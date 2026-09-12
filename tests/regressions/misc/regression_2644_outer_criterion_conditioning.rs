//! #2644: the outer smoothing-parameter optimizer refusing to certify a
//! stationary optimum at an interior PSD minimum.
//!
//! Four gates: one certification gate and three fits that must succeed.
//!
//! ## The gate — `te_with_disparate_scales_certifies`
//!
//! `y ~ te(a,b,k=5)` on 300 rows with `a ∈ [0,1]` against `b ∈ [0,1000]`. This
//! is `misc::mega_batch_k::te_with_disparate_scales`, a `GAM_ERROR` row on both
//! of the last two reference-quality nightlies, and it moved BETWEEN the two
//! shapes the refusal takes:
//!
//! ```text
//! run 30602192415:  cost-only value disagrees with analytic-sample value at
//!                   the same outer point (4.141e-5 vs a 2.842e-5 roundoff bound)
//! run 30619084852:  |g|=|Pg|=2.458e2  bound=1.015e1  hessian_psd=yes
//!                   after exhausting a 400-iteration outer budget
//! ```
//!
//! Both are the same criterion. Its penalty blocks reach `κ(S_λ) = 3.5e19`,
//! where the assembled-matrix Cholesky that used to price `log|S_λ|₊` disagrees
//! with the root-scale spectrum by **20 log units** — and, because that Cholesky
//! FAILS at some trial `rho` and succeeds at others, `V(rho)` was priced by two
//! different formulas with a step of that size between them. 196 of this fit's
//! 1015 penalty-logdet builds took one formula and 819 the other.
//!
//! Measured here: **refused after 400 outer iterations in 0.94 s** before
//! `75563f13e`, **certifies in 0.18 s** after it.
//!
//! The gate lives in `tests/` rather than only in the nightly quality suite
//! because the quality suite runs on a 6-hour schedule and this is a
//! sub-second fit.
//!
//! ## Fits that must succeed
//!
//! These were print-only probes that passed whether the fit certified or refused.
//! Each now fails with the refusal text.
//!
//! * `prostate` — `y ~ s(pc1,k=5) + s(pc2,k=5)`, binomial/logit, 490 rows of
//!   `bench/datasets/prostate.csv`. Three reference-quality rows report the
//!   same `|g|=|Pg|=1.792e-3 bound=1.133e-3` from it. STILL REFUSES after the
//!   `log|S_λ|₊` fix, now at `|Pg|=3.396e-3` against `bound=3.015e-3`, because
//!   its residual criterion noise is `log|H|`, which is priced from the
//!   eigenvalues of the ASSEMBLED `H` (`κ = 3.8e12` ⇒ `6.6e-4` of scatter at
//!   fixed `rho`, against an outer cost floor of `1.8e-5`). Recovering that one
//!   needs a root of `H` that is not formed by summing `XᵀWX` and `S_λ` in f64,
//!   which is a change to what P-IRLS publishes; it is not attempted here.
//! * `matern` — the fit the issue thread recommends as a reproducer. It
//!   certified at `|Pg|=2.025e-5` against `bound=9.876e-5` when measured as a
//!   probe.
//! * `matern low n` — `y ~ matern(x, nu=5/2)` on 15 rows of `t²` plus noise.
//!
use gam::data::EncodedDataset;
use gam::{FitConfig, encode_recordswith_inferred_schema, fit_from_formula, load_csvwith_inferred_schema};
use csv::StringRecord;
use ndarray::Array2;
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand_distr::{Distribution, Normal, Uniform};
use std::path::Path;

const PROSTATE_CSV: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/bench/datasets/prostate.csv");

fn subset_rows(ds: &EncodedDataset, rows: &[usize]) -> EncodedDataset {
    let mut sub = ds.clone();
    let ncol = ds.values.ncols();
    let mut values = Array2::<f64>::zeros((rows.len(), ncol));
    for (new_r, &old_r) in rows.iter().enumerate() {
        for c in 0..ncol {
            values[[new_r, c]] = ds.values[[old_r, c]];
        }
    }
    sub.values = values;
    sub
}

#[test]
fn prostate_binomial_logit_fit_certifies_2644() {
    gam_solve::progress_log::init_logging_at(log::LevelFilter::Info);
    let ds = load_csvwith_inferred_schema(Path::new(PROSTATE_CSV)).expect("load prostate.csv");
    let n = ds.values.nrows();
    let train_rows: Vec<usize> = (0..n).filter(|i| i % 4 != 0).collect();
    let ds_train = subset_rows(&ds, &train_rows);
    let cfg = FitConfig {
        family: Some("binomial".to_string()),
        link: Some("logit".to_string()),
        ..FitConfig::default()
    };
    let started = std::time::Instant::now();
    let outcome = fit_from_formula("y ~ s(pc1, k=5) + s(pc2, k=5)", &ds_train, &cfg);
    println!(
        "[probe-2644-prostate] n_train={} elapsed={:.2}s",
        train_rows.len(),
        started.elapsed().as_secs_f64()
    );
    if let Err(e) = outcome {
        panic!("[2644-prostate] y ~ s(pc1, k=5) + s(pc2, k=5) binomial/logit must fit; got: {e}");
    }
}

const N_TRAIN: usize = 1_500;
const SIGMA: f64 = 0.10;
const TRAIN_SEED: u64 = 1_039;

fn clamp_unit_open(x: f64) -> f64 {
    x.max(1.0e-6).min(1.0 - 1.0e-6)
}

fn latent_to_coords(t: f64) -> [f64; 3] {
    [
        clamp_unit_open(t),
        clamp_unit_open(0.5 + 0.5 * (2.0 * std::f64::consts::PI * t).sin()),
        clamp_unit_open(t * t),
    ]
}

fn truth(t: f64) -> f64 {
    (2.0 * std::f64::consts::PI * t).sin() + 0.5 * (4.0 * std::f64::consts::PI * t).cos()
}

fn build_dataset(n: usize, sigma: f64, seed: u64) -> EncodedDataset {
    let mut rng = StdRng::seed_from_u64(seed);
    let latent = Uniform::new(0.0, 1.0).expect("uniform");
    let noise = Normal::new(0.0, sigma).expect("normal");
    let headers = ["x0", "x1", "x2", "y"]
        .iter()
        .cloned()
        .map(String::from)
        .collect();
    let rows: Vec<StringRecord> = (0..n)
        .map(|_| {
            let t = latent.sample(&mut rng);
            let coords = latent_to_coords(t);
            let y = truth(t) + noise.sample(&mut rng);
            StringRecord::from(vec![
                coords[0].to_string(),
                coords[1].to_string(),
                coords[2].to_string(),
                y.to_string(),
            ])
        })
        .collect();
    encode_recordswith_inferred_schema(headers, rows).expect("encode")
}

#[test]
fn matern_three_axis_k16_fit_certifies_2644() {
    gam_solve::progress_log::init_logging_at(log::LevelFilter::Info);
    let ds = build_dataset(N_TRAIN, SIGMA, TRAIN_SEED);
    let cfg = FitConfig {
        family: Some("gaussian".to_string()),
        ..FitConfig::default()
    };
    let started = std::time::Instant::now();
    let outcome = fit_from_formula("y ~ matern(x0, x1, x2, k=16)", &ds, &cfg);
    println!("[probe-2644] elapsed={:.2}s", started.elapsed().as_secs_f64());
    if let Err(e) = outcome {
        panic!("[2644-matern] y ~ matern(x0, x1, x2, k=16) must fit; got: {e}");
    }
}

// The gate. See the module header.

fn mk_2d(
    n: usize,
    f: impl Fn(f64, f64) -> f64,
    ra: (f64, f64),
    rb: (f64, f64),
    sigma: f64,
    seed: u64,
) -> EncodedDataset {
    let mut rng = StdRng::seed_from_u64(seed);
    let ua = Uniform::new(ra.0, ra.1).expect("finite a range");
    let ub = Uniform::new(rb.0, rb.1).expect("finite b range");
    let noise = Normal::new(0.0, sigma).expect("finite sigma");
    let h = ["a", "b", "y"].into_iter().map(String::from).collect();
    let mut rows = Vec::with_capacity(n);
    for _ in 0..n {
        let a = ua.sample(&mut rng);
        let b = ub.sample(&mut rng);
        let y = f(a, b) + noise.sample(&mut rng);
        rows.push(StringRecord::from(vec![
            a.to_string(),
            b.to_string(),
            y.to_string(),
        ]));
    }
    encode_recordswith_inferred_schema(h, rows).expect("encode")
}

#[test]
fn te_with_disparate_scales_certifies() {
    gam_solve::progress_log::init_logging_at(log::LevelFilter::Info);
    let ds = mk_2d(300, |a, b| a + b, (0.0, 1.0), (0.0, 1000.0), 0.05, 7);
    let cfg = FitConfig {
        family: Some("gaussian".to_string()),
        ..FitConfig::default()
    };
    let started = std::time::Instant::now();
    let outcome = fit_from_formula("y~te(a,b,k=5)", &ds, &cfg);
    println!(
        "[probe-2644-te] elapsed={:.2}s",
        started.elapsed().as_secs_f64()
    );
    let Err(error) = outcome else {
        println!("[probe-2644-te] FIT OK");
        return;
    };
    panic!(
        "#2644: `y~te(a,b,k=5)` on disparate scales must reach a certified outer \
         optimum. This fit refused for 400 outer iterations at |Pg|=2.458e2 while \
         `log|S_lambda|+` was priced from a factorization of the assembled penalty \
         sum, whose error is O(eps*kappa) and whose Cholesky FAILS at some trial \
         rho and succeeds at others -- so the objective carried a step of ~20 log \
         units. A refusal here means that pricing (or an equivalent one) is back: \
         {error}"
    );
}

// ─────────────────────────────────────────────────────────────────────────
// Reporting probe: `misc::broad_sweep_batch_h::matern_low_n_does_not_crash`.
//
// `y ~ matern(x, nu=5/2)`, Gaussian, **15 rows**, one coordinate — the
// smallest witness of an outer-criterion refusal in the suite, and the one
// that fires on the joint-spatial route-AGREEMENT half of the certificate
// rather than its descent half. It refuses in 0.6 s with
//
//   joint_seed = 2.787290435812e0   baseline = 2.787290399076e0
//   gap = 3.674e-8 (1.318e-8 relative)   agreement_tolerance = 2.787e-8
//
// bit-identically across two nightlies on different runners and two local
// runs. This is the residue `9e8ca98da` left open when it closed the
// O(eps*kappa) log-determinant family, and the root-scale work could not have
// touched it.
//
// MEASURED (2026-07-31, at `04671c5b3`), by widening the in-tree
// `[#1271-diag]` from `{:.6}` to `{:.17e}` and adding `beta` and
// `h_total_original` to it. Both routes evaluate at a BIT-IDENTICAL `rho`:
//
//   logS, logH                          bit-identical to 17 digits
//   h_total_original (all 49 entries)   bit-identical (max|H1-H2| = 0.0)
//   pirls_edf                           identical
//   beta                                differs in COORDINATE 0 ALONE, by
//                                       2.13012903753208305e-1; coordinates
//                                       1..6 agree to ~10 digits
//   H[0][0] = 1.50000000100000008e1 = n + 1e-8, so coordinate 0 is the
//   intercept and it carries the absolute `FIXED_STABILIZATION_RIDGE`, which
//   that constant's own doc says `penalty_term` carries as `ridge*||beta||^2`.
//
// The gap is then fully accounted for, with no free parameter:
//
//   ridge * (b0'^2 - b0^2) = 7.7024058e-10   vs measured dDp = 7.7024051e-10
//                                            (ratio 1.00000009)
//   (n/2) * dDp / Dp       = 3.6736410122e-8 vs measured gap = 3.673641e-8
//
// So 100% of the disagreement is `(n/2)*d log(rss+pen)`, 0% is either
// log-determinant, and the ONLY term that notices the intercept moving is the
// 1e-8 ridge. The refusal is the certificate correctly reporting that the
// criterion assigns two values to one predictor.
//
// REFUTED along the way, so nobody re-runs them: the offset seam (instrumented
// `compose_offset` itself; both routes compose an identically ZERO offset from
// an identically zero affine channel), and the Gaussian sufficient-statistic
// cache (`[gaussian-fixed-cache]` is built 11 times, on both routes).
//
// RESOLVED by #2671, and the answer is in the last line above. What made the two
// routes solve different systems is the RESPONSE, not the design: the scalar-rho
// route conditions it (`optimizer.rs`, #1000 centering by `c = mean_w(y-offset)`
// = 2.130e-1 here) and the joint psi route handed `y` to
// `ExternalJointHyperEvaluator::new` verbatim. `beta` differs in coordinate 0
// alone by 2.13012903753208305e-1 = `c` to 10 digits — the intercept absorbing
// exactly the centering constant — which is why `H`, `logS`, `logH` and
// `pirls_edf` are all bit-identical: at `delta = 0` the two are the SAME problem
// under `b_joint = b_scalar + c*e0`, and only the ridge, priced against a fixed
// zero, distinguishes them. `run_exact_joint_spatial_optimization` now conditions
// through the same gate.
//
// This probe PRINTS and never fails; the GATE for the resolved defect is
// `joint_route_outer_criterion_is_invariant_to_the_origin_of_y` below, which
// checks the property (origin-invariance of the criterion) rather than this
// fixture's particular gap.
// ─────────────────────────────────────────────────────────────────────────

fn mk_1d(n: usize, f: impl Fn(f64) -> f64, sigma: f64, seed: u64) -> EncodedDataset {
    let mut rng = StdRng::seed_from_u64(seed);
    let u = Uniform::new(0.0_f64, 1.0).expect("uniform");
    let noise = Normal::new(0.0, sigma).expect("normal");
    let mut x: Vec<f64> = (0..n).map(|_| u.sample(&mut rng)).collect();
    x.sort_by(|a, b| a.partial_cmp(b).expect("finite draws"));
    let y: Vec<f64> = x.iter().map(|&t| f(t) + noise.sample(&mut rng)).collect();
    let headers = ["x", "y"].into_iter().map(String::from).collect();
    let rows: Vec<StringRecord> = x
        .iter()
        .zip(y.iter())
        .map(|(a, b)| StringRecord::from(vec![a.to_string(), b.to_string()]))
        .collect();
    encode_recordswith_inferred_schema(headers, rows).expect("encode")
}

#[test]
fn matern_low_n_fit_certifies_2644() {
    gam_solve::progress_log::init_logging_at(log::LevelFilter::Info);
    let ds = mk_1d(15, |t| t.powi(2), 0.05, 7);
    let cfg = FitConfig {
        family: Some("gaussian".to_string()),
        ..FitConfig::default()
    };
    let started = std::time::Instant::now();
    let outcome = fit_from_formula("y ~ matern(x, nu=5/2)", &ds, &cfg);
    println!(
        "[probe-2644-lown] elapsed={:.2}s",
        started.elapsed().as_secs_f64()
    );
    if let Err(e) = outcome {
        panic!("[2644-lown] y ~ matern(x, nu=5/2) on 15 rows must fit; got: {e}");
    }
}

// ─────────────────────────────────────────────────────────────────────────
// #2671 — THE ORIGIN GUARD, ON A ROUTE THAT CAN FAIL IT.
//
// ## Why the previous guard was retired rather than re-tuned
//
// `outer_criterion_is_invariant_to_the_origin_of_y` fitted `y ~ s(x, k=5)`
// against `y + 10` on `mk_1d(15, t^2, 0.05, 7)`, measured `gap = 4.085621e-14`
// (relative 4.504937e-15) against a `5.092357e-05` prediction, and that nine-
// order miss was recorded — here and in `FIXED_STABILIZATION_RIDGE`'s doc — as a
// REFUTATION of the claim that the ridge shrinks toward a FIXED zero.
//
// It refuted nothing. **THE QUANTITY WAS NEVER FREE TO DISAGREE.** That fixture
// is Gaussian, identity-link, has an unpenalized intercept and no linear
// constraints — every clause of `gaussian_identity_response_center`'s gate
// (`gam-solve/src/estimate/optimizer.rs`) — so the #1000 centering subtracts the
// weighted response mean from BOTH arms before the criterion exists, and
// `reml_score` is the outer value of the CENTERED problem. `y` and `y + 10` map
// to the same conditioned vector. The 4.085621e-14 is the floating-point residue
// of `(y_i + 10) - 10.33` versus `y_i - 0.33`, and the `1e-10` bar was
// denominated in that residue. A guard that cannot fail for the reason its own
// comment gives is not a guard; re-tuning the bar would not have fixed it,
// because no bar can make a constant-by-construction quantity informative.
//
// ## Where the origin was actually live
//
// The joint `[rho, psi]` spatial route (`run_exact_joint_spatial_optimization`)
// handed `y` to `ExternalJointHyperEvaluator::new` VERBATIM — it never called
// the conditioning function — and
// `try_exact_joint_spatial_length_scale_optimization` then graded its criterion
// against the scalar-rho route's `fit_score`, which IS conditioned. With
// `FIXED_STABILIZATION_RIDGE = 1e-8` priced as `delta*||beta||^2` against zero,
// the two routes minimized problems differing by `delta*(2*c*beta0 + c^2)` on
// the intercept axis.
//
// MEASURED at `517b6303f`, release, one run, three arms of THIS fixture, against
// the law `gap = (n/2)/D_p * delta * ((beta0 + m)^2 - beta0^2)` registered as a
// VALUE before the run:
//
//   arm          mean(y)      predicted     measured
//   precentered  -3.70e-17    ~0            fit ACCEPTED
//   asis          2.130e-1    3.674e-8      3.674e-8   REFUSED (tol 2.787e-8)
//   plus10        1.0213e1     5.047e-5     5.047e-5   REFUSED, 1374x worse
//
// Linear-in-`m` is out by 29x, constant by 1374x. The scalar route moved
// 4.085e-14 under the same +10 shift, in the same run: separation 1.24e9.
// User-visible: subtracting `mean(y)` turned a refusing fit into a shipping one.
//
// ## What this test asserts, and why none of it is vacuous
//
// 1. ALL THREE ARMS MUST FIT. Two of the three REFUSED before the fix, so this
//    clause is measured-falsifiable, not assumed.
// 2. TWO-SIDED origin invariance of the reported criterion, relative bar 1e-12:
//    ~67x above the residue the conditioned route actually leaves (~1.5e-14
//    relative) and 1.8e7x BELOW the pre-fix 1.811e-5. It cannot pass by
//    roundoff and it cannot fail on roundoff.
// 3. THE ARMS COULD HAVE DIFFERED — asserted from live values, not from this
//    comment. The fitted coefficient vectors of the `asis` and `plus10` arms
//    must differ by exactly one coordinate moved by `SHIFT` (checked in both the
//    L-infinity and L1 norms, so it is one coordinate and not a spread), and the
//    ridge term the criterion is built from,
//    `delta*(||b'||^2 - ||b||^2)`, must therefore differ between the arms by
//    at least 1e-7 in absolute value — five orders ABOVE the invariance bar.
//    That is the clause the retired guard lacked: the criterion was free to move
//    by ~1e-6 and did not.
//
// The fixture is deliberately the one whose route DOES NOT satisfy the
// `gaussian_identity_response_center` gate upstream of the criterion under test.
// If a future change routes `y ~ matern(x, nu=5/2)` away from the joint spatial
// route, clause 3 is what will notice that this test has gone mute.
// ─────────────────────────────────────────────────────────────────────────

/// `FIXED_STABILIZATION_RIDGE` (`gam-solve/src/pirls/gam_working_model.rs`),
/// which is `pub(crate)` and therefore restated here. Clause 3 only needs its
/// ORDER of magnitude, but the exact value is what makes the 1e-7 floor below a
/// derived number rather than a guess.
const FIXED_STABILIZATION_RIDGE: f64 = 1.0e-8;

#[test]
fn joint_route_outer_criterion_is_invariant_to_the_origin_of_y() {
    gam_solve::progress_log::init_logging_at(log::LevelFilter::Info);
    let cfg = FitConfig {
        family: Some("gaussian".to_string()),
        ..FitConfig::default()
    };
    const SHIFT: f64 = 10.0;
    // Column 1 is `y` (headers are ["x", "y"] in `mk_1d`).
    const Y_COL: usize = 1;

    let base = mk_1d(15, |t| t.powi(2), 0.05, 7);
    let n = base.values.nrows();
    let base_mean: f64 = (0..n).map(|r| base.values[[r, Y_COL]]).sum::<f64>() / n as f64;

    let shifted_by = |delta: f64| -> EncodedDataset {
        let mut ds = base.clone();
        for r in 0..ds.values.nrows() {
            ds.values[[r, Y_COL]] += delta;
        }
        ds
    };

    // Three arms spanning `mean(y)` from ~1e-17 to ~10.2 — the same span the
    // pre-registered sweep used.
    let arms = [
        ("precentered", shifted_by(-base_mean)),
        ("asis", base.clone()),
        ("plus10", shifted_by(SHIFT)),
    ];

    // ── Phase 1: fit every arm, and report EVERY refusal, not the first ──
    //
    // Clause 1. Before the joint route was conditioned, `asis` AND `plus10` both
    // refused (gap 3.674e-8 and 5.047e-5 against agreement_tolerance 2.787e-8)
    // while `precentered` was ACCEPTED. A loop that panicked on the first
    // failure would report one of those two and hide the other -- and the pair
    // is the evidence, because their RATIO is what identifies the mechanism.
    let mut scores: Vec<f64> = Vec::with_capacity(arms.len());
    let mut betas: Vec<ndarray::Array1<f64>> = Vec::with_capacity(arms.len());
    let mut means: Vec<f64> = Vec::with_capacity(arms.len());
    let mut refusals: Vec<String> = Vec::new();
    for (label, ds) in &arms {
        let mean_y: f64 = (0..n).map(|r| ds.values[[r, Y_COL]]).sum::<f64>() / n as f64;
        means.push(mean_y);
        match fit_from_formula("y ~ matern(x, nu=5/2)", ds, &cfg) {
            Ok(gam::FitResult::Standard(fitted)) => {
                let score = fitted.fit.reml_score().unwrap_or(f64::NAN);
                let beta = fitted.fit.beta.clone();
                println!(
                    "[2671-origin] arm={label} mean_y={mean_y:.17e} reml={score:.17e} p={}",
                    beta.len()
                );
                scores.push(score);
                betas.push(beta);
            }
            Ok(_) => refusals.push(format!("{label}: expected a standard fit")),
            Err(error) => {
                println!("[2671-origin] arm={label} mean_y={mean_y:.17e} REFUSED");
                refusals.push(format!("  * {label} (mean_y={mean_y:.6e}): {error}"));
            }
        }
    }
    assert!(
        refusals.is_empty(),
        "#2671 clause 1: `y ~ matern(x, nu=5/2)` must fit regardless of where the origin of the \
         response sits. Whether a fit ships must not depend on the user's choice of response \
         units. {} of {} arms refused:\n{}\n\nBefore the joint [rho, psi] route was conditioned \
         through `gaussian_identity_response_center` the `asis` arm refused with a \
         criterion-agreement gap of 3.674e-8 and `plus10` with 5.047e-5, both against a \
         2.787e-8 tolerance, while `precentered` was ACCEPTED.",
        refusals.len(),
        arms.len(),
        refusals.join("\n"),
    );

    // ── Phase 2: every remaining clause is MEASURED, then reported together ──
    //
    // Collected rather than asserted one by one. A failing supporting clause
    // must never suppress the measurement of the property this gate exists for:
    // an early-aborting assert withdraws the coverage it advertises. (Measured
    // on this very test: a mis-denominated clause-3b bar aborted the run before
    // clause 2 -- the invariance itself -- had been evaluated at all.)
    let mut failures: Vec<String> = Vec::new();

    // Clause 2 is FIRST because it is the property under test: the reported
    // criterion must not move when only the origin of `y` does. Two-sided.
    let bar = 1.0e-12;
    for (idx, label) in [(0_usize, "precentered"), (2_usize, "plus10")] {
        let gap = scores[idx] - scores[1];
        let relative = gap.abs() / scores[1].abs().max(f64::MIN_POSITIVE);
        println!(
            "[2671-origin] CLAUSE2 {label} vs asis: gap={gap:.6e} relative={relative:.6e} \
             bar={bar:.1e}"
        );
        if !(relative < bar) {
            failures.push(format!(
                "CLAUSE 2 ({label}) -- THE PROPERTY UNDER TEST. The outer REML criterion must be \
                 a function of the MODEL, and adding a constant to `y` moves the intercept and \
                 nothing else. Arm `{label}` moved {relative:.6e} relative against a {bar:.1e} \
                 bar. REGISTERED before the fix: 1.811e-5 relative (absolute 5.047e-5) and the \
                 fit REFUSED; the conditioned route measured 1.34e-15. A failure here means an \
                 outer lambda-search is again forming its criterion on an unconditioned \
                 response, so lambda-hat depends on the origin of the user's response units. Do \
                 NOT fix it by exempting the intercept from `FIXED_STABILIZATION_RIDGE` (that \
                 deletes the detector) and do NOT fix it by deleting the #1000 centering (that \
                 is what makes lambda-hat origin-invariant)."
            ));
        }
    }

    // Clause 3a: the arms really are separated along the axis the criterion must
    // be blind to, and the `asis` arm is itself off the origin. Without this a
    // fixture change that silently equalized the arms would make clause 2 pass
    // by construction. These bars are exact arithmetic on the fixture, so they
    // are denominated in nothing but f64.
    let shift_span = means[2] - means[1];
    println!(
        "[2671-origin] CLAUSE3a means: precentered={:.17e} asis={:.17e} span={shift_span:.17e}",
        means[0], means[1]
    );
    if !((shift_span - SHIFT).abs() < 1.0e-9 && means[0].abs() < 1.0e-9 && means[1].abs() > 1.0e-2)
    {
        failures.push(format!(
            "CLAUSE 3a -- the arms must actually differ in the origin of `y`, and the `asis` arm \
             must itself be off the origin (it is the arm whose 3.674e-8 gap opened this issue). \
             Measured mean(y): precentered={:.17e} (must be ~0), asis={:.17e} (must be well away \
             from 0; 2.130e-1 on this fixture), plus10-asis={shift_span:.17e} (must be {SHIFT}). \
             An arm set that does not separate cannot test an invariance.",
            means[0], means[1]
        ));
    }

    // Clause 3b: the shift landed as a pure intercept relabeling -- exactly one
    // coordinate of `beta` moved, and by `SHIFT`. `l1 == linf` is what says "one
    // coordinate", so this does not depend on knowing the intercept's index.
    //
    // BOTH BARS ARE RELATIVE TO `SHIFT`, which is what produces the numbers.
    // They were absolute (1e-6) on first writing and that was a denomination
    // error, not a threshold that was merely too tight: `beta` is converged to a
    // RELATIVE tolerance, so on a coefficient of magnitude 10 the residue scales
    // with 10. MEASURED here: |linf - SHIFT| = 1.6e-7 and l1 - linf = 3.26e-6,
    // i.e. 1.6e-8 and 3.3e-7 OF `SHIFT`. The 1e-5 relative bar leaves 625x and
    // 30x of margin and still catches what this clause exists to catch: a shift
    // spreading into the smooth is O(1), not O(1e-6).
    let relabel_bar = 1.0e-5 * SHIFT;
    let delta_beta = &betas[2] - &betas[1];
    let linf = delta_beta.iter().fold(0.0_f64, |acc, v| acc.max(v.abs()));
    let l1: f64 = delta_beta.iter().map(|v| v.abs()).sum();
    println!(
        "[2671-origin] CLAUSE3b |d beta|_inf={linf:.17e} |d beta|_1={l1:.17e} \
         bar={relabel_bar:.1e}"
    );
    if !((linf - SHIFT).abs() < relabel_bar && (l1 - linf).abs() < relabel_bar) {
        failures.push(format!(
            "CLAUSE 3b -- adding {SHIFT} to `y` must move exactly one coefficient (the intercept) \
             by {SHIFT} and leave every other coefficient fixed. Measured |d beta|_inf={linf:.17e} \
             (miss {:.3e}), |d beta|_1={l1:.17e} (spread into the other coefficients {:.3e}), \
             against a bar of {relabel_bar:.1e} = 1e-5*SHIFT. If these disagree the shift spread \
             across the smooth and the fit is no longer a pure origin relabeling -- in which case \
             clause 2's invariance is not the property this gate means to assert.",
            (linf - SHIFT).abs(),
            (l1 - linf).abs(),
        ));
    }

    // Clause 3c: THE FREEDOM CLAUSE. The criterion therefore carried a term that
    // genuinely differed between the arms. `penalty_term` charges
    // `delta*||beta||^2` against a target of zero (`gam_working_model.rs` and
    // `loop_driver.rs` both spell it literally, with no `prior_mean_target` in
    // the expression), so the two arms priced ridges differing by `ridge_gap` --
    // which must sit far ABOVE clause 2's bar, or clause 2 is a property of the
    // fixture rather than of the code.
    let ridge_asis = FIXED_STABILIZATION_RIDGE * betas[1].dot(&betas[1]);
    let ridge_plus10 = FIXED_STABILIZATION_RIDGE * betas[2].dot(&betas[2]);
    let ridge_gap = (ridge_plus10 - ridge_asis).abs();
    let clause2_absolute_bar = bar * scores[1].abs();
    println!(
        "[2671-origin] CLAUSE3c ridge_term_gap={ridge_gap:.17e} \
         clause2_absolute_bar={clause2_absolute_bar:.6e} \
         headroom={:.3e}x",
        ridge_gap / clause2_absolute_bar.max(f64::MIN_POSITIVE)
    );
    if !(ridge_gap > 1.0e-7) {
        failures.push(format!(
            "CLAUSE 3c -- THE QUANTITY MUST BE FREE TO DISAGREE. The outer criterion is built \
             from a penalty carrying `delta*||beta||^2` against a FIXED zero target \
             (delta={FIXED_STABILIZATION_RIDGE:.1e}), and the arms' coefficient vectors differ by \
             {SHIFT} on the intercept, so that term differs between them by {ridge_gap:.6e} -- \
             which must be far above clause 2's absolute bar of {clause2_absolute_bar:.6e}, or \
             the invariance is a property of the fixture rather than of the code. The retired \
             guard `outer_criterion_is_invariant_to_the_origin_of_y` failed exactly here: its \
             route centered the response upstream, so its two arms were the SAME problem and its \
             4.5e-15 agreement was arithmetic, not evidence."
        ));
    }

    assert!(
        failures.is_empty(),
        "#2671: {} of 4 clauses failed. Every clause was MEASURED before this panic, so the \
         picture below is complete rather than truncated at the first failure.\n\n{}",
        failures.len(),
        failures.join("\n\n"),
    );
}
