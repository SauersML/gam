// `pirls_row.rs` declares this file as `#[cfg(all(test, target_os = "linux"))] mod pirls_row_tests;`;
// declaring the test scope in-file makes that a claim the compiler enforces. Both tests exercise
// Linux-only items: the CPU refusal replay the CUDA launcher consumes, and the generated CUDA sources.
#![cfg(all(test, target_os = "linux"))]

use super::*;
use gam_problem::EstimationError;

#[test]
fn refusal_replay_selects_the_smallest_bad_row_atomically() {
    let eta = [0.0, 0.0, 0.0];
    let y = [0.0, 2.0, -1.0];
    let prior = [1.0; 3];
    let status = [
        status_codes::OK,
        status_codes::RESPONSE,
        status_codes::RESPONSE,
    ];
    assert!(matches!(
        replay_first_refusal(
            PirlsRowFamily::BernoulliLogit,
            CurvatureMode::Fisher,
            1.0,
            &eta,
            &y,
            &prior,
            &status,
        ),
        Err(EstimationError::PirlsRowGeometryUnrepresentable { row: 1, .. })
    ));
}

#[test]
fn generated_sources_have_one_exact_unprojected_contract() {
    let forbidden = [
        "clamp_eta",
        "ETA_CLAMP",
        "MU_FLOOR",
        "W_SOLVER_FLOOR",
        "fmax(",
        "fmin(",
        "flags",
        "1e-12",
        "1e-10",
    ];
    for family in PirlsRowFamily::ALL {
        for curvature in [CurvatureMode::Fisher, CurvatureMode::Observed] {
            for source in [
                cuda_source_for(family, curvature),
                solve_row_source_for(family, curvature),
                ladder_source_for(family, curvature),
            ] {
                for token in forbidden {
                    assert!(!source.contains(token), "{family:?}/{curvature:?}: {token}");
                }
                assert!(source.contains("w_solver = w_hessian"));
                assert!(source.contains("status == PIRLS_OK"));
            }
        }
    }
    let ladder = ladder_source_for(PirlsRowFamily::PoissonLog, CurvatureMode::Fisher);
    assert!(ladder.contains("status_out[k * n + i] = status"));
}


fn bernoulli_row(family: PirlsRowFamily, mode: CurvatureMode, eta: f64, y: f64) -> RowOutput {
    row_reweight_cpu_at(
        0,
        family,
        mode,
        RowInput {
            eta,
            y,
            prior_weight: 1.0,
        },
        1.0,
    )
    .unwrap_or_else(|error| panic!("{family:?}/{mode:?} at eta={eta}, y={y}: {error}"))
}

/// gam#3329: at `η = 4` cloglog's `μ = −expm1(−e⁴)` rounds to exactly 1, so a
/// row built on `μ(1 − μ)` refuses. The log jet reads the row from
/// `b = log μ̄ = −t`, `t = e⁴`, so for `y = 0` the score is `−t` and the
/// observed weight `t` with no rounding at all, and the deviance `−2 log μ̄` is
/// `2t`: one ulp of `exp` in `μ̄` moves `−log μ̄` by at most `ε` (relative `ε/t`),
/// and `bd0`'s log and three sums round at `ε/2` each, so `4ε` bounds it. The
/// Fisher weight `μ a'² + μ̄ b'²` and its closed form `t² μ̄ / μ` each round at
/// most three times from the same `t` and `μ̄`, so `8ε` bounds their gap.
#[test]
fn cloglog_row_is_read_from_the_log_jet_where_mu_rounds_to_one() {
    let eta = 4.0_f64;
    let t = eta.exp();
    let mu_bar = (-t).exp();
    for mode in [CurvatureMode::Fisher, CurvatureMode::Observed] {
        let out = bernoulli_row(PirlsRowFamily::BernoulliCLogLog, mode, eta, 0.0);
        assert_eq!(out.mu, 1.0);
        assert_eq!(out.grad_eta, -t);
        let fisher = t * t * mu_bar / out.mu;
        assert!((out.w_fisher - fisher).abs() <= 8.0 * f64::EPSILON * fisher);
        let expected_hessian = match mode {
            CurvatureMode::Fisher => out.w_fisher,
            CurvatureMode::Observed => t,
        };
        assert_eq!(out.w_hessian, expected_hessian);
        assert!((out.deviance - 2.0 * t).abs() <= 4.0 * f64::EPSILON * 2.0 * t);

        // `y = 1`: `bd0(1, 1) = 0` and `bd0(0, μ̄) = μ̄`, so the deviance is
        // `2μ̄`, which `−2 log1p(−μ̄)` matches to relative `μ̄/2 ≈ 1e−24`.
        let out = bernoulli_row(PirlsRowFamily::BernoulliCLogLog, mode, eta, 1.0);
        assert_eq!(out.deviance, 2.0 * mu_bar);
        assert!(out.grad_eta > 0.0 && out.w_hessian > 0.0);
    }
}

/// Deep in cloglog's left tail, `t = e^{−40}`, the observed weight for `y = 1`
/// is `−a'' = a'(a' + t − 1) ≈ t/2` (relative error `O(t)`), which
/// `1 − a'` computed by subtraction would round to zero. The gap summed from
/// `expm1_minus_x` keeps it: the jet's five roundings from the same `t` stay
/// inside `4ε`.
#[test]
fn cloglog_observed_weight_does_not_cancel_in_the_left_tail() {
    let eta = -40.0_f64;
    let t = eta.exp();
    let out = bernoulli_row(PirlsRowFamily::BernoulliCLogLog, CurvatureMode::Observed, eta, 1.0);
    assert!((out.w_hessian - 0.5 * t).abs() <= 4.0 * f64::EPSILON * 0.5 * t);
}

/// At `η = 9` probit's `μ = Φ(9)` rounds to exactly 1 (`Φ(−9) ≈ 1.1e−19`). For
/// `y = 0` the score is `−R(9)`, `R(x) = φ(x)/Φ(−x)` the Mills ratio, and the
/// observed weight `R(R − x)`. The Sampford and Birnbaum bounds
/// `(3x + √(x² + 8))/4 < R(x) < (x + √(x² + 4))/2` bracket `R`, and `R(R − x)`
/// grows with `R` there, so no tolerance is needed.
#[test]
fn probit_row_is_read_from_the_log_jet_where_mu_rounds_to_one() {
    let x = 9.0_f64;
    let lower = (3.0 * x + (x * x + 8.0).sqrt()) / 4.0;
    let upper = (x + (x * x + 4.0).sqrt()) / 2.0;
    let out = bernoulli_row(PirlsRowFamily::BernoulliProbit, CurvatureMode::Observed, x, 0.0);
    assert_eq!(out.mu, 1.0);
    let mills = -out.grad_eta;
    assert!(lower < mills && mills < upper, "R(9) = {mills}");
    assert!(lower * (lower - x) < out.w_hessian && out.w_hessian < upper * (upper - x));
    let out = bernoulli_row(PirlsRowFamily::BernoulliProbit, CurvatureMode::Fisher, x, 0.0);
    assert!(out.w_fisher > 0.0 && out.w_hessian == out.w_fisher);
}
