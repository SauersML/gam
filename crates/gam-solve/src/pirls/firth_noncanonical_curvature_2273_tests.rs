#![cfg(test)]
//! #2273: the Firth-penalized inner solve must be a Newton solve on EVERY
//! binomial link, not only the canonical one.
//!
//! `WorkingState.hessian` is `XᵀWX + S` and deliberately omits the Jeffreys
//! coefficient Hessian `HΦ`, because the outer Laplace layer consumes `H₀` and
//! `HΦ` separately. Two consumers therefore have to fold it back in: the
//! quadratic model (`objective_hessian_quadratic_correction`, `−dᵀHΦd`) and the
//! linear system the direction is solved from
//! (`objective_hessian_matrix_correction`, subtracted from `H`).
//!
//! Only the first existed. The augmented-square-root direction solve folded
//! `HΦ` in by congruence, but that route is gated on
//! `hessian_curvature == Fisher`, which holds only for the CANONICAL logit
//! link; every non-canonical binomial link (probit, cloglog, …) resolves
//! `Observed` and fell through to a dense solve of `(XᵀWX + S + λD²) d = −g`,
//! with the Jeffreys score in `g` and no Jeffreys curvature in the matrix. That
//! is not a Newton step for any objective, and it contracts LINEARLY: measured
//! at rate 0.4937/iteration on the fixture below, reaching `‖g‖ = 4.3e-7` in 23
//! iterations, failing the exact-decrement certificate
//! (`decrement² = 3.4e-14` against a `2.7e-20` threshold) and returning
//! `StalledAtValidMinimum` — which the fit-assembly gate refuses. The refused β̂
//! was the RIGHT one, which is why the failure was so hard to read.
//!
//! Four places invert that Hessian, not two: the dense unconstrained solve, the
//! constrained/bounded active-set solves, the post-loop undamped Newton polish,
//! and the exact-decrement certificate. All four now fold `HΦ` in through one
//! helper that owns the sign convention.
//!
//! The file then covers what the corrected solve reaches, which is a saturated
//! row — and two further defects that only a converging solve gets far enough to
//! hit. The Bernoulli variance was rebuilt from `μ` alone, so `V = μ(1−μ)`
//! collapsed to a hard zero past the point where `μ` rounds to `1.0`; and the
//! observed weight's closed forms divided by `φV²`, `φV³` and `φV⁴`, which
//! underflow at `V = 2.9e-84`.
//!
//! What the tests assert, in order:
//!
//!   * the cross-link invariant — a Firth solve certifies on EVERY binomial link,
//!     not only the canonical one;
//!   * the probit Firth mode against scipy on the same objective;
//!   * the fold-in's sign convention, where it is defined;
//!   * the noncanonical observed tower on a saturated row, against the exact
//!     identity `−ℓ = e^η` for a cloglog `y = 0` row and against mpmath at 220
//!     decimal digits for `y = 1`;
//!   * the variance's positivity past the `μ`-rounding point;
//!   * the composite bounded links' tail complements, against the absolute
//!     resolution of the naive subtraction they replace.

#![cfg(test)]

use super::*;
use gam_problem::{
    InverseLink, LikelihoodSpec, LogSmoothingParamsView, ResponseFamily, StandardLink,
};
use gam_spec::GlmLikelihoodSpec;
use ndarray::{Array1, Array2};

/// #2273's exact-separation fixture at its smallest reported size: three rows of
/// class 0 at `x ∈ {1.0, 1.1, 1.2}` and three of class 1 at `x ∈ {10.0, 10.1,
/// 10.2}`, with the design centred and scaled the way the engine's parametric
/// column conditioning leaves it.
fn separated_probit_fixture() -> (Array2<f64>, Array1<f64>) {
    let x_raw = [1.0_f64, 1.1, 1.2, 10.0, 10.1, 10.2];
    let n = x_raw.len();
    let mean = x_raw.iter().sum::<f64>() / n as f64;
    let variance = x_raw.iter().map(|v| (v - mean) * (v - mean)).sum::<f64>() / n as f64;
    let sd = variance.sqrt();
    let mut x = Array2::<f64>::zeros((n, 2));
    for (row, &value) in x_raw.iter().enumerate() {
        x[[row, 0]] = 1.0;
        x[[row, 1]] = (value - mean) / sd;
    }
    let y = Array1::from(vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0]);
    (x, y)
}

/// Run one Firth-penalized, unpenalized-in-λ P-IRLS solve on the fixture and
/// report what the inner solve achieved.
fn firth_inner_solve(link: StandardLink) -> PirlsResult {
    let (x, y) = separated_probit_fixture();
    let p = x.ncols();
    let weights = Array1::<f64>::ones(y.len());
    let offset = Array1::<f64>::zeros(y.len());
    let (canonical, _) =
        gam_terms::construction::canonicalize_penalty_specs(&[], &[], p, "#2273 firth curvature")
            .expect("an empty penalty set canonicalizes");
    let config = PirlsConfig {
        likelihood: GlmLikelihoodSpec::canonical(LikelihoodSpec::new(
            ResponseFamily::Binomial,
            InverseLink::Standard(link),
        )),
        link_kind: InverseLink::Standard(link),
        max_iterations: 100,
        convergence_tolerance: 1e-12,
        firth_bias_reduction: true,
        initial_lm_lambda: None,
        arrow_schur: None,
    };
    let problem = PirlsProblem {
        x: x.clone(),
        offset: offset.view(),
        y: y.view(),
        priorweights: weights.view(),
        covariate_se: None,
        gaussian_fixed_cache: None,
        glm_first_step_gram: None,
    };
    let penalty = PenaltyConfig {
        canonical_penalties: &canonical,
        balanced_penalty_root: None,
        reparam_invariant: None,
        p,
        coefficient_lower_bounds: None,
        linear_constraints_original: None,
        kronecker_factored: None,
    };
    let rho = Array1::<f64>::zeros(0);
    let (fit, _) = fit_model_for_fixed_rho(
        LogSmoothingParamsView::new(rho.view()).expect("an empty rho is in the strength domain"),
        problem,
        penalty,
        &config,
        None,
    )
    .expect("the Firth inner solve must not error on a 6-row separated fixture");
    fit
}

/// The cross-link invariant. A Firth solve on a NON-canonical binomial link must
/// certify inner convergence exactly as the canonical one does; the two differ
/// only in which direction-solve branch they take, and both branches have to
/// describe the Firth objective.
///
/// Before the fix: logit `Converged` in 5 iterations at `‖g‖ = 5.5e-14`, probit
/// and cloglog `StalledAtValidMinimum` after 23+ iterations of linear
/// contraction. After: all three certify, and the probit mode agrees with an
/// independent reference to 7 digits (the sibling test).
#[test]
fn firth_inner_solve_converges_on_every_binomial_link_2273() {
    let mut report = String::new();
    let mut failures = Vec::new();
    for link in [StandardLink::Logit, StandardLink::Probit, StandardLink::CLogLog] {
        let fit = firth_inner_solve(link);
        report.push_str(&format!(
            "\n  {link:?}: status={:?} iters={} |g|={:.3e} deviance={:.9e} max|eta|={:.4}",
            fit.status, fit.iteration, fit.lastgradient_norm, fit.deviance, fit.max_abs_eta,
        ));
        if !fit.status.is_converged() {
            failures.push(format!("{link:?} status {:?}", fit.status));
        }
        // A consistent Newton system on a 2-coefficient problem whose Fisher
        // information has condition number 1.0009 reaches the arithmetic floor
        // in single digits (logit, on the root route, takes 5). Contraction at
        // the pre-fix 0.4937/iteration needs ~50 steps to cover the same 8
        // orders, so this budget is what a return to a Hessian-free system would
        // trip.
        if fit.iteration > 30 {
            failures.push(format!("{link:?} took {} iterations", fit.iteration));
        }
        // Deliberately NOT a fixed gradient bound. The attainable `‖g‖` is set
        // by how precisely a minimum's LOCATION is determinable in f64 —
        // `~sqrt(eps)` relative, so `‖g‖ ~ sqrt(eps)·curvature ≈ 2e-8` here —
        // and it differs by route: the canonical link's augmented-root solve
        // never forms the cancelling coefficient-space gradient and reaches
        // 5.5e-14, while the dense route floors out near 2e-8. Asserting a
        // number below a route's own floor is the exact defect this issue is
        // about (see `exact_newton_decrement_threshold`), so the gradient is
        // reported and the CONTRACT — a certified inner mode — is asserted.
        if !fit.lastgradient_norm.is_finite() {
            failures.push(format!("{link:?} |g| is not finite"));
        }
    }
    assert!(
        failures.is_empty(),
        "#2273: the Firth inner solve does not converge alike on every binomial link \
         ({}):{report}",
        failures.join("; "),
    );
}

/// The value check, against a reference that shares no code with the engine.
///
/// `scipy.optimize.minimize(Nelder-Mead)` on `ℓ(β) + ½log|XᵀWX|` with the probit
/// Fisher weight `φ(η)²/(Φ(η)(1−Φ(η)))` gives, on this fixture,
///
/// ```text
///   beta = (−1.3e-9, 1.34683841)   max|eta| = 1.37652   deviance = 1.1201966
///   eig(I) = (1.91830, 1.92011)    cond = 1.00094
/// ```
///
/// The deviance and `max|η|` are invariant to how the design is scaled, so they
/// pin the engine's answer without assuming its coefficient frame. This is the
/// half of the claim the convergence test cannot make: that the mode P-IRLS now
/// certifies is the Firth mode and not merely a point it stopped moving from.
#[test]
fn firth_probit_inner_mode_matches_an_independent_reference_2273() {
    let fit = firth_inner_solve(StandardLink::Probit);
    // Nelder-Mead's own convergence limits the reference to ~1e-6 relative, so
    // the band is set by the reference, not by the engine.
    assert!(
        (fit.deviance - 1.1201966).abs() <= 5e-6,
        "#2273: Firth-probit deviance {:.9e} is not the independent reference 1.1201966 \
         (status {:?}, |g|={:.3e})",
        fit.deviance,
        fit.status,
        fit.lastgradient_norm,
    );
    assert!(
        (fit.max_abs_eta - 1.37652).abs() <= 1e-4,
        "#2273: Firth-probit max|eta| {:.6} is not the independent reference 1.37652",
        fit.max_abs_eta,
    );
}

/// The fold-in's sign convention, asserted where it is defined rather than
/// inferred from a fit. `objective_hessian_quadratic_correction` scores
/// `−dᵀHΦd`, so the matrix form must SUBTRACT `HΦ`; the two are one fact.
#[test]
fn objective_curvature_for_direction_subtracts_the_correction_2273() {
    use super::newton_solve::objective_curvature_for_direction;
    let hessian = ndarray::array![[4.0, 1.0], [1.0, 3.0]];
    let correction = ndarray::array![[1.0, 0.5], [0.5, 2.0]];

    let borrowed = objective_curvature_for_direction(&hessian, None)
        .expect("no correction is not an error");
    assert_eq!(borrowed.as_ref(), &hessian);
    assert!(
        matches!(borrowed, std::borrow::Cow::Borrowed(_)),
        "the uncorrected path must not allocate"
    );

    let folded = objective_curvature_for_direction(&hessian, Some(&correction))
        .expect("a conforming correction folds in");
    assert_eq!(folded.as_ref(), &ndarray::array![[3.0, 0.5], [0.5, 1.0]]);

    let mismatched = Array2::<f64>::zeros((3, 3));
    assert!(
        objective_curvature_for_direction(&hessian, Some(&mismatched)).is_err(),
        "a non-conforming correction must be refused, not broadcast"
    );
}

/// #2273: the noncanonical observed-information tower must evaluate on a
/// SATURATED Bernoulli row.
///
/// Two independent oracles, neither of which shares code with the engine.
///
/// **The exact one.** For a cloglog row with `y = 0` the log-likelihood is
/// `ℓ = log(1−μ) = −e^η` exactly, so
///
/// ```text
///   W_obs = −d²ℓ/dη² = e^η,  and every η-derivative of it is also e^η.
/// ```
///
/// That identity holds at every `η`, including where `V = μ(1−μ)` is `3.6e-84`,
/// so it pins the tower with no tolerance argument to make.
///
/// **The high-precision one.** For `y = 1` there is no such closed form, so the
/// reference is `mpmath` at 220 decimal digits (needed to hold `1 − 3.6e-84`
/// exactly and differentiate through it), computed from `1−μ = exp(−e^η)` and
/// `V = μ(1−μ)` with no gam code involved.
///
/// Before the tower rewrite this refused rather than answered: the closed forms
/// divided by `φV²`, `φV³` and `φV⁴`, and at `η = 5.2593` `V⁴ = 1.7e-334`
/// underflows to zero, so `d²W/dη²` came out NaN and the row was reported as
/// `PIRLS row geometry is not representable` — aborting a whole fit over a
/// number that is `3.87e-75`.
#[test]
fn noncanonical_observed_tower_evaluates_on_a_saturated_row_2273() {
    use gam_problem::InverseLink;

    // `η = 5.2593` is the row the #2273 cloglog n=100 fit refused; 3.6799 is the
    // one it refused before that; the last two are ordinary interior points, so
    // the same code path is checked where nothing is extreme.
    const ETAS: [f64; 4] = [5.259300664374346, 3.679931643574629, 1.25, -0.75];
    // mpmath at 220 dps, `y = 1`: (w_obs, c_obs, d_obs).
    const Y1_REFERENCE: [(f64, f64, f64); 4] = [
        (1.0732398640299e-79, -2.0428230353403e-77, 3.8677001533694e-75),
        (9.2943557520704e-15, -3.4963327742703e-13, 1.2783727780206e-11),
        (0.2854114158076, -0.39029979946258, -0.38865107866873),
        (0.19926932328323, 0.16289984062778, 0.091777041256882),
    ];

    let link = InverseLink::Standard(StandardLink::CLogLog);
    let mut report = String::new();
    for (index, &eta) in ETAS.iter().enumerate() {
        let jet = crate::mixture_link::inverse_link_jet_for_inverse_link(&link, eta)
            .expect("the cloglog jet is defined at these eta");
        let h4 = crate::mixture_link::inverse_link_pdfthird_derivative_for_inverse_link(&link, eta)
            .expect("the cloglog fourth derivative is defined at these eta");
        let one_minus_mu =
            crate::mixture_link::inverse_link_complement_for_inverse_link(&link, eta, jet.mu);
        let vj = variance_jet_for_weight_family(WeightFamily::Binomial, jet.mu, one_minus_mu);

        // y = 0: the exact identity.
        let zero = observed_weight_dispatch(
            WeightFamily::Binomial,
            WeightLink::Other,
            0.0,
            jet.mu,
            one_minus_mu,
            1.0,
            1.0,
            jet,
            h4,
        );
        let exact = eta.exp();
        report.push_str(&format!(
            "\n  eta={eta:>20}  V={:.6e}  y=0 -> ({:.9e}, {:.9e}, {:.9e})  exact e^eta={exact:.9e}",
            vj.v, zero.0, zero.1, zero.2,
        ));
        // The band widens by order, and the reason is arithmetic rather than
        // policy. For cloglog the recurrence's terms nearly cancel — analytically
        // `T₀ = T₁ = T₂ = T₃ = e^η/(1−(1−μ))`, and each order reaches it by
        // subtracting `t²s` from `t²s + ts` with `t = e^η ≈ 192`, so ~log₁₀(t)
        // ≈ 2.3 digits go per order. Measured worst case at this η: `w` to
        // 3e-14, `c` to 1.0e-11, `d` to 6.2e-9 — one order of loss per order of
        // derivative, exactly as the model predicts. That is the generic tower's
        // honest accuracy on a row whose variance is 2.9e-84; the alternative it
        // replaced was NaN.
        for (label, value, band) in
            [("w", zero.0, 1e-12), ("c", zero.1, 1e-10), ("d", zero.2, 1e-7)]
        {
            let relative = (value - exact).abs() / exact.abs();
            assert!(
                relative <= band,
                "#2273: cloglog y=0 {label}_obs at eta={eta} is {value:.9e}, but -l = e^eta \
                 exactly, so it must be {exact:.9e} (relative error {relative:.3e}, \
                 band {band:.0e}){report}"
            );
        }

        // y = 1: against the 220-digit reference.
        let one = observed_weight_dispatch(
            WeightFamily::Binomial,
            WeightLink::Other,
            1.0,
            jet.mu,
            one_minus_mu,
            1.0,
            1.0,
            jet,
            h4,
        );
        let (rw, rc, rd) = Y1_REFERENCE[index];
        report.push_str(&format!(
            "\n  eta={eta:>20}          y=1 -> ({:.9e}, {:.9e}, {:.9e})  mpmath ({rw:.9e}, {rc:.9e}, {rd:.9e})",
            one.0, one.1, one.2,
        ));
        for (label, value, reference, band) in [
            ("w", one.0, rw, 1e-10),
            ("c", one.1, rc, 1e-9),
            ("d", one.2, rd, 1e-7),
        ] {
            let relative = (value - reference).abs() / reference.abs();
            assert!(
                relative <= band,
                "#2273: cloglog y=1 {label}_obs at eta={eta} is {value:.9e} against the \
                 220-digit reference {reference:.9e} (relative error {relative:.3e}, \
                 band {band:.0e}){report}"
            );
        }
    }
    eprintln!("#2273 noncanonical observed tower on saturated cloglog rows:{report}");
}

/// #2273: the complement must reach the variance, not just the working response.
///
/// `inverse_link_complement_for_inverse_link` already existed and the Fisher
/// working-state path already used it; the observed-curvature path rebuilt the
/// variance from `mu` alone. This pins the consequence at the seam rather than
/// through a fit: past the point where `mu` rounds to exactly `1.0`, the
/// complement-fed variance must still be positive.
#[test]
fn saturated_binomial_variance_is_positive_past_the_mu_rounding_point_2273() {
    use gam_problem::InverseLink;

    for (link, first_saturated_eta) in [
        (StandardLink::CLogLog, 3.62_f64),
        (StandardLink::Probit, 8.29_f64),
    ] {
        let inverse_link = InverseLink::Standard(link);
        for eta in [first_saturated_eta, first_saturated_eta + 1.0, first_saturated_eta + 2.0] {
            let jet = crate::mixture_link::inverse_link_jet_for_inverse_link(&inverse_link, eta)
                .expect("the jet is defined");
            let complement = crate::mixture_link::inverse_link_complement_for_inverse_link(
                &inverse_link,
                eta,
                jet.mu,
            );
            let naive = VarianceJet::bernoulli(jet.mu);
            let paired = variance_jet_for_weight_family(WeightFamily::Binomial, jet.mu, complement);
            assert!(
                paired.v > 0.0 && paired.v.is_finite(),
                "#2273: {link:?} at eta={eta} has mu={:.17e}, complement={complement:.6e}, and the \
                 paired variance is {:.6e} — a saturated row's variance must stay positive",
                jet.mu,
                paired.v,
            );
            eprintln!(
                "#2273 saturated variance {link:?} eta={eta}: mu={:.17e} \
                 complement={complement:.6e} V_paired={:.6e} V_naive={:.6e}",
                jet.mu, paired.v, naive.v,
            );
        }
    }
}

/// #2273: the tail complement now covers the COMPOSITE bounded links too.
///
/// `inverse_link_complement_for_inverse_link` wired the standard links and SAS
/// and left the beta-logistic and mixture links on the naive `1.0 - mu`, so
/// those two kept the `V = μ(1−μ) → 0` refusal the standard links no longer
/// have. Both have exact identities:
///
/// * beta-logistic — `μ = I_x(a,b)` with `x = logistic(η)`, so
///   `1 − μ = I_{1−x}(b,a)` by the regularized incomplete beta's reflection, and
///   `1 − x = logistic(−η)` is already carried beside `x`;
/// * mixture — `μ = Σ πᵢ μᵢ`, so `1 − μ = (1 − Σ πᵢ) + Σ πᵢ (1 − μᵢ)`, and each
///   component is a standard link whose complement is already exact.
///
/// The invariant asserted is the one that makes them identities rather than
/// approximations: `1.0 - mu` is a subtraction of two numbers of size 1, so its
/// ABSOLUTE error is `~ε/2` no matter how small the true complement is, and an
/// exact complement must therefore agree with it to that absolute band — for
/// every `η`, saturated or not. That is a two-sided check with teeth: at
/// `η = 12` the beta-logistic naive complement is `5.195844e-14` and the identity
/// says `5.190486e-14`, agreeing to `5.4e-17` while DISAGREEING in the fourth
/// significant digit, because by then the naive value has only three digits left.
/// Past the rounding point the naive value is exactly zero and the identity is
/// still there, which is the whole point.
#[test]
fn composite_bounded_links_carry_an_exact_tail_complement_2273() {
    use gam_problem::{InverseLink, LinkComponent, MixtureLinkState, SasLinkState};
    use ndarray::array;

    let beta_logistic = InverseLink::BetaLogistic(SasLinkState {
        epsilon: 0.35,
        log_delta: 0.6,
        delta: 0.6_f64.exp(),
    });
    let mixture = InverseLink::Mixture(MixtureLinkState {
        components: vec![LinkComponent::CLogLog, LinkComponent::Probit],
        rho: array![0.4],
        pi: array![0.6, 0.4],
    });

    // `1.0 - mu` carries an absolute error of about half an ulp of 1; four of
    // them is the band an exact complement has to sit inside.
    let band = 4.0 * f64::EPSILON;
    for (label, link) in [("beta-logistic", beta_logistic), ("mixture", mixture)] {
        let mut recovered = 0usize;
        for eta in [-1.5_f64, -0.25, 0.0, 0.8, 2.0, 8.0, 12.0, 20.0, 40.0] {
            let Ok(jet) = crate::mixture_link::inverse_link_jet_for_inverse_link(&link, eta) else {
                continue;
            };
            let complement = crate::mixture_link::inverse_link_complement_for_inverse_link(
                &link, eta, jet.mu,
            );
            let naive = 1.0 - jet.mu;
            eprintln!(
                "#2273 composite complement {label} eta={eta}: mu={:.17e} \
                 complement={complement:.6e} naive={naive:.6e} diff={:.3e}",
                jet.mu,
                complement - naive,
            );
            assert!(
                complement >= 0.0 && complement.is_finite(),
                "#2273: {label} complement at eta={eta} is {complement:.6e}"
            );
            assert!(
                (complement - naive).abs() <= band,
                "#2273: {label} complement at eta={eta} is {complement:.17e} against a naive \
                 {naive:.17e}: they differ by {:.3e}, more than the naive subtraction's own \
                 {band:.3e} absolute resolution, so the identity is wrong rather than better \
                 conditioned",
                (complement - naive).abs(),
            );
            if naive == 0.0 && complement > 0.0 {
                recovered += 1;
            }
        }
        assert!(
            recovered > 0,
            "#2273: {label} never reached a saturated eta where the naive complement is a hard \
             zero, so this test is not exercising the repair"
        );
    }
}
