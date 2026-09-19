//! Oracles for the generic exponential-dispersion row kernel
//! ([`crate::edm_row`]).
//!
//! The generic kernel writes ONE expression, `s (y − μ) h′(η)/V(μ)`, and
//! derives every η-derivative of it by the jet algebra. Each oracle here
//! writes the same row's score in the family's own closed form instead — as
//! a function of `η` directly, never through `V` or the inverse-link stack —
//! and checks all five channels `(ℓ′, W_obs, c_obs, d_obs, e_obs)` and the
//! Fisher tower against it, to rounding:
//!
//! * the canonical cells the hand-written kernels serve (Gaussian-identity,
//!   Poisson-log, Gamma-log, Bernoulli logit / probit / cloglog), and
//! * the non-canonical cells only the generic kernel serves (Poisson-identity,
//!   Poisson-sqrt, Gamma-inverse, Gaussian-log, Bernoulli-log), whose
//!   derivatives are elementary power / exponential laws.
//!
//! For the probability links the closed-form score is additionally checked to
//! be the η-derivative of the Bernoulli log-likelihood itself, so the oracle
//! is anchored to the likelihood and not only to another score expression.

use crate::edm_row::{EdmRow, EdmVariance};
use crate::jet_scalar::JetScalar;
use crate::jet_tower::Tower4;
use crate::nested_dual::JetField;
use crate::probability::normal_cdf;

/// Relative-to-magnitude agreement: `|a − b| ≤ tol · max(1, |a|, |b|)`.
fn assert_close(label: &str, generic: f64, oracle: f64, tol: f64) {
    let scale = 1.0_f64.max(generic.abs()).max(oracle.abs());
    assert!(
        (generic - oracle).abs() <= tol * scale,
        "{label}: generic {generic:e} vs oracle {oracle:e}"
    );
}

/// The Taylor coefficients `[f, f′, f″, f‴, f⁗]` of a unary tower at its seed.
fn tower_stack(t: &Tower4<1>) -> [f64; 5] {
    [t.v, t.g[0], t.h[0][0], t.t3[0][0][0], t.t4[0][0][0][0]]
}

/// An inverse link, written as an explicit expression in `η` for the oracle
/// harness. Its six-term stack `[h, …, h⁽⁵⁾]` is read off two independent
/// towers: one of `h(η)` itself and one of the closed-form `h′(η)`.
#[derive(Clone, Copy, Debug)]
enum Link {
    Identity,
    Log,
    Sqrt,
    Inverse,
    Logit,
    Probit,
    Cloglog,
}

impl Link {
    /// `(h(η), h′(η))` as towers, and the cancellation-free `1 − h(η)`.
    fn towers(self, eta: f64) -> (Tower4<1>, Tower4<1>, f64) {
        let x = Tower4::<1>::variable(eta, 0);
        match self {
            Self::Identity => (x, Tower4::constant(1.0), 1.0 - eta),
            Self::Log => (x.exp(), x.exp(), -eta.exp_m1()),
            Self::Sqrt => (x.mul(&x), x.scale(2.0), 1.0 - eta * eta),
            Self::Inverse => {
                let r = x.recip();
                (r, r.mul(&r).neg(), 1.0 - 1.0 / eta)
            }
            Self::Logit => {
                // σ(η) = 1/(1 + e^{−η}),  σ′(η) = e^{−η}/(1 + e^{−η})².
                let e = x.neg().exp();
                let denom = e.add_constant(1.0);
                let sigma = denom.recip();
                let d = e.mul(&sigma).mul(&sigma);
                (sigma, d, e.mul(&sigma).value())
            }
            Self::Probit => {
                // Φ has no elementary jet; its stack is φ times the
                // probabilists' Hermite polynomials, Φ⁽ᵏ⁾ = (−1)ᵏ⁻¹ Heₖ₋₁ φ.
                let phi_value = (-0.5 * eta * eta).exp() / (2.0 * std::f64::consts::PI).sqrt();
                let he = [
                    1.0,
                    eta,
                    eta * eta - 1.0,
                    eta * eta * eta - 3.0 * eta,
                ];
                let cdf = x.compose_unary([
                    normal_cdf(eta),
                    phi_value,
                    -he[1] * phi_value,
                    he[2] * phi_value,
                    -he[3] * phi_value,
                ]);
                // φ(η) = exp(−η²/2)/√(2π) as an elementary expression.
                let pdf = x
                    .mul(&x)
                    .scale(-0.5)
                    .exp()
                    .scale(1.0 / (2.0 * std::f64::consts::PI).sqrt());
                (cdf, pdf, normal_cdf(-eta))
            }
            Self::Cloglog => {
                // h = 1 − exp(−e^η),  h′ = exp(η − e^η).
                let ee = x.exp();
                let survival = ee.neg().exp();
                let d = x.sub(&ee).exp();
                (survival.neg().add_constant(1.0), d, survival.value())
            }
        }
    }

    fn stack(self, eta: f64) -> ([f64; 6], f64) {
        let (h, dh, omm) = self.towers(eta);
        let a = tower_stack(&h);
        let b = tower_stack(&dh);
        // The two towers must agree on their overlap (h′ … h⁗) — a check
        // that the explicit `h′` expression is the derivative of `h`.
        for k in 0..4 {
            assert_close(&format!("{self:?} link stack overlap k={k}"), a[k + 1], b[k], 1e-12);
        }
        ([a[0], b[0], b[1], b[2], b[3], b[4]], omm)
    }
}

/// A cell's score `ℓ′(η)` written directly in `η` (no variance function, no
/// link stack), as a tower whose channels are `ℓ′ … ℓ⁽⁵⁾`.
fn direct_score(variance: EdmVariance, link: Link, y: f64, s: f64, eta: f64) -> Tower4<1> {
    let x = Tower4::<1>::variable(eta, 0);
    let score = match (variance, link) {
        // ℓ = −(y − η)²/2.
        (EdmVariance::Gaussian, Link::Identity) => x.neg().add_constant(y),
        // ℓ = −(y − e^η)²/2  ⇒  ℓ′ = (y − e^η) e^η.
        (EdmVariance::Gaussian, Link::Log) => x.exp().neg().add_constant(y).mul(&x.exp()),
        // ℓ = yη − e^η.
        (EdmVariance::Poisson, Link::Log) => x.exp().neg().add_constant(y),
        // ℓ = y ln η − η.
        (EdmVariance::Poisson, Link::Identity) => x.recip().scale(y).add_constant(-1.0),
        // ℓ = 2y ln η − η².
        (EdmVariance::Poisson, Link::Sqrt) => x.recip().scale(2.0 * y).sub(&x.scale(2.0)),
        // ℓ = −y e^{−η} − η.
        (EdmVariance::Gamma, Link::Log) => x.neg().exp().scale(y).add_constant(-1.0),
        // ℓ = −yη + ln η.
        (EdmVariance::Gamma, Link::Inverse) => x.recip().add_constant(-y),
        // ℓ = yη − ln(1 + e^η).
        (EdmVariance::Bernoulli, Link::Logit) => {
            x.neg().exp().add_constant(1.0).recip().neg().add_constant(y)
        }
        // ℓ = y ln Φ + (1 − y) ln(1 − Φ)  ⇒  ℓ′ = yφ/Φ − (1 − y)φ/(1 − Φ).
        (EdmVariance::Bernoulli, Link::Probit) => {
            let (cdf, pdf, _) = Link::Probit.towers(eta);
            let upper = cdf.neg().add_constant(1.0);
            pdf.mul(&cdf.recip())
                .scale(y)
                .sub(&pdf.mul(&upper.recip()).scale(1.0 - y))
        }
        // ℓ = y ln(1 − e^{−e^η}) − (1 − y) e^η  ⇒  ℓ′ = e^η [y/(e^{e^η} − 1) − (1 − y)].
        (EdmVariance::Bernoulli, Link::Cloglog) => {
            let ee = x.exp();
            ee.mul(&ee.exp().add_constant(-1.0).recip().scale(y).add_constant(y - 1.0))
        }
        // ℓ = yη + (1 − y) ln(1 − e^η)  ⇒  ℓ′ = y − (1 − y) e^η/(1 − e^η).
        (EdmVariance::Bernoulli, Link::Log) => {
            let e = x.exp();
            e.mul(&e.neg().add_constant(1.0).recip())
                .scale(y - 1.0)
                .add_constant(y)
        }
        other => panic!("no direct-score oracle for {other:?}"),
    };
    score.scale(s)
}

/// The closed-form Fisher weight `s h′(η)²/V(h(η))` as a tower in `η`.
fn direct_fisher(variance: EdmVariance, link: Link, s: f64, eta: f64) -> Tower4<1> {
    let x = Tower4::<1>::variable(eta, 0);
    let w = match (variance, link) {
        (EdmVariance::Gaussian, Link::Identity) => Tower4::constant(1.0),
        (EdmVariance::Gaussian, Link::Log) => x.scale(2.0).exp(),
        (EdmVariance::Poisson, Link::Log) => x.exp(),
        (EdmVariance::Poisson, Link::Identity) => x.recip(),
        (EdmVariance::Poisson, Link::Sqrt) => Tower4::constant(4.0),
        (EdmVariance::Gamma, Link::Log) => Tower4::constant(1.0),
        (EdmVariance::Gamma, Link::Inverse) => x.mul(&x).recip(),
        (EdmVariance::Bernoulli, Link::Logit) => {
            let (_, d, _) = Link::Logit.towers(eta);
            d
        }
        (EdmVariance::Bernoulli, Link::Probit) => {
            let (cdf, pdf, _) = Link::Probit.towers(eta);
            pdf.mul(&pdf)
                .mul(&cdf.mul(&cdf.neg().add_constant(1.0)).recip())
        }
        (EdmVariance::Bernoulli, Link::Cloglog) => {
            // h′²/(h(1 − h)) = e^{2η} e^{−e^η}/(1 − e^{−e^η}) = e^{2η}/(e^{e^η} − 1).
            let ee = x.exp();
            x.scale(2.0).exp().mul(&ee.exp().add_constant(-1.0).recip())
        }
        (EdmVariance::Bernoulli, Link::Log) => {
            // e^{2η}/(e^η(1 − e^η)) = e^η/(1 − e^η).
            let e = x.exp();
            e.mul(&e.neg().add_constant(1.0).recip())
        }
        other => panic!("no direct-Fisher oracle for {other:?}"),
    };
    w.scale(s)
}

fn check_cell(variance: EdmVariance, link: Link, y: f64, s: f64, eta: f64, tol: f64) {
    let (stack, omm) = link.stack(eta);
    let row = EdmRow {
        variance,
        y,
        scale: s,
        eta,
        link: stack,
        one_minus_mu: omm,
    };
    let label = format!("{variance:?}-{link:?} y={y} s={s} eta={eta}");
    let observed = row.observed_tower().expect("generic score tower");
    let oracle = tower_stack(&direct_score(variance, link, y, s, eta));
    assert_close(&format!("{label} score"), observed.score, oracle[0], tol);
    assert_close(&format!("{label} W_obs"), observed.w, -oracle[1], tol);
    assert_close(&format!("{label} c_obs"), observed.c, -oracle[2], tol);
    assert_close(&format!("{label} d_obs"), observed.d, -oracle[3], tol);
    assert_close(&format!("{label} e_obs"), observed.e, -oracle[4], tol);

    let fisher = row.fisher_tower().expect("generic Fisher tower");
    let fisher_oracle = tower_stack(&direct_fisher(variance, link, s, eta));
    assert_close(&format!("{label} W_F"), fisher.w, fisher_oracle[0], tol);
    assert_close(&format!("{label} dW_F"), fisher.c, fisher_oracle[1], tol);
    assert_close(&format!("{label} d2W_F"), fisher.d, fisher_oracle[2], tol);

    // At y = μ the observed and Fisher informations coincide.
    let at_mean = EdmRow { y: stack[0], ..row };
    if variance != EdmVariance::Bernoulli || (0.0..=1.0).contains(&stack[0]) {
        let obs = at_mean.observed_tower().expect("tower at the mean");
        assert_close(&format!("{label} W_obs(y=μ) = W_F"), obs.w, fisher.w, tol);
        assert_close(&format!("{label} score(y=μ) = 0"), obs.score, 0.0, tol);
    }
}

/// The rounding tolerance: every channel is a handful of exact jet products,
/// so agreement is to a small multiple of the unit roundoff.
const ROUNDING: f64 = 1e-11;

#[test]
fn generic_kernel_matches_the_canonical_hand_kernels() {
    for &eta in &[-1.3, -0.2, 0.4, 1.7] {
        for &s in &[1.0, 0.37] {
            check_cell(EdmVariance::Gaussian, Link::Identity, 0.8, s, eta, ROUNDING);
            check_cell(EdmVariance::Poisson, Link::Log, 3.0, s, eta, ROUNDING);
            check_cell(EdmVariance::Poisson, Link::Log, 0.0, s, eta, ROUNDING);
            check_cell(EdmVariance::Gamma, Link::Log, 2.4, s, eta, ROUNDING);
            for link in [Link::Logit, Link::Probit, Link::Cloglog] {
                check_cell(EdmVariance::Bernoulli, link, 1.0, s, eta, ROUNDING);
                check_cell(EdmVariance::Bernoulli, link, 0.0, s, eta, ROUNDING);
            }
        }
    }
}

#[test]
fn generic_kernel_matches_closed_forms_on_noncanonical_cells() {
    for &eta in &[0.3, 1.1, 2.6] {
        for &s in &[1.0, 2.5] {
            check_cell(EdmVariance::Poisson, Link::Identity, 4.0, s, eta, ROUNDING);
            check_cell(EdmVariance::Poisson, Link::Identity, 0.0, s, eta, ROUNDING);
            check_cell(EdmVariance::Poisson, Link::Sqrt, 2.0, s, eta, ROUNDING);
            check_cell(EdmVariance::Gamma, Link::Inverse, 1.9, s, eta, ROUNDING);
            check_cell(EdmVariance::Gaussian, Link::Log, 2.2, s, eta, ROUNDING);
        }
    }
    // The log link keeps a Bernoulli mean in (0, 1) only for η < 0. Near the
    // boundary the k-th channel carries terms of size (1 − μ)^{−k} that cancel
    // at y = 1 (where ℓ′ ≡ 1), so the points stay where that conditioning is
    // O(10³) and the rounding tolerance is meaningful.
    for &eta in &[-3.0, -0.9, -0.3] {
        for &y in &[0.0, 1.0] {
            check_cell(EdmVariance::Bernoulli, Link::Log, y, 1.0, eta, ROUNDING);
        }
    }
}

/// The closed-form probability-link scores are themselves checked against the
/// Bernoulli log-likelihood's η-derivatives (through ℓ⁗, the tower's reach),
/// so the canonical oracle is anchored to the likelihood.
#[test]
fn probability_link_oracles_are_likelihood_derivatives() {
    for link in [Link::Logit, Link::Probit, Link::Cloglog, Link::Log] {
        for &eta in &[-1.4, -0.3] {
            for &y in &[0.0, 1.0] {
                let (h, _, _) = link.towers(eta);
                let upper = h.neg().add_constant(1.0);
                let loglik = h.ln().scale(y).add(&upper.ln().scale(1.0 - y));
                let l = tower_stack(&loglik);
                let score = tower_stack(&direct_score(EdmVariance::Bernoulli, link, y, 1.0, eta));
                for k in 0..4 {
                    assert_close(
                        &format!("{link:?} y={y} eta={eta} ℓ^({})", k + 1),
                        score[k],
                        l[k + 1],
                        ROUNDING,
                    );
                }
            }
        }
    }
}

#[test]
fn unit_deviance_matches_twice_the_loglik_gap() {
    // d(y, μ) = 2[ℓ(y; y) − ℓ(y; μ)] for the per-unit log-likelihood kernels.
    let cases = [
        (EdmVariance::Gaussian, 1.3, 0.4),
        (EdmVariance::Poisson, 3.0, 1.7),
        (EdmVariance::Poisson, 0.0, 1.7),
        (EdmVariance::Gamma, 2.0, 0.6),
        (EdmVariance::InverseGaussian, 2.0, 0.6),
        (EdmVariance::Bernoulli, 1.0, 0.3),
        (EdmVariance::Bernoulli, 0.0, 0.3),
    ];
    for (variance, y, mu) in cases {
        let loglik = |m: f64| -> f64 {
            match variance {
                EdmVariance::Gaussian => -0.5 * (y - m) * (y - m),
                EdmVariance::Poisson => {
                    if y == 0.0 {
                        -m
                    } else {
                        y * m.ln() - m
                    }
                }
                EdmVariance::Gamma => -y / m - m.ln(),
                EdmVariance::InverseGaussian => -y / (2.0 * m * m) + 1.0 / m,
                EdmVariance::Bernoulli => {
                    let a = if y == 0.0 { 0.0 } else { y * m.ln() };
                    let b = if y == 1.0 { 0.0 } else { (1.0 - y) * (1.0 - m).ln() };
                    a + b
                }
            }
        };
        let expected = 2.0 * (loglik(y) - loglik(mu));
        assert_close(
            &format!("{variance:?} unit deviance"),
            variance.unit_deviance(y, mu, 1.0 - mu),
            expected,
            ROUNDING,
        );
    }
}
