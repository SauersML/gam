//! Shared test-only oracles for the gamlss family stack.
//!
//! Items here exist purely to pin production fast paths against a dense,
//! single-source reference. They are exercised from more than one gamlss test
//! module (the `dispersion_family` unit tests and the family-level behaviour
//! tests in `tests.rs`), so they live at the common parent rather than as
//! `#[cfg(test)]` items dangling off a production `src/` module — which is the
//! shape `dead_code` cannot see and the build-time ban-scanner rejects.

use gam_math::jet_scalar::JetScalar;
use gam_problem::InverseLink;
use gam_math::nested_dual::JetField;
use statrs::function::gamma::ln_gamma;

use super::dispersion_family::DispersionFamilyKind;

/// Dense all-channel binomial location-scale oracle shared by the row-program
/// pins and the family-level behavior tests. It spells the row NLL in predictor
/// coordinates — `q = −η_t·e^{−η_ls}` through the algebra's own `exp`, then one
/// composition with the q-space stack `[−ℓ, m1, m2, m3, m4]` — so it shares
/// neither the local-coordinate map nor the lowering of the production
/// `binomial_ls_row`.
#[inline]
pub(crate) fn binomial_location_scale_nll_tower(
    y: f64,
    weight: f64,
    eta_t: f64,
    eta_ls: f64,
    q_value: f64,
    mu: f64,
    dmu_dq: f64,
    d2mu_dq2: f64,
    d3mu_dq3: f64,
    link_kind: &InverseLink,
    include_fourth: bool,
) -> Result<gam_math::jet_tower::Tower4<2>, String> {
    use gam_math::jet_tower::Tower4;

    binomial_location_scale_nll_in_predictors::<Tower4<2>>(
        y,
        weight,
        eta_t,
        eta_ls,
        q_value,
        mu,
        dmu_dq,
        d2mu_dq2,
        d3mu_dq3,
        link_kind,
        include_fourth,
    )
}

#[inline]
fn binomial_location_scale_nll_in_predictors<S: JetScalar<2>>(
    y: f64,
    weight: f64,
    eta_t: f64,
    eta_ls: f64,
    q_value: f64,
    mu: f64,
    dmu_dq: f64,
    d2mu_dq2: f64,
    d3mu_dq3: f64,
    link_kind: &InverseLink,
    include_fourth: bool,
) -> Result<S, String> {
    let q = S::variable(eta_t, 0)
        .neg()
        .mul(&S::variable(eta_ls, 1).scale(-1.0).exp());
    let neg_ll =
        -super::binomial_location_scale_log_likelihood(y, weight, q_value, link_kind, mu)?;
    let (m1, m2, m3) = super::binomial_neglog_q_derivatives_dispatch(
        y, weight, q_value, mu, dmu_dq, d2mu_dq2, d3mu_dq3, link_kind,
    );
    let m4 = if include_fourth {
        super::binomial_neglog_q_fourth_derivative_dispatch(
            y, weight, q_value, mu, dmu_dq, d2mu_dq2, d3mu_dq3, link_kind,
        )?
    } else {
        0.0
    };
    Ok(q.compose_unary([neg_ll, m1, m2, m3, m4]))
}

/// Test oracle for the Tweedie series `ln W(y, η_d)` as a jet in `η_d` (#3511).
///
/// `W = Σ_{j≥1} e^{z_j}` with `z_j = j (c₀ + η_d/(p−1)) − ln Γ(j+1) − ln Γ(jα)`,
/// `α = (2−p)/(p−1)` and `c₀ = α (ln y − ln(p−1)) − ln(2−p)`, so each term is
/// the algebra's own `exp` of an affine function of `η_d` and every derivative
/// channel comes from the jet arithmetic, not from the production cumulant
/// formulas. The walk runs upward from `j = 1` and stops once a term is `TAIL`
/// nats below the running maximum: `z_j` is concave in `j`, so every later term
/// is smaller still and falls geometrically, and `e^{−100}` is far below the
/// double-precision resolution of the sum. The terms are scaled by the maximum
/// before summation so the sum is formed at unit magnitude.
pub(crate) fn tweedie_log_series_jet<S: JetScalar<K>, const K: usize>(
    eta_d: &S,
    yi: f64,
    p: f64,
) -> S {
    const TAIL: f64 = 100.0;
    let alpha = (2.0 - p) / (p - 1.0);
    let rate = 1.0 / (p - 1.0);
    let c0 = alpha * (yi.ln() - (p - 1.0).ln()) - (2.0 - p).ln();
    let c = c0 + rate * eta_d.value();
    let mut z_ref = f64::NEG_INFINITY;
    let mut last = 1usize;
    loop {
        let jf = last as f64;
        let z = jf * c - ln_gamma(jf + 1.0) - ln_gamma(jf * alpha);
        assert!(z.is_finite(), "Tweedie series oracle term {last} is not finite");
        if z > z_ref {
            z_ref = z;
        } else if z < z_ref - TAIL {
            break;
        }
        last += 1;
        assert!(last < 10_000_000, "Tweedie series oracle did not reach its tail");
    }
    let mut series: Option<S> = None;
    for j in 1..=last {
        let jf = j as f64;
        let term = eta_d
            .scale(jf * rate)
            .add_constant(jf * c0 - ln_gamma(jf + 1.0) - ln_gamma(jf * alpha) - z_ref)
            .exp();
        series = Some(match series {
            None => term,
            Some(sum) => sum.add(&term),
        });
    }
    series
        .expect("the Tweedie series oracle sums at least one term")
        .ln()
        .add_constant(z_ref)
}

/// Tweedie compound Poisson–Gamma row NLL written ONCE over a generic
/// [`JetScalar<2>`], seeded directly on the PREDICTOR primaries `(η_μ, η_d)`
/// (#932).
///
/// Unlike the NB/Gamma/Beta oracles — which seed on the natural parameters and
/// let the caller apply the precision→η chain via the Fisher-orthogonal
/// `precision²·info` shortcut — this tower carries `μ = exp(η_μ)` and
/// `κ = 1/φ = exp(η_d)` INSIDE the program, so `tower.g[1]` / `tower.h[1][1]`
/// are the η_d-space score and OBSERVED information directly.
///
/// Both density branches are the exact Tweedie density (#3511):
/// * `y > 0` — `ℓ = w·[ −κ (μ^{2−p}/(2−p) − y μ^{1−p}/(1−p)) + ln W(y, η_d) − ln y ]`
///   with the series `ln W` from [`tweedie_log_series_jet`].
/// * `y = 0` — the compound-Poisson point mass `ℓ = w·[ −μ^{2−p}/(φ(2−p)) ]`.
///
/// Production consumes the pruned single-axis `dispersion_tweedie_disp_order2`;
/// this `K=2` generic is the dense oracle / cross-tool witness that pins it.
#[inline]
pub(crate) fn dispersion_tweedie_nll_generic<S: JetScalar<2>>(
    yi: f64,
    eta_mu: f64,
    eta_d: f64,
    p: f64,
    wi: f64,
) -> S {
    let one_minus_p = 1.0 - p;
    let two_minus_p = 2.0 - p;
    // μ = exp(η_μ) and the precision predictor η_d as jets in the predictor
    // primaries, so the whole derivative tower is in η-space.
    let mu = S::variable(eta_mu, 0).exp();
    let eta_d = S::variable(eta_d, 1);
    if yi > 0.0 {
        let kappa = eta_d.exp();
        let kernel = mu
            .powf(two_minus_p)
            .scale(1.0 / two_minus_p)
            .sub(&mu.powf(one_minus_p).scale(yi / one_minus_p));
        let loglik = kappa
            .mul(&kernel)
            .neg()
            .add(&tweedie_log_series_jet::<S, 2>(&eta_d, yi, p))
            .sub(&S::constant(yi.ln()));
        loglik.scale(-wi)
    } else {
        // Exact point mass P(Y=0) = exp(−μ^{2−p}/(φ(2−p))).
        let phi = eta_d.scale(-1.0).exp();
        let c = mu.powf(two_minus_p).scale(1.0 / two_minus_p);
        let loglik = c.mul(&phi.recip()).scale(-1.0);
        loglik.scale(-wi)
    }
}

/// `ln Γ` lifted onto an `Order2<K>` jet by the production `JetScalar::ln_gamma`,
/// shared by the dispersion-family tower oracles.
#[inline]
pub(crate) fn order2_ln_gamma<const K: usize>(
    x: &gam_math::jet_scalar::Order2<K>,
) -> gam_math::jet_scalar::Order2<K> {
    gam_math::jet_scalar::JetScalar::ln_gamma(x)
}

/// `[g, g′, g″, g‴, g⁗]` of the Stirling gap `g(x) = x ln x − x − ln Γ(x)` at
/// `x`, the `compose_unary` stack the Gamma/Beta gap-form oracles lift onto a
/// jet (#4252).
#[inline]
pub(crate) fn stirling_gap_jet_stack(x: f64) -> [f64; 5] {
    let derivatives = gam_math::special::stirling_gap_derivative_stack(x, 4);
    [
        gam_math::special::stirling_gap(x),
        derivatives[0],
        derivatives[1],
        derivatives[2],
        derivatives[3],
    ]
}

/// `[L₁, L₁′, L₁″, L₁‴, L₁⁗]` of `L₁(x) = ln(1 + x) − x` at `x`, with
/// `L₁′ = −x/(1 + x)` and `L₁⁽ᵏ⁾ = (−1)^{k−1}(k − 1)!/(1 + x)^k` for `k ≥ 2`.
#[inline]
pub(crate) fn log1p_minus_x_jet_stack(x: f64) -> [f64; 5] {
    let r = 1.0 / (1.0 + x);
    [
        gam_math::special::log1p_minus_x(x),
        -x * r,
        -r * r,
        2.0 * r * r * r,
        -6.0 * r * r * r * r,
    ]
}

/// Observed η-space row NLL tower, both predictors as jet variables (`η_μ` axis 0,
/// `η_d` axis 1). Oracle for the row-program production row derivatives in
/// `dispersion_family` (`eta_space_row_program_derivatives_match_the_towers`).
pub(crate) fn dispersion_eta_nll_order2(
    kind: DispersionFamilyKind,
    yi: f64,
    em: f64,
    ed: f64,
    wi: f64,
) -> gam_math::jet_scalar::Order2<2> {
    type O2 = gam_math::jet_scalar::Order2<2>;
    let eta_mu = O2::variable(em, 0);
    let eta_d = O2::variable(ed, 1);
    match kind {
        DispersionFamilyKind::NegativeBinomial => {
            // The NB log-likelihood below is written directly in the linear
            // predictors (log-scale) via `log_total`, so the mean `exp(eta_mu)`
            // is never materialized here (unlike the Gamma arm).
            let theta = eta_d.exp();
            let theta_plus_y = theta.add(&O2::constant(yi));
            let log_total = if em >= ed {
                eta_mu.add(&eta_d.sub(&eta_mu).exp().add(&O2::constant(1.0)).ln())
            } else {
                eta_d.add(&eta_mu.sub(&eta_d).exp().add(&O2::constant(1.0)).ln())
            };
            let loglik = order2_ln_gamma(&theta_plus_y)
                .sub(&order2_ln_gamma(&theta))
                .sub(&O2::constant(ln_gamma(yi + 1.0)))
                .add(&theta.mul(&eta_d.sub(&log_total)))
                .add(&eta_mu.sub(&log_total).scale(yi));
            loglik.scale(-wi)
        }
        DispersionFamilyKind::Gamma => {
            let mu = eta_mu.exp();
            let nu = eta_d.exp();
            let y_pos = yi;
            let loglik = nu
                .mul(&nu.ln())
                .sub(&nu.mul(&mu.ln()))
                .sub(&order2_ln_gamma(&nu))
                .add(&nu.sub(&O2::constant(1.0)).scale(y_pos.ln()))
                .sub(&nu.mul(&mu.recip().scale(yi)));
            loglik.scale(-wi)
        }
        DispersionFamilyKind::Beta => {
            let mu = eta_mu.scale(-1.0).exp().add(&O2::constant(1.0)).recip();
            let phi = eta_d.exp();
            let one_minus_mu = O2::constant(1.0).sub(&mu);
            let yc = yi;
            let a = mu.mul(&phi);
            let b = one_minus_mu.mul(&phi);
            let loglik = order2_ln_gamma(&phi)
                .sub(&order2_ln_gamma(&a))
                .sub(&order2_ln_gamma(&b))
                .add(&a.sub(&O2::constant(1.0)).scale(yc.ln()))
                .add(&b.sub(&O2::constant(1.0)).scale((-yc).ln_1p()));
            loglik.scale(-wi)
        }
        DispersionFamilyKind::Tweedie { p } => {
            let one_minus_p = 1.0 - p;
            let two_minus_p = 2.0 - p;
            let mu = eta_mu.exp();
            let phi = eta_d.scale(-1.0).exp();
            if yi > 0.0 {
                let kernel = mu
                    .powf(two_minus_p)
                    .scale(1.0 / two_minus_p)
                    .sub(&mu.powf(one_minus_p).scale(yi / one_minus_p));
                let loglik = eta_d
                    .exp()
                    .mul(&kernel)
                    .neg()
                    .add(&tweedie_log_series_jet::<O2, 2>(&eta_d, yi, p))
                    .sub(&O2::constant(yi.ln()));
                loglik.scale(-wi)
            } else {
                let c = mu.powf(two_minus_p).scale(1.0 / two_minus_p);
                let loglik = c.mul(&phi.recip()).scale(-1.0);
                loglik.scale(-wi)
            }
        }
    }
}

/// Order-3 alias for the two-predictor η-space NLL tower.
type O3 = gam_math::jet_tower::Tower3<2>;

fn o3_exp(x: &O3) -> O3 {
    x.compose_unary_with(|v| {
        let e = v.exp();
        [e, e, e, e]
    })
}

fn o3_ln(x: &O3) -> O3 {
    x.compose_unary_with(|v| [v.ln(), v.recip(), -v.powi(-2), 2.0 * v.powi(-3)])
}

fn o3_recip(x: &O3) -> O3 {
    x.compose_unary_with(|v| [v.recip(), -v.powi(-2), 2.0 * v.powi(-3), -6.0 * v.powi(-4)])
}

fn o3_powf(x: &O3, a: f64) -> O3 {
    x.compose_unary_with(|v| {
        [
            v.powf(a),
            a * v.powf(a - 1.0),
            a * (a - 1.0) * v.powf(a - 2.0),
            a * (a - 1.0) * (a - 2.0) * v.powf(a - 3.0),
        ]
    })
}

fn o3_ln_gamma(x: &O3) -> O3 {
    gam_math::jet_scalar::JetScalar::ln_gamma(x)
}

/// Observed η-space row NLL tower to third order, the order-3 sibling of
/// [`dispersion_eta_nll_order2`] with the identical expression structure per
/// family, so `t3` is the per-row tensor `∂³NLL/∂η_a∂η_b∂η_c`. Oracle for the
/// row-program production directional Hessian derivative.
pub(crate) fn dispersion_eta_nll_order3(
    kind: DispersionFamilyKind,
    yi: f64,
    em: f64,
    ed: f64,
    wi: f64,
) -> gam_math::jet_tower::Tower3<2> {
    let eta_mu = O3::variable(em, 0);
    let eta_d = O3::variable(ed, 1);
    match kind {
        DispersionFamilyKind::NegativeBinomial => {
            let theta = o3_exp(&eta_d);
            let theta_plus_y = theta.add(&O3::constant(yi));
            let log_total = if em >= ed {
                eta_mu.add(&o3_ln(
                    &o3_exp(&eta_d.sub(&eta_mu)).add(&O3::constant(1.0)),
                ))
            } else {
                eta_d.add(&o3_ln(
                    &o3_exp(&eta_mu.sub(&eta_d)).add(&O3::constant(1.0)),
                ))
            };
            let loglik = o3_ln_gamma(&theta_plus_y)
                .sub(&o3_ln_gamma(&theta))
                .sub(&O3::constant(ln_gamma(yi + 1.0)))
                .add(&theta.mul(&eta_d.sub(&log_total)))
                .add(&eta_mu.sub(&log_total).scale(yi));
            loglik.scale(-wi)
        }
        DispersionFamilyKind::Gamma => {
            let mu = o3_exp(&eta_mu);
            let nu = o3_exp(&eta_d);
            let y_pos = yi;
            let loglik = nu
                .mul(&o3_ln(&nu))
                .sub(&nu.mul(&o3_ln(&mu)))
                .sub(&o3_ln_gamma(&nu))
                .add(&nu.sub(&O3::constant(1.0)).scale(y_pos.ln()))
                .sub(&nu.mul(&o3_recip(&mu).scale(yi)));
            loglik.scale(-wi)
        }
        DispersionFamilyKind::Beta => {
            let mu = o3_recip(&o3_exp(&eta_mu.scale(-1.0)).add(&O3::constant(1.0)));
            let phi = o3_exp(&eta_d);
            let one_minus_mu = O3::constant(1.0).sub(&mu);
            let yc = yi;
            let a = mu.mul(&phi);
            let b = one_minus_mu.mul(&phi);
            let loglik = o3_ln_gamma(&phi)
                .sub(&o3_ln_gamma(&a))
                .sub(&o3_ln_gamma(&b))
                .add(&a.sub(&O3::constant(1.0)).scale(yc.ln()))
                .add(&b.sub(&O3::constant(1.0)).scale((-yc).ln_1p()));
            loglik.scale(-wi)
        }
        DispersionFamilyKind::Tweedie { p } => {
            let one_minus_p = 1.0 - p;
            let two_minus_p = 2.0 - p;
            let mu = o3_exp(&eta_mu);
            let phi = o3_exp(&eta_d.scale(-1.0));
            if yi > 0.0 {
                let kernel = o3_powf(&mu, two_minus_p)
                    .scale(1.0 / two_minus_p)
                    .sub(&o3_powf(&mu, one_minus_p).scale(yi / one_minus_p));
                let loglik = o3_exp(&eta_d)
                    .mul(&kernel)
                    .neg()
                    .add(&tweedie_log_series_jet::<O3, 2>(&eta_d, yi, p))
                    .sub(&O3::constant(yi.ln()));
                loglik.scale(-wi)
            } else {
                let c = o3_powf(&mu, two_minus_p).scale(1.0 / two_minus_p);
                let loglik = c.mul(&o3_recip(&phi)).scale(-1.0);
                loglik.scale(-wi)
            }
        }
    }
}
