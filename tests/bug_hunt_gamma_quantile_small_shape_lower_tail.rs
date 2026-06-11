//! Bug hunt: `gam::inference::probability::gamma_quantile` must invert the Gamma
//! CDF in the *lower tail* at small shape (`a ≲ 0.1`).
//!
//! A quantile `Q(p)` of `Gamma(shape = a, scale = 1)` is, by definition, the
//! unique `x ≥ 0` with `P(a, x) = p`, where `P(a, x) = γ(a, x)/Γ(a)` is the
//! regularized lower incomplete gamma (the Gamma CDF). So the round-trip
//! `P(a, gamma_quantile(p, a, 1)) == p` must hold.
//!
//! For strongly over-dispersed Gamma/Tweedie predictives the moment-matched
//! shape `k = μ²/V` drops below `0.1`, and there the lower-tail quantile was
//! pinned near `~1e-15` regardless of how much smaller the true quantile is, so
//! the nominal 2.5% bound carried up to ~19% of the mass and the predictive
//! interval under-covered on the low side (#1018).
//!
//! The reference below is a fully self-contained regularized lower incomplete
//! gamma `P(a, x)` (Numerical Recipes `gammp`: the small-`x`/`x<a+1` power
//! series and the large-`x` Lentz continued fraction), with a Lanczos `ln Γ`.
//! It shares no code with the routine under test, so the round-trip identity it
//! checks is an independent oracle.

use gam::inference::probability::gamma_quantile;

/// Lanczos approximation to `ln Γ(z)` for `z > 0` (g = 7, n = 9), accurate to
/// ~1e-15 relative across the range exercised here.
fn ln_gamma(z: f64) -> f64 {
    const G: f64 = 7.0;
    const C: [f64; 9] = [
        0.999_999_999_999_809_93,
        676.520_368_121_885_1,
        -1_259.139_216_722_402_8,
        771.323_428_777_653_13,
        -176.615_029_162_140_6,
        12.507_343_278_686_905,
        -0.138_571_095_265_720_12,
        9.984_369_578_019_572e-6,
        1.505_632_735_149_311_6e-7,
    ];
    // Reflection is unnecessary: every argument here (`a`, `a + 1`) is positive.
    let z = z - 1.0;
    let mut x = C[0];
    for (i, c) in C.iter().enumerate().skip(1) {
        x += c / (z + i as f64);
    }
    let t = z + G + 0.5;
    0.5 * (2.0 * std::f64::consts::PI).ln() + (z + 0.5) * t.ln() - t + x.ln()
}

/// Regularized lower incomplete gamma `P(a, x) = γ(a, x)/Γ(a)`, the Gamma CDF.
/// Numerical Recipes `gammp`: power series for `x < a + 1`, continued fraction
/// (modified Lentz) otherwise. Pure reference — no dependency on the library.
fn reg_lower_gamma(a: f64, x: f64) -> f64 {
    assert!(a > 0.0 && x >= 0.0);
    if x == 0.0 {
        return 0.0;
    }
    let gln = ln_gamma(a);
    if x < a + 1.0 {
        // Power series: P(a,x) = x^a e^{-x} / Γ(a) · Σ_{n≥0} x^n / (a)(a+1)…(a+n).
        let mut ap = a;
        let mut sum = 1.0 / a;
        let mut del = sum;
        for _ in 0..10_000 {
            ap += 1.0;
            del *= x / ap;
            sum += del;
            if del.abs() < sum.abs() * 1e-16 {
                break;
            }
        }
        // exp(-x + a ln x - ln Γ(a)) is finite even when x^a e^{-x} underflows
        // only after dividing by Γ(a); here we keep it in logs to the end.
        (sum.ln() - x + a * x.ln() - gln).exp()
    } else {
        // Continued fraction for Q(a,x) = 1 - P(a,x).
        const TINY: f64 = 1e-300;
        let mut b = x + 1.0 - a;
        let mut c = 1.0 / TINY;
        let mut d = 1.0 / b;
        let mut h = d;
        for i in 1..10_000 {
            let an = -(i as f64) * (i as f64 - a);
            b += 2.0;
            d = an * d + b;
            if d.abs() < TINY {
                d = TINY;
            }
            c = b + an / c;
            if c.abs() < TINY {
                c = TINY;
            }
            d = 1.0 / d;
            let del = d * c;
            h *= del;
            if (del - 1.0).abs() < 1e-16 {
                break;
            }
        }
        let q = (a * x.ln() - x - gln).exp() * h;
        1.0 - q
    }
}

/// Sanity-check the independent reference against a couple of closed forms so a
/// bug in the *oracle* can never masquerade as a pass.
#[test]
fn reference_cdf_matches_closed_forms() {
    // Gamma(1) is Exp(1): P(1, x) = 1 - e^{-x}.
    for &x in &[0.1_f64, 0.5, 1.0, 2.0, 5.0] {
        let got = reg_lower_gamma(1.0, x);
        let want = 1.0 - (-x).exp();
        assert!(
            (got - want).abs() < 1e-12,
            "P(1,{x}) = {got}, expected {want}"
        );
    }
    // Small-x leading order: P(a, x) ≈ x^a / Γ(a+1).
    let (a, x) = (0.05_f64, 1e-20_f64);
    let got = reg_lower_gamma(a, x);
    let want = (a * x.ln() - ln_gamma(a + 1.0)).exp();
    assert!(
        (got - want).abs() < 1e-6 * want,
        "small-x leading order: P({a},{x}) = {got}, leading {want}"
    );
}

/// The bug: in the working regime the inversion identity is already exact, so
/// this guard documents that the reference and the round-trip agree where the
/// library works.
#[test]
fn gamma_quantile_is_correct_in_the_working_regime() {
    let cases = [
        (2.0_f64, 0.025_f64),
        (2.0, 0.5),
        (2.0, 0.975),
        (1.0, 0.1),
        (1.0, 0.9),
        (0.5, 0.5),
        (0.5, 0.975),
        (0.3, 0.5),
    ];
    for &(a, p) in &cases {
        let q = gamma_quantile(p, a, 1.0);
        let cdf = reg_lower_gamma(a, q);
        assert!(
            (cdf - p).abs() < 1e-9,
            "working regime a={a} p={p}: q={q}, P(a,q)={cdf}, off by {}",
            cdf - p
        );
    }
}

/// The failing case: small shape, lower tail. The returned quantile must invert
/// the CDF — `P(a, gamma_quantile(p, a, 1)) ≈ p` — even when the true quantile
/// underflows far below `~1e-15`.
#[test]
fn gamma_quantile_inverts_its_cdf_at_small_shape() {
    let cases = [
        (0.05_f64, 0.025_f64),
        (0.05, 0.05),
        (0.05, 0.10),
        (0.10, 0.025),
        (0.10, 0.05),
        (0.02, 0.025),
        (0.02, 0.10),
    ];
    for &(a, p) in &cases {
        let q = gamma_quantile(p, a, 1.0);
        assert!(q.is_finite() && q >= 0.0, "a={a} p={p}: non-finite q={q}");
        let cdf = reg_lower_gamma(a, q);
        assert!(
            (cdf - p).abs() < 1e-9,
            "small shape a={a} p={p}: q={q}, P(a,q)={cdf}, off by {} (want ~0)",
            cdf - p
        );
    }
}
