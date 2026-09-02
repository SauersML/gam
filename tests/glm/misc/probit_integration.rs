use gam::pirls::update_glmvectors_by_family;
use gam::types::{GlmLikelihoodSpec, InverseLink, LikelihoodSpec, ResponseFamily, StandardLink};
use ndarray::Array1;

#[test]
fn probitworkingvectors_are_finite_for_extreme_eta() {
    // Eta laid out so we can index the limit cases directly:
    //   0 -> -100 (saturated low), 1 -> -20, 2 -> 0 (peak weight),
    //   3 -> +20, 4 -> +100 (saturated high)
    let y = Array1::from_vec(vec![0.0, 1.0, 0.0, 1.0, 1.0]);
    let eta = Array1::from_vec(vec![-100.0, -20.0, 0.0, 20.0, 100.0]);
    let w = Array1::ones(y.len());
    let mut mu = Array1::zeros(y.len());
    let mut weights = Array1::zeros(y.len());
    let mut z = Array1::zeros(y.len());

    update_glmvectors_by_family(
        y.view(),
        &eta,
        &GlmLikelihoodSpec::canonical(LikelihoodSpec::new(
            ResponseFamily::Binomial,
            InverseLink::Standard(StandardLink::Probit),
        )),
        w.view(),
        &mut mu,
        &mut weights,
        &mut z,
    )
    .expect("probit working-vector update should succeed");

    // --- Finiteness / bounds (preserved from the original smoke test) ---
    assert!(
        mu.iter().all(|v| v.is_finite() && *v >= 0.0 && *v <= 1.0),
        "probit mu out of [0,1] or non-finite: mu={mu:?}"
    );
    assert!(
        weights.iter().all(|v| v.is_finite() && *v >= 0.0),
        "probit weights non-finite or negative: weights={weights:?}"
    );
    assert!(
        z.iter().all(|v| v.is_finite()),
        "probit z non-finite: z={z:?}"
    );

    // --- Mathematical contract for Φ ---
    // mu must implement the standard normal CDF, so the saturated tails
    // collapse to {0, 1} and the centered eta gives 0.5 exactly.
    // Saturation bounds: tolerances are sized to permit any sane numerical
    // clamp the implementation might apply for downstream log-domain safety
    // (epsilons up to ~1e-6) while still failing if the link has the wrong
    // shape (e.g. returning 0.5 everywhere).
    assert!(
        mu[0] < 1e-6,
        "Φ(-100) must collapse to ~0; got mu[0]={}",
        mu[0]
    );
    assert!(mu[1] < 1e-6, "Φ(-20) must be tiny; got mu[1]={}", mu[1]);
    assert!(
        (mu[2] - 0.5).abs() < 1e-9,
        "Φ(0) must equal 0.5 within fp tol; got mu[2]={}",
        mu[2]
    );
    assert!(mu[3] > 1.0 - 1e-6, "Φ(+20) must be ~1; got mu[3]={}", mu[3]);
    assert!(
        mu[4] > 1.0 - 1e-6,
        "Φ(+100) must collapse to ~1; got mu[4]={}",
        mu[4]
    );

    // mu must be monotonically non-decreasing in eta (probit link is
    // strictly increasing).
    for i in 1..mu.len() {
        assert!(
            mu[i] >= mu[i - 1] - 1e-15,
            "probit mu must be non-decreasing in eta; mu[{i}]={mu_i} < mu[{prev_i}]={mu_prev}",
            mu_i = mu[i],
            prev_i = i - 1,
            mu_prev = mu[i - 1]
        );
    }

    // --- IRLS weight contract ---
    // The canonical probit IRLS weight is φ(η)² / [Φ(η)(1-Φ(η))]. At η=0
    // this is (1/√(2π))² / 0.25 = 2/π ≈ 0.6366197723675814 — a closed-form
    // value we can pin exactly. Tying weights[2] to this constant catches
    // implementations that use a non-canonical formula (e.g. forgetting
    // φ², dividing by Φ alone, etc.) that monotonicity alone would miss.
    //
    // The weight is also multiplied by the prior weight (here 1.0), so the
    // constant 2/π is the value we should observe.
    let expected_w_at_zero = 2.0 / std::f64::consts::PI;
    assert!(
        (weights[2] - expected_w_at_zero).abs() < 1e-10,
        "probit IRLS weight at η=0 must equal 2/π = {expected_w_at_zero}; got {}",
        weights[2]
    );

    // The deepest-saturation endpoints (±100) must have ~zero weight. Any
    // reasonable implementation (with or without Mills-ratio fallback)
    // returns a weight close to zero there. The intermediate ±20 region is
    // implementation-dependent (clamp vs Mills-ratio asymptotic) so we do
    // not constrain it here.
    assert!(
        weights[0] < 1e-6,
        "probit weight at η=-100 must be ~0; got w[0]={}",
        weights[0]
    );
    assert!(
        weights[4] < 1e-6,
        "probit weight at η=+100 must be ~0; got w[4]={}",
        weights[4]
    );

    // --- Working-response sign contract ---
    // For a saturated η (μ ≈ y), the residual (y-μ) is tiny; combined with a
    // tiny weight, z is allowed to be wide, but the sign of (z - η) must
    // match the sign of (y - μ): if y is the larger class, z should pull
    // upward (z - η > 0), otherwise downward.
    for i in 0..y.len() {
        let residual = y[i] - mu[i];
        if residual.abs() > 1e-9 {
            let pull = z[i] - eta[i];
            assert!(
                pull * residual >= -1e-12,
                "probit working response must pull η toward y on row {i}: \
                 y={}, mu={}, eta={}, z={}, pull={}, residual={}",
                y[i],
                mu[i],
                eta[i],
                z[i],
                pull,
                residual,
            );
        }
    }
}

#[test]
fn cloglogworkingvectors_are_finite_for_extreme_eta() {
    // Same eta layout as the probit test: index 2 is η=0 where the
    // canonical cloglog mean is exactly 1 - 1/e.
    let y = Array1::from_vec(vec![0.0, 1.0, 0.0, 1.0, 1.0]);
    let eta = Array1::from_vec(vec![-100.0, -20.0, 0.0, 20.0, 100.0]);
    let w = Array1::ones(y.len());
    let mut mu = Array1::zeros(y.len());
    let mut weights = Array1::zeros(y.len());
    let mut z = Array1::zeros(y.len());

    update_glmvectors_by_family(
        y.view(),
        &eta,
        &GlmLikelihoodSpec::canonical(LikelihoodSpec::new(
            ResponseFamily::Binomial,
            InverseLink::Standard(StandardLink::CLogLog),
        )),
        w.view(),
        &mut mu,
        &mut weights,
        &mut z,
    )
    .expect("cloglog working-vector update should succeed");

    // --- Finiteness / bounds (preserved from the original smoke test) ---
    assert!(
        mu.iter().all(|v| v.is_finite() && *v >= 0.0 && *v <= 1.0),
        "cloglog mu out of [0,1] or non-finite: mu={mu:?}"
    );
    assert!(
        weights.iter().all(|v| v.is_finite() && *v >= 0.0),
        "cloglog weights non-finite or negative: weights={weights:?}"
    );
    assert!(
        z.iter().all(|v| v.is_finite()),
        "cloglog z non-finite: z={z:?}"
    );

    // --- Mathematical contract for the cloglog mean function ---
    // μ(η) = 1 - exp(-exp(η))
    //   η = -100: exp(-exp(-100)) ≈ exp(-tiny) ≈ 1, so μ ≈ 0
    //   η =    0: μ = 1 - 1/e
    //   η = +100: μ ≈ 1
    // Same conservative tolerance scheme as the probit test: leave room
    // for any numerical clamp the implementation might apply.
    assert!(
        mu[0] < 1e-6,
        "cloglog μ(-100) must collapse to ~0; got mu[0]={}",
        mu[0]
    );
    let expected_zero = 1.0 - (-1.0_f64).exp();
    assert!(
        (mu[2] - expected_zero).abs() < 1e-9,
        "cloglog μ(0) must equal 1 - exp(-1) = {expected_zero}; got mu[2]={}",
        mu[2]
    );
    assert!(
        mu[3] > 1.0 - 1e-3,
        "cloglog μ(+20) must be ~1 (exp(20) saturates the inner exp); got mu[3]={}",
        mu[3]
    );
    assert!(
        mu[4] > 1.0 - 1e-6,
        "cloglog μ(+100) must collapse to ~1; got mu[4]={}",
        mu[4]
    );

    // mu must be monotonically non-decreasing in eta (cloglog link is
    // strictly increasing).
    for i in 1..mu.len() {
        assert!(
            mu[i] >= mu[i - 1] - 1e-15,
            "cloglog mu must be non-decreasing in eta; mu[{i}]={mu_i} < mu[{prev_i}]={mu_prev}",
            mu_i = mu[i],
            prev_i = i - 1,
            mu_prev = mu[i - 1]
        );
    }
}
