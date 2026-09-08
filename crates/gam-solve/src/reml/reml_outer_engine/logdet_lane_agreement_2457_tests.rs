//! #2457 — the value-only and derivative-bearing lanes must price ONE `log|H|`.
//!
//! `certify_outer_optimality` evaluates the outer criterion twice at the same
//! ρ — once at `OuterEvalOrder::Value`, once at `ValueAndGradient` — and
//! `audit_outer_value_agreement` refuses the fit when the two disagree by more
//! than `outer_value_agreement_bound`. The two lanes reach two different
//! `HessianFactorization`s: the LLT fast path returns the EXACT `Σ ln σ_j`,
//! while `DenseSpectralOperator` returns the smooth-floored `Σ ln r_ε(σ_j)`
//! that the analytic gradient (`tr(G_ε Ḣ)`) and Hessian actually
//! differentiate. Wherever the floor bites those are two different functions
//! of ρ, so the criterion is not single-valued at the point every downstream
//! decision reasons about.
//!
//! Measured instance — `gam::identifiability`
//! `smooths::constant_curvature_smooth::kappa_zero_fit_recovers_planted_flat_signal`:
//! value-only `−9.8977232379128066e2` against analytic-sample
//! `−9.8976254249264389e2`, a `9.781e-3` gap versus a `1.475e-5` envelope
//! (663×). `log|H|` enters the criterion as `+½log|H|` and `r_ε(σ) ≥ σ`, so
//! the derivative lane must read HIGHER — which is the observed sign — and the
//! implied `Δlog|H| = 1.956e-2` is one eigenvalue sitting at ≈`7ε`.

use super::*;
use ndarray::{Array1, Array2};

/// The smooth floor `ε = √(f64::EPSILON)·p` is a function of the DIMENSION
/// alone, so a fixture can place an eigenvalue at an exact multiple of it.
const FIXTURE_DIM: usize = 32;

/// A deterministic, non-trivial orthogonal basis: one Householder reflector
/// `I − 2vvᵀ/vᵀv` with `v = (1, 2, …, n)`.
fn householder_basis(n: usize) -> Array2<f64> {
    let v: Array1<f64> = Array1::from_iter((0..n).map(|i| (i + 1) as f64));
    let vtv = v.dot(&v);
    let mut q = Array2::<f64>::eye(n);
    for i in 0..n {
        for j in 0..n {
            q[[i, j]] -= 2.0 * v[i] * v[j] / vtv;
        }
    }
    q
}

/// `Q diag(σ) Qᵀ`, symmetrized so the LLT sees an exactly symmetric input.
fn spd_with_spectrum(sigma: &[f64]) -> Array2<f64> {
    let n = sigma.len();
    let q = householder_basis(n);
    let mut scaled = q.clone();
    for j in 0..n {
        for i in 0..n {
            scaled[[i, j]] *= sigma[j];
        }
    }
    let h = scaled.dot(&q.t());
    let mut symmetric = Array2::<f64>::zeros((n, n));
    for i in 0..n {
        for j in 0..n {
            symmetric[[i, j]] = 0.5 * (h[[i, j]] + h[[j, i]]);
        }
    }
    symmetric
}

/// Every eigenvalue orders of magnitude above the floor. This is the only
/// regime the pre-existing agreement test for issue #277
/// (`dense_cholesky_value_only_matches_spectral`, a 4×4 with eigenvalues 2–8)
/// ever exercised, which is how the fast path kept a proof of agreement while
/// production disagreed by 663×.
fn well_conditioned_spectrum() -> Vec<f64> {
    (0..FIXTURE_DIM)
        .map(|i| 10f64.powf(-2.0 + 4.0 * i as f64 / (FIXTURE_DIM - 1) as f64))
        .collect()
}

/// The same spectrum with its smallest eigenvalue moved onto the smooth floor
/// (`7ε`) — what a 30-centre radial basis at `λ = e^−8.9` does to
/// `H = XᵀWX + S_λ`. Chosen so the exact/floored gap lands on the `1.956e-2`
/// the failing fit implies.
fn floor_touching_spectrum() -> Vec<f64> {
    let epsilon = spectral_epsilon_for_dim(FIXTURE_DIM);
    let mut sigma: Vec<f64> = (0..FIXTURE_DIM - 1)
        .map(|i| 10f64.powf(-2.0 + 4.0 * i as f64 / (FIXTURE_DIM - 2) as f64))
        .collect();
    sigma.push(7.0 * epsilon);
    sigma
}

/// Exactly the operator selection `build_dense_assembly` and
/// `build_dense_original_assembly` make on a value-only evaluation: try the
/// LLT fast path, fall back to the spectral operator on any decline.
fn value_lane_logdet(h: &Array2<f64>) -> f64 {
    match DenseCholeskyOperator::from_spd_with_smooth_logdet_agreement(h) {
        Ok(operator) => operator.logdet(),
        Err(_) => derivative_lane_logdet(h),
    }
}

fn derivative_lane_logdet(h: &Array2<f64>) -> f64 {
    DenseSpectralOperator::from_symmetric(h)
        .expect("fixture Hessian must decompose")
        .logdet()
}

/// The unguarded exact log-determinant the fast path returned before #2457.
fn raw_cholesky_logdet(h: &Array2<f64>) -> f64 {
    use faer::Side;
    use gam_linalg::faer_ndarray::FaerCholesky;
    let chol = h.cholesky(Side::Lower).expect("fixture must be SPD");
    2.0 * chol.diag().iter().map(|d| d.ln()).sum::<f64>()
}

/// THE CONTRACT. Whatever operator the value-only lane installs must price
/// `log|H|` within the SAME envelope the outer audit applies to the criterion
/// that log-determinant feeds — on a well-conditioned Hessian *and* on one
/// whose smallest eigenvalue sits on the smooth floor.
#[test]
fn value_and_derivative_lanes_price_one_logdet_2457() {
    for (label, sigma) in [
        ("well-conditioned", well_conditioned_spectrum()),
        ("floor-touching", floor_touching_spectrum()),
    ] {
        let h = spd_with_spectrum(&sigma);
        let value = value_lane_logdet(&h);
        let derivative = derivative_lane_logdet(&h);
        let gap = (value - derivative).abs();
        let envelope = crate::rho_optimizer::outer_value_agreement_bound(value, derivative);
        assert!(
            gap <= envelope,
            "{label}: the value lane and the derivative lane price different log|H| \
             (value {value:.17e}, derivative {derivative:.17e}, gap {gap:.3e} = {:.1}x the \
             {envelope:.3e} envelope) — the outer criterion is not a function of rho",
            gap / envelope
        );
    }
}

#[test]
fn strong_penalty_roundoff_uses_one_logdet_kernel_2834() {
    // Every eigenvalue is far above the smooth floor. The competing source of
    // disagreement is the assembled matrix's condition number, as on the
    // shared/group smooth fixture when its deviation penalty tends to infinity.
    let rotation = ndarray::array![
        [0.5, 0.5, 0.5, 0.5],
        [0.5, -0.5, 0.5, -0.5],
        [0.5, 0.5, -0.5, -0.5],
        [0.5, -0.5, -0.5, 0.5]
    ];
    for strength in [1e10, 1e12] {
        let h = rotation
            .dot(&Array2::from_diag(&ndarray::array![1.0, 2.0, 3.0, strength]))
            .dot(&rotation.t());
        assert!(DenseCholeskyOperator::from_spd_with_smooth_logdet_agreement(&h).is_err());
        assert_eq!(value_lane_logdet(&h), derivative_lane_logdet(&h));
    }
    let benign = Array2::from_diag(&ndarray::array![1.0, 2.0, 3.0, 4.0]);
    assert!(DenseCholeskyOperator::from_spd_with_smooth_logdet_agreement(&benign).is_ok());
}

/// CONTROL — the fixture must be able to FAIL, or the assertion above proves
/// nothing. On the floor-touching Hessian the raw LLT log-determinant really
/// does disagree with the floored one by more than the very envelope the
/// production audit refuses on; on the well-conditioned one it does not.
#[test]
fn only_a_floor_touching_hessian_separates_the_two_logdets_2457() {
    let benign = spd_with_spectrum(&well_conditioned_spectrum());
    let benign_raw = raw_cholesky_logdet(&benign);
    let benign_floored = derivative_lane_logdet(&benign);
    assert!(
        (benign_raw - benign_floored).abs()
            <= crate::rho_optimizer::outer_value_agreement_bound(benign_raw, benign_floored),
        "the well-conditioned arm must NOT discriminate, or it is not a control \
         (raw {benign_raw:.17e}, floored {benign_floored:.17e})"
    );
    assert!(
        DenseCholeskyOperator::from_spd_with_smooth_logdet_agreement(&benign).is_ok(),
        "a well-conditioned Hessian must keep the LLT fast path — the #2457 guard is \
         one-sided and must not cost the speedup where the floor cannot bite"
    );

    let hard = spd_with_spectrum(&floor_touching_spectrum());
    let hard_raw = raw_cholesky_logdet(&hard);
    let hard_floored = derivative_lane_logdet(&hard);
    let gap = (hard_raw - hard_floored).abs();
    let envelope = crate::rho_optimizer::outer_value_agreement_bound(hard_raw, hard_floored);
    assert!(
        gap > envelope,
        "the floor-touching arm must be an input the production audit REFUSES, or it \
         cannot witness #2457 (gap {gap:.3e}, envelope {envelope:.3e})"
    );
    assert!(
        hard_floored > hard_raw,
        "the smooth floor can only RAISE log|H| (r_eps(sigma) >= sigma), so the derivative \
         lane must read higher — the sign the failing fit reported \
         (floored {hard_floored:.17e}, exact {hard_raw:.17e})"
    );
    assert!(
        DenseCholeskyOperator::from_spd_with_smooth_logdet_agreement(&hard).is_err(),
        "the fast path must DECLINE a Hessian whose floored and exact log-determinants \
         differ, rather than return a scalar no derivative lane agrees with"
    );
}
