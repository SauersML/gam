//! #3090 — the direct parametric-AFT MLE's step on an indefinite Hessian.
//!
//! The solve used to add a Levenberg ridge `τ·I` whose starting value, growth
//! factor and cap were tuned constants. It now takes the absolute-value Newton
//! step `Q|Λ|⁻¹Qᵀg`, which needs no constant and is an ascent direction for any
//! symmetric `H = −∇²ℓ`.

use super::*;
use crate::survival::location_scale::family_solver::aft_absolute_newton_direction;

fn rotation(angle: f64) -> Array2<f64> {
    let (s, c) = angle.sin_cos();
    array![[c, -s], [s, c]]
}

/// `Q diag(lambda) Qᵀ`, assembled BITWISE symmetric.
///
/// `aft_absolute_newton_direction` hands its argument to
/// `strict_symmetric_eigh` under `SymmetricAssembly::Mirrored`, whose band is
/// exactly zero because a mirrored assembly writes one rounded value into both
/// triangles. Two chained GEMMs do not: entry `(j, k)` of `(Q·D)·Qᵀ` and entry
/// `(k, j)` accumulate the same products in a different association, and IEEE
/// multiplication is commutative but not associative, so the two triangles are
/// free to differ in the last bit. They do here, and the eigendecomposition
/// refused this fixture rather than the code under test.
///
/// `(M + Mᵀ)/2` fixes it without changing the matrix: addition commutes, so the
/// two triangles are the same rounded sum, and halving is exact in binary. The
/// spectrum the assertions below read is unchanged to within one ulp.
fn symmetric_spectral(q: &Array2<f64>, lambda: &Array1<f64>) -> Array2<f64> {
    let m = q.dot(&Array2::from_diag(lambda)).dot(&q.t());
    (&m + &m.t()) * 0.5
}

#[test]
fn absolute_newton_direction_flips_negative_curvature_3090() {
    let q = rotation(0.3);
    let lambda = [3.0, -0.5];
    let h = symmetric_spectral(&q, &Array1::from(lambda.to_vec()));
    let g = array![0.7, -1.1];

    let step = aft_absolute_newton_direction(&h, &g, 0).expect("regular indefinite H");
    let c = q.t().dot(&g);
    let expected = q.dot(&array![c[0] / 3.0, c[1] / 0.5]);
    for k in 0..2 {
        assert!(
            (step.delta[k] - expected[k]).abs() <= 1e-14,
            "δ[{k}] = {} != Q|Λ|⁻¹Qᵀg = {}",
            step.delta[k],
            expected[k]
        );
    }
    assert_eq!(step.negative_curvature, 1);
    assert!((step.min_eigenvalue + 0.5).abs() <= 1e-14);
    // Ascent: g·δ = Σ cᵢ²/|λᵢ| > 0, whereas the Newton step H⁻¹g descends here.
    let decrement = g.dot(&step.delta);
    assert!((decrement - (c[0] * c[0] / 3.0 + c[1] * c[1] / 0.5)).abs() <= 1e-14);
    let newton = q.dot(&array![c[0] / 3.0, c[1] / -0.5]);
    assert!(g.dot(&newton) < 0.0, "the bare Newton step is not an ascent step here");
}

#[test]
fn absolute_newton_direction_is_newton_on_positive_definite_hessian_3090() {
    let h = array![[4.0, 1.0], [1.0, 3.0]];
    let g = array![1.0, -2.0];
    let step = aft_absolute_newton_direction(&h, &g, 0).expect("PD H");
    // H⁻¹g with det 11.
    let newton = array![(3.0 * 1.0 - 1.0 * -2.0) / 11.0, (-1.0 * 1.0 + 4.0 * -2.0) / 11.0];
    for k in 0..2 {
        assert!((step.delta[k] - newton[k]).abs() <= 1e-14);
    }
    assert_eq!(step.negative_curvature, 0);
}

#[test]
fn absolute_newton_direction_refuses_gradient_along_flat_curvature_3090() {
    let q = rotation(0.3);
    let h = symmetric_spectral(&q, &array![2.0, 0.0]);
    // Gradient in the range of H: the flat direction carries no slope, so the
    // step is the pseudo-inverse one.
    let g_range = q.column(0).to_owned() * 1.5;
    let step = aft_absolute_newton_direction(&h, &g_range, 0).expect("flat direction, no slope");
    for k in 0..2 {
        assert!((step.delta[k] - q[[k, 0]] * 0.75).abs() <= 1e-14);
    }
    // A slope along the flat direction has no Newton maximizer.
    let g_flat = &g_range + &(q.column(1).to_owned() * 0.25);
    let error = aft_absolute_newton_direction(&h, &g_flat, 0)
        .err()
        .expect("slope along a flat direction must be refused");
    assert!(
        error.to_string().contains("no resolved curvature"),
        "unexpected error: {error}"
    );
}

/// The gate this file's fixtures stand in front of, asserted from BOTH sides.
///
/// `symmetric_spectral` above keeps the fixtures inside the contract
/// `aft_absolute_newton_direction` declares; production matrices reach it from
/// the row-kernel pullback, whose `(a, b)` channel loop accumulates entry
/// `(i, j)` and entry `(j, i)` separately and so leaves them free to differ in
/// the last bit. `mirror_pullback_in_place` is what that route now applies. This
/// shows the refusal is real on the asymmetric matrix and gone on the mirrored
/// one, so the mirror is what the gate ACCEPTS rather than something it
/// tolerates, and neither half of that claim rests on the other.
#[test]
fn a_pullback_mirrored_in_place_is_what_the_strict_symmetry_gate_accepts_1561() {
    use crate::survival::location_scale::row_kernel::mirror_pullback_in_place;
    // The two values the survival census recorded on the production refusal:
    // one ulp apart, which is what two independent accumulations of one real
    // number differ by.
    let mut h: ndarray::Array2<f64> =
        array![[1.0, -0.024554355106785223], [-0.024554355106785226, 2.0]];
    assert_ne!(
        h[[0, 1]].to_bits(),
        h[[1, 0]].to_bits(),
        "precondition: the fixture must carry the asymmetry the route produced, \
         or the refusal below proves nothing"
    );
    let g = array![0.3, -0.4];
    let refusal = aft_absolute_newton_direction(&h, &g, 0)
        .err()
        .expect("a one-ulp asymmetry must be refused under a band of exactly zero");
    assert!(
        refusal.to_string().contains("not symmetric"),
        "the refusal must name the mechanism, got {refusal}"
    );
    mirror_pullback_in_place(&mut h);
    assert_eq!(
        h[[0, 1]].to_bits(),
        h[[1, 0]].to_bits(),
        "a mirrored pullback writes ONE computed value into both triangles"
    );
    aft_absolute_newton_direction(&h, &g, 0)
        .expect("the mirrored pullback satisfies the band its consumer declares");
}

/// End to end: start the location intercept three standard deviations off the
/// mean. For the lognormal AFT in `(μ, log σ)` the NLL Hessian has determinant
/// `∝ S/n − (ȳ − μ)²`, so this start has an indefinite `H`; the fit must still
/// reach the closed-form MLE.
#[test]
fn reduced_parametric_aft_converges_from_indefinite_start_3090() {
    let n = 400usize;
    let (age_exit, event, log_t) = reduced_aft_lognormal_sample(n, 1.4, 0.5, 3090);
    let (mu_hat, sigma_hat) = lognormal_closed_form_mle(&log_t);
    let mu_start = mu_hat + 3.0 * sigma_hat;
    let log_sigma_start = sigma_hat.ln();
    let mut spec = reduced_aft_lognormal_spec(&age_exit, &event, 1.0);
    let CovariateBlockKind::Static(threshold) = &mut spec.threshold_block else {
        panic!("static threshold block expected");
    };
    threshold.initial_beta = Some(array![mu_start]);
    let CovariateBlockKind::Static(log_sigma) = &mut spec.log_sigma_block else {
        panic!("static log-sigma block expected");
    };
    log_sigma.initial_beta = Some(array![log_sigma_start]);

    // The start is indefinite: det ∝ S/n − (ȳ − μ)² = σ̂²(1 − 9) < 0.
    let d = mu_hat - mu_start;
    assert!(sigma_hat * sigma_hat - d * d < 0.0);

    let prepared = prepare_survival_location_scale_model(&spec).expect("prepare");
    assert!(prepared.is_reduced_parametric_aft());
    let fit = fit_survival_location_scale_spec(spec)
        .unwrap_or_else(|e| panic!("reduced parametric-AFT MLE from an indefinite start: {e}"));
    let loc = fit.beta_threshold()[0];
    let sigma = fit.beta_log_sigma()[0].exp();
    assert!(
        (loc - mu_hat).abs() < 1e-6,
        "location {loc:.9} != closed-form mu {mu_hat:.9}"
    );
    assert!(
        (sigma - sigma_hat).abs() < 1e-6,
        "sigma {sigma:.9} != closed-form sigma {sigma_hat:.9}"
    );
}

/// The log-t collapse check projects `log t` onto the null-space warp images.
/// A null direction with a zero warp image makes `GᵀG` singular; the minimum-norm
/// projection drops it (it used to be ridged by `1e-10·max diag`) and still
/// recognizes the log-t baseline in the other direction.
#[test]
fn reduced_warp_logt_projection_drops_zero_image_direction_3090() {
    let log_t = array![-0.4, 0.1, 0.7, 1.2, 1.9];
    let n = log_t.len();
    let mut design_exit = Array2::<f64>::zeros((n, 2));
    for i in 0..n {
        design_exit[[i, 0]] = 2.0 * log_t[i] + 0.3;
    }
    let z = Array2::<f64>::eye(2);
    assert!(reduced_warp_logt_baseline_usable(&z, &design_exit, log_t.view()));
    // Nothing but a zero warp image: no log-t slope to find.
    let zero_image = Array2::<f64>::zeros((n, 2));
    assert!(!reduced_warp_logt_baseline_usable(&z, &zero_image, log_t.view()));
}
