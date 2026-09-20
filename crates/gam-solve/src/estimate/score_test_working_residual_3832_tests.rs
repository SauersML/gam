//! gam#3832: the fit persists the working residual that the smooth score test's
//! estimated scale is read from, and that residual is the one the test's
//! algebra needs.
//!
//! With `d = (H − G)β̂ = S(λ)β̂ = XᵀW(z − Xβ̂)` at the penalized optimum, the full
//! model's unpenalized residual is
//!
//!     D′ = min_β ‖z − Xβ‖²_W = ‖z − Xβ̂‖²_W − dᵀG⁺d,      ν = n⁺ − rank(G).
//!
//! The score test forms `D′` from the persisted `(H, G, β̂)` and the persisted
//! `‖z − Xβ̂‖²_W`. This fixture fits a penalized Gaussian model with zero-weight
//! rows and an offset, and checks the persisted scalars against the data: `n⁺`
//! counts the positive-weight rows, the norm is the penalized weighted RSS, and
//! `D′` read off the saved fit is the unpenalized weighted least-squares RSS
//! solved directly. The saved form round-trips the residual.

use super::*;
use gam_linalg::faer_ndarray::FaerEigh;
use gam_problem::{InverseLink, LikelihoodSpec, ResponseFamily, StandardLink};
use gam_terms::smooth::BlockwisePenalty;
use ndarray::{Array1, Array2};
use rand::rngs::StdRng;
use rand::{RngExt, SeedableRng};

const N: usize = 120;
const P: usize = 10;
const ZERO_WEIGHT_ROWS: [usize; 4] = [3, 41, 77, 118];

/// Fourier columns on a grid, a truth in the first two harmonics, varied
/// positive weights, four zero-weight rows whose responses are gross outliers
/// (they must carry nothing), and a nonzero offset.
fn fixture() -> (Array2<f64>, Array1<f64>, Array1<f64>, Array1<f64>) {
    let mut rng = StdRng::seed_from_u64(0x3832_0002);
    let mut x = Array2::<f64>::zeros((N, P));
    let mut y = Array1::<f64>::zeros(N);
    let mut weights = Array1::<f64>::zeros(N);
    let mut offset = Array1::<f64>::zeros(N);
    for i in 0..N {
        let t = i as f64 / (N - 1) as f64;
        x[[i, 0]] = 1.0;
        for j in 1..P {
            let arg = std::f64::consts::PI * ((j + 1) / 2) as f64 * t;
            x[[i, j]] = if j % 2 == 1 { arg.sin() } else { arg.cos() };
        }
        let u1: f64 = rng.random::<f64>().max(f64::MIN_POSITIVE);
        let u2: f64 = rng.random::<f64>();
        let noise = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
        offset[i] = 0.5 * t;
        y[i] = offset[i] + x[[i, 1]] + 0.5 * x[[i, 2]] + 0.4 * noise;
        weights[i] = 0.5 + rng.random::<f64>();
    }
    for &row in &ZERO_WEIGHT_ROWS {
        weights[row] = 0.0;
        y[row] = 1.0e3;
    }
    (x, y, weights, offset)
}

/// `vᵀA⁻¹v` of a symmetric positive-definite `A`, from its spectrum.
fn inverse_form(a: &Array2<f64>, v: &Array1<f64>) -> f64 {
    let (values, vectors) = a.eigh(faer::Side::Lower).expect("eigendecomposition");
    values
        .iter()
        .enumerate()
        .map(|(k, &value)| {
            let projection = vectors.column(k).dot(v);
            projection * projection / value
        })
        .sum()
}

#[test]
fn the_fit_persists_the_working_residual_of_its_unpenalized_scale() {
    let (x, y, weights, offset) = fixture();
    let fit = fit_gamwith_heuristic_log_lambdas(
        x.clone(),
        y.view(),
        weights.view(),
        offset.view(),
        &[BlockwisePenalty::new(1..P, Array2::eye(P - 1))],
        None,
        LikelihoodSpec::new(
            ResponseFamily::Gaussian,
            InverseLink::Standard(StandardLink::Identity),
        ),
        &FitOptions {
            compute_inference: true,
            max_iter: 200,
            tol: 1e-11,
            nullspace_dims: vec![0],
            ..FitOptions::default()
        },
    )
    .expect("penalized Gaussian fit");
    let residual = fit
        .inference
        .as_ref()
        .and_then(|inference| inference.working_residual)
        .expect("a Gaussian fit persists its working residual");

    // n⁺: a zero-weight row is an absent row.
    let positive = weights.iter().filter(|&&w| w > 0.0).count();
    assert_eq!(residual.rows, positive);
    assert_eq!(residual.rows, N - ZERO_WEIGHT_ROWS.len());

    // The data side, formed directly: z = y − offset, G_d = XᵀWX,
    // e = XᵀW(z − Xβ̂), the penalized RSS and the unpenalized WLS RSS.
    let z = &y - &offset;
    let fitted = x.dot(&fit.beta);
    let penalized_rss: f64 = (0..N).map(|i| weights[i] * (z[i] - fitted[i]).powi(2)).sum();
    let response_norm: f64 = (0..N).map(|i| weights[i] * z[i] * z[i]).sum();
    let weighted_x = &x * &weights.view().insert_axis(ndarray::Axis(1));
    let gram_direct = x.t().dot(&weighted_x);
    let e = weighted_x.t().dot(&(&z - &fitted));
    let (gram_values, gram_vectors) = gram_direct.eigh(faer::Side::Lower).expect("eigh");
    let lambda_min = gram_values.iter().copied().fold(f64::INFINITY, f64::min);
    let lambda_max = gram_values.iter().copied().fold(0.0_f64, f64::max);
    assert!(lambda_min > 0.0, "the fixture's design has full column rank");
    let condition = lambda_max / lambda_min;
    let mut beta_unpenalized = Array1::<f64>::zeros(P);
    let rhs = weighted_x.t().dot(&z);
    for k in 0..P {
        let column = gram_vectors.column(k);
        beta_unpenalized.scaled_add(column.dot(&rhs) / gram_values[k], &column);
    }
    let fitted_unpenalized = x.dot(&beta_unpenalized);
    let unpenalized_rss: f64 = (0..N)
        .map(|i| weights[i] * (z[i] - fitted_unpenalized[i]).powi(2))
        .sum();

    // Every sum above accumulates at most n·p products of terms bounded by
    // ‖z‖²_W, and the solves amplify that by cond(G).
    let roundoff = condition * gam_linalg::roundoff::accumulation_band(N * P, response_norm);

    // The persisted norm is the penalized weighted RSS (with the offset).
    assert!(
        (residual.weighted_norm - penalized_rss).abs() <= roundoff,
        "persisted ‖z − Xβ̂‖²_W = {} but the data give {penalized_rss} (band {roundoff:e})",
        residual.weighted_norm
    );

    // D′ read off the saved fit, exactly as the score test forms it.
    let hessian = fit
        .saved_frame_penalized_hessian()
        .expect("saved-frame Hessian")
        .expect("inference carries the Hessian")
        .into_owned();
    let gram = fit
        .saved_frame_weighted_gram()
        .expect("saved-frame Gram")
        .expect("inference carries the Gram")
        .into_owned();
    let beta = fit.beta_from_gauge_shift().expect("gauge shift").into_owned();
    let d = (&hessian - &gram).dot(&beta);
    let explained = inverse_form(&gram, &d);
    let d_prime = residual.weighted_norm - explained;

    // First-order perturbation bound of `D′ − D′_direct`, where exact algebra
    // gives `D′_direct = RSS_pen − eᵀG_d⁻¹e`: the saved `d` misses `e` by the
    // fit's measured stationarity defect `δ = e − d`, contributing
    // `2|dᵀG⁻¹δ| + δᵀG⁻¹δ`, and the saved Gram misses `G_d` by `ΔG`,
    // contributing at most `2‖d‖²‖ΔG‖₂/λ_min²` while `‖ΔG‖₂ ≤ λ_min/2`.
    let delta = &e - &d;
    let delta_form = inverse_form(&gram_direct, &delta);
    let gram_defect = (&gram - &gram_direct).mapv(|v| v * v).sum().sqrt();
    assert!(gram_defect <= 0.5 * lambda_min, "the saved Gram is XᵀWX");
    let bound = 2.0 * (explained * delta_form).sqrt()
        + delta_form
        + 2.0 * d.dot(&d) * gram_defect / (lambda_min * lambda_min)
        + roundoff;
    assert!(
        (d_prime - unpenalized_rss).abs() <= bound,
        "D′ from the saved fit = {d_prime} but the unpenalized WLS RSS = {unpenalized_rss} \
         (bound {bound:e})"
    );
    // The penalty is active, so the penalized RSS (the old reference's
    // numerator) is resolved from D′: the check above discriminates.
    assert!(
        penalized_rss - unpenalized_rss > 2.0 * bound,
        "the fixture's penalty must move the residual: RSS_pen = {penalized_rss}, \
         D′ = {unpenalized_rss}, bound {bound:e}"
    );

    // The saved model carries the residual through serde.
    let encoded = serde_json::to_string(&fit).expect("serialize the fit");
    let decoded: UnifiedFitResult = serde_json::from_str(&encoded).expect("parse the fit back");
    assert_eq!(
        decoded.inference.as_ref().and_then(|inference| inference.working_residual),
        Some(residual)
    );
}
