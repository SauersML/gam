//! TEMPORARY diagnostic for #1054 (to be removed). Prints the actual
//! full-conformal envelopes at the small fold and the split coverage + scale
//! structure at the large fold, to localize the failures empirically.

use gam::estimate::{FitOptions, fit_gam};
use gam::matrix::DesignMatrix;
use gam::predict::{
    ConformalCalibrationFold, PredictInput, PredictUncertaintyOptions, StandardPredictor,
    predict_full_uncertainty_conformal,
};
use gam::smooth::BlockwisePenalty;
use gam::types::{InverseLink, LikelihoodSpec, ResponseFamily, StandardLink};
use ndarray::{Array1, Array2};
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand_distr::{Distribution, Normal};

fn poly_design(x: &Array1<f64>) -> Array2<f64> {
    let n = x.len();
    let mut d = Array2::<f64>::zeros((n, 4));
    for i in 0..n {
        let xi = x[i];
        d[[i, 0]] = 1.0;
        d[[i, 1]] = xi;
        d[[i, 2]] = xi * xi;
        d[[i, 3]] = xi * xi * xi;
    }
    d
}
fn true_mean(xi: f64) -> f64 {
    2.0 + 1.5 * xi - 0.8 * xi * xi + 0.3 * xi * xi * xi
}
fn draw(n: usize, sd: f64, rng: &mut StdRng) -> (Array1<f64>, Array1<f64>) {
    let unit = Normal::new(0.0, 1.0).unwrap();
    let mut x = Array1::<f64>::zeros(n);
    let mut y = Array1::<f64>::zeros(n);
    for i in 0..n {
        let xi = -2.0 + 4.0 * (i as f64 + 0.5) / (n as f64) + 0.05 * unit.sample(rng);
        x[i] = xi;
        y[i] = true_mean(xi) + sd * unit.sample(rng);
    }
    (x, y)
}
fn fit_options() -> FitOptions {
    FitOptions {
        latent_cloglog: None,
        mixture_link: None,
        optimize_mixture: false,
        sas_link: None,
        optimize_sas: false,
        compute_inference: true,
        skip_rho_posterior_inference: false,
        max_iter: 120,
        tol: 1e-10,
        nullspace_dims: vec![0],
        linear_constraints: None,
        firth_bias_reduction: false,
        adaptive_regularization: None,
        penalty_shrinkage_floor: None,
        rho_prior: Default::default(),
        kronecker_penalty_system: None,
        kronecker_factored: None,
    }
}
fn gspec() -> LikelihoodSpec {
    LikelihoodSpec::new(
        ResponseFamily::Gaussian,
        InverseLink::Standard(StandardLink::Identity),
    )
}
fn fit_cubic(x: &Array1<f64>, y: &Array1<f64>) -> gam::estimate::UnifiedFitResult {
    let design = poly_design(x);
    let weights = Array1::<f64>::ones(design.nrows());
    let offset = Array1::<f64>::zeros(design.nrows());
    let penalty = BlockwisePenalty::new(1..design.ncols(), Array2::<f64>::eye(design.ncols() - 1));
    fit_gam(design, y.view(), weights.view(), offset.view(), &[penalty], gspec(), &fit_options())
        .expect("fit")
}
fn pin(d: &Array2<f64>) -> PredictInput {
    PredictInput {
        design: DesignMatrix::from(d.clone()),
        offset: Array1::<f64>::zeros(d.nrows()),
        design_noise: None,
        offset_noise: None,
        auxiliary_scalar: None,
        auxiliary_matrix: None,
    }
}
fn predict_conf(
    fit: &gam::estimate::UnifiedFitResult,
    cal_d: &Array2<f64>,
    cal_y: &Array1<f64>,
    test_d: &Array2<f64>,
    level: f64,
) -> gam::predict::PredictUncertaintyResult {
    let predictor = StandardPredictor {
        beta: fit.blocks[0].beta.clone(),
        family: gspec(),
        link_kind: Some(InverseLink::Standard(StandardLink::Identity)),
        covariance: fit.covariance_conditional.clone(),
        link_wiggle: None,
    };
    let mut options = PredictUncertaintyOptions {
        confidence_level: 0.90,
        includeobservation_interval: false,
        apply_bias_correction: false,
        edgeworth_one_sided: false,
        boundary_correction: false,
        ..Default::default()
    };
    options.conformal_level = Some(level);
    let calibration = ConformalCalibrationFold { input: pin(cal_d), y: cal_y.view() };
    predict_full_uncertainty_conformal(&predictor, &pin(test_d), fit, &gspec(), &options, &calibration)
        .expect("conf")
}

#[test]
fn diag_small_fold_envelopes() {
    let mut rng = StdRng::seed_from_u64(424242);
    let (xt, yt) = draw(400, 0.5, &mut rng);
    let fit = fit_cubic(&xt, &yt);
    let (xc, yc) = draw(6, 0.5, &mut rng);
    let (xs, _) = draw(50, 0.5, &mut rng);
    let conf = predict_conf(&fit, &poly_design(&xc), &yc, &poly_design(&xs), 0.90);
    let n_fin = (0..xs.len())
        .filter(|&i| conf.mean_lower[i].is_finite() && conf.mean_upper[i].is_finite())
        .count();
    eprintln!("SMALL n_cal=6: n_finite={}/{}  example=[{}, {}]", n_fin, xs.len(), conf.mean_lower[0], conf.mean_upper[0]);
    assert!(xs.len() > 0);
}

#[test]
fn diag_large_fold_coverage_and_scale() {
    let mut rng = StdRng::seed_from_u64(13);
    let (xt, yt) = draw(600, 0.5, &mut rng);
    let fit = fit_cubic(&xt, &yt);
    let (xc, yc) = draw(300, 0.5, &mut rng);
    let (xs, ys) = draw(2000, 0.5, &mut rng);
    let conf = predict_conf(&fit, &poly_design(&xc), &yc, &poly_design(&xs), 0.90);
    let inside = (0..xs.len())
        .filter(|&i| ys[i] >= conf.mean_lower[i] && ys[i] <= conf.mean_upper[i])
        .count();
    let cov = inside as f64 / xs.len() as f64;
    // width stats
    let widths: Vec<f64> = (0..xs.len()).map(|i| conf.mean_upper[i] - conf.mean_lower[i]).collect();
    let wmin = widths.iter().cloned().fold(f64::INFINITY, f64::min);
    let wmax = widths.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let wmean = widths.iter().sum::<f64>() / widths.len() as f64;
    // SE_mean stats at test
    let se = &conf.mean_standard_error;
    let semin = se.iter().cloned().fold(f64::INFINITY, f64::min);
    let semax = se.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    eprintln!("LARGE n_cal=300: coverage={:.4}  width[min/mean/max]=[{:.3}/{:.3}/{:.3}]  SE_mean[min/max]=[{:.4}/{:.4}]  (noise sd=0.5)", cov, wmin, wmean, wmax, semin, semax);
    assert!(cov.is_finite());
}
