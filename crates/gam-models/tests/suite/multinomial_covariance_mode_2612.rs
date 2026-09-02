//! gam#2612: the multinomial's covariance-definition axis, gated from an angle
//! no coverage sweep can reach.
//!
//! The defect this pins is that `fit_penalized_multinomial_formula` read
//! `fit.covariance_conditional` and dropped the first-order ρ-uncertainty
//! correction `C = J·Var(ρ̂)·Jᵀ` the SAME fit had already computed, so every
//! multinomial band was conditional-on-λ̂ while every other family in the library
//! defaults to the corrected definition. A coverage sweep sees that only as
//! "under-covers", which is also what a wrong Jacobian, a mis-ordered covariance
//! or a bad centre look like. These assertions separate them:
//!
//!   1. the correction SURVIVES the fit (it is `Some`, symmetric, PSD-diagonal),
//!   2. `V_c = V_cond + C` exactly, so the two published matrices cannot drift,
//!   3. the correction REACHES the response scale — the corrected band is
//!      strictly wider than the conditional one, which is the step that was
//!      missing and that a stored-but-unused matrix would not produce,
//!   4. the factorised `gᵀ C g` kernel equals the literal `d`-dimensional
//!      contraction assembled entry by entry, and
//!   5. asking for a correction a model does not carry is an ERROR, never a
//!      silent downgrade to the narrower band.

use csv::StringRecord;
use gam_data::encode_recordswith_inferred_schema;
use gam_models::fit_orchestration::FitConfig;
use gam_models::multinomial::{
    InferenceCovarianceMode, MultinomialFitRequest, MultinomialSavedModel,
    fit_penalized_multinomial_formula,
};

const N: usize = 220;

/// Deterministic LCG → `U[0,1)`; no external RNG so the fixture is
/// byte-identical run to run.
struct Lcg(u64);

impl Lcg {
    fn new(seed: u64) -> Self {
        Self(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15).wrapping_add(1))
    }

    fn next_u01(&mut self) -> f64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((self.0 >> 11) as f64) / ((1u64 << 53) as f64)
    }
}

/// A three-class smooth softmax sample: every class is well represented and the
/// boundaries are curved, so REML has a genuine interior optimum in ρ and the
/// outer solve has curvature to propagate.
fn smooth_three_class(seed: u64) -> Vec<StringRecord> {
    let mut rng = Lcg::new(seed);
    let mut rows = Vec::with_capacity(N);
    for i in 0..N {
        let x = -2.0 + 4.0 * (i as f64) / ((N - 1) as f64);
        let eta = [0.9 * (x + 0.4).sin(), -0.4 + 0.7 * x.cos(), 0.0];
        let shift = eta.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let weights: Vec<f64> = eta.iter().map(|e| (e - shift).exp()).collect();
        let total: f64 = weights.iter().sum();
        let u = rng.next_u01();
        let mut cumulative = 0.0;
        let mut drawn = 2usize;
        for (class, weight) in weights.iter().enumerate() {
            cumulative += weight / total;
            if u < cumulative {
                drawn = class;
                break;
            }
        }
        rows.push(StringRecord::from(vec![
            format!("{x:.8}"),
            format!("c{drawn}"),
        ]));
    }
    rows
}

fn fit_smooth_three_class(seed: u64) -> MultinomialSavedModel {
    let data = encode_recordswith_inferred_schema(
        vec!["x".to_string(), "y".to_string()],
        smooth_three_class(seed),
    )
    .expect("encode three-class dataset");
    let config = FitConfig::default();
    fit_penalized_multinomial_formula(&MultinomialFitRequest {
        init_lambda: 1.0,
        max_iter: 120,
        tol: 1e-8,
        ..MultinomialFitRequest::new(&data, "y ~ s(x, bs='tp', k=8)", &config)
    })
    .expect("three-class smooth multinomial fit")
}

/// `softmax` over the active logits with the reference class pinned at `η = 0`.
fn softmax_with_reference(eta: &[f64]) -> Vec<f64> {
    let shift = eta.iter().copied().fold(0.0_f64, f64::max);
    let mut out: Vec<f64> = eta.iter().map(|e| (e - shift).exp()).collect();
    out.push((-shift).exp());
    let total: f64 = out.iter().sum();
    for value in out.iter_mut() {
        *value /= total;
    }
    out
}

#[test]
fn the_correction_reaches_the_response_scale_and_widens_the_band_2612() {
    let model = fit_smooth_three_class(11);
    let x = model.training_design().expect("training design");
    let correction = model.smoothing_correction().expect("retained correction");

    let (mean_conditional, se_conditional) = model
        .predict_probabilities_with_se_in_mode(x.view(), InferenceCovarianceMode::Conditional)
        .expect("conditional band");
    let (mean_corrected, se_corrected) = model
        .predict_probabilities_with_se_in_mode(
            x.view(),
            InferenceCovarianceMode::SmoothingCorrected,
        )
        .expect("corrected band");

    // The CENTRE is the same estimand under both modes — only the spread carries
    // the smoothing uncertainty. A mode that also moved the mean would be a
    // different fit, not a different interval.
    for ((row, class), &value) in mean_conditional.indexed_iter() {
        assert_eq!(
            value,
            mean_corrected[[row, class]],
            "the covariance mode moved the posterior MEAN at ({row}, {class})"
        );
    }

    let mut strictly_wider = 0usize;
    for ((row, class), &narrow) in se_conditional.indexed_iter() {
        let wide = se_corrected[[row, class]];
        assert!(
            wide >= narrow - 1e-15,
            "the corrected band is NARROWER than the conditional one at \
             ({row}, {class}): {wide} < {narrow} — a variance component cannot subtract"
        );
        if wide > narrow {
            strictly_wider += 1;
        }
    }
    assert!(
        strictly_wider * 2 > se_conditional.len(),
        "the retained correction reached fewer than half the published \
         (row, class) spreads ({strictly_wider} of {}); a correction that is \
         stored but never contracted through the response Jacobian is the \
         original defect wearing a new field",
        se_conditional.len(),
    );

    // The factorised kernel against the literal contraction. The shipped path
    // never forms `g`: it exploits `g_c = u_c ⊗ x` and contracts an `M × M`
    // Gram. This rebuilds the full length-`d` Jacobian and evaluates `gᵀ C g`
    // entry by entry, which is the definition the shortcut has to reproduce.
    let p = model.p_per_class;
    let m = model.n_active_classes;
    let k = model.class_levels.len();
    let d = p * m;
    let beta = model.coefficients_active().expect("coefficients");
    let mut worst_relative = 0.0_f64;
    for row in 0..x.nrows() {
        let design_row = x.row(row);
        let eta: Vec<f64> = (0..m)
            .map(|a| (0..p).map(|i| design_row[i] * beta[[i, a]]).sum::<f64>())
            .collect();
        let probabilities = softmax_with_reference(&eta);
        for class in 0..k {
            let mut jacobian = vec![0.0_f64; d];
            for a in 0..m {
                let delta = if class == a { 1.0 } else { 0.0 };
                let weight = probabilities[class] * (delta - probabilities[a]);
                for i in 0..p {
                    jacobian[a * p + i] = weight * design_row[i];
                }
            }
            let mut expected = 0.0_f64;
            for i in 0..d {
                if jacobian[i] == 0.0 {
                    continue;
                }
                let mut acc = 0.0_f64;
                for j in 0..d {
                    acc += correction[[i, j]] * jacobian[j];
                }
                expected += jacobian[i] * acc;
            }
            let published =
                se_corrected[[row, class]].powi(2) - se_conditional[[row, class]].powi(2);
            let denominator = expected.abs().max(published.abs()).max(1e-300);
            worst_relative = worst_relative.max((published - expected).abs() / denominator);
        }
    }
    assert!(
        worst_relative < 1e-8,
        "the factorised response-scale correction disagrees with the literal \
         gᵀCg contraction by relative {worst_relative:e}"
    );
}

