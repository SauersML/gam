//! gam#2612: the coefficients a multinomial fit PUBLISHES must be the
//! coefficients it FOUND, and a smooth term must produce a non-constant fit.
//!
//! `probe_2612_flat_half` measured a two-class smooth multinomial that reports
//! `edf_per_class = 4.09`, selects interior smoothing parameters, publishes a
//! full-rank posterior covariance and a ρ-uncertainty correction — and saves
//! eight coefficients that are all exactly `0.0`, so every prediction is the
//! uniform simplex. Nothing in the payload contradicts itself loudly enough for
//! any existing gate to notice: the EDF comes from `H⁻¹S_λ` and the covariance
//! from `H`, and neither reads `β`.
//!
//! Two invariants close that gap, and they are deliberately of different kinds.
//!
//!   * SELF-CONSISTENCY, exact and family-general: `deviance` is `−2·log L(β̂)`
//!     computed inside the fit at the fit's own mode. Recomputing it from the
//!     SAVED coefficients against the SAVED training frame must reproduce it.
//!     Any path that loses, zeroes, or re-frames coefficients between the
//!     solver and the payload breaks this for every `K`, whatever the cause.
//!   * SUFFICIENCY: a smooth fitted to a strongly non-constant truth must
//!     recover a non-constant function. A payload can be perfectly
//!     self-consistent and still be the intercept-only model.

use csv::StringRecord;
use gam_data::encode_recordswith_inferred_schema;
use gam_models::fit_orchestration::FitConfig;
use gam_models::multinomial::{
    MultinomialFitRequest, MultinomialSavedModel, fit_penalized_multinomial_formula,
};

const N: usize = 240;

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

fn sigmoid(x: f64) -> f64 {
    if x >= 0.0 {
        1.0 / (1.0 + (-x).exp())
    } else {
        let e = x.exp();
        e / (1.0 + e)
    }
}

/// A strongly non-constant two-class log-odds: `p` sweeps `[0.21, 0.82]` across
/// the covariate range, which no intercept-only model can follow.
fn two_class_truth(x: f64) -> f64 {
    0.1 + 1.4 * (std::f64::consts::TAU * (x + 0.15)).sin()
}

/// A three-class truth whose boundaries move with `x`.
fn three_class_eta(x: f64) -> [f64; 3] {
    [
        1.1 * (x * 3.0 - 1.5).sin(),
        -0.5 + 0.9 * (x * 3.0).cos(),
        0.0,
    ]
}

fn two_class_rows(seed: u64) -> Vec<StringRecord> {
    let mut rng = Lcg::new(seed);
    (0..N)
        .map(|i| {
            let x = i as f64 / (N - 1) as f64;
            let label = if rng.next_u01() < sigmoid(two_class_truth(x)) {
                "hi"
            } else {
                "lo"
            };
            StringRecord::from(vec![format!("{x:.8}"), label.to_string()])
        })
        .collect()
}

fn three_class_rows(seed: u64) -> Vec<StringRecord> {
    let mut rng = Lcg::new(seed);
    (0..N)
        .map(|i| {
            let x = i as f64 / (N - 1) as f64;
            let eta = three_class_eta(x);
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
            StringRecord::from(vec![format!("{x:.8}"), format!("c{drawn}")])
        })
        .collect()
}

fn fit(rows: Vec<StringRecord>, formula: &str) -> MultinomialSavedModel {
    let data = encode_recordswith_inferred_schema(vec!["x".to_string(), "y".to_string()], rows)
        .expect("encode dataset");
    let config = FitConfig::default();
    fit_penalized_multinomial_formula(&MultinomialFitRequest {
        init_lambda: 1.0,
        max_iter: 120,
        tol: 1e-8,
        ..MultinomialFitRequest::new(&data, formula, &config)
    })
    .unwrap_or_else(|error| panic!("multinomial fit `{formula}` failed: {error:?}"))
}

/// `−2·Σ_r w_r·log p_{r, y_r}` from the SAVED payload alone: saved
/// coefficients, saved training design, saved labels, saved weights.
fn deviance_from_payload(model: &MultinomialSavedModel) -> f64 {
    let design = model.training_design().expect("saved training design");
    let beta = model.coefficients_active().expect("saved coefficients");
    let m = model.n_active_classes;
    let mut total = 0.0_f64;
    for row in 0..design.nrows() {
        let design_row = design.row(row);
        let eta: Vec<f64> = (0..m)
            .map(|a| {
                design_row
                    .iter()
                    .zip(beta.column(a).iter())
                    .map(|(x, b)| x * b)
                    .sum::<f64>()
            })
            .collect();
        // `log Σ_k exp(η_k)` over the active logits plus the pinned reference 0.
        let shift = eta.iter().copied().fold(0.0_f64, f64::max);
        let mut partition = (-shift).exp();
        for &value in &eta {
            partition += (value - shift).exp();
        }
        let log_partition = shift + partition.ln();
        let label = model.training_class_index[row] as usize;
        let picked = if label < m { eta[label] } else { 0.0 };
        total += model.training_weights[row] * (picked - log_partition);
    }
    -2.0 * total
}

/// The largest spread of any class's plug-in probability `softmax(η̂)` across the
/// training rows — zero exactly when the fitted surface is constant in `x`. Rebuilt
/// from the payload's training design and active coefficients, with the reference
/// class's `η = 0` last, the same convention the deviance recompute above uses.
fn plugin_probability_spread(model: &MultinomialSavedModel) -> f64 {
    let design = model.training_design().expect("training design");
    let beta = model.coefficients_active().expect("coefficients");
    let m = model.n_active_classes;
    let mut lo = vec![f64::INFINITY; m + 1];
    let mut hi = vec![f64::NEG_INFINITY; m + 1];
    for row in 0..design.nrows() {
        let design_row = design.row(row);
        let eta: Vec<f64> = (0..m)
            .map(|a| {
                design_row
                    .iter()
                    .zip(beta.column(a).iter())
                    .map(|(x, b)| x * b)
                    .sum::<f64>()
            })
            .collect();
        let shift = eta.iter().copied().fold(0.0_f64, f64::max);
        let mut partition = (-shift).exp();
        for &value in &eta {
            partition += (value - shift).exp();
        }
        for class in 0..=m {
            let logit = if class < m { eta[class] } else { 0.0 };
            let probability = (logit - shift).exp() / partition;
            lo[class] = lo[class].min(probability);
            hi[class] = hi[class].max(probability);
        }
    }
    (0..=m).map(|class| hi[class] - lo[class]).fold(0.0_f64, f64::max)
}

/// The exact defect: `K = 2` with a smooth term. Also the class of defect —
/// every published multinomial payload must imply the likelihood the fit
/// reported.
#[test]
fn a_saved_multinomial_payload_implies_the_deviance_the_fit_reported_2612() {
    for (label, rows, formula) in [
        ("K=2 smooth", two_class_rows(7), "y ~ s(x, bs='tp', k=8)"),
        ("K=2 parametric", two_class_rows(7), "y ~ x"),
        ("K=3 smooth", three_class_rows(7), "y ~ s(x, bs='tp', k=8)"),
        ("K=3 parametric", three_class_rows(7), "y ~ x"),
    ] {
        let model = fit(rows, formula);
        let published = model.deviance;
        let implied = deviance_from_payload(&model);
        let scale = published.abs().max(implied.abs()).max(1.0);
        assert!(
            (published - implied).abs() <= 1e-6 * scale,
            "[{label}] the saved coefficients imply deviance {implied:.6} but the fit \
             published {published:.6} (relative {:.3e}); the payload is not the fit — \
             coefficients were lost, zeroed, or left in a different frame between the \
             solver and the saved model",
            (published - implied).abs() / scale,
        );
    }
}

/// A smooth fitted to a truth that sweeps `p ∈ [0.21, 0.82]` must not publish a
/// constant. Self-consistency alone cannot see this: the intercept-only model
/// at the origin is perfectly self-consistent.
#[test]
fn a_two_class_smooth_multinomial_recovers_a_non_constant_surface_2612() {
    let model = fit(two_class_rows(7), "y ~ s(x, bs='tp', k=8)");

    let beta_sup = model
        .coefficients_active()
        .expect("coefficients")
        .iter()
        .fold(0.0_f64, |acc, value| acc.max(value.abs()));
    assert!(
        beta_sup > 0.0,
        "the two-class smooth fit published an identically-zero coefficient vector \
         while reporting edf_per_class={:?} — a model with effective degrees of \
         freedom and no coefficients is two answers to one question",
        model.edf_per_class,
    );

    let spread = plugin_probability_spread(&model);
    assert!(
        spread > 0.30,
        "the fitted class-probability surface spans only {spread:.6} across a truth \
         that spans 0.61; a smooth term that recovers a constant is the intercept-only \
         model wearing eight coefficients"
    );

    // And the fit must beat the constant model on its own likelihood, which is
    // the assumption every downstream calibration bar makes.
    let uniform_deviance = 2.0 * (N as f64) * (2.0_f64).ln();
    assert!(
        model.deviance < uniform_deviance - 1.0,
        "the fit's deviance {:.6} is not better than the uniform-simplex model's \
         {uniform_deviance:.6}",
        model.deviance,
    );
}

/// The same sufficiency question with three classes, so a repair that fixes
/// `K = 2` by special-casing it cannot pass while breaking the general path.
#[test]
fn a_three_class_smooth_multinomial_recovers_a_non_constant_surface_2612() {
    let model = fit(three_class_rows(7), "y ~ s(x, bs='tp', k=8)");
    let spread = plugin_probability_spread(&model);
    assert!(
        spread > 0.20,
        "the three-class fitted surface spans only {spread:.6}"
    );
}
