//! #2612: the posterior-predictive class probability must be the POSTERIOR
//! MEAN, and on a quasi-separated fit the Gaussian-integrated quantity is not.
//!
//! Both bars below are decided against a brute-force posterior computed in the
//! test itself — a dense two-dimensional quadrature of the exact penalized
//! log-posterior, whose window is validated by its own edge mass. No reference
//! implementation, no fitted model, no sampler: the coefficient space is two
//! dimensional on purpose, so "the exact posterior mean" is a quantity this file
//! can compute to quadrature accuracy and assert against.
//!
//! The geometry is the one the penguins arm is in, reduced to its smallest
//! honest form: a binary logit on data that separate except for a thin flipped
//! band, under a nearly-flat proper penalty `WALL = 2.173913043e-4`: the λ floor
//! the multinomial formula path carried at this class balance (8e-4
//! pseudo-observations `· I₁ · n_ref/n_c`) until 14e1ce6d8 deleted it. There
//! `W = diag(p) − ppᵀ → 0` on almost every row, so the
//! likelihood is flat toward more separation and steep away from it and the
//! posterior is strongly skewed — the regime in which integrating a SYMMETRIC
//! Gaussian puts half its mass where the likelihood has already excluded the
//! coefficient.

use gam_models::multinomial_predictive::MultinomialPredictiveModel;
use ndarray::{Array1, Array2};

/// A nearly-flat proper penalty: the λ floor the multinomial formula path carried
/// at this class balance until 14e1ce6d8 deleted it.
const WALL: f64 = 2.173913043e-4;

const N_TRAIN: usize = 200;

/// Rows whose covariate sits far enough from the flip band that the class is
/// decided, plus one inside it where it is not.
const TEST_X: [f64; 5] = [-0.80, -0.30, 0.02, 0.30, 0.80];

fn training_frame() -> (Array2<f64>, Vec<u32>, Array1<f64>) {
    let mut design = Array2::<f64>::zeros((N_TRAIN, 2));
    let mut labels = Vec::with_capacity(N_TRAIN);
    for row in 0..N_TRAIN {
        let x = -1.0 + 2.0 * (row as f64) / ((N_TRAIN - 1) as f64);
        design[[row, 0]] = 1.0;
        design[[row, 1]] = x;
        let separated = if x > 0.0 { 1u32 } else { 0u32 };
        // A thin overlap band, so the maximiser is finite (quasi-separation)
        // rather than at infinity (complete separation).
        let flipped = x.abs() < 0.04;
        labels.push(if flipped { 1 - separated } else { separated });
    }
    (design, labels, Array1::from_elem(N_TRAIN, 1.0))
}

fn log_posterior(design: &Array2<f64>, labels: &[u32], theta: &[f64]) -> f64 {
    let mut total = 0.0;
    for row in 0..design.nrows() {
        let eta = theta[0] * design[[row, 0]] + theta[1] * design[[row, 1]];
        // log p(class 1) = eta - log(1 + e^eta); log p(class 0) = -log(1 + e^eta)
        let log_partition = if eta > 0.0 {
            eta + (1.0 + (-eta).exp()).ln()
        } else {
            (1.0 + eta.exp()).ln()
        };
        let picked = if labels[row] == 0 { eta } else { 0.0 };
        total += picked - log_partition;
    }
    total - 0.5 * WALL * (theta[0] * theta[0] + theta[1] * theta[1])
}

/// Exact posterior mean of `p_class0(x)` at each `TEST_X`, by dense quadrature.
/// Returns the probabilities and the fraction of the quadrature mass sitting on
/// the window's edge, which is what says the window contained the posterior.
fn exact_posterior_means(
    design: &Array2<f64>,
    labels: &[u32],
    centre: [f64; 2],
    half_width: [f64; 2],
    nodes: usize,
) -> ([f64; TEST_X.len()], f64) {
    let mut log_values = vec![0.0_f64; nodes * nodes];
    let mut grid_a = vec![0.0_f64; nodes];
    let mut grid_b = vec![0.0_f64; nodes];
    for i in 0..nodes {
        let t = -1.0 + 2.0 * (i as f64) / ((nodes - 1) as f64);
        grid_a[i] = centre[0] + half_width[0] * t;
        grid_b[i] = centre[1] + half_width[1] * t;
    }
    let mut best = f64::NEG_INFINITY;
    for i in 0..nodes {
        for j in 0..nodes {
            let value = log_posterior(design, labels, &[grid_a[i], grid_b[j]]);
            log_values[i * nodes + j] = value;
            if value > best {
                best = value;
            }
        }
    }
    let mut total = 0.0;
    for value in log_values.iter_mut() {
        *value = (*value - best).exp();
        total += *value;
    }
    let mut edge = 0.0;
    for i in 0..nodes {
        for j in 0..nodes {
            if i == 0 || j == 0 || i == nodes - 1 || j == nodes - 1 {
                edge += log_values[i * nodes + j];
            }
        }
    }
    let mut means = [0.0_f64; TEST_X.len()];
    for (index, x) in TEST_X.iter().enumerate() {
        let mut acc = 0.0;
        for i in 0..nodes {
            for j in 0..nodes {
                let eta = grid_a[i] + grid_b[j] * x;
                // The frame's class 0 is the ACTIVE logit, so p_0 = sigma(eta).
                let p0 = 1.0 / (1.0 + (-eta).exp());
                acc += log_values[i * nodes + j] * p0;
            }
        }
        means[index] = acc / total;
    }
    (means, edge / total)
}

/// The estimand as the Gaussian route computes it: `E[softmax(η)]` with
/// `η ~ N(x'θ̂, x'Σx)`, by Gauss–Hermite in one dimension (exact for this
/// integrand to well past the accuracy under test).
fn gaussian_posterior_mean(eta_hat: f64, sd: f64) -> f64 {
    // The integrand is smooth and bounded in [0, 1] and the weight is Gaussian,
    // so a fine trapezoid over ±12 standard deviations resolves it far past the
    // 1e-3 scale the assertions live at — no quadrature table needed, and no
    // dependence on the shipped integrator, which is the thing under test.
    let steps = 20001usize;
    let span = 12.0;
    let mut acc = 0.0;
    let mut weight_total = 0.0;
    for i in 0..steps {
        let z = -span + 2.0 * span * (i as f64) / ((steps - 1) as f64);
        let w = (-0.5 * z * z).exp();
        let eta = eta_hat + sd * z;
        acc += w / (1.0 + (-eta).exp());
        weight_total += w;
    }
    acc / weight_total
}

#[test]
fn ratio_predictive_matches_the_exact_posterior_where_the_gaussian_does_not_2612() {
    let (design, labels, weights) = training_frame();
    let penalty = Array2::from_diag(&Array1::from_vec(vec![WALL, WALL]));
    let model = MultinomialPredictiveModel {
        training_design: design.view(),
        training_class_index: &labels,
        training_weights: weights.view(),
        joint_penalty: penalty.view(),
        n_classes: 2,
    };

    let mut x_new = Array2::<f64>::zeros((TEST_X.len(), 2));
    for (row, x) in TEST_X.iter().enumerate() {
        x_new[[row, 0]] = 1.0;
        x_new[[row, 1]] = *x;
    }

    // The mode and its Laplace width, computed here so the Gaussian arm is the
    // one the shipped route would build and the ratio arm is handed exactly the
    // coefficients a saved model would carry. `predictive_moments` refuses a
    // start that is not a stationary point of the objective it integrates —
    // that is the guard against publishing a probability from a different
    // posterior than the published mode belongs to — so a zero start is not a
    // legitimate call, and this test exercises the contract as callers meet it.
    let (mode, precision) = newton_mode(&design, &labels);
    let covariance = invert_two_by_two(&precision);
    let start = Array1::from_vec(vec![mode[0], mode[1]]);
    let moments = model
        .predictive_moments(start.view(), x_new.view(), true)
        .expect("ratio predictive moments");

    let (exact, edge_mass) = exact_posterior_means(
        &design,
        &labels,
        mode,
        [
            14.0 * covariance[[0, 0]].sqrt(),
            14.0 * covariance[[1, 1]].sqrt(),
        ],
        601,
    );
    assert!(
        edge_mass < 1e-6,
        "the quadrature window must contain the posterior; edge mass {edge_mass:e}"
    );

    let mut worst_ratio = 0.0_f64;
    let mut worst_gaussian = 0.0_f64;
    for (index, x) in TEST_X.iter().enumerate() {
        let eta_hat = mode[0] + mode[1] * x;
        let variance = covariance[[0, 0]]
            + 2.0 * x * covariance[[0, 1]]
            + x * x * covariance[[1, 1]];
        let gaussian = gaussian_posterior_mean(eta_hat, variance.max(0.0).sqrt());
        let ratio = moments.class_mean[[index, 0]];
        let ratio_error = (ratio - exact[index]).abs();
        let gaussian_error = (gaussian - exact[index]).abs();
        eprintln!(
            "[#2612] x={x:+.2} sd(eta)={:.4} exact={:.8} ratio={:.8} gaussian={:.8} \
             |ratio-exact|={ratio_error:.3e} |gauss-exact|={gaussian_error:.3e} \
             mass_defect={:.3e}",
            variance.max(0.0).sqrt(),
            exact[index],
            ratio,
            gaussian,
            moments.mass_defect[index],
        );
        worst_ratio = worst_ratio.max(ratio_error);
        worst_gaussian = worst_gaussian.max(gaussian_error);
    }

    // The bar the repair has to clear: the published quantity must be the
    // posterior mean to better than a thousandth of a probability.
    assert!(
        worst_ratio < 1.0e-3,
        "ratio predictive is {worst_ratio:e} from the exact posterior mean"
    );
    // And the bar that says the previous route was not merely less accurate but
    // wrong at a scale that changes the answer. Stated as a SEPARATION, not as a
    // bound on the Gaussian alone, so it cannot be satisfied by making the
    // quadrature worse.
    assert!(
        worst_gaussian > 20.0 * worst_ratio,
        "the Gaussian route's error ({worst_gaussian:e}) must be an order of magnitude above \
         the ratio's ({worst_ratio:e}) on this geometry; if it is not, this fixture has stopped \
         reproducing the defect and the bar above is measuring nothing"
    );

    // Second moments come from the same machinery and must respect the exact
    // identities they inherit.
    let second = moments
        .class_second_moment
        .as_ref()
        .expect("second moments requested");
    for index in 0..TEST_X.len() {
        let mut total = 0.0;
        for c in 0..2 {
            for d in 0..2 {
                total += second[[index, c, d]];
            }
        }
        assert!(
            (total - 1.0).abs() < 1e-9,
            "E[(Σ_c p_c)²] must be 1 after normalisation, got {total} at row {index}"
        );
        let mean = moments.class_mean[[index, 0]];
        let variance = second[[index, 0, 0]] - mean * mean;
        assert!(
            variance >= -1e-12,
            "row {index} has negative probability variance {variance:e}"
        );
    }
}

/// Newton on the same objective, used only to build the Gaussian arm.
fn newton_mode(design: &Array2<f64>, labels: &[u32]) -> ([f64; 2], Array2<f64>) {
    let mut theta = [0.0_f64; 2];
    let mut precision = Array2::<f64>::zeros((2, 2));
    for _ in 0..200 {
        let mut gradient = [0.0_f64; 2];
        precision = Array2::from_diag(&Array1::from_vec(vec![WALL, WALL]));
        for row in 0..design.nrows() {
            let x0 = design[[row, 0]];
            let x1 = design[[row, 1]];
            let eta = theta[0] * x0 + theta[1] * x1;
            let p = 1.0 / (1.0 + (-eta).exp());
            let target = if labels[row] == 0 { 1.0 } else { 0.0 };
            gradient[0] += (p - target) * x0;
            gradient[1] += (p - target) * x1;
            let w = p * (1.0 - p);
            precision[[0, 0]] += w * x0 * x0;
            precision[[0, 1]] += w * x0 * x1;
            precision[[1, 0]] += w * x1 * x0;
            precision[[1, 1]] += w * x1 * x1;
        }
        gradient[0] += WALL * theta[0];
        gradient[1] += WALL * theta[1];
        let covariance = invert_two_by_two(&precision);
        let step0 = -(covariance[[0, 0]] * gradient[0] + covariance[[0, 1]] * gradient[1]);
        let step1 = -(covariance[[1, 0]] * gradient[0] + covariance[[1, 1]] * gradient[1]);
        theta[0] += step0;
        theta[1] += step1;
        if step0.abs().max(step1.abs()) < 1e-13 {
            break;
        }
    }
    (theta, precision)
}

fn invert_two_by_two(matrix: &Array2<f64>) -> Array2<f64> {
    let determinant = matrix[[0, 0]] * matrix[[1, 1]] - matrix[[0, 1]] * matrix[[1, 0]];
    let mut out = Array2::<f64>::zeros((2, 2));
    out[[0, 0]] = matrix[[1, 1]] / determinant;
    out[[1, 1]] = matrix[[0, 0]] / determinant;
    out[[0, 1]] = -matrix[[0, 1]] / determinant;
    out[[1, 0]] = -matrix[[1, 0]] / determinant;
    out
}
