//! gam#3335: the conditional-variance and skewness Rao gates run on the
//! conditional-mean residual `û = z − m̂(C)` with the estimated mean stage
//! propagated into their meat.
//!
//! Two arms on the issue's design, `a ~ U(−√3, √3)` and a two-point skewed `ε`:
//!
//!   - power: `z = 0.5a + c·a² + (1 + 0.2a)ε` at `c = −0.6`. On `(z − z̄)²` the
//!     moving mean's odd cross term `c·a·(a² − 1)` cancels the `0.4a` of the
//!     moving variance, and the gate fired in 35 of 4000 replicates.
//!   - size: `z = 0.5a + c·a² + ε`. The variance and third moment do not move
//!     on the linear span (`E[û^k | a]` is even in `a`), so both gates' p-values
//!     are U(0, 1). On `(z − z̄)²` the same cross term fakes a moving variance;
//!     on `û` with the mean stage treated as known, `∂S/∂β ≠ 0` leaves the meat
//!     the variance of the wrong score.

use super::estimated_latent_law::conditional_law_evidence;
use super::{ConditionalLawEvidence, kolmogorov_upper_quantile};
use gam_test_support::calibration::{COVERAGE_FALSE_POSITIVE_RATE, CalibrationRng};
use ndarray::{Array1, Array2};

const N: usize = 2000;
const MEAN_SLOPE: f64 = 0.5;
const CURVATURE: f64 = -0.6;
const SCALE_SLOPE: f64 = 0.2;
/// `ε ∈ {2, −1/2}` with `P(ε = 2) = 1/5`: `E ε = 0`, `E ε² = 1`, `E ε³ = 3/2`.
const EPS_HIGH: f64 = 2.0;
const EPS_LOW: f64 = -0.5;
const EPS_HIGH_PROB: f64 = 0.2;

fn draw(rng: &mut CalibrationRng, scale_slope: f64) -> (Array1<f64>, Array1<f64>, Array2<f64>) {
    let half_width = 3.0_f64.sqrt();
    let mut a_block = Array2::<f64>::zeros((N, 1));
    let mut z = Array1::<f64>::zeros(N);
    let mut weights = Array1::<f64>::zeros(N);
    for i in 0..N {
        let a = half_width * (2.0 * rng.uniform_open01() - 1.0);
        let eps = if rng.uniform_open01() < EPS_HIGH_PROB {
            EPS_HIGH
        } else {
            EPS_LOW
        };
        a_block[[i, 0]] = a;
        z[i] = MEAN_SLOPE * a + CURVATURE * a * a + (1.0 + scale_slope * a) * eps;
        weights[i] = 0.5 + rng.uniform_open01();
    }
    (z, weights, a_block)
}

fn ks_distance(mut p: Vec<f64>) -> f64 {
    p.sort_by(f64::total_cmp);
    let m = p.len() as f64;
    p.iter()
        .enumerate()
        .map(|(i, &u)| (u - i as f64 / m).max((i + 1) as f64 / m - u))
        .fold(0.0, f64::max)
}

#[test]
fn variance_gate_sees_a_moving_variance_under_a_curved_mean_3335() {
    // Population noncentrality of the variance score: `E ψ = Cov(a, E[û² | a])
    // = 2·SCALE_SLOPE·E ε²·Var(a) = 0.4` per unit weight, against the
    // propagated contribution's `Var ψ ≈ 4.5` (the population moments of this
    // design, evaluated on 2·10⁶ rows), so `λ = n·(E ψ)²/Var ψ ≈ 70`. The
    // level-1e-3 χ²₁ gate misses with probability `Φ(3.29 − √70) ≈ 2e-7`: a
    // single miss in the replicates is a defect.
    let replicates = 200;
    let mut rng = CalibrationRng::new(0x3335_0001);
    let mut misses = 0;
    for _ in 0..replicates {
        let (z, weights, a_block) = draw(&mut rng, SCALE_SLOPE);
        let evidence = conditional_law_evidence(&z, &weights, Some(a_block.view()))
            .expect("conditional law evidence");
        if !ConditionalLawEvidence::fires(evidence.variance_p_value, evidence.alpha) {
            misses += 1;
        }
    }
    assert_eq!(
        misses, 0,
        "gam#3335: the variance gate missed a moving variance under a curved mean in {misses} \
         of {replicates} replicates"
    );
}

#[test]
fn residual_moment_gates_are_calibrated_under_a_curved_mean_3335() {
    let replicates = 1000;
    let mut rng = CalibrationRng::new(0x3335_0002);
    let mut variance_p = Vec::with_capacity(replicates);
    let mut skewness_p = Vec::with_capacity(replicates);
    for _ in 0..replicates {
        let (z, weights, a_block) = draw(&mut rng, 0.0);
        let evidence = conditional_law_evidence(&z, &weights, Some(a_block.view()))
            .expect("conditional law evidence");
        variance_p.push(evidence.variance_p_value.expect("variance test is testable"));
        skewness_p.push(evidence.skewness_p_value.expect("skewness test is testable"));
    }
    let m = replicates as f64;
    let root_m = m.sqrt();
    let ks_crit =
        kolmogorov_upper_quantile(COVERAGE_FALSE_POSITIVE_RATE) / (root_m + 0.12 + 0.11 / root_m);
    for (name, p) in [("variance", variance_p), ("skewness", skewness_p)] {
        let ks = ks_distance(p);
        assert!(
            ks <= ks_crit,
            "gam#3335: under a curved mean with a constant residual law the {name} gate's \
             p-values must be U(0, 1); KS distance {ks:.4} exceeds the \
             level-{COVERAGE_FALSE_POSITIVE_RATE} critical value {ks_crit:.4}"
        );
    }
}
