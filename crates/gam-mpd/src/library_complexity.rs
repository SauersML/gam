//! A library's per-token complexity under its posterior (#2951): the expected number of parts
//! active on a token, `E_q[k(x)]`, and its exact derivatives in the posterior's parameters.
//!
//! A part of an MLP is gated when its law is exactly zero below a threshold (a ReLU feature
//! `relu(g·x + c) u`, or a rank-one component read against its threshold): it executes on a token
//! only when its pre-activation `z = g·x + c` is positive. Under the factorized Gaussian posterior
//! `q` (each entry `N(μ, σ²)`), for a fixed input `x` the pre-activation is Gaussian with mean
//! `m = μ_g·x + μ_c` and variance `s² = Σ_j x_j² σ²_gj + σ²_c` (the local reparameterization), so
//! the part is on with probability `P = Φ(m / s)`. Its derivatives are exact:
//! `∂P/∂m = φ(m/s) / s` and `∂P/∂s² = −φ(m/s) m / (2 s³)`, so
//! `∂P/∂μ_gj = (φ/s) x_j`, `∂P/∂μ_c = φ/s`, `∂P/∂σ²_gj = −φ m/(2 s³) x_j²`, `∂P/∂σ²_c = −φ m/(2 s³)`.
//! With `s = 0` (every entry of the gate held exactly) the part is on iff `m > 0`, a step whose
//! derivatives are zero almost everywhere.
//!
//! Every other part (a law that is never exactly zero: GELU, SiLU and gated products of them, the
//! identity) executes on every token and counts one per token, as does every attention head with a
//! surviving value: machinery that runs on all tokens is counted whole.
//!
//! The input `x` is the layer's input on the step's own forward pass (at one weight sample around
//! the iterate): the parts upstream are integrated by that sample, the gate's own entries exactly.
//! The derivatives here are in the gate's own parameters with `x` held at that sample.

use ndarray::{Array1, Array2, ArrayView1, ArrayView2, Axis};
use statrs::function::erf::erfc;

/// One layer's gated parts on a set of rows: the inputs `x` (rows × d), the gate rows' means and
/// variances (parts × d), the thresholds' means and variances (one per part, or none), and which
/// parts survive.
pub struct Gate<'a> {
    pub x: ArrayView2<'a, f64>,
    pub mean: ArrayView2<'a, f64>,
    pub variance: ArrayView2<'a, f64>,
    pub bias: Option<(ArrayView1<'a, f64>, ArrayView1<'a, f64>)>,
    pub alive: &'a [bool],
}

/// `Σ_rows Σ_parts P` and its derivatives in the gate's means and variances (and the thresholds').
#[derive(Debug, Clone)]
pub struct Expected {
    pub count: f64,
    pub rows: usize,
    pub mean: Array2<f64>,
    pub variance: Array2<f64>,
    pub bias_mean: Array1<f64>,
    pub bias_variance: Array1<f64>,
}

/// `Φ(z)`, the standard normal distribution function.
#[must_use]
pub fn normal_cdf(z: f64) -> f64 {
    0.5 * erfc(-z / std::f64::consts::SQRT_2)
}

/// `φ(z)`, the standard normal density.
#[must_use]
pub fn normal_pdf(z: f64) -> f64 {
    (-0.5 * z * z).exp() / (2.0 * std::f64::consts::PI).sqrt()
}

/// The expected count of `gate`'s surviving parts on its rows, `Σ_t Σ_i Φ(m_ti / s_ti)`, and its
/// exact derivatives (module note).
pub fn expected(gate: &Gate<'_>) -> Result<Expected, String> {
    let (rows, d) = gate.x.dim();
    let parts = gate.mean.nrows();
    if gate.mean.ncols() != d || gate.variance.dim() != (parts, d) || gate.alive.len() != parts {
        return Err("library complexity: a gate of other shapes than its input".into());
    }
    if let Some((m, v)) = gate.bias
        && (m.len() != parts || v.len() != parts)
    {
        return Err("library complexity: thresholds of another count than the parts".into());
    }
    let mut m = gate.x.dot(&gate.mean.t());
    let squares = gate.x.mapv(|v| v * v);
    let mut s2 = squares.dot(&gate.variance.t());
    if let Some((bm, bv)) = gate.bias {
        m += &bm;
        s2 += &bv;
    }
    // Per row and part, ∂P/∂m and ∂P/∂s² (zero for a removed part).
    let mut slope = Array2::<f64>::zeros((rows, parts));
    let mut spread = Array2::<f64>::zeros((rows, parts));
    let mut count = 0.0;
    for t in 0..rows {
        for i in 0..parts {
            if !gate.alive[i] {
                continue;
            }
            let (mean, var) = (m[[t, i]], s2[[t, i]]);
            if var > 0.0 {
                let s = var.sqrt();
                let z = mean / s;
                let density = normal_pdf(z);
                count += normal_cdf(z);
                slope[[t, i]] = density / s;
                spread[[t, i]] = -density * mean / (2.0 * var * s);
            } else if mean > 0.0 {
                count += 1.0;
            }
        }
    }
    Ok(Expected {
        count,
        rows,
        mean: slope.t().dot(&gate.x),
        variance: spread.t().dot(&squares),
        bias_mean: slope.sum_axis(Axis(0)),
        bias_variance: spread.sum_axis(Axis(0)),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::{RngExt, SeedableRng, rngs::StdRng};

    fn random(rng: &mut StdRng, rows: usize, cols: usize, scale: f64, shift: f64) -> Array2<f64> {
        Array2::from_shape_fn((rows, cols), |_| shift + scale * (rng.random::<f64>() - 0.5))
    }

    /// The derivatives against central differences of the count in each mean and variance (f64,
    /// relative 1e-6), the count against a Monte Carlo of the gate's own entries, and a removed part
    /// counting nothing.
    #[test]
    fn the_expected_count_and_its_derivatives_are_exact() {
        let mut rng = StdRng::seed_from_u64(5);
        let (rows, d, parts) = (7, 5, 4);
        let x = random(&mut rng, rows, d, 2.0, 0.0);
        let mean = random(&mut rng, parts, d, 1.0, 0.0);
        let variance = random(&mut rng, parts, d, 0.2, 0.15);
        let bias_mean = random(&mut rng, parts, 1, 1.0, 0.0).column(0).to_owned();
        let bias_variance = random(&mut rng, parts, 1, 0.1, 0.1).column(0).to_owned();
        let alive = [true, true, false, true];
        let count = |mean: &Array2<f64>, variance: &Array2<f64>, bm: &Array1<f64>, bv: &Array1<f64>| {
            expected(&Gate { x: x.view(), mean: mean.view(), variance: variance.view(), bias: Some((bm.view(), bv.view())), alive: &alive }).unwrap()
        };
        let at = count(&mean, &variance, &bias_mean, &bias_variance);
        let h = 1e-6;
        let check = |analytic: f64, plus: f64, minus: f64| {
            let numeric = (plus - minus) / (2.0 * h);
            assert!((analytic - numeric).abs() <= 1e-6 * (1.0 + numeric.abs()), "derivative {analytic} against {numeric}");
        };
        for i in 0..parts {
            for j in 0..d {
                for (field, analytic) in [(0, at.mean[[i, j]]), (1, at.variance[[i, j]])] {
                    let (mut up, mut down) = if field == 0 { (mean.clone(), mean.clone()) } else { (variance.clone(), variance.clone()) };
                    up[[i, j]] += h;
                    down[[i, j]] -= h;
                    let (p, m) = if field == 0 {
                        (count(&up, &variance, &bias_mean, &bias_variance).count, count(&down, &variance, &bias_mean, &bias_variance).count)
                    } else {
                        (count(&mean, &up, &bias_mean, &bias_variance).count, count(&mean, &down, &bias_mean, &bias_variance).count)
                    };
                    check(analytic, p, m);
                }
            }
            let (mut up, mut down) = (bias_mean.clone(), bias_mean.clone());
            up[i] += h;
            down[i] -= h;
            check(at.bias_mean[i], count(&mean, &variance, &up, &bias_variance).count, count(&mean, &variance, &down, &bias_variance).count);
            let (mut up, mut down) = (bias_variance.clone(), bias_variance.clone());
            up[i] += h;
            down[i] -= h;
            check(at.bias_variance[i], count(&mean, &variance, &bias_mean, &up).count, count(&mean, &variance, &bias_mean, &down).count);
        }
        // A removed part has no derivatives.
        assert!(at.mean.row(2).iter().chain(at.variance.row(2)).all(|v| *v == 0.0));
        // Monte Carlo over the gate's own entries: per draw of every surviving part's gate row and
        // threshold, the count of positive pre-activations over the rows; within five standard
        // errors of the draws' mean.
        let normal = |rng: &mut StdRng| -> f64 {
            let (a, b): (f64, f64) = (rng.random(), rng.random());
            (-2.0 * (1.0 - a).ln()).sqrt() * (2.0 * std::f64::consts::PI * b).cos()
        };
        let draws = 40_000;
        let (mut sum, mut squares) = (0.0, 0.0);
        for _ in 0..draws {
            let mut on = 0.0;
            for i in (0..parts).filter(|i| alive[*i]) {
                let g: Vec<f64> = (0..d).map(|j| mean[[i, j]] + variance[[i, j]].sqrt() * normal(&mut rng)).collect();
                let c = bias_mean[i] + bias_variance[i].sqrt() * normal(&mut rng);
                on += x.rows().into_iter().filter(|row| row.iter().zip(&g).map(|(a, b)| a * b).sum::<f64>() + c > 0.0).count() as f64;
            }
            sum += on;
            squares += on * on;
        }
        let estimate = sum / draws as f64;
        let standard_error = ((squares / draws as f64 - estimate * estimate) / draws as f64).sqrt();
        assert!((estimate - at.count).abs() <= 5.0 * standard_error, "Monte Carlo {estimate} ± {standard_error} against {}", at.count);
    }

    /// With every variance zero the count is the number of positive pre-activations.
    #[test]
    fn a_gate_held_exactly_counts_its_positive_pre_activations() {
        let mut rng = StdRng::seed_from_u64(9);
        let x = random(&mut rng, 6, 3, 2.0, 0.0);
        let mean = random(&mut rng, 5, 3, 1.0, 0.0);
        let zeros = Array2::zeros((5, 3));
        let alive = [true; 5];
        let at = expected(&Gate { x: x.view(), mean: mean.view(), variance: zeros.view(), bias: None, alive: &alive }).unwrap();
        let positive = x.dot(&mean.t()).iter().filter(|v| **v > 0.0).count() as f64;
        assert_eq!(at.count, positive);
    }
}
