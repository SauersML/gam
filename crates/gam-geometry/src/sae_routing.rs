//! Rust-owned numerical helpers for the torch-only manifold-SAE routing lane.
//!
//! These kernels are intentionally small, deterministic primitives that the
//! Python torch module calls as FFI glue.  Keeping them here preserves the repo
//! doctrine that Python marshals tensors while Rust owns numeric rules.

use ndarray::{Array1, Array2, ArrayView1, ArrayView2};

/// Sinkhorn per-atom log-bias potentials that balance atom usage.
///
/// `log_scores[n, k]` are the (temperature-scaled) unnormalized
/// log-responsibilities of row `n` for atom `k`. This returns the per-atom
/// additive potential `b_k` such that `softmax_k(log_scores + b)` has an
/// approximately uniform column marginal (each atom claims `1/K` of the total
/// assignment mass). The caller forms the balanced log-responsibilities as
/// `log_scores + b`, keeping `log_scores` on the autograd tape while `b` enters
/// as a detached constant — exactly the `@no_grad` steering the torch lane used
/// to perform inline.
///
/// The potentials are the fixed point of the multiplicative Sinkhorn sweep
/// toward the uniform atom marginal `target = log(1/K)`, mean-centered each step
/// (the softmax is shift-invariant, so centering only fixes the gauge and keeps
/// the potentials bounded). A handful of sweeps converge it; `iters` matches the
/// torch lane's fixed count. Returns a zero vector for `K < 2` (nothing to
/// balance).
#[must_use]
pub fn sinkhorn_balance_bias(log_scores: ArrayView2<'_, f64>, iters: usize) -> Array1<f64> {
    let n_atoms = log_scores.ncols();
    let mut bias = Array1::<f64>::zeros(n_atoms);
    let n_rows = log_scores.nrows();
    if n_atoms < 2 || n_rows == 0 {
        return bias;
    }
    let target = (1.0 / n_atoms as f64).ln();
    let inv_rows = 1.0 / n_rows as f64;
    let mut usage = Array1::<f64>::zeros(n_atoms);
    let mut exps = vec![0.0_f64; n_atoms];
    for _ in 0..iters {
        usage.fill(0.0);
        for row in log_scores.rows() {
            // Numerically stable row softmax of `row + bias`.
            let mut max = f64::NEG_INFINITY;
            for k in 0..n_atoms {
                let v = row[k] + bias[k];
                if v > max {
                    max = v;
                }
            }
            let mut sum = 0.0;
            for k in 0..n_atoms {
                let e = (row[k] + bias[k] - max).exp();
                exps[k] = e;
                sum += e;
            }
            let inv_sum = 1.0 / sum;
            for k in 0..n_atoms {
                usage[k] += exps[k] * inv_sum;
            }
        }
        // `usage[k]` now holds Σ_rows softmax(row+bias)[k]; convert to the mean
        // assignment mass and take the multiplicative step toward uniform.
        for k in 0..n_atoms {
            let mean_usage = (usage[k] * inv_rows).max(1e-12);
            bias[k] += target - mean_usage.ln();
        }
        let mean_bias = bias.mean().unwrap_or(0.0);
        bias.mapv_inplace(|v| v - mean_bias);
    }
    bias
}

/// Lift a one-dimensional Duchon center vector into a deterministic `(K, d)`
/// low-discrepancy cloud in `[0, 1]^d`.
///
/// The first coordinate is the caller-provided 1-D center.  Remaining axes use
/// the generalized golden-ratio additive recurrence (`R_d`) keyed only to
/// `(K, d)`, matching the historical torch implementation bit-for-bit for f64
/// inputs before dtype conversion at the Python boundary.
#[must_use]
pub fn duchon_centers_nd(centers_1d: ArrayView1<'_, f64>, d: usize) -> Array2<f64> {
    let k = centers_1d.len();
    let width = d.max(1);
    let mut out = Array2::<f64>::zeros((k, width));
    for (row, center) in centers_1d.iter().enumerate() {
        out[(row, 0)] = *center;
    }
    if d <= 1 || k == 0 {
        return out;
    }

    // Historical R_d generalized golden-ratio fixed point: x^d = x + 1.
    // Thirty-two fixed-point refinements are retained deliberately for
    // bit-level continuity with the previous torch lane.  The count is not a
    // tuning knob: for the smallest routed multi-axis case (d=2) each step
    // roughly doubles correct digits near the fixed point, so 32 iterations is
    // beyond f64 mantissa resolution; larger d is contractive more quickly.
    let mut phi = 2.0_f64;
    for _ in 0..32 {
        phi = (1.0 + phi).powf(1.0 / d as f64);
    }
    for axis in 1..d {
        let alpha = (1.0 / phi).powi(axis as i32).rem_euclid(1.0);
        for row in 0..k {
            let idx = (row + 1) as f64;
            out[(row, axis)] = (idx * alpha + 0.5).rem_euclid(1.0);
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::{duchon_centers_nd, sinkhorn_balance_bias};
    use ndarray::{array, Array2};

    #[test]
    fn duchon_centers_nd_preserves_first_axis_and_shape() {
        let centers = array![0.0, 0.5, 1.0];
        let lifted = duchon_centers_nd(centers.view(), 3);
        assert_eq!(lifted.shape(), &[3, 3]);
        assert_eq!(lifted.column(0), centers);
    }

    #[test]
    fn sinkhorn_balance_bias_equalizes_atom_usage() {
        // Two atoms with a strong per-row preference for atom 0 (column 0 much
        // larger). Unbalanced usage would concentrate on atom 0; the Sinkhorn
        // potentials must push the mean column mass toward 1/K = 0.5.
        let mut scores = Array2::<f64>::zeros((64, 2));
        for i in 0..64 {
            scores[(i, 0)] = 2.0;
            scores[(i, 1)] = -2.0;
        }
        let bias = sinkhorn_balance_bias(scores.view(), 12);
        // Recompute the balanced column marginal.
        let mut usage = [0.0_f64; 2];
        for i in 0..64 {
            let a = scores[(i, 0)] + bias[0];
            let b = scores[(i, 1)] + bias[1];
            let m = a.max(b);
            let (ea, eb) = ((a - m).exp(), (b - m).exp());
            let s = ea + eb;
            usage[0] += ea / s;
            usage[1] += eb / s;
        }
        usage[0] /= 64.0;
        usage[1] /= 64.0;
        assert!(
            (usage[0] - 0.5).abs() < 0.05 && (usage[1] - 0.5).abs() < 0.05,
            "sinkhorn usage not balanced: {usage:?}"
        );
        // Gauge fixed: potentials are mean-centered.
        assert!(bias.mean().unwrap().abs() < 1e-9);
    }

    #[test]
    fn sinkhorn_balance_bias_trivial_for_single_atom() {
        let scores = Array2::<f64>::ones((8, 1));
        let bias = sinkhorn_balance_bias(scores.view(), 12);
        assert_eq!(bias.len(), 1);
        assert_eq!(bias[0], 0.0);
    }

    #[test]
    fn duchon_centers_nd_handles_empty_and_one_dimensional_inputs() {
        let empty = array![];
        assert_eq!(duchon_centers_nd(empty.view(), 4).shape(), &[0, 4]);
        let centers = array![0.25, 0.75];
        assert_eq!(
            duchon_centers_nd(centers.view(), 1),
            centers.into_shape_clone((2, 1)).unwrap()
        );
    }
}
