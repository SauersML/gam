//! Per-feature scalar-gate decoder with masked interchange-swap variant.
//!
//! This primitive is not specific to any one front-end. It is callable from
//! the `gam` Rust library directly, from the CLI (whenever a decoder
//! interchange-intervention probe is needed), and from PyTorch via the
//! `gam-pyffi` bindings. The intended use is *Distributed Alignment Search*
//! (DAS, Geiger et al. CLeaR 2024): given two inputs `a` and `b`, transplant
//! the latent atoms hypothesized to encode a causal variable from `a` into
//! `b`, decode with shared reconstruction weights and a shared per-feature
//! scalar gate, and back-propagate a swap-reconstruction error against a
//! target. The closed-form forward and analytic gradients live here so the
//! exact same arithmetic is used by every caller.
//!
//! Forward
//! -------
//! With latent `Z ∈ ℝ^{B×F}`, scalar gate `g ∈ ℝ^F`, decoder weights
//! `W ∈ ℝ^{D×F}`, and optional bias `b ∈ ℝ^D`,
//!
//!     X̂[i, d] = Σ_f g[f] · Z[i, f] · W[d, f] + b[d]
//!
//! Masked interchange-swap forward composes the latent first,
//!
//!     Z_eff[i, f] = mask[f] ? Z_a[i, f] : Z_b[i, f],
//!
//! then runs the plain decode on `Z_eff`. The gate `g` and the weights `W`
//! are SHARED between the two source decodings — only the latent activations
//! are interchanged. The scalar gate is decoupled from the reconstruction
//! matrix on purpose: that decoupling is what gives DAS a parameter to
//! transplant.
//!
//! Backward
//! --------
//! From upstream `Ȳ = ∂L/∂X̂ ∈ ℝ^{B×D}`,
//!
//!     ∂L/∂Z[i, f] = g[f] · Σ_d Ȳ[i, d] · W[d, f]
//!     ∂L/∂g[f]   = Σ_i Z[i, f] · Σ_d Ȳ[i, d] · W[d, f]
//!     ∂L/∂W[d, f] = g[f] · Σ_i Ȳ[i, d] · Z[i, f]
//!     ∂L/∂b[d]   = Σ_i Ȳ[i, d]
//!
//! For the masked-swap path, `∂L/∂Z_a` keeps the columns where `mask[f]`
//! is true (the rest are zero) and `∂L/∂Z_b` keeps the columns where
//! `mask[f]` is false. All other adjoints (`∂L/∂g`, `∂L/∂W`, `∂L/∂b`)
//! are computed from the composed `Z_eff` exactly as in the plain case.

use ndarray::{Array1, Array2, ArrayView1, ArrayView2};

/// Inputs to the plain (non-swap) gated decode forward.
#[derive(Debug, Clone, Copy)]
pub struct InterchangeDecodeForward<'a> {
    pub z: ArrayView2<'a, f64>,
    pub weights: ArrayView2<'a, f64>,
    pub gate: ArrayView1<'a, f64>,
    pub bias: Option<ArrayView1<'a, f64>>,
}

/// Inputs to the masked-swap forward.
#[derive(Debug, Clone, Copy)]
pub struct InterchangeSwapForward<'a> {
    pub z_a: ArrayView2<'a, f64>,
    pub z_b: ArrayView2<'a, f64>,
    pub mask: ArrayView1<'a, bool>,
    pub weights: ArrayView2<'a, f64>,
    pub gate: ArrayView1<'a, f64>,
    pub bias: Option<ArrayView1<'a, f64>>,
}

/// Adjoints returned by the plain backward.
#[derive(Debug, Clone)]
pub struct InterchangeDecodeBackward {
    pub grad_z: Array2<f64>,
    pub grad_weights: Array2<f64>,
    pub grad_gate: Array1<f64>,
    pub grad_bias: Option<Array1<f64>>,
}

/// Adjoints returned by the masked-swap backward.
#[derive(Debug, Clone)]
pub struct InterchangeSwapBackward {
    pub grad_z_a: Array2<f64>,
    pub grad_z_b: Array2<f64>,
    pub grad_weights: Array2<f64>,
    pub grad_gate: Array1<f64>,
    pub grad_bias: Option<Array1<f64>>,
}

fn check_shapes_forward(
    z_rows: usize,
    z_cols: usize,
    weights: ArrayView2<'_, f64>,
    gate: ArrayView1<'_, f64>,
    bias: Option<ArrayView1<'_, f64>>,
) -> Result<(), String> {
    let (d, f_weights) = weights.dim();
    if f_weights != z_cols {
        return Err(format!(
            "interchange_decode: weights has F={f_weights}, expected {z_cols}"
        ));
    }
    if gate.len() != z_cols {
        return Err(format!(
            "interchange_decode: gate has length {}, expected {z_cols}",
            gate.len()
        ));
    }
    if let Some(b) = bias
        && b.len() != d
    {
        return Err(format!(
            "interchange_decode: bias has length {}, expected D={d}",
            b.len()
        ));
    }
    if z_rows == 0 || z_cols == 0 {
        return Err("interchange_decode: latent must be non-empty".to_string());
    }
    if !weights.iter().all(|v| v.is_finite()) {
        return Err("interchange_decode: weights must be finite".to_string());
    }
    if !gate.iter().all(|v| v.is_finite()) {
        return Err("interchange_decode: gate must be finite".to_string());
    }
    if let Some(b) = bias
        && !b.iter().all(|v| v.is_finite())
    {
        return Err("interchange_decode: bias must be finite".to_string());
    }
    Ok(())
}

/// Plain gated decode: `X̂[i, d] = Σ_f g[f] · Z[i, f] · W[d, f] + b[d]`.
pub fn interchange_decode_forward(
    inputs: InterchangeDecodeForward<'_>,
) -> Result<Array2<f64>, String> {
    let (b_rows, f) = inputs.z.dim();
    check_shapes_forward(b_rows, f, inputs.weights, inputs.gate, inputs.bias)?;
    if !inputs.z.iter().all(|v| v.is_finite()) {
        return Err("interchange_decode: latent must be finite".to_string());
    }

    let d = inputs.weights.nrows();
    let mut z_gated = Array2::<f64>::zeros((b_rows, f));
    for i in 0..b_rows {
        for j in 0..f {
            z_gated[[i, j]] = inputs.z[[i, j]] * inputs.gate[j];
        }
    }
    // out = z_gated · Wᵀ
    let mut out = z_gated.dot(&inputs.weights.t());
    if let Some(bias) = inputs.bias {
        for i in 0..b_rows {
            for k in 0..d {
                out[[i, k]] += bias[k];
            }
        }
    }
    Ok(out)
}

/// Masked-swap forward.
pub fn interchange_swap_forward(inputs: InterchangeSwapForward<'_>) -> Result<Array2<f64>, String> {
    if inputs.z_a.dim() != inputs.z_b.dim() {
        return Err(format!(
            "interchange_swap: z_a {:?} and z_b {:?} must have the same shape",
            inputs.z_a.dim(),
            inputs.z_b.dim()
        ));
    }
    let (b_rows, f) = inputs.z_a.dim();
    if inputs.mask.len() != f {
        return Err(format!(
            "interchange_swap: mask length {} must equal F={f}",
            inputs.mask.len()
        ));
    }
    if !inputs.z_a.iter().all(|v| v.is_finite()) || !inputs.z_b.iter().all(|v| v.is_finite()) {
        return Err("interchange_swap: latents must be finite".to_string());
    }
    let mut z_eff = Array2::<f64>::zeros((b_rows, f));
    for j in 0..f {
        let take_a = inputs.mask[j];
        if take_a {
            for i in 0..b_rows {
                z_eff[[i, j]] = inputs.z_a[[i, j]];
            }
        } else {
            for i in 0..b_rows {
                z_eff[[i, j]] = inputs.z_b[[i, j]];
            }
        }
    }
    interchange_decode_forward(InterchangeDecodeForward {
        z: z_eff.view(),
        weights: inputs.weights,
        gate: inputs.gate,
        bias: inputs.bias,
    })
}

/// Backward for the plain decode. `grad_out` is `∂L/∂X̂`.
pub fn interchange_decode_backward(
    z: ArrayView2<'_, f64>,
    weights: ArrayView2<'_, f64>,
    gate: ArrayView1<'_, f64>,
    grad_out: ArrayView2<'_, f64>,
    with_bias: bool,
) -> Result<InterchangeDecodeBackward, String> {
    let (b_rows, f) = z.dim();
    let (d, f_w) = weights.dim();
    if f_w != f {
        return Err(format!(
            "interchange_decode_backward: weights has F={f_w}, expected {f}"
        ));
    }
    if gate.len() != f {
        return Err(format!(
            "interchange_decode_backward: gate length {} != F={f}",
            gate.len()
        ));
    }
    if grad_out.dim() != (b_rows, d) {
        return Err(format!(
            "interchange_decode_backward: grad_out shape {:?} != ({b_rows}, {d})",
            grad_out.dim()
        ));
    }

    // Working term: G[i, f] = Σ_d grad_out[i, d] · W[d, f]   ( = grad_out · W )
    let g_mat = grad_out.dot(&weights); // (B, F)

    // ∂L/∂Z[i, f] = g[f] · G[i, f]
    let mut grad_z = Array2::<f64>::zeros((b_rows, f));
    for i in 0..b_rows {
        for j in 0..f {
            grad_z[[i, j]] = gate[j] * g_mat[[i, j]];
        }
    }

    // ∂L/∂g[f] = Σ_i Z[i, f] · G[i, f]
    let mut grad_gate = Array1::<f64>::zeros(f);
    for j in 0..f {
        let mut acc = 0.0;
        for i in 0..b_rows {
            acc += z[[i, j]] * g_mat[[i, j]];
        }
        grad_gate[j] = acc;
    }

    // ∂L/∂W[d, f] = g[f] · Σ_i grad_out[i, d] · Z[i, f]
    //             = g[f] · (grad_outᵀ · Z)[d, f]
    let mut grad_weights = grad_out.t().dot(&z); // (D, F)
    for j in 0..f {
        let scale = gate[j];
        for k in 0..d {
            grad_weights[[k, j]] *= scale;
        }
    }

    let grad_bias = if with_bias {
        let mut gb = Array1::<f64>::zeros(d);
        for i in 0..b_rows {
            for k in 0..d {
                gb[k] += grad_out[[i, k]];
            }
        }
        Some(gb)
    } else {
        None
    };

    Ok(InterchangeDecodeBackward {
        grad_z,
        grad_weights,
        grad_gate,
        grad_bias,
    })
}

/// Backward for the masked-swap variant.
pub fn interchange_swap_backward(
    z_a: ArrayView2<'_, f64>,
    z_b: ArrayView2<'_, f64>,
    mask: ArrayView1<'_, bool>,
    weights: ArrayView2<'_, f64>,
    gate: ArrayView1<'_, f64>,
    grad_out: ArrayView2<'_, f64>,
    with_bias: bool,
) -> Result<InterchangeSwapBackward, String> {
    if z_a.dim() != z_b.dim() {
        return Err(format!(
            "interchange_swap_backward: z_a {:?} and z_b {:?} must have the same shape",
            z_a.dim(),
            z_b.dim()
        ));
    }
    let (b_rows, f) = z_a.dim();
    if mask.len() != f {
        return Err(format!(
            "interchange_swap_backward: mask length {} != F={f}",
            mask.len()
        ));
    }

    // Build z_eff and reuse the plain backward.
    let mut z_eff = Array2::<f64>::zeros((b_rows, f));
    for j in 0..f {
        let take_a = mask[j];
        if take_a {
            for i in 0..b_rows {
                z_eff[[i, j]] = z_a[[i, j]];
            }
        } else {
            for i in 0..b_rows {
                z_eff[[i, j]] = z_b[[i, j]];
            }
        }
    }
    let inner = interchange_decode_backward(z_eff.view(), weights, gate, grad_out, with_bias)?;

    // Distribute ∂L/∂Z_eff to ∂L/∂Z_a / ∂L/∂Z_b along the mask.
    let mut grad_z_a = Array2::<f64>::zeros((b_rows, f));
    let mut grad_z_b = Array2::<f64>::zeros((b_rows, f));
    for j in 0..f {
        let take_a = mask[j];
        if take_a {
            for i in 0..b_rows {
                grad_z_a[[i, j]] = inner.grad_z[[i, j]];
            }
        } else {
            for i in 0..b_rows {
                grad_z_b[[i, j]] = inner.grad_z[[i, j]];
            }
        }
    }

    Ok(InterchangeSwapBackward {
        grad_z_a,
        grad_z_b,
        grad_weights: inner.grad_weights,
        grad_gate: inner.grad_gate,
        grad_bias: inner.grad_bias,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{Array1, Array2, array};

    fn approx_eq(a: &Array2<f64>, b: &Array2<f64>, tol: f64) -> bool {
        if a.dim() != b.dim() {
            return false;
        }
        a.iter().zip(b.iter()).all(|(x, y)| (x - y).abs() < tol)
    }

    #[test]
    fn forward_matches_hand_recomputation() {
        let z = array![[1.0, -2.0, 0.5], [0.0, 3.0, -1.0]];
        let w = array![[0.1, 0.2, 0.3], [-0.4, 0.5, 0.6]];
        let g = array![1.0, 0.5, -1.0];
        let bias = array![0.01, -0.02];
        let out = interchange_decode_forward(InterchangeDecodeForward {
            z: z.view(),
            weights: w.view(),
            gate: g.view(),
            bias: Some(bias.view()),
        })
        .unwrap();
        // expected row i, col k: Σ_f g[f] z[i,f] w[k,f] + bias[k]
        let mut expected = Array2::<f64>::zeros((2, 2));
        for i in 0..2 {
            for k in 0..2 {
                let mut acc = bias[k];
                for j in 0..3 {
                    acc += g[j] * z[[i, j]] * w[[k, j]];
                }
                expected[[i, k]] = acc;
            }
        }
        assert!(approx_eq(&out, &expected, 1e-12));
    }

    #[test]
    fn swap_all_true_matches_z_a_forward() {
        let z_a = array![[1.0, -2.0], [3.0, 0.5]];
        let z_b = array![[10.0, 20.0], [-30.0, 40.0]];
        let w = array![[0.1, 0.2], [0.3, -0.4], [0.5, 0.6]];
        let g = array![0.7, -0.3];
        let mask = Array1::from(vec![true, true]);
        let swapped = interchange_swap_forward(InterchangeSwapForward {
            z_a: z_a.view(),
            z_b: z_b.view(),
            mask: mask.view(),
            weights: w.view(),
            gate: g.view(),
            bias: None,
        })
        .unwrap();
        let plain = interchange_decode_forward(InterchangeDecodeForward {
            z: z_a.view(),
            weights: w.view(),
            gate: g.view(),
            bias: None,
        })
        .unwrap();
        assert!(approx_eq(&swapped, &plain, 1e-12));
    }

    #[test]
    fn swap_all_false_matches_z_b_forward() {
        let z_a = array![[1.0, -2.0], [3.0, 0.5]];
        let z_b = array![[10.0, 20.0], [-30.0, 40.0]];
        let w = array![[0.1, 0.2], [0.3, -0.4]];
        let g = array![0.7, -0.3];
        let mask = Array1::from(vec![false, false]);
        let swapped = interchange_swap_forward(InterchangeSwapForward {
            z_a: z_a.view(),
            z_b: z_b.view(),
            mask: mask.view(),
            weights: w.view(),
            gate: g.view(),
            bias: None,
        })
        .unwrap();
        let plain = interchange_decode_forward(InterchangeDecodeForward {
            z: z_b.view(),
            weights: w.view(),
            gate: g.view(),
            bias: None,
        })
        .unwrap();
        assert!(approx_eq(&swapped, &plain, 1e-12));
    }

    #[test]
    fn backward_matches_finite_differences() {
        let z = array![[0.4, -0.7, 1.1], [0.2, 0.8, -0.3]];
        let w = array![[0.1, 0.2, 0.3], [-0.4, 0.5, 0.6]];
        let g = array![0.6, -0.2, 1.3];
        let bias = array![0.05, -0.01];
        let grad_out = array![[1.0, -0.5], [0.3, 0.8]];

        let an = interchange_decode_backward(z.view(), w.view(), g.view(), grad_out.view(), true)
            .unwrap();

        // L = sum(grad_out * forward(z, w, g, bias))
        // ∂L/∂z[i,j] via finite differences
        let eps = 1e-6;
        for i in 0..z.nrows() {
            for j in 0..z.ncols() {
                let mut zp = z.clone();
                let mut zm = z.clone();
                zp[[i, j]] += eps;
                zm[[i, j]] -= eps;
                let fp = interchange_decode_forward(InterchangeDecodeForward {
                    z: zp.view(),
                    weights: w.view(),
                    gate: g.view(),
                    bias: Some(bias.view()),
                })
                .unwrap();
                let fm = interchange_decode_forward(InterchangeDecodeForward {
                    z: zm.view(),
                    weights: w.view(),
                    gate: g.view(),
                    bias: Some(bias.view()),
                })
                .unwrap();
                let lp: f64 = fp.iter().zip(grad_out.iter()).map(|(a, b)| a * b).sum();
                let lm: f64 = fm.iter().zip(grad_out.iter()).map(|(a, b)| a * b).sum();
                let fd = (lp - lm) / (2.0 * eps);
                assert!(
                    (an.grad_z[[i, j]] - fd).abs() < 1e-7,
                    "grad_z mismatch at ({i},{j}): analytic {} vs fd {}",
                    an.grad_z[[i, j]],
                    fd
                );
            }
        }
        // ∂L/∂g[j]
        for j in 0..g.len() {
            let mut gp = g.clone();
            let mut gm = g.clone();
            gp[j] += eps;
            gm[j] -= eps;
            let fp = interchange_decode_forward(InterchangeDecodeForward {
                z: z.view(),
                weights: w.view(),
                gate: gp.view(),
                bias: Some(bias.view()),
            })
            .unwrap();
            let fm = interchange_decode_forward(InterchangeDecodeForward {
                z: z.view(),
                weights: w.view(),
                gate: gm.view(),
                bias: Some(bias.view()),
            })
            .unwrap();
            let lp: f64 = fp.iter().zip(grad_out.iter()).map(|(a, b)| a * b).sum();
            let lm: f64 = fm.iter().zip(grad_out.iter()).map(|(a, b)| a * b).sum();
            let fd = (lp - lm) / (2.0 * eps);
            assert!(
                (an.grad_gate[j] - fd).abs() < 1e-7,
                "grad_gate mismatch at {j}: analytic {} vs fd {}",
                an.grad_gate[j],
                fd
            );
        }
        // ∂L/∂W[d, j]
        for d in 0..w.nrows() {
            for j in 0..w.ncols() {
                let mut wp = w.clone();
                let mut wm = w.clone();
                wp[[d, j]] += eps;
                wm[[d, j]] -= eps;
                let fp = interchange_decode_forward(InterchangeDecodeForward {
                    z: z.view(),
                    weights: wp.view(),
                    gate: g.view(),
                    bias: Some(bias.view()),
                })
                .unwrap();
                let fm = interchange_decode_forward(InterchangeDecodeForward {
                    z: z.view(),
                    weights: wm.view(),
                    gate: g.view(),
                    bias: Some(bias.view()),
                })
                .unwrap();
                let lp: f64 = fp.iter().zip(grad_out.iter()).map(|(a, b)| a * b).sum();
                let lm: f64 = fm.iter().zip(grad_out.iter()).map(|(a, b)| a * b).sum();
                let fd = (lp - lm) / (2.0 * eps);
                assert!(
                    (an.grad_weights[[d, j]] - fd).abs() < 1e-7,
                    "grad_W mismatch at ({d},{j}): analytic {} vs fd {}",
                    an.grad_weights[[d, j]],
                    fd
                );
            }
        }
        // ∂L/∂bias[d]
        let bias_grad = an.grad_bias.as_ref().unwrap();
        for d in 0..bias.len() {
            let mut bp = bias.clone();
            let mut bm = bias.clone();
            bp[d] += eps;
            bm[d] -= eps;
            let fp = interchange_decode_forward(InterchangeDecodeForward {
                z: z.view(),
                weights: w.view(),
                gate: g.view(),
                bias: Some(bp.view()),
            })
            .unwrap();
            let fm = interchange_decode_forward(InterchangeDecodeForward {
                z: z.view(),
                weights: w.view(),
                gate: g.view(),
                bias: Some(bm.view()),
            })
            .unwrap();
            let lp: f64 = fp.iter().zip(grad_out.iter()).map(|(a, b)| a * b).sum();
            let lm: f64 = fm.iter().zip(grad_out.iter()).map(|(a, b)| a * b).sum();
            let fd = (lp - lm) / (2.0 * eps);
            assert!(
                (bias_grad[d] - fd).abs() < 1e-7,
                "grad_bias mismatch at {d}: analytic {} vs fd {}",
                bias_grad[d],
                fd
            );
        }
    }

    #[test]
    fn swap_backward_routes_grad_through_mask() {
        let z_a = array![[1.0, 2.0, 3.0]];
        let z_b = array![[-1.0, -2.0, -3.0]];
        let w = array![[0.5, 0.25, -0.1]];
        let g = array![1.0, 0.5, -1.0];
        let mask = Array1::from(vec![true, false, true]);
        let grad_out = array![[1.0]];
        let bk = interchange_swap_backward(
            z_a.view(),
            z_b.view(),
            mask.view(),
            w.view(),
            g.view(),
            grad_out.view(),
            false,
        )
        .unwrap();
        // For j in {0, 2} (mask true): grad_z_a[0, j] = g[j] * w[0, j]; grad_z_b[0, j] = 0
        // For j=1 (mask false): grad_z_b[0, 1] = g[1] * w[0, 1]; grad_z_a[0, 1] = 0
        assert!((bk.grad_z_a[[0, 0]] - 1.0 * 0.5).abs() < 1e-12);
        assert!((bk.grad_z_a[[0, 1]] - 0.0).abs() < 1e-12);
        assert!((bk.grad_z_a[[0, 2]] - (-1.0) * (-0.1)).abs() < 1e-12);
        assert!((bk.grad_z_b[[0, 0]] - 0.0).abs() < 1e-12);
        assert!((bk.grad_z_b[[0, 1]] - 0.5 * 0.25).abs() < 1e-12);
        assert!((bk.grad_z_b[[0, 2]] - 0.0).abs() < 1e-12);
    }
}
