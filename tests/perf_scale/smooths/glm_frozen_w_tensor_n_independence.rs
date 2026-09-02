//! #1033 mechanism (c): the GLM design-moving ψ-sweep is n-INDEPENDENT.
//!
//! The architectural invariant the issue enforces is: n-dependent work happens
//! ONCE per fit (the sufficient-statistic build); the κ/ψ outer loop manipulates
//! only k×k objects whose per-trial cost is O(D²k²) — independent of n. For the
//! non-Gaussian GLM lane the carrier of that invariant is
//! [`FrozenWeightGramTensor`] (`solver/glm_sufficient_lane.rs`): at the warm β it
//! freezes the working weight `W` and builds the weighted-design Chebyshev-in-ψ
//! tensor once, after which every per-trial accessor — the value Gram `XᵀWX(ψ)`,
//! the RHS `XᵀWz(ψ)`, the gradient pair `(∂G/∂ψ, ∂b/∂ψ)`, and the Fisher Hessian
//! block `(∂²G/∂ψ², ∂²b/∂ψ²)` — is served n-free in k-space.
//!
//! This is the algebraic companion to the wall-clock `perf_kappa_loop_n_scaling`
//! measurement: rather than time a fit (which is noisy and gated behind the
//! iso-κ convergence path), it pins the invariant *exactly*. Replicate the SAME
//! `b` distinct base rows `m` times to form `n = m·b`. The weighted-design Gram
//! and all its ψ-derivatives are additive over rows, so the n-row tensor's
//! accessors equal EXACTLY `m ×` the base-row tensor's accessors at every ψ —
//! hence, after dividing by the replication factor, they are BIT-IDENTICAL as n
//! scales at fixed k. Equivalently: the k×k object the outer trial loop touches
//! does not change shape or content-per-unit-data as n grows; the only thing n
//! buys is a constant scale absorbed by the one-time build. That is exactly the
//! "cost/grad/Hessian identical as n scales at fixed k" acceptance for this lane.

use gam::solver::glm_sufficient_lane::FrozenWeightGramTensor;
use ndarray::{Array1, Array2};

/// Matérn-shaped synthetic design `g(r·e^ψ)`, `g(s) = (1+s)e^{−s}`, plus a
/// ψ-free cubic column — the structural mix of the radial spatial designs the
/// frozen-W lane actually serves. `r` is a deterministic function of the
/// (base-row, column) index so replication is exact: base row `i` reused at
/// global row `i + t·b` gets the IDENTICAL `r`, hence the identical design row.
fn base_design(psi: f64, base_rows: usize, k: usize) -> Array2<f64> {
    let mut x = Array2::<f64>::zeros((base_rows, k));
    for i in 0..base_rows {
        for j in 0..k {
            let r =
                0.05 + (i as f64 + 1.0) * (j as f64 + 1.0) / (base_rows as f64 * k as f64) * 3.0;
            if j == k - 1 {
                x[[i, j]] = r * r * r;
            } else {
                let s = r * psi.exp();
                x[[i, j]] = (1.0 + s) * (-s).exp();
            }
        }
    }
    x
}

/// A non-trivial positive Fisher weight per base row (e.g. Bernoulli μ(1−μ)).
fn base_weights(base_rows: usize) -> Array1<f64> {
    Array1::from_shape_fn(base_rows, |i| {
        let p = 0.1 + 0.8 * ((i as f64 + 0.5) / base_rows as f64);
        p * (1.0 - p)
    })
}

fn base_z(base_rows: usize) -> Array1<f64> {
    Array1::from_shape_fn(base_rows, |i| ((i as f64 * 0.37).sin()) + 0.5)
}

/// Build a frozen-W tensor whose data is the `base` set replicated `reps` times
/// (so `n = reps · base_rows`). The design realizer tiles the base design rows;
/// the weights and working response tile in lockstep. Same `k`, same ψ window —
/// only `n` changes.
fn build_replicated(
    base_rows: usize,
    k: usize,
    reps: usize,
    psi_lo: f64,
    psi_hi: f64,
) -> FrozenWeightGramTensor {
    let bw = base_weights(base_rows);
    let bz = base_z(base_rows);
    let n = base_rows * reps;
    let mut w = Array1::<f64>::zeros(n);
    let mut z = Array1::<f64>::zeros(n);
    for t in 0..reps {
        for i in 0..base_rows {
            w[t * base_rows + i] = bw[i];
            z[t * base_rows + i] = bz[i];
        }
    }
    FrozenWeightGramTensor::build(
        move |psi| {
            let base = base_design(psi, base_rows, k);
            let mut tiled = Array2::<f64>::zeros((n, k));
            for t in 0..reps {
                for i in 0..base_rows {
                    for j in 0..k {
                        tiled[[t * base_rows + i, j]] = base[[i, j]];
                    }
                }
            }
            Ok(tiled)
        },
        w.view(),
        z.view(),
        psi_lo,
        psi_hi,
    )
    .expect("frozen-W tensor must certify on the analytic Matérn-shaped design")
}

fn rel_err_mat(a: &Array2<f64>, b: &Array2<f64>) -> f64 {
    let scale = b
        .iter()
        .fold(0.0_f64, |acc, &v| acc.max(v.abs()))
        .max(1e-300);
    a.iter()
        .zip(b.iter())
        .fold(0.0_f64, |acc, (&x, &y)| acc.max((x - y).abs()))
        / scale
}

fn rel_err_vec(a: &Array1<f64>, b: &Array1<f64>) -> f64 {
    let scale = b
        .iter()
        .fold(0.0_f64, |acc, &v| acc.max(v.abs()))
        .max(1e-300);
    a.iter()
        .zip(b.iter())
        .fold(0.0_f64, |acc, (&x, &y)| acc.max((x - y).abs()))
        / scale
}

