//! Spectral decode of the learned graph atom: eigengap basis, Nyström
//! out-of-sample coordinate + analytic jet, penalty/Dirichlet-form identity, and
//! the reconstruction advantage of a `q`-dimensional spectral decode over a
//! single typed circle atom on a shape outside the typed zoo.

use ndarray::Array2;

/// Explained variance of a multi-output reconstruction `Ŷ` against `Y`.
fn explained_variance(y: &Array2<f64>, fitted: &Array2<f64>) -> f64 {
    let n = y.nrows();
    let p = y.ncols();
    let mut mean = vec![0.0; p];
    for row in 0..n {
        for c in 0..p {
            mean[c] += y[[row, c]];
        }
    }
    for c in 0..p {
        mean[c] /= n as f64;
    }
    let mut sse = 0.0;
    let mut sst = 0.0;
    for row in 0..n {
        for c in 0..p {
            sse += (y[[row, c]] - fitted[[row, c]]).powi(2);
            sst += (y[[row, c]] - mean[c]).powi(2);
        }
    }
    1.0 - sse / sst
}

// (d) — eigengap q-selection recovers q = 2 for the circle, and the embedding
// dimension is deliberately NOT tied to the first Betti number.

// (1) — the decode penalty and the graph Dirichlet form are the SAME object:
// diag(λ) == Φᵀ L_W Φ, where L_W is the atom's own surviving Laplacian.

// (a) — planted noisy circle: Nyström spectral coordinates achieve
// orientation-quotiented circular correlation > 0.95 against the true angle.

// (c) — the analytic Nyström jet matches a central-difference check to 1e-5
// relative (finite differences are legal in test code, banned in src).

// (b) — planted figure-eight (outside the typed zoo): the q-dimensional spectral
// decode reconstructs the shape with EV clearly beating the best single typed
// circle atom fit, both scored through the same public closed-form REML fitter.

// The Nyström extension is a first-class basis evaluator: the trait `evaluate`
// path agrees with the batched coordinate call.
