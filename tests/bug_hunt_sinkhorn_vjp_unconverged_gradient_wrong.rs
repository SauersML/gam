//! Bug hunt: `sinkhorn_barycenter_vjp` returns a gradient that does not match
//! the derivative of `sinkhorn_barycenter` at the SAME arguments when the
//! Sinkhorn iteration has not converged — which is the case at the library's
//! own public default `n_iter = 20` for any small entropic regularization.
//!
//! A vector-Jacobian product is, by definition, the derivative of the forward
//! map it accompanies. The Python wrappers register this exact function as the
//! backward pass of a `torch.autograd.Function` (`gamfit/kernels_torch.py`) and
//! a `jax.custom_vjp` (`gamfit/kernels_jax.py`), both of which default to
//! `eps = 0.01, n_iter = 20`. Whatever `sinkhorn_barycenter(atoms, weights,
//! cost, eps, n_iter)` actually computes, `sinkhorn_barycenter_vjp(...)` with
//! the same `(eps, n_iter)` must return its gradient.
//!
//! It does not. The adjoint iteration in `src/kernels/sinkhorn_barycenter.rs`
//! (the `for _ in 0..n_iter` loop around lines 571-632) propagates cotangents
//! through the transport couplings `P_k`, `Q_k` (lines 499-516) as if the
//! forward dual potentials sat at the converged fixed point. When the forward
//! has not converged (small `eps` needs far more than 20 iterations), those
//! couplings are far from satisfying the marginal constraints, the adjoint
//! recursion is no longer contractive, and the accumulated `g_weights` / atom
//! gradients blow up — wrong sign, many orders of magnitude too large.
//!
//! This test pins `n_iter = 20` (the documented default) and asserts the VJP
//! equals the central finite-difference gradient of the same forward map. It
//! fails today and will pass once the VJP returns the true gradient of what
//! the forward actually computes (e.g. by iterating the adjoint to convergence
//! together with the forward, or by differentiating the truncated iteration).

use gam::kernels::sinkhorn_barycenter::{
    circular_cost, sinkhorn_barycenter, sinkhorn_barycenter_vjp,
};
use ndarray::{Array1, Array2};

/// Deterministic, well-conditioned test problem: three Gaussian-like bumps on a
/// length-6 cyclic support with the canonical squared circular ground cost.
fn problem() -> (Array2<f64>, Array1<f64>, Array2<f64>, Array1<f64>) {
    let k = 3usize;
    let m = 6usize;
    let mut atoms = Array2::<f64>::zeros((k, m));
    for ki in 0..k {
        let centre = (ki as f64 + 1.0) * m as f64 / (k as f64 + 1.0);
        let mut total = 0.0;
        for j in 0..m {
            let v = (-((j as f64 - centre).powi(2)) / 2.0).exp();
            atoms[[ki, j]] = v;
            total += v;
        }
        for j in 0..m {
            atoms[[ki, j]] /= total;
        }
    }
    let weights = Array1::from(vec![0.5, 0.3, 0.2]);
    let cost = circular_cost(m);
    // Cotangent: a fixed linear functional of the barycenter (its centred mean).
    let cotangent = Array1::from_shape_fn(m, |j| j as f64 - (m as f64 - 1.0) / 2.0);
    (atoms, weights, cost, cotangent)
}

fn loss(
    atoms: &Array2<f64>,
    weights: &Array1<f64>,
    cost: &Array2<f64>,
    cotangent: &Array1<f64>,
    eps: f64,
    n_iter: usize,
) -> f64 {
    let bary = sinkhorn_barycenter(atoms.view(), weights.view(), cost.view(), eps, n_iter)
        .expect("forward barycenter must succeed");
    cotangent.dot(&bary)
}

#[test]
fn sinkhorn_vjp_matches_forward_gradient_at_default_iterations() {
    // The library's own public default. `eps = 0.05` is a standard entropic-OT
    // regularization; `n_iter = 20` is the documented default of both Python
    // adapters. (At the literal `eps = 0.01` default the disagreement is ~1e16.)
    let eps = 0.05;
    let n_iter = 20usize;

    let (atoms, weights, cost, cotangent) = problem();
    let (k, m) = atoms.dim();

    let vjp = sinkhorn_barycenter_vjp(
        atoms.view(),
        weights.view(),
        cost.view(),
        eps,
        n_iter,
        cotangent.view(),
    )
    .expect("vjp must succeed");

    // Central finite-difference gradient of the SAME forward map.
    let h = 1.0e-6;

    // Weights gradient.
    let mut fd_weights = Array1::<f64>::zeros(k);
    for ki in 0..k {
        let mut wp = weights.clone();
        let mut wm = weights.clone();
        wp[ki] += h;
        wm[ki] -= h;
        let lp = loss(&atoms, &wp, &cost, &cotangent, eps, n_iter);
        let lm = loss(&atoms, &wm, &cost, &cotangent, eps, n_iter);
        fd_weights[ki] = (lp - lm) / (2.0 * h);
    }

    // Atoms gradient.
    let mut fd_atoms = Array2::<f64>::zeros((k, m));
    for ki in 0..k {
        for j in 0..m {
            let mut ap = atoms.clone();
            let mut am = atoms.clone();
            ap[[ki, j]] += h;
            am[[ki, j]] -= h;
            let lp = loss(&ap, &weights, &cost, &cotangent, eps, n_iter);
            let lm = loss(&am, &weights, &cost, &cotangent, eps, n_iter);
            fd_atoms[[ki, j]] = (lp - lm) / (2.0 * h);
        }
    }

    // Property 1: the weight gradient must at least point the right way. The
    // direction of steepest ascent of a real-valued loss is a basic, scale-free
    // invariant; getting its sign wrong means gradient descent moves uphill.
    for ki in 0..k {
        if fd_weights[ki].abs() > 1.0e-6 {
            assert!(
                vjp.d_weights[ki].signum() == fd_weights[ki].signum(),
                "weight-gradient sign disagrees with finite differences at k={ki}: \
                 vjp={vjp_w}, fd={fd_w} (eps={eps}, n_iter={n_iter})",
                vjp_w = vjp.d_weights[ki],
                fd_w = fd_weights[ki],
            );
        }
    }

    // Property 2: the VJP must equal the finite-difference gradient of the same
    // forward map to a few percent (the tolerance the in-crate VJP/FD unit test
    // already uses for the converged regime).
    let mut max_rel = 0.0_f64;
    for ki in 0..k {
        let denom = vjp.d_weights[ki]
            .abs()
            .max(fd_weights[ki].abs())
            .max(1.0e-6);
        let rel = (vjp.d_weights[ki] - fd_weights[ki]).abs() / denom;
        if rel > max_rel {
            max_rel = rel;
        }
    }
    for ki in 0..k {
        for j in 0..m {
            let denom = vjp.d_atoms[[ki, j]]
                .abs()
                .max(fd_atoms[[ki, j]].abs())
                .max(1.0e-6);
            let rel = (vjp.d_atoms[[ki, j]] - fd_atoms[[ki, j]]).abs() / denom;
            if rel > max_rel {
                max_rel = rel;
            }
        }
    }
    assert!(
        max_rel < 0.05,
        "sinkhorn_barycenter_vjp does not match the gradient of \
         sinkhorn_barycenter at the same arguments: max relative error {max_rel:.3e} \
         (eps={eps}, n_iter={n_iter}). A VJP must be the derivative of its forward map."
    );
}
