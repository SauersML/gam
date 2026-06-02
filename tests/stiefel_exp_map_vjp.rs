//! Finite-difference verification of the analytic Stiefel exponential VJP
//! (`StiefelManifold::exp_map_vjp`) for `k > 1`.
//!
//! The defining identity of a reverse-mode VJP is the directional adjoint
//! relation: for any output cotangent `Ḡ`, point perturbation `δY`, and tangent
//! perturbation `δv`,
//!
//! ```text
//!   ⟨Ḡ, d/dε exp_map(Y + ε·δY, v + ε·δv)⟩|_{ε=0}
//!       = ⟨grad_p, δY⟩ + ⟨grad_v, δv⟩,
//! ```
//!
//! where `(grad_p, grad_v) = exp_map_vjp(Y, v, Ḡ)`. The left-hand directional
//! derivative is computed by central finite differences; the right-hand side by
//! the analytic VJP under test. We sweep several `(n, k)` with `k > 1`, plus a
//! `k == 1` routing check against the sphere VJP.

use gam::{RiemannianManifold, SphereManifold, StiefelManifold};
use ndarray::{Array1, Array2};

/// Tiny deterministic xorshift PRNG: reproducible, no external crates.
struct Rng(u64);

impl Rng {
    fn new(seed: u64) -> Self {
        Rng(seed | 1)
    }
    fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x
    }
    /// Uniform in (-1, 1).
    fn next_signed(&mut self) -> f64 {
        let u = (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64; // [0,1)
        2.0 * u - 1.0
    }
}

/// Random `n×k` matrix with entries in (-1, 1).
fn rand_matrix(rng: &mut Rng, n: usize, k: usize) -> Array2<f64> {
    let mut m = Array2::<f64>::zeros((n, k));
    for i in 0..n {
        for j in 0..k {
            m[[i, j]] = rng.next_signed();
        }
    }
    m
}

/// Orthonormalize the columns of `a` (n×k, n≥k) via modified Gram–Schmidt to get
/// a valid Stiefel point `YᵀY = I_k`.
fn orthonormalize(a: &Array2<f64>) -> Array2<f64> {
    let n = a.nrows();
    let k = a.ncols();
    let mut q = a.clone();
    for j in 0..k {
        for i in 0..j {
            let mut dot = 0.0;
            for r in 0..n {
                dot += q[[r, i]] * q[[r, j]];
            }
            for r in 0..n {
                q[[r, j]] -= dot * q[[r, i]];
            }
        }
        let mut nrm = 0.0;
        for r in 0..n {
            nrm += q[[r, j]] * q[[r, j]];
        }
        let nrm = nrm.sqrt();
        for r in 0..n {
            q[[r, j]] /= nrm;
        }
    }
    q
}

fn flatten(m: &Array2<f64>) -> Array1<f64> {
    let mut v = Array1::<f64>::zeros(m.nrows() * m.ncols());
    let k = m.ncols();
    for i in 0..m.nrows() {
        for j in 0..k {
            v[i * k + j] = m[[i, j]];
        }
    }
    v
}

fn dot(a: &Array1<f64>, b: &Array1<f64>) -> f64 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

/// Verify the directional adjoint identity for a single `(n, k, seed)` case.
fn check_case(n: usize, k: usize, seed: u64) {
    let mut rng = Rng::new(seed);
    let manifold = StiefelManifold::new(k, n).expect("valid Stiefel dimensions");

    // Stiefel point Y (orthonormal columns).
    let y = orthonormalize(&rand_matrix(&mut rng, n, k));
    let y_flat = flatten(&y);

    // Raw tangent input v (the exp_map projects internally), output cotangent Ḡ.
    let v = flatten(&rand_matrix(&mut rng, n, k));
    let g_bar = flatten(&rand_matrix(&mut rng, n, k));

    // Perturbation directions for the point and the tangent.
    let dy = flatten(&rand_matrix(&mut rng, n, k));
    let dv = flatten(&rand_matrix(&mut rng, n, k));

    // Analytic VJP.
    let (grad_p, grad_v) = manifold
        .exp_map_vjp(y_flat.view(), v.view(), g_bar.view())
        .expect("analytic Stiefel VJP must succeed for k>1");
    let analytic = dot(&grad_p, &dy) + dot(&grad_v, &dv);

    // Central finite-difference directional derivative of
    //   ε ↦ ⟨Ḡ, exp_map(Y + ε·δY, v + ε·δv)⟩.
    let h = 1e-6;
    let perturbed = |eps: f64| -> f64 {
        let yp = &y_flat + &(&dy * eps);
        let vp = &v + &(&dv * eps);
        let out = manifold
            .exp_map(yp.view(), vp.view())
            .expect("exp_map at perturbed inputs");
        dot(&g_bar, &out)
    };
    let fd = (perturbed(h) - perturbed(-h)) / (2.0 * h);

    let scale = analytic.abs().max(fd.abs()).max(1e-8);
    let rel = (analytic - fd).abs() / scale;
    assert!(
        rel < 1e-4,
        "VJP adjoint identity failed for St({n},{k}) seed={seed}: \
         analytic={analytic:.10e}, fd={fd:.10e}, rel_err={rel:.3e}"
    );
}

#[test]
fn stiefel_exp_map_vjp_matches_finite_difference_k_gt_1() {
    // Several (n, k) with k > 1, multiple seeds each so the random Y, v, Ḡ, and
    // perturbation directions vary and exercise both point and tangent grads.
    let cases = [(4usize, 2usize), (5, 3), (6, 2), (7, 4)];
    for (n, k) in cases {
        for seed in 1u64..=4 {
            check_case(n, k, seed * 9973 + (n as u64) * 31 + k as u64);
        }
    }
}

#[test]
fn stiefel_exp_map_vjp_point_only_perturbation() {
    // Isolate ⟨grad_p, δY⟩ by setting δv = 0 in the finite-difference probe.
    let n = 5;
    let k = 3;
    let mut rng = Rng::new(424242);
    let manifold = StiefelManifold::new(k, n).expect("valid dims");
    let y = orthonormalize(&rand_matrix(&mut rng, n, k));
    let y_flat = flatten(&y);
    let v = flatten(&rand_matrix(&mut rng, n, k));
    let g_bar = flatten(&rand_matrix(&mut rng, n, k));
    let dy = flatten(&rand_matrix(&mut rng, n, k));

    let (grad_p, _grad_v) = manifold
        .exp_map_vjp(y_flat.view(), v.view(), g_bar.view())
        .expect("VJP");
    let analytic = dot(&grad_p, &dy);

    let h = 1e-6;
    let probe = |eps: f64| -> f64 {
        let yp = &y_flat + &(&dy * eps);
        let out = manifold.exp_map(yp.view(), v.view()).expect("exp_map");
        dot(&g_bar, &out)
    };
    let fd = (probe(h) - probe(-h)) / (2.0 * h);
    let rel = (analytic - fd).abs() / analytic.abs().max(fd.abs()).max(1e-8);
    assert!(rel < 1e-4, "point-only grad mismatch: {analytic} vs {fd}");
}

#[test]
fn stiefel_exp_map_vjp_tangent_only_perturbation() {
    // Isolate ⟨grad_v, δv⟩ by setting δY = 0 in the finite-difference probe.
    let n = 6;
    let k = 2;
    let mut rng = Rng::new(0xDEAD_BEEF);
    let manifold = StiefelManifold::new(k, n).expect("valid dims");
    let y = orthonormalize(&rand_matrix(&mut rng, n, k));
    let y_flat = flatten(&y);
    let v = flatten(&rand_matrix(&mut rng, n, k));
    let g_bar = flatten(&rand_matrix(&mut rng, n, k));
    let dv = flatten(&rand_matrix(&mut rng, n, k));

    let (_grad_p, grad_v) = manifold
        .exp_map_vjp(y_flat.view(), v.view(), g_bar.view())
        .expect("VJP");
    let analytic = dot(&grad_v, &dv);

    let h = 1e-6;
    let probe = |eps: f64| -> f64 {
        let vp = &v + &(&dv * eps);
        let out = manifold.exp_map(y_flat.view(), vp.view()).expect("exp_map");
        dot(&g_bar, &out)
    };
    let fd = (probe(h) - probe(-h)) / (2.0 * h);
    let rel = (analytic - fd).abs() / analytic.abs().max(fd.abs()).max(1e-8);
    assert!(rel < 1e-4, "tangent-only grad mismatch: {analytic} vs {fd}");
}

#[test]
fn stiefel_exp_map_vjp_k1_routes_to_sphere() {
    // St(n, 1) is the unit sphere S^{n-1}; the VJP must delegate to the sphere
    // VJP and produce identical gradients.
    let n = 5;
    let mut rng = Rng::new(13579);
    let stiefel = StiefelManifold::new(1, n).expect("valid dims");
    let sphere = SphereManifold::new(n - 1);

    // Unit point on the sphere.
    let mut p = rand_matrix(&mut rng, n, 1);
    let mut nrm = 0.0;
    for i in 0..n {
        nrm += p[[i, 0]] * p[[i, 0]];
    }
    let nrm = nrm.sqrt();
    for i in 0..n {
        p[[i, 0]] /= nrm;
    }
    let p_flat = flatten(&p);
    let v = flatten(&rand_matrix(&mut rng, n, 1));
    let g_bar = flatten(&rand_matrix(&mut rng, n, 1));

    let (sp_p, sp_v) = stiefel
        .exp_map_vjp(p_flat.view(), v.view(), g_bar.view())
        .expect("stiefel k=1 VJP");
    let (sph_p, sph_v) = sphere
        .exp_map_vjp(p_flat.view(), v.view(), g_bar.view())
        .expect("sphere VJP");

    for i in 0..n {
        assert!(
            (sp_p[i] - sph_p[i]).abs() < 1e-12,
            "k=1 grad_p[{i}] differs from sphere: {} vs {}",
            sp_p[i],
            sph_p[i]
        );
        assert!(
            (sp_v[i] - sph_v[i]).abs() < 1e-12,
            "k=1 grad_v[{i}] differs from sphere: {} vs {}",
            sp_v[i],
            sph_v[i]
        );
    }
}
