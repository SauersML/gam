//! End-to-end quality: gam's affine-invariant SPD exponential/logarithm maps
//! must produce the same Riemannian center of mass (Fréchet mean) as an
//! independent, hand-coded reference descent in Python.
//!
//! gam exposes `gam::SpdManifold` with `exp_map` / `log_map` for the
//! affine-invariant metric but provides **no** `frechet_mean` primitive. The
//! Fréchet (Karcher) mean of SPD matrices `{X_i}` is the fixed point of the
//! identity map under the exponential: the unique `P` with
//! `Σ_i log_P(X_i) = 0`. We compute it the only way the public API allows —
//! the canonical Riemannian gradient-descent fixed-point iteration
//! `P ← exp_P( (1/m) Σ_i log_P(X_i) )` — using gam's `exp_map`/`log_map`, and
//! compare against the *same* iteration coded from scratch in NumPy (matrix
//! sqrt / log / exp via symmetric eigendecomposition). Both consume bit-
//! identical input matrices.
//!
//! This validates that `exp_map` and `log_map` are a correctly paired inverse
//! couple in the affine-invariant metric: a missing factor in the metric (e.g.
//! a wrong `P^{1/2}` conjugation) would move the fixed point and the centers
//! would drift apart. There is no external R/Python tool *call* into gam here;
//! the reference is an independent re-implementation of the identical
//! mathematical recursion, which is the cleanest ground truth for the
//! exp/log pair.

use gam::test_support::reference::{Column, max_abs_diff, relative_l2, run_python};
use gam::{RiemannianManifold, SpdManifold};
use ndarray::{Array1, Array2};
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand_distr::{Distribution, StandardNormal};

const N: usize = 4; // 4x4 SPD matrices
const M: usize = 10; // number of samples
const ITERS: usize = 100; // fixed-point iterations (matches the reference)

/// Symmetrize a square matrix: (A + Aᵀ)/2. SPD points and tangent vectors are
/// symmetric; we symmetrize defensively to kill round-off asymmetry before
/// handing the flat buffer to gam.
fn symmetrize(a: &Array2<f64>) -> Array2<f64> {
    let mut s = a + &a.t();
    s *= 0.5;
    s
}

/// Row-major flatten of an `N×N` matrix into the `N*N` ambient vector that
/// `SpdManifold::exp_map`/`log_map` expect.
fn flat(a: &Array2<f64>) -> Array1<f64> {
    Array1::from_iter(a.iter().copied())
}

/// Reshape an ambient `N*N` flat vector back to an `N×N` matrix (row-major).
fn unflat(v: &Array1<f64>) -> Array2<f64> {
    Array2::from_shape_vec((N, N), v.to_vec()).expect("reshape N*N -> NxN")
}

#[test]
fn spd_frechet_mean_matches_handcoded_riemannian_descent() {
    // ---- generate M random 4x4 SPD matrices: X = AᵀA, A ~ Gaussian --------
    // Deterministic RNG so the data is reproducible and identical across runs.
    let mut rng = StdRng::seed_from_u64(0x5D_7B_3A_19_C4_E2_F0_91);
    let mut samples: Vec<Array2<f64>> = Vec::with_capacity(M);
    for _ in 0..M {
        let mut a = Array2::<f64>::zeros((N, N));
        for v in a.iter_mut() {
            *v = StandardNormal.sample(&mut rng);
        }
        // X = AᵀA is symmetric PSD; with a Gaussian A it is SPD almost surely.
        // Add a small ridge to keep the smallest eigenvalue safely positive so
        // both engines' Cholesky/eigendecompositions stay well-conditioned.
        let mut x = symmetrize(&a.t().dot(&a));
        for i in 0..N {
            x[[i, i]] += 1e-3;
        }
        samples.push(x);
    }

    // ---- gam: Karcher-mean fixed point via exp_map/log_map ----------------
    let spd = SpdManifold::new(N);
    let flat_samples: Vec<Array1<f64>> = samples.iter().map(flat).collect();

    // Initialize at the first sample (any SPD point works; the Karcher mean of
    // a Hadamard space is unique and the descent converges from anywhere).
    let mut p = flat_samples[0].clone();
    for _ in 0..ITERS {
        // Tangent mean at p: (1/M) Σ_i log_p(X_i), all in the tangent space at p.
        let mut acc = Array1::<f64>::zeros(N * N);
        for x in &flat_samples {
            let lg = spd.log_map(p.view(), x.view()).expect("gam log_map");
            acc += &lg;
        }
        acc /= M as f64;
        // Retract along the averaged tangent direction.
        p = spd.exp_map(p.view(), acc.view()).expect("gam exp_map");
    }
    let gam_mean = unflat(&p);

    // ---- reference: the identical recursion, hand-coded in NumPy ----------
    // Pass the EXACT same matrices as flat columns (one column per (row,col)
    // entry, M rows) so the reference consumes bit-identical input.
    let mut columns: Vec<Vec<f64>> = vec![Vec::with_capacity(M); N * N];
    for x in &samples {
        for (k, val) in x.iter().enumerate() {
            columns[k].push(*val);
        }
    }
    let names: Vec<String> = (0..N * N).map(|k| format!("x{k}")).collect();
    let cols: Vec<Column<'_>> = names
        .iter()
        .zip(columns.iter())
        .map(|(name, data)| Column::new(name.as_str(), data.as_slice()))
        .collect();

    let body = format!(
        r#"
N = {N}
M = {M}
ITERS = {ITERS}

# Rebuild the M NxN matrices from the flat columns x0..x{kmax}.
mats = []
for r in range(M):
    flatrow = [float(df["x%d" % k][r]) for k in range(N * N)]
    A = np.asarray(flatrow, dtype=float).reshape(N, N)
    mats.append(0.5 * (A + A.T))  # symmetrize, same as gam

def sym_eig_map(P, f):
    # Symmetric-eigendecomposition functional calculus: P = V diag(w) V^T,
    # return V diag(f(w)) V^T. Used for sqrt / inv-sqrt / log / exp on SPD.
    w, V = np.linalg.eigh(0.5 * (P + P.T))
    return (V * f(w)) @ V.T

def spd_log(P, Q):
    # Affine-invariant log_P(Q) = sqrt(P) . log( isqrt(P) Q isqrt(P) ) . sqrt(P).
    sp = sym_eig_map(P, np.sqrt)
    isp = sym_eig_map(P, lambda x: 1.0 / np.sqrt(x))
    mid = isp @ Q @ isp
    lm = sym_eig_map(mid, np.log)
    return sp @ lm @ sp

def spd_exp(P, U):
    # Affine-invariant exp_P(U) = sqrt(P) . expm( isqrt(P) U isqrt(P) ) . sqrt(P).
    sp = sym_eig_map(P, np.sqrt)
    isp = sym_eig_map(P, lambda x: 1.0 / np.sqrt(x))
    mid = 0.5 * (isp @ U @ isp + (isp @ U @ isp).T)  # exp of a symmetric mat
    em = sym_eig_map(mid, np.exp)
    return sp @ em @ sp

P = mats[0].copy()
for _ in range(ITERS):
    acc = np.zeros((N, N))
    for X in mats:
        acc = acc + spd_log(P, X)
    acc = acc / M
    P = spd_exp(P, acc)

emit("mean", 0.5 * (P + P.T).reshape(-1))
"#,
        N = N,
        M = M,
        ITERS = ITERS,
        kmax = N * N - 1,
    );

    let r = run_python(&cols, &body);
    let scipy_mean = r.vector("mean");
    assert_eq!(scipy_mean.len(), N * N, "reference mean length mismatch");

    // ---- compare ----------------------------------------------------------
    let gam_flat: Vec<f64> = gam_mean.iter().copied().collect();
    let rel = relative_l2(&gam_flat, scipy_mean);
    let mad = max_abs_diff(&gam_flat, scipy_mean);
    let frob: f64 = gam_flat
        .iter()
        .zip(scipy_mean)
        .map(|(a, b)| (a - b) * (a - b))
        .sum::<f64>()
        .sqrt();

    eprintln!(
        "SPD Fréchet mean (4x4, M={M}, {ITERS} iters): \
         frobenius={frob:.3e} rel_l2={rel:.3e} max_abs_diff={mad:.3e}"
    );

    // Both engines run the identical fixed-point recursion to a converged
    // Karcher mean; the only difference is the matrix-function backend (gam's
    // spectral maps vs NumPy's eigh). After 100 iterations the centers must
    // agree to numerical precision. Bounds match the spec.
    assert!(
        frob < 1e-7,
        "SPD Fréchet means diverge in Frobenius norm: {frob:.3e} (>=1e-7) \
         — exp/log pairing or affine-invariant metric is wrong"
    );
    assert!(
        mad < 1e-8,
        "SPD Fréchet means diverge componentwise: max_abs_diff={mad:.3e} (>=1e-8)"
    );
}
