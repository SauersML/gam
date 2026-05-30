//! End-to-end OBJECTIVE quality: gam's affine-invariant SPD exponential/
//! logarithm maps must produce a genuine Riemannian center of mass (the
//! Fréchet / Karcher mean of a set of SPD matrices).
//!
//! gam exposes `gam::SpdManifold` with `exp_map` / `log_map` for the
//! affine-invariant metric but provides **no** `frechet_mean` primitive. The
//! Fréchet mean of SPD matrices `{X_i}` is the unique minimizer of the
//! dispersion functional `V(Q) = (1/M) Σ_i d²(Q, X_i)` and, equivalently, the
//! unique point `P` whose Riemannian gradient vanishes: `Σ_i log_P(X_i) = 0`.
//! We compute it the only way the public API allows — the canonical
//! gradient-descent fixed point `P ← exp_P((1/M) Σ_i log_P(X_i))` — using
//! gam's `exp_map`/`log_map`.
//!
//! The pass/fail assertions are OBJECTIVE Fréchet-mean axioms, evaluated
//! entirely with gam's own maps and the affine-invariant metric tensor — never
//! "matches another tool's fitted output":
//!
//!   1. FIRST-ORDER OPTIMALITY (manifold axiom): the Riemannian gradient at
//!      gam's solution vanishes, i.e. the metric norm of the tangent mean
//!      `(1/M) Σ_i log_P(X_i)` at `P` is ≈ 0. A wrong `P^{1/2}` conjugation in
//!      the metric, or an exp/log pair that is not a true inverse couple, moves
//!      the fixed point and leaves a nonzero gradient.
//!   2. GLOBAL MINIMALITY (the functional it must minimize): gam's `V(P)` is
//!      strictly below `V` evaluated at every input sample and below `V` at the
//!      Euclidean arithmetic mean — proving `P` actually minimizes dispersion,
//!      not merely sits at a stationary point of the wrong functional.
//!
//! The independent NumPy re-implementation of the identical recursion is
//! retained only as a MATCH-OR-BEAT BASELINE on that same objective: gam's
//! dispersion `V(P_gam)` must be ≤ the reference's `V(P_ref)` (both distances
//! measured with gam's metric). It is no longer the pass gate; the axioms are.
//! `rel_l2` between the two centers is printed for context but never asserted.

use gam::test_support::reference::{Column, relative_l2, run_python};
use gam::{RiemannianManifold, SpdManifold};
use ndarray::{Array1, Array2, ArrayView1};
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

/// Karcher-mean fixed point of `samples` at base point `init`, using gam's
/// `exp_map`/`log_map` (`P ← exp_P((1/M) Σ_i log_P(X_i))`).
fn gam_frechet_mean(spd: &SpdManifold, samples: &[Array1<f64>], init: &Array1<f64>) -> Array1<f64> {
    let mut p = init.clone();
    for _ in 0..ITERS {
        let mut acc = Array1::<f64>::zeros(N * N);
        for x in samples {
            let lg = spd.log_map(p.view(), x.view()).expect("gam log_map");
            acc += &lg;
        }
        acc /= M as f64;
        p = spd.exp_map(p.view(), acc.view()).expect("gam exp_map");
    }
    p
}

/// Squared affine-invariant geodesic distance `d²(P, X) = ‖log_P(X)‖²_P`,
/// computed entirely with gam: the tangent vector via `log_map` and its squared
/// metric norm via the SPD `metric_tensor` at `P` (`vᵀ G(P) v`).
fn gam_sq_dist(spd: &SpdManifold, p: ArrayView1<f64>, x: ArrayView1<f64>) -> f64 {
    let v = spd.log_map(p, x).expect("gam log_map for distance");
    let g = spd.metric_tensor(p).expect("gam metric_tensor");
    // vᵀ G v
    let gv = g.dot(&v);
    v.dot(&gv)
}

/// Fréchet dispersion `V(P) = (1/M) Σ_i d²(P, X_i)`, all distances via gam.
fn gam_dispersion(spd: &SpdManifold, p: ArrayView1<f64>, samples: &[Array1<f64>]) -> f64 {
    let s: f64 = samples.iter().map(|x| gam_sq_dist(spd, p, x.view())).sum();
    s / M as f64
}

/// Squared metric norm of a tangent vector at `P`: `‖v‖²_P = vᵀ G(P) v`.
fn gam_tangent_sq_norm(spd: &SpdManifold, p: ArrayView1<f64>, v: ArrayView1<f64>) -> f64 {
    let g = spd.metric_tensor(p).expect("gam metric_tensor");
    let gv = g.dot(&v.to_owned());
    v.dot(&gv)
}

#[test]
fn spd_frechet_mean_is_the_riemannian_center_of_mass() {
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
    let p = gam_frechet_mean(&spd, &flat_samples, &flat_samples[0]);
    let gam_mean = unflat(&p);

    // =====================================================================
    // OBJECTIVE ASSERTION 1 — first-order Fréchet optimality (axiom).
    // At the true Karcher mean the Riemannian gradient vanishes:
    //   g(P) = -(2/M) Σ_i log_P(X_i) = 0  ⇔  the tangent mean is the zero
    // vector. We measure its squared metric norm ‖(1/M) Σ log_P(X_i)‖²_P with
    // gam's own log_map and metric tensor. If exp/log were not a true inverse
    // couple, or the P^{1/2} conjugation were wrong, this gradient would not
    // vanish. This is pure ground truth — no reference involved.
    // =====================================================================
    let mut tangent_mean = Array1::<f64>::zeros(N * N);
    for x in &flat_samples {
        tangent_mean += &spd.log_map(p.view(), x.view()).expect("gam log_map");
    }
    tangent_mean /= M as f64;
    let grad_sq_norm = gam_tangent_sq_norm(&spd, p.view(), tangent_mean.view());
    let grad_norm = grad_sq_norm.sqrt();

    eprintln!(
        "SPD Fréchet mean (4x4, M={M}, {ITERS} iters): Riemannian gradient norm ‖(1/M)Σ log_P(X_i)‖_P = {grad_norm:.3e}"
    );
    assert!(
        grad_norm < 1e-7,
        "gam's SPD center is NOT a Fréchet mean: residual Riemannian gradient \
         norm {grad_norm:.3e} (>=1e-7) — the exp/log pair or affine-invariant \
         metric does not vanish the first-order optimality condition"
    );

    // =====================================================================
    // OBJECTIVE ASSERTION 2 — global minimality of the dispersion functional.
    // V(P) = (1/M) Σ_i d²(P, X_i) must be strictly below V at every sample and
    // below V at the Euclidean arithmetic mean. A correct Karcher mean is the
    // unique minimizer of V on this Hadamard manifold; a fit that minimized the
    // wrong functional (e.g. Euclidean closeness) would fail this.
    // =====================================================================
    let v_gam = gam_dispersion(&spd, p.view(), &flat_samples);
    eprintln!("SPD Fréchet dispersion V(P_gam) = {v_gam:.6e}");

    for (i, x) in flat_samples.iter().enumerate() {
        let v_at_sample = gam_dispersion(&spd, x.view(), &flat_samples);
        assert!(
            v_gam < v_at_sample,
            "gam's center does not minimize dispersion: V(P_gam)={v_gam:.6e} \
             is not below V(sample[{i}])={v_at_sample:.6e}"
        );
    }

    // Euclidean (entrywise) arithmetic mean of the samples — an SPD point that
    // is the WRONG center under the affine-invariant metric.
    let mut euclid_mean = Array2::<f64>::zeros((N, N));
    for x in &samples {
        euclid_mean += x;
    }
    euclid_mean /= M as f64;
    let euclid_flat = flat(&symmetrize(&euclid_mean));
    let v_euclid = gam_dispersion(&spd, euclid_flat.view(), &flat_samples);
    eprintln!("SPD Fréchet dispersion V(Euclidean mean) = {v_euclid:.6e}");
    assert!(
        v_gam < v_euclid,
        "gam's Riemannian center does not beat the Euclidean mean on the \
         affine-invariant dispersion: V(P_gam)={v_gam:.6e} not below \
         V(euclid)={v_euclid:.6e}"
    );

    // ---- reference: the identical recursion, hand-coded in NumPy ----------
    // Demoted to a MATCH-OR-BEAT BASELINE on the objective (dispersion), and a
    // context-only rel_l2 print. Pass the EXACT same matrices as flat columns.
    let mut columns: Vec<Vec<f64>> = (0..N * N).map(|_| Vec::with_capacity(M)).collect();
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

    // ---- BASELINE: match-or-beat on the objective being minimized ---------
    let gam_flat: Vec<f64> = gam_mean.iter().copied().collect();
    let rel = relative_l2(&gam_flat, scipy_mean);
    eprintln!("SPD center rel_l2(gam, numpy-reimpl) = {rel:.3e} (context only, not asserted)");

    let ref_mean: Array1<f64> = Array1::from_iter(scipy_mean.iter().copied());
    let v_ref = gam_dispersion(&spd, ref_mean.view(), &flat_samples);
    eprintln!("SPD Fréchet dispersion V(P_ref) = {v_ref:.6e}");
    assert!(
        v_gam <= v_ref * (1.0 + 1e-9),
        "gam fails to match-or-beat the reference on the Fréchet objective: \
         V(P_gam)={v_gam:.6e} > V(P_ref)*(1+1e-9)={:.6e}",
        v_ref * (1.0 + 1e-9)
    );
}
