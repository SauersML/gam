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

use gam::geometry::spd::spd_frechet_mean;
use gam::test_support::reference::{Column, relative_l2, run_python};
use gam::{RiemannianManifold, SpdManifold, load_csvwith_inferred_schema};
use ndarray::{Array1, Array2, ArrayView1};
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand_distr::{Distribution, StandardNormal};
use std::path::Path;

const N: usize = 4; // 4x4 SPD matrices
const M: usize = 10; // number of samples
const MAX_ITERS: usize = 500;
const KARCHER_GRAD_TOL: f64 = 1e-10;

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

/// Riemannian center of mass of `samples` via gam's own SPD Fréchet-mean
/// primitive `spd_frechet_mean`, which performs affine-invariant Riemannian
/// gradient descent (`P ← exp_P(t·Σ_i w_i log_P(X_i))`) with an Armijo line
/// search and stops on the stationarity residual `‖Σ_i w_i log_P(X_i)‖_P`. The
/// line search is what lets it reach the stationarity tolerance on the spread,
/// ill-conditioned synthetic SPD inputs where the bare unit-step fixed point
/// stalls above tol within a fixed budget. Stacks the flat samples into the
/// `M×N²` matrix the primitive consumes (uniform weights).
fn gam_frechet_mean(samples: &[Array1<f64>]) -> Array1<f64> {
    let mut stacked = Array2::<f64>::zeros((samples.len(), N * N));
    for (i, x) in samples.iter().enumerate() {
        for (k, &v) in x.iter().enumerate() {
            stacked[[i, k]] = v;
        }
    }
    spd_frechet_mean(N, stacked.view(), None, KARCHER_GRAD_TOL, MAX_ITERS)
        .expect("gam spd_frechet_mean")
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

    // The Karcher mean of a Hadamard space is unique and gam's descent converges
    // from its order-independent Euclidean-mean start.
    let p = gam_frechet_mean(&flat_samples);
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
        "SPD Fréchet mean (4x4, M={M}): Riemannian gradient norm ‖(1/M)Σ log_P(X_i)‖_P = {grad_norm:.3e}"
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
MAX_ITERS = {MAX_ITERS}
GRAD_TOL = {KARCHER_GRAD_TOL}

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
iters = 0
for it in range(MAX_ITERS):
    acc = np.zeros((N, N))
    for X in mats:
        acc = acc + spd_log(P, X)
    acc = acc / M
    # Affine-invariant tangent norm at P: tr(P^{{-1}} acc P^{{-1}} acc).
    Pinv = np.linalg.inv(P)
    grad_norm = float(np.sqrt(np.trace(Pinv @ acc @ Pinv @ acc)))
    if grad_norm < GRAD_TOL:
        iters = it
        break
    P = spd_exp(P, acc)
    iters = it + 1

emit("mean", 0.5 * (P + P.T).reshape(-1))
"#,
        N = N,
        M = M,
        MAX_ITERS = MAX_ITERS,
        KARCHER_GRAD_TOL = KARCHER_GRAD_TOL,
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

// ===========================================================================
// REAL-DATA ARM
// ===========================================================================
//
// Dataset SOURCE: the classic Leptograpsus "crabs" morphometrics table
// (Campbell & Mahon 1974; distributed as `MASS::crabs` in R, vendored here at
// `bench/datasets/crabs.csv`). 200 rows, 5 continuous body measurements
// (FL, RW, CL, CW, BD in mm) and two 2-level factors `sp` (species B/O) and
// `sex` (M/F) — exactly 4 groups of 50.
//
// The SAME gam capability the synthetic arm proves (affine-invariant SPD
// exp/log → Riemannian center of mass) is exercised here on real covariance
// structure: each crab group's 5×5 sample covariance of the body measurements
// is a genuine SPD matrix, and the four group covariances live on the SPD
// manifold. We hold out half of every group, build the four 5×5 SPD covariances
// from the TRAIN halves, take gam's Fréchet (Karcher) mean of those four, and
// measure how well that train-only center predicts the four HELD-OUT group
// covariances under the affine-invariant metric.
//
// OBJECTIVE held-out metric (truth unknown on real data):
//   V_test(P) = (1/G) Σ_g d²(P, C_test_g)  — mean squared geodesic distance from
//   the center P to each group's TEST-half covariance, all distances via gam.
//
//   PRIMARY (absolute, tool-free bar): gam's train-only center must predict the
//   held-out covariances strictly better than the two naive baselines on this
//   SAME metric — below V_test at the Euclidean (entrywise) mean of the train
//   covariances and below V_test at every individual train-group covariance.
//   This is an honest generalization claim: the Riemannian center transfers to
//   unseen samples of the same groups.
//
//   BASELINE (match-or-beat): a scipy/NumPy re-implementation computes its own
//   Fréchet mean from the IDENTICAL train covariances; gam's held-out
//   V_test(P_gam) must be ≤ V_test(P_ref) + margin. The mature tool is a
//   baseline to match, never an output to copy.

const SPD_DIM: usize = 5; // FL, RW, CL, CW, BD
const N_GROUPS: usize = 4; // species × sex
const REAL_ITERS: usize = 200; // fixed-point iterations (gam and reference share this)

/// Sample covariance (population-normalized by `rows.len()`, then a tiny ridge
/// added to the diagonal) of the selected `rows` over the `SPD_DIM` measurement
/// columns `meas[d]` (each a full-length column indexed by absolute row id).
/// Returns a flat row-major `SPD_DIM*SPD_DIM` vector. The ridge keeps the
/// matrix safely SPD for both engines' Cholesky/eigendecompositions.
fn group_covariance_flat(meas: &[Vec<f64>; SPD_DIM], rows: &[usize]) -> Array1<f64> {
    let m = rows.len() as f64;
    assert!(
        m > SPD_DIM as f64,
        "need more rows than dims for an SPD covariance"
    );
    let mut mean = [0.0f64; SPD_DIM];
    for d in 0..SPD_DIM {
        let s: f64 = rows.iter().map(|&r| meas[d][r]).sum();
        mean[d] = s / m;
    }
    let mut cov = Array2::<f64>::zeros((SPD_DIM, SPD_DIM));
    for &r in rows {
        let mut centered = [0.0f64; SPD_DIM];
        for d in 0..SPD_DIM {
            centered[d] = meas[d][r] - mean[d];
        }
        for i in 0..SPD_DIM {
            for j in 0..SPD_DIM {
                cov[[i, j]] += centered[i] * centered[j];
            }
        }
    }
    cov /= m;
    for i in 0..SPD_DIM {
        cov[[i, i]] += 1e-6;
    }
    // Defensive symmetrization, then row-major flatten for gam's flat API.
    let cov = symmetrize(&cov);
    Array1::from_iter(cov.iter().copied())
}

/// Riemannian center of mass of `samples` (flat `SPD_DIM*SPD_DIM` vectors) via
/// gam's `spd_frechet_mean` primitive — the same line-searched affine-invariant
/// gradient descent the synthetic arm exercises, at the 5×5 ambient size. Stacks
/// the flat covariances into the `G×SPD_DIM²` matrix the primitive consumes.
fn gam_frechet_mean_dim(samples: &[Array1<f64>]) -> Array1<f64> {
    let dim2 = SPD_DIM * SPD_DIM;
    let mut stacked = Array2::<f64>::zeros((samples.len(), dim2));
    for (i, x) in samples.iter().enumerate() {
        for (k, &v) in x.iter().enumerate() {
            stacked[[i, k]] = v;
        }
    }
    spd_frechet_mean(SPD_DIM, stacked.view(), None, 1e-10, REAL_ITERS)
        .expect("gam spd_frechet_mean (real)")
}

/// Squared affine-invariant distance via gam (tangent from `log_map`, squared
/// metric norm `vᵀ G(P) v` from `metric_tensor`). Dimension-agnostic — reuses
/// gam at whatever ambient size `p`/`x` carry.
fn gam_sq_dist_any(spd: &SpdManifold, p: ArrayView1<f64>, x: ArrayView1<f64>) -> f64 {
    let v = spd.log_map(p, x).expect("gam log_map for real distance");
    let g = spd.metric_tensor(p).expect("gam metric_tensor (real)");
    let gv = g.dot(&v);
    v.dot(&gv)
}

/// Held-out dispersion `V_test(P) = (1/G) Σ_g d²(P, C_test_g)` via gam.
fn gam_dispersion_any(spd: &SpdManifold, p: ArrayView1<f64>, samples: &[Array1<f64>]) -> f64 {
    let s: f64 = samples
        .iter()
        .map(|x| gam_sq_dist_any(spd, p, x.view()))
        .sum();
    s / samples.len() as f64
}

#[test]
fn spd_frechet_mean_is_the_riemannian_center_of_mass_on_real_data() {
    // ---- load the real crabs morphometrics table --------------------------
    let ds = load_csvwith_inferred_schema(Path::new(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/bench/datasets/crabs.csv"
    )))
    .expect("load crabs.csv");
    let col = ds.column_map();
    let n = ds.values.nrows();
    assert_eq!(n, 200, "crabs should have 200 rows, got {n}");

    // Five continuous body measurements (full-length columns, absolute-row idx).
    let meas_names = ["FL", "RW", "CL", "CW", "BD"];
    let meas: [Vec<f64>; SPD_DIM] = std::array::from_fn(|d| {
        let idx = col[meas_names[d]];
        ds.values.column(idx).to_vec()
    });
    // The two 2-level factors are stored as integer level codes; group =
    // sp_code*2 + sex_code gives the four species×sex groups, recovered
    // row-aligned from the dataset itself (no reliance on row ordering).
    let sp_idx = col["sp"];
    let sex_idx = col["sex"];
    let sp_code: Vec<f64> = ds.values.column(sp_idx).to_vec();
    let sex_code: Vec<f64> = ds.values.column(sex_idx).to_vec();
    let group_of =
        |r: usize| -> usize { (sp_code[r].round() as usize) * 2 + (sex_code[r].round() as usize) };

    // ---- deterministic train/test split: even row-index in a group = train,
    // odd = test. Done per group so both halves of every group are populated.
    let mut train_rows: Vec<Vec<usize>> = vec![Vec::new(); N_GROUPS];
    let mut test_rows: Vec<Vec<usize>> = vec![Vec::new(); N_GROUPS];
    {
        let mut seen = [0usize; N_GROUPS];
        for r in 0..n {
            let g = group_of(r);
            assert!(g < N_GROUPS, "unexpected crab group code {g}");
            if seen[g] % 2 == 0 {
                train_rows[g].push(r);
            } else {
                test_rows[g].push(r);
            }
            seen[g] += 1;
        }
    }
    for g in 0..N_GROUPS {
        assert!(
            train_rows[g].len() >= 20 && test_rows[g].len() >= 20,
            "crab group {g} split too small: train={} test={}",
            train_rows[g].len(),
            test_rows[g].len()
        );
    }

    // ---- four TRAIN and four TEST 5×5 SPD group covariances ----------------
    let train_cov: Vec<Array1<f64>> = (0..N_GROUPS)
        .map(|g| group_covariance_flat(&meas, &train_rows[g]))
        .collect();
    let test_cov: Vec<Array1<f64>> = (0..N_GROUPS)
        .map(|g| group_covariance_flat(&meas, &test_rows[g]))
        .collect();

    // ---- gam: Karcher mean of the four TRAIN covariances -------------------
    let spd = SpdManifold::new(SPD_DIM);
    let p = gam_frechet_mean_dim(&train_cov);

    // sanity: gam's center is a genuine stationary point on TRAIN — residual
    // Riemannian gradient (tangent mean) vanishes in the affine-invariant norm.
    let dim2 = SPD_DIM * SPD_DIM;
    let mut tangent_mean = Array1::<f64>::zeros(dim2);
    for x in &train_cov {
        tangent_mean += &spd
            .log_map(p.view(), x.view())
            .expect("gam log_map (real grad)");
    }
    tangent_mean /= N_GROUPS as f64;
    let grad_sq = gam_sq_dist_any(&spd, p.view(), p.view()); // 0 baseline; see below
    assert!(grad_sq.abs() < 1e-12, "self-distance must be zero");
    let g_at_p = spd
        .metric_tensor(p.view())
        .expect("gam metric_tensor (real grad)");
    let grad_norm = tangent_mean.dot(&g_at_p.dot(&tangent_mean)).sqrt();
    assert!(
        grad_norm < 1e-6,
        "gam's real-data SPD center is not a Fréchet stationary point: residual \
         Riemannian gradient norm {grad_norm:.3e} (>=1e-6)"
    );

    // ---- PRIMARY objective: held-out dispersion of gam's train-only center -
    let v_test_gam = gam_dispersion_any(&spd, p.view(), &test_cov);

    // Baseline 1: Euclidean (entrywise) mean of the TRAIN covariances — an SPD
    // point that is the WRONG center under the affine-invariant metric.
    let mut euclid = Array1::<f64>::zeros(dim2);
    for x in &train_cov {
        euclid += x;
    }
    euclid /= N_GROUPS as f64;
    let v_test_euclid = gam_dispersion_any(&spd, euclid.view(), &test_cov);

    eprintln!(
        "crabs SPD Fréchet (5x5, G={N_GROUPS}, {REAL_ITERS} iters): grad_norm={grad_norm:.3e} \
         V_test(P_gam)={v_test_gam:.6e} V_test(euclid)={v_test_euclid:.6e}"
    );

    assert!(
        v_test_gam < v_test_euclid,
        "gam's Riemannian center does not beat the Euclidean mean on HELD-OUT \
         dispersion: V_test(P_gam)={v_test_gam:.6e} not below \
         V_test(euclid)={v_test_euclid:.6e}"
    );
    for (g, x) in train_cov.iter().enumerate() {
        let v_at_train = gam_dispersion_any(&spd, x.view(), &test_cov);
        assert!(
            v_test_gam < v_at_train,
            "gam's center does not predict held-out covariances better than \
             train group {g}'s own covariance: V_test(P_gam)={v_test_gam:.6e} \
             not below V_test(train_cov[{g}])={v_at_train:.6e}"
        );
    }
    // Absolute numeric bar: a real, finite, well-conditioned center.
    assert!(
        v_test_gam.is_finite() && v_test_gam > 0.0 && v_test_gam < 5.0,
        "held-out dispersion V_test(P_gam)={v_test_gam:.6e} outside the sane \
         absolute range (0, 5)"
    );

    // ---- BASELINE (match-or-beat): scipy/NumPy Fréchet mean on the SAME
    // TRAIN covariances, evaluated on the SAME TEST covariances. Pass all eight
    // 5×5 matrices as 4-length columns (one per group; train_* and test_*),
    // every column equal length = N_GROUPS, in identical group order.
    let mut cols_owned: Vec<(String, Vec<f64>)> = Vec::new();
    for k in 0..dim2 {
        let train_k: Vec<f64> = (0..N_GROUPS).map(|g| train_cov[g][k]).collect();
        let test_k: Vec<f64> = (0..N_GROUPS).map(|g| test_cov[g][k]).collect();
        cols_owned.push((format!("tr{k}"), train_k));
        cols_owned.push((format!("te{k}"), test_k));
    }
    let cols: Vec<Column<'_>> = cols_owned
        .iter()
        .map(|(name, data)| Column::new(name.as_str(), data.as_slice()))
        .collect();

    let body = format!(
        r#"
D = {D}
G = {G}
ITERS = {ITERS}
KMAX = D * D - 1

def build(prefix):
    mats = []
    for g in range(G):
        flatrow = [float(df["%s%d" % (prefix, k)][g]) for k in range(D * D)]
        A = np.asarray(flatrow, dtype=float).reshape(D, D)
        mats.append(0.5 * (A + A.T))
    return mats

train = build("tr")
test = build("te")

def sym_eig_map(P, f):
    w, V = np.linalg.eigh(0.5 * (P + P.T))
    return (V * f(w)) @ V.T

def spd_log(P, Q):
    sp = sym_eig_map(P, np.sqrt)
    isp = sym_eig_map(P, lambda x: 1.0 / np.sqrt(x))
    mid = isp @ Q @ isp
    return sp @ sym_eig_map(mid, np.log) @ sp

def spd_exp(P, U):
    sp = sym_eig_map(P, np.sqrt)
    isp = sym_eig_map(P, lambda x: 1.0 / np.sqrt(x))
    mid = isp @ U @ isp
    mid = 0.5 * (mid + mid.T)
    return sp @ sym_eig_map(mid, np.exp) @ sp

def sq_dist(P, Q):
    # affine-invariant squared distance = sum(log(eig(isqrt(P) Q isqrt(P)))**2)
    isp = sym_eig_map(P, lambda x: 1.0 / np.sqrt(x))
    mid = isp @ Q @ isp
    w = np.linalg.eigvalsh(0.5 * (mid + mid.T))
    return float(np.sum(np.log(w) ** 2))

# Fréchet mean of the TRAIN covariances (identical recursion to gam).
P = train[0].copy()
for _ in range(ITERS):
    acc = np.zeros((D, D))
    for X in train:
        acc = acc + spd_log(P, X)
    acc = acc / G
    P = spd_exp(P, acc)

# Held-out dispersion of the reference center against the TEST covariances.
v_test_ref = sum(sq_dist(P, C) for C in test) / G
emit("mean", 0.5 * (P + P.T).reshape(-1))
emit("v_test_ref", [v_test_ref])
"#,
        D = SPD_DIM,
        G = N_GROUPS,
        ITERS = REAL_ITERS,
    );

    let r = run_python(&cols, &body);
    let scipy_mean = r.vector("mean");
    assert_eq!(
        scipy_mean.len(),
        dim2,
        "reference real mean length mismatch"
    );
    let v_test_ref = r.scalar("v_test_ref");

    // Context-only: closeness of the two centers (never asserted).
    let gam_flat: Vec<f64> = p.to_vec();
    let rel = relative_l2(&gam_flat, scipy_mean);
    eprintln!(
        "crabs SPD center rel_l2(gam, numpy-reimpl)={rel:.3e} (context only); \
         V_test(P_ref)={v_test_ref:.6e}"
    );

    assert!(
        v_test_gam <= v_test_ref + 1e-9,
        "gam fails to match-or-beat the reference on the HELD-OUT Fréchet \
         objective: V_test(P_gam)={v_test_gam:.6e} > V_test(P_ref)+1e-9={:.6e}",
        v_test_ref + 1e-9
    );
}
