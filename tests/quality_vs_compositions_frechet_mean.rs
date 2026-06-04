//! End-to-end quality: gam's Fréchet (center-of-mass) mean on the simplex is
//! asserted to be the TRUE Aitchison-geometry barycenter — i.e. the *minimizer*
//! of the weighted sum of squared Aitchison distances — not merely "close to
//! what a reference package prints".
//!
//! ## Objective metric asserted (not "matches a peer tool")
//! The Aitchison Fréchet mean of points `x_1..x_n` with weights `w_i` is, by
//! definition, the unique minimizer over the open simplex of
//!
//!     F(mu) = sum_i w_i * d_A(x_i, mu)^2,
//!
//! where `d_A` is the Aitchison distance (Euclidean distance between centered
//! log-ratio / clr coordinates). We assert gam's output is that minimizer via
//! three OBJECTIVE quality criteria, computed on gam's OWN output:
//!
//!   1. **First-order optimality (stationarity).** In clr coordinates the
//!      gradient of `F` is `2 * sum_i w_i (clr(mu) - clr(x_i))`. At the true
//!      barycenter this is exactly zero. We assert `||grad F(clr(gam_mean))||`
//!      is at round-off (<= 1e-10). This is the defining axiom of a Fréchet mean
//!      and is a pure intrinsic property of gam's result.
//!   2. **Global optimality vs. mature baselines (match-or-beat).** We also fit
//!      the barycenter with R `compositions::mean(acomp())` and Python
//!      `scipy.stats.gmean`, evaluate the SAME objective `F` at each, and assert
//!      `F(gam) <= F(reference) + tol`. gam must do as-well-or-better at the
//!      objective it claims to optimize; the mature tools are demoted to
//!      baselines-to-match-or-beat on that objective, never the pass criterion.
//!   3. **Ground-truth recovery (EXCEPTION: exact closed form).** The Aitchison
//!      barycenter has a known closed form — the closed weighted geometric mean.
//!      We compute it directly in-test (exact ground truth) and assert gam
//!      recovers it to floating-point precision. scipy.stats.gmean is an
//!      independent ground-truth confirmation for the unweighted case.
//!
//! Plus the intrinsic simplex constraints (closure: rows sum to 1, strictly
//! positive) and the uniform-weight invariance of the weighted barycenter.
//!
//! The reference tools therefore remain — but only as (a) baselines we must
//! match-or-beat on the objective `F`, and (b) an independent recomputation of
//! the exact closed form. The pass/fail criterion is gam's own optimality, not
//! agreement with a peer fit.
//!
//! ## Data
//! Identical data is fed to gam and the baselines. We generate, with a fixed
//! seed, 100 Dirichlet(alpha = [1,1,1,1]) draws over 4 components (Dirichlet =
//! normalized independent Gamma(1,1) variates) and a fixed Exp(1) weight vector.

use gam::geometry::simplex::simplex_frechet_mean;
use gam::load_csvwith_inferred_schema;
use gam::test_support::reference::{Column, relative_l2, run_python, run_r};
use ndarray::{Array2, ArrayView1};
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand_distr::{Distribution, Exp, Gamma};
use std::path::Path;

/// Real compositional benchmark: AFM (Na2O+K2O / FeO / MgO) of 23 Skye lava
/// flows. Source: Aitchison, J. (1986) *The Statistical Analysis of
/// Compositional Data*, dataset shipped as `compositions::SkyeAFM` and used
/// throughout the compositional-data literature. Local copy:
/// `bench/datasets/skye_afm_lavas.csv` (columns `A`,`F`,`M`; integer parts that
/// closure normalizes onto the 2-simplex).
const SKYE_CSV: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/bench/datasets/skye_afm_lavas.csv"
);

/// Mean squared Aitchison distance of held-out compositions `points` (N x D,
/// already strictly positive) to a fixed center `mu`. This is the *held-out*
/// value of the per-point Fréchet objective `(1/N) sum_i d_A(x_i, mu)^2`: the
/// out-of-sample generalization error of a candidate barycenter. Lower is
/// better; minimized over `mu` by the population Aitchison mean.
fn mean_sq_aitchison(points: &Array2<f64>, mu: &[f64]) -> f64 {
    let (n, d) = points.dim();
    let clr_mu = clr(mu);
    let mut acc = 0.0_f64;
    for i in 0..n {
        let row: Vec<f64> = (0..d).map(|c| points[[i, c]]).collect();
        let clr_x = clr(&row);
        let sq: f64 = (0..d).map(|k| (clr_x[k] - clr_mu[k]).powi(2)).sum();
        acc += sq;
    }
    acc / n.max(1) as f64
}

/// Centered-log-ratio coordinates of a strictly-positive composition row.
/// `clr(p)_k = ln p_k - mean_j ln p_j`. Aitchison distance is the Euclidean
/// distance between clr coordinates, so the barycenter objective is a plain
/// least-squares problem in clr space.
fn clr(p: &[f64]) -> Vec<f64> {
    let d = p.len();
    let log: Vec<f64> = p.iter().map(|v| v.ln()).collect();
    let mean = log.iter().sum::<f64>() / d as f64;
    log.iter().map(|l| l - mean).collect()
}

/// Weighted Fréchet objective `F(mu) = sum_i w_i * ||clr(x_i) - clr(mu)||^2`
/// evaluated on already-closed rows `points` (N x D) with normalized weights.
fn frechet_objective(points: &Array2<f64>, weights: &[f64], mu: &[f64]) -> f64 {
    let (n, d) = points.dim();
    let clr_mu = clr(mu);
    let mut f = 0.0_f64;
    for i in 0..n {
        let row: Vec<f64> = (0..d).map(|c| points[[i, c]]).collect();
        let clr_x = clr(&row);
        let sq: f64 = (0..d).map(|k| (clr_x[k] - clr_mu[k]).powi(2)).sum();
        f += weights[i] * sq;
    }
    f
}

/// Gradient L2-norm of `F` w.r.t. clr(mu): `grad = 2 * sum_i w_i (clr(mu) -
/// clr(x_i))`. Zero exactly at the barycenter (clr(mu) = weighted mean of the
/// clr(x_i)). Returns `||grad||_2`.
fn frechet_gradient_norm(points: &Array2<f64>, weights: &[f64], mu: &[f64]) -> f64 {
    let (n, d) = points.dim();
    let clr_mu = clr(mu);
    let mut grad = vec![0.0_f64; d];
    for i in 0..n {
        let row: Vec<f64> = (0..d).map(|c| points[[i, c]]).collect();
        let clr_x = clr(&row);
        for k in 0..d {
            grad[k] += weights[i] * (clr_mu[k] - clr_x[k]);
        }
    }
    let g2: f64 = grad.iter().map(|g| (2.0 * g).powi(2)).sum();
    g2.sqrt()
}

/// Exact closed-form Aitchison barycenter: closed weighted geometric mean.
/// `mu_k proportional to exp(sum_i w_i ln x_ik)`, closed to the simplex. This is
/// the analytic ground truth for the minimizer of `F`.
fn closed_weighted_geometric_mean(points: &Array2<f64>, weights: &[f64]) -> Vec<f64> {
    let (n, d) = points.dim();
    let mut log_mean = vec![0.0_f64; d];
    for i in 0..n {
        for k in 0..d {
            log_mean[k] += weights[i] * points[[i, k]].ln();
        }
    }
    let mx = log_mean.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let mut out: Vec<f64> = log_mean.iter().map(|l| (l - mx).exp()).collect();
    let s: f64 = out.iter().sum();
    for v in out.iter_mut() {
        *v /= s;
    }
    out
}

#[test]
fn frechet_mean_is_the_aitchison_barycenter() {
    // ---- generate identical compositional data (seeded, reproducible) -------
    const N: usize = 100;
    const D: usize = 4;
    let mut rng = StdRng::seed_from_u64(20260529);
    let gamma = Gamma::<f64>::new(1.0, 1.0).expect("Gamma(1,1) for Dirichlet(alpha=1)");
    let exp = Exp::<f64>::new(1.0).expect("Exp(1) weights");

    // points: row-major N x D, each row a closed (sums to 1) composition.
    let mut points = Array2::<f64>::zeros((N, D));
    for i in 0..N {
        let mut g = [0.0_f64; D];
        let mut total = 0.0_f64;
        for value in g.iter_mut() {
            let s = gamma.sample(&mut rng);
            *value = s;
            total += s;
        }
        for (col, &value) in g.iter().enumerate() {
            points[[i, col]] = value / total;
        }
    }

    // A single fixed non-uniform weight vector drawn from Exp(1).
    let weights: Vec<f64> = (0..N).map(|_| exp.sample(&mut rng)).collect();
    let wsum: f64 = weights.iter().sum();
    let weights_norm: Vec<f64> = weights.iter().map(|w| w / wsum).collect();
    let uniform_norm = vec![1.0_f64 / N as f64; N];

    // Column-major flattening for the reference engines (one column per part).
    let part_cols: Vec<Vec<f64>> = (0..D)
        .map(|c| (0..N).map(|r| points[[r, c]]).collect::<Vec<f64>>())
        .collect();

    // ---- gam: unweighted and weighted Fréchet means ------------------------
    let gam_unweighted = simplex_frechet_mean(points.view(), None).expect("gam unweighted mean");
    let w_view = ArrayView1::from(weights.as_slice());
    let gam_weighted =
        simplex_frechet_mean(points.view(), Some(w_view)).expect("gam weighted mean");

    assert_eq!(gam_unweighted.len(), D);
    assert_eq!(gam_weighted.len(), D);

    // ---- INTRINSIC property: simplex closure (sum = 1, strictly > 0) --------
    for (label, m) in [("unweighted", &gam_unweighted), ("weighted", &gam_weighted)] {
        let s: f64 = m.iter().sum();
        assert!(
            (s - 1.0).abs() < 1e-12,
            "{label} Fréchet mean must lie on the simplex (sum=1): sum={s:.3e}"
        );
        for (k, &v) in m.iter().enumerate() {
            assert!(
                v > 0.0 && v.is_finite(),
                "{label} Fréchet mean component {k} must be strictly positive: {v:.3e}"
            );
        }
    }

    // ---- OBJECTIVE 1: first-order optimality of gam's own output -----------
    // grad F at the barycenter is exactly zero; we assert it is at round-off.
    let grad_unw = frechet_gradient_norm(&points, &uniform_norm, &gam_unweighted);
    let grad_w = frechet_gradient_norm(&points, &weights_norm, &gam_weighted);
    eprintln!(
        "simplex Fréchet stationarity: ||grad F||(unweighted)={grad_unw:.3e} (weighted)={grad_w:.3e}"
    );
    assert!(
        grad_unw < 1e-10,
        "gam unweighted mean is not a stationary point of the Aitchison Fréchet objective: ||grad F||={grad_unw:.3e}"
    );
    assert!(
        grad_w < 1e-10,
        "gam weighted mean is not a stationary point of the Aitchison Fréchet objective: ||grad F||={grad_w:.3e}"
    );

    // ---- OBJECTIVE 3 (ground truth, exact closed form): truth recovery ------
    // The barycenter equals the closed weighted geometric mean exactly.
    let truth_unweighted = closed_weighted_geometric_mean(&points, &uniform_norm);
    let truth_weighted = closed_weighted_geometric_mean(&points, &weights_norm);
    let truth_unw_rel = relative_l2(&gam_unweighted, &truth_unweighted);
    let truth_w_rel = relative_l2(&gam_weighted, &truth_weighted);
    eprintln!(
        "simplex Fréchet truth-recovery rel_l2: unweighted={truth_unw_rel:.3e} weighted={truth_w_rel:.3e}"
    );
    assert!(
        truth_unw_rel < 1e-12,
        "gam unweighted mean must equal the exact closed weighted geometric mean (ground truth): rel_l2={truth_unw_rel:.3e}"
    );
    assert!(
        truth_w_rel < 1e-12,
        "gam weighted mean must equal the exact closed weighted geometric mean (ground truth): rel_l2={truth_w_rel:.3e}"
    );

    // ---- INTRINSIC: unweighted == uniform-weighted (defining invariance) ----
    let gam_uniform = simplex_frechet_mean(
        points.view(),
        Some(ArrayView1::from(uniform_norm.as_slice())),
    )
    .expect("uniform");
    let uniform_rel = relative_l2(&gam_uniform, &gam_unweighted);
    eprintln!("simplex Fréchet: unweighted-vs-uniform rel_l2={uniform_rel:.3e}");
    assert!(
        uniform_rel < 1e-10,
        "unweighted mean must equal uniform-weighted mean: rel_l2={uniform_rel:.3e}"
    );

    // ---- baselines: R compositions::acomp() + log-space weighted mean -------
    let mut r_columns: Vec<Column<'_>> = (0..D)
        .map(|c| Column::new(part_name(c), &part_cols[c]))
        .collect();
    r_columns.push(Column::new("w", &weights));
    let r = run_r(
        &r_columns,
        r#"
        suppressPackageStartupMessages(library(compositions))
        X <- as.matrix(df[, c("p0","p1","p2","p3")])
        ac <- acomp(X)
        mu <- as.numeric(mean(ac))
        emit("r_unweighted", mu)
        w <- df$w / sum(df$w)
        stopifnot(all(is.finite(X)), all(X > 0), all(is.finite(w)))
        L <- log(X)
        ml <- as.numeric(crossprod(w, L))
        wmu <- exp(ml - max(ml)); wmu <- wmu / sum(wmu)
        stopifnot(all(is.finite(wmu)), all(wmu > 0))
        emit("r_weighted", wmu)
        "#,
    );
    let r_unweighted = r.vector("r_unweighted");
    let r_weighted = r.vector("r_weighted");
    assert_eq!(r_unweighted.len(), D, "R unweighted mean length");
    assert_eq!(r_weighted.len(), D, "R weighted mean length");

    // ---- baseline: scipy.stats.gmean (unweighted) ---------------------------
    let py_columns: Vec<Column<'_>> = (0..D)
        .map(|c| Column::new(part_name(c), &part_cols[c]))
        .collect();
    let py = run_python(
        &py_columns,
        r#"
from scipy.stats import gmean
X = np.column_stack([df["p0"], df["p1"], df["p2"], df["p3"]])
g = gmean(X, axis=0)
g = g / g.sum()
emit("py_unweighted", g)
        "#,
    );
    let py_unweighted = py.vector("py_unweighted");
    assert_eq!(py_unweighted.len(), D, "scipy unweighted mean length");

    // ---- OBJECTIVE 2: match-or-beat the baselines on the objective F --------
    // Evaluate the SAME Fréchet objective at gam's and each baseline's output;
    // gam must achieve a value no larger (modulo round-off). This is the true
    // "as-good-or-better" quality claim, not agreement-with-a-peer.
    let f_gam_unw = frechet_objective(&points, &uniform_norm, &gam_unweighted);
    let f_gam_w = frechet_objective(&points, &weights_norm, &gam_weighted);
    let f_r_unw = frechet_objective(&points, &uniform_norm, r_unweighted);
    let f_r_w = frechet_objective(&points, &weights_norm, r_weighted);
    let f_py_unw = frechet_objective(&points, &uniform_norm, py_unweighted);

    // Context only: how closely the baselines reproduce gam's barycenter.
    let rel_unw_r = relative_l2(&gam_unweighted, r_unweighted);
    let rel_unw_py = relative_l2(&gam_unweighted, py_unweighted);
    let rel_w_r = relative_l2(&gam_weighted, r_weighted);
    eprintln!(
        "simplex Fréchet objective F (lower=better): \
         gam_unw={f_gam_unw:.6e} R_unw={f_r_unw:.6e} scipy_unw={f_py_unw:.6e}; \
         gam_w={f_gam_w:.6e} R_w={f_r_w:.6e}"
    );
    eprintln!(
        "simplex Fréchet baseline rel_l2 (context only): \
         unweighted vs R={rel_unw_r:.3e} vs scipy={rel_unw_py:.3e}; weighted vs R={rel_w_r:.3e}"
    );

    // tol absorbs summation-order round-off in F across BLAS/R/numpy while still
    // catching any genuine sub-optimality of gam relative to the baselines.
    let tol = 1e-9 * f_gam_unw.max(1.0);
    assert!(
        f_gam_unw <= f_r_unw + tol,
        "gam unweighted mean must be at-least-as-good as compositions on the Aitchison objective: F(gam)={f_gam_unw:.6e} > F(R)={f_r_unw:.6e}"
    );
    assert!(
        f_gam_unw <= f_py_unw + tol,
        "gam unweighted mean must be at-least-as-good as scipy.stats.gmean on the Aitchison objective: F(gam)={f_gam_unw:.6e} > F(scipy)={f_py_unw:.6e}"
    );
    let tol_w = 1e-9 * f_gam_w.max(1.0);
    assert!(
        f_gam_w <= f_r_w + tol_w,
        "gam weighted mean must be at-least-as-good as the Aitchison log-space weighted mean baseline: F(gam)={f_gam_w:.6e} > F(R)={f_r_w:.6e}"
    );
}

/// Real-data arm: the Aitchison barycenter is a *predictor of central tendency*
/// for compositions, so on real data (no ground-truth function) its objective
/// quality is its OUT-OF-SAMPLE fit. We split the 23 Skye AFM lavas
/// deterministically into train/test (every 3rd flow held out), estimate the
/// center on TRAIN only with gam, and assert OBJECTIVE held-out quality on gam's
/// OWN output:
///
///   PRIMARY (tool-free):
///     * First-order optimality on TRAIN: gam's center is a stationary point of
///       the training Fréchet objective (||grad F|| at round-off). This is the
///       defining axiom of the Fréchet mean and is intrinsic to gam's result.
///     * Held-out generalization bar: the mean squared Aitchison distance of the
///       HELD-OUT lavas to gam's TRAIN center beats the trivial barycenter (the
///       equal-parts center 1/D, i.e. clr = 0) by a wide margin — gam's center
///       genuinely captures the compositional location of unseen flows.
///
///   BASELINE (match-or-beat): scipy.stats.gmean and R compositions::mean fit the
///     SAME TRAIN rows; gam's HELD-OUT mean squared Aitchison distance must be no
///     worse than the best baseline + round-off tol. The mature tools are
///     baselines on the held-out objective, never an output to replicate.
#[test]
fn frechet_mean_is_the_aitchison_barycenter_on_real_data() {
    // ---- load the real Skye AFM lava compositions --------------------------
    let ds = load_csvwith_inferred_schema(Path::new(SKYE_CSV)).expect("load skye_afm_lavas.csv");
    let col = ds.column_map();
    let a_idx = col["A"];
    let f_idx = col["F"];
    let m_idx = col["M"];
    let a: Vec<f64> = ds.values.column(a_idx).to_vec();
    let f: Vec<f64> = ds.values.column(f_idx).to_vec();
    let m: Vec<f64> = ds.values.column(m_idx).to_vec();
    let n = a.len();
    assert!(n >= 20, "skye AFM should have ~23 flows, got {n}");
    const D: usize = 3;

    // ---- deterministic split: every 3rd flow held out ----------------------
    let is_test = |i: usize| i % 3 == 0;
    let train_rows: Vec<usize> = (0..n).filter(|&i| !is_test(i)).collect();
    let test_rows: Vec<usize> = (0..n).filter(|&i| is_test(i)).collect();
    assert!(
        train_rows.len() >= 12 && test_rows.len() >= 6,
        "split sizes: train={} test={}",
        train_rows.len(),
        test_rows.len()
    );

    // closure-normalized rows (parts -> simplex) for train and test, identical
    // ordering shared with the baselines below.
    let close_row = |i: usize| {
        let s = a[i] + f[i] + m[i];
        [a[i] / s, f[i] / s, m[i] / s]
    };
    let mut train_pts = Array2::<f64>::zeros((train_rows.len(), D));
    for (r, &i) in train_rows.iter().enumerate() {
        let c = close_row(i);
        for k in 0..D {
            train_pts[[r, k]] = c[k];
        }
    }
    let mut test_pts = Array2::<f64>::zeros((test_rows.len(), D));
    for (r, &i) in test_rows.iter().enumerate() {
        let c = close_row(i);
        for k in 0..D {
            test_pts[[r, k]] = c[k];
        }
    }

    // raw (un-closed) train parts as the columns shared with R / Python, so the
    // baselines apply their OWN closure to exactly the same train rows.
    let train_a: Vec<f64> = train_rows.iter().map(|&i| a[i]).collect();
    let train_f: Vec<f64> = train_rows.iter().map(|&i| f[i]).collect();
    let train_m: Vec<f64> = train_rows.iter().map(|&i| m[i]).collect();

    // ---- gam: estimate the (unweighted) barycenter on TRAIN ----------------
    let gam_center = simplex_frechet_mean(train_pts.view(), None).expect("gam train center");
    assert_eq!(gam_center.len(), D);
    let s: f64 = gam_center.iter().sum();
    assert!(
        (s - 1.0).abs() < 1e-12 && gam_center.iter().all(|&v| v > 0.0 && v.is_finite()),
        "gam center must lie on the open simplex: {gam_center:?} (sum={s:.3e})"
    );

    // ---- PRIMARY 1: first-order optimality on TRAIN ------------------------
    let train_w = vec![1.0_f64 / train_rows.len() as f64; train_rows.len()];
    let grad = frechet_gradient_norm(&train_pts, &train_w, &gam_center);
    assert!(
        grad < 1e-10,
        "gam train center is not stationary for the Aitchison Fréchet objective: ||grad F||={grad:.3e}"
    );

    // ---- baseline: scipy.stats.gmean on the SAME train rows ----------------
    let py = run_python(
        &[
            Column::new("A", &train_a),
            Column::new("F", &train_f),
            Column::new("M", &train_m),
        ],
        r#"
from scipy.stats import gmean
X = np.column_stack([df["A"], df["F"], df["M"]])
X = X / X.sum(axis=1, keepdims=True)
g = gmean(X, axis=0)
g = g / g.sum()
emit("py_center", g)
        "#,
    );
    let py_center = py.vector("py_center");
    assert_eq!(py_center.len(), D, "scipy center length");

    // ---- baseline: R compositions::mean(acomp()) on the SAME train rows ----
    let r = run_r(
        &[
            Column::new("A", &train_a),
            Column::new("F", &train_f),
            Column::new("M", &train_m),
        ],
        r#"
        suppressPackageStartupMessages(library(compositions))
        X <- as.matrix(df[, c("A","F","M")])
        ac <- acomp(X)
        mu <- as.numeric(mean(ac))
        mu <- mu / sum(mu)
        emit("r_center", mu)
        "#,
    );
    let r_center = r.vector("r_center");
    assert_eq!(r_center.len(), D, "R center length");

    // ---- OBJECTIVE: held-out mean squared Aitchison distance ---------------
    let gam_heldout = mean_sq_aitchison(&test_pts, &gam_center);
    let py_heldout = mean_sq_aitchison(&test_pts, py_center);
    let r_heldout = mean_sq_aitchison(&test_pts, r_center);
    // Trivial reference: the equal-parts center (clr = 0), i.e. predicting "no
    // compositional signal". The skye lavas are strongly off-center, so a real
    // barycenter must beat this by a wide margin.
    let uniform_center = vec![1.0_f64 / D as f64; D];
    let trivial_heldout = mean_sq_aitchison(&test_pts, &uniform_center);

    // context only: how close the baselines land to gam's center.
    let rel_py = relative_l2(&gam_center, py_center);
    let rel_r = relative_l2(&gam_center, r_center);
    eprintln!(
        "skye AFM held-out mean sq Aitchison dist (lower=better): \
         gam={gam_heldout:.6e} scipy={py_heldout:.6e} R={r_heldout:.6e} trivial={trivial_heldout:.6e}; \
         center rel_l2 vs scipy={rel_py:.3e} vs R={rel_r:.3e} (n_train={} n_test={})",
        train_rows.len(),
        test_rows.len()
    );

    // ---- PRIMARY 2: absolute held-out generalization bar -------------------
    // gam's TRAIN center must explain the held-out flows far better than the
    // equal-parts (no-signal) center.
    assert!(
        gam_heldout < 0.5 * trivial_heldout,
        "gam's held-out mean sq Aitchison distance {gam_heldout:.6e} fails to beat the trivial \
         equal-parts center {trivial_heldout:.6e} by 2x"
    );

    // ---- BASELINE (match-or-beat) on the SAME held-out objective -----------
    let best_baseline = py_heldout.min(r_heldout);
    let tol = 1e-9 * best_baseline.max(1.0);
    assert!(
        gam_heldout <= best_baseline + tol,
        "gam's held-out mean sq Aitchison distance {gam_heldout:.6e} must be no worse than the \
         best mature baseline {best_baseline:.6e} (scipy={py_heldout:.6e}, R={r_heldout:.6e})"
    );
}

/// Stable per-part column header `p0..p3` shared by gam, R, and Python.
fn part_name(col: usize) -> &'static str {
    match col {
        0 => "p0",
        1 => "p1",
        2 => "p2",
        3 => "p3",
        _ => unreachable!("simplex test uses exactly 4 components"),
    }
}
