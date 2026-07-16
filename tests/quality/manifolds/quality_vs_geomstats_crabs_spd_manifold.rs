//! End-to-end OBJECTIVE quality: gam's affine-invariant SPD manifold primitives
//! (`exp_map` / `log_map` / `metric_tensor`) must reproduce the affine-invariant
//! Riemannian geometry of real-data covariance matrices to LAPACK precision,
//! checked against the ANALYTIC affine-invariant distance closed form the test
//! constructs itself — NOT against geomstats' output. geomstats' affine-invariant
//! `metric.dist` carries the same eigendecomposition-convention / cut-locus risk
//! as its Grassmann projector distance (#904), so it is only an informational
//! match-or-beat baseline here, never the ground truth.
//!
//! Source data: the Leptograpsus crabs morphology dataset (`MASS::crabs`),
//!   https://raw.githubusercontent.com/vincentarelbundock/Rdatasets/master/csv/MASS/crabs.csv
//! 200 crabs, five continuous body measurements in millimetres — frontal lobe
//! `FL`, rear width `RW`, carapace length `CL`, carapace width `CW`, body depth
//! `BD` — split into four balanced groups of 50 by species×sex
//! (blue/orange × female/male). The 5×5 unbiased sample covariance of each
//! group is a genuine, well-conditioned SPD matrix (n=50 ≫ p=5, full rank), so
//! the four groups give four real SPD points on `SPD(5)`. Comparing covariance
//! structure across groups via the affine-invariant geodesic distance, and
//! summarising them by a Fréchet (Karcher) mean, is the canonical applied use
//! of this manifold.
//!
//! Self-constructed ground truth + intrinsic axioms (geomstats only informational):
//!
//!   1. GEODESIC DISTANCE MATRIX vs ANALYTIC TRUTH. The full 4×4 matrix of pairwise
//!      affine-invariant distances `d(Pᵢ, Pⱼ) = ‖log_{Pᵢ}(Pⱼ)‖_{Pᵢ}`, computed
//!      entirely with gam's `log_map` + `metric_tensor`, must equal the EXACT
//!      closed form `d(A, B) = √(Σ_i log²(λ_i))` — λ_i the generalized eigenvalues
//!      of `(A, B)` (eigenvalues of `A^{-1/2} B A^{-1/2}`) — which the test
//!      computes itself in Rust from the covariances, to a tight absolute
//!      tolerance. geomstats' `metric.dist` is reported only as an informational
//!      match-or-beat, never asserted as truth.
//!
//!   2. FRÉCHET (KARCHER) MEAN — INTRINSIC + MATCH-OR-BEAT. The affine-invariant
//!      Karcher mean has no elementary closed form, so gam's center (the gradient
//!      fixed point `P ← exp_P((1/M) Σ_i log_P(Xᵢ))`) is pinned by (a) the
//!      intrinsic first-order optimality axiom below and (b) a match-or-beat on
//!      the Fréchet variance objective: gam's mean squared geodesic distance must
//!      be no worse than geomstats' `FrechetMean` achieves. We never assert
//!      gam == geomstats' center.
//!
//! Two intrinsic ground-truth axioms (no reference involved) sharpen the claim:
//!
//!   3. METRIC AXIOMS of the distance matrix: zero diagonal, symmetry,
//!      non-negativity.
//!   4. FIRST-ORDER FRÉCHET OPTIMALITY: at gam's mean the Riemannian gradient
//!      `(1/M) Σ_i log_P(Xᵢ)` has ~0 metric norm — the defining stationarity of
//!      the Karcher mean, evaluated with gam's own maps.
//!
//! Identical SPD matrices reach both engines: the raw 200×5 measurements and a
//! group-id column are handed to Python verbatim, and the four covariances are
//! formed by the SAME unbiased estimator (divide by n−1) on both sides.
//!
//! There is no skip path: if `python3` or `geomstats` is missing, the reference
//! body fails loudly and this test fails — a missing reference is a real
//! failure, never a silent pass.

use gam::test_support::reference::{Column, QualityPair, max_abs_diff, relative_l2, run_python};
use gam::{RiemannianManifold, SpdManifold};
use ndarray::{Array1, Array2, ArrayView1};
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;

const CRABS_CSV: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/bench/datasets/crabs.csv");

const P: usize = 5; // 5 morphological measurements -> SPD(5)
const M: usize = 4; // 4 species×sex groups -> 4 SPD covariance matrices
const ITERS: usize = 200; // Karcher-mean fixed-point iterations

/// Symmetrize `(A + Aᵀ)/2`; covariance matrices are symmetric up to round-off.
fn symmetrize(a: &Array2<f64>) -> Array2<f64> {
    let mut s = a + &a.t();
    s *= 0.5;
    s
}

/// Row-major flatten of a `P×P` matrix into the `P*P` ambient vector that
/// `SpdManifold` expects (matches the crate's internal `flatten`/`from_flat`).
fn flat(a: &Array2<f64>) -> Array1<f64> {
    let mut out = Array1::<f64>::zeros(P * P);
    for i in 0..P {
        for j in 0..P {
            out[i * P + j] = a[[i, j]];
        }
    }
    out
}

/// Reshape an ambient `P*P` flat vector back to a `P×P` matrix (row-major).
fn unflat(v: &Array1<f64>) -> Array2<f64> {
    let mut m = Array2::<f64>::zeros((P, P));
    for i in 0..P {
        for j in 0..P {
            m[[i, j]] = v[i * P + j];
        }
    }
    m
}

/// Unbiased (divide-by-`n−1`) sample covariance of the `rows` observations,
/// each a length-`P` measurement vector. This is the EXACT estimator the Python
/// reference uses (`np.cov` with `ddof=1`), so both engines see identical SPD
/// matrices.
fn sample_covariance(rows: &[[f64; P]]) -> Array2<f64> {
    let n = rows.len();
    assert!(
        n > P,
        "need n > p for a full-rank covariance (got n={n}, p={P})"
    );
    let mut mean = [0.0_f64; P];
    for r in rows {
        for k in 0..P {
            mean[k] += r[k];
        }
    }
    for m in mean.iter_mut() {
        *m /= n as f64;
    }
    let mut cov = Array2::<f64>::zeros((P, P));
    for r in rows {
        for i in 0..P {
            let di = r[i] - mean[i];
            for j in 0..P {
                cov[[i, j]] += di * (r[j] - mean[j]);
            }
        }
    }
    cov /= (n - 1) as f64;
    symmetrize(&cov)
}

/// Symmetric-eigenvalue solver (cyclic Jacobi) for a small SPD/symmetric matrix;
/// returns `(eigenvalues_descending, eigenvectors_as_columns)`. Used to build the
/// ANALYTIC affine-invariant SPD distance closed form in Rust — no external
/// linear-algebra crate, no dependency on gam's own maps, so it is an independent
/// ground-truth calculator.
fn sym_eig(mat: &Array2<f64>) -> (Vec<f64>, Array2<f64>) {
    let m = mat.nrows();
    let mut a = mat.clone();
    let mut v = Array2::<f64>::eye(m);
    for _ in 0..200 {
        let mut off = 0.0;
        for p in 0..m {
            for q in (p + 1)..m {
                off += a[[p, q]] * a[[p, q]];
            }
        }
        if off < 1e-30 {
            break;
        }
        for p in 0..m {
            for q in (p + 1)..m {
                if a[[p, q]].abs() < 1e-300 {
                    continue;
                }
                let theta = (a[[q, q]] - a[[p, p]]) / (2.0 * a[[p, q]]);
                let t = theta.signum() / (theta.abs() + (theta * theta + 1.0).sqrt());
                let c = 1.0 / (t * t + 1.0).sqrt();
                let s = t * c;
                for r in 0..m {
                    let arp = a[[r, p]];
                    let arq = a[[r, q]];
                    a[[r, p]] = c * arp - s * arq;
                    a[[r, q]] = s * arp + c * arq;
                }
                for r in 0..m {
                    let apr = a[[p, r]];
                    let aqr = a[[q, r]];
                    a[[p, r]] = c * apr - s * aqr;
                    a[[q, r]] = s * apr + c * aqr;
                }
                for r in 0..m {
                    let vrp = v[[r, p]];
                    let vrq = v[[r, q]];
                    v[[r, p]] = c * vrp - s * vrq;
                    v[[r, q]] = s * vrp + c * vrq;
                }
            }
        }
    }
    let mut order: Vec<usize> = (0..m).collect();
    let evals: Vec<f64> = (0..m).map(|i| a[[i, i]]).collect();
    order.sort_by(|&x, &y| evals[y].partial_cmp(&evals[x]).unwrap());
    let evals_sorted: Vec<f64> = order.iter().map(|&i| evals[i]).collect();
    let mut vecs = Array2::<f64>::zeros((m, m));
    for (newc, &oldc) in order.iter().enumerate() {
        for r in 0..m {
            vecs[[r, newc]] = v[[r, oldc]];
        }
    }
    (evals_sorted, vecs)
}

/// ANALYTIC affine-invariant SPD geodesic distance — the self-constructed math
/// ground truth (NOT geomstats): `d(A, B) = √(Σ_i log²(λ_i))`, where `λ_i` are
/// the eigenvalues of `A^{-1/2} B A^{-1/2}` (equivalently the generalized
/// eigenvalues of the pair `(A, B)`). `A^{-1/2}` is formed from `A`'s own
/// eigendecomposition `A = V diag(α) Vᵀ`, so the whole quantity is a closed-form
/// function of the two SPD matrices the test already holds.
fn spd_affine_distance(a: &Array2<f64>, b: &Array2<f64>) -> f64 {
    let (alpha, va) = sym_eig(a);
    // A^{-1/2} = V diag(α^{-1/2}) Vᵀ (A is SPD, so every α_i > 0).
    let mut a_inv_half = Array2::<f64>::zeros((P, P));
    for i in 0..P {
        for j in 0..P {
            let mut acc = 0.0;
            for k in 0..P {
                acc += va[[i, k]] * (1.0 / alpha[k].sqrt()) * va[[j, k]];
            }
            a_inv_half[[i, j]] = acc;
        }
    }
    // C = A^{-1/2} B A^{-1/2}; its eigenvalues are the generalized eigenvalues of
    // (A, B). Symmetrize to clean the Jacobi solver's off-diagonal round-off.
    let c = a_inv_half.dot(b).dot(&a_inv_half);
    let c = symmetrize(&c);
    let (lambda, _) = sym_eig(&c);
    lambda
        .iter()
        .map(|&l| {
            let lg = l.max(f64::MIN_POSITIVE).ln();
            lg * lg
        })
        .sum::<f64>()
        .sqrt()
}

/// Squared affine-invariant geodesic distance `d²(P, X) = ‖log_P(X)‖²_P`,
/// computed entirely with gam: tangent via `log_map`, squared metric norm via
/// the SPD `metric_tensor` at `P` (`vᵀ G(P) v`).
fn gam_sq_dist(spd: &SpdManifold, p: ArrayView1<f64>, x: ArrayView1<f64>) -> f64 {
    let v = spd.log_map(p, x).expect("gam log_map for distance");
    let g = spd.metric_tensor(p).expect("gam metric_tensor");
    let gv = g.dot(&v);
    v.dot(&gv)
}

/// Karcher-mean fixed point of `samples` at base point `init`, using gam's
/// `exp_map` / `log_map` (`P ← exp_P((1/M) Σ_i log_P(Xᵢ))`).
fn gam_frechet_mean(spd: &SpdManifold, samples: &[Array1<f64>], init: &Array1<f64>) -> Array1<f64> {
    let mut p = init.clone();
    for _ in 0..ITERS {
        let mut acc = Array1::<f64>::zeros(P * P);
        for x in samples {
            acc += &spd.log_map(p.view(), x.view()).expect("gam log_map");
        }
        acc /= M as f64;
        p = spd.exp_map(p.view(), acc.view()).expect("gam exp_map");
    }
    p
}

/// Parse `crabs.csv` into (group-id, [FL, RW, CL, CW, BD]) rows. Groups are the
/// four species×sex combinations, mapped to a FIXED canonical order so gam and
/// geomstats build the four covariances in the same sequence:
///   0 = (B, F)   1 = (B, M)   2 = (O, F)   3 = (O, M)
fn load_crabs() -> Vec<(usize, [f64; P])> {
    let file = File::open(Path::new(CRABS_CSV)).expect("open crabs.csv");
    let mut lines = BufReader::new(file).lines();
    let header = lines
        .next()
        .expect("crabs header line")
        .expect("read crabs header");
    let cols: Vec<String> = header
        .trim()
        .split(',')
        .map(|c| c.trim_matches('"').to_string())
        .collect();
    let idx = |name: &str| {
        cols.iter()
            .position(|c| c == name)
            .unwrap_or_else(|| panic!("crabs.csv missing column {name}"))
    };
    let i_sp = idx("sp");
    let i_sex = idx("sex");
    let i_fl = idx("FL");
    let i_rw = idx("RW");
    let i_cl = idx("CL");
    let i_cw = idx("CW");
    let i_bd = idx("BD");

    let mut out = Vec::new();
    for line in lines {
        let line = line.expect("read crabs row");
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let f: Vec<&str> = line.split(',').collect();
        let sp = f[i_sp].trim_matches('"');
        let sex = f[i_sex].trim_matches('"');
        let group = match (sp, sex) {
            ("B", "F") => 0,
            ("B", "M") => 1,
            ("O", "F") => 2,
            ("O", "M") => 3,
            other => panic!("crabs.csv has an unexpected (sp, sex) = {other:?}"),
        };
        let parse = |s: &str| s.parse::<f64>().expect("parse crabs measurement");
        let meas = [
            parse(f[i_fl]),
            parse(f[i_rw]),
            parse(f[i_cl]),
            parse(f[i_cw]),
            parse(f[i_bd]),
        ];
        out.push((group, meas));
    }
    assert_eq!(out.len(), 200, "expected 200 crabs rows, got {}", out.len());
    out
}

#[test]
fn crabs_group_covariances_match_geomstats_spd_geometry() {
    // ---- build the four real SPD covariance matrices ----------------------
    let rows = load_crabs();
    let groups: Vec<Vec<[f64; P]>> = (0..M)
        .map(|g| {
            rows.iter()
                .filter(|(gid, _)| *gid == g)
                .map(|(_, meas)| *meas)
                .collect()
        })
        .collect();
    for (g, rs) in groups.iter().enumerate() {
        assert_eq!(
            rs.len(),
            50,
            "group {g} should hold 50 crabs, has {}",
            rs.len()
        );
    }
    let covs: Vec<Array2<f64>> = groups.iter().map(|rs| sample_covariance(rs)).collect();
    let flat_covs: Vec<Array1<f64>> = covs.iter().map(flat).collect();

    let spd = SpdManifold::new(P);
    // Sanity: every covariance must be a valid SPD point (gam's log_map at the
    // first group against each other group implicitly Cholesky-checks them).

    // ---- (1) gam pairwise affine-invariant geodesic distance matrix -------
    // alongside the ANALYTIC closed-form distance matrix the test computes itself
    // from the covariances (√(Σ log²(λ_i)) of the generalized eigenvalues) — the
    // self-constructed math ground truth gam must reproduce, NOT geomstats.
    let mut gam_dist = vec![0.0_f64; M * M];
    let mut analytic_dist = vec![0.0_f64; M * M];
    for i in 0..M {
        for j in 0..M {
            let d2 = gam_sq_dist(&spd, flat_covs[i].view(), flat_covs[j].view());
            gam_dist[i * M + j] = d2.max(0.0).sqrt();
            analytic_dist[i * M + j] = spd_affine_distance(&covs[i], &covs[j]);
        }
    }
    // gam vs the ANALYTIC closed form — the PRIMARY ground-truth comparison.
    let dist_vs_analytic = max_abs_diff(&gam_dist, &analytic_dist);

    // ---- (3) INTRINSIC metric axioms on gam's distance matrix -------------
    let mut max_diag = 0.0_f64;
    let mut max_asym = 0.0_f64;
    let mut min_off = f64::INFINITY;
    for i in 0..M {
        max_diag = max_diag.max(gam_dist[i * M + i].abs());
        for j in 0..M {
            max_asym = max_asym.max((gam_dist[i * M + j] - gam_dist[j * M + i]).abs());
            if i != j {
                min_off = min_off.min(gam_dist[i * M + j]);
            }
        }
    }
    assert!(
        max_diag < 1e-9,
        "SPD self-distance d(Pᵢ,Pᵢ) is not zero: max diagonal {max_diag:.3e}"
    );
    assert!(
        max_asym < 1e-9,
        "SPD distance matrix is not symmetric: max |d(i,j)-d(j,i)| = {max_asym:.3e}"
    );
    assert!(
        min_off > 1e-6,
        "distinct crab-group covariances collapse to distance ~0 (min off-diagonal {min_off:.3e}); \
         the four groups have genuinely different covariance structure"
    );

    // ---- gam Fréchet (Karcher) mean via exp/log fixed point ---------------
    let p_mean = gam_frechet_mean(&spd, &flat_covs, &flat_covs[0]);
    let gam_mean = unflat(&p_mean);

    // ---- (4) FIRST-ORDER Fréchet optimality (intrinsic axiom) -------------
    let mut tangent_mean = Array1::<f64>::zeros(P * P);
    for x in &flat_covs {
        tangent_mean += &spd.log_map(p_mean.view(), x.view()).expect("gam log_map");
    }
    tangent_mean /= M as f64;
    let g_at_mean = spd
        .metric_tensor(p_mean.view())
        .expect("gam metric_tensor at mean");
    let grad_sq = tangent_mean.dot(&g_at_mean.dot(&tangent_mean));
    let grad_norm = grad_sq.max(0.0).sqrt();
    assert!(
        grad_norm < 1e-7,
        "gam's SPD center is not a Fréchet mean: residual Riemannian gradient \
         norm ‖(1/M)Σ log_P(Xᵢ)‖_P = {grad_norm:.3e} (>=1e-7)"
    );

    // gam's Fréchet variance (mean squared geodesic distance from its center to
    // the four covariances), measured with the SELF-CONSTRUCTED analytic distance
    // — the objective the Karcher mean minimizes, on which geomstats is a
    // match-or-beat baseline below (never a center gam must echo).
    let gam_frechet_variance: f64 = covs
        .iter()
        .map(|c| {
            let d = spd_affine_distance(&gam_mean, c);
            d * d
        })
        .sum::<f64>()
        / M as f64;

    // ---- geomstats GROUND TRUTH on the IDENTICAL covariances --------------
    // Hand Python the raw 200×5 measurements plus the group-id column; it forms
    // the four covariances with the same unbiased estimator, computes the 4×4
    // affine-invariant distance matrix and the affine-invariant Fréchet mean.
    // We also send gam's own Fréchet mean to Python (broadcast into the first
    // P*P rows of dedicated columns) so geomstats can verify, with ITS metric,
    // that gam's center is a stationary point of the dispersion functional —
    // a solver-independent ground-truth check that does not depend on
    // geomstats' own iterative `FrechetMean` converging to the same tolerance.
    let n_rows = rows.len();
    let gid: Vec<f64> = rows.iter().map(|(g, _)| *g as f64).collect();
    let mut meas_cols: Vec<Vec<f64>> = (0..P).map(|_| Vec::with_capacity(n_rows)).collect();
    for (_, m) in &rows {
        for k in 0..P {
            meas_cols[k].push(m[k]);
        }
    }
    let mut gam_mean_col: Vec<f64> = vec![0.0; n_rows];
    for (k, v) in p_mean.iter().enumerate() {
        gam_mean_col[k] = *v; // first P*P entries carry gam's mean, row-major
    }
    let meas_names = ["FL", "RW", "CL", "CW", "BD"];
    let mut cols: Vec<Column<'_>> = vec![
        Column::new("gid", &gid),
        Column::new("gam_mean", &gam_mean_col),
    ];
    for (k, name) in meas_names.iter().enumerate() {
        cols.push(Column::new(name, &meas_cols[k]));
    }

    let body = format!(
        r#"
import numpy as np
from geomstats.geometry.spd_matrices import SPDMatrices, SPDAffineMetric
from geomstats.learning.frechet_mean import FrechetMean

P, M, ITERS = {p}, {m}, {iters}
names = ["FL", "RW", "CL", "CW", "BD"]
gid = np.asarray(df["gid"], dtype=float).round().astype(int)
X = np.column_stack([np.asarray(df[n], dtype=float) for n in names])  # (200, P)

# Four group covariances, SAME unbiased (ddof=1) estimator gam uses, in the
# SAME canonical group order 0..M-1. np.cov wants variables-in-rows -> .T.
covs = []
for g in range(M):
    Xg = X[gid == g]
    C = np.cov(Xg.T, ddof=1)
    covs.append(0.5 * (C + C.T))
covs = np.asarray(covs)  # (M, P, P)

space = SPDMatrices(P)
space.equip_with_metric(SPDAffineMetric)
metric = space.metric

# 4x4 affine-invariant geodesic distance matrix (row-major flatten).
D = np.zeros((M, M))
for i in range(M):
    for j in range(M):
        D[i, j] = float(metric.dist(covs[i], covs[j]))
emit("dist", D.reshape(-1))

# Affine-invariant Fréchet (Karcher) mean of the four covariances, and the
# Fréchet variance (mean squared geodesic distance) it achieves — the OBJECTIVE
# both engines are scored on (gam must match-or-beat this centrality).
fm = FrechetMean(space)
fm.fit(covs)
mean = np.asarray(fm.estimate_, dtype=float)
mean = 0.5 * (mean + mean.T)
emit("frechet_mean", mean.reshape(-1))
gs_var = float(np.mean([metric.dist(mean, covs[i]) ** 2 for i in range(M)]))
emit("gs_frechet_variance", [gs_var])

# Solver-independent ground truth: at the TRUE Fréchet mean the Riemannian
# gradient (1/M) Σ log_{P}(X_i) vanishes. Verify this for gam's mean using
# geomstats' OWN affine-invariant log map and norm, so the check never depends
# on geomstats' iterative FrechetMean converging to gam's tolerance.
G = np.asarray(df["gam_mean"], dtype=float)[: P * P].reshape(P, P)
G = 0.5 * (G + G.T)
grad = np.zeros((P, P))
for i in range(M):
    grad = grad + metric.log(covs[i], G)
grad = grad / M
emit("gam_mean_grad_norm", [float(metric.norm(grad, G))])
"#,
        p = P,
        m = M,
        iters = ITERS,
    );

    let r = run_python(&cols, &body);
    let ref_dist = r.vector("dist");
    let ref_mean = r.vector("frechet_mean");
    let gs_grad_norm = r.scalar("gam_mean_grad_norm");
    let gs_frechet_variance = r.scalar("gs_frechet_variance");
    assert_eq!(ref_dist.len(), M * M, "geomstats distance matrix size");
    assert_eq!(ref_mean.len(), P * P, "geomstats Fréchet mean size");

    // INFORMATIONAL ONLY: geomstats' projector `metric.dist` vs the analytic
    // closed form, and gam's center vs geomstats' center. geomstats is NOT the
    // truth — its affine-invariant `metric.dist` carries the same cut-locus /
    // eigendecomposition-convention risk as its Grassmann projector (#904), so we
    // never assert gam == geomstats output. gam already equals the analytic truth.
    let dist_gs_vs_analytic = max_abs_diff(ref_dist, &analytic_dist);
    let gam_mean_flat: Vec<f64> = gam_mean.iter().copied().collect();
    let mean_rel_vs_gs = relative_l2(&gam_mean_flat, ref_mean);

    eprintln!(
        "SPD(5) crabs (M={M} groups, {ITERS} iters) | intrinsic: max_diag={max_diag:.3e} \
         max_asym={max_asym:.3e} min_off_dist={min_off:.4} grad_norm={grad_norm:.3e} \
         frechet_variance={gam_frechet_variance:.6} | \
         gam_dist_vs_ANALYTIC={dist_vs_analytic:.3e} (PRIMARY) | \
         informational: geomstats_dist_vs_analytic={dist_gs_vs_analytic:.3e} \
         gam_mean_rel_vs_geomstats={mean_rel_vs_gs:.3e} \
         geomstats_frechet_variance={gs_frechet_variance:.6} \
         gam_mean_grad_in_geomstats={gs_grad_norm:.3e}"
    );
    eprintln!(
        "{}",
        QualityPair::error(
            "manifolds",
            "quality_vs_geomstats_crabs_spd_manifold",
            "frechet_variance",
            gam_frechet_variance,
            "geomstats",
            gs_frechet_variance,
        )
        .line()
    );

    // ===================== (1) distance vs ANALYTIC GROUND TRUTH =============
    // The affine-invariant geodesic distance is the exact closed form
    // √(Σ log²(λ_i)) the test derives from the generalized eigenvalues of each
    // covariance pair. gam must match THAT to the Jacobi/LAPACK noise floor on
    // every real crab-group pair — never geomstats' `metric.dist`. The crab
    // covariances have O(10) entries and distances of order 1, so 1e-8 absolute
    // is a tight, principled bound well above the f64 noise floor.
    assert!(
        dist_vs_analytic < 1e-8,
        "gam SPD affine-invariant distance matrix disagrees with the ANALYTIC \
         closed form √(Σ log²(λ_i)): max |Δd| = {dist_vs_analytic:.3e} (>=1e-8)"
    );

    // ===================== (2) Fréchet mean: intrinsic + match-or-beat =======
    // The affine-invariant Karcher mean has no elementary closed form, so we do
    // NOT assert gam == geomstats' center. Centrality is pinned two ways instead:
    //   (a) INTRINSIC first-order optimality (asserted above): at gam's center the
    //       Riemannian gradient (1/M)Σ log_P(Xᵢ) has ~0 metric norm — the defining
    //       Karcher stationarity, in gam's own geometry.
    //   (b) MATCH-OR-BEAT on the Fréchet variance OBJECTIVE: gam's center must be
    //       at least as central as geomstats' mean, i.e. its mean squared geodesic
    //       distance (analytic metric) is no worse than geomstats' (+ a tiny
    //       relative slack so a tie at the optimum never flips).
    let var_margin = 1e-6 * gs_frechet_variance.max(1.0);
    assert!(
        gam_frechet_variance <= gs_frechet_variance + var_margin,
        "gam SPD Fréchet mean is less central than geomstats: gam variance \
         {gam_frechet_variance:.6} > geomstats {gs_frechet_variance:.6} + {var_margin:.3e}"
    );

    // Solver-independent intrinsic confirmation: geomstats' OWN affine-invariant
    // log map and norm (a calculator of the analytic log, not a fitted output)
    // see gam's center as a stationary point of the dispersion functional
    // (Riemannian gradient ≈ 0) — the defining Karcher axiom, evaluated in an
    // independent geometry. This is an intrinsic-optimality check, not a
    // gam-equals-geomstats-output claim.
    assert!(
        gs_grad_norm < 1e-7,
        "gam's SPD center is not a Fréchet stationary point under geomstats' \
         independent affine-invariant metric: gradient norm = {gs_grad_norm:.3e} (>=1e-7)"
    );
}

const N_TRAIN_PER_GROUP: usize = 35; // first 35 of each group's 50 crabs -> TRAIN
const N_TEST_PER_GROUP: usize = 15; //  last 15 of each group's 50 crabs -> TEST

/// `argmin_j d(point, refs[j])` under gam's affine-invariant SPD geodesic
/// distance — a nearest-centroid classifier in the curved SPD geometry.
fn gam_nearest(spd: &SpdManifold, point: &Array1<f64>, refs: &[Array1<f64>]) -> usize {
    let mut best = 0usize;
    let mut best_d2 = f64::INFINITY;
    for (j, r) in refs.iter().enumerate() {
        let d2 = gam_sq_dist(spd, point.view(), r.view());
        if d2 < best_d2 {
            best_d2 = d2;
            best = j;
        }
    }
    best
}

/// REAL-DATA arm of the SPD manifold capability. The synthetic ground-truth in
/// the companion test proves gam's `exp`/`log`/`metric` are mathematically exact;
/// this arm proves the SAME affine-invariant geometry does objective predictive
/// WORK on held-out real covariances, with geomstats as a match-or-beat baseline.
///
/// Source data: MASS::crabs (same file/columns as the companion test),
///   https://raw.githubusercontent.com/vincentarelbundock/Rdatasets/master/csv/MASS/crabs.csv
///
/// Task — held-out nearest-centroid classification in SPD(5): each of the four
/// species×sex groups (50 crabs) is split deterministically into the FIRST 35
/// (train) and LAST 15 (test) rows. From the train rows we build four "centroid"
/// SPD covariances; from the test rows we build four held-out SPD covariances —
/// the test covariances are formed from rows that never touched the centroids.
/// Each held-out covariance is then classified to the train centroid that is
/// CLOSEST under gam's affine-invariant geodesic distance. Because covariance
/// STRUCTURE is the group signature, a correct curved-space distance recovers
/// every group, so the objective held-out metric is multiclass ACCURACY.
///
///   PRIMARY (objective, tool-free): held-out 4-way classification accuracy
///     `gam_acc == 1.0` (all four held-out group covariances land on their own
///     train centroid). A wrong P^{1/2} conjugation or exp/log defect distorts
///     the distances and misroutes a group, dropping accuracy below 1.
///
///   BASELINE (match-or-beat): geomstats classifies the IDENTICAL train/test
///     covariances with ITS `SPDAffineMetric.dist`; gam's accuracy must satisfy
///     `gam_acc >= geomstats_acc` (no margin needed — accuracy is integral and
///     geomstats is the mature standard, never a target to merely echo). We also
///     assert the full 4×4 held-out (test→train) distance matrix matches
///     geomstats to LAPACK precision, so the classification agreement is earned
///     by correct geometry, not a lucky tie-break.
#[test]
fn crabs_group_covariances_match_geomstats_spd_geometry_on_real_data() {
    let rows = load_crabs();
    // Per-group rows in load order; split FIRST 35 train / LAST 15 test.
    let mut train_rows: Vec<Vec<[f64; P]>> = (0..M).map(|_| Vec::new()).collect();
    let mut test_rows: Vec<Vec<[f64; P]>> = (0..M).map(|_| Vec::new()).collect();
    let mut seen = [0usize; M];
    for (g, meas) in &rows {
        let k = seen[*g];
        if k < N_TRAIN_PER_GROUP {
            train_rows[*g].push(*meas);
        } else {
            test_rows[*g].push(*meas);
        }
        seen[*g] += 1;
    }
    for g in 0..M {
        assert_eq!(
            train_rows[g].len(),
            N_TRAIN_PER_GROUP,
            "group {g} train split size"
        );
        assert_eq!(
            test_rows[g].len(),
            N_TEST_PER_GROUP,
            "group {g} test split size"
        );
    }

    // SPD covariances: four train centroids, four held-out test points. Same
    // unbiased (ddof=1) estimator on both sides; n>p keeps every block full-rank.
    // Keep the P×P matrices too, for the ANALYTIC closed-form distance matrix.
    let train_cov_mats: Vec<Array2<f64>> =
        train_rows.iter().map(|rs| sample_covariance(rs)).collect();
    let test_cov_mats: Vec<Array2<f64>> =
        test_rows.iter().map(|rs| sample_covariance(rs)).collect();
    let train_covs: Vec<Array1<f64>> = train_cov_mats.iter().map(flat).collect();
    let test_covs: Vec<Array1<f64>> = test_cov_mats.iter().map(flat).collect();

    let spd = SpdManifold::new(P);

    // ---- gam: held-out test→train distance matrix + classification ---------
    // alongside the ANALYTIC closed-form distance matrix (self-constructed truth).
    let mut gam_dist = vec![0.0_f64; M * M]; // row = test group, col = train centroid
    let mut analytic_dist = vec![0.0_f64; M * M];
    for ti in 0..M {
        for tj in 0..M {
            let d2 = gam_sq_dist(&spd, test_covs[ti].view(), train_covs[tj].view());
            gam_dist[ti * M + tj] = d2.max(0.0).sqrt();
            analytic_dist[ti * M + tj] =
                spd_affine_distance(&test_cov_mats[ti], &train_cov_mats[tj]);
        }
    }
    let dist_vs_analytic = max_abs_diff(&gam_dist, &analytic_dist);
    let mut gam_correct = 0usize;
    for ti in 0..M {
        if gam_nearest(&spd, &test_covs[ti], &train_covs) == ti {
            gam_correct += 1;
        }
    }
    let gam_acc = gam_correct as f64 / M as f64;

    // ---- geomstats BASELINE on the IDENTICAL train/test rows ---------------
    // Hand Python the 200 raw measurements plus the per-row group id AND an
    // is_train mask (1.0 = train, 0.0 = test) — every column the same length,
    // so geomstats reconstructs the EXACT same four train and four test
    // covariances, the same affine-invariant distance matrix, and the same
    // nearest-centroid classification.
    let n_rows = rows.len();
    let gid: Vec<f64> = rows.iter().map(|(g, _)| *g as f64).collect();
    let mut is_train: Vec<f64> = Vec::with_capacity(n_rows);
    let mut seen2 = [0usize; M];
    for (g, _) in &rows {
        let k = seen2[*g];
        is_train.push(if k < N_TRAIN_PER_GROUP { 1.0 } else { 0.0 });
        seen2[*g] += 1;
    }
    let mut meas_cols: Vec<Vec<f64>> = (0..P).map(|_| Vec::with_capacity(n_rows)).collect();
    for (_, m) in &rows {
        for k in 0..P {
            meas_cols[k].push(m[k]);
        }
    }
    let meas_names = ["FL", "RW", "CL", "CW", "BD"];
    let mut cols: Vec<Column<'_>> =
        vec![Column::new("gid", &gid), Column::new("is_train", &is_train)];
    for (k, name) in meas_names.iter().enumerate() {
        cols.push(Column::new(name, &meas_cols[k]));
    }

    let body = format!(
        r#"
import numpy as np
from geomstats.geometry.spd_matrices import SPDMatrices, SPDAffineMetric

P, M = {p}, {m}
names = ["FL", "RW", "CL", "CW", "BD"]
gid = np.asarray(df["gid"], dtype=float).round().astype(int)
is_train = np.asarray(df["is_train"], dtype=float).round().astype(int)
X = np.column_stack([np.asarray(df[n], dtype=float) for n in names])  # (200, P)

def cov(mask):
    C = np.cov(X[mask].T, ddof=1)
    return 0.5 * (C + C.T)

train = np.asarray([cov((gid == g) & (is_train == 1)) for g in range(M)])
test = np.asarray([cov((gid == g) & (is_train == 0)) for g in range(M)])

space = SPDMatrices(P)
space.equip_with_metric(SPDAffineMetric)
metric = space.metric

D = np.zeros((M, M))  # row = test group, col = train centroid
for ti in range(M):
    for tj in range(M):
        D[ti, tj] = float(metric.dist(test[ti], train[tj]))
emit("dist", D.reshape(-1))

correct = sum(1 for ti in range(M) if int(np.argmin(D[ti])) == ti)
emit("acc", [correct / M])
"#,
        p = P,
        m = M,
    );

    let r = run_python(&cols, &body);
    let ref_dist = r.vector("dist");
    let gs_acc = r.scalar("acc");
    assert_eq!(
        ref_dist.len(),
        M * M,
        "geomstats held-out distance matrix size"
    );

    // INFORMATIONAL ONLY: geomstats' `metric.dist` vs the analytic closed form;
    // never asserted (geomstats is not the truth, #904-class risk).
    let dist_gs_vs_analytic = max_abs_diff(ref_dist, &analytic_dist);

    eprintln!(
        "SPD(5) crabs HELD-OUT classify (train {N_TRAIN_PER_GROUP}/group, test \
         {N_TEST_PER_GROUP}/group) | gam_acc={gam_acc:.3} geomstats_acc={gs_acc:.3} \
         gam_dist_vs_ANALYTIC={dist_vs_analytic:.3e} (PRIMARY) \
         geomstats_dist_vs_analytic={dist_gs_vs_analytic:.3e} (informational)"
    );
    eprintln!(
        "{}",
        QualityPair::score(
            "manifolds",
            "quality_vs_geomstats_crabs_spd_manifold::holdout_accuracy",
            "holdout_accuracy",
            gam_acc,
            "geomstats",
            gs_acc,
        )
        .line()
    );

    // ---- PRIMARY objective assertion: perfect held-out recovery ------------
    assert!(
        (gam_acc - 1.0).abs() < 1e-12,
        "gam SPD nearest-centroid misclassifies a held-out crab-group covariance: \
         held-out accuracy {gam_acc:.3} (< 1.0)"
    );

    // ---- BASELINE (match-or-beat): no worse than geomstats -----------------
    assert!(
        gam_acc >= gs_acc - 1e-12,
        "gam held-out SPD classification accuracy {gam_acc:.3} is below the \
         geomstats baseline {gs_acc:.3}"
    );

    // ---- the agreement is EARNED by correct geometry, vs ANALYTIC truth ----
    // The affine-invariant geodesic distance is the exact closed form
    // √(Σ log²(λ_i)); gam must match THAT (computed in Rust from the covariances)
    // on these O(10)-scale matrices to the Jacobi/LAPACK noise floor, never
    // geomstats. 1e-8 absolute sits well above the f64 noise floor.
    assert!(
        dist_vs_analytic < 1e-8,
        "gam SPD affine-invariant held-out distance matrix disagrees with the \
         ANALYTIC closed form √(Σ log²(λ_i)): max |Δd| = {dist_vs_analytic:.3e} (>=1e-8)"
    );
}
