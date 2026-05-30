//! End-to-end OBJECTIVE quality: gam's affine-invariant SPD manifold primitives
//! (`exp_map` / `log_map` / `metric_tensor`) must reproduce the affine-invariant
//! Riemannian geometry of real-data covariance matrices to LAPACK precision,
//! checked against Python **geomstats** as MATHEMATICAL GROUND TRUTH.
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
//! Ground-truth comparators (geomstats, the affine-invariant SPD geodesic is an
//! exact closed form — geomstats is independent mathematical truth, not a noisy
//! peer fit):
//!
//!   1. GEODESIC DISTANCE MATRIX. The full 4×4 matrix of pairwise
//!      affine-invariant distances `d(Pᵢ, Pⱼ) = ‖log_{Pᵢ}(Pⱼ)‖_{Pᵢ}`, computed
//!      entirely with gam's `log_map` + `metric_tensor`, must match
//!      `SPDMatrices(5)` under `SPDAffineMetric` (`metric.dist`) to a tight
//!      absolute tolerance.
//!
//!   2. FRÉCHET (KARCHER) MEAN. The affine-invariant center of mass of the four
//!      covariances — computed via gam's canonical gradient fixed point
//!      `P ← exp_P((1/M) Σ_i log_P(Xᵢ))` — must match geomstats'
//!      `FrechetMean(SPDMatrices(5), SPDAffineMetric).fit(...).estimate_` to a
//!      tight relative tolerance.
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

use gam::test_support::reference::{Column, max_abs_diff, relative_l2, run_python};
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
    assert!(n > P, "need n > p for a full-rank covariance (got n={n}, p={P})");
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
        assert_eq!(rs.len(), 50, "group {g} should hold 50 crabs, has {}", rs.len());
    }
    let covs: Vec<Array2<f64>> = groups.iter().map(|rs| sample_covariance(rs)).collect();
    let flat_covs: Vec<Array1<f64>> = covs.iter().map(flat).collect();

    let spd = SpdManifold::new(P);
    // Sanity: every covariance must be a valid SPD point (gam's log_map at the
    // first group against each other group implicitly Cholesky-checks them).

    // ---- (1) gam pairwise affine-invariant geodesic distance matrix -------
    let mut gam_dist = vec![0.0_f64; M * M];
    for i in 0..M {
        for j in 0..M {
            let d2 = gam_sq_dist(&spd, flat_covs[i].view(), flat_covs[j].view());
            gam_dist[i * M + j] = d2.max(0.0).sqrt();
        }
    }

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
    let g_at_mean = spd.metric_tensor(p_mean.view()).expect("gam metric_tensor at mean");
    let grad_sq = tangent_mean.dot(&g_at_mean.dot(&tangent_mean));
    let grad_norm = grad_sq.max(0.0).sqrt();
    assert!(
        grad_norm < 1e-7,
        "gam's SPD center is not a Fréchet mean: residual Riemannian gradient \
         norm ‖(1/M)Σ log_P(Xᵢ)‖_P = {grad_norm:.3e} (>=1e-7)"
    );

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

# Affine-invariant Fréchet (Karcher) mean of the four covariances.
fm = FrechetMean(space)
fm.fit(covs)
mean = np.asarray(fm.estimate_, dtype=float)
mean = 0.5 * (mean + mean.T)
emit("frechet_mean", mean.reshape(-1))

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
    assert_eq!(ref_dist.len(), M * M, "geomstats distance matrix size");
    assert_eq!(ref_mean.len(), P * P, "geomstats Fréchet mean size");

    // ===================== (1) distance matrix vs geomstats =================
    let dist_err = max_abs_diff(&gam_dist, ref_dist);
    // ===================== (2) Fréchet mean vs geomstats ====================
    let gam_mean_flat: Vec<f64> = gam_mean.iter().copied().collect();
    let mean_rel = relative_l2(&gam_mean_flat, ref_mean);
    let mean_abs = max_abs_diff(&gam_mean_flat, ref_mean);

    eprintln!(
        "SPD(5) crabs (M={M} groups, {ITERS} iters) | intrinsic: max_diag={max_diag:.3e} \
         max_asym={max_asym:.3e} min_off_dist={min_off:.4} grad_norm={grad_norm:.3e} | \
         vs geomstats: dist_matrix_max_abs={dist_err:.3e} frechet_rel_l2={mean_rel:.3e} \
         frechet_max_abs={mean_abs:.3e} gam_mean_grad_in_geomstats={gs_grad_norm:.3e}"
    );

    // The affine-invariant geodesic distance is a closed form; gam must match
    // geomstats' eigendecomposition-based value to LAPACK precision. The crab
    // covariances have O(10) entries and distances of order 1, so 1e-8 absolute
    // is a tight, principled bound well above the f64 noise floor.
    assert!(
        dist_err < 1e-8,
        "gam SPD affine-invariant distance matrix disagrees with geomstats ground \
         truth: max |Δd| = {dist_err:.3e} (>=1e-8)"
    );

    // The Fréchet mean is the unique minimizer of an analytic functional; both
    // engines converge to the same SPD point. 1e-6 relative leaves margin for
    // the differing iteration schemes (gam's fixed point vs geomstats' solver
    // stopping tolerance) while still catching any wrong P^{1/2} conjugation or
    // exp/log defect — such a defect would shift the center by O(1), not 1e-6.
    assert!(
        mean_rel < 1e-6,
        "gam SPD Fréchet mean disagrees with geomstats ground truth: \
         rel_l2 = {mean_rel:.3e} (max_abs = {mean_abs:.3e}) (>=1e-6)"
    );

    // Sharper, solver-independent confirmation: geomstats' OWN affine-invariant
    // log map and norm see gam's center as a stationary point of the dispersion
    // functional (Riemannian gradient ≈ 0). This does not depend on geomstats'
    // iterative FrechetMean converging to gam's tolerance — it is the defining
    // Karcher axiom evaluated entirely with the reference's geometry.
    assert!(
        gs_grad_norm < 1e-7,
        "geomstats does not see gam's SPD center as a Fréchet mean: Riemannian \
         gradient norm in geomstats' affine-invariant metric = {gs_grad_norm:.3e} (>=1e-7)"
    );
}
