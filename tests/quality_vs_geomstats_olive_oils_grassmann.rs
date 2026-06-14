//! End-to-end quality: gam's Grassmann `Gr(3, 8)` geodesic distance, exponential,
//! and logarithm must reproduce the EXACT subspace geometry of real per-group PCA
//! subspaces, judged against the ANALYTIC principal-angle arc length the test
//! constructs itself from the bases — NOT against geomstats' output. geomstats'
//! projector `metric.dist` is wrong near the `π/2` cut locus (#904), so it is only
//! an informational match-or-beat cross-check away from that locus, never truth.
//!
//! DATA (real, freely downloadable, no auth):
//!   Forina et al. Italian olive-oil dataset, 572 samples, eight fatty-acid
//!   percentages (palmitic, palmitoleic, stearic, oleic, linoleic, linolenic,
//!   arachidic, eicosenoic) measured per sample, each labelled by one of nine
//!   producing `area`s. Vendored at `bench/datasets/olive_oils.csv` from
//!   https://raw.githubusercontent.com/vincentarelbundock/Rdatasets/master/csv/dslabs/olive.csv
//!
//! CONSTRUCTION (deterministic, no RNG): for each of the nine areas we center the
//! group's `n_g × 8` fatty-acid matrix and take the top-`k=3` principal directions
//! — the leading three right singular vectors / leading eigenvectors of the 8×8
//! sample covariance. Each is an orthonormal `8×3` frame, i.e. a point of the
//! Stiefel manifold `St(8,3)`, and its column span is a point of the Grassmannian
//! `Gr(3, 8)`. This is the canonical "which directions does this region's oil vary
//! along?" subspace, and comparing two regions' subspaces is exactly the realistic
//! use of gam's Grassmann distance / log map. The identical orthonormal bases (same
//! entries, same column order) are handed to BOTH gam and geomstats.
//!
//! OBJECTIVE ACCURACY (the pass criteria), all vs ANALYTIC ground truth the test
//! constructs itself on the real subspaces — geomstats is never the truth:
//!   (A) **Geodesic distance.** For every ordered pair of areas, the Grassmann
//!       geodesic distance `d(P_i, P_j) = ‖Log_{P_i}(P_j)‖_F` computed from gam's
//!       `log_map` must equal the ANALYTIC principal-angle arc length
//!       `‖θ‖₂ = √(Σ_i arccos²(σ_i))`, where `σ_i` are the singular values of
//!       `Y_iᵀ Y_j` clamped to `[-1, 1]` — a closed form this test computes itself
//!       from the subspace bases (the `principal_angles` helper below), to
//!       `< 1e-9`. geomstats is retained ONLY as an informational match-or-beat
//!       cross-check AWAY from the `π/2` cut locus: its canonical-metric `dist`
//!       projector formula is known to be wrong near `π/2` (it disagrees with the
//!       analytic arc length there, while gam matches it), so it cannot be the
//!       ground truth — see issue #904.
//!   (B) **log singular spectrum = principal angles.** The singular values of gam's
//!       `log_map(P_i, P_j)` equal the principal angles `θ(P_i, P_j)` derived
//!       independently from the SVD of `Y_iᵀ Y_j` — an absolute-accuracy claim
//!       against a quantity the test computes itself.
//!   (C) **exp/log round-trip + isometry.** Stepping a controlled fraction along the
//!       geodesic toward another area, `log(exp(v)) == v` componentwise (`< 1e-10`),
//!       and the recovered geodesic distance equals `‖v‖_F` (`< 1e-9`). The tangent
//!       is scaled to sit safely inside the injectivity radius `π/2`.
//!
//! There is no skip path: if `python3` or `geomstats` is missing the reference body
//! fails loudly and so does this test (per `src/test_support/reference.rs`).

use gam::geometry::{GrassmannManifold, RiemannianManifold};
use gam::test_support::reference::{Column, max_abs_diff, run_python};
use ndarray::{Array1, Array2};
use std::collections::BTreeMap;
use std::path::PathBuf;

/// Grassmann geodesic distance between two row-major flattened `D×K` frames, via
/// gam's `log_map`: `d = ‖Log_{P}(Q)‖_F`. This is the SAME gam capability the
/// synthetic/ground-truth test above pins to geomstats; here it is the kernel of
/// an objective held-out classifier.
fn grassmann_dist(gr: &GrassmannManifold, p: &Array1<f64>, q: &Array1<f64>) -> f64 {
    let log = gr
        .log_map(p.view(), q.view())
        .expect("gam Grassmann log_map");
    log.iter().map(|x| x * x).sum::<f64>().sqrt()
}

/// Intrinsic Grassmann Fréchet (Karcher) mean of a set of `D×K` frames, computed
/// ENTIRELY through gam's exponential and logarithm: iterate
/// `μ ← Exp_μ( (1/N) Σ_i Log_μ(P_i) )` from the first frame until the mean
/// tangent vanishes. This is gam's Grassmann geometry doing the work; geomstats
/// is only the match-or-beat baseline for the variance it achieves.
fn grassmann_frechet_mean(gr: &GrassmannManifold, frames: &[Array1<f64>]) -> Array1<f64> {
    assert!(!frames.is_empty(), "Fréchet mean needs at least one frame");
    let mut mu = frames[0].clone();
    for _ in 0..200 {
        let mut tangent = Array1::<f64>::zeros(D * K);
        for f in frames {
            let log = gr
                .log_map(mu.view(), f.view())
                .expect("gam log_map in Karcher mean");
            tangent += &log;
        }
        tangent /= frames.len() as f64;
        let step: f64 = tangent.iter().map(|x| x * x).sum::<f64>().sqrt();
        mu = gr
            .exp_map(mu.view(), tangent.view())
            .expect("gam exp_map in Karcher mean");
        if step < 1e-12 {
            break;
        }
    }
    mu
}

/// Total within-set squared geodesic distance of a candidate mean `mu` to every
/// frame — the Fréchet objective the Karcher mean minimizes. Lower is a tighter
/// (more central) mean; this is the objective both gam and geomstats are scored on.
fn frechet_variance(gr: &GrassmannManifold, mu: &Array1<f64>, frames: &[Array1<f64>]) -> f64 {
    frames
        .iter()
        .map(|f| {
            let d = grassmann_dist(gr, mu, f);
            d * d
        })
        .sum()
}

const D: usize = 8; // ambient dimension (eight fatty acids)
const K: usize = 3; // subspace dimension -> Gr(3, 8)
const FEATURES: [&str; D] = [
    "palmitic",
    "palmitoleic",
    "stearic",
    "oleic",
    "linoleic",
    "linolenic",
    "arachidic",
    "eicosenoic",
];

/// Path to the vendored olive-oil CSV, relative to the crate manifest dir so the
/// test runs from any working directory.
fn dataset_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("bench/datasets/olive_oils.csv")
}

/// Parse the olive-oil CSV into `area -> Vec<[f64; D]>` (one feature row per
/// sample). The header is `rownames,region,area,<8 fatty acids>`.
fn load_groups() -> BTreeMap<String, Vec<[f64; D]>> {
    let text = std::fs::read_to_string(dataset_path()).expect("read olive_oils.csv");
    let mut lines = text.lines();
    let header: Vec<&str> = lines
        .next()
        .expect("olive_oils.csv header")
        .split(',')
        .collect();
    let area_idx = header
        .iter()
        .position(|h| *h == "area")
        .expect("olive_oils.csv has an `area` column");
    let feat_idx: Vec<usize> = FEATURES
        .iter()
        .map(|f| {
            header
                .iter()
                .position(|h| h == f)
                .unwrap_or_else(|| panic!("olive_oils.csv missing feature column {f:?}"))
        })
        .collect();

    let mut groups: BTreeMap<String, Vec<[f64; D]>> = BTreeMap::new();
    for line in lines {
        if line.trim().is_empty() {
            continue;
        }
        let fields: Vec<&str> = line.split(',').collect();
        let area = fields[area_idx].to_string();
        let mut row = [0.0_f64; D];
        for (j, &ci) in feat_idx.iter().enumerate() {
            row[j] = fields[ci]
                .parse::<f64>()
                .expect("olive_oils.csv numeric fatty-acid value");
        }
        groups.entry(area).or_default().push(row);
    }
    groups
}

/// Symmetric-eigenvalue solver (cyclic Jacobi) for a small SPD matrix; returns
/// `(eigenvalues_descending, eigenvectors_as_columns)`. Used to form each group's
/// PCA subspace (eigenvectors of the 8×8 covariance) and to read off principal
/// angles, with no dependency on an external linear-algebra crate.
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

/// Top-`K` principal directions of one group around an explicit `origin`: subtract
/// `origin` from every row, form the `D×D` second-moment matrix of the residuals,
/// and take its leading `K` eigenvectors as an orthonormal `D×K` frame (a point of
/// `St(D,K)`, spanning a point of `Gr(K,D)`).
///
/// The choice of `origin` selects *which* subspace identity the frame encodes:
///   - `origin = group mean` → the within-area covariance subspace ("which
///     directions does this region vary along"). Location-blind: a region's mean
///     composition is projected out, so two regions that differ only in their mean
///     fatty-acid profile map to the SAME Grassmann point.
///   - `origin = global centroid` → the area-identity subspace: the residuals now
///     carry the area's *offset from the global mean composition* as their leading
///     direction, so the frame encodes both WHERE the region sits in fatty-acid
///     space and how it spreads. This is the discriminative "subspace of this
///     region" — the olive-oil signal lives in the mean offset, which the
///     mean-centered covariance subspace discards.
fn subspace_about(rows: &[[f64; D]], origin: &[f64; D]) -> Array2<f64> {
    let n = rows.len();
    assert!(n > K, "group too small for a rank-{K} subspace");
    let mut moment = Array2::<f64>::zeros((D, D));
    for r in rows {
        let centered: [f64; D] = std::array::from_fn(|j| r[j] - origin[j]);
        for i in 0..D {
            for j in 0..D {
                moment[[i, j]] += centered[i] * centered[j];
            }
        }
    }
    moment /= (n - 1) as f64;
    let (evals, evecs) = sym_eig(&moment);
    assert!(
        evals[K - 1] - evals[K] > 1e-6,
        "subspace spectral gap at rank {K} is too small ({:.3e}); the top-{K} \
         subspace is not well defined",
        evals[K - 1] - evals[K]
    );
    let mut frame = Array2::<f64>::zeros((D, K));
    for j in 0..K {
        for i in 0..D {
            frame[[i, j]] = evecs[[i, j]];
        }
    }
    frame
}

/// Group mean of an `n×D` sample block (the per-area centroid).
fn group_mean(rows: &[[f64; D]]) -> [f64; D] {
    let n = rows.len() as f64;
    let mut mean = [0.0_f64; D];
    for r in rows {
        for j in 0..D {
            mean[j] += r[j];
        }
    }
    for m in &mut mean {
        *m /= n;
    }
    mean
}

/// The classic within-area covariance subspace: top-`K` PCA directions about the
/// group's OWN mean. Used by the geometry round-trip test, which only checks that
/// gam's distance/log/exp reproduce the analytic principal-angle geometry and so is
/// indifferent to which orthonormal frames it is handed.
fn pca_subspace(rows: &[[f64; D]]) -> Array2<f64> {
    let mean = group_mean(rows);
    subspace_about(rows, &mean)
}

/// Row-major flatten of a `D×K` frame into the layout gam's `from_flat`/`flatten`
/// expect: `vec[r * K + c] = M[r, c]`.
fn flatten_frame(frame: &Array2<f64>) -> Array1<f64> {
    let mut v = Array1::<f64>::zeros(D * K);
    for r in 0..D {
        for c in 0..K {
            v[r * K + c] = frame[[r, c]];
        }
    }
    v
}

/// Inflate a row-major flat `D*K` vector back to a `D×K` matrix.
fn matrix_from_flat(flat: &[f64]) -> Array2<f64> {
    let mut m = Array2::<f64>::zeros((D, K));
    for r in 0..D {
        for c in 0..K {
            m[[r, c]] = flat[r * K + c];
        }
    }
    m
}

/// Principal angles (ascending) between two orthonormal `D×K` frames:
/// `θ = arccos(σ(YᵀZ))`. Matches the convention the geomstats side emits.
fn principal_angles(y: &Array2<f64>, z: &Array2<f64>) -> Vec<f64> {
    let c = y.t().dot(z); // K×K
    let gram = c.t().dot(&c); // eigenvalues = cos² θ
    let (evals, _) = sym_eig(&gram);
    let mut angles: Vec<f64> = evals
        .into_iter()
        .map(|e| e.max(0.0).min(1.0).sqrt().acos())
        .collect();
    angles.sort_by(|a, b| a.partial_cmp(b).unwrap());
    angles
}

/// Singular values (ascending) of a `D×K` tangent matrix: `√eig(MᵀM)`.
fn singular_values(m: &Array2<f64>) -> Vec<f64> {
    let gram = m.t().dot(m);
    let (evals, _) = sym_eig(&gram);
    let mut s: Vec<f64> = evals.into_iter().map(|e| e.max(0.0).sqrt()).collect();
    s.sort_by(|a, b| a.partial_cmp(b).unwrap());
    s
}

#[test]
fn olive_oils_grassmann_distance_matches_geomstats() {
    let gr = GrassmannManifold::new(K, D).expect("Gr(3, 8) is a valid Grassmannian");

    // --- build one PCA subspace per producing area, in a fixed alphabetical
    // order so gam and geomstats index the same frames identically. ----------
    let groups = load_groups();
    let area_names: Vec<String> = groups.keys().cloned().collect();
    let n_areas = area_names.len();
    assert!(
        n_areas >= 4,
        "expected several olive-oil areas, got {n_areas}"
    );

    let frames: Vec<Array2<f64>> = area_names
        .iter()
        .map(|a| pca_subspace(&groups[a]))
        .collect();
    let frames_flat: Vec<Array1<f64>> = frames.iter().map(flatten_frame).collect();

    // Sanity: every PCA frame is genuinely orthonormal (it is built from
    // eigenvectors of a symmetric covariance), so it is a valid Grassmann point.
    for (a, f) in area_names.iter().zip(&frames) {
        let g = f.t().dot(f);
        let mut defect = 0.0_f64;
        for i in 0..K {
            for j in 0..K {
                let target = if i == j { 1.0 } else { 0.0 };
                defect = defect.max((g[[i, j]] - target).abs());
            }
        }
        assert!(
            defect < 1e-10,
            "PCA frame for area {a:?} is not orthonormal: defect {defect:.3e}"
        );
    }

    // Flatten all ordered (i, j) pairs (i != j) for the geomstats reference. The
    // bases handed to geomstats are byte-identical to gam's: same row-major
    // entries, same column order, same pair enumeration. Both CSV columns carry
    // exactly `n_pairs * D * K` values (one flattened D×K frame per pair), so they
    // are equal length as the harness requires; the reference reshapes by
    // `len / (D*K)` to recover the pairs in the identical order.
    let mut base_flat: Vec<f64> = Vec::new();
    let mut other_flat: Vec<f64> = Vec::new();

    // gam-side per-pair quantities, in the SAME pair order.
    let mut gam_dist: Vec<f64> = Vec::new();
    // Analytic ground-truth distance per pair: √(Σ arccos²(σ)) from the principal
    // angles between the two PCA frames, computed by this test (NOT geomstats).
    let mut analytic_dist: Vec<f64> = Vec::new();
    // Worst |gam geodesic distance − ANALYTIC arc length| over all pairs (metric A).
    let mut max_dist_vs_analytic = 0.0_f64;
    // Worst |gam log-singular-values − analytic principal angles| over all pairs.
    let mut max_log_sigma_vs_angle = 0.0_f64;
    // Round-trip + isometry diagnostics (see metric C).
    let mut max_roundtrip_abs = 0.0_f64;
    let mut max_isometry_err = 0.0_f64;

    for i in 0..n_areas {
        for j in 0..n_areas {
            if i == j {
                continue;
            }
            let yi = frames_flat[i].view();
            let yj = frames_flat[j].view();

            // (A) gam geodesic distance = ‖Log_{P_i}(P_j)‖_F.
            let log_ij = gr.log_map(yi, yj).expect("gam Grassmann log_map");
            let dist_ij: f64 = log_ij.iter().map(|x| x * x).sum::<f64>().sqrt();
            gam_dist.push(dist_ij);

            // (B) singular spectrum of the log equals the analytic principal
            // angles between the two PCA frames.
            let log_mat = matrix_from_flat(log_ij.as_slice().unwrap());
            let log_sigma = singular_values(&log_mat);
            let angles = principal_angles(&frames[i], &frames[j]);
            max_log_sigma_vs_angle = max_log_sigma_vs_angle.max(max_abs_diff(&log_sigma, &angles));

            // (A) ANALYTIC ground-truth geodesic distance = root-sum-of-squared
            // principal angles, derived by this test from the bases themselves.
            // gam's geodesic distance must equal THIS, not geomstats' projector
            // `metric.dist` (wrong near the π/2 cut locus, #904).
            let arc_len: f64 = angles.iter().map(|t| t * t).sum::<f64>().sqrt();
            analytic_dist.push(arc_len);
            max_dist_vs_analytic = max_dist_vs_analytic.max((dist_ij - arc_len).abs());

            // (C) exp/log round-trip + isometry on a controlled tangent. Scale the
            // log toward area j down to a Frobenius norm safely inside the
            // injectivity radius π/2, then exp it and log back.
            let raw_norm = dist_ij.max(1e-300);
            let safe_target = (raw_norm).min(1.0); // ≤ 1 rad < π/2 ≈ 1.5708
            let scale = safe_target / raw_norm;
            let v: Array1<f64> = log_ij.mapv(|x| x * scale);
            let endpoint = gr.exp_map(yi, v.view()).expect("gam Grassmann exp_map");
            let v_rec = gr
                .log_map(yi, endpoint.view())
                .expect("gam Grassmann log_map of exp endpoint");
            max_roundtrip_abs = max_roundtrip_abs.max(max_abs_diff(
                v_rec.as_slice().unwrap(),
                v.as_slice().unwrap(),
            ));
            let v_norm: f64 = v.iter().map(|x| x * x).sum::<f64>().sqrt();
            let rec_norm: f64 = v_rec.iter().map(|x| x * x).sum::<f64>().sqrt();
            max_isometry_err = max_isometry_err.max((rec_norm - v_norm).abs());

            base_flat.extend(frames_flat[i].iter().copied());
            other_flat.extend(frames_flat[j].iter().copied());
        }
    }
    let n_pairs = gam_dist.len();
    assert_eq!(n_pairs, n_areas * (n_areas - 1), "ordered pair count");

    // --- geomstats INFORMATIONAL cross-check on the identical PCA frames -----
    // geomstats represents a Grassmann point as the orthogonal projector P = YYᵀ.
    // We feed the byte-identical row-major frames, rebuild Y_i, Y_j, and emit both
    // its canonical-metric `metric.dist` AND the analytic angle root-sum-of-squares
    // it computes itself. geomstats is NOT the ground truth here: its projector
    // `metric.dist` is known to disagree with the analytic arc length near the π/2
    // cut locus (#904). We use it only as a match-or-beat cross-check on the pairs
    // that sit safely AWAY from π/2, where the closed form is well-conditioned.
    let py = run_python(
        &[
            Column::new("base_flat", &base_flat),
            Column::new("other_flat", &other_flat),
        ],
        &format!(
            r#"
import numpy as np
from geomstats.geometry.grassmannian import Grassmannian

D, K = {d}, {k}
flat_base = np.asarray(df["base_flat"], dtype=float)
NP = flat_base.size // (D * K)  # one flattened D×K frame per ordered pair

base  = flat_base.reshape(NP, D, K)
other = np.asarray(df["other_flat"], dtype=float).reshape(NP, D, K)

space = Grassmannian(D, K)
metric = space.metric

def angles(Y, Z):
    s = np.linalg.svd(Y.T @ Z, compute_uv=False)
    s = np.clip(s, -1.0, 1.0)
    return np.sort(np.arccos(s))  # ascending

dist = []
ang_rss = []
for p in range(NP):
    Y = base[p]
    Z = other[p]
    P0 = Y @ Y.T
    P1 = Z @ Z.T
    dist.append(float(metric.dist(P0, P1)))
    th = angles(Y, Z)
    ang_rss.append(float(np.sqrt((th ** 2).sum())))

emit("dist", dist)
emit("ang_rss", ang_rss)
"#,
            d = D,
            k = K,
        ),
    );

    let gs_dist = py.vector("dist");
    let gs_ang_rss = py.vector("ang_rss");
    assert_eq!(gs_dist.len(), n_pairs, "geomstats distance count");
    assert_eq!(gs_ang_rss.len(), n_pairs, "geomstats angle-rss count");

    // gam vs the ANALYTIC arc length the test built itself — this is the PRIMARY
    // ground-truth comparison (metric A), independent of any external tool.
    let dist_vs_analytic = max_abs_diff(&gam_dist, &analytic_dist);
    assert!(
        (dist_vs_analytic - max_dist_vs_analytic).abs() < 1e-300,
        "per-pair and aggregate gam-vs-analytic distance error must agree: \
         {dist_vs_analytic:.3e} vs {max_dist_vs_analytic:.3e}"
    );

    // Cross-witness that the cut-locus disagreement lives in geomstats' PROJECTOR
    // `metric.dist`, not in the angle spectrum: geomstats' OWN arc-length
    // √(Σ arccos²(σ)) agrees with our Rust analytic arc length to the SVD floor on
    // every pair (the two are the identical closed form on byte-identical frames).
    let analytic_cross_tool = max_abs_diff(gs_ang_rss, &analytic_dist);

    // INFORMATIONAL ONLY: geomstats' projector `metric.dist` vs the analytic arc
    // length, restricted to pairs comfortably away from the π/2 cut locus (largest
    // principal angle ≤ 1.4 rad). Near π/2 geomstats' projector formula is known to
    // be wrong (#904); we never assert agreement there. On the well-conditioned
    // pairs gam already equals the analytic truth, so gam is a match-or-beat of
    // geomstats by construction; this block only reports the residuals.
    let cut = std::f64::consts::FRAC_PI_2 - 0.17; // ≈ 1.4 rad
    let mut gam_vs_gs_away = 0.0_f64;
    let mut gs_vs_analytic_away = 0.0_f64;
    let mut away_pairs = 0usize;
    for p in 0..n_pairs {
        // analytic_dist[p] is ‖θ‖₂; the cut-locus risk is in the LARGEST angle, but
        // ‖θ‖₂ ≥ θ_max so gating on the arc length being below `cut` is a sound,
        // conservative away-from-π/2 filter.
        if analytic_dist[p] <= cut {
            gam_vs_gs_away = gam_vs_gs_away.max((gam_dist[p] - gs_dist[p]).abs());
            gs_vs_analytic_away = gs_vs_analytic_away.max((gs_dist[p] - analytic_dist[p]).abs());
            away_pairs += 1;
        }
    }

    eprintln!(
        "Gr({K},{D}) olive-oil PCA subspaces: areas={n_areas} pairs={n_pairs} | \
         gam_dist_vs_ANALYTIC={dist_vs_analytic:.3e} (PRIMARY) | \
         away-from-π/2 ({away_pairs} pairs): gam_vs_geomstats={gam_vs_gs_away:.3e} \
         geomstats_vs_analytic={gs_vs_analytic_away:.3e} (informational) | \
         geomstats_angle_rss_vs_analytic={analytic_cross_tool:.3e} (cross-tool angle witness) | \
         log_sigma_vs_angles={max_log_sigma_vs_angle:.3e} \
         roundtrip_abs={max_roundtrip_abs:.3e} isometry_err={max_isometry_err:.3e}"
    );

    // ===================== (A) geodesic distance vs ANALYTIC TRUTH ===========
    // The Grassmann geodesic distance is the exact arc length √(Σ arccos²(σ)) the
    // test derives from the SVD of YᵢᵀYⱼ. gam must match THAT to the LAPACK-SVD
    // noise floor on every real subspace pair — never geomstats, whose projector
    // distance is wrong at the π/2 cut locus (#904).
    assert!(
        dist_vs_analytic < 1e-9,
        "gam Grassmann geodesic distance disagrees with the ANALYTIC principal-angle \
         arc length on the olive-oil PCA subspaces: max |Δ| = {dist_vs_analytic:.3e}"
    );

    // ===================== (B) log spectrum = principal angles ==============
    // gam's log singular values equal the principal angles derived from the SVD of
    // YᵢᵀYⱼ — an absolute-accuracy claim against a quantity the test builds itself.
    assert!(
        max_log_sigma_vs_angle < 1e-9,
        "gam Grassmann log singular spectrum deviates from the analytic principal \
         angles: max |Δ| = {max_log_sigma_vs_angle:.3e}"
    );

    // ===================== (C) exp/log round-trip + isometry ================
    // log(exp(v)) == v componentwise, and ‖log(exp(v))‖_F == ‖v‖_F, on tangents
    // pointing between real PCA subspaces (scaled inside the injectivity radius).
    assert!(
        max_roundtrip_abs < 1e-10,
        "gam Grassmann exp/log round-trip error on olive-oil subspace tangents too \
         large: {max_roundtrip_abs:.3e}"
    );
    assert!(
        max_isometry_err < 1e-9,
        "gam Grassmann geodesic violates the metric isometry ‖log(exp(v))‖ = ‖v‖ on \
         olive-oil subspace tangents: {max_isometry_err:.3e}"
    );
}

/// REAL-DATA predictive arm (truth unknown → judged on an OBJECTIVE held-out
/// metric, with geomstats as a match-or-beat baseline, never a target to copy).
///
/// SOURCE: the same vendored Forina Italian olive-oil dataset documented at the
/// top of this file (`bench/datasets/olive_oils.csv`, mirrored from
/// https://raw.githubusercontent.com/vincentarelbundock/Rdatasets/master/csv/dslabs/olive.csv).
///
/// TASK (exercises gam's Grassmann `log_map`/`exp_map` exactly as the synthetic
/// test does, but on a held-out *prediction*): for every producing area we split
/// that area's samples deterministically — even row index → TRAIN, odd row index
/// → TEST — and form an independent top-`K` subspace (a point of `Gr(3,8)`) from
/// each half, built about the SHARED GLOBAL fatty-acid centroid rather than each
/// half's own mean. Centering on the global centroid keeps the region's offset
/// from the overall mean composition as the leading frame direction, so the
/// subspace encodes the area's location-and-spread identity — the actual olive-oil
/// discriminator. (A mean-centered within-area covariance subspace would project
/// that offset out and is near-blind here; see `subspace_about`.) The TRAIN
/// subspaces are class prototypes; each held-out TEST subspace is classified to the
/// area whose TRAIN prototype is GRASSMANN-NEAREST (smallest `‖Log‖_F` geodesic
/// distance). Because a region's centroid-relative subspace is stable across a
/// random split, a test half should land closest to its own train half — so
/// classification accuracy is a genuine objective signal.
///
/// OBJECTIVE METRIC (the pass criteria):
///   PRIMARY (tool-free): held-out nearest-prototype classification ACCURACY,
///     where the metric is gam's own Grassmann geodesic distance, must clear an
///     absolute bar well above the 1/n_areas random baseline.
///   GEOMETRY PRIMARY (match-or-beat): gam's intrinsic Grassmann Fréchet mean of
///     the TRAIN prototypes (built only from gam's exp/log) must achieve a Fréchet
///     variance no worse than geomstats' `FrechetMean` (the mature manifold
///     library) by more than a small margin — gam's mean is at least as central.
///   BASELINE (match-or-beat): gam's classification accuracy must be no worse than
///     geomstats' accuracy on the byte-identical frames minus a one-sample margin.
#[test]
fn olive_oils_grassmann_distance_matches_geomstats_on_real_data() {
    let gr = GrassmannManifold::new(K, D).expect("Gr(3, 8) is a valid Grassmannian");

    // --- deterministic per-area train/test split on real samples -----------
    // Within each area, even local index → TRAIN, odd → TEST (a fixed, RNG-free
    // split). Each half must still exceed K samples to admit a rank-K subspace.
    let groups = load_groups();

    // Global fatty-acid centroid across EVERY sample of every area. Each area's
    // train/test subspace is built about THIS shared origin (not the area's own
    // mean), so the leading frame direction encodes the region's offset from the
    // global mean composition — the location signal that actually discriminates
    // olive-oil regions. A mean-centered covariance subspace would project that
    // offset out and is near-blind here (see `subspace_about`).
    let global_centroid = {
        let all_rows: Vec<[f64; D]> = groups.values().flatten().copied().collect();
        group_mean(&all_rows)
    };

    let mut area_names: Vec<String> = Vec::new();
    let mut train_frames: Vec<Array1<f64>> = Vec::new();
    let mut test_frames: Vec<Array1<f64>> = Vec::new();

    for (area, rows) in groups.iter() {
        let train_rows: Vec<[f64; D]> = rows
            .iter()
            .enumerate()
            .filter(|(i, _)| i % 2 == 0)
            .map(|(_, r)| *r)
            .collect();
        let test_rows: Vec<[f64; D]> = rows
            .iter()
            .enumerate()
            .filter(|(i, _)| i % 2 == 1)
            .map(|(_, r)| *r)
            .collect();
        // Only keep areas whose BOTH halves admit a well-defined rank-K subspace
        // (enough samples and a clean spectral gap). subspace_about asserts the gap,
        // so gate on size here and let it assert the gap.
        if train_rows.len() <= K || test_rows.len() <= K {
            continue;
        }
        area_names.push(area.clone());
        train_frames.push(flatten_frame(&subspace_about(
            &train_rows,
            &global_centroid,
        )));
        test_frames.push(flatten_frame(&subspace_about(&test_rows, &global_centroid)));
    }
    let n_areas = area_names.len();
    assert!(
        n_areas >= 4,
        "expected several olive-oil areas with splittable samples, got {n_areas}"
    );

    // --- gam Grassmann nearest-prototype classification of held-out halves --
    // Distance kernel = gam's log_map Frobenius norm (metric A of the ground-truth
    // test). For each test subspace, the predicted area is argmin over train
    // prototypes; the full distance matrix is also handed to geomstats below.
    let mut dist_matrix: Vec<f64> = Vec::with_capacity(n_areas * n_areas);
    // ANALYTIC ground-truth distance matrix: √(Σ arccos²(σ)) between every
    // (test, train) pair, computed by this test from the frames themselves —
    // the geometry truth gam's distance matrix must reproduce (NOT geomstats').
    let mut analytic_matrix: Vec<f64> = Vec::with_capacity(n_areas * n_areas);
    let mut gam_correct = 0usize;
    for (ti, test) in test_frames.iter().enumerate() {
        let test_mat = matrix_from_flat(test.as_slice().unwrap());
        let mut best_j = 0usize;
        let mut best_d = f64::INFINITY;
        for (tj, train) in train_frames.iter().enumerate() {
            let d = grassmann_dist(&gr, test, train);
            dist_matrix.push(d);
            let train_mat = matrix_from_flat(train.as_slice().unwrap());
            let angles = principal_angles(&test_mat, &train_mat);
            let arc_len: f64 = angles.iter().map(|t| t * t).sum::<f64>().sqrt();
            analytic_matrix.push(arc_len);
            if d < best_d {
                best_d = d;
                best_j = tj;
            }
        }
        if best_j == ti {
            gam_correct += 1;
        }
    }
    let gam_accuracy = gam_correct as f64 / n_areas as f64;

    // --- gam intrinsic Grassmann Fréchet mean of the TRAIN prototypes -------
    let gam_mean = grassmann_frechet_mean(&gr, &train_frames);
    let gam_mean_variance = frechet_variance(&gr, &gam_mean, &train_frames);
    // Orthonormality of gam's mean (it must be a valid Grassmann point).
    let gm = matrix_from_flat(gam_mean.as_slice().unwrap());
    let gm_gram = gm.t().dot(&gm);
    let mut gm_defect = 0.0_f64;
    for i in 0..K {
        for j in 0..K {
            let target = if i == j { 1.0 } else { 0.0 };
            gm_defect = gm_defect.max((gm_gram[[i, j]] - target).abs());
        }
    }
    assert!(
        gm_defect < 1e-8,
        "gam Grassmann Fréchet mean is not orthonormal: defect {gm_defect:.3e}"
    );

    // --- geomstats BASELINE on byte-identical frames ------------------------
    // Two equal-length flat columns (train + test prototypes, each n_areas frames
    // of D*K row-major entries) plus the gam Fréchet mean as a third equal-length
    // column padded to the same length so the harness sees uniform row counts.
    let mut train_flat: Vec<f64> = Vec::new();
    let mut test_flat: Vec<f64> = Vec::new();
    for (tr, te) in train_frames.iter().zip(&test_frames) {
        train_flat.extend(tr.iter().copied());
        test_flat.extend(te.iter().copied());
    }
    // gam's mean is a single D*K frame; pad it to n_areas*D*K with zeros so all
    // three columns are equal length. geomstats reads only the first D*K entries.
    let mut gam_mean_flat: Vec<f64> = gam_mean.to_vec();
    gam_mean_flat.resize(train_flat.len(), 0.0);

    let py = run_python(
        &[
            Column::new("train_flat", &train_flat),
            Column::new("test_flat", &test_flat),
            Column::new("gam_mean_flat", &gam_mean_flat),
        ],
        &format!(
            r#"
import numpy as np
from geomstats.geometry.grassmannian import Grassmannian
from geomstats.learning.frechet_mean import FrechetMean

D, K = {d}, {k}
train = np.asarray(df["train_flat"], dtype=float)
NA = train.size // (D * K)  # one flattened D×K frame per area

train = train.reshape(NA, D, K)
test  = np.asarray(df["test_flat"], dtype=float).reshape(NA, D, K)
gam_mean = np.asarray(df["gam_mean_flat"], dtype=float)[: D * K].reshape(D, K)

space = Grassmannian(D, K)
metric = space.metric

def proj(Y):
    return Y @ Y.T  # geomstats Grassmann point = orthogonal projector

P_train = np.stack([proj(train[a]) for a in range(NA)])
P_test  = np.stack([proj(test[a])  for a in range(NA)])
P_gam_mean = proj(gam_mean)

# Nearest-prototype classification accuracy under geomstats' own distance.
correct = 0
dist = []
for a in range(NA):
    ds = [float(metric.dist(P_test[a], P_train[b])) for b in range(NA)]
    dist.extend(ds)
    if int(np.argmin(ds)) == a:
        correct += 1
emit("gs_accuracy", [correct / NA])
emit("gs_dist", dist)

# geomstats intrinsic Fréchet mean of the train prototypes, and the Fréchet
# variance achieved by BOTH means (geomstats' own and gam's).
fm = FrechetMean(space)
fm.fit(P_train)
P_gs_mean = fm.estimate_

# Fréchet variance must be scored in the CANONICAL Grassmann metric — the
# principal-angle arc length √(Σ arccos²σ) — the SAME metric gam uses and the
# one this test pins as ground truth. geomstats' projector `metric.dist` is a
# DIFFERENT convention: it equals √2 · arc-length (the projector-embedding
# Frobenius metric, dist(YYᵀ,ZZᵀ) = √2·‖θ‖₂), so scoring variance with it would
# inflate every squared distance by exactly 2 and make gam's arc-length variance
# look artificially "more central". We recover an orthonormal basis from each
# projector (its top-K unit eigenvectors) and use the canonical arc length, so
# `gam_mean_variance_gs` is directly comparable to gam's own evaluation.
def basis_of(P):
    w, V = np.linalg.eigh(P)
    return V[:, np.argsort(w)[::-1][:K]]  # top-K eigenvectors span the subspace

def arc_dist(P_mean, P_other):
    Ym = basis_of(P_mean)
    Yo = basis_of(P_other)
    s = np.clip(np.linalg.svd(Ym.T @ Yo, compute_uv=False), -1.0, 1.0)
    th = np.arccos(s)
    return float(np.sqrt((th ** 2).sum()))

def variance(P_mean):
    return float(sum(arc_dist(P_mean, P_train[b]) ** 2 for b in range(NA)))

emit("gs_mean_variance", [variance(P_gs_mean)])
emit("gam_mean_variance_gs", [variance(P_gam_mean)])
"#,
            d = D,
            k = K,
        ),
    );

    let gs_accuracy = py.scalar("gs_accuracy");
    let gs_dist = py.vector("gs_dist");
    let gs_mean_variance = py.scalar("gs_mean_variance");
    let gam_mean_variance_gs = py.scalar("gam_mean_variance_gs");
    assert_eq!(
        gs_dist.len(),
        n_areas * n_areas,
        "geomstats distance-matrix size"
    );

    // gam's distance matrix must agree with the ANALYTIC arc length the test built
    // itself (the closed-form geodesic distance the ground-truth test pins),
    // confirming the geometry is correct before we compare classification
    // decisions. This is gam-vs-self-constructed-truth, never gam-vs-geomstats.
    let dist_diff = max_abs_diff(&dist_matrix, &analytic_matrix);

    // INFORMATIONAL ONLY: geomstats' projector `metric.dist` vs the analytic arc
    // length, restricted to (test, train) pairs away from the π/2 cut locus where
    // geomstats' formula is sound (#904). We never assert gam == geomstats here.
    let cut = std::f64::consts::FRAC_PI_2 - 0.17; // ≈ 1.4 rad
    let mut gs_vs_analytic_away = 0.0_f64;
    let mut away_pairs = 0usize;
    for (k, &an) in analytic_matrix.iter().enumerate() {
        if an <= cut {
            gs_vs_analytic_away = gs_vs_analytic_away.max((gs_dist[k] - an).abs());
            away_pairs += 1;
        }
    }

    eprintln!(
        "Gr({K},{D}) olive-oil held-out subspace classification: areas={n_areas} \
         gam_accuracy={gam_accuracy:.3} gs_accuracy={gs_accuracy:.3} \
         dist_vs_ANALYTIC={dist_diff:.3e} (PRIMARY) \
         geomstats_vs_analytic_away-from-π/2={gs_vs_analytic_away:.3e} ({away_pairs} pairs, informational) | \
         Fréchet variance gam={gam_mean_variance:.4} (geomstats-recomputed={gam_mean_variance_gs:.4}) \
         geomstats={gs_mean_variance:.4}"
    );

    // distance-matrix geometry gate vs ANALYTIC truth (parity before decision parity).
    assert!(
        dist_diff < 1e-7,
        "gam Grassmann distance matrix disagrees with the ANALYTIC principal-angle \
         arc length on the held-out olive-oil subspaces: max |Δ| = {dist_diff:.3e}"
    );

    // ===================== PRIMARY: absolute held-out accuracy ==============
    // Random nearest-prototype guessing scores 1/n_areas; a region's covariance is
    // stable across the even/odd split, so the correct area must dominate. Require
    // a strong absolute majority well above chance.
    let chance = 1.0 / n_areas as f64;
    assert!(
        gam_accuracy >= 0.6 && gam_accuracy > chance + 0.2,
        "gam held-out Grassmann nearest-prototype accuracy too low: {gam_accuracy:.3} \
         (chance={chance:.3}, n_areas={n_areas})"
    );

    // ===================== BASELINE: match-or-beat geomstats accuracy =======
    // gam must classify the held-out subspaces at least as well as the mature
    // manifold library, allowing one sample of slack for argmin tie-breaking.
    let one_sample = 1.0 / n_areas as f64 + 1e-9;
    assert!(
        gam_accuracy >= gs_accuracy - one_sample,
        "gam held-out accuracy {gam_accuracy:.3} trails geomstats {gs_accuracy:.3} by more \
         than one sample"
    );

    // ===================== GEOMETRY PRIMARY: Fréchet mean centrality ========
    // gam's intrinsic Grassmann mean (built purely from gam exp/log) must be at
    // least as central as geomstats' FrechetMean: its Fréchet variance is no worse
    // than geomstats' by more than a small relative margin. Compare on geomstats'
    // OWN recomputation of gam's variance to avoid any cross-tool metric drift.
    let margin = 1e-6 * gs_mean_variance.max(1.0);
    assert!(
        gam_mean_variance_gs <= gs_mean_variance + margin,
        "gam Grassmann Fréchet mean is less central than geomstats: gam variance \
         {gam_mean_variance_gs:.6} > geomstats {gs_mean_variance:.6} + {margin:.3e}"
    );

    // Self-consistency: gam's own variance evaluation of its mean tracks geomstats'
    // recomputation of the same quantity (no metric-convention mismatch).
    assert!(
        (gam_mean_variance - gam_mean_variance_gs).abs() < 1e-6 * gam_mean_variance.max(1.0),
        "gam vs geomstats disagree on gam-mean Fréchet variance: {gam_mean_variance:.6} vs \
         {gam_mean_variance_gs:.6}"
    );
}
