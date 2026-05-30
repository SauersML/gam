//! End-to-end quality: gam's Grassmann `Gr(3, 8)` geodesic distance, exponential,
//! and logarithm must reproduce the EXACT subspace geometry of real per-group PCA
//! subspaces, judged against geomstats GROUND TRUTH on byte-identical bases.
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
//! OBJECTIVE ACCURACY (the pass criteria), all vs ground truth on real subspaces:
//!   (A) **Geodesic distance.** For every ordered pair of areas, the Grassmann
//!       geodesic distance `d(P_i, P_j) = ‖Log_{P_i}(P_j)‖_F = ‖θ‖₂` (root-sum of
//!       squared principal angles) computed from gam's `log_map` must match
//!       geomstats' canonical-metric distance to `< 1e-9`. The angle spectrum is an
//!       exact closed form (SVD of `Y_iᵀ Y_j`), so geomstats is mathematical truth,
//!       not a noisy peer fit.
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
    let header: Vec<&str> = lines.next().expect("olive_oils.csv header").split(',').collect();
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

/// Top-`K` principal directions of one group: center the `n×D` matrix, form the
/// `D×D` sample covariance, and take its leading `K` eigenvectors as an
/// orthonormal `D×K` frame (a point of `St(D,K)`, spanning a point of `Gr(K,D)`).
fn pca_subspace(rows: &[[f64; D]]) -> Array2<f64> {
    let n = rows.len();
    assert!(n > K, "group too small for a rank-{K} PCA subspace");
    let mut mean = [0.0_f64; D];
    for r in rows {
        for j in 0..D {
            mean[j] += r[j];
        }
    }
    for m in &mut mean {
        *m /= n as f64;
    }
    let mut cov = Array2::<f64>::zeros((D, D));
    for r in rows {
        let centered: [f64; D] = std::array::from_fn(|j| r[j] - mean[j]);
        for i in 0..D {
            for j in 0..D {
                cov[[i, j]] += centered[i] * centered[j];
            }
        }
    }
    cov /= (n - 1) as f64;
    let (evals, evecs) = sym_eig(&cov);
    assert!(
        evals[K - 1] - evals[K] > 1e-6,
        "PCA spectral gap at rank {K} is too small ({:.3e}); the top-{K} subspace is \
         not well defined",
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
            max_log_sigma_vs_angle =
                max_log_sigma_vs_angle.max(max_abs_diff(&log_sigma, &angles));

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
            max_roundtrip_abs = max_roundtrip_abs
                .max(max_abs_diff(v_rec.as_slice().unwrap(), v.as_slice().unwrap()));
            let v_norm: f64 = v.iter().map(|x| x * x).sum::<f64>().sqrt();
            let rec_norm: f64 = v_rec.iter().map(|x| x * x).sum::<f64>().sqrt();
            max_isometry_err = max_isometry_err.max((rec_norm - v_norm).abs());

            base_flat.extend(frames_flat[i].iter().copied());
            other_flat.extend(frames_flat[j].iter().copied());
        }
    }
    let n_pairs = gam_dist.len();
    assert_eq!(n_pairs, n_areas * (n_areas - 1), "ordered pair count");

    // --- geomstats GROUND TRUTH on the identical PCA frames -----------------
    // geomstats represents a Grassmann point as the orthogonal projector P = YYᵀ.
    // We feed the byte-identical row-major frames, rebuild Y_i, Y_j, and emit the
    // canonical-metric geodesic distance for every ordered pair. The Grassmann
    // distance is an exact closed form (the principal-angle root-sum-of-squares),
    // so geomstats is mathematical truth, not a noisy peer fit.
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

    // geomstats self-consistency: its canonical distance equals the principal-angle
    // root-sum-of-squares it computes independently (closed-form truth, both ways).
    let gs_internal = max_abs_diff(gs_dist, gs_ang_rss);

    // PRIMARY (A): gam's geodesic distance vs geomstats ground truth.
    let dist_diff = max_abs_diff(&gam_dist, gs_dist);

    eprintln!(
        "Gr({K},{D}) olive-oil PCA subspaces: areas={n_areas} pairs={n_pairs} | \
         gam_dist_vs_geomstats={dist_diff:.3e} (geomstats internal dist-vs-angles={gs_internal:.3e}) \
         log_sigma_vs_angles={max_log_sigma_vs_angle:.3e} \
         roundtrip_abs={max_roundtrip_abs:.3e} isometry_err={max_isometry_err:.3e}"
    );

    // ===================== (A) geodesic distance vs GROUND TRUTH =============
    // The Grassmann geodesic distance is an exact closed form, so gam must match
    // geomstats to its LAPACK-SVD noise floor on every real subspace pair.
    assert!(
        dist_diff < 1e-9,
        "gam Grassmann geodesic distance disagrees with geomstats ground truth on \
         the olive-oil PCA subspaces: max |Δ| = {dist_diff:.3e}"
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
