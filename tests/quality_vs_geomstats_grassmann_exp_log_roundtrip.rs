//! End-to-end quality: gam's Grassmann exponential/logarithm maps must satisfy
//! the exact Riemannian-geodesic axioms of `Gr(3, 10)` against ANALYTIC ground
//! truth — not merely reproduce a peer library's output.
//!
//! The Grassmannian `Gr(k, n)` is a quotient manifold: a point is a
//! `k`-dimensional subspace of `ℝⁿ` (equivalently an orthonormal `n×k` frame,
//! modulo `O(k)`). Its Riemannian exponential and logarithm have a closed form
//! built from the *compact SVD* of the tangent matrix and `cos`/`sin` of the
//! singular values (= principal angles); there is no optimization, so the maps
//! are EXACT mathematical objects whose properties are derivable a priori.
//!
//! OBJECTIVE METRIC (the pass criterion): for a base frame `P` and a horizontal
//! tangent `v` with compact-SVD spectrum `σ` (constructed by the test, hence
//! known in closed form), the geodesic `t ↦ exp_P(t v)` obeys three analytic
//! axioms that gam must reproduce to linear-algebra precision, WITHOUT reference
//! to any external tool:
//!   (a) **Round-trip / involution.** `log_P(exp_P(v)) == v` to
//!       `max_abs_diff < 1e-10`, Frobenius `< 1e-9`. exp pushes the angles
//!       through `cos`/`sin`; log pulls them back through `atan`; the residual is
//!       pure f64 rounding.
//!   (b) **Principal-angle law θ = σ.** The principal angles between `P` and the
//!       geodesic endpoint `exp_P(v)` equal the input singular spectrum `σ`
//!       exactly; likewise the singular spectrum of the recovered `log` equals
//!       `σ`. `σ` is computed directly from `v` by this test (analytic truth),
//!       so this is an absolute-accuracy claim, not a cross-tool agreement.
//!   (c) **Isometry of the metric.** The geodesic distance
//!       `dist(P, exp_P(v)) = ‖log_P(exp_P(v))‖_F = ‖v‖_F = ‖σ‖₂`. gam's exp/log
//!       must preserve the canonical Frobenius length to `< 1e-9`.
//! All three are asserted against quantities the test derives itself; the worst
//! deviation across all `N_CASES` must clear the bar.
//!
//! geomstats is retained ONLY as an independent GROUND-TRUTH cross-check (the
//! Grassmann geodesic is an exact closed form, so geomstats is mathematical
//! truth, not a noisy peer fit — the EXCEPTION case): we hand it the identical
//! base/tangent and additionally require gam to match its exact endpoint and
//! log invariants. But the primary, sufficient pass criterion is gam-vs-analytic
//! axioms above; geomstats only sharpens it.
//!
//! There is no skip path: if `python3` or `geomstats` is missing the reference
//! body fails loudly and so does this test.

use gam::geometry::{GrassmannManifold, RiemannianManifold};
use gam::test_support::reference::{Column, max_abs_diff, run_python};
use ndarray::{Array1, Array2};

const N: usize = 10; // ambient dimension
const K: usize = 3; // subspace dimension -> Gr(3, 10)
const N_CASES: usize = 16; // independent (base, tangent) draws

/// Deterministic xorshift64* PRNG so gam and geomstats see byte-identical data.
struct Rng(u64);
impl Rng {
    fn new(seed: u64) -> Self {
        Rng(seed ^ 0x9E37_79B9_7F4A_7C15)
    }
    fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        x.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }
    /// Standard-normal via Box-Muller (one of the two deviates per call).
    fn next_normal(&mut self) -> f64 {
        let u1 = ((self.next_u64() >> 11) as f64 + 1.0) / (9_007_199_254_740_992.0 + 1.0);
        let u2 = (self.next_u64() >> 11) as f64 / 9_007_199_254_740_992.0;
        (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
    }
}

/// Thin Gram-Schmidt QR: orthonormal `n×k` basis of the column space of `a`.
fn qr_orthonormal(a: &Array2<f64>) -> Array2<f64> {
    let (n, k) = (a.nrows(), a.ncols());
    let mut q = Array2::<f64>::zeros((n, k));
    for j in 0..k {
        let mut v = a.column(j).to_owned();
        for i in 0..j {
            let qi = q.column(i);
            let proj: f64 = qi.iter().zip(v.iter()).map(|(p, x)| p * x).sum();
            for r in 0..n {
                v[r] -= proj * qi[r];
            }
        }
        let nrm: f64 = v.iter().map(|x| x * x).sum::<f64>().sqrt();
        assert!(nrm > 1e-9, "QR encountered a rank-deficient column");
        for r in 0..n {
            q[[r, j]] = v[r] / nrm;
        }
    }
    q
}

/// Symmetric-eigenvalue solver (cyclic Jacobi) for a small `k×k` SPD matrix;
/// returns eigenvalues (descending) used only to read off singular values.
fn sym_eigvals(mat: &Array2<f64>) -> Vec<f64> {
    let m = mat.nrows();
    let mut a = mat.clone();
    for _ in 0..100 {
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
            }
        }
    }
    let mut ev: Vec<f64> = (0..m).map(|i| a[[i, i]]).collect();
    ev.sort_by(|x, y| y.partial_cmp(x).unwrap());
    ev
}

/// Principal angles (descending) between two orthonormal `n×k` frames `y`, `z`:
/// `θ = arccos(σ(yᵀz))`, the canonical `O(k)`-invariant subspace distance.
fn principal_angles(y: &Array2<f64>, z: &Array2<f64>) -> Vec<f64> {
    let c = y.t().dot(z); // k×k
    let gram = c.t().dot(&c); // (yᵀz)ᵀ(yᵀz), eigenvalues = cos² θ
    let mut angles: Vec<f64> = sym_eigvals(&gram)
        .into_iter()
        .map(|e| e.max(0.0).min(1.0).sqrt().acos())
        .collect();
    angles.sort_by(|a, b| a.partial_cmp(b).unwrap()); // ascending, matches geomstats side
    angles
}

#[test]
fn grassmann_exp_log_roundtrip_matches_geomstats() {
    let gr = GrassmannManifold::new(K, N).expect("Gr(3,10) is a valid Grassmannian");
    let mut rng = Rng::new(0x6772_6173_736D_6e00); // "grassmn"

    // Flattened (row-major) base frames and tangents shared with geomstats.
    let mut base_flat: Vec<f64> = Vec::with_capacity(N_CASES * N * K);
    let mut tangent_flat: Vec<f64> = Vec::with_capacity(N_CASES * N * K);

    let mut max_roundtrip_abs = 0.0_f64;
    let mut max_roundtrip_frob = 0.0_f64;
    // Worst deviation of gam's geodesic invariants from ANALYTIC ground truth
    // (the input spectrum σ that the test itself constructs): endpoint principal
    // angles vs σ, log-recovered singular values vs σ, and geodesic distance
    // dist(P, exp_P v) = ‖v‖_F vs ‖σ‖₂. These are the primary pass criteria.
    let mut max_exp_angle_truth_err = 0.0_f64;
    let mut max_log_sigma_truth_err = 0.0_f64;
    let mut max_isometry_err = 0.0_f64;
    // gam's principal angles of the geodesic endpoint and of its recovered log,
    // per case (flattened K-vectors), retained for the geomstats ground-truth
    // cross-check only.
    let mut gam_exp_angles: Vec<f64> = Vec::with_capacity(N_CASES * K);
    let mut gam_log_angles: Vec<f64> = Vec::with_capacity(N_CASES * K);

    for case in 0..N_CASES {
        // --- base subspace: orthonormalize a random n×k matrix -------------
        let mut a = Array2::<f64>::zeros((N, K));
        for r in 0..N {
            for c in 0..K {
                a[[r, c]] = rng.next_normal();
            }
        }
        let y = qr_orthonormal(&a);

        // --- raw random ambient direction, then project onto T_Y Gr --------
        let mut raw = Array2::<f64>::zeros((N, K));
        for r in 0..N {
            for c in 0..K {
                raw[[r, c]] = rng.next_normal();
            }
        }
        let y_flat: Array1<f64> = {
            let mut v = Array1::<f64>::zeros(N * K);
            for r in 0..N {
                for c in 0..K {
                    v[r * K + c] = y[[r, c]];
                }
            }
            v
        };
        let raw_flat: Array1<f64> = {
            let mut v = Array1::<f64>::zeros(N * K);
            for r in 0..N {
                for c in 0..K {
                    v[r * K + c] = raw[[r, c]];
                }
            }
            v
        };
        let h_flat = gr
            .project_tangent(y_flat.view(), raw_flat.view())
            .expect("project random direction onto tangent space at Y");
        // Rescale the tangent so its Frobenius norm lands in [1e-7, π/3], well
        // inside the injectivity radius (π/2) where exp/log are bijective.
        let cur_norm: f64 = h_flat.iter().map(|x| x * x).sum::<f64>().sqrt();
        assert!(cur_norm > 1e-12, "projected tangent collapsed to zero");
        let frac = case as f64 / (N_CASES - 1) as f64;
        let target = 1e-7 * (((std::f64::consts::PI / 3.0) / 1e-7).powf(frac)); // geometric sweep
        let scale = target / cur_norm;
        let v_flat: Array1<f64> = h_flat.mapv(|x| x * scale);

        // --- (1) intrinsic gam round-trip: log(exp(P,v)) == v --------------
        let endpoint = gr
            .exp_map(y_flat.view(), v_flat.view())
            .expect("gam exp_map");
        let v_rec = gr
            .log_map(y_flat.view(), endpoint.view())
            .expect("gam log_map");
        let abs = max_abs_diff(v_rec.as_slice().unwrap(), v_flat.as_slice().unwrap());
        let frob: f64 = v_rec
            .iter()
            .zip(v_flat.iter())
            .map(|(a, b)| (a - b) * (a - b))
            .sum::<f64>()
            .sqrt();
        max_roundtrip_abs = max_roundtrip_abs.max(abs);
        max_roundtrip_frob = max_roundtrip_frob.max(frob);

        // --- ANALYTIC ground truth for this case --------------------------
        // The principal angles of exp_P(v) equal the singular values σ of the
        // tangent matrix v (the θ = σ law); the geodesic distance is ‖v‖_F.
        // Both are read off v directly, independent of any library.
        let v_in_mat = {
            let mut m = Array2::<f64>::zeros((N, K));
            for r in 0..N {
                for c in 0..K {
                    m[[r, c]] = v_flat[r * K + c];
                }
            }
            m
        };
        let mut truth_sigma: Vec<f64> = sym_eigvals(&v_in_mat.t().dot(&v_in_mat))
            .into_iter()
            .map(|e| e.max(0.0).sqrt())
            .collect();
        truth_sigma.sort_by(|a, b| a.partial_cmp(b).unwrap()); // ascending
        let truth_dist: f64 = truth_sigma.iter().map(|s| s * s).sum::<f64>().sqrt();

        // gam's principal angles at the endpoint and from the recovered log.
        let z = {
            let mut m = Array2::<f64>::zeros((N, K));
            for r in 0..N {
                for c in 0..K {
                    m[[r, c]] = endpoint[r * K + c];
                }
            }
            m
        };
        let exp_angles = principal_angles(&y, &z); // ascending
        // (b) endpoint principal angles must equal the analytic spectrum σ.
        max_exp_angle_truth_err =
            max_exp_angle_truth_err.max(max_abs_diff(&exp_angles, &truth_sigma));
        gam_exp_angles.extend(exp_angles.iter().copied());

        // Recovered-log principal angles = singular values of the recovered
        // tangent matrix = sqrt of eigenvalues of (V_recᵀ V_rec).
        let v_rec_mat = {
            let mut m = Array2::<f64>::zeros((N, K));
            for r in 0..N {
                for c in 0..K {
                    m[[r, c]] = v_rec[r * K + c];
                }
            }
            m
        };
        let mut log_sigma: Vec<f64> = sym_eigvals(&v_rec_mat.t().dot(&v_rec_mat))
            .into_iter()
            .map(|e| e.max(0.0).sqrt())
            .collect();
        log_sigma.sort_by(|a, b| a.partial_cmp(b).unwrap()); // ascending
        // (b) log-recovered singular spectrum must equal the analytic spectrum.
        max_log_sigma_truth_err =
            max_log_sigma_truth_err.max(max_abs_diff(&log_sigma, &truth_sigma));
        // (c) isometry: geodesic distance ‖log_P(exp_P v)‖_F must equal ‖v‖_F.
        let gam_dist: f64 = v_rec.iter().map(|x| x * x).sum::<f64>().sqrt();
        max_isometry_err = max_isometry_err.max((gam_dist - truth_dist).abs());
        gam_log_angles.extend(log_sigma);

        base_flat.extend(y_flat.iter().copied());
        tangent_flat.extend(v_flat.iter().copied());
    }

    // --- geomstats reference on the identical base frames + tangents -------
    // geomstats represents a Grassmann point as the orthogonal projector
    // P = YYᵀ (n×n) and a tangent as W = HYᵀ + YHᵀ. We rebuild Y and H from
    // the shared row-major flat vectors, hand geomstats the projector forms,
    // and read back the principal angles of exp's endpoint and log's recovered
    // tangent — the representation-invariant invariants gam also reports.
    let py = run_python(
        &[
            Column::new("base_flat", &base_flat),
            Column::new("tangent_flat", &tangent_flat),
        ],
        &format!(
            r#"
import numpy as np
from geomstats.geometry.grassmannian import Grassmannian

N, K, NCASES = {n}, {k}, {ncases}
base = np.asarray(df["base_flat"], dtype=float).reshape(NCASES, N, K)
tang = np.asarray(df["tangent_flat"], dtype=float).reshape(NCASES, N, K)

space = Grassmannian(N, K)
metric = space.metric

def principal_angles(Y, Z):
    # Y, Z: orthonormal n x k frames.  theta = arccos(sigma(Y^T Z)).
    s = np.linalg.svd(Y.T @ Z, compute_uv=False)
    s = np.clip(s, -1.0, 1.0)
    return np.sort(np.arccos(s))  # ascending, matches the gam side

def frame_of_projector(P):
    # Top-k eigenvectors of the symmetric projector P span the subspace.
    w, V = np.linalg.eigh(P)
    return V[:, ::-1][:, :K]

exp_angles = []
log_angles = []
for i in range(NCASES):
    Y = base[i]                      # n x k, orthonormal
    H = tang[i]                      # n x k, horizontal (Y^T H = 0)
    P0 = Y @ Y.T                     # projector representation of the base
    W = H @ Y.T + Y @ H.T            # projector-tangent representation
    P1 = metric.exp(W, P0)           # geodesic endpoint (projector)
    Z = frame_of_projector(P1)
    ang = principal_angles(Y, Z)
    exp_angles.extend(ang.tolist())
    # log recovers a projector-tangent W_rec = Hr Y^T + Y Hr^T; its principal
    # angles are the singular values of the frame block Hr = W_rec @ Y.
    W_rec = metric.log(P1, P0)
    Hr = W_rec @ Y
    s = np.linalg.svd(Hr, compute_uv=False)
    log_angles.extend(np.sort(s).tolist())  # ascending, matches the gam side

emit("exp_angles", exp_angles)
emit("log_angles", log_angles)
"#,
            n = N,
            k = K,
            ncases = N_CASES,
        ),
    );

    let gs_exp = py.vector("exp_angles");
    let gs_log = py.vector("log_angles");
    assert_eq!(gs_exp.len(), N_CASES * K, "geomstats exp angle count");
    assert_eq!(gs_log.len(), N_CASES * K, "geomstats log angle count");

    let exp_angle_diff = max_abs_diff(&gam_exp_angles, gs_exp);
    let log_angle_diff = max_abs_diff(&gam_log_angles, gs_log);

    eprintln!(
        "Gr({K},{N}) n_cases={N_CASES} | gam vs ANALYTIC truth: roundtrip_abs={max_roundtrip_abs:.3e} \
         roundtrip_frob={max_roundtrip_frob:.3e} exp_angle_vs_σ={max_exp_angle_truth_err:.3e} \
         log_σ_vs_σ={max_log_sigma_truth_err:.3e} isometry_dist_err={max_isometry_err:.3e} | \
         geomstats ground-truth cross-check: exp_endpoint_diff={exp_angle_diff:.3e} \
         log_recovered_diff={log_angle_diff:.3e}"
    );

    // ===================== PRIMARY: gam vs ANALYTIC geodesic axioms =========
    // The Grassmann exp/log are exact closed forms; their defining properties
    // are derivable a priori from the input spectrum σ that this test builds.
    // gam must satisfy them to f64 linear-algebra precision with NO reference
    // to any external tool. 1e-10 / 1e-9 leave a sane margin above machine
    // epsilon while still catching any real SVD/trig/inverse defect.

    // (a) Involution: log_P(exp_P(v)) == v.
    assert!(
        max_roundtrip_abs < 1e-10,
        "gam Grassmann exp/log round-trip component error too large: {max_roundtrip_abs:.3e}"
    );
    assert!(
        max_roundtrip_frob < 1e-9,
        "gam Grassmann exp/log round-trip Frobenius error too large: {max_roundtrip_frob:.3e}"
    );

    // (b) Principal-angle law θ = σ: the angles between P and exp_P(v), and the
    // singular spectrum recovered by log, both equal the analytic input σ.
    assert!(
        max_exp_angle_truth_err < 1e-9,
        "gam exp-endpoint principal angles deviate from analytic spectrum σ: {max_exp_angle_truth_err:.3e}"
    );
    assert!(
        max_log_sigma_truth_err < 1e-9,
        "gam log-recovered singular spectrum deviates from analytic σ: {max_log_sigma_truth_err:.3e}"
    );

    // (c) Isometry: geodesic distance dist(P, exp_P(v)) = ‖v‖_F = ‖σ‖₂.
    assert!(
        max_isometry_err < 1e-9,
        "gam Grassmann geodesic violates the metric isometry ‖log(exp(v))‖ = ‖v‖: {max_isometry_err:.3e}"
    );

    // ===================== GROUND-TRUTH CROSS-CHECK vs geomstats ============
    // The Grassmann geodesic is an exact closed form (no fit, no noise), so
    // geomstats is an independent computation of the SAME mathematical truth,
    // not a peer tool whose noisy fit we chase. We require gam to additionally
    // match geomstats' exact endpoint and log invariants to its LAPACK-SVD
    // noise floor (1e-9) — a sharper, redundant confirmation of the axioms
    // above, not the primary claim.
    assert!(
        exp_angle_diff < 1e-9,
        "gam exp-endpoint principal angles disagree with geomstats ground truth: {exp_angle_diff:.3e}"
    );
    assert!(
        log_angle_diff < 1e-9,
        "gam log-recovered principal angles disagree with geomstats ground truth: {log_angle_diff:.3e}"
    );
}
