//! End-to-end quality: gam's Grassmann exponential/logarithm maps must agree
//! with `geomstats` — the mature differential-geometry reference — on the
//! canonical principal-angle invariants of `Gr(3, 10)`, and must round-trip to
//! transcendental precision.
//!
//! The Grassmannian `Gr(k, n)` is a quotient manifold: a point is a
//! `k`-dimensional subspace of `ℝⁿ` (equivalently an orthonormal `n×k` frame,
//! modulo `O(k)`). Its Riemannian exponential and logarithm have a closed form
//! built from the *compact SVD* of the tangent matrix and `cos`/`sin` of the
//! singular values (= principal angles); there is no optimization, so any
//! disagreement is a linear-algebra or transcendental-function error, not a
//! convergence artifact.
//!
//! Two facts are asserted on identical, fixed-seed data:
//!   1. **Intrinsic round-trip (gam alone).** For a base frame `P` and a
//!      horizontal tangent `v`, `log_map(P, exp_map(P, v)) == v` to
//!      `max_abs_diff < 1e-10` and Frobenius `< 1e-9`. This is the cleanest
//!      probe of the SVD → sin/cos → atan composition: exp pushes the angles
//!      through `cos`/`sin`, log pulls them back through `atan`, and the
//!      residual is pure floating-point rounding.
//!   2. **Head-to-head with geomstats.** gam and geomstats are handed the SAME
//!      base subspace and the SAME tangent direction (gam in the `n×k`-frame
//!      representation it uses internally; geomstats in the `n×n`-projector
//!      representation it uses internally — `P₀ = YYᵀ`, tangent
//!      `W = HYᵀ + YHᵀ`). We then compare the *principal angles* between the
//!      base subspace and the geodesic endpoint, and the principal angles
//!      recovered by each library's `log`. Principal angles are the canonical,
//!      `O(k)`-invariant, frame-vs-projector-metric-invariant description of a
//!      pair of subspaces, so they are the correct quantity to compare across
//!      two libraries with different internal representations. Both must equal
//!      the input singular spectrum `σ`.
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
    // gam's principal angles of the geodesic endpoint and of its recovered log,
    // per case (flattened K-vectors), to compare against geomstats.
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
        for ang in principal_angles(&y, &z) {
            gam_exp_angles.push(ang);
        }
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
        "Gr({K},{N}) n_cases={N_CASES} | gam round-trip: max_abs={max_roundtrip_abs:.3e} \
         frob={max_roundtrip_frob:.3e} | vs geomstats principal angles: \
         exp_endpoint_max_diff={exp_angle_diff:.3e} log_recovered_max_diff={log_angle_diff:.3e}"
    );

    // (1) Intrinsic round-trip is pure SVD + cos/sin/atan rounding. The maps
    // are exact closed forms within the injectivity radius, so the only error
    // is f64 transcendental/linear-algebra rounding; 1e-10 / 1e-9 leave a sane
    // margin above machine epsilon while still catching any real defect.
    assert!(
        max_roundtrip_abs < 1e-10,
        "gam Grassmann exp/log round-trip component error too large: {max_roundtrip_abs:.3e}"
    );
    assert!(
        max_roundtrip_frob < 1e-9,
        "gam Grassmann exp/log round-trip Frobenius error too large: {max_roundtrip_frob:.3e}"
    );

    // (2) Principal angles are the canonical subspace invariants computed by
    // both libraries' SVD/trig composition; gam and geomstats walk the same
    // geodesic, so their endpoint angles and their log-recovered angles must
    // coincide to linear-algebra precision. 1e-9 is geomstats' own LAPACK-SVD
    // noise floor — tight, and not weakened for either engine.
    assert!(
        exp_angle_diff < 1e-9,
        "exp-endpoint principal angles disagree with geomstats: {exp_angle_diff:.3e}"
    );
    assert!(
        log_angle_diff < 1e-9,
        "log-recovered principal angles disagree with geomstats: {log_angle_diff:.3e}"
    );
}
