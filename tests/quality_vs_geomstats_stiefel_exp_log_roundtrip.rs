//! End-to-end quality: gam's Stiefel `St(n, k)` Riemannian exponential must
//! agree with `geomstats` — the mature, standard differential-geometry library
//! — on the *same* base points and tangent vectors, and gam's exponential must
//! be the genuine inverse of geomstats' logarithm.
//!
//! gam implements the Edelman–Arias–Smith closed-form geodesic for the
//! canonical metric on `St(8, 3)`: with `A = YᵀΔ`, compact QR `(I − YYᵀ)Δ = QR`,
//! `Exp_Y(Δ) = [Y Q]·exp([[A,−Rᵀ],[R,0]])·[[I_k],[0]]` (see
//! `src/geometry/stiefel.rs`). geomstats' `StiefelCanonicalMetric` implements the
//! identical construction. Because the formula is closed-form (a `2k×2k` matrix
//! exponential then a QR projection), any disagreement is a real bug in gam's
//! block-matrix assembly, matrix exponential, or QR — not an optimization
//! artifact.
//!
//! gam deliberately refuses a closed-form Stiefel `log_map` for `k > 1` (no
//! elementary inverse exists; `src/geometry/stiefel.rs` returns
//! `GeometryError::Unsupported` rather than a wrong projected difference), so the
//! round-trip cannot be closed *inside* gam. We therefore close it through the
//! mature comparator: `||geomstats.log(gam.exp(Y, v)) − v||_F` must collapse to
//! geomstats' own iterative-log convergence floor, which confirms gam's
//! exponential is the true geometric inverse of geomstats' logarithm. We also
//! assert head-to-head that gam's exp output matches geomstats' exp output
//! elementwise. Identical fixed-seed data is fed to both engines.

use gam::geometry::{RiemannianManifold, StiefelManifold};
use gam::test_support::reference::{Column, max_abs_diff, run_python};
use ndarray::Array1;

const N: usize = 8;
const K: usize = 3;
const AMBIENT: usize = N * K; // 24, row-major flatten vec[i*K + j] = M[i, j]

/// Deterministic xorshift64* PRNG so gam and the CSV handed to geomstats are
/// built from one fixed bit-stream (no external rng-crate dependency, no
/// nondeterminism, no `env`).
struct Rng(u64);
impl Rng {
    fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        x.wrapping_mul(0x2545_f491_4f6c_dd1d)
    }
    /// Uniform in (-1, 1).
    fn unit(&mut self) -> f64 {
        // 53-bit mantissa fraction in [0, 1), mapped to (-1, 1).
        let frac = (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64;
        2.0 * frac - 1.0
    }
}

/// Thin QR of an `N×K` matrix (classical Gram–Schmidt with sign fixed so the
/// `R` diagonal is non-negative), returning an orthonormal `N×K` frame `Y`
/// flattened row-major. Used only to manufacture a valid St(8, 3) base point.
fn orthonormal_frame(rng: &mut Rng) -> Array1<f64> {
    let mut cols: Vec<Vec<f64>> = Vec::with_capacity(K);
    for _ in 0..K {
        let mut v: Vec<f64> = (0..N).map(|_| rng.unit()).collect();
        for q in &cols {
            let proj: f64 = (0..N).map(|i| q[i] * v[i]).sum();
            for i in 0..N {
                v[i] -= proj * q[i];
            }
        }
        let nrm: f64 = v.iter().map(|x| x * x).sum::<f64>().sqrt();
        for x in &mut v {
            *x /= nrm;
        }
        cols.push(v);
    }
    let mut y = Array1::<f64>::zeros(AMBIENT);
    for i in 0..N {
        for (j, col) in cols.iter().enumerate() {
            y[i * K + j] = col[i];
        }
    }
    y
}

#[test]
fn gam_stiefel_exp_matches_geomstats_and_inverts_its_log() {
    let manifold = StiefelManifold::new(K, N).expect("St(8, 3) is a valid frame manifold");

    // Tangent norms spanning the requested dynamic range [1e-7, 0.5]: a
    // near-zero step (exp must be ~identity) up to a moderate geodesic.
    let target_norms = [1e-7_f64, 1e-4, 1e-2, 0.1, 0.3, 0.5];

    let mut rng = Rng(0x5EED_1234_ABCD_0001);
    let n_samples = target_norms.len();

    // Stacked layout: each sample contributes AMBIENT consecutive rows carrying
    // its base point (yflat) and tangent (vflat). sample/entry index columns let
    // Python reshape back to (n_samples, N, K). Identical bytes feed both engines.
    let mut sample_col = Vec::with_capacity(n_samples * AMBIENT);
    let mut entry_col = Vec::with_capacity(n_samples * AMBIENT);
    let mut yflat = Vec::with_capacity(n_samples * AMBIENT);
    let mut vflat = Vec::with_capacity(n_samples * AMBIENT);

    // gam side: collect exp outputs per sample for the head-to-head comparison,
    // and remember the (Y, v) so we can reuse them in metrics.
    let mut gam_exp_all = Vec::with_capacity(n_samples * AMBIENT);
    let mut v_all = Vec::with_capacity(n_samples * AMBIENT);

    for (s, &target) in target_norms.iter().enumerate() {
        let y = orthonormal_frame(&mut rng);

        // Raw ambient vector, projected onto the tangent space at Y (gam's own
        // projection), then rescaled to the exact target tangent norm.
        let raw: Array1<f64> = (0..AMBIENT).map(|_| rng.unit()).collect();
        let tangent = manifold
            .project_tangent(y.view(), raw.view())
            .expect("project raw vector onto T_Y St(8, 3)");
        let nrm = tangent.dot(&tangent).sqrt();
        assert!(nrm > 1e-12, "projected tangent collapsed to zero");
        let v: Array1<f64> = &tangent * (target / nrm);

        // gam exponential of this exact (Y, v).
        let q_gam = manifold
            .exp_map(y.view(), v.view())
            .expect("gam Stiefel exp_map");
        assert_eq!(q_gam.len(), AMBIENT);

        for e in 0..AMBIENT {
            sample_col.push(s as f64);
            entry_col.push(e as f64);
            yflat.push(y[e]);
            vflat.push(v[e]);
            gam_exp_all.push(q_gam[e]);
            v_all.push(v[e]);
        }
    }

    // ---- geomstats reference: identical Y and v, canonical metric ----------
    // geomstats Stiefel(n, p) defaults to the canonical metric; signatures are
    // metric.exp(tangent_vec, base_point) and metric.log(point, base_point).
    // We emit: (a) its exp output, to compare head-to-head with gam, and
    // (b) log(gam_exp, Y), to confirm gam's exp inverts geomstats' log.
    let r = run_python(
        &[
            Column::new("sample", &sample_col),
            Column::new("entry", &entry_col),
            Column::new("yflat", &yflat),
            Column::new("vflat", &vflat),
            Column::new("gam_exp", &gam_exp_all),
        ],
        r#"
import numpy as np
import geomstats.backend as gs
from geomstats.geometry.stiefel import Stiefel

N, K = 8, 3
AMB = N * K
space = Stiefel(N, K)
metric = space.metric

sample = np.asarray(df["sample"], dtype=int)
entry  = np.asarray(df["entry"],  dtype=int)
yflat  = np.asarray(df["yflat"],  dtype=float)
vflat  = np.asarray(df["vflat"],  dtype=float)
gflat  = np.asarray(df["gam_exp"], dtype=float)

n_samples = int(sample.max()) + 1
geo_exp = np.zeros((n_samples, AMB))
geo_log_of_gam = np.zeros((n_samples, AMB))

for s in range(n_samples):
    mask = sample == s
    # rows are already ordered by entry within a sample, but sort to be safe
    order = np.argsort(entry[mask])
    yv = yflat[mask][order]
    vv = vflat[mask][order]
    gv = gflat[mask][order]
    Y = gs.array(yv.reshape(N, K))
    V = gs.array(vv.reshape(N, K))
    Q = gs.array(gv.reshape(N, K))   # gam's exp(Y, V)

    ge = metric.exp(V, Y)            # geomstats exp(tangent, base)
    gl = metric.log(Q, Y)            # geomstats log(gam_exp, base) -> should == V
    geo_exp[s] = np.asarray(ge, dtype=float).reshape(-1)
    geo_log_of_gam[s] = np.asarray(gl, dtype=float).reshape(-1)

emit("geo_exp", geo_exp.reshape(-1))
emit("geo_log_of_gam", geo_log_of_gam.reshape(-1))
"#,
    );

    let geo_exp = r.vector("geo_exp");
    let geo_log = r.vector("geo_log_of_gam");
    assert_eq!(geo_exp.len(), n_samples * AMBIENT, "geomstats exp length");
    assert_eq!(geo_log.len(), n_samples * AMBIENT, "geomstats log length");

    // Metric 1 (head-to-head): gam's exp vs geomstats' exp, elementwise on the
    // flattened frames over all samples.
    let exp_max_abs = max_abs_diff(&gam_exp_all, geo_exp);

    // Metric 2 (round-trip through the mature comparator): geomstats.log of
    // gam's exp must recover the original tangent v. Frobenius norm of the
    // residual, summed over all samples (each frame is one matrix).
    let mut frob_sq = 0.0;
    let mut log_max_abs = 0.0_f64;
    for i in 0..geo_log.len() {
        let d = geo_log[i] - v_all[i];
        frob_sq += d * d;
        log_max_abs = log_max_abs.max(d.abs());
    }
    let roundtrip_frob = frob_sq.sqrt();

    eprintln!(
        "Stiefel St(8,3) exp vs geomstats: samples={n_samples} \
         exp_max_abs={exp_max_abs:.3e} roundtrip_frob={roundtrip_frob:.3e} \
         roundtrip_max_abs={log_max_abs:.3e}"
    );

    // Both engines implement the identical closed-form canonical-metric geodesic
    // (the 2k×2k block exponential `[[A,−Rᵀ],[R,0]]` then a QR projection — see
    // `_StiefelLogSolver`/`StiefelCanonicalMetric.exp` in geomstats and
    // `src/geometry/stiefel.rs`), so the only difference between the two exp
    // outputs is floating-point round-off in `expm` and `qr`. Agreement to
    // ~1e-10 is the correct head-to-head bar; a wider divergence signals a
    // genuine block-assembly / exp / QR bug.
    assert!(
        exp_max_abs < 1e-10,
        "gam Stiefel exp diverges from geomstats: max_abs_diff={exp_max_abs:.3e}"
    );

    // The round-trip is closed through geomstats' *iterative* Stiefel logarithm
    // (`_StiefelLogSolver`), which terminates as soon as its internal residual
    // norm falls below its convergence tolerance `tol = 1e-8` (with
    // `imag_tol = 1e-6`, `max_iter = 500`). The recovered tangent therefore
    // carries the solver's own ~1e-8 convergence floor, accumulated in
    // quadrature across the `n_samples` frames. Asserting tighter than that
    // floor would test geomstats' iteration count, not gam: the principled bar
    // is one comfortably above the comparator's documented `tol` yet far below
    // the O(1e-1)+ residual a real exp/QR/block-assembly bug would produce, so
    // it still catches any genuine inverse-relation failure in gam's exp.
    assert!(
        roundtrip_frob < 1e-6,
        "gam exp is not the inverse of geomstats log: ||log(exp(v)) - v||_F={roundtrip_frob:.3e}"
    );
}
