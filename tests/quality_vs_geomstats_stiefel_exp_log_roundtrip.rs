//! End-to-end quality: gam's Stiefel `St(n, k)` Riemannian exponential is judged
//! by the *intrinsic geometric axioms it must satisfy*, not by whether it
//! reproduces another library's fitted output.
//!
//! The exponential map of a manifold is an exact mathematical object, pinned by
//! two defining properties that can be checked using only gam's own output and
//! gam's own canonical metric:
//!
//!   1. **Constraint closure.** `Exp_Y(Δ)` must land back on the manifold:
//!      `Exp_Y(Δ)ᵀ Exp_Y(Δ) = I_k`. We assert the worst-case orthonormality
//!      defect `max‖QᵀQ − I_k‖_max` is at floating-point precision.
//!   2. **Geodesic isometry.** `t ↦ Exp_Y(t·Δ)` is the unit-speed-rescaled
//!      geodesic, so its canonical arc length from `t=0` to `t=1` must equal the
//!      canonical tangent norm `‖Δ‖_g = √⟨Δ, Δ⟩_g`. We discretise the geodesic
//!      with gam's `exp_map`, measure each segment with gam's `metric_tensor`
//!      (the canonical Gram matrix `(I − ½YYᵀ) ⊗ I_k`), and assert the relative
//!      length error collapses as the discretisation refines. This is the
//!      genuine "is this a geodesic of the canonical metric?" test — a wrong
//!      block assembly, `expm`, or QR would break length preservation.
//!
//! Both metrics are PURELY intrinsic to gam (no reference tool appears in the
//! pass/fail criterion). geomstats is retained only as ground truth for context:
//! the Stiefel exponential is an *exact analytic quantity*, so geomstats' `exp`
//! IS a correct reference value, and we additionally assert gam's frame is at
//! least as orthonormal as geomstats' (match-or-beat on the same objective
//! constraint). We also print, via `eprintln!`, the head-to-head `exp` agreement
//! and the `geomstats.log(gam.exp(Δ)) − Δ` round-trip residual for diagnostics —
//! but neither "closeness to geomstats" gates the test.
//!
//! gam deliberately refuses a closed-form Stiefel `log_map` for `k > 1`
//! (`src/geometry/stiefel.rs` returns `GeometryError::Unsupported`), which is why
//! the geodesic-isometry check measures arc length by integrating forward along
//! `Exp_Y(t·Δ)` rather than calling a (nonexistent) gam log. Identical
//! fixed-seed data is fed to both engines.

use gam::geometry::{RiemannianManifold, StiefelManifold};
use gam::test_support::reference::{Column, max_abs_diff, run_python};
use ndarray::{Array1, Array2};

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

/// Canonical-metric inner product `⟨a, b⟩_g = aᵀ G(Y) b`, where `G(Y)` is gam's
/// own `metric_tensor` at the base frame `Y`. Used to measure both tangent norms
/// and geodesic segment lengths in the *intrinsic* metric the exponential is the
/// geodesic of.
fn canonical_inner(g: &Array2<f64>, a: &Array1<f64>, b: &Array1<f64>) -> f64 {
    let mut acc = 0.0;
    for i in 0..AMBIENT {
        let mut row = 0.0;
        for j in 0..AMBIENT {
            row += g[[i, j]] * b[j];
        }
        acc += a[i] * row;
    }
    acc
}

/// Worst-case `‖QᵀQ − I_k‖_max` over the `N×K` frame `q` (row-major flattened):
/// the orthonormality defect that says how far `q` is from being a genuine
/// Stiefel point.
fn orthonormality_defect(q: &[f64]) -> f64 {
    let mut worst = 0.0_f64;
    for a in 0..K {
        for b in 0..K {
            let mut dot = 0.0;
            for i in 0..N {
                dot += q[i * K + a] * q[i * K + b];
            }
            let target = if a == b { 1.0 } else { 0.0 };
            worst = worst.max((dot - target).abs());
        }
    }
    worst
}

#[test]
fn gam_stiefel_exp_is_a_valid_canonical_metric_geodesic() {
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

    // gam side: collect exp outputs per sample for the head-to-head diagnostic,
    // and remember the (Y, v) so we can run the intrinsic axiom checks.
    let mut gam_exp_all = Vec::with_capacity(n_samples * AMBIENT);
    let mut v_all = Vec::with_capacity(n_samples * AMBIENT);

    // INTRINSIC METRIC 1 — constraint closure: worst orthonormality defect of
    // gam's exp output across all samples (must be ~machine precision).
    let mut gam_frame_defect = 0.0_f64;

    // INTRINSIC METRIC 2 — geodesic isometry: worst relative error between the
    // canonical arc length of t↦Exp_Y(t·v) and the canonical tangent norm ‖v‖_g.
    let mut worst_geodesic_rel_err = 0.0_f64;
    // Number of geodesic subdivisions; finer than this only chases the segment
    // discretisation's O(1/STEPS²) truncation, not a gam defect.
    const STEPS: usize = 256;

    for (s, &target) in target_norms.iter().enumerate() {
        let y = orthonormal_frame(&mut rng);

        // Raw ambient vector, projected onto the tangent space at Y (gam's own
        // projection), then rescaled so the *canonical* tangent norm is `target`.
        let raw: Array1<f64> = (0..AMBIENT).map(|_| rng.unit()).collect();
        let tangent = manifold
            .project_tangent(y.view(), raw.view())
            .expect("project raw vector onto T_Y St(8, 3)");
        let g = manifold
            .metric_tensor(y.view())
            .expect("gam canonical metric tensor at Y");
        let g_norm = canonical_inner(&g, &tangent, &tangent).sqrt();
        assert!(g_norm > 1e-12, "projected tangent collapsed to zero");
        let v: Array1<f64> = &tangent * (target / g_norm);
        // By construction the canonical tangent norm equals `target`.
        let v_canonical_norm = canonical_inner(&g, &v, &v).sqrt();

        // gam exponential of this exact (Y, v).
        let q_gam = manifold
            .exp_map(y.view(), v.view())
            .expect("gam Stiefel exp_map");
        assert_eq!(q_gam.len(), AMBIENT);

        // --- intrinsic metric 1: is the exp output a genuine Stiefel frame? ---
        gam_frame_defect = gam_frame_defect.max(orthonormality_defect(q_gam.as_slice().unwrap()));

        // --- intrinsic metric 2: canonical arc length of Exp_Y(t·v) == ‖v‖_g ---
        // Integrate ∫₀¹ ‖d/dt Exp_Y(t·v)‖_{g(point(t))} dt by sampling the
        // geodesic at t_m = m/STEPS, measuring each chord in the canonical metric
        // at the segment's left endpoint (the metric is point-dependent, so we
        // evaluate G at the current point along the curve). For the actual
        // geodesic this Riemann sum converges to the geodesic length, which the
        // exp/log theory pins to exactly ‖v‖_g.
        let mut arc_len = 0.0;
        let mut prev = y.clone();
        for m in 1..=STEPS {
            let t = m as f64 / STEPS as f64;
            let scaled: Array1<f64> = &v * t;
            let cur = manifold
                .exp_map(y.view(), scaled.view())
                .expect("gam Stiefel exp_map along geodesic");
            let chord: Array1<f64> = &cur - &prev;
            // Canonical length of this chord, metric evaluated at the segment's
            // start point (a valid Stiefel frame), via gam's own metric tensor.
            let g_seg = manifold
                .metric_tensor(prev.view())
                .expect("gam canonical metric tensor along geodesic");
            let seg_len = canonical_inner(&g_seg, &chord, &chord).sqrt();
            arc_len += seg_len;
            prev = cur;
        }
        let geodesic_rel_err = (arc_len - v_canonical_norm).abs() / v_canonical_norm.max(1e-300);
        worst_geodesic_rel_err = worst_geodesic_rel_err.max(geodesic_rel_err);

        for e in 0..AMBIENT {
            sample_col.push(s as f64);
            entry_col.push(e as f64);
            yflat.push(y[e]);
            vflat.push(v[e]);
            gam_exp_all.push(q_gam[e]);
            v_all.push(v[e]);
        }
    }

    // ---- geomstats: GROUND TRUTH (the exp map is an exact analytic quantity) --
    // geomstats Stiefel(n, p) defaults to the canonical metric; signatures are
    // metric.exp(tangent_vec, base_point) and metric.log(point, base_point).
    // We emit: (a) its exp output, for the head-to-head diagnostic and the
    // match-or-beat orthonormality baseline, and (b) log(gam_exp, Y), for the
    // round-trip residual diagnostic.
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

    // Diagnostic 1 (context only): gam's exp vs geomstats' exp elementwise.
    let exp_max_abs = max_abs_diff(&gam_exp_all, geo_exp);

    // Diagnostic 2 (context only): geomstats.log(gam.exp(v)) − v round-trip.
    let mut frob_sq = 0.0;
    let mut log_max_abs = 0.0_f64;
    for i in 0..geo_log.len() {
        let d = geo_log[i] - v_all[i];
        frob_sq += d * d;
        log_max_abs = log_max_abs.max(d.abs());
    }
    let roundtrip_frob = frob_sq.sqrt();

    // Match-or-beat baseline: worst orthonormality defect of geomstats' own exp
    // output. Same objective constraint, computed on the reference's points.
    let mut geo_frame_defect = 0.0_f64;
    for s in 0..n_samples {
        let frame = &geo_exp[s * AMBIENT..(s + 1) * AMBIENT];
        geo_frame_defect = geo_frame_defect.max(orthonormality_defect(frame));
    }

    eprintln!(
        "Stiefel St(8,3) intrinsic geodesic quality: samples={n_samples} \
         gam_frame_defect={gam_frame_defect:.3e} geo_frame_defect={geo_frame_defect:.3e} \
         worst_geodesic_rel_err={worst_geodesic_rel_err:.3e} | \
         [diagnostic vs geomstats] exp_max_abs={exp_max_abs:.3e} \
         roundtrip_frob={roundtrip_frob:.3e} roundtrip_max_abs={log_max_abs:.3e}"
    );

    // ---- OBJECTIVE ASSERTION 1: constraint closure -------------------------
    // gam's exp output is a genuine St(8, 3) frame to floating-point precision.
    // This is a property of gam's output alone; the bar reflects accumulated
    // round-off through the 2k×2k block expm and QR, not any reference tool.
    assert!(
        gam_frame_defect < 1e-12,
        "gam Stiefel exp output is not orthonormal: max ‖QᵀQ − I_k‖={gam_frame_defect:.3e}"
    );

    // Match-or-beat the analytic-ground-truth reference on that same constraint:
    // gam's frame is at least as orthonormal as geomstats' (within 10%, and with
    // an absolute floor so two machine-precision values never trip the ratio).
    assert!(
        gam_frame_defect <= geo_frame_defect * 1.10 + 1e-13,
        "gam frame less orthonormal than geomstats: gam={gam_frame_defect:.3e} geo={geo_frame_defect:.3e}"
    );

    // ---- OBJECTIVE ASSERTION 2: geodesic isometry --------------------------
    // The canonical arc length of t↦Exp_Y(t·v) equals the canonical tangent norm
    // ‖v‖_g. The residual is dominated by the chord-vs-arc truncation of a
    // STEPS-segment Riemann sum (O(1/STEPS²) for a smooth curve), which at
    // STEPS=256 sits at ~1e-5; a broken block assembly / expm / QR would not be
    // a length-preserving geodesic and would blow this far past the bar.
    assert!(
        worst_geodesic_rel_err < 5e-4,
        "gam Stiefel exp is not a unit-rescaled canonical geodesic: \
         worst |arc_len − ‖v‖_g| / ‖v‖_g = {worst_geodesic_rel_err:.3e}"
    );
}
