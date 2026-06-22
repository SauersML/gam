//! End-to-end quality: gam's Poincaré-ball tangent-space exp/log maps at the
//! origin must satisfy the INTRINSIC geometric axioms of the hyperbolic
//! manifold — not merely reproduce a peer tool's floating-point output.
//!
//! OBJECTIVE METRIC ASSERTED (no reference needed for the pass criterion):
//!   1. ROUND-TRIP IDENTITY (manifold axiom): `log_0(exp_0(v)) = v`. With the
//!      closed forms `exp_0(v) = tanh(√k|v|)/(√k|v|)·v` and
//!      `log_0(y) = atanh(√k|y|)/(√k|y|)·y`, `log_0 ∘ exp_0` is the identity in
//!      exact arithmetic; away from the saturation boundary the only deviation
//!      is f64 rounding of tanh/atanh — a handful of ULPs.
//!   2. RADIAL GEODESIC ISOMETRY (manifold axiom): the exp map at the origin is
//!      a radial isometry in the RIEMANNIAN tangent norm, so
//!      `d_c(0, exp_0(v)) = λ_0|v| = 2|v|` exactly (at the origin the conformal
//!      factor is `λ_0 = 2/(1-0) = 2`). This pins the geodesic distance, the
//!      exp map and the metric together against an independently-known scalar
//!      (the Riemannian tangent norm) — no reference involved.
//!   3. CLOSED-FORM GROUND TRUTH: `exp_0(v)` equals the textbook analytic
//!      coefficient `tanh(√k|v|)/(√k|v|)` recomputed independently inside the
//!      test from `v`'s own norm. Matching that analytic expression is a
//!      correctness claim against mathematical ground truth, not "same output
//!      as geomstats".
//!
//! geomstats (the mature differential-geometry reference) is still COMPUTED and
//! its own axiom errors are printed for context, demoted to a BASELINE-TO-BEAT:
//! we additionally assert gam's round-trip / closed-form errors are no worse
//! than geomstats' (match-or-beat on accuracy). We deliberately do NOT make
//! "gam's exp image ≈ geomstats' exp image" the pass criterion — reproducing a
//! peer tool's noisy float output proves nothing about correctness; satisfying
//! the manifold axioms to float precision does.
//!
//! We fix curvature `c = -1` (`k = 1`, ball radius `1/√k = 1`) — the default
//! `geomstats.geometry.poincare_ball.PoincareBall(dim).metric` — feed BOTH
//! engines the *same* fixed-seed random tangent vectors `v` whose norms span
//! `[1e-8, 0.95]` (strictly below the saturation radius) across dimensions
//! `d ∈ {2, 4, 8}`.
//!
//! No data-driven fitting is involved; this checks the manifold primitives that
//! back `gamfit.PoincareAtoms` against their defining geometric axioms.

use gam::geometry::poincare::{exp_origin, log_origin, poincare_distance};
use gam::test_support::reference::{Column, max_abs_diff, relative_l2, run_python};
use ndarray::Array1;

/// Curvature exposed to both engines. `c = -1` ⇒ `k = 1` ⇒ unit ball, which is
/// the geomstats Poincaré-ball default and makes the closed forms collapse to
/// the textbook `tanh`/`atanh` of the raw norm.
const CURVATURE: f64 = -1.0;

/// Deterministic LCG (Numerical Recipes constants) so the test is fully
/// reproducible and the *identical* draws are shipped to both engines without
/// depending on any RNG crate or platform float formatting on the Python side.
struct Lcg {
    state: u64,
}

impl Lcg {
    fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    /// Uniform in [0, 1).
    fn next_unit(&mut self) -> f64 {
        self.state = self
            .state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        // Top 53 bits → [0, 1).
        ((self.state >> 11) as f64) / ((1u64 << 53) as f64)
    }

    /// Symmetric uniform in (-1, 1).
    fn next_signed(&mut self) -> f64 {
        2.0 * self.next_unit() - 1.0
    }
}

/// Build one tangent vector of dimension `d` with a *prescribed* Euclidean norm
/// `target_norm`: draw a random direction, normalise it, then scale. This lets
/// us deterministically sweep the radial coordinate across the full interior
/// `[1e-8, 0.95]` (the regime where the closed forms are exact) instead of
/// leaving the norm to chance.
fn tangent_with_norm(rng: &mut Lcg, d: usize, target_norm: f64) -> Vec<f64> {
    let mut v: Vec<f64> = (0..d).map(|_| rng.next_signed()).collect();
    let raw = v.iter().map(|x| x * x).sum::<f64>().sqrt();
    // raw is ~O(1) and almost surely > 0 for these seeds; guard anyway.
    let scale = if raw > 0.0 { target_norm / raw } else { 0.0 };
    for x in v.iter_mut() {
        *x *= scale;
    }
    v
}

/// Euclidean norm of a slice.
fn norm(v: &[f64]) -> f64 {
    v.iter().map(|x| x * x).sum::<f64>().sqrt()
}

#[test]
fn gam_poincare_exp_log_satisfies_manifold_axioms() {
    // ---- fixed-seed bank of tangent vectors at the origin -----------------
    // Dimensions and, per dimension, a geometric-ish sweep of target norms
    // spanning the interior from the origin floor to 0.95·radius (radius = 1).
    let dims = [2usize, 4, 8];
    let target_norms = [
        1.0e-8, 1.0e-6, 1.0e-4, 1.0e-3, 1.0e-2, 5.0e-2, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95,
    ];
    let radius = 1.0 / (-CURVATURE).sqrt();
    let sqrt_k = (-CURVATURE).sqrt();
    for &nrm in &target_norms {
        assert!(
            nrm < 0.95 * radius + 1e-12,
            "target norm {nrm} must stay below 0.95·radius (={})",
            0.95 * radius
        );
    }

    let mut rng = Lcg::new(0x5EED_C0FF_EE15_600D);
    // Flatten every vector into a single padded matrix so it can ride the CSV
    // wire to Python: row = one tangent vector, columns c0..c7 (zeros beyond
    // the row's own dimension), plus the row's true dimension `dim` so Python
    // reads only the first `dim` entries of each padded row.
    let max_d = *dims.iter().max().expect("dims non-empty");
    let mut vectors: Vec<Vec<f64>> = Vec::new();
    let mut row_dim: Vec<f64> = Vec::new();
    for &d in &dims {
        for &nrm in &target_norms {
            vectors.push(tangent_with_norm(&mut rng, d, nrm));
            row_dim.push(d as f64);
        }
    }
    let n = vectors.len();

    // ---- gam: per-vector manifold-axiom checks ---------------------------
    // For each tangent v we evaluate exp_0(v), then assert the three intrinsic
    // properties directly against quantities recomputed from v alone:
    //   * round-trip:   log_0(exp_0(v)) == v
    //   * radial isometry: d_c(0, exp_0(v)) == |v|
    //   * closed form:  exp_0(v) == (tanh(sqrt_k*|v|)/(sqrt_k*|v|)) * v
    let mut gam_exp_flat = vec![0.0f64; n * max_d];
    let mut gam_roundtrip_l2_max = 0.0f64;
    let mut gam_roundtrip_maxabs = 0.0f64;
    let mut gam_isometry_maxabs = 0.0f64;
    let mut gam_closedform_maxabs = 0.0f64;
    let origin_max = Array1::<f64>::zeros(max_d);
    for (i, v) in vectors.iter().enumerate() {
        let vv = Array1::from(v.clone());
        let y = exp_origin(vv.view(), CURVATURE).expect("gam exp_origin");
        let back = log_origin(y.view(), CURVATURE).expect("gam log_origin");

        // (1) round-trip identity log_0(exp_0(v)) == v.
        let l2 = relative_l2(back.as_slice().expect("contig"), v);
        let mab = max_abs_diff(back.as_slice().expect("contig"), v);
        gam_roundtrip_l2_max = gam_roundtrip_l2_max.max(l2);
        gam_roundtrip_maxabs = gam_roundtrip_maxabs.max(mab);

        // (2) radial geodesic isometry: d_c(0, exp_0(v)) == λ_0|v| == 2|v|.
        // The exp map at the origin is a radial isometry in the RIEMANNIAN
        // (not Euclidean) tangent norm: the conformal factor is
        // λ_x = 2/(1 - k|x|²), so at the origin λ_0 = 2/(1-0) = 2 and the
        // Riemannian length of the tangent v is λ_0|v| = 2|v|. Concretely
        // |exp_0(v)| = tanh(s)/√k (s = √k|v|), so the standard distance
        // d = 2·asinh(√δ)/√k = 2s/√k = 2|v| — matching geomstats'
        // `PoincareBallMetric.dist` (acosh(1+2δ)) and the in-crate textbook
        // formula test, not the bare Euclidean |v|.
        let d_geo = poincare_distance(
            origin_max.slice(ndarray::s![0..v.len()]),
            y.view(),
            CURVATURE,
        )
        .expect("gam poincare_distance");
        gam_isometry_maxabs = gam_isometry_maxabs.max((d_geo - 2.0 * norm(v)).abs());

        // (3) closed-form ground truth: exp_0(v) == phi * v with the textbook
        // radial coefficient phi = tanh(sqrt_k*|v|)/(sqrt_k*|v|), recomputed
        // here independently from v's own norm (no reference tool involved).
        let nv = norm(v);
        let s = sqrt_k * nv;
        // s -> 0 limit of tanh(s)/s is 1; for the tiny norms in the sweep this
        // is the well-conditioned branch.
        let phi = if s > 0.0 { s.tanh() / s } else { 1.0 };
        for j in 0..v.len() {
            let analytic = phi * v[j];
            gam_closedform_maxabs = gam_closedform_maxabs.max((y[j] - analytic).abs());
        }

        for j in 0..v.len() {
            gam_exp_flat[i * max_d + j] = y[j];
        }
    }

    // ---- ship the SAME vectors (flattened) to geomstats (BASELINE) --------
    // geomstats is fit purely to obtain a peer's own axiom errors as a
    // match-or-beat baseline; it is NOT the pass criterion. One column per
    // slot c0..c{max_d-1}; rows shorter than max_d are zero-padded and Python
    // only reads the first `dim` entries of each row.
    let mut columns: Vec<Column<'_>> = Vec::with_capacity(max_d + 1);
    let mut col_storage: Vec<Vec<f64>> = Vec::with_capacity(max_d);
    for j in 0..max_d {
        let col: Vec<f64> = (0..n)
            .map(|i| {
                if j < vectors[i].len() {
                    vectors[i][j]
                } else {
                    0.0
                }
            })
            .collect();
        col_storage.push(col);
    }
    let col_names: Vec<String> = (0..max_d).map(|j| format!("c{j}")).collect();
    for j in 0..max_d {
        columns.push(Column::new(
            col_names[j].as_str(),
            col_storage[j].as_slice(),
        ));
    }
    columns.push(Column::new("dim", &row_dim));

    let py = run_python(
        &columns,
        r#"
import numpy as np
from geomstats.geometry.poincare_ball import PoincareBall

# Column-major rebuild of every tangent vector (rows shorter than max_d are
# zero-padded; only the first `dim` entries are physically meaningful).
cols = [k for k in df.keys()] if isinstance(df, dict) else list(df.columns)
slot_names = sorted([c for c in cols if c.startswith('c')], key=lambda s: int(s[1:]))
max_d = len(slot_names)
dim_arr = np.asarray(df['dim']).astype(int)
n = len(dim_arr)
mat = np.column_stack([np.asarray(df[s], dtype=float) for s in slot_names])

exp_flat = np.zeros((n, max_d), dtype=float)
self_l2 = []
self_maxabs = []
closed_maxabs = []

# Group rows by dimension so we hand each PoincareBall metric vectors of its
# own intrinsic dimension. The default-scale PoincareBall is the curvature
# c = -1 (k = 1) unit ball — the same metric gam evaluates. `PoincareBall`
# attaches its PoincareBallMetric in __init__; .metric.exp / .metric.log are
# the closed-form tanh / artanh tangent maps.
for d in sorted(set(int(x) for x in dim_arr)):
    idx = np.where(dim_arr == d)[0]
    metric = PoincareBall(d).metric
    V = mat[idx, :d]                              # tangent vectors at the origin
    # exp_0(v) and log_0(exp_0(v)); base_point is the origin for every row.
    # geomstats signature: exp(tangent_vec, base_point), log(point, base_point).
    origin = np.zeros((len(idx), d), dtype=float)
    Y = np.asarray(metric.exp(V, origin), dtype=float)   # points on the ball
    B = np.asarray(metric.log(Y, origin), dtype=float)   # back to tangent space
    exp_flat[np.ix_(idx, range(d))] = Y
    # geomstats' OWN axiom errors per row (its baseline against the same axioms).
    for r in range(len(idx)):
        v = V[r]
        b = B[r]
        nv = np.sqrt(np.sum(v * v))
        denom = max(nv, 1e-300)
        # round-trip identity
        self_l2.append(np.sqrt(np.sum((b - v) ** 2)) / denom)
        self_maxabs.append(np.max(np.abs(b - v)))
        # closed-form ground truth: phi = tanh(s)/s, s = nv (k = 1).
        s = nv
        phi = (np.tanh(s) / s) if s > 0.0 else 1.0
        closed_maxabs.append(np.max(np.abs(Y[r, :d] - phi * v)))

emit("exp_flat", exp_flat.reshape(-1))
emit("self_l2_max", [float(np.max(self_l2))])
emit("self_maxabs_max", [float(np.max(self_maxabs))])
emit("closed_maxabs_max", [float(np.max(closed_maxabs))])
"#,
    );

    let geo_exp_flat = py.vector("exp_flat");
    let geo_self_l2_max = py.scalar("self_l2_max");
    let geo_self_maxabs_max = py.scalar("self_maxabs_max");
    let geo_closed_maxabs_max = py.scalar("closed_maxabs_max");

    assert_eq!(
        geo_exp_flat.len(),
        n * max_d,
        "geomstats exp length mismatch"
    );

    // Cross-engine rel-L2 on the exp map: printed for CONTEXT only, never the
    // pass criterion (matching a peer tool's float output is not a quality
    // claim).
    let cross_exp_rel_l2 = relative_l2(&gam_exp_flat, geo_exp_flat);

    eprintln!(
        "poincare exp/log manifold axioms (c={CURVATURE}, n={n} vectors over d={dims:?}):\n  \
         gam       round-trip: L2_rel_max={gam_roundtrip_l2_max:.3e} maxabs={gam_roundtrip_maxabs:.3e}\n  \
         gam       radial isometry |d(0,exp(v))-|v|| max={gam_isometry_maxabs:.3e}\n  \
         gam       closed-form maxabs={gam_closedform_maxabs:.3e}\n  \
         geomstats round-trip: L2_rel_max={geo_self_l2_max:.3e} maxabs={geo_self_maxabs_max:.3e} (baseline)\n  \
         geomstats closed-form maxabs={geo_closed_maxabs_max:.3e} (baseline)\n  \
         context: cross-engine exp_rel_l2={cross_exp_rel_l2:.3e} (NOT asserted)"
    );

    // --- PRIMARY OBJECTIVE BOUND 1: round-trip identity log∘exp == v ---
    // log∘exp is the identity in exact arithmetic; away from the saturation
    // boundary (max norm 0.95 < 1) the only error is f64 rounding of
    // tanh/atanh and the radial scaling, a handful of ULPs. 1e-13 max-abs /
    // 1e-12 relative-L2 is generous against ULP noise yet tight enough that a
    // missing √k factor or a tanh/atanh swap (errors of order 0.1–1) trips it.
    assert!(
        gam_roundtrip_maxabs < 1.0e-13,
        "gam log∘exp must recover v (round-trip axiom): maxabs={gam_roundtrip_maxabs:.3e}"
    );
    assert!(
        gam_roundtrip_l2_max < 1.0e-12,
        "gam log∘exp relative-L2 too large (round-trip axiom): {gam_roundtrip_l2_max:.3e}"
    );

    // --- PRIMARY OBJECTIVE BOUND 2: radial geodesic isometry ---
    // The origin exp map is a radial isometry in the RIEMANNIAN tangent norm,
    // so the geodesic distance from the origin to exp_0(v) must equal the
    // Riemannian tangent length λ_0|v| = 2|v| exactly (the conformal factor at
    // the origin is λ_0 = 2/(1-0) = 2). This pins exp, log-free, the geodesic
    // distance, and the metric simultaneously against 2|v| (recomputed from v
    // alone). acosh near arg=1 amplifies ULPs by ~1/√(2(arg-1)); the worst case
    // here (|v|=1e-8) gives ~1e-12 slack, so 1e-10 is a safe absolute bar that
    // still catches any √k or formula error.
    assert!(
        gam_isometry_maxabs < 1.0e-10,
        "gam exp map must be a radial geodesic isometry d_c(0,exp(v))==2|v|: \
         maxabs deviation={gam_isometry_maxabs:.3e}"
    );

    // --- PRIMARY OBJECTIVE BOUND 3: closed-form ground truth ---
    // exp_0(v) must equal the textbook analytic coefficient applied to v. The
    // coefficient is recomputed inside the test from v's own norm — this is a
    // correctness check against mathematical ground truth, not peer output.
    assert!(
        gam_closedform_maxabs < 1.0e-13,
        "gam exp_0(v) must match the analytic tanh closed form: maxabs={gam_closedform_maxabs:.3e}"
    );

    // --- BASELINE-TO-BEAT: gam must be at least as accurate as geomstats ---
    // The mature reference also satisfies the same axioms; gam's axiom errors
    // must be no worse than geomstats' (with a 1-ULP-ish additive slack so a
    // tie at the float floor never flips). This is a match-or-beat ACCURACY
    // claim on an objective metric — not "gam reproduces geomstats' output".
    let slack = 1.0e-14;
    assert!(
        gam_roundtrip_maxabs <= geo_self_maxabs_max + slack,
        "gam round-trip error {gam_roundtrip_maxabs:.3e} should be <= geomstats \
         baseline {geo_self_maxabs_max:.3e} (+slack)"
    );
    assert!(
        gam_roundtrip_l2_max <= geo_self_l2_max + slack,
        "gam round-trip rel-L2 {gam_roundtrip_l2_max:.3e} should be <= geomstats \
         baseline {geo_self_l2_max:.3e} (+slack)"
    );
    assert!(
        gam_closedform_maxabs <= geo_closed_maxabs_max + slack,
        "gam closed-form error {gam_closedform_maxabs:.3e} should be <= geomstats \
         baseline {geo_closed_maxabs_max:.3e} (+slack)"
    );
}
