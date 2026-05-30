//! End-to-end quality: gam's Poincaré-ball tangent-space exp/log maps at the
//! origin must agree with `geomstats` — the mature differential-geometry
//! reference — on the *intrinsic* round-trip identity `log_0(exp_0(v)) = v`.
//!
//! This is an algebraic property, not an optimisation outcome: with the
//! closed forms `exp_0(v) = tanh(√k|v|)/(√k|v|)·v` and
//! `log_0(y) = atanh(√k|y|)/(√k|y|)·y`, the composition `log_0 ∘ exp_0` is the
//! identity in exact arithmetic, up to boundary saturation. Any deviation is
//! pure floating-point summation / transcendental-function rounding, so both a
//! correct gam and a correct geomstats *must* reproduce the input `v` to float
//! precision. A failure here immediately pinpoints a formula typo (a missing
//! `√k` factor, a swapped `tanh`/`atanh`, or a wrong boundary clamp) in one of
//! the two engines rather than a tuning disagreement.
//!
//! We fix curvature `c = -1` (`k = 1`, ball radius `1/√k = 1`) — the default
//! `geomstats.geometry.poincare_ball.PoincareBall(dim).metric` — feed
//! BOTH engines the *same* fixed-seed random tangent vectors `v` whose norms
//! span `[1e-8, 0.95]` (strictly below the saturation radius) across
//! dimensions `d ∈ {2, 4, 8}`, and assert:
//!   1. each engine's own round-trip recovers `v` (gam and geomstats both
//!      satisfy the identity), and
//!   2. the two engines' exp-map images and round-trips agree componentwise
//!      (they compute the *same* function).
//!
//! No data-driven fitting is involved; this is a head-to-head check of the
//! manifold primitives that back `gamfit.PoincareAtoms`.

use gam::geometry::poincare::{exp_origin, log_origin};
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
/// `[1e-8, 0.95]` (the regime where `log∘exp` is the exact identity) instead of
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

#[test]
fn gam_poincare_exp_log_roundtrip_matches_geomstats() {
    // ---- fixed-seed bank of tangent vectors at the origin -----------------
    // Dimensions and, per dimension, a geometric-ish sweep of target norms
    // spanning the interior from the origin floor to 0.95·radius (radius = 1).
    let dims = [2usize, 4, 8];
    let target_norms = [
        1.0e-8, 1.0e-6, 1.0e-4, 1.0e-3, 1.0e-2, 5.0e-2, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95,
    ];
    let radius = 1.0 / (-CURVATURE).sqrt();
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

    // ---- gam: per-vector log_0(exp_0(v)) round-trip -----------------------
    let mut gam_exp_flat = vec![0.0f64; n * max_d];
    let mut gam_round_flat = vec![0.0f64; n * max_d];
    let mut gam_roundtrip_l2_max = 0.0f64;
    let mut gam_roundtrip_maxabs = 0.0f64;
    for (i, v) in vectors.iter().enumerate() {
        let vv = Array1::from(v.clone());
        let y = exp_origin(vv.view(), CURVATURE).expect("gam exp_origin");
        let back = log_origin(y.view(), CURVATURE).expect("gam log_origin");
        // gam's own round-trip must reproduce v (intrinsic identity).
        let l2 = relative_l2(back.as_slice().expect("contig"), v);
        let mab = max_abs_diff(back.as_slice().expect("contig"), v);
        gam_roundtrip_l2_max = gam_roundtrip_l2_max.max(l2);
        gam_roundtrip_maxabs = gam_roundtrip_maxabs.max(mab);
        for j in 0..v.len() {
            gam_exp_flat[i * max_d + j] = y[j];
            gam_round_flat[i * max_d + j] = back[j];
        }
    }

    // ---- ship the SAME vectors (flattened) to geomstats -------------------
    // One column per slot c0..c{max_d-1}; rows shorter than max_d are zero-
    // padded and Python only reads the first `dim` entries of each row.
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
round_flat = np.zeros((n, max_d), dtype=float)
self_l2 = []
self_maxabs = []

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
    round_flat[np.ix_(idx, range(d))] = B
    # geomstats' OWN round-trip error per row.
    for r in range(len(idx)):
        v = V[r]
        b = B[r]
        denom = np.sqrt(np.sum(v * v))
        l2 = np.sqrt(np.sum((b - v) ** 2)) / max(denom, 1e-300)
        self_l2.append(l2)
        self_maxabs.append(np.max(np.abs(b - v)))

emit("exp_flat", exp_flat.reshape(-1))
emit("round_flat", round_flat.reshape(-1))
emit("self_l2_max", [float(np.max(self_l2))])
emit("self_maxabs_max", [float(np.max(self_maxabs))])
"#,
    );

    let geo_exp_flat = py.vector("exp_flat");
    let geo_round_flat = py.vector("round_flat");
    let geo_self_l2_max = py.scalar("self_l2_max");
    let geo_self_maxabs_max = py.scalar("self_maxabs_max");

    assert_eq!(
        geo_exp_flat.len(),
        n * max_d,
        "geomstats exp length mismatch"
    );
    assert_eq!(
        geo_round_flat.len(),
        n * max_d,
        "geomstats round length mismatch"
    );

    // ---- cross-engine agreement on the exp map and the round-trip ---------
    let exp_maxabs = max_abs_diff(&gam_exp_flat, geo_exp_flat);
    let round_maxabs = max_abs_diff(&gam_round_flat, geo_round_flat);
    let exp_l2 = relative_l2(&gam_exp_flat, geo_exp_flat);

    eprintln!(
        "poincare exp/log roundtrip (c={CURVATURE}, n={n} vectors over d={dims:?}):\n  \
         gam   self-roundtrip: L2_rel_max={gam_roundtrip_l2_max:.3e} maxabs={gam_roundtrip_maxabs:.3e}\n  \
         geomstats self-roundtrip: L2_rel_max={geo_self_l2_max:.3e} maxabs={geo_self_maxabs_max:.3e}\n  \
         cross-engine: exp_maxabs={exp_maxabs:.3e} exp_rel_l2={exp_l2:.3e} round_maxabs={round_maxabs:.3e}"
    );

    // --- bound 1: each engine's intrinsic round-trip recovers v ---
    // log∘exp is the identity in exact arithmetic; away from the saturation
    // boundary (max norm 0.95 < 1) the only error is f64 rounding of
    // tanh/atanh and the radial scaling, which is a handful of ULPs.
    // 1e-13 max-abs / 1e-12 relative-L2 is the SPEC bound — generous against
    // ULP-level noise yet tight enough that a missing √k factor or a
    // tanh/atanh swap (errors of order 0.1–1) trips it instantly.
    assert!(
        gam_roundtrip_maxabs < 1.0e-13,
        "gam log∘exp must recover v: maxabs={gam_roundtrip_maxabs:.3e}"
    );
    assert!(
        gam_roundtrip_l2_max < 1.0e-12,
        "gam log∘exp relative-L2 too large: {gam_roundtrip_l2_max:.3e}"
    );
    assert!(
        geo_self_maxabs_max < 1.0e-13,
        "geomstats log∘exp must recover v: maxabs={geo_self_maxabs_max:.3e}"
    );
    assert!(
        geo_self_l2_max < 1.0e-12,
        "geomstats log∘exp relative-L2 too large: {geo_self_l2_max:.3e}"
    );

    // --- bound 2: gam and geomstats compute the SAME maps ---
    // Both evaluate the identical closed forms on identical inputs, so the
    // images must coincide to floating-point precision (transcendental-fn
    // rounding only). The same 1e-13 / 1e-12 SPEC bounds apply head-to-head.
    assert!(
        exp_maxabs < 1.0e-13,
        "gam vs geomstats exp_0(v) disagree: maxabs={exp_maxabs:.3e}"
    );
    assert!(
        exp_l2 < 1.0e-12,
        "gam vs geomstats exp_0(v) relative-L2 disagree: {exp_l2:.3e}"
    );
    assert!(
        round_maxabs < 1.0e-13,
        "gam vs geomstats log∘exp disagree: maxabs={round_maxabs:.3e}"
    );
}
