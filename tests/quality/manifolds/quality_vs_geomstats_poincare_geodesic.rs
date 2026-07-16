//! End-to-end quality: gam's Poincaré-ball geodesic distance is asserted against
//! the EXACT closed-form geodesic distance and against the defining axioms of a
//! Riemannian metric — OBJECTIVE correctness, not "we match geomstats".
//!
//! The primary pass/fail criterion is gam vs MATHEMATICAL GROUND TRUTH:
//!
//!   1. CLOSED-FORM TRUTH. The Poincaré-ball geodesic distance has the analytic
//!      form
//!          d(a, b) = (1/√(-c)) · arccosh( 1 + 2·(-c)·|a-b|²
//!                                           / ((1 + c|a|²)(1 + c|b|²)) ).
//!      We evaluate this exact expression in-test (independent of gam's internal
//!      summation order) and require gam to reproduce it to f64-ulp scale. This
//!      is correctness against an analytic quantity, not agreement with a peer
//!      fit, so it is a legitimate accuracy claim.
//!
//!   2. METRIC AXIOMS. A geodesic distance on a Riemannian manifold MUST satisfy
//!      the metric-space axioms, regardless of any reference implementation:
//!        * non-negativity:           d(a, b) ≥ 0
//!        * identity of indiscernibles: d(a, a) = 0  (exactly, via acosh(1)=0)
//!        * symmetry:                 d(a, b) = d(b, a)
//!        * triangle inequality:      d(a, c) ≤ d(a, b) + d(b, c)
//!      These are structural correctness properties of gam's own output; they
//!      hold or fail without ever consulting geomstats.
//!
//! geomstats remains only as a BASELINE CROSS-CHECK: it computes the same exact
//! arccosh distance, so it is a second independent witness of ground truth. We
//! print gam-vs-geomstats divergence for context and assert gam's error against
//! the analytic truth is no worse than geomstats' error against that same truth
//! (match-or-beat on accuracy). The claim "gam is correct" rests on items 1–2,
//! not on reproducing geomstats.
//!
//! Identical data is fed to gam, the in-test closed form, and geomstats via the
//! shared CSV harness: a fixed-seed batch of point pairs for `d ∈ {2, 4, 8}`,
//! each point sampled in `[-0.99, 0.99]^d` then rescaled to norm ≤ 0.9 so it
//! lies strictly inside the open unit ball where the arccosh formula is smooth.

use gam::geometry::poincare::poincare_distance;
use gam::test_support::reference::{
    Column, QualityPair, max_abs_diff, relative_l2, rmse, run_python,
};
use ndarray::ArrayView1;

/// Deterministic SplitMix64 -> uniform f64 in `[lo, hi)`. Fixed seed makes the
/// point batch reproducible and lets gam, the in-test closed form, and geomstats
/// see byte-identical data (the CSV is written from these same values).
fn splitmix_uniform(state: &mut u64, lo: f64, hi: f64) -> f64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^= z >> 31;
    // 53-bit mantissa fraction in [0, 1).
    let unit = (z >> 11) as f64 / (1u64 << 53) as f64;
    lo + (hi - lo) * unit
}

/// Exact analytic Poincaré-ball geodesic distance, evaluated independently of
/// gam's implementation. This is the mathematical ground truth gam must match.
/// `curvature` is `c < 0`; with `c = -1` this is the canonical unit ball.
fn closed_form_distance(a: &[f64], b: &[f64], curvature: f64) -> f64 {
    let mut diff_sq = 0.0;
    let mut a_sq = 0.0;
    let mut b_sq = 0.0;
    for i in 0..a.len() {
        let d = a[i] - b[i];
        diff_sq += d * d;
        a_sq += a[i] * a[i];
        b_sq += b[i] * b[i];
    }
    let denom_a = 1.0 + curvature * a_sq;
    let denom_b = 1.0 + curvature * b_sq;
    let arg = 1.0 + 2.0 * (-curvature) * diff_sq / (denom_a * denom_b);
    // arccosh is defined for arg ≥ 1; strictly-interior points keep arg ≥ 1, and
    // acosh(1) = 0 gives exact-zero distance for identical points.
    arg.max(1.0).acosh() / (-curvature).sqrt()
}

#[test]
fn poincare_geodesic_is_a_correct_metric() {
    // The Poincare ball is the OPEN unit ball: every point must satisfy |x| < 1,
    // otherwise (1 + c|x|²) flips sign and the arccosh distance is undefined. A
    // per-coordinate box like [-0.99, 0.99]^d does NOT enforce this (|x|² up to
    // 1.96 at d=2, up to 7.84 at d=8). We sample a raw direction in the box then
    // rescale each point to a fixed radius MAX_RADIUS < 1, keeping every engine
    // strictly inside the smooth interior of the formula with (1 + c|x|²) bounded
    // away from 0. The rescaled vectors are exactly what gam evaluates, what the
    // in-test closed form evaluates, and what is written to the CSV for geomstats.
    const N_PAIRS: usize = 50;
    const BOX: f64 = 0.99;
    const MAX_RADIUS: f64 = 0.9;
    const CURVATURE: f64 = -1.0;

    // Rescale each consecutive `dim`-vector to have norm <= MAX_RADIUS.
    fn clamp_into_ball(flat: &mut [f64], dim: usize) {
        for chunk in flat.chunks_mut(dim) {
            let norm = chunk.iter().map(|x| x * x).sum::<f64>().sqrt();
            if norm > MAX_RADIUS {
                let scale = MAX_RADIUS / norm;
                for x in chunk.iter_mut() {
                    *x *= scale;
                }
            }
        }
    }

    fn point(flat: &[f64], dim: usize, idx: usize) -> &[f64] {
        &flat[idx * dim..(idx + 1) * dim]
    }
    fn gam_dist(flat_a: &[f64], flat_b: &[f64], dim: usize, i: usize, j: usize) -> f64 {
        let av = ArrayView1::from(point(flat_a, dim, i));
        let bv = ArrayView1::from(point(flat_b, dim, j));
        poincare_distance(av, bv, CURVATURE).expect("gam poincare_distance")
    }

    for &dim in &[2usize, 4, 8] {
        // ---- fixed-seed point pairs (identical bytes to every engine) ------
        let mut rng = 0x5DEE_CE66_2024_0529u64 ^ (dim as u64).wrapping_mul(0x100_0001);
        let mut a = vec![0.0f64; N_PAIRS * dim];
        let mut b = vec![0.0f64; N_PAIRS * dim];
        for v in a.iter_mut() {
            *v = splitmix_uniform(&mut rng, -BOX, BOX);
        }
        for v in b.iter_mut() {
            *v = splitmix_uniform(&mut rng, -BOX, BOX);
        }
        clamp_into_ball(&mut a, dim);
        clamp_into_ball(&mut b, dim);

        // ---- gam vs EXACT CLOSED-FORM TRUTH (primary accuracy claim) -------
        let gam_d: Vec<f64> = (0..N_PAIRS).map(|p| gam_dist(&a, &b, dim, p, p)).collect();
        let truth_d: Vec<f64> = (0..N_PAIRS)
            .map(|p| closed_form_distance(point(&a, dim, p), point(&b, dim, p), CURVATURE))
            .collect();
        let gam_truth_mad = max_abs_diff(&gam_d, &truth_d);
        let gam_truth_rmse = rmse(&gam_d, &truth_d);
        eprintln!(
            "poincare d={dim}: n={N_PAIRS} gam-vs-closed-form max_abs={gam_truth_mad:.3e} rmse={gam_truth_rmse:.3e}"
        );
        // gam and the analytic formula differ only by f64 summation order /
        // intermediate rounding on these strictly-interior points; 5e-10 is at
        // the ulp scale of the distances and would catch any genuine
        // formula/acosh bug while leaving headroom for benign rounding.
        assert!(
            gam_truth_mad < 5e-10,
            "d={dim}: gam diverged from the exact closed-form geodesic distance: max_abs_diff={gam_truth_mad:.3e}"
        );

        // ---- METRIC AXIOMS on gam's own distances (structural truth) -------
        // (1) identity of indiscernibles: d(x, x) = 0 exactly.
        for p in 0..N_PAIRS {
            let daa = gam_dist(&a, &a, dim, p, p);
            assert_eq!(
                daa, 0.0,
                "d={dim}: d(a_{p}, a_{p}) = {daa:.3e}, expected exactly 0 (acosh(1)=0)"
            );
        }
        // (2) non-negativity and (3) symmetry across all sampled cross pairs.
        let mut max_asymmetry = 0.0f64;
        for i in 0..N_PAIRS {
            for j in 0..N_PAIRS {
                let dab = gam_dist(&a, &b, dim, i, j);
                assert!(
                    dab >= 0.0 && dab.is_finite(),
                    "d={dim}: d(a_{i}, b_{j}) = {dab:.3e} violates non-negativity/finiteness"
                );
                let dba = gam_dist(&b, &a, dim, j, i);
                max_asymmetry = max_asymmetry.max((dab - dba).abs());
            }
        }
        eprintln!("poincare d={dim}: max symmetry violation |d(a,b)-d(b,a)| = {max_asymmetry:.3e}");
        assert!(
            max_asymmetry < 1e-12,
            "d={dim}: distance is not symmetric: max |d(a,b)-d(b,a)| = {max_asymmetry:.3e}"
        );
        // (4) triangle inequality: d(a_i, b_j) <= d(a_i, a_k) + d(a_k, b_j) for
        // a deterministic sweep of intermediate points a_k. A small tolerance
        // absorbs rounding in the sum of two acosh evaluations.
        let mut worst_triangle = 0.0f64;
        for i in 0..N_PAIRS {
            for j in 0..N_PAIRS {
                let direct = gam_dist(&a, &b, dim, i, j);
                for k in 0..N_PAIRS {
                    let via = gam_dist(&a, &a, dim, i, k) + gam_dist(&a, &b, dim, k, j);
                    // positive => violation magnitude
                    worst_triangle = worst_triangle.max(direct - via);
                }
            }
        }
        eprintln!("poincare d={dim}: worst triangle-inequality excess = {worst_triangle:.3e}");
        assert!(
            worst_triangle < 1e-9,
            "d={dim}: triangle inequality violated by {worst_triangle:.3e}"
        );

        // ---- geomstats: BASELINE CROSS-CHECK (a second ground-truth witness)
        // Same exact arccosh distance; used only to confirm gam's error against
        // the analytic truth is no worse than geomstats' error against it.
        let mut columns: Vec<Column<'_>> = Vec::with_capacity(2 * dim);
        let mut a_cols: Vec<Vec<f64>> = (0..dim).map(|_| Vec::with_capacity(N_PAIRS)).collect();
        let mut b_cols: Vec<Vec<f64>> = (0..dim).map(|_| Vec::with_capacity(N_PAIRS)).collect();
        for p in 0..N_PAIRS {
            for j in 0..dim {
                a_cols[j].push(a[p * dim + j]);
                b_cols[j].push(b[p * dim + j]);
            }
        }
        let a_names: Vec<String> = (0..dim).map(|j| format!("a{j}")).collect();
        let b_names: Vec<String> = (0..dim).map(|j| format!("b{j}")).collect();
        for j in 0..dim {
            columns.push(Column::new(&a_names[j], &a_cols[j]));
            columns.push(Column::new(&b_names[j], &b_cols[j]));
        }

        let body = format!(
            r#"
import numpy as np
from geomstats.geometry.poincare_ball import PoincareBall
dim = {dim}
acols = [df["a%d" % j] for j in range(dim)]
bcols = [df["b%d" % j] for j in range(dim)]
A = np.column_stack([np.asarray(c, dtype=float) for c in acols])
B = np.column_stack([np.asarray(c, dtype=float) for c in bcols])
# Curvature -1 Poincare ball of dimension `dim`; .metric.dist is the closed-form
# arccosh geodesic distance d(a,b)=arccosh(1 + 2|a-b|^2/((1-|a|^2)(1-|b|^2))).
ball = PoincareBall(dim)
d = np.array([float(ball.metric.dist(A[i], B[i])) for i in range(A.shape[0])])
emit("dist", d)
"#
        );
        let r = run_python(&columns, &body);
        let ref_dist = r.vector("dist");
        assert_eq!(
            ref_dist.len(),
            N_PAIRS,
            "geomstats returned {} distances, expected {N_PAIRS}",
            ref_dist.len()
        );

        let gam_vs_ref = relative_l2(&gam_d, ref_dist);
        let ref_truth_rmse = rmse(ref_dist, &truth_d);
        eprintln!(
            "poincare d={dim}: gam-vs-geomstats relative_l2={gam_vs_ref:.3e} | geomstats-vs-closed-form rmse={ref_truth_rmse:.3e}"
        );
        eprintln!(
            "{}",
            QualityPair::error(
                "manifolds",
                &format!("quality_vs_geomstats_poincare_geodesic::d{dim}"),
                "rmse_to_truth",
                gam_truth_rmse,
                "geomstats",
                ref_truth_rmse,
            )
            .line()
        );
        // Match-or-beat on ACCURACY against the shared analytic truth: gam's
        // RMSE to ground truth must not exceed geomstats' RMSE by more than a
        // small absolute slack (both are at ulp scale, so a 1e-12 floor avoids
        // flagging which engine happens to round last).
        assert!(
            gam_truth_rmse <= ref_truth_rmse + 1e-12,
            "d={dim}: gam is less accurate against the closed-form truth than geomstats: \
             gam_rmse={gam_truth_rmse:.3e} > geomstats_rmse={ref_truth_rmse:.3e}"
        );
    }
}
