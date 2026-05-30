//! End-to-end quality: gam's closed-form Poincaré-ball geodesic distance must
//! agree to float-ulp precision with `geomstats`, the mature reference for
//! Riemannian-manifold primitives.
//!
//! Both `gam::geometry::poincare::poincare_distance` and
//! `geomstats.geometry.hyperbolic.PoincareBall(dim).metric.dist` evaluate the
//! *same* closed-form arccosh geodesic distance on the open Poincaré ball with
//! sectional curvature `c = -1`:
//!
//!     d(a, b) = arccosh( 1 + 2 |a-b|^2 / ((1 - |a|^2)(1 - |b|^2)) ).
//!
//! This is not an optimization/recovery comparison — there is no fitting and no
//! convergence variation. Two implementations of one transcendental closed form
//! must converge to the IEEE-754 floating-point limit; the only achievable
//! differences come from summation order and intermediate rounding in the
//! `log`/`acosh` evaluation. We therefore hold an extremely tight bound and a
//! divergence above it signals a real transcendental-function or
//! distance-formula bug, not statistical noise.
//!
//! Identical data is fed to both engines via the shared CSV harness: a
//! fixed-seed batch of 50 point pairs sampled strictly interior to
//! `[-0.99, 0.99]^d` for `d ∈ {2, 4, 8}`.

use gam::geometry::poincare::poincare_distance;
use gam::test_support::reference::{Column, max_abs_diff, relative_l2, run_python};
use ndarray::ArrayView1;

/// Deterministic SplitMix64 -> uniform f64 in `[lo, hi)`. Fixed seed makes the
/// point batch reproducible and lets gam and geomstats see byte-identical data
/// (the CSV is written from these same values).
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

#[test]
fn poincare_geodesic_matches_geomstats_to_float_ulp() {
    // Strictly interior box: |coord| <= 0.99 keeps every point well inside the
    // open unit ball so the (1 - |x|^2) denominators are bounded away from 0
    // and both engines stay in the smooth interior of the arccosh formula.
    const N_PAIRS: usize = 50;
    const INTERIOR: f64 = 0.99;
    const CURVATURE: f64 = -1.0;

    for &dim in &[2usize, 4, 8] {
        // ---- fixed-seed point pairs (identical bytes to both engines) ------
        let mut rng = 0x5DEE_CE66_2024_0529u64 ^ (dim as u64).wrapping_mul(0x100_0001);
        let mut a = vec![0.0f64; N_PAIRS * dim];
        let mut b = vec![0.0f64; N_PAIRS * dim];
        for v in a.iter_mut() {
            *v = splitmix_uniform(&mut rng, -INTERIOR, INTERIOR);
        }
        for v in b.iter_mut() {
            *v = splitmix_uniform(&mut rng, -INTERIOR, INTERIOR);
        }

        // ---- gam: closed-form Riemannian distance per pair -----------------
        let gam_dist: Vec<f64> = (0..N_PAIRS)
            .map(|p| {
                let av = ArrayView1::from(&a[p * dim..(p + 1) * dim]);
                let bv = ArrayView1::from(&b[p * dim..(p + 1) * dim]);
                poincare_distance(av, bv, CURVATURE).expect("gam poincare_distance")
            })
            .collect();

        // ---- columns: one per coordinate, identical values to Python -------
        // Flattened row-major (pair, coord); the Python side reshapes to (N, d).
        let mut columns: Vec<Column<'_>> = Vec::with_capacity(2 * dim);
        let mut a_cols: Vec<Vec<f64>> = vec![Vec::with_capacity(N_PAIRS); dim];
        let mut b_cols: Vec<Vec<f64>> = vec![Vec::with_capacity(N_PAIRS); dim];
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

        // ---- geomstats: the mature reference, same closed-form distance ----
        let body = format!(
            r#"
import numpy as np
from geomstats.geometry.hyperbolic import Hyperbolic
dim = {dim}
acols = [df["a%d" % j] for j in range(dim)]
bcols = [df["b%d" % j] for j in range(dim)]
A = np.column_stack([np.asarray(c, dtype=float) for c in acols])
B = np.column_stack([np.asarray(c, dtype=float) for c in bcols])
ball = Hyperbolic(dim=dim, default_coords_type="ball")
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

        // ---- compare -------------------------------------------------------
        let mad = max_abs_diff(&gam_dist, ref_dist);
        let rel = relative_l2(&gam_dist, ref_dist);
        eprintln!("poincare d={dim}: n={N_PAIRS} max_abs_diff={mad:.3e} relative_l2={rel:.3e}");

        // Two implementations of the identical closed-form arccosh distance can
        // only differ by float summation order / intermediate rounding on these
        // strictly-interior points. 5e-10 (abs) and 1e-9 (relative) are at the
        // ulp scale of the distances involved and would catch any genuine
        // log/acosh or formula bug while leaving headroom for benign rounding.
        assert!(
            mad < 5e-10,
            "d={dim}: gam vs geomstats geodesic distance diverged: max_abs_diff={mad:.3e}"
        );
        assert!(
            rel < 1e-9,
            "d={dim}: gam vs geomstats geodesic distance diverged: relative_l2={rel:.3e}"
        );
    }
}
