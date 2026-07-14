//! #1731 regression: `matern(x1, x2, k=K)` must produce a basis whose column
//! count grows monotonically with the requested center count `k` (up to the
//! genuine numerical-rank limit), instead of saturating at ~54 and then
//! DECREASING for large `k`.
//!
//! Root cause & mechanism. The Matérn forward design rank-reduces an
//! over-specified center set (`matern_rank_reduce_centers`) by probing the
//! realized kernel block's numerical rank *at the kernel's `length_scale`*. When
//! the length scale out-runs the inter-center spacing, densely-but-distinctly
//! placed centers go numerically collinear and RRQR truncates them — capping the
//! basis (~54 for n=500) and shrinking it as `k` grows. The reporter hit this
//! because their build's length scale was on the over-smoothed (coarse) side of
//! the center spacing.
//!
//! The auto length-scale seed used by `matern(...)` is `max_range / sqrt(n)`
//! (gam#1629's resolving seed). Because `k <= n`, that seed is always <= the
//! inter-center spacing `~max_range / sqrt(k)` in 2-D, so every requested center
//! is resolved and the basis stays full rank: the column count tracks `k`. This
//! test pins that invariant through the public build path so a regression to a
//! coarse (over-smoothed) construction seed — which would re-introduce the
//! gam#1731 cap — is caught.
//!
//! On the issue's n=500 `[0, 5]²` cloud the counts are 19, 49, 74, 99, 149
//! (= k − 1 from the sum-to-zero constraint), growing monotonically and far past
//! the old ~54 cap.

use gam::basis::{
    CenterStrategy, MaternBasisSpec, MaternIdentifiability, MaternNu, build_matern_basis,
};
use gam::smooth::auto_initial_length_scale;
use gam::smooth::input_standardization::estimate_isotropic_scale;
use ndarray::Array2;

/// Deterministic n×2 cloud on `[0, 5]²` (xorshift64*, no RNG crate dependency),
/// matching the gam#1731 reproducer's `uniform(0, 5, n)` shape.
fn make_issue_cloud(n: usize) -> Array2<f64> {
    let mut pts = Array2::<f64>::zeros((n, 2));
    let mut state: u64 = 0x1234_5678_9ABC_DEF0;
    let mut next = || {
        state ^= state >> 12;
        state ^= state << 25;
        state ^= state >> 27;
        let v = state.wrapping_mul(0x2545_F491_4F6C_DD1D);
        ((v >> 11) as f64) / ((1u64 << 53) as f64)
    };
    for r in 0..n {
        pts[[r, 0]] = 5.0 * next();
        pts[[r, 1]] = 5.0 * next();
    }
    pts
}

/// Basis column count for a default `matern(x1, x2, k=k)` term, reproducing the
/// `matern(...)` build dispatch exactly: an explicit `k` in a low-D smooth
/// selects FarthestPoint centers with the default ν = 5/2; each axis is
/// standardized to unit variance; the auto length scale `max_range / sqrt(n)` is
/// σ_geom-compensated for that standardization; the basis is built on the
/// standardized data.
fn matern_basis_cols(data: &Array2<f64>, k: usize) -> usize {
    let mut x = data.clone();
    let raw_length_scale = auto_initial_length_scale(x.view(), &[0, 1]);
    let input_scale =
        estimate_isotropic_scale(x.view()).expect("2-D cloud with n>=2 has a spatial scale");
    input_scale.standardize(&mut x);
    let length_scale = input_scale.to_standardized_units(raw_length_scale);

    let spec = MaternBasisSpec {
        center_strategy: CenterStrategy::FarthestPoint { num_centers: k },
        periodic: None,
        length_scale: gam::terms::basis::MaternLengthScale::fixed(length_scale),
        nu: MaternNu::FiveHalves,
        include_intercept: false,
        double_penalty: false,
        identifiability: MaternIdentifiability::CenterSumToZero,
        aniso_log_scales: None,
    };
    let basis = build_matern_basis(x.view(), &spec)
        .unwrap_or_else(|e| panic!("matern basis build failed for k={k}: {e:?}"));
    basis.design.ncols()
}

#[test]
fn matern_basis_size_grows_monotonically_with_k() {
    let n = 500usize;
    let data = make_issue_cloud(n);

    let ks = [20usize, 50, 75, 100, 150];
    let cols: Vec<usize> = ks.iter().map(|&k| matern_basis_cols(&data, k)).collect();

    for (k, c) in ks.iter().zip(&cols) {
        eprintln!("k={k:>4}  basis_cols={c:>4}");
    }

    // (a) Monotone non-decreasing in k: a richer request must never yield a LESS
    //     expressive basis (the pre-fix DECREASE at large k).
    for w in cols.windows(2) {
        assert!(
            w[1] >= w[0],
            "basis column count must be monotonically non-decreasing in k, but it \
             dropped: {cols:?} for k={ks:?}"
        );
    }

    // (b) The ~54 cap is gone: for k >= 75 the basis is far wider than the old
    //     saturation point. Sum-to-zero drops exactly one column, so a healthy
    //     k=75 basis is ~74 columns; require comfortably > 54.
    for (i, &k) in ks.iter().enumerate() {
        if k >= 75 {
            assert!(
                cols[i] > 54,
                "matern(x1,x2,k={k}) produced only {} basis columns (expected > 54); \
                 the rank cap that made large k a no-op is still present: {cols:?}",
                cols[i]
            );
        }
    }

    // (c) k must still actually buy columns: the k=100 basis must be strictly
    //     wider than the k=50 basis (otherwise k saturated in between).
    assert!(
        cols[3] > cols[1],
        "increasing k from 50 to 100 did not increase the basis width: {cols:?}"
    );
}
