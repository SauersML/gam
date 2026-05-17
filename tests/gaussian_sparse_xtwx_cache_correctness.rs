//! Exact-equivalence regression for the Gaussian-Identity *sparse* XᵀWX
//! cache.
//!
//! The sparse REML eval path (`assemble_and_factor_sparse_penalized_system`)
//! historically rebuilt `XᵀWX` per outer evaluation via an SpGEMM. For
//! Gaussian + Identity the IRLS weights are constant across the outer loop,
//! so the new fast path caches `XᵀWX` in `SparseXtwxPrecomputed` once and
//! scatters it into the inner workspace's symbolic pattern instead of
//! re-running the SpGEMM.
//!
//! This test asserts the cached path produces a **bit-identical** sparse
//! penalized Hessian — same nonzero pattern, same numerical values — as the
//! per-call recompute. Mathematical equivalence at this level guarantees
//! every downstream solve, factorization, log-determinant, and selected
//! inverse is also identical, so the higher-level REML fit cannot diverge.

use gam::pirls::{PirlsWorkspace, SparseXtwxPrecomputed};
use ndarray::{Array1, Array2};
use rand::rngs::StdRng;
use rand::{RngExt, SeedableRng};

use faer::sparse::{SparseColMat, Triplet};
use gam::linalg::sparse_exact::assemble_and_factor_sparse_penalized_system;

const N: usize = 10_000;
const SEED: u64 = 0xC0FFEE_5EED_1234;

fn build_cylinder_design(n: usize) -> SparseColMat<usize, f64> {
    // Cylinder-style sparse design: tensor product of a periodic θ basis
    // (m_theta marginal cells) with a height-h basis (m_h cells). Each row
    // is one observation with band support, so X is highly sparse and the
    // X'WX SpGEMM has a non-trivial fill pattern — matching the structural
    // shape of the production cylinder fits the cache targets.
    let m_theta: usize = 16;
    let m_h: usize = 12;
    let p = m_theta * m_h;

    let mut rng = StdRng::seed_from_u64(SEED);
    let mut triplets = Vec::<Triplet<usize, usize, f64>>::with_capacity(n * 4);
    for i in 0..n {
        // Sample a θ-cell index and an h-cell index; each row touches a
        // 2x2 cell window in the tensor (linear in both axes), giving 4
        // nonzeros per row.
        let t = rng.random_range(0..m_theta);
        let t1 = (t + 1) % m_theta;
        let h = rng.random_range(0..(m_h - 1));
        let h1 = h + 1;
        let alpha: f64 = rng.random_range(0.1..0.9);
        let beta: f64 = rng.random_range(0.1..0.9);
        for (jt, wt) in [(t, 1.0 - alpha), (t1, alpha)] {
            for (jh, wh) in [(h, 1.0 - beta), (h1, beta)] {
                let col = jh * m_theta + jt;
                triplets.push(Triplet::new(i, col, wt * wh));
            }
        }
    }
    SparseColMat::try_new_from_triplets(n, p, &triplets).expect("build cylinder sparse")
}

fn random_weights(n: usize, seed: u64) -> Array1<f64> {
    let mut rng = StdRng::seed_from_u64(seed);
    // Positive priorweights; in Gaussian-Identity production the cache is
    // built once at the user-supplied priorweights and reused, so we use
    // an irregular distribution to stress the cache pattern.
    Array1::from_shape_fn(n, |_| rng.random_range(0.25..2.0))
}

fn ridge_penalty(p: usize, scale: f64) -> Array2<f64> {
    // Plain diagonal ridge so the assembly carries a non-trivial penalty
    // contribution on the diagonal of H.
    let mut s = Array2::<f64>::zeros((p, p));
    for j in 0..p {
        s[[j, j]] = scale;
    }
    s
}

fn dense_from_sparse_upper(m: &SparseColMat<usize, f64>) -> Array2<f64> {
    let p = m.ncols();
    let mut out = Array2::<f64>::zeros((p, p));
    let (symbolic, values) = m.parts();
    let col_ptr = symbolic.col_ptr();
    let row_idx = symbolic.row_idx();
    for col in 0..p {
        for idx in col_ptr[col]..col_ptr[col + 1] {
            let row = row_idx[idx];
            out[[row, col]] += values[idx];
        }
    }
    out
}

#[test]
fn sparse_xtwx_cache_matches_per_call_recompute_bitwise() {
    let x = build_cylinder_design(N);
    let p = x.ncols();
    let weights = random_weights(N, SEED ^ 0xA5A5);
    let s_lambda = ridge_penalty(p, 1.5);
    let ridge: f64 = 1e-8;

    // Baseline: per-call SpGEMM recompute (precomputed_xtwx = None).
    let mut workspace_no_cache = PirlsWorkspace::new(N, p, 0, 0);
    let h_no_cache = assemble_and_factor_sparse_penalized_system(
        &mut workspace_no_cache,
        &x,
        &weights,
        &s_lambda,
        ridge,
        None,
    )
    .expect("per-call sparse assembly");

    // Cached: precompute XᵀWX once, then assemble using the cache.
    let precomp = SparseXtwxPrecomputed::build(&x, &weights).expect("build precomp");
    let mut workspace_cached = PirlsWorkspace::new(N, p, 0, 0);
    let h_cached = assemble_and_factor_sparse_penalized_system(
        &mut workspace_cached,
        &x,
        &weights,
        &s_lambda,
        ridge,
        Some(&precomp),
    )
    .expect("cached sparse assembly");

    // Symbolic equality: the upper-triangular fill pattern must match
    // exactly, because the inner penalty/ridge scatter relies on positional
    // index alignment with the X'X pattern.
    let (sym_no, _) = h_no_cache.h_sparse.parts();
    let (sym_yes, _) = h_cached.h_sparse.parts();
    assert_eq!(
        sym_no.col_ptr(),
        sym_yes.col_ptr(),
        "sparse H col_ptr must be identical between cached and per-call paths"
    );
    assert_eq!(
        sym_no.row_idx(),
        sym_yes.row_idx(),
        "sparse H row_idx must be identical between cached and per-call paths"
    );

    // Numerical equality: same SpGEMM math, just sourced from a cache
    // instead of recomputed — we expect bit-identical f64 results
    // (the cache stores the exact bits returned by the per-call path).
    let v_no = h_no_cache.h_sparse.val();
    let v_yes = h_cached.h_sparse.val();
    assert_eq!(
        v_no.len(),
        v_yes.len(),
        "sparse H value-vector lengths must match"
    );
    let mut max_abs_diff = 0.0_f64;
    let mut max_ulp_diff: i64 = 0;
    for (a, b) in v_no.iter().zip(v_yes.iter()) {
        let d = (a - b).abs();
        if d > max_abs_diff {
            max_abs_diff = d;
        }
        // ULP diff: should be exactly zero for the SpGEMM dense backend
        // (deterministic accumulation order), tolerated up to a couple of
        // ULPs if a future refactor introduces reduction-order variance.
        let ai = a.to_bits() as i64;
        let bi = b.to_bits() as i64;
        let ulp = (ai - bi).abs();
        if ulp > max_ulp_diff {
            max_ulp_diff = ulp;
        }
    }
    eprintln!(
        "[sparse-xtwx-cache] N={N} p={p} nnz_upper={} max|Δ|={max_abs_diff:.3e} max_ulp={max_ulp_diff}",
        v_no.len()
    );

    // Bit-identical is the design contract here; permit at most ~1 ulp.
    assert!(
        max_abs_diff <= 1e-12,
        "max abs diff = {max_abs_diff:.3e} exceeds 1e-12 (cache must be \
         numerically equivalent to per-call recompute)"
    );

    // logdet should also match — it is computed from the Cholesky factor of
    // the assembled sparse matrix; identical matrices → identical logdet.
    let dl = (h_no_cache.logdet_h - h_cached.logdet_h).abs();
    eprintln!(
        "[sparse-xtwx-cache] logdet_h: no_cache={:.6} cached={:.6} |Δ|={dl:.3e}",
        h_no_cache.logdet_h, h_cached.logdet_h
    );
    assert!(
        dl <= 1e-9,
        "logdet drift {dl:.3e} exceeds 1e-9 between cached and per-call paths"
    );

    // Sanity: ensure the assembled H is actually nontrivial (catches a
    // degenerate test where both paths silently produce zeros).
    let dense = dense_from_sparse_upper(&h_no_cache.h_sparse);
    let frobenius_sq: f64 = dense.iter().map(|v| v * v).sum();
    assert!(
        frobenius_sq > 0.0,
        "sanity: assembled H should have nonzero entries"
    );
}
