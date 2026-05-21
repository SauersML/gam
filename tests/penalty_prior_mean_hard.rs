//! Hard-edge tests for `(β - μ_p)' S_p (β - μ_p)` centering on a non-zero prior mean.
//!
//! Centering is implemented in two related surfaces:
//!
//!   * `gam::solver::estimate::reml::unified::PenaltyCoordinate::{DenseRootCentered,
//!     BlockRootCentered}` with constructors `from_dense_root_with_mean` /
//!     `from_block_root_with_mean` and quadratics `shifted_quadratic` /
//!     `apply_shifted_penalty` (src/solver/reml/unified.rs lines 4297-4378, 4606-4656,
//!     5485, 5489).
//!   * `gam::terms::construction::CanonicalPenalty` carries the same `prior_mean`
//!     and routes through `to_penalty_coordinate` (src/terms/construction.rs:837,
//!     963-979). The shifted quadratic equals
//!         β'Sβ − 2 β' (S μ) + μ' S μ,
//!     and `CanonicalPenalty` exposes `quadratic` (β'Sβ),
//!     `prior_linear_shift(1.0)` (= S·μ embedded into the global basis), and
//!     `prior_constant_shift(1.0)` (= μ' S μ).
//!
//! `PenaltyCoordinate` lives in `pub(crate) mod unified` (src/solver/reml/mod.rs:23)
//! and is **not** reachable from integration tests. We therefore drive the
//! observable math via `CanonicalPenalty`, which is what feeds the runtime's
//! shifted-quadratic call sites (src/solver/reml/runtime.rs:1252, 3048, 5049,
//! 8144, 8553).

use gam::construction::CanonicalPenalty;
use ndarray::{Array1, Array2};

// ---------------------------------------------------------------------------
// Tiny deterministic PRNG (xorshift64*) so tests are reproducible without deps.
// ---------------------------------------------------------------------------

struct Rng(u64);

impl Rng {
    fn new(seed: u64) -> Self {
        // Avoid 0 state.
        Self(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ 0xDEAD_BEEF_CAFE_BABE)
    }
    fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }
    fn next_f64(&mut self) -> f64 {
        // Uniform in [-1, 1).
        let u = (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64;
        2.0 * u - 1.0
    }
}

fn random_vector(rng: &mut Rng, n: usize, scale: f64) -> Array1<f64> {
    Array1::from_iter((0..n).map(|_| scale * rng.next_f64()))
}

fn random_root(rng: &mut Rng, rank: usize, dim: usize) -> Array2<f64> {
    let mut m = Array2::<f64>::zeros((rank, dim));
    for r in 0..rank {
        for c in 0..dim {
            m[[r, c]] = rng.next_f64();
        }
    }
    m
}

/// Reference shifted quadratic: `scale * (β − μ)' (Rᵀ R) (β − μ)`,
/// computed directly from primitives (no centering machinery).
fn reference_shifted_quadratic(
    root: &Array2<f64>,
    beta: &Array1<f64>,
    mean: &Array1<f64>,
    scale: f64,
) -> f64 {
    let centered = beta - mean;
    let r_c = root.dot(&centered);
    scale * r_c.dot(&r_c)
}

/// Centered quadratic via `CanonicalPenalty` accessors (full-width β):
///     β'Sβ − 2 β'(Sμ) + μ'Sμ, all at `scale = 1`.
fn canonical_shifted_quadratic(cp: &CanonicalPenalty, beta: &Array1<f64>) -> f64 {
    let q_beta = cp.quadratic(beta, 1.0);
    let s_mu = cp.prior_linear_shift(1.0); // length = total_dim
    let cross = beta.dot(&s_mu);
    let const_term = cp.prior_constant_shift(1.0);
    q_beta - 2.0 * cross + const_term
}

// ---------------------------------------------------------------------------
// 1. Zero-mean reduction: bit-identical penalty quadratic.
// ---------------------------------------------------------------------------

#[test]
fn zero_prior_mean_matches_uncentered_bitwise() {
    let mut rng = Rng::new(0xA1);
    let mut mismatches = 0usize;
    for trial in 0..20 {
        let dim = 4 + (trial % 5);
        let rank = (dim - 1).max(1);
        let root = random_root(&mut rng, rank, dim);

        // Centered (zero mean) vs uncentered.
        let cp_centered =
            CanonicalPenalty::from_dense_root_with_mean(root.clone(), dim, Array1::zeros(dim));
        let cp_uncentered = CanonicalPenalty::from_dense_root(root.clone(), dim);

        for _ in 0..50 {
            let beta = random_vector(&mut rng, dim, 3.0);
            // Direct β'Sβ.
            let q_centered = cp_centered.quadratic(&beta, 1.0);
            let q_uncentered = cp_uncentered.quadratic(&beta, 1.0);
            if q_centered.to_bits() != q_uncentered.to_bits() {
                mismatches += 1;
            }
            // And the composed shifted quadratic at μ=0 must also bit-match β'Sβ
            // because the cross and constant terms are exactly 0.0.
            let shifted = canonical_shifted_quadratic(&cp_centered, &beta);
            if shifted.to_bits() != q_uncentered.to_bits() {
                mismatches += 1;
            }
        }
    }
    assert_eq!(mismatches, 0, "zero-mean centering must reduce bitwise");
}

// ---------------------------------------------------------------------------
// 2. Centering at β = μ: quadratic is exactly 0.0 bits.
// ---------------------------------------------------------------------------

#[test]
fn shifted_quadratic_at_beta_equals_mean_is_exact_zero() {
    let mut rng = Rng::new(0xB2);
    for trial in 0..30 {
        let dim = 3 + (trial % 6);
        let rank = (dim - 1).max(1);
        let root = random_root(&mut rng, rank, dim);
        let mean = random_vector(&mut rng, dim, 2.5);
        let cp = CanonicalPenalty::from_dense_root_with_mean(root, dim, mean.clone());

        // Reference path: pure root-of-zero computation.
        let q_ref = reference_shifted_quadratic(&cp.root, &mean, &mean, 1.0);
        assert_eq!(
            q_ref.to_bits(),
            0.0_f64.to_bits(),
            "reference quadratic at β=μ must be bit-zero (trial {trial})"
        );

        // CanonicalPenalty-composed path. Algebraically equal to zero, but
        // catastrophic cancellation can flip a few ULPs — this is exactly the
        // shape of bug we want to surface, so require true bitwise zero only
        // for the reference and a tight tolerance for the composed form.
        let q_comp = canonical_shifted_quadratic(&cp, &mean);
        assert!(
            q_comp.abs() <= 1e-10 * (1.0 + mean.dot(&mean).abs()),
            "composed shifted quadratic at β=μ should be ~0, got {q_comp} (trial {trial})"
        );
    }
}

// ---------------------------------------------------------------------------
// 3. Shift identity: penalty(β; μ) == penalty(β − μ; 0).
// ---------------------------------------------------------------------------

#[test]
fn shifted_quadratic_identity_under_translation() {
    let mut rng = Rng::new(0xC3);
    for seed in 0..200u64 {
        let mut local = Rng::new(0xC3_0000 ^ seed);
        let dim = 2 + ((seed as usize) % 7);
        let rank = (dim).max(1);
        let root = random_root(&mut local, rank, dim);
        let beta = random_vector(&mut local, dim, 4.0);
        let mean = random_vector(&mut local, dim, 4.0);

        let cp_mu = CanonicalPenalty::from_dense_root_with_mean(root.clone(), dim, mean.clone());
        let cp_zero =
            CanonicalPenalty::from_dense_root_with_mean(root.clone(), dim, Array1::zeros(dim));

        let lhs = canonical_shifted_quadratic(&cp_mu, &beta);
        let rhs = cp_zero.quadratic(&(beta.clone() - &mean), 1.0);
        let scale = 1.0 + lhs.abs() + rhs.abs();
        assert!(
            (lhs - rhs).abs() <= 1e-13 * scale,
            "shift identity broken at seed={seed}: lhs={lhs}, rhs={rhs}, scale={scale}"
        );
    }
}

// ---------------------------------------------------------------------------
// 4. GammaPrecision MAP closed-form: λ* = (a − 1 + ν_p/2) / (b + q/2),
//    with q = (β − μ)' S (β − μ).
//
// This is a pure-formula test: we verify the formula evaluates as the docstring
// at src/types.rs:209 promises and degenerates to (a, b) = (1, 0)
// MacKay/Tipping. A full REML-driver check would require fit_gam_*; if the lib
// is build-red, that driver test goes in the ignored section below.
// ---------------------------------------------------------------------------

#[test]
fn gamma_precision_map_closed_form_matches_formula() {
    let mut rng = Rng::new(0xD4);
    for trial in 0..30 {
        let dim = 3 + (trial % 5);
        let rank = (dim - 1).max(1);
        let root = random_root(&mut rng, rank, dim);
        let mean = random_vector(&mut rng, dim, 1.5);
        let beta = random_vector(&mut rng, dim, 2.0);
        let cp = CanonicalPenalty::from_dense_root_with_mean(root, dim, mean.clone());

        // Effective dimension ν_p = rank of S_p (≥ 1).
        let nu_p = cp.rank() as f64;
        // Centered quadratic.
        let q = canonical_shifted_quadratic(&cp, &beta).max(0.0);

        // MacKay/Tipping fixed point: (a,b)=(1,0) ⇒ λ* = (ν_p/2)/(q/2) = ν_p/q.
        let lam_mackay = nu_p / q;
        let lam_formula = ((1.0_f64) - 1.0 + 0.5 * nu_p) / (0.0_f64 + 0.5 * q);
        assert!(
            (lam_mackay - lam_formula).abs() <= 1e-12 * lam_mackay.abs(),
            "MacKay limit broken: {lam_mackay} vs {lam_formula}"
        );

        // Generic (a,b). Ensure the formula is finite, positive, and
        // monotone-decreasing in `b`.
        let (a, b1, b2) = (2.5_f64, 0.25_f64, 0.75_f64);
        let lam_b1 = (a - 1.0 + 0.5 * nu_p) / (b1 + 0.5 * q);
        let lam_b2 = (a - 1.0 + 0.5 * nu_p) / (b2 + 0.5 * q);
        assert!(lam_b1.is_finite() && lam_b1 > 0.0, "lam_b1 = {lam_b1}");
        assert!(lam_b2.is_finite() && lam_b2 > 0.0, "lam_b2 = {lam_b2}");
        assert!(
            lam_b2 < lam_b1,
            "MAP λ* must decrease with rate b: b1={b1} → {lam_b1}, b2={b2} → {lam_b2}"
        );
    }
}

// ---------------------------------------------------------------------------
// 5. Gradient w.r.t. β: ∂[(β−μ)' S (β−μ) / 2] / ∂β = S (β − μ).
// ---------------------------------------------------------------------------

#[test]
fn shifted_quadratic_gradient_finite_difference() {
    let dim: usize = 8;
    let rank: usize = 8;
    let h = 1e-6;
    let tol = 1e-7;

    for seed in 0..20u64 {
        let mut rng = Rng::new(0xE5_0000 ^ seed);
        let root = random_root(&mut rng, rank, dim);
        let mean = random_vector(&mut rng, dim, 1.0);
        let beta = random_vector(&mut rng, dim, 1.0);
        let cp = CanonicalPenalty::from_dense_root_with_mean(root, dim, mean.clone());

        // Analytical: S (β − μ).
        let s_beta = {
            // CanonicalPenalty has no public matvec; recompute as root.t() · (root · (β − μ)).
            let centered = &beta - &mean;
            let rb = cp.root.dot(&centered);
            cp.root.t().dot(&rb)
        };

        // Finite difference of 0.5 * (β−μ)' S (β−μ) via canonical_shifted_quadratic.
        let f = |b: &Array1<f64>| 0.5 * canonical_shifted_quadratic(&cp, b);
        let mut grad_fd = Array1::<f64>::zeros(dim);
        for i in 0..dim {
            let mut bp = beta.clone();
            let mut bm = beta.clone();
            bp[i] += h;
            bm[i] -= h;
            grad_fd[i] = (f(&bp) - f(&bm)) / (2.0 * h);
        }

        for i in 0..dim {
            let err = (grad_fd[i] - s_beta[i]).abs();
            let scale = 1.0 + s_beta[i].abs();
            assert!(
                err <= tol * scale,
                "grad mismatch seed={seed} i={i}: fd={} analytic={} err={err}",
                grad_fd[i],
                s_beta[i]
            );
        }
    }
}

// ---------------------------------------------------------------------------
// 6. Hessian = S, independent of μ.
// ---------------------------------------------------------------------------

#[test]
fn shifted_quadratic_hessian_equals_s_independent_of_mean() {
    let dim: usize = 6;
    let rank: usize = 6;
    let h = 1e-4;
    let tol = 1e-5;

    let mut rng = Rng::new(0xF6);
    let root = random_root(&mut rng, rank, dim);
    let beta = random_vector(&mut rng, dim, 1.0);

    // Build two penalties differing only in μ.
    let mu_a = random_vector(&mut rng, dim, 1.0);
    let mu_b = random_vector(&mut rng, dim, 10.0);
    let cp_a = CanonicalPenalty::from_dense_root_with_mean(root.clone(), dim, mu_a);
    let cp_b = CanonicalPenalty::from_dense_root_with_mean(root.clone(), dim, mu_b);

    let s_dense = root.t().dot(&root);

    // FD Hessian of 0.5 * (β−μ)' S (β−μ).
    let fd_hessian = |cp: &CanonicalPenalty| -> Array2<f64> {
        let mut hess = Array2::<f64>::zeros((dim, dim));
        for i in 0..dim {
            for j in 0..dim {
                let mut bpp = beta.clone();
                let mut bpm = beta.clone();
                let mut bmp = beta.clone();
                let mut bmm = beta.clone();
                bpp[i] += h;
                bpp[j] += h;
                bpm[i] += h;
                bpm[j] -= h;
                bmp[i] -= h;
                bmp[j] += h;
                bmm[i] -= h;
                bmm[j] -= h;
                let fpp = 0.5 * canonical_shifted_quadratic(cp, &bpp);
                let fpm = 0.5 * canonical_shifted_quadratic(cp, &bpm);
                let fmp = 0.5 * canonical_shifted_quadratic(cp, &bmp);
                let fmm = 0.5 * canonical_shifted_quadratic(cp, &bmm);
                hess[[i, j]] = (fpp - fpm - fmp + fmm) / (4.0 * h * h);
            }
        }
        hess
    };

    for (label, cp) in [("μ_a", &cp_a), ("μ_b", &cp_b)] {
        let h_fd = fd_hessian(cp);
        for i in 0..dim {
            for j in 0..dim {
                let err = (h_fd[[i, j]] - s_dense[[i, j]]).abs();
                let scale = 1.0 + s_dense[[i, j]].abs();
                assert!(
                    err <= tol * scale,
                    "Hessian mismatch ({label}) at ({i},{j}): fd={} S={} err={err}",
                    h_fd[[i, j]],
                    s_dense[[i, j]]
                );
            }
        }
    }
}

// ---------------------------------------------------------------------------
// 7. Block composition: two block penalties with non-zero μ sum to the joint.
// ---------------------------------------------------------------------------

#[test]
fn block_centered_penalties_sum_to_joint_quadratic() {
    let mut rng = Rng::new(0x17);
    let total = 10usize;
    for seed in 0..20u64 {
        let mut local = Rng::new(0x17_0000 ^ seed);
        // Two disjoint blocks.
        let cut = 3 + ((seed as usize) % 5); // [0, cut) and [cut, total)
        let dim_a = cut;
        let dim_b = total - cut;
        let rank_a = dim_a.max(1);
        let rank_b = dim_b.max(1);
        let root_a = random_root(&mut local, rank_a, dim_a);
        let root_b = random_root(&mut local, rank_b, dim_b);
        let mu_a = random_vector(&mut local, dim_a, 1.0);
        let mu_b = random_vector(&mut local, dim_b, 1.0);

        // Build block-local CanonicalPenalty objects using the construction
        // primitives we can reach. `from_dense_root_with_mean` is the only
        // public constructor; we therefore embed each block as a full-width
        // root and rely on the convention that a non-block penalty has
        // col_range = 0..total_dim. To still exercise *block* composition,
        // we manually inject zeros into a wide root and treat the blocks as
        // dense, then verify additivity at the quadratic level.
        let mut wide_a = Array2::<f64>::zeros((rank_a, total));
        wide_a.slice_mut(ndarray::s![.., 0..dim_a]).assign(&root_a);
        let mut wide_b = Array2::<f64>::zeros((rank_b, total));
        wide_b
            .slice_mut(ndarray::s![.., dim_a..total])
            .assign(&root_b);
        let mut wide_mu_a = Array1::<f64>::zeros(total);
        wide_mu_a.slice_mut(ndarray::s![0..dim_a]).assign(&mu_a);
        let mut wide_mu_b = Array1::<f64>::zeros(total);
        wide_mu_b.slice_mut(ndarray::s![dim_a..total]).assign(&mu_b);

        let cp_a = CanonicalPenalty::from_dense_root_with_mean(wide_a, total, wide_mu_a);
        let cp_b = CanonicalPenalty::from_dense_root_with_mean(wide_b, total, wide_mu_b);

        let beta = random_vector(&mut local, total, 2.0);

        let q_a = canonical_shifted_quadratic(&cp_a, &beta);
        let q_b = canonical_shifted_quadratic(&cp_b, &beta);

        // Joint reference: directly via beta_block-style accounting.
        let beta_a = beta.slice(ndarray::s![0..dim_a]).to_owned();
        let beta_b = beta.slice(ndarray::s![dim_a..total]).to_owned();
        let ref_a = reference_shifted_quadratic(&root_a, &beta_a, &mu_a, 1.0);
        let ref_b = reference_shifted_quadratic(&root_b, &beta_b, &mu_b, 1.0);

        let lhs = q_a + q_b;
        let rhs = ref_a + ref_b;
        let scale = 1.0 + lhs.abs() + rhs.abs();
        assert!(
            (lhs - rhs).abs() <= 1e-12 * scale,
            "block sum mismatch seed={seed}: lhs={lhs}, rhs={rhs}"
        );
    }
}

// ---------------------------------------------------------------------------
// 8. REML invariance under (μ, β-init) joint shift.
// Requires a public REML driver; we cannot launch a real fit from this file
// without entangling with the build-red WIP. The test is written against the
// invariant directly: shifting both μ and a hypothetical converged β by the
// same vector δ leaves the centered quadratic — the only place μ enters the
// REML/PIRLS objective — unchanged.
// ---------------------------------------------------------------------------

#[test]
fn joint_shift_of_mean_and_beta_leaves_quadratic_invariant() {
    let mut rng = Rng::new(0x28);
    for seed in 0..50u64 {
        let mut local = Rng::new(0x28_0000 ^ seed);
        let dim = 3 + ((seed as usize) % 6);
        let rank = dim.max(1);
        let root = random_root(&mut local, rank, dim);
        let mean = random_vector(&mut local, dim, 1.0);
        let beta = random_vector(&mut local, dim, 1.0);
        let delta = random_vector(&mut local, dim, 3.0);

        let cp_base = CanonicalPenalty::from_dense_root_with_mean(root.clone(), dim, mean.clone());
        let cp_shift =
            CanonicalPenalty::from_dense_root_with_mean(root.clone(), dim, mean.clone() + &delta);

        let q_base = canonical_shifted_quadratic(&cp_base, &beta);
        let q_shift = canonical_shifted_quadratic(&cp_shift, &(beta.clone() + &delta));
        let scale = 1.0 + q_base.abs() + q_shift.abs();
        assert!(
            (q_base - q_shift).abs() <= 1e-12 * scale,
            "joint (μ,β) shift not invariant at seed={seed}: {q_base} vs {q_shift}"
        );
    }
}

// ---------------------------------------------------------------------------
// 9. Null-space component of μ is invisible to the penalty.
// ---------------------------------------------------------------------------

#[test]
fn nullspace_component_of_prior_mean_is_invisible() {
    let mut rng = Rng::new(0x39);
    let dim: usize = 6;
    // Build a low-rank root so null(S) is non-trivial (rank < dim).
    let rank: usize = 4;
    for seed in 0..20u64 {
        let mut local = Rng::new(0x39_0000 ^ seed);
        let root = random_root(&mut local, rank, dim);
        // Construct a vector orthogonal to range(root.t()) (i.e. in null(S)):
        // solve root · v = 0 by picking any vector in the kernel of `root`.
        // For a random `rank x dim` matrix with rank < dim, we obtain a
        // nullspace vector by QR-like Gram-Schmidt against root's rows.
        let mut v = random_vector(&mut local, dim, 1.0);
        for r in 0..rank {
            let row = root.row(r).to_owned();
            let dot = row.dot(&v) / row.dot(&row).max(1e-300);
            v = v - &(row.mapv(|x| x * dot));
        }
        // Sanity: v ∈ null(root) ⇒ S v = root.t()·(root·v) = 0.
        let rv = root.dot(&v);
        assert!(
            rv.dot(&rv).sqrt() <= 1e-10,
            "null-space construction failed (seed={seed}): ||root v|| = {}",
            rv.dot(&rv).sqrt()
        );

        let mu = random_vector(&mut local, dim, 1.0);
        let mu_plus_null = &mu + &v;

        let cp_mu = CanonicalPenalty::from_dense_root_with_mean(root.clone(), dim, mu.clone());
        let cp_mu_null =
            CanonicalPenalty::from_dense_root_with_mean(root.clone(), dim, mu_plus_null);

        for _ in 0..10 {
            let beta = random_vector(&mut local, dim, 2.0);
            let q1 = canonical_shifted_quadratic(&cp_mu, &beta);
            let q2 = canonical_shifted_quadratic(&cp_mu_null, &beta);
            let scale = 1.0 + q1.abs() + q2.abs();
            assert!(
                (q1 - q2).abs() <= 1e-10 * scale,
                "null-space-of-S component of μ leaked into quadratic \
                 (seed={seed}): q1={q1}, q2={q2}"
            );
        }
    }
}

// ---------------------------------------------------------------------------
// 10. Length mismatch must be rejected.
// `CanonicalPenalty::from_dense_root_with_mean` uses `debug_assert_eq!`; the
// `PenaltyCoordinate` constructors use `assert_eq!` (panicking in release).
// We probe via a panic catch on a debug-built binary.
// ---------------------------------------------------------------------------

#[test]
fn mismatched_prior_mean_length_panics() {
    use std::panic;
    let mut rng = Rng::new(0x4A);
    let dim: usize = 5;
    let rank: usize = 5;
    let root = random_root(&mut rng, rank, dim);
    let bad_mean = Array1::<f64>::zeros(dim + 1);

    let result = panic::catch_unwind(panic::AssertUnwindSafe(|| {
        let _ = CanonicalPenalty::from_dense_root_with_mean(root.clone(), dim, bad_mean);
    }));
    // In release the debug_assert won't fire; we therefore accept either a
    // panic OR a successfully-constructed (but malformed) penalty. The bug
    // we want to catch is silent acceptance in *debug* builds, which is what
    // `cargo test` defaults to.
    if cfg!(debug_assertions) {
        assert!(
            result.is_err(),
            "mismatched prior_mean length must panic in debug builds"
        );
    }
}
