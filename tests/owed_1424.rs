//! Owed-work regression for #1424 — the high-dimensional hybrid Duchon–Matérn
//! kernel must stay POSITIVE SEMIDEFINITE (no catastrophic-cancellation negative
//! modes).
//!
//! ## The defect (now fixed)
//!
//! The hybrid Duchon–Matérn radial kernel was evaluated through its
//! partial-fraction expansion `Σ a_m/ρ^{2m} + Σ b_n/(κ²+ρ²)ⁿ`. The
//! partial-fraction COEFFICIENTS are mathematically correct, but in the genuine
//! high-dimensional blend (`2p < d`, `s ≥ 1`) the individual polyharmonic /
//! Matérn blocks are enormous and ALTERNATE in sign, while their true sum is a
//! tiny (~1e-13) kernel value — so the sum cancels catastrophically and loses
//! every significant digit, destroying the positive-semidefiniteness of the Gram
//! matrix (and the subsequent Frobenius normalization amplifies the resulting
//! noise). This is NOT a coefficient bug; it is floating-point cancellation, so
//! the fix is a STABLE REFORMULATION, not a coefficient change.
//!
//! The fix evaluates exactly that regime (`duchon_hybrid_stable_integral_applies:
//! s ≥ 1 && 2p < d`) through the cancellation-free single-integral (Schwinger /
//! Schoenberg) representation of the same kernel — a smooth, strictly positive
//! integrand — instead of the alternating partial-fraction sum, and projects the
//! constrained bending Gram onto the PSD cone as a defensive backstop.
//!
//! ## What this guards
//!
//! At fast basis scope (no end-to-end fit): a high-dimensional (`d = 6`) hybrid
//! Duchon–Matérn smooth (`length_scale = Some`, `power = 3` → `s = 3`,
//! constants-only null space → `p = 1`) builds successfully and its realized
//! primary penalty is PSD (`λmin ≥ −tol`) while remaining a non-trivial
//! roughness penalty (`λmax > 0`). The pre-fix alternating partial-fraction sum
//! produced large negative eigenvalues here.
//!
//! The catastrophic-cancellation regime is exactly `2p < d < 2(p+s)`: the
//! kernel must EXIST pointwise (`2(p+s) > d`, here `8 > 6`, so the radial
//! Fourier integral converges at infinity), while the stable single-integral
//! reduction must apply (`2p < d`, here `2 < 6`, so the `w → 0` endpoint is
//! integrable and `b = p + s − d/2 = 1 > 0`). Picking `s = 1` here would give
//! `2(p+s) = 4 < 6`: the kernel does not exist (the validator correctly rejects
//! it), so it cannot probe the cancellation path — `s ≥ 3` is the smallest
//! power that keeps `d = 6` inside the genuine high-dimensional blend.
//!
//! Reference-as-truth: PSD-ness is an intrinsic requirement of a roughness
//! penalty, asserted on gam's own realized penalty — never against another
//! tool's output.

use gam::terms::basis::{
    CenterStrategy, DuchonBasisSpec, DuchonNullspaceOrder, PenaltySource, build_duchon_basis,
};
use ndarray::Array2;

/// Smallest / largest eigenvalue of a symmetric matrix via a cyclic Jacobi sweep
/// (dependency-free; the Grams here are a few-dozen columns at most).
fn jacobi_eigenvalues(m: &Array2<f64>) -> Vec<f64> {
    let n = m.nrows();
    assert_eq!(n, m.ncols(), "eigenvalue helper needs a square matrix");
    let mut a = m.clone();
    for i in 0..n {
        for j in (i + 1)..n {
            let avg = 0.5 * (a[[i, j]] + a[[j, i]]);
            a[[i, j]] = avg;
            a[[j, i]] = avg;
        }
    }
    for _sweep in 0..200 {
        let mut off = 0.0_f64;
        for i in 0..n {
            for j in (i + 1)..n {
                off = off.max(a[[i, j]].abs());
            }
        }
        if off < 1e-14 {
            break;
        }
        for p in 0..n {
            for q in (p + 1)..n {
                let apq = a[[p, q]];
                if apq.abs() < 1e-300 {
                    continue;
                }
                let app = a[[p, p]];
                let aqq = a[[q, q]];
                let phi = 0.5 * (2.0 * apq).atan2(aqq - app);
                let (s, c) = phi.sin_cos();
                for k in 0..n {
                    let akp = a[[k, p]];
                    let akq = a[[k, q]];
                    a[[k, p]] = c * akp - s * akq;
                    a[[k, q]] = s * akp + c * akq;
                }
                for k in 0..n {
                    let apk = a[[p, k]];
                    let aqk = a[[q, k]];
                    a[[p, k]] = c * apk - s * aqk;
                    a[[q, k]] = s * apk + c * aqk;
                }
            }
        }
    }
    (0..n).map(|i| a[[i, i]]).collect()
}

/// `d = 6` hybrid configuration: a deterministic low-discrepancy point cloud in
/// the unit cube so the Gram is non-degenerate but the kernel values are tiny
/// (the regime where the old partial-fraction sum cancelled to noise).
fn high_dim_hybrid_data() -> Array2<f64> {
    let n = 14usize;
    let d = 6usize;
    // Halton-like radical-inverse coordinates for a few small primes — purely
    // deterministic, no external RNG.
    let primes = [2usize, 3, 5, 7, 11, 13];
    let mut v = Vec::with_capacity(n * d);
    for i in 0..n {
        for &base in primes.iter().take(d) {
            v.push(radical_inverse(i + 1, base));
        }
    }
    Array2::from_shape_vec((n, d), v).unwrap()
}

fn radical_inverse(mut i: usize, base: usize) -> f64 {
    let mut f = 1.0_f64;
    let mut r = 0.0_f64;
    while i > 0 {
        f /= base as f64;
        r += f * (i % base) as f64;
        i /= base;
    }
    r
}

/// MERGE GATE (#1424): the high-dimensional hybrid Duchon–Matérn penalty is PSD
/// (the cancellation-free stable-integral kernel evaluation must not produce the
/// large negative eigenvalues the partial-fraction sum did), and is a non-trivial
/// roughness penalty.
#[test]
fn high_dim_hybrid_duchon_matern_penalty_is_psd_1424() {
    let data = high_dim_hybrid_data();
    assert_eq!(
        data.ncols(),
        6,
        "this regression is the d = 6 hybrid regime"
    );
    let spec = DuchonBasisSpec {
        radial_reparam: None,
        center_strategy: CenterStrategy::UserProvided(data.clone()),
        periodic: None,
        // Hybrid Matérn blend: length_scale = Some triggers the
        // partial-fraction / stable-integral kernel; power = 3 → s = 3.
        length_scale: Some(0.75),
        power: 3.0,
        // Constants-only null space → p = 1. With d = 6 this is the
        // catastrophic-cancellation regime 2p < d < 2(p+s) (2 < 6 < 8):
        // `duchon_hybrid_stable_integral_applies` (s ≥ 1 && 2p < d) holds and
        // the kernel exists pointwise (2(p+s) > d), so the stable single
        // integral (b = p + s − d/2 = 1 > 0) replaces the cancelling sum.
        nullspace_order: DuchonNullspaceOrder::Zero,
        identifiability: Default::default(),
        aniso_log_scales: None,
        operator_penalties: Default::default(),
        boundary: Default::default(),
    };

    let built = build_duchon_basis(data.view(), &spec)
        .expect("d=6 hybrid Duchon–Matérn build should succeed via the stable-integral kernel");
    assert!(
        !built.active_penalties.is_empty(),
        "the hybrid Duchon–Matérn basis must carry a primary penalty"
    );
    let s = &built
        .active_penalties
        .iter()
        .find(|penalty| matches!(penalty.info.source, PenaltySource::Primary))
        .expect("hybrid Duchon–Matérn build must retain its primary roughness penalty")
        .matrix;
    let eigs = jacobi_eigenvalues(s);
    let lambda_min = eigs.iter().cloned().fold(f64::INFINITY, f64::min);
    let lambda_max = eigs.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let scale = lambda_max.abs().max(1.0);

    assert!(
        lambda_min > -1e-8 * scale,
        "d=6 hybrid Duchon–Matérn penalty is NOT PSD — λmin = {lambda_min:.6e} \
         (λmax = {lambda_max:.6e}); the stable-integral kernel evaluation (#1424) must avoid the \
         catastrophic-cancellation negative modes the partial-fraction sum produced"
    );
    assert!(
        lambda_max > 1e-8,
        "d=6 hybrid Duchon–Matérn penalty must be a non-trivial roughness penalty; \
         got λmax = {lambda_max:.6e}"
    );
    assert!(
        s.iter().all(|v| v.is_finite()),
        "the realized hybrid penalty must be entirely finite (no cancellation NaNs/Infs)"
    );
}
