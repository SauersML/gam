//! Owed-work regression for #1422 — the mixed-periodicity (cylinder / torus)
//! Duchon smooth must be POSITIVE SEMIDEFINITE.
//!
//! ## The defect (now fixed)
//!
//! The mixed-periodicity Duchon penalty used to be built from a polyharmonic
//! kernel `r^(2p+2s−d)` evaluated at a *generalized chord distance* that wrapped
//! periodic axes. That chord-polyharmonic kernel is only CONDITIONALLY positive
//! definite, and the constants-only constraint left behind it does NOT annihilate
//! its indefinite linear modes — so the realized center Gram had genuinely
//! negative eigenvalues (cylinder λmin ≈ −0.43, torus λmin ≈ −0.89). A penalty
//! `S = ZᵀKZ` inheriting those negative modes is not a valid roughness penalty:
//! the REML/LAML `log|λS|₊` and the whitening transform break on the negative
//! spectrum.
//!
//! The fix replaces the chord-polyharmonic kernel with an ADDITIVE (ANOVA) sum of
//! per-axis reproducing kernels — the periodic Bernoulli Green's function on
//! periodic axes and the 1-D Sobolev smoothing-spline kernel on non-periodic axes
//! — each individually PSD. A sum of PSD kernels is PSD, so the constrained Gram
//! `Ω = Zᵀ K_CC Z` is PSD by congruence.
//!
//! ## What this guards
//!
//! At fast basis scope (no end-to-end fit): the realized primary penalty of a
//! mixed-periodicity Duchon basis — on both a cylinder (one periodic axis) and a
//! torus (two periodic axes) — has a minimum eigenvalue ≥ −tol (PSD), while still
//! being a non-trivial penalty (a strictly positive largest eigenvalue). The
//! pre-fix chord-polyharmonic kernel would fail the PSD floor by a wide margin.
//!
//! Reference-as-truth: the PSD property is an intrinsic mathematical requirement
//! of a roughness penalty, asserted on gam's own realized penalty matrix — never
//! against another tool's output.

use gam::terms::basis::{
    CenterStrategy, DuchonBasisSpec, DuchonNullspaceOrder, build_duchon_basis,
};
use ndarray::{Array1, Array2};

/// Smallest eigenvalue of a small symmetric matrix via a cyclic Jacobi sweep
/// (dependency-free, exact for the few-column Grams these tests produce).
fn min_eigenvalue_symmetric(m: &Array2<f64>) -> f64 {
    let n = m.nrows();
    assert_eq!(n, m.ncols(), "eigenvalue helper needs a square matrix");
    if n == 0 {
        return 0.0;
    }
    let mut a = m.clone();
    // Symmetrize defensively (the penalty is symmetric up to rounding).
    for i in 0..n {
        for j in (i + 1)..n {
            let avg = 0.5 * (a[[i, j]] + a[[j, i]]);
            a[[i, j]] = avg;
            a[[j, i]] = avg;
        }
    }
    for _sweep in 0..100 {
        // Largest off-diagonal magnitude.
        let mut off = 0.0_f64;
        for i in 0..n {
            for j in (i + 1)..n {
                off = off.max(a[[i, j]].abs());
            }
        }
        if off < 1e-15 {
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
    let mut lambda = Array1::<f64>::zeros(n);
    for i in 0..n {
        lambda[i] = a[[i, i]];
    }
    lambda.iter().cloned().fold(f64::INFINITY, f64::min)
}

fn max_eigenvalue_symmetric(m: &Array2<f64>) -> f64 {
    // The Jacobi sweep above leaves the diagonal as the eigenvalues; reuse it.
    let n = m.nrows();
    if n == 0 {
        return 0.0;
    }
    let mut a = m.clone();
    for i in 0..n {
        for j in (i + 1)..n {
            let avg = 0.5 * (a[[i, j]] + a[[j, i]]);
            a[[i, j]] = avg;
            a[[j, i]] = avg;
        }
    }
    for _sweep in 0..100 {
        let mut off = 0.0_f64;
        for i in 0..n {
            for j in (i + 1)..n {
                off = off.max(a[[i, j]].abs());
            }
        }
        if off < 1e-15 {
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
    (0..n).map(|i| a[[i, i]]).fold(f64::NEG_INFINITY, f64::max)
}

/// Cylinder: axis 0 periodic (period 1), axis 1 non-periodic. A spread of
/// distinct points so the Gram is non-degenerate.
fn cylinder_data() -> Array2<f64> {
    Array2::from_shape_vec(
        (8, 2),
        vec![
            0.05, -0.6, 0.20, 0.1, 0.37, 0.9, 0.51, -0.3, 0.66, 0.5, 0.78, -0.1, 0.90, 0.8, 0.97,
            -0.45,
        ],
    )
    .unwrap()
}

/// Torus: both axes periodic (periods 1 and 1).
fn torus_data() -> Array2<f64> {
    Array2::from_shape_vec(
        (8, 2),
        vec![
            0.05, 0.10, 0.20, 0.55, 0.37, 0.88, 0.51, 0.22, 0.66, 0.71, 0.78, 0.40, 0.90, 0.15,
            0.97, 0.63,
        ],
    )
    .unwrap()
}

fn mixed_periodicity_spec(data: &Array2<f64>, periodic: Vec<Option<f64>>) -> DuchonBasisSpec {
    DuchonBasisSpec {
        radial_reparam: None,
        center_strategy: CenterStrategy::UserProvided(data.clone()),
        periodic: Some(periodic),
        // Pure polyharmonic spectrum: mixed-periodicity Duchon supports only
        // length_scale=None / power=0 (the supported path).
        length_scale: None,
        power: 0.0,
        nullspace_order: DuchonNullspaceOrder::Linear, // m = 2 (cylinder null = {1, y})
        identifiability: Default::default(),
        aniso_log_scales: None,
        operator_penalties: Default::default(),
        boundary: Default::default(),
    }
}

/// MERGE GATE (#1422): the realized mixed-periodicity Duchon penalty is PSD on
/// both the cylinder and the torus (the chord-polyharmonic kernel the fix
/// replaced produced λmin ≈ −0.43 / −0.89), and it is a non-trivial penalty
/// (strictly positive top eigenvalue).
#[test]
fn mixed_periodicity_duchon_penalty_is_psd_cylinder_and_torus_1422() {
    for (label, data, periodic) in [
        ("cylinder", cylinder_data(), vec![Some(1.0), None]),
        ("torus", torus_data(), vec![Some(1.0), Some(1.0)]),
    ] {
        let spec = mixed_periodicity_spec(&data, periodic);
        let built = build_duchon_basis(data.view(), &spec)
            .unwrap_or_else(|e| panic!("{label} mixed-periodicity Duchon build failed: {e:?}"));
        assert!(
            !built.penalties.is_empty(),
            "{label}: a mixed-periodicity Duchon basis must carry a primary penalty"
        );
        let s = &built.penalties[0];
        let lambda_min = min_eigenvalue_symmetric(s);
        let lambda_max = max_eigenvalue_symmetric(s);
        // PSD floor scaled to the penalty magnitude: a roughness penalty must
        // have no genuinely-negative eigenvalues. The pre-fix chord-polyharmonic
        // kernel violated this by O(0.1)·scale; rounding noise is ~1e-12·scale.
        let scale = lambda_max.abs().max(1.0);
        assert!(
            lambda_min > -1e-9 * scale,
            "{label}: mixed-periodicity Duchon penalty is NOT PSD — λmin = {lambda_min:.6e} \
             (λmax = {lambda_max:.6e}); the additive ANOVA kernel fix (#1422) must keep it PSD, \
             the old chord-polyharmonic kernel gave λmin ≈ −0.43 (cyl) / −0.89 (torus)"
        );
        assert!(
            lambda_max > 1e-8,
            "{label}: mixed-periodicity Duchon penalty must be a NON-TRIVIAL roughness penalty \
             (positive top eigenvalue); got λmax = {lambda_max:.6e}"
        );
    }
}
