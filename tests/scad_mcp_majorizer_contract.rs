//! Regression: `ScadMcpPenalty`'s PSD majorizer obeys its trait contract from
//! several angles, not just the single near-zero probe of the bug-hunt test for
//! #804.
//!
//! The `AnalyticPenalty` contract for `psd_majorizer_diag` is a PSD majorizer
//! `B` with `B ⪰ 0` AND `B ⪰ ∂²P` everywhere. Beyond those two legs, a *good*
//! majorizer is also tight: it should equal the exact Hessian wherever the
//! penalty is already convex (SCAD's first L¹ region and both variants' flat
//! tail) and exceed it by exactly the dropped concave constant in the concave
//! taper. This test sweeps a fine grid across every region for both variants
//! (smoothed and sharp `ε`, fixed and learnable weight) and checks all of that,
//! plus that `psd_majorizer_hvp` is the diagonal `B ⊙ v` (coordinate
//! separability) rather than the indefinite exact HVP.

use gam::terms::analytic_penalties::{AnalyticPenalty, PenaltyConcavity, PsiSlice, ScadMcpPenalty};
use ndarray::{Array1, ArrayView1};

fn sweep() -> Vec<f64> {
    // Dense coverage of t from deep-active to far-past-cutoff, both signs and 0.
    let mut v = vec![0.0_f64];
    let mut t = 1e-3;
    while t <= 12.0 {
        v.push(t);
        v.push(-t);
        t *= 1.3;
    }
    v
}

#[test]
fn majorizer_dominates_and_is_psd_across_all_regions() {
    let weight = 0.7_f64;
    for &eps in &[1e-4_f64, 1e-2, 0.1] {
        for variant in [PenaltyConcavity::Mcp, PenaltyConcavity::Scad] {
            for &learnable in &[false, true] {
                let gamma = match variant {
                    PenaltyConcavity::Mcp => 2.5,
                    PenaltyConcavity::Scad => 3.7,
                };
                let probe = Array1::from(sweep());
                let n = probe.len();
                let target = PsiSlice::full(n, Some(1));
                let pen = ScadMcpPenalty::new(target, weight, n, gamma, eps, variant, learnable)
                    .expect("valid penalty");
                // Learnable weight resolves through rho: λ = weight·exp(rho).
                let rho: Array1<f64> = if learnable {
                    Array1::from(vec![0.3_f64])
                } else {
                    Array1::zeros(0)
                };
                let lam = if learnable {
                    weight * 0.3_f64.exp()
                } else {
                    weight
                };

                let maj = pen
                    .psd_majorizer_diag(probe.view(), rho.view())
                    .expect("diagonal majorizer exists");
                let hess = pen
                    .hessian_diag(probe.view(), rho.view())
                    .expect("diagonal Hessian exists");

                for i in 0..n {
                    let t = probe[i];
                    let r = (t * t + eps * eps).sqrt();
                    let tag = format!("{variant:?} eps={eps} learnable={learnable} t={t} r={r}");

                    // Leg 1: B ⪰ 0.
                    assert!(maj[i] >= -1e-12, "B<0 ({}): {}", tag, maj[i]);
                    // Leg 2: B ⪰ ∂²P.
                    assert!(
                        maj[i] >= hess[i] - 1e-9,
                        "B<H ({}): B={} H={}",
                        tag,
                        maj[i],
                        hess[i]
                    );
                    // Finiteness (no division blow-ups; r >= eps > 0).
                    assert!(maj[i].is_finite(), "non-finite B ({})", tag);

                    // Tightness: where the penalty is convex the majorizer is
                    // the exact Hessian; in the concave taper it exceeds it by
                    // exactly the dropped constant.
                    let gap = maj[i] - hess[i];
                    match variant {
                        PenaltyConcavity::Mcp => {
                            if r <= gamma * lam {
                                // Concave taper: gap == 1/γ.
                                assert!(
                                    (gap - 1.0 / gamma).abs() < 1e-9,
                                    "MCP active gap {} != 1/γ ({})",
                                    gap,
                                    tag
                                );
                            } else {
                                // Flat tail: both 0.
                                assert!(gap.abs() < 1e-12, "MCP tail gap {} ({})", gap, tag);
                            }
                        }
                        PenaltyConcavity::Scad => {
                            if r <= lam {
                                // Convex L¹ region: majorizer == Hessian.
                                assert!(gap.abs() < 1e-9, "SCAD L1 gap {} ({})", gap, tag);
                            } else if r <= gamma * lam {
                                // Concave middle: gap == 1/(γ−1).
                                assert!(
                                    (gap - 1.0 / (gamma - 1.0)).abs() < 1e-9,
                                    "SCAD mid gap {} != 1/(γ-1) ({})",
                                    gap,
                                    tag
                                );
                            } else {
                                assert!(gap.abs() < 1e-12, "SCAD tail gap {} ({})", gap, tag);
                            }
                        }
                    }
                }

                // psd_majorizer_hvp must be the diagonal applied to v, NOT the
                // indefinite exact hvp. Probe with a non-trivial vector.
                let v: Array1<f64> =
                    Array1::from((0..n).map(|k| ((k % 5) as f64) - 2.0).collect::<Vec<_>>());
                let hv = pen.psd_majorizer_hvp(probe.view(), rho.view(), v.view());
                for i in 0..n {
                    assert!(
                        (hv[i] - maj[i] * v[i]).abs() < 1e-12,
                        "hvp != B⊙v at {i}: {} vs {}",
                        hv[i],
                        maj[i] * v[i]
                    );
                }
            }
        }
    }
}

/// The majorizer of a *block* of mixed-magnitude coordinates is a genuine PSD
/// diagonal: `vᵀ B v ≥ 0` for arbitrary `v`, and `vᵀ B v ≥ vᵀ H v` whenever the
/// exact Hessian's quadratic form would have been driven negative by the
/// concave entries. This is the property the inner solve actually relies on.
#[test]
fn block_quadratic_form_is_nonnegative_and_dominates() {
    let probe = ndarray::array![0.02_f64, 0.3, 0.6, 0.9, -1.2, -0.04, 0.15, 2.5];
    let n = probe.len();
    let target = PsiSlice::full(n, Some(1));
    let rho = Array1::<f64>::zeros(0);

    for (variant, gamma) in [
        (PenaltyConcavity::Mcp, 3.0_f64),
        (PenaltyConcavity::Scad, 3.7_f64),
    ] {
        let pen = ScadMcpPenalty::new(target.clone(), 0.5, n, gamma, 1e-4, variant, false)
            .expect("valid penalty");
        let maj = pen.psd_majorizer_diag(probe.view(), rho.view()).unwrap();
        let hess = pen.hessian_diag(probe.view(), rho.view()).unwrap();

        // The exact Hessian's quadratic form on the all-ones direction is
        // negative here (proves the bug was real); the majorizer's is ≥ 0.
        let ones = Array1::<f64>::ones(n);
        let qf = |d: &Array1<f64>, w: ArrayView1<'_, f64>| -> f64 {
            (0..n).map(|i| d[i] * w[i] * w[i]).sum()
        };
        assert!(
            qf(&hess, ones.view()) < 0.0,
            "{variant:?}: exact Hessian quadratic form should be negative on this block"
        );
        assert!(
            qf(&maj, ones.view()) >= -1e-12,
            "{variant:?}: majorizer quadratic form must be ≥ 0"
        );
        // Domination for several arbitrary directions.
        for seed in 0..6 {
            let d: Array1<f64> = Array1::from(
                (0..n)
                    .map(|k| (((k * 7 + seed * 3) % 11) as f64) - 5.0)
                    .collect::<Vec<_>>(),
            );
            assert!(
                qf(&maj, d.view()) >= qf(&hess, d.view()) - 1e-9,
                "{variant:?}: majorizer must dominate Hessian quadratic form (seed {seed})"
            );
        }
    }
}
