//! #2027 — deterministic repro for the K≥2 whitened dictionary CO-COLLAPSE, and
//! the regression guard for the disjoint-subspace / ownership-anchor / reseed-
//! hysteresis fix.
//!
//! Two planted circles live in DISJOINT 2-planes of an ambient `p`-dim cloud; the
//! per-column-standardized ("whitened") target is their sum, so a faithful K=2
//! reconstruction REQUIRES both atoms to carry signal on different subspaces.
//! Before the fix the joint decoder refit at the co-collapse reseed re-spread one
//! residual direction across both atoms and the gate let them trade rows, so the
//! dictionary re-symmetrised into a single shared basin and the reconstruction EV
//! collapsed to the no-signal level. With the greedy disjoint-subspace decoder
//! refit + soft row-ownership anchor + reseed cooldown the two atoms hold distinct
//! territories and the fit recovers a materially positive EV.

use super::tests::deterministic_circle_noise;
use super::*;

/// Whitened two-circle target: circle A lives on the even ambient columns, circle
/// B on the odd ones (disjoint deterministic near-orthonormal 2-frames), driven by
/// two INCOMMENSURATE phases so the circles are not row-aligned. Each column is
/// standardized to zero mean / unit variance (the whitening proxy that puts both
/// circles on a common scale, the regime the real-data co-collapse lives in).
fn two_circle_whitened_target(n: usize, p: usize, sigma: f64) -> Array2<f64> {
    let mut fa = Array2::<f64>::zeros((2, p));
    let mut fb = Array2::<f64>::zeros((2, p));
    for j in 0..p {
        if j % 2 == 0 {
            fa[[0, j]] = deterministic_circle_noise(j, 0);
            fa[[1, j]] = deterministic_circle_noise(j, 1);
        } else {
            fb[[0, j]] = deterministic_circle_noise(j, 2);
            fb[[1, j]] = deterministic_circle_noise(j, 3);
        }
    }
    for f in [&mut fa, &mut fb] {
        for r in 0..2 {
            let nrm = (0..p).map(|j| f[[r, j]] * f[[r, j]]).sum::<f64>().sqrt();
            for j in 0..p {
                f[[r, j]] /= nrm.max(1.0e-300);
            }
        }
    }
    let mut z = Array2::<f64>::zeros((n, p));
    for row in 0..n {
        let ta = std::f64::consts::TAU * (row as f64) / (n as f64);
        let tb = std::f64::consts::TAU * (2.0 * row as f64 + 0.37) / (n as f64);
        let (ca, sa) = (ta.cos(), ta.sin());
        let (cb, sb) = (tb.cos(), tb.sin());
        for j in 0..p {
            z[[row, j]] = ca * fa[[0, j]]
                + sa * fa[[1, j]]
                + cb * fb[[0, j]]
                + sb * fb[[1, j]]
                + sigma * deterministic_circle_noise(row, j + 7);
        }
    }
    for j in 0..p {
        let mut mean = 0.0_f64;
        for row in 0..n {
            mean += z[[row, j]];
        }
        mean /= n as f64;
        let mut var = 0.0_f64;
        for row in 0..n {
            let d = z[[row, j]] - mean;
            var += d * d;
        }
        let sd = (var / n as f64).sqrt().max(1.0e-12);
        for row in 0..n {
            z[[row, j]] = (z[[row, j]] - mean) / sd;
        }
    }
    z
}

/// Two circles of UNEQUAL amplitude on disjoint output-channel parities: circle A
/// (unit amplitude, winding 1) on the even channels {0, 2}, circle B (`amp_b < 1`)
/// on the odd channels {1, 3}. Circle B winds THREE times per revolution of A, a
/// harmonic index BEYOND the atoms' order-2 (`m = 5`) chart span, so neither
/// circle lies in the other's harmonic reach: atom A's decoder cannot absorb B as
/// one of its own harmonics (which an absorbable 2× winding would let it do,
/// collapsing the residual to zero and making the second reseed correctly
/// terminal). A DOMINATES, so a co-collapse reseed that seeds both atoms from the
/// same residual reads circle A for BOTH (re-collision); only a sequential-
/// deflation reseed — peel A onto atom 0, then seed atom 1 from what A genuinely
/// leaves behind (circle B) — separates them onto the two disjoint circles.
fn two_amplitude_circle_target(n: usize, amp_b: f64) -> Array2<f64> {
    let p = 4usize;
    let mut z = Array2::<f64>::zeros((n, p));
    for row in 0..n {
        let ta = std::f64::consts::TAU * (row as f64) / (n as f64);
        let tb = std::f64::consts::TAU * (3.0 * row as f64 + 0.37) / (n as f64);
        z[[row, 0]] = ta.cos();
        z[[row, 2]] = ta.sin();
        z[[row, 1]] = amp_b * tb.cos();
        z[[row, 3]] = amp_b * tb.sin();
    }
    z
}

