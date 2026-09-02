//! zz_measure diagnostic (2026-07-10, #2234 blocker): every Python-entry
//! `sae_manifold_fit` on HEAD refuses to mint — the outer search descends,
//! then freezes with a bit-stable projected gradient (synthetic planted
//! circle: 63 iterations, objective 1.216074e2, |g_proj| = 1.381e0) and the
//! #2241 certified-termination contract refuses. Lane-consistency (#1224) and
//! every FD gate PASS, so this drives the SAME planted circle through the
//! plain RUST engine (seed builders + `OuterProblem`) to split the fault:
//!
//! - engine converges here  ⇒ the stall lives in the pyffi orchestration
//!   above the engine (topology/promotion/alternation), not the optimizer;
//! - engine stalls here     ⇒ run per-coordinate central differences of
//!   `eval_cost` against `eval`'s analytic gradient AT the stalled ρ and
//!   print both (the desync, if any, named coordinate by coordinate).

// `manifold/mod.rs` declares this module as
// `#[cfg(test)] mod tests_stall_diagnostic_2234;` — its single declaration. Saying so in-file
// makes the test scope a claim the compiler enforces rather than one the
// filename merely implies, which is what puts the fixture helpers below in
// the same scope as the `#[test]` fns they serve.
#![cfg(test)]
use super::*;

fn planted_circle_cloud() -> (Array2<f64>, usize) {
    // Mirrors the frozen #2253 weekday-L17 discriminator after its exact
    // orthonormal reduction: K=1, d_atom=1, n=42, p=48. The prior n=200,
    // p=8, d_atom=3 fixture did not enter the single-circle log-det seam whose
    // analytic smoothing derivative is under audit.
    let n = 42usize;
    let p = 48usize;
    let mut state = 0x2468_ace0_1357_9bdfu64;
    let mut unit = move || {
        // LCG → [0,1); NO rand, NO clock (repo #932 rules).
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((state >> 11) as f64) / ((1u64 << 53) as f64)
    };
    let two_pi = std::f64::consts::TAU;
    let b0: Vec<f64> = (0..p).map(|_| 2.0 * unit() - 1.0).collect();
    let b1: Vec<f64> = (0..p).map(|_| 2.0 * unit() - 1.0).collect();
    let mut z = Array2::<f64>::zeros((n, p));
    for i in 0..n {
        let theta = two_pi * unit();
        for j in 0..p {
            let noise = 0.01 * (2.0 * unit() - 1.0);
            z[[i, j]] = theta.cos() * b0[j] + theta.sin() * b1[j] + noise;
        }
    }
    (z, p)
}

struct LogdetAuditPoint {
    term: SaeManifoldTerm,
    criterion: SaeCriterion,
    components: SaeOuterRhoGradientComponents,
    raw_cache_components: Result<SaeOuterRhoGradientComponents, String>,
    log_det: f64,
    kkt_grad_norm: f64,
    quotient_kkt_grad_norm: f64,
    kkt_tolerance: f64,
    branch_certificate: BranchCertificate,
    exact_chart_gauge_count: usize,
    solver_gauge_count: usize,
    cache_beta_quotient_dim: usize,
    loss_smoothness: f64,
    raw_smoothness_sum: f64,
    smooth_renorm: f64,
}

fn decoder_frames_match_exactly(
    current: &SaeManifoldTerm,
    expected: &[Option<GrassmannFrame>],
) -> bool {
    current.atoms.len() == expected.len()
        && current.atoms.iter().zip(expected).all(|(atom, saved)| {
            match (&atom.decoder_frame, saved) {
                (Some(current), Some(expected)) => {
                    current.frame() == expected.frame()
                        && current.gauge_singular_values() == expected.gauge_singular_values()
                }
                (None, None) => true,
                _ => false,
            }
        })
}

