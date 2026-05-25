//! RED REPRO: PIRLS joint-Newton "linearized-rate stall early-exit"
//!
//! Reproduces — at the linear-algebra level — the exact diagnostic
//! signature observed in production survival_marginal_slope fits at
//! n=195,780 (5/5 outer seeds rejected). From the production trace:
//!
//!   block_widths   = [12, 11, 10]            # time, marginal, logslope
//!   block_beta_inf = [2.3e-4, 15.3, 20.0]    # time barely moved
//!   block_grad_inf = [5.6e8,  1.5e3, 2.3e3]  # time carries the gradient
//!   cycle 0 unconstrained proposal:  |prop|∞ = 2.173e5
//!   cycle 0 after TR truncation:     |δ|∞    = 20.0
//!   linearized_rel = ‖g+Hδ‖∞ / (1+‖g‖∞)  ≈ 0.97  for 15+ cycles
//!   exit reason    = "linearized-rate stall early-exit:
//!                    linearized_rel ≥ 0.9 for 15 consecutive cycles"
//!
//! Root cause (under test): gam's joint-Newton trust region uses a
//! single scalar L2 norm over the concatenated δ vector
//! (`src/families/custom_family.rs:10835-10847`). When ONE block has a
//! near-null direction in its joint Hessian, the unconstrained Newton
//! step is huge along that direction. The L2 rescale uniformly multiplies
//! the entire δ by `radius / ‖δ‖₂ ≈ 1e-4`, crushing every block's
//! Newton step by the same factor. The well-conditioned blocks lose
//! their meaningful contribution; their gradient is essentially
//! unchanged at the next cycle. `‖g + Hδ‖∞ ≈ ‖g‖∞` → `linearized_rel ≈ 1`
//! → 15-cycle stall detector fires → seed rejected.

use ndarray::{Array1, Array2};

// ────────────────────────────────────────────────────────────────────
// Verbatim copies of gam's internal trust-region primitives.
// Source: src/families/custom_family.rs:10835-10847 (gam v0.2.1).
// These are private fns inside the gam crate; we inline them so the
// red repro is independent of test-mod compile state and survives
// in-progress refactors of crate::test_support.
// ────────────────────────────────────────────────────────────────────

fn joint_trust_region_step_norm(delta: &Array1<f64>) -> f64 {
    delta.iter().map(|v| v * v).sum::<f64>().sqrt()
}

fn truncate_joint_step_to_radius(delta: &mut Array1<f64>, radius: f64) -> f64 {
    let norm = joint_trust_region_step_norm(delta);
    if norm.is_finite() && norm > radius && radius > 0.0 {
        let scale = radius / norm;
        delta.mapv_inplace(|v| v * scale);
        radius
    } else {
        norm
    }
}

// ────────────────────────────────────────────────────────────────────
// Block-diagonal Hessian fixture.
// ────────────────────────────────────────────────────────────────────

/// Build a synthetic joint Hessian H = blockdiag(H_time, H_marg, H_logsl)
/// where H_logsl carries one near-null direction (σ_min ≈ 1e-10), the
/// other blocks are well-conditioned, and a gradient g aligned so that:
///
///   * δ̂ = H⁻¹ (-g) has a single huge component (~2e5) in the
///     near-null direction of the logslope block
///   * the other components of δ̂ are O(1)
///
/// Returns (H, g, unconstrained δ̂).
fn build_anisotropic_block_fixture() -> (Array2<f64>, Array1<f64>, Array1<f64>) {
    const TIME_W: usize = 12;
    const MARG_W: usize = 11;
    const LOGSL_W: usize = 10;
    const P: usize = TIME_W + MARG_W + LOGSL_W;

    let mut h = Array2::<f64>::zeros((P, P));
    let mut g = Array1::<f64>::zeros(P);

    // Time block: well-conditioned (large eigenvalues, no near-null
    // direction) but its FIRST coordinate carries the dominant gradient.
    // Matching production block_grad_inf[time] = 5.6e8: pick H_time[0,0]
    // such that the unconstrained Newton step in this direction equals
    // the production block_beta_inf[time] / rescale ≈ 2.5 (i.e., the
    // unconstrained step is modest; the L2 rescale then crushes it to
    // O(1e-4) because the OTHER block's near-null direction dominates
    // the L2 norm). H_time[0,0] = 5.6e8 / 2.5 = 2.24e8.
    h[[0, 0]] = 2.24e8;
    g[0] = -5.6e8;
    for i in 1..TIME_W {
        h[[i, i]] = 1.0 + (i as f64) * 0.3;
        g[i] = -0.3 - 0.07 * (i as f64);
    }

    // Marginal block: well-conditioned diag with eigenvalues in [1.2, 3.5].
    for j in 0..MARG_W {
        let row = TIME_W + j;
        h[[row, row]] = 1.2 + (j as f64) * 0.2;
        g[row] = if j == 0 {
            -0.9
        } else {
            -0.2 + 0.01 * (j as f64)
        };
    }

    // Logslope block: one near-null direction (σ_min = 1e-10) plus 9
    // well-conditioned diagonal entries. The first coordinate of the
    // logslope block IS the near-null direction; its gradient component
    // is O(1) but its inverse-Hessian image is ~1e10, the source of
    // the huge unconstrained Newton step.
    for k in 0..LOGSL_W {
        let row = TIME_W + MARG_W + k;
        if k == 0 {
            // Near-singular eigenvalue of the joint Hessian. Pick the
            // magnitude so that g_logslope[0] / H_logslope[0,0] gives
            // the production cycle-0 |prop|∞ = 2.173e5.
            h[[row, row]] = 1.0e-5;
            g[row] = -2.173e0;
        } else {
            h[[row, row]] = 1.5 + (k as f64) * 0.1;
            g[row] = -0.4;
        }
    }

    // Unconstrained Newton step δ̂ = H⁻¹(-g). For our diagonal H this is
    // trivially δ̂_i = -g_i / H_{i,i}.
    let mut delta_hat = Array1::<f64>::zeros(P);
    for i in 0..P {
        delta_hat[i] = -g[i] / h[[i, i]];
    }

    (h, g, delta_hat)
}

// ────────────────────────────────────────────────────────────────────
// The test.
// ────────────────────────────────────────────────────────────────────

#[test]
fn joint_newton_isotropic_l2_tr_produces_production_linearized_rate_stall() {
    let (h, g, delta_hat) = build_anisotropic_block_fixture();

    // PRODUCTION CHECK 1: unconstrained Newton step inf-norm matches
    // production cycle 0 (|prop|∞ = 2.173e5).
    let prop_inf = delta_hat.iter().map(|v| v.abs()).fold(0.0_f64, f64::max);
    assert!(
        (prop_inf - 2.173e5).abs() < 5.0,
        "fixture |prop|∞ = {:.3e} should match production cycle-0 value 2.173e5",
        prop_inf,
    );

    // Snapshot block dimensions & unconstrained block inf-norms.
    const TIME_W: usize = 12;
    const MARG_W: usize = 11;
    let block_inf = |start: usize, len: usize, v: &Array1<f64>| {
        v.slice(ndarray::s![start..start + len])
            .iter()
            .map(|x| x.abs())
            .fold(0.0_f64, f64::max)
    };
    let pre_time_inf = block_inf(0, TIME_W, &delta_hat);
    let pre_marg_inf = block_inf(TIME_W, MARG_W, &delta_hat);

    // Apply gam's actual joint-Newton trust-region truncation, with the
    // same radius the production trust-region loop landed on after its
    // cycle-0 shrink cascade (|δ|∞ = 20.0).
    let mut delta = delta_hat.clone();
    let pre_l2 = joint_trust_region_step_norm(&delta);
    let radius = 20.0_f64;
    let _post_l2 = truncate_joint_step_to_radius(&mut delta, radius);
    let scale = radius / pre_l2;

    // PRODUCTION CHECK 2: after L2 rescale, |δ|∞ ≤ radius.  Equality
    // holds when one component dominates the L2 norm completely; for
    // our fixture other components are small but non-zero so post_inf
    // is just under radius.  Production trace: |δ|∞ = 20.000 (the
    // 'shrink_marginal_accept' decision rounded to the cap).
    let post_inf = delta.iter().map(|v| v.abs()).fold(0.0_f64, f64::max);
    assert!(
        post_inf <= 20.0 && post_inf > 19.0,
        "after isotropic-L2 truncation, |δ|∞ should be in (19, 20] (production cap = 20.0); got {}",
        post_inf,
    );

    // PRODUCTION CHECK 3: the time block's |δ_time|∞ collapses to the
    // production block_beta_inf[time] = 2.3e-4 (within an order of
    // magnitude — the production value comes from the actual joint
    // Hessian, our fixture differs in exact magnitude but reproduces
    // the same collapse).
    let post_time_inf = block_inf(0, TIME_W, &delta);
    let post_marg_inf = block_inf(TIME_W, MARG_W, &delta);
    assert!(
        post_time_inf < 1.0e-3,
        "PRODUCTION-MATCH: time block |δ|∞ should collapse to O(1e-4) under \
         isotropic-L2 rescale (production observed 2.3e-4 in block_beta_inf[0]); \
         got {:.3e}",
        post_time_inf,
    );

    // ── EXACT PRODUCTION DIAGNOSTIC: compute linearized_rel ──
    //
    // gam's [PIRLS/JN/math] cycle log prints
    //     linearized_next_kkt_inf = ‖g + Hδ‖∞
    //     old_kkt_inf             = ‖g‖∞
    //     linearized_rel          = linearized_next_kkt_inf / (1 + old_kkt_inf)
    //
    // The "linearized-rate stall early-exit" fires when
    //     linearized_rel ≥ LINEARIZED_STALL_REL_THRESHOLD  (0.9)
    // holds for LINEARIZED_STALL_CYCLES (15) consecutive cycles AND
    // residual ≥ LINEARIZED_STALL_RESIDUAL_FACTOR × residual_tol (50×).
    //
    // Production observed linearized_rel ≈ 0.97 — exactly the signal we
    // expect here under the isotropic-L2 rescale of a step proposal with
    // a single huge near-null component.

    let h_delta = h.dot(&delta);
    let g_plus_h_delta = &g + &h_delta;
    let linearized_next_kkt_inf = g_plus_h_delta
        .iter()
        .map(|v| v.abs())
        .fold(0.0_f64, f64::max);
    let old_kkt_inf = g.iter().map(|v| v.abs()).fold(0.0_f64, f64::max);
    let linearized_rel = linearized_next_kkt_inf / (1.0 + old_kkt_inf);

    eprintln!(
        "[isotropic-TR-repro] |prop|₂ = {:.3e}  radius = {:.1}  rescale = {:.3e}",
        pre_l2, radius, scale,
    );
    eprintln!(
        "[isotropic-TR-repro] time  block |δ|∞: {:.3e} -> {:.3e}  ({:.4}% retained)",
        pre_time_inf,
        post_time_inf,
        100.0 * post_time_inf / pre_time_inf,
    );
    eprintln!(
        "[isotropic-TR-repro] marg  block |δ|∞: {:.3e} -> {:.3e}  ({:.4}% retained)",
        pre_marg_inf,
        post_marg_inf,
        100.0 * post_marg_inf / pre_marg_inf,
    );
    eprintln!(
        "[isotropic-TR-repro] production-style PIRLS/JN/math line:\n  \
         old_kkt_inf=‖g‖∞={:.3e}  linearized_next_kkt_inf=‖g+Hδ‖∞={:.3e}  \
         linearized_rel = {:.3e}  (LINEARIZED_STALL_REL_THRESHOLD = 0.9)",
        old_kkt_inf, linearized_next_kkt_inf, linearized_rel,
    );

    // ── RED ASSERTION ──
    //
    // After one Newton step (this is exactly the per-cycle stall test),
    // the linear solve PLUS trust-region truncation must reduce the
    // gradient meaningfully — otherwise the inner Newton loop runs in
    // place for the next 15+ cycles and the linearized-rate stall
    // detector rejects the seed.
    //
    // A healthy Newton step on a well-formed quadratic has
    // linearized_rel ≪ 0.9 even after truncation (the well-conditioned
    // blocks pull g down). The L2 rescale here violates this by giving
    // linearized_rel ≈ 1, exactly matching the production
    // "linearized-rate stall early-exit" diagnostic.
    let stall_threshold = 0.9_f64;
    assert!(
        linearized_rel < stall_threshold,
        "ISOTROPIC-TR LINEARIZED-RATE STALL: ‖g + Hδ‖∞ / (1 + ‖g‖∞) = {:.3e} \
         ≥ {:.1} after a single Newton step with the joint trust region. \
         This is exactly the diagnostic gam prints right before the \
         'linearized-rate stall early-exit' fires (after 15 consecutive \
         cycles like this one, the outer optimizer rejects the seed). \
         \
         The Newton step itself is correct (δ̂ = H⁻¹(-g) gives perfect \
         linearized_rel ≈ 0). The bug is in the joint trust-region \
         TRUNCATION: src/families/custom_family.rs:10835-10847 computes \
         the L2 norm of the WHOLE concatenated δ and applies a single \
         scalar rescale `δ *= radius / ‖δ‖₂`. When one block has a \
         near-null Hessian direction → that block's unconstrained step \
         has component ~|g|/σ_min(H_block) = O(1e5), the rescale factor \
         becomes ~radius / 1e5 = 1e-4, and the well-conditioned blocks' \
         Newton steps are crushed by the same 1e-4 factor. Hδ for those \
         blocks shrinks too, so g + Hδ ≈ g, linearized_rel ≈ 1, stall \
         detector fires. \
         \
         Fix: per-block trust radii (each block updated from its own \
         actual/predicted reduction), or a diagonal preconditioner \
         D = diag(H)^{{1/2}} so the trust region is ellipsoidal in the \
         scaled metric and respects each block's local curvature scale. \
         \
         Production observed: block_beta_inf=[2.3e-4, 15.3, 20.0], \
         linearized_rel ≈ 0.97 for 15+ cycles, seed rejected. \
         Reproduced here: post_time_inf={:.3e}, linearized_rel={:.3e}.",
        linearized_rel,
        stall_threshold,
        post_time_inf,
        linearized_rel,
    );
}
