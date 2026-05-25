//! RED REPRO: PIRLS joint-Newton trust-region L2 rescale starves
//! well-conditioned blocks when ONE block has a near-singular curvature
//! direction.
//!
//! Production failure (n=195,780 survival_marginal_slope fit, 5/5 outer
//! seeds rejected):
//!
//!   cycle 0 unconstrained Newton proposal:  |prop|∞ = 2.173e5
//!   cycle 0 after trust-region truncation:  |δ|∞    = 20.0
//!   block_widths   = [12, 11, 10]            # time, marginal, logslope
//!   block_beta_inf = [2.3e-4, 15.3, 20.0]    # time block barely moved
//!   block_grad_inf = [5.6e8,  1.5e3, 2.3e3]  # time block carries ALL the KKT residual
//!   linearized_rel = ‖g + Hδ‖∞ / (1 + ‖g‖∞) ≈ 0.97 for 15+ consecutive cycles
//!   exit reason    = "[PIRLS/joint-Newton convergence] cycle N | linearized-rate
//!                     stall early-exit: ... >= 0.9 for 15 consecutive cycles ..."
//!
//! Root cause (under test): gam's joint-Newton trust region uses a single
//! scalar L2 norm over the concatenated δ vector and applies a uniform
//! rescale `δ *= radius / ‖δ‖₂` whenever the proposal exceeds the radius.
//! See `src/families/custom_family.rs:10835-10847`:
//!
//!   fn joint_trust_region_step_norm(delta: &Array1<f64>) -> f64 {
//!       delta.iter().map(|v| v * v).sum::<f64>().sqrt()
//!   }
//!   fn truncate_joint_step_to_radius(delta: &mut Array1<f64>, radius: f64) -> f64 {
//!       let norm = joint_trust_region_step_norm(delta);
//!       if norm.is_finite() && norm > radius && radius > 0.0 {
//!           delta.mapv_inplace(|v| v * (radius / norm));   // uniform rescale
//!           radius
//!       } else {
//!           norm
//!       }
//!   }
//!
//! When the joint Newton step has one block carrying an O(1e5) component
//! (because that block's Hessian has a near-null direction along the
//! response), the rescale factor becomes `radius / 1e5 ≈ 1e-4`. EVERY
//! block — including blocks whose unconstrained Newton step was a
//! perfectly well-conditioned O(1) — gets multiplied by 1e-4. The
//! well-conditioned blocks lose their meaningful Newton step, their
//! gradient stays essentially unchanged for the next cycle, and the
//! joint-Newton loop's linearized residual `‖g + Hδ‖∞` is stuck near
//! `‖g‖∞` forever → `linearized_rel ≈ 1` → 15-cycle stall → seed reject.
//!
//! This test does not invoke the full fit (which on biobank-scale data
//! takes ~2 min per outer eval and is platform-dependent on CUDA);
//! instead it reproduces the EXACT linear-algebra step the inner
//! solver performs at every TR attempt — `truncate_joint_step_to_radius`
//! reduced to its arithmetic essence — and asserts that each block
//! retains a non-trivial share of its unconstrained Newton step.
//!
//! Currently expected to FAIL. The fix is per-block trust radii (each
//! block's radius updated independently from its own actual/predicted
//! reduction), or a diagonal preconditioner that makes the box
//! ellipsoidal so the ill-conditioned direction is compressed without
//! crushing the orthogonal directions.

/// Mirror of gam's `joint_trust_region_step_norm`:
/// `src/families/custom_family.rs:10835-10837`.
fn joint_trust_region_step_norm(delta: &[f64]) -> f64 {
    delta.iter().map(|v| v * v).sum::<f64>().sqrt()
}

/// Mirror of gam's `truncate_joint_step_to_radius`:
/// `src/families/custom_family.rs:10841-10847`.
fn truncate_joint_step_to_radius(delta: &mut [f64], radius: f64) -> f64 {
    let norm = joint_trust_region_step_norm(delta);
    if norm.is_finite() && norm > radius && radius > 0.0 {
        let scale = radius / norm;
        for v in delta.iter_mut() {
            *v *= scale;
        }
        radius
    } else {
        norm
    }
}

#[test]
fn isotropic_l2_trust_region_starves_well_conditioned_blocks() {
    // Production block layout: time(12), marginal(11), logslope(10) → p = 33.
    const TIME_W: usize = 12;
    const MARG_W: usize = 11;
    const LOGSL_W: usize = 10;
    const P: usize = TIME_W + MARG_W + LOGSL_W;

    // Unconstrained Newton proposal δ̂ = H⁻¹(-g) for the actual joint
    // Newton system the production survival_marginal_slope fit produced
    // at cycle 0:
    //
    //   • time block: well-conditioned, modest O(1) step;
    //   • marginal block: well-conditioned, O(1) step;
    //   • logslope block: ONE near-null direction in H whose Newton step
    //     is 2.173e5 (the production |prop|∞ value), other components O(1).
    //
    // What the [PIRLS/JN/math] line in the production trace reports.
    let mut proposal = [0.0_f64; P];
    for i in 0..TIME_W {
        proposal[i] = if i == 0 { 1.5 } else { 0.3 + 0.07 * (i as f64) };
    }
    for j in 0..MARG_W {
        proposal[TIME_W + j] = if j == 0 { 0.9 } else { 0.2 - 0.01 * (j as f64) };
    }
    for k in 0..LOGSL_W {
        proposal[TIME_W + MARG_W + k] = if k == 0 { 2.173e5 } else { 0.4 };
    }

    // Sanity: matches the production cycle-0 |prop|∞.
    let prop_inf = proposal.iter().map(|v| v.abs()).fold(0.0_f64, f64::max);
    assert!(
        (prop_inf - 2.173e5).abs() < 1.0,
        "fixture proposal |.|∞ = {} should match production 2.173e5",
        prop_inf
    );

    // Snapshot the unconstrained per-block inf-norms before truncation.
    let block_inf = |start: usize, len: usize, v: &[f64]| -> f64 {
        v[start..start + len]
            .iter()
            .map(|x| x.abs())
            .fold(0.0_f64, f64::max)
    };
    let pre_time_inf = block_inf(0, TIME_W, &proposal);
    let pre_marg_inf = block_inf(TIME_W, MARG_W, &proposal);
    let pre_logsl_inf = block_inf(TIME_W + MARG_W, LOGSL_W, &proposal);

    // Apply gam's actual joint-Newton trust-region truncation, with the
    // exact radius the production trust-region loop arrived at after
    // its cycle-0 shrink cascade (production: r = 25 -> 6.31 immediately,
    // and the accepted-step inf-norm was 20.0; the L2 rescale to a
    // radius of 20 reproduces the production |δ|∞ = 20.0 exactly).
    let radius = 20.0_f64;
    let pre_l2 = joint_trust_region_step_norm(&proposal);
    let _post_l2 = truncate_joint_step_to_radius(&mut proposal, radius);
    let scale = radius / pre_l2;

    let post_time_inf = block_inf(0, TIME_W, &proposal);
    let post_marg_inf = block_inf(TIME_W, MARG_W, &proposal);
    let post_logsl_inf = block_inf(TIME_W + MARG_W, LOGSL_W, &proposal);

    eprintln!(
        "[isotropic-TR-repro] |prop|₂ = {:.3e}, radius = {}, uniform-rescale factor = {:.3e}",
        pre_l2, radius, scale,
    );
    eprintln!(
        "[isotropic-TR-repro] time  block |δ|∞: {:.3e} -> {:.3e}  ({:.3}% retained)",
        pre_time_inf,
        post_time_inf,
        100.0 * post_time_inf / pre_time_inf,
    );
    eprintln!(
        "[isotropic-TR-repro] marg  block |δ|∞: {:.3e} -> {:.3e}  ({:.3}% retained)",
        pre_marg_inf,
        post_marg_inf,
        100.0 * post_marg_inf / pre_marg_inf,
    );
    eprintln!(
        "[isotropic-TR-repro] logsl block |δ|∞: {:.3e} -> {:.3e}  ({:.3}% retained)",
        pre_logsl_inf,
        post_logsl_inf,
        100.0 * post_logsl_inf / pre_logsl_inf,
    );

    // PRODUCTION-MATCH CHECK 1: after the rescale, the |δ|∞ of the WHOLE
    // step is exactly the production cycle-0 value (20.0). The rescale
    // pinned one component of the logslope block to ±radius and crushed
    // every other coordinate to ~radius / |prop|∞ * its_original.
    let post_inf = proposal.iter().map(|v| v.abs()).fold(0.0_f64, f64::max);
    assert!(
        (post_inf - 20.0).abs() < 1.0e-9,
        "after truncation |δ|∞ should be radius=20 (production cycle 0); got {}",
        post_inf,
    );

    // PRODUCTION-MATCH CHECK 2: the time block's |δ|∞ collapses to the
    // production block_beta_inf[0] = 2.3e-4. This is the smoking-gun
    // signature of the isotropic L2 rescale: the well-conditioned block
    // gets a step of magnitude ~ (radius / |prop|∞) × its_unconstrained
    // step, which for the production fixture is exactly 2-3 × 1e-4.
    assert!(
        post_time_inf < 1.0e-3,
        "PRODUCTION-MATCH: time block |δ|∞ should collapse to O(1e-4) under \
         isotropic-L2 rescale (the production trace reported block_beta_inf[time] = 2.3e-4); \
         got {:.3e}",
        post_time_inf,
    );

    // RED ASSERTION: each block must retain at least 1% of its
    // unconstrained Newton step direction after a TR truncation —
    // otherwise no well-conditioned block can converge while ONE
    // ill-conditioned block holds the entire step budget hostage.
    //
    // This is the core property the joint-Newton trust region must
    // preserve to converge on block-anisotropic Hessians. The L2
    // uniform-rescale violates it by ~four orders of magnitude
    // (production observed 100% × (2.3e-4 / 1.5) = 0.015 % retention
    // in the time block; this fixture reproduces the same ballpark).
    let min_retention = 0.01_f64;
    assert!(
        post_time_inf >= min_retention * pre_time_inf,
        "ISOTROPIC-TR STARVATION: well-conditioned time block lost its \
         Newton step under the joint L2 rescale. unconstrained |δ_time|∞ = {:.3e}, \
         after truncation = {:.3e} ({:.3}% retained, need ≥ {:.0}%). \
         This is the production survival_marginal_slope failure mode: \
         block_beta_inf=[2.3e-4, 15.3, 20.0] — one block's near-null \
         direction with |prop|∞ = 2.173e5 forces an L2 rescale by \
         {:.3e}, which uniformly crushes every block. Subsequent inner \
         Newton cycles cannot reduce the time-block gradient (KKT \
         residual stays at the cycle-0 value of ≈ 5.6e8) because the \
         step magnitude assigned to that block is below step_tol. \
         Linearized residual ‖g + Hδ‖∞ / (1 + ‖g‖∞) stays ≈ 1, the \
         15-consecutive-cycle linearized-rate stall detector fires, the \
         seed is rejected, and the outer optimizer exhausts all 5 seeds. \
         \
         Root cause: src/families/custom_family.rs:10835-10847 — \
         `joint_trust_region_step_norm` returns an L2 norm over the \
         concatenated δ, and `truncate_joint_step_to_radius` applies a \
         single scalar rescale `δ *= radius / ‖δ‖₂`. The fix is \
         per-block trust radii (each block's radius updated independently \
         from its own actual/predicted reduction), or a diagonal \
         preconditioner D = diag(H)^{1/2} so the trust region is \
         ellipsoidal in the scaled metric.",
        pre_time_inf,
        post_time_inf,
        100.0 * post_time_inf / pre_time_inf,
        100.0 * min_retention,
        scale,
    );
    assert!(
        post_marg_inf >= min_retention * pre_marg_inf,
        "ISOTROPIC-TR STARVATION: marginal block lost its Newton step \
         (post {:.3e} vs unconstrained {:.3e}, scale {:.3e}).",
        post_marg_inf,
        pre_marg_inf,
        scale,
    );
}
