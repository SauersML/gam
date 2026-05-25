//! Mechanism repro: PIRLS joint-Newton "linearized-rate stall early-exit"
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
// Copy of the old raw-L2 primitive, plus the fixed block-local metric
// primitive. These are private inside the crate; this integration test keeps
// the math mechanism visible without depending on test-only exports.
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

fn block_metric_norm(delta: &Array1<f64>, metric_diag: &Array1<f64>) -> f64 {
    delta
        .iter()
        .zip(metric_diag)
        .map(|(step, weight)| step * step * weight.abs().max(1.0e-10))
        .sum::<f64>()
        .sqrt()
}

fn block_metric_norms(
    delta: &Array1<f64>,
    ranges: &[(usize, usize)],
    metric_diag: &Array1<f64>,
) -> Vec<f64> {
    ranges
        .iter()
        .map(|(start, end)| {
            block_metric_norm(
                &delta.slice(ndarray::s![*start..*end]).to_owned(),
                &metric_diag.slice(ndarray::s![*start..*end]).to_owned(),
            )
        })
        .collect()
}

fn truncate_block_metric(
    delta: &mut Array1<f64>,
    ranges: &[(usize, usize)],
    metric_diag: &Array1<f64>,
    radii: &[f64],
) {
    for (block_idx, (start, end)) in ranges.iter().copied().enumerate() {
        let metric = metric_diag.slice(ndarray::s![start..end]).to_owned();
        let mut block = delta.slice_mut(ndarray::s![start..end]);
        let norm = block_metric_norm(&block.to_owned(), &metric);
        let radius = radii[block_idx];
        if norm.is_finite() && norm > radius && radius > 0.0 {
            block.mapv_inplace(|v| v * (radius / norm));
        }
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

    println!(
        "[isotropic-TR-repro] |prop|₂ = {:.3e}  radius = {:.1}  rescale = {:.3e}",
        pre_l2, radius, scale,
    );
    println!(
        "[isotropic-TR-repro] time  block |δ|∞: {:.3e} -> {:.3e}  ({:.4}% retained)",
        pre_time_inf,
        post_time_inf,
        100.0 * post_time_inf / pre_time_inf,
    );
    println!(
        "[isotropic-TR-repro] marg  block |δ|∞: {:.3e} -> {:.3e}  ({:.4}% retained)",
        pre_marg_inf,
        post_marg_inf,
        100.0 * post_marg_inf / pre_marg_inf,
    );
    println!(
        "[isotropic-TR-repro] production-style PIRLS/JN/math line:\n  \
         old_kkt_inf=‖g‖∞={:.3e}  linearized_next_kkt_inf=‖g+Hδ‖∞={:.3e}  \
         linearized_rel = {:.3e}  (LINEARIZED_STALL_REL_THRESHOLD = 0.9)",
        old_kkt_inf, linearized_next_kkt_inf, linearized_rel,
    );

    // ── MECHANISM ASSERTION ──
    //
    // After one Newton step (this is exactly the per-cycle stall test),
    // the linear solve PLUS trust-region truncation must reduce the
    // gradient meaningfully — otherwise the inner Newton loop runs in
    // place for the next 15+ cycles and the linearized-rate stall
    // detector rejects the seed.
    //
    // The old raw L2 rescale gives linearized_rel ≈ 1, exactly matching the
    // production "linearized-rate stall early-exit" diagnostic.
    let stall_threshold = 0.9_f64;
    assert!(
        linearized_rel >= stall_threshold,
        "raw concatenated L2 should reproduce the stall; got linearized_rel={linearized_rel:.3e}",
    );

    let ranges = vec![
        (0, TIME_W),
        (TIME_W, TIME_W + MARG_W),
        (TIME_W + MARG_W, h.nrows()),
    ];
    let metric_diag = h.diag().to_owned();
    let full_block_norms = block_metric_norms(&delta_hat, &ranges, &metric_diag);
    let mut fixed_delta = delta_hat.clone();
    let block_radii = vec![full_block_norms[0], full_block_norms[1], radius];
    truncate_block_metric(&mut fixed_delta, &ranges, &metric_diag, &block_radii);
    let fixed_linearized_rel = (&g + &h.dot(&fixed_delta))
        .iter()
        .map(|v| v.abs())
        .fold(0.0_f64, f64::max)
        / (1.0 + old_kkt_inf);
    assert!(
        fixed_linearized_rel < 1.0e-6,
        "block-local curvature metric should remove the starvation mechanism; got {fixed_linearized_rel:.3e}",
    );
}

// ────────────────────────────────────────────────────────────────────
// MISTAKE B: per-block trust radii share the joint scalar ρ.
//
// Setup: two blocks. Block A's δ would reduce the objective. Block B's δ
// would increase it. The JOINT actual-reduction sums both → can be small
// or negative even though Block A made real progress. gam's update loop
// then shrinks BOTH block radii by the same factor (since they both see
// the same actual_reduction and predicted_reduction inputs).
//
// The right behaviour: keep / grow Block A's radius, shrink Block B's.
// The current code can't because the per-block ρ is not separable
// without per-block likelihood/penalty contribution accounting.
// ────────────────────────────────────────────────────────────────────

/// Mirror of gam's `update_joint_trust_region_radius` signature: returns
/// the new radius given a step norm and the JOINT actual/predicted
/// reduction. (Inlined here so this test is independent of the test
/// support module that currently does not compile.)
fn update_radius_joint_rho(
    old_radius: f64,
    step_norm: f64,
    actual_reduction: f64,
    predicted_reduction: f64,
) -> f64 {
    let rho = if predicted_reduction > 0.0 {
        actual_reduction / predicted_reduction
    } else if actual_reduction >= 0.0 {
        1.0
    } else {
        f64::NEG_INFINITY
    };
    let accepted = rho.is_finite() && rho > 0.0;
    let mut radius = old_radius;
    if !accepted {
        radius *= 0.25;
        if step_norm.is_finite() && step_norm > 0.0 {
            radius = radius.min(0.5 * step_norm);
        }
    } else if rho < 0.25 {
        radius *= 0.25;
    } else if rho > 0.75 && step_norm >= 0.99 * old_radius {
        radius *= 2.0;
    }
    radius.clamp(1.0e-12, 1.0e6)
}

#[test]
fn mistake_b_per_block_radii_share_joint_rho_punishes_progressing_block() {
    // Block A: tiny well-chosen step gives actual_a = 100, predicted_a = 100.
    // Block B: large badly-chosen step gives actual_b = -200, predicted_b = +20.
    let actual_a = 100.0_f64;
    let predicted_a = 100.0_f64;
    let actual_b = -200.0_f64;
    let predicted_b = 20.0_f64;

    // Per-block ρ (the "ideal" view)
    let rho_a = actual_a / predicted_a; //  +1.00
    let rho_b = actual_b / predicted_b; //  −10.00 — bad
    assert!(rho_a >= 0.75 && rho_b <= 0.0);

    // JOINT view (what gam actually feeds the radius update):
    let actual_joint = actual_a + actual_b; //  −100
    let predicted_joint = predicted_a + predicted_b; //  +120
    let rho_joint = actual_joint / predicted_joint; //  −0.83

    let r_a_old = 1.0;
    let r_b_old = 5.0;
    let step_a = 0.1; // tiny progressive step in block A
    let step_b = 4.9; // large bad step in block B

    let r_a_new = update_radius_joint_rho(r_a_old, step_a, actual_joint, predicted_joint);
    let r_b_new = update_radius_joint_rho(r_b_old, step_b, actual_joint, predicted_joint);

    eprintln!(
        "[mistake-B] rho_per_block = (A={:.2}, B={:.2}) but rho_joint = {:.2}",
        rho_a, rho_b, rho_joint
    );
    eprintln!(
        "[mistake-B] r_A: {:.3} -> {:.3} (block A made real progress, radius MUST not shrink)",
        r_a_old, r_a_new
    );
    eprintln!(
        "[mistake-B] r_B: {:.3} -> {:.3} (block B caused damage, radius SHOULD shrink)",
        r_b_old, r_b_new
    );

    // PROGRESSING block must not be punished by another block's failure.
    // Currently gam's loop applies the joint ρ to every block → A shrinks.
    assert!(
        r_a_new >= r_a_old,
        "MISTAKE B: progressing block A had ρ_per_block = +1.00 (perfect Newton step) \
         but joint ρ = {:.2} dragged its trust radius from {:.3} down to {:.3}. \
         A well-conditioned block making real progress must NOT have its trust \
         radius collapsed because an unrelated block (B, ρ={:.2}) overshot. The \
         per-block radii in `joint_block_trust_radii` are nominally independent, \
         but they are updated using the JOINT scalar (actual_reduction, \
         predicted_reduction) at custom_family.rs:12184-12196, which couples \
         them. The fix is to compute per-block (actual, predicted) reductions \
         and update each radius with its OWN ρ.",
        rho_joint, r_a_old, r_a_new, rho_b,
    );
}

// ────────────────────────────────────────────────────────────────────
// MISTAKE C: survival_marginal_slope's per-block feasibility shrinks one
// block while leaving others at their unconstrained Newton magnitude.
//
// Setup: time block's `max_feasible_time_step` returns α = 1e-4 (because
// the unconstrained δ_time would cross a monotonicity guard). gam's
// `apply_block_local_feasibility_limits` multiplies ONLY the time block
// by α → time goes to ~0, logslope and marginal blocks keep their full
// unconstrained δ.
//
// The "joint" step the line search evaluates is then:
//   δ = (≈0 time, large marg, huge logslope)
// which is NOT a Newton step on the joint objective. The line search ρ
// is dominated by whatever marg/logslope do; the time-block descent
// direction is effectively lost. Production observes exactly this:
// time stays at 2.3e-4 forever.
// ────────────────────────────────────────────────────────────────────

#[test]
fn mistake_c_per_block_feasibility_alpha_destroys_joint_newton_direction() {
    // Build a tiny 2-block problem where the Newton step is δ̂ = (1, 1)
    // and the time-block feasibility forces α_time = 0.01 (so δ_time
    // becomes 0.01 instead of 1) while the other block stays at δ_other = 1.
    // Block 0: time, well-conditioned, H = 1, g = -1, so δ̂_time = 1
    // Block 1: other,                 H = 1, g = -1, so δ̂_other = 1
    let h = ndarray::array![[1.0, 0.0], [0.0, 1.0]];
    let g = ndarray::array![-1.0, -1.0];
    let delta_hat = ndarray::array![1.0, 1.0];

    // Sanity: full Newton kills g exactly.
    let exact_residual = &g + &h.dot(&delta_hat);
    assert!(
        exact_residual.iter().all(|v| v.abs() < 1e-12),
        "unconstrained Newton must be exact on this 2-block linear system"
    );

    // Now apply gam's per-block-only feasibility shrink to block 0.
    let alpha_time = 0.01_f64;
    let mut delta = delta_hat.clone();
    delta[0] *= alpha_time;

    let residual = &g + &h.dot(&delta);
    let linearized_rel = residual.iter().map(|v| v.abs()).fold(0.0_f64, f64::max)
        / (1.0 + g.iter().map(|v| v.abs()).fold(0.0_f64, f64::max));
    eprintln!(
        "[mistake-C] delta after per-block α_time={:.0e}: ({:.3e}, {:.3e})",
        alpha_time, delta[0], delta[1]
    );
    eprintln!("[mistake-C] linearized_rel = {:.3e}", linearized_rel);

    // The CORRECT step that respects time's feasibility while remaining a
    // joint Newton descent direction is α · δ̂ = (0.01, 0.01) — both
    // coordinates scaled by the same α, so the direction is preserved.
    // Per-block-only scaling gives (0.01, 1) — collinear with neither
    // δ̂ nor α·δ̂, and not a descent direction on the JOINT quadratic.
    let scaled_correctly = ndarray::array![alpha_time, alpha_time];
    let scaled_residual = &g + &h.dot(&scaled_correctly);
    let scaled_linearized_rel = scaled_residual.iter().map(|v| v.abs()).fold(0.0_f64, f64::max)
        / (1.0 + g.iter().map(|v| v.abs()).fold(0.0_f64, f64::max));
    eprintln!(
        "[mistake-C] correct α·δ̂ = ({:.3e}, {:.3e}) gives linearized_rel = {:.3e}",
        scaled_correctly[0], scaled_correctly[1], scaled_linearized_rel,
    );

    // Direction-preservation check: the two coordinates must scale by the
    // same factor relative to δ̂. The per-block-only scheme violates this.
    let ratio_0 = delta[0] / delta_hat[0];
    let ratio_1 = delta[1] / delta_hat[1];
    assert!(
        (ratio_0 - ratio_1).abs() < 1e-9,
        "MISTAKE C: per-block feasibility α scales only block 0 while leaving \
         block 1 at its unconstrained Newton step. After scaling: \
         δ/δ̂ = ({:.3e}, {:.3e}). The two coordinates moved by different \
         factors, so the JOINT Newton direction is destroyed. The right \
         answer is α·δ̂ = ({:.3e}, {:.3e}) which preserves direction and \
         gives linearised_rel = {:.3e}. The buggy step gives \
         linearised_rel = {:.3e} (≈ 1 − α along the unscaled axis). \
         Fix: when one block's α_max < 1, scale the JOINT step by α_max, \
         or add the violating hyperplane to the QP and re-solve.",
        ratio_0, ratio_1,
        scaled_correctly[0], scaled_correctly[1],
        scaled_linearized_rel,
        linearized_rel,
    );
}
