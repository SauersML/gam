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
//! Historical root cause (under test): the old joint-Newton trust region used
//! a single scalar L2 norm over the concatenated δ vector. When ONE block has
//! a near-null direction in its joint Hessian, the unconstrained Newton step is
//! huge along that direction. The L2 rescale uniformly multiplies the entire δ
//! by `radius / ‖δ‖₂ ≈ 1e-4`, crushing every block's Newton step by the same
//! factor. The well-conditioned blocks lose their meaningful contribution;
//! their gradient is essentially unchanged at the next cycle.
//! `‖g + Hδ‖∞ ≈ ‖g‖∞` → `linearized_rel ≈ 1` → 15-cycle stall detector fires
//! → seed rejected.

use ndarray::{Array1, Array2};

// ────────────────────────────────────────────────────────────────────
// Copy of the old raw-L2 primitive, plus the fixed block-local metric
// primitive. The production helper is tested directly in
// `src/families/custom_family.rs`; this integration test keeps the historical
// failure mechanism visible without re-exporting private internals.
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

    // Apply the historical raw-L2 joint-Newton trust-region truncation, with
    // the same radius the production trust-region loop used in the failing
    // trace after its cycle-0 shrink cascade (|δ|∞ = 20.0).
    let mut delta = delta_hat.clone();
    let pre_l2 = joint_trust_region_step_norm(&delta);
    let radius = 20.0_f64;
    let post_l2 = truncate_joint_step_to_radius(&mut delta, radius);
    assert!(post_l2 <= radius + 1.0e-12);
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
// Mechanism B: rejected joint steps must not shrink interior blocks.
//
// Setup: two blocks. Block A's δ would reduce the objective. Block B's δ
// would increase it. The joint actual-reduction sums both and is negative.
// A naive per-block radius update using the joint rho shrinks both radii,
// even though Block A was an interior step and cannot be the trust-region
// bottleneck.
//
// The fixed behavior: only blocks on their own local trust boundary are
// shrunk on a rejected joint step. Interior blocks keep their radius.
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

fn block_hit_boundary(step_norm: f64, radius: f64) -> bool {
    step_norm.is_finite() && radius > 0.0 && step_norm >= 0.99 * radius
}

fn shrink_active_block_radii(radii: &mut [f64], step_norms: &[f64], factor: f64) {
    let any_boundary_block = radii
        .iter()
        .zip(step_norms)
        .any(|(radius, step_norm)| block_hit_boundary(*step_norm, *radius));
    for (radius, step_norm) in radii.iter_mut().zip(step_norms) {
        if !any_boundary_block || block_hit_boundary(*step_norm, *radius) {
            *radius = (*radius * factor).clamp(1.0e-12, 1.0e6);
        }
    }
}

#[test]
fn rejected_step_shrinks_only_boundary_block_radii() {
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
    let step_b = 5.0; // large bad step on block B's trust boundary

    let old_r_a_new = update_radius_joint_rho(r_a_old, step_a, actual_joint, predicted_joint);
    let old_r_b_new = update_radius_joint_rho(r_b_old, step_b, actual_joint, predicted_joint);
    assert!(old_r_a_new < r_a_old && old_r_b_new < r_b_old);

    let mut fixed_radii = vec![r_a_old, r_b_old];
    shrink_active_block_radii(&mut fixed_radii, &[step_a, step_b], 0.25);
    assert!(
        (fixed_radii[0] - r_a_old).abs() < 1.0e-12,
        "interior block A must keep its trust radius on a rejected joint step; \
         rho_a={rho_a:.2}, rho_b={rho_b:.2}, rho_joint={rho_joint:.2}, \
         old scalar update would have moved A to {old_r_a_new:.3e}",
    );
    assert!(
        fixed_radii[1] < r_b_old,
        "boundary block B caused the rejected joint step and must shrink; \
         old scalar update would have moved B to {old_r_b_new:.3e}",
    );
}

// ────────────────────────────────────────────────────────────────────
// Mechanism C: feasibility limiting must preserve the joint Newton direction.
//
// Setup: one block's feasibility guard returns α < 1. Scaling only that
// block mutates δ̂ into a different direction. With coupled blocks, that
// direction can be an ascent direction for the quadratic model even though
// the original Newton direction is descent. The principled feasibility
// operation is therefore a scalar α applied to the whole joint step; the
// block-local trust-region metric above is what prevents unrelated blocks
// from being starved by near-null curvature.
// ────────────────────────────────────────────────────────────────────

#[test]
fn feasibility_alpha_preserves_the_joint_newton_direction() {
    // SPD Hessian with strong cross-block coupling. The exact Newton
    // direction is intentionally mixed-sign; this makes per-block scaling
    // mathematically different from a line search along δ̂.
    let h = ndarray::array![[1.0_f64, 0.99_f64], [0.99_f64, 1.0_f64]];
    let delta_hat = ndarray::array![50.0_f64, -1.0_f64];
    let g = -h.dot(&delta_hat);

    // Sanity: full Newton kills g exactly.
    let exact_residual = &g + &h.dot(&delta_hat);
    assert!(
        exact_residual.iter().all(|v| *v > -1.0e-12 && *v < 1.0e-12),
        "unconstrained Newton must be exact on this 2-block linear system"
    );

    let alpha = 0.01_f64;
    let global_alpha_delta = alpha * &delta_hat;
    let mut block_local_delta = delta_hat.clone();
    block_local_delta[0] *= alpha;

    // Linearized objective change for q(δ) = g'δ + 0.5 δ'Hδ.
    let predicted_change =
        |step: &Array1<f64>| -> f64 { g.dot(step) + 0.5 * step.dot(&h.dot(step)) };
    let global_change = predicted_change(&global_alpha_delta);
    let block_local_change = predicted_change(&block_local_delta);

    assert!(
        global_change < 0.0,
        "global α preserves the Newton descent direction; got model change {global_change:.3e}",
    );
    assert!(
        block_local_change > 0.0,
        "block-local feasibility scaling can turn a coupled Newton direction \
         into ascent; got model change {block_local_change:.3e}",
    );
}

// ────────────────────────────────────────────────────────────────────
// Mechanism D: penalised Newton on a *genuinely ill-posed* problem.
//
// Setup: H_pen = -∇²ℓ + S has a null direction u (eigenvalue 0). The
// gradient g satisfies uᵀg ≠ 0.
//
// First-principles math: along the line β + t·u, the penalised
// objective is
//
//     f(β + t·u) = f(β) + t·(uᵀ∇f)        (curvature term uᵀH_pen u = 0)
//
// linear in t. There is NO minimum — the objective slopes to ±∞.
// No β* with ∇f(β*) = 0 exists; the penalised likelihood is
// genuinely unbounded below along u.
//
// Consequence for Newton: the truncated-pseudoinverse step
// δ = -H_pen⁺ g lives in range(H_pen) by construction, so Hδ ∈
// range(H_pen) and the null component of g + Hδ is exactly uᵀg ≠ 0.
// The unprojected residual cannot shrink; linearised_rel is bounded
// below by |uᵀg| / (1 + ‖g‖).
//
// The only mathematically valid responses are:
//
//   (i)  Detect ill-posedness at construction time and fail with a
//        clear diagnostic that names the null direction. The user
//        must add a constraint, drop a basis column, or supply a
//        prior.
//   (ii) Add an explicit, documented prior on u (i.e. a small ridge
//        in the null direction). This changes the model, not just
//        the solver — so it must be opt-in at the smooth-construction
//        level, not silently added inside the inner solver.
//   (iii) Reparameterise to remove u from the free parameters.
//
// Range-projecting the convergence test (`‖U_rangeᵀ g‖ < tol`) would
// silently certify a non-stationary β and break the IFT/envelope
// identity dV/dρ = ∂f/∂ρ for the original objective, because the
// outer derivation assumes g(β*(ρ)) = 0 in the full space.
// ────────────────────────────────────────────────────────────────────

#[test]
fn null_subspace_of_h_pen_with_nonzero_gradient_is_ill_posed() {
    // H_pen with rank 1 in a 2-D system. Eigenvalues: 1, 0. The null
    // direction is e_2.
    let h = ndarray::array![[1.0, 0.0], [0.0, 0.0]];
    let g = ndarray::array![-1.0, -1.0];

    // The truncated-pseudoinverse Newton step. Only the rank-1
    // component contributes; δ lives in range(H) = span(e_1).
    let delta = ndarray::array![1.0_f64, 0.0];
    let residual = &g + &h.dot(&delta);
    let g_inf = g.iter().map(|v| v.abs()).fold(0.0_f64, f64::max);
    let residual_inf = residual.iter().map(|v| v.abs()).fold(0.0_f64, f64::max);
    let linearized_rel = residual_inf / (1.0 + g_inf);

    eprintln!(
        "[mechanism-D] H_pen rank-deficient, uᵀg = -1: δ̂ = ({:.3}, {:.3}), \
         residual = ({:.3}, {:.3}), linearised_rel = {:.3}",
        delta[0], delta[1], residual[0], residual[1], linearized_rel,
    );

    // Math fact: every Newton-type step lives in range(H_pen), so the
    // null component of the residual equals the null component of g
    // forever. There is no β at which ∇f = 0.
    assert!(
        linearized_rel >= 0.5,
        "for H_pen with null direction u and uᵀg ≠ 0, ‖g + Hδ‖∞ / (1 + ‖g‖∞) \
         is bounded below by |uᵀg| / (1 + ‖g‖∞). No Newton-type step can \
         reduce it. Got linearised_rel = {linearized_rel:.3}.",
    );

    // The unambiguous detector of ill-posedness: for any u ∈ null(H_pen),
    // check whether uᵀg / ‖u‖ exceeds the gradient tolerance. If so the
    // objective is non-stationary along u and the fit is ill-posed.
    let u = ndarray::array![0.0_f64, 1.0]; // spans null(H)
    let u_dot_g = u.dot(&g);
    let u_norm = u.dot(&u).sqrt();
    let ill_posedness_score = u_dot_g.abs() / u_norm;
    assert!(
        ill_posedness_score > 1.0e-6,
        "null(H_pen) carries gradient mass uᵀg/‖u‖ = {ill_posedness_score:.3e}; \
         the penalised objective slopes linearly to ±∞ along u and admits \
         no Newton stationary point. The principled responses are (i) error \
         at construction with a diagnostic that names u, (ii) opt in to an \
         explicit prior on null(S) at the smooth definition site, or \
         (iii) reparameterise to drop u from the free parameters. \
         Silently certifying ‖U_rangeᵀ g‖ < tol would break the IFT envelope \
         identity for the original objective.",
    );
}
