#![cfg(test)]
//! #2515 — the dense and streaming evidence routes must rank one exact
//! observed information `A = B_raw + ΔC`.
//!
//! These gates run at the #2330 residual-excited deflated anchor from
//! `tests_deflated_from_probes_2712`: the dense and arrow routes must
//! materialize the same raw `A`, and the forced streaming outer route must
//! admit a deflating state and return the dense route's value and gradient
//! wherever the dense route ranks one.

use super::*;
use approx::assert_abs_diff_eq;

/// #2515 — both routes materialize the ONE raw objective Hessian
/// `A = B_raw + ΔC` before applying any evidence classification.
///
/// The dense route used to add ΔC to the already-conditioned cache operator
/// `B_tilde`, while the arrow route added it to `B_raw` and conditioned the
/// result. This gate requires operator equality itself; it no longer blesses the
/// old mismatch by merely attributing it to `B_tilde - B_raw`.
#[test]
fn dense_and_arrow_materialize_the_same_raw_exact_a_2515() {
    let (mut term, rho, target, b_cache) =
        super::tests_deflated_from_probes_2712::residual_excited_deflated_anchor(
            "#2515 the B-conditioning increment is the whole residual",
        );
    let a_dense = term
        .materialize_exact_hessian_dense(&rho, target.view(), &b_cache)
        .expect("#2515: the deflated anchor's exact Hessian materializes");
    let sys = term
        .assemble_arrow_schur(target.view(), &rho, None)
        .expect("#2515: the deflated anchor assembles");
    let a_sys = term
        .exact_a_evidence_system(target.view(), &rho, &sys, 1.0)
        .expect("#2515: the deflated anchor builds its exact-A evidence system");

    // Reconstruct the entire arrow operator, not only its diagonal row blocks.
    // Equality on H_tt alone cannot certify one A: a stale H_tβ operator changes
    // both the reduced Schur and every selected-inverse/logdet channel while the
    // row-only comparison remains exactly green.
    let total_t = b_cache.delta_t_len();
    let dim = total_t + a_sys.k;
    assert_eq!(a_dense.dim(), (dim, dim));
    let mut a_arrow = Array2::<f64>::zeros((dim, dim));
    for (row_index, row) in a_sys.rows.iter().enumerate() {
        let q = a_sys.row_dims[row_index];
        let base = a_sys.row_offsets[row_index];
        a_arrow
            .slice_mut(s![base..base + q, base..base + q])
            .assign(&row.htt);

        let use_dense = a_sys.htbeta_dense_supplement || a_sys.htbeta_matvec.is_none();
        let mut cross = if use_dense && row.htbeta.dim() == (q, a_sys.k) {
            row.htbeta.clone()
        } else {
            Array2::<f64>::zeros((q, a_sys.k))
        };
        if let Some(operator) = a_sys.htbeta_matvec.as_ref() {
            let mut basis = Array1::<f64>::zeros(a_sys.k);
            let mut column = Array1::<f64>::zeros(q);
            for beta in 0..a_sys.k {
                basis[beta] = 1.0;
                column.fill(0.0);
                operator(row_index, basis.view(), &mut column);
                basis[beta] = 0.0;
                cross.column_mut(beta).scaled_add(1.0, &column);
            }
        }
        a_arrow
            .slice_mut(s![base..base + q, total_t..])
            .assign(&cross);
        a_arrow
            .slice_mut(s![total_t.., base..base + q])
            .assign(&cross.t());
    }
    // The border block is the exact-A system's own effective shared operator:
    // the majorizer's penalty operator composed with leg (5) of ΔC, the β-tier
    // decoder priors' exact-minus-majorizer remainder (#2828). The legacy `hbb`
    // slab may be empty, and copying the dense block here instead would bless an
    // arrow route that prices B_ββ where the dense route prices A_ββ.
    a_arrow
        .slice_mut(s![total_t.., total_t..])
        .assign(&a_sys.effective_penalty_op().to_dense());

    let mut worst_operator_gap = 0.0_f64;
    let mut worst_block_scale = 0.0_f64;
    let mut worst_location = (0usize, 0usize);
    for row in 0..dim {
        for column in 0..dim {
            let gap = (a_dense[[row, column]] - a_arrow[[row, column]]).abs();
            if gap > worst_operator_gap {
                worst_operator_gap = gap;
                worst_location = (row, column);
            }
            worst_block_scale = worst_block_scale.max(a_dense[[row, column]].abs());
        }
    }
    let mut deflated_directions = 0usize;
    for row in 0..b_cache.n_rows() {
        let q = b_cache.row_dims[row];
        let base = b_cache.row_offsets[row];
        let a_dense_block = a_dense.slice(s![base..base + q, base..base + q]).to_owned();
        let a_arrow_block = &a_sys.rows[row].htt;
        for a in 0..q {
            for b in 0..q {
                worst_operator_gap = worst_operator_gap
                    .max((a_dense_block[[a, b]] - a_arrow_block[[a, b]]).abs());
                worst_block_scale = worst_block_scale.max(a_dense_block[[a, b]].abs());
            }
        }
        // On each direction the majorizer factorization declared null, report what
        // each route's exact-`A` actually prices there.
        for direction in b_cache
            .deflated_row_directions
            .get(row)
            .map(Vec::as_slice)
            .unwrap_or(&[])
        {
            if direction.len() != q {
                continue;
            }
            deflated_directions += 1;
            let dense_curvature = direction.dot(&a_dense_block.dot(direction));
            let arrow_curvature = direction.dot(&a_arrow_block.dot(direction));
            println!(
                "[#2515 ORDERING] row {row}: dense v'Av={dense_curvature:.10e} \
                 arrow v'Av={arrow_curvature:.10e} gap={:.3e}",
                (dense_curvature - arrow_curvature).abs(),
            );
        }
    }
    println!(
        "[#2515 ORDERING] |A_dense − A_arrow|∞ = {worst_operator_gap:.6e} at \
         {worst_location:?} over full-operator scale {worst_block_scale:.6e}; \
         deflated directions inspected = {deflated_directions}"
    );

    assert!(
        deflated_directions > 0,
        "#2515: this anchor must carry at least one majorizer-deflated direction, or \
         the ordering claim is about nothing"
    );
    assert!(
        worst_block_scale.is_finite() && worst_block_scale > 1.0e-6,
        "#2515: the exact-A row blocks must be non-trivial for the identity to mean \
         anything; block scale {worst_block_scale:.6e}"
    );
    assert!(
        worst_operator_gap <= 1.0e-12 * worst_block_scale,
        "#2515: the dense and arrow routes materialized different raw exact-A row \
         blocks (|A_dense − A_arrow|∞={worst_operator_gap:.6e} over block scale \
         {worst_block_scale:.6e}). Evidence conditioning must happen only after \
         the one objective Hessian `B_raw + ΔC` exists."
    );
}

/// #2515 — THE LIFTED GATE, END TO END: a state whose evidence factorization
/// spectrally deflates now gets a streaming outer gradient instead of a typed
/// refusal, and that gradient is the dense one.
///
/// The two tests either side of this measure the ASSEMBLERS at a fixed state.
/// This one drives the production entry point — `evaluate_outer_criterion_route`
/// with `direct_logdet_admitted = false`, the branch the memory planner selects
/// at production `p` — so a regression that re-armed the refusal, or that
/// admitted it while silently returning a `B`-rooted gradient, is caught where a
/// fit would actually meet it.
///
/// Before the lift this returned
/// `"streaming outer derivative is not admitted: the … evidence factorization
/// spectrally deflates row R in N direction(s)"`, so on a deflating state the
/// streaming lane had no answer at all — and at production `p` the streaming lane
/// is the only lane there is. That is the residual route-dependence this issue
/// was left with once the operator halves were closed: not a wrong criterion on
/// one route, but a criterion on one route and nothing on the other.
#[test]
fn forced_streaming_admits_a_deflating_state_and_matches_dense_2515() {
    // Both routes' `[SAE-CRITERION]` term lines are `log::debug!`; they say which
    // term of the value the two routes disagree on.
    gam_runtime::test_support::install_diagnostic_logger();
    let (term, rho, target, b_cache) =
        super::tests_deflated_from_probes_2712::residual_excited_deflated_anchor(
            "#2515 the lifted deflation gate, end to end",
        );
    let anchor_deflated_rows = b_cache
        .deflated_row_directions
        .iter()
        .filter(|directions| !directions.is_empty())
        .count();
    assert!(
        anchor_deflated_rows > 0,
        "#2515: the anchor must deflate, or this exercises the ordinary lane"
    );

    let mut dense = SaeManifoldOuterObjective::new(
        term.clone(),
        target.clone(),
        None,
        rho.clone(),
        40,
        0.4,
        1.0e-6,
        1.0e-6,
    );
    let mut streaming =
        SaeManifoldOuterObjective::new(term, target, None, rho.clone(), 40, 0.4, 1.0e-6, 1.0e-6);
    let rho_flat = dense.baseline_rho.flat_coordinates();
    let route_rho = streaming
        .baseline_rho
        .from_flat(rho_flat.view())
        .expect("#2515: both objectives own the same typed rho layout");

    let dense_artifact = dense
        .evaluate_outer_criterion_route(&route_rho, true, false)
        .expect("#2515: the dense route is the authority this parity is against");
    let dense_gradient = dense
        .analytic_gradient_for_outer_evaluation(&route_rho, &dense_artifact)
        .expect("#2515: the dense route's analytic gradient");

    let streaming_artifact = streaming
        .evaluate_outer_criterion_route(&route_rho, false, false)
        .expect(
            "#2515: the forced streaming route must ADMIT a deflating state. A typed \
             `streaming outer derivative is not admitted: … spectrally deflates row …` \
             here means the lifted refusal has been re-armed",
        );
    let streaming_gradient = streaming
        .analytic_gradient_for_outer_evaluation(&route_rho, &streaming_artifact)
        .expect("#2515: the forced streaming route's analytic gradient");

    let mut worst = 0.0_f64;
    let mut scale = 0.0_f64;
    for (coordinate, (&streamed, &direct)) in streaming_gradient
        .iter()
        .zip(dense_gradient.iter())
        .enumerate()
    {
        assert!(
            streamed.is_finite() && direct.is_finite(),
            "#2515: gradient coordinate {coordinate} is non-finite \
             (streaming={streamed}, dense={direct})"
        );
        worst = worst.max((streamed - direct).abs());
        scale = scale.max(direct.abs());
    }
    println!(
        "[#2515 LIFTED] anchor deflated rows={anchor_deflated_rows} \
         cost dense={:.10e} streaming={:.10e} \
         gradient max|Δ|={worst:.6e} against ‖g‖∞={scale:.6e}",
        dense_artifact.cost, streaming_artifact.cost
    );

    assert_eq!(
        streaming_gradient.len(),
        dense_gradient.len(),
        "#2515: the two routes must own the same outer coordinate layout"
    );
    assert!(
        scale.is_finite() && scale > 1.0e-9,
        "#2515: route parity must exercise a nonzero analytic gradient; ‖g‖∞={scale:.6e}"
    );
    assert_abs_diff_eq!(
        streaming_artifact.cost,
        dense_artifact.cost,
        epsilon = 1.0e-7
    );
    assert!(
        worst <= 1.0e-6 * scale.max(1.0),
        "#2515: the forced streaming gradient departs from the dense one \
         (max|Δ|={worst:.6e} against ‖g‖∞={scale:.6e}). The streaming lane is steering \
         a fit with the derivative of an operator the dense criterion does not rank — \
         the defect the spectral-deflation refusal used to hide behind."
    );
}

/// #2515 — THE STREAMING OUTER GRADIENT EXISTS ON EVERY STATE THE DENSE ROUTE
/// DIFFERENTIATES, and this is the gate that says so at the production entry
/// point.
///
/// Before the gate freeze was scoped to the whole criterion evaluation, the
/// streaming lane returned a VALUE and then refused its GRADIENT across a whole
/// band of smoothing strengths:
///
/// ```text
/// smooth=-1.10  dense     cost=1.8195496423e1  ‖g‖∞=1.580471e1
///               streaming cost=1.8195496415e1  GRADIENT REFUSED: … refuses a stale
///                         matrix-free system/cache pair (row fingerprints DIFFER,
///                         manifold fingerprints EQUAL)
/// smooth=-1.20  same
/// smooth=-1.40  same
/// ```
///
/// The cache was factored inside `converge_inner_for_undamped_logdet`'s frozen
/// window and the system was assembled after that window closed, so
/// `assemble_arrow_schur_scaled` re-refreshed all three collapse-prevention gates
/// from the moved state and the pair described two operators. Equal manifold
/// fingerprints with unequal row ones is exactly the signature
/// `evidence_assembly_row_fingerprint_sources_2515` attributes to the gate state.
///
/// This walks the same band and requires BOTH routes to produce a `(value,
/// gradient)` pair and to agree on it. A refusal on either side fails, and so does
/// a disagreement — the two are the same defect seen from opposite ends, and a
/// gate that accepted a refusal as "well, it declined safely" would have passed
/// throughout the era this fixes.
#[test]
fn forced_streaming_has_a_gradient_wherever_the_dense_route_does_2515() {
    // Both routes' `[SAE-CRITERION]` term lines are `log::debug!`.
    gam_runtime::test_support::install_diagnostic_logger();
    let (term, anchor_rho, target, _cache) =
        super::tests_deflated_from_probes_2712::residual_excited_deflated_anchor(
            "#2515 the streaming gradient exists wherever the dense one does",
        );
    let mut compared = 0usize;
    for smooth in [-0.9_f64, -1.05, -1.1, -1.4] {
        let mut rho = anchor_rho.clone();
        for value in rho.log_lambda_smooth.iter_mut() {
            *value = smooth;
        }
        let mut dense = SaeManifoldOuterObjective::new(
            term.clone(),
            target.clone(),
            None,
            rho.clone(),
            40,
            0.4,
            1.0e-6,
            1.0e-6,
        );
        let mut streaming = SaeManifoldOuterObjective::new(
            term.clone(),
            target.clone(),
            None,
            rho.clone(),
            40,
            0.4,
            1.0e-6,
            1.0e-6,
        );
        let rho_flat = dense.baseline_rho.flat_coordinates();
        let route_rho = streaming
            .baseline_rho
            .from_flat(rho_flat.view())
            .expect("#2515: both objectives own the same typed rho layout");

        // The dense route decides whether this state is rankable at all. Where it
        // declines (an exact-A saddle, say), there is nothing for the streaming
        // lane to match and the rung is not a counter-example to anything.
        let Ok(dense_artifact) = dense.evaluate_outer_criterion_route(&route_rho, true, false)
        else {
            println!("[#2515 EXISTS] smooth={smooth:.2}: dense route declines this state");
            continue;
        };
        let Ok(dense_gradient) =
            dense.analytic_gradient_for_outer_evaluation(&route_rho, &dense_artifact)
        else {
            println!("[#2515 EXISTS] smooth={smooth:.2}: dense gradient declines this state");
            continue;
        };

        let streaming_artifact = streaming
            .evaluate_outer_criterion_route(&route_rho, false, false)
            .unwrap_or_else(|err| {
                panic!(
                    "#2515: the dense route ranks smooth={smooth:.2} and the forced streaming \
                     route must too. Got: {err}"
                )
            });
        let streaming_gradient = streaming
            .analytic_gradient_for_outer_evaluation(&route_rho, &streaming_artifact)
            .unwrap_or_else(|err| {
                panic!(
                    "#2515: the dense route DIFFERENTIATES smooth={smooth:.2} and the forced \
                     streaming route refused. A `stale matrix-free system/cache pair` here is \
                     the gate freeze stopping one call short of the evidence assembly that \
                     prices the criterion; anything else is a new defect at the same seam. \
                     Got: {err}"
                )
            });

        let mut worst = 0.0_f64;
        let mut scale = 0.0_f64;
        for (streamed, direct) in streaming_gradient.iter().zip(dense_gradient.iter()) {
            worst = worst.max((streamed - direct).abs());
            scale = scale.max(direct.abs());
        }
        println!(
            "[#2515 EXISTS] smooth={smooth:.2}: cost dense={:.10e} streaming={:.10e} \
             gradient max|Δ|={worst:.6e} against ‖g‖∞={scale:.6e}",
            dense_artifact.cost, streaming_artifact.cost
        );
        assert_abs_diff_eq!(
            streaming_artifact.cost,
            dense_artifact.cost,
            epsilon = 1.0e-7
        );
        assert!(
            worst <= 1.0e-6 * scale.max(1.0),
            "#2515: the two routes both produced a gradient at smooth={smooth:.2} and they \
             disagree (max|Δ|={worst:.6e} against ‖g‖∞={scale:.6e})"
        );
        compared += 1;
    }
    assert!(
        compared >= 3,
        "#2515: at least three rungs must be rankable by the dense route, or this gate is \
         about a band the dense route also declines; got {compared}"
    );
}
