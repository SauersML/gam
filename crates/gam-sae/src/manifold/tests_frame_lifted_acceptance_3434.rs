#![cfg(test)]
//! #3434 — the evidence root the criterion prices is a minimum in the lifted chart
//! `ξ = (vec δC, vec W)` of its learned frames, not only in the fixed-frame `(t, C)`.
//!
//! The fixed-frame inner solve moves `(t, C)` with the frame `U` held, so a frame
//! turned off its root orientation stays turned: the polish reaches a root in
//! `(t, C)` at which `Tᵀ·∇_B L` is live along `W`, and the frame-integrated
//! information `A_ξ` the criterion prices there belongs to no minimum. The control
//! arm measures exactly that state; the criterion must then leave it for a root the
//! lifted chart certifies, with no resolved negative lifted curvature.
use super::tests_fitted_response_edf_2933::{ROOT_GRADIENT_CEILING, polish_to_root, root_norm};
use super::tests_fitted_response_frames_2933::framed_circle_in;
use super::*;
use ndarray::Array2;

/// The normal velocity the control turns the frame by, toward a data-free output
/// axis: an order-one-percent tilt of an order-one decoder, far outside the
/// decrement tolerance and far inside the chart's injectivity radius.
const FRAME_TURN: f64 = 0.05;

#[test]
fn the_priced_root_is_a_minimum_in_the_lifted_frame_chart_3434() {
    for (label, rank, off_span_constant) in [
        ("rank 2 with an absorbed off-span constant", 2usize, 0.3_f64),
        ("rank 3", 3, 0.0),
    ] {
        let (mut term, target, rho) = framed_circle_in(12, rank, off_span_constant);
        term.recompute_joint_shape_uncertainty(target.view(), &rho, None, 40, 0.4, 1.0e-6, 1.0e-6)
            .unwrap_or_else(|error| panic!("{label}: the framed fixture fits: {error}"));
        assert_eq!(
            term.atoms[0].decoder_frame.as_ref().map(|frame| frame.rank()),
            Some(rank),
            "{label}: the fit must activate a frame carrying the decoder's rank"
        );
        let gates = term.collapse_prevention_gates();
        term.declare_collapse_prevention_gates(&gates);

        // Control: turn the frame along the chart toward the last output axis, which
        // carries no data, and polish `(t, C)` to its fixed-frame root.
        let frame = term.atoms[0]
            .decoder_frame
            .as_ref()
            .expect("the framed state carries its frame")
            .frame()
            .to_owned();
        let p = frame.nrows();
        let m = term.atoms[0].basis_size();
        let mut axis = Array2::<f64>::zeros((p, 1));
        axis[[p - 1, 0]] = 1.0;
        let normal_axis = &axis - &frame.dot(&frame.t().dot(&axis));
        let mut normal_velocity = Array2::<f64>::zeros((p, rank));
        for row in 0..p {
            normal_velocity[[row, 0]] = FRAME_TURN * normal_axis[[row, 0]];
        }
        term.atoms[0]
            .advance_decoder_frame_along_tangent(
                Array2::<f64>::zeros((m, rank)).view(),
                normal_velocity.view(),
                1.0,
            )
            .unwrap_or_else(|error| panic!("{label}: the frame turns along the chart: {error}"));
        let (turned_cache, trajectory) = polish_to_root(&mut term, target.view(), &rho);
        let turned_norm = root_norm(&trajectory);
        assert!(
            turned_norm <= ROOT_GRADIENT_CEILING,
            "{label}: the turned frame's (t, C) polish stopped at ‖g‖={turned_norm:.3e}"
        );
        let turned = term
            .frame_lifted_certificate(target.view(), &rho, None, &turned_cache)
            .unwrap_or_else(|error| panic!("{label}: the turned root has a certificate: {error}"))
            .unwrap_or_else(|| panic!("{label}: the default host prices the dense lifted chart"));
        eprintln!(
            "[#3434 lifted root {label}] fixed-frame root of the turned frame: (t, C) ‖g‖ \
             {turned_norm:.3e}; lifted min μ {:.6e}, resolved negative {:?}, λ² {:.6e}, \
             ½λ²/scale {:.6e}, certified {}",
            turned.min_curvature,
            turned.resolved_negative,
            turned.lambda_sq,
            turned.relative,
            turned.certified,
        );
        assert!(
            !turned.certified,
            "{label}: control: a fixed-frame root of a turned frame must fail the lifted \
             certificate, or this fixture cannot tell the acceptance from its absence"
        );

        // The criterion's acceptance leaves that root for a certified lifted minimum.
        let (value, _loss, cache) = term
            .penalized_quasi_laplace_criterion_with_cache(
                target.view(),
                &rho,
                None,
                40,
                0.4,
                1.0e-6,
                1.0e-6,
            )
            .unwrap_or_else(|error| panic!("{label}: the criterion prices the framed fit: {error}"));
        assert_eq!(
            term.atoms[0].decoder_frame.as_ref().map(|frame| frame.rank()),
            Some(rank),
            "{label}: the priced state keeps its frame"
        );
        let priced = term
            .frame_lifted_certificate(target.view(), &rho, None, &cache)
            .unwrap_or_else(|error| panic!("{label}: the priced root has a certificate: {error}"))
            .unwrap_or_else(|| panic!("{label}: the default host prices the dense lifted chart"));
        let turn_left = term.atoms[0]
            .decoder_frame
            .as_ref()
            .expect("the priced state carries its frame")
            .frame()
            .row(p - 1)
            .iter()
            .map(|value| value * value)
            .sum::<f64>()
            .sqrt();
        eprintln!(
            "[#3434 lifted root {label}] priced root: value {value:.10e}; lifted min μ {:.6e}, \
             resolved negative {:?}, clamp explained {}, λ² {:.6e}, ½λ²/scale {:.6e}; frame \
             weight on the data-free axis {turn_left:.3e} (turned to {FRAME_TURN:.3e})",
            priced.min_curvature,
            priced.resolved_negative,
            priced.clamp_explained,
            priced.lambda_sq,
            priced.relative,
        );
        assert!(
            value.is_finite(),
            "{label}: the priced criterion is finite"
        );
        assert!(
            priced.resolved_negative.is_none(),
            "{label}: the priced root carries resolved negative lifted curvature {:?}",
            priced.resolved_negative
        );
        assert!(
            priced.certified,
            "{label}: the priced root is not certified in the lifted chart: ½λ²/scale {:.6e}",
            priced.relative
        );
    }
}
