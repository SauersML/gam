//! Split-out sibling test module, carved from `tests.rs` to keep that tracked
//! file under the 10k-line build gate (`build.rs` `MAX_TRACKED_FILE_LINES`),
//! mirroring the existing `tests_parallelism_invariance_1557` /
//! `tests_logdet_adjoint_780` splits:
//!
//! * `streaming_polar_refresh_reorients_frame` — the closed-form streaming polar
//!   frame refresh re-orients the decoder frame toward an accumulated
//!   cross-moment span while staying column-orthonormal.
//! * `small_p_zero_decoder_stays_full_b` — a zero decoder at small ambient `p`
//!   keeps the full-`B` border (frame activation is a no-op).

use super::*;
use approx::assert_abs_diff_eq;

/// Streaming polar refresh from an accumulated cross-moment re-orients the
/// frame toward the cross-moment span and keeps `B_k`'s in-span component
/// while staying column-orthonormal (the closed-form streaming step).
#[test]
pub(crate) fn streaming_polar_refresh_reorients_frame() {
    let m = 4usize;
    let p = 12usize;
    let r = 2usize;
    let mut frame0 = Array2::<f64>::zeros((p, r));
    frame0[[0, 0]] = 1.0;
    frame0[[1, 1]] = 1.0;
    let mut c0 = Array2::<f64>::zeros((m, r));
    for mu in 0..m {
        c0[[mu, 0]] = 1.0 + mu as f64;
        c0[[mu, 1]] = 0.5 - mu as f64;
    }
    let decoder = fast_abt(&c0, &frame0);
    let mut phi = Array2::<f64>::zeros((m, m));
    let mut jet = Array3::<f64>::zeros((m, m, 1));
    for mu in 0..m {
        phi[[mu, mu]] = 1.0;
        jet[[mu, mu, 0]] = 1.0;
    }
    let s_raw = gam_terms::basis::create_difference_penalty_matrix(m, 2, None).unwrap();
    let mut atom = SaeManifoldAtom::new_with_provided_function_gram(
        "stream",
        SaeAtomBasisKind::EuclideanPatch,
        1,
        phi,
        jet,
        decoder,
        s_raw,
    )
    .unwrap();
    atom.maybe_activate_decoder_frame().expect("activate");
    // New cross-moment pointing at axes {2,3}: refreshed frame must span them.
    let mut cross = Array2::<f64>::zeros((p, r));
    cross[[2, 0]] = 3.0;
    cross[[3, 1]] = 2.0;
    atom.refresh_frame_from_cross_moment(cross.view())
        .expect("refresh");
    let frame = atom.decoder_frame.as_ref().expect("frame");
    // Frame stays orthonormal.
    let gram = fast_atb(&frame.frame().to_owned(), &frame.frame().to_owned());
    for i in 0..r {
        for j in 0..r {
            let expect = if i == j { 1.0 } else { 0.0 };
            assert_abs_diff_eq!(gram[[i, j]], expect, epsilon = 1.0e-9);
        }
    }
    // Refreshed span aligns with the cross-moment axes {2,3} (angle ~0).
    let mut target_span = Array2::<f64>::zeros((p, r));
    target_span[[2, 0]] = 1.0;
    target_span[[3, 1]] = 1.0;
    let angle = frame
        .max_principal_angle(target_span.view())
        .expect("angle");
    assert_abs_diff_eq!(angle, 0.0, epsilon = 1.0e-9);
}

#[test]
pub(crate) fn small_p_zero_decoder_stays_full_b() {
    let m = 3usize;
    let p = 8usize;
    let mut phi = Array2::<f64>::zeros((m, m));
    let mut jet = Array3::<f64>::zeros((m, m, 1));
    for row in 0..m {
        phi[[row, row]] = 1.0;
        jet[[row, row, 0]] = 1.0;
    }
    let smooth_penalty = gam_terms::basis::create_difference_penalty_matrix(m, 2, None).unwrap();
    let mut atom = SaeManifoldAtom::new_with_provided_function_gram(
        "small-p-zero",
        SaeAtomBasisKind::EuclideanPatch,
        1,
        phi,
        jet,
        Array2::<f64>::zeros((m, p)),
        smooth_penalty,
    )
    .unwrap();

    assert_eq!(atom.decoder_frame_activation_rank().unwrap(), None);
    assert_eq!(atom.maybe_activate_decoder_frame().unwrap(), None);
    assert_eq!(atom.border_frame_rank(), p);
}
