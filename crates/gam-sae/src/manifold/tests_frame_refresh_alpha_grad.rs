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
