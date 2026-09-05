//! Shared polar step and stationarity certificate for tied block projectors.

use crate::frames::GrassmannFrame;
use gam_linalg::faer_ndarray::FaerCholesky;
use ndarray::{Array2, ArrayView2, ArrayViewMut2};

/// Increase the Rayleigh surrogate `tr(U' H U)` by polarizing `(H+sI)U`,
/// where the caller supplies a shift that makes `H+sI` positive semidefinite.
/// `current` stores U transposed; `action` initially contains H U.
///
/// Simultaneous updates add a normal majorization term to H U. Subtracting
/// `normal_multiplier * U * code_second` from the certificate's denominator
/// recovers the conditional objective's action. Neither that normal term nor
/// the spectral shift changes its tangent gradient. Sequential block updates
/// use a zero multiplier. Both fitting paths therefore certify the same tied
/// reconstruction objective, independently of how conservative the step is.
pub(super) fn polar_tied_frame_step(
    current: ArrayView2<'_, f32>,
    mut action: ArrayViewMut2<'_, f64>,
    code_second: ArrayView2<'_, f64>,
    normal_multiplier: f64,
    shift: f64,
    mut proposal: ArrayViewMut2<'_, f32>,
) -> Result<f64, String> {
    let (b, p) = current.dim();
    if action.iter().all(|&value| value == 0.0) {
        proposal.assign(&current);
        return Ok(0.0);
    }
    let gram = Array2::from_shape_fn((b, b), |(axis, column)| {
        (0..p)
            .map(|feature| current[[axis, feature]] as f64 * current[[column, feature]] as f64)
            .sum::<f64>()
    });
    let normal_rhs = Array2::from_shape_fn((b, b), |(axis, column)| {
        (0..p)
            .map(|feature| current[[axis, feature]] as f64 * action[[feature, column]])
            .sum::<f64>()
    });
    // The stored f32 frame is only approximately orthonormal. UU' is therefore
    // not its orthogonal projector: it leaks an O(u32) normal component into
    // the purported tangent gradient even when the represented subspace is
    // exactly invariant. Solve the tiny Gram system to apply U(U'U)^-1 U'.
    // This removes normal leakage algebraically, without subtracting a tolerance.
    let normal = gram
        .cholesky(faer::Side::Lower)
        .map_err(|error| format!("tied frame projector Gram factorization: {error}"))?
        .solve_mat(&normal_rhs);
    let mut tangent_sq = 0.0;
    let mut conditional_sq = 0.0;
    for feature in 0..p {
        for column in 0..b {
            let mut projected = 0.0;
            let mut normal_correction = 0.0;
            for axis in 0..b {
                let direction = current[[axis, feature]] as f64;
                projected += direction * normal[[axis, column]];
                normal_correction += direction * code_second[[axis, column]];
            }
            tangent_sq += (action[[feature, column]] - projected).powi(2);
            conditional_sq +=
                (action[[feature, column]] - normal_multiplier * normal_correction).powi(2);
            action[[feature, column]] += shift * current[[column, feature]] as f64;
        }
    }
    let stationarity = if conditional_sq == 0.0 {
        0.0
    } else {
        (tangent_sq / conditional_sq).sqrt()
    };
    if action.iter().all(|&value| value == 0.0) {
        proposal.assign(&current);
        return Ok(stationarity);
    }
    let frame = GrassmannFrame::polar_update(action.view())?;
    let u = frame.frame();
    for column in 0..b {
        // The Grassmann frame canonicalizes column signs. Undo that gauge
        // convention to recover the maximizing polar factor of H U + s U.
        let alignment = (0..p)
            .map(|feature| u[[feature, column]] * action[[feature, column]])
            .sum::<f64>();
        let orientation = if alignment < 0.0 { -1.0 } else { 1.0 };
        for feature in 0..p {
            proposal[[column, feature]] = (orientation * u[[feature, column]]) as f32;
        }
    }
    Ok(stationarity)
}
