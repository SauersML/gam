//! Shared polar step and stationarity certificate for tied block projectors.

use crate::frames::GrassmannFrame;
use gam_linalg::faer_ndarray::{FaerCholesky, FaerSvd};
use ndarray::{Array2, ArrayView2, ArrayViewMut2};

/// The resolution of a frame residual read off `f32`-stored frames.
///
/// The block dictionary is stored as `f32`. A stationarity residual computed
/// from those frames is a difference of quantities carrying `f32` round-off, so
/// below `f32::EPSILON` it is not SMALL, it is UNREPRESENTABLE: the number is
/// the storage noise, not the distance to the fixed point. A convergence bar
/// placed under it asks the frames a question their own precision cannot
/// answer, and no amount of iterating can clear it.
///
/// This is not a chosen tolerance. It is the machine epsilon of the type the
/// frames are stored in — the instrument reporting its own noise floor. The
/// workspace already states the principle for the timing gates, in
/// `gam_math::paired_timing`: "There is no chosen tolerance here: the
/// instrument reports its noise floor, and that is the only denominator a
/// parity bar can honestly be stated in." Same argument, different instrument.
///
/// Callers take `tolerance.max(STORED_FRAME_RESOLUTION)`, so a bar ABOVE the
/// floor is honoured exactly as configured and only a bar below it is lifted to
/// the floor (#2825).
pub(super) const STORED_FRAME_RESOLUTION: f64 = f32::EPSILON as f64;

/// Relative distance of the actual stored projectors, without subtracting
/// order-one overlap traces to recover a tiny squared displacement.
pub(super) fn stored_projector_distance(
    current: ArrayView2<'_, f32>,
    next: ArrayView2<'_, f32>,
) -> Result<f64, String> {
    if current
        .iter()
        .zip(next.iter())
        .all(|(left, right)| left == right)
    {
        return Ok(0.0);
    }
    let u = current.t().mapv(f64::from);
    let v = next.t().mapv(f64::from);
    let gram_u = u.t().dot(&u);
    let gram_v = v.t().dot(&v);
    let scale = gram_u
        .iter()
        .chain(gram_v.iter())
        .map(|value| value * value)
        .sum::<f64>();
    if scale == 0.0 {
        return Ok(0.0);
    }
    // Orthogonal Procrustes changes only the gauge of V, including when the
    // overlap is singular. It makes U'V symmetric, so the midpoint identity
    // below has nonnegative terms in exact arithmetic.
    let (left, _, right_t) = u
        .t()
        .dot(&v)
        .svd(true, true)
        .map_err(|error| format!("frame projector alignment: {error}"))?;
    let left = left.ok_or("frame projector alignment omitted the left factor")?;
    let right_t = right_t.ok_or("frame projector alignment omitted the right factor")?;
    let aligned = v.dot(&right_t.t().dot(&left.t()));
    let difference = &aligned - &u;
    let midpoint = (&aligned + &u) * 0.5;
    let gram_midpoint = midpoint.t().dot(&midpoint);
    let gram_difference = difference.t().dot(&difference);
    let mixed = midpoint.t().dot(&difference);
    // VV'-UU' = SD'+DS'. Its squared Frobenius norm is
    // 2 tr(S'S D'D) + 2 tr((S'D)^2). Compute the latter trace explicitly,
    // retaining any roundoff skew rather than assuming exact symmetry.
    let mut distance_sq = 0.0;
    for i in 0..mixed.nrows() {
        for j in 0..mixed.ncols() {
            distance_sq += 2.0
                * (gram_midpoint[[i, j]] * gram_difference[[j, i]] + mixed[[i, j]] * mixed[[j, i]]);
        }
    }
    Ok((distance_sq.max(0.0) / scale).sqrt())
}

/// The secant pair a frame proposal leaves for the step after it:
/// `step_sq = ⟨S, S⟩` and `step_dot_gradient = ⟨S, G⟩`, where `S` is the
/// proposal minus the stored frame it was formed at and `G` is that frame's
/// tangent gradient (see [`secant_tied_frame_step`]).
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub(super) struct FrameSecant {
    pub(super) step_sq: f64,
    pub(super) step_dot_gradient: f64,
}

/// The tangent gradient of the tied surrogate at the stored frame, and the
/// stationarity certificate read off it.
///
/// `current` stores U transposed and `action` holds H U. Returns `G`, the part
/// of H U normal to span U removed, as a `P×b` array, together with
/// `‖G‖ / ‖H U − normal_multiplier·U·code_second‖`.
///
/// Simultaneous updates add a normal majorization term to H U. Subtracting
/// `normal_multiplier * U * code_second` from the certificate's denominator
/// recovers the conditional objective's action. Neither that normal term nor
/// any spectral shift or step size changes the tangent gradient, so every frame
/// step certifies the same tied reconstruction objective, independently of how
/// the step is sized. Sequential block updates use a zero multiplier.
fn tied_frame_tangent(
    current: ArrayView2<'_, f32>,
    action: ArrayView2<'_, f64>,
    code_second: ArrayView2<'_, f64>,
    normal_multiplier: f64,
) -> Result<(Array2<f64>, f64), String> {
    let (b, p) = current.dim();
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
    let mut tangent = Array2::<f64>::zeros((p, b));
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
            let component = action[[feature, column]] - projected;
            tangent[[feature, column]] = component;
            tangent_sq += component.powi(2);
            conditional_sq +=
                (action[[feature, column]] - normal_multiplier * normal_correction).powi(2);
        }
    }
    let stationarity = if conditional_sq == 0.0 {
        0.0
    } else {
        (tangent_sq / conditional_sq).sqrt()
    };
    Ok((tangent, stationarity))
}

/// Write the maximizing polar factor of `action` (`P×b`) into `proposal`, which
/// stores the frame transposed. The Grassmann frame canonicalizes column signs,
/// so each column's orientation is restored to that of its action column.
fn polarize_into(
    action: ArrayView2<'_, f64>,
    mut proposal: ArrayViewMut2<'_, f32>,
) -> Result<(), String> {
    let (p, b) = action.dim();
    let frame = GrassmannFrame::polar_update(action)?;
    let u = frame.frame();
    for column in 0..b {
        let alignment = (0..p)
            .map(|feature| u[[feature, column]] * action[[feature, column]])
            .sum::<f64>();
        let orientation = if alignment < 0.0 { -1.0 } else { 1.0 };
        for feature in 0..p {
            proposal[[column, feature]] = (orientation * u[[feature, column]]) as f32;
        }
    }
    Ok(())
}

/// Increase the Rayleigh surrogate `tr(U' H U)` by polarizing `(H+sI)U`,
/// where the caller supplies a shift that makes `H+sI` positive semidefinite.
/// `current` stores U transposed; `action` initially contains H U. Returns the
/// stationarity certificate of `tied_frame_tangent`.
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
    let (_, stationarity) =
        tied_frame_tangent(current, action.view(), code_second, normal_multiplier)?;
    for feature in 0..p {
        for column in 0..b {
            action[[feature, column]] += shift * current[[column, feature]] as f64;
        }
    }
    if action.iter().all(|&value| value == 0.0) {
        proposal.assign(&current);
        return Ok(stationarity);
    }
    polarize_into(action.view(), proposal)?;
    Ok(stationarity)
}

/// The streaming lane's frame step.
///
/// With the secant pair left by the step that produced the stored frame, take
/// the Riemannian gradient step `polar(U + G/κ)`, whose curvature is the secant
/// measured along that step: `κ = (⟨S, G_prev⟩ − ⟨S, G⟩) / ⟨S, S⟩` for
/// `S = U − U_prev`. Without a pair, or when the secant measures no positive
/// curvature, take the majorizer step of [`polar_tied_frame_step`].
///
/// The majorizer's shift sums row-level bounds over the block's rows, where only
/// a spectral bound is needed, so it dwarfs the curvature along the gradient and
/// each step moves the frame by a small fraction of it (#2502). The secant step
/// is sized by the measured curvature instead. Neither step is trusted by
/// itself: the caller commits a proposal only after a paired full pass proves
/// strict RSS descent, and bisects toward the baseline otherwise. Returns the
/// stationarity certificate, which does not depend on the step, and the
/// proposal's own secant pair.
pub(super) fn secant_tied_frame_step(
    current: ArrayView2<'_, f32>,
    mut action: ArrayViewMut2<'_, f64>,
    code_second: ArrayView2<'_, f64>,
    normal_multiplier: f64,
    shift: f64,
    previous: Option<(ArrayView2<'_, f32>, FrameSecant)>,
    mut proposal: ArrayViewMut2<'_, f32>,
) -> Result<(f64, FrameSecant), String> {
    let (b, p) = current.dim();
    if action.iter().all(|&value| value == 0.0) {
        proposal.assign(&current);
        return Ok((0.0, FrameSecant::default()));
    }
    let (tangent, stationarity) =
        tied_frame_tangent(current, action.view(), code_second, normal_multiplier)?;
    let curvature = previous.and_then(|(prior, secant)| {
        if !(secant.step_sq > 0.0) {
            return None;
        }
        let mut step_dot_gradient = 0.0;
        for axis in 0..b {
            for feature in 0..p {
                let step = current[[axis, feature]] as f64 - prior[[axis, feature]] as f64;
                step_dot_gradient += step * tangent[[feature, axis]];
            }
        }
        let curvature = (secant.step_dot_gradient - step_dot_gradient) / secant.step_sq;
        (curvature.is_finite() && curvature > 0.0).then_some(curvature)
    });
    for feature in 0..p {
        for column in 0..b {
            let direction = current[[column, feature]] as f64;
            action[[feature, column]] = match curvature {
                Some(curvature) => direction + tangent[[feature, column]] / curvature,
                None => action[[feature, column]] + shift * direction,
            };
        }
    }
    if action.iter().all(|&value| value == 0.0) {
        proposal.assign(&current);
        return Ok((stationarity, FrameSecant::default()));
    }
    polarize_into(action.view(), proposal.view_mut())?;
    let mut step_sq = 0.0;
    let mut step_dot_gradient = 0.0;
    for axis in 0..b {
        for feature in 0..p {
            let step = proposal[[axis, feature]] as f64 - current[[axis, feature]] as f64;
            step_sq += step * step;
            step_dot_gradient += step * tangent[[feature, axis]];
        }
    }
    Ok((
        stationarity,
        FrameSecant {
            step_sq,
            step_dot_gradient,
        },
    ))
}
