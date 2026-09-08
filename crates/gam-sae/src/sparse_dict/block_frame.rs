//! Shared polar step and stationarity certificate for tied block projectors.

use crate::frames::GrassmannFrame;
use gam_linalg::faer_ndarray::{FaerCholesky, FaerQr, FaerSvd, strict_symmetric_eigh};
use ndarray::{Array2, ArrayView2, ArrayViewMut2};

pub(super) use crate::frames::stored_projector_distance;

/// Scaled sum of squares: individual finite entries are never squared at their
/// original scale. The stationarity ratio need not materialize either norm.
#[derive(Default)]
struct FrameNorm {
    scale: f64,
    squared: f64,
}

impl FrameNorm {
    fn add(&mut self, value: f64) -> Result<(), String> {
        if !value.is_finite() {
            return Err("tied frame stationarity entry is not finite".into());
        }
        let magnitude = value.abs();
        if magnitude > self.scale {
            self.squared = 1.0 + self.squared * (self.scale / magnitude).powi(2);
            self.scale = magnitude;
        } else if magnitude != 0.0 {
            self.squared += (magnitude / self.scale).powi(2);
        }
        Ok(())
    }

    fn relative_to(&self, denominator: &Self) -> Result<f64, String> {
        if self.scale == 0.0 {
            return Ok(0.0);
        }
        if denominator.scale == 0.0 {
            return Err("nonzero tied frame tangent has zero conditional action norm".into());
        }
        let ratio = (self.scale / denominator.scale) * (self.squared / denominator.squared).sqrt();
        if !ratio.is_finite() || ratio == 0.0 {
            return Err("tied frame stationarity ratio is not representable".into());
        }
        Ok(ratio)
    }
}

/// Tangent of the actual current frame, before choosing a numerical update.
fn tied_frame_tangent(
    current: ArrayView2<'_, f64>,
    action: ArrayView2<'_, f64>,
    code_second: ArrayView2<'_, f64>,
    normal_multiplier: f64,
) -> Result<(Array2<f64>, f64), String> {
    let (b, p) = current.dim();
    if current
        .iter()
        .chain(action.iter())
        .chain(code_second.iter())
        .any(|value| !value.is_finite())
    {
        return Err("tied frame action contains non-finite values".into());
    }
    let mut tangent = action.to_owned();
    let mut conditional_norm = FrameNorm::default();
    for feature in 0..p {
        for column in 0..b {
            if normal_multiplier != 0.0 {
                let normal_correction = (0..b)
                    .map(|axis| current[[axis, feature]] * code_second[[axis, column]])
                    .sum::<f64>();
                tangent[[feature, column]] -= normal_multiplier * normal_correction;
            }
            conditional_norm.add(tangent[[feature, column]])?;
        }
    }
    let gram = Array2::from_shape_fn((b, b), |(axis, column)| {
        (0..p)
            .map(|feature| current[[axis, feature]] as f64 * current[[column, feature]] as f64)
            .sum::<f64>()
    });
    let normal_rhs = Array2::from_shape_fn((b, b), |(axis, column)| {
        (0..p)
            .map(|feature| current[[axis, feature]] as f64 * tangent[[feature, column]])
            .sum::<f64>()
    });
    // The stored f64 frame is only approximately orthonormal. UU' is therefore
    // not its orthogonal projector: it leaks a roundoff normal component into
    // the purported tangent gradient even when the represented subspace is
    // exactly invariant. Solve the tiny Gram system to apply U(U'U)^-1 U'.
    // This removes normal leakage algebraically, without subtracting a tolerance.
    let normal = gram
        .cholesky(faer::Side::Lower)
        .map_err(|error| format!("tied frame projector Gram factorization: {error}"))?
        .solve_mat(&normal_rhs);
    let mut tangent_norm = FrameNorm::default();
    for feature in 0..p {
        for column in 0..b {
            let mut projected = 0.0;
            for axis in 0..b {
                let direction = current[[axis, feature]] as f64;
                projected += direction * normal[[axis, column]];
            }
            tangent[[feature, column]] -= projected;
            tangent_norm.add(tangent[[feature, column]])?;
        }
    }
    let stationarity = tangent_norm.relative_to(&conditional_norm)?;
    Ok((tangent, stationarity))
}

/// Increase the Rayleigh surrogate by polarizing `(H+sI)U`, with a caller-owned
/// PSD shift. Streaming moments supply only H U at the frozen frame; they cannot
/// evaluate H on additional search directions without replaying the observations.
/// The normal majorization term and spectral shift do not change the tangent.
pub(super) fn polar_tied_frame_step(
    current: ArrayView2<'_, f64>,
    mut action: ArrayViewMut2<'_, f64>,
    code_second: ArrayView2<'_, f64>,
    normal_multiplier: f64,
    shift: f64,
    mut proposal: ArrayViewMut2<'_, f64>,
) -> Result<f64, String> {
    let (b, p) = current.dim();
    let (_, stationarity) =
        tied_frame_tangent(current, action.view(), code_second, normal_multiplier)?;
    if action.iter().all(|&value| value == 0.0) {
        proposal.assign(&current);
        return Ok(stationarity);
    }
    for feature in 0..p {
        for column in 0..b {
            action[[feature, column]] += shift * current[[column, feature]];
        }
    }
    if action.iter().any(|value| !value.is_finite()) {
        return Err("shifted tied frame action is not finite".into());
    }
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
            proposal[[column, feature]] = (orientation * u[[feature, column]]) as f64;
        }
    }
    Ok(stationarity)
}

/// One locally optimal Rayleigh step in span(U, (I-P_U) H U).
///
/// `project_operator` supplies the analytic compression Q' H Q for an orthonormal
/// trial basis Q. The trial space contains the current frame, so selecting its b
/// largest Ritz values cannot decrease tr(U' H U), including for indefinite H.
/// Scalar shifts leave this subproblem unchanged. Storage is O(p b + b²).
/// Returns the current frame's full tangent certificate, not a Ritz-space residual.
pub(super) fn rayleigh_ritz_tied_frame_step<F>(
    current: ArrayView2<'_, f64>,
    action: ArrayView2<'_, f64>,
    mut project_operator: F,
    mut proposal: ArrayViewMut2<'_, f64>,
) -> Result<f64, String>
where
    F: for<'a> FnMut(ArrayView2<'a, f64>) -> Result<Array2<f64>, String>,
{
    let (b, p) = current.dim();
    let second = Array2::zeros((b, b));
    let (mut tangent, stationarity) = tied_frame_tangent(current, action, second.view(), 0.0)?;
    if tangent.iter().all(|&value| value == 0.0) || b == p {
        proposal.assign(&current);
        return Ok(stationarity);
    }
    let (current_basis, _) = current
        .t()
        .qr()
        .map_err(|error| format!("tied frame trial basis QR: {error}"))?;
    // Reorthogonalization keeps the residual's numerical rank decision separate
    // from the order-one current frame. An absolute rank floor would incorrectly
    // remove a small but accurately represented tangent.
    for _ in 0..2 {
        let overlap = current_basis.t().dot(&tangent);
        tangent -= &current_basis.dot(&overlap);
    }
    let (directions, singular, _) = tangent
        .svd(true, false)
        .map_err(|error| format!("tied frame tangent SVD: {error}"))?;
    let directions = directions.ok_or("tied frame tangent SVD omitted directions")?;
    if singular
        .iter()
        .chain(directions.iter())
        .any(|value| !value.is_finite())
    {
        return Err("tied frame tangent SVD is not finite".into());
    }
    let largest = singular.iter().copied().fold(0.0_f64, f64::max);
    let cutoff = f64::EPSILON * p.max(b) as f64 * largest;
    let retained: Vec<usize> = singular
        .iter()
        .enumerate()
        .filter_map(|(index, &value)| (value > cutoff).then_some(index))
        .take(p - b)
        .collect();
    if retained.is_empty() {
        proposal.assign(&current);
        return Ok(stationarity);
    }
    let m = b + retained.len();
    let mut trial = Array2::zeros((p, m));
    trial.slice_mut(ndarray::s![.., ..b]).assign(&current_basis);
    for (slot, &index) in retained.iter().enumerate() {
        trial.column_mut(b + slot).assign(&directions.column(index));
    }
    // Orthogonalize the assembled basis once more before projecting H. This
    // changes its coordinates, not its span or the current-frame inclusion.
    let (trial, _) = trial
        .qr()
        .map_err(|error| format!("tied frame Ritz basis QR: {error}"))?;
    let reduced = project_operator(trial.view())?;
    if reduced.dim() != (m, m) {
        return Err("tied frame projected operator has the wrong shape".into());
    }
    let (values, vectors) = strict_symmetric_eigh(&reduced, faer::Side::Lower)
        .map_err(|error| format!("tied frame Ritz eigensolve: {error}"))?;
    let mut order: Vec<usize> = (0..m).collect();
    order.sort_by(|&left, &right| {
        values[right]
            .total_cmp(&values[left])
            .then(left.cmp(&right))
    });
    let selected = Array2::from_shape_fn((m, b), |(row, column)| vectors[[row, order[column]]]);
    let next = trial.dot(&selected);
    if next.iter().any(|value| !value.is_finite()) {
        return Err("tied frame Ritz proposal is not finite".into());
    }
    proposal.assign(&next.t());
    Ok(stationarity)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tangent_certificate_preserves_scale_and_handles_zero_conditional_action() {
        let current = ndarray::array![[1.0, 0.0]];
        let second = Array2::zeros((1, 1));
        for scale in [1.0e-200, 1.0, 1.0e200] {
            let action = ndarray::array![[3.0 * scale], [4.0 * scale]];
            let (_, measured) =
                tied_frame_tangent(current.view(), action.view(), second.view(), 0.0)
                    .expect("scaled tangent certificate");
            assert!(
                (measured - 0.8).abs() <= 4.0 * f64::EPSILON,
                "scale={scale}, measured={measured}"
            );
        }
        let current = ndarray::array![[0.6, 0.8]];
        let second = ndarray::array![[3.0]];
        let action = current.t().dot(&second);
        let (tangent, measured) =
            tied_frame_tangent(current.view(), action.view(), second.view(), 1.0)
                .expect("zero conditional action under a normal majorizer");
        assert!(tangent.iter().all(|&value| value == 0.0));
        assert_eq!(measured, 0.0);
        let mut nonzero = FrameNorm::default();
        nonzero.add(1.0e-200).expect("finite norm entry");
        assert!(nonzero.relative_to(&FrameNorm::default()).is_err());
    }

    #[test]
    fn ritz_step_solves_indefinite_plane_and_retains_current_certificate() {
        let current = ndarray::array![[1.0, 0.0]];
        let ratio = (5.0_f64.sqrt() - 1.0) / 2.0;
        let expected = ndarray::array![[1.0, ratio]] / (1.0 + ratio * ratio).sqrt();
        // A relative rank decision must retain the same accurately represented
        // direction when the entire objective is many orders below one.
        for scale in [1.0, 1.0e-120] {
            let h = ndarray::array![[-1.0, 2.0], [2.0, -3.0]] * scale;
            let action = h.dot(&current.t());
            let mut proposal = Array2::zeros((1, 2));
            let mut projected_dimension = 0;
            let measured = rayleigh_ritz_tied_frame_step(
                current.view(),
                action.view(),
                |basis| {
                    projected_dimension = basis.ncols();
                    Ok(basis.t().dot(&h.dot(&basis)))
                },
                proposal.view_mut(),
            )
            .expect("indefinite plane Ritz solve");
            assert_eq!(projected_dimension, 2);
            assert!((measured - 2.0 / 5.0_f64.sqrt()).abs() <= 16.0 * f64::EPSILON);
            let distance = stored_projector_distance(expected.view(), proposal.view())
                .expect("Ritz projector comparison");
            assert!(distance <= 64.0 * f64::EPSILON, "distance={distance}");
            let rayleigh = proposal.dot(&h.dot(&proposal.t()))[[0, 0]] / scale;
            assert!((rayleigh - (-2.0 + 5.0_f64.sqrt())).abs() <= 64.0 * f64::EPSILON);
        }
    }

    #[test]
    fn ritz_step_deflates_dependent_tangent_columns_without_losing_the_frame() {
        let current = ndarray::array![[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]];
        let h = ndarray::array![
            [3.0, 0.0, 0.4, 0.0],
            [0.0, 2.0, 0.0, 0.0],
            [0.4, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, -2.0],
        ];
        let mut proposal = Array2::zeros((2, 4));
        let mut projected_dimension = 0;
        rayleigh_ritz_tied_frame_step(
            current.view(),
            h.dot(&current.t()).view(),
            |basis| {
                projected_dimension = basis.ncols();
                Ok(basis.t().dot(&h.dot(&basis)))
            },
            proposal.view_mut(),
        )
        .expect("rank-one tangent Ritz solve");
        assert_eq!(projected_dimension, 3);
        let ratio = ((1.16_f64).sqrt() - 1.0) / 0.4;
        let norm = (1.0 + ratio * ratio).sqrt();
        let expected = ndarray::array![[1.0 / norm, 0.0, ratio / norm, 0.0], [0.0, 1.0, 0.0, 0.0]];
        let distance = stored_projector_distance(expected.view(), proposal.view())
            .expect("rank-deficient Ritz projector comparison");
        assert!(distance <= 128.0 * f64::EPSILON, "distance={distance}");
    }

    #[test]
    fn ritz_step_is_quiescent_at_an_exact_invariant_frame() {
        let current = ndarray::array![[1.0, 0.0, 0.0]];
        let action = ndarray::array![[3.0], [0.0], [0.0]];
        let mut proposal = Array2::zeros((1, 3));
        let measured = rayleigh_ritz_tied_frame_step(
            current.view(),
            action.view(),
            |_| Err("an invariant frame needs no extra observation pass".into()),
            proposal.view_mut(),
        )
        .expect("invariant frame");
        assert_eq!(measured, 0.0);
        assert_eq!(current, proposal);
    }
}
