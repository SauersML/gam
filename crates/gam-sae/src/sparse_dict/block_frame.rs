//! Frame steps and the stationarity certificate for tied block projectors: the
//! one-shot lane's polar step and the streaming lane's Rayleigh–Ritz step.

use crate::frames::GrassmannFrame;
use gam_linalg::faer_ndarray::{FaerCholesky, FaerEigh, FaerSvd};
use ndarray::{Array2, ArrayView2, ArrayViewMut2, Axis, s};

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

/// Write each column of `columns` (`P×c`), scaled to unit length, into the
/// matching row of `rows` (`c×P`). A column without a finite positive length
/// leaves its row as it was.
fn write_unit_columns(columns: ArrayView2<'_, f64>, mut rows: ArrayViewMut2<'_, f32>) {
    for (index, column) in columns.axis_iter(Axis(1)).enumerate() {
        let norm = column.dot(&column).sqrt();
        if norm.is_finite() && norm > 0.0 {
            for (entry, &value) in rows.row_mut(index).iter_mut().zip(column.iter()) {
                *entry = (value / norm) as f32;
            }
        }
    }
}

/// The streaming lane's frame step: Rayleigh–Ritz on the tied surrogate.
///
/// With supports and γ held fixed, `‖Σ_h ΔP_h x‖² ≤ k Σ_h ‖ΔP_h x‖²` majorizes
/// the tied loss under a simultaneous update of every block. Each block's
/// majorizer is linear in its new projector, so lowering it means raising
/// `tr(U'ᵀ H U')` for `H = (2γ − kγ²)XᵀX + γ²(XᵀV + VᵀX)`, and it equals the loss
/// at the stored frame. Any `U'` that raises the trace therefore lowers the tied
/// loss at fixed supports. On a subspace containing U, Ky Fan's principle makes
/// the top-`b` Ritz vectors the largest such raise, for an indefinite `H` too, so
/// the step needs no spectral shift and no step length (#2502).
///
/// `current` stores U transposed (`b×P`), `directions` stores the search rows
/// `[R, P]` (`2b×P`) the previous step left, and `action` holds `H·[U, R, P]`
/// (`P×3b`) from this pass. The top-`b` Ritz vectors on `span[U, R, P]`, rotated
/// into the gauge closest to U, go to `proposal`. The next search rows go to
/// `next_directions`: the Ritz residual `HU' − U'(U'ᵀHU')` and the part of U
/// outside `span U'`, each scaled to unit length. This is block LOBPCG whose
/// residual is one pass behind, because `H·R` exists only once a pass has
/// accumulated it. Returns the stationarity certificate of `tied_frame_tangent`,
/// read at U, which does not depend on the step.
pub(super) fn ritz_tied_frame_step(
    current: ArrayView2<'_, f32>,
    directions: ArrayView2<'_, f32>,
    action: ArrayView2<'_, f64>,
    code_second: ArrayView2<'_, f64>,
    normal_multiplier: f64,
    mut proposal: ArrayViewMut2<'_, f32>,
    mut next_directions: ArrayViewMut2<'_, f32>,
) -> Result<f64, String> {
    let (b, p) = current.dim();
    next_directions.fill(0.0);
    let own_action = action.slice(s![.., ..b]);
    if own_action.iter().all(|&value| value == 0.0) {
        proposal.assign(&current);
        return Ok(0.0);
    }
    let (tangent, stationarity) =
        tied_frame_tangent(current, own_action, code_second, normal_multiplier)?;
    // A zero search row has a zero action column and spans nothing.
    let searched: Vec<usize> = (0..directions.nrows())
        .filter(|&row| directions.row(row).iter().any(|&value| value != 0.0))
        .collect();
    if searched.is_empty() {
        // span W = span U, so Rayleigh–Ritz can only re-express U. The tangent
        // gradient becomes the next pass's first search direction.
        proposal.assign(&current);
        write_unit_columns(tangent.view(), next_directions.slice_mut(s![..b, ..]));
        return Ok(stationarity);
    }
    let width = b + searched.len();
    let mut basis = Array2::<f64>::zeros((p, width));
    let mut image = Array2::<f64>::zeros((p, width));
    for axis in 0..b {
        for (entry, &value) in basis.column_mut(axis).iter_mut().zip(current.row(axis)) {
            *entry = value as f64;
        }
        image.column_mut(axis).assign(&action.column(axis));
    }
    for (offset, &row) in searched.iter().enumerate() {
        for (entry, &value) in basis
            .column_mut(b + offset)
            .iter_mut()
            .zip(directions.row(row))
        {
            *entry = value as f64;
        }
        image.column_mut(b + offset).assign(&action.column(b + row));
    }
    let gram = basis.t().dot(&basis);
    let crossed = basis.t().dot(&image);
    // H is symmetric; WᵀHW picks up asymmetric roundoff from separate products.
    let rayleigh = (&crossed + &crossed.t()) * 0.5;
    let (gram_values, gram_vectors) = gram
        .eigh(faer::Side::Lower)
        .map_err(|error| format!("search subspace Gram eigendecomposition: {error}"))?;
    // The stored rows carry f32 rounding E with |E_ij| ≤ ½u₃₂|W_ij|, which moves
    // the Gram's eigenvalues by at most 2‖W‖₂‖E‖_F ≤ u₃₂·√width·λ_max. A Gram
    // direction below that bound is rounding, not a direction the rows resolve.
    let largest = gram_values.iter().copied().fold(0.0_f64, f64::max);
    let floor = STORED_FRAME_RESOLUTION * (width as f64).sqrt() * largest;
    let kept: Vec<usize> = (0..width)
        .filter(|&index| gram_values[index] > floor)
        .collect();
    if kept.len() < b {
        return Err(format!(
            "search subspace resolves {} directions, fewer than the frame's {b}",
            kept.len()
        ));
    }
    let whitening = Array2::from_shape_fn((width, kept.len()), |(row, column)| {
        gram_vectors[[row, kept[column]]] / gram_values[kept[column]].sqrt()
    });
    let reduced = whitening.t().dot(&rayleigh).dot(&whitening);
    let reduced = (&reduced + &reduced.t()) * 0.5;
    let (ritz_values, ritz_vectors) = reduced
        .eigh(faer::Side::Lower)
        .map_err(|error| format!("reduced Rayleigh–Ritz eigendecomposition: {error}"))?;
    let mut order: Vec<usize> = (0..kept.len()).collect();
    order.sort_by(|&left, &right| ritz_values[right].total_cmp(&ritz_values[left]));
    let top = &order[..b];
    let ritz_sum: f64 = top.iter().map(|&index| ritz_values[index]).sum();
    // The surrogate trace at the stored frame, tr((UᵀU)⁻¹UᵀHU), because the f32
    // frame is only approximately orthonormal.
    let current_sum: f64 = {
        let solved = gram
            .slice(s![..b, ..b])
            .to_owned()
            .cholesky(faer::Side::Lower)
            .map_err(|error| format!("stored frame Gram factorization: {error}"))?
            .solve_mat(&rayleigh.slice(s![..b, ..b]).to_owned());
        (0..b).map(|axis| solved[[axis, axis]]).sum()
    };
    if !(ritz_sum > current_sum) {
        proposal.assign(&current);
        write_unit_columns(tangent.view(), next_directions.slice_mut(s![..b, ..]));
        return Ok(stationarity);
    }
    let coefficients = whitening.dot(&ritz_vectors.select(Axis(1), top));
    // Rotate the Ritz basis into the gauge closest to U (orthogonal Procrustes),
    // so a bisected trial averages matching columns.
    let (left, _, right_t) = basis
        .dot(&coefficients)
        .t()
        .dot(&basis.slice(s![.., ..b]))
        .svd(true, true)
        .map_err(|error| format!("Ritz frame alignment: {error}"))?;
    let left = left.ok_or("Ritz frame alignment omitted the left factor")?;
    let right_t = right_t.ok_or("Ritz frame alignment omitted the right factor")?;
    let coefficients = coefficients.dot(&left.dot(&right_t));
    let ritz = basis.dot(&coefficients);
    for (mut row, column) in proposal.outer_iter_mut().zip(ritz.axis_iter(Axis(1))) {
        for (entry, &value) in row.iter_mut().zip(column.iter()) {
            *entry = value as f32;
        }
    }
    // The next search rows, read off this pass's H: the Ritz residual, and the
    // part of the stored frame outside the proposal's span.
    let ritz_rayleigh = coefficients.t().dot(&rayleigh).dot(&coefficients);
    let residual = image.dot(&coefficients) - ritz.dot(&ritz_rayleigh);
    let frame = basis.slice(s![.., ..b]);
    let previous = &frame - &ritz.dot(&ritz.t().dot(&frame));
    write_unit_columns(residual.view(), next_directions.slice_mut(s![..b, ..]));
    write_unit_columns(previous.view(), next_directions.slice_mut(s![b.., ..]));
    Ok(stationarity)
}
