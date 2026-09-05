//! Exact conditional direction update after profiling one scalar code per row.
//!
//! For a unit direction d, c_i=(x_i·d)/(1+rho). Substitution into the penalized
//! loss leaves a constant minus sum_i(x_i·d)^2/(1+rho); the decoder ridge is
//! constant on the unit sphere. Thus every shared rho has the same conditional
//! direction: a leading eigenvector of the uncentered cluster scatter matrix.

use gam_linalg::faer_ndarray::FaerEigh;
use ndarray::{Array1, Array2, ArrayView1, ArrayView2};

/// Return the maximizing unit direction nearest the previous direction within
/// the numerically unresolved leading eigenspace. Form the smaller Gram matrix
/// in tiles of its own dimension: scratch and eigensystem storage are O(q²),
/// q=min(cluster rows, features), independent of the longer axis. No N×K
/// routing matrix, ambient covariance per dictionary atom, or cluster copy is
/// retained. The caller owns routing, significance admission, and normalization.
pub(super) fn profiled_direction(
    x: ArrayView2<'_, f32>,
    rows: &[usize],
    previous: ArrayView1<'_, f32>,
) -> Result<Array1<f32>, String> {
    let m = rows.len();
    let p = x.ncols();
    let q = m.min(p);
    if q == 0 {
        return Err("single-atom profiling requires a nonempty cluster and feature axis".into());
    }
    let primal = p <= m;
    let long_axis = m.max(p);
    let mut gram = Array2::<f64>::zeros((q, q));
    for start in (0..long_axis).step_by(q) {
        let end = (start + q).min(long_axis);
        let tile = if primal {
            Array2::from_shape_fn((end - start, p), |(i, j)| x[[rows[start + i], j]] as f64)
        } else {
            Array2::from_shape_fn((m, end - start), |(i, j)| x[[rows[i], start + j]] as f64)
        };
        if primal {
            gram += &tile.t().dot(&tile);
        } else {
            gram += &tile.dot(&tile.t());
        }
    }
    let (values, vectors) = gram
        .eigh(faer::Side::Lower)
        .map_err(|error| format!("single-atom scatter eigensolve failed: {error}"))?;
    let leading = (0..q)
        .max_by(|&a, &b| values[a].total_cmp(&values[b]))
        .expect("nonempty scatter spectrum");
    let largest = values[leading];
    if !(largest.is_finite() && largest > 0.0) {
        return Err(format!(
            "single-atom scatter has no positive finite energy: {largest}"
        ));
    }
    // Gram accumulation and the q-dimensional eigensolve both contribute
    // roundoff. Scaling by scatter trace bounds cancellation in a rank-deficient
    // Gram; the longest dot product, not just q, determines its resolution.
    let work = (long_axis as f64 + q as f64) * f64::EPSILON;
    let trace = gram.diag().sum();
    let resolution = work / (1.0 - work) * trace;
    if !(resolution.is_finite() && resolution >= 0.0 && resolution < largest) {
        return Err(
            "single-atom scatter spectrum is unresolved at its arithmetic precision".into(),
        );
    }
    if values
        .iter()
        .any(|&value| !value.is_finite() || value < -resolution)
    {
        return Err("single-atom scatter spectrum is not positive semidefinite".into());
    }
    let previous_coordinates = if primal {
        previous.mapv(f64::from)
    } else {
        Array1::from_shape_fn(m, |i| {
            (0..p)
                .map(|j| x[[rows[i], j]] as f64 * previous[j] as f64)
                .sum()
        })
    };
    let mut coordinates = Array1::<f64>::zeros(q);
    for index in 0..q {
        if largest - values[index] <= resolution {
            let eigenvector = vectors.column(index);
            let mut weight = eigenvector.dot(&previous_coordinates);
            if !primal {
                weight /= values[index];
            }
            coordinates.scaled_add(weight, &eigenvector);
        }
    }
    // If the old direction is orthogonal to the maximizing eigenspace, every
    // maximizer is equally near. Select the eigensolver's leading basis vector.
    if coordinates.iter().all(|&value| value == 0.0) {
        coordinates.assign(&vectors.column(leading));
    }
    let direction = if primal {
        coordinates
    } else {
        Array1::from_shape_fn(p, |j| {
            rows.iter()
                .zip(coordinates.iter())
                .map(|(&row, &weight)| x[[row, j]] as f64 * weight)
                .sum()
        })
    };
    let norm = direction.dot(&direction).sqrt();
    if !(norm.is_finite() && norm > 0.0) {
        return Err("single-atom maximizing direction is not finite and nonzero".into());
    }
    Ok(direction.mapv(|value| (value / norm) as f32))
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn profiling_solves_the_direction_instead_of_taking_one_power_step() {
        let x = array![[2.0_f32, 0.0], [0.0, 1.0], [-2.0, 0.0], [0.0, -1.0]];
        let previous = array![0.5_f32.sqrt(), 0.5_f32.sqrt()];
        let direction = profiled_direction(x.view(), &[0, 1, 2, 3], previous.view()).unwrap();
        assert!((direction[0].abs() - 1.0).abs() < f32::EPSILON);
        assert!(direction[1].abs() < f32::EPSILON);
        // MOD plus normalization gives (4,1)/sqrt(17), which captures less
        // scatter energy than the profiled maximizer's exact value 8.
        let power_energy = (8.0 * 16.0 + 2.0) / 17.0;
        let captured: f64 = x
            .outer_iter()
            .map(|row| row.dot(&direction).powi(2) as f64)
            .sum();
        assert!((captured - 8.0).abs() < f64::EPSILON);
        assert!(captured > power_energy);
    }

    #[test]
    fn dual_gram_recovers_the_same_direction_after_zero_feature_padding() {
        let x = array![[2.0_f32, 1.0], [-1.0, 1.0], [1.0, 0.0]];
        let previous = array![0.6_f32, 0.8];
        let primal = profiled_direction(x.view(), &[0, 1, 2], previous.view()).unwrap();
        let wide = Array2::from_shape_fn((3, 19), |(i, j)| if j < 2 { x[[i, j]] } else { 0.0 });
        let wide_previous = Array1::from_shape_fn(19, |j| if j < 2 { previous[j] } else { 0.0 });
        let dual = profiled_direction(wide.view(), &[0, 1, 2], wide_previous.view()).unwrap();
        for i in 0..2 {
            for j in 0..2 {
                assert!((primal[i] * primal[j] - dual[i] * dual[j]).abs() <= 4.0 * f32::EPSILON);
            }
        }
        assert!(dual.iter().skip(2).all(|&value| value == 0.0));
    }

    #[test]
    fn isotropic_cluster_preserves_the_previous_maximizer() {
        let x = array![[1.0_f32, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, -1.0]];
        let previous = array![0.6_f32, 0.8];
        let direction = profiled_direction(x.view(), &[0, 1, 2, 3], previous.view()).unwrap();
        for (&actual, &expected) in direction.iter().zip(previous.iter()) {
            assert!((actual - expected).abs() < f32::EPSILON);
        }
    }

    #[test]
    fn dense_wide_and_tall_clusters_recover_the_known_leading_projector() {
        for (n, p) in [(32usize, 2048usize), (512, 128)] {
            let scale = (p as f64).sqrt();
            // Two orthogonal feature directions and orthogonal row scores.
            // Both are dense, so this exercises every Gram-tile coordinate.
            let x = Array2::from_shape_fn((n, p), |(i, j)| {
                let a = if i % 2 == 0 { 2.0 } else { -2.0 };
                let b = if i % 4 < 2 { 1.0 } else { -1.0 };
                ((a + if j % 2 == 0 { b } else { -b }) / scale) as f32
            });
            let previous = Array1::from_elem(p, (1.0 / scale) as f32);
            let rows: Vec<usize> = (0..n).collect();
            let start = std::time::Instant::now();
            let direction = profiled_direction(x.view(), &rows, previous.view()).unwrap();
            let alignment = direction
                .iter()
                .map(|&v| v as f64 / scale)
                .sum::<f64>()
                .abs();
            assert!(
                (alignment - 1.0).abs() < 8.0 * f32::EPSILON as f64,
                "N={n} P={p}: projector alignment {alignment}"
            );
            eprintln!("single-atom profile N={n} P={p}: {:?}", start.elapsed());
        }
    }
}
