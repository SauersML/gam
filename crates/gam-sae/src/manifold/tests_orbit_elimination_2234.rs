//! #2234 — the orbit-stiffened operator and the orbit-eliminated exact-`A` pseudo-inverse.
//!
//! The dense evaluation prices `A_s = P_ΦᵀAP_Φ + ΦτN⁻¹τᵀΦ` and solves with `A` through the eliminated
//! orbit coordinate, off one decomposition. The fixture carries a NEGATIVE chord along `τ` under a
//! nonorthogonal `Φ`: the connection-term state of the #2234 stall pin, where pricing `A` refuses.

use super::tests_pencil_classification_2933::DensePencilMetric;
use super::*;
use ndarray::{Array1, Array2};

const DIM: usize = 6;

/// A symmetric positive-definite metric that is not a multiple of the identity.
fn fixture_metric() -> Array2<f64> {
    let lower = Array2::from_shape_fn((DIM, DIM), |(i, j)| {
        if j > i {
            0.0
        } else if i == j {
            1.0 + 0.3 * (i as f64)
        } else {
            0.25 * ((i * 3 + j) as f64 + 1.0).cos()
        }
    });
    lower.dot(&lower.t())
}

fn fixture_tangent() -> Array1<f64> {
    Array1::from_vec(vec![1.0, 0.5, -0.3, 0.2, 0.1, -0.4])
}

/// `A = CCᵀ + I − λ·(Φτ)(Φτ)ᵀ`, with `λ` set so the chord `τᵀAτ = −τᵀ(CCᵀ + I)τ` is negative.
fn fixture_operator(metric: &Array2<f64>, tangent: &Array1<f64>) -> Array2<f64> {
    let factor = Array2::from_shape_fn((DIM, DIM), |(i, j)| ((1 + i + 2 * j) as f64).sin());
    let mut base = factor.dot(&factor.t());
    for index in 0..DIM {
        base[[index, index]] += 1.0;
    }
    let image = metric.dot(tangent);
    let norm = tangent.dot(&image);
    let base_chord = tangent.dot(&base.dot(tangent));
    let lambda = 2.0 * base_chord / (norm * norm);
    let mut operator = base;
    for row in 0..DIM {
        for col in 0..DIM {
            operator[[row, col]] -= lambda * image[row] * image[col];
        }
    }
    operator
}

fn stiffened_block(
    operator: &Array2<f64>,
    metric: &Array2<f64>,
    tangent: &Array1<f64>,
    eliminate: bool,
) -> ExactHessianSpectralBlock {
    let pencil = DensePencilMetric::new(metric.clone(), metric).expect("positive-definite metric");
    let tangents = tangent.clone().insert_axis(ndarray::Axis(1));
    let (stiffened, stiffening) =
        SaeManifoldTerm::stiffen_compact_orbits(operator.clone(), tangents, &pencil).expect("stiffening");
    let mut block =
        SaeManifoldTerm::exact_hessian_spectral_block(stiffened, &pencil).expect("pencil geometry");
    if eliminate {
        block.orbit = Some(
            stiffening
                .expect("one tangent")
                .eliminate(&block, &pencil)
                .expect("orbit elimination"),
        );
    }
    block
}

fn border(v: &Array1<f64>) -> SaeArrowVector {
    SaeArrowVector {
        t: Array1::zeros(0),
        beta: v.clone(),
    }
}

#[test]
fn orbit_stiffening_prices_the_tangent_at_unit_metric_curvature_2234() {
    let metric = fixture_metric();
    let tangent = fixture_tangent();
    let operator = fixture_operator(&metric, &tangent);
    let chord = tangent.dot(&operator.dot(&tangent)) / tangent.dot(&metric.dot(&tangent));
    assert!(chord < 0.0, "the fixture's chord along τ must be negative, got {chord}");
    let pencil = DensePencilMetric::new(metric.clone(), &metric).expect("positive-definite metric");
    let (stiffened, _) = SaeManifoldTerm::stiffen_compact_orbits(
        operator.clone(),
        tangent.clone().insert_axis(ndarray::Axis(1)),
        &pencil,
    )
    .expect("stiffening");
    let scale = operator.iter().map(|v| v.abs()).fold(0.0_f64, f64::max)
        + metric.iter().map(|v| v.abs()).fold(0.0_f64, f64::max);
    let band = 64.0 * f64::EPSILON * scale * (DIM * DIM) as f64;
    // A_s τ = Φ τ: the orbit carries the metric's own curvature.
    let gap = &stiffened.dot(&tangent) - &metric.dot(&tangent);
    let gap_norm = gap.iter().map(|v| v.abs()).fold(0.0_f64, f64::max);
    eprintln!("[#2234 orbit elimination] chord μ = {chord:e}, |A_sτ − Φτ|_∞ = {gap_norm:e}, band {band:e}");
    assert!(gap_norm <= band, "A_s τ misses Φτ by {gap_norm} against band {band}");
    // A Φ-orthogonal complement vector keeps its quadratic form: qᵀA_s q = qᵀAq.
    let image = metric.dot(&tangent);
    let raw = Array1::from_vec(vec![0.2, -1.0, 0.7, 0.4, -0.6, 0.3]);
    let complement = &raw - &(&tangent * (raw.dot(&image) / tangent.dot(&image)));
    assert!(complement.dot(&image).abs() <= band * complement.dot(&complement).sqrt());
    let form_gap = (complement.dot(&stiffened.dot(&complement)) - complement.dot(&operator.dot(&complement))).abs();
    assert!(form_gap <= band * complement.dot(&complement), "complement form moved by {form_gap}");
    // The stiffened pencil has no negative direction left.
    let block = stiffened_block(&operator, &metric, &tangent, false);
    let negative = (0..DIM)
        .filter(|&index| block.eigenvalues[index] < -block.rank_floor(index))
        .count();
    assert_eq!(negative, 0, "the stiffened pencil kept a negative direction: {:?}", block.eigenvalues);
}

#[test]
fn orbit_elimination_returns_the_unstiffened_inverse_2234() {
    let metric = fixture_metric();
    let tangent = fixture_tangent();
    let operator = fixture_operator(&metric, &tangent);
    let rhs = Array1::<f64>::from_vec(vec![0.9, -0.4, 1.3, 0.05, -0.8, 0.6]);
    let rhs_norm = rhs.dot(&rhs).sqrt();
    let operator_norm = operator.iter().map(|v| v * v).sum::<f64>().sqrt();

    let eliminated = stiffened_block(&operator, &metric, &tangent, true);
    let solve = eliminated.solve_stationarity(&border(&rhs)).expect("eliminated solve");
    let solution = solve.step.beta.clone();
    let residual = &operator.dot(&solution) - &rhs;
    let residual_norm = residual.dot(&residual).sqrt();
    let scale = operator_norm * solution.dot(&solution).sqrt() + rhs_norm;
    let band = f64::EPSILON.sqrt() * scale;
    eprintln!(
        "[#2234 orbit elimination] |A x − rhs| = {residual_norm:e}, scale {scale:e}, retained {}",
        solve.retained_rank
    );
    assert_eq!(solve.retained_rank, DIM, "every direction of the fixture is resolved");
    assert!(residual_norm <= band, "the eliminated solve misses A x = rhs by {residual_norm} (band {band})");

    // The factors carry the same inverse.
    let (basis, weights) = eliminated.retained_pseudo_inverse_factors();
    let pseudo_inverse = basis.dot(&Array2::from_diag(&weights)).dot(&basis.t());
    let identity_gap = &operator.dot(&pseudo_inverse) - &Array2::<f64>::eye(DIM);
    let identity_norm = identity_gap.iter().map(|v| v.abs()).fold(0.0_f64, f64::max);
    assert!(
        identity_norm <= f64::EPSILON.sqrt() * operator_norm * pseudo_inverse.iter().map(|v| v.abs()).fold(0.0_f64, f64::max) * DIM as f64,
        "A·(B diag(w) Bᵀ) misses the identity by {identity_norm}"
    );

    // Positive control: the same stiffened block WITHOUT elimination inverts A_s, not A.
    let stiffened_only = stiffened_block(&operator, &metric, &tangent, false);
    let wrong = stiffened_only
        .solve_stationarity(&border(&rhs))
        .expect("stiffened solve")
        .step
        .beta;
    let wrong_residual = &operator.dot(&wrong) - &rhs;
    let wrong_norm = wrong_residual.dot(&wrong_residual).sqrt();
    eprintln!("[#2234 orbit elimination] control |A x_s − rhs| = {wrong_norm:e}");
    assert!(
        wrong_norm > 1.0e3 * band,
        "the stiffened pseudo-inverse already solves A (residual {wrong_norm}), so the pin cannot see the elimination"
    );
}

/// With `A = 0` the stiffened operator resolves only the declared tangent (`A_sτ = Φτ`) and the orbit
/// curvature sits in its band, so `A` retains no direction. The pseudo-inverse still carries the
/// tangent's projected column, so a rank read off the column count would report 1 and hide the
/// root step's all-in-band skip.
#[test]
fn a_tangent_only_orbit_retains_no_direction_2234() {
    let metric = fixture_metric();
    let tangent = fixture_tangent();
    let operator = Array2::<f64>::zeros((DIM, DIM));
    let rhs = Array1::<f64>::from_vec(vec![0.9, -0.4, 1.3, 0.05, -0.8, 0.6]);

    let eliminated = stiffened_block(&operator, &metric, &tangent, true);
    let (_, weights) = eliminated.retained_pseudo_inverse_factors();
    let solve = eliminated.solve_stationarity(&border(&rhs)).expect("tangent-only solve");
    eprintln!(
        "[#2234 orbit elimination] tangent-only: pseudo-inverse columns {}, retained {}, band {}",
        weights.len(),
        solve.retained_rank,
        solve.band.len()
    );
    // Premise: the only column is the tangent's, which a column count reads as rank 1.
    assert_eq!(weights.len(), 1, "the stiffened block must resolve exactly the declared tangent");
    assert_eq!(solve.retained_rank, 0, "a tangent-only orbit leaves A no resolved direction");
    assert_eq!(solve.band.len(), DIM, "every direction of the zero operator is held in the band");
}

/// The collapsed-chart state of the #2263 item-4 replay: every direction in band, the orbit held out
/// with them, and a metric whose spectrum spans many decades. The held-out band is removed through
/// its `Φ`-orthonormal primal vectors, so the solve certifies. The control is the superseded
/// normal-equations projection onto the same images, rebuilt here from the block's own images: it
/// rounds at `Φ`'s conditioning and misses the √ε bar.
#[test]
fn an_in_band_orbit_solve_certifies_on_a_wide_metric_spectrum_2234() {
    let scales = [1.0e-3, 1.0e-2, 1.0e-1, 1.0e1, 1.0e2, 1.0e3];
    let base = fixture_metric();
    let metric = Array2::from_shape_fn((DIM, DIM), |(i, j)| scales[i] * base[[i, j]] * scales[j]);
    let tangent = fixture_tangent();
    let operator = Array2::<f64>::zeros((DIM, DIM));
    let rhs = Array1::<f64>::from_vec(vec![0.9, -0.4, 1.3, 0.05, -0.8, 0.6]);

    let eliminated = stiffened_block(&operator, &metric, &tangent, true);
    let solve = eliminated
        .solve_stationarity(&border(&rhs))
        .expect("the all-band orbit solve certifies");
    assert_eq!(solve.retained_rank, 0, "a zero operator leaves A no resolved direction");
    assert_eq!(solve.band.len(), DIM, "every direction, the orbit's included, is held out");

    // Control: the superseded removal, (HᵀH)⁻¹Hᵀr onto H = [P_ΦᵀΦW_Z, ΦTU], of the same residual.
    let orbit = eliminated.orbit.as_ref().expect("the block carries its eliminated orbit");
    let band_orbit: Vec<usize> = (0..orbit.curvatures.len())
        .filter(|&index| orbit.curvatures[index].abs() <= orbit.edges[index])
        .collect();
    let mut images = Array2::<f64>::zeros((DIM, eliminated.band.len() + band_orbit.len()));
    for position in 0..eliminated.band.len() {
        images
            .column_mut(position)
            .assign(&orbit.stiffening.project_dual(eliminated.band_metric_images.column(position)));
    }
    for (offset, &index) in band_orbit.iter().enumerate() {
        images
            .column_mut(eliminated.band.len() + offset)
            .assign(&orbit.direction_metric_images.column(index));
    }
    let residual = rhs.mapv(|value| -value);
    let gram = images.t().dot(&images);
    let (gram_values, _) = gram.eigh(Side::Lower).expect("image Gram spectrum");
    let norm = |vector: &Array1<f64>| vector.dot(vector).sqrt();
    let bar = f64::EPSILON.sqrt() * 2.0 * norm(&rhs);
    let superseded = match symmetric_positive_function(&gram, "control", f64::recip) {
        Ok(inverse) => {
            let coefficients = inverse.dot(&images.t().dot(&residual));
            Some(norm(&(&residual - &images.dot(&coefficients))))
        }
        Err(_) => None,
    };
    eprintln!(
        "[#2234 orbit elimination] wide metric: image Gram spectrum [{:e}, {:e}], superseded remainder {superseded:?} \
         against bar {bar:e}",
        gram_values[0],
        gram_values[gram_values.len() - 1],
    );
    assert!(
        superseded.is_none_or(|remainder| remainder > bar),
        "the normal-equations removal already certifies here (remainder {superseded:?}, bar {bar}), so the pin \
         cannot see the pairing"
    );
}

/// The certificate still refuses a step that does not solve `A`. After the decomposition, the block's
/// operator gains `c·(Φw)(Φw)ᵀ` along a retained direction `w`, so the eliminated step's physical
/// residual gains `c·Φw·(wᵀΦx)` along a direction the band does not hold out. The same block
/// without the plant certifies.
#[test]
fn a_residual_off_the_held_out_band_is_still_refused_2234() {
    let metric = fixture_metric();
    let tangent = fixture_tangent();
    let operator = fixture_operator(&metric, &tangent);
    let rhs = Array1::<f64>::from_vec(vec![0.9, -0.4, 1.3, 0.05, -0.8, 0.6]);

    let clean = stiffened_block(&operator, &metric, &tangent, true);
    let solve = clean.solve_stationarity(&border(&rhs)).expect("the unplanted solve certifies");
    assert_eq!(solve.retained_rank, DIM, "the fixture resolves every direction");

    let mut planted = stiffened_block(&operator, &metric, &tangent, true);
    let retained = (0..DIM)
        .find(|&index| planted.eigenvalues[index].abs() > planted.rank_floor(index) && (planted.eigenvalues[index] - 1.0).abs() > 0.5)
        .expect("a retained complement direction away from the orbit's unit curvature");
    let image = metric.dot(&planted.eigenvectors.column(retained));
    let operator_norm = operator.iter().map(|v| v * v).sum::<f64>().sqrt();
    let strength = 1.0e-1 * operator_norm / image.dot(&image);
    for row in 0..DIM {
        for col in 0..DIM {
            planted.operator[[row, col]] += strength * image[row] * image[col];
        }
    }
    let step = solve.step.beta.clone();
    let planted_residual = image.dot(&step).abs() * strength * image.dot(&image).sqrt();
    let bar = f64::EPSILON.sqrt() * (operator_norm * step.dot(&step).sqrt() + rhs.dot(&rhs).sqrt());
    eprintln!("[#2234 orbit elimination] planted residual {planted_residual:e} against bar {bar:e}");
    assert!(planted_residual > 1.0e3 * bar, "the plant must sit far above the bar ({planted_residual} vs {bar})");
    let refused = planted.solve_stationarity(&border(&rhs));
    assert!(
        matches!(&refused, Err(message) if message.contains("failed certification")),
        "a step with a residual off the held-out band must be refused, got {:?}",
        refused.map(|solve| solve.retained_rank)
    );
}
