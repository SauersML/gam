//! #2234 step 1a — the arrow orbit lane prices the dense route's orbit-eliminated criterion.
//!
//! Two arbiters. On arrow operators small enough to hold densely, the bordered elimination is
//! checked against the dense route's own stiffening and pencil classification: its log-determinant,
//! its shifted inertia, its solves, and the verdicts of the certificate, which must refuse a band
//! direction, a saddle and a direction inside its own rounding margin. On the planted-circle and the
//! two-circle compact TopK fixtures the lane's value and ρ-gradient are checked against the dense
//! route's at one converged state, to the dense route's own finite-difference agreement there, and
//! a mutant that drops the orbit integral's legs must miss it.

use super::tests_orbit_gradient_2234::{
    arrow_dot, arrow_norm, displaced, displaced_compact, inner_gradient, periodic_fixture,
    richardson, topk_two_circle_fixture,
};
use super::tests_pencil_classification_2933::DensePencilMetric;
use super::*;
use ndarray::{Array1, Array2};

/// A deterministic stream in `[−1, 1)`.
fn stream(seed: u64) -> impl FnMut() -> f64 {
    let mut state = seed;
    move || {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        2.0 * (((state >> 11) as f64) / ((1u64 << 53) as f64)) - 1.0
    }
}

fn offsets(rows: usize, q: usize) -> Vec<usize> {
    (0..=rows).map(|row| row * q).collect()
}

fn dense_of(blocks: &ArrowJointBlocks) -> Array2<f64> {
    let dim = blocks.dim();
    let total_t = blocks.total_t;
    let mut dense = Array2::<f64>::zeros((dim, dim));
    for (row, block) in blocks.rows.iter().enumerate() {
        let (start, end) = blocks.row_range(row);
        dense.slice_mut(ndarray::s![start..end, start..end]).assign(block);
    }
    dense
        .slice_mut(ndarray::s![..total_t, total_t..])
        .assign(&blocks.cross);
    dense
        .slice_mut(ndarray::s![total_t.., ..total_t])
        .assign(&blocks.cross.t());
    dense
        .slice_mut(ndarray::s![total_t.., total_t..])
        .assign(&blocks.border);
    dense
}

/// A random symmetric positive-definite arrow: `diagonal`-shifted row and border blocks and a
/// small coordinate–border coupling.
fn random_arrow(rows: usize, q: usize, k: usize, seed: u64, diagonal: f64) -> ArrowJointBlocks {
    let mut unit = stream(seed);
    let mut blocks = ArrowJointBlocks::zeros(&offsets(rows, q), k);
    for block in &mut blocks.rows {
        let factor = Array2::from_shape_fn((q, q), |_| unit());
        let mut spd = factor.dot(&factor.t()) * 0.3;
        for index in 0..q {
            spd[[index, index]] += diagonal;
        }
        *block = spd;
    }
    blocks.cross = Array2::from_shape_fn(blocks.cross.raw_dim(), |_| 0.1 * unit());
    let factor = Array2::from_shape_fn((k, k), |_| unit());
    let mut border = factor.dot(&factor.t()) * 0.3;
    for index in 0..k {
        border[[index, index]] += diagonal * (rows as f64);
    }
    blocks.border = border;
    blocks
}

/// `(ΦT, N, N⁻¹, ‖Φ‖_F)` for the tangents `T`.
fn orbit_images(
    metric: &ArrowJointBlocks,
    tangents: &Array2<f64>,
) -> (Array2<f64>, Array2<f64>, Array2<f64>, f64) {
    let mut images = Array2::<f64>::zeros(tangents.raw_dim());
    for column in 0..tangents.ncols() {
        images
            .column_mut(column)
            .assign(&metric.apply(tangents.column(column)));
    }
    let mut gram = tangents.t().dot(&images);
    symmetrized_in_place(&mut gram);
    let gram_inverse =
        symmetric_positive_function(&gram, "test", f64::recip).expect("a positive-definite Gram");
    (images, gram, gram_inverse, metric.frobenius())
}

fn describe(verdict: &ArrowOrbitPencilVerdict) -> String {
    match verdict {
        ArrowOrbitPencilVerdict::Certified(certificate) => format!("certified ({certificate})"),
        ArrowOrbitPencilVerdict::Refused(refusal) => {
            format!("refused by the {}: {refusal}", refusal.lane())
        }
    }
}

fn verdict_of(
    operator: &ArrowJointBlocks,
    metric: &ArrowJointBlocks,
    tangents: &Array2<f64>,
) -> (ArrowOrbitPencilVerdict, BorderedArrowFactor) {
    let (images, gram, gram_inverse, metric_frobenius) = orbit_images(metric, tangents);
    let factor = BorderedArrowFactor::eliminate(operator, None, images.view(), true)
        .expect("the bordered elimination at τ = 0");
    let verdict = certify_orbit_pencil(
        operator,
        metric,
        tangents.view(),
        images.view(),
        &gram,
        &gram_inverse,
        metric_frobenius,
        &factor,
    )
    .expect("a verdict");
    (verdict, factor)
}

/// The dense route's own classification of the same state: `stiffen_compact_orbits` and the
/// pencil block, as `materialize_dense_exact_a_geometry` builds them.
fn dense_route_block(
    operator: &ArrowJointBlocks,
    metric: &ArrowJointBlocks,
    tangents: &Array2<f64>,
) -> (ExactHessianSpectralBlock, DensePencilMetric) {
    let metric_dense = dense_of(metric);
    let pencil = DensePencilMetric::new(metric_dense.clone(), &metric_dense).expect("a PD metric");
    let (stiffened, _) =
        SaeManifoldTerm::stiffen_compact_orbits(dense_of(operator), tangents.clone(), &pencil)
            .expect("the dense stiffening");
    let block = SaeManifoldTerm::exact_hessian_spectral_block(stiffened, &pencil)
        .expect("the dense pencil block");
    (block, pencil)
}

/// The bordered elimination prices `log|A_s|`, counts the stiffened pencil below every shift, and
/// solves `K` exactly as the dense route's own stiffening and classification do: on a random arrow
/// with two orbit tangents spanning rows and border, and on one whose single border orbit carries a
/// negative chord, so `A` is indefinite and `A_s` is not.
#[test]
fn the_bordered_elimination_is_the_dense_stiffened_pencil_2234() {
    let (rows, q, k) = (7usize, 2usize, 3usize);
    let metric = random_arrow(rows, q, k, 11, 1.2);
    let operator = random_arrow(rows, q, k, 29, 0.8);
    let dim = operator.dim();
    let mut unit = stream(53);
    let tangents = Array2::from_shape_fn((dim, 2), |_| unit());
    check_against_the_dense_route("general", &operator, &metric, &tangents, false);

    // A metric with no coordinate–border coupling keeps a border orbit's image `Φτ` on the border,
    // so `A − s·(Φτ)(Φτ)ᵀ` is still an arrow and `A_s` does not see `s`: the chord is flipped
    // negative with the complement untouched.
    let mut metric = random_arrow(rows, q, k, 71, 1.2);
    metric.cross.fill(0.0);
    let mut operator = random_arrow(rows, q, k, 83, 0.8);
    let total_t = operator.total_t;
    let mut tangents = Array2::<f64>::zeros((dim, 1));
    for slot in 0..k {
        tangents[[total_t + slot, 0]] = unit();
    }
    let tangent = tangents.column(0).to_owned();
    let image = metric.apply(tangent.view());
    let chord = tangent.dot(&operator.apply(tangent.view()));
    let norm = tangent.dot(&image);
    let scale = 2.0 * chord / (norm * norm);
    for row in 0..k {
        for column in 0..k {
            operator.border[[row, column]] -= scale * image[total_t + row] * image[total_t + column];
        }
    }
    let flipped = tangent.dot(&operator.apply(tangent.view()));
    assert!(flipped < 0.0, "premise: the flipped chord {flipped} is negative");
    check_against_the_dense_route("negative chord", &operator, &metric, &tangents, true);
}

fn check_against_the_dense_route(
    label: &str,
    operator: &ArrowJointBlocks,
    metric: &ArrowJointBlocks,
    tangents: &Array2<f64>,
    indefinite: bool,
) {
    let dim = operator.dim();
    let total_t = operator.total_t;
    let mut unit = stream(97);
    let k = operator.k;
    let (operator_eigen, _) = dense_of(operator).eigh(Side::Lower).expect("dense eigh");
    let negative_a = operator_eigen.iter().filter(|&&value| value < 0.0).count();
    let (block, _) = dense_route_block(operator, metric, tangents);
    let dense_negative = (0..block.eigenvalues.len())
        .filter(|&index| block.eigenvalues[index] < -block.rank_floor(index))
        .count();
    let dense_band = (0..block.eigenvalues.len())
        .filter(|&index| block.eigenvalues[index].abs() <= block.rank_floor(index))
        .count();
    let dense_log_det = block.eigenvalues.iter().map(|value| value.ln()).sum::<f64>() + block.metric_log_det;
    let (verdict, factor) = verdict_of(operator, metric, tangents);
    let (images, gram, _, _) = orbit_images(metric, tangents);
    let (gram_values, _) = gram.eigh(Side::Lower).expect("Gram eigh");
    let arrow_log_det = factor.log_abs_det - gram_values.iter().map(|value| value.ln()).sum::<f64>();
    eprintln!(
        "[#2234 1a primitive {label}] A negatives={negative_a} dense stiffened negatives={dense_negative} \
         band={dense_band} log|A_s| dense={dense_log_det:.15e} arrow={arrow_log_det:.15e} verdict={}",
        describe(&verdict)
    );
    assert_eq!(negative_a > 0, indefinite, "{label}: premise on the definiteness of A");
    assert_eq!(
        (dense_negative, dense_band),
        (0, 0),
        "{label}: premise — the dense route retains every direction"
    );
    assert!(
        matches!(verdict, ArrowOrbitPencilVerdict::Certified(_)),
        "{label}: a state the dense route retains whole must certify, got {}",
        describe(&verdict)
    );
    let bar = 1.0e-10 * (1.0 + dense_log_det.abs());
    assert!(
        (arrow_log_det - dense_log_det).abs() <= bar,
        "{label}: log|A_s| of the bordered elimination {arrow_log_det} against the dense pencil \
         {dense_log_det} (|Δ| {:e}, bar {bar:e})",
        (arrow_log_det - dense_log_det).abs()
    );

    // Shifted inertia against the dense pencil, at every gap between consecutive curvatures below 1.
    let mut curvatures = block.eigenvalues.to_vec();
    curvatures.sort_by(f64::total_cmp);
    let mut checked = 0usize;
    for pair in curvatures.windows(2) {
        let tau = 0.5 * (pair[0] + pair[1]);
        if tau >= 1.0 || (pair[1] - pair[0]) < 1.0e-6 {
            continue;
        }
        let below = curvatures.iter().filter(|&&value| value < tau).count();
        let shifted = BorderedArrowFactor::eliminate(operator, Some((metric, tau)), images.view(), false)
            .expect("a shifted elimination");
        assert_eq!(
            shifted.negative.checked_sub(tangents.ncols()),
            Some(below),
            "{label}: at τ = {tau} the bordered count and the dense pencil disagree"
        );
        checked += 1;
    }
    eprintln!("[#2234 1a primitive {label}] shifted inertia checked at {checked} shifts");
    assert!(checked >= 3, "{label}: premise — the pencil has curvatures below its orbit's own unit");

    // `K⁻¹` against a dense solve of the bordered operator.
    let orbits = tangents.ncols();
    let mut bordered = Array2::<f64>::zeros((dim + orbits, dim + orbits));
    bordered.slice_mut(ndarray::s![..dim, ..dim]).assign(&dense_of(operator));
    bordered.slice_mut(ndarray::s![..dim, dim..]).assign(&images);
    bordered.slice_mut(ndarray::s![dim.., ..dim]).assign(&images.t());
    let (values, vectors) = bordered.eigh(Side::Lower).expect("bordered eigh");
    let rhs = Array1::from_shape_fn(dim + orbits, |_| unit());
    let dense_solution = vectors.dot(&(vectors.t().dot(&rhs) / &values));
    let mut border_rhs = Array1::<f64>::zeros(k + orbits);
    border_rhs.slice_mut(ndarray::s![..k]).assign(&rhs.slice(ndarray::s![total_t..dim]));
    border_rhs.slice_mut(ndarray::s![k..]).assign(&rhs.slice(ndarray::s![dim..]));
    let (coordinates, border) =
        factor.solve(&operator.row_offsets, rhs.slice(ndarray::s![..total_t]), border_rhs.view());
    let mut arrow_solution = Array1::<f64>::zeros(dim + orbits);
    arrow_solution.slice_mut(ndarray::s![..total_t]).assign(&coordinates);
    arrow_solution.slice_mut(ndarray::s![total_t..dim]).assign(&border.slice(ndarray::s![..k]));
    arrow_solution.slice_mut(ndarray::s![dim..]).assign(&border.slice(ndarray::s![k..]));
    let gap = (&arrow_solution - &dense_solution).iter().fold(0.0_f64, |m, v| m.max(v.abs()));
    let scale = dense_solution.iter().fold(0.0_f64, |m, v| m.max(v.abs()));
    eprintln!("[#2234 1a primitive {label}] K⁻¹ solve max|Δ| {gap:.3e} against max|x| {scale:.3e}");
    assert!(gap <= 1.0e-10 * scale.max(1.0), "{label}: the bordered solve misses the dense one by {gap:e}");
}

/// The certificate's `λ_min(Φ)` bound encloses the true smallest eigenvalue from below on every
/// arrow, weakly and strongly coupled, and on a row-heterogeneous layout, measured against a dense
/// eigensolve.
#[test]
fn the_metric_floor_encloses_the_smallest_metric_eigenvalue_2234() {
    let mut cases = Vec::new();
    for (seed, coupling) in [(3u64, 0.1), (5, 1.0), (7, 3.0), (13, 0.0)] {
        let mut metric = random_arrow(6, 2, 4, seed, 0.9);
        metric.cross.mapv_inplace(|value| value * coupling / 0.1);
        // Keep Φ positive definite at the stronger couplings by stiffening its border.
        let shift = 1.0 + coupling * coupling * 20.0;
        for index in 0..metric.k {
            metric.border[[index, index]] += shift;
        }
        cases.push((format!("uniform seed={seed} coupling={coupling}"), metric));
    }
    let heterogeneous_offsets = vec![0usize, 1, 4, 4, 6, 9];
    let mut unit = stream(17);
    let mut metric = ArrowJointBlocks::zeros(&heterogeneous_offsets, 3);
    for block in &mut metric.rows {
        let q = block.nrows();
        let factor = Array2::from_shape_fn((q, q), |_| unit());
        *block = factor.dot(&factor.t()) + Array2::<f64>::eye(q) * 0.2;
    }
    metric.cross = Array2::from_shape_fn(metric.cross.raw_dim(), |_| 0.4 * unit());
    metric.border = Array2::<f64>::eye(3) * 20.0;
    cases.push(("heterogeneous rows, an empty row".to_string(), metric));
    for (label, metric) in cases {
        let (values, _) = dense_of(&metric).eigh(Side::Lower).expect("dense eigh");
        let smallest = values.iter().copied().fold(f64::INFINITY, f64::min);
        let floor = metric_smallest_eigenvalue_bound(&metric).expect("a positive-definite metric");
        eprintln!(
            "[#2234 1a floor] {label}: λ_min(Φ) = {smallest:.6e}, bound {floor:.6e}, ratio {:.3}",
            floor / smallest
        );
        assert!(smallest > 0.0, "{label}: premise — Φ is positive definite");
        assert!(
            floor > 0.0 && floor <= smallest * (1.0 + 1.0e-12),
            "{label}: the bound {floor:e} does not enclose λ_min(Φ) = {smallest:e} from below"
        );
    }
}

/// A diagonal arrow in the identity metric with its orbit on border slot 0, whose chord is
/// negative: the stiffened pencil is `{1} ∪ {d_i}` over the other slots.
fn diagonal_instance(complement: &[f64]) -> (ArrowJointBlocks, ArrowJointBlocks, Array2<f64>) {
    let (rows, q, k) = (3usize, 2usize, 3usize);
    let mut operator = ArrowJointBlocks::zeros(&offsets(rows, q), k);
    let mut metric = ArrowJointBlocks::zeros(&offsets(rows, q), k);
    let total_t = rows * q;
    assert_eq!(complement.len(), total_t + k - 1);
    for row in 0..rows {
        for local in 0..q {
            operator.rows[row][[local, local]] = complement[row * q + local];
            metric.rows[row][[local, local]] = 1.0;
        }
    }
    operator.border[[0, 0]] = -0.3;
    metric.border[[0, 0]] = 1.0;
    for slot in 1..k {
        operator.border[[slot, slot]] = complement[total_t + slot - 1];
        metric.border[[slot, slot]] = 1.0;
    }
    let mut tangents = Array2::<f64>::zeros((total_t + k, 1));
    tangents[[total_t, 0]] = 1.0;
    (operator, metric, tangents)
}

/// The certificate's negative controls: a band direction, a saddle, and a direction inside the
/// certificate's own rounding margin each refuse by name; the same state without them certifies
/// though its chord is negative, and the dense route classifies each the same way.
#[test]
fn the_certificate_refuses_a_band_a_saddle_and_its_own_margin_2234() {
    let certified = [2.0, 3.0, 0.5, 1.5, 4.0, 2.5, 0.7, 1.1];
    let (operator, metric, tangents) = diagonal_instance(&certified);
    let (verdict, _) = verdict_of(&operator, &metric, &tangents);
    eprintln!("[#2234 1a controls] certified: {}", describe(&verdict));
    let certificate = match &verdict {
        ArrowOrbitPencilVerdict::Certified(certificate) => *certificate,
        other => panic!("the positive complement must certify, got {}", describe(other)),
    };

    let mut band = certified;
    band[2] = 1.0e-12;
    let (operator, metric, tangents) = diagonal_instance(&band);
    let (verdict, _) = verdict_of(&operator, &metric, &tangents);
    let (block, _) = dense_route_block(&operator, &metric, &tangents);
    let dense_band = (0..block.eigenvalues.len())
        .filter(|&index| block.eigenvalues[index].abs() <= block.rank_floor(index))
        .count();
    eprintln!("[#2234 1a controls] band: {} / dense in-band {dense_band}", describe(&verdict));
    assert_eq!(dense_band, 1, "premise: the dense route holds the planted direction in its band");
    assert!(
        matches!(
            verdict,
            ArrowOrbitPencilVerdict::Refused(ArrowOrbitRefusal::Band { count: 1, .. })
        ),
        "a band direction must refuse as the band, got {}",
        describe(&verdict)
    );

    let mut saddle = certified;
    saddle[4] = -0.2;
    let (operator, metric, tangents) = diagonal_instance(&saddle);
    let (verdict, _) = verdict_of(&operator, &metric, &tangents);
    let (block, _) = dense_route_block(&operator, &metric, &tangents);
    let dense_negative = (0..block.eigenvalues.len())
        .filter(|&index| block.eigenvalues[index] < -block.rank_floor(index))
        .count();
    eprintln!("[#2234 1a controls] saddle: {} / dense negative {dense_negative}", describe(&verdict));
    assert_eq!(dense_negative, 1, "premise: the dense route resolves the planted negative direction");
    assert!(
        matches!(
            verdict,
            ArrowOrbitPencilVerdict::Refused(ArrowOrbitRefusal::NegativeInertia { count: 1, .. })
        ),
        "a saddle of the complement must refuse as negative inertia, got {}",
        describe(&verdict)
    );

    let mut margin = certified;
    margin[6] = certificate.threshold;
    let (operator, metric, tangents) = diagonal_instance(&margin);
    let (verdict, _) = verdict_of(&operator, &metric, &tangents);
    let text = describe(&verdict);
    eprintln!("[#2234 1a controls] margin: {text}");
    assert!(
        matches!(
            verdict,
            ArrowOrbitPencilVerdict::Refused(ArrowOrbitRefusal::Undecided { .. })
        ),
        "a curvature at the band edge itself must refuse as undecided, got {text}"
    );
    assert!(
        text.contains("undecided") && text.contains("margin δ"),
        "the undecided refusal must name itself and print its margin: {text}"
    );
}

/// One route comparison at a converged state: the value and every coordinate's fixed-state partial
/// and implicit correction, dense against the arrow orbit lane, each to the dense route's own
/// finite-difference agreement; the mutant without the orbit integral's legs must miss.
fn compare_routes_at_state(
    label: &str,
    state: &mut SaeManifoldTerm,
    target: &Array2<f64>,
    anchor: &SaeManifoldRho,
    coordinates: &[(&str, usize)],
    displace: &dyn Fn(&SaeManifoldTerm, &SaeArrowVector, f64) -> SaeManifoldTerm,
) {
    let (_, loss, cache, geometry) = state
        .penalized_quasi_laplace_criterion_with_geometry(
            target.view(),
            anchor,
            None,
            0,
            0.05,
            1.0e-6,
            1.0e-6,
            true,
        )
        .expect("the criterion prices the converged state");
    let geometry = geometry.expect("the dense criterion hands out the block it priced");
    assert!(
        !geometry.orbit_generators.is_empty(),
        "premise: the dense route integrates the orbits"
    );
    let dense = state
        .analytic_outer_rho_gradient_components_with_bundle(
            target.view(),
            anchor,
            &loss,
            &cache,
            None,
            None,
            Some(&geometry),
        )
        .expect("dense gradient components");
    let (dense_log_det, _) = state
        .exact_observed_information_log_dets_with_saddle_directions(anchor, target.view(), &cache, &mut Vec::new())
        .expect("the dense log-determinant");
    let generators: Vec<CircleOrbitGenerator> = state
        .separated_compact_orbit_pricing(anchor, target.view(), &cache)
        .expect("orbit pricing")
        .into_iter()
        .filter_map(|pricing| match pricing {
            CompactOrbitPricing::ExactCircle(generator) => Some(generator),
            CompactOrbitPricing::Laplace { .. } => None,
        })
        .collect();
    let mut arrow_geometry = state
        .arrow_orbit_geometry(anchor, target.view(), &cache, generators)
        .expect("the arrow orbit lane certifies the converged state");
    let arrow = state
        .analytic_outer_rho_gradient_components_arrow_orbit(
            target.view(),
            anchor,
            &loss,
            &cache,
            &arrow_geometry,
        )
        .expect("arrow orbit gradient components");
    // The mutant: the same lane with every leg of −2·Σ log I_k dropped, which the lane reads off
    // each orbit's coupling forms and multiplier images alone.
    arrow_geometry.orbits.clear();
    arrow_geometry.multiplier_images.clear();
    let mutant = state
        .analytic_outer_rho_gradient_components_arrow_orbit(
            target.view(),
            anchor,
            &loss,
            &cache,
            &arrow_geometry,
        )
        .expect("mutant gradient components");

    let arrow_log_det = arrow_geometry.log_det();
    let value_bar = 1.0e-9 * (1.0 + dense_log_det.abs());
    eprintln!(
        "[#2234 1a {label}] log det dense={dense_log_det:.15e} arrow={arrow_log_det:.15e} \
         (|Δ| {:.3e}, bar {value_bar:.3e}; orbit correction {:.6e})",
        (arrow_log_det - dense_log_det).abs(),
        arrow_geometry.log_det_correction
    );
    assert!(
        (arrow_log_det - dense_log_det).abs() <= value_bar,
        "{label}: the arrow orbit lane's log-determinant misses the dense route's"
    );
    assert!(
        arrow_geometry.log_det_correction.abs() > 10.0 * value_bar,
        "{label}: premise — the orbit correction must be visible at the value bar"
    );

    let residual = inner_gradient(state, target.view(), anchor);
    let flat = anchor.flat_coordinates();
    let price = |at_state: &SaeManifoldTerm, at: &SaeManifoldRho| -> f64 {
        let mut arm = at_state.clone();
        arm.penalized_quasi_laplace_criterion_with_cache(target.view(), at, None, 0, 0.05, 1.0e-6, 1.0e-6)
            .expect("the criterion prices the fixed state")
            .0
    };
    let moved_rho = |index: usize, step: f64| -> SaeManifoldRho {
        let mut moved = flat.clone();
        moved[index] += step;
        anchor.from_flat(moved.view()).expect("a nearby ρ")
    };
    let mut mutant_misses = Vec::new();
    for &(name, index) in coordinates {
        let partial = |components: &SaeOuterRhoGradientComponents| {
            components.explicit[index] + components.logdet_trace[index] + components.occam[index]
        };
        let cost_over_rho = |step: f64| -> f64 {
            (price(state, &moved_rho(index, step)) - price(state, &moved_rho(index, -step))) / (2.0 * step)
        };
        let step = 1.0e-3_f64;
        let (partial_fd, partial_spread) = richardson(cost_over_rho(step), cost_over_rho(0.5 * step));
        let g_rho = state
            .outer_rho_gradient_ift_rhs(anchor, index, &cache)
            .expect("implicit right-hand side");
        let a_pinv_g = state
            .solve_exact_stationarity(anchor, target.view(), &cache, &g_rho)
            .expect("A⁺ g_ρ");
        let theta_hat = SaeArrowVector {
            t: a_pinv_g.t.mapv(|value| -value),
            beta: a_pinv_g.beta.mapv(|value| -value),
        };
        let response_step = 1.0e-4 / arrow_norm(&theta_hat).max(1.0);
        let cost_along_response = |eps: f64| -> f64 {
            (price(&displace(state, &theta_hat, eps), anchor) - price(&displace(state, &theta_hat, -eps), anchor))
                / (2.0 * eps)
        };
        let (directional_fd, directional_spread) =
            richardson(cost_along_response(response_step), cost_along_response(0.5 * response_step));
        let implicit_fd = directional_fd - arrow_dot(&residual, &theta_hat);
        // The dense route's own agreement with the frozen criterion, never tighter than the
        // difference's resolution.
        let partial_bar = (partial(&dense) - partial_fd)
            .abs()
            .max(10.0 * partial_spread + 1.0e-6 * partial_fd.abs().max(1.0));
        let implicit_bar = (dense.third_order_correction[index] - implicit_fd)
            .abs()
            .max(10.0 * directional_spread + 1.0e-6 * directional_fd.abs().max(1.0));
        eprintln!(
            "[#2234 1a {label}] {name}: partial dense={:.12e} arrow={:.12e} mutant={:.12e} fd={partial_fd:.12e} \
             bar={partial_bar:.3e}; implicit dense={:.12e} arrow={:.12e} fd={implicit_fd:.12e} bar={implicit_bar:.3e}",
            partial(&dense),
            partial(&arrow),
            partial(&mutant),
            dense.third_order_correction[index],
            arrow.third_order_correction[index],
        );
        assert!(
            (partial(&arrow) - partial(&dense)).abs() <= partial_bar,
            "{label} {name}: the arrow partial {} misses the dense {} (|Δ| {:e}, bar {partial_bar:e})",
            partial(&arrow),
            partial(&dense),
            (partial(&arrow) - partial(&dense)).abs()
        );
        assert!(
            (arrow.third_order_correction[index] - dense.third_order_correction[index]).abs() <= implicit_bar,
            "{label} {name}: the arrow implicit correction {} misses the dense {} (|Δ| {:e}, bar {implicit_bar:e})",
            arrow.third_order_correction[index],
            dense.third_order_correction[index],
            (arrow.third_order_correction[index] - dense.third_order_correction[index]).abs()
        );
        if name.starts_with("ard") {
            assert!(
                partial(&dense).abs() > 10.0 * partial_bar,
                "{label} {name}: premise — the compared partial {} must clear ten bars {partial_bar:e}",
                partial(&dense)
            );
            mutant_misses.push((partial(&mutant) - partial(&dense)).abs() / partial_bar);
        }
    }
    let misses_text = mutant_misses.iter().map(|miss| format!("{miss:.3e}")).collect::<Vec<_>>().join(", ");
    eprintln!("[#2234 1a {label}] mutant |Δ|/bar per ARD axis [{misses_text}]");
    assert!(
        mutant_misses.iter().any(|&miss| miss > 10.0),
        "{label}: the mutant without the orbit integral's legs matches the dense route on every ARD \
         axis ([{misses_text}]), so the comparison cannot see them"
    );
}

#[test]
fn the_arrow_orbit_lane_prices_the_dense_criterion_and_its_gradient_2234() {
    let (term, target, rho) = periodic_fixture();
    let mut objective =
        SaeManifoldOuterObjective::new(term, target.clone(), None, rho, 40, 0.05, 1.0e-6, 1.0e-6);
    let anchor = objective.baseline_rho.clone();
    objective
        .evaluate_outer_criterion_route(&anchor, true, false)
        .expect("the dense route converges the anchor");
    let mut state = objective.term.clone();
    let coordinates = [
        ("ard", anchor.ard_flat_index(0, 0)),
        ("smooth", anchor.smooth_flat_index(0)),
    ];
    compare_routes_at_state("circle", &mut state, &target, &anchor, &coordinates, &displaced);
}

#[test]
fn the_arrow_orbit_lane_prices_two_compact_topk_orbits_2234() {
    let (term, target, rho) = topk_two_circle_fixture();
    let mut objective =
        SaeManifoldOuterObjective::new(term, target.clone(), None, rho, 40, 0.05, 1.0e-6, 1.0e-6);
    let anchor = objective.baseline_rho.clone();
    objective
        .evaluate_outer_criterion_route(&anchor, true, false)
        .expect("the dense route converges the anchor");
    let mut state = objective.term.clone();
    let row_offsets = {
        let mut probe = state.clone();
        let (_, _, cache) = probe
            .penalized_quasi_laplace_criterion_with_cache(target.view(), &anchor, None, 0, 0.05, 1.0e-6, 1.0e-6)
            .expect("the criterion prices the converged state");
        cache.row_offsets.to_vec()
    };
    let coordinates = [
        ("ard0", anchor.ard_flat_index(0, 0)),
        ("ard1", anchor.ard_flat_index(1, 0)),
        ("smooth0", anchor.smooth_flat_index(0)),
    ];
    let displace = |term: &SaeManifoldTerm, step: &SaeArrowVector, scale: f64| {
        displaced_compact(term, &row_offsets, step, scale)
    };
    compare_routes_at_state("topk", &mut state, &target, &anchor, &coordinates, &displace);
}
