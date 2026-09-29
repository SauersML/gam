#![cfg(test)]
//! The fibre oracle against planted ranks, and the declared gauge census against the
//! planted toys' hand-derived fibres: the census is a lower bound and says so.

use super::*;
use crate::gauge::{HiddenUnits, LinearPassthrough, RotaryQueryKey};
use crate::test_support::planted_toys::{
    MlpToy, RANDOM_NULL_SEED, ROUTING_HEADS, RoutingToy, cross_edge, dyadic, hadamard_modules, hadamard_modules_truth, paired_copy,
    paired_copy_truth, product_band, random_mlp, uniform_inputs,
};
use crate::test_support::test_governor;
use gam_math::gaussian_activation::GaussianActivation;
use ndarray::Axis;
use rand::SeedableRng;
use rand::rngs::StdRng;

fn exact_value(status: &EvidenceStatus<(), ParameterJacobian>) -> f64 {
    match status {
        EvidenceStatus::Exact { value, .. } => *value,
        other => panic!("expected an exact nullity, got {other:?}"),
    }
}

/// The oracle's certified side and the hand-derived side of a fibre meet: `J T` vanishes
/// within the Jacobian's formation bound and the product's rounding, and the tangents
/// resolve `expected` independent directions, so the fibre has at least that dimension by
/// algebra, and at most `nullity_at_most` by the oracle.
fn assert_tangents_span_the_kernel(jacobian: &Array2<f64>, formation: f64, tangents: &Array2<f64>, fibre: &ParameterFibre) {
    let moved = jacobian.dot(tangents);
    let tangent_norm = tangents.iter().map(|value| value * value).sum::<f64>().sqrt();
    let moved_norm = moved.iter().map(|value| value * value).sum::<f64>().sqrt();
    let reach = formation * tangent_norm + product_band(jacobian, tangents);
    assert!(moved_norm <= reach, "‖J T‖ = {moved_norm:e} beyond {reach:e}");
    let independent = parameter_fibre(test_governor(), &tangents.t().to_owned(), 0.0).expect("tangent rank");
    assert_eq!(independent.resolved_rank, tangents.ncols(), "the hand-derived tangents are independent");
    assert_eq!(fibre.nullity_at_most(), tangents.ncols(), "the oracle's bound meets the hand-derived fibre");
}

/// A product of dyadic factors is formed exactly, so its rank is the planted one: the
/// nullity is certified from above, and exact when the rank reaches the ceiling. A
/// formation bound larger than the smallest singular value hides it, and the oracle then
/// reports a larger nullity, never a smaller one.
#[test]
fn a_planted_rank_bounds_the_nullity_from_above() {
    let mut rng = StdRng::seed_from_u64(1);
    let left = dyadic(&mut rng, 30, 5, 8, 8.0);
    let right = dyadic(&mut rng, 5, 12, 8, 8.0);
    let jacobian = left.dot(&right);
    let fibre = parameter_fibre(test_governor(), &jacobian, 0.0).expect("fibre");
    assert_eq!(fibre.resolved_rank, 5);
    assert_eq!(fibre.nullity_at_most(), 7);
    assert_eq!(fibre.nullity_at_least(), 0);
    assert!(matches!(fibre.nullity().expect("status"), EvidenceStatus::Unresolved { .. }));
    let (smallest, largest) = fibre.gap();
    assert!(smallest.expect("a resolved value") > fibre.band && largest.expect("an unresolved value") <= fibre.band);
    assert_eq!(fibre.undeclared_at_most(3), 4);

    let full = parameter_fibre(test_governor(), &left, 0.0).expect("fibre");
    assert_eq!(exact_value(&full.nullity().expect("status")), 0.0);

    let smallest = smallest.expect("a resolved value");
    let blurred = parameter_fibre(test_governor(), &jacobian, smallest).expect("fibre");
    assert!(blurred.resolved_rank < 5 && blurred.nullity_at_most() > 7);
}

/// Toy 1, the paired copy: the declared GELU unit family is permutations only, so the
/// census charges no continuous coordinate, yet the function-level fibre is
/// `{[M; −M], [N, −N], NM = I, [β; −β], −Nβ}` of dimension `d² + d = 72`. The oracle's
/// upper bound meets the hand-derived tangents exactly, so the census undercounts by 72,
/// and it reports its count as `real_coordinates_at_most`, a bound, not the fibre.
#[test]
fn toy1_paired_copy_fibre_exceeds_the_declared_census() {
    let width = 8;
    let block = paired_copy(width);
    let truth = paired_copy_truth(width);
    let inputs = uniform_inputs(2, 120, width, 3.0);
    let (jacobian, formation) = block.jacobian(&inputs);
    let fibre = parameter_fibre(test_governor(), &jacobian, formation).expect("fibre");
    assert_eq!(fibre.nullity_at_most(), truth.fibre_dimension);

    // The hand family's tangent at `M = N = I`, `β = 0`: `δW_in = [X; −X]`,
    // `δW_out = [−X, X]`, `δb_in = [y; −y]`, `δb_out = −y`.
    let hidden = 2 * width;
    let (bias_in, weight_out) = (hidden * width, hidden * width + hidden);
    let bias_out = weight_out + width * hidden;
    let mut tangents = Array2::<f64>::zeros((block.parameters(), width * width + width));
    for (column, (a, b)) in (0..width).flat_map(|a| (0..width).map(move |b| (a, b))).enumerate() {
        tangents[[a * width + b, column]] = 1.0;
        tangents[[(width + a) * width + b, column]] = -1.0;
        tangents[[weight_out + a * hidden + b, column]] = -1.0;
        tangents[[weight_out + a * hidden + width + b, column]] = 1.0;
    }
    for coordinate in 0..width {
        let column = width * width + coordinate;
        tangents[[bias_in + coordinate, column]] = 1.0;
        tangents[[bias_in + width + coordinate, column]] = -1.0;
        tangents[[bias_out + coordinate, column]] = -1.0;
    }
    assert_tangents_span_the_kernel(&jacobian, formation, &tangents, &fibre);

    let census = HiddenUnits::new(block.w_in.clone(), block.b_in.clone(), block.w_out.clone(), GaussianActivation::ExactGelu)
        .expect("units")
        .family();
    let declared = census.orbit_dimension.resolved + census.null_coordinates;
    assert_eq!(declared, truth.declared_continuous);
    assert_eq!(fibre.undeclared_at_most(declared), truth.fibre_dimension);
    // The census charges nothing against the unit tensors (`b_out` is outside the family).
    assert_eq!(census.real_coordinates_at_most(), census.parameter_coordinates);
}

fn assert_no_continuous_fibre(block: &MlpToy, inputs: &Array2<f64>, context: &str) {
    let (jacobian, formation) = block.jacobian(inputs);
    let fibre = parameter_fibre(test_governor(), &jacobian, formation).expect("fibre");
    assert_eq!(exact_value(&fibre.nullity().expect("status")), 0.0, "{context}");
    let census = HiddenUnits::new(block.w_in.clone(), block.b_in.clone(), block.w_out.clone(), GaussianActivation::ExactGelu)
        .expect("units")
        .family();
    assert_eq!(census.orbit_dimension.resolved + census.null_coordinates, 0, "{context}");
}

/// Toys 2, 3 and 7: planted modules, the same with a linear cross edge (its skip counted as
/// parameters), and a random block. Generic GELU units carry only permutations, and the
/// oracle certifies an exact nullity of zero: the census is the whole fibre here.
#[test]
fn toys_two_three_and_seven_have_no_continuous_fibre() {
    let (mut block, modules) = hadamard_modules(2951);
    assert_eq!(hadamard_modules_truth(&modules).fibre_dimension, 0);
    let inputs = uniform_inputs(3, 150, 8, 3.0);
    assert_no_continuous_fibre(&block, &inputs, "planted modules");
    for epsilon in [1e-6, 1e-3, 1e-1] {
        block.skip = Some(cross_edge(epsilon));
        assert_no_continuous_fibre(&block, &inputs, &format!("cross edge ε = {epsilon}"));
    }
    let random = random_mlp(RANDOM_NULL_SEED, 64, 16);
    assert_no_continuous_fibre(&random, &uniform_inputs(4, 160, 16, 2.0), "random block");
}

/// Toy 5, the routing toy: the declared families are the rotary commutant per key/value
/// head (`3 × 4 = 12`) and a `GL(2)` pass-through per head (`3 × 4 = 12`), 24 in all. The
/// fibre also holds the cross-head `GL(4)` of heads 1 and 2, whose patterns are identical:
/// `12 + 16 + 4 = 32`. The oracle's bound meets the hand-derived tangents, so the census
/// undercounts by `16 − 2·4 = 8`.
#[test]
fn toy5_routing_law_fibre_exceeds_the_declared_census() {
    let toy = RoutingToy::new(2951);
    let truth = RoutingToy::truth();
    let (jacobian, formation) = toy.jacobian(test_governor());
    let fibre = parameter_fibre(test_governor(), &jacobian, formation).expect("fibre");
    assert_eq!(fibre.nullity_at_most(), truth.fibre_dimension);
    assert_tangents_span_the_kernel(&jacobian, formation, &toy.fibre_tangents(), &fibre);

    let native = toy.native(&toy.value, &toy.output);
    let rotary = RotaryQueryKey::new(&native).expect("rotary").family().expect("family");
    let mut declared = rotary.orbit_dimension.resolved + rotary.null_coordinates;
    assert_eq!(declared, 12);
    for head in 0..ROUTING_HEADS {
        let family = LinearPassthrough::new(toy.value[head].clone(), toy.output[head].clone())
            .expect("pass-through")
            .family()
            .expect("family");
        declared += family.orbit_dimension.resolved + family.null_coordinates;
    }
    assert_eq!(declared, truth.declared_per_head);
    assert_eq!(fibre.undeclared_at_most(declared), truth.fibre_dimension - truth.declared_per_head);
    // Merging heads 1 and 2 into one pass-through of order 4 declares the missing `GL(4)`.
    let merged = LinearPassthrough::new(
        ndarray::concatenate(Axis(0), &[toy.value[0].view(), toy.value[1].view()]).expect("stack"),
        ndarray::concatenate(Axis(1), &[toy.output[0].view(), toy.output[1].view()]).expect("stack"),
    )
    .expect("merged pass-through")
    .family()
    .expect("family");
    let third = LinearPassthrough::new(toy.value[2].clone(), toy.output[2].clone()).expect("pass-through").family().expect("family");
    let by_law = rotary.orbit_dimension.resolved + merged.orbit_dimension.resolved + third.orbit_dimension.resolved;
    assert_eq!(fibre.undeclared_at_most(by_law), 0);
}
