#![cfg(test)]
//! Observability and the linear quotient on the planted toys of #2951: the Krylov closure
//! of a rotation block with a repeated angle, routing-law letters against per-head letters
//! under the value/output gauges, and a data Gramian that cannot see a direction.

use crate::parameter_decomposition::state::{
    LinearStateQuotient, ObservabilityLetter, ObservabilityStep, WeightedObservability, resolve_stacked_factor,
};
use crate::parameter_decomposition::supports::EvidenceStatus;
use crate::parameter_decomposition::test_support::planted_toys::{ROUTING_WIDTH, RoutingToy, hadamard};
use crate::parameter_decomposition::test_support::{plant, test_governor};
use ndarray::{Array2, Axis, array, concatenate, s};
use rand::rngs::StdRng;
use rand::{RngExt, SeedableRng};

/// Toy 4: a rotation `R` with angles `0.3, 0.3, 1.1` in a hidden orthonormal basis, a
/// generic readout `c` and the transition `R − I`. The closure of `c` is its Krylov space,
/// of dimension 4: one direction pair per distinct eigenvalue pair `e^{±1.1i} − 1`,
/// `e^{±0.3i} − 1`. It holds the whole 1.1 plane and only a 2-dimensional slice of the
/// repeated 0.3 space.
///
/// The chart is a computed invariant subspace, off an exact one by at most its measured
/// section, readout and quotient defects, and the planted basis carries its own defect.
/// With those as the formation of the stacked rows, the 1.1 plane is not separated from
/// the chart (the stack resolves rank 4), and the 0.3 space is certified to leave it by
/// two dimensions (the stack resolves rank 6).
#[test]
fn toy4_rotation_closure_is_the_krylov_space() {
    let planted = plant(6, &[0.3, 0.3, 1.1], 0, 0x2951_0004);
    let transition = &planted.matrix - &Array2::<f64>::eye(6);
    let readout = array![[0.7, -0.3, 1.1, 0.4, -0.9, 0.2]];
    let quotient = LinearStateQuotient::close(test_governor(), &[readout.view()], &[transition.view()]).expect("closure");
    assert_eq!(quotient.chart.nrows(), 4);
    let defect = quotient.section_bounds.upper
        + quotient.readout_bounds[0].upper
        + quotient.quotient_bounds[0].upper
        + planted.matrix_defect
        + planted.basis_defect;
    let stack = |rows: Array2<f64>| {
        let stacked = concatenate(Axis(0), &[quotient.chart.view(), rows.view()]).expect("stack");
        resolve_stacked_factor(test_governor(), &stacked, defect).expect("rank")
    };
    let plane = planted.basis.slice(s![.., 4..6]).t().to_owned();
    let with_plane = stack(plane);
    assert_eq!(with_plane.resolved_rank, 4, "the 1.1 plane is inside the chart: {:?}", with_plane.singular_values);
    let repeated = planted.basis.slice(s![.., 0..4]).t().to_owned();
    let with_repeated = stack(repeated);
    assert_eq!(with_repeated.resolved_rank, 6, "only a slice of the 0.3 space: {:?}", with_repeated.singular_values);
}

fn identity() -> Array2<f64> {
    Array2::<f64>::eye(ROUTING_WIDTH)
}

fn observe(toy: &RoutingToy, letters: &[Array2<f64>]) -> WeightedObservability {
    WeightedObservability::pull_back(
        test_governor(),
        &[ObservabilityStep {
            letters: letters.iter().map(|letter| ObservabilityLetter::Linear(letter.view())).collect(),
            readouts: vec![toy.readout.view()],
        }],
    )
    .expect("pull back")
}

/// Toy 5: one attention step with the residual stream and the readout `c`. With one letter
/// per routing law, `[I, C₁ + C₂, C₃]`, the observable directions are `c`, `c(C₁ + C₂)` and
/// `c C₃`: rank 3. One letter per head, `[I, C₁, C₂, C₃]`, over-counts to 4.
///
/// Under the per-head gauge `V_h ↦ S_h V_h`, `O_h ↦ O_h S_h⁻¹` every letter of both
/// alphabets is carried exactly. Under the cross-head `GL(4)` of heads 1 and 2, whose
/// patterns are identical, the law letters are still carried exactly, so the law Gramian
/// does not move, while the per-head letters `C₁`, `C₂` move and the per-head observable
/// space moves by a certified angle: per-head observability is not gauge-invariant.
#[test]
fn toy5_routing_law_letters_are_gauge_invariant_and_head_letters_are_not() {
    let toy = RoutingToy::new(2951);
    let law_letters = |value: &[Array2<f64>], output: &[Array2<f64>]| {
        vec![identity(), RoutingToy::transport(value, output, &[0, 1]), RoutingToy::transport(value, output, &[2])]
    };
    let head_letters = |value: &[Array2<f64>], output: &[Array2<f64>]| {
        let mut letters = vec![identity()];
        letters.extend((0..3).map(|head| RoutingToy::transport(value, output, &[head])));
        letters
    };
    let by_law = observe(&toy, &law_letters(&toy.value, &toy.output));
    let by_head = observe(&toy, &head_letters(&toy.value, &toy.output));
    assert_eq!(by_law.spectrum().resolved_rank, 3);
    assert!(matches!(by_law.spectrum().rank_evidence().expect("rank"), EvidenceStatus::Exact { value, .. } if value == 3.0));
    assert_eq!(by_head.spectrum().resolved_rank, 4);

    let (value, output) = toy.per_head_gauge(5);
    assert_eq!(law_letters(&value, &output), law_letters(&toy.value, &toy.output));
    assert_eq!(head_letters(&value, &output), head_letters(&toy.value, &toy.output));

    let (value, output) = toy.cross_head_gauge(6);
    let moved_law = observe(&toy, &law_letters(&value, &output));
    assert_eq!(moved_law.factor, by_law.factor, "the law Gramian is carried exactly");
    let moved_head = observe(&toy, &head_letters(&value, &output));
    assert_eq!(moved_head.spectrum().resolved_rank, 4);
    let base_space = by_head.directions.slice(s![..4, ..]).to_owned();
    let capture = moved_head.capture(test_governor(), base_space.view()).expect("capture");
    let own = by_head.capture(test_governor(), base_space.view()).expect("capture");
    let smallest = capture.principal_cosines.iter().copied().fold(1.0_f64, f64::min);
    let sine = (1.0 - smallest * smallest).max(0.0).sqrt();
    assert!(
        sine > capture.angle_perturbation + own.angle_perturbation,
        "the per-head space moves by sin θ = {sine} beyond {} + {}",
        capture.angle_perturbation,
        own.angle_perturbation
    );
}

/// Toy 6: data on a 4-dimensional subspace `S` of `ℝ⁸` (rows `0..4` of a Hadamard matrix,
/// integer coefficients, so every datum is exact) and a direction `v ∈ S^⊥` (row 4). The
/// data Gramian `XᵀX` resolves rank 4, and `v` carries none of its energy: component reads
/// `a + t v` and `a` are indistinguishable on the data for every `t`, so reads are identified
/// only modulo `S^⊥`. A direction inside `S` is seen.
#[test]
fn toy6_data_gramian_cannot_see_the_orthogonal_complement() {
    let basis = hadamard();
    let span = basis.slice(s![..4, ..]).to_owned();
    let mut rng = StdRng::seed_from_u64(6);
    let coefficients = Array2::from_shape_simple_fn((256, 4), || f64::from(rng.random_range(-4_i32..=4)));
    let data = coefficients.dot(&span);
    let observed = WeightedObservability::pull_back(
        test_governor(),
        &[ObservabilityStep {
            letters: vec![ObservabilityLetter::Linear(Array2::<f64>::eye(8).view())],
            readouts: vec![data.view()],
        }],
    )
    .expect("pull back");
    assert_eq!(observed.spectrum().resolved_rank, 4);
    assert!(matches!(
        observed.spectrum().rank_evidence().expect("rank"),
        EvidenceStatus::Unresolved { lower, .. } if lower == 4.0
    ));
    let hidden = basis.slice(s![4..5, ..]).to_owned();
    assert_eq!(data.dot(&hidden.t()).iter().fold(0.0_f64, |largest, value| largest.max(value.abs())), 0.0);
    let invisible = observed.capture(test_governor(), hidden.view()).expect("capture");
    match &invisible.energy_fraction {
        EvidenceStatus::Exact { value, numerical_error, .. } => {
            assert!(value <= numerical_error, "v holds {value:e} of the data energy, beyond {numerical_error:e}");
        }
        other => panic!("expected an exact energy fraction, got {other:?}"),
    }
    let seen = observed.capture(test_governor(), basis.slice(s![0..1, ..])).expect("capture");
    assert!(seen.energy_fraction.lower_bound().expect("exact") > 0.0);
}
