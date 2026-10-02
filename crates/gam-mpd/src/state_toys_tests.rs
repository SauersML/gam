#![cfg(test)]
//! Observability on the planted toys of #2951: routing-law letters against per-head letters
//! under the value/output gauges, and a data Gramian that cannot see a direction.

use crate::joint_operators::attention_letters;
use crate::state::{ObservabilityLetter, ObservabilityStep, WeightedObservability};
use crate::supports::EvidenceStatus;
use crate::test_support::planted_toys::{ROUTING_WIDTH, RoutingToy, data_subspace_toy};
use crate::test_support::test_governor;
use ndarray::{Array2, Axis, s};

fn identity() -> Array2<f64> {
    Array2::<f64>::eye(ROUTING_WIDTH)
}

fn observe<'a>(toy: &'a RoutingToy, letters: Vec<ObservabilityLetter<'a>>) -> WeightedObservability {
    WeightedObservability::pull_back(
        test_governor(),
        &[ObservabilityStep {
            letters,
            readouts: vec![toy.readout.view()],
        }],
    )
    .expect("pull back")
}

fn linear(letters: &[Array2<f64>]) -> Vec<ObservabilityLetter<'_>> {
    letters.iter().map(|letter| ObservabilityLetter::Linear(letter.view())).collect()
}

/// Toy 5: one attention step with the residual stream and the readout `c`. The attention
/// block enters as its routing-law letters (`joint_operators::attention_letters`), `C₁ + C₂`
/// and `C₃`, so the observable directions are `c`, `c(C₁ + C₂)` and `c C₃`: rank 3. One
/// linear letter per head, `[I, C₁, C₂, C₃]` (what the owner no longer lets an attention
/// step declare), over-counts to 4.
///
/// Under the per-head gauge `V_h ↦ S_h V_h`, `O_h ↦ O_h S_h⁻¹` every letter of both
/// alphabets is carried exactly. Under the cross-head `GL(4)` of heads 1 and 2, whose
/// patterns are identical, the law letters are still carried exactly, so the law Gramian
/// does not move, while the per-head letters `C₁`, `C₂` move and the per-head observable
/// space moves by a certified angle: per-head observability is not gauge-invariant.
#[test]
fn toy5_routing_law_letters_are_gauge_invariant_and_head_letters_are_not() {
    let toy = RoutingToy::new(2951);
    let truth = RoutingToy::truth();
    let identity = identity();
    let derived = |value: &[Array2<f64>], output: &[Array2<f64>]| {
        let letters = attention_letters(test_governor(), &toy.native(value, output), None, None).expect("letters");
        assert_eq!(letters.laws.laws, truth.laws);
        letters
    };
    let head_letters = |value: &[Array2<f64>], output: &[Array2<f64>]| {
        let mut letters = vec![identity.clone()];
        letters.extend((0..3).map(|head| RoutingToy::transport(value, output, &[head])));
        letters
    };
    let base_letters = derived(&toy.value, &toy.output);
    let by_law = observe(&toy, vec![ObservabilityLetter::Linear(identity.view()), ObservabilityLetter::RoutingLaws(&base_letters)]);
    let base_heads = head_letters(&toy.value, &toy.output);
    let by_head = observe(&toy, linear(&base_heads));
    let (per_law, per_head) = (truth.observable_rank_per_law, truth.observable_rank_per_head);
    assert_eq!(by_law.spectrum().resolved_rank, per_law);
    assert!(matches!(by_law.spectrum().rank_evidence().expect("rank"), EvidenceStatus::Exact { value, .. } if value == per_law as f64));
    assert_eq!(by_head.spectrum().resolved_rank, per_head);

    let (value, output) = toy.per_head_gauge(5);
    assert_eq!(derived(&value, &output).transports(), base_letters.transports());
    assert_eq!(head_letters(&value, &output), base_heads);

    let (value, output) = toy.cross_head_gauge(6);
    let moved_letters = derived(&value, &output);
    assert_eq!(moved_letters.transports(), base_letters.transports(), "the law letters are carried exactly");
    let moved_law = observe(&toy, vec![ObservabilityLetter::Linear(identity.view()), ObservabilityLetter::RoutingLaws(&moved_letters)]);
    assert_eq!(moved_law.factor, by_law.factor, "the law Gramian is carried exactly");
    let moved_heads = head_letters(&value, &output);
    let moved_head = observe(&toy, linear(&moved_heads));
    assert_eq!(moved_head.spectrum().resolved_rank, per_head);
    let base_space = by_head.directions.slice(s![..per_head, ..]).to_owned();
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
    let toy = data_subspace_toy(6, 256);
    let data = &toy.data;
    let identity = Array2::<f64>::eye(8);
    let observed = WeightedObservability::pull_back(
        test_governor(),
        &[ObservabilityStep {
            letters: vec![ObservabilityLetter::Linear(identity.view())],
            readouts: vec![data.view()],
        }],
    )
    .expect("pull back");
    assert_eq!(observed.spectrum().resolved_rank, toy.data_rank);
    assert!(matches!(
        observed.spectrum().rank_evidence().expect("rank"),
        EvidenceStatus::Unresolved { lower, .. } if lower == toy.data_rank as f64
    ));
    let hidden = toy.hidden.clone().insert_axis(Axis(0));
    assert_eq!(data.dot(&hidden.t()).iter().fold(0.0_f64, |largest, value| largest.max(value.abs())), 0.0);
    let invisible = observed.capture(test_governor(), hidden.view()).expect("capture");
    match &invisible.energy_fraction {
        EvidenceStatus::Exact { value, numerical_error, .. } => {
            assert!(value <= numerical_error, "v holds {value:e} of the data energy, beyond {numerical_error:e}");
        }
        other => panic!("expected an exact energy fraction, got {other:?}"),
    }
    let seen = observed.capture(test_governor(), toy.span.slice(s![0..1, ..])).expect("capture");
    assert!(seen.energy_fraction.lower_bound().expect("exact") > 0.0);
}
