//! Blind re-derivation oracles for #2951 results that no owner test pins, run against the landed APIs (mpd-verify).
//!
//! - P7 on a tied mask domain: the union theorem needs only that clamping intersects the admissible set, which the
//!   all-on mask keeps nonempty, and never a product domain.

use super::supports::{
    ComponentSet, EvidenceStatus, EvidenceStatusError, ExactBasis, InputSupport, SeparationOracle,
    sufficient_union,
};

/// The declared mask levels.
const LEVELS: [f64; 3] = [0.0, 0.5, 1.0];

/// Exhaustive separation over the grid `{0, ½, 1}³`, restricted to the tied masks `m₀ = m₁` when `tied`. The responses
/// are dyadic polynomials in dyadic levels, so every distance is exact and the numerical error is zero.
struct GridOracle {
    tied: bool,
    response: fn(&[f64]) -> f64,
}

impl GridOracle {
    fn domain(&self) -> &'static str {
        if self.tied {
            "{0, 1/2, 1}^3 with m0 = m1"
        } else {
            "{0, 1/2, 1}^3"
        }
    }

    fn distance(&self, mask: &[f64]) -> f64 {
        ((self.response)(mask) - (self.response)(&[1.0; 3])).abs()
    }

    /// `R(S) = sup {d(m) : m admissible, m_S = 1}`, an attaining mask, and the number of masks visited. The all-on mask
    /// is admissible under every support, so the supremum is over a nonempty set.
    fn risk(&self, support: &ComponentSet) -> (f64, Vec<f64>, u64) {
        let mut best = (f64::NEG_INFINITY, vec![1.0; 3]);
        let mut visited = 0u64;
        for index in 0..27usize {
            let mask: Vec<f64> = (0..3u32).map(|slot| LEVELS[index / 3usize.pow(slot) % 3]).collect();
            let clamped = support.members().iter().all(|&component| mask[component] == 1.0);
            if !clamped || (self.tied && mask[0] != mask[1]) {
                continue;
            }
            visited += 1;
            let distance = self.distance(&mask);
            if distance > best.0 {
                best = (distance, mask);
            }
        }
        (best.0, best.1, visited)
    }
}

impl SeparationOracle for GridOracle {
    type Mask = Vec<f64>;
    type Domain = &'static str;
    type Error = EvidenceStatusError;

    fn components(&self) -> usize {
        3
    }

    fn perturbed_components(&self, mask: &Vec<f64>) -> Vec<usize> {
        (0..mask.len()).filter(|&component| mask[component] != 1.0).collect()
    }

    fn separate(
        &mut self,
        support: &ComponentSet,
    ) -> Result<EvidenceStatus<Vec<f64>, &'static str>, EvidenceStatusError> {
        let (value, witness, cardinality) = self.risk(support);
        EvidenceStatus::exact(
            value,
            0.0,
            ExactBasis::Exhaustive { cardinality },
            Some(witness),
            self.domain(),
        )
    }

    fn evaluate(
        &mut self,
        mask: &Vec<f64>,
    ) -> Result<EvidenceStatus<Vec<f64>, &'static str>, EvidenceStatusError> {
        EvidenceStatus::exact(
            self.distance(mask),
            0.0,
            ExactBasis::Exhaustive { cardinality: 1 },
            Some(mask.clone()),
            "one mask",
        )
    }
}

/// `m₁ + (1 − m₂)/16`. Clamping `m₀` pins `m₁` only through the tie.
fn first_response(mask: &[f64]) -> f64 {
    mask[1] + (1.0 - mask[2]) / 16.0
}

/// `m₂ + (1 − m₁)/16`.
fn second_response(mask: &[f64]) -> f64 {
    mask[2] + (1.0 - mask[1]) / 16.0
}

fn component_set(members: Vec<usize>) -> ComponentSet {
    ComponentSet::new(3, members).expect("members in range")
}

#[test]
fn a_union_of_per_input_supports_is_sufficient_on_a_tied_mask_domain_2951() {
    // P7: S ⊆ S′ clamps more controls, so {m admissible : m_S′ = 1} ⊆ {m admissible : m_S = 1} and R(S′) ≤ R(S) for ANY
    // admissible set containing the all-on mask. The owner test pins the union on a product grid. Here the domain ties
    // m₀ = m₁, so clamping {0} clamps m₁ as well.
    let tolerance = 0.125;
    let mut first = GridOracle { tied: true, response: first_response };
    let mut second = GridOracle { tied: true, response: second_response };
    let first_support = component_set(vec![0]);
    let second_support = component_set(vec![2]);
    // R₁({0}) = sup (1 − m₂)/16 = 1/16 and R₂({2}) = sup (1 − m₁)/16 = 1/16, both at most the tolerance.
    let first_evidence = first.separate(&first_support).expect("exhaustive separation");
    let second_evidence = second.separate(&second_support).expect("exhaustive separation");
    assert_eq!(first_evidence.upper_bound(), Some(0.0625));
    assert_eq!(second_evidence.upper_bound(), Some(0.0625));
    assert!(first_evidence.certifies_at_most(tolerance) && second_evidence.certifies_at_most(tolerance));

    // Positive control: the tie is load-bearing. On the product grid {0} leaves m₁ free, and the unique attaining mask
    // (1, 0, 1) keeps the support on at distance 1.
    let mut product = GridOracle { tied: false, response: first_response };
    let refuted = product.separate(&first_support).expect("exhaustive separation");
    assert!(refuted.refutes_at_most(tolerance));
    assert_eq!(refuted.lower_bound(), Some(1.0));
    let witness = refuted.witness().expect("an attaining mask").clone();
    assert_eq!(witness, vec![1.0, 0.0, 1.0]);
    assert_eq!(product.perturbed_components(&witness), vec![1]);
    assert_eq!(product.evaluate(&witness).expect("exact evaluation").upper_bound(), Some(1.0));

    // Neither per-input support is sufficient at the other input, so the union is needed.
    assert!(second.separate(&first_support).expect("exhaustive separation").refutes_at_most(tolerance));
    assert!(first.separate(&second_support).expect("exhaustive separation").refutes_at_most(tolerance));

    let union = sufficient_union(
        3,
        &[
            InputSupport { support: first_support, tolerance, evidence: first_evidence },
            InputSupport { support: second_support, tolerance, evidence: second_evidence },
        ],
    )
    .expect("both inputs are certified");
    assert_eq!(union.support.members(), &[0, 2]);
    let direct = [
        first.separate(&union.support).expect("exhaustive separation"),
        second.separate(&union.support).expect("exhaustive separation"),
    ];
    for (inherited, direct) in union.per_input.iter().zip(&direct) {
        assert!(matches!(inherited, EvidenceStatus::UniformBound { .. }), "got {inherited:?}");
        assert!(inherited.certifies_at_most(tolerance));
        assert!(direct.upper_bound().expect("exact") <= inherited.upper_bound().expect("a bound"));
    }

    // Monotonicity on the tied domain, exhaustively: R(S′) ≤ R(S) for every S ⊆ S′, at both inputs.
    let members = |bits: u32| (0..3usize).filter(|component| bits >> component & 1 == 1).collect::<Vec<usize>>();
    for oracle in [&first, &second] {
        for subset in 0u32..8 {
            for superset in (0u32..8).filter(|superset| superset & subset == subset) {
                let small = oracle.risk(&component_set(members(subset))).0;
                let large = oracle.risk(&component_set(members(superset))).0;
                assert!(large <= small, "R({superset:03b}) = {large} above R({subset:03b}) = {small}");
            }
        }
    }
}
