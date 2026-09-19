//! Blind re-derivation oracles for #2951 results that no owner test pins, run against the landed APIs (mpd-verify).
//!
//! - P7 on a tied mask domain: the union theorem needs only that clamping intersects the admissible set, which the
//!   all-on mask keeps nonempty, and never a product domain.
//! - P15 for the reverse divergence, which the oscillation bound also claims and which can exceed the forward one.
//! - P17 for exact-constraint conditioning: null-space elimination equals the Schur formula, and re-expressing the
//!   constraint as `TAβ = Ty` keeps the posterior while `−log p` moves by `log|det T|`.

use std::f64::consts::PI;

use faer::Side;
use gam_linalg::faer_ndarray::{FaerCholesky, FaerEigh};
use gam_linalg::matrix::symmetrize_in_place;
use gam_linalg::roundoff::{accumulation_band, accumulation_growth};
use gam_math::categorical::categorical_kl_from_logits_with_error;
use gam_solve::gaussian_marginal::condition_on_exact_constraint;
use ndarray::{Array1, Array2, ArrayView1, array};

use super::bounds::softmax_kl_oscillation_bound;
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

#[test]
fn the_oscillation_bound_covers_the_reverse_divergence_where_it_exceeds_the_forward_one_2951() {
    // z = (c, 0) and z′ = (0, 0), with gap δ = z′ − z and osc(δ) = c, so P15 bounds both directions by c²/8. Under
    // p′ = (½, ½) the reverse divergence is log E_{p′} e^{−δ} − E_{p′}(−δ) = log((e^c + 1)/2) − c/2 = log cosh(c/2).
    for c in [1e-3, 0.5, 3.0, 5.0] {
        let logits = [c, 0.0];
        let perturbed = [0.0, 0.0];
        let upper = softmax_kl_oscillation_bound(ArrayView1::from(&logits[..]), ArrayView1::from(&perturbed[..]))
            .expect("finite logits")
            .upper_bound()
            .expect("a uniform bound has an upper side");
        let (forward, forward_error) =
            categorical_kl_from_logits_with_error(&logits, &perturbed).expect("finite logits");
        let (reverse, reverse_error) =
            categorical_kl_from_logits_with_error(&perturbed, &logits).expect("finite logits");
        assert!(forward - forward_error <= upper, "c = {c}: forward KL {forward} above bound {upper}");
        assert!(reverse - reverse_error <= upper, "c = {c}: reverse KL {reverse} above bound {upper}");
        // The argument order names the reverse direction: the evaluation matches the closed form. `c/2` is exact, and
        // cosh and ln each round within one unit in the last place, which moves the logarithm by at most 2u + u·|log|.
        let closed_form = (c / 2.0).cosh().ln();
        assert!(
            (reverse - closed_form).abs() <= reverse_error + accumulation_band(4, 1.0 + closed_form),
            "c = {c}: reverse KL {reverse}, log cosh(c/2) = {closed_form}"
        );
        if c < 1.0 {
            // Tight to second order: log cosh a ≥ a²/2 − a⁴/12, so the exact reverse divergence is at least
            // (c²/8)(1 − c²/24). The gaps (−c, 0) are exact, so the computed bound exceeds c²/8 by the band γ₃·3c on the
            // oscillation and two `next_up`s, below γ₂₄ relatively; forming the floor rounds five more times.
            let floor = upper * (1.0 - c * c / 24.0) / (1.0 + accumulation_growth(24));
            assert!(
                reverse + reverse_error + accumulation_growth(8) * upper >= floor,
                "c = {c}: reverse KL {reverse}, floor {floor}"
            );
        } else {
            // Positive control: the reverse divergence alone breaks a halved constant here, so a forward-only test
            // would not pin the reverse claim.
            let halved = c * c / 16.0;
            assert!(reverse - reverse_error > halved, "c = {c}: reverse KL {reverse}, halved bound {halved}");
            assert!(forward + forward_error < halved, "c = {c}: forward KL {forward}, halved bound {halved}");
        }
    }
}

fn euclidean(vector: &Array1<f64>) -> f64 {
    vector.dot(vector).sqrt()
}

fn frobenius(matrix: &Array2<f64>) -> f64 {
    matrix.iter().map(|value| value * value).sum::<f64>().sqrt()
}

/// The smallest and largest eigenvalue of a symmetric fixture matrix.
fn spectrum(matrix: &Array2<f64>) -> (f64, f64) {
    let values = matrix.eigh(Side::Lower).expect("fixture matrices are symmetric").0;
    (
        values.iter().copied().fold(f64::INFINITY, f64::min),
        values.iter().copied().fold(f64::NEG_INFINITY, f64::max),
    )
}

/// `‖ΔM‖₂` bound of a strict Cholesky solve against an SPD `dim × dim` matrix with largest eigenvalue `largest`: Higham,
/// ASNA 2nd ed., Thm 10.4, `|ΔM| ≤ γ_{3·dim+1}·|L||Lᵀ|`, with `‖|L||Lᵀ|‖₂ ≤ ‖L‖_F² = tr M ≤ dim·‖M‖₂`.
fn cholesky_backward_band(dim: usize, largest: f64) -> f64 {
    dim as f64 * accumulation_growth(3 * dim + 1) * largest
}

/// First-order rounding bands of [`condition_on_exact_constraint`] and of its covariance applied to `I`.
struct ConditioningBands {
    quadratic: f64,
    log_det: f64,
    mean: f64,
    covariance: f64,
}

/// Bands for the production route: `Ẑ = Q⁻¹Aᵀ` by a certified solve, `V̂ = sym(AẐ)`, `ŵ = V̂⁻¹y`, `q = yᵀŵ`, `log|V̂|`
/// off the factor, `mean = Ẑŵ`, and `Cov·I = Q⁻¹I − Ẑ·V̂⁻¹ẐᵀI`. Accumulation bands are Higham ASNA Lemma 3.1
/// (`γ_k·Σ|terms|`). A solve against `M` perturbed by `‖ΔM‖₂` moves its solution by `‖M⁻¹‖₂·‖ΔM‖₂·‖x‖`, a quadratic
/// form `yᵀM⁻¹y` by `‖w‖²·‖ΔM‖₂`, and `log|M|` by `|tr(M⁻¹ΔM)| ≤ dim·‖ΔM‖₂/λ_min`; each pivot logarithm lies within
/// `max|ln λ|`.
fn conditioning_bands(constraint: &Array2<f64>, value: &Array1<f64>, prior: &Array2<f64>) -> ConditioningBands {
    let gamma = accumulation_growth;
    let (n, p) = constraint.dim();
    let prior_spectrum = spectrum(prior);
    let prior_factor = prior.cholesky(Side::Lower).expect("fixture prior is SPD");
    let readout = prior_factor.solve_mat(&constraint.t().to_owned());
    let prior_covariance = prior_factor.solve_mat(&Array2::<f64>::eye(p));
    let prior_perturbation = cholesky_backward_band(p, prior_spectrum.1);
    let readout_error = prior_perturbation * frobenius(&readout) / prior_spectrum.0;
    let abs_readout = readout.mapv(f64::abs);
    let mut covariance = constraint.dot(&readout);
    symmetrize_in_place(&mut covariance);
    let covariance_spectrum = spectrum(&covariance);
    let assembly = 2.0 * gamma(p + 1) * frobenius(&constraint.mapv(f64::abs).dot(&abs_readout))
        + frobenius(constraint) * readout_error;
    let perturbation = assembly + cholesky_backward_band(n, covariance_spectrum.1);
    let covariance_factor = covariance.cholesky(Side::Lower).expect("fixture constraint covariance is SPD");
    let weights = covariance_factor.solvevec(value);
    let weights_norm = euclidean(&weights);
    let weights_error = perturbation * weights_norm / covariance_spectrum.0;
    // With B = I: ẐᵀB rounds by γ_p·‖|Ẑ|‖_F and carries ‖ΔẐ‖_F·‖I‖₂ = readout_error.
    let correction = covariance_factor.solve_mat(&readout.t().to_owned());
    let correction_norm = frobenius(&correction);
    let correction_error =
        (perturbation * correction_norm + gamma(p) * frobenius(&abs_readout) + readout_error) / covariance_spectrum.0;
    let product_terms = frobenius(&abs_readout.dot(&correction.mapv(f64::abs)));
    ConditioningBands {
        quadratic: weights_norm * weights_norm * perturbation
            + gamma(n) * value.mapv(f64::abs).dot(&weights.mapv(f64::abs)),
        log_det: n as f64 * perturbation / covariance_spectrum.0
            + gamma(2 * n) * n as f64 * covariance_spectrum.0.ln().abs().max(covariance_spectrum.1.ln().abs()),
        mean: readout_error * weights_norm
            + frobenius(&readout) * weights_error
            + gamma(n) * euclidean(&abs_readout.dot(&weights.mapv(f64::abs))),
        covariance: prior_perturbation * frobenius(&prior_covariance) / prior_spectrum.0
            + readout_error * correction_norm
            + frobenius(&readout) * correction_error
            + gamma(n) * product_terms
            + gamma(1) * (frobenius(&prior_covariance) + product_terms),
    }
}

/// A dyadic SPD prior `Q` (Gershgorin: every row's diagonal exceeds its off-diagonal sum) and a dyadic full-rank
/// constraint `A`, with `y = Aβ_p` at `β_p = (1, ½, −½)`. Every product and partial sum of these entries is a dyadic
/// with few bits, so `Aβ_p`, `TA` and `Ty` below are exact in f64.
fn conditioning_fixture() -> (Array2<f64>, Array2<f64>, Array1<f64>) {
    let prior = array![[2.0, 0.5, -0.25], [0.5, 1.5, 0.25], [-0.25, 0.25, 1.0]];
    let constraint = array![[1.0, 0.5, 0.25], [0.5, -0.5, 0.75]];
    let value = array![1.125, -0.125];
    (prior, constraint, value)
}

#[test]
fn exact_constraint_conditioning_equals_null_space_elimination_2951() {
    // P17: with N spanning null(A), every β with Aβ = y is β_p + Nt. The prior energy βᵀQβ restricted to that line has
    // curvature NᵀQN, so the conditional mean is β_p − N·(NᵀQβ_p)/(NᵀQN) and the conditional covariance N(NᵀQN)⁻¹Nᵀ.
    // The Schur route Q⁻¹AᵀV⁻¹y and Q⁻¹ − Q⁻¹AᵀV⁻¹AQ⁻¹ must equal both. The owner test pins A·Cov = 0, which leaves the
    // variance along N free.
    let (prior, constraint, value) = conditioning_fixture();
    let particular = array![1.0, 0.5, -0.5];
    assert_eq!(constraint.dot(&particular), value);
    // N = a₁ × a₂.
    let null = array![0.5, -0.625, -0.75];
    assert_eq!(constraint.dot(&null), Array1::<f64>::zeros(2));
    // NᵀQN = 1.7578125 and NᵀQβ_p = 0.953125 are exact dyadic sums; only the quotient and what follows round.
    let curvature = null.dot(&prior.dot(&null));
    let slope = null.dot(&prior.dot(&particular));
    assert_eq!((curvature, slope), (1.7578125, 0.953125));
    let step = slope / curvature;
    let eliminated_mean = &particular - &(&null * step);
    // The quotient, the scaling and the subtraction each round once.
    let eliminated_mean_band = accumulation_growth(3) * (euclidean(&particular) + step.abs() * euclidean(&null));
    let outer = Array2::from_shape_fn((3, 3), |(row, col)| null[row] * null[col]);
    let eliminated_covariance = &outer / curvature;
    let eliminated_covariance_band = accumulation_growth(1) * frobenius(&outer) / curvature;

    let posterior =
        condition_on_exact_constraint(constraint.view(), value.view(), prior.view()).expect("full-rank constraint");
    let bands = conditioning_bands(&constraint, &value, &prior);
    let mean_band = bands.mean + eliminated_mean_band;
    let mean_difference = euclidean(&(&posterior.mean().to_owned() - &eliminated_mean));
    assert!(mean_difference <= mean_band, "Schur mean differs from elimination by {mean_difference:e}, band {mean_band:e}");
    let covariance = posterior.covariance_times(&Array2::<f64>::eye(3)).expect("covariance columns");
    let covariance_band = bands.covariance + eliminated_covariance_band;
    let covariance_difference = frobenius(&(&covariance - &eliminated_covariance));
    assert!(
        covariance_difference <= covariance_band,
        "Schur covariance differs from elimination by {covariance_difference:e}, band {covariance_band:e}"
    );

    // Positive controls: the particular solution is not the conditional mean, and the prior covariance is not the
    // conditional covariance.
    assert!(euclidean(&(&posterior.mean().to_owned() - &particular)) > mean_band);
    let prior_covariance = prior.cholesky(Side::Lower).expect("fixture prior is SPD").solve_mat(&Array2::<f64>::eye(3));
    assert!(frobenius(&(&covariance - &prior_covariance)) > covariance_band);
}

#[test]
fn re_expressing_an_exact_constraint_keeps_the_posterior_and_moves_the_evidence_by_log_det_t_2951() {
    // P17: for invertible T, TAβ = Ty is the same event as Aβ = y, so the posterior is unchanged. The density of TAβ at Ty
    // is the density of Aβ at y divided by |det T|: V_T = TVTᵀ, y_Tᵀ V_T⁻¹ y_T = yᵀV⁻¹y and log|V_T| = log|V| + 2·log|det T|.
    // The owner tests pin a coefficient reparametrization, which moves no evidence.
    let (prior, constraint, value) = conditioning_fixture();
    let original =
        condition_on_exact_constraint(constraint.view(), value.view(), prior.view()).expect("full-rank constraint");
    let original_bands = conditioning_bands(&constraint, &value, &prior);
    let original_evidence = original.evidence();
    // A mixing transform with det 2, and a row swap with |det| = 1 as the invariance control.
    let mixing = array![[1.0, 1.0], [-1.0, 1.0]];
    let swap = array![[0.0, 1.0], [1.0, 0.0]];
    for (transform, log_abs_det) in [(mixing, 2.0_f64.ln()), (swap, 0.0)] {
        let transformed_constraint = transform.dot(&constraint);
        let transformed_value = transform.dot(&value);
        let transformed =
            condition_on_exact_constraint(transformed_constraint.view(), transformed_value.view(), prior.view())
                .expect("full-rank constraint");
        let bands = conditioning_bands(&transformed_constraint, &transformed_value, &prior);
        let evidence = transformed.evidence();

        let quadratic_band = original_bands.quadratic + bands.quadratic;
        let quadratic_difference = (evidence.quadratic - original_evidence.quadratic).abs();
        assert!(
            quadratic_difference <= quadratic_band,
            "log|det T| = {log_abs_det}: yᵀV⁻¹y moved by {quadratic_difference:e}, band {quadratic_band:e}"
        );
        // The test's own shift, its subtraction from the expected value and ln 2 each round once.
        let shift = evidence.log_det - original_evidence.log_det;
        let shift_band = original_bands.log_det
            + bands.log_det
            + accumulation_growth(3) * (evidence.log_det.abs() + original_evidence.log_det.abs() + 2.0 * log_abs_det);
        assert!(
            (shift - 2.0 * log_abs_det).abs() <= shift_band,
            "log|det T| = {log_abs_det}: log|V| moved by {shift:e}, band {shift_band:e}"
        );
        // log p = −½(q + log|V| + n·log 2π) rounds four times per evaluation, and the comparison twice more.
        let log_evidence_band = 0.5 * (quadratic_band + shift_band)
            + accumulation_growth(6)
                * (evidence.quadratic.abs()
                    + evidence.log_det.abs()
                    + original_evidence.quadratic.abs()
                    + original_evidence.log_det.abs()
                    + 4.0 * (2.0 * PI).ln()
                    + log_abs_det);
        let log_evidence_shift = evidence.log_evidence() - original_evidence.log_evidence();
        assert!(
            (log_evidence_shift + log_abs_det).abs() <= log_evidence_band,
            "log|det T| = {log_abs_det}: log p moved by {log_evidence_shift:e}, band {log_evidence_band:e}"
        );
        let mean_band = original_bands.mean + bands.mean;
        let mean_difference = euclidean(&(&transformed.mean().to_owned() - &original.mean().to_owned()));
        assert!(
            mean_difference <= mean_band,
            "log|det T| = {log_abs_det}: the posterior mean moved by {mean_difference:e}, band {mean_band:e}"
        );
        if log_abs_det > 0.0 {
            // Positive control: the shift is resolved, so an evidence reported as invariant under re-expression fails.
            assert!(shift.abs() > shift_band, "log|V| moved by {shift:e}, inside the band {shift_band:e}");
        }
    }
}
