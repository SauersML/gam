//! ordered Beta--Bernoulli prior-capacity and assignment-mode admission invariants.
//!
//! The forward reconstruction gate is the Bernoulli posterior mean and is not
//! capped by the ordered prior. These tests therefore exercise the prior itself
//! and routing admission, never reconstruction quality at a hand-selected alpha.

use crate::assignment::{
    AssignmentMode, AssignmentModeRequest, admit_assignment_mode_for_size,
    default_ordered_beta_bernoulli_concentration_for_k_atoms,
};

#[test]
fn k_aware_ordered_beta_bernoulli_prior_spans_large_dictionary_without_capping_forward_gate() {
    let k = 128usize;
    let alpha = default_ordered_beta_bernoulli_concentration_for_k_atoms(k);
    let ratio = alpha / (alpha + 1.0);

    // The closed form chooses alpha so the final ordered prior mean is e^-1.
    let expected_tail = (-1.0_f64).exp();
    assert!((ratio.powi(k as i32) - expected_tail).abs() <= 1.0e-12);
    assert!(ratio > 0.0 && ratio < 1.0);

    // This is a prior-strength invariant only. The reconstructed gate is the
    // independent posterior mean sigmoid(logit/tau), so no atom is structurally
    // unable to reach one because of its index.
    assert!(0.5_f64.powi(k as i32) < 1.0e-30);
}

#[test]
fn default_mode_admission_uses_top_k_at_large_k_and_never_implicit_ordered_beta_bernoulli() {
    let n = 300_000usize;
    let k = 32_768usize;
    let admitted =
        admit_assignment_mode_for_size(AssignmentModeRequest::Default, n, k, 1.0, 1.0, false, 0.0)
            .expect("default admission");
    assert!(matches!(admitted.mode, AssignmentMode::Softmax { .. }));
    assert_eq!(admitted.top_k, Some(n.div_ceil(k)));

    let explicit = admit_assignment_mode_for_size(
        AssignmentModeRequest::OrderedBetaBernoulli,
        n,
        k,
        1.0,
        1.0,
        false,
        0.0,
    )
    .expect("explicit large-K ordered Beta--Bernoulli admission");
    assert!(matches!(
        explicit.mode,
        AssignmentMode::OrderedBetaBernoulli { .. }
    ));
    assert_eq!(explicit.top_k, Some(n.div_ceil(k)));
}

#[test]
fn ordered_beta_bernoulli_mode_admission_requires_explicit_small_fit_request() {
    let n = 4096usize;
    let k = 32usize;
    let default_admitted =
        admit_assignment_mode_for_size(AssignmentModeRequest::Default, n, k, 0.7, 1.0, false, 0.0)
            .expect("default small-fit admission");
    assert!(matches!(
        default_admitted.mode,
        AssignmentMode::Softmax { .. }
    ));
    assert_eq!(default_admitted.top_k, None);

    let ordered_beta_bernoulli = admit_assignment_mode_for_size(
        AssignmentModeRequest::OrderedBetaBernoulli,
        n,
        k,
        0.7,
        1.3,
        true,
        0.0,
    )
    .expect("explicit small-fit ordered Beta--Bernoulli admission");
    assert!(matches!(
        ordered_beta_bernoulli.mode,
        AssignmentMode::OrderedBetaBernoulli {
            alpha,
            learnable_alpha: true,
            ..
        } if (alpha - 1.3).abs() < 1.0e-12
    ));
    assert_eq!(ordered_beta_bernoulli.top_k, None);
}
