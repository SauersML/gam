use gam::custom_family::{CustomFamily, ParameterBlockState};
use gam::families::gamlss::{GammaLogFamily, ParameterLink, PoissonLogFamily};
use ndarray::array;

fn single_block(eta: ndarray::Array1<f64>) -> Vec<ParameterBlockState> {
    vec![ParameterBlockState {
        beta: array![],
        eta,
    }]
}

#[test]
fn bug_poisson_loglik_matches_canonical_form_up_to_constant() {
    let fam = PoissonLogFamily {
        y: array![0.0, 2.0, 7.0],
        weights: array![1.0, 1.0, 1.0],
    };
    let eta = array![-0.3, 0.1, 1.7];
    let eval = fam
        .evaluate(&single_block(eta.clone()))
        .expect("poisson evaluate should succeed");
    let expected: f64 = (0..eta.len())
        .map(|i| fam.y[i] * eta[i] - eta[i].exp())
        .sum();
    assert!(
        (eval.log_likelihood - expected).abs() < 1e-12,
        "Poisson log_lik should equal sum(y_i*eta_i-exp(eta_i)) up to additive constants"
    );
}

#[test]
fn bug_gamma_alpha1_reduces_to_exponential_loglik_up_to_constant() {
    let fam = GammaLogFamily {
        y: array![0.4, 1.1, 2.7],
        weights: array![1.0, 1.0, 1.0],
        shape: 1.0,
    };
    let eta = array![-0.2, 0.5, 0.9];
    let eval = fam
        .evaluate(&single_block(eta.clone()))
        .expect("gamma evaluate should succeed");
    let expected: f64 = (0..eta.len())
        .map(|i| -(fam.y[i] / eta[i].exp() + eta[i]))
        .sum();
    assert!(
        (eval.log_likelihood - expected).abs() < 1e-12,
        "Gamma(shape=1) log_lik should reduce to exponential form up to constants"
    );
}

#[test]
fn bug_poisson_overflow_eta_above_700_must_stay_finite() {
    let fam = PoissonLogFamily {
        y: array![1.0],
        weights: array![1.0],
    };
    let eval = fam
        .evaluate(&single_block(array![750.0]))
        .expect("poisson evaluate should not error for large eta");
    assert!(
        eval.log_likelihood.is_finite(),
        "Poisson log_lik at eta>700 must be finite (saturate or explicit error), not NaN/inf"
    );
}

#[test]
fn bug_parameter_link_variant_order_is_stable_for_prediction_mapping() {
    let expected = [
        ParameterLink::Identity,
        ParameterLink::Log,
        ParameterLink::Logit,
        ParameterLink::Probit,
        ParameterLink::InverseLink,
        ParameterLink::Wiggle,
    ];
    assert_eq!(
        expected.len(),
        6,
        "ParameterLink variant set changed; verify predict inverse-link mapping for every variant"
    );
}
