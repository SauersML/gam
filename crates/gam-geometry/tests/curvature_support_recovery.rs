//! #2818 recovery: curvature point estimates retain the #2687 box provenance.

use gam_geometry::curvature_estimand::{KappaEstimateSupport, profile_ci_walk};

#[test]
fn a_monotone_criterion_rails_kappa_hat_and_the_walk_declares_it_2687() {
    for upper in [1.389_f64, 2.78, 40.0] {
        let ci = profile_ci_walk(|kappa| Ok(-kappa), upper, -1.0, -upper, upper, 0.95, 1e-8)
            .expect("a monotone profile must report its constrained optimum");
        assert_eq!(ci.kappa_hat, upper);
        assert_eq!(
            ci.kappa_hat_support,
            KappaEstimateSupport::RailedAtUpperBound
        );
        assert_eq!(ci.kappa_hat_support.label(), "railed_at_upper_bound");
        assert!(ci.hi_at_bound);
        assert_eq!(ci.ci_hi, upper);
    }
    let ci = profile_ci_walk(|kappa| Ok(kappa), -2.0, -1.0, -2.0, 2.0, 0.95, 1e-8)
        .expect("the mirrored monotone profile must report the lower rail");
    assert_eq!(
        ci.kappa_hat_support,
        KappaEstimateSupport::RailedAtLowerBound
    );
    assert_eq!(ci.kappa_hat_support.label(), "railed_at_lower_bound");
    assert!(ci.lo_at_bound);
    assert_eq!(ci.ci_lo, -2.0);
}

#[test]
fn an_interior_optimum_is_not_declared_railed_2687() {
    let optimum = -0.37_f64;
    let curvature = 16.0_f64;
    let quadratic = |kappa: f64| Ok(7.0 + 0.5 * curvature * (kappa - optimum) * (kappa - optimum));
    let ci = profile_ci_walk(quadratic, optimum, curvature, -3.0, 3.0, 0.95, 1e-8)
        .expect("the interior quadratic has a closed confidence interval");
    assert_eq!(ci.kappa_hat_support, KappaEstimateSupport::Interior);
    assert!(!ci.lo_at_bound && !ci.hi_at_bound);
    assert!(ci.ci_lo < optimum && ci.ci_hi > optimum);
    assert!((ci.ci_lo + ci.ci_hi - 2.0 * optimum).abs() < 2e-8);
    // Only the chart box moves: the profile and its optimum are unchanged.
    // Provenance must reflect that the point is now on the lower endpoint.
    let squeezed = profile_ci_walk(quadratic, optimum, curvature, optimum, 3.0, 0.95, 1e-8)
        .expect("a box touching the optimum remains a supported profile");
    assert_eq!(
        squeezed.kappa_hat_support,
        KappaEstimateSupport::RailedAtLowerBound
    );
    assert!(squeezed.lo_at_bound);
    assert_eq!(squeezed.ci_lo, optimum);
    assert!((squeezed.ci_hi - ci.ci_hi).abs() < 1e-8);
}
