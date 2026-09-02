use gam::mixture_link::{state_from_beta_logisticspec, state_from_sasspec, state_fromspec};
use gam::topology_selector::{
    AutoTopologyKind, TopologyAutoFitEvidence, TopologyAutoSelector, select_topology_with_fit,
};
use gam::types::{LinkComponent, MixtureLinkSpec, SasLinkSpec};
use ndarray::array;

#[test]
fn mixture_state_fromspec_uses_last_zero_logit_softmax_that_sums_to_one() {
    let spec = MixtureLinkSpec {
        components: vec![
            LinkComponent::Probit,
            LinkComponent::Logit,
            LinkComponent::CLogLog,
        ],
        initial_rho: array![2.0, -1.0],
    };
    let state =
        state_fromspec(&spec).expect("MixtureLinkSpec should build a valid MixtureLinkState");
    let exp_logits = [2.0_f64.exp(), (-1.0_f64).exp(), 0.0_f64.exp()];
    let z = exp_logits.iter().sum::<f64>();
    let expected = [exp_logits[0] / z, exp_logits[1] / z, exp_logits[2] / z];

    assert!(
        (state.pi.sum() - 1.0).abs() <= 1e-12,
        "MixtureLinkState pi should be a probability vector with sum exactly one"
    );
    for (i, expected_i) in expected.into_iter().enumerate() {
        assert!(
            (state.pi[i] - expected_i).abs() <= 1e-12,
            "MixtureLinkState pi should match softmax(rho with final fixed zero logit) for every component"
        );
    }
}

#[test]
fn sas_state_fromspec_bounds_delta_with_sas_log_delta_bound_transform() {
    let spec = SasLinkSpec {
        initial_epsilon: 0.3,
        initial_log_delta: 100.0,
    };
    let state =
        state_from_sasspec(spec).expect("SAS spec with finite parameters should be accepted");
    // `SAS_LOG_DELTA_BOUND`. The bounded latent map is the interior-exact
    // compact-support splice `smooth_bound_jet`, NOT `B*tanh(x/B)`: it is the
    // identity on `|x| <= SPLICE_INTERIOR_FRAC*B = 9.6` and saturates to exactly
    // `+/-B` for `|x| >= (2 - SPLICE_INTERIOR_FRAC)*B = 14.4`. `log_delta = 100`
    // is deep in the saturated branch, so the effective log-delta is exactly
    // `B` and `delta` is exactly `exp(B)` -- a bit-exact expectation, which is
    // strictly stronger than the old `tanh` form it replaces.
    let bound = 12.0_f64;
    assert!(
        spec.initial_log_delta.abs() >= (2.0 - 0.8) * bound,
        "fixture must sit in the saturated branch for the exact-saturation expectation to bite"
    );
    let expected_delta = bound.exp();

    assert!(
        (state.delta - expected_delta).abs() <= 1e-12,
        "SAS state delta must saturate to exp(B) at |log_delta| >= 1.2*B so raw log_delta is bounded; got {} want {}",
        state.delta,
        expected_delta
    );
}

#[test]
fn beta_logistic_state_fromspec_uses_same_bounded_delta_parameterization_as_sas() {
    let spec = SasLinkSpec {
        initial_epsilon: -0.2,
        initial_log_delta: 100.0,
    };
    let state = state_from_beta_logisticspec(spec)
        .expect("Beta-logistic spec with finite parameters should be accepted");
    // Same saturated-branch reasoning as the SAS test above: the shared bounded
    // map is the compact-support splice, exactly `+/-B` past `1.2*B`.
    let bound = 12.0_f64;
    assert!(
        spec.initial_log_delta.abs() >= (2.0 - 0.8) * bound,
        "fixture must sit in the saturated branch for the exact-saturation expectation to bite"
    );
    let expected_delta = bound.exp();

    assert!(
        (state.delta - expected_delta).abs() <= 1e-12,
        "Beta-logistic state delta must use the same bounded SAS transform and saturate to exp(B); got {} want {}",
        state.delta,
        expected_delta
    );
}

#[test]
fn topology_selector_picks_lowest_cost_and_returns_fit_metadata() {
    // `tk_score` is a minimised REML / TK cost (lower is better; see issue
    // #396 and `solver::evidence`). Circle (raw_reml 10) must beat Sphere
    // (raw_reml 20). The pre-fix descending sort returned the worst topology.
    let selector = TopologyAutoSelector::new(vec![
        AutoTopologyKind::Circle,
        AutoTopologyKind::Sphere,
    ]);
    let out = select_topology_with_fit(&selector, |kind| {
        Ok::<_, String>(match kind {
            AutoTopologyKind::Circle => TopologyAutoFitEvidence {
                topology_name: "circle".to_string(),
                raw_reml: 10.0,
                // Exact synthetic literal: zero accumulated forward error. `None` would
                // state that no bound was established (#2729), which makes the selector
                // refuse to certify any margin -- the opposite of this fixture\'s intent.
                raw_reml_roundoff: Some(0.0),
                null_dim: 0.0,
                null_space_logdet: None,
                effective_dim: 2.0,
                n_obs: 100,
                fit_handle: 7_i32,
            },
            AutoTopologyKind::Sphere => TopologyAutoFitEvidence {
                topology_name: "sphere".to_string(),
                raw_reml: 20.0,
                // Exact synthetic literal: zero accumulated forward error. `None` would
                // state that no bound was established (#2729), which makes the selector
                // refuse to certify any margin -- the opposite of this fixture\'s intent.
                raw_reml_roundoff: Some(0.0),
                null_dim: 0.0,
                null_space_logdet: None,
                effective_dim: 2.0,
                n_obs: 100,
                fit_handle: 9_i32,
            },
            _ => unreachable!(),
        })
    })
    .expect("Topology selection should succeed when at least one candidate fits");

    let winner = out
        .winner()
        .expect("Topology selection should return a winner");
    assert_eq!(
        winner.topology_name, "circle",
        "select_topology_with_fit should return the lowest-cost (best) topology"
    );
    assert_eq!(
        winner.fit_handle, 7,
        "select_topology_with_fit should preserve fit metadata for the winning candidate"
    );
    assert_eq!(
        winner.n_obs, 100,
        "select_topology_with_fit should preserve n_obs metadata for the winning candidate"
    );
}

#[test]
fn topology_selector_breaks_exact_ties_deterministically_by_candidate_order() {
    let selector = TopologyAutoSelector::new(vec![
        AutoTopologyKind::Torus,
        AutoTopologyKind::Cylinder,
    ]);
    let out = select_topology_with_fit(&selector, |kind| {
        Ok::<_, String>(TopologyAutoFitEvidence {
            topology_name: kind.display_name(),
            raw_reml: 4.0,
            // Exact synthetic literal: zero accumulated forward error. `None` would
            // state that no bound was established (#2729), which makes the selector
            // refuse to certify any margin -- the opposite of this fixture\'s intent.
            raw_reml_roundoff: Some(0.0),
            null_dim: 0.0,
            null_space_logdet: None,
            effective_dim: 2.0,
            n_obs: 50,
            fit_handle: kind.display_name(),
        })
    })
    .expect("Topology selection should succeed on tied candidates");

    let winner = out
        .winner()
        .expect("Tied topology selection should still produce a winner");
    assert_eq!(
        winner.topology_name, "torus",
        "When topology scores tie exactly, selection should be deterministic and prefer the first candidate in input order"
    );
}
