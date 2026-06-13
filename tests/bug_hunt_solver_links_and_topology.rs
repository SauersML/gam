use gam::mixture_link::{
    inverse_link_jet_for_family, state_from_beta_logisticspec, state_from_sasspec, state_fromspec,
};
use gam::topology_selector::{
    AutoTopologyKind, TopologyAutoFitEvidence, TopologyAutoSelector, select_topology_with_fit,
};
use gam::types::{
    InverseLink, LikelihoodSpec, LinkComponent, MixtureLinkSpec, ResponseFamily, SasLinkSpec,
    StandardLink,
};
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
    let bound = 12.0_f64;
    let expected_delta = (bound * (spec.initial_log_delta / bound).tanh()).exp();

    assert!(
        (state.delta - expected_delta).abs() <= 1e-12,
        "SAS state delta should use exp(B * tanh(log_delta / B)) so raw log_delta is bounded"
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
    let bound = 12.0_f64;
    let expected_delta = (bound * (spec.initial_log_delta / bound).tanh()).exp();

    assert!(
        (state.delta - expected_delta).abs() <= 1e-12,
        "Beta-logistic state delta should use the bounded SAS transform rather than raw exp(log_delta)"
    );
}

#[test]
fn inverse_link_jet_for_family_uses_parameterized_state_for_mixture_and_sas_links() {
    let mixture_state = state_fromspec(&MixtureLinkSpec {
        components: vec![LinkComponent::Logit, LinkComponent::Probit],
        initial_rho: array![5.0],
    })
    .expect("Mixture spec should build state");

    let sas_state = state_from_sasspec(SasLinkSpec {
        initial_epsilon: 1.7,
        initial_log_delta: -0.4,
    })
    .expect("SAS spec should build state");

    let eta = 0.8;
    let mix_spec = LikelihoodSpec::new(
        ResponseFamily::Binomial,
        InverseLink::Mixture(mixture_state),
    );
    let sas_spec = LikelihoodSpec::new(ResponseFamily::Binomial, InverseLink::Sas(sas_state));
    let logit_spec = LikelihoodSpec::new(
        ResponseFamily::Binomial,
        InverseLink::Standard(StandardLink::Logit),
    );

    let mix_jet = inverse_link_jet_for_family(&mix_spec, eta)
        .expect("Mixture inverse-link jet should evaluate");
    let sas_jet =
        inverse_link_jet_for_family(&sas_spec, eta).expect("SAS inverse-link jet should evaluate");
    let logit_jet = inverse_link_jet_for_family(&logit_spec, eta)
        .expect("Logit inverse-link jet should evaluate");

    assert!(
        (mix_jet.mu - logit_jet.mu).abs() > 1e-8,
        "Mixture inverse-link evaluation should come from its MixtureLinkState and not silently fall back to a default link"
    );
    assert!(
        (sas_jet.mu - logit_jet.mu).abs() > 1e-8,
        "SAS inverse-link evaluation should come from its SasLinkState and not silently fall back to a default link"
    );
}

#[test]
fn topology_selector_picks_lowest_cost_and_returns_fit_metadata() {
    // `tk_score` is a minimised REML / TK cost (lower is better; see issue
    // #396 and `solver::evidence`). Circle (raw_reml 10) must beat Sphere
    // (raw_reml 20). The pre-fix descending sort returned the worst topology.
    let selector = TopologyAutoSelector::new(Some(vec![
        AutoTopologyKind::Circle,
        AutoTopologyKind::Sphere,
    ]));
    let out = select_topology_with_fit(&selector, |kind| {
        Ok::<_, String>(match kind {
            AutoTopologyKind::Circle => TopologyAutoFitEvidence {
                topology_name: "circle".to_string(),
                raw_reml: 10.0,
                null_dim: 0.0,
                null_space_logdet: None,
                effective_dim: 2.0,
                n_obs: 100,
                fit_handle: 7_i32,
            },
            AutoTopologyKind::Sphere => TopologyAutoFitEvidence {
                topology_name: "sphere".to_string(),
                raw_reml: 20.0,
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
fn topology_selector_parallel_matches_sequential_winner() {
    // #1017 Phase 0: the driver-level parallel topology race must return the
    // bit-identical winner to the sequential loop (results come back in input
    // order, ranked through the same deterministic priority selector).
    use gam::topology_selector::select_topology_with_fit_parallel;
    let selector = TopologyAutoSelector::new(Some(vec![
        AutoTopologyKind::Circle,
        AutoTopologyKind::Sphere,
        AutoTopologyKind::Torus,
    ]));
    let fit_one = |kind: AutoTopologyKind| {
        Ok::<_, String>(match kind {
            AutoTopologyKind::Circle => TopologyAutoFitEvidence {
                topology_name: "circle".to_string(),
                raw_reml: 10.0,
                null_dim: 0.0,
                null_space_logdet: None,
                effective_dim: 2.0,
                n_obs: 100,
                fit_handle: 7_i32,
            },
            AutoTopologyKind::Sphere => TopologyAutoFitEvidence {
                topology_name: "sphere".to_string(),
                raw_reml: 20.0,
                null_dim: 0.0,
                null_space_logdet: None,
                effective_dim: 2.0,
                n_obs: 100,
                fit_handle: 9_i32,
            },
            AutoTopologyKind::Torus => TopologyAutoFitEvidence {
                topology_name: "torus".to_string(),
                raw_reml: 15.0,
                null_dim: 0.0,
                null_space_logdet: None,
                effective_dim: 2.0,
                n_obs: 100,
                fit_handle: 11_i32,
            },
            _ => unreachable!(),
        })
    };
    let parallel = select_topology_with_fit_parallel(&selector, &fit_one)
        .expect("parallel topology selection should succeed");
    let sequential = select_topology_with_fit(&selector, fit_one)
        .expect("sequential topology selection should succeed");
    let pw = parallel.winner().expect("parallel winner");
    let sw = sequential.winner().expect("sequential winner");
    assert_eq!(
        pw.topology_name, sw.topology_name,
        "parallel and sequential topology races must agree on the winner"
    );
    assert_eq!(pw.topology_name, "circle", "circle (raw_reml 10) is best");
    assert_eq!(
        pw.fit_handle, sw.fit_handle,
        "parallel race must preserve the winning fit metadata"
    );
}

#[test]
fn topology_selector_breaks_exact_ties_deterministically_by_candidate_order() {
    let selector = TopologyAutoSelector::new(Some(vec![
        AutoTopologyKind::Torus,
        AutoTopologyKind::Cylinder,
    ]));
    let out = select_topology_with_fit(&selector, |kind| {
        Ok::<_, String>(TopologyAutoFitEvidence {
            topology_name: kind.as_str().to_string(),
            raw_reml: 4.0,
            null_dim: 0.0,
            null_space_logdet: None,
            effective_dim: 2.0,
            n_obs: 50,
            fit_handle: kind.as_str().to_string(),
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
