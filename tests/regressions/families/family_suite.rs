use gam::families::family_runtime::{FamilyStrategy, strategy_for_spec};
use gam::families::marginal_slope_shared::{outer_row_weights_by_index, outer_weighted_rows};
use gam::families::scale_design::{
    apply_scale_deviation_transform, build_scale_deviation_transform,
};
use gam::families::survival::latent::fixed_latent_hazard_frailty;
use gam::families::survival::lognormal_kernel::{FrailtySpec, HazardLoading};
use gam::families::vector_response::{GaussianVectorLikelihood, VectorNoise, VectorResponseTarget};
use gam::outer_subsample::{OuterScoreSubsample, WeightedOuterRow};
use gam::types::inverse_link_to_binomial_spec;
use gam::types::{InverseLink, LatentCLogLogState, LikelihoodSpec, ResponseFamily, StandardLink};
use ndarray::{Array1, Array2, array};

#[test]
fn bug_family_meta_binomial_inverse_links_round_trip_identity() {
    let links = vec![
        InverseLink::Standard(StandardLink::Logit),
        InverseLink::Standard(StandardLink::Probit),
        InverseLink::Standard(StandardLink::CLogLog),
        InverseLink::LatentCLogLog(LatentCLogLogState { latent_sd: 0.4 }),
    ];
    for link in links {
        let spec = inverse_link_to_binomial_spec(&link)
            .expect("Every supported binomial inverse link must resolve to a binomial likelihood specification.");
        assert_eq!(
            spec.response,
            ResponseFamily::Binomial,
            "Each supported binomial inverse link must map to ResponseFamily::Binomial."
        );
        assert_eq!(
            spec.link, link,
            "Round-tripping through inverse_link_to_binomial_spec must preserve inverse-link identity."
        );
    }
}

#[test]
fn bug_strategy_for_spec_preserves_family_marker_for_all_response_variants() {
    let specs = vec![
        LikelihoodSpec::new(
            ResponseFamily::Gaussian,
            InverseLink::Standard(StandardLink::Identity),
        ),
        LikelihoodSpec::new(
            ResponseFamily::Binomial,
            InverseLink::Standard(StandardLink::Logit),
        ),
        LikelihoodSpec::new(
            ResponseFamily::Poisson,
            InverseLink::Standard(StandardLink::Log),
        ),
        LikelihoodSpec::new(
            ResponseFamily::Tweedie { p: 1.5 },
            InverseLink::Standard(StandardLink::Log),
        ),
        LikelihoodSpec::new(
            ResponseFamily::NegativeBinomial {
                theta: 2.0,
                theta_fixed: false,
            },
            InverseLink::Standard(StandardLink::Log),
        ),
        LikelihoodSpec::new(
            ResponseFamily::Beta { phi: 3.0 },
            InverseLink::Standard(StandardLink::Logit),
        ),
        LikelihoodSpec::new(
            ResponseFamily::Gamma,
            InverseLink::Standard(StandardLink::Log),
        ),
        LikelihoodSpec::new(
            ResponseFamily::RoystonParmar,
            InverseLink::Standard(StandardLink::Identity),
        ),
    ];
    for spec in specs {
        let strategy = strategy_for_spec(&spec);
        assert_eq!(
            strategy.family().response,
            spec.response,
            "strategy_for_spec must preserve the response-family marker for each LikelihoodSpec variant."
        );
    }
}

#[test]
fn bug_marginal_slope_outer_weighted_rows_match_documented_row_weights() {
    let rows = vec![
        WeightedOuterRow {
            index: 1,
            weight: 2.5,
            stratum: 4,
        },
        WeightedOuterRow {
            index: 3,
            weight: 5.0,
            stratum: 7,
        },
    ];
    let mut opts = gam::families::custom_family::BlockwiseFitOptions::default();
    opts.outer_score_subsample = Some(OuterScoreSubsample::from_weighted_rows(rows, 5, 123).into());

    let weighted = outer_weighted_rows(&opts, 5);
    assert_eq!(
        weighted.len(),
        2,
        "outer_weighted_rows must return only retained rows when a subsample is active."
    );
    assert!(
        (weighted[0].weight - 2.5).abs() < 1e-12 && (weighted[1].weight - 5.0).abs() < 1e-12,
        "outer_weighted_rows must preserve per-row Horvitz-Thompson weights exactly."
    );

    let dense = outer_row_weights_by_index(&opts, 5);
    assert!(
        (dense[1] - 2.5).abs() < 1e-12 && (dense[3] - 5.0).abs() < 1e-12,
        "outer_row_weights_by_index must place retained-row weights at their original row indices."
    );
}

#[test]
fn bug_vector_response_rejects_row_weight_dimension_mismatch_and_noise_diagonals_are_documented() {
    let y = Array2::<f64>::zeros((3, 2));
    let target = VectorResponseTarget::new(y.clone(), VectorNoise::Isotropic(2.0));
    let err = target.with_row_weights(array![1.0, 2.0]).expect_err(
        "VectorResponseTarget must reject row_weights vectors whose length does not match N.",
    );
    assert!(
        format!("{err}").contains("length 2"),
        "Mismatched row-weight dimensions should report the observed size."
    );

    let iso = VectorNoise::Isotropic(2.0)
        .diag_precision(2)
        .expect("Isotropic noise with sigma=2 must be valid.");
    assert!(
        (iso[0] - 0.25).abs() < 1e-12 && (iso[1] - 0.25).abs() < 1e-12,
        "Isotropic VectorNoise diag_precision must return 1/sigma^2 in every column."
    );

    let diag = VectorNoise::Diagonal(array![2.0, 4.0])
        .diag_precision(2)
        .expect("Positive diagonal sigmas must be valid.");
    assert!(
        (diag[0] - 0.25).abs() < 1e-12 && (diag[1] - 0.0625).abs() < 1e-12,
        "Diagonal VectorNoise diag_precision must return elementwise 1/sigma_j^2."
    );

    let low_rank = VectorNoise::LowRank {
        diag: array![3.0, 5.0],
        factor: Array2::zeros((2, 1)),
    }
    .diag_precision(2)
    .expect("LowRank noise with positive precision diagonal must be valid.");
    assert!(
        (low_rank[0] - 3.0).abs() < 1e-12 && (low_rank[1] - 5.0).abs() < 1e-12,
        "LowRank VectorNoise diag_precision must return the diagonal precision unchanged."
    );

    GaussianVectorLikelihood::from_target(&VectorResponseTarget::new(
        y,
        VectorNoise::Diagonal(array![1.0, 1.0]),
    ))
    .expect("Well-formed vector-response targets must construct a GaussianVectorLikelihood.");
}

#[test]
fn bug_scale_design_apply_then_inverse_apply_round_trips_identity() {
    let n = 64;
    let mut primary = Array2::<f64>::zeros((n, 3));
    let mut noise = Array2::<f64>::zeros((n, 3));
    let weights = Array1::from_elem(n, 1.0);
    for i in 0..n {
        let t = i as f64 / n as f64;
        primary[[i, 0]] = 1.0;
        primary[[i, 1]] = t;
        primary[[i, 2]] = (3.0 * t).sin();
        noise[[i, 0]] = 1.0;
        noise[[i, 1]] = 0.8 * primary[[i, 1]] - 0.2 * primary[[i, 2]];
        noise[[i, 2]] = 0.4 * primary[[i, 2]] + 0.1 * primary[[i, 1]];
    }
    let transform = build_scale_deviation_transform(&primary, &noise, &weights, 1)
        .expect("A well-conditioned primary/noise design pair should produce a valid scale deviation transform.");
    let transformed = apply_scale_deviation_transform(&primary, &noise, &transform)
        .expect("Applying a valid scale deviation transform should succeed.");

    for i in 0..n {
        assert_eq!(
            transformed[[i, 0]],
            1.0,
            "Scale-deviation apply should leave the intercept/pass-through column unchanged."
        );
    }
    let max_abs: f64 = transformed
        .slice(ndarray::s![.., 1..])
        .iter()
        .fold(0.0_f64, |acc, &v| acc.max(v.abs()));
    assert!(
        max_abs < 1e-6,
        "Scale-deviation apply should numerically eliminate noise components in the primary-design span."
    );
}

#[test]
fn bug_latent_survival_fixed_frailty_accepts_valid_hazard_multiplier_spec() {
    let frailty = FrailtySpec::HazardMultiplier {
        sigma_fixed: Some(0.7),
        loading: HazardLoading::Full,
    };
    let (sigma, loading) = fixed_latent_hazard_frailty(&frailty, "latent-survival")
        .expect("fixed_latent_hazard_frailty must accept a finite non-negative fixed hazard-multiplier sigma.");
    assert!(
        (sigma - 0.7).abs() < 1e-12,
        "fixed_latent_hazard_frailty must return the configured sigma value exactly."
    );
    assert_eq!(
        loading,
        HazardLoading::Full,
        "fixed_latent_hazard_frailty must preserve hazard-loading identity."
    );
}
