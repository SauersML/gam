#![cfg(test)]
//! #2902 item 15: a latent survival search takes the ρ half of its θ box from the
//! seed blocks it realizes. The #2714 loaded/unloaded fixture carries three
//! smoothing coordinates, while its mean term collection declares one penalty,
//! so before the domain was derived over the realized blocks the driver kept the
//! precision box on all three and every fitted strength railed on its face.

use super::*;
use crate::fit_orchestration::{FitConfig, FitRequest, materialize};
use csv::StringRecord;
use gam_data::encode_recordswith_inferred_schema;

const N_ROWS: usize = 160;
const TRUE_BETA: f64 = 0.7;
const TRUE_SIGMA: f64 = 0.5;
const TRUE_RATE: f64 = 0.01;
const TRUE_SHAPE: f64 = 0.08;
const TRUE_MAKEHAM: f64 = 0.02;
const CENSOR_SPAN: f64 = 60.0;

/// SplitMix64 uniforms in `(0, 1)`: the #2714 acceptance fixture's rows, bit for bit.
struct DetRng {
    state: u64,
}

impl DetRng {
    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    fn uniform(&mut self) -> f64 {
        ((self.next_u64() >> 11) as f64 + 0.5) / (1u64 << 53) as f64
    }

    fn normal(&mut self) -> f64 {
        let u1 = self.uniform();
        let u2 = self.uniform();
        (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
    }
}

fn loaded_vs_unloaded_rows() -> gam_data::EncodedDataset {
    let mut rng = DetRng { state: 0x2714_0913 };
    let mut records = Vec::with_capacity(N_ROWS);
    for _ in 0..N_ROWS {
        let x = rng.uniform();
        let frailty = (TRUE_SIGMA * rng.normal()).exp();
        let background_time = -rng.uniform().ln() / TRUE_MAKEHAM;
        let multiplier = TRUE_RATE * frailty * (TRUE_BETA * x).exp();
        let loaded_time = (1.0 + TRUE_SHAPE * -rng.uniform().ln() / multiplier).ln() / TRUE_SHAPE;
        let censor_time = CENSOR_SPAN * (0.5 + rng.uniform());
        let event_time = background_time.min(loaded_time);
        let (time, status) = if event_time <= censor_time {
            (event_time, 1.0)
        } else {
            (censor_time, 0.0)
        };
        records.push(StringRecord::from(vec![
            time.to_string(),
            status.to_string(),
            x.to_string(),
        ]));
    }
    encode_recordswith_inferred_schema(
        vec!["time".to_string(), "status".to_string(), "x".to_string()],
        records,
    )
    .expect("encode the loaded/unloaded survival rows")
}

/// (a) The search domain's edges are `realized_blocks_rho_domain` of the fit's own
/// seed blocks, and not the precision box. (b) Each coordinate certifies railed at
/// its derived upper edge, where its term switches off: the certificate names it
/// railed and records the box it was judged against. The seed blocks are realized
/// through the builders `fit_latent_survival_terms` uses.
#[test]
fn latent_rho_domain_is_its_realized_blocks_domain_and_rails_at_switch_off_2902() {
    drop(gam_problem::laplace_sampler_contract::set_laplace_marginal_corrector(Box::new(
        gam_inference::hmc_io::HmcIoLaplaceMarginalCorrector,
    )));
    drop(gam_problem::rho_posterior::set_rho_posterior_escalator(Box::new(
        gam_inference::rho_posterior::HmcIoRhoPosteriorEscalator,
    )));
    let data = loaded_vs_unloaded_rows();
    let cfg = FitConfig {
        survival_likelihood: Some("latent".to_string()),
        baseline_target: "gompertz-makeham".to_string(),
        time_basis: "ispline".to_string(),
        frailty: FrailtySpec::HazardMultiplier {
            scale: FrailtyScale::Fixed { sigma: TRUE_SIGMA },
            loading: HazardLoading::LoadedVsUnloaded,
        },
        ..FitConfig::default()
    };
    let model = materialize("Surv(time, status) ~ x", &data, &cfg)
        .expect("materialize the #2714 loaded/unloaded fixture");
    let FitRequest::LatentSurvival(request) = model.request else {
        panic!("expected a latent survival request for survival_likelihood=latent");
    };

    let mut seed_spec = request.spec.clone();
    install_latent_time_nullspace_shrinkage_penalty(&mut seed_spec.time_block)
        .expect("time null-space shrinkage penalty");
    let mean_design = build_term_collection_design(request.data, &seed_spec.meanspec)
        .expect("mean block design");
    let mean_offset = mean_design
        .compose_offset(seed_spec.mean_offset.view(), "latent-survival mean block")
        .expect("mean block offset");
    let time_prepared = prepare_latent_time_block(
        &seed_spec.time_block,
        seed_spec.time_design_right.as_ref(),
        seed_spec.derivative_guard,
    )
    .expect("prepared time block");
    let seed_blocks = vec![
        build_time_blockspec(&time_prepared, &seed_spec.time_block),
        build_mean_blockspec(&mean_design, mean_offset),
    ];
    let rho_dim: usize = seed_blocks.iter().map(|block| block.penalties.len()).sum();
    let (lower, upper) = crate::fit_orchestration::drivers::realized_blocks_rho_domain(
        &seed_blocks,
        &request.options,
        rho_dim,
    )
    .expect("realized-blocks rho domain");
    let precision_box = gam_problem::precision_box();

    let pool = rayon::ThreadPoolBuilder::new()
        .stack_size(64 << 20)
        .build()
        .expect("survival worker pool");
    let fit = pool
        .install(|| {
            fit_latent_survival_terms(request.data, request.spec, request.frailty, &request.options)
        })
        .expect("#2714: the loaded/unloaded latent fit must converge and return a fit");
    let certificate = fit
        .fit
        .artifacts
        .criterion_certificate
        .as_ref()
        .expect("a latent chart fit is minted from a certified outer search");

    for k in 0..rho_dim {
        assert!(
            (lower[k], upper[k]) != precision_box,
            "#2902: coordinate {k}'s realized-blocks domain [{}, {}] is the precision box",
            lower[k],
            upper[k]
        );
        let fact = certificate
            .railed_facts
            .iter()
            .find(|fact| fact.index == k)
            .unwrap_or_else(|| {
                panic!(
                    "#2902: rho coordinate {k} is not railed; railed={:?} facts={:?}",
                    certificate.lambdas_railed, certificate.railed_facts
                )
            });
        assert!(
            fact.lower == lower[k] && fact.upper == upper[k],
            "#2902 (a): coordinate {k} was searched on [{}, {}], not its realized-blocks domain \
             [{}, {}] (precision box {precision_box:?})",
            fact.lower,
            fact.upper,
            lower[k],
            upper[k]
        );
        assert!(
            certificate.lambdas_railed.contains(&k),
            "#2902 (b): coordinate {k} has a railed fact but is not in lambdas_railed={:?}",
            certificate.lambdas_railed
        );
        assert!(
            fact.upper - fact.theta <= fact.margin && fact.theta - fact.lower > fact.margin,
            "#2902 (b): coordinate {k} is railed but not at its upper switch-off edge: {fact}"
        );
    }
    // (b) The rails are certified, not only detected: with the outward components of
    // the railed coordinates projected away, the gradient is inside the certificate's
    // own bound.
    match &certificate.stationarity {
        gam_solve::model_types::OuterStationarityCertificate::AnalyticGradient {
            projected_grad_norm,
            bound,
            ..
        } => assert!(
            *projected_grad_norm <= *bound,
            "#2902 (b): projected gradient {projected_grad_norm:e} exceeds its bound {bound:e}"
        ),
        gam_solve::model_types::OuterStationarityCertificate::AsymptoteRail { rails, .. } => {
            for k in 0..rho_dim {
                assert!(
                    rails.iter().any(|rail| rail.index == k),
                    "#2902 (b): rho coordinate {k} is not among the certified rails {rails:?}"
                );
            }
        }
        other => panic!("#2902 (b): a railed latent fit certified by {other:?}"),
    }
}
