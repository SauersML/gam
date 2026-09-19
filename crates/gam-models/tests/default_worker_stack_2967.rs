//! gam#2967: a survival marginal-slope fit must run on Rayon workers with std's default 2 MiB
//! thread stack.
//!
//! gam-models compiles here as a plain dependency, without its `cfg(test)` modules, the way a
//! library consumer builds it. The fit is gnomon calibrate's shape: 2,000 rows entering at ages
//! 20-24 and leaving at 40-52, about a third of them events, a rigid probit marginal slope on a
//! declared latent law (the anchored frame), and 12 workers. It runs inside a local pool whose
//! workers get 2 MiB explicitly, so neither `RUST_MIN_STACK` nor the process's global pool can
//! give it more.
//!
//! The anchored row program's fourth-order tower evaluation reserves about 76 KiB of stack.
//! Before the fix it was the body of the closure every row-parallel tower build handed to Rayon,
//! and a build that inlined that closure into Rayon's recursive split helper reserved the whole
//! row program again at every split level: gnomon's gdb trace at the overflow shows split frames
//! of 77,880 and 80,000 bytes, 72 frames and six nested steals deep. A stack overflow aborts the
//! process instead of returning an error. The tower builds now fill heap slots through one
//! out-of-line writer, so a split frame holds a row index and a slot pointer whatever the
//! inliner does.
//!
//! gam's own test profile keeps that closure out of line, so this fit completes there with or
//! without the fix; it is the consumer contract, not the discriminating pin. The pin is the
//! root build.rs rule `scan_for_marginal_slope_tower_seeds`, which refuses a primary tower seeded
//! anywhere in the module's production code except the out-of-line writer.

use csv::StringRecord;
use gam_data::encode_recordswith_inferred_schema;
use gam_linalg::utils::splitmix64;
use gam_models::fit_orchestration::{DeclaredLatentLaw, FitConfig, FitResult, fit_from_formula};

const N: usize = 2_000;
const WORKERS: usize = 12;
/// std's default stack for a spawned thread, which is what a Rayon worker gets when neither the
/// pool builder nor `RUST_MIN_STACK` sets one.
const DEFAULT_WORKER_STACK: usize = 2 << 20;

fn next_unit(state: &mut u64) -> f64 {
    (splitmix64(state) >> 11) as f64 / (1u64 << 53) as f64
}

fn next_gauss(state: &mut u64) -> f64 {
    let u1 = next_unit(state).max(1e-12);
    let u2 = next_unit(state);
    (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
}

/// The #2923 skewed two-component law on 41 nodes, standardised to zero mean and unit variance.
fn skewed_law() -> DeclaredLatentLaw {
    let raw_nodes: Vec<f64> = (0..41).map(|k| -2.5 + 0.15 * k as f64).collect();
    let raw: Vec<f64> = raw_nodes
        .iter()
        .map(|&u| {
            (-0.5 * ((u + 0.9) / 0.5).powi(2)).exp()
                + 0.20 * (-0.5 * ((u - 2.8) / 0.5).powi(2)).exp()
        })
        .collect();
    let total: f64 = raw.iter().sum();
    let weights: Vec<f64> = raw.into_iter().map(|w| w / total).collect();
    let mean: f64 = raw_nodes.iter().zip(&weights).map(|(u, w)| u * w).sum();
    let var: f64 = raw_nodes
        .iter()
        .zip(&weights)
        .map(|(u, w)| (u - mean).powi(2) * w)
        .sum();
    let nodes = raw_nodes.iter().map(|u| (u - mean) / var.sqrt()).collect();
    DeclaredLatentLaw { nodes, weights }
}

/// Draw a node with its weight as probability.
fn draw(law: &DeclaredLatentLaw, u: f64) -> f64 {
    let mut cumulative = 0.0;
    for (&node, &weight) in law.nodes.iter().zip(&law.weights) {
        cumulative += weight;
        if u < cumulative {
            return node;
        }
    }
    *law.nodes.last().expect("non-empty law")
}

fn dataset(law: &DeclaredLatentLaw) -> (gam_data::EncodedDataset, usize) {
    let headers = ["entry_age", "exit_age", "event", "z"]
        .iter()
        .map(|s| s.to_string())
        .collect::<Vec<_>>();
    let mut state: u64 = 0x2967_0000_0000_0001;
    let mut rows = Vec::with_capacity(N);
    let mut events = 0usize;
    for _ in 0..N {
        let z = draw(law, next_unit(&mut state));
        let entry = 20.0 + 4.0 * next_unit(&mut state);
        let exit = 40.0 + 12.0 * next_unit(&mut state);
        let event = u8::from(0.8 * z + next_gauss(&mut state) > 0.5);
        events += usize::from(event);
        rows.push(StringRecord::from(vec![
            entry.to_string(),
            exit.to_string(),
            event.to_string(),
            z.to_string(),
        ]));
    }
    let data = encode_recordswith_inferred_schema(headers, rows)
        .expect("encode the #2967 default-stack fixture");
    (data, events)
}

#[test]
fn anchored_survival_marginal_slope_fit_runs_on_default_worker_stacks_2967() {
    drop(
        gam_problem::laplace_sampler_contract::set_laplace_marginal_corrector(Box::new(
            gam_inference::hmc_io::HmcIoLaplaceMarginalCorrector,
        )),
    );
    drop(gam_problem::rho_posterior::set_rho_posterior_escalator(
        Box::new(gam_inference::rho_posterior::HmcIoRhoPosteriorEscalator),
    ));
    #[cfg(target_os = "macos")]
    gam_gpu::configure_global_policy(gam_gpu::GpuPolicy::Off);

    let law = skewed_law();
    let (data, events) = dataset(&law);
    let config = FitConfig {
        survival_likelihood: Some("marginal-slope".to_string()),
        z_column: Some("z".to_string()),
        slope_formula: Some("1".to_string()),
        time_num_internal_knots: 3,
        declared_latent_law: Some(law),
        ..FitConfig::default()
    };
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(WORKERS)
        .stack_size(DEFAULT_WORKER_STACK)
        .build()
        .expect("build the default-stack worker pool");
    let result = pool
        .install(|| fit_from_formula("Surv(entry_age, exit_age, event) ~ 1", &data, &config))
        .expect("the anchored survival marginal-slope fit on default worker stacks");
    let FitResult::SurvivalMarginalSlope(fit) = result else {
        panic!("expected a SurvivalMarginalSlope fit result");
    };
    eprintln!(
        "[2967] n={N} events={events} workers={WORKERS} stack={DEFAULT_WORKER_STACK} log-likelihood={:.6}",
        fit.fit.log_likelihood
    );
    // The declared law must be the fit's measure, so the anchored frame, the one whose tower
    // builds overflowed, is the frame that ran.
    assert!(
        matches!(
            fit.latent_measure,
            gam_models::bms::LatentMeasureKind::GlobalEmpirical { .. }
        ),
        "the declared latent law must be the fit's measure"
    );
    assert!(
        fit.fit.log_likelihood.is_finite(),
        "the fit's log-likelihood must be finite, got {}",
        fit.fit.log_likelihood
    );
}
