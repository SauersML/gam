//! A plain Poisson `y ~ te(x0, x1)` fit must finish: the #784 block-local
//! correction on a multi-axis block is integrated axis by axis.
//!
//! The pyGAM audit's speed cell (`bench/pygam_audit/speed/worker.py gamfit
//! poisson 10000 te 2`) never returned: killed at 900 s, where seeds 0 and 1
//! took about 13 s. At its certified Laplace optimum the skewness verdict
//! admitted an `m = 7` block (seed 0: `m = 8`), and the order search raised
//! a tensor Gauss–Hermite rule over all of it — `4^7 = 16384` nodes, about
//! 5.5 min per rule on one thread, with paired differences still near `1e-4`
//! against a `1e-8` target, while each axis alone resolves near order 10. No
//! tensor rule over that block is feasible.
//!
//! The correction is now `Σ_r Δ_r + Φ`: an exact one-dimensional quadrature per
//! curvature axis plus the analytic mixed-axis Laplace term, the part of the
//! second-order expansion no single axis carries. What is left is `O(n_eff⁻²)`,
//! below the correction's own resolution target, and the cost is linear in `m`.
//!
//! The fixture is a smaller instance of the same model, on the speed cell's
//! generator: at `n = 2000` its optimum admits an `m = 3` block, resolved at
//! axis orders `[9, 9, 8]` — 26 nodes where the tensor rule needs 648 per
//! evaluation, and the gap grows geometrically with `m`. There the split and
//! the tensor rule agree to `9.9e-7` on `Δ_b = −1.96e-3`, while `Φ = 1.1e-4`,
//! so the mixed-axis term is what closes the gap between them. The test
//! asserts that the fit certifies, that the correction engaged on a block of
//! more than one axis — so the axis split is what ran, not a decline — and
//! that every axis resolved at a one-dimensional order.

use csv::StringRecord;
use gam::estimate::outer_eval_capture::{enable_rho_outer_audit, take_rho_outer_audit};
use gam::estimate::{ExternalOptimOptions, optimize_external_designwith_heuristic_log_lambdas};
use gam::smooth::build_term_collection_design;
use gam::types::{InverseLink, LikelihoodSpec, ResponseFamily, StandardLink};
use gam::{FitConfig, FitRequest, encode_recordswith_inferred_schema, init_parallelism};
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand_distr::{Distribution, Poisson, Uniform};
use std::f64::consts::PI;

const SEED: u64 = 0;
const ROWS: usize = 2000;

/// The speed cell's generator: `x0, x1 ~ U(0, 1)`,
/// `log μ = 0.5 + 0.7·sin(2πx0)·cos(2πx1)`, Poisson counts.
fn tensor_poisson_dataset(seed: u64, n: usize) -> gam::data::EncodedDataset {
    let mut rng = StdRng::seed_from_u64(seed);
    let unit = Uniform::new(0.0_f64, 1.0).expect("uniform [0,1]");
    let mut rows: Vec<StringRecord> = Vec::with_capacity(n);
    for _ in 0..n {
        let x0 = unit.sample(&mut rng);
        let x1 = unit.sample(&mut rng);
        let eta = 0.5 + 0.7 * (2.0 * PI * x0).sin() * (2.0 * PI * x1).cos();
        let count: f64 = Poisson::new(eta.exp())
            .expect("a finite positive Poisson rate")
            .sample(&mut rng);
        rows.push(StringRecord::from(vec![
            x0.to_string(),
            x1.to_string(),
            count.to_string(),
        ]));
    }
    let headers = ["x0", "x1", "y"].into_iter().map(String::from).collect();
    encode_recordswith_inferred_schema(headers, rows).expect("encode the fixture")
}

#[test]
fn poisson_tensor_fit_integrates_a_multi_axis_block_axis_by_axis() {
    // Registers the Laplace marginal corrector; without it the correction
    // declines before the diagnostic and the assertions below are vacuous.
    init_parallelism();
    let data = tensor_poisson_dataset(SEED, ROWS);
    let config = FitConfig {
        family: Some("poisson".to_string()),
        ..FitConfig::default()
    };
    let model = gam::materialize("y ~ te(x0, x1)", &data, &config).expect("materialize");
    let FitRequest::Standard(request) = model.request else {
        panic!("a Poisson tensor smooth is a standard request");
    };
    let design = build_term_collection_design(request.data.view(), &request.spec)
        .expect("build the tensor design");
    let opts = ExternalOptimOptions {
        family: LikelihoodSpec::new(
            ResponseFamily::Poisson,
            InverseLink::Standard(StandardLink::Log),
        ),
        latent_cloglog: None,
        mixture_link: None,
        optimize_mixture: false,
        sas_link: None,
        optimize_sas: false,
        compute_inference: false,
        skip_rho_posterior_inference: true,
        max_iter: 300,
        tol: 1.0e-8,
        nullspace_dims: design.nullspace_dims.clone(),
        linear_constraints: design.linear_constraints.clone(),
        firth_bias_reduction: Some(false),
        rho_prior: Default::default(),
        persistent_warm_start_store: None,
    };
    let offset = &*request.offset + &design.affine_offset;

    enable_rho_outer_audit();
    let fit = optimize_external_designwith_heuristic_log_lambdas(
        request.y.view(),
        request.weights.view(),
        design.design.clone(),
        offset.view(),
        design.penalties.clone(),
        None,
        &opts,
    )
    .expect("the Poisson tensor fit must fit");
    let audit = take_rho_outer_audit().expect("the fit's outer evaluations ran on this thread");

    assert!(
        fit.outer_converged,
        "the Poisson tensor fit's outer search must certify"
    );
    let record = audit.quadrature_marginal.unwrap_or_else(|| {
        panic!(
            "the fixture's Laplace optimum must admit the #784 correction; a decline makes \
             this test vacuous"
        )
    });
    eprintln!(
        "[poisson te] block_cols={:?} axis_orders={:?} axis_errors={:?} delta_b={:.6e} \
         node_count={} log_lambdas={:?}",
        record.block_cols,
        record.axis_orders,
        record.axis_quadrature_errors,
        record.delta_b,
        record.node_count,
        fit.log_lambdas.to_vec()
    );
    let m = record.axis_orders.len();
    assert!(
        m >= 2,
        "the fixture must admit a multi-axis block (got m = {m}), or the axis split is not \
         what this test exercised"
    );
    assert_eq!(record.block_cols.len(), m);
    // One rule per axis: the nodes evaluated are the SUM of the axis orders,
    // not their product.
    assert_eq!(
        record.node_count,
        record.axis_orders.iter().sum::<usize>(),
        "a multi-axis block must be integrated one axis at a time"
    );
    assert!(record.delta_b.is_finite());
    assert!(
        record
            .axis_quadrature_errors
            .iter()
            .all(|error| error.is_finite()),
        "every axis must have resolved at a one-dimensional order: {:?}",
        record.axis_quadrature_errors
    );
}
