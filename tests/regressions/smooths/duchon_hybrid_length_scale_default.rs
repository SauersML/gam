//! Regression test for #750.
//!
//! A `duchon(...)` smooth with an explicit `length_scale` (the documented hybrid
//! Duchon–Matérn mode) but **no explicit `power=`** crashed at basis-generation
//! time for even covariate dimensions `d ≥ 4` (originally surfaced as a
//! "non-finite values in self-adjoint eigendecomposition" error inside a
//! `bernoulli-marginal-slope` logslope surface; later a clean fit-time
//! "Duchon pointwise kernel values require 2*(p+s) > dimension" validation
//! error).
//!
//! Root cause: the request-layer cubic structural default resolves the spectral
//! power to the fractional `s = (d-1)/2`, which is a half-integer for even `d`.
//! The hybrid Matérn-blended kernel requires an *integer* `s`, and the basis
//! builder's `power_as_usize` maps a non-integer to `0` (not its floor). For
//! `d ≥ 4` that yields `2(p+s) = 2p = 4 ≤ d`, an inadmissible kernel that is
//! non-finite at the origin.
//!
//! Fix: for the hybrid + cubic-default case the request layer resolves the
//! smallest admissible *integer* `(nullspace, s)` via `resolve_duchon_orders`,
//! honoring the collocation order of the default operator penalties. This
//! recovers the canonical thin-plate smoothness order `m = ⌊d/2⌋ + 1` and agrees
//! with the fractional cubic default for odd `d`.

use gam::ResourcePolicy;
use gam::basis::DuchonNullspaceOrder;
use gam::estimate::FitOptions;
use gam::inference::data::EncodedDataset;
use gam::inference::formula_dsl::parse_formula;
use gam::inference::model::{ColumnKindTag, DataSchema, SchemaColumn};
use gam::smooth::{SmoothBasisSpec, TermCollectionSpec};
use gam::terms::term_builder::build_termspec;
use gam::types::{InverseLink, LikelihoodSpec, ResponseFamily, StandardLink};
use ndarray::{Array1, Array2};
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand_distr::{Distribution, Normal};

/// `p = m` (polynomial null space contains all polynomials of degree `< m`).
fn p_from_nullspace(order: DuchonNullspaceOrder) -> usize {
    match order {
        DuchonNullspaceOrder::Zero => 1,
        DuchonNullspaceOrder::Linear => 2,
        DuchonNullspaceOrder::Degree(k) => k + 1,
    }
}

/// Build an encoded dataset whose first column is the response `y` followed by
/// `d` continuous covariate columns `x1..xd`, filled with deterministic noise.
fn dataset_with_covariates(n: usize, d: usize) -> EncodedDataset {
    let mut rng = StdRng::seed_from_u64(0x0750_0750_0750_0750);
    let normal = Normal::new(0.0, 1.0).unwrap();
    let ncol = d + 1;
    let mut values = Array2::<f64>::zeros((n, ncol));
    for i in 0..n {
        // y is just structured noise; this test is about basis generation, not recovery.
        values[[i, 0]] = normal.sample(&mut rng);
        for j in 0..d {
            values[[i, 1 + j]] = normal.sample(&mut rng);
        }
    }
    let mut headers = vec!["y".to_string()];
    let mut columns = vec![SchemaColumn {
        name: "y".into(),
        kind: ColumnKindTag::Continuous,
        levels: vec![],
    }];
    let mut column_kinds = vec![ColumnKindTag::Continuous];
    for j in 0..d {
        let name = format!("x{}", j + 1);
        headers.push(name.clone());
        columns.push(SchemaColumn {
            name,
            kind: ColumnKindTag::Continuous,
            levels: vec![],
        });
        column_kinds.push(ColumnKindTag::Continuous);
    }
    EncodedDataset {
        headers,
        values,
        schema: DataSchema { columns },
        column_kinds,
    }
}

fn resolve_hybrid_default_spec(n: usize, d: usize) -> TermCollectionSpec {
    let ds = dataset_with_covariates(n, d);
    let cmap = ds.column_map();
    let vars: Vec<String> = (0..d).map(|j| format!("x{}", j + 1)).collect();
    let formula = format!(
        "y ~ duchon({}, centers=12, length_scale=1.0)",
        vars.join(", ")
    );
    let parsed = parse_formula(&formula).unwrap_or_else(|e| panic!("{formula}: {e}"));
    let mut notes = Vec::new();
    build_termspec(
        &parsed.terms,
        &ds,
        &cmap,
        &mut notes,
        &ResourcePolicy::default_library(),
    )
    .unwrap_or_else(|e| panic!("{formula}: build_termspec failed: {e:?}"))
}

/// The hybrid cubic default must resolve to an admissible *integer* spectral
/// power for every covariate dimension — in particular the even `d ≥ 4` cases
/// that previously truncated to `s = 0` and crashed.
#[test]
fn hybrid_duchon_cubic_default_resolves_to_admissible_integer_power() {
    for d in 1..=8usize {
        let spec = resolve_hybrid_default_spec(96, d);
        let smooth = &spec.smooth_terms[0];
        let (power, nullspace, length_scale) = match &smooth.basis {
            SmoothBasisSpec::Duchon { spec, .. } => {
                (spec.power, spec.nullspace_order, spec.length_scale)
            }
            other => panic!("d={d}: expected Duchon basis, got {other:?}"),
        };

        assert_eq!(
            length_scale,
            Some(1.0),
            "d={d}: hybrid length_scale must survive resolution"
        );
        assert!(
            power.is_finite() && power >= 0.0 && power.fract() == 0.0,
            "d={d}: hybrid Duchon power must resolve to a non-negative integer, got {power}"
        );

        let p = p_from_nullspace(nullspace);
        let s = power as usize;
        let spectral = 2 * (p + s);
        // Kernel existence (finite at the origin) AND D1 collocation for the
        // default mass+tension penalties.
        assert!(
            spectral > d,
            "d={d}: 2(p+s)={spectral} must exceed dimension (p={p}, s={s})"
        );
        assert!(
            spectral > d + 1,
            "d={d}: 2(p+s)={spectral} must clear D1 collocation d+1={} (p={p}, s={s})",
            d + 1
        );

        // For odd d the fractional cubic default (d-1)/2 is already an integer,
        // and the collocation floor forces exactly that value.
        if d % 2 == 1 {
            assert_eq!(
                s,
                (d - 1) / 2,
                "d={d}: odd-dimension hybrid default should match the cubic power"
            );
        }
    }
}

/// End-to-end guard: a 4D hybrid Duchon smooth with the cubic default — the exact
/// shape from the issue repro — must build its basis and fit without the
/// non-finite eigendecomposition crash.
#[test]
fn hybrid_duchon_4d_default_builds_and_fits_via_formula() {
    let n = 240usize;
    let d = 4usize;
    let ds = dataset_with_covariates(n, d);
    let cmap = ds.column_map();
    let vars: Vec<String> = (0..d).map(|j| format!("x{}", j + 1)).collect();
    let formula = format!(
        "y ~ duchon({}, centers=10, length_scale=1.0)",
        vars.join(", ")
    );
    let parsed = parse_formula(&formula).unwrap();
    let mut notes = Vec::new();
    let spec = build_termspec(
        &parsed.terms,
        &ds,
        &cmap,
        &mut notes,
        &ResourcePolicy::default_library(),
    )
    .expect("4D hybrid Duchon default termspec must resolve");

    let y = ds.values.column(0).to_owned();
    let weights = Array1::ones(n);
    let offset = Array1::zeros(n);

    let fitted = gam::smooth::fit_term_collection_forspec(
        ds.values.view(),
        y.view(),
        weights.view(),
        offset.view(),
        &spec,
        gaussian_identity_likelihood(),
        &fit_options(),
    )
    .expect(
        "#750: 4D hybrid Duchon default must build its basis (no non-finite eigendecomposition)",
    );

    assert!(
        fitted.fit.beta.iter().all(|v| v.is_finite()),
        "#750: fitted coefficients must be finite"
    );
}

fn gaussian_identity_likelihood() -> LikelihoodSpec {
    LikelihoodSpec::new(
        ResponseFamily::Gaussian,
        InverseLink::Standard(StandardLink::Identity),
    )
}

fn fit_options() -> FitOptions {
    FitOptions {
        latent_cloglog: None,
        mixture_link: None,
        optimize_mixture: false,
        sas_link: None,
        optimize_sas: false,
        compute_inference: false,
        skip_rho_posterior_inference: false,
        max_iter: 12,
        tol: 1e-4,
        nullspace_dims: vec![],
        linear_constraints: None,
        firth_bias_reduction: false,
        adaptive_regularization: None,
        penalty_shrinkage_floor: None,
        rho_prior: Default::default(),
        kronecker_penalty_system: None,
        kronecker_factored: None,
        persist_warm_start_disk: false,
    }
}
