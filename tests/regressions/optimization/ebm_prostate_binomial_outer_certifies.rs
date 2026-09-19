//! Binomial-logit fit on the prostate benchmark whose pc2 range penalty runs
//! toward `λ = ∞` must return a certified outer optimum.
//!
//! This is the fit the EBM binomial-logit quality comparison makes
//! (`y ~ s(pc1, k=5) + s(pc2, k=5)` on the 3-of-4 training rows). The #1561
//! sweep recorded it refused with coordinate 2 held on its rail:
//!
//! ```text
//! |Pg|=2.280e-5 > bound=7.302e-6, railed=[2] theta=22.73,
//! rho_checkpoint=[2.8066877696589136, -2.340994673400271,
//!                 22.73260730311278, -3.0301791821018074]
//! ```
//!
//! On the tree immediately before the rail-face KKT certificate (#3212,
//! parent `658fec0f^1`) this test still fails, now with nothing railed and the
//! search stopped just over its bound:
//!
//! ```text
//! |Pg|=2.249e-10 > bound=1.975e-10 (rung=coordinate-band) railed=[]
//! termination=gradient_tolerance(|g|=2.835361e-11 < 1.000000e-10)
//! ```
//!
//! After #3212 the returned point certifies: `λ₂ ≈ 1.67e9` (`ρ₂ ≈ 21.2`, where
//! coordinate 2's tail `−c·e^{−ρ}` is already under the bound) and an analytic
//! projected gradient of `9.0e-11` against a bound of `2.5e-10`.
//!
//! The assertions are on the certificate itself, not on the route. A rail
//! certificate for coordinate 2 is as valid as a gradient certificate, but any
//! rail it reports must carry evidence its own well-formedness rule admits.

use gam::inference::data::EncodedDataset;
use gam::{FitConfig, FitResult, fit_from_formula, init_parallelism, load_csvwith_inferred_schema};
use ndarray::Array2;
use std::path::Path;

const PROSTATE_CSV: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/bench/datasets/prostate.csv");

#[test]
fn prostate_binomial_logit_outer_optimum_certifies() {
    assert_prostate_binomial_logit_outer_optimum_certifies(4);
}

/// The same fit on the 4-of-5 training rows that the pyGAM binomial-logit
/// comparison makes (fold 0 of its `i % 5` split, and its single real-data
/// split). After the per-coordinate Theorem 9 band (#2954) this fit refused on
/// the reference-quality run with coordinate 2 railed and the search
/// cost-stalled on BFGS curvature:
///
/// ```text
/// |Pg|=2.421e-8 bound=2.536e-10 (rung=coordinate-band) curvature_source=unavailable
/// railed=[2] origin=BfgsCostStallExit search_hessian_source=BfgsApprox
/// rho_checkpoint=[20.97, -2.378, 22.62, -3.060]
/// ```
#[test]
fn prostate_binomial_logit_outer_optimum_certifies_on_the_five_fold_split() {
    assert_prostate_binomial_logit_outer_optimum_certifies(5);
}

/// Fit `y ~ s(pc1, k=5) + s(pc2, k=5)` (binomial logit) on the prostate rows
/// with `i % holdout_modulus != 0` and assert the returned outer optimum carries
/// a certificate that certifies.
fn assert_prostate_binomial_logit_outer_optimum_certifies(holdout_modulus: usize) {
    init_parallelism();
    let ds = load_csvwith_inferred_schema(Path::new(PROSTATE_CSV)).expect("load prostate.csv");
    let train_rows: Vec<usize> = (0..ds.values.nrows())
        .filter(|i| i % holdout_modulus != 0)
        .collect();
    let mut values = Array2::<f64>::zeros((train_rows.len(), ds.headers.len()));
    for (r, &i) in train_rows.iter().enumerate() {
        values.row_mut(r).assign(&ds.values.row(i));
    }
    let train = EncodedDataset {
        headers: ds.headers.clone(),
        values,
        schema: ds.schema.clone(),
        column_kinds: ds.column_kinds.clone(),
    };
    let config = FitConfig {
        family: Some("binomial".to_string()),
        link: Some("logit".to_string()),
        ..FitConfig::default()
    };
    let fit = fit_from_formula("y ~ s(pc1, k=5) + s(pc2, k=5)", &train, &config)
        .expect("prostate binomial-logit fit must return a certified outer optimum");
    let FitResult::Standard(standard) = &fit else {
        panic!("a two-smooth binomial formula fits as a standard model");
    };

    let lambdas = &standard.fit.lambdas;
    assert_eq!(lambdas.len(), 4, "two smooths, two penalties each: {lambdas:?}");
    assert!(
        lambdas.iter().all(|l| l.is_finite() && *l > 0.0),
        "every returned smoothing parameter is a finite positive number: {lambdas:?}"
    );
    assert!(
        lambdas[2].ln() > 15.0,
        "pc2's range penalty is the coordinate running toward λ = ∞ that this \
         regression is about; ρ₂ = {} (λ = {lambdas:?})",
        lambdas[2].ln()
    );
    let score = standard
        .fit
        .reml_score()
        .expect("a binomial fit with a positive-rank penalized Hessian has a LAML criterion");
    assert!(score.is_finite(), "LAML criterion {score}");

    let certificate = standard
        .fit
        .convergence_evidence()
        .outer_certificate()
        .expect("the outer optimum carries its certificate");
    assert!(
        certificate.certifies(),
        "outer certificate refused: {:?}\n{certificate:#?}",
        certificate.refusal()
    );
    let stationarity = &certificate.stationarity;
    assert!(
        stationarity.projected_norm() <= stationarity.bound(),
        "load-bearing residual {:.3e} exceeds its bound {:.3e} ({})",
        stationarity.projected_norm(),
        stationarity.bound(),
        stationarity.rung().label
    );
    for rail in stationarity.rails() {
        assert!(
            rail.evidence.admits(rail.tail_constant),
            "rail on coordinate {} carries {} evidence that does not admit its \
             tail constant {}: {rail:?}",
            rail.index,
            rail.evidence.route(),
            rail.tail_constant
        );
    }
}
