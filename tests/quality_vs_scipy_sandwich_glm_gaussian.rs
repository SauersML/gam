//! End-to-end quality: gam's ALO sandwich standard error on the linear
//! predictor must reproduce the *exact* model-based prediction covariance of a
//! mature OLS/GLM reference on an unpenalized Gaussian linear model.
//!
//! Mature comparator: **Python `statsmodels` (OLS / Gaussian GLM)** — the
//! standard, widely-trusted regression stack. For an unpenalized Gaussian
//! linear model the model-based ("sandwich-with-correct-likelihood") covariance
//! of the coefficients is the textbook OLS covariance
//!
//!   Cov(beta_hat) = sigma^2 (X'X)^{-1},      sigma^2 = RSS / (n - p),
//!
//! and the standard error of the in-sample linear predictor at row i is
//!
//!   SE(eta_i) = sqrt( x_i' Cov(beta_hat) x_i ) = sqrt( sigma^2 x_i' (X'X)^{-1} x_i ).
//!
//! gam's ALO sandwich formula (alo.rs line ~120) is
//!
//!   Var_sandwich(eta_i) = phi * ( x_i' H^{-1} x_i  -  ||E t||^2  -  ridge ||t||^2 ),
//!     H = X'WX + S + ridge*I,   t = H^{-1} x_i,   phi = RSS/(n - edf).
//!
//! With NO penalty (a purely parametric `y ~ x1 + ... + x5`, so S = 0, E absent,
//! ridge = 0, W = I for Gaussian-identity) this collapses *exactly* to
//! phi * x_i' (X'X)^{-1} x_i with phi = RSS/(n - p) — bit-for-bit the standard
//! OLS prediction variance. Comparing the two on a no-smooth linear model with
//! identical data is therefore a direct, unambiguous test that gam's sandwich
//! implementation equals the canonical definition; any divergence is a real bug
//! in ALO, not an approximation gap. The bound is tight (< 1e-8 absolute) because
//! both engines compute the identical unpenalized matrix.

use csv::StringRecord;
use gam::inference::alo::compute_alo_diagnostics_from_fit;
use gam::test_support::reference::{Column, max_abs_diff, pearson, run_python};
use gam::types::LinkFunction;
use gam::{
    FitConfig, FitResult, encode_recordswith_inferred_schema, fit_from_formula, init_parallelism,
};
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand_distr::{Distribution, Normal};

#[test]
fn gam_alo_sandwich_se_matches_statsmodels_ols_on_linear_gaussian() {
    init_parallelism();

    // ---- synthetic unpenalized Gaussian linear model -----------------------
    // n = 100, p = 5 covariates + intercept; y = X*beta + N(0, sigma^2).
    // Fixed seed => both engines see byte-identical data.
    const N: usize = 100;
    const P: usize = 5;
    let mut rng = StdRng::seed_from_u64(20240529);
    let xdist = Normal::new(0.0_f64, 1.0).expect("normal x");
    let edist = Normal::new(0.0_f64, 0.7).expect("normal eps");
    let beta_true = [1.3_f64, -0.8, 0.5, 2.1, -1.4];
    let intercept_true = 0.4_f64;

    // Column-major storage so we can hand identical vectors to the reference.
    let mut xcols: Vec<Vec<f64>> = (0..P).map(|_| Vec::with_capacity(N)).collect();
    let mut yvec: Vec<f64> = Vec::with_capacity(N);
    for _ in 0..N {
        let mut eta = intercept_true;
        let mut xrow = [0.0_f64; P];
        for j in 0..P {
            let xij = xdist.sample(&mut rng);
            xrow[j] = xij;
            eta += beta_true[j] * xij;
        }
        for j in 0..P {
            xcols[j].push(xrow[j]);
        }
        yvec.push(eta + edist.sample(&mut rng));
    }

    // ---- build the gam dataset (identical numbers) -------------------------
    let headers: Vec<String> = (0..P)
        .map(|j| format!("x{}", j + 1))
        .chain(std::iter::once("y".to_string()))
        .collect();
    let mut records: Vec<StringRecord> = Vec::with_capacity(N);
    for i in 0..N {
        let mut fields: Vec<String> = (0..P).map(|j| format!("{:.17e}", xcols[j][i])).collect();
        fields.push(format!("{:.17e}", yvec[i]));
        records.push(StringRecord::from(fields));
    }
    let data = encode_recordswith_inferred_schema(headers, records).expect("encode dataset");

    // ---- fit gam: y ~ x1 + ... + x5, Gaussian-identity, NO smooth ----------
    // A purely parametric formula => no penalty, no ridge => the ALO sandwich
    // reduces to the exact OLS prediction covariance.
    let formula = "y ~ x1 + x2 + x3 + x4 + x5";
    let cfg = FitConfig {
        family: Some("gaussian".to_string()),
        ..FitConfig::default()
    };
    let result = fit_from_formula(formula, &data, &cfg).expect("gam fit");
    let FitResult::Standard(fit) = result else {
        panic!("expected a standard GAM fit for an unpenalized linear model");
    };
    let edf = fit.fit.edf_total().expect("gam reports total edf");
    // For an unpenalized model with intercept + 5 slopes, edf must be 6.
    assert!(
        (edf - (P as f64 + 1.0)).abs() < 1e-6,
        "unpenalized linear model should have edf = p+1 = {}; got {edf:.6}",
        P + 1
    );

    let yview = data.values.column(data.column_map()["y"]);
    let alo = compute_alo_diagnostics_from_fit(&fit.fit, yview, LinkFunction::Identity)
        .expect("gam ALO diagnostics");
    let gam_se_sandwich: Vec<f64> = alo.se_sandwich.to_vec();
    assert_eq!(gam_se_sandwich.len(), N, "ALO se_sandwich length mismatch");
    // With no penalty the sandwich SE must equal the Bayesian SE (S = 0).
    let se_bayes: Vec<f64> = alo.se_bayes.to_vec();
    let sandwich_vs_bayes = max_abs_diff(&gam_se_sandwich, &se_bayes);
    assert!(
        sandwich_vs_bayes < 1e-12,
        "unpenalized => sandwich SE must equal Bayesian SE; max diff = {sandwich_vs_bayes:.3e}"
    );

    // ---- reference: statsmodels OLS prediction SE on eta -------------------
    // SE(eta_i) = sqrt( x_i' [sigma^2 (X'X)^{-1}] x_i ), sigma^2 = RSS/(n-p).
    // statsmodels.OLS.cov_params() = sigma^2 (X'X)^{-1} exactly, so the
    // per-row prediction SE is the canonical sandwich SE we compare against.
    let xnames: Vec<String> = (0..P).map(|j| format!("x{}", j + 1)).collect();
    let mut columns: Vec<Column> = (0..P)
        .map(|j| Column::new(xnames[j].as_str(), &xcols[j]))
        .collect();
    columns.push(Column::new("y", &yvec));

    let r = run_python(
        &columns,
        r#"
import numpy as np
import statsmodels.api as sm
Xcols = [df["x%d" % (j + 1)] for j in range(5)]
X = np.column_stack([np.asarray(c, dtype=float) for c in Xcols])
y = np.asarray(df["y"], dtype=float)
Xd = sm.add_constant(X, has_constant="add")  # intercept first
model = sm.OLS(y, Xd).fit()
cov = np.asarray(model.cov_params(), dtype=float)  # sigma^2 (X'X)^{-1}
# Per-row prediction SE on the linear predictor eta_i = x_i' beta.
se = np.sqrt(np.einsum("ij,jk,ik->i", Xd, cov, Xd))
emit("se_sandwich", se)
emit("sigma2", [model.mse_resid])  # RSS/(n-p)
"#,
    );
    let ref_se = r.vector("se_sandwich");
    assert_eq!(ref_se.len(), N, "statsmodels SE length mismatch");

    // ---- compare -----------------------------------------------------------
    let maxabs = max_abs_diff(&gam_se_sandwich, ref_se);
    let corr = pearson(&gam_se_sandwich, ref_se);
    eprintln!(
        "ALO sandwich vs statsmodels OLS: n={N} p={P} edf={edf:.3} sigma2={:.6} \
         max_abs_diff={maxabs:.3e} pearson={corr:.10}",
        r.scalar("sigma2")
    );

    // Both engines compute the identical unpenalized matrix sigma^2 x' (X'X)^{-1} x.
    // The only sources of difference are the linear-algebra paths (gam's
    // StableSolver factorization vs statsmodels' pinv) and float rounding, so a
    // tight absolute bound is the correct expectation; a real divergence here
    // signals a bug in ALO's sandwich formula or its dispersion estimate.
    assert!(
        maxabs < 1e-8,
        "ALO sandwich SE diverges from statsmodels OLS prediction SE: max_abs_diff={maxabs:.3e}"
    );
    assert!(
        corr > 0.99999,
        "ALO sandwich SE shape disagrees with statsmodels: pearson={corr:.10}"
    );
}
