//! Cross-family regression guard for #679: the Negative-Binomial(log)
//! coefficient covariance `Vb` must NOT carry a spurious dispersion factor.
//!
//! ## Why a *separate* family test
//!
//! The #679 fix replaced a per-family covariance multiplier with the single
//! invariant "Vb = inverse of the penalized Hessian the solver actually
//! minimizes; restore only the dispersion the working weight does not already
//! carry." For Negative-Binomial with a known size `θ` that scale is `1.0`,
//! because NB(θ) is a one-parameter (mean) exponential family: its IRLS working
//! weight `W = prior·μθ/(θ+μ)` is the *complete* Fisher weight and `φ ≡ 1`.
//!
//! The OLD code instead multiplied `H⁻¹` by `θ` (the misread `dispersion_phi`
//! for NB), inflating every SE by `√θ` — a factor with no covariance meaning at
//! all (θ is the overdispersion size, not a scale). This is the most dramatic
//! arm of the cross-family double-count and had **zero** test coverage: the
//! existing `quality_vs_statsmodels_negbin` asserts mean recovery and a χ²/n
//! dispersion statistic, never the coefficient SE scale. With `θ = 3` the buggy
//! SEs were `√3 ≈ 1.73×` too large.
//!
//! ## The objective check
//!
//! On a PARAMETRIC NB(log) model (`y ~ x`, no smooth ⇒ `S_λ = 0`) gam's
//! coefficient covariance must equal the model-based reference exactly:
//! `Vb = H⁻¹ = (XᵀWX)⁻¹`. That is precisely statsmodels' `cov_params()` for
//! `GLM(family=NegativeBinomial(alpha=1/θ)).fit(scale=1.)` — the model-based
//! (Fisher) covariance for a known-size NB, the same convention mgcv's `nb()`
//! reports. We compare the per-row link-scale SE `SE(η_i) = √(xᵢᵀ Vb xᵢ)`. The
//! corrected gam matches statsmodels; the buggy `×θ` path lands at `√θ ≈ 1.73`.

use csv::StringRecord;
use gam::matrix::LinearOperator;
use gam::predict::{
    InferenceCovarianceMode, MeanIntervalMethod, PredictUncertaintyOptions,
    predict_gamwith_uncertainty,
};
use gam::smooth::build_term_collection_design;
use gam::test_support::reference::{Column, run_python};
use gam::types::{InverseLink, LikelihoodSpec, ResponseFamily, StandardLink};
use gam::{
    FitConfig, FitResult, encode_recordswith_inferred_schema, fit_from_formula, init_parallelism,
};
use ndarray::{Array1, Array2};
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand_distr::{Distribution, Gamma, Poisson, Uniform};

#[test]
fn negbin_log_coefficient_se_matches_statsmodels_no_theta_inflation() {
    init_parallelism();

    // ---- synthetic overdispersed counts (NB2, known size theta) ----------
    // truth: eta = 0.8 + 0.7*x;  mu = exp(eta);  y ~ NB2(mu, theta)
    // sampled as a Gamma-Poisson mixture: lambda ~ Gamma(theta, mu/theta),
    // y ~ Poisson(lambda) ⇒ E[y]=mu, Var[y]=mu + mu^2/theta.
    // theta = 3 makes the buggy SE inflation sqrt(theta) ~= 1.732 unmistakable.
    let n = 400usize;
    let seed = 911u64;
    let theta = 3.0_f64;
    let mut rng = StdRng::seed_from_u64(seed);
    let ux = Uniform::new(-1.0_f64, 1.0_f64).expect("uniform -1..1");

    let mut x = Vec::<f64>::with_capacity(n);
    let mut y = Vec::<f64>::with_capacity(n);
    for _ in 0..n {
        let xi = ux.sample(&mut rng);
        let eta = 0.8 + 0.7 * xi;
        let mu = eta.exp();
        let lam_gamma = Gamma::new(theta, mu / theta).expect("gamma(theta, mu/theta)");
        let lambda = lam_gamma.sample(&mut rng).max(1e-12);
        let pois = Poisson::new(lambda).expect("poisson(lambda)");
        let yi = pois.sample(&mut rng);
        x.push(xi);
        y.push(yi);
    }
    assert!(
        y.iter().all(|&v| v >= 0.0 && v.fract() == 0.0),
        "NB outcomes must be non-negative integers"
    );

    // ---- encode for gam ---------------------------------------------------
    let headers: Vec<String> = ["y", "x"].into_iter().map(String::from).collect();
    let rows: Vec<StringRecord> = (0..n)
        .map(|i| StringRecord::from(vec![(y[i] as i64).to_string(), x[i].to_string()]))
        .collect();
    let ds = encode_recordswith_inferred_schema(headers, rows).expect("encode negbin dataset");
    let col = ds.column_map();
    let x_idx = col["x"];

    // ---- fit gam: PARAMETRIC NB(log), y ~ x (no smooth ⇒ S=0), known theta ----
    let cfg = FitConfig {
        family: Some("negative-binomial".to_string()),
        negative_binomial_theta: Some(theta),
        ..FitConfig::default()
    };
    let result = fit_from_formula("y ~ x", &ds, &cfg).expect("gam parametric negbin fit");
    let FitResult::Standard(fit) = result else {
        panic!("expected a standard GAM fit for NegativeBinomial(log)");
    };

    // Rebuild gam's frozen design at the training points (log link: eta = X*beta).
    let mut grid = Array2::<f64>::zeros((n, ds.headers.len()));
    for i in 0..n {
        grid[[i, x_idx]] = x[i];
    }
    let design = build_term_collection_design(grid.view(), &fit.resolvedspec)
        .expect("rebuild design at training points");
    let dense = design.design.to_dense();
    let p = dense.ncols();
    assert_eq!(dense.nrows(), n, "design row count must match data");

    // gam's per-point linear-predictor SE SE(eta_i) = sqrt(xᵢᵀ Vb xᵢ).
    // `theta` was supplied to the fit as a held-fixed user value (via
    // `FitConfig::negative_binomial_theta`); the prediction-time spec must
    // mirror that contract so the IRLS weight `W = μθ/(θ+μ)` used to build
    // `Vb` is computed with the exact same overdispersion.
    let nb_log = LikelihoodSpec::new(
        ResponseFamily::NegativeBinomial {
            theta,
            theta_fixed: true,
        },
        InverseLink::Standard(StandardLink::Log),
    );
    let offset = Array1::<f64>::zeros(n);
    let pred = predict_gamwith_uncertainty(
        dense.clone(),
        fit.fit.beta.view(),
        offset.view(),
        nb_log,
        &fit.fit,
        &PredictUncertaintyOptions {
            confidence_level: 0.95,
            covariance_mode: InferenceCovarianceMode::Conditional,
            mean_interval_method: MeanIntervalMethod::Delta,
            includeobservation_interval: false,
            apply_bias_correction: false,
            edgeworth_one_sided: false,
            boundary_correction: false,
            ood_inflation: false,
            multi_point_joint: false,
            ..PredictUncertaintyOptions::default()
        },
    )
    .expect("gam negbin eta-SE prediction");
    let gam_eta_se = pred.eta_standard_error.to_vec();
    assert_eq!(gam_eta_se.len(), n, "one eta SE per training point");

    // ---- statsmodels GLM(NegativeBinomial(alpha=1/theta), log) on the SAME basis ----
    // Known-size NB is a one-parameter (mean) exponential family ⇒ the
    // model-based covariance uses scale = 1: cov_params() = (XᵀWX)⁻¹. We force
    // scale=1. so the comparison is to the Fisher covariance gam's corrected Vb
    // reproduces (the default Pearson scale would itself rescale the SEs).
    let design_cols: Vec<Vec<f64>> = (0..p)
        .map(|j| {
            let mut e = Array1::<f64>::zeros(p);
            e[j] = 1.0;
            design.design.apply(&e).to_vec()
        })
        .collect();
    let p_col = vec![p as f64; n];
    let theta_col = vec![theta; n];

    let mut columns: Vec<Column<'_>> = Vec::with_capacity(p + 3);
    columns.push(Column::new("y", &y));
    columns.push(Column::new("p", &p_col));
    columns.push(Column::new("theta", &theta_col));
    let col_names: Vec<String> = (0..p).map(|j| format!("d{j}")).collect();
    for j in 0..p {
        columns.push(Column::new(&col_names[j], &design_cols[j]));
    }

    let r = run_python(
        &columns,
        r#"
import numpy as np
import statsmodels.api as sm

n = len(df["y"])
p = int(df["p"][0])
theta = float(df["theta"][0])
X = np.column_stack([np.asarray(df["d%d" % j], dtype=float) for j in range(p)])
yv = np.asarray(df["y"], dtype=float)

# gam's basis already includes its own intercept column, so do NOT add one.
# Known-size NB2: alpha = 1/theta. Force scale=1 for the model-based (Fisher)
# covariance that a correctly-specified known-theta NB reports.
fam = sm.families.NegativeBinomial(alpha=1.0 / theta)
model = sm.GLM(yv, X, family=fam)
res = model.fit(scale=1.0)

cov = np.asarray(res.cov_params(), dtype=float)
eta_se = np.sqrt(np.einsum("ij,jk,ik->i", X, cov, X))
emit("eta_se", eta_se)
emit("scale", [res.scale])
"#,
    );
    let ref_eta_se = r.vector("eta_se");
    let sm_scale = r.scalar("scale");
    assert_eq!(ref_eta_se.len(), n, "statsmodels eta SE length mismatch");

    // ---- OBJECTIVE assertion: SE ratio ~ 1 (NOT sqrt(theta)) ---------------
    let mut ratios: Vec<f64> = gam_eta_se
        .iter()
        .zip(ref_eta_se.iter())
        .filter(|&(_, &s)| s > 0.0)
        .map(|(&g, &s)| g / s)
        .collect();
    assert!(!ratios.is_empty(), "no positive reference SEs to compare");
    ratios.sort_by(|a, b| a.partial_cmp(b).expect("finite SE ratio"));
    let median_ratio = ratios[ratios.len() / 2];

    let buggy_ratio = theta.sqrt(); // ~1.732: the theta-inflation signature

    eprintln!(
        "negbin(log) coef SE vs statsmodels: n={n} p={p} theta={theta:.3} \
         sm_scale={sm_scale:.4} median(gam_eta_se/sm_eta_se)={median_ratio:.4} \
         (correct ~= 1.0; buggy theta-inflation ~= {buggy_ratio:.4})"
    );

    // PRIMARY: gam's coefficient-covariance scale matches the model-based
    // reference. 8% tolerance absorbs the tiny ridge (1e-6) gam adds and the
    // slight leverage spread — far tighter than the 73% inflation the bug gives.
    assert!(
        (median_ratio - 1.0).abs() < 0.08,
        "NB(log) coefficient SE off from statsmodels by >8%: median ratio \
         {median_ratio:.4} (expected ~1.0; the #679 theta-inflation would give \
         ~{buggy_ratio:.4})"
    );

    // GUARD: explicitly reject the theta-inflation signature.
    assert!(
        (median_ratio - 1.0).abs() < (median_ratio - buggy_ratio).abs(),
        "NB(log) SE ratio {median_ratio:.4} is closer to the #679 theta-inflation \
         value {buggy_ratio:.4} than to the correct 1.0 — Vb carries a spurious θ"
    );
}
