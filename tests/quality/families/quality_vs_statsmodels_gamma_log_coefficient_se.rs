//! Regression guard for #679: the Gamma(log) coefficient covariance `Vb` must
//! NOT double-count the dispersion.
//!
//! ## What was wrong
//!
//! The Gamma-log IRLS working weight already carries the shape `ν = 1/φ`
//! (`W_i = prior_i · ν`), while the penalty is added to the Hessian unscaled, so
//! the stored penalized Hessian is `H = ν·XᵀΛX + S_λ` — exactly mgcv's effective
//! penalized Hessian `XᵀW_sfX/φ + S_λ`. The correct coefficient covariance is
//! therefore `Vb = H⁻¹`. The old code multiplied by `φ̂ = 1/shape` a second time
//! (`Vb = H⁻¹·φ`), shrinking every standard error by `√φ = 1/√shape`. No prior
//! Gamma test caught this — they assert mean/η recovery and a Pearson dispersion
//! statistic, never the SE/CI scale.
//!
//! ## The objective check
//!
//! On a PARAMETRIC Gamma(log) model (`y ~ x + z`, no smooth ⇒ `S_λ = 0`) gam's
//! coefficient covariance must equal the mature reference's exactly: with no
//! penalty `Vb = H⁻¹ = φ·(XᵀW_sfX)⁻¹`, which is precisely statsmodels'
//! `cov_params() = scale·(XᵀW_sfX)⁻¹` (Pearson `scale = φ̂`) on the identical
//! frozen basis. We compare the per-point linear-predictor standard error
//! `SE(η_i) = √(xᵢᵀ Vb xᵢ)` — the quantity that flows straight into every CI —
//! against statsmodels' link-scale SE `√(xᵢᵀ·cov_params·xᵢ)`. The corrected
//! gam matches statsmodels to a tight tolerance; the BUGGY gam would come out a
//! factor `√shape ≈ 1.6` too small, which this test rejects.

use csv::StringRecord;
use gam::matrix::LinearOperator;
use gam::smooth::build_term_collection_design;
use gam::test_support::reference::{Column, run_python};
use gam::types::{InverseLink, LikelihoodSpec, ResponseFamily, StandardLink};
use gam::{
    FitConfig, FitResult, encode_recordswith_inferred_schema, fit_from_formula, init_parallelism,
};
use gam_predict::{
    InferenceCovarianceMode, MeanIntervalMethod, PredictUncertaintyOptions,
    predict_gamwith_uncertainty,
};
use ndarray::{Array1, Array2};
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand_distr::{Distribution, Gamma, Uniform};

#[test]
fn gamma_log_coefficient_se_matches_statsmodels_no_double_count() {
    init_parallelism();

    // ---- synthetic positive-continuous data (canonical Gamma parametrization) ----
    // truth: eta = 1.5 + 0.6*x - 0.4*z;  y ~ Gamma(shape, scale=exp(eta)/shape)
    // so E[y] = shape*scale = exp(eta), Var(y) = mu^2/shape (=> phi = 1/shape).
    // shape = 2.5 makes the buggy SE deficit sqrt(1/shape) ~= 0.632 unmistakable.
    let n = 240usize;
    let seed = 781u64;
    let shape = 2.5_f64;
    let mut rng = StdRng::seed_from_u64(seed);
    let ux = Uniform::new(-1.0_f64, 1.0_f64).expect("uniform -1..1");

    let mut x = Vec::<f64>::with_capacity(n);
    let mut z = Vec::<f64>::with_capacity(n);
    let mut y = Vec::<f64>::with_capacity(n);
    for _ in 0..n {
        let xi = ux.sample(&mut rng);
        let zi = ux.sample(&mut rng);
        let eta: f64 = 1.5 + 0.6 * xi - 0.4 * zi;
        let scale = eta.exp() / shape; // shape*scale = exp(eta) = E[y]
        let g = Gamma::new(shape, scale).expect("gamma(shape,scale)");
        let yi = g.sample(&mut rng);
        x.push(xi);
        z.push(zi);
        y.push(yi);
    }
    assert!(
        y.iter().all(|&v| v > 0.0),
        "Gamma outcomes must be positive"
    );

    // ---- encode for gam ---------------------------------------------------
    let headers: Vec<String> = ["y", "x", "z"].into_iter().map(String::from).collect();
    let rows: Vec<StringRecord> = (0..n)
        .map(|i| StringRecord::from(vec![y[i].to_string(), x[i].to_string(), z[i].to_string()]))
        .collect();
    let ds = encode_recordswith_inferred_schema(headers, rows).expect("encode gamma dataset");
    let col = ds.column_map();
    let x_idx = col["x"];
    let z_idx = col["z"];

    // ---- fit with gam: PARAMETRIC Gamma(log), y ~ x + z (no smooth ⇒ S=0) ----
    let cfg = FitConfig {
        family: Some("gamma".to_string()),
        ..FitConfig::default()
    };
    let result = fit_from_formula("y ~ x + z", &ds, &cfg).expect("gam parametric gamma fit");
    let FitResult::Standard(fit) = result else {
        panic!("expected a standard GAM fit for Gamma(log)");
    };

    // Rebuild gam's frozen design at the training points (log link: eta = X*beta).
    let mut grid = Array2::<f64>::zeros((n, ds.headers.len()));
    for i in 0..n {
        grid[[i, x_idx]] = x[i];
        grid[[i, z_idx]] = z[i];
    }
    let design = build_term_collection_design(grid.view(), &fit.resolvedspec)
        .expect("rebuild design at training points");
    let dense = design.design.to_dense();
    let p = dense.ncols();
    assert_eq!(dense.nrows(), n, "design row count must match data");

    // gam's per-point linear-predictor standard error SE(eta_i) = sqrt(xᵢᵀ Vb xᵢ),
    // conditional on the (here trivial) smoothing parameters — Vb is exactly the
    // coefficient covariance the fix corrects.
    let gamma_log = LikelihoodSpec::new(
        ResponseFamily::Gamma,
        InverseLink::Standard(StandardLink::Log),
    );
    let offset = Array1::<f64>::zeros(n);
    let pred = predict_gamwith_uncertainty(
        dense.clone(),
        fit.fit.beta.view(),
        offset.view(),
        gamma_log,
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
    .expect("gam gamma eta-SE prediction");
    let gam_eta_se = pred.eta_standard_error.to_vec();
    assert_eq!(gam_eta_se.len(), n, "one eta SE per training point");

    // ---- statsmodels GLM(Gamma, log) on the IDENTICAL frozen basis ----------
    // gam's design already carries its own intercept column, so we fit with no
    // extra constant. statsmodels' linear-predictor prediction SE uses
    // cov_params() = scale * (XᵀW X)⁻¹ with Pearson scale = φ̂ — the
    // mgcv/statsmodels convention gam's corrected Vb must reproduce.
    let design_cols: Vec<Vec<f64>> = (0..p)
        .map(|j| {
            let mut e = Array1::<f64>::zeros(p);
            e[j] = 1.0;
            design.design.apply(&e).to_vec()
        })
        .collect();
    let p_col = vec![p as f64; n];

    let mut columns: Vec<Column<'_>> = Vec::with_capacity(p + 2);
    columns.push(Column::new("y", &y));
    columns.push(Column::new("p", &p_col));
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
X = np.column_stack([np.asarray(df["d%d" % j], dtype=float) for j in range(p)])
yv = np.asarray(df["y"], dtype=float)

# gam's basis already includes its own intercept column, so do NOT add one.
model = sm.GLM(yv, X, family=sm.families.Gamma(link=sm.families.links.Log()))
res = model.fit()

# Linear-predictor (link-scale) standard error per training row, computed
# straight from the coefficient covariance to stay version-independent:
#   SE(eta_i) = sqrt(x_i' cov_params x_i).
# statsmodels' cov_params() = scale * (X'WX)^-1 with the scale-free Gamma
# Fisher weight and Pearson scale = phi_hat — the mgcv/statsmodels convention
# gam's corrected Vb must reproduce.
cov = np.asarray(res.cov_params(), dtype=float)
eta_se = np.sqrt(np.einsum("ij,jk,ik->i", X, cov, X))
emit("eta_se", eta_se)
emit("scale", [res.scale])
"#,
    );
    let ref_eta_se = r.vector("eta_se");
    let sm_scale = r.scalar("scale");
    assert_eq!(ref_eta_se.len(), n, "statsmodels eta SE length mismatch");

    // ---- OBJECTIVE assertion: SE ratio ~ 1 (NOT 1/sqrt(shape)) -------------
    // Median ratio is robust to the few high-leverage rows. The corrected gam
    // reproduces statsmodels' link-scale SE on the shared unpenalized basis; the
    // buggy path would land at sqrt(1/shape) ~= 0.632.
    let mut ratios: Vec<f64> = gam_eta_se
        .iter()
        .zip(ref_eta_se.iter())
        .filter(|&(_, &s)| s > 0.0)
        .map(|(&g, &s)| g / s)
        .collect();
    assert!(!ratios.is_empty(), "no positive reference SEs to compare");
    ratios.sort_by(|a, b| a.partial_cmp(b).expect("finite SE ratio"));
    let median_ratio = ratios[ratios.len() / 2];

    let buggy_ratio = (1.0_f64 / shape).sqrt(); // ~0.632: the double-count signature

    eprintln!(
        "gamma(log) coef SE vs statsmodels: n={n} p={p} shape={shape:.3} \
         sm_scale={sm_scale:.4} median(gam_eta_se/sm_eta_se)={median_ratio:.4} \
         (correct ~= 1.0; buggy double-count ~= {buggy_ratio:.4})"
    );

    // PRIMARY: gam's coefficient-covariance scale matches the mature reference.
    // 6% tolerance absorbs the Pearson-vs-deviance scale-estimate difference
    // (statsmodels uses Pearson φ̂; gam profiles the Gamma shape) plus the slight
    // intercept/leverage spread — far tighter than the ~37% deficit the bug
    // would produce.
    assert!(
        (median_ratio - 1.0).abs() < 0.06,
        "Gamma(log) coefficient SE off from statsmodels by >6%: \
         median ratio {median_ratio:.4} (expected ~1.0; the #679 double-count \
         would give ~{buggy_ratio:.4})"
    );

    // GUARD: explicitly reject the double-count signature — the ratio must be
    // unambiguously closer to the correct 1.0 than to the buggy 1/sqrt(shape).
    assert!(
        (median_ratio - 1.0).abs() < (median_ratio - buggy_ratio).abs(),
        "Gamma(log) SE ratio {median_ratio:.4} is closer to the #679 double-count \
         value {buggy_ratio:.4} than to the correct 1.0 — Vb is double-counting φ"
    );
}
