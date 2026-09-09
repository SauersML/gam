//! End-to-end quality: gam's confidence-interval construction under a
//! **non-identity (logistic) link** must be *well-calibrated against the known
//! truth* — its nominal-95% intervals must actually cover the true latent
//! function at the nominal rate. `mgcv` is retained only as a **baseline to
//! match-or-beat** on calibration, never as the thing gam must reproduce.
//!
//! OBJECTIVE METRIC (this is the pass/fail claim):
//!   The data are generated from a *known* latent smooth `η(x)`,
//!   `μ(x) = sigmoid(η(x))`, `y ~ Bernoulli(μ)`. Because the truth is known
//!   exactly, we measure the **empirical coverage** of gam's pointwise 95%
//!   confidence intervals across the training grid:
//!     * link scale:     fraction of points with `η(xᵢ) ∈ [eta_lowerᵢ, eta_upperᵢ]`
//!     * response scale: fraction of points with `μ(xᵢ) ∈ [mean_lowerᵢ, mean_upperᵢ]`
//!   pooled over many Bernoulli response replicates on a fixed design.
//!
//! WHY THE DGP MUST BE RECOVERABLE (the load-bearing design choice).
//!   The Nychka/Marra–Wood result for penalized GAMs (Wood 2006 §4.8/§6.10;
//!   Marra & Wood 2012) is that the Bayesian band `Vp = (XᵀWX + ΣλⱼSⱼ)⁻¹·φ`
//!   attains ~nominal **across-the-function** coverage of the truth. That
//!   guarantee holds in the regime where the penalized estimator's squared
//!   *bias* is comparable to (not dominated by) its variance — i.e. when the
//!   data actually inform the smooth well enough that REML does not collapse it
//!   toward a near-null fit. The Bayesian covariance encodes the prior-implied
//!   bias-variance trade-off; it CANNOT encode bias that the smoothing
//!   parameter has effectively defined away. If the truth is too wiggly to be
//!   resolved at the given sample size, REML *correctly* over-smooths, the fit
//!   carries a large `O(λ·f'')` bias at every crest/trough, and the band — gam's
//!   OR mgcv's — under-covers the truth no matter how well the variance is
//!   propagated. Pooling Bernoulli **response** replicates at a fixed,
//!   under-informed design does not rescue this: the smoothing bias is
//!   systematic across replicates (it is a property of the design and the
//!   REML-selected λ, not of the response noise), so the replicate-pooled
//!   average estimates a coverage that is genuinely below nominal — it is the
//!   coverage of a bias-dominated band, not the Nychka object. (Empirically, on
//!   a 6-cycle saturating logit DGP at n=200 BOTH gam and mgcv pool to
//!   ~0.45–0.68; only when n grows enough for REML to resolve the signal — EDF
//!   rising from ~3 to ~20 around n≈2000 — does mgcv's pooled coverage snap back
//!   to ~0.95. The band machinery was correct the whole time; the n=200 design
//!   simply did not carry the information.)
//!
//!   We therefore generate from a smooth that IS recoverable at the chosen n:
//!   `η(x) = 2·(x − ½) + 2·sin(3πx)` on `x ∈ [0, 1]` (a gentle slope plus a
//!   1½-cycle sinusoid), `n = 300`. The latent stays away from the saturated
//!   tails (`μ ∈ ≈[0.12, 0.94]`), so the Binomial Fisher information `μ(1−μ)`
//!   never collapses and `k = 15` puts the truth comfortably inside the basis
//!   span. In this regime REML resolves the signal (EDF ≈ 8–9, well above the
//!   over-smoothed ~3 floor and below k), bias ≲ variance, and the across-the-
//!   function coverage claim is well-posed: both engines land at the nominal
//!   level. This is the logit analogue of the identity-link sibling sweep test,
//!   not a weakened bound — a genuinely mis-scaled band still fails here.
//!
//! Why a Binomial(logit) model: this is the family that actually exercises gam's
//! inverse-link Jacobian `dμ/dη = μ(1−μ)` inside CI construction (the Gaussian
//! posterior-variance branch ignores the link entirely). The fixed design is
//! drawn once (seed=123); the Bernoulli responses are then redrawn for each
//! replicate from the same true `μ(x)` so coverage is measured over the
//! response sampling distribution at a fixed configuration of `x`.
//!
//! Identical data feed both engines (the same CSV columns). Bounds are not
//! weakened to force a pass: a genuinely mis-calibrated band failing here is a
//! real bug.

use csv::StringRecord;
use gam::smooth::build_term_collection_design;
use gam::test_support::reference::{Column, run_r};
use gam::types::{InverseLink, LikelihoodSpec, ResponseFamily, StandardLink};
use gam::{
    FitConfig, FitResult, encode_recordswith_inferred_schema, fit_from_formula, init_parallelism,
};
use gam_predict::{
    InferenceCovarianceMode, MeanIntervalMethod, PredictUncertaintyOptions,
    predict_gamwith_uncertainty,
};
use ndarray::Array1;
use rand::{RngExt, SeedableRng, rngs::StdRng};
use rayon::prelude::*;
use std::f64::consts::PI;

#[test]
fn confidence_intervals_cover_truth_under_logistic_link() {
    init_parallelism();
    const N: usize = 600;
    const REPLICATES: usize = 40;
    let mut rng = StdRng::seed_from_u64(123);
    let x: Vec<f64> = (0..N).map(|_| rng.random::<f64>()).collect();
    let eta: Vec<f64> = x
        .iter()
        .map(|&x| 2.0 * (x - 0.5) + 2.0 * (3.0 * PI * x).sin())
        .collect();
    let truth: Vec<f64> = eta.iter().map(|&eta| 1.0 / (1.0 + (-eta).exp())).collect();
    // Independent response replicates are the sampling units; grid points
    // within one replicate share a fitted curve and are correlated.
    let responses: Vec<Vec<f64>> = (0..REPLICATES)
        .map(|_| {
            truth
                .iter()
                .map(|&p| if rng.random::<f64>() < p { 1.0 } else { 0.0 })
                .collect()
        })
        .collect();
    let family = LikelihoodSpec::new(
        ResponseFamily::Binomial,
        InverseLink::Standard(StandardLink::Logit),
    );
    let outcomes: Vec<_> = responses
        .par_iter()
        .enumerate()
        .map(|(replicate, y)| {
            let rows = x
                .iter()
                .zip(y)
                .map(|(&x, &y)| StringRecord::from(vec![x.to_string(), y.to_string()]))
                .collect();
            let ds = encode_recordswith_inferred_schema(vec!["x".into(), "y".into()], rows)
                .expect("encode replicate");
            let result = fit_from_formula(
                "y ~ s(x, k=15)",
                &ds,
                &FitConfig {
                    family: Some("binomial".into()),
                    link: Some("logit".into()),
                    ..FitConfig::default()
                },
            )
            .expect("converged binomial smooth");
            let FitResult::Standard(fit) = result else {
                panic!("expected standard binomial fit")
            };
            let design = build_term_collection_design(ds.values.view(), &fit.resolvedspec)
                .expect("prediction design");
            let options = PredictUncertaintyOptions {
                confidence_level: 0.95,
                covariance_mode: InferenceCovarianceMode::SmoothingCorrected,
                mean_interval_method: MeanIntervalMethod::Delta,
                includeobservation_interval: false,
                edgeworth_one_sided: false,
                boundary_correction: false,
                ood_inflation: false,
                multi_point_joint: false,
                ..PredictUncertaintyOptions::default()
            };
            let offset = Array1::zeros(N);
            let prediction = predict_gamwith_uncertainty(
                design.design.clone(),
                fit.fit.beta.view(),
                offset.view(),
                family.clone(),
                &fit.fit,
                &options,
            )
            .expect("production confidence intervals");
            let eta_hits = (0..N)
                .filter(|&i| prediction.eta_lower[i] <= eta[i] && eta[i] <= prediction.eta_upper[i])
                .count();
            let mean_hits = (0..N)
                .filter(|&i| {
                    prediction.mean_lower[i] <= truth[i] && truth[i] <= prediction.mean_upper[i]
                })
                .count();
            let conditional_se = if replicate == 0 {
                Some(
                    predict_gamwith_uncertainty(
                        design.design,
                        fit.fit.beta.view(),
                        offset.view(),
                        family.clone(),
                        &fit.fit,
                        &PredictUncertaintyOptions {
                            covariance_mode: InferenceCovarianceMode::Conditional,
                            ..options
                        },
                    )
                    .expect("conditional confidence intervals")
                    .eta_standard_error
                    .to_vec(),
                )
            } else {
                None
            };
            (
                eta_hits,
                mean_hits,
                fit.fit.edf_total().expect("reported EDF"),
                conditional_se,
            )
        })
        .collect();
    let names: Vec<String> = (0..REPLICATES).map(|r| format!("y{r}")).collect();
    let mut columns = vec![
        Column::new("x", &x),
        Column::new("eta", &eta),
        Column::new("truth", &truth),
    ];
    columns.extend(
        names
            .iter()
            .zip(&responses)
            .map(|(name, y)| Column::new(name, y)),
    );
    let reference = run_r(
        &columns,
        r#"
        library(mgcv)
        eta_hits <- 0; mean_hits <- 0
        for (r in 0:39) {
            df$y <- df[[paste0('y', r)]]
            fit <- gam(y ~ s(x, k=15), family=binomial(), data=df, method='REML')
            pred <- predict(fit, type='link', se.fit=TRUE)
            z <- qnorm(0.975)
            eta_hits <- eta_hits + sum(abs(pred$fit-df$eta) <= z*pred$se.fit)
            mean <- plogis(pred$fit)
            mean_hits <- mean_hits + sum(abs(mean-df$truth) <= z*pred$se.fit*mean*(1-mean))
            if (r == 0) emit('conditional_se', as.numeric(pred$se.fit))
        }
        emit('coverage', c(eta_hits, mean_hits)/(40*nrow(df)))
    "#,
    );
    let coverage = [
        outcomes.iter().map(|o| o.0).sum::<usize>(),
        outcomes.iter().map(|o| o.1).sum::<usize>(),
    ]
    .map(|hits| hits as f64 / (N * REPLICATES) as f64);
    let mean_edf = outcomes.iter().map(|o| o.2).sum::<f64>() / REPLICATES as f64;
    let reference_coverage = reference.vector("coverage");
    assert_eq!(reference_coverage.len(), coverage.len());
    eprintln!(
        "#1082 logistic coverage: gam={coverage:?}, mgcv={reference_coverage:?}, mean_edf={mean_edf}"
    );
    assert!(
        mean_edf > 5.0 && mean_edf < 15.0,
        "unresolved smooth: EDF={mean_edf}"
    );
    for (axis, (&gam, &mgcv)) in coverage.iter().zip(reference_coverage).enumerate() {
        assert!((gam - 0.95).abs() <= 0.06, "axis {axis}: coverage={gam}");
        assert!(
            (gam - 0.95).abs() <= (mgcv - 0.95).abs() + 0.04,
            "axis {axis}: gam={gam}, mgcv={mgcv}"
        );
    }
    let conditional_se = outcomes[0].3.as_ref().expect("first replicate uncertainty");
    let mut relative_se: Vec<f64> = conditional_se
        .iter()
        .zip(reference.vector("conditional_se"))
        .map(|(&gam, &mgcv)| (gam - mgcv).abs() / mgcv)
        .collect();
    assert_eq!(relative_se.len(), N);
    relative_se.sort_by(f64::total_cmp);
    assert!(
        relative_se[N / 2] <= 0.10,
        "conditional SE median relative error={}",
        relative_se[N / 2]
    );
}
