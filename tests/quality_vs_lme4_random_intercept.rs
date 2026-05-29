//! End-to-end quality: gam's random-intercept term must match `lme4::lmer` —
//! the de-facto standard for mixed-effects variance-component estimation in R —
//! on a synthetic random-intercept design.
//!
//! `lme4::lmer(y ~ 1 + (1|g))` fits a Gaussian linear mixed model by REML and
//! reports (a) the random-intercept variance σ_g² and residual variance σ_ε²
//! (`VarCorr`/`sigma`) and (b) the per-group conditional modes (BLUPs,
//! `ranef`). gam expresses the same structure via `group(g)` / `s(g, bs="re")`:
//! the random intercept is a ridge-penalized factor block whose penalty is
//! selected by REML, exactly the criterion lme4 optimizes. With a smooth main
//! effect added, the gam model fit here is
//!
//!     y ~ s(x) + group(g)
//!
//! which exercises the cross-feature combination that matters: a penalized
//! smooth additive with a random intercept (a by/RE-x-smooth interaction of
//! penalized blocks). Both engines target the same REML objective, so they must
//! agree on:
//!   1. the per-group predicted intercepts (Pearson r vs lme4 conditional
//!      modes — the *shape* of the group effects), and
//!   2. the estimated variance components σ_g² and σ_ε².
//!
//! The data is generated once and handed *identically* to gam and to lme4. A
//! genuine divergence here is a real bug in gam's random-effect machinery, not
//! something to paper over by loosening the bound.

use gam::matrix::LinearOperator;
use gam::smooth::build_term_collection_design;
use gam::test_support::reference::{Column, pearson, run_r};
use gam::{
    FitConfig, FitResult, encode_recordswith_inferred_schema, fit_from_formula, init_parallelism,
};
use csv::StringRecord;
use ndarray::Array2;
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand_distr::{Distribution, Normal, Uniform};

const N_GROUPS: usize = 8;
const PER_GROUP: usize = 60;
// True group intercepts μ_g (population draws fixed by the spec). Their sample
// variance is the target σ_g² that lme4 estimates from the data.
const MU_G: [f64; N_GROUPS] = [-1.5, -0.9, -0.4, 0.0, 0.3, 0.7, 1.1, 1.5];
// Residual noise: ε ~ N(mean=0, variance=0.25) ⇒ standard deviation 0.5.
// rand_distr::Normal is parameterised by (mean, std_dev), so we pass 0.5; the
// residual *variance* σ_ε² we compare against is therefore 0.25.
const RESID_SD: f64 = 0.5;
const RESID_VAR: f64 = RESID_SD * RESID_SD;
const SEED: u64 = 42;

#[test]
fn gam_random_intercept_matches_lme4() {
    init_parallelism();

    // ---- synthesize the random-intercept dataset --------------------------
    // Rows are emitted group-blocked (all of g0, then g1, …). Group labels are
    // strings ("g0".."g7") so the schema inferrer treats `g` as categorical;
    // first-appearance order then makes the encoded level index equal the group
    // number, which we rely on when rebuilding the design at group anchors.
    let n = N_GROUPS * PER_GROUP;
    let mut rng = StdRng::seed_from_u64(SEED);
    let ux = Uniform::new(0.0, 1.0).expect("uniform");
    let noise = Normal::new(0.0, RESID_SD).expect("normal");
    let two_pi = 2.0 * std::f64::consts::PI;

    let mut x = Vec::<f64>::with_capacity(n);
    let mut y = Vec::<f64>::with_capacity(n);
    let mut g_code = Vec::<f64>::with_capacity(n); // numeric group index for lme4
    let mut rows = Vec::<StringRecord>::with_capacity(n);
    for grp in 0..N_GROUPS {
        for _ in 0..PER_GROUP {
            let xi = ux.sample(&mut rng);
            let yi = two_pi * xi.sin() + MU_G[grp] + noise.sample(&mut rng);
            x.push(xi);
            y.push(yi);
            g_code.push(grp as f64);
            rows.push(StringRecord::from(vec![
                format!("{xi}"),
                format!("g{grp}"),
                format!("{yi}"),
            ]));
        }
    }

    let headers = vec!["x".to_string(), "g".to_string(), "y".to_string()];
    let ds = encode_recordswith_inferred_schema(headers, rows).expect("encode RE dataset");
    let col = ds.column_map();
    let x_idx = col["x"];
    let g_idx = col["g"];

    // ---- fit with gam: y ~ s(x) + group(g), REML --------------------------
    let cfg = FitConfig {
        family: Some("gaussian".to_string()),
        ..FitConfig::default()
    };
    let result = fit_from_formula("y ~ s(x) + group(g)", &ds, &cfg).expect("gam RE fit");
    let FitResult::Standard(fit) = result else {
        panic!("expected a standard GAM fit for a Gaussian random-intercept model");
    };
    // gam's residual standard deviation on the response scale (Gaussian identity
    // stores σ_ε here per the UnifiedFitResult contract).
    let gam_resid_var = fit.fit.standard_deviation * fit.fit.standard_deviation;

    // Predicted per-group intercept: evaluate the fitted model at a *single*
    // common x reference for every group. The s(x) contribution and the global
    // intercept are then identical across the 8 rows, so the row-to-row spread
    // isolates the estimated group effect (the gam BLUP). We use the mean x as
    // the reference (well inside the data support, no extrapolation).
    let x_ref = x.iter().sum::<f64>() / n as f64;
    let mut anchor = Array2::<f64>::zeros((N_GROUPS, ds.headers.len()));
    for grp in 0..N_GROUPS {
        anchor[[grp, x_idx]] = x_ref;
        // Encoded categorical level index equals the group number (see above).
        anchor[[grp, g_idx]] = grp as f64;
    }
    let anchor_design = build_term_collection_design(anchor.view(), &fit.resolvedspec)
        .expect("rebuild design at group anchors");
    let gam_group_pred: Vec<f64> = anchor_design.design.apply(&fit.fit.beta).to_vec();
    // Centre to per-group *deviations* (mean-zero), matching lme4 conditional
    // modes which are deviations from the fixed-effect intercept.
    let gam_mean = gam_group_pred.iter().sum::<f64>() / N_GROUPS as f64;
    let gam_dev: Vec<f64> = gam_group_pred.iter().map(|v| v - gam_mean).collect();
    // Sample variance of the predicted group deviations is gam's estimate of
    // σ_g². With 60 observations/group and σ_ε²=0.25 the BLUP shrinkage factor
    // σ_g²/(σ_g²+σ_ε²/60) ≈ 0.99, so this is an essentially unbiased read of the
    // variance component and the apples-to-apples match for lme4's VarCorr.
    let gam_sigma_g2 =
        gam_dev.iter().map(|v| v * v).sum::<f64>() / (N_GROUPS as f64 - 1.0);

    // ---- fit the SAME model with lme4 (the mature reference) ---------------
    // lme4 separates fixed and random parts: the smooth main effect of x is a
    // fixed-effect natural spline (df matched to a typical s(x) edf), and the
    // random intercept is (1|g). VarCorr gives σ_g², sigma² gives σ_ε², and
    // ranef gives the per-group conditional modes in factor-level order g0..g7.
    let r = run_r(
        &[
            Column::new("x", &x),
            Column::new("g", &g_code),
            Column::new("y", &y),
        ],
        r#"
        suppressPackageStartupMessages(library(lme4))
        suppressPackageStartupMessages(library(splines))
        df$g <- factor(df$g, levels = as.character(sort(unique(df$g))))
        m <- lmer(y ~ ns(x, df = 6) + (1 | g), data = df, REML = TRUE)
        vc <- as.data.frame(VarCorr(m))
        sigma_g2 <- vc$vcov[vc$grp == "g"]
        sigma_e2 <- vc$vcov[vc$grp == "Residual"]
        re <- ranef(m)$g[, "(Intercept)"]
        emit("sigma_g2", sigma_g2)
        emit("sigma_e2", sigma_e2)
        emit("ranef", as.numeric(re))
        "#,
    );
    let lme4_sigma_g2 = r.scalar("sigma_g2");
    let lme4_sigma_e2 = r.scalar("sigma_e2");
    let lme4_ranef = r.vector("ranef");
    assert_eq!(
        lme4_ranef.len(),
        N_GROUPS,
        "lme4 returned {} conditional modes, expected {N_GROUPS}",
        lme4_ranef.len()
    );

    // ---- compare ----------------------------------------------------------
    let corr = pearson(&gam_dev, lme4_ranef);
    let var_g_rel = (gam_sigma_g2 - lme4_sigma_g2).abs() / lme4_sigma_g2.abs().max(1e-12);
    let var_e_rel = (gam_resid_var - lme4_sigma_e2).abs() / lme4_sigma_e2.abs().max(1e-12);

    eprintln!(
        "random-intercept vs lme4: n={n} groups={N_GROUPS} \
         pearson(group effects)={corr:.5} \
         sigma_g2 gam={gam_sigma_g2:.4} lme4={lme4_sigma_g2:.4} rel={var_g_rel:.4} \
         sigma_e2 gam={gam_resid_var:.4} lme4={lme4_sigma_e2:.4} (truth {RESID_VAR:.4}) rel={var_e_rel:.4}"
    );

    // (1) The per-group effects are well separated (8 distinct μ_g spanning
    // [-1.5, 1.5]) and far above the per-group noise floor (σ_ε²/60 ≈ 0.004), so
    // both REML engines recover essentially the same BLUP ordering and
    // magnitudes. Pearson r > 0.99 is the principled bound: anything lower means
    // gam's random-intercept shrinkage/estimation genuinely disagrees with the
    // mixed-model standard.
    assert!(
        corr > 0.99,
        "gam per-group intercepts disagree with lme4 conditional modes: pearson={corr:.5}"
    );
    // (2) Both fit residual variance by REML on the same data; σ_ε² is the
    // best-determined component (n=480 residual d.f.), so it must match tightly.
    assert!(
        var_e_rel < 0.10,
        "residual variance disagrees with lme4: gam={gam_resid_var:.4} lme4={lme4_sigma_e2:.4} (rel={var_e_rel:.4})"
    );
    // (3) The random-intercept variance is estimated from only 8 groups, so it
    // is the noisiest component for *both* engines; they nonetheless target the
    // same REML estimand. 0.10 relative is the principled bound — both engines
    // see the identical 8 group means, and shrinkage is ~1% at 60 obs/group, so
    // a larger gap signals a real divergence in the variance-component machinery.
    assert!(
        var_g_rel < 0.10,
        "random-intercept variance disagrees with lme4: gam={gam_sigma_g2:.4} lme4={lme4_sigma_g2:.4} (rel={var_g_rel:.4})"
    );
}
