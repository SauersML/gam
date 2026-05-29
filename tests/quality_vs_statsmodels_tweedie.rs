//! End-to-end quality: gam's Tweedie (compound Poisson-Gamma, power-variance)
//! family must agree with **statsmodels** — the mature, standard Python GLM
//! implementation — on the same insurance-style zero-inflated positive data.
//!
//! Tweedie with `p ∈ (1, 2)` is the bridge between Poisson (`p = 1`) and Gamma
//! (`p = 2`): `Var(y) = φ·μ^p`. It is the workhorse family for zero-inflated
//! non-negative data (insurance claim totals, rainfall, biomass). `p = 1.5` is
//! the canonical semi-Poisson case and is mgcv's default `tw()` power. gam
//! fixes the Tweedie link to `log`, exactly matching
//! `statsmodels.api.GLM(family=Tweedie(var_power=1.5, link=log()))`.
//!
//! To make the comparison element-wise honest we hand **statsmodels the exact
//! same basis** gam builds: gam constructs the penalized design for
//! `y ~ s(x1, k=4) + s(x2, k=4) + linear(offset)`, and we ship that dense
//! design matrix straight into a plain statsmodels Tweedie GLM. Both engines
//! then maximize the identical Tweedie (`p = 1.5`, log-link) likelihood over
//! the identical column space; gam additionally applies a *mild* wiggliness
//! penalty (the smooths are low-rank, k = 4, so they are nearly saturated and
//! the penalty barely bites). The two fitted linear predictors `η = Xβ` must
//! therefore essentially coincide.
//!
//! We assert:
//!   1. relative-L2 of the fitted log-scale linear predictor `η` < 0.10
//!      (principled: same likelihood + same basis; the only difference is gam's
//!      light k=4 ridge, which moves η by only a few percent), and
//!   2. the `linear(offset)` term carries the offset signal — refitting WITHOUT
//!      the offset and differencing the two η vectors recovers a contribution
//!      that is strongly aligned (Pearson > 0.95) with the offset column,
//!      confirming the additive log-link offset adjustment.
//!
//! A genuine divergence (wrong variance power, wrong link, broken Tweedie
//! working-response) makes this test fail — which is the point.

use gam::matrix::LinearOperator;
use gam::smooth::build_term_collection_design;
use gam::test_support::reference::{Column, pearson, relative_l2, rmse, run_python};
use gam::{
    FitConfig, FitResult, encode_recordswith_inferred_schema, fit_from_formula, init_parallelism,
};
use ndarray::{Array1, Array2};
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand_distr::{Distribution, Gamma, Poisson, Uniform};

/// Draw one compound Poisson-Gamma (Tweedie) variate with mean `mu`, power
/// `p ∈ (1,2)`, dispersion `phi`. This is the exact exponential-dispersion
/// construction: `N ~ Poisson(λ)`, `y = Σ_{i=1}^N G_i`, `G_i ~ Gamma(α, θ)`,
/// with `λ = μ^{2-p}/(φ(2-p))`, `α = (2-p)/(p-1)`, `θ = φ(p-1)μ^{p-1}`.
/// `N = 0` yields the exact zero, giving the characteristic zero-inflation.
fn tweedie_sample(mu: f64, p: f64, phi: f64, rng: &mut StdRng) -> f64 {
    let lambda = mu.powf(2.0 - p) / (phi * (2.0 - p));
    let alpha = (2.0 - p) / (p - 1.0);
    let theta = phi * (p - 1.0) * mu.powf(p - 1.0);
    let n = Poisson::new(lambda).expect("poisson rate").sample(rng) as u64;
    if n == 0 {
        return 0.0;
    }
    let gamma = Gamma::new(alpha, theta).expect("gamma shape/scale");
    (0..n).map(|_| gamma.sample(rng)).sum()
}

#[test]
fn gam_tweedie_matches_statsmodels_power_variance() {
    init_parallelism();

    // ---- synthesize identical Tweedie data for both engines (seed=890) -----
    let n = 200usize;
    let p = 1.5_f64;
    let phi = 2.0_f64;
    let mut rng = StdRng::seed_from_u64(890);
    let ux = Uniform::new(0.0, 8.0).expect("uniform 0..8");
    let uo = Uniform::new(0.0, 1.0).expect("uniform offset");

    let mut x1 = Vec::with_capacity(n);
    let mut x2 = Vec::with_capacity(n);
    let mut offset = Vec::with_capacity(n);
    let mut y = Vec::with_capacity(n);
    for _ in 0..n {
        let a = ux.sample(&mut rng);
        let b = ux.sample(&mut rng);
        let o = uo.sample(&mut rng);
        // Truth on the log-mean scale, with a multiplicative offset:
        //   mu_eta = 1.5 + 0.4 sin(x1 π/6) + 0.3 cos(x2 π/5);  mu = exp(mu_eta)*o
        let mu_eta = 1.5
            + 0.4 * (a * std::f64::consts::PI / 6.0).sin()
            + 0.3 * (b * std::f64::consts::PI / 5.0).cos();
        let mu = mu_eta.exp() * o;
        let yi = tweedie_sample(mu, p, phi, &mut rng);
        x1.push(a);
        x2.push(b);
        offset.push(o);
        y.push(yi);
    }
    let zeros = y.iter().filter(|&&v| v == 0.0).count();
    assert!(
        zeros > 0,
        "Tweedie p=1.5 data should be zero-inflated; got {zeros} exact zeros"
    );

    // ---- fit gam: y ~ s(x1, k=4) + s(x2, k=4) + linear(offset), Tweedie ----
    let headers: Vec<String> = ["y", "x1", "x2", "offset"].into_iter().map(String::from).collect();
    let rows: Vec<csv::StringRecord> = (0..n)
        .map(|i| {
            csv::StringRecord::from(vec![
                y[i].to_string(),
                x1[i].to_string(),
                x2[i].to_string(),
                offset[i].to_string(),
            ])
        })
        .collect();
    let ds = encode_recordswith_inferred_schema(headers, rows).expect("encode tweedie dataset");
    let col = ds.column_map();
    let (x1_idx, x2_idx, off_idx) = (col["x1"], col["x2"], col["offset"]);
    let width = ds.headers.len();

    let cfg = FitConfig {
        family: Some("tweedie".to_string()),
        ..FitConfig::default()
    };
    let result = fit_from_formula(
        "y ~ s(x1, k=4) + s(x2, k=4) + linear(offset)",
        &ds,
        &cfg,
    )
    .expect("gam tweedie fit");
    let FitResult::Standard(fit) = result else {
        panic!("Tweedie(log) is a scalar GLM family => expected FitResult::Standard");
    };
    let gam_edf = fit.fit.edf_total().expect("gam reports total edf");

    // Rebuild gam's design at the training rows from the frozen spec. With a
    // log link, `η = X β` is the linear predictor on the log-mean scale.
    let mut grid = Array2::<f64>::zeros((n, width));
    for i in 0..n {
        grid[[i, x1_idx]] = x1[i];
        grid[[i, x2_idx]] = x2[i];
        grid[[i, off_idx]] = offset[i];
    }
    let design = build_term_collection_design(grid.view(), &fit.resolvedspec)
        .expect("rebuild gam design at training rows");
    let gam_eta: Vec<f64> = design.design.apply(&fit.fit.beta).to_vec();
    let ncols = design.design.ncols();
    assert_eq!(
        fit.fit.beta.len(),
        ncols,
        "beta length must match design columns"
    );

    // Materialize the dense design column-by-column (apply unit basis vectors)
    // so statsmodels fits the SAME column space gam penalized over.
    let mut dense = Array2::<f64>::zeros((n, ncols));
    for j in 0..ncols {
        let mut e = Array1::<f64>::zeros(ncols);
        e[j] = 1.0;
        let colj = design.design.apply(&e);
        for i in 0..n {
            dense[[i, j]] = colj[i];
        }
    }

    // ---- fit the SAME basis with statsmodels Tweedie(var_power=1.5, log) ---
    // Ship y plus every design column (D0..D{ncols-1}) as flat columns.
    let mut cols: Vec<Column<'_>> = Vec::with_capacity(ncols + 1);
    cols.push(Column::new("y", &y));
    let colnames: Vec<String> = (0..ncols).map(|j| format!("D{j}")).collect();
    let dense_cols: Vec<Vec<f64>> = (0..ncols).map(|j| dense.column(j).to_vec()).collect();
    for (name, data) in colnames.iter().zip(dense_cols.iter()) {
        cols.push(Column::new(name, data));
    }

    let body = format!(
        r#"
import numpy as np
import statsmodels.api as sm
ncols = {ncols}
X = np.column_stack([np.asarray(df["D%d" % j], dtype=float) for j in range(ncols)])
yv = np.asarray(df["y"], dtype=float)
fam = sm.families.Tweedie(var_power=1.5, link=sm.families.links.Log())
# gam already carries its own intercept/constant column in the basis, so do
# NOT add another constant here — that would make the column space differ.
m = sm.GLM(yv, X, family=fam).fit(maxiter=200)
eta = m.predict(X, linear=True)
emit("eta", np.asarray(eta, dtype=float))
"#
    );
    let r = run_python(&cols, &body);
    let sm_eta = r.vector("eta");
    assert_eq!(sm_eta.len(), n, "statsmodels eta length mismatch");

    // ---- compare linear predictors on the log scale -----------------------
    let rel = relative_l2(&gam_eta, sm_eta);
    let corr = pearson(&gam_eta, sm_eta);
    let mu_gam: Vec<f64> = gam_eta.iter().map(|e| e.exp()).collect();
    let mu_sm: Vec<f64> = sm_eta.iter().map(|e| e.exp()).collect();
    let resp_rmse = rmse(&mu_gam, &mu_sm);

    eprintln!(
        "[tweedie p=1.5] n={n} zeros={zeros} gam_edf={gam_edf:.3} \
         rel_l2(eta)={rel:.4} pearson(eta)={corr:.5} rmse(mu)={resp_rmse:.4}"
    );

    // ---- offset verification: differencing the with/without-offset fits ---
    // Refit WITHOUT the linear(offset) term; the difference in fitted η must be
    // the offset term's additive log-link contribution, hence aligned with the
    // offset column itself.
    let no_off = fit_from_formula("y ~ s(x1, k=4) + s(x2, k=4)", &ds, &cfg)
        .expect("gam tweedie fit without offset");
    let FitResult::Standard(fit_no) = no_off else {
        panic!("expected FitResult::Standard for no-offset Tweedie fit");
    };
    let design_no = build_term_collection_design(grid.view(), &fit_no.resolvedspec)
        .expect("rebuild no-offset gam design");
    let eta_no: Vec<f64> = design_no.design.apply(&fit_no.fit.beta).to_vec();
    let offset_contrib: Vec<f64> = gam_eta
        .iter()
        .zip(eta_no.iter())
        .map(|(a, b)| a - b)
        .collect();
    let off_align = pearson(&offset_contrib, &offset);
    eprintln!("[tweedie offset] pearson(eta_with - eta_without, offset) = {off_align:.4}");

    // (1) Same Tweedie likelihood + same basis: η must essentially coincide.
    // The only wedge is gam's light k=4 wiggliness penalty, worth a few percent
    // on η; 0.10 is a principled bound that still catches a wrong power/link.
    assert!(
        corr > 0.99,
        "fitted Tweedie η should track statsmodels closely: pearson={corr:.5}"
    );
    assert!(
        rel < 0.10,
        "gam Tweedie η diverges from statsmodels: rel_l2={rel:.4} (bound 0.10)"
    );
    // (2) The linear(offset) term must additively encode the offset on the log
    // scale: removing it shifts η by a contribution co-linear with `offset`.
    assert!(
        off_align > 0.95,
        "linear(offset) contribution not aligned with offset column: pearson={off_align:.4}"
    );
}
