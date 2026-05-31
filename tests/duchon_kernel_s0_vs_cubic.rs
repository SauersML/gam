//! Head-to-head: the two Duchon spectral powers, reachable through ONE gam
//! construction, on identical 2-D data.
//!
//! The non-periodic Euclidean Duchon kernel exponent is `2(p + s) − d`, where
//! `p` is the polynomial (null-space) order and `s` is the Duchon spectral
//! power. With an affine null space (`p = 2`) in `d = 2`:
//!   * `s = 0`        → exponent 2 → `r²·log r`  (the integer-order Duchon
//!                       kernel, equal to the classic thin-plate kernel)
//!   * `s = (d−1)/2`  → exponent 3 → `r³`        (the fractional cubic, gam's
//!                       magic default)
//!
//! These are NOT two different bases — they are the SAME Duchon construction at
//! two values of `s`. `s = ½` is gam's default `duchon(x, z, k)`; `s = 0` is the
//! explicit `duchon(x, z, k, power=0)`. This test exercises BOTH and asserts:
//!   (1) each recovers the known surface (objective truth recovery), and matches
//!       or beats mgcv's Duchon (`bs="ds"`) on truth-recovery accuracy;
//!   (2) `power=0` is genuinely REACHABLE — it produces a fit demonstrably
//!       different from the `r³` default (if `power=0` were silently swallowed
//!       back to the cubic default, the two gam fits would be identical).
//! It also prints both truth-recovery RMSEs so the kernels can be compared.

use gam::matrix::LinearOperator;
use gam::smooth::build_term_collection_design;
use gam::test_support::reference::{Column, relative_l2, rmse, run_r};
use gam::{
    FitConfig, FitResult, encode_recordswith_inferred_schema, fit_from_formula, init_parallelism,
};
use ndarray::Array2;
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand_distr::{Distribution, Normal, Uniform};

/// Known smooth 2-D surface: a sum of two Gaussian bumps over the unit square.
fn truth_surface(x: f64, z: f64) -> f64 {
    let bump = |cx: f64, cz: f64, s: f64, a: f64| {
        let d2 = (x - cx).powi(2) + (z - cz).powi(2);
        a * (-d2 / (2.0 * s * s)).exp()
    };
    bump(0.3, 0.3, 0.18, 1.0) + bump(0.7, 0.65, 0.22, 0.8)
}

/// Fit `formula` with gam (gaussian, REML default) and return fitted values on
/// the supplied test grid via the frozen spec (identity link ⇒ design·β = mean).
fn gam_fit_on_grid(
    formula: &str,
    ds: &gam::data::EncodedDataset,
    x_idx: usize,
    z_idx: usize,
    gx: &[f64],
    gz: &[f64],
) -> Vec<f64> {
    let cfg = FitConfig {
        family: Some("gaussian".to_string()),
        ..FitConfig::default()
    };
    let result =
        fit_from_formula(formula, ds, &cfg).unwrap_or_else(|e| panic!("gam fit '{formula}': {e}"));
    let FitResult::Standard(fit) = result else {
        panic!("expected a standard GAM fit for '{formula}'");
    };
    let m = gx.len();
    let mut grid = Array2::<f64>::zeros((m, ds.headers.len()));
    for i in 0..m {
        grid[[i, x_idx]] = gx[i];
        grid[[i, z_idx]] = gz[i];
    }
    let design = build_term_collection_design(grid.view(), &fit.resolvedspec)
        .unwrap_or_else(|e| panic!("rebuild design for '{formula}': {e}"));
    design.design.apply(&fit.fit.beta).to_vec()
}

#[test]
fn duchon_s0_thinplate_and_cubic_are_both_reachable_and_recover() {
    init_parallelism();

    // ---- synthetic data: (x, z) ~ U[0,1]^2, y = f(x,z) + N(0, 0.10), n=400 ----
    let n = 400usize;
    let mut rng = StdRng::seed_from_u64(20260531);
    let u = Uniform::new(0.0_f64, 1.0).expect("uniform [0,1]");
    let noise = Normal::new(0.0, 0.10).expect("normal");
    let (mut x, mut z, mut y) = (Vec::new(), Vec::new(), Vec::new());
    for _ in 0..n {
        let xi = u.sample(&mut rng);
        let zi = u.sample(&mut rng);
        x.push(xi);
        z.push(zi);
        y.push(truth_surface(xi, zi) + noise.sample(&mut rng));
    }

    let headers = ["x", "z", "y"].into_iter().map(String::from).collect();
    let rows = (0..n)
        .map(|i| {
            csv::StringRecord::from(vec![x[i].to_string(), z[i].to_string(), y[i].to_string()])
        })
        .collect();
    let ds = encode_recordswith_inferred_schema(headers, rows).expect("encode 2d dataset");
    let col = ds.column_map();
    let (x_idx, z_idx) = (col["x"], col["z"]);

    // ---- dense interior test grid on [0.05, 0.95]^2 ------------------------
    let g = 25usize;
    let coord = |i: usize| 0.05 + 0.90 * i as f64 / (g as f64 - 1.0);
    let (mut gx, mut gz, mut y_truth) = (Vec::new(), Vec::new(), Vec::new());
    for i in 0..g {
        for j in 0..g {
            let (xi, zi) = (coord(i), coord(j));
            gx.push(xi);
            gz.push(zi);
            y_truth.push(truth_surface(xi, zi));
        }
    }
    let m = gx.len();

    // ---- gam: BOTH Duchon spectral powers through one construction ---------
    // Default (no power) is the magic cubic s=(d-1)/2 → r³; explicit power=0 is
    // the integer-order Duchon kernel s=0 → r²·log r (the thin-plate kernel).
    let gam_cubic = gam_fit_on_grid("y ~ duchon(x, z, k=49)", &ds, x_idx, z_idx, &gx, &gz);
    let gam_thinplate = gam_fit_on_grid(
        "y ~ duchon(x, z, k=49, power=0)",
        &ds,
        x_idx,
        z_idx,
        &gx,
        &gz,
    );

    // ---- mgcv Duchon bs="ds", m=c(2,0): the r²·log r reference -------------
    let mut x_all = x.clone();
    x_all.extend_from_slice(&gx);
    let mut z_all = z.clone();
    z_all.extend_from_slice(&gz);
    let mut y_all = y.clone();
    y_all.extend(std::iter::repeat_n(0.0, m));
    let mut is_train = vec![1.0; n];
    is_train.extend(std::iter::repeat_n(0.0, m));
    let r = run_r(
        &[
            Column::new("x", &x_all),
            Column::new("z", &z_all),
            Column::new("y", &y_all),
            Column::new("is_train", &is_train),
        ],
        r#"
        suppressPackageStartupMessages(library(mgcv))
        train <- df[df$is_train > 0.5, ]
        grid  <- df[df$is_train < 0.5, ]
        m <- gam(y ~ s(x, z, bs = "ds", k = 49, m = c(2, 0)), data = train, method = "REML")
        emit("fitted", as.numeric(predict(m, newdata = grid)))
        "#,
    );
    let mgcv_fitted = r.vector("fitted");
    assert_eq!(mgcv_fitted.len(), m, "mgcv prediction length mismatch");

    // ---- objective truth recovery -----------------------------------------
    let rmse_cubic = rmse(&gam_cubic, &y_truth);
    let rmse_thinplate = rmse(&gam_thinplate, &y_truth);
    let rmse_mgcv = rmse(mgcv_fitted, &y_truth);
    // How different are the two gam fits from each other (reachability proof)?
    let rel_cubic_vs_thinplate = relative_l2(&gam_cubic, &gam_thinplate);
    let rms_truth = (y_truth.iter().map(|t| t * t).sum::<f64>() / y_truth.len() as f64).sqrt();

    eprintln!(
        "duchon-kernel-compare-2d: n={n} grid={m} sigma=0.10 rms_truth={rms_truth:.4}\n  \
         gam r³ (s=½, default)         truth_rmse={rmse_cubic:.4}\n  \
         gam r²·log r (s=0, power=0)   truth_rmse={rmse_thinplate:.4}\n  \
         mgcv bs=ds m=c(2,0) (r²·log r) truth_rmse={rmse_mgcv:.4}\n  \
         rel_l2(gam r³, gam r²·log r)  = {rel_cubic_vs_thinplate:.4}"
    );

    // (1) Both Duchon spectral powers recover the surface (non-degeneracy +
    //     denoising; a constant predictor scores ≈ rms_truth ≈ 0.5).
    assert!(
        rmse_cubic < 0.15,
        "gam cubic (r³) failed to recover the surface: rmse={rmse_cubic:.4}"
    );
    assert!(
        rmse_thinplate < 0.15,
        "gam thin-plate (r²·log r, power=0) failed to recover the surface: rmse={rmse_thinplate:.4}"
    );

    // (2) power=0 is genuinely REACHABLE: the s=0 fit is demonstrably different
    //     from the r³ default. If power=0 were swallowed back to the cubic
    //     default (the bug this guards), the two fits would be identical and
    //     this relative difference would be ~0.
    assert!(
        rel_cubic_vs_thinplate > 1e-3,
        "power=0 produced the SAME fit as the r³ default (rel_l2={rel_cubic_vs_thinplate:.2e}): \
         the s=0 thin-plate Duchon kernel is not reachable — power=0 is being swallowed."
    );

    // (3) Match-or-beat mgcv on truth recovery, for BOTH spectral powers (mgcv
    //     uses the same r²·log r kernel as the s=0 variant; a 10% accuracy margin
    //     absorbs differing low-rank constructions and REML λ selection).
    assert!(
        rmse_cubic <= rmse_mgcv * 1.10,
        "gam cubic recovers worse than mgcv: {rmse_cubic:.4} > 1.10*{rmse_mgcv:.4}"
    );
    assert!(
        rmse_thinplate <= rmse_mgcv * 1.10,
        "gam thin-plate recovers worse than mgcv: {rmse_thinplate:.4} > 1.10*{rmse_mgcv:.4}"
    );
}
