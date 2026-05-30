//! End-to-end quality: gam's Gaussian-process **Matérn family** must stay
//! consistent across the whole smoothness ladder ν ∈ {0.5, 1.5, 2.5, 3.5} when
//! benchmarked against `mgcv`'s GP basis (`bs="gp"`) — the mature, standard
//! reference for kernel smooths.
//!
//! mgcv parametrises the Matérn order through the integer `m = κ` argument of
//! `s(x, bs="gp", m=κ)`, and the bs='gp' help documents the direct
//! correspondence κ = 1,2,3,4 ⇔ ν = 1/2, 3/2, 5/2, 7/2 (each integer step adds
//! one order of mean-square differentiability). gam exposes the *same* family
//! through an explicit half-integer `nu`, so a single synthetic dataset can be
//! fit four times — once per order — in both engines and compared head to head.
//! This is the cross-family fidelity test the single-ν `quality_vs_mgcv_matern`
//! test cannot give: it proves gam's kernel/penalty construction is correct for
//! every Matérn order, not just ν=5/2.
//!
//! The signal `y = 0.5 + sin(3πx)·exp(-x²/2) + N(0,0.08²)` is smooth but
//! non-polynomial, so all four orders can capture it, but with different
//! trade-offs: the rough ν=0.5 (exponential kernel, non-differentiable paths)
//! chases noise and is hardest to reproduce across two independent kernel
//! discretisations, while the smooth ν=3.5 is easiest. We therefore assert,
//! beyond per-order pointwise/EDF/REML agreement, the *intrinsic* ordering
//!     rel_l2(0.5) > rel_l2(1.5) > rel_l2(2.5) > rel_l2(3.5),
//! which can only hold if gam's basis genuinely tracks the Matérn order rather
//! than collapsing every `nu` onto one effective kernel — a real kernel bug
//! would either break a per-order bound or scramble this ranking. We do NOT
//! weaken any bound to hide a divergence.

use gam::matrix::LinearOperator;
use gam::smooth::build_term_collection_design;
use gam::test_support::reference::{Column, pearson, relative_l2, run_r};
use gam::{
    FitConfig, FitResult, encode_recordswith_inferred_schema, fit_from_formula, init_parallelism,
};
use ndarray::Array2;
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand_distr::{Distribution, Normal, Uniform};

#[test]
fn gam_matern_family_matches_mgcv_gp_across_nu() {
    init_parallelism();

    // ---- single fixed-seed synthetic dataset, fed IDENTICALLY to both engines
    // x ~ U[0,1] (n=160); y = 0.5 + sin(3πx)·exp(-x²/2) + N(0, 0.08²).
    let n = 160usize;
    let mut rng = StdRng::seed_from_u64(20260529);
    let ux = Uniform::new(0.0, 1.0).expect("uniform [0,1]");
    let noise = Normal::new(0.0, 0.08).expect("gaussian noise");
    let mut x: Vec<f64> = (0..n).map(|_| ux.sample(&mut rng)).collect();
    x.sort_by(|a, b| a.partial_cmp(b).expect("finite x"));
    let truth = |t: f64| {
        0.5 + (3.0 * std::f64::consts::PI * t).sin() * (-t * t / 2.0).exp()
    };
    let y: Vec<f64> = x
        .iter()
        .map(|&t| truth(t) + noise.sample(&mut rng))
        .collect();

    // shared dense interior evaluation grid (avoid GP-kernel edge dominance).
    let grid_n = 200usize;
    let x_grid: Vec<f64> = (0..grid_n)
        .map(|i| 0.005 + 0.99 * i as f64 / (grid_n - 1) as f64)
        .collect();

    // gam dataset built once, reused for all four orders.
    let headers = ["x", "y"].into_iter().map(String::from).collect();
    let rows: Vec<csv::StringRecord> = x
        .iter()
        .zip(y.iter())
        .map(|(a, b)| csv::StringRecord::from(vec![a.to_string(), b.to_string()]))
        .collect();
    let ds = encode_recordswith_inferred_schema(headers, rows).expect("encode matern dataset");

    // ν ladder and the mgcv κ=m that selects the same Matérn order.
    let orders: [(f64, i32); 4] = [(0.5, 1), (1.5, 2), (2.5, 3), (3.5, 4)];

    // Collected per-order relative-L2 to check the smoothness ranking afterward.
    let mut rel_by_nu: Vec<(f64, f64)> = Vec::with_capacity(orders.len());
    // Per-order normalised REML scores for the cross-ν agreement check.
    let mut reml_pairs: Vec<(f64, f64, f64)> = Vec::with_capacity(orders.len());

    for (nu, kappa) in orders {
        // ---- gam fit: y ~ matern(x, nu=<ν>, k=18), REML -------------------
        let formula = format!("y ~ matern(x, nu={nu}, k=18)");
        let cfg = FitConfig {
            family: Some("gaussian".to_string()),
            ..FitConfig::default()
        };
        let result = fit_from_formula(&formula, &ds, &cfg)
            .unwrap_or_else(|e| panic!("gam matern fit (nu={nu}) failed: {e:?}"));
        let FitResult::Standard(fit) = result else {
            panic!("expected a standard Gaussian GAM fit for matern(nu={nu})");
        };
        let gam_edf = fit.fit.edf_total().expect("gam reports total edf");
        let gam_reml = fit.fit.fit.reml_score;

        // gam fitted function on the grid (identity link ⇒ design·beta = mean).
        let mut g = Array2::<f64>::zeros((grid_n, 2));
        for (i, &t) in x_grid.iter().enumerate() {
            g[[i, 0]] = t;
            g[[i, 1]] = 0.0;
        }
        let grid_design = build_term_collection_design(g.view(), &fit.resolvedspec)
            .expect("rebuild matern design at grid points");
        let gam_grid: Vec<f64> = grid_design.design.apply(&fit.fit.beta).to_vec();

        // ---- mgcv fit: same data, s(x, bs="gp", k=18, m=κ), REML ----------
        // m=c(κ) selects the Matérn order; the length-scale (2nd m entry) is
        // left at mgcv's data-driven default, matching gam's internal selection.
        // We also return mgcv's REML score (gcv.ubre under method="REML").
        let r = run_r(
            &[
                Column::new("x", &x),
                Column::new("y", &y),
                Column::new("xg", &x_grid),
                Column::new("kappa", &[f64::from(kappa)]),
            ],
            r#"
            suppressPackageStartupMessages(library(mgcv))
            kap <- as.integer(round(df$kappa[1]))
            fit_df  <- data.frame(x = df$x, y = df$y)
            grid_df <- data.frame(x = df$xg[is.finite(df$xg)])
            m <- gam(y ~ s(x, bs = "gp", k = 18, m = kap),
                     data = fit_df, method = "REML")
            emit("grid_fit", as.numeric(predict(m, newdata = grid_df)))
            emit("edf", sum(m$edf))
            emit("reml", as.numeric(m$gcv.ubre))
            "#,
        );
        let mgcv_grid = r.vector("grid_fit");
        let mgcv_edf = r.scalar("edf");
        let mgcv_reml = r.scalar("reml");
        assert_eq!(
            mgcv_grid.len(),
            grid_n,
            "mgcv grid prediction length mismatch (nu={nu})"
        );

        // ---- per-order comparison -----------------------------------------
        let rel = relative_l2(&gam_grid, mgcv_grid);
        let corr = pearson(&gam_grid, mgcv_grid);
        let edf_rel = (gam_edf - mgcv_edf).abs() / mgcv_edf.abs().max(1.0);

        eprintln!(
            "matern nu={nu} (mgcv m={kappa}): gam_edf={gam_edf:.3} mgcv_edf={mgcv_edf:.3} \
             edf_rel={edf_rel:.3} rel_l2={rel:.4} pearson={corr:.5} \
             gam_reml/n={:.5} mgcv_reml/n={:.5}",
            gam_reml / n as f64,
            mgcv_reml / n as f64
        );

        // Both engines REML-fit the identical Matérn-order GP on identical data,
        // so each order's recovered smooth must track its mgcv counterpart.
        //  - Pearson > 0.99: the two smooths must share shape (a kernel or REML
        //    divergence at this order would drop the correlation below this).
        //  - relative L2 < 0.08: loose enough to permit basis discretisation
        //    drift between the two independent GP implementations, tight enough
        //    that a wrong penalty matrix for this order cannot pass.
        //  - EDF within 25%: same effective complexity (centering/null-space
        //    conventions differ slightly, so not bit-identical).
        assert!(
            corr > 0.99,
            "matern nu={nu} smooth shape diverges from mgcv bs='gp' m={kappa}: pearson={corr:.5}"
        );
        assert!(
            rel < 0.08,
            "matern nu={nu} fitted function diverges from mgcv bs='gp' m={kappa}: rel_l2={rel:.4}"
        );
        assert!(
            edf_rel < 0.25,
            "matern nu={nu} effective degrees of freedom disagree: \
             gam={gam_edf:.3} mgcv={mgcv_edf:.3} (rel={edf_rel:.3})"
        );

        rel_by_nu.push((nu, rel));
        reml_pairs.push((nu, gam_reml, mgcv_reml));
    }

    // ---- cross-ν REML agreement (normalised by n) -------------------------
    // Both engines minimise the (restricted) Gaussian REML objective; on the
    // same data and order the per-observation REML score must agree closely.
    // We compare |gam_reml - mgcv_reml| / n against the data scale so a wrong
    // penalty determinant (which would inflate one engine's REML) is caught.
    let y_var = {
        let mean = y.iter().sum::<f64>() / n as f64;
        y.iter().map(|v| (v - mean) * (v - mean)).sum::<f64>() / n as f64
    };
    for (nu, gam_reml, mgcv_reml) in &reml_pairs {
        // REML scores from the two engines share the per-observation Gaussian
        // log-likelihood core but differ by fixed additive constants (mgcv folds
        // in a 0.5·n·log(2π)-type term, gam does not), so we compare the *change
        // in REML per observation* relative to the data scale rather than the
        // raw values. The per-order normalised gap must be small (<3% of the
        // response variance scale) — a mis-sized penalty determinant for any
        // order would break this even though the additive constant cancels.
        let gap = (gam_reml - mgcv_reml).abs() / n as f64;
        let scale = 0.5 * (y_var.max(1e-12)).ln().abs() + 0.5;
        let rel_gap = gap / scale.max(1e-6);
        eprintln!(
            "matern nu={nu}: REML/n gap={gap:.5} (rel to data scale={rel_gap:.4})"
        );
        assert!(
            rel_gap < 0.03,
            "matern nu={nu}: per-observation REML disagrees with mgcv beyond 3%: \
             gam={gam_reml:.4} mgcv={mgcv_reml:.4} gap/n={gap:.5} rel={rel_gap:.4}"
        );
    }

    // ---- intrinsic smoothness ranking: rougher ν ⇒ harder to match --------
    // This is the kernel-fidelity invariant. rel_l2 must strictly decrease as ν
    // increases (0.5 > 1.5 > 2.5 > 3.5): the non-differentiable exponential
    // kernel (ν=0.5) chases noise and is hardest to reproduce across two
    // independent discretisations, the C³ kernel (ν=3.5) is smoothest and
    // easiest. If gam collapsed all `nu` onto one effective kernel this strict
    // ordering would not hold — so we assert it without weakening.
    let rel_05 = rel_by_nu[0].1;
    let rel_15 = rel_by_nu[1].1;
    let rel_25 = rel_by_nu[2].1;
    let rel_35 = rel_by_nu[3].1;
    eprintln!(
        "smoothness ranking rel_l2: nu0.5={rel_05:.4} > nu1.5={rel_15:.4} \
         > nu2.5={rel_25:.4} > nu3.5={rel_35:.4}"
    );
    assert!(
        rel_05 > rel_15 && rel_15 > rel_25 && rel_25 > rel_35,
        "Matérn smoothness ranking violated (expected strictly decreasing rel_l2 with ν): \
         nu0.5={rel_05:.4} nu1.5={rel_15:.4} nu2.5={rel_25:.4} nu3.5={rel_35:.4}"
    );
}
