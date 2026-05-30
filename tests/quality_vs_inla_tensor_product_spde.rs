//! End-to-end quality: gam's REML tensor-product 2-D smooth must agree with
//! **R-INLA's SPDE/Matérn latent-Gaussian field** — the de-facto standard for
//! scalable approximate-Bayesian spatial inference — on the *same* spatial data.
//!
//! Reference tool: `R-INLA` (the INLA package) with `fmesher`, fitting the
//! spatial surface as a continuously-indexed Gaussian random field via the
//! Stochastic-PDE representation `inla.spde2.matern(mesh)` on a Delaunay mesh
//! over the (Longitude, Latitude) domain. INLA integrates out the latent field
//! and hyperparameters with its nested-Laplace scheme and returns a fully
//! marginalized posterior for the fitted linear predictor (mean + SD).
//!
//! gam represents the *same* 2-D spatial field with a classical tensor-product
//! smooth `te(lon, lat)` — the row-wise Kronecker product of two 1-D marginal
//! bases with a per-margin penalty — and selects the smoothing parameters by
//! REML (Laplace marginal likelihood). The two engines therefore parameterize a
//! 2-D spatial field in fundamentally different ways: INLA via a stochastic-PDE
//! Matérn covariance solved on a triangulation, gam via a Cartesian
//! tensor-product penalized basis. This test checks that despite that
//! difference, gam's Laplace+REML fit lands on essentially the same fitted
//! surface and uncertainty as INLA's marginalized SPDE posterior. A divergence
//! beyond the bounds below signals either a bug in gam's 2-D tensor smooth or a
//! genuine, documentable difference in how the two engines regularize spatial
//! fields.
//!
//! NOTE on the gam formula. The SPEC names `te(lon, lat, bs=c('tp','tp'))`
//! (per-margin thin-plate marginals, an mgcv idiom). gam's tensor-product
//! constructor `te(...)` builds B-spline marginal bases and does not expose
//! per-margin thin-plate selection, so we fit gam's native expression of the
//! same capability — a Cartesian tensor-product 2-D smooth `te(lon, lat)`,
//! REML-selected. This is exactly the "classical Cartesian product" the SPEC's
//! rationale contrasts against INLA's SPDE basis; the B-spline-vs-thin-plate
//! marginal choice changes the within-span parameterization, not the fitted
//! spatial field that both engines target.
//!
//! Data: n=500 samples from the HGDP+1kG panel — response = PC1, covariates =
//! (Longitude, Latitude). The *identical* 500 rows are handed to both engines.
//! PC1 of a worldwide genotype panel is a strong, smooth geographic gradient, so
//! both a Matérn field and a tensor-product spline should comfortably capture it
//! and their fitted surfaces should track closely.

use gam::matrix::LinearOperator;
use gam::smooth::build_term_collection_design;
use gam::test_support::reference::{Column, pearson, relative_l2, rmse, run_r};
use gam::{
    FitConfig, FitResult, encode_recordswith_inferred_schema, fit_from_formula, init_parallelism,
};
use csv::StringRecord;
use ndarray::{Array1, Array2};
use std::path::Path;

const HGDP_TSV: &str =
    concat!(env!("CARGO_MANIFEST_DIR"), "/bench/datasets/hgdp_1kg_pc_data.tsv");

/// Number of samples handed identically to gam and INLA.
const N: usize = 500;

#[test]
fn gam_tensor_product_matches_inla_spde_on_hgdp_pc1() {
    init_parallelism();

    // ---- load the first N rows of (Longitude, Latitude, PC1) from the TSV ---
    // We parse the tab-separated panel directly so the *exact same* 500 rows are
    // encoded for gam and emitted to INLA. No RNG, no subsampling jitter: row i
    // is row i in both engines.
    let mut rdr = csv::ReaderBuilder::new()
        .delimiter(b'\t')
        .has_headers(true)
        .from_path(Path::new(HGDP_TSV))
        .expect("open hgdp_1kg_pc_data.tsv");
    let headers = rdr.headers().expect("tsv header row").clone();
    let pc1_col = headers
        .iter()
        .position(|h| h == "PC1")
        .expect("PC1 column present");
    let lat_col = headers
        .iter()
        .position(|h| h == "Latitude")
        .expect("Latitude column present");
    let lon_col = headers
        .iter()
        .position(|h| h == "Longitude")
        .expect("Longitude column present");

    let mut lon: Vec<f64> = Vec::with_capacity(N);
    let mut lat: Vec<f64> = Vec::with_capacity(N);
    let mut pc1: Vec<f64> = Vec::with_capacity(N);
    for rec in rdr.records() {
        if lon.len() == N {
            break;
        }
        let rec = rec.expect("read tsv record");
        let lo: f64 = rec[lon_col].parse().expect("parse Longitude");
        let la: f64 = rec[lat_col].parse().expect("parse Latitude");
        let y: f64 = rec[pc1_col].parse().expect("parse PC1");
        if lo.is_finite() && la.is_finite() && y.is_finite() {
            lon.push(lo);
            lat.push(la);
            pc1.push(y);
        }
    }
    assert_eq!(
        lon.len(),
        N,
        "expected {N} finite (lon,lat,PC1) rows from the HGDP panel, got {}",
        lon.len()
    );

    // ---- fit with gam: PC1 ~ te(lon, lat), Gaussian, REML ------------------
    let hdrs: Vec<String> = vec!["lon".into(), "lat".into(), "pc1".into()];
    let rows: Vec<StringRecord> = (0..N)
        .map(|r| {
            StringRecord::from(vec![
                format!("{:.17e}", lon[r]),
                format!("{:.17e}", lat[r]),
                format!("{:.17e}", pc1[r]),
            ])
        })
        .collect();
    let ds = encode_recordswith_inferred_schema(hdrs, rows).expect("encode hgdp subset");
    let col = ds.column_map();
    let lon_idx = col["lon"];
    let lat_idx = col["lat"];

    let cfg = FitConfig {
        family: Some("gaussian".to_string()),
        ..FitConfig::default()
    };
    let result = fit_from_formula("pc1 ~ te(lon, lat)", &ds, &cfg).expect("gam te(lon,lat) fit");
    let FitResult::Standard(fit) = result else {
        panic!("expected a standard GAM fit for a Gaussian 2-D tensor-product smooth");
    };
    let gam_edf = fit.fit.edf_total().expect("gam reports total edf");

    // gam fitted surface at the observed sites: rebuild the design from the
    // frozen spec at the observed (lon, lat) (identity link => design*beta = mean).
    let mut grid = Array2::<f64>::zeros((N, ds.headers.len()));
    for r in 0..N {
        grid[[r, lon_idx]] = lon[r];
        grid[[r, lat_idx]] = lat[r];
    }
    let design = build_term_collection_design(grid.view(), &fit.resolvedspec)
        .expect("rebuild te design at observed sites");
    let gam_fitted: Vec<f64> = design.design.apply(&fit.fit.beta).to_vec();

    // gam posterior SD of the fitted linear predictor: s_i = sqrt(x_iᵀ Vp x_i),
    // where Vp is the smoothing-parameter-corrected Bayesian covariance (the
    // analogue of mgcv's `predict(se.fit=TRUE)` and the right counterpart to
    // INLA's marginal posterior SD of the linear predictor). We materialize the
    // design's columns by applying the operator to canonical basis vectors —
    // exact, and cheap for a 2-D tensor basis — then form the row quadratic form.
    let p = fit.fit.beta.len();
    let vp = fit
        .fit
        .beta_covariance_vp()
        .or_else(|| fit.fit.beta_covariance_vb())
        .expect("gam reports a posterior coefficient covariance (Vp or Vb)");
    assert_eq!(vp.nrows(), p, "Vp dimension matches coefficient count");
    let mut xmat = Array2::<f64>::zeros((N, p));
    for j in 0..p {
        let mut ej = Array1::<f64>::zeros(p);
        ej[j] = 1.0;
        let colj = design.design.apply(&ej); // length N: column j of the design
        for i in 0..N {
            xmat[[i, j]] = colj[i];
        }
    }
    let gam_sd: Vec<f64> = (0..N)
        .map(|i| {
            let xi = xmat.row(i);
            let vx = vp.dot(&xi); // Vp x_i
            xi.dot(&vx).max(0.0).sqrt() // sqrt(x_iᵀ Vp x_i)
        })
        .collect();

    // ---- fit the SAME 500 rows with R-INLA's SPDE/Matérn field -------------
    // Build a Delaunay mesh over the (lon,lat) sites (fmesher), an SPDE Matérn
    // model on it, project sites onto mesh nodes with the A matrix, and fit a
    // Gaussian observation model through inla.stack. We emit the marginalized
    // fitted linear-predictor mean and SD per site, and INLA's effective number
    // of parameters (neffp) as the EDF analogue.
    let r = run_r(
        &[
            Column::new("lon", &lon),
            Column::new("lat", &lat),
            Column::new("pc1", &pc1),
        ],
        r#"
        suppressPackageStartupMessages(library(INLA))
        loc <- cbind(df$lon, df$lat)
        n <- nrow(loc)
        # Delaunay mesh over the observed sites (SPDE domain triangulation).
        rng <- apply(loc, 2, function(z) diff(range(z)))
        ms <- max(rng)
        mesh <- inla.mesh.2d(loc = loc,
                             max.edge = c(ms / 12, ms / 3),
                             cutoff   = ms / 50,
                             offset   = c(ms / 10, ms / 3))
        # SPDE Matern model (alpha=2 => nu=1 in 2D), default PC-style priors.
        spde <- inla.spde2.matern(mesh = mesh, alpha = 2)
        s.index <- inla.spde.make.index(name = "spatial", n.spde = spde$n.spde)
        A <- inla.spde.make.A(mesh = mesh, loc = loc)
        stk <- inla.stack(
          data    = list(y = df$pc1),
          A       = list(A, 1),
          effects = list(c(s.index, list(Intercept = 1)),
                         list()),
          tag     = "est")
        form <- y ~ -1 + Intercept + f(spatial, model = spde)
        m <- inla(form,
                  data = inla.stack.data(stk),
                  family = "gaussian",
                  control.predictor = list(A = inla.stack.A(stk), compute = TRUE),
                  control.compute = list(dic = TRUE, config = TRUE))
        idx <- inla.stack.index(stk, tag = "est")$data
        fitted_mean <- m$summary.fitted.values$mean[idx]
        fitted_sd   <- m$summary.fitted.values$sd[idx]
        emit("fitted", as.numeric(fitted_mean))
        emit("sd", as.numeric(fitted_sd))
        # Effective number of parameters: INLA's neffp (mean) is the natural
        # EDF analogue of the latent-Gaussian field's complexity.
        emit("edf", as.numeric(m$neffp["mean of the neffp", 1]))
        "#,
    );
    let inla_fitted = r.vector("fitted");
    let inla_sd = r.vector("sd");
    let inla_edf = r.scalar("edf");
    assert_eq!(inla_fitted.len(), N, "INLA fitted length mismatch");
    assert_eq!(inla_sd.len(), N, "INLA sd length mismatch");

    // ---- compare the quantities that matter --------------------------------
    let rel = relative_l2(&gam_fitted, inla_fitted);
    let corr = pearson(&gam_fitted, inla_fitted);
    let sd_rmse = rmse(&gam_sd, inla_sd);
    let inla_sd_mean = inla_sd.iter().sum::<f64>() / N as f64;
    let sd_tol = 0.15 * inla_sd_mean;
    let edf_rel = (gam_edf - inla_edf).abs() / inla_edf.abs().max(1.0);

    eprintln!(
        "te(lon,lat) vs INLA-SPDE: n={N} gam_edf={gam_edf:.3} inla_edf={inla_edf:.3} \
         edf_rel={edf_rel:.4} rel_l2={rel:.4} pearson={corr:.5} \
         sd_rmse={sd_rmse:.5} 0.15*mean(inla_sd)={sd_tol:.5} (inla_sd_mean={inla_sd_mean:.5})"
    );

    // 1. Fitted surface. Both engines represent the same smooth geographic PC1
    //    field, but via different basis families (Cartesian tensor-product vs
    //    SPDE Matérn), so 2-D fits admit larger discrepancy than 1-D: a 10%
    //    relative-L2 ceiling still pins the two surfaces to the same field while
    //    leaving room for the genuine basis difference. A divergence past 10%
    //    means gam's 2-D smooth fits a materially different surface than the
    //    mature SPDE reference — a real finding.
    assert!(
        rel < 0.10,
        "gam tensor-product surface diverges from INLA SPDE: rel_l2={rel:.4} (pearson={corr:.5})"
    );

    // 2. Posterior SD agreement. The two marginal-uncertainty fields must agree
    //    in absolute terms relative to INLA's own SD scale: RMSE below 15% of
    //    the mean INLA posterior SD. This catches a mis-scaled or structurally
    //    wrong uncertainty surface from gam's Laplace covariance.
    assert!(
        sd_rmse < sd_tol,
        "gam posterior SD disagrees with INLA: rmse={sd_rmse:.5} >= 0.15*mean(inla_sd)={sd_tol:.5}"
    );

    // 3. Effective degrees of freedom. Both engines should select comparable
    //    spatial-field complexity for the same data; within 25% relative is a
    //    principled "same-ballpark complexity" bound given the basis/null-space
    //    and prior conventions differ between a tensor-product penalty and an
    //    SPDE hyperprior.
    assert!(
        edf_rel < 0.25,
        "effective degrees of freedom disagree: gam={gam_edf:.3} inla={inla_edf:.3} (rel={edf_rel:.4})"
    );
}
