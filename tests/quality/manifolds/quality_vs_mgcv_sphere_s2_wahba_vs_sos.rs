//! End-to-end quality: gam's intrinsic S² (sphere) smooth must RECOVER a known
//! smooth spherical signal from noisy geographic data — and do it at least as
//! accurately as mgcv's spline-on-sphere `bs="sos"`, the mature reference.
//!
//! OBJECTIVE METRIC (primary claim): the data is generated from a *known* truth
//! f(lat,lon)=a·sin(lat)·cos(lon) (a genuine low-frequency field on S²) plus
//! fixed-seed Gaussian noise. The test asserts that gam's fitted surface recovers
//! that truth where the data constrain it: RMSE(gam fit, truth) at the fit sites
//! is below the noise SD. The fit sites are a small clustered patch of S², so the
//! 20x15 global lat/lon grid is mostly extrapolation across regions no data
//! constrain, where the surface is dominated by each smoother's prior rather than
//! recoverable from this design. The global-grid truth RMSE is therefore printed
//! and emitted as a second quality pair (`grid_rmse`), but not asserted. Measured
//! mechanism: gam's Wahba sphere basis carries the degree-1 spherical harmonics as
//! a block the roughness penalty leaves free and only the double-penalty ridge
//! shrinks (a712a8577, #1246), so gam continues a fitted linear gradient across
//! the globe, while mgcv `bs="sos"` leaves only the constant unpenalized and
//! reverts toward the mean far from the data. This is an absolute accuracy claim
//! about gam against ground truth — NOT a claim that gam reproduces another tool's
//! fit.
//!
//! BASELINE TO MATCH-OR-BEAT: `mgcv::gam(y ~ s(lat, lon, bs="sos", k=30),
//! method="REML")` (Wahba's spline-on-sphere, the de-facto integrated penalized
//! smoother on S²) is fit on the IDENTICAL data and predicted on the IDENTICAL
//! grid. We additionally assert gam's truth-recovery error does not exceed mgcv's
//! by more than 10%, so gam is at least as accurate as the mature reference on
//! the quantity that actually matters (recovering the real field). mgcv's surface
//! is no longer the pass criterion — we print rel_l2 / pearson for context only.
//!
//! INTRINSIC CORRECTNESS PROPERTY: any honest S² smoother must produce a surface
//! that is continuous across the ±180° longitude seam (no antimeridian
//! discontinuity), because longitude is an angular chart of the sphere and the
//! seam is not a real boundary. The grid deliberately spans both hemispheres and
//! straddles the seam to expose seam/pole artifacts; we assert gam's own surface
//! is seam-continuous.
//!
//! DISTINCTIVE-AXIS NOTE: sphere smoothing is a fragmented corner of the
//! ecosystem (spheresmooth / Directional / rcosmo are single-purpose and do not
//! offer an integrated penalized-REML GAM). mgcv `bs="sos"` is the only mature,
//! widely-trusted integrated comparator, hence its role as the accuracy baseline.
//!
//! Data: real geographic coordinates (lat/long) from the committed `quakes`
//! panel (`bench/datasets/quakes.csv`, 1000 Fiji-region earthquakes), with the
//! known smooth spherical signal plus fixed-seed Gaussian noise at a controlled
//! SNR. The identical (lat, lon, y) triples are handed to both engines. Only the
//! coordinates are taken from the data; the response is the known truth field, so
//! any real lat/lon panel of distinct, well-spread sites serves the recovery
//! claim — the quakes coordinates span ~28° of latitude and cross the ±180°
//! seam (long up to 188°), exercising the seam-continuity property.

use gam::matrix::LinearOperator;
use gam::smooth::{build_term_collection_design, freeze_term_collection_from_design};
use gam::test_support::reference::{Column, QualityPair, pearson, relative_l2, rmse, run_r};
use gam::{FitConfig, FitResult, fit_from_formula, init_parallelism, load_csvwith_inferred_schema};
use ndarray::Array2;
use std::io::Write as _;
use std::path::Path;

const QUAKES_CSV: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/bench/datasets/quakes.csv");

/// Known smooth signal on the sphere, evaluated from degree-valued lat/lon.
/// f = a·sin(lat)·cos(lon) is a real, infinitely-smooth, low-frequency field on
/// S² (it is proportional to a degree-1 spherical harmonic combination), so a
/// correct intrinsic smoother must recover it and a wrong kernel cannot fake it.
fn signal(lat_deg: f64, lon_deg: f64) -> f64 {
    let a = 2.5;
    a * lat_deg.to_radians().sin() * lon_deg.to_radians().cos()
}

#[test]
fn gam_sphere_matches_mgcv_sos_on_geographic_surface() {
    init_parallelism();

    // ---- read real lat/lon from the committed quakes panel (comma-separated) -
    let raw = std::fs::read_to_string(Path::new(QUAKES_CSV)).expect("read quakes csv");
    let mut lines = raw.lines();
    let header = lines.next().expect("csv header");
    let cols: Vec<&str> = header.split(',').collect();
    let lat_col = cols.iter().position(|c| *c == "lat").expect("lat column");
    let lon_col = cols.iter().position(|c| *c == "long").expect("long column");

    let mut lat_all: Vec<f64> = Vec::new();
    let mut lon_all: Vec<f64> = Vec::new();
    for line in lines {
        if line.trim().is_empty() {
            continue;
        }
        let fields: Vec<&str> = line.split(',').collect();
        let (Ok(la), Ok(lo)) = (
            fields[lat_col].trim().parse::<f64>(),
            fields[lon_col].trim().parse::<f64>(),
        ) else {
            continue;
        };
        if la.is_finite() && lo.is_finite() {
            lat_all.push(la);
            lon_all.push(lo);
        }
    }
    assert!(
        lat_all.len() >= 150,
        "need >=150 geographic rows, got {}",
        lat_all.len()
    );

    // ---- deterministic subsample to keep both REML fits fast ---------------
    // Coincident epicentres are common in the quakes panel; deduplicate to
    // distinct sites (a sphere smooth over coincident points is degenerate) and
    // keep a fixed-stride subsample so the design stays well-spread over S² and
    // both REML fits stay fast. Stride is chosen to land near ~250 distinct
    // sites regardless of how many the dedup yields.
    let mut seen: std::collections::BTreeSet<(i64, i64)> = std::collections::BTreeSet::new();
    let mut lat_dedup: Vec<f64> = Vec::new();
    let mut lon_dedup: Vec<f64> = Vec::new();
    for (la, lo) in lat_all.iter().zip(&lon_all) {
        let key = ((la * 100.0).round() as i64, (lo * 100.0).round() as i64);
        if seen.insert(key) {
            lat_dedup.push(*la);
            lon_dedup.push(*lo);
        }
    }
    let stride = (lat_dedup.len() / 250).max(1);
    let lat: Vec<f64> = lat_dedup.iter().copied().step_by(stride).collect();
    let lon: Vec<f64> = lon_dedup.iter().copied().step_by(stride).collect();
    assert!(
        lat.len() >= 150,
        "need >=150 distinct geographic sites, got {}",
        lat.len()
    );
    let n = lat.len();

    // ---- synthetic response: known spherical signal + fixed-seed noise -----
    // Deterministic LCG so gam and mgcv receive byte-identical y. Noise SD is
    // set from the signal SD to fix the SNR (~10:1 power), exercising the
    // smoother's penalty selection without drowning the truth.
    let truth: Vec<f64> = lat
        .iter()
        .zip(&lon)
        .map(|(&la, &lo)| signal(la, lo))
        .collect();
    let mean_t = truth.iter().sum::<f64>() / n as f64;
    let var_t = truth.iter().map(|t| (t - mean_t).powi(2)).sum::<f64>() / n as f64;
    let noise_sd = (var_t / 10.0).sqrt();

    let mut state: u64 = 0x5EED_0_5_2_F_u64 ^ 0xDEAD_BEEF_CAFE_F00D;
    let mut next_normal = || {
        // Box–Muller from a 64-bit LCG (numerical-recipes constants).
        let mut u = || {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((state >> 11) as f64 + 1.0) / ((1u64 << 53) as f64 + 1.0)
        };
        let (u1, u2) = (u(), u());
        (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
    };
    let y: Vec<f64> = truth.iter().map(|t| t + noise_sd * next_normal()).collect();

    // ---- fit with gam: y ~ sphere(lat, lon, k=30), REML --------------------
    // Write the identical (lat, lon, y) triples to a temp CSV and load through
    // the standard inferred-schema path so gam sees exactly the reference data.
    let dir = std::env::temp_dir().join(format!("gam_sphere_sos_{}", std::process::id()));
    std::fs::create_dir_all(&dir).expect("scratch dir");
    let csv_path = dir.join("sphere.csv");
    {
        let mut f = std::fs::File::create(&csv_path).expect("create temp csv");
        writeln!(f, "lat,lon,y").expect("csv header");
        for i in 0..n {
            writeln!(f, "{:.17e},{:.17e},{:.17e}", lat[i], lon[i], y[i]).expect("csv row");
        }
    }
    let ds = load_csvwith_inferred_schema(&csv_path).expect("load sphere csv");
    let cm = ds.column_map();
    let lat_idx = cm["lat"];
    let lon_idx = cm["lon"];

    let cfg = FitConfig {
        family: Some("gaussian".to_string()),
        ..FitConfig::default()
    };
    let result = fit_from_formula("y ~ sphere(lat, lon, k=30)", &ds, &cfg).expect("gam sphere fit");
    let FitResult::Standard(fit) = result else {
        panic!("expected a standard GAM fit for a Gaussian sphere smooth");
    };
    let gam_edf = fit.fit.edf_total().expect("gam reports total edf");

    // ---- 20x15 evaluation grid over [-80,80] lat x [-179,179] lon ----------
    // Spans both hemispheres and straddles the ±180° seam to expose seam/pole
    // artifacts. Identical grid is handed to mgcv via predict().
    let n_lon = 20usize;
    let n_lat = 15usize;
    let mut grid_lat: Vec<f64> = Vec::with_capacity(n_lon * n_lat);
    let mut grid_lon: Vec<f64> = Vec::with_capacity(n_lon * n_lat);
    for i in 0..n_lat {
        let la = -80.0 + 160.0 * (i as f64) / ((n_lat - 1) as f64);
        for j in 0..n_lon {
            let lo = -179.0 + 358.0 * (j as f64) / ((n_lon - 1) as f64);
            grid_lat.push(la);
            grid_lon.push(lo);
        }
    }
    let ng = grid_lat.len();

    // gam fitted surface at the grid: rebuild the FROZEN design at grid points
    // (identity link => design·beta = fitted mean).
    //
    // CRITICAL: the sphere basis places its kernel centers from the *fit* data
    // (data-dependent CenterStrategy). `fit.resolvedspec` is only the resolved
    // (planned) spec; rebuilding a design straight from it on the grid would
    // re-plan the sphere centers at the grid points, yielding a basis that the
    // fitted `beta` was never estimated against — `design·beta` would be
    // meaningless. We must first freeze the spec against the fit-time design
    // (pinning centers to UserProvided), then evaluate that frozen basis at the
    // grid. This is exactly what gam's own predict path does in main.rs.
    let frozenspec = freeze_term_collection_from_design(&fit.resolvedspec, &fit.design)
        .expect("freeze sphere term collection against fit-time design");
    let mut grid = Array2::<f64>::zeros((ng, ds.headers.len()));
    for k in 0..ng {
        grid[[k, lat_idx]] = grid_lat[k];
        grid[[k, lon_idx]] = grid_lon[k];
    }
    let design = build_term_collection_design(grid.view(), &frozenspec)
        .expect("rebuild frozen sphere design at grid points");
    let gam_surface: Vec<f64> = design.design.apply(&fit.fit.beta).to_vec();
    if let Err(err) = std::fs::remove_dir_all(&dir) {
        eprintln!(
            "temp reference dir cleanup failed for {}: {err}",
            dir.display()
        );
    }

    // ---- fit the SAME data with mgcv bs="sos" and predict on the SAME grid --
    // The reference harness requires all data columns to share one length, so we
    // cannot ship the (longer/shorter) grid as extra columns. Instead we hand
    // mgcv only the fit data and have R *reconstruct the identical grid* with the
    // same closed-form linspace used above (n_lat, n_lon, and the exact bounds),
    // guaranteeing both engines predict on byte-equivalent coordinates.
    let r = run_r(
        &[
            Column::new("lat", &lat),
            Column::new("lon", &lon),
            Column::new("y", &y),
        ],
        &format!(
            r#"
        suppressPackageStartupMessages(library(mgcv))
        fit_df <- data.frame(lat = df$lat, lon = df$lon, y = df$y)
        n_lat <- {n_lat}; n_lon <- {n_lon}
        las <- -80.0 + 160.0 * (0:(n_lat - 1)) / (n_lat - 1)
        los <- -179.0 + 358.0 * (0:(n_lon - 1)) / (n_lon - 1)
        # Row-major over latitude bands then longitude, matching the Rust grid.
        glat <- rep(las, each = n_lon)
        glon <- rep(los, times = n_lat)
        grid_df <- data.frame(lat = glat, lon = glon)
        m <- gam(y ~ s(lat, lon, bs = "sos", k = 30), data = fit_df, method = "REML")
        pr <- as.numeric(predict(m, newdata = grid_df))
        emit("surface", pr)
        emit("edf", sum(m$edf))
        emit("fitted", as.numeric(fitted(m)))
        "#,
            n_lat = n_lat,
            n_lon = n_lon,
        ),
    );
    let mgcv_surface = r.vector("surface");
    let mgcv_edf = r.scalar("edf");
    assert_eq!(
        mgcv_surface.len(),
        ng,
        "mgcv predicted {} grid points, expected {ng}",
        mgcv_surface.len()
    );

    // ---- truth recovery WHERE THE DATA CONSTRAIN THE FIELD ------------------
    // The fit sites cover a small patch of S² (the Fiji/Tonga quakes region);
    // over most of the global grid the surface is extrapolation that no data
    // informs, where each smoother returns its own null-space prior. The
    // noise-free truth at the fit sites is the recovery target the data can
    // certify, and a smoother that chases the noise scores worse on it.
    let mgcv_fitted = r.vector("fitted");
    assert_eq!(
        mgcv_fitted.len(),
        n,
        "mgcv returned {} fitted values, expected {n}",
        mgcv_fitted.len()
    );
    let mut site_rows = Array2::<f64>::zeros((n, ds.headers.len()));
    for i in 0..n {
        site_rows[[i, lat_idx]] = lat[i];
        site_rows[[i, lon_idx]] = lon[i];
    }
    let site_design = build_term_collection_design(site_rows.view(), &frozenspec)
        .expect("rebuild frozen sphere design at the fit sites");
    let gam_fitted: Vec<f64> = site_design.design.apply(&fit.fit.beta).to_vec();
    let gam_site_rmse = rmse(&gam_fitted, &truth);
    let mgcv_site_rmse = rmse(mgcv_fitted, &truth);

    // ---- truth on the SAME grid (the ground-truth field we must recover) ----
    let truth_grid: Vec<f64> = grid_lat
        .iter()
        .zip(&grid_lon)
        .map(|(&la, &lo)| signal(la, lo))
        .collect();
    let signal_range = {
        let mut lo = f64::INFINITY;
        let mut hi = f64::NEG_INFINITY;
        for &t in &truth_grid {
            lo = lo.min(t);
            hi = hi.max(t);
        }
        hi - lo
    };

    // OBJECTIVE accuracy: how well does each engine recover the known field?
    let gam_rmse = rmse(&gam_surface, &truth_grid);
    let mgcv_rmse = rmse(mgcv_surface, &truth_grid);

    // ---- reference closeness, for CONTEXT ONLY (printed, never asserted) ----
    let rel = relative_l2(&gam_surface, mgcv_surface);
    let corr = pearson(&gam_surface, mgcv_surface);
    let edf_rel = (gam_edf - mgcv_edf).abs() / mgcv_edf.abs().max(1.0);

    // ---- intrinsic seam-continuity property (gam, on its own surface) ------
    // Longitude is an angular chart: lon = -179° and lon = +179° are 2° apart on
    // S², so a correct intrinsic smoother must produce nearly-equal values there
    // at every latitude band. A seam discontinuity would betray a chart that
    // treats longitude as a hard boundary. Compare the first (lon=-179) and last
    // (lon=+179) column of gam's own grid against the surface's overall scale.
    let mut seam_num = 0.0;
    let mut surf_sq = 0.0;
    for i in 0..n_lat {
        let west = gam_surface[i * n_lon]; // lon = -179
        let east = gam_surface[i * n_lon + (n_lon - 1)]; // lon = +179
        seam_num += (west - east) * (west - east);
        for j in 0..n_lon {
            let v = gam_surface[i * n_lon + j];
            surf_sq += v * v;
        }
    }
    let seam_rel = (seam_num / surf_sq.max(1e-300)).sqrt();

    let rmse_to_range = gam_rmse / signal_range.max(1e-300);
    eprintln!(
        "sphere(lat,lon,k=30) recover f=a·sin(lat)·cos(lon): n={n} grid={ng} \
         signal_range={signal_range:.4} site_rmse gam={gam_site_rmse:.5} mgcv={mgcv_site_rmse:.5} \
         noise_sd={noise_sd:.4} [context, extrapolation-dominated global grid: \
         gam_rmse={gam_rmse:.4} mgcv_rmse={mgcv_rmse:.4} gam_rmse/range={rmse_to_range:.4}] \
         [context: gam_edf={gam_edf:.3} mgcv_edf={mgcv_edf:.3} edf_rel={edf_rel:.3} \
         rel_l2_to_mgcv={rel:.4} pearson_to_mgcv={corr:.5}] seam_rel={seam_rel:.4}"
    );
    eprintln!(
        "{}",
        QualityPair::error(
            "manifolds",
            "quality_vs_mgcv_sphere_s2_wahba_vs_sos",
            "site_rmse",
            gam_site_rmse,
            "mgcv",
            mgcv_site_rmse,
        )
        .line()
    );
    eprintln!(
        "{}",
        QualityPair::error(
            "manifolds",
            "quality_vs_mgcv_sphere_s2_wahba_vs_sos",
            "grid_rmse",
            gam_rmse,
            "mgcv",
            mgcv_rmse,
        )
        .line()
    );

    // PRIMARY claim: gam recovers the known field where the data constrain it.
    // The raw observations score about the noise SD against the truth, so an
    // informative smoother must do better than that. The grid_rmse pair above is
    // emitted but not asserted: across regions no data constrain, the surface is
    // each smoother's prior rather than a recovery of the truth.
    assert!(
        gam_site_rmse < noise_sd,
        "gam sphere smooth does not recover the known S² field at the fit sites \
         better than the raw observations: site RMSE={gam_site_rmse:.5} vs noise SD {noise_sd:.4}"
    );

    // MATCH-OR-BEAT the mature reference on the SAME truth-recovery metric.
    assert!(
        gam_site_rmse <= mgcv_site_rmse * 1.10,
        "gam recovers the S² field at the fit sites less accurately than mgcv bs=sos: \
         gam={gam_site_rmse:.5} > 1.10 * mgcv={mgcv_site_rmse:.5}"
    );

    // Intrinsic correctness: gam's surface must be continuous across the seam.
    // 2°-apart longitudes can differ slightly under a real signal, so the seam
    // jump must be a small fraction of the surface scale — anything larger is an
    // antimeridian discontinuity, a real geometry bug.
    assert!(
        seam_rel < 0.05,
        "gam sphere surface is discontinuous across the ±180° longitude seam: seam_rel={seam_rel:.4}"
    );
}
