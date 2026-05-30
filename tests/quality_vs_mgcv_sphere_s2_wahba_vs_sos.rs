//! End-to-end quality: gam's intrinsic S² (sphere) smooth must reproduce the
//! *same fitted spherical surface* as mgcv's spline-on-sphere `bs="sos"` — the
//! mature, standard penalized smoother on the 2-sphere.
//!
//! Mature comparator: `mgcv::gam(y ~ s(lat, lon, bs="sos", k=30), method="REML")`.
//! mgcv's `bs="sos"` is Wahba's spline-on-sphere: a penalized thin-plate-style
//! spectral kernel on S² that takes latitude/longitude **in degrees** and is the
//! de-facto reference for smoothing geographically-distributed data. gam's
//! `sphere(lat, lon, k=...)` builds an intrinsic Wahba/Sobolev kernel on the same
//! manifold (also degree-valued lat/lon by default). Both are REML-fit with the
//! same basis count `k`, so they target the same penalized objective on S² and
//! their fitted surfaces must essentially coincide.
//!
//! DISTINCTIVE-AXIS NOTE: sphere smoothing is a fragmented corner of the
//! ecosystem (spheresmooth / Directional / rcosmo are single-purpose and do not
//! offer an integrated penalized-REML GAM). mgcv `bs="sos"` is the *only* mature,
//! widely-trusted integrated comparator, so we (a) compare gam head-to-head with
//! it on a 20x15 lat/lon evaluation grid, and (b) additionally assert an
//! INTRINSIC correctness property that any honest S² smoother must satisfy: the
//! fitted surface is continuous across the ±180° longitude seam (no antimeridian
//! discontinuity), because longitude is an angular chart of the sphere and the
//! seam is not a real boundary. The grid deliberately spans both hemispheres and
//! straddles the seam to expose seam/pole artifacts.
//!
//! Data: real geographic coordinates (Latitude/Longitude) from the HGDP+1kG
//! panel, with a *known* smooth spherical signal f(lat,lon)=a·sin(lat)·cos(lon)
//! (a genuine low-frequency field on S²) plus fixed-seed Gaussian noise at a
//! controlled SNR. The identical (lat, lon, y) triples are handed to both engines.

use gam::matrix::LinearOperator;
use gam::smooth::build_term_collection_design;
use gam::test_support::reference::{Column, pearson, relative_l2, run_r};
use gam::{
    FitConfig, FitResult, fit_from_formula, init_parallelism, load_csvwith_inferred_schema,
};
use ndarray::Array2;
use std::io::Write as _;
use std::path::Path;

const HGDP_TSV: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/bench/datasets/hgdp_1kg_pc_data.tsv"
);

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

    // ---- read real lat/lon from the HGDP+1kG TSV (tab-separated) -----------
    let raw = std::fs::read_to_string(Path::new(HGDP_TSV)).expect("read hgdp tsv");
    let mut lines = raw.lines();
    let header = lines.next().expect("tsv header");
    let cols: Vec<&str> = header.split('\t').collect();
    let lat_col = cols
        .iter()
        .position(|c| *c == "Latitude")
        .expect("Latitude column");
    let lon_col = cols
        .iter()
        .position(|c| *c == "Longitude")
        .expect("Longitude column");

    let mut lat_all: Vec<f64> = Vec::new();
    let mut lon_all: Vec<f64> = Vec::new();
    for line in lines {
        if line.trim().is_empty() {
            continue;
        }
        let fields: Vec<&str> = line.split('\t').collect();
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
    // The HGDP coordinates are heavily duplicated (many samples per population),
    // which makes a sphere smooth degenerate; deduplicate to distinct sites and
    // keep a fixed-stride subsample so the design is well-spread over S².
    let mut seen: std::collections::BTreeSet<(i64, i64)> = std::collections::BTreeSet::new();
    let mut lat: Vec<f64> = Vec::new();
    let mut lon: Vec<f64> = Vec::new();
    for (la, lo) in lat_all.iter().zip(&lon_all) {
        let key = ((la * 100.0).round() as i64, (lo * 100.0).round() as i64);
        if seen.insert(key) {
            lat.push(*la);
            lon.push(*lo);
        }
    }
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
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((state >> 11) as f64 + 1.0) / ((1u64 << 53) as f64 + 1.0)
        };
        let (u1, u2) = (u(), u());
        (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
    };
    let y: Vec<f64> = truth
        .iter()
        .map(|t| t + noise_sd * next_normal())
        .collect();

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
    let result =
        fit_from_formula("y ~ sphere(lat, lon, k=30)", &ds, &cfg).expect("gam sphere fit");
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

    // gam fitted surface at the grid: rebuild the frozen design at grid points
    // (identity link => design·beta = fitted mean).
    let mut grid = Array2::<f64>::zeros((ng, ds.headers.len()));
    for k in 0..ng {
        grid[[k, lat_idx]] = grid_lat[k];
        grid[[k, lon_idx]] = grid_lon[k];
    }
    let design = build_term_collection_design(grid.view(), &fit.resolvedspec)
        .expect("rebuild sphere design at grid points");
    let gam_surface: Vec<f64> = design.design.apply(&fit.fit.beta).to_vec();
    std::fs::remove_dir_all(&dir).ok();

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

    // ---- compare fitted surfaces -------------------------------------------
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

    eprintln!(
        "sphere(lat,lon,k=30) vs mgcv bs=sos: n={n} grid={ng} \
         gam_edf={gam_edf:.3} mgcv_edf={mgcv_edf:.3} edf_rel={edf_rel:.3} \
         rel_l2={rel:.4} pearson={corr:.5} seam_rel={seam_rel:.4}"
    );

    // Both engines REML-fit the SAME penalized Wahba kernel on S² with the same
    // basis count, so their fitted surfaces must track each other closely. The
    // two kernels differ slightly in truncation/center placement, so we allow a
    // looser-than-1D margin while still asserting genuine surface agreement:
    // pearson > 0.95 (shape) and rel_l2 < 0.20 (magnitude). These are tight
    // enough that a wrong kernel, seam wrap, or penalty mismatch fails.
    assert!(
        corr > 0.95,
        "gam and mgcv sphere surfaces should agree in shape: pearson={corr:.5}"
    );
    assert!(
        rel < 0.20,
        "gam and mgcv sphere surfaces diverge in magnitude: rel_l2={rel:.4}"
    );
    // Same basis count k=30 under REML on the same manifold => EDF must be in the
    // same ballpark; 15% per the spec (basis-convention differences allowed).
    assert!(
        edf_rel < 0.15,
        "effective degrees of freedom disagree: gam={gam_edf:.3} mgcv={mgcv_edf:.3} (rel={edf_rel:.3})"
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
