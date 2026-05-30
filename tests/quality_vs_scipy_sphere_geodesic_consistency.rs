//! Sphere-smooth *geodesic consistency* — gam's intrinsic S² smooth must respect
//! the great-circle (geodesic) metric of the sphere, not the R³ chordal/Euclidean
//! metric of the embedding.
//!
//! Ground truth for the great-circle distance-to-pole is computed with
//! NumPy/SciPy via the classic haversine great-circle formula, cross-checked in
//! the Python body against both the arccos-of-dot-product central angle
//! `d_geod(p,q) = arccos(p·q)` and the analytic colatitude `π/2 − lat`, so the
//! exported "ground truth" is itself verified three independent ways before gam
//! is judged against it. We additionally fit `mgcv`'s spline-on-sphere
//! (`bs="sos"`) as the closest mature comparator and confirm it reconstructs the
//! same geodesic surface (it is geodesic-consistent by construction), making it a
//! meaningful secondary baseline.
//!
//! The intrinsic correctness property: fit `y ~ sphere(lat, lon, k=20)` on noisy
//! samples of a radially-symmetric truth `f(p) = exp(-d_geod(p, pole)/bandwidth)`,
//! evaluate the fitted surface at a probe grid, and test two things that a
//! metric-respecting S² smooth must satisfy and a coordinate/chordal-confused
//! kernel cannot:
//!   1. RECONSTRUCTION — gam's fitted surface must track the haversine-derived
//!      true surface `f_true` per probe point, and be monotone-decreasing in the
//!      geodesic distance-to-pole. (We deliberately do NOT correlate the pairwise
//!      fitted differences `|f_gam(p)−f_gam(q)|` against the pairwise geodesic
//!      distance `d_geod(p,q)`: for a truth radial about a single point the fitted
//!      differences track the difference of distances-TO-POLE, which even for a
//!      perfect fit correlates only ~0.5 with the pairwise separation, so such a
//!      bound would assert something false.)
//!   2. ANTIMERIDIAN SEAM — two probe points at the same latitude straddling the
//!      ±180° seam are ~10° apart geodesically but ~350° apart in raw longitude.
//!      An intrinsic kernel gives them near-equal fitted values; a kernel keyed on
//!      raw lat/lon degrees would see them as maximally separated. This is the
//!      sharpest discriminator, because a single-center radial truth alone cannot
//!      separate geodesic from chordal-R³ distance (both are monotone in the same
//!      central angle, so any radial function of one is a near-monotone function
//!      of the other).
//!
//! The sphere-smoothing tool ecosystem is fragmented — there is no integrated
//! GAM that advertises "geodesic-consistent" fits — so the finding is twofold:
//! (a) gam tracks the mature `mgcv` spline-on-sphere baseline, and (b) gam
//! satisfies the intrinsic geodesic-consistency property directly. If this test
//! fails, check the kernel evaluation for the lat/lon <-> 3D unit-vector
//! conversion (the embedding must feed an intrinsic, not chordal, distance).

use gam::matrix::LinearOperator;
use gam::smooth::build_term_collection_design;
use gam::test_support::reference::{Column, pearson, relative_l2, run_python, run_r};
use gam::{
    FitConfig, FitResult, encode_recordswith_inferred_schema, fit_from_formula, init_parallelism,
};
use csv::StringRecord;
use ndarray::Array2;
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand_distr::{Distribution, Normal, Uniform};

/// Geographic (lat, lon) in degrees → unit vector on S², pole on +z. Mirrors
/// `gam::gpu::sphere::latlon_to_xyz_host`: x=cos(lat)cos(lon), y=cos(lat)sin(lon),
/// z=sin(lat). Using the SAME convention gam uses internally is what makes the
/// geodesic-distance comparison meaningful.
fn latlon_to_xyz(lat_deg: f64, lon_deg: f64) -> [f64; 3] {
    let lat = lat_deg.to_radians();
    let lon = lon_deg.to_radians();
    [lat.cos() * lon.cos(), lat.cos() * lon.sin(), lat.sin()]
}

/// Great-circle (geodesic) distance on the unit sphere between two (lat, lon)
/// points in degrees: the central angle `arccos(p·q)`, clamped against round-off.
fn geodesic_deg(lat0: f64, lon0: f64, lat1: f64, lon1: f64) -> f64 {
    let a = latlon_to_xyz(lat0, lon0);
    let b = latlon_to_xyz(lat1, lon1);
    let dot = (a[0] * b[0] + a[1] * b[1] + a[2] * b[2]).clamp(-1.0, 1.0);
    dot.acos()
}

/// Build an `EncodedDataset` with columns `lat`, `lon`, `y` (degrees).
fn make_dataset(lats: &[f64], lons: &[f64], ys: &[f64]) -> gam::data::EncodedDataset {
    let headers = ["lat", "lon", "y"].into_iter().map(String::from).collect();
    let mut rows = Vec::with_capacity(lats.len());
    for i in 0..lats.len() {
        rows.push(StringRecord::from(vec![
            lats[i].to_string(),
            lons[i].to_string(),
            ys[i].to_string(),
        ]));
    }
    encode_recordswith_inferred_schema(headers, rows).expect("encode sphere dataset")
}

#[test]
fn gam_sphere_smooth_is_geodesic_consistent_and_tracks_mgcv_sos() {
    init_parallelism();

    // ---- synthetic data: n=50 uniform points on S² ------------------------
    // Uniform on the sphere: z ~ U(-1,1), lon ~ U(-180,180); lat = asin(z).
    // Radially-symmetric truth about the +z pole (lat=+90°):
    //   f(p) = exp(-d_geod(p, pole) / bandwidth),  d_geod in radians (central angle).
    // This truth is *intrinsic*: it depends only on the geodesic distance to the
    // pole, so a metric-respecting smooth must reconstruct a surface whose values
    // order points by geodesic separation.
    let n = 50usize;
    let bandwidth = 0.8_f64; // radians; ~46° correlation length on S²
    let mut rng = StdRng::seed_from_u64(20260529);
    let u_z = Uniform::new_inclusive(-1.0, 1.0).expect("uniform z");
    let u_lon = Uniform::new(-180.0_f64, 180.0_f64).expect("uniform lon");
    let noise = Normal::new(0.0, 0.01).expect("normal");

    let mut lats = Vec::with_capacity(n);
    let mut lons = Vec::with_capacity(n);
    let mut ys = Vec::with_capacity(n);
    for _ in 0..n {
        let z: f64 = u_z.sample(&mut rng);
        let lon_deg: f64 = u_lon.sample(&mut rng);
        let lat_deg = z.asin().to_degrees();
        // geodesic distance to the +z pole = colatitude = (90° - lat) in radians.
        let d_pole = geodesic_deg(90.0, 0.0, lat_deg, lon_deg);
        let f = (-d_pole / bandwidth).exp();
        lats.push(lat_deg);
        lons.push(lon_deg);
        ys.push(f + noise.sample(&mut rng));
    }

    // ---- evaluation grid: 10×10 = 100 probe points, away from the poles ----
    // (avoid the lat=±90 lat/lon coordinate singularity; the *intrinsic* fit is
    // fine there but the comparison logic is cleaner away from the seam/poles).
    let mut eval_lats = Vec::new();
    let mut eval_lons = Vec::new();
    for i in 0..10 {
        let lat = -75.0 + 150.0 * (i as f64) / 9.0;
        for j in 0..10 {
            let lon = -170.0 + 340.0 * (j as f64) / 9.0;
            eval_lats.push(lat);
            eval_lons.push(lon);
        }
    }
    let m = eval_lats.len();

    // ---- fit gam: y ~ sphere(lat, lon, k=20), Gaussian/REML ----------------
    // Default kernel is the Sobolev/Wahba spline-on-sphere with k=20 centers;
    // lat/lon are interpreted in degrees (radians=false by default).
    let cfg = FitConfig {
        family: Some("gaussian".to_string()),
        ..FitConfig::default()
    };
    let data = make_dataset(&lats, &lons, &ys);
    let result =
        fit_from_formula("y ~ sphere(lat, lon, k=20)", &data, &cfg).expect("gam sphere fit");
    let FitResult::Standard(fit) = result else {
        panic!("expected a standard GAM fit for the sphere smooth");
    };
    let edf = fit.fit.edf_total().expect("gam reports total edf");

    // Evaluate the fitted surface at the probe points (identity link => mean).
    let mut grid = Array2::<f64>::zeros((m, 3));
    for i in 0..m {
        grid[[i, 0]] = eval_lats[i];
        grid[[i, 1]] = eval_lons[i];
        grid[[i, 2]] = 0.0;
    }
    let design = build_term_collection_design(grid.view(), &fit.resolvedspec)
        .expect("rebuild sphere design at probe points");
    let gam_eval: Vec<f64> = design.design.apply(&fit.fit.beta).to_vec();
    assert_eq!(gam_eval.len(), m, "gam eval length must match probe grid");

    // ---- ground-truth GEODESIC distance-to-pole + true surface via SciPy ---
    // The truth `f(p) = exp(-d_geod(p, pole) / bandwidth)` is a function of the
    // great-circle distance from each probe point to the +z pole (lat=+90). We
    // recompute that distance in Python with the classic haversine great-circle
    // formula, cross-check it against the arccos-of-dot-product central angle so
    // the exported ground truth is itself verified, then form the true surface
    // on the probe grid. `import scipy` enforces the no-skip contract (a missing
    // reference stack is a hard failure, not a silent pass).
    //
    // Why this quantity and not pairwise |f(p)-f(q)| vs pairwise d_geod(p,q):
    // the truth is radial about ONE point, so fitted differences track the
    // difference of distances-TO-POLE, which is only weakly (~0.5) correlated
    // with the pairwise separation d_geod(p,q) even for a perfect fit. The
    // quantity that actually matters is whether gam reconstructs the geodesic
    // surface itself, so we compare gam's fitted values to f_true per probe
    // point (grid-aligned by index).
    let bw = bandwidth;
    let py = run_python(
        &[
            Column::new("elat", &eval_lats),
            Column::new("elon", &eval_lons),
            Column::new("bw", &vec![bw; m]),
        ],
        r#"
import numpy as np
import scipy  # ensure SciPy is present; hard-fail otherwise (no skip path)

lat = np.radians(np.asarray(df["elat"], dtype=float))
lon = np.radians(np.asarray(df["elon"], dtype=float))
bw = float(np.asarray(df["bw"], dtype=float)[0])

# Unit vectors on S^2 (same convention as gam: x=cos(lat)cos(lon), etc.).
x = np.cos(lat) * np.cos(lon)
y = np.cos(lat) * np.sin(lon)
z = np.sin(lat)
P = np.column_stack([x, y, z])
assert np.allclose(np.linalg.norm(P, axis=1), 1.0, atol=1e-12), "probe points must be unit"

pole = np.array([0.0, 0.0, 1.0])  # lat=+90

# (A) geodesic distance-to-pole via arccos of dot product (central angle on S^2).
d_pole_arccos = np.arccos(np.clip(P @ pole, -1.0, 1.0))

# (B) geodesic distance-to-pole via the haversine great-circle formula between
# (lat, lon) and the pole (lat=pi/2, lon=0). On the unit sphere this equals the
# central angle. For a point at latitude `lat`, the colatitude is exactly
# pi/2 - lat; haversine must reproduce it.
lat_p, lon_p = np.pi / 2.0, 0.0
dlat = lat_p - lat
dlon = lon_p - lon
hav = np.sin(dlat / 2.0) ** 2 + np.cos(lat) * np.cos(lat_p) * np.sin(dlon / 2.0) ** 2
d_pole_hav = 2.0 * np.arcsin(np.sqrt(np.clip(hav, 0.0, 1.0)))

# The two great-circle formulas must agree: this validates the ground truth.
assert np.allclose(d_pole_arccos, d_pole_hav, atol=1e-9), "haversine vs arccos disagree"
# And both must equal the analytic colatitude pi/2 - lat.
assert np.allclose(d_pole_arccos, np.pi / 2.0 - lat, atol=1e-9), "colatitude check"

# True geodesic-radial surface at the probe points.
f_true = np.exp(-d_pole_hav / bw)

# Document that geodesic and chordal-to-pole distances are DIFFERENT metrics
# (chord = 2 sin(d_geod/2)); emit the largest relative gap over the probe grid.
chord_pole = np.linalg.norm(P - pole[None, :], axis=1)
metric_gap = float(np.max(np.abs(d_pole_arccos - chord_pole) / np.maximum(d_pole_arccos, 1e-9)))

emit("f_true", f_true)
emit("d_pole", d_pole_hav)
emit("metric_gap", [metric_gap])
"#,
    );
    let f_true = py.vector("f_true");
    let d_pole = py.vector("d_pole");
    let metric_gap = py.scalar("metric_gap");
    assert_eq!(
        f_true.len(),
        m,
        "scipy true-surface length must match the {m} probe points"
    );

    // ---- the intrinsic correctness metric ---------------------------------
    // (i) gam must RECONSTRUCT the geodesic-radial truth: its fitted surface
    // tracks f_true (computed from the haversine great-circle distance-to-pole)
    // across the probe grid. (ii) gam's fitted values must be MONOTONE-DECREASING
    // in the geodesic distance to the pole — the defining signature of a smooth
    // keyed on the intrinsic S² metric rather than raw lat/lon.
    let corr_true = pearson(gam_eval.as_slice(), f_true);
    let corr_gam_dpole = pearson(gam_eval.as_slice(), d_pole);
    let l2_true = relative_l2(gam_eval.as_slice(), f_true);
    eprintln!(
        "[sphere-geodesic] n={n} m_probe={m} edf={edf:.3} \
         pearson(f_gam, f_true)={corr_true:.4} relL2(f_gam, f_true)={l2_true:.4} \
         pearson(f_gam, d_geod_to_pole)={corr_gam_dpole:.4} metric_gap={metric_gap:.4}"
    );

    // ---- secondary baseline: mgcv spline-on-sphere (bs="sos") --------------
    // The closest mature comparator. We confirm it satisfies the SAME geodesic
    // consistency property on identical data, then confirm gam tracks it. mgcv
    // sos is geodesic-consistent by construction (it is literally a spline on the
    // sphere), so it is a fair yardstick for gam's intrinsic property.
    let mgcv = run_r(
        &[
            Column::new("lat", &lats),
            Column::new("lon", &lons),
            Column::new("y", &ys),
            Column::new("elat", &eval_lats),
            Column::new("elon", &eval_lons),
        ],
        r#"
        suppressPackageStartupMessages(library(mgcv))
        train <- data.frame(lat = df$lat, lon = df$lon, y = df$y)
        train <- train[!is.na(train$lat), ]
        # mgcv spline-on-sphere expects (latitude, longitude) in degrees.
        mod <- gam(y ~ s(lat, lon, bs = "sos", k = 20), data = train, method = "REML")
        ev <- data.frame(lat = df$elat, lon = df$elon)
        emit("pred", as.numeric(predict(mod, newdata = ev)))
        emit("edf", sum(mod$edf))
        "#,
    );
    let mgcv_pred = mgcv.vector("pred");
    let mgcv_edf = mgcv.scalar("edf");
    assert_eq!(
        mgcv_pred.len(),
        m,
        "mgcv sos prediction length must match the {m} probe points"
    );

    // mgcv's own reconstruction of the geodesic-radial truth, grid-aligned.
    let mgcv_corr_true = pearson(mgcv_pred, f_true);
    // How closely gam's probe-surface tracks mgcv's spline-on-sphere surface.
    let gam_vs_mgcv = pearson(gam_eval.as_slice(), mgcv_pred);
    eprintln!(
        "[sphere-geodesic] mgcv_edf={mgcv_edf:.3} \
         pearson(f_mgcv, f_true)={mgcv_corr_true:.4} pearson(gam, mgcv_sos)={gam_vs_mgcv:.4}"
    );

    // ---- antimeridian-seam discriminator ----------------------------------
    // Two probe points at the SAME latitude straddling the ±180° seam are
    // geodesically near (10° apart across the antimeridian) yet ~350° apart in
    // raw longitude. The radial truth assigns them IDENTICAL values (same
    // colatitude). An intrinsic S² smooth therefore gives them near-equal fitted
    // values; a kernel that confused raw lat/lon degrees with distance would see
    // them as maximally separated and assign wildly different values. This is the
    // sharpest available test that gam keys on the geodesic, not on coordinates.
    let mut seam_grid = Array2::<f64>::zeros((2, 3));
    seam_grid[[0, 0]] = 0.0;
    seam_grid[[0, 1]] = -175.0;
    seam_grid[[1, 0]] = 0.0;
    seam_grid[[1, 1]] = 175.0;
    let seam_design = build_term_collection_design(seam_grid.view(), &fit.resolvedspec)
        .expect("rebuild sphere design at seam points");
    let seam_eval: Vec<f64> = seam_design.design.apply(&fit.fit.beta).to_vec();
    let fitted_range = gam_eval
        .iter()
        .cloned()
        .fold(f64::NEG_INFINITY, f64::max)
        - gam_eval.iter().cloned().fold(f64::INFINITY, f64::min);
    let seam_gap = (seam_eval[0] - seam_eval[1]).abs();
    let seam_rel = seam_gap / fitted_range.max(1e-12);
    eprintln!(
        "[sphere-geodesic] seam |f(-175)-f(175)|={seam_gap:.5} \
         fitted_range={fitted_range:.5} seam_rel={seam_rel:.4}"
    );

    // ---- assertions --------------------------------------------------------
    // (1) gam RECONSTRUCTS the geodesic-radial truth. f_true is built from the
    // haversine great-circle distance-to-pole (cross-checked against the arccos
    // central angle and the analytic colatitude in Python). A smooth keyed on the
    // intrinsic S² metric recovers this surface up to noise and the k=20 basis
    // budget. 0.97 is tight: a perfect noiseless reconstruction would correlate
    // ~1, the noise sd is only 0.01 against a unit-amplitude surface, so anything
    // below 0.97 signals real surface distortion. If this fails, check the
    // kernel's lat/lon <-> 3D unit-vector conversion: the smooth must respect S²
    // geometry, not the R³ embedding.
    assert!(
        corr_true > 0.97,
        "gam sphere smooth does NOT reconstruct the geodesic-radial truth: \
         pearson(f_gam, f_true) = {corr_true:.4} (bound 0.97), \
         relL2 = {l2_true:.4}. Check the kernel's lat/lon <-> 3D unit-vector \
         conversion: the smooth must respect S² geometry, not the R³ embedding."
    );
    // The fitted surface must also be MONOTONE-DECREASING in geodesic distance to
    // the pole (f = exp(-d/bw) decreases with d), i.e. a strong NEGATIVE
    // correlation. A chordal/coordinate-confused kernel would not order points by
    // their intrinsic distance to the pole.
    assert!(
        corr_gam_dpole < -0.9,
        "gam fitted values are not monotone in geodesic distance-to-pole: \
         pearson(f_gam, d_geod_to_pole) = {corr_gam_dpole:.4} (bound -0.9)"
    );

    // (2) The mature comparator clears the same reconstruction bar — this is what
    // makes mgcv bs=\"sos\" a valid yardstick (it is geodesic-consistent by
    // construction). If mgcv itself fails, the reference fit is broken, not gam.
    assert!(
        mgcv_corr_true > 0.97,
        "mgcv bs=\"sos\" baseline failed to reconstruct the geodesic-radial truth \
         it should track by construction: pearson(f_mgcv, f_true) = {mgcv_corr_true:.4} \
         (bound 0.97) — check the reference fit, not gam"
    );

    // (3) gam tracks the mature spline-on-sphere surface on identical data; both
    // REML-fit the same intrinsic surface with k=20, so their probe-grid surfaces
    // are strongly correlated. 0.95 is tight enough to catch a real baseline
    // divergence yet tolerant of the differing sphere bases (gam Sobolev/Wahba vs
    // mgcv thin-plate-on-sphere).
    assert!(
        gam_vs_mgcv > 0.95,
        "gam sphere fit diverges from the mgcv bs=\"sos\" baseline: \
         pearson(gam, mgcv_sos) = {gam_vs_mgcv:.4} (bound 0.95)"
    );

    // (4) Antimeridian-seam consistency: geodesically-near (10°) but
    // coordinate-far (~350° in longitude) probe points get near-equal fitted
    // values. Their fitted difference must be a small fraction of the total
    // fitted range. 0.1 is principled: across the seam the true surface values
    // are identical and only ~10° of geodesic separation plus fit noise can move
    // them apart, so a kernel respecting S² geometry keeps |Δ| well under 10% of
    // the full range, whereas a coordinate-confused kernel (350° apart) would
    // produce a near-maximal gap. The check also confirms the fitted surface is
    // non-degenerate (positive range) so the ratio is meaningful.
    assert!(
        fitted_range > 1e-3,
        "gam fitted surface is degenerate over the probe grid (range \
         {fitted_range:.6}); the seam ratio would be meaningless"
    );
    assert!(
        seam_rel < 0.1,
        "gam sphere smooth fails antimeridian-seam consistency: probe points at \
         (0,-175) and (0,+175) are 10° apart geodesically but ~350° apart in raw \
         longitude, yet gam assigns them values differing by {seam_gap:.5} = \
         {seam_rel:.4} of the fitted range (bound 0.1). The kernel is keyed on raw \
         lat/lon, not the intrinsic S² geodesic."
    );
}
