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
//! The intrinsic correctness property: fit gam's spline-on-sphere — the SAME
//! Laplace-Beltrami spherical-harmonic construction as mgcv `bs="sos"`, i.e.
//! `y ~ sphere(lat, lon, kernel=harmonic, degree=4)` — on noisy samples of a
//! radially-symmetric truth `f(p) = exp(-d_geod(p, pole)/bandwidth)`,
//! evaluate the fitted surface at a probe grid, and test two things that a
//! metric-respecting S² smooth must satisfy and a coordinate/chordal-confused
//! kernel cannot:
//!   1. RECONSTRUCTION — gam's fitted surface must recover the haversine-derived
//!      true surface `f_true` per probe point with small absolute RMSE, and be
//!      monotone-decreasing in the geodesic distance-to-pole. (We deliberately
//!      do NOT correlate the pairwise
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
//! OBJECTIVE metric asserted (no "matches a peer tool" criterion):
//!   * PRIMARY — TRUTH RECOVERY: `RMSE(f_gam, f_true)` against the analytic
//!     geodesic-radial truth `f_true = exp(-d_geod(p,pole)/bw)`, where the
//!     distance-to-pole is the EXACT great-circle central angle (cross-checked
//!     three ways in Python: arccos-of-dot-product, haversine, and the analytic
//!     colatitude `π/2 − lat`). That makes `f_true` mathematical GROUND TRUTH,
//!     not a peer-tool fit, so matching it IS an objective accuracy claim. We
//!     require gam's reconstruction error to be a small fraction of the unit
//!     signal amplitude.
//!   * STRUCTURE: the fitted surface is monotone-decreasing in geodesic
//!     distance-to-pole, and antimeridian-seam consistent (geodesically-near but
//!     coordinate-far probe points get near-equal values).
//!   * BASELINE TO MATCH-OR-BEAT: `mgcv` spline-on-sphere (`bs="sos"`) is fit on
//!     identical data and its OWN reconstruction error against the same truth is
//!     measured. gam must be at least as ACCURATE (RMSE within 10% of mgcv's).
//!     We never assert that gam's fitted surface is *close to mgcv's* fitted
//!     surface — two tools agreeing proves nothing about correctness; only error
//!     against the analytic truth is an objective quality claim.
//! If this test fails, check the kernel evaluation for the lat/lon <-> 3D
//! unit-vector conversion (the embedding must feed an intrinsic, not chordal,
//! distance).

use csv::StringRecord;
use gam::matrix::LinearOperator;
use gam::smooth::build_term_collection_design;
use gam::test_support::reference::{Column, pearson, relative_l2, rmse, run_python, run_r};
use gam::{
    FitConfig, FitResult, encode_recordswith_inferred_schema, fit_from_formula, init_parallelism,
};
use ndarray::Array2;
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand_distr::{Distribution, Normal, Uniform};

/// Geographic (lat, lon) in degrees → unit vector on S², pole on +z. Mirrors
/// `gam::terms::basis::sphere_gpu::latlon_to_xyz_host`: x=cos(lat)cos(lon), y=cos(lat)sin(lon),
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
fn gam_sphere_smooth_recovers_geodesic_truth_at_least_as_well_as_mgcv_sos() {
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

    // ---- fit gam's spline-on-sphere, the SAME construction as mgcv bs="sos" -
    // mgcv `bs="sos"` is the Laplace-Beltrami spherical-harmonic spline on S²
    // (Wood §5.6.2). gam's apples-to-apples analog is the `harmonic` sphere method
    // (real spherical harmonics with the `[l(l+1)]^m` Laplace-Beltrami eigenvalue
    // penalty), NOT the Sobolev reproducing-kernel `sphere()` default, which is a
    // related-but-distinct smoother. Degree 4 gives a basis dimension
    // `L(L+2) = 24`, matching mgcv's `k=20`, so the two tools are fit at comparable
    // resolution and the match-or-beat claim is between the SAME smoother family.
    // lat/lon are interpreted in degrees (radians=false by default).
    let cfg = FitConfig {
        family: Some("gaussian".to_string()),
        ..FitConfig::default()
    };
    let data = make_dataset(&lats, &lons, &ys);
    let result = fit_from_formula(
        "y ~ sphere(lat, lon, kernel=harmonic, degree=4)",
        &data,
        &cfg,
    )
    .expect("gam sphere fit");
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
    // PRIMARY objective metric: absolute reconstruction error against the
    // analytic geodesic-radial truth. The truth ranges over (0, 1] (a unit
    // amplitude exp decay), so an RMSE expressed as a fraction of that amplitude
    // is directly interpretable.
    let gam_rmse = rmse(gam_eval.as_slice(), f_true);
    let signal_amp = f_true.iter().cloned().fold(f64::NEG_INFINITY, f64::max)
        - f_true.iter().cloned().fold(f64::INFINITY, f64::min);
    eprintln!(
        "[sphere-geodesic] n={n} m_probe={m} edf={edf:.3} \
         rmse(f_gam, f_true)={gam_rmse:.5} signal_amp={signal_amp:.4} \
         pearson(f_gam, f_true)={corr_true:.4} relL2(f_gam, f_true)={l2_true:.4} \
         pearson(f_gam, d_geod_to_pole)={corr_gam_dpole:.4} metric_gap={metric_gap:.4}"
    );

    // ---- baseline to match-or-beat: mgcv spline-on-sphere (bs="sos") -------
    // The closest mature comparator. We fit it on IDENTICAL data and measure its
    // OWN reconstruction error against the analytic truth, turning mgcv into an
    // accuracy yardstick gam must match-or-beat. We do NOT assert that gam's
    // fitted surface resembles mgcv's fitted surface — two tools agreeing on a
    // noisy fit is not a correctness claim; only error against the analytic
    // geodesic truth is.
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

    // mgcv's OWN reconstruction error against the analytic geodesic-radial truth,
    // grid-aligned. This is the baseline accuracy gam must match-or-beat.
    let mgcv_rmse = rmse(mgcv_pred, f_true);
    let mgcv_corr_true = pearson(mgcv_pred, f_true);
    eprintln!(
        "[sphere-geodesic] mgcv_edf={mgcv_edf:.3} rmse(f_mgcv, f_true)={mgcv_rmse:.5} \
         pearson(f_mgcv, f_true)={mgcv_corr_true:.4} \
         rmse_ratio(gam/mgcv)={:.4}",
        gam_rmse / mgcv_rmse.max(1e-12)
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
    let fitted_range = gam_eval.iter().cloned().fold(f64::NEG_INFINITY, f64::max)
        - gam_eval.iter().cloned().fold(f64::INFINITY, f64::min);
    let seam_gap = (seam_eval[0] - seam_eval[1]).abs();
    let seam_rel = seam_gap / fitted_range.max(1e-12);
    eprintln!(
        "[sphere-geodesic] seam |f(-175)-f(175)|={seam_gap:.5} \
         fitted_range={fitted_range:.5} seam_rel={seam_rel:.4}"
    );

    // ---- assertions --------------------------------------------------------
    // (1) PRIMARY — TRUTH RECOVERY. gam must RECONSTRUCT the analytic
    // geodesic-radial truth with small absolute error. f_true is built from the
    // exact great-circle distance-to-pole (cross-checked against the arccos
    // central angle and the analytic colatitude in Python), so it is mathematical
    // ground truth, not a peer-tool fit — asserting gam recovers it IS an
    // objective accuracy claim. The truth has unit amplitude (~1.0) and the
    // additive noise sd is only 0.01, so a smooth that respects the intrinsic S²
    // metric reconstructs it to a small fraction of the signal amplitude. We
    // require RMSE <= 0.05 = 5% of the unit signal amplitude: well within reach of
    // a metric-correct fit at k=20, while a chordal/coordinate-confused kernel
    // distorts the surface and blows past it. If this fails, check the kernel's
    // lat/lon <-> 3D unit-vector conversion: the smooth must respect S² geometry,
    // not the R³ embedding.
    assert!(
        gam_rmse <= 0.05,
        "gam sphere smooth does NOT recover the geodesic-radial truth: \
         rmse(f_gam, f_true) = {gam_rmse:.5} (bound 0.05, signal amplitude \
         {signal_amp:.4}); pearson = {corr_true:.4}, relL2 = {l2_true:.4}. Check \
         the kernel's lat/lon <-> 3D unit-vector conversion: the smooth must \
         respect S² geometry, not the R³ embedding."
    );

    // (2) STRUCTURE — the fitted surface must be MONOTONE-DECREASING in geodesic
    // distance to the pole (f = exp(-d/bw) decreases with d), i.e. a strong
    // NEGATIVE correlation. This is a property of the fit itself (gam's values vs
    // the analytic distance), not a comparison to any tool. A chordal/
    // coordinate-confused kernel would not order points by their intrinsic
    // distance to the pole.
    assert!(
        corr_gam_dpole < -0.9,
        "gam fitted values are not monotone in geodesic distance-to-pole: \
         pearson(f_gam, d_geod_to_pole) = {corr_gam_dpole:.4} (bound -0.9)"
    );

    // (3) MATCH-OR-BEAT the mature comparator on ACCURACY. mgcv bs=\"sos\" is fit
    // on identical data; we compare each tool's error against the SAME analytic
    // truth. gam must be at least as accurate as the mature spline-on-sphere, up
    // to a 10% slack on RMSE. We assert error-vs-truth, NOT that gam's surface
    // resembles mgcv's surface — two tools agreeing on a noisy fit is not a
    // correctness claim. The 0.10 truth-pearson floor on mgcv is a sanity gate
    // that the reference fit itself is not broken (if mgcv produced garbage its
    // RMSE would be a meaningless yardstick); it never gates gam.
    assert!(
        mgcv_corr_true > 0.10,
        "mgcv bs=\"sos\" baseline fit is broken (it does not even correlate with \
         the geodesic-radial truth it is built to recover): \
         pearson(f_mgcv, f_true) = {mgcv_corr_true:.4} — its RMSE \
         ({mgcv_rmse:.5}) cannot serve as a yardstick; check the reference fit"
    );
    assert!(
        gam_rmse <= mgcv_rmse * 1.10,
        "gam is LESS accurate than the mature mgcv bs=\"sos\" spline-on-sphere on \
         identical data: rmse(f_gam, f_true) = {gam_rmse:.5} vs mgcv {mgcv_rmse:.5} \
         (allowed up to {:.5} = mgcv * 1.10). gam must match-or-beat the mature \
         tool's reconstruction accuracy against the analytic geodesic truth.",
        mgcv_rmse * 1.10
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
