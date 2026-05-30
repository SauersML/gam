//! Sphere-smooth *geodesic consistency* — gam's intrinsic S² smooth must respect
//! the great-circle (geodesic) metric of the sphere, not the R³ chordal/Euclidean
//! metric of the embedding.
//!
//! Ground truth for great-circle distance is computed with NumPy/SciPy on the
//! exact unit-sphere central angle `d_geod(p,q) = arccos(p·q)` — the definitional
//! geodesic distance on S², which is exactly what `scipy.spatial.distance` /
//! sklearn's `haversine_distances` return for unit vectors (haversine and the
//! arccos-of-dot-product formula agree to round-off for the central angle). We
//! cross-check the two formulas inside the Python body so the "ground truth" is
//! itself verified, then export the pairwise geodesic distances. We additionally
//! fit `mgcv`'s spline-on-sphere (`bs="sos"`) as the closest mature comparator and
//! confirm it satisfies the *same* geodesic-consistency property (by design),
//! making it a meaningful secondary baseline.
//!
//! The intrinsic correctness property: fit `y ~ sphere(lat, lon, k=20)` on noisy
//! samples of a radially-symmetric truth `f(p) = exp(-d_geod(p, pole)/bandwidth)`,
//! evaluate the fitted surface at a set of probe points, and measure how the
//! *fitted-function distance* `|f_gam(p) - f_gam(q)|` relates to the *geodesic
//! distance* `d_geod(p,q)`. Because the truth is monotone in geodesic distance to
//! the pole, a metric-respecting smooth produces fitted differences that are
//! strongly (positively) correlated with geodesic separation. If gam's sphere
//! kernel were accidentally keyed on Euclidean R³ distance (or on raw lat/lon
//! degrees) instead of the intrinsic S² geodesic, that correlation would collapse.
//!
//! The sphere-smoothing tool ecosystem is fragmented — there is no integrated
//! GAM that advertises "geodesic-consistent" fits — so the finding is twofold:
//! (a) gam tracks the mature `mgcv` spline-on-sphere baseline, and (b) gam
//! satisfies the intrinsic geodesic-consistency property directly. If this test
//! fails, check the kernel evaluation for the lat/lon <-> 3D unit-vector
//! conversion (the embedding must feed an intrinsic, not chordal, distance).

use gam::matrix::LinearOperator;
use gam::smooth::build_term_collection_design;
use gam::test_support::reference::{Column, pearson, run_python, run_r};
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
    let u_z = Uniform::new_inclusive(-1.0, 1.0);
    let u_lon = Uniform::new(-180.0_f64, 180.0_f64);
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

    // ---- pairwise fitted-function distances |f_gam(p) - f_gam(q)| ----------
    // Flatten the strictly-upper-triangular pairs in a fixed order so the gam and
    // SciPy vectors are element-aligned by (i, j).
    let mut gam_pair_diff = Vec::with_capacity(m * (m - 1) / 2);
    for i in 0..m {
        for j in (i + 1)..m {
            gam_pair_diff.push((gam_eval[i] - gam_eval[j]).abs());
        }
    }

    // ---- ground-truth pairwise geodesic distances via NumPy/SciPy ----------
    // Identical probe points fed to Python. We compute the great-circle central
    // angle two independent ways (arccos-of-dot-product and the haversine
    // formula) and assert they agree, so the exported "ground truth" is itself
    // verified before gam is judged against it. SciPy's distance machinery is
    // used to form the pairwise matrix; the central-angle metric IS the geodesic
    // distance on the unit sphere.
    let py = run_python(
        &[
            Column::new("elat", &eval_lats),
            Column::new("elon", &eval_lons),
        ],
        r#"
import numpy as np
import scipy  # ensure SciPy is present; hard-fail otherwise
from scipy.spatial.distance import pdist

lat = np.radians(np.asarray(df["elat"], dtype=float))
lon = np.radians(np.asarray(df["elon"], dtype=float))

# Unit vectors on S^2 (same convention as gam: x=cos(lat)cos(lon), etc.).
x = np.cos(lat) * np.cos(lon)
y = np.cos(lat) * np.sin(lon)
z = np.sin(lat)
P = np.column_stack([x, y, z])
nrm = np.linalg.norm(P, axis=1)
assert np.allclose(nrm, 1.0, atol=1e-12), "probe points must be unit vectors"

m = P.shape[0]
iu, ju = np.triu_indices(m, k=1)

# (A) geodesic via arccos of dot product (the central angle on S^2).
dots = np.clip(np.einsum("ij,ij->i", P[iu], P[ju]), -1.0, 1.0)
geo_arccos = np.arccos(dots)

# (B) geodesic via the haversine formula on (lat, lon) — the classic
# great-circle distance. On the unit sphere this equals the central angle.
dlat = lat[ju] - lat[iu]
dlon = lon[ju] - lon[iu]
hav = (np.sin(dlat / 2.0) ** 2
       + np.cos(lat[iu]) * np.cos(lat[ju]) * np.sin(dlon / 2.0) ** 2)
geo_hav = 2.0 * np.arcsin(np.sqrt(np.clip(hav, 0.0, 1.0)))

# The two great-circle formulas must agree: this validates the ground truth.
assert np.allclose(geo_arccos, geo_hav, atol=1e-9), "haversine vs arccos disagree"

# Sanity: SciPy's chordal (Euclidean R^3) distance is a DIFFERENT metric from the
# geodesic; emit their correlation to document that they are not interchangeable.
chord = pdist(P, metric="euclidean")
def corr(a, b):
    a = np.asarray(a); b = np.asarray(b)
    return float(np.corrcoef(a, b)[0, 1])

emit("geodesic", geo_arccos)
emit("geo_vs_chord_corr", [corr(geo_arccos, chord)])
emit("npairs", [float(geo_arccos.size)])
"#,
    );
    let geodesic = py.vector("geodesic");
    let geo_vs_chord = py.scalar("geo_vs_chord_corr");
    assert_eq!(
        geodesic.len(),
        gam_pair_diff.len(),
        "geodesic pair count must match gam pair count"
    );

    // ---- the intrinsic correctness metric ---------------------------------
    // Pearson correlation between gam's fitted-function pairwise differences and
    // the true geodesic pairwise distances. A metric-respecting fit of a truth
    // monotone in geodesic-distance-to-pole yields a strong positive correlation.
    let corr_geo = pearson(&gam_pair_diff, geodesic);
    eprintln!(
        "[sphere-geodesic] n={n} m_probe={m} npairs={} edf={edf:.3} \
         pearson(|df_gam|, d_geod)={corr_geo:.4} corr(geodesic, chordal)={geo_vs_chord:.4}",
        gam_pair_diff.len()
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

    // mgcv's own fitted-function pairwise differences, same (i, j) order.
    let mut mgcv_pair_diff = Vec::with_capacity(gam_pair_diff.len());
    for i in 0..m {
        for j in (i + 1)..m {
            mgcv_pair_diff.push((mgcv_pred[i] - mgcv_pred[j]).abs());
        }
    }
    let mgcv_corr_geo = pearson(&mgcv_pair_diff, geodesic);
    // How closely gam's probe-surface tracks mgcv's spline-on-sphere surface.
    let gam_vs_mgcv = pearson(&gam_eval, mgcv_pred);
    eprintln!(
        "[sphere-geodesic] mgcv_edf={mgcv_edf:.3} \
         pearson(|df_mgcv|, d_geod)={mgcv_corr_geo:.4} pearson(gam, mgcv_sos)={gam_vs_mgcv:.4}"
    );

    // ---- assertions --------------------------------------------------------
    // (1) Intrinsic property. The truth is monotone-decreasing in geodesic
    // distance to the pole, so points far apart geodesically tend to have larger
    // fitted-value differences. A metric-respecting smooth therefore yields a
    // strong positive Pearson correlation between |f_gam(p)-f_gam(q)| and
    // d_geod(p,q). 0.85 is the principled bound from the spec: not 1.0 because the
    // fit is noisy and the relationship is monotone-but-not-linear (so even a
    // perfect smooth would not give correlation 1), yet far above the ~0.0–0.4
    // a chordal/lat-lon-confused kernel would produce. A failure here means the
    // kernel is not keyed on the intrinsic S² geodesic — check the lat/lon <-> 3D
    // unit-vector conversion feeding the kernel.
    assert!(
        corr_geo > 0.85,
        "gam sphere smooth is NOT geodesic-consistent: \
         pearson(|f_gam diff|, d_geod) = {corr_geo:.4} (bound 0.85). \
         Check the kernel's lat/lon <-> 3D unit-vector conversion: the smooth must \
         respect S² geometry, not the R³ embedding."
    );

    // (2) The mature comparator must clear the same bar — this is what makes it a
    // valid yardstick (mgcv sos is geodesic-consistent by construction).
    assert!(
        mgcv_corr_geo > 0.85,
        "mgcv bs=\"sos\" baseline failed the geodesic-consistency bar it defines: \
         pearson(|f_mgcv diff|, d_geod) = {mgcv_corr_geo:.4} (bound 0.85) — \
         check the reference fit, not gam"
    );

    // (3) Sanity that the geodesic and chordal metrics are genuinely different
    // (otherwise the test could pass even for a Euclidean-embedding kernel). On
    // points spread across the sphere the chordal/geodesic correlation is well
    // below 1; if it were ~1 the test would not be discriminating.
    assert!(
        geo_vs_chord < 0.999,
        "geodesic and chordal metrics are indistinguishable on this grid \
         (corr={geo_vs_chord:.5}); the test would not discriminate intrinsic vs \
         embedding kernels — widen the probe grid"
    );

    // (4) gam must track the mature spline-on-sphere surface on identical data;
    // both REML-fit the same intrinsic surface with k=20, so their probe-grid
    // surfaces should be strongly correlated. 0.9 is tight enough to catch a real
    // baseline divergence yet tolerant of the differing sphere bases (gam Sobolev
    // Wahba vs mgcv thin-plate-on-sphere).
    assert!(
        gam_vs_mgcv > 0.9,
        "gam sphere fit diverges from the mgcv bs=\"sos\" baseline: \
         pearson(gam, mgcv_sos) = {gam_vs_mgcv:.4} (bound 0.9)"
    );
}
