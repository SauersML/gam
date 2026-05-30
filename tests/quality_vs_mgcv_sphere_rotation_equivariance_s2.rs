//! Intrinsic S² rotation equivariance — gam's spherical smooth vs `mgcv`'s
//! spline-on-sphere (`bs="sos"`) as the closest mature comparator.
//!
//! Sphere smooths are intrinsically defined on S² via Riemannian geometry. A
//! spherical-harmonic basis spans a function space that is *exactly* invariant
//! under the rotation group SO(3): if you rotate every data point by a fixed 3D
//! rotation R, refit, and then evaluate the refit at the rotated locations, you
//! must recover the original fitted surface bit-for-bit (modulo floating point),
//! because the harmonic subspace is mapped to itself by R and the penalized
//! least-squares solution is therefore identical up to the same rotation. This
//! is an *intrinsic correctness property* gam must satisfy independent of any
//! comparator.
//!
//! mgcv `bs="sos"` uses a fixed (lat, lon) parameterization and is **not**
//! designed to handle rotated coordinate frames: rotating the data in 3D and
//! re-expressing it as (lat', lon') silently moves data across the lat/lon
//! singularities (poles, ±180° seam), so mgcv's surface is not expected to be
//! rotation-equivariant the way an intrinsic harmonic basis is. We therefore
//! (a) assert the intrinsic equivariance property on gam directly with a tight
//! absolute bound, and (b) use mgcv's `sos` fit only as a relative baseline to
//! confirm gam's *unrotated* surface tracks the mature spline-on-sphere on the
//! same data — divergence in either is a real finding. The fragmentation of the
//! sphere-smoothing tool ecosystem (no integrated rotation-equivariant GAM
//! reference exists) is itself part of the finding documented here.

use gam::matrix::LinearOperator;
use gam::smooth::build_term_collection_design;
use gam::test_support::reference::{Column, relative_l2, run_r};
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
/// z=sin(lat). Equivariance depends on using the SAME map gam uses internally.
fn latlon_to_xyz(lat_deg: f64, lon_deg: f64) -> [f64; 3] {
    let lat = lat_deg.to_radians();
    let lon = lon_deg.to_radians();
    [
        lat.cos() * lon.cos(),
        lat.cos() * lon.sin(),
        lat.sin(),
    ]
}

/// Unit vector on S² → geographic (lat, lon) in degrees, inverse of the map
/// above. lon ∈ (−180, 180], lat ∈ [−90, 90].
fn xyz_to_latlon(v: [f64; 3]) -> (f64, f64) {
    let z = v[2].clamp(-1.0, 1.0);
    let lat = z.asin().to_degrees();
    let lon = v[1].atan2(v[0]).to_degrees();
    (lat, lon)
}

/// Apply a 3×3 rotation matrix (row-major) to a unit vector.
fn rotate(r: &[[f64; 3]; 3], v: [f64; 3]) -> [f64; 3] {
    [
        r[0][0] * v[0] + r[0][1] * v[1] + r[0][2] * v[2],
        r[1][0] * v[0] + r[1][1] * v[1] + r[1][2] * v[2],
        r[2][0] * v[0] + r[2][1] * v[1] + r[2][2] * v[2],
    ]
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

/// Fit `y ~ sphere(lat, lon, method=harmonic, max_degree=4)` and evaluate the
/// fitted surface at the given (lat, lon) degrees. The harmonic basis (no
/// data-dependent centers) spans an SO(3)-invariant function space, which is
/// what makes intrinsic rotation equivariance hold to floating point.
fn fit_and_eval(
    data: &gam::data::EncodedDataset,
    eval_lats: &[f64],
    eval_lons: &[f64],
) -> (Vec<f64>, f64) {
    let cfg = FitConfig {
        family: Some("gaussian".to_string()),
        ..FitConfig::default()
    };
    // max_degree=4 -> harmonic basis dim L(L+2)=24 (≈ k=25 in the spec).
    let result = fit_from_formula(
        "y ~ sphere(lat, lon, method=harmonic, max_degree=4)",
        data,
        &cfg,
    )
    .expect("gam sphere harmonic fit");
    let FitResult::Standard(fit) = result else {
        panic!("expected a standard GAM fit for the sphere smooth");
    };
    let edf = fit.fit.edf_total().expect("gam reports total edf");

    let n = eval_lats.len();
    let mut grid = Array2::<f64>::zeros((n, 3));
    for i in 0..n {
        grid[[i, 0]] = eval_lats[i];
        grid[[i, 1]] = eval_lons[i];
        grid[[i, 2]] = 0.0;
    }
    let design = build_term_collection_design(grid.view(), &fit.resolvedspec)
        .expect("rebuild sphere design at eval points");
    (design.design.apply(&fit.fit.beta).to_vec(), edf)
}

#[test]
fn gam_sphere_smooth_is_rotation_equivariant_and_tracks_mgcv_sos() {
    init_parallelism();

    // ---- synthetic data: n=100 uniform points on S² -----------------------
    // Uniform on the sphere: z ~ U(-1,1), lon ~ U(-180,180); lat = asin(z).
    // Ground-truth f(lat,lon) = sin(lat)·cos(2·lon) (radians), σ=0.01 noise.
    let n = 100usize;
    let mut rng = StdRng::seed_from_u64(20260529);
    let u_z = Uniform::new_inclusive(-1.0_f64, 1.0).expect("uniform z");
    let u_lon = Uniform::new(-180.0_f64, 180.0_f64).expect("uniform lon");
    let noise = Normal::new(0.0, 0.01).expect("normal");

    let mut lats = Vec::with_capacity(n);
    let mut lons = Vec::with_capacity(n);
    let mut ys = Vec::with_capacity(n);
    for _ in 0..n {
        let z: f64 = u_z.sample(&mut rng);
        let lon_deg: f64 = u_lon.sample(&mut rng);
        let lat_deg = z.asin().to_degrees();
        let f = lat_deg.to_radians().sin() * (2.0 * lon_deg.to_radians()).cos();
        lats.push(lat_deg);
        lons.push(lon_deg);
        ys.push(f + noise.sample(&mut rng));
    }

    // ---- fixed 180°-about-the-y-axis rotation (a real SO(3) element) -------
    // R_y(180°) = diag(-1, 1, -1): swaps hemispheres about the equatorial
    // y–z structure, dragging points across both the pole and the ±180° seam
    // when re-expressed as (lat', lon'). A pure (lat,lon) parameterization
    // cannot be invariant to this; an intrinsic harmonic basis must be.
    let rmat: [[f64; 3]; 3] = [[-1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, -1.0]];

    // Rotate every data point: same responses y, new (lat', lon') coordinates.
    let mut lats_rot = Vec::with_capacity(n);
    let mut lons_rot = Vec::with_capacity(n);
    for i in 0..n {
        let p = latlon_to_xyz(lats[i], lons[i]);
        let (la, lo) = xyz_to_latlon(rotate(&rmat, p));
        lats_rot.push(la);
        lons_rot.push(lo);
    }

    // ---- evaluation grid: away from the poles (avoid singular lat=±90) -----
    // 12×12 = 144 evaluation points. We evaluate the original fit at each p and
    // the rotated fit at R·p; intrinsic equivariance requires the two to match.
    let mut eval_lats = Vec::new();
    let mut eval_lons = Vec::new();
    for i in 0..12 {
        let lat = -75.0 + 150.0 * (i as f64) / 11.0;
        for j in 0..12 {
            let lon = -170.0 + 340.0 * (j as f64) / 11.0;
            eval_lats.push(lat);
            eval_lons.push(lon);
        }
    }
    let mut eval_lats_rot = Vec::with_capacity(eval_lats.len());
    let mut eval_lons_rot = Vec::with_capacity(eval_lats.len());
    for k in 0..eval_lats.len() {
        let p = latlon_to_xyz(eval_lats[k], eval_lons[k]);
        let (la, lo) = xyz_to_latlon(rotate(&rmat, p));
        eval_lats_rot.push(la);
        eval_lons_rot.push(lo);
    }

    // ---- two gam fits: original frame and rotated frame --------------------
    let orig = make_dataset(&lats, &lons, &ys);
    let rotd = make_dataset(&lats_rot, &lons_rot, &ys);

    // Original fit evaluated at original eval points.
    let (fit_orig_at_p, edf_orig) = fit_and_eval(&orig, &eval_lats, &eval_lons);
    // Rotated fit evaluated at the ROTATED eval points (R·p). Intrinsic
    // equivariance: g_R(R·p) must equal g(p).
    let (fit_rot_at_rp, edf_rot) = fit_and_eval(&rotd, &eval_lats_rot, &eval_lons_rot);

    let max_abs = fit_orig_at_p
        .iter()
        .zip(&fit_rot_at_rp)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f64, f64::max);
    eprintln!(
        "[s2-equivariance] n={n} edf_orig={edf_orig:.3} edf_rot={edf_rot:.3} \
         max|g(p) - g_R(Rp)|={max_abs:.3e}"
    );

    // ---- intrinsic correctness: ABSOLUTE rotation equivariance -------------
    // The harmonic function space is exactly closed under SO(3); the penalized
    // normal equations are invariant under the same rotation, so the fitted
    // surfaces must coincide to floating-point round-off. Absolute (not
    // relative) error is the right metric: equivariance is a structural
    // property, so any non-trivial breakage produces a detectably non-zero
    // gap. 1e-5 leaves headroom for f64 round-off through the rotation,
    // re-encoding, refit, and design rebuild while still catching any genuine
    // basis-level frame dependence.
    assert!(
        max_abs < 1e-5,
        "gam's intrinsic S² smooth is NOT rotation-equivariant: \
         max|g(p) - g_R(Rp)| = {max_abs:.3e} (bound 1e-5)"
    );

    // ---- relative baseline vs mgcv bs="sos" on the UNROTATED data ----------
    // mgcv's spline-on-sphere is the closest mature comparator. On the original
    // (un-rotated) frame both engines smooth the same intrinsic surface, so
    // gam's fit should track mgcv's. (We do NOT ask mgcv to be rotation
    // equivariant — its fixed (lat,lon) parameterization is not designed for
    // rotated frames; that gap is the documented finding.)
    let r = run_r(
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
        # mgcv sos expects (latitude, longitude) in degrees.
        m <- gam(y ~ s(lat, lon, bs = "sos", k = 25), data = train, method = "REML")
        ev <- data.frame(lat = df$elat[!is.na(df$elat)], lon = df$elon[!is.na(df$elon)])
        ev <- ev[!is.na(ev$lat), ]
        emit("pred", as.numeric(predict(m, newdata = ev)))
        emit("edf", sum(m$edf))
        "#,
    );
    let mgcv_pred = r.vector("pred");
    let mgcv_edf = r.scalar("edf");
    assert_eq!(
        mgcv_pred.len(),
        fit_orig_at_p.len(),
        "mgcv sos prediction length must match the {} eval points",
        fit_orig_at_p.len()
    );

    let rel = relative_l2(&fit_orig_at_p, mgcv_pred);
    eprintln!(
        "[s2-equivariance] mgcv_edf={mgcv_edf:.3} rel_l2(gam, mgcv_sos)={rel:.4}"
    );

    // Both engines fit the same smooth surface on the same data via REML; their
    // surfaces should track closely. 0.05 relative L2 is tight enough to catch
    // a genuine baseline divergence yet tolerant of the differing sphere bases
    // (gam harmonic L=4 vs mgcv sos k=25 thin-plate-on-sphere).
    assert!(
        rel < 0.05,
        "gam sphere fit diverges from mgcv bs=\"sos\" baseline: rel_l2={rel:.4} (bound 0.05)"
    );
}
