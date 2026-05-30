//! Intrinsic S² spherical smooth — OBJECTIVE quality on two axes:
//!   (1) STRUCTURE/AXIOM: rotation equivariance under SO(3) (max |g(p) − g_R(Rp)|).
//!   (2) TRUTH RECOVERY: RMSE of the fitted surface against the KNOWN generating
//!       function on a dense evaluation grid, with gam additionally required to
//!       match-or-beat mgcv `bs="sos"` on that same truth-recovery accuracy.
//!
//! Neither assertion is "gam reproduces a reference tool's fitted output". The
//! data are generated from a known closed-form surface, so accuracy is measured
//! against ground truth; mgcv is demoted to a baseline-to-beat ON ACCURACY, not
//! a target to match.
//!
//! AXIOM (1). Sphere smooths are intrinsically defined on S² via Riemannian
//! geometry. A spherical-harmonic basis spans a function space that is *exactly*
//! invariant under the rotation group SO(3): if you rotate every data point by a
//! fixed 3D rotation R, refit, and then evaluate the refit at the rotated
//! locations, you recover the original fitted surface, because the harmonic
//! subspace is mapped to itself by R, the Laplace-Beltrami penalty is
//! rotation-invariant, and so the penalized REML problem is the *same* problem
//! expressed in a rotated coordinate frame. The two fits are NOT bit-identical:
//! the rotated design matrix carries different floating-point harmonic values,
//! and the rotated REML optimization is an independent run that certifies its
//! smoothing parameter against the same convergence tolerance (default 1e-6)
//! rather than reproducing the original ρ̂ exactly. The residual gap is set by
//! REML convergence and basis-construction round-off, not by f64 round-off
//! alone — but it stays far below the O(1) surface scale a genuinely
//! frame-dependent basis would produce. This is an *intrinsic correctness
//! property* gam must satisfy independent of any comparator.
//!
//! TRUTH RECOVERY (2). The responses are y = sin(lat)·cos(2·lon) + N(0, σ²),
//! σ = 0.01, a degree-2 spherical surface that lives exactly in the harmonic
//! function space (max_degree=4 ⊇ degree 2). With σ this small, a faithful
//! smoother must reconstruct the surface to well within the signal range
//! [−1, 1]; we assert RMSE(gam, truth) below an absolute bar tied to σ and the
//! signal scale. mgcv `bs="sos"` is fit on the IDENTICAL data and evaluated on
//! the IDENTICAL grid as a mature accuracy baseline: gam must recover the truth
//! at least as well as mgcv (gam_rmse ≤ mgcv_rmse · 1.10). We print rel_l2 and
//! Pearson vs mgcv for context only — they are no longer pass/fail criteria,
//! because matching mgcv's (also noisy) fit proves nothing about correctness.

use csv::StringRecord;
use gam::matrix::LinearOperator;
use gam::smooth::build_term_collection_design;
use gam::test_support::reference::{Column, pearson, relative_l2, rmse, run_r};
use gam::{
    FitConfig, FitResult, encode_recordswith_inferred_schema, fit_from_formula, init_parallelism,
};
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
    [lat.cos() * lon.cos(), lat.cos() * lon.sin(), lat.sin()]
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
fn gam_sphere_smooth_is_rotation_equivariant_and_recovers_truth() {
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
    // The harmonic function space is exactly closed under SO(3) and the
    // Laplace-Beltrami penalty is rotation-invariant, so the rotated fit solves
    // the same penalized REML problem in a rotated frame. Absolute (not
    // relative) error is the right metric: the truth surface
    // sin(lat)·cos(2·lon) has range [−1, 1], so a genuinely frame-dependent
    // basis (e.g. a fixed (lat,lon) parameterization dragged across the poles
    // and seam by this 180° rotation) produces O(0.1–1) gaps. The two fits are
    // independent REML runs over bit-different rotated designs, so the residual
    // is bounded by REML convergence (default tol 1e-6) and basis round-off,
    // not f64 alone. 5e-3 is well below the O(1) breakage scale a real frame
    // dependence yields, yet comfortably above the convergence-limited floor
    // (cf. the 0.02 mirror-symmetry refit bound in
    // tests/symmetric_truth_predict_symmetric.rs).
    assert!(
        max_abs < 5e-3,
        "gam's intrinsic S² smooth is NOT rotation-equivariant: \
         max|g(p) - g_R(Rp)| = {max_abs:.3e} (bound 5e-3)"
    );

    // ---- TRUTH RECOVERY: ground-truth surface on the eval grid -------------
    // The data were generated from the closed-form surface
    // f(lat,lon) = sin(lat)·cos(2·lon) (radians). It is a degree-2 spherical
    // function and therefore lies exactly inside the degree-4 harmonic space we
    // fit, so a correct smoother must reconstruct it; the only obstruction is
    // the σ=0.01 observation noise and the 100-point sampling density. We
    // evaluate the truth on the SAME 12×12 grid both engines predict on.
    let truth: Vec<f64> = eval_lats
        .iter()
        .zip(&eval_lons)
        .map(|(lat, lon)| lat.to_radians().sin() * (2.0 * lon.to_radians()).cos())
        .collect();

    // ---- mgcv bs="sos" as a TRUTH-RECOVERY accuracy baseline ---------------
    // mgcv's spline-on-sphere is the closest mature comparator. We fit it on the
    // IDENTICAL unrotated data and predict on the IDENTICAL grid, then compare
    // how well EACH engine recovers the known truth. mgcv is a baseline to
    // match-or-beat on accuracy, NOT a target to reproduce.
    // The reference harness requires every column to share one length, so the
    // 144-point eval grid cannot ride along as extra columns next to the
    // 100-row fit data. Instead we hand mgcv only the fit data and have R
    // reconstruct the IDENTICAL 12×12 grid with the same closed-form linspace
    // the Rust side used (lat = −75 + 150·i/11, lon = −170 + 340·j/11, row-major
    // i over j), so both engines predict on byte-equivalent coordinates.
    let r = run_r(
        &[
            Column::new("lat", &lats),
            Column::new("lon", &lons),
            Column::new("y", &ys),
        ],
        r#"
        suppressPackageStartupMessages(library(mgcv))
        train <- data.frame(lat = df$lat, lon = df$lon, y = df$y)
        # mgcv sos expects (latitude, longitude) in degrees.
        las <- -75.0 + 150.0 * (0:11) / 11.0
        los <- -170.0 + 340.0 * (0:11) / 11.0
        glat <- rep(las, each = 12)   # i (latitude band) is the outer loop
        glon <- rep(los, times = 12)  # j (longitude) is the inner loop
        ev <- data.frame(lat = glat, lon = glon)
        m <- gam(y ~ s(lat, lon, bs = "sos", k = 25), data = train, method = "REML")
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

    // ---- OBJECTIVE accuracy vs ground truth --------------------------------
    let gam_truth_rmse = rmse(&fit_orig_at_p, &truth);
    let mgcv_truth_rmse = rmse(mgcv_pred, &truth);

    // Context only (NOT pass/fail): how closely the two fits track each other.
    // Matching a peer tool's noisy fit is not a quality claim, so these are
    // printed for diagnostics rather than asserted.
    let rel = relative_l2(&fit_orig_at_p, mgcv_pred);
    let corr = pearson(&fit_orig_at_p, mgcv_pred);
    eprintln!(
        "[s2-truth] gam_truth_rmse={gam_truth_rmse:.4} mgcv_truth_rmse={mgcv_truth_rmse:.4} \
         mgcv_edf={mgcv_edf:.3} | context: rel_l2(gam,mgcv)={rel:.4} pearson={corr:.5}"
    );

    // PRIMARY CLAIM: gam recovers the known surface. The truth has range [−1, 1]
    // (peak-to-peak 2) and the observation noise is σ=0.01. A faithful penalized
    // smoother reconstructs a degree-2 surface that lives inside its degree-4
    // harmonic space from 100 points to far better than the signal scale; we
    // require RMSE ≤ 0.10, i.e. ≤ 5% of the signal range and ~10× the noise σ,
    // which leaves headroom for the finite-sample smoothing bias while still
    // failing hard on any genuine reconstruction defect.
    assert!(
        gam_truth_rmse <= 0.10,
        "gam does not recover the known S² surface: RMSE(gam, truth)={gam_truth_rmse:.4} \
         (bound 0.10, signal range 2, noise σ=0.01)"
    );

    // MATCH-OR-BEAT (accuracy, not reproduction): gam must recover the truth at
    // least as well as the mature spline-on-sphere, within a 10% margin for the
    // legitimate basis/penalty difference (gam harmonic L=4 dim 24 vs mgcv sos
    // k=25 thin-plate-on-sphere).
    assert!(
        gam_truth_rmse <= mgcv_truth_rmse * 1.10,
        "gam recovers the S² truth worse than the mgcv bs=\"sos\" baseline: \
         gam_rmse={gam_truth_rmse:.4} > 1.10 · mgcv_rmse={mgcv_truth_rmse:.4}"
    );
}
