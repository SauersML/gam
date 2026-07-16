//! End-to-end quality: gam's 2-D spatial Duchon spline `duchon(x, z, k=...)`
//! must RECOVER the known smooth surface it was trained on, and do so at least
//! as accurately as mgcv — the mature, standard GAM implementation — fit on the
//! identical data.
//!
//! OBJECTIVE METRIC (the pass/fail claim): TRUTH RECOVERY. The data are
//! generated from a KNOWN surface `f(x, z)` (a sum of two Gaussian bumps over
//! the unit square) plus i.i.d. Gaussian noise with σ=0.10, n=400. The primary
//! assertion is that gam's fitted smooth recovers `f` on a dense interior grid
//! with `RMSE(gam_fit, truth)` below a principled bar tied to the noise level —
//! i.e. the smoother denoises rather than tracking the noise. This is an
//! objective accuracy claim about gam alone; it does NOT depend on any
//! reference tool.
//!
//! BASELINE TO MATCH-OR-BEAT: mgcv is fit on byte-identical data and its own
//! truth-recovery RMSE is computed; we additionally assert gam's
//! truth-recovery error is no worse than mgcv's within a 10% accuracy margin.
//! This demotes mgcv from "the answer gam must reproduce" to "a respected
//! baseline gam must match or beat on ACCURACY VS THE TRUTH". The gam-vs-mgcv
//! relative-L2 is printed for context only and is NOT a pass criterion.
//!
//! THE OBJECT UNDER TEST. gam's redesigned non-periodic Euclidean Duchon is a
//! structural amplitude/slope/curvature smoother on the cubic (`r³`)
//! polyharmonic basis with an affine (`Linear`, d+1 = 3 in 2D) polynomial null
//! space and the default spectral power `s = (d−1)/2 = 0.5` in 2D. We test the
//! DEFAULT `duchon(x, z, k=...)` — the magic structural surface smoother — the
//! object users actually get for a 2-D spatial smooth.
//!
//! BASELINE COMPARATOR: `mgcv::gam(y ~ s(x, z, bs="ds", k=49, m=c(2,0)),
//! method="REML")`. For `bs="ds"` the pair `m=c(m1,m2)` sets `m1` = order of the
//! squared-derivative penalty `‖D^{m1} f‖²` and `m2` = the extra radial-kernel
//! power; the radial kernel exponent is `2*m1 + 2*m2 − d`. We pick `m=c(2,0)`
//! because (a) `m1=2` gives an AFFINE polynomial null space of dimension d+1=3,
//! matching gam's affine (`Linear`) null space, and (b) `m2=0` keeps the basis
//! conditionally-positive-definite (`2*m2 < d`), the standard 2-D thin-plate-
//! equivalent Duchon smoother and the canonical mature analogue of gam's default
//! 2-D structural smoother. NOTE: an EXACT `r³` kernel match would need
//! `2*m1 + 2*m2 − 2 = 3` ⇒ `m1+m2 = 2.5`, which has no integer mgcv setting, so
//! `m=c(2,0)` (kernel `r²·log r`, the conventional 2-D Duchon) is the closest
//! standard comparator — appropriate precisely because this is a match-or-beat
//! ACCURACY baseline, not a fitted-surface reproduction.

use gam::matrix::LinearOperator;
use gam::smooth::build_term_collection_design;
use gam::test_support::reference::{Column, QualityPair, relative_l2, rmse, run_r};
use gam::{
    FitConfig, FitResult, encode_recordswith_inferred_schema, fit_from_formula, init_parallelism,
};
use ndarray::Array2;
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand_distr::{Distribution, Normal, Uniform};

/// Known smooth 2-D surface: a sum of two Gaussian bumps over the unit square.
/// Smooth, low-curvature, and clearly above the noise floor, so a competent
/// spatial smoother must recover it. Range is roughly [0, ~1.4].
fn truth_surface(x: f64, z: f64) -> f64 {
    let bump = |cx: f64, cz: f64, s: f64, a: f64| {
        let d2 = (x - cx).powi(2) + (z - cz).powi(2);
        a * (-d2 / (2.0 * s * s)).exp()
    };
    bump(0.3, 0.3, 0.18, 1.0) + bump(0.7, 0.65, 0.22, 0.8)
}

#[test]
fn gam_duchon_2d_surface_matches_mgcv_ds() {
    init_parallelism();

    // ---- synthetic data: (x, z) ~ U[0,1]^2, y = f(x,z) + N(0, 0.10), n=400 --
    // Fixed seed so gam and mgcv see byte-identical data.
    let n = 400usize;
    let mut rng = StdRng::seed_from_u64(20260530);
    let u = Uniform::new(0.0_f64, 1.0).expect("uniform [0,1]");
    let noise = Normal::new(0.0, 0.10).expect("normal");

    let mut x = Vec::with_capacity(n);
    let mut z = Vec::with_capacity(n);
    let mut y = Vec::with_capacity(n);
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
    let ds =
        encode_recordswith_inferred_schema(headers, rows).expect("encode synthetic 2d dataset");
    let col = ds.column_map();
    let x_idx = col["x"];
    let z_idx = col["z"];

    // ---- fit with gam: y ~ duchon(x, z, k=49), REML -----------------------
    // The default (no order=/power=) is the structural cubic surface smoother:
    // affine null space + power s=(d-1)/2 in 2D => the 2-D analogue of mgcv's
    // bs="ds", m=c(2,0). REML is gam's default. k=49 (~7x7) gives the surface
    // enough centers to resolve two bumps without over-fitting.
    let cfg = FitConfig {
        family: Some("gaussian".to_string()),
        ..FitConfig::default()
    };
    let result = fit_from_formula("y ~ duchon(x, z, k=49)", &ds, &cfg).expect("gam 2d duchon fit");
    let FitResult::Standard(fit) = result else {
        panic!("expected a standard GAM fit for a gaussian 2-D Duchon smooth");
    };

    // ---- dense interior test grid on [0.05, 0.95]^2 (avoid extrapolation) ---
    let g = 25usize; // 25x25 = 625 interior grid points
    let coord = |i: usize| 0.05 + 0.90 * i as f64 / (g as f64 - 1.0);
    let mut gx = Vec::with_capacity(g * g);
    let mut gz = Vec::with_capacity(g * g);
    let mut y_truth = Vec::with_capacity(g * g);
    for i in 0..g {
        for j in 0..g {
            let (xi, zi) = (coord(i), coord(j));
            gx.push(xi);
            gz.push(zi);
            y_truth.push(truth_surface(xi, zi));
        }
    }
    let m = gx.len();

    // gam fitted values at the test grid: rebuild the design from the frozen
    // spec (identity link => design*beta = mean).
    let mut grid = Array2::<f64>::zeros((m, ds.headers.len()));
    for i in 0..m {
        grid[[i, x_idx]] = gx[i];
        grid[[i, z_idx]] = gz[i];
    }
    let design = build_term_collection_design(grid.view(), &fit.resolvedspec)
        .expect("rebuild 2-D Duchon design at test grid");
    let gam_fitted: Vec<f64> = design.design.apply(&fit.fit.beta).to_vec();

    // ---- fit the SAME model with mgcv bs="ds" (the mature reference) -------
    // Pass training data plus the test grid (trailing rows, sentinel y) so mgcv
    // predicts on exactly the same grid. m=c(2,0): 2nd-derivative penalty,
    // affine null space, CPD-valid — the standard 2-D Duchon analogue.
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

    // ---- OBJECTIVE METRIC: truth recovery on the interior grid ------------
    let gam_truth_rmse = rmse(&gam_fitted, &y_truth);
    let mgcv_truth_rmse = rmse(mgcv_fitted, &y_truth);
    let rel_gam_vs_mgcv = relative_l2(&gam_fitted, mgcv_fitted);

    // RMS of the truth: the error a constant (zero) predictor would incur, the
    // "do-nothing" baseline that the recovery must clearly beat.
    let rms_truth = (y_truth.iter().map(|t| t * t).sum::<f64>() / y_truth.len() as f64).sqrt();

    eprintln!(
        "duchon-truth-recovery-2d: n={n} grid={m} sigma=0.10 \
         gam_truth_rmse={gam_truth_rmse:.4} mgcv_truth_rmse={mgcv_truth_rmse:.4} \
         rms_truth={rms_truth:.4} (context: rel_l2(gam,mgcv)={rel_gam_vs_mgcv:.4})"
    );
    eprintln!(
        "{}",
        QualityPair::error(
            "smooths",
            "quality_vs_mgcv_duchon_2d",
            "truth_rmse",
            gam_truth_rmse,
            "mgcv",
            mgcv_truth_rmse,
        )
        .line()
    );

    // (1) ABSOLUTE non-degeneracy bar: gam must genuinely recover the surface,
    // not blow up or collapse to a trivial predictor. The two-bump surface is
    // smooth and well above the σ=0.10 noise; a competent spatial smoother
    // reconstructs it to a small fraction of the signal RMS (~0.5 here). A
    // constant (zero/mean) predictor scores RMSE ≈ rms_truth, so 0.15 is a
    // principled "clearly better-than-trivial AND denoising" floor that still
    // catches a blown-up or non-recovering fit.
    assert!(
        gam_truth_rmse < 0.15,
        "gam 2-D Duchon smooth failed to recover the surface: \
         RMSE-vs-truth={gam_truth_rmse:.4} (signal RMS≈{rms_truth:.4}, bar 0.15)"
    );

    // (2) MATCH-OR-BEAT mgcv ON ACCURACY. mgcv is the mature baseline; gam must
    // recover the truth at least as well, within a 10% accuracy margin. This is
    // a comparison of ERRORS AGAINST THE TRUTH, not closeness of the two fits.
    assert!(
        gam_truth_rmse <= mgcv_truth_rmse * 1.10,
        "gam recovers the 2-D truth worse than mgcv: gam RMSE-vs-truth={gam_truth_rmse:.4} \
         > 1.10 * mgcv RMSE-vs-truth={mgcv_truth_rmse:.4}"
    );
}
