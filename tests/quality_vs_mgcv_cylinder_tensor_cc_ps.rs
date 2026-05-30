//! End-to-end quality: gam's *mixed-boundary* tensor smooth on the cylinder
//! S¹ × [0,1] must match **mgcv** — the mature, de-facto standard GAM
//! implementation — on the same data, and must satisfy the two intrinsic
//! boundary properties a cylinder basis exists to guarantee.
//!
//! A cylinder is the product of a circle and an interval: **periodic** in the
//! azimuthal angle θ (wrapping at 0 ≡ 2π) and **non-periodic / clamped** in the
//! height z (free at the two ends, no wrap). mgcv builds exactly this with
//! `te(theta, z, bs = c("cc", "ps"))` — the row-wise Kronecker product of a
//! cyclic-cubic ("cc") margin in θ and an ordinary penalized-B-spline ("ps")
//! margin in z. gam exposes the same construction through
//! `te(theta, z, boundary=['periodic','clamped'], period=[2*pi, None])`,
//! which builds one periodic B-spline margin (θ) and one clamped, non-periodic
//! B-spline margin (z) and forms their tensor product.
//!
//! Both engines fit by REML against a Gaussian likelihood, so they target the
//! *same* penalized objective; on a low-noise truth the fitted surfaces must
//! essentially coincide. We additionally assert the TWO defining contracts of a
//! cylinder smooth:
//!   1. **Azimuthal seam continuity (θ):** the fitted surface must be identical
//!      at θ=0 and θ=2π for every z — the load-bearing property of the cyclic
//!      margin, exact up to float error.
//!   2. **Non-periodic z boundary (asymmetry):** the z margin must NOT wrap. If
//!      gam mistakenly applied periodicity to z, the surface would close at
//!      z=0 ≡ z=1; we assert that gam's own z-seam gap is *large* (the surface
//!      is free to differ at the two ends) AND that gam's z-direction behavior
//!      tracks mgcv's "ps" margin (which is likewise non-wrapping). The
//!      intrinsic asymmetry — θ closes, z does not — is what distinguishes a
//!      true cylinder from a torus and from a flat-bed double-clamped sheet.
//!
//! Data: deterministic 15×20 grid (n=300), θ uniform on [0,2π) (last grid point
//! stops short of 2π so the seam is not duplicated in training), z uniform on
//! [0,1], truth f(θ,z)=sin(2θ)·(1+z), Gaussian noise σ=0.03 from a fixed seed.
//! The identical (θ,z,y) rows are handed to both gam and mgcv.

use csv::StringRecord;
use gam::matrix::LinearOperator;
use gam::smooth::build_term_collection_design;
use gam::test_support::reference::{Column, pearson, relative_l2, run_r};
use gam::{
    FitConfig, FitResult, encode_recordswith_inferred_schema, fit_from_formula, init_parallelism,
};
use ndarray::Array2;
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand_distr::{Distribution, Normal};
use std::f64::consts::TAU;

#[test]
fn gam_cylinder_tensor_cc_ps_matches_mgcv_and_mixes_boundaries() {
    init_parallelism();

    // ---- deterministic cylinder truth on a 15×20 grid ---------------------
    // f(θ,z) = sin(2θ)·(1+z) over [0,2π) × [0,1]. The θ grid stops short of 2π
    // so the seam is never duplicated in training; z spans the closed [0,1].
    // Gaussian noise σ=0.03 from a fixed seed makes the rows reproducible and
    // identical across both engines. The truth is genuinely periodic in θ (so
    // the cyclic margin is exercised) and genuinely *non*-periodic in z:
    // f(θ,0)=sin(2θ) ≠ f(θ,1)=2·sin(2θ), so a correct z margin must NOT wrap.
    const G_THETA: usize = 15;
    const G_Z: usize = 20;
    let n = G_THETA * G_Z;
    let sigma = 0.03_f64;
    let mut rng = StdRng::seed_from_u64(20240529);
    let noise = Normal::new(0.0, sigma).expect("normal");

    let mut theta: Vec<f64> = Vec::with_capacity(n);
    let mut z: Vec<f64> = Vec::with_capacity(n);
    let mut y: Vec<f64> = Vec::with_capacity(n);
    for i in 0..G_THETA {
        let th = TAU * (i as f64) / (G_THETA as f64);
        for j in 0..G_Z {
            let zz = (j as f64) / ((G_Z - 1) as f64);
            let f = (2.0 * th).sin() * (1.0 + zz);
            theta.push(th);
            z.push(zz);
            y.push(f + noise.sample(&mut rng));
        }
    }

    // ---- fit with gam: mixed-boundary tensor smooth, REML -----------------
    // `boundary=['periodic','clamped']` + `period=[2*pi, None]` is gam's exact
    // analog of mgcv's te(bs=c('cc','ps')): a cyclic θ margin tensor-producted
    // with a clamped, non-periodic z margin on the cylinder.
    let headers = ["theta", "z", "y"].into_iter().map(String::from).collect();
    let rows: Vec<StringRecord> = (0..n)
        .map(|r| {
            StringRecord::from(vec![
                theta[r].to_string(),
                z[r].to_string(),
                y[r].to_string(),
            ])
        })
        .collect();
    let ds = encode_recordswith_inferred_schema(headers, rows).expect("encode cylinder dataset");
    let col = ds.column_map();
    let theta_idx = col["theta"];
    let z_idx = col["z"];

    let cfg = FitConfig {
        family: Some("gaussian".to_string()),
        ..FitConfig::default()
    };
    let formula = "y ~ te(theta, z, boundary=['periodic','clamped'], period=[2*pi, None], k=8)";
    let result = fit_from_formula(formula, &ds, &cfg).expect("gam cylinder tensor fit");
    let FitResult::Standard(fit) = result else {
        panic!("expected a standard GAM fit for the cylinder tensor smooth");
    };
    let gam_edf = fit.fit.edf_total().expect("gam reports total edf");

    // Helper: evaluate gam's fitted surface at arbitrary (θ,z) rows by rebuilding
    // the design from the frozen spec (identity link => design·beta = mean).
    let gam_predict = |ths: &[f64], zs: &[f64]| -> Vec<f64> {
        assert_eq!(ths.len(), zs.len());
        let m = ths.len();
        let mut pts = Array2::<f64>::zeros((m, ds.headers.len()));
        for r in 0..m {
            pts[[r, theta_idx]] = ths[r];
            pts[[r, z_idx]] = zs[r];
        }
        let d = build_term_collection_design(pts.view(), &fit.resolvedspec)
            .expect("rebuild cylinder design");
        d.design.apply(&fit.fit.beta).to_vec()
    };

    // gam fitted surface at the training grid.
    let gam_fitted = gam_predict(&theta, &z);

    // ---- fit the SAME model with mgcv te(bs=c("cc","ps")) (the reference) --
    // mgcv needs an explicit cyclic knot range [0, 2π] for the θ margin so its
    // cyclic closure matches the [0, 2π) data support. The z margin ("ps") gets
    // mgcv's default penalized-B-spline knot placement over observed z, with no
    // wrap — exactly the non-periodic boundary gam declares as 'clamped'. We
    // emit mgcv's fitted surface at the training grid AND on a dense
    // z-direction probe so the Rust side can compare the non-wrapping shape.
    let r = run_r(
        &[
            Column::new("theta", &theta),
            Column::new("z", &z),
            Column::new("y", &y),
        ],
        r#"
        suppressPackageStartupMessages(library(mgcv))
        m <- gam(y ~ te(theta, z, bs = c("cc", "ps"), k = c(8, 8)),
                 data = df, method = "REML",
                 knots = list(theta = c(0, 2 * pi)))
        emit("fitted", as.numeric(fitted(m)))
        emit("edf", sum(m$edf))
        # z-direction probe at a fixed interior angle: predict along z in [0,1]
        # at theta = pi/4 to characterize the (non-periodic) z marginal shape.
        zg <- seq(0, 1, length.out = 41)
        pg <- data.frame(theta = rep(pi / 4, length(zg)), z = zg)
        emit("zprobe", as.numeric(predict(m, newdata = pg)))
        # mgcv's own z endpoints at theta = pi/4: f(pi/4, 0) and f(pi/4, 1).
        emit("zends", as.numeric(predict(m,
              newdata = data.frame(theta = c(pi / 4, pi / 4), z = c(0, 1)))))
        "#,
    );
    let mgcv_fitted = r.vector("fitted");
    let mgcv_edf = r.scalar("edf");
    let mgcv_zprobe = r.vector("zprobe");
    let mgcv_zends = r.vector("zends");
    assert_eq!(mgcv_fitted.len(), n, "mgcv fitted length mismatch");
    assert_eq!(mgcv_zprobe.len(), 41, "mgcv z-probe length mismatch");

    // ---- pointwise agreement on the training grid --------------------------
    let rel = relative_l2(&gam_fitted, mgcv_fitted);
    let corr = pearson(&gam_fitted, mgcv_fitted);
    let edf_rel = (gam_edf - mgcv_edf).abs() / mgcv_edf.abs().max(1.0);

    // ---- (1) intrinsic θ-seam continuity (the cyclic-margin contract) ------
    // Evaluate at a dense set of z values, comparing θ=0 vs θ=2π. A genuine
    // cyclic θ margin has identical design rows — hence identical fitted values
    // — at coordinates separated by exactly one period in θ, for every z.
    let z_grid: Vec<f64> = (0..41).map(|k| (k as f64) / 40.0).collect();
    let theta_zeros: Vec<f64> = std::iter::repeat_n(0.0, z_grid.len()).collect();
    let theta_taus: Vec<f64> = std::iter::repeat_n(TAU, z_grid.len()).collect();
    let theta_seam_0 = gam_predict(&theta_zeros, &z_grid);
    let theta_seam_tau = gam_predict(&theta_taus, &z_grid);
    let theta_seam_gap = theta_seam_0
        .iter()
        .zip(theta_seam_tau.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f64, f64::max);

    // ---- (2) intrinsic z non-periodicity (the asymmetry that defines a -----
    //          cylinder vs a torus) --------------------------------------------
    // gam's own z-direction probe at θ=π/4, plus its z endpoints f(π/4,0) and
    // f(π/4,1). The ground truth is sin(π/2)·(1+z) = 1+z, so f(π/4,1)-f(π/4,0)
    // ≈ 1.0: the surface MUST be free to differ across the z ends. If gam wrongly
    // wrapped z, this gap would collapse toward 0. We require the gap to be a
    // sizeable fraction of the truth's span and to match mgcv's "ps" margin.
    let theta_quarter: Vec<f64> =
        std::iter::repeat_n(std::f64::consts::FRAC_PI_4, z_grid.len()).collect();
    let gam_zprobe = gam_predict(&theta_quarter, &z_grid);
    let gam_z0 = gam_zprobe[0];
    let gam_z1 = gam_zprobe[gam_zprobe.len() - 1];
    let gam_z_endpoint_gap = (gam_z1 - gam_z0).abs();
    let mgcv_z_endpoint_gap = (mgcv_zends[1] - mgcv_zends[0]).abs();

    // How well gam's z-marginal shape tracks mgcv's "ps" z-marginal shape.
    let zprobe_rel = relative_l2(&gam_zprobe, mgcv_zprobe);
    let zprobe_corr = pearson(&gam_zprobe, mgcv_zprobe);

    eprintln!(
        "cylinder te(cc,ps): n={n} gam_edf={gam_edf:.3} mgcv_edf={mgcv_edf:.3} \
         rel_l2={rel:.5} pearson={corr:.6} edf_rel={edf_rel:.3} \
         theta_seam_gap={theta_seam_gap:.3e} \
         gam_z_gap={gam_z_endpoint_gap:.4} mgcv_z_gap={mgcv_z_endpoint_gap:.4} \
         zprobe_rel={zprobe_rel:.4} zprobe_corr={zprobe_corr:.5}"
    );

    // Both engines REML-fit identical low-noise cylinder data in matched mixed
    // cyclic×ps tensor spaces (k=8 per margin), so the fitted surfaces must
    // essentially coincide. The spec bounds (rel_l2 < 0.04, pearson > 0.999) are
    // tight for σ=0.03 yet absorb the small basis/centering convention gap; a
    // real divergence is a real bug in gam's mixed-boundary Kronecker build.
    assert!(
        corr > 0.999,
        "cylinder fitted surfaces should be near-identical to mgcv: pearson={corr:.6}"
    );
    assert!(
        rel < 0.04,
        "cylinder fitted surface diverges from mgcv te(cc,ps): rel_l2={rel:.5}"
    );
    // EDF is basis/null-space-convention sensitive; same-ballpark complexity
    // (within 20% relative) is the right expectation for matched k and REML.
    assert!(
        edf_rel < 0.20,
        "effective degrees of freedom disagree: gam={gam_edf:.3} mgcv={mgcv_edf:.3} (rel={edf_rel:.3})"
    );

    // (1) The defining contract of the cyclic θ margin: value continuity across
    // the azimuthal wrap, exact up to float error. The θ-seam must close to
    // < 1e-6; any larger gap is a sign/threshold bug in gam's periodic closure.
    assert!(
        theta_seam_gap < 1e-6,
        "θ-seam not closed: max |f(0,z) - f(2π,z)| = {theta_seam_gap:.3e}"
    );

    // (2) The z margin must NOT wrap. The truth's z-endpoint span at θ=π/4 is
    // ≈ 1.0; a correctly non-periodic z marginal keeps the two ends free, so the
    // recovered gap must be a large fraction of that span (we require > 0.5 — at
    // least half the truth, far from the ~0 a wrongly-wrapped z would force) and
    // must agree with mgcv's "ps" margin gap to within 25% relative. This is the
    // intrinsic asymmetry distinguishing a cylinder from a torus.
    assert!(
        gam_z_endpoint_gap > 0.5,
        "z margin appears to wrap (cylinder collapsed to torus): \
         |f(π/4,1) - f(π/4,0)| = {gam_z_endpoint_gap:.4} (truth span ≈ 1.0)"
    );
    let z_gap_rel =
        (gam_z_endpoint_gap - mgcv_z_endpoint_gap).abs() / mgcv_z_endpoint_gap.abs().max(1e-6);
    assert!(
        z_gap_rel < 0.25,
        "gam's non-periodic z-endpoint span disagrees with mgcv 'ps': \
         gam={gam_z_endpoint_gap:.4} mgcv={mgcv_z_endpoint_gap:.4} (rel={z_gap_rel:.3})"
    );
    // The whole z-marginal shape at θ=π/4 must track mgcv's "ps" marginal, not
    // merely the endpoints: a non-wrapping curve that nonetheless mis-shapes the
    // interior would be caught here. Tight for matched k and REML on σ=0.03.
    assert!(
        zprobe_corr > 0.999,
        "z-marginal shape diverges from mgcv 'ps': pearson={zprobe_corr:.5}"
    );
    assert!(
        zprobe_rel < 0.05,
        "z-marginal shape diverges from mgcv 'ps': rel_l2={zprobe_rel:.4}"
    );
}
