//! End-to-end OBJECTIVE quality: gam's *mixed-boundary* tensor smooth on the
//! cylinder S¹ × [0,1] must RECOVER the known truth and satisfy the two
//! intrinsic boundary contracts a cylinder basis exists to guarantee.
//!
//! A cylinder is the product of a circle and an interval: **periodic** in the
//! azimuthal angle θ (wrapping at 0 ≡ 2π) and **non-periodic / clamped** in the
//! height z (free at the two ends, no wrap). gam builds this with
//! `te(theta, z, boundary=['periodic','clamped'], period=[2*pi, None])`: one
//! periodic B-spline margin (θ) tensor-producted with a clamped, non-periodic
//! B-spline margin (z). mgcv builds the identical construction with
//! `te(theta, z, bs = c("cc", "ps"))` and serves here purely as a
//! BASELINE-TO-MATCH-OR-BEAT on recovery accuracy — never as the definition of
//! "correct". The data are generated from a KNOWN truth, so the pass/fail
//! criterion is recovery of that truth, not agreement with any peer tool.
//!
//! OBJECTIVE METRICS ASSERTED (none is "close to a reference tool's output"):
//!   1. TRUTH RECOVERY (primary): RMSE(gam_fit, truth) on a dense held-out
//!      (θ,z) probe grid is small relative to the noise scale —
//!      RMSE ≤ 1.5·σ — and gam matches-or-beats mgcv's recovery RMSE on the
//!      SAME truth to within 10% (mgcv stays only as an accuracy baseline).
//!   2. AZIMUTHAL SEAM CONTINUITY (θ contract, exact math): the fitted surface
//!      is identical at θ=0 and θ=2π for every z, to float error — the
//!      load-bearing property of the cyclic margin.
//!   3. Z NON-PERIODICITY (the asymmetry that defines a cylinder vs a torus):
//!      the recovered z-endpoint span at θ=π/4 tracks the TRUTH span (≈1.0), so
//!      the z margin demonstrably does NOT wrap. Asserted against the truth, not
//!      against mgcv.
//!
//! Data: deterministic 15×20 grid (n=300), θ uniform on [0,2π) (last grid point
//! stops short of 2π so the seam is not duplicated in training), z uniform on
//! [0,1], truth f(θ,z)=sin(2θ)·(1+z), Gaussian noise σ=0.03 from a fixed seed.
//! The identical (θ,z,y) rows are handed to both gam and mgcv.

use csv::StringRecord;
use gam::matrix::LinearOperator;
use gam::smooth::build_term_collection_design;
use gam::test_support::reference::{Column, relative_l2, rmse, run_r};
use gam::{
    FitConfig, FitResult, encode_recordswith_inferred_schema, fit_from_formula, init_parallelism,
};
use ndarray::Array2;
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand_distr::{Distribution, Normal};
use std::f64::consts::TAU;

/// Closed-form cylinder truth f(θ,z) = sin(2θ)·(1+z).
fn truth(theta: f64, z: f64) -> f64 {
    (2.0 * theta).sin() * (1.0 + z)
}

#[test]
fn gam_cylinder_tensor_cc_ps_recovers_truth_and_mixes_boundaries() {
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
            theta.push(th);
            z.push(zz);
            y.push(truth(th, zz) + noise.sample(&mut rng));
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

    // ---- dense held-out probe grid for truth-recovery scoring -------------
    // A 31×21 (θ,z) lattice over [0,2π) × [0,1], DISTINCT from the training
    // nodes (different counts, interior offset in θ), so RMSE-to-truth measures
    // generalization of the fitted surface, not interpolation at the data.
    const P_THETA: usize = 31;
    const P_Z: usize = 21;
    let mut probe_theta: Vec<f64> = Vec::with_capacity(P_THETA * P_Z);
    let mut probe_z: Vec<f64> = Vec::with_capacity(P_THETA * P_Z);
    let mut probe_truth: Vec<f64> = Vec::with_capacity(P_THETA * P_Z);
    for i in 0..P_THETA {
        // Offset the probe θ off the training θ nodes; stay strictly in [0,2π).
        let th = TAU * (i as f64 + 0.5) / (P_THETA as f64);
        for j in 0..P_Z {
            let zz = (j as f64) / ((P_Z - 1) as f64);
            probe_theta.push(th);
            probe_z.push(zz);
            probe_truth.push(truth(th, zz));
        }
    }
    let gam_probe = gam_predict(&probe_theta, &probe_z);
    let gam_truth_rmse = rmse(&gam_probe, &probe_truth);

    // ---- mgcv as an accuracy BASELINE on the SAME truth -------------------
    // mgcv te(bs=c("cc","ps")) is fit to the identical rows and asked to predict
    // the identical probe grid; we score ITS error against the same truth. mgcv
    // is the bar to match-or-beat on recovery accuracy, not the definition of
    // correctness. The cyclic θ margin needs an explicit [0,2π] knot range.
    // mgcv rebuilds the identical probe lattice inside R from the same scalars
    // (no need to ship it as columns — the lattice is fully determined by
    // P_THETA, P_Z and the same generating formulae used on the Rust side).
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
        emit("edf", sum(m$edf))
        # Rebuild the IDENTICAL 31x21 held-out probe lattice used on the Rust
        # side and predict the fitted surface there, so we can score mgcv's
        # recovery RMSE against the same closed-form truth.
        P_THETA <- 31L; P_Z <- 21L
        ig <- 0:(P_THETA - 1); jg <- 0:(P_Z - 1)
        th <- 2 * pi * (ig + 0.5) / P_THETA
        zz <- jg / (P_Z - 1)
        grid <- expand.grid(j = jg, i = ig)           # j fastest -> row-major (i,j)
        pth <- th[grid$i + 1]
        pz  <- zz[grid$j + 1]
        pg  <- data.frame(theta = pth, z = pz)
        emit("probe_pred", as.numeric(predict(m, newdata = pg)))
        # mgcv's own z endpoints at theta = pi/4: f(pi/4, 0) and f(pi/4, 1),
        # for context only (printed, not asserted against).
        emit("zends", as.numeric(predict(m,
              newdata = data.frame(theta = c(pi / 4, pi / 4), z = c(0, 1)))))
        "#,
    );
    let mgcv_probe = r.vector("probe_pred");
    let mgcv_edf = r.scalar("edf");
    let mgcv_zends = r.vector("zends");
    assert_eq!(
        mgcv_probe.len(),
        P_THETA * P_Z,
        "mgcv probe-grid length mismatch"
    );
    let mgcv_truth_rmse = rmse(mgcv_probe, &probe_truth);

    // ---- (2) intrinsic θ-seam continuity (the cyclic-margin contract) ------
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

    // ---- (3) intrinsic z non-periodicity (cylinder vs torus) --------------
    // gam's z-endpoints at θ=π/4. The ground truth is sin(π/2)·(1+z) = 1+z, so
    // truth f(π/4,1)-f(π/4,0) = 1.0 exactly: the surface MUST be free to differ
    // across the z ends. If gam wrongly wrapped z, this gap would collapse to ~0.
    // We assert the recovered gap tracks the TRUTH span, not any peer tool.
    let theta_quarter: Vec<f64> =
        std::iter::repeat_n(std::f64::consts::FRAC_PI_4, z_grid.len()).collect();
    let gam_zprobe = gam_predict(&theta_quarter, &z_grid);
    let gam_z0 = gam_zprobe[0];
    let gam_z1 = gam_zprobe[gam_zprobe.len() - 1];
    let gam_z_endpoint_gap = (gam_z1 - gam_z0).abs();
    let truth_z_span = (truth(std::f64::consts::FRAC_PI_4, 1.0)
        - truth(std::f64::consts::FRAC_PI_4, 0.0))
    .abs(); // = 1.0
    let gam_z_span_err = (gam_z_endpoint_gap - truth_z_span).abs();
    let mgcv_z_endpoint_gap = (mgcv_zends[1] - mgcv_zends[0]).abs();

    // Context-only relative distance to mgcv on the probe grid (printed, never
    // asserted): handy when diagnosing a regression, but NOT a pass criterion.
    let rel_to_mgcv = relative_l2(&gam_probe, mgcv_probe);

    eprintln!(
        "cylinder te(cc,ps): n={n} sigma={sigma} \
         gam_truth_rmse={gam_truth_rmse:.5} mgcv_truth_rmse={mgcv_truth_rmse:.5} \
         gam_edf={gam_edf:.3} mgcv_edf={mgcv_edf:.3} \
         theta_seam_gap={theta_seam_gap:.3e} \
         gam_z_gap={gam_z_endpoint_gap:.4} truth_z_span={truth_z_span:.4} \
         mgcv_z_gap={mgcv_z_endpoint_gap:.4} rel_to_mgcv(context)={rel_to_mgcv:.5}"
    );

    // ---- (1) PRIMARY: truth recovery --------------------------------------
    // The cylinder surface is smooth and low-noise (σ=0.03). A correct REML fit
    // on n=300 must recover it on the held-out probe lattice to within a small
    // multiple of the noise scale. 1.5·σ is a principled, un-weakened bar: it is
    // tighter than σ would be alone for a 2-D surface only if the fit truly
    // generalizes; a real basis/penalty bug blows well past it.
    assert!(
        gam_truth_rmse <= 1.5 * sigma,
        "gam failed to recover the cylinder truth: held-out RMSE={gam_truth_rmse:.5} > {:.5} (=1.5σ)",
        1.5 * sigma
    );
    // Match-or-beat mgcv on the SAME recovery task: gam's error must not exceed
    // the mature baseline's by more than 10%. mgcv is here only as an accuracy
    // yardstick on the identical truth, not as a definition of correctness.
    assert!(
        gam_truth_rmse <= mgcv_truth_rmse * 1.10,
        "gam recovery worse than mgcv baseline: gam_rmse={gam_truth_rmse:.5} > 1.10*mgcv_rmse={:.5}",
        mgcv_truth_rmse * 1.10
    );

    // ---- (2) θ-seam continuity (exact cyclic-margin math) -----------------
    // Value continuity across the azimuthal wrap, exact up to float error. The
    // θ-seam must close to < 1e-6; any larger gap is a sign/threshold bug in
    // gam's periodic closure. This is a mathematical contract of the basis.
    assert!(
        theta_seam_gap < 1e-6,
        "θ-seam not closed: max |f(0,z) - f(2π,z)| = {theta_seam_gap:.3e}"
    );

    // ---- (3) z non-periodicity vs the TRUTH span --------------------------
    // The truth's z-endpoint span at θ=π/4 is exactly 1.0; a correctly
    // non-periodic z marginal keeps the two ends free and recovers that span.
    // We require the recovered gap to land within 0.15 of the truth span (and,
    // implicitly, far from the ~0 a wrongly-wrapped z would force). This is the
    // intrinsic asymmetry distinguishing a cylinder from a torus, asserted
    // against ground truth — not against mgcv.
    assert!(
        gam_z_span_err <= 0.15,
        "z margin mis-recovers the (non-periodic) endpoint span: \
         |f(π/4,1)-f(π/4,0)|={gam_z_endpoint_gap:.4} vs truth span={truth_z_span:.4} \
         (err={gam_z_span_err:.4}); a value near 0 means z wrongly wrapped"
    );
}
