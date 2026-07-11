//! End-to-end SCALE + MEMORY tests for gam's non-periodic Euclidean Duchon
//! smoother. The claim under test is operational, not comparative: the
//! redesigned default `duchon(x, k=...)` — a cubic (`r³`) polyharmonic
//! structural smoother with an analytic native reproducing-norm Gram penalty
//! plus null-space ridge — must FIT AT SCALE without OOM and still RECOVER a
//! known smooth truth.
//!
//! OBJECTIVE METRICS (the only pass/fail claims here): for each fit we assert
//! (a) the fit completes, (b) every fitted value is finite, and (c) the
//! truth-recovery RMSE on a held-out interior grid is clearly below the trivial
//! mean/zero predictor (RMS of the demeaned truth). We do NOT compare against
//! any reference tool and we do NOT assert closeness to a reference output —
//! these tests stand on gam's own truth recovery at scale. (The companion file
//! `quality_vs_mgcv_duchon_smooth.rs` owns the match-or-beat-mgcv comparison.)
//!
//! RAM DISCIPLINE. The whole point of the lazy path is to NOT allocate the
//! dense `n × p` design when it would be large, so we deliberately keep `n`
//! bounded (≤ 40_000) and exercise the chunked operator's *correctness*, not an
//! actual multi-GB allocation. Caps and their rationale are documented at each
//! test below. With n ≤ 40_000 the raw input array is a few hundred KiB and the
//! p × p normal-equations Gram is single-digit MiB, so CI memory stays bounded
//! even on the lazy path (which streams the design in row chunks rather than
//! materializing it).

use gam::ResourcePolicy;
use gam::matrix::LinearOperator;
use gam::smooth::build_term_collection_design;
use gam::test_support::reference::rmse;
use gam::{
    FitConfig, FitResult, encode_recordswith_inferred_schema, fit_from_formula, init_parallelism,
};
use ndarray::Array2;
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand_distr::{Distribution, Normal};

/// Root-mean-square of a slice (used for the trivial-predictor floor).
fn rms(v: &[f64]) -> f64 {
    (v.iter().map(|&t| t * t).sum::<f64>() / v.len() as f64).sqrt()
}

/// Build a single-feature dataset `{x, y}` from parallel vectors.
fn encode_xy(x: &[f64], y: &[f64]) -> gam::data::EncodedDataset {
    let headers = ["x", "y"].into_iter().map(String::from).collect();
    let rows = x
        .iter()
        .zip(y.iter())
        .map(|(a, b)| csv::StringRecord::from(vec![a.to_string(), b.to_string()]))
        .collect();
    encode_recordswith_inferred_schema(headers, rows).expect("encode synthetic 1-D dataset")
}

/// Build a two-feature dataset `{x, z, y}` from parallel vectors.
fn encode_xzy(x: &[f64], z: &[f64], y: &[f64]) -> gam::data::EncodedDataset {
    let headers = ["x", "z", "y"].into_iter().map(String::from).collect();
    let rows = (0..x.len())
        .map(|i| {
            csv::StringRecord::from(vec![x[i].to_string(), z[i].to_string(), y[i].to_string()])
        })
        .collect();
    encode_recordswith_inferred_schema(headers, rows).expect("encode synthetic 2-D dataset")
}

/// Fit `formula` as a Gaussian GAM and return the standard fit, panicking with a
/// clear message on the non-standard arm.
fn fit_gaussian(formula: &str, ds: &gam::data::EncodedDataset) -> gam::StandardFitResult {
    let cfg = FitConfig {
        family: Some("gaussian".to_string()),
        ..FitConfig::default()
    };
    let result = fit_from_formula(formula, ds, &cfg)
        .unwrap_or_else(|e| panic!("gam duchon fit failed for `{formula}`: {e}"));
    match result {
        FitResult::Standard(fit) => fit,
        _ => panic!("expected a standard GAM fit for a gaussian Duchon smooth: `{formula}`"),
    }
}

/// Fit with the caller's resource policy so a memory-routing fixture can pass
/// the exact admission boundary that it asserted.
fn fit_gaussian_with_resource_policy(
    formula: &str,
    ds: &gam::data::EncodedDataset,
    resource_policy: ResourcePolicy,
) -> gam::StandardFitResult {
    let cfg = FitConfig {
        family: Some("gaussian".to_string()),
        resource_policy: Some(resource_policy),
        ..FitConfig::default()
    };
    let result = fit_from_formula(formula, ds, &cfg)
        .unwrap_or_else(|e| panic!("gam duchon fit failed for `{formula}`: {e}"));
    match result {
        FitResult::Standard(fit) => fit,
        _ => panic!("expected a standard GAM fit for a gaussian Duchon smooth: `{formula}`"),
    }
}

#[test]
fn duchon_1d_recovers_truth_across_increasing_n() {
    init_parallelism();

    // Low-frequency truth f(x) = sin(2π·x): exactly ONE period over [0,1], which
    // a k=40 cubic Duchon basis resolves comfortably — so the achievable error
    // is the noise floor, not under-resolution bias. We sweep n to prove the
    // standard (dense) Duchon path scales: n ∈ {2_000, 10_000, 40_000}.
    //
    // n CAP = 40_000. WHY: at k=40 the dense design is n·~42·8 bytes ≈ 13 MiB at
    // n=40_000 — well under the 256 MiB default materialization budget, so this
    // arm stays on the dense path on purpose (the lazy path gets its own test
    // below). 40_000 is large enough to be a genuine scale test while keeping
    // the raw input array (~0.6 MiB) and all linear algebra trivially in-RAM for
    // CI; pushing n higher buys no extra coverage here and only burns CI time.
    let sigma = 0.05;
    for &n in &[2_000usize, 10_000, 40_000] {
        let mut rng = StdRng::seed_from_u64(0xD0_C0_00 ^ n as u64);
        let noise = Normal::new(0.0, sigma).expect("normal");
        let mut x: Vec<f64> = (0..n).map(|i| i as f64 / (n as f64 - 1.0)).collect();
        x.sort_by(|a, b| a.partial_cmp(b).expect("finite x"));
        let two_pi = 2.0 * std::f64::consts::PI;
        let y: Vec<f64> = x
            .iter()
            .map(|&t| (two_pi * t).sin() + noise.sample(&mut rng))
            .collect();

        let ds = encode_xy(&x, &y);
        let x_idx = ds.column_map()["x"];
        let fit = fit_gaussian("y ~ duchon(x, k=40)", &ds);

        // Every training fitted value must be finite. With identity link the
        // mean is design*beta; apply the (possibly chunked) operator to beta.
        let train_fitted: Vec<f64> = fit.design.design.apply(&fit.fit.beta).to_vec();
        assert!(
            train_fitted.iter().all(|v| v.is_finite()),
            "duchon 1d n={n}: non-finite fitted value among training points"
        );

        // Truth recovery on a dense interior grid (avoid extrapolation edges).
        let m = 201usize;
        let x_test: Vec<f64> = (0..m)
            .map(|i| 0.01 + 0.98 * i as f64 / (m as f64 - 1.0))
            .collect();
        let y_truth: Vec<f64> = x_test.iter().map(|&t| (two_pi * t).sin()).collect();

        let mut grid = Array2::<f64>::zeros((m, ds.headers.len()));
        for (i, &t) in x_test.iter().enumerate() {
            grid[[i, x_idx]] = t;
        }
        let design = build_term_collection_design(grid.view(), &fit.resolvedspec)
            .expect("rebuild Duchon design at 1-D test grid");
        let gam_fitted: Vec<f64> = design.design.apply(&fit.fit.beta).to_vec();
        assert!(
            gam_fitted.iter().all(|v| v.is_finite()),
            "duchon 1d n={n}: non-finite fitted value on the test grid"
        );

        let recovery_rmse = rmse(&gam_fitted, &y_truth);
        // Trivial-predictor floor: a constant predictor scores RMS of the
        // demeaned truth = RMS(sin over interior) ≈ 0.707. A real reconstruction
        // of a single-period sine from n≥2000 points at σ=0.05 sits far below
        // that; 0.20 is a principled non-degeneracy bar that still catches a
        // blown-up or collapsed fit without encoding the noise floor.
        let truth_mean = y_truth.iter().sum::<f64>() / m as f64;
        let demeaned: Vec<f64> = y_truth.iter().map(|&t| t - truth_mean).collect();
        let trivial = rms(&demeaned);
        eprintln!(
            "duchon-scale-1d: n={n} sigma={sigma} k=40 recovery_rmse={recovery_rmse:.4} \
             trivial_predictor_rms={trivial:.4}"
        );
        assert!(
            recovery_rmse < 0.20,
            "duchon 1d n={n}: failed to recover sin(2πx): recovery_rmse={recovery_rmse:.4} \
             (trivial-predictor RMS≈{trivial:.4}); fit is degenerate"
        );
    }
}

#[test]
fn duchon_2d_recovers_smooth_surface() {
    init_parallelism();

    // 2-D Duchon syntax is the multi-arg term `duchon(x, z, k=...)` (each smooth
    // coordinate is a separate argument; `k`/`centers` set the number of radial
    // centers). The default power in 2D is s=(d-1)/2 = 0.5 over the affine null
    // space — the magic structural surface smoother.
    //
    // Known smooth truth on [0,1]²: a separable low-frequency surface
    // f(x,z) = sin(2π·x) · cos(2π·z). It is globally smooth (one period per
    // axis) so a moderate center count resolves it; recovery error is the noise
    // floor, not under-resolution.
    //
    // n CAP = 4_000, k = 60. WHY: 2-D fitting on a uniform random cloud needs
    // enough points to pin the surface but the test is about correctness at
    // moderate scale, not OOM. The dense design here is 4_000·~63·8 ≈ 2 MiB and
    // the 63×63 Gram is trivial; keeping n=4_000 keeps CI fast while still being
    // a genuine surface-recovery check.
    let n = 4_000usize;
    let k = 60usize;
    let sigma = 0.05;
    let mut rng = StdRng::seed_from_u64(0x2D_C0_FE);
    let unit = rand_distr::Uniform::new(0.0, 1.0).expect("uniform");
    let noise = Normal::new(0.0, sigma).expect("normal");
    let two_pi = 2.0 * std::f64::consts::PI;

    let x: Vec<f64> = (0..n).map(|_| unit.sample(&mut rng)).collect();
    let z: Vec<f64> = (0..n).map(|_| unit.sample(&mut rng)).collect();
    let truth_at = |a: f64, b: f64| (two_pi * a).sin() * (two_pi * b).cos();
    let y: Vec<f64> = (0..n)
        .map(|i| truth_at(x[i], z[i]) + noise.sample(&mut rng))
        .collect();

    let ds = encode_xzy(&x, &z, &y);
    let col = ds.column_map();
    let (x_idx, z_idx) = (col["x"], col["z"]);
    let fit = fit_gaussian(&format!("y ~ duchon(x, z, k={k})"), &ds);

    let train_fitted: Vec<f64> = fit.design.design.apply(&fit.fit.beta).to_vec();
    assert!(
        train_fitted.iter().all(|v| v.is_finite()),
        "duchon 2d: non-finite fitted value among training points"
    );

    // Interior tensor grid for truth recovery (avoid the convex-hull edges where
    // a scattered-data smoother extrapolates).
    let g = 25usize;
    let coords: Vec<f64> = (0..g)
        .map(|i| 0.08 + 0.84 * i as f64 / (g as f64 - 1.0))
        .collect();
    let mut grid = Array2::<f64>::zeros((g * g, ds.headers.len()));
    let mut y_truth = Vec::with_capacity(g * g);
    for (i, &gx) in coords.iter().enumerate() {
        for (j, &gz) in coords.iter().enumerate() {
            let row = i * g + j;
            grid[[row, x_idx]] = gx;
            grid[[row, z_idx]] = gz;
            y_truth.push(truth_at(gx, gz));
        }
    }
    let design = build_term_collection_design(grid.view(), &fit.resolvedspec)
        .expect("rebuild Duchon design at 2-D test grid");
    let gam_fitted: Vec<f64> = design.design.apply(&fit.fit.beta).to_vec();
    assert!(
        gam_fitted.iter().all(|v| v.is_finite()),
        "duchon 2d: non-finite fitted value on the test grid"
    );

    let recovery_rmse = rmse(&gam_fitted, &y_truth);
    let truth_mean = y_truth.iter().sum::<f64>() / y_truth.len() as f64;
    let demeaned: Vec<f64> = y_truth.iter().map(|&t| t - truth_mean).collect();
    let trivial = rms(&demeaned);
    eprintln!(
        "duchon-scale-2d: n={n} k={k} sigma={sigma} recovery_rmse={recovery_rmse:.4} \
         trivial_predictor_rms={trivial:.4}"
    );
    // The surface has RMS amplitude ≈ 0.5 over the interior; a constant
    // predictor scores ≈ that. Half of it is a principled non-degeneracy bar
    // that a real reconstruction clears with room to spare while still failing a
    // collapsed/blown-up fit.
    assert!(
        recovery_rmse < 0.5 * trivial,
        "duchon 2d: failed to recover sin(2πx)cos(2πz): recovery_rmse={recovery_rmse:.4} \
         vs trivial-predictor RMS={trivial:.4}; fit is degenerate"
    );
}

#[test]
fn duchon_lazy_chunked_path_is_exercised_and_recovers_truth() {
    init_parallelism();

    // This test forces the MEMORY-SAVING lazy chunked design operator
    // (`ChunkedKernelDesignOperator`, gated by `should_use_lazy_spatial_design`)
    // and asserts the streamed path is still correct. The lazy path activates
    // when the would-be dense `n × p` design exceeds the active policy's
    // `max_single_materialization_bytes`. The fixture derives that boundary
    // from the adjacent design with one fewer basis column, so it is stable
    // across hosts without an environment override or a fixed magic cap.
    //
    // n = 40_000, k = 900. WHY THESE NUMBERS: the constrained radial block has
    // k-2 columns and the affine polynomial block restores 2, so the lazy gate
    // sees exactly p=k before global identifiability. The forbidden dense block
    // is 40_000·900·8 = 288,000,000 bytes (≈275 MiB), while streamed row
    // chunks and the p×p normal-equation folds remain bounded independently of
    // n. The test therefore exercises the operator's correctness without ever
    // allocating the full n×p design.
    let n = 40_000usize;
    let k = 900usize;

    // The policy admits the otherwise-identical (k-1)-column design and rejects
    // this k-column design. Checked arithmetic makes dimension overflow a test
    // construction error rather than a falsely tiny routing footprint.
    let dense_design_bytes = n
        .checked_mul(k)
        .and_then(|cells| cells.checked_mul(std::mem::size_of::<f64>()))
        .expect("lazy-path dense design footprint must fit usize");
    let fixture_budget = n
        .checked_mul(k.checked_sub(1).expect("lazy-path k must be positive"))
        .and_then(|cells| cells.checked_mul(std::mem::size_of::<f64>()))
        .expect("lazy-path adjacent-design budget must fit usize");
    let mut policy = ResourcePolicy::analytic_operator_required();
    policy.max_single_materialization_bytes = fixture_budget;
    assert!(
        dense_design_bytes > policy.max_single_materialization_bytes,
        "lazy-path test misconfigured: dense design {dense_design_bytes} bytes does not exceed \
         the mechanism-derived materialization budget {} bytes for n={n}, k={k}",
        policy.max_single_materialization_bytes
    );

    // Low-frequency truth f(x) = sin(2π·x), one period over [0,1]; 900 centers
    // resolve it to the noise floor.
    let sigma = 0.05;
    let mut rng = StdRng::seed_from_u64(0x1A_2B_3C);
    let noise = Normal::new(0.0, sigma).expect("normal");
    let mut x: Vec<f64> = (0..n).map(|i| i as f64 / (n as f64 - 1.0)).collect();
    x.sort_by(|a, b| a.partial_cmp(b).expect("finite x"));
    let two_pi = 2.0 * std::f64::consts::PI;
    let y: Vec<f64> = x
        .iter()
        .map(|&t| (two_pi * t).sin() + noise.sample(&mut rng))
        .collect();

    let ds = encode_xy(&x, &y);
    let x_idx = ds.column_map()["x"];
    let fit =
        fit_gaussian_with_resource_policy(&format!("y ~ duchon(x, k={k})"), &ds, policy.clone());

    let duchon_design = fit
        .design
        .smooth
        .term_designs
        .first()
        .expect("lazy-path fit must contain its Duchon term design");
    assert!(
        duchon_design.is_operator_backed(),
        "Duchon term crossed the fixture budget but was not operator-backed"
    );
    assert!(
        duchon_design.as_dense_ref().is_none(),
        "operator-backed Duchon term exposed a hidden full-design materialization"
    );

    let train_fitted: Vec<f64> = fit.design.design.apply(&fit.fit.beta).to_vec();
    assert!(
        train_fitted.iter().all(|v| v.is_finite()),
        "duchon lazy path: non-finite fitted value among training points"
    );

    let m = 201usize;
    let x_test: Vec<f64> = (0..m)
        .map(|i| 0.01 + 0.98 * i as f64 / (m as f64 - 1.0))
        .collect();
    let y_truth: Vec<f64> = x_test.iter().map(|&t| (two_pi * t).sin()).collect();

    let mut grid = Array2::<f64>::zeros((m, ds.headers.len()));
    for (i, &t) in x_test.iter().enumerate() {
        grid[[i, x_idx]] = t;
    }
    let design = build_term_collection_design(grid.view(), &fit.resolvedspec)
        .expect("rebuild Duchon design at lazy-path test grid");
    let gam_fitted: Vec<f64> = design.design.apply(&fit.fit.beta).to_vec();
    assert!(
        gam_fitted.iter().all(|v| v.is_finite()),
        "duchon lazy path: non-finite fitted value on the test grid"
    );

    let recovery_rmse = rmse(&gam_fitted, &y_truth);
    let truth_mean = y_truth.iter().sum::<f64>() / m as f64;
    let demeaned: Vec<f64> = y_truth.iter().map(|&t| t - truth_mean).collect();
    let trivial = rms(&demeaned);
    eprintln!(
        "duchon-scale-lazy: n={n} k={k} sigma={sigma} \
         dense_design_MiB={:.1} fixture_budget_MiB={:.1} recovery_rmse={recovery_rmse:.4} \
         trivial_predictor_rms={trivial:.4}",
        dense_design_bytes as f64 / (1024.0 * 1024.0),
        policy.max_single_materialization_bytes as f64 / (1024.0 * 1024.0),
    );
    assert!(
        recovery_rmse < 0.20,
        "duchon lazy path: failed to recover sin(2πx) on the chunked operator: \
         recovery_rmse={recovery_rmse:.4} (trivial-predictor RMS≈{trivial:.4})"
    );
}
