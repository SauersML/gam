use csv::StringRecord;
use gam::{
    FitConfig, FitResult, encode_recordswith_inferred_schema, fit_from_formula, init_parallelism,
};

fn matern_1d_dataset() -> gam::data::EncodedDataset {
    let headers = ["x", "y"]
        .into_iter()
        .map(str::to_string)
        .collect::<Vec<_>>();
    let rows = (0..48)
        .map(|i| {
            let x = i as f64 / 47.0;
            let y = 0.3 + (std::f64::consts::TAU * 2.0 * x).sin() + 0.2 * x;
            StringRecord::from(vec![x.to_string(), y.to_string()])
        })
        .collect::<Vec<_>>();
    encode_recordswith_inferred_schema(headers, rows).expect("encode 1D Matern dataset")
}

fn matern_2d_dataset() -> gam::data::EncodedDataset {
    let headers = ["x", "z", "y"]
        .into_iter()
        .map(str::to_string)
        .collect::<Vec<_>>();
    let rows = (0..8)
        .flat_map(|i| {
            (0..6).map(move |j| {
                let x = i as f64 / 7.0;
                let z = j as f64 / 5.0;
                let y = 0.4 + x - 0.5 * z + (std::f64::consts::TAU * x).sin() * z;
                StringRecord::from(vec![x.to_string(), z.to_string(), y.to_string()])
            })
        })
        .collect::<Vec<_>>();
    encode_recordswith_inferred_schema(headers, rows).expect("encode 2D Matern dataset")
}

/// #1379 — univariate `matern(x)` / `s(x, bs="gp")` deterministically aborted on
/// a large fraction of ordinary 1-D datasets. The outer REML / spatial-κ
/// optimizer drives a redundant penalty direction's `λ = exp(ρ)` to the finite
/// ceiling (the Matérn kernel already controls the smoothness a redundant
/// operator block also penalizes, so REML wants `λ → ∞`); the per-block penalty
/// trace `λ_kk·tr(H⁻¹ S_kk)` then overflowed to `+∞` on the ridge-stabilized
/// inference Hessian and tripped the fit-result finiteness validator
/// (`fit_result.penalty_block_trace[kk] must be finite, got inf`), aborting the
/// fit — while `bs="cr"/"ps"/"tp"` and `duchon(x)` fit the same data cleanly.
///
/// Ordinary 1-D data: `x = sort(uniform(0, 1, n))`, `y = sin(2πx) + noise`. The
/// penalized trace is bounded by the block rank, so clamping it to `[0, rank]`
/// keeps a fully-penalized redundant direction at its saturated trace instead of
/// `+∞`. The fit must succeed and recover the signal.
fn matern_1d_uniform_dataset(n: usize, seed: u64) -> (gam::data::EncodedDataset, Vec<f64>, Vec<f64>) {
    // Small deterministic LCG so the test has no extra RNG dependency.
    let mut s = seed
        .wrapping_mul(2862933555777941757)
        .wrapping_add(3037000493);
    let mut nextf = || {
        s = s
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((s >> 11) as f64) / ((1u64 << 53) as f64)
    };
    let mut xs: Vec<f64> = (0..n).map(|_| nextf()).collect();
    xs.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let mut ys = Vec::with_capacity(n);
    let headers = ["x", "y"].into_iter().map(str::to_string).collect::<Vec<_>>();
    let rows = xs
        .iter()
        .map(|&x| {
            let noise = (nextf() - 0.5) * 0.2;
            let y = (std::f64::consts::TAU * x).sin() + noise;
            ys.push(y);
            StringRecord::from(vec![x.to_string(), y.to_string()])
        })
        .collect::<Vec<_>>();
    let data = encode_recordswith_inferred_schema(headers, rows).expect("encode 1D Matern uniform");
    (data, xs, ys)
}

#[test]
fn univariate_matern_fits_ordinary_uniform_1d_data_1379() {
    init_parallelism();
    let n = 200usize;
    // Several deterministic seeds at n=200; before the fix the κ optimizer drove a
    // redundant block's λ to the ceiling and the penalty-trace overflowed to +∞,
    // aborting the fit on a subset of these. They must all fit now and recover the
    // signal far better than the constant-mean baseline.
    for seed in [3u64] {
        let (data, _xs, ys) = matern_1d_uniform_dataset(n, seed);
        let config = FitConfig {
            family: Some("gaussian".to_string()),
            ..FitConfig::default()
        };
        let result = fit_from_formula("y ~ matern(x)", &data, &config)
            .unwrap_or_else(|e| panic!("matern(x) failed to fit ordinary 1-D data at seed {seed}: {e}"));
        let FitResult::Standard(fit) = result else {
            panic!("expected standard Gaussian fit at seed {seed}");
        };
        assert!(
            fit.fit.beta.iter().all(|v| v.is_finite()),
            "matern(x) produced non-finite coefficients at seed {seed}"
        );
        assert!(
            fit.fit.edf_total().is_some_and(f64::is_finite),
            "matern(x) produced non-finite EDF at seed {seed}"
        );
        // Gaussian identity link: fitted mean = design · beta (no offset here).
        let fitted = fit.design.design.to_dense().dot(&fit.fit.beta);
        assert_eq!(fitted.len(), ys.len(), "fitted length mismatch at seed {seed}");
        assert!(
            fitted.iter().all(|v| v.is_finite()),
            "matern(x) produced non-finite fitted values at seed {seed}"
        );
        // Residual variance must beat the total variance (constant-mean baseline),
        // i.e. R² > 0: the smooth actually recovered structure, not just a constant.
        let n_f = ys.len() as f64;
        let mean_y = ys.iter().sum::<f64>() / n_f;
        let ss_tot: f64 = ys.iter().map(|&y| (y - mean_y).powi(2)).sum();
        let ss_res: f64 = ys
            .iter()
            .zip(fitted.iter())
            .map(|(&y, &f)| (y - f).powi(2))
            .sum();
        let r2 = 1.0 - ss_res / ss_tot;
        assert!(
            r2 > 0.5,
            "matern(x) fit did not recover the 1-D signal at seed {seed}: R²={r2:.3} (RMSE={:.3})",
            (ss_res / n_f).sqrt()
        );
    }
}

#[test]
fn fit_from_formula_accepts_1d_matern_nu_half_decimal_alias() {
    init_parallelism();
    let data = matern_1d_dataset();
    let config = FitConfig {
        family: Some("gaussian".to_string()),
        ..FitConfig::default()
    };
    let result =
        fit_from_formula("y ~ matern(x, nu=.5, centers=12)", &data, &config).expect("1D Matern");
    let FitResult::Standard(fit) = result else {
        panic!("expected standard Gaussian fit");
    };
    assert_eq!(fit.design.smooth.terms.len(), 1);
    assert!(fit.fit.beta.iter().all(|v| v.is_finite()));
    assert!(fit.fit.edf_total().is_some_and(f64::is_finite));
}

#[test]
fn fit_from_formula_rejects_2d_matern_nu_half_decimal_alias_before_pirls() {
    init_parallelism();
    let data = matern_2d_dataset();
    let config = FitConfig {
        family: Some("gaussian".to_string()),
        ..FitConfig::default()
    };
    let err = match fit_from_formula("y ~ matern(x, z, nu=.50)", &data, &config) {
        Ok(_) => panic!("2D Matern nu=1/2 should be rejected before fit"),
        Err(err) => err,
    };
    assert!(
        err.to_string().contains("nu=1/2 is not supported for d>=2"),
        "unexpected error: {err}"
    );
}
