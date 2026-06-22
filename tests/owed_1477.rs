//! Owed-work regression gate for #1477: a Tweedie + default P-spline `s(x)` fit
//! must ship an UNBIASED mean — no systematic tilt and no right-boundary
//! blow-up.
//!
//! On `main` the default `s(x)` basis is the double-penalty P-spline (`bs='ps'`).
//! The PS Gaussian failures in this cluster were fixed (#1392/#1401/#1364), but
//! the fixes never reached the NON-Gaussian path: `relax_smoothing_rho_prior`
//! bailed on a `gaussian_identity` gate for every non-Gaussian family, leaving
//! the Gaussian-tuned symmetric `Normal{0,3}` log-λ cap (a #1089 termination
//! stabiliser centred at λ=1, with no smoothing-selection justification) on the
//! Tweedie smooth. The cap drags the bending log-λ off the REML optimum mgcv
//! reaches, so the Tweedie `ps` fit ships a systematically biased mean (low over
//! the peak, high over the trough) with a hard right-boundary blow-up — while
//! gam's OWN `cr` basis and mgcv's SAME `ps` basis both recover truth on the
//! identical data. EDF stays sane (~5-7), so this is a BIAS, not an overfit.
//!
//! The fix frees length-safe non-Gaussian smooths from the symmetric cap (pure
//! REML = mgcv on the bending coordinate; one-sided select-out on the
//! double-penalty null space). This gate is R-free: it asserts OBJECTIVE truth
//! recovery of the Tweedie mean on the default `ps` basis, pins it within a
//! small factor of gam's `cr` basis on the same data (the issue's contrast), and
//! bounds the right-boundary prediction at x=1.0 against the boundary blow-up.
//!
//! No `let _`, no `#[allow(...)]`, no env vars, no `#[cfg(feature=...)]`.

use csv::StringRecord;
use gam::matrix::LinearOperator;
use gam::smooth::build_term_collection_design;
use gam::test_support::reference::rmse;
use gam::{
    FitConfig, FitResult, encode_recordswith_inferred_schema, fit_from_formula, init_parallelism,
};
use ndarray::Array2;
use std::f64::consts::PI;

/// Deterministic SplitMix64 stream (no external RNG dependency / Python).
struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    fn new(seed: u64) -> Self {
        Self { state: seed }
    }
    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }
    fn next_unit(&mut self) -> f64 {
        let bits = self.next_u64() >> 11; // 53 bits
        ((bits as f64) + 0.5) / (1u64 << 53) as f64
    }
    /// Poisson(rate) by Knuth's product-of-uniforms method.
    fn next_poisson(&mut self, rate: f64) -> u64 {
        if !(rate.is_finite() && rate > 0.0) {
            return 0;
        }
        let l = (-rate).exp();
        let mut k: u64 = 0;
        let mut prod = 1.0;
        loop {
            prod *= self.next_unit();
            if prod <= l {
                return k;
            }
            k += 1;
            if k > 10_000 {
                return k; // safety valve; never reached at these rates
            }
        }
    }
    /// Gamma(shape, scale) for real shape > 0 via Marsaglia–Tsang, using the
    /// SplitMix64 unit stream for both the normal (Box–Muller) and the uniform.
    fn next_gamma(&mut self, shape: f64, scale: f64) -> f64 {
        if shape < 1.0 {
            let g = self.next_gamma(shape + 1.0, scale);
            let u = self.next_unit();
            return g * u.powf(1.0 / shape);
        }
        let d = shape - 1.0 / 3.0;
        let c = 1.0 / (9.0 * d).sqrt();
        loop {
            // Box–Muller standard normal.
            let u1 = self.next_unit();
            let u2 = self.next_unit();
            let n = (-2.0 * u1.ln()).sqrt() * (2.0 * PI * u2).cos();
            let v = (1.0 + c * n).powi(3);
            if v <= 0.0 {
                continue;
            }
            let u = self.next_unit();
            if u.ln() < 0.5 * n * n + d - d * v + d * (v.ln()) {
                return d * v * scale;
            }
        }
    }
    /// Tweedie(mean=mu, power=p, dispersion=phi) for p in (1,2): the compound
    /// Poisson–Gamma. N ~ Poisson(lambda), lambda = mu^(2-p)/(phi(2-p)); each of
    /// the N jumps ~ Gamma(alpha, theta) with alpha = (2-p)/(p-1),
    /// theta = phi(p-1)mu^(p-1); Y = sum of jumps (0 when N = 0). E[Y] = mu.
    fn next_tweedie(&mut self, mu: f64, p: f64, phi: f64) -> f64 {
        let lambda = mu.powf(2.0 - p) / (phi * (2.0 - p));
        let alpha = (2.0 - p) / (p - 1.0);
        let theta = phi * (p - 1.0) * mu.powf(p - 1.0);
        let n = self.next_poisson(lambda);
        let mut y = 0.0;
        for _ in 0..n {
            y += self.next_gamma(alpha, theta);
        }
        y
    }
}

const N: usize = 600;
const P_TWEEDIE: f64 = 1.5;
const PHI: f64 = 0.6;

/// The true mean: a clear unimodal-ish sinusoid on the log scale, large enough
/// amplitude to surface the boundary bias (#1477 notes a low-amplitude scenario
/// hid it). `tm(x) = exp(0.9·sin(2πx) + 0.4)`. tm(1) = exp(0.4) ≈ 1.49: a benign
/// interior-level boundary value, so a blown-up boundary prediction (≈ 2.4× truth
/// in the bug) is unambiguous.
fn truth_mean(x: f64) -> f64 {
    ((0.9 * (2.0 * PI * x).sin()) + 0.4).exp()
}

fn build_data(seed: u64) -> (gam::data::EncodedDataset, Vec<f64>) {
    let mut rng = SplitMix64::new(seed);
    let mut x = Vec::with_capacity(N);
    let mut y = Vec::with_capacity(N);
    for _ in 0..N {
        let xi = rng.next_unit();
        let yi = rng.next_tweedie(truth_mean(xi), P_TWEEDIE, PHI);
        assert!(yi.is_finite() && yi >= 0.0, "Tweedie y must be non-negative finite");
        x.push(xi);
        y.push(yi);
    }
    let headers = ["x", "y"].into_iter().map(String::from).collect();
    let rows: Vec<StringRecord> = x
        .iter()
        .zip(y.iter())
        .map(|(a, b)| StringRecord::from(vec![a.to_string(), b.to_string()]))
        .collect();
    let ds = encode_recordswith_inferred_schema(headers, rows).expect("encode tweedie xy");
    (ds, x)
}

struct FitOut {
    edf: f64,
    /// Fitted mean (response scale) on the supplied evaluation grid.
    mean_on_grid: Vec<f64>,
}

/// Fit Tweedie (fixed p=1.5) `y ~ s(x, bs=<basis>, k=10)` and evaluate the fitted
/// mean on `grid_x` on the response scale (log link → mean = exp(eta)).
fn fit_tweedie(basis: &str, data: &gam::data::EncodedDataset, grid_x: &[f64]) -> FitOut {
    let cfg = FitConfig {
        family: Some("tweedie".to_string()),
        ..FitConfig::default()
    };
    let formula = format!("y ~ s(x, bs=\"{basis}\", k=10)");
    let result = fit_from_formula(&formula, data, &cfg)
        .unwrap_or_else(|e| panic!("tweedie {basis} fit failed: {e:?}"));
    let FitResult::Standard(fit) = result else {
        panic!("expected a Standard GAM fit for tweedie + {basis}");
    };
    let edf = fit.fit.edf_total().expect("edf_total");

    let col = data.column_map();
    let x_idx = col["x"];
    let mut grid = Array2::<f64>::zeros((grid_x.len(), data.headers.len()));
    for (i, &xv) in grid_x.iter().enumerate() {
        grid[[i, x_idx]] = xv;
    }
    let design = build_term_collection_design(grid.view(), &fit.resolvedspec)
        .expect("rebuild design on grid");
    let eta = design.design.apply(&fit.fit.beta);
    let mean_on_grid: Vec<f64> = eta.iter().map(|e| e.exp()).collect();
    FitOut { edf, mean_on_grid }
}

/// A dense evaluation grid INCLUDING the right boundary x=1.0 (the #1477
/// blow-up location) and the left boundary x=0.0.
fn eval_grid() -> Vec<f64> {
    let m = 101usize;
    (0..m).map(|i| i as f64 / (m as f64 - 1.0)).collect()
}

/// #1477: the Tweedie default `ps` smooth must recover the true mean — no
/// systematic tilt, no right-boundary blow-up. We assert OBJECTIVE truth recovery
/// on the response scale across a dense grid, pin `ps` within a small factor of
/// gam's `cr` basis on the same data (the issue's contrast), and bound the
/// boundary prediction at x=1.0.
#[test]
fn tweedie_default_ps_smooth_recovers_unbiased_mean_no_boundary_blowup_1477() {
    init_parallelism();
    let grid = eval_grid();
    let truth: Vec<f64> = grid.iter().map(|&x| truth_mean(x)).collect();
    let mu_min = truth.iter().copied().fold(f64::INFINITY, f64::min);
    let mu_max = truth.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let mu_range = mu_max - mu_min;

    // Deterministic seed block (the bug was robust across seeds: ps worse on 7/8).
    let seeds: &[u64] = &[481001, 481002, 481003, 481004, 481005, 481006];
    let mut ps_worse = 0usize;
    let mut ps_rmse_all = Vec::new();
    let mut ps_boundary_ratio_worst = 0.0_f64;

    for &seed in seeds {
        let (data, _x) = build_data(seed);
        let ps = fit_tweedie("ps", &data, &grid);
        let cr = fit_tweedie("cr", &data, &grid);

        let ps_rmse = rmse(&ps.mean_on_grid, &truth);
        let cr_rmse = rmse(&cr.mean_on_grid, &truth);

        // Right-boundary prediction at x = 1.0 (last grid point).
        let last = grid.len() - 1;
        assert!((grid[last] - 1.0).abs() < 1e-12, "grid must end at x=1.0");
        let boundary_pred = ps.mean_on_grid[last];
        let boundary_truth = truth[last];
        let boundary_ratio = boundary_pred / boundary_truth;
        ps_boundary_ratio_worst = ps_boundary_ratio_worst.max(boundary_ratio);

        eprintln!(
            "[#1477] seed={seed} ps_rmse={ps_rmse:.4} (edf {:.2}) cr_rmse={cr_rmse:.4} \
             (edf {:.2}) x=1.0 ps_pred={boundary_pred:.3} truth={boundary_truth:.3} \
             ratio={boundary_ratio:.3}",
            ps.edf, cr.edf
        );

        // PER-SEED HARD BOUND: the right-boundary prediction must not blow up.
        // The bug shipped ≈2.4× truth at x=1.0; truth recovery sits near 1.0×.
        // 1.6× is comfortably above honest sampling scatter and far below the
        // blow-up, so it fails on the regression and passes on the fix.
        assert!(
            boundary_ratio < 1.6 && boundary_ratio > 0.55,
            "seed {seed}: Tweedie ps right-boundary prediction at x=1.0 is \
             {boundary_pred:.3} vs truth {boundary_truth:.3} (ratio {boundary_ratio:.3}); \
             a blown-up boundary is the #1477 PS-Tweedie defect."
        );

        ps_rmse_all.push(ps_rmse);
        if ps_rmse > cr_rmse {
            ps_worse += 1;
        }
        // PER-SEED CONTRAST: ps must not be dramatically worse than cr on the
        // SAME data (the issue measured ratio ≈ 2.4). A factor of 1.8 catches the
        // bias while tolerating that the two bases legitimately differ a little.
        assert!(
            ps_rmse <= 1.8 * cr_rmse + 1e-9,
            "seed {seed}: Tweedie ps RMSE-to-truth {ps_rmse:.4} >> cr RMSE {cr_rmse:.4} \
             (ratio {:.2}) on identical data — the non-Gaussian PS mean path is biased (#1477).",
            ps_rmse / cr_rmse
        );
    }

    // AGGREGATE truth-recovery bar (R-free, OBJECTIVE): the median ps RMSE must
    // recover the mean within a noise-scaled fraction of the signal range. The
    // bug's median ps RMSE ≈ 0.47 with mu_range ≈ 3-4; a recovered fit lands well
    // under 0.30·range. This is the primary unbiasedness gate.
    ps_rmse_all.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let median_ps = ps_rmse_all[ps_rmse_all.len() / 2];
    let abs_bar = 0.30 * mu_range;
    eprintln!(
        "[#1477] mu_range={mu_range:.3} median ps_rmse={median_ps:.4} abs_bar={abs_bar:.4} \
         ps_worse_than_cr={ps_worse}/{} worst_boundary_ratio={ps_boundary_ratio_worst:.3}",
        seeds.len()
    );
    assert!(
        median_ps <= abs_bar,
        "Tweedie default ps median RMSE-to-truth {median_ps:.4} > {abs_bar:.4} (0.30·range): \
         the non-Gaussian PS mean is systematically biased (#1477)."
    );

    // The PS path must not be the systematically-worse basis: the bug had ps
    // worse on 7/8 seeds. After the fix ps and cr are both clean, so ps must NOT
    // be worse on a strict majority of seeds.
    assert!(
        ps_worse <= seeds.len() / 2,
        "Tweedie ps RMSE-to-truth is worse than cr on {ps_worse}/{} seeds — the PS-specific \
         non-Gaussian bias (#1477) is still present.",
        seeds.len()
    );
}
