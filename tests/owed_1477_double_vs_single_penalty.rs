//! Owed-work regression gate for #1477, from a DIFFERENT angle than the
//! boundary / `cr`-contrast in `owed_1477.rs`: isolate the double-penalty
//! null-space coordinate itself.
//!
//! #1477's root cause is that the non-Gaussian double-penalty `ps` path drove its
//! null-space log-λ with an aggressive penalized-complexity SELECT-OUT prior in
//! every determinacy regime. On data whose null space (the linear trend) is
//! GENUINELY SUPPORTED — `sin(2πx)` over `[0,1]` has a real linear component,
//! `∫₀¹ sin(2πx)(x−½)dx ≠ 0` — that prior OVER-SHRINKS the supported trend (the
//! #1476 over-shrink failure mode, surfacing here as a Tweedie right-boundary
//! blow-up), and on hard seeds it carves a competing high-λ_null basin the outer
//! optimizer mis-navigates into a falsely-`converged` EDF-inflated overfit.
//!
//! The fix unified the null-space prior with the Gaussian path's wide,
//! weakly-informative degeneracy prior (no select-out bias when well-determined)
//! and made the cost-stalled flat valley certify by a score-relative stationarity
//! bound. The cleanest statement of the resulting contract is BASIS-CONTROLLED:
//! on identical Tweedie data the DEFAULT double-penalty `ps` smooth must recover
//! the mean essentially as well as the SINGLE-penalty `ps` smooth — same basis,
//! the ONLY difference being whether the null-space ridge is present — with no EDF
//! inflation and no boundary blow-up. The pre-fix double penalty lands ≈2× worse
//! RMSE-to-truth and rails one seed's EDF to ≈k; the single penalty (which carries
//! no null-space ridge to over-shrink) is the clean reference.
//!
//! R-free / Python-free: the contrast is gam-internal (double vs single penalty),
//! not a tool comparison. No `let _`, no `#[allow]`, no env vars.

use csv::StringRecord;
use gam::matrix::LinearOperator;
use gam::smooth::build_term_collection_design;
use gam::test_support::reference::rmse;
use gam::{
    FitConfig, FitResult, encode_recordswith_inferred_schema, fit_from_formula, init_parallelism,
};
use ndarray::Array2;
use std::f64::consts::PI;

/// Deterministic SplitMix64 → Tweedie(p, φ) compound Poisson–Gamma draws.
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
        let bits = self.next_u64() >> 11;
        ((bits as f64) + 0.5) / (1u64 << 53) as f64
    }
    fn next_poisson(&mut self, rate: f64) -> u64 {
        if !(rate.is_finite() && rate > 0.0) {
            return 0;
        }
        let l = (-rate).exp();
        let mut k = 0u64;
        let mut prod = 1.0;
        loop {
            prod *= self.next_unit();
            if prod <= l {
                return k;
            }
            k += 1;
            if k > 10_000 {
                return k;
            }
        }
    }
    fn next_gamma(&mut self, shape: f64, scale: f64) -> f64 {
        if shape < 1.0 {
            let g = self.next_gamma(shape + 1.0, scale);
            let u = self.next_unit();
            return g * u.powf(1.0 / shape);
        }
        let d = shape - 1.0 / 3.0;
        let c = 1.0 / (9.0 * d).sqrt();
        loop {
            let u1 = self.next_unit();
            let u2 = self.next_unit();
            let nrm = (-2.0 * u1.ln()).sqrt() * (2.0 * PI * u2).cos();
            let v = (1.0 + c * nrm).powi(3);
            if v <= 0.0 {
                continue;
            }
            let u = self.next_unit();
            if u.ln() < 0.5 * nrm * nrm + d - d * v + d * (v.ln()) {
                return d * v * scale;
            }
        }
    }
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

/// Log-mean truth with a SUPPORTED linear null component (sin over [0,1]).
fn truth_mean(x: f64) -> f64 {
    ((0.9 * (2.0 * PI * x).sin()) + 0.4).exp()
}

struct FitOut {
    edf: f64,
    /// Fitted mean on the response scale across the evaluation grid.
    mean_on_grid: Vec<f64>,
}

fn build_data(seed: u64) -> gam::data::EncodedDataset {
    let mut rng = SplitMix64::new(seed);
    let mut x = Vec::with_capacity(N);
    let mut y = Vec::with_capacity(N);
    for _ in 0..N {
        let xi = rng.next_unit();
        let yi = rng.next_tweedie(truth_mean(xi), P_TWEEDIE, PHI);
        assert!(
            yi.is_finite() && yi >= 0.0,
            "Tweedie y must be non-negative finite"
        );
        x.push(xi);
        y.push(yi);
    }
    let headers = ["x", "y"].into_iter().map(String::from).collect();
    let rows: Vec<StringRecord> = x
        .iter()
        .zip(y.iter())
        .map(|(a, b)| StringRecord::from(vec![a.to_string(), b.to_string()]))
        .collect();
    encode_recordswith_inferred_schema(headers, rows).expect("encode tweedie xy")
}

fn fit_tweedie(double_penalty: bool, data: &gam::data::EncodedDataset, grid_x: &[f64]) -> FitOut {
    let cfg = FitConfig {
        family: Some("tweedie".to_string()),
        ..FitConfig::default()
    };
    let formula = format!("y ~ s(x, bs=\"ps\", k=10, double_penalty={double_penalty})");
    let FitResult::Standard(fit) = fit_from_formula(&formula, data, &cfg)
        .unwrap_or_else(|e| panic!("tweedie ps double_penalty={double_penalty} fit failed: {e:?}"))
    else {
        panic!("expected a Standard GAM fit");
    };
    let edf = fit.fit.edf_total().expect("edf_total");
    let col = data.column_map();
    let x_idx = col["x"];
    let mut grid = Array2::<f64>::zeros((grid_x.len(), data.headers.len()));
    for (i, &xv) in grid_x.iter().enumerate() {
        grid[[i, x_idx]] = xv;
    }
    let design =
        build_term_collection_design(grid.view(), &fit.resolvedspec).expect("rebuild design");
    let eta = design.design.apply(&fit.fit.beta);
    let mean_on_grid: Vec<f64> = eta.iter().map(|e| e.exp()).collect();
    FitOut { edf, mean_on_grid }
}

#[test]
fn tweedie_default_double_penalty_matches_single_penalty_no_overshrink_1477() {
    init_parallelism();
    let grid: Vec<f64> = (0..101).map(|i| i as f64 / 100.0).collect();
    let truth: Vec<f64> = grid.iter().map(|&x| truth_mean(x)).collect();

    let seeds: &[u64] = &[481001, 481003, 481004, 481007, 481011];
    let mut dp_worse = 0usize;
    for &seed in seeds {
        let data = build_data(seed);
        let dp = fit_tweedie(true, &data, &grid);
        let sp = fit_tweedie(false, &data, &grid);

        let dp_rmse = rmse(&dp.mean_on_grid, &truth);
        let sp_rmse = rmse(&sp.mean_on_grid, &truth);
        let last = grid.len() - 1;
        eprintln!(
            "[#1477 dp/sp] seed={seed} dp_rmse={dp_rmse:.4} (edf {:.2}, conv certified) \
             sp_rmse={sp_rmse:.4} (edf {:.2}, conv certified) x=1: dp={:.3} sp={:.3} truth={:.3}",
            dp.edf,
            sp.edf,
            dp.mean_on_grid[last],
            sp.mean_on_grid[last],
            truth[last]
        );

        // CONTRAST: the double-penalty `ps` mean must track the single-penalty
        // `ps` mean on the SAME data — the only difference is the null-space
        // ridge. The pre-fix over-shrink lands ≈2× worse; 1.5× catches it and
        // tolerates the legitimate small regularization difference.
        assert!(
            dp_rmse <= 1.5 * sp_rmse + 1e-9,
            "seed {seed}: Tweedie double-penalty ps RMSE-to-truth {dp_rmse:.4} >> single-penalty \
             ps {sp_rmse:.4} (ratio {:.2}) on identical data — the null-space select-out prior \
             over-shrinks the supported linear trend (#1477).",
            dp_rmse / sp_rmse.max(1e-12)
        );

        // NO EDF INFLATION: the double penalty must not rail the EDF toward the
        // full basis (k=10 → 9 after centering). The #1426-class false-converged
        // overfit shipped EDF ≈ 9.5 here; the single penalty sits ~6.
        assert!(
            dp.edf <= sp.edf + 1.5,
            "seed {seed}: Tweedie double-penalty ps EDF {:.2} inflated far above single-penalty \
             EDF {:.2} — a near-full-basis overfit (#1477/#1426).",
            dp.edf,
            sp.edf
        );

        // HONEST VERDICT: a certified double-penalty fit (fit existence is the
        // sealed convergence proof, SPEC 20) must not be a near-full-basis
        // overfit (the #1426 silent-overfit contract).
        assert!(
            dp.edf < 8.5,
            "seed {seed}: certified Tweedie double-penalty ps fit with EDF {:.2} \
             (near the k=10 full basis) — a silently-certified overfit (#1426/#1477).",
            dp.edf
        );

        if dp_rmse > sp_rmse {
            dp_worse += 1;
        }
    }

    // The double penalty must not be the systematically-worse penalty model: the
    // pre-fix bug had it worse on (nearly) every seed.
    assert!(
        dp_worse <= seeds.len() / 2,
        "Tweedie double-penalty ps RMSE-to-truth is worse than single-penalty on \
         {dp_worse}/{} seeds — the non-Gaussian null-space over-shrink (#1477) persists.",
        seeds.len()
    );
}
