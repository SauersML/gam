//! Regression for issue #1426: gamma/log default-k REML flat-valley overfit.
//!
//! On `family="gamma", link="log"` with the DEFAULT basis (k=24), the outer
//! REML smoothing-parameter optimizer fails to converge on a small fraction of
//! datasets and SHIPS a near-unpenalized overfit: the total EDF rails to the
//! full basis (~24) when the correct EDF is ~8-9 (mgcv converges to EDF ~8 on
//! identical data). The engine logs `NON-CONVERGED` internally but nothing
//! surfaces through the public fit, so the user silently gets a wrong point
//! estimate.
//!
//! Root cause: the inner PIRLS hits its iteration cap at the under-penalized
//! (λ→0) ridge, leaving the cached cost and the analytic gradient inconsistent.
//! The outer optimizer's flat-valley / "near-separable" cost-stall guard then
//! mis-accepts that ρ as the answer even though the projected gradient is far
//! above tolerance and the candidate λ is at the under-penalized extreme (EDF
//! near the full basis). The optimizer must NOT ship that point: it must recover
//! to the well-penalized converged optimum.
//!
//! DGP (from the issue), driven through the public `fit_from_formula` path:
//!   N=1500; tm(x) = exp(0.6*sin(2πx) + 0.6)
//!   x ~ Uniform(0,1); y ~ Gamma(shape=2.0, scale=tm(x)/2.0)
//!   fit  y ~ s(x)  family=gamma  link=log  (default k)
//!
//! Broken (pre-fix) seeds 770013 / 900005 ship EDF ≈ 24; the correct EDF is
//! ~8-9. Healthy seed 900000 already lands at EDF ~9.4 (control). The RNG is a
//! deterministic SplitMix64 so the test is reproducible with no Python.

use csv::StringRecord;
use gam::{FitConfig, FitResult, encode_recordswith_inferred_schema, fit_from_formula};

/// Deterministic SplitMix64 stream of uniforms in (0, 1).
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

    /// Uniform in (0, 1), 53-bit mantissa, nudged off the open endpoints so an
    /// inverse-CDF `-ln(1-u)` exponential draw stays finite.
    fn next_unit(&mut self) -> f64 {
        let bits = self.next_u64() >> 11; // 53 bits
        ((bits as f64) + 0.5) / (1u64 << 53) as f64
    }

    /// One Exponential(1) draw via inverse CDF.
    fn next_exp1(&mut self) -> f64 {
        -(1.0 - self.next_unit()).ln()
    }

    /// One Gamma(shape=2, scale) draw: sum of two Exponential(scale) draws.
    /// (Exact for integer shape: Gamma(k, θ) = Σ_{i=1..k} Exp(θ).)
    fn next_gamma_shape2(&mut self, scale: f64) -> f64 {
        scale * (self.next_exp1() + self.next_exp1())
    }
}

/// Build the issue's gamma/log DGP at `seed`, returning the encoded dataset.
fn build_data(seed: u64) -> gam::data::EncodedDataset {
    let n = 1500usize;
    let mut rng = SplitMix64::new(seed);

    let truth_mean = |x: f64| ((0.6 * (2.0 * std::f64::consts::PI * x).sin()) + 0.6).exp();

    let mut x = Vec::with_capacity(n);
    let mut y = Vec::with_capacity(n);
    for _ in 0..n {
        let xi = rng.next_unit();
        let mu = truth_mean(xi);
        // Gamma(shape=2, scale=mu/2) has mean mu and CV = 1/sqrt(2).
        let yi = rng.next_gamma_shape2(mu / 2.0);
        assert!(
            yi.is_finite() && yi > 0.0,
            "constructed y must be positive finite"
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
    encode_recordswith_inferred_schema(headers, rows).expect("encode")
}

/// Fit `y ~ s(x)` with default k, family=gamma, link=log, returning total EDF.
fn fit_edf(data: &gam::data::EncodedDataset) -> f64 {
    let cfg = FitConfig {
        family: Some("gamma".to_string()),
        link: Some("log".to_string()),
        ..FitConfig::default()
    };
    let result = fit_from_formula("y ~ s(x)", data, &cfg).expect("gamma/log gam fit");
    let FitResult::Standard(fit) = result else {
        panic!("expected a standard GAM fit");
    };
    fit.fit.edf_total().expect("gam reports total edf")
}

/// The under-penalized overfit rails EDF to the full default basis (~24). The
/// correct (mgcv-converged) EDF is ~8-9. Assert EDF <= 12 so the broken
/// near-full-basis ship fails and the recovered well-penalized fit passes. The
/// bound is deliberately well below the broken ~24 and above the correct ~9 so
/// it cannot be satisfied by the overfit.
const EDF_OVERFIT_CEILING: f64 = 12.0;

/// Healthy (well-penalized) band for this DGP+basis: mgcv lands at EDF ~8-9; the
/// fit must converge into this band on benign seeds.
const HEALTHY_LO: f64 = 6.0;
const HEALTHY_HI: f64 = 12.0;

/// Deterministic seed block scanned for the overfit invariant. The issue reports
/// the pathology on ~5-8% of datasets, so a block of this size reliably contains
/// at least one seed that, BEFORE the fix, ships the non-converged near-full-
/// basis overfit (EDF ≈ 24) — which makes this test RED pre-fix and GREEN after.
/// The block is anchored at the issue's reported failing seed (770013) so the
/// reported case is covered even though the Rust RNG differs from the issue's
/// numpy draws.
const SCAN_SEEDS: &[u64] = &[
    770013, 770014, 770015, 770016, 770017, 770018, 770019, 770020, 770021, 770022, 900000, 900001,
    900002, 900003, 900004, 900005, 900006, 900007, 900008, 900009,
];

#[test]
fn gamma_log_default_k_never_ships_full_basis_overfit() {
    // Invariant: across the deterministic seed block, the gamma/log default-k
    // fit must NEVER ship a near-full-basis overfit. Before the fix, the outer
    // REML optimizer cost-stalls on the under-penalized (λ→0) ridge for a
    // fraction of these seeds and reports a NON-CONVERGED EDF ≈ 24 — this
    // assertion fails there. After the fix the heavier-seed multistart recovers
    // the well-penalized (mgcv-like) optimum and every seed stays <= ceiling.
    let mut max_edf = f64::NEG_INFINITY;
    let mut max_seed = 0u64;
    let mut any_healthy = false;
    for &seed in SCAN_SEEDS {
        let data = build_data(seed);
        let edf = fit_edf(&data);
        eprintln!("[#1426] seed={seed} edf_total={edf:.3}");
        if edf > max_edf {
            max_edf = edf;
            max_seed = seed;
        }
        if (HEALTHY_LO..=HEALTHY_HI).contains(&edf) {
            any_healthy = true;
        }
        assert!(
            edf <= EDF_OVERFIT_CEILING,
            "seed {seed}: total EDF {edf:.3} rails toward the full basis (~24); the gamma/log \
             REML fit shipped a near-unpenalized overfit instead of recovering the well-\
             penalized optimum (correct EDF ~8-9, like mgcv). Flat-valley-stall \
             mis-acceptance (#1426)."
        );
    }
    eprintln!("[#1426] scan max EDF = {max_edf:.3} at seed {max_seed}");
    // Control: the optimizer genuinely reaches the well-penalized basin on at
    // least one benign seed — a fix that simply forced EDF down everywhere
    // (e.g. by over-penalizing) would be caught by the healthy-band requirement.
    assert!(
        any_healthy,
        "no scanned seed produced a healthy converged EDF in [{HEALTHY_LO}, {HEALTHY_HI}]; the \
         optimizer is not reaching the well-penalized basin at all (mgcv lands at EDF ~8-9)."
    );
}
