//! Regression + diagnostic for #1426: a Gamma/log REML fit must not SILENTLY
//! ship a non-converged near-full-basis overfit. Under SPEC 20 the old
//! boolean verdict is unrepresentable: a `UnifiedFitResult` that exists IS the
//! sealed convergence certificate, and a fit that cannot certify returns a
//! typed error (which would trip the `expect` below). What remains testable —
//! and what this guards end-to-end — is that every minted fit is genuinely
//! stationary (small `outer_gradient_norm`, not the 70271226e fake-zero
//! fast-path overfit at projected |g|≈10.9) and is NOT a near-unpenalized
//! near-full-basis overfit (EDF well below the full basis).
//!
//! DGP (the issue's): N=1500; tm(x)=exp(0.6·sin(2πx)+0.6); x~U(0,1);
//! y~Gamma(shape=2, scale=tm(x)/2). Deterministic SplitMix64 (no Python).

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
    fn next_unit(&mut self) -> f64 {
        let bits = self.next_u64() >> 11; // 53 bits
        ((bits as f64) + 0.5) / (1u64 << 53) as f64
    }
    fn next_exp1(&mut self) -> f64 {
        -(1.0 - self.next_unit()).ln()
    }
    /// Gamma(shape=2, scale) = sum of two Exponential(scale) draws.
    fn next_gamma_shape2(&mut self, scale: f64) -> f64 {
        scale * (self.next_exp1() + self.next_exp1())
    }
}

const N: usize = 1500;

fn build_data(seed: u64) -> gam::data::EncodedDataset {
    let mut rng = SplitMix64::new(seed);
    let truth_mean = |x: f64| ((0.6 * (2.0 * std::f64::consts::PI * x).sin()) + 0.6).exp();
    let mut x = Vec::with_capacity(N);
    let mut y = Vec::with_capacity(N);
    for _ in 0..N {
        let xi = rng.next_unit();
        let yi = rng.next_gamma_shape2(truth_mean(xi) / 2.0);
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

struct FitProbe {
    edf: f64,
    grad_norm: Option<f64>,
}

fn fit_probe(data: &gam::data::EncodedDataset) -> FitProbe {
    let cfg = FitConfig {
        family: Some("gamma".to_string()),
        link: Some("log".to_string()),
        ..FitConfig::default()
    };
    let result = fit_from_formula("y ~ s(x)", data, &cfg).expect("gamma/log gam fit");
    let FitResult::Standard(fit) = result else {
        panic!("expected a standard GAM fit");
    };
    FitProbe {
        edf: fit.fit.edf_total().expect("gam reports total edf"),
        grad_norm: fit.fit.outer_gradient_norm,
    }
}

/// The full default basis is ~24; a well-penalized recovery lands near EDF ~8-9.
/// A `converged == true` verdict must not coincide with EDF anywhere near the
/// full basis: that is precisely the silent overfit #1426 forbids.
const FULL_BASIS_OVERFIT_FLOOR: f64 = 18.0;

/// Outer gradient tolerance band: a certified fit must have a small projected
/// outer gradient. mgcv-equivalent stationary REML fits land orders of
/// magnitude below this on this DGP; the broken overfit carried |g|≈10.9.
const CONVERGED_GRAD_CEILING: f64 = 1.0;

/// Deterministic seed block, anchored at the issue's reported failing seed and
/// covering the ~5-8% pathology rate so at least one seed historically shipped
/// the non-converged near-full-basis overfit.
const SCAN_SEEDS: &[u64] = &[
    770013, 770014, 770015, 770016, 770017, 770018, 770019, 770020, 770021, 770022, 900000, 900001,
    900002, 900003, 900004, 900005, 900006, 900007, 900008, 900009,
];

/// Every seed's fit must mint (fit existence is the sealed convergence proof,
/// SPEC 20 — a genuinely non-stationary run returns a typed error and trips the
/// `expect` in `fit_probe`), must be genuinely stationary (small outer
/// gradient), and must NOT be a near-full-basis overfit. This is the end-to-end
/// public-contract gate for #1426 — the silent overfit can no longer ship
/// mislabelled; it must not ship at all.
#[test]
fn gamma_log_convergence_verdict_is_honest_not_silent_overfit() {
    let mut min_edf = f64::INFINITY;
    for &seed in SCAN_SEEDS {
        let data = build_data(seed);
        let p = fit_probe(&data);
        eprintln!(
            "[#1426] seed={seed} edf_total={:.3} converged=certified grad_norm={:?}",
            p.edf, p.grad_norm
        );
        min_edf = min_edf.min(p.edf);

        // (1) A certified fit must be genuinely stationary: its reported
        //     outer gradient must clear the convergence band. A certificate
        //     with a large residual gradient is the fake-zero mislabelling
        //     (#1426, the 70271226e fast-path bug).
        if let Some(g) = p.grad_norm {
            assert!(
                g.is_finite() && g <= CONVERGED_GRAD_CEILING,
                "seed {seed}: certified fit but outer_gradient_norm={g:.4e} > \
                 {CONVERGED_GRAD_CEILING} — a non-stationary point certified as converged \
                 (silent overfit, #1426)."
            );
        }

        // (2) A certified fit must NOT be a near-full-basis overfit: the
        //     silent-overfit ship rails EDF toward the full basis (~24).
        assert!(
            p.edf < FULL_BASIS_OVERFIT_FLOOR,
            "seed {seed}: certified fit with EDF={:.3} (>= {FULL_BASIS_OVERFIT_FLOOR}, \
             near the full basis ~24) — a near-unpenalized overfit certified as a converged \
             optimum (silent overfit, #1426). A correct Gamma/log REML recovers EDF ~8-9.",
            p.edf
        );
    }

    eprintln!(
        "[#1426] {}/{} seeds minted certified fits; min EDF = {:.3}",
        SCAN_SEEDS.len(),
        SCAN_SEEDS.len(),
        min_edf
    );
}
