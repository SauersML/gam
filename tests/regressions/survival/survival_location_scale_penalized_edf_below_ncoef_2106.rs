//! Regression guard for #2106: a penalized survival (Royston-Parmar) location-scale
//! `summary()` must report the EFFECTIVE degrees of freedom `tr(F)`,
//! `F = (X'WX + S)⁻¹ X'WX`, NOT the nominal coefficient count.
//!
//! Before the fix the survival finalize path hardcoded every per-block penalty
//! trace to 0, so `edf_total` equaled the raw coefficient count and each smooth
//! term's EDF equaled its column count — unchanged no matter how heavily the
//! penalty shrank the fit. That inflated AIC / evidence and broke survival
//! `compare_models`. The Gaussian `noise_formula` location-scale sibling on the
//! same shape already reports `edf_total < ncoef`.
//!
//! Here the truth is a FLAT location and FLAT log-scale (no x dependence), so
//! REML drives the smoothing parameters large and the smooth terms shrink toward
//! their unpenalized null space. The effective EDF must therefore land strictly
//! below the coefficient count, and each smooth term's EDF strictly below its own
//! column count. With the bug the assertion `edf_total < ncoef` fails because
//! `edf_total == ncoef` exactly.

use gam::estimate::BlockRole;
use gam::{
    FitConfig, FitResult, encode_recordswith_inferred_schema, fit_from_formula, init_parallelism,
};

/// Numerical-Recipes 64-bit LCG → deterministic uniforms in [0,1).
struct Lcg {
    state: u64,
}
impl Lcg {
    fn new(seed: u64) -> Self {
        Self { state: seed }
    }
    fn unit(&mut self) -> f64 {
        self.state = self
            .state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((self.state >> 11) as f64) / ((1u64 << 53) as f64)
    }
    /// Box–Muller standard normal.
    fn normal(&mut self) -> f64 {
        let u1 = self.unit().max(1e-300);
        let u2 = self.unit();
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
    }
}

#[test]
fn survival_location_scale_penalized_edf_below_ncoef_2106() {
    init_parallelism();

    // Clean Gaussian AFT on log-time with a FLAT truth in x: log T = 0 + 1·ε.
    // The covariate x is pure noise, so REML shrinks both the location and the
    // log-σ smooth of x heavily.
    let n = 240usize;
    let k_loc = 8usize;
    let k_scale = 8usize;
    let mut rng = Lcg::new(20250703);

    let mut x = Vec::with_capacity(n);
    let mut exit = Vec::with_capacity(n);
    let mut event = Vec::with_capacity(n);
    for _ in 0..n {
        let xi = -2.0 + 4.0 * rng.unit();
        let log_t = rng.normal(); // flat μ = 0, flat log σ = 0
        let t = log_t.exp().max(1e-6);
        // Light censoring so the fit stays well-posed.
        let c = (1.5 + 3.0 * rng.unit()).exp();
        let (obs, ev) = if t <= c { (t, 1.0) } else { (c, 0.0) };
        x.push(xi);
        exit.push(obs);
        event.push(ev);
    }

    let headers: Vec<String> = ["entry", "exit", "event", "x"]
        .into_iter()
        .map(str::to_string)
        .collect();
    let rows: Vec<csv::StringRecord> = (0..n)
        .map(|i| {
            csv::StringRecord::from(vec![
                "0".to_string(),
                format!("{:.17e}", exit[i]),
                format!("{:.17e}", event[i]),
                format!("{:.17e}", x[i]),
            ])
        })
        .collect();
    let ds = encode_recordswith_inferred_schema(headers, rows).expect("encode edf data");

    let cfg = FitConfig {
        survival_likelihood: "location-scale".to_string(),
        survival_distribution: "gaussian".to_string(),
        noise_formula: Some(format!("s(x, k={k_scale})")),
        ..FitConfig::default()
    };
    let result = fit_from_formula(
        &format!("Surv(entry, exit, event) ~ s(x, k={k_loc})"),
        &ds,
        &cfg,
    )
    .expect("gam survival location-scale fit");
    let FitResult::SurvivalLocationScale(fit) = result else {
        panic!("expected a survival location-scale fit result");
    };
    let unified = &fit.fit.fit;

    let ncoef = unified.beta_flat().len();
    // The reported total effective d.f. is the sum of the per-block EDFs (what
    // `summary()` / the report builder aggregate). Before the fix every block's
    // EDF was its raw column count, so this equals `ncoef` exactly.
    let edf_total: f64 = unified.blocks.iter().map(|b| b.edf).sum();

    for b in &unified.blocks {
        eprintln!(
            "[#2106] block {:?} ncoef={} edf={:.4} lambdas={:?}",
            b.role,
            b.beta.len(),
            b.edf,
            b.lambdas.to_vec()
        );
    }
    eprintln!("[#2106] ncoef={ncoef} edf_total={edf_total:.4}");

    assert!(
        edf_total.is_finite() && edf_total > 0.0,
        "#2106: edf_total must be a finite positive effective d.f., got {edf_total}"
    );

    // Core regression: the effective d.f. must be STRICTLY below the nominal
    // coefficient count once a positive-rank penalty is active. Before the fix
    // this equals `ncoef` exactly (penalty traces hardcoded to 0).
    assert!(
        edf_total < ncoef as f64 - 1.0,
        "#2106: effective edf_total ({edf_total:.4}) must be meaningfully below the coefficient \
         count ({ncoef}); a hardcoded nominal count would report edf_total == ncoef"
    );

    // Each penalized smooth term (location + scale) must also spend fewer than
    // its full column count of degrees of freedom.
    for block in &unified.blocks {
        if matches!(block.role, BlockRole::Threshold | BlockRole::Scale) {
            let cols = block.beta.len();
            // Only assert when the block actually carries a penalty.
            if !block.lambdas.is_empty() {
                assert!(
                    block.edf < cols as f64 - 0.25,
                    "#2106: penalized {:?} smooth EDF ({:.4}) must be below its column count ({cols})",
                    block.role,
                    block.edf,
                );
                assert!(
                    block.edf > 0.0 && block.edf.is_finite(),
                    "#2106: {:?} smooth EDF must be finite and positive, got {}",
                    block.role,
                    block.edf,
                );
            }
        }
    }
}
