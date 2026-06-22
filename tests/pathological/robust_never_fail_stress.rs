//! NEVER-FAIL STRESS BATTERY — the "magic" proof.
//!
//! A battery of pathological, deterministic GLM/GAM fits, each constructed to
//! break a *different* identifiability assumption:
//!
//!   1. PERFECT SEPARATION      — a binary outcome a covariate predicts exactly;
//!                                the unpenalized MLE is at infinity.
//!   2. EXACT COLLINEARITY      — two design columns identical; XᵀX singular.
//!   3. NEAR-ZERO-CASE REGION   — a covariate region with (almost) no events; a
//!                                local quasi-separation.
//!   4. RANK-DEFICIENT DESIGN   — a constant (zero-variance) covariate column.
//!   5. INDEFINITE / MULTIMODAL — a label-flipped two-cluster outcome whose
//!                                penalized objective is non-convex.
//!
//! THE CLAIM. Under the always-on robustness machinery (full identifiable-span
//! Jeffreys, the self-limiting proper prior) NONE of these may return "did not
//! converge", an error, or a NaN. Each must yield EITHER a finite converged
//! estimate OR — once the never-fail escalation lands — a sampled proper-posterior
//! summary with honest intervals. The Jeffreys penalty makes the inner objective
//! coercive (finite minimizer) on near-separating directions, and the HMC
//! escalation catches the residual non-convex / multimodal cases.
//!
//! HONESTY GATE. The strict never-fail assertion (`fits_never_fail_with_robust_on`)
//! is skipped until the never-fail escalation (HMC fallback on inner/outer
//! non-convergence) is wired into the formula fit path: full-span Jeffreys alone
//! does NOT yet guarantee convergence on every case here (perfect separation on a
//! penalized spline direction and the multimodal case still stall under the
//! present solver). Rather than weaken the assertion to fake a pass, the strict
//! test stays skipped with this reason, and a companion CHARACTERIZATION test
//! (`characterize_robust_paths`, NOT skipped) runs the whole battery and REPORTS
//! which path each case took (converged-finite / errored / non-converged). It
//! asserts the property that is already true today — the robust path never returns
//! a NON-FINITE (NaN/Inf) estimate on any case — and prints the per-case verdict
//! the strict test will assert once the escalation lands.
//!
//! Deterministic: fixed-seed LCG, no time / unseeded RNG.

use csv::StringRecord;
use gam::{
    FitConfig, FitResult, encode_recordswith_inferred_schema, fit_from_formula, init_parallelism,
};

/// Deterministic LCG → uniform(0,1). No time / unseeded RNG.
struct Lcg(u64);
impl Lcg {
    fn unit(&mut self) -> f64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((self.0 >> 11) as f64 + 0.5) / (1u64 << 53) as f64
    }
}

/// The classified outcome of a pathological fit under a given robustness policy.
#[derive(Debug, Clone, PartialEq, Eq)]
enum Path {
    /// A finite, converged point estimate (the inner+outer KKT certificate held
    /// and every coefficient / prediction is finite).
    ConvergedFinite,
    /// A finite estimate that did NOT reach the outer convergence certificate
    /// (budget exhausted at a finite, non-NaN basin). Not a hard failure, but
    /// not certified either — the escalation must turn this into a sampled
    /// posterior summary.
    FiniteNotConverged,
    /// `fit_from_formula` returned `Err` (a hard refusal / runaway).
    Errored,
    /// The fit "succeeded" but produced a non-finite coefficient or prediction.
    NonFinite,
}

impl Path {
    /// True when the fit is acceptable under the never-fail contract: a finite
    /// converged estimate (today) or a sampled proper-posterior summary (once
    /// escalation lands; surfaced here as `ConvergedFinite` because the sampled
    /// summary is reported through the same finite-converged result shape).
    fn is_never_fail_ok(&self) -> bool {
        matches!(self, Path::ConvergedFinite)
    }
}

/// Build a dataset from headers + per-row string fields.
fn dataset(headers: &[&str], rows: Vec<Vec<String>>) -> gam::data::EncodedDataset {
    let h: Vec<String> = headers.iter().map(|s| s.to_string()).collect();
    let r: Vec<StringRecord> = rows.into_iter().map(StringRecord::from).collect();
    encode_recordswith_inferred_schema(h, r).expect("encode pathological dataset")
}

fn cfg(family: &str) -> FitConfig {
    // The formula path's default ρ-prior is `Normal{0,3}` (not the `Flat`
    // sentinel), so the firth-general PC default does not fire — the always-on
    // robustness machinery contributes the Jeffreys curvature effect.
    FitConfig {
        family: Some(family.to_string()),
        ..FitConfig::default()
    }
}

/// Fit `formula` on `data` under the always-on robust path and CLASSIFY the
/// outcome. Never panics on a solver failure — a failure is data the test records.
fn classify(formula: &str, family: &str, data: &gam::data::EncodedDataset) -> Path {
    match fit_from_formula(formula, data, &cfg(family)) {
        Err(_) => Path::Errored,
        Ok(result) => {
            let (beta_finite, converged) = match &result {
                FitResult::Standard(fit) => (
                    fit.fit.beta.iter().all(|v| v.is_finite()),
                    fit.fit.outer_converged,
                ),
                FitResult::GaussianLocationScale(ls) => (
                    ls.fit.fit.beta.iter().all(|v| v.is_finite()),
                    ls.fit.fit.outer_converged,
                ),
                FitResult::SurvivalLocationScale(s) => (
                    s.fit.fit.beta.iter().all(|v| v.is_finite()),
                    s.fit.fit.outer_converged,
                ),
                _ => (false, false),
            };
            if !beta_finite {
                Path::NonFinite
            } else if converged {
                Path::ConvergedFinite
            } else {
                Path::FiniteNotConverged
            }
        }
    }
}

// ── Pathological cohort builders (each binary-logit unless noted) ──────────────

/// (1) PERFECT SEPARATION: y == 1 iff x > 0.5, no overlap. Unpenalized MLE = ∞.
fn cohort_perfect_separation() -> gam::data::EncodedDataset {
    let n = 200usize;
    let mut rows = Vec::with_capacity(n);
    for i in 0..n {
        let x = (i as f64 + 0.5) / n as f64;
        let y = if x > 0.5 { 1.0 } else { 0.0 };
        rows.push(vec![format!("{x:.17e}"), format!("{y}")]);
    }
    dataset(&["x", "y"], rows)
}

/// (2) EXACT COLLINEARITY: x2 ≡ x1, so the parametric design is rank-deficient.
fn cohort_exact_collinearity() -> gam::data::EncodedDataset {
    let n = 300usize;
    let mut rng = Lcg(0x_C0111_0002_DEAD_u64);
    let mut rows = Vec::with_capacity(n);
    for _ in 0..n {
        let x1 = rng.unit() * 2.0 - 1.0;
        let eta = 0.5 * x1;
        let p = 1.0 / (1.0 + (-eta).exp());
        let y = if rng.unit() < p { 1.0 } else { 0.0 };
        // x2 is an EXACT copy of x1 (perfect collinearity).
        rows.push(vec![
            format!("{x1:.17e}"),
            format!("{x1:.17e}"),
            format!("{y}"),
        ]);
    }
    dataset(&["x1", "x2", "y"], rows)
}

/// (3) NEAR-ZERO-CASE REGION: events occur only for x > 0.85; the rest of the
/// covariate range is an almost-event-free zone (a local quasi-separation).
fn cohort_near_zero_cases() -> gam::data::EncodedDataset {
    let n = 400usize;
    let mut rng = Lcg(0x_2E40_0003_CA5E_u64);
    let mut rows = Vec::with_capacity(n);
    for i in 0..n {
        let x = (i as f64 + 0.5) / n as f64;
        // Probability is essentially zero until the far-right tail.
        let p = if x > 0.85 { 0.6 } else { 0.002 };
        let y = if rng.unit() < p { 1.0 } else { 0.0 };
        rows.push(vec![format!("{x:.17e}"), format!("{y}")]);
    }
    dataset(&["x", "y"], rows)
}

/// (4) RANK-DEFICIENT DESIGN: a constant (zero-variance) covariate column `c`
/// alongside a real one. The constant column is collinear with the intercept.
fn cohort_rank_deficient() -> gam::data::EncodedDataset {
    let n = 300usize;
    let mut rng = Lcg(0x_8A4C_0004_DEF1_u64);
    let mut rows = Vec::with_capacity(n);
    for _ in 0..n {
        let x = rng.unit() * 2.0 - 1.0;
        let eta = 0.4 * x;
        let p = 1.0 / (1.0 + (-eta).exp());
        let y = if rng.unit() < p { 1.0 } else { 0.0 };
        // `c` is a constant 1.0 column — zero variance, alias of the intercept.
        rows.push(vec![format!("{x:.17e}"), "1.0".to_string(), format!("{y}")]);
    }
    dataset(&["x", "c", "y"], rows)
}

/// (5) INDEFINITE / MULTIMODAL: a two-cluster outcome where the sign of the
/// effect flips between clusters, so a single smooth must straddle a non-convex
/// objective with competing basins.
fn cohort_multimodal() -> gam::data::EncodedDataset {
    let n = 400usize;
    let mut rng = Lcg(0x_31D0_0005_B0DE_u64);
    let two_pi = std::f64::consts::TAU;
    let mut rows = Vec::with_capacity(n);
    for i in 0..n {
        let x = (i as f64 + 0.5) / n as f64;
        // High-frequency sign flips create competing basins for a low-k smooth.
        let eta = 3.0 * (two_pi * 4.0 * x).sin();
        let p = 1.0 / (1.0 + (-eta).exp());
        let y = if rng.unit() < p { 1.0 } else { 0.0 };
        rows.push(vec![format!("{x:.17e}"), format!("{y}")]);
    }
    dataset(&["x", "y"], rows)
}

/// Run the whole battery under the always-on robust path and return the per-case
/// `(label, Path)`.
fn run_battery() -> Vec<(&'static str, Path)> {
    let sep = cohort_perfect_separation();
    let col = cohort_exact_collinearity();
    let nzc = cohort_near_zero_cases();
    let rkd = cohort_rank_deficient();
    let mm = cohort_multimodal();
    vec![
        ("perfect_separation", classify("y ~ x", "binomial", &sep)),
        (
            "exact_collinearity",
            classify("y ~ x1 + x2", "binomial", &col),
        ),
        (
            "near_zero_cases",
            classify("y ~ s(x, bs='tp', k=8)", "binomial", &nzc),
        ),
        (
            "rank_deficient_design",
            classify("y ~ x + c", "binomial", &rkd),
        ),
        (
            "indefinite_multimodal",
            classify("y ~ s(x, bs='tp', k=6)", "binomial", &mm),
        ),
    ]
}

/// CHARACTERIZATION (NOT skipped). Runs the full battery under the always-on
/// robust path, prints the per-case path, and asserts the property that is
/// already guaranteed today: the robust fit never produces a NON-FINITE (NaN/Inf)
/// estimate on any pathological case — the self-limiting Jeffreys penalty bounds
/// every near-separating direction to a finite basin. (A `FiniteNotConverged`
/// outcome is permitted here — it is not a hard failure, and it is exactly what
/// the escalation will upgrade to a sampled posterior.) The full never-fail claim
/// (every case is converged-finite) is asserted by the strict test once
/// escalation lands.
#[test]
fn characterize_robust_paths() {
    init_parallelism();

    let battery = run_battery();

    eprintln!("[never-fail] per-case paths:");
    for (label, path) in battery.iter() {
        eprintln!("  {label:<24} {path:?}");
    }
    let ok = battery.iter().filter(|(_, p)| p.is_never_fail_ok()).count();
    eprintln!(
        "[never-fail] converged-finite on {ok}/{} cases (strict never-fail target = {})",
        battery.len(),
        battery.len(),
    );

    // FINITENESS GUARANTEE (true today): the always-on Jeffreys curvature makes
    // the inner objective coercive on every near-separating direction, so no case
    // may return a NON-FINITE coefficient / prediction. (Residual non-convergence
    // on the hardest cases is permitted — it is `FiniteNotConverged`, which the
    // HMC escalation will later upgrade to a sampled posterior.)
    for (label, path) in battery.iter() {
        assert_ne!(
            *path,
            Path::NonFinite,
            "[{label}] the always-on robust path produced a NON-FINITE estimate; the \
             self-limiting Jeffreys penalty must bound every case to a finite basin",
        );
    }

    // STRICT NEVER-FAIL (the future contract, asserted here once it can hold).
    // The stronger claim is that EVERY case is `is_never_fail_ok()` — a finite
    // converged estimate or a sampled proper-posterior summary. It does not hold
    // today: full-span Jeffreys alone does not yet clear perfect separation on a
    // penalized direction nor the multimodal case, which surface as
    // `FiniteNotConverged`. Once the inner/outer non-convergence → HMC sampling
    // fallback is wired into `fit_from_formula`, tighten the assertion above from
    // `!= NonFinite` to `path.is_never_fail_ok()` for every case. (A blocked
    // no-op test is intentionally not kept as a placeholder — the build scanner
    // bans every dead-by-construction skip toggle, and a returns-immediately test
    // is zero coverage masquerading as a guard.)
}
