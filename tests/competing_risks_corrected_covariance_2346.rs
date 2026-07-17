//! #2346 acceptance witness — a unified competing-risks (cause-specific) fit
//! must carry the smoothing-corrected joint coefficient covariance with typed
//! provenance, so interval requests at the DEFAULT covariance mode
//! (`SmoothingCorrected`, which per the #2296 provenance contract never falls
//! back) stop hard-erroring on competing-risks models.
//!
//! The fit-side chain under test (gam-custom-family):
//! `joint_smoothing_correction` builds the first-order ρ-uncertainty inflation
//! `C = A·V_ρ·Aᵀ` (`A = V_cond·U`, `U[:,o] = (∂S_λ/∂ρ_o)·β̂`) from the SAME
//! analytic outer ρ-Hessian the criterion certificate judged, excludes rail
//! coordinates (#2337 Thm 2.3), lifts `C` through the identifiability gauge
//! alongside the conditional covariance, and publishes
//! `beta_covariance_corrected = V_cond + C` with
//! `FirstOrderIdentifiedSubspace` provenance on the fit inference.
//!
//! Assertions:
//! 1. presence — the corrected covariance and its typed method are on the fit;
//! 2. shape — corrected matches the conditional covariance dimensions
//!    (cross-cause blocks retained, nothing collapsed per-cause);
//! 3. inflation — `C = V_c − V_cond` is the PSD ρ-uncertainty term, so every
//!    diagonal of the corrected matrix is ≥ the conditional diagonal (within
//!    roundoff) and at least one strictly grows (the smoothing parameters of a
//!    REML fit on finite data are not known exactly);
//! 4. symmetry + finiteness of the corrected matrix.

use csv::StringRecord;
use gam::{FitConfig, FitResult, encode_recordswith_inferred_schema, fit_from_formula};

/// Deterministic SplitMix64 → byte-identical data run-to-run (no external RNG).
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
    fn unit(&mut self) -> f64 {
        ((self.next_u64() >> 11) as f64 + 0.5) / (1u64 << 53) as f64
    }
}

#[test]
fn competing_risks_fit_carries_smoothing_corrected_covariance_2346() {
    // Two-cause competing-risks data with asymmetric cause-specific hazards
    // (cause 1 rises with x, cause 2 falls) — the same generator shape as the
    // #1593 invariance guard, smaller n for a CI-affordable REML solve.
    let n = 600usize;
    let mut rng = SplitMix64::new(2346);
    let horizon = 6.0_f64;
    let mut rows = Vec::with_capacity(n);
    for _ in 0..n {
        let xi = rng.unit();
        let rate_a = 0.18 * (0.9 * (xi - 0.5)).exp();
        let rate_b = 0.18 * (-0.9 * (xi - 0.5)).exp();
        let t_a = -(rng.unit().ln()) / rate_a;
        let t_b = -(rng.unit().ln()) / rate_b;
        let (t_event, cause) = if t_a <= t_b { (t_a, 1.0) } else { (t_b, 2.0) };
        let (t_obs, code) = if t_event > horizon {
            (horizon, 0.0)
        } else {
            (t_event, cause)
        };
        rows.push(StringRecord::from(vec![
            t_obs.to_string(),
            code.to_string(),
            xi.to_string(),
        ]));
    }
    let headers = vec!["time".to_string(), "event".to_string(), "x".to_string()];
    let data =
        encode_recordswith_inferred_schema(headers, rows).expect("encode competing-risks data");

    let cfg = FitConfig {
        survival_likelihood: Some("weibull".to_string()),
        ..FitConfig::default()
    };
    let result = fit_from_formula("Surv(time, event) ~ s(x, bs='tp')", &data, &cfg)
        .expect("unified competing-risks Weibull fit");
    let FitResult::SurvivalTransformation(fit) = result else {
        panic!("expected a SurvivalTransformation fit for the unified competing-risks model");
    };

    // (1) Presence: corrected covariance + typed provenance.
    let conditional = fit
        .fit
        .covariance_conditional
        .as_ref()
        .expect("#2346: a converged competing-risks REML fit carries the conditional covariance");
    let corrected = fit.fit.beta_covariance_corrected().expect(
        "#2346: a converged competing-risks REML fit must carry the smoothing-corrected \
         covariance so DEFAULT-mode interval requests stop hard-erroring",
    );
    let method = fit
        .fit
        .smoothing_correction_method()
        .expect("#2346: the corrected covariance must carry its typed method provenance");
    match method {
        gam::solver::model_types::SmoothingCorrectionMethod::FirstOrderIdentifiedSubspace {
            active_rank,
            rho_dimension,
        } => {
            assert!(
                active_rank >= 1 && active_rank <= rho_dimension,
                "#2346: identified interior rank {active_rank} must be within the \
                 rho dimension {rho_dimension}",
            );
        }
        other => panic!(
            "#2346: expected FirstOrderIdentifiedSubspace provenance, got {other:?} — \
             a named approximation must not be reported as the exact first-order channel"
        ),
    }

    // (2) Shape: the corrected matrix lives on the SAME stacked coefficient
    // space (cross-cause blocks retained).
    assert_eq!(
        corrected.dim(),
        conditional.dim(),
        "#2346: corrected covariance must match the conditional stacked dimensions"
    );

    // (3) The correction C = V_c − V_cond is PSD, so corrected diagonals never
    // shrink; and REML smoothing parameters on finite data carry genuine
    // uncertainty, so at least one variance strictly grows.
    let p = corrected.nrows();
    let mut any_strict_growth = false;
    for i in 0..p {
        let vc = corrected[[i, i]];
        let v0 = conditional[[i, i]];
        assert!(
            vc >= v0 - 1e-10 * (1.0 + v0.abs()),
            "#2346: corrected variance must not shrink below conditional at \
             coordinate {i}: corrected {vc:.6e} vs conditional {v0:.6e}"
        );
        if vc > v0 * (1.0 + 1e-9) + 1e-14 {
            any_strict_growth = true;
        }
    }
    assert!(
        any_strict_growth,
        "#2346: the rho-uncertainty inflation must strictly widen at least one \
         coefficient variance (a zero correction means the channel is dead)"
    );

    // (4) Symmetry + finiteness.
    for i in 0..p {
        for j in 0..p {
            let v = corrected[[i, j]];
            assert!(v.is_finite(), "#2346: corrected[{i},{j}] must be finite");
            let asym = (v - corrected[[j, i]]).abs();
            assert!(
                asym <= 1e-9 * (1.0 + v.abs()),
                "#2346: corrected covariance must be symmetric at [{i},{j}]"
            );
        }
    }
}
