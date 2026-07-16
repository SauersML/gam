//! Regression repro for #2299: the dense outer-REML plateau stall on a
//! near-linear `y ~ smooth(x)` Gaussian fit with a nonzero model offset.
//!
//! ROOT CAUSE (see the #2299 entrance-diff investigation): the Python and CLI
//! entrances build a byte-identical `StandardFitRequest` and both reach the same
//! dense `fit_model`; there is NO entrance-configuration divergence. The observed
//! "Python stalls at 200 iters, CLI converges in 5" was CSV-round-trip input
//! perturbation landing in different basins of a near-flat REML surface.
//!
//! The surface is near-flat because the fixture's signal (`0.6 + 1.2·x`, plus a
//! small offset the model subtracts exactly) lies in the smooth's polynomial
//! NULL SPACE `{1, x}`. The REML criterion is then asymptotically flat in `ρ` as
//! `λ → ∞` (the range-space EDF → 0), so the bending-penalty coordinate rails at
//! the `+rho_bound` infinite-smoothing ceiling — locus line checkpoint `ρ ≈
//! [−7.73, 29.994]` at the `+30` rail, `hessian_psd=NO` — and the projected
//! gradient plateaus at `|Pg| ≈ 1.1e-1`, far above the n-scaled stationarity
//! bound, until the standard-REML general outer engine
//! (`gam-solve rho_optimizer/run.rs`, `certify_outer_optimality`) exhausts its
//! 200-iteration cap.
//!
//! FIX (landed): the outer certificate's large-step flatness certificate — which
//! removes a saturated railed coordinate from `|Pg|` — was gated on the FULL
//! Hessian being PSD, but a coordinate railed at the infinite-smoothing ceiling
//! makes the full Hessian indefinite along its own flat direction, so the
//! certificate that exists to handle rails was disabled by the rail. The box-KKT
//! reduced-Hessian gate (`certificate_hessian_is_psd_off_railed`) judges PSD on
//! the INTERIOR (un-railed) sub-block for both the flatness-certificate gate and
//! the final curvature verdict, so the railed coordinate's outward pull is a KKT
//! multiplier (dropped from the residual) rather than a stationarity obstruction.
//! The fit certifies RAILED-CONVERGED; `lambdas_railed` stays reported, so the
//! verdict is converged-with-a-rail-flag, not clean. The fixture's TRUTH is the
//! null-space (linear) fit, so the rail is the correct answer, not a stall.
//! (Same pathology class as #1788,
//! [`super::gaussian_reml_stall_edf_collapse_1788_tests`], whose interior-optimum
//! fix does not cover the near-null-space rail case.)
//!
//! The top-level `gam` crate cannot build here (a `build.rs` author tripwire), so
//! the fixture is exercised through `fit_from_formula` in `gam-models`, which
//! builds standalone — mirroring the #1788 repro.

use super::entry::fit_from_formula;
use super::request::{FitConfig, FitResult, StandardFitResult};
use csv::StringRecord;
use gam_data::encode_recordswith_inferred_schema;
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand_distr::{Distribution, Normal, Uniform};

/// The #2299 fixture, structurally mirroring the Python repro
/// (`bug_hunt_2299_affine_design_link_wiggle_test.py`): a near-linear signal
/// `0.6 + 1.2·x` plus a small non-linear model offset `0.35·sin(1.7·x) − 0.1`,
/// low Gaussian noise, fit `y ~ smooth(x)` with the offset threaded as a model
/// offset. The exact numpy PCG64 stream is not reproducible under `StdRng`, but
/// the plateau pathology is a STRUCTURAL property of the near-null-space signal,
/// not of one seed — so a deterministic near-linear fixture reproduces it and is
/// a stabler regression pin than a single hard-coded draw.
fn fit_near_linear_with_offset(n: usize, seed: u64) -> Result<StandardFitResult, String> {
    let mut rng = StdRng::seed_from_u64(seed);
    let ux = Uniform::new(-1.5_f64, 1.5).expect("valid uniform range");
    let noise = Normal::new(0.0, 0.08).expect("valid normal");

    let headers: Vec<String> = ["x", "offset", "y"].iter().map(|s| s.to_string()).collect();
    let mut rows = Vec::with_capacity(n);
    for _ in 0..n {
        let x = ux.sample(&mut rng);
        let offset = 0.35 * (1.7 * x).sin() - 0.1;
        let y = 0.6 + 1.2 * x + offset + noise.sample(&mut rng);
        rows.push(StringRecord::from(vec![
            x.to_string(),
            offset.to_string(),
            y.to_string(),
        ]));
    }
    let ds = encode_recordswith_inferred_schema(headers, rows).expect("encode fixture");

    let cfg = FitConfig {
        family: Some("gaussian".to_string()),
        offset_column: Some("offset".to_string()),
        ..FitConfig::default()
    };
    // The nonzero offset routes this to the dense outer-REML path (the exact O(n)
    // spline scan bails on any nonzero offset), so this exercises exactly the
    // stalling surface — no explicit forcing needed.
    match fit_from_formula("y ~ smooth(x)", &ds, &cfg).map_err(|error| error.to_string())? {
        FitResult::Standard(standard) => Ok(standard),
        _ => Err("expected a Standard fit from a single-smooth Gaussian formula".to_string()),
    }
}

/// #2299 regression gate (upgraded for #2348 Inc 1): the near-linear `s(x)` +
/// offset fit MINTS a *typed* stationary-at-asymptote rail certificate instead of
/// grinding the outer REML to its iteration cap.
///
/// The bending penalty's REML optimum is at ρ→+∞ for a signal that lives in the
/// smooth's polynomial null space `{1, x}`, so it rails at the +`rho_bound`
/// infinite-smoothing ceiling. The asymptote-rail certificate
/// (`rho_optimizer/run.rs`, `asymptote_certificate.rs`, #2337 Thm 2.1)
/// reconstructs the coordinate's exponential tail, confirms the pencil constant
/// `ĉ = −e^{ρ}·∂V/∂ρ` is constant on a finite-difference-clean run, and proves
/// both the remaining criterion improvement (`value_gap = |∂V/∂ρ|`) and the
/// remaining coefficient travel to the rail limit are below tolerance. The fit
/// then converges via the typed `OuterStationarityCertificate::AsymptoteRail`
/// (not the untyped `lambdas_railed` flag), and the truth being the null-space
/// (linear) fit, the estimator is the honest answer.
#[test]
fn near_linear_offset_fit_converges_railed_off_the_infinite_smoothing_plateau_2299() {
    use gam_solve::estimate::OuterStationarityCertificate;

    let result = fit_near_linear_with_offset(160, 2299).expect(
        "#2299/#2348: the near-linear s(x)+offset fit must MINT a stationary-at-asymptote \
         rail certificate — the bending penalty rails at the +rho_bound ceiling and the \
         outer certificate positively certifies the confirmed exponential tail, instead \
         of grinding to the iteration cap",
    );
    // The plateau no longer grinds: a railed-converged fit lands well under the
    // 200-iteration outer cap.
    assert!(
        result.fit.outer_iterations < 200,
        "#2299: railed-converged fit must land under the outer iteration cap, got {} \
         iterations (still grinding the infinite-smoothing plateau)",
        result.fit.outer_iterations,
    );

    // The stationarity certificate is the TYPED asymptote rail, not the generic
    // gradient/criterion-flat verdict: the railed coordinate is positively
    // certified on its confirmed tail.
    let certificate = result
        .fit
        .convergence_evidence()
        .outer_certificate()
        .expect("#2299: a smoothing-optimized fit carries an analytic outer certificate");
    let rails = match &certificate.stationarity {
        OuterStationarityCertificate::AsymptoteRail {
            interior_projected_grad_norm,
            bound,
            rails,
        } => {
            assert!(
                interior_projected_grad_norm.is_finite() && interior_projected_grad_norm <= bound,
                "#2299: the interior (non-railed) projected gradient {interior_projected_grad_norm} \
                 must be stationary within the bound {bound}",
            );
            rails.clone()
        }
        other => panic!(
            "#2299/#2348: expected a typed AsymptoteRail stationarity certificate, got {other:?}"
        ),
    };
    assert!(
        !rails.is_empty(),
        "#2299: the AsymptoteRail certificate must carry at least one certified rail",
    );
    for rail in &rails {
        assert!(
            rail.tail_constant.is_finite() && rail.tail_constant > 0.0,
            "#2299: rail #{} must carry a positive pencil constant ĉ, got {}",
            rail.index,
            rail.tail_constant,
        );
        // The remaining criterion improvement to the rail (|∂V/∂ρ|) is below the
        // outer tolerance scale — the criterion has reached its asymptote.
        assert!(
            rail.value_gap.is_finite() && rail.value_gap < 1.0e-2,
            "#2299: rail #{} value_gap must be below tolerance, got {}",
            rail.index,
            rail.value_gap,
        );
        // The fitted coefficients have reached the rail limit: the bounded
        // remaining travel is a negligible fraction of the coefficient scale.
        assert!(
            rail.estimand_travel_bound.is_finite() && rail.estimand_travel_bound < 1.0e-2,
            "#2299: rail #{} estimand_travel_bound must be below tolerance, got {}",
            rail.index,
            rail.estimand_travel_bound,
        );
    }

    // The truth lives in the polynomial null space `{1, x}`, so the range-space EDF
    // is driven out and the reported EDF is finite and low — never NaN or the
    // algebraic ceiling.
    let edf_total = result
        .fit
        .edf_total()
        .expect("converged fit reports edf_total");
    assert!(
        edf_total.is_finite() && (0.5..8.0).contains(&edf_total),
        "#2299: a near-linear fit must report a finite, low null-space EDF, got \
         edf_total={edf_total}",
    );
}
