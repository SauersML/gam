//! E2E gate: the measure-jet smooth (`mjs(...)`, canonical `type=measurejet`)
//! works as the Bernoulli marginal-slope (BMS) surface backend, exactly like
//! the duchon/matern backbones do.
//!
//! Pattern mirrored from `tests/marginal_slope_neyman_orthogonal_reference.rs`
//! (the shipped `fit_from_formula` + `FitConfig{family, logslope_formula,
//! z_column}` arm, its SplitMix64 data law, and its Sim-B truth-recovery
//! bound): a known smooth marginal surface alpha(x1, x2) plus a planted
//! logslope surface beta(x1) = 0.2 + 0.9*x1 drive a single principled
//! Bernoulli-probit draw per row on the latent driver z ~ N(0,1), which is
//! handed to Stage 2 directly as the `z` column. Both the marginal and the
//! logslope surfaces are declared as `mjs(x1, x2, ...)`.
//!
//! Assertions, in order of importance:
//!   (a) the fit SUCCEEDS through the standard `fit_from_formula` entry point
//!       and returns the BernoulliMarginalSlope variant with finite
//!       coefficients and finite smoothing parameters;
//!   (b) the recovered logslope surface tracks the planted truth: Pearson
//!       correlation of beta_hat(grid) with beta_true(grid) > 0.8 — the exact
//!       truth-recovery bound the template's Sim B asserts for its own
//!       logslope readout (read off `logslopespec_resolved` + block 1 +
//!       `baseline_logslope`, the same reconstruction the family uses);
//!   (c) the measure-jet diagnostic analogue of the template family's
//!       EDF/lambda-count checks: in spectral (per-level) mode each mjs term
//!       contributes one penalty candidate PER realized scale plus the
//!       double-penalty ridge, so each surface's design must carry exactly
//!       band+1 penalties, with the band read off the frozen quadrature the
//!       fit persisted onto the resolved spec.

use gam::families::bms::BernoulliMarginalSlopeFitResult;
use gam::terms::smooth::{SmoothBasisSpec, TermCollectionSpec, build_term_collection_design};
use gam::test_support::reference::pearson;
use gam::{FitConfig, FitResult, encode_recordswith_inferred_schema, fit_from_formula};
use ndarray::Array2;

/// Both surfaces share one declaration so the marginal and logslope blocks
/// exercise the identical measure-jet backbone. `scales=3` pins the realized
/// band (the unit square is far from degenerate), `centers=16` keeps the
/// quadrature modest at n=1500, and the default tau (1e-3 > 0) keeps the
/// term's (alpha, ln-tau) geometry dials enrolled in the outer psi engine —
/// the full BMS hyper path, not a dial-frozen shortcut.
const MJS_SURFACE: &str = "mjs(x1, x2, centers=16, scales=3)";

// ---------------------------------------------------------------------------
// Deterministic RNG — SplitMix64, copied from the template so every platform
// draws the identical data without pulling an RNG crate into the test.
// ---------------------------------------------------------------------------
struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// Uniform on (0, 1).
    fn next_unit(&mut self) -> f64 {
        ((self.next_u64() >> 11) as f64 + 0.5) / (1u64 << 53) as f64
    }

    /// Standard normal via Box-Muller (one of the pair, regenerated per call
    /// so downstream consumption order does not perturb the stream).
    fn next_normal(&mut self) -> f64 {
        let u1 = self.next_unit().max(1.0e-300);
        let u2 = self.next_unit();
        (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
    }
}

fn normal_cdf(x: f64) -> f64 {
    0.5 * (1.0 + statrs::function::erf::erf(x / std::f64::consts::SQRT_2))
}

/// Planted logslope truth: monotone in x1, flat in x2 — the template Sim B's
/// slope law lifted onto the 2-covariate surface.
fn beta_true(x1: f64) -> f64 {
    0.2 + 0.9 * x1
}

/// Planted marginal surface: smooth in both ambient coordinates.
fn alpha_true(x1: f64, x2: f64) -> f64 {
    -0.2 + 0.7 * (std::f64::consts::PI * x1).sin() + 0.3 * (std::f64::consts::PI * x2).cos()
}

/// In-memory dataset with columns `x1`, `x2`, `y`, `z` — the covariates first
/// so the resolved specs' feature columns (0, 1) line up with the 2-column
/// grid arrays used for the logslope readout, mirroring the template.
fn build_dataset(x1: &[f64], x2: &[f64], y: &[f64], z: &[f64]) -> gam::data::EncodedDataset {
    let n = x1.len();
    assert_eq!(x2.len(), n, "x1/x2 length mismatch");
    assert_eq!(y.len(), n, "x1/y length mismatch");
    assert_eq!(z.len(), n, "x1/z length mismatch");
    let headers = vec![
        "x1".to_string(),
        "x2".to_string(),
        "y".to_string(),
        "z".to_string(),
    ];
    let records: Vec<csv::StringRecord> = (0..n)
        .map(|i| {
            csv::StringRecord::from(vec![
                format!("{:.17e}", x1[i]),
                format!("{:.17e}", x2[i]),
                format!("{:.17e}", y[i]),
                format!("{:.17e}", z[i]),
            ])
        })
        .collect();
    encode_recordswith_inferred_schema(headers, records).expect("encode mjs BMS dataset")
}

/// Evaluate the fitted logslope surface beta_hat at grid points, reconstructed
/// exactly as the family does (block 1 = logslope, `bms/block_specs.rs`):
/// `beta_hat(x_i) = baseline_logslope + design(x_i) . beta_logslope`. Same
/// readout as the template's `beta_of_x`, widened to two ambient coordinates.
fn logslope_surface(fit: &BernoulliMarginalSlopeFitResult, grid: &[(f64, f64)]) -> Vec<f64> {
    let n = grid.len();
    let mut data = Array2::<f64>::zeros((n, 2));
    for (i, &(g1, g2)) in grid.iter().enumerate() {
        data[[i, 0]] = g1;
        data[[i, 1]] = g2;
    }
    let design = build_term_collection_design(data.view(), &fit.logslopespec_resolved)
        .expect("rebuild mjs logslope design from the frozen resolved spec");
    let dense = design.design.to_dense();
    let beta_logslope = &fit.fit.blocks[1].beta;
    assert_eq!(
        dense.ncols(),
        beta_logslope.len(),
        "mjs logslope design width {} != logslope beta length {}",
        dense.ncols(),
        beta_logslope.len()
    );
    let mut out = vec![0.0; n];
    for i in 0..n {
        let mut acc = fit.baseline_logslope;
        for j in 0..dense.ncols() {
            acc += dense[[i, j]] * beta_logslope[j];
        }
        out[i] = acc;
    }
    out
}

/// Realized scale-band length of the (single) mjs term in a resolved surface
/// spec, read off the frozen fit-time quadrature the freeze step persisted.
fn mjs_band_len(spec: &TermCollectionSpec, what: &str) -> usize {
    let mj = spec
        .smooth_terms
        .iter()
        .find_map(|term| match &term.basis {
            SmoothBasisSpec::MeasureJet { spec, .. } => Some(spec),
            _ => None,
        })
        .unwrap_or_else(|| panic!("{what} resolved spec must carry the mjs term"));
    let frozen = mj.frozen_quadrature.as_ref().unwrap_or_else(|| {
        panic!("{what} resolved mjs term must be frozen with its fit-time quadrature")
    });
    assert!(
        !frozen.eps_band.is_empty(),
        "{what} frozen mjs quadrature must carry a non-empty scale band"
    );
    frozen.eps_band.len()
}

#[test]
fn bms_marginal_slope_accepts_measure_jet_backbone() {
    gam::init_parallelism();
    const N: usize = 1500;

    // Covariates, latent driver, and a single principled Bernoulli draw per
    // row from the generative law eta = alpha(x1,x2) + beta(x1)*z — the
    // template Sim B's construction with the surface lifted to 2 covariates.
    let mut rng = SplitMix64::new(0x315A_2026_0612_0001);
    let mut x1 = vec![0.0; N];
    let mut x2 = vec![0.0; N];
    let mut z = vec![0.0; N];
    for i in 0..N {
        x1[i] = rng.next_unit();
        x2[i] = rng.next_unit();
        z[i] = rng.next_normal();
    }
    let mut rng_y = SplitMix64::new(0x315A_2026_0612_0002);
    let mut y = vec![0.0; N];
    for i in 0..N {
        let eta = alpha_true(x1[i], x2[i]) + beta_true(x1[i]) * z[i];
        let p = normal_cdf(eta).clamp(1e-9, 1.0 - 1e-9);
        y[i] = if rng_y.next_unit() < p { 1.0 } else { 0.0 };
    }

    let data = build_dataset(&x1, &x2, &y, &z);
    let config = FitConfig {
        family: Some("bernoulli-marginal-slope".to_string()),
        logslope_formula: Some(MJS_SURFACE.to_string()),
        z_column: Some("z".to_string()),
        ..FitConfig::default()
    };

    // (a) The fit must SUCCEED through the standard entry point with the
    // measure-jet backbone on BOTH surfaces, exactly as duchon/matern do.
    let result = fit_from_formula(&format!("y ~ {MJS_SURFACE}"), &data, &config)
        .expect("BMS fit with mjs marginal + mjs logslope surfaces must succeed");
    let out = match result {
        FitResult::BernoulliMarginalSlope(out) => out,
        _ => panic!("mjs-backed BMS fit returned the wrong family variant"),
    };
    assert!(
        out.fit.beta.iter().all(|coef| coef.is_finite()),
        "mjs-backed BMS fit should produce finite coefficients, got {:?}",
        out.fit.beta
    );
    assert!(
        out.fit.log_lambdas.iter().all(|rho| rho.is_finite()),
        "mjs-backed BMS fit must report finite smoothing parameters; got log_lambdas={:?}",
        out.fit.log_lambdas
    );

    // (b) Truth recovery: the fitted logslope surface must track the planted
    // increasing slope. Same readout and the same Pearson > 0.8 bound the
    // template's Sim B asserts for its logslope surface.
    let mut grid = Vec::with_capacity(49);
    for k1 in 0..7 {
        for k2 in 0..7 {
            grid.push((k1 as f64 / 6.0, k2 as f64 / 6.0));
        }
    }
    let beta_hat = logslope_surface(&out, &grid);
    let truth: Vec<f64> = grid.iter().map(|&(g1, _)| beta_true(g1)).collect();
    let corr = pearson(&beta_hat, &truth);
    assert!(
        corr > 0.8,
        "mjs-backed BMS logslope surface failed to track the planted increasing \
         slope beta(x1) = 0.2 + 0.9*x1 (Pearson {corr:.3} <= 0.8); the measure-jet \
         backbone is not recovering the marginal-slope signal"
    );

    // (c) Measure-jet diagnostic: with `centers=16` and no `multiscale` opt-in
    // (#1116) the mjs term resolves to single-scale mode — exactly ONE fused
    // penalty per surface (the jet-energy with the affine-preserving nullspace
    // ridge folded in at a fixed identifiability fraction, not a 2nd λ), the
    // one-λ Duchon/Matérn footprint (#1039). The per-scale spectral split is
    // reserved for the explicit multiscale opt-in where the spectrum is
    // identifiable; opting it in here would only inflate the marginal-slope
    // family's O(n) per-evaluation cost for no benefit. The band is still
    // realized (frozen quadrature non-empty), it just feeds one fused penalty.
    assert!(
        mjs_band_len(&out.marginalspec_resolved, "marginal") >= 1,
        "marginal mjs surface must still realize a scale band"
    );
    assert_eq!(
        out.marginal_design.penalties.len(),
        1,
        "small-centers mjs surface must be single-scale: one fused penalty (ridge folded), got {}",
        out.marginal_design.penalties.len()
    );
    assert_eq!(
        out.logslope_design.penalties.len(),
        1,
        "small-centers logslope mjs surface must be single-scale: one fused penalty (ridge folded), got {}",
        out.logslope_design.penalties.len()
    );
}
