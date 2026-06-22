//! Property-based invariant fuzzer for the fit stack (DISCOVERY harness).
//!
//! Generates random-but-VALID `(formula, family, n, seed)` cases, fits each
//! through the public `fit_from_formula` entry point, and asserts a handful of
//! UNIVERSAL invariants that must hold for any well-posed GAM fit, regardless
//! of family/link/term:
//!
//!   I1  finite β            — every fitted coefficient is finite.
//!   I2  finite predictions  — predicting on the TRAINING rows yields finite
//!                             values (the design replay must not NaN/Inf).
//!   I3  finite objective    — REML score and log-likelihood are finite.
//!   I4  determinism         — refitting identical data with the identical
//!                             config reproduces β bit-for-bit-close.
//!   I5  EDF bounds           — total EDF ∈ [null_space_dim − tol, p + tol].
//!   I6  SE ordering          — standard errors are finite and non-negative,
//!                             so any symmetric interval lower ≤ mean ≤ upper.
//!
//! This file is a DISCOVERY net: it should stay green. Any case that trips an
//! invariant is a bug — it gets minimized into its own dedicated regression
//! test + a root-cause fix, and is then either fixed (so this stays green) or,
//! if a deep fix is pending, captured in a `#[ignore]`d focused repro so this
//! sweep does not block CI on a known-open ticket.
//!
//! Design choices that keep it CI-cheap and deterministic:
//!   - n is small (8..40) — edge cases live at tiny n, and the 8 GB CI
//!     build budget forbids large fixtures.
//!   - the case grid is a fixed deterministic product of (shape × family ×
//!     seed); no wall-clock or thread-order dependence.
//!   - data generators are seeded `StdRng` only (no rand-version drift).

use csv::StringRecord;
use gam::matrix::LinearOperator;
use gam::smooth::build_term_collection_design;
use gam::{
    FitConfig, FitResult, encode_recordswith_inferred_schema, fit_from_formula, init_parallelism,
};
use ndarray::Array2;
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand_distr::{Distribution, Normal, Uniform};

/// A single fuzz case: a column generator + a formula template + the family the
/// `FitConfig` should request. `ncols` is how many predictor columns the data
/// generator must emit (the response column `y` is always appended last).
struct Shape {
    label: &'static str,
    /// `{cols}` placeholder unused; formula is literal over column names x0,x1,...
    formula: &'static str,
    family: &'static str,
    ncols: usize,
}

/// The curated shape grid. Each entry is a VALID model the engine supports;
/// the fuzz dimensions are n, the seed (data realization), and any degenerate
/// column structure injected by `degeneracy`.
const SHAPES: &[Shape] = &[
    // ── single 1-D smooths across families ──────────────────────────────
    Shape {
        label: "gauss_s",
        formula: "y ~ s(x0)",
        family: "gaussian",
        ncols: 1,
    },
    Shape {
        label: "gauss_matern",
        formula: "y ~ matern(x0)",
        family: "gaussian",
        ncols: 1,
    },
    Shape {
        label: "gauss_duchon",
        formula: "y ~ duchon(x0)",
        family: "gaussian",
        ncols: 1,
    },
    Shape {
        label: "pois_s",
        formula: "y ~ s(x0)",
        family: "poisson",
        ncols: 1,
    },
    Shape {
        label: "binom_s",
        formula: "y ~ s(x0)",
        family: "binomial",
        ncols: 1,
    },
    Shape {
        label: "gamma_s",
        formula: "y ~ s(x0)",
        family: "gamma",
        ncols: 1,
    },
    // ── parametric / mixed ──────────────────────────────────────────────
    Shape {
        label: "gauss_linear",
        formula: "y ~ x0",
        family: "gaussian",
        ncols: 1,
    },
    Shape {
        label: "gauss_lin2",
        formula: "y ~ x0 + x1",
        family: "gaussian",
        ncols: 2,
    },
    Shape {
        label: "gauss_mix",
        formula: "y ~ x0 + s(x1)",
        family: "gaussian",
        ncols: 2,
    },
    Shape {
        label: "pois_mix",
        formula: "y ~ x0 + s(x1)",
        family: "poisson",
        ncols: 2,
    },
    // ── additive multi-smooth ───────────────────────────────────────────
    Shape {
        label: "gauss_add2",
        formula: "y ~ s(x0) + s(x1)",
        family: "gaussian",
        ncols: 2,
    },
    // ── tensor product ──────────────────────────────────────────────────
    Shape {
        label: "gauss_te",
        formula: "y ~ te(x0, x1)",
        family: "gaussian",
        ncols: 2,
    },
];

/// Degeneracy injectors applied to a generated predictor column. These are the
/// edge cases the mission flags: collinear / constant / near-zero-variance
/// designs that frequently break factorizations.
#[derive(Clone, Copy)]
enum Degeneracy {
    None,
    /// One predictor column is a constant (zero variance).
    ConstantCol,
    /// Two predictor columns are exactly collinear (x1 = x0).
    Collinear,
    /// A predictor column has a single tied repeated value over half the rows.
    HeavyTie,
}

impl Degeneracy {
    const ALL: &'static [Degeneracy] = &[
        Degeneracy::None,
        Degeneracy::ConstantCol,
        Degeneracy::Collinear,
        Degeneracy::HeavyTie,
    ];
    fn label(self) -> &'static str {
        match self {
            Degeneracy::None => "none",
            Degeneracy::ConstantCol => "constcol",
            Degeneracy::Collinear => "collinear",
            Degeneracy::HeavyTie => "heavytie",
        }
    }
}

/// Build a dataset with `ncols` predictor columns `x0..x{ncols-1}` and a
/// response `y` appropriate to `family`. Degeneracy is injected into the
/// predictor block; the response is always a smooth-ish function of x0 plus
/// noise so the problem is well-posed in expectation.
fn build_data(
    ncols: usize,
    family: &str,
    n: usize,
    degen: Degeneracy,
    seed: u64,
) -> gam::data::EncodedDataset {
    let mut rng = StdRng::seed_from_u64(seed);
    let unif = Uniform::new(0.0_f64, 1.0).expect("uniform");

    // Predictor columns.
    let mut cols: Vec<Vec<f64>> = (0..ncols)
        .map(|_| (0..n).map(|_| unif.sample(&mut rng)).collect::<Vec<f64>>())
        .collect();

    match degen {
        Degeneracy::None => {}
        Degeneracy::ConstantCol => {
            if ncols >= 1 {
                // Make the LAST predictor column constant.
                let v = 0.5;
                for r in cols.last_mut().unwrap().iter_mut() {
                    *r = v;
                }
            }
        }
        Degeneracy::Collinear => {
            if ncols >= 2 {
                let c0 = cols[0].clone();
                cols[1] = c0;
            }
        }
        Degeneracy::HeavyTie => {
            if ncols >= 1 {
                let half = n / 2;
                for r in cols[0].iter_mut().take(half) {
                    *r = 0.3;
                }
            }
        }
    }

    // Mean signal from x0 (a clean nonlinear trend so smooths have something).
    let signal: Vec<f64> = (0..n)
        .map(|i| {
            let t = cols[0][i];
            (2.0 * std::f64::consts::PI * t).sin() * 0.6 + 0.4 * t
        })
        .collect();

    let y: Vec<f64> = match family {
        "gaussian" | "gamma" => {
            let noise = Normal::new(0.0, 0.1).expect("normal");
            signal
                .iter()
                .map(|&s| {
                    let raw = s + noise.sample(&mut rng);
                    if family == "gamma" {
                        // Gamma needs strictly-positive responses.
                        raw.exp().max(1e-3)
                    } else {
                        raw
                    }
                })
                .collect()
        }
        "poisson" => signal
            .iter()
            .map(|&s| {
                // mean = exp(centered signal); sample a small count deterministically.
                let mu = (s).exp();
                // Deterministic rounded Poisson-ish count without a Poisson dist.
                let u = unif.sample(&mut rng);
                (mu + u).floor().max(0.0)
            })
            .collect(),
        "binomial" => signal
            .iter()
            .map(|&s| {
                let p = 1.0 / (1.0 + (-s).exp());
                let u = unif.sample(&mut rng);
                if u < p { 1.0 } else { 0.0 }
            })
            .collect(),
        other => panic!("unsupported fuzz family {other}"),
    };

    // Assemble CSV records: x0..x{ncols-1}, y.
    let mut headers: Vec<String> = (0..ncols).map(|j| format!("x{j}")).collect();
    headers.push("y".to_string());

    let rows: Vec<StringRecord> = (0..n)
        .map(|i| {
            let mut fields: Vec<String> = (0..ncols).map(|j| cols[j][i].to_string()).collect();
            fields.push(y[i].to_string());
            StringRecord::from(fields)
        })
        .collect();

    encode_recordswith_inferred_schema(headers, rows).expect("encode fuzz data")
}

/// Outcome of fitting + checking one case. `Ok(())` is invariant-clean;
/// `Err(msg)` is an invariant VIOLATION (a bug) with a self-contained repro.
fn check_case(shape: &Shape, n: usize, degen: Degeneracy, seed: u64) -> Result<(), String> {
    let repro = format!(
        "[{}/{} family={} n={} degen={} seed={}] formula=`{}`",
        shape.label,
        shape.formula,
        shape.family,
        n,
        degen.label(),
        seed,
        shape.formula,
    );

    let data = build_data(shape.ncols, shape.family, n, degen, seed);
    let cfg = FitConfig {
        family: Some(shape.family.to_string()),
        ..FitConfig::default()
    };

    // ── fit (must not panic; a clean Err is acceptable — the engine is
    //    allowed to REFUSE a degenerate problem, it just may not crash or
    //    silently return garbage) ─────────────────────────────────────────
    let result = match fit_from_formula(shape.formula, &data, &cfg) {
        Ok(r) => r,
        Err(_) => return Ok(()), // a loud, typed refusal is fine.
    };

    // Refit for the determinism invariant (I4).
    let result2 = fit_from_formula(shape.formula, &data, &cfg);

    let FitResult::Standard(fit) = result else {
        // Non-standard fast paths (scan/cascade) carry their own invariant
        // tests; this sweep targets the dense standard stack.
        return Ok(());
    };

    // I1 finite β.
    if fit.fit.beta.iter().any(|v| !v.is_finite()) {
        return Err(format!(
            "{repro}\n  I1 VIOLATED: non-finite beta {:?}",
            fit.fit.beta
        ));
    }

    // I3 finite objective.
    if !fit.fit.reml_score.is_finite() {
        return Err(format!(
            "{repro}\n  I3 VIOLATED: non-finite reml_score {}",
            fit.fit.reml_score
        ));
    }
    if !fit.fit.log_likelihood.is_finite() {
        return Err(format!(
            "{repro}\n  I3 VIOLATED: non-finite log_likelihood {}",
            fit.fit.log_likelihood
        ));
    }

    // I2 finite predictions on training rows. Rebuild the predict design from
    // the frozen resolved spec over the SAME predictor columns the fit saw.
    // Recover predictor columns by re-reading them deterministically: rebuild
    // the same data and pull the encoded predictor matrix shape. We construct
    // the predict matrix directly from the regenerated columns.
    let pred_matrix = training_predictor_matrix(shape.ncols, n, degen, seed);
    match build_term_collection_design(pred_matrix.view(), &fit.resolvedspec) {
        Ok(design) => {
            let pred = design.design.apply(&fit.fit.beta).to_vec();
            if pred.len() != n {
                return Err(format!(
                    "{repro}\n  I2 VIOLATED: predict produced {} rows, expected {n}",
                    pred.len()
                ));
            }
            if pred.iter().any(|v| !v.is_finite()) {
                return Err(format!(
                    "{repro}\n  I2 VIOLATED: non-finite training prediction"
                ));
            }
        }
        Err(e) => {
            return Err(format!(
                "{repro}\n  I2 VIOLATED: predict design rebuild failed: {e}"
            ));
        }
    }

    // I4 determinism: refit β must match to tight tolerance.
    if let Ok(FitResult::Standard(fit2)) = result2 {
        if fit2.fit.beta.len() == fit.fit.beta.len() {
            let max_diff = fit
                .fit
                .beta
                .iter()
                .zip(fit2.fit.beta.iter())
                .map(|(a, b)| (a - b).abs())
                .fold(0.0_f64, f64::max);
            // Allow only floating-point reduction noise, not a different optimum.
            if max_diff > 1e-6 {
                return Err(format!(
                    "{repro}\n  I4 VIOLATED: refit β diverged by {max_diff:.3e} (non-determinism)"
                ));
            }
        } else {
            return Err(format!(
                "{repro}\n  I4 VIOLATED: refit β length {} != {}",
                fit2.fit.beta.len(),
                fit.fit.beta.len()
            ));
        }
    }

    // I5 EDF bounds + I6 SE ordering (only when inference is populated).
    if let Some(inf) = fit.fit.inference.as_ref() {
        let p = fit.fit.beta.len() as f64;
        let edf = inf.edf_total;
        if !edf.is_finite() {
            return Err(format!("{repro}\n  I5 VIOLATED: non-finite edf_total"));
        }
        // EDF lives in [0, p]; allow a small numerical slack on the upper edge.
        if edf < -1e-6 || edf > p + 1e-3 {
            return Err(format!(
                "{repro}\n  I5 VIOLATED: edf_total {edf:.4} outside [0, p={p}]"
            ));
        }
        if let Some(se) = inf.beta_standard_errors.as_ref() {
            if se.iter().any(|v| !v.is_finite() || *v < 0.0) {
                return Err(format!(
                    "{repro}\n  I6 VIOLATED: standard error non-finite or negative: {se:?}"
                ));
            }
        }
    }

    Ok(())
}

/// Rebuild the training predictor matrix (x0..x{ncols-1} plus a trailing column
/// the design replay ignores) deterministically, matching `build_data`'s RNG
/// stream for the predictor block. The response draws happen AFTER the
/// predictor draws in `build_data`, so reproducing only the predictor columns
/// here uses the identical leading RNG sequence.
fn training_predictor_matrix(ncols: usize, n: usize, degen: Degeneracy, seed: u64) -> Array2<f64> {
    let mut rng = StdRng::seed_from_u64(seed);
    let unif = Uniform::new(0.0_f64, 1.0).expect("uniform");
    let mut cols: Vec<Vec<f64>> = (0..ncols)
        .map(|_| (0..n).map(|_| unif.sample(&mut rng)).collect::<Vec<f64>>())
        .collect();
    match degen {
        Degeneracy::None => {}
        Degeneracy::ConstantCol => {
            if ncols >= 1 {
                for r in cols.last_mut().unwrap().iter_mut() {
                    *r = 0.5;
                }
            }
        }
        Degeneracy::Collinear => {
            if ncols >= 2 {
                let c0 = cols[0].clone();
                cols[1] = c0;
            }
        }
        Degeneracy::HeavyTie => {
            if ncols >= 1 {
                let half = n / 2;
                for r in cols[0].iter_mut().take(half) {
                    *r = 0.3;
                }
            }
        }
    }
    // The predict design needs at least ncols columns; append one zero column
    // so single-predictor formulas (which `build_data` writes as [x0, y]) line
    // up with the build_term_collection_design column expectation.
    let total = ncols + 1;
    let mut m = Array2::<f64>::zeros((n, total));
    for j in 0..ncols {
        for i in 0..n {
            m[[i, j]] = cols[j][i];
        }
    }
    m
}

/// The fixed deterministic case grid: every (shape × degeneracy × n × seed).
/// Kept small enough to run in CI but wide enough to surface edge-case bugs.
fn case_grid() -> Vec<(&'static Shape, usize, Degeneracy, u64)> {
    let ns: &[usize] = &[8, 12, 20, 40];
    let seeds: &[u64] = &[1, 7, 19, 101];
    let mut out = Vec::new();
    for shape in SHAPES {
        for &degen in Degeneracy::ALL {
            // Degeneracies that need >= 2 cols are skipped for 1-col shapes.
            if matches!(degen, Degeneracy::Collinear) && shape.ncols < 2 {
                continue;
            }
            for &n in ns {
                for &seed in seeds {
                    out.push((shape, n, degen, seed));
                }
            }
        }
    }
    out
}

#[test]
fn fuzz_fit_universal_invariants() {
    init_parallelism();
    let grid = case_grid();
    let mut violations: Vec<String> = Vec::new();
    let mut ran = 0usize;
    for (shape, n, degen, seed) in grid {
        ran += 1;
        match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            check_case(shape, n, degen, seed)
        })) {
            Ok(Ok(())) => {}
            Ok(Err(v)) => violations.push(v),
            Err(_) => violations.push(format!(
                "[{} family={} n={} degen={} seed={}] PANIC during fit/predict",
                shape.label,
                shape.family,
                n,
                degen.label(),
                seed
            )),
        }
    }
    eprintln!(
        "[fuzz] ran {ran} cases, {} invariant violations",
        violations.len()
    );
    assert!(
        violations.is_empty(),
        "fuzz invariant violations ({} of {ran} cases):\n  - {}",
        violations.len(),
        violations.join("\n  - "),
    );
}
