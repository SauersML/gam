//! Owed-work regression gate for issue #1378 — the default univariate thin-plate
//! smooth `s(x, bs="tp")` was NOT invariant to a pure row permutation of the
//! training data.
//!
//! ## The defect (now fixed)
//!
//! A GAM fit is a functional of the *unordered set* of `(x, y)` observations, so
//! reordering the rows of an identical dataset must reproduce the identical fit
//! (up to round-off). The local bases `bs="cr"` and `bs="ps"` honour this
//! bit-for-bit. The default `bs="tp"` did not: permuting the rows moved the
//! fitted curve by ~3% of the signal range (the issue measured a worst-case
//! prediction drift of 0.0756, ~1.5× the fit's own RMSE-to-truth).
//!
//! Two mechanisms, both fixed on `main`:
//!
//!   (a) `select_thin_plate_knots` (`src/terms/basis/duchon_thinplate.rs`) broke
//!       the greedy maximin tie (equal `min_dist2` to the chosen set, common on
//!       near-regular 1-D data) by ROW INDEX. A pure permutation then selected a
//!       different tied knot, yielding a different basis, conditioning, and REML
//!       λ̂. Fixed (`93f938970`) by breaking ties value-lexicographically — a
//!       pure function of the unordered data value set (the `value_less` closure,
//!       used for both the seed and every maximin step).
//!
//!   (b) The default 1-D tp basis dimension was n-scaled and oversized; the
//!       resulting REML ARC sub-problem could cost-stall NON-converged, leaving
//!       the selected λ̂ order-dependent. Fixed (`980725fc2`) by sizing the
//!       default univariate tp basis to the modest mgcv-style ceiling
//!       `THIN_PLATE_1D_DEFAULT_BASIS_DIM = 10` (`src/terms/term_builder.rs`).
//!
//! The knot-set invariance is pinned in-crate by
//! `terms::basis::duchon_thinplate::tests::knot_set_is_row_permutation_invariant_gh1378`.
//! This file is the end-to-end complement: it drives the PUBLIC fit path on the
//! issue's own reproducer geometry and asserts that every row permutation reaches
//! the same certified optimum.
//!
//! ## Why the tp contract is a certified optimum, not bit-identity
//!
//! A permutation reorders every O(n) reduction, so even a correct pipeline stops
//! at a different point inside its certificate's tolerance ball, which
//! `tests/pyffi/optimization/warm_start_invariance_contract.rs` documents for a
//! donated seed. Bit-identity of λ̂ is therefore not the contract.
//!
//! What lane-only diagnostics measured: the base tp fit and six same-order fits
//! with `y` nudged by one ulp land on one λ̂ with criteria within 8 ulps (pool jobs
//! 599189 at `583577152` and 602711 at `0243a382d`). The six permutations land on
//! two other λ̂, up to 3e-6 away, with criteria 1.3e8 to 1.7e8 ulps from the base
//! (curve drift 1.2e-7 of a 2.4 signal range), while ps and cr move their criteria
//! by at most 240 ulps. Six one-ulp nudges of `x` in the SAME row order move the tp
//! criterion 1.2e8 to 6.3e8 ulps (job 609264 at `2886c2803`). That arm reorders
//! nothing, so the spread is rounding amplified through the tp design's
//! conditioning rather than a row-order computation, and no ulp budget over O(n)
//! reductions describes it.
//!
//! The tp gate therefore judges both arms on what the two certificates publish: the
//! stationarity bound `b` every certified point satisfies, and `V_ρ`.
//!   * each permutation's log-λ̂ lies inside the ball both certificates license,
//!     `‖Δρ̂‖∞ ≤ ‖V_ρ‖∞·(b_base + b_perm)`;
//!   * the criteria agree within the quadratic-model value gap of two certified
//!     points, `|ΔV| ≤ ½·m·‖V_ρ‖∞·(b_base² + b_perm²)` for `m` smoothing coordinates.
//! Measured at `3ebdcf869` (job 644261): the tp permutations move the criterion
//! 4.7e-7 to 6.1e-7 against a bar of 8.2e-6 and log-λ̂ 6.0e-8 to 3.1e-6 against a
//! ball of 5.8e-3. Seeding the knot recursion at data row 0, the index-dependent
//! defect class this issue names, moves them 4.5e-4 to 2.3e-3 and 8.9e-3 to 0.13, so
//! both arms fail on it. The curve is a function of the basis and λ̂, so its drift
//! is reported rather than bounded a second time.

use csv::StringRecord;
use gam::matrix::LinearOperator;
use gam::smooth::build_term_collection_design;
use gam::{
    FitConfig, FitResult, encode_recordswith_inferred_schema, fit_from_formula, init_parallelism,
};
use ndarray::Array2;
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rand_distr::{Distribution, Normal, Uniform};


/// Signed ulp distance between two `f64`s, using the monotone ordering of their
/// bit patterns so the count stays meaningful across zero.
fn ulp_distance(left: f64, right: f64) -> i128 {
    let order = |value: f64| -> i128 {
        let bits = value.to_bits() as i64;
        if bits < 0 {
            (i64::MIN - bits) as i128
        } else {
            bits as i128
        }
    };
    order(left) - order(right)
}

/// Build the issue-#1378 dataset: `y = sin(2x) + 0.3x + N(0, 0.2)` on `n` points
/// with `x` drawn uniform on `[-2, 2]` and SORTED ascending (so the canonical row
/// order is the value order — exactly the regime where an index-based tie-break
/// silently masquerades as a value-based one until the rows are permuted).
fn make_data(n: usize, seed: u64) -> (Vec<f64>, Vec<f64>) {
    let mut rng = StdRng::seed_from_u64(seed);
    let unif = Uniform::new(-2.0_f64, 2.0).expect("uniform");
    let noise = Normal::new(0.0, 0.2).expect("normal");
    let mut x: Vec<f64> = (0..n).map(|_| unif.sample(&mut rng)).collect();
    x.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let y: Vec<f64> = x
        .iter()
        .map(|&xi| (2.0 * xi).sin() + 0.3 * xi + noise.sample(&mut rng))
        .collect();
    (x, y)
}

/// Apply a row permutation to `(x, y)` and return the reordered columns.
fn permute(x: &[f64], y: &[f64], perm: &[usize]) -> (Vec<f64>, Vec<f64>) {
    (
        perm.iter().map(|&i| x[i]).collect(),
        perm.iter().map(|&i| y[i]).collect(),
    )
}

/// The certified outer optimum a dense REML fit publishes.
struct CertifiedOptimum {
    criterion: f64,
    log_lambdas: Vec<f64>,
    /// The KKT-projected outer gradient norm the certificate carries.
    projected_grad_norm: Option<f64>,
    /// The stationarity bound the certificate was decided at: every point whose
    /// projected gradient is within it is certified.
    stationarity_bound: Option<f64>,
    /// `‖V_ρ‖∞`, the max absolute row sum of the published inverse outer Hessian.
    rho_covariance_inf_norm: Option<f64>,
}

/// One fit's predicted curve, and its certified optimum when the fit selected λ
/// by an outer search. The exact state-space scan and the residual cascade do not.
struct FitOutcome {
    predictions: Vec<f64>,
    optimum: Option<CertifiedOptimum>,
}

/// Fit `y ~ s(x, bs="<bs>")` on `(x, y)` and return the predicted curve on a
/// fixed grid in `[-1.8, 1.8]` (the issue's prediction grid), with the fit's
/// certified optimum.
fn fit_predict(bs: &str, x: &[f64], y: &[f64], grid: &[f64]) -> FitOutcome {
    let n = x.len();
    let headers: Vec<String> = ["x", "y"].into_iter().map(String::from).collect();
    let rows: Vec<StringRecord> = (0..n)
        .map(|i| StringRecord::from(vec![x[i].to_string(), y[i].to_string()]))
        .collect();
    let data = encode_recordswith_inferred_schema(headers, rows).expect("encode");
    let x_idx = data.column_map()["x"];

    let cfg = FitConfig {
        family: Some("gaussian".to_string()),
        ..FitConfig::default()
    };
    let formula = format!("y ~ s(x, bs=\"{bs}\")");
    let result = fit_from_formula(&formula, &data, &cfg).expect("fit");

    match result {
        FitResult::Standard(fit) => {
            // Identity-link Gaussian: prediction = design(grid) · beta.
            let mut grid_design = Array2::<f64>::zeros((grid.len(), data.headers.len()));
            for (row, &g) in grid.iter().enumerate() {
                grid_design[[row, x_idx]] = g;
            }
            let dm = build_term_collection_design(grid_design.view(), &fit.resolvedspec)
                .expect("rebuild design at grid");
            let optimum = CertifiedOptimum {
                criterion: fit
                    .fit
                    .reml_score()
                    .expect("a dense REML fit reports its criterion"),
                log_lambdas: fit.fit.log_lambdas.to_vec(),
                projected_grad_norm: fit
                    .fit
                    .convergence_evidence()
                    .outer_certificate()
                    .map(|certificate| certificate.stationarity.projected_norm()),
                stationarity_bound: fit
                    .fit
                    .convergence_evidence()
                    .outer_certificate()
                    .map(|certificate| certificate.stationarity.bound()),
                rho_covariance_inf_norm: fit.fit.artifacts.rho_covariance.as_ref().map(
                    |covariance| {
                        covariance
                            .rows()
                            .into_iter()
                            .map(|row| row.iter().map(|value| value.abs()).sum::<f64>())
                            .fold(0.0_f64, f64::max)
                    },
                ),
            };
            FitOutcome {
                predictions: dm.design.apply(&fit.fit.beta).to_vec(),
                optimum: Some(optimum),
            }
        }
        // A 1-D cubic Gaussian smooth (e.g. bs="cr") routes through the exact
        // O(n) state-space smoothing-spline scan, which carries its own exact
        // per-abscissa posterior rather than a dense design + beta. Predict the
        // posterior mean directly so the control base is exercised on the
        // representation the fit actually produced.
        FitResult::SplineScan(scan) => FitOutcome {
            predictions: grid
                .iter()
                .map(|&g| scan.predict(g).expect("spline scan predict").0)
                .collect(),
            optimum: None,
        },
        FitResult::ResidualCascade(cascade) => FitOutcome {
            predictions: grid
                .iter()
                .map(|&g| cascade.predict(&[g]).expect("residual cascade predict").0)
                .collect(),
            optimum: None,
        },
        _ => panic!("unexpected fit variant for {formula}"),
    }
}

/// Every way `candidate` fails to be the certified optimum `reference` located again.
///
/// Both bars come from what the two certificates publish. A point is certified
/// when its projected gradient is within the stationarity bound `b`, and near the
/// optimum `ρ − ρ⋆ = V_ρ·g`, so a certified point lies within `‖V_ρ‖∞·b` of `ρ⋆` in
/// log-λ and within `½·gᵀV_ρg ≤ ½·m·‖V_ρ‖∞·b²` of `V⋆` in criterion, where `m` is the
/// number of smoothing coordinates (`‖g‖₂² ≤ m·‖g‖∞²`, `‖V_ρ‖₂ ≤ ‖V_ρ‖∞`). Two certified
/// fits of the same unordered data therefore differ by at most the sum of the two.
fn optimum_disagreements(reference: &CertifiedOptimum, candidate: &CertifiedOptimum) -> Vec<String> {
    let mut disagreements = Vec::new();
    if reference.log_lambdas.len() != candidate.log_lambdas.len() {
        disagreements.push(format!(
            "log-λ layout differs: {} vs {}",
            reference.log_lambdas.len(),
            candidate.log_lambdas.len()
        ));
        return disagreements;
    }
    let criterion_ulps = ulp_distance(reference.criterion, candidate.criterion);
    let criterion_gap = (reference.criterion - candidate.criterion).abs();
    let displacement = reference
        .log_lambdas
        .iter()
        .zip(&candidate.log_lambdas)
        .fold(0.0_f64, |mx, (a, b)| mx.max((a - b).abs()));
    match (
        reference.stationarity_bound,
        candidate.stationarity_bound,
        reference.rho_covariance_inf_norm,
        candidate.rho_covariance_inf_norm,
    ) {
        (
            Some(reference_bound),
            Some(candidate_bound),
            Some(reference_covariance),
            Some(candidate_covariance),
        ) => {
            let covariance = reference_covariance.max(candidate_covariance);
            let dimension = reference.log_lambdas.len() as f64;
            let ball = covariance * (reference_bound + candidate_bound);
            let value_gap = 0.5
                * dimension
                * covariance
                * (reference_bound * reference_bound + candidate_bound * candidate_bound);
            eprintln!(
                "#1378 criterion gap={criterion_gap:.3e} ({criterion_ulps} ulps) bar={value_gap:.3e} \
                 log-λ displacement={displacement:.3e} ball={ball:.3e} (‖V_ρ‖∞={covariance:.3e}, \
                 bound=({reference_bound:.3e}, {candidate_bound:.3e}), ‖Pg‖=({:.3e}, {:.3e}))",
                reference.projected_grad_norm.unwrap_or(f64::NAN),
                candidate.projected_grad_norm.unwrap_or(f64::NAN)
            );
            if !(criterion_gap <= value_gap) {
                disagreements.push(format!(
                    "criterion {:.17e} vs {:.17e} differs by {criterion_gap:.3e} ({criterion_ulps} ulps), \
                     beyond the certified value gap {value_gap:.3e}",
                    reference.criterion, candidate.criterion
                ));
            }
            if !(displacement <= ball) {
                disagreements.push(format!(
                    "log-λ moved {displacement:.3e}, outside the certificate ball {ball:.3e} \
                     (‖V_ρ‖∞={covariance:.3e}, bound=({reference_bound:.3e}, {candidate_bound:.3e}))"
                ));
            }
        }
        _ => disagreements.push(format!(
            "log-λ moved {displacement:.3e} and the criterion {criterion_gap:.3e}, and a fit published \
             no outer certificate or no V_ρ, so there is no certified region to judge them against"
        )),
    }
    disagreements
}

/// How far a battery of pure row permutations moves one basis's fit.
struct PermutationReport {
    worst_drift: f64,
    signal_range: f64,
    /// Whether the unpermuted fit published a certified outer optimum.
    certified: bool,
    /// Every way a permuted fit left the unpermuted fit's certified optimum.
    optimum_violations: Vec<String>,
}

/// The worst |prediction drift| of `bs` under a battery of pure row permutations,
/// with the baseline curve's signal range and every certified-optimum violation.
fn worst_permutation_drift(bs: &str) -> PermutationReport {
    let n = 300usize;
    let (x, y) = make_data(n, 7);
    let grid: Vec<f64> = (0..40).map(|i| -1.8 + 3.6 * i as f64 / 39.0).collect();

    let base = fit_predict(bs, &x, &y, &grid);
    let signal_range = {
        let (mut lo, mut hi) = (f64::INFINITY, f64::NEG_INFINITY);
        for &v in &base.predictions {
            lo = lo.min(v);
            hi = hi.max(v);
        }
        (hi - lo).max(1e-12)
    };

    let mut worst_drift = 0.0_f64;
    let mut optimum_violations = Vec::new();
    for s in 0..6u64 {
        // A pure shuffle of the row order; the (x, y) value set is untouched.
        let mut perm: Vec<usize> = (0..n).collect();
        let mut rng = StdRng::seed_from_u64(200 + s);
        perm.shuffle(&mut rng);
        let (xp, yp) = permute(&x, &y, &perm);
        let permuted = fit_predict(bs, &xp, &yp, &grid);
        let drift = base
            .predictions
            .iter()
            .zip(&permuted.predictions)
            .fold(0.0_f64, |mx, (a, c)| mx.max((a - c).abs()));
        worst_drift = worst_drift.max(drift);
        match (&base.optimum, &permuted.optimum) {
            (Some(reference), Some(candidate)) => optimum_violations.extend(
                optimum_disagreements(reference, candidate)
                    .into_iter()
                    .map(|reason| format!("bs={bs} permutation {s}: {reason}")),
            ),
            (None, None) => {}
            _ => optimum_violations.push(format!(
                "bs={bs} permutation {s}: the fit variant changed with row order"
            )),
        }
    }
    PermutationReport {
        worst_drift,
        signal_range,
        certified: base.optimum.is_some(),
        optimum_violations,
    }
}

/// #1378: the default `s(x, bs="tp")` fit must be invariant to a pure row
/// permutation of the training data. A correct fit is a functional of the
/// unordered observation set, so every reordering must reach the same certified
/// optimum.
#[test]
fn default_thin_plate_fit_is_row_permutation_invariant_1378() {
    init_parallelism();

    let tp = worst_permutation_drift("tp");
    eprintln!(
        "#1378 bs=tp row-permutation drift = {:.3e} ({:.4}% of signal range {:.3e})",
        tp.worst_drift,
        100.0 * tp.worst_drift / tp.signal_range,
        tp.signal_range
    );

    // Anchors: the value-based local bases guard against a regression in the
    // fit/predict harness itself masking the tp result. Both now select λ by the
    // same certified dense REML search as tp, so they are judged the way tp is:
    // on the optimum their certificates license, not on a borrowed curve-drift
    // literal. A permutation reorders every O(n) reduction, and a certified search
    // may stop anywhere inside its stationarity ball. Measured at 95115c8a1f (sw4l
    // job 1244874): the ps permutations stop up to 1.6e-7 apart in log-λ̂ against
    // a ball of 2.2e-2, with criteria within 24 ulps and a curve drift of 2.3e-9.
    // The cr permutations are within 72 ulps, with a drift of 1.2e-12.
    let cr = worst_permutation_drift("cr");
    let ps = worst_permutation_drift("ps");
    eprintln!("#1378 bs=cr row-permutation drift = {:.3e}", cr.worst_drift);
    eprintln!("#1378 bs=ps row-permutation drift = {:.3e}", ps.worst_drift);
    for (bs, report) in [("cr", &cr), ("ps", &ps)] {
        assert!(
            report.certified,
            "anchor s(x, bs=\"{bs}\") published no certified outer optimum, so it cannot \
             anchor the harness"
        );
        assert!(
            report.optimum_violations.is_empty(),
            "anchor s(x, bs=\"{bs}\") is value-anchored and must be row-permutation invariant, \
             but a permuted fit left the unpermuted fit's certified optimum (harness drift \
             {:.3e}):\n{}",
            report.worst_drift,
            report.optimum_violations.join("\n")
        );
    }

    // The fix: a value-lexicographic knot tie-break plus the mgcv-sized default
    // basis make the selected knot set, basis, and REML criterion a pure function
    // of the unordered data, so every permutation locates the same certified
    // optimum. Before #1378 the curve moved by ~0.0756 (~3% of the signal range).
    assert!(
        tp.certified,
        "default s(x, bs=\"tp\") published no certified outer optimum, so row-permutation \
         invariance of its λ̂ cannot be judged"
    );
    assert!(
        tp.optimum_violations.is_empty(),
        "default s(x, bs=\"tp\") is NOT row-permutation invariant: a permuted fit left the \
         unpermuted fit's certified optimum (worst prediction drift {:.3e}, {:.4}% of signal \
         range). The greedy maximin knot tie-break must break ties value-lexicographically \
         (duchon_thinplate.rs select_thin_plate_knots `value_less`, #93f938970) and the default \
         1-D tp basis must be sized to THIN_PLATE_1D_DEFAULT_BASIS_DIM=10 (term_builder.rs, \
         #980725fc2) so REML λ̂ does not drift with row order.\n{}",
        tp.worst_drift,
        100.0 * tp.worst_drift / tp.signal_range,
        tp.optimum_violations.join("\n")
    );
}
