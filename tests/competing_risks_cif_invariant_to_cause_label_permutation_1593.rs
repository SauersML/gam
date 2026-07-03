//! #1593 gauge-invariance guard — a fitted competing-risks model's cumulative
//! incidence functions (CIFs) are invariant to an arbitrary PERMUTATION of the
//! cause labels (which physical competing event is coded `1` vs `2` vs ...).
//!
//! The cause-label coding is a pure relabelling: codes `{0 = censored, k =
//! cause k}` only NAME the competing events; the physical data (who failed,
//! when, from which event) are identical under any bijection of the nonzero
//! codes. A correct competing-risks fit + CIF assembly must therefore produce
//! the SAME cumulative incidence for a given physical cause no matter which
//! integer label that cause was assigned.
//!
//! This matters because the unified cause-specific Royston-Parmar fit indexes
//! its per-cause coefficient blocks by the SORTED event level (cause code `k`
//! → block `k-1`), and the joint identifiability path assigns DESCENDING gauge
//! priorities to those blocks (`gauge_priority = 100 + (cause_count - cause)`,
//! cause 0 highest) so the channel-aware audit can resolve the K shared,
//! near-aliased time-basis columns
//! (`crates/gam-models/src/fit_orchestration/fit.rs` and the mirrored note in
//! `crates/gam-models/src/survival/predict.rs` around the per-cause block
//! loop). Permuting the cause labels permutes which physical cause lands in
//! which sorted-level block — and thus which physical cause receives the
//! highest gauge priority. That is exactly a frame-anchored choice the fit
//! could silently depend on (were the per-cause baseline anchored to the sorted
//! event-level index rather than to a canonical, label-free cause identity).
//! It SHOULD be invariant, so this test LOCKS THAT IN as a green guard — the
//! competing-risks sibling of the multinomial reference class (#1587), the
//! simplex ALR reference (#1549), the cyclic period origin and the `ti` margin
//! order guards in this #1593 family.
//!
//! It fits the SAME physical two-cause competing-risks data twice — once under
//! the natural labelling `{1, 2}` and once under the swapped labelling (the
//! two nonzero codes exchanged) — derives each fit's per-cause cumulative
//! hazards from its per-cause coefficient blocks exactly as the
//! `quality_vs_lifelines_competing_risks_cif` test does, assembles the
//! per-cause CIF surfaces through gam's own competing-risks integrator
//! `assemble_competing_risks_cif`, REALIGNS the swapped fit's blocks back to a
//! common physical-cause identity, and asserts the two fits agree on each
//! physical cause's CIF to a tight fraction of the signal range — while a refit
//! under the SAME labelling is confirmed deterministic, so any cross-labelling
//! drift would be a real frame dependence and not refit noise.

use csv::StringRecord;
use gam::families::survival::assemble_competing_risks_cif;
use gam::matrix::LinearOperator;
use gam::smooth::build_term_collection_design;
use gam::{
    FitConfig, FitResult, encode_recordswith_inferred_schema, fit_from_formula, init_parallelism,
};
use ndarray::{Array2, Array3};

/// Deterministic SplitMix64 → byte-identical data run-to-run (no external RNG),
/// so any cross-labelling disagreement is a fit property, not sampling noise.
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
    /// Uniform on (0, 1).
    fn unit(&mut self) -> f64 {
        ((self.next_u64() >> 11) as f64 + 0.5) / (1u64 << 53) as f64
    }
}

/// Synthetic competing-risks data with TWO causes on the same physical event
/// times, with deliberately ASYMMETRIC cause-specific hazards (cause 1's hazard
/// rises with the covariate `x`, cause 2's falls) so the two causes are
/// genuinely distinct fits and a label-dependent identifiability resolution
/// would have something to disagree about. The natural labelling uses codes
/// `{0 = censored, 1 = cause A, 2 = cause B}`.
///
/// Returns the per-row (covariate x, exit time, natural event code) so the
/// caller can relabel the codes for the permuted fit while holding x and time
/// fixed (the SAME physical data).
struct CrData {
    x: Vec<f64>,
    time: Vec<f64>,
    code: Vec<f64>,
}

fn build(seed: u64) -> CrData {
    let n = 1500usize;
    let mut rng = SplitMix64::new(seed);
    let mut x = Vec::with_capacity(n);
    let mut time = Vec::with_capacity(n);
    let mut code = Vec::with_capacity(n);
    // Administrative censoring horizon; events beyond it are right-censored.
    let horizon = 6.0_f64;
    for _ in 0..n {
        let xi = rng.unit(); // covariate on (0, 1)
        // Cause A: hazard rises with x. Cause B: hazard falls with x.
        // Exponential cause-specific event times via inverse-CDF.
        let rate_a = 0.18 * (0.9 * (xi - 0.5)).exp();
        let rate_b = 0.18 * (-0.9 * (xi - 0.5)).exp();
        let u_a = rng.unit();
        let u_b = rng.unit();
        let t_a = -(u_a.ln()) / rate_a;
        let t_b = -(u_b.ln()) / rate_b;
        // First event wins; censor at the horizon.
        let (t_event, cause) = if t_a <= t_b { (t_a, 1.0) } else { (t_b, 2.0) };
        let (t_obs, c) = if t_event > horizon {
            (horizon, 0.0)
        } else {
            (t_event, cause)
        };
        x.push(xi);
        time.push(t_obs);
        code.push(c);
    }
    CrData { x, time, code }
}

/// Fit the unified competing-risks Weibull model `Surv(time, event) ~ s(x)` on
/// the given (x, time, code) data and return, for each cause block in
/// sorted-event-level order, the per-subject cumulative hazard `H_k(t | x_i)`
/// on the shared `(grid_x, grid_t)` evaluation grid.
///
/// The per-cause cumulative hazard is reconstructed from the unified fit's
/// per-cause coefficient block exactly as the
/// `quality_vs_lifelines_competing_risks_cif` quality test does for the
/// single-cause fits:
///   H_k(t | x) = (t / scale)^shape_k * exp(eta_k(x)),
/// where `scale` is the shared fitted Weibull anchor (`baseline_cfg.scale`),
/// `shape_k = beta_k[1]` (the slope on log-time of the anchor-centered linear
/// `[1, log t]` time basis for cause k), and `eta_k(x)` is the centered
/// covariate smooth `cov_design(x) · beta_k[time_base_ncols..]`.
///
/// Returns one `[grid_x, grid_t]` matrix per cause, in sorted block order
/// (block index `c` ⇔ event code `c + 1`).
fn fit_and_cause_cumulative_hazards(
    x: &[f64],
    time: &[f64],
    code: &[f64],
    grid_x: &[f64],
    grid_t: &[f64],
) -> Vec<Array2<f64>> {
    let n = x.len();
    let rows: Vec<StringRecord> = (0..n)
        .map(|i| {
            StringRecord::from(vec![
                time[i].to_string(),
                code[i].to_string(),
                x[i].to_string(),
            ])
        })
        .collect();
    let headers = vec!["time".to_string(), "event".to_string(), "x".to_string()];
    let data =
        encode_recordswith_inferred_schema(headers, rows).expect("encode competing-risks data");

    let cfg = FitConfig {
        survival_likelihood: "weibull".to_string(),
        ..FitConfig::default()
    };
    let result = fit_from_formula("Surv(time, event) ~ s(x, bs='tp')", &data, &cfg)
        .expect("unified competing-risks Weibull fit");
    let FitResult::SurvivalTransformation(fit) = result else {
        panic!("expected a SurvivalTransformation fit for the unified competing-risks model");
    };

    let cause_count = fit.fit.blocks.len();
    assert!(
        cause_count >= 2,
        "expected at least two cause blocks, got {cause_count}"
    );

    // Shared Weibull anchor (scale) recovered from the fit; per-cause shape is
    // the log-time slope beta_k[1] of the anchor-centered linear time basis.
    let scale = fit
        .baseline_cfg
        .scale
        .expect("fitted competing-risks Weibull anchor (scale)");
    assert!(
        scale.is_finite() && scale > 0.0,
        "fitted Weibull scale must be positive finite, got {scale}"
    );
    let time_base = fit.time_base_ncols;
    assert!(
        time_base >= 2,
        "Weibull time basis must carry at least [1, log t] = 2 columns, got {time_base}"
    );

    // Build the centered covariate smooth design at every grid x value.
    let headers_ref = ["time".to_string(), "event".to_string(), "x".to_string()];
    let x_idx = headers_ref
        .iter()
        .position(|h| h == "x")
        .expect("x column index");
    let m = grid_x.len();
    let mut covgrid = Array2::<f64>::zeros((m, headers_ref.len()));
    for (i, &xi) in grid_x.iter().enumerate() {
        covgrid[[i, x_idx]] = xi;
    }
    let design = build_term_collection_design(covgrid.view(), &fit.resolvedspec)
        .expect("rebuild covariate design at grid x values");
    let cov_ncols = design.design.ncols();

    let mut out = Vec::with_capacity(cause_count);
    for c in 0..cause_count {
        let beta = &fit.fit.blocks[c].beta;
        assert_eq!(
            beta.len(),
            time_base + cov_ncols,
            "cause {} beta layout mismatch: len={} time_base={} cov_ncols={}",
            c + 1,
            beta.len(),
            time_base,
            cov_ncols
        );
        let shape = beta[1];
        assert!(
            shape.is_finite() && shape > 0.0,
            "cause {} fitted Weibull shape must be positive finite, got {shape}",
            c + 1
        );
        let cov_beta = beta.slice(ndarray::s![time_base..]).to_owned();
        let eta = design.design.apply(&cov_beta);
        assert_eq!(
            eta.len(),
            m,
            "cause {} covariate eta length mismatch",
            c + 1
        );

        // H_k(t | x_i) = (t / scale)^shape * exp(eta_i) on the shared grid.
        let mut h = Array2::<f64>::zeros((m, grid_t.len()));
        for i in 0..m {
            let mult = eta[i].exp();
            for (j, &t) in grid_t.iter().enumerate() {
                let h0 = if t <= 0.0 {
                    0.0
                } else {
                    (t / scale).powf(shape)
                };
                h[[i, j]] = h0 * mult;
            }
        }
        out.push(h);
    }
    out
}

/// Assemble the per-cause CIF surfaces from per-cause cumulative hazards (in
/// sorted block order) via gam's own competing-risks integrator, returning one
/// `[grid_x, grid_t]` CIF matrix per cause (same block order as the input).
fn assemble_cifs(cause_hazards: &[Array2<f64>], grid_t: &[f64]) -> Vec<Array2<f64>> {
    let cause_count = cause_hazards.len();
    let (m, t_cols) = cause_hazards[0].dim();
    let mut cumhaz = Array3::<f64>::zeros((cause_count, m, t_cols));
    for (c, h) in cause_hazards.iter().enumerate() {
        cumhaz.index_axis_mut(ndarray::Axis(0), c).assign(h);
    }
    let assembled = assemble_competing_risks_cif(ndarray::aview1(grid_t), cumhaz.view())
        .expect("assemble competing-risks CIF");
    assembled.cif
}

#[test]
fn competing_risks_cif_is_invariant_to_cause_label_permutation_1593() {
    init_parallelism();

    // A correct competing-risks fit is label-invariant by construction: the
    // cause-specific likelihood routes each cause to disjoint risk sets and the
    // CIF assembly's split (ΔH_k / ΔH_total) is symmetric in the causes, so a
    // relabelling is a pure permutation of the per-cause blocks. We hold a tight
    // 1e-3 of the signal range — mirroring the cyclic period-origin sibling — so
    // this is a real (non-vacuous) guard: a regression that anchored a per-cause
    // baseline to the sorted event-level index (rather than to the label-free
    // cause identity) would trip it.
    const REL_TOL: f64 = 1.0e-3;

    // Shared physical evaluation grid: a spread of covariate values × a strictly
    // increasing time grid that reaches a regime with real failure mass.
    let grid_x: Vec<f64> = (0..25).map(|i| (i as f64 + 0.5) / 25.0).collect();
    let grid_t: Vec<f64> = vec![0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 6.0];

    let mut worst_rel = 0.0_f64;
    let mut worst_seed = 0_u64;
    let mut worst_cause = 0usize;

    for seed in [1_u64, 3, 5] {
        let d = build(seed);
        let n_a = d.code.iter().filter(|&&c| c == 1.0).count();
        let n_b = d.code.iter().filter(|&&c| c == 2.0).count();
        assert!(
            n_a > 100 && n_b > 100,
            "seed {seed}: need substantial events per cause (A={n_a}, B={n_b}) for a non-vacuous guard"
        );

        // Natural labelling: code 1 = physical cause A, code 2 = physical cause B.
        let nat = fit_and_cause_cumulative_hazards(&d.x, &d.time, &d.code, &grid_x, &grid_t);
        let cif_nat = assemble_cifs(&nat, &grid_t);

        // Refit under the SAME labelling must be deterministic, else cross-
        // labelling drift could not be attributed to the cause frame.
        let nat_again = fit_and_cause_cumulative_hazards(&d.x, &d.time, &d.code, &grid_x, &grid_t);
        let cif_nat_again = assemble_cifs(&nat_again, &grid_t);
        let refit_noise: f64 = cif_nat
            .iter()
            .zip(&cif_nat_again)
            .flat_map(|(a, b)| a.iter().zip(b.iter()))
            .map(|(a, b)| (a - b).abs())
            .fold(0.0, f64::max);
        assert!(
            refit_noise < 1e-9,
            "seed {seed}: same-labelling refit is non-deterministic (max|ΔCIF|={refit_noise:.3e}); \
             cannot attribute cross-labelling drift to the cause frame"
        );

        // Permuted labelling on the SAME physical data: swap the two nonzero
        // codes (1 ↔ 2), so the sorted-event-level → block mapping now places
        // physical cause B in block 0 and physical cause A in block 1. Censored
        // rows (code 0) are untouched.
        let code_swapped: Vec<f64> = d
            .code
            .iter()
            .map(|&c| {
                if c == 1.0 {
                    2.0
                } else if c == 2.0 {
                    1.0
                } else {
                    c
                }
            })
            .collect();
        let perm = fit_and_cause_cumulative_hazards(&d.x, &d.time, &code_swapped, &grid_x, &grid_t);
        let cif_perm = assemble_cifs(&perm, &grid_t);

        assert_eq!(
            cif_nat.len(),
            cif_perm.len(),
            "seed {seed}: cause count must match across labellings"
        );
        assert_eq!(cif_nat.len(), 2, "seed {seed}: expected exactly two causes");

        // Realign: under the swap, swapped block index `b` corresponds to the
        // natural block index that physical cause now occupies. Physical cause A
        // (natural block 0) is swapped block 1; physical cause B (natural block
        // 1) is swapped block 0. So natural[c] must match perm[1 - c].
        for c in 0..cif_nat.len() {
            let reference = &cif_nat[c];
            let realigned = &cif_perm[cif_nat.len() - 1 - c];

            // Non-degeneracy: the CIF for this physical cause must actually vary
            // across the (x, t) grid, else the invariant is vacuous.
            let max = reference.iter().cloned().fold(f64::MIN, f64::max);
            let min = reference.iter().cloned().fold(f64::MAX, f64::min);
            let range = max - min;
            assert!(
                range > 0.02,
                "seed {seed} cause {}: degenerate (constant) CIF surface (range {range:.5}); \
                 the invariant would be vacuous",
                c + 1
            );

            let max_abs: f64 = reference
                .iter()
                .zip(realigned.iter())
                .map(|(a, b)| (a - b).abs())
                .fold(0.0, f64::max);
            let rel = max_abs / range;
            if rel > worst_rel {
                worst_rel = rel;
                worst_seed = seed;
                worst_cause = c + 1;
            }
            eprintln!(
                "[cr-cause-permut] seed={seed} cause={} | max|ΔCIF|/range={rel:.3e} \
                 (refit noise {refit_noise:.3e}, signal range {range:.4})",
                c + 1
            );
        }
    }
    eprintln!(
        "[cr-cause-permut] worst max|ΔCIF|/range across seeds/causes = {worst_rel:.3e} \
         (seed {worst_seed}, cause {worst_cause})"
    );

    assert!(
        worst_rel < REL_TOL,
        "competing-risks CIF DIFFERS under a permutation of the cause labels: worst max|ΔCIF| \
         across seeds/causes is {worst_rel:.3e} of the signal range (seed {worst_seed}, cause \
         {worst_cause}, tol {REL_TOL:.0e}). The cause-label coding is a pure relabelling of the \
         competing events, so the per-cause cumulative incidence must be invariant to it \
         (#1593 gauge-invariance class). A drift here is a real cause-label-frame dependence of \
         the #1549/#1587 family — most likely a per-cause baseline anchored to the sorted \
         event-level index rather than to the label-free cause identity."
    );
}
