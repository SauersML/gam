//! gam#2484 — the Murphy–Topel generalization to a `GlobalEmpirical` second-stage
//! latent measure, gated by finite differences of the PRODUCTION objective.
//!
//! Every derivative here is checked against a difference quotient of code that
//! is on the fit path — `build_empirical_z_grid_with_alpha` for the measure and
//! `empirical_intercept` + `signed_probit_logcdf_and_mills_ratio`
//! for the row. Nothing is checked against a reimplementation of the formula it
//! is testing, because a probe that reconstructs the object it measures is
//! measuring its own reconstruction.
//!
//! The ladder, cheapest first:
//!
//! 1. the recorded allocation is the fill loop's own (mass conservation, both
//!    margins);
//! 2. `D = ∂node/∂ζ` against a central difference of the production builder;
//! 3. the tie certificate fires exactly where the compression stops being
//!    differentiable and nowhere else;
//! 4. the TOTAL `∂²(log L)/∂β∂ζ_j` — the number the seam actually consumes —
//!    against a double central difference of the production log-likelihood with
//!    the grid REBUILT at every perturbed `ζ`. That last one is the acceptance
//!    gate: it is blind to how the channels are split and would fail on an IFT
//!    sign error, a missing cross-row term, or a wrong `1/sd`.

#![cfg(test)]

use super::empirical_measure_sensitivity::{
    build_empirical_z_grid_with_alpha, rigid_empirical_score_zeta_channels,
};
use super::gradient_paths::empirical_intercept;
use crate::probability::signed_probit_logcdf_and_mills_ratio;
use gam_problem::{InverseLink, StandardLink};
use ndarray::{Array1, Array2};

const PROBIT: InverseLink = InverseLink::Standard(StandardLink::Probit);

/// A deterministic fixture that exercises the two structural cases a naive
/// implementation gets wrong: a row heavier than the per-bin target (so it
/// spans several bins and appears in `α` more than twice), and a zero-weight
/// row (filtered out of the compression entirely, so its column of `D` must be
/// exactly zero).
fn zeta_and_weights() -> (Array1<f64>, Array1<f64>) {
    let zeta = Array1::from(vec![
        -1.83, -1.10, -0.74, -0.31, -0.05, 0.17, 0.42, 0.68, 0.95, 1.31, 1.77, 2.40,
    ]);
    // Total weight 14.9 over 5 bins ⇒ per-bin target 2.98. The 6.0-weight row
    // is more than TWICE that and starts mid-bin, so it spans three bins and
    // appears in `α` three times — the case the "at most two nonzeros per
    // column" claim in the original design would have got wrong.
    let weights = Array1::from(vec![
        0.7, 1.2, 0.9, 6.0, 0.0, 1.1, 0.8, 1.4, 0.6, 0.9, 0.7, 0.6,
    ]);
    (zeta, weights)
}

const GRID_SIZE: usize = 5;

#[test]
fn recorded_allocation_conserves_mass_on_both_margins_2484() {
    let (zeta, weights) = zeta_and_weights();
    let build = build_empirical_z_grid_with_alpha(zeta.view(), weights.view(), GRID_SIZE, "test")
        .expect("grid builds");

    // Row margin: every positive-weight row's mass is fully allocated, and a
    // zero-weight row appears nowhere.
    let mut per_row = vec![0.0_f64; zeta.len()];
    for &(_, row, mass) in &build.alpha {
        per_row[row] += mass;
    }
    for (row, (&expected, &got)) in weights.iter().zip(per_row.iter()).enumerate() {
        assert!(
            (expected - got).abs() <= 1.0e-12 * (1.0 + expected),
            "gam#2484: row {row} allocated {got} of its weight {expected}"
        );
    }

    // Bin margin: the recorded per-bin mass is the sum of what landed in it.
    let mut per_bin = vec![0.0_f64; build.grid.nodes.len()];
    for &(node, _, mass) in &build.alpha {
        per_bin[node] += mass;
    }
    for (bin, (&expected, &got)) in build.bin_mass.iter().zip(per_bin.iter()).enumerate() {
        assert!(
            (expected - got).abs() <= 1.0e-12 * (1.0 + expected),
            "gam#2484: bin {bin} records mass {expected} but the allocation sums to {got}"
        );
    }

    // The heavy row must genuinely span more than two bins, or this fixture is
    // not testing the case it exists for.
    let heavy_entries = build.alpha.iter().filter(|&&(_, row, _)| row == 3).count();
    assert!(
        heavy_entries > 2,
        "gam#2484: the fixture's heavy row landed in {heavy_entries} bins; the >2-per-row case is \
         then untested"
    );
    assert!(
        build.standardization_sd.is_some(),
        "gam#2484: this fixture has a healthy spread, so the standardization must have run"
    );
}

#[test]
fn node_zeta_sensitivity_matches_central_fd_of_the_production_builder_2484() {
    let (zeta, weights) = zeta_and_weights();
    let build = build_empirical_z_grid_with_alpha(zeta.view(), weights.view(), GRID_SIZE, "test")
        .expect("grid builds");
    let m = build.grid.nodes.len();
    let n = zeta.len();

    // Column `b` of `Dᵀ` is `(∂x_b/∂ζ_i)_i`, so a one-hot right-hand side reads
    // one node's whole gradient.
    let mut analytic = Array2::<f64>::zeros((n, m));
    for b in 0..m {
        let mut e_b = Array2::<f64>::zeros((m, 1));
        e_b[[b, 0]] = 1.0;
        let column = build.node_zeta_vjp(e_b.view()).expect("differentiable");
        for i in 0..n {
            analytic[[i, b]] = column[[i, 0]];
        }
    }

    let h = 1.0e-6;
    for i in 0..n {
        let mut plus = zeta.clone();
        let mut minus = zeta.clone();
        plus[i] += h;
        minus[i] -= h;
        let grid_plus =
            build_empirical_z_grid_with_alpha(plus.view(), weights.view(), GRID_SIZE, "fd")
                .expect("perturbed grid builds")
                .grid;
        let grid_minus =
            build_empirical_z_grid_with_alpha(minus.view(), weights.view(), GRID_SIZE, "fd")
                .expect("perturbed grid builds")
                .grid;
        assert_eq!(
            grid_plus.nodes.len(),
            m,
            "the perturbation changed the node count"
        );
        for b in 0..m {
            let fd = (grid_plus.nodes[b] - grid_minus.nodes[b]) / (2.0 * h);
            let tol = 1.0e-6 * (1.0 + fd.abs());
            assert!(
                (analytic[[i, b]] - fd).abs() <= tol,
                "gam#2484: ∂node[{b}]/∂ζ[{i}] analytic {} vs central FD {}",
                analytic[[i, b]],
                fd
            );
        }
    }
}

#[test]
fn zero_weight_rows_get_an_exactly_zero_node_sensitivity_2484() {
    let (zeta, weights) = zeta_and_weights();
    let build = build_empirical_z_grid_with_alpha(zeta.view(), weights.view(), GRID_SIZE, "test")
        .expect("grid builds");
    let m = build.grid.nodes.len();
    let ones = Array2::<f64>::from_elem((m, 1), 1.0);
    let pulled = build.node_zeta_vjp(ones.view()).expect("differentiable");
    assert_eq!(pulled.nrows(), zeta.len());
    for (row, &weight) in weights.iter().enumerate() {
        if weight == 0.0 {
            assert_eq!(
                pulled[[row, 0]],
                0.0,
                "gam#2484: row {row} carries no weight, so it cannot move the measure; got {}",
                pulled[[row, 0]]
            );
        }
    }
}

#[test]
fn the_tie_certificate_fires_only_where_a_boundary_cuts_a_tie_2484() {
    // Two rows tied at 0.5 with unit weights, four bins over total weight 6 ⇒
    // per-bin target 1.5, so a boundary at 3.0 falls in the middle of the tied
    // pair, which spans (2.0, 4.0).
    let zeta = Array1::from(vec![-2.0, -1.0, 0.5, 0.5, 1.0, 2.0]);
    let weights = Array1::from(vec![1.0, 1.0, 1.0, 1.0, 1.0, 1.0]);
    let straddling = build_empirical_z_grid_with_alpha(zeta.view(), weights.view(), 4, "tie")
        .expect("grid builds");
    let tie = straddling
        .tie_straddle
        .as_ref()
        .expect("gam#2484: a boundary cutting a tied pair must be certified");
    assert_eq!(tie.rows, 2);
    assert!((tie.value - 0.5).abs() < 1.0e-12);
    assert!(
        straddling
            .node_zeta_vjp(Array2::<f64>::zeros((straddling.grid.nodes.len(), 1)).view())
            .is_err(),
        "gam#2484: a certified tie straddle must refuse the derivative, not return one"
    );

    // The same tie at three bins: per-bin target 2.0, so the interior
    // boundaries at 2.0 and 4.0 land exactly on the tied pair's own edges. The
    // whole tie then sits inside one bin, every member contributes its full
    // weight to it whatever order the sort chose, and the allocation is
    // order-invariant.
    let inside =
        build_empirical_z_grid_with_alpha(zeta.view(), weights.view(), 3, "tie").expect("builds");
    assert!(
        inside.tie_straddle.is_none(),
        "gam#2484: a tie that lies inside a single bin is order-invariant and must NOT refuse; \
         got {:?}",
        inside.tie_straddle
    );
    assert!(
        inside
            .node_zeta_vjp(Array2::<f64>::zeros((inside.grid.nodes.len(), 1)).view())
            .is_ok()
    );

    // And a sample with no ties at all is unconditionally differentiable.
    let (untied_z, untied_w) = zeta_and_weights();
    assert!(
        build_empirical_z_grid_with_alpha(untied_z.view(), untied_w.view(), GRID_SIZE, "tie")
            .expect("builds")
            .tie_straddle
            .is_none()
    );
}

// ---------------------------------------------------------------------------
// The acceptance gate: the TOTAL mixed derivative, against the production
// objective with the grid rebuilt at every perturbed ζ.
// ---------------------------------------------------------------------------

/// A two-block rigid empirical-grid BMS fixture: `n` rows, a `p_m`-column
/// marginal design and a `p_g`-column slope design, so `β = [β_m; β_g]`.
struct RowFixture {
    marginal_design: Array2<f64>,
    slope_design: Array2<f64>,
    y: Array1<f64>,
    weights: Array1<f64>,
    zeta: Array1<f64>,
    grid_weights: Array1<f64>,
    beta: Array1<f64>,
    probit_scale: f64,
}

impl RowFixture {
    fn new() -> Self {
        let n = 10;
        let mut marginal_design = Array2::<f64>::zeros((n, 2));
        let mut slope_design = Array2::<f64>::zeros((n, 2));
        let mut y = Array1::<f64>::zeros(n);
        let mut zeta = Array1::<f64>::zeros(n);
        let mut weights = Array1::<f64>::zeros(n);
        let mut grid_weights = Array1::<f64>::zeros(n);
        for i in 0..n {
            let t = (i as f64) / (n as f64 - 1.0);
            marginal_design[[i, 0]] = 1.0;
            marginal_design[[i, 1]] = 2.0 * t - 1.0;
            slope_design[[i, 0]] = 1.0;
            slope_design[[i, 1]] = (3.0 * t - 1.4).sin();
            y[i] = if i % 3 == 0 { 1.0 } else { 0.0 };
            // Deliberately untied and irregularly spaced.
            zeta[i] = -1.7 + 0.41 * (i as f64) + 0.07 * ((i * i) as f64).sqrt();
            weights[i] = 0.6 + 0.13 * ((i % 4) as f64);
            grid_weights[i] = weights[i];
        }
        Self {
            marginal_design,
            slope_design,
            y,
            weights,
            zeta,
            grid_weights,
            beta: Array1::from(vec![0.15, -0.42, 0.55, 0.23]),
            probit_scale: 0.87,
        }
    }

    fn p_m(&self) -> usize {
        self.marginal_design.ncols()
    }

    fn p_beta(&self) -> usize {
        self.marginal_design.ncols() + self.slope_design.ncols()
    }

    fn linear_predictors(&self, beta: &Array1<f64>) -> (Array1<f64>, Array1<f64>) {
        let p_m = self.p_m();
        let marginal = self.marginal_design.dot(&beta.slice(ndarray::s![..p_m]));
        let slope = self.slope_design.dot(&beta.slice(ndarray::s![p_m..]));
        (marginal, slope)
    }

    /// The production log-likelihood of the rigid empirical-grid kernel at
    /// `(β, ζ)`, WITH the latent measure rebuilt from `ζ` — which is the whole
    /// point: `ζ` reaches the objective through the row AND through the grid.
    fn log_likelihood(&self, beta: &Array1<f64>, zeta: &Array1<f64>) -> f64 {
        let grid = build_empirical_z_grid_with_alpha(
            zeta.view(),
            self.grid_weights.view(),
            GRID_SIZE,
            "fd objective",
        )
        .expect("grid builds")
        .grid;
        let (marginal_eta, slope_eta) = self.linear_predictors(beta);
        let mut total = 0.0;
        for i in 0..zeta.len() {
            let marginal = super::family::bernoulli_marginal_link_map(&PROBIT, marginal_eta[i])
                .expect("link map");
            let a = empirical_intercept(
                marginal.q,
                slope_eta[i],
                self.probit_scale,
                &grid.nodes,
                &grid.weights,
            )
            .expect("intercept solves");
            let e = a + self.probit_scale * slope_eta[i] * zeta[i];
            let (logcdf, _) = signed_probit_logcdf_and_mills_ratio((2.0 * self.y[i] - 1.0) * e);
            total += self.weights[i] * logcdf;
        }
        total
    }

    /// `S_eff` — the total per-row sensitivity the seam feeds to the
    /// Murphy–Topel chain: the direct channel plus the grid channel pulled back
    /// through `D`.
    fn analytic_total_sensitivity(&self) -> Array2<f64> {
        let build = build_empirical_z_grid_with_alpha(
            self.zeta.view(),
            self.grid_weights.view(),
            GRID_SIZE,
            "analytic",
        )
        .expect("grid builds");
        let (marginal_eta, slope_eta) = self.linear_predictors(&self.beta);
        let channels = rigid_empirical_score_zeta_channels(
            &PROBIT,
            &marginal_eta,
            &slope_eta,
            &self.zeta,
            &self.y,
            &self.weights,
            self.probit_scale,
            &build.grid,
            self.marginal_design.view(),
            self.slope_design.view(),
            self.p_beta(),
        )
        .expect("channels");
        let cross_row = build
            .node_zeta_vjp(channels.node.view())
            .expect("differentiable");
        channels.direct + cross_row
    }
}

#[test]
fn the_total_zeta_sensitivity_matches_a_double_central_fd_of_the_objective_2484() {
    let fixture = RowFixture::new();
    let analytic = fixture.analytic_total_sensitivity();
    let p_beta = fixture.p_beta();
    let n = fixture.zeta.len();

    let h_beta = 1.0e-4;
    let h_zeta = 1.0e-4;
    let mut worst = 0.0_f64;
    for j in 0..n {
        for k in 0..p_beta {
            let mut value = 0.0;
            for (sign_beta, sign_zeta) in [(1.0, 1.0), (-1.0, -1.0), (1.0, -1.0), (-1.0, 1.0)] {
                let mut beta = fixture.beta.clone();
                let mut zeta = fixture.zeta.clone();
                beta[k] += sign_beta * h_beta;
                zeta[j] += sign_zeta * h_zeta;
                value += sign_beta * sign_zeta * fixture.log_likelihood(&beta, &zeta);
            }
            let fd = value / (4.0 * h_beta * h_zeta);
            let error = (analytic[[j, k]] - fd).abs();
            let tol = 5.0e-5 * (1.0 + fd.abs());
            worst = worst.max(error / (1.0 + fd.abs()));
            assert!(
                error <= tol,
                "gam#2484: ∂²(log L)/∂β[{k}]∂ζ[{j}] analytic {} vs double central FD {} \
                 (|Δ| = {error:.3e})",
                analytic[[j, k]],
                fd
            );
        }
    }
    assert!(worst.is_finite());
}

#[test]
fn the_standardization_annihilates_location_and_scale_2484() {
    // An exact structural property of `D`, and the reason the cross-row channel
    // is smaller in the published standard error than its matrix norm suggests
    // (measured in `scripts/probe_2484_empirical_measure_sensitivity.py`).
    //
    // The grid is STANDARDIZED: its nodes carry weighted mean 0 and weighted sd
    // 1 by construction. So no perturbation of `ζ` can move the measure's
    // location or its scale — only its SHAPE. Two directions witness that
    // exactly:
    //
    //   * shifting every `ζ_i` by the same δ shifts every RAW node by δ and
    //     leaves the standardized nodes fixed  ⇒  `D·1 = 0`;
    //   * scaling every `ζ_i` by `1 + ε` scales every raw node the same way and
    //     again leaves the standardized nodes fixed  ⇒  `D·ζ = 0`.
    //
    // These are identities, not tolerances, so they hold to machine precision
    // and would break under a dropped `−w_c·Σ v` or `−w_c·x_c·Σ x v` term in
    // the `M` factor — the two terms a hand-rolled `∂x/∂n` most easily loses.
    let (zeta, weights) = zeta_and_weights();
    let build = build_empirical_z_grid_with_alpha(zeta.view(), weights.view(), GRID_SIZE, "test")
        .expect("grid builds");
    let m = build.grid.nodes.len();
    let n = zeta.len();
    let mut d_transpose = Array2::<f64>::zeros((n, m));
    for b in 0..m {
        let mut e_b = Array2::<f64>::zeros((m, 1));
        e_b[[b, 0]] = 1.0;
        let column = build.node_zeta_vjp(e_b.view()).expect("differentiable");
        for i in 0..n {
            d_transpose[[i, b]] = column[[i, 0]];
        }
    }
    for b in 0..m {
        let shift: f64 = (0..n).map(|i| d_transpose[[i, b]]).sum();
        let rescale: f64 = (0..n).map(|i| d_transpose[[i, b]] * zeta[i]).sum();
        assert!(
            shift.abs() <= 1.0e-10,
            "gam#2484: a uniform shift of ζ must not move standardized node {b}; got {shift:.3e}"
        );
        assert!(
            rescale.abs() <= 1.0e-10,
            "gam#2484: a uniform rescale of ζ must not move standardized node {b}; got \
             {rescale:.3e}"
        );
    }
}

#[test]
fn the_cross_row_channel_is_a_first_class_part_of_the_total_2484() {
    // The direct channel alone is what the standard-normal branch would have
    // supplied. If the grid channel were negligible the whole issue would be a
    // rounding error, so assert it is not: it must be a large fraction of the
    // total on this ordinary fixture.
    //
    // This is a claim about the SENSITIVITY MATRIX, which is what this file can
    // see. It is deliberately NOT a claim about the published standard error:
    // three contractions (`G = S_effᵀJ`, `Vb·G`, the `V₁` congruence) sit
    // between the two, and the sweep in
    // `scripts/probe_2484_empirical_measure_sensitivity.py` measures the SE
    // effect at 1.2e-5 to 9.0e-3 relative, growing with the SLOPE rather
    // than with the grid size. Small enough that a direct-only implementation
    // would pass casual inspection, which is the reason to be exact rather than
    // a reason not to bother.
    let fixture = RowFixture::new();
    let build = build_empirical_z_grid_with_alpha(
        fixture.zeta.view(),
        fixture.grid_weights.view(),
        GRID_SIZE,
        "split",
    )
    .expect("grid builds");
    let (marginal_eta, slope_eta) = fixture.linear_predictors(&fixture.beta);
    let channels = rigid_empirical_score_zeta_channels(
        &PROBIT,
        &marginal_eta,
        &slope_eta,
        &fixture.zeta,
        &fixture.y,
        &fixture.weights,
        fixture.probit_scale,
        &build.grid,
        fixture.marginal_design.view(),
        fixture.slope_design.view(),
        fixture.p_beta(),
    )
    .expect("channels");
    let cross_row = build
        .node_zeta_vjp(channels.node.view())
        .expect("differentiable");
    let direct_norm = channels.direct.iter().map(|v| v * v).sum::<f64>().sqrt();
    let cross_norm = cross_row.iter().map(|v| v * v).sum::<f64>().sqrt();
    assert!(
        cross_norm > 0.05 * direct_norm,
        "gam#2484: the cross-row channel is {cross_norm:.6e} against a direct channel of \
         {direct_norm:.6e}; if it were really negligible the correction would not need it"
    );
}

// ---------------------------------------------------------------------------
// Which fits get the correction, and which are still withheld
// ---------------------------------------------------------------------------

use super::LatentMeasureKind;
use super::empirical_measure_sensitivity::{
    EmpiricalGeneratedRegressorChannel, classify_empirical_generated_regressor_channel,
};

fn clean_build() -> super::empirical_measure_sensitivity::EmpiricalZGridBuild {
    let (zeta, weights) = zeta_and_weights();
    build_empirical_z_grid_with_alpha(zeta.view(), weights.view(), GRID_SIZE, "classify")
        .expect("grid builds")
}

fn assert_unavailable_because(
    channel: EmpiricalGeneratedRegressorChannel<'_>,
    measure: &str,
    needle: &str,
) {
    match channel {
        EmpiricalGeneratedRegressorChannel::Unavailable {
            latent_measure,
            unavailable_channel,
        } => {
            assert_eq!(latent_measure, measure);
            assert!(
                unavailable_channel.contains(needle),
                "gam#2484: the withholding must say WHICH channel is missing; expected a mention \
                 of {needle:?}, got: {unavailable_channel}"
            );
        }
        EmpiricalGeneratedRegressorChannel::ClosedForm => {
            panic!("gam#2484: classified as the closed-form kernel, which it is not")
        }
        EmpiricalGeneratedRegressorChannel::Empirical(_) => panic!(
            "gam#2484: classified as correctable. The correction would then be computed from a \
             channel that does not describe this fit, which is worse than withholding."
        ),
    }
}

#[test]
fn the_ordinary_rigid_global_empirical_fit_is_correctable_2484() {
    let build = clean_build();
    let measure = LatentMeasureKind::GlobalEmpirical {
        grid: build.grid.clone(),
    };
    assert!(
        matches!(
            classify_empirical_generated_regressor_channel(&measure, Some(&build), false),
            EmpiricalGeneratedRegressorChannel::Empirical(_)
        ),
        "gam#2484: this is the state the issue is about, and it is the one that must now be \
         corrected rather than declined"
    );
    assert!(
        matches!(
            classify_empirical_generated_regressor_channel(
                &LatentMeasureKind::StandardNormal,
                None,
                false
            ),
            EmpiricalGeneratedRegressorChannel::ClosedForm
        ),
        "gam#2484: the standard-normal branch must be untouched"
    );
}

#[test]
fn the_shapes_with_no_channel_are_still_withheld_and_say_which_2484() {
    let build = clean_build();
    let global = LatentMeasureKind::GlobalEmpirical {
        grid: build.grid.clone(),
    };

    // A score-warp / link-deviation block sees the latent score through a basis
    // as well as through the calibration intercept.
    assert_unavailable_because(
        classify_empirical_generated_regressor_channel(&global, Some(&build), true),
        "global-empirical",
        "score-warp",
    );

    // A measure with no build record: the allocation behind its nodes is
    // unknown, so `D` cannot be formed. Correcting anyway would mean inventing
    // an allocation.
    assert_unavailable_because(
        classify_empirical_generated_regressor_channel(&global, None, false),
        "global-empirical",
        "no build record",
    );

    // Data on which the compression is genuinely non-differentiable.
    let tied_z = Array1::from(vec![-2.0, -1.0, 0.5, 0.5, 1.0, 2.0]);
    let tied_w = Array1::from(vec![1.0; 6]);
    let tied = build_empirical_z_grid_with_alpha(tied_z.view(), tied_w.view(), 4, "tie")
        .expect("grid builds");
    let tied_measure = LatentMeasureKind::GlobalEmpirical {
        grid: tied.grid.clone(),
    };
    assert_unavailable_because(
        classify_empirical_generated_regressor_channel(&tied_measure, Some(&tied), false),
        "global-empirical",
        "not differentiable",
    );

    // A per-row local-empirical measure, which only a deserialized model has.
    let local = LatentMeasureKind::LocalEmpirical {
        feature_cols: vec![0],
        input_scales: None,
        centers: vec![vec![0.0], vec![1.0]],
        grids: vec![build.grid.clone(), build.grid.clone()],
        top_k: 1,
        bandwidth: 1.0,
        mixture: crate::bms::LocalLawMixture::default(),
        train_row_mixtures: std::sync::Arc::new(Vec::new()),
    };
    assert_unavailable_because(
        classify_empirical_generated_regressor_channel(&local, Some(&build), false),
        "local-empirical",
        "no fit-time build record",
    );
}

#[test]
fn the_assembled_correction_is_psd_and_strictly_widens_the_naive_interval_2484() {
    // The statistical content of the whole issue: the naive second-stage
    // covariance treats `ζ` as known, so it omits the first-stage uncertainty
    // and is TOO NARROW. The correction's job is to add that back, and a
    // correction that came out zero (or indefinite) would be worse than
    // withholding, because it reads as corrected on the wire.
    let fixture = RowFixture::new();
    let s_eff = fixture.analytic_total_sensitivity();
    let p_beta = fixture.p_beta();

    // A first-stage calibration on a one-column conditioning basis, with a PSD
    // joint covariance. The numbers are a fixture, not a fit; what is under
    // test is the congruence the seam performs with them.
    let basis_ncols = 1;
    let theta1_dim = 2 * (basis_ncols + 1);
    let mut theta1_cov = Array2::<f64>::zeros((theta1_dim, theta1_dim));
    for i in 0..theta1_dim {
        for j in 0..theta1_dim {
            // A Kac-Murdock-Szego correlation, scaled: symmetric, strictly PD.
            theta1_cov[[i, j]] = 0.04 * 0.6_f64.powi((i as i32 - j as i32).abs());
        }
    }
    let calibration = super::LatentZConditionalCalibration {
        mean_coeffs: vec![0.05, 0.31],
        var_coeffs: vec![0.9, 0.12],
        basis_ncols,
        var_floor: 1.0e-6,
        homoskedastic_var: 0.9,
        post_mean: 0.0,
        post_sd: 1.0,
        theta1_cov,
    };
    let a_block = Array2::<f64>::from_shape_fn((fixture.zeta.len(), basis_ncols), |(i, _)| {
        ((i as f64) - 4.5) / 4.5
    });
    let raw_z = Array1::from_shape_fn(fixture.zeta.len(), |i| 0.4 * fixture.zeta[i] + 0.05);
    let vb =
        Array2::<f64>::from_shape_fn((p_beta, p_beta), |(i, j)| if i == j { 0.25 } else { 0.02 });

    let correction = calibration
        .generated_regressor_correction(s_eff.view(), raw_z.view(), a_block.view(), vb.view())
        .expect("the correction assembles");

    let mut max_diagonal = 0.0_f64;
    for i in 0..p_beta {
        max_diagonal = max_diagonal.max(correction[[i, i]]);
        assert!(
            correction[[i, i]] >= -1.0e-14,
            "gam#2484: the correction is a congruence of a PSD V₁, so no diagonal may be \
             negative; got {} at {i}",
            correction[[i, i]]
        );
        for j in 0..p_beta {
            assert!(
                (correction[[i, j]] - correction[[j, i]]).abs() <= 1.0e-12,
                "gam#2484: the correction must be symmetric at ({i},{j})"
            );
        }
    }
    assert!(
        max_diagonal > 1.0e-8,
        "gam#2484: the correction must actually widen the intervals; its largest diagonal is \
         {max_diagonal:.6e}, which is indistinguishable from publishing the naive covariance"
    );

    // PSD in every direction, not just on the diagonal: a congruence cannot be
    // indefinite, and an implementation that lost the symmetry of `V₁` could be.
    for k in 0..p_beta {
        for l in 0..p_beta {
            let mut direction = Array1::<f64>::zeros(p_beta);
            direction[k] += 1.0;
            direction[l] -= 1.0;
            let quadratic = direction.dot(&correction.dot(&direction));
            assert!(
                quadratic >= -1.0e-12,
                "gam#2484: the correction is indefinite along e_{k} − e_{l}: {quadratic:.6e}"
            );
        }
    }
}

#[test]
fn the_batched_directional_chunk_size_never_inverts_its_clamp() {
    // Not a gam#2484 finding by intent — a hard PANIC found by gam#2484's
    // score-warp arm, which is the first fixture on this path with fewer rows
    // than the amortization floor.
    //
    // `batched_directional_derivative_chunk_rows` clamped a budget-derived row
    // count into `[1024, n]`, and `usize::clamp` panics when `min > max`. Every
    // BMS fit below 1024 rows that reached either `dense_contiguous_rows`
    // batched directional-derivative call site aborted the process with
    // `min > max. min = 1024, max = <n>`.
    //
    // The contract, at every size: at least one row, never more than `n`, and
    // never a panic.
    for &n in &[0usize, 1, 2, 63, 96, 1023, 1024, 1025, 50_000] {
        for &n_dirs in &[1usize, 3, 17, 4096] {
            let (rows, gpu) =
                super::family::BernoulliMarginalSlopeFamily::batched_directional_derivative_chunk_rows(
                    n, n_dirs,
                );
            assert!(
                rows >= 1,
                "chunk size must be at least one row at n={n}, n_dirs={n_dirs}; got {rows}"
            );
            assert!(
                rows <= n.max(1),
                "chunk size {rows} exceeds the row count at n={n}, n_dirs={n_dirs}"
            );
            assert!(!gpu, "there is no live device backend in this build");
        }
    }
}
