//! Joint co-fitting of the linear block tier and the curved chart tier
//! (residual-orthogonality trap closure).
//!
//! # The trap this closes
//!
//! The block-chart compose lane ([`super::block_chart`]) fits curved charts to
//! the **least-squares residual** of a frozen linear dictionary. But an LS
//! residual is orthogonal to the fitted span: the linear tiling has already
//! absorbed the local tangent *and* the curvature into where it placed its
//! atoms, so what is left in the residual is high-frequency sawtooth
//! quantisation noise between atoms — exactly the thing a *smooth* chart cannot
//! represent. The one-shot fit-curved-on-linear-residual protocol therefore
//! hands the curved lane a target from which the very structure it is meant to
//! find has been removed by construction.
//!
//! # The fix: monotone two-block coordinate descent
//!
//! Model the reconstruction as two **additive** tiers,
//! `x̂ = L(codes) + C(charts)`, and alternate two block solves that both descend
//! the SAME penalised joint objective
//!
//! ```text
//!   J(codes, charts) = ‖target − L − C‖²_F  +  λ_lin · ‖codes‖²_F
//! ```
//!
//! (the linear tier's ridge is explicit in `J`; the curved tier's complexity
//! penalty is realised as the compose lane's cross-fit BIC acceptance charge,
//! which admits a chart only when its cross-validated deviance gain exceeds its
//! `½·d_eff·log n_eff` information charge — a descriptive per-chart BIC gate, not
//! an FDR-controlled e-BH discovery — surfaced per round as
//! [`CofitRound::curved_charge`]).
//!
//! * **Block A — linear tier refit.** With the charts (hence `C`) held fixed and
//!   the block routing frozen, re-solve the per-row active-set ridge
//!   least-squares codes against the *chart-adjusted* target `target − C`. This
//!   is an exact block minimisation of `J` over the linear codes (the previous
//!   codes are always feasible), so it is **provably monotone**: `J` cannot
//!   increase. It is precisely the step the one-shot protocol never takes — the
//!   linear tier stops chasing the curvature that the chart already explains, so
//!   its atoms are freed to model the genuinely linear part.
//! * **Block B — curved joint fit.** With the linear codes held fixed, re-fit the
//!   charts against the *linear-adjusted* target `target − L` through the
//!   existing curved surface ([`compose_block_coordinate_charts`]). The compose
//!   lane's acceptance is cross-fit gated rather than a pure held-in minimiser,
//!   so this step is **guarded**: the candidate is committed only when it does
//!   not increase `J`. The previous chart set is always available as the
//!   fallback, so the round is monotone by construction either way.
//!
//! Each committed round therefore has `J[r] ≤ J[r-1]` up to numerical
//! tolerance. Convergence requires an entire deterministic A/B replay to leave
//! codes, chart ownership, and both reconstruction components bit-identical; an
//! objective stall alone never mints a fit. The curved solver's internals are
//! **untouched** — it is called through its existing public surface with an
//! adjusted target.

use ndarray::{Array2, Array3};

use super::block_chart::{BlockChartComposeConfig, BlockChartComposeResult};

/// Configuration for [`cofit_block_and_curved`].
#[derive(Clone, Debug)]
pub struct CofitConfig {
    /// Maximum number of complete deterministic A/B replays. Exhaustion is a
    /// non-convergence error; a [`CofitReport`] is created only after one replay
    /// leaves the complete fitted state bit-identical.
    pub max_rounds: usize,
    /// Linear-tier ridge `λ_lin` on the per-row active-set least-squares codes.
    pub code_ridge: f32,
    /// Relative slack for the monotone-non-increase invariant. A round whose
    /// objective exceeds the previous by more than
    /// `monotone_slack · (|J_prev| + 1)` is a bug and aborts the fit.
    pub monotone_slack: f64,
    /// Curved-tier compose configuration. Its `block_size`, `block_topk` and
    /// `gamma` are overwritten from the passed routing/frames each round so the
    /// tiers always agree on geometry; `residual_target` is forced on (the
    /// co-fit *is* the principled residual protocol).
    pub chart: BlockChartComposeConfig,
}

impl Default for CofitConfig {
    fn default() -> Self {
        Self {
            max_rounds: 256,
            code_ridge: 1.0e-6,
            monotone_slack: 1.0e-6,
            chart: BlockChartComposeConfig::default(),
        }
    }
}

/// Per-round telemetry of the co-fit alternation.
#[derive(Clone, Debug)]
pub struct CofitRound {
    /// Round index (`0` is the one-shot fit-curved-on-linear-residual baseline;
    /// `≥1` are A/B alternation rounds).
    pub round: usize,
    /// Joint objective `J = ‖target − (L+C)‖²_F + λ_lin‖codes‖²_F` at round end.
    pub objective: f64,
    /// Reconstruction term `‖target − (L+C)‖²_F` (Frobenius SSE).
    pub recon_sse: f64,
    /// Linear-tier ridge energy `λ_lin · ‖codes‖²_F`.
    pub linear_ridge: f64,
    /// Total BIC complexity charge (`Σ ½·d_eff·log n_eff`) of the descriptively
    /// accepted charts this round — the curved tier's information penalty,
    /// enforced as a per-chart BIC acceptance gate (not an FDR-controlled e-BH
    /// discovery).
    pub curved_charge: f64,
    /// Composed explained variance (`1 − RSS/TSS`, mean baseline).
    pub explained_variance: f64,
    /// Number of accepted curved charts (single blocks + pairs) this round.
    pub n_accepted_charts: usize,
    /// Whether the linear block A step strictly reduced the objective this round
    /// (always a non-increase; `false` when it was already at the block optimum).
    pub linear_improved: bool,
    /// Whether the curved block B candidate was committed (`false` = the guard
    /// kept the previous chart set because the candidate did not reduce `J`).
    pub curved_committed: bool,
}

/// Result of a co-fit run.
#[derive(Clone, Debug)]
pub struct CofitReport {
    /// Composed reconstruction `L + C`, `N×P`.
    pub reconstructed: Array2<f32>,
    /// Linear-tier reconstruction `L` over the chart-*unowned* blocks, `N×P`
    /// (the blocks a chart replaced are excluded — they live in `C`).
    pub linear_reconstruction: Array2<f32>,
    /// Additive curved correction `C = composed − L`, `N×P` (the lifted chart
    /// coordinates of the accepted, chart-owned blocks).
    pub curved_correction: Array2<f32>,
    /// Refit linear-tier codes, `N×k×b`, at the frozen routing.
    pub codes: Array3<f32>,
    /// Final composed explained variance.
    pub explained_variance: f64,
    /// Per-round telemetry (index 0 is the one-shot baseline).
    pub rounds: Vec<CofitRound>,
    /// Final curved compose result (chart records, acceptances, screens).
    pub compose: BlockChartComposeResult,
}

#[cfg(test)]
mod cofit_tests {
    use super::*;
    use crate::sparse_dict::reconstruct_block_sparse_rows;

    /// Orthonormal-per-block decoder for the planted trap. Three blocks of size
    /// b=2 in P=5:
    ///   block 0 = {e0, e1}          (linear, disjoint span)
    ///   block 1 = {(e1+e2)/√2, (e1−e2)/√2}  (linear, OVERLAPS block 0 on e1)
    ///   block 2 = {e3, e4}          (the planted circle's plane)
    /// The block-0 / block-1 overlap on e1 makes the tied per-block projection
    /// codes suboptimal (they double-count e1), so the block-A least-squares
    /// refit strictly improves the linear tier.
    fn planted_decoder() -> Array2<f32> {
        let s = 1.0f32 / 2.0f32.sqrt();
        // Rows are atoms; 6 atoms × 5 dims.
        Array2::from_shape_vec(
            (6, 5),
            vec![
                1.0, 0.0, 0.0, 0.0, 0.0, // e0
                0.0, 1.0, 0.0, 0.0, 0.0, // e1
                0.0, s, s, 0.0, 0.0, // (e1+e2)/√2
                0.0, s, -s, 0.0, 0.0, // (e1−e2)/√2
                0.0, 0.0, 0.0, 1.0, 0.0, // e3
                0.0, 0.0, 0.0, 0.0, 1.0, // e4
            ],
        )
        .unwrap()
    }

    /// Planted data: a linear part in span{e0,e1,e2} plus a unit circle in the
    /// {e3,e4} plane, tiny noise. Returns (target, planted angles).
    fn planted_data(n: usize) -> (Array2<f32>, Vec<f64>) {
        let mut x = Array2::<f32>::zeros((n, 5));
        let mut theta = Vec::with_capacity(n);
        for i in 0..n {
            // Deterministic pseudo-random coefficients (no rng dependency).
            let a = ((i * 7 + 1) % 17) as f32 / 17.0 - 0.5;
            let bb = ((i * 13 + 5) % 19) as f32 / 19.0 - 0.5;
            let cc = ((i * 5 + 3) % 23) as f32 / 23.0 - 0.5;
            let t = 2.0 * std::f64::consts::PI * (i as f64) / (n as f64);
            let noise = 0.002 * (((i * 3) % 11) as f32 / 11.0 - 0.5);
            x[[i, 0]] = a;
            x[[i, 1]] = bb;
            x[[i, 2]] = cc;
            x[[i, 3]] = t.cos() as f32 + noise;
            x[[i, 4]] = t.sin() as f32 - noise;
            theta.push(t);
        }
        (x, theta)
    }

    /// Tied per-block routing: every row fires all three blocks; the within-block
    /// code is the tied projection `z_g = x D_gᵀ` (γ = 1). This is exactly what a
    /// converged block-sparse fit stores.
    fn tied_routing(
        x: &Array2<f32>,
        decoder: &Array2<f32>,
        b: usize,
    ) -> (Array2<u32>, Array3<f32>) {
        let n = x.nrows();
        let g = decoder.nrows() / b;
        let mut blocks = Array2::<u32>::zeros((n, g));
        let mut codes = Array3::<f32>::zeros((n, g, b));
        for i in 0..n {
            for gg in 0..g {
                blocks[[i, gg]] = gg as u32;
                for r in 0..b {
                    let atom = decoder.row(gg * b + r);
                    let mut dot = 0.0f32;
                    for c in 0..decoder.ncols() {
                        dot += x[[i, c]] * atom[c];
                    }
                    codes[[i, gg, r]] = dot;
                }
            }
        }
        (blocks, codes)
    }

    fn ev(x: &Array2<f32>, recon: &Array2<f32>) -> f64 {
        explained_variance_from_reconstruction(x.view(), recon.view()).unwrap()
    }

    fn chart_cfg_small() -> BlockChartComposeConfig {
        BlockChartComposeConfig {
            block_size: 2,
            block_topk: 3,
            min_firings: 8,
            crossfit_folds: 4,
            pair_screen: false,
            ..BlockChartComposeConfig::default()
        }
    }

    #[test]
    fn cofit_beats_one_shot_and_recovers_angle() {
        let n = 240;
        let decoder = planted_decoder();
        let (x, theta) = planted_data(n);
        let (blocks, codes) = tied_routing(&x, &decoder, 2);

        let config = CofitConfig {
            code_ridge: 1.0e-6,
            chart: chart_cfg_small(),
            ..CofitConfig::default()
        };
        let report = cofit_block_and_curved(
            x.view(),
            decoder.view(),
            blocks.view(),
            codes.view(),
            1.0,
            &config,
        )
        .expect("cofit runs");

        // (a) The co-fit's composed reconstruction beats the one-shot baseline
        //     (round 0) on explained variance — the linear tier, freed of the
        //     curvature the chart explains, stops double-counting e1.
        let one_shot_ev = report.rounds[0].explained_variance;
        assert!(
            report.explained_variance > one_shot_ev + 1.0e-4,
            "co-fit EV {} should beat one-shot EV {}",
            report.explained_variance,
            one_shot_ev
        );

        // A curved chart must actually have been accepted for block 2 (the plane
        // holding the circle) — otherwise this is a pure-linear result.
        assert!(
            report.rounds.last().unwrap().n_accepted_charts >= 1,
            "expected at least one accepted curved chart"
        );

        // (a, cont.) The curved atom's recovered coordinates correlate with the
        //     planted angle. The chart lives in {e3, e4}; read the recovered
        //     angle off the curved correction there. Use the rotation-invariant
        //     complex correlation magnitude to allow a global chart-frame gauge.
        let c = &report.curved_correction;
        let mut re = 0.0f64;
        let mut im = 0.0f64;
        let mut mag = 0.0f64;
        for i in 0..n {
            let hat = (c[[i, 4]] as f64).atan2(c[[i, 3]] as f64);
            let d = theta[i] - hat;
            re += d.cos();
            im += d.sin();
            mag += 1.0;
        }
        let rho = (re * re + im * im).sqrt() / mag.max(1.0);
        assert!(
            rho > 0.9,
            "recovered circle angle should track the planted angle (ρ={rho})"
        );
    }

    #[test]
    fn objective_is_monotone_across_rounds() {
        let n = 240;
        let decoder = planted_decoder();
        let (x, _theta) = planted_data(n);
        let (blocks, codes) = tied_routing(&x, &decoder, 2);

        let config = CofitConfig {
            chart: chart_cfg_small(),
            ..CofitConfig::default()
        };
        let report = cofit_block_and_curved(
            x.view(),
            decoder.view(),
            blocks.view(),
            codes.view(),
            1.0,
            &config,
        )
        .expect("cofit runs");

        assert!(report.rounds.len() >= 2, "expected multiple rounds");
        let slack = config.monotone_slack;
        for w in report.rounds.windows(2) {
            let prev = w[0].objective;
            let cur = w[1].objective;
            assert!(
                cur <= prev + slack * (prev.abs() + 1.0),
                "objective rose from {prev} to {cur}"
            );
        }
    }

    #[test]
    fn insufficient_rounds_return_error_instead_of_an_open_cofit_2023() {
        let decoder = planted_decoder();
        let (x, _theta) = planted_data(240);
        let (blocks, codes) = tied_routing(&x, &decoder, 2);
        let config = CofitConfig {
            max_rounds: 1,
            chart: chart_cfg_small(),
            ..CofitConfig::default()
        };

        let error = cofit_block_and_curved(
            x.view(),
            decoder.view(),
            blocks.view(),
            codes.view(),
            1.0,
            &config,
        )
        .expect_err("a still-moving A/B replay must not mint CofitReport");
        assert!(
            error.contains("did not reach an idempotent fixed point"),
            "unexpected non-convergence error: {error}"
        );
    }

    #[test]
    fn empty_curved_tier_reproduces_pure_linear_fit() {
        let n = 200;
        let decoder = planted_decoder();
        let (x, _theta) = planted_data(n);
        let (blocks, codes) = tied_routing(&x, &decoder, 2);

        // Force the curved tier empty: select no blocks ⇒ no charts accepted.
        let mut chart = chart_cfg_small();
        chart.max_blocks = 0;
        let config = CofitConfig {
            code_ridge: 1.0e-6,
            chart,
            ..CofitConfig::default()
        };
        let report = cofit_block_and_curved(
            x.view(),
            decoder.view(),
            blocks.view(),
            codes.view(),
            1.0,
            &config,
        )
        .expect("cofit runs");

        // No charts anywhere.
        assert_eq!(report.rounds.last().unwrap().n_accepted_charts, 0);
        // The curved correction is identically zero.
        let max_c = report
            .curved_correction
            .iter()
            .fold(0.0f32, |m, &v| m.max(v.abs()));
        assert!(
            max_c < 1.0e-5,
            "curved correction should vanish (max {max_c})"
        );

        // The composed reconstruction equals an independent per-row least-squares
        // linear solve over all fired blocks (the pure linear fit).
        let b = 2usize;
        let mut ref_codes = Array3::<f32>::zeros((n, decoder.nrows() / b, b));
        for i in 0..n {
            let mut active: Vec<(u32, f32)> = Vec::new();
            for gg in 0..(decoder.nrows() / b) {
                for r in 0..b {
                    active.push(((gg * b + r) as u32, 0.0));
                }
            }
            let s = active.len();
            let solved = solve_row_codes(x.row(i), decoder.view(), &active, s, 1.0e-6);
            for (t, code) in solved.codes.iter().enumerate() {
                ref_codes[[i, t / b, t % b]] = *code;
            }
        }
        let ref_recon =
            reconstruct_block_sparse_rows(decoder.view(), blocks.view(), ref_codes.view(), b)
                .unwrap();
        let ev_ref = ev(&x, &ref_recon);
        assert!(
            (report.explained_variance - ev_ref).abs() < 1.0e-6,
            "empty-curved co-fit EV {} should match the pure linear LS fit EV {}",
            report.explained_variance,
            ev_ref
        );
        // And it must strictly beat the tied one-shot baseline (LS fixes the e1
        // double-count the tied projection introduced).
        assert!(report.explained_variance > report.rounds[0].explained_variance + 1.0e-4);
    }
}
