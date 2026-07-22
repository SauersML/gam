// #2155 inner-mode geography measurement harness (zz_measure diagnostics),
// split from `tests.rs` to respect the crate-wide source-file length budget.
// Same conventions: `super::*` resolves to the parent `gamlss` module's flat
// re-exports; fn-scoped imports carry the rest.

use super::*;
use ndarray::{Array1, Array2};

// =====================================================================
// #2155 inner-mode geography measurement harness (zz_measure diagnostics).
//
// The outer-optimizer half of #2155 mode (b) landed (6966b2a31 / 8739251b6 /
// ece539920); the residual blocker is the measured warm/cold inner-solve
// bimodality: a warm-started binomial mean-wiggle solve reaches a strictly
// lower mode than a cold solve at the same ρ, so the cold-reproducible
// terminal state is not the mode the search descended. These zz tests map the
// mode geography of the REAL #2155 fixture (600 rows, seed 2155, y ~ x with a
// flexible link) at FIXED wiggle log-λ, comparing:
//   (a) the production cold seed (pilot β, β_w = 0) jumped straight to the
//       target λ, against
//   (b) a deterministic warp-penalty continuation: the same solve reached
//       through a descending λ ladder anchored at the exact large-λ limit
//       (the pilot fit — see fit_orchestration/fit.rs on the wiggle model
//       containing the baseline as its large-λ limiting case).
// If (b) reaches a strictly lower penalized objective than (a) in a λ region,
// the graduated continuation is the correct canonical cold solve for this
// family and becomes the production fix.
// =====================================================================

fn zz2155_splitmix_u01(state: &mut u64) -> f64 {
    *state = state.wrapping_add(0x9e37_79b9_7f4a_7c15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    z ^= z >> 31;
    ((z >> 11) as f64 + 0.5) / (1u64 << 53) as f64
}

/// The exact #2155 fixture from
/// `tests/bug_hunt_flexible_loglog_cauchit_binomial_wiggle.rs`: x ~ U(-2,2),
/// p = logistic(0.8 x), y ~ Bernoulli(p), splitmix64 stream seeded at 2155.
fn zz2155_fixture(n: usize, seed: u64) -> (Array1<f64>, Array1<f64>) {
    let mut s = seed;
    let mut y = Vec::with_capacity(n);
    let mut x = Vec::with_capacity(n);
    for _ in 0..n {
        let xv = -2.0 + 4.0 * zz2155_splitmix_u01(&mut s);
        let p = 1.0 / (1.0 + (-(0.8 * xv)).exp());
        let yv = if zz2155_splitmix_u01(&mut s) < p { 1.0 } else { 0.0 };
        x.push(xv);
        y.push(yv);
    }
    (Array1::from(y), Array1::from(x))
}

/// Unpenalized 2-parameter binomial GLM pilot (intercept + slope) by expected-
/// Fisher scoring — the same estimand as the production no-wiggle pilot fit.
fn zz2155_pilot(
    y: &Array1<f64>,
    x: &Array1<f64>,
    link: &InverseLink,
) -> (Array1<f64>, Array1<f64>) {
    let n = y.len();
    let mut beta = Array1::<f64>::zeros(2);
    for _ in 0..80 {
        let mut a00 = 0.0;
        let mut a01 = 0.0;
        let mut a11 = 0.0;
        let mut b0 = 0.0;
        let mut b1 = 0.0;
        for i in 0..n {
            let eta = beta[0] + beta[1] * x[i];
            let jet = inverse_link_jet_for_inverse_link(link, eta)
                .expect("pilot inverse-link jet");
            let mu = jet.mu.clamp(1e-12, 1.0 - 1e-12);
            let d1 = jet.d1;
            let w = (d1 * d1 / (mu * (1.0 - mu))).max(1e-12);
            let z = eta + (y[i] - mu) / if d1.abs() > 1e-12 { d1 } else { 1e-12 };
            a00 += w;
            a01 += w * x[i];
            a11 += w * x[i] * x[i];
            b0 += w * z;
            b1 += w * z * x[i];
        }
        let det = a00 * a11 - a01 * a01;
        let nb0 = (a11 * b0 - a01 * b1) / det;
        let nb1 = (a00 * b1 - a01 * b0) / det;
        let delta = (nb0 - beta[0]).abs().max((nb1 - beta[1]).abs());
        beta[0] = nb0;
        beta[1] = nb1;
        if delta < 1e-12 {
            break;
        }
    }
    let eta = Array1::from_shape_fn(n, |i| beta[0] + beta[1] * x[i]);
    (beta, eta)
}

/// One fixed-λ frozen-basis Gauss-Newton solve, mirroring
/// `fit_binomial_mean_wiggle`'s freeze-refit loop with the wiggle log-λ held
/// FIXED (no outer REML search): freeze `B(η̂)`, residualize against the mean
/// columns in observation space, run the joint two-block inner solve through
/// the public fixed-log-λ entry, re-freeze at the refit η̂, until the frozen
/// index is a fixed point. Returns
/// `(penalized_objective, deviance, beta_eta, beta_w, eta_hat, cycles)`.
struct Zz2155Problem {
    y: Array1<f64>,
    x: Array1<f64>,
    link: InverseLink,
    knots: Array1<f64>,
    degree: usize,
    wiggle_template: ParameterBlockInput,
}

impl Zz2155Problem {
    fn solve_fixed_lambda_freeze_refit(
        &self,
        rho_w: &Array1<f64>,
        eta0: &Array1<f64>,
        beta_eta0: &Array1<f64>,
        beta_w0: Option<&Array1<f64>>,
    ) -> Result<(f64, f64, Array1<f64>, Array1<f64>, Array1<f64>, usize), String> {
    use std::sync::Arc;
    let (y, x, link, knots, degree, wiggle_template) = (
        &self.y,
        &self.x,
        &self.link,
        &self.knots,
        self.degree,
        &self.wiggle_template,
    );
    let n = y.len();
    let mut x_dense = Array2::<f64>::zeros((n, 2));
    for i in 0..n {
        x_dense[[i, 0]] = 1.0;
        x_dense[[i, 1]] = x[i];
    }
    let base_family = BinomialMeanWiggleFamily {
        y: y.clone(),
        weights: Array1::from_elem(n, 1.0),
        link_kind: link.clone(),
        wiggle_knots: knots.clone(),
        wiggle_degree: degree,
        policy: gam_runtime::resource::ResourcePolicy::default_library(),
        frozen_warp_design: None,
    };
    let mut frozen_eta = eta0.clone();
    let mut beta_eta = beta_eta0.clone();
    let mut beta_w: Option<Array1<f64>> = beta_w0.cloned();
    for cycle in 0..60 {
        let b_full = base_family
            .wiggle_design(frozen_eta.view())
            .map_err(|e| format!("wiggle design: {e}"))?;
        // Observation-space de-aliasing B⊥ = B - X (XᵀX)⁻¹ XᵀB (closed-form
        // 2×2 mean Gram — the pilot design is intercept + slope).
        let a00: f64 = x_dense.column(0).dot(&x_dense.column(0));
        let a01: f64 = x_dense.column(0).dot(&x_dense.column(1));
        let a11: f64 = x_dense.column(1).dot(&x_dense.column(1));
        let det = a00 * a11 - a01 * a01;
        let xtb = x_dense.t().dot(&b_full);
        let mut alias = Array2::<f64>::zeros((2, b_full.ncols()));
        for j in 0..b_full.ncols() {
            alias[[0, j]] = (a11 * xtb[[0, j]] - a01 * xtb[[1, j]]) / det;
            alias[[1, j]] = (a00 * xtb[[1, j]] - a01 * xtb[[0, j]]) / det;
        }
        let bda = &b_full - &x_dense.dot(&alias);

        let eta_input = ParameterBlockInput {
            design: DesignMatrix::Dense(gam_linalg::matrix::DenseDesignMatrix::from(
                x_dense.clone(),
            )),
            offset: Array1::zeros(n),
            penalties: vec![],
            nullspace_dims: vec![],
            initial_log_lambdas: Some(Array1::zeros(0)),
            initial_beta: Some(beta_eta.clone()),
        };
        let mut wiggle_input = wiggle_template.clone();
        wiggle_input.design =
            DesignMatrix::Dense(gam_linalg::matrix::DenseDesignMatrix::from(bda.clone()));
        wiggle_input.offset = Array1::zeros(n);
        wiggle_input.initial_log_lambdas = Some(rho_w.clone());
        wiggle_input.initial_beta = Some(match &beta_w {
            Some(b) => b.clone(),
            None => Array1::zeros(bda.ncols()),
        });
        let specs = vec![
            eta_input.intospec("eta").map_err(|e| e.to_string())?,
            wiggle_input.intospec("wiggle").map_err(|e| e.to_string())?,
        ];
        let mut fam = base_family.clone();
        fam.frozen_warp_design = Some(Arc::new(bda));
        let options = BlockwiseFitOptions::default();
        let fit = crate::custom_family::fit_custom_family_fixed_log_lambdas(
            &fam, &specs, &options, None,
        )
        .map_err(|e| format!("fixed-λ inner solve (cycle {cycle}): {e:?}"))?;
        let new_beta_eta = fit.block_states[BinomialMeanWiggleFamily::BLOCK_ETA]
            .beta
            .clone();
        let new_beta_w = fit.block_states[BinomialMeanWiggleFamily::BLOCK_WIGGLE]
            .beta
            .clone();
        let eta_hat = x_dense.dot(&new_beta_eta);
        let delta = eta_hat
            .iter()
            .zip(frozen_eta.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f64, f64::max);
        beta_eta = new_beta_eta;
        beta_w = Some(new_beta_w.clone());
        if delta <= 1.0e-9 {
            return Ok((
                fit.penalized_objective,
                fit.deviance,
                beta_eta,
                new_beta_w,
                eta_hat,
                cycle + 1,
            ));
        }
        frozen_eta = eta_hat;
    }
    Err("freeze-refit did not reach a fixed point in 60 cycles".to_string())
    }
}

/// Map cold-direct vs warp-penalty-continuation modes for a base link across a
/// small fixed-λ set. Pure measurement: every branch eprintln's its verdict;
/// the test itself only requires that at least one strategy converges per λ.
fn zz2155_mode_geography_for_link(label: &str, link: InverseLink) {
    let (y, x) = zz2155_fixture(600, 2155);
    let (pilot_beta, pilot_eta) = zz2155_pilot(&y, &x, &link);
    eprintln!(
        "[zz2155:{label}] pilot beta = ({:+.6}, {:+.6})",
        pilot_beta[0], pilot_beta[1]
    );
    let selected = select_wiggle_basis_from_seed(
        pilot_eta.view(),
        &WiggleBlockConfig {
            degree: 3,
            num_internal_knots: 8,
            penalty_order: 2,
            double_penalty: true,
        },
        &[1, 2, 3],
    )
    .expect("wiggle basis selection");
    let problem = Zz2155Problem {
        y: y.clone(),
        x: x.clone(),
        link: link.clone(),
        knots: selected.knots.clone(),
        degree: selected.degree,
        wiggle_template: selected.block.clone(),
    };
    let k_w = selected.block.penalties.len();
    eprintln!(
        "[zz2155:{label}] wiggle: {} coefficients, {k_w} penalties, nullspace {:?}",
        match &selected.block.design {
            DesignMatrix::Dense(d) => d.to_dense().ncols(),
            DesignMatrix::Sparse(_) => 0,
        },
        selected.block.nullspace_dims,
    );

    for rho_center in [0.0_f64, 2.0] {
        let rho = Array1::from_elem(k_w, rho_center);
        // (a) production-shaped cold seed: pilot β, β_w = 0, straight to λ.
        let cold = problem.solve_fixed_lambda_freeze_refit(&rho, &pilot_eta, &pilot_beta, None);
        // (b) warp-penalty continuation: descend τ ∈ {8, 5, 3, 1.5, 0.5, 0}
        // added to every wiggle log-λ, warm-chaining (β, η̂) down the ladder.
        let mut carried: Option<(Array1<f64>, Array1<f64>, Array1<f64>)> = None;
        let mut cont: Result<(f64, f64, Array1<f64>, Array1<f64>, Array1<f64>, usize), String> =
            Err("continuation never ran".to_string());
        for tau in [8.0_f64, 5.0, 3.0, 1.5, 0.5, 0.0] {
            let rho_tau = rho.mapv(|v| v + tau);
            let (eta_seed, beta_eta_seed, beta_w_seed) = match &carried {
                Some((be, bw, eh)) => (eh.clone(), be.clone(), Some(bw.clone())),
                None => (pilot_eta.clone(), pilot_beta.clone(), None),
            };
            cont = problem.solve_fixed_lambda_freeze_refit(
                &rho_tau,
                &eta_seed,
                &beta_eta_seed,
                beta_w_seed.as_ref(),
            );
            match &cont {
                Ok((pen, dev, be, bw, eh, cycles)) => {
                    eprintln!(
                        "[zz2155:{label}] rho={rho_center:+.1} tau={tau:+.1}: pen_obj={pen:.10e} dev={dev:.6e} |beta_w|_1={:.6e} cycles={cycles}",
                        bw.iter().map(|v| v.abs()).sum::<f64>()
                    );
                    carried = Some((be.clone(), bw.clone(), eh.clone()));
                }
                Err(e) => {
                    eprintln!("[zz2155:{label}] rho={rho_center:+.1} tau={tau:+.1}: FAILED {e}");
                    carried = None;
                }
            }
        }
        match (&cold, &cont) {
            (Ok((pc, dc, _, bwc, _, cyc_c)), Ok((pk, dk, _, bwk, _, cyc_k))) => {
                let dbw = bwc
                    .iter()
                    .zip(bwk.iter())
                    .map(|(a, b)| (a - b).abs())
                    .fold(0.0_f64, f64::max);
                eprintln!(
                    "[zz2155:{label}] VERDICT rho={rho_center:+.1}: cold pen={pc:.10e} (dev {dc:.6e}, {cyc_c} cyc) vs continuation pen={pk:.10e} (dev {dk:.6e}, {cyc_k} cyc); pen gap (cold-cont)={:+.6e}; max|Δβ_w|={dbw:.3e}",
                    pc - pk
                );
            }
            (Ok((pc, ..)), Err(e)) => {
                eprintln!(
                    "[zz2155:{label}] VERDICT rho={rho_center:+.1}: cold pen={pc:.10e}; continuation FAILED: {e}"
                );
            }
            (Err(e), Ok((pk, ..))) => {
                eprintln!(
                    "[zz2155:{label}] VERDICT rho={rho_center:+.1}: cold FAILED ({e}); continuation pen={pk:.10e}"
                );
            }
            (Err(ec), Err(ek)) => {
                eprintln!(
                    "[zz2155:{label}] VERDICT rho={rho_center:+.1}: BOTH FAILED cold={ec} cont={ek}"
                );
            }
        }
        assert!(
            cold.is_ok() || cont.is_ok(),
            "[zz2155:{label}] rho={rho_center:+.1}: no strategy converged"
        );
    }
}

#[test]
fn zz_measure_2155_loglog_mode_geography() {
    zz2155_mode_geography_for_link("loglog", InverseLink::Standard(StandardLink::LogLog));
}

#[test]
fn zz_measure_2155_cauchit_mode_geography() {
    zz2155_mode_geography_for_link("cauchit", InverseLink::Standard(StandardLink::Cauchit));
}

#[test]
fn zz_measure_2155_cloglog_control_mode_geography() {
    zz2155_mode_geography_for_link("cloglog", InverseLink::Standard(StandardLink::CLogLog));
}

