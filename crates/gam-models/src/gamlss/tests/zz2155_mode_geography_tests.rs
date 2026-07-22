// #2155 mode-geography measurement block, split out of gamlss/tests.rs to keep
// that file under the #780 10k-line gate. Child module of `tests`, so the
// zz2155 fixture/problem helpers stay private to the test tree.
use super::*;

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

