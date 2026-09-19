//! #2817: a cost stall above the certificate's band carries no convergence claim.
//!
//! The pin is a curved valley `u = ρ₀ − ρ₁²`,
//! `f = V₀ + ½·wall·u² + ½·slope·(ρ₁ − 3)²`, searched on the gradient-only BFGS
//! route. Scan v2 (job 1128119 at 7eb9b6a531) measured the old guard claiming at
//! both starts: `origin=BfgsCostStallExit, claimed_converged=true` at
//! |Pg| 2.289e-1 against a certificate bound of 1.073e-1 from (0.5, 0), and at
//! |Pg| 2.019e-1 against 6.532e-2 from (4, −2). The claims came through the
//! guard's score-relative term `min(1e-3·(1 + |V|), 1) = 1.0`, which the
//! certificate never applied. A stall now claims only inside the band the
//! certificate applies at its incumbent, so neither refusal may carry a
//! cost-stall claim.

use super::*;
use ndarray::array;

const WALL: f64 = 1.0e4;
const SLOPE: f64 = 1.0e-2;
const OFFSET: f64 = 1.0e3;
const TARGET: f64 = 3.0;

fn valley_eval(rho: &Array1<f64>) -> OuterEval {
    let r1 = rho[1];
    let u = rho[0] - r1 * r1;
    let cross = -2.0 * WALL * r1;
    OuterEval {
        cost: OFFSET + 0.5 * WALL * u * u + 0.5 * SLOPE * (r1 - TARGET) * (r1 - TARGET),
        gradient: array![WALL * u, cross * u + SLOPE * (r1 - TARGET)],
        hessian: HessianValue::Dense(array![
            [WALL, cross],
            [cross, WALL * (4.0 * r1 * r1 - 2.0 * u) + SLOPE],
        ]),
        inner_beta_hint: None,
    }
}

fn run_valley(start: &Array1<f64>) -> Result<OuterResult, EstimationError> {
    let problem = OuterProblem::new(2)
        .with_gradient(Derivative::Analytic)
        .with_hessian(DeclaredHessianForm::Dense)
        .with_prefer_gradient_only(true)
        .with_bounds(Array1::from_elem(2, -20.0), Array1::from_elem(2, 20.0))
        .with_initial_rho(start.clone())
        .with_screen_initial_rho(false)
        .with_seed_config(gam_problem::SeedConfig {
            max_seeds: 1,
            seed_budget: 1,
            ..Default::default()
        });
    let mut obj = problem.build_objective(
        (),
        |_: &mut (), rho: &Array1<f64>| Ok(valley_eval(rho).cost),
        |_: &mut (), rho: &Array1<f64>| Ok(valley_eval(rho)),
        None::<fn(&mut ())>,
        None::<fn(&mut (), &Array1<f64>) -> Result<EfsEval, EstimationError>>,
    );
    problem.run(&mut obj, "curved valley claim pin #2817")
}

#[test]
fn a_curved_valley_stall_above_the_band_is_not_claimed_2817() {
    for start in [array![0.5, 0.0], array![4.0, -2.0]] {
        match run_valley(&start) {
            Ok(result) => {
                eprintln!(
                    "[#2817 valley claim pin] start={:?} OK rho={:?} |g|={:?} origin={:?}",
                    start.to_vec(),
                    result.rho.to_vec(),
                    result.final_grad_norm,
                    result.origin,
                );
                assert!(
                    result.final_grad_norm.is_some_and(f64::is_finite),
                    "a certified valley optimum must report its final gradient norm (start {:?})",
                    start.to_vec()
                );
            }
            Err(error) => {
                let text = error.to_string();
                eprintln!("[#2817 valley claim pin] start={:?} ERR {text}", start.to_vec());
                assert!(
                    !(text.contains("origin=BfgsCostStallExit")
                        && text.contains("claimed_converged=true")),
                    "a cost stall the certificate refused must not carry a convergence claim \
                     (start {:?}): {text}",
                    start.to_vec()
                );
            }
        }
    }
}
