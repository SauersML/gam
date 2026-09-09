// Child module of `run_plan::run_plan_tests` (see the `#[path]` declaration
// there): #2334 terminal-coefficient-mode reset for cap-less mode owners, and
// the #2357/#2155 interior and railed strict-saddle escape. Scope comes from
// the parent via `use super::*`; the split is purely physical.

use super::*;
use ndarray::array;

// ─── #2334 terminal-coefficient-mode reset for cap-less mode owners ──
//
// The terminal certification sequence installs the selected point twice at
// `result.rho`: once via `finalize_outer_result` (where a mode-owning
// objective installs its owned coefficient mode) and once via the analytic
// re-eval in `certify_outer_optimality` (which sets `result.final_value`). On
// a nonconvex / bimodal inner solve those two evaluations can land in
// DIFFERENT coefficient basins unless each re-installs from the same clean
// baseline through `reset()`. That reset was gated solely on
// `config.outer_inner_cap.is_some()`, which the custom-family fit never sets
// (it holds its inner cap in a different field), so the reset never fired and
// the downstream bitwise bind `terminal_mode.objective == final_value` could
// spuriously fail. `owns_terminal_coefficient_mode()` now forces that reset
// independently of the cap.

/// A deterministic stand-in for a warm-start-sensitive bimodal inner solve.
///
/// Each derivative-bearing terminal evaluation (`eval` / `eval_with_order` /
/// `finalize`) reads a warm-parity flag: in the "bumped" parity the inner
/// solve lands in a basin whose objective is offset by `BASIN_BUMP`; every
/// such evaluation flips the parity, modeling an inner solve whose consecutive
/// warm-started solves alternate basins. `reset()` re-baselines the parity to
/// the un-bumped basin. The outer gradient is basin-independent (`= ρ`), so
/// the fit still certifies stationarity at ρ = 0 regardless of basin — exactly
/// the real defect's shape, where the two inner optima share the outer
/// gradient but differ in objective value.
///
/// Consequence: with NO terminal reset, `finalize` and the certifying re-eval
/// are one flip apart, so their objective values differ by `BASIN_BUMP`
/// (whole-basin disagreement, not roundoff). With the terminal reset, both
/// start from the un-bumped baseline and agree bitwise.
struct BimodalTerminalObjective {
    owns_terminal: bool,
    warm_bumped: Arc<Mutex<bool>>,
    finalize_installed: Arc<Mutex<Option<f64>>>,
}

const BASIN_BUMP: f64 = 2.6; // ~ the observed terminal(9.1931e2) − certified(9.1671e2) gap

impl BimodalTerminalObjective {
    /// The objective in the CURRENTLY installed basin, without changing it.
    ///
    /// Reading the installed state is not a solve, so it must not flip the
    /// parity — and, critically, every lane reading the same instant must see
    /// the SAME basin. A mock whose value lane and derivative lane disagree at
    /// one ρ is not a bimodal inner solve; it is a value/gradient desync, and
    /// the terminal certificate's same-ρ value-agreement audit refuses it on
    /// sight (measured: `value-only=0.0, analytic-sample=2.6,
    /// disagreement=2.600e0, roundoff bound=3.874e-8`). That refusal is the
    /// audit working correctly; it was the fixture that was modelling the wrong
    /// thing.
    fn basin_value(&self, rho: &Array1<f64>) -> f64 {
        let bump = if *self
            .warm_bumped
            .lock()
            .expect("the warm_bumped mutex is not poisoned")
        {
            BASIN_BUMP
        } else {
            0.0
        };
        0.5 * rho.dot(rho) + bump
    }

    /// A warm-started SOLVE: reads the current basin and lands the next one in
    /// the other basin. This is what makes two consecutive installations
    /// disagree unless a `reset()` re-baselines between them.
    fn basin_solve(&self, rho: &Array1<f64>) -> OuterEval {
        let cost = self.basin_value(rho);
        let mut warm = self
            .warm_bumped
            .lock()
            .expect("the warm_bumped mutex is not poisoned");
        *warm = !*warm; // consecutive warm-started solves alternate basins
        OuterEval {
            cost,
            gradient: rho.clone(),
            hessian: HessianValue::Unavailable,
            inner_beta_hint: None,
        }
    }
}

impl OuterObjective for BimodalTerminalObjective {
    fn capability(&self) -> OuterCapability {
        OuterCapability {
            gradient: Derivative::Analytic,
            hessian: DeclaredHessianForm::Unavailable,
            n_params: 1,
            psi_dim: 0,
            fixed_point_available: false,
            barrier_config: None,
            prefer_gradient_only: false,
            disable_fixed_point: true,
        }
    }
    fn eval_cost(&mut self, rho: &Array1<f64>) -> Result<f64, EstimationError> {
        // Reads the installed basin; does not solve, so it does not flip.
        Ok(self.basin_value(rho))
    }
    fn eval(&mut self, rho: &Array1<f64>) -> Result<OuterEval, EstimationError> {
        Ok(self.basin_solve(rho))
    }
    fn eval_with_order(
        &mut self,
        rho: &Array1<f64>,
        order: OuterEvalOrder,
    ) -> Result<OuterEval, EstimationError> {
        match order {
            // The value-only lane is the scalar authority sampling the state
            // that is already installed — the same read `eval_cost` performs.
            // It must agree with the derivative-bearing sample taken at the
            // same ρ, or the fixture is a desync rather than a bimodal solve.
            OuterEvalOrder::Value => Ok(OuterEval {
                cost: self.basin_value(rho),
                gradient: rho.clone(),
                hessian: HessianValue::Unavailable,
                inner_beta_hint: None,
            }),
            // Derivative-bearing orders are warm-started solves and DO advance
            // the basin, which is what leaves `finalize` and the certifying
            // re-eval one flip apart when no reset intervenes.
            OuterEvalOrder::ValueAndGradient | OuterEvalOrder::ValueGradientHessian => {
                Ok(self.basin_solve(rho))
            }
        }
    }
    fn finalize_outer_result(
        &mut self,
        rho: &Array1<f64>,
        plan: &OuterPlan,
    ) -> Result<(), EstimationError> {
        // This fixture's whole premise is the warm-started, DERIVATIVE-bearing
        // inner solve of `basin_solve` (see `eval_with_order`). A solver that
        // never requests derivatives would not exercise the parity flip these
        // tests measure, so refuse it rather than record an install whose
        // meaning the assertions do not cover.
        if matches!(plan.solver, Solver::Efs | Solver::HybridEfs) {
            return Err(EstimationError::RemlOptimizationFailed(format!(
                "bimodal terminal fixture reached finalize under solver {:?}, which never \
                 requests a derivative-bearing evaluation",
                plan.solver
            )));
        }
        // Install the owned coefficient mode: record the objective value the
        // mode carries, exactly as the custom-family evaluator does.
        let installed = self.basin_solve(rho).cost;
        *self
            .finalize_installed
            .lock()
            .expect("the finalize_installed mutex is not poisoned") = Some(installed);
        Ok(())
    }
    fn owns_terminal_coefficient_mode(&self) -> bool {
        self.owns_terminal
    }
    fn reset(&mut self) {
        *self
            .warm_bumped
            .lock()
            .expect("the warm_bumped mutex is not poisoned") = false;
    }
    fn seed_inner_state(&mut self, beta: &Array1<f64>) -> Result<SeedOutcome, EstimationError> {
        // The bimodal basin is carried by `warm_bumped`, not an inner-β slot —
        // but the offered seed must still be finite, or the driver handed this
        // fixture a state no warm start could legitimately resume from.
        if beta.iter().any(|value| !value.is_finite()) {
            return Err(EstimationError::RemlOptimizationFailed(format!(
                "bimodal terminal fixture was offered a non-finite inner seed of length {}",
                beta.len()
            )));
        }
        Ok(SeedOutcome::NoSlot)
    }
}

fn run_bimodal_terminal(owns_terminal: bool) -> (f64, f64) {
    let warm_bumped = Arc::new(Mutex::new(false));
    let finalize_installed = Arc::new(Mutex::new(None));
    let mut obj = BimodalTerminalObjective {
        owns_terminal,
        warm_bumped: Arc::clone(&warm_bumped),
        finalize_installed: Arc::clone(&finalize_installed),
    };
    // Seed AT the stationary point (∇ = ρ = 0) so the outer search certifies
    // at iteration 0 and control passes straight into the terminal sequence.
    let problem = OuterProblem::new(1)
        .with_gradient(Derivative::Analytic)
        .with_hessian(DeclaredHessianForm::Unavailable)
        .with_initial_rho(array![0.0])
        .with_screen_initial_rho(false)
        .with_seed_config(gam_problem::SeedConfig {
            max_seeds: 1,
            seed_budget: 1,
            ..Default::default()
        });
    let result = problem
        .run(&mut obj, "bimodal-terminal")
        .expect("stationary seed must certify");
    assert!(result.converged(), "stationary seed must converge");
    let installed = finalize_installed
        .lock()
        .expect("the finalize_installed mutex is not poisoned")
        .expect("finalize must have installed a terminal mode");
    (installed, result.final_value)
}

#[test]
fn terminal_reset_binds_bimodal_mode_owner_bitwise() {
    // WITH the ownership flag: the terminal reset fires before BOTH the
    // finalize install and the certifying re-eval, so the mode's objective and
    // the certified value come from the same un-bumped baseline — bitwise equal.
    let (installed, final_value) = run_bimodal_terminal(true);
    assert_eq!(
        installed.to_bits(),
        final_value.to_bits(),
        "terminal-coefficient-mode owner: finalize-installed objective ({installed:.17e}) \
         must bitwise-match the certified final_value ({final_value:.17e})",
    );
}

#[test]
fn without_ownership_flag_bimodal_terminal_bind_fails() {
    // WITHOUT the flag (and no outer_inner_cap wired, exactly the custom-family
    // situation): no terminal reset, so finalize and the certifying re-eval are
    // one warm-parity flip apart and settle in different basins — a whole-basin
    // bitwise mismatch, i.e. the spurious bind failure this fix removes.
    let (installed, final_value) = run_bimodal_terminal(false);
    assert_ne!(
        installed.to_bits(),
        final_value.to_bits(),
        "control: without the ownership flag the bimodal finalize/certify pair must disagree",
    );
    assert!(
        (installed - final_value).abs() >= BASIN_BUMP - 1e-9,
        "the disagreement must be a whole-basin gap (~{BASIN_BUMP}), not roundoff: \
         installed={installed:.17e} final_value={final_value:.17e}",
    );
}

// ─── #2357 interior strict-saddle escape ──────────────────────────

// A 2-D outer objective with a genuine interior saddle at ρ=(0,0) and a pair of
// PSD minima at ρ=(0,±1):
//
//   f(ρ) = ½·ρ₀²  +  ¼·ρ₁⁴ − ½·ρ₁²
//   ∇f    = (ρ₀,  ρ₁³ − ρ₁)
//   H     = [[1, 0], [0, 3ρ₁² − 1]]
//
// At ρ=(0,0): ∇=0 (first-order stationary) but H=diag(1,−1) is INDEFINITE — a
// strict saddle. The two minima ρ₁=±1 carry H=diag(1,2) (PSD) and f=−¼. A
// gradient-only convergence gate that arrives at the saddle stops there; only
// the certified negative-curvature escape reaches a minimum. This is the
// synthetic distillation of the periodic-te() cold-start refusal in #2357.
fn saddle_cost(rho: &Array1<f64>) -> f64 {
    let r0 = rho[0];
    let r1 = rho[1];
    0.5 * r0 * r0 + 0.25 * r1 * r1 * r1 * r1 - 0.5 * r1 * r1
}

fn saddle_eval(rho: &Array1<f64>) -> OuterEval {
    let r0 = rho[0];
    let r1 = rho[1];
    OuterEval {
        cost: saddle_cost(rho),
        gradient: array![r0, r1 * r1 * r1 - r1],
        hessian: HessianValue::Dense(array![[1.0, 0.0], [0.0, 3.0 * r1 * r1 - 1.0]]),
        inner_beta_hint: None,
    }
}

fn saddle_problem() -> OuterProblem {
    OuterProblem::new(2)
        .with_gradient(Derivative::Analytic)
        .with_hessian(DeclaredHessianForm::Dense)
        .with_initial_rho(array![0.0, 0.0])
        .with_screen_initial_rho(false)
        .with_seed_config(gam_problem::SeedConfig {
            max_seeds: 1,
            seed_budget: 1,
            ..Default::default()
        })
}

#[test]
fn certify_mints_saddle_escape_reseed_at_interior_saddle() {
    // Directly audit the saddle point: the certificate must refuse it for
    // indefinite interior curvature AND publish a negative-curvature escape
    // reseed strictly BELOW the saddle objective, moving along the indefinite
    // axis rather than the positive-curvature one (#2357).
    let problem = saddle_problem();
    let mut obj = problem.build_objective(
        (),
        |_: &mut (), rho: &Array1<f64>| Ok(saddle_cost(rho)),
        |_: &mut (), rho: &Array1<f64>| Ok(saddle_eval(rho)),
        None::<fn(&mut ())>,
        None::<fn(&mut (), &Array1<f64>) -> Result<EfsEval, EstimationError>>,
    );
    let rejection = audit_stationary_point(&mut obj, array![0.0, 0.0], "saddle-escape #2357")
        .expect_err("an interior strict saddle must be refused, not certified");
    let result = &rejection.result;
    let cert = result
        .criterion_certificate
        .as_ref()
        .expect("refused certificate must be recorded");
    assert!(
        cert.is_stationary(),
        "the gradient must clear the stationarity band at the saddle"
    );
    assert_eq!(
        cert.hessian_psd(),
        Some(false),
        "the saddle Hessian must read indefinite"
    );
    assert!(
        cert.lambdas_railed.is_empty(),
        "the saddle is interior, not railed"
    );
    let reseed = result
        .saddle_escape_reseed
        .as_ref()
        .expect("a refused interior saddle must mint a negative-curvature escape reseed");
    let saddle_f = saddle_cost(&array![0.0, 0.0]);
    assert!(
        saddle_cost(reseed) < saddle_f - 1e-9,
        "escape reseed {reseed:?} (f={}) must strictly descend below the saddle f={saddle_f}",
        saddle_cost(reseed),
    );
    assert!(
        reseed[1].abs() > 1e-6,
        "escape must step along the indefinite ρ₁ axis, not the PSD ρ₀ axis: {reseed:?}"
    );
}

// #2155 — the SAME double-well saddle in (ρ₀, ρ₁), plus a third coordinate ρ₂
// pulled by a constant negative gradient onto the +rho_bound box rail. At the
// railed point the FULL Hessian is diag(1, −1, 0), but the load-bearing verdict
// is on the REDUCED (off-railed) sub-block diag(1, −1) over {ρ₀, ρ₁}, which is
// indefinite. This is the flexible-link binomial-wiggle failure genus (#2155):
// a genuine interior saddle in the free directions while a smoothing coordinate
// rails, where the #2357 escape used to be waived by the mere presence of a
// rail (`lambdas_railed.is_empty()`), so the fit was refused despite a real,
// box-feasible descent along ρ₁ that holds ρ₂ fixed on its rail.
fn railed_saddle_cost(rho: &Array1<f64>) -> f64 {
    let r0 = rho[0];
    let r1 = rho[1];
    let r2 = rho[2];
    0.5 * r0 * r0 + 0.25 * r1 * r1 * r1 * r1 - 0.5 * r1 * r1 - 0.3 * r2
}

fn railed_saddle_eval(rho: &Array1<f64>) -> OuterEval {
    let r0 = rho[0];
    let r1 = rho[1];
    OuterEval {
        cost: railed_saddle_cost(rho),
        // ∂/∂ρ₂ = −0.3: a constant outward pull that drives ρ₂ to the +rho_bound
        // rail and keeps it there. ρ₂ carries zero curvature, so the full Hessian
        // is only semidefinite; the indefinite direction lives entirely in the
        // un-railed {ρ₀, ρ₁} block.
        gradient: array![r0, r1 * r1 * r1 - r1, -0.3],
        hessian: HessianValue::Dense(array![
            [1.0, 0.0, 0.0],
            [0.0, 3.0 * r1 * r1 - 1.0, 0.0],
            [0.0, 0.0, 0.0],
        ]),
        inner_beta_hint: None,
    }
}

fn railed_saddle_problem() -> OuterProblem {
    // Default box is ±rho_bound = ±30, so ρ₂ = 30 sits exactly on the upper rail.
    OuterProblem::new(3)
        .with_gradient(Derivative::Analytic)
        .with_hessian(DeclaredHessianForm::Dense)
        .with_initial_rho(array![0.0, 0.0, 30.0])
        .with_screen_initial_rho(false)
        .with_seed_config(gam_problem::SeedConfig {
            max_seeds: 1,
            seed_budget: 1,
            ..Default::default()
        })
}

#[test]
fn certify_mints_saddle_escape_reseed_at_railed_saddle_2155() {
    // A saddle whose indefinite curvature is confined to the free (un-railed)
    // directions must STILL be refused and STILL publish a negative-curvature
    // escape reseed — one that steps along the interior indefinite axis (ρ₁)
    // while holding the railed coordinate (ρ₂) fixed on its bound (#2155).
    let problem = railed_saddle_problem();
    let mut obj = problem.build_objective(
        (),
        |_: &mut (), rho: &Array1<f64>| Ok(railed_saddle_cost(rho)),
        |_: &mut (), rho: &Array1<f64>| Ok(railed_saddle_eval(rho)),
        None::<fn(&mut ())>,
        None::<fn(&mut (), &Array1<f64>) -> Result<EfsEval, EstimationError>>,
    );
    let rejection =
        audit_stationary_point(&mut obj, array![0.0, 0.0, 30.0], "railed-saddle-escape #2155")
            .expect_err("a railed strict saddle in the free directions must be refused");
    let result = &rejection.result;
    let cert = result
        .criterion_certificate
        .as_ref()
        .expect("refused certificate must be recorded");
    assert!(
        cert.is_stationary(),
        "the interior gradient must clear the stationarity band at the railed saddle"
    );
    assert_eq!(
        cert.hessian_psd(),
        Some(false),
        "the REDUCED (off-railed) Hessian must read indefinite"
    );
    assert!(
        cert.lambdas_railed.contains(&2),
        "ρ₂ must be recorded as railed: {:?}",
        cert.lambdas_railed
    );
    let reseed = result.saddle_escape_reseed.as_ref().expect(
        "a refused railed saddle with an indefinite reduced Hessian must mint a reseed (#2155)",
    );
    let saddle_f = railed_saddle_cost(&array![0.0, 0.0, 30.0]);
    assert!(
        railed_saddle_cost(reseed) < saddle_f - 1e-9,
        "escape reseed {reseed:?} (f={}) must strictly descend below the saddle f={saddle_f}",
        railed_saddle_cost(reseed),
    );
    assert!(
        reseed[1].abs() > 1e-6,
        "escape must step along the free indefinite ρ₁ axis: {reseed:?}"
    );
    assert!(
        (reseed[2] - 30.0).abs() < 1e-12,
        "escape must hold the railed ρ₂ fixed on its bound: {reseed:?}"
    );
}

#[test]
fn outer_search_escapes_railed_saddle_and_certifies_minimum_2155() {
    // Full pipeline: seeded at the railed saddle, the outer search must escape
    // via the one-shot reseed and certify a genuine reduced-PSD minimum at
    // ρ₁=±1 with ρ₂ still on its rail — not refuse at the saddle (#2155).
    let problem = railed_saddle_problem();
    let mut obj = problem.build_objective(
        (),
        |_: &mut (), rho: &Array1<f64>| Ok(railed_saddle_cost(rho)),
        |_: &mut (), rho: &Array1<f64>| Ok(railed_saddle_eval(rho)),
        None::<fn(&mut ())>,
        None::<fn(&mut (), &Array1<f64>) -> Result<EfsEval, EstimationError>>,
    );
    let result = problem
        .run(&mut obj, "railed-saddle-escape pipeline #2155")
        .expect("the outer search must escape the railed saddle and certify a minimum");
    assert!(
        result.converged(),
        "must converge at a reduced-PSD minimum, not refuse at the railed saddle: rho={:?}",
        result.rho
    );
    assert!(
        (result.rho[1].abs() - 1.0).abs() < 1e-4,
        "must land on the free-direction minimum ρ₁=±1, got ρ={:?}",
        result.rho,
    );
    assert!(
        (result.rho[2] - 30.0).abs() < 1e-4,
        "ρ₂ must remain on its rail at the minimum: {:?}",
        result.rho,
    );
    assert!(
        result
            .criterion_certificate
            .as_ref()
            .is_some_and(|c| c.certifies() && c.hessian_psd() == Some(true)),
        "the escaped minimum must carry a reduced-PSD certificate",
    );
}

#[test]
fn outer_search_escapes_interior_saddle_and_certifies_minimum() {
    // Full pipeline: seeded AT the saddle, the outer search must escape via the
    // one-shot negative-curvature reseed and certify a genuine PSD minimum at
    // ρ₁=±1 — not refuse the whole fit at the saddle (#2357).
    let problem = saddle_problem();
    let mut obj = problem.build_objective(
        (),
        |_: &mut (), rho: &Array1<f64>| Ok(saddle_cost(rho)),
        |_: &mut (), rho: &Array1<f64>| Ok(saddle_eval(rho)),
        None::<fn(&mut ())>,
        None::<fn(&mut (), &Array1<f64>) -> Result<EfsEval, EstimationError>>,
    );
    let result = problem
        .run(&mut obj, "saddle-escape pipeline #2357")
        .expect("the outer search must escape the saddle and certify a minimum");
    assert!(
        result.converged(),
        "must converge at a minimum, not refuse at the saddle"
    );
    assert!(
        (result.rho[1].abs() - 1.0).abs() < 1e-4,
        "must land on a PSD minimum ρ₁=±1, got ρ={:?}",
        result.rho,
    );
    assert!(
        result.rho[0].abs() < 1e-4,
        "ρ₀ must be 0 at the minimum: {:?}",
        result.rho,
    );
    assert!(
        result
            .criterion_certificate
            .as_ref()
            .is_some_and(|c| c.certifies() && c.hessian_psd() == Some(true)),
        "the escaped minimum must carry a PSD certificate",
    );
}

// ─── #2612 the criterion adjudicates the matrix ───────────────────

// A LYING Hessian: a strictly convex bowl `f(ρ) = ½‖ρ‖²` whose declared
// curvature is `diag(1, −1)`.
//
// This is #2665's measured shape reduced to two coordinates — there the
// analytic `λ_min = −1721.5` had an actual objective curvature of `+121.6`
// along its OWN eigenvector, established by `Δ ∝ α²` over three decades. No
// resolution bound can catch that: the matrix is not imprecise there, it is
// wrong, and it is wrong by 14× in magnitude and by sign.
//
// The point ρ = 0 is the global minimum and the gradient vanishes there, so
// every first-order fact certifies. Only the declared curvature refuses, and
// the criterion contradicts it: `f(±αe₁) = ½α² > 0` for every α.
fn lying_hessian_cost(rho: &Array1<f64>) -> f64 {
    0.5 * rho.dot(rho)
}

fn lying_hessian_eval(rho: &Array1<f64>) -> OuterEval {
    OuterEval {
        cost: lying_hessian_cost(rho),
        gradient: rho.clone(),
        // The true Hessian is `I`. This one is sign-flipped in ρ₁.
        hessian: HessianValue::Dense(array![[1.0, 0.0], [0.0, -1.0]]),
        inner_beta_hint: None,
    }
}

#[test]
fn criterion_contradicts_a_lying_hessian_and_the_point_certifies_2612() {
    let problem = OuterProblem::new(2)
        .with_gradient(Derivative::Analytic)
        .with_hessian(DeclaredHessianForm::Dense)
        .with_initial_rho(array![0.0, 0.0])
        .with_screen_initial_rho(false)
        .with_seed_config(gam_problem::SeedConfig {
            max_seeds: 1,
            seed_budget: 1,
            ..Default::default()
        });
    let mut obj = problem.build_objective(
        (),
        |_: &mut (), rho: &Array1<f64>| Ok(lying_hessian_cost(rho)),
        |_: &mut (), rho: &Array1<f64>| Ok(lying_hessian_eval(rho)),
        None::<fn(&mut ())>,
        None::<fn(&mut (), &Array1<f64>) -> Result<EfsEval, EstimationError>>,
    );
    let result = audit_stationary_point(&mut obj, array![0.0, 0.0], "lying-hessian #2612")
        .expect("a global minimum must not be refused on curvature the criterion contradicts");
    let cert = result
        .criterion_certificate
        .as_ref()
        .expect("a certified point carries its certificate");
    assert!(
        cert.certifies(),
        "the certificate must accept: {}",
        cert.summary()
    );
    assert_eq!(
        cert.curvature,
        CurvatureEvidence::CriterionContradicted,
        "the withdrawn verdict must be RECORDED as contradicted, not quietly relabelled \
         `Measured {{ psd: true }}` — nothing here established that the point is a minimum, \
         only that this matrix's negative direction has no operational content: {}",
        cert.summary()
    );
    assert_eq!(
        cert.hessian_psd(),
        None,
        "the published `hessian_psd` contract is null|true|false and a contradicted verdict is \
         not a PSD claim"
    );
    assert!(
        result.saddle_escape_reseed.is_none(),
        "there is nothing to escape from: {:?}",
        result.saddle_escape_reseed
    );
}

// The NEGATIVE CONTROL for the test above, and the reason it cannot pass by
// simply never refusing: the same audit on a REAL saddle must still refuse and
// still mint the escape. `certify_mints_saddle_escape_reseed_at_interior_saddle`
// above is that control, unchanged.

// A saddle whose descent is real but lives BELOW the old fixed ladder's last
// rung.
//
//   f(ρ) = ½ρ₀² + 2ρ₁⁴ − 0.005ρ₁²        H = diag(1, 24ρ₁² − 0.01)
//
// At ρ = 0 the curvature is `−0.01` and the objective descends along ρ₁ only
// while `2ρ₁⁴ < 0.005ρ₁²`, i.e. `|ρ₁| < 0.05`. The old ladder's rungs were
// `[1, 0.5, 0.25, 0.125, 0.0625]` — every one of them OUTSIDE that well, so it
// found no descending trial and the fit was refused at a point with a genuine,
// exploitable descent one halving further down.
//
// The derived ladder runs to `α_min = sqrt(2·objective_resolution/|λ_min|)`,
// which at the default `1e-7` resolution and `|λ_min| = 0.01` is `4.5e-3`, so
// it reaches `0.03125` — inside the well — and mints the escape.
fn narrow_well_cost(rho: &Array1<f64>) -> f64 {
    let r0 = rho[0];
    let r1 = rho[1];
    0.5 * r0 * r0 + 2.0 * r1 * r1 * r1 * r1 - 0.005 * r1 * r1
}

fn narrow_well_eval(rho: &Array1<f64>) -> OuterEval {
    let r0 = rho[0];
    let r1 = rho[1];
    OuterEval {
        cost: narrow_well_cost(rho),
        gradient: array![r0, 8.0 * r1 * r1 * r1 - 0.01 * r1],
        hessian: HessianValue::Dense(array![[1.0, 0.0], [0.0, 24.0 * r1 * r1 - 0.01]]),
        inner_beta_hint: None,
    }
}

#[test]
fn escape_reaches_a_descent_below_the_old_fixed_ladder_2612() {
    // Pin the fixture's own arithmetic first, so a failure below is about the
    // ladder and not about the well having moved.
    let outside = narrow_well_cost(&array![0.0, 0.0625]);
    let inside = narrow_well_cost(&array![0.0, 0.03125]);
    assert!(
        outside > 0.0,
        "the old ladder's last rung must sit OUTSIDE the well (f={outside:.6e})"
    );
    assert!(
        inside < 0.0,
        "the next halving must sit INSIDE it (f={inside:.6e})"
    );

    let problem = OuterProblem::new(2)
        .with_gradient(Derivative::Analytic)
        .with_hessian(DeclaredHessianForm::Dense)
        .with_initial_rho(array![0.0, 0.0])
        .with_screen_initial_rho(false)
        .with_seed_config(gam_problem::SeedConfig {
            max_seeds: 1,
            seed_budget: 1,
            ..Default::default()
        });
    let mut obj = problem.build_objective(
        (),
        |_: &mut (), rho: &Array1<f64>| Ok(narrow_well_cost(rho)),
        |_: &mut (), rho: &Array1<f64>| Ok(narrow_well_eval(rho)),
        None::<fn(&mut ())>,
        None::<fn(&mut (), &Array1<f64>) -> Result<EfsEval, EstimationError>>,
    );
    let rejection = audit_stationary_point(&mut obj, array![0.0, 0.0], "narrow-well #2612")
        .expect_err("a saddle with a real descent must be refused, not certified");
    let result = &rejection.result;
    let cert = result
        .criterion_certificate
        .as_ref()
        .expect("refused certificate must be recorded");
    assert_eq!(
        cert.hessian_psd(),
        Some(false),
        "the curvature here is genuinely indefinite AND exploitable, so the verdict must stand \
         rather than be withdrawn: {}",
        cert.summary()
    );
    let reseed = result
        .saddle_escape_reseed
        .as_ref()
        .expect("the derived ladder must reach into the well and mint an escape reseed");
    assert!(
        narrow_well_cost(reseed) < -1e-12,
        "the reseed {reseed:?} must land strictly below the saddle (f={:.6e})",
        narrow_well_cost(reseed),
    );
    assert!(
        reseed[1].abs() < 0.0625,
        "the descent only exists below the OLD ladder's last rung, so a reseed at or beyond it \
         would mean the well moved: {reseed:?}"
    );
}

// ─── #2612: a decrease the criterion cannot resolve is not an escape ──────────
//
// The two tests above pin the ladder's REACH. This one pins its ACCEPTANCE, and
// it is the production failure's exact shape: a genuinely negative eigenvalue,
// a ladder rung that genuinely lands inside the well, and a measured decrease
// that is BELOW the criterion's own resolution.
//
//   f(ρ) = ½ρ₀² + 0.02ρ₁⁴ − 5e-5·ρ₁²        H = diag(1, 0.24ρ₁² − 1e-4)
//
// At ρ = 0 the curvature is `−1e-4` — four orders above the
// `sqrt(EPSILON)·max(1,|H_kk|) ≈ 1.5e-8` roundoff margin, so this is a
// certified strict saddle and not a flat direction. The well is WIDE
// (`f < 0` for `|ρ₁| < 0.05`) and SHALLOW (`min f = −3.125e-8`), so:
//
//   * `α_min = sqrt(2·1e-7/1e-4) = 4.47e-2`, so the ladder runs
//     `1 → 0.5 → … → 0.03125` and the last rung lands INSIDE the well;
//   * the decrease there is `2.976e-8`, against `objective_resolution = 1e-7`.
//
// With the acceptance floor at the ARITHMETIC's resolution (`16ε|V|` ≈ 3.6e-15,
// what it used to be) that rung reads as a descent, the one-shot reseed is spent
// on it, and the retry — which used to be forbidden from adjudicating — refuses
// on the matrix's word. That is exactly what penguins' unbiased probe did four
// times over, at `λ_min ≈ −1e-6` and decreases of `2e-6` against a `1.228e-3`
// resolution.
//
// With the floor at the CRITERION's resolution the same rung is what it is: a
// number the criterion cannot distinguish from zero. The claim is unfalsifiable
// over its whole derived range, the verdict is withdrawn, and the point
// certifies.
/// The criterion's OWN evaluation error, as a deterministic function of `ρ`
/// (#2748).
///
/// "The criterion's resolution" has to be a property of the CRITERION. The
/// smooth well below is a polynomial, exact to `~1e-24` in binary64, so its
/// `2.976e-8` decrease is not merely representable there — it is EXACT, and
/// declaring it unresolvable was reading the optimizer's DECLARED tolerance as
/// if it were the criterion's error. That is the borrowed-`ε_f` defect #2690
/// exists to stop, and the ladder that now runs inside the adjudication
/// measures the real thing.
///
/// So the fixture supplies one. A REML criterion's value carries a component
/// that does not vary smoothly with `ρ`: the inner solve stops on its own
/// tolerance and a warm start from a neighbouring point lands on a slightly
/// different mode. This models that with a deterministic term — same `ρ`, same
/// value, in every lane and on every host, no RNG — of amplitude
/// [`WELL_EVALUATION_ERROR`], varying far faster than any ladder rung, so the
/// ladder measures it as `ε_f` exactly as it would in production.
///
/// It is EXACTLY ZERO at `ρ = 0`, so the baseline the escape compares against
/// is the smooth value and the test's arithmetic below stays legible.
/// # ⚠ The phase is load-bearing, and it is not a fudge
///
/// A symmetric second difference ANNIHILATES any odd perturbation exactly —
/// that is what it is for, and this module's `curvature_resolution` header says
/// so about the first-order term. An "evaluation error" modelled as
/// `A·sin(F·ρ₁)` is odd in `ρ₁`, so `f(+α) + f(−α)` cancels it to the last bit
/// and the ladder measures `ε_f = 2.6e-17` on a fixture planted with `5e-8`.
/// Measured, on the first attempt at this fixture. Real evaluation error has no
/// such symmetry, so the phase offset is what makes this model one.
fn well_evaluation_error(rho: &Array1<f64>) -> f64 {
    WELL_EVALUATION_ERROR
        * (WELL_ERROR_FREQUENCY * rho[1] + 3.0 * rho[0] + WELL_ERROR_PHASE).sin()
}

/// Amplitude of the well criterion's evaluation error.
///
/// Sized from the two properties the fixture has to have, both measured on it
/// rather than assumed:
///
/// * the ladder measures `ε_f = 4.6e-8` here, so `2ε_f = 9.2e-8` — larger than
///   the deepest descent the well offers (`3.125e-8`). The well is below what
///   this criterion can distinguish from zero, which is the fixture's thesis;
/// * Law 1's floor from the measured `(ε_f, M₄) = (4.6e-8, 0.48)` comes out
///   `1.7e-4`, ABOVE the `|λ_min| = 1e-4` in dispute — so the criterion cannot
///   resolve that curvature either, and the adjudication declines on the
///   measurement rather than on a declared tolerance.
///
/// It is also small enough that no ladder rung dips below the DECLARED
/// `objective_resolution` of `1e-7` (deepest trial: `-6.6e-8`), so the original
/// acceptance path is not what this test exercises.
const WELL_EVALUATION_ERROR: f64 = 3.0e-8;

/// How fast the error term varies in `ρ₁`. A power of two so the arithmetic is
/// exact, and large enough that consecutive ladder rungs — which halve — see
/// uncorrelated values rather than a smooth trend the fit would absorb into
/// `M₄`.
const WELL_ERROR_FREQUENCY: f64 = 1_048_576.0;

/// Phase offset, so the error term is neither even nor odd. See
/// [`well_evaluation_error`] — an odd one is cancelled exactly by the symmetric
/// average and models no error at all.
const WELL_ERROR_PHASE: f64 = 1.0;

fn unresolvable_well_cost(rho: &Array1<f64>) -> f64 {
    unresolvable_well_smooth(rho) + well_evaluation_error(rho)
}

/// The well itself, with no evaluation error: `f(ρ) = ½ρ₀² + 0.02ρ₁⁴ − 5e-5·ρ₁²`.
fn unresolvable_well_smooth(rho: &Array1<f64>) -> f64 {
    let r0 = rho[0];
    let r1 = rho[1];
    0.5 * r0 * r0 + 0.02 * r1 * r1 * r1 * r1 - 5.0e-5 * r1 * r1
}

fn unresolvable_well_eval(rho: &Array1<f64>) -> OuterEval {
    let r0 = rho[0];
    let r1 = rho[1];
    OuterEval {
        cost: unresolvable_well_cost(rho),
        gradient: array![r0, 0.08 * r1 * r1 * r1 - 1.0e-4 * r1],
        hessian: HessianValue::Dense(array![[1.0, 0.0], [0.0, 0.24 * r1 * r1 - 1.0e-4]]),
        inner_beta_hint: None,
    }
}

#[test]
fn a_descent_below_the_criterion_resolution_is_not_an_escape_2612() {
    // Pin the fixture's own arithmetic first, so a failure below is about the
    // acceptance floor and not about the well having moved.
    let rung = unresolvable_well_smooth(&array![0.0, 0.031_25]);
    assert!(
        rung < 0.0,
        "the ladder's last rung must land INSIDE the well, or this test would be about the \
         ladder's reach instead of its acceptance (f={rung:.6e})"
    );
    // And the decrease there must be below the CRITERION's own evaluation
    // error, not below a declared tolerance (#2748). Two evaluations each
    // accurate to `ε_f` carry `2ε_f`, so that is the smallest difference this
    // criterion can distinguish from zero — and the well is shallower than it,
    // everywhere.
    let deepest = -(5.0e-5_f64 * 5.0e-5) / (4.0 * 0.02);
    assert!(
        rung.abs() < 2.0 * WELL_EVALUATION_ERROR
            && deepest.abs() < 2.0 * WELL_EVALUATION_ERROR,
        "the well must be shallower than the criterion's own evaluation error, or the descent \
         is REAL and must be minted: rung={rung:.6e}, deepest={deepest:.6e}, \
         2*amplitude={:.6e}",
        2.0 * WELL_EVALUATION_ERROR
    );
    // And no trial may dip below the DECLARED resolution either, or the
    // original acceptance path fires and this test stops being about the
    // measured one.
    let deepest_trial = [1.0_f64, 0.5, 0.25, 0.125, 0.062_5, 0.031_25]
        .into_iter()
        .flat_map(|alpha| [alpha, -alpha])
        .map(|r1| unresolvable_well_cost(&array![0.0, r1]))
        .fold(f64::INFINITY, f64::min);
    assert!(
        deepest_trial > -1.0e-7,
        "no ladder rung may beat the DECLARED objective resolution, or the escape fires \
         through the pre-#2748 path: deepest trial={deepest_trial:.6e}"
    );

    let problem = OuterProblem::new(2)
        .with_gradient(Derivative::Analytic)
        .with_hessian(DeclaredHessianForm::Dense)
        .with_initial_rho(array![0.0, 0.0])
        .with_screen_initial_rho(false)
        .with_seed_config(gam_problem::SeedConfig {
            max_seeds: 1,
            seed_budget: 1,
            ..Default::default()
        });
    let mut obj = problem.build_objective(
        (),
        |_: &mut (), rho: &Array1<f64>| Ok(unresolvable_well_cost(rho)),
        |_: &mut (), rho: &Array1<f64>| Ok(unresolvable_well_eval(rho)),
        None::<fn(&mut ())>,
        None::<fn(&mut (), &Array1<f64>) -> Result<EfsEval, EstimationError>>,
    );
    let result = audit_stationary_point(&mut obj, array![0.0, 0.0], "unresolvable-well #2612")
        .expect(
            "a point whose only available descent is below the criterion's own resolution must \
             not be refused: no optimizer can reach past it, so the negative direction has no \
             operational content",
        );
    let cert = result
        .criterion_certificate
        .as_ref()
        .expect("a certified point carries its certificate");
    assert!(
        cert.certifies(),
        "the certificate must accept: {}",
        cert.summary()
    );
    assert_eq!(
        cert.curvature,
        CurvatureEvidence::CriterionContradicted,
        "the verdict must be recorded as WITHDRAWN, not as a PSD claim — the matrix is still \
         indefinite and nothing here showed the point is a minimum: {}",
        cert.summary()
    );
    assert!(
        result.saddle_escape_reseed.is_none(),
        "the one-shot reseed must NOT be spent on a decrease the criterion cannot resolve; \
         spending it is what left the retry pass with nothing to adjudicate: {:?}",
        result.saddle_escape_reseed
    );
    // NEGATIVE CONTROL for the fixture itself (#2748): the SMOOTH well really
    // does descend, and by a margin the arithmetic can carry. So this test is
    // about the criterion's evaluation error hiding that descent — not about
    // there being nothing to find, which would make the assertions above pass
    // vacuously.
    let roundoff = 16.0 * f64::EPSILON;
    assert!(
        rung < -roundoff,
        "the smooth well must offer a descent the ARITHMETIC can resolve, or this fixture \
         proves nothing about resolution: rung={rung:.6e} vs roundoff={roundoff:.6e}"
    );
}

// ─── #2612 the escape step is the objective's, not the falsifier's ───
//
// The #2357/#2155 fixtures above are DOUBLE WELLS: the saddle at ρ₁ = 0 sits
// exactly one unit from its minima, so a reseed capped at the falsifiability
// ladder's largest rung (α = 1, one e-fold in log-λ) lands on the answer by
// coincidence of scale. Nothing above can see a cap, because nothing above
// needs to travel.
//
// The #2612 multinomial ridge is the other shape, and it is the common one for
// a smoothing parameter: the criterion is stationary, gently CONCAVE, and falls
// monotonically along the free direction all the way to the box wall — the
// λ → 0 face — with no interior minimiser at all. Distilled:
//
//   f(ρ) = ½·ρ₀²  −  ½·ε·ρ₁²,        ε = 1e-3
//   ∇f    = (ρ₀, −ε·ρ₁)
//   H     = diag(1, −ε)
//
// At ρ = (0, 0) the gradient vanishes and `H` is indefinite, exactly as in the
// wells above; unlike them, `argmin` over the box is the FACE ρ₁ = ±rho_bound
// and the descent runs the whole width of the box. A reseed capped at α = 1
// covers 1/30th of it, which on the real fixture cost one
// `OUTER_SADDLE_ESCAPE_BUDGET` unit per e-fold and refused the fit six e-folds
// short.
//
// Both directions descend identically here, so the sign is a tie and either
// face is correct; the assertions are on |ρ₁|.
const RIDGE_CURVATURE: f64 = 1e-3;

fn ridge_cost(rho: &Array1<f64>) -> f64 {
    0.5 * rho[0] * rho[0] - 0.5 * RIDGE_CURVATURE * rho[1] * rho[1]
}

fn ridge_eval(rho: &Array1<f64>) -> OuterEval {
    OuterEval {
        cost: ridge_cost(rho),
        gradient: array![rho[0], -RIDGE_CURVATURE * rho[1]],
        hessian: HessianValue::Dense(array![[1.0, 0.0], [0.0, -RIDGE_CURVATURE]]),
        inner_beta_hint: None,
    }
}

/// The face this ridge runs to. Stated by the fixture rather than inherited
/// from the default box, so the assertions below are about the escape and not
/// about `rho_bound`'s value.
const RIDGE_BOX_FACE: f64 = 30.0;

fn ridge_problem() -> OuterProblem {
    OuterProblem::new(2)
        .with_gradient(Derivative::Analytic)
        .with_hessian(DeclaredHessianForm::Dense)
        .with_bounds(
            array![-RIDGE_BOX_FACE, -RIDGE_BOX_FACE],
            array![RIDGE_BOX_FACE, RIDGE_BOX_FACE],
        )
        .with_initial_rho(array![0.0, 0.0])
        .with_screen_initial_rho(false)
        .with_seed_config(gam_problem::SeedConfig {
            max_seeds: 1,
            seed_budget: 1,
            ..Default::default()
        })
}

#[test]
fn saddle_escape_reseed_travels_to_the_box_face_on_a_monotone_ridge_2612() {
    // The load-bearing statement of #2612, at the unit that decides it: on a
    // stationary point whose free direction descends monotonically to the box,
    // the minted reseed must be the step the OBJECTIVE supports — the face —
    // and not the largest rung of the falsifiability ladder.
    let problem = ridge_problem();
    let mut obj = problem.build_objective(
        (),
        |_: &mut (), rho: &Array1<f64>| Ok(ridge_cost(rho)),
        |_: &mut (), rho: &Array1<f64>| Ok(ridge_eval(rho)),
        None::<fn(&mut ())>,
        None::<fn(&mut (), &Array1<f64>) -> Result<EfsEval, EstimationError>>,
    );
    let rejection = audit_stationary_point(&mut obj, array![0.0, 0.0], "monotone-ridge #2612")
        .expect_err("a stationary point on an indefinite ridge must be refused, not certified");
    let result = &rejection.result;
    let cert = result
        .criterion_certificate
        .as_ref()
        .expect("refused certificate must be recorded");
    assert!(
        cert.is_stationary(),
        "the gradient vanishes identically at the ridge point: {}",
        cert.summary()
    );
    assert_eq!(
        cert.hessian_psd(),
        Some(false),
        "diag(1, -{RIDGE_CURVATURE}) must read indefinite: {}",
        cert.summary()
    );
    let reseed = result
        .saddle_escape_reseed
        .as_ref()
        .expect("a refused indefinite ridge must mint a negative-curvature escape reseed");
    let face = RIDGE_BOX_FACE;
    assert!(
        (reseed[1].abs() - face).abs() < 1e-6,
        "the reseed must land on the box face |ρ₁| = {face}, which is where the descent this \
         escape confirmed actually ends; got ρ = {reseed:?}. A reseed at |ρ₁| ≈ 1 is the \
         falsifiability ladder's largest rung being reused as a step length — the #2612 defect, \
         which cost one escape-budget unit per e-fold and refused the fit six e-folds short"
    );
    assert!(
        reseed[0].abs() < 1e-9,
        "the escape must move ONLY the indefinite coordinate, holding the PSD ρ₀ at its \
         optimum: {reseed:?}"
    );
    // The fixture's own negative control: the face really is far from the rung
    // the old code could reach, so passing is not an accident of scale (this is
    // exactly what the double wells above cannot say).
    assert!(
        face > 10.0,
        "the ridge fixture needs a box many e-folds wide or it cannot distinguish a travelling \
         escape from a capped one: face={face}"
    );
}

#[test]
fn saddle_escape_expansion_does_not_overshoot_a_genuine_well_2612() {
    // The guard on the repair: extending the step must stop where the objective
    // stops improving, so a genuine interior well is still escaped TO ITS
    // MINIMUM and not past it to the box. The #2357 double well has its minima
    // at ρ₁ = ±1 and rises steeply beyond (f(±2) = +2 against f(±1) = −¼), so an
    // expansion that ignored the objective would land at ρ₁ = ±30 and be caught
    // here.
    let problem = saddle_problem();
    let mut obj = problem.build_objective(
        (),
        |_: &mut (), rho: &Array1<f64>| Ok(saddle_cost(rho)),
        |_: &mut (), rho: &Array1<f64>| Ok(saddle_eval(rho)),
        None::<fn(&mut ())>,
        None::<fn(&mut (), &Array1<f64>) -> Result<EfsEval, EstimationError>>,
    );
    let rejection = audit_stationary_point(&mut obj, array![0.0, 0.0], "well-overshoot #2612")
        .expect_err("an interior strict saddle must be refused, not certified");
    let reseed = rejection
        .result
        .saddle_escape_reseed
        .as_ref()
        .expect("a refused interior saddle must mint a negative-curvature escape reseed")
        .clone();
    assert!(
        reseed[1].abs() <= 1.0 + 1e-9,
        "the double well's descent ENDS at |ρ₁| = 1 (f = −¼, against f = +2 at |ρ₁| = 2), so an \
         expanded step must stop there rather than running to the box: {reseed:?}"
    );
    assert!(
        saddle_cost(&reseed) < saddle_cost(&array![0.0, 0.0]) - 1e-9,
        "and it must still be a strict descent: f={} against the saddle's {}",
        saddle_cost(&reseed),
        saddle_cost(&array![0.0, 0.0]),
    );
}

#[test]
fn outer_search_clears_a_monotone_ridge_on_the_gradient_only_plan_2612() {
    // End to end, on the plan the multinomial fit actually runs. `#2612`'s fit
    // is a custom family, and `fit_custom_family` sets `prefer_gradient_only`
    // (the generic REML/LAML Hessian consumes the order-four family tower,
    // so the exact Hessian is reserved for the terminal certificate), which
    // routes the search to BFGS. That matters here: an ARC search reads the
    // analytic Hessian and can follow a negative-curvature direction on its
    // own, so a pipeline test that let the planner pick ARC would be green
    // whether or not the escape can travel. This one pins the gradient-only
    // plan, where the escape IS the only thing that can cross the ridge.
    //
    // The sharp discriminator for the step rule is the unit test above; this
    // asserts the property the fit needs — reach the face and certify there,
    // with ρ₁ railed the reduced Hessian is the 1×1 PSD block [1].
    let problem = ridge_problem().with_prefer_gradient_only(true);
    let mut obj = problem.build_objective(
        (),
        |_: &mut (), rho: &Array1<f64>| Ok(ridge_cost(rho)),
        |_: &mut (), rho: &Array1<f64>| Ok(ridge_eval(rho)),
        None::<fn(&mut ())>,
        None::<fn(&mut (), &Array1<f64>) -> Result<EfsEval, EstimationError>>,
    );
    let result = problem
        .run(&mut obj, "monotone-ridge pipeline #2612")
        .expect("the outer search must reach the ridge's box face and certify there");
    let face = RIDGE_BOX_FACE;
    assert!(
        result.converged(),
        "must converge on the face, not refuse on the ridge: rho={:?}",
        result.rho
    );
    assert!(
        (result.rho[1].abs() - face).abs() < 1e-4,
        "must land on the box face |ρ₁| = {face}: rho={:?}",
        result.rho,
    );
    assert!(
        result
            .criterion_certificate
            .as_ref()
            .is_some_and(|c| c.certifies() && c.hessian_psd() == Some(true)),
        "the face is a constrained minimum: with ρ₁ railed the reduced Hessian is [1], PSD",
    );
}

// ─── #2612 a criterion whose minimum over the box is a CORNER ────────
//
// The ridge fixture above has ONE indefinite coordinate, so one escape clears
// it. That is still inside the premise `OUTER_SADDLE_ESCAPE_BUDGET = 3` rests
// on — *"a genuine saddle is cleared in one escape"* — and so it cannot test
// that premise.
//
// This one is outside it, and it is the shape the multinomial fit has:
//
//   f(ρ) = ½·ρ₀²  −  ½·Σ_{j=1..K} ε_j·ρ_j²,     ε = (5, 4, 3, 2, 1)·1e-3
//
// stationary at the origin, `H = diag(1, −ε₁, …, −ε_K)` indefinite in EVERY
// ρ_j, and `argmin` over the box is the CORNER `ρ₀ = 0, ρ_j = ±30`. There is
// nothing pathological here — a concave quadratic on a box attains its minimum
// at a vertex, and refusing that point is refusing the answer.
//
// The escape can only retire the coordinates one at a time: the eigenvector of
// the most negative curvature is a single axis, the expanded step runs to the
// face along it, and that coordinate rails. `judged_subspace_basis` then makes
// the next escape's direction exactly zero there, so the free block shrinks by
// one per escape and the sequence is monotone in TWO quantities at once — the
// criterion strictly decreases and the free dimension strictly shrinks. It is
// finite by construction and needs `K` escapes.
//
// With `K = 5` against a budget of 3 the fit is refused with two coordinates
// still un-retired. That is the measurement the premise cannot survive, and it
// is here as a fixture rather than as an argument.
const CORNER_CURVATURES: [f64; 5] = [5.0e-3, 4.0e-3, 3.0e-3, 2.0e-3, 1.0e-3];
const CORNER_BOX_FACE: f64 = 30.0;

fn corner_cost(rho: &Array1<f64>) -> f64 {
    let mut cost = 0.5 * rho[0] * rho[0];
    for (j, &curvature) in CORNER_CURVATURES.iter().enumerate() {
        cost -= 0.5 * curvature * rho[j + 1] * rho[j + 1];
    }
    cost
}

fn corner_eval(rho: &Array1<f64>) -> OuterEval {
    let dim = CORNER_CURVATURES.len() + 1;
    let mut gradient = Array1::<f64>::zeros(dim);
    let mut hessian = Array2::<f64>::zeros((dim, dim));
    gradient[0] = rho[0];
    hessian[[0, 0]] = 1.0;
    for (j, &curvature) in CORNER_CURVATURES.iter().enumerate() {
        gradient[j + 1] = -curvature * rho[j + 1];
        hessian[[j + 1, j + 1]] = -curvature;
    }
    OuterEval {
        cost: corner_cost(rho),
        gradient,
        hessian: HessianValue::Dense(hessian),
        inner_beta_hint: None,
    }
}

fn corner_problem() -> OuterProblem {
    let dim = CORNER_CURVATURES.len() + 1;
    OuterProblem::new(dim)
        .with_gradient(Derivative::Analytic)
        .with_hessian(DeclaredHessianForm::Dense)
        .with_prefer_gradient_only(true)
        .with_bounds(
            Array1::from_elem(dim, -CORNER_BOX_FACE),
            Array1::from_elem(dim, CORNER_BOX_FACE),
        )
        .with_initial_rho(Array1::<f64>::zeros(dim))
        .with_screen_initial_rho(false)
        .with_seed_config(gam_problem::SeedConfig {
            max_seeds: 1,
            seed_budget: 1,
            ..Default::default()
        })
}

#[test]
fn outer_search_reaches_a_corner_minimum_that_needs_more_than_one_escape_2612() {
    // The premise `OUTER_SADDLE_ESCAPE_BUDGET` rests on, as a fixture. Every
    // escape here is a strict descent AND retires one coordinate onto a rail, so
    // the sequence is finite by construction; capping it by a count refuses a
    // point the criterion is still descending toward.
    let problem = corner_problem();
    let mut obj = problem.build_objective(
        (),
        |_: &mut (), rho: &Array1<f64>| Ok(corner_cost(rho)),
        |_: &mut (), rho: &Array1<f64>| Ok(corner_eval(rho)),
        None::<fn(&mut ())>,
        None::<fn(&mut (), &Array1<f64>) -> Result<EfsEval, EstimationError>>,
    );
    let result = problem
        .run(&mut obj, "corner-minimum pipeline #2612")
        .expect(
            "a concave quadratic on a box attains its minimum at a VERTEX; refusing that point \
             is refusing the answer",
        );
    assert!(
        result.converged(),
        "must converge at the corner: rho={:?}",
        result.rho
    );
    assert!(
        result.rho[0].abs() < 1e-4,
        "the convex coordinate must sit at its optimum ρ₀ = 0: {:?}",
        result.rho
    );
    for j in 1..=CORNER_CURVATURES.len() {
        assert!(
            (result.rho[j].abs() - CORNER_BOX_FACE).abs() < 1e-4,
            "every concave coordinate must be retired onto a face: coordinate {j} sits at {} in \
             {:?}",
            result.rho[j],
            result.rho
        );
    }
    assert!(
        result
            .criterion_certificate
            .as_ref()
            .is_some_and(|c| c.certifies() && c.hessian_psd() == Some(true)),
        "with every concave coordinate railed the reduced Hessian is [1], PSD",
    );
}
