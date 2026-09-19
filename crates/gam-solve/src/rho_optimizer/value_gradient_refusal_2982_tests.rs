use super::*;
use ndarray::array;
use std::sync::Mutex;

/// The planted refusal. Only records carrying it are kept, so another test's
/// log traffic in the same process cannot satisfy or spoil the assertions.
const PLANTED_REASON: &str = "planted gam#2982 derivative-assembly refusal";

static CAPTURED: Mutex<Vec<String>> = Mutex::new(Vec::new());

struct PlantedReasonLogger;

impl log::Log for PlantedReasonLogger {
    fn enabled(&self, _metadata: &log::Metadata<'_>) -> bool {
        true
    }

    fn log(&self, record: &log::Record<'_>) {
        let line = record.args().to_string();
        if line.contains(PLANTED_REASON)
            && let Ok(mut captured) = CAPTURED.lock()
        {
            captured.push(line);
        }
    }

    fn flush(&self) {}
}

static PLANTED_REASON_LOGGER: PlantedReasonLogger = PlantedReasonLogger;

/// gam#2982: a value+gradient evaluation the first-order bridge refuses must
/// hand `opt` the refusal's own reason and write that reason to the log.
///
/// `opt`'s backtracking and Strong-Wolfe searches evaluate the gradient at every
/// trial that clears Armijo. On a recoverable failure they halve the step and
/// keep only a `nonfinite_seen` flag, so the refusal's reason survives only in
/// the record this bridge writes. On gnomon's 300-row survival calibrate fixture
/// 408 of 839 such evaluations failed with no record at all. The line search
/// discarded steps up to 4.36 below the incumbent and ended "Line search failed
/// (nonfinite seen)" with no cause named.
///
/// The record is read from the process's `log` backend, which this test
/// installs. nextest runs every test in its own process, so the backend is this
/// test's to install.
#[test]
fn refused_value_gradient_evaluation_names_its_reason_2982() {
    assert!(
        log::set_logger(&PLANTED_REASON_LOGGER).is_ok(),
        "another `log` backend owns this process, so the bridge's own record cannot be read here"
    );
    log::set_max_level(log::LevelFilter::Info);

    let problem = OuterProblem::new(1).with_gradient(Derivative::Analytic);
    let mut obj = problem.build_objective_with_eval_order(
        (),
        |_: &mut (), _: &Array1<f64>| Ok(1.0),
        |_: &mut (), _: &Array1<f64>| {
            Err(EstimationError::InvalidInput(
                "legacy eager eval should not run".to_string(),
            ))
        },
        |_: &mut (), _: &Array1<f64>, _: OuterEvalOrder| {
            Err(EstimationError::TrialPointRefused {
                reason: PLANTED_REASON.to_string(),
            })
        },
        None::<fn(&mut ())>,
        None::<fn(&mut (), &Array1<f64>) -> Result<EfsEval, EstimationError>>,
    );
    let mut bridge = OuterFirstOrderBridge {
        obj: &mut obj,
        layout: OuterThetaLayout::new(1, 0),
        outer_inner_cap: None,
        first_order_evals: 0,
        g_norm_initial: None,
        last_g_norm: None,
        last_value_grad_rho: None,
        value_probe_cache: Vec::new(),
        cost_stall: None,
        cost_stall_bounds: None,
        consecutive_probe_refusals: 0,
        accepted_steps: None,
        pending_first_order: Vec::new(),
        incumbent: None,
        stratum_rank: None,
        stratum_probe: None,
    };

    let refusal = match FirstOrderObjective::eval_grad(&mut bridge, &array![0.5]) {
        Ok(_) => panic!("a refused value+gradient evaluation must not return a sample"),
        Err(err) => err,
    };
    assert!(
        refusal.is_recoverable(),
        "a trial-point refusal stays recoverable, so the line search shortens the step: {refusal}"
    );
    assert!(
        refusal.to_string().contains(PLANTED_REASON),
        "the error handed to opt must carry the refusal's own reason, got: {refusal}"
    );

    let captured = CAPTURED
        .lock()
        .map(|records| records.clone())
        .unwrap_or_default();
    assert!(
        captured.iter().any(|record| {
            record.contains("outer eval end order=ValueAndGradient")
                && record.contains("outcome=recoverable")
                && record.contains("theta=[+0.5000]")
        }),
        "the bridge must write the refused evaluation's order, outcome, point and reason; \
         records naming the planted reason: {captured:?}"
    );
}
