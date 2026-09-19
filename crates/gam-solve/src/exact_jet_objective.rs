//! A second-order objective built from one closure that returns the exact
//! value, gradient and Hessian at a point, and the certified second-order
//! minimisation that consumes it.
//!
//! `opt` publishes [`opt::FusedObjective`] for first-order callers; this is
//! its second-order counterpart. Small application-specific outer problems
//! (a one-dimensional smoothing coordinate, a block's own log-precisions)
//! hand their analytic jet here instead of iterating a hand-rolled Newton,
//! so the step geometry, the stationarity certificate and the refusal of an
//! unconverged iterate all come from `opt`.

use ndarray::{Array1, Array2};
use opt::{
    Bounds, FallbackPolicy, FirstOrderObjective, FirstOrderSample, GradientTolerance,
    MaxIterations, NewtonTrustRegion, NewtonTrustRegionError, ObjectiveEvalError,
    SecondOrderObjective, SecondOrderSample, Solution, ZerothOrderObjective,
};

/// A [`SecondOrderObjective`] from a fused `(value, gradient, Hessian)`
/// evaluator, with a bit-exact cache of the last point so a cost, gradient
/// and Hessian request at one point evaluate it once.
pub struct ExactJetObjective<F> {
    evaluator: F,
    cached: Option<(Array1<f64>, SecondOrderSample)>,
}

impl<F> ExactJetObjective<F>
where
    F: FnMut(&Array1<f64>) -> Result<SecondOrderSample, ObjectiveEvalError>,
{
    pub fn new(evaluator: F) -> Self {
        Self {
            evaluator,
            cached: None,
        }
    }

    fn sample(&mut self, x: &Array1<f64>) -> Result<SecondOrderSample, ObjectiveEvalError> {
        if let Some((cached_x, sample)) = self.cached.as_ref()
            && cached_x.len() == x.len()
            && cached_x
                .iter()
                .zip(x.iter())
                .all(|(a, b)| a.to_bits() == b.to_bits())
        {
            return Ok(sample.clone());
        }
        let sample = (self.evaluator)(x)?;
        let n = x.len();
        if sample.gradient.len() != n {
            return Err(ObjectiveEvalError::fatal(format!(
                "exact-jet objective returned gradient length {}, expected {n}",
                sample.gradient.len()
            )));
        }
        if let Some(hessian) = sample.hessian.as_ref()
            && hessian.dim() != (n, n)
        {
            return Err(ObjectiveEvalError::fatal(format!(
                "exact-jet objective returned a {:?} Hessian, expected ({n}, {n})",
                hessian.dim()
            )));
        }
        // A non-finite jet is a property of this trial point: the solver
        // regularises away from it rather than accepting it.
        let finite = sample.value.is_finite()
            && sample.gradient.iter().all(|g| g.is_finite())
            && sample
                .hessian
                .as_ref()
                .is_some_and(|h| h.iter().all(|v| v.is_finite()));
        if !finite {
            return Err(ObjectiveEvalError::recoverable(
                "exact-jet objective is not finite at this point",
            ));
        }
        self.cached = Some((x.clone(), sample.clone()));
        Ok(sample)
    }
}

impl<F> ZerothOrderObjective for ExactJetObjective<F>
where
    F: FnMut(&Array1<f64>) -> Result<SecondOrderSample, ObjectiveEvalError>,
{
    fn eval_cost(&mut self, x: &Array1<f64>) -> Result<f64, ObjectiveEvalError> {
        self.sample(x).map(|sample| sample.value)
    }
}

impl<F> FirstOrderObjective for ExactJetObjective<F>
where
    F: FnMut(&Array1<f64>) -> Result<SecondOrderSample, ObjectiveEvalError>,
{
    fn eval_grad(&mut self, x: &Array1<f64>) -> Result<FirstOrderSample, ObjectiveEvalError> {
        self.sample(x).map(|sample| FirstOrderSample {
            value: sample.value,
            gradient: sample.gradient,
        })
    }
}

impl<F> SecondOrderObjective for ExactJetObjective<F>
where
    F: FnMut(&Array1<f64>) -> Result<SecondOrderSample, ObjectiveEvalError>,
{
    fn eval_hessian(&mut self, x: &Array1<f64>) -> Result<SecondOrderSample, ObjectiveEvalError> {
        self.sample(x)
    }
}

/// The exact jet `(value, gradient, Hessian)` as a [`SecondOrderSample`].
pub fn exact_jet(value: f64, gradient: Array1<f64>, hessian: Array2<f64>) -> SecondOrderSample {
    SecondOrderSample {
        value,
        gradient,
        hessian: Some(hessian),
        decrement_bands: None,
    }
}

/// Why [`certified_newton_minimum`] has no minimiser to return.
#[derive(Debug)]
pub enum CertifiedMinimumError {
    /// The solver failed: a fatal evaluation, a non-finite seed, a
    /// saturated regularisation it could not step past.
    Solver(NewtonTrustRegionError),
    /// The solver stopped without a stationarity certificate.
    Uncertified(Box<Solution>),
}

impl CertifiedMinimumError {
    /// The last iterate the solver reported, when it reported one: the
    /// checkpoint a caller can resume from.
    pub fn last_point(&self) -> Option<&Array1<f64>> {
        match self {
            Self::Solver(
                NewtonTrustRegionError::TrustRegionRejectFloor { last_solution }
                | NewtonTrustRegionError::MaxIterationsReached { last_solution },
            ) => Some(&last_solution.final_point),
            Self::Solver(_) => None,
            Self::Uncertified(solution) => Some(&solution.final_point),
        }
    }
}

impl std::fmt::Display for CertifiedMinimumError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Solver(error) => write!(f, "Newton trust region failed: {error}"),
            Self::Uncertified(solution) => write!(
                f,
                "Newton trust region stopped uncertified ({}) at {:?} with value {} and \
                 projected gradient norm {:?}",
                solution.termination.label(),
                solution.final_point.to_vec(),
                solution.final_value,
                solution.final_gradient_norm,
            ),
        }
    }
}

impl std::error::Error for CertifiedMinimumError {}

/// Minimise `evaluator` from `x0` with `opt`'s Newton trust region on the
/// exact jet: no iteration budget, no demotion to a quasi-Newton model, and
/// only a certified stationary point is a result. Everything else is an
/// error.
///
/// A sample that carries [`opt::DecrementBands`] is certified when its
/// Newton decrement falls inside the objective's own rounding band, and the
/// gradient tolerance is not consulted; pass `None` for it then. A sample
/// without bands is certified by a projected gradient under
/// `gradient_tolerance` with a positive semidefinite reduced Hessian.
pub fn certified_newton_minimum<F>(
    x0: Array1<f64>,
    bounds: Option<Bounds>,
    gradient_tolerance: Option<GradientTolerance>,
    evaluator: F,
) -> Result<Solution, CertifiedMinimumError>
where
    F: FnMut(&Array1<f64>) -> Result<SecondOrderSample, ObjectiveEvalError>,
{
    let max_iterations = MaxIterations::new(usize::MAX)
        .expect("usize::MAX is a valid iteration count");
    let mut solver = NewtonTrustRegion::new(x0, ExactJetObjective::new(evaluator))
        .with_fallback_policy(FallbackPolicy::Never)
        .with_max_iterations(max_iterations);
    if let Some(gradient_tolerance) = gradient_tolerance {
        solver = solver.with_gradient_tolerance(gradient_tolerance);
    }
    if let Some(bounds) = bounds {
        solver = solver.with_bounds(bounds);
    }
    let solution = solver.run().map_err(CertifiedMinimumError::Solver)?;
    if solution.status().is_success() {
        Ok(solution)
    } else {
        Err(CertifiedMinimumError::Uncertified(Box::new(solution)))
    }
}

