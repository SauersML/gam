//! Alternating fit of a decomposition's pieces and its per-position minimal supports
//! (#2951).
//!
//! # The problem
//!
//! Pieces are parameterized by a vector `θ` such that they sum to the teacher's
//! parameters exactly for every `θ` (a frame and its canonical dual, for instance).
//! At a declared or derived fidelity `ε`, the objective is the smallest admissible
//! supports: every position `t` keeps a set `S_t` with `KL_t(θ; S) ≤ ε`, and the pieces
//! are chosen so that those sets can be small.
//!
//! # The alternation
//!
//! 1. **Supports.** [`minimal_support`] from the current supports (a warm start),
//!    at the current `θ`.
//! 2. **Pieces.** At fixed supports, `θ` moves to the analytic centre of the feasible
//!    set, minimizing the log barrier
//!
//!    ```text
//!    B(θ) = −Σ_t log(ε − KL_t(θ; S)).
//!    ```
//!
//!    `B` is finite exactly on the admissible set, so no step can leave it, and its
//!    minimizer maximizes every position's slack together, with nothing to weigh.
//!    The step is `gam_geometry`'s trust region with Steihaug-CG on the exact second-order
//!    model of `B`:
//!
//!    ```text
//!    ∇B  = Σ_t w_t ∇KL_t,                             w_t = 1 / (ε − KL_t),
//!    H v = Σ_t [ w_t ∇²KL_t v + w_t² (∇KL_t · v) ∇KL_t ],
//!    ```
//!
//!    with `∇²KL_t` the exact Hessian of `KL_t` in `θ` (its product comes from the
//!    executor). The Gauss–Newton curvature would drop the residual term
//!    `Σ (q − p)·∂²z`, negligible only near the target; continuation starts far from it
//!    (tens of nats), where that model mispredicts every step by a constant factor, the
//!    radius never grows, and the fit stalls (measured on the 4-layer target). The
//!    exact model may be indefinite, which Steihaug-CG handles. Slack opened here lets
//!    step 1 remove more.
//!
//! Within one fidelity level, supports never regain a piece and `θ` never leaves the
//! admissible set, so the kept count is non-increasing. Each alternation takes one
//! trust-region iteration resumed from the previous one (its radius and certificate
//! scale carried), so the pieces never converge to the centre for supports the next
//! search replaces. The level ends when the supports stop shrinking and the
//! certificate holds: the fit comes only from a converged optimization.
//!
//! # Continuation in the fidelity
//!
//! Started at the declared `ε` from the full support, the alternation walks down from
//! dense supports, and pieces are reshaped only to open slack for nearly full supports.
//! The fit instead follows a path of levels from sparse to the declared fidelity. The
//! first level is set by the empty support: `ε₀` is its largest per-position
//! divergence, so the empty support is admissible at the first level
//! `ε·2^K > ε₀` and pieces are reshaped for sparse supports from the start. Each next
//! level halves the previous one down to `ε`; the supports carried over are repaired
//! (`minimal_support` restores pieces where they now violate) and searched again.
//! Halving discretizes the path; it is not a property of the result.
//!
//! The executor runs the model: per-position divergences, and the three products
//! above. The barrier, its weights, the trust region and the stopping rule live here.
use gam_geometry::manifolds::euclidean::EuclideanManifold;
use gam_geometry::{GeometryError, GeometryResult, RiemannianObjective, RiemannianTrustRegion, TrustRegionTermination};
use ndarray::{Array1, Array2, ArrayView1, ArrayView2};
use std::fmt;

use super::minimal_support::{
    Fidelity, MinimalSupport, SupportError, SupportEvaluation, SupportExecutor, minimal_support,
};

/// Runs the model at pieces `θ` and supports `keep` (`P × C`, false = removed).
pub trait PieceExecutor {
    /// Per-position divergence and removal predictions, as [`SupportExecutor`].
    fn supports(&mut self, theta: ArrayView1<'_, f64>, keep: ArrayView2<'_, bool>) -> Result<SupportEvaluation, String>;
    /// `KL_t` per position.
    fn divergence(&mut self, theta: ArrayView1<'_, f64>, keep: ArrayView2<'_, bool>) -> Result<Array1<f64>, String>;
    /// `Σ_t w_t ∇_θ KL_t`.
    fn weighted_gradient(&mut self, theta: ArrayView1<'_, f64>, keep: ArrayView2<'_, bool>, weights: ArrayView1<'_, f64>) -> Result<Array1<f64>, String>;
    /// `(∇_θ KL_t · v)_t`.
    fn directional(&mut self, theta: ArrayView1<'_, f64>, keep: ArrayView2<'_, bool>, v: ArrayView1<'_, f64>) -> Result<Array1<f64>, String>;
    /// `Σ_t w_t ∇²KL_t v`, with `∇²KL_t` the exact Hessian of `KL_t` in `θ`.
    fn weighted_hessian(&mut self, theta: ArrayView1<'_, f64>, keep: ArrayView2<'_, bool>, weights: ArrayView1<'_, f64>, v: ArrayView1<'_, f64>) -> Result<Array1<f64>, String>;
}

/// Why the alternation stopped without a result.
#[derive(Debug)]
pub enum SupportFitError {
    Support(SupportError),
    Executor(String),
    Geometry(GeometryError),
}

impl fmt::Display for SupportFitError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Support(e) => write!(f, "support_fit: {e}"),
            Self::Executor(e) => write!(f, "support_fit: executor failed: {e}"),
            Self::Geometry(e) => write!(f, "support_fit: trust region failed: {e}"),
        }
    }
}

impl std::error::Error for SupportFitError {}

/// One alternation.
#[derive(Clone, Debug, PartialEq)]
pub struct Alternation {
    /// The fidelity level of this alternation.
    pub level: f64,
    /// Kept pieces summed over positions after the support step.
    pub kept: usize,
    /// Barrier value before and after the piece step.
    pub barrier_before: f64,
    pub barrier_after: f64,
    /// Trust-region iterations and whether its first-order certificate held.
    pub iterations: usize,
    pub certified: bool,
}

/// The fitted pieces and supports.
#[derive(Clone, Debug)]
pub struct SupportFit {
    pub theta: Array1<f64>,
    pub supports: MinimalSupport,
    pub alternations: Vec<Alternation>,
}

struct AtTheta<'a, E: PieceExecutor> {
    executor: &'a mut E,
    theta: ArrayView1<'a, f64>,
}

impl<E: PieceExecutor> SupportExecutor for AtTheta<'_, E> {
    fn evaluate(&mut self, keep: ArrayView2<'_, bool>) -> Result<SupportEvaluation, String> {
        self.executor.supports(self.theta, keep)
    }
}

struct Barrier<'a, E: PieceExecutor> {
    executor: &'a mut E,
    keep: &'a Array2<bool>,
    fidelity: Fidelity,
    error: Option<String>,
}

impl<E: PieceExecutor> Barrier<'_, E> {
    /// The barrier value and the per-position weights `w_t` of `∇B = Σ_t w_t ∇KL_t`, or
    /// `None` outside the admissible set.
    ///
    /// Per position, `B = −Σ_t log(ε − KL_t)` and `w_t = 1/(ε − KL_t)`. Under the mean
    /// form, `B = −log(ε − m)` with `m` the mean divergence, and `w_t = 1/(P (ε − m))`.
    fn weights(&mut self, theta: ArrayView1<'_, f64>) -> Result<Option<(f64, Array1<f64>)>, String> {
        let kl = self.executor.divergence(theta, self.keep.view())?;
        match self.fidelity {
            Fidelity::PerPosition(eps) => {
                if kl.iter().any(|&v| !(v < eps)) {
                    return Ok(None);
                }
                let value = -kl.iter().map(|&v| (eps - v).ln()).sum::<f64>();
                Ok(Some((value, kl.mapv(|v| 1.0 / (eps - v)))))
            }
            Fidelity::Mean(eps) => {
                let p = kl.len() as f64;
                let mean = kl.sum() / p;
                if !(mean < eps) {
                    return Ok(None);
                }
                Ok(Some((-(eps - mean).ln(), Array1::from_elem(kl.len(), 1.0 / (p * (eps - mean))))))
            }
        }
    }

    /// Per-position weights of the barrier's rank-one Hessian term along `d_t = ∇KL_t · v`:
    /// per position `w_t² d_t`; under the mean form the one constraint's
    /// `(Σ d / P) / (ε − m)² / P` at every position.
    fn rank_one_weights(&self, w: &Array1<f64>, d: &Array1<f64>) -> Array1<f64> {
        match self.fidelity {
            Fidelity::PerPosition(_) => w * w * d,
            Fidelity::Mean(_) => {
                let p = d.len() as f64;
                // w_t = 1/(P (ε − m)), so 1/(ε − m)² = (P w)².
                let scale = (p * w[0]).powi(2) * d.sum() / p / p;
                Array1::from_elem(d.len(), scale)
            }
        }
    }

    fn fail(&mut self, e: String) -> GeometryError {
        self.error = Some(e);
        GeometryError::InvalidPoint("support_fit: the executor failed")
    }
}

impl<E: PieceExecutor> RiemannianObjective for Barrier<'_, E> {
    fn value_gradient(&mut self, point: ArrayView1<'_, f64>) -> GeometryResult<(f64, Array1<f64>)> {
        match self.weights(point) {
            Err(e) => Err(self.fail(e)),
            // Outside the admissible set the barrier is +∞; a trial there reaches the trust
            // region's acceptance ratio as a non-finite reduction and is rejected.
            Ok(None) => Ok((f64::INFINITY, Array1::zeros(point.len()))),
            Ok(Some((value, w))) => {
                let g = self.executor.weighted_gradient(point, self.keep.view(), w.view()).map_err(|e| self.fail(e))?;
                Ok((value, g))
            }
        }
    }

    fn hessian_vector_product(&mut self, point: ArrayView1<'_, f64>, tangent: ArrayView1<'_, f64>) -> GeometryResult<Option<Array1<f64>>> {
        let (_, w) = match self.weights(point) {
            Err(e) => return Err(self.fail(e)),
            Ok(None) => return Ok(None),
            Ok(Some(pair)) => pair,
        };
        let mut out = self.executor.weighted_hessian(point, self.keep.view(), w.view(), tangent).map_err(|e| self.fail(e))?;
        let d = self.executor.directional(point, self.keep.view(), tangent).map_err(|e| self.fail(e))?;
        let rank_one = self.rank_one_weights(&w, &d);
        out += &self.executor.weighted_gradient(point, self.keep.view(), rank_one.view()).map_err(|e| self.fail(e))?;
        Ok(Some(out))
    }
}

/// Alternate supports and pieces along the fidelity path to `eps` (module docs).
pub fn fit_supports_and_pieces<E: PieceExecutor>(
    executor: &mut E,
    theta: Array1<f64>,
    positions: usize,
    pieces: usize,
    eps: f64,
    mean: bool,
    sequence: usize,
) -> Result<SupportFit, SupportFitError> {
    let form = |level: f64| if mean { Fidelity::Mean(level) } else { Fidelity::PerPosition(level) };
    let mut theta = theta;
    let empty = Array2::from_elem((positions, pieces), false);
    let widest = executor
        .divergence(theta.view(), empty.view())
        .map_err(SupportFitError::Executor)?
        .iter()
        .copied()
        .fold(0.0, f64::max);
    let mut levels = vec![eps];
    while levels.last().is_some_and(|&l| l <= widest) {
        let next = 2.0 * levels.last().copied().unwrap_or(eps);
        levels.push(next);
    }
    levels.reverse();
    let mut keep = if levels.len() > 1 { empty } else { Array2::from_elem((positions, pieces), true) };
    let mut alternations = Vec::new();
    let manifold = EuclideanManifold::new(theta.len());
    let mut supports = None;
    // Each support search resumes the proposal sizes the previous one learned.
    let mut radius: Option<Vec<usize>> = None;
    for &level in &levels {
        let fidelity = form(level);
        let mut current = minimal_support(&mut AtTheta { executor: &mut *executor, theta: theta.view() }, positions, pieces, fidelity, sequence, Some(keep.clone()), radius.take())
            .map_err(SupportFitError::Support)?;
        // One resumed trust-region iteration per alternation: the supports the barrier
        // holds fixed are re-decided between iterations, and the resume keeps the
        // learned radius and the certificate's scale. The level ends only when the
        // supports stop shrinking and the certificate holds, so every level (and the
        // fit) comes from a converged optimization.
        let step = RiemannianTrustRegion { max_iter: 1, ..RiemannianTrustRegion::default() };
        let mut state: Option<TrustRegionTermination> = None;
        loop {
            let kept_before = current.keep.iter().filter(|&&k| k).count();
            let at = current.keep.clone();
            let (termination, before, after) = {
                let mut barrier = Barrier { executor: &mut *executor, keep: &at, fidelity, error: None };
                let before = barrier.value_gradient(theta.view()).map_err(SupportFitError::Geometry)?.0;
                let termination = match &state {
                    None => step.minimize_reporting_termination(&manifold, &mut barrier, theta.view()),
                    Some(previous) => step.resume(
                        &manifold,
                        &mut barrier,
                        &TrustRegionTermination { point: theta.clone(), ..previous.clone() },
                    ),
                };
                if let Some(e) = barrier.error.take() {
                    return Err(SupportFitError::Executor(e));
                }
                let termination = termination.map_err(SupportFitError::Geometry)?;
                let after = barrier.value_gradient(termination.point.view()).map_err(SupportFitError::Geometry)?.0;
                (termination, before, after)
            };
            theta = termination.point.clone();
            let certified = termination.residual <= termination.tolerance;
            let carried = Some(current.radius.clone());
            current = minimal_support(&mut AtTheta { executor: &mut *executor, theta: theta.view() }, positions, pieces, fidelity, sequence, Some(at), carried)
                .map_err(SupportFitError::Support)?;
            let kept = current.keep.iter().filter(|&&k| k).count();
            alternations.push(Alternation {
                level,
                kept,
                barrier_before: before,
                barrier_after: after,
                iterations: termination.iterations,
                certified,
            });
            if kept >= kept_before && certified {
                break;
            }
            state = Some(termination);
        }
        keep = current.keep.clone();
        radius = Some(current.radius.clone());
        supports = Some(current);
    }
    let supports = supports.ok_or(SupportFitError::Executor("support_fit: no fidelity level".to_string()))?;
    Ok(SupportFit { theta, supports, alternations })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `KL_t = ½ Σ_{c removed} (a_tc e^{θ_c})²`: shrinking a piece's scale `e^{θ_c}` lowers
    /// the cost of removing it everywhere (a stand-in for pieces that can be reshaped).
    /// The piece step opens slack, the next support step removes more, and every
    /// position stays admissible throughout.
    struct Scaled {
        a: Array2<f64>,
    }

    impl Scaled {
        fn removed_energy(&self, theta: ArrayView1<'_, f64>, keep: ArrayView2<'_, bool>) -> Array2<f64> {
            Array2::from_shape_fn(self.a.dim(), |(t, c)| {
                if keep[[t, c]] { 0.0 } else { (self.a[[t, c]] * theta[c].exp()).powi(2) }
            })
        }
    }

    impl PieceExecutor for Scaled {
        fn supports(&mut self, theta: ArrayView1<'_, f64>, keep: ArrayView2<'_, bool>) -> Result<SupportEvaluation, String> {
            let divergence = self.divergence(theta, keep)?;
            let removal_cost = Array2::from_shape_fn(self.a.dim(), |(t, c)| 0.5 * (self.a[[t, c]] * theta[c].exp()).powi(2));
            Ok(SupportEvaluation { divergence, restore_gain: removal_cost.clone(), removal_cost })
        }
        fn divergence(&mut self, theta: ArrayView1<'_, f64>, keep: ArrayView2<'_, bool>) -> Result<Array1<f64>, String> {
            Ok(self.removed_energy(theta, keep).sum_axis(ndarray::Axis(1)) * 0.5)
        }
        fn weighted_gradient(&mut self, theta: ArrayView1<'_, f64>, keep: ArrayView2<'_, bool>, weights: ArrayView1<'_, f64>) -> Result<Array1<f64>, String> {
            // d KL_t / d θ_c = removed energy of (t, c).
            Ok(self.removed_energy(theta, keep).t().dot(&weights))
        }
        fn directional(&mut self, theta: ArrayView1<'_, f64>, keep: ArrayView2<'_, bool>, v: ArrayView1<'_, f64>) -> Result<Array1<f64>, String> {
            Ok(self.removed_energy(theta, keep).dot(&v))
        }
        fn weighted_hessian(&mut self, theta: ArrayView1<'_, f64>, keep: ArrayView2<'_, bool>, weights: ArrayView1<'_, f64>, v: ArrayView1<'_, f64>) -> Result<Array1<f64>, String> {
            // KL_t = ½ Σ a_tc² e^{2θ_c} over removed pieces: the exact Hessian is diagonal,
            // ∂²KL_t/∂θ_c² = 2 a_tc² e^{2θ_c}, twice the removed energy.
            let e = self.removed_energy(theta, keep);
            Ok(e.t().dot(&weights) * &v * 2.0)
        }
    }

    #[test]
    fn reshaping_pieces_lets_the_supports_shrink() {
        let a = ndarray::array![[1.0, 0.3, 0.2, 0.9], [0.25, 1.0, 0.3, 0.2], [0.3, 0.2, 1.0, 0.25]];
        let mut executor = Scaled { a };
        let fit = fit_supports_and_pieces(&mut executor, Array1::zeros(4), 3, 4, 0.1, false, 1).unwrap();
        let first = fit.alternations.first().unwrap();
        assert!(first.barrier_after <= first.barrier_before, "{:?}", fit.alternations);
        // The path starts sparse (above the empty support's divergence) and ends at ε.
        assert!(first.level > 0.1 && fit.alternations.last().unwrap().level == 0.1, "{:?}", fit.alternations);
        for w in fit.alternations.windows(2) {
            if w[1].level == w[0].level {
                assert!(w[1].kept <= w[0].kept, "{:?}", fit.alternations);
            }
        }
        assert!(fit.supports.divergence.iter().all(|&v| v <= 0.1));
        let kl = executor.divergence(fit.theta.view(), fit.supports.keep.view()).unwrap();
        assert!(kl.iter().all(|&v| v <= 0.1), "{kl:?}");
    }

    /// Under the mean form the fit ends with the mean divergence below ε (one shared
    /// budget), and removes at least as much as the per-position form.
    #[test]
    fn the_mean_form_shares_one_budget() {
        let a = ndarray::array![[1.0, 0.3, 0.2, 0.9], [0.25, 1.0, 0.3, 0.2], [0.3, 0.2, 1.0, 0.25]];
        let per = fit_supports_and_pieces(&mut Scaled { a: a.clone() }, Array1::zeros(4), 3, 4, 0.1, false, 1).unwrap();
        let mut executor = Scaled { a };
        let mean = fit_supports_and_pieces(&mut executor, Array1::zeros(4), 3, 4, 0.1, true, 1).unwrap();
        let kl = executor.divergence(mean.theta.view(), mean.supports.keep.view()).unwrap();
        assert!(kl.sum() / 3.0 < 0.1, "{kl:?}");
        let removed = |k: &Array2<bool>| k.iter().filter(|x| !**x).count();
        assert!(removed(&mean.supports.keep) >= removed(&per.supports.keep));
    }
}
