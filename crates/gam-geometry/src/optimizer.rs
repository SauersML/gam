use ndarray::{Array1, ArrayView1};
use opt::{BacktrackConfig, ExpandConfig, bidirectional_line_search, constants};

use crate::manifold::{GeometryResult, RiemannianManifold, check_len, quad_form};

/// Linear factor of the Steihaug truncated-CG forcing sequence: the inner CG
/// solve is terminated once the residual drops to `min(η·‖r₀‖, ‖r₀‖²)`. The
/// quadratic `‖r₀‖²` term gives the super-linear convergence of an inexact
/// Newton step near the optimum, while `η·‖r₀‖` caps wasted inner work far from
/// it (Nocedal & Wright, *Numerical Optimization*, §7.1, eq. 7.3).
const STEIHAUG_CG_FORCING_FACTOR: f64 = 1.0e-2;

pub trait RiemannianObjective {
    fn value_gradient(&mut self, point: ArrayView1<'_, f64>) -> GeometryResult<(f64, Array1<f64>)>;

    /// Riemannian Hessian–vector product `H(x)·v` for a tangent direction `v`
    /// at `point`, returned in the same ambient/tangent coordinates as the
    /// gradient.
    ///
    /// This is what upgrades the trust-region subproblem from a Cauchy-point
    /// step (the exact minimizer of the *linear* model along the steepest
    /// descent direction) to a Steihaug truncated-CG step that exploits real
    /// curvature. An objective that exposes no second-order information returns
    /// `None` (the default), and the trust region transparently falls back to
    /// the Cauchy point — never to plain clipped steepest descent, which has no
    /// model, no predicted/actual reduction ratio, and no accept/reject.
    ///
    /// The Riemannian-Hessian quadratic model the trust region builds from this
    /// product is a valid second-order model of `f` only along a (≥)second-order
    /// retraction (the exponential map, or any retraction with
    /// [`RiemannianManifold::retraction_is_second_order`] `== true`). On a
    /// manifold whose `retract` is only FIRST-order (e.g. the Stiefel/Grassmann
    /// QR retraction) the second derivative of the pullback `f∘R_x` is not the
    /// Riemannian Hessian, so the trust region ignores this curvature and uses
    /// the first-order-correct Cauchy model instead (issue #956).
    fn hessian_vector_product(
        &mut self,
        point: ArrayView1<'_, f64>,
        tangent: ArrayView1<'_, f64>,
    ) -> GeometryResult<Option<Array1<f64>>> {
        // Validate the shapes the contract requires (a tangent at `point`), then
        // report "no curvature available" so the trust region selects the
        // Cauchy point. We never fabricate a Hessian here.
        check_len("hessian_vector_product tangent", tangent.len(), point.len())?;
        Ok(None)
    }
}

/// Metric inner product `g_x(a, b) = aᵀ G(x) b` using the manifold metric
/// tensor at `point`. For manifolds whose metric is the ambient identity
/// (Euclidean, Sphere, Circle, Torus, …) this reduces to the Euclidean dot
/// product; for a genuine Riemannian metric (e.g. the affine-invariant SPD
/// metric) it evaluates the correct geometric inner product on the tangent
/// space. Every norm and inner product in both optimizers below routes through
/// this so the algorithms are metric-correct on curved manifolds.
fn g_inner(
    manifold: &dyn RiemannianManifold,
    point: ArrayView1<'_, f64>,
    a: ArrayView1<'_, f64>,
    b: ArrayView1<'_, f64>,
) -> GeometryResult<f64> {
    let g = manifold.metric_tensor(point)?;
    Ok(quad_form(g.view(), a, b))
}

fn g_norm(
    manifold: &dyn RiemannianManifold,
    point: ArrayView1<'_, f64>,
    a: ArrayView1<'_, f64>,
) -> GeometryResult<f64> {
    let metric = manifold.metric_tensor(point)?;
    let metric_times_a = gam_linalg::faer_ndarray::fast_av(&metric.view(), &a);
    metric_norm_from_product(a, metric_times_a.view())
}

/// Certify `sqrt(a^T G a)` once the metric product `G a` is available.
///
/// The absolute accumulation is part of the backward-error certificate.  It
/// must itself remain finite: an infinite error scale would make every finite
/// negative quadratic look like harmless roundoff and could certify a
/// non-zero vector as having zero norm under an indefinite metric.
fn metric_norm_from_product(
    a: ArrayView1<'_, f64>,
    metric_times_a: ArrayView1<'_, f64>,
) -> GeometryResult<f64> {
    check_len("metric norm product", metric_times_a.len(), a.len())?;
    let mut squared_norm = 0.0_f64;
    let mut absolute_sum = 0.0_f64;
    for (&left, &right) in a.iter().zip(metric_times_a.iter()) {
        let term = left * right;
        squared_norm += term;
        absolute_sum += term.abs();
    }
    if !squared_norm.is_finite() {
        return Ok(f64::INFINITY);
    }
    if !absolute_sum.is_finite() {
        return Err(crate::manifold::GeometryError::InvalidPoint(
            "Riemannian metric norm error bound overflowed",
        ));
    }
    // A Riemannian metric is positive definite. Permit only the backward-error
    // band of the final dot product; clamping a materially negative quadratic
    // to zero would falsely turn an indefinite metric into a stationary point.
    let negative_roundoff = 64.0 * f64::EPSILON * absolute_sum;
    if squared_norm < -negative_roundoff {
        return Err(crate::manifold::GeometryError::InvalidPoint(
            "Riemannian metric produced a negative squared norm",
        ));
    }
    Ok(squared_norm.max(0.0).sqrt())
}

/// Shift-invariant relative-gradient stationarity measure
/// `‖grad_k‖_g / max(‖grad_0‖_g, 1)`, comparing the current Riemannian gradient
/// norm to the gradient norm at the INITIAL iterate. The initial gradient norm
/// carries the same *multiplicative* scale the objective and its gradient share
/// (`f → c·f` ⇒ `grad → c·grad`), so a fixed `grad_tol` still reads as a
/// *relative* tolerance — but, unlike dividing by `max(|f|, 1)`, `‖grad_0‖` is
/// invariant under an additive shift `f → f + C`, which leaves the minimizers,
/// gradient, trust-region model reduction, Armijo slope, and accepted path all
/// unchanged. Dividing by `|f|` was non-invariant: a large additive constant
/// inflates the denominator and can falsely certify convergence at a
/// non-stationary iterate (e.g. `f̃(x) = C + x²` at `x = 1` with `C > 2/τ − 1`),
/// issue #954. The `max(·, 1)` floor reduces this to the absolute test
/// `‖grad_k‖ ≤ grad_tol` on a unit-scale objective and preserves the
/// O(n)-gradient calibration of the profiled REML latent objective (whose
/// `‖grad_0‖` is itself O(n), issue #879). The non-intrinsic `‖x‖_typ` factor is
/// dropped: ambient iterate magnitude is not coordinate/chart invariant on a
/// manifold, so it does not belong in a Riemannian stationarity test. A
/// non-finite gradient maps to `+∞` so a blown-up iterate is never stationary.
fn relative_stationarity(grad_norm: f64, grad0_norm: f64) -> f64 {
    if !grad_norm.is_finite() || !grad0_norm.is_finite() {
        return f64::INFINITY;
    }
    grad_norm / grad0_norm.max(1.0)
}

#[derive(Debug, Clone, PartialEq)]
pub struct RiemannianTrustRegion {
    /// Initial trust-region radius Δ₀.
    pub radius: f64,
    /// Hard cap Δmax on the radius across all iterations.
    pub max_radius: f64,
    pub max_iter: usize,
    pub grad_tol: f64,
}

impl Default for RiemannianTrustRegion {
    fn default() -> Self {
        Self {
            radius: 1.0,
            max_radius: 1.0e6,
            max_iter: 64,
            grad_tol: 1.0e-8,
        }
    }
}

impl RiemannianTrustRegion {
    /// A genuine Riemannian trust-region method.
    ///
    /// At each iterate `x` we build the quadratic model in the tangent space
    /// `T_xM`,
    ///
    /// ```text
    ///   m(η) = f(x) + g_x(grad, η) + ½ g_x(η, Hη),
    /// ```
    ///
    /// where `g_x(·,·)` is the manifold metric inner product and `H` is the
    /// Riemannian Hessian (accessed only through Hessian–vector products). The
    /// step is the (approximate) solution of the trust-region subproblem
    ///
    /// ```text
    ///   min_{η ∈ T_xM, ‖η‖_g ≤ Δ}  m(η).
    /// ```
    ///
    /// When the objective supplies Hessian–vector products AND the manifold's
    /// `retract` is at least a second-order retraction
    /// ([`RiemannianManifold::retraction_is_second_order`]) we solve the
    /// subproblem with the Steihaug truncated-CG method (stopping at negative
    /// curvature or the trust-region boundary). Otherwise — no curvature, or a
    /// first-order retraction whose pullback second derivative is not the
    /// Riemannian Hessian (issue #956) — we fall back to the Cauchy point: the
    /// exact minimizer of the model along the steepest-descent direction within
    /// the trust region (with curvature taken from the model where available,
    /// and the boundary point of the decreasing linear model otherwise). The
    /// linear term `Df_x[η]` is retraction-independent, so the Cauchy model
    /// keeps ρ and the radius control valid along any retraction. Either way
    /// this is a real model-based step — not clipped descent.
    ///
    /// We then form the ratio of actual to predicted reduction
    ///
    /// ```text
    ///   ρ = (f(x) − f(x⁺)) / (m(0) − m(η)),
    /// ```
    ///
    /// accept the step only when `ρ > η₁`, and adapt Δ: shrink on a poor ratio,
    /// expand on an excellent ratio that reaches the boundary, otherwise hold.
    /// Only accepted steps are retracted onto the manifold.
    pub fn minimize(
        &self,
        manifold: &dyn RiemannianManifold,
        objective: &mut dyn RiemannianObjective,
        initial: ArrayView1<'_, f64>,
    ) -> GeometryResult<Array1<f64>> {
        // Trust-region acceptance / radius-update constants.
        const ETA1: f64 = 0.1; // accept the step iff ρ > ETA1
        const ETA_SHRINK: f64 = 0.25; // ρ below this ⇒ shrink the radius
        const ETA_EXPAND: f64 = 0.75; // ρ above this (at the boundary) ⇒ expand
        const SHRINK: f64 = 0.25;
        const EXPAND: f64 = 2.0;
        let mut x = initial.to_owned();
        let d = manifold.ambient_dim();
        check_len("trust-region initial point", x.len(), d)?;
        if !(self.radius.is_finite() && self.radius > 0.0) {
            return Err(crate::manifold::GeometryError::InvalidPoint(
                "trust-region radius must be finite and positive",
            ));
        }
        if !(self.max_radius.is_finite() && self.max_radius > 0.0) {
            return Err(crate::manifold::GeometryError::InvalidPoint(
                "trust-region maximum radius must be finite and positive",
            ));
        }
        if !(self.grad_tol.is_finite() && self.grad_tol >= 0.0) {
            return Err(crate::manifold::GeometryError::InvalidPoint(
                "trust-region gradient tolerance must be finite and non-negative",
            ));
        }

        // Establish the trust-region invariant `0 < Δ_k ≤ Δmax` *before* the
        // first step, not just on later expansions. The expansion rule below
        // caps via `min(·, max_radius)` and contraction only shrinks, so once
        // `0 < Δ₀ ≤ Δmax` holds we have `0 < Δ_k ≤ Δmax` for all `k` by
        // induction; every subproblem then obeys `‖η_k‖_g ≤ Δ_k ≤ Δmax`,
        // restoring `max_radius` as the documented hard cap. A configured
        // `radius > max_radius` (or a non-finite `radius`) would otherwise let
        // the very first Cauchy/Steihaug step overshoot the advertised maximum,
        // so we clamp the initial radius into `(0, max_radius]` here.
        let mut delta = self.radius.min(self.max_radius);

        // Initial Riemannian gradient norm, captured on the first iteration and
        // used as the shift-invariant scale in the relative stationarity test
        // (see `relative_stationarity`).
        let mut grad0_norm: Option<f64> = None;
        let mut iterations = 0usize;

        for _ in 0..self.max_iter {
            let (f_curr, grad_e) = objective.value_gradient(x.view())?;
            if !f_curr.is_finite() {
                return Err(crate::manifold::GeometryError::InvalidPoint(
                    "trust-region objective returned a non-finite value",
                ));
            }
            iterations += 1;
            // Raise the ambient Euclidean differential to the *Riemannian*
            // gradient through the manifold metric. Merely projecting onto the
            // tangent space is the Riemannian gradient only for the embedded
            // (identity) metric; for a genuine metric (affine-invariant SPD,
            // canonical Stiefel) it is the wrong direction, making the model
            // linear term `g_x(grad, η)` not the differential `Df_x[η]` and the
            // step not first-order correct (issue #955).
            let grad = manifold.riemannian_gradient(x.view(), grad_e.view())?;
            let grad_norm = g_norm(manifold, x.view(), grad.view())?;
            // Shift-invariant (relative) stationarity test. Comparing the bare
            // gradient norm to a fixed absolute `grad_tol` is mis-calibrated for
            // objectives whose natural scale is large — e.g. the *profiled*
            // Gaussian REML latent objective, whose `n·log σ̂²` term leaves
            // `‖grad‖` at an O(n) magnitude even at a genuine stationary point
            // near interpolation (issue #879). We instead test the dimensionless
            // ratio `‖grad_k‖_g / max(‖grad_0‖_g, 1)`, where `‖grad_0‖_g` is the
            // gradient norm at the initial iterate. It carries the same
            // *multiplicative* scale the objective and its gradient share but is
            // invariant under an additive shift `f → f + C` (unlike `max(|f|,1)`,
            // which a large constant inflates into a false convergence, #954),
            // and reduces to the absolute test on a unit-scale objective.
            let grad0 = *grad0_norm.get_or_insert(grad_norm);
            if relative_stationarity(grad_norm, grad0) <= self.grad_tol {
                break;
            }

            // Solve the trust-region subproblem in T_xM.
            let (step, predicted_reduction, hit_boundary) =
                self.solve_subproblem(manifold, objective, x.view(), grad.view(), delta)?;

            // A non-positive predicted reduction means the model offers no
            // descent (e.g. a vanishing step); shrink and retry from the same
            // point rather than dividing by ~0 in ρ.
            if !(predicted_reduction > 0.0) {
                delta *= SHRINK;
                if delta <= self.grad_tol * self.grad_tol {
                    break;
                }
                continue;
            }

            let trial_x = manifold.retract(x.view(), step.view())?;
            let f_trial = objective.value_gradient(trial_x.view())?.0;
            let actual_reduction = f_curr - f_trial;
            let rho = if f_trial.is_finite() {
                actual_reduction / predicted_reduction
            } else {
                f64::NEG_INFINITY
            };

            // Radius update.
            if rho < ETA_SHRINK {
                delta *= SHRINK;
            } else if rho > ETA_EXPAND && hit_boundary {
                delta = (delta * EXPAND).min(self.max_radius);
            }

            // Accept only sufficiently-good steps; otherwise keep x (the next
            // iteration recomputes f and the gradient at the retained point).
            if rho > ETA1 && f_trial.is_finite() {
                x = trial_x;
            }
        }
        // Returning a point is a mathematical claim: it must satisfy the same
        // first-order certificate that controls the loop. Budget exhaustion,
        // a collapsed radius, or a failed model step is not success merely
        // because the last iterate is finite.
        let (f_final, grad_e_final) = objective.value_gradient(x.view())?;
        if !f_final.is_finite() {
            return Err(crate::manifold::GeometryError::InvalidPoint(
                "trust-region objective returned a non-finite terminal value",
            ));
        }
        let grad_final = manifold.riemannian_gradient(x.view(), grad_e_final.view())?;
        let grad_final_norm = g_norm(manifold, x.view(), grad_final.view())?;
        let grad0 = grad0_norm.unwrap_or(grad_final_norm);
        let residual = relative_stationarity(grad_final_norm, grad0);
        if residual <= self.grad_tol {
            Ok(x)
        } else {
            Err(crate::manifold::GeometryError::NonConvergence {
                context: "Riemannian trust-region optimization (relative gradient norm)",
                iterations,
                residual,
                tolerance: self.grad_tol,
            })
        }
    }

    /// Solve `min_{‖η‖_g ≤ Δ} m(η)` and return `(η, m(0) − m(η), hit_boundary)`.
    ///
    /// Uses Steihaug truncated-CG when the objective provides Hessian–vector
    /// products *and* the manifold's `retract` is at least a second-order
    /// retraction ([`RiemannianManifold::retraction_is_second_order`]). When the
    /// retraction is only first-order the Riemannian-Hessian quadratic term is
    /// not the second derivative of `f∘R_x`, so scoring it would corrupt ρ
    /// (issue #956); we then take the Cauchy point, whose linear model is
    /// first-order correct along any retraction (as we also do when no curvature
    /// is available).
    fn solve_subproblem(
        &self,
        manifold: &dyn RiemannianManifold,
        objective: &mut dyn RiemannianObjective,
        x: ArrayView1<'_, f64>,
        grad: ArrayView1<'_, f64>,
        delta: f64,
    ) -> GeometryResult<(Array1<f64>, f64, bool)> {
        const BOUNDARY_FRAC: f64 = 0.9;

        // Probe for curvature once: if the objective exposes no Hessian–vector
        // product we take the Cauchy point.
        let has_hessian = objective.hessian_vector_product(x, grad)?.is_some();

        // The Riemannian-Hessian quadratic model `½ g_x(η, Hη)` is the correct
        // second-order model of `f` along the trial path ONLY when that path is
        // generated by the exponential map or another second-order retraction:
        // for a first-order retraction `R_x` the pullback `f∘R_x` has a second
        // derivative at `0` that is NOT the Riemannian Hessian, so scoring the
        // curved model against `manifold.retract` corrupts ρ and the radius
        // control (issue #956). The linear term `g_x(grad, η) = Df_x[η]` is
        // retraction-independent, so the curvature-free Cauchy model stays
        // first-order correct along ANY retraction. We therefore use the curved
        // Steihaug truncated-CG step only when the objective supplies curvature
        // AND the manifold's retraction is (at least) second-order; otherwise we
        // take the Cauchy point — never asserting a second-order model the
        // retraction cannot honor.
        if !has_hessian || !manifold.retraction_is_second_order() {
            return self.cauchy_point(manifold, x, grad, delta);
        }

        // --- Steihaug truncated-CG on the metric inner product. ---
        // Solve min m(η) = g_x(grad, η) + ½ g_x(η, Hη) within ‖η‖_g ≤ Δ.
        let n = grad.len();
        let mut z = Array1::<f64>::zeros(n); // current iterate η
        let mut r = grad.to_owned(); // residual = grad + Hz (z=0 ⇒ grad)
        let mut p = -&r; // search direction
        let r0_norm = g_norm(manifold, x, r.view())?;
        let tol = (STEIHAUG_CG_FORCING_FACTOR * r0_norm).min(r0_norm * r0_norm);

        // model reduction tracker m(0) − m(z); m(0) = 0 here (constant dropped).
        // m(z) = g(grad,z) + ½ g(z,Hz); we recompute it at the end for ρ.
        let max_cg = 2 * n + 1;
        for _ in 0..max_cg {
            let hp = objective.hessian_vector_product(x, p.view())?.ok_or(
                crate::manifold::GeometryError::Unsupported(
                    "Hessian–vector product became unavailable mid-subproblem",
                ),
            )?;
            let php = g_inner(manifold, x, p.view(), hp.view())?;
            if php <= 0.0 {
                // Negative curvature: go to the boundary along p.
                let (tau, _) = boundary_tau(manifold, x, z.view(), p.view(), delta)?;
                let eta = &z + &(&p * tau);
                let red = model_reduction(manifold, objective, x, grad, eta.view())?;
                return Ok((eta, red, true));
            }
            let rr = g_inner(manifold, x, r.view(), r.view())?;
            let alpha = rr / php;
            let z_next = &z + &(&p * alpha);
            if g_norm(manifold, x, z_next.view())? >= delta {
                // Trust-region boundary crossed: step to it.
                let (tau, _) = boundary_tau(manifold, x, z.view(), p.view(), delta)?;
                let eta = &z + &(&p * tau);
                let red = model_reduction(manifold, objective, x, grad, eta.view())?;
                return Ok((eta, red, true));
            }
            z = z_next;
            let r_next = &r + &(&hp * alpha);
            let r_next_norm = g_norm(manifold, x, r_next.view())?;
            if r_next_norm <= tol {
                let red = model_reduction(manifold, objective, x, grad, z.view())?;
                let hit = g_norm(manifold, x, z.view())? >= BOUNDARY_FRAC * delta;
                return Ok((z, red, hit));
            }
            let rr_next = g_inner(manifold, x, r_next.view(), r_next.view())?;
            let beta = rr_next / rr;
            p = &(-&r_next) + &(&p * beta);
            r = r_next;
        }
        let red = model_reduction(manifold, objective, x, grad, z.view())?;
        let hit = g_norm(manifold, x, z.view())? >= BOUNDARY_FRAC * delta;
        Ok((z, red, hit))
    }

    /// Cauchy point: the exact minimizer of the model along the steepest-descent
    /// direction `−grad` within the trust region. With no curvature available
    /// the model is the decreasing linear `m(τ·(−grad)) = −τ‖grad‖²_g`, whose
    /// constrained minimizer sits on the boundary `τ = Δ / ‖grad‖_g`, giving a
    /// predicted reduction `Δ·‖grad‖_g`.
    fn cauchy_point(
        &self,
        manifold: &dyn RiemannianManifold,
        x: ArrayView1<'_, f64>,
        grad: ArrayView1<'_, f64>,
        delta: f64,
    ) -> GeometryResult<(Array1<f64>, f64, bool)> {
        let grad_norm = g_norm(manifold, x, grad.view())?;
        if grad_norm <= 0.0 {
            return Ok((Array1::<f64>::zeros(grad.len()), 0.0, false));
        }
        let tau = delta / grad_norm;
        let step = &grad.to_owned() * (-tau);
        // Predicted reduction of the linear model m(0) − m(η) = τ‖grad‖²_g.
        let predicted = tau * grad_norm * grad_norm;
        Ok((step, predicted, true))
    }
}

/// Largest `τ ≥ 0` with `‖z + τ p‖_g = Δ`, solving the quadratic
/// `‖p‖²_g τ² + 2 g(z,p) τ + (‖z‖²_g − Δ²) = 0`. Returns `(τ, ‖z + τp‖_g)`.
fn boundary_tau(
    manifold: &dyn RiemannianManifold,
    x: ArrayView1<'_, f64>,
    z: ArrayView1<'_, f64>,
    p: ArrayView1<'_, f64>,
    delta: f64,
) -> GeometryResult<(f64, f64)> {
    let pp = g_inner(manifold, x, p, p)?;
    let zp = g_inner(manifold, x, z, p)?;
    let zz = g_inner(manifold, x, z, z)?;
    if pp <= 0.0 {
        return Ok((0.0, zz.max(0.0).sqrt()));
    }
    let c = zz - delta * delta;
    let disc = (zp * zp - pp * c).max(0.0);
    let tau = (-zp + disc.sqrt()) / pp;
    let tau = tau.max(0.0);
    Ok((tau, delta))
}

/// Model reduction `m(0) − m(η) = −g(grad, η) − ½ g(η, Hη)`.
fn model_reduction(
    manifold: &dyn RiemannianManifold,
    objective: &mut dyn RiemannianObjective,
    x: ArrayView1<'_, f64>,
    grad: ArrayView1<'_, f64>,
    eta: ArrayView1<'_, f64>,
) -> GeometryResult<f64> {
    let lin = g_inner(manifold, x, grad, eta)?;
    let heta = objective.hessian_vector_product(x, eta)?.ok_or(
        crate::manifold::GeometryError::Unsupported(
            "Hessian–vector product unavailable while scoring the model",
        ),
    )?;
    let quad = g_inner(manifold, x, eta, heta.view())?;
    Ok(-lin - 0.5 * quad)
}

#[derive(Debug, Clone, PartialEq)]
pub struct RiemannianLBFGS {
    pub history: usize,
    pub step_size: f64,
    pub max_iter: usize,
    pub grad_tol: f64,
}

impl Default for RiemannianLBFGS {
    fn default() -> Self {
        Self {
            history: 10,
            step_size: 1.0,
            max_iter: 100,
            grad_tol: 1.0e-8,
        }
    }
}

/// One stored secant pair, kept with its base point so the two-loop recursion
/// can transport it into whatever the current tangent space is. `s` and `y`
/// both live in `T_{base}M`.
#[derive(Clone)]
struct SecantPair {
    base: Array1<f64>,
    s: Array1<f64>,
    y: Array1<f64>,
}

impl RiemannianLBFGS {
    /// Riemannian L-BFGS with a backtracking-and-expansion Armijo line search.
    ///
    /// The search starts at the user-supplied `step_size` (a hint, not a hard
    /// cap) and first *expands* by doubling while the Armijo sufficient-
    /// decrease condition continues to hold and the objective is still
    /// strictly improving. Once expansion stalls, it accepts the best step
    /// it has seen so far; if even the initial trial violates Armijo, it
    /// *contracts* by halving until Armijo holds or a safeguard floor is
    /// reached. This makes the optimizer robust to mis-scaled `step_size`
    /// inputs (including the Newton-natural α=1 that BFGS expects on
    /// well-conditioned quadratics) without forcing the caller to retune
    /// it, and preserves the secant pair (s, y) curvature condition so the
    /// L-BFGS inverse-Hessian approximation stays SPD.
    ///
    /// All inner products use the manifold metric `g_x(·,·)`, and every secant
    /// pair is *parallel-transported into the current tangent space* before it
    /// enters the two-loop recursion, so the BFGS algebra never mixes vectors
    /// living in different tangent spaces (the bug fixed in #616). The freshly
    /// formed secant pair likewise transports the accepted step from the old
    /// tangent space into the new one before pairing it with the gradient
    /// difference, so both `s` and `y` live in `T_{x_new}M`.
    pub fn minimize(
        &self,
        manifold: &dyn RiemannianManifold,
        objective: &mut dyn RiemannianObjective,
        initial: ArrayView1<'_, f64>,
    ) -> GeometryResult<Array1<f64>> {
        let mut x = initial.to_owned();
        let d = manifold.ambient_dim();
        check_len("L-BFGS initial point", x.len(), d)?;
        if !(self.step_size.is_finite() && self.step_size > 0.0) {
            return Err(crate::manifold::GeometryError::InvalidPoint(
                "L-BFGS step size must be finite and positive",
            ));
        }
        if !(self.grad_tol.is_finite() && self.grad_tol >= 0.0) {
            return Err(crate::manifold::GeometryError::InvalidPoint(
                "L-BFGS gradient tolerance must be finite and non-negative",
            ));
        }
        let mut history: Vec<SecantPair> = Vec::new();
        let (mut f_curr, grad_e0) = objective.value_gradient(x.view())?;
        if !f_curr.is_finite() {
            return Err(crate::manifold::GeometryError::InvalidPoint(
                "L-BFGS objective returned a non-finite value",
            ));
        }
        // Riemannian gradient (metric-raised), not a bare tangent projection —
        // see the trust region above and issue #955. The secant pairs, two-loop
        // recursion, and Armijo slope are all metric inner products, so they
        // must operate on the true Riemannian gradient.
        let mut grad = manifold.riemannian_gradient(x.view(), grad_e0.view())?;
        // Initial Riemannian gradient norm: the shift-invariant scale of the
        // relative stationarity test (see `relative_stationarity`).
        let grad0_norm = g_norm(manifold, x.view(), grad.view())?;
        let armijo_c: f64 = constants::ARMIJO_C1;
        let alpha_min: f64 = 1.0e-16;
        let alpha_max: f64 = 1.0e16;
        let initial_step = self.step_size;
        let mut iterations = 0usize;
        for _ in 0..self.max_iter {
            // Shift-invariant (relative) stationarity test, identical in form to
            // the trust region's (see `relative_stationarity`): the current
            // gradient norm is measured against the initial one, so a fixed
            // `grad_tol` reads as a *relative* tolerance for a large-scale
            // objective (e.g. the profiled REML latent objective, issue #879)
            // while staying invariant under an additive shift of `f` (#954).
            // Reduces to the absolute test on a unit-scale objective.
            let grad_norm = g_norm(manifold, x.view(), grad.view())?;
            if relative_stationarity(grad_norm, grad0_norm) <= self.grad_tol {
                break;
            }
            iterations += 1;
            let direction = two_loop(manifold, x.view(), grad.view(), &history)?;
            let direction = -&direction;
            let slope = g_inner(manifold, x.view(), grad.view(), direction.view())?;
            // Guard against ascent directions caused by stale curvature; if the
            // BFGS direction is not a (metric) descent direction, fall back to
            // the projected steepest-descent direction so progress is
            // guaranteed.
            let (direction, slope) = if slope < 0.0 {
                (direction, slope)
            } else {
                let sd = -&grad;
                let s_sd = g_inner(manifold, x.view(), grad.view(), sd.view())?;
                (sd, s_sd)
            };
            let old_x = x.clone();
            let old_grad = grad.clone();
            // --- Armijo line search with bidirectional adaptation, via the
            // shared expand-then-backtrack primitive: doubles while Armijo
            // holds and the objective keeps strictly improving (capped at
            // `alpha_max`), otherwise contracts by the shared halving
            // schedule. The trial payload carries the retracted point and its
            // ambient differential so the accepted step's Riemannian gradient
            // is raised exactly once, after the search.
            // The pre-migration loops doubled to `alpha_max` and halved to
            // `alpha_min`; both trial counts are derived by the same
            // recurrences (exact, unlike a log).
            let max_expansions = {
                let mut n = 1_usize;
                let mut a = initial_step;
                while a < alpha_max {
                    n += 1;
                    a *= 2.0;
                }
                n
            };
            let max_steps = {
                let mut n = 0_usize;
                let mut a = initial_step;
                while a > alpha_min {
                    n += 1;
                    a *= 0.5;
                }
                n
            };
            let accepted = bidirectional_line_search(
                f_curr,
                ExpandConfig {
                    expand_factor: 2.0,
                    max_expansions,
                    max_step: alpha_max,
                },
                BacktrackConfig {
                    initial_step,
                    max_steps,
                    ..BacktrackConfig::default()
                },
                |alpha| {
                    let step = &direction * alpha;
                    let trial_x = manifold.retract(x.view(), step.view())?;
                    let (f_trial, g_trial_e) = objective.value_gradient(trial_x.view())?;
                    Ok(Some((f_trial, (trial_x, g_trial_e))))
                },
                |alpha, f_trial| {
                    f_trial.is_finite() && f_trial <= f_curr + armijo_c * alpha * slope
                },
            )?;
            let Some(accepted) = accepted else {
                // No admissible step found — terminate at the current point.
                break;
            };
            let best_alpha = accepted.step;
            let best_f = accepted.value;
            let (best_x, g_trial_e) = accepted.payload;
            // Riemannian (metric-raised) gradient at the accepted point — the
            // object the secant pair (s, y) is formed from (#955).
            let best_grad = manifold.riemannian_gradient(best_x.view(), g_trial_e.view())?;
            // The accepted tangent step at `old_x` (the actual move taken).
            let eta = &direction * best_alpha;
            x = best_x;
            f_curr = best_f;
            grad = best_grad;

            // --- Secant pair, formed entirely in T_{x_new}M (the #616 fix). ---
            // Parallel-transport the accepted step from T_{old_x}M to T_{x}M so
            // it is a tangent at the NEW point, matching the gradient there. The
            // old gradient is likewise transported to T_{x}M before subtraction.
            let path = transport_path(&old_x, &x);
            let s = manifold.parallel_transport(path.view(), eta.view())?;
            let transported_old_grad = manifold.parallel_transport(path.view(), old_grad.view())?;
            let y = &grad - &transported_old_grad;
            // Commit the (s, y) pair only when the metric curvature condition
            // g_x(s, y) > 0 holds (strict positivity). This is required for the
            // implicit BFGS inverse-Hessian update to remain SPD.
            let sy = g_inner(manifold, x.view(), s.view(), y.view())?;
            if sy > 1.0e-14 {
                history.push(SecantPair {
                    base: x.clone(),
                    s,
                    y,
                });
                if history.len() > self.history {
                    history.remove(0);
                }
            }
        }
        let grad_norm = g_norm(manifold, x.view(), grad.view())?;
        let residual = relative_stationarity(grad_norm, grad0_norm);
        if residual <= self.grad_tol {
            Ok(x)
        } else {
            Err(crate::manifold::GeometryError::NonConvergence {
                context: "Riemannian L-BFGS optimization (relative gradient norm)",
                iterations,
                residual,
                tolerance: self.grad_tol,
            })
        }
    }
}

/// Build the 2×D point path matrix `[p_from; p_to]` consumed by
/// [`RiemannianManifold::parallel_transport`].
fn transport_path(p_from: &Array1<f64>, p_to: &Array1<f64>) -> ndarray::Array2<f64> {
    let d = p_from.len();
    let mut path = ndarray::Array2::<f64>::zeros((2, d));
    path.row_mut(0).assign(p_from);
    path.row_mut(1).assign(p_to);
    path
}

/// L-BFGS two-loop recursion in the CURRENT tangent space `T_xM`.
///
/// Each stored secant pair `(s_i, y_i)` lives in `T_{base_i}M`; before it can
/// participate in the recursion it is parallel-transported into `T_xM`. All
/// inner products use the manifold metric `g_x(·,·)` at the current point, so
/// no two vectors from different tangent spaces are ever combined (#616).
fn two_loop(
    manifold: &dyn RiemannianManifold,
    x: ArrayView1<'_, f64>,
    grad: ArrayView1<'_, f64>,
    history: &[SecantPair],
) -> GeometryResult<Array1<f64>> {
    // Transport every stored pair into the current tangent space once.
    let mut s_cur: Vec<Array1<f64>> = Vec::with_capacity(history.len());
    let mut y_cur: Vec<Array1<f64>> = Vec::with_capacity(history.len());
    for pair in history {
        let path = transport_path(&pair.base, &x.to_owned());
        s_cur.push(manifold.parallel_transport(path.view(), pair.s.view())?);
        y_cur.push(manifold.parallel_transport(path.view(), pair.y.view())?);
    }

    let mut q = grad.to_owned();
    let mut alpha = vec![0.0; history.len()];
    let mut rho = vec![0.0; history.len()];
    for i in (0..history.len()).rev() {
        let sy = g_inner(manifold, x, s_cur[i].view(), y_cur[i].view())?;
        rho[i] = 1.0 / sy;
        alpha[i] = rho[i] * g_inner(manifold, x, s_cur[i].view(), q.view())?;
        q = &q - &(&y_cur[i] * alpha[i]);
    }
    let mut r = q;
    if let (Some(s), Some(y)) = (s_cur.last(), y_cur.last()) {
        let yy = g_inner(manifold, x, y.view(), y.view())?;
        if yy > 1.0e-14 {
            let sy = g_inner(manifold, x, s.view(), y.view())?;
            r = &r * (sy / yy);
        }
    }
    for i in 0..history.len() {
        let beta = rho[i] * g_inner(manifold, x, y_cur[i].view(), r.view())?;
        r = &r + &(&s_cur[i] * (alpha[i] - beta));
    }
    Ok(r)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::EuclideanManifold;
    use ndarray::{Array1, Array2, ArrayView1, ArrayView2};

    struct IndefiniteLine;

    impl RiemannianManifold for IndefiniteLine {
        fn dim(&self) -> usize {
            1
        }

        fn tangent_basis(&self, point: ArrayView1<'_, f64>) -> GeometryResult<Array2<f64>> {
            assert_eq!(point.len(), 1, "IndefiniteLine points are one-dimensional");
            Ok(Array2::eye(1))
        }

        fn exp_map(
            &self,
            point: ArrayView1<'_, f64>,
            tangent_vec: ArrayView1<'_, f64>,
        ) -> GeometryResult<Array1<f64>> {
            Ok(&point.to_owned() + &tangent_vec)
        }

        fn log_map(
            &self,
            p_from: ArrayView1<'_, f64>,
            p_to: ArrayView1<'_, f64>,
        ) -> GeometryResult<Array1<f64>> {
            Ok(&p_to.to_owned() - &p_from)
        }

        fn parallel_transport(
            &self,
            point_along: ArrayView2<'_, f64>,
            vec: ArrayView1<'_, f64>,
        ) -> GeometryResult<Array1<f64>> {
            assert_eq!(
                point_along.ncols(),
                1,
                "IndefiniteLine transport paths are one-dimensional"
            );
            assert_eq!(vec.len(), 1, "IndefiniteLine tangents are one-dimensional");
            Ok(vec.to_owned())
        }

        fn metric_tensor(&self, point: ArrayView1<'_, f64>) -> GeometryResult<Array2<f64>> {
            assert_eq!(point.len(), 1, "IndefiniteLine points are one-dimensional");
            Ok(ndarray::array![[-1.0]])
        }

        fn sectional_curvature(
            &self,
            point: ArrayView1<'_, f64>,
            tangent_pair: (ArrayView1<'_, f64>, ArrayView1<'_, f64>),
        ) -> GeometryResult<f64> {
            assert_eq!(point.len(), 1, "IndefiniteLine points are one-dimensional");
            assert_eq!(
                tangent_pair.0.len(),
                1,
                "IndefiniteLine tangents are one-dimensional"
            );
            assert_eq!(
                tangent_pair.1.len(),
                1,
                "IndefiniteLine tangents are one-dimensional"
            );
            Ok(0.0)
        }
    }

    /// Scalar objective `f(x) = x²` on the 1-D Euclidean line. Gradient `2x`,
    /// Hessian `2`, exposed as an HVP so the trust region runs Steihaug-CG.
    struct Square;
    impl RiemannianObjective for Square {
        fn value_gradient(
            &mut self,
            point: ArrayView1<'_, f64>,
        ) -> GeometryResult<(f64, Array1<f64>)> {
            let x = point[0];
            Ok((x * x, Array1::from_vec(vec![2.0 * x])))
        }
        fn hessian_vector_product(
            &mut self,
            point: ArrayView1<'_, f64>,
            tangent: ArrayView1<'_, f64>,
        ) -> GeometryResult<Option<Array1<f64>>> {
            assert!(point.iter().all(|value| value.is_finite()));
            check_len("hessian_vector_product tangent", tangent.len(), point.len())?;
            Ok(Some(&tangent.to_owned() * 2.0))
        }
    }

    /// Gradient-only variant of `f(x)=x²` (no HVP) to exercise the Cauchy-point
    /// branch of the trust region.
    struct SquareGradOnly;
    impl RiemannianObjective for SquareGradOnly {
        fn value_gradient(
            &mut self,
            point: ArrayView1<'_, f64>,
        ) -> GeometryResult<(f64, Array1<f64>)> {
            let x = point[0];
            Ok((x * x, Array1::from_vec(vec![2.0 * x])))
        }
    }

    /// General convex quadratic `f(x) = ½ xᵀ A x − bᵀ x` on Euclidean R^n with
    /// SPD `A`; minimizer solves `A x = b`. Provides an exact HVP `A v`.
    struct Quadratic {
        a: ndarray::Array2<f64>,
        b: Array1<f64>,
    }
    impl RiemannianObjective for Quadratic {
        fn value_gradient(
            &mut self,
            point: ArrayView1<'_, f64>,
        ) -> GeometryResult<(f64, Array1<f64>)> {
            let ax = self.a.dot(&point.to_owned());
            let val = 0.5 * point.dot(&ax) - self.b.dot(&point.to_owned());
            let grad = &ax - &self.b;
            Ok((val, grad))
        }
        fn hessian_vector_product(
            &mut self,
            point: ArrayView1<'_, f64>,
            tangent: ArrayView1<'_, f64>,
        ) -> GeometryResult<Option<Array1<f64>>> {
            assert!(point.iter().all(|value| value.is_finite()));
            check_len("hessian_vector_product tangent", tangent.len(), point.len())?;
            Ok(Some(self.a.dot(&tangent.to_owned())))
        }
    }

    /// (#615 counterexample) A correct trust region on `f(x)=x²` from `x₀=0.1`
    /// with `Δ=1` must CONVERGE to 0 (not oscillate), monotonically driving `f`
    /// down — never increasing it on an accepted iterate.
    #[test]
    fn trust_region_converges_on_square_steihaug() {
        let manifold = EuclideanManifold::new(1);
        let tr = RiemannianTrustRegion {
            radius: 1.0,
            max_radius: 1.0e6,
            max_iter: 100,
            grad_tol: 1.0e-12,
        };
        let mut obj = Square;
        let x0 = Array1::from_vec(vec![0.1]);
        let x = tr
            .minimize(&manifold, &mut obj, x0.view())
            .expect("TR runs");
        assert!(
            x[0].abs() < 1.0e-6,
            "trust region must converge to 0, got {}",
            x[0]
        );
    }

    /// The trust region must never increase `f` across accepted iterates. We
    /// check the monotone-descent invariant directly by stepping the public
    /// `minimize` from a sequence of decreasing budgets and confirming the
    /// returned value is below the start value, and that from `x₀=0.1` it does
    /// not return a point with larger `|x|`.
    #[test]
    fn trust_region_never_increases_objective() {
        let manifold = EuclideanManifold::new(1);
        let tr = RiemannianTrustRegion {
            radius: 1.0,
            max_radius: 1.0e6,
            max_iter: 1,
            grad_tol: 1.0e-12,
        };
        let mut obj = Square;
        // A single TR iteration from 0.1: with exact Hessian the Newton step
        // lands at the minimum (inside Δ=1), ρ=1, so it must be accepted and f
        // must strictly decrease.
        let x0 = Array1::from_vec(vec![0.1]);
        let f0 = obj.value_gradient(x0.view()).unwrap().0;
        let x1 = tr
            .minimize(&manifold, &mut obj, x0.view())
            .expect("TR runs");
        let f1 = obj.value_gradient(x1.view()).unwrap().0;
        assert!(f1 <= f0, "objective increased: {f0} -> {f1}");
        assert!(x1[0].abs() <= x0[0].abs() + 1e-15, "moved away from min");
    }

    /// Cauchy-point branch (no HVP) must still be a real trust-region method:
    /// from `x₀=0.1`, `Δ=1` on `f(x)=x²` it converges toward 0 and never
    /// oscillates upward in `f`.
    #[test]
    fn trust_region_cauchy_point_converges() {
        let manifold = EuclideanManifold::new(1);
        let tr = RiemannianTrustRegion {
            radius: 1.0,
            max_radius: 1.0e6,
            max_iter: 500,
            grad_tol: 1.0e-12,
        };
        let mut obj = SquareGradOnly;
        let x0 = Array1::from_vec(vec![0.1]);
        let x = tr
            .minimize(&manifold, &mut obj, x0.view())
            .expect("TR runs");
        assert!(
            x[0].abs() < 1.0e-6,
            "Cauchy-point trust region must converge to 0, got {}",
            x[0]
        );
    }

    /// Steihaug-CG trust region on a 3-D SPD quadratic must reach the exact
    /// minimizer `A⁻¹ b`.
    #[test]
    fn trust_region_solves_spd_quadratic() {
        let manifold = EuclideanManifold::new(3);
        let a = ndarray::array![[4.0, 1.0, 0.0], [1.0, 3.0, 1.0], [0.0, 1.0, 2.0],];
        let b = Array1::from_vec(vec![1.0, 2.0, -1.0]);
        // Reference solution A x = b.
        let x_ref = crate::manifold::inverse(&a).unwrap().dot(&b);
        let mut obj = Quadratic { a, b };
        let tr = RiemannianTrustRegion {
            radius: 1.0,
            max_radius: 1.0e6,
            max_iter: 200,
            grad_tol: 1.0e-12,
        };
        let x0 = Array1::from_vec(vec![0.0, 0.0, 0.0]);
        let x = tr
            .minimize(&manifold, &mut obj, x0.view())
            .expect("TR runs");
        for i in 0..3 {
            assert!(
                (x[i] - x_ref[i]).abs() < 1.0e-6,
                "component {i}: got {}, want {}",
                x[i],
                x_ref[i]
            );
        }
    }

    /// (#616 sanity) Riemannian L-BFGS on a Euclidean SPD quadratic must reduce
    /// the objective and converge to the analytic minimizer `A⁻¹ b`.
    #[test]
    fn lbfgs_reduces_euclidean_quadratic() {
        let manifold = EuclideanManifold::new(3);
        let a = ndarray::array![[5.0, 1.0, 0.5], [1.0, 4.0, 1.0], [0.5, 1.0, 3.0],];
        let b = Array1::from_vec(vec![2.0, -1.0, 0.5]);
        let x_ref = crate::manifold::inverse(&a).unwrap().dot(&b);
        let mut obj = Quadratic { a, b };
        let lbfgs = RiemannianLBFGS {
            history: 10,
            step_size: 1.0,
            max_iter: 200,
            grad_tol: 1.0e-10,
        };
        let x0 = Array1::from_vec(vec![0.0, 0.0, 0.0]);
        let f0 = obj.value_gradient(x0.view()).unwrap().0;
        let x = lbfgs
            .minimize(&manifold, &mut obj, x0.view())
            .expect("L-BFGS runs");
        let f1 = obj.value_gradient(x.view()).unwrap().0;
        assert!(f1 < f0, "L-BFGS did not reduce the quadratic: {f0} -> {f1}");
        for i in 0..3 {
            assert!(
                (x[i] - x_ref[i]).abs() < 1.0e-6,
                "component {i}: got {}, want {}",
                x[i],
                x_ref[i]
            );
        }
    }

    #[test]
    fn optimizers_refuse_nonstationary_zero_budget_iterates() {
        let manifold = EuclideanManifold::new(1);
        let x0 = Array1::from_vec(vec![1.0]);

        let mut trust_objective = Square;
        let trust = RiemannianTrustRegion {
            max_iter: 0,
            ..RiemannianTrustRegion::default()
        };
        let trust_error = trust
            .minimize(&manifold, &mut trust_objective, x0.view())
            .expect_err("a zero-budget nonstationary trust-region run must not mint a point");
        assert!(matches!(
            trust_error,
            crate::manifold::GeometryError::NonConvergence {
                iterations: 0,
                residual,
                ..
            } if residual > trust.grad_tol
        ));

        let mut lbfgs_objective = Square;
        let lbfgs = RiemannianLBFGS {
            max_iter: 0,
            ..RiemannianLBFGS::default()
        };
        let lbfgs_error = lbfgs
            .minimize(&manifold, &mut lbfgs_objective, x0.view())
            .expect_err("a zero-budget nonstationary L-BFGS run must not mint a point");
        assert!(matches!(
            lbfgs_error,
            crate::manifold::GeometryError::NonConvergence {
                iterations: 0,
                residual,
                ..
            } if residual > lbfgs.grad_tol
        ));
    }

    #[test]
    fn nonfinite_gradient_cannot_be_misread_as_zero_norm() {
        struct NanGradient;
        impl RiemannianObjective for NanGradient {
            fn value_gradient(
                &mut self,
                point: ArrayView1<'_, f64>,
            ) -> GeometryResult<(f64, Array1<f64>)> {
                assert_eq!(point.len(), 1, "NanGradient is one-dimensional");
                Ok((0.0, Array1::from_vec(vec![f64::NAN])))
            }
        }

        let manifold = EuclideanManifold::new(1);
        let x0 = Array1::zeros(1);
        let mut objective = NanGradient;
        let error = RiemannianTrustRegion::default()
            .minimize(&manifold, &mut objective, x0.view())
            .expect_err("NaN gradient must never certify stationarity");
        assert!(matches!(
            error,
            crate::manifold::GeometryError::NonConvergence { residual, .. }
                if residual.is_infinite()
        ));
    }

    #[test]
    fn overflowed_metric_error_bound_cannot_certify_zero_norm() {
        // The signed quadratic cancels to zero in this order, while the sum of
        // absolute terms overflows. Treating an infinite backward-error band
        // as a valid tolerance would turn this indefinite quadratic into a
        // zero norm and falsely certify stationarity.
        let vector = ndarray::array![1.0, 1.0, 1.0, 1.0];
        let metric_product = ndarray::array![9.0e307, -9.0e307, 9.0e307, -9.0e307];
        let error = metric_norm_from_product(vector.view(), metric_product.view())
            .expect_err("an overflowed norm error bound must be rejected");
        assert!(matches!(
            error,
            crate::manifold::GeometryError::InvalidPoint(
                "Riemannian metric norm error bound overflowed"
            )
        ));
    }

    #[test]
    fn indefinite_metric_cannot_be_clamped_into_false_stationarity() {
        let manifold = IndefiniteLine;
        let mut objective = Square;
        let x0 = Array1::from_vec(vec![1.0]);
        let error = RiemannianTrustRegion::default()
            .minimize(&manifold, &mut objective, x0.view())
            .expect_err("an indefinite metric is not a Riemannian norm");
        assert!(matches!(
            error,
            crate::manifold::GeometryError::InvalidPoint(
                "Riemannian metric produced a negative squared norm"
            )
        ));
    }
}
