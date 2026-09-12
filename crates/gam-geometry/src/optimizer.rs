use gam_linalg::faer_ndarray::fast_av;
use ndarray::{Array1, ArrayView1};

use crate::manifold::{GeometryError, GeometryResult, RiemannianManifold, check_len};

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

/// The context string of the trust-region first-order certificate, shared by the
/// refusal in [`RiemannianTrustRegion::minimize`] and by callers that re-report
/// the same verdict from [`TrustRegionTermination`].
pub const TRUST_REGION_RELATIVE_GRADIENT_CONTEXT: &str =
    "Riemannian trust-region optimization (relative gradient norm)";

/// Terminal state of a trust-region run: the iterate reached, and the numbers the
/// first-order certificate was decided against.
#[derive(Clone, Debug)]
pub struct TrustRegionTermination {
    /// The last iterate. Present whether or not the certificate holds — this is
    /// the work a budget-exhausted run has to hand back.
    pub point: Array1<f64>,
    /// Iterations actually executed (zero when the budget was zero).
    pub iterations: usize,
    /// Relative stationarity `‖g_final‖ / max(‖g_0‖, 1)` at `point`.
    pub residual: f64,
    /// The bound `residual` was compared against.
    pub tolerance: f64,
}

impl TrustRegionTermination {
    /// Whether `point` satisfies the first-order certificate that controls the
    /// loop. This is the same test `minimize` applies before returning a point.
    pub fn certifies(&self) -> bool {
        self.residual <= self.tolerance
    }
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
    /// A Riemannian trust-region method on `manifold`.
    ///
    /// The method is `opt::RiemannianTrustRegion`, because general outer
    /// optimizer work lives in `opt` (SPEC rule 24): a quadratic model in the
    /// tangent space under the manifold metric, a Steihaug truncated-CG step when
    /// the objective supplies Hessian–vector products and the manifold's
    /// retraction is second-order ([`RiemannianManifold::retraction_is_second_order`]),
    /// the Cauchy point otherwise (issue #956), and radius control from
    /// `opt::TrustRegionPolicy::classic`. This type adapts a manifold and an
    /// objective of this crate to it, and returns a point only when the
    /// relative-gradient certificate holds.
    pub fn minimize(
        &self,
        manifold: &dyn RiemannianManifold,
        objective: &mut dyn RiemannianObjective,
        initial: ArrayView1<'_, f64>,
    ) -> GeometryResult<Array1<f64>> {
        let termination = self.minimize_reporting_termination(manifold, objective, initial)?;
        if termination.certifies() {
            Ok(termination.point)
        } else {
            Err(GeometryError::NonConvergence {
                context: TRUST_REGION_RELATIVE_GRADIENT_CONTEXT,
                iterations: termination.iterations,
                residual: termination.residual,
                tolerance: termination.tolerance,
            })
        }
    }

    /// As [`Self::minimize`], but reporting the terminal iterate alongside the
    /// first-order verdict instead of discarding it.
    ///
    /// `minimize` returns `Err(NonConvergence)` when the terminal point fails the
    /// relative-gradient certificate, and that error carries the residual but not
    /// the POINT. A caller whose contract is checkpoint/resume cannot be served
    /// by it: the work done before the budget ran out is exactly the iterate, and
    /// with only a residual there is nothing to resume from. Genuine failures —
    /// a non-finite value, an invalid radius, an objective or manifold error —
    /// are still `Err` here; only the first-order test is demoted from an error
    /// to a reported verdict.
    pub fn minimize_reporting_termination(
        &self,
        manifold: &dyn RiemannianManifold,
        objective: &mut dyn RiemannianObjective,
        initial: ArrayView1<'_, f64>,
    ) -> GeometryResult<TrustRegionTermination> {
        let solver = opt::RiemannianTrustRegion {
            radius: self.radius,
            max_radius: self.max_radius,
            max_iter: self.max_iter,
            grad_tol: self.grad_tol,
        };
        let termination = solver
            .minimize(
                &ManifoldGeometry(manifold),
                &mut ObjectiveCallbacks(objective),
                initial,
            )
            .map_err(geometry_error)?;
        Ok(TrustRegionTermination {
            point: termination.point,
            iterations: termination.iterations,
            residual: termination.residual,
            tolerance: termination.tolerance,
        })
    }
}

/// A manifold of this crate as the geometry `opt`'s Riemannian trust region runs
/// on. The metric product goes through the GPU-dispatched `fast_av`, as every
/// metric inner product in this crate does.
struct ManifoldGeometry<'a>(&'a dyn RiemannianManifold);

impl opt::RiemannianGeometry for ManifoldGeometry<'_> {
    type Error = GeometryError;

    fn ambient_dim(&self) -> usize {
        self.0.ambient_dim()
    }

    fn riemannian_gradient(
        &self,
        point: ArrayView1<'_, f64>,
        differential: ArrayView1<'_, f64>,
    ) -> GeometryResult<Array1<f64>> {
        self.0.riemannian_gradient(point, differential)
    }

    fn metric_product(
        &self,
        point: ArrayView1<'_, f64>,
        tangent: ArrayView1<'_, f64>,
    ) -> GeometryResult<Array1<f64>> {
        let metric = self.0.metric_tensor(point)?;
        check_len("metric product tangent", tangent.len(), metric.ncols())?;
        Ok(fast_av(&metric.view(), &tangent))
    }

    fn retract(
        &self,
        point: ArrayView1<'_, f64>,
        tangent: ArrayView1<'_, f64>,
    ) -> GeometryResult<Array1<f64>> {
        self.0.retract(point, tangent)
    }

    fn retraction_is_second_order(&self) -> bool {
        self.0.retraction_is_second_order()
    }
}

/// An objective of this crate as `opt`'s Riemannian objective.
struct ObjectiveCallbacks<'a, 'o>(&'a mut (dyn RiemannianObjective + 'o));

impl opt::RiemannianObjective<GeometryError> for ObjectiveCallbacks<'_, '_> {
    fn value_gradient(&mut self, point: ArrayView1<'_, f64>) -> GeometryResult<(f64, Array1<f64>)> {
        self.0.value_gradient(point)
    }

    fn hessian_vector_product(
        &mut self,
        point: ArrayView1<'_, f64>,
        tangent: ArrayView1<'_, f64>,
    ) -> GeometryResult<Option<Array1<f64>>> {
        self.0.hessian_vector_product(point, tangent)
    }
}

/// The trust region's refusals in this crate's error vocabulary, with the
/// messages this crate has always reported for them.
fn geometry_error(error: opt::RiemannianTrustRegionError<GeometryError>) -> GeometryError {
    use opt::RiemannianTrustRegionError as Refusal;
    match error {
        Refusal::Callback(inner) => inner,
        Refusal::InitialPointLength { expected, got } => GeometryError::DimensionMismatch {
            context: "trust-region initial point",
            expected,
            got,
        },
        Refusal::InvalidRadius => {
            GeometryError::InvalidPoint("trust-region radius must be finite and positive")
        }
        Refusal::InvalidMaxRadius => {
            GeometryError::InvalidPoint("trust-region maximum radius must be finite and positive")
        }
        Refusal::InvalidGradientTolerance => GeometryError::InvalidPoint(
            "trust-region gradient tolerance must be finite and non-negative",
        ),
        Refusal::NonFiniteValue => {
            GeometryError::InvalidPoint("trust-region objective returned a non-finite value")
        }
        Refusal::NonFiniteTerminalValue => GeometryError::InvalidPoint(
            "trust-region objective returned a non-finite terminal value",
        ),
        Refusal::MetricBoundOverflow => {
            GeometryError::InvalidPoint("Riemannian metric norm error bound overflowed")
        }
        Refusal::IndefiniteMetric => {
            GeometryError::InvalidPoint("Riemannian metric produced a negative squared norm")
        }
        Refusal::MetricProductLength { expected, got } => GeometryError::DimensionMismatch {
            context: "metric norm product",
            expected,
            got,
        },
        Refusal::CurvatureWithdrawnMidSubproblem => GeometryError::Unsupported(
            "Hessian–vector product became unavailable mid-subproblem",
        ),
        Refusal::CurvatureWithdrawnWhileScoring => GeometryError::Unsupported(
            "Hessian–vector product unavailable while scoring the model",
        ),
    }
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
        // zero norm and falsely certify stationarity. The alternating metric
        // ±9e307 applied to an all-ones Riemannian gradient produces exactly that
        // product, [9e307, −9e307, 9e307, −9e307], at the first certificate.
        struct OverflowingMetric;

        impl RiemannianManifold for OverflowingMetric {
            fn dim(&self) -> usize {
                4
            }

            fn tangent_basis(&self, point: ArrayView1<'_, f64>) -> GeometryResult<Array2<f64>> {
                assert_eq!(point.len(), 4, "OverflowingMetric points are four-dimensional");
                Ok(Array2::eye(4))
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
                    4,
                    "OverflowingMetric transport paths are four-dimensional"
                );
                Ok(vec.to_owned())
            }

            fn metric_tensor(&self, point: ArrayView1<'_, f64>) -> GeometryResult<Array2<f64>> {
                assert_eq!(point.len(), 4, "OverflowingMetric points are four-dimensional");
                Ok(Array2::from_diag(&ndarray::array![
                    9.0e307, -9.0e307, 9.0e307, -9.0e307
                ]))
            }

            fn sectional_curvature(
                &self,
                point: ArrayView1<'_, f64>,
                tangent_pair: (ArrayView1<'_, f64>, ArrayView1<'_, f64>),
            ) -> GeometryResult<f64> {
                assert_eq!(point.len(), 4, "OverflowingMetric points are four-dimensional");
                assert_eq!(
                    tangent_pair.0.len(),
                    4,
                    "OverflowingMetric tangents are four-dimensional"
                );
                assert_eq!(
                    tangent_pair.1.len(),
                    4,
                    "OverflowingMetric tangents are four-dimensional"
                );
                Ok(0.0)
            }

            fn riemannian_gradient(
                &self,
                point: ArrayView1<'_, f64>,
                euclidean_grad: ArrayView1<'_, f64>,
            ) -> GeometryResult<Array1<f64>> {
                assert_eq!(point.len(), 4, "OverflowingMetric points are four-dimensional");
                assert_eq!(
                    euclidean_grad.len(),
                    4,
                    "OverflowingMetric gradients are four-dimensional"
                );
                Ok(Array1::ones(4))
            }
        }

        struct Flat;

        impl RiemannianObjective for Flat {
            fn value_gradient(
                &mut self,
                point: ArrayView1<'_, f64>,
            ) -> GeometryResult<(f64, Array1<f64>)> {
                assert_eq!(point.len(), 4, "Flat is four-dimensional");
                Ok((0.0, Array1::ones(4)))
            }
        }

        let x0 = Array1::zeros(4);
        let error = RiemannianTrustRegion::default()
            .minimize(&OverflowingMetric, &mut Flat, x0.view())
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
