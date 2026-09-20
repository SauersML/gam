#![cfg(test)]
//! #3285 — each joint Newton step takes the route its solve's own measurements
//! price cheaper, CG or the dense build, and the route changes what the solve
//! costs, never the mode it certifies.
//!
//! A ridge-penalized logistic regression on overlapping Gaussian bumps. The
//! overlap spreads the penalized Hessian's spectrum, so Jacobi-preconditioned
//! CG needs several products, and the workspace serves the Hessian as an
//! operator (a diagonal, products, and a structural dense build), so every step
//! may take either route. Two workspaces serve the same Hessian and differ only
//! in which operation is slow, so the measured costs send one solve through CG
//! and the other through the dense build.
//!
//! The penalized objective `F(β) = −ℓ(β) + ½λ‖β‖²` is λ-strongly convex, so two
//! points whose gradients meet the certificate's target `t` in the `∞`-norm lie
//! within `2√p·t/λ` of each other: that is the resolution two certified modes
//! can differ by.
use super::*;
use std::time::Duration;

const ROWS: usize = 400;
const BUMPS: usize = 24;
const INNER_TOL: f64 = 1.0e-9;
/// The delay of the slow operation. Each is over a hundred times the other
/// operation's cost at this size (a dense build is about `n·p²` = 2.3e5 flops,
/// a product `2·n·p` = 1.9e4), so the measured comparison has one answer on any
/// machine load.
const DENSE_BUILD_DELAY: Duration = Duration::from_millis(200);
const PRODUCT_DELAY: Duration = Duration::from_millis(20);

/// Which of the workspace's two Hessian operations is slow.
#[derive(Clone, Copy, Debug)]
enum SlowOperation {
    DenseBuild,
    Product,
}

/// How often a solve used each Hessian operation.
#[derive(Default)]
struct OperationCounts {
    dense_builds: AtomicUsize,
    products: AtomicUsize,
}

#[derive(Clone)]
struct BumpLogisticFamily {
    design: Arc<Array2<f64>>,
    response: Arc<Array1<f64>>,
    slow: SlowOperation,
    counts: Arc<OperationCounts>,
}

/// The log-likelihood, its gradient and the IRLS weights at one linear predictor.
struct LogisticFit {
    log_likelihood: f64,
    gradient: Array1<f64>,
    weights: Array1<f64>,
}

impl BumpLogisticFamily {
    fn new(slow: SlowOperation) -> Self {
        let width = 2.0 / (BUMPS - 1) as f64;
        let position = |row: usize| (row as f64 + 0.5) / ROWS as f64;
        let design = Array2::from_shape_fn((ROWS, BUMPS), |(row, bump)| {
            let center = bump as f64 / (BUMPS - 1) as f64;
            (-((position(row) - center) / width).powi(2)).exp()
        });
        let mut state = 0x3285_u64;
        let response = (0..ROWS)
            .map(|row| {
                let logit = 2.0 * (std::f64::consts::TAU * position(row)).sin();
                let probability = 1.0 / (1.0 + (-logit).exp());
                let unit = (gam_linalg::utils::splitmix64(&mut state) >> 11) as f64
                    / (1u64 << 53) as f64;
                f64::from(u8::from(unit < probability))
            })
            .collect();
        Self {
            design: Arc::new(design),
            response: Arc::new(response),
            slow,
            counts: Arc::new(OperationCounts::default()),
        }
    }

    fn fit_at(&self, eta: &Array1<f64>) -> LogisticFit {
        let mut log_likelihood = 0.0;
        let mut residual = Array1::<f64>::zeros(ROWS);
        let mut weights = Array1::<f64>::zeros(ROWS);
        for row in 0..ROWS {
            let linear = eta[row];
            let mean = 1.0 / (1.0 + (-linear).exp());
            let softplus = linear.max(0.0) + (-linear.abs()).exp().ln_1p();
            log_likelihood += self.response[row] * linear - softplus;
            residual[row] = self.response[row] - mean;
            weights[row] = mean * (1.0 - mean);
        }
        LogisticFit {
            log_likelihood,
            gradient: self.design.t().dot(&residual),
            weights,
        }
    }

    fn spec(&self, rho: f64) -> ParameterBlockSpec {
        ParameterBlockSpec {
            name: "bumps".to_string(),
            design: DesignMatrix::Dense(gam_linalg::matrix::DenseDesignMatrix::from(
                (*self.design).clone(),
            )),
            offset: Array1::zeros(ROWS),
            penalties: vec![PenaltyMatrix::Dense(Array2::eye(BUMPS))],
            nullspace_dims: vec![],
            initial_log_lambdas: array![rho],
            initial_beta: None,
            gauge_priority: 100,
            jacobian_callback: None,
            stacked_design: None,
            stacked_offset: None,
        }
    }

    /// The `∞`-norm of `∇F(β)` at `λ = e^ρ`, and the certificate's target for
    /// it, `INNER_TOL·(1 + max(‖∇ℓ‖∞, ‖λβ‖∞))`.
    fn stationarity(&self, rho: f64, beta: &Array1<f64>) -> (f64, f64) {
        let fit = self.fit_at(&self.design.dot(beta));
        let penalty_gradient = beta * rho.exp();
        let residual = (&fit.gradient - &penalty_gradient)
            .iter()
            .fold(0.0_f64, |largest, value| largest.max(value.abs()));
        let scale = fit
            .gradient
            .iter()
            .chain(penalty_gradient.iter())
            .fold(0.0_f64, |largest, value| largest.max(value.abs()));
        (residual, INNER_TOL * (1.0 + scale))
    }
}

fn weighted_gram(design: &Array2<f64>, weights: &Array1<f64>) -> Array2<f64> {
    let weighted = design * &weights.view().insert_axis(ndarray::Axis(1));
    design.t().dot(&weighted)
}

struct BumpLogisticWorkspace {
    design: Arc<Array2<f64>>,
    weights: Array1<f64>,
    slow: SlowOperation,
    counts: Arc<OperationCounts>,
}

impl ExactNewtonJointHessianWorkspace for BumpLogisticWorkspace {
    fn warm_up_outer_caches_for_mode(&self, eval_mode: EvalMode) -> Result<(), String> {
        // No directional cache to prime, in any mode.
        match eval_mode {
            EvalMode::ValueOnly | EvalMode::ValueAndGradient | EvalMode::ValueGradientHessian => {
                Ok(())
            }
        }
    }

    /// The structural dense build: the dense route's cost.
    fn hessian_dense_forced(&self) -> Result<Option<Array2<f64>>, String> {
        self.counts.dense_builds.fetch_add(1, Ordering::Relaxed);
        if matches!(self.slow, SlowOperation::DenseBuild) {
            std::thread::sleep(DENSE_BUILD_DELAY);
        }
        Ok(Some(weighted_gram(&self.design, &self.weights)))
    }

    fn hessian_matvec_available(&self) -> bool {
        true
    }

    /// One product: the CG route's cost per iteration.
    fn hessian_matvec(&self, arr: &Array1<f64>) -> Result<Option<Array1<f64>>, String> {
        assert_direction_finite(arr, "bump logistic workspace Hessian product");
        self.counts.products.fetch_add(1, Ordering::Relaxed);
        if matches!(self.slow, SlowOperation::Product) {
            std::thread::sleep(PRODUCT_DELAY);
        }
        let weighted = &self.weights * &self.design.dot(arr);
        Ok(Some(self.design.t().dot(&weighted)))
    }

    fn hessian_diagonal(&self) -> Result<Option<Array1<f64>>, String> {
        let squared = self.design.mapv(|value| value * value);
        Ok(Some(squared.t().dot(&self.weights)))
    }

    fn directional_derivative(
        &self,
        direction: &Array1<f64>,
    ) -> Result<Option<Array2<f64>>, String> {
        assert_direction_finite(direction, "bump logistic workspace directional derivative");
        Ok(None)
    }
}

impl CustomFamily for BumpLogisticFamily {
    fn evaluate(&self, block_states: &[ParameterBlockState]) -> Result<FamilyEvaluation, String> {
        assert_states_finite(block_states, "bump logistic evaluation");
        let fit = self.fit_at(&block_states[0].eta);
        Ok(FamilyEvaluation {
            log_likelihood: fit.log_likelihood,
            blockworking_sets: vec![BlockWorkingSet::ExactNewton {
                hessian: SymmetricMatrix::Dense(weighted_gram(&self.design, &fit.weights)),
                gradient: fit.gradient,
            }],
        })
    }

    fn inner_coefficient_hessian_hvp_available(&self, specs: &[ParameterBlockSpec]) -> bool {
        assert_specs_consistent(specs, "bump logistic HVP availability");
        true
    }

    fn exact_newton_joint_hessian_workspace(
        &self,
        block_states: &[ParameterBlockState],
        specs: &[ParameterBlockSpec],
    ) -> Result<Option<Arc<dyn ExactNewtonJointHessianWorkspace>>, String> {
        assert_specs_consistent(specs, "bump logistic workspace");
        assert_states_finite(block_states, "bump logistic workspace");
        Ok(Some(Arc::new(BumpLogisticWorkspace {
            design: Arc::clone(&self.design),
            weights: self.fit_at(&block_states[0].eta).weights,
            slow: self.slow,
            counts: Arc::clone(&self.counts),
        })))
    }
}

/// A cold inner solve at `ρ`, with the dense builds and products it used.
fn solve(slow: SlowOperation, rho: f64) -> (BlockwiseInnerResult, usize, usize) {
    let family = BumpLogisticFamily::new(slow);
    let options = BlockwiseFitOptions {
        inner_max_cycles: 400,
        inner_tol: INNER_TOL,
        use_remlobjective: false,
        compute_covariance: false,
        ..BlockwiseFitOptions::default()
    };
    let result = inner_blockwise_fit(&family, &[family.spec(rho)], &[array![rho]], &options, None)
        .expect("a bump logistic inner solve ends with an inner result, not a refusal");
    let dense_builds = family.counts.dense_builds.load(Ordering::Relaxed);
    let products = family.counts.products.load(Ordering::Relaxed);
    (result, dense_builds, products)
}

/// The solve whose dense build is slow takes CG after its first, dense step,
/// the solve whose products are slow keeps the dense route, and both certify one
/// mode (#3285).
#[test]
fn a_joint_newton_step_takes_the_route_its_solve_measured_cheaper_3285() {
    let rho = 0.0;
    let (cg, cg_dense_builds, cg_products) = solve(SlowOperation::DenseBuild, rho);
    let (dense, dense_dense_builds, dense_products) = solve(SlowOperation::Product, rho);
    assert!(
        cg_dense_builds < dense_dense_builds && cg_products > dense_products,
        "slow dense build: {cg_dense_builds} dense builds, {cg_products} products over {} cycles; \
         slow products: {dense_dense_builds} dense builds, {dense_products} products over {} cycles",
        cg.cycles,
        dense.cycles
    );
    // Both workspaces serve one objective, so either family measures stationarity.
    let family = BumpLogisticFamily::new(SlowOperation::Product);
    let mut targets = Vec::with_capacity(2);
    for (label, result) in [("slow dense build", &cg), ("slow products", &dense)] {
        let (residual, target) = family.stationarity(rho, &result.block_states[0].beta);
        assert!(
            result.converged && residual <= target,
            "{label}: converged={} after {} cycles with ‖∇F‖∞ = {residual:.3e} against the target {target:.3e}",
            result.converged,
            result.cycles
        );
        targets.push(target);
    }
    let difference = &cg.block_states[0].beta - &dense.block_states[0].beta;
    let distance = difference.dot(&difference).sqrt();
    let resolution = 2.0 * (BUMPS as f64).sqrt() * targets[0].max(targets[1]) / rho.exp();
    assert!(
        distance <= resolution,
        "the two routes' modes lie {distance:.3e} apart, beyond the resolution {resolution:.3e}"
    );
}
