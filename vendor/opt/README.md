# opt

[![Crates.io](https://img.shields.io/crates/v/opt.svg)](https://crates.io/crates/opt)
[![Docs.rs](https://docs.rs/opt/badge.svg)](https://docs.rs/opt)
[![Build Status](https://github.com/SauersML/opt/actions/workflows/test.yml/badge.svg)](https://github.com/SauersML/opt/actions)

Dense nonlinear optimization in Rust with:
- `Bfgs` for first-order dense quasi-Newton optimization
- `NewtonTrustRegion` for Hessian-based trust-region optimization
- `Arc` for adaptive regularization with cubics
- `FixedPoint` for bounded fixed-point iteration
- automatic solver selection through `Problem`, `SecondOrderProblem`, and `optimize`

This crate is designed for practical nonlinear objectives, including optional simple box constraints, and is built around robustness for noisy or non-ideal functions.

This work is a rewrite of the original `bfgs` crate by Paul Kernfeld.

## Features

- Strong Wolfe line search with practical fallback behavior for difficult first-order problems
- Dense BFGS with inverse-Hessian updates and stability safeguards
- Newton trust-region steps using supplied Hessians
- ARC with adaptive cubic regularization updates
- Fixed-point iteration with projection and step-norm termination
- Automatic solver selection for second-order objectives
- Optional simple box constraints with projected gradients
- Internal finite-difference support for cost-only and Hessian-optional objectives
- Structured error reporting with recoverable optimization failures

## Usage

Add this to your `Cargo.toml`:

```toml
[dependencies]
opt = "0.2.0"
```

### Example: First-Order Optimization

```rust
use opt::{
    optimize, FirstOrderObjective, FirstOrderSample, MaxIterations, Problem, Profile, Solution,
    Tolerance, ZerothOrderObjective,
};
use ndarray::{array, Array1};

struct Rosenbrock;

impl ZerothOrderObjective for Rosenbrock {
    fn eval_cost(&mut self, x: &Array1<f64>) -> Result<f64, opt::ObjectiveEvalError> {
        let a = 1.0;
        let b = 100.0;
        Ok((a - x[0]).powi(2) + b * (x[1] - x[0].powi(2)).powi(2))
    }
}

impl FirstOrderObjective for Rosenbrock {
    fn eval_grad(&mut self, x: &Array1<f64>) -> Result<FirstOrderSample, opt::ObjectiveEvalError> {
        let a = 1.0;
        let b = 100.0;
        let f = (a - x[0]).powi(2) + b * (x[1] - x[0].powi(2)).powi(2);
        Ok(FirstOrderSample {
            value: f,
            gradient: array![
                -2.0 * (a - x[0]) - 4.0 * b * (x[1] - x[0].powi(2)) * x[0],
                2.0 * b * (x[1] - x[0].powi(2)),
            ],
        })
    }
}

let x0 = array![-1.2, 1.0];

let Solution {
    final_point: x_min,
    final_value,
    final_gradient_norm: Some(grad_norm),
    iterations,
    ..
} = optimize(Problem::new(x0, Rosenbrock))
    .with_tolerance(Tolerance::new(1e-6).unwrap())
    .with_max_iterations(MaxIterations::new(100).unwrap())
    .with_profile(Profile::Robust)
    .run()
    .expect("optimization failed");

assert!((x_min[0] - 1.0).abs() < 1e-5);
assert!((x_min[1] - 1.0).abs() < 1e-5);
assert!(final_value.is_finite());
assert!(grad_norm < 1e-5);
assert!(iterations > 0);
```

For cost-only objectives, wrap a `ZerothOrderObjective` with `FiniteDiffGradient`.

### Example: Second-Order Optimization

Use `SecondOrderProblem` with `optimize` for automatic solver selection, or construct `NewtonTrustRegion` and `Arc` directly when you want explicit control over the algorithm choice.

## Testing

Run the crate tests from the repository root:

```bash
cargo test -p opt
```

The crate also includes comparison tests against SciPy through `opt/optimization_harness.py`.

## License

Licensed under either of:
- Apache License, Version 2.0
- MIT license

at your option.
