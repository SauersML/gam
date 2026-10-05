//! A bounded native-space query translation candidate. This module does not replace
//! upstream states, discover semantic roles, or make whole-model claims.
use crate::{
    operator_program::{Rotary, Scale},
    tiled_attention,
};
use gam_linalg::faer_ndarray::{FaerArrayView, FaerLlt};
use ndarray::{Array1, Array2};
use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SolverMethod {
    #[default]
    Lbfgs,
    DampedNewton,
}
fn default_newton_damping() -> f64 {
    1e-8
}

/// Rows are `scale * R_query^T R_key[j] key[j]`. No rotary implementation is duplicated.
pub fn native_features(
    keys: &Array2<f64>,
    positions: &[u32],
    query_position: u32,
    scale: Scale,
    rotary: Option<Rotary>,
) -> Result<Array2<f64>, String> {
    if keys.nrows() == 0
        || keys.ncols() == 0
        || positions.len() != keys.nrows()
        || keys.iter().any(|v| !v.is_finite())
    {
        return Err("invalid native keys or positions".into());
    }
    if let Some(r) = rotary {
        if r.base == 0 || r.dims == 0 || r.dims % 2 != 0 || r.dims as usize > keys.ncols() {
            return Err("invalid native rotary declaration".into());
        }
    }
    if !scale.value().is_finite() {
        return Err("invalid attention scale".into());
    }
    let rotated = tiled_attention::rotate(keys, rotary, positions, false);
    let positions = vec![query_position; keys.nrows()];
    Ok(tiled_attention::rotate(&rotated, rotary, &positions, true).into_owned() * scale.value())
}

#[derive(Clone)]
pub struct Sample {
    pub features: Array2<f64>,
    pub previous_logits: Array1<f64>,
    pub target: Array1<f64>,
}
impl Sample {
    pub fn new(
        features: Array2<f64>,
        previous_query: &[f64],
        target: Vec<f64>,
    ) -> Result<Self, String> {
        if features.ncols() == 0
            || features.ncols() != previous_query.len()
            || features.nrows() != target.len()
            || target.is_empty()
            || target.iter().any(|p| !p.is_finite() || *p < 0.)
            || (target.iter().sum::<f64>() - 1.).abs() > 1e-10
            || previous_query
                .iter()
                .chain(features.iter())
                .any(|v| !v.is_finite())
        {
            return Err("invalid query-transition sample".into());
        }
        let previous_logits = features.dot(&Array1::from_vec(previous_query.to_vec()));
        Ok(Self {
            features,
            previous_logits,
            target: Array1::from_vec(target),
        })
    }
    pub fn log_probabilities(&self, delta: &[f64]) -> Array1<f64> {
        let logits = &self.previous_logits + &self.features.dot(&Array1::from_vec(delta.to_vec()));
        let max = logits.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let logz = max + logits.iter().map(|v| (v - max).exp()).sum::<f64>().ln();
        logits - logz
    }
    pub fn kl(&self, delta: &[f64]) -> f64 {
        let logp = self.log_probabilities(delta);
        self.target
            .iter()
            .zip(logp.iter())
            .filter(|(t, _)| **t > 0.)
            .map(|(t, p)| t * (t.ln() - p))
            .sum::<f64>()
            .max(0.)
    }
}

#[derive(Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SolverSettings {
    #[serde(default)]
    pub method: SolverMethod,
    #[serde(default = "default_newton_damping")]
    pub newton_damping: f64,
    pub ridge: f64,
    pub max_iterations: usize,
    pub gradient_tolerance: f64,
    pub history: usize,
    pub max_backtracks: usize,
    pub armijo: f64,
}
impl Default for SolverSettings {
    fn default() -> Self {
        Self {
            method: SolverMethod::Lbfgs,
            newton_damping: default_newton_damping(),
            ridge: 1e-6,
            max_iterations: 300,
            gradient_tolerance: 1e-8,
            history: 10,
            max_backtracks: 40,
            armijo: 1e-4,
        }
    }
}
#[derive(Serialize)]
pub struct Fit {
    pub delta: Vec<f64>,
    pub objective: f64,
    pub mean_kl: f64,
    pub ridge_penalty: f64,
    pub gradient_norm: f64,
    pub iterations: usize,
    pub objective_evaluations: usize,
    pub stop_reason: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub newton_diagnostics: Option<NewtonDiagnostics>,
}
#[derive(Serialize)]
pub struct NewtonDiagnostics {
    pub hessian_evaluations: usize,
    pub linear_solves: usize,
    pub damping_increases: usize,
    pub last_damping: f64,
    pub maximum_relative_linear_solve_residual: f64,
    pub accepted_objectives: Vec<f64>,
    /// Analytic strong-convexity bound evaluated using the floating-point gradient;
    /// not an interval-arithmetic optimality certificate.
    pub ridge_gradient_objective_gap_bound: f64,
}
fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}
fn objective(samples: &[&Sample], delta: &[f64], ridge: f64) -> (f64, Vec<f64>) {
    let mut value = 0.;
    let mut gradient = Array1::<f64>::zeros(delta.len());
    for sample in samples {
        let logp = sample.log_probabilities(delta);
        value += sample
            .target
            .iter()
            .zip(logp.iter())
            .filter(|(t, _)| **t > 0.)
            .map(|(t, p)| t * (t.ln() - p))
            .sum::<f64>();
        let residual = logp.mapv(f64::exp) - &sample.target;
        gradient += &sample.features.t().dot(&residual);
    }
    let count = samples.len() as f64;
    value = value / count + 0.5 * ridge * dot(delta, delta);
    let gradient = gradient
        .iter()
        .zip(delta)
        .map(|(g, d)| g / count + ridge * d)
        .collect();
    (value, gradient)
}

/// Exact smooth Hessian: mean F^T (diag(p)-p p^T) F + ridge I.
/// Centering features before the outer product avoids cancellation of raw moments.
fn hessian(samples: &[&Sample], delta: &[f64], ridge: f64) -> Array2<f64> {
    let width = delta.len();
    let mut result = Array2::<f64>::zeros((width, width));
    for sample in samples {
        let p = sample.log_probabilities(delta).mapv(f64::exp);
        let mean = sample.features.t().dot(&p);
        let mut centered = sample.features.clone();
        for (mut row, probability) in centered.rows_mut().into_iter().zip(p.iter()) {
            for (value, mean) in row.iter_mut().zip(mean.iter()) {
                *value = (*value - mean) * probability.sqrt();
            }
        }
        result += &centered.t().dot(&centered);
    }
    result /= samples.len() as f64;
    for i in 0..width {
        result[(i, i)] += ridge;
    }
    result
}

fn newton_direction(
    hessian: &Array2<f64>,
    gradient: &[f64],
    damping: f64,
) -> Result<(Vec<f64>, f64), String> {
    let mut system = hessian.clone();
    for i in 0..gradient.len() {
        system[(i, i)] += damping;
    }
    if system.iter().any(|v| !v.is_finite()) {
        return Err("nonfinite Newton system".into());
    }
    let factor = FaerLlt::new(FaerArrayView::new(&system).as_ref(), faer::Side::Lower)
        .map_err(|e| format!("Newton SPD factorization: {e:?}"))?;
    let rhs = Array2::from_shape_fn((gradient.len(), 1), |(i, _)| -gradient[i]);
    let solved = factor.solve(FaerArrayView::new(&rhs).as_ref());
    let direction: Vec<f64> = (0..gradient.len()).map(|i| solved[(i, 0)]).collect();
    if direction.iter().any(|v| !v.is_finite()) {
        return Err("nonfinite Newton direction".into());
    }
    let applied = system.dot(&Array1::from_vec(direction.clone()));
    let residual = applied
        .iter()
        .zip(gradient)
        .map(|(a, g)| (a + g) * (a + g))
        .sum::<f64>()
        .sqrt();
    let relative = residual / dot(gradient, gradient).sqrt().max(f64::MIN_POSITIVE);
    if !relative.is_finite() || relative > 1e-6 {
        return Err(format!("Newton linear solve relative residual {relative}"));
    }
    Ok((direction, relative))
}

fn fit_newton(samples: &[&Sample], settings: &SolverSettings) -> Result<Fit, String> {
    let mut delta = vec![0.; samples[0].features.ncols()];
    let (mut value, mut gradient) = objective(samples, &delta, settings.ridge);
    let mut evaluations = 1;
    let mut diagnostics = NewtonDiagnostics {
        hessian_evaluations: 0,
        linear_solves: 0,
        damping_increases: 0,
        last_damping: settings.newton_damping,
        maximum_relative_linear_solve_residual: 0.,
        accepted_objectives: vec![value],
        ridge_gradient_objective_gap_bound: 0.,
    };
    let mut damping = settings.newton_damping;
    let mut iterations = 0;
    let mut stop = "iteration_limit";
    for iteration in 0..settings.max_iterations {
        if !value.is_finite() || gradient.iter().any(|v| !v.is_finite()) {
            return Err("nonfinite Newton state".into());
        }
        if dot(&gradient, &gradient).sqrt() <= settings.gradient_tolerance {
            stop = "gradient_tolerance";
            break;
        }
        let hessian = hessian(samples, &delta, settings.ridge);
        diagnostics.hessian_evaluations += 1;
        let mut accepted = None;
        // Damping is a direction safeguard, never part of the optimized objective.
        for _ in 0..12 {
            diagnostics.linear_solves += 1;
            if let Ok((direction, residual)) = newton_direction(&hessian, &gradient, damping) {
                diagnostics.maximum_relative_linear_solve_residual = diagnostics
                    .maximum_relative_linear_solve_residual
                    .max(residual);
                let slope = dot(&gradient, &direction);
                if slope.is_finite() && slope < 0. {
                    let mut step = 1.;
                    for _ in 0..settings.max_backtracks {
                        let candidate = delta
                            .iter()
                            .zip(&direction)
                            .map(|(d, p)| d + step * p)
                            .collect::<Vec<_>>();
                        let (next, grad) = objective(samples, &candidate, settings.ridge);
                        evaluations += 1;
                        if next.is_finite()
                            && grad.iter().all(|g| g.is_finite())
                            && next <= value + settings.armijo * step * slope
                        {
                            accepted = Some((candidate, next, grad, step));
                            break;
                        }
                        step *= 0.5;
                    }
                }
            }
            if accepted.is_some() {
                break;
            }
            damping = (damping * 10.).max(settings.ridge * 1e-6).max(1e-15);
            diagnostics.damping_increases += 1;
        }
        diagnostics.last_damping = damping;
        let Some((candidate, next, grad, step)) = accepted else {
            stop = "newton_direction_or_backtracking_limit";
            break;
        };
        delta = candidate;
        value = next;
        gradient = grad;
        iterations = iteration + 1;
        diagnostics.accepted_objectives.push(value);
        if step == 1. {
            damping *= 0.1;
        }
    }
    if !value.is_finite() || delta.iter().chain(gradient.iter()).any(|v| !v.is_finite()) {
        return Err("nonfinite final Newton state".into());
    }
    let gradient_norm = dot(&gradient, &gradient).sqrt();
    if gradient_norm <= settings.gradient_tolerance {
        stop = "gradient_tolerance";
    }
    diagnostics.ridge_gradient_objective_gap_bound =
        gradient_norm * gradient_norm / (2. * settings.ridge);
    let ridge_penalty = 0.5 * settings.ridge * dot(&delta, &delta);
    Ok(Fit {
        mean_kl: samples.iter().map(|s| s.kl(&delta)).sum::<f64>() / samples.len() as f64,
        delta,
        objective: value,
        ridge_penalty,
        gradient_norm,
        iterations,
        objective_evaluations: evaluations,
        stop_reason: stop.into(),
        newton_diagnostics: Some(diagnostics),
    })
}

/// Fixed-family optimization with true smooth-objective Armijo backtracking.
/// L-BFGS remains the default; exact damped Newton is an explicit option.
/// Finite termination is reported; no numerical optimality certificate is asserted.
pub fn fit(samples: &[&Sample], settings: &SolverSettings) -> Result<Fit, String> {
    if samples.is_empty()
        || !settings.newton_damping.is_finite()
        || settings.newton_damping < 0.
        || !settings.ridge.is_finite()
        || settings.ridge <= 0.
        || !settings.gradient_tolerance.is_finite()
        || settings.gradient_tolerance <= 0.
        || settings.history == 0
        || settings.max_iterations == 0
        || settings.max_backtracks == 0
        || !(0. ..0.5).contains(&settings.armijo)
        || settings.armijo == 0.
    {
        return Err("invalid solver settings or empty training fold".into());
    }
    let width = samples[0].features.ncols();
    if width == 0 || samples.iter().any(|s| s.features.ncols() != width) {
        return Err("query widths differ".into());
    }
    if matches!(settings.method, SolverMethod::DampedNewton) {
        return fit_newton(samples, settings);
    }
    let mut delta = vec![0.; width];
    let (mut value, mut gradient) = objective(samples, &delta, settings.ridge);
    let mut evaluations = 1;
    let mut history: Vec<(Vec<f64>, Vec<f64>, f64)> = vec![];
    let mut iterations = 0;
    let mut stop = "iteration_limit";
    for iteration in 0..settings.max_iterations {
        if !value.is_finite() || gradient.iter().any(|g| !g.is_finite()) {
            return Err("nonfinite solver state".into());
        }
        if dot(&gradient, &gradient).sqrt() <= settings.gradient_tolerance {
            stop = "gradient_tolerance";
            break;
        }
        let mut direction = gradient.clone();
        let mut alphas = Vec::with_capacity(history.len());
        for (s, y, rho) in history.iter().rev() {
            let alpha = rho * dot(s, &direction);
            for (q, y) in direction.iter_mut().zip(y) {
                *q -= alpha * y;
            }
            alphas.push(alpha);
        }
        let initial = history
            .last()
            .map(|(s, y, _)| dot(s, y) / dot(y, y))
            .unwrap_or(1.);
        for q in &mut direction {
            *q *= initial;
        }
        for ((s, y, rho), alpha) in history.iter().zip(alphas.iter().rev()) {
            let beta = rho * dot(y, &direction);
            for (q, s) in direction.iter_mut().zip(s) {
                *q += (alpha - beta) * s;
            }
        }
        for q in &mut direction {
            *q = -*q;
        }
        let mut slope = dot(&direction, &gradient);
        if !slope.is_finite() || slope >= 0. {
            direction = gradient.iter().map(|g| -g).collect();
            slope = -dot(&gradient, &gradient);
            history.clear();
        }
        let mut step = 1.;
        let mut accepted = None;
        for _ in 0..settings.max_backtracks {
            let candidate: Vec<_> = delta
                .iter()
                .zip(&direction)
                .map(|(d, p)| d + step * p)
                .collect();
            let (next, grad) = objective(samples, &candidate, settings.ridge);
            evaluations += 1;
            if next.is_finite() && next <= value + settings.armijo * step * slope {
                accepted = Some((candidate, next, grad));
                break;
            }
            step *= 0.5;
        }
        let Some((candidate, next, grad)) = accepted else {
            stop = "backtracking_limit";
            break;
        };
        let s: Vec<_> = candidate.iter().zip(&delta).map(|(a, b)| a - b).collect();
        let y: Vec<_> = grad.iter().zip(&gradient).map(|(a, b)| a - b).collect();
        let sy = dot(&s, &y);
        if sy > 1e-14 * dot(&s, &s).sqrt() * dot(&y, &y).sqrt() && sy.is_finite() {
            if history.len() == settings.history {
                history.remove(0);
            }
            history.push((s, y, 1. / sy));
        }
        delta = candidate;
        value = next;
        gradient = grad;
        iterations = iteration + 1;
    }
    if !value.is_finite()
        || gradient.iter().any(|g| !g.is_finite())
        || delta.iter().any(|d| !d.is_finite())
    {
        return Err("nonfinite final solver state".into());
    }
    if dot(&gradient, &gradient).sqrt() <= settings.gradient_tolerance {
        stop = "gradient_tolerance";
    }
    let ridge_penalty = 0.5 * settings.ridge * dot(&delta, &delta);
    Ok(Fit {
        mean_kl: samples.iter().map(|s| s.kl(&delta)).sum::<f64>() / samples.len() as f64,
        delta,
        objective: value,
        ridge_penalty,
        gradient_norm: dot(&gradient, &gradient).sqrt(),
        iterations,
        objective_evaluations: evaluations,
        stop_reason: stop.into(),
        newton_diagnostics: None,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;
    fn calibration() -> (Vec<Sample>, Vec<f64>) {
        let delta = vec![0.7, -0.4];
        let features = array![[1., 0.], [0., 1.], [-1., -1.], [0.5, -0.7]];
        let samples = [vec![-0.2, 0.5], vec![0.3, -0.1], vec![0.8, 0.6]]
            .iter()
            .map(|q| {
                let mut sample = Sample::new(features.clone(), q, vec![0.25; 4]).unwrap();
                sample.target = sample.log_probabilities(&delta).mapv(f64::exp);
                sample
            })
            .collect();
        (samples, delta)
    }
    #[test]
    fn smooth_gradient_matches_finite_difference() {
        let (samples, _) = calibration();
        let refs = samples.iter().collect::<Vec<_>>();
        let delta = vec![0.2, -0.6];
        let (_, gradient) = objective(&refs, &delta, 1e-3);
        for i in 0..2 {
            let mut a = delta.clone();
            let mut b = delta.clone();
            a[i] += 1e-6;
            b[i] -= 1e-6;
            let numerical = (objective(&refs, &a, 1e-3).0 - objective(&refs, &b, 1e-3).0) / 2e-6;
            assert!((gradient[i] - numerical).abs() < 1e-8);
        }
    }
    #[test]
    fn translated_query_calibration_and_oracle() {
        let (samples, delta) = calibration();
        let refs = samples.iter().collect::<Vec<_>>();
        assert!(samples.iter().all(|s| s.kl(&delta) < 1e-14));
        let settings = SolverSettings {
            ridge: 1e-10,
            gradient_tolerance: 1e-10,
            ..Default::default()
        };
        let result = fit(&refs, &settings).unwrap();
        assert!(result.mean_kl < 1e-12);
        assert!(result.gradient_norm < 1e-9);
        assert!(
            result
                .delta
                .iter()
                .zip(delta)
                .all(|(a, b)| (a - b).abs() < 1e-6)
        );
    }
    #[test]
    fn exact_hessian_vector_matches_gradient_difference() {
        let (samples, _) = calibration();
        let refs = samples.iter().collect::<Vec<_>>();
        let delta = vec![0.2, -0.6];
        let direction = vec![0.3, -0.8];
        let epsilon = 1e-5;
        let plus = delta
            .iter()
            .zip(&direction)
            .map(|(d, v)| d + epsilon * v)
            .collect::<Vec<_>>();
        let minus = delta
            .iter()
            .zip(&direction)
            .map(|(d, v)| d - epsilon * v)
            .collect::<Vec<_>>();
        let hp = hessian(&refs, &delta, 1e-3).dot(&Array1::from_vec(direction));
        let a = objective(&refs, &plus, 1e-3).1;
        let b = objective(&refs, &minus, 1e-3).1;
        for i in 0..delta.len() {
            assert!((hp[i] - (a[i] - b[i]) / (2. * epsilon)).abs() < 1e-9);
        }
    }
    #[test]
    fn newton_calibrates_ill_conditioned_realizable_translation() {
        // Feature curvature spans 12 orders of magnitude and includes an exactly
        // unobservable native coordinate. Only ridge constrains the final axis.
        let scales = [100., 1., 0.01, 0.001, 0.0001, 0.];
        let truth = vec![0.007, -0.4, 2., -3., 4., 0.];
        let mut samples = vec![];
        for context in 0..5 {
            let features = Array2::from_shape_fn((18, 6), |(j, d)| {
                scales[d] * (((j + 1) * (d + 1)) as f64 * 0.71 + context as f64 * 0.13).sin()
            });
            let query = vec![0.001, 0.2, -0.4, 3., -5., 0.];
            let mut sample = Sample::new(features, &query, vec![1. / 18.; 18]).unwrap();
            sample.target = sample.log_probabilities(&truth).mapv(f64::exp);
            samples.push(sample);
        }
        let refs = samples.iter().collect::<Vec<_>>();
        let settings = SolverSettings {
            method: SolverMethod::DampedNewton,
            ridge: 1e-14,
            max_iterations: 100,
            gradient_tolerance: 1e-12,
            ..Default::default()
        };
        let result = fit(&refs, &settings).unwrap();
        assert!(result.mean_kl < 1e-13, "KL {}", result.mean_kl);
        assert!(
            result.gradient_norm < 1e-10,
            "gradient {}",
            result.gradient_norm
        );
        assert!(
            result
                .delta
                .iter()
                .zip(&truth)
                .all(|(a, b)| (a - b).abs() < 0.002),
            "delta {:?}",
            result.delta
        );
        let diagnostics = result.newton_diagnostics.unwrap();
        assert!(
            diagnostics
                .accepted_objectives
                .windows(2)
                .all(|w| w[1] <= w[0])
        );
        assert!(diagnostics.maximum_relative_linear_solve_residual < 1e-6);
        assert!(diagnostics.ridge_gradient_objective_gap_bound < 1e-6);
    }
    #[test]
    fn feature_inverse_rotary_matches_native_attend() {
        use crate::operator_program::{
            Declarations, FamilyInputs, Node, OperatorProgram, SequenceLayout, Slot, SlotValues,
        };
        let keys = array![
            [0.1, 0.3, 0.6, -0.2],
            [-0.4, 0.2, -0.1, 0.7],
            [0.5, 0.4, -0.3, -0.6]
        ];
        let queries = array![[0., 0., 0., 0.], [0., 0., 0., 0.], [0.2, -0.5, 0.8, 0.1]];
        let positions = vec![1, 4, 8];
        let rotary = Some(Rotary {
            base: 10000,
            dims: 4,
            half_split: true,
        });
        let scale = Scale::InverseSqrt(4);
        let program = OperatorProgram {
            declarations: Declarations {
                domains: vec![],
                slots: vec![Slot::Raw { width: 4 }; 3],
                parameters: 0,
            },
            bases: vec![],
            operators: vec![],
            rules: vec![],
            nodes: vec![
                Node::Raw { slot: 0 },
                Node::Raw { slot: 1 },
                Node::Raw { slot: 2 },
                Node::Attend {
                    query: 0,
                    key: 1,
                    value: 2,
                    scale,
                    rotary,
                    causal: true,
                },
            ],
            output: 3,
        };
        let mut values = Array2::zeros((3, 4));
        for i in 0..3 {
            values[(i, i)] = 1.;
        }
        let inputs = FamilyInputs {
            rows: 3,
            slots: vec![
                SlotValues::Raw(queries.clone()),
                SlotValues::Raw(keys.clone()),
                SlotValues::Raw(values),
            ],
            layout: Some(SequenceLayout {
                sequence: vec![0; 3],
                position: positions.clone(),
            }),
        };
        let trace = program.execute(&inputs, false).unwrap();
        let features = native_features(&keys, &positions, 8, scale, rotary).unwrap();
        let sample = Sample::new(
            features,
            queries.row(2).as_slice().unwrap(),
            vec![1. / 3.; 3],
        )
        .unwrap();
        let p = sample.log_probabilities(&[0.; 4]).mapv(f64::exp);
        for i in 0..3 {
            assert!((p[i] - trace.values[3][(2, i)]).abs() < 1e-14);
        }
    }
}
