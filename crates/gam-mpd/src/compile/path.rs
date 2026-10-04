//! Requirements that hold downstream of norms and MLPs: edits at one or more linear sites
//! solved jointly through the exact forward.
//!
//! A linear-site requirement ([`super::linear`]) constrains what one stored matrix writes.
//! A requirement on what a whole path writes, through an RMSNorm, a gated MLP or a residual
//! stream, is nonlinear in the edits. This module declares the path as a small program of
//! native primitives ([`PathLayer`]), sets its output on declared inputs to declared targets
//! (set-type), and solves for global edits of the path's editable linear sites together.
//!
//! # The solve
//!
//! Gauss–Newton on the set-type residual `r(Δ) = T − F_{θ+Δ}(X)`: each step is the
//! minimum-norm solution of `J s = r` with `J` the exact Jacobian of the path's output with
//! respect to every editable site's weights at the current point. `J` is never formed: its
//! products `J v` (forward tangents) and `Jᵀ u` (reverse cotangents) are derived primitive by
//! primitive,
//!
//! ```text
//! linear      y = W x + b           ẏ = W ẋ + Ẇ x               x̄ = Wᵀ ȳ,  W̄ += ȳ xᵀ
//! RMSNorm     y = g ⊙ x / ρ(x)      ẏ = g ⊙ (ẋ/ρ − x (xᵀẋ)/(d ρ³))   x̄ = h/ρ − x (xᵀh)/(d ρ³), h = g ⊙ ȳ
//! activation  y = φ(z)              ẏ = φ′(z) ẋ                  x̄ = φ′(z) ȳ
//! SwiGLU      y = silu(W_g x) ⊙ W_u x    (product rule on both reads)
//! residual    y = x + f(x)          ẏ = ẋ + ḟ                    x̄ = ȳ + f̄
//! ```
//!
//! with `ρ(x) = √(ε + ‖x‖²/d)`, `silu′(z) = σ(z)(1 + z(1 − σ(z)))`, `gelu′(z) = Φ(z) + z φ(z)`,
//! and ReLU's derivative taken as `1{z > 0}`. The minimum-norm step is LSQR (Paige &
//! Saunders), which in exact arithmetic ends within `rank J ≤ min(parameters, outputs)`
//! iterations; it stops there, once its residual estimate falls to
//! `γ_p (‖J‖_F ‖s‖ + ‖r‖)` (a consistent system solved to rounding), or once the
//! normal-equation estimate `‖Jᵀ r‖` falls to `γ_p ‖J‖_F ‖r‖` (the least-squares optimum). An
//! early stop costs only another Gauss–Newton step. A step is accepted only if the executed
//! residual decreases; otherwise it is halved until it no longer moves any weight. The outer
//! loop runs until the residual certifies or the declared iteration budget ends.
//!
//! # Certificate
//!
//! Each editable site's accumulated edit is compressed to the singular values above its SVD
//! band and stored as factors (or stored exactly, one factor the identity, when the
//! compression's rounding would lose the certificate the uncompressed edit holds). The edited path is then executed with the stored factors,
//! `W + L Rᵀ`, and a first-order running error bound: each primitive propagates its input
//! band through `|∂y/∂x|` and adds its own rounding (`γ_{n+1}(|W||x| + |b|)` and the formation
//! of `W + L Rᵀ` for a linear read, `γ_{d+4}|y|` for a norm, the special function's proven
//! relative error for an activation). A requirement is certified when every entry of the
//! executed residual lies within its band plus the declared target radius. The control is
//! then [`ControlRealization::ExactlyRealized`] on the declared inputs (an exhaustive status
//! over them); otherwise [`ControlRealization::EmpiricallyValidated`] with the residual it
//! reached. Off-target inputs, when declared, report `sup ‖F_{θ+Δ}(x) − F_θ(x)‖` over them.

use gam_linalg::roundoff::{UNIT_ROUNDOFF, accumulation_growth, factor_singular_band};
use gam_math::probability::{NORMAL_CDF_RELATIVE_ERROR, NORMAL_CDF_UNDERFLOW_FLOOR, normal_cdf_and_pdf};
use gam_math::roundoff::inflated;
use gam_math::special::logistic;
use ndarray::{Array1, Array2, ArrayView1, ArrayView2, s};

use super::super::apply::FactoredEdit;
use gam_linalg::decompose::svd;
use super::super::lift::{TensorId, TensorRegistry};
use super::super::supports::{EvidenceStatus, ExactBasis};
use super::{
    CompileError, CompiledControl, CompiledParameterEdit, ControlRealization, NativeEditPlan, require_finite,
    require_shape,
};

/// A linear read of the path: `y = W x + b` with a stored `W` (`out × in`).
#[derive(Clone, Debug)]
pub struct PathSite<'a> {
    pub storage: TensorId,
    pub weight: ArrayView2<'a, f64>,
    pub bias: Option<ArrayView1<'a, f64>>,
    /// Whether the solve may edit this site.
    pub editable: bool,
}

/// An elementwise activation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PathActivation {
    Silu,
    ExactGelu,
    Relu,
}

/// One primitive of a path; sites are indices into [`PathProblem::sites`], so a site read
/// twice (a shared body) is one tensor and its edit moves both reads.
#[derive(Clone, Debug)]
pub enum PathLayer {
    Linear { site: usize },
    RmsNorm { gain: Option<Vec<f64>>, epsilon: f64 },
    Activation(PathActivation),
    /// `silu(W_gate x) ⊙ (W_up x)`.
    Swiglu { gate: usize, up: usize },
    /// `x + f(x)`.
    Residual(Vec<PathLayer>),
}

/// A set-type requirement on a path's output.
#[derive(Clone, Debug)]
pub struct PathProblem<'a> {
    pub registry: &'a TensorRegistry,
    pub sites: Vec<PathSite<'a>>,
    pub layers: Vec<PathLayer>,
    /// `n × input width`, rows are observations.
    pub inputs: ArrayView2<'a, f64>,
    /// What the path must write on them after the edit (`n × output width`).
    pub targets: ArrayView2<'a, f64>,
    pub target_radius: f64,
    /// Inputs the edit should leave alone; their damage is reported.
    pub off_target: Option<ArrayView2<'a, f64>>,
    /// Declared Gauss–Newton budget.
    pub max_iterations: usize,
}

/// The family a path residual is stated over: the declared inputs.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PathFamily {
    pub inputs: usize,
}

/// Where the largest residual or damage sits: observation and output coordinate.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PathWitness {
    pub observation: usize,
    pub coordinate: usize,
}

/// What [`compile_path`] found.
#[derive(Clone, Debug)]
pub struct PathReport {
    pub compiled: CompiledControl<PathWitness, PathFamily>,
    pub iterations: usize,
    /// The executed residual `F_{θ+Δ}(X) − T` of the stored edit and its entrywise band
    /// (radius included).
    pub residual: Array2<f64>,
    pub residual_band: Array2<f64>,
    pub off_target_damage: Option<EvidenceStatus<PathWitness, PathFamily>>,
}

/// The weights one forward reads, with the entrywise band of their formation.
struct Weights {
    values: Vec<Array2<f64>>,
    bands: Vec<Array2<f64>>,
    biases: Vec<Option<Array1<f64>>>,
}

enum Cache {
    Linear { input: Array1<f64> },
    RmsNorm { input: Array1<f64>, rms: f64 },
    Activation { pre: Array1<f64> },
    Swiglu { input: Array1<f64>, gate: Array1<f64>, up: Array1<f64> },
    Residual { inner: Vec<Cache> },
}

fn activation(kind: PathActivation, z: f64) -> (f64, f64, f64) {
    // (value, derivative, relative evaluation error of the value)
    match kind {
        PathActivation::Silu => {
            let sigma = logistic(z);
            (z * sigma, sigma * (1.0 + z * (1.0 - sigma)), 6.0 * UNIT_ROUNDOFF)
        }
        PathActivation::ExactGelu => {
            let (cdf, pdf) = normal_cdf_and_pdf(z);
            (z * cdf, cdf + z * pdf, NORMAL_CDF_RELATIVE_ERROR + 2.0 * UNIT_ROUNDOFF)
        }
        PathActivation::Relu => {
            if z > 0.0 {
                (z, 1.0, 0.0)
            } else {
                (0.0, 0.0, 0.0)
            }
        }
    }
}

fn rms(x: &Array1<f64>, epsilon: f64) -> f64 {
    (epsilon + x.dot(x) / x.len() as f64).sqrt()
}

/// The output width of `layers` on inputs of `width`, refusing a shape the sites do not fit.
fn check_layers(layers: &[PathLayer], sites: &[PathSite<'_>], width: usize) -> Result<usize, CompileError> {
    let mut current = width;
    for layer in layers {
        current = match layer {
            PathLayer::Linear { site } => {
                let site = sites.get(*site).ok_or(CompileError::InvalidDeclaration {
                    what: "path site",
                    reason: format!("site index {site} is out of range"),
                })?;
                require_shape("path linear input", (site.weight.ncols(), 1), (current, 1))?;
                site.weight.nrows()
            }
            PathLayer::RmsNorm { gain, epsilon } => {
                if !(epsilon.is_finite() && *epsilon > 0.0) {
                    return Err(CompileError::InvalidDeclaration {
                        what: "RMSNorm epsilon",
                        reason: format!("must be positive and finite; got {epsilon}"),
                    });
                }
                if let Some(gain) = gain {
                    require_shape("RMSNorm gain", (current, 1), (gain.len(), 1))?;
                    require_finite("RMSNorm gain", gain.iter().copied())?;
                }
                current
            }
            PathLayer::Activation(_) => current,
            PathLayer::Swiglu { gate, up } => {
                let (Some(gate), Some(up)) = (sites.get(*gate), sites.get(*up)) else {
                    return Err(CompileError::InvalidDeclaration {
                        what: "path site",
                        reason: "a SwiGLU site index is out of range".to_string(),
                    });
                };
                require_shape("SwiGLU gate input", (gate.weight.ncols(), 1), (current, 1))?;
                require_shape("SwiGLU up", gate.weight.dim(), up.weight.dim())?;
                gate.weight.nrows()
            }
            PathLayer::Residual(inner) => {
                let out = check_layers(inner, sites, current)?;
                require_shape("residual branch", (current, 1), (out, 1))?;
                current
            }
        };
    }
    Ok(current)
}

fn linear_value(weights: &Weights, site: usize, x: &Array1<f64>) -> Array1<f64> {
    let mut y = weights.values[site].dot(x);
    if let Some(bias) = &weights.biases[site] {
        y += bias;
    }
    y
}

/// The forward of one row, with its caches.
fn forward(layers: &[PathLayer], weights: &Weights, x: Array1<f64>) -> (Array1<f64>, Vec<Cache>) {
    let mut caches = Vec::with_capacity(layers.len());
    let mut current = x;
    for layer in layers {
        current = match layer {
            PathLayer::Linear { site } => {
                let y = linear_value(weights, *site, &current);
                caches.push(Cache::Linear { input: current });
                y
            }
            PathLayer::RmsNorm { gain, epsilon } => {
                let r = rms(&current, *epsilon);
                let mut y = current.mapv(|value| value / r);
                if let Some(gain) = gain {
                    y.iter_mut().zip(gain).for_each(|(value, g)| *value *= g);
                }
                caches.push(Cache::RmsNorm { input: current, rms: r });
                y
            }
            PathLayer::Activation(kind) => {
                let y = current.mapv(|z| activation(*kind, z).0);
                caches.push(Cache::Activation { pre: current });
                y
            }
            PathLayer::Swiglu { gate, up } => {
                let a = linear_value(weights, *gate, &current);
                let b = linear_value(weights, *up, &current);
                let y = Array1::from_shape_fn(a.len(), |i| activation(PathActivation::Silu, a[i]).0 * b[i]);
                caches.push(Cache::Swiglu { input: current, gate: a, up: b });
                y
            }
            PathLayer::Residual(inner) => {
                let (branch, inner_caches) = forward(inner, weights, current.clone());
                caches.push(Cache::Residual { inner: inner_caches });
                current + &branch
            }
        };
    }
    (current, caches)
}

/// The forward of one row with a first-order running error bound (module docs).
fn banded_forward(layers: &[PathLayer], weights: &Weights, x: Array1<f64>, band: Array1<f64>) -> (Array1<f64>, Array1<f64>) {
    let mut current = x;
    let mut error = band;
    let linear = |site: usize, x: &Array1<f64>, e: &Array1<f64>| {
        let w = &weights.values[site];
        let terms = w.ncols() + 1;
        let y = linear_value(weights, site, x);
        let magnitude = w.mapv(f64::abs).dot(&x.mapv(f64::abs))
            + weights.biases[site].as_ref().map_or_else(|| Array1::zeros(y.len()), |b| b.mapv(f64::abs));
        let e_y = magnitude * accumulation_growth(terms) + w.mapv(f64::abs).dot(e) + weights.bands[site].dot(&x.mapv(f64::abs));
        (y, e_y)
    };
    for layer in layers {
        let (y, e_y) = match layer {
            PathLayer::Linear { site } => linear(*site, &current, &error),
            PathLayer::RmsNorm { gain, epsilon } => {
                let d = current.len() as f64;
                let r = rms(&current, *epsilon);
                let absolute = current.mapv(f64::abs);
                let coupling = absolute.dot(&error) / (d * r * r * r);
                let mut y = current.mapv(|value| value / r);
                let mut e = Array1::from_shape_fn(current.len(), |i| error[i] / r + absolute[i] * coupling);
                if let Some(gain) = gain {
                    for i in 0..y.len() {
                        y[i] *= gain[i];
                        e[i] *= gain[i].abs();
                    }
                }
                let rounding = accumulation_growth(current.len() + 4);
                let e = &e + &y.mapv(|value| rounding * value.abs());
                (y, e)
            }
            PathLayer::Activation(kind) => {
                let mut y = Array1::zeros(current.len());
                let mut e = Array1::zeros(current.len());
                for i in 0..current.len() {
                    let (value, derivative, relative) = activation(*kind, current[i]);
                    y[i] = value;
                    e[i] = derivative.abs() * error[i] + relative * value.abs() + NORMAL_CDF_UNDERFLOW_FLOOR;
                }
                (y, e)
            }
            PathLayer::Swiglu { gate, up } => {
                let (a, e_a) = linear(*gate, &current, &error);
                let (b, e_b) = linear(*up, &current, &error);
                let mut y = Array1::zeros(a.len());
                let mut e = Array1::zeros(a.len());
                for i in 0..a.len() {
                    let (value, derivative, relative) = activation(PathActivation::Silu, a[i]);
                    y[i] = value * b[i];
                    e[i] = derivative.abs() * e_a[i] * b[i].abs()
                        + value.abs() * e_b[i]
                        + 1.1 * e_a[i] * e_b[i]
                        + (relative + UNIT_ROUNDOFF) * y[i].abs();
                }
                (y, e)
            }
            PathLayer::Residual(inner) => {
                let (branch, e_branch) = banded_forward(inner, weights, current.clone(), error.clone());
                let y = &current + &branch;
                let e = &error + &e_branch + &y.mapv(|value| UNIT_ROUNDOFF * value.abs());
                (y, e)
            }
        };
        current = y;
        error = e_y.mapv(|value| inflated(value, 1));
    }
    (current, error)
}

/// `J v` for one row: the output tangent of weight tangents `tangents` (editable sites).
fn tangent(layers: &[PathLayer], weights: &Weights, caches: &[Cache], dx: Array1<f64>, tangents: &[Option<Array2<f64>>]) -> Array1<f64> {
    let mut current = dx;
    for (layer, cache) in layers.iter().zip(caches) {
        current = match (layer, cache) {
            (PathLayer::Linear { site }, Cache::Linear { input }) => {
                let mut y = weights.values[*site].dot(&current);
                if let Some(t) = &tangents[*site] {
                    y += &t.dot(input);
                }
                y
            }
            (PathLayer::RmsNorm { gain, .. }, Cache::RmsNorm { input, rms }) => {
                let d = input.len() as f64;
                let projection = input.dot(&current) / (d * rms * rms * rms);
                let mut y = Array1::from_shape_fn(input.len(), |i| current[i] / rms - input[i] * projection);
                if let Some(gain) = gain {
                    y.iter_mut().zip(gain).for_each(|(value, g)| *value *= g);
                }
                y
            }
            (PathLayer::Activation(kind), Cache::Activation { pre }) => {
                Array1::from_shape_fn(pre.len(), |i| activation(*kind, pre[i]).1 * current[i])
            }
            (PathLayer::Swiglu { gate, up }, Cache::Swiglu { input, gate: a, up: b }) => {
                let mut da = weights.values[*gate].dot(&current);
                if let Some(t) = &tangents[*gate] {
                    da += &t.dot(input);
                }
                let mut db = weights.values[*up].dot(&current);
                if let Some(t) = &tangents[*up] {
                    db += &t.dot(input);
                }
                Array1::from_shape_fn(a.len(), |i| {
                    let (value, derivative, _) = activation(PathActivation::Silu, a[i]);
                    derivative * da[i] * b[i] + value * db[i]
                })
            }
            (PathLayer::Residual(inner), Cache::Residual { inner: inner_caches }) => {
                let branch = tangent(inner, weights, inner_caches, current.clone(), tangents);
                current + &branch
            }
            _ => current,
        };
    }
    current
}

/// `Jᵀ u` for one row: accumulates the weight cotangents of editable sites into `grads` and
/// returns the input cotangent.
fn cotangent(layers: &[PathLayer], weights: &Weights, caches: &[Cache], gy: Array1<f64>, grads: &mut [Option<Array2<f64>>]) -> Array1<f64> {
    let outer = |g: &Array1<f64>, x: &Array1<f64>| {
        Array2::from_shape_fn((g.len(), x.len()), |(i, j)| g[i] * x[j])
    };
    let mut current = gy;
    for (layer, cache) in layers.iter().zip(caches).rev() {
        current = match (layer, cache) {
            (PathLayer::Linear { site }, Cache::Linear { input }) => {
                if let Some(grad) = &mut grads[*site] {
                    *grad += &outer(&current, input);
                }
                weights.values[*site].t().dot(&current)
            }
            (PathLayer::RmsNorm { gain, .. }, Cache::RmsNorm { input, rms }) => {
                let mut h = current.clone();
                if let Some(gain) = gain {
                    h.iter_mut().zip(gain).for_each(|(value, g)| *value *= g);
                }
                let d = input.len() as f64;
                let projection = input.dot(&h) / (d * rms * rms * rms);
                Array1::from_shape_fn(input.len(), |i| h[i] / rms - input[i] * projection)
            }
            (PathLayer::Activation(kind), Cache::Activation { pre }) => {
                Array1::from_shape_fn(pre.len(), |i| activation(*kind, pre[i]).1 * current[i])
            }
            (PathLayer::Swiglu { gate, up }, Cache::Swiglu { input, gate: a, up: b }) => {
                let ga = Array1::from_shape_fn(a.len(), |i| activation(PathActivation::Silu, a[i]).1 * b[i] * current[i]);
                let gb = Array1::from_shape_fn(a.len(), |i| activation(PathActivation::Silu, a[i]).0 * current[i]);
                if let Some(grad) = &mut grads[*gate] {
                    *grad += &outer(&ga, input);
                }
                if let Some(grad) = &mut grads[*up] {
                    *grad += &outer(&gb, input);
                }
                weights.values[*gate].t().dot(&ga) + weights.values[*up].t().dot(&gb)
            }
            (PathLayer::Residual(inner), Cache::Residual { inner: inner_caches }) => {
                let branch = cotangent(inner, weights, inner_caches, current.clone(), grads);
                current + &branch
            }
            _ => current,
        };
    }
    current
}

type Parameters = Vec<Option<Array2<f64>>>;

fn parameter_dot(a: &Parameters, b: &Parameters) -> f64 {
    a.iter()
        .zip(b)
        .map(|pair| match pair {
            (Some(x), Some(y)) => (x * y).sum(),
            _ => 0.0,
        })
        .sum()
}

fn parameter_combine(a: &Parameters, alpha: f64, b: &Parameters, beta: f64) -> Parameters {
    a.iter()
        .zip(b)
        .map(|pair| match pair {
            (Some(x), Some(y)) => Some(x * alpha + y * beta),
            _ => None,
        })
        .collect()
}

/// The Jacobian of the path's outputs at one point, as products.
struct Linearization<'w> {
    layers: &'w [PathLayer],
    weights: &'w Weights,
    caches: Vec<Vec<Cache>>,
    shapes: Vec<Option<(usize, usize)>>,
    output_width: usize,
    input_width: usize,
}

impl Linearization<'_> {
    fn zeros(&self) -> Parameters {
        self.shapes.iter().map(|shape| shape.map(Array2::zeros)).collect()
    }

    fn apply(&self, v: &Parameters) -> Array2<f64> {
        let mut out = Array2::zeros((self.caches.len(), self.output_width));
        for (row, caches) in self.caches.iter().enumerate() {
            let y = tangent(self.layers, self.weights, caches, Array1::zeros(self.input_width), v);
            out.row_mut(row).assign(&y);
        }
        out
    }

    fn adjoint(&self, u: &Array2<f64>) -> Parameters {
        let mut grads = self.zeros();
        for (row, caches) in self.caches.iter().enumerate() {
            cotangent(self.layers, self.weights, caches, u.row(row).to_owned(), &mut grads);
        }
        grads
    }
}

/// LSQR for the minimum-norm solution of `J s = b` (module docs).
fn lsqr(linearization: &Linearization<'_>, b: &Array2<f64>) -> Parameters {
    let norm2 = |m: &Array2<f64>| m.iter().map(|v| v * v).sum::<f64>().sqrt();
    let unknowns: usize = linearization.shapes.iter().flatten().map(|(r, c)| r * c).sum();
    let limit = unknowns.min(b.len()).max(1);
    let mut x = linearization.zeros();
    let mut beta = norm2(b);
    if beta == 0.0 {
        return x;
    }
    let mut u = b / beta;
    let mut v = linearization.adjoint(&u);
    let mut alpha = parameter_dot(&v, &v).sqrt();
    if alpha == 0.0 {
        return x;
    }
    v = parameter_combine(&v, 1.0 / alpha, &v, 0.0);
    let mut w = v.clone();
    let (mut phibar, mut rhobar) = (beta, alpha);
    let mut operator_norm_squared = alpha * alpha;
    for _ in 0..limit {
        let next = linearization.apply(&v) - &u * alpha;
        beta = norm2(&next);
        if beta > 0.0 {
            u = next / beta;
        }
        let next_v = parameter_combine(&linearization.adjoint(&u), 1.0, &v, -beta);
        alpha = parameter_dot(&next_v, &next_v).sqrt();
        if alpha > 0.0 {
            v = parameter_combine(&next_v, 1.0 / alpha, &next_v, 0.0);
        }
        operator_norm_squared += alpha * alpha + beta * beta;
        let rho = (rhobar * rhobar + beta * beta).sqrt();
        let (c, sn) = (rhobar / rho, beta / rho);
        let theta = sn * alpha;
        rhobar = -c * alpha;
        let phi = c * phibar;
        phibar *= sn;
        x = parameter_combine(&x, 1.0, &w, phi / rho);
        w = parameter_combine(&v, 1.0, &w, -theta / rho);
        let normal = phibar * alpha * c.abs();
        let solution_norm = parameter_dot(&x, &x).sqrt();
        let rounding = accumulation_growth(unknowns);
        let consistent = phibar <= rounding * (operator_norm_squared.sqrt() * solution_norm + norm2(b));
        let optimal = normal <= rounding * operator_norm_squared.sqrt() * phibar;
        if consistent || optimal || alpha == 0.0 || beta == 0.0 {
            break;
        }
    }
    x
}

/// The weights `W + Δ` a forward reads, with the band of forming them: one rounding of the
/// sum, plus `formation` (the band of `Δ = L Rᵀ` formed from stored factors) when given.
fn weights_at(problem: &PathProblem<'_>, deltas: &Parameters, formation: Option<&Parameters>) -> Weights {
    let mut values = Vec::with_capacity(problem.sites.len());
    let mut bands = Vec::with_capacity(problem.sites.len());
    for (index, (site, delta)) in problem.sites.iter().zip(deltas).enumerate() {
        match delta {
            Some(delta) => {
                values.push(&site.weight + delta);
                let mut band = (site.weight.mapv(f64::abs) + delta.mapv(f64::abs)) * UNIT_ROUNDOFF;
                if let Some(Some(extra)) = formation.map(|bands| &bands[index]) {
                    band += extra;
                }
                bands.push(band);
            }
            None => {
                values.push(site.weight.to_owned());
                bands.push(Array2::zeros(site.weight.dim()));
            }
        }
    }
    Weights {
        values,
        bands,
        biases: problem.sites.iter().map(|site| site.bias.map(|b| b.to_owned())).collect(),
    }
}

fn outputs(problem: &PathProblem<'_>, weights: &Weights, inputs: ArrayView2<'_, f64>, width: usize) -> Array2<f64> {
    let mut out = Array2::zeros((inputs.nrows(), width));
    for (row, x) in inputs.rows().into_iter().enumerate() {
        out.row_mut(row).assign(&forward(&problem.layers, weights, x.to_owned()).0);
    }
    out
}

fn residual_norm(problem: &PathProblem<'_>, weights: &Weights, width: usize) -> f64 {
    let out = outputs(problem, weights, problem.inputs, width);
    (&out - &problem.targets).iter().map(|v| v * v).sum::<f64>().sqrt()
}

/// Solves the set-type requirement on the path for edits of its editable sites, certifies
/// the stored edit by executing the edited path, and reports its status.
pub fn compile_path(problem: &PathProblem<'_>, control: &str) -> Result<PathReport, CompileError> {
    let n = problem.inputs.nrows();
    let input_width = problem.inputs.ncols();
    let width = check_layers(&problem.layers, &problem.sites, input_width)?;
    require_shape("path targets", (n, width), problem.targets.dim())?;
    require_finite("path inputs", problem.inputs.iter().copied())?;
    require_finite("path targets", problem.targets.iter().copied())?;
    if !(problem.target_radius.is_finite() && problem.target_radius >= 0.0) {
        return Err(CompileError::InvalidDeclaration {
            what: "target radius",
            reason: format!("must be finite and nonnegative; got {}", problem.target_radius),
        });
    }
    if n == 0 {
        return Err(CompileError::InvalidDeclaration {
            what: "path inputs",
            reason: "a path requirement needs at least one input".to_string(),
        });
    }
    for site in &problem.sites {
        require_finite("path weight", site.weight.iter().copied())?;
        if let Some(bias) = site.bias {
            require_shape("path bias", (site.weight.nrows(), 1), (bias.len(), 1))?;
        }
    }
    let shapes: Vec<Option<(usize, usize)>> =
        problem.sites.iter().map(|site| site.editable.then(|| site.weight.dim())).collect();
    let mut deltas: Parameters = shapes.iter().map(|shape| shape.map(Array2::zeros)).collect();
    let mut iterations = 0;
    let certify = |deltas: &Parameters, formation: Option<&Parameters>| -> Result<(Array2<f64>, Array2<f64>, bool, Weights), CompileError> {
        let weights = weights_at(problem, deltas, formation);
        let mut residual = Array2::zeros((n, width));
        let mut band = Array2::zeros((n, width));
        for (row, x) in problem.inputs.rows().into_iter().enumerate() {
            let (y, e) = banded_forward(&problem.layers, &weights, x.to_owned(), Array1::zeros(input_width));
            for j in 0..width {
                let value = y[j] - problem.targets[[row, j]];
                residual[[row, j]] = value;
                band[[row, j]] = inflated(e[j] + problem.target_radius + UNIT_ROUNDOFF * value.abs(), 1);
            }
        }
        let certified = residual.iter().zip(band.iter()).all(|(r, b)| r.abs() <= *b);
        Ok((residual, band, certified, weights))
    };
    let (_, _, mut certified, _) = certify(&deltas, None)?;
    while !certified && iterations < problem.max_iterations {
        iterations += 1;
        let weights = weights_at(problem, &deltas, None);
        let mut caches = Vec::with_capacity(n);
        let mut current = Array2::zeros((n, width));
        for (row, x) in problem.inputs.rows().into_iter().enumerate() {
            let (y, cache) = forward(&problem.layers, &weights, x.to_owned());
            current.row_mut(row).assign(&y);
            caches.push(cache);
        }
        let linearization = Linearization {
            layers: &problem.layers,
            weights: &weights,
            caches,
            shapes: shapes.clone(),
            output_width: width,
            input_width,
        };
        let demand = &problem.targets - &current;
        let step = lsqr(&linearization, &demand);
        let start = residual_norm(problem, &weights, width);
        let mut scale = 1.0;
        let mut accepted = None;
        loop {
            let trial = parameter_combine(&deltas, 1.0, &step, scale);
            if trial.iter().zip(&deltas).all(|pair| match pair {
                (Some(a), Some(b)) => a == b,
                _ => true,
            }) {
                break;
            }
            if residual_norm(problem, &weights_at(problem, &trial, None), width) < start {
                accepted = Some(trial);
                break;
            }
            scale *= 0.5;
        }
        let Some(trial) = accepted else {
            break;
        };
        deltas = trial;
        certified = certify(&deltas, None)?.2;
    }
    // Store each edit compressed to its resolved rank; if the compression's own rounding
    // loses a certificate the uncompressed edit holds, store it exactly instead (one factor
    // the identity, so `L Rᵀ` reproduces `Δ` bit for bit). Either way the stored factors are
    // what is certified.
    let factor_sets = |compress: bool| -> Result<(Vec<CompiledParameterEdit>, Parameters, Parameters), CompileError> {
        let mut edits = Vec::new();
        let mut stored: Parameters = shapes.iter().map(|_| None).collect();
        let mut formation: Parameters = shapes.iter().map(|_| None).collect();
        for (index, (site, delta)) in problem.sites.iter().zip(&deltas).enumerate() {
            let Some(delta) = delta else {
                continue;
            };
            if delta.iter().all(|value| *value == 0.0) {
                continue;
            }
            let (left, right) = if compress {
                let decomposed = svd(delta.view(), false)?;
                let sigma_max = decomposed.singular_values.first().copied().unwrap_or(0.0);
                let band = factor_singular_band(delta.nrows(), delta.ncols(), sigma_max);
                let rank = decomposed.singular_values.iter().filter(|&&value| value > band).count();
                if rank == 0 {
                    continue;
                }
                let mut left = decomposed.u.slice(s![.., ..rank]).to_owned();
                for (mut column, &value) in left.columns_mut().into_iter().zip(decomposed.singular_values.iter()) {
                    column.mapv_inplace(|entry| entry * value);
                }
                (left, decomposed.vt.slice(s![..rank, ..]).t().to_owned())
            } else if delta.nrows() <= delta.ncols() {
                (Array2::eye(delta.nrows()), delta.t().to_owned())
            } else {
                (delta.clone(), Array2::eye(delta.ncols()))
            };
            let terms = left.ncols();
            stored[index] = Some(left.dot(&right.t()));
            formation[index] = Some(left.mapv(f64::abs).dot(&right.mapv(f64::abs).t()) * accumulation_growth(terms));
            edits.push(CompiledParameterEdit {
                storage: site.storage.clone(),
                delta: FactoredEdit::new(left, right)?,
            });
        }
        Ok((edits, stored, formation))
    };
    let (mut edits, stored, formation) = factor_sets(true)?;
    let (mut residual, mut residual_band, mut certified, mut weights) = certify(&stored, Some(&formation))?;
    if !certified && certify(&deltas, None)?.2 {
        let (exact_edits, exact_stored, exact_formation) = factor_sets(false)?;
        edits = exact_edits;
        (residual, residual_band, certified, weights) = certify(&exact_stored, Some(&exact_formation))?;
    }
    let family = PathFamily { inputs: n };
    let (mut worst, mut value, mut error) = (PathWitness { observation: 0, coordinate: 0 }, 0.0_f64, 0.0_f64);
    for ((row, col), &r) in residual.indexed_iter() {
        if r.abs() > value {
            value = r.abs();
            worst = PathWitness {
                observation: row,
                coordinate: col,
            };
        }
        error = error.max(residual_band[[row, col]]);
    }
    let status = EvidenceStatus::exact(
        value,
        error,
        ExactBasis::Exhaustive { cardinality: n as u64 },
        Some(worst),
        family,
    )?;
    let realization = if certified {
        ControlRealization::exactly_realized_on_family(status)?
    } else {
        ControlRealization::empirically_validated(status)?
    };
    let off_target_damage = match problem.off_target {
        None => None,
        Some(off) if off.nrows() == 0 => None,
        Some(off) => {
            require_shape("off-target inputs", (off.nrows(), input_width), off.dim())?;
            require_finite("off-target inputs", off.iter().copied())?;
            let native = weights_at(problem, &shapes.iter().map(|_| None).collect::<Parameters>(), None);
            let (mut worst, mut value, mut error) = (PathWitness { observation: 0, coordinate: 0 }, 0.0_f64, 0.0_f64);
            for (row, x) in off.rows().into_iter().enumerate() {
                let (before, e_before) = banded_forward(&problem.layers, &native, x.to_owned(), Array1::zeros(input_width));
                let (after, e_after) = banded_forward(&problem.layers, &weights, x.to_owned(), Array1::zeros(input_width));
                let difference = &after - &before;
                let damage = difference.dot(&difference).sqrt();
                let band = (&e_before + &e_after).dot(&(&e_before + &e_after)).sqrt();
                if damage > value {
                    let coordinate = (0..width).fold(0, |best, j| if difference[j].abs() > difference[best].abs() { j } else { best });
                    worst = PathWitness { observation: row, coordinate };
                    value = damage;
                }
                error = error.max(inflated(band + UNIT_ROUNDOFF * damage, 1));
            }
            Some(EvidenceStatus::exact(
                value,
                error,
                ExactBasis::Exhaustive { cardinality: off.nrows() as u64 },
                Some(worst),
                PathFamily { inputs: off.nrows() },
            )?)
        }
    };
    let plan = NativeEditPlan::new(problem.registry, edits)?;
    Ok(PathReport {
        compiled: CompiledControl::new(control.to_string(), Some(plan), realization)?,
        iterations,
        residual,
        residual_band,
        off_target_damage,
    })
}
