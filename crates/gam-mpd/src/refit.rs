//! Refitting the continuous reals of an operator program to its contract (#2951).
//!
//! A structural change (a dropped block, a coarser lattice, a basis change) moves the executed
//! function; the surviving reals can often absorb it. Refitting is part of the search, not of the
//! acceptance: a refitted program is accepted only through the same decoded-bits-plus-contract
//! rule as any other candidate.
//!
//! # The convex case: operators feeding a readout
//!
//! For the affine node `y = Σ_t x_t A_tᵀ + b` whose output a readout maps to logits `z = y Φᵀ`,
//! with every other node held at its current value, the objective
//!
//! ```text
//! f(A, b) = Σ_rows KL(p ‖ softmax z) = Σ_rows [lse(z) − pᵀz] + const
//! ```
//!
//! is convex in the present reals of `A_t` and `b` (a log-sum-exp of a linear map). Its gradient is
//! `∂f/∂A_t = (G Φ)ᵀ x_t` with `G = softmax(z) − p` per row, and its Hessian acts on a direction
//! `(V_t, v)` as `(W Φ)ᵀ x_t` with `W = q ⊙ δz − q (qᵀδz)`, `δz = (Σ_t x_t V_tᵀ + v) Φᵀ`: both exact,
//! no finite differences. [`refit_readout`] takes Newton steps whose direction solves the Newton
//! system by conjugate gradients on those products, each step accepted only when it lowers `f`
//! (halving otherwise), and rounds the result to each operator's lattice. The iteration counts are
//! search settings: they change how good a refit is found, never what is accepted.

use super::engine::EngineError;
use super::operator_program::{FamilyInputs, Node, Operator, OperatorBody, OperatorProgram, Provenance};
use gam_linalg::roundoff::accumulation_growth;
use gam_linalg::utils::solve_spd_pcg_bounded_into;
use ndarray::{Array1, Array2, Axis, s};
use std::sync::Arc;

/// How hard [`refit_readout`] searches.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RefitSearch {
    pub newton_steps: usize,
    pub conjugate_gradient_steps: usize,
}

/// `Σ_rows [lse(z) − pᵀz]` and `softmax(z)` per row.
fn objective(logits: &Array2<f64>, target: &Array2<f64>) -> (f64, Array2<f64>) {
    let mut q = logits.clone();
    let mut total = 0.0;
    for (mut row, p) in q.outer_iter_mut().zip(target.outer_iter()) {
        let m = row.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let sum: f64 = row.iter().map(|v| (v - m).exp()).sum();
        let lse = m + sum.ln();
        total += lse - row.dot(&p);
        row.mapv_inplace(|v| (v - lse).exp());
    }
    (total, q)
}

/// The refit parameters: each operator's present-entry mask and its current matrix.
struct Parameters {
    masks: Vec<Array2<f64>>,
    values: Vec<Array2<f64>>,
}

/// Refit the present reals of `operators` (terms or the bias of the affine node the readout reads)
/// to minimize `Σ KL(p ‖ softmax z)` against the model distributions `target` (rows of
/// probabilities), holding every other node at its value on `inputs`.
pub fn refit_readout(
    program: &OperatorProgram,
    inputs: &FamilyInputs,
    target: &Array2<f64>,
    operators: &[usize],
    search: RefitSearch,
) -> Result<OperatorProgram, EngineError> {
    let Node::Readout { input: y_node, basis } = program.nodes[program.output] else {
        return Err(EngineError::Primitive("the output is not a readout".to_string()));
    };
    let Node::Affine { terms, bias } = program.nodes[y_node].clone() else {
        return Err(EngineError::Primitive("the readout does not read an affine node".to_string()));
    };
    let trace = program.execute(inputs, false)?;
    let size = program.declarations.domains[match &program.bases[basis] {
        super::operator_program::Basis::Indicator { domain } | super::operator_program::Basis::Characters { domain, .. } => *domain,
    }]
    .size;
    // The readout's table Φ (classes × width); an indicator basis is the identity, never formed.
    let phi = match &program.bases[basis] {
        super::operator_program::Basis::Indicator { .. } => None,
        other => Some(other.evaluate(&program.declarations, &(0..size as u32).collect::<Vec<u32>>())?.values),
    };
    let forward = |y: Array2<f64>| -> Array2<f64> {
        match phi.as_ref() {
            None => y,
            Some(phi) => y.dot(&phi.t()),
        }
    };
    let backward = |g: Array2<f64>| -> Array2<f64> {
        match phi.as_ref() {
            None => g,
            Some(phi) => g.dot(phi),
        }
    };
    // Each refit slot: (operator, the node it reads, or None for the bias).
    let mut slots: Vec<(usize, Option<usize>)> = Vec::new();
    for &op in operators {
        if let Some((argument, _)) = terms.iter().find(|(_, t)| *t == op) {
            slots.push((op, Some(*argument)));
        } else if bias == Some(op) {
            slots.push((op, None));
        } else {
            return Err(EngineError::Primitive(format!("operator {op} does not feed the readout's affine node")));
        }
        if !matches!(program.operators[op].body, OperatorBody::Dense { .. }) {
            return Err(EngineError::Primitive(format!("operator {op} is not dense")));
        }
    }
    let masks: Vec<Array2<f64>> = slots
        .iter()
        .map(|(op, _)| {
            let operator = &program.operators[*op];
            let mut mask = Array2::<f64>::zeros(operator.matrix().dim());
            if let OperatorBody::Dense { present, .. } = &operator.body {
                for ((r, c), keep) in present.indexed_iter() {
                    if *keep {
                        mask.slice_mut(s![operator.rows.range(r), operator.cols.range(c)]).fill(1.0);
                    }
                }
            }
            mask
        })
        .collect();
    let mut parameters = Parameters { values: slots.iter().map(|(op, _)| program.operators[*op].matrix()).collect(), masks };
    let fixed_y = &trace.values[y_node] - &slot_output(&slots, &parameters.values, &trace.values);
    let inputs_of = |slot: &Option<usize>| -> Array2<f64> {
        match slot {
            Some(node) => trace.values[*node].clone(),
            None => Array2::ones((inputs.rows, 1)),
        }
    };
    let xs: Vec<Array2<f64>> = slots.iter().map(|(_, n)| inputs_of(n)).collect();
    let logits_of = |values: &[Array2<f64>]| -> Array2<f64> {
        let mut y = fixed_y.clone();
        for (x, a) in xs.iter().zip(values) {
            y += &x.dot(&a.t());
        }
        forward(y)
    };
    let gradient_of = |q: &Array2<f64>| -> Vec<Array2<f64>> {
        let g_y = backward(q - target);
        xs.iter().zip(&parameters.masks).map(|(x, m)| g_y.t().dot(x) * m).collect()
    };
    for _ in 0..search.newton_steps {
        let (value, q) = objective(&logits_of(&parameters.values), target);
        let gradient = gradient_of(&q);
        let hessian_times = |direction: &[Array2<f64>]| -> Vec<Array2<f64>> {
            let mut dy = Array2::<f64>::zeros(fixed_y.dim());
            for (x, v) in xs.iter().zip(direction) {
                dy += &x.dot(&v.t());
            }
            let dz = forward(dy);
            let mut w = &q * &dz;
            let inner = (&q * &dz).sum_axis(Axis(1));
            for (mut row, (qr, i)) in w.outer_iter_mut().zip(q.outer_iter().zip(inner.iter())) {
                row.scaled_add(-*i, &qr);
            }
            let wy = backward(w);
            xs.iter().zip(&parameters.masks).map(|(x, m)| wy.t().dot(x) * m).collect()
        };
        // The Newton direction solves H d = −g by the linear-algebra owner's preconditioned
        // conjugate gradients on the packed present reals (H is positive semidefinite: the exact
        // Hessian of a log-sum-exp of a linear map). Its tolerance is the packing's accumulation
        // band; the iteration count is a search setting, and a truncated iterate is still a descent
        // direction.
        let rhs = pack(&parameters.masks, &gradient.iter().map(|g| -g).collect::<Vec<_>>());
        if rhs.iter().all(|v| *v == 0.0) {
            break;
        }
        let shapes: Vec<(usize, usize)> = parameters.masks.iter().map(|m| m.dim()).collect();
        let apply = |x: &Array1<f64>, out: &mut Array1<f64>| {
            let direction = unpack(&parameters.masks, &shapes, x);
            out.assign(&pack(&parameters.masks, &hessian_times(&direction)));
        };
        let unit = Array1::<f64>::ones(rhs.len());
        let tolerance = accumulation_growth(rhs.len());
        let Some((solution, _, _)) =
            solve_spd_pcg_bounded_into(apply, &rhs, &unit, tolerance, search.conjugate_gradient_steps)
        else {
            break;
        };
        let d = unpack(&parameters.masks, &shapes, &solution);
        let mut step = 1.0;
        let mut improved = false;
        while step > f64::EPSILON {
            let trial: Vec<Array2<f64>> = parameters.values.iter().zip(&d).map(|(v, dv)| v + &(dv * step)).collect();
            let (trial_value, _) = objective(&logits_of(&trial), target);
            if trial_value < value {
                parameters.values = trial;
                improved = true;
                break;
            }
            step *= 0.5;
        }
        if !improved {
            break;
        }
    }
    let mut refit = program.clone();
    for ((op, _), values) in slots.iter().zip(&parameters.values) {
        let old = &program.operators[*op];
        let OperatorBody::Dense { present, precision, .. } = &old.body else { continue };
        refit.operators[*op] = Arc::new(Operator::blocks(
            old.name.clone(),
            old.rows.clone(),
            old.cols.clone(),
            values.clone(),
            present.clone(),
            *precision,
            Provenance::derived(&[&old.provenance], "refit to the contract's readout".to_string()),
        )?);
    }
    Ok(refit)
}

/// The present entries of `values` (masked by `masks`), in order.
fn pack(masks: &[Array2<f64>], values: &[Array2<f64>]) -> Array1<f64> {
    let mut out = Vec::new();
    for (mask, value) in masks.iter().zip(values) {
        for (m, v) in mask.iter().zip(value.iter()) {
            if *m != 0.0 {
                out.push(*v);
            }
        }
    }
    Array1::from(out)
}

/// The inverse of [`pack`]: absent entries are zero.
fn unpack(masks: &[Array2<f64>], shapes: &[(usize, usize)], packed: &Array1<f64>) -> Vec<Array2<f64>> {
    let mut next = packed.iter();
    masks
        .iter()
        .zip(shapes)
        .map(|(mask, shape)| {
            let mut out = Array2::<f64>::zeros(*shape);
            for (o, m) in out.iter_mut().zip(mask.iter()) {
                if *m != 0.0 {
                    *o = next.next().copied().unwrap_or(0.0);
                }
            }
            out
        })
        .collect()
}

fn slot_output(slots: &[(usize, Option<usize>)], values: &[Array2<f64>], trace: &[Array2<f64>]) -> Array2<f64> {
    let rows = trace[0].nrows();
    let width = values.first().map_or(0, |v| v.nrows());
    let mut y = Array2::<f64>::zeros((rows, width));
    for ((_, node), a) in slots.iter().zip(values) {
        match node {
            Some(node) => y += &trace[*node].dot(&a.t()),
            None => y += &a.column(0),
        }
    }
    y
}
