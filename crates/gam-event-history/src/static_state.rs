//! Adaptive integration for genuinely static Gaussian frailties.
//! Grid placement uses the whole observed likelihood; chronological updates
//! on that one grid telescope to the same integral for every event placement.

use crate::chain::{Grid, normal_density};
use crate::cohort::EventHistoryError;
use crate::marginal::{ForwardPass, SubjectInputs, centred_baseline, condition, node_likelihood};
use crate::scalar::{add_real, div, exp, ln, sqrt};
use gam_math::nested_dual::JetField;

fn failure(reason: &str) -> EventHistoryError {
    EventHistoryError::NumericalFailure { reason: reason.to_string() }
}

pub(crate) fn filter<S: JetField>(inputs: &SubjectInputs<'_, S>, initial: Option<(&Grid<S>, &[S])>,
    compensated: &[bool]) -> Result<ForwardPass<S>, EventHistoryError> {
    let like = &inputs.eta0[0];
    let marks = inputs.nodes.counts.ncols();
    let atoms = inputs.rates.len();
    let (grid, mut density) = match initial {
        Some((grid, density)) => (grid.clone(), density.to_vec()),
        None => {
            let grid = posterior_grid(inputs, compensated)?;
            let density = prior(&grid, like);
            (grid, density)
        }
    };
    let mut pass = ForwardPass { grids: Vec::new(), alpha: Vec::new(), predicted: Vec::new(), log_normalisers: Vec::new() };
    for n in 0..inputs.nodes.len() {
        let likelihood = node_likelihood(&grid, &inputs.eta0[n * marks..(n + 1) * marks],
            inputs.loadings, &inputs.nodes.counts.row(n).to_vec(), &inputs.nodes.exposure_row(n),
            Some(compensated), inputs.log_normaliser.map(|m| &m[n * marks..(n + 1) * marks]), marks, atoms, false);
        let (updated, mass) = condition(&grid, &density, &likelihood.ell, likelihood.shift, "static frailty")?;
        pass.predicted.push(density);
        pass.grids.push(grid.clone());
        pass.log_normalisers.push(add_real(&ln(&mass), likelihood.shift));
        pass.alpha.push(updated.clone());
        density = updated;
    }
    Ok(pass)
}

fn solve<S: JetField>(matrix: &[S], rhs: &[S]) -> Result<Vec<S>, EventHistoryError> {
    let n = rhs.len();
    let zero = rhs[0].constant_like(0.0);
    let mut lower = vec![zero.clone(); n * n];
    for i in 0..n {
        for j in 0..=i {
            let mut value = matrix[i * n + j].clone();
            for k in 0..j { value = value.sub(&lower[i * n + k].mul(&lower[j * n + k])); }
            lower[i * n + j] = if i == j {
                if !(value.value().is_finite() && value.value() > 0.0) {
                    return Err(failure("static posterior precision is not positive definite"));
                }
                sqrt(&value)
            } else { div(&value, &lower[j * n + j]) };
        }
    }
    let mut out = rhs.to_vec();
    for i in 0..n {
        for j in 0..i { out[i] = out[i].sub(&lower[i * n + j].mul(&out[j])); }
        out[i] = div(&out[i], &lower[i * n + i]);
    }
    for i in (0..n).rev() {
        for j in i + 1..n { out[i] = out[i].sub(&lower[j * n + i].mul(&out[j])); }
        out[i] = div(&out[i], &lower[i * n + i]);
    }
    Ok(out)
}

pub(crate) fn prior<S: JetField>(grid: &Grid<S>, like: &S) -> Vec<S> {
    let zero = like.constant_like(0.0);
    let unit = like.constant_like(1.0);
    (0..grid.size()).map(|i| (0..grid.dimension()).fold(unit.clone(), |p, k|
        p.mul(&normal_density(grid.coordinate(i, k), &zero, &unit)))).collect()
}

pub(crate) fn posterior_grid<S: JetField>(
    inputs: &SubjectInputs<'_, S>, compensated: &[bool],
) -> Result<Grid<S>, EventHistoryError> {
    let atoms = inputs.rates.len();
    let marks = inputs.nodes.counts.ncols();
    let like = &inputs.eta0[0];
    let zero = like.constant_like(0.0);
    let mut linear = vec![zero.clone(); atoms];
    let mut log_hazards = Vec::with_capacity(marks);
    for d in 0..marks {
        let loadings = &inputs.loadings[d * atoms..(d + 1) * atoms];
        let mut terms = Vec::new();
        for n in 0..inputs.nodes.len() {
            for k in 0..atoms {
                linear[k] = linear[k].add(&loadings[k].scale(inputs.nodes.counts[[n, d]]));
            }
            let exposure = inputs.nodes.exposure_row(n)[d];
            if compensated[d] && exposure > 0.0 {
                terms.push(add_real(&centred_baseline(&inputs.eta0[n * marks + d], loadings,
                    inputs.log_normaliser.map(|m| &m[n * marks + d])), exposure.ln()));
            }
        }
        log_hazards.push(if terms.is_empty() { None } else {
            Some(crate::chain::log_sum_exp(&terms))
        });
    }
    let evaluate = |z: &[S]| {
        let mut value = zero.clone();
        let mut gradient = linear.to_vec();
        let mut precision = vec![zero.clone(); atoms * atoms];
        for k in 0..atoms {
            value = value.add(&linear[k].mul(&z[k])).sub(&z[k].mul(&z[k]).scale(0.5));
            gradient[k] = gradient[k].sub(&z[k]);
            precision[k * atoms + k] = like.constant_like(1.0);
        }
        for (d, log_hazard) in log_hazards.iter().enumerate() {
            if let Some(log_hazard) = log_hazard {
                let a = &inputs.loadings[d * atoms..(d + 1) * atoms];
                let log_rate = a.iter().zip(z).fold(log_hazard.clone(), |acc, (a, z)| acc.add(&a.mul(z)));
                let rate = exp(&log_rate);
                value = value.sub(&rate);
                for k in 0..atoms {
                    gradient[k] = gradient[k].sub(&rate.mul(&a[k]));
                    for j in 0..atoms {
                        precision[k * atoms + j] = precision[k * atoms + j].add(&rate.mul(&a[k]).mul(&a[j]));
                    }
                }
            }
        }
        (value, gradient, precision)
    };
    let mut means = vec![zero.clone(); atoms];
    // Fixed iteration depth carries parameter sensitivities through the
    // converged mode, including when its primal value has stopped moving.
    for _ in 0..24 {
        let (value, gradient, precision) = evaluate(&means);
        let step = solve(&precision, &gradient)?;
        let mut scale = 1.0;
        let mut next: Vec<S> = means.iter().zip(&step).map(|(z, d)| z.add(d)).collect();
        let mut accepted = false;
        for _ in 0..40 {
            let proposed = evaluate(&next).0.value();
            if proposed.is_finite() && proposed >= value.value() - 16.0 * f64::EPSILON * (1.0 + value.value().abs()) {
                accepted = true;
                break;
            }
            scale *= 0.5;
            next = means.iter().zip(&step).map(|(z, d)| z.add(&d.scale(scale))).collect();
        }
        if !accepted { return Err(failure("static posterior grid placement did not converge")); }
        means = next;
    }
    let (value, gradient, precision) = evaluate(&means);
    if !value.value().is_finite() || gradient.iter().any(|g| !g.value().is_finite() || g.value().abs() > 1e-8) {
        return Err(failure("static posterior grid placement has an unresolved score"));
    }
    let mut scales = Vec::with_capacity(atoms);
    for k in 0..atoms {
        let mut unit = vec![zero.clone(); atoms];
        unit[k] = like.constant_like(1.0);
        scales.push(sqrt(&solve(&precision, &unit)?[k]));
    }
    Ok(Grid::new(inputs.gh, &means, &scales, like))
}
