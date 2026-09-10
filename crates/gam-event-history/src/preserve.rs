//! Risk-set reference evolution for the shared latent point process.
//!
//! In continuous time, dividing each mark's intensity by its survivor-law
//! moment makes the reference risk-set mean equal to the baseline intensity.
//! Numerical evolution uses symmetric OU splitting and an implicit midpoint
//! killing step. Finite steps approximate that identity; the fitting driver
//! must resolve both normalisers and risk masses by time refinement.
//!
use super::chain::{GaussHermite, Grid, normal_density};
use super::cohort::{EventHistoryError, MarkKind};
use super::marginal::{condition, node_likelihood, predict, transitions_across, weighted_sum};
use super::scalar::{exp, ln};
use gam_math::nested_dual::JetField;

/// The reference population of every stratum: which covariate row its profile
/// is, and which stratum every subject belongs to.
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct ReferenceStrata {
    /// Row of the cohort's covariate table holding each stratum's profile.
    pub rows: Vec<usize>,
    /// The stratum of every subject, in the cohort's subject order.
    pub subject: Vec<usize>,
}

impl ReferenceStrata {
    /// One stratum at the given covariate row: every subject divides by the
    /// same reference population's normaliser.
    pub fn single(row: usize, subjects: usize) -> Self {
        Self {
            rows: vec![row],
            subject: vec![0; subjects],
        }
    }

    pub fn strata(&self) -> usize {
        self.rows.len()
    }

    pub fn validate(&self, subjects: usize, rows: usize) -> Result<(), EventHistoryError> {
        if self.rows.is_empty() {
            return Err(EventHistoryError::InvalidInput {
                reason: "reference strata need at least one profile".to_string(),
            });
        }
        if let Some(row) = self.rows.iter().find(|&&row| row >= rows) {
            return Err(EventHistoryError::InvalidInput {
                reason: format!(
                    "a reference profile names covariate row {row} of a {rows}-row table"
                ),
            });
        }
        if self.subject.len() != subjects {
            return Err(EventHistoryError::InvalidInput {
                reason: format!(
                    "{} subject strata for {subjects} subjects",
                    self.subject.len()
                ),
            });
        }
        if let Some(stratum) = self.subject.iter().find(|&&s| s >= self.rows.len()) {
            return Err(EventHistoryError::InvalidInput {
                reason: format!(
                    "a subject is in stratum {stratum} of {} strata",
                    self.rows.len()
                ),
            });
        }
        Ok(())
    }
}

/// The killing of every mark's risk set, and the grouping of marks that share
/// one: a subject leaves a once-only mark's risk set when that mark fires or
/// when a terminal mark does, and leaves any other mark's when a terminal mark
/// does.
pub(crate) fn killing_masks(kinds: &[MarkKind]) -> (Vec<Vec<bool>>, Vec<usize>) {
    let marks = kinds.len();
    let terminal: Vec<bool> = kinds.iter().map(|k| *k == MarkKind::Terminal).collect();
    let mut masks: Vec<Vec<bool>> = Vec::new();
    let mut of_mark = vec![0usize; marks];
    for d in 0..marks {
        let mut mask = terminal.clone();
        if kinds[d] == MarkKind::Once {
            mask[d] = true;
        }
        of_mark[d] = match masks.iter().position(|m| *m == mask) {
            Some(index) => index,
            None => {
                masks.push(mask);
                masks.len() - 1
            }
        };
    }
    (masks, of_mark)
}

/// The reference grid one stratum's population is run forward on: the node
/// times, the gaps between them and the quadrature weight each node carries.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct ReferenceGrid {
    pub times: Vec<f64>,
    pub gaps: Vec<f64>,
}

impl ReferenceGrid {
    pub fn len(&self) -> usize {
        self.times.len()
    }

    pub fn is_empty(&self) -> bool {
        self.times.is_empty()
    }

    /// Where a supported time sits on this grid: the node below it and the weight of
    /// the node above, for a linear interpolation in the log of the
    /// normaliser — the scale the normaliser varies on.
    pub fn locate(&self, t: f64) -> Result<(usize, f64), EventHistoryError> {
        if self.times.len() < 2 || !t.is_finite()
            || t < self.times[0] || t > self.times[self.times.len() - 1] {
            return Err(EventHistoryError::InvalidInput {
                reason: format!("time {t} is outside the supported reference interval {:?}..{:?}",
                    self.times.first(), self.times.last()),
            });
        }
        let last = self.times.len() - 1;
        if t == self.times[0] {
            return Ok((0, 0.0));
        }
        if t == self.times[last] {
            return Ok((last - 1, 1.0));
        }
        let upper = self.times.partition_point(|&node| node <= t);
        let lower = upper - 1;
        let width = self.times[upper] - self.times[lower];
        let weight = if width > 0.0 {
            (t - self.times[lower]) / width
        } else {
            0.0
        };
        Ok((lower, weight))
    }
}

/// One stratum's log normalisers, `nodes × marks`, and the log of the risk
/// mass its population retains, `nodes × masks` — the marginal survival of
/// each risk set, which the identity makes `−∫ e^{η⁰}` for a single mark.
pub(crate) struct Normalisers<S> {
    pub log_normaliser: Vec<S>,
    pub log_risk_mass: Vec<S>,
    pub masks: usize,
}

/// Log survivor-law activity moments, with a numerical log-sum-exp shift.
fn moments<S: JetField>(
    populations: &[(Grid<S>, Vec<S>)], of_mark: &[usize], loadings: &[S],
    atoms: usize, like: &S,
) -> Result<Vec<S>, EventHistoryError> {
    let mut out = Vec::with_capacity(of_mark.len());
    for (d, &mask) in of_mark.iter().enumerate() {
        let (grid, density) = &populations[mask];
        let exponents: Vec<S> = (0..grid.size()).map(|i| {
            let mut value = like.constant_like(0.0);
            for k in 0..atoms {
                value = value.add(&loadings[d * atoms + k].mul(grid.coordinate(i, k)));
            }
            value
        }).collect();
        let shift = exponents.iter().map(JetField::value).fold(f64::NEG_INFINITY, f64::max);
        let weighted: Vec<S> = exponents.iter().zip(density.iter()).map(|(value, p)|
            p.mul(&exp(&value.sub(&like.constant_like(shift))))).collect();
        let numerator = weighted_sum(&grid.weights, &weighted);
        let denominator = weighted_sum(&grid.weights, density);
        if !(numerator.value().is_finite() && numerator.value() > 0.0
            && denominator.value().is_finite() && denominator.value() > 0.0) {
            return Err(EventHistoryError::NumericalFailure {
                reason: format!("reference population: invalid activity moment for mark {d}"),
            });
        }
        out.push(ln(&numerator).sub(&ln(&denominator)).add(&like.constant_like(shift)));
    }
    Ok(out)
}

/// Advance all risk sets with a common set of per-mark normalisers. A terminal
/// hazard uses the living-law normaliser even in a once-only disease's risk set.
fn kill<S: JetField>(
    populations: &[(Grid<S>, Vec<S>)], masks: &[Vec<bool>], eta0: &[S],
    loadings: &[S], shift: &[S], exposure: f64, atoms: usize,
) -> Result<(Vec<(Grid<S>, Vec<S>)>, Vec<S>), EventHistoryError> {
    let marks = eta0.len();
    let no_counts = vec![0.0; marks];
    let mut out = Vec::with_capacity(masks.len());
    let mut masses = Vec::with_capacity(masks.len());
    for ((grid, density), mask) in populations.iter().zip(masks) {
        let exposures: Vec<f64> = mask.iter().map(|&killed|
            if killed { exposure } else { 0.0 }).collect();
        let likelihood = node_likelihood(grid, eta0, loadings, &no_counts,
            &exposures, None, Some(shift), marks, atoms, false);
        let (alpha, mass) = condition(grid, density, &likelihood.ell,
            likelihood.shift, "reference population")?;
        masses.push(ln(&mass).add(&eta0[0].constant_like(likelihood.shift)));
        out.push((grid.clone(), alpha));
    }
    Ok((out, masses))
}

/// Evolve from the reference entry to each endpoint, reporting the law AT
/// that time. Every coefficient derivative propagates through the same steps.
pub(crate) fn stratum_normalisers<S: JetField>(
    grid: &ReferenceGrid, eta0: &[S], loadings: &[S], rates: &[S],
    time_scale: f64, gh: &GaussHermite, kinds: &[MarkKind], atoms: usize,
) -> Result<Normalisers<S>, EventHistoryError> {
    let marks = kinds.len();
    let nodes = grid.len();
    if nodes < 2 || marks == 0 || eta0.len() != nodes * marks
        || loadings.len() != marks * atoms || rates.len() != atoms
        || grid.gaps.len() != nodes - 1
        || grid.times.iter().any(|t| !t.is_finite())
        || grid.times.windows(2).zip(&grid.gaps).any(|(w, gap)|
            !(w[1] > w[0] && *gap == w[1] - w[0])) {
        return Err(EventHistoryError::InvalidInput {
            reason: "reference evolution needs increasing endpoints and conformable parameters".to_string(),
        });
    }
    let (masks, of_mark) = killing_masks(kinds);
    let like = &eta0[0];
    let zero: Vec<S> = (0..atoms).map(|_| like.constant_like(0.0)).collect();
    let unit: Vec<S> = (0..atoms).map(|_| like.constant_like(1.0)).collect();
    let initial = Grid::new(gh, &zero, &unit, like);
    let density: Vec<S> = (0..initial.size()).map(|i| {
        let mut p = like.constant_like(1.0);
        for k in 0..atoms {
            p = p.mul(&normal_density(initial.coordinate(i, k), &zero[k], &unit[k]));
        }
        p
    }).collect();
    let mut populations = vec![(initial, density); masks.len()];
    let mut carried = vec![like.constant_like(0.0); masks.len()];
    let mut log_normaliser = Vec::with_capacity(nodes * marks);
    let mut log_risk_mass = Vec::with_capacity(nodes * masks.len());
    for n in 0..nodes {
        log_normaliser.extend(moments(&populations, &of_mark, loadings, atoms, like)?);
        log_risk_mass.extend(carried.iter().cloned());
        if n + 1 == nodes { break; }
        let dt = grid.gaps[n];
        let transitions = transitions_across(rates, 0.5 * dt, time_scale)?;
        let diffuse = |populations: &[(Grid<S>, Vec<S>)]| -> Result<Vec<(Grid<S>, Vec<S>)>, EventHistoryError> {
            populations.iter().map(|(grid, density)|
                predict(gh, like, grid, density, &transitions, "reference population")).collect()
        };
        let middle = diffuse(&populations)?;
        let eta_mid: Vec<S> = (0..marks).map(|d|
            eta0[n * marks + d].add(&eta0[(n + 1) * marks + d]).scale(0.5)).collect();
        let mut shift = moments(&middle, &of_mark, loadings, atoms, like)?;
        let mut residual = f64::INFINITY;
        // Fixed arithmetic depth: derivative channels follow every iteration.
        // Refinement, not clipping, handles an unresolved midpoint equation.
        for _ in 0..12 {
            let (selected, _) = kill(&middle, &masks, &eta_mid, loadings, &shift, 0.5 * dt, atoms)?;
            let next = moments(&selected, &of_mark, loadings, atoms, like)?;
            residual = shift.iter().zip(&next).map(|(a, b)|
                (a.value() - b.value()).abs()).fold(0.0, f64::max);
            shift = next;
        }
        if !(residual.is_finite() && residual <= 1e-10) {
            return Err(EventHistoryError::NumericalFailure {
                reason: format!("reference midpoint unresolved on interval {n}: {residual:.3e}; refine the reference grid"),
            });
        }
        let (selected, masses) = kill(&middle, &masks, &eta_mid, loadings, &shift, dt, atoms)?;
        for (mass, increment) in carried.iter_mut().zip(masses) { *mass = mass.add(&increment); }
        populations = diffuse(&selected)?;
    }
    Ok(Normalisers { log_normaliser, log_risk_mass, masks: masks.len() })
}
