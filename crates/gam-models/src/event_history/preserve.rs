//! Risk-set marginal preservation: what makes `η⁰` the marginal rate.
//!
//! The latent term enters the intensity as a deviation from a population
//! rate, and something has to say which population. The family's default
//! answer is the *stationary prior*: `−½|a_d|²` is exactly `−log E_z[e^{a_d·z}]`
//! under `z ~ N(0, I)`, so `E_z λ_d = exp(η⁰_d)` averaged over everybody the
//! cohort started with.
//!
//! That is the wrong population for an incidence rate. An incidence rate is a
//! rate *among those still at risk*, and the people still at risk are not a
//! draw from the prior: they are the ones whose own latent state kept them
//! event-free, which is a selection toward the low activities. The gap is not
//! second order. Take two equal halves of a cohort at relative hazards `0.2`
//! and `1.8` — mean one — and a baseline hazard of `0.1` over ten years:
//!
//! ```text
//! S(10) = ½e^{−0.2} + ½e^{−1.8} ≈ 0.492,   not   e^{−1} ≈ 0.368.
//! ```
//!
//! Nothing is wrong with that mixture. What is wrong is calling its baseline
//! the population's continuing hazard: the two differ by a third of the risk
//! after one mean event time, and by more as the heterogeneity grows.
//!
//! This module replaces the constant shift with the predictable one it should
//! have been,
//!
//! ```text
//! log M_d(t) = log E[ e^{a_d·z(t)} | still at risk for d at t⁻ ],
//! ```
//!
//! so that
//!
//! ```text
//! E[λ_d(t) | at risk at t⁻] = exp(η⁰_d(t))
//! ```
//!
//! exactly, at every age rather than only at the cohort's start. The
//! consequence worth having is the one the constant shift cannot give: for a
//! single first-occurrence mark the *marginal* survival of the reference
//! population is exactly `exp(−∫ e^{η⁰})`, because `dR/dt = −R · E[λ | at
//! risk]`. Raising the latent heterogeneity then moves risk between people at
//! a fixed population rate instead of moving the population's rate.
//!
//! ## How it is computed, and what that costs
//!
//! `M` is a property of the reference population's own evolution, so it is
//! computed by running that population forward: the same Gauss-Hermite filter
//! the fit uses, from the stationary prior, with no events and the killing of
//! the marks that remove a subject from the risk set in question. The
//! *predicted* density at a node — before that node's own factor — is the law
//! of the state among those still at risk just before it, and `M` is the
//! expectation of `e^{a_d·z}` under it. Taking the predicted density rather
//! than the filtered one is what makes `M` predictable, and it is also what
//! breaks the circularity: a node's normaliser is a function of the killing
//! strictly before it.
//!
//! The risk set differs by mark kind. A once-only mark leaves its own risk set
//! when it fires, so its killing is the terminal marks and itself; a recurrent
//! or terminal mark's risk set is simply the living, so its killing is the
//! terminal marks. Marks that share a killing share a filter, so the cost is
//! one filter per once-only mark plus one, over the reference grid — a
//! population quantity, independent of the number of subjects.
//!
//! The reference population is stratified: within a stratum the identity above
//! is exact, and a subject divides by its own stratum's normaliser. Across
//! strata it is exact only where the covariates the strata condition on are
//! the covariates the selection acts through. A single stratum is a claim
//! about one reference profile, not about everybody.

use super::chain::{GaussHermite, Grid, normal_density};
use super::cohort::{EventHistoryError, MarkKind};
use super::marginal::{condition, node_likelihood, predict, transitions_across, weighted_sum};
use super::scalar::{div, exp, ln};
use gam_math::nested_dual::JetField;

/// The reference population of every stratum: which covariate row its profile
/// is, and which stratum every subject belongs to.
#[derive(Clone, Debug, PartialEq)]
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
#[derive(Clone, Debug)]
pub struct ReferenceGrid {
    pub times: Vec<f64>,
    pub gaps: Vec<f64>,
    pub weights: Vec<f64>,
}

impl ReferenceGrid {
    pub fn len(&self) -> usize {
        self.times.len()
    }

    pub fn is_empty(&self) -> bool {
        self.times.is_empty()
    }

    /// Where a time sits on this grid: the node below it and the weight of
    /// the node above, for a linear interpolation in the log of the
    /// normaliser — the scale the normaliser varies on.
    pub fn locate(&self, t: f64) -> (usize, f64) {
        let last = self.times.len() - 1;
        if t <= self.times[0] {
            return (0, 0.0);
        }
        if t >= self.times[last] {
            return (last - 1, 1.0);
        }
        let upper = self.times.partition_point(|&node| node <= t);
        let lower = upper - 1;
        let width = self.times[upper] - self.times[lower];
        let weight = if width > 0.0 {
            (t - self.times[lower]) / width
        } else {
            0.0
        };
        (lower, weight)
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

/// Run one stratum's reference population forward and read its normalisers.
///
/// `eta0` is `nodes × marks` at the stratum's profile. The recursion is
/// exactly the fit's own filter: predict across the gap, take the normaliser
/// from the *predicted* density, then condition on the node's killing.
pub(crate) fn stratum_normalisers<S: JetField>(
    grid: &ReferenceGrid,
    eta0: &[S],
    loadings: &[S],
    rates: &[S],
    time_scale: f64,
    gh: &GaussHermite,
    kinds: &[MarkKind],
    atoms: usize,
) -> Result<Normalisers<S>, EventHistoryError> {
    let marks = kinds.len();
    let nodes = grid.len();
    if eta0.len() != nodes * marks {
        return Err(EventHistoryError::InvalidInput {
            reason: format!(
                "reference η⁰ has {} entries for {nodes} nodes and {marks} marks",
                eta0.len()
            ),
        });
    }
    let (masks, mask_of) = killing_masks(kinds);
    let like = &eta0[0];
    let zero: Vec<S> = (0..atoms).map(|_| like.constant_like(0.0)).collect();
    let unit: Vec<S> = (0..atoms).map(|_| like.constant_like(1.0)).collect();
    let prior_density = |grid: &Grid<S>| -> Vec<S> {
        (0..grid.size())
            .map(|i| {
                let mut density = like.constant_like(1.0);
                for k in 0..atoms {
                    density =
                        density.mul(&normal_density(grid.coordinate(i, k), &zero[k], &unit[k]));
                }
                density
            })
            .collect()
    };
    let mut states: Vec<Option<(Grid<S>, Vec<S>)>> = (0..masks.len()).map(|_| None).collect();
    let mut log_normaliser: Vec<S> = Vec::with_capacity(nodes * marks);
    let mut log_risk_mass: Vec<S> = Vec::with_capacity(nodes * masks.len());
    let mut carried: Vec<S> = (0..masks.len()).map(|_| like.constant_like(0.0)).collect();
    let no_counts = vec![0.0; marks];
    for n in 0..nodes {
        // Predict every risk set's population to this node.
        let mut predicted: Vec<(Grid<S>, Vec<S>)> = Vec::with_capacity(masks.len());
        for state in states.iter() {
            match state {
                None => {
                    let grid = Grid::new(gh, &zero, &unit, like);
                    let density = prior_density(&grid);
                    predicted.push((grid, density));
                }
                Some((previous_grid, previous_alpha)) => {
                    let transitions = transitions_across(rates, grid.gaps[n - 1], time_scale)?;
                    let (next_grid, next_density) = predict(
                        gh,
                        like,
                        previous_grid,
                        previous_alpha,
                        &transitions,
                        "reference population",
                    )?;
                    predicted.push((next_grid, next_density));
                }
            }
        }
        // The normaliser of every mark, under the predicted law of its own
        // risk set: `M_d = E[e^{a_d·z} | at risk at t⁻]`. Taking the density
        // before this node's own killing is what makes it predictable.
        for d in 0..marks {
            let (mask_grid, mask_density) = &predicted[mask_of[d]];
            let loadings_d = &loadings[d * atoms..(d + 1) * atoms];
            let mut weighted = Vec::with_capacity(mask_grid.size());
            for i in 0..mask_grid.size() {
                let mut exponent = like.constant_like(0.0);
                for (k, a) in loadings_d.iter().enumerate() {
                    exponent = exponent.add(&a.mul(mask_grid.coordinate(i, k)));
                }
                weighted.push(mask_density[i].mul(&exp(&exponent)));
            }
            let numerator = weighted_sum(&mask_grid.weights, &weighted);
            let denominator = weighted_sum(&mask_grid.weights, mask_density);
            if !(denominator.value() > 0.0) || !numerator.value().is_finite() {
                return Err(EventHistoryError::NumericalFailure {
                    reason: format!(
                        "reference population: the risk set of mark {d} at node {n} has mass {}",
                        denominator.value()
                    ),
                });
            }
            log_normaliser.push(ln(&div(&numerator, &denominator)));
        }
        // Condition every risk set on its own killing over this node.
        let shift = &log_normaliser[n * marks..(n + 1) * marks];
        for (m, mask) in masks.iter().enumerate() {
            let exposures: Vec<f64> = mask
                .iter()
                .map(|&killed| if killed { grid.weights[n] } else { 0.0 })
                .collect();
            let (mask_grid, mask_density) = &predicted[m];
            let likelihood = node_likelihood(
                mask_grid,
                &eta0[n * marks..(n + 1) * marks],
                loadings,
                &no_counts,
                &exposures,
                None,
                Some(shift),
                marks,
                atoms,
                false,
            );
            let (alpha, normaliser) = condition(
                mask_grid,
                mask_density,
                &likelihood.ell,
                likelihood.shift,
                "reference population",
            )?;
            carried[m] = carried[m]
                .add(&ln(&normaliser))
                .add(&like.constant_like(likelihood.shift));
            log_risk_mass.push(carried[m].clone());
            states[m] = Some((mask_grid.clone(), alpha));
        }
    }
    Ok(Normalisers {
        log_normaliser,
        log_risk_mass,
        masks: masks.len(),
    })
}
