//! Risk-set reference evolution for the shared latent point process.
//!
//! In continuous time, dividing each mark's intensity by its survivor-law
//! moment makes the reference risk-set mean equal to the baseline intensity.
//! Numerical evolution uses symmetric OU splitting and an implicit midpoint
//! killing step. Finite steps approximate that identity; the fitting driver
//! must resolve both normalisers and risk masses by time refinement.
//!
use super::chain::{GaussHermite, Grid, log_standard_prior, log_sum_exp};
use super::cohort::{EventHistoryError, MarkKind};
use super::marginal::{condition, node_likelihood, predict, transitions_across};
use super::scalar::ln;
use gam_math::nested_dual::JetField;
use gam_math::roundoff::{UNIT_ROUNDOFF, accumulation_growth};

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

/// Log survivor-law activity moments of risk sets held as log densities,
/// `ln ∫ e^{a_d·z} α(z) dz − ln ∫ α(z) dz`, each a difference of two
/// log-sum-exps over the grid, with the absolute rounding error of the
/// largest moment beyond the error of the log densities it reads.
///
/// Each log-sum-exp over `n` terms rounds `t_i − s` (an error `u|t_i − s|`
/// in a term of share at most `e^{t_i − s}`, so at most `u·n/e` in all),
/// every `exp` and the fold of the sum (relative `u + γ_{n−1}`), the `ln`
/// (`u ln n`) and the restoring shift (`u` of the result's magnitude): at
/// most `2γ_{n+1} + u|lse|`. Two of them and their difference give
/// `4γ_{n+1} + u(|lse_num| + |lse_den| + |moment|)`. At each point the term
/// `ln w_i + ln α_i`, the exponent `Σ_k a_dk z_ik` (a fold of `2·atoms`
/// operations) and their sum round with at most
/// `u(|ln w_i| + |a_i| + |a_i + e_i|) + γ_{2·atoms} Σ_k |a_dk z_ik|`; an
/// error `δ_i` in the point terms moves the moment by
/// `Σ_i (π_i − ρ_i) δ_i` for the numerator and denominator shares `π`, `ρ`,
/// at most twice the largest `δ_i`.
fn moments<S: JetField>(
    populations: &[(Grid<S>, Vec<S>)], of_mark: &[usize], loadings: &[S],
    atoms: usize, like: &S,
) -> Result<(Vec<S>, f64), EventHistoryError> {
    let u = UNIT_ROUNDOFF;
    let mut out = Vec::with_capacity(of_mark.len());
    let mut error = 0.0_f64;
    for (d, &mask) in of_mark.iter().enumerate() {
        let (grid, log_density) = &populations[mask];
        let size = grid.size();
        let mut point_error = 0.0_f64;
        let mut denominator_terms = Vec::with_capacity(size);
        let mut numerator_terms = Vec::with_capacity(size);
        for i in 0..size {
            let log_weight = ln(&grid.weights[i]);
            let mut exponent = like.constant_like(0.0);
            let mut spread = 0.0;
            for k in 0..atoms {
                let term = loadings[d * atoms + k].mul(grid.coordinate(i, k));
                spread += term.value().abs();
                exponent = exponent.add(&term);
            }
            let base = log_weight.add(&log_density[i]);
            let tilted = base.add(&exponent);
            point_error = point_error.max(
                u * (log_weight.value().abs() + base.value().abs() + tilted.value().abs())
                    + accumulation_growth(2 * atoms) * spread,
            );
            denominator_terms.push(base);
            numerator_terms.push(tilted);
        }
        let numerator = log_sum_exp(&numerator_terms);
        let denominator = log_sum_exp(&denominator_terms);
        if !(numerator.value().is_finite() && denominator.value().is_finite()) {
            return Err(EventHistoryError::NumericalFailure {
                reason: format!("reference population: invalid activity moment for mark {d}"),
            });
        }
        let moment = numerator.sub(&denominator);
        error = error.max(
            4.0 * accumulation_growth(size + 1)
                + u * (numerator.value().abs() + denominator.value().abs() + moment.value().abs())
                + 2.0 * point_error,
        );
        out.push(moment);
    }
    Ok((out, error))
}

/// Advance all risk sets with a common set of per-mark normalisers. A terminal
/// hazard uses the living-law normaliser even in a once-only disease's risk set.
///
/// Also returns the largest absolute rounding error of a conditioned log
/// density point, beyond the error its log normaliser shares with every
/// point (which cancels from each moment). A killed mark's log intensity
/// `η⁰_d − shift_d + Σ_k a_dk z_ik + ln exposure` is formed in `2·atoms + 3`
/// operations on quantities no larger than
/// `Λ_i = |η⁰_d| + |shift_d| + Σ_k |a_dk z_ik| + |ln exposure|`; its `exp`
/// and the sum of the `marks` compensators give `ell_i` a relative error of
/// at most `γ_{2·atoms+4+marks}(1 + Λ_i)`. Adding the log density, removing
/// the likelihood shift and the log normaliser round once each relative to
/// their results.
fn kill<S: JetField>(
    populations: &[(Grid<S>, Vec<S>)], masks: &[Vec<bool>], eta0: &[S],
    loadings: &[S], shift: &[S], exposure: f64, atoms: usize,
) -> Result<(Vec<(Grid<S>, Vec<S>)>, Vec<S>, f64), EventHistoryError> {
    let u = UNIT_ROUNDOFF;
    let marks = eta0.len();
    let no_counts = vec![0.0; marks];
    let intensity_growth = accumulation_growth(2 * atoms + 4 + marks);
    let mut out = Vec::with_capacity(masks.len());
    let mut masses = Vec::with_capacity(masks.len());
    let mut error = 0.0_f64;
    for ((grid, log_density), mask) in populations.iter().zip(masks) {
        let exposures: Vec<f64> = mask.iter().map(|&killed|
            if killed { exposure } else { 0.0 }).collect();
        let likelihood = node_likelihood(grid, eta0, loadings, &no_counts,
            &exposures, None, Some(shift), marks, atoms, false);
        let state = condition(grid, log_density, &likelihood.ell,
            likelihood.shift, "reference population")?;
        let log_normaliser = state.log_normaliser.value();
        for i in 0..grid.size() {
            let magnitude = (0..marks).filter(|&d| mask[d]).map(|d| {
                eta0[d].value().abs() + shift[d].value().abs() + exposure.ln().abs()
                    + (0..atoms).map(|k|
                        (loadings[d * atoms + k].value() * grid.coordinate(i, k).value()).abs())
                        .sum::<f64>()
            }).fold(0.0_f64, f64::max);
            let ell = likelihood.ell[i].value();
            let joined = log_density[i].value() + ell;
            let raw = joined - likelihood.shift;
            error = error.max(
                intensity_growth * ell.abs() * (1.0 + magnitude)
                    + u * (joined.abs() + raw.abs() + (raw - log_normaliser).abs()),
            );
        }
        masses.push(state.log_normaliser.add(&eta0[0].constant_like(likelihood.shift)));
        out.push((grid.clone(), state.log_alpha));
    }
    Ok((out, masses, error))
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
    let log_density = log_standard_prior(&initial, like);
    let mut populations = vec![(initial, log_density); masks.len()];
    let mut carried = vec![like.constant_like(0.0); masks.len()];
    let mut log_normaliser = Vec::with_capacity(nodes * marks);
    let mut log_risk_mass = Vec::with_capacity(nodes * masks.len());
    for n in 0..nodes {
        log_normaliser.extend(moments(&populations, &of_mark, loadings, atoms, like)?.0);
        log_risk_mass.extend(carried.iter().cloned());
        if n + 1 == nodes { break; }
        let dt = grid.gaps[n];
        let transitions = transitions_across(rates, 0.5 * dt, time_scale)?;
        let diffuse = |populations: &[(Grid<S>, Vec<S>)]| -> Result<Vec<(Grid<S>, Vec<S>)>, EventHistoryError> {
            populations.iter().map(|(grid, log_density)|
                predict(gh, like, grid, log_density, &transitions, "reference population")).collect()
        };
        let middle = diffuse(&populations)?;
        let eta_mid: Vec<S> = (0..marks).map(|d|
            eta0[n * marks + d].add(&eta0[(n + 1) * marks + d]).scale(0.5)).collect();
        let (mut shift, _) = moments(&middle, &of_mark, loadings, atoms, like)?;
        // The midpoint shift is the fixed point of `shift ↦ moments(kill(middle, shift))`,
        // iterated until the value is resolved. Derivative channels follow every
        // iteration, so at a contraction they converge with the value. Two evaluations
        // of the map agree only to their pairwise rounding `2·band`: below it they have
        // converged, and a contraction ratio formed from changes inside it is noise.
        // Above it, a ratio ≥ 1 is a map that does not contract at this step length,
        // the one refusal, which a finer reference grid answers. A ratio below 1
        // stops once the geometric remainder `change·q/(1 − q)` is within the band.
        let mut previous: Option<f64> = None;
        loop {
            let (selected, _, conditioning) =
                kill(&middle, &masks, &eta_mid, loadings, &shift, 0.5 * dt, atoms)?;
            let (next, moment_error) = moments(&selected, &of_mark, loadings, atoms, like)?;
            let change = shift.iter().zip(&next).map(|(a, b)|
                (a.value() - b.value()).abs()).fold(0.0, f64::max);
            shift = next;
            if !change.is_finite() {
                return Err(EventHistoryError::NumericalFailure {
                    reason: format!("reference midpoint on interval {n}: non-finite change {change}"),
                });
            }
            // One evaluation's rounding: its own moment error and twice the
            // conditioned point error each moment propagates (`moments`, `kill`).
            let band = moment_error + 2.0 * conditioning;
            if change <= 2.0 * band { break; }
            if let Some(prior) = previous
                && prior > 2.0 * band
            {
                let contraction = change / prior;
                if contraction >= 1.0 {
                    return Err(EventHistoryError::ReferenceStep {
                        interval: n, change, contraction, band,
                    });
                }
                if change * contraction / (1.0 - contraction) <= band { break; }
            }
            previous = Some(change);
        }
        let (selected, masses, _) = kill(&middle, &masks, &eta_mid, loadings, &shift, dt, atoms)?;
        for (mass, increment) in carried.iter_mut().zip(masses) { *mass = mass.add(&increment); }
        populations = diffuse(&selected)?;
    }
    Ok(Normalisers { log_normaliser, log_risk_mass, masks: masks.len() })
}
