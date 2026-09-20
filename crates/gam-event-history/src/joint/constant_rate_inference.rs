//! Exact coefficient integration for constant counting-process rates (#2961).
//!
//! Every mark's rate has the prior `r_d ~ Exponential(c)` at one shared
//! strength `c`, a rate in the data's time unit, so
//!
//! ```text
//! p(data | c) = product_d c Gamma(y_d+1) / (E_d+c)^(y_d+1)
//! r_d | data, c ~ Gamma(shape = y_d+1, rate = E_d+c)
//! ```
//!
//! and `c` maximizes this evidence. An exposed mark adds `(E_d - y_d c)/(E_d+c)`
//! to the score in `log c` and `-(y_d+1) E_d c/(E_d+c)^2` to its curvature. The
//! score is strictly decreasing, so its root is unique once an event has been
//! observed, and `opt` finds it inside a data-derived bracket. An event-free
//! exposed cohort has its supremum at the boundary `c -> infinity`: the exact
//! zero-rate law, a point mass on zero rates.
//!
//! Conditioning on a new history adds its events to the shapes and its at-risk
//! exposure to the rates, at the fitted strength. A forecast averages the Gamma
//! posterior; it never inserts a fitted rate.

use super::law::{JointHistory, invalid, numerical};
use super::model::JointForecast;
use crate::{EventHistoryError, MarkKind};
use gam_math::sparse_grid::CompensatedSum;
use gam_math::special::{gauss_legendre, logaddexp, logistic, softplus};
use ndarray::Array2;
use serde::{Deserialize, Serialize};

/// The posterior Gamma law of one mark's rate.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
struct GammaRate {
    shape: f64,
    log_rate: f64,
}

/// Independent Gamma posteriors of the mark rates at the empirical-Bayes
/// strength.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub(super) struct ConstantRatePosterior {
    /// `None` is the exact zero-rate law of an event-free cohort.
    rates: Option<Vec<GammaRate>>,
}



/// The negative log evidence as a function of `z = log c - x0`.
struct Evidence {
    /// The event count and `x0 - log E_d` of every exposed mark.
    marks: Vec<(f64, f64)>,
}

impl Evidence {
    /// The first three derivatives in `z`. An exposed mark contributes
    /// `(y_d+1)p_d - 1`, `(y_d+1)p_d q_d` and `(y_d+1)p_d q_d (q_d - p_d)`, with
    /// `p_d = c/(E_d+c)` and `q_d = E_d/(E_d+c)`.
    fn derivatives(&self, z: f64) -> [f64; 3] {
        let mut sums = [CompensatedSum::default(); 3];
        for &(count, logit) in &self.marks {
            let p = logistic(logit + z);
            let q = logistic(-(logit + z));
            sums[0].add((count + 1.0) * p - 1.0);
            sums[1].add((count + 1.0) * p * q);
            sums[2].add((count + 1.0) * p * q * (q - p));
        }
        sums.map(CompensatedSum::value)
    }

    /// The rounding resolution of the gradient at `z`. Each term `(y_d+1)p_d - 1`
    /// carries the relative rounding of its two parts, one representable step
    /// in its argument `x0 + z - log E_d` moves it by its curvature times that
    /// step, and the sum over marks adds its summation floor.
    fn resolution(&self, x0: f64, z: f64) -> f64 {
        let spread = 1.0 + x0.abs() + z.abs();
        let terms: f64 = self
            .marks
            .iter()
            .map(|&(count, logit)| {
                let p = logistic(logit + z);
                let q = logistic(-(logit + z));
                (count + 1.0) * p + 1.0 + (count + 1.0) * p * q * (spread + logit.abs())
            })
            .sum();
        f64::EPSILON * terms * (self.marks.len() as f64).log2().ceil().max(1.0)
    }
}

/// `log c` at the evidence maximum, over the exposed marks.
///
/// Let `D >= 1` be the number of exposed marks, `E_min` and `E_max` their
/// smallest and largest exposures, and `Y = sum_d y_d`. An event node always
/// follows exposure in its own risk set, so every event belongs to an exposed
/// mark and `Y >= 1` here. With `p_d = c/(E_d+c)` and `q_d = 1 - p_d`, the
/// score in `log c` is `S(c) = sum_d (E_d - y_d c)/(E_d+c) = sum_d (q_d - y_d p_d)`,
/// and `dS/dlog c = -sum_d (y_d+1) p_d q_d < 0`, so `S` has at most one root.
///
/// At `c_lo = E_min/(2Y)`: `q_d = 1/(1 + c_lo/E_d) >= 1/(1 + c_lo/E_min) =
/// 2Y/(2Y+1) >= 2/3`, and `y_d p_d <= y_d c_lo/E_d <= y_d c_lo/E_min = y_d/(2Y)`,
/// so `sum_d y_d p_d <= 1/2` and `S(c_lo) >= 2D/3 - 1/2 > 0`.
///
/// At `c_hi = 2D E_max/Y`: each term is `1 - (y_d+1) c/(E_d+c)`, and
/// `E_d <= E_max` gives `(y_d+1) c/(E_d+c) >= (y_d+1) c/(E_max+c)`, so
/// `S(c) <= D - (Y+D) c/(E_max+c) = (D E_max - Y c)/(E_max+c)`. At `c_hi` that
/// bound is `-D E_max/(E_max+c_hi) < 0`.
///
/// `S` is continuous, so its unique root lies strictly inside `[c_lo, c_hi]`.
/// The search starts at `c0 = sum_d E_d / Y`, which is the root itself when
/// every exposure is equal. The refinement halves the bracket at least every
/// two iterations, so twice the halvings that reach the gradient's resolution
/// bound its iterations. The strength is accepted only where the gradient is
/// within that resolution and the curvature is positive.
fn strength(counts: &[u64], exposure: &[f64]) -> Result<f64, EventHistoryError> {
    let exposed: Vec<(f64, f64)> = counts
        .iter()
        .zip(exposure)
        .filter(|pair| *pair.1 > 0.0)
        .map(|(&count, &e)| (count as f64, e))
        .collect();
    let mut total_exposure = CompensatedSum::default();
    for &(.., e) in &exposed {
        total_exposure.add(e);
    }
    let total_count = exposed.iter().map(|mark| mark.0).sum::<f64>();
    let x0 = total_exposure.value().ln() - total_count.ln();
    let evidence = Evidence {
        marks: exposed
            .iter()
            .map(|&(count, e)| (count, x0 - e.ln()))
            .collect(),
    };
    let (smallest, largest) = exposed
        .iter()
        .fold((f64::INFINITY, 0.0_f64), |(lo, hi), mark| {
            (lo.min(mark.1), hi.max(mark.1))
        });
    let bracket = (
        (smallest / (2.0 * total_count)).ln() - x0,
        (2.0 * exposed.len() as f64 * largest / total_count).ln() - x0,
    );
    let tolerance = evidence.resolution(x0, 0.0);
    let halvings = ((bracket.1 - bracket.0) / tolerance).log2().ceil();
    let config = opt::RootConfig::new(tolerance, 0, 2 * (halvings as usize + 1));
    let oracle = |z: f64| {
        let [gradient, curvature, third] = evidence.derivatives(z);
        if [gradient, curvature, third].iter().all(|v| v.is_finite()) {
            Ok(opt::RootSample {
                value: gradient,
                d1: curvature,
                d2: third,
            })
        } else {
            Err(format!(
                "rate strength evidence derivatives are not finite at log c = {}",
                x0 + z
            ))
        }
    };
    let z = opt::find_root_monotone(oracle, 0.0, &config, Some(bracket))
        .map_err(|error| numerical(format!("rate strength root solve failed: {error:?}")))?
        .root;
    let [gradient, curvature, ..] = evidence.derivatives(z);
    if !(gradient.abs() <= evidence.resolution(x0, z) && curvature > 0.0) {
        return Err(numerical(format!(
            "rate strength evidence is not stationary at log c = {}: gradient {gradient}, curvature {curvature}",
            x0 + z
        )));
    }
    Ok(x0 + z)
}

impl ConstantRatePosterior {
    /// The rate posterior of a cohort at its evidence-maximizing strength.
    pub(super) fn infer(
        marks: &[MarkKind],
        histories: impl Iterator<Item = Result<JointHistory, EventHistoryError>>,
    ) -> Result<Self, EventHistoryError> {
        let mut counts = vec![0u64; marks.len()];
        let mut exposure = vec![CompensatedSum::default(); marks.len()];
        for history in histories {
            let (history_counts, history_exposure) = history?.rate_statistics(marks);
            for d in 0..marks.len() {
                counts[d] += history_counts[d];
                exposure[d].add(history_exposure[d]);
            }
        }
        let exposure: Vec<f64> = exposure.into_iter().map(CompensatedSum::value).collect();
        if exposure.iter().any(|e| !e.is_finite()) {
            return Err(numerical(
                "cohort risk exposure exceeds floating-point range",
            ));
        }
        if exposure.iter().all(|&e| e == 0.0) {
            return Err(invalid(
                "no mark has risk exposure, so the rate strength is unidentified",
            ));
        }
        if counts.iter().all(|&n| n == 0) {
            return Ok(Self { rates: None });
        }
        let log_c = strength(&counts, &exposure)?;
        Ok(Self {
            rates: Some(
                counts
                    .iter()
                    .zip(&exposure)
                    .map(|(&n, &e)| GammaRate {
                        shape: n as f64 + 1.0,
                        log_rate: logaddexp(e.ln(), log_c),
                    })
                    .collect(),
            ),
        })
    }

    /// The posterior given one more history: its events join the shapes and
    /// its at-risk exposure joins the rates.
    pub(super) fn condition(
        &self,
        marks: &[MarkKind],
        history: &JointHistory,
    ) -> Result<Self, EventHistoryError> {
        let (counts, exposure) = history.rate_statistics(marks);
        let Some(rates) = &self.rates else {
            if counts.iter().any(|&n| n > 0) {
                return Err(invalid(
                    "a history with an event has probability zero under the zero-rate law of an event-free cohort",
                ));
            }
            return Ok(self.clone());
        };
        Ok(Self {
            rates: Some(
                rates
                    .iter()
                    .zip(counts.iter().zip(&exposure))
                    .map(|(rate, (&n, &e))| GammaRate {
                        shape: rate.shape + n as f64,
                        log_rate: logaddexp(e.ln(), rate.log_rate),
                    })
                    .collect(),
            ),
        })
    }

    /// Refuse a saved posterior this law cannot hold.
    pub(super) fn validate(&self, marks: &[MarkKind]) -> Result<(), EventHistoryError> {
        match &self.rates {
            None => Ok(()),
            Some(rates)
                if rates.len() == marks.len()
                    && rates.iter().all(|rate| {
                        rate.shape >= 1.0 && rate.shape.fract() == 0.0 && rate.log_rate.is_finite()
                    }) =>
            {
                Ok(())
            }
            Some(_) => Err(invalid(
                "a rate posterior needs one Gamma law per mark, each with an integer shape of at least one and a finite log rate",
            )),
        }
    }

    /// Posterior-predictive absolute risks after a history whose open risk set
    /// is `at_risk`: at every horizon `u`, the survival `P(T_dagger > s+u | H_s)`
    /// and the incidence `P(s < T_d <= s+u, T_d < T_dagger | H_s)` of every
    /// mark's next occurrence.
    pub(super) fn forecast(
        &self,
        marks: &[MarkKind],
        at_risk: &[bool],
        horizons: &[f64],
    ) -> Result<JointForecast, EventHistoryError> {
        if horizons.iter().any(|u| !u.is_finite() || *u < 0.0) {
            return Err(invalid("forecast horizons must be finite and nonnegative"));
        }
        let dimensions = (horizons.len(), marks.len());
        let mut forecast = JointForecast {
            horizons: horizons.to_vec(),
            survival: vec![1.0; horizons.len()],
            incidence: Array2::zeros(dimensions),
            incidence_error: Array2::zeros(dimensions),
        };
        let Some(rates) = &self.rates else {
            return Ok(forecast);
        };
        let terminal: Vec<usize> = (0..marks.len())
            .filter(|&d| marks[d] == MarkKind::Terminal)
            .collect();
        let mut rules = GaussLegendreRules::default();
        for (h, &u) in horizons.iter().enumerate() {
            if u == 0.0 {
                continue;
            }
            forecast.survival[h] = log_survival(rates, &terminal, u).exp();
            for d in (0..marks.len()).filter(|&d| at_risk[d]) {
                let causes: Vec<usize> = std::iter::once(d)
                    .chain(terminal.iter().copied().filter(|&j| j != d))
                    .collect();
                let (value, error) = incidence(rates, &causes, u, &mut rules);
                forecast.incidence[[h, d]] = value;
                forecast.incidence_error[[h, d]] = error;
            }
        }
        Ok(forecast)
    }
}

/// `log G(t) = -sum_j a_j log(1 + t/b_j)`: the log probability that none of
/// `causes` fires by `t` under their Lomax predictive laws.
fn log_survival(rates: &[GammaRate], causes: &[usize], t: f64) -> f64 {
    -causes
        .iter()
        .map(|&j| rates[j].shape * softplus(t.ln() - rates[j].log_rate))
        .sum::<f64>()
}

/// Gauss-Legendre rules on `[-1, 1]` by order, computed once per forecast.
#[derive(Default)]
struct GaussLegendreRules(Vec<(usize, Vec<f64>, Vec<f64>)>);

impl GaussLegendreRules {
    fn rule(&mut self, order: usize) -> (&[f64], &[f64]) {
        let index = match self.0.iter().position(|rule| rule.0 == order) {
            Some(index) => index,
            None => {
                let (nodes, weights) = gauss_legendre(order);
                self.0.push((order, nodes, weights));
                self.0.len() - 1
            }
        };
        let rule = &self.0[index];
        (&rule.1, &rule.2)
    }
}

/// `P(T_d <= u, T_d before every other cause)` for independent Lomax predictive
/// times, where `causes[0] = d`: the integral over `(0, u]` of
/// `f_d(t) = a_d/(b_d+t) G(t)`. Returns the value and a bound on its error.
///
/// When every cause has the same rate `b`, the integral is `a_d/sum_j a_j
/// (1 - G(u))`. Otherwise `f_d` and the other causes' `f_rest` are completely
/// monotone, so every Gauss-Legendre sum underestimates its integral (the error
/// term has the sign of an even derivative), while `f_d + f_rest = -G'`
/// integrates to `1 - G(u)` exactly. Together they bracket the incidence:
/// `[GL(f_d), 1 - G(u) - GL(f_rest)]`. The panels `b_min (2^k - 1)` keep each
/// panel as long as its distance from the nearest pole, and the order doubles
/// until the bracket is within its rounding band or stops shrinking. The
/// midpoint is returned with half the bracket plus that band.
fn incidence(
    rates: &[GammaRate],
    causes: &[usize],
    u: f64,
    rules: &mut GaussLegendreRules,
) -> (f64, f64) {
    let d = causes[0];
    let complement = -log_survival(rates, causes, u).exp_m1();
    if causes
        .iter()
        .all(|&j| rates[j].log_rate == rates[d].log_rate)
    {
        let total: f64 = causes.iter().map(|&j| rates[j].shape).sum();
        return (rates[d].shape / total * complement, 0.0);
    }
    let nearest_pole = causes
        .iter()
        .map(|&j| rates[j].log_rate)
        .fold(f64::INFINITY, f64::min)
        .exp();
    let mut breaks = vec![0.0];
    let mut width = nearest_pole;
    while width > 0.0 && breaks[breaks.len() - 1] + width < u {
        breaks.push(breaks[breaks.len() - 1] + width);
        width *= 2.0;
    }
    breaks.push(u);
    let hazard = |j: usize, t: f64| rates[j].shape * (-logaddexp(rates[j].log_rate, t.ln())).exp();
    let mut order = 1;
    let mut previous = f64::INFINITY;
    loop {
        let (nodes, weights) = rules.rule(order);
        let mut lower = CompensatedSum::default();
        let mut rest = CompensatedSum::default();
        for panel in breaks.windows(2) {
            let half = 0.5 * (panel[1] - panel[0]);
            let middle = 0.5 * (panel[0] + panel[1]);
            for (&x, &w) in nodes.iter().zip(weights) {
                let t = middle + half * x;
                let weight = half * w * log_survival(rates, causes, t).exp();
                lower.add(weight * hazard(d, t));
                for &j in &causes[1..] {
                    rest.add(weight * hazard(j, t));
                }
            }
        }
        let lower = lower.value();
        let upper = complement - rest.value();
        let bracket = upper - lower;
        let terms = causes.len() * order * (breaks.len() - 1);
        let band = f64::EPSILON * (terms as f64 + 1.0) * complement;
        if bracket <= band || bracket >= previous {
            return (0.5 * (lower + upper), 0.5 * bracket.abs() + band);
        }
        previous = bracket;
        order *= 2;
    }
}

#[cfg(test)]
mod tests {
    use super::super::data::{FrozenJointSchema, JointTables};
    use super::super::model::rank_zero_declarations;
    use super::*;

    /// The rank-zero histories of `subjects` as `(entry, exit, events)`, events
    /// as `(time, mark index)`, encoded through the production encoder with the
    /// declared marks, as `fit_joint_event_model` encodes them. Returns the
    /// marks' kinds and one history per subject.
    fn histories(
        marks: &[(&str, MarkKind)],
        subjects: &[(f64, f64, &[(f64, usize)])],
    ) -> Result<(Vec<MarkKind>, Vec<JointHistory>), EventHistoryError> {
        let declared: Vec<(String, MarkKind)> = marks
            .iter()
            .map(|&(name, kind)| (name.to_string(), kind))
            .collect();
        let mut tables = JointTables::default();
        for (i, &(entry, exit, events)) in subjects.iter().enumerate() {
            let id = format!("s{i}");
            tables.subjects.id.push(id.clone());
            tables.subjects.entry.push(entry);
            tables.subjects.exit.push(exit);
            for &(time, mark) in events {
                tables.events.id.push(id.clone());
                tables.events.time.push(time);
                tables.events.mark.push(marks[mark].0.to_string());
            }
        }
        let (schema, encoded) =
            FrozenJointSchema::fit(&rank_zero_declarations(Some(declared)), &tables)?;
        Ok((
            schema.mark_kinds,
            encoded.into_iter().map(|subject| subject.history).collect(),
        ))
    }

    const COMPETING: [(&str, MarkKind); 4] = [
        ("diagnosis", MarkKind::Once),
        ("cvd_death", MarkKind::Terminal),
        ("other_death", MarkKind::Terminal),
        ("visit", MarkKind::Recurrent),
    ];

    /// Roundoff bar of two routes to one value: `eps * S_abs * ceil(log2 n)`
    /// over the `n` summands the routes accumulate.
    fn roundoff(summands: &[f64]) -> f64 {
        f64::EPSILON
            * summands.iter().map(|s| s.abs()).sum::<f64>()
            * (summands.len() as f64).log2().ceil().max(1.0)
    }

    /// A central difference with its bar: the Richardson truncation estimate
    /// plus the roundoff of `f` over the step.
    fn central_difference(f: &dyn Fn(f64) -> f64, z: f64, roundoff: f64) -> (f64, f64) {
        let h = f64::EPSILON.cbrt();
        let difference = |step: f64| (f(z + step) - f(z - step)) / (2.0 * step);
        let (fine, coarse) = (difference(h), difference(2.0 * h));
        (fine, (coarse - fine).abs() / 3.0 + roundoff / h)
    }

    #[test]
    fn evidence_derivatives_match_the_marginal_likelihood() {
        // Marks with (y, E) = (1, 1) and (0, 3), started at c0 = sum E / sum y = 4.
        let marks = [(1.0_f64, 1.0_f64), (0.0, 3.0)];
        let x0 = 4.0_f64.ln();
        let evidence = Evidence {
            marks: marks
                .iter()
                .map(|&(count, e)| (count, x0 - e.ln()))
                .collect(),
        };
        // -log p(data | c) without constants: sum_d (y_d+1) log(E_d+c) - log c.
        let terms = |z: f64| -> Vec<f64> {
            marks
                .iter()
                .flat_map(|&(count, e)| [(count + 1.0) * (e + (x0 + z).exp()).ln(), -(x0 + z)])
                .collect()
        };
        let value = |z: f64| terms(z).iter().sum::<f64>();
        let bound: f64 = marks.iter().map(|mark| mark.0 + 2.0).sum();
        let shared = &evidence;
        let derivative = |order: usize| move |v: f64| shared.derivatives(v)[order];
        for z in [-4.0, -1.0, 0.0, 0.7, 3.0] {
            let [gradient, curvature, third] = evidence.derivatives(z);
            let (fd, bar) = central_difference(&value, z, roundoff(&terms(z)));
            assert!(
                gradient.abs() > bar && (gradient - fd).abs() <= bar,
                "z {z}: {gradient} vs {fd}, bar {bar}"
            );
            let (fd, bar) = central_difference(&derivative(0), z, roundoff(&[bound, bound]));
            assert!(
                curvature.abs() > bar && (curvature - fd).abs() <= bar,
                "z {z}: {curvature} vs {fd}, bar {bar}"
            );
            let (fd, bar) = central_difference(&derivative(1), z, roundoff(&[bound, bound]));
            assert!(
                third.abs() > bar && (third - fd).abs() <= bar,
                "z {z}: {third} vs {fd}, bar {bar}"
            );
        }
    }

    #[test]
    fn the_strength_is_the_algebraic_root_in_every_time_unit() {
        // Score (1-c)/(1+c) + 3/(3+c) = 0 gives c^2 - c - 6 = 0, so c = 3.
        for scale in [1.0, 1e100, 1e-100] {
            let log_c = strength(&[1, 0], &[scale, 3.0 * scale]).unwrap();
            let x0 = (4.0 * scale).ln();
            let evidence = Evidence {
                marks: vec![(1.0, x0 - scale.ln()), (0.0, x0 - (3.0 * scale).ln())],
            };
            let root = (3.0 * scale).ln();
            let z = root - x0;
            let [.., curvature, third] = evidence.derivatives(z);
            // The certificate places log c within the gradient resolution over
            // the curvature, plus one representable step of each log.
            let location =
                evidence.resolution(x0, z) / curvature + roundoff(&[log_c, x0, root, scale.ln()]);
            assert!(z.abs() > location && third.is_finite());
            assert!(
                (log_c - root).abs() <= location,
                "scale {scale}: {log_c} vs {root}, bar {location}"
            );
        }
    }

    #[test]
    fn incidence_bracket_matches_partial_fractions_and_the_closed_form() {
        // Once-only target Gamma(1, 5) against a terminal cause Gamma(3, 11).
        let (b_d, b_t, delta) = (5.0_f64, 11.0_f64, 6.0_f64);
        let rates = [
            GammaRate {
                shape: 1.0,
                log_rate: b_d.ln(),
            },
            GammaRate {
                shape: 3.0,
                log_rate: b_t.ln(),
            },
        ];
        // Antiderivative pieces of 1/(x^2 (x+delta)^3), by partial fractions.
        let pieces = |x: f64| {
            [
                3.0 * ((x + delta) / x).ln() / delta.powi(4),
                -1.0 / (delta.powi(3) * x),
                -2.0 / (delta.powi(3) * (x + delta)),
                -0.5 / (delta.powi(2) * (x + delta).powi(2)),
            ]
        };
        let mut rules = GaussLegendreRules::default();
        for u in [0.5, 3.0, 40.0, 1e4] {
            let parts: Vec<f64> = pieces(b_d + u)
                .into_iter()
                .chain(pieces(b_d).map(|piece| -piece))
                .map(|piece| b_d * b_t.powi(3) * piece)
                .collect();
            let oracle: f64 = parts.iter().sum();
            let oracle_error = roundoff(&parts);
            let (value, error) = incidence(&rates, &[0, 1], u, &mut rules);
            assert!(
                error > 0.0 && oracle > error + oracle_error,
                "u {u}: error {error}"
            );
            assert!(
                (value - oracle).abs() <= error + oracle_error,
                "u {u}: {value} vs {oracle}"
            );
            let survival = (b_d / (b_d + u)) * (b_t / (b_t + u)).powi(3);
            let (other, other_error) = incidence(&rates, &[1, 0], u, &mut rules);
            let complement_error = roundoff(&[other, oracle, 1.0, survival]);
            assert!(
                (other - (1.0 - survival - oracle)).abs()
                    <= other_error + oracle_error + complement_error
            );
        }
        // Equal rates have the closed form a_d / sum_j a_j (1 - G(u)).
        let equal = [
            GammaRate {
                shape: 2.0,
                log_rate: 7.0_f64.ln(),
            },
            GammaRate {
                shape: 3.0,
                log_rate: 7.0_f64.ln(),
            },
        ];
        let (value, error) = incidence(&equal, &[0, 1], 2.0, &mut rules);
        let log_g = 5.0 * (7.0_f64 / 9.0).ln();
        let expected = 0.4 * (1.0 - log_g.exp());
        assert_eq!(error, 0.0);
        assert!(
            (value - expected).abs() <= roundoff(&[value, expected, 0.4 * log_g, 0.4 * log_g, 0.4])
        );
    }

    #[test]
    fn competing_forecasts_are_the_gamma_posterior_predictive() {
        // A death of the first cause at 4, and a subject prevalent for the
        // diagnosis with two visits, censored at 6.
        let (marks, cohort) = histories(
            &COMPETING,
            &[
                (0.0, 4.0, &[(4.0, 1)]),
                (0.0, 6.0, &[(0.0, 0), (1.0, 3), (5.0, 3)]),
            ],
        )
        .unwrap();
        // The encoded cohort's sufficient statistics, re-derived from the
        // tables. Subject 0 is at risk for every mark on its one cell (0, 4] and
        // fires cvd_death at 4. Subject 1 enters prevalent for the diagnosis (its
        // event at 0 equals its entry), so that mark starts outside its risk
        // set; it is at risk for the other three marks on cells (0, 1], (1, 5],
        // (5, 6] of weights 1, 4, 1, and records two visits. Hence exposures
        // (4, 10, 10, 10) and counts (0, 1, 0, 2), every weight an exact integer.
        assert!(cohort[0].initially_at_risk[0] && !cohort[1].initially_at_risk[0]);
        let mut counts_total = [0u64; 4];
        let mut exposure_total = [0.0_f64; 4];
        for history in &cohort {
            let (counts, exposure) = history.rate_statistics(&marks);
            for d in 0..4 {
                counts_total[d] += counts[d];
                exposure_total[d] += exposure[d];
            }
        }
        assert_eq!(counts_total, [0, 1, 0, 2]);
        assert_eq!(
            exposure_total.map(f64::to_bits),
            [4.0_f64, 10.0, 10.0, 10.0].map(f64::to_bits)
        );
        let posterior = ConstantRatePosterior::infer(&marks, cohort.into_iter().map(Ok)).unwrap();
        // Exposures (4, 10, 10, 10) and counts (0, 1, 0, 2): the score
        // sum_d (E_d - y_d c)/(E_d + c) vanishes where 3c^2 - 22c - 160 = 0.
        let counts = [0.0_f64, 1.0, 0.0, 2.0];
        let exposure = [4.0_f64, 10.0, 10.0, 10.0];
        let c: f64 = (22.0 + 2404.0_f64.sqrt()) / 6.0;
        let x0 = 34.0_f64.ln() - 3.0_f64.ln();
        let evidence = Evidence {
            marks: counts
                .iter()
                .zip(exposure)
                .map(|(&count, e)| (count, x0 - e.ln()))
                .collect(),
        };
        let z = c.ln() - x0;
        let curvature = evidence.derivatives(z)[1];
        let location = evidence.resolution(x0, z) / curvature + roundoff(&[c.ln(), x0, 22.0 / 6.0]);
        assert!(z.abs() > location);
        let rates = posterior.rates.clone().unwrap();
        for d in 0..4 {
            assert_eq!(rates[d].shape, counts[d] + 1.0);
            let expected = (exposure[d] + c).ln();
            let bar =
                location * c / (exposure[d] + c) + roundoff(&[expected, exposure[d].ln(), c.ln()]);
            assert!((rates[d].log_rate - expected).abs() <= bar, "mark {d}");
        }

        // A new history adds exposure 2 to every mark.
        let (_, new) = histories(&COMPETING, &[(0.0, 2.0, &[])]).unwrap();
        let conditioned = posterior.condition(&marks, &new[0]).unwrap();
        let rates = conditioned.rates.clone().unwrap();
        assert!(rates[1].log_rate == rates[2].log_rate && rates[2].log_rate == rates[3].log_rate);
        let (b_d, b_t) = (rates[0].log_rate.exp(), rates[1].log_rate.exp());
        let delta = b_t - b_d;
        let horizons = [0.0, 0.5, 3.0, 40.0];
        let forecast = conditioned.forecast(&marks, &[true; 4], &horizons).unwrap();
        assert_eq!(forecast.survival[0], 1.0);
        assert!(forecast.incidence.row(0).iter().all(|&p| p == 0.0));
        for (h, &u) in horizons.iter().enumerate().skip(1) {
            let lomax = |shape: f64| {
                let log_g = shape * (b_t / (b_t + u)).ln();
                (
                    log_g.exp(),
                    log_g.exp() * roundoff(&[log_g, log_g, 1.0, 1.0]),
                )
            };
            // Survival: the Lomax law of the three terminal shapes. Inserting
            // the posterior mean rates instead misses it by more than its bar.
            let (survival, survival_bar) = lomax(3.0);
            assert!((forecast.survival[h] - survival).abs() <= survival_bar);
            assert!(((-3.0 * u / b_t).exp() - forecast.survival[h]).abs() > survival_bar);
            // Terminal causes and the recurrent visit share the terminal rate:
            // a_d / A (1 - G(u)).
            for (d, share, (g, g_bar)) in [
                (1, 2.0 / 3.0, lomax(3.0)),
                (2, 1.0 / 3.0, lomax(3.0)),
                (3, 0.5, lomax(6.0)),
            ] {
                let expected = share * (1.0 - g);
                let bar = share * g_bar + roundoff(&[forecast.incidence[[h, d]], share, share * g]);
                assert!(
                    (forecast.incidence[[h, d]] - expected).abs() <= bar,
                    "u {u}, mark {d}"
                );
                assert_eq!(forecast.incidence_error[[h, d]], 0.0);
            }
            // The diagnosis against both terminal causes:
            // b_d b_t^3 times the integral of (b_d+t)^-2 (b_t+t)^-3.
            let pieces = |x: f64| {
                [
                    3.0 * ((x + delta) / x).ln() / delta.powi(4),
                    -1.0 / (delta.powi(3) * x),
                    -2.0 / (delta.powi(3) * (x + delta)),
                    -0.5 / (delta.powi(2) * (x + delta).powi(2)),
                ]
            };
            let parts: Vec<f64> = pieces(b_d + u)
                .into_iter()
                .chain(pieces(b_d).map(|piece| -piece))
                .map(|piece| b_d * b_t.powi(3) * piece)
                .collect();
            let oracle: f64 = parts.iter().sum();
            let (value, error) = (forecast.incidence[[h, 0]], forecast.incidence_error[[h, 0]]);
            assert!(error > 0.0 && oracle > error + roundoff(&parts));
            assert!(
                (value - oracle).abs() <= error + roundoff(&parts),
                "u {u}: {value} vs {oracle}"
            );
        }
    }

    #[test]
    fn an_event_free_cohort_is_the_exact_zero_rate_law() {
        let marks = [("diagnosis", MarkKind::Once), ("death", MarkKind::Terminal)];
        let (kinds, cohort) = histories(&marks, &[(0.0, 3.0, &[]), (0.0, 5.0, &[])]).unwrap();
        let posterior = ConstantRatePosterior::infer(&kinds, cohort.into_iter().map(Ok)).unwrap();
        assert_eq!(posterior.rates, None);
        let forecast = posterior
            .forecast(&kinds, &[true, true], &[1.0, 1e300])
            .unwrap();
        assert_eq!(forecast.survival, [1.0, 1.0]);
        assert!(forecast.incidence.iter().all(|&p| p == 0.0));
        let (_, quiet) = histories(&marks, &[(0.0, 2.0, &[])]).unwrap();
        assert_eq!(posterior.condition(&kinds, &quiet[0]).unwrap(), posterior);
        let (_, diagnosed) = histories(&marks, &[(0.0, 2.0, &[(1.0, 0)])]).unwrap();
        let error = posterior
            .condition(&kinds, &diagnosed[0])
            .err()
            .unwrap()
            .to_string();
        assert!(error.contains("probability zero"), "{error}");
        // Everyone prevalent for the only mark: no exposure identifies the strength.
        let (prevalent, histories_prevalent) =
            histories(&[("only", MarkKind::Once)], &[(0.0, 2.0, &[(0.0, 0)])]).unwrap();
        let error =
            ConstantRatePosterior::infer(&prevalent, histories_prevalent.into_iter().map(Ok))
                .err()
                .unwrap()
                .to_string();
        assert!(error.contains("unidentified"), "{error}");
    }
}
