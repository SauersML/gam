//! Posterior-predictive forecasts of the joint model with latent signatures (#2961).
//!
//! For a person alive at the assessment time `s` with history `H_s`, a forecast returns at every
//! horizon the terminal survival `P(T_dagger > s+u | H_s)` and, for every mark still at risk, the
//! probability of its next occurrence before termination `P(s < T_d <= s+u, T_d < T_dagger | H_s)`.
//! The rank-zero law has its exact route in `constant_rate_inference.rs`; this module serves
//! models with signatures.
//!
//! A coefficient draw `theta_j` carries its own reference evolution and its own
//! history-conditioned state law. Other diagnoses occur in the future and jump the state, death
//! stops the trajectory, and unobserved future measurements integrate to one. The final
//! probabilities are averaged over the draws with the weights updated by `L(H_s | theta_j)`
//! (`coefficient_prediction.rs`), never evaluated at averaged coefficients (SPEC 3, #2964).
//!
//! # Particle steps
//!
//! One run per coefficient draw is killed by the terminal marks. Its weighted particles move
//! through the reference kernel's steps, with the draw's reference normaliser `M_d(t_mid)`. Within
//! one step every hazard is frozen at the step midpoint, and the kernel returns each particle's
//! integrated killing hazards `h_{p,e} = lambda_{p,e}(t_mid) dt`. With `H_p = sum_e h_{p,e}`,
//!
//! ```text
//! S_p(t1) = S_p(t0) exp(-H_p)
//! integral_{t0}^{t1} S_p(t) lambda_{p,e} dt = S_p(t0) h_{p,e} (1 - exp(-H_p)) / H_p,
//! ```
//!
//! so a particle's killing increments sum to its survival decrement exactly. These are exact
//! only where the rates are constant within the step. The integrator's comparison of a cell
//! against its halves checks the rest.
//!
//! Every other at-risk mark's next occurrence comes from the same run. Up to a mark's first
//! event after `s` the dynamics equal those of a run that mark also kills, so its sub-density is
//! `E[S(t) lambda_d(t) 1{d has not fired since s}]`. With the decoder's linear coefficients
//! `pi_d0, pi_dk` and the step's scale `w_d = exp(b_d - m_d)`, a step contributes
//!
//! ```text
//! w_d [pi_d0 sum_p m_p r_pd + sum_k pi_dk sum_p m_p r_pd softplus(x_pk)],
//! ```
//!
//! with `m_p = S_p(t0) (1 - exp(-H_p)) / H_p` the particle's occupancy of the step and `r_pd`
//! whether it has not fired `d`. The full channel sums cost `O(P K)` per step. A (mark, channel)
//! pair whose fired particles hold at most half its mass costs only its fired particles, while one
//! whose fired particles hold more rescans all `P` particles. So the worst case, many marks fired by
//! most of the mass, is `O(P D (K + 1))`, and the common case of rare firings is `O(P K + fired)`.
//!
//! The kernel's `event_step` returns the killing hazards untouched. It adds only the retained
//! non-killing event's importance ratio to a particle's log weight, never renormalizes a set,
//! and updates the particle's risk set when a once-only mark fires. So survival is applied here,
//! once, and the particles keep absolute weights.

use super::coefficient_prediction::sum;
use crate::EventHistoryError;

fn numerical(reason: impl Into<String>) -> EventHistoryError {
    EventHistoryError::NumericalFailure {
        reason: reason.into(),
    }
}

/// One killed run's weighted particles across a cell, relative to the run's survival at the cell
/// start: `log_mass[p] = log(w_p S_p(t) / S(a))`, where `w_p` are the particles' absolute weights.
pub struct ParticleRun {
    log_mass: Vec<f64>,
    /// Whether each mark kills the run.
    exposed: Vec<bool>,
    /// Per mark, the terms of `integral_a^t S(t')/S(a) E[lambda_e | alive] dt'`.
    terms: Vec<Vec<f64>>,
}

impl ParticleRun {
    /// Open a cell from every particle's absolute log weight times its survival so far.
    pub fn opening(log_weights: &[f64], exposed: Vec<bool>) -> Result<Self, EventHistoryError> {
        let refuse = || numerical("a forecast cell needs a population with finite total weight");
        if log_weights.is_empty() {
            return Err(refuse());
        }
        let total = crate::chain::log_sum_exp(log_weights);
        if !total.is_finite() {
            return Err(refuse());
        }
        let marks = exposed.len();
        Ok(Self {
            log_mass: log_weights.iter().map(|w| w - total).collect(),
            exposed,
            terms: vec![Vec::new(); marks],
        })
    }

    /// Carry the run across one kernel step. `log_hazard[offsets[p]..offsets[p + 1]]` holds
    /// particle `p`'s `log(lambda_e(t_mid) dt)` for the marks that kill the run and that `p` is at
    /// risk for (`risk[p]`), in mark order, as the kernel's `EventStep` returns them.
    pub fn step(
        &mut self,
        log_hazard: &[f64],
        offsets: &[usize],
        risk: &[Vec<bool>],
    ) -> Result<(), EventHistoryError> {
        let particles = self.log_mass.len();
        let marks = self.exposed.len();
        if offsets.len() != particles + 1
            || risk.len() != particles
            || offsets.last() != Some(&log_hazard.len())
            || log_hazard.iter().any(|h| h.is_nan())
        {
            return Err(numerical(
                "kernel hazards need one row per particle with finite or -inf log hazards",
            ));
        }
        for (p, mass) in self.log_mass.iter_mut().enumerate() {
            let killing: Vec<usize> = (0..marks)
                .filter(|&d| self.exposed[d] && risk[p].get(d) == Some(&true))
                .collect();
            if risk[p].len() != marks
                || offsets[p] > offsets[p + 1]
                || offsets[p + 1] - offsets[p] != killing.len()
            {
                return Err(numerical(
                    "a particle's hazard row does not match its killing marks at risk",
                ));
            }
            let row = &log_hazard[offsets[p]..offsets[p + 1]];
            let total = if row.is_empty() {
                0.0
            } else {
                crate::chain::log_sum_exp(row).exp()
            };
            // (1 - exp(-H)) / H, which is one at H = 0.
            let occupancy = gam_math::special::log_exprel(-total);
            for (&d, &h) in killing.iter().zip(row) {
                if h > f64::NEG_INFINITY {
                    self.terms[d].push((*mass + h + occupancy).exp());
                }
            }
            *mass -= total;
        }
        Ok(())
    }

    /// Add the kernel's retained-event log weights, `log target - log proposal`, per particle.
    pub fn reweight(&mut self, log_ratios: &[f64]) -> Result<(), EventHistoryError> {
        if log_ratios.len() != self.log_mass.len() || log_ratios.iter().any(|r| r.is_nan()) {
            return Err(numerical("one retained-event log weight per particle"));
        }
        for (mass, ratio) in self.log_mass.iter_mut().zip(log_ratios) {
            *mass += ratio;
        }
        Ok(())
    }

    /// The cell's log survival decrement and sub-density integral per mark.
    pub fn finish(&self) -> Result<(f64, Vec<f64>), EventHistoryError> {
        let log_decrement = crate::chain::log_sum_exp(&self.log_mass);
        let sub_densities: Vec<f64> = self.terms.iter().map(|t| sum(t.iter().copied())).collect();
        if log_decrement.is_nan() || sub_densities.iter().any(|v| !v.is_finite()) {
            return Err(numerical("a particle forecast cell is not representable"));
        }
        Ok((log_decrement, sub_densities))
    }
}

/// Per mark, the occupancy-weighted channel sums over the particles that have not fired that mark:
/// `[sum_p m_p r_pd, sum_p m_p r_pd softplus(x_p1), ..., sum_p m_p r_pd softplus(x_pK)]`.
///
/// `log_occupancy[p] = log m_p`, `softplus` is `particles x signatures`, and `fired[p]` lists the
/// marks particle `p` has fired since the assessment time. A channel is the full sum minus the
/// fired particles' part when that part `R` is no larger than what remains. The subtraction's
/// absolute rounding `eps (F + R)` is then at most `3 eps (F - R)`, a fixed count of the direct
/// route's. Otherwise the channel is summed directly over the particles that have not fired, so
/// no channel is formed by cancelling most of its own mass.
pub fn masked_mark_sums(
    log_occupancy: &[f64],
    softplus: &[f64],
    signatures: usize,
    fired: &[Vec<usize>],
    marks: usize,
) -> Result<Vec<Vec<f64>>, EventHistoryError> {
    let particles = log_occupancy.len();
    if fired.len() != particles
        || softplus.len() != particles * signatures
        || softplus.iter().any(|s| !s.is_finite() || *s < 0.0)
        || log_occupancy.iter().any(|m| m.is_nan() || *m == f64::INFINITY)
        || fired.iter().flatten().any(|&d| d >= marks)
    {
        return Err(numerical(
            "masked mark sums need one occupancy, one softplus row and one fired list per particle, with known marks",
        ));
    }
    let channels = signatures + 1;
    let occupancy: Vec<f64> = log_occupancy.iter().map(|m| m.exp()).collect();
    // Channel 0 is the intercept; channel k + 1 is signature axis k.
    let channel = |p: usize, c: usize| -> f64 {
        if c == 0 {
            occupancy[p]
        } else {
            occupancy[p] * softplus[p * signatures + c - 1]
        }
    };
    let full: Vec<f64> = (0..channels)
        .map(|c| sum((0..particles).map(|p| channel(p, c))))
        .collect();
    let mut by_mark: Vec<Vec<usize>> = vec![Vec::new(); marks];
    for (p, particle_fired) in fired.iter().enumerate() {
        let mut distinct = particle_fired.clone();
        distinct.sort_unstable();
        distinct.dedup();
        for d in distinct {
            by_mark[d].push(p);
        }
    }
    let mut fired_particle = vec![false; particles];
    let mut out = Vec::with_capacity(marks);
    for members in &by_mark {
        let mut sums = Vec::with_capacity(channels);
        for c in 0..channels {
            let removed = sum(members.iter().map(|&p| channel(p, c)));
            let remaining = full[c] - removed;
            if removed <= remaining {
                sums.push(remaining.max(0.0));
            } else {
                for &p in members {
                    fired_particle[p] = true;
                }
                sums.push(sum((0..particles).filter(|&p| !fired_particle[p]).map(|p| channel(p, c))));
                for &p in members {
                    fired_particle[p] = false;
                }
            }
        }
        out.push(sums);
    }
    Ok(out)
}

/// Every mark's step sub-density `w_d [pi_d0 S_d0 + sum_k pi_dk S_dk]` from its masked sums, with
/// `log_scale[d] = b_d - m_d` and `rows[d] = [pi_d0, pi_d1, ..., pi_dK]`, all nonnegative.
pub fn reported_sub_densities(
    log_scale: &[f64],
    rows: &[Vec<f64>],
    masked: &[Vec<f64>],
) -> Result<Vec<f64>, EventHistoryError> {
    if rows.len() != log_scale.len()
        || masked.len() != log_scale.len()
        || rows.iter().zip(masked).any(|(r, m)| r.len() != m.len())
        || rows.iter().flatten().any(|pi| !pi.is_finite() || *pi < 0.0)
        || log_scale.iter().any(|s| s.is_nan() || *s == f64::INFINITY)
    {
        return Err(numerical(
            "reported sub-densities need one log scale, one nonnegative coefficient row and one masked sum per mark",
        ));
    }
    let values: Vec<f64> = log_scale
        .iter()
        .zip(rows)
        .zip(masked)
        .map(|((scale, row), sums)| scale.exp() * sum(row.iter().zip(sums).map(|(pi, s)| pi * s)))
        .collect();
    if values.iter().any(|v| !v.is_finite()) {
        return Err(numerical("a reported sub-density is not representable"));
    }
    Ok(values)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Rounding band of an accumulated quantity: one ulp per rounded term plus one per
    /// transform, doubled for the independent oracle, at the largest magnitude formed.
    fn band(terms: usize, scale: f64) -> f64 {
        2.0 * (terms + 1) as f64 * f64::EPSILON * scale.abs().max(1.0)
    }

    /// The kernel's ragged layout for particles at risk for every killing mark.
    fn ragged(rates: &[Vec<f64>], exposed: &[bool], dt: f64) -> (Vec<f64>, Vec<usize>, Vec<Vec<bool>>) {
        let mut log_hazard = Vec::new();
        let mut offsets = vec![0];
        for particle in rates {
            for (d, &rate) in particle.iter().enumerate() {
                if exposed[d] {
                    log_hazard.push((rate * dt).ln());
                }
            }
            offsets.push(log_hazard.len());
        }
        (log_hazard, offsets, vec![vec![true; exposed.len()]; rates.len()])
    }

    #[test]
    fn constant_rates_are_exact_across_uneven_steps_and_killing_mass_is_conserved() {
        // Marks 0 and 1 kill the run; mark 2 does not, so the kernel returns no hazard for it.
        let rates = vec![vec![0.3_f64, 0.5, 0.2]];
        let exposed = vec![true, true, false];
        let mut run = ParticleRun::opening(&[0.0], exposed.clone()).unwrap();
        let steps = [0.25_f64, 1.0, 0.125, 2.5];
        for &dt in &steps {
            let (log_hazard, offsets, risk) = ragged(&rates, &exposed, dt);
            run.step(&log_hazard, &offsets, &risk).unwrap();
        }
        let (log_decrement, sub) = run.finish().unwrap();
        let t: f64 = steps.iter().sum();
        let dead = -(-0.8 * t).exp_m1();
        assert!((log_decrement + 0.8 * t).abs() <= band(steps.len(), 0.8 * t));
        for d in 0..2 {
            let exact = rates[0][d] / 0.8 * dead;
            assert!((sub[d] - exact).abs() <= band(steps.len(), 1.0), "mark {d}: {} vs {exact}", sub[d]);
        }
        assert_eq!(sub[2], 0.0);
        let killed = sub[0] + sub[1];
        assert!((killed - (1.0 - log_decrement.exp())).abs() <= band(2 * steps.len(), 1.0));
    }

    #[test]
    fn a_weighted_mixture_matches_its_exact_mixture_and_follows_risk_and_reweighting() {
        let weights = [0.25_f64, 0.75];
        let rates = vec![vec![0.4_f64, 0.1], vec![1.5, 0.6]];
        let exposed = vec![true, true];
        let log_weights: Vec<f64> = weights.iter().map(|w| (7.0 * w).ln()).collect();
        let mut run = ParticleRun::opening(&log_weights, exposed.clone()).unwrap();
        let dt = 0.3;
        for n in 0..4 {
            let (log_hazard, offsets, risk) = ragged(&rates, &exposed, dt);
            run.step(&log_hazard, &offsets, &risk).unwrap();
            assert_eq!(run.terms[0].len(), 2 * (n + 1));
        }
        let (log_decrement, sub) = run.finish().unwrap();
        let t = 1.2_f64;
        let total = |r: &Vec<f64>| r[0] + r[1];
        let survival: f64 = weights.iter().zip(&rates).map(|(w, r)| w * (-total(r) * t).exp()).sum();
        assert!((log_decrement - survival.ln()).abs() <= band(8, survival.ln()));
        for d in 0..2 {
            let exact: f64 = weights
                .iter()
                .zip(&rates)
                .map(|(w, r)| w * r[d] / total(r) * -(-total(r) * t).exp_m1())
                .sum();
            assert!((sub[d] - exact).abs() <= band(8, 1.0), "mark {d}: {} vs {exact}", sub[d]);
        }
        // The second particle leaves risk for mark 1 after the first step: its row shrinks to mark 0,
        // and it contributes to mark 1 only over that first step.
        let mut once = ParticleRun::opening(&log_weights, exposed.clone()).unwrap();
        let (log_hazard, offsets, risk) = ragged(&rates, &exposed, dt);
        once.step(&log_hazard, &offsets, &risk).unwrap();
        let risk = vec![vec![true, true], vec![true, false]];
        let log_hazard = vec![(0.4 * dt).ln(), (0.1 * dt).ln(), (1.5 * dt).ln()];
        once.step(&log_hazard, &[0, 2, 3], &risk).unwrap();
        let (.., sub) = once.finish().unwrap();
        let first = -(-2.1 * dt).exp_m1();
        let expected: f64 = weights[0] * 0.1 / 0.5 * -(-0.5 * 2.0 * dt).exp_m1() + weights[1] * 0.6 / 2.1 * first;
        assert!((sub[1] - expected).abs() <= band(4, 1.0), "{} vs {expected}", sub[1]);
        // A retained-event log weight of 1 on the second particle after the first step multiplies
        // everything that particle contributes from then on by e.
        let mut weighted = ParticleRun::opening(&log_weights, exposed.clone()).unwrap();
        let (log_hazard, offsets, risk) = ragged(&rates, &exposed, dt);
        weighted.step(&log_hazard, &offsets, &risk).unwrap();
        weighted.reweight(&[0.0, 1.0]).unwrap();
        weighted.step(&log_hazard, &offsets, &risk).unwrap();
        let (log_decrement, ..) = weighted.finish().unwrap();
        let expected = (weights[0] * (-0.5 * 0.6_f64).exp() + weights[1] * 1.0_f64.exp() * (-2.1 * 0.6_f64).exp()).ln();
        assert!((log_decrement - expected).abs() <= band(4, 2.0));
        // Refusals: an empty or weightless population, a row that disagrees with the risk set, and
        // a malformed reweight.
        assert!(ParticleRun::opening(&[], vec![true]).is_err());
        assert!(ParticleRun::opening(&[f64::NEG_INFINITY], vec![true]).is_err());
        assert!(weighted.step(&log_hazard, &[0, 1, 4], &risk).is_err());
        assert!(weighted.step(&[f64::NAN; 4], &offsets, &risk).is_err());
        assert!(weighted.step(&log_hazard, &offsets, &risk[..1]).is_err());
        assert!(weighted.reweight(&[0.0]).is_err());
    }

    #[test]
    fn masked_sums_and_reported_sub_densities_match_the_per_particle_per_mark_rescan() {
        // Six particles, two signatures, four marks. Mark 1 is fired by the particles holding most of
        // the occupancy (direct route); marks 2 and 3 by a few (subtraction route); mark 0 by none.
        let log_occupancy = [0.3_f64.ln(), 0.25_f64.ln(), 0.2_f64.ln(), 0.1_f64.ln(), 0.15_f64.ln(), f64::NEG_INFINITY];
        let softplus = [0.7_f64, 1.9, 0.2, 0.4, 2.5, 0.05, 1.1, 1.1, 0.3, 2.2, 0.9, 0.6];
        let fired = vec![vec![1], vec![1, 3], vec![1, 1], vec![], vec![2], vec![1]];
        let (particles, signatures, marks) = (6, 2, 4);
        let masked = masked_mark_sums(&log_occupancy, &softplus, signatures, &fired, marks).unwrap();
        let log_scale = [-1.2_f64, 0.4, -0.3, 2.0];
        let rows = vec![vec![0.2_f64, 0.5, 0.3], vec![0.1, 0.0, 0.9], vec![0.6, 0.2, 0.2], vec![0.05, 0.45, 0.5]];
        let sub = reported_sub_densities(&log_scale, &rows, &masked).unwrap();
        for d in 0..marks {
            let not_fired = |p: usize| !fired[p].contains(&d);
            for c in 0..=signatures {
                let oracle: f64 = (0..particles)
                    .filter(|&p| not_fired(p))
                    .map(|p| log_occupancy[p].exp() * if c == 0 { 1.0 } else { softplus[p * signatures + c - 1] })
                    .sum();
                let scale: f64 = (0..particles)
                    .map(|p| log_occupancy[p].exp() * if c == 0 { 1.0 } else { softplus[p * signatures + c - 1] })
                    .sum();
                assert!((masked[d][c] - oracle).abs() <= 3.0 * band(particles, scale), "mark {d} channel {c}");
            }
            let rescan: f64 = (0..particles)
                .filter(|&p| not_fired(p))
                .map(|p| {
                    let activity = rows[d][0]
                        + (0..signatures).map(|k| rows[d][k + 1] * softplus[p * signatures + k]).sum::<f64>();
                    log_occupancy[p].exp() * log_scale[d].exp() * activity
                })
                .sum();
            // The largest magnitude either route forms is the scale times the unmasked channel sums.
            let unmasked: f64 = (0..=signatures)
                .map(|c| {
                    rows[d][c]
                        * (0..particles)
                            .map(|p| log_occupancy[p].exp() * if c == 0 { 1.0 } else { softplus[p * signatures + c - 1] })
                            .sum::<f64>()
                })
                .sum();
            let largest = log_scale[d].exp() * unmasked;
            assert!((sub[d] - rescan).abs() <= 3.0 * band(particles * (signatures + 1), largest), "mark {d}");
        }
        // Positive control for the route choice: one particle fired the mark and holds 10^16 of the
        // occupancy against 1. The naive full-minus-fired subtraction misses the remaining mass by
        // far more than the rounding band the masked sum achieves at the result's scale, and the
        // masked sum recovers it within that band.
        let heavy = [1e16_f64.ln(), 0.0];
        let heavy_masked = masked_mark_sums(&heavy, &[1.0, 1.0], 1, &[vec![0], vec![]], 1).unwrap();
        let naive = (1e16_f64 + 1.0) - 1e16;
        assert!((naive - 1.0).abs() > band(2, 1.0));
        assert!((heavy_masked[0][0] - 1.0).abs() <= band(2, 1.0));
        assert!((heavy_masked[0][1] - 1.0).abs() <= band(2, 1.0));
        // Refusals: an unknown mark, a malformed softplus row, a negative coefficient.
        assert!(masked_mark_sums(&log_occupancy, &softplus, signatures, &[vec![4], vec![], vec![], vec![], vec![], vec![]], marks).is_err());
        assert!(masked_mark_sums(&log_occupancy, &softplus[..10], signatures, &fired, marks).is_err());
        let mut negative = rows.clone();
        negative[0][1] = -0.1;
        assert!(reported_sub_densities(&log_scale, &negative, &masked).is_err());
    }
}
