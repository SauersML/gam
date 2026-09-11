//! Conditional continuation densities use a ratio of JOINT integrals.
//! Earlier outcomes update coefficient weights as well as the latent law.
use super::*;

fn check_prefix(prefix: &JointHistory, extended: &JointHistory) -> Result<(), EventHistoryError> {
    let n = prefix.times.len();
    if extended.times.len() < n
        || prefix.times != extended.times[..n]
        || prefix.exposure != extended.exposure[..n]
        || prefix.events != extended.events[..n]
        || prefix.initially_at_risk != extended.initially_at_risk
        || prefix.entry_design != extended.entry_design
        || prefix.genetics != extended.genetics
        || prefix.baseline_design != extended.baseline_design.slice(ndarray::s![..n, ..])
        || prefix.drive_design != extended.drive_design.slice(ndarray::s![..n - 1, ..])
    {
        return Err(invalid(
            "conditional prediction requires an unchanged prefix grid, exposure, events, designs, entry state, and genetics",
        ));
    }
    let mut past = extended.measurements.iter().filter(|m| m.node < n);
    for a in &prefix.measurements {
        let Some(b) = past.next() else {
            return Err(invalid("conditional prediction removed a past measurement"));
        };
        if a.node != b.node
            || a.channel != b.channel
            || a.value != b.value
            || a.after_event != b.after_event
        {
            return Err(invalid("conditional prediction changed a past measurement"));
        }
    }
    if past.next().is_some() {
        return Err(invalid(
            "conditional prediction inserted a measurement into the past",
        ));
    }
    Ok(())
}

fn normalized(
    log_weights: &[f64],
    likelihoods: &[f64],
) -> Result<(f64, Vec<f64>), EventHistoryError> {
    let terms: Vec<_> = log_weights
        .iter()
        .zip(likelihoods)
        .map(|(w, l)| w + l)
        .collect();
    let total = log_sum_exp(&terms);
    if !total.is_finite() {
        return Err(numerical("conditional predictive integral is unresolved"));
    }
    let mut weights: Vec<_> = terms.iter().map(|v| (v - total).exp()).collect();
    let sum_weights = sum(weights.iter().copied());
    for w in &mut weights {
        *w /= sum_weights;
    }
    Ok((total, weights))
}

fn conditional_mixture(
    log_weights: &[f64],
    prefix: &[f64],
    extended: &[f64],
    inner_error: f64,
    options: &PredictiveDensityOptions,
) -> Result<PredictiveHistoryDensity, EventHistoryError> {
    let (denominator, before) = normalized(log_weights, prefix)?;
    let (numerator, after) = normalized(log_weights, extended)?;
    let n = log_weights.len() as f64;
    let se = before
        .iter()
        .zip(&after)
        .fold(0.0_f64, |s, (a, b)| s.hypot(a - b))
        * (n / (n - 1.0)).sqrt();
    // BOTH conditioned integrals need adequate coverage. Checking only the
    // final numerator could hide a missed part of the conditioning history.
    let ess = (1.0 / sum(before.iter().map(|w| w * w))).min(1.0 / sum(after.iter().map(|w| w * w)));
    finish(
        numerator - denominator,
        se,
        options.standard_error_multiplier * se + inner_error,
        ess,
        options,
    )
}

impl JointCohortIntegration<'_, '_> {
    /// Density of the newly appended outcomes conditional on the earlier
    /// histories and the training data. The paired subjects must have the
    /// same order, stratum and exact prefix mesh/data. A genuine continuation
    /// keeps entry fixed; opening a new stationary prior at the cutoff would
    /// define a different model.
    ///
    /// Computes E[L(extended)|training] / E[L(prefix)|training]. It does not
    /// average L(extended)/L(prefix) under unchanged training weights.
    /// With no appended events or observed measurements, this is the
    /// probability of no event of ANY mark over the appended windows.
    /// It is not terminal survival with other event marks marginalized out.
    pub fn conditional_history_density(
        &self,
        inference: &JointCoefficientInference<'_, '_>,
        prefix: &JointCohortIntegration<'_, '_>,
        extended: &JointCohortIntegration<'_, '_>,
        accuracy: &IntegrationAccuracy,
        tolerance: &CohortScoreTolerance,
        options: &PredictiveDensityOptions,
    ) -> Result<PredictiveHistoryDensity, EventHistoryError> {
        self.validate_prediction(inference, prefix, options)?;
        self.validate_prediction(inference, extended, options)?;
        if prefix.subjects.len() != extended.subjects.len() {
            return Err(invalid(
                "conditional histories require the same paired subjects",
            ));
        }
        for (i, (a, b)) in prefix.subjects.iter().zip(extended.subjects).enumerate() {
            if !std::ptr::eq(
                &prefix.references[prefix.strata[i]],
                &extended.references[extended.strata[i]],
            ) {
                return Err(invalid(
                    "conditional prediction changed a subject's reference stratum",
                ));
            }
            check_prefix(a.history, b.history)?;
        }
        let no_outcomes = prefix.subjects.iter().zip(extended.subjects).all(|(a, b)| {
            let n = a.history.times.len();
            b.history.events[n..].iter().all(Option::is_none)
                && b.history
                    .measurements
                    .iter()
                    .filter(|m| m.node >= n)
                    .all(|m| m.value.is_none())
        });
        let result = match &inference.law {
            CoefficientLaw::ConstantRates(law) => Ok(PredictiveHistoryDensity {
                log_density: law.conditional_predictive_log_density(
                    prefix,
                    extended,
                    options.memory_limit_bytes,
                )?,
                coefficient_log_standard_error: None,
                log_error_estimate: 0.0,
                effective_coefficient_samples: None,
            }),
            CoefficientLaw::Sampled(law) => {
                if prefix
                    .subjects
                    .iter()
                    .zip(extended.subjects)
                    .all(|(a, b)| a.history.times.len() == b.history.times.len())
                {
                    // The prefix checks establish identical observations.
                    // Every valid finite-parameter sampled model gives them
                    // positive density, so the exact conditional identity is 1.
                    return Ok(PredictiveHistoryDensity {
                        log_density: 0.0,
                        coefficient_log_standard_error: None,
                        log_error_estimate: 0.0,
                        effective_coefficient_samples: None,
                    });
                }
                let n = law.integral.draws.len();
                if n.checked_mul(8 * std::mem::size_of::<f64>())
                    .is_none_or(|v| v > options.memory_limit_bytes)
                {
                    return Err(invalid(
                        "conditional prediction exceeds its coefficient-workspace memory budget",
                    ));
                }
                let mut before = Vec::with_capacity(n);
                let mut after = Vec::with_capacity(n);
                let mut prefix_error = 0.0_f64;
                let mut extended_error = 0.0_f64;
                for draw in &law.integral.draws {
                    let past = prefix.resolved_score(&draw.coefficients, accuracy, tolerance)?;
                    before.push(*past.score().evaluation().log_likelihood());
                    prefix_error = prefix_error.max(past.report().log_error_estimate);
                    let future =
                        extended.resolved_score(&draw.coefficients, accuracy, tolerance)?;
                    after.push(*future.score().evaluation().log_likelihood());
                    extended_error = extended_error.max(future.report().log_error_estimate);
                }
                conditional_mixture(
                    &law.evidence.log_weights,
                    &before,
                    &after,
                    2.0 * law.evidence.inner_log_error_estimate() + prefix_error + extended_error,
                    options,
                )
            }
        }?;
        if no_outcomes && result.log_density() > 0.0 {
            return Err(numerical(
                "conditional no-event probability exceeds one; refine the predictive integrals",
            ));
        }
        Ok(result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn conditional_density_reweights_the_past_and_cancels_shared_sampling_error() {
        let weights = [0.2_f64, 0.3, 0.5];
        let log_weights: Vec<_> = weights.iter().map(|w| w.ln()).collect();
        let past = [-2.0_f64, -1.0, 0.0];
        let future = [-0.2_f64, -1.0, -3.0];
        let extended: Vec<_> = past.iter().zip(future).map(|(p, f)| p + f).collect();
        let options = PredictiveDensityOptions {
            log_error_tolerance: 10.0,
            minimum_effective_samples: 1.0,
            ..PredictiveDensityOptions::default()
        };
        let value = conditional_mixture(&log_weights, &past, &extended, 0.04, &options).unwrap();
        let denominator: f64 = weights.iter().zip(past).map(|(w, p)| w * p.exp()).sum();
        let numerator: f64 = weights
            .iter()
            .zip(&extended)
            .map(|(w, p)| w * p.exp())
            .sum();
        let naive: f64 = weights.iter().zip(future).map(|(w, p)| w * p.exp()).sum();
        assert!((value.log_density() - (numerator / denominator).ln()).abs() < 1e-14);
        assert!((value.log_density().exp() - naive).abs() > 0.1);
        let variance: f64 = weights
            .iter()
            .zip(past)
            .zip(&extended)
            .map(|((w, p), e)| (w * (e.exp() / numerator - p.exp() / denominator)).powi(2))
            .sum();
        let se = (1.5 * variance).sqrt();
        assert!((value.coefficient_log_standard_error().unwrap() - se).abs() < 1e-14);
        assert!((value.log_error_estimate() - 3.0 * se - 0.04).abs() < 1e-14);
        let constant: Vec<_> = past.iter().map(|p| p - 0.7).collect();
        let identity = conditional_mixture(&log_weights, &past, &constant, 0.0, &options).unwrap();
        assert!((identity.log_density() + 0.7).abs() < 1e-14);
        assert!(identity.coefficient_log_standard_error().unwrap() < 1e-14);
        let recovered = conditional_mixture(
            &[-1000.0, 0.0],
            &[1000.0, 0.0],
            &[999.6, -0.4],
            0.0,
            &options,
        )
        .unwrap();
        assert!((recovered.log_density() + 0.4).abs() < 1e-12);
        let adequate = PredictiveDensityOptions {
            minimum_effective_samples: 2.0,
            ..options
        };
        assert!(
            conditional_mixture(
                &log_weights,
                &[0.0, -1000.0, -1000.0],
                &log_weights.iter().map(|w| -w).collect::<Vec<_>>(),
                0.0,
                &adequate
            )
            .is_err()
        );
    }

    #[test]
    fn continuation_rejects_changes_to_every_past_observation_channel() {
        let prefix = JointHistory {
            times: vec![0.0, 1.0, 2.0],
            exposure: vec![0.0, 2.0, 0.0],
            events: vec![None; 3],
            initially_at_risk: vec![true],
            baseline_design: Array2::ones((3, 1)),
            drive_design: Array2::ones((2, 1)),
            entry_design: vec![0.2],
            genetics: vec![Some(0.1), None],
            measurements: vec![MeasurementRecord {
                node: 1,
                channel: 0,
                value: Some(0.4),
                after_event: false,
            }],
        };
        let extended = JointHistory {
            times: vec![0.0, 1.0, 2.0, 3.0, 4.0],
            exposure: vec![0.0, 2.0, 0.0, 2.0, 0.0],
            events: vec![None; 5],
            baseline_design: Array2::ones((5, 1)),
            drive_design: Array2::ones((4, 1)),
            measurements: vec![
                prefix.measurements[0].clone(),
                MeasurementRecord {
                    node: 3,
                    channel: 0,
                    value: Some(0.5),
                    after_event: false,
                },
            ],
            ..prefix.clone()
        };
        check_prefix(&prefix, &extended).unwrap();
        check_prefix(&prefix, &prefix).unwrap();
        for channel in 0..12 {
            let mut changed = extended.clone();
            match channel {
                0 => changed.times[1] = 0.5,
                1 => changed.exposure[1] = 1.0,
                2 => changed.events[2] = Some(0),
                3 => changed.initially_at_risk[0] = false,
                4 => changed.baseline_design[[1, 0]] = 2.0,
                5 => changed.drive_design[[0, 0]] = 2.0,
                6 => changed.entry_design[0] = 0.3,
                7 => changed.genetics[1] = Some(0.0),
                8 => changed.measurements[0].value = Some(0.6),
                9 => changed.measurements[0].after_event = true,
                10 => {
                    changed.measurements.remove(0);
                }
                11 => changed.measurements.push(prefix.measurements[0].clone()),
                _ => unreachable!(),
            }
            assert!(
                check_prefix(&prefix, &changed).is_err(),
                "channel {channel}"
            );
        }
    }
}
