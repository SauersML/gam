//! Value-only reference assessment for coefficient quadrature and prediction.
//! Hyperparameter derivatives use the cached likelihood values; constructing
//! every coefficient gradient at those points is unnecessary.
use super::*;

pub(in crate::joint) struct ValueAssessment {
    pub assessment: FunctionalAssessment,
    /// Conditional SE of the paired log-integral differences, using the SAME
    /// subject innovations on both reference curves.
    pub conditional_difference: [f64; 2],
}

/// Remove one of `replicates` pooled populations from the pooled curve: with
/// `lm = log(M_r / sum M)` and `la = log(M_r m_r / sum M m)`, the remaining log
/// mass is `full + log R - log(R - 1) + log(1 - exp lm)` and the remaining log
/// moment is `full + log(1 - exp la) - log(1 - exp lm)`.
pub(super) fn delete_population<S: JetField>(
    full: &JointReferenceEvolution<S>,
    part: &JointReferenceEvolution<S>,
    replicates: usize,
) -> Result<JointReferenceEvolution<S>, EventHistoryError> {
    if full.times != part.times
        || full.marks != part.marks
        || replicates < 2
        || full.theta.len() != part.theta.len()
        || full
            .theta
            .iter()
            .zip(&part.theta)
            .any(|(a, b)| a.value() != b.value())
    {
        return Err(invalid(
            "reference deletion needs matching coefficient/grid states",
        ));
    }
    // log R and log(R - 1) are formed over `S` from the exact counts.
    let like = &full.log_moments[0];
    let count = ln(&like.constant_like(replicates as f64));
    let remaining = ln(&like.constant_like((replicates - 1) as f64));
    let mut mass = Vec::with_capacity(full.log_risk_mass.len());
    let mut moments = Vec::with_capacity(full.log_moments.len());
    for i in 0..full.log_moments.len() {
        let lm = part.log_risk_mass[i].sub(&full.log_risk_mass[i]).sub(&count);
        let la = lm.add(&part.log_moments[i]).sub(&full.log_moments[i]);
        let mass_remainder = emission::expm1(&lm).neg();
        let activity_remainder = emission::expm1(&la).neg();
        if !(mass_remainder.value() > 0.0 && activity_remainder.value() > 0.0) {
            return Err(numerical(
                "reference value deletion unresolved: one population dominates",
            ));
        }
        let log_mass = ln(&mass_remainder);
        mass.push(full.log_risk_mass[i].add(&count).sub(&remaining).add(&log_mass));
        moments.push(full.log_moments[i].add(&ln(&activity_remainder)).sub(&log_mass));
    }
    Ok(JointReferenceEvolution {
        theta: full.theta.clone(),
        times: full.times.clone(),
        row_times: full.row_times.clone(),
        marks: full.marks,
        log_moments: moments,
        log_risk_mass: mass,
        diagnostics: full.diagnostics.clone(),
    })
}

impl ResolvedReference {
    fn value_point<F>(
        &self,
        theta: &[f64],
        banks: &[JointReferenceBank],
        evaluate: &mut F,
    ) -> Result<(JointReferenceEvolution<f64>, FunctionalPoint), EventHistoryError>
    where
        F: FnMut(&JointReferenceEvolution<f64>) -> Result<FunctionalValue, EventHistoryError>,
    {
        let first = banks[0].evolve(&self.model, theta)?;
        // Live curve-sized arrays: R replicate curves, the pooled and deleted
        // curves, aggregate's 2R interpolated copies, and the two pooled curves
        // `assess_values` keeps while the next ensemble is evaluated.
        let required = banks
            .len()
            .checked_mul(3)
            .and_then(|n| n.checked_add(4))
            .and_then(|n| n.checked_mul(curve_bytes(&first).ok()?));
        let budget = materialization_budget();
        if required.is_none_or(|n| n > budget) {
            return Err(numerical(
                "reference value assessment exceeds this machine's materialization budget",
            ));
        }
        let mut curves = vec![first];
        for bank in &banks[1..] {
            curves.push(bank.evolve(&self.model, theta)?);
        }
        let pooled = aggregate(&curves, curves[0].times(), &curves[0].row_times)?.reference;
        let full = evaluate(&pooled)?;
        full.validate(1)?;
        let mut jackknife = JackknifeMoments::new(1, &0.0);
        for curve in &curves {
            let deleted = delete_population(&pooled, curve, banks.len())?;
            let output = evaluate(&deleted)?;
            output.validate(1)?;
            jackknife.add(&output.values);
        }
        Ok((pooled, jackknife.finish(full)?))
    }

    pub(in crate::joint) fn assess_values<F, G>(
        &self,
        theta: &[f64],
        mut evaluate: F,
        mut difference: G,
    ) -> Result<ValueAssessment, EventHistoryError>
    where
        F: FnMut(&JointReferenceEvolution<f64>) -> Result<FunctionalValue, EventHistoryError>,
        G: FnMut(
            &JointReferenceEvolution<f64>,
            &JointReferenceEvolution<f64>,
        ) -> Result<f64, EventHistoryError>,
    {
        let (coarse, c) = self.value_point(theta, &self.coarse, &mut evaluate)?;
        let (fine, f) = self.value_point(theta, &self.fine, &mut evaluate)?;
        let time_error = difference(&coarse, &fine)?;
        drop(coarse);
        let (large, h) = self.value_point(theta, &self.large, &mut evaluate)?;
        let particle_error = difference(&fine, &large)?;
        if [time_error, particle_error]
            .iter()
            .any(|v| !v.is_finite() || *v < 0.0)
        {
            return Err(numerical("non-finite paired reference comparison error"));
        }
        Ok(ValueAssessment {
            assessment: FunctionalAssessment { points: [c, f, h] },
            conditional_difference: [time_error, particle_error],
        })
    }
}
