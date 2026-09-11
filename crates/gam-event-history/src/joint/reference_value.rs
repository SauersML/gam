//! Value-only reference assessment for coefficient quadrature and prediction.
//! Hyperparameter derivatives use the cached likelihood values; constructing
//! every coefficient Jacobian at those points is unnecessary.
use super::*;

pub(in crate::joint) struct ValueAssessment {
    pub assessment: FunctionalAssessment,
    /// Conditional SE of the paired log-integral differences, using the SAME
    /// subject innovations on both reference curves.
    pub conditional_difference: [f64; 2],
}

pub(super) fn delete_population(
    full: &JointReferenceEvolution<f64>,
    part: &JointReferenceEvolution<f64>,
    replicates: usize,
) -> Result<JointReferenceEvolution<f64>, EventHistoryError> {
    if full.theta != part.theta
        || full.times != part.times
        || full.marks != part.marks
        || replicates < 2
    {
        return Err(invalid(
            "reference deletion needs matching coefficient/grid states",
        ));
    }
    let count = (replicates as f64).ln();
    let remaining = ((replicates - 1) as f64).ln();
    let mut mass = Vec::with_capacity(full.log_risk_mass.len());
    let mut moments = Vec::with_capacity(full.log_moments.len());
    for i in 0..full.log_moments.len() {
        let lm = part.log_risk_mass[i] - full.log_risk_mass[i] - count;
        let la = lm + part.log_moments[i] - full.log_moments[i];
        let mass_remainder = -lm.exp_m1();
        let activity_remainder = -la.exp_m1();
        if !(mass_remainder > 0.0 && activity_remainder > 0.0) {
            return Err(numerical(
                "reference value deletion unresolved: one population dominates",
            ));
        }
        let log_mass = mass_remainder.ln();
        mass.push(full.log_risk_mass[i] + count - remaining + log_mass);
        moments.push(full.log_moments[i] + activity_remainder.ln() - log_mass);
    }
    Ok(JointReferenceEvolution {
        theta: full.theta.clone(),
        times: full.times.clone(),
        marks: full.marks,
        log_moments: moments,
        log_risk_mass: mass,
        diagnostics: full.diagnostics.clone(),
    })
}

impl ResolvedReference<'_> {
    fn value_point<F>(
        &self,
        theta: &[f64],
        banks: &[JointReferenceBank<'_>],
        evaluate: &mut F,
    ) -> Result<(JointReferenceEvolution<f64>, FunctionalPoint), EventHistoryError>
    where
        F: FnMut(&JointReferenceEvolution<f64>) -> Result<FunctionalValue, EventHistoryError>,
    {
        let first = banks[0].evolve_for_resolution(theta, &self.step_limits)?;
        // Includes the two previously completed pooled curves, deletion and
        // aggregate work arrays while these replicate curves remain live.
        let required = banks
            .len()
            .checked_mul(3)
            .and_then(|n| n.checked_add(8))
            .and_then(|n| n.checked_mul(curve_bytes(&first).ok()?));
        if required.is_none_or(|n| n > self.options.memory_limit_bytes) {
            return Err(numerical(
                "reference value assessment exceeds its workspace budget",
            ));
        }
        let mut curves = vec![first];
        for bank in &banks[1..] {
            curves.push(bank.evolve_for_resolution(theta, &self.step_limits)?);
        }
        let pooled = aggregate(&curves, curves[0].times())?.reference;
        let value = evaluate(&pooled)?;
        value.validate(1)?;
        let mut jackknife = JackknifeMoments::new(1);
        for curve in &curves {
            let deleted = delete_population(&pooled, curve, banks.len())?;
            let output = evaluate(&deleted)?;
            output.validate(1)?;
            jackknife.add(&output.values);
        }
        Ok((pooled, jackknife.finish(value)?))
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
