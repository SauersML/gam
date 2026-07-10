//! Full-gradient runtime audit for the SAE LAML criterion.
//!
//! The criterion value and analytic-gradient paths are independently complex.
//! After fitting, this audit central-differences every outer-ρ coordinate, with
//! a Richardson companion step, and compares the resulting numerical gradient
//! to the complete production gradient. It is diagnostic only: finite
//! differences never produce a fitted value or update.

/// A complete first-order value/gradient consistency audit.
#[derive(Debug, Clone)]
pub struct GradientCriterionCertificate {
    /// Euclidean norm of the production analytic gradient.
    pub grad_norm: f64,
    /// Central-difference gradient of the production value path.
    // FD-OK: post-fit audit oracle; never consumed by fitting math
    pub fd_gradient: Vec<f64>,
    // END-FD-OK
    /// Production analytic gradient, in the same outer-ρ layout.
    pub analytic_gradient: Vec<f64>,
    /// Coordinatewise Richardson truncation-error estimates.
    // FD-OK: post-fit audit error bars
    pub fd_error_bars: Vec<f64>,
    // END-FD-OK
    /// Coordinatewise finite-difference steps.
    pub steps: Vec<f64>,
    /// Whether every probe was finite and its underlying solve well posed.
    pub well_posed: bool,
}

impl GradientCriterionCertificate {
    /// Largest relative coordinate error that is resolved beyond its numerical
    /// error bar. A gap equal to the unresolved FD bar contributes zero.
    #[must_use]
    pub fn agreement_rel(&self) -> f64 {
        if self.fd_gradient.len() != self.analytic_gradient.len()
            || self.fd_gradient.len() != self.fd_error_bars.len()
        {
            return f64::INFINITY;
        }
        self.fd_gradient
            .iter()
            .zip(self.analytic_gradient.iter())
            .zip(self.fd_error_bars.iter())
            .map(|((&fd, &analytic), &error)| {
                let resolved = ((fd - analytic).abs() - error).max(0.0);
                resolved / fd.abs().max(analytic.abs()).max(1.0e-12)
            })
            .fold(0.0_f64, f64::max)
    }

    #[must_use]
    pub fn passes(&self, relative_tolerance: f64) -> bool {
        self.well_posed
            && relative_tolerance.is_finite()
            && relative_tolerance >= 0.0
            && self.agreement_rel() <= relative_tolerance
    }
}

/// Relative probe step for one outer-ρ coordinate.
#[must_use]
pub fn probe_step_for(rho_i: f64) -> f64 {
    const BASE: f64 = 1.0e-4;
    BASE * rho_i.abs().max(1.0)
}

/// Four value-path samples for one coordinate and its analytic derivative.
#[derive(Debug, Clone, Copy)]
pub struct CoordinateSamples {
    pub plus_h: f64,
    pub minus_h: f64,
    pub plus_2h: f64,
    pub minus_2h: f64,
    pub step: f64,
    pub analytic: f64,
    pub well_posed: bool,
}

/// Assemble a full-gradient certificate from every coordinate's samples.
#[must_use]
pub fn certificate_from_samples(samples: &[CoordinateSamples]) -> GradientCriterionCertificate {
    let mut fd_gradient = Vec::with_capacity(samples.len());
    let mut analytic_gradient = Vec::with_capacity(samples.len());
    let mut fd_error_bars = Vec::with_capacity(samples.len());
    let mut steps = Vec::with_capacity(samples.len());
    let mut well_posed = !samples.is_empty();
    for sample in samples {
        let d_h = (sample.plus_h - sample.minus_h) / (2.0 * sample.step);
        let d_2h = (sample.plus_2h - sample.minus_2h) / (4.0 * sample.step);
        fd_gradient.push(d_h); // fd-ok: post-fit value-path audit
        analytic_gradient.push(sample.analytic);
        fd_error_bars.push((d_h - d_2h).abs() / 3.0); // fd-ok: Richardson audit bar
        steps.push(sample.step);
        well_posed &= sample.well_posed
            && sample.step.is_finite()
            && sample.step > 0.0
            && sample.plus_h.is_finite()
            && sample.minus_h.is_finite()
            && sample.plus_2h.is_finite()
            && sample.minus_2h.is_finite();
    }
    let grad_norm = analytic_gradient
        .iter()
        .map(|value| value * value)
        .sum::<f64>()
        .sqrt();
    GradientCriterionCertificate {
        grad_norm,
        fd_gradient,
        analytic_gradient,
        fd_error_bars,
        steps,
        well_posed,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn full_gradient_catches_error_orthogonal_to_any_single_probe() {
        let rho = [0.7_f64, -1.3];
        let analytic = [2.0 * rho[0] + 1.0, 3.0 * rho[1] - 2.0];
        let value = |point: [f64; 2]| {
            point[0] * point[0] + 1.5 * point[1] * point[1] + point[0] - 2.0 * point[1]
        };
        let mut samples = Vec::new();
        for axis in 0..2 {
            let h = probe_step_for(rho[axis]);
            let at = |multiple: f64| {
                let mut point = rho;
                point[axis] += multiple * h;
                value(point)
            };
            samples.push(CoordinateSamples {
                plus_h: at(1.0),
                minus_h: at(-1.0),
                plus_2h: at(2.0),
                minus_2h: at(-2.0),
                step: h,
                analytic: analytic[axis],
                well_posed: true,
            });
        }
        let exact = certificate_from_samples(&samples);
        assert!(exact.passes(1.0e-6));

        samples[1].analytic += 1.0;
        let broken = certificate_from_samples(&samples);
        assert!(!broken.passes(1.0e-3));
    }

    #[test]
    fn unresolved_gap_is_not_reported_as_unit_disagreement() {
        let certificate = GradientCriterionCertificate {
            grad_norm: 1.0,
            fd_gradient: vec![2.0],
            analytic_gradient: vec![1.0],
            fd_error_bars: vec![1.0],
            steps: vec![1.0e-4],
            well_posed: true,
        };
        assert_eq!(certificate.agreement_rel(), 0.0);
    }

    #[test]
    fn nonfinite_probe_is_not_well_posed() {
        let certificate = certificate_from_samples(&[CoordinateSamples {
            plus_h: f64::NAN,
            minus_h: 0.0,
            plus_2h: 0.0,
            minus_2h: 0.0,
            step: 1.0e-4,
            analytic: 0.0,
            well_posed: true,
        }]);
        assert!(!certificate.well_posed);
        assert!(!certificate.passes(1.0));
    }
}
