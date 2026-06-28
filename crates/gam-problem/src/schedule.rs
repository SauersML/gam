/// Decay law for deterministic Gumbel/concrete assignment temperature.
#[derive(Debug, Clone)]
pub enum ScheduleKind {
    Geometric { rate: f64 },
    Linear { steps: usize },
    ReciprocalIter,
}

/// Outer-state temperature annealing for SAE assignment relaxations.
///
/// Annealing drives the continuous concrete/softmax assignment toward the
/// discrete argmax or IBP active-set solution while PIRLS solves smooth
/// positive-temperature subproblems. In the zero-floor limit, softmax becomes
/// argmax and the IBP-MAP sigmoid active set becomes exact; a positive
/// `tau_min` optimizes the corresponding near-discrete MAP problem.
#[derive(Debug, Clone)]
pub struct GumbelTemperatureSchedule {
    pub tau_start: f64,
    pub tau_min: f64,
    pub decay: ScheduleKind,
    pub iter_count: usize,
}

impl GumbelTemperatureSchedule {
    #[must_use = "build error must be handled"]
    pub fn new(tau_start: f64, tau_min: f64, decay: ScheduleKind) -> Result<Self, String> {
        let sched = Self {
            tau_start,
            tau_min,
            decay,
            iter_count: 0,
        };
        sched.validate()?;
        Ok(sched)
    }

    pub fn validate(&self) -> Result<(), String> {
        if !(self.tau_start.is_finite() && self.tau_start > 0.0) {
            return Err(format!(
                "GumbelTemperatureSchedule: tau_start must be finite and positive; got {}",
                self.tau_start
            ));
        }
        if !(self.tau_min.is_finite() && self.tau_min > 0.0) {
            return Err(format!(
                "GumbelTemperatureSchedule: tau_min must be finite and positive; got {}",
                self.tau_min
            ));
        }
        if self.tau_min > self.tau_start {
            return Err(format!(
                "GumbelTemperatureSchedule: tau_min ({}) cannot exceed tau_start ({})",
                self.tau_min, self.tau_start
            ));
        }
        match self.decay {
            ScheduleKind::Geometric { rate } => {
                if !(rate.is_finite() && rate > 0.0 && rate < 1.0) {
                    return Err(format!(
                        "GumbelTemperatureSchedule::Geometric: rate must be in (0, 1); got {rate}"
                    ));
                }
            }
            ScheduleKind::Linear { steps } => {
                if steps == 0 {
                    return Err("GumbelTemperatureSchedule::Linear: steps must be positive".into());
                }
            }
            ScheduleKind::ReciprocalIter => {}
        }
        Ok(())
    }

    pub fn current_tau(&self, iter: usize) -> f64 {
        let raw = match self.decay {
            ScheduleKind::Geometric { rate } => self.tau_start * rate.powf(iter as f64),
            ScheduleKind::Linear { steps } => {
                if iter >= steps {
                    self.tau_min
                } else {
                    let frac = iter as f64 / steps as f64;
                    self.tau_start + frac * (self.tau_min - self.tau_start)
                }
            }
            ScheduleKind::ReciprocalIter => self.tau_start / (1.0 + iter as f64),
        };
        raw.max(self.tau_min)
    }

    pub fn step(&mut self) -> f64 {
        let tau = self.current_tau(self.iter_count);
        self.iter_count += 1;
        tau
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum SearchStrategy {
    Fixed,
    ExponentialSweep { values: Vec<f64> },
}

impl SearchStrategy {
    #[must_use]
    pub fn is_fixed(&self) -> bool {
        matches!(self, Self::Fixed)
    }

    #[must_use]
    pub fn sweep_values(&self) -> Option<&[f64]> {
        match self {
            Self::Fixed => None,
            Self::ExponentialSweep { values } => Some(values),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn geometric(rate: f64) -> GumbelTemperatureSchedule {
        GumbelTemperatureSchedule::new(1.0, 0.01, ScheduleKind::Geometric { rate }).unwrap()
    }

    // ── GumbelTemperatureSchedule validation ──────────────────────────────────

    #[test]
    fn new_ok_for_valid_geometric() {
        assert!(GumbelTemperatureSchedule::new(
            1.0,
            0.1,
            ScheduleKind::Geometric { rate: 0.9 }
        )
        .is_ok());
    }

    #[test]
    fn new_err_for_non_positive_tau_start() {
        assert!(GumbelTemperatureSchedule::new(0.0, 0.1, ScheduleKind::ReciprocalIter).is_err());
        assert!(GumbelTemperatureSchedule::new(f64::NAN, 0.1, ScheduleKind::ReciprocalIter)
            .is_err());
    }

    #[test]
    fn new_err_for_tau_min_exceeds_tau_start() {
        assert!(GumbelTemperatureSchedule::new(
            0.5,
            1.0,
            ScheduleKind::Geometric { rate: 0.9 }
        )
        .is_err());
    }

    #[test]
    fn new_err_for_geometric_rate_out_of_range() {
        assert!(GumbelTemperatureSchedule::new(
            1.0,
            0.1,
            ScheduleKind::Geometric { rate: 1.0 }
        )
        .is_err());
        assert!(GumbelTemperatureSchedule::new(
            1.0,
            0.1,
            ScheduleKind::Geometric { rate: 0.0 }
        )
        .is_err());
    }

    #[test]
    fn new_err_for_linear_zero_steps() {
        assert!(
            GumbelTemperatureSchedule::new(1.0, 0.1, ScheduleKind::Linear { steps: 0 }).is_err()
        );
    }

    // ── current_tau: Geometric ────────────────────────────────────────────────

    #[test]
    fn geometric_iter_zero_returns_tau_start() {
        let s = geometric(0.5);
        assert!((s.current_tau(0) - 1.0).abs() < 1e-14);
    }

    #[test]
    fn geometric_decays_by_rate_each_step() {
        let s = geometric(0.5);
        // iter 2: 1.0 * 0.5^2 = 0.25
        assert!((s.current_tau(2) - 0.25).abs() < 1e-12);
    }

    #[test]
    fn geometric_clamps_at_tau_min() {
        let s = GumbelTemperatureSchedule::new(
            1.0,
            0.5,
            ScheduleKind::Geometric { rate: 0.1 },
        )
        .unwrap();
        // 1.0 * 0.1^5 = 1e-5 < tau_min=0.5 → clamped
        assert!((s.current_tau(5) - 0.5).abs() < 1e-14);
    }

    // ── current_tau: Linear ───────────────────────────────────────────────────

    #[test]
    fn linear_iter_zero_returns_tau_start() {
        let s = GumbelTemperatureSchedule::new(2.0, 0.5, ScheduleKind::Linear { steps: 10 }).unwrap();
        assert!((s.current_tau(0) - 2.0).abs() < 1e-14);
    }

    #[test]
    fn linear_at_steps_returns_tau_min() {
        let s = GumbelTemperatureSchedule::new(2.0, 0.5, ScheduleKind::Linear { steps: 10 }).unwrap();
        assert!((s.current_tau(10) - 0.5).abs() < 1e-14);
    }

    // ── current_tau: ReciprocalIter ───────────────────────────────────────────

    #[test]
    fn reciprocal_iter_zero_returns_tau_start() {
        let s = GumbelTemperatureSchedule::new(4.0, 0.1, ScheduleKind::ReciprocalIter).unwrap();
        assert!((s.current_tau(0) - 4.0).abs() < 1e-14);
    }

    #[test]
    fn reciprocal_iter_one_halves_tau_start() {
        let s = GumbelTemperatureSchedule::new(4.0, 0.1, ScheduleKind::ReciprocalIter).unwrap();
        assert!((s.current_tau(1) - 2.0).abs() < 1e-14);
    }

    // ── step() increments iter_count ──────────────────────────────────────────

    #[test]
    fn step_increments_iter_count() {
        let mut s = geometric(0.5);
        assert_eq!(s.iter_count, 0);
        s.step();
        assert_eq!(s.iter_count, 1);
        s.step();
        assert_eq!(s.iter_count, 2);
    }

    // ── SearchStrategy ────────────────────────────────────────────────────────

    #[test]
    fn fixed_is_fixed_and_has_no_sweep_values() {
        let s = SearchStrategy::Fixed;
        assert!(s.is_fixed());
        assert!(s.sweep_values().is_none());
    }

    #[test]
    fn exponential_sweep_is_not_fixed_and_returns_values() {
        let s = SearchStrategy::ExponentialSweep { values: vec![1.0, 2.0, 3.0] };
        assert!(!s.is_fixed());
        assert_eq!(s.sweep_values().unwrap(), &[1.0, 2.0, 3.0]);
    }
}
