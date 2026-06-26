//! PSIS diagnostic for marginal smoothing-parameter uncertainty.
//!
//! The diagnostic treats the exact outer Hessian at `rho_hat` as a Laplace
//! proposal, evaluates the exact profiled criterion at a deterministic finite
//! set of proposal draws, and fits a GPD tail to the resulting importance
//! weights. A large `k_hat` is evidence that fixed-`rho` REML/LAML intervals are
//! inadequate for this fit; a small `k_hat` is only absence of heavy-tail
//! evidence at the probed points, not a proof about unprobed tails. A criterion
//! closure can agree with the Gaussian proposal at every deterministic draw and
//! still have catastrophic heavier tails elsewhere.

use crate::psis::{MIN_TAIL_COUNT, pareto_smooth_weights};
use ndarray::{Array1, Array2};

const DEFAULT_SAMPLE_COUNT: usize = 32;
const MAX_AUTO_RHO_DIM: usize = 4;
const MAX_AUTO_WORK_UNITS: usize = 2_000_000;

#[derive(Clone, Debug, PartialEq)]
pub struct RhoUncertaintyDiagnostic {
    pub k_hat: Option<f64>,
    pub n_evaluations: usize,
    pub status: RhoUncertaintyStatus,
}

#[derive(Clone, Debug, PartialEq)]
pub enum RhoUncertaintyStatus {
    NoEvidenceOfHeavyTails,
    HeavyTailsDetected { k_hat: f64 },
    Skipped { reason: String },
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct RhoUncertaintyProblemSize {
    pub n_obs: Option<usize>,
    pub p_coefficients: Option<usize>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RhoUncertaintyCostGate {
    pub sample_count: usize,
    pub problem_size: RhoUncertaintyProblemSize,
}

impl Default for RhoUncertaintyCostGate {
    fn default() -> Self {
        Self {
            sample_count: DEFAULT_SAMPLE_COUNT,
            problem_size: RhoUncertaintyProblemSize::default(),
        }
    }
}

impl RhoUncertaintyDiagnostic {
    pub fn skipped(reason: impl Into<String>, n_evaluations: usize) -> Self {
        Self {
            k_hat: None,
            n_evaluations,
            status: RhoUncertaintyStatus::Skipped {
                reason: reason.into(),
            },
        }
    }
}

pub fn cost_gate_allows(rho_dim: usize, gate: RhoUncertaintyCostGate) -> Result<usize, String> {
    if rho_dim == 0 {
        return Err("no smoothing parameters".to_string());
    }
    if rho_dim > MAX_AUTO_RHO_DIM {
        return Err(format!(
            "rho dimension {rho_dim} exceeds automatic PSIS diagnostic limit {MAX_AUTO_RHO_DIM}"
        ));
    }
    let sample_count = gate.sample_count.max(2 * MIN_TAIL_COUNT);
    let n = gate.problem_size.n_obs.unwrap_or(1);
    let p = gate.problem_size.p_coefficients.unwrap_or(1);
    let work_units = sample_count
        .saturating_add(1)
        .saturating_mul(rho_dim.max(1))
        .saturating_mul(n.max(1))
        .saturating_mul(p.max(1));
    if work_units > MAX_AUTO_WORK_UNITS {
        return Err(format!(
            "estimated diagnostic cost {work_units} work units exceeds automatic limit {MAX_AUTO_WORK_UNITS} \
             (M={sample_count}, K={rho_dim}, n={}, p={})",
            gate.problem_size.n_obs.unwrap_or(0),
            gate.problem_size.p_coefficients.unwrap_or(0),
        ));
    }
    Ok(sample_count)
}

pub fn rho_uncertainty_diagnostic<F>(
    rho_hat: &Array1<f64>,
    outer_hessian_rho: &Array2<f64>,
    gate: RhoUncertaintyCostGate,
    mut criterion: F,
) -> RhoUncertaintyDiagnostic
where
    F: FnMut(&Array1<f64>) -> Option<f64>,
{
    let rho_dim = rho_hat.len();
    let sample_count = match cost_gate_allows(rho_dim, gate) {
        Ok(sample_count) => sample_count,
        Err(reason) => return RhoUncertaintyDiagnostic::skipped(reason, 0),
    };
    if outer_hessian_rho.nrows() != rho_dim || outer_hessian_rho.ncols() != rho_dim {
        return RhoUncertaintyDiagnostic::skipped(
            format!(
                "outer rho Hessian shape {}x{} does not match K={rho_dim}",
                outer_hessian_rho.nrows(),
                outer_hessian_rho.ncols()
            ),
            0,
        );
    }
    let Some(cost_hat) = criterion(rho_hat).filter(|value| value.is_finite()) else {
        return RhoUncertaintyDiagnostic::skipped("criterion was not finite at rho_hat", 1);
    };
    let Some(proposal_factor) = proposal_factor_from_hessian(outer_hessian_rho) else {
        return RhoUncertaintyDiagnostic::skipped("outer rho Hessian was not positive definite", 1);
    };

    let mut rng = DeterministicNormal::new(seed_from_problem(rho_hat, gate.problem_size));
    let mut log_weights = Vec::with_capacity(sample_count);
    let mut n_evaluations = 1usize;
    for _draw in 0..sample_count {
        let z = Array1::from_iter((0..rho_dim).map(|coord| rng.normal(coord)));
        let rho = rho_hat + &proposal_factor.dot(&z);
        let half_norm_sq = 0.5 * z.iter().map(|value| value * value).sum::<f64>();
        let log_weight = match criterion(&rho) {
            Some(cost) if cost.is_finite() => -cost + cost_hat + half_norm_sq,
            _ => f64::NEG_INFINITY,
        };
        log_weights.push(log_weight);
        n_evaluations = n_evaluations.saturating_add(1);
    }

    let max_log_weight = log_weights
        .iter()
        .copied()
        .filter(|value| value.is_finite())
        .fold(f64::NEG_INFINITY, f64::max);
    if !max_log_weight.is_finite() {
        return RhoUncertaintyDiagnostic::skipped(
            "all proposal draws had non-finite criterion values",
            n_evaluations,
        );
    }
    let weights: Vec<f64> = log_weights
        .iter()
        .map(|&value| {
            if value.is_finite() {
                (value - max_log_weight).exp()
            } else {
                0.0
            }
        })
        .collect();
    let (min_weight, max_weight) = weights
        .iter()
        .copied()
        .fold((f64::INFINITY, f64::NEG_INFINITY), |(min_w, max_w), w| {
            (min_w.min(w), max_w.max(w))
        });
    if max_weight.is_finite()
        && min_weight.is_finite()
        && max_weight > 0.0
        && (max_weight - min_weight) <= 1e-12 * max_weight.max(1.0)
    {
        return RhoUncertaintyDiagnostic {
            k_hat: Some(0.0),
            n_evaluations,
            status: RhoUncertaintyStatus::NoEvidenceOfHeavyTails,
        };
    }
    let Some(psis) = pareto_smooth_weights(&weights) else {
        return RhoUncertaintyDiagnostic::skipped(
            "PSIS tail fit failed for rho-importance weights",
            n_evaluations,
        );
    };
    let k_hat = psis.k_hat;
    let status = if k_hat < 0.5 {
        RhoUncertaintyStatus::NoEvidenceOfHeavyTails
    } else {
        RhoUncertaintyStatus::HeavyTailsDetected { k_hat }
    };
    RhoUncertaintyDiagnostic {
        k_hat: Some(k_hat),
        n_evaluations,
        status,
    }
}

fn proposal_factor_from_hessian(hessian: &Array2<f64>) -> Option<Array2<f64>> {
    let chol = cholesky_lower(hessian)?;
    let n = chol.nrows();
    let mut inverse_lower = Array2::<f64>::zeros((n, n));
    for col in 0..n {
        for row in 0..n {
            let mut acc = if row == col { 1.0 } else { 0.0 };
            for k in 0..row {
                acc -= chol[[row, k]] * inverse_lower[[k, col]];
            }
            let diagonal = chol[[row, row]];
            if !(diagonal.is_finite() && diagonal > 0.0) {
                return None;
            }
            inverse_lower[[row, col]] = acc / diagonal;
        }
    }
    let mut factor = Array2::<f64>::zeros((n, n));
    for row in 0..n {
        for col in 0..n {
            factor[[row, col]] = inverse_lower[[col, row]];
        }
    }
    Some(factor)
}

fn cholesky_lower(matrix: &Array2<f64>) -> Option<Array2<f64>> {
    let n = matrix.nrows();
    if n == 0 || matrix.ncols() != n || matrix.iter().any(|value| !value.is_finite()) {
        return None;
    }
    let mut lower = Array2::<f64>::zeros((n, n));
    for row in 0..n {
        for col in 0..=row {
            let mut acc = matrix[[row, col]];
            for k in 0..col {
                acc -= lower[[row, k]] * lower[[col, k]];
            }
            if row == col {
                if !(acc.is_finite() && acc > 0.0) {
                    return None;
                }
                lower[[row, col]] = acc.sqrt();
            } else {
                let diagonal = lower[[col, col]];
                if !(diagonal.is_finite() && diagonal > 0.0) {
                    return None;
                }
                lower[[row, col]] = acc / diagonal;
            }
        }
    }
    Some(lower)
}

fn seed_from_problem(rho_hat: &Array1<f64>, size: RhoUncertaintyProblemSize) -> u64 {
    let mut state = 0xcbf2_9ce4_8422_2325_u64;
    mix_u64(&mut state, size.n_obs.unwrap_or(0) as u64);
    mix_u64(&mut state, size.p_coefficients.unwrap_or(0) as u64);
    mix_u64(&mut state, rho_hat.len() as u64);
    for value in rho_hat {
        mix_u64(&mut state, value.to_bits());
    }
    state
}

fn mix_u64(state: &mut u64, value: u64) {
    for byte in value.to_le_bytes() {
        *state ^= u64::from(byte);
        *state = state.wrapping_mul(0x0000_0100_0000_01b3);
    }
}

struct DeterministicNormal {
    state: u64,
    spare: Option<f64>,
}

impl DeterministicNormal {
    fn new(seed: u64) -> Self {
        Self {
            state: seed,
            spare: None,
        }
    }

    fn normal(&mut self, coord: usize) -> f64 {
        if let Some(value) = self.spare.take() {
            return value;
        }
        mix_u64(&mut self.state, coord as u64);
        let u1 = self.uniform().max(1e-300);
        let u2 = self.uniform();
        let radius = (-2.0 * u1.ln()).sqrt();
        let angle = 2.0 * std::f64::consts::PI * u2;
        self.spare = Some(radius * angle.sin());
        radius * angle.cos()
    }

    fn uniform(&mut self) -> f64 {
        self.state = self.state.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^= z >> 31;
        ((z >> 11) as f64 + 0.5) / (1_u64 << 53) as f64
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    fn gaussian_criterion(
        rho_hat: Array1<f64>,
        hessian: Array2<f64>,
    ) -> impl FnMut(&Array1<f64>) -> Option<f64> {
        move |rho: &Array1<f64>| {
            let delta = rho - &rho_hat;
            Some(0.5 * delta.dot(&hessian.dot(&delta)))
        }
    }

    #[test]
    fn near_gaussian_target_has_no_heavy_tail_evidence_at_probe_points() {
        let rho_hat = array![0.2, -0.3];
        let hessian = array![[2.5, 0.2], [0.2, 1.8]];
        let diagnostic = rho_uncertainty_diagnostic(
            &rho_hat,
            &hessian,
            RhoUncertaintyCostGate {
                sample_count: 32,
                problem_size: RhoUncertaintyProblemSize {
                    n_obs: Some(40),
                    p_coefficients: Some(8),
                },
            },
            gaussian_criterion(rho_hat.clone(), hessian.clone()),
        );
        assert!(
            matches!(
                diagnostic.status,
                RhoUncertaintyStatus::NoEvidenceOfHeavyTails
            ),
            "near-Gaussian rho posterior should not show heavy-tail evidence at the probe \
             points, got {diagnostic:?}"
        );
        assert!(
            diagnostic.k_hat.expect("k_hat") < 0.5,
            "near-Gaussian target should have k_hat below 0.5"
        );
    }

    #[test]
    fn weak_identification_orders_above_gaussian_case() {
        let rho_hat = array![0.0];
        let hessian = array![[5.0]];
        let gate = RhoUncertaintyCostGate {
            sample_count: 64,
            problem_size: RhoUncertaintyProblemSize {
                n_obs: Some(12),
                p_coefficients: Some(4),
            },
        };
        let gaussian = rho_uncertainty_diagnostic(
            &rho_hat,
            &hessian,
            gate,
            gaussian_criterion(rho_hat.clone(), hessian.clone()),
        );
        let weak = rho_uncertainty_diagnostic(&rho_hat, &hessian, gate, |rho| {
            Some((1.0 + rho[0] * rho[0]).ln())
        });
        assert!(
            weak.k_hat.expect("weak k_hat") > gaussian.k_hat.expect("gaussian k_hat"),
            "weak rho identification should increase k_hat: weak={weak:?} gaussian={gaussian:?}"
        );
    }

    #[test]
    fn diagnostic_is_bit_deterministic() {
        let rho_hat = array![0.7];
        let hessian = array![[1.4]];
        let gate = RhoUncertaintyCostGate {
            sample_count: 32,
            problem_size: RhoUncertaintyProblemSize {
                n_obs: Some(80),
                p_coefficients: Some(9),
            },
        };
        let a = rho_uncertainty_diagnostic(
            &rho_hat,
            &hessian,
            gate,
            gaussian_criterion(rho_hat.clone(), hessian.clone()),
        );
        let b = rho_uncertainty_diagnostic(
            &rho_hat,
            &hessian,
            gate,
            gaussian_criterion(rho_hat.clone(), hessian.clone()),
        );
        assert_eq!(a, b);
    }

    #[test]
    fn cost_gate_skips_large_problem() {
        let rho_hat = array![0.0, 0.0, 0.0, 0.0, 0.0];
        let hessian = array![
            [1.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 1.0],
        ];
        let diagnostic = rho_uncertainty_diagnostic(
            &rho_hat,
            &hessian,
            RhoUncertaintyCostGate::default(),
            |_| Some(0.0),
        );
        assert!(matches!(
            diagnostic.status,
            RhoUncertaintyStatus::Skipped { .. }
        ));
    }
}
