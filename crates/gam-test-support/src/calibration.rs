//! Standing simulation-calibration helpers for uncertainty-quality tests.
//!
//! The harness deliberately lives in `gam-test-support`: production crates expose
//! uncertainty payloads, while quality tests register those surfaces here and
//! audit the calibration contract with deterministic planted draws.

use std::collections::BTreeSet;
use std::fmt;

/// Deterministic per-target replicate budget for fast calibration gates.
///
/// The value is the smallest multiple of the three nominal coverage levels used
/// by this harness that leaves at least twenty expected misses at 95% coverage,
/// so binomial noise cannot hide a halved-width interval in the self-test.
pub const FAST_COVERAGE_REPLICATES: usize = 420;

/// Deterministic SBC draw budget for fast gates.
///
/// The value gives ten expected counts in each decile of the rank histogram, the
/// conventional minimum for the chi-square approximation used by the verdict.
pub const FAST_SBC_DRAWS: usize = 100;

/// Rank-histogram bin count for the fast SBC gate.
///
/// Deciles are the coarsest symmetric partition that still distinguishes the
/// lower-tail and upper-tail skew planted by the self-test.
pub const SBC_DECILE_BINS: usize = 10;

/// Calibration audit family for a registered uncertainty surface.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd)]
pub enum CalibrationKind {
    Interval,
    TestPValue,
    PosteriorSbc,
}

/// Registry entry naming one public uncertainty surface.
#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd)]
pub struct CalibrationTarget {
    pub name: &'static str,
    pub kind: CalibrationKind,
    pub seed: u64,
}

impl CalibrationTarget {
    pub const fn new(name: &'static str, kind: CalibrationKind, seed: u64) -> Self {
        Self { name, kind, seed }
    }
}

/// Public uncertainty field discovered from result payloads.
#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd)]
pub struct UncertaintyField {
    pub payload: &'static str,
    pub field: &'static str,
}

impl UncertaintyField {
    pub const fn new(payload: &'static str, field: &'static str) -> Self {
        Self { payload, field }
    }

    pub fn target_name(&self) -> String {
        format!("{}.{}", self.payload, self.field)
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CompletenessFailure {
    pub missing_targets: Vec<String>,
    pub duplicate_targets: Vec<&'static str>,
}

impl fmt::Display for CompletenessFailure {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "calibration registry is incomplete; missing={:?}; duplicate_targets={:?}",
            self.missing_targets, self.duplicate_targets
        )
    }
}

/// Assert that every uncertainty-bearing payload field has a calibration target.
pub fn audit_registry_completeness(
    targets: &[CalibrationTarget],
    uncertainty_fields: &[UncertaintyField],
) -> Result<(), CompletenessFailure> {
    let mut names = BTreeSet::new();
    let mut duplicate_targets = Vec::new();
    for target in targets {
        if !names.insert(target.name) {
            duplicate_targets.push(target.name);
        }
    }

    let missing_targets = uncertainty_fields
        .iter()
        .map(UncertaintyField::target_name)
        .filter(|name| !names.contains(name.as_str()))
        .collect::<Vec<_>>();

    if missing_targets.is_empty() && duplicate_targets.is_empty() {
        Ok(())
    } else {
        Err(CompletenessFailure {
            missing_targets,
            duplicate_targets,
        })
    }
}

#[derive(Clone, Copy, Debug)]
pub struct IntervalDraw {
    pub truth: f64,
    pub estimate: f64,
    pub standard_error: f64,
}

#[derive(Clone, Copy, Debug)]
pub struct CoverageLevel {
    pub nominal: f64,
    pub z: f64,
}

/// Central normal coverage levels audited by the fast interval gate.
///
/// The z-values are the closed-form standard-normal central quantiles rounded to
/// the precision at which the Monte Carlo standard error from
/// `FAST_COVERAGE_REPLICATES` dominates the quantile rounding error.
pub const CENTRAL_NORMAL_LEVELS: [CoverageLevel; 3] = [
    CoverageLevel {
        nominal: 0.80,
        z: 1.281_551_565_545,
    },
    CoverageLevel {
        nominal: 0.90,
        z: 1.644_853_626_951,
    },
    CoverageLevel {
        nominal: 0.95,
        z: 1.959_963_984_540,
    },
];

#[derive(Clone, Debug)]
pub struct CoverageVerdict {
    pub target_name: &'static str,
    pub nominal: f64,
    pub empirical: f64,
    pub lower_gate: f64,
    pub passed: bool,
}

/// Audit interval coverage, hard-failing only anti-conservative under-coverage.
pub fn audit_interval_coverage(
    target: &CalibrationTarget,
    draws: &[IntervalDraw],
) -> Vec<CoverageVerdict> {
    assert_eq!(target.kind, CalibrationKind::Interval);
    let n = draws.len() as f64;
    CENTRAL_NORMAL_LEVELS
        .iter()
        .map(|level| {
            let hits = draws
                .iter()
                .filter(|draw| {
                    let half_width = level.z * draw.standard_error;
                    draw.truth >= draw.estimate - half_width
                        && draw.truth <= draw.estimate + half_width
                })
                .count() as f64;
            let empirical = hits / n;
            // Wilson-style lower gate: three Monte Carlo standard errors below
            // nominal, derived from the binomial null for this exact replicate
            // count. Conservative over-coverage is reported by the caller but is
            // not a hard failure for anti-conservatism.
            let gate_sigma = (level.nominal * (1.0 - level.nominal) / n).sqrt();
            let lower_gate = level.nominal - 3.0 * gate_sigma;
            CoverageVerdict {
                target_name: target.name,
                nominal: level.nominal,
                empirical,
                lower_gate,
                passed: empirical >= lower_gate,
            }
        })
        .collect()
}

#[derive(Clone, Debug)]
pub struct SbcVerdict {
    pub target_name: &'static str,
    pub counts: Vec<usize>,
    pub chi_square: f64,
    pub critical_value: f64,
    pub passed: bool,
}

/// Audit posterior rank uniformity with a fixed-bin chi-square verdict.
pub fn audit_sbc_ranks(
    target: &CalibrationTarget,
    ranks: &[usize],
    posterior_draws_per_rank: usize,
    bins: usize,
) -> SbcVerdict {
    assert_eq!(target.kind, CalibrationKind::PosteriorSbc);
    assert!(bins > 1);
    let mut counts = vec![0usize; bins];
    for &rank in ranks {
        assert!(rank <= posterior_draws_per_rank);
        let numerator = rank * bins;
        let denominator = posterior_draws_per_rank + 1;
        let bin = (numerator / denominator).min(bins - 1);
        counts[bin] += 1;
    }
    let expected = ranks.len() as f64 / bins as f64;
    let chi_square = counts
        .iter()
        .map(|&count| {
            let diff = count as f64 - expected;
            diff * diff / expected
        })
        .sum::<f64>();
    // Laurent-Massart 99% chi-square upper bound for k=bins-1 degrees of
    // freedom: k + 2 sqrt(k x) + 2x with x=ln(100). This is a principled fixed
    // false-positive rate for a standing gate, not a tuned threshold.
    let degrees = (bins - 1) as f64;
    let x = 100.0_f64.ln();
    let critical_value = degrees + 2.0 * (degrees * x).sqrt() + 2.0 * x;
    SbcVerdict {
        target_name: target.name,
        counts,
        chi_square,
        critical_value,
        passed: chi_square <= critical_value,
    }
}

/// Small deterministic RNG for calibration self-tests.
pub struct CalibrationRng {
    state: u64,
    spare_normal: Option<f64>,
}

impl CalibrationRng {
    pub const fn new(seed: u64) -> Self {
        Self {
            state: seed,
            spare_normal: None,
        }
    }

    pub fn uniform_open01(&mut self) -> f64 {
        // PCG-family 64-bit linear-congruential multiplier/increment from the
        // reference generator. The constants are the generator definition; the
        // calibration harness owns only the seed.
        self.state = self
            .state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        let bits = self.state >> 11;
        (bits as f64 + 0.5) / ((1_u64 << 53) as f64)
    }

    pub fn standard_normal(&mut self) -> f64 {
        if let Some(value) = self.spare_normal.take() {
            return value;
        }
        let radius = (-2.0 * self.uniform_open01().ln()).sqrt();
        let angle = std::f64::consts::TAU * self.uniform_open01();
        let first = radius * angle.cos();
        self.spare_normal = Some(radius * angle.sin());
        first
    }
}

#[cfg(test)]
mod tests {
    use super::{
        CalibrationKind, CalibrationRng, CalibrationTarget, FAST_COVERAGE_REPLICATES,
        FAST_SBC_DRAWS, IntervalDraw, SBC_DECILE_BINS, UncertaintyField, audit_interval_coverage,
        audit_registry_completeness, audit_sbc_ranks,
    };

    const GAUSSIAN_REFERENCE_TARGET: CalibrationTarget = CalibrationTarget::new(
        "GaussianReference.mean_interval",
        CalibrationKind::Interval,
        0x1891_1869_1878,
    );
    const NARROW_INTERVAL_TARGET: CalibrationTarget = CalibrationTarget::new(
        "PlantedMiscalibration.narrow_interval",
        CalibrationKind::Interval,
        0x1891_1870_1875,
    );
    const SBC_REFERENCE_TARGET: CalibrationTarget = CalibrationTarget::new(
        "GaussianReference.posterior_rank",
        CalibrationKind::PosteriorSbc,
        0x1891_1841_1810,
    );
    const SKEWED_SBC_TARGET: CalibrationTarget = CalibrationTarget::new(
        "PlantedMiscalibration.skewed_rank",
        CalibrationKind::PosteriorSbc,
        0x1891_1878_1841,
    );

    fn gaussian_interval_draws(seed: u64, standard_error_scale: f64) -> Vec<IntervalDraw> {
        let mut rng = CalibrationRng::new(seed);
        (0..FAST_COVERAGE_REPLICATES)
            .map(|_| {
                let error = rng.standard_normal();
                IntervalDraw {
                    truth: 0.0,
                    estimate: error,
                    standard_error: standard_error_scale,
                }
            })
            .collect()
    }

    fn uniform_sbc_ranks(draws_per_rank: usize) -> Vec<usize> {
        (0..FAST_SBC_DRAWS)
            .map(|draw| draw % (draws_per_rank + 1))
            .collect()
    }

    #[test]
    fn registry_completeness_lint_rejects_unregistered_uncertainty_payload_field() {
        let fields = [
            UncertaintyField::new("PredictiveMean", "se"),
            UncertaintyField::new("PredictiveMean", "interval"),
            UncertaintyField::new("LikelihoodRatio", "p_value"),
        ];
        let incomplete = [CalibrationTarget::new(
            "PredictiveMean.se",
            CalibrationKind::Interval,
            0x1891_0001,
        )];
        let failure = audit_registry_completeness(&incomplete, &fields).expect_err(
            "an uncertainty payload carrying interval/p_value fields must be registered before it ships",
        );
        assert_eq!(
            failure.missing_targets,
            vec![
                "LikelihoodRatio.p_value".to_string(),
                "PredictiveMean.interval".to_string()
            ]
        );
    }

    #[test]
    fn gaussian_reference_interval_passes_and_planted_narrow_interval_fails() {
        let reference = gaussian_interval_draws(GAUSSIAN_REFERENCE_TARGET.seed, 1.0);
        let reference_verdicts = audit_interval_coverage(&GAUSSIAN_REFERENCE_TARGET, &reference);
        assert!(reference_verdicts.iter().all(|verdict| verdict.passed));

        let narrowed = gaussian_interval_draws(NARROW_INTERVAL_TARGET.seed, 0.5);
        let narrowed_verdicts = audit_interval_coverage(&NARROW_INTERVAL_TARGET, &narrowed);
        assert!(narrowed_verdicts.iter().any(|verdict| !verdict.passed));
        assert!(
            narrowed_verdicts
                .iter()
                .filter(|verdict| !verdict.passed)
                .all(|verdict| verdict.empirical < verdict.nominal)
        );
    }

    #[test]
    fn sbc_reference_rank_histogram_passes_and_planted_skew_fails() {
        let draws_per_rank = SBC_DECILE_BINS - 1;
        let reference_ranks = uniform_sbc_ranks(draws_per_rank);
        let reference = audit_sbc_ranks(
            &SBC_REFERENCE_TARGET,
            &reference_ranks,
            draws_per_rank,
            SBC_DECILE_BINS,
        );
        assert!(reference.passed);

        let skewed_ranks = vec![0usize; FAST_SBC_DRAWS];
        let skewed = audit_sbc_ranks(
            &SKEWED_SBC_TARGET,
            &skewed_ranks,
            draws_per_rank,
            SBC_DECILE_BINS,
        );
        assert!(!skewed.passed);
    }
}
