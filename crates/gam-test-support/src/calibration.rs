//! Standing calibration-audit harness for uncertainty-producing APIs.
//!
//! The production crates own the statistical estimators.  This module owns the
//! reusable *quality gate*: every uncertainty surface is represented by a
//! [`CalibrationTarget`], is checked against a registry, and is audited with the
//! mode appropriate for its payload (coverage/size or SBC ranks).  The harness is
//! intentionally model-agnostic so new UQ surfaces can register cheap planted
//! truth probes without depending on one large integration fixture.

use std::collections::BTreeSet;
use std::fmt;

/// Default replicate budget for fast calibration probes.
pub const FAST_REPLICATES: usize = 200;
/// Default replicate budget for slow-suite calibration probes.
pub const SLOW_REPLICATES: usize = 2_000;
/// Default number of observations per planted-truth replicate.
pub const FAST_ROWS: usize = 64;
/// Nominal interval coverage levels audited by the standing gate.
pub const COVERAGE_LEVELS: [f64; 3] = [0.80, 0.90, 0.95];
/// Test-size levels audited by the standing gate.
pub const TEST_ALPHA_LEVELS: [f64; 3] = [0.01, 0.05, 0.10];

const ANTI_CONSERVATIVE_SLACK: f64 = 0.015;
const CONSERVATIVE_REPORT_SLACK: f64 = 0.06;
const SBC_ECDF_SLACK: f64 = 0.12;
const SBC_CHI_SQUARE_PER_DOF_MAX: f64 = 2.25;

/// The family of calibration audit a target requires.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CalibrationKind {
    /// Confidence, credible, conformal, or predictive intervals/bands.
    Coverage,
    /// Null p-values from LR/Wald/score-style tests.
    TestSize,
    /// Posterior draws/ranks checked by simulation-based calibration.
    SbcRanks,
}

/// A source-code/public-payload field that carries uncertainty and therefore
/// must have a named calibration target in the registry.
#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd)]
pub struct UncertaintySurface {
    pub name: &'static str,
    pub owner: &'static str,
    pub field: &'static str,
}

impl UncertaintySurface {
    pub const fn new(name: &'static str, owner: &'static str, field: &'static str) -> Self {
        Self { name, owner, field }
    }
}

/// A registered UQ surface.  Tests may attach the appropriate audit vectors;
/// production-specific fitting remains in the test that constructs the target.
#[derive(Clone, Debug)]
pub struct CalibrationTarget {
    pub name: &'static str,
    pub kind: CalibrationKind,
    pub seed: u64,
    pub rows: usize,
    pub replicates: usize,
    pub coverage: Vec<CoverageObservation>,
    pub test_size: Vec<TestSizeObservation>,
    pub sbc: Vec<SbcRankObservation>,
    pub forensic_notes: Vec<String>,
}

impl CalibrationTarget {
    pub fn new(name: &'static str, kind: CalibrationKind) -> Self {
        Self {
            name,
            kind,
            seed: stable_seed(name),
            rows: FAST_ROWS,
            replicates: FAST_REPLICATES,
            coverage: Vec::new(),
            test_size: Vec::new(),
            sbc: Vec::new(),
            forensic_notes: Vec::new(),
        }
    }

    pub fn with_budget(mut self, rows: usize, replicates: usize) -> Self {
        self.rows = rows;
        self.replicates = replicates;
        self
    }

    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = seed;
        self
    }

    pub fn with_forensic_note(mut self, note: impl Into<String>) -> Self {
        self.forensic_notes.push(note.into());
        self
    }
}

#[derive(Clone, Copy, Debug)]
pub struct CoverageObservation {
    pub nominal: f64,
    pub covered: bool,
}

#[derive(Clone, Copy, Debug)]
pub struct TestSizeObservation {
    pub alpha: f64,
    pub rejected: bool,
}

#[derive(Clone, Copy, Debug)]
pub struct SbcRankObservation {
    /// Zero-based rank of the planted parameter among posterior draws.
    pub rank: usize,
    /// Number of posterior draws used to form the rank; ranks are in
    /// `0..=draws`.
    pub draws: usize,
}

#[derive(Clone, Debug, PartialEq)]
pub struct CalibrationVerdict {
    pub target: &'static str,
    pub passed: bool,
    pub diagnostics: Vec<String>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CompletenessError {
    pub missing: Vec<UncertaintySurface>,
}

impl fmt::Display for CompletenessError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "uncertainty surfaces missing calibration targets:")?;
        for surface in &self.missing {
            write!(
                f,
                " {}::{} ({})",
                surface.owner, surface.field, surface.name
            )?;
        }
        Ok(())
    }
}

impl std::error::Error for CompletenessError {}

/// Enforce the registry contract: every discovered `se`/`interval`/`p_value`-
/// carrying payload field must appear as a registered calibration target.
pub fn assert_registry_complete(
    discovered: &[UncertaintySurface],
    registry: &[CalibrationTarget],
) -> Result<(), CompletenessError> {
    let registered = registry
        .iter()
        .map(|target| target.name)
        .collect::<BTreeSet<_>>();
    let missing = discovered
        .iter()
        .filter(|surface| !registered.contains(surface.name))
        .cloned()
        .collect::<Vec<_>>();
    if missing.is_empty() {
        Ok(())
    } else {
        Err(CompletenessError { missing })
    }
}

/// Canonical target names for the uncertainty surfaces that motivated the
/// standing gate.  Production-specific tests can extend this vector locally,
/// but these names keep the defect-class registry explicit and searchable.
pub fn core_uncertainty_registry() -> Vec<CalibrationTarget> {
    vec![
        CalibrationTarget::new("alo_exact_frozen_curvature_loo_se", CalibrationKind::Coverage),
        CalibrationTarget::new("mean_prediction_interval", CalibrationKind::Coverage),
        CalibrationTarget::new("smooth_posterior_band", CalibrationKind::Coverage),
        CalibrationTarget::new("post_selection_lr_p_value", CalibrationKind::TestSize),
        CalibrationTarget::new("bartlett_lawley_p_value", CalibrationKind::TestSize),
        CalibrationTarget::new("joint_response_posterior_se", CalibrationKind::Coverage),
        CalibrationTarget::new(
            "delta_method_vs_posterior_simulation_interval",
            CalibrationKind::Coverage,
        ),
        CalibrationTarget::new("rho_posterior_certificate", CalibrationKind::SbcRanks),
        CalibrationTarget::new("nuts_posterior_surface", CalibrationKind::SbcRanks),
        CalibrationTarget::new("polya_gamma_posterior_surface", CalibrationKind::SbcRanks),
        CalibrationTarget::new("conformal_prediction_interval", CalibrationKind::Coverage),
        CalibrationTarget::new("predictive_interval", CalibrationKind::Coverage),
    ]
}

pub fn audit_target(target: &CalibrationTarget) -> CalibrationVerdict {
    match target.kind {
        CalibrationKind::Coverage => audit_coverage(target),
        CalibrationKind::TestSize => audit_test_size(target),
        CalibrationKind::SbcRanks => audit_sbc(target),
    }
}

pub fn audit_registry(registry: &[CalibrationTarget]) -> Vec<CalibrationVerdict> {
    registry.iter().map(audit_target).collect()
}

fn audit_coverage(target: &CalibrationTarget) -> CalibrationVerdict {
    let mut diagnostics = Vec::new();
    let mut passed = !target.coverage.is_empty();
    for &nominal in &COVERAGE_LEVELS {
        let (hits, total) = count_coverage(&target.coverage, nominal);
        if total == 0 {
            continue;
        }
        let empirical = hits as f64 / total as f64;
        diagnostics.push(format!(
            "coverage nominal={nominal:.2} empirical={empirical:.3} n={total}"
        ));
        if empirical + ANTI_CONSERVATIVE_SLACK < nominal {
            passed = false;
        }
        if empirical > nominal + CONSERVATIVE_REPORT_SLACK {
            diagnostics.push(format!(
                "conservative slack exceeded at nominal={nominal:.2}"
            ));
        }
    }
    append_forensics(target, &mut diagnostics);
    CalibrationVerdict {
        target: target.name,
        passed,
        diagnostics,
    }
}

fn audit_test_size(target: &CalibrationTarget) -> CalibrationVerdict {
    let mut diagnostics = Vec::new();
    let mut passed = !target.test_size.is_empty();
    for &alpha in &TEST_ALPHA_LEVELS {
        let (rejects, total) = count_rejections(&target.test_size, alpha);
        if total == 0 {
            continue;
        }
        let empirical = rejects as f64 / total as f64;
        diagnostics.push(format!(
            "test-size alpha={alpha:.2} empirical={empirical:.3} n={total}"
        ));
        if empirical > alpha + ANTI_CONSERVATIVE_SLACK {
            passed = false;
        }
    }
    append_forensics(target, &mut diagnostics);
    CalibrationVerdict {
        target: target.name,
        passed,
        diagnostics,
    }
}

fn audit_sbc(target: &CalibrationTarget) -> CalibrationVerdict {
    let mut diagnostics = Vec::new();
    if target.sbc.is_empty() {
        return CalibrationVerdict {
            target: target.name,
            passed: false,
            diagnostics: vec!["no SBC ranks supplied".to_string()],
        };
    }
    let draws = target.sbc[0].draws;
    let bins = (draws + 1).min(20);
    let mut counts = vec![0usize; bins];
    for obs in &target.sbc {
        if obs.draws != draws || obs.rank > obs.draws {
            return CalibrationVerdict {
                target: target.name,
                passed: false,
                diagnostics: vec!["invalid/nonconformable SBC ranks".to_string()],
            };
        }
        let idx = (obs.rank * bins) / (draws + 1);
        counts[idx.min(bins - 1)] += 1;
    }
    let n = target.sbc.len() as f64;
    let expected = n / bins as f64;
    let chi_per_dof = counts
        .iter()
        .map(|&c| {
            let d = c as f64 - expected;
            d * d / expected
        })
        .sum::<f64>()
        / (bins.saturating_sub(1).max(1)) as f64;
    let ecdf_max = sbc_ecdf_supremum(&target.sbc, draws);
    diagnostics.push(format!(
        "sbc bins={bins} chi_square_per_dof={chi_per_dof:.3} ecdf_sup={ecdf_max:.3}"
    ));
    append_forensics(target, &mut diagnostics);
    CalibrationVerdict {
        target: target.name,
        passed: chi_per_dof <= SBC_CHI_SQUARE_PER_DOF_MAX && ecdf_max <= SBC_ECDF_SLACK,
        diagnostics,
    }
}

fn count_coverage(obs: &[CoverageObservation], nominal: f64) -> (usize, usize) {
    obs.iter()
        .filter(|o| (o.nominal - nominal).abs() < 1.0e-12)
        .fold((0, 0), |(h, n), o| (h + usize::from(o.covered), n + 1))
}

fn count_rejections(obs: &[TestSizeObservation], alpha: f64) -> (usize, usize) {
    obs.iter()
        .filter(|o| (o.alpha - alpha).abs() < 1.0e-12)
        .fold((0, 0), |(r, n), o| (r + usize::from(o.rejected), n + 1))
}

fn sbc_ecdf_supremum(obs: &[SbcRankObservation], draws: usize) -> f64 {
    let mut ranks = obs.iter().map(|o| o.rank).collect::<Vec<_>>();
    ranks.sort_unstable();
    let n = ranks.len() as f64;
    ranks
        .iter()
        .enumerate()
        .map(|(i, &rank)| {
            let empirical = (i + 1) as f64 / n;
            let uniform = (rank + 1) as f64 / (draws + 1) as f64;
            (empirical - uniform).abs()
        })
        .fold(0.0, f64::max)
}

fn append_forensics(target: &CalibrationTarget, diagnostics: &mut Vec<String>) {
    diagnostics.extend(
        target
            .forensic_notes
            .iter()
            .map(|note| format!("forensics: {note}")),
    );
}

fn stable_seed(name: &str) -> u64 {
    name.bytes().fold(0xcbf29ce484222325, |hash, byte| {
        (hash ^ u64::from(byte)).wrapping_mul(0x100000001b3)
    })
}
