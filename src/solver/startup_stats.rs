//! Honest accounting for outer-solver seed validation.
//!
//! The legacy aggregate error built three integers — `generated`,
//! `attempted`, `rejected` — and dumped each rejection's text into a
//! single comma-joined `reasons: [...]` blob. Those names lied:
//! `attempted = min(generated, seed_budget)` rather than "number of
//! seeds we actually ran inner solves on", and `rejected` lumped
//! NaN-domain failures together with structural rank deficiencies and
//! never named *why* the cascade is unable to land on any seed.
//!
//! [`StartupStats`] replaces those counters with a category breakdown
//! derived from the structured [`InnerFailure`] classifier. The same
//! string-sentinel parser the inner-status module uses powers the
//! breakdown here, so a refactor of the inner solver to emit
//! [`InnerFailure`] natively flows through this module without changes
//! at the call site.
//!
//! The struct also drives the seed-loop's structural early-exit: when
//! every observed failure carries the same `(diagnosis, carrying_block)`
//! pair, every remaining ρ candidate will fail the same way, so the
//! outer skips them instead of paying a full joint-Newton inner solve
//! per duplicate.

use std::fmt::Write;

use crate::families::custom_family::KktRefusalDiagnosis;
use crate::families::inner_status::{InnerFailure, classify_inner_error};

/// Records one failed seed candidate along with its structured failure
/// classification, the validation phase tag that produced it, and the
/// 0-based seed index in the candidate list.
#[derive(Clone, Debug)]
pub(crate) struct SeedRejection {
    pub seed_idx: usize,
    pub phase: &'static str,
    pub failure: InnerFailure,
}

impl SeedRejection {
    pub(crate) fn from_message(seed_idx: usize, phase: &'static str, message: String) -> Self {
        Self {
            seed_idx,
            phase,
            failure: classify_inner_error(message),
        }
    }
}

/// Per-category counters built from a collection of [`SeedRejection`].
/// All counts are honest: `screened`, `exact_validated`, and
/// `solver_started` are populated by the seed loop directly; the
/// rejection categories are summed from the `InnerFailure` variants.
#[derive(Clone, Debug, Default)]
pub(crate) struct StartupStats {
    pub generated: usize,
    pub screened: usize,
    pub exact_validated: usize,
    pub solver_started: usize,
    pub rejected_by_kkt: usize,
    pub rejected_by_domain: usize,
    pub rejected_by_objective: usize,
    pub rejected_by_budget: usize,
    pub rejected_other: usize,
}

impl StartupStats {
    pub(crate) fn from_rejections(
        generated: usize,
        screened: usize,
        exact_validated: usize,
        solver_started: usize,
        rejections: &[SeedRejection],
    ) -> Self {
        let mut stats = Self {
            generated,
            screened,
            exact_validated,
            solver_started,
            ..Self::default()
        };
        for rej in rejections {
            match &rej.failure {
                InnerFailure::CertRefused { .. } => stats.rejected_by_kkt += 1,
                InnerFailure::LikelihoodFailure(_) => stats.rejected_by_domain += 1,
                InnerFailure::BudgetExhausted { .. } | InnerFailure::TrustRegionFloor { .. } => {
                    stats.rejected_by_budget += 1
                }
                InnerFailure::Other(msg) => {
                    if msg.contains("non-finite")
                        || msg.contains("not finite")
                        || msg.contains("Infinity")
                        || msg.contains("inf")
                    {
                        stats.rejected_by_objective += 1;
                    } else {
                        stats.rejected_other += 1;
                    }
                }
            }
        }
        stats
    }

    pub(crate) fn total_rejected(&self) -> usize {
        self.rejected_by_kkt
            + self.rejected_by_domain
            + self.rejected_by_objective
            + self.rejected_by_budget
            + self.rejected_other
    }
}

/// `(diagnosis, carrying_block)` key shared by all CertRefused
/// rejections. When every observed rejection carries the same key, the
/// outer seed loop short-circuits — there is no point burning a full
/// inner solve on each remaining ρ candidate just to watch the same
/// structural rank deficiency reject it.
pub(crate) type StructuralKey = (KktRefusalDiagnosis, Option<String>);

pub(crate) fn structural_key(failure: &InnerFailure) -> Option<StructuralKey> {
    match failure {
        InnerFailure::CertRefused {
            diagnosis,
            carrying_block,
            ..
        } => Some((*diagnosis, carrying_block.clone())),
        _ => None,
    }
}

/// `Some(key)` when every rejection in `rejections` is a CertRefused
/// failure with an identical `(diagnosis, carrying_block)` pair, and
/// the cascade has produced at least `min_count` observations. The
/// caller uses this to break the seed loop early and to format the
/// structural-cause diagnosis in the final error.
pub(crate) fn uniform_structural_key(
    rejections: &[SeedRejection],
    min_count: usize,
) -> Option<StructuralKey> {
    if rejections.len() < min_count {
        return None;
    }
    let mut iter = rejections.iter();
    let key = structural_key(&iter.next()?.failure)?;
    for rej in iter {
        let candidate = structural_key(&rej.failure)?;
        if candidate != key {
            return None;
        }
    }
    Some(key)
}

/// Render a structural-cause diagnosis hint based on the agreed-upon
/// `(diagnosis, carrying_block)` pair. The phrasing names the user's
/// next step (reduce knots / increase λ / wait for null-space
/// absorption) so the failure is actionable rather than just labelled.
pub(crate) fn structural_diagnosis_hint(key: &StructuralKey) -> String {
    let (diagnosis, carrying) = key;
    let carrying_label = carrying
        .as_deref()
        .map(|name| format!("smooth '{name}'"))
        .unwrap_or_else(|| "the smooth carrying the dominant KKT residual".to_string());
    match diagnosis {
        KktRefusalDiagnosis::RankDeficientHPen => format!(
            "structural rank deficiency in {carrying_label} — no seed is solvable. \
             Either reduce the smooth's knot count, increase its smoothing parameter, \
             or rely on the smooth-construction null-space absorption pass once it lands."
        ),
        KktRefusalDiagnosis::PhantomMultiplierWithWellConditionedH => format!(
            "every seed terminates at a phantom multiplier in {carrying_label} while H_pen \
             is well-conditioned — the active-set projection captures part but not all of \
             the gradient. Likely an incomplete inequality-constraint set or a basis whose \
             range still hides a near-null direction the data does not constrain."
        ),
        KktRefusalDiagnosis::ActiveSetIncomplete => format!(
            "every seed exits with an incomplete active set on {carrying_label}. The \
             outer cascade cannot grow the active set further without changing the \
             smooth's constraint family."
        ),
    }
}

/// Format the structured "no candidate seeds passed outer startup
/// validation" payload. Returns a single multi-line `String` because
/// `EstimationError::RemlOptimizationFailed` carries a single message
/// field.
pub(crate) fn format_no_seeds_passed(
    context: &str,
    stats: &StartupStats,
    rejections: &[SeedRejection],
    structural: Option<&StructuralKey>,
    early_exit_note: &str,
) -> String {
    let mut out = String::new();
    let _ = writeln!(
        &mut out,
        "no candidate seeds passed outer startup validation ({context}):"
    );
    let _ = writeln!(
        &mut out,
        "  generated={}, screened={}, exact_validated={}, solver_started={}",
        stats.generated, stats.screened, stats.exact_validated, stats.solver_started,
    );
    let _ = writeln!(
        &mut out,
        "  rejection breakdown: rejected_by_kkt={}, rejected_by_domain={}, \
         rejected_by_objective={}, rejected_by_budget={}, rejected_other={} (total={})",
        stats.rejected_by_kkt,
        stats.rejected_by_domain,
        stats.rejected_by_objective,
        stats.rejected_by_budget,
        stats.rejected_other,
        stats.total_rejected(),
    );
    if let Some(key) = structural {
        let _ = writeln!(
            &mut out,
            "  uniform CertRefused: diagnosis={}, carrying-block={}",
            key.0.as_str(),
            key.1.as_deref().unwrap_or("<unknown>"),
        );
        let _ = writeln!(&mut out, "  diagnosis: {}", structural_diagnosis_hint(key));
    }
    if !early_exit_note.is_empty() {
        let _ = writeln!(&mut out, "  {early_exit_note}");
    }
    if !rejections.is_empty() {
        let _ = writeln!(&mut out, "  per-seed reasons:");
        for rej in rejections {
            let _ = writeln!(
                &mut out,
                "    seed {} ({}): {}",
                rej.seed_idx,
                rej.phase,
                rej.failure.message(),
            );
        }
    }
    // Trim the trailing newline so the message embeds cleanly inside
    // `EstimationError::RemlOptimizationFailed("...")`.
    while out.ends_with('\n') {
        out.pop();
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cert_refused(seed_idx: usize, block: &str) -> SeedRejection {
        SeedRejection::from_message(
            seed_idx,
            "validation",
            format!(
                "cycle=7 cert REFUSED: residual=5.0e+05 > 4·tol=4.0e+03; \
                 carrying-block: {block} (idx=0, |g|=5.0e+05, |Sβ|=1.0e-03, \
                 |∇L-Sβ|=5.0e+05, |β|=1.0e+00, width=12); diagnosis: rank_deficient_H_pen; \
                 reduce knots"
            ),
        )
    }

    #[test]
    fn structural_key_extracts_diagnosis_only_for_cert_refused() {
        let cert = cert_refused(0, "time_surface").failure;
        let key = structural_key(&cert).expect("CertRefused must yield a structural key");
        assert_eq!(key.0, KktRefusalDiagnosis::RankDeficientHPen);
        assert_eq!(key.1.as_deref(), Some("time_surface"));

        let domain = SeedRejection::from_message(
            0,
            "validation",
            "likelihood evaluation failed: NaN response".to_string(),
        )
        .failure;
        assert!(
            structural_key(&domain).is_none(),
            "non-cert-refused failures must not present a structural key"
        );
    }

    #[test]
    fn startup_stats_categorises_cert_refused() {
        let rejections = vec![
            cert_refused(0, "time_surface"),
            cert_refused(1, "time_surface"),
        ];
        let stats = StartupStats::from_rejections(5, 5, 5, 0, &rejections);
        assert_eq!(stats.generated, 5);
        assert_eq!(stats.solver_started, 0);
        assert_eq!(stats.rejected_by_kkt, 2);
        assert_eq!(stats.rejected_by_domain, 0);
        assert_eq!(stats.total_rejected(), 2);
    }

    #[test]
    fn uniform_structural_key_detects_repeating_cert_refused() {
        let rejections = vec![
            cert_refused(0, "time_surface"),
            cert_refused(1, "time_surface"),
            cert_refused(2, "time_surface"),
        ];
        let key = uniform_structural_key(&rejections, 2).expect("uniform key");
        assert_eq!(key.0, KktRefusalDiagnosis::RankDeficientHPen);
        assert_eq!(key.1.as_deref(), Some("time_surface"));
    }

    #[test]
    fn uniform_structural_key_rejects_mixed_blocks() {
        let rejections = vec![cert_refused(0, "time_surface"), cert_refused(1, "marginal")];
        assert!(uniform_structural_key(&rejections, 2).is_none());
    }

    #[test]
    fn uniform_structural_key_rejects_mixed_failure_kinds() {
        let cert = cert_refused(0, "time_surface");
        let domain = SeedRejection::from_message(
            1,
            "validation",
            "likelihood evaluation failed: NaN response".to_string(),
        );
        assert!(uniform_structural_key(&[cert, domain], 2).is_none());
    }

    /// Simulates the outer seed loop's iterative behaviour: failures
    /// arrive one at a time, and after each one we probe whether
    /// `uniform_structural_key` is ready to fire the structural
    /// early-exit (Stage 3). The contract is:
    ///   - after one failure the key is not yet stable (min_count=2)
    ///   - after two identical failures it fires
    ///   - if the third failure deviates the key would no longer be uniform
    /// The seed loop in `outer_strategy.rs` mirrors this exact pattern,
    /// so the test pins the behaviour without needing to spin up the
    /// full outer optimiser.
    #[test]
    fn iterative_loop_triggers_early_exit_at_second_uniform_failure() {
        const MIN_COUNT: usize = 2;
        let mut rejections: Vec<SeedRejection> = Vec::new();

        rejections.push(cert_refused(0, "time_surface"));
        assert!(
            uniform_structural_key(&rejections, MIN_COUNT).is_none(),
            "single failure must not trigger early-exit; threshold guards \
             against transient one-off CertRefused at exploration seeds"
        );

        rejections.push(cert_refused(1, "time_surface"));
        let key = uniform_structural_key(&rejections, MIN_COUNT)
            .expect("second matching failure must trigger early-exit");
        assert_eq!(key.0, KktRefusalDiagnosis::RankDeficientHPen);
        assert_eq!(key.1.as_deref(), Some("time_surface"));

        // If we kept iterating past early-exit (hypothetically) a
        // deviating third failure would invalidate the key. Verify the
        // equality check is strict on (diagnosis, block) — never on
        // diagnosis alone.
        rejections.push(cert_refused(2, "marginal"));
        assert!(
            uniform_structural_key(&rejections, MIN_COUNT).is_none(),
            "structural key must be invalidated when a sibling block \
             carries the residual at a later seed"
        );
    }

    /// Pins the structural-cause hint copy. The phrasing names the
    /// user's next action so the error is actionable; the test guards
    /// against accidental message regressions when the diagnosis enum
    /// is extended.
    #[test]
    fn structural_diagnosis_hint_names_next_action_per_diagnosis() {
        let rank = structural_diagnosis_hint(&(
            KktRefusalDiagnosis::RankDeficientHPen,
            Some("time_surface".to_string()),
        ));
        assert!(rank.contains("structural rank deficiency"));
        assert!(rank.contains("time_surface"));
        assert!(rank.contains("reduce the smooth's knot count"));

        let phantom = structural_diagnosis_hint(&(
            KktRefusalDiagnosis::PhantomMultiplierWithWellConditionedH,
            None,
        ));
        assert!(phantom.contains("phantom multiplier"));
        assert!(phantom.contains("the smooth carrying the dominant KKT residual"));

        let active = structural_diagnosis_hint(&(
            KktRefusalDiagnosis::ActiveSetIncomplete,
            Some("constraint_block".to_string()),
        ));
        assert!(active.contains("incomplete active set"));
        assert!(active.contains("constraint_block"));
    }

    /// Smoke test that the full final-error formatter (used by
    /// `run_outer_with_plan` when no seed converges) builds a payload
    /// that names every field the user needs to triage a failed fit:
    /// honest counters, per-category breakdown, structural hint, and
    /// the original per-seed messages.
    #[test]
    fn format_no_seeds_passed_payload_carries_full_triage_surface() {
        let rejections = vec![
            cert_refused(0, "time_surface"),
            cert_refused(1, "time_surface"),
            cert_refused(2, "time_surface"),
        ];
        let stats = StartupStats::from_rejections(5, 5, 3, 0, &rejections);
        let key = uniform_structural_key(&rejections, 2);
        let msg = format_no_seeds_passed(
            "custom family",
            &stats,
            &rejections,
            key.as_ref(),
            "early-exit triggered: every observed seed reported the same structural CertRefused",
        );
        // Honest counters
        assert!(msg.contains("generated=5"));
        assert!(msg.contains("exact_validated=3"));
        assert!(msg.contains("solver_started=0"));
        // Per-category breakdown
        assert!(msg.contains("rejected_by_kkt=3"));
        // Structural diagnosis
        assert!(msg.contains("diagnosis=rank_deficient_H_pen"));
        assert!(msg.contains("carrying-block=time_surface"));
        assert!(msg.contains("structural rank deficiency"));
        // Early-exit note
        assert!(msg.contains("early-exit triggered"));
        // Per-seed reasons preserved (the original messages still bubble)
        assert!(msg.contains("seed 0 (validation)"));
        assert!(msg.contains("seed 2 (validation)"));
    }

    #[test]
    fn format_no_seeds_passed_emits_structured_payload() {
        let rejections = vec![
            cert_refused(0, "time_surface"),
            cert_refused(1, "time_surface"),
        ];
        let stats = StartupStats::from_rejections(5, 5, 5, 0, &rejections);
        let key = uniform_structural_key(&rejections, 2);
        let msg = format_no_seeds_passed("custom family", &stats, &rejections, key.as_ref(), "");
        assert!(msg.contains("generated=5"));
        assert!(msg.contains("solver_started=0"));
        assert!(msg.contains("rejected_by_kkt=2"));
        assert!(msg.contains("diagnosis=rank_deficient_H_pen"));
        assert!(msg.contains("carrying-block=time_surface"));
        assert!(msg.contains("structural rank deficiency"));
    }
}
