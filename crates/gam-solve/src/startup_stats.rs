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
//! [`StartupStats`] replaces those counters with a category breakdown derived
//! from [`InnerFailure`]. Typed objective sources are projected directly;
//! string classification is reserved for producers that emitted only prose.
//!
//! When every observed failure carries the same genuinely structural
//! `(diagnosis, carrying_block)` pair, the refusal names that structural
//! cause. Numerical certificate refusals, such as a phantom multiplier with
//! well-conditioned `H_pen`, are deliberately excluded: they are statements
//! about one ρ, not about the problem.

use std::fmt::Write;

use crate::inner_status::{InnerFailure, classify_estimation_error, classify_inner_error};
use gam_problem::{EstimationError, diagnostics::KktRefusalDiagnosis};
use opt::ObjectiveEvalError;

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
    /// Preserve an objective producer's typed source before any orchestration
    /// layer renders it. `into_objective_error` attaches the originating
    /// [`EstimationError`], so a custom-family
    /// `InnerSolveNotConverged` reaches startup accounting with every terminal
    /// field intact.
    pub(crate) fn from_objective_error(
        seed_idx: usize,
        phase: &'static str,
        error: &ObjectiveEvalError,
    ) -> Self {
        let message = error.message().to_string();
        let failure = error
            .downcast_ref::<EstimationError>()
            .map(|source| classify_estimation_error(source, message.clone()))
            .unwrap_or_else(|| classify_inner_error(message));
        Self {
            seed_idx,
            phase,
            failure,
        }
    }

    /// Preserve a direct engine error at rejection sites that do not cross the
    /// `opt` objective boundary.
    pub(crate) fn from_estimation_error(
        seed_idx: usize,
        phase: &'static str,
        error: &EstimationError,
    ) -> Self {
        Self {
            seed_idx,
            phase,
            failure: classify_estimation_error(error, error.to_string()),
        }
    }

    /// Record a rejection that reaches startup accounting only as prose.
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
    pub rejected_by_nonconvergence: usize,
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
                InnerFailure::InnerSolveNotConverged { .. } => {
                    stats.rejected_by_nonconvergence += 1
                }
                InnerFailure::CertRefused { .. } => stats.rejected_by_kkt += 1,
                InnerFailure::LikelihoodFailure(_) => stats.rejected_by_domain += 1,
                InnerFailure::BudgetExhausted { .. } | InnerFailure::TrustRegionFloor { .. } => {
                    stats.rejected_by_budget += 1
                }
                // A pre-fit identifiability failure is structural in
                // the same shape as a KKT cert refusal — bucket it
                // with `rejected_by_kkt` so the seed-screening
                // structural early-exit accounting sees it.
                InnerFailure::IdentifiabilityFailure { .. } => stats.rejected_by_kkt += 1,
                // `Other` is the variant `classify_inner_error` reaches when
                // none of its sentinels matched: "still rejected, the cascade
                // cannot say why". Counting it under a bucket that names a
                // cause re-asserts by substring exactly what the classifier
                // just declined to conclude.
                //
                // The bucket this used to feed, `rejected_by_objective`, is
                // gone rather than repaired, because it had no correct
                // producer to repair it for (gam#2651):
                //
                //   * `non-finite` / `not finite` were unreachable here —
                //     `classify_inner_error` routes both to `LikelihoodFailure`
                //     before this point, i.e. to `rejected_by_domain`, which is
                //     where a genuinely non-finite objective has always been
                //     counted;
                //   * `Infinity` is subsumed by `inf`, which is tested first;
                //   * `inf` matched EVERY joint-Newton refusal, because the
                //     terminal-state Display renders the field NAME `step_inf=`.
                //     Measured on the binomial location-scale wiggle spatial
                //     fixture: four non-convergences with finite beta and finite
                //     objective reported as `rejected_by_objective=4`, which sent
                //     a reader hunting a non-finite objective that never existed.
                //
                // So the test could never separate anything: `rejected_other`
                // was unreachable for that whole family and the bucket was a
                // confident wrong label. This is the same defect
                // `classify_inner_error` records thirty lines above for
                // `rejected_by_budget`, and it gets the same answer — an honest
                // "unclassified" beats a confident wrong label.
                InnerFailure::Other(_) => stats.rejected_other += 1,
            }
        }
        stats
    }

    pub(crate) fn total_rejected(&self) -> usize {
        self.rejected_by_kkt
            + self.rejected_by_domain
            + self.rejected_by_nonconvergence
            + self.rejected_by_budget
            + self.rejected_other
    }
}

/// `(diagnosis, carrying_block)` key shared by genuinely structural
/// rejections. When every observed rejection carries the same key, the
/// outer seed loop short-circuits — there is no point burning a full
/// inner solve on each remaining ρ candidate just to watch the same
/// structural rank/alias/active-set defect reject it.
pub(crate) type StructuralKey = (KktRefusalDiagnosis, Option<String>);

pub(crate) fn structural_key(failure: &InnerFailure) -> Option<StructuralKey> {
    match failure {
        InnerFailure::CertRefused {
            diagnosis,
            carrying_block,
            ..
        } => match diagnosis {
            KktRefusalDiagnosis::RankDeficientHPen
            | KktRefusalDiagnosis::ActiveSetIncomplete
            | KktRefusalDiagnosis::AliasingDetectedAtFit => {
                Some((*diagnosis, carrying_block.clone()))
            }
            KktRefusalDiagnosis::PhantomMultiplierWithWellConditionedH => None,
        },
        _ => None,
    }
}

/// `Some(key)` when every rejection in `rejections` is a genuinely
/// structural failure with an identical `(diagnosis, carrying_block)`
/// pair, and the cascade has produced at least `min_count`
/// observations. The caller uses this to break the seed loop early and
/// to format the structural-cause diagnosis in the final error.
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
        KktRefusalDiagnosis::AliasingDetectedAtFit => format!(
            "cross-block identifiability aliasing surfaced at {carrying_label} during the \
             inner solve — a binding active set or λ-dependent direction created an alias \
             the pre-fit audit could not see. Structural fix only: drop or reparameterise \
             the aliased block; no rho-anneal will recover."
        ),
    }
}

/// One line naming the stage that actually failed, when the counters say it
/// unambiguously.
///
/// The headline of this refusal is "no candidate seeds passed outer startup
/// validation", which points a reader at seed GENERATION. That is often not
/// where the failure is, and the counters printed directly beneath it say so —
/// but only if the reader knows to divide them.
///
/// MEASURED 2026-09-05, `bench/gha_results/python-contracts/py1512_junit.xml`
/// (CI run 33941725421): six of the eight tests carrying this refusal report
///
/// ```text
/// generated=13, screened=13, exact_validated=13, solver_started=0
/// rejected_by_kkt=0, rejected_by_domain=13, rejected_by_nonconvergence=0,
/// rejected_by_budget=0, rejected_other=0
/// all 13 seeds, phase=validation:
///     "outer eval failed: objective returned a non-finite cost"
/// ```
///
/// Every seed was generated, screened AND exact-validated, and the objective
/// then returned a non-finite cost at all thirteen. Nothing about seeding
/// failed. Reading that took someone an hour of auditing seed generation
/// because the headline named the stage that REPORTED the failure rather than
/// the stage that CAUSED it.
///
/// This is deliberately conservative: it speaks only when the evidence is
/// unambiguous — every seed survived to the same stage, no seed reached the
/// solver, every rejection fell in ONE category, and every per-seed reason is
/// the SAME string. Any heterogeneity and it says nothing, because a summary
/// that guesses is worse than a summary that is absent.
fn uniform_failure_attribution(stats: &StartupStats, rejections: &[SeedRejection]) -> Option<String> {
    if stats.generated == 0 || stats.solver_started > 0 {
        return None;
    }
    // Exactly one non-empty rejection category, holding every rejection.
    let categories: [(&str, usize); 5] = [
        ("KKT", stats.rejected_by_kkt),
        ("domain", stats.rejected_by_domain),
        ("non-convergence", stats.rejected_by_nonconvergence),
        ("budget", stats.rejected_by_budget),
        ("other", stats.rejected_other),
    ];
    let mut only: Option<(&str, usize)> = None;
    for (name, count) in categories {
        if count == 0 {
            continue;
        }
        if only.is_some() {
            return None;
        }
        only = Some((name, count));
    }
    let (category, count) = only?;
    if count != stats.total_rejected() {
        return None;
    }
    // Every per-seed reason identical, and every seed accounted for.
    let first = rejections.first()?;
    let reason = first.failure.message();
    if rejections.len() != count
        || rejections.iter().any(|r| r.failure.message() != reason)
    {
        return None;
    }
    // How far the seeds got before the uniform rejection.
    let reached = if stats.exact_validated == stats.generated {
        "generated, screened and exact-validated"
    } else if stats.screened == stats.generated {
        "generated and screened"
    } else {
        "generated"
    };
    Some(format!(
        "attribution: this is NOT a seeding failure. All {} candidate seeds were {}, \
         none reached the solver, and all {count} were rejected in the {category} phase for \
         the SAME reason: {reason}. Look there, not at seed generation.",
        stats.generated, reached,
    ))
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
    writeln!(
        &mut out,
        "no candidate seeds passed outer startup validation ({context}):"
    )
    .expect("writing to String cannot fail");
    writeln!(
        &mut out,
        "  generated={}, screened={}, exact_validated={}, solver_started={}",
        stats.generated, stats.screened, stats.exact_validated, stats.solver_started,
    )
    .expect("writing to String cannot fail");
    writeln!(
        &mut out,
        "  rejection breakdown: rejected_by_kkt={}, rejected_by_domain={}, \
         rejected_by_nonconvergence={}, rejected_by_budget={}, rejected_other={} (total={})",
        stats.rejected_by_kkt,
        stats.rejected_by_domain,
        stats.rejected_by_nonconvergence,
        stats.rejected_by_budget,
        stats.rejected_other,
        stats.total_rejected(),
    )
    .expect("writing to String cannot fail");
    if let Some(attribution) = uniform_failure_attribution(stats, rejections) {
        writeln!(&mut out, "  {attribution}").expect("writing to String cannot fail");
    }
    if let Some(key) = structural {
        writeln!(
            &mut out,
            "  uniform CertRefused: diagnosis={}, carrying-block={}",
            key.0.as_str(),
            key.1.as_deref().unwrap_or("<unknown>"),
        )
        .expect("writing to String cannot fail");
        writeln!(&mut out, "  diagnosis: {}", structural_diagnosis_hint(key))
            .expect("writing to String cannot fail");
    }
    if !early_exit_note.is_empty() {
        writeln!(&mut out, "  {early_exit_note}").expect("writing to String cannot fail");
    }
    if !rejections.is_empty() {
        writeln!(&mut out, "  per-seed reasons:").expect("writing to String cannot fail");
        for rej in rejections {
            writeln!(
                &mut out,
                "    seed {} ({}): {}",
                rej.seed_idx,
                rej.phase,
                rej.failure.message(),
            )
            .expect("writing to String cannot fail");
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

    fn phantom_refused(seed_idx: usize, block: &str) -> SeedRejection {
        SeedRejection::from_message(
            seed_idx,
            "validation",
            format!(
                "cycle=7 cert REFUSED: residual=5.0e+00 > 4·tol=4.0e-06; \
                 carrying-block: {block} (idx=0, |g|=5.0e+00, |Sβ|=1.0e-03, \
                 |∇L-Sβ|=5.0e+00, |β|=5.0e+01, width=20); \
                 H_pen spectrum: λ_max=1.0e+03, λ_min=1.0e+00, cond=1.0e+03; \
                 diagnosis: phantom_multiplier_with_well_conditioned_H"
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

        let phantom = phantom_refused(0, "marginal_surface").failure;
        assert!(
            structural_key(&phantom).is_none(),
            "well-conditioned phantom multipliers are rho-local certificate refusals, not structural seed-loop keys"
        );
    }

    /// gam#2651: a joint-Newton non-convergence must not be counted under a
    /// bucket that names a cause it cannot know. The message is the real one —
    /// `InnerConvergenceTerminalState::JointNewton`'s Display renders the field
    /// NAME `step_inf=`, which is what the deleted `rejected_by_objective`
    /// substring test matched on every such refusal.
    #[test]
    fn joint_newton_non_convergence_is_unclassified_not_an_objective_failure_2651() {
        // Verbatim shape of the refusal, from the binomial location-scale
        // wiggle spatial fixture at origin/main.
        let message = "custom-family inner solve did not converge after 48 cycle(s)             [joint-Newton terminal cycle 47: stationarity_residual=6.950377e-1             (tol=1.677281e-11), step_inf=1.734422e0 (tol=1.026406e-10),             resolvable_negative_curvature=false, best_stationarity_residual=8.095899e-3             (last improved 4 cycle(s) before this one)]"
            .to_string();

        // The trap must still be present in the input, or this test proves
        // nothing about the classification that has to survive it.
        assert!(
            message.contains("inf"),
            "fixture must still carry the `inf` substring the old test matched;              if the terminal-state field names changed, re-derive this fixture              from a real refusal rather than deleting the assertion"
        );
        assert!(
            !message.contains("non-finite") && !message.contains("NaN"),
            "fixture must NOT claim a non-finite quantity: the point is that a              finite-beta, finite-objective refusal was being counted as one"
        );

        let rejection = SeedRejection::from_message(0, "validation", message);
        assert!(
            matches!(rejection.failure, InnerFailure::Other(_)),
            "a non-convergence with no budget/floor/cert/domain sentinel is              unclassified, got {:?}",
            rejection.failure
        );

        let stats = StartupStats::from_rejections(4, 4, 4, 0, &[rejection]);
        assert_eq!(
            stats.rejected_other, 1,
            "an unclassified refusal belongs in `rejected_other`"
        );
        assert_eq!(
            stats.rejected_by_domain, 0,
            "nothing here reported a domain failure"
        );
        assert_eq!(
            stats.rejected_by_kkt, 0,
            "no certificate refusal was produced"
        );
        assert_eq!(
            stats.rejected_by_budget, 0,
            "no budget was reported exhausted"
        );
        assert_eq!(stats.total_rejected(), 1);

        // The rendered breakdown must not offer a bucket with no producer.
        let rendered = format_no_seeds_passed(
            "custom family",
            &StartupStats::from_rejections(4, 4, 4, 0, &[]),
            &[],
            None,
            "",
        );
        assert!(
            !rendered.contains("rejected_by_objective"),
            "the breakdown must not print a bucket nothing can correctly produce, got:
{rendered}"
        );
    }

    /// Non-vacuity for the above: a refusal that genuinely reports a non-finite
    /// quantity is still counted, and still counted as a DOMAIN failure — the
    /// bucket a non-finite objective has always belonged in.
    #[test]
    fn genuinely_non_finite_refusal_is_still_counted_as_domain_2651() {
        let rejection = SeedRejection::from_message(
            0,
            "validation",
            "custom-family objective returned a non-finite cost at the seed".to_string(),
        );
        assert!(
            matches!(rejection.failure, InnerFailure::LikelihoodFailure(_)),
            "a non-finite report is a likelihood/domain failure, got {:?}",
            rejection.failure
        );
        let stats = StartupStats::from_rejections(1, 1, 1, 0, &[rejection]);
        assert_eq!(
            stats.rejected_by_domain, 1,
            "a real non-finite failure must still be counted"
        );
        assert_eq!(
            stats.rejected_other, 0,
            "and must not fall through to unclassified"
        );
        assert_eq!(stats.total_rejected(), 1);
    }

    #[test]
    fn objective_boundary_preserves_typed_joint_newton_terminal_state_2658() {
        let terminal = gam_problem::InnerConvergenceTerminalState::JointNewton {
            cycle: 47,
            stationarity_residual: 6.950377e-1,
            residual_tol: 1.677281e-11,
            // Consistent with the tol above: `tol = 1e-11 · (1 + scale)`.
            stationarity_scale: 0.677281,
            step_inf: 1.734422,
            step_tol: 1.026406e-10,
            resolvable_negative_curvature: false,
            best_stationarity_residual: 8.095899e-3,
            cycles_since_best_residual: 4,
            termination_reason: gam_problem::JointNewtonTerminalReason::CycleBudget,
        };
        let source =
            EstimationError::CustomFamily(gam_problem::CustomFamilyError::InnerSolveNotConverged {
                cycles: 48,
                terminal: Some(terminal.clone()),
                kkt_residual: Some(6.950377e-1),
                kkt_tol: Some(1.677281e-11),
                theta_dim: 7,
                rho_dim: 5,
                psi_dim: 2,
                cycle_budget: None,
                carrying_block: None,
            });
        let objective_error =
            ObjectiveEvalError::recoverable_from(source).with_context("outer eval failed");
        let rejection = SeedRejection::from_objective_error(3, "validation", &objective_error);

        match &rejection.failure {
            InnerFailure::InnerSolveNotConverged { message } => {
                assert!(message.starts_with("outer eval failed:"));
            }
            other => panic!("typed refusal was reclassified: {other:?}"),
        }

        let stats = StartupStats::from_rejections(4, 4, 4, 0, &[rejection.clone()]);
        assert_eq!(stats.rejected_by_nonconvergence, 1);
        assert_eq!(stats.rejected_by_budget, 0);
        assert_eq!(stats.rejected_other, 0);
        assert_eq!(stats.total_rejected(), 1);
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

    #[test]
    fn uniform_structural_key_ignores_repeated_phantom_multiplier_refusals() {
        let rejections = vec![
            phantom_refused(0, "marginal_surface"),
            phantom_refused(1, "marginal_surface"),
            phantom_refused(2, "marginal_surface"),
        ];
        assert!(
            uniform_structural_key(&rejections, 2).is_none(),
            "phantom_multiplier_with_well_conditioned_H is recoverable by trying another rho seed; startup must not skip sibling seeds"
        );
    }

    #[test]
    fn no_seeds_payload_does_not_call_phantom_refusals_structural() {
        let rejections = vec![
            phantom_refused(0, "marginal_surface"),
            phantom_refused(1, "marginal_surface"),
        ];
        let stats = StartupStats::from_rejections(5, 5, 2, 0, &rejections);
        let key = uniform_structural_key(&rejections, 2);
        let msg = format_no_seeds_passed("custom family", &stats, &rejections, key.as_ref(), "");
        assert!(msg.contains("rejected_by_kkt=2"));
        assert!(!msg.contains("uniform CertRefused"));
        assert!(!msg.contains("early-exit triggered"));
        assert!(
            msg.contains("phantom_multiplier_with_well_conditioned_H"),
            "per-seed diagnostics must still preserve the actual refusal"
        );
    }

    /// Simulates the outer seed loop's iterative behaviour: failures
    /// arrive one at a time, and after each one we probe whether
    /// `uniform_structural_key` is ready to fire the structural
    /// early-exit (Stage 3). The contract is:
    ///   - after one failure the key is not yet stable (min_count=2)
    ///   - after two identical failures it fires
    ///   - if the third failure deviates the key would no longer be uniform
    /// The seed loop in `rho_optimizer.rs` mirrors this exact pattern,
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

    /// Reproduces the exact shape measured in
    /// `bench/gha_results/python-contracts/py1512_junit.xml` (CI run
    /// 33941725421) for the six `test_sae_manifold_regularizer_noops_issue_240`
    /// tests: 13 generated, 13 screened, 13 exact-validated, 0 reaching the
    /// solver, all 13 rejected in the domain phase with one identical reason.
    #[test]
    fn a_uniform_non_seeding_failure_is_attributed_to_the_stage_that_caused_it() {
        let reason = "outer eval failed: objective returned a non-finite cost";
        let rejections: Vec<SeedRejection> = (0..13)
            .map(|i| {
                SeedRejection::from_message(i, "validation", reason.to_string())
            })
            .collect();
        let stats = StartupStats::from_rejections(13, 13, 13, 0, &rejections);
        assert_eq!(
            stats.rejected_by_domain, 13,
            "fixture must place every rejection in one category"
        );

        let rendered = format_no_seeds_passed("SAE manifold", &stats, &rejections, None, "");
        assert!(
            rendered.contains("NOT a seeding failure"),
            "a run where every seed was exact-validated and none reached the solver must say so \
             instead of leaving the reader with a headline that names seed generation:\n{rendered}"
        );
        assert!(
            rendered.contains("generated, screened and exact-validated"),
            "the attribution must say HOW FAR the seeds got, not merely that seeding is innocent:\n{rendered}"
        );
        assert!(
            rendered.contains(reason),
            "the attribution must carry the shared reason so the reader has the next question \
             without scrolling:\n{rendered}"
        );
    }

    /// Silence is the correct output when the evidence is mixed. A summary that
    /// guesses is worse than one that is absent, and every one of these cases
    /// is a real shape the seed loop produces.
    #[test]
    fn a_mixed_failure_is_not_attributed_at_all() {
        let uniform = "outer eval failed: objective returned a non-finite cost";

        // (a) Two different reasons in the same category.
        let mixed_reasons = vec![
            SeedRejection::from_message(0, "validation", uniform.to_string()),
            SeedRejection::from_message(1, "validation", "a different refusal".to_string()),
        ];
        let stats = StartupStats::from_rejections(2, 2, 2, 0, &mixed_reasons);
        let rendered = format_no_seeds_passed("ctx", &stats, &mixed_reasons, None, "");
        assert!(
            !rendered.contains("NOT a seeding failure"),
            "two distinct reasons is not one cause; the attribution must stay silent:\n{rendered}"
        );

        // (b) A seed DID reach the solver, so seeding is not exonerated.
        let started = vec![SeedRejection::from_message(0, "validation", uniform.to_string())];
        let stats = StartupStats::from_rejections(2, 2, 2, 1, &started);
        let rendered = format_no_seeds_passed("ctx", &stats, &started, None, "");
        assert!(
            !rendered.contains("NOT a seeding failure"),
            "solver_started > 0 means the pipeline did reach the solver:\n{rendered}"
        );

        // (c) No rejections recorded at all: nothing to attribute.
        let stats = StartupStats::from_rejections(3, 3, 3, 0, &[]);
        let rendered = format_no_seeds_passed("ctx", &stats, &[], None, "");
        assert!(
            !rendered.contains("NOT a seeding failure"),
            "an empty rejection list carries no evidence:\n{rendered}"
        );
    }

    /// The counters and the per-seed list must agree before anything is
    /// concluded from them: a truncated rejection list beside a larger count is
    /// exactly the shape that would let a summary generalise from a prefix.
    #[test]
    fn a_rejection_list_shorter_than_its_count_is_not_attributed() {
        let reason = "outer eval failed: objective returned a non-finite cost";
        let all: Vec<SeedRejection> = (0..5)
            .map(|i| SeedRejection::from_message(i, "validation", reason.to_string()))
            .collect();
        let stats = StartupStats::from_rejections(5, 5, 5, 0, &all);
        assert_eq!(stats.total_rejected(), 5);

        // Same stats, but only the first two rejections survived to the render.
        let rendered = format_no_seeds_passed("ctx", &stats, &all[..2], None, "");
        assert!(
            !rendered.contains("NOT a seeding failure"),
            "2 listed reasons cannot establish that all 5 shared one cause:\n{rendered}"
        );
    }

}
