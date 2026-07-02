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
//! every observed failure carries the same genuinely structural
//! `(diagnosis, carrying_block)` pair, every remaining ρ candidate will
//! fail the same way, so the outer skips them instead of paying a full
//! joint-Newton inner solve per duplicate. Numerical certificate
//! refusals, such as a phantom multiplier with well-conditioned
//! `H_pen`, are deliberately excluded: continuation treats them as
//! recoverable by changing the ρ path, so startup must not infer that
//! sibling seeds are impossible.

use std::fmt::Write;

use gam_problem::diagnostics::KktRefusalDiagnosis;
use crate::inner_status::{InnerFailure, classify_inner_error};

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
                // A pre-fit identifiability failure is structural in
                // the same shape as a KKT cert refusal — bucket it
                // with `rejected_by_kkt` so the seed-screening
                // structural early-exit accounting sees it.
                InnerFailure::IdentifiabilityFailure { .. } => stats.rejected_by_kkt += 1,
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

/// Coarse discriminant of an [`InnerFailure`] variant, used as the first
/// half of the generic cross-seed failure signature. The `uniform_structural_key`
/// path above only fires for genuinely structural `CertRefused` diagnoses; this
/// tag is deliberately broader so the *generic* consecutive-run detector can
/// also catch the `RemlConvergenceError` / non-PD-pivot / KKT-stuck class
/// (#1036) that classifies as `BudgetExhausted`, `TrustRegionFloor`, or
/// `Other` and never reaches a structural diagnosis.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum FailureVariantTag {
    CertRefused,
    BudgetExhausted,
    TrustRegionFloor,
    Likelihood,
    Identifiability,
    Other,
}

fn variant_tag(failure: &InnerFailure) -> FailureVariantTag {
    match failure {
        InnerFailure::CertRefused { .. } => FailureVariantTag::CertRefused,
        InnerFailure::BudgetExhausted { .. } => FailureVariantTag::BudgetExhausted,
        InnerFailure::TrustRegionFloor { .. } => FailureVariantTag::TrustRegionFloor,
        InnerFailure::LikelihoodFailure(_) => FailureVariantTag::Likelihood,
        InnerFailure::IdentifiabilityFailure { .. } => FailureVariantTag::Identifiability,
        InnerFailure::Other(_) => FailureVariantTag::Other,
    }
}

/// True only for repeated generic failures that are safe to treat as a
/// cross-seed structural fingerprint.  Non-finite objective/domain failures are
/// deliberately excluded even when they carry a repeated numeric marker: those
/// are often rho-local trial-point pathologies on spatial/Duchon/sphere bases,
/// and bailing early turns "the first few seeds were numerically bad" into the
/// fatal and misleading "no candidate seeds passed" outcome before the stable
/// heavy-smoothing candidates are ever tried (#1802).
fn eligible_for_generic_structural_bail(failure: &InnerFailure) -> bool {
    match failure {
        InnerFailure::CertRefused { .. }
        | InnerFailure::BudgetExhausted { .. }
        | InnerFailure::TrustRegionFloor { .. }
        | InnerFailure::IdentifiabilityFailure { .. } => true,
        InnerFailure::LikelihoodFailure(_) => false,
        InnerFailure::Other(message) => {
            let lower = message.to_ascii_lowercase();
            !(lower.contains("non-finite")
                || lower.contains("not finite")
                || lower.contains("nan")
                || lower.contains("infinite"))
        }
    }
}

/// Signed order-of-magnitude bucket of the dominant diagnostic numeric:
/// `sign` is the value's sign (`-1`/`0`/`+1`) and `order` is
/// `floor(log10(|value|))`. Kept as two independent fields rather than a
/// single packed int because the magnitude order is itself signed (a tiny
/// pivot `-6e-11` has order `-11`), so folding the value's sign into it would
/// be ambiguous — `-6e-11` and `-6e+11` must not collide. Two seeds match
/// only when BOTH fields agree.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct MagnitudeBucket {
    pub sign: i32,
    pub order: i32,
}

/// Generic cross-seed failure signature: the failure-variant discriminant
/// paired with the signed order-of-magnitude bucket of the dominant
/// pivot/KKT numeric parsed from the message. Two seeds that reject with the
/// same variant AND the same magnitude bucket are the "same failure class"
/// the issue (#1036) calls structural — the per-row Hessian pivot and KKT
/// residual reproduce to the same order of magnitude across seeds when the
/// blocker is the design, not the warm-start. The magnitude is `Option`:
/// a message with no parseable diagnostic numeric carries `None`, and a run
/// of `None`-magnitude failures is NOT eligible for the generic bail (we
/// refuse to call an unquantified failure structural).
pub(crate) type GenericFailureSignature = (FailureVariantTag, Option<MagnitudeBucket>);

/// Markers, in priority order, that precede the dominant diagnostic numeric
/// in a bubbled inner-solver error. The first one present wins: the KKT/cert
/// residual and the per-row Hessian pivot are the two quantities the issue
/// names as the structural fingerprint. Each marker is matched
/// case-insensitively on the lowercased message.
const DOMINANT_NUMERIC_MARKERS: &[&str] = &[
    "residual=",
    "pivot=",
    "pivot ~",
    "pivot~",
    "min_pivot=",
    // The grid-spline factor writes `pivot {j} (value {s})`, where `{j}` is the
    // INDEX and `{s}` is the offending diagonal value — so the value follows
    // `(value `, which must out-rank the bare `pivot ` marker below (that would
    // otherwise grab the integer index). Placed first so the genuine value wins.
    "(value ",
    // The Arrow-Schur row factor's genuinely-non-PD bail formats the pivot
    // space-delimited — `non-PD pivot {sum} at index {i}` (arrow_schur.rs) —
    // the exact `RemlConvergenceError` / non-PD-`H_tt` autopsy class #1036 must
    // catch. The earlier `=`/`~`-delimited pivot markers still win when present;
    // this bare-space form is the real solver's wording and parses `{sum}`.
    "pivot ",
    "kkt=",
    "|∇l-sβ|=",
    "|g|=",
    // P-IRLS inner-loop non-convergence (`estimate.rs`) reports the dominant
    // diagnostic as the final gradient norm; that scalar is the stable
    // cross-seed fingerprint for the GLM inner-stall class.
    "gradient norm was ",
];

/// Parse a leading floating-point number (optionally signed, optionally in
/// scientific notation) from the start of `s`. Returns the value and the
/// number of bytes consumed.
fn parse_leading_f64(s: &str) -> Option<f64> {
    let bytes = s.as_bytes();
    let mut end = 0usize;
    let mut seen_digit = false;
    let mut seen_exp = false;
    let mut seen_dot = false;
    while end < bytes.len() {
        let c = bytes[end] as char;
        match c {
            '0'..='9' => {
                seen_digit = true;
                end += 1;
            }
            '+' | '-' => {
                // Sign is only valid at the very start or right after an
                // exponent marker.
                if end == 0 || matches!(bytes[end - 1] as char, 'e' | 'E') {
                    end += 1;
                } else {
                    break;
                }
            }
            '.' if !seen_dot && !seen_exp => {
                seen_dot = true;
                end += 1;
            }
            'e' | 'E' if seen_digit && !seen_exp => {
                seen_exp = true;
                end += 1;
            }
            _ => break,
        }
    }
    if !seen_digit {
        return None;
    }
    s[..end].parse::<f64>().ok()
}

/// Extract the dominant diagnostic magnitude bucket from a bubbled inner
/// error: the value's sign and `floor(log10(|value|))` for the first
/// dominant-numeric marker present. `None` when no marker yields a finite,
/// non-zero value — such a failure has no quantified fingerprint and is
/// excluded from the generic structural bail.
pub(crate) fn dominant_magnitude_bucket(message: &str) -> Option<MagnitudeBucket> {
    let lower = message.to_ascii_lowercase();
    for marker in DOMINANT_NUMERIC_MARKERS {
        if let Some(pos) = lower.find(marker) {
            let tail = lower[pos + marker.len()..].trim_start();
            if let Some(value) = parse_leading_f64(tail) {
                if value.is_finite() && value != 0.0 {
                    return Some(MagnitudeBucket {
                        sign: value.signum() as i32,
                        order: value.abs().log10().floor() as i32,
                    });
                }
            }
        }
    }
    None
}

pub(crate) fn generic_signature(failure: &InnerFailure) -> GenericFailureSignature {
    (
        variant_tag(failure),
        dominant_magnitude_bucket(failure.message()),
    )
}

/// `Some((signature, run_len))` when the LAST `min_run` rejections all carry
/// an identical generic signature with a *quantified* magnitude bucket —
/// the generic cross-seed structural-failure detector (#1036). Distinct from
/// [`uniform_structural_key`] in three ways:
///   - it covers every failure variant, not only structural `CertRefused`;
///   - it keys on the order-of-magnitude pivot/KKT bucket, not the
///     `(diagnosis, carrying_block)` pair, so it fires on the
///     `RemlConvergenceError` / non-PD-pivot class the structural-diagnosis
///     path never sees;
///   - it requires the run to be the *trailing* `min_run` seeds, so a single
///     deviating signature breaks the run and the cascade keeps going (genuine
///     seed-luck stays a full cascade).
/// A `None`-magnitude signature is never eligible: an unquantified failure is
/// not called structural.
pub(crate) fn consecutive_generic_signature(
    rejections: &[SeedRejection],
    min_run: usize,
) -> Option<(GenericFailureSignature, usize)> {
    if min_run == 0 || rejections.len() < min_run {
        return None;
    }
    let tail = &rejections[rejections.len() - min_run..];
    if tail
        .iter()
        .any(|rej| !eligible_for_generic_structural_bail(&rej.failure))
    {
        return None;
    }
    let sig = generic_signature(&tail[0].failure);
    // An unquantified (None-magnitude) signature is excluded by contract.
    sig.1?;
    for rej in &tail[1..] {
        if generic_signature(&rej.failure) != sig {
            return None;
        }
    }
    Some((sig, min_run))
}

/// Render the generic structural-failure signature for the aggregated bail
/// message: `"<variant>@<sign>1e<order>"`, e.g. `"budget_exhausted@1e3"` or
/// `"other@-1e-11"` (a negative pivot of order `1e-11`). The phrasing names
/// the variant and the signed order of magnitude so two operators reading two
/// failed fits can tell at a glance whether they hit the same blocker.
pub(crate) fn generic_signature_label(sig: &GenericFailureSignature) -> String {
    let (tag, bucket) = sig;
    let variant = match tag {
        FailureVariantTag::CertRefused => "cert_refused",
        FailureVariantTag::BudgetExhausted => "budget_exhausted",
        FailureVariantTag::TrustRegionFloor => "trust_region_floor",
        FailureVariantTag::Likelihood => "likelihood",
        FailureVariantTag::Identifiability => "identifiability",
        FailureVariantTag::Other => "other",
    };
    match bucket {
        Some(b) => {
            let sign = if b.sign < 0 { "-" } else { "" };
            format!("{variant}@{sign}1e{}", b.order)
        }
        None => format!("{variant}@<unquantified>"),
    }
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
         rejected_by_objective={}, rejected_by_budget={}, rejected_other={} (total={})",
        stats.rejected_by_kkt,
        stats.rejected_by_domain,
        stats.rejected_by_objective,
        stats.rejected_by_budget,
        stats.rejected_other,
        stats.total_rejected(),
    )
    .expect("writing to String cannot fail");
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

    /// A `RemlConvergenceError`-class rejection in the shape #1036 autopsies:
    /// a non-PD per-row H_tt pivot and a stuck KKT residual, with no
    /// structural `CertRefused` diagnosis. Classifies as `Other` and so is
    /// invisible to `uniform_structural_key`, but carries a quantified
    /// pivot/KKT fingerprint the generic detector keys on.
    fn reml_nonpd(seed_idx: usize, pivot: &str, kkt: &str) -> SeedRejection {
        SeedRejection::from_message(
            seed_idx,
            "validation",
            format!(
                "RemlConvergenceError: inner Newton stalled; non-PD per-row H_tt \
                 pivot={pivot}; KKT residual=stuck (|∇L-Sβ|={kkt} > 1.0e-03 tol)"
            ),
        )
    }

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

    // ─── #1036 generic cross-seed structural-failure detector ────────────

    #[test]
    fn dominant_magnitude_buckets_signed_order_of_magnitude() {
        // Negative tiny pivot ~ -6e-11 → sign=-1, order=floor(log10(6e-11))=-11.
        assert_eq!(
            dominant_magnitude_bucket("non-PD pivot=-6e-11; rest"),
            Some(MagnitudeBucket {
                sign: -1,
                order: -11
            })
        );
        // KKT residual stuck at 1e3 → sign=+1, order=3.
        assert_eq!(
            dominant_magnitude_bucket("residual=5.0e+03 > 4·tol=4.0e+03"),
            Some(MagnitudeBucket { sign: 1, order: 3 })
        );
        // No parseable diagnostic numeric → None (unquantified).
        assert_eq!(dominant_magnitude_bucket("some opaque failure"), None);
        // residual= present but non-numeric falls through to the next marker.
        assert_eq!(
            dominant_magnitude_bucket("residual=stuck; |∇L-Sβ|=2.5e+05 vs tol"),
            Some(MagnitudeBucket { sign: 1, order: 5 })
        );
        // A negative value of order 1e+11 must NOT collide with -6e-11.
        assert_ne!(
            dominant_magnitude_bucket("pivot=-6e-11"),
            dominant_magnitude_bucket("pivot=-6e+11"),
        );
    }

    #[test]
    fn dominant_magnitude_bucket_parses_real_solver_wordings() {
        // #1036 regression: the ACTUAL Arrow-Schur non-PD bail is space-delimited
        // (`non-PD pivot {sum} at index {i}`), NOT `pivot=`. The detector must
        // parse the real wording or it never fires on the sphere autopsy class.
        assert_eq!(
            dominant_magnitude_bucket(
                "row 3 H_tt is non-PD at base ridge 0e0; non-PD pivot -6e-11 at index 2 \
                 (matrix is not positive definite)"
            ),
            Some(MagnitudeBucket {
                sign: -1,
                order: -11
            })
        );
        // Grid-spline factor: `pivot {j} (value {s})` — the VALUE follows
        // `(value `, which must out-rank the bare `pivot ` (an integer index).
        assert_eq!(
            dominant_magnitude_bucket(
                "grid spline 2d: penalized system not positive definite at pivot 4 (value -2.5e-09)"
            ),
            Some(MagnitudeBucket {
                sign: -1,
                order: -9
            })
        );
        // P-IRLS inner-loop stall: the final gradient norm is the fingerprint.
        assert_eq!(
            dominant_magnitude_bucket(
                "The P-IRLS inner loop did not converge within 200 iterations. \
                 Last gradient norm was 3.400000e+02."
            ),
            Some(MagnitudeBucket { sign: 1, order: 2 })
        );
    }

    /// #1036 end-to-end: three seeds whose REAL Arrow-Schur non-PD message (the
    /// space-delimited `non-PD pivot {sum}` wording the solver actually emits)
    /// repeats at the same order-of-magnitude pivot must trigger the generic
    /// structural bail — the exact sphere-autopsy class that previously burned
    /// all 12 seeds because the detector keyed only on `pivot=`.
    #[test]
    fn generic_detector_fires_on_real_arrow_nonpd_wording() {
        let real = |seed: usize, pivot: &str| {
            SeedRejection::from_message(
                seed,
                "validation",
                format!(
                    "RemlConvergenceError: row 3 H_tt is non-PD at base ridge 0e0; \
                     non-PD pivot {pivot} at index 2 (matrix is not positive definite)"
                ),
            )
        };
        // Three consecutive seeds, same signed pivot order (≈ -6e-11), with the
        // KKT residual deliberately NOT in the message — the pivot is the stable
        // cross-seed invariant the autopsy identified.
        let rejections = vec![
            real(0, "-6.1e-11"),
            real(1, "-5.8e-11"),
            real(2, "-6.4e-11"),
        ];
        let (sig, run) = consecutive_generic_signature(&rejections, 3)
            .expect("three identical real-wording non-PD pivots must trigger the bail");
        assert_eq!(run, 3);
        assert_eq!(sig.0, FailureVariantTag::Other);
        assert_eq!(
            sig.1,
            Some(MagnitudeBucket {
                sign: -1,
                order: -11
            })
        );
        // The aggregated label is the human-readable bail signature.
        assert_eq!(generic_signature_label(&sig), "other@-1e-11");
    }

    #[test]
    fn generic_signature_pairs_variant_with_magnitude() {
        let rej = reml_nonpd(0, "-6e-11", "1.0e+03");
        let sig = generic_signature(&rej.failure);
        assert_eq!(sig.0, FailureVariantTag::Other);
        // pivot= marker wins over |∇l-sβ|=: -6e-11 → sign=-1, order=-11.
        assert_eq!(
            sig.1,
            Some(MagnitudeBucket {
                sign: -1,
                order: -11
            })
        );
        assert_eq!(generic_signature_label(&sig), "other@-1e-11");
    }

    /// The #1036 structural class: three consecutive seeds reject with the
    /// SAME `RemlConvergenceError` non-PD-pivot signature. The generic
    /// detector must fire at run length 3 even though none of these are a
    /// structural `CertRefused` (so `uniform_structural_key` stays silent).
    #[test]
    fn generic_detector_fires_on_repeated_reml_nonpd_pivot() {
        let rejections = vec![
            reml_nonpd(0, "-6e-11", "1.0e+03"),
            reml_nonpd(1, "-6e-11", "5.0e+03"),
            reml_nonpd(2, "-6e-11", "8.0e+03"),
        ];
        // uniform_structural_key never sees this class.
        assert!(
            uniform_structural_key(&rejections, 2).is_none(),
            "non-cert-refused RemlConvergenceError must not be a structural-diagnosis key"
        );
        let (sig, run) = consecutive_generic_signature(&rejections, 3)
            .expect("three identical pivot signatures must trigger the generic bail");
        assert_eq!(run, 3);
        assert_eq!(
            sig,
            (
                FailureVariantTag::Other,
                Some(MagnitudeBucket {
                    sign: -1,
                    order: -11
                })
            )
        );
    }

    /// #1802: a repeated non-finite objective at the first few trial rhos is a
    /// numeric startup miss, not proof that the remaining spatial/Duchon/sphere
    /// seed lattice is infeasible.  The live per-seed breakdown must keep
    /// running so an over-smoothed or manifold-consistent seed can pass.
    #[test]
    fn generic_detector_does_not_bail_on_repeated_nonfinite_objectives() {
        let nonfinite = |seed: usize| {
            SeedRejection::from_message(
                seed,
                "validation",
                "outer eval failed: non-finite objective at trial rho; \
                 non-PD pivot -6.0e-11 at index 2"
                    .into(),
            )
        };
        let rejections = vec![nonfinite(0), nonfinite(1), nonfinite(2)];
        assert!(
            consecutive_generic_signature(&rejections, 3).is_none(),
            "non-finite objective rejections are rho-local startup failures; \
             the seed cascade must keep evaluating later candidates"
        );
    }

    /// Control: genuine seed-luck. The trailing run of identical signatures is
    /// broken by a deviating final seed, so the generic detector must NOT fire
    /// and the cascade keeps running every seed.
    #[test]
    fn generic_detector_silent_when_signatures_differ() {
        let rejections = vec![
            reml_nonpd(0, "-6e-11", "1.0e+03"),
            reml_nonpd(1, "-6e-11", "5.0e+03"),
            // Different pivot order of magnitude → different signature.
            reml_nonpd(2, "-3e-04", "8.0e+03"),
        ];
        assert!(
            consecutive_generic_signature(&rejections, 3).is_none(),
            "a deviating trailing signature is seed-luck, not structural — full cascade must run"
        );
    }

    /// The detector keys on the TRAILING run: an early-cascade deviation that
    /// is later followed by `min_run` identical signatures still fires (the
    /// blocker surfaced once the cascade settled into the structural basin).
    #[test]
    fn generic_detector_keys_on_trailing_run() {
        let rejections = vec![
            // A one-off domain miss at an exploration seed.
            SeedRejection::from_message(
                0,
                "validation",
                "likelihood evaluation failed: NaN".into(),
            ),
            reml_nonpd(1, "-6e-11", "1.0e+03"),
            reml_nonpd(2, "-6e-11", "5.0e+03"),
            reml_nonpd(3, "-6e-11", "8.0e+03"),
        ];
        let (sig, run) = consecutive_generic_signature(&rejections, 3)
            .expect("trailing run of three identical signatures must fire");
        assert_eq!(run, 3);
        assert_eq!(sig.0, FailureVariantTag::Other);
        assert_eq!(
            sig.1,
            Some(MagnitudeBucket {
                sign: -1,
                order: -11
            })
        );
    }

    /// An unquantified failure run (no parseable pivot/KKT numeric) is never
    /// called structural — we refuse to bail on a fingerprint we cannot
    /// quantify.
    #[test]
    fn generic_detector_excludes_unquantified_runs() {
        let rejections = vec![
            SeedRejection::from_message(0, "validation", "opaque legacy failure".into()),
            SeedRejection::from_message(1, "validation", "opaque legacy failure".into()),
            SeedRejection::from_message(2, "validation", "opaque legacy failure".into()),
        ];
        assert!(
            consecutive_generic_signature(&rejections, 3).is_none(),
            "an unquantified (None-magnitude) run must not trigger the generic bail"
        );
    }

    /// Below `min_run` the detector stays silent: two structural rejections
    /// are not yet enough to declare the candidate dead under the generic
    /// rule (default n_struct = 3).
    #[test]
    fn generic_detector_needs_min_run_observations() {
        let rejections = vec![
            reml_nonpd(0, "-6e-11", "1.0e+03"),
            reml_nonpd(1, "-6e-11", "5.0e+03"),
        ];
        assert!(consecutive_generic_signature(&rejections, 3).is_none());
    }

    #[test]
    fn generic_signature_label_renders_signed_buckets() {
        assert_eq!(
            generic_signature_label(&(
                FailureVariantTag::BudgetExhausted,
                Some(MagnitudeBucket { sign: 1, order: 3 })
            )),
            "budget_exhausted@1e3"
        );
        assert_eq!(
            generic_signature_label(&(
                FailureVariantTag::CertRefused,
                Some(MagnitudeBucket {
                    sign: -1,
                    order: -11
                })
            )),
            "cert_refused@-1e-11"
        );
        assert_eq!(
            generic_signature_label(&(FailureVariantTag::Other, None)),
            "other@<unquantified>"
        );
    }
}
