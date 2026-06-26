//! The unified certificate contract (task #16).
//!
//! Across the program a dozen independent analyses each emit a "certificate":
//! the outer-optimum first-order self-audit ([`CriterionCertificate`]), the
//! sensitivity-coreset error budget ([`CoresetCertificate`]), the log-det
//! enclosure ([`LogdetEnclosure`]), the Kantorovich encode atlas
//! ([`EncodeResult`]), the exact-orbit residual-gauge report, the dictionary
//! incoherence / global-optimality report, the structure-search collapse
//! events, and the topology evidence certification. Each grew its own struct,
//! its own verdict enum, and its own scattered payload key.
//!
//! This module gives them ONE shared contract so a fit returns a single
//! inspectable certificate ledger — the program's signature artifact. The
//! contract is the [`Certificate`] trait: every certificate states
//!
//! 1. **the CLAIM** it certifies — a stable machine id plus a human sentence
//!    ([`Certificate::claim`]);
//! 2. **the EVIDENCE** quantities behind the claim ([`Certificate::evidence`]),
//!    as named scalars/flags/text;
//! 3. **a conservative VERDICT** ([`Certificate::verdict`]) drawn from
//!    [`Verdict`], in which *certified-but-wrong is structurally impossible*:
//!    the verdict can only STRENGTHEN as evidence accrues, the weakest state is
//!    the default, and there are explicit [`Verdict::Insufficient`] /
//!    [`Verdict::Unavailable`] states so a missing or below-margin certificate
//!    never silently reads as "certified".
//!
//! Migration rule (task #16): the existing certificate types KEEP their math
//! unchanged; they merely implement [`Certificate`]. Their bespoke methods
//! (`passes`, `certify_margin`, `decide_within_margin`, `is_certified`, …) stay
//! as-is and the trait's [`Certificate::verdict`] is defined in terms of them,
//! so there is exactly one source of truth for each verdict.

use std::collections::BTreeMap;

/// The conservative verdict ladder shared by every certificate.
///
/// The ordering is a soundness lattice, weakest → strongest:
/// `Unavailable < Insufficient < Certified`. A verdict may only move UP this
/// ladder as evidence accrues; it can never claim more than the evidence
/// supports. The weakest state ([`Verdict::Unavailable`]) is the default, so a
/// certificate that was never computed, or whose inputs were degenerate, reads
/// as "no claim" — never as a silent pass. This is what makes
/// "certified-but-wrong" structurally impossible: `Certified` is reachable only
/// when the owning certificate's own (unchanged) decision rule says the
/// evidence strictly clears its required margin.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Verdict {
    /// The certificate could not be evaluated: inputs were missing, degenerate,
    /// or non-finite. No claim is made. This is the default — the absence of a
    /// certificate is this state, not `Certified`.
    Unavailable,
    /// The certificate was evaluated and the evidence is present, but it does
    /// NOT clear the margin required to certify the claim. The consumer must
    /// escalate (refine, gather more evidence, or fall back to the exact path);
    /// it must NOT treat this as a pass.
    Insufficient,
    /// The evidence strictly clears the claim's required margin. The claim holds
    /// — and, by construction of each owning decision rule, the conservative
    /// (worst-case) bound was used, so this verdict cannot be falsely positive.
    Certified,
}

impl Verdict {
    /// Whether the claim is certified. The ONLY `true` case is
    /// [`Verdict::Certified`]; `Insufficient` and `Unavailable` are both `false`
    /// so no caller can read a missing or below-margin certificate as a pass.
    pub fn is_certified(self) -> bool {
        matches!(self, Verdict::Certified)
    }

    /// Stable machine label for payloads.
    pub fn label(self) -> &'static str {
        match self {
            Verdict::Unavailable => "unavailable",
            Verdict::Insufficient => "insufficient",
            Verdict::Certified => "certified",
        }
    }

    /// Combine two verdicts about the SAME claim conservatively: the result is
    /// the WEAKER of the two (the meet on the soundness lattice). Aggregating a
    /// batch of per-item verdicts this way guarantees the summary can never be
    /// stronger than its weakest member — one uncertified row makes the batch
    /// uncertified.
    pub fn meet(self, other: Verdict) -> Verdict {
        self.min(other)
    }
}

/// The claim a certificate makes: a stable machine id and a human sentence.
///
/// `id` is a kebab-case key stable across runs (used as the payload sub-key and
/// for programmatic lookup); `statement` is the one-line human description of
/// exactly what is being certified.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Claim {
    pub id: &'static str,
    pub statement: String,
}

impl Claim {
    pub fn new(id: &'static str, statement: impl Into<String>) -> Self {
        Self {
            id,
            statement: statement.into(),
        }
    }
}

/// A single named evidence quantity behind a claim. Evidence is reported as
/// typed values so the ledger is machine-inspectable, not just a string blob.
#[derive(Debug, Clone, PartialEq)]
pub enum EvidenceValue {
    Scalar(f64),
    Integer(i64),
    Flag(bool),
    Text(String),
    /// A short list of scalars (e.g. per-atom statistics). Kept small; large
    /// arrays belong in the typed diagnostics, not the certificate ledger.
    Vector(Vec<f64>),
}

impl From<f64> for EvidenceValue {
    fn from(v: f64) -> Self {
        EvidenceValue::Scalar(v)
    }
}
impl From<usize> for EvidenceValue {
    fn from(v: usize) -> Self {
        EvidenceValue::Integer(v as i64)
    }
}
impl From<i64> for EvidenceValue {
    fn from(v: i64) -> Self {
        EvidenceValue::Integer(v)
    }
}
impl From<bool> for EvidenceValue {
    fn from(v: bool) -> Self {
        EvidenceValue::Flag(v)
    }
}
impl From<String> for EvidenceValue {
    fn from(v: String) -> Self {
        EvidenceValue::Text(v)
    }
}
impl From<&str> for EvidenceValue {
    fn from(v: &str) -> Self {
        EvidenceValue::Text(v.to_string())
    }
}
impl From<Vec<f64>> for EvidenceValue {
    fn from(v: Vec<f64>) -> Self {
        EvidenceValue::Vector(v)
    }
}

/// The ordered set of evidence quantities behind a claim. Ordering is stable
/// (`BTreeMap`) so payloads and snapshots are deterministic.
pub type Evidence = BTreeMap<&'static str, EvidenceValue>;

/// The shared contract every certificate in the program implements (task #16).
///
/// Implementors do NOT change their math — they expose their existing claim,
/// evidence, and (unchanged) decision rule through this uniform shape. The
/// default [`Certificate::ledger_entry`] folds the three into one inspectable
/// record so the fit can assemble a single certificate ledger.
pub trait Certificate {
    /// What this certificate certifies — stable id + human sentence.
    fn claim(&self) -> Claim;

    /// The named evidence quantities behind the claim.
    fn evidence(&self) -> Evidence;

    /// The conservative verdict. MUST be derived from the certificate's own
    /// (unchanged) decision rule, and MUST return [`Verdict::Unavailable`] /
    /// [`Verdict::Insufficient`] rather than a silent pass when the evidence is
    /// missing or below margin.
    fn verdict(&self) -> Verdict;

    /// Fold claim + evidence + verdict into one ledger record.
    fn ledger_entry(&self) -> LedgerEntry {
        LedgerEntry {
            claim: self.claim(),
            evidence: self.evidence(),
            verdict: self.verdict(),
        }
    }
}

/// One certificate's contribution to the ledger: its claim, evidence, and
/// conservative verdict, frozen at the time the fit recorded it.
#[derive(Debug, Clone, PartialEq)]
pub struct LedgerEntry {
    pub claim: Claim,
    pub evidence: Evidence,
    pub verdict: Verdict,
}

/// The fit's certificate ledger: every certificate the fit produced, keyed by
/// claim id, in stable order. This is the single inspectable artifact that
/// replaces the scattered per-feature payload keys.
///
/// The ledger never fabricates a verdict: a claim that was not evaluated simply
/// is absent (queried as [`Verdict::Unavailable`] via [`Self::verdict_of`]).
#[derive(Debug, Clone, Default, PartialEq)]
pub struct CertificateLedger {
    entries: BTreeMap<&'static str, LedgerEntry>,
}

impl CertificateLedger {
    pub fn new() -> Self {
        Self::default()
    }

    /// Record one certificate. If two certificates share a claim id, they are
    /// combined conservatively: the retained verdict is the WEAKER of the two
    /// (so duplicate evidence can never upgrade a claim past its weakest
    /// witness), and the evidence of the weaker verdict is kept.
    pub fn record<C: Certificate>(&mut self, certificate: &C) {
        self.record_entry(certificate.ledger_entry());
    }

    /// Record a pre-built entry (for certificates whose owning type lives behind
    /// a boundary that only hands back the folded record).
    pub fn record_entry(&mut self, entry: LedgerEntry) {
        match self.entries.get(entry.claim.id) {
            Some(existing) if existing.verdict <= entry.verdict => {
                // Existing is weaker-or-equal: keep the conservative one.
            }
            _ => {
                self.entries.insert(entry.claim.id, entry);
            }
        }
    }

    /// The verdict for a claim id, or [`Verdict::Unavailable`] if the fit never
    /// recorded it — the absence of a certificate is "no claim", never a pass.
    pub fn verdict_of(&self, claim_id: &str) -> Verdict {
        self.entries
            .get(claim_id)
            .map(|e| e.verdict)
            .unwrap_or(Verdict::Unavailable)
    }

    /// All recorded entries in stable (claim-id) order.
    pub fn entries(&self) -> impl Iterator<Item = &LedgerEntry> {
        self.entries.values()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// The conservative roll-up across the whole ledger: the WEAKEST verdict of
    /// any recorded claim (the meet over the soundness lattice). An empty ledger
    /// rolls up to [`Verdict::Unavailable`]. This is the single number that
    /// answers "did everything this fit could certify, certify?" — and it cannot
    /// be stronger than its weakest member.
    pub fn overall(&self) -> Verdict {
        self.entries
            .values()
            .map(|e| e.verdict)
            .fold(Verdict::Certified, Verdict::meet)
            // An empty ledger has nothing to certify → Unavailable, not the
            // vacuous-Certified fold seed.
            .min(if self.entries.is_empty() {
                Verdict::Unavailable
            } else {
                Verdict::Certified
            })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct FakeCert {
        id: &'static str,
        verdict: Verdict,
    }
    impl Certificate for FakeCert {
        fn claim(&self) -> Claim {
            Claim::new(self.id, "fake claim")
        }
        fn evidence(&self) -> Evidence {
            let mut e = Evidence::new();
            e.insert("x", 1.0.into());
            e
        }
        fn verdict(&self) -> Verdict {
            self.verdict
        }
    }

    #[test]
    fn verdict_ladder_orders_weakest_to_strongest() {
        assert!(Verdict::Unavailable < Verdict::Insufficient);
        assert!(Verdict::Insufficient < Verdict::Certified);
        assert!(!Verdict::Insufficient.is_certified());
        assert!(!Verdict::Unavailable.is_certified());
        assert!(Verdict::Certified.is_certified());
    }

    #[test]
    fn meet_is_conservative() {
        assert_eq!(
            Verdict::Certified.meet(Verdict::Insufficient),
            Verdict::Insufficient
        );
        assert_eq!(
            Verdict::Insufficient.meet(Verdict::Unavailable),
            Verdict::Unavailable
        );
        assert_eq!(
            Verdict::Certified.meet(Verdict::Certified),
            Verdict::Certified
        );
    }

    #[test]
    fn absent_claim_reads_as_unavailable_never_pass() {
        let ledger = CertificateLedger::new();
        assert_eq!(ledger.verdict_of("nonexistent"), Verdict::Unavailable);
        assert!(!ledger.verdict_of("nonexistent").is_certified());
        // Empty ledger rolls up to Unavailable, not a vacuous pass.
        assert_eq!(ledger.overall(), Verdict::Unavailable);
    }

    #[test]
    fn overall_is_weakest_member() {
        let mut ledger = CertificateLedger::new();
        ledger.record(&FakeCert {
            id: "a",
            verdict: Verdict::Certified,
        });
        ledger.record(&FakeCert {
            id: "b",
            verdict: Verdict::Insufficient,
        });
        assert_eq!(ledger.overall(), Verdict::Insufficient);
        assert!(!ledger.overall().is_certified());
    }

    #[test]
    fn duplicate_record_keeps_weaker_verdict() {
        let mut ledger = CertificateLedger::new();
        ledger.record(&FakeCert {
            id: "a",
            verdict: Verdict::Certified,
        });
        ledger.record(&FakeCert {
            id: "a",
            verdict: Verdict::Insufficient,
        });
        assert_eq!(ledger.verdict_of("a"), Verdict::Insufficient);
        assert_eq!(ledger.len(), 1);
    }
}
