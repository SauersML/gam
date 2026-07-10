//! The one unified SAE migration ledger (sae-unification Increment 3).
//!
//! Before this module the fit path carried TWO parallel move-accounting
//! currencies:
//!
//!   * the tiered driver's `MigrationLedger` (`tiered/fit.rs`) — promotions /
//!     demotions / deaths priced in a `dl_bits` description-length charge, plus
//!     the `pc_reseed_events == 0` invariant; and
//!   * the structure-search stream (`structure_harvest.rs` →
//!     `gam_solve::structure_search::SearchLedger`) — births / deaths / fusions /
//!     fissions / glues adjudicated by an e-process and priced in a banked
//!     `log_e` evidence value.
//!
//! Both are the SAME accounting: an atom is born, dies, or a proposed move is
//! refused, and the move pays evidence. [`SaeMigrationLedger`] is that one
//! currency. A move is a [`SaeMove`] — `Birth` (residual → linear → curved),
//! `Death` (the reverse fall back to the residual-factor pool), or `Refuse` (a
//! proposed move the evidence did not buy) — and every move carries the single
//! [`MoveEvidence`] currency: a REML/LAML criterion delta, the rank/complexity
//! charge it spends, and the net description-length change in **bits** (`dl_bits`)
//! that unifies the tiered `curved_charge` and the e-process `log_e` (a log-e
//! value in nats is a description-length saving; [`bits_from_nats`] converts it).
//!
//! The ledger carries the architecture's global invariant as a counter:
//! `pc_reseed_events` MUST be `0` — every birth seeds from the residual-factor
//! pool, never from a principal component. A stray PC seed is not silently
//! dropped; it is recorded as [`BirthSeed::PrincipalComponent`] and trips
//! [`SaeMigrationLedger::pc_reseed_events`] so the acceptance bar
//! (`assert_no_pc_reseed`) fails loudly.

use std::collections::HashMap;

use gam_solve::structure_search::{MoveVerdict, SearchLedger, StructureMove};

/// Natural-log base, the nats → bits conversion constant (`ln 2`).
const LN_2: f64 = std::f64::consts::LN_2;

/// Convert an evidence quantity measured in **nats** (a log-e value, a REML
/// criterion delta) into the ledger's **bits** description-length currency.
#[inline]
#[must_use]
pub fn bits_from_nats(nats: f64) -> f64 {
    nats / LN_2
}

/// The stage an atom occupies on the residual → linear → curved ladder. A move's
/// stage is the ladder rung it lands on (`Birth`) or leaves (`Death`/`Refuse`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MoveStage {
    /// The residual-factor pool: directions the current dictionary does not
    /// reconstruct. Births originate here; deaths fall back here.
    Residual,
    /// A linear (Euclidean `d = 1`) atom — the collapsed-linear bulk.
    Linear,
    /// A curved chart atom.
    Curved,
}

impl MoveStage {
    /// Stable integer legend for FFI marshalling (`0` residual, `1` linear,
    /// `2` curved).
    #[must_use]
    pub fn code(self) -> u64 {
        match self {
            MoveStage::Residual => 0,
            MoveStage::Linear => 1,
            MoveStage::Curved => 2,
        }
    }
}

/// Where a [`SaeMove::Birth`] seeded from. The residual-factor pool is the ONLY
/// admissible source; every other variant is an accounting record of a seed that
/// must NOT occur, kept so it is loud rather than silent.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BirthSeed {
    /// The residual-factor subspace (worst-reconstructed residual directions /
    /// rows). The architecture's only sanctioned birth seed.
    ResidualFactor,
    /// Promoted from an existing linear atom (linear → curved co-fit promotion).
    LinearAtom,
    /// Promoted / refined from an existing curved chart.
    CurvedChart,
    /// A principal-component reseed — FORBIDDEN. Present only so a stray PC seed
    /// is recorded and trips [`SaeMigrationLedger::pc_reseed_events`]; the tiered
    /// and structure-search paths never emit it.
    PrincipalComponent,
}

impl BirthSeed {
    /// `true` for the one forbidden seed (a principal-component reseed).
    #[must_use]
    pub fn is_pc_reseed(self) -> bool {
        matches!(self, BirthSeed::PrincipalComponent)
    }

    /// Stable integer legend for FFI marshalling.
    #[must_use]
    pub fn code(self) -> u64 {
        match self {
            BirthSeed::ResidualFactor => 0,
            BirthSeed::LinearAtom => 1,
            BirthSeed::CurvedChart => 2,
            BirthSeed::PrincipalComponent => 3,
        }
    }
}

/// The single evidence currency every move pays. `dl_bits` is the net
/// description-length change in **bits** and is the unified quantity across the
/// tiered `curved_charge` and the e-process `log_e`; `reml_delta` and
/// `rank_charge` carry the two half-ledgers (fit gain vs complexity spent) when a
/// path exposes them, and are `NaN` / `0.0` when it does not.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MoveEvidence {
    /// Change in the REML/LAML evidence criterion attributable to the move
    /// (nats; positive ⇒ the move improved the evidence). `NaN` when the move is
    /// a structural tally not scored by a criterion delta.
    pub reml_delta: f64,
    /// The rank / effective-d.o.f. charge the move spends — the complexity side
    /// of the description-length ledger (`0.0` for deaths and refusals, which
    /// free or spend no rank).
    pub rank_charge: f64,
    /// Net description-length change in **bits**: the tiered co-fit's
    /// `curved_charge` for a curved promotion, or the banked e-process evidence
    /// `bits_from_nats(log_e)` for a structure-search move.
    pub dl_bits: f64,
}

impl MoveEvidence {
    /// Evidence carrying only a `dl_bits` charge (the tiered co-fit currency);
    /// the REML delta is unscored and no rank is charged at this granularity.
    #[must_use]
    pub fn from_dl_bits(dl_bits: f64) -> Self {
        Self {
            reml_delta: f64::NAN,
            rank_charge: 0.0,
            dl_bits,
        }
    }

    /// Evidence for a structure-search move whose e-process banked `log_e` nats;
    /// the description-length charge is that evidence in bits.
    #[must_use]
    pub fn from_log_e(log_e: f64) -> Self {
        Self {
            reml_delta: f64::NAN,
            rank_charge: 0.0,
            dl_bits: bits_from_nats(log_e),
        }
    }

    /// The zero-charge evidence a structural tally carries (a dead-routing death,
    /// a budget-deferred refusal): no criterion delta, no rank, no bits.
    #[must_use]
    pub fn none() -> Self {
        Self {
            reml_delta: f64::NAN,
            rank_charge: 0.0,
            dl_bits: 0.0,
        }
    }
}

/// A move in the unified ledger: an atom born onto a ladder rung, an atom that
/// died back toward the residual pool, or a proposed move the evidence refused.
#[derive(Clone, Debug, PartialEq)]
pub enum SaeMove {
    /// An atom was born onto `stage` from `seed`. On the sanctioned path `seed`
    /// is [`BirthSeed::ResidualFactor`] (or a linear/curved promotion); a PC
    /// reseed here trips the invariant.
    Birth { stage: MoveStage, seed: BirthSeed },
    /// An atom on `stage` died and fell back toward the residual-factor pool,
    /// for `reason`.
    Death {
        stage: MoveStage,
        reason: MoveReason,
    },
    /// A proposed move onto `stage` was refused (the evidence did not buy it),
    /// for `reason`. The prior structure is kept.
    Refuse {
        stage: MoveStage,
        reason: MoveReason,
    },
}

impl SaeMove {
    /// Stable integer legend for FFI marshalling (`0` birth, `1` death,
    /// `2` refuse).
    #[must_use]
    pub fn kind_code(&self) -> u64 {
        match self {
            SaeMove::Birth { .. } => 0,
            SaeMove::Death { .. } => 1,
            SaeMove::Refuse { .. } => 2,
        }
    }

    /// The ladder rung this move lands on / leaves.
    #[must_use]
    pub fn stage(&self) -> MoveStage {
        match self {
            SaeMove::Birth { stage, .. }
            | SaeMove::Death { stage, .. }
            | SaeMove::Refuse { stage, .. } => *stage,
        }
    }
}

/// Why an atom died or a proposed move was refused. A `Custom` label carries the
/// path-specific reason (a structure-search verdict, a revival trigger) without a
/// cross-crate enum change.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum MoveReason {
    /// A linear block / atom ended dead — no row selected it — and fell back to
    /// the residual-factor pool.
    DeadRouting,
    /// A curved candidate's evidence did not beat the linear/flat alternative;
    /// the simpler atom is kept (the co-fit's `Θ→0` verdict).
    EvidenceInsufficient,
    /// A death proposal on an already-certified atom — refused (Ville crossings
    /// are permanent).
    CertifiedVeto,
    /// The move-budget was exhausted before this proposal was reached.
    BudgetDeferred,
    /// A proposal duplicated / went stale against an earlier move this round.
    StaleOrDuplicate,
    /// A path-specific reason carried verbatim.
    Custom(String),
}

/// One recorded move, at the granularity the emitting path exposes (per co-fit
/// round; per structure-search round; one structural entry for a death tally).
#[derive(Clone, Debug, PartialEq)]
pub struct MigrationMove {
    /// What the move was (birth / death / refuse) and its stage + seed/reason.
    pub kind: SaeMove,
    /// The round the move was adjudicated in; `None` for a structural tally not
    /// tied to a co-fit / search round.
    pub round: Option<usize>,
    /// Number of atoms / charts / blocks affected.
    pub count: usize,
    /// The evidence currency the move paid.
    pub evidence: MoveEvidence,
    /// Joint objective `J` after the round (`NaN` for a structural tally).
    pub objective: f64,
    /// #2233 closed-form MDL birth pre-screen: the predicted net
    /// description-length change (bits) the pre-screen computed for this move at
    /// PROPOSAL time, before any refit. `Some` only for a residual-factor
    /// [`SaeMove::Birth`] the pre-screen scored (the prediction the post-refit
    /// `evidence.dl_bits` realizes — a logged predicted-vs-realized calibration
    /// pair); `None` for every move the pre-screen does not price (deaths,
    /// refusals, fusions/fissions/glues, curl births, structural tallies).
    pub predicted_dl_bits: Option<f64>,
}

/// The unified migration ledger: every birth / death / refusal, in order, plus
/// the running tallies and the `pc_reseed_events == 0` global invariant. Replaces
/// the tiered `MigrationLedger` and subsumes the structure-search move stream and
/// the sparse-dict dead-atom revival into one accounting currency.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct SaeMigrationLedger {
    /// The adjudicated moves in order.
    pub moves: Vec<MigrationMove>,
    /// Principal-component reseed events. **Must be `0`** — births only draw from
    /// the residual-factor pool. Incremented whenever a birth records
    /// [`BirthSeed::PrincipalComponent`] so callers can assert the "no PC reseed
    /// in the log" acceptance bar.
    pub pc_reseed_events: usize,
    /// Total births (residual → linear → curved).
    pub n_births: usize,
    /// Total deaths (fell back toward the residual-factor pool).
    pub n_deaths: usize,
    /// Total refusals (proposed move the evidence did not buy).
    pub n_refusals: usize,
}

impl SaeMigrationLedger {
    /// An empty ledger.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Record one move, updating the tallies and tripping `pc_reseed_events` if it
    /// is a forbidden principal-component birth.
    pub fn record(&mut self, mv: MigrationMove) {
        match &mv.kind {
            SaeMove::Birth { seed, .. } => {
                self.n_births += mv.count;
                if seed.is_pc_reseed() {
                    self.pc_reseed_events += mv.count;
                }
            }
            SaeMove::Death { .. } => self.n_deaths += mv.count,
            SaeMove::Refuse { .. } => self.n_refusals += mv.count,
        }
        self.moves.push(mv);
    }

    /// Record a birth onto `stage` from `seed`.
    pub fn birth(
        &mut self,
        stage: MoveStage,
        seed: BirthSeed,
        count: usize,
        round: Option<usize>,
        evidence: MoveEvidence,
        objective: f64,
    ) {
        self.record(MigrationMove {
            kind: SaeMove::Birth { stage, seed },
            round,
            count,
            evidence,
            objective,
            predicted_dl_bits: None,
        });
    }

    /// Record a death on `stage` for `reason`.
    pub fn death(
        &mut self,
        stage: MoveStage,
        reason: MoveReason,
        count: usize,
        round: Option<usize>,
        evidence: MoveEvidence,
        objective: f64,
    ) {
        self.record(MigrationMove {
            kind: SaeMove::Death { stage, reason },
            round,
            count,
            evidence,
            objective,
            predicted_dl_bits: None,
        });
    }

    /// Record a refused move onto `stage` for `reason`.
    pub fn refuse(
        &mut self,
        stage: MoveStage,
        reason: MoveReason,
        count: usize,
        round: Option<usize>,
        evidence: MoveEvidence,
        objective: f64,
    ) {
        self.record(MigrationMove {
            kind: SaeMove::Refuse { stage, reason },
            round,
            count,
            evidence,
            objective,
            predicted_dl_bits: None,
        });
    }

    /// `true` iff the log records zero principal-component reseeds — the global
    /// invariant every sanctioned fit path must hold.
    #[must_use]
    pub fn is_pc_reseed_free(&self) -> bool {
        self.pc_reseed_events == 0
    }

    /// Assert the `pc_reseed_events == 0` invariant, returning a diagnostic
    /// `Err` (never a panic) so callers can surface the breach honestly.
    pub fn assert_no_pc_reseed(&self) -> Result<(), String> {
        if self.is_pc_reseed_free() {
            Ok(())
        } else {
            Err(format!(
                "SaeMigrationLedger invariant breached: {} principal-component \
                 reseed event(s) recorded (births must draw from the \
                 residual-factor pool only)",
                self.pc_reseed_events
            ))
        }
    }

    /// Fold one structure-search round's [`SearchLedger`] into the unified
    /// currency, mapping each adjudicated move + verdict onto a birth / death /
    /// refusal priced by the banked e-process evidence (`bits_from_nats(log_e)`).
    /// Keeps the e-process gating untouched — this is the read-out of its
    /// verdicts into the one move currency, not a second gate.
    ///
    /// `birth_predictions` maps a birth candidate index (the index a
    /// [`StructureMove::Birth`] carries) to the #2233 closed-form pre-screen's
    /// predicted ΔMDL (bits) for that residual-factor birth. Each proposed birth's
    /// prediction is stamped onto its folded record's `predicted_dl_bits`, so the
    /// post-refit verdict (`evidence.dl_bits`) and the pre-refit prediction sit on
    /// the SAME record — the predicted-vs-realized calibration pair. A birth not in
    /// the map (a curl birth, or any round scored before the pre-screen existed) is
    /// left `None`; pass an empty map when no predictions are available.
    pub fn record_search_round(
        &mut self,
        round: usize,
        ledger: &SearchLedger,
        birth_predictions: &HashMap<usize, f64>,
    ) {
        for record in &ledger.moves {
            let stage = structure_move_stage(&record.mv);
            // The pre-screen prices residual-factor births only; every other move
            // (and every unscored birth) carries no prediction.
            let predicted = match &record.mv {
                StructureMove::Birth { candidate } => birth_predictions.get(candidate).copied(),
                _ => None,
            };
            match &record.verdict {
                // An accepted birth / fission / fusion / glue is a birth from the
                // residual-factor pool (structure search never PC-reseeds); an
                // accepted death is a demotion recorded below.
                MoveVerdict::Accepted { log_e } => match &record.mv {
                    StructureMove::Death { .. } => self.death(
                        stage,
                        MoveReason::DeadRouting,
                        1,
                        Some(round),
                        MoveEvidence::from_log_e(*log_e),
                        f64::NAN,
                    ),
                    _ => self.birth(
                        stage,
                        BirthSeed::ResidualFactor,
                        1,
                        Some(round),
                        MoveEvidence::from_log_e(*log_e),
                        f64::NAN,
                    ),
                },
                // A never-certified atom demoted to ~0 routing: a death.
                MoveVerdict::Demoted { log_e } => self.death(
                    stage,
                    MoveReason::DeadRouting,
                    1,
                    Some(round),
                    MoveEvidence::from_log_e(*log_e),
                    f64::NAN,
                ),
                // Gate did not certify: the move is refused, structure kept.
                MoveVerdict::Contested { log_e } => self.refuse(
                    stage,
                    MoveReason::EvidenceInsufficient,
                    1,
                    Some(round),
                    MoveEvidence::from_log_e(*log_e),
                    f64::NAN,
                ),
                // Death refused on a certified atom (permanent Ville crossing).
                MoveVerdict::Vetoed { log_e } => self.refuse(
                    stage,
                    MoveReason::CertifiedVeto,
                    1,
                    Some(round),
                    MoveEvidence::from_log_e(*log_e),
                    f64::NAN,
                ),
                MoveVerdict::Deduplicated | MoveVerdict::Stale => self.refuse(
                    stage,
                    MoveReason::StaleOrDuplicate,
                    1,
                    Some(round),
                    MoveEvidence::none(),
                    f64::NAN,
                ),
                MoveVerdict::Deferred => self.refuse(
                    stage,
                    MoveReason::BudgetDeferred,
                    1,
                    Some(round),
                    MoveEvidence::none(),
                    f64::NAN,
                ),
            }
            // Every verdict arm records exactly one move; stamp the pre-screen
            // prediction onto it when this move is a scored residual-factor birth.
            if predicted.is_some() {
                if let Some(last) = self.moves.last_mut() {
                    last.predicted_dl_bits = predicted;
                }
            }
        }
    }

    /// Record a sparse-dict dead-atom revival: a dead linear atom reseeded onto a
    /// worst-reconstructed residual row (never a principal component). The
    /// revival must pay evidence, not just RSS — the `dl_bits` reconstruction
    /// charge it banks. This is the seam the streaming linear kernel wires onto
    /// (its call site lives in `sparse_dict/stream.rs`; re-pointed in a later
    /// increment).
    pub fn record_revival(
        &mut self,
        count: usize,
        round: Option<usize>,
        dl_bits: f64,
        objective: f64,
    ) {
        // A revival is a death (the dead atom) immediately re-born from the
        // residual pool: record the birth in the residual-seed currency.
        self.birth(
            MoveStage::Linear,
            BirthSeed::ResidualFactor,
            count,
            round,
            MoveEvidence::from_dl_bits(dl_bits),
            objective,
        );
    }
}

/// The ladder rung a structure-search move lands on: a `Birth` candidate is a new
/// atom raced off the residual factor (curved by intent — the topology race
/// decides its manifold); a `Death` leaves the linear/curved bulk; fusions,
/// fissions, and glues restructure curved charts.
fn structure_move_stage(mv: &StructureMove) -> MoveStage {
    match mv {
        StructureMove::Birth { .. }
        | StructureMove::Fusion { .. }
        | StructureMove::Fission { .. }
        | StructureMove::Glue { .. } => MoveStage::Curved,
        StructureMove::Death { .. } => MoveStage::Curved,
    }
}

#[cfg(test)]
mod ledger_tests {
    use super::*;

    #[test]
    fn nats_to_bits_is_log2_scaling() {
        // ln 2 nats == exactly 1 bit.
        assert!((bits_from_nats(LN_2) - 1.0).abs() < 1e-12);
        assert!((bits_from_nats(2.0 * LN_2) - 2.0).abs() < 1e-12);
    }

    #[test]
    fn record_updates_tallies_and_holds_pc_invariant() {
        let mut ledger = SaeMigrationLedger::new();
        ledger.birth(
            MoveStage::Curved,
            BirthSeed::ResidualFactor,
            2,
            Some(0),
            MoveEvidence::from_dl_bits(4.0),
            -1.0,
        );
        ledger.death(
            MoveStage::Linear,
            MoveReason::DeadRouting,
            1,
            None,
            MoveEvidence::none(),
            f64::NAN,
        );
        ledger.refuse(
            MoveStage::Curved,
            MoveReason::EvidenceInsufficient,
            1,
            Some(1),
            MoveEvidence::from_dl_bits(0.0),
            -0.5,
        );
        assert_eq!(ledger.n_births, 2);
        assert_eq!(ledger.n_deaths, 1);
        assert_eq!(ledger.n_refusals, 1);
        assert_eq!(ledger.moves.len(), 3);
        assert!(ledger.is_pc_reseed_free());
        assert!(ledger.assert_no_pc_reseed().is_ok());
    }

    #[test]
    fn principal_component_birth_trips_the_invariant() {
        let mut ledger = SaeMigrationLedger::new();
        ledger.birth(
            MoveStage::Curved,
            BirthSeed::PrincipalComponent,
            1,
            Some(0),
            MoveEvidence::none(),
            f64::NAN,
        );
        assert_eq!(ledger.pc_reseed_events, 1);
        assert!(!ledger.is_pc_reseed_free());
        assert!(ledger.assert_no_pc_reseed().is_err());
    }
}
