//! The one unified SAE migration ledger (sae-unification Increment 3).
//!
//! Before this module the fit path carried TWO parallel move-accounting
//! currencies:
//!
//!   * the tiered driver's `MigrationLedger` (`tiered/fit.rs`) — promotions /
//!     demotions / deaths priced in a `dl_bits` description-length charge; and
//!   * the structure-search stream (`structure_harvest.rs` →
//!     `gam_solve::structure_search::SearchLedger`) — births / deaths / fusions /
//!     fissions / glues adjudicated by an e-process and priced in a banked
//!     `log_e` evidence value.
//!
//! Both are the SAME accounting: an atom is born, dies, or a proposed move is
//! refused, and the move pays evidence. [`SaeMigrationLedger`] is that one
//! currency. A move is a [`SaeMove`] — `Birth` (residual → linear → curved),
//! `Death` (the reverse fall back to the residual-factor pool), `Refuse` (a
//! proposed move the evidence did not buy), `Admit` (a proposed move the
//! evidence bought that the fit reports without installing), or `Restructure` (an
//! installed move that adds and removes no atom) — and every move carries the single
//! [`MoveEvidence`] currency: a REML/LAML criterion delta, the rank/complexity
//! charge it spends, and the net description-length change in **bits** (`dl_bits`)
//! that unifies the tiered `curved_charge` and the e-process `log_e` (a log-e
//! value in nats is a description-length saving; [`bits_from_nats`] converts it).
//!
use std::collections::HashMap;

use gam_solve::structure_search::{ChartGlueOutcome, MoveVerdict, SearchLedger, StructureMove};

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
/// admissible source of new structure; the atom-derived variants record a birth
/// that copies or promotes an atom already in the dictionary.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BirthSeed {
    /// The residual-factor subspace (worst-reconstructed residual directions /
    /// rows). The architecture's only sanctioned seed of new structure.
    ResidualFactor,
    /// Promoted from an existing linear atom (linear → curved co-fit promotion).
    LinearAtom,
    /// Promoted / refined from an existing curved chart, or a fission child
    /// cloned from the chart it splits.
    CurvedChart,
}

impl BirthSeed {
    /// Stable integer legend for FFI marshalling.
    #[must_use]
    pub fn code(self) -> u64 {
        match self {
            BirthSeed::ResidualFactor => 0,
            BirthSeed::LinearAtom => 1,
            BirthSeed::CurvedChart => 2,
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
    pub(crate) fn from_dl_bits(dl_bits: f64) -> Self {
        Self {
            reml_delta: f64::NAN,
            rank_charge: 0.0,
            dl_bits,
        }
    }

    /// Evidence for a structure-search move whose e-process banked `log_e` nats;
    /// the description-length charge is that evidence in bits.
    #[must_use]
    pub(crate) fn from_log_e(log_e: f64) -> Self {
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
    /// is [`BirthSeed::ResidualFactor`] (or a linear/curved promotion).
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
    /// A proposed move onto `stage` from `seed` that the evidence bought but this fit
    /// reports without installing. The code-space census adjudicates each linear
    /// community's curved replacement in bits and mutates nothing, so its accepted
    /// proposals are admitted moves: no atom joined the model, and nothing was refused.
    Admit { stage: MoveStage, seed: BirthSeed },
    /// An installed move on `stage` that adds and removes no atom: an atlas
    /// registration keeps both charts and records the transition between them.
    Restructure { stage: MoveStage },
}

impl SaeMove {
    /// The ladder rung this move lands on / leaves.
    #[must_use]
    pub fn stage(&self) -> MoveStage {
        match self {
            SaeMove::Birth { stage, .. }
            | SaeMove::Death { stage, .. }
            | SaeMove::Refuse { stage, .. }
            | SaeMove::Admit { stage, .. }
            | SaeMove::Restructure { stage } => *stage,
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
    /// An accepted fusion or chart glue folded this atom's routing mass into the
    /// surviving atom of the pair.
    Fused,
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
    /// #2233 birth proposal priority: the heuristic net description-length change
    /// (bits) computed for this move at PROPOSAL time, before any refit. It orders
    /// proposals and certifies nothing (#2933 F22). `Some` only for a
    /// residual-factor [`SaeMove::Birth`] with a finite priority (paired with the
    /// post-refit `evidence.dl_bits` — a logged predicted-vs-realized calibration
    /// pair); `None` for every move the priority does not price (deaths, refusals,
    /// fusions/fissions/glues, curl births, structural tallies, inconclusive
    /// priorities).
    pub predicted_dl_bits: Option<f64>,
}

/// The unified migration ledger: every birth / death / refusal, in order, plus
/// the running tallies. Replaces
/// the tiered `MigrationLedger` and subsumes the structure-search move stream and
/// the sparse-dict dead-atom revival into one accounting currency.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct SaeMigrationLedger {
    /// The adjudicated moves in order.
    pub moves: Vec<MigrationMove>,
    /// Total births (residual → linear → curved).
    pub n_births: usize,
    /// Total deaths (fell back toward the residual-factor pool).
    pub n_deaths: usize,
    /// Total refusals (proposed move the evidence did not buy).
    pub n_refusals: usize,
    /// Total admitted moves (the evidence bought them; the fit did not install them).
    pub n_admitted: usize,
    /// Total installed moves that added and removed no atom (atlas registrations).
    pub n_restructures: usize,
}

impl SaeMigrationLedger {
    /// An empty ledger.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Record one move, updating the tallies.
    pub fn record(&mut self, mv: MigrationMove) {
        match &mv.kind {
            SaeMove::Birth { .. } => self.n_births += mv.count,
            SaeMove::Death { .. } => self.n_deaths += mv.count,
            SaeMove::Refuse { .. } => self.n_refusals += mv.count,
            SaeMove::Admit { .. } => self.n_admitted += mv.count,
            SaeMove::Restructure { .. } => self.n_restructures += mv.count,
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

    /// Record a move onto `stage` from `seed` that the evidence bought but the fit
    /// does not install.
    pub fn admit(
        &mut self,
        stage: MoveStage,
        seed: BirthSeed,
        count: usize,
        round: Option<usize>,
        evidence: MoveEvidence,
        objective: f64,
    ) {
        self.record(MigrationMove {
            kind: SaeMove::Admit { stage, seed },
            round,
            count,
            evidence,
            objective,
            predicted_dl_bits: None,
        });
    }

    /// Record an installed move on `stage` that adds and removes no atom.
    pub fn restructure(
        &mut self,
        stage: MoveStage,
        count: usize,
        round: Option<usize>,
        evidence: MoveEvidence,
        objective: f64,
    ) {
        self.record(MigrationMove {
            kind: SaeMove::Restructure { stage },
            round,
            count,
            evidence,
            objective,
            predicted_dl_bits: None,
        });
    }

    /// The ledger as a JSON record for a fitted model's payload: the tallies and
    /// every move with its stage, seed or reason, count, round and evidence. Unscored evidence (`NaN`) is `null`.
    #[must_use]
    pub fn to_json(&self) -> serde_json::Value {
        let moves = self
            .moves
            .iter()
            .map(|mv| {
                let (kind, stage, seed, reason) = match &mv.kind {
                    SaeMove::Birth { stage, seed } => ("birth", stage, Some(*seed), None),
                    SaeMove::Death { stage, reason } => ("death", stage, None, Some(reason)),
                    SaeMove::Refuse { stage, reason } => ("refuse", stage, None, Some(reason)),
                    SaeMove::Admit { stage, seed } => ("admitted", stage, Some(*seed), None),
                    SaeMove::Restructure { stage } => ("restructure", stage, None, None),
                };
                serde_json::json!({
                    "kind": kind,
                    "stage": stage.code(),
                    "seed": seed.map(BirthSeed::code),
                    "reason": reason.map(|reason| format!("{reason:?}")),
                    "round": mv.round,
                    "count": mv.count,
                    "reml_delta": mv.evidence.reml_delta,
                    "rank_charge": mv.evidence.rank_charge,
                    "dl_bits": mv.evidence.dl_bits,
                    "objective": mv.objective,
                    "predicted_dl_bits": mv.predicted_dl_bits,
                })
            })
            .collect::<Vec<_>>();
        serde_json::json!({
            "n_births": self.n_births,
            "n_deaths": self.n_deaths,
            "n_refusals": self.n_refusals,
            "n_admitted": self.n_admitted,
            "n_restructures": self.n_restructures,
            "moves": moves,
        })
    }

    /// Fold one structure-search round's [`SearchLedger`] into the unified
    /// currency, mapping each adjudicated move + verdict onto a birth / death /
    /// refusal / restructure priced by the banked e-process evidence
    /// (`bits_from_nats(log_e)`). An accepted move is booked by what
    /// `apply_structure_move` does to the dictionary, so across the accepted
    /// moves `births − deaths` tracks additions less demotions: a residual-factor
    /// birth adds an atom seeded from the residual pool; a fission adds a child
    /// cloned from an existing curved chart; a fusion and a destructive glue
    /// fold one atom into the other (a death); an atlas registration keeps both
    /// charts (a restructure). Demotions retain index-stable slots within the
    /// round; this is not the immediate change in the atom vector length.
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
    pub(crate) fn record_search_round(
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
                // Structure search never PC-reseeds: a birth seeds from the
                // residual-factor pool and a fission from the chart it splits.
                MoveVerdict::Accepted { log_e } => {
                    let evidence = MoveEvidence::from_log_e(*log_e);
                    match &record.mv {
                        StructureMove::Birth { .. } => self.birth(
                            stage,
                            BirthSeed::ResidualFactor,
                            1,
                            Some(round),
                            evidence,
                            f64::NAN,
                        ),
                        StructureMove::Fission { .. } => self.birth(
                            stage,
                            BirthSeed::CurvedChart,
                            1,
                            Some(round),
                            evidence,
                            f64::NAN,
                        ),
                        StructureMove::Death { .. } => self.death(
                            stage,
                            MoveReason::DeadRouting,
                            1,
                            Some(round),
                            evidence,
                            f64::NAN,
                        ),
                        StructureMove::Fusion { .. }
                        | StructureMove::Glue {
                            outcome: ChartGlueOutcome::Fuse,
                            ..
                        } => self.death(
                            stage,
                            MoveReason::Fused,
                            1,
                            Some(round),
                            evidence,
                            f64::NAN,
                        ),
                        StructureMove::Glue {
                            outcome: ChartGlueOutcome::RegisterAtlas,
                            ..
                        } => self.restructure(stage, 1, Some(round), evidence, f64::NAN),
                    }
                }
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

    /// #2023 criterion 3: an admitted move (a census verdict the fit does not install)
    /// is counted apart from births and refusals, and the payload record names it.
    #[test]
    fn admitted_moves_are_counted_apart_from_births_and_refusals_2023() {
        let mut ledger = SaeMigrationLedger::new();
        ledger.admit(
            MoveStage::Curved,
            BirthSeed::LinearAtom,
            2,
            None,
            MoveEvidence::from_dl_bits(5.0),
            f64::NAN,
        );
        assert_eq!(ledger.n_admitted, 2);
        assert_eq!(
            (ledger.n_births, ledger.n_deaths, ledger.n_refusals),
            (0, 0, 0),
            "an admitted move adds no atom and refuses nothing"
        );
        let record = ledger.to_json();
        assert_eq!(record["n_admitted"].as_u64(), Some(2));
        assert_eq!(record["moves"][0]["kind"].as_str(), Some("admitted"));
        assert_eq!(
            record["moves"][0]["seed"].as_u64(),
            Some(BirthSeed::LinearAtom.code())
        );
    }

    /// #3771: an accepted structure move is booked by what it does to the
    /// dictionary, so `births − deaths` over the accepted moves counts additions
    /// less demotions (+1 birth, +1 fission, −1 fusion, −1 fuse glue, 0 atlas
    /// registration, −1 death = −1), and no merge is reported as a birth.
    #[test]
    fn accepted_structure_moves_are_booked_by_their_atom_count_change_3771() {
        use gam_solve::structure_search::MoveRecord;
        use gam_terms::inference::structure_evidence::ClaimKind;
        let accepted = |mv: StructureMove| MoveRecord {
            mv,
            trigger: 0.0,
            structure_hash: 0,
            claim: ClaimKind::Custom {
                label: String::new(),
            },
            verdict: MoveVerdict::Accepted { log_e: LN_2 },
        };
        let search = SearchLedger {
            alpha: 0.05,
            moves: vec![
                accepted(StructureMove::Birth { candidate: 0 }),
                accepted(StructureMove::Fission { atom: 0 }),
                accepted(StructureMove::Fusion { a: 0, b: 1 }),
                accepted(StructureMove::Glue {
                    a: 0,
                    b: 2,
                    outcome: ChartGlueOutcome::Fuse,
                }),
                accepted(StructureMove::Glue {
                    a: 0,
                    b: 3,
                    outcome: ChartGlueOutcome::RegisterAtlas,
                }),
                accepted(StructureMove::Death { atom: 4 }),
            ],
            collapse_events: Vec::new(),
        };
        let mut ledger = SaeMigrationLedger::new();
        ledger.record_search_round(0, &search, &HashMap::new());

        assert_eq!(
            ledger.moves.len(),
            6,
            "every verdict records exactly one move"
        );
        assert_eq!(
            (
                ledger.n_births,
                ledger.n_deaths,
                ledger.n_restructures,
                ledger.n_refusals
            ),
            (2, 3, 1, 0)
        );
        assert_eq!(ledger.n_births as isize - ledger.n_deaths as isize, -1);
        let kinds: Vec<SaeMove> = ledger.moves.iter().map(|mv| mv.kind.clone()).collect();
        assert_eq!(
            kinds,
            vec![
                SaeMove::Birth {
                    stage: MoveStage::Curved,
                    seed: BirthSeed::ResidualFactor,
                },
                SaeMove::Birth {
                    stage: MoveStage::Curved,
                    seed: BirthSeed::CurvedChart,
                },
                SaeMove::Death {
                    stage: MoveStage::Curved,
                    reason: MoveReason::Fused,
                },
                SaeMove::Death {
                    stage: MoveStage::Curved,
                    reason: MoveReason::Fused,
                },
                SaeMove::Restructure {
                    stage: MoveStage::Curved,
                },
                SaeMove::Death {
                    stage: MoveStage::Curved,
                    reason: MoveReason::DeadRouting,
                },
            ]
        );
        for mv in &ledger.moves {
            assert_eq!(
                mv.evidence.dl_bits, 1.0,
                "ln 2 nats of banked evidence is one bit"
            );
        }
        let record = ledger.to_json();
        assert_eq!(record["n_restructures"].as_u64(), Some(1));
        assert_eq!(record["moves"][4]["kind"].as_str(), Some("restructure"));
    }
}
