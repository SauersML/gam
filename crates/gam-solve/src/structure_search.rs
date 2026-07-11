//! #976 — evidence-guarded dictionary structure search: atom birth / death /
//! fission / fusion as anytime-valid hypothesis tests, with a deterministic,
//! serializable [`SearchLedger`] as the honesty surface.
//!
//! # What this is
//!
//! The two documented SAE pathologies, restated statistically:
//!
//! * **Feature absorption** (an A⇒B hierarchy makes sparsity fold B's content
//!   into A's direction): an absorbing atom's code distribution carries
//!   substructure — detectable misspecification, found by a within-atom audit
//!   and corrected by a FISSION move.
//! * **Feature shattering** (one curved family smeared across many
//!   near-duplicate flat atoms): shattered atoms have dependent codes
//!   (`gam_sae::atom_codes::CoactivationStats::dependence`) and joint
//!   structure when refit together — corrected by a FUSION move.
//!
//! This module owns the MOVE ENGINE: canonical deterministic proposal order,
//! structural-hash deduplication, e-process-gated acceptance, and the ledger.
//! It is generic over the fitter — the caller supplies the state type and four
//! closures (apply / evaluate / null-sup / refit), exactly the surface
//! [`run_atom_birth_gate`] already pins down. Warm structure inheritance is
//! enforced by construction: a candidate state is built FROM the parent state
//! (`apply_move(&parent, &mv)`), never from scratch — cold restarts after
//! structure moves are both slow and collapse-prone, so the API gives them no
//! entry point.
//!
//! # Acceptance is a hypothesis test, not a threshold (#984)
//!
//! The original #976 design accepted a move when
//! `Δ(neg log evidence) < −margin` under the Laplace normalizer. That is the
//! K vs K+1 boundary / Davies-regime comparison where likelihood-ratio
//! thresholds are invalid (the null sits on the boundary of the alternative;
//! the new atom's parameters vanish under the null). Acceptance here is
//! therefore routed through the universal-inference e-process gates of
//! [`gam_terms::inference::structure_evidence`]:
//!
//! * **Birth / fission / fusion** each assert structure BEYOND what the
//!   current dictionary class expresses, so each runs an [`AtomBirthGate`]
//!   (the mechanics are claim-generic: predictable alternative, honest
//!   null sup, Ville threshold at the α fixed in [`MoveBudget`]). A move is
//!   applied only when its claim is **Certified**; otherwise the structure is
//!   unchanged and the claim stays **Contested** in the [`StructureLedger`]
//!   with its banked evidence — the input to the #984 probe-design loop.
//! * **Death is never certifiable, by construction.** The K−1 class is nested
//!   inside the current class, so the split-likelihood e-value satisfies
//!   `E ≤ 1` pointwise (the null sup dominates any sub-model fit): no amount
//!   of data can *prove* an atom unnecessary — only fail to prove it
//!   necessary. The demote-never-reject philosophy is therefore not a policy
//!   choice here, it is what the math leaves: a death proposal DEMOTES an atom
//!   whose `AtomExists` claim has never certified (trigger: diverged ARD
//!   precision), and is VETOED for a certified atom (a Ville crossing is
//!   permanent — later evidence retreat cannot un-prove existence).
//!
//! # Determinism
//!
//! No RNG, no clock. Proposals are sorted by the canonical order (deaths by
//! ARD precision descending, fissions by audit significance ascending, fusions
//! by code dependence descending, births last by proposal mass descending; ties
//! broken by structural hash), deduplicated by the caller-computed structural
//! hash (the `TermCollectionSpec` hash machinery, #869), and processed
//! sequentially. Identical inputs ⇒ identical serialized [`SearchLedger`] —
//! which is what keeps replicate-null comparisons (#910/#943) valid across
//! structure changes.
//!
//! The ledger reports a certified **local** mode: the moves explored, the
//! evidence for accepted ones, and the evidence gaps to rejected alternatives.
//! No global-optimality theater.

use serde::{Deserialize, Serialize};
use std::collections::HashSet;

use gam_terms::inference::structure_evidence::{
    ClaimKind, GateVerdict, StructureLedger, run_atom_birth_gate,
};

/// One proposed structural move. Atom indices are STABLE IDENTIFIERS for the
/// duration of one [`search`] round: the caller's `apply_move` must not
/// reindex surviving atoms (mark dead atoms inactive, append born atoms) —
/// the engine relies on this to detect conflicting proposals.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum StructureMove {
    /// Add a new atom. `candidate` indexes the caller's proposal list (e.g.
    /// scaffold clusters on first build; whitened residual-factor directions
    /// thereafter — see #974's rescope: proposals must come from the WHITENED
    /// residual subspace, raw-Euclidean Λ skews loud-but-inert).
    Birth { candidate: usize },
    /// Demote an atom whose existence was never certified (ARD precision
    /// diverged). Never applies to a certified atom.
    Death { atom: usize },
    /// Split an atom along detected substructure (within-atom audit / #975
    /// vanished-interaction carve).
    Fission { atom: usize },
    /// Merge two atoms into one joint structure (dependent codes + joint
    /// interaction evidence — #975's binding, in reverse).
    Fusion { a: usize, b: usize },
    /// Glue two CHARTS of one manifold into one atom (#1890). Distinct from
    /// [`StructureMove::Fusion`]: the co-activation fusion lane fires on
    /// DEPENDENT (co-firing) codes, but atoms over-tiling a single manifold have
    /// DISJOINT supports (each owns its own arc/patch) and therefore
    /// anti-correlated codes — invisible to fusion. The glue lane proposes such a
    /// pair on a GEOMETRIC pre-screen (decoder-frame principal angles × latent-
    /// support adjacency) and its acceptance is an EQUIVALENCE e-value on the
    /// seam (the two decoded charts coincide within an isometry tolerance,
    /// against the churn null) — a pre-computed e-value carried on the proposal's
    /// `trigger`, not the held-out fit-improvement gate the other moves use (a
    /// clean glue leaves EV tied, so a likelihood-ratio gate could never accept
    /// it). [`ChartGlueOutcome::Fuse`] folds an ordinary over-tile;
    /// [`ChartGlueOutcome::RegisterAtlas`] preserves a seam whose local charts
    /// cannot be replaced by one global chart (orientation reversal / pole).
    Glue {
        a: usize,
        b: usize,
        outcome: ChartGlueOutcome,
    },
}

/// Structural outcome certified by a chart-gluing seam.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum ChartGlueOutcome {
    /// A single orientable chart covers the union, so the redundant chart is
    /// physically folded and removed.
    Fuse,
    /// Both local charts are required.  Keep them as the partition-of-unity
    /// cover of one semantic atlas atom and persist their transition map.
    RegisterAtlas,
}

impl StructureMove {
    /// Atoms whose state this move modifies (births create, so touch none).
    fn touches(&self) -> Vec<usize> {
        match self {
            StructureMove::Birth { .. } => Vec::new(),
            StructureMove::Death { atom } | StructureMove::Fission { atom } => vec![*atom],
            StructureMove::Fusion { a, b } | StructureMove::Glue { a, b, .. } => vec![*a, *b],
        }
    }

    /// Canonical kind rank: deaths, fissions, fusions, glues, births. Glue sorts
    /// after fusion (both merge a pair) and before birth (#1890).
    fn kind_rank(&self) -> u8 {
        match self {
            StructureMove::Death { .. } => 0,
            StructureMove::Fission { .. } => 1,
            StructureMove::Fusion { .. } => 2,
            StructureMove::Glue { .. } => 3,
            StructureMove::Birth { .. } => 4,
        }
    }

    /// Whether the canonical order sorts this kind's trigger ascending
    /// (fission audits report significance levels — smaller is more urgent)
    /// or descending (ARD precision, code dependence, proposal mass).
    fn trigger_ascending(&self) -> bool {
        matches!(self, StructureMove::Fission { .. })
    }
}

/// One proposal: the move, its trigger statistic (the canonical-order key,
/// kind-specific — see [`StructureMove`] docs), the caller-computed structural
/// hash of the POST-move specification (dedup key), and the structural claim
/// the move asserts (registered in the [`StructureLedger`] so the dictionary
/// certificate covers it).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct MoveProposal {
    pub mv: StructureMove,
    /// Canonical-order key. Deaths: ARD amplitude precision (descending).
    /// Fissions: within-atom audit significance (ascending). Fusions: code
    /// dependence (descending). Births: explained proposal mass (descending).
    /// Must be finite.
    pub trigger: f64,
    /// Structural hash of the specification the move produces (#869
    /// `TermCollectionSpec` machinery). Two proposals with the same hash are
    /// the same structure; only the canonically-first is gated.
    pub structure_hash: u64,
    /// The claim this move asserts. Births: `AtomExists`. Fusions:
    /// `BindingEdge`. Fissions: a `Custom`/`GeometryKind` substructure claim.
    /// Deaths: the `AtomExists` claim CONSULTED for the veto/demote decision.
    pub claim: ClaimKind,
}

/// The search round's budget and error level.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct MoveBudget {
    /// Maximum structure-changing moves (accepted + demoted) applied this
    /// round; remaining proposals are recorded as `Deferred`, never silently
    /// dropped.
    pub max_moves: usize,
    /// The level every gate certifies at; fixed for the round so verdicts
    /// cannot be shopped.
    pub alpha: f64,
}

/// The per-proposal outcome. Every proposal handed to [`search`] gets exactly
/// one record — the no-silent-caps rule.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum MoveVerdict {
    /// Gate certified at α; the move was applied and the claim's evidence
    /// banked in the ledger.
    Accepted { log_e: f64 },
    /// Gate did not certify; structure unchanged, claim stays contested in
    /// the ledger with this evidence (the probe loop's input).
    Contested { log_e: f64 },
    /// Death applied to a never-certified atom (its contested evidence at the
    /// time of demotion is recorded).
    Demoted { log_e: f64 },
    /// Death proposal on a CERTIFIED atom — refused; Ville crossings are
    /// permanent.
    Vetoed { log_e: f64 },
    /// Same structural hash as a canonically-earlier proposal this round.
    Deduplicated,
    /// References an atom already modified this round; triggers are stale —
    /// re-propose next round against the new structure.
    Stale,
    /// Move budget exhausted before this proposal was reached.
    Deferred,
}

/// One ledger line: the proposal exactly as ranked, plus its verdict.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct MoveRecord {
    pub mv: StructureMove,
    pub trigger: f64,
    pub structure_hash: u64,
    pub claim: ClaimKind,
    pub verdict: MoveVerdict,
}

/// An assignment-collapse event from the joint fit (#976 Layer-1 guard): an
/// atom's support fell below the active-mass floor and was either re-seeded
/// (bounded budget) or recorded as terminally collapsed — an observable event,
/// never a silent death and never a fit error. Terminal collapses are the
/// natural death-proposal feed for the next [`search`] round.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct CollapseEvent {
    /// Outer iteration of the joint fit at which the breach was observed.
    pub iteration: usize,
    /// The collapsed atom.
    pub atom: usize,
    /// The atom's maximum active mass over rows at the breach (the collapse
    /// statistic: a legitimately sparse atom has small MEAN mass but high
    /// mass on its rows; only an atom with no material support anywhere has a
    /// small MAX).
    pub max_active_mass: f64,
    /// The floor breached.
    pub floor: f64,
    /// What the guard did.
    pub action: CollapseAction,
}

/// The guard's response to an active-mass breach.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum CollapseAction {
    /// The atom's gate logits were re-seeded to a mode-appropriate neutral
    /// (one second chance from a fresh basin; bounded budget per atom).
    Reseeded,
    /// Re-seed budget exhausted and the atom collapsed again: the collapse is
    /// (locally) the objective's verdict. Recorded once; the structure-search
    /// death move owns the decision from here.
    Terminal,
}

/// The serialized honesty surface of one search round: every proposal in
/// canonical order with its verdict, plus any collapse events the joint fit
/// recorded. Identical inputs produce a byte-identical serialization.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SearchLedger {
    /// The α every verdict in this round was gated at.
    pub alpha: f64,
    /// One record per proposal, in canonical processing order.
    pub moves: Vec<MoveRecord>,
    /// Layer-1 guard events carried from the joint fit (see
    /// [`CollapseEvent`]); attached by the caller.
    pub collapse_events: Vec<CollapseEvent>,
}

/// Result of one search round: the (possibly restructured) state and the
/// ledger.
pub struct SearchOutcome<S> {
    pub state: S,
    pub ledger: SearchLedger,
}

/// Sort proposals into the canonical deterministic order: kind rank (deaths,
/// fissions, fusions, births), then the kind's trigger direction, then
/// structural hash. Pure — no RNG, no clock — so the search path, and with it
/// the ledger, is a function of the inputs alone.
pub fn canonical_order(proposals: &mut [MoveProposal]) {
    proposals.sort_by(|x, y| {
        let xr = x.mv.kind_rank();
        let yr = y.mv.kind_rank();
        xr.cmp(&yr)
            .then_with(|| {
                let (xt, yt) = if x.mv.trigger_ascending() {
                    (x.trigger, y.trigger)
                } else {
                    (-x.trigger, -y.trigger)
                };
                xt.total_cmp(&yt)
            })
            .then_with(|| x.structure_hash.cmp(&y.structure_hash))
    });
}

/// Run one evidence-guarded structure-search round.
///
/// * `state` — the current fitted structure (dictionary). Moves are applied
///   sequentially; later gates run against the updated state.
/// * `proposals` — trigger-ranked candidate moves (any order; the engine
///   canonicalizes). Triggers must be finite.
/// * `shards` — the evaluation stream for the gates. Each certifiable move
///   streams over ALL shards (or until certified) under the universal-
///   inference contract of [`run_atom_birth_gate`]: the candidate is evaluated
///   on a shard strictly before being refit with it, so the plug-in is
///   predictable and the e-process valid under optional stopping. Validity
///   requires these shards be data the TRIGGERS were not tuned on (the same
///   estimation/evaluation split discipline as every e-value here).
/// * `ledger` — the dictionary's claim ledger, carried ACROSS rounds: claims
///   keep their banked evidence (idempotent registration), so a structure
///   contested this round resumes from its evidence next round, and the death
///   veto sees certifications from any earlier round.
/// * `apply_move` — build the candidate state from the PARENT state (warm
///   inheritance by construction). For deaths this is the demotion itself.
/// * `eval_log_lik(candidate, shard)` — evaluation log-likelihood of a shard
///   under the candidate as currently fit (prior shards only — the engine
///   guarantees the call order).
/// * `null_sup_log_lik(state, shard)` — the HONEST sup: the current structure
///   refit on the shard. Under-maximizing this side inflates every e-value
///   and voids validity; it is the one closure that must genuinely optimize.
/// * `refit(candidate, shard)` — fold the shard into the candidate. Likelihood
///   evaluation and refitting are fallible: undefined scores or non-convergence
///   abort the round rather than becoming neutral evidence.
pub fn search<S, Sh>(
    mut state: S,
    mut proposals: Vec<MoveProposal>,
    shards: &[Sh],
    budget: &MoveBudget,
    ledger: &mut StructureLedger,
    mut apply_move: impl FnMut(&S, &StructureMove) -> Result<S, String>,
    mut eval_log_lik: impl FnMut(&S, &Sh) -> Result<f64, String>,
    mut null_sup_log_lik: impl FnMut(&S, &Sh) -> Result<f64, String>,
    mut refit: impl FnMut(S, &Sh) -> Result<S, String>,
) -> Result<SearchOutcome<S>, String> {
    if !(budget.alpha > 0.0 && budget.alpha < 1.0) {
        return Err(format!(
            "structure_search: alpha must be in (0,1), got {}",
            budget.alpha
        ));
    }
    if let Some(bad) = proposals.iter().find(|p| !p.trigger.is_finite()) {
        return Err(format!(
            "structure_search: non-finite trigger {} on {:?}",
            bad.trigger, bad.mv
        ));
    }
    canonical_order(&mut proposals);

    let mut seen_hashes: HashSet<u64> = HashSet::new();
    let mut touched: Vec<usize> = Vec::new();
    let mut moves_applied = 0usize;
    let mut records: Vec<MoveRecord> = Vec::with_capacity(proposals.len());

    for prop in proposals {
        // Dedup is a property of the proposal stream (a duplicate structural
        // hash describes a proposal that the engine has already considered),
        // so it is decided BEFORE the budget gate: a duplicate of an
        // already-applied move stays a duplicate even when the budget is
        // exhausted. Reversing this order mislabels duplicates as deferred,
        // which breaks the dedup-vs-defer accounting downstream (a deferred
        // record is replayed by the next round; a deduplicated one is not).
        let verdict = if !seen_hashes.insert(prop.structure_hash) {
            MoveVerdict::Deduplicated
        } else if moves_applied >= budget.max_moves {
            MoveVerdict::Deferred
        } else if prop.mv.touches().iter().any(|a| touched.contains(a)) {
            MoveVerdict::Stale
        } else {
            match &prop.mv {
                StructureMove::Death { atom } => {
                    let idx = ledger.register(prop.claim.clone());
                    let evidence = &ledger.claims()[idx].evidence;
                    let log_e = evidence.log_evidence();
                    if evidence.rejects_at(budget.alpha) {
                        MoveVerdict::Vetoed { log_e }
                    } else {
                        state = apply_move(&state, &prop.mv)?;
                        touched.push(*atom);
                        moves_applied += 1;
                        MoveVerdict::Demoted { log_e }
                    }
                }
                StructureMove::Glue { a, b, .. } => {
                    // Equivalence acceptance (#1890): the seam e-value was
                    // computed at harvest against the churn null (the two charts
                    // coincide within an isometry tolerance) and carried on
                    // `trigger`. Bank it and glue when the accumulated evidence
                    // certifies at α — never the held-out fit-improvement gate
                    // the merge/split/birth moves use, which a clean glue's tied
                    // EV could never clear. Composes with the e-BH ledger like
                    // every other claim.
                    let idx = ledger.register(prop.claim.clone());
                    ledger.absorb_log(idx, prop.trigger)?;
                    let evidence = &ledger.claims()[idx].evidence;
                    let log_e = evidence.log_evidence();
                    if evidence.rejects_at(budget.alpha) {
                        state = apply_move(&state, &prop.mv)?;
                        touched.push(*a);
                        touched.push(*b);
                        moves_applied += 1;
                        MoveVerdict::Accepted { log_e }
                    } else {
                        MoveVerdict::Contested { log_e }
                    }
                }
                mv @ (StructureMove::Birth { .. }
                | StructureMove::Fission { .. }
                | StructureMove::Fusion { .. }) => {
                    let candidate = apply_move(&state, mv)?;
                    // `shards.iter()` makes the gate's shard item `&Sh`, so the
                    // closures receive `&&Sh`; deref once back to the caller's
                    // `&Sh` surface.
                    let (gate, folded) = run_atom_birth_gate(
                        budget.alpha,
                        candidate,
                        shards.iter(),
                        |c, sh| eval_log_lik(c, *sh),
                        |sh| null_sup_log_lik(&state, *sh),
                        |c, sh| refit(c, *sh),
                    )?;
                    let idx = ledger.register(prop.claim.clone());
                    match gate.verdict() {
                        GateVerdict::Certified { log_e } => {
                            ledger.absorb_log(idx, log_e)?;
                            state = folded;
                            touched.extend(mv.touches());
                            moves_applied += 1;
                            MoveVerdict::Accepted { log_e }
                        }
                        GateVerdict::Contested { log_e } => {
                            ledger.absorb_log(idx, log_e)?;
                            MoveVerdict::Contested { log_e }
                        }
                    }
                }
            }
        };
        records.push(MoveRecord {
            mv: prop.mv,
            trigger: prop.trigger,
            structure_hash: prop.structure_hash,
            claim: prop.claim,
            verdict,
        });
    }

    Ok(SearchOutcome {
        state,
        ledger: SearchLedger {
            alpha: budget.alpha,
            moves: records,
            collapse_events: Vec::new(),
        },
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Test fixture: a "dictionary" is a sorted set of atom labels; the
    /// per-shard log-likelihood advantage of a candidate over the honest null
    /// sup is scripted per label, so the statistics are exact and the tests
    /// exercise the ENGINE (ordering, gating, veto, budget, determinism) —
    /// the e-process statistics themselves are pinned in structure_evidence.
    type Dict = Vec<&'static str>;

    /// Per-shard advantage of a state over the null sup: +0.8 nats/shard when
    /// the planted "real" atom is present, −0.2 when the spurious "fake" fused
    /// atom is present, 0 otherwise.
    fn advantage(state: &Dict) -> f64 {
        let mut adv = 0.0;
        if state.contains(&"real") {
            adv += 0.8;
        }
        if state.contains(&"fake") {
            adv -= 0.2;
        }
        adv
    }

    fn apply(state: &Dict, mv: &StructureMove) -> Result<Dict, String> {
        let mut next = state.clone();
        match mv {
            StructureMove::Birth { candidate } => {
                next.push(if *candidate == 0 { "real" } else { "extra" });
            }
            StructureMove::Death { atom } => {
                if *atom < next.len() {
                    next[*atom] = "dead";
                }
            }
            StructureMove::Fusion { .. } => next.push("fake"),
            StructureMove::Glue { .. } => next.push("glued"),
            StructureMove::Fission { .. } => next.push("split"),
        }
        Ok(next)
    }

    fn run(
        state: Dict,
        proposals: Vec<MoveProposal>,
        n_shards: usize,
        budget: &MoveBudget,
        ledger: &mut StructureLedger,
    ) -> SearchOutcome<Dict> {
        let shards: Vec<f64> = vec![1.0; n_shards];
        search(
            state,
            proposals,
            &shards,
            budget,
            ledger,
            apply,
            |c, _sh| Ok(-100.0 + advantage(c)),
            |_s, _sh| Ok(-100.0),
            |c, _sh| Ok(c),
        )
        .expect("search runs")
    }

    fn birth(candidate: usize, trigger: f64, hash: u64) -> MoveProposal {
        MoveProposal {
            mv: StructureMove::Birth { candidate },
            trigger,
            structure_hash: hash,
            claim: ClaimKind::AtomExists {
                atom: 100 + candidate,
            },
        }
    }

    /// Canonical order: deaths → fissions → fusions → births, direction-aware
    /// triggers, hash tiebreak.
    #[test]
    fn canonical_order_ranks_kinds_and_triggers() {
        let mut props = vec![
            birth(0, 0.5, 7),
            MoveProposal {
                mv: StructureMove::Fusion { a: 1, b: 2 },
                trigger: 0.9,
                structure_hash: 3,
                claim: ClaimKind::BindingEdge { a: 1, b: 2 },
            },
            MoveProposal {
                mv: StructureMove::Death { atom: 4 },
                trigger: 1e6,
                structure_hash: 1,
                claim: ClaimKind::AtomExists { atom: 4 },
            },
            MoveProposal {
                mv: StructureMove::Fission { atom: 3 },
                trigger: 0.01,
                structure_hash: 2,
                claim: ClaimKind::Custom {
                    label: "fission:3".to_string(),
                },
            },
            MoveProposal {
                mv: StructureMove::Fission { atom: 5 },
                trigger: 0.001,
                structure_hash: 9,
                claim: ClaimKind::Custom {
                    label: "fission:5".to_string(),
                },
            },
        ];
        canonical_order(&mut props);
        assert!(matches!(props[0].mv, StructureMove::Death { atom: 4 }));
        // Fissions ascending by significance: 0.001 before 0.01.
        assert!(matches!(props[1].mv, StructureMove::Fission { atom: 5 }));
        assert!(matches!(props[2].mv, StructureMove::Fission { atom: 3 }));
        assert!(matches!(props[3].mv, StructureMove::Fusion { .. }));
        assert!(matches!(props[4].mv, StructureMove::Birth { .. }));
    }

    /// A planted birth certifies (0.8 nats/shard × 10 shards crosses ln 20),
    /// updates the state, and banks certified evidence in the claim ledger; a
    /// spurious fusion stays contested, leaves the state unchanged, and its
    /// claim keeps (negative) evidence for the probe loop.
    #[test]
    fn birth_certifies_and_null_fusion_stays_contested() {
        let mut ledger = StructureLedger::new();
        let budget = MoveBudget {
            max_moves: 8,
            alpha: 0.05,
        };
        let proposals = vec![
            birth(0, 1.0, 11),
            MoveProposal {
                mv: StructureMove::Fusion { a: 0, b: 1 },
                trigger: 0.8,
                structure_hash: 12,
                claim: ClaimKind::BindingEdge { a: 0, b: 1 },
            },
        ];
        let out = run(vec!["a", "b"], proposals, 10, &budget, &mut ledger);

        // Fusion is gated first (canonical order) and must NOT certify.
        let fusion_rec = &out.ledger.moves[0];
        assert!(matches!(fusion_rec.mv, StructureMove::Fusion { .. }));
        match fusion_rec.verdict {
            MoveVerdict::Contested { log_e } => assert!(log_e < 0.0),
            ref v => panic!("spurious fusion must stay contested, got {v:?}"),
        }
        // Birth certifies and the atom is in the final state.
        let birth_rec = &out.ledger.moves[1];
        match birth_rec.verdict {
            MoveVerdict::Accepted { log_e } => assert!(log_e >= -(0.05f64.ln())),
            ref v => panic!("planted birth must certify, got {v:?}"),
        }
        assert!(out.state.contains(&"real"));
        assert!(!out.state.contains(&"fake"));

        // Ledger: birth claim certified, fusion claim contested with evidence.
        let cert = ledger.certify(0.05);
        let confirmed: Vec<_> = cert.confirmed().map(|e| e.kind.clone()).collect();
        assert!(confirmed.contains(&ClaimKind::AtomExists { atom: 100 }));
        assert!(
            cert.contested()
                .any(|e| e.kind == ClaimKind::BindingEdge { a: 0, b: 1 } && e.log_e < 0.0)
        );
    }

    /// Death is vetoed for a certified atom (Ville permanence) and demotes a
    /// never-certified one; a later proposal touching the demoted atom is
    /// stale.
    #[test]
    fn death_vetoes_certified_demotes_contested_and_staleness_propagates() {
        let mut ledger = StructureLedger::new();
        let certified = ledger.register(ClaimKind::AtomExists { atom: 0 });
        ledger.absorb_log(certified, 5.0).unwrap(); // > ln 20 ⇒ certified at 0.05
        let weak = ledger.register(ClaimKind::AtomExists { atom: 1 });
        ledger.absorb_log(weak, -1.0).unwrap();

        let budget = MoveBudget {
            max_moves: 8,
            alpha: 0.05,
        };
        let proposals = vec![
            MoveProposal {
                mv: StructureMove::Death { atom: 0 },
                trigger: 9.0,
                structure_hash: 21,
                claim: ClaimKind::AtomExists { atom: 0 },
            },
            MoveProposal {
                mv: StructureMove::Death { atom: 1 },
                trigger: 8.0,
                structure_hash: 22,
                claim: ClaimKind::AtomExists { atom: 1 },
            },
            MoveProposal {
                mv: StructureMove::Fusion { a: 1, b: 2 },
                trigger: 0.9,
                structure_hash: 23,
                claim: ClaimKind::BindingEdge { a: 1, b: 2 },
            },
        ];
        let out = run(vec!["a", "b", "c"], proposals, 4, &budget, &mut ledger);

        assert!(matches!(
            out.ledger.moves[0].verdict,
            MoveVerdict::Vetoed { .. }
        ));
        match out.ledger.moves[1].verdict {
            MoveVerdict::Demoted { log_e } => assert!((log_e - (-1.0)).abs() < 1e-12),
            ref v => panic!("contested atom must demote, got {v:?}"),
        }
        assert_eq!(out.state[1], "dead");
        assert_eq!(out.state[0], "a", "vetoed death must not touch the atom");
        // Fusion references the demoted atom ⇒ stale, not gated.
        assert!(matches!(out.ledger.moves[2].verdict, MoveVerdict::Stale));
    }

    /// Budget exhaustion defers (records, never silently drops), and duplicate
    /// structural hashes are deduplicated.
    #[test]
    fn budget_defers_and_hash_dedups() {
        let mut ledger = StructureLedger::new();
        let budget = MoveBudget {
            max_moves: 1,
            alpha: 0.05,
        };
        let proposals = vec![
            birth(0, 1.0, 31),
            birth(0, 0.9, 31), // same structure, lower trigger ⇒ dedup
            birth(1, 0.5, 32), // budget exhausted by then ⇒ deferred
        ];
        let out = run(vec!["a"], proposals, 10, &budget, &mut ledger);
        assert!(matches!(
            out.ledger.moves[0].verdict,
            MoveVerdict::Accepted { .. }
        ));
        assert!(matches!(
            out.ledger.moves[1].verdict,
            MoveVerdict::Deduplicated
        ));
        assert!(matches!(out.ledger.moves[2].verdict, MoveVerdict::Deferred));
    }

    /// Identical inputs ⇒ byte-identical serialized ledger (the replicate-null
    /// validity requirement). Proposals are supplied in scrambled orders.
    #[test]
    fn ledger_is_deterministic_across_runs() {
        let props = || {
            vec![
                birth(0, 1.0, 41),
                MoveProposal {
                    mv: StructureMove::Death { atom: 1 },
                    trigger: 3.0,
                    structure_hash: 42,
                    claim: ClaimKind::AtomExists { atom: 1 },
                },
                MoveProposal {
                    mv: StructureMove::Fusion { a: 0, b: 2 },
                    trigger: 0.7,
                    structure_hash: 43,
                    claim: ClaimKind::BindingEdge { a: 0, b: 2 },
                },
            ]
        };
        let budget = MoveBudget {
            max_moves: 8,
            alpha: 0.05,
        };
        let mut scrambled = props();
        scrambled.reverse();

        let mut ledger_a = StructureLedger::new();
        let out_a = run(vec!["a", "b", "c"], props(), 6, &budget, &mut ledger_a);
        let mut ledger_b = StructureLedger::new();
        let out_b = run(vec!["a", "b", "c"], scrambled, 6, &budget, &mut ledger_b);

        let ser_a = serde_json::to_string(&out_a.ledger).expect("serialize");
        let ser_b = serde_json::to_string(&out_b.ledger).expect("serialize");
        assert_eq!(ser_a, ser_b);
        assert_eq!(out_a.state, out_b.state);
    }

    /// Non-finite triggers and degenerate α are rejected loudly.
    #[test]
    fn invalid_inputs_error() {
        let mut ledger = StructureLedger::new();
        let shards: Vec<f64> = vec![1.0];
        let bad_alpha = search(
            vec!["a"],
            Vec::<MoveProposal>::new(),
            &shards,
            &MoveBudget {
                max_moves: 1,
                alpha: 1.0,
            },
            &mut ledger,
            apply,
            |_c: &Dict, _sh| Ok(0.0),
            |_s, _sh| Ok(0.0),
            |c, _sh| Ok(c),
        );
        assert!(bad_alpha.is_err());

        let bad_trigger = search(
            vec!["a"],
            vec![birth(0, f64::NAN, 1)],
            &shards,
            &MoveBudget {
                max_moves: 1,
                alpha: 0.05,
            },
            &mut ledger,
            apply,
            |_c: &Dict, _sh| Ok(0.0),
            |_s, _sh| Ok(0.0),
            |c, _sh| Ok(c),
        );
        assert!(bad_trigger.is_err());
    }

    #[test]
    fn likelihood_and_refit_failures_abort_the_search() {
        let shards = vec![1.0_f64];
        let budget = MoveBudget {
            max_moves: 1,
            alpha: 0.05,
        };

        for failing_stage in 0..3 {
            let mut ledger = StructureLedger::new();
            let result = search(
                vec!["a"],
                vec![birth(0, 1.0, 99)],
                &shards,
                &budget,
                &mut ledger,
                apply,
                |_candidate, _shard| {
                    if failing_stage == 0 {
                        Err("candidate likelihood failed".to_string())
                    } else {
                        Ok(-1.0)
                    }
                },
                |_null, _shard| {
                    if failing_stage == 1 {
                        Err("null fit failed".to_string())
                    } else {
                        Ok(-2.0)
                    }
                },
                |state, _shard| {
                    if failing_stage == 2 {
                        Err("alternative refit failed".to_string())
                    } else {
                        Ok(state)
                    }
                },
            );
            assert!(
                result.is_err(),
                "failure stage {failing_stage} must abort rather than mint evidence"
            );
        }
    }
}
