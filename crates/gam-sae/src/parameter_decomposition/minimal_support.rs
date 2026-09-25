//! Per-position minimal supports at a declared fidelity, found against the model
//! itself (#2951).
//!
//! # The object
//!
//! A decomposition's pieces `P_1, ..., P_C` sum exactly to the teacher's parameters.
//! At every position `t` of an executed batch, a support is the set of pieces kept on:
//! the model runs with the others removed at that position. A support is admissible
//! when every position's divergence from the teacher, `KL_t`, is at most the declared
//! fidelity `ε`. The problem is the smallest admissible supports.
//!
//! No selector is learned. A learned gate is a restriction of this search: any gate's
//! output is one admissible support or not, and the search below can return it.
//!
//! # The search
//!
//! Start from the full support, where `KL_t = 0` up to roundoff, or from a caller's
//! supports (a warm start). An inadmissible start is first repaired: removed pieces are
//! restored in decreasing predicted gain wherever a violation can come from, until the
//! supports are admissible (the full support always is, or the start is refused).
//! Then each round:
//!
//! 1. the executor returns, at the current supports, each position's exact `KL_t` and a
//!    predicted increase `ĉ_tc` of removing each kept piece `c` at `t`;
//! 2. at every position, the kept pieces are ordered by increasing `ĉ_tc` (ties to the
//!    lower index) and the first `r_t` of them are proposed for removal, where `r_t` is
//!    the position's trust radius (a count of pieces, starting at one). Under the mean
//!    form one budget is shared and its exact slack `P·ε − Σ_t KL_t` is known, so all
//!    kept pieces are ordered together and the proposal is the longest prefix whose
//!    predicted increase, calibrated by the ratio of exact to predicted increase on the
//!    latest evaluated proposal, stays below the slack (at least the cheapest piece).
//!    The raw prediction underestimates the joint effect of many removals; the ratio is
//!    measured, not declared;
//! 3. the executor evaluates the proposal exactly. A position's divergence depends only
//!    on removals at that position and at earlier positions of its own sequence (causal
//!    attention; independent examples are sequences of length one), a structure the
//!    executor declares. While some position exceeds `ε`, every proposal in its causal
//!    cone (itself and the earlier positions of its sequence) is halved, dropping its
//!    costliest half, and the proposal is re-evaluated. Under the mean form the budget is
//!    shared, so a violation instead drops the costliest half of all proposed removals,
//!    wherever they are. Every halving shrinks the
//!    proposal and an empty proposal is the current, admissible supports, so this ends.
//! 4. (per-position form) a position whose proposal was accepted whole doubles its radius; one whose
//!    proposal was halved keeps the size that was accepted (at least one). The radius is
//!    the classic trust-region update on the count: it is learned from exact
//!    acceptances, not from the prediction's scale, which underestimates the joint
//!    effect of many removals.
//!
//! The prediction only orders proposals; acceptance reads the exact
//! divergence, so a poor prediction costs rounds, never fidelity. Every accepted round
//! removes at least one piece and none is ever restored, so the search ends after at
//! most `P·C` accepted rounds, at a fixed point of the round rule: no proposal the rule
//! can build is admissible. That is a local, not a certified global, minimum, and the
//! report says so by carrying the rule's last refusal.
use ndarray::{Array1, Array2, ArrayView2};
use std::fmt;

/// What "admissible" means, declared by the experiment.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Fidelity {
    /// Every position's divergence is below `ε` (strictly, so a barrier on the
    /// admissible set is finite at every admissible point).
    PerPosition(f64),
    /// The mean divergence over the positions is below `ε` (the form a batch-mean
    /// metric such as VPD's reports); positions share one budget.
    Mean(f64),
}

impl Fidelity {
    fn bound(self) -> f64 {
        match self {
            Self::PerPosition(eps) | Self::Mean(eps) => eps,
        }
    }

    /// Positions whose proposals a violation can be blamed on, before causal
    /// widening: the violating positions, or every position for a shared budget.
    fn violators(self, divergence: &Array1<f64>) -> Vec<usize> {
        match self {
            Self::PerPosition(eps) => (0..divergence.len()).filter(|&t| divergence[t] >= eps).collect(),
            Self::Mean(eps) => {
                let mean = divergence.iter().sum::<f64>() / divergence.len() as f64;
                if mean >= eps { (0..divergence.len()).collect() } else { Vec::new() }
            }
        }
    }
}

/// One exact evaluation at a set of supports.
#[derive(Clone, Debug)]
pub struct SupportEvaluation {
    /// `KL_t` per position (`P`).
    pub divergence: Array1<f64>,
    /// Predicted increase of `KL_t` from removing piece `c` at position `t` (`P × C`);
    /// read only where the piece is kept.
    pub removal_cost: Array2<f64>,
    /// Predicted decrease of `KL_t` from restoring piece `c` at position `t` (`P × C`);
    /// read only where the piece is removed (repairing an inadmissible start).
    pub restore_gain: Array2<f64>,
}

/// Runs the model with pieces removed where `keep` is false (`P × C`).
pub trait SupportExecutor {
    fn evaluate(&mut self, keep: ArrayView2<'_, bool>) -> Result<SupportEvaluation, String>;
}

/// Why a minimal-support search stopped without a result.
#[derive(Debug)]
pub enum SupportError {
    /// `ε` is negative or not finite.
    Fidelity(f64),
    /// The executor returned arrays of the wrong shape.
    Shape { what: &'static str, expected: (usize, usize), found: (usize, usize) },
    /// A divergence or prediction is not finite.
    NonFinite { what: &'static str },
    /// The starting supports (the full support when none is given) already exceed `ε`
    /// at some position.
    StartInadmissible { position: usize, divergence: f64 },
    /// The positions are not a whole number of sequences of the declared length.
    Sequence { positions: usize, sequence: usize },
    /// The executor failed.
    Executor(String),
}

impl fmt::Display for SupportError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Fidelity(eps) => write!(f, "minimal_support: fidelity {eps} is not a finite non-negative number"),
            Self::Shape { what, expected, found } => {
                write!(f, "minimal_support: {what} has shape {found:?}, expected {expected:?}")
            }
            Self::NonFinite { what } => write!(f, "minimal_support: {what} has a non-finite entry"),
            Self::StartInadmissible { position, divergence } => write!(
                f,
                "minimal_support: the starting supports already have KL {divergence:.3e} > ε at position {position}"
            ),
            Self::Sequence { positions, sequence } => {
                write!(f, "minimal_support: {positions} positions are not whole sequences of length {sequence}")
            }
            Self::Executor(e) => write!(f, "minimal_support: executor failed: {e}"),
        }
    }
}

impl std::error::Error for SupportError {}

/// One round of the search.
#[derive(Clone, Debug, PartialEq)]
pub struct SupportRound {
    /// Pieces proposed before any halving.
    pub proposed: usize,
    /// Pieces removed by the accepted proposal (0 when the round accepted nothing).
    pub removed: usize,
    /// Pieces restored by a repair round (0 in a removal round).
    pub restored: usize,
    /// Halvings before acceptance or exhaustion.
    pub halvings: usize,
    /// Kept pieces summed over positions after the round.
    pub kept: usize,
    /// Largest `KL_t` after the round.
    pub max_divergence: f64,
}

/// The supports at the search's fixed point.
#[derive(Clone, Debug)]
pub struct MinimalSupport {
    /// `P × C`, true where a piece is kept.
    pub keep: Array2<bool>,
    /// `KL_t` at `keep`.
    pub divergence: Array1<f64>,
    pub rounds: Vec<SupportRound>,
    /// Each position's trust radius at the end: a later search resumed with these
    /// continues at the learned proposal sizes instead of relearning them from one.
    pub radius: Vec<usize>,
    /// Each position's cheapest predicted removal among its kept pieces at the final
    /// supports (`+∞` where nothing is kept): a caller can tell whether slack it opens
    /// could buy any removal before paying for another search.
    pub cheapest: Array1<f64>,
}

/// Per position, the smallest predicted removal cost over its kept pieces.
fn cheapest_removals(keep: &Array2<bool>, cost: &Array2<f64>) -> Array1<f64> {
    Array1::from_iter(keep.outer_iter().zip(cost.outer_iter()).map(|(k, c)| {
        k.iter().zip(c.iter()).filter(|(k, _)| **k).map(|(_, v)| v.max(0.0)).fold(f64::INFINITY, f64::min)
    }))
}

fn checked(eval: SupportEvaluation, positions: usize, pieces: usize) -> Result<SupportEvaluation, SupportError> {
    if eval.divergence.len() != positions {
        return Err(SupportError::Shape {
            what: "divergence",
            expected: (positions, 1),
            found: (eval.divergence.len(), 1),
        });
    }
    if eval.removal_cost.dim() != (positions, pieces) {
        return Err(SupportError::Shape {
            what: "removal_cost",
            expected: (positions, pieces),
            found: eval.removal_cost.dim(),
        });
    }
    if !eval.divergence.iter().all(|v| v.is_finite()) {
        return Err(SupportError::NonFinite { what: "divergence" });
    }
    if !eval.removal_cost.iter().all(|v| v.is_finite()) {
        return Err(SupportError::NonFinite { what: "removal_cost" });
    }
    if eval.restore_gain.dim() != (positions, pieces) {
        return Err(SupportError::Shape {
            what: "restore_gain",
            expected: (positions, pieces),
            found: eval.restore_gain.dim(),
        });
    }
    if !eval.restore_gain.iter().all(|v| v.is_finite()) {
        return Err(SupportError::NonFinite { what: "restore_gain" });
    }
    Ok(eval)
}

/// The first `count` of `candidates` under the total order `before`, in that order: a
/// selection (`O(n)`) and a sort of the selected prefix only, which returns exactly the
/// prefix a full sort would.
fn first_in_order<F: Fn(usize, usize) -> std::cmp::Ordering>(mut candidates: Vec<usize>, count: usize, before: F) -> Vec<usize> {
    if count == 0 {
        return Vec::new();
    }
    if count < candidates.len() {
        candidates.select_nth_unstable_by(count - 1, |&a, &b| before(a, b));
        candidates.truncate(count);
    }
    candidates.sort_by(|&a, &b| before(a, b));
    candidates
}

/// Each position's proposal: its first `radius[t]` kept pieces in increasing predicted
/// cost (ties to the lower index).
fn propose(keep: ArrayView2<'_, bool>, cost: ArrayView2<'_, f64>, radius: &[usize]) -> Vec<Vec<usize>> {
    let (positions, pieces) = keep.dim();
    (0..positions)
        .map(|t| {
            let kept: Vec<usize> = (0..pieces).filter(|&c| keep[[t, c]]).collect();
            first_in_order(kept, radius[t], |a, b| cost[[t, a]].total_cmp(&cost[[t, b]]).then(a.cmp(&b)))
        })
        .collect()
}

/// The mean form's proposal: the longest prefix of all kept pieces in increasing
/// predicted cost (ties to the lower position, then piece) whose calibrated predicted
/// increase `scale · Σ ĉ` stays below the budget's exact `slack`, and at least the
/// cheapest piece, so the search ends only on an exact refusal.
fn within_slack(keep: ArrayView2<'_, bool>, cost: ArrayView2<'_, f64>, slack: f64, scale: f64) -> Vec<Vec<usize>> {
    let (positions, pieces) = keep.dim();
    let kept = || (0..positions).flat_map(move |t| (0..pieces).map(move |c| (t, c))).filter(move |&(t, c)| keep[[t, c]]);
    let cheapest = kept().min_by(|a, b| cost[[a.0, a.1]].total_cmp(&cost[[b.0, b.1]]).then(a.cmp(b)));
    // A piece whose own calibrated cost exceeds the slack can join a prefix only after
    // pieces of negative cost pay for it: only pieces that fit with every negative cost
    // spent are candidates.
    let negative: f64 = kept().map(|(t, c)| cost[[t, c]].min(0.0)).sum();
    let mut candidates: Vec<(usize, usize)> = kept().filter(|&(t, c)| scale * (cost[[t, c]] + negative) < slack).collect();
    candidates.sort_by(|a, b| cost[[a.0, a.1]].total_cmp(&cost[[b.0, b.1]]).then(a.cmp(b)));
    let mut taken = 0;
    let mut total = 0.0;
    for (k, &(t, c)) in candidates.iter().enumerate() {
        total += cost[[t, c]];
        if scale * total < slack {
            taken = k + 1;
        }
    }
    let mut proposal = vec![Vec::new(); positions];
    if taken == 0 {
        if let Some((t, c)) = cheapest {
            proposal[t].push(c);
        }
    }
    for &(t, c) in &candidates[..taken] {
        proposal[t].push(c);
    }
    proposal
}

/// The smallest admissible supports the round rule reaches (module docs).
pub fn minimal_support<E: SupportExecutor>(
    executor: &mut E,
    positions: usize,
    pieces: usize,
    fidelity: Fidelity,
    sequence: usize,
    start: Option<Array2<bool>>,
    start_radius: Option<Vec<usize>>,
) -> Result<MinimalSupport, SupportError> {
    let eps = fidelity.bound();
    if !(eps.is_finite() && eps >= 0.0) {
        return Err(SupportError::Fidelity(eps));
    }
    if sequence == 0 || positions % sequence != 0 {
        return Err(SupportError::Sequence { positions, sequence });
    }
    let mut keep = match start {
        Some(start) if start.dim() != (positions, pieces) => {
            return Err(SupportError::Shape { what: "start", expected: (positions, pieces), found: start.dim() });
        }
        Some(start) => start,
        None => Array2::from_elem((positions, pieces), true),
    };
    let mut current = checked(executor.evaluate(keep.view()).map_err(SupportError::Executor)?, positions, pieces)?;
    let mut rounds = Vec::new();
    // Repair: while the start is inadmissible, restore removed pieces in decreasing
    // predicted gain at every position in a violation's causal cone (all positions
    // under a shared budget), doubling each such position's count per repair round.
    // Every round restores at least one piece and the full support bounds the count,
    // so this ends; a full support that still violates is refused.
    let mut restore_radius = vec![1usize; positions];
    loop {
        let violators = fidelity.violators(&current.divergence);
        let Some(&first_violator) = violators.first() else { break };
        let mut repair = vec![false; positions];
        for &v in &violators {
            let first = v - v % sequence;
            for flag in &mut repair[first..=v] {
                *flag = true;
            }
        }
        let mut restored = 0usize;
        for t in (0..positions).filter(|&t| repair[t]) {
            let removed: Vec<usize> = (0..pieces).filter(|&c| !keep[[t, c]]).collect();
            let gain = &current.restore_gain;
            let chosen = first_in_order(removed, restore_radius[t], |a, b| {
                gain[[t, b]].total_cmp(&gain[[t, a]]).then(a.cmp(&b))
            });
            for &c in &chosen {
                keep[[t, c]] = true;
                restored += 1;
            }
            restore_radius[t] *= 2;
        }
        if restored == 0 {
            return Err(SupportError::StartInadmissible {
                position: first_violator,
                divergence: current.divergence[first_violator],
            });
        }
        current = checked(executor.evaluate(keep.view()).map_err(SupportError::Executor)?, positions, pieces)?;
        rounds.push(SupportRound {
            proposed: 0,
            removed: 0,
            restored,
            halvings: 0,
            kept: keep.iter().filter(|&&k| k).count(),
            max_divergence: current.divergence.iter().copied().fold(0.0, f64::max),
        });
    }
    let mut radius = match start_radius {
        Some(r) if r.len() != positions => {
            return Err(SupportError::Shape { what: "start_radius", expected: (positions, 1), found: (r.len(), 1) });
        }
        Some(r) => r.into_iter().map(|v| v.max(1)).collect(),
        None => vec![1usize; positions],
    };
    // Under the mean form: the ratio of the exact increase to the predicted one, measured
    // on the latest exact evaluation of a proposal (module docs, step 2).
    let mut scale = 1.0;
    loop {
        let mut proposal = match fidelity {
            Fidelity::Mean(eps) => {
                let slack = positions as f64 * eps - current.divergence.sum();
                within_slack(keep.view(), current.removal_cost.view(), slack, scale)
            }
            Fidelity::PerPosition(_) => propose(keep.view(), current.removal_cost.view(), &radius),
        };
        let offered: Vec<usize> = proposal.iter().map(Vec::len).collect();
        let proposed: usize = proposal.iter().map(Vec::len).sum();
        if proposed == 0 {
            {
                let cheapest = cheapest_removals(&keep, &current.removal_cost);
                return Ok(MinimalSupport { keep, divergence: current.divergence, rounds, radius, cheapest });
            }
        }
        let mut halvings = 0;
        let accepted = loop {
            let count: usize = proposal.iter().map(Vec::len).sum();
            if count == 0 {
                break None;
            }
            let mut trial = keep.clone();
            for (t, removal) in proposal.iter().enumerate() {
                for &c in removal {
                    trial[[t, c]] = false;
                }
            }
            let evaluation = checked(executor.evaluate(trial.view()).map_err(SupportError::Executor)?, positions, pieces)?;
            if let Fidelity::Mean(_) = fidelity {
                let predicted: f64 = proposal.iter().enumerate().flat_map(|(t, r)| r.iter().map(move |&c| (t, c))).map(|(t, c)| current.removal_cost[[t, c]]).sum();
                let actual = evaluation.divergence.sum() - current.divergence.sum();
                if predicted > 0.0 && actual > 0.0 {
                    scale = actual / predicted;
                }
            }
            let violators = fidelity.violators(&evaluation.divergence);
            if violators.is_empty() {
                break Some((trial, evaluation, count));
            }
            halvings += 1;
            if let Fidelity::Mean(_) = fidelity {
                // One shared budget: keep the cheapest half of all proposed removals,
                // wherever they are, and drop the costliest half.
                let mut all: Vec<(usize, usize)> =
                    proposal.iter().enumerate().flat_map(|(t, r)| r.iter().map(move |&c| (t, c))).collect();
                all.sort_by(|a, b| {
                    current.removal_cost[[a.0, a.1]].total_cmp(&current.removal_cost[[b.0, b.1]]).then(a.cmp(b))
                });
                let keep_count = all.len() / 2;
                let mut kept_pairs = vec![Vec::new(); positions];
                for &(t, c) in &all[..keep_count] {
                    kept_pairs[t].push(c);
                }
                if all.is_empty() {
                    break None;
                }
                proposal = kept_pairs;
                continue;
            }
            let mut halve = vec![false; positions];
            for &v in &violators {
                let first = v - v % sequence;
                for flag in &mut halve[first..=v] {
                    *flag = true;
                }
            }
            // A violation nothing upstream can explain (the executor's own nondeterminism at
            // the boundary) leaves nothing to halve: the round accepts nothing.
            if !halve.iter().zip(&proposal).any(|(&flag, removal)| flag && !removal.is_empty()) {
                break None;
            }
            for (removal, flag) in proposal.iter_mut().zip(halve) {
                if flag {
                    removal.truncate(removal.len() / 2);
                }
            }
        };
        match accepted {
            Some((trial, evaluation, removed)) => {
                keep = trial;
                current = evaluation;
                for t in 0..positions {
                    radius[t] = if proposal[t].len() == offered[t] { 2 * offered[t].max(1) } else { proposal[t].len().max(1) };
                }
                rounds.push(SupportRound {
                    proposed,
                    removed,
                    restored: 0,
                    halvings,
                    kept: keep.iter().filter(|&&k| k).count(),
                    max_divergence: current.divergence.iter().copied().fold(0.0, f64::max),
                });
            }
            // Under the mean form every refused round ended on the single cheapest piece,
            // exactly: nothing smaller is left to try.
            None if matches!(fidelity, Fidelity::PerPosition(_)) && radius.iter().any(|&r| r > 1) => {
                // Nothing was admissible at these sizes: shrink every radius and retry.
                for r in &mut radius {
                    *r = (*r / 2).max(1);
                }
                rounds.push(SupportRound {
                    proposed,
                    removed: 0,
                    restored: 0,
                    halvings,
                    kept: keep.iter().filter(|&&k| k).count(),
                    max_divergence: current.divergence.iter().copied().fold(0.0, f64::max),
                });
            }
            None => {
                rounds.push(SupportRound {
                    proposed,
                    removed: 0,
                    restored: 0,
                    halvings,
                    kept: keep.iter().filter(|&&k| k).count(),
                    max_divergence: current.divergence.iter().copied().fold(0.0, f64::max),
                });
                {
                let cheapest = cheapest_removals(&keep, &current.removal_cost);
                return Ok(MinimalSupport { keep, divergence: current.divergence, rounds, radius, cheapest });
            }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A linear-Gaussian teacher per position: `KL_t(keep) = ½ Σ_{c removed} a_tc²`, with
    /// the exact removal cost as the prediction.
    struct Quadratic {
        weight: Array2<f64>,
        evaluations: usize,
    }

    impl SupportExecutor for Quadratic {
        fn evaluate(&mut self, keep: ArrayView2<'_, bool>) -> Result<SupportEvaluation, String> {
            self.evaluations += 1;
            let divergence = Array1::from_iter(keep.outer_iter().zip(self.weight.outer_iter()).map(|(k, w)| {
                0.5 * k.iter().zip(w.iter()).filter(|(k, _)| !**k).map(|(_, a)| a * a).sum::<f64>()
            }));
            Ok(SupportEvaluation {
                divergence,
                removal_cost: self.weight.mapv(|a| 0.5 * a * a),
                restore_gain: self.weight.mapv(|a| 0.5 * a * a),
            })
        }
    }

    /// With an exact additive divergence the smallest admissible support drops the
    /// cheapest pieces until the next one would exceed ε, and the search reaches it.
    #[test]
    fn additive_divergence_reaches_the_greedy_optimum() {
        let weight = ndarray::array![[3.0, 0.1, 0.2, 2.0, 0.05], [0.01, 0.02, 5.0, 0.03, 1.0]];
        let mut executor = Quadratic { weight: weight.clone(), evaluations: 0 };
        let eps = 0.1;
        let result = minimal_support(&mut executor, 2, 5, Fidelity::PerPosition(eps), 1, None, None).unwrap();
        // Position 0 drops 0.05, 0.1, 0.2 (½ Σ a² = 0.02625); 2.0 would add 2.0.
        assert_eq!(result.keep.row(0).to_vec(), vec![true, false, false, true, false]);
        // Position 1 drops 0.01, 0.02, 0.03; 1.0 would add 0.5.
        assert_eq!(result.keep.row(1).to_vec(), vec![false, false, true, false, true]);
        assert!(result.divergence.iter().all(|&v| v <= eps));
    }

    /// A prediction that underestimates is corrected by exact acceptance: the result is
    /// admissible although every prediction is zero.
    #[test]
    fn a_wrong_prediction_never_costs_fidelity() {
        struct Blind(Quadratic);
        impl SupportExecutor for Blind {
            fn evaluate(&mut self, keep: ArrayView2<'_, bool>) -> Result<SupportEvaluation, String> {
                let mut e = self.0.evaluate(keep)?;
                e.removal_cost.fill(0.0);
                Ok(e)
            }
        }
        let weight = ndarray::array![[3.0, 0.1, 0.2, 2.0, 0.05, 0.3, 0.02, 1.5]];
        let mut executor = Blind(Quadratic { weight, evaluations: 0 });
        let result = minimal_support(&mut executor, 1, 8, Fidelity::PerPosition(0.1), 1, None, None).unwrap();
        assert!(result.divergence[0] <= 0.1, "{:?}", result.divergence);
        assert!(result.rounds.iter().any(|r| r.halvings > 0));
        for w in result.rounds.windows(2) {
            assert!(w[1].kept <= w[0].kept);
        }
    }

    /// One position whose predictions are useless does not stop the others: its
    /// violations halve only its own proposal (independent positions), so position 1
    /// still reaches its optimum.
    #[test]
    fn a_bad_position_does_not_block_the_others() {
        struct HalfBlind(Quadratic);
        impl SupportExecutor for HalfBlind {
            fn evaluate(&mut self, keep: ArrayView2<'_, bool>) -> Result<SupportEvaluation, String> {
                let mut e = self.0.evaluate(keep)?;
                e.removal_cost.row_mut(0).fill(0.0);
                Ok(e)
            }
        }
        let weight = ndarray::array![[3.0, 2.5, 4.0, 2.0, 3.5], [0.01, 0.02, 5.0, 0.03, 1.0]];
        let mut executor = HalfBlind(Quadratic { weight, evaluations: 0 });
        let result = minimal_support(&mut executor, 2, 5, Fidelity::PerPosition(0.1), 1, None, None).unwrap();
        assert_eq!(result.keep.row(1).to_vec(), vec![false, false, true, false, true]);
        assert!(result.divergence.iter().all(|&v| v <= 0.1));
    }

    /// A causal sequence: removals at position 0 raise position 1's divergence too. A
    /// violation at position 1 whose own proposal is empty halves position 0's.
    #[test]
    fn an_upstream_violation_halves_the_upstream_proposal() {
        struct Causal;
        impl SupportExecutor for Causal {
            fn evaluate(&mut self, keep: ArrayView2<'_, bool>) -> Result<SupportEvaluation, String> {
                let removed0 = keep.row(0).iter().filter(|k| !**k).count() as f64;
                let removed1 = keep.row(1).iter().filter(|k| !**k).count() as f64;
                let divergence = ndarray::array![0.01 * removed0, 0.04 * removed0 + 0.01 * removed1];
                Ok(SupportEvaluation {
                    divergence,
                    removal_cost: Array2::from_elem(keep.dim(), 0.01),
                    restore_gain: Array2::from_elem(keep.dim(), 0.01),
                })
            }
        }
        let result = minimal_support(&mut Causal, 2, 4, Fidelity::PerPosition(0.1), 2, None, None).unwrap();
        assert!(result.divergence.iter().all(|&v| v <= 0.1 + 1e-12), "{:?}", result.divergence);
        assert!(result.keep.iter().any(|k| !*k), "nothing was removed");
    }

    /// An empty start is inadmissible: repair restores the largest-gain pieces until the
    /// supports are admissible, and the removal rounds then reach the same optimum as
    /// from the full support.
    #[test]
    fn an_empty_start_is_repaired_to_the_same_optimum() {
        let weight = ndarray::array![[3.0, 0.1, 0.2, 2.0, 0.05], [0.01, 0.02, 5.0, 0.03, 1.0]];
        let empty = Array2::from_elem((2, 5), false);
        let result = minimal_support(&mut Quadratic { weight, evaluations: 0 }, 2, 5, Fidelity::PerPosition(0.1), 1, Some(empty), None).unwrap();
        assert!(result.rounds.iter().any(|r| r.restored > 0));
        assert_eq!(result.keep.row(0).to_vec(), vec![true, false, false, true, false]);
        assert_eq!(result.keep.row(1).to_vec(), vec![false, false, true, false, true]);
    }

    /// A shared budget spends slack where removal is cheap: under the mean form, the
    /// cheap position removes more than its own per-position bound would allow, and the
    /// mean still meets ε.
    #[test]
    fn a_mean_budget_is_shared_across_positions() {
        let weight = ndarray::array![[0.3, 0.3, 0.3, 0.3], [2.0, 2.0, 2.0, 2.0]];
        let per = minimal_support(&mut Quadratic { weight: weight.clone(), evaluations: 0 }, 2, 4, Fidelity::PerPosition(0.1), 1, None, None).unwrap();
        let mean = minimal_support(&mut Quadratic { weight, evaluations: 0 }, 2, 4, Fidelity::Mean(0.1), 1, None, None).unwrap();
        let removed = |k: &Array2<bool>| k.iter().filter(|x| !**x).count();
        assert!(mean.divergence.iter().sum::<f64>() / 2.0 <= 0.1);
        assert!(removed(&mean.keep) >= removed(&per.keep), "{:?} vs {:?}", mean.keep, per.keep);
    }

    /// With an exact additive prediction the mean form's first proposal is the greedy
    /// optimum (the cheapest removals whose summed cost fits the slack), accepted at once;
    /// the next round's single cheapest removal is refused and the search ends: three
    /// exact evaluations in all.
    #[test]
    fn the_mean_proposal_is_sized_by_the_exact_slack() {
        // Costs ½a²: .005 .02 .045 .5 | .00125 .01125 2.0 .03125; budget 2 × 0.1 = 0.2.
        let weight = ndarray::array![[0.1, 0.2, 0.3, 1.0], [0.05, 0.15, 2.0, 0.25]];
        let mut executor = Quadratic { weight, evaluations: 0 };
        let result = minimal_support(&mut executor, 2, 4, Fidelity::Mean(0.1), 1, None, None).unwrap();
        assert_eq!(result.keep.row(0).to_vec(), vec![false, false, false, true]);
        assert_eq!(result.keep.row(1).to_vec(), vec![false, false, true, false]);
        assert!(result.divergence.sum() / 2.0 < 0.1);
        assert_eq!(executor.evaluations, 3);
    }

    /// A prediction that sees nothing (every cost zero) proposes every kept piece; exact
    /// acceptance halves it back, so the mean form still ends, admissible. (Ties fall to
    /// the lowest index, whose removal alone breaks this budget, so the fixed point it
    /// reaches is the full support: a blind prediction may cost pieces, never fidelity.)
    #[test]
    fn a_blind_prediction_never_costs_the_mean_budget() {
        struct Blind(Quadratic);
        impl SupportExecutor for Blind {
            fn evaluate(&mut self, keep: ArrayView2<'_, bool>) -> Result<SupportEvaluation, String> {
                let mut e = self.0.evaluate(keep)?;
                e.removal_cost.fill(0.0);
                Ok(e)
            }
        }
        let weight = ndarray::array![[3.0, 0.1, 0.2, 2.0, 0.05, 0.3, 0.02, 1.5], [0.4, 0.01, 1.0, 0.2, 0.03, 2.5, 0.1, 0.6]];
        let mut executor = Blind(Quadratic { weight, evaluations: 0 });
        let result = minimal_support(&mut executor, 2, 8, Fidelity::Mean(0.1), 1, None, None).unwrap();
        assert!(result.divergence.sum() / 2.0 < 0.1, "{:?}", result.divergence);
        // one round, halving 16 proposed removals down to none: the start and five trials
        assert_eq!(executor.0.evaluations, 6);
    }

    /// The selection returns exactly the prefix a full sort under the same total
    /// order returns, ties included.
    #[test]
    fn a_selected_prefix_equals_the_sorted_prefix() {
        let cost: [f64; 8] = [0.5, 0.1, 0.5, 0.3, 0.1, 0.9, 0.0, 0.3];
        let order = |a: usize, b: usize| cost[a].total_cmp(&cost[b]).then(a.cmp(&b));
        let mut sorted: Vec<usize> = (0..cost.len()).collect();
        sorted.sort_by(|&a, &b| order(a, b));
        for count in 0..=cost.len() + 1 {
            let expected: Vec<usize> = sorted.iter().copied().take(count).collect();
            assert_eq!(first_in_order((0..cost.len()).rev().collect(), count, order), expected, "count {count}");
        }
    }

    /// A warm start continues from the given supports and never restores a piece the
    /// start removed.
    #[test]
    fn a_warm_start_only_removes_more() {
        let weight = ndarray::array![[3.0, 0.1, 0.2, 2.0, 0.05]];
        let mut executor = Quadratic { weight, evaluations: 0 };
        let start = ndarray::array![[true, true, false, true, true]];
        let result = minimal_support(&mut executor, 1, 5, Fidelity::PerPosition(0.1), 1, Some(start.clone()), None).unwrap();
        for (k, s) in result.keep.iter().zip(start.iter()) {
            assert!(!*k || *s, "a piece the start removed came back");
        }
        assert_eq!(result.keep.row(0).to_vec(), vec![true, false, false, true, false]);
    }

    #[test]
    fn an_inadmissible_full_support_is_refused() {
        struct Broken;
        impl SupportExecutor for Broken {
            fn evaluate(&mut self, keep: ArrayView2<'_, bool>) -> Result<SupportEvaluation, String> {
                let (p, c) = keep.dim();
                Ok(SupportEvaluation {
                    divergence: Array1::from_elem(p, 1.0),
                    removal_cost: Array2::zeros((p, c)),
                    restore_gain: Array2::zeros((p, c)),
                })
            }
        }
        assert!(matches!(
            minimal_support(&mut Broken, 2, 3, Fidelity::PerPosition(0.5), 1, None, None),
            Err(SupportError::StartInadmissible { position: 0, .. })
        ));
    }
}
