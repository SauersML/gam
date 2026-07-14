//! Shared priority-ordered candidate selection.
//!
//! Topology ranking, model comparison, and seed screening all rank candidates by
//! a lower-is-better scalar score with deterministic tie handling. Keeping that
//! ordering contract here prevents the three call sites from drifting on score
//! direction, finite-score filtering, or original-order tie breaks.

#[derive(Clone, Debug)]
pub(crate) struct PriorityCandidate<T> {
    pub item: T,
    pub original_index: usize,
    pub score: f64,
    pub tie_break: usize,
}

impl<T> PriorityCandidate<T> {
    pub(crate) fn new(item: T, original_index: usize, score: f64, tie_break: usize) -> Self {
        Self {
            item,
            original_index,
            score,
            tie_break,
        }
    }
}

pub(crate) fn rank_priority_candidates<T>(
    mut candidates: Vec<PriorityCandidate<T>>,
) -> Vec<PriorityCandidate<T>> {
    candidates.sort_by(|lhs, rhs| {
        lhs.score
            .total_cmp(&rhs.score)
            .then_with(|| lhs.tie_break.cmp(&rhs.tie_break))
            .then_with(|| lhs.original_index.cmp(&rhs.original_index))
    });
    candidates
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct PriorityBudgetStage {
    pub cap: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct PriorityStageSummary {
    pub stage: usize,
    pub cap: usize,
    pub ranked: usize,
    pub rejected: usize,
}

#[derive(Clone, Debug)]
pub(crate) struct PriorityCascadeResult {
    pub ranked_indices: Vec<usize>,
    pub rejected: usize,
    pub final_cap: usize,
    pub stages_consumed: usize,
}

pub(crate) fn rank_indices_with_budget_cascade<E>(
    item_count: usize,
    stages: &[PriorityBudgetStage],
    mut evaluate: impl FnMut(usize, usize, usize) -> Result<f64, E>,
    mut on_stage_complete: impl FnMut(PriorityStageSummary),
) -> Result<PriorityCascadeResult, E> {
    assert!(
        !stages.is_empty(),
        "priority-selection cascade requires at least one budget stage"
    );
    let mut ranked = Vec::<PriorityCandidate<usize>>::with_capacity(item_count);
    let mut rejected = 0usize;
    let mut final_cap = stages[0].cap;
    let mut stages_consumed = 0usize;

    for (stage, budget) in stages.iter().enumerate() {
        ranked.clear();
        rejected = 0;
        for idx in 0..item_count {
            match evaluate(stage, budget.cap, idx) {
                Ok(score) if score.is_finite() => {
                    ranked.push(PriorityCandidate::new(idx, idx, score, 0));
                }
                Ok(_) => {
                    rejected += 1;
                }
                Err(error) => return Err(error),
            }
        }
        final_cap = budget.cap;
        stages_consumed = stage + 1;
        on_stage_complete(PriorityStageSummary {
            stage,
            cap: budget.cap,
            ranked: ranked.len(),
            rejected,
        });
        if !ranked.is_empty() {
            break;
        }
    }

    let ranked_indices = rank_priority_candidates(ranked)
        .into_iter()
        .map(|row| row.item)
        .collect();
    Ok(PriorityCascadeResult {
        ranked_indices,
        rejected,
        final_cap,
        stages_consumed,
    })
}

#[cfg(test)]
mod tests {
    use super::{
        PriorityBudgetStage, PriorityCandidate, rank_indices_with_budget_cascade,
        rank_priority_candidates,
    };

    #[test]
    fn rank_priority_candidates_uses_score_tie_break_then_original_order() {
        let ranked = rank_priority_candidates(vec![
            PriorityCandidate::new("late", 3, 1.0, 1),
            PriorityCandidate::new("winner", 2, 0.5, 9),
            PriorityCandidate::new("early", 1, 1.0, 1),
            PriorityCandidate::new("simple", 0, 1.0, 0),
        ]);
        let names: Vec<_> = ranked.into_iter().map(|row| row.item).collect();
        assert_eq!(names, vec!["winner", "simple", "early", "late"]);
    }

    #[test]
    fn budget_cascade_escalates_until_a_finite_stage_and_sorts_scores() {
        let stages = [
            PriorityBudgetStage { cap: 2 },
            PriorityBudgetStage { cap: 8 },
            PriorityBudgetStage { cap: 0 },
        ];
        let mut seen = Vec::new();
        let out = rank_indices_with_budget_cascade(
            3,
            &stages,
            |_, cap, idx| -> Result<f64, ()> {
                if cap < 8 {
                    Ok(f64::NAN)
                } else {
                    Ok([3.0, 1.0, 2.0][idx])
                }
            },
            |summary| seen.push((summary.stage, summary.cap, summary.ranked)),
        )
        .expect("the fixture evaluator is infallible");
        assert_eq!(seen, vec![(0, 2, 0), (1, 8, 3)]);
        assert_eq!(out.final_cap, 8);
        assert_eq!(out.stages_consumed, 2);
        assert_eq!(out.ranked_indices, vec![1, 2, 0]);
    }
}
