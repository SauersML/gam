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

#[cfg(test)]
mod tests {
    use super::{PriorityCandidate, rank_priority_candidates};

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
}
