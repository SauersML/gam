use std::cmp::Ordering;

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct PriorityRanking {
    pub ranked_indices: Vec<usize>,
    pub rejected_indices: Vec<usize>,
}

pub(crate) fn rank_min_finite_scores<I>(scores: I) -> PriorityRanking
where
    I: IntoIterator<Item = (usize, f64)>,
{
    let mut ranked = Vec::new();
    let mut rejected = Vec::new();
    for (index, score) in scores {
        if score.is_finite() {
            ranked.push((index, score));
        } else {
            rejected.push(index);
        }
    }
    ranked.sort_by(|(idx_a, score_a), (idx_b, score_b)| {
        compare_min_finite_scores(*score_a, *score_b, || idx_a.cmp(idx_b))
    });
    PriorityRanking {
        ranked_indices: ranked.into_iter().map(|(idx, _)| idx).collect(),
        rejected_indices: rejected,
    }
}

pub(crate) fn compare_min_finite_scores(
    lhs: f64,
    rhs: f64,
    tie_break: impl FnOnce() -> Ordering,
) -> Ordering {
    match (lhs.is_finite(), rhs.is_finite()) {
        (true, true) => lhs.total_cmp(&rhs).then_with(tie_break),
        (true, false) => Ordering::Less,
        (false, true) => Ordering::Greater,
        (false, false) => tie_break(),
    }
}

pub(crate) fn candidate_improves_min_score(
    candidate_score: f64,
    candidate_converged: bool,
    best_score: f64,
    best_converged: bool,
) -> bool {
    if candidate_converged != best_converged {
        return candidate_converged;
    }
    compare_min_finite_scores(candidate_score, best_score, || Ordering::Equal) == Ordering::Less
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rank_min_finite_scores_rejects_nonfinite_and_tie_breaks_by_index() {
        let ranking = rank_min_finite_scores([(2, 1.0), (0, f64::NAN), (1, 1.0), (3, -2.0)]);
        assert_eq!(ranking.ranked_indices, vec![3, 1, 2]);
        assert_eq!(ranking.rejected_indices, vec![0]);
    }

    #[test]
    fn converged_candidate_beats_lower_nonconverged_cost() {
        assert!(candidate_improves_min_score(10.0, true, 1.0, false));
        assert!(!candidate_improves_min_score(1.0, false, 10.0, true));
        assert!(candidate_improves_min_score(1.0, true, 2.0, true));
    }
}
