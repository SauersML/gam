//! REML model comparison: ranking, deltas, Bayes-factor summary.
//!
//! Pure-Rust core consumed by the PyO3 wrapper in `crates/gam-pyffi`
//! (`compare_reml_fits`). Callers marshal their fits into [`RemlCandidate`]
//! and receive a fully-typed [`RemlComparison`] back; only presentation-layer
//! formatting differs per surface.
//!
//! Sign convention: `RemlCandidate::score` is the same minimised cost the
//! REML/LAML optimiser stores on each fit (penalised negative log marginal
//! likelihood + Laplace correction), so **lower is better**. The comparator
//! sorts ascending; the Bayes factor in favour of the winner over a row is
//! `exp(row.score − best_score) ≥ 1`. Pre-fix this module sorted descending
//! and treated the cost as a log-evidence, so the worst model won and every
//! Bayes factor was inverted — see issue #396.

use std::cmp::Ordering;

/// One candidate fit in a REML model comparison.
#[derive(Clone, Debug)]
pub struct RemlCandidate {
    /// Caller's original index into the fits array — preserved through
    /// sorting so e.g. external CV-score arrays stay aligned.
    pub index: usize,
    pub name: String,
    /// Minimised REML / LAML cost (penalised negative log marginal
    /// likelihood; lower is better). Bayes factors in favour of the
    /// winner over this row are `exp(row.score − best_score)`.
    pub score: f64,
    pub edf: Option<f64>,
}

/// Result of comparing `Vec<RemlCandidate>` — ranking sorted best-first.
#[derive(Clone, Debug)]
pub struct RemlComparison {
    pub ranking: Vec<RankedRow>,
    pub winner: String,
    pub evidence_summary: String,
    pub score_table: Vec<ScoreRow>,
}

#[derive(Clone, Debug)]
pub struct RankedRow {
    pub name: String,
    pub score: f64,
    /// Cost gap from the winning (lowest-score) model: `score - best_score`,
    /// i.e. the log Bayes factor in favour of the winner over this row.
    pub delta: f64,
    /// Bayes factor in favor of the winning model over this row.
    pub bayes_factor: f64,
    pub edf: Option<f64>,
}

#[derive(Clone, Debug)]
pub struct ScoreRow {
    pub name: String,
    pub reml_score: f64,
    /// Cost gap from the winning model: `reml_score - best_score`.
    pub delta_reml: f64,
    /// Bayes factor in favor of the winning model over this row.
    pub bayes_factor_best_over_model: f64,
    pub effective_dof: Option<f64>,
}

pub fn compare_reml_fits(mut candidates: Vec<RemlCandidate>) -> Result<RemlComparison, String> {
    if candidates.is_empty() {
        return Err("compare_models requires at least one fit".to_string());
    }

    // Lowest-cost model wins: `RemlCandidate::score` is the optimiser's
    // minimised cost (issue #396 was the wrong direction here).
    candidates.sort_by(|left, right| {
        left.score
            .partial_cmp(&right.score)
            .unwrap_or(Ordering::Equal)
    });

    let best_score = candidates[0].score;
    let winner = candidates[0].name.clone();
    let mut ranking = Vec::with_capacity(candidates.len());
    let mut score_table = Vec::with_capacity(candidates.len());

    for row in candidates.iter() {
        // `delta = row.score - best_score ≥ 0` is the cost gap from the
        // winner. Under the Laplace approximation that gap equals
        // log(P(D|winner)/P(D|row)) up to common normalising constants, so
        // its exponential is the Bayes factor in favour of the winner.
        let delta = row.score - best_score;
        let bayes_factor = delta.exp();
        ranking.push(RankedRow {
            name: row.name.clone(),
            score: row.score,
            delta,
            bayes_factor,
            edf: row.edf,
        });
        score_table.push(ScoreRow {
            name: row.name.clone(),
            reml_score: row.score,
            delta_reml: delta,
            bayes_factor_best_over_model: bayes_factor,
            effective_dof: row.edf,
        });
    }

    let evidence_summary = if candidates.len() >= 2 {
        let runner_up = &candidates[1];
        let best_log_bayes_factor_vs_runner_up = runner_up.score - best_score;
        format!(
            "{} wins by Bayes factor {} over {}",
            winner,
            format_bayes_factor(best_log_bayes_factor_vs_runner_up),
            runner_up.name
        )
    } else {
        format!("{winner} (single fit; no comparison)")
    };

    Ok(RemlComparison {
        ranking,
        winner,
        evidence_summary,
        score_table,
    })
}

/// Format a log-Bayes-factor as a presentation-ready string. Large
/// magnitudes (≥ ln(10)·3) get scientific notation; small ones get
/// three significant figures on the linear scale.
pub fn format_bayes_factor(log_bf: f64) -> String {
    if !log_bf.is_finite() {
        return "inf".to_string();
    }
    if log_bf.abs() >= std::f64::consts::LN_10 * 3.0 {
        return format!("1e{:+.1}", log_bf / std::f64::consts::LN_10);
    }
    format_three_significant(log_bf.exp())
}

pub fn format_three_significant(value: f64) -> String {
    if value == 0.0 {
        return "0".to_string();
    }
    if !value.is_finite() {
        return format!("{value}");
    }
    let abs = value.abs();
    // Choose decimal places so we emit exactly three significant digits.
    // For |v| in [1, 10) we keep 2 decimals (e.g. "1.23"); for [10, 100)
    // we keep 1 decimal (e.g. "12.3"); for [100, 1000) we keep 0 decimals
    // (e.g. "123"). Above 1000 the integer width itself already exceeds
    // three sig figs, so we fall back to scientific notation so we don't
    // silently emit four-or-more-significant-digit integers like "1234".
    let exponent = abs.log10().floor() as i32;
    if exponent >= 3 {
        // |v| >= 1000 — use scientific with two fractional digits (3 sig figs).
        return format!("{value:.2e}");
    }
    let decimals = (2 - exponent).max(0) as usize;
    // Round half-away-from-zero in *decimal* (not in IEEE binary) so that
    // values such as 1.005 round to "1.01" rather than "1.00". The naive
    // `{value:.decimals$}` uses banker's rounding on the binary
    // representation, and 1.005 is actually 1.00499999... in IEEE-754,
    // which yields the wrong-looking "1.00".
    let scale = 10f64.powi(decimals as i32);
    let rounded = (value * scale).abs().round() / scale * value.signum();
    let formatted = format!("{rounded:.decimals$}");
    // Do NOT strip trailing zeros: "1.00" and "10.0" are intentional and
    // carry the three-significant-digit precision contract. Stripping them
    // would silently turn "10.0" into "10" (two sig figs) and "1.00" into
    // "1" (one sig fig).
    formatted
}

#[cfg(test)]
mod tests {
    use super::{RemlCandidate, compare_reml_fits};

    /// The optimiser stores a minimised REML cost on each fit; the lowest
    /// cost should win, with `delta = row.score - best_score ≥ 0` and the
    /// Bayes factor `exp(delta)` reading "in favour of the winner over this
    /// row". This locks the post-issue-#396 sign convention.
    #[test]
    fn ranks_lowest_reml_cost_as_winner() {
        let comparison = compare_reml_fits(vec![
            RemlCandidate {
                index: 0,
                name: "low_cost".to_string(),
                score: 2.0,
                edf: None,
            },
            RemlCandidate {
                index: 1,
                name: "high_cost".to_string(),
                score: 5.0,
                edf: Some(4.0),
            },
        ])
        .expect("finite REML candidates should compare");

        assert_eq!(comparison.winner, "low_cost");
        assert_eq!(comparison.ranking[0].name, "low_cost");
        assert_eq!(comparison.ranking[0].delta, 0.0);
        assert_eq!(comparison.ranking[0].bayes_factor, 1.0);
        assert_eq!(comparison.ranking[1].name, "high_cost");
        assert_eq!(comparison.ranking[1].delta, 3.0);
        assert!((comparison.ranking[1].bayes_factor - 3.0_f64.exp()).abs() < 1e-12);
        assert_eq!(comparison.score_table[1].name, "high_cost");
        assert_eq!(comparison.score_table[1].reml_score, 5.0);
        assert_eq!(comparison.score_table[1].delta_reml, 3.0);
        assert!(
            (comparison.score_table[1].bayes_factor_best_over_model - 3.0_f64.exp()).abs() < 1e-12
        );
        assert!(comparison.evidence_summary.contains("low_cost"));
    }

    /// Regression for issue #396: pre-fix this exact configuration declared
    /// `m1_xonly` (the WORSE model) the winner because the comparator sorted
    /// REML costs descending. The minimised-cost convention now ranks
    /// `m2_true` first, with the score-table fields preserving the raw
    /// per-model REML cost for downstream reporting.
    #[test]
    fn issue_396_lower_reml_cost_wins_against_higher_cost() {
        let comparison = compare_reml_fits(vec![
            RemlCandidate {
                index: 0,
                name: "m1_xonly".to_string(),
                score: 786.32,
                edf: Some(9.49),
            },
            RemlCandidate {
                index: 1,
                name: "m2_true".to_string(),
                score: 359.06,
                edf: Some(18.58),
            },
        ])
        .expect("finite REML candidates should compare");

        assert_eq!(
            comparison.winner, "m2_true",
            "issue #396: the lower-cost (better) model must win"
        );
        // Winner row carries the raw cost it was supplied, plus a zero gap.
        let winner_row = &comparison.score_table[0];
        assert_eq!(winner_row.name, "m2_true");
        assert!((winner_row.reml_score - 359.06).abs() < 1.0e-12);
        assert_eq!(winner_row.delta_reml, 0.0);
        assert_eq!(winner_row.bayes_factor_best_over_model, 1.0);
        // Loser row reports the cost gap (here 427.26 nats) and a Bayes
        // factor in favour of the winner of ~exp(427), reported on a log
        // scale by `format_bayes_factor` because it exceeds the threshold.
        let loser_row = &comparison.score_table[1];
        assert_eq!(loser_row.name, "m1_xonly");
        assert!((loser_row.reml_score - 786.32).abs() < 1.0e-12);
        let expected_delta = 786.32 - 359.06;
        assert!((loser_row.delta_reml - expected_delta).abs() < 1.0e-12);
        // The Bayes factor is `exp(delta) ≈ exp(427)` — overflows
        // `f64::MAX ≈ 1.8e308`, so the field stores `+∞`; the test simply
        // checks that the direction (huge in favour of winner) is right.
        assert!(loser_row.bayes_factor_best_over_model > 1.0e+100);
        assert!(comparison.evidence_summary.starts_with("m2_true wins"));
    }
}
