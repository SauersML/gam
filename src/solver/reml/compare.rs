use crate::solver::evidence::TopologyScoreScale;
use std::cmp::Ordering;

#[derive(Clone, Debug)]
pub struct RemlCandidate {
    pub index: usize,
    pub name: String,
    pub raw_reml: f64,
    pub null_dim: Option<f64>,
    pub null_space_logdet: Option<f64>,
    pub effective_dof: Option<f64>,
}

#[derive(Clone, Debug)]
pub struct CompareReport {
    pub ranking: Vec<RankedRow>,
    pub winner: String,
    pub evidence_summary: String,
    pub score_table: Vec<ScoreRow>,
}

#[derive(Clone, Debug)]
pub struct RankedRow {
    pub index: usize,
    pub name: String,
    pub score: f64,
    pub delta: f64,
    pub bayes_factor: f64,
    pub effective_dof: Option<f64>,
}

#[derive(Clone, Debug)]
pub struct ScoreRow {
    pub name: String,
    pub reml_score: f64,
    pub delta_reml: f64,
    pub bayes_factor_vs_best: f64,
    pub effective_dof: Option<f64>,
}

pub fn compare_fits(candidates: Vec<RemlCandidate>) -> Result<CompareReport, String> {
    if candidates.is_empty() {
        return Err("compare_models requires at least one fit".to_string());
    }

    let mut scored = Vec::with_capacity(candidates.len());
    for candidate in candidates {
        let score = normalized_reml_score(&candidate)?;
        scored.push((candidate, score));
    }
    scored.sort_by(|left, right| right.1.partial_cmp(&left.1).unwrap_or(Ordering::Equal));

    let best_score = scored[0].1;
    let winner = scored[0].0.name.clone();
    let mut ranking = Vec::with_capacity(scored.len());
    let mut score_table = Vec::with_capacity(scored.len());

    for (candidate, score) in scored.iter() {
        let delta = *score - best_score;
        let bayes_factor = delta.exp();
        ranking.push(RankedRow {
            index: candidate.index,
            name: candidate.name.clone(),
            score: *score,
            delta,
            bayes_factor,
            effective_dof: candidate.effective_dof,
        });
        score_table.push(ScoreRow {
            name: candidate.name.clone(),
            reml_score: *score,
            delta_reml: delta,
            bayes_factor_vs_best: bayes_factor,
            effective_dof: candidate.effective_dof,
        });
    }

    let evidence_summary = if scored.len() >= 2 {
        let runner_up = &scored[1].0;
        let log_bf = best_score - scored[1].1;
        format!(
            "{} wins by Bayes factor {} over {}",
            winner,
            format_bayes_factor(log_bf),
            runner_up.name
        )
    } else {
        format!("{winner} (single fit; no comparison)")
    };

    Ok(CompareReport {
        ranking,
        winner,
        evidence_summary,
        score_table,
    })
}

pub fn tierney_kadane_normalized_score(
    raw_reml: f64,
    null_dim: f64,
    null_space_logdet: Option<f64>,
) -> Result<f64, String> {
    crate::solver::topology_selector::tk_normalized_score(
        raw_reml,
        null_dim,
        null_space_logdet,
        1.0,
        1,
        TopologyScoreScale::PerObservation,
    )
}

fn normalized_reml_score(candidate: &RemlCandidate) -> Result<f64, String> {
    match candidate.null_dim {
        Some(null_dim) => tierney_kadane_normalized_score(
            candidate.raw_reml,
            null_dim,
            candidate.null_space_logdet,
        ),
        None => Ok(candidate.raw_reml),
    }
}

fn format_bayes_factor(log_bf: f64) -> String {
    if !log_bf.is_finite() {
        return "inf".to_string();
    }
    if log_bf.abs() >= std::f64::consts::LN_10 * 3.0 {
        return format!("1e{:+.1}", log_bf / std::f64::consts::LN_10);
    }
    format_three_significant(log_bf.exp())
}

fn format_three_significant(value: f64) -> String {
    if value == 0.0 {
        return "0".to_string();
    }
    let abs = value.abs();
    let decimals = if abs >= 100.0 {
        0
    } else if abs >= 10.0 {
        1
    } else if abs >= 1.0 {
        2
    } else {
        let exponent = abs.log10().floor();
        (-exponent as i32 + 2).max(0) as usize
    };
    let mut formatted = format!("{value:.decimals$}");
    if formatted.contains('.') {
        while formatted.ends_with('0') {
            formatted.pop();
        }
        if formatted.ends_with('.') {
            formatted.pop();
        }
    }
    formatted
}
