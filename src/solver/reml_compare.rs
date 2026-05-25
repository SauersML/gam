//! REML model comparison: ranking, deltas, Bayes-factor summary.
//!
//! Pure-Rust core shared by the CLI (`src/main.rs` → `gam compare_models`),
//! the library, and the PyO3 wrapper in `crates/gam-pyffi`. Callers
//! marshal their fits into [`RemlCandidate`] and receive a fully-typed
//! [`RemlComparison`] back; only presentation-layer formatting differs
//! per surface.

use std::cmp::Ordering;

/// One candidate fit in a REML model comparison.
#[derive(Clone, Debug)]
pub struct RemlCandidate {
    /// Caller's original index into the fits array — preserved through
    /// sorting so e.g. external CV-score arrays stay aligned.
    pub index: usize,
    pub name: String,
    /// REML / LAML evidence score on whatever scale the caller chose
    /// (Tierney–Kadane normalised when applicable; see `gamfit._compare`).
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
    pub delta: f64,
    pub bayes_factor: f64,
    pub edf: Option<f64>,
}

#[derive(Clone, Debug)]
pub struct ScoreRow {
    pub name: String,
    pub reml_score: f64,
    pub delta_reml: f64,
    pub bayes_factor_vs_best: f64,
    pub effective_dof: Option<f64>,
}

pub fn compare_reml_fits(mut candidates: Vec<RemlCandidate>) -> Result<RemlComparison, String> {
    if candidates.is_empty() {
        return Err("compare_models requires at least one fit".to_string());
    }

    candidates.sort_by(|left, right| {
        right
            .score
            .partial_cmp(&left.score)
            .unwrap_or(Ordering::Equal)
    });

    let best_score = candidates[0].score;
    let winner = candidates[0].name.clone();
    let mut ranking = Vec::with_capacity(candidates.len());
    let mut score_table = Vec::with_capacity(candidates.len());

    for row in candidates.iter() {
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
            bayes_factor_vs_best: bayes_factor,
            effective_dof: row.edf,
        });
    }

    let evidence_summary = if candidates.len() >= 2 {
        let runner_up = &candidates[1];
        let log_bf = best_score - runner_up.score;
        format!(
            "{} wins by Bayes factor {} over {}",
            winner,
            format_bayes_factor(log_bf),
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
