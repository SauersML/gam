//! Runnable identifiability-theorem precondition checks.
//!
//! Python, the CLI, and other bindings consume the same `Vec<TheoremResult>`
//! from this module. Thresholds are explicit call inputs via [`Thresholds`].

use std::collections::BTreeMap;

use ndarray::{Array2, ArrayView2, Axis};
use serde::{Deserialize, Serialize};

use gam_linalg::faer_ndarray::FaerSvd;

/// Below this std the aux column is "constant" (Khemakhem 2107.10098 Thm. 1
/// — a constant column carries zero conditioning information).
pub const DEFAULT_IVAE_AUX_VAR_FLOOR: f64 = 1.0e-9;

/// Tolerance used by the truncated-SVD rank routine for the aux column-rank
/// check (Khemakhem 2107.10098 §3 parametric-richness assumption).
pub const DEFAULT_IVAE_AUX_RANK_RTOL: f64 = 1.0e-8;

/// Khemakhem 2107.10098 §3: encoder must be "non-trivially nonlinear" — bare
/// linear (1 affine layer) does not satisfy the universal-approximation
/// argument that pushes identifiability through the encoder.
pub const DEFAULT_IVAE_MIN_ENCODER_LAYERS: i64 = 2;

/// Lachapelle 2401.04890 §2.4: at L1 equilibrium >=50% of the decoder
/// Jacobian entries on the free block are near zero.
pub const DEFAULT_MECH_SPARSITY_FRACTION: f64 = 0.50;

/// Relative threshold for "near-zero" decoder entry — mirrors the paper's
/// column-relative thresholding.
pub const DEFAULT_MECH_SPARSITY_ZERO_TOL: f64 = 1.0e-3;

/// Khemakhem App. A.3: encoder activation variance must be bounded. We treat
/// activation variances above this ceiling as a hard fail.
pub const DEFAULT_RANDPROJ_VAR_CEILING: f64 = 1.0e6;

/// Variances above this floor (but below the ceiling) downgrade the random
/// projection check to a warn — encoder is large but not yet unbounded.
pub const DEFAULT_RANDPROJ_VAR_WARN: f64 = 1.0e3;

/// Tunable thresholds — every field has a paper-backed default and can be
/// overridden per call (constructor kwargs in Python, struct literal here).
#[derive(Debug, Clone, Copy, Deserialize, Serialize)]
pub struct Thresholds {
    pub ivae_aux_var_floor: f64,
    pub ivae_aux_rank_rtol: f64,
    pub ivae_min_encoder_layers: i64,
    pub mech_sparsity_fraction: f64,
    pub mech_sparsity_zero_tol: f64,
    pub randproj_var_warn: f64,
    pub randproj_var_ceiling: f64,
}

impl Default for Thresholds {
    fn default() -> Self {
        Self {
            ivae_aux_var_floor: DEFAULT_IVAE_AUX_VAR_FLOOR,
            ivae_aux_rank_rtol: DEFAULT_IVAE_AUX_RANK_RTOL,
            ivae_min_encoder_layers: DEFAULT_IVAE_MIN_ENCODER_LAYERS,
            mech_sparsity_fraction: DEFAULT_MECH_SPARSITY_FRACTION,
            mech_sparsity_zero_tol: DEFAULT_MECH_SPARSITY_ZERO_TOL,
            randproj_var_warn: DEFAULT_RANDPROJ_VAR_WARN,
            randproj_var_ceiling: DEFAULT_RANDPROJ_VAR_CEILING,
        }
    }
}

/// Outcome of a single per-theorem precondition check.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TheoremResult {
    pub theorem_name: String,
    pub status: TheoremStatus,
    pub reason: String,
    pub metric: BTreeMap<String, f64>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum TheoremStatus {
    Pass,
    Warn,
    Fail,
}

impl TheoremStatus {
    fn rank(&self) -> u8 {
        match self {
            TheoremStatus::Pass => 0,
            TheoremStatus::Warn => 1,
            TheoremStatus::Fail => 2,
        }
    }
    fn worse(self, other: TheoremStatus) -> TheoremStatus {
        if other.rank() > self.rank() {
            other
        } else {
            self
        }
    }
}

/// Caller-supplied summary of the fit. All numerical evidence is in here —
/// the Rust check needs no Python objects.
#[derive(Debug, Clone, Default, Deserialize, Serialize)]
pub struct FitSummary {
    /// Auxiliary covariates of shape `(n_obs, n_supervised)`. Row-major. If
    /// `None`, the iVAE check downgrades to a warn-with-skip.
    pub aux: Option<Vec<Vec<f64>>>,
    /// Declared supervised latent dim.
    pub n_supervised: Option<i64>,
    /// Declared free latent dim.
    pub n_free: Option<i64>,
    /// Decoder of shape `(n_features, n_supervised + n_free)`.
    pub decoder: Option<Vec<Vec<f64>>>,
    /// Number of affine (Linear) layers in the encoder.
    pub encoder_depth: Option<i64>,
    /// Sparsity penalty weight used at fit time.
    pub mech_sparsity_weight: Option<f64>,
    /// Latent samples / encoder activations of shape `(n_obs, latent_dim)`.
    pub activations: Option<Vec<Vec<f64>>>,
    /// Ground-truth latent dim (e.g. from a simulator). Optional.
    pub ground_truth_dim: Option<i64>,
    /// Threshold overrides (defaults to paper-cited values when missing).
    #[serde(default)]
    pub thresholds: Option<Thresholds>,
}

fn rows_to_array(rows: &[Vec<f64>]) -> Result<Array2<f64>, String> {
    if rows.is_empty() {
        return Ok(Array2::<f64>::zeros((0, 0)));
    }
    let ncols = rows[0].len();
    for (i, row) in rows.iter().enumerate() {
        if row.len() != ncols {
            return Err(format!(
                "ragged matrix: row 0 has {ncols} cols but row {i} has {} cols",
                row.len()
            ));
        }
    }
    let nrows = rows.len();
    let mut flat = Vec::with_capacity(nrows * ncols);
    for row in rows {
        flat.extend_from_slice(row);
    }
    Array2::from_shape_vec((nrows, ncols), flat).map_err(|e| e.to_string())
}

fn column_std(mat: ArrayView2<f64>) -> Vec<f64> {
    let n = mat.nrows() as f64;
    if n <= 0.0 {
        return vec![0.0; mat.ncols()];
    }
    let mut out = Vec::with_capacity(mat.ncols());
    for col in mat.axis_iter(Axis(1)) {
        let mean = col.sum() / n;
        let mut var = 0.0_f64;
        for v in col.iter() {
            let d = v - mean;
            var += d * d;
        }
        out.push((var / n).sqrt());
    }
    out
}

fn column_var(mat: ArrayView2<f64>) -> Vec<f64> {
    column_std(mat).into_iter().map(|s| s * s).collect()
}

/// Tolerance-based rank via faer SVD: count singular values larger than
/// `rtol * max_singular_value`.
fn matrix_rank(mat: ArrayView2<f64>, rtol: f64) -> Result<usize, String> {
    if mat.nrows() == 0 || mat.ncols() == 0 {
        return Ok(0);
    }
    let owned = mat.to_owned();
    let (_u, sigma, _vt) = owned.svd(false, false).map_err(|e| format!("{e:?}"))?;
    if sigma.is_empty() {
        return Ok(0);
    }
    let smax = sigma.iter().cloned().fold(0.0_f64, f64::max);
    if smax <= 0.0 {
        return Ok(0);
    }
    let cutoff = smax * rtol;
    Ok(sigma.iter().filter(|s| **s > cutoff).count())
}

/// Khemakhem 2107.10098 Theorem 1 preconditions.
pub fn check_ivae(summary: &FitSummary, thr: &Thresholds) -> TheoremResult {
    let mut metric: BTreeMap<String, f64> = BTreeMap::new();
    let mut issues: Vec<String> = Vec::new();
    let mut status = TheoremStatus::Pass;

    let aux_rows = match summary.aux.as_ref() {
        Some(a) => a,
        None => {
            return TheoremResult {
                theorem_name: "iVAE".to_string(),
                status: TheoremStatus::Warn,
                reason: "iVAE check skipped: no aux provided in fit summary.".to_string(),
                metric,
            };
        }
    };
    let n_supervised = match summary.n_supervised {
        Some(v) => v,
        None => {
            return TheoremResult {
                theorem_name: "iVAE".to_string(),
                status: TheoremStatus::Warn,
                reason: "iVAE check skipped: n_supervised missing.".to_string(),
                metric,
            };
        }
    };

    let aux = match rows_to_array(aux_rows) {
        Ok(a) => a,
        Err(e) => {
            return TheoremResult {
                theorem_name: "iVAE".to_string(),
                status: TheoremStatus::Fail,
                reason: format!("aux is malformed: {e}"),
                metric,
            };
        }
    };

    let stds = column_std(aux.view());
    let min_std = stds.iter().cloned().fold(f64::INFINITY, f64::min);
    metric.insert(
        "aux_min_std".to_string(),
        if stds.is_empty() { 0.0 } else { min_std },
    );
    if stds.is_empty() || stds.iter().any(|s| *s <= thr.ivae_aux_var_floor) {
        let zeros: Vec<usize> = stds
            .iter()
            .enumerate()
            .filter(|(_, s)| **s <= thr.ivae_aux_var_floor)
            .map(|(i, _)| i)
            .collect();
        issues.push(format!(
            "iVAE identifiability requires auxiliary covariate variation; \
             aux axes {zeros:?} are constant across observations (min std \
             {min_std:.3e} <= {:.0e}); Khemakhem 2107.10098 Thm. 1 \
             conditioning rank is zero.",
            thr.ivae_aux_var_floor,
        ));
        status = status.worse(TheoremStatus::Fail);
    }

    let rank = match matrix_rank(aux.view(), thr.ivae_aux_rank_rtol) {
        Ok(r) => r,
        Err(e) => {
            return TheoremResult {
                theorem_name: "iVAE".to_string(),
                status: TheoremStatus::Fail,
                reason: format!("aux SVD failed: {e}"),
                metric,
            };
        }
    };
    metric.insert("aux_column_rank".to_string(), rank as f64);
    metric.insert("n_supervised".to_string(), n_supervised as f64);
    if (rank as i64) < n_supervised {
        issues.push(format!(
            "aux column rank {rank} < n_supervised={n_supervised}: \
             Khemakhem 2107.10098 §3 parametric-richness fails."
        ));
        status = status.worse(TheoremStatus::Fail);
    }

    match summary.encoder_depth {
        None => {
            issues.push(
                "encoder depth unknown — cannot verify the >=2-layer \
                 requirement of Khemakhem 2107.10098 §3."
                    .to_string(),
            );
            status = status.worse(TheoremStatus::Warn);
        }
        Some(depth) => {
            metric.insert("encoder_depth".to_string(), depth as f64);
            if depth < 1 {
                issues.push(format!("encoder depth {depth} < 1; no encoder is present."));
                status = status.worse(TheoremStatus::Fail);
            } else if depth == 1 {
                issues.push(
                    "encoder depth == 1 (bare linear); Khemakhem 2107.10098 \
                     §3 requires non-linear encoder. Identifiability voided."
                        .to_string(),
                );
                status = status.worse(TheoremStatus::Fail);
            } else if depth < thr.ivae_min_encoder_layers {
                issues.push(format!(
                    "encoder depth {depth} < canonical min={}: \
                     Khemakhem 2107.10098 §3 universal-approximation \
                     argument is weakened.",
                    thr.ivae_min_encoder_layers,
                ));
                status = status.worse(TheoremStatus::Warn);
            }
        }
    }

    let reason = if matches!(status, TheoremStatus::Pass) {
        "all Khemakhem 2107.10098 Thm. 1 preconditions hold".to_string()
    } else {
        issues.join(" | ")
    };
    TheoremResult {
        theorem_name: "iVAE".to_string(),
        status,
        reason,
        metric,
    }
}

/// Lachapelle 2401.04890 Theorem preconditions.
pub fn check_mechanism_sparsity(summary: &FitSummary, thr: &Thresholds) -> TheoremResult {
    let mut metric: BTreeMap<String, f64> = BTreeMap::new();
    let mut issues: Vec<String> = Vec::new();
    let mut status = TheoremStatus::Pass;

    let decoder_rows = match summary.decoder.as_ref() {
        Some(d) => d,
        None => {
            return TheoremResult {
                theorem_name: "MechanismSparsity".to_string(),
                status: TheoremStatus::Warn,
                reason: "MechanismSparsity skipped: no decoder in fit summary.".to_string(),
                metric,
            };
        }
    };
    let n_sup = summary.n_supervised.unwrap_or(0);
    let n_free = match summary.n_free {
        Some(v) => v,
        None => {
            return TheoremResult {
                theorem_name: "MechanismSparsity".to_string(),
                status: TheoremStatus::Warn,
                reason: "MechanismSparsity skipped: n_free missing.".to_string(),
                metric,
            };
        }
    };

    let decoder = match rows_to_array(decoder_rows) {
        Ok(d) => d,
        Err(e) => {
            return TheoremResult {
                theorem_name: "MechanismSparsity".to_string(),
                status: TheoremStatus::Fail,
                reason: format!("decoder is malformed: {e}"),
                metric,
            };
        }
    };

    let total_cols = decoder.ncols() as i64;
    if n_sup + n_free > total_cols || n_sup < 0 || n_free < 0 {
        return TheoremResult {
            theorem_name: "MechanismSparsity".to_string(),
            status: TheoremStatus::Fail,
            reason: format!(
                "decoder has {total_cols} columns but n_supervised + n_free \
                 = {} + {}.",
                n_sup, n_free,
            ),
            metric,
        };
    }
    let free_cols = decoder.slice(ndarray::s![
        ..,
        (n_sup as usize)..((n_sup + n_free) as usize)
    ]);
    metric.insert(
        "free_block_shape_rows".to_string(),
        free_cols.nrows() as f64,
    );
    metric.insert(
        "free_block_shape_cols".to_string(),
        free_cols.ncols() as f64,
    );

    // Column-relative thresholded zero-fraction.
    let mut col_max = vec![0.0_f64; free_cols.ncols()];
    for col_idx in 0..free_cols.ncols() {
        let col = free_cols.column(col_idx);
        col_max[col_idx] = col.iter().fold(0.0_f64, |acc, v| acc.max(v.abs()));
    }
    let mut zeros: u64 = 0;
    let mut total: u64 = 0;
    for col_idx in 0..free_cols.ncols() {
        let safe_max = if col_max[col_idx] > 0.0 {
            col_max[col_idx]
        } else {
            1.0
        };
        for row_idx in 0..free_cols.nrows() {
            let rel = free_cols[[row_idx, col_idx]].abs() / safe_max;
            if rel <= thr.mech_sparsity_zero_tol {
                zeros += 1;
            }
            total += 1;
        }
    }
    let zero_fraction = if total == 0 {
        0.0
    } else {
        zeros as f64 / total as f64
    };
    metric.insert("decoder_zero_fraction".to_string(), zero_fraction);

    let rank = match matrix_rank(free_cols.view(), 1.0e-8) {
        Ok(r) => r,
        Err(e) => {
            return TheoremResult {
                theorem_name: "MechanismSparsity".to_string(),
                status: TheoremStatus::Fail,
                reason: format!("decoder SVD failed: {e}"),
                metric,
            };
        }
    };
    metric.insert("decoder_free_rank".to_string(), rank as f64);
    if (rank as i64) < n_free {
        issues.push(format!(
            "decoder Jacobian on the free block has rank {rank} < \
             n_free={n_free}; Lachapelle 2401.04890 Thm. requires full \
             rank on the free latents."
        ));
        status = status.worse(TheoremStatus::Fail);
    }

    match summary.mech_sparsity_weight {
        None => {
            issues.push(
                "mech sparsity weight unknown — cannot confirm L1 prox \
                 was active."
                    .to_string(),
            );
            status = status.worse(TheoremStatus::Warn);
        }
        Some(w) => {
            metric.insert("mech_sparsity_weight".to_string(), w);
            if !(w > 0.0) {
                issues.push(format!(
                    "mech sparsity weight = {w} is not strictly positive; \
                     Lachapelle 2401.04890 identification voided."
                ));
                status = status.worse(TheoremStatus::Fail);
            }
        }
    }

    if zero_fraction < thr.mech_sparsity_fraction {
        issues.push(format!(
            "decoder zero-fraction {zero_fraction:.3} < {:.2} threshold \
             from Lachapelle 2401.04890 §2.4: L1 prox has not reached \
             equilibrium, identification weakened.",
            thr.mech_sparsity_fraction,
        ));
        status = status.worse(TheoremStatus::Warn);
    }

    let state_dim = n_sup + n_free;
    if let Some(gt) = summary.ground_truth_dim {
        metric.insert("state_dim".to_string(), state_dim as f64);
        metric.insert("ground_truth_dim".to_string(), gt as f64);
        if state_dim < gt {
            issues.push(format!(
                "state_dim={state_dim} < ground_truth_dim={gt}: Lachapelle \
                 2401.04890 requires at least as many latents as the data \
                 generating process."
            ));
            status = status.worse(TheoremStatus::Fail);
        }
    }

    let reason = if matches!(status, TheoremStatus::Pass) {
        "all Lachapelle 2401.04890 preconditions hold".to_string()
    } else {
        issues.join(" | ")
    };
    TheoremResult {
        theorem_name: "MechanismSparsity".to_string(),
        status,
        reason,
        metric,
    }
}

/// Random-projection identifiability precondition (Khemakhem App. A.3).
pub fn check_random_projection(summary: &FitSummary, thr: &Thresholds) -> TheoremResult {
    let mut metric: BTreeMap<String, f64> = BTreeMap::new();

    let act_rows = match summary.activations.as_ref() {
        Some(a) => a,
        None => {
            return TheoremResult {
                theorem_name: "RandomProjection".to_string(),
                status: TheoremStatus::Warn,
                reason: "RandomProjection skipped: no activations provided.".to_string(),
                metric,
            };
        }
    };
    let act = match rows_to_array(act_rows) {
        Ok(a) => a,
        Err(e) => {
            return TheoremResult {
                theorem_name: "RandomProjection".to_string(),
                status: TheoremStatus::Fail,
                reason: format!("activations malformed: {e}"),
                metric,
            };
        }
    };
    if act.nrows() == 0 || act.ncols() == 0 {
        return TheoremResult {
            theorem_name: "RandomProjection".to_string(),
            status: TheoremStatus::Fail,
            reason: "activations are empty.".to_string(),
            metric,
        };
    }
    let variances = column_var(act.view());
    let var_max = variances.iter().cloned().fold(0.0_f64, f64::max);
    let var_min = variances.iter().cloned().fold(f64::INFINITY, f64::min);
    metric.insert("activation_var_max".to_string(), var_max);
    metric.insert("activation_var_min".to_string(), var_min);
    if variances.iter().any(|v| !v.is_finite()) {
        return TheoremResult {
            theorem_name: "RandomProjection".to_string(),
            status: TheoremStatus::Fail,
            reason: "activations contain non-finite variance; Khemakhem App. A.3 \
                 requires bounded variance."
                .to_string(),
            metric,
        };
    }
    if var_max > thr.randproj_var_ceiling {
        return TheoremResult {
            theorem_name: "RandomProjection".to_string(),
            status: TheoremStatus::Fail,
            reason: format!(
                "max activation variance {var_max:.3e} > ceiling \
                 {:.3e}; encoder is unbounded.",
                thr.randproj_var_ceiling,
            ),
            metric,
        };
    }
    if var_max > thr.randproj_var_warn {
        return TheoremResult {
            theorem_name: "RandomProjection".to_string(),
            status: TheoremStatus::Warn,
            reason: format!(
                "max activation variance {var_max:.3e} > warn-floor \
                 {:.3e}; encoder is large but not yet unbounded.",
                thr.randproj_var_warn,
            ),
            metric,
        };
    }
    TheoremResult {
        theorem_name: "RandomProjection".to_string(),
        status: TheoremStatus::Pass,
        reason: "encoder activation variance is bounded.".to_string(),
        metric,
    }
}

/// Run every applicable identifiability theorem check.
pub fn identifiability_check(summary: &FitSummary) -> Vec<TheoremResult> {
    let thr = summary.thresholds.unwrap_or_default();
    vec![
        check_ivae(summary, &thr),
        check_mechanism_sparsity(summary, &thr),
        check_random_projection(summary, &thr),
    ]
}

/// JSON adaptor: caller serializes a `FitSummary`, gets back a JSON array of
/// `TheoremResult`. The single FFI surface — Python, the CLI, and any
/// future binding all consume this.
pub fn identifiability_check_json(input: &str) -> Result<String, String> {
    let summary: FitSummary =
        serde_json::from_str(input).map_err(|e| format!("invalid FitSummary JSON: {e}"))?;
    let report = identifiability_check(&summary);
    serde_json::to_string(&report).map_err(|e| format!("serialise: {e}"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn constant_aux_fails_ivae() {
        let summary = FitSummary {
            aux: Some(vec![vec![1.0]; 32]),
            n_supervised: Some(1),
            n_free: Some(2),
            encoder_depth: Some(3),
            mech_sparsity_weight: Some(1.0),
            decoder: Some(vec![vec![1.0, 0.5, 0.0, 0.0, 0.0]; 12]),
            activations: Some(vec![vec![0.0; 3]; 32]),
            ground_truth_dim: None,
            thresholds: None,
        };
        let report = identifiability_check(&summary);
        let ivae = report.iter().find(|t| t.theorem_name == "iVAE").unwrap();
        assert_eq!(ivae.status, TheoremStatus::Fail);
        assert!(ivae.reason.to_lowercase().contains("constant"));
        assert_eq!(
            ivae.metric.get("aux_min_std").copied().unwrap_or(f64::NAN),
            0.0
        );
    }

    #[test]
    fn json_roundtrip() {
        let summary = FitSummary {
            aux: Some(vec![vec![1.0, 2.0], vec![3.0, 4.0], vec![5.0, 6.0]]),
            n_supervised: Some(2),
            n_free: Some(1),
            encoder_depth: Some(3),
            mech_sparsity_weight: Some(1.0),
            decoder: Some(vec![vec![1.0, 0.0, 1.0], vec![0.0, 1.0, 1.0]]),
            activations: Some(vec![vec![0.1, 0.2, 0.3], vec![0.4, 0.5, 0.6]]),
            ground_truth_dim: None,
            thresholds: None,
        };
        let json = serde_json::to_string(&summary).unwrap();
        let out = identifiability_check_json(&json).unwrap();
        let parsed: Vec<TheoremResult> = serde_json::from_str(&out).unwrap();
        assert_eq!(parsed.len(), 3);
    }
}
