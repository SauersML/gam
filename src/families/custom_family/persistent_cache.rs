//! Persistent (on-disk) warm-start cache for the custom-family blockwise fit:
//! fingerprint hashing, cache-key derivation, and record load/store.
//!
//! Pure relocation from `custom_family.rs` (issue #780 decomposition): the
//! `hash_cf_*` fingerprint primitives, the persistent cache-key derivation
//! (`persistent_custom_family_key`, `custom_family_cache_shape`), the
//! load/store of the on-disk `PersistentBlockWarmStartRecord`
//! (`load_persistent_custom_family_warm_start`,
//! `persistent_block_inner_summary`, `store_persistent_custom_family_warm_start`),
//! and the outer→inner iteration-cap update driven by the restored warm start.
//! No behavior change — bodies are byte-identical and the three entry points
//! consumed elsewhere in the parent are re-imported so every call site is
//! unchanged.

use super::{
    BlockwiseFitOptions, CachedInnerMode, ConstrainedWarmStart, CustomFamily, ParameterBlockSpec,
    PenaltyMatrix, normalize_active_sets,
};
use crate::cache::Fingerprinter;
use crate::matrix::DesignMatrix;
use crate::solver::persistent_warm_start::{
    PersistentBlockInnerSummary, PersistentBlockWarmStartRecord, load_block_record,
    store_block_record,
};
use ndarray::{Array1, Array2};
use std::any::type_name;
use std::sync::atomic::Ordering;

pub(crate) fn hash_cf_array_view(hasher: &mut Fingerprinter, values: ndarray::ArrayView1<'_, f64>) {
    hasher.write_usize(values.len());
    for &value in values {
        hasher.write_f64(value);
    }
}

pub(crate) fn hash_cf_array2(hasher: &mut Fingerprinter, values: &Array2<f64>) {
    hasher.write_usize(values.nrows());
    hasher.write_usize(values.ncols());
    for &value in values {
        hasher.write_f64(value);
    }
}

pub(crate) fn hash_cf_design_matrix(hasher: &mut Fingerprinter, design: &DesignMatrix) -> Result<(), String> {
    let n = design.nrows();
    let p = design.ncols();
    hasher.write_usize(n);
    hasher.write_usize(p);
    let bytes_per_row = p.saturating_mul(std::mem::size_of::<f64>()).max(1);
    let chunk_rows = ((8 * 1024 * 1024) / bytes_per_row).clamp(1, 4096);
    for start in (0..n).step_by(chunk_rows) {
        let end = (start + chunk_rows).min(n);
        let chunk = design
            .try_row_chunk(start..end)
            .map_err(|e| format!("custom-family persistent warm-start design hash failed: {e}"))?;
        hash_cf_array2(hasher, &chunk);
    }
    Ok(())
}

pub(crate) fn hash_cf_penalty(hasher: &mut Fingerprinter, penalty: &PenaltyMatrix) {
    match penalty {
        PenaltyMatrix::Dense(matrix) => {
            hasher.write_str("dense");
            hash_cf_array2(hasher, matrix);
        }
        PenaltyMatrix::KroneckerFactored { left, right } => {
            hasher.write_str("kron");
            hash_cf_array2(hasher, left);
            hash_cf_array2(hasher, right);
        }
        PenaltyMatrix::Blockwise {
            local,
            col_range,
            total_dim,
        } => {
            hasher.write_str("blockwise");
            hasher.write_usize(col_range.start);
            hasher.write_usize(col_range.end);
            hasher.write_usize(*total_dim);
            hash_cf_array2(hasher, local);
        }
        PenaltyMatrix::Labeled { label, inner } => {
            hasher.write_str("labeled");
            hasher.write_str(label);
            hash_cf_penalty(hasher, inner);
        }
        PenaltyMatrix::Fixed { log_lambda, inner } => {
            hasher.write_str("fixed");
            hasher.write_u64(log_lambda.to_bits());
            hash_cf_penalty(hasher, inner);
        }
    }
}

pub(crate) fn persistent_custom_family_key<F: CustomFamily + ?Sized>(
    family: &F,
    specs: &[ParameterBlockSpec],
    options: &BlockwiseFitOptions,
) -> Option<String> {
    let mut hasher = Fingerprinter::new();
    hasher.write_str("gamfit-persistent-block-warm-start");
    hasher.write_str(&crate::solver::persistent_warm_start::cache_schema_tag());
    hasher.write_str(type_name::<F>());
    hasher.write_str(&family.persistent_warm_start_fingerprint(specs, options)?);
    hasher.write_usize(specs.len());
    for spec in specs {
        hasher.write_str(&spec.name);
        hash_cf_design_matrix(&mut hasher, &spec.design).ok()?;
        hash_cf_array_view(&mut hasher, spec.offset.view());
        hasher.write_usize(spec.penalties.len());
        for penalty in &spec.penalties {
            hash_cf_penalty(&mut hasher, penalty);
        }
        hasher.write_usize(spec.nullspace_dims.len());
        for &dim in &spec.nullspace_dims {
            hasher.write_usize(dim);
        }
        hash_cf_array_view(&mut hasher, spec.initial_log_lambdas.view());
    }
    hasher.write_usize(options.inner_max_cycles);
    hasher.write_f64(options.inner_tol);
    hasher.write_usize(options.outer_max_iter);
    hasher.write_f64(options.outer_tol);
    hasher.write_f64(options.minweight);
    hasher.write_f64(options.ridge_floor);
    hasher.write_str(&format!("{:?}", options.ridge_policy));
    hasher.write_bool(options.use_remlobjective);
    hasher.write_bool(options.use_outer_hessian);
    hasher.write_bool(options.compute_covariance);
    hasher.write_bool(options.early_exit_threshold.is_some());
    if let Some(value) = options.early_exit_threshold {
        hasher.write_f64(value);
    }
    hasher.write_bool(options.outer_score_subsample.is_some());
    hasher.write_bool(options.auto_outer_subsample);
    Some(format!("cf-{}", hasher.finish_hex()))
}

pub(crate) fn custom_family_cache_shape(specs: &[ParameterBlockSpec]) -> (usize, Vec<String>, Vec<usize>) {
    let n_rows = specs.first().map(|spec| spec.design.nrows()).unwrap_or(0);
    let block_names = specs.iter().map(|spec| spec.name.clone()).collect();
    let block_dims = specs.iter().map(|spec| spec.design.ncols()).collect();
    (n_rows, block_names, block_dims)
}

pub(crate) fn load_persistent_custom_family_warm_start<F: CustomFamily + ?Sized>(
    family: &F,
    specs: &[ParameterBlockSpec],
    options: &BlockwiseFitOptions,
    rho_len: usize,
) -> (Option<String>, Option<ConstrainedWarmStart>) {
    let Some(key) = persistent_custom_family_key::<F>(family, specs, options) else {
        return (None, None);
    };
    let (n_rows, block_names, block_dims) = custom_family_cache_shape(specs);
    let Some(record) = load_block_record(&key) else {
        return (Some(key), None);
    };
    if !record.is_compatible(&key, n_rows, &block_names, &block_dims, rho_len) {
        return (Some(key), None);
    }
    let active_sets = normalize_active_sets(record.active_sets);
    let cached_inner = record.inner.map(|inner| CachedInnerMode {
        log_likelihood: inner.log_likelihood,
        penalty_value: inner.penalty_value,
        cycles: inner.cycles,
        converged: inner.converged,
        block_logdet_h: inner.block_logdet_h,
        block_logdet_s: inner.block_logdet_s,
        joint_workspace: None,
        // Persistent warm-start records don't carry the KKT-residual or
        // active-constraint diagnostics (they're not serialized on disk;
        // they're rebuilt from the inner solve on next visit), so a
        // restored cache replay forces the unified evaluator's IFT
        // correction path to degrade to its no-data branch until a fresh
        // joint-Newton pass produces them.
        kkt_residual: None,
        active_constraints: None,
    });
    let inner_status = cached_inner.as_ref().map_or("missing", |inner| {
        if inner.converged {
            "converged"
        } else {
            "partial"
        }
    });
    log::info!(
        "[warm-start-cache] restored custom-family persistent warm start key={key} inner={inner_status}"
    );
    (
        Some(key),
        Some(ConstrainedWarmStart {
            rho: Array1::from_vec(record.rho),
            block_beta: record
                .block_beta
                .into_iter()
                .map(Array1::from_vec)
                .collect(),
            active_sets,
            cached_inner,
        }),
    )
}

pub(crate) fn persistent_block_inner_summary(
    warm_start: &ConstrainedWarmStart,
) -> Option<PersistentBlockInnerSummary> {
    warm_start.cached_inner.as_ref().and_then(|cached| {
        (cached.log_likelihood.is_finite()
            && cached.penalty_value.is_finite()
            && cached.block_logdet_h.is_finite()
            && cached.block_logdet_s.is_finite())
        .then_some(PersistentBlockInnerSummary {
            log_likelihood: cached.log_likelihood,
            penalty_value: cached.penalty_value,
            cycles: cached.cycles,
            converged: cached.converged,
            block_logdet_h: cached.block_logdet_h,
            block_logdet_s: cached.block_logdet_s,
        })
    })
}

pub(crate) fn store_persistent_custom_family_warm_start(
    key: Option<&str>,
    specs: &[ParameterBlockSpec],
    warm_start: &ConstrainedWarmStart,
) {
    let Some(key) = key else {
        return;
    };
    let (n_rows, block_names, block_dims) = custom_family_cache_shape(specs);
    if warm_start.block_beta.len() != block_dims.len()
        || warm_start
            .block_beta
            .iter()
            .zip(block_dims.iter())
            .any(|(beta, dim)| beta.len() != *dim || beta.iter().any(|v| !v.is_finite()))
        || warm_start.rho.iter().any(|v| !v.is_finite())
    {
        return;
    }
    // Saturation gate: never persist ρ that hit the outer optimizer's
    // box (|ρ_i| ≥ 9). Those iterates are either at a legitimate active
    // bound or a non-converged intermediate; either way they make poor
    // seed material because the load-side clamp pulls them back into
    // the interior anyway (see `outer_strategy.rs` `[CACHE] hit-clamp`).
    pub(crate) const SATURATION_THRESHOLD: f64 = 9.0;
    if warm_start
        .rho
        .iter()
        .any(|&v| v.abs() >= SATURATION_THRESHOLD)
    {
        log::debug!(
            "[warm-start-cache] skip persist custom-family key={} \
             reason=rho-saturated threshold=±{:.1} rho_inf_norm={:.3e}",
            key,
            SATURATION_THRESHOLD,
            warm_start
                .rho
                .iter()
                .fold(0.0_f64, |acc, &v| acc.max(v.abs())),
        );
        return;
    }
    let mut record =
        PersistentBlockWarmStartRecord::new(key.to_string(), n_rows, block_names, block_dims);
    record.updated_unix_secs = record.created_unix_secs;
    record.rho = warm_start.rho.to_vec();
    record.block_beta = warm_start
        .block_beta
        .iter()
        .map(|beta| beta.to_vec())
        .collect();
    record.active_sets = warm_start.active_sets.clone();
    record.inner = persistent_block_inner_summary(warm_start);
    if let Err(err) = store_block_record(&record) {
        log::warn!("[warm-start-cache] failed to persist custom-family warm start: {err}");
    }
}

pub(crate) const CUSTOM_OUTER_INNER_CAP_MARGIN: usize = 5;

pub(crate) fn update_custom_outer_inner_cap_from_warm_start(
    options: &BlockwiseFitOptions,
    warm_start: &ConstrainedWarmStart,
    gradient_norm: Option<f64>,
    initial_gradient_norm: &mut Option<f64>,
) {
    let Some(outer_cap) = options.outer_inner_max_iterations.as_ref() else {
        return;
    };
    let full_budget = options.inner_max_cycles.max(1);
    let Some(cached_inner) = warm_start.cached_inner.as_ref() else {
        outer_cap.store(full_budget, Ordering::Relaxed);
        return;
    };

    if let Some(norm) = gradient_norm.filter(|value| value.is_finite() && *value > 0.0) {
        if initial_gradient_norm.is_none() {
            *initial_gradient_norm = Some(norm);
        }
        if matches!(*initial_gradient_norm, Some(initial) if initial > 0.0 && norm / initial < 0.01)
        {
            outer_cap.store(full_budget, Ordering::Relaxed);
            return;
        }
    }

    let next_cap = if cached_inner.converged {
        cached_inner
            .cycles
            .saturating_add(CUSTOM_OUTER_INNER_CAP_MARGIN)
    } else {
        cached_inner.cycles.saturating_mul(2).max(
            cached_inner
                .cycles
                .saturating_add(CUSTOM_OUTER_INNER_CAP_MARGIN),
        )
    }
    .clamp(1, full_budget);
    outer_cap.store(next_cap, Ordering::Relaxed);
}
