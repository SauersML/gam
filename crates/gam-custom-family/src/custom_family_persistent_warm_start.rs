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

use crate::{CachedInnerMode, ConstrainedWarmStart, normalize_active_sets};
use gam_linalg::matrix::DesignMatrix;
use gam_model_api::families::custom_family::{BlockwiseFitOptions, CustomFamily};
use gam_problem::{ParameterBlockSpec, PenaltyMatrix};
use gam_runtime::warm_start::Fingerprinter;
use gam_solve::persistent_warm_start::{
    PersistentBlockInnerSummary, PersistentBlockWarmStartRecord, load_block_record,
    store_block_record,
};
use ndarray::{Array1, Array2};
use std::any::type_name;
use std::sync::atomic::Ordering;

use gam_solve::warm_start_artifact::{
    FIT_ARTIFACT_SCHEMA, FitArtifact, FitDescriptor, GlobalFitSummary, ResponseSig,
    RowPopulationTag, SerializableBasisMeta, TermArtifact, TermIdentityKey, TermRole,
    TransferProvenance, term_identity_from_block,
};
use gam_solve::warm_start_transfer::{TermBuildContext, TransferConfig, build_warm_start};

/// Build the structural identity of each block at the fit-spec layer. The
/// returned `TermIdentityKey` is fold-invariant (keyed on block name + penalty
/// structure + realized reduced β-width, never on row-dependent values), which
/// is exactly what lets an LOSO fold match a prior full-data artifact while
/// splitting two diseases whose spatial basis collapses to a different
/// effective support (different `design.ncols()` ⇒ different per-block β
/// dimension ⇒ distinct identity, so a p=37 fit never matches a p=85 artifact).
fn block_term_identity(spec: &ParameterBlockSpec) -> TermIdentityKey {
    let role = TermRole::from_block_name(&spec.name);
    let labels: Vec<Option<String>> = spec
        .penalties
        .iter()
        .map(|p| p.precision_label().map(str::to_owned))
        .collect();
    term_identity_from_block(
        role,
        &spec.name,
        &labels,
        &spec.nullspace_dims,
        spec.design.ncols(),
    )
}

/// Map each block to the set of OUTER ρ indices its penalties occupy, using
/// the label layout's flat `physical_to_outer` table sliced by per-block
/// penalty counts. Fixed (`None`) penalties contribute no outer slot.
fn per_block_rho_slots(
    specs: &[ParameterBlockSpec],
    physical_to_outer: &[Option<usize>],
) -> Vec<Vec<usize>> {
    let mut out = Vec::with_capacity(specs.len());
    let mut physical = 0usize;
    for spec in specs {
        let mut slots = Vec::new();
        for _ in 0..spec.penalties.len() {
            if let Some(Some(outer)) = physical_to_outer.get(physical) {
                slots.push(*outer);
            }
            physical += 1;
        }
        out.push(slots);
    }
    out
}

/// Build the [`FitDescriptor`] for a fit from its specs and family kind. The
/// descriptor key (used for cross-fit lookup) excludes rows/response by
/// construction (see [`FitDescriptor::descriptor_key`]).
fn descriptor_for(specs: &[ParameterBlockSpec], family_kind: &str, n_rows: usize) -> FitDescriptor {
    let term_identities = specs.iter().map(block_term_identity).collect();
    FitDescriptor {
        family_kind: family_kind.to_string(),
        term_identities,
        response_signature: ResponseSig {
            family_kind: family_kind.to_string(),
            n_response_channels: 1,
        },
        row_population: Some(RowPopulationTag {
            n_rows,
            label: None,
        }),
    }
}

/// Capture and persist a cross-fit [`FitArtifact`] from a converged fit.
///
/// `gauge` lifts the converged REDUCED per-block β to RAW coordinates (the
/// identifiability transform is fit-specific, so we store gauge-free raw
/// coefficients). `rho` is the converged outer log-smoothing vector;
/// per-block slices are taken via the label layout. Best-effort: any anomaly
/// (length mismatch, non-finite, lift failure) silently skips the capture —
/// a missing artifact only forfeits a future warm start, never the fit.
pub(crate) fn capture_fit_artifact<F: CustomFamily + ?Sized>(
    specs: &[ParameterBlockSpec],
    gauge: &gam_solve::gauge::Gauge,
    reduced_block_beta: &[Array1<f64>],
    rho: &Array1<f64>,
    physical_to_outer: &[Option<usize>],
    outer_objective: f64,
    converged: bool,
) {
    let family_kind = type_name::<F>();
    let (n_rows, ..) = custom_family_cache_shape(specs);
    if reduced_block_beta.len() != specs.len() || gauge.n_blocks() != specs.len() {
        return;
    }
    // Lift reduced θ -> raw β per block. Pre-validate every block's reduced
    // width against the gauge so we degrade to a skip on any disagreement
    // rather than tripping the gauge's internal width assertion.
    if gauge.block_starts_reduced.len() != specs.len() + 1 {
        return;
    }
    for (b, beta) in reduced_block_beta.iter().enumerate() {
        let expected = gauge.block_starts_reduced[b + 1] - gauge.block_starts_reduced[b];
        if beta.len() != expected {
            return;
        }
    }
    let raw_block_beta = gauge.lift_block_betas(reduced_block_beta);
    let slots = per_block_rho_slots(specs, physical_to_outer);
    let mut terms = Vec::with_capacity(specs.len());
    for (idx, spec) in specs.iter().enumerate() {
        let raw_beta: Vec<f64> = raw_block_beta[idx].iter().copied().collect();
        if raw_beta.iter().any(|v| !v.is_finite()) {
            return;
        }
        let rho_for_term: Vec<f64> = slots[idx]
            .iter()
            .filter_map(|&s| rho.get(s).copied())
            .collect();
        if rho_for_term.iter().any(|v| !v.is_finite()) {
            return;
        }
        terms.push(TermArtifact {
            identity: block_term_identity(spec),
            role: TermRole::from_block_name(&spec.name),
            // No BasisMetadata at this layer; record the minimal structural
            // stub so the serialized term is self-describing.
            basis_meta: SerializableBasisMeta {
                kind: "block-spec".to_string(),
                degree: None,
                num_knots: None,
                n_centers: Some(spec.design.ncols() as u64),
                nullspace_order: None,
                matern_nu: None,
                periodic: false,
            },
            joint_null_rotation: None,
            raw_beta,
            rho_for_term,
        });
    }
    let artifact = FitArtifact {
        schema: FIT_ARTIFACT_SCHEMA,
        created_unix_secs: now_unix_secs(),
        descriptor: descriptor_for(specs, family_kind, n_rows),
        // family_kind is the static type name (fold-invariant).
        terms,
        global: GlobalFitSummary {
            outer_objective,
            converged,
            n_rows,
        },
    };
    if let Err(err) = gam_solve::persistent_warm_start::store_fit_artifact(&artifact) {
        log::debug!("[fit-artifact] capture skipped: {err}");
    }
}

/// Extract block `b`'s sub-matrix of the gauge lift `T : reduced → raw`
/// (shape `raw_width_b × reduced_width_b`), used to project the parent's RAW β
/// into this fold's reduced coordinates. The gauge may be block-diagonal or
/// block-upper-triangular; we take the on-diagonal `(raw_b, reduced_b)` block,
/// which is exactly the per-block reduction section. Returns `None` on any
/// partition/shape anomaly so the term falls back to a cold β.
fn gauge_block_t(gauge: &gam_solve::gauge::Gauge, b: usize) -> Option<Array2<f64>> {
    let r0 = *gauge.block_starts_raw.get(b)?;
    let r1 = *gauge.block_starts_raw.get(b + 1)?;
    let c0 = *gauge.block_starts_reduced.get(b)?;
    let c1 = *gauge.block_starts_reduced.get(b + 1)?;
    if r1 < r0 || c1 < c0 || r1 > gauge.t_full.nrows() || c1 > gauge.t_full.ncols() {
        return None;
    }
    Some(gauge.t_full.slice(ndarray::s![r0..r1, c0..c1]).to_owned())
}

/// Consume the best matching parent [`FitArtifact`] (exact descriptor-key
/// match) and build a warm `(ρ, β)` start for the current fit.
///
/// ρ transfers per matched term (the marquee LOSO win); β is least-squares
/// projected from the parent's RAW coefficients onto this fold's reduced
/// subspace via the new gauge lift `T_b` — this is the function-space
/// cross-fit transfer that survives a differing reduced width (e.g. p=37 vs
/// p=35). Any per-block anomaly degrades that block to a cold zero β; any
/// whole-artifact anomaly returns `None` (full cold start). This can never
/// error a fit: a warm start only seeds the inner Newton's starting iterate,
/// which still runs to its KKT/REML certificate.
///
/// The returned `ConstrainedWarmStart` carries ρ in the OUTER coordinate
/// system (same length as `rho_default`) and per-block β in the reduced
/// (`spec.design.ncols()`) coordinates, with `cached_inner = None` so the
/// inner solve replays from the seed rather than reusing a stale mode.
pub(crate) fn consume_fit_artifact<F: CustomFamily + ?Sized>(
    specs: &[ParameterBlockSpec],
    gauge: &gam_solve::gauge::Gauge,
    physical_to_outer: &[Option<usize>],
    rho_default: &Array1<f64>,
) -> Option<ConstrainedWarmStart> {
    let family_kind = type_name::<F>();
    let (n_rows, ..) = custom_family_cache_shape(specs);
    let descriptor = descriptor_for(specs, family_kind, n_rows);
    let key_hex = descriptor.descriptor_key().to_hex();
    let parent = gam_solve::persistent_warm_start::load_fit_artifact_by_descriptor(&key_hex)?;

    // The gauge must partition into exactly the spec blocks for the per-block
    // T extraction to be meaningful; otherwise we transfer ρ only.
    let gauge_aligned = gauge.n_blocks() == specs.len();

    let slots = per_block_rho_slots(specs, physical_to_outer);
    let new_terms: Vec<TermBuildContext> = specs
        .iter()
        .enumerate()
        .zip(slots.into_iter())
        .map(|((b, spec), rho_slots)| TermBuildContext {
            identity: block_term_identity(spec),
            rho_slots,
            reduced_width: spec.design.ncols(),
            gauge_t_block: if gauge_aligned {
                gauge_block_t(gauge, b)
            } else {
                None
            },
        })
        .collect();

    match build_warm_start(
        &descriptor,
        &new_terms,
        rho_default,
        &parent,
        TransferConfig::default(),
    ) {
        Ok(result) => {
            let n_rho = result
                .provenance
                .iter()
                .filter(|p| {
                    matches!(
                        p,
                        TransferProvenance::RhoOnly | TransferProvenance::Projected
                    )
                })
                .count();
            let n_beta = result
                .provenance
                .iter()
                .filter(|p| matches!(p, TransferProvenance::Projected))
                .count();
            if n_rho == 0 && n_beta == 0 {
                return None;
            }
            // Final finite-guard: a non-finite warm iterate is never seeded.
            if result.rho.iter().any(|v| !v.is_finite())
                || result
                    .block_beta
                    .iter()
                    .any(|b| b.iter().any(|v| !v.is_finite()))
                || result.block_beta.len() != specs.len()
            {
                log::debug!("[fit-artifact] cross-fit transfer skipped: non-finite warm iterate");
                return None;
            }
            log::info!(
                "[CACHE] beta-warm action=projected source=cross-fit descriptor={key_hex} \
                 terms={} rho_warm={n_rho} beta_projected={n_beta}",
                new_terms.len(),
            );
            Some(ConstrainedWarmStart {
                rho: result.rho,
                block_beta: result.block_beta,
                active_sets: vec![None; specs.len()],
                cached_inner: None,
            })
        }
        Err(err) => {
            log::debug!("[CACHE] beta-warm action=cold-fallback reason={err:?}");
            None
        }
    }
}

fn now_unix_secs() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

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

pub(crate) fn hash_cf_design_matrix(
    hasher: &mut Fingerprinter,
    design: &DesignMatrix,
) -> Result<(), String> {
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
    hasher.write_str(&gam_solve::persistent_warm_start::cache_schema_tag());
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
    hasher.write_bool(options.outer_rel_cost_tol.is_some());
    if let Some(value) = options.outer_rel_cost_tol {
        hasher.write_f64(value);
    }
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

pub(crate) fn custom_family_cache_shape(
    specs: &[ParameterBlockSpec],
) -> (usize, Vec<String>, Vec<usize>) {
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
    // the interior anyway (see `rho_optimizer.rs` `[CACHE] hit-clamp`).
    const SATURATION_THRESHOLD: f64 = 9.0;
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

    // Keep the adaptive cap as a performance budget, not a mathematical
    // admissibility constraint.  Some valid smooth/nonlinear custom-family
    // profiles (notably negative-binomial dispersion location-scale) need a
    // few dozen polishing cycles at nearby trial rho values even after a
    // warm-started evaluation converged quickly.  Capping the next derivative
    // evaluation at `last_cycles + 5` can then reject an otherwise valid rho
    // point before the KKT certificate has a chance to fire, causing the outer
    // optimizer to exhaust fallbacks on a spurious "inner solve did not
    // converge".  Preserve the adaptive shrink for easy regions, but never
    // drive the ordinary outer-evaluation budget below a small polishing floor.
    let cap_floor = full_budget.min(64);
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
    .clamp(cap_floor, full_budget);
    outer_cap.store(next_cap, Ordering::Relaxed);
}
