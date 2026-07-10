//! Typed unified-engine entry for manifold crosscoders (#2231 Inc D).
//!
//! A crosscoder is not a separate optimizer.  This module only owns the
//! multi-layer schedule around [`SaeManifoldOuterObjective`]: validate and stack
//! row-aligned targets once, install the block-relevance coordinates on the
//! shared outer objective, run the same REML engine as a plain manifold SAE,
//! and materialize honest-unit layer reports from the fitted stacked decoder.

use std::sync::Arc;
use std::sync::atomic::AtomicBool;

use gam_solve::rho_optimizer::{OuterProblem, OuterResult};
use gam_solve::seeding::SeedConfig;
use gam_terms::analytic_penalties::AnalyticPenaltyRegistry;
use ndarray::{Array2, s};
use serde::Serialize;

use super::*;

/// One named, row-aligned non-anchor target in a manifold crosscoder fit.
#[derive(Clone, Debug)]
pub struct NamedCrosscoderTarget {
    pub label: String,
    pub target: Array2<f64>,
}

/// Pair the parallel representation used by array-oriented bindings without
/// allowing `zip` truncation at the boundary.
pub fn pair_crosscoder_targets(
    labels: Vec<String>,
    targets: Vec<Array2<f64>>,
) -> Result<Vec<NamedCrosscoderTarget>, String> {
    if labels.len() != targets.len() {
        return Err(format!(
            "pair_crosscoder_targets: labels length {} != targets length {}",
            labels.len(),
            targets.len()
        ));
    }
    Ok(labels
        .into_iter()
        .zip(targets)
        .map(|(label, target)| NamedCrosscoderTarget { label, target })
        .collect())
}

/// Fully typed request for the crosscoder schedule over the unified engine.
///
/// `base_term` must be seeded at the augmented width
/// `anchor.ncols() + sum(block.target.ncols())`.  The target matrices are kept
/// unscaled here; [`SaeManifoldOuterObjective::with_crosscoder_blocks`] owns the
/// idempotent `sqrt(lambda_l)` materialization at every rho evaluation.
pub struct SaeCrosscoderFitRequest {
    pub anchor_label: String,
    pub anchor: Array2<f64>,
    pub blocks: Vec<NamedCrosscoderTarget>,
    pub base_term: SaeManifoldTerm,
    pub registry: AnalyticPenaltyRegistry,
    pub initial_rho: SaeManifoldRho,
    pub max_iter: usize,
    pub learning_rate: f64,
    pub ridge_ext_coord: f64,
    pub ridge_beta: f64,
    pub run_outer_rho_search: bool,
    pub cancel: Option<Arc<AtomicBool>>,
}

/// Single source of truth for the automatic circle-crosscoder fit controls used
/// by the Python and CLI front doors. Callers may replace any field, but neither
/// binding owns a second set of defaults.
#[derive(Clone, Debug)]
pub struct SaeCrosscoderAutoFitConfig {
    pub n_atoms: usize,
    pub n_harmonics: usize,
    pub sparsity_strength: f64,
    pub smoothness: f64,
    pub max_iter: usize,
    pub learning_rate: f64,
    pub ridge_ext_coord: f64,
    pub ridge_beta: f64,
    pub random_state: u64,
    pub run_outer_rho_search: bool,
}

impl SaeCrosscoderAutoFitConfig {
    /// Established manifold-SAE defaults, centralized in the Rust owner. Atom
    /// count and harmonic order are structural choices and therefore required.
    pub fn standard(n_atoms: usize, n_harmonics: usize) -> Self {
        Self {
            n_atoms,
            n_harmonics,
            sparsity_strength: 1.0,
            smoothness: 1.0,
            max_iter: 50,
            learning_rate: 0.05,
            ridge_ext_coord: 1.0e-6,
            ridge_beta: 1.0e-6,
            random_state: 0,
            run_outer_rho_search: true,
        }
    }

    fn validate(&self) -> Result<(), String> {
        if self.n_atoms == 0 {
            return Err("SaeCrosscoderAutoFitConfig: n_atoms must be positive".to_string());
        }
        if self.n_harmonics == 0 {
            return Err("SaeCrosscoderAutoFitConfig: n_harmonics must be positive".to_string());
        }
        if self.max_iter == 0 {
            return Err("SaeCrosscoderAutoFitConfig: max_iter must be positive".to_string());
        }
        for (name, value) in [
            ("sparsity_strength", self.sparsity_strength),
            ("smoothness", self.smoothness),
        ] {
            if !value.is_finite() || value < 0.0 {
                return Err(format!(
                    "SaeCrosscoderAutoFitConfig: {name} must be finite and non-negative; got {value}"
                ));
            }
        }
        for (name, value) in [
            ("learning_rate", self.learning_rate),
            ("ridge_ext_coord", self.ridge_ext_coord),
            ("ridge_beta", self.ridge_beta),
        ] {
            if !value.is_finite() || value <= 0.0 {
                return Err(format!(
                    "SaeCrosscoderAutoFitConfig: {name} must be finite and positive; got {value}"
                ));
            }
        }
        Ok(())
    }
}

/// Optional binding/CLI overrides. Resolution onto the Rust-owned standard
/// config happens here so every front door has identical defaults.
#[derive(Clone, Debug, Default)]
pub struct SaeCrosscoderAutoFitOverrides {
    pub sparsity_strength: Option<f64>,
    pub smoothness: Option<f64>,
    pub max_iter: Option<usize>,
    pub learning_rate: Option<f64>,
    pub ridge_ext_coord: Option<f64>,
    pub ridge_beta: Option<f64>,
    pub random_state: Option<u64>,
    pub run_outer_rho_search: Option<bool>,
}

impl SaeCrosscoderAutoFitOverrides {
    pub fn resolve(self, n_atoms: usize, n_harmonics: usize) -> SaeCrosscoderAutoFitConfig {
        let mut config = SaeCrosscoderAutoFitConfig::standard(n_atoms, n_harmonics);
        if let Some(value) = self.sparsity_strength {
            config.sparsity_strength = value;
        }
        if let Some(value) = self.smoothness {
            config.smoothness = value;
        }
        if let Some(value) = self.max_iter {
            config.max_iter = value;
        }
        if let Some(value) = self.learning_rate {
            config.learning_rate = value;
        }
        if let Some(value) = self.ridge_ext_coord {
            config.ridge_ext_coord = value;
        }
        if let Some(value) = self.ridge_beta {
            config.ridge_beta = value;
        }
        if let Some(value) = self.random_state {
            config.random_state = value;
        }
        if let Some(value) = self.run_outer_rho_search {
            config.run_outer_rho_search = value;
        }
        config
    }
}

/// Automatic crosscoder request shared by non-Rust front doors. It owns only
/// row-aligned activations and one Rust-owned config; seed construction and all
/// model policy stay below the bindings.
pub struct SaeCrosscoderAutoFitRequest {
    pub anchor_label: String,
    pub anchor: Array2<f64>,
    pub blocks: Vec<NamedCrosscoderTarget>,
    pub config: SaeCrosscoderAutoFitConfig,
    pub cancel: Option<Arc<AtomicBool>>,
}

/// Optional scientific measurements to materialize from a completed fit.
/// Transport is not run implicitly: its grid resolution is a caller-owned
/// experimental resolution, and its law threshold is an optional claim rule.
#[derive(Clone, Copy, Debug, Default)]
pub struct SaeCrosscoderEvaluationConfig {
    pub transport_grid_resolution: Option<usize>,
    pub law_gap_tolerance: Option<f64>,
}

/// Honest-unit reconstruction and per-atom decoders for one fitted layer.
#[derive(Clone, Debug)]
pub struct CrosscoderLayerFit {
    pub label: String,
    pub target: Array2<f64>,
    pub fitted: Array2<f64>,
    pub reconstruction_r2: f64,
    pub decoders: Vec<Array2<f64>>,
}

/// Drift is defined only when consecutive layers share an ambient width.
/// Ragged crosscoders remain valid fits, but cannot manufacture a Frobenius
/// difference or principal angle between matrices in different ambient spaces.
#[derive(Clone, Debug)]
pub enum CrosscoderDriftStatus {
    Measured(CrosscoderDriftReport),
    Undefined { reason: String },
}

/// Completed crosscoder fit.  `term` retains the engine's scaled decoder
/// parameterization and has `layout` installed; `layers` is the public,
/// honest-unit view in anchor/block order.
pub struct SaeCrosscoderFitReport {
    pub term: SaeManifoldTerm,
    pub rho: SaeManifoldRho,
    pub loss: SaeManifoldLoss,
    pub termination: SaeOuterTermination,
    pub layout: CrosscoderLayout,
    pub layers: Vec<CrosscoderLayerFit>,
    pub drift: CrosscoderDriftStatus,
}

#[derive(Clone, Debug, Serialize)]
pub struct SaeCrosscoderWireLayout {
    pub anchor_label: String,
    pub anchor_dim: usize,
    pub block_dims: Vec<usize>,
    pub labels: Vec<String>,
    pub log_lambda_block: Vec<f64>,
}

#[derive(Clone, Debug, Serialize)]
pub struct SaeCrosscoderWireLayer {
    pub label: String,
    pub fitted: Vec<Vec<f64>>,
    pub reconstruction_r2: f64,
    pub decoders: Vec<Vec<Vec<f64>>>,
}

#[derive(Clone, Debug, Serialize)]
pub struct SaeCrosscoderWireLoss {
    pub data_fit: f64,
    pub assignment_sparsity: f64,
    pub smoothness: f64,
    pub ard: f64,
    pub total_penalized_loss: f64,
}

#[derive(Clone, Debug, Serialize)]
pub struct SaeCrosscoderWireTermination {
    pub verdict: String,
    pub evals: u64,
    pub evals_since_improvement: u64,
    pub wall_seconds: f64,
}

#[derive(Clone, Debug, Serialize)]
pub struct SaeCrosscoderWireDriftStep {
    pub atom: usize,
    pub source: String,
    pub target: String,
    pub drift: f64,
    pub principal_angles: Vec<f64>,
    pub max_principal_angle: f64,
}

#[derive(Clone, Debug, Serialize)]
#[serde(tag = "status", rename_all = "snake_case")]
pub enum SaeCrosscoderWireDrift {
    Measured {
        num_atoms: usize,
        mean_drift: f64,
        most_drifting_atom: Option<usize>,
        most_stable_atom: Option<usize>,
        layer_chain: Vec<String>,
        steps: Vec<SaeCrosscoderWireDriftStep>,
    },
    Undefined {
        reason: String,
    },
}

#[derive(Clone, Debug, Serialize)]
pub struct SaeCrosscoderWireTransport {
    pub atom: usize,
    pub source: String,
    pub target: String,
    pub grid_resolution: usize,
    pub n_harmonics: usize,
    pub phase_shift: (f64, f64),
    pub phase_r2: f64,
    pub smooth_r2: f64,
    pub law_gap: f64,
    pub law_holds: Option<bool>,
    pub deviation_locus: Option<f64>,
    pub drift: f64,
    pub principal_angles: Vec<f64>,
    pub transport_grid: Vec<(f64, f64)>,
}

/// Stable, binding-neutral report shape owned by GAM-SAE. pyffi and CLI only
/// serialize this value; neither derives diagnostics or chooses measurements.
#[derive(Clone, Debug, Serialize)]
pub struct SaeCrosscoderWireReport {
    pub layout: SaeCrosscoderWireLayout,
    pub log_lambda_block: Vec<f64>,
    pub log_lambda_sparse: f64,
    pub log_lambda_smooth: Vec<f64>,
    pub assignments: Vec<Vec<f64>>,
    pub logits: Vec<Vec<f64>>,
    pub coords: Vec<Vec<Vec<f64>>>,
    pub loss: SaeCrosscoderWireLoss,
    pub termination: SaeCrosscoderWireTermination,
    pub layers: Vec<SaeCrosscoderWireLayer>,
    pub drift: SaeCrosscoderWireDrift,
    pub transport: Vec<SaeCrosscoderWireTransport>,
    /// Serialization unit consumed by `ManifoldSaePayload::crosscoder`.
    pub crosscoder: SaeCrosscoderWirePersistence,
}

#[derive(Clone, Debug, Serialize)]
pub struct SaeCrosscoderWirePersistence {
    pub anchor_label: String,
    pub anchor_dim: usize,
    pub block_dims: Vec<usize>,
    pub labels: Vec<String>,
    pub log_lambda_block: Vec<f64>,
    pub drift: SaeCrosscoderWireDrift,
    pub transport: Vec<SaeCrosscoderWireTransport>,
}

fn validate_label(label: &str, role: &str) -> Result<(), String> {
    if label.trim().is_empty() {
        return Err(format!(
            "run_sae_crosscoder_fit: {role} label must be non-empty"
        ));
    }
    Ok(())
}

/// Validate and stack unscaled crosscoder targets in
/// `[anchor | block_0 | ...]` order.  This allocation happens exactly once;
/// block-weight movement thereafter is in-place from the objective's pristine
/// copy.
pub fn stack_crosscoder_targets(
    anchor_label: &str,
    anchor: &Array2<f64>,
    blocks: &[NamedCrosscoderTarget],
) -> Result<(Array2<f64>, Vec<usize>, Vec<String>), String> {
    validate_label(anchor_label, "anchor")?;
    let (n, p_x) = anchor.dim();
    if n == 0 || p_x == 0 {
        return Err(format!(
            "run_sae_crosscoder_fit: anchor must be non-empty; got shape ({n}, {p_x})"
        ));
    }
    if !anchor.iter().all(|value| value.is_finite()) {
        return Err("run_sae_crosscoder_fit: anchor contains non-finite values".to_string());
    }
    if blocks.is_empty() {
        return Err(
            "run_sae_crosscoder_fit: at least one named non-anchor target is required".to_string(),
        );
    }

    let mut labels = Vec::with_capacity(blocks.len());
    let mut dims = Vec::with_capacity(blocks.len());
    let mut total_dim = p_x;
    let mut seen = std::collections::BTreeSet::new();
    seen.insert(anchor_label.to_string());
    for (index, block) in blocks.iter().enumerate() {
        validate_label(&block.label, &format!("block {index}"))?;
        if !seen.insert(block.label.clone()) {
            return Err(format!(
                "run_sae_crosscoder_fit: layer label {:?} is duplicated",
                block.label
            ));
        }
        let (block_n, block_p) = block.target.dim();
        if block_n != n {
            return Err(format!(
                "run_sae_crosscoder_fit: block {index} ({:?}) has {block_n} rows; expected the anchor's {n} row-aligned observations",
                block.label
            ));
        }
        if block_p == 0 {
            return Err(format!(
                "run_sae_crosscoder_fit: block {index} ({:?}) has zero columns",
                block.label
            ));
        }
        if !block.target.iter().all(|value| value.is_finite()) {
            return Err(format!(
                "run_sae_crosscoder_fit: block {index} ({:?}) contains non-finite values",
                block.label
            ));
        }
        total_dim = total_dim.checked_add(block_p).ok_or_else(|| {
            "run_sae_crosscoder_fit: augmented target width overflowed usize".to_string()
        })?;
        labels.push(block.label.clone());
        dims.push(block_p);
    }

    let mut stacked = Array2::<f64>::zeros((n, total_dim));
    stacked.slice_mut(s![.., 0..p_x]).assign(anchor);
    let mut offset = p_x;
    for block in blocks {
        let width = block.target.ncols();
        stacked
            .slice_mut(s![.., offset..offset + width])
            .assign(&block.target);
        offset += width;
    }
    Ok((stacked, dims, labels))
}

fn reconstruction_r2(target: &Array2<f64>, fitted: &Array2<f64>) -> Result<f64, String> {
    if target.dim() != fitted.dim() {
        return Err(format!(
            "crosscoder reconstruction R2 shape mismatch: target {:?}, fitted {:?}",
            target.dim(),
            fitted.dim()
        ));
    }
    let (n, p) = target.dim();
    let mut means = vec![0.0; p];
    for row in target.rows() {
        for j in 0..p {
            means[j] += row[j];
        }
    }
    for mean in &mut means {
        *mean /= n as f64;
    }
    let mut rss = 0.0;
    let mut tss = 0.0;
    for i in 0..n {
        for j in 0..p {
            let residual = target[[i, j]] - fitted[[i, j]];
            let centered = target[[i, j]] - means[j];
            rss += residual * residual;
            tss += centered * centered;
        }
    }
    Ok(if tss > 0.0 { 1.0 - rss / tss } else { f64::NAN })
}

/// Build the production circle seed and run the typed crosscoder schedule. This
/// is the one automatic front door shared by Python and CLI.
pub fn run_auto_sae_crosscoder_fit(
    request: SaeCrosscoderAutoFitRequest,
) -> Result<SaeCrosscoderFitReport, SaeFitError> {
    request.config.validate()?;
    let (stacked, _, _) =
        stack_crosscoder_targets(&request.anchor_label, &request.anchor, &request.blocks)?;
    let assignment = SaeFitAssignmentKind::Softmax;
    let minimal = build_sae_minimal_seed(SaeMinimalSeedRequest {
        target: stacked.view(),
        atom_basis: vec!["periodic".to_string(); request.config.n_atoms],
        atom_dim: vec![request.config.n_harmonics; request.config.n_atoms],
        assignment_kind: assignment,
        alpha: 1.0,
        tau: 1.0,
        threshold: 0.0,
        top_k: None,
        ibp_alpha_override: None,
        random_state: request.config.random_state,
        initial_logits: None,
        initial_coords: None,
    })?;
    let SaeMinimalSeedReport {
        atom_basis,
        effective_atom_dim,
        atom_centers,
        basis_values,
        basis_jacobian,
        basis_sizes,
        decoder_coefficients,
        smooth_penalties,
        initial_logits,
        initial_coords,
        refine_routing,
    } = minimal;
    let registry = AnalyticPenaltyRegistry::new();
    let seed = build_sae_fit_seed(SaeFitSeedRequest {
        target: stacked.view(),
        atom_basis: &atom_basis,
        atom_dim: &effective_atom_dim,
        atom_centers: &atom_centers,
        basis_values: basis_values.view(),
        basis_jacobian: basis_jacobian.view(),
        basis_sizes: &basis_sizes,
        decoder_coefficients: decoder_coefficients.view(),
        smooth_penalties: smooth_penalties.view(),
        initial_logits: initial_logits.view(),
        initial_coords: initial_coords.view(),
        alpha: 1.0,
        tau: 1.0,
        learnable_alpha: false,
        assignment_kind: assignment,
        sparsity_strength: request.config.sparsity_strength,
        smoothness: request.config.smoothness,
        max_iter: request.config.max_iter,
        learning_rate: request.config.learning_rate,
        ridge_ext_coord: request.config.ridge_ext_coord,
        ridge_beta: request.config.ridge_beta,
        top_k: None,
        threshold: 0.0,
        native_ard_enabled: true,
        seed_refine_routing: refine_routing,
        seed_refine_random_state: request.config.random_state,
        data_row_reseed: false,
        fit_config: SaeFitConfig::default(),
        temperature_schedule: None,
        fisher_metric: None,
        row_loss_weights: None,
        registry: &registry,
    })?;
    run_sae_crosscoder_fit(SaeCrosscoderFitRequest {
        anchor_label: request.anchor_label,
        anchor: request.anchor,
        blocks: request.blocks,
        base_term: seed.base_term,
        registry,
        initial_rho: seed.initial_rho,
        max_iter: request.config.max_iter,
        learning_rate: request.config.learning_rate,
        ridge_ext_coord: request.config.ridge_ext_coord,
        ridge_beta: request.config.ridge_beta,
        run_outer_rho_search: request.config.run_outer_rho_search,
        cancel: request.cancel,
    })
}

fn array2_to_nested(array: &Array2<f64>) -> Vec<Vec<f64>> {
    array.rows().into_iter().map(|row| row.to_vec()).collect()
}

fn wire_layer_label(
    layer: CrosscoderLayer,
    anchor_label: &str,
    block_labels: &[String],
) -> Result<String, String> {
    match layer {
        CrosscoderLayer::Anchor => Ok(anchor_label.to_string()),
        CrosscoderLayer::Block(index) => block_labels
            .get(index)
            .cloned()
            .ok_or_else(|| format!("crosscoder wire report: block layer {index} is out of range")),
    }
}

impl SaeCrosscoderFitReport {
    pub fn layer_from_label(&self, label: &str) -> Result<CrosscoderLayer, String> {
        let anchor = self
            .layers
            .first()
            .ok_or_else(|| "crosscoder report has no anchor layer".to_string())?;
        if label == anchor.label {
            return Ok(CrosscoderLayer::Anchor);
        }
        self.layout
            .labels()
            .iter()
            .position(|candidate| candidate == label)
            .map(CrosscoderLayer::Block)
            .ok_or_else(|| format!("crosscoder layer label {label:?} is not fitted"))
    }

    pub fn steer_layer_delta(
        &self,
        atom: usize,
        layer_label: &str,
        rows: &[usize],
        delta: ndarray::ArrayView1<'_, f64>,
    ) -> Result<Array2<f64>, String> {
        self.term.steer_layer_delta(
            atom,
            self.layer_from_label(layer_label)?,
            rows,
            delta,
        )
    }

    pub fn steer_layer_decode(
        &self,
        atom: usize,
        layer_label: &str,
        rows: &[usize],
        delta: ndarray::ArrayView1<'_, f64>,
    ) -> Result<Array2<f64>, String> {
        self.term.steer_layer_decode(
            atom,
            self.layer_from_label(layer_label)?,
            rows,
            delta,
        )
    }

    /// Materialize the stable report shared by bindings. The optional transport
    /// experiment is evaluated here, not in pyffi/CLI.
    pub fn wire_report(
        &self,
        evaluation: SaeCrosscoderEvaluationConfig,
    ) -> Result<SaeCrosscoderWireReport, String> {
        if let Some(tolerance) = evaluation.law_gap_tolerance {
            if !tolerance.is_finite() || tolerance < 0.0 {
                return Err(format!(
                    "SaeCrosscoderEvaluationConfig: law_gap_tolerance must be finite and non-negative; got {tolerance}"
                ));
            }
            if evaluation.transport_grid_resolution.is_none() {
                return Err(
                    "SaeCrosscoderEvaluationConfig: law_gap_tolerance requires a transport grid"
                        .to_string(),
                );
            }
        }
        let anchor_label = self
            .layers
            .first()
            .map(|layer| layer.label.as_str())
            .ok_or_else(|| "crosscoder report has no anchor layer".to_string())?;
        let block_labels = self.layout.labels();
        let layout = SaeCrosscoderWireLayout {
            anchor_label: anchor_label.to_string(),
            anchor_dim: self.layout.anchor_dim(),
            block_dims: self.layout.block_dims().to_vec(),
            labels: block_labels.to_vec(),
            log_lambda_block: self.layout.block_log_lambda().to_vec(),
        };
        let drift = match &self.drift {
            CrosscoderDriftStatus::Measured(report) => {
                let layer_chain = report
                    .layer_chain
                    .iter()
                    .map(|&layer| wire_layer_label(layer, anchor_label, block_labels))
                    .collect::<Result<Vec<_>, _>>()?;
                let steps = report
                    .steps
                    .iter()
                    .map(|step| {
                        Ok(SaeCrosscoderWireDriftStep {
                            atom: step.atom,
                            source: wire_layer_label(step.source, anchor_label, block_labels)?,
                            target: wire_layer_label(step.target, anchor_label, block_labels)?,
                            drift: step.drift,
                            principal_angles: step.principal_angles.clone(),
                            max_principal_angle: step.max_principal_angle(),
                        })
                    })
                    .collect::<Result<Vec<_>, String>>()?;
                SaeCrosscoderWireDrift::Measured {
                    num_atoms: report.num_atoms,
                    mean_drift: report.mean_drift(),
                    most_drifting_atom: report.most_drifting_atom(),
                    most_stable_atom: report.most_stable_atom(),
                    layer_chain,
                    steps,
                }
            }
            CrosscoderDriftStatus::Undefined { reason } => SaeCrosscoderWireDrift::Undefined {
                reason: reason.clone(),
            },
        };
        let mut transport = Vec::new();
        if let Some(grid_resolution) = evaluation.transport_grid_resolution {
            let chain: Vec<CrosscoderLayer> = std::iter::once(CrosscoderLayer::Anchor)
                .chain((0..self.layout.num_blocks()).map(CrosscoderLayer::Block))
                .collect();
            for atom in 0..self.term.k_atoms() {
                for pair in chain.windows(2) {
                    let measured = measure_atom_transport_between(
                        &self.term,
                        &self.layout,
                        atom,
                        pair[0],
                        pair[1],
                        grid_resolution,
                    )?;
                    transport.push(SaeCrosscoderWireTransport {
                        atom,
                        source: wire_layer_label(pair[0], anchor_label, block_labels)?,
                        target: wire_layer_label(pair[1], anchor_label, block_labels)?,
                        grid_resolution: measured.grid_resolution,
                        n_harmonics: measured.n_harmonics,
                        phase_shift: measured.phase_shift,
                        phase_r2: measured.phase_r2,
                        smooth_r2: measured.smooth_r2,
                        law_gap: measured.law_gap(),
                        law_holds: evaluation
                            .law_gap_tolerance
                            .map(|tolerance| measured.law_holds(tolerance)),
                        deviation_locus: measured.deviation_locus(),
                        drift: measured.drift,
                        principal_angles: measured.principal_angles,
                        transport_grid: measured.transport_grid,
                    });
                }
            }
        }
        let crosscoder = SaeCrosscoderWirePersistence {
            anchor_label: layout.anchor_label.clone(),
            anchor_dim: layout.anchor_dim,
            block_dims: layout.block_dims.clone(),
            labels: layout.labels.clone(),
            log_lambda_block: layout.log_lambda_block.clone(),
            drift: drift.clone(),
            transport: transport.clone(),
        };
        Ok(SaeCrosscoderWireReport {
            layout,
            log_lambda_block: self.rho.log_lambda_block.clone(),
            log_lambda_sparse: self.rho.log_lambda_sparse,
            log_lambda_smooth: self.rho.log_lambda_smooth.clone(),
            assignments: array2_to_nested(&self.term.assignment.assignments()),
            logits: array2_to_nested(&self.term.assignment.logits),
            coords: self
                .term
                .assignment
                .coords
                .iter()
                .map(|coord| array2_to_nested(&coord.as_matrix()))
                .collect(),
            loss: SaeCrosscoderWireLoss {
                data_fit: self.loss.data_fit,
                assignment_sparsity: self.loss.assignment_sparsity,
                smoothness: self.loss.smoothness,
                ard: self.loss.ard,
                total_penalized_loss: self.loss.total(),
            },
            termination: SaeCrosscoderWireTermination {
                verdict: self.termination.verdict.as_str().to_string(),
                evals: self.termination.evals,
                evals_since_improvement: self.termination.evals_since_improvement,
                wall_seconds: self.termination.wall.as_secs_f64(),
            },
            layers: self
                .layers
                .iter()
                .map(|layer| SaeCrosscoderWireLayer {
                    label: layer.label.clone(),
                    fitted: array2_to_nested(&layer.fitted),
                    reconstruction_r2: layer.reconstruction_r2,
                    decoders: layer.decoders.iter().map(array2_to_nested).collect(),
                })
                .collect(),
            drift,
            transport,
            crosscoder,
        })
    }
}

fn certify_crosscoder_outer(
    objective: SaeManifoldOuterObjective,
    result: Result<OuterResult, gam_problem::EstimationError>,
) -> Result<SaeManifoldOuterObjective, SaeFitError> {
    super::fit_entry::certify_outer_stage(objective, SaeFitStage::Primary, result)
}

/// Fit a multi-layer manifold crosscoder through the unified outer objective.
pub fn run_sae_crosscoder_fit(
    mut request: SaeCrosscoderFitRequest,
) -> Result<SaeCrosscoderFitReport, SaeFitError> {
    let (stacked, block_dims, labels) =
        stack_crosscoder_targets(&request.anchor_label, &request.anchor, &request.blocks)?;
    let p_x = request.anchor.ncols();
    if request.base_term.output_dim() != stacked.ncols() {
        return Err(SaeFitError::Fit(format!(
            "run_sae_crosscoder_fit: base term output_dim {} != augmented target width {}",
            request.base_term.output_dim(),
            stacked.ncols()
        )));
    }
    if request.initial_rho.log_lambda_block.is_empty() {
        request.initial_rho.log_lambda_block = vec![0.0; block_dims.len()];
    }
    if request.initial_rho.log_lambda_block.len() != block_dims.len() {
        return Err(SaeFitError::Fit(format!(
            "run_sae_crosscoder_fit: initial rho has {} block coordinates; expected {}",
            request.initial_rho.log_lambda_block.len(),
            block_dims.len()
        )));
    }
    if !request
        .initial_rho
        .log_lambda_block
        .iter()
        .all(|value| value.is_finite())
    {
        return Err(SaeFitError::Fit(
            "run_sae_crosscoder_fit: initial block log lambdas must be finite".to_string(),
        ));
    }

    let initial_flat = request.initial_rho.to_flat();
    let n_params = initial_flat.len();
    let cancel = request
        .cancel
        .unwrap_or_else(|| Arc::new(AtomicBool::new(false)));
    let mut objective = SaeManifoldOuterObjective::new(
        request.base_term,
        stacked,
        Some(request.registry),
        request.initial_rho,
        request.max_iter,
        request.learning_rate,
        request.ridge_ext_coord,
        request.ridge_beta,
    )
    .with_crosscoder_blocks(p_x, block_dims.clone())?;
    super::fit_entry::scope_outer_checkpoint_to_stage(&mut objective, SaeFitStage::Primary);
    objective.set_cancel_flag(cancel);

    let objective = if request.run_outer_rho_search {
        let search_initial = match objective.try_resume_from_checkpoint(n_params) {
            Some(banked) => ndarray::Array1::from(banked),
            None => initial_flat,
        };
        let problem = OuterProblem::new(n_params)
            .with_initial_rho(search_initial)
            .with_seed_config(SeedConfig {
                max_seeds: 1,
                seed_budget: 1,
                ..Default::default()
            });
        let result = problem.run(&mut objective, "SAE manifold crosscoder");
        certify_crosscoder_outer(objective, result)?
    } else {
        objective.fit_at_fixed_rho(initial_flat.view())?;
        objective
    };
    objective.remove_checkpoint();
    let fitted_result = objective.into_fitted();
    let mut term = fitted_result.term;
    let rho = fitted_result.rho;
    let loss = fitted_result.loss;
    let termination = fitted_result.termination;
    let layout = CrosscoderLayout::new(p_x, block_dims, labels, rho.log_lambda_block.clone())?;
    term.set_crosscoder_layout(layout.clone())?;

    let scaled_fitted = term.try_fitted()?;
    let mut layers = Vec::with_capacity(1 + request.blocks.len());
    let anchor_fitted = scaled_fitted.slice(s![.., 0..p_x]).to_owned();
    let anchor_decoders = term
        .atoms
        .iter()
        .map(|atom| {
            atom.physical_full_width_decoder()
                .slice(s![.., 0..p_x])
                .to_owned()
        })
        .collect();
    layers.push(CrosscoderLayerFit {
        label: request.anchor_label,
        reconstruction_r2: reconstruction_r2(&request.anchor, &anchor_fitted)?,
        target: request.anchor,
        fitted: anchor_fitted,
        decoders: anchor_decoders,
    });
    for (block_index, block) in request.blocks.into_iter().enumerate() {
        let scale = layout.sqrt_lambda(block_index);
        let honest_fitted = scaled_fitted
            .slice(s![.., layout.block_range(block_index)])
            .mapv(|value| value / scale);
        let decoders = (0..term.k_atoms())
            .map(|atom| term.layer_decoder(atom, block_index))
            .collect::<Result<Vec<_>, _>>()?;
        layers.push(CrosscoderLayerFit {
            label: block.label,
            reconstruction_r2: reconstruction_r2(&block.target, &honest_fitted)?,
            target: block.target,
            fitted: honest_fitted,
            decoders,
        });
    }

    let drift = match measure_crosscoder_drift(&term, &layout) {
        Ok(report) => CrosscoderDriftStatus::Measured(report),
        Err(reason)
            if layers
                .windows(2)
                .any(|pair| pair[0].target.ncols() != pair[1].target.ncols()) =>
        {
            CrosscoderDriftStatus::Undefined { reason }
        }
        Err(reason) => return Err(SaeFitError::Fit(reason)),
    };

    Ok(SaeCrosscoderFitReport {
        term,
        rho,
        loss,
        termination,
        layout,
        layers,
        drift,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stacking_is_unscaled_ordered_and_validated() {
        let anchor = ndarray::array![[1.0, 2.0], [3.0, 4.0]];
        let blocks = vec![
            NamedCrosscoderTarget {
                label: "middle".to_string(),
                target: ndarray::array![[5.0], [6.0]],
            },
            NamedCrosscoderTarget {
                label: "late".to_string(),
                target: ndarray::array![[7.0, 8.0], [9.0, 10.0]],
            },
        ];
        let (stacked, dims, labels) = stack_crosscoder_targets("early", &anchor, &blocks).unwrap();
        assert_eq!(
            stacked,
            ndarray::array![[1.0, 2.0, 5.0, 7.0, 8.0], [3.0, 4.0, 6.0, 9.0, 10.0]]
        );
        assert_eq!(dims, vec![1, 2]);
        assert_eq!(labels, vec!["middle", "late"]);
    }

    #[test]
    fn stacking_rejects_unaligned_or_duplicate_layers() {
        let anchor = Array2::<f64>::zeros((2, 2));
        let unaligned = vec![NamedCrosscoderTarget {
            label: "late".to_string(),
            target: Array2::<f64>::zeros((3, 2)),
        }];
        assert!(stack_crosscoder_targets("early", &anchor, &unaligned).is_err());
        let duplicate = vec![NamedCrosscoderTarget {
            label: "early".to_string(),
            target: Array2::<f64>::zeros((2, 2)),
        }];
        assert!(stack_crosscoder_targets("early", &anchor, &duplicate).is_err());
    }
}
