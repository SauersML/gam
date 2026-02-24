use crate::basis::{self, BasisOptions, Dense, KnotSource, create_basis};
use crate::construction::EngineLayout;
use crate::estimate::EstimationError;
use crate::hull::PeeledHull;
use crate::survival::{
    self, DEFAULT_RISK_EPSILON, SurvivalError, SurvivalModelArtifacts, SurvivalSpec,
};
pub use crate::types::LinkFunction;
use ndarray::{Array1, Array2, ArrayView1, ArrayView2, Axis, Zip, s};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs;
use std::io::{BufWriter, Write};
use thiserror::Error;

// --- Public Data Structures ---
// These structs define the public, human-readable format of the trained model
// when serialized to a TOML file.

/// Enumerates the supported working model families.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ModelFamily {
    Gam(LinkFunction),
    Survival(SurvivalSpec),
}

/// Configuration toggle for tensor-product interaction penalties.
///
/// * `Isotropic` matches the historical behavior with a single penalty parameter
///   using whitened marginal bases.
/// * `Anisotropic` builds separate penalties for each marginal direction using
///   the unwhitened bases. This is the new default.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum InteractionPenaltyKind {
    Isotropic,
    Anisotropic,
}

/// Configuration for a single basis expansion (e.g., for one variable).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BasisConfig {
    pub num_knots: usize,
    pub degree: usize,
}

/// Configuration for a single Principal Component, bundling related data together.
/// This makes invalid states unrepresentable by ensuring each PC has all required
/// configuration elements together.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PrincipalComponentConfig {
    /// Name of this principal component (e.g., "PC1")
    pub name: String,
    /// Basis configuration for this principal component
    pub basis_config: BasisConfig,
    /// Value range for this principal component
    pub range: (f64, f64),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SurvivalTimeVaryingConfig {
    #[serde(default)]
    pub label: Option<String>,
    pub pgs_basis: BasisConfig,
    pub pgs_penalty_order: usize,
    pub lambda_age: f64,
    pub lambda_pgs: f64,
    pub lambda_null: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SurvivalModelConfig {
    pub baseline_basis: BasisConfig,
    pub guard_delta: f64,
    pub monotonic_grid_size: usize,
    #[serde(default)]
    pub time_varying: Option<SurvivalTimeVaryingConfig>,
    #[serde(default)]
    pub model_competing_risk: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SurvivalRiskType {
    /// Hypothetical risk assuming no competing death (Net Risk).
    Net,
    /// Actuarial risk accounting for probability of death (Crude Risk).
    /// Requires `model_competing_risk = true` during training.
    Crude,
}

/// Holds the transformation matrix for a sum-to-zero constraint.
/// This is serializable so it can be saved to the TOML file.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Constraint {
    /// The Z matrix that transforms an unconstrained basis B to a constrained one B_c = B.dot(Z)
    pub z_transform: Array2<f64>,
}

pub fn default_reml_parallel_threshold() -> usize {
    4
}

pub fn default_mcmc_enabled() -> bool {
    true
}

pub fn default_calibrator_enabled() -> bool {
    true
}

/// The complete blueprint of a trained model.
/// Contains all hyperparameters and structural information needed for prediction.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelConfig {
    pub model_family: ModelFamily,
    pub penalty_order: usize,
    pub convergence_tolerance: f64,
    pub max_iterations: usize,
    pub reml_convergence_tolerance: f64,
    pub reml_max_iterations: u64,
    /// Firth bias reduction for logit - auto-detected in training if not set.
    /// Enabled when: prevalence < 10%, n < 500, or p/n > 0.1
    #[serde(default)]
    pub firth_bias_reduction: bool,
    /// Minimum number of penalties required before enabling REML gradient parallelization.
    /// Set to zero to force sequential execution regardless of penalty count.
    #[serde(default = "default_reml_parallel_threshold")]
    pub reml_parallel_threshold: usize,
    pub pgs_basis_config: BasisConfig,
    /// Bundled configuration for each principal component.
    /// This ensures that for each PC, we have the correct basis config, name, and range together.
    pub pc_configs: Vec<PrincipalComponentConfig>,
    // Data-dependent parameters saved from training are crucial for prediction.
    pub pgs_range: (f64, f64),
    pub interaction_penalty: InteractionPenaltyKind,

    /// Sum-to-zero constraint transformations (separate from range transforms)
    /// Maps term names (e.g., "pgs_main") to their sum-to-zero transformation matrices
    pub sum_to_zero_constraints: HashMap<String, Array2<f64>>,
    /// Knot vectors used during training, required for exact reproduction during prediction
    pub knot_vectors: HashMap<String, Array1<f64>>,
    /// Range transformation matrices for functional ANOVA decomposition
    /// Maps variable names (e.g., "pgs", "PC1") to their range-space transformation matrices
    pub range_transforms: HashMap<String, Array2<f64>>,
    /// Weighted column means used for centering interaction marginals during training
    /// Maps variable names (e.g., "pgs", "PC1") to their weighted column means for ANOVA centering
    pub interaction_centering_means: HashMap<String, Array1<f64>>,
    /// Linear maps that orthogonalize each interaction block against the main effects (pure ti())
    /// Keyed by interaction term name (e.g., "f(PGS,PC1)"). Shape: m_cols x t_cols.
    pub interaction_orth_alpha: HashMap<String, Array2<f64>>,
    /// Null-space transformation matrices for PC main effects (unpenalized part).
    /// Maps PC name to Z_null (columns span the penalty null space after dropping intercept).
    pub pc_null_transforms: HashMap<String, Array2<f64>>,
    /// Enable MCMC posterior sampling after PIRLS (expensive but honest uncertainty quantification).
    /// When true, runs NUTS sampler to generate posterior samples stored in TrainedModel.
    #[serde(default)]
    pub mcmc_enabled: bool,
    /// Enable post-process calibration when a calibrator is available.
    #[serde(default = "default_calibrator_enabled")]
    pub calibrator_enabled: bool,
    #[serde(default)]
    pub survival: Option<SurvivalModelConfig>,
}

impl ModelConfig {
    /// Minimal configuration for external designs (calibrator adapter).
    /// Only the fields used by PIRLS/REML are populated; others are left empty.
    pub fn external(
        link: LinkFunction,
        reml_tol: f64,
        reml_max_iter: usize,
        firth_bias_reduction: bool,
    ) -> Self {
        ModelConfig {
            model_family: ModelFamily::Gam(link),
            penalty_order: 2,
            // Align PIRLS tolerance with external REML tolerance to avoid over-iterating
            // on small calibrator fits where the outer loop is already coarse.
            convergence_tolerance: reml_tol,
            max_iterations: 500,
            reml_convergence_tolerance: reml_tol,
            reml_max_iterations: reml_max_iter as u64,
            firth_bias_reduction,
            reml_parallel_threshold: default_reml_parallel_threshold(),
            pgs_basis_config: BasisConfig {
                num_knots: 0,
                degree: 0,
            },
            pc_configs: Vec::new(),
            pgs_range: (0.0, 1.0),
            interaction_penalty: InteractionPenaltyKind::Anisotropic,
            sum_to_zero_constraints: HashMap::new(),
            knot_vectors: HashMap::new(),
            range_transforms: HashMap::new(),
            interaction_centering_means: HashMap::new(),
            interaction_orth_alpha: HashMap::new(),
            pc_null_transforms: HashMap::new(),
            mcmc_enabled: true, // Honest uncertainty by default
            calibrator_enabled: true,
            survival: None,
        }
    }

    pub fn link_function(&self) -> Result<LinkFunction, &'static str> {
        match &self.model_family {
            ModelFamily::Gam(link) => Ok(*link),
            ModelFamily::Survival(_) => {
                Err("link_function is not applicable for survival model family")
            }
        }
    }

    pub fn survival_spec(&self) -> Option<SurvivalSpec> {
        match &self.model_family {
            ModelFamily::Gam(_) => None,
            ModelFamily::Survival(spec) => Some(*spec),
        }
    }
}

/// A structured representation of the fitted model coefficients, designed for
/// human interpretation and sharing. This structure is used in the TOML file.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MappedCoefficients {
    pub intercept: f64,
    pub main_effects: MainEffects,
    /// Flat map for tensor product interaction terms.
    /// - Key: Interaction term name (e.g., "f(PGS,PC1)").
    /// - Value: The vector of coefficients for the entire tensor product surface.
    pub interaction_effects: HashMap<String, Vec<f64>>,
}

impl Default for MappedCoefficients {
    fn default() -> Self {
        Self {
            intercept: 0.0,
            main_effects: MainEffects::default(),
            interaction_effects: HashMap::new(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MainEffects {
    /// Coefficient for the main effect of sex (binary indicator).
    #[serde(default)]
    pub sex: f64,
    /// Coefficients for the main effect of PGS (for basis functions m > 0).
    pub pgs: Vec<f64>,
    /// Coefficients for the main effects of each PC, keyed by PC name.
    pub pcs: HashMap<String, Vec<f64>>,
}

impl Default for MainEffects {
    fn default() -> Self {
        Self {
            sex: 0.0,
            pgs: Vec::new(),
            pcs: HashMap::new(),
        }
    }
}

/// The top-level, self-contained, trained model artifact.
/// This is the structure that gets saved to and loaded from a file.
#[derive(Debug, Serialize, Deserialize)]
pub struct TrainedModel {
    pub config: ModelConfig,
    #[serde(default)]
    pub coefficients: MappedCoefficients,
    /// Estimated smoothing parameters from REML
    pub lambdas: Vec<f64>,
    /// Robust geometric clamping hull.
    #[serde(default)]
    pub hull: Option<PeeledHull>,
    /// Optional penalized Hessian (X'WX + S) at convergence in the model's coefficient order.
    /// When present, enables standard error estimates at prediction time via var = x^T H^{-1} x.
    #[serde(default)]
    pub penalized_hessian: Option<Array2<f64>>,
    /// Optional scale parameter (Gaussian identity link). Used for mean SE scaling when applicable.
    #[serde(default)]
    pub scale: Option<f64>,
    /// Optional post-fit calibration payload (adapter-owned in gnomon).
    #[serde(default)]
    pub calibrator: Option<()>,
    /// Optional joint link calibration for single-index models.
    #[serde(default)]
    pub joint_link: Option<JointLinkModel>,
    /// Optional survival-specific artifacts (present when training survival models).
    #[serde(default)]
    pub survival: Option<SurvivalModelArtifacts>,
    /// Optional registry of companion survival models keyed by handle reference.
    #[serde(default)]
    pub survival_companions: HashMap<String, SurvivalModelArtifacts>,
    /// Optional MCMC posterior samples [n_samples, n_coeffs] in original coefficient space.
    /// Present when mcmc_enabled=true during training. Enables honest uncertainty quantification.
    #[serde(default)]
    pub mcmc_samples: Option<Array2<f64>>,
    /// Optional smoothing parameter uncertainty correction matrix V_corr = J * V_ρ * J^T.
    /// When present, the variance estimate for predictions becomes: Var(η) = x^T H^{-1} x + x^T V_corr x.
    /// This implements the Wood et al. (2016) correction for smoothing parameter uncertainty,
    /// ensuring that standard errors properly account for the fact that λ was estimated from data.
    #[serde(default)]
    pub smoothing_correction: Option<Array2<f64>>,
}

/// Optional joint single-index link data for calibrated predictions.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JointLinkModel {
    pub knot_range: (f64, f64),
    pub knot_vector: Array1<f64>,
    pub link_transform: Array2<f64>,
    pub beta_link: Array1<f64>,
    pub degree: usize,
}

#[derive(Debug, Clone)]
pub struct SurvivalPrediction {
    pub cumulative_hazard_entry: Array1<f64>,
    pub cumulative_hazard_exit: Array1<f64>,
    pub cumulative_incidence_entry: Array1<f64>,
    pub cumulative_incidence_exit: Array1<f64>,
    pub conditional_risk: Array1<f64>,
    pub logit_risk: Array1<f64>,
    pub logit_risk_se: Option<Array1<f64>>,
    /// Jacobian matrix for logit risk (N × P). Optional to avoid memory waste
    /// for large cohorts when only risk scores are needed without SE computation.
    pub logit_risk_design: Option<Array2<f64>>,
}

/// Custom error type for model loading, saving, and prediction.
#[derive(Error, Debug)]
pub enum ModelError {
    #[error("Failed to read or write model file: {0}")]
    IoError(#[from] std::io::Error),
    #[error("Failed to parse TOML model file: {0}")]
    TomlParseError(#[from] toml::de::Error),
    #[error("Failed to serialize model to TOML format: {0}")]
    TomlSerializeError(#[from] toml::ser::Error),
    #[error("Prediction data has {found} PC columns, but the model was trained on {expected}.")]
    MismatchedPcCount { found: usize, expected: usize },
    #[error("Underlying basis function generation failed during prediction: {0}")]
    BasisError(#[from] basis::BasisError),
    #[error("Dimension mismatch: {0}")]
    DimensionMismatch(String),
    #[error(
        "Internal error: failed to stack design matrix columns or constraint matrix dimensions don't match basis dimensions during prediction."
    )]
    InternalStackingError,
    #[error(
        "Constraint transformation matrix missing for term '{0}'. This usually indicates a model format mismatch."
    )]
    ConstraintMissing(String),
    #[error("Coefficient vector for term '{0}' is missing from the model file.")]
    CoefficientMissing(String),

    #[error(
        "Calibrated prediction requested but no calibrator is present (disabled or missing from model)."
    )]
    CalibratorMissing,

    #[error("Estimation error: {0}")]
    EstimationError(#[from] crate::estimate::EstimationError),

    #[error("Survival prediction error: {0}")]
    SurvivalPrediction(#[from] SurvivalError),

    #[error("Survival artifacts are missing from the trained model.")]
    MissingSurvivalArtifacts,

    #[error("Operation '{0}' is not supported for survival model family.")]
    UnsupportedForSurvival(&'static str),
}

impl TrainedModel {
    fn ensure_gam_link(&self, operation: &'static str) -> Result<LinkFunction, ModelError> {
        match &self.config.model_family {
            ModelFamily::Gam(link) => Ok(*link),
            ModelFamily::Survival(_) => Err(ModelError::UnsupportedForSurvival(operation)),
        }
    }

    fn survival_artifacts(&self) -> Result<&SurvivalModelArtifacts, ModelError> {
        self.survival
            .as_ref()
            .ok_or(ModelError::MissingSurvivalArtifacts)
    }

    fn assemble_survival_covariates(
        &self,
        p_new: ArrayView1<f64>,
        sex_new: ArrayView1<f64>,
        pcs_new: ArrayView2<f64>,
        artifacts: &SurvivalModelArtifacts,
    ) -> Result<Array2<f64>, ModelError> {
        let n = p_new.len();
        if sex_new.len() != n {
            return Err(ModelError::DimensionMismatch(format!(
                "Sample count mismatch: p_new has {} samples but sex_new has {}",
                n,
                sex_new.len()
            )));
        }
        if pcs_new.nrows() != n {
            return Err(ModelError::DimensionMismatch(format!(
                "Sample count mismatch: pcs_new has {} rows but p_new has {} samples",
                pcs_new.nrows(),
                n
            )));
        }
        if pcs_new.ncols() != self.config.pc_configs.len() {
            return Err(ModelError::MismatchedPcCount {
                found: pcs_new.ncols(),
                expected: self.config.pc_configs.len(),
            });
        }

        let expected_cols = artifacts.static_covariate_layout.column_names.len();
        let required_cols = 2 + pcs_new.ncols();
        if expected_cols != required_cols {
            return Err(ModelError::DimensionMismatch(format!(
                "Survival covariate width mismatch: artifacts expect {} columns but received {} (2 + {} PCs)",
                expected_cols,
                required_cols,
                pcs_new.ncols()
            )));
        }

        let mut matrix = Array2::<f64>::zeros((n, expected_cols));
        for i in 0..n {
            matrix[[i, 0]] = p_new[i];
            matrix[[i, 1]] = sex_new[i];
            for j in 0..pcs_new.ncols() {
                matrix[[i, 2 + j]] = pcs_new[[i, j]];
            }
        }
        Ok(matrix)
    }

    fn build_prediction_design(
        &self,
        p_new: ArrayView1<f64>,
        sex_new: ArrayView1<f64>,
        pcs_new: ArrayView2<f64>,
    ) -> Result<(Array2<f64>, Array1<f64>), ModelError> {
        if pcs_new.ncols() != self.config.pc_configs.len() {
            return Err(ModelError::MismatchedPcCount {
                found: pcs_new.ncols(),
                expected: self.config.pc_configs.len(),
            });
        }

        let raw = internal::assemble_raw_from_p_and_pcs(p_new, pcs_new);
        let (signed_dist, x_corr) = if let Some(hull) = &self.hull {
            hull.signed_distance_and_project_many(raw.view())
        } else {
            (Array1::zeros(raw.nrows()), raw.clone())
        };
        let (p_corr, pcs_corr) = internal::split_p_and_pcs_from_raw(x_corr.view());

        let x_new = internal::construct_design_matrix(
            p_corr.view(),
            sex_new,
            pcs_corr.view(),
            &self.config,
            &self.coefficients,
        )?;

        Ok((x_new, signed_dist))
    }

    fn compute_se_eta_from_hessian(
        &self,
        x_new: &Array2<f64>,
        link_function: LinkFunction,
    ) -> Option<Array1<f64>> {
        let h = self.penalized_hessian.as_ref()?;
        if h.nrows() != h.ncols() || h.ncols() != x_new.ncols() {
            return None;
        }
        use crate::faer_ndarray::FaerCholesky;
        use faer::Side;
        let chol = match h.clone().cholesky(Side::Lower) {
            Ok(c) => c,
            Err(_) => return None,
        };

        // Check if smoothing correction is available and compatible
        let v_corr = self
            .smoothing_correction
            .as_ref()
            .filter(|v| v.nrows() == x_new.ncols() && v.ncols() == x_new.ncols());

        let mut vars = Array1::zeros(x_new.nrows());
        for i in 0..x_new.nrows() {
            let x_row = x_new.row(i).to_owned();
            let v = chol.solve_vec(&x_row);
            // Base variance: x^T H^{-1} x
            let var_base = x_row.dot(&v);

            // Add smoothing correction: x^T V_corr x (Wood et al. 2016)
            // This accounts for uncertainty in the smoothing parameter estimation.
            let var_corr = if let Some(corr) = v_corr {
                let corr_x = corr.dot(&x_row);
                x_row.dot(&corr_x)
            } else {
                0.0
            };

            // For Gaussian models, var_base is unitless and needs scaling by σ²,
            // but var_corr already has variance units (y²) because β has units of y.
            // Scaling var_corr again would give y⁴ units - a catastrophic error!
            vars[i] = if link_function == LinkFunction::Identity {
                let scale = self.scale.unwrap_or(1.0);
                (var_base * scale) + var_corr // Scale ONLY the base variance
            } else {
                // For GLMs, both terms already have correct units
                var_base + var_corr
            };
        }
        Some(vars.mapv(|v| v.max(0.0).sqrt()))
    }

    fn build_joint_link_basis(
        &self,
        eta_base: &Array1<f64>,
        joint: &JointLinkModel,
    ) -> Option<Array2<f64>> {
        if joint.beta_link.is_empty() {
            return None;
        }
        let (min_u, max_u) = joint.knot_range;
        let range_width = (max_u - min_u).max(1e-6);
        let z_raw: Array1<f64> = eta_base.mapv(|u| (u - min_u) / range_width);
        let z_c: Array1<f64> = z_raw.mapv(|z| z.clamp(0.0, 1.0));

        let (basis, _) = crate::basis::create_basis::<Dense>(
            z_c.view(),
            KnotSource::Provided(joint.knot_vector.view()),
            joint.degree,
            BasisOptions::value(),
        )
        .ok()?;
        let mut raw = basis.as_ref().clone();
        // Math note: to avoid a C^1 discontinuity from hard clamping, extend the basis
        // linearly outside [0,1]: B_ext(z_raw) = B(z_c) + (z_raw - z_c) * B'(z_c).
        let mut needs_ext = false;
        for i in 0..z_raw.len() {
            if (z_raw[i] - z_c[i]).abs() > 1e-12 {
                needs_ext = true;
                break;
            }
        }
        if needs_ext {
            if let Ok((b_prime_arc, _)) = crate::basis::create_basis::<Dense>(
                z_c.view(),
                KnotSource::Provided(joint.knot_vector.view()),
                joint.degree,
                BasisOptions::first_derivative(),
            ) {
                let b_prime = b_prime_arc.as_ref();
                for i in 0..z_raw.len() {
                    let dz = z_raw[i] - z_c[i];
                    if dz.abs() <= 1e-12 {
                        continue;
                    }
                    for j in 0..raw.ncols() {
                        raw[[i, j]] += dz * b_prime[[i, j]];
                    }
                }
            }
        }
        if joint.link_transform.nrows() != raw.ncols()
            || joint.link_transform.ncols() != joint.beta_link.len()
        {
            return None;
        }

        Some(raw.dot(&joint.link_transform))
    }

    fn compute_joint_link_derivative(
        &self,
        eta_base: &Array1<f64>,
        joint: &JointLinkModel,
    ) -> Option<Array1<f64>> {
        use crate::basis::evaluate_bspline_derivative_scalar_into;
        use crate::basis::internal::BsplineScratch;

        let n = eta_base.len();
        let mut deriv = Array1::<f64>::ones(n);
        if joint.beta_link.is_empty() {
            return Some(deriv);
        }

        let (min_u, max_u) = joint.knot_range;
        let range_width = (max_u - min_u).max(1e-6);
        let _z_raw: Array1<f64> = eta_base.mapv(|u| (u - min_u) / range_width);
        let z_c: Array1<f64> = _z_raw.mapv(|z| z.clamp(0.0, 1.0));

        let n_raw = joint.knot_vector.len().saturating_sub(joint.degree + 1);
        let n_constrained = joint.link_transform.ncols();
        if n_raw == 0 || n_constrained == 0 || joint.beta_link.len() != n_constrained {
            return None;
        }
        if joint.link_transform.nrows() != n_raw {
            return None;
        }

        let mut deriv_raw = vec![0.0; n_raw];
        let num_basis_lower = joint.knot_vector.len().saturating_sub(joint.degree);
        let mut lower_basis = vec![0.0; num_basis_lower];
        let mut lower_scratch = BsplineScratch::new(joint.degree.saturating_sub(1));

        for i in 0..n {
            // For the linear extension, d/dz_raw B_ext(z_raw) = B'(z_c) everywhere,
            // so we always evaluate B' at the clamped coordinate.
            let z_i = z_c[i];
            deriv_raw.fill(0.0);
            if evaluate_bspline_derivative_scalar_into(
                z_i,
                joint.knot_vector.view(),
                joint.degree,
                &mut deriv_raw,
                &mut lower_basis,
                &mut lower_scratch,
            )
            .is_err()
            {
                continue;
            }

            let mut d_wiggle_dz = 0.0;
            for c in 0..n_constrained {
                let mut b_prime_c = 0.0;
                for r in 0..n_raw {
                    b_prime_c += deriv_raw[r] * joint.link_transform[[r, c]];
                }
                d_wiggle_dz += b_prime_c * joint.beta_link[c];
            }
            deriv[i] = 1.0 + d_wiggle_dz / range_width;
        }

        Some(deriv)
    }

    /// Compute SE for joint model using the full joint Hessian.
    ///
    /// For joint models η = g(Xβ; θ) = Xβ + B(z)θ where z = (u - min)/(max - min) and u = Xβ,
    /// the full variance propagation is:
    ///
    ///   Var(η_i) = J_i^T H_joint^{-1} J_i
    ///
    /// where J_i = [g'(u_i) × x_i, b_i] is the Jacobian row.
    ///
    /// This properly accounts for uncertainty in BOTH β and θ, not just the simplified
    /// g'×SE approximation which ignores Var(θ).
    fn compute_joint_se_from_hessian(
        &self,
        x_new: &Array2<f64>,
        eta_base: &Array1<f64>,
        joint: &JointLinkModel,
        link_function: LinkFunction,
    ) -> Option<Array1<f64>> {
        let h = self.penalized_hessian.as_ref()?;
        let p_base = x_new.ncols();
        let p_link = joint.beta_link.len();
        let dim = p_base + p_link;

        // Check dimensions: joint Hessian should be (p_base + p_link) × (p_base + p_link)
        if h.nrows() != dim || h.ncols() != dim {
            // Fall back to base-only SE propagation if dimension mismatch
            return None;
        }

        use crate::faer_ndarray::FaerCholesky;
        use faer::Side;
        let chol = match h.clone().cholesky(Side::Lower) {
            Ok(c) => c,
            Err(_) => return None,
        };

        let n = x_new.nrows();
        let b_wiggle = self.build_joint_link_basis(eta_base, joint)?;
        if b_wiggle.ncols() != p_link {
            return None;
        }
        let g_prime = self
            .compute_joint_link_derivative(eta_base, joint)
            .unwrap_or_else(|| Array1::<f64>::ones(n));

        let mut vars = Array1::zeros(n);
        let mut j_row = Array1::<f64>::zeros(dim);

        for i in 0..n {
            j_row.fill(0.0);
            for j in 0..p_base {
                j_row[j] = g_prime[i] * x_new[[i, j]];
            }
            for j in 0..p_link {
                j_row[p_base + j] = b_wiggle[[i, j]];
            }

            let v = chol.solve_vec(&j_row);
            let var_i = j_row.dot(&v);
            vars[i] = if link_function == LinkFunction::Identity {
                if let Some(scale) = self.scale {
                    var_i * scale
                } else {
                    var_i
                }
            } else {
                var_i
            };
        }

        Some(vars.mapv(|v| v.max(0.0).sqrt()))
    }

    fn mcmc_mean_logit_predictions(&self, x_new: &Array2<f64>) -> Result<Array1<f64>, ModelError> {
        const ROW_CHUNK_SIZE: usize = 2048;
        let samples = self.mcmc_samples.as_ref().ok_or_else(|| {
            ModelError::DimensionMismatch("MCMC samples missing for logit prediction.".to_string())
        })?;
        if samples.nrows() == 0 {
            return Err(ModelError::DimensionMismatch(
                "MCMC samples are empty.".to_string(),
            ));
        }
        if samples.ncols() != x_new.ncols() {
            return Err(ModelError::DimensionMismatch(format!(
                "MCMC sample width {} does not match design columns {}",
                samples.ncols(),
                x_new.ncols()
            )));
        }

        let mut sum = Array1::<f64>::zeros(x_new.nrows());
        let samples_t = samples.t();
        let n_samples = samples.nrows() as f64;
        let mut start = 0;
        while start < x_new.nrows() {
            let end = (start + ROW_CHUNK_SIZE).min(x_new.nrows());
            let x_chunk = x_new.slice(s![start..end, ..]);
            let eta = x_chunk.dot(&samples_t);
            for (i, row) in eta.outer_iter().enumerate() {
                let mut acc = 0.0;
                for &e_raw in row.iter() {
                    let e = e_raw.clamp(-700.0, 700.0);
                    acc += 1.0 / (1.0 + f64::exp(-e));
                }
                sum[start + i] = acc;
            }
            start = end;
        }

        let scale = 1.0 / n_samples;
        let mut mean = sum.mapv(|v| v * scale);
        mean.mapv_inplace(|p| p.clamp(1e-8, 1.0 - 1e-8));
        Ok(mean)
    }

    /// Compute mean η and SE η directly from MCMC samples for coherent uncertainty.
    ///
    /// Unlike the previous approach which computed logit(E[σ(η)]) and paired it with
    /// Hessian-based SE (an incoherent pairing), this computes:
    ///   - mean_eta = E[η] = (1/S) Σ_s (X β_s)
    ///   - se_eta = sqrt(Var[η]) = sqrt((1/S) Σ_s (X β_s - mean_eta)²)
    ///
    /// This provides a coherent (mean, variance) summary on the logit scale.
    fn mcmc_mean_and_se_eta(
        &self,
        x_new: &Array2<f64>,
    ) -> Result<(Array1<f64>, Array1<f64>, Array1<f64>), ModelError> {
        const ROW_CHUNK_SIZE: usize = 2048;
        let samples = self.mcmc_samples.as_ref().ok_or_else(|| {
            ModelError::DimensionMismatch("MCMC samples missing for prediction.".to_string())
        })?;
        let n_samples = samples.nrows();
        if n_samples == 0 {
            return Err(ModelError::DimensionMismatch(
                "MCMC samples are empty.".to_string(),
            ));
        }
        if samples.ncols() != x_new.ncols() {
            return Err(ModelError::DimensionMismatch(format!(
                "MCMC sample width {} does not match design columns {}",
                samples.ncols(),
                x_new.ncols()
            )));
        }

        let n = x_new.nrows();
        let mut sum_eta = Array1::<f64>::zeros(n);
        let mut sum_eta_sq = Array1::<f64>::zeros(n);
        let mut sum_prob = Array1::<f64>::zeros(n);

        let samples_t = samples.t();
        let mut start = 0;
        while start < n {
            let end = (start + ROW_CHUNK_SIZE).min(n);
            let x_chunk = x_new.slice(s![start..end, ..]);
            let etas = x_chunk.dot(&samples_t); // [chunk_size, n_samples]

            for (i, row) in etas.outer_iter().enumerate() {
                let idx = start + i;
                for &eta_s in row.iter() {
                    let eta_clamped = eta_s.clamp(-700.0, 700.0);
                    sum_eta[idx] += eta_clamped;
                    sum_eta_sq[idx] += eta_clamped * eta_clamped;
                    sum_prob[idx] += 1.0 / (1.0 + f64::exp(-eta_clamped));
                }
            }
            start = end;
        }

        let scale = 1.0 / (n_samples as f64);
        let mean_eta = sum_eta.mapv(|v| v * scale);
        let mean_prob = sum_prob.mapv(|v| (v * scale).clamp(1e-8, 1.0 - 1e-8));

        // Var[η] = E[η²] - E[η]²
        let mut se_eta = Array1::<f64>::zeros(n);
        for i in 0..n {
            let var_eta = (sum_eta_sq[i] * scale) - (mean_eta[i] * mean_eta[i]);
            se_eta[i] = var_eta.max(0.0).sqrt();
        }

        Ok((mean_prob, mean_eta, se_eta))
    }

    /// Posterior mean probability using joint MCMC samples over (β, θ).
    /// For each sample [β_s | θ_s], computes η_s = Xβ_s + B(Xβ_s)θ_s
    /// and returns (mean_prob, mean_eta, se_eta) for proper uncertainty quantification.
    fn joint_mcmc_mean_predictions(
        &self,
        x_new: &Array2<f64>,
        joint: &JointLinkModel,
    ) -> Result<(Array1<f64>, Array1<f64>, Array1<f64>), ModelError> {
        let samples = self.mcmc_samples.as_ref().ok_or_else(|| {
            ModelError::DimensionMismatch("MCMC samples missing for joint prediction.".to_string())
        })?;
        let n_samples = samples.nrows();
        let n_points = x_new.nrows();
        if n_samples == 0 {
            return Err(ModelError::DimensionMismatch(
                "MCMC samples are empty.".to_string(),
            ));
        }

        let p_base = x_new.ncols();
        let p_link = joint.beta_link.len();
        let expected_cols = p_base + p_link;
        if samples.ncols() != expected_cols {
            return Err(ModelError::DimensionMismatch(format!(
                "Joint MCMC samples have {} cols, expected {} (p_base={}, p_link={})",
                samples.ncols(),
                expected_cols,
                p_base,
                p_link
            )));
        }

        let (min_u, max_u) = joint.knot_range;
        let range_width = (max_u - min_u).max(1e-6);
        let n_raw = joint.knot_vector.len().saturating_sub(joint.degree + 1);
        let n_constrained = joint.link_transform.ncols();

        let mut prob_sum = Array1::<f64>::zeros(n_points);
        let mut eta_sum = Array1::<f64>::zeros(n_points);
        let mut eta_sq_sum = Array1::<f64>::zeros(n_points);
        let mut raw = vec![0.0; n_raw];
        let mut constrained = vec![0.0; n_constrained];
        let mut scratch = basis::SplineScratch::new(joint.degree);

        for s in 0..n_samples {
            let beta_s = samples.slice(s![s, 0..p_base]);
            let theta_s = samples.slice(s![s, p_base..]);
            let u = x_new.dot(&beta_s.to_owned());

            for i in 0..n_points {
                let u_i = u[i];
                let z_i = ((u_i - min_u) / range_width).clamp(0.0, 1.0);

                raw.fill(0.0);
                let eta_i = if basis::evaluate_bspline_basis_scalar(
                    z_i,
                    joint.knot_vector.view(),
                    joint.degree,
                    &mut raw,
                    &mut scratch,
                )
                .is_ok()
                    && joint.link_transform.nrows() == n_raw
                    && n_constrained > 0
                {
                    constrained.fill(0.0);
                    for r in 0..n_raw {
                        let v = raw[r];
                        for c in 0..n_constrained {
                            constrained[c] += v * joint.link_transform[[r, c]];
                        }
                    }
                    let mut wiggle = 0.0;
                    for c in 0..n_constrained.min(theta_s.len()) {
                        wiggle += constrained[c] * theta_s[c];
                    }
                    u_i + wiggle
                } else {
                    u_i
                };

                let eta_clamped = eta_i.clamp(-700.0, 700.0);
                let p_i = 1.0 / (1.0 + f64::exp(-eta_clamped));
                prob_sum[i] += p_i;
                eta_sum[i] += eta_i;
                eta_sq_sum[i] += eta_i * eta_i;
            }
        }

        let ns = n_samples as f64;
        let mean_prob = prob_sum.mapv(|p| (p / ns).clamp(1e-8, 1.0 - 1e-8));
        let mean_eta = eta_sum.mapv(|e| e / ns);
        let var_eta = (&eta_sq_sum / ns) - (&mean_eta * &mean_eta);
        let se_eta = var_eta.mapv(|v| v.max(0.0).sqrt());

        Ok((mean_prob, mean_eta, se_eta))
    }

    /// Detailed predictions including linear predictor, mean response, signed distance
    /// to the peeled hull boundary (negative inside), and optional SEs for eta.
    pub fn predict_detailed(
        &self,
        p_new: ArrayView1<f64>,
        sex_new: ArrayView1<f64>,
        pcs_new: ArrayView2<f64>,
    ) -> Result<
        (
            Array1<f64>,         // eta (linear predictor)
            Array1<f64>,         // mean (after inverse link)
            Array1<f64>,         // signed distance to peeled hull (negative inside)
            Option<Array1<f64>>, // se_eta if available
        ),
        ModelError,
    > {
        if matches!(self.config.model_family, ModelFamily::Survival(_)) {
            return Err(ModelError::UnsupportedForSurvival("predict_detailed"));
        }
        // --- Build design and coefficients ---
        let (x_new, signed_dist) = self.build_prediction_design(p_new, sex_new, pcs_new)?;
        let beta = internal::flatten_coefficients(&self.coefficients, &self.config)?;
        if x_new.ncols() != beta.len() {
            return Err(ModelError::DimensionMismatch(format!(
                "prediction rebuild mismatch: design matrix columns ({}) != coefficient length ({})",
                x_new.ncols(),
                beta.len()
            )));
        }

        // Ensure the model family is supported before invoking link-specific logic
        let link_function = self.ensure_gam_link("prediction")?;

        if link_function == LinkFunction::Logit
            && self.mcmc_samples.is_some()
            && self.joint_link.is_some()
        {
            // Joint posterior predictive: get mean prob, mean eta, and SE from samples
            let joint = self.joint_link.as_ref().unwrap();
            let (mean, eta, se_eta) = self.joint_mcmc_mean_predictions(&x_new, joint)?;
            return Ok((eta, mean, signed_dist, Some(se_eta)));
        }

        if link_function == LinkFunction::Logit
            && self.mcmc_samples.is_some()
            && self.joint_link.is_none()
        {
            // Use coherent MCMC-based mean and SE instead of mixing quantities
            let (mean, eta, se_eta) = self.mcmc_mean_and_se_eta(&x_new)?;
            return Ok((eta, mean, signed_dist, Some(se_eta)));
        }

        // --- Linear predictor and mean ---
        let eta_base = x_new.dot(&beta);
        let (eta, se_eta_opt) = if let Some(joint) = &self.joint_link {
            // For joint models, first try to use the full joint Hessian for SE
            let joint_se =
                self.compute_joint_se_from_hessian(&x_new, &eta_base, joint, link_function);

            // Compute η_cal = u + wiggle(u) (no SE needed from this call since we compute properly above)
            let (eta_cal, _) = self.apply_joint_link_to_eta(&eta_base, None, joint);

            // Use joint SE if available, otherwise fall back to simplified propagation
            let final_se = if joint_se.is_some() {
                joint_se
            } else {
                // Fall back to base SE propagated through g' (imperfect but better than nothing)
                let base_se = self.compute_se_eta_from_hessian(&x_new, link_function);
                let (_, propagated_se) =
                    self.apply_joint_link_to_eta(&eta_base, base_se.as_ref(), joint);
                propagated_se
            };

            (eta_cal, final_se)
        } else {
            let se = self.compute_se_eta_from_hessian(&x_new, link_function);
            (eta_base, se)
        };
        let mean = match link_function {
            LinkFunction::Logit => {
                let eta_clamped = eta.mapv(|e| e.clamp(-700.0, 700.0));
                let mut probs = eta_clamped.mapv(|e| 1.0 / (1.0 + f64::exp(-e)));
                probs.mapv_inplace(|p| p.clamp(1e-8, 1.0 - 1e-8));
                probs
            }
            LinkFunction::Identity => eta.clone(),
        };

        // --- Optional SE for eta using the penalized Hessian ---
        //
        // IMPORTANT: Smoothing bias limitation
        //
        // This computes the CONDITIONAL variance: Var(η̂ | u) = x' H⁻¹ x
        // where H is the penalized Hessian and u represents the spline coefficients.
        //
        // This estimate does NOT account for smoothing bias. Penalized splines
        // systematically:
        //   - Flatten peaks (estimates too low at maxima)
        //   - Fill valleys (estimates too high at minima)
        //   - Round sharp corners
        //
        // The conditional CI centered at η̂ ± 1.96·SE will:
        //   - Have correct coverage where the true function is smooth
        //   - UNDER-cover at peaks, valleys, and sharp features (bias pulls
        //     the estimate away from truth, but SE doesn't account for this)
        //
        // The unconditional approach (averaging over u's prior) would give wider
        // intervals but is "too large where bias is small and too small where
        // bias is large" (Nychka 1988, RWC Ch. 6).
        //
        // For clinical use: treat these SEs as approximate. They are most reliable
        // in smooth regions of the predictor space and may be overconfident at
        // extremes of PGS or ancestry.
        //
        // See: Ruppert, Wand, Carroll "Semiparametric Regression" Ch. 6.6-6.9
        //
        // Note on ridge regularization: If the Hessian was ill-conditioned during
        // fitting and ridge regularization was applied to make it invertible, the
        // stored Hessian has artificially increased curvature. This causes the SE
        // computed here to be a LOWER BOUND on true uncertainty. In sparse regions
        // of covariate space, the reported SE may be overconfident.
        //
        Ok((eta, mean, signed_dist, se_eta_opt))
    }

    fn apply_joint_link_to_eta(
        &self,
        eta_base: &Array1<f64>,
        se_eta_opt: Option<&Array1<f64>>,
        joint: &JointLinkModel,
    ) -> (Array1<f64>, Option<Array1<f64>>) {
        let b_wiggle = match self.build_joint_link_basis(eta_base, joint) {
            Some(basis) => basis,
            None => {
                let se_out = se_eta_opt.map(|se| se.clone());
                return (eta_base.clone(), se_out);
            }
        };

        let wiggle = b_wiggle.dot(&joint.beta_link);
        let eta_cal = eta_base + &wiggle;

        let eff_se = se_eta_opt.map(|se| {
            let g_prime = self
                .compute_joint_link_derivative(eta_base, joint)
                .unwrap_or_else(|| Array1::<f64>::ones(eta_base.len()));
            g_prime.mapv(f64::abs) * se
        });

        (eta_cal, eff_se)
    }
    /// Predicts outcomes for new individuals using the trained model.
    ///
    /// This is the core inference engine. It is a fast, non-iterative process that:
    /// - Reconstructs the mathematical model (design matrix and coefficient vector)
    ///   from the stored configuration.
    /// - Computes the final prediction via matrix algebra.
    ///
    /// # Important: ALO vs Identity Predictor
    ///
    /// During **training**, the model uses Approximate Leave-One-Out (ALO) projections
    /// for stable REML smoothing parameter selection. However, for **prediction**,
    /// this function uses the standard identity predictor `η = Xβ`. This is intentional:
    /// - ALO is a diagnostic/tuning tool, not meant for inference
    /// - New data was not part of the LOO procedure, so ALO doesn't apply
    /// - Standard GLM prediction semantics are preserved
    ///
    /// If you need ALO-adjusted predictions on training data, use the residual
    /// diagnostics functions instead.
    ///
    /// # Arguments
    /// * `p_new`: A 1D array view of new PGS values.
    /// * `pcs_new`: A 2D array view of new PC values, with shape `[n_samples, n_pcs]`.
    ///              The order of PC columns must match `config.pc_configs` names.
    ///
    /// # Returns
    /// A `Result` containing an `Array1<f64>` of predicted outcomes (e.g., probabilities
    /// or continuous values), or a `ModelError`.
    pub fn predict(
        &self,
        p_new: ArrayView1<f64>,
        sex_new: ArrayView1<f64>,
        pcs_new: ArrayView2<f64>,
    ) -> Result<Array1<f64>, ModelError> {
        if matches!(self.config.model_family, ModelFamily::Survival(_)) {
            return Err(ModelError::UnsupportedForSurvival("predict"));
        }
        let link_function = self.ensure_gam_link("prediction")?;
        if link_function == LinkFunction::Logit
            && self.mcmc_samples.is_some()
            && self.joint_link.is_none()
        {
            let (x_new, _) = self.build_prediction_design(p_new, sex_new, pcs_new)?;
            return self.mcmc_mean_logit_predictions(&x_new);
        }

        let (_, mean, _, _) = self.predict_detailed(p_new, sex_new, pcs_new)?;
        Ok(mean)
    }

    /// Predicts outcomes using the posterior mean (via Gauss-Hermite quadrature).
    ///
    /// Unlike `predict` which returns g⁻¹(η̂) (the posterior mode), this method
    /// returns E[g⁻¹(η)] where η ~ N(η̂, σ²). This is the Bayes-optimal predictor
    /// that minimizes squared prediction error (Brier score).
    ///
    /// For the logit link, this means predictions at extreme values are pulled
    /// slightly toward 50%, which is statistically correct behavior given
    /// parameter uncertainty.
    ///
    /// Requires the penalized Hessian to be stored in the model. If unavailable,
    /// falls back to mode-based predictions (equivalent to `predict`).
    ///
    /// For identity link (Gaussian), mean equals mode, so this is identical to `predict`.
    pub fn predict_mean(
        &self,
        p_new: ArrayView1<f64>,
        sex_new: ArrayView1<f64>,
        pcs_new: ArrayView2<f64>,
    ) -> Result<Array1<f64>, ModelError> {
        if matches!(self.config.model_family, ModelFamily::Survival(_)) {
            return Err(ModelError::UnsupportedForSurvival("predict_mean"));
        }

        let link_function = self.ensure_gam_link("predict_mean")?;
        if link_function == LinkFunction::Logit
            && self.mcmc_samples.is_some()
            && self.joint_link.is_none()
        {
            let (x_new, _) = self.build_prediction_design(p_new, sex_new, pcs_new)?;
            return self.mcmc_mean_logit_predictions(&x_new);
        }

        let (eta, mode_mean, _, se_eta_opt) = self.predict_detailed(p_new, sex_new, pcs_new)?;

        let quad_ctx = crate::quadrature::QuadratureContext::new();
        match link_function {
            LinkFunction::Identity => {
                // For identity link, mean = mode, no quadrature needed
                Ok(mode_mean)
            }
            LinkFunction::Logit => {
                // Use quadrature if SE is available
                match se_eta_opt {
                    Some(se_eta) => Ok(crate::quadrature::logit_posterior_mean_batch(
                        &quad_ctx, &eta, &se_eta,
                    )),
                    None => {
                        // No SE available, fall back to mode
                        Ok(mode_mean)
                    }
                }
            }
        }
    }

    /// Predicts outcomes applying the optional post-process calibrator.
    /// Baseline predictions are computed first, then the calibrator adjusts them.
    pub fn predict_calibrated(
        &self,
        p_new: ArrayView1<f64>,
        sex_new: ArrayView1<f64>,
        pcs_new: ArrayView2<f64>,
    ) -> Result<Array1<f64>, ModelError> {
        let _ = (p_new, sex_new, pcs_new);
        Err(ModelError::CalibratorMissing)
    }

    /// Returns the linear predictor (η = Xβ) for new data without applying the inverse link.
    ///
    /// This is useful for diagnostics and tests that want to validate design-matrix
    /// consistency (e.g., checking that training-time `X.dot(beta)` equals prediction-time
    /// reconstruction). For logistic models, prefer `predict` for probabilities.
    pub fn predict_linear(
        &self,
        p_new: ArrayView1<f64>,
        sex_new: ArrayView1<f64>,
        pcs_new: ArrayView2<f64>,
    ) -> Result<Array1<f64>, ModelError> {
        if matches!(self.config.model_family, ModelFamily::Survival(_)) {
            return Err(ModelError::UnsupportedForSurvival("predict_linear"));
        }
        if pcs_new.ncols() != self.config.pc_configs.len() {
            return Err(ModelError::MismatchedPcCount {
                found: pcs_new.ncols(),
                expected: self.config.pc_configs.len(),
            });
        }

        // Assemble raw predictors, optionally project via PHC, and split back
        let mut raw = internal::assemble_raw_from_p_and_pcs(p_new, pcs_new);
        let num_projected = if let Some(hull) = &self.hull {
            hull.project_in_place(raw.view_mut())
        } else {
            0
        };
        if raw.nrows() > 0 && num_projected > 0 {
            let rate = 100.0 * (num_projected as f64) / (raw.nrows() as f64);
            println!(
                "[PHC] Projected {} of {} points ({:.1}%).",
                num_projected,
                raw.nrows(),
                rate
            );
        }
        let (p_corr, pcs_corr) = internal::split_p_and_pcs_from_raw(raw.view());

        let x_new = internal::construct_design_matrix(
            p_corr.view(),
            sex_new,
            pcs_corr.view(),
            &self.config,
            &self.coefficients,
        )?;
        let flattened_coeffs = internal::flatten_coefficients(&self.coefficients, &self.config)?;

        if x_new.ncols() != flattened_coeffs.len() {
            return Err(ModelError::InternalStackingError);
        }

        Ok(x_new.dot(&flattened_coeffs))
    }

    pub fn predict_survival(
        &self,
        age_entry: ArrayView1<f64>,
        age_exit: ArrayView1<f64>,
        p_new: ArrayView1<f64>,
        sex_new: ArrayView1<f64>,
        pcs_new: ArrayView2<f64>,
        risk_type: SurvivalRiskType,
        companion_registry: Option<&HashMap<String, SurvivalModelArtifacts>>,
    ) -> Result<SurvivalPrediction, ModelError> {
        if !matches!(self.config.model_family, ModelFamily::Survival(_)) {
            return Err(ModelError::UnsupportedForSurvival("predict_survival"));
        }

        let n = age_entry.len();
        if age_exit.len() != n {
            return Err(ModelError::DimensionMismatch(format!(
                "age_exit has {} elements but age_entry has {}",
                age_exit.len(),
                n
            )));
        }
        if p_new.len() != n {
            return Err(ModelError::DimensionMismatch(format!(
                "p_new has {} elements but age_entry has {}",
                p_new.len(),
                n
            )));
        }

        let artifacts = self.survival_artifacts()?;
        let covariates = self.assemble_survival_covariates(p_new, sex_new, pcs_new, artifacts)?;

        let registry = companion_registry.unwrap_or(&self.survival_companions);

        let mortality_model = if let SurvivalRiskType::Crude = risk_type {
            if let Some(model) = registry.get("__internal_mortality") {
                Some(model)
            } else {
                return Err(ModelError::SurvivalPrediction(
                    SurvivalError::CompanionModelUnavailable {
                        reference: "__internal_mortality".to_string(),
                    },
                ));
            }
        } else {
            None
        };

        let coeffs = &artifacts.coefficients;
        let design_width = coeffs.len();
        let mortality_width = mortality_model
            .map(|model| model.coefficients.len())
            .unwrap_or(0);

        if let Some(samples) = self.mcmc_samples.as_ref() {
            const MCMC_CHUNK_SIZE: usize = 32;
            const ROW_CHUNK_SIZE: usize = 2048;
            if samples.nrows() == 0 {
                return Err(ModelError::DimensionMismatch(
                    "MCMC samples are empty.".to_string(),
                ));
            }
            if samples.ncols() != design_width {
                return Err(ModelError::DimensionMismatch(format!(
                    "MCMC sample width {} does not match survival design width {}",
                    samples.ncols(),
                    design_width
                )));
            }

            // Pre-allocate scratch buffer for zero-allocation loop
            let max_basis = artifacts.age_basis.knot_vector.len().max(100);
            let max_pgs = 10;
            let mut scratch =
                survival::SurvivalScratch::new(artifacts.age_basis.degree, max_basis, max_pgs);
            let mut entry_buf = Array1::<f64>::zeros(design_width);
            let mut exit_buf = Array1::<f64>::zeros(design_width);

            let mut design_entry = Array2::<f64>::zeros((n, design_width));
            let mut design_exit = Array2::<f64>::zeros((n, design_width));
            for i in 0..n {
                let cov_row = covariates.row(i);
                let entry_age = age_entry[i];
                let exit_age = age_exit[i];
                survival::design_row_at_age_scratch(
                    entry_age,
                    cov_row,
                    artifacts,
                    &mut entry_buf,
                    &mut scratch,
                )?;
                survival::design_row_at_age_scratch(
                    exit_age,
                    cov_row,
                    artifacts,
                    &mut exit_buf,
                    &mut scratch,
                )?;
                if entry_buf.len() != design_width || exit_buf.len() != design_width {
                    return Err(ModelError::DimensionMismatch(
                        "Survival design reconstruction mismatch".to_string(),
                    ));
                }
                design_entry.row_mut(i).assign(&entry_buf);
                design_exit.row_mut(i).assign(&exit_buf);
            }

            let mut hazard_entry = Array1::<f64>::zeros(n);
            let mut hazard_exit = Array1::<f64>::zeros(n);
            let mut cif_entry = Array1::<f64>::zeros(n);
            let mut cif_exit = Array1::<f64>::zeros(n);
            let mut conditional_risk = Array1::<f64>::zeros(n);
            let mut gradient = Array2::<f64>::zeros((n, design_width));
            let mut mortality_gradient = if matches!(risk_type, SurvivalRiskType::Crude)
                && mortality_width > 0
                && mortality_model
                    .and_then(|model| model.hessian_factor.as_ref())
                    .is_some()
            {
                Some(Array2::<f64>::zeros((n, mortality_width)))
            } else {
                None
            };
            let mut logit_risk_sum = Array1::<f64>::zeros(n);
            let mut logit_risk_sq_sum = Array1::<f64>::zeros(n);
            let mortality_ref = if matches!(risk_type, SurvivalRiskType::Crude) {
                Some(mortality_model.expect("checked above"))
            } else {
                None
            };

            let mut n_samples = samples.nrows();
            if let Some(mortality) = mortality_ref {
                if let Some(mcmc) = mortality.mcmc_samples.as_ref() {
                    n_samples = n_samples.min(mcmc.nrows());
                }
            }
            if n_samples == 0 {
                return Err(ModelError::DimensionMismatch(
                    "MCMC samples are empty.".to_string(),
                ));
            }
            let mut row_start = 0;
            while row_start < n {
                let row_end = (row_start + ROW_CHUNK_SIZE).min(n);
                let design_entry_chunk = design_entry.slice(s![row_start..row_end, ..]);
                let design_exit_chunk = design_exit.slice(s![row_start..row_end, ..]);
                let age_entry_chunk = age_entry.slice(s![row_start..row_end]);
                let age_exit_chunk = age_exit.slice(s![row_start..row_end]);

                let mut sample_start = 0;
                while sample_start < n_samples {
                    let sample_end = (sample_start + MCMC_CHUNK_SIZE).min(n_samples);
                    let chunk = samples.slice(s![sample_start..sample_end, ..]);
                    let mortality_chunk = mortality_ref
                        .and_then(|model| model.mcmc_samples.as_ref())
                        .filter(|mcmc| {
                            mcmc.nrows() >= n_samples
                                && mcmc.ncols() == mortality_width
                                && mortality_width > 0
                        })
                        .map(|mcmc| mcmc.slice(s![sample_start..sample_end, ..]));
                    let eta_entry = design_entry_chunk.dot(&chunk.t());
                    let eta_exit = design_exit_chunk.dot(&chunk.t());

                    for j in 0..(sample_end - sample_start) {
                        for i in 0..(row_end - row_start) {
                            let idx = row_start + i;
                            let h_entry = eta_entry[[i, j]].exp();
                            let h_exit = eta_exit[[i, j]].exp();
                            hazard_entry[idx] += h_entry;
                            hazard_exit[idx] += h_exit;

                            let cif_entry_val = 1.0 - (-h_entry).exp();
                            let cif_exit_val = 1.0 - (-h_exit).exp();
                            cif_entry[idx] += cif_entry_val;
                            cif_exit[idx] += cif_exit_val;

                            let mut grad_row_opt = None;
                            let mut mort_grad_row_opt = None;
                            let mut dr_deta_entry = 0.0;
                            let mut dr_deta_exit = 0.0;
                            let risk_val = match risk_type {
                                SurvivalRiskType::Net => {
                                    let delta_h = (h_exit - h_entry).max(0.0);
                                    let risk = 1.0 - (-delta_h).exp();

                                    let exp_neg_entry = (-h_entry).exp();
                                    let exp_neg_exit = (-h_exit).exp();
                                    let delta_raw = exp_neg_entry - exp_neg_exit;
                                    let denom_raw = exp_neg_entry;
                                    let delta = delta_raw.max(0.0);
                                    let denom = denom_raw.max(DEFAULT_RISK_EPSILON);

                                    let d_f_entry = h_entry * exp_neg_entry;
                                    let d_f_exit = h_exit * exp_neg_exit;
                                    dr_deta_exit = if delta_raw > 0.0 {
                                        d_f_exit / denom
                                    } else {
                                        0.0
                                    };
                                    let numerator = if delta_raw > 0.0 { delta } else { 0.0 };
                                    let dnum = if delta_raw > 0.0 { -d_f_entry } else { 0.0 };
                                    let dden = -d_f_entry;
                                    dr_deta_entry = if denom_raw > DEFAULT_RISK_EPSILON {
                                        (dnum * denom_raw - numerator * dden)
                                            / (denom_raw * denom_raw)
                                    } else {
                                        0.0
                                    };
                                    risk
                                }
                                SurvivalRiskType::Crude => {
                                    let mortality = mortality_ref.expect("checked above");
                                    let mortality_coeffs =
                                        mortality_chunk.as_ref().map(|chunk| chunk.row(j));
                                    let result = survival::calculate_crude_risk_quadrature(
                                        age_entry_chunk[i],
                                        age_exit_chunk[i],
                                        covariates.row(idx),
                                        artifacts,
                                        mortality,
                                        Some(chunk.row(j)),
                                        mortality_coeffs,
                                    )?;
                                    grad_row_opt = Some(result.disease_gradient);
                                    mort_grad_row_opt = Some(result.mortality_gradient);
                                    result.risk
                                }
                            };

                            conditional_risk[idx] += risk_val;
                            let risk_clamped = risk_val.max(1e-12).min(1.0 - 1e-12);
                            let logit_r = (risk_clamped / (1.0 - risk_clamped)).ln();
                            logit_risk_sum[idx] += logit_r;
                            logit_risk_sq_sum[idx] += logit_r * logit_r;
                            let logistic_scale = 1.0 / (risk_clamped * (1.0 - risk_clamped));
                            match risk_type {
                                SurvivalRiskType::Net => {
                                    let entry_row = design_entry_chunk.row(i);
                                    let exit_row = design_exit_chunk.row(i);
                                    let mut grad_acc = gradient.row_mut(idx);
                                    for k in 0..design_width {
                                        grad_acc[k] += (exit_row[k] * dr_deta_exit
                                            + entry_row[k] * dr_deta_entry)
                                            * logistic_scale;
                                    }
                                }
                                SurvivalRiskType::Crude => {
                                    if let Some(mut grad_row) = grad_row_opt {
                                        grad_row.mapv_inplace(|v| v * logistic_scale);
                                        let mut grad_acc = gradient.row_mut(idx);
                                        Zip::from(&mut grad_acc)
                                            .and(&grad_row)
                                            .for_each(|a, &b| *a += b);
                                    }
                                    if let (Some(mut mort_grad_row), Some(ref mut mort_acc)) =
                                        (mort_grad_row_opt, mortality_gradient.as_mut())
                                    {
                                        mort_grad_row.mapv_inplace(|v| v * logistic_scale);
                                        let mut mort_row = mort_acc.row_mut(idx);
                                        Zip::from(&mut mort_row)
                                            .and(&mort_grad_row)
                                            .for_each(|a, &b| *a += b);
                                    }
                                }
                            }
                        }
                    }
                    sample_start = sample_end;
                }
                row_start = row_end;
            }

            let scale = 1.0 / (samples.nrows() as f64);
            hazard_entry.mapv_inplace(|v| v * scale);
            hazard_exit.mapv_inplace(|v| v * scale);
            cif_entry.mapv_inplace(|v| v * scale);
            cif_exit.mapv_inplace(|v| v * scale);
            conditional_risk.mapv_inplace(|v| v * scale);
            gradient.mapv_inplace(|v| v * scale);
            if let Some(ref mut mort_grad) = mortality_gradient {
                mort_grad.mapv_inplace(|v| v * scale);
            }

            let mut logit_risk = Array1::<f64>::zeros(n);
            for i in 0..n {
                let p = conditional_risk[i].max(1e-12).min(1.0 - 1e-12);
                logit_risk[i] = (p / (1.0 - p)).ln();
            }

            let mut logit_risk_se = if n_samples > 1 {
                let mut se = Array1::<f64>::zeros(n);
                let n_f = n_samples as f64;
                for i in 0..n {
                    let mean = logit_risk_sum[i] / n_f;
                    let mean_sq = logit_risk_sq_sum[i] / n_f;
                    let var = (n_f / (n_f - 1.0)) * (mean_sq - mean * mean);
                    se[i] = if var > 0.0 { var.sqrt() } else { 0.0 };
                }
                Some(se)
            } else {
                None
            };
            if let (Some(mort_grad), Some(mortality)) = (mortality_gradient.as_ref(), mortality_ref)
            {
                if let Some(factor) = mortality.hessian_factor.as_ref() {
                    let mort_se = survival::delta_method_standard_errors(factor, mort_grad)?;
                    let combined = match logit_risk_se.as_mut() {
                        Some(se) => {
                            for i in 0..se.len() {
                                let var = se[i] * se[i] + mort_se[i] * mort_se[i];
                                se[i] = var.max(0.0).sqrt();
                            }
                            None
                        }
                        None => Some(mort_se),
                    };
                    if let Some(se) = combined {
                        logit_risk_se = Some(se);
                    }
                }
            }

            return Ok(SurvivalPrediction {
                cumulative_hazard_entry: hazard_entry,
                cumulative_hazard_exit: hazard_exit,
                cumulative_incidence_entry: cif_entry,
                cumulative_incidence_exit: cif_exit,
                conditional_risk,
                logit_risk,
                logit_risk_se,
                logit_risk_design: Some(gradient),
            });
        }

        let mut hazard_entry = Array1::<f64>::zeros(n);
        let mut hazard_exit = Array1::<f64>::zeros(n);
        let mut cif_entry = Array1::<f64>::zeros(n);
        let mut cif_exit = Array1::<f64>::zeros(n);
        let mut conditional_risk = Array1::<f64>::zeros(n);
        let mut logit_risk = Array1::<f64>::zeros(n);
        let mut gradient = Array2::<f64>::zeros((n, design_width));
        let mut mortality_gradient = if matches!(risk_type, SurvivalRiskType::Crude)
            && mortality_width > 0
            && mortality_model
                .and_then(|model| model.hessian_factor.as_ref())
                .is_some()
        {
            Some(Array2::<f64>::zeros((n, mortality_width)))
        } else {
            None
        };

        // Pre-allocate scratch buffer for zero-allocation loop
        let max_basis = artifacts.age_basis.knot_vector.len().max(100);
        let max_pgs = 10;
        let mut scratch =
            survival::SurvivalScratch::new(artifacts.age_basis.degree, max_basis, max_pgs);
        let mut design_entry = Array1::<f64>::zeros(design_width);
        let mut design_exit = Array1::<f64>::zeros(design_width);

        for i in 0..n {
            let cov_row = covariates.row(i);
            let cov_row_owned = cov_row.to_owned(); // For functions requiring Array1
            let entry_age = age_entry[i];
            let exit_age = age_exit[i];

            let hazard_entry_val =
                survival::cumulative_hazard(entry_age, &cov_row_owned, artifacts)?;
            let hazard_exit_val = survival::cumulative_hazard(exit_age, &cov_row_owned, artifacts)?;
            hazard_entry[i] = hazard_entry_val;
            hazard_exit[i] = hazard_exit_val;

            let cif_entry_val =
                survival::cumulative_incidence(entry_age, &cov_row_owned, artifacts)?;
            let cif_exit_val = survival::cumulative_incidence(exit_age, &cov_row_owned, artifacts)?;
            cif_entry[i] = cif_entry_val;
            cif_exit[i] = cif_exit_val;

            survival::design_row_at_age_scratch(
                entry_age,
                cov_row,
                artifacts,
                &mut design_entry,
                &mut scratch,
            )?;
            survival::design_row_at_age_scratch(
                exit_age,
                cov_row,
                artifacts,
                &mut design_exit,
                &mut scratch,
            )?;
            if design_entry.len() != design_width || design_exit.len() != design_width {
                return Err(ModelError::DimensionMismatch(
                    "Survival design reconstruction mismatch".to_string(),
                ));
            }

            let (risk_val, mut grad_row, mut mort_grad_row_opt) = match risk_type {
                SurvivalRiskType::Net => {
                    let risk = survival::conditional_absolute_risk(
                        entry_age,
                        exit_age,
                        &cov_row_owned,
                        artifacts,
                    )?;

                    let eta_entry = design_entry.dot(coeffs);
                    let eta_exit = design_exit.dot(coeffs);
                    let h_entry = eta_entry.exp();
                    let h_exit = eta_exit.exp();
                    let exp_neg_entry = (-h_entry).exp();
                    let exp_neg_exit = (-h_exit).exp();
                    let f_entry = 1.0 - exp_neg_entry;
                    let f_exit = 1.0 - exp_neg_exit;
                    let delta_raw = f_exit - f_entry;
                    let denom_raw = 1.0 - f_entry;
                    let delta = delta_raw.max(0.0);
                    let denom = denom_raw.max(DEFAULT_RISK_EPSILON);

                    let d_f_entry = h_entry * exp_neg_entry;
                    let d_f_exit = h_exit * exp_neg_exit;
                    let dr_deta_exit = if delta_raw > 0.0 {
                        d_f_exit / denom
                    } else {
                        0.0
                    };
                    let numerator = if delta_raw > 0.0 { delta } else { 0.0 };
                    let dnum = if delta_raw > 0.0 { -d_f_entry } else { 0.0 };
                    let dden = -d_f_entry;
                    let dr_deta_entry = if denom_raw > DEFAULT_RISK_EPSILON {
                        (dnum * denom_raw - numerator * dden) / (denom_raw * denom_raw)
                    } else {
                        0.0
                    };

                    let grad_exit = design_exit.mapv(|v| v * dr_deta_exit);
                    let grad_entry = design_entry.mapv(|v| v * dr_deta_entry);
                    let grad = grad_exit + grad_entry;

                    (risk, grad, None)
                }
                SurvivalRiskType::Crude => {
                    let mortality = mortality_model.expect("checked above");
                    let result = survival::calculate_crude_risk_quadrature(
                        entry_age,
                        exit_age,
                        cov_row.view(),
                        artifacts,
                        mortality,
                        None,
                        None,
                    )?;
                    (
                        result.risk,
                        result.disease_gradient,
                        Some(result.mortality_gradient),
                    )
                }
            };
            conditional_risk[i] = risk_val;
            let risk_clamped = risk_val.max(1e-12).min(1.0 - 1e-12);
            logit_risk[i] = (risk_clamped / (1.0 - risk_clamped)).ln();
            let logistic_scale = 1.0 / (risk_clamped * (1.0 - risk_clamped));

            grad_row.mapv_inplace(|v| v * logistic_scale);
            gradient.row_mut(i).assign(&grad_row);
            if let (Some(mut mort_grad_row), Some(ref mut mort_acc)) =
                (mort_grad_row_opt.take(), mortality_gradient.as_mut())
            {
                mort_grad_row.mapv_inplace(|v| v * logistic_scale);
                mort_acc.row_mut(i).assign(&mort_grad_row);
            }
        }

        let logit_risk_se = match risk_type {
            SurvivalRiskType::Net => {
                if let Some(factor) = artifacts.hessian_factor.as_ref() {
                    Some(survival::delta_method_standard_errors(factor, &gradient)?)
                } else {
                    None
                }
            }
            SurvivalRiskType::Crude => {
                let disease_factor = artifacts.hessian_factor.as_ref();
                let mortality_factor =
                    mortality_model.and_then(|mortality| mortality.hessian_factor.as_ref());
                let mortality_grad = mortality_gradient.as_ref();
                let cross_cov = mortality_model
                    .and_then(|mortality| mortality.cross_covariance_to_primary.as_ref());

                let disease_se = disease_factor
                    .map(|factor| survival::delta_method_standard_errors(factor, &gradient))
                    .transpose()?;
                let mortality_se = if let (Some(mg), Some(mf)) = (mortality_grad, mortality_factor)
                {
                    Some(survival::delta_method_standard_errors(mf, mg)?)
                } else {
                    None
                };

                match (disease_se, mortality_se) {
                    (Some(d), Some(m)) => {
                        let mut out = d.clone();
                        if let (Some(mg), Some(cov_dm)) = (mortality_grad, cross_cov) {
                            for i in 0..out.len() {
                                let g_d = gradient.row(i);
                                let g_m = mg.row(i);
                                let cov_gm = cov_dm.dot(&g_m);
                                let cross = g_d.dot(&cov_gm);
                                let var = d[i] * d[i] + m[i] * m[i] + 2.0 * cross;
                                out[i] = var.max(0.0).sqrt();
                            }
                        } else {
                            for i in 0..out.len() {
                                let var = d[i] * d[i] + m[i] * m[i];
                                out[i] = var.max(0.0).sqrt();
                            }
                        }
                        Some(out)
                    }
                    (Some(d), None) => Some(d),
                    (None, Some(m)) => Some(m),
                    (None, None) => None,
                }
            }
        };

        Ok(SurvivalPrediction {
            cumulative_hazard_entry: hazard_entry,
            cumulative_hazard_exit: hazard_exit,
            cumulative_incidence_entry: cif_entry,
            cumulative_incidence_exit: cif_exit,
            conditional_risk,
            logit_risk,
            logit_risk_se,
            logit_risk_design: Some(gradient),
        })
    }

    pub fn predict_survival_calibrated(
        &self,
        age_entry: ArrayView1<f64>,
        age_exit: ArrayView1<f64>,
        p_new: ArrayView1<f64>,
        sex_new: ArrayView1<f64>,
        pcs_new: ArrayView2<f64>,
        companion_registry: Option<&HashMap<String, SurvivalModelArtifacts>>,
    ) -> Result<Array1<f64>, ModelError> {
        if !matches!(self.config.model_family, ModelFamily::Survival(_)) {
            return Err(ModelError::UnsupportedForSurvival(
                "predict_survival_calibrated",
            ));
        }
        let artifacts = self.survival_artifacts()?;
        if artifacts.calibrator.is_none() {
            return Err(ModelError::CalibratorMissing);
        }

        // Compute hull distance for ancestry-domain calibration
        let signed_dist = if let Some(hull) = &self.hull {
            let raw = internal::assemble_raw_from_p_and_pcs(p_new, pcs_new);
            Some(hull.signed_distance_many(raw.view()))
        } else {
            None
        };

        let baseline = self.predict_survival(
            age_entry,
            age_exit,
            p_new,
            sex_new,
            pcs_new,
            SurvivalRiskType::Net,
            companion_registry,
        )?;
        // logit_risk_design is required for calibration SE computation
        let design = baseline.logit_risk_design.as_ref().ok_or_else(|| {
            ModelError::DimensionMismatch("logit_risk_design is required for calibration".into())
        })?;
        artifacts
            .apply_logit_risk_calibrator(
                &baseline.conditional_risk,
                baseline.logit_risk_se.as_ref(),
                design,
                signed_dist.as_ref(),
            )
            .map_err(ModelError::from)
    }

    /// Saves the trained model to a file in a human-readable TOML format.
    pub fn save(&self, path: &str) -> Result<(), ModelError> {
        let toml_string = toml::to_string_pretty(self)?;
        let mut file = BufWriter::new(fs::File::create(path)?);
        file.write_all(toml_string.as_bytes())?;
        Ok(())
    }

    /// Loads a trained model from a TOML file.
    pub fn load(path: &str) -> Result<Self, ModelError> {
        let toml_string = fs::read_to_string(path)?;
        let model: TrainedModel = toml::from_str(&toml_string)?;
        model.assert_layout_consistency_from_config()?;
        Ok(model)
    }

    fn rebuild_layout_from_config(&self) -> Result<EngineLayout, ModelError> {
        if matches!(self.config.model_family, ModelFamily::Survival(_)) {
            return Err(ModelError::UnsupportedForSurvival(
                "rebuild_layout_from_config",
            ));
        }
        let mut pc_null_ncols = Vec::with_capacity(self.config.pc_configs.len());
        let mut pc_range_ncols = Vec::with_capacity(self.config.pc_configs.len());
        let mut pc_int_ncols = Vec::with_capacity(self.config.pc_configs.len());

        for pc in &self.config.pc_configs {
            let range_cols = self
                .config
                .range_transforms
                .get(&pc.name)
                .ok_or_else(|| {
                    ModelError::ConstraintMissing(format!("range transform for {}", pc.name))
                })?
                .ncols();
            pc_range_ncols.push(range_cols);

            let interaction_cols = match self.config.interaction_penalty {
                InteractionPenaltyKind::Isotropic => range_cols,
                InteractionPenaltyKind::Anisotropic => {
                    pc.basis_config.num_knots + pc.basis_config.degree
                }
            };
            pc_int_ncols.push(interaction_cols);

            let null_cols = self
                .config
                .pc_null_transforms
                .get(&pc.name)
                .ok_or_else(|| {
                    ModelError::ConstraintMissing(format!("pc_null_transform for {}", pc.name))
                })?
                .ncols();
            pc_null_ncols.push(null_cols);
        }

        let pgs_main_cols = self.coefficients.main_effects.pgs.len();
        let pgs_range_cols = self
            .config
            .range_transforms
            .get("pgs")
            .ok_or_else(|| ModelError::ConstraintMissing("pgs range transform".to_string()))?
            .ncols();
        let pgs_int_cols = match self.config.interaction_penalty {
            InteractionPenaltyKind::Isotropic => pgs_range_cols,
            InteractionPenaltyKind::Anisotropic => {
                self.config.pgs_basis_config.num_knots + self.config.pgs_basis_config.degree
            }
        };

        let sex_main_cols = 1;

        EngineLayout::new(
            &self.config,
            &pc_null_ncols,
            &pc_range_ncols,
            sex_main_cols,
            pgs_main_cols,
            &pc_int_ncols,
            pgs_int_cols,
        )
        .map_err(ModelError::EstimationError)
    }

    fn assert_layout_consistency_from_config(&self) -> Result<(), ModelError> {
        if matches!(self.config.model_family, ModelFamily::Survival(_)) {
            return Ok(());
        }
        let layout = self.rebuild_layout_from_config()?;
        self.assert_layout_consistency_with_layout(&layout)?;
        Ok(())
    }

    pub(crate) fn assert_layout_consistency_with_layout(
        &self,
        layout: &EngineLayout,
    ) -> Result<(), ModelError> {
        if matches!(self.config.model_family, ModelFamily::Survival(_)) {
            return Ok(());
        }
        assert_eq!(
            self.lambdas.len(),
            layout.num_penalties,
            "Mismatch: saved lambdas vs penalties (saved {}, expected {})",
            self.lambdas.len(),
            layout.num_penalties
        );

        if let Some(sex_col) = layout.sex_col {
            if sex_col >= layout.total_coeffs {
                return Err(ModelError::DimensionMismatch(format!(
                    "Sex column index {} exceeds total coefficients {}",
                    sex_col, layout.total_coeffs
                )));
            }
        } else {
            return Err(ModelError::DimensionMismatch(
                "Layout is missing a sex column for the main effect".to_string(),
            ));
        }

        for (pc_idx, pc_config) in self.config.pc_configs.iter().enumerate() {
            let key = format!("f(PGS,{})", pc_config.name);
            if let Some(stored) = self.coefficients.interaction_effects.get(&key) {
                if pc_idx >= layout.interaction_factor_widths.len() {
                    return Err(ModelError::DimensionMismatch(format!(
                        "Layout missing interaction factor widths for {} (index {})",
                        pc_config.name, pc_idx
                    )));
                }

                let (pgs_dim_dbg, pc_dim_dbg) = layout.interaction_factor_widths[pc_idx];
                let expected = pgs_dim_dbg * pc_dim_dbg;

                assert_eq!(
                    stored.len(),
                    expected,
                    "Mismatch: stored vs rebuilt interaction width for {key} (stored {}, expected {} = {}×{})",
                    stored.len(),
                    expected,
                    pgs_dim_dbg,
                    pc_dim_dbg
                );

                if pc_idx < layout.interaction_block_idx.len() {
                    let layout_cols = layout.penalty_map[layout.interaction_block_idx[pc_idx]]
                        .col_range
                        .len();
                    assert_eq!(
                        layout_cols, expected,
                        "Mismatch: layout vs rebuilt interaction width for {key} (layout {}, expected {} = {}×{})",
                        layout_cols, expected, pgs_dim_dbg, pc_dim_dbg
                    );
                }
            }
        }

        let flattened_coeffs = internal::flatten_coefficients(&self.coefficients, &self.config)?;
        if flattened_coeffs.len() != layout.total_coeffs {
            return Err(ModelError::DimensionMismatch(format!(
                "Flattened coefficient length ({}) does not match layout.total_coeffs ({}). Likely causes: knot vectors/degree drift, transform set drift (range/null/orth), or penalty mode change (isotropic/anisotropic).",
                flattened_coeffs.len(),
                layout.total_coeffs
            )));
        }

        Ok(())
    }
}

/// Internal module for prediction-specific implementation details.
mod internal {
    use super::*;
    // no extra imports

    /// Assemble raw predictors into an n x d matrix [PGS | PC1 | PC2 | ...]
    pub(super) fn assemble_raw_from_p_and_pcs(
        p: ArrayView1<f64>,
        pcs: ArrayView2<f64>,
    ) -> Array2<f64> {
        let n = p.len();
        let d = 1 + pcs.ncols();
        let mut x = Array2::zeros((n, d));
        // first column is p
        x.column_mut(0).assign(&p);
        // remaining are pcs
        if pcs.ncols() > 0 {
            x.slice_mut(s![.., 1..]).assign(&pcs);
        }
        x
    }

    /// Split raw matrix [PGS | PCs...] into (p, pcs)
    pub(super) fn split_p_and_pcs_from_raw(x: ArrayView2<f64>) -> (Array1<f64>, Array2<f64>) {
        let n = x.nrows();
        let d = x.ncols();
        let mut p = Array1::zeros(n);
        p.assign(&x.column(0));
        let pcs = if d > 1 {
            x.slice(s![.., 1..]).to_owned()
        } else {
            Array2::zeros((n, 0))
        };
        (p, pcs)
    }

    /// Computes the row-wise tensor product (Khatri-Rao product) of two matrices.
    /// This creates the design matrix columns for tensor product interactions.
    /// Each row of the result is the outer product of the corresponding rows from A and B.
    fn row_wise_tensor_product(a: &Array2<f64>, b: &Array2<f64>) -> Array2<f64> {
        let n_samples = a.nrows();
        assert_eq!(
            n_samples,
            b.nrows(),
            "Matrices must have same number of rows"
        );

        let a_cols = a.ncols();
        let b_cols = b.ncols();
        let mut result = Array2::zeros((n_samples, a_cols * b_cols));

        for row in 0..n_samples {
            let mut col_idx = 0;
            for i in 0..a_cols {
                for j in 0..b_cols {
                    result[[row, col_idx]] = a[[row, i]] * b[[row, j]];
                    col_idx += 1;
                }
            }
        }

        result
    }

    /// Constructs the design matrix `X` for new data following a strict canonical order.
    /// This order is the implicit contract that allows the flattened coefficients to work correctly.
    pub(super) fn construct_design_matrix(
        p_new: ArrayView1<f64>,
        sex_new: ArrayView1<f64>,
        pcs_new: ArrayView2<f64>,
        config: &ModelConfig,
        coeffs: &MappedCoefficients,
    ) -> Result<Array2<f64>, ModelError> {
        // Validate that prediction data dimensions are consistent
        if p_new.len() != pcs_new.nrows() {
            return Err(ModelError::DimensionMismatch(format!(
                "Sample count mismatch: p_new has {} samples but pcs_new has {} rows",
                p_new.len(),
                pcs_new.nrows()
            )));
        }
        if sex_new.len() != p_new.len() {
            return Err(ModelError::DimensionMismatch(format!(
                "Sample count mismatch: p_new has {} samples but sex_new has {}",
                p_new.len(),
                sex_new.len()
            )));
        }
        // Stage: Generate the PGS basis using the saved knot vector if available
        // Only use saved knot vectors - remove fallback to ensure consistency
        let saved_knots = config
            .knot_vectors
            .get("pgs")
            .ok_or(ModelError::InternalStackingError)?;

        let (pgs_basis_unc, _) = create_basis::<Dense>(
            p_new,
            KnotSource::Provided(saved_knots.view()),
            config.pgs_basis_config.degree,
            BasisOptions::value(),
        )?;

        // Build PGS main basis to MATCH TRAINING exactly.
        // Start from the unconstrained PGS basis (drop intercept col).
        let pgs_main_basis_unc = pgs_basis_unc.slice(s![.., 1..]);

        // Require sum-to-zero constraint for PGS main effect
        let pgs_z = config
            .sum_to_zero_constraints
            .get("pgs_main")
            .ok_or_else(|| ModelError::ConstraintMissing("pgs_main".to_string()))?;

        // Note: z_range_pgs is still required for interactions later, but not needed for main effects

        // Target width must match trained coefficient count
        let pgs_coef_len = coeffs.main_effects.pgs.len();

        // For PGS main effect, we use ONLY the sum-to-zero constraint transform.
        // The range transform (z_range_pgs) is used ONLY for interaction terms.

        // Check shapes: B_unc (N x m), Z (m x c)
        let m = pgs_main_basis_unc.ncols();
        let c = pgs_z.ncols();

        // Verify dimensions match before matrix multiplication
        if m != pgs_z.nrows() || c != pgs_coef_len {
            return Err(ModelError::DimensionMismatch(format!(
                "PGS basis transform dimensions mismatch: B_unc {}×{}, Z {}×{}, coefs {}",
                pgs_main_basis_unc.nrows(),
                m,
                pgs_z.nrows(),
                c,
                pgs_coef_len
            )));
        }

        // Apply ONLY the sum-to-zero constraint to PGS main effect basis
        // This matches the training behavior in construction.rs
        let pgs_main_basis = pgs_main_basis_unc.dot(pgs_z);

        // The interaction basis MUST be constructed from the UNCONSTRAINED PGS basis.
        // The model's coefficient layout (defined in construction.rs -> EngineLayout::new) is derived
        // from the dimensions of the unconstrained bases. Changing this to use a constrained basis
        // without a corresponding change to the layout logic will cause a dimension mismatch and lead
        // to silently incorrect predictions. The debug_assert in TrainedModel::predict guards against this.

        // Stage: Generate bases for PCs (functional ANOVA decomposition: null + range)
        let mut pc_range_bases = Vec::new();
        let mut pc_null_bases: Vec<Option<Array2<f64>>> = Vec::new();
        let mut pc_unconstrained_bases_main = Vec::new();
        for i in 0..config.pc_configs.len() {
            let pc_col = pcs_new.column(i);
            let pc_name = &config.pc_configs[i].name;
            // Only use saved knot vectors - remove fallback to ensure consistency
            let saved_knots = config
                .knot_vectors
                .get(pc_name)
                .ok_or(ModelError::InternalStackingError)?;

            let (pc_basis_unc, _) = create_basis::<Dense>(
                pc_col,
                KnotSource::Provided(saved_knots.view()),
                config.pc_configs[i].basis_config.degree,
                BasisOptions::value(),
            )?;

            // Slice the basis to remove the intercept term, just like in the training code
            let pc_main_basis_unc = pc_basis_unc.slice(s![.., 1..]);
            pc_unconstrained_bases_main.push(pc_main_basis_unc.to_owned());

            // Apply the SAVED null and range transformations for functional ANOVA
            if let Some(range_transform) = config.range_transforms.get(pc_name) {
                // Check dimensions before matrix multiplication to prevent panic
                if pc_main_basis_unc.ncols() != range_transform.nrows() {
                    return Err(ModelError::InternalStackingError);
                }

                // Use range-only transformation for PC main effects (fully penalized)
                let pc_range_basis = pc_main_basis_unc.dot(range_transform);
                pc_range_bases.push(pc_range_basis);
                // Get the required null-space transform
                let z_null = config.pc_null_transforms.get(pc_name).ok_or_else(|| {
                    ModelError::ConstraintMissing(format!("pc_null_transform for {}", pc_name))
                })?;

                if pc_main_basis_unc.ncols() != z_null.nrows() {
                    return Err(ModelError::DimensionMismatch(format!(
                        "PC null transform dimensions mismatch for {}: basis {}×{}, transform {}×{}",
                        pc_name,
                        pc_main_basis_unc.nrows(),
                        pc_main_basis_unc.ncols(),
                        z_null.nrows(),
                        z_null.ncols()
                    )));
                }

                // Build null-space basis (may have 0 columns if PC has no null space)
                let pc_null_basis = pc_main_basis_unc.dot(z_null);
                pc_null_bases.push(if z_null.ncols() > 0 {
                    Some(pc_null_basis)
                } else {
                    None
                });
            } else {
                // No range transform found for this PC
                return Err(ModelError::ConstraintMissing(format!(
                    "range transform for {}",
                    pc_name
                )));
            }
        }

        // Stage: Assemble the design matrix following the canonical order
        let n_samples = p_new.len();
        let mut owned_cols: Vec<Array1<f64>> = Vec::new();

        // Stage: Populate the intercept column
        owned_cols.push(Array1::<f64>::ones(n_samples));

        // Stage: Add sex main effect column
        owned_cols.push(sex_new.to_owned());

        // Stage: Add main PC effects per PC—null first (if any), then range
        for pc_idx in 0..config.pc_configs.len() {
            if let Some(ref null_basis) = pc_null_bases[pc_idx] {
                for col in null_basis.axis_iter(Axis(1)) {
                    owned_cols.push(col.to_owned());
                }
            }
            for col in pc_range_bases[pc_idx].axis_iter(Axis(1)) {
                owned_cols.push(col.to_owned());
            }
        }

        // Stage: Add the main PGS effect
        for col in pgs_main_basis.axis_iter(Axis(1)) {
            owned_cols.push(col.to_owned());
        }

        // Stage: Add the sex×PGS varying-coefficient interaction if present
        let sex_pgs_key = "f(PGS,sex)";
        if let Some(sex_pgs_coeffs) = coeffs.interaction_effects.get(sex_pgs_key) {
            let mut sex_pgs_basis = pgs_main_basis.to_owned();
            for (mut row, &sex_value) in sex_pgs_basis.axis_iter_mut(Axis(0)).zip(sex_new.iter()) {
                row *= sex_value;
            }

            let alpha = config
                .interaction_orth_alpha
                .get(sex_pgs_key)
                .ok_or_else(|| {
                    ModelError::ConstraintMissing(format!(
                        "interaction_orth_alpha for {}",
                        sex_pgs_key
                    ))
                })?;

            let intercept = Array1::<f64>::ones(n_samples);
            let sex_col = sex_new.to_owned();
            let m_cols = 1 + 1 + pgs_main_basis.ncols();
            if alpha.nrows() != m_cols {
                return Err(ModelError::DimensionMismatch(format!(
                    "Orth map stored with unexpected row count for {}: got {}, expected {}",
                    sex_pgs_key,
                    alpha.nrows(),
                    m_cols
                )));
            }
            if alpha.ncols() != sex_pgs_basis.ncols() {
                return Err(ModelError::DimensionMismatch(format!(
                    "Orth map stored with unexpected column count for {}: got {}, expected {}",
                    sex_pgs_key,
                    alpha.ncols(),
                    sex_pgs_basis.ncols()
                )));
            }

            let mut m_matrix = Array2::<f64>::zeros((n_samples, m_cols));
            let mut offset = 0;
            m_matrix.column_mut(offset).assign(&intercept);
            offset += 1;
            m_matrix.column_mut(offset).assign(&sex_col);
            offset += 1;
            m_matrix
                .slice_mut(s![.., offset..offset + pgs_main_basis.ncols()])
                .assign(&pgs_main_basis);

            let sex_pgs_orth = &sex_pgs_basis - &m_matrix.dot(alpha);

            if sex_pgs_coeffs.len() != sex_pgs_orth.ncols() {
                return Err(ModelError::DimensionMismatch(format!(
                    "Stored interaction coefficient count for {} is {}, but constructed design has {} columns.",
                    sex_pgs_key,
                    sex_pgs_coeffs.len(),
                    sex_pgs_orth.ncols()
                )));
            }

            for col in sex_pgs_orth.axis_iter(Axis(1)) {
                owned_cols.push(col.to_owned());
            }
        }

        // Stage: Add tensor product interaction effects (only if PCs are present)
        if !config.pc_configs.is_empty() {
            // Reconstruct interaction marginals using the same basis choice used during training
            let pgs_int_basis = match config.interaction_penalty {
                InteractionPenaltyKind::Isotropic => {
                    let z_range_pgs_pred = config.range_transforms.get("pgs").ok_or_else(|| {
                        ModelError::ConstraintMissing("pgs range transform".to_string())
                    })?;

                    if pgs_main_basis_unc.ncols() != z_range_pgs_pred.nrows() {
                        return Err(ModelError::InternalStackingError);
                    }

                    pgs_main_basis_unc.dot(z_range_pgs_pred)
                }
                InteractionPenaltyKind::Anisotropic => pgs_main_basis_unc.to_owned(),
            };

            for pc_idx in 0..config.pc_configs.len() {
                let pc_name = &config.pc_configs[pc_idx].name;
                let tensor_key = format!("f(PGS,{})", pc_name);

                // Only build the interaction if the trained model contains coefficients for it
                if coeffs.interaction_effects.contains_key(&tensor_key) {
                    let pc_int_basis = match config.interaction_penalty {
                        InteractionPenaltyKind::Isotropic => {
                            let z_range_pc_pred = config
                                .range_transforms
                                .get(pc_name)
                                .ok_or(ModelError::InternalStackingError)?;

                            if pc_unconstrained_bases_main[pc_idx].ncols()
                                != z_range_pc_pred.nrows()
                            {
                                return Err(ModelError::InternalStackingError);
                            }

                            pc_unconstrained_bases_main[pc_idx].dot(z_range_pc_pred)
                        }
                        InteractionPenaltyKind::Anisotropic => {
                            pc_unconstrained_bases_main[pc_idx].to_owned()
                        }
                    };

                    let mut tensor_interaction =
                        row_wise_tensor_product(&pgs_int_basis, &pc_int_basis);

                    // Apply stored orthogonalization to remove main-effect components (pure interaction)
                    let alpha =
                        config
                            .interaction_orth_alpha
                            .get(&tensor_key)
                            .ok_or_else(|| {
                                ModelError::ConstraintMissing(format!(
                                    "interaction_orth_alpha for {}",
                                    tensor_key
                                ))
                            })?;

                    // Build M = [Intercept | Sex? | PGS_main | PC_main_for_this_pc (null + range)]
                    let intercept = Array1::<f64>::ones(n_samples);
                    let sex_col = sex_new.to_owned();
                    let pc_null_cols = pc_null_bases
                        .get(pc_idx)
                        .and_then(|opt| opt.as_ref().map(|arr| arr.ncols()))
                        .unwrap_or(0);
                    let pc_range_cols = pc_range_bases[pc_idx].ncols();
                    let base_cols_without_sex =
                        1 + pgs_main_basis.ncols() + pc_null_cols + pc_range_cols;
                    let base_cols_with_sex = base_cols_without_sex + 1;

                    let include_sex = match alpha.nrows() {
                        n if n == base_cols_with_sex => true,
                        n if n == base_cols_without_sex => false,
                        n => {
                            return Err(ModelError::DimensionMismatch(format!(
                                "Orth map stored with unexpected row count for {tensor_key}: got {n}, expected {base_cols_without_sex} (without sex) or {base_cols_with_sex} (with sex)",
                            )));
                        }
                    };

                    let mut m_cols: Vec<Array1<f64>> = Vec::new();
                    m_cols.push(intercept);
                    if include_sex {
                        m_cols.push(sex_col);
                    }
                    m_cols.extend(pgs_main_basis.axis_iter(Axis(1)).map(|c| c.to_owned()));
                    if let Some(ref pc_null) = pc_null_bases[pc_idx] {
                        m_cols.extend(pc_null.axis_iter(Axis(1)).map(|c| c.to_owned()));
                    }
                    m_cols.extend(
                        pc_range_bases[pc_idx]
                            .axis_iter(Axis(1))
                            .map(|c| c.to_owned()),
                    );
                    let m_matrix = ndarray::stack(
                        Axis(1),
                        &m_cols.iter().map(|c| c.view()).collect::<Vec<_>>(),
                    )
                    .map_err(|_| ModelError::InternalStackingError)?;

                    // Dimension check: alpha: m_cols x t_cols
                    if m_matrix.ncols() != alpha.nrows()
                        || tensor_interaction.ncols() != alpha.ncols()
                    {
                        return Err(ModelError::DimensionMismatch(format!(
                            "Orth map dims mismatch for {}: M {}x{}, alpha {}x{}, T {}x{}",
                            tensor_key,
                            m_matrix.nrows(),
                            m_matrix.ncols(),
                            alpha.nrows(),
                            alpha.ncols(),
                            tensor_interaction.nrows(),
                            tensor_interaction.ncols()
                        )));
                    }
                    tensor_interaction = &tensor_interaction - &m_matrix.dot(alpha);

                    // Align the number of interaction columns with the stored coefficients if available.
                    let constructed_cols = tensor_interaction.ncols();
                    if let Some(stored) = coeffs.interaction_effects.get(&tensor_key) {
                        let stored_cols = stored.len();
                        let expected_product = pgs_int_basis.ncols() * pc_int_basis.ncols();
                        if stored_cols != expected_product {
                            return Err(ModelError::DimensionMismatch(format!(
                                "Stored interaction coefficient count for {tensor_key} is {stored_cols}, but the marginal bases imply {expected_product} columns ({}×{}). Common causes: knot vector length/degree drift or stale range/orthogonalization transforms.",
                                pgs_int_basis.ncols(),
                                pc_int_basis.ncols(),
                            )));
                        }

                        if stored_cols != constructed_cols {
                            return Err(ModelError::DimensionMismatch(format!(
                                "Interaction columns mismatch for {tensor_key}: constructed {constructed_cols} ({}×{}) vs stored {stored_cols}. Compare knot_vectors['pgs'], knot_vectors['{pc_name}'], degree, range_transforms, sum_to_zero constraints, pc_null_transforms, and interaction_orth_alpha shapes.",
                                pgs_int_basis.ncols(),
                                pc_int_basis.ncols(),
                                pc_name = pc_name.as_str(),
                            )));
                        }
                    }

                    for col in tensor_interaction.axis_iter(Axis(1)) {
                        owned_cols.push(col.to_owned());
                    }
                }
            }
        }
        // No fallback - only modern approach is supported now

        // Stack all column views into the final design matrix
        let col_views: Vec<_> = owned_cols.iter().map(|col| col.view()).collect();
        ndarray::stack(Axis(1), &col_views).map_err(|_| ModelError::InternalStackingError)
    }

    /// Flattens the structured `MappedCoefficients` into a single `Array1` vector,
    /// following the exact same canonical order used in `construct_design_matrix`.
    /// This function is the "mirror image" of the design matrix construction.
    pub(super) fn flatten_coefficients(
        coeffs: &MappedCoefficients,
        config: &ModelConfig,
    ) -> Result<Array1<f64>, ModelError> {
        let mut flattened: Vec<f64> = Vec::new();

        // Order of concatenation MUST exactly match `construct_design_matrix`.
        // Stage: Intercept
        flattened.push(coeffs.intercept);

        // Stage: Sex main effect
        flattened.push(coeffs.main_effects.sex);

        // Stage: Main PC effects (ordered by pc_configs for determinism)
        for pc_config in &config.pc_configs {
            let pc_name = &pc_config.name;
            let c = coeffs
                .main_effects
                .pcs
                .get(pc_name)
                .ok_or_else(|| ModelError::CoefficientMissing(pc_name.clone()))?;
            flattened.extend_from_slice(c);
        }

        // Stage: Main PGS effects (for basis functions m > 0)
        flattened.extend_from_slice(&coeffs.main_effects.pgs);

        // Stage: Sex×PGS varying-coefficient interaction (if present)
        let sex_pgs_key = "f(PGS,sex)";
        if let Some(sex_pgs_coeffs) = coeffs.interaction_effects.get(sex_pgs_key) {
            flattened.extend_from_slice(sex_pgs_coeffs);
        }

        // Stage: Tensor product interaction effects (ordered by PC)
        for pc_config in &config.pc_configs {
            let tensor_key = format!("f(PGS,{})", pc_config.name);
            if let Some(tensor_coeffs) = coeffs.interaction_effects.get(&tensor_key) {
                flattened.extend_from_slice(tensor_coeffs);
            }
        }

        Ok(Array1::from_vec(flattened))
    }
}

pub fn internal_construct_design_matrix(
    p_new: ArrayView1<f64>,
    sex_new: ArrayView1<f64>,
    pcs_new: ArrayView2<f64>,
    config: &ModelConfig,
    coeffs: &MappedCoefficients,
) -> Result<Array2<f64>, ModelError> {
    internal::construct_design_matrix(p_new, sex_new, pcs_new, config, coeffs)
}

pub fn internal_flatten_coefficients(
    coeffs: &MappedCoefficients,
    config: &ModelConfig,
) -> Result<Array1<f64>, ModelError> {
    internal::flatten_coefficients(coeffs, config)
}

#[cfg(test)]
mod tests {
    use super::*;
    // TrainingData is not used in tests
    use approx::assert_abs_diff_eq;
    use ndarray::{Array1, Array2, array, s};

    fn sigmoid(x: f64) -> f64 {
        let x = x.clamp(-700.0, 700.0);
        1.0 / (1.0 + f64::exp(-x))
    }

    /// Test a trivially simple case with degree 1 B-spline (piecewise linear interpolation).
    /// This test uses a simple ground truth we can calculate by hand to verify the prediction.
    #[test]
    fn test_trained_model_predict() {
        // --- Setup a simple, known model configuration ---
        let knot_vector = array![0.0, 0.0, 0.5, 1.0, 1.0];
        let degree = 1;
        // For a degree=1 spline with this knot vector, there are 3 unconstrained basis functions.
        // The main effect (excluding the intercept) has 2 basis functions.
        // The sum-to-zero constraint will transform these 2 functions into 1 constrained function.
        // However, the test below will derive this transformation programmatically.

        // Create a dummy unconstrained basis to derive the constraint matrix Z
        let (unconstrained_basis_for_constraint, _) = create_basis::<Dense>(
            array![0.25, 0.75].view(), // two arbitrary points
            KnotSource::Provided(knot_vector.view()),
            degree,
            BasisOptions::value(),
        )
        .unwrap();
        let unconstrained_main_basis = unconstrained_basis_for_constraint.slice(s![.., 1..]);
        let (_, z_transform) =
            basis::apply_sum_to_zero_constraint(unconstrained_main_basis.view(), None).unwrap();

        let model = TrainedModel {
            config: ModelConfig {
                model_family: ModelFamily::Gam(LinkFunction::Identity),
                penalty_order: 2,
                convergence_tolerance: 1e-6,
                max_iterations: 100,
                reml_convergence_tolerance: 1e-6,
                reml_max_iterations: 50,
                firth_bias_reduction: false,
                reml_parallel_threshold: default_reml_parallel_threshold(),
                pgs_basis_config: BasisConfig {
                    num_knots: 1, // Number of internal knots
                    degree,
                },
                pc_configs: vec![],
                pgs_range: (0.0, 1.0),
                interaction_penalty: InteractionPenaltyKind::Anisotropic,
                sum_to_zero_constraints: {
                    let mut constraints = HashMap::new();
                    constraints.insert("pgs_main".to_string(), z_transform.clone());
                    constraints
                },
                knot_vectors: {
                    let mut knots = HashMap::new();
                    knots.insert("pgs".to_string(), knot_vector.clone());
                    knots
                },
                range_transforms: HashMap::new(),
                pc_null_transforms: HashMap::new(),
                interaction_centering_means: HashMap::new(),
                interaction_orth_alpha: HashMap::new(),
                mcmc_enabled: false,
                calibrator_enabled: false,
                survival: None,
            },
            coefficients: MappedCoefficients {
                intercept: 0.5, // Added an intercept for a more complete test
                main_effects: MainEffects {
                    sex: 0.5,
                    // There is only 1 coefficient after constraint for this simple case
                    pgs: vec![2.0],
                    pcs: HashMap::new(),
                },
                interaction_effects: HashMap::new(),
            },
            lambdas: vec![],
            hull: None,
            penalized_hessian: None,
            scale: None,
            calibrator: None,
            joint_link: None,
            survival: None,
            survival_companions: HashMap::new(),
            mcmc_samples: None,
            smoothing_correction: None,
        };

        // --- Define Test Points ---
        let test_points = array![0.25, 0.75];
        let sex_points = array![0.0, 1.0];
        let empty_pcs = Array2::<f64>::zeros((2, 0));

        // --- Calculate the expected result CORRECTLY ---
        // The manual calculation was flawed. Here's the correct way to derive the ground truth:
        // Stage: Generate the raw, unconstrained basis at the test points
        let (full_basis_unc, _) = create_basis::<Dense>(
            test_points.view(),
            KnotSource::Provided(knot_vector.view()),
            degree,
            BasisOptions::value(),
        )
        .unwrap();

        // Stage: Isolate the main effect part of the basis (all columns except the intercept)
        let pgs_main_basis_unc = full_basis_unc.slice(s![.., 1..]);

        // Stage: Apply the same sum-to-zero constraint transformation
        let pgs_main_basis_con = pgs_main_basis_unc.dot(&z_transform);

        // Stage: Get the coefficients for the constrained basis
        let coeffs = Array1::from(model.coefficients.main_effects.pgs.clone());

        // Stage: Calculate the final expected linear predictor as intercept + constrained_basis * coeffs
        let mut expected_values = pgs_main_basis_con.dot(&coeffs);
        expected_values += &sex_points.mapv(|v| v * model.coefficients.main_effects.sex);
        expected_values += model.coefficients.intercept;

        // Get the model's prediction using the actual `predict` method
        let predictions = model
            .predict(test_points.view(), sex_points.view(), empty_pcs.view())
            .unwrap();

        // Verify the results match our correctly calculated ground truth
        assert_eq!(predictions.len(), 2);
        assert_abs_diff_eq!(
            predictions.as_slice().unwrap(),
            expected_values.as_slice().unwrap(),
            epsilon = 1e-10
        );
    }

    #[test]
    fn predict_detailed_uses_mcmc_mean_for_logit() {
        let knot_vector = array![0.0, 0.0, 0.5, 1.0, 1.0];
        let degree = 1;
        let (unconstrained_basis_for_constraint, _) = create_basis::<Dense>(
            array![0.25, 0.75].view(),
            KnotSource::Provided(knot_vector.view()),
            degree,
            BasisOptions::value(),
        )
        .unwrap();
        let unconstrained_main_basis = unconstrained_basis_for_constraint.slice(s![.., 1..]);
        let (_, z_transform) =
            basis::apply_sum_to_zero_constraint(unconstrained_main_basis.view(), None).unwrap();

        let mut model = TrainedModel {
            config: ModelConfig {
                model_family: ModelFamily::Gam(LinkFunction::Logit),
                penalty_order: 2,
                convergence_tolerance: 1e-6,
                max_iterations: 100,
                reml_convergence_tolerance: 1e-6,
                reml_max_iterations: 50,
                firth_bias_reduction: false,
                reml_parallel_threshold: default_reml_parallel_threshold(),
                pgs_basis_config: BasisConfig {
                    num_knots: 1,
                    degree,
                },
                pc_configs: vec![],
                pgs_range: (0.0, 1.0),
                interaction_penalty: InteractionPenaltyKind::Anisotropic,
                sum_to_zero_constraints: {
                    let mut constraints = HashMap::new();
                    constraints.insert("pgs_main".to_string(), z_transform.clone());
                    constraints
                },
                knot_vectors: {
                    let mut knots = HashMap::new();
                    knots.insert("pgs".to_string(), knot_vector.clone());
                    knots
                },
                range_transforms: HashMap::new(),
                pc_null_transforms: HashMap::new(),
                interaction_centering_means: HashMap::new(),
                interaction_orth_alpha: HashMap::new(),
                mcmc_enabled: true,
                calibrator_enabled: false,
                survival: None,
            },
            coefficients: MappedCoefficients {
                intercept: 0.5,
                main_effects: MainEffects {
                    sex: 0.5,
                    pgs: vec![2.0],
                    pcs: HashMap::new(),
                },
                interaction_effects: HashMap::new(),
            },
            lambdas: vec![],
            hull: None,
            penalized_hessian: None,
            scale: None,
            calibrator: None,
            joint_link: None,
            survival: None,
            survival_companions: HashMap::new(),
            mcmc_samples: None,
            smoothing_correction: None,
        };

        let test_points = array![0.25, 0.75];
        let sex_points = array![0.0, 1.0];
        let empty_pcs = Array2::<f64>::zeros((2, 0));
        let x_new = internal_construct_design_matrix(
            test_points.view(),
            sex_points.view(),
            empty_pcs.view(),
            &model.config,
            &model.coefficients,
        )
        .unwrap();
        let p = x_new.ncols();
        let mut samples = Array2::<f64>::zeros((2, p));
        samples.row_mut(1).fill(1.0);
        model.mcmc_samples = Some(samples);

        let (eta, mean, _, _) = model
            .predict_detailed(test_points.view(), sex_points.view(), empty_pcs.view())
            .unwrap();
        let pred = model
            .predict(test_points.view(), sex_points.view(), empty_pcs.view())
            .unwrap();

        let eta_ones = x_new.dot(&Array1::ones(p));
        let expected_mean = eta_ones.mapv(|e| 0.5 * (0.5 + sigmoid(e)));
        let expected_eta = eta_ones.mapv(|e| 0.5 * e);

        assert_abs_diff_eq!(
            mean.as_slice().unwrap(),
            expected_mean.as_slice().unwrap(),
            epsilon = 1e-10
        );
        assert_abs_diff_eq!(
            pred.as_slice().unwrap(),
            expected_mean.as_slice().unwrap(),
            epsilon = 1e-10
        );
        assert_abs_diff_eq!(
            eta.as_slice().unwrap(),
            expected_eta.as_slice().unwrap(),
            epsilon = 1e-10
        );
    }

    #[test]
    fn identity_prediction_ignores_mcmc_samples() {
        let knot_vector = array![0.0, 0.0, 0.5, 1.0, 1.0];
        let degree = 1;
        let (unconstrained_basis_for_constraint, _) = create_basis::<Dense>(
            array![0.25, 0.75].view(),
            KnotSource::Provided(knot_vector.view()),
            degree,
            BasisOptions::value(),
        )
        .unwrap();
        let unconstrained_main_basis = unconstrained_basis_for_constraint.slice(s![.., 1..]);
        let (_, z_transform) =
            basis::apply_sum_to_zero_constraint(unconstrained_main_basis.view(), None).unwrap();

        let model = TrainedModel {
            config: ModelConfig {
                model_family: ModelFamily::Gam(LinkFunction::Identity),
                penalty_order: 2,
                convergence_tolerance: 1e-6,
                max_iterations: 100,
                reml_convergence_tolerance: 1e-6,
                reml_max_iterations: 50,
                firth_bias_reduction: false,
                reml_parallel_threshold: default_reml_parallel_threshold(),
                pgs_basis_config: BasisConfig {
                    num_knots: 1,
                    degree,
                },
                pc_configs: vec![],
                pgs_range: (0.0, 1.0),
                interaction_penalty: InteractionPenaltyKind::Anisotropic,
                sum_to_zero_constraints: {
                    let mut constraints = HashMap::new();
                    constraints.insert("pgs_main".to_string(), z_transform.clone());
                    constraints
                },
                knot_vectors: {
                    let mut knots = HashMap::new();
                    knots.insert("pgs".to_string(), knot_vector.clone());
                    knots
                },
                range_transforms: HashMap::new(),
                pc_null_transforms: HashMap::new(),
                interaction_centering_means: HashMap::new(),
                interaction_orth_alpha: HashMap::new(),
                mcmc_enabled: true,
                calibrator_enabled: false,
                survival: None,
            },
            coefficients: MappedCoefficients {
                intercept: 0.5,
                main_effects: MainEffects {
                    sex: 0.5,
                    pgs: vec![2.0],
                    pcs: HashMap::new(),
                },
                interaction_effects: HashMap::new(),
            },
            lambdas: vec![],
            hull: None,
            penalized_hessian: None,
            scale: None,
            calibrator: None,
            joint_link: None,
            survival: None,
            survival_companions: HashMap::new(),
            mcmc_samples: Some(Array2::<f64>::ones((2, 3))),
            smoothing_correction: None,
        };

        let test_points = array![0.25, 0.75];
        let sex_points = array![0.0, 1.0];
        let empty_pcs = Array2::<f64>::zeros((2, 0));
        let predictions = model
            .predict(test_points.view(), sex_points.view(), empty_pcs.view())
            .unwrap();

        let (full_basis_unc, _) = create_basis::<Dense>(
            test_points.view(),
            KnotSource::Provided(knot_vector.view()),
            degree,
            BasisOptions::value(),
        )
        .unwrap();
        let pgs_main_basis_unc = full_basis_unc.slice(s![.., 1..]);
        let pgs_main_basis_con = pgs_main_basis_unc.dot(&z_transform);
        let coeffs = Array1::from(model.coefficients.main_effects.pgs.clone());
        let mut expected_values = pgs_main_basis_con.dot(&coeffs);
        expected_values += &sex_points.mapv(|v| v * model.coefficients.main_effects.sex);
        expected_values += model.coefficients.intercept;

        assert_abs_diff_eq!(
            predictions.as_slice().unwrap(),
            expected_values.as_slice().unwrap(),
            epsilon = 1e-10
        );
    }

    #[test]
    fn survival_prediction_produces_risk_and_se() {
        use crate::survival::{
            BasisDescriptor, CholeskyFactor, HessianFactor, SurvivalModelArtifacts,
            SurvivalLayoutInputs, build_survival_layout,
        };

        let data = SurvivalLayoutInputs {
            age_entry: array![50.0, 55.0],
            age_exit: array![55.0, 60.0],
            event_target: array![1, 0],
            event_competing: array![0, 0],
            sample_weight: array![1.0, 1.0],
            static_covariates: array![[0.2, 0.0], [-0.1, 1.0]],
            static_covariate_names: vec!["pgs".to_string(), "sex".to_string()],
            time_varying_covariate: Some(array![0.2, -0.1]),
        };
        let basis = BasisDescriptor {
            knot_vector: array![0.0, 0.0, 0.0, 0.5, 1.0, 1.0, 1.0],
            degree: 2,
        };
        let layout_bundle = build_survival_layout(&data, &basis, 0.1, 2, 4, None).unwrap();
        let layout = layout_bundle.layout;
        let coeffs = Array1::from_elem(layout.combined_exit.ncols(), 0.1);

        let column_names: Vec<String> = (0..layout.static_covariates.ncols())
            .map(|idx| format!("cov{idx}"))
            .collect();
        let mut ranges = Vec::new();
        for col_idx in 0..layout.static_covariates.ncols() {
            let column = layout.static_covariates.column(col_idx);
            let min_val = column.iter().copied().fold(f64::INFINITY, f64::min);
            let max_val = column.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            ranges.push(crate::survival::ValueRange {
                min: min_val,
                max: max_val,
            });
        }
        let artifacts = SurvivalModelArtifacts {
            coefficients: coeffs.clone(),
            age_basis: basis.clone(),
            time_varying_basis: None,
            static_covariate_layout: crate::survival::CovariateLayout {
                column_names,
                ranges,
            },
            penalties: Vec::new(),
            age_transform: layout.age_transform,
            reference_constraint: layout.reference_constraint.clone(),
            monotonicity: layout.monotonicity.clone(),
            interaction_metadata: Vec::new(),
            companion_models: Vec::new(),
            hessian_factor: Some(HessianFactor::Expected {
                factor: CholeskyFactor {
                    lower: Array2::eye(layout.combined_exit.ncols()),
                },
            }),
            calibrator: None,
            mcmc_samples: None,
            cross_covariance_to_primary: None,
        };

        let config = ModelConfig {
            model_family: ModelFamily::Survival(SurvivalSpec::default()),
            penalty_order: 2,
            convergence_tolerance: 1e-6,
            max_iterations: 20,
            reml_convergence_tolerance: 1e-6,
            reml_max_iterations: 20,
            firth_bias_reduction: false,
            reml_parallel_threshold: default_reml_parallel_threshold(),
            pgs_basis_config: BasisConfig {
                num_knots: 0,
                degree: 0,
            },
            pc_configs: Vec::new(),
            pgs_range: (0.0, 1.0),
            interaction_penalty: InteractionPenaltyKind::Anisotropic,
            sum_to_zero_constraints: HashMap::new(),
            knot_vectors: HashMap::new(),
            range_transforms: HashMap::new(),
            interaction_centering_means: HashMap::new(),
            interaction_orth_alpha: HashMap::new(),
            pc_null_transforms: HashMap::new(),
            mcmc_enabled: false,
            calibrator_enabled: false,
            survival: Some(SurvivalModelConfig {
                baseline_basis: BasisConfig {
                    num_knots: basis.knot_vector.len(),
                    degree: basis.degree,
                },
                guard_delta: 0.1,
                monotonic_grid_size: 4,
                time_varying: None,
                model_competing_risk: false,
            }),
        };

        let model = TrainedModel {
            config,
            coefficients: MappedCoefficients::default(),
            lambdas: Vec::new(),
            hull: None,
            penalized_hessian: None,
            scale: None,
            calibrator: None,
            joint_link: None,
            survival: Some(artifacts),
            survival_companions: HashMap::new(),
            mcmc_samples: None,
            smoothing_correction: None,
        };

        let result = model
            .predict_survival(
                data.age_entry.view(),
                data.age_exit.view(),
                data.static_covariates.column(0),
                data.static_covariates.column(1),
                data.static_covariates.slice(s![.., 2..]),
                SurvivalRiskType::Net,
                None,
            )
            .expect("survival prediction succeeded");

        assert_eq!(result.conditional_risk.len(), 2);
        assert!(
            result
                .conditional_risk
                .iter()
                .all(|value| value.is_finite())
        );
        let se = result.logit_risk_se.expect("delta-method se available");
        assert_eq!(se.len(), 2);
        for i in 0..se.len() {
            let design = result.logit_risk_design.as_ref().expect("design available");
            let grad = design.row(i);
            let expected = grad.iter().map(|v| v * v).sum::<f64>().sqrt();
            assert!((se[i] - expected).abs() < 1e-9);
        }
    }

    /// Tests that the prediction fails appropriately with invalid input dimensions.
    #[test]
    fn test_trained_model_predict_invalid_input() {
        let model = TrainedModel {
            config: ModelConfig {
                model_family: ModelFamily::Gam(LinkFunction::Identity),
                penalty_order: 2,
                convergence_tolerance: 1e-6,
                max_iterations: 100,
                reml_convergence_tolerance: 1e-6,
                reml_max_iterations: 50,
                firth_bias_reduction: false,
                reml_parallel_threshold: default_reml_parallel_threshold(),
                pgs_basis_config: BasisConfig {
                    num_knots: 2,
                    degree: 1,
                },
                pc_configs: vec![PrincipalComponentConfig {
                    name: "PC1".to_string(),
                    basis_config: BasisConfig {
                        num_knots: 2,
                        degree: 1,
                    },
                    range: (-1.0, 1.0),
                }],
                pgs_range: (-2.0, 2.0),
                interaction_penalty: InteractionPenaltyKind::Anisotropic,
                sum_to_zero_constraints: HashMap::new(),
                knot_vectors: HashMap::new(),
                range_transforms: HashMap::new(),
                pc_null_transforms: HashMap::new(),
                interaction_centering_means: HashMap::new(),
                interaction_orth_alpha: HashMap::new(),
                mcmc_enabled: false,
                calibrator_enabled: false,
                survival: None,
            },
            coefficients: MappedCoefficients {
                intercept: 0.0,
                main_effects: MainEffects {
                    sex: 0.0,
                    pgs: vec![],
                    pcs: HashMap::new(),
                },
                interaction_effects: HashMap::new(),
            },
            lambdas: vec![],
            hull: None,
            penalized_hessian: None,
            scale: None,
            calibrator: None,
            joint_link: None,
            survival: None,
            survival_companions: HashMap::new(),
            mcmc_samples: None,
            smoothing_correction: None,
        };

        // Test with mismatched PC dimensions (model expects 1 PC, but we provide 2)
        let pgs = Array1::linspace(0.0, 1.0, 5);
        let sex = Array1::zeros(5);
        let pcs = Array2::zeros((5, 2)); // 2 PC columns, but model expects 1

        let result = model.predict(pgs.view(), sex.view(), pcs.view());

        // Verify we get the expected error
        assert!(
            result.is_err(),
            "Should return an error when PC dimensions don't match"
        );
        if let Err(ModelError::MismatchedPcCount { found, expected }) = result {
            assert_eq!(found, 2);
            assert_eq!(expected, 1);
        } else {
            panic!("Expected MismatchedPcCount error");
        }
    }

    /// Tests that the coefficient flattening logic works correctly.
    /// This test verifies that the structured MappedCoefficients are correctly
    /// flattened into a single vector following the canonical order.
    #[test]
    fn test_coefficient_flattening() {
        // Create a small structured coefficient object with known values
        let coeffs = MappedCoefficients {
            intercept: 1.0,
            main_effects: MainEffects {
                sex: 0.5,
                pgs: vec![2.0, 3.0], // 2 PGS main effect coefficients
                pcs: {
                    let mut pc_map = HashMap::new();
                    pc_map.insert("PC1".to_string(), vec![4.0, 5.0]); // 2 PC1 coefficients
                    pc_map.insert("PC2".to_string(), vec![6.0, 7.0]); // 2 PC2 coefficients
                    pc_map
                },
            },
            interaction_effects: {
                let mut interactions = HashMap::new();

                interactions.insert("f(PGS,sex)".to_string(), vec![16.0, 17.0]);

                // Unified tensor product interactions - flattened coefficient vectors
                // For tensor products, coefficients are flattened in the order: pgs_basis × pc_basis
                interactions.insert("f(PGS,PC1)".to_string(), vec![8.0, 9.0, 12.0, 13.0]);
                interactions.insert("f(PGS,PC2)".to_string(), vec![10.0, 11.0, 14.0, 15.0]);

                interactions
            },
        };

        // Create a simple model config for testing
        let config = ModelConfig {
            model_family: ModelFamily::Gam(LinkFunction::Identity),
            penalty_order: 2,
            convergence_tolerance: 1e-6,
            max_iterations: 100,
            reml_convergence_tolerance: 1e-6,
            reml_max_iterations: 50,
            firth_bias_reduction: false,
            reml_parallel_threshold: default_reml_parallel_threshold(),
            pgs_basis_config: BasisConfig {
                num_knots: 3,
                degree: 3,
            },
            pc_configs: vec![
                PrincipalComponentConfig {
                    name: "PC1".to_string(),
                    basis_config: BasisConfig {
                        num_knots: 3,
                        degree: 3,
                    },
                    range: (0.0, 1.0),
                },
                PrincipalComponentConfig {
                    name: "PC2".to_string(),
                    basis_config: BasisConfig {
                        num_knots: 3,
                        degree: 3,
                    },
                    range: (0.0, 1.0),
                },
            ],
            pgs_range: (0.0, 1.0),
            interaction_penalty: InteractionPenaltyKind::Anisotropic,
            sum_to_zero_constraints: HashMap::new(),
            knot_vectors: HashMap::new(),
            range_transforms: HashMap::new(),
            pc_null_transforms: HashMap::new(),
            interaction_centering_means: HashMap::new(),
            interaction_orth_alpha: HashMap::new(),
            mcmc_enabled: false,
            calibrator_enabled: false,
            survival: None,
        };

        // Use the internal flatten_coefficients function
        use super::internal::flatten_coefficients;
        let flattened = flatten_coefficients(&coeffs, &config)
            .expect("flatten_coefficients should succeed for well-formed inputs");

        // Verify the canonical order of flattening:
        // Stage: Intercept
        // Stage: PC main effects (ordered by config.pc_configs)
        // Stage: PGS main effects
        // Stage: Interaction effects (PGS basis function 1, then PC1, PC2, etc.; followed by additional PGS basis functions)

        // Define expected order based on the canonical ordering rules
        let expected = vec![
            1.0, // Intercept
            0.5, // Sex main effect
            4.0, 5.0, // PC1 main effects
            6.0, 7.0, // PC2 main effects
            2.0, 3.0, // PGS main effects
            16.0, 17.0, // f(PGS,sex) varying coefficient
            8.0, 9.0, 12.0, 13.0, // f(PGS,PC1) tensor product interaction
            10.0, 11.0, 14.0, 15.0, // f(PGS,PC2) tensor product interaction
        ];

        // Convert to vectors for easier comparison
        let flat_vec = flattened.to_vec();

        // Check total length
        assert_eq!(
            flat_vec.len(),
            expected.len(),
            "Flattened vector has incorrect length: expected {}, got {}",
            expected.len(),
            flat_vec.len()
        );

        // Check each element in the expected order
        for (i, &expected_val) in expected.iter().enumerate() {
            assert_eq!(
                flat_vec[i], expected_val,
                "Mismatch at position {}: expected {}, got {}",
                i, expected_val, flat_vec[i]
            );
        }
    }

    /// Tests that the model can be saved to and loaded from a file,
    /// preserving all its contents exactly.
    #[test]
    fn test_save_load_functionality() {
        use crate::construction::TrainingData;

        // Define the BasisConfig to use
        let pgs_basis_config = BasisConfig {
            num_knots: 6,
            degree: 3,
        };
        let pc1_basis_config = BasisConfig {
            num_knots: 6,
            degree: 3,
        };

        // Create a ModelConfig to use for both generation and testing
        let model_config = ModelConfig {
            model_family: ModelFamily::Gam(LinkFunction::Logit),
            penalty_order: 2,
            convergence_tolerance: 1e-6,
            max_iterations: 100,
            reml_convergence_tolerance: 1e-6,
            reml_max_iterations: 50,
            firth_bias_reduction: false,
            reml_parallel_threshold: default_reml_parallel_threshold(),
            pgs_basis_config: pgs_basis_config.clone(),
            pc_configs: vec![PrincipalComponentConfig {
                name: "PC1".to_string(),
                basis_config: pc1_basis_config.clone(),
                range: (-0.5, 0.5),
            }],
            pgs_range: (-1.0, 1.0),
            interaction_penalty: InteractionPenaltyKind::Anisotropic,
            sum_to_zero_constraints: HashMap::new(), // Will be populated by build_design_and_penalty_matrices
            knot_vectors: HashMap::new(), // Will be populated by build_design_and_penalty_matrices
            range_transforms: HashMap::new(), // Will be populated by build_design_and_penalty_matrices
            pc_null_transforms: HashMap::new(),
            interaction_centering_means: HashMap::new(),
            interaction_orth_alpha: HashMap::new(),
            mcmc_enabled: false,
            calibrator_enabled: false,
            survival: None,
        };

        // Create a dummy dataset for generating the correct model structure
        // Choose a sample size that comfortably exceeds the total number of coefficients in this
        // configuration (currently 101) so that the design matrix construction succeeds even as
        // new unpenalized effects such as sex are introduced.
        let n_samples = 150;
        let dummy_data = TrainingData {
            y: Array1::linspace(0.0, 1.0, n_samples),
            p: Array1::linspace(-1.0, 1.0, n_samples),
            sex: Array1::from_iter((0..n_samples).map(|i| (i % 2) as f64)),
            pcs: Array2::from_shape_vec(
                (n_samples, 1),
                Array1::linspace(-0.5, 0.5, n_samples).to_vec(),
            )
            .unwrap(),
            weights: Array1::<f64>::ones(n_samples),
        };

        // Generate the correct constraints, structure, and range transforms using the actual model-building code
        let (
            _,
            _,
            layout,
            sum_to_zero_constraints,
            knot_vectors,
            range_transforms,
            pc_null_transforms,
            interaction_centering_means,
            interaction_orth_alpha,
            _,
        ) = crate::construction::build_design_and_penalty_matrices(
            &dummy_data,
            &model_config,
        )
        .expect("Failed to build model matrices");

        // Now we can create a test model with the correct structure
        let original_model = TrainedModel {
            config: ModelConfig {
                model_family: ModelFamily::Gam(LinkFunction::Logit),
                penalty_order: 2,
                convergence_tolerance: 1e-6,
                max_iterations: 100,
                reml_convergence_tolerance: 1e-6,
                reml_max_iterations: 50,
                firth_bias_reduction: false,
                reml_parallel_threshold: default_reml_parallel_threshold(),
                pgs_basis_config: pgs_basis_config.clone(),
                pc_configs: vec![PrincipalComponentConfig {
                    name: "PC1".to_string(),
                    basis_config: pc1_basis_config.clone(),
                    range: (-0.5, 0.5),
                }],
                pgs_range: (-1.0, 1.0),
                interaction_penalty: InteractionPenaltyKind::Anisotropic,
                sum_to_zero_constraints: sum_to_zero_constraints.clone(), // Clone the new field
                knot_vectors, // Use the knot vectors generated by the model-building code
                range_transforms: range_transforms.clone(), // Use full Option 3 pipeline
                pc_null_transforms: pc_null_transforms.clone(),
                interaction_centering_means: interaction_centering_means.clone(),
                interaction_orth_alpha: interaction_orth_alpha.clone(),
                mcmc_enabled: false,
                calibrator_enabled: false,
                survival: None,
            },
            coefficients: MappedCoefficients {
                intercept: 0.5,
                main_effects: MainEffects {
                    sex: 0.75,
                    // The number of PGS main effect coefficients must match dimensions after constraint
                    // This size is derived from the layout object which reflects the actual model structure
                    pgs: {
                        let pgs_dim = layout.pgs_main_cols.end - layout.pgs_main_cols.start;
                        // Fill with increasing values 1.0, 2.0, 3.0, ...
                        (1..=pgs_dim).map(|i| i as f64).collect()
                    },
                    pcs: {
                        // Use null + range dimension for PC1
                        let pc1_dim_null = pc_null_transforms
                            .get("PC1")
                            .map(|z| z.ncols())
                            .unwrap_or(0);
                        let pc1_dim_range = range_transforms.get("PC1").unwrap().ncols();

                        let mut pc_map = HashMap::new();
                        pc_map.insert(
                            "PC1".to_string(),
                            (1..=(pc1_dim_null + pc1_dim_range))
                                .map(|i| i as f64)
                                .collect(),
                        );
                        pc_map
                    },
                },
                interaction_effects: {
                    let mut interactions = HashMap::new();

                    if let Some(block_idx) = layout.sex_pgs_block_idx {
                        let block = layout
                            .penalty_map
                            .get(block_idx)
                            .expect("sex×PGS block missing from layout.penalty_map");
                        let num_cols = block.col_range.end - block.col_range.start;
                        let coeffs: Vec<f64> = (1..=num_cols).map(|i| i as f64 * 5.0).collect();
                        interactions.insert("f(PGS,sex)".to_string(), coeffs);
                    }

                    for (pc_idx, pc_cfg) in model_config.pc_configs.iter().enumerate() {
                        let block_idx = *layout
                            .interaction_block_idx
                            .get(pc_idx)
                            .expect("interaction block index missing");
                        let block = layout
                            .penalty_map
                            .get(block_idx)
                            .expect("interaction block missing from layout.penalty_map");
                        let num_cols = block.col_range.end - block.col_range.start;

                        let coeffs: Vec<f64> = (1..=num_cols).map(|i| i as f64 * 10.0).collect();

                        interactions.insert(format!("f(PGS,{})", pc_cfg.name), coeffs);
                    }

                    interactions
                },
            },
            // Match layout.num_penalties dynamically (PC null + PC range + interaction)
            lambdas: vec![0.1; layout.num_penalties],
            hull: None,
            penalized_hessian: None,
            scale: None,
            calibrator: None,
            joint_link: None,
            survival: None,
            survival_companions: HashMap::new(),
            mcmc_samples: None,
            smoothing_correction: None,
        };

        // Create a temporary file for testing
        use tempfile::NamedTempFile;
        let temp_file = NamedTempFile::new().expect("Failed to create temp file");
        let file_path = temp_file.path().to_str().unwrap();

        // Save the model
        original_model
            .save(file_path)
            .expect("Failed to save model");

        // Load the model back
        let loaded_model = TrainedModel::load(file_path).expect("Failed to load model");

        // Compare the models

        // Stage: Check model configuration
        assert_eq!(
            loaded_model
                .config
                .link_function()
                .expect("link_function called on survival model"),
            original_model
                .config
                .link_function()
                .expect("link_function called on survival model")
        );
        assert_eq!(
            loaded_model.config.penalty_order,
            original_model.config.penalty_order
        );
        assert_eq!(
            loaded_model.config.pgs_basis_config.num_knots,
            original_model.config.pgs_basis_config.num_knots
        );
        assert_eq!(
            loaded_model.config.pgs_basis_config.degree,
            original_model.config.pgs_basis_config.degree
        );
        assert_eq!(
            loaded_model.config.pc_configs.len(),
            original_model.config.pc_configs.len()
        );
        // Check that each PC config matches
        for i in 0..loaded_model.config.pc_configs.len() {
            assert_eq!(
                loaded_model.config.pc_configs[i].name,
                original_model.config.pc_configs[i].name
            );
            assert_eq!(
                loaded_model.config.pc_configs[i].range,
                original_model.config.pc_configs[i].range
            );
            assert_eq!(
                loaded_model.config.pc_configs[i].basis_config.num_knots,
                original_model.config.pc_configs[i].basis_config.num_knots
            );
            assert_eq!(
                loaded_model.config.pc_configs[i].basis_config.degree,
                original_model.config.pc_configs[i].basis_config.degree
            );
        }
        assert_eq!(
            loaded_model.config.pgs_range,
            original_model.config.pgs_range
        );

        // Stage: Confirm sum_to_zero_constraints transformations exist for the PGS main effect only
        assert!(
            loaded_model
                .config
                .sum_to_zero_constraints
                .contains_key("pgs_main")
        );

        // Stage: Check coefficient values
        assert_eq!(
            loaded_model.coefficients.intercept,
            original_model.coefficients.intercept
        );
        assert_eq!(
            loaded_model.coefficients.main_effects.pgs,
            original_model.coefficients.main_effects.pgs
        );
        assert_eq!(
            loaded_model.coefficients.main_effects.pcs.len(),
            original_model.coefficients.main_effects.pcs.len()
        );

        if let Some(pc1_loaded) = loaded_model.coefficients.main_effects.pcs.get("PC1") {
            if let Some(pc1_orig) = original_model.coefficients.main_effects.pcs.get("PC1") {
                assert_eq!(pc1_loaded, pc1_orig);
            } else {
                panic!("Original model missing PC1 main effects");
            }
        } else {
            panic!("Loaded model missing PC1 main effects");
        }

        // Stage: Check interaction effects
        assert_eq!(
            loaded_model.coefficients.interaction_effects.len(),
            original_model.coefficients.interaction_effects.len()
        );

        // Check for interaction effect f(PGS,PC1)
        let key_interaction = "f(PGS,PC1)";
        if let (Some(interaction_loaded), Some(interaction_orig)) = (
            loaded_model
                .coefficients
                .interaction_effects
                .get(key_interaction),
            original_model
                .coefficients
                .interaction_effects
                .get(key_interaction),
        ) {
            assert_eq!(
                interaction_loaded, interaction_orig,
                "Mismatch in {}",
                key_interaction
            );
        } else {
            panic!("Missing {} interaction effect", key_interaction);
        }

        // Check for sex×PGS interaction effect if present
        let key_sex = "f(PGS,sex)";
        if let Some(interaction_orig) = original_model.coefficients.interaction_effects.get(key_sex)
        {
            let interaction_loaded = loaded_model
                .coefficients
                .interaction_effects
                .get(key_sex)
                .expect("Missing f(PGS,sex) interaction effect");
            assert_eq!(
                interaction_loaded, interaction_orig,
                "Mismatch in {}",
                key_sex
            );
        }

        // Stage: Check lambdas
        assert_eq!(loaded_model.lambdas, original_model.lambdas);

        // Stage: Verify that we can use the loaded model for prediction
        let test_pgs = Array1::linspace(-0.5, 0.5, 3);
        let test_pcs = Array2::from_shape_fn((3, 1), |(i, _)| {
            (i as f64 - 1.0) * 0.25 // Values: -0.25, 0, 0.25
        });

        let test_sex = Array1::zeros(test_pgs.len());

        let predictions_orig = original_model
            .predict(test_pgs.view(), test_sex.view(), test_pcs.view())
            .expect("Prediction with original model failed");

        let predictions_loaded = loaded_model
            .predict(test_pgs.view(), test_sex.view(), test_pcs.view())
            .expect("Prediction with loaded model failed");

        // Verify that predictions are identical
        for i in 0..predictions_orig.len() {
            assert_eq!(
                predictions_orig[i], predictions_loaded[i],
                "Predictions differ at position {}",
                i
            );
        }
    }
}

/// Maps the flattened coefficient vector to a structured representation.
pub(crate) fn map_coefficients(
    beta: &Array1<f64>,
    layout: &EngineLayout,
) -> Result<MappedCoefficients, EstimationError> {
    let intercept = beta[layout.intercept_col];
    let mut pcs = HashMap::new();
    let mut pgs = vec![];
    let mut interaction_effects = HashMap::new();

    let sex = layout.sex_col.map(|col| beta[col]).unwrap_or(0.0);

    // Extract the unpenalized PGS main effect coefficients
    if !layout.pgs_main_cols.is_empty() {
        pgs = beta.slice(s![layout.pgs_main_cols.clone()]).to_vec();
    }

    // Build PC main-effect coefficients deterministically by layout indices
    // For each PCj: concatenate [null-space coeffs for PCj (if any), range-space coeffs for PCj].
    // Infer PC count from layout vectors; layout invariants guarantee alignment.
    let num_pcs = layout.pc_main_block_idx.len();
    for i in 0..num_pcs {
        let range_block_idx = layout.pc_main_block_idx[i];
        let range_block = &layout.penalty_map[range_block_idx];
        let mut full = Vec::new();
        let null_range = &layout.pc_null_cols[i];
        if !null_range.is_empty() {
            full.extend_from_slice(&beta.slice(s![null_range.clone()]).to_vec());
        }
        full.extend_from_slice(&beta.slice(s![range_block.col_range.clone()]).to_vec());

        // Derive PC name robustly from the range block term name, which is guaranteed to be f(PCname)
        let name_str = &range_block.term_name;
        let pc_name = if let Some(start) = name_str.find("f(") {
            let rest = &name_str[start + 2..];
            let end = rest.find(')').ok_or_else(|| {
                EstimationError::LayoutError(format!("Malformed PC term name: {}", name_str))
            })?;
            rest[..end].to_string()
        } else {
            return Err(EstimationError::LayoutError(format!(
                "Unexpected PC term name (missing prefix): {}",
                name_str
            )));
        };
        pcs.insert(pc_name, full);
    }

    // Build interaction effects deterministically by layout.interaction_block_idx order.
    if let Some(block_idx) = layout.sex_pgs_block_idx {
        let block = &layout.penalty_map[block_idx];
        let coeffs = beta.slice(s![block.col_range.clone()]).to_vec();
        interaction_effects.insert(block.term_name.to_string(), coeffs);
    }
    for &blk_idx in &layout.interaction_block_idx {
        let block = &layout.penalty_map[blk_idx];
        let coeffs = beta.slice(s![block.col_range.clone()]).to_vec();
        interaction_effects.insert(block.term_name.to_string(), coeffs);
    }

    Ok(MappedCoefficients {
        intercept,
        main_effects: MainEffects { sex, pgs, pcs },
        interaction_effects,
    })
}
