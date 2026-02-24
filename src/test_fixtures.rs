//! Test fixtures and builders for gnomon calibration tests.
//!
//! This module provides reusable builders for synthetic data generation
//! and default model configurations, reducing boilerplate across test files.

use crate::model::{
    BasisConfig, InteractionPenaltyKind, LinkFunction, MainEffects, MappedCoefficients,
    ModelConfig, ModelFamily, PrincipalComponentConfig, TrainedModel,
};
use ndarray::{Array1, Array2};
use rand::rngs::StdRng;
use rand::{RngExt, SeedableRng};
use rand_distr::StandardNormal;
use std::collections::HashMap;

/// Type of signal to inject into synthetic data.
#[derive(Clone, Copy, Debug)]
pub enum SignalType {
    /// Linear relationship: y ~ β₀ + β₁·score
    Linear,
    /// Quadratic relationship: y ~ β₀ + β₁·score + β₂·score²  
    Quadratic,
    /// No signal (pure noise)
    Null,
}

/// Synthetic test data for GAM fitting.
#[derive(Clone)]
pub struct TestData {
    /// Response variable (binary for logit, continuous for identity)
    pub y: Array1<f64>,
    /// Polygenic score values
    pub pgs: Array1<f64>,
    /// Sex covariate (binary: 0 or 1)
    pub sex: Array1<f64>,
    /// Principal components matrix (n_samples × n_pcs)
    pub pcs: Array2<f64>,
    /// Prior weights (defaults to ones)
    pub weights: Array1<f64>,
}

impl TestData {
    /// Number of samples in the dataset.
    pub fn n(&self) -> usize {
        self.y.len()
    }

    /// Number of principal components.
    pub fn n_pcs(&self) -> usize {
        self.pcs.ncols()
    }
}

/// Builder for creating synthetic test data with configurable properties.
pub struct SyntheticDataBuilder {
    n_samples: usize,
    n_pcs: usize,
    prevalence: f64,
    signal_strength: f64,
    signal_type: SignalType,
    seed: u64,
    male_fraction: f64,
}

impl SyntheticDataBuilder {
    /// Create a new builder with defaults.
    pub fn new(n_samples: usize) -> Self {
        Self {
            n_samples,
            n_pcs: 2,
            prevalence: 0.1,
            signal_strength: 0.5,
            signal_type: SignalType::Linear,
            seed: 42,
            male_fraction: 0.5,
        }
    }

    /// Set the number of principal components.
    pub fn with_pcs(mut self, k: usize) -> Self {
        self.n_pcs = k;
        self
    }

    /// Set the case prevalence (for binary outcomes).
    pub fn with_prevalence(mut self, p: f64) -> Self {
        self.prevalence = p.clamp(0.01, 0.99);
        self
    }

    /// Set the signal strength (effect size).
    pub fn with_signal_strength(mut self, s: f64) -> Self {
        self.signal_strength = s;
        self
    }

    /// Set the signal type.
    pub fn with_signal(mut self, s: SignalType) -> Self {
        self.signal_type = s;
        self
    }

    /// Set the random seed for reproducibility.
    pub fn seed(mut self, s: u64) -> Self {
        self.seed = s;
        self
    }

    /// Set the fraction of male samples.
    pub fn with_male_fraction(mut self, f: f64) -> Self {
        self.male_fraction = f.clamp(0.0, 1.0);
        self
    }

    /// Build synthetic data for a logistic regression model (binary outcome).
    pub fn build_logit(self) -> TestData {
        let mut rng = StdRng::seed_from_u64(self.seed);
        let n = self.n_samples;

        // Generate PGS (standardized)
        let pgs: Array1<f64> = Array1::from_iter((0..n).map(|_| rng.sample(StandardNormal)));

        // Generate PCs (standardized)
        let pcs = Array2::from_shape_fn((n, self.n_pcs), |_| rng.sample(StandardNormal));

        // Generate sex
        let sex: Array1<f64> = Array1::from_iter((0..n).map(|_| {
            if rng.random::<f64>() < self.male_fraction {
                1.0
            } else {
                0.0
            }
        }));

        // Generate liability based on signal type
        let liability: Array1<f64> = match self.signal_type {
            SignalType::Linear => pgs.mapv(|p| self.signal_strength * p),
            SignalType::Quadratic => pgs.mapv(|p| self.signal_strength * p + 0.3 * p * p),
            SignalType::Null => Array1::zeros(n),
        };

        // Add noise to liability
        let noise: Array1<f64> = Array1::from_iter((0..n).map(|_| rng.sample(StandardNormal)));
        let total_liability = &liability + &noise;

        // Convert to binary outcome via threshold (to match desired prevalence)
        let mut indices: Vec<usize> = (0..n).collect();
        indices.sort_by(|&a, &b| total_liability[b].partial_cmp(&total_liability[a]).unwrap());

        let n_cases = ((n as f64) * self.prevalence).round() as usize;
        let mut y = Array1::<f64>::zeros(n);
        for &idx in indices.iter().take(n_cases) {
            y[idx] = 1.0;
        }

        let weights = Array1::ones(n);

        TestData {
            y,
            pgs,
            sex,
            pcs,
            weights,
        }
    }

    /// Build synthetic data for a Gaussian regression model (continuous outcome).
    pub fn build_identity(self) -> TestData {
        let mut rng = StdRng::seed_from_u64(self.seed);
        let n = self.n_samples;

        // Generate PGS (standardized)
        let pgs: Array1<f64> = Array1::from_iter((0..n).map(|_| rng.sample(StandardNormal)));

        // Generate PCs (standardized)
        let pcs = Array2::from_shape_fn((n, self.n_pcs), |_| rng.sample(StandardNormal));

        // Generate sex
        let sex: Array1<f64> = Array1::from_iter((0..n).map(|_| {
            if rng.random::<f64>() < self.male_fraction {
                1.0
            } else {
                0.0
            }
        }));

        // Generate continuous outcome based on signal type
        let signal: Array1<f64> = match self.signal_type {
            SignalType::Linear => pgs.mapv(|p| self.signal_strength * p),
            SignalType::Quadratic => pgs.mapv(|p| self.signal_strength * p + 0.3 * p * p),
            SignalType::Null => Array1::zeros(n),
        };

        // Add noise
        let noise: Array1<f64> = Array1::from_iter((0..n).map(|_| rng.sample(StandardNormal)));
        let y = &signal + &noise;

        let weights = Array1::ones(n);

        TestData {
            y,
            pgs,
            sex,
            pcs,
            weights,
        }
    }
}

/// Create a default `ModelConfig` for logistic regression tests.
pub fn default_model_config_logit() -> ModelConfig {
    default_model_config_with_link(LinkFunction::Logit)
}

/// Create a default `ModelConfig` for Gaussian regression tests.
pub fn default_model_config_identity() -> ModelConfig {
    default_model_config_with_link(LinkFunction::Identity)
}

/// Create a default `ModelConfig` with the specified link function.
pub fn default_model_config_with_link(link: LinkFunction) -> ModelConfig {
    ModelConfig {
        model_family: ModelFamily::Gam(link),
        penalty_order: 2,
        convergence_tolerance: 1e-6,
        max_iterations: 100,
        reml_convergence_tolerance: 1e-5,
        reml_max_iterations: 50,
        firth_bias_reduction: false,
        reml_parallel_threshold: 4,
        pgs_basis_config: BasisConfig {
            num_knots: 8,
            degree: 3,
        },
        pc_configs: vec![
            PrincipalComponentConfig {
                name: "PC1".to_string(),
                basis_config: BasisConfig {
                    num_knots: 6,
                    degree: 3,
                },
                range: (-3.0, 3.0),
            },
            PrincipalComponentConfig {
                name: "PC2".to_string(),
                basis_config: BasisConfig {
                    num_knots: 6,
                    degree: 3,
                },
                range: (-3.0, 3.0),
            },
        ],
        pgs_range: (-3.0, 3.0),
        interaction_penalty: InteractionPenaltyKind::Anisotropic,
        sum_to_zero_constraints: HashMap::new(),
        knot_vectors: HashMap::new(),
        range_transforms: HashMap::new(),
        interaction_centering_means: HashMap::new(),
        interaction_orth_alpha: HashMap::new(),
        pc_null_transforms: HashMap::new(),
        mcmc_enabled: false,       // Disable MCMC for faster tests
        calibrator_enabled: false, // Disable calibrator for faster tests
        survival: None,
    }
}

/// Create a minimal `ModelConfig` suitable for quick unit tests.
pub fn minimal_model_config_logit() -> ModelConfig {
    ModelConfig {
        model_family: ModelFamily::Gam(LinkFunction::Logit),
        penalty_order: 2,
        convergence_tolerance: 1e-4,
        max_iterations: 50,
        reml_convergence_tolerance: 1e-3,
        reml_max_iterations: 20,
        firth_bias_reduction: false,
        reml_parallel_threshold: 4,
        pgs_basis_config: BasisConfig {
            num_knots: 4,
            degree: 3,
        },
        pc_configs: vec![],
        pgs_range: (-3.0, 3.0),
        interaction_penalty: InteractionPenaltyKind::Anisotropic,
        sum_to_zero_constraints: HashMap::new(),
        knot_vectors: HashMap::new(),
        range_transforms: HashMap::new(),
        interaction_centering_means: HashMap::new(),
        interaction_orth_alpha: HashMap::new(),
        pc_null_transforms: HashMap::new(),
        mcmc_enabled: false,
        calibrator_enabled: false,
        survival: None,
    }
}

/// Create a dummy `TrainedModel` for testing prediction paths.
pub fn dummy_trained_model(config: ModelConfig) -> TrainedModel {
    TrainedModel {
        config,
        coefficients: MappedCoefficients {
            intercept: 0.0,
            main_effects: MainEffects {
                sex: 0.0,
                pgs: vec![0.0; 10],
                pcs: HashMap::new(),
            },
            interaction_effects: HashMap::new(),
        },
        lambdas: vec![1.0],
        hull: None,
        penalized_hessian: None,
        scale: None,
        calibrator: None,
        joint_link: None,
        survival: None,
        survival_companions: HashMap::new(),
        mcmc_samples: None,
        smoothing_correction: None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_synthetic_data_builder_logit() {
        let data = SyntheticDataBuilder::new(100)
            .with_pcs(2)
            .with_prevalence(0.1)
            .seed(12345)
            .build_logit();

        assert_eq!(data.n(), 100);
        assert_eq!(data.n_pcs(), 2);
        assert_eq!(data.y.len(), 100);
        assert_eq!(data.pgs.len(), 100);
        assert_eq!(data.sex.len(), 100);
        assert_eq!(data.pcs.nrows(), 100);
        assert_eq!(data.pcs.ncols(), 2);

        // Check that y is binary
        assert!(data.y.iter().all(|&v| v == 0.0 || v == 1.0));

        // Check approximate prevalence (within tolerance)
        let actual_prevalence = data.y.sum() / data.n() as f64;
        assert!((actual_prevalence - 0.1).abs() < 0.05);
    }

    #[test]
    fn test_synthetic_data_builder_identity() {
        let data = SyntheticDataBuilder::new(100)
            .with_pcs(3)
            .seed(12345)
            .build_identity();

        assert_eq!(data.n(), 100);
        assert_eq!(data.n_pcs(), 3);

        // Check that y is continuous (has variance)
        let mean = data.y.sum() / data.n() as f64;
        let variance: f64 =
            data.y.iter().map(|&v| (v - mean).powi(2)).sum::<f64>() / data.n() as f64;
        assert!(variance > 0.1);
    }

    #[test]
    fn test_default_model_configs() {
        let logit = default_model_config_logit();
        assert!(matches!(
            logit.model_family,
            ModelFamily::Gam(LinkFunction::Logit)
        ));

        let identity = default_model_config_identity();
        assert!(matches!(
            identity.model_family,
            ModelFamily::Gam(LinkFunction::Identity)
        ));
    }
}
