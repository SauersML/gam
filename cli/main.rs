#![deny(unused_variables)]
#![deny(dead_code)]
#![deny(unused_imports)]
#![deny(clippy::no_effect_underscore_binding)]

use clap::{Args, CommandFactory, Parser, Subcommand, ValueEnum};
use ndarray::{Array1, ArrayView1};
use std::collections::HashSet;
use std::path::PathBuf;
use std::process;

use gnomon::calibrate::data::{load_prediction_data, load_training_data};
use gnomon::calibrate::estimate::train_model;
#[cfg(feature = "survival-data")]
use gnomon::calibrate::estimate::train_survival_model;
use gnomon::calibrate::model::BasisConfig;
#[cfg(feature = "survival-data")]
use gnomon::calibrate::model::SurvivalModelConfig;
#[cfg(feature = "survival-data")]
use gnomon::calibrate::model::SurvivalPrediction;
#[cfg(feature = "survival-data")]
use gnomon::calibrate::model::SurvivalRiskType;
#[cfg(feature = "survival-data")]
use gnomon::calibrate::model::SurvivalTimeVaryingConfig;
use gnomon::calibrate::model::{
    InteractionPenaltyKind, LinkFunction, ModelConfig, ModelFamily, TrainedModel,
};
#[cfg(feature = "survival-data")]
use gnomon::calibrate::survival::SurvivalSpec;
#[cfg(feature = "survival-data")]
use gnomon::calibrate::survival_data::{
    SurvivalPredictionData, load_survival_prediction_data, load_survival_training_data,
};
use gnomon::map::main as map_cli;
use gnomon::map::{DEFAULT_LD_WINDOW, LdWindow};
use gnomon::terms::infer_sex_to_tsv;
use std::collections::HashMap;

#[derive(Clone, ValueEnum)]
pub enum ModelFamilyCli {
    Gam,
    Survival,
}

#[derive(Args)]
pub struct TrainArgs {
    #[arg(long, value_enum, default_value_t = ModelFamilyCli::Gam)]
    pub model_family: ModelFamilyCli,

    /// Path to training TSV file with phenotype,score,PC1,PC2,... columns
    pub training_data: String,

    /// Number of principal components to use from the data
    #[arg(long, value_name = "N")]
    pub num_pcs: usize,

    /// Number of internal knots for PGS spline basis
    #[arg(long, default_value = "10")]
    pub pgs_knots: usize,

    /// Polynomial degree for PGS spline basis
    #[arg(long, default_value = "3")]
    pub pgs_degree: usize,

    /// Number of internal knots for PC spline bases
    #[arg(long, default_value = "5")]
    pub pc_knots: usize,

    /// Polynomial degree for PC spline bases  
    #[arg(long, default_value = "2")]
    pub pc_degree: usize,

    /// Order of the difference penalty matrix
    #[arg(long, default_value = "2")]
    pub penalty_order: usize,

    /// Maximum number of P-IRLS iterations for the inner loop (per REML step)
    #[arg(long, default_value = "50")]
    pub max_iterations: usize,

    /// Convergence tolerance for the P-IRLS inner loop deviance change
    #[arg(long, default_value = "1e-7")]
    pub convergence_tolerance: f64,

    /// Maximum number of iterations for the outer REML/BFGS optimization loop
    #[arg(long, default_value = "100")]
    pub reml_max_iterations: u64,

    /// Convergence tolerance for the gradient norm in the outer REML/BFGS loop
    #[arg(long, default_value = "1e-3")]
    pub reml_convergence_tolerance: f64,

    /// Disable the optional post-process calibration layer (enabled by default)
    #[arg(long)]
    pub no_calibration: bool,

    /// Guard delta for the survival age transform
    #[arg(long, default_value = "0.1")]
    pub survival_guard_delta: f64,

    /// Number of internal knots for the survival baseline spline
    #[arg(long, default_value = "6")]
    pub survival_baseline_knots: usize,

    /// Degree for the survival baseline spline
    #[arg(long, default_value = "3")]
    pub survival_baseline_degree: usize,

    /// Grid size for the survival monotonicity penalty
    #[arg(long, default_value = "64")]
    pub survival_monotonic_grid: usize,

    /// Derivative guard threshold used inside the survival likelihood
    #[arg(long, default_value = "1e-8")]
    pub survival_derivative_guard: f64,

    /// Use expected information instead of observed Hessian when fitting survival models
    #[arg(long)]
    pub survival_expected_information: bool,

    /// Enable the optional time-varying PGS × age interaction
    #[arg(long)]
    pub survival_enable_time_varying: bool,

    /// Number of internal knots for the time-varying PGS spline
    #[arg(long, default_value = "5")]
    pub survival_time_varying_pgs_knots: usize,

    /// Degree for the time-varying PGS spline
    #[arg(long, default_value = "3")]
    pub survival_time_varying_pgs_degree: usize,

    /// Difference-penalty order for the time-varying PGS spline
    #[arg(long, default_value = "2")]
    pub survival_time_varying_pgs_penalty_order: usize,
}

#[derive(Args)]
pub struct InferArgs {
    /// Path to test TSV file with score,PC1,PC2,... columns (no phenotype needed)
    pub test_data: String,

    /// Path to trained model file (.toml)
    #[arg(long)]
    pub model: String,

    /// Disable applying the post-process calibration layer (enabled by default)
    #[arg(long)]
    pub no_calibration: bool,
}

#[derive(Args)]
pub struct TermsArgs {
    /// Path to genotype dataset (PLINK .bed/.bim/.fam prefix or VCF/BCF file)
    #[arg(value_name = "GENOTYPE_PATH")]
    pub genotype_path: PathBuf,

    /// Run sex inference on the provided genotype dataset
    #[arg(long)]
    pub sex: bool,
}

pub fn train(args: TrainArgs) -> Result<(), Box<dyn std::error::Error>> {
    gnomon::calibrate::model::set_calibrator_enabled(!args.no_calibration);
    if args.no_calibration {
        println!("Post-process calibration disabled via --no-calibration flag.");
    } else {
        println!("Post-process calibration enabled (default).");
    }

    match args.model_family {
        ModelFamilyCli::Gam => {
            println!("Loading training data from: {}", args.training_data);
            let data = load_training_data(&args.training_data, args.num_pcs)?;
            println!(
                "Loaded {} samples with {} PCs",
                data.y.len(),
                data.pcs.ncols()
            );

            let link_function = detect_link_function(&data.y);
            println!("Auto-detected link function: {link_function:?}");

            let pgs_range = calculate_range(data.p.view());
            let pc_ranges: Vec<(f64, f64)> = (0..data.pcs.ncols())
                .map(|i| calculate_range(data.pcs.column(i)))
                .collect();

            println!("PGS range: ({:.3}, {:.3})", pgs_range.0, pgs_range.1);
            println!(
                "PC ranges: {}",
                pc_ranges
                    .iter()
                    .enumerate()
                    .map(|(i, &(min, max))| format!("PC{}: ({:.3}, {:.3})", i + 1, min, max))
                    .collect::<Vec<_>>()
                    .join(", ")
            );

            let pgs_basis_config = BasisConfig {
                num_knots: args.pgs_knots,
                degree: args.pgs_degree,
            };

            let pc_configs = (0..args.num_pcs)
                .map(|i| gnomon::calibrate::model::PrincipalComponentConfig {
                    name: format!("PC{}", i + 1),
                    basis_config: BasisConfig {
                        num_knots: args.pc_knots,
                        degree: args.pc_degree,
                    },
                    range: pc_ranges[i],
                })
                .collect();

            println!("Training model with REML estimation of smoothing parameters");
            let config = ModelConfig {
                model_family: ModelFamily::Gam(link_function),
                penalty_order: args.penalty_order,
                convergence_tolerance: args.convergence_tolerance,
                max_iterations: args.max_iterations,
                reml_convergence_tolerance: args.reml_convergence_tolerance,
                reml_max_iterations: args.reml_max_iterations,
                firth_bias_reduction: false,
                reml_parallel_threshold: gnomon::calibrate::model::default_reml_parallel_threshold(
                ),
                pgs_basis_config,
                pc_configs,
                pgs_range,
                interaction_penalty: InteractionPenaltyKind::Anisotropic,
                sum_to_zero_constraints: HashMap::new(),
                knot_vectors: HashMap::new(),
                range_transforms: HashMap::new(),
                interaction_centering_means: HashMap::new(),
                interaction_orth_alpha: HashMap::new(),
                pc_null_transforms: HashMap::new(),
                survival: None,
            };

            println!("Training final model...");
            let trained_model = train_model(&data, &config)?;
            trained_model.save("model.toml")?;
            println!("Model saved to: model.toml");
        }
        #[cfg(feature = "survival-data")]
        ModelFamilyCli::Survival => {
            println!(
                "Loading survival training data from: {}",
                args.training_data
            );
            let bundle = load_survival_training_data(
                &args.training_data,
                args.num_pcs,
                args.survival_guard_delta,
            )?;
            println!(
                "Loaded {} samples with {} PCs",
                bundle.data.age_entry.len(),
                bundle.data.pcs.ncols()
            );

            let mut spec = SurvivalSpec::default();
            spec.derivative_guard = args.survival_derivative_guard;
            spec.use_expected_information = args.survival_expected_information;

            let time_varying = if args.survival_enable_time_varying {
                Some(SurvivalTimeVaryingConfig {
                    label: Some("pgs_by_age".to_string()),
                    pgs_basis: BasisConfig {
                        num_knots: args.survival_time_varying_pgs_knots,
                        degree: args.survival_time_varying_pgs_degree,
                    },
                    pgs_penalty_order: args.survival_time_varying_pgs_penalty_order,
                    lambda_age: 0.0,
                    lambda_pgs: 0.0,
                    lambda_null: 0.0,
                })
            } else {
                None
            };

            if let Some(settings) = &time_varying {
                println!(
                    "Enabling time-varying PGS × age interaction (knots={}, degree={}, penalty order={})",
                    settings.pgs_basis.num_knots,
                    settings.pgs_basis.degree,
                    settings.pgs_penalty_order
                );
            }

            let survival_config = SurvivalModelConfig {
                baseline_basis: BasisConfig {
                    num_knots: args.survival_baseline_knots,
                    degree: args.survival_baseline_degree,
                },
                guard_delta: args.survival_guard_delta,
                monotonic_grid_size: args.survival_monotonic_grid,
                time_varying,
                model_competing_risk: false,
            };

            let config = ModelConfig {
                model_family: ModelFamily::Survival(spec),
                penalty_order: args.penalty_order,
                convergence_tolerance: args.convergence_tolerance,
                max_iterations: args.max_iterations,
                reml_convergence_tolerance: args.reml_convergence_tolerance,
                reml_max_iterations: args.reml_max_iterations,
                firth_bias_reduction: false,
                reml_parallel_threshold: gnomon::calibrate::model::default_reml_parallel_threshold(
                ),
                pgs_basis_config: BasisConfig {
                    num_knots: 0,
                    degree: 0,
                },
                pc_configs: Vec::new(),
                pgs_range: (0.0, 0.0),
                interaction_penalty: InteractionPenaltyKind::Anisotropic,
                sum_to_zero_constraints: HashMap::new(),
                knot_vectors: HashMap::new(),
                range_transforms: HashMap::new(),
                interaction_centering_means: HashMap::new(),
                interaction_orth_alpha: HashMap::new(),
                pc_null_transforms: HashMap::new(),
                survival: Some(survival_config),
            };

            println!("Training survival model...");
            let trained_model = train_survival_model(&bundle, &config)?;
            trained_model.save("model.toml")?;
            println!("Model saved to: model.toml");
        }
        #[cfg(not(feature = "survival-data"))]
        ModelFamilyCli::Survival => {
            eprintln!("Error: Survival model support requires the 'survival-data' feature");
            std::process::exit(1);
        }
    }

    Ok(())
}

pub fn infer(args: InferArgs) -> Result<(), Box<dyn std::error::Error>> {
    println!("Loading model from: {}", args.model);

    // Load trained model
    let model = TrainedModel::load(&args.model)?;
    let num_pcs = model.config.pc_configs.len();
    println!("Model expects {num_pcs} PCs");

    match &model.config.model_family {
        ModelFamily::Gam(link_function) => {
            // Load test data
            println!("Loading test data from: {}", args.test_data);
            let data = load_prediction_data(&args.test_data, num_pcs)?;
            println!("Loaded {} samples for prediction", data.p.len());

            // Make detailed predictions
            println!("Generating predictions with diagnostics...");
            let (eta, mean, signed_dist, se_eta_opt) =
                model.predict_detailed(data.p.view(), data.sex.view(), data.pcs.view())?;

            // Check if calibrator is available
            let calibrated_mean_opt = if args.no_calibration {
                println!("Skipping calibration via --no-calibration flag.");
                None
            } else if model.calibrator.is_some() {
                println!("Calibrator detected. Generating calibrated predictions.");
                // Get calibrated predictions but don't error if calibrator is missing
                match model.predict_calibrated(data.p.view(), data.sex.view(), data.pcs.view()) {
                    Ok(calibrated) => Some(calibrated),
                    Err(_) => None,
                }
            } else {
                None
            };

            // Save predictions with required columns to hardcoded output path
            let output_path = "predictions.tsv";
            save_predictions_detailed(
                &data.sample_ids,
                &signed_dist,
                &eta,
                &mean,
                se_eta_opt.as_ref(),
                *link_function,
                calibrated_mean_opt.as_ref(),
                output_path,
            )?;
            println!("Predictions saved to: {output_path}");
        }
        ModelFamily::Survival(_) => {
            #[cfg(not(feature = "survival-data"))]
            {
                println!(
                    "Loaded survival model, but CLI was built without 'survival-data' feature. \
                     Rebuild with '--features survival-data' to enable survival inference."
                );
                return Ok(());
            }

            #[cfg(feature = "survival-data")]
            {
                println!("Loading survival prediction data from: {}", args.test_data);
                let data = load_survival_prediction_data(&args.test_data, num_pcs)?;
                println!(
                    "Loaded {} samples for survival prediction",
                    data.age_entry.len()
                );

                println!("Generating survival predictions with diagnostics...");
                let prediction = model.predict_survival(
                    data.age_entry.view(),
                    data.age_exit.view(),
                    data.pgs.view(),
                    data.sex.view(),
                    data.pcs.view(),
                    SurvivalRiskType::Net,
                    Some(&model.survival_companions),
                )?;

                let calibrated_risk = if args.no_calibration {
                    println!("Skipping calibration via --no-calibration flag.");
                    None
                } else if model
                    .survival
                    .as_ref()
                    .and_then(|artifacts| artifacts.calibrator.as_ref())
                    .is_some()
                {
                    println!("Calibrator detected. Generating calibrated survival predictions.");
                    match model.predict_survival_calibrated(
                        data.age_entry.view(),
                        data.age_exit.view(),
                        data.pgs.view(),
                        data.sex.view(),
                        data.pcs.view(),
                        Some(&model.survival_companions),
                    ) {
                        Ok(calibrated) => Some(calibrated),
                        Err(_) => None,
                    }
                } else {
                    None
                };

                let output_path = "predictions.tsv";
                save_survival_predictions(
                    output_path,
                    &data,
                    &prediction,
                    calibrated_risk.as_ref(),
                )?;
                println!("Predictions saved to: {output_path}");
            }
        }
    }

    Ok(())
}

fn detect_link_function(phenotype: &Array1<f64>) -> LinkFunction {
    let unique_values: HashSet<_> = phenotype.iter().map(|&x| x as i64).collect();

    if unique_values.len() == 2 {
        // Binary case/control data
        LinkFunction::Logit
    } else {
        // Continuous quantitative trait
        LinkFunction::Identity
    }
}

fn calculate_range(data: ArrayView1<f64>) -> (f64, f64) {
    let min_val = data.iter().fold(f64::INFINITY, |a, &b| a.min(b));
    let max_val = data.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
    (min_val, max_val)
}

fn save_predictions_detailed(
    sample_ids: &Vec<String>,
    signed_distance: &Array1<f64>,
    eta: &Array1<f64>,
    mean: &Array1<f64>,
    se_eta_opt: Option<&Array1<f64>>,
    link: LinkFunction,
    calibrated_mean_opt: Option<&Array1<f64>>,
    output_path: &str,
) -> Result<(), std::io::Error> {
    use std::io::Write;

    let mut file = std::fs::File::create(output_path)?;
    match link {
        LinkFunction::Logit => {
            // Header for binary target - include uncalibrated_prediction column when calibration is enabled
            if calibrated_mean_opt.is_some() {
                writeln!(
                    file,
                    "sample_id\thull_signed_distance\tlog_odds\tstandard_error_log_odds\tuncalibrated_prediction\tprediction\tprobability_lower_95\tprobability_upper_95"
                )?
            } else {
                writeln!(
                    file,
                    "sample_id\thull_signed_distance\tlog_odds\tstandard_error_log_odds\tprediction\tprobability_lower_95\tprobability_upper_95"
                )?
            };

            for i in 0..eta.len() {
                let id = &sample_ids[i];
                let sd = signed_distance[i];
                let log_odds = eta[i];
                let uncalibrated_p = mean[i];
                // If calibrated predictions are available, use them, otherwise use uncalibrated
                let p = calibrated_mean_opt.map_or(uncalibrated_p, |cal| cal[i]);

                let (se_str, lo_str, hi_str) = if let Some(se_eta) = se_eta_opt {
                    let se = se_eta[i];
                    let lo_eta = log_odds - 1.959964 * se;
                    let hi_eta = log_odds + 1.959964 * se;
                    let lo_p = 1.0 / (1.0 + (-lo_eta).exp());
                    let hi_p = 1.0 / (1.0 + (-hi_eta).exp());
                    (
                        se.to_string(),
                        lo_p.max(0.0).min(1.0).to_string(),
                        hi_p.max(0.0).min(1.0).to_string(),
                    )
                } else {
                    ("NA".to_string(), "NA".to_string(), "NA".to_string())
                };

                // Include uncalibrated prediction column when calibration is enabled
                if calibrated_mean_opt.is_some() {
                    writeln!(
                        file,
                        "{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}",
                        id, sd, log_odds, se_str, uncalibrated_p, p, lo_str, hi_str
                    )?
                } else {
                    writeln!(
                        file,
                        "{}\t{}\t{}\t{}\t{}\t{}\t{}",
                        id, sd, log_odds, se_str, p, lo_str, hi_str
                    )?
                };
            }
        }
        LinkFunction::Identity => {
            // Header for continuous target - include uncalibrated_prediction column when calibration is enabled
            if calibrated_mean_opt.is_some() {
                writeln!(
                    file,
                    "sample_id\thull_signed_distance\tuncalibrated_prediction\tprediction\tstandard_error_mean\tmean_lower_95\tmean_upper_95"
                )?
            } else {
                writeln!(
                    file,
                    "sample_id\thull_signed_distance\tprediction\tstandard_error_mean\tmean_lower_95\tmean_upper_95"
                )?
            };

            for i in 0..eta.len() {
                let id = &sample_ids[i];
                let sd = signed_distance[i];
                let uncalibrated_pred = mean[i];
                // If calibrated predictions are available, use them, otherwise use uncalibrated
                let pred = calibrated_mean_opt.map_or(uncalibrated_pred, |cal| cal[i]);

                let (se_str, lo_str, hi_str) = if let Some(se_eta) = se_eta_opt {
                    let se = se_eta[i];
                    let lo = pred - 1.959964 * se;
                    let hi = pred + 1.959964 * se;
                    (se.to_string(), lo.to_string(), hi.to_string())
                } else {
                    ("NA".to_string(), "NA".to_string(), "NA".to_string())
                };

                // Include uncalibrated prediction column when calibration is enabled
                if calibrated_mean_opt.is_some() {
                    writeln!(
                        file,
                        "{}\t{}\t{}\t{}\t{}\t{}\t{}",
                        id, sd, uncalibrated_pred, pred, se_str, lo_str, hi_str
                    )?
                } else {
                    writeln!(
                        file,
                        "{}\t{}\t{}\t{}\t{}\t{}",
                        id, sd, pred, se_str, lo_str, hi_str
                    )?
                };
            }
        }
    }
    Ok(())
}

#[cfg(feature = "survival-data")]
fn save_survival_predictions(
    output_path: &str,
    data: &SurvivalPredictionData,
    prediction: &SurvivalPrediction,
    calibrated_risk: Option<&Array1<f64>>,
) -> Result<(), std::io::Error> {
    use std::io::Write;

    let mut file = std::fs::File::create(output_path)?;
    if calibrated_risk.is_some() {
        writeln!(
            file,
            "sample_id\tage_entry\tage_exit\tcumulative_hazard_entry\tcumulative_hazard_exit\tcumulative_incidence_entry\tcumulative_incidence_exit\tconditional_risk\tlogit_risk\tlogit_risk_standard_error\tcalibrated_risk"
        )?;
    } else {
        writeln!(
            file,
            "sample_id\tage_entry\tage_exit\tcumulative_hazard_entry\tcumulative_hazard_exit\tcumulative_incidence_entry\tcumulative_incidence_exit\tconditional_risk\tlogit_risk\tlogit_risk_standard_error"
        )?;
    }

    let count = prediction.conditional_risk.len();
    for idx in 0..count {
        let sample_id = (idx + 1).to_string();
        let hazard_entry = prediction.cumulative_hazard_entry[idx];
        let hazard_exit = prediction.cumulative_hazard_exit[idx];
        let incidence_entry = prediction.cumulative_incidence_entry[idx];
        let incidence_exit = prediction.cumulative_incidence_exit[idx];
        let conditional_risk = prediction.conditional_risk[idx];
        let logit_risk = prediction.logit_risk[idx];
        let se = prediction
            .logit_risk_se
            .as_ref()
            .map(|array| array[idx].to_string())
            .unwrap_or_else(|| "NA".to_string());

        if let Some(calibrated) = calibrated_risk {
            writeln!(
                file,
                "{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}",
                sample_id,
                data.age_entry[idx],
                data.age_exit[idx],
                hazard_entry,
                hazard_exit,
                incidence_entry,
                incidence_exit,
                conditional_risk,
                logit_risk,
                se,
                calibrated[idx]
            )?;
        } else {
            writeln!(
                file,
                "{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}",
                sample_id,
                data.age_entry[idx],
                data.age_exit[idx],
                hazard_entry,
                hazard_exit,
                incidence_entry,
                incidence_exit,
                conditional_risk,
                logit_risk,
                se
            )?;
        }
    }

    Ok(())
}

// Import score functionality
#[path = "../score/main.rs"]
mod score_main;

#[derive(Parser)]
#[command(
    name = "gnomon",
    about = "High-performance polygenic score calculation and calibration toolkit",
    long_about = "A comprehensive toolkit for calculating polygenic scores from genotype data \
                 and training GAM models for score calibration."
)]
struct Cli {
    #[command(subcommand)]
    command: Option<Commands>,
}

#[derive(Subcommand)]
enum Commands {
    /// Calculate polygenic scores from genotype data
    #[command(about = "Calculate raw polygenic scores")]
    Score {
        /// Path to score file or directory containing score files
        #[arg(value_name = "SCORE_PATH")]
        score: PathBuf,

        /// Path to PLINK .bed file or directory containing .bed files
        #[arg(value_name = "GENOTYPE_PATH")]
        input_path: PathBuf,

        /// Path to file containing list of individual IDs to include (optional)
        #[arg(long)]
        keep: Option<PathBuf>,
    },

    /// Fit an HWE PCA model from the provided genotype dataset
    #[command(about = "Fit an HWE PCA model")]
    Fit {
        /// Path to PLINK .bed file or directory containing .bed files
        #[arg(value_name = "GENOTYPE_PATH")]
        genotype_path: PathBuf,

        /// Optional variant list limiting SNVs used for PCA fitting
        #[arg(long, value_name = "PATH")]
        list: Option<PathBuf>,

        /// Number of principal components to retain when fitting the HWE PCA model
        #[arg(long, value_name = "N")]
        components: usize,

        /// Enable LD normalization when fitting the PCA model
        #[arg(long)]
        ld: bool,

        /// LD window expressed as the number of sites (must be odd)
        #[arg(
            long = "sites_window",
            value_name = "SITES",
            requires = "ld",
            conflicts_with = "bp_window"
        )]
        sites_window: Option<usize>,

        /// LD window expressed as the total span in base pairs
        #[arg(
            long = "bp_window",
            value_name = "BP",
            requires = "ld",
            conflicts_with = "sites_window"
        )]
        bp_window: Option<u64>,
    },

    /// Project samples using an existing HWE PCA model
    #[command(about = "Project samples using an existing HWE PCA model")]
    Project {
        /// Path to PLINK .bed file or directory containing .bed files
        #[arg(value_name = "GENOTYPE_PATH")]
        genotype_path: PathBuf,
    },

    /// Infer sample-level terms from genotype data
    #[command(about = "Infer sample metadata terms (outputs: sex.tsv)")]
    Terms(TermsArgs),

    /// Train a GAM calibration model from training data
    #[command(about = "Train GAM calibration model (outputs: model.toml)")]
    Train(TrainArgs),

    /// Apply trained calibration model to new data
    #[command(about = "Apply calibration model to new data (outputs: predictions.tsv)")]
    Infer(InferArgs),
}

fn main() {
    let cli = Cli::parse();
    let Cli { command } = cli;

    // Ensure each invocation starts with calibration enabled unless explicitly disabled.
    gnomon::calibrate::model::reset_calibrator_flag();

    let result = match command {
        Some(Commands::Score {
            score,
            input_path,
            keep,
        }) => run_score(input_path, score, keep),
        Some(Commands::Fit {
            genotype_path,
            list,
            components,
            ld,
            sites_window,
            bp_window,
        }) => run_map_fit(genotype_path, list, components, ld, sites_window, bp_window),
        Some(Commands::Project { genotype_path }) => run_map_project(genotype_path),
        Some(Commands::Terms(args)) => run_terms(args),
        Some(Commands::Train(args)) => train(args),
        Some(Commands::Infer(args)) => infer(args),
        None => {
            Cli::command().print_help().expect("print help");
            println!();
            Ok(())
        }
    };

    if let Err(e) = result {
        eprintln!("Error: {e}");
        process::exit(1);
    }
}

fn run_map_fit(
    genotype_path: PathBuf,
    list: Option<PathBuf>,
    components: usize,
    ld: bool,
    sites_window: Option<usize>,
    bp_window: Option<u64>,
) -> Result<(), Box<dyn std::error::Error>> {
    let ld_window = if ld {
        if let Some(bp) = bp_window {
            Some(LdWindow::BasePairs(bp))
        } else {
            let window = sites_window.unwrap_or(DEFAULT_LD_WINDOW);
            if window == 0 {
                return Err(Box::new(std::io::Error::new(
                    std::io::ErrorKind::InvalidInput,
                    "--sites_window must be at least 1",
                )));
            }
            if window % 2 == 0 {
                return Err(Box::new(std::io::Error::new(
                    std::io::ErrorKind::InvalidInput,
                    "--sites_window must be an odd number",
                )));
            }
            Some(LdWindow::Sites(window))
        }
    } else {
        None
    };

    map_cli::run(map_cli::MapCommand::Fit {
        genotype_path,
        variant_list: list,
        components,
        ld: ld_window,
    })
    .map_err(|err| Box::new(err) as Box<dyn std::error::Error>)
}

fn run_map_project(genotype_path: PathBuf) -> Result<(), Box<dyn std::error::Error>> {
    map_cli::run(map_cli::MapCommand::Project { genotype_path })
        .map_err(|err| Box::new(err) as Box<dyn std::error::Error>)
}

fn run_terms(args: TermsArgs) -> Result<(), Box<dyn std::error::Error>> {
    if !args.sex {
        return Err(Box::new(std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            "No term inference selected. Use --sex to run sex inference.",
        )));
    }

    let output_path = infer_sex_to_tsv(&args.genotype_path, None)
        .map_err(|err| Box::new(err) as Box<dyn std::error::Error>)?;
    println!("Sex inference results written to {}", output_path.display());
    Ok(())
}

fn run_score(
    input_path: PathBuf,
    score: PathBuf,
    keep: Option<PathBuf>,
) -> Result<(), Box<dyn std::error::Error>> {
    // Call the score calculation logic directly
    score_main::run_gnomon_with_args(input_path, score, keep)
        .map_err(|e| e as Box<dyn std::error::Error>)
}
