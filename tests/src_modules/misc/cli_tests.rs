use super::{
    BlockRole, BoundedCoefficientPriorSpec, CliFirthValidation, DataSchema,
    FAMILY_GAUSSIAN_LOCATION_SCALE, FamilyArg, FittedFamily, LikelihoodSpec, LinkChoice, LinkMode,
    ResponseFamily, SavedFitSummary, SavedModel, SurvivalArgs, SurvivalBaselineTarget,
    SurvivalLikelihoodMode, SurvivalTimeBasisConfig, build_survival_time_basis, classify_cli_error,
    collect_smooth_structure_warnings, compact_fit_result_for_batch,
    compact_saved_multiblock_fit_result, compute_probit_q0_from_eta, core_saved_fit_result,
    covariance_from_model, effectivelinkwiggle_formulaspec, family_arg_canonical_name,
    load_dataset_projected, parse_formula, parse_link_choice, parse_matching_auxiliary_formula,
    parse_surv_response, parse_survival_time_basis_config, predict_gam,
    prepend_id_column_to_prediction_csv, required_columns_for_fit, required_columns_for_formula,
    route_marginal_slope_deviation_blocks, summarizewiggle_domain,
    validate_cli_firth_configuration, validate_fit_args_preflight,
    write_estimand_explicit_prediction_csv, write_prediction_csv,
    write_survival_binary_prediction_csv, write_survival_prediction_csv,
};
use super::{
    Cli, Command, FitArgs, InferenceCovarianceMode, PredictArgs, PredictModeArg, SampleArgs,
    run_fit, run_predict, run_sample, write_model_json,
};
use crate::config_resolve::{
    SurvivalInverseLinkInput, parse_survival_inverse_link as parse_config_survival_inverse_link,
};
use clap::Parser;

/// Delete a test's temporary output file, reporting rather than swallowing a
/// failure. Cleanup runs after the assertions, so a failure cannot invalidate
/// the test that just passed — but a leaked temp file can collide with a later
/// run's fixture, and an error nobody prints makes that impossible to diagnose.
fn remove_temp_file(path: &std::path::Path) {
    if let Err(err) = fs::remove_file(path) {
        eprintln!(
            "cli test cleanup: could not remove {}: {err}",
            path.display()
        );
    }
}
use csv::StringRecord;
use gam::MatrixMaterializationError;
use gam::basis::{BSplineBasisSpec, BSplineBoundaryConditions, BSplineIdentifiability, BSplineKnotSpec, BasisOptions, CenterStrategy, DuchonBasisSpec, DuchonNullspaceOrder, DuchonOperatorPenaltySpec, MaternBasisSpec, MaternNu, OneDimensionalBoundary, SpatialIdentifiability, ThinPlateBasisSpec};
use gam::estimate::{
    FitGeometry, FitInference, FittedBlock, FittedLinkState, UnifiedFitResultParts,
};
use gam::families::bms::LatentMeasureKind;
use gam::families::cubic_cell_kernel as exact_kernel;
use gam::families::survival::construction::build_survival_baseline_offsets;
use gam::families::survival::construction::build_survival_timewiggle_from_baseline;
use gam::families::survival::construction::parse_survival_baseline_config;
use gam::families::survival::construction::{SurvivalBaselineConfig, evaluate_survival_baseline};
use gam::families::survival::location_scale::{
    ResidualDistribution, SurvivalLocationScaleTimeParameterization,
    project_onto_linear_constraints,
};
use gam::families::survival::lognormal_kernel::FrailtyScale;
use gam::families::wiggle::monotone_wiggle_basis_with_derivative_order;
use gam::generative::sampleobservation_seeded_replicates;
use gam::inference::data::{
    EncodedDataset as Dataset, UnseenCategoryPolicy, encode_recordswith_schema,
};
use gam::inference::formula_dsl::{ParsedTerm, parse_linkwiggle_formulaspec};
use gam::inference::model::{
    ColumnKindTag, FittedModelPayload, MODEL_PAYLOAD_VERSION, ModelKind, SavedCompiledFlexBlock,
    SavedLatentZNormalization, SavedSurvivalLocationScaleStructure, SchemaColumn,
};
use gam::inference::model_payload_builders::{
    BernoulliMarginalSlopeInputs, SavedModelSourceMetadata,
    assemble_bernoulli_marginal_slope_payload,
};
use gam::matrix::{DenseDesignMatrix, DenseDesignOperator, DesignMatrix, LinearOperator};
use gam::mixture_link::mixture_inverse_link_jet;
use gam::probability::normal_cdf;
use gam::smooth::{
    LinearCoefficientGeometry, LinearTermSpec, SmoothBasisSpec, SmoothTermSpec, TermCollectionSpec,
    build_term_collection_design,
};
use gam::solver::gauge::Gauge;
use gam::term_builder::{heuristic_knots_for_column, parse_duchon_order, unique_count_column};
use gam::types::{
    InverseLink, LikelihoodScaleMetadata, LinkComponent, LinkFunction, LogLikelihoodNormalization,
    ResponseColumnKind, StandardLink, WigglePenaltyConfig,
};
use gam_predict::{
    FittedModelPredictExt, PredictableModel, SavedGenerativeInput, generative_spec_for_saved_model,
};
use ndarray::{Array1, Array2, ArrayView1, ArrayViewMut2, array, s};
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand_distr::{Distribution, StandardNormal};
use std::collections::{BTreeMap, HashMap};
use std::fs;
use std::ops::Range;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};
use tempfile::tempdir;

fn resolve_family(
    arg: FamilyArg,
    negative_binomial_theta: Option<f64>,
    link_choice: Option<LinkChoice>,
    y: ArrayView1<'_, f64>,
    y_kind: ResponseColumnKind,
    response_name: &str,
) -> Result<LikelihoodSpec, String> {
    if negative_binomial_theta.is_some() && !matches!(arg, FamilyArg::NegativeBinomial) {
        return Err("--negative-binomial-theta requires --family negative-binomial".to_string());
    }
    gam::families::fit_orchestration::resolve_family(
        family_arg_canonical_name(arg),
        negative_binomial_theta,
        link_choice.as_ref(),
        y,
        y_kind,
        response_name,
    )
}

fn test_saved_linkwiggle_design(
    q0: &Array1<f64>,
    model: &SavedModel,
) -> Result<Option<Array2<f64>>, String> {
    test_saved_linkwiggle_basis(q0, model, BasisOptions::value())
}

fn test_saved_linkwiggle_basis(
    q0: &Array1<f64>,
    model: &SavedModel,
    basis_options: BasisOptions,
) -> Result<Option<Array2<f64>>, String> {
    match model.saved_link_wiggle()? {
        None => Ok(None),
        Some(runtime) => {
            runtime.derivative_q0(q0)?;
            runtime
                .constrained_basis(q0, basis_options)
                .map(Some)
                .map_err(String::from)
        }
    }
}

fn test_saved_linkwiggle_derivative_q0(
    q0: &Array1<f64>,
    model: &SavedModel,
) -> Result<Array1<f64>, String> {
    match model.saved_link_wiggle()? {
        Some(runtime) => runtime.derivative_q0(q0).map_err(String::from),
        None => Ok(Array1::ones(q0.len())),
    }
}

fn empty_termspec() -> TermCollectionSpec {
    TermCollectionSpec {
        linear_terms: vec![],
        random_effect_terms: vec![],
        smooth_terms: vec![],
    }
}

fn bounded_cli_schema() -> DataSchema {
    DataSchema {
        columns: vec![
            SchemaColumn {
                name: "x".to_string(),
                kind: ColumnKindTag::Continuous,
                levels: vec![],
            },
            SchemaColumn {
                name: "y".to_string(),
                kind: ColumnKindTag::Continuous,
                levels: vec![],
            },
        ],
    }
}

fn bounded_cli_dataset() -> Dataset {
    Dataset {
        headers: vec!["x".to_string(), "y".to_string()],
        values: array![[0.0, 0.0], [0.5, 1.0], [1.0, 1.0], [1.5, 2.0]],
        schema: bounded_cli_schema(),
        column_kinds: vec![ColumnKindTag::Continuous, ColumnKindTag::Continuous],
    }
}

fn bounded_cli_termspec() -> TermCollectionSpec {
    let parsed = parse_formula("y ~ bounded(x, min=-2, max=2) + link(type=logit)")
        .unwrap_or_else(|e| panic!("{} failed: {:?}", "formula", e));
    let ds = bounded_cli_dataset();
    let col_map = HashMap::from([("x".to_string(), 0usize), ("y".to_string(), 1usize)]);
    let mut inference_notes = Vec::<String>::new();
    super::build_termspec(
        &parsed.terms,
        &ds,
        &col_map,
        &mut inference_notes,
        &gam::ResourcePolicy::default_library(),
    )
    .unwrap_or_else(|e| panic!("{} failed: {:?}", "bounded term spec", e))
}

fn saved_fit_summary_fixture() -> SavedFitSummary {
    SavedFitSummary {
        training_sample_size: 1,
        likelihood_family: Some(LikelihoodSpec::new(
            ResponseFamily::Gaussian,
            InverseLink::Standard(StandardLink::Identity),
        )),
        likelihood_scale: LikelihoodScaleMetadata::ProfiledGaussian,
        log_likelihood_normalization: LogLikelihoodNormalization::Full,
        log_likelihood: 0.0,
        iterations: 0,
        finalgrad_norm: 0.0,
        pirls_status: gam::pirls::PirlsStatus::Converged,
        deviance: 0.0,
        stable_penalty_term: 0.0,
        max_abs_eta: 0.0,
        reml_score: Some(0.0),
        // A CONVERGED summary, which is what every user of this fixture means
        // by it. SPEC 20 -- "a fit object must only ever come from a converged
        // optimization" -- is enforced by the sealed constructor, and
        // `core_saved_fit_result_preserves_summary_metrics` says so in as many
        // words: "a non-converged saved summary is unrepresentable now ... pin
        // that a CONVERGED summary mints".
        //
        // With `criterion_certificate: None` this fixture was not that. Any
        // test that set `iterations > 0` on it described a fit whose outer loop
        // ran and produced no stationarity proof, and assembly refused it with
        // "Fit assembly rejected a non-converged optimization state: inner
        // status Converged, outer status outer iterations ran without an
        // analytic stationarity certificate". The gate is right; the fixture
        // was contradicting the thing it exists to represent.
        criterion_certificate: Some(gam::estimate::OuterCriterionCertificate {
            stationarity: gam::estimate::OuterStationarityCertificate::AnalyticGradient {
                grad_norm: 1e-8,
                projected_grad_norm: 1e-8,
                bound: 1e-4,
                rung: gam::model_types::CertifiedRung {
                    label: "solver-band".to_string(),
                    derived_standard: false,
                },
            },
            curvature: gam::model_types::CurvatureEvidence::Measured { psd: true },
            lambdas_railed: Vec::new(),
            railed_facts: Vec::new(),
            curvature_floor: None,
        }),
    }
}

#[test]
fn core_saved_fit_result_preserves_summary_metrics() {
    // A non-converged saved summary is unrepresentable now: the sealed
    // constructor refuses to mint a UnifiedFitResult from a stalled fit
    // (SPEC 20). Pin that a converged summary mints and its optimizer
    // metrics survive the round-trip.
    let mut summary = saved_fit_summary_fixture();
    summary.iterations = 60;
    summary.finalgrad_norm = 42.0;

    // The smoothing vector has to be NON-EMPTY for an outer iteration count to
    // mean anything. With zero smoothing coordinates there is no equation for
    // the outer loop to solve, so `UnifiedFitResult` canonicalizes the pair to
    // the exact `Fixed` representation -- `(0, FitOuterConvergenceEvidence::
    // Fixed)` -- on the documented grounds that dimensionality, not an
    // implementation counter, is the semantic authority. Asserting that 60
    // survives `Array1::zeros(0)` asked the constructor to contradict itself,
    // and it cannot exercise "metrics survive the round-trip" on a path where
    // the metric is definitionally absent.
    let fit = core_saved_fit_result(array![1.0], array![1.0], 1.0, None, None, summary);

    assert_eq!(fit.outer_iterations, 60);
    assert_eq!(fit.outer_gradient_norm, Some(42.0));
    assert_eq!(
        fit.convergence_evidence().inner_status(),
        gam::pirls::PirlsStatus::Converged
    );
}

mod saved_survival_marginal_slope_test_support {
    use super::exact_kernel;
    use super::{Array1, SavedCompiledFlexBlock};
    use gam::families::marginal_slope_shared::{probit_frailty_scale, scale_coeff4};
    use gam::probability::normal_cdf;

    fn saved_survival_default_score_span() -> exact_kernel::LocalSpanCubic {
        exact_kernel::LocalSpanCubic {
            left: 0.0,
            right: 1.0,
            c0: 0.0,
            c1: 0.0,
            c2: 0.0,
            c3: 0.0,
        }
    }

    fn saved_survival_default_link_span() -> exact_kernel::LocalSpanCubic {
        exact_kernel::LocalSpanCubic {
            left: 0.0,
            right: 1.0,
            c0: 0.0,
            c1: 0.0,
            c2: 0.0,
            c3: 0.0,
        }
    }

    fn saved_survival_denested_partition_cells(
        a: f64,
        b: f64,
        gaussian_frailty_sd: Option<f64>,
        score_runtime: Option<&SavedCompiledFlexBlock>,
        score_beta: Option<&Array1<f64>>,
        link_runtime: Option<&SavedCompiledFlexBlock>,
        link_beta: Option<&Array1<f64>>,
    ) -> Result<Vec<exact_kernel::DenestedPartitionCell>, String> {
        let score_breaks = if let Some(runtime) = score_runtime {
            runtime.breakpoints()?
        } else {
            Vec::new()
        };
        let link_breaks = if let Some(runtime) = link_runtime {
            runtime.breakpoints()?
        } else {
            Vec::new()
        };
        let mut cells = exact_kernel::build_denested_partition_cells_with_tails(
            a,
            b,
            &score_breaks,
            &link_breaks,
            |z| {
                if let (Some(runtime), Some(beta)) = (score_runtime, score_beta) {
                    runtime.local_cubic_at(beta.view(), z).map_err(String::from)
                } else {
                    Ok(saved_survival_default_score_span())
                }
            },
            |u| {
                if let (Some(runtime), Some(beta)) = (link_runtime, link_beta) {
                    runtime.local_cubic_at(beta.view(), u).map_err(String::from)
                } else {
                    Ok(saved_survival_default_link_span())
                }
            },
        )?;
        let scale = probit_frailty_scale(gaussian_frailty_sd);
        if scale != 1.0 {
            for partition_cell in &mut cells {
                partition_cell.cell.c0 *= scale;
                partition_cell.cell.c1 *= scale;
                partition_cell.cell.c2 *= scale;
                partition_cell.cell.c3 *= scale;
            }
        }
        Ok(cells)
    }

    fn evaluate_saved_survival_calibration(
        a: f64,
        q: f64,
        slope: f64,
        gaussian_frailty_sd: Option<f64>,
        score_runtime: Option<&SavedCompiledFlexBlock>,
        score_beta: Option<&Array1<f64>>,
        link_runtime: Option<&SavedCompiledFlexBlock>,
        link_beta: Option<&Array1<f64>>,
    ) -> Result<(f64, f64), String> {
        let cells = saved_survival_denested_partition_cells(
            a,
            slope,
            gaussian_frailty_sd,
            score_runtime,
            score_beta,
            link_runtime,
            link_beta,
        )?;
        let scale = probit_frailty_scale(gaussian_frailty_sd);
        let mut f = -gam::probability::normal_cdf(-q);
        let mut f_a = 0.0;
        for partition_cell in cells {
            let pos_cell = partition_cell.cell;
            let neg_cell = exact_kernel::DenestedCubicCell {
                left: pos_cell.left,
                right: pos_cell.right,
                c0: -pos_cell.c0,
                c1: -pos_cell.c1,
                c2: -pos_cell.c2,
                c3: -pos_cell.c3,
            };
            let state = exact_kernel::evaluate_cell_moments(neg_cell, 3)?;
            f += state.value;
            let (dc_da_pos, _) = exact_kernel::denested_cell_coefficient_partials(
                partition_cell.score_span,
                partition_cell.link_span,
                a,
                slope,
            );
            let dc_da = scale_coeff4(dc_da_pos, -scale);
            f_a += exact_kernel::cell_first_derivative_from_moments(&dc_da, &state.moments)?;
        }
        Ok((f, f_a))
    }

    fn solve_saved_survival_intercept(
        q: f64,
        slope: f64,
        gaussian_frailty_sd: Option<f64>,
        score_runtime: Option<&SavedCompiledFlexBlock>,
        score_beta: Option<&Array1<f64>>,
        link_runtime: Option<&SavedCompiledFlexBlock>,
        link_beta: Option<&Array1<f64>>,
    ) -> Result<f64, String> {
        let eval = |a: f64| -> Result<(f64, f64, f64), String> {
            let (f, f_a) = evaluate_saved_survival_calibration(
                a,
                q,
                slope,
                gaussian_frailty_sd,
                score_runtime,
                score_beta,
                link_runtime,
                link_beta,
            )?;
            Ok((f, f_a, 0.0))
        };
        let scale = probit_frailty_scale(gaussian_frailty_sd);
        let a_init = q * (1.0 + (scale * slope) * (scale * slope)).sqrt();
        let (root, _, residual) = gam::families::monotone_root::solve_monotone_root(
            eval,
            a_init,
            "saved survival intercept",
            1e-12,
            64,
            64,
        )
        // Top-level CLI entry returns Result<_, String>; stringify the typed
        // monotone-root error here to keep the surrounding pipeline uniform.
        .map_err(|e| e.to_string())?;
        let target_survival = gam::probability::normal_cdf(-q);
        let tail_mass = target_survival.min(1.0 - target_survival).max(0.0);
        let probability_tol = 1e-12_f64.max(1e-8 * tail_mass);
        let mut residual_ok = residual.abs() <= probability_tol;
        if target_survival < 1e-8 {
            let achieved_survival = target_survival + residual;
            residual_ok = if target_survival.is_finite()
                && target_survival > 0.0
                && achieved_survival.is_finite()
                && achieved_survival > 0.0
            {
                (achieved_survival.ln() - target_survival.ln()).abs() <= 1e-8
            } else {
                residual_ok
            };
        }
        if !residual_ok {
            return Err(format!(
                "saved survival marginal-slope intercept solve failed: \
                     residual={residual:.3e} at a={root:.6}, target survival={target_survival:.6e}, \
                     probability_tol={probability_tol:.3e}"
            ));
        }
        Ok(root)
    }

    struct SavedSurvivalMarginalSlopeEtaTransport {
        eta: Array1<f64>,
        mean: Array1<f64>,
    }

    fn saved_survival_marginal_slope_eta_transport(
        q_exit: &Array1<f64>,
        slope: &Array1<f64>,
        z: &Array1<f64>,
        gaussian_frailty_sd: Option<f64>,
        score_runtime: Option<&SavedCompiledFlexBlock>,
        score_beta: Option<&Array1<f64>>,
        link_runtime: Option<&SavedCompiledFlexBlock>,
        link_beta: Option<&Array1<f64>>,
    ) -> Result<SavedSurvivalMarginalSlopeEtaTransport, String> {
        let n = q_exit.len();
        if slope.len() != n || z.len() != n {
            return Err(format!(
                "saved survival marginal-slope transport length mismatch: q={} slope={} z={}",
                n,
                slope.len(),
                z.len()
            ));
        }
        if score_runtime.is_some() != score_beta.is_some() {
            return Err(
                "saved survival marginal-slope score-warp runtime/coefficients are inconsistent"
                    .to_string(),
            );
        }
        if link_runtime.is_some() != link_beta.is_some() {
            return Err(
                    "saved survival marginal-slope link-deviation runtime/coefficients are inconsistent"
                        .to_string(),
                );
        }
        let scale = probit_frailty_scale(gaussian_frailty_sd);
        let flex_active = score_runtime.is_some() || link_runtime.is_some();
        if !flex_active {
            let sb = slope.mapv(|value| scale * value);
            let c = sb.mapv(|value| (1.0 + value * value).sqrt());
            let eta = q_exit * &c + &sb * z;
            let mean = eta.mapv(normal_cdf);
            return Ok(SavedSurvivalMarginalSlopeEtaTransport { eta, mean });
        }

        let score_obs_design = if let Some(runtime) = score_runtime {
            Some(runtime.design(z).map_err(|err| {
                format!("saved survival marginal-slope score-warp design failed: {err}")
            })?)
        } else {
            None
        };
        let score_dev_obs =
            if let (Some(design), Some(beta)) = (score_obs_design.as_ref(), score_beta) {
                design.dot(beta)
            } else {
                Array1::zeros(n)
            };

        let mut intercepts = Array1::<f64>::zeros(n);
        for row in 0..n {
            intercepts[row] = solve_saved_survival_intercept(
                q_exit[row],
                slope[row],
                gaussian_frailty_sd,
                score_runtime,
                score_beta,
                link_runtime,
                link_beta,
            )?;
        }

        let eta_base = &intercepts + &(slope * z);
        let link_dev_obs = if let (Some(runtime), Some(beta)) = (link_runtime, link_beta) {
            runtime
                .design(&eta_base)
                .map_err(|err| {
                    format!("saved survival marginal-slope link-deviation design failed: {err}")
                })?
                .dot(beta)
        } else {
            Array1::zeros(n)
        };
        let eta =
            (&eta_base + &(slope * &score_dev_obs) + &link_dev_obs).mapv(|value| scale * value);
        let mean = eta.mapv(normal_cdf);
        Ok(SavedSurvivalMarginalSlopeEtaTransport { eta, mean })
    }

    pub(super) fn predict_saved_survival_marginal_slope_flex_exit(
        q_exit: &Array1<f64>,
        slope: &Array1<f64>,
        z: &Array1<f64>,
        gaussian_frailty_sd: Option<f64>,
        score_runtime: Option<&SavedCompiledFlexBlock>,
        score_beta: Option<&Array1<f64>>,
        link_runtime: Option<&SavedCompiledFlexBlock>,
        link_beta: Option<&Array1<f64>>,
    ) -> Result<(Array1<f64>, Array1<f64>), String> {
        let transport = saved_survival_marginal_slope_eta_transport(
            q_exit,
            slope,
            z,
            gaussian_frailty_sd,
            score_runtime,
            score_beta,
            link_runtime,
            link_beta,
        )?;
        Ok((transport.eta, transport.mean))
    }
}

fn csv_mean_at(path: &std::path::Path, row_idx: usize) -> f64 {
    let mut rdr = csv::Reader::from_path(path)
        .unwrap_or_else(|e| panic!("{} failed: {:?}", "open prediction csv", e));
    let rows = rdr
        .deserialize::<BTreeMap<String, String>>()
        .collect::<Result<Vec<_>, _>>()
        .unwrap_or_else(|e| panic!("{} failed: {:?}", "parse prediction csv", e));
    rows[row_idx]["mean"]
        .parse::<f64>()
        .unwrap_or_else(|e| panic!("{} failed: {:?}", "mean should parse", e))
}

fn write_binomial_location_scale_train_csv(path: &std::path::Path) {
    fs::write(
            path,
            "x1,x2,y\n-2.0,-1.2,0\n-1.7,0.4,0\n-1.5,-0.7,0\n-1.2,1.1,1\n-1.0,-0.3,0\n-0.8,0.9,0\n-0.5,-1.1,1\n-0.2,0.2,0\n0.0,-0.8,1\n0.3,1.0,0\n0.5,-0.4,1\n0.7,0.6,1\n0.9,-1.3,0\n1.1,0.3,1\n1.4,-0.2,1\n1.8,1.2,1\n",
        )
        .expect("write training csv");
}

fn write_bernoulli_marginal_slope_train_csv(path: &std::path::Path) {
    fs::write(
            path,
            "x,z,y\n-1.4,-1.2816,0\n-1.1,-0.8416,0\n-0.9,-0.5244,0\n-0.6,-0.2533,0\n-0.3,0.0000,1\n0.0,0.2533,0\n0.2,0.5244,1\n0.5,0.8416,1\n0.8,1.2816,1\n1.0,-0.5244,0\n1.2,0.5244,1\n1.4,0.8416,1\n",
        )
        .unwrap_or_else(|e| panic!("{} failed: {:?}", "write marginal-slope training csv", e));
}

fn location_scale_fit_args(
    data: PathBuf,
    out: PathBuf,
    formula: &str,
    noise_formula: &str,
) -> FitArgs {
    FitArgs {
        inference: true,
        expectile_tau: None,
        data,
        request: None,
        formula_positional: Some(formula.to_string()),
        ctn_stage1: None,
        precision_hyperpriors: None,
        latent_coordinates: None,
        analytic_penalties: None,
        smooth_descriptors: None,
        predict_noise: Some(noise_formula.to_string()),
        slope_formula: None,
        z_column: None,
        weights_column: None,
        offset_column: None,
        noise_offset_column: None,
        frailty_kind: None,
        frailty_sd: None,
        hazard_loading: None,
        transformation_normal: false,
        firth: false,
        family: FamilyArg::Auto,
        negative_binomial_theta: None,
        survival_likelihood: Some("transformation".to_string()),
        survival_time_anchor: None,
        baseline_target: "linear".to_string(),
        baseline_scale: None,
        baseline_shape: None,
        baseline_rate: None,
        baseline_makeham: None,
        time_basis: "ispline".to_string(),
        time_degree: 3,
        time_num_internal_knots: 8,
        time_smooth_lambda: 1e-2,
        ridge_lambda: 1e-6,
        threshold_time_k: None,
        threshold_time_degree: 3,
        sigma_time_k: None,
        sigma_time_degree: 3,
        slope_time_k: None,
        slope_time_degree: 3,
        adaptive_regularization: false,
        scale_dimensions: false,
        precompute_conformal: true,
        persistent_warm_start_root: None,
        pilot_subsample_threshold: 0,
        out: Some(out),
    }
}

#[test]
fn cli_predict_defaults_to_posterior_mean_instead_of_map() {
    let cli = Cli::parse_from([
        "gam",
        "predict",
        "model.json",
        "new_data.csv",
        "--out",
        "predictions.csv",
    ]);
    let Command::Predict(args) = cli.command else {
        panic!("expected predict command");
    };
    assert_eq!(args.mode, PredictModeArg::PosteriorMean);
    assert_ne!(args.mode, PredictModeArg::Map);
}

#[test]
fn cli_log_level_is_typed_and_rejects_unknown_values_2670() {
    let parsed = Cli::try_parse_from(["gam", "--log-level", "debug", "report", "model.json"])
        .expect("a canonical log level must parse");
    assert_eq!(parsed.log_level, Some(log::LevelFilter::Debug));

    let error = Cli::try_parse_from([
        "gam",
        "--log-level",
        "verbose",
        "report",
        "model.json",
    ])
    .expect_err("an unknown log level must be rejected rather than guessed as info");
    assert_eq!(error.kind(), clap::error::ErrorKind::ValueValidation);
    let rendered = error.to_string();
    assert!(
        rendered.contains("accepted values: off, error, warn, info, debug, trace"),
        "the parser error must enumerate the canonical levels: {rendered}"
    );
}

#[test]
fn cli_fit_request_replaces_formula_and_scientific_flags() {
    let cli = Cli::try_parse_from([
        "gam",
        "fit",
        "train.csv",
        "--request",
        "request.json",
        "--out",
        "model.json",
    ])
    .expect("request mode should require only DATA, --request, and --out");
    let Command::Fit(args) = cli.command else {
        panic!("expected fit command");
    };
    assert_eq!(args.request, Some(PathBuf::from("request.json")));
    assert_eq!(args.formula_positional, None);

    for conflicting_args in [
        vec!["y ~ x"],
        vec!["--family", "auto"],
        vec!["--ridge-lambda", "0.000001"],
        vec!["--transformation-normal"],
        vec!["--latent-coordinates", "latents.json"],
        vec!["--persistent-warm-start-root", "caller-owned/warm"],
    ] {
        let mut argv = vec![
            "gam",
            "fit",
            "train.csv",
            "--request",
            "request.json",
            "--out",
            "model.json",
        ];
        argv.extend(conflicting_args);
        assert!(
            Cli::try_parse_from(argv).is_err(),
            "request mode must reject formula/scientific flag overlays"
        );
    }
}

#[test]
fn cli_persistent_warm_start_root_is_explicit_and_preserved_exactly_2639() {
    let cli = Cli::try_parse_from([
        "gam",
        "fit",
        "train.csv",
        "y ~ x",
        "--persistent-warm-start-root",
        "caller-owned/../warm",
        "--out",
        "model.json",
    ])
    .expect("an explicit persistent warm-start root should parse");
    let Command::Fit(args) = cli.command else {
        panic!("expected fit command");
    };
    assert_eq!(
        args.persistent_warm_start_root,
        Some(PathBuf::from("caller-owned/../warm")),
        "the CLI must not canonicalize or relocate the requested root"
    );
}

#[test]
fn cli_firth_validation_uses_shared_family_support_rule() {
    let err = validate_cli_firth_configuration(CliFirthValidation {
        enabled: true,
        family: LikelihoodSpec::poisson_log(),
        predict_noise: false,
        is_survival: false,
        link_choice: None,
    })
    .expect_err("Poisson Firth should be rejected through the shared family policy");

    let err = err.to_string();
    assert!(
        err.contains("Binomial inverse link with a Fisher-weight jet"),
        "unexpected error message: {err}"
    );
}

#[test]
fn cli_sample_bounded_model_reaches_sampler_config_validation() {
    let td = tempdir().unwrap_or_else(|e| panic!("{} failed: {:?}", "tempdir", e));
    let model_path = td.path().join("bounded.model.json");
    let data_path = td.path().join("bounded.csv");
    let out_path = td.path().join("draws.csv");

    fs::write(&data_path, "x,y\n0.0,0.0\n0.5,1.0\n1.0,1.0\n")
        .unwrap_or_else(|e| panic!("{} failed: {:?}", "write data", e));

    let mut payload = test_payload(
        "y ~ bounded(x, min=-2, max=2)",
        ModelKind::Standard,
        FittedFamily::Standard {
            likelihood: LikelihoodSpec::gaussian_identity(),
            link: Some(StandardLink::Identity),
            latent_cloglog_state: None,
            mixture_state: None,
            sas_state: None,
        },
        LikelihoodSpec::gaussian_identity().name(),
    );
    // The bounded design is `[intercept, bounded(x)]` (2 coefficients). A
    // persistable model requires a canonical fit_result + training headers, so
    // build a single Mean block with a well-conditioned user-scale penalized
    // Hessian; this lets `run_sample` load the model and reach the unified
    // sampler dispatch (which validates the NUTS config before drawing).
    let fit_result = compact_saved_multiblock_fit_result(
        vec![FittedBlock {
            beta: array![0.1, 0.2],
            role: BlockRole::Mean,
            edf: 2.0,
            lambdas: Array1::zeros(0),
        }],
        Array1::zeros(0),
        1.0,
        None,
        None,
        Some(FitGeometry {
            coefficient_gauge: Gauge::identity(&[2]),
            penalized_hessian: array![[4.0, 1.0], [1.0, 3.0]].into(),
            constrained_posterior: None,
            working: Some(gam::estimate::WorkingGeometry {
                weights: array![1.0, 1.0, 1.0],
                response: array![0.0, 1.0, 1.0],
            }),
        }),
        {
            // The shared fixture declares `training_sample_size: 1`, but the
            // working geometry above carries three rows. The sampler checks the
            // two agree, so the fixture has to state the size it actually has;
            // the shared default is left alone because its other dependents
            // carry different row counts.
            let mut summary = saved_fit_summary_fixture();
            summary.training_sample_size = 3;
            summary
        },
    );
    payload.fit_result = Some(fit_result);
    payload.data_schema = Some(bounded_cli_schema());
    payload.resolved_termspec = Some(bounded_cli_termspec());
    payload.set_training_feature_metadata(vec!["x".to_string(), "y".to_string()], vec![]);
    write_model_json(&model_path, &SavedModel::from_payload(payload))
        .unwrap_or_else(|e| panic!("{} failed: {:?}", "write model", e));

    let err = run_sample(SampleArgs {
        model: model_path,
        data: data_path,
        chains: Some(1),
        samples: Some(1),
        warmup: Some(1),
        seed: Some(760),
        out: Some(out_path),
    })
    .expect_err("invalid draw count should fail inside sampler validation");

    assert!(
        err.contains("NUTS n_samples"),
        "bounded sample dispatch should reach sampler validation, got {err}"
    );
    assert!(
        !err.to_ascii_lowercase().contains("bounded"),
        "sample must not reject bounded() coefficients before sampler dispatch: {err}"
    );
}

#[test]
fn required_columns_for_fit_includes_auxiliary_formula_columns() {
    let parsed = parse_formula("y ~ x + s(pc1, pc2, type=tensor)")
        .unwrap_or_else(|e| panic!("{} failed: {:?}", "parse main formula", e));
    let mut args = location_scale_fit_args(
        PathBuf::from("train.csv"),
        PathBuf::from("model.json"),
        "y ~ x + s(pc1, pc2, type=tensor)",
        "z + smooth(w)",
    );
    args.slope_formula = Some("slope_x + slope_z".to_string());
    args.z_column = Some("z_anchor".to_string());

    let required = required_columns_for_fit(&args, &parsed)
        .unwrap_or_else(|e| panic!("{} failed: {:?}", "required columns", e));

    assert_eq!(
        required,
        vec![
            "pc1".to_string(),
            "pc2".to_string(),
            "slope_x".to_string(),
            "slope_z".to_string(),
            "w".to_string(),
            "x".to_string(),
            "y".to_string(),
            "z".to_string(),
            "z_anchor".to_string(),
        ]
    );
}

#[test]
fn load_dataset_projected_keeps_only_requested_columns() {
    let dir = tempdir().unwrap_or_else(|e| panic!("{} failed: {:?}", "tempdir", e));
    let csv_path = dir.path().join("projected.csv");
    fs::write(
        &csv_path,
        "unused_a,x,unused_b,y\n1,10,100,0\n2,11,101,1\n3,12,102,0\n",
    )
    .unwrap_or_else(|e| panic!("{} failed: {:?}", "write csv", e));

    let ds = load_dataset_projected(&csv_path, &["x".to_string(), "y".to_string()])
        .unwrap_or_else(|e| panic!("{} failed: {:?}", "load projected csv", e));

    assert_eq!(ds.headers, vec!["x".to_string(), "y".to_string()]);
    assert_eq!(ds.values.nrows(), 3);
    assert_eq!(ds.values.ncols(), 2);
    assert_eq!(ds.values[[1, 0]], 11.0);
    assert_eq!(ds.values[[1, 1]], 1.0);
}

/// The CLI flag rule `--negative-binomial-theta` requires
/// `--family negative-binomial` is a surface concern owned by the CLI
/// adapter (the canonical resolver only rejects a theta with no family at
/// all). Guard it explicitly so the adapter keeps enforcing it.
#[test]
fn cli_resolve_family_rejects_theta_without_negative_binomial() {
    let y = array![0.0, 1.0, 2.0, 3.0];
    let err = resolve_family(
        FamilyArg::PoissonLog,
        Some(2.0),
        None,
        y.view(),
        ResponseColumnKind::Numeric,
        "y",
    )
    .expect_err("theta without negative-binomial family must be rejected");
    assert_eq!(
        err,
        "--negative-binomial-theta requires --family negative-binomial"
    );
}

#[test]
fn cli_firth_validation_rejects_survival_models() {
    let err = validate_cli_firth_configuration(CliFirthValidation {
        enabled: true,
        family: LikelihoodSpec::royston_parmar(),
        predict_noise: false,
        is_survival: true,
        link_choice: None,
    })
    .expect_err("survival Firth should be rejected");

    assert_eq!(
        err.to_string(),
        "--firth is not supported for survival models"
    );
}

#[test]
fn cli_firth_preflight_accepts_redundant_survival_marginal_slope_flag() {
    let parsed = parse_formula("Surv(t0, t1, event) ~ x")
        .unwrap_or_else(|e| panic!("{} failed: {:?}", "parse survival formula", e));
    let mut args = location_scale_fit_args(
        PathBuf::from("train.csv"),
        PathBuf::from("model.json"),
        "Surv(t0, t1, event) ~ x",
        "unused",
    );
    args.predict_noise = None;
    args.slope_formula = Some("1".to_string());
    args.z_column = Some("z".to_string());
    args.survival_likelihood = Some("marginal-slope".to_string());
    args.firth = true;

    let fit_config = super::resolve_fit_invocation(&args)
        .expect("fit config should resolve")
        .fit_config;
    let result = validate_fit_args_preflight(&args, &parsed, &fit_config);
    assert!(
        result.is_ok(),
        "--firth is redundant, not rejected, for marginal-slope: {result:?}"
    );
}

#[test]
fn issue_2116_cli_standard_fit_gates_duchon_operator_penalties_for_poisson() {
    // #2116: the `gam` CLI and the `gamfit` Python API are two front-ends of ONE
    // shared engine (#1191/#1196). The Python/materialize standard path drops the
    // Duchon *operator* penalties (the mass/tension collocation-Gram blocks) for a
    // non-Gaussian-identity family via `gate_duchon_operator_penalties_for_family`
    // (materialize/standard.rs), but the CLI's hand-built `StandardFitRequest`
    // never applied that gate — so a Duchon smooth under e.g. Poisson fit a
    // DIFFERENT penalty structure through the CLI than through Python, a genuine
    // single-engine-contract violation. `run_fit` now applies the SAME gate. This
    // test drives the real CLI fit end-to-end and pins that the persisted (frozen)
    // Duchon term carries ALL operator penalties DISABLED under Poisson — matching
    // the materialize path. Before the fix the frozen term kept the default
    // (mass + tension Active), so the assertion failed; after the fix it passes.
    let td = tempdir().unwrap_or_else(|e| panic!("{} failed: {:?}", "tempdir", e));
    let train_path = td.path().join("duchon_poisson.csv");
    let model_path = td.path().join("model.json");

    // Deterministic 7x7 spatial grid with a smooth log-linear Poisson mean; every
    // count is a non-negative integer so the Poisson support check passes.
    let mut csv = String::from("y,pc1,pc2\n");
    for i in 0..7i32 {
        for j in 0..7i32 {
            let pc1 = f64::from(i) / 6.0;
            let pc2 = f64::from(j) / 6.0;
            let mean = (0.4 + 1.1 * pc1 + 0.7 * pc2).exp();
            let y = mean.round() as i64;
            csv.push_str(&format!("{y},{pc1:.6},{pc2:.6}\n"));
        }
    }
    fs::write(&train_path, csv).unwrap_or_else(|e| panic!("{} failed: {:?}", "write csv", e));

    run_fit(FitArgs {
        inference: true,
        expectile_tau: None,
        data: train_path,
        request: None,
        formula_positional: Some("y ~ s(pc1, pc2, type=duchon, centers=6)".to_string()),
        ctn_stage1: None,
        precision_hyperpriors: None,
        latent_coordinates: None,
        analytic_penalties: None,
        smooth_descriptors: None,
        predict_noise: None,
        slope_formula: None,
        z_column: None,
        weights_column: None,
        offset_column: None,
        noise_offset_column: None,
        frailty_kind: None,
        frailty_sd: None,
        hazard_loading: None,
        transformation_normal: false,
        firth: false,
        family: FamilyArg::PoissonLog,
        negative_binomial_theta: None,
        // `survival_likelihood` is read exclusively by the survival fit path.
        // On a Poisson response nothing consumes it, so the requested survival
        // model would silently degrade to an ordinary GAM -- the fit now refuses
        // it rather than ignoring it. This test gates Duchon operator penalties
        // and never needed the option.
        survival_likelihood: None,
        survival_time_anchor: None,
        baseline_target: "linear".to_string(),
        baseline_scale: None,
        baseline_shape: None,
        baseline_rate: None,
        baseline_makeham: None,
        time_basis: "ispline".to_string(),
        time_degree: 3,
        time_num_internal_knots: 8,
        time_smooth_lambda: 1e-2,
        ridge_lambda: 1e-6,
        threshold_time_k: None,
        threshold_time_degree: 3,
        sigma_time_k: None,
        sigma_time_degree: 3,
        slope_time_k: None,
        slope_time_degree: 3,
        adaptive_regularization: false,
        scale_dimensions: false,
        precompute_conformal: true,
        persistent_warm_start_root: None,
        pilot_subsample_threshold: 0,
        out: Some(model_path.clone()),
    })
    .unwrap_or_else(|e| {
        panic!(
            "{} failed: {:?}",
            "CLI Poisson Duchon fit should succeed", e
        )
    });

    let saved = SavedModel::load_from_path(&model_path)
        .unwrap_or_else(|e| panic!("{} failed: {:?}", "load fitted model", e));
    let spec = saved
        .resolved_termspec
        .as_ref()
        .expect("standard fit must persist a resolved termspec");
    let duchon = spec
        .smooth_terms
        .iter()
        .find_map(|term| match &term.basis {
            SmoothBasisSpec::Duchon { spec, .. } => Some(spec),
            _ => None,
        })
        .expect("resolved termspec must contain the Duchon smooth");

    use gam::basis::OperatorPenaltySpec::Disabled;
    assert!(
        matches!(duchon.operator_penalties.mass, Disabled)
            && matches!(duchon.operator_penalties.tension, Disabled)
            && matches!(duchon.operator_penalties.stiffness, Disabled),
        "CLI standard fit under Poisson must gate the Duchon operator penalties \
         (mass/tension collocation-Gram blocks) off, matching the Python/materialize \
         path (#2116); got {:?}",
        duchon.operator_penalties
    );
}

/// #2631: the CLI and the engine must resolve the SAME baseline time anchor for
/// the same formula, data and config.
///
/// This is the issue's title as an assertion. `run_survival` used to carry its
/// own copy of the anchor rule that promoted the robust interior anchor for
/// marginal-slope only, while `materialize_survival` promoted it for any
/// left-truncated dataset — so on this fixture the CLI centered a location-scale
/// fit at the earliest entry (10) and `fit_from_formula` centered it at the
/// median exit (80). Both were internally consistent, which is exactly why
/// nothing caught it: the fits simply differed.
///
/// The fixture is chosen so the two candidate anchors are far apart and the
/// median exit is exact: entries all delayed (min 10), six exits whose two
/// central values are 60 and 100, so the median exit is 80.
#[test]
fn cli_and_engine_agree_on_the_left_truncated_survival_anchor_2631() {
    const EARLIEST_ENTRY: f64 = 10.0;
    const MEDIAN_EXIT: f64 = 80.0;
    let td = tempdir().unwrap_or_else(|e| panic!("{} failed: {:?}", "tempdir", e));
    let train_path = td.path().join("left_truncated_anchor.csv");
    let model_path = td.path().join("left_truncated_anchor.model.json");
    fs::write(
        &train_path,
        "entry,exit,event,x\n\
         10,15,1,-0.8\n\
         20,35,0,0.4\n\
         40,60,1,-0.2\n\
         80,100,0,0.7\n\
         120,150,1,0.1\n\
         160,220,1,-0.5\n",
    )
    .unwrap_or_else(|e| panic!("{} failed: {:?}", "write left-truncated csv", e));

    let formula = "Surv(entry, exit, event) ~ x";

    // ── Engine arm: the same request through `materialize`, which is what
    // `fit_from_formula` and the Python FFI use.
    let dataset = load_dataset_projected(
        &train_path,
        &[
            "entry".to_string(),
            "exit".to_string(),
            "event".to_string(),
            "x".to_string(),
        ],
    )
    .unwrap_or_else(|e| panic!("{} failed: {:?}", "load left-truncated dataset", e));
    let engine_config = gam::families::fit_orchestration::FitConfig {
        survival_likelihood: Some("location-scale".to_string()),
        ..gam::families::fit_orchestration::FitConfig::default()
    }
    .resolve()
    .unwrap_or_else(|e| panic!("{} failed: {:?}", "resolve engine fit config", e));
    let materialized =
        gam::families::fit_orchestration::materialize(formula, &dataset, &engine_config)
            .unwrap_or_else(|e| panic!("{} failed: {:?}", "engine materialization", e));
    let engine_anchor = materialized
        .survival_time_basis
        .expect("a survival materialization must carry its realised time basis")
        .anchor;

    // ── CLI arm: the same request through `run_fit` -> `run_survival`.
    run_fit(FitArgs {
        inference: true,
        expectile_tau: None,
        data: train_path.clone(),
        request: None,
        formula_positional: Some(formula.to_string()),
        ctn_stage1: None,
        precision_hyperpriors: None,
        latent_coordinates: None,
        analytic_penalties: None,
        smooth_descriptors: None,
        predict_noise: None,
        slope_formula: None,
        z_column: None,
        weights_column: None,
        offset_column: None,
        noise_offset_column: None,
        frailty_kind: None,
        frailty_sd: None,
        hazard_loading: None,
        transformation_normal: false,
        firth: false,
        family: FamilyArg::Auto,
        negative_binomial_theta: None,
        survival_likelihood: Some("location-scale".to_string()),
        survival_time_anchor: None,
        baseline_target: "linear".to_string(),
        baseline_scale: None,
        baseline_shape: None,
        baseline_rate: None,
        baseline_makeham: None,
        time_basis: "ispline".to_string(),
        time_degree: 2,
        time_num_internal_knots: 4,
        time_smooth_lambda: 1e-2,
        ridge_lambda: 1e-6,
        threshold_time_k: None,
        threshold_time_degree: 3,
        sigma_time_k: None,
        sigma_time_degree: 3,
        slope_time_k: None,
        slope_time_degree: 3,
        adaptive_regularization: false,
        scale_dimensions: false,
        precompute_conformal: true,
        persistent_warm_start_root: None,
        pilot_subsample_threshold: 0,
        out: Some(model_path.clone()),
    })
    .unwrap_or_else(|e| {
        panic!(
            "{} failed: {:?}",
            "left-truncated location-scale CLI fit should succeed", e
        )
    });
    let saved = SavedModel::load_from_path(&model_path)
        .unwrap_or_else(|e| panic!("{} failed: {:?}", "load CLI-fitted model", e));
    let cli_anchor = saved
        .survival_time_anchor
        .expect("a saved survival model must carry its time anchor");

    assert!(
        (cli_anchor - engine_anchor).abs() <= 1e-12,
        "the CLI and the engine must resolve the same survival time anchor for \
         identical inputs; CLI got {cli_anchor}, engine got {engine_anchor}"
    );
    assert!(
        (engine_anchor - MEDIAN_EXIT).abs() <= 1e-12,
        "left-truncated data must anchor at the robust median exit \
         ({MEDIAN_EXIT}), got {engine_anchor}"
    );
    assert!(
        (cli_anchor - EARLIEST_ENTRY).abs() > 1.0,
        "the CLI must not fall back to the earliest entry ({EARLIEST_ENTRY}) — \
         that is the #2631 divergence"
    );
    remove_temp_file(&model_path);
}

/// #2631: `--survival-time-anchor` must be honored on the CLI's DEFAULT survival
/// route.
///
/// `Transformation` and `Weibull` short-circuit to
/// `run_canonical_survival_transformation`, which delegates to
/// `fit_from_formula`. The anchor override used to be read only by the CLI's own
/// anchor computation further down the function, which that route never reaches,
/// and `FitConfig` had no field to carry it — so the flag was parsed, validated,
/// and dropped on the floor for the default likelihood. The CLI's own comment
/// claimed it was "honored by all paths".
///
/// The explicit value (25) is neither the earliest entry nor the median exit of
/// this fixture, so only an honored override can produce it. The fixture is
/// right-censored (`Surv(exit, event)`, entry synthesized at the origin) because
/// the assertion is about the OVERRIDE reaching the fit, and the default anchor
/// there is the time-origin floor — as far from 25 as the left-truncated default
/// would be, with none of the left-truncated transformation fit's convergence
/// fragility in the way.
#[test]
fn cli_survival_time_anchor_is_honored_on_the_default_transformation_route_2631() {
    const EXPLICIT_ANCHOR: f64 = 25.0;
    let td = tempdir().unwrap_or_else(|e| panic!("{} failed: {:?}", "tempdir", e));
    let train_path = td.path().join("explicit_anchor.csv");
    let model_path = td.path().join("explicit_anchor.model.json");
    fs::write(
        &train_path,
        "exit,event,x\n\
         15,1,-0.8\n\
         35,0,0.4\n\
         60,1,-0.2\n\
         100,0,0.7\n\
         150,1,0.1\n\
         220,1,-0.5\n",
    )
    .unwrap_or_else(|e| panic!("{} failed: {:?}", "write survival csv", e));

    run_fit(FitArgs {
        inference: true,
        expectile_tau: None,
        data: train_path.clone(),
        request: None,
        formula_positional: Some("Surv(exit, event) ~ x".to_string()),
        ctn_stage1: None,
        precision_hyperpriors: None,
        latent_coordinates: None,
        analytic_penalties: None,
        smooth_descriptors: None,
        predict_noise: None,
        slope_formula: None,
        z_column: None,
        weights_column: None,
        offset_column: None,
        noise_offset_column: None,
        frailty_kind: None,
        frailty_sd: None,
        hazard_loading: None,
        transformation_normal: false,
        firth: false,
        family: FamilyArg::Auto,
        negative_binomial_theta: None,
        // The default route — the one that delegated to the engine and lost the
        // flag. `Weibull` short-circuits through the same branch and is covered
        // by the sibling test below.
        survival_likelihood: Some("transformation".to_string()),
        survival_time_anchor: Some(EXPLICIT_ANCHOR),
        baseline_target: "linear".to_string(),
        baseline_scale: None,
        baseline_shape: None,
        baseline_rate: None,
        baseline_makeham: None,
        time_basis: "ispline".to_string(),
        time_degree: 2,
        time_num_internal_knots: 4,
        time_smooth_lambda: 1e-2,
        ridge_lambda: 1e-6,
        threshold_time_k: None,
        threshold_time_degree: 3,
        sigma_time_k: None,
        sigma_time_degree: 3,
        slope_time_k: None,
        slope_time_degree: 3,
        adaptive_regularization: false,
        scale_dimensions: false,
        precompute_conformal: true,
        persistent_warm_start_root: None,
        pilot_subsample_threshold: 0,
        out: Some(model_path.clone()),
    })
    .unwrap_or_else(|e| {
        panic!(
            "{} failed: {:?}",
            "explicit-anchor transformation CLI fit should succeed", e
        )
    });

    let saved = SavedModel::load_from_path(&model_path)
        .unwrap_or_else(|e| panic!("{} failed: {:?}", "load fitted model", e));
    let anchor = saved
        .survival_time_anchor
        .expect("a saved survival model must carry its time anchor");
    assert!(
        (anchor - EXPLICIT_ANCHOR).abs() <= 1e-12,
        "--survival-time-anchor must be honored on the default transformation \
         route; requested {EXPLICIT_ANCHOR}, saved {anchor}"
    );
    remove_temp_file(&model_path);
}

/// #2631, the left-truncated half of the default route: `Weibull` takes the same
/// `run_canonical_survival_transformation` short-circuit as `Transformation`, and
/// converges on a thin left-truncated fixture where the Royston-Parmar
/// transformation fit does not — so it is where both halves of the fix can be
/// asserted on ONE dataset.
///
/// Default: the robust median exit (80), not the earliest entry (10).
/// Override: honored (25), where the old code discarded it and produced 80.
#[test]
fn cli_weibull_route_anchors_left_truncated_data_and_honors_the_override_2631() {
    const MEDIAN_EXIT: f64 = 80.0;
    const EXPLICIT_ANCHOR: f64 = 25.0;
    let td = tempdir().unwrap_or_else(|e| panic!("{} failed: {:?}", "tempdir", e));
    let train_path = td.path().join("weibull_left_truncated.csv");
    fs::write(
        &train_path,
        "entry,exit,event,x\n\
         10,15,1,-0.8\n\
         20,35,0,0.4\n\
         40,60,1,-0.2\n\
         80,100,0,0.7\n\
         120,150,1,0.1\n\
         160,220,1,-0.5\n",
    )
    .unwrap_or_else(|e| panic!("{} failed: {:?}", "write left-truncated csv", e));

    for (requested, expected) in [
        (None, MEDIAN_EXIT),
        (Some(EXPLICIT_ANCHOR), EXPLICIT_ANCHOR),
    ] {
        let model_path = td.path().join(match requested {
            Some(_) => "weibull_explicit.model.json",
            None => "weibull_default.model.json",
        });
        run_fit(FitArgs {
            inference: true,
            expectile_tau: None,
            data: train_path.clone(),
            request: None,
            formula_positional: Some("Surv(entry, exit, event) ~ x".to_string()),
            ctn_stage1: None,
            precision_hyperpriors: None,
            latent_coordinates: None,
            analytic_penalties: None,
            smooth_descriptors: None,
            predict_noise: None,
            slope_formula: None,
            z_column: None,
            weights_column: None,
            offset_column: None,
            noise_offset_column: None,
            frailty_kind: None,
            frailty_sd: None,
            hazard_loading: None,
            transformation_normal: false,
            firth: false,
            family: FamilyArg::Auto,
            negative_binomial_theta: None,
            survival_likelihood: Some("weibull".to_string()),
            survival_time_anchor: requested,
            baseline_target: "linear".to_string(),
            baseline_scale: None,
            baseline_shape: None,
            baseline_rate: None,
            baseline_makeham: None,
            time_basis: "ispline".to_string(),
            time_degree: 3,
            time_num_internal_knots: 8,
            time_smooth_lambda: 1e-2,
            ridge_lambda: 1e-6,
            threshold_time_k: None,
            threshold_time_degree: 3,
            sigma_time_k: None,
            sigma_time_degree: 3,
            slope_time_k: None,
            slope_time_degree: 3,
            adaptive_regularization: false,
            scale_dimensions: false,
            precompute_conformal: true,
            persistent_warm_start_root: None,
            pilot_subsample_threshold: 0,
            out: Some(model_path.clone()),
        })
        .unwrap_or_else(|e| {
            panic!(
                "{} failed: {:?}",
                "left-truncated Weibull CLI fit should succeed", e
            )
        });
        let saved = SavedModel::load_from_path(&model_path)
            .unwrap_or_else(|e| panic!("{} failed: {:?}", "load fitted model", e));
        let anchor = saved
            .survival_time_anchor
            .expect("a saved survival model must carry its time anchor");
        assert!(
            (anchor - expected).abs() <= 1e-12,
            "requested anchor {requested:?} must resolve to {expected}, got {anchor}"
        );
        remove_temp_file(&model_path);
    }
}

/// #2631: a `--request` document's `survival_time_anchor` must reach the fit on
/// EVERY survival route, including the ones the CLI still materializes itself.
///
/// `--survival-time-anchor` declares `conflicts_with = --request` precisely
/// because the document is supposed to carry the complete scientific model
/// configuration. Under `--request` the flag is therefore always `None`, so a
/// survival route that read the anchor from `FitArgs` rather than from the
/// resolved `FitConfig` would drop a document-supplied anchor without a word.
/// This exercises the location-scale route, which `run_survival` materializes
/// itself rather than delegating to the engine.
#[test]
fn cli_request_document_survival_time_anchor_reaches_the_fit_2631() {
    const EXPLICIT_ANCHOR: f64 = 25.0;
    let td = tempdir().unwrap_or_else(|e| panic!("{} failed: {:?}", "tempdir", e));
    let train_path = td.path().join("request_anchor.csv");
    let request_path = td.path().join("request_anchor.request.json");
    let model_path = td.path().join("request_anchor.model.json");
    fs::write(
        &train_path,
        "entry,exit,event,x\n\
         10,15,1,-0.8\n\
         20,35,0,0.4\n\
         40,60,1,-0.2\n\
         80,100,0,0.7\n\
         120,150,1,0.1\n\
         160,220,1,-0.5\n",
    )
    .unwrap_or_else(|e| panic!("{} failed: {:?}", "write survival csv", e));
    fs::write(
        &request_path,
        format!(
            r#"{{"schema":"gam.fit-request","schema_version":1,
                 "formula":"Surv(entry, exit, event) ~ x",
                 "config":{{"survival_likelihood":"location-scale",
                            "survival_time_anchor":{EXPLICIT_ANCHOR}}}}}"#
        ),
    )
    .unwrap_or_else(|e| panic!("{} failed: {:?}", "write fit-request document", e));

    let mut args = location_scale_fit_args(
        train_path.clone(),
        model_path.clone(),
        "unused ~ when --request is supplied",
        "1",
    );
    // `--request` carries the formula and the whole model configuration; the CLI
    // rejects the conflicting flags, so they must be cleared here too.
    args.request = Some(request_path);
    args.formula_positional = None;
    args.predict_noise = None;
    args.survival_likelihood = None;
    args.family = FamilyArg::Auto;
    run_fit(args).unwrap_or_else(|e| {
        panic!(
            "{} failed: {:?}",
            "fit from a --request document with an explicit anchor should succeed", e
        )
    });

    let saved = SavedModel::load_from_path(&model_path)
        .unwrap_or_else(|e| panic!("{} failed: {:?}", "load fitted model", e));
    let anchor = saved
        .survival_time_anchor
        .expect("a saved survival model must carry its time anchor");
    assert!(
        (anchor - EXPLICIT_ANCHOR).abs() <= 1e-12,
        "a --request document's survival_time_anchor must reach the fit; \
         requested {EXPLICIT_ANCHOR}, saved {anchor}"
    );
    remove_temp_file(&model_path);
}

#[test]
fn cli_surv_predict_noise_routes_to_survival_location_scale() {
    let td = tempdir().unwrap_or_else(|e| panic!("{} failed: {:?}", "tempdir", e));
    let train_path = td.path().join("survival_train.csv");
    let model_path = td.path().join("survival.model.json");
    let pred_path = td.path().join("survival.pred.csv");
    fs::write(
        &train_path,
        "entry,exit,event\n\
             10,15,1\n\
             20,35,0\n\
             40,60,1\n\
             80,100,0\n\
             120,150,1\n\
             160,220,1\n",
    )
    .unwrap_or_else(|e| panic!("{} failed: {:?}", "write survival training csv", e));

    run_fit(FitArgs {
        inference: true,
        expectile_tau: None,
        data: train_path.clone(),
        request: None,
        formula_positional: Some("Surv(entry, exit, event) ~ 1".to_string()),
        ctn_stage1: None,
        precision_hyperpriors: None,
        latent_coordinates: None,
        analytic_penalties: None,
        smooth_descriptors: None,
        predict_noise: Some("1".to_string()),
        slope_formula: None,
        z_column: None,
        weights_column: None,
        offset_column: None,
        noise_offset_column: None,
        frailty_kind: None,
        frailty_sd: None,
        hazard_loading: None,
        transformation_normal: false,
        firth: false,
        family: FamilyArg::Auto,
        negative_binomial_theta: None,
        survival_likelihood: Some("transformation".to_string()),
        survival_time_anchor: None,
        baseline_target: "linear".to_string(),
        baseline_scale: None,
        baseline_shape: None,
        baseline_rate: None,
        baseline_makeham: None,
        time_basis: "ispline".to_string(),
        time_degree: 2,
        time_num_internal_knots: 4,
        time_smooth_lambda: 1e-2,
        ridge_lambda: 1e-6,
        threshold_time_k: None,
        threshold_time_degree: 3,
        sigma_time_k: None,
        sigma_time_degree: 3,
        slope_time_k: None,
        slope_time_degree: 3,
        adaptive_regularization: false,
        scale_dimensions: false,
        precompute_conformal: true,
        persistent_warm_start_root: None,
        pilot_subsample_threshold: 0,
        out: Some(model_path.clone()),
    })
    .unwrap_or_else(|e| {
        panic!(
            "{} failed: {:?}",
            "survival predict-noise fit should succeed", e
        )
    });

    let saved = SavedModel::load_from_path(&model_path)
        .unwrap_or_else(|e| panic!("{} failed: {:?}", "load fitted survival model", e));
    assert_eq!(saved.formula, "Surv(entry, exit, event) ~ 1");
    assert_eq!(saved.formula_noise.as_deref(), Some("1"));
    assert_eq!(saved.survival_likelihood.as_deref(), Some("location-scale"));
    assert!(saved.survival_beta_log_sigma.is_some());
    assert!(saved.resolved_termspec_noise.is_some());
    let fit_result = saved
        .fit_result
        .as_ref()
        .unwrap_or_else(|| panic!("{} failed", "saved fit_result"));
    let covariance = fit_result
        .beta_covariance()
        .or(fit_result.beta_covariance_corrected())
        .unwrap_or_else(|| panic!("{} failed", "saved survival fit covariance"));
    let expected_p = saved
        .survival_beta_time
        .as_ref()
        .unwrap_or_else(|| panic!("{} failed", "saved beta_time"))
        .len()
        + saved
            .survival_beta_threshold
            .as_ref()
            .expect("saved beta_threshold")
            .len()
        + saved
            .survival_beta_log_sigma
            .as_ref()
            .expect("saved beta_log_sigma")
            .len()
        + saved.beta_link_wiggle.as_ref().map_or(0, Vec::len);
    assert_eq!(covariance.nrows(), expected_p);
    assert_eq!(covariance.ncols(), expected_p);

    run_predict(PredictArgs {
        model: model_path,
        new_data: train_path,
        out: pred_path.clone(),
        offset_column: None,
        noise_offset_column: None,
        id_column: None,
        uncertainty: false,
        level: 0.95,
        covariance_mode: Some(InferenceCovarianceMode::SmoothingCorrected),
        mode: PredictModeArg::PosteriorMean,
        no_bias_correction: false,
    })
    .unwrap_or_else(|e| {
        panic!(
            "{} failed: {:?}",
            "saved survival posterior-mean predict should succeed", e
        )
    });

    let pred_text = fs::read_to_string(&pred_path)
        .unwrap_or_else(|e| panic!("{} failed: {:?}", "read survival prediction csv", e));
    // Exact header pin (the writer's column order is deterministic): a
    // substring check could pass with reordered/renamed/duplicated columns.
    let header = pred_text.lines().next().unwrap_or("");
    assert_eq!(
        header, "eta,survival_prob,failure_prob,risk_score,std_error,mean_lower,mean_upper",
        "posterior-mean survival prediction header drifted"
    );
}

#[test]
fn saved_prediction_runtime_rejects_location_scale_survival_payload_drift() {
    let blocks = vec![
        gam::estimate::FittedBlock {
            beta: array![0.1],
            role: BlockRole::Time,
            edf: 1.0,
            lambdas: Array1::zeros(0),
        },
        gam::estimate::FittedBlock {
            beta: array![0.2],
            role: BlockRole::Threshold,
            edf: 1.0,
            lambdas: Array1::zeros(0),
        },
        gam::estimate::FittedBlock {
            beta: array![-0.3],
            role: BlockRole::Scale,
            edf: 1.0,
            lambdas: Array1::zeros(0),
        },
    ];
    let fit_result = compact_saved_multiblock_fit_result(
        blocks,
        Array1::zeros(0),
        1.0,
        Some(Array2::<f64>::eye(3)),
        None,
        None,
        saved_fit_summary_fixture(),
    );
    let mut payload = test_payload(
        "Surv(entry, exit, event) ~ 1",
        ModelKind::Survival,
        FittedFamily::Survival {
            likelihood: LikelihoodSpec::new(
                ResponseFamily::RoystonParmar,
                InverseLink::Standard(StandardLink::Identity),
            ),
            survival_likelihood: Some("location-scale".to_string()),
            survival_distribution: Some(ResidualDistribution::Gaussian),
            frailty: gam::families::survival::lognormal_kernel::FrailtySpec::None,
        },
        "survival",
    );
    payload.fit_result = Some(fit_result.clone());
    payload.unified = Some(fit_result);
    payload.survival_likelihood = Some("location-scale".to_string());
    // Every location-scale survival artifact must carry the exact replay
    // structure: the field deliberately has no serde default, so a v11 artifact
    // states `null` for non-location-scale families or carries the complete
    // structure. Without it the runtime refuses on the MISSING STRUCTURE check
    // before it ever reaches the coefficient-drift refusal this test pins, and
    // the assertion reads a message about the wrong defect.
    payload.survival_location_scale_structure = Some(SavedSurvivalLocationScaleStructure {
        time_parameterization: SurvivalLocationScaleTimeParameterization::MonotoneWarp,
        threshold_time_basis: None,
        log_sigma_time_basis: None,
    });
    payload.survival_beta_time = Some(vec![9.9]);
    payload.survival_beta_threshold = Some(vec![0.2]);
    payload.survival_beta_log_sigma = Some(vec![-0.3]);
    let model = SavedModel::from_payload(payload);

    let err = model
        .saved_prediction_runtime()
        .expect_err("payload drift should be rejected");
    assert!(
        err.to_string()
            .contains("saved time coefficients disagree with fit_result")
    );
}

#[test]
fn cli_predict_noise_with_explicit_probit_keeps_binomial_probit_base_link() {
    let td = tempdir().unwrap_or_else(|e| panic!("{} failed: {:?}", "tempdir", e));
    let train_path = td.path().join("train.csv");
    let model_path = td.path().join("model.json");
    write_binomial_location_scale_train_csv(&train_path);

    run_fit(location_scale_fit_args(
        train_path,
        model_path.clone(),
        "y ~ x1 + link(type=probit)",
        "x2",
    ))
    .unwrap_or_else(|e| {
        panic!(
            "{} failed: {:?}",
            "explicit probit location-scale fit should succeed", e
        )
    });

    let saved = SavedModel::load_from_path(&model_path)
        .unwrap_or_else(|e| panic!("{} failed: {:?}", "load fitted model", e));
    assert_eq!(
        saved.link.as_ref(),
        Some(&InverseLink::Standard(StandardLink::Probit))
    );
    match &saved.family_state {
        FittedFamily::LocationScale {
            likelihood,
            base_link,
        } => {
            assert_eq!(*likelihood, LikelihoodSpec::binomial_probit());
            assert!(matches!(
                base_link.as_ref(),
                Some(InverseLink::Standard(StandardLink::Probit))
            ));
        }
        other => panic!("expected location-scale family state, got {other:?}"),
    }
}

#[test]
fn cli_bernoulli_marginal_slope_fit_saves_covariance_so_default_predict_succeeds() {
    let td = tempdir().unwrap_or_else(|e| panic!("{} failed: {:?}", "tempdir", e));
    let train_path = td.path().join("train.csv");
    let model_path = td.path().join("model.json");
    let pred_path = td.path().join("pred.csv");
    write_bernoulli_marginal_slope_train_csv(&train_path);

    run_fit(FitArgs {
        inference: true,
        expectile_tau: None,
        data: train_path.clone(),
        request: None,
        formula_positional: Some("y ~ x".to_string()),
        ctn_stage1: None,
        precision_hyperpriors: None,
        latent_coordinates: None,
        analytic_penalties: None,
        smooth_descriptors: None,
        predict_noise: None,
        slope_formula: Some("1".to_string()),
        z_column: Some("z".to_string()),
        weights_column: None,
        offset_column: None,
        noise_offset_column: None,
        frailty_kind: None,
        frailty_sd: None,
        hazard_loading: None,
        transformation_normal: false,
        firth: true,
        family: FamilyArg::Auto,
        negative_binomial_theta: None,
        // #2301: `survival_likelihood` is `Option<String>` defaulting to `None`,
        // and the single canonical default ("transformation") is resolved at the
        // `Surv(...)` seam rather than stored. `None` means unset; ANY `Some(mode)`
        // is an explicit request, and on a non-survival response the materializer
        // that reads it is never reached, so the knob would be silently dropped and
        // the requested model would degrade to an ordinary GAM (#1767). That is why
        // `reject_survival_only_config_for_nonsurvival` refuses it.
        //
        // This fixture is a NON-survival fit, so `Some("transformation")` here was
        // asking to be rejected. The guard is correct; the fixture predates it.
        survival_likelihood: None,
        survival_time_anchor: None,
        baseline_target: "linear".to_string(),
        baseline_scale: None,
        baseline_shape: None,
        baseline_rate: None,
        baseline_makeham: None,
        time_basis: "ispline".to_string(),
        time_degree: 3,
        time_num_internal_knots: 8,
        time_smooth_lambda: 1e-2,
        ridge_lambda: 1e-6,
        threshold_time_k: None,
        threshold_time_degree: 3,
        sigma_time_k: None,
        sigma_time_degree: 3,
        slope_time_k: None,
        slope_time_degree: 3,
        adaptive_regularization: false,
        scale_dimensions: false,
        precompute_conformal: true,
        persistent_warm_start_root: None,
        pilot_subsample_threshold: 0,
        out: Some(model_path.clone()),
    })
    .unwrap_or_else(|e| {
        panic!(
            "{} failed: {:?}",
            "bernoulli marginal-slope fit should succeed", e
        )
    });

    let saved = SavedModel::load_from_path(&model_path)
        .unwrap_or_else(|e| panic!("{} failed: {:?}", "load fitted model", e));
    let fit_result = saved
        .fit_result
        .as_ref()
        .unwrap_or_else(|| panic!("{} failed", "fit_result should be saved"));
    assert!(saved.payload().latent_z_normalization.is_some());
    assert!(
        fit_result.beta_covariance().is_some() || fit_result.beta_covariance_corrected().is_some(),
        "CLI marginal-slope fit should save covariance for default posterior-mean prediction",
    );

    run_predict(PredictArgs {
        model: model_path,
        new_data: train_path,
        out: pred_path.clone(),
        offset_column: None,
        noise_offset_column: None,
        id_column: None,
        uncertainty: false,
        level: 0.95,
        covariance_mode: Some(InferenceCovarianceMode::SmoothingCorrected),
        mode: PredictModeArg::PosteriorMean,
        no_bias_correction: false,
    })
    .unwrap_or_else(|e| {
        panic!(
            "{} failed: {:?}",
            "default posterior-mean marginal-slope predict should succeed", e
        )
    });

    let pred_text = fs::read_to_string(&pred_path)
        .unwrap_or_else(|e| panic!("{} failed: {:?}", "read prediction csv", e));
    let header = pred_text.lines().next().unwrap_or("");
    // The bernoulli marginal slope reports an EVENT probability -- its
    // likelihood is `binomial_probit()` and `mean_from_eta` is
    // `eta.mapv(normal_cdf)` -- so the CLI writes the survival-BINARY schema,
    // which keeps `mean` and adds the derived probabilities beside it. The
    // plain-survival header this used to expect belongs to a model whose mean
    // already IS a survival probability, and it drops the `mean` column that
    // `csv_mean_at` reads back elsewhere in this file.
    //
    // And `uncertainty: false` is point-only by contract:
    // `resolve_prediction_request` routes through
    // `PosteriorMeanOptions::point_only()` absent a confidence level, because
    // "passing a confidence level is the switch that populates SE/bounds"
    // (#2136). So the default predict carries no `std_error`/bands, and the
    // banded schema is asserted separately below.
    assert_eq!(
        header, "eta,mean,event_prob,failure_prob,survival_prob,risk_score",
        "posterior-mean marginal-slope prediction header drifted"
    );
}

#[test]
fn cli_bernoulli_marginal_slope_rejects_z_column_in_main_formula() {
    let td = tempdir().unwrap_or_else(|e| panic!("{} failed: {:?}", "tempdir", e));
    let train_path = td.path().join("train.csv");
    write_bernoulli_marginal_slope_train_csv(&train_path);

    let err = run_fit(FitArgs {
        inference: true,
        expectile_tau: None,
        data: train_path,
        request: None,
        formula_positional: Some("y ~ x + z".to_string()),
        ctn_stage1: None,
        precision_hyperpriors: None,
        latent_coordinates: None,
        analytic_penalties: None,
        smooth_descriptors: None,
        predict_noise: None,
        slope_formula: Some("1".to_string()),
        z_column: Some("z".to_string()),
        weights_column: None,
        offset_column: None,
        noise_offset_column: None,
        frailty_kind: None,
        frailty_sd: None,
        hazard_loading: None,
        transformation_normal: false,
        firth: false,
        family: FamilyArg::Auto,
        negative_binomial_theta: None,
        survival_likelihood: Some("transformation".to_string()),
        survival_time_anchor: None,
        baseline_target: "linear".to_string(),
        baseline_scale: None,
        baseline_shape: None,
        baseline_rate: None,
        baseline_makeham: None,
        time_basis: "ispline".to_string(),
        time_degree: 3,
        time_num_internal_knots: 8,
        time_smooth_lambda: 1e-2,
        ridge_lambda: 1e-6,
        threshold_time_k: None,
        threshold_time_degree: 3,
        sigma_time_k: None,
        sigma_time_degree: 3,
        slope_time_k: None,
        slope_time_degree: 3,
        adaptive_regularization: false,
        scale_dimensions: false,
        precompute_conformal: true,
        persistent_warm_start_root: None,
        pilot_subsample_threshold: 0,
        out: None,
    })
    .expect_err("main formula should reject z-column reuse");

    assert!(err.contains("bernoulli marginal-slope reserves z column 'z'"));
    assert!(err.contains("main formula"));
}

#[test]
fn cli_bernoulli_marginal_slope_rejects_z_column_in_slope_formula() {
    let td = tempdir().unwrap_or_else(|e| panic!("{} failed: {:?}", "tempdir", e));
    let train_path = td.path().join("train.csv");
    write_bernoulli_marginal_slope_train_csv(&train_path);

    let err = run_fit(FitArgs {
        inference: true,
        expectile_tau: None,
        data: train_path,
        request: None,
        formula_positional: Some("y ~ x".to_string()),
        ctn_stage1: None,
        precision_hyperpriors: None,
        latent_coordinates: None,
        analytic_penalties: None,
        smooth_descriptors: None,
        predict_noise: None,
        slope_formula: Some("1 + s(z, type=duchon, centers=6)".to_string()),
        z_column: Some("z".to_string()),
        weights_column: None,
        offset_column: None,
        noise_offset_column: None,
        frailty_kind: None,
        frailty_sd: None,
        hazard_loading: None,
        transformation_normal: false,
        firth: false,
        family: FamilyArg::Auto,
        negative_binomial_theta: None,
        survival_likelihood: Some("transformation".to_string()),
        survival_time_anchor: None,
        baseline_target: "linear".to_string(),
        baseline_scale: None,
        baseline_shape: None,
        baseline_rate: None,
        baseline_makeham: None,
        time_basis: "ispline".to_string(),
        time_degree: 3,
        time_num_internal_knots: 8,
        time_smooth_lambda: 1e-2,
        ridge_lambda: 1e-6,
        threshold_time_k: None,
        threshold_time_degree: 3,
        sigma_time_k: None,
        sigma_time_degree: 3,
        slope_time_k: None,
        slope_time_degree: 3,
        adaptive_regularization: false,
        scale_dimensions: false,
        precompute_conformal: true,
        persistent_warm_start_root: None,
        pilot_subsample_threshold: 0,
        out: None,
    })
    .expect_err("slope formula should reject z-column reuse");

    assert!(err.contains("bernoulli marginal-slope reserves z column 'z'"));
    assert!(err.contains("--slope-formula"));
}

#[test]
fn saved_bernoulli_marginal_slope_replays_main_and_slope_deviation_runtimes() {
    let saved_runtime = || SavedCompiledFlexBlock {
        kernel: exact_kernel::ANCHORED_DEVIATION_KERNEL.to_string(),
        breakpoints: vec![-1.0, 1.0],
        basis_dim: 1,
        span_c0: vec![vec![0.0]],
        span_c1: vec![vec![0.0]],
        span_c2: vec![vec![0.0]],
        span_c3: vec![vec![0.0]],
        anchor_correction: None,
        anchor_components: Vec::new(),
    };
    let fit_result = compact_saved_multiblock_fit_result(
        vec![
            FittedBlock {
                beta: array![0.0],
                role: BlockRole::Mean,
                edf: 0.0,
                lambdas: Array1::zeros(0),
            },
            FittedBlock {
                beta: array![0.0],
                role: BlockRole::Scale,
                edf: 0.0,
                lambdas: Array1::zeros(0),
            },
            FittedBlock {
                beta: array![0.0],
                role: BlockRole::Scale,
                edf: 0.0,
                lambdas: Array1::zeros(0),
            },
            FittedBlock {
                beta: array![0.0],
                role: BlockRole::LinkWiggle,
                edf: 0.0,
                lambdas: Array1::zeros(0),
            },
        ],
        Array1::zeros(0),
        1.0,
        None,
        None,
        None,
        SavedFitSummary {
            likelihood_family: Some(LikelihoodSpec::new(
                ResponseFamily::Binomial,
                InverseLink::Standard(StandardLink::Probit),
            )),
            // Binomial has NO free dispersion: phi is identically 1, so the resolved
            // scale is Unit. `Unspecified` is not "the family decides" -- it means the
            // caller never said, and `LikelihoodSpec::resolved_scale` refuses it for
            // binomial/Poisson because the saved-model contract requires "the
            // response-scale summary paired with explicit likelihood-scale metadata for
            // non-Gaussian models" (core_saved_fit_result). Leaving it Unspecified made
            // these fixtures build a saved model the loader is right to reject:
            //   invalid resolved likelihood scale: family binomial requires exact
            //   FixedDispersion { phi: 1.0 } metadata, got Unspecified
            // The validator is not loosened -- an Unspecified reaching it still means a
            // real caller dropped the metadata, which is exactly what it should catch.
            likelihood_scale: LikelihoodScaleMetadata::FixedDispersion { phi: 1.0 },
            log_likelihood_normalization: LogLikelihoodNormalization::UserProvided,
            ..saved_fit_summary_fixture()
        },
    );
    let mut payload = FittedModelPayload::new(
        MODEL_PAYLOAD_VERSION,
        "y ~ x + link(type=probit) + linkwiggle(degree=3, internal_knots=4, penalty_order=\"1\")"
            .to_string(),
        ModelKind::MarginalSlope,
        FittedFamily::MarginalSlope {
            likelihood: LikelihoodSpec::new(
                ResponseFamily::Binomial,
                InverseLink::Standard(StandardLink::Probit),
            ),
            base_link: InverseLink::Standard(StandardLink::Probit),
            frailty: gam::families::survival::lognormal_kernel::FrailtySpec::None,
        },
        "bernoulli-marginal-slope".to_string(),
    );
    payload.unified = Some(fit_result.clone());
    payload.fit_result = Some(fit_result);
    payload.data_schema = Some(DataSchema { columns: vec![] });
    payload.set_training_feature_metadata(vec![], vec![]);
    payload.resolved_termspec = Some(empty_termspec());
    payload.resolved_slopespec = Some(empty_termspec());
    payload.slope_formula =
        Some("1 + linkwiggle(degree=3, internal_knots=4, penalty_order=\"2\")".to_string());
    payload.z_column = Some("z".to_string());
    payload.latent_z_normalization = Some(SavedLatentZNormalization { mean: 0.0, sd: 1.0 });
    payload.marginal_baseline = Some(0.0);
    payload.baseline_slope = Some(0.0);
    payload.link = Some(InverseLink::Standard(StandardLink::Probit));
    payload.score_warp_runtime = Some(saved_runtime());
    payload.link_deviation_runtime = Some(saved_runtime());

    let saved = SavedModel::from_payload(payload);
    let runtime = saved.saved_prediction_runtime().unwrap_or_else(|e| {
        panic!(
            "{} failed: {:?}",
            "saved marginal-slope runtime should replay", e
        )
    });
    assert!(
        runtime.score_warp.is_some(),
        "slope-formula linkwiggle should persist score-warp runtime"
    );
    assert!(
        runtime.link_deviation.is_some(),
        "main-formula linkwiggle should persist link-deviation runtime"
    );
    assert_eq!(
        saved
            .resolved_inverse_link()
            .unwrap_or_else(|e| panic!("{} failed: {:?}", "resolved inverse link", e)),
        Some(InverseLink::Standard(StandardLink::Probit))
    );
}

#[test]
fn nonlinear_saved_model_with_hessian_only_remains_persistable_and_predictable() {
    let td = tempdir().unwrap_or_else(|e| panic!("{} failed: {:?}", "tempdir", e));
    let model_path = td.path().join("model.json");
    let fit_result = gam::estimate::UnifiedFitResult::try_from_parts(UnifiedFitResultParts {
        blocks: vec![FittedBlock {
            beta: array![0.25],
            role: BlockRole::Mean,
            edf: 0.0,
            lambdas: Array1::zeros(0),
        }],
        training_sample_size: 12,
        log_lambdas: Array1::zeros(0),
        lambdas: Array1::zeros(0),
        likelihood_family: Some(LikelihoodSpec::new(
            ResponseFamily::Binomial,
            InverseLink::Standard(StandardLink::Logit),
        )),
        likelihood_scale: LikelihoodScaleMetadata::FixedDispersion { phi: 1.0 },
        log_likelihood_normalization: LogLikelihoodNormalization::UserProvided,
        log_likelihood: -1.0,
        deviance: 2.0,
        reml_score: Some(0.0),
        stable_penalty_term: 0.0,
        penalized_objective: Some(1.0),
        used_device: false,
        outer_iterations: 0,
        outer_converged: true,
        outer_gradient_norm: None,
        standard_deviation: 1.0,
        covariance_conditional: None,
        covariance_corrected: None,
        inference: None,
        fitted_link: FittedLinkState::Standard(None),
        geometry: Some(FitGeometry {
            coefficient_gauge: Gauge::identity(&[1]),
            penalized_hessian: array![[2.0]].into(),
            constrained_posterior: None,
            working: None,
        }),
        block_states: Vec::new(),
        pirls_status: gam::pirls::PirlsStatus::Converged,
        max_abs_eta: 0.0,
        constraint_kkt: None,
        artifacts: Default::default(),
        inner_cycles: 0,
    })
    .unwrap_or_else(|e| panic!("{} failed: {:?}", "construct hessian-only fit result", e));

    let mut payload = FittedModelPayload::new(
        MODEL_PAYLOAD_VERSION,
        "y ~ x".to_string(),
        ModelKind::Standard,
        FittedFamily::Standard {
            likelihood: LikelihoodSpec::new(
                ResponseFamily::Binomial,
                InverseLink::Standard(StandardLink::Logit),
            ),
            link: Some(StandardLink::Logit),
            latent_cloglog_state: None,
            mixture_state: None,
            sas_state: None,
        },
        "binomial-logit".to_string(),
    );
    payload.fit_result = Some(fit_result.clone());
    payload.unified = Some(fit_result);
    payload.data_schema = Some(DataSchema {
        columns: vec![
            SchemaColumn {
                name: "x".to_string(),
                kind: ColumnKindTag::Continuous,
                levels: Vec::new(),
            },
            SchemaColumn {
                name: "y".to_string(),
                kind: ColumnKindTag::Binary,
                levels: Vec::new(),
            },
        ],
    });
    payload.set_training_feature_metadata(
        vec!["x".to_string(), "y".to_string()],
        vec![(0.0, 1.0), (0.0, 1.0)],
    );
    payload.resolved_termspec = Some(empty_termspec());

    let model = SavedModel::from_payload(payload);
    model.save_to_path(&model_path).unwrap_or_else(|e| {
        panic!(
            "{} failed: {:?}",
            "hessian-only nonlinear model should save", e
        )
    });
    let loaded = SavedModel::load_from_path(&model_path)
        .unwrap_or_else(|e| panic!("{} failed: {:?}", "reload hessian-only model", e));
    let covariance = covariance_from_model(&loaded, InferenceCovarianceMode::Conditional)
        .unwrap_or_else(|e| {
            panic!(
                "{} failed: {:?}",
                "recover covariance from saved penalized Hessian", e
            )
        });
    assert_eq!(covariance.dim(), (1, 1));
    assert!((covariance[[0, 0]] - 0.5).abs() < 1e-12);
}

/// #2385: the CLI's Hessian-only covariance fallback must report the covariance
/// of the **truncated** posterior, not the ambient Gaussian's.
///
/// The fixture makes the inequality maximally ACTIVE rather than merely present:
/// one coefficient, `H = [[2]]` so the ambient `Σ = φ·H⁻¹ = 1/2`, and the single
/// row `β ≥ 0.25` placed exactly at the ambient centre `β_unc = 0.25`. The
/// truncated posterior is then the half-normal on `[0.25, ∞)` built from
/// `N(0.25, 1/2)`, whose variance is the closed form `Σ·(1 − 2/π)` — an
/// analytic reference this test does not obtain from the code under test.
///
/// This is the case an interval on an active face gets wrong in the direction
/// that matters: the ambient `1/2` is 2.75× the truncated variance, so an
/// interval built from it is far too wide and its lower endpoint crosses the
/// very wall the fit was constrained by. Before the fix
/// `covariance_from_model` built its backend with `from_factorized_hessian`,
/// which passes no correction, and returned exactly the ambient `0.5`.
#[test]
fn hessian_only_saved_model_reports_the_truncated_covariance_on_an_active_face() {
    use gam::pirls::LinearInequalityConstraints;
    use gam::solver::constrained_posterior::{
        ConstrainedPosteriorGeometry, constrained_posterior_correction_from_covariance,
    };

    let td = tempdir().unwrap_or_else(|e| panic!("{} failed: {:?}", "tempdir", e));
    let model_path = td.path().join("model.json");

    // Σ = φ·H⁻¹ with φ = 1 (Binomial: the IRLS weight already carries the
    // dispersion), so the ambient posterior variance is exactly 1/2.
    let ambient = array![[0.5]];
    let center = array![0.25];
    let constraints = LinearInequalityConstraints::new(array![[1.0]], array![0.25])
        .unwrap_or_else(|e| panic!("{} failed: {:?}", "build the active constraint row", e));
    let correction =
        constrained_posterior_correction_from_covariance(&ambient, &center, &constraints)
            .unwrap_or_else(|e| panic!("{} failed: {:?}", "build the truncation correction", e))
            .unwrap_or_else(|| {
                panic!(
                    "a row whose wall sits exactly at the ambient centre must be retained; \
             a `None` correction here would make this test pass by absence"
                )
            });
    // Guard the fixture itself: the face is retained AND it removes variance.
    assert_eq!(
        correction.rows,
        vec![0],
        "the single constraint row must be the retained face"
    );
    assert!(
        correction.removed_normal_variance[[0, 0]] > 0.0,
        "an active face must remove normal-coordinate variance, got {:?}",
        correction.removed_normal_variance
    );

    let fit_result = gam::estimate::UnifiedFitResult::try_from_parts(UnifiedFitResultParts {
        blocks: vec![FittedBlock {
            beta: array![0.25],
            role: BlockRole::Mean,
            edf: 0.0,
            lambdas: Array1::zeros(0),
        }],
        training_sample_size: 12,
        log_lambdas: Array1::zeros(0),
        lambdas: Array1::zeros(0),
        likelihood_family: Some(LikelihoodSpec::new(
            ResponseFamily::Binomial,
            InverseLink::Standard(StandardLink::Logit),
        )),
        likelihood_scale: LikelihoodScaleMetadata::FixedDispersion { phi: 1.0 },
        log_likelihood_normalization: LogLikelihoodNormalization::UserProvided,
        log_likelihood: -1.0,
        deviance: 2.0,
        reml_score: Some(0.0),
        stable_penalty_term: 0.0,
        penalized_objective: Some(1.0),
        used_device: false,
        outer_iterations: 0,
        outer_converged: true,
        outer_gradient_norm: None,
        standard_deviation: 1.0,
        covariance_conditional: None,
        covariance_corrected: None,
        inference: None,
        fitted_link: FittedLinkState::Standard(None),
        geometry: Some(FitGeometry {
            coefficient_gauge: Gauge::identity(&[1]),
            penalized_hessian: array![[2.0]].into(),
            constrained_posterior: Some(ConstrainedPosteriorGeometry::with_moments(
                constraints,
                array![0.25],
                center,
                Some(correction),
            )),
            working: None,
        }),
        block_states: Vec::new(),
        pirls_status: gam::pirls::PirlsStatus::Converged,
        max_abs_eta: 0.0,
        constraint_kkt: None,
        artifacts: Default::default(),
        inner_cycles: 0,
    })
    .unwrap_or_else(|e| {
        panic!(
            "{} failed: {:?}",
            "construct constrained hessian-only fit result", e
        )
    });

    let mut payload = FittedModelPayload::new(
        MODEL_PAYLOAD_VERSION,
        "y ~ x".to_string(),
        ModelKind::Standard,
        FittedFamily::Standard {
            likelihood: LikelihoodSpec::new(
                ResponseFamily::Binomial,
                InverseLink::Standard(StandardLink::Logit),
            ),
            link: Some(StandardLink::Logit),
            latent_cloglog_state: None,
            mixture_state: None,
            sas_state: None,
        },
        "binomial-logit".to_string(),
    );
    payload.fit_result = Some(fit_result.clone());
    payload.unified = Some(fit_result);
    payload.data_schema = Some(DataSchema {
        columns: vec![
            SchemaColumn {
                name: "x".to_string(),
                kind: ColumnKindTag::Continuous,
                levels: Vec::new(),
            },
            SchemaColumn {
                name: "y".to_string(),
                kind: ColumnKindTag::Binary,
                levels: Vec::new(),
            },
        ],
    });
    payload.set_training_feature_metadata(
        vec!["x".to_string(), "y".to_string()],
        vec![(0.0, 1.0), (0.0, 1.0)],
    );
    payload.resolved_termspec = Some(empty_termspec());

    let model = SavedModel::from_payload(payload);
    model.save_to_path(&model_path).unwrap_or_else(|e| {
        panic!(
            "{} failed: {:?}",
            "constrained hessian-only model should save", e
        )
    });
    let loaded = SavedModel::load_from_path(&model_path).unwrap_or_else(|e| {
        panic!(
            "{} failed: {:?}",
            "reload constrained hessian-only model", e
        )
    });
    let covariance = covariance_from_model(&loaded, InferenceCovarianceMode::Conditional)
        .unwrap_or_else(|e| {
            panic!(
                "{} failed: {:?}",
                "recover truncated covariance from saved penalized Hessian", e
            )
        });
    assert_eq!(covariance.dim(), (1, 1));

    // Closed form for the half-normal on [0.25, ∞) obtained from N(0.25, 1/2).
    let expected = 0.5 * (1.0 - 2.0 / std::f64::consts::PI);
    let relative = (covariance[[0, 0]] - expected).abs() / expected;
    assert!(
        relative < 2e-3,
        "truncated posterior variance on an active face should be {expected:.9e} \
         (the ambient 0.5 is 2.75x too wide and puts the interval's lower endpoint \
         through the constraint wall), got {:.9e} (relative {relative:.3e})",
        covariance[[0, 0]]
    );
    assert!(
        covariance[[0, 0]] < 0.4,
        "the reported covariance must be strictly narrower than the ambient 0.5; \
         got {:.9e}, which is the untruncated Gaussian",
        covariance[[0, 0]]
    );
}

#[test]
fn cli_fit_saves_covariance_so_default_binomial_predict_succeeds() {
    let td = tempdir().unwrap_or_else(|e| panic!("{} failed: {:?}", "tempdir", e));
    let train_path = td.path().join("train.csv");
    let model_path = td.path().join("model.json");
    let pred_path = td.path().join("pred.csv");

    fs::write(
            &train_path,
            "x1,x2,y\n-1.0,-0.5,0\n-0.8,0.2,0\n-0.3,-0.1,0\n0.1,0.0,0\n0.4,0.2,1\n0.8,0.5,1\n1.1,0.9,1\n1.4,1.0,1\n",
        )
        .unwrap_or_else(|e| panic!("{} failed: {:?}", "write training csv", e));

    let fit_args = FitArgs {
        inference: true,
        expectile_tau: None,
        data: train_path.clone(),
        request: None,
        formula_positional: Some("y ~ x1 + x2".to_string()),
        ctn_stage1: None,
        precision_hyperpriors: None,
        latent_coordinates: None,
        analytic_penalties: None,
        smooth_descriptors: None,
        predict_noise: None,
        slope_formula: None,
        z_column: None,
        weights_column: None,
        offset_column: None,
        noise_offset_column: None,
        frailty_kind: None,
        frailty_sd: None,
        hazard_loading: None,
        transformation_normal: false,
        firth: false,
        family: FamilyArg::Auto,
        negative_binomial_theta: None,
        // #2301: `survival_likelihood` is `Option<String>` defaulting to `None`,
        // and the single canonical default ("transformation") is resolved at the
        // `Surv(...)` seam rather than stored. `None` means unset; ANY `Some(mode)`
        // is an explicit request, and on a non-survival response the materializer
        // that reads it is never reached, so the knob would be silently dropped and
        // the requested model would degrade to an ordinary GAM (#1767). That is why
        // `reject_survival_only_config_for_nonsurvival` refuses it.
        //
        // This fixture is a NON-survival fit, so `Some("transformation")` here was
        // asking to be rejected. The guard is correct; the fixture predates it.
        survival_likelihood: None,
        survival_time_anchor: None,
        baseline_target: "linear".to_string(),
        baseline_scale: None,
        baseline_shape: None,
        baseline_rate: None,
        baseline_makeham: None,
        time_basis: "ispline".to_string(),
        time_degree: 3,
        time_num_internal_knots: 8,
        time_smooth_lambda: 1e-2,
        ridge_lambda: 1e-6,
        threshold_time_k: None,
        threshold_time_degree: 3,
        sigma_time_k: None,
        sigma_time_degree: 3,
        slope_time_k: None,
        slope_time_degree: 3,
        adaptive_regularization: true,
        scale_dimensions: false,
        precompute_conformal: true,
        persistent_warm_start_root: None,
        pilot_subsample_threshold: 0,
        out: Some(model_path.clone()),
    };
    run_fit(fit_args).unwrap_or_else(|e| panic!("{} failed: {:?}", "fit should succeed", e));

    let saved = SavedModel::load_from_path(&model_path)
        .unwrap_or_else(|e| panic!("{} failed: {:?}", "load fitted model", e));
    let fit_result = saved
        .fit_result
        .as_ref()
        .unwrap_or_else(|| panic!("{} failed", "fit_result should be saved"));
    assert!(
        fit_result.beta_covariance().is_some() || fit_result.beta_covariance_corrected().is_some(),
        "CLI fit should save covariance for default posterior-mean prediction",
    );

    let predict_args = PredictArgs {
        model: model_path.clone(),
        new_data: train_path.clone(),
        out: pred_path.clone(),
        offset_column: None,
        noise_offset_column: None,
        id_column: None,
        uncertainty: false,
        level: 0.95,
        covariance_mode: Some(InferenceCovarianceMode::SmoothingCorrected),
        mode: PredictModeArg::PosteriorMean,
        no_bias_correction: false,
    };
    run_predict(predict_args).unwrap_or_else(|e| {
        panic!(
            "{} failed: {:?}",
            "default posterior-mean predict should succeed", e
        )
    });

    // The DEFAULT predict (`uncertainty: false`) is point-only, and that is the
    // documented contract rather than an omission: `resolve_prediction_request`
    // routes a curved link through `PosteriorMeanOptions::point_only()` and says
    // so in as many words -- "passing a confidence level is the switch that
    // populates SE/bounds" (#2136). The default therefore emits `eta,mean`, and
    // demanding `std_error` from it (as this test used to) asserts against the
    // design; it failed with `missing std_error column: eta,mean`.
    let pred_text = fs::read_to_string(&pred_path)
        .unwrap_or_else(|e| panic!("{} failed: {:?}", "read prediction csv", e));
    let header = pred_text.lines().next().unwrap_or("");
    assert!(
        header.contains("mean"),
        "default posterior-mean prediction must emit the point column: {header}"
    );

    // What saving the covariance actually BUYS is the other half of this test's
    // name: the bands become computable on request. Asking for them is what
    // proves the persisted covariance is real and reachable -- a point-only
    // predict would pass identically with no covariance saved at all, so the
    // column check belongs behind `uncertainty: true`.
    let band_path = pred_path.with_extension("band.csv");
    let band_args = PredictArgs {
        model: model_path,
        new_data: train_path,
        out: band_path.clone(),
        offset_column: None,
        noise_offset_column: None,
        id_column: None,
        uncertainty: true,
        level: 0.95,
        covariance_mode: Some(InferenceCovarianceMode::SmoothingCorrected),
        mode: PredictModeArg::PosteriorMean,
        no_bias_correction: false,
    };
    run_predict(band_args).unwrap_or_else(|e| {
        panic!(
            "{} failed: {:?}",
            "posterior-mean predict with uncertainty on a saved covariance", e
        )
    });
    let band_text = fs::read_to_string(&band_path)
        .unwrap_or_else(|e| panic!("{} failed: {:?}", "read band csv", e));
    let band_header = band_text.lines().next().unwrap_or("");
    for required in [
        "posterior_mean",
        "posterior_mean_standard_error",
        "posterior_mean_lower",
        "posterior_mean_upper",
    ] {
        assert!(
            band_header.contains(required),
            "posterior-mean prediction with uncertainty is missing {required}: {band_header}"
        );
    }
}

/// Build a standard (non-survival, non-location-scale) binomial `FitArgs` for the
/// given formula, writing the model to `out`. Shared by the parameterized-link
/// cap-guard regression tests below.
fn binomial_link_fit_args(data: PathBuf, out: PathBuf, formula: &str) -> FitArgs {
    FitArgs {
        inference: true,
        expectile_tau: None,
        data,
        request: None,
        formula_positional: Some(formula.to_string()),
        ctn_stage1: None,
        precision_hyperpriors: None,
        latent_coordinates: None,
        analytic_penalties: None,
        smooth_descriptors: None,
        predict_noise: None,
        slope_formula: None,
        z_column: None,
        weights_column: None,
        offset_column: None,
        noise_offset_column: None,
        frailty_kind: None,
        frailty_sd: None,
        hazard_loading: None,
        transformation_normal: false,
        firth: false,
        family: FamilyArg::Auto,
        negative_binomial_theta: None,
        // #2301: `survival_likelihood` is `Option<String>` defaulting to `None`,
        // and the single canonical default ("transformation") is resolved at the
        // `Surv(...)` seam rather than stored. `None` means unset; ANY `Some(mode)`
        // is an explicit request, and on a non-survival response the materializer
        // that reads it is never reached, so the knob would be silently dropped and
        // the requested model would degrade to an ordinary GAM (#1767). That is why
        // `reject_survival_only_config_for_nonsurvival` refuses it.
        //
        // This fixture is a NON-survival fit, so `Some("transformation")` here was
        // asking to be rejected. The guard is correct; the fixture predates it.
        survival_likelihood: None,
        survival_time_anchor: None,
        baseline_target: "linear".to_string(),
        baseline_scale: None,
        baseline_shape: None,
        baseline_rate: None,
        baseline_makeham: None,
        time_basis: "ispline".to_string(),
        time_degree: 3,
        time_num_internal_knots: 8,
        time_smooth_lambda: 1e-2,
        ridge_lambda: 1e-6,
        threshold_time_k: None,
        threshold_time_degree: 3,
        sigma_time_k: None,
        sigma_time_degree: 3,
        slope_time_k: None,
        slope_time_degree: 3,
        adaptive_regularization: false,
        scale_dimensions: false,
        precompute_conformal: true,
        persistent_warm_start_root: None,
        pilot_subsample_threshold: 0,
        out: Some(out),
    }
}

/// Regression for #1571: a binomial `s(x) + link(type=sas)` fit through the CLI
/// fit path must not abort with "Lambda count mismatch".
///
/// The sinh-arcsinh link optimizes its two parameters (ε, log δ) jointly with the
/// smoothing log-λ in one augmented outer vector θ = [ρ_smooth (k) | ε, log δ].
/// On realistic data the outer-aware inner-PIRLS schedule lifts its iteration cap
/// during the search, which fires the post-convergence cap guard
/// (`run_outer_inner_cap_guard`). That guard used to forward the FULL augmented θ
/// to `compute_cost`, which exponentiates the whole vector into the penalty λ
/// vector — handing `k + 2` "lambdas" to the `k`-penalty reparameterizer and
/// faulting. The fix routes θ through `apply_link_theta` first (installing the
/// converged link state and slicing the smoothing-only ρ block), exactly as the
/// outer evaluator and the accept-fit already do.
///
/// Data is the committed n≈2000 binomial fixture, large enough to drive the
/// schedule into lifting the cap so the guard actually runs.
#[test]
fn cli_binomial_sas_link_fit_survives_outer_inner_cap_guard() {
    let fixture = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/bug_hunt_sas_link_cap_guard.csv");
    assert!(
        fixture.exists(),
        "missing committed fixture: {}",
        fixture.display()
    );
    let td = tempdir().unwrap_or_else(|e| panic!("{} failed: {:?}", "tempdir", e));
    let model_path = td.path().join("model.json");

    let fit_args = binomial_link_fit_args(fixture, model_path.clone(), "y ~ s(x) + link(type=sas)");
    let result = run_fit(fit_args);
    if let Err(e) = &result {
        let msg = format!("{e:?}");
        assert!(
            !msg.contains("Lambda count mismatch"),
            "SAS cap-guard regression (#1571): augmented θ leaked into compute_cost: {msg}"
        );
        panic!("binomial s(x) + link(type=sas) fit should succeed, got: {msg}");
    }

    // The fit must have persisted a usable model.
    let saved = SavedModel::load_from_path(&model_path)
        .unwrap_or_else(|e| panic!("{} failed: {:?}", "load fitted SAS model", e));
    assert!(
        saved.fit_result.is_some(),
        "SAS fit should persist a fit result"
    );
}

/// Regression for #1571 from a different angle: a PARAMETRIC-ONLY binomial
/// `x + link(type=beta-logistic)` fit (0 penalty blocks, 2 link parameters) must
/// also survive the shared `MixtureSas` cap guard. Here the augmented θ is the
/// pure link block [ε, log δ] with k = 0, so the pre-fix guard faulted with
/// "expected 0 lambdas for 0 penalties, got 2" — the same leak, the opposite
/// extreme (no smoothing block at all), through a different parameterized link.
#[test]
fn cli_binomial_beta_logistic_parametric_fit_survives_outer_inner_cap_guard() {
    let fixture = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/bug_hunt_sas_link_cap_guard.csv");
    assert!(
        fixture.exists(),
        "missing committed fixture: {}",
        fixture.display()
    );
    let td = tempdir().unwrap_or_else(|e| panic!("{} failed: {:?}", "tempdir", e));
    let model_path = td.path().join("model.json");

    let fit_args = binomial_link_fit_args(
        fixture,
        model_path.clone(),
        "y ~ x + link(type=beta-logistic)",
    );
    let result = run_fit(fit_args);
    if let Err(e) = &result {
        let msg = format!("{e:?}");
        assert!(
            !msg.contains("Lambda count mismatch"),
            "beta-logistic cap-guard regression (#1571): augmented θ leaked into compute_cost: {msg}"
        );
        panic!("binomial x + link(type=beta-logistic) fit should succeed, got: {msg}");
    }

    let saved = SavedModel::load_from_path(&model_path)
        .unwrap_or_else(|e| panic!("{} failed: {:?}", "load fitted beta-logistic model", e));
    assert!(
        saved.fit_result.is_some(),
        "beta-logistic fit should persist a fit result"
    );
}

#[test]
fn cli_firth_fit_saves_covariance_so_default_binomial_predict_succeeds() {
    let td = tempdir().unwrap_or_else(|e| panic!("{} failed: {:?}", "tempdir", e));
    let train_path = td.path().join("train.csv");
    let model_path = td.path().join("model.json");
    let pred_path = td.path().join("pred.csv");

    fs::write(
            &train_path,
            "x1,x2,y\n-1.0,-0.5,0\n-0.8,0.2,0\n-0.3,-0.1,0\n0.1,0.0,0\n0.4,0.2,1\n0.8,0.5,1\n1.1,0.9,1\n1.4,1.0,1\n",
        )
        .unwrap_or_else(|e| panic!("{} failed: {:?}", "write training csv", e));

    let fit_args = FitArgs {
        inference: true,
        expectile_tau: None,
        data: train_path.clone(),
        // Firth bias-reduction is only implemented for the binomial logit
        // likelihood. The auto-detect default for binary responses is
        // probit (96df9f5/b0590db), so the formula must request logit
        // explicitly for this CLI Firth-fit smoke to exercise the actual
        // Firth code path.
        request: None,
        formula_positional: Some("y ~ x1 + x2 + link(type=logit)".to_string()),
        ctn_stage1: None,
        precision_hyperpriors: None,
        latent_coordinates: None,
        analytic_penalties: None,
        smooth_descriptors: None,
        predict_noise: None,
        slope_formula: None,
        z_column: None,
        weights_column: None,
        offset_column: None,
        noise_offset_column: None,
        frailty_kind: None,
        frailty_sd: None,
        hazard_loading: None,
        transformation_normal: false,
        firth: true,
        family: FamilyArg::Auto,
        negative_binomial_theta: None,
        // #2301, same as the three sibling fixtures: this is a NON-survival
        // binomial Firth fit, so an explicit survival_likelihood is a knob the
        // materializer that reads it never sees. It would be dropped silently and
        // degrade the requested model to an ordinary GAM (#1767), which is why
        // reject_survival_only_config_for_nonsurvival refuses it.
        survival_likelihood: None,
        survival_time_anchor: None,
        baseline_target: "linear".to_string(),
        baseline_scale: None,
        baseline_shape: None,
        baseline_rate: None,
        baseline_makeham: None,
        time_basis: "ispline".to_string(),
        time_degree: 3,
        time_num_internal_knots: 8,
        time_smooth_lambda: 1e-2,
        ridge_lambda: 1e-6,
        threshold_time_k: None,
        threshold_time_degree: 3,
        sigma_time_k: None,
        sigma_time_degree: 3,
        slope_time_k: None,
        slope_time_degree: 3,
        adaptive_regularization: false,
        scale_dimensions: false,
        precompute_conformal: true,
        persistent_warm_start_root: None,
        pilot_subsample_threshold: 0,
        out: Some(model_path.clone()),
    };
    run_fit(fit_args).unwrap_or_else(|e| panic!("{} failed: {:?}", "Firth fit should succeed", e));

    let saved = SavedModel::load_from_path(&model_path)
        .unwrap_or_else(|e| panic!("{} failed: {:?}", "load fitted model", e));
    let fit_result = saved
        .fit_result
        .as_ref()
        .unwrap_or_else(|| panic!("{} failed", "fit_result should be saved"));
    assert!(
        fit_result.beta_covariance().is_some() || fit_result.beta_covariance_corrected().is_some(),
        "CLI Firth fit should save covariance for default posterior-mean prediction",
    );

    let predict_args = PredictArgs {
        model: model_path.clone(),
        new_data: train_path.clone(),
        out: pred_path.clone(),
        offset_column: None,
        noise_offset_column: None,
        id_column: None,
        uncertainty: false,
        level: 0.95,
        covariance_mode: Some(InferenceCovarianceMode::SmoothingCorrected),
        mode: PredictModeArg::PosteriorMean,
        no_bias_correction: false,
    };
    run_predict(predict_args).unwrap_or_else(|e| {
        panic!(
            "{} failed: {:?}",
            "default posterior-mean predict should succeed after Firth fit", e
        )
    });

    // The DEFAULT predict (`uncertainty: false`) is point-only, and that is the
    // documented contract rather than an omission: `resolve_prediction_request`
    // routes a curved link through `PosteriorMeanOptions::point_only()` and says
    // so in as many words -- "passing a confidence level is the switch that
    // populates SE/bounds" (#2136). The default therefore emits `eta,mean`, and
    // demanding `std_error` from it (as this test used to) asserts against the
    // design; it failed with `missing std_error column: eta,mean`.
    let pred_text = fs::read_to_string(&pred_path)
        .unwrap_or_else(|e| panic!("{} failed: {:?}", "read prediction csv", e));
    let header = pred_text.lines().next().unwrap_or("");
    assert!(
        header.contains("mean"),
        "default posterior-mean prediction must emit the point column: {header}"
    );

    // What saving the covariance actually BUYS is the other half of this test's
    // name: the bands become computable on request. Asking for them is what
    // proves the persisted covariance is real and reachable -- a point-only
    // predict would pass identically with no covariance saved at all, so the
    // column check belongs behind `uncertainty: true`.
    let band_path = pred_path.with_extension("band.csv");
    let band_args = PredictArgs {
        model: model_path,
        new_data: train_path,
        out: band_path.clone(),
        offset_column: None,
        noise_offset_column: None,
        id_column: None,
        uncertainty: true,
        level: 0.95,
        covariance_mode: Some(InferenceCovarianceMode::SmoothingCorrected),
        mode: PredictModeArg::PosteriorMean,
        no_bias_correction: false,
    };
    run_predict(band_args).unwrap_or_else(|e| {
        panic!(
            "{} failed: {:?}",
            "posterior-mean predict with uncertainty on a saved covariance", e
        )
    });
    let band_text = fs::read_to_string(&band_path)
        .unwrap_or_else(|e| panic!("{} failed: {:?}", "read band csv", e));
    let band_header = band_text.lines().next().unwrap_or("");
    for required in [
        "posterior_mean",
        "posterior_mean_standard_error",
        "posterior_mean_lower",
        "posterior_mean_upper",
    ] {
        assert!(
            band_header.contains(required),
            "posterior-mean prediction with uncertainty is missing {required}: {band_header}"
        );
    }
}

fn test_payload(
    formula: impl Into<String>,
    model_kind: ModelKind,
    family_state: FittedFamily,
    family: impl Into<String>,
) -> FittedModelPayload {
    let mut payload = FittedModelPayload::new(
        MODEL_PAYLOAD_VERSION,
        formula.into(),
        model_kind,
        family_state,
        family.into(),
    );
    payload.data_schema = Some(DataSchema { columns: vec![] });
    payload
}

fn intercept_only_gaussian_location_scale_model(
    beta_mu: f64,
    beta_log_sigma: f64,
    response_scale: f64,
) -> SavedModel {
    let fit_result = compact_saved_multiblock_fit_result(
        vec![
            gam::estimate::FittedBlock {
                beta: array![beta_mu],
                role: BlockRole::Location,
                edf: 1.0,
                lambdas: Array1::zeros(0),
            },
            gam::estimate::FittedBlock {
                beta: array![beta_log_sigma],
                role: BlockRole::Scale,
                edf: 1.0,
                lambdas: Array1::zeros(0),
            },
        ],
        Array1::zeros(0),
        1.0,
        None,
        None,
        None,
        saved_fit_summary_fixture(),
    );
    let mut payload = test_payload(
        "y ~ 1",
        ModelKind::LocationScale,
        FittedFamily::LocationScale {
            likelihood: LikelihoodSpec::new(
                ResponseFamily::Gaussian,
                InverseLink::Standard(StandardLink::Identity),
            ),
            base_link: None,
        },
        FAMILY_GAUSSIAN_LOCATION_SCALE,
    );
    payload.fit_result = Some(fit_result);
    payload.formula_noise = Some("1".to_string());
    payload.beta_noise = Some(vec![beta_log_sigma]);
    payload.gaussian_response_scale = Some(response_scale);
    payload.set_training_feature_metadata(vec![], vec![]);
    payload.resolved_termspec = Some(empty_termspec());
    payload.resolved_termspec_noise = Some(empty_termspec());
    SavedModel::from_payload(payload)
}

fn intercept_only_binomial_location_scale_model(
    beta_t: f64,
    beta_ls: f64,
    covariance: Array2<f64>,
    beta_link_wiggle: Option<Vec<f64>>,
    wiggle_knots: Option<Vec<f64>>,
    wiggle_degree: Option<usize>,
) -> SavedModel {
    let mut blocks = vec![
        gam::estimate::FittedBlock {
            beta: array![beta_t],
            role: BlockRole::Location,
            edf: 1.0,
            lambdas: Array1::zeros(0),
        },
        gam::estimate::FittedBlock {
            beta: array![beta_ls],
            role: BlockRole::Scale,
            edf: 1.0,
            lambdas: Array1::zeros(0),
        },
    ];
    if let Some(beta_wiggle) = beta_link_wiggle.as_ref() {
        blocks.push(gam::estimate::FittedBlock {
            beta: Array1::from_vec(beta_wiggle.clone()),
            role: BlockRole::LinkWiggle,
            edf: beta_wiggle.len() as f64,
            lambdas: Array1::zeros(0),
        });
    }
    let fit_result = compact_saved_multiblock_fit_result(
        blocks,
        Array1::zeros(0),
        1.0,
        Some(covariance.clone()),
        Some(covariance),
        None,
        saved_fit_summary_fixture(),
    );
    let mut payload = test_payload(
        "y ~ 1",
        ModelKind::LocationScale,
        FittedFamily::LocationScale {
            likelihood: LikelihoodSpec::new(
                ResponseFamily::Binomial,
                InverseLink::Standard(StandardLink::Probit),
            ),
            base_link: Some(InverseLink::Standard(StandardLink::Probit)),
        },
        "binomial-location-scale",
    );
    payload.fit_result = Some(fit_result);
    payload.link = Some(InverseLink::Standard(StandardLink::Probit));
    payload.formula_noise = Some("1".to_string());
    payload.beta_noise = Some(vec![beta_ls]);
    payload.linkwiggle_knots = wiggle_knots;
    payload.linkwiggle_degree = wiggle_degree;
    payload.beta_link_wiggle = beta_link_wiggle;
    payload.set_training_feature_metadata(vec![], vec![]);
    payload.resolved_termspec = Some(empty_termspec());
    payload.resolved_termspec_noise = Some(empty_termspec());
    SavedModel::from_payload(payload)
}

fn posterior_mean_prediction_for_model(model: &SavedModel) -> f64 {
    let td = tempdir().unwrap_or_else(|e| panic!("{} failed: {:?}", "tempdir", e));
    let model_path = td.path().join("model.json");
    let data_path = td.path().join("new_data.csv");
    let out_path = td.path().join("pred.csv");
    write_model_json(&model_path, model)
        .unwrap_or_else(|e| panic!("{} failed: {:?}", "write saved model", e));
    fs::write(&data_path, "unused\n0\n")
        .unwrap_or_else(|e| panic!("{} failed: {:?}", "write prediction data", e));
    let args = PredictArgs {
        model: model_path,
        new_data: data_path,
        out: out_path.clone(),
        offset_column: None,
        noise_offset_column: None,
        id_column: None,
        uncertainty: false,
        level: 0.95,
        covariance_mode: Some(InferenceCovarianceMode::SmoothingCorrected),
        mode: PredictModeArg::PosteriorMean,
        no_bias_correction: false,
    };
    run_predict(args)
        .unwrap_or_else(|e| panic!("{} failed: {:?}", "predict binomial location-scale", e));
    csv_mean_at(&out_path, 0)
}

fn mc_nonwiggle_posterior_mean(
    beta_t: f64,
    beta_ls: f64,
    cov: &Array2<f64>,
    draws: usize,
    seed: u64,
) -> f64 {
    assert_eq!(cov.dim(), (2, 2));
    let var_t = cov[[0, 0]].max(0.0);
    let var_ls = cov[[1, 1]].max(0.0);
    let cov_tl = cov[[0, 1]];
    let l11 = var_t.sqrt();
    let l21 = if l11 > 0.0 { cov_tl / l11 } else { 0.0 };
    let l22 = (var_ls - l21 * l21).max(0.0).sqrt();
    let mut rng = StdRng::seed_from_u64(seed);
    let mut acc = 0.0;
    for _ in 0..draws {
        let z1: f64 = StandardNormal.sample(&mut rng);
        let z2: f64 = StandardNormal.sample(&mut rng);
        let t = beta_t + l11 * z1;
        let ls = beta_ls + l21 * z1 + l22 * z2;
        acc += gam::probability::normal_cdf(
            -t * gam::families::sigma_link::exp_sigma_inverse_from_eta_scalar(ls),
        );
    }
    acc / draws.max(1) as f64
}

fn mcwiggle_posterior_mean(
    beta_t: f64,
    beta_ls: f64,
    beta_link_wiggle: &[f64],
    cov_diag: &[f64],
    model: &SavedModel,
    draws: usize,
    seed: u64,
) -> f64 {
    assert_eq!(cov_diag.len(), 2 + beta_link_wiggle.len());
    let mut rng = StdRng::seed_from_u64(seed);
    let mut beta_draws = Array2::<f64>::zeros((draws, beta_link_wiggle.len()));
    let mut q0_draws = Array1::<f64>::zeros(draws);
    for i in 0..draws {
        let z_t: f64 = StandardNormal.sample(&mut rng);
        let z_ls: f64 = StandardNormal.sample(&mut rng);
        let t = beta_t + cov_diag[0].max(0.0).sqrt() * z_t;
        let ls = beta_ls + cov_diag[1].max(0.0).sqrt() * z_ls;
        q0_draws[i] = -t * gam::families::sigma_link::exp_sigma_inverse_from_eta_scalar(ls);
        for j in 0..beta_link_wiggle.len() {
            let zw: f64 = StandardNormal.sample(&mut rng);
            beta_draws[[i, j]] = beta_link_wiggle[j] + cov_diag[2 + j].max(0.0).sqrt() * zw;
        }
    }
    let wiggle_design = test_saved_linkwiggle_design(&q0_draws, model)
        .unwrap_or_else(|e| panic!("{} failed: {:?}", "wiggle design", e))
        .expect("wiggle model should produce basis");
    let mut acc = 0.0;
    for i in 0..draws {
        let q = q0_draws[i] + wiggle_design.row(i).dot(&beta_draws.row(i));
        acc += gam::probability::normal_cdf(q);
    }
    acc / draws.max(1) as f64
}

#[test]
fn classify_cli_errorspecializes_thin_plate_knot_count_error() {
    let err = classify_cli_error(
            "failed to build term collection design: Invalid input: thin-plate spline requires at least d+1 knots (7), got 3"
                .to_string(),
        );
    let advice = err
        .advice()
        .unwrap_or_else(|| panic!("{} failed", "thin-plate advice"));
    assert!(advice.contains("Increase the number of centers/knots"));
    assert!(!advice.contains("Shape mismatch detected"));
}

#[test]
fn classify_cli_errorspecializes_duchon_power_too_low() {
    // A Duchon admissibility error mentions "dimension=N" literally; ensure
    // it is NOT misclassified as a data-shape mismatch and that the advice
    // points at raising the power.
    let err = classify_cli_error(
        "transformation-normal fit failed: Underlying basis function generation failed: \
             Invalid input: Duchon collision derivative phi^(2) psi triplet requires \
             2*(p+s) > dimension+2; got 2*(p+s)=18, dimension=16, p=1, s=8. \
             The exact two-block / transformation-normal path needs analytic length-scale \
             derivatives of the kernel, which are finite only for a smoother spline: \
             raise power to >= 9 (or reduce the joint smooth's dimension)."
            .to_string(),
    );
    let advice = err
        .advice()
        .unwrap_or_else(|| panic!("{} failed", "duchon advice"));
    assert!(advice.contains("power"));
    assert!(!advice.contains("Shape mismatch detected"));
}

#[test]
fn classify_cli_errorspecializes_thin_plate_knot_error() {
    let err = classify_cli_error(
            "failed to build term collection design: Invalid input: thin-plate spline requires at least d+1 knots (13), got 12"
                .to_string(),
        );
    let advice = err
        .advice()
        .unwrap_or_else(|| panic!("{} failed", "thin-plate advice"));
    assert!(advice.contains("Increase the number of centers/knots"));
    assert!(!advice.contains("Shape mismatch detected"));
}

#[test]
fn compact_fit_result_for_batch_preserves_unified_geometry_invariant() {
    let hessian = array![[4.0, 0.2], [0.2, 3.0]];
    let working_weights = array![1.0, 0.75, 0.5];
    let working_response = array![0.2, -0.1, 0.4];
    let lambdas = array![0.5];
    let mut fit = gam::estimate::UnifiedFitResult::try_from_parts(UnifiedFitResultParts {
        blocks: vec![FittedBlock {
            beta: array![0.1, -0.2],
            role: BlockRole::Mean,
            edf: 1.5,
            lambdas: lambdas.clone(),
        }],
        training_sample_size: 3,
        log_lambdas: lambdas.mapv(f64::ln),
        lambdas,
        likelihood_family: Some(LikelihoodSpec::new(
            ResponseFamily::Binomial,
            InverseLink::Standard(StandardLink::Logit),
        )),
        likelihood_scale: LikelihoodScaleMetadata::FixedDispersion { phi: 1.0 },
        log_likelihood_normalization: LogLikelihoodNormalization::UserProvided,
        log_likelihood: -2.0,
        deviance: 4.0,
        reml_score: Some(0.0),
        stable_penalty_term: 0.25,
        penalized_objective: Some(2.25),
        used_device: false,
        outer_iterations: 2,
        outer_converged: true,
        outer_gradient_norm: Some(1e-8),
        standard_deviation: 1.0,
        covariance_conditional: None,
        covariance_corrected: None,
        inference: Some(FitInference {
            edf_by_block: vec![1.5],
            penalty_block_trace: vec![],
            edf_total: 1.5,
            smoothing_correction: None,
            smoothing_correction_method: None,
            smoothing_correction_first_order: None,
            smoothing_correction_method_first_order: None,
            penalized_hessian: hessian.clone().into(),
            reparam_qs: Some(Array2::eye(2)),
            dispersion: gam::estimate::Dispersion::known(1.0)
                .expect("unit known dispersion is valid"),
            beta_covariance: None,
            beta_standard_errors: None,
            beta_covariance_corrected: None,
            beta_standard_errors_corrected: None,
            beta_covariance_frequentist: None,
            coefficient_influence: None,
            weighted_gram: None,
            bias_correction_beta: None,
            bias_correction_jacobian: None,
        }),
        fitted_link: FittedLinkState::Standard(Some(StandardLink::Logit)),
        geometry: Some(FitGeometry {
            coefficient_gauge: Gauge::identity(&[2]),
            penalized_hessian: hessian.into(),
            constrained_posterior: None,
            working: Some(gam::estimate::WorkingGeometry {
                weights: working_weights,
                response: working_response,
            }),
        }),
        block_states: Vec::new(),
        pirls_status: gam::pirls::PirlsStatus::Converged,
        max_abs_eta: 0.4,
        constraint_kkt: None,
        // This fixture declares two outer iterations over a non-empty smoothing
        // vector, so it describes a fit whose outer loop RAN. Assembly then
        // requires the analytic stationarity certificate that proves the run
        // reached a stationary point -- a default `FitArtifacts` carries none,
        // and the constructor correctly refuses:
        //
        //   outer iterations ran without an analytic stationarity certificate
        //
        // The values mirror the repository's canonical clean certificate in
        // tests/identifiability/misc/certificate_ledger_unified.rs: a projected
        // gradient far inside its bound, a solver-band rung, and measured PSD
        // curvature.
        artifacts: gam::estimate::FitArtifacts {
            criterion_certificate: Some(gam::solver::rho_optimizer::OuterCriterionCertificate {
                stationarity:
                    gam::solver::rho_optimizer::OuterStationarityCertificate::AnalyticGradient {
                        grad_norm: 1e-9,
                        projected_grad_norm: 1e-9,
                        bound: 1e-6,
                        rung: gam::solver::rho_optimizer::CertifiedRung {
                            label: "solver-band".to_string(),
                            derived_standard: false,
                        },
                    },
                curvature: gam::solver::rho_optimizer::CurvatureEvidence::Measured { psd: true },
                lambdas_railed: Vec::new(),
                railed_facts: Vec::new(),
                curvature_floor: None,
            }),
            ..Default::default()
        },
        inner_cycles: 3,
    })
    .unwrap_or_else(|e| {
        panic!(
            "{} failed: {:?}",
            "construct compactable unified fit result", e
        )
    });

    compact_fit_result_for_batch(&mut fit);

    let inf = fit
        .inference
        .as_ref()
        .unwrap_or_else(|| panic!("{} failed", "inference kept"));
    let geom = fit
        .geometry
        .as_ref()
        .unwrap_or_else(|| panic!("{} failed", "geometry kept"));
    assert!(inf.reparam_qs.is_none());
    let working = geom
        .working
        .as_ref()
        .unwrap_or_else(|| panic!("{} failed", "working geometry kept"));
    assert_eq!(working.weights.len(), 3);
    assert_eq!(working.response.len(), 3);
    fit.validate_numeric_finiteness().unwrap_or_else(|e| {
        panic!(
            "{} failed: {:?}",
            "compacted fit result remains persistable", e
        )
    });
}

#[test]
fn core_saved_fit_result_json_roundtripswith_finite_summary() {
    let fit = core_saved_fit_result(
        Array1::from_vec(vec![0.1, -0.2]),
        Array1::from_vec(vec![1e-3]),
        1.0,
        None,
        None,
        SavedFitSummary {
            training_sample_size: 1,
            likelihood_family: Some(LikelihoodSpec::new(
                ResponseFamily::Gaussian,
                InverseLink::Standard(StandardLink::Identity),
            )),
            likelihood_scale: LikelihoodScaleMetadata::ProfiledGaussian,
            log_likelihood_normalization: LogLikelihoodNormalization::Full,
            log_likelihood: -0.75,
            iterations: 3,
            finalgrad_norm: 0.25,
            pirls_status: gam::pirls::PirlsStatus::Converged,
            deviance: 1.5,
            stable_penalty_term: 0.4,
            max_abs_eta: 2.0,
            reml_score: Some(0.95),
            // Same contract as the shared fixture: a saved fit is a CONVERGED
            // fit (SPEC 20), and without a certificate the assembly refuses to
            // mint one from a summary whose outer loop ran.
            criterion_certificate: Some(gam::estimate::OuterCriterionCertificate {
                stationarity: gam::estimate::OuterStationarityCertificate::AnalyticGradient {
                    grad_norm: 1e-8,
                    projected_grad_norm: 1e-8,
                    bound: 1e-4,
                    rung: gam::model_types::CertifiedRung {
                        label: "solver-band".to_string(),
                        derived_standard: false,
                    },
                },
                curvature: gam::model_types::CurvatureEvidence::Measured { psd: true },
                lambdas_railed: Vec::new(),
                railed_facts: Vec::new(),
                curvature_floor: None,
            }),
        },
    );
    let payload = serde_json::to_string(&fit)
        .unwrap_or_else(|e| panic!("{} failed: {:?}", "serialize fit result", e));
    let parsed: gam::estimate::UnifiedFitResult = serde_json::from_str(&payload)
        .unwrap_or_else(|e| panic!("{} failed: {:?}", "deserialize fit result", e));
    assert_eq!(parsed.outer_gradient_norm, Some(0.25));
    assert_eq!(parsed.deviance, 1.5);
    assert_eq!(parsed.reml_score(), Some(0.95));
}

#[test]
fn parse_bounded_linear_term_defaults_to_no_prior() {
    let parsed = parse_formula("y ~ bounded(mu_hat, min=0, max=1) + z")
        .unwrap_or_else(|e| panic!("{} failed: {:?}", "formula", e));
    assert_eq!(parsed.terms.len(), 2);
    match &parsed.terms[0] {
        ParsedTerm::BoundedLinear {
            name,
            min,
            max,
            prior,
            double_penalty,
        } => {
            assert_eq!(name, "mu_hat");
            assert_eq!((*min, *max), (0.0, 1.0));
            match prior {
                BoundedCoefficientPriorSpec::None => {}
                other => panic!("unexpected prior: {other:?}"),
            }
            assert!(!*double_penalty);
        }
        other => panic!("expected bounded linear term, got {other:?}"),
    }
}

#[test]
fn parse_bounded_linear_termwith_center_pull() {
    let parsed = parse_formula("y ~ bounded(mu_hat, min=0, max=1, pull=\"center\") + z")
        .unwrap_or_else(|e| panic!("{} failed: {:?}", "formula", e));
    assert_eq!(parsed.terms.len(), 2);
    match &parsed.terms[0] {
        ParsedTerm::BoundedLinear {
            name,
            min,
            max,
            prior,
            double_penalty,
        } => {
            assert_eq!(name, "mu_hat");
            assert_eq!((*min, *max), (0.0, 1.0));
            match prior {
                BoundedCoefficientPriorSpec::Beta { a, b } => {
                    assert_eq!((*a, *b), (2.0, 2.0));
                }
                other => panic!("unexpected prior: {other:?}"),
            }
            assert!(!*double_penalty);
        }
        other => panic!("expected bounded linear term, got {other:?}"),
    }
}

#[test]
fn parse_bounded_linear_termwith_uniform_prior() {
    let parsed = parse_formula("y ~ bounded(mu_hat, min=0, max=1, prior=\"uniform\") + z")
        .unwrap_or_else(|e| panic!("{} failed: {:?}", "formula", e));
    assert_eq!(parsed.terms.len(), 2);
    match &parsed.terms[0] {
        ParsedTerm::BoundedLinear {
            name,
            min,
            max,
            prior,
            double_penalty,
        } => {
            assert_eq!(name, "mu_hat");
            assert_eq!(*min, 0.0);
            assert_eq!(*max, 1.0);
            match prior {
                BoundedCoefficientPriorSpec::Uniform => {}
                other => panic!("unexpected prior: {other:?}"),
            }
            assert!(!*double_penalty);
        }
        other => panic!("unexpected term: {other:?}"),
    }
}

#[test]
fn parse_bounded_linear_target_strength_maps_to_beta_prior() {
    let parsed = parse_formula("y ~ bounded(mu_hat, min=-1, max=1, target=0.5, strength=4)")
        .unwrap_or_else(|e| panic!("{} failed: {:?}", "formula", e));
    match &parsed.terms[0] {
        ParsedTerm::BoundedLinear { prior, .. } => match prior {
            BoundedCoefficientPriorSpec::Beta { a, b } => {
                assert!((*a - 4.0).abs() < 1e-12);
                assert!((*b - 2.0).abs() < 1e-12);
            }
            other => panic!("unexpected prior: {other:?}"),
        },
        other => panic!("expected bounded linear term, got {other:?}"),
    }
}

#[test]
fn warns_for_repeated_univariate_duchon_spatial_terms() {
    let spec = TermCollectionSpec {
        linear_terms: vec![],
        random_effect_terms: vec![],
        smooth_terms: vec![
            SmoothTermSpec {
                frozen_parametric_residualization: None,
                name: "pc1".to_string(),
                basis: SmoothBasisSpec::Duchon {
                    feature_cols: vec![0],
                    spec: DuchonBasisSpec {
                        radial_reparam: None,
                        center_strategy: CenterStrategy::FarthestPoint { num_centers: 12 },
                        length_scale: Some(1.0),
                        power: 1.0,
                        nullspace_order: DuchonNullspaceOrder::Linear,
                        identifiability: SpatialIdentifiability::default(),
                        aniso_log_scales: None,
                        operator_penalties: DuchonOperatorPenaltySpec::default(),
                        boundary: OneDimensionalBoundary::Open,
                        periodic: None,
                    },
                    input_scale: None,
                },
                shape: gam::smooth::ShapeConstraint::None,
                joint_null_rotation: None,
            },
            SmoothTermSpec {
                frozen_parametric_residualization: None,
                name: "pc2".to_string(),
                basis: SmoothBasisSpec::Duchon {
                    feature_cols: vec![1],
                    spec: DuchonBasisSpec {
                        radial_reparam: None,
                        center_strategy: CenterStrategy::FarthestPoint { num_centers: 12 },
                        length_scale: Some(1.0),
                        power: 1.0,
                        nullspace_order: DuchonNullspaceOrder::Linear,
                        identifiability: SpatialIdentifiability::default(),
                        aniso_log_scales: None,
                        operator_penalties: DuchonOperatorPenaltySpec::default(),
                        boundary: OneDimensionalBoundary::Open,
                        periodic: None,
                    },
                    input_scale: None,
                },
                shape: gam::smooth::ShapeConstraint::None,
                joint_null_rotation: None,
            },
            SmoothTermSpec {
                frozen_parametric_residualization: None,
                name: "pc3".to_string(),
                basis: SmoothBasisSpec::Duchon {
                    feature_cols: vec![2],
                    spec: DuchonBasisSpec {
                        radial_reparam: None,
                        center_strategy: CenterStrategy::FarthestPoint { num_centers: 12 },
                        length_scale: Some(1.0),
                        power: 1.0,
                        nullspace_order: DuchonNullspaceOrder::Linear,
                        identifiability: SpatialIdentifiability::default(),
                        aniso_log_scales: None,
                        operator_penalties: DuchonOperatorPenaltySpec::default(),
                        boundary: OneDimensionalBoundary::Open,
                        periodic: None,
                    },
                    input_scale: None,
                },
                shape: gam::smooth::ShapeConstraint::None,
                joint_null_rotation: None,
            },
        ],
    };
    let headers = vec!["pc1".to_string(), "pc2".to_string(), "pc3".to_string()];

    let warnings = collect_smooth_structure_warnings(&spec, &headers, "model");

    assert_eq!(warnings.len(), 1);
    assert!(warnings[0].contains("3 separate 1D duchon spatial smooths"));
    assert!(warnings[0].contains("[pc1, pc2, pc3]"));
    assert!(warnings[0].contains("TIP:"));
    assert!(
        warnings[0].contains("s(pc1, type=duchon) + s(pc2, type=duchon) + s(pc3, type=duchon)")
    );
    assert!(warnings[0].contains("duchon(pc1, pc2, pc3)"));
}

#[test]
fn does_notwarn_for_singlemultivariate_matern_spatial_term() {
    let spec = TermCollectionSpec {
        linear_terms: vec![],
        random_effect_terms: vec![],
        smooth_terms: vec![SmoothTermSpec {
            frozen_parametric_residualization: None,
            name: "matern".to_string(),
            basis: SmoothBasisSpec::Matern {
                feature_cols: vec![0, 1, 2],
                spec: MaternBasisSpec {
                    center_strategy: CenterStrategy::FarthestPoint { num_centers: 12 },
                    length_scale: gam::terms::basis::MaternLengthScale::fixed(1.0),
                    nu: MaternNu::ThreeHalves,
                    double_penalty: true,
                    include_intercept: false,
                    identifiability: gam::basis::MaternIdentifiability::default(),
                    aniso_log_scales: None,
                    periodic: None,
                },
                input_scale: None,
            },
            shape: gam::smooth::ShapeConstraint::None,
            joint_null_rotation: None,
        }],
    };
    let headers = vec!["pc1".to_string(), "pc2".to_string(), "pc3".to_string()];

    let warnings = collect_smooth_structure_warnings(&spec, &headers, "model");

    assert!(warnings.is_empty());
}

#[test]
fn warns_for_repeated_univariate_thinplate_spatial_terms() {
    let spec = TermCollectionSpec {
        linear_terms: vec![],
        random_effect_terms: vec![],
        smooth_terms: vec![
            SmoothTermSpec {
                frozen_parametric_residualization: None,
                name: "pc1".to_string(),
                basis: SmoothBasisSpec::ThinPlate {
                    feature_cols: vec![0],
                    spec: ThinPlateBasisSpec {
                        center_strategy: CenterStrategy::FarthestPoint { num_centers: 8 },
                        length_scale: 1.0,
                        double_penalty: true,
                        identifiability: SpatialIdentifiability::default(),
                        radial_reparam: None,
                        periodic: None,
                    },
                    input_scale: None,
                },
                shape: gam::smooth::ShapeConstraint::None,
                joint_null_rotation: None,
            },
            SmoothTermSpec {
                frozen_parametric_residualization: None,
                name: "pc2".to_string(),
                basis: SmoothBasisSpec::ThinPlate {
                    feature_cols: vec![1],
                    spec: ThinPlateBasisSpec {
                        center_strategy: CenterStrategy::FarthestPoint { num_centers: 8 },
                        length_scale: 1.0,
                        double_penalty: true,
                        identifiability: SpatialIdentifiability::default(),
                        radial_reparam: None,
                        periodic: None,
                    },
                    input_scale: None,
                },
                shape: gam::smooth::ShapeConstraint::None,
                joint_null_rotation: None,
            },
        ],
    };
    let headers = vec!["pc1".to_string(), "pc2".to_string()];

    let warnings = collect_smooth_structure_warnings(&spec, &headers, "model");

    assert_eq!(warnings.len(), 1);
    assert!(warnings[0].contains("2 separate 1D thinplate/tps spatial smooths"));
    assert!(warnings[0].contains("s(pc1, type=tps) + s(pc2, type=tps)"));
    assert!(warnings[0].contains("thinplate(pc1, pc2)"));
}

#[test]
fn warns_for_linear_terms_overlappingwith_smoothvariables() {
    let spec = TermCollectionSpec {
        linear_terms: vec![LinearTermSpec {
            name: "pc1".to_string(),
            feature_col: 0,
            feature_cols: vec![0],
            categorical_levels: vec![],
            double_penalty: true,
            coefficient_geometry: LinearCoefficientGeometry::default(),
            coefficient_min: None,
            coefficient_max: None,
            frozen_function_mass: None,
        }],
        random_effect_terms: vec![],
        smooth_terms: vec![SmoothTermSpec {
            frozen_parametric_residualization: None,
            name: "duchon(pc1, pc2, pc3)".to_string(),
            basis: SmoothBasisSpec::Duchon {
                feature_cols: vec![0, 1, 2],
                spec: DuchonBasisSpec {
                    radial_reparam: None,
                    center_strategy: CenterStrategy::FarthestPoint { num_centers: 12 },
                    length_scale: Some(1.0),
                    power: 1.0,
                    nullspace_order: DuchonNullspaceOrder::Linear,
                    identifiability: SpatialIdentifiability::default(),
                    aniso_log_scales: None,
                    operator_penalties: DuchonOperatorPenaltySpec::default(),
                    boundary: OneDimensionalBoundary::Open,
                    periodic: None,
                },
                input_scale: None,
            },
            shape: gam::smooth::ShapeConstraint::None,
            joint_null_rotation: None,
        }],
    };
    let headers = vec!["pc1".to_string(), "pc2".to_string(), "pc3".to_string()];

    let warnings = collect_smooth_structure_warnings(&spec, &headers, "model");

    assert_eq!(warnings.len(), 1);
    assert!(warnings[0].contains("feature(s) [pc1]"));
    assert!(warnings[0].contains("duchon(pc1, pc2, pc3)"));
    assert!(warnings[0].contains("linear(pc1)"));
    assert!(warnings[0].contains("residualizes the smooth against the intercept"));
    assert!(warnings[0].contains("nonlinear remainder"));
}

#[test]
fn warns_for_nested_smooth_terms_with_hierarchical_ownership() {
    let spec = TermCollectionSpec {
        linear_terms: vec![],
        random_effect_terms: vec![],
        smooth_terms: vec![
            SmoothTermSpec {
                frozen_parametric_residualization: None,
                name: "duchon(pc1, pc2)".to_string(),
                basis: SmoothBasisSpec::Duchon {
                    feature_cols: vec![0, 1],
                    spec: DuchonBasisSpec {
                        radial_reparam: None,
                        center_strategy: CenterStrategy::FarthestPoint { num_centers: 6 },
                        length_scale: Some(1.0),
                        power: 1.0,
                        nullspace_order: DuchonNullspaceOrder::Linear,
                        identifiability: SpatialIdentifiability::default(),
                        aniso_log_scales: None,
                        operator_penalties: DuchonOperatorPenaltySpec::default(),
                        boundary: OneDimensionalBoundary::Open,
                        periodic: None,
                    },
                    input_scale: None,
                },
                shape: gam::smooth::ShapeConstraint::None,
                joint_null_rotation: None,
            },
            SmoothTermSpec {
                frozen_parametric_residualization: None,
                name: "s(pc1)".to_string(),
                basis: SmoothBasisSpec::BSpline1D {
                    feature_col: 0,
                    spec: BSplineBasisSpec {
                        degree: 3,
                        penalty_order: 2,
                        knotspec: BSplineKnotSpec::Generate {
                            data_range: (0.0, 1.0),
                            num_internal_knots: 4,
                        },
                        double_penalty: false,
                        identifiability: BSplineIdentifiability::default(),
                        boundary: OneDimensionalBoundary::Open,
                        boundary_conditions: BSplineBoundaryConditions::default(),
                    },
                },
                shape: gam::smooth::ShapeConstraint::None,
                joint_null_rotation: None,
            },
        ],
    };
    let headers = vec!["pc1".to_string(), "pc2".to_string()];

    let warnings = collect_smooth_structure_warnings(&spec, &headers, "model");

    assert_eq!(warnings.len(), 1);
    assert!(warnings[0].contains("duchon(pc1, pc2)"));
    assert!(warnings[0].contains("s(pc1)"));
    assert!(warnings[0].contains("automatic hierarchical ownership"));
    assert!(warnings[0].contains("residualized against that overlap"));
}

#[test]
fn parse_linear_termwith_box_constraints() {
    let parsed = parse_formula("y ~ linear(mu_hat, min=0, max=1) + nonpositive(z)")
        .unwrap_or_else(|e| panic!("{} failed: {:?}", "formula", e));
    assert_eq!(parsed.terms.len(), 2);
    match &parsed.terms[0] {
        ParsedTerm::Linear {
            name,
            explicit,
            double_penalty,
            coefficient_min,
            coefficient_max,
        } => {
            assert_eq!(name, "mu_hat");
            assert!(*explicit);
            assert!(!*double_penalty);
            assert_eq!(*coefficient_min, Some(0.0));
            assert_eq!(*coefficient_max, Some(1.0));
        }
        other => panic!("expected constrained linear term, got {other:?}"),
    }
    match &parsed.terms[1] {
        ParsedTerm::Linear {
            name,
            coefficient_min,
            coefficient_max,
            ..
        } => {
            assert_eq!(name, "z");
            assert_eq!(*coefficient_min, None);
            assert_eq!(*coefficient_max, Some(0.0));
        }
        other => panic!("expected nonpositive linear term, got {other:?}"),
    }
}

#[test]
fn build_termspec_leaves_parametric_linear_terms_unpenalized_by_default() {
    let parsed = parse_formula("y ~ x + linear(z) + nonnegative(w)")
        .unwrap_or_else(|e| panic!("{} failed: {:?}", "formula", e));
    let ds = Dataset {
        headers: vec!["x".to_string(), "z".to_string(), "w".to_string()],
        values: array![[1.0, 2.0, 3.0], [1.5, 2.5, 3.5], [2.0, 3.0, 4.0],],
        schema: DataSchema {
            columns: vec![
                SchemaColumn {
                    name: "x".to_string(),
                    kind: ColumnKindTag::Continuous,
                    levels: vec![],
                },
                SchemaColumn {
                    name: "z".to_string(),
                    kind: ColumnKindTag::Continuous,
                    levels: vec![],
                },
                SchemaColumn {
                    name: "w".to_string(),
                    kind: ColumnKindTag::Continuous,
                    levels: vec![],
                },
            ],
        },
        column_kinds: vec![
            ColumnKindTag::Continuous,
            ColumnKindTag::Continuous,
            ColumnKindTag::Continuous,
        ],
    };
    let col_map = HashMap::from([
        ("x".to_string(), 0usize),
        ("z".to_string(), 1usize),
        ("w".to_string(), 2usize),
    ]);
    let mut inference_notes = Vec::<String>::new();
    let spec = super::build_termspec(
        &parsed.terms,
        &ds,
        &col_map,
        &mut inference_notes,
        &gam::ResourcePolicy::default_library(),
    )
    .unwrap_or_else(|e| panic!("{} failed: {:?}", "term spec", e));

    assert_eq!(spec.linear_terms.len(), 3);
    assert!(
        spec.linear_terms.iter().all(|term| !term.double_penalty),
        "parametric linear terms should be unpenalized by default: {:?}",
        spec.linear_terms
            .iter()
            .map(|term| (&term.name, term.double_penalty))
            .collect::<Vec<_>>()
    );
}

#[test]
fn parametric_double_penalty_is_an_explicit_opt_in() {
    let parsed = parse_formula("y ~ linear(x, double_penalty=true) + z:w")
        .unwrap_or_else(|e| panic!("{} failed: {:?}", "formula", e));
    assert_eq!(parsed.terms.len(), 2);
    match &parsed.terms[0] {
        ParsedTerm::Linear { double_penalty, .. } => assert!(*double_penalty),
        other => panic!("expected explicit linear term, got {other:?}"),
    }
    match &parsed.terms[1] {
        ParsedTerm::Interaction { double_penalty, .. } => assert!(!*double_penalty),
        other => panic!("expected bare interaction term, got {other:?}"),
    }
}

#[test]
fn build_termspec_accepts_joint_thinplate_above_three_dimensions() {
    // TPS supports arbitrary dimensions via the general polyharmonic kernel
    // with auto-selected penalty order m = floor(d/2) + 1.
    let parsed = parse_formula("y ~ thinplate(pc1, pc2, pc3, pc4, centers=6)")
        .unwrap_or_else(|e| panic!("{} failed: {:?}", "formula", e));
    let n = 20;
    let mut rng = 42u64;
    let mut vals = Array2::<f64>::zeros((n, 4));
    for v in vals.iter_mut() {
        // simple LCG for deterministic pseudo-random data
        rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
        *v = (rng >> 33) as f64 / (1u64 << 31) as f64;
    }
    let ds = Dataset {
        headers: vec![
            "pc1".to_string(),
            "pc2".to_string(),
            "pc3".to_string(),
            "pc4".to_string(),
        ],
        values: vals,
        schema: DataSchema {
            columns: vec![
                SchemaColumn {
                    name: "pc1".to_string(),
                    kind: ColumnKindTag::Continuous,
                    levels: vec![],
                },
                SchemaColumn {
                    name: "pc2".to_string(),
                    kind: ColumnKindTag::Continuous,
                    levels: vec![],
                },
                SchemaColumn {
                    name: "pc3".to_string(),
                    kind: ColumnKindTag::Continuous,
                    levels: vec![],
                },
                SchemaColumn {
                    name: "pc4".to_string(),
                    kind: ColumnKindTag::Continuous,
                    levels: vec![],
                },
            ],
        },
        column_kinds: vec![
            ColumnKindTag::Continuous,
            ColumnKindTag::Continuous,
            ColumnKindTag::Continuous,
            ColumnKindTag::Continuous,
        ],
    };
    let col_map = HashMap::from([
        ("pc1".to_string(), 0usize),
        ("pc2".to_string(), 1usize),
        ("pc3".to_string(), 2usize),
        ("pc4".to_string(), 3usize),
    ]);
    let mut inference_notes = Vec::<String>::new();
    let spec = super::build_termspec(
        &parsed.terms,
        &ds,
        &col_map,
        &mut inference_notes,
        &gam::ResourcePolicy::default_library(),
    )
    .unwrap_or_else(|e| panic!("{} failed: {:?}", "4-d TPS should be accepted", e));
    assert_eq!(spec.smooth_terms.len(), 1, "should have one smooth term");
}

#[test]
fn parse_linkwiggle_defaults_to_all_penalty_orders() {
    let parsed = parse_formula("y ~ x + linkwiggle(degree=4, internal_knots=9)")
        .unwrap_or_else(|e| panic!("{} failed: {:?}", "formula", e));
    let lw = parsed
        .linkwiggle
        .unwrap_or_else(|| panic!("{} failed", "expected linkwiggle config"));
    assert_eq!(lw.degree, 4);
    assert_eq!(lw.num_internal_knots, 9);
    assert_eq!(lw.penalty_orders, vec![1, 2, 3]);
    assert!(lw.double_penalty);
}

#[test]
fn parse_linkwiggle_rejects_unknown_options() {
    let err = parse_formula("y ~ x + linkwiggle(knots=9)")
        .expect_err("unknown linkwiggle options should be rejected");
    assert!(
        err.to_string()
            .contains("linkwiggle() does not support option(s) knots")
    );
}

#[test]
fn marginal_slope_linkwiggle_routes_into_anchored_deviation_config() {
    let parsed = parse_formula(
            "y ~ x + linkwiggle(degree=3, internal_knots=9, penalty_order=\"1,3\", double_penalty=false)",
        )
        .unwrap_or_else(|e| panic!("{} failed: {:?}", "formula", e));
    let routed = route_marginal_slope_deviation_blocks(parsed.linkwiggle.as_ref(), None)
        .expect("cubic linkwiggle must route into the deviation config")
        .link_dev
        .expect("main linkwiggle must produce a link-deviation block");
    assert_eq!(routed.degree, 3);
    assert_eq!(routed.num_internal_knots, 9);
    assert_eq!(routed.penalty_order, 3);
    assert_eq!(routed.penalty_orders, vec![1, 3]);
    assert!(!routed.double_penalty);
}

#[test]
fn marginal_slope_linkwiggle_rejects_non_cubic_degree_at_routing_boundary() {
    // Regression for #384: the score-warp / link-deviation block is a
    // structurally cubic I-spline `DeviationRuntime`, so only degree 3 is
    // realizable. The shared formula parser stays general (it also feeds
    // arbitrary-degree timewiggle / location-scale wiggles), so non-cubic
    // linkwiggle degrees must be rejected at this routing boundary — up
    // front, with a clear cubic-only message — instead of parsing fine and
    // then blowing up deep in the fit with "structural deviation runtime is
    // cubic; degree must be 3, got k". On the pre-fix code these degrees
    // routed successfully (the test would fail at the `expect_err`).
    for deg in [1usize, 2, 4, 5, 10] {
        let parsed = parse_formula(&format!(
            "y ~ x + linkwiggle(degree={deg}, internal_knots=9)"
        ))
        .unwrap_or_else(|e| {
            panic!(
                "{} failed: {:?}",
                "non-cubic linkwiggle must still parse at the shared layer", e
            )
        });
        let err = route_marginal_slope_deviation_blocks(parsed.linkwiggle.as_ref(), None)
            .expect_err("non-cubic linkwiggle must be rejected when routed into the cubic block");
        assert!(
            err.contains("degree must be 3"),
            "error should state degree must be 3, got: {err}"
        );
        assert!(
            err.contains("cubic"),
            "error should explain the runtime is cubic, got: {err}"
        );
        assert!(
            err.contains(&format!("degree={deg}")),
            "error should echo the rejected degree, got: {err}"
        );
    }
}

#[test]
fn marginal_slope_deviation_routing_splits_main_and_slope_linkwiggles() {
    let parsed_main = parse_formula(
            "y ~ x + linkwiggle(degree=3, internal_knots=9, penalty_order=\"1,3\", double_penalty=false)",
        )
        .unwrap_or_else(|e| panic!("{} failed: {:?}", "main formula", e));
    let (_, parsed_slope) = parse_matching_auxiliary_formula(
        "1 + linkwiggle(degree=3, internal_knots=7, penalty_order=\"2,3\")",
        "y",
        "--slope-formula",
    )
    .unwrap_or_else(|e| panic!("{} failed: {:?}", "slope formula", e));
    let routed = super::route_marginal_slope_deviation_blocks(
        parsed_main.linkwiggle.as_ref(),
        parsed_slope.linkwiggle.as_ref(),
    )
    .unwrap_or_else(|e| panic!("{} failed: {:?}", "routing", e));
    let link_dev = routed
        .link_dev
        .unwrap_or_else(|| panic!("{} failed", "main link-deviation config"));
    let score_warp = routed
        .score_warp
        .unwrap_or_else(|| panic!("{} failed", "slope score-warp config"));
    assert_eq!(link_dev.degree, 3);
    assert_eq!(link_dev.num_internal_knots, 9);
    assert_eq!(link_dev.penalty_order, 3);
    assert_eq!(link_dev.penalty_orders, vec![1, 3]);
    assert!(!link_dev.double_penalty);
    assert_eq!(score_warp.degree, 3);
    assert_eq!(score_warp.num_internal_knots, 7);
    assert_eq!(score_warp.penalty_order, 3);
    assert_eq!(score_warp.penalty_orders, vec![2, 3]);
    assert!(score_warp.double_penalty);
}

#[test]
fn marginal_slope_routing_rejects_non_cubic_in_either_slot() {
    // #384: rejection must trigger from either the main (link-deviation)
    // or slope (score-warp) slot, since both feed the cubic runtime.
    let parsed_main = parse_formula("y ~ x + linkwiggle(degree=4, internal_knots=9)")
        .unwrap_or_else(|e| panic!("{} failed: {:?}", "main formula parses", e));
    let err = super::route_marginal_slope_deviation_blocks(parsed_main.linkwiggle.as_ref(), None)
        .expect_err("non-cubic main linkwiggle must be rejected at routing");
    assert!(err.contains("degree must be 3"), "got: {err}");

    let (_, parsed_slope) = parse_matching_auxiliary_formula(
        "1 + linkwiggle(degree=5, internal_knots=7)",
        "y",
        "--slope-formula",
    )
    .unwrap_or_else(|e| panic!("{} failed: {:?}", "slope formula parses", e));
    let err =
        super::route_marginal_slope_deviation_blocks(None, parsed_slope.linkwiggle.as_ref())
            .expect_err("non-cubic slope linkwiggle must be rejected at routing");
    assert!(err.contains("degree must be 3"), "got: {err}");
}

#[test]
fn bernoulli_marginal_slope_accepts_only_probit_base_link() {
    let parsed = parse_formula("y ~ x + link(type=probit)")
        .unwrap_or_else(|e| panic!("{} failed: {:?}", "main formula", e));
    let resolved = super::resolve_bernoulli_marginal_slope_base_link(
        parsed.linkspec.as_ref(),
        "bernoulli marginal-slope",
    )
    .unwrap_or_else(|e| panic!("{} failed: {:?}", "explicit probit base link", e));
    assert_eq!(resolved, InverseLink::Standard(StandardLink::Probit));

    for formula in [
        "y ~ x + link(type=logit)",
        "y ~ x + link(type=sas, sas_init=\"0.1,-0.2\")",
        "y ~ x + link(type=beta-logistic, beta_logistic_init=\"0.3,0.7\")",
        "y ~ x + link(type=blended(logit,probit,cloglog), rho=\"0.4,-0.1\")",
    ] {
        let parsed =
            parse_formula(formula).unwrap_or_else(|e| panic!("{} failed: {:?}", "main formula", e));
        let err = super::resolve_bernoulli_marginal_slope_base_link(
            parsed.linkspec.as_ref(),
            "bernoulli marginal-slope",
        )
        .expect_err("non-probit marginal-slope link should be rejected");
        assert!(
            err.contains("requires link(type=probit)"),
            "unexpected error for {formula}: {err}"
        );
    }
}

#[test]
fn bernoulli_marginal_slope_rejects_flexible_and_unbounded_base_links() {
    let parsed = parse_formula("y ~ x + link(type=flexible(logit))")
        .unwrap_or_else(|e| panic!("{} failed: {:?}", "main formula", e));
    let err = super::resolve_bernoulli_marginal_slope_base_link(
        parsed.linkspec.as_ref(),
        "bernoulli marginal-slope",
    )
    .expect_err("flexible link should be rejected");
    assert!(err.contains("does not accept flexible"));

    let parsed = parse_formula("y ~ x + link(type=log)")
        .unwrap_or_else(|e| panic!("{} failed: {:?}", "main formula", e));
    let err = super::resolve_bernoulli_marginal_slope_base_link(
        parsed.linkspec.as_ref(),
        "bernoulli marginal-slope",
    )
    .expect_err("log link should be rejected");
    assert!(err.contains("requires link(type=probit)"));
}

#[test]
fn parse_timewiggle_defaults_to_all_penalty_orders() {
    let parsed = parse_formula("Surv(entry, exit, event) ~ timewiggle(degree=4, internal_knots=9)")
        .unwrap_or_else(|e| panic!("{} failed: {:?}", "formula", e));
    let tw = parsed
        .timewiggle
        .unwrap_or_else(|| panic!("{} failed", "expected timewiggle config"));
    assert_eq!(tw.degree, 4);
    assert_eq!(tw.num_internal_knots, 9);
    assert_eq!(tw.penalty_orders, vec![1, 2, 3]);
    assert!(tw.double_penalty);
}

#[test]
fn parse_timewiggle_rejects_unknown_options() {
    let err = parse_formula("Surv(entry, exit, event) ~ timewiggle(knots=9)")
        .expect_err("unknown timewiggle options should be rejected");
    assert!(
        err.to_string()
            .contains("timewiggle() does not support option(s) knots")
    );
}

#[test]
fn bernoulli_marginal_slope_saved_model_persists_exact_kernel_metadata_only() {
    let model = super::build_bernoulli_marginal_slope_saved_model(
        "y ~ 1".to_string(),
        DataSchema { columns: vec![] },
        "y ~ 1".to_string(),
        "z".to_string(),
        vec![],
        vec![],
        empty_termspec(),
        empty_termspec(),
        core_saved_fit_result(
            array![0.0],
            Array1::zeros(0),
            1.0,
            None,
            None,
            saved_fit_summary_fixture(),
        ),
        // Single marginal coefficient, no influence absorber → truncation
        // is a no-op (p_marginal == block-0 width).
        1,
        0.0,
        0.0,
        SavedLatentZNormalization { mean: 0.2, sd: 1.3 },
        LatentMeasureKind::StandardNormal,
        None,
        None,
        None,
        None,
        InverseLink::Standard(StandardLink::Probit),
        gam::families::survival::lognormal_kernel::FrailtySpec::None,
    )
    .unwrap_or_else(|e| {
        panic!(
            "{} failed: {:?}",
            "build bernoulli marginal-slope saved model", e
        )
    });
    assert_eq!(
        model.payload().latent_z_normalization,
        Some(SavedLatentZNormalization { mean: 0.2, sd: 1.3 })
    );
    assert_eq!(model.payload().marginal_baseline, Some(0.0));
    assert_eq!(model.payload().baseline_slope, Some(0.0));
    assert_eq!(
        model.payload().link.as_ref(),
        Some(&InverseLink::Standard(StandardLink::Probit))
    );
    assert_eq!(
        model
            .resolved_inverse_link()
            .unwrap_or_else(|e| panic!("{} failed: {:?}", "resolved inverse link", e)),
        Some(InverseLink::Standard(StandardLink::Probit))
    );
}

/// Snapshot parity: the CLI and PyFFI save paths feed identical *semantic*
/// inputs into the same shared core assembler
/// (`assemble_bernoulli_marginal_slope_payload`), differing only in
/// source-specific metadata — the CLI supplies per-feature training ranges,
/// the FFI supplies offset columns. This test drives the assembler with both
/// source-metadata shapes and asserts the serialized payloads are
/// byte-identical once the legitimately source-specific fields are
/// normalized away. Any drift in the semantic contract between the two save
/// routes (the exact failure mode #402 closes) would break this assertion.
#[test]
fn cli_and_ffi_bernoulli_marginal_slope_payloads_have_one_contract() {
    let schema = DataSchema {
        columns: vec![SchemaColumn {
            name: "z".to_string(),
            kind: ColumnKindTag::Continuous,
            levels: vec![],
        }],
    };
    // Build the resolved semantic inputs once; clone into the two source
    // shapes so the *only* differences are the source-specific fields.
    let make_inputs = || BernoulliMarginalSlopeInputs {
        formula: "y ~ 1".to_string(),
        data_schema: schema.clone(),
        slope_formula: "y ~ z".to_string(),
        z_column: "z".to_string(),
        resolved_marginalspec: empty_termspec(),
        resolved_slopespec: empty_termspec(),
        fit_result: core_saved_fit_result(
            array![0.3],
            Array1::zeros(0),
            1.0,
            None,
            None,
            saved_fit_summary_fixture(),
        ),
        // Single marginal coefficient, no influence absorber ⇒ truncation
        // is a no-op (p_marginal == block-0 width).
        p_marginal: 1,
        baseline_marginal: -0.2,
        baseline_slope: 0.7,
        latent_z_normalization: SavedLatentZNormalization { mean: 1.1, sd: 2.2 },
        latent_measure: LatentMeasureKind::StandardNormal,
        latent_z_rank_int_calibration: None,
        latent_z_conditional_calibration: None,
        score_warp_runtime: None,
        link_dev_runtime: None,
        base_link: InverseLink::Standard(StandardLink::Probit),
        frailty: gam::families::survival::lognormal_kernel::FrailtySpec::None,
    };

    // CLI source metadata: headers + per-feature ranges, no offset columns.
    let cli_payload = assemble_bernoulli_marginal_slope_payload(
        make_inputs(),
        SavedModelSourceMetadata {
            training_headers: vec!["z".to_string()],
            training_feature_ranges: Some(vec![(0.0, 4.0)]),
            offset_column: None,
            noise_offset_column: None,
        },
    )
    .unwrap_or_else(|e| panic!("{} failed: {:?}", "CLI-shaped payload", e));

    // FFI source metadata: headers only (no ranges), offset columns present.
    let ffi_payload = assemble_bernoulli_marginal_slope_payload(
        make_inputs(),
        SavedModelSourceMetadata {
            training_headers: vec!["z".to_string()],
            training_feature_ranges: None,
            offset_column: Some("off".to_string()),
            noise_offset_column: Some("noff".to_string()),
        },
    )
    .unwrap_or_else(|e| panic!("{} failed: {:?}", "FFI-shaped payload", e));

    // The semantic mirror fields the marginal-slope contract depends on must
    // match exactly between the two routes — this is what used to drift.
    assert_eq!(cli_payload.slope_formula, ffi_payload.slope_formula);
    assert_eq!(cli_payload.slope_formulas, ffi_payload.slope_formulas);
    assert_eq!(cli_payload.z_column, ffi_payload.z_column);
    assert_eq!(cli_payload.z_columns, ffi_payload.z_columns);
    assert_eq!(cli_payload.baseline_slope, ffi_payload.baseline_slope);
    assert_eq!(
        cli_payload.baseline_slopes,
        ffi_payload.baseline_slopes
    );
    assert_eq!(cli_payload.marginal_baseline, ffi_payload.marginal_baseline);
    // `TermCollectionSpec` is not `PartialEq`; the resolved-termspec
    // singular/vector mirrors are covered by the full serialized snapshot
    // equality at the end of this test.
    assert_eq!(
        cli_payload.latent_z_normalization,
        ffi_payload.latent_z_normalization
    );
    assert_eq!(cli_payload.latent_measure, ffi_payload.latent_measure);

    // The vector mirror fields must be the singletons of their scalar peers
    // — the core assembler is the single place that guarantees this.
    assert_eq!(
        cli_payload.slope_formulas.as_deref(),
        Some([cli_payload.slope_formula.clone().unwrap()].as_slice())
    );
    assert_eq!(
        cli_payload.z_columns.as_deref(),
        Some([cli_payload.z_column.clone().unwrap()].as_slice())
    );
    assert_eq!(
        cli_payload.baseline_slopes.as_deref(),
        Some([cli_payload.baseline_slope.unwrap()].as_slice())
    );

    // Full snapshot parity: serialize both, normalize away the
    // deliberately source-specific fields, and require byte equality.
    let mut cli_json = serde_json::to_value(&cli_payload)
        .unwrap_or_else(|e| panic!("{} failed: {:?}", "serialize CLI payload", e));
    let mut ffi_json = serde_json::to_value(&ffi_payload)
        .unwrap_or_else(|e| panic!("{} failed: {:?}", "serialize FFI payload", e));
    for json in [&mut cli_json, &mut ffi_json] {
        let obj = json
            .as_object_mut()
            .unwrap_or_else(|| panic!("{} failed", "payload serializes to an object"));
        obj.remove("training_feature_ranges");
        obj.remove("offset_column");
        obj.remove("noise_offset_column");
    }
    assert_eq!(
        cli_json, ffi_json,
        "CLI- and FFI-shaped marginal-slope payloads diverged in their semantic contract"
    );
}

#[test]
fn saved_bernoulli_marginal_slope_prediction_replays_latent_z_normalization() {
    let td = tempdir().unwrap_or_else(|e| panic!("{} failed: {:?}", "tempdir", e));
    let model_path = td.path().join("model.json");
    let data_path = td.path().join("predict.csv");
    let out_path = td.path().join("pred.csv");
    let fit_result = compact_saved_multiblock_fit_result(
        vec![
            FittedBlock {
                beta: array![0.0],
                role: BlockRole::Mean,
                edf: 1.0,
                lambdas: Array1::zeros(0),
            },
            FittedBlock {
                beta: array![0.0],
                role: BlockRole::Scale,
                edf: 1.0,
                lambdas: Array1::zeros(0),
            },
        ],
        Array1::zeros(0),
        1.0,
        // Minimal beta_covariance to satisfy the saved-model invariant
        // (`needs_covariance` for nonlinear families): the test exercises
        // latent-z normalization replay, not covariance accuracy, so the
        // identity is fine.
        Some(Array2::eye(2)),
        None,
        None,
        SavedFitSummary {
            likelihood_family: Some(LikelihoodSpec::new(
                ResponseFamily::Binomial,
                InverseLink::Standard(StandardLink::Probit),
            )),
            likelihood_scale: LikelihoodScaleMetadata::FixedDispersion { phi: 1.0 },
            log_likelihood_normalization: LogLikelihoodNormalization::UserProvided,
            ..saved_fit_summary_fixture()
        },
    );
    let model = super::build_bernoulli_marginal_slope_saved_model(
        "y ~ 1".to_string(),
        DataSchema {
            columns: vec![SchemaColumn {
                name: "z".to_string(),
                kind: ColumnKindTag::Continuous,
                levels: vec![],
            }],
        },
        "y ~ 1".to_string(),
        "z".to_string(),
        vec!["z".to_string()],
        vec![(0.0, 4.0)],
        empty_termspec(),
        empty_termspec(),
        fit_result,
        // Block-0 ("Mean") has a single coefficient — no influence absorber
        // is present in the fixture, so p_marginal == block-0 width = 1.
        1,
        0.0,
        1.0,
        SavedLatentZNormalization { mean: 1.0, sd: 2.0 },
        LatentMeasureKind::StandardNormal,
        None,
        None,
        None,
        None,
        InverseLink::Standard(StandardLink::Probit),
        gam::families::survival::lognormal_kernel::FrailtySpec::None,
    )
    .unwrap_or_else(|e| {
        panic!(
            "{} failed: {:?}",
            "build bernoulli marginal-slope saved model", e
        )
    });
    write_model_json(&model_path, &model)
        .unwrap_or_else(|e| panic!("{} failed: {:?}", "write saved marginal-slope model", e));
    fs::write(&data_path, "z\n3.0\n")
        .unwrap_or_else(|e| panic!("{} failed: {:?}", "write prediction data", e));

    run_predict(PredictArgs {
        model: model_path,
        new_data: data_path,
        out: out_path.clone(),
        offset_column: None,
        noise_offset_column: None,
        id_column: None,
        uncertainty: false,
        level: 0.95,
        covariance_mode: Some(InferenceCovarianceMode::SmoothingCorrected),
        mode: PredictModeArg::Map,
        no_bias_correction: false,
    })
    .unwrap_or_else(|e| {
        panic!(
            "{} failed: {:?}",
            "saved marginal-slope predict should succeed", e
        )
    });

    let predicted = csv_mean_at(&out_path, 0);
    let expected = normal_cdf(1.0);
    assert!(
        (predicted - expected).abs() <= 1e-12,
        "saved marginal-slope prediction should use normalized z: predicted={predicted}, expected={expected}"
    );
}

#[test]
fn saved_marginal_slope_models_require_latent_z_normalization() {
    let mut bernoulli = super::build_bernoulli_marginal_slope_saved_model(
        "y ~ 1".to_string(),
        DataSchema { columns: vec![] },
        "y ~ 1".to_string(),
        "z".to_string(),
        vec![],
        vec![],
        empty_termspec(),
        empty_termspec(),
        core_saved_fit_result(
            array![0.0],
            Array1::zeros(0),
            1.0,
            None,
            None,
            saved_fit_summary_fixture(),
        ),
        // Single marginal coefficient, no influence absorber → truncation
        // is a no-op (p_marginal == block-0 width).
        1,
        0.0,
        0.0,
        SavedLatentZNormalization { mean: 0.0, sd: 1.0 },
        LatentMeasureKind::StandardNormal,
        None,
        None,
        None,
        None,
        InverseLink::Standard(StandardLink::Probit),
        gam::families::survival::lognormal_kernel::FrailtySpec::None,
    )
    .unwrap_or_else(|e| {
        panic!(
            "{} failed: {:?}",
            "build bernoulli marginal-slope saved model", e
        )
    })
    .payload()
    .clone();
    bernoulli.latent_z_normalization = None;
    let err = SavedModel::from_payload(bernoulli)
        .validate_for_persistence()
        .expect_err("bernoulli marginal-slope payload without z normalization should fail");
    assert!(err.to_string().contains("latent_z_normalization"));

    let mut survival = test_payload(
        "Surv(entry, exit, event) ~ 1",
        ModelKind::Survival,
        FittedFamily::Survival {
            likelihood: LikelihoodSpec::new(
                ResponseFamily::RoystonParmar,
                InverseLink::Standard(StandardLink::Identity),
            ),
            survival_likelihood: Some("marginal-slope".to_string()),
            survival_distribution: Some(ResidualDistribution::Gaussian),
            frailty: gam::families::survival::lognormal_kernel::FrailtySpec::None,
        },
        "survival",
    );
    survival.fit_result = Some(core_saved_fit_result(
        array![0.0],
        Array1::zeros(0),
        1.0,
        None,
        None,
        saved_fit_summary_fixture(),
    ));
    survival.data_schema = Some(DataSchema { columns: vec![] });
    survival.set_training_feature_metadata(vec![], vec![]);
    survival.resolved_termspec = Some(empty_termspec());
    survival.resolved_termspec_noise = Some(empty_termspec());
    survival.slope_formula = Some("1".to_string());
    survival.z_column = Some("z".to_string());
    survival.baseline_slope = Some(0.0);
    survival.survival_entry = Some("entry".to_string());
    survival.survival_exit = Some("exit".to_string());
    survival.survival_event = Some("event".to_string());
    survival.survival_likelihood = Some("marginal-slope".to_string());
    let err = SavedModel::from_payload(survival)
        .validate_for_persistence()
        .expect_err("survival marginal-slope payload without z normalization should fail");
    assert!(err.to_string().contains("latent_z_normalization"));
}

#[test]
fn parse_survival_formula_allows_timewiggle_and_linkwiggle_together() {
    let parsed = parse_formula(
            "Surv(entry, exit, event) ~ x + timewiggle(degree=3, internal_knots=5) + linkwiggle(degree=4, internal_knots=6)",
        )
        .unwrap_or_else(|e| panic!("{} failed: {:?}", "formula should parse", e));
    // Pin the parsed fields per block: is_some()-only would let the two
    // wiggle configs cross-contaminate undetected.
    let timewiggle = parsed.timewiggle.expect("timewiggle parsed");
    assert_eq!(timewiggle.degree, 3);
    assert_eq!(timewiggle.num_internal_knots, 5);
    let linkwiggle = parsed.linkwiggle.expect("linkwiggle parsed");
    assert_eq!(linkwiggle.degree, 4);
    assert_eq!(linkwiggle.num_internal_knots, 6);
}

#[test]
fn parse_link_formula_config_extracts_link_and_inits() {
    let parsed = parse_formula(
            "y ~ x + link(type=sas, sas_init=\"0.1,-0.2\", rho=\"0.3\", beta_logistic_init=\"0.0,0.0\")",
        )
        .unwrap_or_else(|e| panic!("{} failed: {:?}", "formula", e));
    let cfg = parsed
        .linkspec
        .unwrap_or_else(|| panic!("{} failed", "expected link formula config"));
    assert_eq!(cfg.link, "sas");
    assert_eq!(cfg.sas_init.as_deref(), Some("0.1,-0.2"));
    assert_eq!(cfg.mixture_rho.as_deref(), Some("0.3"));
    assert_eq!(cfg.beta_logistic_init.as_deref(), Some("0.0,0.0"));
}

#[test]
fn parse_survmodel_formula_config_extractsspec_and_distribution() {
    let parsed = parse_formula("__survival__ ~ x + survmodel(spec=crude, distribution=gaussian)")
        .unwrap_or_else(|e| panic!("{} failed: {:?}", "formula", e));
    let cfg = parsed
        .survivalspec
        .unwrap_or_else(|| panic!("{} failed", "expected survival formula config"));
    assert_eq!(cfg.spec.as_deref(), Some("crude"));
    assert_eq!(cfg.survival_distribution.as_deref(), Some("gaussian"));
}

#[test]
fn parse_duchon_order_accepts_supportedvalues() {
    let options = BTreeMap::new();
    assert_eq!(
        parse_duchon_order(&options)
            .unwrap_or_else(|e| panic!("{} failed: {:?}", "default Duchon order", e)),
        DuchonNullspaceOrder::Linear
    );

    let mut linear = BTreeMap::new();
    linear.insert("order".to_string(), "1".to_string());
    assert_eq!(
        parse_duchon_order(&linear)
            .unwrap_or_else(|e| panic!("{} failed: {:?}", "linear Duchon order", e)),
        DuchonNullspaceOrder::Linear
    );
}

#[test]
fn parse_duchon_order_accepts_higher_polynomial_degrees_and_rejects_malformedvalues() {
    let mut quadratic = BTreeMap::new();
    quadratic.insert("order".to_string(), "2".to_string());
    assert_eq!(
        parse_duchon_order(&quadratic)
            .unwrap_or_else(|e| panic!("{} failed: {:?}", "quadratic Duchon order", e)),
        DuchonNullspaceOrder::Degree(2)
    );

    let mut malformed = BTreeMap::new();
    malformed.insert("order".to_string(), "linear".to_string());
    let malformed_err =
        parse_duchon_order(&malformed).expect_err("malformed Duchon order should fail");
    assert!(malformed_err.contains("invalid Duchon order"));
}

#[test]
fn parse_formula_retains_explicit_duchon_power_and_order_options() {
    let parsed = parse_formula("y ~ s(pc1, type=duchon, centers=12, power=0, order=1)")
        .unwrap_or_else(|e| panic!("{} failed: {:?}", "formula", e));
    match &parsed.terms[0] {
        ParsedTerm::Smooth { options, .. } => {
            assert_eq!(options.get("power").map(String::as_str), Some("0"));
            assert_eq!(options.get("order").map(String::as_str), Some("1"));
        }
        other => panic!("expected smooth term, got {other:?}"),
    }
}

#[test]
fn build_termspec_rejects_duchon_double_penalty_option() {
    let parsed = parse_formula("y ~ s(pc1, pc2, type=duchon, centers=8, double_penalty=true)")
        .unwrap_or_else(|e| {
            panic!(
                "{} failed: {:?}",
                "formula should parse before basis validation", e
            )
        });
    let ds = Dataset {
        headers: vec!["pc1".to_string(), "pc2".to_string()],
        values: array![[0.1, 0.2], [0.2, 0.3], [0.3, 0.4]],
        schema: DataSchema {
            columns: vec![
                SchemaColumn {
                    name: "pc1".to_string(),
                    kind: ColumnKindTag::Continuous,
                    levels: vec![],
                },
                SchemaColumn {
                    name: "pc2".to_string(),
                    kind: ColumnKindTag::Continuous,
                    levels: vec![],
                },
            ],
        },
        column_kinds: vec![ColumnKindTag::Continuous, ColumnKindTag::Continuous],
    };
    let col_map = HashMap::from([("pc1".to_string(), 0usize), ("pc2".to_string(), 1usize)]);
    let mut inference_notes = Vec::<String>::new();
    let err = super::build_termspec(
        &parsed.terms,
        &ds,
        &col_map,
        &mut inference_notes,
        &gam::ResourcePolicy::default_library(),
    )
    .expect_err("Duchon double_penalty should be rejected");
    assert!(err.to_string().contains("does not support double_penalty"));
    assert!(inference_notes.is_empty());
}

#[test]
fn build_termspec_honors_explicit_duchon_power_and_builds_well_posed() {
    // PgsCalibration's defaults expand into
    // `duchon(pc1, pc2, pc3, pc4, centers=N, order=1, power=1, length_scale=1)`.
    //
    // Historical intent (now superseded): when an operator-penalty triple
    // (mass + tension + stiffness) is active, D2 collocation requires
    // `2(p+s) > d+2`, and an early design escalated the explicit *power* to the
    // minimum admissible `s`. That contract is UNSOUND: at `d=4` it lifts `s`
    // to 2, giving `2s = 4 = d`, which VIOLATES the pure-Duchon conditional-
    // positive-definiteness gate `2s < d`. Commit f59909437 (#1817, "Auto-raise
    // Duchon nullspace order to satisfy operator collocation margin") replaced
    // power-escalation with nullspace-ORDER escalation at basis-build time
    // (`duchon_order_for_operator_margin`, unit-tested in `duchon_kernel_math`):
    // lifting `p` is monotone-safe (`2s < d` is untouched, so it can never
    // invalidate a power the user already satisfied).
    //
    // At the term-spec layer the explicit power/order are honored VERBATIM, and
    // the formula builder sets every operator penalty `Disabled` for a
    // formula-built duchon (`term_builder`), so there is no D2 collocation
    // constraint to trip in the first place — `power=1` is already well-posed.
    // This test therefore asserts the *current* contract: the explicit power is
    // honored (not silently bumped to 2) and the design builds well-posed rather
    // than emitting the old opaque "Duchon D2 collocation requires …" reject
    // that once broke every PgsCalibration fit.
    let formula = "y ~ s(pc1, pc2, pc3, pc4, type=duchon, centers=8, order=1, \
                       power=1, length_scale=1)";
    let parsed = parse_formula(formula)
        .unwrap_or_else(|e| panic!("{} failed: {:?}", "formula should parse", e));
    let ds = Dataset {
        headers: vec![
            "pc1".to_string(),
            "pc2".to_string(),
            "pc3".to_string(),
            "pc4".to_string(),
        ],
        values: array![
            [0.10, 0.20, 0.30, 0.40],
            [0.15, 0.25, 0.35, 0.45],
            [0.20, 0.30, 0.40, 0.50],
            [0.25, 0.35, 0.45, 0.55],
            [0.30, 0.40, 0.50, 0.60],
            [0.35, 0.45, 0.55, 0.65],
            [0.40, 0.50, 0.60, 0.70],
            [0.45, 0.55, 0.65, 0.75],
            [0.50, 0.60, 0.70, 0.80],
            [0.55, 0.65, 0.75, 0.85],
        ],
        schema: DataSchema {
            columns: vec![
                SchemaColumn {
                    name: "pc1".to_string(),
                    kind: ColumnKindTag::Continuous,
                    levels: vec![],
                },
                SchemaColumn {
                    name: "pc2".to_string(),
                    kind: ColumnKindTag::Continuous,
                    levels: vec![],
                },
                SchemaColumn {
                    name: "pc3".to_string(),
                    kind: ColumnKindTag::Continuous,
                    levels: vec![],
                },
                SchemaColumn {
                    name: "pc4".to_string(),
                    kind: ColumnKindTag::Continuous,
                    levels: vec![],
                },
            ],
        },
        column_kinds: vec![
            ColumnKindTag::Continuous,
            ColumnKindTag::Continuous,
            ColumnKindTag::Continuous,
            ColumnKindTag::Continuous,
        ],
    };
    let col_map = HashMap::from([
        ("pc1".to_string(), 0usize),
        ("pc2".to_string(), 1usize),
        ("pc3".to_string(), 2usize),
        ("pc4".to_string(), 3usize),
    ]);
    let mut inference_notes = Vec::<String>::new();
    let spec = super::build_termspec(
        &parsed.terms,
        &ds,
        &col_map,
        &mut inference_notes,
        &gam::ResourcePolicy::default_library(),
    )
    .unwrap_or_else(|e| {
        panic!(
            "{} failed: {:?}",
            "explicit power=1 must be honored, not rejected", e
        )
    });
    assert_eq!(spec.smooth_terms.len(), 1);
    match &spec.smooth_terms[0].basis {
        gam::smooth::SmoothBasisSpec::Duchon { spec: duchon, .. } => {
            assert_eq!(
                duchon.power, 1.0,
                "explicit power=1 must be honored verbatim (order, not power, is \
                     escalated for the operator collocation margin — #1817): got power={}",
                duchon.power
            );
            assert_eq!(
                duchon.nullspace_order,
                gam::basis::DuchonNullspaceOrder::Linear,
                "user-requested nullspace order=Linear must be preserved at the spec layer",
            );
        }
        other => panic!("expected Duchon basis, got {other:?}"),
    }
    // The end-to-end contract that mattered for PgsCalibration: explicit power=1
    // builds a well-posed design instead of the opaque "Duchon D2 collocation
    // requires …" reject (basis-time nullspace-order escalation, #1817).
    let design = build_term_collection_design(ds.values.view(), &spec).unwrap_or_else(|e| {
        panic!(
            "{} failed: {:?}",
            "explicit power=1 duchon must build a well-posed design, not reject", e
        )
    });
    assert_eq!(design.smooth.terms.len(), 1, "one built smooth term");
    // Power is honored, never silently bumped to 2 (which would violate 2s < d).
    assert!(
        !inference_notes.iter().any(|note| note.contains("power=2")),
        "no power-escalation note should be emitted — power is honored verbatim: got {inference_notes:?}"
    );
}

#[test]
fn survival_prediction_csv_includes_explicit_semantics_columns() {
    let mut path = std::env::temp_dir();
    let ts = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_else(|e| panic!("{} failed: {:?}", "clock", e))
        .as_nanos();
    path.push(format!("gam_survival_pred_schema_{ts}.csv"));

    let eta: Array1<f64> = array![0.5, -0.25];
    let surv = eta.mapv(|v| (-v.exp()).exp().clamp(0.0, 1.0));
    write_survival_prediction_csv(&path, eta.view(), surv.view(), None, None, None)
        .unwrap_or_else(|e| panic!("{} failed: {:?}", "write survival prediction csv", e));

    let text =
        fs::read_to_string(&path).unwrap_or_else(|e| panic!("{} failed: {:?}", "read csv", e));
    let header = text.lines().next().unwrap_or("");
    assert_eq!(
        header, "eta,survival_prob,failure_prob,risk_score",
        "survival output schema changed unexpectedly"
    );

    remove_temp_file(&path);
}

#[test]
fn survival_binary_prediction_csv_includes_explicit_semantics_columns() {
    let mut path = std::env::temp_dir();
    let ts = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_else(|e| panic!("{} failed: {:?}", "clock", e))
        .as_nanos();
    path.push(format!("gam_survival_binary_pred_schema_{ts}.csv"));

    let eta: Array1<f64> = array![0.5, -0.25];
    let event = array![0.7, 0.2];
    write_survival_binary_prediction_csv(&path, eta.view(), event.view(), None, None, None)
        .unwrap_or_else(|e| panic!("{} failed: {:?}", "write survival binary prediction csv", e));

    let text =
        fs::read_to_string(&path).unwrap_or_else(|e| panic!("{} failed: {:?}", "read csv", e));
    let header = text.lines().next().unwrap_or("");
    assert_eq!(
        header, "eta,mean,event_prob,failure_prob,survival_prob,risk_score",
        "survival binary output schema changed unexpectedly"
    );

    remove_temp_file(&path);
}

#[test]
fn survival_prediction_csv_emits_bounds_without_std_error() {
    // Contract invariant: when a caller supplies interval bounds without
    // `eta_se` (e.g. latent-window survival predictions: see
    // SavedLatentWindowKind::Survival::write_predictions), the writer must
    // still emit mean_lower / mean_upper columns instead of silently
    // discarding them.
    let mut path = std::env::temp_dir();
    let ts = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_else(|e| panic!("{} failed: {:?}", "clock", e))
        .as_nanos();
    path.push(format!("gam_survival_pred_bounds_only_{ts}.csv"));

    let eta: Array1<f64> = array![0.5, -0.25];
    let surv = eta.mapv(|v| (-v.exp()).exp().clamp(0.0, 1.0));
    let lower = array![0.3, 0.4];
    let upper = array![0.9, 0.8];
    write_survival_prediction_csv(
        &path,
        eta.view(),
        surv.view(),
        None,
        Some(lower.view()),
        Some(upper.view()),
    )
    .unwrap_or_else(|e| {
        panic!(
            "{} failed: {:?}",
            "write survival prediction csv with bounds", e
        )
    });

    let text =
        fs::read_to_string(&path).unwrap_or_else(|e| panic!("{} failed: {:?}", "read csv", e));
    let header = text.lines().next().unwrap_or("");
    assert_eq!(
        header, "eta,survival_prob,failure_prob,risk_score,mean_lower,mean_upper",
        "survival output must include bounds when supplied without std_error",
    );

    remove_temp_file(&path);
}

#[test]
fn survival_prediction_csv_errors_on_half_supplied_bounds() {
    // Contract invariant: lower XOR upper is structurally invalid and must
    // return an error rather than produce a malformed CSV.
    let mut path = std::env::temp_dir();
    let ts = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_else(|e| panic!("{} failed: {:?}", "clock", e))
        .as_nanos();
    path.push(format!("gam_survival_pred_half_bounds_{ts}.csv"));

    let eta: Array1<f64> = array![0.0];
    let surv = array![0.5];
    let lower = array![0.1];
    let upper = array![0.9];

    let err_lower_only = write_survival_prediction_csv(
        &path,
        eta.view(),
        surv.view(),
        None,
        Some(lower.view()),
        None,
    )
    .expect_err("lower-only survival bounds must be rejected");
    assert!(
        err_lower_only
            .to_string()
            .contains("survival_upper missing"),
        "lower-only error message wrong: {err_lower_only}"
    );

    let err_upper_only = write_survival_prediction_csv(
        &path,
        eta.view(),
        surv.view(),
        None,
        None,
        Some(upper.view()),
    )
    .expect_err("upper-only survival bounds must be rejected");
    assert!(
        err_upper_only
            .to_string()
            .contains("survival_lower missing"),
        "upper-only error message wrong: {err_upper_only}"
    );

    remove_temp_file(&path);
}

#[test]
fn survival_binary_prediction_csv_emits_bounds_without_std_error() {
    // Parallel contract invariant to
    // survival_prediction_csv_emits_bounds_without_std_error: the binary
    // writer (used by SavedLatentWindowKind::EventProbability) must emit
    // mean_lower / mean_upper when the caller supplies bounds without
    // `eta_se`.
    let mut path = std::env::temp_dir();
    let ts = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_else(|e| panic!("{} failed: {:?}", "clock", e))
        .as_nanos();
    path.push(format!("gam_survival_binary_pred_bounds_only_{ts}.csv"));

    let eta: Array1<f64> = array![0.5, -0.25];
    let event = array![0.7, 0.2];
    let lower = array![0.5, 0.1];
    let upper = array![0.9, 0.4];
    write_survival_binary_prediction_csv(
        &path,
        eta.view(),
        event.view(),
        None,
        Some(lower.view()),
        Some(upper.view()),
    )
    .unwrap_or_else(|e| {
        panic!(
            "{} failed: {:?}",
            "write survival binary prediction csv with bounds", e
        )
    });

    let text =
        fs::read_to_string(&path).unwrap_or_else(|e| panic!("{} failed: {:?}", "read csv", e));
    let header = text.lines().next().unwrap_or("");
    assert_eq!(
        header, "eta,mean,event_prob,failure_prob,survival_prob,risk_score,mean_lower,mean_upper",
        "survival binary output must include bounds when supplied without std_error",
    );

    remove_temp_file(&path);
}

#[test]
fn survival_binary_prediction_csv_errors_on_half_supplied_bounds() {
    // Parallel contract invariant: lower XOR upper is structurally invalid.
    let mut path = std::env::temp_dir();
    let ts = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_else(|e| panic!("{} failed: {:?}", "clock", e))
        .as_nanos();
    path.push(format!("gam_survival_binary_pred_half_bounds_{ts}.csv"));

    let eta: Array1<f64> = array![0.0];
    let event = array![0.5];
    let lower = array![0.1];
    let upper = array![0.9];

    let err_lower_only = write_survival_binary_prediction_csv(
        &path,
        eta.view(),
        event.view(),
        None,
        Some(lower.view()),
        None,
    )
    .expect_err("lower-only binary bounds must be rejected");
    assert!(
        err_lower_only.to_string().contains("event_upper missing"),
        "lower-only binary error message wrong: {err_lower_only}"
    );

    let err_upper_only = write_survival_binary_prediction_csv(
        &path,
        eta.view(),
        event.view(),
        None,
        None,
        Some(upper.view()),
    )
    .expect_err("upper-only binary bounds must be rejected");
    assert!(
        err_upper_only.to_string().contains("event_lower missing"),
        "upper-only binary error message wrong: {err_upper_only}"
    );

    remove_temp_file(&path);
}

#[test]
fn prediction_csv_can_prepend_id_column() {
    let mut path = std::env::temp_dir();
    let ts = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_else(|e| panic!("{} failed: {:?}", "clock", e))
        .as_nanos();
    path.push(format!("gam_prediction_id_passthrough_{ts}.csv"));

    let eta = array![0.5, -0.25];
    let mean = array![0.62, 0.44];
    write_prediction_csv(&path, eta.view(), mean.view(), None, None, None)
        .unwrap_or_else(|e| panic!("{} failed: {:?}", "write prediction csv", e));
    prepend_id_column_to_prediction_csv(&path, "person_id", &["p1".to_string(), "p2".to_string()])
        .unwrap_or_else(|e| panic!("{} failed: {:?}", "prepend id column", e));

    let text = fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("{} failed: {:?}", "read prediction csv", e));
    let mut lines = text.lines();
    assert_eq!(lines.next(), Some("person_id,eta,mean"));
    assert_eq!(lines.next(), Some("p1,0.500000000000,0.620000000000"));
    assert_eq!(lines.next(), Some("p2,-0.250000000000,0.440000000000"));

    remove_temp_file(&path);
}

#[test]
fn location_scale_prediction_csv_uses_estimand_explicit_schema() {
    let mut path = std::env::temp_dir();
    let ts = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_else(|e| panic!("{} failed: {:?}", "clock", e))
        .as_nanos();
    path.push(format!("gam_gaussian_loc_scale_pred_schema_{ts}.csv"));

    let eta = array![0.5, -0.25];
    let mean = eta.clone();
    let sigma = array![0.3, 0.7];
    write_estimand_explicit_prediction_csv(
        &path,
        eta.view(),
        mean.view(),
        Some(mean.view()),
        Some(sigma.view()),
        None,
        None,
        None,
    )
    .unwrap_or_else(|e| {
        panic!(
            "{} failed: {:?}",
            "write location-scale prediction csv", e
        )
    });

    let text =
        fs::read_to_string(&path).unwrap_or_else(|e| panic!("{} failed: {:?}", "read csv", e));
    let header = text.lines().next().unwrap_or("");
    assert_eq!(
        header, "linear_predictor_plugin,mean_plugin,posterior_mean,noise_scale",
        "location-scale output must use the model-owned estimand-explicit schema"
    );

    remove_temp_file(&path);
}

#[test]
fn location_scale_map_prediction_omits_the_posterior_estimand() {
    let mut path = std::env::temp_dir();
    let ts = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_else(|e| panic!("{} failed: {:?}", "clock", e))
        .as_nanos();
    path.push(format!("gam_gaussian_loc_scale_pred_bounds_{ts}.csv"));

    let eta = array![1.0];
    let mean = array![1.0];
    let sigma = array![0.4];
    write_estimand_explicit_prediction_csv(
        &path,
        eta.view(),
        mean.view(),
        None,
        Some(sigma.view()),
        None,
        None,
        None,
    )
    .unwrap_or_else(|e| {
        panic!(
            "{} failed: {:?}",
            "write location-scale MAP prediction csv", e
        )
    });

    let text =
        fs::read_to_string(&path).unwrap_or_else(|e| panic!("{} failed: {:?}", "read csv", e));
    let header = text.lines().next().unwrap_or("");
    assert_eq!(
        header, "linear_predictor_plugin,mean_plugin,noise_scale",
        "MAP output must not relabel the plug-in mean as a posterior estimand"
    );

    remove_temp_file(&path);
}

#[test]
fn location_scale_prediction_csv_names_posterior_uncertainty_explicitly() {
    let mut path = std::env::temp_dir();
    let ts = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_else(|e| panic!("{} failed: {:?}", "clock", e))
        .as_nanos();
    path.push(format!("gam_gaussian_loc_scale_pred_se_{ts}.csv"));

    let eta = array![1.0];
    let mean = array![1.0];
    let sigma = array![0.4];
    let std_error = array![0.3];
    let mean_lower = array![0.2];
    let mean_upper = array![1.8];
    write_estimand_explicit_prediction_csv(
        &path,
        eta.view(),
        mean.view(),
        Some(mean.view()),
        Some(sigma.view()),
        Some(std_error.view()),
        Some(mean_lower.view()),
        Some(mean_upper.view()),
    )
    .unwrap_or_else(|e| {
        panic!(
            "{} failed: {:?}",
            "write location-scale prediction csv with posterior uncertainty", e
        )
    });

    let text =
        fs::read_to_string(&path).unwrap_or_else(|e| panic!("{} failed: {:?}", "read csv", e));
    let mut lines = text.lines();
    assert_eq!(
        lines.next(),
        Some(
            "linear_predictor_plugin,mean_plugin,posterior_mean,noise_scale,posterior_mean_standard_error,posterior_mean_lower,posterior_mean_upper"
        ),
        "location-scale uncertainty output must name the posterior estimand"
    );
    assert_eq!(
        lines.next(),
        Some(
            "1.000000000000,1.000000000000,1.000000000000,0.400000000000,0.300000000000,0.200000000000,1.800000000000"
        )
    );

    remove_temp_file(&path);
}

#[test]
fn gaussian_location_scale_generate_restores_sigma_to_response_units() {
    // The persisted log-σ coefficient is already in RAW response units:
    // `rescale_gaussian_location_scale_to_raw` shifts that intercept by
    // `+ln(response_scale)` at fit time, so `exp(η_ls)` carries one factor of
    // the scale on its own. The soft floor sits OUTSIDE the exp and cannot ride
    // that shift, so it is the only piece still standardized and the only piece
    // multiplied here:
    //
    //   σ_raw = response_scale·LOGB_SIGMA_FLOOR + exp(η_ls)
    //         = response_scale·(LOGB_SIGMA_FLOOR + exp(η_internal))
    //
    // Scaling the whole `(floor + exp(η_ls))` instead would apply
    // `response_scale` twice on the exp term and break σ's response-scale
    // equivariance -- the defect behind #1874/#1928, and precisely what this
    // fixture used to assert. See `GaussianLocationScalePredictor::compute_sigma`.
    //
    // Pick the input so σ exits at 2.0 exactly under the real convention:
    // exp(η_ls) = 2.0 − 8·0.01 = 1.92.
    let model = intercept_only_gaussian_location_scale_model(-3.0, (1.92f64).ln(), 8.0);
    let data = ndarray::Array2::<f64>::zeros((2, 0));
    let headers = vec![];
    let col_map = HashMap::new();
    let spec = super::run_generate_unified(
        &model,
        data.view(),
        &col_map,
        Some(&headers),
        &Array1::zeros(data.nrows()),
        &Array1::zeros(data.nrows()),
        false,
    )
    .unwrap_or_else(|e| panic!("{} failed: {:?}", "generate gaussian location-scale", e));
    assert_eq!(spec.mean.to_vec(), vec![-3.0, -3.0]);
    match spec.noise {
        gam::generative::NoiseModel::Gaussian { sigma } => {
            assert!(sigma.iter().all(|&v| (v - 2.0).abs() < 1e-12));
        }
        _ => panic!("expected Gaussian noise model"),
    }
}

#[test]
fn parse_survival_time_basis_accepts_ispline() {
    let args = SurvivalArgs {
        data: std::path::PathBuf::from("dummy.csv"),
        entry: Some("entry".to_string()),
        exit: "exit".to_string(),
        event: "event".to_string(),
        formula: "1".to_string(),
        predict_noise: None,
        survival_likelihood: "transformation".to_string(),
        survival_distribution: "gaussian".to_string(),
        link: None,
        mixture_rho: None,
        sas_init: None,
        beta_logistic_init: None,
        survival_time_anchor: None,
        baseline_target: "linear".to_string(),
        baseline_scale: None,
        baseline_shape: None,
        baseline_rate: None,
        baseline_makeham: None,
        time_basis: "ispline".to_string(),
        time_degree: 2,
        time_num_internal_knots: 6,
        time_smooth_lambda: 1e-2,
        ridge_lambda: 1e-6,
        threshold_time_k: None,
        threshold_time_degree: 3,
        sigma_time_k: None,
        sigma_time_degree: 3,
        slope_time_k: None,
        slope_time_degree: 3,
        scale_dimensions: false,
        pilot_subsample_threshold: 0,
        out: None,
        slope_formula: None,
        z_column: None,
        weights_column: None,
        offset_column: None,
        noise_offset_column: None,
        frailty_kind: None,
        frailty_sd: None,
        hazard_loading: None,
        persistent_warm_start_store: None,
    };
    let cfg = parse_survival_time_basis_config(
        &args.time_basis,
        args.time_degree,
        args.time_num_internal_knots,
        args.time_smooth_lambda,
    )
    .unwrap_or_else(|e| panic!("{} failed: {:?}", "parse ispline time basis", e));
    assert!(matches!(cfg, SurvivalTimeBasisConfig::ISpline { .. }));
}

#[test]
fn parse_survival_time_basis_rejects_nonstructural_bases() {
    let mut args = SurvivalArgs {
        data: std::path::PathBuf::from("dummy.csv"),
        entry: Some("entry".to_string()),
        exit: "exit".to_string(),
        event: "event".to_string(),
        formula: "1".to_string(),
        predict_noise: None,
        survival_likelihood: "transformation".to_string(),
        survival_distribution: "gaussian".to_string(),
        link: None,
        mixture_rho: None,
        sas_init: None,
        beta_logistic_init: None,
        survival_time_anchor: None,
        baseline_target: "linear".to_string(),
        baseline_scale: None,
        baseline_shape: None,
        baseline_rate: None,
        baseline_makeham: None,
        time_basis: "linear".to_string(),
        time_degree: 2,
        time_num_internal_knots: 6,
        time_smooth_lambda: 1e-2,
        ridge_lambda: 1e-6,
        threshold_time_k: None,
        threshold_time_degree: 3,
        sigma_time_k: None,
        sigma_time_degree: 3,
        slope_time_k: None,
        slope_time_degree: 3,
        scale_dimensions: false,
        pilot_subsample_threshold: 0,
        out: None,
        slope_formula: None,
        z_column: None,
        weights_column: None,
        offset_column: None,
        noise_offset_column: None,
        frailty_kind: None,
        frailty_sd: None,
        hazard_loading: None,
        persistent_warm_start_store: None,
    };
    let err = parse_survival_time_basis_config(
        &args.time_basis,
        args.time_degree,
        args.time_num_internal_knots,
        args.time_smooth_lambda,
    )
    .expect_err("linear survival time basis should be rejected");
    assert!(err.contains("structural"));
    assert!(err.contains("ispline"));
    assert!(err.contains("survival semantics"));

    args.time_basis = "bspline".to_string();
    let err = parse_survival_time_basis_config(
        &args.time_basis,
        args.time_degree,
        args.time_num_internal_knots,
        args.time_smooth_lambda,
    )
    .expect_err("bspline survival time basis should be rejected");
    assert!(err.contains("structural"));
    assert!(err.contains("ispline"));
    assert!(err.contains("non-monotone"));
}

#[test]
fn structural_survival_basis_error_explainswhy_bspline_is_rejected() {
    let err = super::require_structural_survival_time_basis("bspline", "survival benchmark")
        .expect_err("bspline should be rejected");
    assert!(err.contains("survival benchmark"));
    assert!(err.contains("Only `ispline` is accepted"));
    assert!(err.contains("monotone cumulative time effect"));
    assert!(err.contains("survival semantics"));
    assert!(err.contains("`--time-basis ispline`"));
}

#[test]
fn structural_survival_basis_detection_is_ispline_only() {
    assert!(
        gam::families::survival::construction::survival_basis_supports_structural_monotonicity(
            "ispline"
        )
    );
    assert!(
        gam::families::survival::construction::survival_basis_supports_structural_monotonicity(
            "ISPLINE"
        )
    );
    assert!(
        !gam::families::survival::construction::survival_basis_supports_structural_monotonicity(
            "linear"
        )
    );
    assert!(
        !gam::families::survival::construction::survival_basis_supports_structural_monotonicity(
            "bspline"
        )
    );
}

#[test]
fn normalize_survival_time_pair_rejects_invalid_raw_times() {
    let err = super::normalize_survival_time_pair(1.0, f64::NAN, 2)
        .expect_err("non-finite exit time should fail");
    assert!(err.contains("non-finite survival times at row 3"));

    let err = super::normalize_survival_time_pair(-1.0, 2.0, 4)
        .expect_err("negative entry time should fail");
    assert!(err.contains("negative survival times at row 5"));
}

#[test]
fn saved_survival_model_requires_time_basis_metadata() {
    let mut payload = test_payload(
        "Surv(start, stop, event) ~ x",
        ModelKind::Survival,
        FittedFamily::Survival {
            likelihood: LikelihoodSpec::new(
                ResponseFamily::RoystonParmar,
                InverseLink::Standard(StandardLink::Identity),
            ),
            survival_likelihood: Some("transformation".to_string()),
            survival_distribution: Some(ResidualDistribution::Gaussian),
            frailty: gam::families::survival::lognormal_kernel::FrailtySpec::None,
        },
        "survival",
    );
    payload.survival_entry = Some("start".to_string());
    payload.survival_exit = Some("stop".to_string());
    payload.survival_event = Some("event".to_string());
    payload.survivalspec = Some("net".to_string());
    payload.survival_baseline_target = Some("linear".to_string());
    payload.survival_likelihood = Some("transformation".to_string());
    payload.survival_distribution = Some(ResidualDistribution::Gaussian);
    let model = SavedModel::from_payload(payload);

    let err = super::load_survival_time_basis_config_from_model(&model)
        .expect_err("survival model without basis metadata should fail");
    assert!(err.to_string().contains("missing survival_time_basis"));
}

#[test]
fn saved_survival_flex_exit_helper_matches_rigid_when_deviations_absent() {
    let q_exit = array![-0.4, 0.2, 1.1];
    let slope = array![-0.7, 0.0, 0.9];
    let z = array![-1.0, 0.5, 1.3];

    let (eta, mean) =
            saved_survival_marginal_slope_test_support::predict_saved_survival_marginal_slope_flex_exit(
            &q_exit, &slope, &z, None, None, None, None, None,
        )
        .unwrap_or_else(|e| panic!("{} failed: {:?}", "flex exit helper should reduce to rigid model", e));

    for i in 0..q_exit.len() {
        let c = (1.0 + slope[i] * slope[i]).sqrt();
        let expected_eta = q_exit[i] * c + slope[i] * z[i];
        let expected_mean = super::normal_cdf(expected_eta);
        assert!(
            (eta[i] - expected_eta).abs() <= 1e-10,
            "row {i}: eta mismatch: got {}, expected {}",
            eta[i],
            expected_eta
        );
        assert!(
            (mean[i] - expected_mean).abs() <= 1e-10,
            "row {i}: mean mismatch: got {}, expected {}",
            mean[i],
            expected_mean
        );
    }
}

#[test]
fn saved_prediction_runtime_validates_survival_anchored_deviation_runtime() {
    let mut payload = test_payload(
        "Surv(start, stop, event) ~ x",
        ModelKind::Survival,
        FittedFamily::Survival {
            likelihood: LikelihoodSpec::new(
                ResponseFamily::RoystonParmar,
                InverseLink::Standard(StandardLink::Identity),
            ),
            survival_likelihood: Some("marginal-slope".to_string()),
            survival_distribution: Some(ResidualDistribution::Gaussian),
            frailty: gam::families::survival::lognormal_kernel::FrailtySpec::None,
        },
        "survival",
    );
    payload.score_warp_runtime = Some(SavedCompiledFlexBlock {
        kernel: "BadKernel".to_string(),
        breakpoints: vec![-1.0, 1.0],
        basis_dim: 2,
        span_c0: vec![vec![0.0, 0.0]],
        span_c1: vec![vec![0.0, 0.0]],
        span_c2: vec![vec![0.0, 0.0]],
        span_c3: vec![vec![0.0, 0.0]],
        anchor_correction: None,
        anchor_components: Vec::new(),
    });
    let model = SavedModel::from_payload(payload);

    let err = model
        .saved_prediction_runtime()
        .expect_err("invalid survival anchored deviation runtime should fail validation");
    assert!(err.to_string().contains("unsupported kernel"));
    assert!(err.to_string().contains("anchored score-warp"));
}

#[test]
fn saved_survival_flex_exit_helper_with_zero_scorewarp_matches_rigid() {
    let saved_runtime = SavedCompiledFlexBlock {
        kernel: gam::families::cubic_cell_kernel::ANCHORED_DEVIATION_KERNEL.to_string(),
        breakpoints: vec![-1.0, 1.0],
        basis_dim: 1,
        span_c0: vec![vec![0.0]],
        span_c1: vec![vec![0.0]],
        span_c2: vec![vec![0.0]],
        span_c3: vec![vec![0.0]],
        anchor_correction: None,
        anchor_components: Vec::new(),
    };
    let zero_beta = Array1::zeros(saved_runtime.basis_dim);

    let q_exit = array![-0.8, 0.4];
    let slope = array![0.3, -1.1];
    let z = array![0.2, -0.7];

    let (eta, mean) =
            saved_survival_marginal_slope_test_support::predict_saved_survival_marginal_slope_flex_exit(
            &q_exit,
            &slope,
            &z,
            None,
            Some(&saved_runtime),
            Some(&zero_beta),
            None,
            None,
        )
        .unwrap_or_else(|e| panic!("{} failed: {:?}", "zero score-warp should still predict", e));

    for i in 0..q_exit.len() {
        let c = (1.0 + slope[i] * slope[i]).sqrt();
        let expected_eta = q_exit[i] * c + slope[i] * z[i];
        let expected_mean = super::normal_cdf(expected_eta);
        assert!((eta[i] - expected_eta).abs() <= 1e-10);
        assert!((mean[i] - expected_mean).abs() <= 1e-10);
    }
}

#[test]
fn saved_survival_flex_exit_helper_matches_gaussian_frailty_rigid_formula() {
    let q_exit = array![-0.8, 0.4];
    let slope = array![0.3, -1.1];
    let z = array![0.2, -0.7];
    let gaussian_frailty_sd = Some(0.9);

    let (eta, mean) =
            saved_survival_marginal_slope_test_support::predict_saved_survival_marginal_slope_flex_exit(
            &q_exit,
            &slope,
            &z,
            gaussian_frailty_sd,
            None,
            None,
            None,
            None,
        )
        .unwrap_or_else(|e| panic!("{} failed: {:?}", "rigid frailty path should predict", e));

    let scale = gam::families::marginal_slope_shared::probit_frailty_scale(gaussian_frailty_sd);
    for i in 0..q_exit.len() {
        let sb = scale * slope[i];
        let c = (1.0 + sb * sb).sqrt();
        let expected_eta = q_exit[i] * c + sb * z[i];
        let expected_mean = super::normal_cdf(expected_eta);
        assert!((eta[i] - expected_eta).abs() <= 1e-10);
        assert!((mean[i] - expected_mean).abs() <= 1e-10);
    }
}

#[test]
fn saved_survival_marginal_slope_predictor_keeps_operator_backed_designs_lazy() {
    #[derive(Clone)]
    struct NoDensifyTestOperator {
        dense: Array2<f64>,
    }

    impl LinearOperator for NoDensifyTestOperator {
        fn nrows(&self) -> usize {
            self.dense.nrows()
        }

        fn ncols(&self) -> usize {
            self.dense.ncols()
        }

        fn apply(&self, vector: &Array1<f64>) -> Array1<f64> {
            self.dense.dot(vector)
        }

        fn apply_transpose(&self, vector: &Array1<f64>) -> Array1<f64> {
            self.dense.t().dot(vector)
        }

        fn diag_xtw_x(&self, weights: &Array1<f64>) -> Result<Array2<f64>, String> {
            if weights.len() != self.nrows() {
                return Err(format!(
                    "NoDensifyTestOperator weight length mismatch: weights={}, nrows={}",
                    weights.len(),
                    self.nrows()
                ));
            }
            let p = self.ncols();
            let mut out = Array2::<f64>::zeros((p, p));
            for i in 0..self.nrows() {
                let w = weights[i].max(0.0);
                for a in 0..p {
                    let xia = self.dense[[i, a]];
                    for b in 0..p {
                        out[[a, b]] += w * xia * self.dense[[i, b]];
                    }
                }
            }
            Ok(out)
        }
    }

    impl DenseDesignOperator for NoDensifyTestOperator {
        fn row_chunk_into(
            &self,
            rows: Range<usize>,
            mut out: ArrayViewMut2<'_, f64>,
        ) -> Result<(), MatrixMaterializationError> {
            out.assign(&self.dense.slice(s![rows, ..]));
            Ok(())
        }

        fn to_dense(&self) -> Array2<f64> {
            panic!("saved survival marginal-slope predictor should not densify this operator")
        }
    }

    fn nondensify_design(dense: Array2<f64>) -> DesignMatrix {
        DesignMatrix::from(DenseDesignMatrix::from(Arc::new(NoDensifyTestOperator {
            dense,
        })))
    }

    let time_entry_dense = array![[0.1], [0.4]];
    let time_exit_dense = array![[0.2], [0.6]];
    let time_deriv_dense = array![[1.0], [1.0]];
    let cov_dense = array![[1.0, -0.5], [0.3, 0.8]];
    let slope_dense = array![[0.7], [-0.2]];
    let time_build = gam::families::survival::construction::SurvivalTimeBuildOutput {
        x_entry_time: nondensify_design(time_entry_dense.clone()),
        x_exit_time: nondensify_design(time_exit_dense.clone()),
        x_derivative_time: nondensify_design(time_deriv_dense.clone()),
        penalties: vec![],
        nullspace_dims: vec![],
        basisname: "ispline".to_string(),
        degree: Some(1),
        knots: None,
        keep_cols: None,
        smooth_lambda: None,
    };
    let fit_saved = compact_saved_multiblock_fit_result(
        vec![
            FittedBlock {
                beta: array![0.6],
                role: BlockRole::Mean,
                edf: 1.0,
                lambdas: Array1::zeros(0),
            },
            FittedBlock {
                beta: array![0.5, -0.25],
                role: BlockRole::Mean,
                edf: 2.0,
                lambdas: Array1::zeros(0),
            },
            FittedBlock {
                beta: array![0.8],
                role: BlockRole::Scale,
                edf: 1.0,
                lambdas: Array1::zeros(0),
            },
        ],
        Array1::zeros(0),
        1.0,
        None,
        None,
        None,
        saved_fit_summary_fixture(),
    );

    let mut payload = test_payload(
        "Surv(entry, exit, event) ~ x1 + x2",
        ModelKind::Survival,
        FittedFamily::Survival {
            likelihood: LikelihoodSpec::new(
                ResponseFamily::RoystonParmar,
                InverseLink::Standard(StandardLink::Identity),
            ),
            survival_likelihood: Some("marginal-slope".to_string()),
            survival_distribution: Some(ResidualDistribution::Gaussian),
            frailty: gam::families::survival::lognormal_kernel::FrailtySpec::None,
        },
        "survival",
    );
    payload.fit_result = Some(fit_saved.clone());
    payload.unified = Some(fit_saved.clone());
    payload.survival_entry = Some("entry".to_string());
    payload.survival_exit = Some("exit".to_string());
    payload.survival_event = Some("event".to_string());
    payload.survivalspec = Some("net".to_string());
    payload.survival_baseline_target = Some("linear".to_string());
    payload.survival_likelihood = Some("marginal-slope".to_string());
    payload.survival_distribution = Some(ResidualDistribution::Gaussian);
    payload.survival_time_basis = Some("ispline".to_string());
    payload.slope_formula = Some("ls ~ 1".to_string());
    payload.z_column = Some("z".to_string());
    payload.latent_z_normalization = Some(SavedLatentZNormalization { mean: 0.0, sd: 1.0 });
    // Marginal-slope saved-model invariant requires `latent_measure` to be
    // populated; the standard-normal default matches the test's frozen
    // latent-z policy.
    payload.latent_measure = Some(LatentMeasureKind::StandardNormal);
    // The marginal-slope saved-model invariant also requires an exact
    // latent-score covariance before it will validate a payload. These tests
    // exercise latent-z replay and lazy operator-backed designs, not covariance
    // accuracy -- the z normalization under test is `latent_z_normalization`,
    // a separate field -- so a minimal 1x1 unit covariance satisfies the
    // invariant without standing in for anything the assertions read. Same
    // rationale as the minimal `beta_covariance` above.
    payload.survival_marginal_slope_score_covariance = Some(vec![vec![1.0]]);
    payload.baseline_slope = Some(0.0);
    payload.link = Some(InverseLink::Standard(StandardLink::Probit));
    let model = SavedModel::from_payload(payload);

    let cov_design = nondensify_design(cov_dense.clone());
    let slope_design = nondensify_design(slope_dense.clone());
    let z = array![-1.0, 0.5];
    let eta_offset_entry = array![0.05, -0.02];
    let eta_offset_exit = array![0.1, -0.03];
    let derivative_offset_exit = array![0.0, 0.0];
    let primary_offset = array![0.2, -0.15];
    let noise_offset = array![0.04, -0.01];

    let (predictor, pred_input, _) = super::build_saved_survival_marginal_slope_predictor(
        &model,
        &fit_saved,
        "z",
        &z,
        &cov_design,
        &slope_design,
        &time_build,
        &eta_offset_entry,
        &eta_offset_exit,
        &derivative_offset_exit,
        &primary_offset,
        &noise_offset,
    )
    .unwrap_or_else(|e| {
        panic!(
            "{} failed: {:?}",
            "operator-backed saved survival predictor should build without densifying", e
        )
    });

    assert!(
        pred_input.design.as_dense_ref().is_none(),
        "saved survival predictor should keep the rebuilt q design operator-backed"
    );
    assert!(
        pred_input
            .design_noise
            .as_ref()
            .unwrap_or_else(|| panic!("{} failed", "slope design"))
            .as_dense_ref()
            .is_none(),
        "saved survival predictor should keep the slope design operator-backed"
    );

    let prediction = predictor
        .predict_plugin_response(&pred_input)
        .unwrap_or_else(|e| {
            panic!(
                "{} failed: {:?}",
                "operator-backed saved survival predictor should score", e
            )
        });
    let q_exit = time_exit_dense.dot(&array![0.6])
        + cov_dense.dot(&array![0.5, -0.25])
        + &eta_offset_exit
        + &primary_offset;
    let slope = slope_dense.dot(&array![0.8]) + &noise_offset;
    let (expected_eta, expected_mean) =
            saved_survival_marginal_slope_test_support::predict_saved_survival_marginal_slope_flex_exit(
                &q_exit,
                &slope,
                &z,
                None,
                None,
                None,
                None,
                None,
            )
            .unwrap_or_else(|e| panic!("{} failed: {:?}", "closed-form saved survival helper should evaluate", e));

    for i in 0..expected_eta.len() {
        assert!(
            (prediction.eta[i] - expected_eta[i]).abs() <= 1e-10,
            "row {i}: eta mismatch: got {}, expected {}",
            prediction.eta[i],
            expected_eta[i]
        );
        assert!(
            (prediction.mean[i] - expected_mean[i]).abs() <= 1e-10,
            "row {i}: mean mismatch: got {}, expected {}",
            prediction.mean[i],
            expected_mean[i]
        );
    }
}

#[test]
fn saved_survival_marginal_slope_prediction_replays_latent_z_normalization() {
    let fit_saved = compact_saved_multiblock_fit_result(
        vec![
            FittedBlock {
                beta: array![0.4],
                role: BlockRole::Mean,
                edf: 1.0,
                lambdas: Array1::zeros(0),
            },
            FittedBlock {
                beta: Array1::zeros(0),
                role: BlockRole::Mean,
                edf: 0.0,
                lambdas: Array1::zeros(0),
            },
            FittedBlock {
                beta: array![1.0],
                role: BlockRole::Scale,
                edf: 1.0,
                lambdas: Array1::zeros(0),
            },
        ],
        Array1::zeros(0),
        1.0,
        // Minimal beta_covariance: total beta = 1 + 0 + 1 = 2. Saved-model
        // invariant requires either a covariance or a penalized Hessian for
        // nonlinear families; the test exercises latent-z replay, not
        // covariance accuracy.
        Some(Array2::eye(2)),
        None,
        None,
        saved_fit_summary_fixture(),
    );

    let mut payload = test_payload(
        "Surv(entry, exit, event) ~ 1",
        ModelKind::Survival,
        FittedFamily::Survival {
            likelihood: LikelihoodSpec::new(
                ResponseFamily::RoystonParmar,
                InverseLink::Standard(StandardLink::Identity),
            ),
            survival_likelihood: Some("marginal-slope".to_string()),
            survival_distribution: Some(ResidualDistribution::Gaussian),
            frailty: gam::families::survival::lognormal_kernel::FrailtySpec::None,
        },
        "survival",
    );
    payload.fit_result = Some(fit_saved.clone());
    payload.unified = Some(fit_saved.clone());
    payload.data_schema = Some(DataSchema {
        columns: vec![
            SchemaColumn {
                name: "entry".to_string(),
                kind: ColumnKindTag::Continuous,
                levels: vec![],
            },
            SchemaColumn {
                name: "exit".to_string(),
                kind: ColumnKindTag::Continuous,
                levels: vec![],
            },
            SchemaColumn {
                name: "event".to_string(),
                kind: ColumnKindTag::Binary,
                levels: vec![],
            },
            SchemaColumn {
                name: "z".to_string(),
                kind: ColumnKindTag::Continuous,
                levels: vec![],
            },
        ],
    });
    payload.set_training_feature_metadata(
        vec![
            "entry".to_string(),
            "exit".to_string(),
            "event".to_string(),
            "z".to_string(),
        ],
        vec![(0.0, 0.0); 4],
    );
    payload.resolved_termspec = Some(empty_termspec());
    payload.resolved_termspec_noise = Some(empty_termspec());
    payload.resolved_slopespec = Some(empty_termspec());
    payload.survival_entry = Some("entry".to_string());
    payload.survival_exit = Some("exit".to_string());
    payload.survival_event = Some("event".to_string());
    payload.survivalspec = Some("net".to_string());
    payload.survival_baseline_target = Some("linear".to_string());
    payload.survival_likelihood = Some("marginal-slope".to_string());
    payload.survival_distribution = Some(ResidualDistribution::Gaussian);
    payload.survival_time_basis = Some("ispline".to_string());
    payload.survival_time_anchor = Some(0.0);
    payload.slope_formula = Some("1".to_string());
    payload.z_column = Some("z".to_string());
    payload.latent_z_normalization = Some(SavedLatentZNormalization { mean: 1.0, sd: 2.0 });
    // Marginal-slope saved-model invariant requires `latent_measure`; this
    // test exercises latent-z normalization replay, so a standard-normal
    // measure (the frozen default) is correct.
    payload.latent_measure = Some(LatentMeasureKind::StandardNormal);
    // The marginal-slope saved-model invariant also requires an exact
    // latent-score covariance before it will validate a payload. These tests
    // exercise latent-z replay and lazy operator-backed designs, not covariance
    // accuracy -- the z normalization under test is `latent_z_normalization`,
    // a separate field -- so a minimal 1x1 unit covariance satisfies the
    // invariant without standing in for anything the assertions read. Same
    // rationale as the minimal `beta_covariance` above.
    payload.survival_marginal_slope_score_covariance = Some(vec![vec![1.0]]);
    payload.baseline_slope = Some(0.0);
    payload.link = Some(InverseLink::Standard(StandardLink::Probit));
    let model = SavedModel::from_payload(payload);
    model.validate_for_persistence().unwrap_or_else(|e| {
        panic!(
            "{} failed: {:?}",
            "saved survival marginal-slope payload should validate", e
        )
    });

    let time_build = gam::families::survival::construction::SurvivalTimeBuildOutput {
        x_entry_time: DesignMatrix::from(array![[1.0]]),
        x_exit_time: DesignMatrix::from(array![[1.0]]),
        x_derivative_time: DesignMatrix::from(array![[1.0]]),
        penalties: vec![],
        nullspace_dims: vec![],
        basisname: "ispline".to_string(),
        degree: Some(1),
        knots: None,
        keep_cols: None,
        smooth_lambda: None,
    };
    let cov_design = DesignMatrix::from(Array2::<f64>::zeros((1, 0)));
    let slope_design = DesignMatrix::from(array![[1.0]]);
    let z_raw = array![3.0];
    let eta_offset_entry = array![0.0];
    let eta_offset_exit = array![0.0];
    let derivative_offset_exit = array![0.0];
    let primary_offset = array![0.0];
    let noise_offset = array![0.0];

    let (predictor, pred_input, _) = super::build_saved_survival_marginal_slope_predictor(
        &model,
        &fit_saved,
        "z",
        &z_raw,
        &cov_design,
        &slope_design,
        &time_build,
        &eta_offset_entry,
        &eta_offset_exit,
        &derivative_offset_exit,
        &primary_offset,
        &noise_offset,
    )
    .unwrap_or_else(|e| {
        panic!(
            "{} failed: {:?}",
            "saved survival marginal-slope predictor should build", e
        )
    });
    let prediction = predictor
        .predict_plugin_response(&pred_input)
        .unwrap_or_else(|e| {
            panic!(
                "{} failed: {:?}",
                "saved survival marginal-slope predictor should score", e
            )
        });

    let z_normalized = array![1.0];
    let (expected_eta, expected_mean) =
            saved_survival_marginal_slope_test_support::predict_saved_survival_marginal_slope_flex_exit(
                &array![0.4],
                &array![1.0],
                &z_normalized,
                None,
                None,
                None,
                None,
                None,
            )
            .unwrap_or_else(|e| panic!("{} failed: {:?}", "saved survival helper should evaluate", e));
    assert!((prediction.eta[0] - expected_eta[0]).abs() <= 1e-12);
    assert!((prediction.mean[0] - expected_mean[0]).abs() <= 1e-12);
}

#[test]
fn saved_baseline_timewiggle_components_return_none_without_metadata() {
    let eta = array![0.1, 0.2];
    let deriv = array![0.3, 0.4];
    let mut payload = test_payload(
        "Surv(entry, exit, event) ~ timewiggle(degree=3, internal_knots=5)",
        ModelKind::Survival,
        FittedFamily::Survival {
            likelihood: LikelihoodSpec::new(
                ResponseFamily::RoystonParmar,
                InverseLink::Standard(StandardLink::Identity),
            ),
            survival_likelihood: Some("transformation".to_string()),
            survival_distribution: Some(ResidualDistribution::Gaussian),
            frailty: gam::families::survival::lognormal_kernel::FrailtySpec::None,
        },
        "survival",
    );
    payload.survival_entry = Some("entry".to_string());
    payload.survival_exit = Some("exit".to_string());
    payload.survival_event = Some("event".to_string());
    payload.survivalspec = Some("net".to_string());
    payload.survival_baseline_target = Some("weibull".to_string());
    payload.survival_baseline_scale = Some(10.0);
    payload.survival_baseline_shape = Some(1.2);
    payload.survival_time_basis = Some("none".to_string());
    payload.survival_likelihood = Some("transformation".to_string());
    payload.survival_distribution = Some(ResidualDistribution::Gaussian);
    payload.set_training_feature_metadata(vec![], vec![]);
    payload.resolved_termspec = Some(empty_termspec());
    let model = SavedModel::from_payload(payload);
    let got = super::saved_baseline_timewiggle_components(&eta, &eta, &deriv, &model)
        .unwrap_or_else(|e| panic!("{} failed: {:?}", "baseline-timewiggle metadata check", e));
    assert!(got.is_none());
}

#[test]
fn run_predict_survival_supports_saved_baseline_timewiggle_model() {
    let age_entry = array![10.0, 12.0];
    let age_exit = array![20.0, 24.0];
    let baseline_cfg = SurvivalBaselineConfig {
        target: SurvivalBaselineTarget::Weibull,
        scale: Some(15.0),
        shape: Some(1.3),
        rate: None,
        makeham: None,
    };
    let (eta_entry, eta_exit, derivative_exit) =
        build_survival_baseline_offsets(&age_entry, &age_exit, &baseline_cfg)
            .unwrap_or_else(|e| panic!("{} failed: {:?}", "baseline offsets", e));
    let wiggle_cfg = parse_linkwiggle_formulaspec(
        &BTreeMap::from([
            ("degree".to_string(), "3".to_string()),
            ("internal_knots".to_string(), "4".to_string()),
        ]),
        "timewiggle(degree=3, internal_knots=4)",
    )
    .unwrap_or_else(|e| panic!("{} failed: {:?}", "baseline-timewiggle cfg", e));
    let built = build_survival_timewiggle_from_baseline(
        &eta_entry,
        &eta_exit,
        &derivative_exit,
        &wiggle_cfg,
    )
    .unwrap_or_else(|e| panic!("{} failed: {:?}", "baseline-timewiggle build", e));
    let beta = Array1::from_iter((0..built.ncols).map(|j| 0.05 * (j as f64 + 1.0)));
    let mut fit_beta = Array1::<f64>::zeros(beta.len() + 1);
    fit_beta.slice_mut(s![..beta.len()]).assign(&beta);
    let p = fit_beta.len();
    let fit_result = core_saved_fit_result(
        fit_beta,
        Array1::zeros(built.penalties.len()),
        1.0,
        Some(Array2::<f64>::eye(p)),
        None,
        saved_fit_summary_fixture(),
    );
    let mut payload = test_payload(
        "Surv(entry, exit, event) ~ timewiggle(degree=3, internal_knots=4)",
        ModelKind::Survival,
        FittedFamily::Survival {
            likelihood: LikelihoodSpec::new(
                ResponseFamily::RoystonParmar,
                InverseLink::Standard(StandardLink::Identity),
            ),
            survival_likelihood: Some("transformation".to_string()),
            survival_distribution: Some(ResidualDistribution::Gaussian),
            frailty: gam::families::survival::lognormal_kernel::FrailtySpec::None,
        },
        "survival",
    );
    payload.fit_result = Some(fit_result);
    payload.baseline_timewiggle_knots = Some(built.knots.to_vec());
    payload.baseline_timewiggle_degree = Some(built.degree);
    payload.baseline_timewiggle_penalty_orders = Some(wiggle_cfg.penalty_orders.clone());
    payload.baseline_timewiggle_double_penalty = Some(wiggle_cfg.double_penalty);
    payload.beta_baseline_timewiggle = Some(Array1::<f64>::zeros(built.ncols).to_vec());
    payload.survival_entry = Some("entry".to_string());
    payload.survival_exit = Some("exit".to_string());
    payload.survival_event = Some("event".to_string());
    payload.survivalspec = Some("net".to_string());
    payload.survival_baseline_target = Some("weibull".to_string());
    payload.survival_baseline_scale = Some(15.0);
    payload.survival_baseline_shape = Some(1.3);
    payload.survival_time_basis = Some("none".to_string());
    payload.survivalridge_lambda = Some(1e-4);
    payload.survival_likelihood = Some("transformation".to_string());
    payload.survival_distribution = Some(ResidualDistribution::Gaussian);
    payload.set_training_feature_metadata(
        vec!["entry".to_string(), "exit".to_string()],
        vec![(0.0, 0.0); 2],
    );
    payload.resolved_termspec = Some(empty_termspec());
    payload.data_schema = Some(DataSchema {
        columns: vec![
            SchemaColumn {
                name: "entry".to_string(),
                kind: ColumnKindTag::Continuous,
                levels: vec![],
            },
            SchemaColumn {
                name: "exit".to_string(),
                kind: ColumnKindTag::Continuous,
                levels: vec![],
            },
        ],
    });
    let model = SavedModel::from_payload(payload);
    let data = array![[10.0, 20.0], [12.0, 24.0]];
    let col_map = HashMap::from([("entry".to_string(), 0usize), ("exit".to_string(), 1usize)]);
    let out_dir = tempdir().unwrap_or_else(|e| panic!("{} failed: {:?}", "tempdir", e));
    let out_path = out_dir.path().join("survival_baseline_timewiggle_pred.csv");
    let args = PredictArgs {
        model: PathBuf::from("unused.model.json"),
        new_data: PathBuf::from("unused.csv"),
        out: out_path.clone(),
        offset_column: None,
        noise_offset_column: None,
        id_column: None,
        uncertainty: false,
        level: 0.95,
        covariance_mode: Some(InferenceCovarianceMode::SmoothingCorrected),
        mode: PredictModeArg::Map,
        no_bias_correction: false,
    };
    super::run_predict_survival(
        &args,
        &model,
        data.view(),
        &col_map,
        model.training_headers.as_ref(),
        &Array1::zeros(data.nrows()),
        &Array1::zeros(data.nrows()),
    )
    .unwrap_or_else(|e| panic!("{} failed: {:?}", "survival predict with timewiggle", e));
    let (_, exit_w, _) = super::saved_baseline_timewiggle_components(
        &eta_entry,
        &eta_exit,
        &derivative_exit,
        &model,
    )
    .unwrap_or_else(|e| panic!("{} failed: {:?}", "rebuild saved baseline-timewiggle", e))
    .expect("saved baseline-timewiggle metadata");
    let expected = predict_gam(
        exit_w,
        beta.view(),
        eta_exit.view(),
        LikelihoodSpec::royston_parmar(),
    )
    .unwrap_or_else(|e| panic!("{} failed: {:?}", "expected survival predict", e));

    let mut rdr = csv::Reader::from_path(&out_path)
        .unwrap_or_else(|e| panic!("{} failed: {:?}", "open prediction csv", e));
    let rows = rdr
        .deserialize::<BTreeMap<String, String>>()
        .collect::<Result<Vec<_>, _>>()
        .unwrap_or_else(|e| panic!("{} failed: {:?}", "parse prediction csv", e));
    assert_eq!(rows.len(), 2);
    for i in 0..rows.len() {
        let eta = rows[i]["eta"]
            .parse::<f64>()
            .unwrap_or_else(|e| panic!("{} failed: {:?}", "eta should parse", e));
        let survival_prob = rows[i]["survival_prob"]
            .parse::<f64>()
            .unwrap_or_else(|e| panic!("{} failed: {:?}", "survival_prob should parse", e));
        assert!(
            (eta - expected.eta[i]).abs() <= 1e-12,
            "row {i}: eta mismatch: got {eta}, expected {}",
            expected.eta[i]
        );
        let expected_survival_prob = expected.mean[i].clamp(0.0, 1.0);
        assert!(
            (survival_prob - expected_survival_prob).abs() <= 1e-12,
            "row {i}: survival_prob mismatch: got {survival_prob}, expected {expected_survival_prob}",
        );
    }
}

#[test]
fn run_predict_survival_supports_saved_latent_survival_model() {
    let data = array![[10.0, 20.0], [12.0, 24.0]];
    let age_entry = data.column(0).to_owned();
    let age_exit = data.column(1).to_owned();
    let time_cfg = gam::families::survival::construction::SurvivalTimeBasisConfig::ISpline {
        degree: 2,
        knots: Array1::zeros(0),
        keep_cols: Vec::new(),
        smooth_lambda: 1e-4,
    };
    let time_build = gam::families::survival::construction::build_survival_time_basis(
        &age_entry,
        &age_exit,
        time_cfg,
        Some((2, 1e-4)),
    )
    .unwrap_or_else(|e| {
        panic!(
            "{} failed: {:?}",
            "build latent survival test time basis", e
        )
    });
    let p_time = time_build.x_exit_time.ncols();
    let time_anchor =
        gam::families::survival::construction::survival_earliest_entry_time_anchor(&age_entry)
            .unwrap_or_else(|e| {
                panic!(
                    "{} failed: {:?}",
                    "resolve latent survival test time anchor", e
                )
            });
    let blocks = vec![
        gam::estimate::FittedBlock {
            beta: Array1::zeros(p_time),
            role: BlockRole::Time,
            edf: p_time as f64,
            lambdas: Array1::zeros(0),
        },
        gam::estimate::FittedBlock {
            beta: array![0.0],
            role: BlockRole::Mean,
            edf: 1.0,
            lambdas: Array1::zeros(0),
        },
    ];
    let fit_result = compact_saved_multiblock_fit_result(
        blocks,
        Array1::zeros(0),
        1.0,
        Some(Array2::<f64>::eye(p_time + 1)),
        None,
        None,
        saved_fit_summary_fixture(),
    );
    let mut payload = test_payload(
        "Surv(entry, exit, event) ~ 1",
        ModelKind::Survival,
        FittedFamily::LatentSurvival {
            frailty: gam::families::survival::lognormal_kernel::FrailtySpec::HazardMultiplier {
                scale: FrailtyScale::Fixed { sigma: 0.3 },
                loading: gam::families::survival::lognormal_kernel::HazardLoading::Full,
            },
        },
        "latent-survival",
    );
    payload.fit_result = Some(fit_result.clone());
    payload.unified = Some(fit_result);
    payload.survival_entry = Some("entry".to_string());
    payload.survival_exit = Some("exit".to_string());
    payload.survival_event = Some("event".to_string());
    payload.survivalspec = Some("net".to_string());
    payload.survival_baseline_target = Some("weibull".to_string());
    payload.survival_baseline_scale = Some(15.0);
    payload.survival_baseline_shape = Some(1.3);
    payload.survival_time_basis = Some("ispline".to_string());
    payload.survival_time_degree = time_build.degree;
    payload.survival_time_knots = time_build.knots.clone();
    payload.survival_time_keep_cols = time_build.keep_cols.clone();
    payload.survival_time_smooth_lambda = Some(1e-4);
    payload.survival_time_anchor = Some(time_anchor);
    payload.survival_beta_time = Some(vec![0.0; p_time]);
    payload.survival_likelihood = Some("latent".to_string());
    payload.set_training_feature_metadata(
        vec!["entry".to_string(), "exit".to_string()],
        vec![(0.0, 0.0); 2],
    );
    payload.resolved_termspec = Some(empty_termspec());
    let model = SavedModel::from_payload(payload);

    let col_map = HashMap::from([("entry".to_string(), 0usize), ("exit".to_string(), 1usize)]);
    let out_dir = tempdir().unwrap_or_else(|e| panic!("{} failed: {:?}", "tempdir", e));
    let out_path = out_dir.path().join("latent_survival_pred.csv");
    let args = PredictArgs {
        model: PathBuf::from("unused.model.json"),
        new_data: PathBuf::from("unused.csv"),
        out: out_path.clone(),
        offset_column: None,
        noise_offset_column: None,
        id_column: None,
        uncertainty: false,
        level: 0.95,
        covariance_mode: Some(InferenceCovarianceMode::SmoothingCorrected),
        mode: PredictModeArg::Map,
        no_bias_correction: false,
    };

    super::run_predict_survival(
        &args,
        &model,
        data.view(),
        &col_map,
        model.training_headers.as_ref(),
        &Array1::zeros(data.nrows()),
        &Array1::zeros(data.nrows()),
    )
    .unwrap_or_else(|e| {
        panic!(
            "{} failed: {:?}",
            "latent survival predict should succeed", e
        )
    });

    let csv = fs::read_to_string(&out_path)
        .unwrap_or_else(|e| panic!("{} failed: {:?}", "prediction csv", e));
    let lines = csv.lines().collect::<Vec<_>>();
    assert_eq!(lines.len(), 3);
    assert_eq!(lines[0], "eta,survival_prob,failure_prob,risk_score");

    let zero = Array1::zeros(data.nrows());
    let spec = generative_spec_for_saved_model(
        &model,
        SavedGenerativeInput {
            data: data.view(),
            col_map: &col_map,
            training_headers: model.training_headers.as_ref(),
            offset: &zero,
            offset_noise: &zero,
            noise_offset_supplied: false,
            prior_weights: None,
        },
    )
    .expect("saved latent survival model should expose its exact event-window law");
    assert!(
        spec.mean
            .iter()
            .all(|probability| { probability.is_finite() && (0.0..=1.0).contains(probability) })
    );
    let draws = sampleobservation_seeded_replicates(&spec, 0, 31, 2300)
        .expect("sample saved latent survival event-window law");
    assert_eq!(draws.shape(), [31, data.nrows()]);
    assert!(draws.iter().all(|value| *value == 0.0 || *value == 1.0));

    let mut binary_payload = model.payload().clone();
    binary_payload.family_state = FittedFamily::LatentBinary {
        frailty: gam::families::survival::lognormal_kernel::FrailtySpec::HazardMultiplier {
            scale: FrailtyScale::Fixed { sigma: 0.3 },
            loading: gam::families::survival::lognormal_kernel::HazardLoading::Full,
        },
    };
    binary_payload.family = "latent-binary".to_string();
    binary_payload.survival_likelihood = Some("latent-binary".to_string());
    let binary_model = SavedModel::from_payload(binary_payload);
    let binary_spec = generative_spec_for_saved_model(
        &binary_model,
        SavedGenerativeInput {
            data: data.view(),
            col_map: &col_map,
            training_headers: binary_model.training_headers.as_ref(),
            offset: &zero,
            offset_noise: &zero,
            noise_offset_supplied: false,
            prior_weights: None,
        },
    )
    .expect("saved latent-binary model should expose its exact event-window law");
    assert_eq!(spec.mean, binary_spec.mean);
}

#[test]
fn explicit_latent_binary_family_requires_matching_saved_likelihood_metadata() {
    let mut payload = test_payload(
        "Surv(entry, exit, event) ~ 1",
        ModelKind::Survival,
        FittedFamily::LatentBinary {
            frailty: gam::families::survival::lognormal_kernel::FrailtySpec::HazardMultiplier {
                scale: FrailtyScale::Fixed { sigma: 0.3 },
                loading: gam::families::survival::lognormal_kernel::HazardLoading::Full,
            },
        },
        "latent-binary",
    );
    payload.survival_likelihood = Some("latent-binary".to_string());
    let model = SavedModel::from_payload(payload);
    let mode = super::require_saved_survival_likelihood_mode(&model)
        .unwrap_or_else(|e| panic!("{} failed: {:?}", "latent-binary mode", e));
    assert_eq!(mode, SurvivalLikelihoodMode::LatentBinary);
}

#[test]
fn explicit_latent_survival_family_requires_matching_saved_likelihood_metadata() {
    let mut payload = test_payload(
        "Surv(entry, exit, event) ~ 1",
        ModelKind::Survival,
        FittedFamily::LatentSurvival {
            frailty: gam::families::survival::lognormal_kernel::FrailtySpec::HazardMultiplier {
                scale: FrailtyScale::Fixed { sigma: 0.3 },
                loading: gam::families::survival::lognormal_kernel::HazardLoading::Full,
            },
        },
        "latent-survival",
    );
    payload.survival_likelihood = Some("latent".to_string());
    let model = SavedModel::from_payload(payload);
    let mode = super::require_saved_survival_likelihood_mode(&model)
        .unwrap_or_else(|e| panic!("{} failed: {:?}", "latent mode", e));
    assert_eq!(mode, SurvivalLikelihoodMode::Latent);
}

#[test]
fn saved_baseline_timewiggle_reconstruction_keeps_requested_order_one_penalty() {
    let age_entry = array![10.0, 12.0];
    let age_exit = array![20.0, 24.0];
    let baseline_cfg = SurvivalBaselineConfig {
        target: SurvivalBaselineTarget::Weibull,
        scale: Some(15.0),
        shape: Some(1.3),
        rate: None,
        makeham: None,
    };
    let (eta_entry, eta_exit, derivative_exit) =
        build_survival_baseline_offsets(&age_entry, &age_exit, &baseline_cfg)
            .unwrap_or_else(|e| panic!("{} failed: {:?}", "baseline offsets", e));
    let wiggle_cfg = parse_linkwiggle_formulaspec(
        &BTreeMap::from([
            ("degree".to_string(), "3".to_string()),
            ("internal_knots".to_string(), "4".to_string()),
        ]),
        "timewiggle(degree=3, internal_knots=4)",
    )
    .unwrap_or_else(|e| panic!("{} failed: {:?}", "baseline-timewiggle cfg", e));
    let built = build_survival_timewiggle_from_baseline(
        &eta_entry,
        &eta_exit,
        &derivative_exit,
        &wiggle_cfg,
    )
    .unwrap_or_else(|e| panic!("{} failed: {:?}", "baseline-timewiggle build", e));
    let mut payload = test_payload(
        "Surv(entry, exit, event) ~ timewiggle(degree=3, internal_knots=4)",
        ModelKind::Survival,
        FittedFamily::Survival {
            likelihood: LikelihoodSpec::new(
                ResponseFamily::RoystonParmar,
                InverseLink::Standard(StandardLink::Identity),
            ),
            survival_likelihood: Some("transformation".to_string()),
            survival_distribution: Some(ResidualDistribution::Gaussian),
            frailty: gam::families::survival::lognormal_kernel::FrailtySpec::None,
        },
        "survival",
    );
    payload.fit_result = Some(core_saved_fit_result(
        Array1::zeros(1),
        Array1::zeros(0),
        1.0,
        None,
        None,
        saved_fit_summary_fixture(),
    ));
    payload.baseline_timewiggle_knots = Some(built.knots.to_vec());
    payload.baseline_timewiggle_degree = Some(built.degree);
    payload.baseline_timewiggle_penalty_orders = Some(vec![1, 2, 3]);
    payload.baseline_timewiggle_double_penalty = Some(false);
    payload.beta_baseline_timewiggle = Some(vec![0.0; built.ncols]);
    payload.survival_entry = Some("entry".to_string());
    payload.survival_exit = Some("exit".to_string());
    payload.survival_event = Some("event".to_string());
    payload.survivalspec = Some("net".to_string());
    payload.survival_baseline_target = Some("weibull".to_string());
    payload.survival_baseline_scale = Some(15.0);
    payload.survival_baseline_shape = Some(1.3);
    payload.survival_time_basis = Some("none".to_string());
    payload.survivalridge_lambda = Some(1e-4);
    payload.survival_likelihood = Some("transformation".to_string());
    payload.survival_distribution = Some(ResidualDistribution::Gaussian);
    payload.set_training_feature_metadata(
        vec!["entry".to_string(), "exit".to_string()],
        vec![(0.0, 0.0); 2],
    );
    payload.resolved_termspec = Some(empty_termspec());
    let model = SavedModel::from_payload(payload);

    let saved_cfg = gam::sample::saved_baseline_timewiggle_spec(&model)
        .unwrap_or_else(|e| panic!("{} failed: {:?}", "saved baseline-timewiggle spec", e))
        .expect("timewiggle metadata");
    let wiggle_knots = Array1::from_vec(
        model
            .baseline_timewiggle_knots
            .clone()
            .unwrap_or_else(|| panic!("{} failed", "saved knots")),
    );
    let mut seed = Array1::<f64>::zeros(2 * eta_entry.len());
    for i in 0..eta_entry.len() {
        seed[i] = eta_entry[i];
        seed[eta_entry.len() + i] = eta_exit[i];
    }
    let (primary_order, extra_orders) =
        gam::families::wiggle::split_wiggle_penalty_orders(2, &saved_cfg.penalty_orders)
            .expect("saved positive penalty orders are valid");
    let mut derivative_orders = Vec::with_capacity(1 + extra_orders.len());
    derivative_orders.push(primary_order);
    derivative_orders.extend(extra_orders.iter().copied());
    let block = gam::families::wiggle::buildwiggle_block_input_from_orders(
        seed.view(),
        &wiggle_knots,
        saved_cfg.degree,
        &derivative_orders,
        saved_cfg.double_penalty,
    )
    .unwrap_or_else(|e| {
        panic!(
            "{} failed: {:?}",
            "rebuild saved baseline-timewiggle block", e
        )
    });
    assert_eq!(wiggle_cfg.penalty_orders, vec![1, 2, 3]);
    assert_eq!(saved_cfg.penalty_orders, vec![1, 2, 3]);
    assert_eq!(primary_order, 1);
    assert_eq!(extra_orders, vec![2, 3]);
    assert_eq!(block.penalties.len(), 3);
    // Anchored I-spline value basis (#2306): the anchoring removes the constant
    // direction, so the order-m roughness nullity is m-1 -- the order-1 penalty
    // is positive definite (nullity 0). The old [1, 2, 3] encoded the
    // UNANCHORED convention. `gam-models`'
    // `survival_timewiggle_keeps_requested_order_one_penalty` was already moved
    // to [0, 1, 2] with this reasoning; this copy of the same assertion was
    // left behind.
    assert_eq!(block.nullspace_dims, vec![0, 1, 2]);
}

#[test]
fn parse_survival_baseline_accepts_gompertz_makeham() {
    let cfg = parse_survival_baseline_config(
        "gompertz-makeham",
        None,
        Some(0.08),
        Some(0.015),
        Some(0.002),
    )
    .unwrap_or_else(|e| panic!("{} failed: {:?}", "parse gompertz-makeham baseline", e));
    assert_eq!(cfg.target, SurvivalBaselineTarget::GompertzMakeham);
    assert_eq!(cfg.shape, Some(0.08));
    assert_eq!(cfg.rate, Some(0.015));
    assert_eq!(cfg.makeham, Some(0.002));
}

#[test]
fn parse_survival_baseline_seeds_missing_gompertz_makeham_terms() {
    let cfg =
        parse_survival_baseline_config("gompertz-makeham", None, Some(0.08), Some(0.015), None)
            .unwrap_or_else(|e| {
                panic!(
                    "{} failed: {:?}",
                    "missing makeham should seed a default", e
                )
            });
    assert_eq!(cfg.target, SurvivalBaselineTarget::GompertzMakeham);
    assert_eq!(cfg.shape, Some(0.08));
    assert_eq!(cfg.rate, Some(0.015));
    assert_eq!(cfg.makeham, Some(0.5));
}

#[test]
fn evaluate_survival_baseline_matches_gompertz_makeham_formula() {
    let cfg = SurvivalBaselineConfig {
        target: SurvivalBaselineTarget::GompertzMakeham,
        scale: None,
        shape: Some(0.07),
        rate: Some(0.012),
        makeham: Some(0.003),
    };
    let age = 11.5;
    let (eta, derivative) = evaluate_survival_baseline(age, &cfg)
        .unwrap_or_else(|e| panic!("{} failed: {:?}", "evaluate gompertz-makeham baseline", e));
    let shape = cfg.shape.unwrap_or_else(|| panic!("{} failed", "shape"));
    let rate = cfg.rate.unwrap_or_else(|| panic!("{} failed", "rate"));
    let makeham = cfg
        .makeham
        .unwrap_or_else(|| panic!("{} failed", "makeham"));
    let cumulative_hazard = makeham * age + (rate / shape) * ((shape * age).exp() - 1.0);
    let expected_eta = cumulative_hazard.ln();
    let expected_derivative = (makeham + rate * (shape * age).exp()) / cumulative_hazard;
    assert!((eta - expected_eta).abs() <= 1e-12);
    assert!((derivative - expected_derivative).abs() <= 1e-12);
}

#[test]
fn evaluate_survival_baseline_handles_nearzero_gompertz_makeham_shape() {
    let cfg = SurvivalBaselineConfig {
        target: SurvivalBaselineTarget::GompertzMakeham,
        scale: None,
        shape: Some(1e-14),
        rate: Some(0.012),
        makeham: Some(0.003),
    };
    let age = 11.5;
    let (eta, derivative) = evaluate_survival_baseline(age, &cfg)
        .unwrap_or_else(|e| panic!("{} failed: {:?}", "evaluate near-zero gompertz-makeham", e));
    let cumulative_hazard = (cfg.rate.unwrap_or_else(|| panic!("{} failed", "rate"))
        + cfg.makeham.expect("makeham"))
        * age;
    let expected_eta = cumulative_hazard.ln();
    let expected_derivative = 1.0 / age;
    assert!((eta - expected_eta).abs() <= 1e-12);
    assert!((derivative - expected_derivative).abs() <= 1e-12);
}

#[test]
fn parse_link_choice_rejects_flexible_beta_logistic() {
    let err = parse_link_choice(Some("flexible(beta-logistic)"), false)
        .expect_err("flexible(beta-logistic) should be rejected");
    assert!(
        err.to_string()
            .contains("does not support sas/beta-logistic")
    );
}

#[test]
fn parse_link_choice_rejects_flexible_sas() {
    let err = parse_link_choice(Some("flexible(sas)"), false)
        .expect_err("flexible(sas) should be rejected");
    assert!(
        err.to_string()
            .contains("does not support sas/beta-logistic")
    );
}

#[test]
fn parse_link_choice_rejects_flexible_blended_link() {
    let err = parse_link_choice(Some("flexible(blended(logit,probit))"), false)
        .expect_err("flexible(blended(...)) should be rejected");
    assert!(
        err.to_string()
            .contains("does not support blended(...)/mixture(...)")
    );
}

#[test]
fn parse_link_choice_accepts_binomial_aliases() {
    let probit = parse_link_choice(Some("binomial-probit"), false)
        .unwrap_or_else(|e| panic!("{} failed: {:?}", "parse binomial-probit", e))
        .expect("link choice");
    assert!(matches!(probit.link, LinkFunction::Probit));
    assert!(probit.mixture_components.is_none());

    let logit = parse_link_choice(Some("binomial-logit"), false)
        .unwrap_or_else(|e| panic!("{} failed: {:?}", "parse binomial-logit", e))
        .expect("link choice");
    assert!(matches!(logit.link, LinkFunction::Logit));
    assert!(logit.mixture_components.is_none());

    let cloglog = parse_link_choice(Some("binomial-cloglog"), false)
        .unwrap_or_else(|e| panic!("{} failed: {:?}", "parse binomial-cloglog", e))
        .expect("link choice");
    assert!(matches!(cloglog.link, LinkFunction::CLogLog));
    assert!(cloglog.mixture_components.is_none());
}

#[test]
fn parse_link_choice_flexible_shorthand_defaults_to_probit() {
    let choice = parse_link_choice(None, true)
        .unwrap_or_else(|e| panic!("{} failed: {:?}", "parse flexible shorthand", e))
        .expect("link choice");
    assert!(matches!(choice.mode, LinkMode::Flexible));
    assert!(matches!(choice.link, LinkFunction::Probit));
    assert!(choice.mixture_components.is_none());
}

fn parse_survival_inverse_link(args: &SurvivalArgs) -> Result<InverseLink, String> {
    parse_config_survival_inverse_link(SurvivalInverseLinkInput {
        link: args.link.as_deref(),
        mixture_rho: args.mixture_rho.as_deref(),
        sas_init: args.sas_init.as_deref(),
        beta_logistic_init: args.beta_logistic_init.as_deref(),
        survival_distribution: &args.survival_distribution,
    })
}

#[test]
fn parse_survival_inverse_link_accepts_sas_init() {
    let mut args = SurvivalArgs {
        data: std::path::PathBuf::from("dummy.csv"),
        entry: Some("entry".to_string()),
        exit: "exit".to_string(),
        event: "event".to_string(),
        formula: "1".to_string(),
        predict_noise: None,
        survival_likelihood: "location-scale".to_string(),
        survival_distribution: "gaussian".to_string(),
        link: Some("logit".to_string()),
        mixture_rho: None,
        sas_init: None,
        beta_logistic_init: None,
        survival_time_anchor: None,
        baseline_target: "linear".to_string(),
        baseline_scale: None,
        baseline_shape: None,
        baseline_rate: None,
        baseline_makeham: None,
        time_basis: "linear".to_string(),
        time_degree: 3,
        time_num_internal_knots: 8,
        time_smooth_lambda: 1e-2,
        ridge_lambda: 1e-6,
        threshold_time_k: None,
        threshold_time_degree: 3,
        sigma_time_k: None,
        sigma_time_degree: 3,
        slope_time_k: None,
        slope_time_degree: 3,
        scale_dimensions: false,
        pilot_subsample_threshold: 0,
        out: None,
        slope_formula: None,
        z_column: None,
        weights_column: None,
        offset_column: None,
        noise_offset_column: None,
        frailty_kind: None,
        frailty_sd: None,
        hazard_loading: None,
        persistent_warm_start_store: None,
    };
    args.link = Some("sas".to_string());
    args.sas_init = Some("0.15,-0.70".to_string());
    let link = parse_survival_inverse_link(&args)
        .unwrap_or_else(|e| panic!("{} failed: {:?}", "sas survival link", e));
    match link {
        InverseLink::Sas(state) => {
            assert!((state.epsilon - 0.15).abs() < 1e-12);
            assert!((state.log_delta - (-0.70)).abs() < 1e-12);
        }
        other => panic!("expected sas inverse link, got {other:?}"),
    }
}

/// Default `SurvivalArgs` shape shared by the
/// `parse_survival_inverse_link_*` test set. Real fields are picked so
/// the inverse-link validator path is the only thing being tested:
/// `formula = "1"`, single-knot time basis, no frailty, no extra
/// columns. Tests override `link` / `sas_init` / `beta_logistic_init`
/// (and occasionally one more) to exercise the validation branches.
fn survival_args_for_inverse_link_test() -> SurvivalArgs {
    SurvivalArgs {
        data: std::path::PathBuf::from("dummy.csv"),
        entry: Some("entry".to_string()),
        exit: "exit".to_string(),
        event: "event".to_string(),
        formula: "1".to_string(),
        predict_noise: None,
        survival_likelihood: "location-scale".to_string(),
        survival_distribution: "gaussian".to_string(),
        link: Some("logit".to_string()),
        mixture_rho: None,
        sas_init: None,
        beta_logistic_init: None,
        survival_time_anchor: None,
        baseline_target: "linear".to_string(),
        baseline_scale: None,
        baseline_shape: None,
        baseline_rate: None,
        baseline_makeham: None,
        time_basis: "linear".to_string(),
        time_degree: 3,
        time_num_internal_knots: 8,
        time_smooth_lambda: 1e-2,
        ridge_lambda: 1e-6,
        threshold_time_k: None,
        threshold_time_degree: 3,
        sigma_time_k: None,
        sigma_time_degree: 3,
        slope_time_k: None,
        slope_time_degree: 3,
        scale_dimensions: false,
        pilot_subsample_threshold: 0,
        out: None,
        slope_formula: None,
        z_column: None,
        weights_column: None,
        offset_column: None,
        noise_offset_column: None,
        frailty_kind: None,
        frailty_sd: None,
        hazard_loading: None,
        persistent_warm_start_store: None,
    }
}

/// Shared test driver for the four "init-flag rejected when link does
/// not match" guards. Builds the default args, overrides
/// (`link`, `sas_init`, `beta_logistic_init`), runs the validator, and
/// pins the per-case expected error substring.
fn assert_inverse_link_init_rejected(
    link: &str,
    sas_init: Option<&str>,
    beta_logistic_init: Option<&str>,
    expected_error_substr: &str,
) {
    let mut args = survival_args_for_inverse_link_test();
    args.link = Some(link.to_string());
    args.sas_init = sas_init.map(String::from);
    args.beta_logistic_init = beta_logistic_init.map(String::from);
    let err = parse_survival_inverse_link(&args).expect_err("expected arg validation error");
    assert!(
        err.contains(expected_error_substr),
        "validation error '{err}' does not contain '{expected_error_substr}'"
    );
}

#[test]
fn parse_survival_inverse_link_rejects_beta_logistic_init_for_sas() {
    assert_inverse_link_init_rejected(
        "sas",
        None,
        Some("0.1,0.2"),
        "--beta-logistic-init requires --link beta-logistic",
    );
}

#[test]
fn parse_survival_inverse_link_rejects_sas_init_for_logit() {
    assert_inverse_link_init_rejected(
        "logit",
        Some("0.1,0.2"),
        None,
        "--sas-init requires --link sas",
    );
}

#[test]
fn parse_survival_inverse_link_accepts_beta_logistic_init() {
    let mut args = survival_args_for_inverse_link_test();
    args.link = Some("beta-logistic".to_string());
    args.beta_logistic_init = Some("0.25,0.80".to_string());
    let link = parse_survival_inverse_link(&args)
        .unwrap_or_else(|e| panic!("{} failed: {:?}", "beta-logistic survival link", e));
    match link {
        InverseLink::BetaLogistic(state) => {
            assert!((state.epsilon - 0.25).abs() < 1e-12);
            assert!((state.log_delta - 0.80).abs() < 1e-12);
        }
        other => panic!("expected beta-logistic inverse link, got {other:?}"),
    }
}

#[test]
fn parse_survival_inverse_link_rejects_sas_init_for_beta_logistic() {
    assert_inverse_link_init_rejected(
        "beta-logistic",
        Some("0.1,0.2"),
        None,
        "--sas-init requires --link sas",
    );
}

#[test]
fn parse_survival_inverse_link_rejects_beta_logistic_init_for_logit() {
    assert_inverse_link_init_rejected(
        "logit",
        None,
        Some("0.1,0.2"),
        "--beta-logistic-init requires --link beta-logistic",
    );
}

#[test]
fn parse_survival_inverse_link_supports_loglog_and_cauchit() {
    let mut args = SurvivalArgs {
        data: std::path::PathBuf::from("dummy.csv"),
        entry: Some("entry".to_string()),
        exit: "exit".to_string(),
        event: "event".to_string(),
        formula: "1".to_string(),
        predict_noise: None,
        survival_likelihood: "location-scale".to_string(),
        survival_distribution: "gaussian".to_string(),
        link: Some("loglog".to_string()),
        mixture_rho: None,
        sas_init: None,
        beta_logistic_init: None,
        survival_time_anchor: None,
        baseline_target: "linear".to_string(),
        baseline_scale: None,
        baseline_shape: None,
        baseline_rate: None,
        baseline_makeham: None,
        time_basis: "linear".to_string(),
        time_degree: 3,
        time_num_internal_knots: 8,
        time_smooth_lambda: 1e-2,
        ridge_lambda: 1e-6,
        threshold_time_k: None,
        threshold_time_degree: 3,
        sigma_time_k: None,
        sigma_time_degree: 3,
        slope_time_k: None,
        slope_time_degree: 3,
        scale_dimensions: false,
        pilot_subsample_threshold: 0,
        out: None,
        slope_formula: None,
        z_column: None,
        weights_column: None,
        offset_column: None,
        noise_offset_column: None,
        frailty_kind: None,
        frailty_sd: None,
        hazard_loading: None,
        persistent_warm_start_store: None,
    };
    // `loglog` and `cauchit` are supported survival --link values (issue #1829). Each
    // routes through a single-component MixtureLinkSpec (weight 1.0) — a pure link, not
    // an under-identified blend — so `validate_mixturespec` accepts it (the anchor
    // requirement only applies to genuine multi-component blends). Numeric mu checks
    // live in `parse_survival_inverse_link_accepts_loglog_and_cauchit`.
    let loglog = parse_survival_inverse_link(&args)
        .expect("loglog survival link parses to a single-component mixture");
    match &loglog {
        InverseLink::Mixture(state) => {
            assert_eq!(state.components, vec![LinkComponent::LogLog]);
            assert!((state.pi[0] - 1.0).abs() < 1e-12);
        }
        other => panic!("expected loglog to route through a mixture, got {other:?}"),
    }

    args.link = Some("cauchit".to_string());
    let cauchit = parse_survival_inverse_link(&args)
        .expect("cauchit survival link parses to a single-component mixture");
    match &cauchit {
        InverseLink::Mixture(state) => {
            assert_eq!(state.components, vec![LinkComponent::Cauchit]);
            assert!((state.pi[0] - 1.0).abs() < 1e-12);
        }
        other => panic!("expected cauchit to route through a mixture, got {other:?}"),
    }
}

#[test]
fn flexible_link_injects_default_linkwiggle_config() {
    let link_choice = parse_link_choice(Some("flexible(logit)"), false)
        .unwrap_or_else(|e| panic!("{} failed: {:?}", "parse flexible link choice", e));
    let cfg = effectivelinkwiggle_formulaspec(None, link_choice.as_ref())
        .unwrap_or_else(|| panic!("{} failed", "flexible link should inject wiggle config"));
    let defaults = WigglePenaltyConfig::cubic_triple_operator_default();
    assert_eq!(cfg.degree, 3);
    assert_eq!(cfg.num_internal_knots, defaults.num_internal_knots);
    assert_eq!(cfg.penalty_orders, vec![1, 2, 3]);
    assert!(cfg.double_penalty);
}

#[test]
fn parse_survival_inverse_link_accepts_flexible_standard_links() {
    let mut args = SurvivalArgs {
        data: std::path::PathBuf::from("dummy.csv"),
        entry: Some("entry".to_string()),
        exit: "exit".to_string(),
        event: "event".to_string(),
        formula: "1".to_string(),
        predict_noise: None,
        survival_likelihood: "location-scale".to_string(),
        survival_distribution: "gaussian".to_string(),
        link: Some("logit".to_string()),
        mixture_rho: None,
        sas_init: None,
        beta_logistic_init: None,
        survival_time_anchor: None,
        baseline_target: "linear".to_string(),
        baseline_scale: None,
        baseline_shape: None,
        baseline_rate: None,
        baseline_makeham: None,
        time_basis: "linear".to_string(),
        time_degree: 3,
        time_num_internal_knots: 8,
        time_smooth_lambda: 1e-2,
        ridge_lambda: 1e-6,
        threshold_time_k: None,
        threshold_time_degree: 3,
        sigma_time_k: None,
        sigma_time_degree: 3,
        slope_time_k: None,
        slope_time_degree: 3,
        scale_dimensions: false,
        pilot_subsample_threshold: 0,
        out: None,
        slope_formula: None,
        z_column: None,
        weights_column: None,
        offset_column: None,
        noise_offset_column: None,
        frailty_kind: None,
        frailty_sd: None,
        hazard_loading: None,
        persistent_warm_start_store: None,
    };
    args.link = Some("flexible(logit)".to_string());
    let link = parse_survival_inverse_link(&args)
        .unwrap_or_else(|e| panic!("{} failed: {:?}", "flexible survival link", e));
    assert!(matches!(link, InverseLink::Standard(StandardLink::Logit)));
}

#[test]
fn parse_survival_inverse_link_rejects_flexible_blended_links() {
    let mut args = SurvivalArgs {
        data: std::path::PathBuf::from("dummy.csv"),
        entry: Some("entry".to_string()),
        exit: "exit".to_string(),
        event: "event".to_string(),
        formula: "1".to_string(),
        predict_noise: None,
        survival_likelihood: "location-scale".to_string(),
        survival_distribution: "gaussian".to_string(),
        link: Some("logit".to_string()),
        mixture_rho: None,
        sas_init: None,
        beta_logistic_init: None,
        survival_time_anchor: None,
        baseline_target: "linear".to_string(),
        baseline_scale: None,
        baseline_shape: None,
        baseline_rate: None,
        baseline_makeham: None,
        time_basis: "linear".to_string(),
        time_degree: 3,
        time_num_internal_knots: 8,
        time_smooth_lambda: 1e-2,
        ridge_lambda: 1e-6,
        threshold_time_k: None,
        threshold_time_degree: 3,
        sigma_time_k: None,
        sigma_time_degree: 3,
        slope_time_k: None,
        slope_time_degree: 3,
        scale_dimensions: false,
        pilot_subsample_threshold: 0,
        out: None,
        slope_formula: None,
        z_column: None,
        weights_column: None,
        offset_column: None,
        noise_offset_column: None,
        frailty_kind: None,
        frailty_sd: None,
        hazard_loading: None,
        persistent_warm_start_store: None,
    };
    args.link = Some("flexible(blended(logit,probit))".to_string());
    args.mixture_rho = Some("0.2".to_string());
    let err = parse_survival_inverse_link(&args)
        .expect_err("flexible blended survival link should be rejected");
    assert!(err.contains("does not support blended(...)/mixture(...)"));
}

#[test]
fn parse_survival_inverse_link_reports_survival_specific_supported_links() {
    let mut args = SurvivalArgs {
        data: std::path::PathBuf::from("dummy.csv"),
        entry: Some("entry".to_string()),
        exit: "exit".to_string(),
        event: "event".to_string(),
        formula: "1".to_string(),
        predict_noise: None,
        survival_likelihood: "location-scale".to_string(),
        survival_distribution: "gaussian".to_string(),
        link: Some("logit".to_string()),
        mixture_rho: None,
        sas_init: None,
        beta_logistic_init: None,
        survival_time_anchor: None,
        baseline_target: "linear".to_string(),
        baseline_scale: None,
        baseline_shape: None,
        baseline_rate: None,
        baseline_makeham: None,
        time_basis: "linear".to_string(),
        time_degree: 3,
        time_num_internal_knots: 8,
        time_smooth_lambda: 1e-2,
        ridge_lambda: 1e-6,
        threshold_time_k: None,
        threshold_time_degree: 3,
        sigma_time_k: None,
        sigma_time_degree: 3,
        slope_time_k: None,
        slope_time_degree: 3,
        scale_dimensions: false,
        pilot_subsample_threshold: 0,
        out: None,
        slope_formula: None,
        z_column: None,
        weights_column: None,
        offset_column: None,
        noise_offset_column: None,
        frailty_kind: None,
        frailty_sd: None,
        hazard_loading: None,
        persistent_warm_start_store: None,
    };
    args.link = Some("bogus".to_string());
    let err = parse_survival_inverse_link(&args).expect_err("expected unsupported survival link");
    assert!(err.contains("unsupported survival --link 'bogus'"));
    // `loglog` and `cauchit` are now genuinely implemented survival links (routed
    // through the single-component mixture kernels), so the usage line must advertise
    // them alongside the other supported survival links.
    assert!(err.contains("use identity|logit|probit|cloglog|loglog|cauchit|sas|beta-logistic|blended(...)/mixture(...) or flexible(...)"));
}

#[test]
fn parse_survival_inverse_link_accepts_loglog_and_cauchit() {
    let mut args = SurvivalArgs {
        data: std::path::PathBuf::from("dummy.csv"),
        entry: Some("entry".to_string()),
        exit: "exit".to_string(),
        event: "event".to_string(),
        formula: "1".to_string(),
        predict_noise: None,
        survival_likelihood: "location-scale".to_string(),
        survival_distribution: "gaussian".to_string(),
        link: Some("loglog".to_string()),
        mixture_rho: None,
        sas_init: None,
        beta_logistic_init: None,
        survival_time_anchor: None,
        baseline_target: "linear".to_string(),
        baseline_scale: None,
        baseline_shape: None,
        baseline_rate: None,
        baseline_makeham: None,
        time_basis: "linear".to_string(),
        time_degree: 3,
        time_num_internal_knots: 8,
        time_smooth_lambda: 1e-2,
        ridge_lambda: 1e-6,
        threshold_time_k: None,
        threshold_time_degree: 3,
        sigma_time_k: None,
        sigma_time_degree: 3,
        slope_time_k: None,
        slope_time_degree: 3,
        scale_dimensions: false,
        pilot_subsample_threshold: 0,
        out: None,
        slope_formula: None,
        z_column: None,
        weights_column: None,
        offset_column: None,
        noise_offset_column: None,
        frailty_kind: None,
        frailty_sd: None,
        hazard_loading: None,
        persistent_warm_start_store: None,
    };

    // `--link loglog` parses to a single-component LogLog mixture (weight 1.0), which
    // evaluates as the exact loglog inverse link mu = exp(-exp(-eta)).
    args.link = Some("loglog".to_string());
    let loglog = parse_survival_inverse_link(&args).expect("loglog survival link parses");
    let loglog_state = match &loglog {
        InverseLink::Mixture(state) => state,
        other => panic!("expected loglog to route through a mixture, got {other:?}"),
    };
    assert_eq!(loglog_state.components, vec![LinkComponent::LogLog]);
    assert!((loglog_state.pi[0] - 1.0).abs() < 1e-12);
    let eta = 0.3_f64;
    let jet = mixture_inverse_link_jet(loglog_state, eta);
    let expected_loglog_mu = (-((-eta).exp())).exp();
    assert!(
        (jet.mu - expected_loglog_mu).abs() < 1e-10,
        "loglog mu mismatch: {} vs {}",
        jet.mu,
        expected_loglog_mu
    );

    // `--link cauchit` parses to a single-component Cauchit mixture, evaluating as the
    // exact cauchit inverse link mu = 0.5 + atan(eta)/pi.
    args.link = Some("cauchit".to_string());
    let cauchit = parse_survival_inverse_link(&args).expect("cauchit survival link parses");
    let cauchit_state = match &cauchit {
        InverseLink::Mixture(state) => state,
        other => panic!("expected cauchit to route through a mixture, got {other:?}"),
    };
    assert_eq!(cauchit_state.components, vec![LinkComponent::Cauchit]);
    assert!((cauchit_state.pi[0] - 1.0).abs() < 1e-12);
    let cjet = mixture_inverse_link_jet(cauchit_state, eta);
    let expected_cauchit_mu = 0.5 + eta.atan() / std::f64::consts::PI;
    assert!(
        (cjet.mu - expected_cauchit_mu).abs() < 1e-10,
        "cauchit mu mismatch: {} vs {}",
        cjet.mu,
        expected_cauchit_mu
    );
}

#[test]
fn structural_survival_fit_is_time_unit_invariant() {
    let fit_structural_survival_eta = |age_entry: &Array1<f64>,
                                       age_exit: &Array1<f64>,
                                       event_target: &Array1<u8>,
                                       knots| {
        let time_build = build_survival_time_basis(
            age_entry,
            age_exit,
            SurvivalTimeBasisConfig::ISpline {
                degree: 2,
                knots,
                keep_cols: Vec::new(),
                smooth_lambda: 5e-1,
            },
            None,
        )
        .unwrap_or_else(|e| panic!("{} failed: {:?}", "build structural survival time basis", e));
        let p_time = time_build.x_exit_time.ncols();
        let penalties = gam::families::survival::PenaltyBlocks::new(
            time_build
                .penalties
                .iter()
                .enumerate()
                .filter(|(_, s)| s.nrows() == p_time && s.ncols() == p_time)
                .map(|(idx, s)| gam::families::survival::PenaltyBlock {
                    matrix: s.clone(),
                    lambda: 5e-1,
                    range: 0..p_time,
                    nullspace_dim: time_build.nullspace_dims.get(idx).copied().unwrap_or(0),
                })
                .collect(),
        );
        let event_competing = Array1::zeros(age_entry.len());
        let weights = Array1::ones(age_entry.len());
        let eta_offset_entry = Array1::zeros(age_entry.len());
        let eta_offset_exit = Array1::zeros(age_entry.len());
        let derivative_offset_exit = Array1::zeros(age_entry.len());
        let tb_entry_d = time_build.x_entry_time.to_dense();
        let tb_exit_d = time_build.x_exit_time.to_dense();
        let tb_deriv_d = time_build.x_derivative_time.to_dense();
        let mut model = gam::families::survival::royston_parmar::working_model_from_flattened(
            penalties,
            gam::families::survival::SurvivalMonotonicityPenalty { tolerance: 0.0 },
            gam::families::survival::SurvivalSpec::Net,
            gam::families::survival::royston_parmar::RoystonParmarInputs {
                age_entry: age_entry.view(),
                age_exit: age_exit.view(),
                event_target: event_target.view(),
                event_competing: event_competing.view(),
                weights: weights.view(),
                x_entry: tb_entry_d.view(),
                x_exit: tb_exit_d.view(),
                x_derivative: tb_deriv_d.view(),
                monotonicity_constraint_rows: None,
                monotonicity_constraint_offsets: None,
                eta_offset_entry: Some(eta_offset_entry.view()),
                eta_offset_exit: Some(eta_offset_exit.view()),
                derivative_offset_exit: Some(derivative_offset_exit.view()),
            },
        )
        .unwrap_or_else(|e| panic!("{} failed: {:?}", "construct structural survival model", e));
        model
            .set_structural_monotonicity(true, p_time)
            .unwrap_or_else(|e| panic!("{} failed: {:?}", "enable structural monotonicity", e));
        let mut beta0 = Array1::<f64>::zeros(p_time);
        beta0.fill(0.1);
        let mut constrained_model = model;
        let lb = Array1::from_elem(p_time, 0.0_f64);
        let summary = gam::pirls::runworking_model_pirls(
            &mut constrained_model,
            gam::types::Coefficients::new(beta0),
            &gam::pirls::WorkingModelPirlsOptions {
                max_iterations: 400,
                convergence_tolerance: 1e-6,
                max_step_halving: 40,
                min_step_size: 1e-12,
                firth_bias_reduction: false,
                coefficient_lower_bounds: Some(lb),
                linear_constraints: None,
                initial_lm_lambda: None,
                adaptive_kkt_tolerance: None,
                arrow_schur: None,
            },
            None,
        )
        .unwrap_or_else(|e| panic!("{} failed: {:?}", "fit structural survival model", e));
        assert!(
            matches!(
                summary.status,
                gam::pirls::PirlsStatus::Converged | gam::pirls::PirlsStatus::StalledAtValidMinimum
            ),
            "unexpected PIRLS status: {:?} after {} iterations, grad_norm={:.3e}",
            summary.status,
            summary.iterations,
            summary.lastgradient_norm
        );
        let beta = summary.beta.as_ref().to_owned();
        let eta = time_build.x_exit_time.dot(&beta);
        let surv = eta.mapv(|v| (-v.exp()).exp().clamp(0.0, 1.0));
        let state = constrained_model.update_state(&beta).unwrap_or_else(|e| {
            panic!(
                "{} failed: {:?}",
                "evaluate fitted structural survival state", e
            )
        });
        (eta, surv, state.deviance)
    };

    let age_entry_days = Array1::from_vec(vec![10.0, 20.0, 40.0, 80.0, 120.0, 160.0]);
    let age_exit_days = Array1::from_vec(vec![15.0, 35.0, 60.0, 100.0, 150.0, 220.0]);
    let event_target = Array1::from_vec(vec![1u8, 0u8, 1u8, 0u8, 1u8, 1u8]);
    let knots_days = Array1::from_vec(vec![2.0, 2.0, 2.0, 2.0, 4.0, 5.5, 5.5, 5.5, 5.5]);

    let (eta_days, surv_days, deviance_days) = fit_structural_survival_eta(
        &age_entry_days,
        &age_exit_days,
        &event_target,
        knots_days.clone(),
    );

    let time_scale = 365.25;
    let age_entry_years = age_entry_days.mapv(|v| v / time_scale);
    let age_exit_years = age_exit_days.mapv(|v| v / time_scale);
    let knots_years = knots_days.mapv(|v| v - time_scale.ln());
    let (eta_years, surv_years, deviance_years) = fit_structural_survival_eta(
        &age_entry_years,
        &age_exit_years,
        &event_target,
        knots_years,
    );

    assert_eq!(eta_days.len(), eta_years.len());
    assert_eq!(surv_days.len(), surv_years.len());
    for i in 0..eta_days.len() {
        assert!(
            (eta_days[i] - eta_years[i]).abs() <= 1e-5,
            "fitted eta mismatch at row {i}: days={} years={}",
            eta_days[i],
            eta_years[i]
        );
        assert!(
            (surv_days[i] - surv_years[i]).abs() <= 1e-6,
            "fitted survival mismatch at row {i}: days={} years={}",
            surv_days[i],
            surv_years[i]
        );
    }

    let event_count = event_target.iter().map(|d| f64::from(*d)).sum::<f64>();
    let expected_deviance_shift = -2.0 * event_count * time_scale.ln();
    assert!(
        (deviance_years - deviance_days - expected_deviance_shift).abs() <= 1e-5,
        "fitted deviance shift mismatch: years={} days={} expected_shift={expected_deviance_shift}",
        deviance_years,
        deviance_days
    );
}

/// Integration test: a small survival dataset (6 rows, intercept-only
/// formula) run through the full `run_survival` pipeline must converge.
/// This exercises the entire path a real user hits: CSV loading, I-spline
/// time basis construction, REML smoothing parameter selection, and
/// constrained PIRLS fitting.  The user never specifies a penalty — REML
/// picks it automatically.
///
/// Exercises the PIRLS eta-guard and stall-detection on a small,
/// underdetermined I-spline survival problem.
#[test]
fn survival_integration_small_dataset_converges() {
    let dir = tempdir().unwrap_or_else(|e| panic!("{} failed: {:?}", "tempdir", e));
    let csv_path = dir.path().join("small_surv.csv");
    let out_path = dir.path().join("model.json");
    std::fs::write(
        &csv_path,
        "entry,exit,event\n\
             10,15,1\n\
             20,35,0\n\
             40,60,1\n\
             80,100,0\n\
             120,150,1\n\
             160,220,1\n",
    )
    .unwrap_or_else(|e| panic!("{} failed: {:?}", "write csv", e));
    let args = SurvivalArgs {
        data: csv_path,
        entry: Some("entry".to_string()),
        exit: "exit".to_string(),
        event: "event".to_string(),
        formula: "1".to_string(),
        predict_noise: None,
        survival_likelihood: "transformation".to_string(),
        survival_distribution: "gaussian".to_string(),
        link: None,
        mixture_rho: None,
        sas_init: None,
        beta_logistic_init: None,
        survival_time_anchor: None,
        baseline_target: "linear".to_string(),
        baseline_scale: None,
        baseline_shape: None,
        baseline_rate: None,
        baseline_makeham: None,
        time_basis: "ispline".to_string(),
        time_degree: 2,
        time_num_internal_knots: 4,
        time_smooth_lambda: 1e-2,
        ridge_lambda: 1e-6,
        threshold_time_k: None,
        threshold_time_degree: 3,
        sigma_time_k: None,
        sigma_time_degree: 3,
        slope_time_k: None,
        slope_time_degree: 3,
        scale_dimensions: false,
        pilot_subsample_threshold: 0,
        out: Some(out_path.clone()),
        slope_formula: None,
        z_column: None,
        weights_column: None,
        offset_column: None,
        noise_offset_column: None,
        frailty_kind: None,
        frailty_sd: None,
        hazard_loading: None,
        persistent_warm_start_store: None,
    };
    let result = super::run_survival(args);
    assert!(
        result.is_ok(),
        "survival integration fit failed on 6-row dataset: {}",
        result.unwrap_err()
    );
    assert!(out_path.exists(), "model output file should be written");
}

#[test]
fn survival_timewiggle_with_parametric_baseline_skips_base_basis_requirement() {
    let dir = tempdir().unwrap_or_else(|e| panic!("{} failed: {:?}", "tempdir", e));
    let csv_path = dir.path().join("small_surv_timewiggle.csv");
    let out_path = dir.path().join("timewiggle.model.json");
    std::fs::write(
        &csv_path,
        "entry,exit,event\n\
             10,15,1\n\
             20,35,0\n\
             40,60,1\n\
             80,100,0\n\
             120,150,1\n\
             160,220,1\n",
    )
    .unwrap_or_else(|e| panic!("{} failed: {:?}", "write csv", e));
    let args = SurvivalArgs {
        data: csv_path,
        entry: Some("entry".to_string()),
        exit: "exit".to_string(),
        event: "event".to_string(),
        formula: "timewiggle(degree=3, internal_knots=4)".to_string(),
        predict_noise: None,
        survival_likelihood: "transformation".to_string(),
        survival_distribution: "gaussian".to_string(),
        link: None,
        mixture_rho: None,
        sas_init: None,
        beta_logistic_init: None,
        survival_time_anchor: None,
        baseline_target: "gompertz-makeham".to_string(),
        baseline_scale: None,
        baseline_shape: None,
        baseline_rate: None,
        baseline_makeham: None,
        time_basis: "ispline".to_string(),
        time_degree: 2,
        time_num_internal_knots: 4,
        time_smooth_lambda: 1e-2,
        ridge_lambda: 1e-6,
        threshold_time_k: None,
        threshold_time_degree: 3,
        sigma_time_k: None,
        sigma_time_degree: 3,
        slope_time_k: None,
        slope_time_degree: 3,
        scale_dimensions: false,
        pilot_subsample_threshold: 0,
        out: Some(out_path.clone()),
        slope_formula: None,
        z_column: None,
        weights_column: None,
        offset_column: None,
        noise_offset_column: None,
        frailty_kind: None,
        frailty_sd: None,
        hazard_loading: None,
        persistent_warm_start_store: None,
    };
    super::run_survival(args).unwrap_or_else(|e| {
        panic!(
            "{} failed: {:?}",
            "survival timewiggle fit should succeed", e
        )
    });

    let saved = SavedModel::load_from_path(&out_path)
        .unwrap_or_else(|e| panic!("{} failed: {:?}", "load fitted survival model", e));
    assert_eq!(saved.survival_time_basis.as_deref(), Some("none"));
    assert!(saved.baseline_timewiggle_knots.is_some());
    assert!(saved.beta_baseline_timewiggle.is_some());
}

#[test]
fn survival_location_scale_rejects_linkwiggle_for_mixture_inverse_link() {
    let dir = tempdir().unwrap_or_else(|e| panic!("{} failed: {:?}", "tempdir", e));
    let csv_path = dir.path().join("small_surv_linkwiggle_reject.csv");
    std::fs::write(
        &csv_path,
        "entry,exit,event\n\
             10,15,1\n\
             20,35,0\n\
             40,60,1\n\
             80,100,0\n\
             120,150,1\n\
             160,220,1\n",
    )
    .unwrap_or_else(|e| panic!("{} failed: {:?}", "write csv", e));
    let err = super::run_survival(SurvivalArgs {
        data: csv_path,
        entry: Some("entry".to_string()),
        exit: "exit".to_string(),
        event: "event".to_string(),
        formula: "1 + linkwiggle(degree=2, internal_knots=2)".to_string(),
        predict_noise: None,
        survival_likelihood: "location-scale".to_string(),
        survival_distribution: "gaussian".to_string(),
        link: Some("loglog".to_string()),
        mixture_rho: None,
        sas_init: None,
        beta_logistic_init: None,
        survival_time_anchor: None,
        baseline_target: "linear".to_string(),
        baseline_scale: None,
        baseline_shape: None,
        baseline_rate: None,
        baseline_makeham: None,
        time_basis: "ispline".to_string(),
        time_degree: 2,
        time_num_internal_knots: 4,
        time_smooth_lambda: 1e-2,
        ridge_lambda: 1e-6,
        threshold_time_k: None,
        threshold_time_degree: 3,
        sigma_time_k: None,
        sigma_time_degree: 3,
        slope_time_k: None,
        slope_time_degree: 3,
        scale_dimensions: false,
        pilot_subsample_threshold: 0,
        out: None,
        slope_formula: None,
        z_column: None,
        weights_column: None,
        offset_column: None,
        noise_offset_column: None,
        frailty_kind: None,
        frailty_sd: None,
        hazard_loading: None,
        persistent_warm_start_store: None,
    })
    .expect_err("mixture-backed survival linkwiggle should be rejected before fitting");
    assert!(
        err.contains(
            "linkwiggle(...) does not support latent-cloglog, SAS, BetaLogistic, or Mixture links"
        ),
        "unexpected error: {err}",
    );
}

#[test]
fn survival_location_scale_saved_fit_preserves_linkwiggle_metadata() {
    let dir = tempdir().unwrap_or_else(|e| panic!("{} failed: {:?}", "tempdir", e));
    let csv_path = dir.path().join("small_surv_linkwiggle.csv");
    let out_path = dir.path().join("surv_linkwiggle.model.json");
    std::fs::write(
        &csv_path,
        "entry,exit,event\n\
             10,15,1\n\
             20,35,0\n\
             40,60,1\n\
             80,100,0\n\
             120,150,1\n\
             160,220,1\n",
    )
    .unwrap_or_else(|e| panic!("{} failed: {:?}", "write csv", e));
    super::run_survival(SurvivalArgs {
        data: csv_path,
        entry: Some("entry".to_string()),
        exit: "exit".to_string(),
        event: "event".to_string(),
        formula: "1 + linkwiggle(degree=2, internal_knots=2)".to_string(),
        predict_noise: None,
        survival_likelihood: "location-scale".to_string(),
        survival_distribution: "gaussian".to_string(),
        link: None,
        mixture_rho: None,
        sas_init: None,
        beta_logistic_init: None,
        survival_time_anchor: None,
        baseline_target: "linear".to_string(),
        baseline_scale: None,
        baseline_shape: None,
        baseline_rate: None,
        baseline_makeham: None,
        time_basis: "ispline".to_string(),
        time_degree: 2,
        time_num_internal_knots: 4,
        time_smooth_lambda: 1e-2,
        ridge_lambda: 1e-6,
        threshold_time_k: None,
        threshold_time_degree: 3,
        sigma_time_k: None,
        sigma_time_degree: 3,
        slope_time_k: None,
        slope_time_degree: 3,
        scale_dimensions: false,
        pilot_subsample_threshold: 0,
        out: Some(out_path.clone()),
        slope_formula: None,
        z_column: None,
        weights_column: None,
        offset_column: None,
        noise_offset_column: None,
        frailty_kind: None,
        frailty_sd: None,
        hazard_loading: None,
        persistent_warm_start_store: None,
    })
    .unwrap_or_else(|e| {
        panic!(
            "{} failed: {:?}",
            "survival location-scale linkwiggle fit should succeed", e
        )
    });

    let saved = SavedModel::load_from_path(&out_path)
        .unwrap_or_else(|e| panic!("{} failed: {:?}", "load fitted survival model", e));
    let fit = saved
        .fit_result
        .as_ref()
        .unwrap_or_else(|| panic!("{} failed", "saved survival fit_result should be present"));
    assert!(saved.linkwiggle_knots.is_some());
    assert!(saved.linkwiggle_degree.is_some());
    assert!(saved.beta_link_wiggle.is_some());
    assert!(fit.block_by_role(BlockRole::LinkWiggle).is_some());
    assert_eq!(
        fit.artifacts.survival_link_wiggle_degree,
        saved.linkwiggle_degree,
    );
    assert_eq!(
        fit.artifacts
            .survival_link_wiggle_knots
            .as_ref()
            .map(|knots| knots.to_vec()),
        saved.linkwiggle_knots.clone(),
    );
}

#[test]
fn survival_time_basis_inference_rejects_nonfinite_times_before_knot_retry() {
    let age_entry = Array1::from_vec(vec![1e-9; 4]);
    let age_exit = Array1::from_vec(vec![0.5, 1.0, f64::NAN, 4.0]);
    let err = match build_survival_time_basis(
        &age_entry,
        &age_exit,
        SurvivalTimeBasisConfig::BSpline {
            degree: 3,
            knots: Array1::zeros(0),
            smooth_lambda: 1e-2,
        },
        Some((4, 1e-6)),
    ) {
        Ok(_) => panic!("non-finite times should not retry through uniform knots"),
        Err(err) => err,
    };

    assert!(err.contains("survival time basis requires finite exit times (row 3)"));
}

#[test]
fn survival_feasible_initial_beta_handles_sparse_overlapping_constraints() {
    let constraints = gam::pirls::LinearInequalityConstraints {
        a: Array2::from_shape_vec((3, 3), vec![1.0, 0.0, 0.0, 0.5, 1.0, 0.0, 0.0, 1.0, 1.0])
            .unwrap_or_else(|e| panic!("{} failed: {:?}", "constraint rows", e)),
        b: Array1::from_vec(vec![0.25, 0.5, 0.75]),
    };

    let beta0 = project_onto_linear_constraints(3, &constraints, None)
        .expect("projection from origin onto well-formed constraints must succeed");

    assert!(beta0.iter().all(|v| v.is_finite()));
    for i in 0..constraints.a.nrows() {
        let slack = constraints.a.row(i).dot(&beta0) - constraints.b[i];
        assert!(slack >= -1e-9, "constraint {i} violated by {slack}");
    }
}

#[test]
fn survival_feasible_initial_beta_respects_offset_shifted_constraints() {
    let constraints = gam::pirls::LinearInequalityConstraints {
        a: Array2::from_shape_vec((2, 2), vec![1.0, 0.0, 0.25, 1.0])
            .unwrap_or_else(|e| panic!("{} failed: {:?}", "constraint rows", e)),
        b: Array1::from_vec(vec![-0.5, 0.4]),
    };

    let beta0 = project_onto_linear_constraints(2, &constraints, None)
        .expect("projection from origin onto well-formed constraints must succeed");

    assert!(beta0.iter().all(|v| v.is_finite()));
    assert!(constraints.a.row(0).dot(&beta0) - constraints.b[0] >= -1e-9);
    assert!(constraints.a.row(1).dot(&beta0) - constraints.b[1] >= -1e-9);
}

#[test]
fn survival_time_basis_rejects_reversed_intervals_before_basis_construction() {
    let age_entry = Array1::from_vec(vec![1.0, 3.0]);
    let age_exit = Array1::from_vec(vec![2.0, 2.5]);
    let err = match build_survival_time_basis(
        &age_entry,
        &age_exit,
        SurvivalTimeBasisConfig::BSpline {
            degree: 3,
            knots: Array1::zeros(0),
            smooth_lambda: 1e-2,
        },
        Some((4, 1e-6)),
    ) {
        Ok(_) => panic!("exit before entry should fail"),
        Err(err) => err,
    };

    assert!(err.contains("survival time basis requires exit times >= entry times (row 2)"));
}

#[test]
fn survival_time_basiszerowidth_data_surfaces_range_errorwithout_uniform_retry() {
    let age_entry = Array1::from_vec(vec![1.0; 4]);
    let age_exit = Array1::from_vec(vec![1.0; 4]);
    let err = match build_survival_time_basis(
        &age_entry,
        &age_exit,
        SurvivalTimeBasisConfig::BSpline {
            degree: 3,
            knots: Array1::zeros(0),
            smooth_lambda: 1e-2,
        },
        Some((4, 1e-6)),
    ) {
        Ok(_) => panic!("zero-width time support should fail"),
        Err(err) => err,
    };

    assert!(err.contains("Data range has zero width"));
}

#[test]
fn ispline_time_basis_reuses_saved_keep_cols_on_narrow_prediction_range() {
    let train_entry = Array1::from_vec(vec![1.0, 1.5, 2.0, 2.5, 3.5, 4.5]);
    let train_exit = Array1::from_vec(vec![1.2, 1.9, 2.8, 3.1, 4.2, 5.0]);
    let knots = Array1::from_vec(vec![0.0, 0.0, 0.0, 0.0, 0.8, 1.2, 1.6, 1.9, 1.9, 1.9, 1.9]);

    let trained = build_survival_time_basis(
        &train_entry,
        &train_exit,
        SurvivalTimeBasisConfig::ISpline {
            degree: 2,
            knots: knots.clone(),
            keep_cols: Vec::new(),
            smooth_lambda: 1e-2,
        },
        None,
    )
    .unwrap_or_else(|e| panic!("{} failed: {:?}", "build training ispline basis", e));

    let pred_entry = Array1::from_vec(vec![1.0, 1.1, 1.2]);
    let pred_exit = Array1::from_vec(vec![1.25, 1.3, 1.35]);
    let rebuilt = build_survival_time_basis(
        &pred_entry,
        &pred_exit,
        SurvivalTimeBasisConfig::ISpline {
            degree: 2,
            knots,
            keep_cols: trained
                .keep_cols
                .clone()
                .unwrap_or_else(|| panic!("{} failed", "saved keep cols")),
            smooth_lambda: 1e-2,
        },
        None,
    )
    .expect("rebuild prediction ispline basis");

    assert_eq!(rebuilt.x_entry_time.ncols(), trained.x_entry_time.ncols());
    assert_eq!(rebuilt.x_exit_time.ncols(), trained.x_exit_time.ncols());
    assert_eq!(
        rebuilt.x_derivative_time.ncols(),
        trained.x_derivative_time.ncols()
    );
    assert_eq!(rebuilt.keep_cols, trained.keep_cols);
}

#[test]
fn saved_linkwiggle_derivative_matches_exact_constrained_basis_chain_rule() {
    let q0 = array![-1.25, -0.2, 0.35, 1.4];
    let knots = vec![-2.0, -2.0, -2.0, -2.0, -0.5, 0.5, 2.0, 2.0, 2.0, 2.0];
    let knot_arr = Array1::from_vec(knots.clone());
    let constrained_cols = monotone_wiggle_basis_with_derivative_order(q0.view(), &knot_arr, 3, 0)
        .unwrap_or_else(|e| panic!("{} failed: {:?}", "build monotone link-wiggle basis", e))
        .ncols();
    let beta_link_wiggle = (0..constrained_cols)
        .map(|j| match j % 5 {
            0 => 0.2,
            1 => 0.15,
            2 => 0.05,
            3 => 0.1,
            _ => 0.08,
        })
        .collect::<Vec<_>>();
    let mut payload = test_payload(
        "y ~ x",
        ModelKind::LocationScale,
        FittedFamily::LocationScale {
            likelihood: LikelihoodSpec::new(
                ResponseFamily::Binomial,
                InverseLink::Standard(StandardLink::Probit),
            ),
            base_link: Some(InverseLink::Standard(StandardLink::Probit)),
        },
        "binomial-location-scale",
    );
    payload.link = Some(InverseLink::Standard(StandardLink::Probit));
    payload.linkwiggle_knots = Some(knots);
    payload.linkwiggle_degree = Some(3);
    payload.beta_link_wiggle = Some(beta_link_wiggle.clone());
    let model = SavedModel::from_payload(payload);

    let exact = test_saved_linkwiggle_derivative_q0(&q0, &model)
        .unwrap_or_else(|e| panic!("{} failed: {:?}", "exact derivative", e));
    let constrained_deriv = test_saved_linkwiggle_design(&q0, &model)
        .unwrap_or_else(|e| panic!("{} failed: {:?}", "design path should succeed", e))
        .expect("wiggle design")
        .ncols();
    assert_eq!(constrained_deriv, beta_link_wiggle.len());

    let d_basis = test_saved_linkwiggle_basis(&q0, &model, BasisOptions::first_derivative())
        .unwrap_or_else(|e| panic!("{} failed: {:?}", "derivative basis", e))
        .expect("wiggle derivative basis");
    let expected = d_basis.dot(&Array1::from_vec(beta_link_wiggle)) + 1.0;
    for i in 0..q0.len() {
        assert!(
            (exact[i] - expected[i]).abs() <= 1e-12,
            "wiggle dq/dq0 mismatch at row {i}: got {}, expected {}",
            exact[i],
            expected[i]
        );
    }
}

#[test]
fn parse_formula_allows_nested_expression_arguments_in_smooth_calls() {
    let parsed = parse_formula("y ~ s(log(x + 1), type=duchon, centers=12, power=0, order=1)")
        .unwrap_or_else(|e| panic!("{} failed: {:?}", "formula", e));
    let ParsedTerm::Smooth { vars, options, .. } = &parsed.terms[0] else {
        panic!("expected smooth term");
    };
    assert_eq!(vars, &vec!["log(x + 1)".to_string()]);
    assert_eq!(options.get("type").map(String::as_str), Some("duchon"));
    assert_eq!(options.get("power").map(String::as_str), Some("0"));
    assert_eq!(options.get("order").map(String::as_str), Some("1"));
}

#[test]
fn required_columns_include_the_by_smooth_grouping_variable() {
    // Regression for #807: a `by=` smooth carries its grouping/scaling
    // variable in options["by"], not in the positional `vars`. The CLI's
    // required-column set must still list it, or the data file loads without
    // that column and the fit aborts before any numerics. Covers the factor
    // (`s(x, by=g)`), numeric varying-coefficient, and tensor (`te(..., by=w)`)
    // forms — all share the ParsedTerm::Smooth representation.
    let factor = parse_formula("y ~ s(x, by=g)")
        .unwrap_or_else(|e| panic!("{} failed: {:?}", "parse factor by-smooth", e));
    let cols = required_columns_for_formula(&factor)
        .unwrap_or_else(|e| panic!("{} failed: {:?}", "required columns", e));
    assert!(
        cols.contains(&"g".to_string()),
        "by= grouping column 'g' must be required, got {cols:?}"
    );
    assert!(cols.contains(&"x".to_string()) && cols.contains(&"y".to_string()));

    let tensor = parse_formula("y ~ te(x, z, by=w)")
        .unwrap_or_else(|e| panic!("{} failed: {:?}", "parse tensor by-smooth", e));
    let tcols = required_columns_for_formula(&tensor)
        .unwrap_or_else(|e| panic!("{} failed: {:?}", "required columns", e));
    for needed in ["x", "y", "z", "w"] {
        assert!(
            tcols.contains(&needed.to_string()),
            "te(x, z, by=w) must require '{needed}', got {tcols:?}"
        );
    }
}

#[test]
fn parse_formula_reports_unbalanced_parentheses() {
    let err = parse_formula("y ~ s(x, k=10").expect_err("expected parse failure");
    assert!(err.to_string().contains("unbalanced parentheses"));
}

#[test]
fn auxiliary_formula_accepts_rhs_only_input() {
    let (normalized, parsed) = parse_matching_auxiliary_formula("s(x)", "y", "--predict-noise")
        .unwrap_or_else(|e| panic!("{} failed: {:?}", "auxiliary formula", e));
    assert_eq!(normalized, "s(x)");
    assert_eq!(parsed.response, "y");
}

#[test]
fn auxiliary_formula_rejects_explicit_response_column() {
    let err = parse_matching_auxiliary_formula("noise ~ s(x)", "y", "--predict-noise")
        .expect_err("explicit response should fail");
    assert_eq!(
        err.to_string(),
        "--predict-noise expects only the terms after '~', not a full 'response ~ terms' formula; use --predict-noise 's(x)' instead of --predict-noise 'y ~ s(x)' (or pass '1' for an intercept-only noise model)"
    );
}

#[test]
fn auxiliary_formula_rejects_explicit_survival_response() {
    let err = parse_matching_auxiliary_formula(
        "Surv(entry,exit,event) ~ s(x)",
        "Surv(entry, exit, event)",
        "--predict-noise",
    )
    .expect_err("explicit survival response should fail");
    assert_eq!(
        err.to_string(),
        "--predict-noise expects only the terms after '~', not a full 'response ~ terms' formula; use --predict-noise 's(x)' instead of --predict-noise 'y ~ s(x)' (or pass '1' for an intercept-only noise model)"
    );
}

#[test]
fn parse_surv_response_extracts_entry_exit_event_columns() {
    let surv = parse_surv_response("Surv(entry_time, exit_time, event)")
        .unwrap_or_else(|e| panic!("{} failed: {:?}", "parse Surv lhs", e));
    assert_eq!(
        surv,
        Some((
            Some("entry_time".to_string()),
            "exit_time".to_string(),
            "event".to_string()
        ))
    );
}

#[test]
fn parse_surv_response_accepts_two_arg_right_censored_shorthand() {
    // Surv(time, event): R survival / mgcv default, entry synthesized
    // as zero downstream. Confirmed by the None in slot 0.
    let surv = parse_surv_response("Surv(exit_time, event)")
        .unwrap_or_else(|e| panic!("{} failed: {:?}", "parse 2-arg Surv lhs", e));
    assert_eq!(
        surv,
        Some((None, "exit_time".to_string(), "event".to_string()))
    );
}

#[test]
fn parse_surv_response_rejectswrong_arity() {
    // 1-arg, 4-arg, etc are still rejected.
    let err = parse_surv_response("Surv(entry_time)").expect_err("invalid Surv arity should fail");
    assert!(
        err.to_string().contains("Surv(time, event)")
            || err.to_string().contains("Surv(entry, exit, event)"),
        "expected actionable arity error, got: {err}"
    );
}

#[test]
fn data_schema_encodes_categorical_levels_deterministically() {
    let schema = DataSchema {
        columns: vec![SchemaColumn {
            name: "group".to_string(),
            kind: ColumnKindTag::Categorical,
            levels: vec!["ControlGroup".to_string(), "Treatment".to_string()],
        }],
    };
    let headers = vec!["group".to_string()];
    let records = vec![
        StringRecord::from(vec!["ControlGroup"]),
        StringRecord::from(vec!["Treatment"]),
    ];
    let ds = encode_recordswith_schema(headers, records, &schema, UnseenCategoryPolicy::Error)
        .unwrap_or_else(|e| panic!("{} failed: {:?}", "dataset", e));
    assert_eq!(ds.values[[0, 0]], 0.0);
    assert_eq!(ds.values[[1, 0]], 1.0);
}

#[test]
fn data_schema_rejects_unseen_categorical_levels() {
    let schema = DataSchema {
        columns: vec![SchemaColumn {
            name: "group".to_string(),
            kind: ColumnKindTag::Categorical,
            levels: vec!["ControlGroup".to_string(), "Treatment".to_string()],
        }],
    };
    let headers = vec!["group".to_string()];
    let records = vec![StringRecord::from(vec!["NewGroup"])];
    let err = encode_recordswith_schema(headers, records, &schema, UnseenCategoryPolicy::Error)
        .expect_err("should fail");
    assert!(err.contains("unseen level"));
}

#[test]
fn probit_q0_helper_matches_manual_threshold_over_sigma() {
    let eta_t = array![0.8, -0.4, 1.2];
    let eta_ls = array![-1.0, 0.0, 1.5];
    let q0 = compute_probit_q0_from_eta(eta_t.view(), eta_ls.view())
        .unwrap_or_else(|e| panic!("{} failed: {:?}", "compute probit q0", e));
    for i in 0..q0.len() {
        let expected =
            -eta_t[i] * gam::families::sigma_link::exp_sigma_inverse_from_eta_scalar(eta_ls[i]);
        assert!((q0[i] - expected).abs() < 1e-12);
    }
}

#[test]
fn wiggle_domain_summary_counts_out_of_range_q0() {
    let q0 = array![-2.5, -0.5, 0.0, 1.0, 2.5];
    let knots = array![-1.0, -1.0, -1.0, -0.25, 0.25, 1.0, 1.0, 1.0];
    let summary = summarizewiggle_domain(q0.view(), knots.view(), 2)
        .unwrap_or_else(|e| panic!("{} failed: {:?}", "summarize wiggle domain", e));
    assert_eq!(summary.domain_min, -1.0);
    assert_eq!(summary.domain_max, 1.0);
    assert_eq!(summary.outside_count, 2);
    assert!((summary.outside_fraction - 0.4).abs() < 1e-12);
}

#[test]
fn wiggle_domain_summary_inside_range_reportszero_outside() {
    let q0 = array![-0.75, -0.25, 0.0, 0.6];
    let knots = array![-1.0, -1.0, -1.0, -0.2, 0.2, 1.0, 1.0, 1.0];
    let summary = summarizewiggle_domain(q0.view(), knots.view(), 2)
        .unwrap_or_else(|e| panic!("{} failed: {:?}", "summarize wiggle domain", e));
    assert_eq!(summary.outside_count, 0);
    assert!((summary.outside_fraction - 0.0).abs() < 1e-12);
}

#[test]
fn saved_linkwiggle_design_returnsnonewhen_metadata_missing() {
    let q0 = array![-0.3, 0.2];
    let mut payload = test_payload(
        "y ~ x",
        ModelKind::LocationScale,
        FittedFamily::LocationScale {
            likelihood: LikelihoodSpec::new(
                ResponseFamily::Binomial,
                InverseLink::Standard(StandardLink::Probit),
            ),
            base_link: Some(InverseLink::Standard(StandardLink::Probit)),
        },
        "binomial-location-scale",
    );
    payload.link = Some(InverseLink::Standard(StandardLink::Probit));
    let model = SavedModel::from_payload(payload);
    let design = test_saved_linkwiggle_design(&q0, &model)
        .unwrap_or_else(|e| panic!("{} failed: {:?}", "wiggle design", e));
    assert!(design.is_none());
}

#[test]
fn saved_linkwiggle_runtime_rejects_partial_metadata() {
    let mut payload = test_payload(
        "y ~ x",
        ModelKind::LocationScale,
        FittedFamily::LocationScale {
            likelihood: LikelihoodSpec::new(
                ResponseFamily::Binomial,
                InverseLink::Standard(StandardLink::Probit),
            ),
            base_link: Some(InverseLink::Standard(StandardLink::Probit)),
        },
        "binomial-location-scale",
    );
    payload.link = Some(InverseLink::Standard(StandardLink::Probit));
    payload.linkwiggle_knots = Some(vec![-1.0, -1.0, -1.0, 1.0, 1.0, 1.0]);
    payload.linkwiggle_degree = Some(2);
    let model = SavedModel::from_payload(payload);
    let err = model
        .saved_link_wiggle()
        .expect_err("expected partial-metadata error");
    assert!(err.to_string().contains("link-wiggle"));
}

#[test]
fn heuristic_knots_for_column_uses_uniquevalue_rule() {
    // Few unique values → `unique/4` clamped up to the 4-knot floor.
    let col = array![0.0, 0.0, 1.0, 1.0, 2.0, 3.0, 4.0, 5.0];
    assert_eq!(unique_count_column(col.view()), 6);
    assert_eq!(heuristic_knots_for_column(col.view()), 4);
    // Many unique values → clamped to the flat mgcv-like default cap of 8
    // internal knots (cubic basis ≈ 12 functions), NOT grown with n. A larger
    // column used to return 20 internal knots (a 24-function basis); that
    // over-rich default over-parameterized weak-signal additive fits and the
    // penalty could not shrink it away cleanly (gam#1680). The cap is flat in n:
    // users opt *in* to a wigglier fit by raising `k` explicitly.
    let bigger = Array1::from_iter((0..200).map(|v| v as f64));
    assert_eq!(heuristic_knots_for_column(bigger.view()), 8);
    // The 32-unique boundary is exactly where `unique/4` meets the cap, so
    // columns at or below it keep their previous knot count unchanged.
    let boundary = Array1::from_iter((0..32).map(|v| v as f64));
    assert_eq!(heuristic_knots_for_column(boundary.view()), 8);
}

#[test]
fn probit_location_scale_posterior_mean_matches_mcwhen_uncertainty_is_small() {
    let beta_t = -0.25;
    let beta_ls = -0.2;
    let cov = array![[0.01, 0.002], [0.002, 0.015]];
    let model = intercept_only_binomial_location_scale_model(
        beta_t,
        beta_ls,
        cov.clone(),
        None,
        None,
        None,
    );
    let predicted = posterior_mean_prediction_for_model(&model);
    let mc = mc_nonwiggle_posterior_mean(beta_t, beta_ls, &cov, 80_000, 42);
    assert!(
        (predicted - mc).abs() < 0.015,
        "small-uncertainty posterior mean should stay close to Monte Carlo: predicted={predicted}, mc={mc}"
    );
}

#[test]
fn binomial_location_scale_wiggle_uses_unified_generate_path() {
    let model = intercept_only_binomial_location_scale_model(
        -0.4,
        -1.3,
        Array2::eye(6),
        Some(vec![0.25, 0.1, 0.05, 0.02]),
        Some(vec![-3.0, -3.0, -3.0, -3.0, 0.0, 3.0, 3.0, 3.0, 3.0]),
        Some(3),
    );
    assert!(model.predictor().is_some());
    let data = ndarray::Array2::<f64>::zeros((2, 0));
    let headers = vec![];
    let col_map = HashMap::new();
    let spec = super::run_generate_unified(
        &model,
        data.view(),
        &col_map,
        Some(&headers),
        &Array1::zeros(data.nrows()),
        &Array1::zeros(data.nrows()),
        false,
    )
    .unwrap_or_else(|e| {
        panic!(
            "{} failed: {:?}",
            "generate binomial location-scale through unified predictor", e
        )
    });
    assert!(spec.mean.iter().all(|value| value.is_finite()));
    assert!(matches!(spec.noise, gam::generative::NoiseModel::Bernoulli));
}

#[test]
fn probit_location_scale_posterior_mean_matches_mc_in_largevariance_correlated_regime() {
    let beta_t = -0.4;
    let beta_ls = -1.3;
    let cov = array![[0.2, 1.5], [1.5, 20.0]];
    let model = intercept_only_binomial_location_scale_model(
        beta_t,
        beta_ls,
        cov.clone(),
        None,
        None,
        None,
    );
    let predicted = posterior_mean_prediction_for_model(&model);
    let mc = mc_nonwiggle_posterior_mean(beta_t, beta_ls, &cov, 120_000, 7);
    assert!(
        (predicted - mc).abs() < 0.03,
        "posterior mean should match Monte Carlo in the hard correlated regime: predicted={predicted}, mc={mc}"
    );
}

#[test]
fn probit_location_scalewiggle_posterior_mean_matches_mc_in_largevariance_regime() {
    let beta_t = -0.4;
    let beta_ls = -1.3;
    let beta_link_wiggle = vec![0.25, 0.10, 0.05, 0.02];
    let cov_diag = vec![0.2, 10.0, 0.4, 0.3, 0.2, 0.1];
    let cov = Array2::from_diag(&Array1::from_vec(cov_diag.clone()));
    let model = intercept_only_binomial_location_scale_model(
        beta_t,
        beta_ls,
        cov,
        Some(beta_link_wiggle.clone()),
        Some(vec![-3.0, -3.0, -3.0, -3.0, 0.0, 3.0, 3.0, 3.0, 3.0]),
        Some(3),
    );
    let predicted = posterior_mean_prediction_for_model(&model);
    let mc = mcwiggle_posterior_mean(
        beta_t,
        beta_ls,
        &beta_link_wiggle,
        &cov_diag,
        &model,
        80_000,
        99,
    );
    assert!(
        (predicted - mc).abs() < 0.03,
        "wiggle posterior mean should match Monte Carlo in the hard regime: predicted={predicted}, mc={mc}"
    );
}

// TEMPORARY gam#2695 probe — a linkwiggle fit whose warp is genuinely ON.
#[test]
fn probe_2695_live_warp() {
    gam_runtime::test_support::install_diagnostic_logger();
    let dir = tempdir().unwrap_or_else(|e| panic!("{} failed: {:?}", "tempdir", e));
    let csv_path = dir.path().join("live_warp.csv");
    // A log-logistic AFT truth fitted with a GAUSSIAN location-scale link: the
    // monotone link warp is the only block that can absorb the difference, so
    // it is driven away from zero rather than shrunk to it.
    let mut rows = String::from("entry,exit,event,x\n");
    let n = 240usize;
    for i in 0..n {
        let u = (i as f64 + 0.5) / (n as f64);
        let x = -1.0 + 2.0 * u;
        // log-logistic quantile with scale exp(0.4 x) and shape 1.6
        let t = (u / (1.0 - u)).powf(1.0 / 1.6) * (0.4 * x).exp() * 10.0 + 1.0;
        let event = usize::from(i % 5 != 0);
        rows.push_str(&format!("0,{t:.6},{event},{x:.6}\n"));
    }
    std::fs::write(&csv_path, rows).unwrap_or_else(|e| panic!("{} failed: {:?}", "write csv", e));
    for (degree, knots) in [(0usize, 0usize), (2, 3), (3, 4), (4, 4)] {
        let out_path = dir.path().join(format!("live_warp_{degree}.model.json"));
        let result = super::run_survival(SurvivalArgs {
            data: csv_path.clone(),
            entry: Some("entry".to_string()),
            exit: "exit".to_string(),
            event: "event".to_string(),
            formula: if degree == 0 {
                "1 + x".to_string()
            } else {
                format!("1 + x + linkwiggle(degree={degree}, internal_knots={knots})")
            },
            predict_noise: None,
            survival_likelihood: "location-scale".to_string(),
            survival_distribution: "gaussian".to_string(),
            link: None,
            mixture_rho: None,
            sas_init: None,
            beta_logistic_init: None,
            survival_time_anchor: None,
            baseline_target: "linear".to_string(),
            baseline_scale: None,
            baseline_shape: None,
            baseline_rate: None,
            baseline_makeham: None,
            time_basis: "ispline".to_string(),
            time_degree: 3,
            time_num_internal_knots: 6,
            time_smooth_lambda: 1e-2,
            ridge_lambda: 1e-6,
            threshold_time_k: None,
            threshold_time_degree: 3,
            sigma_time_k: None,
            sigma_time_degree: 3,
            slope_time_k: None,
            slope_time_degree: 3,
            scale_dimensions: false,
            pilot_subsample_threshold: 0,
            out: Some(out_path.clone()),
            slope_formula: None,
            z_column: None,
            weights_column: None,
            offset_column: None,
            noise_offset_column: None,
            frailty_kind: None,
            frailty_sd: None,
            hazard_loading: None,
            persistent_warm_start_store: None,
        });
        match result {
            Ok(_) => {
                let saved = SavedModel::load_from_path(&out_path).expect("load");
                let beta_w = saved.beta_link_wiggle.clone().unwrap_or_default();
                let amp = beta_w.iter().fold(0.0_f64, |a, b| a.max(b.abs()));
                println!(
                    "[2695-LIVE] degree={degree} FIT OK  max|beta_w|={amp:.6e} p_w={}",
                    beta_w.len()
                );
            }
            Err(e) => {
                let rejects: Vec<String> = e
                    .match_indices("rejects [model,likelihood,objective,feasibility]")
                    .map(|(i, _)| e[i..(i + 70).min(e.len())].to_string())
                    .collect();
                println!(
                    "[2695-LIVE] degree={degree} FAILED; reject buckets: {:?}",
                    rejects
                );
            }
        }
    }
}
