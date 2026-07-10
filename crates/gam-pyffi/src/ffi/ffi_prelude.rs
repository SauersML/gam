//! Shared pyo3 / numpy / ndarray / `gam`-engine imports for the FFI boundary.
//!
//! Every FFI submodule pulls this in via `use crate::ffi_prelude::*;` so the
//! large, hand-curated set of engine re-exports lives in exactly one place
//! instead of being duplicated across each concern module. Re-exports are
//! `pub(crate)` so the glob is visible to sibling FFI modules but never
//! escapes the crate.

pub(crate) use csv::StringRecord;

pub(crate) use faer::Side;

pub(crate) use gam::families::fit_orchestration::DispersionLocationScaleFitResult;

pub(crate) use gam::terms::basis::create_duchon_basis_1d_derivative_dense;

pub(crate) use gam::solver::estimate::{
    BlockRole, EstimationError, ExternalOptimOptions,
    optimize_external_designwith_heuristic_lambdas, saved_latent_cloglog_state_from_fit,
    saved_mixture_state_from_fit, saved_sas_state_from_fit,
};

pub(crate) use gam::linalg::faer_ndarray::{
    FaerCholesky, FaerSvd, array2_to_matmut, factorize_symmetricwith_fallback, fast_ata, fast_atb,
    fast_xt_diag_x,
};

pub(crate) use gam::families::bms::BernoulliMarginalSlopeFitResult;

pub(crate) use gam::families::inverse_link::{apply_inverse_link_spec_vec, apply_inverse_link_vec};

pub(crate) use gam::families::scale_design::{
    build_scale_deviation_transform, infer_non_intercept_start,
};

pub(crate) use gam::families::survival::construction::{
    SavedSurvivalTimeBasis, survival_likelihood_modename,
};

pub(crate) use gam::families::survival::predict::{
    apply_inverse_link_state_to_fit_result, fit_result_from_saved_model_for_prediction,
};

pub(crate) use gam::families::gamlss::{
    BinomialLocationScaleFitResult, DispersionFamilyKind, GaussianLocationScaleFitResult,
};

pub(crate) use gam::solver::gaussian_reml::{
    GaussianRemlMultiBackwardProblem, build_gaussian_reml_eigen_cache_batched,
    gaussian_reml_blocks_orthogonal_shared_scale, gaussian_reml_free_b_score,
    gaussian_reml_multi_closed_form_backward, gaussian_reml_multi_closed_form_backward_batch,
    gaussian_reml_multi_closed_form_backward_from_fit, gaussian_reml_multi_closed_form_with_cache,
};

pub(crate) use gam::geometry::manifold::GeometryError as EngineGeometryError;

pub(crate) use gam::geometry::poincare::{
    conformal_factor as poincare_conformal_factor_impl, distance_batch as poincare_distance_batch_impl,
    exp_map as poincare_exp_map_impl, exp_map_batch as poincare_exp_map_batch_impl,
    exp_origin as poincare_exp_origin_impl, from_lorentz as poincare_from_lorentz_impl,
    log_map as poincare_log_map_impl, log_map_batch as poincare_log_map_batch_impl,
    log_origin as poincare_log_origin_impl,
    lorentz_decode_backward as poincare_lorentz_decode_backward_impl,
    lorentz_decode_forward as poincare_lorentz_decode_forward_impl,
    lorentz_exp_origin as poincare_lorentz_exp_origin_impl,
    lorentz_log_origin as poincare_lorentz_log_origin_impl, mobius_add as poincare_mobius_add_impl,
    metric_tensor_batch as poincare_metric_tensor_batch_impl,
    poincare_distance as poincare_distance_impl, project_into_ball as poincare_project_into_ball_impl,
    project_into_ball_batch as poincare_project_into_ball_batch_impl,
    tangent_decode_backward as poincare_tangent_decode_backward_impl,
    tangent_decode_forward as poincare_tangent_decode_forward_impl,
    to_lorentz as poincare_to_lorentz_impl,
};

pub(crate) use gam::geometry::simplex::{closure as simplex_closure, simplex_frechet_mean};

pub(crate) use gam::sample::{NutsConfig, NutsResult};

pub(crate) use gam::data::{
    ColumnKindTag, DataSchema, EncodedDataset, SchemaColumn, UnseenCategoryPolicy,
    encode_recordswith_schema, infer_and_encode_column_major,
};

pub(crate) use gam::inference::formula_dsl::{parse_formula, parse_surv_response};

pub(crate) use gam::inference::model::{
    FittedFamily, FittedModel, FittedModelPayload, GroupMetadata, MODEL_PAYLOAD_VERSION, ModelKind,
    PredictModelClass, SavedDeploymentExtension, SavedLatentZNormalization,
    append_deployment_extension_columns,
};

pub(crate) use gam::inference::model_payload_builders::{
    BernoulliMarginalSlopeInputs, LatentWindowInputs, LocationScaleInputs, LocationScaleResponse,
    LocationScaleWiggle, SavedModelSourceMetadata, SurvivalLocationScaleInputs,
    SurvivalMarginalSlopeInputs, SurvivalTimewiggle, SurvivalTimewiggleBeta,
    SurvivalTransformationInputs, TransformationNormalInputs,
    assemble_bernoulli_marginal_slope_payload, assemble_latent_window_payload,
    assemble_location_scale_payload, assemble_survival_location_scale_payload,
    assemble_survival_marginal_slope_payload, assemble_survival_transformation_payload,
    assemble_transformation_normal_payload,
};

pub(crate) use gam_predict::posterior_bands::{self, PosteriorPredictBandsPayload};

pub(crate) use gam_predict::FittedModelPredictExt;
pub(crate) use gam_predict::input::build_predict_input_for_model;

pub(crate) use gam::geometry::sae_routing::apply_anchor_rule as sae_apply_anchor_rule_impl;
pub(crate) use gam::geometry::sae_routing::assign_ema_update as sae_assign_ema_update_impl;
pub(crate) use gam::geometry::sae_routing::direction_cluster_anchor as sae_direction_cluster_anchor_impl;
pub(crate) use gam::geometry::sae_routing::duchon_centers_nd as sae_duchon_centers_nd_impl;
pub(crate) use gam::geometry::sae_routing::matching_pursuit_commit as sae_matching_pursuit_commit_impl;
pub(crate) use gam::geometry::sae_routing::quadratic_subspace_anchor as sae_quadratic_subspace_anchor_impl;
pub(crate) use gam::geometry::sae_routing::sinkhorn_balance_bias as sae_sinkhorn_balance_bias_impl;

pub(crate) use gam::geometry::sinkhorn_barycenter::{
    circular_cost as sinkhorn_circular_cost_impl, euclidean_cost as sinkhorn_euclidean_cost_impl,
    geodesic_sphere_cost as sinkhorn_geodesic_sphere_cost_impl,
    sinkhorn_barycenter as sinkhorn_barycenter_impl,
    sinkhorn_barycenter_vjp as sinkhorn_barycenter_vjp_impl,
};

pub(crate) use gam::report::{CoefficientRow, EdfBlockRow, ReportInput, render_html};

pub(crate) use gam::families::fit_orchestration::drivers::freeze_term_collection_from_design;
pub(crate) use gam::terms::smooth::{
    TermCollectionDesign, TermCollectionSpec, smooth_term_feature_cols,
};
// #1521: relocated DOWN into gam_terms::smooth (was families::...::drivers).
pub(crate) use gam::terms::smooth::{
    build_term_collection_derivative_design, build_term_collection_design,
};

pub(crate) use gam::families::fit_orchestration::descriptors::build_analytic_penalty_registry_from_descriptors as build_analytic_penalty_registry_from_json;

pub(crate) use gam::solver::evidence::{
    RemlCandidate, compare_reml_fits as compare_reml_fits_core, log_bayes_factor,
};

pub(crate) use gam::families::survival::marginal_slope::SurvivalMarginalSlopeFitResult;

pub(crate) use gam::terms::basis::{
    BasisOptions, CenterStrategy, Dense, DuchonBasisSpec, DuchonNullspaceOrder,
    DuchonOperatorPenaltySpec, MaternBasisSpec, MaternIdentifiability, MaternNu,
    OneDimensionalBoundary, OperatorPenaltySpec, PeriodicBSplineBasisSpec, SpatialIdentifiability,
    SphereMethod, SphereWahbaKernel, SphericalSplineBasisSpec, SphericalSplineIdentifiability,
    SplineScratch, auto_centers_1d_equal_mass, auto_knot_vector_1d_quantile,
    bspline_tensor_first_derivative, build_duchon_basis, build_duchon_basis_mixed_periodicity_auto,
    build_duchon_operator_penalty_matrices, build_matern_basis, build_matern_basis_literal_aniso,
    build_periodic_bspline_basis_1d, build_spherical_spline_basis, build_thin_plate_penalty_matrix,
    create_basis, create_cyclic_difference_penalty_matrix, create_difference_penalty_matrix,
    duchon_cubic_default, duchon_nullspace_dimension, duchon_polynomial_first_derivative_nd,
    duchon_pure_kernel_amplification, duchon_radial_first_derivative_nd,
    duchon_sae_atom_basis_with_jet, evaluate_bspline_basis_scalar,
    matern_input_location_hessian_nd, matern_input_location_jet_nd,
    matern_radial_first_derivative_nd, monomial_exponents, periodic_bspline_first_derivative_nd,
    resolve_duchon_orders, select_spherical_farthest_point_centers, sphere_first_derivative_nd,
    spherical_spline_design_jet,
};

pub(crate) use gam::terms::basis::input_loc_derivatives::contract_input_loc_gradient;

pub(crate) use gam::terms::decoders::interchange_decoder::{
    InterchangeDecodeForward as CoreInterchangeDecodeForward,
    InterchangeSwapForward as CoreInterchangeSwapForward,
    interchange_decode_backward as core_interchange_decode_backward,
    interchange_decode_forward as core_interchange_decode_forward,
    interchange_swap_backward as core_interchange_swap_backward,
    interchange_swap_forward as core_interchange_swap_forward,
};

pub(crate) use gam::terms::latent::{AuxPriorFamily, aux_prior_targets};

pub(crate) use gam::terms::dictionary::{
    LinearDictionaryAssignment, LinearDictionaryConfig, fit_linear_dictionary,
    linear_dictionary_transform,
};

pub(crate) use gam::terms::sae::sparse_dict::{
    BlockChartComposeConfig, BlockChartRecord, BlockSeedManifest, BlockSeedManifestConfig,
    BlockSeedRecord, BlockSparseConfig, BlockSparseStreamState, MdlFeaturizerRow, SparseDictConfig,
    SparseDictStreamState, block_sparse_dictionary_block_coords, block_sparse_dictionary_firings,
    block_sparse_dictionary_lift_block, block_sparse_dictionary_project_residual,
    block_sparse_dictionary_seed_manifest, block_sparse_dictionary_transform,
    compose_block_coordinate_charts, fit_block_sparse_dictionary, fit_sparse_dictionary,
    reconstruct_block_sparse_rows, reconstruct_sparse_rows, sparse_dictionary_transform_with_mode,
};

pub(crate) use gam::terms::sae::manifold::{
    AssignmentMode, CylinderHarmonicEvaluator, DuchonCoordinateEvaluator, EuclideanPatchEvaluator,
    GumbelTemperatureSchedule, MobiusHarmonicEvaluator, PeriodicHarmonicEvaluator,
    SPHERE_CHART_PENALTY_DIAGONAL,
    SaeAtomBasisKind, SaeBasisEvaluator, SaeBasisSecondJet, SaeManifoldRho, ScheduleKind,
    SphereChartEvaluator, TorusHarmonicEvaluator, sphere_chart_basis_jet,
    term_from_padded_blocks_with_mode,
};

pub(crate) use gam::terms::decoders::skip_transcoder::{
    SkipTranscoderRemlInputs, skip_transcoder_reml_metrics as skip_transcoder_reml_metrics_core,
};

pub(crate) use gam::terms::smooth::BlockwisePenalty;

pub(crate) use gam::terms::basis::matern_gradient::{
    MaternBasisGradientTarget, StreamingMaternBasisGradientEvaluator,
};
pub(crate) use gam::terms::decoders::gated_decoder::GatedSAEDecoder;
pub(crate) use gam::terms::{
    AnalyticPenalty as AnalyticPenaltyTrait, AnalyticPenaltyKind, AnalyticPenaltyRegistry,
    EdgeRestriction as CoreEdgeRestriction, IvaeRidgeMeanGauge as IvaeRidgeMeanGaugePenalty,
    MechanismSparsityPenalty as CoreMechanismSparsityPenalty, ParametricRowPrecisionPriorPenalty,
    PenaltyTier, PsiSlice, RowPrecisionPriorPenalty,
    SheafConsistencyPenalty as CoreSheafConsistencyPenalty,
};

pub(crate) use gam::families::transformation_normal::TransformationNormalFitResult;

pub(crate) use gam::types::{InverseLink, LikelihoodSpec, ResponseFamily, RhoPrior, StandardLink};

pub(crate) use gam::families::fit_orchestration::{
    FitConfig, FitRequest, FitResult, WorkflowError, fit_model, materialize, resolve_offset_column,
    resolve_weight_column,
};

pub(crate) use ndarray::{
    Array1, Array2, Array3, Array4, ArrayView1, ArrayView2, ArrayView3, ArrayView4, Axis, IxDyn, s,
};

pub(crate) use numpy::{
    IntoPyArray, PyArray1, PyArray2, PyArray3, PyArray4, PyArrayDyn, PyArrayMethods,
    PyReadonlyArray1, PyReadonlyArray2, PyReadonlyArray3, PyReadonlyArray4, PyReadonlyArrayDyn,
};

pub(crate) use pyo3::IntoPyObjectExt;

pub(crate) type PyObject = pyo3::Py<pyo3::PyAny>;

pub(crate) use pyo3::exceptions::{PyKeyError, PyNotImplementedError, PyTypeError, PyValueError};

pub(crate) use pyo3::prelude::*;

pub(crate) use pyo3::types::{PyAny, PyBytes, PyDict, PyList, PyString, PyTuple, PyType};

pub(crate) use serde::de::{MapAccess, Visitor};

pub(crate) use serde::{Deserialize, Serialize};

pub(crate) use std::cmp::Ordering;

pub(crate) use std::collections::{BTreeMap, BTreeSet, HashMap};

pub(crate) use std::fmt;

pub(crate) use std::panic::{AssertUnwindSafe, catch_unwind};

pub(crate) use std::sync::Arc;

pub(crate) use std::sync::{Mutex, OnceLock};
