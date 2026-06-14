use csv::StringRecord;

use faer::Side;

use gam::DispersionLocationScaleFitResult;

use gam::basis::create_duchon_basis_1d_derivative_dense;

use gam::estimate::{
    BlockRole, EstimationError, ExternalOptimOptions,
    optimize_external_designwith_heuristic_lambdas, saved_latent_cloglog_state_from_fit,
    saved_mixture_state_from_fit, saved_sas_state_from_fit,
};

use gam::faer_ndarray::{
    FaerCholesky, FaerSvd, array2_to_matmut, factorize_symmetricwith_fallback, fast_ata, fast_atb,
    fast_xt_diag_x,
};

use gam::families::bms::BernoulliMarginalSlopeFitResult;

use gam::families::inverse_link::apply_inverse_link_vec;

use gam::families::scale_design::{build_scale_deviation_transform, infer_non_intercept_start};

use gam::families::survival_construction::{SavedSurvivalTimeBasis, survival_likelihood_modename};

use gam::families::survival_predict::{
    apply_inverse_link_state_to_fit_result, fit_result_from_saved_model_for_prediction,
};

use gam::gamlss::{
    BinomialLocationScaleFitResult, DispersionFamilyKind, GaussianLocationScaleFitResult,
};

use gam::gaussian_reml::{
    GaussianRemlMultiBackwardProblem, build_gaussian_reml_eigen_cache_batched,
    gaussian_reml_blocks_orthogonal_shared_scale, gaussian_reml_free_b_score,
    gaussian_reml_multi_closed_form_backward, gaussian_reml_multi_closed_form_backward_batch,
    gaussian_reml_multi_closed_form_backward_from_fit, gaussian_reml_multi_closed_form_with_cache,
};

use gam::geometry::manifold::GeometryError as EngineGeometryError;

use gam::geometry::poincare::{
    conformal_factor as poincare_conformal_factor_impl, exp_map as poincare_exp_map_impl,
    exp_origin as poincare_exp_origin_impl, from_lorentz as poincare_from_lorentz_impl,
    log_map as poincare_log_map_impl, log_origin as poincare_log_origin_impl,
    lorentz_decode_backward as poincare_lorentz_decode_backward_impl,
    lorentz_decode_forward as poincare_lorentz_decode_forward_impl,
    lorentz_exp_origin as poincare_lorentz_exp_origin_impl,
    lorentz_log_origin as poincare_lorentz_log_origin_impl, mobius_add as poincare_mobius_add_impl,
    poincare_distance as poincare_distance_impl,
    project_into_ball as poincare_project_into_ball_impl,
    tangent_decode_backward as poincare_tangent_decode_backward_impl,
    tangent_decode_forward as poincare_tangent_decode_forward_impl,
    to_lorentz as poincare_to_lorentz_impl,
};

use gam::geometry::simplex::{closure as simplex_closure, simplex_frechet_mean};

use gam::hmc::{NutsConfig, NutsResult};

use gam::inference::data::{
    EncodedDataset, UnseenCategoryPolicy, encode_recordswith_inferred_schema,
    encode_recordswith_schema,
};

use gam::inference::formula_dsl::{parse_formula, parse_surv_response};

use gam::inference::model::{
    ColumnKindTag, DataSchema, FittedFamily, FittedModel, FittedModelPayload, GroupMetadata,
    MODEL_PAYLOAD_VERSION, ModelKind, PredictModelClass, SavedDeploymentExtension,
    SavedLatentZNormalization, SchemaColumn, append_deployment_extension_columns,
};

use gam::inference::model_payload_builders::{
    BernoulliMarginalSlopeInputs, LatentWindowInputs, LocationScaleInputs, LocationScaleResponse,
    LocationScaleWiggle, SavedModelSourceMetadata, SurvivalLocationScaleInputs,
    SurvivalMarginalSlopeInputs, SurvivalTimewiggle, SurvivalTimewiggleBeta,
    SurvivalTransformationInputs, TransformationNormalInputs,
    assemble_bernoulli_marginal_slope_payload, assemble_latent_window_payload,
    assemble_location_scale_payload, assemble_survival_location_scale_payload,
    assemble_survival_marginal_slope_payload, assemble_survival_transformation_payload,
    assemble_transformation_normal_payload,
};

use gam::inference::posterior_bands::{self, PosteriorPredictBandsPayload};

use gam::inference::predict::input::build_predict_input_for_model;

use gam::kernels::sinkhorn_barycenter::{
    circular_cost as sinkhorn_circular_cost_impl, euclidean_cost as sinkhorn_euclidean_cost_impl,
    geodesic_sphere_cost as sinkhorn_geodesic_sphere_cost_impl,
    sinkhorn_barycenter as sinkhorn_barycenter_impl,
    sinkhorn_barycenter_vjp as sinkhorn_barycenter_vjp_impl,
};

use gam::report::{CoefficientRow, EdfBlockRow, ReportInput, render_html};

use gam::smooth::{
    TermCollectionDesign, TermCollectionSpec, build_term_collection_design,
    freeze_term_collection_from_design, smooth_term_feature_cols,
};

use gam::solver::build_analytic_penalty_registry_from_descriptors as build_analytic_penalty_registry_from_json;

use gam::solver::evidence::{
    RemlCandidate, compare_reml_fits as compare_reml_fits_core, log_bayes_factor,
};

use gam::survival_marginal_slope::SurvivalMarginalSlopeFitResult;

use gam::terms::basis::{
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

use gam::terms::input_loc_derivatives::contract_input_loc_gradient;

use gam::terms::interchange_decoder::{
    InterchangeDecodeForward as CoreInterchangeDecodeForward,
    InterchangeSwapForward as CoreInterchangeSwapForward,
    interchange_decode_backward as core_interchange_decode_backward,
    interchange_decode_forward as core_interchange_decode_forward,
    interchange_swap_backward as core_interchange_swap_backward,
    interchange_swap_forward as core_interchange_swap_forward,
};

use gam::terms::latent_coord::{AuxPriorFamily, aux_prior_targets};

use gam::terms::linear_dictionary::{
    LinearDictionaryAssignment, LinearDictionaryConfig, fit_linear_dictionary,
};

use gam::terms::sae_manifold::{
    AssignmentMode, DuchonCoordinateEvaluator, EuclideanPatchEvaluator, GumbelTemperatureSchedule,
    PeriodicHarmonicEvaluator, SPHERE_CHART_PENALTY_DIAGONAL, SaeAtomBasisKind, SaeBasisEvaluator,
    SaeManifoldRho, ScheduleKind, SphereChartEvaluator, TorusHarmonicEvaluator,
    sphere_chart_basis_jet, term_from_padded_blocks_with_mode,
};

use gam::terms::skip_transcoder::{
    SkipTranscoderRemlInputs, skip_transcoder_reml_metrics as skip_transcoder_reml_metrics_core,
};

use gam::terms::smooth::BlockwisePenalty;

use gam::terms::{
    AnalyticPenalty as AnalyticPenaltyTrait, AnalyticPenaltyKind, AnalyticPenaltyRegistry,
    EdgeRestriction as CoreEdgeRestriction, GatedSAEDecoder,
    IvaeRidgeMeanGauge as IvaeRidgeMeanGaugePenalty, MaternBasisGradientTarget,
    MechanismSparsityPenalty as CoreMechanismSparsityPenalty, ParametricRowPrecisionPriorPenalty,
    PenaltyTier, PsiSlice, RowPrecisionPriorPenalty,
    SheafConsistencyPenalty as CoreSheafConsistencyPenalty, StreamingMaternBasisGradientEvaluator,
};

use gam::transformation_normal::TransformationNormalFitResult;

use gam::types::{InverseLink, LikelihoodSpec, ResponseFamily, RhoPrior, StandardLink};

use gam::{
    FitConfig, FitRequest, FitResult, WorkflowError, fit_model, materialize, resolve_offset_column,
};

use ndarray::{
    Array1, Array2, Array3, Array4, ArrayView1, ArrayView2, ArrayView3, ArrayView4, Axis, IxDyn, s,
};

use numpy::{
    IntoPyArray, PyArray1, PyArray2, PyArray3, PyArray4, PyArrayDyn, PyArrayMethods,
    PyReadonlyArray1, PyReadonlyArray2, PyReadonlyArray3, PyReadonlyArray4, PyReadonlyArrayDyn,
};

use pyo3::IntoPyObjectExt;

pub(crate) type PyObject = pyo3::Py<pyo3::PyAny>;

use pyo3::exceptions::{PyKeyError, PyNotImplementedError, PyTypeError, PyValueError};

use pyo3::prelude::*;

use pyo3::types::{PyAny, PyBytes, PyDict, PyList, PyString, PyTuple, PyType};

use serde::de::{MapAccess, Visitor};

use serde::{Deserialize, Serialize};

use std::cmp::Ordering;

use std::collections::{BTreeMap, BTreeSet, HashMap};

use std::fmt;

use std::panic::{AssertUnwindSafe, catch_unwind};

use std::sync::Arc;
