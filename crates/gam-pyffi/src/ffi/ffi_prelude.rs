//! Shared pyo3 / numpy / ndarray / `gam`-engine imports for the FFI boundary.
//!
//! Every FFI submodule pulls this in via `use crate::ffi_prelude::*;` so the
//! large, hand-curated set of engine re-exports lives in exactly one place
//! instead of being duplicated across each concern module. Re-exports are
//! `pub(crate)` so the glob is visible to sibling FFI modules but never
//! escapes the crate.

pub(crate) use csv::StringRecord;

pub(crate) use faer::Side;

pub(crate) use gam::solver::estimate::{
    EstimationError, ExternalOptimOptions,
    optimize_external_designwith_heuristic_log_lambdas,
};

pub(crate) use gam::linalg::faer_ndarray::{
    array2_to_matmut, factorize_symmetricwith_fallback,
};

pub(crate) use gam::families::inverse_link::{apply_inverse_link_spec_vec, apply_inverse_link_vec};

pub(crate) use gam::families::survival::predict::fit_result_from_saved_model_for_prediction;

pub(crate) use gam::solver::gaussian_reml::{
    GaussianRemlMultiBackwardProblem, build_gaussian_reml_eigen_cache_batched,
    gaussian_reml_blocks_orthogonal_shared_scale, gaussian_reml_free_b_score,
    gaussian_reml_multi_closed_form_backward, gaussian_reml_multi_closed_form_backward_batch,
    gaussian_reml_multi_closed_form_backward_from_fit, gaussian_reml_multi_closed_form_with_cache,
};

pub(crate) use gam::geometry::manifold::GeometryError as EngineGeometryError;

pub(crate) use gam::geometry::poincare::{
    conformal_factor as poincare_conformal_factor_impl,
    distance_batch as poincare_distance_batch_impl, exp_map as poincare_exp_map_impl,
    exp_map_batch as poincare_exp_map_batch_impl, exp_origin as poincare_exp_origin_impl,
    from_lorentz as poincare_from_lorentz_impl, log_map as poincare_log_map_impl,
    log_map_batch as poincare_log_map_batch_impl, log_origin as poincare_log_origin_impl,
    lorentz_decode_backward as poincare_lorentz_decode_backward_impl,
    lorentz_decode_forward as poincare_lorentz_decode_forward_impl,
    lorentz_exp_origin as poincare_lorentz_exp_origin_impl,
    lorentz_log_origin as poincare_lorentz_log_origin_impl,
    metric_tensor_batch as poincare_metric_tensor_batch_impl,
    mobius_add as poincare_mobius_add_impl, poincare_distance as poincare_distance_impl,
    project_into_ball as poincare_project_into_ball_impl,
    project_into_ball_batch as poincare_project_into_ball_batch_impl,
    tangent_decode_backward as poincare_tangent_decode_backward_impl,
    tangent_decode_forward as poincare_tangent_decode_forward_impl,
    to_lorentz as poincare_to_lorentz_impl,
};

pub(crate) use gam::geometry::simplex::{closure as simplex_closure, simplex_frechet_mean};

pub(crate) use gam::sample::{NutsConfig, NutsResult};

pub(crate) use gam::data::{
    ColumnKindTag, DataSchema, EncodedDataset, SchemaColumn, UnseenCategoryPolicy,
    encode_recordswith_schema,
};

pub(crate) use gam::inference::model::{
    FittedFamily, FittedModel, FittedModelPayload, GroupMetadata, PredictModelClass,
};

pub(crate) use gam::inference::model_extension::ExtendGroupRequest;

pub(crate) use gam_predict::posterior_bands::{self, PosteriorPredictBandsPayload};

pub(crate) use gam_predict::FittedModelPredictExt;
pub(crate) use gam_predict::input::{
    build_predict_input_for_model, build_transformation_normal_observed_scores,
};

pub(crate) use gam::geometry::sinkhorn_barycenter::{
    circular_cost as sinkhorn_circular_cost_impl, euclidean_cost as sinkhorn_euclidean_cost_impl,
    geodesic_sphere_cost as sinkhorn_geodesic_sphere_cost_impl,
    sinkhorn_barycenter as sinkhorn_barycenter_impl,
    sinkhorn_barycenter_vjp as sinkhorn_barycenter_vjp_impl,
};

pub(crate) use gam::report::render_html;

pub(crate) use gam::terms::smooth::{TermCollectionSpec, smooth_term_feature_cols};
// #1521: relocated DOWN into gam_terms::smooth (was families::...::drivers).
pub(crate) use gam::terms::smooth::{
    build_term_collection_derivative_design, build_term_collection_design,
};

pub(crate) use gam::families::fit_orchestration::descriptors::build_analytic_penalty_registry_from_descriptors as build_analytic_penalty_registry_from_json;

pub(crate) use gam::terms::basis::{
    BasisOptions, CenterStrategy, Dense, DuchonBasisSpec, DuchonNullspaceOrder,
    DuchonOperatorPenaltySpec, MaternBasisSpec, MaternIdentifiability, MaternLengthScale, MaternNu,
    OneDimensionalBoundary, OperatorPenaltySpec, PeriodicBSplineBasisSpec, SpatialIdentifiability,
    SphereMethod, SphereWahbaKernel, SphericalSplineBasisSpec, SphericalSplineIdentifiability,
    bspline_derivative_penalty_matrix, bspline_tensor_first_derivative, build_duchon_basis,
    build_duchon_basis_mixed_periodicity_auto, build_duchon_basis_spec_chart,
    build_duchon_operator_penalty_matrices,
    build_matern_basis_literal_aniso, build_periodic_bspline_basis_1d,
    build_spherical_spline_basis, build_thin_plate_penalty_matrix, create_basis,
    cyclic_bspline_derivative_penalty_matrix,
    duchon_nullspace_order_from_m,
    duchon_sae_atom_basis_with_jet,
    matern_input_location_hessian_nd, matern_input_location_jet_nd,
    periodic_bspline_derivative_nd,
    periodic_bspline_first_derivative_nd,
    resolve_duchon_orders, select_spherical_farthest_point_centers, sphere_first_derivative_nd,
    spherical_spline_design_hessian, spherical_spline_design_jet,
};

pub(crate) use gam::terms::basis::input_loc_derivatives::contract_input_loc_gradient;

pub(crate) use gam::terms::basis::position_basis::{
    PositionBasisKind, PositionBasisLocations, PositionPenaltyRequest, ResolvedPositionBasis,
    resolve_position_basis,
    validate_position_period,
};

pub(crate) use gam::terms::basis::{
    duchon_cubic_default_with_periodicity,
    duchon_function_norm_penalty as core_duchon_function_norm_penalty,
};

pub(crate) use gam::terms::decoders::interchange_decoder::{
    InterchangeDecodeForward as CoreInterchangeDecodeForward,
    InterchangeSwapForward as CoreInterchangeSwapForward,
    interchange_decode_backward as core_interchange_decode_backward,
    interchange_decode_forward as core_interchange_decode_forward,
    interchange_swap_backward as core_interchange_swap_backward,
    interchange_swap_forward as core_interchange_swap_forward,
};

pub(crate) use gam::terms::latent::{AuxPriorFamily, aux_prior_targets};
pub(crate) use gam::terms::basis::latent_design::{latent_basis_kind, latent_input_location_jet, periodic_bspline_basis_dense_via_spec, build_latent_duchon_design, build_latent_forward_design};
pub(crate) use gam::terms::latent::{LatentAuxStrengthState, latent_aux_prior_stats, ValidatedDimSelectionPrecisions, latent_prior_score_and_aux_state_for_t, latent_analytic_penalty_value};
pub(crate) use gam::families::latent_outer::{LatentOuterProblem, LatentOuterObjective, latent_manifold_periodic_descriptor, build_latent_outer_manifold, latent_spectral_seed_start, gaussian_reml_weight_vector_local, latent_scalar_weights_with_fisher, latent_row_weights, validate_dense_fisher_w, gaussian_reml_fit_latent_impl};

pub(crate) use gam::terms::dictionary::{
    LinearDictionaryAssignment, LinearDictionaryConfig, LinearDictionaryError,
    fit_linear_dictionary, linear_dictionary_transform,
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
    AmbientSphereHarmonicEvaluator, GumbelTemperatureSchedule, SaeAtomGeometryPlan,
    SaeFisherRowMetricRequest, SaeFitAssignmentKind, SaeFitSeedReport, SaeFitSeedRequest,
    SaeMinimalSeedReport, SaeMinimalSeedRequest, build_sae_fisher_row_metric,
    SaeBasisEvaluator, build_sae_fit_seed, build_sae_minimal_seed, sae_atom_basis_kind_name,
    sae_fitted_atom_plans,
};

pub(crate) use gam::terms::smooth::BlockwisePenalty;

pub(crate) use gam::terms::basis::matern_gradient::{
    MaternBasisGradientTarget, StreamingMaternBasisGradientEvaluator,
};
pub(crate) use gam::terms::decoders::gated_decoder::GatedSAEDecoder;
pub(crate) use gam::terms::{
    AnalyticPenalty as AnalyticPenaltyTrait, AnalyticPenaltyKind, AnalyticPenaltyRegistry,
    EdgeRestriction as CoreEdgeRestriction, IsometryEvaluationOrder,
    IvaeRidgeMeanGauge as IvaeRidgeMeanGaugePenalty,
    MechanismSparsityPenalty as CoreMechanismSparsityPenalty, ParametricRowPrecisionPriorPenalty,
    PenaltyTier, PsiSlice, RowPrecisionPriorPenalty,
    SheafConsistencyPenalty as CoreSheafConsistencyPenalty,
};

pub(crate) use gam::types::{InverseLink, LikelihoodSpec, ResponseFamily, RhoPrior, StandardLink};

pub(crate) use gam::families::fit_orchestration::{
    FitConfig, FitRequest, WorkflowError, materialize, materialize_structural,
    resolve_offset_column, resolve_weight_column,
};

pub(crate) use ndarray::{
    Array1, Array2, Array3, ArrayView1, ArrayView2, ArrayView3, ArrayView4, Axis, s,
};

pub(crate) use numpy::{
    IntoPyArray, PyArray1, PyArray2, PyArray3, PyArray4, PyArrayDyn, PyArrayMethods,
    PyReadonlyArray1, PyReadonlyArray2, PyReadonlyArray3, PyReadonlyArray4, PyReadonlyArrayDyn,
};

pub(crate) use pyo3::IntoPyObjectExt;

pub(crate) type PyObject = pyo3::Py<pyo3::PyAny>;

pub(crate) use pyo3::exceptions::{PyTypeError, PyValueError};

pub(crate) use pyo3::prelude::*;

pub(crate) use pyo3::types::{PyAny, PyBool, PyBytes, PyDict, PyInt, PyList, PyString, PyTuple, PyType};

pub(crate) use serde::{Deserialize, Serialize};

pub(crate) use std::collections::{BTreeMap, BTreeSet, HashMap};

pub(crate) use std::panic::{AssertUnwindSafe, catch_unwind};

pub(crate) use std::sync::Arc;
