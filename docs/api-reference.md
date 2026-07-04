# API reference

Generated from docstrings and type hints in the `gamfit` source. See the
topical guides for narrative explanations.

## Top-level functions

::: gamfit.fit
    options:
      show_root_heading: true
      heading_level: 3

::: gamfit.fit_array
    options:
      show_root_heading: true
      heading_level: 3

::: gamfit.load
    options:
      show_root_heading: true
      heading_level: 3

::: gamfit.loads
    options:
      show_root_heading: true
      heading_level: 3

::: gamfit.save
    options:
      show_root_heading: true
      heading_level: 3

::: gamfit.load_posterior
    options:
      show_root_heading: true
      heading_level: 3

::: gamfit.competing_risks_cif
    options:
      show_root_heading: true
      heading_level: 3

::: gamfit.compare_models
    options:
      show_root_heading: true
      heading_level: 3

::: gamfit.cross_fit_shared_precision_groups
    options:
      show_root_heading: true
      heading_level: 3

::: gamfit.validate_formula
    options:
      show_root_heading: true
      heading_level: 3

::: gamfit.build_info
    options:
      show_root_heading: true
      heading_level: 3

::: gamfit.cuda_diagnostics
    options:
      show_root_heading: true
      heading_level: 3

::: gamfit.format_cuda_diagnostics
    options:
      show_root_heading: true
      heading_level: 3

::: gamfit.cuda_subprocess_env
    options:
      show_root_heading: true
      heading_level: 3

::: gamfit.cuda_subprocess_library_dirs
    options:
      show_root_heading: true
      heading_level: 3

::: gamfit.explain_error
    options:
      show_root_heading: true
      heading_level: 3

::: gamfit.mechanism_sparsity_jacobian
    options:
      show_root_heading: true
      heading_level: 3

::: gamfit.linear_dictionary_fit
    options:
      show_root_heading: true
      heading_level: 3

::: gamfit.conditional_prior_ivae
    options:
      show_root_heading: true
      heading_level: 3

::: gamfit.glm_full_conformal
    options:
      show_root_heading: true
      heading_level: 3

::: gamfit.lawley_bartlett_factor
    options:
      show_root_heading: true
      heading_level: 3

::: gamfit.layer_transport_fit
    options:
      show_root_heading: true
      heading_level: 3

::: gamfit.layer_transport_ladder
    options:
      show_root_heading: true
      heading_level: 3

::: gamfit.sae_checkpoint_dynamics
    options:
      show_root_heading: true
      heading_level: 3

## Fitted model

::: gamfit.Model
    options:
      show_root_heading: true
      heading_level: 3
      members_order: source
      filters:
        - "!^_"
        - "^__init__$"
        - "^__repr__$"

::: gamfit.SurvivalPrediction
    options:
      show_root_heading: true
      heading_level: 3
      members_order: source

::: gamfit.PredictionResult
    options:
      show_root_heading: true
      heading_level: 3
      members_order: source

::: gamfit.CompetingRisksPrediction
    options:
      show_root_heading: true
      heading_level: 3
      members_order: source

::: gamfit.CompetingRisksCIF
    options:
      show_root_heading: true
      heading_level: 3
      members_order: source

::: gamfit.MultinomialModel
    options:
      show_root_heading: true
      heading_level: 3
      members_order: source

::: gamfit.MultinomialPrediction
    options:
      show_root_heading: true
      heading_level: 3
      members_order: source

## Posterior sampling

::: gamfit.SamplingConfig
    options:
      show_root_heading: true
      heading_level: 3

::: gamfit.PosteriorSamples
    options:
      show_root_heading: true
      heading_level: 3
      members_order: source

::: gamfit.PairedPosteriorSamples
    options:
      show_root_heading: true
      heading_level: 3
      members_order: source

::: gamfit.PosteriorPredictive
    options:
      show_root_heading: true
      heading_level: 3
      members_order: source

::: gamfit.CumulativeIncidenceDraws
    options:
      show_root_heading: true
      heading_level: 3
      members_order: source

## Diagnostics and metadata

::: gamfit.Summary
    options:
      show_root_heading: true
      heading_level: 3

::: gamfit.Diagnostics
    options:
      show_root_heading: true
      heading_level: 3

::: gamfit.SchemaCheck
    options:
      show_root_heading: true
      heading_level: 3

::: gamfit.SchemaIssue
    options:
      show_root_heading: true
      heading_level: 3

::: gamfit.FormulaValidation
    options:
      show_root_heading: true
      heading_level: 3

::: gamfit.SharedPrecisionGroup
    options:
      show_root_heading: true
      heading_level: 3

::: gamfit.LinearDictionaryFit
    options:
      show_root_heading: true
      heading_level: 3

## Basis and ridge primitives

::: gamfit.bspline_basis
    options:
      show_root_heading: true
      heading_level: 3

::: gamfit.bspline_basis_derivative
    options:
      show_root_heading: true
      heading_level: 3

::: gamfit.duchon_basis
    options:
      show_root_heading: true
      heading_level: 3

::: gamfit.duchon_function_norm_penalty
    options:
      show_root_heading: true
      heading_level: 3

::: gamfit.matern_basis
    options:
      show_root_heading: true
      heading_level: 3

::: gamfit.periodic_spline_curve_basis
    options:
      show_root_heading: true
      heading_level: 3

::: gamfit.sphere_basis
    options:
      show_root_heading: true
      heading_level: 3

::: gamfit.sphere_basis_jet
    options:
      show_root_heading: true
      heading_level: 3

::: gamfit.smoothness_penalty
    options:
      show_root_heading: true
      heading_level: 3

::: gamfit.gaussian_weighted_ridge
    options:
      show_root_heading: true
      heading_level: 3

::: gamfit.gaussian_weighted_ridge_batch
    options:
      show_root_heading: true
      heading_level: 3

## Gaussian REML primitives

::: gamfit.gaussian_reml_fit
    options:
      show_root_heading: true
      heading_level: 3

::: gamfit.gaussian_reml_fit_backward
    options:
      show_root_heading: true
      heading_level: 3

::: gamfit.gaussian_reml_fit_batched
    options:
      show_root_heading: true
      heading_level: 3

::: gamfit.gaussian_reml_fit_batched_backward
    options:
      show_root_heading: true
      heading_level: 3

::: gamfit.gaussian_reml_fit_positions
    options:
      show_root_heading: true
      heading_level: 3

::: gamfit.gaussian_reml_fit_positions_backward
    options:
      show_root_heading: true
      heading_level: 3

::: gamfit.gaussian_reml_fit_positions_batched
    options:
      show_root_heading: true
      heading_level: 3

::: gamfit.gaussian_reml_fit_positions_batched_backward
    options:
      show_root_heading: true
      heading_level: 3

::: gamfit.gaussian_reml_fit_blocks_forward
    options:
      show_root_heading: true
      heading_level: 3

::: gamfit.gaussian_reml_fit_blocks_backward
    options:
      show_root_heading: true
      heading_level: 3

::: gamfit.gaussian_reml_fit_with_constraints_forward
    options:
      show_root_heading: true
      heading_level: 3

::: gamfit.gaussian_reml_fit_with_constraints_backward
    options:
      show_root_heading: true
      heading_level: 3

::: gamfit.gaussian_reml_fit_formula
    options:
      show_root_heading: true
      heading_level: 3

::: gamfit.gaussian_reml_fit_latent
    options:
      show_root_heading: true
      heading_level: 3

::: gamfit.gaussian_reml_fit_latent_backward
    options:
      show_root_heading: true
      heading_level: 3

::: gamfit.gaussian_reml_optimize_latent
    options:
      show_root_heading: true
      heading_level: 3

::: gamfit.glm_reml_fit_latent
    options:
      show_root_heading: true
      heading_level: 3

::: gamfit.glm_reml_fit_latent_backward
    options:
      show_root_heading: true
      heading_level: 3

## Smooth term builders

::: gamfit.smooth.Smooth
    options:
      show_root_heading: true
      heading_level: 3

::: gamfit.smooth.BSpline
    options:
      show_root_heading: true
      heading_level: 3

::: gamfit.smooth.TensorBSpline
    options:
      show_root_heading: true
      heading_level: 3

::: gamfit.smooth.PeriodicSplineCurve
    options:
      show_root_heading: true
      heading_level: 3

::: gamfit.smooth.Duchon
    options:
      show_root_heading: true
      heading_level: 3

::: gamfit.smooth.Matern
    options:
      show_root_heading: true
      heading_level: 3

::: gamfit.smooth.Sphere
    options:
      show_root_heading: true
      heading_level: 3

::: gamfit.smooth.MeasureJet
    options:
      show_root_heading: true
      heading_level: 3

::: gamfit.smooth.Pca
    options:
      show_root_heading: true
      heading_level: 3

::: gamfit.smooth.LatentCoord
    options:
      show_root_heading: true
      heading_level: 3

::: gamfit.smooth.Categorical
    options:
      show_root_heading: true
      heading_level: 3

::: gamfit.smooth.ShapeConstraintLiteral
    options:
      show_root_heading: true
      heading_level: 3

::: gamfit.TermBlock
    options:
      show_root_heading: true
      heading_level: 3

## Topology and smooth descriptors

::: gamfit.Circle
    options:
      show_root_heading: true
      heading_level: 3

::: gamfit.Cylinder
    options:
      show_root_heading: true
      heading_level: 3

::: gamfit.Torus
    options:
      show_root_heading: true
      heading_level: 3

::: gamfit.TopologySphere
    options:
      show_root_heading: true
      heading_level: 3

::: gamfit.EuclideanPatch
    options:
      show_root_heading: true
      heading_level: 3

::: gamfit.PeriodicHarmonic
    options:
      show_root_heading: true
      heading_level: 3

::: gamfit.Fourier
    options:
      show_root_heading: true
      heading_level: 3

## Penalties and latent-coordinate tools

::: gamfit.ARDPenalty
    options:
      show_root_heading: true
      heading_level: 3

::: gamfit.IBPPenalty
    options:
      show_root_heading: true
      heading_level: 3

::: gamfit.OrthogonalityPenalty
    options:
      show_root_heading: true
      heading_level: 3

::: gamfit.BlockOrthogonalityPenalty
    options:
      show_root_heading: true
      heading_level: 3

::: gamfit.SparsityPenalty
    options:
      show_root_heading: true
      heading_level: 3

::: gamfit.SoftmaxAssignmentSparsityPenalty
    options:
      show_root_heading: true
      heading_level: 3

::: gamfit.TopKActivationPenalty
    options:
      show_root_heading: true
      heading_level: 3

::: gamfit.JumpReLUPenalty
    options:
      show_root_heading: true
      heading_level: 3

::: gamfit.NuclearNormPenalty
    options:
      show_root_heading: true
      heading_level: 3

::: gamfit.ScadMcpPenalty
    options:
      show_root_heading: true
      heading_level: 3

::: gamfit.CompositePenalty
    options:
      show_root_heading: true
      heading_level: 3

::: gamfit.Penalty
    options:
      show_root_heading: true
      heading_level: 3

::: gamfit.AnalyticPenaltyKind
    options:
      show_root_heading: true
      heading_level: 3

::: gamfit.ScalarWeightSchedule
    options:
      show_root_heading: true
      heading_level: 3

::: gamfit.PENALTY_MANIFEST
    options:
      show_root_heading: true
      heading_level: 3

::: gamfit.AuxConditionalPriorPenalty
    options:
      show_root_heading: true
      heading_level: 3

::: gamfit.ParametricAuxConditionalPriorPenalty
    options:
      show_root_heading: true
      heading_level: 3

::: gamfit.BlockSparsityPenalty
    options:
      show_root_heading: true
      heading_level: 3

::: gamfit.IBPAssignmentPenalty
    options:
      show_root_heading: true
      heading_level: 3

::: gamfit.IsometryPenalty
    options:
      show_root_heading: true
      heading_level: 3

::: gamfit.IvaeRidgeMeanGauge
    options:
      show_root_heading: true
      heading_level: 3

::: gamfit.MechanismSparsityPenalty
    options:
      show_root_heading: true
      heading_level: 3

::: gamfit.TotalVariationPenalty
    options:
      show_root_heading: true
      heading_level: 3

::: gamfit.SheafConsistencyPenalty
    options:
      show_root_heading: true
      heading_level: 3

::: gamfit.GatedSAEDecoder
    options:
      show_root_heading: true
      heading_level: 3

::: gamfit.torch.modules.AdaptiveTopK
    options:
      show_root_heading: true
      heading_level: 3

## Descriptor protocol

::: gamfit.BasisDescriptor
    options:
      show_root_heading: true
      heading_level: 3

::: gamfit.ManifoldDescriptor
    options:
      show_root_heading: true
      heading_level: 3

::: gamfit.PenaltyDescriptor
    options:
      show_root_heading: true
      heading_level: 3

::: gamfit.BlockOrthogonalityDescriptor
    options:
      show_root_heading: true
      heading_level: 3

::: gamfit.MechanismSparsityDescriptor
    options:
      show_root_heading: true
      heading_level: 3

## scikit-learn integration

::: gamfit.sklearn.GAMRegressor
    options:
      show_root_heading: true
      heading_level: 3
      inherited_members: false

::: gamfit.sklearn.GAMClassifier
    options:
      show_root_heading: true
      heading_level: 3
      inherited_members: false

## Exceptions

::: gamfit.GamError
    options:
      show_root_heading: true
      heading_level: 3

::: gamfit.FormulaError
    options:
      show_root_heading: true
      heading_level: 3

::: gamfit.SchemaMismatchError
    options:
      show_root_heading: true
      heading_level: 3

::: gamfit.PredictionError
    options:
      show_root_heading: true
      heading_level: 3

::: gamfit.RustExtensionUnavailableError
    options:
      show_root_heading: true
      heading_level: 3

::: gamfit.AloError
    options:
      show_root_heading: true
      heading_level: 3

::: gamfit.ArrowSchurError
    options:
      show_root_heading: true
      heading_level: 3

::: gamfit.BasisError
    options:
      show_root_heading: true
      heading_level: 3

::: gamfit.CacheStoreError
    options:
      show_root_heading: true
      heading_level: 3

::: gamfit.CalibratorError
    options:
      show_root_heading: true
      heading_level: 3

::: gamfit.ColumnNotFoundError
    options:
      show_root_heading: true
      heading_level: 3

::: gamfit.CorrectedCovarianceError
    options:
      show_root_heading: true
      heading_level: 3

::: gamfit.CubicCellKernelError
    options:
      show_root_heading: true
      heading_level: 3

::: gamfit.CustomFamilyError
    options:
      show_root_heading: true
      heading_level: 3

::: gamfit.DataError
    options:
      show_root_heading: true
      heading_level: 3

::: gamfit.DeviationRuntimeError
    options:
      show_root_heading: true
      heading_level: 3

::: gamfit.EigendecompositionError
    options:
      show_root_heading: true
      heading_level: 3

::: gamfit.FittedModelError
    options:
      show_root_heading: true
      heading_level: 3

::: gamfit.GamlssError
    options:
      show_root_heading: true
      heading_level: 3

::: gamfit.GeometryError
    options:
      show_root_heading: true
      heading_level: 3

::: gamfit.GpuError
    options:
      show_root_heading: true
      heading_level: 3

::: gamfit.GradientUnavailableError
    options:
      show_root_heading: true
      heading_level: 3

::: gamfit.HessianNotPositiveDefiniteError
    options:
      show_root_heading: true
      heading_level: 3

::: gamfit.HmcError
    options:
      show_root_heading: true
      heading_level: 3

::: gamfit.IdentifiabilityCompilerError
    options:
      show_root_heading: true
      heading_level: 3

::: gamfit.IllConditionedError
    options:
      show_root_heading: true
      heading_level: 3

::: gamfit.IntegrationError
    options:
      show_root_heading: true
      heading_level: 3

::: gamfit.InvalidConfigurationError
    options:
      show_root_heading: true
      heading_level: 3

::: gamfit.InvalidInputError
    options:
      show_root_heading: true
      heading_level: 3

::: gamfit.InvalidSpecificationError
    options:
      show_root_heading: true
      heading_level: 3

::: gamfit.JointPenaltyError
    options:
      show_root_heading: true
      heading_level: 3

::: gamfit.LatentSurvivalError
    options:
      show_root_heading: true
      heading_level: 3

::: gamfit.LayoutError
    options:
      show_root_heading: true
      heading_level: 3

::: gamfit.LinearAlgebraError
    options:
      show_root_heading: true
      heading_level: 3

::: gamfit.LinearSystemSolveError
    options:
      show_root_heading: true
      heading_level: 3

::: gamfit.LognormalKernelError
    options:
      show_root_heading: true
      heading_level: 3

::: gamfit.MapUniquenessError
    options:
      show_root_heading: true
      heading_level: 3

::: gamfit.MatrixError
    options:
      show_root_heading: true
      heading_level: 3

::: gamfit.MatrixMaterializationError
    options:
      show_root_heading: true
      heading_level: 3

::: gamfit.MissingDependencyError
    options:
      show_root_heading: true
      heading_level: 3

::: gamfit.ModelOverparameterizedError
    options:
      show_root_heading: true
      heading_level: 3

::: gamfit.MonotoneRootError
    options:
      show_root_heading: true
      heading_level: 3

::: gamfit.OuterStrategyError
    options:
      show_root_heading: true
      heading_level: 3

::: gamfit.ParameterConstraintError
    options:
      show_root_heading: true
      heading_level: 3

::: gamfit.PenaltySpectrumError
    options:
      show_root_heading: true
      heading_level: 3

::: gamfit.PerfectSeparationError
    options:
      show_root_heading: true
      heading_level: 3

::: gamfit.PirlsConvergenceError
    options:
      show_root_heading: true
      heading_level: 3

::: gamfit.PredictInputError
    options:
      show_root_heading: true
      heading_level: 3

::: gamfit.RemlConvergenceError
    options:
      show_root_heading: true
      heading_level: 3

::: gamfit.ScaleDesignError
    options:
      show_root_heading: true
      heading_level: 3

::: gamfit.SmoothError
    options:
      show_root_heading: true
      heading_level: 3

::: gamfit.SurvivalConstructionError
    options:
      show_root_heading: true
      heading_level: 3

::: gamfit.SurvivalError
    options:
      show_root_heading: true
      heading_level: 3

::: gamfit.SurvivalLocationScaleError
    options:
      show_root_heading: true
      heading_level: 3

::: gamfit.SurvivalMarginalSlopeError
    options:
      show_root_heading: true
      heading_level: 3

::: gamfit.SurvivalPredictError
    options:
      show_root_heading: true
      heading_level: 3

::: gamfit.TermBuilderError
    options:
      show_root_heading: true
      heading_level: 3

::: gamfit.TransformationNormalError
    options:
      show_root_heading: true
      heading_level: 3

::: gamfit.UnsupportedLinkError
    options:
      show_root_heading: true
      heading_level: 3

## Manifold SAE

See the [Manifold SAE dictionary guide](manifold-sae.md) for the narrative.

::: gamfit.sae_manifold_fit
    options:
      show_root_heading: true
      heading_level: 3

::: gamfit.atom_trust_scores
    options:
      show_root_heading: true
      heading_level: 3

::: gamfit.sae_trust_diagnostics
    options:
      show_root_heading: true
      heading_level: 3

::: gamfit.ManifoldSAE
    options:
      show_root_heading: true
      heading_level: 3
      members_order: source
      filters:
        - "!^_"
        - "^__init__$"
        - "^__repr__$"

::: gamfit.SaeManifoldAtomFit
    options:
      show_root_heading: true
      heading_level: 3

::: gamfit.SaeManifoldFitResult
    options:
      show_root_heading: true
      heading_level: 3

::: gamfit.GumbelTemperatureSchedule
    options:
      show_root_heading: true
      heading_level: 3

::: gamfit.gumbel_geometric_schedule
    options:
      show_root_heading: true
      heading_level: 3

::: gamfit.gumbel_linear_schedule
    options:
      show_root_heading: true
      heading_level: 3

::: gamfit.gumbel_reciprocal_iter_schedule
    options:
      show_root_heading: true
      heading_level: 3

## Equivariant smooths

::: gamfit.EquivariantPenalty
    options:
      show_root_heading: true
      heading_level: 3

::: gamfit.GaugeCompanion
    options:
      show_root_heading: true
      heading_level: 3

::: gamfit.LieAtom
    options:
      show_root_heading: true
      heading_level: 3

::: gamfit.equivariant_smooth
    options:
      show_root_heading: true
      heading_level: 3

::: gamfit.gauge_companion
    options:
      show_root_heading: true
      heading_level: 3

::: gamfit.rho_so2
    options:
      show_root_heading: true
      heading_level: 3

::: gamfit.rho_so2_jvp
    options:
      show_root_heading: true
      heading_level: 3

::: gamfit.rho_so3
    options:
      show_root_heading: true
      heading_level: 3

::: gamfit.rho_so3_jvp
    options:
      show_root_heading: true
      heading_level: 3

## Manifolds

::: gamfit.CircleManifold
    options:
      show_root_heading: true
      heading_level: 3

::: gamfit.EuclideanManifold
    options:
      show_root_heading: true
      heading_level: 3

::: gamfit.GrassmannManifold
    options:
      show_root_heading: true
      heading_level: 3

::: gamfit.ProductManifold
    options:
      show_root_heading: true
      heading_level: 3

::: gamfit.SpdManifold
    options:
      show_root_heading: true
      heading_level: 3

::: gamfit.SphereManifold
    options:
      show_root_heading: true
      heading_level: 3

::: gamfit.StiefelManifold
    options:
      show_root_heading: true
      heading_level: 3

::: gamfit.TorusManifold
    options:
      show_root_heading: true
      heading_level: 3

## Topology selection

::: gamfit.BasisSpec
    options:
      show_root_heading: true
      heading_level: 3

::: gamfit.ScoreKind
    options:
      show_root_heading: true
      heading_level: 3

::: gamfit.ScoreScale
    options:
      show_root_heading: true
      heading_level: 3

::: gamfit.SelectTopologyResult
    options:
      show_root_heading: true
      heading_level: 3

::: gamfit.TopologyAutoSelector
    options:
      show_root_heading: true
      heading_level: 3

::: gamfit.TopologyAutoSelectorRank
    options:
      show_root_heading: true
      heading_level: 3

::: gamfit.TopologyAutoSelectorResult
    options:
      show_root_heading: true
      heading_level: 3

::: gamfit.TopologyStack
    options:
      show_root_heading: true
      heading_level: 3

::: gamfit.select_topology
    options:
      show_root_heading: true
      heading_level: 3

::: gamfit.stack_topologies
    options:
      show_root_heading: true
      heading_level: 3

## Identifiability

::: gamfit.IdentifiabilityReport
    options:
      show_root_heading: true
      heading_level: 3

::: gamfit.IdentifiabilityTheoremResult
    options:
      show_root_heading: true
      heading_level: 3

::: gamfit.IdentifiableFactorFitResult
    options:
      show_root_heading: true
      heading_level: 3

::: gamfit.identifiability_check
    options:
      show_root_heading: true
      heading_level: 3

::: gamfit.identifiable_factor_fit
    options:
      show_root_heading: true
      heading_level: 3

## Structure discovery

::: gamfit.atom_birth_gate
    options:
      show_root_heading: true
      heading_level: 3

::: gamfit.e_bh_dictionary_certificate
    options:
      show_root_heading: true
      heading_level: 3

::: gamfit.expected_resolution_budget
    options:
      show_root_heading: true
      heading_level: 3

::: gamfit.log_e_from_p_value
    options:
      show_root_heading: true
      heading_level: 3

::: gamfit.plan_probe_for_contested_claim
    options:
      show_root_heading: true
      heading_level: 3

::: gamfit.select_probe_by_expected_evidence
    options:
      show_root_heading: true
      heading_level: 3

::: gamfit.split_likelihood_log_e
    options:
      show_root_heading: true
      heading_level: 3

## Partial supervision

::: gamfit.PartialSupervisionExample
    options:
      show_root_heading: true
      heading_level: 3

::: gamfit.PartialSupervisionFit
    options:
      show_root_heading: true
      heading_level: 3

::: gamfit.SaeSupervisedFit
    options:
      show_root_heading: true
      heading_level: 3

::: gamfit.partial_supervision
    options:
      show_root_heading: true
      heading_level: 3

::: gamfit.sae_supervised
    options:
      show_root_heading: true
      heading_level: 3

## Manifold SAE helpers

::: gamfit.featurize
    options:
      show_root_heading: true
      heading_level: 3

::: gamfit.sae_fit
    options:
      show_root_heading: true
      heading_level: 3

::: gamfit.plot
    options:
      show_root_heading: true
      heading_level: 3

::: gamfit.plot_atom
    options:
      show_root_heading: true
      heading_level: 3

::: gamfit.plot_fit
    options:
      show_root_heading: true
      heading_level: 3

## Distillation and staged coordinates

::: gamfit.DistilledEncoder
    options:
      show_root_heading: true
      heading_level: 3

::: gamfit.EncoderFallbackStats
    options:
      show_root_heading: true
      heading_level: 3

::: gamfit.CtnStage1
    options:
      show_root_heading: true
      heading_level: 3

## Compositional smooth specs

::: gamfit.Smooth
    options:
      show_root_heading: true
      heading_level: 3

::: gamfit.SmoothSum
    options:
      show_root_heading: true
      heading_level: 3

::: gamfit.SmoothSpec
    options:
      show_root_heading: true
      heading_level: 3

## Torch-optional primitives

These symbols are re-exported at the top level for convenience but their
implementations live behind the optional `torch` dependency; accessing them
without torch installed raises `AttributeError`. See [torch.md](torch.md).

::: gamfit.crosscoder.Crosscoder
    options:
      show_root_heading: true
      heading_level: 3

::: gamfit.torch.interchange.InterchangeSwapDecoder
    options:
      show_root_heading: true
      heading_level: 3

::: gamfit.torch.hyperbolic.PoincareAtoms
    options:
      show_root_heading: true
      heading_level: 3

## Response geometry

::: gamfit.ResponseGeometryModel
    options:
      show_root_heading: true
      heading_level: 3
      inherited_members: false

::: gamfit.clr
    options:
      show_root_heading: true
      heading_level: 3

::: gamfit.alr
    options:
      show_root_heading: true
      heading_level: 3

::: gamfit.closure
    options:
      show_root_heading: true
      heading_level: 3

::: gamfit.simplex_frechet_mean
    options:
      show_root_heading: true
      heading_level: 3

::: gamfit.sphere_frechet_mean
    options:
      show_root_heading: true
      heading_level: 3
