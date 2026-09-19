# API reference

Generated from docstrings and type hints in the `gamfit` source. See the
topical guides for narrative explanations.

## Namespace layout

`import gamfit` exposes only the fit / load entry points and the fitted-model
classes (`gamfit.__all__`):

`fit`, `fit_array`, `load`, `loads`, `validate_formula`, `explain_error`,
`compare_models`, `build_info`, `competing_risks_cif`, `fit_event_history`,
`fit_joint_event_model`, `load_joint_event_model`, `Model`,
`MultinomialModel`, `ResponseGeometryModel`, `EventHistoryModel`,
`JointEventModel`, `CtnStage1`, `__version__`.

Everything else lives in a public submodule, imported on first access
(`gamfit.errors.GamError`, or `from gamfit.errors import GamError`):

| Submodule | Contents |
| --- | --- |
| `gamfit.errors` | exception hierarchy rooted at `GamError`, plus `GamInferenceWarning` |
| `gamfit.results` | prediction, summary, diagnostics, and posterior-sample result types |
| `gamfit.plot` | matplotlib plotting (optional `gamfit[plot]` extra) |
| `gamfit.smooth` | formula term specifications (`BSpline`, `Duchon`, `LatentCoord`, ...) |
| `gamfit.basis` | raw basis / penalty matrix builders and compositional `Smooth` |
| `gamfit.penalties` | analytic latent-coordinate penalties |
| `gamfit.reml` | array-level Gaussian / GLM REML and ridge primitives |
| `gamfit.topology` | latent topologies and topology selection |
| `gamfit.manifolds`, `gamfit.geometry` | manifold descriptors |
| `gamfit.sae` | sparse-dictionary and manifold-SAE tools |
| `gamfit.identifiability` | identifiable latent-factor fits and checks |
| `gamfit.inference` | conformal, Bartlett, and shared-precision inference helpers |
| `gamfit.response_geometry` | compositional / spherical response transforms |
| `gamfit.cuda` | CUDA runtime diagnostics |
| `gamfit.diagnostics`, `gamfit.kernels`, `gamfit.examples` | diagnostic tools, kernels, worked examples |
| `gamfit.sklearn`, `gamfit.torch` | scikit-learn and PyTorch integrations (optional extras) |

## Entry points

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

::: gamfit.competing_risks_cif
    options:
      show_root_heading: true
      heading_level: 3

::: gamfit.compare_models
    options:
      show_root_heading: true
      heading_level: 3

::: gamfit.inference.cross_fit_shared_precision_groups
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

::: gamfit.cuda.cuda_diagnostics
    options:
      show_root_heading: true
      heading_level: 3

::: gamfit.cuda.format_cuda_diagnostics
    options:
      show_root_heading: true
      heading_level: 3

::: gamfit.cuda.cuda_subprocess_env
    options:
      show_root_heading: true
      heading_level: 3

::: gamfit.cuda.cuda_subprocess_library_dirs
    options:
      show_root_heading: true
      heading_level: 3

::: gamfit.explain_error
    options:
      show_root_heading: true
      heading_level: 3

::: gamfit.identifiability.mechanism_sparsity_jacobian
    options:
      show_root_heading: true
      heading_level: 3

::: gamfit.sae.linear_dictionary_fit
    options:
      show_root_heading: true
      heading_level: 3

::: gamfit.identifiability.conditional_prior_ivae
    options:
      show_root_heading: true
      heading_level: 3

::: gamfit.inference.glm_full_conformal
    options:
      show_root_heading: true
      heading_level: 3

::: gamfit.inference.lawley_bartlett_factor
    options:
      show_root_heading: true
      heading_level: 3

::: gamfit.sae.layer_transport_fit
    options:
      show_root_heading: true
      heading_level: 3

::: gamfit.sae.layer_transport_ladder
    options:
      show_root_heading: true
      heading_level: 3

::: gamfit.sae.sae_checkpoint_dynamics
    options:
      show_root_heading: true
      heading_level: 3

::: gamfit.sae.audit_sae
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

::: gamfit.results.SurvivalPrediction
    options:
      show_root_heading: true
      heading_level: 3
      members_order: source

::: gamfit.results.PredictionResult
    options:
      show_root_heading: true
      heading_level: 3
      members_order: source

::: gamfit.results.CompetingRisksPrediction
    options:
      show_root_heading: true
      heading_level: 3
      members_order: source

::: gamfit.results.CompetingRisksCIF
    options:
      show_root_heading: true
      heading_level: 3
      members_order: source

::: gamfit.MultinomialModel
    options:
      show_root_heading: true
      heading_level: 3
      members_order: source

::: gamfit.results.MultinomialPrediction
    options:
      show_root_heading: true
      heading_level: 3
      members_order: source

## Posterior sampling

::: gamfit.results.SamplingConfig
    options:
      show_root_heading: true
      heading_level: 3

::: gamfit.results.PosteriorSamples
    options:
      show_root_heading: true
      heading_level: 3
      members_order: source

::: gamfit.results.PairedPosteriorSamples
    options:
      show_root_heading: true
      heading_level: 3
      members_order: source

::: gamfit.results.PosteriorPredictive
    options:
      show_root_heading: true
      heading_level: 3
      members_order: source

::: gamfit.results.CumulativeIncidenceDraws
    options:
      show_root_heading: true
      heading_level: 3
      members_order: source

## Diagnostics and metadata

::: gamfit.results.Summary
    options:
      show_root_heading: true
      heading_level: 3

::: gamfit.results.Diagnostics
    options:
      show_root_heading: true
      heading_level: 3

::: gamfit.results.SchemaCheck
    options:
      show_root_heading: true
      heading_level: 3

::: gamfit.results.SchemaIssue
    options:
      show_root_heading: true
      heading_level: 3

::: gamfit.results.FormulaValidation
    options:
      show_root_heading: true
      heading_level: 3

::: gamfit.inference.SharedPrecisionGroup
    options:
      show_root_heading: true
      heading_level: 3

::: gamfit.sae.LinearDictionaryFit
    options:
      show_root_heading: true
      heading_level: 3

## Basis and ridge primitives

::: gamfit.basis.bspline_basis
    options:
      show_root_heading: true
      heading_level: 3

::: gamfit.basis.bspline_basis_derivative
    options:
      show_root_heading: true
      heading_level: 3

::: gamfit.basis.duchon_basis
    options:
      show_root_heading: true
      heading_level: 3

::: gamfit.basis.duchon_function_norm_penalty
    options:
      show_root_heading: true
      heading_level: 3

::: gamfit.basis.matern_basis
    options:
      show_root_heading: true
      heading_level: 3

::: gamfit.basis.periodic_spline_curve_basis
    options:
      show_root_heading: true
      heading_level: 3

::: gamfit.basis.sphere_basis
    options:
      show_root_heading: true
      heading_level: 3

::: gamfit.basis.sphere_basis_jet
    options:
      show_root_heading: true
      heading_level: 3

::: gamfit.basis.smoothness_penalty
    options:
      show_root_heading: true
      heading_level: 3

::: gamfit.reml.gaussian_weighted_ridge
    options:
      show_root_heading: true
      heading_level: 3

::: gamfit.reml.gaussian_weighted_ridge_batch
    options:
      show_root_heading: true
      heading_level: 3

## Gaussian REML primitives

::: gamfit.reml.gaussian_reml_fit
    options:
      show_root_heading: true
      heading_level: 3

::: gamfit.reml.gaussian_reml_fit_backward
    options:
      show_root_heading: true
      heading_level: 3

::: gamfit.reml.gaussian_reml_fit_batched
    options:
      show_root_heading: true
      heading_level: 3

::: gamfit.reml.gaussian_reml_fit_batched_backward
    options:
      show_root_heading: true
      heading_level: 3

::: gamfit.reml.gaussian_reml_fit_positions
    options:
      show_root_heading: true
      heading_level: 3

::: gamfit.reml.gaussian_reml_fit_positions_backward
    options:
      show_root_heading: true
      heading_level: 3

::: gamfit.reml.gaussian_reml_fit_positions_batched
    options:
      show_root_heading: true
      heading_level: 3

::: gamfit.reml.gaussian_reml_fit_positions_batched_backward
    options:
      show_root_heading: true
      heading_level: 3

::: gamfit.reml.gaussian_reml_fit_blocks_forward
    options:
      show_root_heading: true
      heading_level: 3

::: gamfit.reml.gaussian_reml_fit_blocks_backward
    options:
      show_root_heading: true
      heading_level: 3

::: gamfit.reml.gaussian_reml_fit_with_constraints_forward
    options:
      show_root_heading: true
      heading_level: 3

::: gamfit.reml.gaussian_reml_fit_with_constraints_backward
    options:
      show_root_heading: true
      heading_level: 3

::: gamfit.reml.gaussian_reml_fit_formula
    options:
      show_root_heading: true
      heading_level: 3

::: gamfit.reml.gaussian_reml_fit_latent
    options:
      show_root_heading: true
      heading_level: 3

::: gamfit.reml.gaussian_reml_fit_latent_backward
    options:
      show_root_heading: true
      heading_level: 3

::: gamfit.reml.gaussian_reml_optimize_latent
    options:
      show_root_heading: true
      heading_level: 3

::: gamfit.reml.glm_reml_fit_latent
    options:
      show_root_heading: true
      heading_level: 3

::: gamfit.reml.glm_reml_fit_latent_backward
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

::: gamfit.results.TermBlock
    options:
      show_root_heading: true
      heading_level: 3

## Topology and smooth descriptors

::: gamfit.topology.Circle
    options:
      show_root_heading: true
      heading_level: 3

::: gamfit.topology.Cylinder
    options:
      show_root_heading: true
      heading_level: 3

::: gamfit.topology.Torus
    options:
      show_root_heading: true
      heading_level: 3

::: gamfit.topology.Sphere
    options:
      show_root_heading: true
      heading_level: 3

::: gamfit.topology.EuclideanPatch
    options:
      show_root_heading: true
      heading_level: 3

::: gamfit.basis.PeriodicHarmonic
    options:
      show_root_heading: true
      heading_level: 3

## Penalties and latent-coordinate tools

::: gamfit.penalties.ARDPenalty
    options:
      show_root_heading: true
      heading_level: 3

::: gamfit.penalties.OrderedBetaBernoulliPenalty
    options:
      show_root_heading: true
      heading_level: 3

::: gamfit.penalties.OrthogonalityPenalty
    options:
      show_root_heading: true
      heading_level: 3

::: gamfit.penalties.BlockOrthogonalityPenalty
    options:
      show_root_heading: true
      heading_level: 3

::: gamfit.penalties.SparsityPenalty
    options:
      show_root_heading: true
      heading_level: 3

::: gamfit.penalties.SoftmaxAssignmentSparsityPenalty
    options:
      show_root_heading: true
      heading_level: 3

::: gamfit.penalties.TopKActivationPenalty
    options:
      show_root_heading: true
      heading_level: 3

::: gamfit.penalties.SmoothThresholdPenalty
    options:
      show_root_heading: true
      heading_level: 3

::: gamfit.penalties.NuclearNormPenalty
    options:
      show_root_heading: true
      heading_level: 3

::: gamfit.penalties.ScadMcpPenalty
    options:
      show_root_heading: true
      heading_level: 3

::: gamfit.penalties.CompositePenalty
    options:
      show_root_heading: true
      heading_level: 3

::: gamfit.penalties.Penalty
    options:
      show_root_heading: true
      heading_level: 3

::: gamfit.penalties.AnalyticPenaltyKind
    options:
      show_root_heading: true
      heading_level: 3

::: gamfit.penalties.ScalarWeightSchedule
    options:
      show_root_heading: true
      heading_level: 3

::: gamfit.penalties.PENALTY_MANIFEST
    options:
      show_root_heading: true
      heading_level: 3

::: gamfit.penalties.AuxConditionalPriorPenalty
    options:
      show_root_heading: true
      heading_level: 3

::: gamfit.penalties.ParametricAuxConditionalPriorPenalty
    options:
      show_root_heading: true
      heading_level: 3

::: gamfit.penalties.BlockSparsityPenalty
    options:
      show_root_heading: true
      heading_level: 3

::: gamfit.penalties.OrderedBetaBernoulliPenalty
    options:
      show_root_heading: true
      heading_level: 3

::: gamfit.penalties.IsometryPenalty
    options:
      show_root_heading: true
      heading_level: 3

::: gamfit.penalties.IvaeRidgeMeanGauge
    options:
      show_root_heading: true
      heading_level: 3

::: gamfit.penalties.MechanismSparsityPenalty
    options:
      show_root_heading: true
      heading_level: 3

::: gamfit.penalties.TotalVariationPenalty
    options:
      show_root_heading: true
      heading_level: 3

::: gamfit.penalties.SheafConsistencyPenalty
    options:
      show_root_heading: true
      heading_level: 3

::: gamfit.penalties.GatedSAEDecoder
    options:
      show_root_heading: true
      heading_level: 3

## Descriptor protocol

::: gamfit.basis.BasisDescriptor
    options:
      show_root_heading: true
      heading_level: 3

::: gamfit.manifolds.ManifoldDescriptor
    options:
      show_root_heading: true
      heading_level: 3

::: gamfit.penalties.PenaltyDescriptor
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

::: gamfit.errors.GamError
    options:
      show_root_heading: true
      heading_level: 3

::: gamfit.errors.FormulaError
    options:
      show_root_heading: true
      heading_level: 3

::: gamfit.errors.SchemaMismatchError
    options:
      show_root_heading: true
      heading_level: 3

::: gamfit.errors.PredictionError
    options:
      show_root_heading: true
      heading_level: 3

::: gamfit.errors.RustExtensionUnavailableError
    options:
      show_root_heading: true
      heading_level: 3

::: gamfit.errors.AloError
    options:
      show_root_heading: true
      heading_level: 3

::: gamfit.errors.ArrowSchurError
    options:
      show_root_heading: true
      heading_level: 3

::: gamfit.errors.BasisError
    options:
      show_root_heading: true
      heading_level: 3

::: gamfit.errors.CacheStoreError
    options:
      show_root_heading: true
      heading_level: 3

::: gamfit.errors.CalibratorError
    options:
      show_root_heading: true
      heading_level: 3

::: gamfit.errors.ColumnNotFoundError
    options:
      show_root_heading: true
      heading_level: 3

::: gamfit.errors.CorrectedCovarianceError
    options:
      show_root_heading: true
      heading_level: 3

::: gamfit.errors.CubicCellKernelError
    options:
      show_root_heading: true
      heading_level: 3

::: gamfit.errors.CustomFamilyError
    options:
      show_root_heading: true
      heading_level: 3

::: gamfit.errors.DataError
    options:
      show_root_heading: true
      heading_level: 3

::: gamfit.errors.DeviationRuntimeError
    options:
      show_root_heading: true
      heading_level: 3

::: gamfit.errors.EigendecompositionError
    options:
      show_root_heading: true
      heading_level: 3

::: gamfit.errors.FitConvergenceError
    options:
      show_root_heading: true
      heading_level: 3

::: gamfit.errors.InnerModeConvergenceError
    options:
      show_root_heading: true
      heading_level: 3

::: gamfit.errors.FitError
    options:
      show_root_heading: true
      heading_level: 3

::: gamfit.errors.FitInputError
    options:
      show_root_heading: true
      heading_level: 3

::: gamfit.errors.FitInvariantError
    options:
      show_root_heading: true
      heading_level: 3

::: gamfit.errors.FitNumericalError
    options:
      show_root_heading: true
      heading_level: 3

::: gamfit.errors.FitSeedError
    options:
      show_root_heading: true
      heading_level: 3

::: gamfit.errors.FittedModelError
    options:
      show_root_heading: true
      heading_level: 3

::: gamfit.errors.GamlssError
    options:
      show_root_heading: true
      heading_level: 3

::: gamfit.errors.GeometryError
    options:
      show_root_heading: true
      heading_level: 3

::: gamfit.errors.GpuError
    options:
      show_root_heading: true
      heading_level: 3

::: gamfit.errors.GradientUnavailableError
    options:
      show_root_heading: true
      heading_level: 3

::: gamfit.errors.HessianNotPositiveDefiniteError
    options:
      show_root_heading: true
      heading_level: 3

::: gamfit.errors.HmcError
    options:
      show_root_heading: true
      heading_level: 3

::: gamfit.errors.IdentifiabilityCompilerError
    options:
      show_root_heading: true
      heading_level: 3

::: gamfit.errors.IllConditionedError
    options:
      show_root_heading: true
      heading_level: 3

::: gamfit.errors.IntegrationError
    options:
      show_root_heading: true
      heading_level: 3

::: gamfit.errors.InvalidConfigurationError
    options:
      show_root_heading: true
      heading_level: 3

::: gamfit.errors.InvalidInputError
    options:
      show_root_heading: true
      heading_level: 3

::: gamfit.errors.InvalidSpecificationError
    options:
      show_root_heading: true
      heading_level: 3

::: gamfit.errors.JointPenaltyError
    options:
      show_root_heading: true
      heading_level: 3

::: gamfit.errors.LatentSurvivalError
    options:
      show_root_heading: true
      heading_level: 3

::: gamfit.errors.LayoutError
    options:
      show_root_heading: true
      heading_level: 3

::: gamfit.errors.LinearAlgebraError
    options:
      show_root_heading: true
      heading_level: 3

::: gamfit.errors.LinearSystemSolveError
    options:
      show_root_heading: true
      heading_level: 3

::: gamfit.errors.LognormalKernelError
    options:
      show_root_heading: true
      heading_level: 3

::: gamfit.errors.MapUniquenessError
    options:
      show_root_heading: true
      heading_level: 3

::: gamfit.errors.MatrixError
    options:
      show_root_heading: true
      heading_level: 3

::: gamfit.errors.MatrixMaterializationError
    options:
      show_root_heading: true
      heading_level: 3

::: gamfit.errors.MissingDependencyError
    options:
      show_root_heading: true
      heading_level: 3

::: gamfit.errors.ModelOverparameterizedError
    options:
      show_root_heading: true
      heading_level: 3

::: gamfit.errors.MonotoneRootError
    options:
      show_root_heading: true
      heading_level: 3

::: gamfit.errors.OuterStrategyError
    options:
      show_root_heading: true
      heading_level: 3

::: gamfit.errors.ParameterConstraintError
    options:
      show_root_heading: true
      heading_level: 3

::: gamfit.errors.PenaltySpectrumError
    options:
      show_root_heading: true
      heading_level: 3

::: gamfit.errors.PerfectSeparationError
    options:
      show_root_heading: true
      heading_level: 3

::: gamfit.errors.PirlsConvergenceError
    options:
      show_root_heading: true
      heading_level: 3

::: gamfit.errors.PredictInputError
    options:
      show_root_heading: true
      heading_level: 3

::: gamfit.errors.RemlConvergenceError
    options:
      show_root_heading: true
      heading_level: 3

::: gamfit.errors.ScaleDesignError
    options:
      show_root_heading: true
      heading_level: 3

::: gamfit.errors.SmoothError
    options:
      show_root_heading: true
      heading_level: 3

::: gamfit.errors.SurvivalConstructionError
    options:
      show_root_heading: true
      heading_level: 3

::: gamfit.errors.SurvivalError
    options:
      show_root_heading: true
      heading_level: 3

::: gamfit.errors.SurvivalLocationScaleError
    options:
      show_root_heading: true
      heading_level: 3

::: gamfit.errors.SurvivalMarginalSlopeError
    options:
      show_root_heading: true
      heading_level: 3

::: gamfit.errors.SurvivalPredictError
    options:
      show_root_heading: true
      heading_level: 3

::: gamfit.errors.TermBuilderError
    options:
      show_root_heading: true
      heading_level: 3

::: gamfit.errors.TransformationNormalError
    options:
      show_root_heading: true
      heading_level: 3

::: gamfit.errors.UnsupportedLinkError
    options:
      show_root_heading: true
      heading_level: 3

## Manifold SAE

See the [Manifold SAE dictionary guide](manifold-sae.md) for the narrative.

::: gamfit.sae.sae_manifold_fit
    options:
      show_root_heading: true
      heading_level: 3

::: gamfit.sae.atom_trust_scores
    options:
      show_root_heading: true
      heading_level: 3

::: gamfit.sae.sae_trust_diagnostics
    options:
      show_root_heading: true
      heading_level: 3

::: gamfit.sae.ManifoldSAE
    options:
      show_root_heading: true
      heading_level: 3
      members_order: source
      filters:
        - "!^_"
        - "^__init__$"
        - "^__repr__$"

::: gamfit.sae.GumbelTemperatureSchedule
    options:
      show_root_heading: true
      heading_level: 3

::: gamfit.sae.gumbel_geometric_schedule
    options:
      show_root_heading: true
      heading_level: 3

::: gamfit.sae.gumbel_linear_schedule
    options:
      show_root_heading: true
      heading_level: 3

::: gamfit.sae.gumbel_reciprocal_iter_schedule
    options:
      show_root_heading: true
      heading_level: 3

## Equivariant smooths

::: gamfit.sae.GaugeCompanion
    options:
      show_root_heading: true
      heading_level: 3

::: gamfit.sae.gauge_companion
    options:
      show_root_heading: true
      heading_level: 3

::: gamfit.sae.rho_so2
    options:
      show_root_heading: true
      heading_level: 3

::: gamfit.sae.rho_so2_jvp
    options:
      show_root_heading: true
      heading_level: 3

::: gamfit.sae.rho_so3
    options:
      show_root_heading: true
      heading_level: 3

::: gamfit.sae.rho_so3_jvp
    options:
      show_root_heading: true
      heading_level: 3

## Manifolds

::: gamfit.geometry.CircleManifold
    options:
      show_root_heading: true
      heading_level: 3

::: gamfit.geometry.EuclideanManifold
    options:
      show_root_heading: true
      heading_level: 3

::: gamfit.geometry.GrassmannManifold
    options:
      show_root_heading: true
      heading_level: 3

::: gamfit.geometry.ProductManifold
    options:
      show_root_heading: true
      heading_level: 3

::: gamfit.geometry.SpdManifold
    options:
      show_root_heading: true
      heading_level: 3

::: gamfit.geometry.SphereManifold
    options:
      show_root_heading: true
      heading_level: 3

::: gamfit.geometry.StiefelManifold
    options:
      show_root_heading: true
      heading_level: 3

::: gamfit.geometry.TorusManifold
    options:
      show_root_heading: true
      heading_level: 3

## Topology selection

::: gamfit.topology.BasisSpec
    options:
      show_root_heading: true
      heading_level: 3

::: gamfit.topology.ScoreKind
    options:
      show_root_heading: true
      heading_level: 3

::: gamfit.topology.ScoreScale
    options:
      show_root_heading: true
      heading_level: 3

::: gamfit.topology.SelectTopologyResult
    options:
      show_root_heading: true
      heading_level: 3

::: gamfit.topology.TopologyAutoSelector
    options:
      show_root_heading: true
      heading_level: 3

::: gamfit.topology.TopologyAutoSelectorRank
    options:
      show_root_heading: true
      heading_level: 3

::: gamfit.topology.TopologyAutoSelectorResult
    options:
      show_root_heading: true
      heading_level: 3

::: gamfit.topology.TopologyStack
    options:
      show_root_heading: true
      heading_level: 3

::: gamfit.topology.select_topology
    options:
      show_root_heading: true
      heading_level: 3

::: gamfit.topology.stack_topologies
    options:
      show_root_heading: true
      heading_level: 3

## Identifiability

::: gamfit.identifiability.IdentifiabilityReport
    options:
      show_root_heading: true
      heading_level: 3

::: gamfit.identifiability.IdentifiabilityTheoremResult
    options:
      show_root_heading: true
      heading_level: 3

::: gamfit.identifiability.IdentifiableFactorFitResult
    options:
      show_root_heading: true
      heading_level: 3

::: gamfit.identifiability.check
    options:
      show_root_heading: true
      heading_level: 3

::: gamfit.identifiability.identifiable_factor_fit
    options:
      show_root_heading: true
      heading_level: 3

## Structure discovery

::: gamfit.sae.atom_birth_gate
    options:
      show_root_heading: true
      heading_level: 3

::: gamfit.sae.e_bh_dictionary_certificate
    options:
      show_root_heading: true
      heading_level: 3

::: gamfit.sae.expected_resolution_budget
    options:
      show_root_heading: true
      heading_level: 3

::: gamfit.sae.log_e_from_p_value
    options:
      show_root_heading: true
      heading_level: 3

::: gamfit.sae.plan_probe_for_contested_claim
    options:
      show_root_heading: true
      heading_level: 3

::: gamfit.sae.select_probe_by_expected_evidence
    options:
      show_root_heading: true
      heading_level: 3

::: gamfit.sae.split_likelihood_log_e
    options:
      show_root_heading: true
      heading_level: 3

## Partial supervision

::: gamfit.examples.PartialSupervisionExample
    options:
      show_root_heading: true
      heading_level: 3

::: gamfit.examples.PartialSupervisionFit
    options:
      show_root_heading: true
      heading_level: 3

::: gamfit.examples.SaeSupervisedFit
    options:
      show_root_heading: true
      heading_level: 3

::: gamfit.examples.partial_supervision
    options:
      show_root_heading: true
      heading_level: 3

::: gamfit.examples.sae_supervised
    options:
      show_root_heading: true
      heading_level: 3

## Plotting

`gamfit.plot` needs the optional matplotlib extra (`pip install 'gamfit[plot]'`);
the module imports without it and each function raises `ImportError` naming
the extra when matplotlib is missing.

::: gamfit.plot.model
    options:
      show_root_heading: true
      heading_level: 3

::: gamfit.plot.trace
    options:
      show_root_heading: true
      heading_level: 3

::: gamfit.plot.sae_atom
    options:
      show_root_heading: true
      heading_level: 3

::: gamfit.plot.sae_fit
    options:
      show_root_heading: true
      heading_level: 3

## Staged coordinates

::: gamfit.CtnStage1
    options:
      show_root_heading: true
      heading_level: 3

## Compositional smooth specs

::: gamfit.basis.Smooth
    options:
      show_root_heading: true
      heading_level: 3

::: gamfit.basis.SmoothSum
    options:
      show_root_heading: true
      heading_level: 3

::: gamfit.smooth.Smooth
    options:
      show_root_heading: true
      heading_level: 3

## Torch-optional primitives

These live in `gamfit.torch`, behind the optional `torch` dependency;
accessing `gamfit.torch` without torch installed raises `AttributeError`.
See [torch.md](torch.md).

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

::: gamfit.response_geometry.clr
    options:
      show_root_heading: true
      heading_level: 3

::: gamfit.response_geometry.alr
    options:
      show_root_heading: true
      heading_level: 3

::: gamfit.response_geometry.closure
    options:
      show_root_heading: true
      heading_level: 3

::: gamfit.response_geometry.simplex_frechet_mean
    options:
      show_root_heading: true
      heading_level: 3

::: gamfit.response_geometry.sphere_frechet_mean
    options:
      show_root_heading: true
      heading_level: 3
