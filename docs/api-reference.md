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

::: gamfit.periodic_spline_curve_basis
    options:
      show_root_heading: true
      heading_level: 3

::: gamfit.sphere_basis
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

::: gamfit.sae_benchmark
    options:
      show_root_heading: true
      heading_level: 3

::: gamfit.sweep_sae_benchmark
    options:
      show_root_heading: true
      heading_level: 3

::: gamfit.format_sae_benchmark_markdown
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
