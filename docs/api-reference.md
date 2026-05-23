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

::: gamfit.explain_error
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
