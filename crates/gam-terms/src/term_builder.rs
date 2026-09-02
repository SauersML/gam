//! Term construction: bridge from parsed formula terms to `TermCollectionSpec`.
//!
//! This module takes the AST produced by `inference::formula_dsl` and a loaded
//! dataset, resolves column references, infers knot counts and center strategies,
//! and produces a `TermCollectionSpec` ready for `build_term_collection_design`.

use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::path::PathBuf;

use ndarray::{Array2, ArrayView1};

use crate::basis::{
    BSplineBasisSpec, BSplineBoundaryConditions, BSplineEndpointBoundaryCondition,
    BSplineIdentifiability, BSplineKnotSpec, CenterCountRequest, CenterStrategy,
    ConstantCurvatureBasisSpec, ConstantCurvatureIdentifiability, DuchonBasisSpec,
    DuchonNullspaceOrder, DuchonOperatorPenaltySpec, DuchonSpectralBasis, MaternBasisSpec,
    MaternIdentifiability, MaternLengthScale, MaternNu, MeasureJetBasisSpec,
    MeasureJetIdentifiability, OneDimensionalBoundary, SpatialIdentifiability, SphereMethod,
    SphereWahbaKernel, SphericalSplineBasisSpec, SphericalSplineIdentifiability,
    ThinPlateBasisSpec, auto_spatial_center_strategy, count_unique_coordinate_rows,
    default_num_centers, default_spatial_center_strategy, default_spherical_harmonic_degree,
    plan_spatial_basis, select_r_uniform_subsample_centers, thin_plate_penalty_order,
};
use crate::inference::formula_dsl::{
    ParsedTerm, SmoothKind, option_bool, option_f64, option_f64_strict, option_usize,
    option_usize_any, option_usize_any_strict, option_usize_strict, parsed_term_column_names,
    strip_quotes,
};
use crate::smooth::{
    BySmoothKind, ByVarKind, ByVariableSpec, FactorSmoothFlavour, FactorSmoothSpec,
    LinearCoefficientGeometry, LinearTermSpec, RandomEffectTermSpec, ShapeConstraint,
    SmoothBasisSpec, SmoothTermSpec, TensorBSplineIdentifiability,
    TensorBSplinePenaltyDecomposition, TensorBSplineSpec, TermCollectionSpec,
};
use gam_data::{ColumnKindTag, DataError, EncodedDataset as Dataset};
use gam_problem::types::ColIdx;
use gam_runtime::resource::ResourcePolicy;

/// Default B-spline degree when a smooth's `degree=` option is absent. Cubic
/// (degree 3) is the standard GAM convention: C² continuity with a low knot
/// count.
const DEFAULT_BSPLINE_DEGREE: usize = 3;

/// Default difference-penalty order when a smooth's `penalty_order=` (alias
/// `m=`) option is absent. Second-order (curvature) is the standard P-spline
/// convention.
const DEFAULT_PENALTY_ORDER: usize = 2;

/// Admissible `lmax=` for the truncated Wahba sphere kernels, matching the
/// documented range on [`SphereWahbaKernel::SobolevTruncated`]. The lower end
/// keeps at least a few degrees of resolution; the upper end is the bound the
/// device kernel bakes in as a compile-time `#define`.
const SPHERE_TRUNCATION_LMAX_RANGE: std::ops::RangeInclusive<usize> = 5..=200;

/// Default basis dimension for one-dimensional cyclic cubic P-splines.
///
/// Periodic smooths spend no coefficients on free endpoints, so they should not
/// inherit the larger open B-spline knot ceiling by default.  This is still only
/// a default: callers can request a richer periodic space with `k=`.
const CYCLIC_DEFAULT_BASIS_DIM: usize = 12;

/// Default shared-marginal basis dimension for `bs="fs"`/`bs="sz"` factor smooths,
/// matching mgcv's factor-smooth default `k=10`. A factor smooth shares one
/// marginal across all levels; a modest basis recovers the shared signal without
/// over-fitting each group's within-group noise (gam#903). Overridden by an
/// explicit `k`/`basis_dim`.
const FACTOR_SMOOTH_DEFAULT_BASIS_DIM: usize = 10;

/// Default row-chunk size for the out-of-core PCA-basis smooth when the
/// `chunk_size=` option is absent. Streams the design in row blocks to bound
/// peak memory independent of the dataset row count.
const DEFAULT_PCA_CHUNK_SIZE: usize = 4096;

// ---------------------------------------------------------------------------
// Typed errors
// ---------------------------------------------------------------------------

/// Typed errors emitted by term-builder helpers. `Display` reproduces the exact
/// pre-refactor `format!(...)` text byte-for-byte, so callers that string-match
/// on the message (tests, log assertions) keep working unchanged. Public-API
/// functions still return `Result<_, String>` and use `.to_string()` shims at
/// their boundary to stay compatible with callers in protected modules.
#[derive(Clone, Debug)]
pub enum TermBuilderError {
    /// Column-resolution / column-kind lookup failures whose context is purely
    /// internal (column-kind table out-of-sync, alias map missing an entry,
    /// etc.). User-facing "this formula references a column that doesn't
    /// exist" diagnostics use the dedicated `ColumnNotFound` variant so the
    /// FFI boundary can lift the structured payload into a Python
    /// `ColumnNotFoundError` without parsing prose.
    MissingColumn { reason: String },
    /// A formula referenced a column that is not present in the input data.
    /// Mirrors `DataError::ColumnNotFound` field-for-field so the conversion
    /// across module boundaries is a pure data move (no re-derivation, no
    /// string re-parsing). Public callers see byte-identical `Display`
    /// output to the legacy `missing_column_message` text.
    ColumnNotFound {
        name: String,
        role: Option<String>,
        available: Vec<String>,
        similar: Vec<String>,
        tsv_hint: bool,
    },
    /// User-specified configuration is internally inconsistent (e.g. too few
    /// variables for a smooth type, conflicting size options, requested basis
    /// dimension below the polynomial nullspace).
    IncompatibleConfig { reason: String },
    /// Option parsing failure: malformed numeric expression, unknown option
    /// key, out-of-range integer, list-length mismatch, etc.
    InvalidOption { reason: String },
    /// User requested a feature that is intentionally not supported (unknown
    /// smooth type / method / kernel / identifiability, non-zero anchor,
    /// internal-only token, etc.).
    UnsupportedFeature { reason: String },
    /// Input data is degenerate for the requested term (constant column,
    /// non-finite categorical entries, ...).
    DegenerateData { reason: String },
    /// Term-collection-stage formula error — a node that the caller was
    /// supposed to resolve upstream reached the builder.
    MalformedFormula { reason: String },
}

impl std::fmt::Display for TermBuilderError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TermBuilderError::MissingColumn { reason }
            | TermBuilderError::IncompatibleConfig { reason }
            | TermBuilderError::InvalidOption { reason }
            | TermBuilderError::UnsupportedFeature { reason }
            | TermBuilderError::DegenerateData { reason }
            | TermBuilderError::MalformedFormula { reason } => f.write_str(reason),
            // Delegate to the canonical `DataError::ColumnNotFound` formatter
            // so a single source of truth defines the human text. The
            // intermediate `DataError` constructed here owns its strings only
            // for the duration of the Display call — no allocation cost
            // beyond the original payload that this variant already holds.
            TermBuilderError::ColumnNotFound {
                name,
                role,
                available,
                similar,
                tsv_hint,
            } => {
                let canonical = DataError::ColumnNotFound {
                    name: name.clone(),
                    role: role.clone(),
                    available: available.clone(),
                    similar: similar.clone(),
                    tsv_hint: *tsv_hint,
                };
                std::fmt::Display::fmt(&canonical, f)
            }
        }
    }
}

impl From<TermBuilderError> for String {
    fn from(err: TermBuilderError) -> String {
        err.to_string()
    }
}

/// Catchall lift for the term-builder's internal `Result<_, String>` helpers
/// (numeric expression parsing, option lookup, boundary-condition parsing,
/// ...) that flow into `build_termspec` via `?`. Maps to
/// `IncompatibleConfig`, which is the most appropriate generic bucket for
/// option/config-style failures — leaf sites that emit structured payloads
/// (`From<DataError>` for column-not-found) bypass this fallback.
impl From<String> for TermBuilderError {
    fn from(reason: String) -> Self {
        Self::IncompatibleConfig { reason }
    }
}

/// Typed lift from data-layer errors. `DataError::ColumnNotFound` becomes
/// `TermBuilderError::ColumnNotFound` field-for-field — no stringification,
/// no information loss — so the FFI boundary downstream can dispatch on
/// the typed variant. Other `DataError` variants degrade into
/// `MissingColumn` since they describe column-resolution-time failures
/// without a dedicated structured destination.
impl From<DataError> for TermBuilderError {
    fn from(err: DataError) -> Self {
        match err {
            DataError::ColumnNotFound {
                name,
                role,
                available,
                similar,
                tsv_hint,
            } => Self::ColumnNotFound {
                name,
                role,
                available,
                similar,
                tsv_hint,
            },
            DataError::SchemaMismatch { reason }
            | DataError::ParseError { reason }
            | DataError::EncodingFailure { reason }
            | DataError::EmptyInput { reason }
            | DataError::InvalidValue { reason } => Self::MissingColumn { reason },
        }
    }
}

// Constructor helpers — keep error-site code compact and consistent.
impl TermBuilderError {
    #[inline]
    fn missing_column(reason: impl Into<String>) -> Self {
        TermBuilderError::MissingColumn {
            reason: reason.into(),
        }
    }
    #[inline]
    fn incompatible_config(reason: impl Into<String>) -> Self {
        TermBuilderError::IncompatibleConfig {
            reason: reason.into(),
        }
    }
    #[inline]
    fn invalid_option(reason: impl Into<String>) -> Self {
        TermBuilderError::InvalidOption {
            reason: reason.into(),
        }
    }
    #[inline]
    fn unsupported_feature(reason: impl Into<String>) -> Self {
        TermBuilderError::UnsupportedFeature {
            reason: reason.into(),
        }
    }
    #[inline]
    fn degenerate_data(reason: impl Into<String>) -> Self {
        TermBuilderError::DegenerateData {
            reason: reason.into(),
        }
    }
    #[inline]
    fn malformed_formula(reason: impl Into<String>) -> Self {
        TermBuilderError::MalformedFormula {
            reason: reason.into(),
        }
    }
}

// ---------------------------------------------------------------------------
// Column resolution
// ---------------------------------------------------------------------------

/// Resolve a bare column name to its index, returning a typed
/// `DataError::ColumnNotFound` on miss so the FFI boundary can surface a
/// structured `gamfit.ColumnNotFoundError(column=…, available=…)` rather
/// than rely on string-classification of human prose. Internal callers that
/// still flow `Result<_, String>` get byte-identical text via
/// `From<DataError> for String`.
pub fn resolve_col(col_map: &HashMap<String, usize>, name: &str) -> Result<usize, DataError> {
    col_map
        .get(name)
        .copied()
        .ok_or_else(|| DataError::column_not_found(col_map, name, None))
}

/// Like `resolve_col` but tags the missing-column payload with a role label
/// (`"response"`, `"entry"`, `"exit"`, `"event"`, `"z"`, `"id"`, …) so the
/// boundary-side Python exception can disambiguate which formula slot held
/// the bad reference.
pub fn resolve_role_col(
    col_map: &HashMap<String, usize>,
    name: &str,
    role: &str,
) -> Result<usize, DataError> {
    col_map
        .get(name)
        .copied()
        .ok_or_else(|| DataError::column_not_found(col_map, name, Some(role)))
}

fn encoded_levels_for_column(ds: &Dataset, col: ColIdx) -> Vec<(u64, String)> {
    let mut seen = BTreeSet::<u64>::new();
    for value in ds.values.column(col.get()) {
        if value.is_finite() {
            seen.insert(gam_data::canonical_level_bits(*value));
        }
    }
    let schema_levels = ds
        .schema
        .columns
        .get(col.get())
        .map(|column| column.levels.as_slice())
        .unwrap_or(&[]);
    seen.into_iter()
        .enumerate()
        .map(|(idx, bits)| {
            let fallback = format!("level{}", idx + 1);
            let label = schema_levels.get(idx).cloned().unwrap_or(fallback);
            (bits, label)
        })
        .collect()
}

/// Internal option key carrying the row count that n-scaling BASIS DEFAULTS
/// (radial center counts, spatial plans) must size from. A factor-by smooth
/// expands into per-level blocks that each see ONLY their level's rows, so
/// sizing the default from the pooled row count over-provisions every level —
/// measured on the #1561 by-group location-scale fixture: `s(x, bs='tp',
/// by=group)` at n=200 (100/group) got ~50 centers PER LEVEL, an
/// ill-conditioned 100-column mean block whose truth-recovery floor (0.111)
/// no λ could beat, while the same smooth sized for the level's own 100 rows
/// recovers to ~0.036. Explicit user `centers=`/`k=` bypass the default and
/// are unaffected. Stripped at the top of [`build_smooth_basis`] like
/// `__by_col`, so per-kind option allow-lists never see it.
const DEFAULT_SIZING_ROWS_OPTION: &str = "__default_sizing_rows";

/// The smallest per-level row count of a categorical by-column: the effective
/// sample size each by-level smooth block actually fits. `None` when the
/// column has no finite rows (callers fall back to the pooled count).
fn min_categorical_by_level_rows(ds: &Dataset, by_col: usize) -> Option<usize> {
    let mut counts: BTreeMap<u64, usize> = BTreeMap::new();
    for value in ds.values.column(by_col) {
        if value.is_finite() {
            *counts
                .entry(gam_data::canonical_level_bits(*value))
                .or_insert(0) += 1;
        }
    }
    counts.values().copied().min()
}

/// Insert [`DEFAULT_SIZING_ROWS_OPTION`] into `inner_options` when the by
/// column is categorical (numeric-by smooths keep one shared block over all
/// rows, so pooled sizing stays correct there).
fn inject_by_level_sizing_rows(
    inner_options: &mut BTreeMap<String, String>,
    ds: &Dataset,
    by_col: usize,
) {
    if matches!(
        ds.column_kinds.get(by_col).copied(),
        Some(ColumnKindTag::Categorical)
    ) && let Some(min_rows) = min_categorical_by_level_rows(ds, by_col)
    {
        inner_options.insert(DEFAULT_SIZING_ROWS_OPTION.to_string(), min_rows.to_string());
    }
}

pub fn column_map_with_alias(
    col_map: &HashMap<String, usize>,
    alias: &str,
    target_column: &str,
) -> HashMap<String, usize> {
    let mut aliased = col_map.clone();
    if let Some(idx) = col_map.get(target_column).copied() {
        aliased.entry(alias.to_string()).or_insert(idx);
    }
    aliased
}

/// The canonical marginal-slope alias: `z` in a formula binds to the column
/// named by `z_column`.
pub const MARGINAL_SLOPE_Z_ALIAS: &str = "z";

/// Whether writing `z` in a formula would resolve to `z_column` for this frame.
///
/// `column_map_with_alias` inserts with `or_insert`, so a frame that carries its
/// own real `z` column keeps it and the alias is inert — writing `z` there means
/// that column, which is legitimate. The alias is live only when `z_column`
/// exists and `z` does not, and only then does `z` silently denote the score.
pub fn marginal_slope_z_alias_is_live(col_map: &HashMap<String, usize>, z_column: &str) -> bool {
    col_map.contains_key(z_column) && !col_map.contains_key(MARGINAL_SLOPE_Z_ALIAS)
}

// ---------------------------------------------------------------------------
// ParsedTerm[] + Dataset → TermCollectionSpec
// ---------------------------------------------------------------------------

pub fn build_termspec(
    terms: &[ParsedTerm],
    ds: &Dataset,
    col_map: &HashMap<String, usize>,
    inference_notes: &mut Vec<String>,
    policy: &ResourcePolicy,
) -> Result<TermCollectionSpec, TermBuilderError> {
    // Generic ingestion deliberately preserves missing cells because it runs
    // before a formula exists. This is the first layer that knows the complete
    // set of columns consumed by term construction (including `by=` and nested
    // slope surfaces), so completeness is enforced here and nowhere wider.
    let mut consumed_columns = BTreeSet::new();
    parsed_term_column_names(terms, &mut consumed_columns);
    for name in consumed_columns {
        let column = resolve_col(col_map, &name)?;
        if let Some(row) = ds
            .values
            .column(column)
            .iter()
            .position(|value| !value.is_finite())
        {
            return Err(TermBuilderError::degenerate_data(format!(
                "model term column '{name}' contains a non-finite value at row {}",
                row + 1
            )));
        }
    }

    let mut linear_terms = Vec::<LinearTermSpec>::new();
    let mut random_terms = Vec::<RandomEffectTermSpec>::new();
    let mut smooth_terms = Vec::<SmoothTermSpec>::new();
    let smooth_coordinate_count = terms
        .iter()
        .map(|term| match term {
            ParsedTerm::Smooth { vars, .. } => vars.len(),
            _ => 0,
        })
        .sum::<usize>();

    for t in terms {
        match t {
            ParsedTerm::Linear {
                name,
                explicit,
                double_penalty,
                coefficient_min,
                coefficient_max,
            } => {
                let col = resolve_col(col_map, name)?;
                let auto_kind = ds.column_kinds.get(col).copied().ok_or_else(|| {
                    TermBuilderError::missing_column(format!(
                        "internal column-kind lookup failed for '{name}'"
                    ))
                    .to_string()
                })?;
                if *explicit {
                    linear_terms.push(LinearTermSpec {
                        name: name.clone(),
                        feature_col: col,
                        feature_cols: vec![col],
                        categorical_levels: vec![],
                        // Parametric terms are unpenalized/MLE by default.
                        // `double_penalty=true` is an explicit shrinkage choice
                        // carried by the parsed term.
                        double_penalty: *double_penalty,
                        coefficient_geometry: LinearCoefficientGeometry::Unconstrained,
                        coefficient_min: *coefficient_min,
                        coefficient_max: *coefficient_max,
                        frozen_function_mass: None,
                    });
                } else {
                    match auto_kind {
                        ColumnKindTag::Continuous | ColumnKindTag::Binary => {
                            linear_terms.push(LinearTermSpec {
                                name: name.clone(),
                                feature_col: col,
                                feature_cols: vec![col],
                                categorical_levels: vec![],
                                // Preserve the parser's explicit opt-in. Bare
                                // numeric terms arrive as `false`.
                                double_penalty: *double_penalty,
                                coefficient_geometry: LinearCoefficientGeometry::Unconstrained,
                                coefficient_min: *coefficient_min,
                                coefficient_max: *coefficient_max,
                                frozen_function_mass: None,
                            });
                        }
                        ColumnKindTag::Categorical => {
                            if coefficient_min.is_some() || coefficient_max.is_some() {
                                return Err(TermBuilderError::incompatible_config(format!(
                                    "coefficient constraints are not supported for categorical auto-random-effect term '{name}'; use group({name}) or an unconstrained numeric term"
                                )));
                            }
                            random_terms.push(RandomEffectTermSpec {
                                name: name.clone(),
                                feature_col: col,
                                drop_first_level: false,
                                penalized: true,
                                frozen_levels: None,
                                // A BARE categorical main effect (`+ g`) is a FIXED
                                // parametric factor. Although it is auto-promoted to
                                // a penalized random block above, an *unseen* level
                                // at predict must raise a schema mismatch rather than
                                // be mapped to the factor's centering point (#2102).
                                lenient_unseen: false,
                            });
                        }
                    }
                }
            }
            ParsedTerm::BoundedLinear {
                name,
                min,
                max,
                prior,
                double_penalty,
            } => {
                let col = resolve_col(col_map, name)?;
                let auto_kind = ds.column_kinds.get(col).copied().ok_or_else(|| {
                    TermBuilderError::missing_column(format!(
                        "internal column-kind lookup failed for '{name}'"
                    ))
                    .to_string()
                })?;
                if !matches!(auto_kind, ColumnKindTag::Continuous | ColumnKindTag::Binary) {
                    return Err(TermBuilderError::incompatible_config(format!(
                        "bounded() currently supports only numeric columns, got categorical '{name}'"
                    )));
                }
                linear_terms.push(LinearTermSpec {
                    name: name.clone(),
                    feature_col: col,
                    feature_cols: vec![col],
                    categorical_levels: vec![],
                    double_penalty: *double_penalty,
                    coefficient_geometry: LinearCoefficientGeometry::Bounded {
                        min: *min,
                        max: *max,
                        prior: prior.clone(),
                    },
                    coefficient_min: None,
                    coefficient_max: None,
                    frozen_function_mass: None,
                });
            }
            ParsedTerm::RandomEffect {
                name,
                lenient_unseen,
            } => {
                let col = resolve_col(col_map, name)?;
                random_terms.push(RandomEffectTermSpec {
                    name: name.clone(),
                    feature_col: col,
                    drop_first_level: false,
                    penalized: true,
                    frozen_levels: None,
                    // Unseen-level policy is fixed by the wrapper the user wrote
                    // (`formula_dsl`): a genuine random effect
                    // (`group(g)`/`re(g)`/`s(g, bs="re")`) shrinks a held-out
                    // group to the population mean and so tolerates unseen
                    // levels; a fixed `factor(g)`, like a bare `+ g` categorical
                    // main effect, must reject an unseen level rather than
                    // collapse onto the centering point (#2137/#2102).
                    lenient_unseen: *lenient_unseen,
                });
            }
            ParsedTerm::Smooth {
                label,
                vars,
                kind,
                options,
            } => {
                let smooth_vars = vars.clone();
                let by_name = options.get("by").cloned();
                // `bs="sz"` (sum-to-zero), like `bs="fs"`/`bs="re"`, is a
                // factor-smooth family handled natively by `build_smooth_basis`'s
                // fs/sz/re path: it detects the categorical factor among the
                // variables and emits a `SmoothBasisSpec::FactorSmooth { Sz }`
                // with the correct single-penalty marginal and modest default
                // basis. Route sz straight through `build_smooth_basis` rather
                // than intercepting it into a legacy `FactorSumToZero` envelope
                // here (which left `sz(fac, x)` mis-typed as `FactorSumToZero`
                // instead of the expected `FactorSmooth { Sz }`).
                let cols = smooth_vars
                    .iter()
                    .map(|v| resolve_col(col_map, v))
                    .collect::<Result<Vec<_>, _>>()?;
                let mut inner_options = options.clone();
                inner_options.remove("by");
                // `ordered=` is consumed here (ByVarKind::Factor routing) and
                // must not propagate to the inner basis builder, which has no
                // allow-list entry for it and would reject it as an unknown option.
                inner_options.remove("ordered");
                // Pop the shape constraint before `build_smooth_basis` runs so
                // it never reaches the per-kind `validate_known_options`
                // allow-lists (the constraint is a property of the smooth term,
                // not of any one basis kind). Basis-incompatible requests still
                // fail loudly downstream via `shape_supports_basis`.
                let shape = match inner_options.remove("shape") {
                    None => ShapeConstraint::None,
                    Some(raw) => crate::smooth::parse_shape_constraint(&raw)
                        .map_err(TermBuilderError::invalid_option)?,
                };
                // A categorical by= expands into per-level blocks below; size
                // the inner basis's n-scaling defaults from the smallest
                // level's rows, not the pooled count (see
                // `DEFAULT_SIZING_ROWS_OPTION`).
                if let Some(by_name) = by_name.as_deref() {
                    let by_col = resolve_col(col_map, by_name)?;
                    inject_by_level_sizing_rows(&mut inner_options, ds, by_col);
                }
                let inner_basis = build_smooth_basis(
                    *kind,
                    &smooth_vars,
                    &cols,
                    &inner_options,
                    ds,
                    inference_notes,
                    policy,
                    smooth_coordinate_count,
                )?;
                // `bs="sz"` deliberately stays typed as `SmoothBasisSpec::FactorSmooth
                // { Sz }` (#1403, owner-confirmed in #1887): the `FactorSumToZero`
                // envelope is the *legacy, mis-typed* representation. `build_factor_smooth`
                // reuses the sum-to-zero construction internally as its single source of
                // truth for the zero-sum geometry (term_specs.rs) while keeping the
                // freeze-consistent `FactorSmooth` metadata shape shared by fs/sz/re, so
                // there is no reason to re-wrap the spec into the legacy envelope here —
                // doing so (#1981) mis-typed `sz(fac, x)` back to `FactorSumToZero` and
                // broke the refit/predict freeze path's `(FactorSmooth, …)` metadata match.
                if let Some(by_name) = by_name {
                    let by_col = resolve_col(col_map, &by_name)?;
                    match ds.column_kinds.get(by_col).copied().ok_or_else(|| {
                        format!("internal column-kind lookup failed for by variable '{by_name}'")
                    })? {
                        ColumnKindTag::Categorical => {
                            let levels = encoded_levels_for_column(ds, ColIdx::new(by_col));
                            // A penalized random block for this factor already
                            // owns its full level offsets when EITHER an explicit
                            // `group(factor)` appears, OR a *bare* categorical
                            // `+ factor` does — the latter is auto-promoted to a
                            // penalized random-effect block (see the
                            // `ParsedTerm::Linear` / `ColumnKindTag::Categorical`
                            // arm above, `penalized: true`). Both representations
                            // carry the same per-level offsets, so #1457: the
                            // `by=` branch must NOT additionally add its own
                            // unpenalized treatment-coded main effect, which would
                            // double-represent the factor (two `g` design blocks +
                            // a spurious extra smoothing parameter).
                            let penalized_group_owner_present =
                                terms.iter().any(|other| match other {
                                    ParsedTerm::RandomEffect { name, .. } => name == &by_name,
                                    ParsedTerm::Linear {
                                        name,
                                        explicit: false,
                                        ..
                                    } if name == &by_name => col_map
                                        .get(name)
                                        .and_then(|c| ds.column_kinds.get(*c).copied())
                                        .map(|kind| matches!(kind, ColumnKindTag::Categorical))
                                        .unwrap_or(false),
                                    _ => false,
                                });
                            // Add an unpenalized treatment-coded fixed main
                            // effect for a standalone factor-by smooth, unless
                            // the same factor already has an explicit
                            // `group(factor)` term OR a bare categorical `+
                            // factor` that was auto-promoted to a penalized
                            // random block (#1457).  In those mixed-model forms
                            // the penalized random intercept is the coherent
                            // owner of level offsets; adding a no-pooling fixed
                            // factor effect would bypass random-effect
                            // shrinkage and degrade BLUP-style predictions.
                            if !random_terms.iter().any(|rt| rt.name == by_name)
                                && !penalized_group_owner_present
                            {
                                random_terms.push(RandomEffectTermSpec {
                                    name: by_name.clone(),
                                    feature_col: by_col,
                                    drop_first_level: true,
                                    penalized: false,
                                    frozen_levels: None,
                                    // Unpenalized treatment-coded FIXED factor main
                                    // effect for a factor-by smooth: an unseen level
                                    // is out of contract and must raise, not center
                                    // (#2102).
                                    lenient_unseen: false,
                                });
                            }
                            // Unordered factor-by smooths are independent
                            // level-specific smooths. Preserve that
                            // term-spec structure explicitly so later
                            // hierarchy/identifiability passes can see the
                            // per-level ownership rather than a generic
                            // BySmooth envelope.
                            for (level_bits, level_label) in levels {
                                smooth_terms.push(SmoothTermSpec {
                                    frozen_parametric_residualization: None,
                                    name: format!("{label}:by={by_name}[{level_label}]"),
                                    basis: SmoothBasisSpec::ByVariable {
                                        inner: Box::new(inner_basis.clone()),
                                        by_col,
                                        kind: BySmoothKind::Level { level_bits },
                                        by: ByVariableSpec::Level {
                                            value_bits: level_bits,
                                            label: level_label,
                                        },
                                    },
                                    shape: shape.clone(),
                                    joint_null_rotation: None,
                                });
                            }
                        }
                        ColumnKindTag::Binary | ColumnKindTag::Continuous => {
                            smooth_terms.push(SmoothTermSpec {
                                frozen_parametric_residualization: None,
                                name: label.clone(),
                                basis: SmoothBasisSpec::ByVariable {
                                    inner: Box::new(inner_basis),
                                    by_col,
                                    kind: BySmoothKind::Numeric,
                                    by: ByVariableSpec::Numeric,
                                },
                                shape,
                                joint_null_rotation: None,
                            });
                        }
                    }
                } else {
                    smooth_terms.push(SmoothTermSpec {
                        frozen_parametric_residualization: None,
                        name: label.clone(),
                        basis: inner_basis,
                        shape,
                        joint_null_rotation: None,
                    });
                }
            }
            ParsedTerm::LinkWiggle { .. }
            | ParsedTerm::TimeWiggle { .. }
            | ParsedTerm::LinkConfig { .. }
            | ParsedTerm::SurvivalConfig { .. } => {
                // Consumed at formula level, not design terms.
            }
            ParsedTerm::SlopeSurface { .. } => {
                return Err(TermBuilderError::malformed_formula(
                    "slope(...) declarations must be resolved by the marginal-slope formula path before building a term spec",
                ));
            }
            ParsedTerm::Interaction {
                vars,
                double_penalty,
            } => {
                // A linear `:` interaction realizes one design column equal to
                // the elementwise product of its operands. Numeric (continuous/
                // binary) operands multiply directly; a categorical operand is
                // a factor, so the product is expanded factor-aware: one design
                // column per surviving cell of the factor(s), each an indicator
                // `1[factor == level]` gating the numeric product.
                //
                // Coding is MARGINALITY-AWARE (gam#1158, gam#1159). A categorical
                // operand `g` is treatment-coded (its lexicographically first
                // reference level dropped) ONLY when the lower-order term obtained
                // by removing `g` from this interaction is also present in the
                // model — that lower-order term is what makes the dropped level
                // identifiable, exactly mgcv's marginality rule. When that parent
                // is ABSENT (the interaction-only form), dropping the reference
                // level instead pins a group to the reference fit (a rank-deficient
                // design), so we keep ALL levels (full dummy coding) and rely on a
                // single intercept cell-drop below for identifiability:
                //   * `y ~ x:g` with no `x` main effect → "common intercept,
                //     separate slopes": every group keeps its own x-slope.
                //   * `y ~ g:h` with no `g`/`h` main effects → the saturated
                //     cell-means model: full cross of all levels minus one
                //     reference cell absorbed by the intercept.
                // When the parents ARE present (`x + x:g`, or `g*h` = `g + h +
                // g:h`), the historical treatment coding is preserved so those
                // forms stay correct.
                //
                // A main effect for var V is a `Linear`/`BoundedLinear`/
                // `RandomEffect` ParsedTerm whose referenced name is V (an
                // auto-detected categorical `Linear` becomes a RandomEffect main
                // effect; either spelling counts). We only treat such standalone
                // main-effect terms as parents — not V appearing inside another
                // interaction.
                let main_effect_present = |target: &str| -> bool {
                    terms.iter().any(|other| match other {
                        ParsedTerm::Linear { name, .. }
                        | ParsedTerm::BoundedLinear { name, .. }
                        | ParsedTerm::RandomEffect { name, .. } => name == target,
                        _ => false,
                    })
                };
                // The lower-order parent of dropping operand `drop_var` from this
                // interaction is present iff EVERY other operand is a main effect.
                // For the two cases we care about (`x:g`, `g:h`) the interaction
                // has two operands, so this reduces to "is the single remaining
                // operand a main effect"; the general form handles any arity.
                let parent_present = |drop_var: &str| -> bool {
                    vars.iter()
                        .filter(|v| v.as_str() != drop_var)
                        .all(|v| main_effect_present(v))
                };

                let mut numeric_cols = Vec::<usize>::new();
                // Per categorical operand: (var name, col, kept levels, was the
                // reference level dropped / treatment-coded?).
                let mut categorical_factors =
                    Vec::<(String, usize, Vec<(u64, String)>, bool)>::new();
                for var in vars {
                    let col = resolve_col(col_map, var)?;
                    let kind = ds.column_kinds.get(col).copied().ok_or_else(|| {
                        TermBuilderError::missing_column(format!(
                            "internal column-kind lookup failed for '{var}'"
                        ))
                        .to_string()
                    })?;
                    match kind {
                        ColumnKindTag::Continuous | ColumnKindTag::Binary => numeric_cols.push(col),
                        ColumnKindTag::Categorical => {
                            let mut levels = encoded_levels_for_column(ds, ColIdx::new(col));
                            // Treatment-code (drop the reference level) only when
                            // the marginal parent that identifies it is present;
                            // otherwise keep every level (full dummy coding).
                            let treatment_coded = parent_present(var);
                            if treatment_coded && levels.len() > 1 {
                                levels.remove(0);
                            }
                            if levels.is_empty() {
                                return Err(TermBuilderError::incompatible_config(format!(
                                    "interaction `{}` references categorical column `{var}` with no usable levels",
                                    vars.join(":")
                                )));
                            }
                            categorical_factors.push((var.clone(), col, levels, treatment_coded));
                        }
                    }
                }

                let label = vars.join(":");

                if categorical_factors.is_empty() {
                    // Pure numeric `:` interaction — single product column,
                    // identical to the historical behaviour.
                    linear_terms.push(LinearTermSpec {
                        name: label,
                        feature_col: numeric_cols[0],
                        feature_cols: numeric_cols,
                        categorical_levels: vec![],
                        // Interactions are recoverable as zero by default.
                        double_penalty: *double_penalty,
                        coefficient_geometry: LinearCoefficientGeometry::Unconstrained,
                        coefficient_min: None,
                        coefficient_max: None,
                        frozen_function_mass: None,
                    });
                    inference_notes.push(format!(
                        "wired linear interaction `{}` as product of numeric columns",
                        vars.join(":")
                    ));
                } else {
                    // Factor-aware expansion: cartesian product over the kept
                    // levels of every categorical operand. Each cell yields one
                    // column gating the numeric product (or, with no numeric
                    // operand, a pure cell indicator).
                    let mut cells: Vec<Vec<(usize, u64, String)>> = vec![Vec::new()];
                    for (_var, col, levels, _treatment_coded) in &categorical_factors {
                        let mut next = Vec::with_capacity(cells.len() * levels.len());
                        for cell in &cells {
                            for (bits, level_label) in levels {
                                let mut extended = cell.clone();
                                extended.push((*col, *bits, level_label.clone()));
                                next.push(extended);
                            }
                        }
                        cells = next;
                    }

                    // Intercept-identifiability cell drop. When the cells are PURE
                    // INDICATORS (no numeric operand) and at least one factor was
                    // dummy-coded (kept all its levels), the full set of cell
                    // columns sums to the all-ones intercept and is rank-deficient
                    // against it. Drop exactly ONE reference cell — the cell where
                    // every factor sits at its reference (lexicographically first)
                    // level — so the remaining saturated cells are identifiable
                    // (rank n_g*n_h - 1 cells + intercept). With a numeric operand
                    // the cells gate `x` and sum to `x`, not the intercept, so no
                    // cell is dropped (the collinearity there is with the absent
                    // `x` main effect, which is exactly why full coding is right).
                    let any_dummy_coded = categorical_factors
                        .iter()
                        .any(|(_, _, _, treatment_coded)| !*treatment_coded);
                    if numeric_cols.is_empty() && any_dummy_coded {
                        // The reference cell pairs each factor's column with the
                        // bits of its lexicographically-first (index 0) level.
                        let reference_cell: Vec<(usize, u64)> = categorical_factors
                            .iter()
                            .map(|(_, col, _, _)| {
                                let levels = encoded_levels_for_column(ds, ColIdx::new(*col));
                                (*col, levels[0].0)
                            })
                            .collect();
                        cells.retain(|cell| {
                            !reference_cell.iter().all(|(rcol, rbits)| {
                                cell.iter()
                                    .any(|(col, bits, _)| col == rcol && bits == rbits)
                            })
                        });
                    }

                    let n_cells = cells.len();
                    for cell in cells {
                        let cell_suffix = cell
                            .iter()
                            .map(|(_, _, level_label)| level_label.as_str())
                            .collect::<Vec<_>>()
                            .join(":");
                        let categorical_levels =
                            cell.iter().map(|(col, bits, _)| (*col, *bits)).collect();
                        // `feature_col` is required to point at a real column;
                        // use the first numeric operand when present, otherwise
                        // the first categorical column (its raw value is never
                        // multiplied — `realized_design_column` starts from ones
                        // and only gates by the level indicators).
                        let feature_col = numeric_cols
                            .first()
                            .copied()
                            .unwrap_or(categorical_factors[0].1);
                        linear_terms.push(LinearTermSpec {
                            name: format!("{label}:{cell_suffix}"),
                            feature_col,
                            feature_cols: numeric_cols.clone(),
                            categorical_levels,
                            double_penalty: *double_penalty,
                            coefficient_geometry: LinearCoefficientGeometry::Unconstrained,
                            coefficient_min: None,
                            coefficient_max: None,
                            frozen_function_mass: None,
                        });
                    }
                    let all_treatment_coded = !any_dummy_coded;
                    let coding = if all_treatment_coded {
                        "treatment-coded"
                    } else {
                        "marginality-aware (full dummy / saturated)"
                    };
                    inference_notes.push(format!(
                        "wired factor-aware linear interaction `{}` as {} {} cell column(s)",
                        vars.join(":"),
                        n_cells,
                        coding
                    ));
                }
            }
        }
    }

    Ok(TermCollectionSpec {
        linear_terms,
        random_effect_terms: random_terms,
        smooth_terms,
    })
}

fn split_list_option(raw: &str) -> Vec<String> {
    let t = raw.trim();
    // Accept the Python/JSON list form `[a, b]` AND mgcv's R-vector forms
    // `c(a, b)` / `(a, b)` as bracketed wrappers around a comma-separated body.
    // mgcv-style formulas pass per-margin numeric options as `k=c(5,5)` /
    // `period=c(2*pi, pi)`; without R-vector peeling here those entries were
    // split into `["c(5", "5)"]` and the downstream numeric parser then
    // misreported the leading garbage as the invalid digit.
    let inner = t
        .strip_prefix('[')
        .and_then(|u| u.strip_suffix(']'))
        .or_else(|| {
            t.strip_prefix("c(")
                .or_else(|| t.strip_prefix("C("))
                .or_else(|| t.strip_prefix('('))
                .and_then(|u| u.strip_suffix(')'))
        })
        .unwrap_or(t);
    inner
        .split(',')
        .map(|v| v.trim().to_string())
        .filter(|v| !v.is_empty())
        .collect()
}

fn parse_numeric_expr(raw: &str) -> Result<f64, String> {
    let mut acc = 1.0f64;
    let normalized = raw.replace(' ', "");
    if normalized.eq_ignore_ascii_case("none") {
        return Err("None is not numeric".to_string());
    }
    for factor in normalized.split('*') {
        if factor.is_empty() {
            return Err(format!("invalid numeric expression '{raw}'"));
        }
        let value = if factor.eq_ignore_ascii_case("pi") || factor == "π" {
            std::f64::consts::PI
        } else if factor.eq_ignore_ascii_case("tau") || factor == "τ" {
            std::f64::consts::TAU
        } else if let Some(prefix) = factor
            .strip_suffix("pi")
            .or_else(|| factor.strip_suffix("π"))
        {
            let coefficient = if prefix.is_empty() {
                1.0
            } else {
                prefix
                    .parse::<f64>()
                    .map_err(|err| format!("invalid numeric expression '{raw}': {err}"))?
            };
            coefficient * std::f64::consts::PI
        } else if let Some(prefix) = factor
            .strip_suffix("tau")
            .or_else(|| factor.strip_suffix("τ"))
        {
            let coefficient = if prefix.is_empty() {
                1.0
            } else {
                prefix
                    .parse::<f64>()
                    .map_err(|err| format!("invalid numeric expression '{raw}': {err}"))?
            };
            coefficient * std::f64::consts::TAU
        } else {
            factor
                .parse::<f64>()
                .map_err(|err| format!("invalid numeric expression '{raw}': {err}"))?
        };
        acc *= value;
    }
    Ok(acc)
}

/// Read an endpoint/period option as a numeric *expression* (`2*pi`, `tau`,
/// `0.5*tau`, `6.283185307179586`, ...) — the same grammar that `period=` and
/// `origin=` already accept via [`parse_numeric_expr`].
///
/// Returns `Ok(None)` when the key is absent, `Ok(Some(v))` when it parses, and
/// a hard `Err` when the key is *present but unparseable*. The crucial contrast
/// is with the lenient [`option_f64`], which collapses an unparseable value to
/// `None` and lets the caller silently substitute the data range — wrapping a
/// cyclic smooth at the wrong period with no diagnostic (the #815 failure mode).
fn option_numeric_expr(
    options: &BTreeMap<String, String>,
    key: &str,
) -> Result<Option<f64>, String> {
    match options.get(key) {
        None => Ok(None),
        Some(raw) => parse_numeric_expr(raw)
            .map(Some)
            .map_err(|err| format!("option `{key}={raw}` is not a valid numeric value: {err}")),
    }
}

fn parse_periods_option(
    options: &BTreeMap<String, String>,
    dim: usize,
) -> Result<Option<Vec<Option<f64>>>, String> {
    let Some(raw) = options.get("period") else {
        return Ok(None);
    };
    let values = split_list_option(raw);
    let mut periods = vec![None; dim];
    if values.len() == 1 && dim == 1 {
        periods[0] = Some(parse_numeric_expr(&values[0])?);
    } else {
        if values.len() != dim {
            return Err(format!(
                "period list length {} must match smooth dimension {}",
                values.len(),
                dim
            ));
        }
        for (i, v) in values.iter().enumerate() {
            if v.eq_ignore_ascii_case("none") {
                continue;
            }
            periods[i] = Some(parse_numeric_expr(v)?);
        }
    }
    Ok(Some(periods))
}

fn parse_periodic_axes_option(
    options: &BTreeMap<String, String>,
    dim: usize,
) -> Result<Option<Vec<Option<f64>>>, String> {
    // `cyclic=` is whitelisted as the alias of `periodic=` on every radial arm,
    // so read it here too; it was previously accepted and dropped (#2781).
    let Some(raw_axes) = options.get("periodic").or_else(|| options.get("cyclic")) else {
        // No periodicity FLAG — but a declared period is itself the declaration.
        // A period is not a property an aperiodic basis has, so an axis that
        // carries one is periodic, exactly as on the 1-D B-spline and tensor
        // paths (`axes_with_declared_period`). Before #2781 this early return
        // dropped `matern(x, z, period=[2*pi, None])` on the floor: the option
        // was validated by the arm's whitelist and then never read.
        let declared = parse_periods_option(options, dim)?;
        return Ok(match declared {
            Some(periods) if periods.iter().any(Option::is_some) => Some(periods),
            _ => None,
        });
    };
    let mut periods = parse_periods_option(options, dim)?.unwrap_or_else(|| vec![None; dim]);
    // Scalar boolean form (`periodic=true` / `false`, `yes` / `no`) applies to
    // every axis — the documented per-axis-flag broadcast (see the doc on
    // `parse_periodic_axes`, the tensor sibling that already accepts it). A
    // 1-D `duchon(x, periodic=true)` lands here: the cyclic *domain* is then
    // resolved from the data range by `parse_cyclic_boundary` (the 1-D builder
    // consults `boundary` first), so a finite explicit period is NOT required —
    // we only need to NOT mis-read "true" as an axis index (#1074). `false`
    // means no axis is periodic.
    let lowered = raw_axes.trim().to_ascii_lowercase();
    if matches!(lowered.as_str(), "true" | "yes" | "y") {
        return Ok(Some(periods));
    }
    // `false` means NO axis is periodic. Return `None` — NOT
    // `Some(vec![None; dim])` — because the radial 1-D consumer treats a
    // `Some([None])` as "periodicity requested, derive the wrap period from
    // the data range" (see the Duchon builder arm below, which back-fills
    // `axes[0] = data_span` for a lone `None`) and the 1-D builder routes on
    // `spec.periodic.is_some()`. Emitting `Some([None])` here therefore
    // silently produced a *periodic* smooth for an explicit `periodic=false`
    // — the exact regression this branch now avoids, matching the bracketed
    // `[false]` form handled by the per-axis boolean block below.
    if matches!(lowered.as_str(), "false" | "no" | "n") {
        return Ok(None);
    }
    let axes = split_list_option(raw_axes);
    if axes.is_empty() {
        return Ok(Some(periods));
    }

    // Boolean forms `periodic=true` / `periodic=[true, false, ...]`, mirroring
    // `parse_tensor_periodic_axes`. The radial 1-D builders (`duchon`/`tps`/
    // `matern`) intentionally DERIVE the wrap period from the closed center
    // lattice when none is supplied (`prepare_periodic_duchon_centers_1d_with_period`,
    // gam#580: `None => span`), so a boolean-selected periodic axis legitimately
    // omits `period`. Without this branch, `duchon(x, periodic=true)`-style
    // radial formulas failed with the misleading "invalid periodic axis 'true'".
    let is_bool = |t: &str| {
        matches!(
            t.to_ascii_lowercase().as_str(),
            "true" | "yes" | "y" | "false" | "no" | "n"
        )
    };
    let is_truthy = |t: &str| matches!(t.to_ascii_lowercase().as_str(), "true" | "yes" | "y");

    // Scalar boolean: `periodic=true` / `periodic=false`.
    if axes.len() == 1 && is_bool(&axes[0]) {
        if !is_truthy(&axes[0]) {
            // Non-periodic: return None so the 1-D builder (which routes on
            // `spec.periodic.is_some()`) does NOT take the periodic path.
            return Ok(None);
        }
        // Every axis periodic; honor any explicit per-axis period, else leave
        // `None` for the caller (formula arm) / builder to derive the span.
        return Ok(Some(periods));
    }

    // Per-axis boolean list: `periodic=[true, false, ...]` (length must match dim).
    if axes.iter().all(|a| is_bool(a)) {
        if axes.len() != dim {
            return Err(format!(
                "periodic flag list length {} must match smooth dimension {dim}",
                axes.len()
            ));
        }
        if !axes.iter().any(|a| is_truthy(a)) {
            return Ok(None);
        }
        for (i, a) in axes.iter().enumerate() {
            if !is_truthy(a) {
                periods[i] = None;
            }
        }
        return Ok(Some(periods));
    }

    // Index-list form: `periodic=[0, 2]`. Each listed axis must carry an
    // explicit finite period — an index gives no per-axis span-derive hint.
    for a in &axes {
        let axis = a
            .parse::<usize>()
            .map_err(|err| format!("invalid periodic axis '{a}': {err}"))?;
        if axis >= dim {
            return Err(format!(
                "periodic axis {axis} out of range for {dim}D smooth"
            ));
        }
        if periods[axis].is_none() {
            return Err(format!(
                "periodic axis {axis} requires period[{axis}] to be finite"
            ));
        }
    }
    // Axes not listed are non-periodic even if period list has a finite placeholder.
    let listed: std::collections::BTreeSet<usize> = axes
        .iter()
        .filter_map(|a| a.parse::<usize>().ok())
        .collect();
    for i in 0..dim {
        if !listed.contains(&i) {
            periods[i] = None;
        }
    }
    Ok(Some(periods))
}

// ---------------------------------------------------------------------------
// Smooth basis spec construction
// ---------------------------------------------------------------------------

fn parse_option_list(raw: &str) -> Vec<String> {
    let trimmed = raw.trim();
    // Accept both the Python/JSON list form `[a, b]` and mgcv's R vector form
    // `c(a, b)` (and a bare `(a, b)`) as the bracketed wrapper around a
    // comma-separated option list. mgcv writes per-margin options as
    // `bs=c('tp','tp')` / `m=c(2,2)`, so the `c(...)` form must round-trip
    // through the same splitter the `[...]` form uses.
    let inner = trimmed
        .strip_prefix('[')
        .and_then(|v| v.strip_suffix(']'))
        .or_else(|| {
            trimmed
                .strip_prefix("c(")
                .or_else(|| trimmed.strip_prefix("C("))
                .or_else(|| trimmed.strip_prefix('('))
                .and_then(|v| v.strip_suffix(')'))
        })
        .unwrap_or(trimmed);
    inner
        .split(',')
        .map(|v| {
            v.trim()
                .trim_matches('"')
                .trim_matches('\'')
                .to_ascii_lowercase()
        })
        .filter(|v| !v.is_empty())
        .collect()
}

/// Axes for which the caller has explicitly declared a period.
///
/// A period is not a property an aperiodic basis has: declaring one *is* the
/// periodicity declaration, and `periodic=` / `bc='periodic'` is a second,
/// redundant spelling of the same fact for the axes it names. Before #2781 the
/// axis resolvers read only that second spelling, so `s(t, period=24)`,
/// `s(t, period_start=0, period_end=24)` and
/// `te(th, h, periods=[2*pi, None])` were each validated as a legal option and
/// then dropped on the floor — the caller asked for a cyclic smooth, got an
/// aperiodic one with a discontinuity at the seam, and was never told.
///
/// Only *unambiguous* declarations are read here: a per-axis list (which
/// includes the scalar form on a 1-D smooth, where the list has length one).
/// A bare scalar on a multi-margin tensor does not say which margin it belongs
/// to, so it is left to [`parse_periods`], which either broadcasts it onto a
/// lone axis already flagged periodic or refuses the length mismatch.
fn axes_with_declared_period(
    options: &BTreeMap<String, String>,
    dim: usize,
) -> Result<Vec<bool>, String> {
    let mut axes = vec![false; dim];
    if let Some(raw) = options.get("period").or_else(|| options.get("periods")) {
        let values = split_list_option(raw);
        if values.len() == dim {
            for (axis, value) in values.iter().enumerate() {
                if !value.trim().eq_ignore_ascii_case("none") {
                    axes[axis] = true;
                }
            }
        }
    }
    // The half-open endpoint spelling (`period_start=`/`period_end=`, aliases
    // `start=`/`end=`) declares the periodic DOMAIN of one axis, so it only has
    // a referent on a 1-D smooth. `parse_periodic_domain_1d` is what reads it.
    if dim == 1
        && PERIOD_ENDPOINT_OPTION_KEYS
            .iter()
            .any(|key| options.contains_key(*key))
    {
        axes[0] = true;
    }
    Ok(axes)
}

/// Option keys that declare a periodic domain by its endpoints.
const PERIOD_ENDPOINT_OPTION_KEYS: [&str; 4] = ["period_start", "period_end", "start", "end"];

/// Option keys that declare a period length.
const PERIOD_LENGTH_OPTION_KEYS: [&str; 2] = ["period", "periods"];

/// Option keys that place the start of a periodic domain.
const PERIOD_ORIGIN_OPTION_KEYS: [&str; 5] = [
    "origin",
    "origins",
    "period_origin",
    "period-origin",
    "domain_origin",
];

/// Refuse a period declaration that no axis of the smooth can consume.
///
/// After [`axes_with_declared_period`] has folded every unambiguous declaration
/// into the periodic-axis set, the only ways to still hold a period option that
/// nothing reads are (a) a bare scalar `period=` on a multi-margin tensor, which
/// does not name its margin, (b) an `origin=` with no period to be the origin
/// of, and (c) an explicit `periodic=false` that contradicts the declaration.
/// Each of those is refused here rather than discarded, which is the contract
/// `docs/formulas.md` already states for this family of options: "an unparseable
/// endpoint or an unknown option is rejected rather than silently dropped"
/// (#2781).
fn reject_unconsumable_period_declaration(
    term_name: &str,
    options: &BTreeMap<String, String>,
    periodic_axes: &[bool],
) -> Result<(), String> {
    if periodic_axes.iter().any(|periodic| *periodic) {
        return Ok(());
    }
    let dim = periodic_axes.len();
    if let Some(key) = PERIOD_LENGTH_OPTION_KEYS
        .iter()
        .find(|key| options.contains_key(**key))
    {
        let hint = if dim > 1 {
            format!(
                "a scalar `{key}=` does not say which of the {dim} margins wraps; write one entry \
                 per margin (e.g. {key}=[<value>, None]) or name the axis with periodic=<axis>"
            )
        } else {
            "declare it on a periodic axis or drop it".to_string()
        };
        return Err(TermBuilderError::invalid_option(format!(
            "{term_name}(): `{key}=` declares a period, but no axis of this smooth is periodic — {hint}"
        ))
        .to_string());
    }
    if let Some(key) = PERIOD_ORIGIN_OPTION_KEYS
        .iter()
        .find(|key| options.contains_key(**key))
    {
        return Err(TermBuilderError::invalid_option(format!(
            "{term_name}(): `{key}=` places the start of a periodic domain, but this smooth \
             declares no period; add period=<value> or drop it"
        ))
        .to_string());
    }
    if let Some(key) = PERIOD_ENDPOINT_OPTION_KEYS
        .iter()
        .find(|key| options.contains_key(**key))
    {
        return Err(TermBuilderError::invalid_option(format!(
            "{term_name}(): `{key}=` declares a periodic domain endpoint, but no axis of this \
             smooth is periodic; on a tensor smooth use periods=[...] with origins=[...], which \
             name their margin"
        ))
        .to_string());
    }
    Ok(())
}

/// The radial (`thinplate` / `matern` / `duchon`) counterpart of
/// [`reject_unconsumable_period_declaration`] (#2781).
///
/// [`parse_periodic_axes_option`] returns the per-axis period vector these arms
/// actually consume, and an axis wraps only when that vector carries a finite
/// period for it — with the one exception that a ONE-dimensional radial smooth
/// derives its period from the closed center lattice, which tiles a full period
/// exactly (gam#580), unlike the sample-dependent data-range derive the B-spline
/// path refuses (#1771). Every other spelling used to be validated by the arm's
/// whitelist and then dropped: `matern(x, z, period=0.7)` and
/// `matern(x, z, periodic=true)` were each bit-identical to the plain aperiodic
/// fit, with no error and no warning.
///
/// Two refusals:
///
/// * a periodicity or period declaration that leaves no axis periodic — which
///   on a multi-dimensional radial smooth is exactly what `periodic=true` alone
///   does, since there is no per-axis span to derive from;
/// * `period_start=` / `period_end=` on a multi-dimensional radial smooth.
///   Those name ONE axis's domain and are read by `parse_cyclic_boundary`,
///   which these arms consult only when `d == 1`.
fn reject_unconsumable_radial_period_declaration(
    term_name: &str,
    options: &BTreeMap<String, String>,
    dim: usize,
    periodic: Option<&[Option<f64>]>,
    boundary_is_cyclic: bool,
) -> Result<(), String> {
    if dim > 1
        && let Some(key) = PERIOD_ENDPOINT_OPTION_KEYS
            .iter()
            .find(|key| options.contains_key(**key))
    {
        return Err(TermBuilderError::invalid_option(format!(
            "{term_name}(): `{key}=` names one axis's periodic domain and is only read on a \
             one-dimensional radial smooth; this one has {dim} covariates, so give the wrap as \
             period=[…] with one entry per axis"
        ))
        .to_string());
    }
    let any_axis_wraps = boundary_is_cyclic
        || periodic.is_some_and(|axes| {
            (dim == 1 && !axes.is_empty()) || axes.iter().any(Option::is_some)
        });
    if any_axis_wraps {
        return Ok(());
    }
    let declared = ["periodic", "cyclic"]
        .iter()
        .chain(PERIOD_LENGTH_OPTION_KEYS.iter())
        .chain(PERIOD_ENDPOINT_OPTION_KEYS.iter())
        .find(|key| options.contains_key(**key));
    let Some(key) = declared else {
        return Ok(());
    };
    // `periodic=false` is a denial, not a declaration: it legitimately leaves
    // every axis open.
    if matches!(*key, "periodic" | "cyclic")
        && options
            .get(*key)
            .map(|raw| raw.trim().to_ascii_lowercase())
            .is_some_and(|raw| matches!(raw.as_str(), "false" | "no" | "n"))
    {
        return Ok(());
    }
    Err(TermBuilderError::invalid_option(format!(
        "{term_name}(): `{key}=` declares periodicity, but no axis of this smooth ends up \
         periodic. A radial smooth derives its wrap from the center lattice only in one \
         dimension (this one has {dim}), so name the period per axis: \
         period=[<value>, None, …]"
    ))
    .to_string())
}

fn parse_periodic_axes(
    options: &BTreeMap<String, String>,
    dim: usize,
) -> Result<Vec<bool>, String> {
    let mut axes = vec![false; dim];
    // `periodic=false` is an explicit denial, not merely the absence of a
    // declaration: it suppresses the `boundary=` spelling below, and it
    // CONTRADICTS a period declaration rather than silently outranking it.
    let mut explicitly_aperiodic = false;
    if let Some(raw) = options.get("periodic").or_else(|| options.get("cyclic")) {
        let lowered = raw.trim().to_ascii_lowercase();
        if matches!(lowered.as_str(), "true" | "yes" | "y") {
            axes.fill(true);
        } else if matches!(lowered.as_str(), "false" | "no" | "n") {
            explicitly_aperiodic = true;
        } else {
            for axis_raw in parse_option_list(raw) {
                let axis = axis_raw
                    .parse::<usize>()
                    .map_err(|err| format!("invalid periodic axis '{axis_raw}': {err}"))?;
                if axis >= dim {
                    return Err(format!(
                        "periodic axis {axis} out of range for {dim}D smooth"
                    ));
                }
                axes[axis] = true;
            }
        }
    }
    if !explicitly_aperiodic
        && let Some(raw) = options.get("boundary").or_else(|| options.get("bc"))
    {
        let boundary = parse_option_list(raw);
        if boundary.len() == dim {
            for (axis, value) in boundary.iter().enumerate() {
                if matches!(value.as_str(), "periodic" | "cyclic" | "cc") {
                    axes[axis] = true;
                }
            }
        } else if dim == 1
            && matches!(
                boundary.first().map(String::as_str),
                Some("periodic" | "cyclic" | "cc")
            )
        {
            axes[0] = true;
        }
    }
    fold_in_declared_periods(options, dim, &mut axes, explicitly_aperiodic)?;
    Ok(axes)
}

/// Fold every unambiguous period declaration into `axes` (#2781), refusing a
/// declaration that an explicit `periodic=false` contradicts.
///
/// Shared by the 1-D and tensor axis resolvers so one rule — "a declared period
/// makes its axis periodic" — holds on both paths.
fn fold_in_declared_periods(
    options: &BTreeMap<String, String>,
    dim: usize,
    axes: &mut [bool],
    explicitly_aperiodic: bool,
) -> Result<(), String> {
    let declared = axes_with_declared_period(options, dim)?;
    if explicitly_aperiodic && declared.iter().any(|d| *d) {
        return Err(TermBuilderError::incompatible_config(
            "periodic=false denies the periodicity that the smooth's own period declaration \
             asserts; drop one of the two",
        )
        .to_string());
    }
    for (axis, declared_axis) in declared.into_iter().enumerate() {
        axes[axis] |= declared_axis;
    }
    Ok(())
}

fn parse_optional_numeric_list(
    options: &BTreeMap<String, String>,
    keys: &[&str],
    dim: usize,
) -> Result<Vec<Option<f64>>, String> {
    let Some(raw) = keys.iter().find_map(|key| options.get(*key)) else {
        return Ok(vec![None; dim]);
    };
    let values = split_list_option(raw);
    let mut out = vec![None; dim];
    if values.len() == 1 && dim == 1 {
        if !values[0].eq_ignore_ascii_case("none") {
            out[0] = Some(parse_numeric_expr(&values[0])?);
        }
        return Ok(out);
    }
    if values.len() != dim {
        return Err(format!(
            "numeric option list length {} must match smooth dimension {}",
            values.len(),
            dim
        ));
    }
    for (i, value) in values.iter().enumerate() {
        if !value.eq_ignore_ascii_case("none") {
            out[i] = Some(parse_numeric_expr(value)?);
        }
    }
    Ok(out)
}

fn parse_periods(
    options: &BTreeMap<String, String>,
    periodic_axes: &[bool],
) -> Result<Vec<Option<f64>>, String> {
    let dim = periodic_axes.len();
    // Broadcast a single-element `period=[v]` onto the lone periodic axis
    // of a multi-axis smooth (e.g. `te(th, h, bc=['periodic','natural'],
    // period=[2*pi])`): with only one periodic margin, the value can only
    // belong there.
    let lone_periodic_broadcast = options
        .get("period")
        .or_else(|| options.get("periods"))
        .and_then(|raw| {
            let values = split_list_option(raw);
            if values.len() != 1 || dim <= 1 {
                return None;
            }
            let mut iter = periodic_axes.iter().enumerate().filter(|(_, p)| **p);
            let first = iter.next()?;
            if iter.next().is_some() {
                return None;
            }
            Some((first.0, values.into_iter().next()?))
        });
    let periods = if let Some((axis, value)) = lone_periodic_broadcast {
        let mut out = vec![None; dim];
        if !value.eq_ignore_ascii_case("none") {
            out[axis] = Some(parse_numeric_expr(&value)?);
        }
        out
    } else {
        parse_optional_numeric_list(options, &["period", "periods"], dim)?
    };
    for (axis, (periodic, period)) in periodic_axes.iter().zip(periods.iter()).enumerate() {
        if *periodic
            && let Some(value) = period
            && (!value.is_finite() || *value <= 0.0)
        {
            return Err(format!(
                "period for periodic axis {axis} must be finite and positive, got {value}"
            ));
        }
    }
    Ok(periods)
}

fn parse_period_origins(
    options: &BTreeMap<String, String>,
    periodic_axes: &[bool],
) -> Result<Vec<Option<f64>>, String> {
    parse_optional_numeric_list(
        options,
        &[
            "origin",
            "origins",
            "period_origin",
            "period-origin",
            "domain_origin",
        ],
        periodic_axes.len(),
    )
}

/// Parse a per-axis periodic flag list for tensor smooths. Accepts three forms:
/// - `periodic=true` / `periodic=false` (scalar applied to every axis),
/// - `periodic=[true, false, ...]` (one flag per axis, length `dim`),
/// - `periodic=c(1, 1)` / `c(0, 0)` (a length-`dim` 0/1 mask, mgcv's
///   per-margin spelling — distinguished from an axis-index list by the
///   repeated 0/1 value), and
/// - `periodic=[0, 2, ...]` (axis indices that are periodic; others are not).
///
/// `boundary=[..., "periodic"/"cyclic"/"cc", ...]` may also flip individual
/// axes on; non-matching tokens leave the existing flag unchanged.
fn parse_tensor_periodic_axes(
    options: &BTreeMap<String, String>,
    dim: usize,
) -> Result<Vec<bool>, String> {
    let mut axes = vec![false; dim];
    if let Some(raw) = options.get("periodic").or_else(|| options.get("cyclic")) {
        let lowered = raw.trim().to_ascii_lowercase();
        match lowered.as_str() {
            "true" | "yes" | "y" => {
                axes.fill(true);
            }
            "false" | "no" | "n" => {
                // Already false; allow `boundary=` below to flip axes if set.
            }
            _ => {
                let entries = parse_option_list(raw);
                let all_bool = !entries.is_empty()
                    && entries.iter().all(|v| {
                        matches!(
                            v.as_str(),
                            "true" | "yes" | "y" | "false" | "no" | "n" | "none"
                        )
                    });
                // mgcv writes per-margin flag vectors as `periodic=c(1,1)` /
                // `periodic=c(0,0)` — a length-`dim` mask where each entry is a
                // 0/1 flag for THAT margin, not an axis index. A bare axis-index
                // list (`periodic=[0,1]`, `periodic=[0]`) lists DISTINCT margin
                // indices to turn on. The two collide only when the list is all
                // 0/1 of length `dim`; disambiguate by the repeated-value
                // signature `c(1,1)`/`c(0,0)` (a valid axis-index set never
                // repeats an index), which is the canonical mask spelling. This
                // is what makes the leading tensor margin honor its periodic flag
                // (#1751: `periodic=c(1,1)` previously parsed `1,1` as axis
                // indices, marking only axis 1 and dropping axis 0).
                let all_zero_one =
                    !entries.is_empty() && entries.iter().all(|v| v == "0" || v == "1");
                let has_repeat = {
                    let mut seen = std::collections::BTreeSet::new();
                    !entries.iter().all(|v| seen.insert(v.clone()))
                };
                let numeric_mask = all_zero_one && entries.len() == dim && has_repeat;
                if all_bool || numeric_mask {
                    if entries.len() != dim {
                        return Err(format!(
                            "periodic list length {} must match smooth dimension {}",
                            entries.len(),
                            dim
                        ));
                    }
                    for (i, v) in entries.iter().enumerate() {
                        axes[i] = matches!(v.as_str(), "true" | "yes" | "y" | "1");
                    }
                } else {
                    for axis_raw in entries {
                        let axis = axis_raw
                            .parse::<usize>()
                            .map_err(|err| format!("invalid periodic axis '{axis_raw}': {err}"))?;
                        if axis >= dim {
                            return Err(format!(
                                "periodic axis {axis} out of range for {dim}D smooth"
                            ));
                        }
                        axes[axis] = true;
                    }
                }
            }
        }
    }
    if let Some(raw) = options.get("boundary").or_else(|| options.get("bc")) {
        let boundary = parse_option_list(raw);
        // A scalar token applies to every margin; `validate_tensor_boundary_tokens`
        // has already refused any other length (#2782).
        if boundary.len() == 1 {
            if matches!(boundary[0].as_str(), "periodic" | "cyclic" | "cc") {
                axes.fill(true);
            }
        } else if boundary.len() == dim {
            for (axis, value) in boundary.iter().enumerate() {
                if matches!(value.as_str(), "periodic" | "cyclic" | "cc") {
                    axes[axis] = true;
                }
            }
        }
    }
    // A per-margin basis vector (`bs=c('cc','ps')` / `type=[...]`) declares each
    // margin's basis family, and a cyclic family (`cc`/`cp`/`cyclic`) makes THAT
    // margin periodic — exactly as the 1-D `s(x, bs='cc')` smooth wraps its lone
    // axis. Without this, the per-margin `cc` token was validated but discarded:
    // every `bs=c(...)` spelling collapsed to the same open B-spline tensor
    // (#1752). Only honor the vector form here; a scalar `bs='cc'` on a tensor is
    // ambiguous about which margins wrap, so it does not flip any axis on.
    if let Some(raw) = options.get("bs").or_else(|| options.get("type"))
        && bs_selector_is_vector(raw)
    {
        let per_margin = parse_option_list(raw);
        if per_margin.len() == dim {
            for (axis, margin_bs) in per_margin.iter().enumerate() {
                if matches!(canonicalize_smooth_type(margin_bs), "cc" | "cp" | "cyclic") {
                    axes[axis] = true;
                }
            }
        }
    }
    // A per-margin period list names its own margins, so it declares
    // periodicity just as `periodic=`/`bc=` do (#2781). Without this,
    // `te(th, h, periods=[2*pi, None])` was bit-identical to `te(th, h)`.
    let explicitly_aperiodic = options
        .get("periodic")
        .or_else(|| options.get("cyclic"))
        .is_some_and(|raw| {
            matches!(
                raw.trim().to_ascii_lowercase().as_str(),
                "false" | "no" | "n"
            )
        });
    fold_in_declared_periods(options, dim, &mut axes, explicitly_aperiodic)?;
    Ok(axes)
}

/// Validate the per-margin `boundary=`/`bc=` tokens on a tensor-product smooth.
///
/// The tensor `boundary`/`bc` list selects, per margin, whether the margin
/// *wraps* (a `periodic`/`cyclic`/`cc` token, consumed by
/// [`parse_tensor_periodic_axes`]) or is an ordinary non-periodic margin. In the
/// tensor DSL a *non-periodic* margin is spelled `clamped` — in the B-spline
/// sense of a **clamped knot vector**, i.e. the standard open spline that is
/// free at its two ends and does not wrap (exactly how the callers document it:
/// "non-periodic / clamped … free at the two ends, no wrap"). It is therefore an
/// inert marker here, not a zero-derivative endpoint reparameterization: a
/// cylinder `te(theta, z, boundary=['periodic','clamped'], …)` is a cyclic θ
/// margin tensor-producted with an ordinary open z margin, the direct analog of
/// mgcv `te(bs=c("cc","ps"))` / `te(bs=c("cc","cr"))`.
///
/// The periodic selectors and the inert non-periodic markers
/// (`clamped`/`open`/`natural`/`free`/`none`/empty) are accepted; anything else
/// (e.g. a genuine `anchored` zero-value endpoint constraint, which has no
/// ordinary-margin meaning in a tensor) is surfaced as a clean
/// unsupported-feature error rather than silently dropped. Previously `clamped`
/// itself was rejected, so the cylinder/torus mixed-boundary tensors — the exact
/// construction the manifold quality suite builds — could not be fit at all.
fn validate_tensor_boundary_tokens(
    options: &BTreeMap<String, String>,
    dim: usize,
) -> Result<(), String> {
    let Some(raw) = options.get("boundary").or_else(|| options.get("bc")) else {
        return Ok(());
    };
    let entries = parse_option_list(raw);
    // A scalar token applies to every margin (the same broadcast `k=`, `bs=` and
    // `degree=` use); any other length names margins that do not exist. Both were
    // previously accepted and then dropped by the `len() == dim` guard in
    // `parse_tensor_periodic_axes`, so `te(x, z, bc='periodic')` silently built
    // an aperiodic tensor (#2782).
    if entries.len() != 1 && entries.len() != dim {
        return Err(TermBuilderError::invalid_option(format!(
            "tensor smooth bc/boundary={raw:?} has {} entries but the smooth has {dim} margins; \
             pass one token per margin or a single token for all of them",
            entries.len()
        ))
        .to_string());
    }
    for (axis, value) in entries.iter().enumerate() {
        let inert = matches!(
            value.trim().to_ascii_lowercase().as_str(),
            "clamped" | "open" | "natural" | "free" | "none" | "" | "periodic" | "cyclic" | "cc"
        );
        if !inert {
            return Err(TermBuilderError::unsupported_feature(format!(
                "tensor smooth margin {axis} boundary token '{value}' is not supported \
                 (got bc/boundary={raw:?} on a {dim}-D tensor); tensor margins accept the periodic \
                 selectors (periodic/cyclic/cc) or the non-periodic markers (clamped/open/natural/free). \
                 Apply anchored/zero-value endpoint constraints with a 1-D s(x, bc=...) term instead."
            ))
            .to_string());
        }
    }
    Ok(())
}

fn tensor_k_axis_option_axis(
    key: &str,
    cols: &[usize],
    ds: &Dataset,
) -> Result<Option<usize>, String> {
    let Some(suffix) = key.strip_prefix("k_") else {
        return Ok(None);
    };
    if suffix.is_empty() {
        return Err("tensor k axis option must be named k_<axis> or k_<variable>".to_string());
    }
    if let Ok(axis) = suffix.parse::<usize>() {
        return if axis < cols.len() {
            Ok(Some(axis))
        } else {
            Err(format!(
                "tensor k axis option `{key}` references axis {axis}, but the smooth has {} margins",
                cols.len()
            ))
        };
    }

    let mut matches = cols
        .iter()
        .enumerate()
        .filter(|(_, col)| ds.headers.get(**col).is_some_and(|name| name == suffix))
        .map(|(axis, _)| axis);
    let first = matches.next();
    if matches.next().is_some() {
        return Err(format!(
            "tensor k axis option `{key}` matches more than one margin named `{suffix}`"
        ));
    }
    first.map(Some).ok_or_else(|| {
        let margin_names = cols
            .iter()
            .enumerate()
            .map(|(axis, col)| {
                let name = ds
                    .headers
                    .get(*col)
                    .map(String::as_str)
                    .unwrap_or("<unnamed>");
                format!("{axis}:{name}")
            })
            .collect::<Vec<_>>()
            .join(", ");
        format!(
            "tensor k axis option `{key}` does not match a margin index or name; tensor margins are [{margin_names}]"
        )
    })
}

fn is_tensor_k_axis_option_key(key: &str) -> bool {
    key.strip_prefix("k_")
        .is_some_and(|suffix| !suffix.is_empty())
}

/// Parse a per-margin basis dimension list (`k=<scalar>`, `k=[k0, k1, ...]`,
/// or axis aliases like `k_x=...` / `k_0=...`). A scalar is broadcast across
/// all axes; `None` returns the heuristic from the data column.
fn parse_tensor_k_list(
    options: &BTreeMap<String, String>,
    cols: &[usize],
    ds: &Dataset,
) -> Result<(Vec<usize>, bool), String> {
    let mut axis_values = vec![None; cols.len()];
    let mut saw_axis_alias = false;
    for (key, value) in options {
        let Some(axis) = tensor_k_axis_option_axis(key, cols, ds)? else {
            continue;
        };
        saw_axis_alias = true;
        if axis_values[axis].is_some() {
            return Err(format!("tensor k axis {axis} is specified more than once"));
        }
        let k: usize = value
            .parse()
            .map_err(|err| format!("invalid tensor k option `{key}={value}`: {err}"))?;
        axis_values[axis] = Some(k);
    }

    let raw = options
        .get("k")
        .or_else(|| options.get("basis_dim"))
        .or_else(|| options.get("basis-dim"))
        .or_else(|| options.get("basisdim"));
    if saw_axis_alias {
        if raw.is_some() {
            return Err(
                "tensor k axis aliases cannot be combined with k= or basis_dim=".to_string(),
            );
        }
        if let Some(missing_axis) = axis_values.iter().position(Option::is_none) {
            let margin_name = cols
                .get(missing_axis)
                .and_then(|col| ds.headers.get(*col))
                .map(String::as_str)
                .unwrap_or("<unnamed>");
            return Err(format!(
                "tensor k axis aliases must specify every margin; missing axis {missing_axis} ({margin_name})"
            ));
        }
        return Ok((
            axis_values
                .into_iter()
                .map(|k| k.expect("missing axis values rejected above"))
                .collect(),
            false,
        ));
    }
    let Some(raw) = raw else {
        let inferred = heuristic_tensor_margin_knots(cols, ds);
        return Ok((inferred, true));
    };
    let entries = split_list_option(raw);
    if entries.len() == 1 {
        let k: usize = entries[0]
            .parse()
            .map_err(|err| format!("invalid tensor k '{}': {err}", entries[0]))?;
        return Ok((vec![k; cols.len()], false));
    }
    if entries.len() != cols.len() {
        return Err(format!(
            "tensor k list length {} must match smooth dimension {}",
            entries.len(),
            cols.len()
        ));
    }
    let mut out = Vec::with_capacity(entries.len());
    for entry in entries {
        let k: usize = entry
            .parse()
            .map_err(|err| format!("invalid tensor k '{entry}': {err}"))?;
        out.push(k);
    }
    Ok((out, false))
}

/// Parse the `identifiability=` option for tensor-product smooths. Mirrors the
/// vocabulary of the Matern/Duchon parsers so the formula DSL is consistent.
///
/// `kind` selects the default identifiability when no explicit
/// `identifiability=` option is supplied: `te(...)` ([`SmoothKind::Te`]) keeps
/// the full-tensor sum-to-zero default, while `ti(...)` ([`SmoothKind::Ti`])
/// defaults to per-margin sum-to-zero so the marginal main effects are excluded
/// (the mgcv tensor-interaction semantics). An explicit option always wins.
fn parse_tensor_identifiability(
    options: &BTreeMap<String, String>,
    kind: SmoothKind,
) -> Result<TensorBSplineIdentifiability, String> {
    let Some(raw) = options.get("identifiability").map(String::as_str) else {
        return Ok(match kind {
            SmoothKind::Ti => TensorBSplineIdentifiability::MarginalSumToZero,
            _ => TensorBSplineIdentifiability::default(),
        });
    };
    match raw.trim().to_ascii_lowercase().as_str() {
        "none" => Ok(TensorBSplineIdentifiability::None),
        "sum_tozero" | "sum-to-zero" | "center_sum_tozero" | "center-sum-to-zero" | "centered"
        | "sumtozero" => Ok(TensorBSplineIdentifiability::SumToZero),
        "marginal_sum_tozero" | "marginal-sum-to-zero" | "marginal_sumtozero"
        | "marginalsumtozero" | "interaction" => {
            Ok(TensorBSplineIdentifiability::MarginalSumToZero)
        }
        other => Err(TermBuilderError::unsupported_feature(format!(
            "invalid tensor identifiability '{other}'; expected one of: none, sum_tozero, marginal_sum_tozero"
        ))
        .to_string()),
    }
}

/// Parse the `identifiability=` option for every 1-D B-spline family arm —
/// `s()` / `bs='ps'|'bspline'|'cr'|'cs'` and the cyclic `cc`/`cp`/`periodic`
/// selector.
///
/// Returns `Ok(None)` when the option is absent so each arm can keep applying
/// its own *structural* default: an anchored endpoint is already the model's
/// level gauge and therefore defaults to [`BSplineIdentifiability::None`],
/// while every other 1-D smooth defaults to sum-to-zero centering. An explicit
/// token always wins over the default, and an unrecognised one is refused
/// rather than silently discarded (#2783).
///
/// The vocabulary deliberately mirrors [`parse_tensor_identifiability`],
/// [`parse_matern_identifiability`] and [`parse_spatial_identifiability`] so a
/// token means the same thing on every smooth kind: `none` keeps the
/// unconstrained basis columns, the `sum_tozero` family centers, and `linear`
/// removes the constant *and* linear directions (the 1-D Greville-geometry
/// analogue of the Matérn `CenterLinearOrthogonal` policy).
///
/// [`BSplineIdentifiability::OrthogonalToDesignColumns`] and
/// [`BSplineIdentifiability::FrozenTransform`] are engine-internal: the first
/// needs a design-column block no formula can name, the second is minted by
/// design freezing at fit time. Both are refused with a message that says so,
/// exactly as `parse_spatial_identifiability` refuses `frozen`.
fn parse_bspline_identifiability(
    options: &BTreeMap<String, String>,
) -> Result<Option<BSplineIdentifiability>, String> {
    let Some(raw) = options.get("identifiability").map(String::as_str) else {
        return Ok(None);
    };
    match raw.trim().to_ascii_lowercase().as_str() {
        "none" => Ok(Some(BSplineIdentifiability::None)),
        "sum_tozero" | "sum-to-zero" | "center_sum_tozero" | "center-sum-to-zero" | "centered"
        | "sumtozero" => Ok(Some(BSplineIdentifiability::WeightedSumToZero {
            weights: None,
        })),
        "linear" | "remove_linear_trend" | "remove-linear-trend" | "removelineartrend"
        | "center_linear_orthogonal" | "center-linear-orthogonal" => {
            Ok(Some(BSplineIdentifiability::RemoveLinearTrend))
        }
        "frozen" | "frozen_transform" | "orthogonal" | "orthogonal_to_design_columns" => {
            Err(TermBuilderError::unsupported_feature(format!(
                "B-spline identifiability '{}' is internal-only (it is minted by design freezing \
                 or needs an explicit design-column block); use one of: none, sum_tozero, linear",
                raw.trim()
            ))
            .to_string())
        }
        other => Err(TermBuilderError::unsupported_feature(format!(
            "invalid B-spline identifiability '{other}'; expected one of: none, sum_tozero, linear"
        ))
        .to_string()),
    }
}

/// The structural facts about a 1-D B-spline arm that decide which
/// identifiability policies are simultaneously satisfiable with the basis the
/// arm is about to build.
#[derive(Debug, Clone, Copy, Default)]
struct BSplineIdentifiabilityContext {
    /// An endpoint is pinned to a value, so the smooth already carries its own
    /// level gauge and the global intercept is suppressed.
    has_anchor: bool,
    /// The basis wraps, so no aperiodic (constant + linear) chart applies.
    periodic: bool,
    /// The basis is a natural cubic regression spline indexed by value-at-knot
    /// (`bs="cr"`/`"cs"`), which carries no B-spline knot/degree geometry for a
    /// Greville-abscissae chart to be built from.
    natural_cubic_regression: bool,
}

/// Resolve the 1-D B-spline identifiability policy from the caller's
/// `identifiability=` token (if any) and the structural default the arm would
/// otherwise apply, refusing the combinations that are not simultaneously
/// satisfiable.
///
/// Three refusals, each of which would otherwise be a silent mis-fit rather
/// than a mere style violation:
///
/// * **anchored endpoint + a centering policy.** An anchored endpoint pins the
///   function's absolute level and suppresses the global intercept, so the
///   *fitted function* — not a centered deviation — obeys the pin. Layering
///   sum-to-zero on top demands the same function additionally have sample mean
///   zero, which excludes every non-zero-mean anchored curve from the model
///   space before REML is even evaluated (#1867, #2297). The implicit default
///   already resolves this by choosing `None`; an explicit request for the
///   incompatible policy is refused rather than quietly overridden.
/// * **periodic basis + `linear`.** [`BSplineIdentifiability::RemoveLinearTrend`]
///   builds its transform from the Greville abscissae of an *open* knot vector
///   and removes the constant and linear directions. A linear trend is not a
///   periodic function, so it is not in the span of a cyclic basis at all: the
///   constraint is ill-posed there, and the transform would in any case be
///   derived from the wrong knot geometry.
/// * **`bs="cr"`/`"cs"` + `linear`.** The natural cubic regression basis is
///   parameterized by function values at its knots, so a Greville-based linear
///   removal would be applied to the wrong coordinates.
///   [`crate::basis::build_cubic_regression_basis_1d`] refuses this too; doing
///   it here turns a fit-time basis error into a formula-time configuration
///   error naming the option the user actually wrote.
fn resolve_bspline_identifiability(
    options: &BTreeMap<String, String>,
    structural_default: BSplineIdentifiability,
    context: BSplineIdentifiabilityContext,
) -> Result<BSplineIdentifiability, String> {
    let Some(explicit) = parse_bspline_identifiability(options)? else {
        return Ok(structural_default);
    };
    if context.has_anchor && !matches!(explicit, BSplineIdentifiability::None) {
        return Err(TermBuilderError::incompatible_config(
            "an anchored endpoint already fixes the smooth's level (the global intercept is \
             suppressed), so it cannot also carry a centering identifiability constraint; \
             drop the anchor or use identifiability='none'",
        )
        .to_string());
    }
    if matches!(explicit, BSplineIdentifiability::RemoveLinearTrend) {
        if context.periodic {
            return Err(TermBuilderError::incompatible_config(
                "identifiability='linear' removes the constant and linear directions using \
                 open-knot Greville geometry, which a periodic basis does not span; use 'none' \
                 or 'sum_tozero' on a periodic smooth",
            )
            .to_string());
        }
        if context.natural_cubic_regression {
            return Err(TermBuilderError::incompatible_config(
                "identifiability='linear' needs B-spline knot/degree geometry, which the natural \
                 cubic regression basis (bs='cr'/'cs') does not carry; use 'none' or 'sum_tozero', \
                 or switch to bs='ps'",
            )
            .to_string());
        }
    }
    Ok(explicit)
}

fn bspline_boundary_declares_periodic_axis(options: &BTreeMap<String, String>) -> bool {
    options
        .get("boundary")
        .or_else(|| options.get("bc"))
        .map(|raw| {
            parse_option_list(raw)
                .into_iter()
                .any(|value| matches!(value.as_str(), "periodic" | "cyclic" | "cc"))
        })
        .unwrap_or(false)
}

/// Canonical-name lookup for the `bs=`/`type=` smooth selector.
///
/// User-facing names — including mgcv-compatible spellings whose semantics
/// match an existing gamfit smooth exactly — collapse to the engine-internal
/// canonical names used by the dispatch in [`build_smooth_basis`]. Adding a
/// new exactly-equivalent alias is a one-line entry here; the match arms
/// below remain the single dispatch site.
///
/// Aliases listed here MUST be true semantic equivalents of the canonical
/// target, not approximations. mgcv names whose semantics differ from any
/// gamfit smooth (e.g. `bs="ts"` shrinkage thin-plate, `bs="ad"` adaptive)
/// are intentionally NOT mapped here — they should reach the unsupported-type
/// path so users get a real diagnostic instead of a silent semantic
/// substitution. mgcv's `bs="cr"`/`"cs"` (cubic regression and its shrinkage
/// twin) are handled directly in the [`build_smooth_basis`] dispatch — they
/// are not aliased here because the `cr`/`cs` distinction controls a default
/// (`double_penalty`) that the canonical-name layer cannot see.
///
/// Unrecognised inputs pass through unchanged so the dispatch can produce its
/// usual "unsupported smooth type" error, preserving the existing diagnostic
/// surface for genuine typos.
pub(crate) fn canonicalize_smooth_type(raw: &str) -> &str {
    match raw {
        // Thin-plate spline. mgcv `bs="tp"` is the default thin-plate
        // regression spline — exact semantic equivalent of gamfit's `"tps"`.
        "tp" => "tps",
        // Gaussian process / Matérn. mgcv `bs="gp"` defaults to a Matérn
        // covariance kernel with REML smoothing parameter selection, which
        // matches gamfit's `"matern"` exactly (same kernel-Gram identity,
        // same REML route).
        "gp" => "matern",
        // Constant-curvature (M_κ) geodesic-kernel smooth (#944). All aliases
        // collapse to one canonical type so `bs="curv"`/`bs="mkappa"` cannot
        // diverge from `curv(...)`.
        "curv" | "constant_curvature" | "mkappa" => "curvature",
        // Measure-jet spline: multiscale local-jet-residual energy of the
        // empirical measure. No mgcv equivalent (mgcv has no measure-learned
        // geometry smooth), so no mgcv alias is mapped.
        "mjs" | "measure_jet" | "web" => "measurejet",
        other => other,
    }
}

/// Is `margin_bs` a per-margin basis name that the tensor builder realizes as a
/// penalized 1-D B-spline margin?
///
/// gam's tensor product is built from penalized B-spline marginals. mgcv's
/// thin-plate (`tp`/`tps`), P-spline (`ps`), B-spline (`bs`), cubic-regression
/// (`cr`/`cs`), and cyclic (`cc`/`cp`/`cyclic`) marginals are all penalized
/// splines spanning the same per-axis smoothing space, so a B-spline margin
/// reproduces the same tensor smoothing class. Margin kinds with fundamentally
/// different structure (adaptive, random-effect, sphere) are NOT accepted as
/// tensor margins.
pub(crate) fn tensor_margin_bs_is_supported(margin_bs: &str) -> bool {
    matches!(
        canonicalize_smooth_type(margin_bs),
        "tps" | "ps" | "bs" | "bspline" | "cr" | "cs" | "cc" | "cp" | "cyclic"
    )
}

/// Does the smooth request a periodic/cyclic axis via its options?
///
/// Mirrors the boundary-condition reading used by the periodic-aware dispatch
/// branches. Factored out so the type resolver and `build_smooth_basis` agree
/// on a single notion of "periodic requested".
pub(crate) fn smooth_options_declare_periodic(options: &BTreeMap<String, String>) -> bool {
    options.contains_key("periodic")
        || options.contains_key("cyclic")
        || options
            .get("boundary")
            .or_else(|| options.get("bc"))
            .map(|boundary| {
                boundary.to_ascii_lowercase().contains("periodic")
                    || boundary.to_ascii_lowercase().contains("cyclic")
            })
            .unwrap_or(false)
}

/// Resolve the canonical engine-internal smooth-type name for a term.
///
/// Reads the user-facing `type=`/`bs=` selector and collapses mgcv-compatible
/// aliases (`tp`→`tps`, `gp`→`matern`) via [`canonicalize_smooth_type`], or
/// derives the default from the smooth kind/arity when no selector is given.
/// This is the single source of truth for the dispatch in
/// [`build_smooth_basis`]; other call sites (e.g. predictor-specific basis
/// policy) use it so the classification never drifts from the dispatch.
/// Is the raw `bs=`/`type=` selector a vector literal (`c('tp','tp')`,
/// `['tp','tp']`, `(tp, tp)`) rather than a scalar smooth-type name?
///
/// mgcv's tensor smooths take a *per-margin* basis vector
/// (`te(x1, x2, bs=c('tp','tp'))`). Such a value is not a scalar canonical
/// type and must not be fed through [`canonicalize_smooth_type`] — it has to be
/// recognized as a tensor request and split into per-margin types. A scalar
/// selector (`bs="tp"`) is left untouched.
pub(crate) fn bs_selector_is_vector(raw: &str) -> bool {
    let trimmed = raw.trim();
    let bracketed = (trimmed.starts_with('[') && trimmed.ends_with(']'))
        || (trimmed.starts_with("c(") || trimmed.starts_with("C(")) && trimmed.ends_with(')')
        || (trimmed.starts_with('(') && trimmed.ends_with(')'));
    bracketed && !parse_option_list(trimmed).is_empty()
}

pub fn resolve_smooth_type_name(
    kind: SmoothKind,
    n_cols: usize,
    options: &BTreeMap<String, String>,
) -> String {
    let selector = options.get("type").or_else(|| options.get("bs"));
    // A per-margin basis vector is a tensor request, never a scalar type. Route
    // it to the tensor builder, which reads the per-margin types out of the
    // same `bs=` option. (A vector on a non-tensor smooth is ill-formed and
    // falls through to the scalar path below so the existing diagnostic fires.)
    if let Some(raw) = selector
        && bs_selector_is_vector(raw)
        && matches!(kind, SmoothKind::Te | SmoothKind::Ti | SmoothKind::T2)
    {
        return "tensor".to_string();
    }
    selector
        .map(|s| canonicalize_smooth_type(&s.to_ascii_lowercase()).to_string())
        .unwrap_or_else(|| match kind {
            SmoothKind::Te | SmoothKind::Ti | SmoothKind::T2 => "tensor".to_string(),
            SmoothKind::S if n_cols == 1 => "bspline".to_string(),
            // Mixed periodic Euclidean radial kernels are not separable on the
            // cylinder. Use a tensor product with a cyclic margin so s(theta,h)
            // honors seam continuity while preserving the formula-level s(...).
            SmoothKind::S if smooth_options_declare_periodic(options) => "tensor".to_string(),
            SmoothKind::S => "tps".to_string(),
        })
}

/// Does this canonical smooth type size its basis through the generous spatial
/// center heuristic ([`crate::basis::default_num_centers`])?
///
/// Only the radial spatial bases (thin-plate, Matérn/GP, Duchon) route their
/// default basis dimension through `plan_spatial_basis(.., Default, ..)`. The
/// B-spline, cyclic, tensor, and factor-smooth bases use their own modest
/// knot-based defaults, so they are unaffected by — and must not be perturbed
/// by — secondary-predictor basis-parsimony adjustments (#501).
pub fn smooth_type_uses_spatial_center_heuristic(canonical_type: &str) -> bool {
    matches!(canonical_type, "tps" | "matern" | "duchon")
}

pub fn build_smooth_basis(
    kind: SmoothKind,
    vars: &[String],
    cols: &[usize],
    options: &BTreeMap<String, String>,
    ds: &Dataset,
    inference_notes: &mut Vec<String>,
    policy: &ResourcePolicy,
    smooth_coordinate_count: usize,
) -> Result<SmoothBasisSpec, String> {
    // Strip the internal by-level sizing carrier before any per-kind option
    // allow-list runs (the `__by_col` pattern): `sizing_rows` feeds every
    // n-scaling BASIS DEFAULT below; explicit user counts are untouched.
    let stripped_sizing_options;
    let (options, sizing_rows) = match options.get(DEFAULT_SIZING_ROWS_OPTION) {
        Some(raw) => {
            let rows = raw.parse::<usize>().map_err(|_| {
                format!("internal by-level sizing rows carrier is not a count: '{raw}'")
            })?;
            let mut cleaned = options.clone();
            cleaned.remove(DEFAULT_SIZING_ROWS_OPTION);
            stripped_sizing_options = cleaned;
            (&stripped_sizing_options, rows)
        }
        None => (options, ds.values.nrows()),
    };
    // Fail fast on degenerate input: a smooth whose (non-categorical) coordinate
    // columns collapse to a SINGLE distinct point can only ever fit the response
    // mean — its design matrix is rank-1. For a UNIVARIATE smooth this is exactly
    // "the one column is constant": `smooth(x)`/`matern(x)` on constant `x` would
    // otherwise silently fit the mean of `y` with no visible cue (Duchon already
    // errors loudly via the basis layer; this makes the diagnosis explicit and
    // uniform). For a general MULTIVARIATE Euclidean smooth (tensor, tps,
    // matern, ...) a single constant coordinate is NOT degenerate — the basis
    // still varies along the other coordinate(s) and the penalty absorbs the
    // rank-deficient direction (a constant-`x2` slice of `tps(x1, x2)` is a
    // well-posed 1-D function of `x1`). Such a term is degenerate only when
    // EVERY coordinate is constant at once, i.e. the joint input is a single
    // point. Test the JOINT cardinality, not each column independently, so the
    // loud diagnosis still fires for the genuinely rank-1 case without rejecting
    // well-posed lower-dimensional slices.
    //
    // The SPHERE/SOS term is the exception (handled separately just below): its
    // spherical-harmonic / Wahba basis is intrinsically a function of BOTH
    // angular coordinates, so a constant latitude or longitude is not an honest
    // lower-D slice but an unidentifiable axis (every point on a single meridian
    // or parallel) — that case is rejected per-coordinate.
    let coord_cols: Vec<(&String, usize)> = vars
        .iter()
        .zip(cols.iter().copied())
        .filter(|(_, col)| !matches!(ds.column_kinds.get(*col), Some(ColumnKindTag::Categorical)))
        .collect();
    if !coord_cols.is_empty() {
        let views: Vec<ArrayView1<'_, f64>> = coord_cols
            .iter()
            .map(|(_, col)| ds.values.column(*col))
            .collect();
        let n_rows = views[0].len();
        let mut distinct_points = std::collections::HashSet::<Vec<u64>>::new();
        for r in 0..n_rows {
            let key: Vec<u64> = views
                .iter()
                .map(|v| gam_data::canonical_level_bits(v[r]))
                .collect();
            distinct_points.insert(key);
            if distinct_points.len() > 1 {
                break;
            }
        }
        if distinct_points.len() <= 1 {
            return Err(TermBuilderError::degenerate_data(if coord_cols.len() == 1 {
                let var = coord_cols[0].0;
                format!(
                    "smooth term over '{var}' has only one unique value in the training data \
                     — a smooth on a constant column is degenerate and would only fit the response mean. \
                     Remove `{var}` from the smooth, drop the term, or check the data."
                )
            } else {
                let names = coord_cols
                    .iter()
                    .map(|(v, _)| v.as_str())
                    .collect::<Vec<_>>()
                    .join(", ");
                format!(
                    "smooth term over ({names}) has only one unique joint coordinate in the training \
                     data — every coordinate is constant, so the smooth is degenerate and would only \
                     fit the response mean. Drop the term or check the data."
                )
            })
            .to_string());
        }

        // Sphere/SOS exception: the S² smooth is intrinsically a function of
        // BOTH angular coordinates, so a single constant axis is unidentifiable
        // (every point on one meridian or one parallel), not an honest 1-D
        // slice. Reject it per-coordinate at fit-time with a coordinate-named
        // error. This runs ONLY during term construction (build_smooth_basis);
        // predict rebuilds the design from the frozen resolvedspec and never
        // re-enters this path, so a constant predict grid (e.g. a single query
        // point on a fixed meridian) is never re-validated (#frozen-mass).
        if matches!(
            resolve_smooth_type_name(kind, cols.len(), options).as_str(),
            "sphere" | "s2" | "sos"
        ) {
            for (axis, (var, col)) in coord_cols.iter().enumerate() {
                let column = ds.values.column(*col);
                let mut distinct = std::collections::HashSet::<u64>::new();
                for &value in column.iter() {
                    distinct.insert(gam_data::canonical_level_bits(value));
                    if distinct.len() > 1 {
                        break;
                    }
                }
                if distinct.len() <= 1 {
                    // Axis 0 is latitude, axis 1 longitude (formula order
                    // `sphere(lat, lon)`); name the collapsed slice accordingly.
                    let slice = if axis == 0 {
                        "a single parallel (constant latitude)"
                    } else {
                        "a single meridian (constant longitude)"
                    };
                    return Err(TermBuilderError::degenerate_data(format!(
                        "sphere smooth has a constant '{var}' column — every point lies on \
                         {slice}, so the 2-sphere term is degenerate and unidentifiable along \
                         that axis. A spherical smooth needs genuine variation in BOTH latitude \
                         and longitude; vary '{var}', drop the term, or fit a 1-D smooth on the \
                         varying coordinate."
                    ))
                    .to_string());
                }
            }
        }
    }
    if let Some(by_name) = options.get("by").cloned() {
        let by_col = options
            .get("__by_col")
            .and_then(|raw| raw.parse::<usize>().ok())
            .or_else(|| vars.iter().position(|v| v == &by_name).map(|idx| cols[idx]))
            .ok_or_else(|| format!("unknown by= column '{by_name}'"))?;
        let mut inner_options = options.clone();
        inner_options.remove("by");
        inner_options.remove("__by_col");
        inner_options.remove("id");
        // Size the inner basis's n-scaling defaults from the smallest
        // by-level's rows (see `DEFAULT_SIZING_ROWS_OPTION`); numeric-by
        // smooths keep pooled sizing.
        inject_by_level_sizing_rows(&mut inner_options, ds, by_col);
        let inner = build_smooth_basis(
            kind,
            vars,
            cols,
            &inner_options,
            ds,
            inference_notes,
            policy,
            smooth_coordinate_count,
        )?;
        let by_kind = match ds.column_kinds.get(by_col).copied() {
            Some(ColumnKindTag::Categorical) => ByVarKind::Factor {
                feature_col: by_col,
                ordered: option_bool(options, "ordered").unwrap_or(false),
                frozen_levels: None,
            },
            Some(ColumnKindTag::Continuous | ColumnKindTag::Binary) => ByVarKind::Numeric {
                feature_col: by_col,
            },
            None => {
                return Err(format!(
                    "internal column-kind lookup failed for by='{by_name}'"
                ));
            }
        };
        return Ok(SmoothBasisSpec::BySmooth {
            smooth: Box::new(inner),
            by_kind,
        });
    }

    let smooth_double_penalty = option_bool(options, "double_penalty").unwrap_or(true);
    let type_opt = resolve_smooth_type_name(kind, cols.len(), options);

    if matches!(type_opt.as_str(), "fs" | "sz" | "re") {
        if type_opt == "re" {
            validate_random_effect_smooth_options(options)?;
        } else {
            validate_known_options(type_opt.as_str(), options, FACTOR_SMOOTH_OPTION_KEYS)?;
        }
        if cols.len() != 2 {
            return Err(format!(
                "{} factor-smooth currently expects exactly two variables (one numeric, one categorical)",
                type_opt
            ));
        }
        let kinds = cols
            .iter()
            .map(|&c| ds.column_kinds.get(c).copied())
            .collect::<Vec<_>>();
        let (cont_idx, group_idx) = if type_opt == "re" {
            // mgcv random-slope examples are often s(g, x, bs="re").
            match (kinds[0], kinds[1]) {
                (Some(ColumnKindTag::Categorical), _) => (1usize, 0usize),
                (_, Some(ColumnKindTag::Categorical)) => (0usize, 1usize),
                _ => (1usize, 0usize),
            }
        } else {
            match (kinds[0], kinds[1]) {
                (_, Some(ColumnKindTag::Categorical)) => (0usize, 1usize),
                (Some(ColumnKindTag::Categorical), _) => (1usize, 0usize),
                _ => {
                    return Err(format!(
                        "{} factor-smooth requires one categorical factor variable",
                        type_opt
                    ));
                }
            }
        };
        let c = cols[cont_idx];
        let (minv, maxv) = col_minmax(ds.values.column(c))?;
        let degree = if type_opt == "re" {
            1
        } else {
            option_usize(options, "degree").unwrap_or(DEFAULT_BSPLINE_DEGREE)
        };
        // For a factor smooth every group's curve is fit from THAT group's rows
        // alone, so the marginal's flexibility must respect the least-resolved
        // group, not the pooled column. The pooled heuristic can hand the marginal
        // a basis that saturates (or exceeds) a small group's sample — e.g. the
        // sleepstudy panel has 8 training days per subject, and a default cubic
        // basis of 8 functions interpolates each subject's 8 points, leaving no
        // room for the wiggliness penalty to collapse the curve toward the
        // per-subject line. The factor smooth then fits within-group noise and
        // extrapolates badly (held-out forecast worse than the population mean).
        //
        // Cap the marginal basis below the minimum per-group covariate resolution
        // so the penalty always retains residual degrees of freedom to shrink each
        // group's curvature toward its linear null space (the random-slope
        // estimand). This small-group cap composes with a separate upper bound at
        // mgcv's factor-smooth default k=10 (FACTOR_SMOOTH_DEFAULT_BASIS_DIM,
        // applied below), so even ample-data groups get the modest SHARED marginal
        // a factor smooth wants rather than the full pooled basis. The explicit
        // `re` random-effect form takes neither cap: it is a raw linear `[1, x]`
        // random effect (0 internal knots), handled in the branch above.
        let pooled_internal = heuristic_knots_for_column(ds.values.column(c));
        let default_internal = if type_opt == "re" {
            // `bs="re"` is a PARAMETRIC random effect, not a smooth of the
            // covariate: `s(x, g, bs="re")` is the mgcv random intercept+slope
            // `(1 + x | g)`, i.e. a per-group line `[1, x]`, penalized by an iid
            // ridge. A degree-1 marginal with ZERO internal knots spans exactly
            // that linear space (2 coefficients per group). Using the pooled
            // knot heuristic here instead turned the marginal into a
            // piecewise-linear B-spline (e.g. 6 functions/group on sleepstudy),
            // i.e. a *smooth* with kinks rather than a random slope — many extra
            // collinear-across-levels coefficients that ill-condition the joint
            // Newton/REML solve (minutes-long fits, and a singular block when
            // combined with a separate random intercept `s(g, bs="re")`). The
            // raw linear basis is both the correct `re` semantics and fast.
            0
        } else {
            let min_group_resolution =
                min_per_group_unique_count(ds.values.column(c), ds.values.column(cols[group_idx]));
            // Per-group basis dim = degree + 1 + internal. Hold it well below the
            // smallest group's resolution (leave at least two residual points per
            // group) so the smooth cannot interpolate that group and the
            // wiggliness penalty retains the room to collapse each curve toward
            // its linear null space. Never drop below `degree + 2`, which keeps
            // exactly the linear span plus a single curvature direction — the
            // minimal smoother that can still bend if the data demand it.
            let basis_cap = min_group_resolution.saturating_sub(2).max(degree + 2);
            let internal_cap = basis_cap.saturating_sub(degree + 1);
            let capped = pooled_internal.min(internal_cap.max(1));
            // A factor smooth (`fs` AND `sz`) shares ONE marginal across ALL
            // levels, each level's curve fit from that group's rows alone. The
            // pooled knot heuristic (driven by the full column's sample) hands it
            // a much richer basis than the shared signal needs — ~24
            // functions/group on the gam#903 factor-smooth-recovery fixtures — so
            // REML has the capacity to fit within-group noise and over-fits the
            // shared shape (fs: edf 58 vs mgcv's k=10/edf 39; sz: gam 0.068 vs
            // mgcv 0.046 truth RMSE), losing the truth-recovery head-to-head with
            // the mature tool. mgcv's factor-smooth default `k=10` embodies the
            // right convention: a modest shared marginal. Cap the marginal there
            // (basis ≈ degree+1+internal ≈ 10) for both flavours when the
            // small-group cap above is not already tighter, so REML is not handed
            // noise-fitting capacity it does not need. An explicit `k`/`basis_dim`
            // overrides this (parse_ps_internal_knots); `re` is the raw linear
            // effect handled above.
            let fs_default_internal = FACTOR_SMOOTH_DEFAULT_BASIS_DIM
                .saturating_sub(degree + 1)
                .max(1);
            capped.min(fs_default_internal)
        };
        let (n_knots, _, effective_degree) =
            parse_ps_internal_knots(options, degree, default_internal)?;
        // `m=` is mgcv's spelling of `penalty_order=` and is resolved as its
        // alias here (#2791). It used to be read further down as a boolean gate
        // on the `Fs` null-penalty path instead, which made every value >= 1 the
        // same model and dropped the key entirely on `sz`.
        let penalty_order = parse_penalty_order_alias(options)?
            .unwrap_or(if effective_degree > 1 { 2 } else { 1 })
            .min(effective_degree);
        // All factor-smooth flavours (`fs`, `sz`, `re`) place their per-level
        // marginal on the SAME penalized B-spline (P-spline) basis. The flavours
        // differ ONLY in their penalty/constraint structure (handled below) —
        // sz: zero-sum deviation blocks with the per-level null space left
        // unpenalized; fs: random-effect double penalty; re: identity ridge.
        //
        // `sz` USED to route its default-degree marginal to a NATURAL cubic
        // regression spline (`cr`), on the belief that mgcv's `bs="sz"` does the
        // same and that cr recovers smooth signals more efficiently than the
        // (then uncapped) B-spline margin (#1074). That introduced a consistency
        // failure (#1605): the `cr` basis enforces the natural boundary
        // conditions f''(x_1)=f''(x_k)=0 and extrapolates linearly past the end
        // knots, so it CANNOT represent a per-group deviation curve with non-zero
        // curvature at the data boundary. Phase-shifted deviation shapes
        // (f''(0) = -(2π)² sin(φ) ≠ 0) are then biased toward "free linear +
        // anchored wiggle", under-shooting the amplitude — a bias that does NOT
        // vanish as n→∞ (n-independent: a genuine consistency failure, not
        // finite-sample shrinkage). The earlier #700/#1074 sz fixtures used
        // d_g ∝ sin(2πx), whose f'' happens to vanish at x=0 and x=1, so they
        // accidentally satisfied the natural BC and never exposed the gap; the
        // `fs` sibling, on this very B-spline marginal, recovers the SAME
        // phase-shifted data to the noise floor.
        //
        // The penalized B-spline marginal makes no boundary assumption, so it
        // represents arbitrary deviation shapes, and — with the
        // FACTOR_SMOOTH_DEFAULT_BASIS_DIM cap above already removing the
        // noise-fitting capacity that originally motivated leaving B-splines —
        // it recovers the BC-satisfying #700/#1074 signals just as well. Sharing
        // one marginal basis across all flavours also lets the B-spline degree/
        // knot degradation handle low-cardinality covariates uniformly (what
        // `fs` already does), so the `sz`-only cr data-support cap (#1541/#1542)
        // — and the asymmetry where only the cr-marginal `sz` spelling hard-
        // failed a 3-level ordinal — is no longer needed.
        let marginal_knotspec = resolve_nonperiodic_bspline_knotspec(
            options,
            ds.values.column(c),
            (minv, maxv),
            effective_degree,
            n_knots,
        )?;
        let marginal = BSplineBasisSpec {
            degree: effective_degree,
            penalty_order,
            knotspec: marginal_knotspec,
            // mgcv's `bs="fs"` is a random-effect-style smooth: EVERY per-level
            // coefficient, including the marginal null space, is penalized so
            // unobserved groups can be predicted — so `fs` keeps the null-space
            // (double) penalty. mgcv's `bs="sz"` is a pure across-level
            // *deviation* smooth that, under the default `select=FALSE`, leaves
            // the per-level null space UNPENALIZED; carrying the double penalty
            // there shrinks the genuine deviation signal and over-smooths the
            // recovered curves relative to mgcv (gam#700). `re` carries its own
            // identity ridge below and ignores this flag. Honour an explicit
            // user `double_penalty=` either way.
            double_penalty: option_bool(options, "double_penalty")
                .unwrap_or(type_opt.as_str() != "sz"),
            identifiability: BSplineIdentifiability::None,
            boundary_conditions: Default::default(),
            boundary: OneDimensionalBoundary::Open,
        };
        let flavour = match type_opt.as_str() {
            "fs" => FactorSmoothFlavour::Fs {},
            "sz" => FactorSmoothFlavour::Sz,
            "re" => FactorSmoothFlavour::Re,
            // Outer `matches!` already restricts to fs/sz/re.
            other => {
                return Err(format!(
                    "internal: factor-smooth flavour dispatch reached unexpected type `{}`",
                    other
                ));
            }
        };
        return Ok(SmoothBasisSpec::FactorSmooth {
            spec: FactorSmoothSpec {
                continuous_cols: vec![c],
                group_col: cols[group_idx],
                marginal,
                flavour,
                group_frozen_levels: None,
                frozen_global_orthogonality: None,
            },
        });
    }

    match type_opt.as_str() {
        // `periodic` is the generic spelling for a periodic (wrap-continuous)
        // B-spline; it names the SAME `SmoothBasisSpec::BSpline1D {
        // PeriodicUniform }` the mgcv-style cyclic selectors (`cc`/`cp`/`cyclic`)
        // build, and is already recognized as that basis kind by the JSON /
        // override path (`smooth_overrides`) and accepted by the formula parser.
        // Route it through the cyclic arm so the formula path agrees with the
        // rest of the codebase instead of rejecting it as an unsupported type.
        "cyclic" | "cc" | "cp" | "cyclic-ps" | "periodic" => {
            validate_known_options("cyclic", options, CYCLIC_SMOOTH_OPTION_KEYS)?;
            if cols.len() != 1 {
                return Err(format!(
                    "periodic smooth expects one variable, got {}",
                    cols.len()
                ));
            }
            let c = cols[0];
            let (minv, maxv) = col_minmax(ds.values.column(c))?;
            let degree = option_usize(options, "degree").unwrap_or(DEFAULT_BSPLINE_DEGREE);
            let mut default_internal = heuristic_knots_for_column(ds.values.column(c));
            if ds.values.nrows() <= 32 && smooth_coordinate_count >= 5 {
                default_internal = default_internal.min(1);
            }
            // A periodic cubic spline has no free endpoint behaviour to spend
            // degrees of freedom on: the wrap constraint removes the ordinary
            // boundary wiggle, and the cyclic second-difference penalty leaves
            // only the constant direction (handled by the smooth
            // identifiability constraint).  An over-rich default would give
            // small binomial/continuation-ratio fits a large penalized nuisance
            // space whose REML/LAML optimum is driven by finite-sample Bernoulli
            // noise rather than the low-frequency periodic signal.  Cap the
            // cyclic default in the mgcv `bs="cc"` spirit: a modest basis unless
            // the caller explicitly requests `k=...`; high-frequency periodic
            // structure remains available through that explicit contract.  Since
            // gam#1680 lowered the open-spline univariate default to ≈12
            // functions this cap and the open-spline default coincide, so it now
            // acts as an explicit floor/guard that keeps the cyclic default lean
            // even if the open-spline heuristic is later widened.
            let cyclic_default_basis_cap = CYCLIC_DEFAULT_BASIS_DIM.max(degree + 1);
            let default_basis = (default_internal + degree + 1).min(cyclic_default_basis_cap);
            let num_basis = option_usize_any(options, &["k", "basis_dim", "basis-dim", "basisdim"])
                .unwrap_or(default_basis);
            if num_basis < degree + 1 {
                return Err(format!(
                    "periodic smooth: k={} too small for degree {}; expected k >= {}",
                    num_basis,
                    degree,
                    degree + 1
                ));
            }
            // The cyclic arm is periodic on its single axis by construction, so
            // resolve the period exactly the way the `s()`/`ps` arm does: honour
            // `period=`/`periods=` first (with `origin=` setting the domain
            // start), and fall back to the `period_start`/`period_end` endpoint
            // form only when `period=` is absent. Previously this arm jumped
            // straight to `parse_periodic_domain_1d`, so a `period=<v>`
            // declaration was silently dropped and the smooth wrapped at the
            // data range (#816). All three helpers route through
            // `parse_numeric_expr`, so `period=2*pi` and `period_end=2*pi` parse
            // identically (#815).
            let periodic_axes = [true];
            let periods = parse_periods(options, &periodic_axes)?;
            let origins = parse_period_origins(options, &periodic_axes)?;
            // Distinguish a *cyclic basis selector* (`bs='cc'`/`cp'`/`cyclic`,
            // this whole arm) from a generic B-spline forced periodic by a
            // `periodic=`/`boundary=` flag (the `ps`/`bspline` arm). Only the
            // latter carries the sample-dependent off-by-ε seam that #1771's
            // guard in `parse_periodic_domain_1d` requires an explicit period
            // to avoid. A bare `s(x, bs='cc')` opts INTO mgcv's `bs="cc"`
            // semantics — the wrap IS the observed data range — exactly like
            // the tensor cc-margin fallback (`te(x, z, bs=c('cc','cc'))`). The
            // cyclic arm was left routing through the now-strict helper when
            // #1771 tightened it, so a bare cyclic smooth hard-errored with
            // "periodic B-spline smooth requires an explicit period" even
            // though its period is well-defined. Honor `period=`/`periods=`
            // first, then the half-open `period_start`/`period_end` endpoint
            // form, and only otherwise wrap at the observed `[min, max]` span.
            let has_endpoint_decl = ["period_start", "start", "period_end", "end"]
                .iter()
                .any(|key| options.contains_key(*key));
            let (domain_start, period) = if let Some(p) = periods[0] {
                (origins[0].unwrap_or(minv), p)
            } else if has_endpoint_decl {
                parse_periodic_domain_1d(options, minv, maxv)?
            } else {
                let span = maxv - minv;
                if !(span.is_finite() && span > 0.0) {
                    return Err(format!(
                        "cyclic smooth requires a positive observed data range to derive \
                         its period, got [{minv}, {maxv}]"
                    ));
                }
                (origins[0].unwrap_or(minv), span)
            };
            // This arm is periodic by construction, so its structural default is
            // the ordinary sum-to-zero centering (the cyclic penalty leaves the
            // constant direction unpenalized). An explicit `identifiability=`
            // token overrides it; before #2783 the option was whitelisted here
            // and then hardcoded away, so `cyclic(x, identifiability='none')`
            // was bit-identical to the centered default.
            let identifiability = resolve_bspline_identifiability(
                options,
                BSplineIdentifiability::default(),
                BSplineIdentifiabilityContext {
                    periodic: true,
                    ..Default::default()
                },
            )?;
            Ok(SmoothBasisSpec::BSpline1D {
                feature_col: c,
                spec: BSplineBasisSpec {
                    degree,
                    penalty_order: option_usize(options, "penalty_order")
                        .unwrap_or(DEFAULT_PENALTY_ORDER),
                    knotspec: BSplineKnotSpec::PeriodicUniform {
                        data_range: (domain_start, domain_start + period),
                        num_basis,
                    },
                    double_penalty: smooth_double_penalty,
                    identifiability,
                    boundary_conditions: Default::default(),
                    boundary: OneDimensionalBoundary::Cyclic {
                        start: domain_start,
                        end: domain_start + period,
                    },
                },
            })
        }
        "bspline" | "ps" | "p-spline" | "cr" | "cs" => {
            // mgcv's `bs="cr"` (cubic regression spline) and `bs="cs"` (its
            // shrinkage twin) are penalized cubic-regression smooths that span
            // the same per-axis function space as gamfit's `bspline` (cubic
            // B-spline, second-derivative penalty). Route both through the
            // 1-D B-spline arm. Both recover unsupported null-space effects by
            // default; `double_penalty=false` is the explicit unpenalized
            // opt-out. Without this route, a stand-alone
            // `s(x, bs='cr')` (which is otherwise a routine 1-D smooth in
            // mgcv-compatible formulae) reached the dispatch's default arm
            // and aborted the whole fit with `unsupported smooth type 'cr'`,
            // even though the same name was already recognized as a tensor
            // margin (`tensor_margin_bs_is_supported`).
            let validation_name = match type_opt.as_str() {
                "cr" => "cr",
                "cs" => "cs",
                _ => "bspline",
            };
            validate_known_options(validation_name, options, BSPLINE_SMOOTH_OPTION_KEYS)?;
            if cols.len() != 1 {
                return Err(TermBuilderError::incompatible_config(format!(
                    "bspline smooth expects one variable, got {}",
                    cols.len()
                ))
                .to_string());
            }
            let c = cols[0];
            let (minv, maxv) = col_minmax(ds.values.column(c))?;
            let degree = option_usize(options, "degree").unwrap_or(DEFAULT_BSPLINE_DEGREE);
            let default_internal = heuristic_knots_for_column(ds.values.column(c));
            let (mut n_knots, inferred, effective_degree) =
                parse_ps_internal_knots(options, degree, default_internal)?;
            let periodic_axes = parse_periodic_axes(options, 1).map_err(|e| e.to_string())?;
            // Every period/origin declaration this arm accepts is read only
            // inside the `periodic_axes[0]` branch below, so one that leaves the
            // axis aperiodic would be silently discarded (#2781).
            reject_unconsumable_period_declaration(validation_name, options, &periodic_axes)?;
            // Periodic margins still need enough basis functions to wrap, so
            // surface the per-axis degree reduction as a config error when the
            // user explicitly asked for a periodic-but-too-small basis. The
            // non-periodic path silently degrades degree to match mgcv.
            if periodic_axes[0] && effective_degree != degree {
                return Err(TermBuilderError::invalid_option(format!(
                    "periodic smooth: k={} too small for degree {}; expected k >= {}",
                    effective_degree + 1,
                    degree,
                    degree + 1
                ))
                .to_string());
            }
            let heuristic_knots = n_knots;
            if inferred && ds.values.nrows() <= 32 && smooth_coordinate_count >= 5 {
                n_knots = n_knots.min(1);
            }
            if inferred {
                let unique = unique_count_column(ds.values.column(c));
                // State the rule the engine actually applied
                // (`heuristic_knots_for_column`: `clamp(unique/4, 4..8)`), and
                // the small-data reduction when it fired. The note used to
                // announce a `max(20, cbrt(unique))` ceiling that no code path
                // computed, so for every column with 36 or more unique values
                // it printed a rule whose own arithmetic disagreed with the
                // count beside it.
                let mut note = format!(
                    "Automatically set {} internal knots for smooth '{}' from {} unique values (rule: clamp(unique/4, 4..{}) = {}; basis dimension = internal knots + degree + 1).",
                    n_knots,
                    vars.join(","),
                    unique,
                    MAX_DEFAULT_INTERNAL_KNOTS,
                    heuristic_knots,
                );
                if n_knots != heuristic_knots {
                    note.push_str(&format!(
                        " Reduced to {} because the fit has only {} rows and {} smooth coordinates.",
                        n_knots,
                        ds.values.nrows(),
                        smooth_coordinate_count,
                    ));
                }
                note.push_str(" Override with knots=... or k=....");
                inference_notes.push(note);
            }
            let boundary_conditions =
                if periodic_axes[0] && bspline_boundary_declares_periodic_axis(options) {
                    BSplineBoundaryConditions::default()
                } else {
                    parse_bspline_boundary_conditions(options).map_err(|e| e.to_string())?
                };
            // An anchored endpoint (one *or* both sides) is already the model's
            // level-setting gauge: term-design construction suppresses the
            // global intercept so the fitted function itself, rather than only a
            // centered deviation, obeys the endpoint pin. Applying the ordinary
            // sum-to-zero chart as well would force the entire anchored function
            // to have sample mean zero. In #1867 that made a positive one-sided
            // anchored bump mathematically unrecoverable before REML was even
            // evaluated; for a two-sided anchor it additionally strips the
            // interior level the two pins bracket (#2297).
            let structural_identifiability = if boundary_conditions.has_anchor() {
                BSplineIdentifiability::None
            } else {
                BSplineIdentifiability::default()
            };
            // An explicit `identifiability=` token overrides that structural
            // default, and an unrecognised one is refused. Before #2783 the
            // option was whitelisted by `validate_known_options` above and then
            // never read, so every value — including nonsense — was accepted
            // and inert on this arm alone.
            let identifiability = resolve_bspline_identifiability(
                options,
                structural_identifiability,
                BSplineIdentifiabilityContext {
                    has_anchor: boundary_conditions.has_anchor(),
                    periodic: periodic_axes[0],
                    natural_cubic_regression: !periodic_axes[0]
                        && (type_opt == "cr" || type_opt == "cs"),
                },
            )?;
            let periods = parse_periods(options, &periodic_axes).map_err(|e| e.to_string())?;
            let origins =
                parse_period_origins(options, &periodic_axes).map_err(|e| e.to_string())?;
            let (knotspec, boundary) = if periodic_axes[0] {
                if !boundary_conditions.is_free() {
                    return Err(TermBuilderError::incompatible_config(
                        "periodic B-splines cannot also declare endpoint boundary conditions",
                    )
                    .to_string());
                }
                {
                    let (domain_start, p_value) = if let Some(period) = periods[0] {
                        (origins[0].unwrap_or(minv), period)
                    } else {
                        parse_periodic_domain_1d(options, minv, maxv).map_err(|e| e.to_string())?
                    };
                    let domain_end = domain_start + p_value;
                    (
                        BSplineKnotSpec::PeriodicUniform {
                            data_range: (domain_start, domain_end),
                            num_basis: n_knots + effective_degree + 1,
                        },
                        OneDimensionalBoundary::Cyclic {
                            start: domain_start,
                            end: domain_end,
                        },
                    )
                }
            } else if type_opt == "cr" || type_opt == "cs" {
                // mgcv `bs="cr"`/`"cs"`: a natural cubic regression spline whose
                // basis is indexed by `k` values at quantile-placed knots (#1074),
                // NOT a B-spline knot vector. Match gam's `k=` convention by
                // requesting the same total basis size the B-spline arm would
                // produce (`n_knots` internal + degree + 1), floored at the cr
                // minimum of 3 knots. `cr` vs `cs` (shrinkage) is carried by the
                // `double_penalty` flag resolved below, which the cr builder reads.
                //
                // Cap that request to the covariate's data support (#1541): a cr
                // basis cannot place more value-knots than there are distinct
                // covariate values, so an unclamped `k` on a low-cardinality
                // predictor (binary indicator, 3-level ordinal, small count) used
                // to hard-fail in `select_cr_knots` instead of reducing like mgcv
                // and gam's tensor path. Below the cr minimum (a binary covariate)
                // degrade to the B-spline marginal the default `s(x, k=..)` basis
                // already fits on the same data — never a hard error.
                let k_cr = (n_knots + effective_degree + 1).max(CR_MIN_KNOTS);
                let knotspec = match capped_cr_marginal_knotspec(
                    ds.values.column(c),
                    k_cr,
                    &vars.join(","),
                    inference_notes,
                )? {
                    Some(cr_knotspec) => cr_knotspec,
                    None => resolve_nonperiodic_bspline_knotspec(
                        options,
                        ds.values.column(c),
                        (minv, maxv),
                        effective_degree,
                        n_knots,
                    )?,
                };
                (knotspec, parse_cyclic_boundary(options, minv, maxv)?)
            } else {
                (
                    resolve_nonperiodic_bspline_knotspec(
                        options,
                        ds.values.column(c),
                        (minv, maxv),
                        effective_degree,
                        n_knots,
                    )?,
                    parse_cyclic_boundary(options, minv, maxv)?,
                )
            };
            // Both cubic-regression spellings recover unsupported null-space
            // effects by default. An explicit `double_penalty=false` is the
            // MLE-style opt-out.
            let double_penalty = smooth_double_penalty;
            // Clamp the marginal difference penalty to `<= effective_degree`
            // so it stays well-defined when the per-axis degree was reduced
            // (mirrors the tensor margin path: `create_difference_penalty_matrix`
            // requires order < num_basis_functions).
            let penalty_order = option_usize(options, "penalty_order")
                .unwrap_or(DEFAULT_PENALTY_ORDER)
                .min(effective_degree);
            Ok(SmoothBasisSpec::BSpline1D {
                feature_col: c,
                spec: BSplineBasisSpec {
                    degree: effective_degree,
                    penalty_order,
                    knotspec,
                    double_penalty,
                    identifiability,
                    boundary,
                    boundary_conditions,
                },
            })
        }
        "tps" | "thinplate" | "thin-plate" => {
            validate_known_options("thinplate", options, THINPLATE_SMOOTH_OPTION_KEYS)?;
            let plan = plan_spatial_basis(
                sizing_rows,
                cols.len(),
                CenterCountRequest::Default,
                DuchonNullspaceOrder::Linear,
                option_bool(options, "scale_dims").unwrap_or(false),
                policy,
            )
            .map_err(|e| e.to_string())?;
            // #1074: the mgcv-sized basis cap (`k = 10·3^(d-1)`) that used to live
            // here was DELETED. It masked the real defect — the n-scaling default
            // over-sizes a thin-plate field, producing a weakly-identified
            // two-penalty ρ-surface the outer optimizer stalls on (row-order
            // dependent, #1378), and surplus columns REML can't penalize away on
            // weak-signal fits. Capping the basis hid that stall instead of fixing
            // it. The default now uses the generic spatial center heuristic; the
            // root fix (a well-identified ρ-surface / optimizer that doesn't stall)
            // is tracked separately. Explicit `k`/`centers` still take full effect.
            let default_centers = plan.centers;
            let centers = parse_countwith_basis_alias(
                options,
                "centers",
                cap_default_spatial_centers(options, default_centers),
            )?;
            let center_strategy = if has_explicit_countwith_basis_alias(options, "centers") {
                spatial_center_strategy_for_dimension(centers, cols.len())
            } else {
                auto_spatial_center_strategy(centers, cols.len())
            };
            // `include_intercept` appends a constant column to a KERNEL basis
            // that has no polynomial null space of its own — which is exactly
            // what the Matérn basis is, and exactly what the thin-plate basis is
            // not: a TPS ships its polynomial null space (constant and linear)
            // by construction, sized by
            // `thin_plate_polynomial_basis_dimension`. An appended constant
            // would therefore be exactly collinear with a column already in the
            // span. The option was whitelisted here anyway and read by nothing,
            // so it was accepted and silently discarded (#2781's family).
            if options.contains_key("include_intercept") {
                return Err(TermBuilderError::unsupported_feature(
                    "thinplate() does not support include_intercept: the thin-plate basis already                      spans its polynomial null space (the constant and linear terms), so an                      appended constant column would be exactly collinear with it. matern() takes                      the option because its kernel basis carries no polynomial null space.",
                )
                .to_string());
            }
            let periodic = parse_periodic_axes_option(options, cols.len())?;
            reject_unconsumable_radial_period_declaration(
                "thinplate",
                options,
                cols.len(),
                periodic.as_deref(),
                false,
            )?;
            Ok(SmoothBasisSpec::ThinPlate {
                feature_cols: cols.to_vec(),
                spec: ThinPlateBasisSpec {
                    center_strategy,
                    periodic,
                    // Sentinel: leave at 0.0 when the user didn't pass an
                    // explicit length_scale so `auto_init_length_scale_in_place`
                    // can replace it with a data-derived initialization. The
                    // old hard-coded 1.0 was the documented basin (see
                    // smooth.rs `auto_init_length_scale_in_place`) that the
                    // spatial optimizer could not escape, leaving TPS terms
                    // initialized off the data scale.
                    length_scale: option_f64(options, "length_scale").unwrap_or(0.0),
                    double_penalty: smooth_double_penalty,
                    identifiability: parse_spatial_identifiability(options)
                        .map_err(|e| e.to_string())?,
                    radial_reparam: None,
                },
                input_scale: None,
            })
        }
        "sphere" | "s2" | "sos" => {
            validate_known_options("sphere", options, SPHERE_SMOOTH_OPTION_KEYS)?;
            if cols.len() != 2 {
                return Err(format!(
                    "sphere smooth expects exactly two variables (lat, lon), got {}",
                    cols.len()
                ));
            }
            let radians = option_bool(options, "radians").unwrap_or_else(|| {
                options
                    .get("units")
                    .map(|u| u.eq_ignore_ascii_case("radian") || u.eq_ignore_ascii_case("radians"))
                    .unwrap_or(false)
            });
            // An explicit `degree`/`l`/`max_degree` names a spherical-harmonic
            // truncation, so with no explicit kernel/method it selects the
            // Harmonic construction (the Wahba kernel ignores `degree` and would
            // silently emit a 1-column kernel design). An explicit kernel/method
            // still wins.
            let degree_requested = options.contains_key("degree")
                || options.contains_key("l")
                || options.contains_key("max_degree")
                || options.contains_key("max-degree");
            let kernel = options
                .get("kernel")
                .or_else(|| options.get("method"))
                .map(|raw| strip_quotes(raw).trim().to_ascii_lowercase())
                .unwrap_or_else(|| {
                    if degree_requested {
                        "harmonic".to_string()
                    } else {
                        "sobolev".to_string()
                    }
                });
            let (method, wahba_kernel) = match kernel.as_str() {
                "sobolev" | "wahba" | "wahba_sobolev" | "wahba-sobolev" => {
                    (SphereMethod::Wahba, SphereWahbaKernel::Sobolev)
                }
                "pseudo" | "mgcv" | "sos" | "wahba_pseudo" | "wahba-pseudo" => {
                    (SphereMethod::Wahba, SphereWahbaKernel::Pseudo)
                }
                "harmonic" | "spherical_harmonic" | "spherical-harmonic" => {
                    (SphereMethod::Harmonic, SphereWahbaKernel::Sobolev)
                }
                other => {
                    return Err(format!(
                        "unsupported sphere kernel '{other}'; expected sobolev, pseudo, or harmonic"
                    ));
                }
            };
            // `lmax=` states a finite spectral resolution for a Wahba kernel,
            // selecting the truncated variant `Σ_{ℓ=1..lmax} c_ℓ P_ℓ(cos γ)`
            // instead of the closed form. This is the only route from the
            // formula surface to `SobolevTruncated`/`PseudoTruncated`, and it
            // is what makes `m=1` expressible at all: the untruncated Sobolev
            // `m = 1` kernel is log-singular at coincidence, so it has no Gram
            // diagonal and the basis builder refuses it (#2475). Before this
            // option the refusal named a remedy no formula could reach.
            let wahba_kernel = match option_usize_any(options, &["lmax", "l_max", "l-max"]) {
                None => wahba_kernel,
                Some(_) if matches!(method, SphereMethod::Harmonic) => {
                    return Err(
                        "sphere smooth: lmax= states the truncation of a Wahba reproducing kernel \
                         and does not apply to kernel=harmonic; use degree=/max_degree= to set the \
                         harmonic degree"
                            .to_string(),
                    );
                }
                Some(lmax) => {
                    if !(SPHERE_TRUNCATION_LMAX_RANGE).contains(&lmax) {
                        return Err(format!(
                            "sphere smooth: lmax={lmax} is out of range; the truncated Wahba \
                             kernels support lmax in {}..={} (the device kernel bakes it in as a \
                             compile-time bound)",
                            SPHERE_TRUNCATION_LMAX_RANGE.start(),
                            SPHERE_TRUNCATION_LMAX_RANGE.end()
                        ));
                    }
                    let lmax = lmax as u16;
                    match wahba_kernel {
                        SphereWahbaKernel::Sobolev | SphereWahbaKernel::SobolevTruncated { .. } => {
                            SphereWahbaKernel::SobolevTruncated { lmax }
                        }
                        SphereWahbaKernel::Pseudo | SphereWahbaKernel::PseudoTruncated { .. } => {
                            SphereWahbaKernel::PseudoTruncated { lmax }
                        }
                    }
                }
            };
            let max_degree = if matches!(method, SphereMethod::Harmonic) {
                let degree =
                    option_usize_any(options, &["degree", "l", "max_degree", "max-degree"])
                        .or_else(|| option_usize(options, "centers"))
                        .or_else(|| {
                            option_usize_any(options, &["k", "basis_dim", "basis-dim", "basisdim"])
                                .and_then(|k| (1..=128).find(|&l| l * (l + 2) >= k))
                        })
                        .unwrap_or_else(|| default_spherical_harmonic_degree(sizing_rows));
                if degree == 0 {
                    return Err("sphere smooth requires degree/max_degree >= 1".to_string());
                }
                if degree > 32 {
                    return Err(format!(
                        "sphere smooth max_degree={} is too large for the dense harmonic engine (limit 32)",
                        degree
                    ));
                }
                Some(degree)
            } else {
                None
            };
            let penalty_order =
                parse_penalty_order_alias(options)?.unwrap_or(DEFAULT_PENALTY_ORDER);
            let center_strategy = if matches!(method, SphereMethod::Wahba) {
                let mut centers = parse_countwith_basis_alias(
                    options,
                    "centers",
                    default_num_centers(sizing_rows, cols.len()),
                )?;
                if penalty_order >= 4 {
                    centers = centers.max(30);
                }
                CenterStrategy::FarthestPoint {
                    num_centers: centers,
                }
            } else {
                CenterStrategy::FarthestPoint { num_centers: 0 }
            };
            Ok(SmoothBasisSpec::Sphere {
                feature_cols: cols.to_vec(),
                spec: SphericalSplineBasisSpec {
                    center_strategy,
                    penalty_order,
                    double_penalty: smooth_double_penalty,
                    radians,
                    method,
                    max_degree,
                    wahba_kernel,
                    identifiability: SphericalSplineIdentifiability::CenterSumToZero,
                },
            })
        }
        "curvature" => {
            // Constant-curvature (M_κ) geodesic-kernel smooth (#944): the
            // κ-generic sibling of the intrinsic S² smooth above. The feature
            // columns are κ-stereographic chart coordinates and the geometry
            // comes from `geometry::constant_curvature::ConstantCurvature`.
            // `kappa=` follows the mgcv-`sp=` convention (gam#2152): an EXPLICIT
            // value is a FIXED sectional curvature that selects the geometry
            // (`Sᵈ` for κ>0, `ℝᵈ` for κ=0, `Hᵈ` for κ<0) and is honoured verbatim
            // by the fit; OMITTING `kappa=` leaves κ free for the #944/#1464
            // outer ψ-coordinate estimation, seeded at the flat default 0.
            validate_known_options("curvature", options, CURVATURE_SMOOTH_OPTION_KEYS)?;
            // `kappa=` follows the mgcv-`sp=` convention: an EXPLICIT value pins
            // the sectional curvature (fixed geometry, honoured verbatim by the
            // fit — gam#2152); an OMITTED `kappa=` leaves κ free for the
            // #944/#1464 outer estimation, seeded at the flat default 0.
            let kappa_opt = option_f64(options, "kappa");
            let kappa_fixed = kappa_opt.is_some();
            let kappa = kappa_opt.unwrap_or(0.0);
            if !kappa.is_finite() {
                return Err("curvature smooth requires a finite kappa".to_string());
            }
            // `length_scale=` follows the SAME mgcv-`sp=` convention as `kappa=`
            // (gam#2747): an EXPLICIT value pins the kernel resolution and the fit
            // honours it verbatim; an OMITTED one leaves η = ln ℓ free for the
            // outer estimation, seeded by the auto rule. The range must be fitted
            // by default because it is confounded with κ — pinning it makes κ
            // absorb the range error rather than measure curvature.
            let length_scale_opt = option_f64(options, "length_scale");
            let length_scale_fixed = length_scale_opt.is_some();
            let length_scale = length_scale_opt.unwrap_or(0.0);
            if !length_scale.is_finite() || length_scale < 0.0 {
                return Err(format!(
                    "curvature smooth length_scale must be positive (or omitted for auto); got {length_scale}"
                ));
            }
            let centers = parse_countwith_basis_alias(
                options,
                "centers",
                default_num_centers(sizing_rows, cols.len()),
            )?;
            if centers < 2 {
                return Err("curvature smooth requires at least 2 centers".to_string());
            }
            let center_strategy = if has_explicit_countwith_basis_alias(options, "centers") {
                spatial_center_strategy_for_dimension(centers, cols.len())
            } else {
                auto_spatial_center_strategy(centers, cols.len())
            };
            Ok(SmoothBasisSpec::ConstantCurvature {
                feature_cols: cols.to_vec(),
                spec: ConstantCurvatureBasisSpec {
                    center_strategy,
                    kappa,
                    kappa_fixed,
                    // 0.0 sentinel = κ-independent auto initialization in the
                    // basis builder (median chart center spacing, doubled).
                    length_scale,
                    length_scale_fixed,
                    // Curvature smooth defaults to NO double-penalty ridge
                    // (#1464): the curvature-blind ridge `I` absorbs the data fit
                    // independently of κ and rails the fitted curvature to the
                    // +chart bound (hyperbolic truth recovered as spherical). The
                    // RKHS Gram penalty is already full-rank PD, so the ridge adds
                    // no stability. Honour an EXPLICIT `double_penalty=` only.
                    double_penalty: option_bool(options, "double_penalty").unwrap_or(false),
                    identifiability: ConstantCurvatureIdentifiability::CenterSumToZero,
                },
            })
        }
        "measurejet" => {
            // Measure-jet spline: multiscale local-jet-residual energy of the
            // empirical measure. The feature columns are ambient coordinates
            // of data concentrated near an unknown low-dimensional set; the
            // geometry (centers, masses, scale band) is read off the measure
            // at build time — magic by default, every option optional.
            validate_known_options("measurejet", options, MEASURE_JET_SMOOTH_OPTION_KEYS)?;
            let order_s = option_f64(options, "s").unwrap_or(0.0);
            // 0.0 = auto sentinel; explicit values must sit inside the
            // admissible order interval of the affine-jet (r = 2) energy.
            if !(order_s.is_finite() && (order_s == 0.0 || (order_s > 0.0 && order_s < 2.0))) {
                return Err(format!(
                    "measurejet smooth s must lie in (0, 2) (or be omitted for auto); got {order_s}"
                ));
            }
            // Default to the spec Default (α = 1, density-WEIGHTED Hessian
            // energy — the module-header default). The density-free α = 3/2
            // (q^{−2}) over-smooths low-intrinsic-dimension manifolds where the
            // local mass q is tiny and varies along the stratum (#1116:
            // 13×-worse-than-matérn on a 1-D curve in 3-D); α = 1's q^{−1} is
            // gentler and robust across intrinsic dimensions. An explicit
            // `alpha=` still overrides for full-dimensional density-free use.
            let alpha =
                option_f64(options, "alpha").unwrap_or(MeasureJetBasisSpec::default().alpha);
            if !alpha.is_finite() {
                return Err("measurejet smooth requires a finite alpha".to_string());
            }
            let tau0 = option_f64(options, "tau").unwrap_or(1e-3);
            if !(tau0.is_finite() && tau0 >= 0.0) {
                return Err(format!(
                    "measurejet smooth tau must be finite and nonnegative; got {tau0}"
                ));
            }
            let num_scales = option_usize(options, "scales").unwrap_or(0);
            let length_scale = option_f64(options, "length_scale").unwrap_or(0.0);
            if !length_scale.is_finite() || length_scale < 0.0 {
                return Err(format!(
                    "measurejet smooth length_scale must be positive (or omitted for auto); got {length_scale}"
                ));
            }
            let centers = parse_countwith_basis_alias(
                options,
                "centers",
                default_num_centers(sizing_rows, cols.len()),
            )?;
            if centers < 3 {
                return Err("measurejet smooth requires at least 3 centers".to_string());
            }
            let center_strategy = if has_explicit_countwith_basis_alias(options, "centers") {
                spatial_center_strategy_for_dimension(centers, cols.len())
            } else {
                auto_spatial_center_strategy(centers, cols.len())
            };
            // Multiscale (per-scale spectral split + (α, lnτ) ψ dials + the
            // affine-preserving ridge) is an explicit opt-in (#1116): default
            // single-scale at any center count, the Duchon/Matérn footprint.
            let multiscale = option_bool(options, "multiscale").unwrap_or(false);
            // The representer range ℓ is a design-moving basis coordinate of the
            // same kind as the Matérn κ: λ shrinks inside a span and cannot move
            // one, so a frozen ℓ is an error no smoothing parameter can repair
            // (#2761 measured it at 13.4x held-out RMSE on a 1-D curve in 3-D,
            // with the design's own span floor sitting AT the fitted value).
            // REML therefore selects it by default.
            //
            // An explicit `length_scale=` is a request, not a seed, so it pins ℓ
            // — the same short-circuit `all_spatial_terms_kappa_fixed` gives an
            // explicitly-scaled Matérn. `learn_length_scale=` overrides either
            // way.
            let learn_length_scale =
                option_bool(options, "learn_length_scale").unwrap_or(length_scale == 0.0);
            Ok(SmoothBasisSpec::MeasureJet {
                feature_cols: cols.to_vec(),
                spec: MeasureJetBasisSpec {
                    center_strategy,
                    order_s,
                    alpha,
                    tau0,
                    num_scales,
                    // 0.0 sentinel = auto initialization in the basis builder
                    // (median nearest-center spacing).
                    length_scale,
                    double_penalty: smooth_double_penalty,
                    learn_length_scale,
                    multiscale,
                    identifiability: MeasureJetIdentifiability::CenterSumToZero,
                    frozen_quadrature: None,
                },
                input_scale: None,
            })
        }
        "matern" => {
            // Catch typos like `lengt_scale=` / `nyu=` / `centerz=` before
            // they get silently ignored and the user wonders why their
            // option had no effect. The matern() term accepts exactly
            // these options.
            validate_known_options("matern", options, MATERN_SMOOTH_OPTION_KEYS)?;
            let plan = plan_spatial_basis(
                sizing_rows,
                cols.len(),
                CenterCountRequest::Default,
                DuchonNullspaceOrder::Zero,
                option_bool(options, "scale_dims").unwrap_or(false),
                policy,
            )
            .map_err(|e| e.to_string())?;
            // #1867: spline-equivalent floor so a 1-D radial basis is not
            // dimensioned coarser than the competing `s(x)` on identical data.
            let univariate_floor = if cols.len() == 1 {
                heuristic_knots_for_column(ds.values.column(cols[0]))
                    .saturating_add(DEFAULT_BSPLINE_DEGREE + 1)
            } else {
                0
            };
            let centers = parse_countwith_basis_alias(
                options,
                "centers",
                cap_default_spatial_centers(
                    options,
                    default_matern_center_count(
                        sizing_rows,
                        cols.len(),
                        plan.centers,
                        univariate_floor,
                    ),
                ),
            )?;
            let center_strategy = if has_explicit_countwith_basis_alias(options, "centers") {
                spatial_center_strategy_for_dimension(centers, cols.len())
            } else {
                auto_spatial_center_strategy(centers, cols.len())
            };
            let nu = parse_matern_nu(options.get("nu").map(String::as_str).unwrap_or("5/2"))?;
            // The exponential (ν = 1/2) Matérn kernel has a singular Laplacian
            // at zero in d ≥ 2, so the operator-collocation penalty machinery
            // hits a non-invertible matrix during fit. Surface the cause
            // up-front instead of letting the user see the generic
            // "Matrix conditioning issue detected" wrapper from PIRLS.
            if matches!(nu, MaternNu::Half) && cols.len() >= 2 {
                return Err(TermBuilderError::unsupported_feature(format!(
                    "matern() with nu=1/2 is not supported for d>=2 (got {} covariates): \
                     the exponential kernel's Laplacian is singular at center collisions, \
                     which makes the operator-collocation penalty non-invertible. \
                     Choose nu>=3/2 (e.g. nu=3/2 or the default nu=5/2) for multi-dimensional smooths.",
                    cols.len()
                ))
                .to_string());
            }
            let aniso_log_scales = if option_bool(options, "scale_dims").unwrap_or(false) {
                Some(vec![0.0; cols.len()])
            } else {
                None
            };
            let periodic = parse_periodic_axes_option(options, cols.len())?;
            reject_unconsumable_radial_period_declaration(
                "matern",
                options,
                cols.len(),
                periodic.as_deref(),
                false,
            )?;
            Ok(SmoothBasisSpec::Matern {
                feature_cols: cols.to_vec(),
                spec: MaternBasisSpec {
                    center_strategy,
                    periodic,
                    // Preserve whether the user supplied `length_scale` as typed
                    // provenance. The planner resolves `Auto` to the same
                    // data-derived wiggly-side initialization the thin-plate path
                    // uses (`max_range / sqrt(n)`), then lets the κ-optimizer refine
                    // it without ever turning it into a user-fixed scale.
                    //
                    // gam#1629: the previous `default_matern_length_scale` seeded
                    // the FULL data diameter — the maximally over-smoothed corner.
                    // Because that value looked explicit, the old auto-init was a
                    // no-op for Matérn, so the κ-optimizer started in the flat
                    // over-smoothed basin and parked there, leaving high-frequency
                    // 2-D surfaces unresolved (truth-RMSE ~6× worse than
                    // thin-plate/tensor on identical data, and insensitive to `k`).
                    // Typed Auto starts REML in the resolving regime it can escape
                    // from and cannot be confused with explicit zero.
                    length_scale: option_f64(options, "length_scale")
                        .map(MaternLengthScale::fixed)
                        .unwrap_or_else(MaternLengthScale::auto),
                    nu,
                    include_intercept: option_bool(options, "include_intercept").unwrap_or(false),
                    double_penalty: smooth_double_penalty,
                    identifiability: parse_matern_identifiability(options)
                        .map_err(|e| e.to_string())?,
                    aniso_log_scales,
                    // Cold build: let the bootstrap-κ spectral test decide whether
                    // the double-penalty nullspace shrinkage survives; the freeze
                    // step then pins that decision into the FrozenTransform so the
                    // κ-optimizer's rebuilds keep the count invariant (gam#787/#860).
                },
                input_scale: None,
            })
        }
        "duchon" | "ds" => {
            validate_known_options("duchon", options, DUCHON_SMOOTH_OPTION_KEYS)?;
            if options.contains_key("double_penalty") {
                return Err(TermBuilderError::incompatible_config(format!(
                    "Duchon smooth '{}' does not support double_penalty; the Duchon smoother already ships its native reproducing-norm penalty plus a null-space shrinkage ridge.",
                    vars.join(", ")
                ))
                .to_string());
            }
            let requested_nullspace_order = parse_duchon_order_opt(options)?;
            let length_scale = option_f64_strict(options, "length_scale")?;
            // Resolve `(nullspace_order, power)`. The default (magic) path is a
            // structural amplitude/slope/curvature smoother: an affine (`Linear`)
            // polynomial nullspace and spectral power `s = (d - 1)/2`, giving the
            // cubic kernel `r^3` in 1D. There is no nullspace-order escalation —
            // the structural cubic smoother is well-defined for every dimension.
            //
            // Explicit `power=...` honors the user's value verbatim against their
            // requested nullspace order; the kernel validator emits a precise
            // diagnostic for any inadmissible combination. In the scale-free
            // (non-hybrid) regime fractional powers are admitted and threaded as
            // `f64`. The hybrid Duchon-Matérn kernel (`length_scale=Some`) is
            // restricted to integer powers.
            let (nullspace_order, power) = match parse_duchon_power_policy(options)? {
                DuchonPowerPolicy::Explicit(req_power) => {
                    if length_scale.is_some() && req_power.fract() != 0.0 {
                        return Err(TermBuilderError::incompatible_config(format!(
                            "hybrid Duchon-Matern smooth '{}' (length_scale=...) requires an integer power, got power={}; \
                             drop length_scale to use the scale-free structural kernel with a fractional power.",
                            vars.join(", "),
                            req_power,
                        ))
                        .to_string());
                    }
                    (
                        requested_nullspace_order.unwrap_or(DuchonNullspaceOrder::Linear),
                        req_power,
                    )
                }
                DuchonPowerPolicy::CubicStructuralDefault => {
                    // Magic cubic rule (REQUEST-LAYER default): no explicit power ⇒
                    // affine null space + fractional spectral power s = (d-1)/2, i.e.
                    // the Duchon kernel φ(r)=r³ in every dimension. An EXPLICIT
                    // `power=0` is handled above and is honored as the s=0 Duchon
                    // kernel (r²·log r ≡ the thin-plate kernel in even d) — the magic
                    // default lives here, not in the basis builder.
                    // An explicit `order=` names the polynomial null space; the
                    // structural default then supplies only the spectral power.
                    // Taking the whole PAIR from the default discarded a
                    // caller's `order=` whenever no `power=` accompanied it, so
                    // `duchon(x, z, order=0)` and `order=2` were parsed,
                    // validated, and thrown away (#2781's family). `order=1` is
                    // the default null space, so every shipped
                    // `duchon(..., order=1)` formula is unaffected.
                    match length_scale {
                        None => {
                            let (default_order, s) =
                                crate::basis::duchon_cubic_default(cols.len());
                            (requested_nullspace_order.unwrap_or(default_order), s)
                        }
                        Some(_) => {
                            // The hybrid Matérn-blended kernel (`length_scale=Some`)
                            // requires an INTEGER spectral power `s` (the partial-
                            // fraction split `1/(ρ^{2p}(κ²+ρ²)^s)` is only defined for
                            // integer `s`). The fractional cubic default `s=(d-1)/2` is
                            // a half-integer for even `d`, and the basis builder's
                            // `power_as_usize` maps a NON-integer to `0` (not its
                            // floor) — so for even `d ≥ 4` the realized kernel has
                            // `2(p+s) = 2p = 4 ≤ d`, which is non-finite at the origin
                            // and crashes the fit (historically a non-finite
                            // eigendecomposition; now a fit-time validation error).
                            //
                            // Resolve to the same structural cubic default the
                            // scale-free path uses (affine `Linear` null space, `r³`
                            // kernel, fractional power `s = (d-1)/2`) but take the
                            // largest admissible INTEGER at or below it — `⌊(d-1)/2⌋`.
                            // For odd `d` this is exactly the cubic power (the hybrid
                            // default then agrees with the scale-free cubic default);
                            // for even `d` it is the nearest integer below. Either way
                            // `p = 2` (affine) gives spectral order
                            // `2(p+s) = d+3` (odd `d`) or `d+2` (even `d`), which
                            // clears both kernel existence `2(p+s) > d` and the D1
                            // collocation floor `2(p+s) > d+1` for every `d ≥ 1`.
                            // Flooring here at the request layer avoids the
                            // `power_as_usize` truncation-to-zero on the fractional
                            // half-integer.
                            let (default_order, s_frac) =
                                crate::basis::duchon_cubic_default(cols.len());
                            (
                                requested_nullspace_order.unwrap_or(default_order),
                                s_frac.floor(),
                            )
                        }
                    }
                }
            };
            let plan = plan_spatial_basis(
                sizing_rows,
                cols.len(),
                CenterCountRequest::Default,
                nullspace_order,
                option_bool(options, "scale_dims").unwrap_or(false),
                policy,
            )
            .map_err(|e| e.to_string())?;
            let centers_explicit = has_explicit_countwith_basis_alias(options, "centers");
            let polynomial_cols = match nullspace_order {
                DuchonNullspaceOrder::Zero => 1,
                DuchonNullspaceOrder::Linear => cols.len() + 1,
                DuchonNullspaceOrder::Degree(degree) => {
                    crate::basis::duchon_nullspace_dimension(cols.len(), degree)
                }
            };
            // #1867: spline-equivalent floor so a 1-D radial basis is not
            // dimensioned coarser than the competing `s(x)` on identical data.
            let univariate_floor = if cols.len() == 1 {
                heuristic_knots_for_column(ds.values.column(cols[0]))
                    .saturating_add(DEFAULT_BSPLINE_DEGREE + 1)
            } else {
                0
            };
            let default_centers = default_duchon_center_count(
                sizing_rows,
                cols.len(),
                plan.centers,
                polynomial_cols,
                univariate_floor,
            );
            let spectral_rank = option_usize(options, "rank");
            let center_default = if spectral_rank.is_some() {
                // mgcv's Duchon constructor runs `uniquecombs` FIRST and caps
                // at `max.knots` afterwards, so its knot budget is
                // `min(n_unique, 2000)`. This took the RAW row count and let
                // `select_r_uniform_subsample_centers` deduplicate later — so
                // on any data carrying a repeated coordinate row with fewer
                // than 2000 rows, the budget exceeded what the sampler could
                // supply and the fit hard-refused rather than degrading
                // (#2623: `prostate_gamair`, 523 requested vs 522 unique).
                // Counting distinct rows here makes the budget satisfiable by
                // construction, and leaves the retained spectral `rank` — a
                // separate option — untouched.
                count_unique_coordinate_rows(ds.values.view(), &cols).min(2000)
            } else {
                cap_default_spatial_centers(options, default_centers)
            };
            let requested_centers =
                parse_countwith_basis_alias(options, "centers", center_default)?;
            if requested_centers > ds.values.nrows() {
                return Err(TermBuilderError::incompatible_config(format!(
                    "Duchon smooth '{}' requested {requested_centers} centers but only {} rows are available",
                    vars.join(", "),
                    ds.values.nrows(),
                ))
                .to_string());
            }
            if requested_centers <= polynomial_cols {
                return Err(TermBuilderError::incompatible_config(format!(
                    "Duchon smooth '{}' requested basis dimension {} but order={:?} in {}D needs {} polynomial null-space columns; choose centers/k > {}",
                    vars.join(", "),
                    requested_centers,
                    nullspace_order,
                    cols.len(),
                    polynomial_cols,
                    polynomial_cols,
                ))
                .to_string());
            }
            if let Some(rank) = spectral_rank
                && (rank <= polynomial_cols || rank > requested_centers)
            {
                return Err(TermBuilderError::incompatible_config(format!(
                    "Duchon smooth '{}' spectral rank must satisfy {} < rank <= centers (got rank={rank}, centers={requested_centers})",
                    vars.join(", "),
                    polynomial_cols,
                ))
                .to_string());
            }
            let mut centers = requested_centers;
            if !centers_explicit && ds.values.nrows() <= 32 && smooth_coordinate_count >= 5 {
                centers = centers.max(polynomial_cols + 4);
            }
            let aniso_log_scales = if option_bool(options, "scale_dims").unwrap_or(false) {
                Some(vec![0.0; cols.len()])
            } else {
                None
            };
            // Formula-level `duchon(...)` is the native Duchon reproducing-norm
            // smoother: the always-on Primary Gram plus the polynomial trend
            // ridge. Do not silently add collocated mass/tension penalties here.
            // They add extra REML hyperparameters and an O(k)-support quadrature
            // build to the default 2-D path, making `duchon(x, z)` materially
            // slower than the equivalent thin-plate fit without a principled
            // accuracy gain (gam#1718). Lower-order Hilbert-scale penalties remain
            // available to callers that construct an explicit DuchonBasisSpec.
            let operator_penalties = DuchonOperatorPenaltySpec::all_disabled();
            // For a 1-D periodic Duchon with no EXPLICIT period, anchor the wrap
            // to the covariate DATA range rather than letting the basis builder
            // derive it from the (k-subsampled) center span. The center span is a
            // strict subset of the data and undershoots the true period, seaming
            // the curve (f(0) ≠ f(2π)); the data range is the caller's actual
            // domain. Honors any explicit `period=` (parse_periodic_axes_option
            // already threaded it) and leaves multi-D / non-periodic untouched.
            let mut periodic = parse_periodic_axes_option(options, cols.len())?;
            if cols.len() == 1
                && let Some(axes) = periodic.as_mut()
                && axes.len() == 1
                && axes[0].is_none()
            {
                let (minv, maxv) = col_minmax(ds.values.column(cols[0]))?;
                if maxv > minv {
                    axes[0] = Some(maxv - minv);
                }
            }
            let boundary = if cols.len() == 1 {
                let c = cols[0];
                let (minv, maxv) = col_minmax(ds.values.column(c))?;
                parse_cyclic_boundary(options, minv, maxv)?
            } else {
                OneDimensionalBoundary::Open
            };
            let is_periodic = periodic
                .as_ref()
                .is_some_and(|axes| axes.iter().any(Option::is_some))
                || matches!(boundary, OneDimensionalBoundary::Cyclic { .. });
            reject_unconsumable_radial_period_declaration(
                "duchon",
                options,
                cols.len(),
                periodic.as_deref(),
                matches!(boundary, OneDimensionalBoundary::Cyclic { .. }),
            )?;
            if spectral_rank.is_some() && is_periodic {
                return Err(TermBuilderError::incompatible_config(
                    "Duchon spectral rank is defined for the scale-free open-domain kernel, \
                     not a periodic image expansion"
                        .to_string(),
                )
                .to_string());
            }
            let center_strategy = if spectral_rank.is_some() {
                // Freeze the exact fixed-seed uniform landmark experiment used
                // by mgcv's Duchon constructor. Spectral rank parity requires
                // the same kernel matrix, not merely the same retained column
                // count: maximin/equal-mass landmarks define a different
                // finite-sample eigenspace and confound accuracy comparisons.
                // Materializing 2,000×d coordinates here is cheap, avoids an
                // O(nk) maximin pass, and makes prediction replay explicit.
                let mut coordinates = Array2::<f64>::zeros((ds.values.nrows(), cols.len()));
                for (axis, &column) in cols.iter().enumerate() {
                    coordinates
                        .column_mut(axis)
                        .assign(&ds.values.column(column));
                }
                let sampled = select_r_uniform_subsample_centers(coordinates.view(), centers, 1)
                    .map_err(|error| error.to_string())?;
                CenterStrategy::UserProvided(sampled)
            } else if is_periodic {
                if centers_explicit {
                    spatial_center_strategy_for_dimension(centers, cols.len())
                } else {
                    auto_spatial_center_strategy(centers, cols.len())
                }
            } else {
                duchon_center_strategy(centers, cols.len(), !centers_explicit)
            };
            let center_strategy = match spectral_rank {
                Some(rank) => CenterStrategy::DuchonSpectral {
                    knots: Box::new(center_strategy),
                    basis: DuchonSpectralBasis::Fresh { rank },
                },
                None => center_strategy,
            };
            Ok(SmoothBasisSpec::Duchon {
                feature_cols: cols.to_vec(),
                spec: DuchonBasisSpec {
                    center_strategy,
                    periodic,
                    length_scale,
                    power,
                    nullspace_order,
                    identifiability: parse_spatial_identifiability(options)
                        .map_err(|e| e.to_string())?,
                    aniso_log_scales,
                    operator_penalties,
                    boundary,
                    radial_reparam: None,
                },
                input_scale: None,
            })
        }
        "tensor" | "te" | "ti" | "t2" => {
            validate_known_options("tensor", options, TENSOR_SMOOTH_OPTION_KEYS)?;
            if cols.len() < 2 {
                return Err(TermBuilderError::incompatible_config(format!(
                    "tensor smooth expects at least 2 variables, got {}",
                    cols.len()
                ))
                .to_string());
            }
            let dim = cols.len();

            // Tensor-product contract (#1082). `te(x1, x2, ...)` ALWAYS builds a
            // genuine anisotropic tensor product of per-margin bases (the arm
            // below), exactly as mgcv's `te()` does — one smoothing parameter per
            // margin, a marginal-Kronecker-sum penalty, and a separate default
            // function-space ridge on the joint polynomial null space. A margin
            // vector `bs=c('tp','tp')` requests a thin-plate FUNCTION SPACE per
            // axis; the tensor realizes each axis as a 1-D penalized B-spline
            // margin spanning that same per-axis space (tp/ps/cr/bs/cc all share
            // it). We deliberately do NOT silently swap the requested tensor for a
            // single multi-D ISOTROPIC thin-plate radial smooth (`s(x,y,bs='tp')`):
            // that is a different model — one isotropic smoothing parameter, no
            // per-margin anisotropy — and substituting it while the user wrote a
            // tensor formula is dishonest. A user who genuinely wants the isotropic
            // radial smooth asks for it directly with `s(x1, x2, bs='tp')`.
            // Per-margin basis vector (`bs=c('tp','tp')` / `bs=['ps','cr']`):
            // validate each requested margin is a penalized-spline basis that
            // the tensor product realizes as a 1-D B-spline margin. mgcv's
            // `tp`/`ps`/`cr`/`bs`/`cc` margins are all penalized splines over
            // the same per-axis function space, so a B-spline margin recovers
            // the same tensor smoothing space; genuinely different margin kinds
            // (e.g. adaptive `ad`, random `re`) are rejected loudly rather than
            // silently substituted.
            if let Some(raw) = options.get("bs").or_else(|| options.get("type"))
                && bs_selector_is_vector(raw)
            {
                let per_margin = parse_option_list(raw);
                if per_margin.len() != dim {
                    return Err(TermBuilderError::invalid_option(format!(
                        "tensor smooth per-margin bs vector has {} entries but the smooth has {} margins",
                        per_margin.len(),
                        dim
                    ))
                    .to_string());
                }
                for (axis, margin_bs) in per_margin.iter().enumerate() {
                    if !tensor_margin_bs_is_supported(margin_bs) {
                        return Err(TermBuilderError::unsupported_feature(format!(
                            "tensor smooth margin {axis} basis '{margin_bs}' is not a supported penalized-spline margin; \
                             tensor margins accept tp/tps/ps/bs/cr/cc"
                        ))
                        .to_string());
                    }
                }
            }
            // Validate the boundary tokens BEFORE the axis resolver reads them,
            // so a malformed list is refused by name rather than silently
            // failing the resolver's length guard.
            validate_tensor_boundary_tokens(options, dim)?;
            let periodic_axes = parse_tensor_periodic_axes(options, dim)?;
            reject_unconsumable_period_declaration("tensor", options, &periodic_axes)?;
            // The half-open endpoint spelling names a single axis's domain and
            // has no per-margin form, so the tensor arm never reads it — it went
            // in through `validate_known_options` and straight out again (#2781).
            if let Some(key) = PERIOD_ENDPOINT_OPTION_KEYS
                .iter()
                .find(|key| options.contains_key(**key))
            {
                return Err(TermBuilderError::invalid_option(format!(
                    "tensor(): `{key}=` declares one axis's periodic domain and has no per-margin \
                     form; on a tensor smooth give periods=[...] (with origins=[...] for the \
                     domain start), which name their margin"
                ))
                .to_string());
            }
            let periods_opt = parse_periods(options, &periodic_axes)?;
            let origins_opt = parse_period_origins(options, &periodic_axes)?;
            // Per-margin `degree=` / `penalty_order=`. Both keep the caller's
            // request as `Option` rather than collapsing it onto the default
            // immediately: the cr-margin routing below has to know whether the
            // default was ASKED FOR or merely not overridden (#2782).
            let requested_degrees = parse_tensor_per_axis_usize(options, "degree", dim)?;
            let requested_penalty_orders =
                parse_tensor_per_axis_usize(options, "penalty_order", dim)?;
            let axis_degree = |axis: usize| -> usize {
                requested_degrees[axis].unwrap_or(DEFAULT_BSPLINE_DEGREE)
            };
            let axis_penalty_order = |axis: usize| -> usize {
                requested_penalty_orders[axis]
                    .unwrap_or(if axis_degree(axis) > 1 { 2 } else { 1 })
            };
            let (mut k_list, k_inferred) = parse_tensor_k_list(options, cols, ds)?;
            if ds.values.nrows() <= 32 && smooth_coordinate_count >= 5 {
                for (axis, k) in k_list.iter_mut().enumerate() {
                    *k = (*k).min(axis_degree(axis) + 2);
                }
            }
            if k_inferred {
                inference_notes.push(format!(
                    "Automatically set per-margin basis sizes {:?} for tensor smooth '{}' \
                     (dimension-aware tensor budget: total ∏k kept near the mgcv-te default \
                     and within the data support, distributed geometrically across margins and \
                     capped per margin by each column's resolution). \
                     Override with k=<int> or k=[k0,k1,...].",
                    k_list,
                    vars.join(",")
                ));
            }
            // Per-axis requested marginal basis family. mgcv's `te()`/`ti()`
            // default marginal basis is the cubic regression spline (`cr`), and
            // the te_3d quality gap (#1074) is precisely the marginal-basis
            // resolution at small `k`: a `cr` margin places k value-knots at
            // data quantiles (finer interior resolution under natural boundary
            // constraints) where the cubic B-spline margin has only
            // `k-degree-1` interior knots. Resolve each axis to either an
            // explicit per-margin `bs` (vector `bs=c('cr','ps')`), a single
            // scalar `bs`, or the unset default — and route
            // `cr`/`cs`/unset/`tp`/`tps` margins through the natural cubic
            // regression builder (`NaturalCubicRegression` knotspec), keeping
            // explicit `ps`/`bs`/`bspline` on the B-spline margin.
            let per_axis_bs: Vec<Option<String>> =
                match options.get("bs").or_else(|| options.get("type")) {
                    Some(raw) if bs_selector_is_vector(raw) => {
                        let list = parse_option_list(raw);
                        (0..dim).map(|a| list.get(a).cloned()).collect()
                    }
                    Some(raw) => {
                        let scalar = raw
                            .trim()
                            .trim_matches('"')
                            .trim_matches('\'')
                            .to_ascii_lowercase();
                        vec![Some(scalar); dim]
                    }
                    None => vec![None; dim],
                };
            // A margin is realized as a natural cubic regression spline when it
            // is the (unset) mgcv default, an explicit `cr`/`cs`, or a
            // `tp`/`tps` (same per-axis penalized-spline space). Explicit
            // B-spline-family margins (`ps`/`bs`/`bspline`/`p-spline`) keep the
            // open B-spline margin.
            let margin_wants_cr = |bs: &Option<String>| -> bool {
                matches!(
                    bs.as_deref(),
                    None | Some("cr") | Some("cs") | Some("tp") | Some("tps")
                )
            };
            let requested_knot_placement = explicit_knot_placement(options)?;
            let mut margins: Vec<BSplineBasisSpec> = Vec::with_capacity(dim);
            let mut emitted_periods: Vec<Option<f64>> = Vec::with_capacity(dim);
            for axis in 0..dim {
                let c = cols[axis];
                let (data_min, data_max) = col_minmax(ds.values.column(c))?;
                // mgcv reduces a tensor margin's basis dimension to what its data
                // can support: a cr or B-spline margin cannot place more value
                // knots / basis functions than there are DISTINCT covariate
                // values on that axis. Without this cap an explicit `k` on a
                // low-cardinality margin — e.g. the binary `badh ∈ {0,1}` in
                // `te(age, badh, k=5)` — hard-failed in `select_cr_knots` ("cubic
                // regression spline with k=5 requires at least 5 distinct values,
                // got 2") instead of degrading to the 2-function (linear) margin
                // mgcv builds there. The auto-`k` path already caps per margin via
                // `heuristic_tensor_margin_knots`; mirror that for explicit `k`.
                // The cap propagates correctly: every per-axis quantity below
                // (effective degree, knot set, penalty order) is derived from
                // `k_axis`, and the marginal basis size is read from the resulting
                // knot spec — never from `k_list`. Floor at 2 so a margin still
                // carries at least a linear basis (tensor margins require k >= 2).
                let k_requested = k_list[axis];
                let n_distinct_axis = unique_count_column(ds.values.column(c));
                let k_axis = k_requested.min(n_distinct_axis).max(2);
                if k_axis < k_requested {
                    log::info!(
                        "tensor smooth: margin axis {axis} requested k={k_requested}, but the \
                         covariate has only {n_distinct_axis} distinct value(s); reducing this \
                         margin to k={k_axis} (mgcv-style data-support cap on the per-axis basis)."
                    );
                }
                // Per-axis effective spline degree. The B-spline basis with `k`
                // functions is well-defined for any `degree <= k - 1`; mgcv's
                // `te(...)` exploits this so a binary tensor margin
                // (`k=2` → linear basis) or a ternary margin (`k=3` → quadratic)
                // can coexist with a smoother continuous margin under one
                // shared `degree=` request. We mirror that: if the caller
                // explicitly asks for `k < degree + 1`, drop the degree on
                // THAT axis only to the largest feasible spline, and track the
                // penalty order so the marginal difference penalty stays
                // well-defined (`order < num_basis_functions` is required by
                // `create_difference_penalty_matrix`). Apply the same
                // per-margin degree shrinkage to periodic tensor margins too:
                // a cyclic marginal basis with k=3 cannot be cubic, but it is
                // still a valid lower-degree cyclic margin with dimension k,
                // matching mgcv's small-k tensor-margin behavior.
                if k_axis < 2 {
                    return Err(TermBuilderError::invalid_option(format!(
                        "tensor smooth: k[{axis}]={k_axis} too small; tensor margins require k >= 2"
                    ))
                    .to_string());
                }
                let degree = axis_degree(axis);
                let penalty_order = axis_penalty_order(axis);
                let effective_degree = degree.min(k_axis - 1).max(1);
                let effective_penalty_order = penalty_order.min(effective_degree);
                // A `cc`/`cp`/`cyclic` per-margin basis declares periodicity
                // without necessarily supplying a `period=`: mgcv's `bs="cc"`
                // wraps at the covariate's observed data range. Mirror the 1-D
                // cyclic fallback (`parse_periodic_domain_1d`) here so a bare
                // `te(x, z, bs=c('cc','cc'))` wraps each margin on its own
                // [min, max] span instead of hard-erroring (#1752).
                let margin_is_cc = matches!(
                    canonicalize_smooth_type(per_axis_bs[axis].as_deref().unwrap_or("")),
                    "cc" | "cp" | "cyclic"
                );
                let (knotspec, boundary, axis_period) = if periodic_axes[axis] {
                    // A `cc`/`cp`/`cyclic` per-margin basis declares periodicity
                    // without necessarily supplying a `period=`; in that case wrap
                    // at the covariate's observed [min, max] span, mirroring the
                    // 1-D cyclic fallback (`parse_periodic_domain_1d`) so a bare
                    // `te(x, z, bs=c('cc','cc'))` wraps each margin on its own
                    // range instead of hard-erroring (#1752). An axis made
                    // periodic by an explicit `periodic=`/`boundary=` selector
                    // (not a cyclic margin basis) still requires an explicit
                    // `period=`: a data-derived period there is a sample-dependent
                    // off-by-ε seam and is not inferred.
                    let (domain_start, period_value) = match periods_opt[axis] {
                        Some(period_value) => {
                            if !period_value.is_finite() || period_value <= 0.0 {
                                return Err(format!(
                                    "tensor smooth axis {axis}: period must be a positive finite value, got {period_value}"
                                ));
                            }
                            (origins_opt[axis].unwrap_or(data_min), period_value)
                        }
                        None if margin_is_cc => {
                            let span = data_max - data_min;
                            if !span.is_finite() || span <= 0.0 {
                                return Err(format!(
                                    "tensor smooth axis {axis}: cyclic margin requires a positive \
                                     observed data range to derive its period, got [{data_min}, {data_max}]"
                                ));
                            }
                            (origins_opt[axis].unwrap_or(data_min), span)
                        }
                        None => {
                            return Err(format!(
                                "tensor smooth axis {axis} is periodic but requires an explicit \
                                 period: pass period=<value> (scalar) or period=[..., <value>, ...]. \
                                 Deriving the period from the observed data range is sample-dependent \
                                 (off-by-ε seam), so it is not inferred."
                            ));
                        }
                    };
                    let domain_end = domain_start + period_value;
                    (
                        BSplineKnotSpec::PeriodicUniform {
                            data_range: (domain_start, domain_end),
                            num_basis: k_axis,
                        },
                        OneDimensionalBoundary::Cyclic {
                            start: domain_start,
                            end: domain_end,
                        },
                        Some(period_value),
                    )
                } else if margin_wants_cr(&per_axis_bs[axis])
                    && requested_knot_placement.is_none()
                    && requested_degrees[axis].is_none_or(|d| d == CR_MARGIN_DEGREE)
                    && requested_penalty_orders[axis]
                        .is_none_or(|m| m == CR_MARGIN_PENALTY_ORDER)
                    && k_axis >= 3
                {
                    // mgcv `te()`/`ti()` default cr margin: place exactly
                    // `k_axis` Lancaster–Salkauskas value-knots at data
                    // quantiles. The cr basis dimension equals the knot count,
                    // so this reproduces the requested per-margin `k` directly.
                    // A natural cubic regression spline needs at least 3 knots
                    // (one interior); a `k_axis < 3` margin (e.g. a binary
                    // tensor axis requesting a linear margin) falls through to
                    // the B-spline branch below, exactly as before #1074 — mgcv
                    // likewise does not build a `cr` margin below k=3. An
                    // explicit `knot_placement=quantile` also falls through:
                    // that option selects the generated B-spline knot strategy
                    // represented by `Automatic { Quantile }`, whereas the cr
                    // margin has already materialized its quantile value-knots.
                    let cr_knots = crate::basis::select_cr_knots(ds.values.column(c), k_axis)
                        .map_err(|e| e.to_string())?;
                    (
                        BSplineKnotSpec::NaturalCubicRegression { knots: cr_knots },
                        OneDimensionalBoundary::Open,
                        None,
                    )
                } else {
                    // `num_internal_knots = k - effective_degree - 1` is the
                    // only count that realises the requested per-margin basis
                    // size: a clamped degree-`d` B-spline with `m` internal
                    // knots has exactly `m + d + 1` functions, and zero
                    // internal knots (`k = degree + 1`, one polynomial piece)
                    // is a valid margin. A legacy `.max(1)` floor on the
                    // un-reduced path used to turn `k=4` into a five-function
                    // cubic margin, so `k=4` and `k=5` built the same tensor.
                    // `k_axis >= 2` and `effective_degree <= k_axis - 1`, so
                    // this cannot underflow.
                    let num_internal_knots = k_axis - effective_degree - 1;
                    let knotspec = match requested_knot_placement
                        .unwrap_or(crate::basis::BSplineKnotPlacement::Uniform)
                    {
                        crate::basis::BSplineKnotPlacement::Uniform => BSplineKnotSpec::Generate {
                            data_range: (data_min, data_max),
                            num_internal_knots,
                        },
                        crate::basis::BSplineKnotPlacement::Quantile => {
                            crate::basis::auto_knot_vector_1d_quantile(
                                ds.values.column(c),
                                num_internal_knots,
                                effective_degree,
                            )
                            .map_err(|e| e.to_string())?;
                            BSplineKnotSpec::Automatic {
                                num_internal_knots: Some(num_internal_knots),
                                placement: crate::basis::BSplineKnotPlacement::Quantile,
                            }
                        }
                    };
                    (knotspec, OneDimensionalBoundary::Open, None)
                };
                // Margins contribute only their roughness operators. The tensor
                // builder constructs exactly one joint function-space null
                // penalty, avoiding unused per-margin ridge candidates and
                // duplicate λ coordinates.
                margins.push(BSplineBasisSpec {
                    degree: effective_degree,
                    penalty_order: effective_penalty_order,
                    knotspec,
                    double_penalty: false,
                    identifiability: BSplineIdentifiability::None,
                    boundary,
                    boundary_conditions: BSplineBoundaryConditions::default(),
                });
                emitted_periods.push(axis_period);
            }
            // #1593: canonicalize the margin order so a tensor smooth is invariant
            // to the typed order of its covariates. `te(x, z)` and `te(z, x)` span
            // the IDENTICAL tensor-product space under the identical per-margin
            // penalty family, but the design is the Khatri–Rao product
            // `B_first ⊙ B_second`, so the typed order permutes the design columns
            // (and the per-margin penalty blocks `S_first⊗I`, `I⊗S_second`). That
            // permutation is a pure relabelling in exact arithmetic — REML is
            // invariant to it — yet it reorders the penalized normal-equation / REML
            // eigen/Cholesky linear algebra, and the resulting sub-ULP differences
            // route the outer λ optimizer to a different terminal point in te's flat
            // REML valley (the over-smoothed margin rails to the ρ bound while the
            // other lands on a materially different λ̂). So the shipped surface
            // drifted ~2–6 % of range with a cosmetic swap of the covariate order
            // (the #1378 row-permutation / #1456 rotation flat-valley gauge family).
            // Sorting the margins by their source feature-column index makes the same
            // physical model build the identical problem regardless of typed order,
            // so the fit — and every prediction rebuilt from the resolved spec — is
            // genuinely order-invariant. `ti`/`t2` share this arm and become exactly
            // invariant too (they were already ~1e-5 by centring each margin
            // separately; canonicalization makes the swap bit-identical).
            let canon_cols: Vec<usize> = {
                let mut perm: Vec<usize> = (0..dim).collect();
                perm.sort_by_key(|&a| cols[a]);
                if perm.iter().enumerate().any(|(i, &a)| i != a) {
                    margins = perm.iter().map(|&a| margins[a].clone()).collect();
                    emitted_periods = perm.iter().map(|&a| emitted_periods[a]).collect();
                }
                perm.iter().map(|&a| cols[a]).collect()
            };
            let any_periodic = emitted_periods.iter().any(|p| p.is_some());
            let periods_vec = if any_periodic {
                emitted_periods
            } else {
                Vec::new()
            };
            // The tensor's joint polynomial null space is independently
            // shrinkable by default, so REML can recover an unsupported surface
            // as zero. Explicit `double_penalty=false` remains the MLE opt-out.
            let tensor_double_penalty = smooth_double_penalty;
            Ok(SmoothBasisSpec::TensorBSpline {
                feature_cols: canon_cols,
                spec: TensorBSplineSpec {
                    marginalspecs: margins,
                    periods: periods_vec,
                    double_penalty: tensor_double_penalty,
                    identifiability: parse_tensor_identifiability(options, kind)?,
                    // `t2` selects mgcv's separable (Wood, Scheipl & Faraway
                    // 2013) decomposition. It can arrive either as the `t2(...)`
                    // function form (`SmoothKind::T2`) or as a `type="t2"` /
                    // `bs="t2"` option on an `s(...)`/`te(...)` term, in which
                    // case `kind` is *not* `T2` but the resolved type string is
                    // "t2". Keying only off `kind` silently aliased the option
                    // form to `te`'s Kronecker-sum penalty (gam#1185); key off
                    // the resolved type string as well so both routes build the
                    // separable penalty.
                    penalty_decomposition: if matches!(kind, SmoothKind::T2)
                        || type_opt.as_str() == "t2"
                    {
                        TensorBSplinePenaltyDecomposition::Separable
                    } else {
                        TensorBSplinePenaltyDecomposition::MarginalKroneckerSum
                    },
                },
            })
        }
        "pca" => {
            validate_known_options("pca", options, PCA_SMOOTH_OPTION_KEYS)?;
            let path = options
                .get("lazy_path")
                .or_else(|| options.get("pca_basis_path"))
                .or_else(|| options.get("path"))
                .map(|raw| PathBuf::from(strip_quotes(raw)));
            let Some(path) = path else {
                return Err(TermBuilderError::incompatible_config(
                    "pca smooth requires lazy_path=... on the formula path",
                )
                .to_string());
            };
            let k = option_usize_any(options, &["k", "basis_dim", "basis-dim", "basisdim"])
                .unwrap_or(0);
            let chunk_size = option_usize(options, "chunk_size").unwrap_or(DEFAULT_PCA_CHUNK_SIZE);
            Ok(SmoothBasisSpec::Pca {
                feature_cols: cols.to_vec(),
                basis_matrix: Array2::<f64>::zeros((cols.len(), k)),
                centered: option_bool(options, "centered").unwrap_or(true),
                smooth_penalty: option_f64(options, "smooth_penalty").unwrap_or(1.0),
                center_mean: None,
                pca_basis_path: Some(path),
                chunk_size,
            })
        }
        other => Err(TermBuilderError::unsupported_feature(format!(
            "unsupported smooth type '{other}'"
        ))
        .to_string()),
    }
}

/// Initialise per-axis anisotropic log-scales on eligible spatial smooth specs.
pub fn enable_scale_dimensions(spec: &mut TermCollectionSpec) {
    for smooth in spec.smooth_terms.iter_mut() {
        // A multi-axis thin-plate term cannot carry per-axis anisotropy on its
        // single curvature penalty, so `scale_dimensions` was historically a
        // silent no-op for `bs="tp"` (gam#1676). Rewrite it to the
        // mathematically-equivalent anisotropic s=0 Duchon spline first; the
        // Duchon arm below then sees an already-seeded `aniso_log_scales` and
        // leaves it untouched.
        promote_thin_plate_for_scale_dimensions(&mut smooth.basis);
        match &mut smooth.basis {
            SmoothBasisSpec::Matern {
                feature_cols,
                spec: matern,
                ..
            } => {
                if matern.aniso_log_scales.is_none() {
                    let d = feature_cols.len();
                    matern.aniso_log_scales = Some(vec![0.0; d]);
                }
            }
            SmoothBasisSpec::Duchon {
                feature_cols,
                spec: duchon,
                ..
            } => {
                if duchon.aniso_log_scales.is_none() {
                    let d = feature_cols.len();
                    duchon.aniso_log_scales = Some(vec![0.0; d]);
                }
            }
            // Bases with no per-axis length-scale vector to seed: either
            // single-axis, factor-indexed, or (tensor) already anisotropic by
            // construction through their marginals. Enumerated rather than
            // wildcarded so a new basis kind has to answer this question.
            SmoothBasisSpec::ByVariable { .. }
            | SmoothBasisSpec::FactorSumToZero { .. }
            | SmoothBasisSpec::BSpline1D { .. }
            | SmoothBasisSpec::BySmooth { .. }
            | SmoothBasisSpec::FactorSmooth { .. }
            | SmoothBasisSpec::ThinPlate { .. }
            | SmoothBasisSpec::Sphere { .. }
            | SmoothBasisSpec::ConstantCurvature { .. }
            | SmoothBasisSpec::MeasureJet { .. }
            | SmoothBasisSpec::Pca { .. }
            | SmoothBasisSpec::TensorBSpline { .. } => {}
        }
    }
}

/// Rewrite a multi-axis thin-plate term into the mathematically-equivalent
/// anisotropic s=0 Duchon spline so that `scale_dimensions` genuinely engages
/// (gam#1676).
///
/// ## Why a rewrite rather than a new field on the TPS builder
///
/// A canonical thin-plate regression spline carries a *single* curvature
/// penalty — the exact `∫|Dᵐ f|²` reproducing-kernel Gram. That penalty has no
/// per-axis structure to make one direction more or less relevant than another,
/// so per-axis anisotropy (`scale_dimensions`) cannot be expressed on it. The
/// flag was therefore a silent no-op for `bs="tp"` while it engaged for
/// `duchon()`/`matern()`.
///
/// The thin-plate kernel `r^{2m−d}` (the `r²·log r` log-case in even `d`) is
/// *exactly* the s=0 Duchon kernel (`DuchonBasisSpec::power = 0`,
/// `length_scale = None`) at the matching polynomial null-space order
/// `m = thin_plate_penalty_order(d)`. The Duchon polyharmonic family already
/// carries the per-axis tension ARD that `scale_dimensions` requests: its
/// isotropic first-order roughness penalty `Σ‖∇f‖²` splits into `d` directional
/// penalties `Σ(∂f/∂x_a)²`, each with its own REML `λ_a`
/// (`duchon_operator_penalty_candidates`). So the well-posed *anisotropic
/// thin-plate spline is the anisotropic s=0 Duchon spline*. Rewriting to that
/// representation reuses the battle-tested Duchon anisotropy / ψ-derivative /
/// freeze / predict machinery instead of duplicating it onto the TPS metadata
/// path, and keeps the polyharmonic family internally consistent. The codebase
/// already promotes infeasible-`k` TPS to Duchon for the same reason (the
/// canonical TPS single curvature penalty cannot deliver a requested
/// capability); per-axis anisotropy is another such capability.
///
/// This fires *only* when the user opts into `scale_dimensions`; the default
/// thin-plate path (`scale_dimensions` off) is left bit-for-bit unchanged.
/// A 1-D thin-plate term is left untouched — anisotropy is meaningless on a
/// single axis (its `Σ η = 0` contrast vector is empty), exactly as for a 1-D
/// Matérn/Duchon term.
fn promote_thin_plate_for_scale_dimensions(basis: &mut SmoothBasisSpec) {
    let SmoothBasisSpec::ThinPlate {
        feature_cols,
        spec,
        input_scale,
    } = &*basis
    else {
        return;
    };
    let d = feature_cols.len();
    if d <= 1 {
        return;
    }
    // m = thin_plate_penalty_order(d) is the TPS penalty order; the Duchon
    // null-space order naming is `Zero → m=1`, `Linear → m=2`,
    // `Degree(g) → m=g+1`, so the s=0 Duchon kernel exponent
    // `2(p+s) − d = 2m − d` reproduces the TPS kernel exactly.
    let m = thin_plate_penalty_order(d);
    let nullspace_order = match m {
        0 | 1 => DuchonNullspaceOrder::Zero,
        2 => DuchonNullspaceOrder::Linear,
        _ => DuchonNullspaceOrder::Degree(m - 1),
    };
    let duchon_spec = DuchonBasisSpec {
        center_strategy: spec.center_strategy.clone(),
        periodic: spec.periodic.clone(),
        // Pure, scale-free Duchon — the thin-plate kernel has no length scale
        // (a global TPS kernel scale is non-identifiable once REML learns the
        // smoothing penalty: gam#718/#721/#731/#732). The per-axis relevance
        // the user asked for is carried by the tension-ARD `λ_a`, not a κ axis.
        length_scale: None,
        // s = 0  ⇒  thin-plate kernel `r^{2m−d}`.
        power: 0.0,
        nullspace_order,
        identifiability: spec.identifiability.clone(),
        // All-zero geometry seed sentinel: `auto_seed_aniso_contrasts` resolves
        // it from the (standardized) knot cloud, and the per-axis tension split
        // engages on `aniso.is_some()`.
        aniso_log_scales: Some(vec![0.0; d]),
        operator_penalties: DuchonOperatorPenaltySpec::default(),
        boundary: OneDimensionalBoundary::Open,
        radial_reparam: None,
    };
    let feature_cols = feature_cols.clone();
    let input_scale = *input_scale;
    // All borrows of `*basis` (the `&*basis` destructure above) end with the
    // clones on the two preceding lines, so the reassignment is sound.
    *basis = SmoothBasisSpec::Duchon {
        feature_cols,
        spec: duchon_spec,
        input_scale,
    };
}

// ---------------------------------------------------------------------------
// Data-aware helpers
// ---------------------------------------------------------------------------

pub fn spatial_center_strategy_for_dimension(num_centers: usize, d: usize) -> CenterStrategy {
    if d <= 3 {
        // In low-dimensional spatial smooths, an explicit `k` is a resolution
        // request rather than a request for marginal quantile-midpoint centers.
        // Use deterministic maximin geometry so Matérn/GP and Duchon REML see a
        // well-resolved native kernel block with small fill distance instead of
        // compensating for holes or endpoint under-resolution by over-smoothing
        // low-noise signals (#504).
        CenterStrategy::FarthestPoint { num_centers }
    } else {
        default_spatial_center_strategy(num_centers, d)
    }
}

/// Center geometry for a non-periodic Duchon smooth.
///
/// In one dimension the represented domain is the interval between the observed
/// extrema.  Equally spaced centers are the exact minimax design for that
/// interval: among all `k`-point center sets they minimize the largest uncovered
/// gap.  Greedy farthest-point sampling instead produces a dyadic mesh whose
/// partially filled final level clusters centers and leaves wider holes whenever
/// `k` is not a power-of-two refinement.  Those holes reduce the effective
/// resolution of an explicit `k` and caused the low-noise k=20 Duchon fit to miss
/// the mature-smoother accuracy bar despite having the same basis dimension.
///
/// Multidimensional Duchon terms keep the rotation-equivariant farthest-point /
/// equal-mass strategies, where there is no canonical coordinate-aligned grid.
/// The `Auto` wrapper is retained for inferred 1-D counts so adaptive resolution
/// can still resize the interval grid before freezing its realized centers.
fn duchon_center_strategy(num_centers: usize, d: usize, automatic: bool) -> CenterStrategy {
    let realized = if d == 1 {
        CenterStrategy::UniformGrid {
            points_per_dim: num_centers,
        }
    } else {
        spatial_center_strategy_for_dimension(num_centers, d)
    };
    if automatic {
        CenterStrategy::Auto(Box::new(realized))
    } else {
        realized
    }
}

pub fn col_minmax(col: ArrayView1<'_, f64>) -> Result<(f64, f64), String> {
    let min = col.iter().fold(f64::INFINITY, |a, &b| a.min(b));
    let max = col.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
    if !min.is_finite() || !max.is_finite() {
        return Err(TermBuilderError::degenerate_data(
            "non-finite data encountered while inferring knot range",
        )
        .to_string());
    }
    if (max - min).abs() < 1e-12 {
        Ok((min, min + 1e-6))
    } else {
        Ok((min, max))
    }
}

pub fn unique_count_column(col: ArrayView1<'_, f64>) -> usize {
    use std::collections::HashSet;
    let mut set = HashSet::<u64>::with_capacity(col.len());
    for &v in col {
        set.insert(gam_data::canonical_level_bits(v));
    }
    set.len().max(1)
}

/// Minimum knot count for a natural cubic regression spline: `select_cr_knots`
/// places one value-knot per basis function and needs at least an interior knot,
/// so the sparsest representable cr basis is `{const, linear, curvature}` at
/// three knots. Below this a cr spline is not constructible and the caller must
/// degrade to the linear B-spline marginal.
pub(crate) const CR_MIN_KNOTS: usize = 3;

/// Build a cubic-regression marginal knot spec capped to the covariate's data
/// support, mgcv-style.
///
/// A `cr`/`cs`/`sz` marginal places exactly one basis function per value-knot,
/// so `select_cr_knots` cannot place more knots than the covariate has DISTINCT
/// values — it `bail`s with "cubic regression spline with k=N requires at least
/// N distinct values" otherwise. An unclamped `k` on an ordinary low-cardinality
/// covariate (a binary indicator, a 3-level ordinal/Likert score, a small count)
/// therefore hard-failed the whole fit instead of reducing the basis the way
/// mgcv — and gam's own tensor-margin path (996f829d7, `term_builder.rs:2986` /
/// the `k_axis >= 3` cr gate at `:3047`) — do. This is the univariate / factor-
/// smooth sibling of that tensor cap (#1541, #1542).
///
/// Returns:
/// - `Some(NaturalCubicRegression { .. })` with `k = min(k_requested, n_distinct)`
///   value-knots when the data supports a cr spline (`n_distinct >= CR_MIN_KNOTS`).
///   A cr basis of exactly `n_distinct` knots is full-rank for the data — it can
///   represent any per-distinct-value structure (e.g. 3 arbitrary group means on
///   a ternary covariate) — so the cap never costs recoverable signal.
/// - `None` when `n_distinct < CR_MIN_KNOTS` (a binary covariate): too few
///   distinct values for ANY cr spline, so the caller degrades to the linear
///   B-spline marginal — exactly what the default `s(x, k=..)` basis already
///   builds on the same data, and what the tensor path's `< 3` branch builds.
///
/// `inference_notes` records any reduction so the user sees that `k` was capped
/// (mgcv emits a warning in the same situation).
fn capped_cr_marginal_knotspec(
    col: ArrayView1<'_, f64>,
    k_cr_requested: usize,
    label: &str,
    inference_notes: &mut Vec<String>,
) -> Result<Option<BSplineKnotSpec>, String> {
    let n_distinct = unique_count_column(col);
    let k_cr = k_cr_requested.min(n_distinct);
    if k_cr < CR_MIN_KNOTS {
        inference_notes.push(format!(
            "Smooth '{label}': cubic-regression ('cr'/'cs'/'sz') basis requested k={k_cr_requested}, \
             but the covariate has only {n_distinct} distinct value(s) — too few to support a cubic \
             regression spline (needs >= {CR_MIN_KNOTS} distinct values). Degraded to the linear \
             B-spline marginal the default basis builds on the same data."
        ));
        return Ok(None);
    }
    if k_cr < k_cr_requested {
        inference_notes.push(format!(
            "Smooth '{label}': cubic-regression ('cr'/'cs'/'sz') basis reduced from k={k_cr_requested} \
             to k={k_cr} to match the covariate's {n_distinct} distinct value(s) (mgcv-style \
             data-support cap; a cr basis cannot place more value-knots than the data has)."
        ));
    }
    let cr_knots = crate::basis::select_cr_knots(col, k_cr).map_err(|e| e.to_string())?;
    Ok(Some(BSplineKnotSpec::NaturalCubicRegression {
        knots: cr_knots,
    }))
}

/// Smallest number of distinct covariate values seen within any single group
/// of `group_col`. For a factor smooth this is the resolution that bounds the
/// marginal basis: a group with `m` distinct covariate values can only inform
/// `m` basis coefficients, so a marginal richer than that interpolates the
/// group instead of estimating a penalized trend. Bits are compared exactly so
/// integer-valued covariates (days, dose levels) collapse to their true count.
fn min_per_group_unique_count(
    feature_col: ArrayView1<'_, f64>,
    group_col: ArrayView1<'_, f64>,
) -> usize {
    use std::collections::{HashMap, HashSet};
    let mut per_group: HashMap<u64, HashSet<u64>> = HashMap::new();
    for (xi, gi) in feature_col.iter().zip(group_col.iter()) {
        per_group
            .entry(gam_data::canonical_level_bits(*gi))
            .or_default()
            .insert(gam_data::canonical_level_bits(*xi));
    }
    per_group
        .values()
        .map(|s| s.len())
        .min()
        .unwrap_or(1)
        .max(1)
}

/// Cap on the automatically inferred internal-knot count of a 1-D smooth.
/// Default cubic basis ≈ `MAX_DEFAULT_INTERNAL_KNOTS + degree + 1` = 12
/// functions, matching mgcv's lean univariate default. Named at module level
/// so the inference note reports the ceiling the engine applies rather than
/// one of its own.
pub(crate) const MAX_DEFAULT_INTERNAL_KNOTS: usize = 8;

/// Default internal-knot count for an *additive* univariate smooth, derived
/// from the column's unique-value count.
///
/// The basis dimension is `internal_knots + degree + 1`, so the cap below maps
/// to a default cubic basis of ~12 functions — deliberately close to mgcv's
/// univariate default (`k = 10`). A penalized smooth controls its wiggliness
/// through the *penalty*, not the basis size: REML/LAML shrinks a too-rich
/// basis toward the null, but it cannot do so cleanly when the basis is so
/// over-sized that the design becomes weakly identified. Growing the basis with
/// `n` (the old `n^(1/3)`-ceilinged `unique/4` rule, which pinned to 20 internal
/// knots ⇒ a 24-function basis for any column with ≥80 unique values) therefore
/// *hurts* recovery on finite, weak-signal fits: a 4-smooth additive model on
/// n=120 asks for ~92 coefficients, the outer optimizer stalls on the resulting
/// flat two-penalty (range + null-space) REML surface, and the truth leaks into
/// surplus columns the penalty can't shrink away (gam#1680; the same defect was
/// documented for thin-plate fields in gam#1074). A k-sweep on the #1680 design
/// confirms a basis of ~10–15 recovers truth at RMSE ≈ 0.12 while the old
/// 24-function default lands at ≈ 0.39 (~3× worse) — *whether or not* the
/// covariates are collinear, so this is basis over-richness, not collinearity.
///
/// The cap is flat in `n`: a user who genuinely needs a wigglier fit raises `k`
/// explicitly (mgcv's contract — opt *in* to more flexibility), and the SPEC
/// requires the default to allow recovering the null rather than forcing the
/// user to opt out of overfitting. The 4-knot floor stays put because we still
/// need enough basis functions to fit a non-trivial smooth at all, and the
/// `unique/4` growth below the cap keeps small/sparse columns (n ≤ 32, where
/// `unique/4 ≤ 8`) on exactly their previous knot count.
pub fn heuristic_knots_for_column(col: ArrayView1<'_, f64>) -> usize {
    let unique = unique_count_column(col);
    (unique / 4).clamp(4, MAX_DEFAULT_INTERNAL_KNOTS)
}

/// Per-margin basis sizes for a tensor-product smooth (`te`/`ti`/`t2`).
///
/// The 1-D heuristic [`heuristic_knots_for_column`] is calibrated for an
/// *additive* margin: a well-resolved column asks for the lean univariate
/// default (≈12 basis functions, the mgcv-like cap of 8 internal knots; see
/// gam#1680), which is sensible for a single `s(x)` term.
/// A tensor product, however, multiplies the per-margin sizes:
/// `p = ∏_d k_d`. Reusing the 1-D rule per margin makes `p` explode with the
/// tensor dimension — a 3-D `te(x,y,z)` at the 1-D ceiling of 12/margin is
/// `12³ ≈ 1728` columns, and every REML evaluation pays an O(p³) dense
/// penalty reparameterization (the full-tensor sum-to-zero constraint is not
/// Kronecker-factorable), turning model selection over tensor candidates into
/// a multi-minute single-threaded stall (gam#813). It also requests far more
/// coefficients than the data can identify whenever `p ≫ n`.
///
/// mgcv's `te(...)` uses a small per-margin default (`k = 5`, i.e. `5^d`).
/// We match that spirit while staying data-adaptive: budget the *total* tensor
/// column count `p_target` and distribute it geometrically across the margins
/// so `∏ k_d ≈ p_target`, never asking a margin for more functions than its
/// own unique values (and the data set) can support.
fn heuristic_tensor_margin_knots(cols: &[usize], ds: &Dataset) -> Vec<usize> {
    let d = cols.len().max(1);
    let degree = DEFAULT_BSPLINE_DEGREE;
    let min_k = degree + 2; // smallest margin that carries a difference penalty
    let n = ds.values.nrows();

    // Per-margin 1-D ceiling: never request more basis functions than the
    // margin's own resolution (unique values) supports. This caps each axis
    // independently before the joint budget is applied.
    let per_margin_cap: Vec<usize> = cols
        .iter()
        .map(|&c| heuristic_knots_for_column(ds.values.column(c)).max(min_k))
        .collect();

    // Total-basis budget. A tensor with ∏k ≫ n coefficients is rank-deficient
    // and pure REML cost; cap the product at a generous fraction of n while
    // honoring mgcv's small default for the common small-d case. The budget
    // grows with n but the geometric split below keeps each margin modest.
    //   d=2 → up to ~7²=49 (mgcv-`te`-like), d=3 → ~5³=125, larger d shrinks
    // per-margin further so the product never blows past the data support.
    let mgcv_like_per_margin = match d {
        2 => 7usize,
        3 => 5usize,
        _ => 4usize,
    };
    let mgcv_like_total = (mgcv_like_per_margin as f64).powi(d as i32);
    let data_budget = (n as f64) * 0.8;
    let p_target = mgcv_like_total
        .max(min_k.pow(d as u32) as f64)
        .min(data_budget);

    // Geometric per-margin target so ∏k ≈ p_target, then clamp each margin to
    // its own 1-D resolution cap and the difference-penalty floor.
    let geo_per_margin = p_target.powf(1.0 / d as f64).round() as usize;
    let unclamped: Vec<usize> = per_margin_cap
        .iter()
        .map(|&cap| geo_per_margin.clamp(min_k, cap))
        .collect();

    // The per-margin clamps can pull some axes below `geo_per_margin` (a
    // low-resolution column), leaving headroom in the joint budget. Redistribute
    // that headroom to the margins that can still grow, so the realized ∏k stays
    // close to p_target instead of systematically under-shooting it.
    let mut k_list = unclamped;
    loop {
        let product: f64 = k_list.iter().map(|&k| k as f64).product();
        if product >= p_target {
            break;
        }
        // Grow the axis with the most remaining headroom (cap − current),
        // breaking ties toward the largest cap. Stop when none can grow.
        let Some(idx) = k_list
            .iter()
            .zip(per_margin_cap.iter())
            .enumerate()
            .filter(|&(_, (k, cap))| k < cap)
            .max_by_key(|&(_, (k, cap))| (cap - k, *cap))
            .map(|(i, _)| i)
        else {
            break;
        };
        k_list[idx] += 1;
    }
    k_list
}

// ---------------------------------------------------------------------------
// Smooth option parsers
// ---------------------------------------------------------------------------

fn parse_endpoint_side(
    value: &str,
    context: &str,
) -> Result<BSplineEndpointBoundaryCondition, String> {
    match value.trim().to_ascii_lowercase().as_str() {
        "" | "none" | "open" | "unconstrained" | "free" => {
            Ok(BSplineEndpointBoundaryCondition::Free)
        }
        "clamped" | "clamp" | "zero_derivative" | "zero-derivative" => {
            Ok(BSplineEndpointBoundaryCondition::Clamped)
        }
        "anchored" | "anchor" | "zero" | "zero_value" | "zero-value" => {
            Ok(BSplineEndpointBoundaryCondition::Anchored { value: 0.0 })
        }
        other => Err(format!(
            "unsupported {context} boundary condition '{other}'; expected free, clamped, or anchored"
        )),
    }
}

fn boundary_anchor_value(
    options: &BTreeMap<String, String>,
    side: &str,
    fallback: Option<f64>,
) -> Option<f64> {
    [
        format!("anchor_{side}"),
        format!("{side}_anchor"),
        format!("anchor-value-{side}"),
    ]
    .iter()
    .find_map(|key| option_f64(options, key))
    .or(fallback)
}

fn apply_anchor_value(
    cond: BSplineEndpointBoundaryCondition,
    value: Option<f64>,
) -> BSplineEndpointBoundaryCondition {
    match cond {
        BSplineEndpointBoundaryCondition::Anchored { .. } => {
            BSplineEndpointBoundaryCondition::Anchored {
                value: value.unwrap_or(0.0),
            }
        }
        other => other,
    }
}

fn parse_bspline_boundary_conditions(
    options: &BTreeMap<String, String>,
) -> Result<BSplineBoundaryConditions, String> {
    let fallback_anchor = option_f64(options, "anchor")
        .or_else(|| option_f64(options, "anchor_value"))
        .or_else(|| option_f64(options, "value"));
    // `boundary` is whitelisted on this arm as the third spelling of `bc` /
    // `boundary_conditions` and was read by NEITHER of the two functions that
    // consume the option (`parse_periodic_axes` reads it, but only for the
    // periodic tokens), so `s(x, boundary=clamped)` was accepted and inert
    // (#2781's family). A periodic token never reaches here: the arm skips this
    // function entirely once `bspline_boundary_declares_periodic_axis` fires.
    let global_boundary_conditions = options
        .get("boundary_conditions")
        .or_else(|| options.get("bc"))
        .or_else(|| options.get("boundary"));
    let mut boundary_conditions = BSplineBoundaryConditions::default();

    if let Some(raw_boundary_conditions) = global_boundary_conditions {
        let cond = parse_endpoint_side(raw_boundary_conditions, "boundary_conditions")?;
        let side = options
            .get("side")
            .map(|s| s.trim().to_ascii_lowercase())
            .unwrap_or_else(|| "both".to_string());
        match side.as_str() {
            "both" | "all" | "endpoints" => {
                boundary_conditions.left = cond;
                boundary_conditions.right = cond;
            }
            "left" | "start" | "lower" => boundary_conditions.left = cond,
            "right" | "end" | "upper" => boundary_conditions.right = cond,
            other => {
                return Err(format!(
                    "unsupported B-spline boundary side '{other}'; expected left, right, or both"
                ));
            }
        }
    }

    if let Some(raw) = options
        .get("bc_left")
        .or_else(|| options.get("left_bc"))
        .or_else(|| options.get("bc_start"))
        .or_else(|| options.get("start_bc"))
    {
        boundary_conditions.left = parse_endpoint_side(raw, "left endpoint")?;
    }
    if let Some(raw) = options
        .get("bc_right")
        .or_else(|| options.get("right_bc"))
        .or_else(|| options.get("bc_end"))
        .or_else(|| options.get("end_bc"))
    {
        boundary_conditions.right = parse_endpoint_side(raw, "right endpoint")?;
    }

    boundary_conditions.left = apply_anchor_value(
        boundary_conditions.left,
        boundary_anchor_value(options, "left", fallback_anchor),
    );
    boundary_conditions.right = apply_anchor_value(
        boundary_conditions.right,
        boundary_anchor_value(options, "right", fallback_anchor),
    );

    // `side=` says WHICH endpoint the global condition applies to, and an
    // anchor value says WHAT an anchored endpoint is pinned to. Neither means
    // anything on its own, and both were previously accepted and discarded, so
    // `s(x, bc_left=anchored, anchor=2.5)` pinned the endpoint at 2.5 while
    // `s(x, anchor=2.5)` silently pinned nothing at all (#2781's family).
    if options.contains_key("side") && global_boundary_conditions.is_none() {
        return Err(TermBuilderError::invalid_option(
            "`side=` selects which endpoint a boundary condition applies to, but this smooth              declares none; add bc=<condition> or drop it",
        )
        .to_string());
    }
    if !boundary_conditions.has_anchor()
        && let Some(key) = ANCHOR_VALUE_OPTION_KEYS
            .iter()
            .find(|key| options.contains_key(**key))
    {
        return Err(TermBuilderError::invalid_option(format!(
            "`{key}=` sets the value an ANCHORED endpoint is pinned to, but no endpoint of this              smooth is anchored; add bc=anchored (or bc_left=/bc_right=anchored) or drop it"
        ))
        .to_string());
    }

    Ok(boundary_conditions)
}

/// Option keys that carry the value an anchored endpoint is pinned to. Each is
/// meaningless without an `anchored` endpoint to attach it to.
const ANCHOR_VALUE_OPTION_KEYS: [&str; 8] = [
    "anchor",
    "anchor_value",
    "value",
    "anchor_left",
    "left_anchor",
    "anchor_right",
    "right_anchor",
    "anchor-value-left",
];

/// Resolve the requested internal-knot count and effective spline degree for
/// a 1-D penalized B-spline smooth. This mirrors the tensor-margin per-axis
/// degree-reduction policy: a 1-D B-spline basis with `k` functions
/// is well-defined for any `degree <= k - 1`, so an explicit
/// `s(x, bs="ps", k=3)` with default `degree=3` is interpreted as the
/// largest representable spline (`effective_degree = k - 1 = 2`, quadratic)
/// rather than rejected. The `penalty_order` carried by the caller must be
/// clamped to `<= effective_degree` so the marginal roughness penalty
/// stays well-defined; the returned `effective_degree` makes that explicit.
///
/// An explicit `k` is honoured EXACTLY: a clamped degree-`d` B-spline with
/// `m` internal knots has `m + d + 1` functions, so the returned count is
/// `k - effective_degree - 1`, zero included. `s(x, k=4)` therefore names the
/// same four-function cubic as `s(x, knots=0)` — the documented identity
/// `k = internal_knots + degree + 1` (docs/formulas.md) holds for every `k`.
///
/// Mirrors the tensor margin treatment in the `te(...)` builder so a
/// standalone smooth, a factor smooth, and a tensor margin all interpret
/// "small k" the same way.
fn parse_ps_internal_knots(
    options: &BTreeMap<String, String>,
    degree: usize,
    default_internal_knots: usize,
) -> Result<(usize, bool, usize), String> {
    // Strict variants: reject `k=-1`, `k=1.5`, `knots=-2` etc. with a
    // focused error instead of silently dropping the value and using the
    // default. Lenient `option_usize` / `option_usize_any` silently swallow
    // unparseable values, which leaves the user thinking they configured
    // something when they did not.
    // A list-valued `knots=[...]` carries explicit internal positions, not a
    // count; it is consumed by `parse_explicit_internal_knots`. Treat it as
    // "count not specified" here so the strict integer parse does not reject
    // the bracketed value (the Provided path ignores the returned count).
    let knots_internal = if knots_option_is_list(options) {
        None
    } else {
        option_usize_strict(options, "knots")?
    };
    let basis_dim = option_usize_any_strict(options, &["k", "basis_dim", "basis-dim", "basisdim"])?;
    if knots_internal.is_some() && basis_dim.is_some() {
        return Err(TermBuilderError::incompatible_config(
            "ps/bspline smooth: specify either knots=<internal_knots> or k=<basis_dim> (not both)",
        )
        .to_string());
    }
    if let Some(k) = basis_dim {
        if k < 2 {
            return Err(TermBuilderError::invalid_option(format!(
                "ps/bspline smooth: k={} too small; B-spline basis requires k >= 2",
                k
            ))
            .to_string());
        }
        // `degree <= k - 1` is required for the B-spline basis to be
        // well-defined; reduce on this axis only when the user asked for
        // a smaller k than the cubic default supports. This matches mgcv's
        // behaviour (e.g. `s(x, bs="ps", k=3)` becomes a quadratic basis)
        // and the per-axis reduction the tensor builder already does.
        let effective_degree = degree.min(k - 1).max(1);
        // `k >= 2` and `effective_degree <= k - 1`, so this cannot underflow.
        // A floor of two internal knots used to sit on the un-reduced branch
        // and silently turned `k=4` and `k=5` into the six-function cubic, so
        // three requested dimensions collapsed onto one bit-identical fit
        // while the `knots=` spelling of the same bases was honoured exactly.
        // Zero internal knots is a valid basis (a single polynomial piece).
        let num_internal_knots = k - effective_degree - 1;
        Ok((num_internal_knots, false, effective_degree))
    } else {
        Ok((
            knots_internal.unwrap_or(default_internal_knots),
            knots_internal.is_none(),
            degree,
        ))
    }
}

/// True when the `knots` option value is a *list* literal (`[...]`, `c(...)`,
/// or `(...)`) rather than a scalar count. mgcv's `knots=` accepts both: a
/// single integer is an internal-knot count, while a vector is explicit
/// internal knot positions. We disambiguate purely on the wrapper syntax so a
/// bare `knots=5` keeps its historical count meaning.
fn knots_option_is_list(options: &BTreeMap<String, String>) -> bool {
    options
        .get("knots")
        .map(|raw| {
            let t = raw.trim();
            t.starts_with('[') || t.starts_with("c(") || t.starts_with("C(") || t.starts_with('(')
        })
        .unwrap_or(false)
}

/// Parse `knots=[k0, k1, ...]` (or `c(...)` / `(...)`) into explicit internal
/// knot positions. Returns `Ok(None)` when `knots` is absent or a scalar count
/// (handled by [`parse_ps_internal_knots`]); `Ok(Some(positions))` when it is a
/// non-empty numeric list; and an error for an empty or unparseable list.
fn parse_explicit_internal_knots(
    options: &BTreeMap<String, String>,
) -> Result<Option<Vec<f64>>, String> {
    if !knots_option_is_list(options) {
        return Ok(None);
    }
    let raw = options
        .get("knots")
        .expect("knots_option_is_list implies the key is present");
    let tokens = split_list_option(raw);
    if tokens.is_empty() {
        return Err(TermBuilderError::invalid_option(format!(
            "knots={raw} is an empty list; supply at least one internal knot position \
             (e.g. knots=[0.2, 0.5, 0.8]) or a scalar count (e.g. knots=8)"
        ))
        .to_string());
    }
    let mut positions = Vec::with_capacity(tokens.len());
    for tok in &tokens {
        let value = parse_numeric_expr(tok).map_err(|err| {
            TermBuilderError::invalid_option(format!(
                "knots list entry '{tok}' is not a numeric position: {err}"
            ))
            .to_string()
        })?;
        positions.push(value);
    }
    Ok(Some(positions))
}

/// Resolve the `knot_placement=` option for an automatically generated knot
/// vector. Accepts `"uniform"` (the default, equal spacing on the data range)
/// and `"quantile"` (interior knots at empirical data quantiles, better for
/// skewed covariates). Unknown values are rejected so typos do not silently
/// fall back to uniform.
/// Parse a per-margin unsigned-integer tensor option (`degree=`,
/// `penalty_order=`).
///
/// Accepts the scalar form (`degree=2`), which broadcasts to every margin as
/// `docs/formulas.md` promises ("Margins requested as a single value are
/// broadcast across all margins"), and the per-margin list form
/// (`degree=[1, 3]`, `degree=c(1, 3)`), with `none` selecting the default on
/// that margin. Returns `None` per axis when the caller said nothing, so the
/// margin loop can tell "asked for the default" apart from "asked for a value
/// that happens to equal the default" — a distinction the cr-margin routing
/// below depends on.
///
/// Before #2782 both options were read with `option_usize`, which parses only a
/// bare integer: a list form silently fell back to the default, so
/// `te(x, z, degree=[1,3])` was bit-identical to `te(x, z)`.
fn parse_tensor_per_axis_usize(
    options: &BTreeMap<String, String>,
    key: &str,
    dim: usize,
) -> Result<Vec<Option<usize>>, String> {
    let Some(raw) = options.get(key) else {
        return Ok(vec![None; dim]);
    };
    let values = split_list_option(raw);
    let parse_one = |value: &str| -> Result<Option<usize>, String> {
        let trimmed = value.trim().trim_matches('"').trim_matches('\'').trim();
        if trimmed.is_empty() || trimmed.eq_ignore_ascii_case("none") {
            return Ok(None);
        }
        trimmed.parse::<usize>().map(Some).map_err(|err| {
            TermBuilderError::invalid_option(format!(
                "tensor smooth `{key}={raw}`: '{trimmed}' is not a non-negative integer ({err})"
            ))
            .to_string()
        })
    };
    if values.len() == 1 {
        let shared = parse_one(&values[0])?;
        return Ok(vec![shared; dim]);
    }
    if values.len() != dim {
        return Err(TermBuilderError::invalid_option(format!(
            "tensor smooth `{key}={raw}` has {} entries but the smooth has {dim} margins; pass one \
             value per margin or a single value for all of them",
            values.len()
        ))
        .to_string());
    }
    values.iter().map(|value| parse_one(value)).collect()
}

/// The polynomial degree of the natural cubic regression margin. It is not a
/// parameter of that basis — a "cubic regression spline" IS cubic — so a margin
/// that asks for any other degree cannot be realized as one.
const CR_MARGIN_DEGREE: usize = 3;

/// The derivative order the natural cubic regression penalty integrates. Like
/// [`CR_MARGIN_DEGREE`], this is definitional rather than adjustable: the cr
/// penalty is the exact integrated squared SECOND derivative of the
/// interpolating cubic.
const CR_MARGIN_PENALTY_ORDER: usize = 2;

fn parse_knot_placement(
    options: &BTreeMap<String, String>,
) -> Result<crate::basis::BSplineKnotPlacement, String> {
    use crate::basis::BSplineKnotPlacement;
    match options
        .get("knot_placement")
        .or_else(|| options.get("knot-placement"))
        .or_else(|| options.get("knotplacement"))
    {
        None => Ok(BSplineKnotPlacement::Uniform),
        Some(raw) => match raw
            .trim()
            .trim_matches('"')
            .trim_matches('\'')
            .to_ascii_lowercase()
            .as_str()
        {
            "uniform" | "even" | "equal" => Ok(BSplineKnotPlacement::Uniform),
            "quantile" | "quantiles" | "data" | "empirical" => Ok(BSplineKnotPlacement::Quantile),
            other => Err(TermBuilderError::invalid_option(format!(
                "knot_placement={other} is not recognised; expected \"uniform\" or \"quantile\""
            ))
            .to_string()),
        },
    }
}

/// Like [`parse_knot_placement`] but distinguishes "unset" from an explicit
/// `knot_placement=uniform`.
///
/// The two are not the same request on a tensor margin: unset means "give me
/// mgcv's default margin", which is a natural cubic regression spline on
/// QUANTILE value-knots, while an explicit `uniform` asks for evenly spaced
/// knots — something the cr margin cannot do. Collapsing them made
/// `te(x, z, knot_placement='uniform')` a silent no-op that returned
/// quantile-placed knots (#2782).
fn explicit_knot_placement(
    options: &BTreeMap<String, String>,
) -> Result<Option<crate::basis::BSplineKnotPlacement>, String> {
    let declared = ["knot_placement", "knot-placement", "knotplacement"]
        .iter()
        .any(|key| options.contains_key(*key));
    if !declared {
        return Ok(None);
    }
    parse_knot_placement(options).map(Some)
}

/// Build the non-periodic 1D B-spline knot spec for the `ps`/`bspline` and
/// factor-smooth marginal paths, honoring (in priority order):
///   1. `knots=[...]` explicit internal positions  → [`BSplineKnotSpec::Provided`]
///   2. `knot_placement="quantile"`                 → [`BSplineKnotSpec::Automatic`]
///   3. uniform generation                          → [`BSplineKnotSpec::Generate`]
///
/// `data` is the covariate column (used to clamp explicit positions to the
/// observed range and to drive quantile placement); `n_knots` is the resolved
/// internal-knot count from [`parse_ps_internal_knots`] used for the automatic
/// strategies.
fn resolve_nonperiodic_bspline_knotspec(
    options: &BTreeMap<String, String>,
    data: ArrayView1<'_, f64>,
    data_range: (f64, f64),
    degree: usize,
    n_knots: usize,
) -> Result<BSplineKnotSpec, String> {
    use crate::basis::{BSplineKnotPlacement, clamped_knot_vector_from_internal_positions};
    if let Some(positions) = parse_explicit_internal_knots(options)? {
        if option_usize_any_strict(options, &["k", "basis_dim", "basis-dim", "basisdim"])?.is_some()
        {
            return Err(TermBuilderError::incompatible_config(
                "ps/bspline smooth: specify either explicit knots=[...] positions or \
                 k=<basis_dim> (not both); the basis size is fixed by the knot vector",
            )
            .to_string());
        }
        let knots = clamped_knot_vector_from_internal_positions(data_range, &positions, degree)
            .map_err(|e| e.to_string())?;
        return Ok(BSplineKnotSpec::Provided(knots));
    }
    match parse_knot_placement(options)? {
        BSplineKnotPlacement::Uniform => Ok(BSplineKnotSpec::Generate {
            data_range,
            num_internal_knots: n_knots,
        }),
        BSplineKnotPlacement::Quantile => {
            // Validate the column up-front so an unfittable request surfaces a
            // user-correctable error at parse time rather than deep in basis
            // construction. The same data drives the eventual quantile knots.
            crate::basis::auto_knot_vector_1d_quantile(data, n_knots, degree)
                .map_err(|e| e.to_string())?;
            Ok(BSplineKnotSpec::Automatic {
                num_internal_knots: Some(n_knots),
                placement: BSplineKnotPlacement::Quantile,
            })
        }
    }
}

/// Reject unknown option keys with a focused error that names the term and
/// the offending key, plus suggests near-matches from the known-key list.
/// Without this, typos like `lengt_scale=0.1` or `nyu=5/2` are silently
/// dropped, the term uses the default, and the user has no idea why their
/// option had no effect.

// ---------------------------------------------------------------------------
// Per-smooth-kind option whitelists
//
// Hoisted out of the `validate_known_options` call sites so the guard test
// `no_whitelisted_smooth_option_is_accepted_and_inert` can enumerate them.
// `validate_known_options` answers "is this key spelled right?"; that guard
// answers the different question these three lists silently got wrong in
// #2781/#2782/#2783 — "does this key do anything?".
// ---------------------------------------------------------------------------
/// Options of the PENALIZED factor smooths, `bs='fs'` and `bs='sz'`: a shared
/// B-spline marginal replicated once per level of the grouping factor. Every
/// key here shapes that marginal, so every key here reaches the built design.
///
/// `bs='re'` used to share this list even though it builds no spline at all;
/// see [`RANDOM_EFFECT_SMOOTH_OPTION_KEYS`] and #2791.
pub(crate) const FACTOR_SMOOTH_OPTION_KEYS: &[&str] = &[
    "type",
    "bs",
    "k",
    "basis_dim",
    "basis-dim",
    "basisdim",
    "knots",
    "knot_placement",
    "knot-placement",
    "knotplacement",
    "degree",
    "penalty_order",
    "m",
    "double_penalty",
    "ordered",
];

/// Options of `bs='re'`, the PARAMETRIC random intercept + slope.
///
/// `s(x, g, bs='re')` is mgcv's `(1 + x | g)`: the per-level design is the raw
/// line `[1, x − c]` and the penalty is an identity ridge per parametric
/// coordinate. There is no spline marginal, no knot vector and no difference
/// penalty, so none of the basis-shaping keys of
/// [`FACTOR_SMOOTH_OPTION_KEYS`] can be honoured — and until #2791 all ten of
/// them were accepted and silently discarded.
pub(crate) const RANDOM_EFFECT_SMOOTH_OPTION_KEYS: &[&str] = &["type", "bs", "ordered"];

/// The keys `bs='re'` refuses with a reason rather than a bare "unknown
/// option": they are all spelled correctly and all valid on `bs='fs'`, so the
/// user's mistake is the flavour, not the spelling.
const RANDOM_EFFECT_UNSHAPEABLE_OPTION_KEYS: &[&str] = &[
    "k",
    "basis_dim",
    "basis-dim",
    "basisdim",
    "knots",
    "knot_placement",
    "knot-placement",
    "knotplacement",
    "degree",
    "penalty_order",
    "m",
    "double_penalty",
];

/// Validate the option map of a `bs='re'` term.
///
/// Refuses the basis-shaping keys with a message that names the flavour that
/// does honour them, then falls through to the ordinary spelling check.
fn validate_random_effect_smooth_options(
    options: &BTreeMap<String, String>,
) -> Result<(), String> {
    if let Some(key) = RANDOM_EFFECT_UNSHAPEABLE_OPTION_KEYS
        .iter()
        .find(|key| options.contains_key(**key))
    {
        return Err(TermBuilderError::incompatible_config(format!(
            "bs='re' is a parametric random intercept + slope — the per-level line \
             [1, x] under an i.i.d. ridge — not a spline, so it has no basis to shape \
             and `{key}=` cannot be honoured. Use bs='fs' for a penalized random \
             smooth of x within each level (it accepts {key}=), or drop the option."
        ))
        .to_string());
    }
    validate_known_options("re", options, RANDOM_EFFECT_SMOOTH_OPTION_KEYS)
}

pub(crate) const CYCLIC_SMOOTH_OPTION_KEYS: &[&str] = &[
    "type",
    "bs",
    "by",
    "k",
    "basis_dim",
    "basis-dim",
    "basisdim",
    "degree",
    "penalty_order",
    "period",
    "periods",
    "period_start",
    "period_end",
    "start",
    "end",
    "origin",
    "origins",
    "period_origin",
    "period-origin",
    "domain_origin",
    "double_penalty",
    "id",
    "identifiability",
];

pub(crate) const BSPLINE_SMOOTH_OPTION_KEYS: &[&str] = &[
    "type",
    "bs",
    "by",
    "k",
    "basis_dim",
    "basis-dim",
    "basisdim",
    "knots",
    "knot_placement",
    "knot-placement",
    "knotplacement",
    "degree",
    "penalty_order",
    "boundary",
    "bc",
    "boundary_conditions",
    "bc_left",
    "bc_right",
    "left_bc",
    "right_bc",
    "start_bc",
    "end_bc",
    "side",
    "anchor",
    "anchor_value",
    "value",
    "anchor_left",
    "left_anchor",
    "anchor_right",
    "right_anchor",
    "periodic",
    "period",
    "periods",
    "period_start",
    "period_end",
    "origin",
    "double_penalty",
    "id",
    "identifiability",
];

pub(crate) const THINPLATE_SMOOTH_OPTION_KEYS: &[&str] = &[
    "type",
    "bs",
    "by",
    "length_scale",
    "centers",
    "k",
    "basis_dim",
    "basis-dim",
    "basisdim",
    "knots",
    "include_intercept",
    "double_penalty",
    "id",
    "identifiability",
    "periodic",
    "cyclic",
    "period",
    "period_start",
    "period_end",
    "scale_dims",
];

pub(crate) const SPHERE_SMOOTH_OPTION_KEYS: &[&str] = &[
    "type",
    "bs",
    "by",
    "centers",
    "k",
    "basis_dim",
    "basis-dim",
    "basisdim",
    "knots",
    "penalty_order",
    "m",
    "double_penalty",
    "id",
    "kernel",
    "method",
    "radians",
    "units",
    "degree",
    "l",
    "max_degree",
    "max-degree",
    "lmax",
    "l_max",
    "l-max",
];

pub(crate) const CURVATURE_SMOOTH_OPTION_KEYS: &[&str] = &[
    "type",
    "bs",
    "by",
    "centers",
    "k",
    "basis_dim",
    "basis-dim",
    "basisdim",
    "knots",
    "kappa",
    "length_scale",
    "double_penalty",
    "id",
];

pub(crate) const MEASURE_JET_SMOOTH_OPTION_KEYS: &[&str] = &[
    "type",
    "bs",
    "by",
    "centers",
    "k",
    "basis_dim",
    "basis-dim",
    "basisdim",
    "knots",
    "s",
    "alpha",
    "tau",
    "scales",
    "length_scale",
    "double_penalty",
    "multiscale",
    "learn_length_scale",
    "id",
];

pub(crate) const MATERN_SMOOTH_OPTION_KEYS: &[&str] = &[
    "type",
    "bs",
    "by",
    "nu",
    "length_scale",
    "centers",
    "k",
    "basis_dim",
    "basis-dim",
    "basisdim",
    "knots",
    "include_intercept",
    "double_penalty",
    "id",
    "identifiability",
    "periodic",
    "cyclic",
    "period",
    "period_start",
    "period_end",
    "scale_dims",
];

pub(crate) const DUCHON_SMOOTH_OPTION_KEYS: &[&str] = &[
    "type",
    "bs",
    "by",
    "length_scale",
    "centers",
    "k",
    "basis_dim",
    "basis-dim",
    "basisdim",
    "knots",
    "rank",
    "power",
    "p",
    "nullspace_order",
    "order",
    "identifiability",
    "periodic",
    "cyclic",
    "period",
    "period_start",
    "period_end",
    "scale_dims",
    "double_penalty",
    "id",
];

pub(crate) const TENSOR_SMOOTH_OPTION_KEYS: &[&str] = &[
    "type",
    "bs",
    "by",
    "k",
    "basis_dim",
    "basis-dim",
    "basisdim",
    "knot_placement",
    "knot-placement",
    "knotplacement",
    "degree",
    "penalty_order",
    "double_penalty",
    "periodic",
    "cyclic",
    "period",
    "periods",
    "period_start",
    "period_end",
    "origin",
    "origins",
    "period_origin",
    "period-origin",
    "domain_origin",
    "boundary",
    "bc",
    "identifiability",
    "id",
];

pub(crate) const PCA_SMOOTH_OPTION_KEYS: &[&str] = &[
    "type",
    "bs",
    "by",
    "k",
    "basis_dim",
    "basis-dim",
    "basisdim",
    "lazy_path",
    "path",
    "pca_basis_path",
    "chunk_size",
    "smooth_penalty",
    "centered",
    "double_penalty",
    "id",
];

/// Prefix of the engine-injected option namespace. Options the pipeline adds
/// to a term's option map on the user's behalf — the `by=` column index the
/// `BySmooth` wrapper resolves (`__by_col`), the secondary-predictor center cap
/// ([`SECONDARY_CENTER_CAP_OPTION`]) — carry this prefix. They are never typed
/// in a formula, so [`validate_known_options`] does not judge them: that
/// validator answers "is this USER key spelled right?", and an engine key is
/// neither a user key nor a member of any one arm's vocabulary. Listing
/// `__by_col` in every arm's whitelist, and the cap in none, is how a
/// location-scale `noise_formula` with a `thinplate()` smooth came to refuse on
/// its own injected option (gam#2781 tightened the whitelists).
pub const ENGINE_OPTION_PREFIX: &str = "__";

/// Whether `key` belongs to the engine-injected option namespace.
pub fn is_engine_option(key: &str) -> bool {
    key.starts_with(ENGINE_OPTION_PREFIX)
}

pub fn validate_known_options(
    term_name: &str,
    options: &BTreeMap<String, String>,
    known: &[&str],
) -> Result<(), String> {
    let known_set: std::collections::BTreeSet<&&str> = known.iter().collect();
    for key in options.keys() {
        if is_engine_option(key) {
            continue;
        }
        if !known_set.contains(&key.as_str()) {
            if term_name == "tensor" && is_tensor_k_axis_option_key(key) {
                continue;
            }
            // Suggest near-matches (substring or shared prefix ≥ 3).
            let key_l = key.to_ascii_lowercase();
            let mut suggestions: Vec<&str> = known
                .iter()
                .filter(|k| {
                    let kl = k.to_ascii_lowercase();
                    kl.contains(&key_l) || key_l.contains(&kl) || {
                        let n = kl
                            .chars()
                            .zip(key_l.chars())
                            .take_while(|(a, b)| a == b)
                            .count();
                        n >= 3
                    }
                })
                .copied()
                .collect();
            suggestions.sort_unstable();
            suggestions.dedup();
            let hint = if suggestions.is_empty() {
                String::new()
            } else {
                format!(" — did you mean one of [{}]?", suggestions.join(", "))
            };
            return Err(TermBuilderError::invalid_option(format!(
                "{term_name}() does not accept option `{key}`{hint}. Valid options: [{}]",
                {
                    let mut sorted = known.to_vec();
                    sorted.sort_unstable();
                    sorted.join(", ")
                }
            ))
            .to_string());
        }
    }
    Ok(())
}

/// Private (engine-injected) option that caps the *default* spatial center
/// count for a secondary (distributional) predictor's smooth — see
/// `solver::fit_orchestration::apply_secondary_predictor_basis_parsimony` and #501.
///
/// It is deliberately NOT one of the user-facing count aliases recognised by
/// [`has_explicit_countwith_basis_alias`], so it never flips the spatial basis
/// onto the explicit (hard) center-placement strategy: the cap lowers the
/// *default* count while the `Auto` strategy is retained, so the count is still
/// softly reduced when the data can't support it.
pub const SECONDARY_CENTER_CAP_OPTION: &str = "__secondary_center_cap";

/// Apply the secondary-predictor center cap to a *default* spatial center
/// count. A no-op when the cap option is absent (the common case) or when the
/// user supplied an explicit count (then `default_count` is ignored downstream
/// by [`parse_countwith_basis_alias`] anyway).
pub(crate) fn cap_default_spatial_centers(
    options: &BTreeMap<String, String>,
    default_count: usize,
) -> usize {
    match option_usize(options, SECONDARY_CENTER_CAP_OPTION) {
        Some(cap) => default_count.min(cap),
        None => default_count,
    }
}

fn default_matern_center_count(
    n: usize,
    d: usize,
    planned_count: usize,
    univariate_floor: usize,
) -> usize {
    // #1074: the mgcv-sized basis cap (`k = 10·3^(d-1)`) was DELETED here too — it
    // masked the same over-sizing/under-penalization defect by shrinking the basis
    // rather than fixing the optimizer. The default now uses the generic n-scaling
    // plan. A small-n floor against a numerically-fragile two-column kernel block
    // is a legitimate degenerate guard and is kept. Explicit `k`/`centers` still
    // take full effect upstream.
    let low_n_floor = (d + 4).min(n);
    // #1867: at small n the generic conditioning cap (`n / COND_N_DIVISOR`) in
    // `default_num_centers` starves a 1-D radial basis BELOW the resolution the
    // univariate B-spline `s(x)` is handed on the SAME data (e.g. 7 vs 11 basis
    // functions at n=30), so `matern(x)`/`duchon(x)` over-smooth sparse
    // oscillations that `s(x)` recovers cleanly. Smoothness is set by the REML
    // penalty λ, not by the raw center count (see `default_num_centers`), so a
    // radial smooth competing with `s(x)` must not be dimensioned coarser than
    // it. `univariate_floor` carries that spline-equivalent resolution for a 1-D
    // smooth (0 for d>1, where there is no direct univariate analogue) and is
    // bounded by n. Explicit `k`/`centers` still override upstream.
    planned_count
        .max(low_n_floor)
        .max(univariate_floor.min(n))
        .max(1)
}

fn default_duchon_center_count(
    n: usize,
    d: usize,
    planned_count: usize,
    polynomial_cols: usize,
    univariate_floor: usize,
) -> usize {
    // #1757: Duchon fits pay a larger setup cost than Matérn/TPS because the
    // constrained radial block is rotated through its center Gram and several
    // operator-collocation penalties.  The old generic spatial default handed a
    // 2-D Gaussian Duchon at n≈500 more than one hundred centers, so cold fits
    // spent most of their time in dense O(k³) eigensolves even though the REML
    // smoother uses a low-rank basis.  mgcv's Duchon spline default is the
    // thin-plate-style `k = 10 * 3^(d - 1)` (30 in 2-D); use that as the
    // implicit low-rank cap while preserving the user's explicit `centers=`/`k=`
    // request above.  The polynomial null space must still fit, so tiny
    // high-order bases are raised to the smallest admissible count.
    let mgcv_default = 10usize.saturating_mul(3usize.saturating_pow(d.saturating_sub(1) as u32));
    let low_n_floor = (polynomial_cols + 1).min(n).max(1);
    // #1867: at small n the generic conditioning cap (`n / COND_N_DIVISOR`) in
    // `default_num_centers` starves `planned_count` below the univariate spline
    // resolution the competing `s(x)` gets on the SAME data, so `duchon(x)`
    // over-smooths sparse oscillations. `univariate_floor` (0 for d>1) carries
    // that spline-equivalent basis dimension and floors the 1-D default,
    // bounded by n; smoothness is set by the REML penalty, not the raw count.
    // Explicit `k`/`centers` still override upstream.
    planned_count
        .min(mgcv_default)
        .max(low_n_floor)
        .max(univariate_floor.min(n))
}

pub fn parse_countwith_basis_alias(
    options: &BTreeMap<String, String>,
    primarykey: &str,
    default_count: usize,
) -> Result<usize, String> {
    // Strict: reject unparseable values (e.g. `centers=many`, `centers=-1`,
    // `centers=1.5`) instead of silently dropping them and falling through
    // to the default. Without this the user gets the auto-inferred count
    // silently and never realizes their explicit option was ignored.
    let primary = option_usize_strict(options, primarykey)?;
    let basis_dim = option_usize_any_strict(
        options,
        &["k", "basis_dim", "basis-dim", "basisdim", "knots"],
    )?;
    if primary.is_some() && basis_dim.is_some() {
        return Err(TermBuilderError::incompatible_config(format!(
            "specify either {}=<count> or k=<basis_dim> (not both)",
            primarykey
        ))
        .to_string());
    }
    Ok(primary.or(basis_dim).unwrap_or(default_count))
}

/// Resolve the wiggliness penalty's derivative/difference order from its two
/// documented spellings.
///
/// `penalty_order=` and `m=` name the SAME knob here and in mgcv. Reading one
/// and falling through to the other (`option_usize(.., "penalty_order")
/// .or_else(|| option_usize(.., "m"))`) silently drops the loser when both are
/// given, and the lenient `option_usize` silently drops an unparseable value on
/// top of that. This refuses the conflict the way `parse_countwith_basis_alias`
/// refuses `centers=` together with `k=`, and parses strictly so `m=1.5` is a
/// user mistake rather than "m not specified".
pub fn parse_penalty_order_alias(
    options: &BTreeMap<String, String>,
) -> Result<Option<usize>, String> {
    let primary = option_usize_strict(options, "penalty_order")?;
    let alias = option_usize_strict(options, "m")?;
    match (primary, alias) {
        (Some(primary), Some(alias)) if primary != alias => {
            Err(TermBuilderError::incompatible_config(format!(
                "penalty_order={primary} and m={alias} are two spellings of the same \
                 option (the order of the penalised derivative), so they cannot disagree; \
                 specify one of them"
            ))
            .to_string())
        }
        (Some(primary), _) => Ok(Some(primary)),
        (None, alias) => Ok(alias),
    }
}

pub fn has_explicit_countwith_basis_alias(
    options: &BTreeMap<String, String>,
    primarykey: &str,
) -> bool {
    options.contains_key(primarykey)
        || ["k", "basis_dim", "basis-dim", "basisdim", "knots"]
            .iter()
            .any(|alias| options.contains_key(*alias))
}

pub fn parse_cyclic_boundary(
    options: &BTreeMap<String, String>,
    minv: f64,
    maxv: f64,
) -> Result<OneDimensionalBoundary, String> {
    let cyclic = option_bool(options, "cyclic")
        .or_else(|| option_bool(options, "periodic"))
        .unwrap_or(false);
    if !cyclic {
        return Ok(OneDimensionalBoundary::Open);
    }
    let start = match option_numeric_expr(options, "period_start")? {
        Some(v) => v,
        None => option_numeric_expr(options, "start")?.unwrap_or(minv),
    };
    let end = match option_numeric_expr(options, "period_end")? {
        Some(v) => v,
        None => option_numeric_expr(options, "end")?.unwrap_or(maxv),
    };
    if end <= start {
        return Err(format!(
            "cyclic smooth requires period_end/end ({end}) > period_start/start ({start})"
        ));
    }
    Ok(OneDimensionalBoundary::Cyclic { start, end })
}

/// Parse the periodic-uniform domain for a one-dimensional cyclic smooth.
///
/// Returns the `(domain_start, period)` pair derived from
/// `period_start` / `start`, `period_end` / `end`, falling back to the
/// data range `[minv, maxv)` when neither bound is provided. The period
/// must be strictly positive.
pub fn parse_periodic_domain_1d(
    options: &BTreeMap<String, String>,
    minv: f64,
    maxv: f64,
) -> Result<(f64, f64), String> {
    let start_opt = match option_numeric_expr(options, "period_start")? {
        Some(v) => Some(v),
        None => option_numeric_expr(options, "start")?,
    };
    let end_opt = match option_numeric_expr(options, "period_end")? {
        Some(v) => Some(v),
        None => option_numeric_expr(options, "end")?,
    };
    // Reject the pure data-range fallback. A B-spline periodic smooth that takes
    // its wrap from the observed [min, max] is sample-dependent and silently
    // wrong: uniform draws on a true period of 2π land on [ε, 2π−ε], so using
    // (max−min) as the period seams the curve with an off-by-ε discontinuity and
    // the fit drifts with the sample. (Unlike the radial closed-lattice Duchon
    // path, whose centers DO tile a full period, so its span-derive is exact —
    // see `parse_periodic_axes_option`.) Require the caller to name the period
    // explicitly via `period=`/`period_end`. The end is only defaulted to `maxv`
    // when a `period_start`/`start` was given (a half-open declaration); a bare
    // periodic smooth with neither bound is an error.
    if end_opt.is_none() && start_opt.is_none() {
        return Err(
            "periodic B-spline smooth requires an explicit period: pass period=<value> \
             (e.g. period=2*pi) or period_start=/period_end=. Deriving the period from the \
             observed data range is sample-dependent and produces an off-by-ε seam, so it is \
             not inferred."
                .to_string(),
        );
    }
    let start = start_opt.unwrap_or(minv);
    let end = end_opt.unwrap_or(maxv);
    if !(start.is_finite() && end.is_finite()) {
        return Err(format!(
            "periodic smooth domain requires finite endpoints, got ({start}, {end})"
        ));
    }
    if end <= start {
        return Err(format!(
            "periodic smooth requires period_end/end ({end}) > period_start/start ({start})"
        ));
    }
    Ok((start, end - start))
}

fn parse_matern_nu(raw: &str) -> Result<MaternNu, String> {
    let trimmed = raw.trim();
    let lowered = trimmed.to_ascii_lowercase();
    // Exact spellings of the half-integer smoothnesses that have closed-form
    // kernels; anything else falls through to the numeric parse below.
    let named = match lowered.as_str() {
        "1/2" | "0.5" | "half" => Some(MaternNu::Half),
        "3/2" | "1.5" => Some(MaternNu::ThreeHalves),
        "5/2" | "2.5" => Some(MaternNu::FiveHalves),
        "7/2" | "3.5" => Some(MaternNu::SevenHalves),
        "9/2" | "4.5" => Some(MaternNu::NineHalves),
        _ => None,
    };
    if let Some(nu) = named {
        return Ok(nu);
    }

    let value = if let Some((num, den)) = trimmed.split_once('/') {
        let num = num
            .trim()
            .parse::<f64>()
            .map_err(|err| format!("{}: {err}", unsupported_matern_nu_message(raw)))?;
        let den = den
            .trim()
            .parse::<f64>()
            .map_err(|err| format!("{}: {err}", unsupported_matern_nu_message(raw)))?;
        if den == 0.0 || !num.is_finite() || !den.is_finite() {
            return Err(unsupported_matern_nu_message(raw));
        }
        num / den
    } else {
        trimmed
            .parse::<f64>()
            .map_err(|err| format!("{}: {err}", unsupported_matern_nu_message(raw)))?
    };

    const TOL: f64 = 1e-12;
    if (value - 0.5).abs() <= TOL {
        Ok(MaternNu::Half)
    } else if (value - 1.5).abs() <= TOL {
        Ok(MaternNu::ThreeHalves)
    } else if (value - 2.5).abs() <= TOL {
        Ok(MaternNu::FiveHalves)
    } else if (value - 3.5).abs() <= TOL {
        Ok(MaternNu::SevenHalves)
    } else if (value - 4.5).abs() <= TOL {
        Ok(MaternNu::NineHalves)
    } else {
        Err(unsupported_matern_nu_message(raw))
    }
}

fn unsupported_matern_nu_message(raw: &str) -> String {
    TermBuilderError::unsupported_feature(format!(
        "unsupported Matern nu '{raw}'; supported half-integer values are 1/2, 3/2, 5/2, 7/2, and 9/2"
    ))
    .to_string()
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub enum DuchonPowerPolicy {
    Explicit(f64),
    /// No explicit `power=` given: defer to the cubic structural default, which
    /// the builder resolves dimension-aware as `s = (d − 1)/2` (so `φ(r) = r³`
    /// in every dimension). There is no triple-operator minimum any more.
    CubicStructuralDefault,
}

pub fn parse_duchon_power_policy(
    options: &BTreeMap<String, String>,
) -> Result<DuchonPowerPolicy, String> {
    if let Some(raw_nu) = options.get("nu") {
        return Err(TermBuilderError::incompatible_config(format!(
            "Duchon smooths use power=<number>, not nu='{}'. Use power=1.5, power=2, etc.",
            raw_nu
        ))
        .to_string());
    }
    // `p` is the Duchon whitelist's alias of `power` and was read nowhere, so
    // `duchon(x, z, p=2)` was accepted and silently used the structural default
    // (#2781's family).
    match options.get("power").or_else(|| options.get("p")) {
        Some(raw) => {
            let value = raw.parse::<f64>().map_err(|err| {
                TermBuilderError::invalid_option(format!(
                    "invalid Duchon power '{}'; expected a non-negative number such as power=1.5 or power=2: {}",
                    raw, err
                ))
                .to_string()
            })?;
            if !value.is_finite() || value < 0.0 {
                return Err(TermBuilderError::invalid_option(format!(
                    "invalid Duchon power '{}'; expected a finite non-negative number such as power=1.5 or power=2",
                    raw
                ))
                .to_string());
            }
            Ok(DuchonPowerPolicy::Explicit(value))
        }
        None => Ok(DuchonPowerPolicy::CubicStructuralDefault),
    }
}

/// Like [`parse_duchon_order`] but reports ABSENCE, so a caller whose default
/// happens to equal `Linear` can still tell "the user named the affine null
/// space" apart from "the user named nothing". The `duchon` arm needs that
/// distinction: its structural cubic default supplies a jointly chosen
/// `(order, power)` PAIR, and it used to take the order from that pair even
/// when the caller had named one (#2781's family) — contradicting this module's
/// own contract that "an explicit `order=0` still selects the constant-only
/// space".
pub fn parse_duchon_order_opt(
    options: &BTreeMap<String, String>,
) -> Result<Option<DuchonNullspaceOrder>, String> {
    if !options.contains_key("order") && !options.contains_key("nullspace_order") {
        return Ok(None);
    }
    parse_duchon_order(options).map(Some)
}

pub fn parse_duchon_order(
    options: &BTreeMap<String, String>,
) -> Result<DuchonNullspaceOrder, String> {
    // `nullspace_order` is the whitelist's alias of `order` and was read
    // nowhere (#2781's family).
    match options.get("order").or_else(|| options.get("nullspace_order")) {
        // Structural cubic Duchon is affine-by-default: an unspecified order is
        // the `Linear` (constant + linear) null space, matching the magic
        // default. An explicit `order=0` still selects the constant-only space.
        None => Ok(DuchonNullspaceOrder::Linear),
        Some(raw) => match raw.parse::<usize>() {
            Ok(0) => Ok(DuchonNullspaceOrder::Zero),
            Ok(1) => Ok(DuchonNullspaceOrder::Linear),
            Ok(other) => Ok(DuchonNullspaceOrder::Degree(other)),
            Err(_) => Err(TermBuilderError::invalid_option(format!(
                "invalid Duchon order '{}'; expected a non-negative integer such as order=0, order=1, or order=2",
                raw
            ))
            .to_string()),
        },
    }
}

fn parse_matern_identifiability(
    options: &BTreeMap<String, String>,
) -> Result<MaternIdentifiability, TermBuilderError> {
    let Some(raw) = options.get("identifiability").map(String::as_str) else {
        return Ok(MaternIdentifiability::default());
    };
    match raw.trim().to_ascii_lowercase().as_str() {
        "none" => Ok(MaternIdentifiability::None),
        "sum_tozero" | "sum-to-zero" | "center_sum_tozero" | "center-sum-to-zero" | "centered" => {
            Ok(MaternIdentifiability::CenterSumToZero)
        }
        "linear" | "center_linear_orthogonal" | "center-linear-orthogonal" => {
            Ok(MaternIdentifiability::CenterLinearOrthogonal)
        }
        other => Err(TermBuilderError::unsupported_feature(format!(
            "invalid Matérn identifiability '{other}'; expected one of: none, sum_tozero, linear"
        ))),
    }
}

fn parse_spatial_identifiability(
    options: &BTreeMap<String, String>,
) -> Result<SpatialIdentifiability, TermBuilderError> {
    let Some(raw) = options.get("identifiability").map(String::as_str) else {
        return Ok(SpatialIdentifiability::default());
    };
    match raw.trim().to_ascii_lowercase().as_str() {
        "none" => Ok(SpatialIdentifiability::None),
        "orthogonal"
        | "orthogonal_to_parametric"
        | "orthogonal-to-parametric"
        | "parametric_orthogonal" => Ok(SpatialIdentifiability::OrthogonalToParametric),
        "frozen" => Err(TermBuilderError::unsupported_feature(
            "spatial identifiability 'frozen' is internal-only; use none or orthogonal_to_parametric",
        )),
        other => Err(TermBuilderError::unsupported_feature(format!(
            "invalid spatial identifiability '{other}'; expected one of: none, orthogonal_to_parametric"
        ))),
    }
}

#[cfg(test)]
mod tests;
