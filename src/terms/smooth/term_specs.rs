use bspline_boundary::bspline_boundary_linear_constraints;

use coefficient_transforms::{
    convex_divided_difference_transform_matrix, cumulative_exp, cumulative_sum_transform_matrix,
    second_cumulative_exp,
};

pub use error::SmoothError;

use input_standardization::{
    apply_input_standardization, compensate_length_scale_for_standardization,
    compensate_optional_length_scale_for_standardization, compute_spatial_input_scales,
};

use shape_constraints::{
    build_shape_constraint_design_1d, build_shape_linear_constraints_1d,
    linear_constraints_from_lower_bounds_global, merge_linear_constraints_global,
    shape_lower_bounds_local, shape_order_and_sign, shape_supports_basis,
    shape_uses_box_reparameterization,
};

fn describe_thin_plate_center_request(strategy: &CenterStrategy) -> String {
    match strategy {
        CenterStrategy::Auto(inner) => describe_thin_plate_center_request(inner),
        CenterStrategy::UserProvided(centers) => format!("{} centers", centers.nrows()),
        CenterStrategy::EqualMass { num_centers }
        | CenterStrategy::EqualMassCovarRepresentative { num_centers }
        | CenterStrategy::FarthestPoint { num_centers }
        | CenterStrategy::KMeans { num_centers, .. } => format!("{num_centers} centers"),
        CenterStrategy::UniformGrid { points_per_dim } => {
            format!("uniform grid with {points_per_dim} points per dimension")
        }
    }
}

fn rewrite_thin_plate_knots_error(
    err: BasisError,
    termname: &str,
    feature_count: usize,
    spec: &ThinPlateBasisSpec,
) -> BasisError {
    match err {
        // Polynomial-nullspace shortfall reported directly by the kernel
        // builder ("thin-plate spline requires at least N centers to span ...").
        BasisError::InvalidInput(msg)
            if msg.contains("thin-plate spline requires at least")
                && (msg.contains("centers to span") || msg.contains("knots to span")) =>
        {
            let min_centers = crate::basis::thin_plate_polynomial_basis_dimension(feature_count);
            let requested = describe_thin_plate_center_request(&spec.center_strategy);
            BasisError::InvalidInput(format!(
                "joint TPS term '{termname}' over {feature_count} covariates with {requested} is invalid; minimum centers is {min_centers}"
            ))
        }
        // Insufficient-rows shortfall raised by `select_thin_plate_knots` when
        // the requested center count exceeds the available row count. Rewrite
        // it in term language so the diagnostic points at the smooth term and
        // the polynomial-nullspace minimum the user needs to satisfy.
        BasisError::InvalidInput(msg)
            if msg.starts_with("requested ") && msg.contains(" knots but only ") =>
        {
            let min_centers = crate::basis::thin_plate_polynomial_basis_dimension(feature_count);
            let requested = describe_thin_plate_center_request(&spec.center_strategy);
            BasisError::InvalidInput(format!(
                "joint TPS term '{termname}' over {feature_count} covariates with {requested} is invalid; minimum centers is {min_centers}"
            ))
        }
        other => other,
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ShapeConstraint {
    None,
    MonotoneIncreasing,
    MonotoneDecreasing,
    Convex,
    Concave,
}

/// Parse a shape-constraint string into a [`ShapeConstraint`].
///
/// This is the single source of truth shared by the formula DSL
/// (`s(x, shape=...)`) and the `smooths={...}` override path
/// (`Smooth.shape_constraint`). The accepted spellings cover the canonical
/// Python `ShapeConstraintLiteral` strings exactly
/// (`"none"` / `"monotone_increasing"` / `"monotone_decreasing"` /
/// `"convex"` / `"concave"`) plus a few common aliases. Hyphens and case are
/// normalized, so `"Monotone-Increasing"` and `"mono_inc"` both resolve to
/// [`ShapeConstraint::MonotoneIncreasing`].
pub fn parse_shape_constraint(raw: &str) -> Result<ShapeConstraint, String> {
    let normalized = raw.trim().to_ascii_lowercase().replace('-', "_");
    match normalized.as_str() {
        "" | "none" => Ok(ShapeConstraint::None),
        "monotone_increasing" | "monotonic_increasing" | "increasing" | "mono_inc" | "mpi" => {
            Ok(ShapeConstraint::MonotoneIncreasing)
        }
        "monotone_decreasing" | "monotonic_decreasing" | "decreasing" | "mono_dec" | "mpd" => {
            Ok(ShapeConstraint::MonotoneDecreasing)
        }
        "convex" | "cvx" => Ok(ShapeConstraint::Convex),
        "concave" | "ccv" => Ok(ShapeConstraint::Concave),
        other => Err(format!(
            "unknown shape constraint {other:?}; expected one of \
             \"none\", \"monotone_increasing\", \"monotone_decreasing\", \
             \"convex\", \"concave\""
        )),
    }
}

impl ShapeConstraint {
    /// Canonical formula-DSL spelling, i.e. the text emitted into
    /// `s(x, shape=...)`. Round-trips through [`parse_shape_constraint`].
    pub fn dsl_str(&self) -> &'static str {
        match self {
            ShapeConstraint::None => "none",
            ShapeConstraint::MonotoneIncreasing => "monotone_increasing",
            ShapeConstraint::MonotoneDecreasing => "monotone_decreasing",
            ShapeConstraint::Convex => "convex",
            ShapeConstraint::Concave => "concave",
        }
    }
}

/// Smooth-term head keywords recognised by the formula DSL. A `shape=` option
/// may be attached to any term whose head is one of these.
const SMOOTH_HEAD_KEYWORDS: [&str; 11] = [
    "s",
    "smooth",
    "te",
    "tensor",
    "thinplate",
    "tps",
    "duchon",
    "matern",
    "sphere",
    "bs",
    "bspline",
];

/// Rewrite smooth-term calls in `formula` so each named smooth carries a
/// `shape=<kind>` option understood by the formula DSL.
///
/// `constraints` pairs the smooth-term text as it appears in the formula
/// (e.g. `"s(x)"` or `"s(x, type=duchon, centers=8)"`) with a shape-constraint
/// spelling accepted by [`parse_shape_constraint`]; comparison is exact after
/// whitespace removal. A `"none"` constraint is a no-op. Referencing a term not
/// present in the formula is an error.
///
/// This is the single source of truth for the `gamfit.fit(..., constraints=…)`
/// rewrite — the Python wrapper only marshals the mapping across the FFI and
/// holds no formula-parsing or alias-normalization logic of its own.
pub fn apply_shape_constraints_to_formula(
    formula: &str,
    constraints: &[(String, String)],
) -> Result<String, String> {
    use std::collections::{BTreeMap, BTreeSet};

    if constraints.is_empty() {
        return Ok(formula.to_string());
    }
    let strip_ws = |s: &str| -> String { s.chars().filter(|c| !c.is_whitespace()).collect() };

    // Whitespace-stripped term text -> canonical shape spelling.
    let mut wanted: BTreeMap<String, &'static str> = BTreeMap::new();
    // Whitespace-stripped term text -> original key (for error labels).
    let mut originals: BTreeMap<String, String> = BTreeMap::new();
    for (key, kind_raw) in constraints {
        let kind = parse_shape_constraint(kind_raw)?;
        let nk = strip_ws(key);
        originals.entry(nk.clone()).or_insert_with(|| key.clone());
        if kind != ShapeConstraint::None {
            wanted.insert(nk, kind.dsl_str());
        }
    }
    if wanted.is_empty() {
        return Ok(formula.to_string());
    }

    let chars: Vec<char> = formula.chars().collect();
    let n = chars.len();
    let is_ident = |c: char| c.is_ascii_alphanumeric() || c == '_';

    let mut out = String::with_capacity(formula.len() + 32);
    let mut matched: BTreeSet<String> = BTreeSet::new();
    let mut i = 0usize;
    while i < n {
        // Locate the next smooth-term head (`<keyword> \s* (`) at or after `i`,
        // respecting word boundaries so `abs(` never matches the `s(` head.
        let mut head: Option<(usize, usize)> = None; // (head_start, paren_index)
        let mut p = i;
        while p < n {
            let boundary = p == 0 || !is_ident(chars[p - 1]);
            if boundary {
                for kw in SMOOTH_HEAD_KEYWORDS.iter() {
                    let klen = kw.chars().count();
                    if p + klen > n || chars[p..p + klen].iter().collect::<String>() != **kw {
                        continue;
                    }
                    let mut q = p + klen;
                    while q < n && chars[q].is_whitespace() {
                        q += 1;
                    }
                    if q < n && chars[q] == '(' {
                        head = Some((p, q));
                        break;
                    }
                }
            }
            if head.is_some() {
                break;
            }
            p += 1;
        }
        let (head_start, paren_open) = match head {
            Some(h) => h,
            None => {
                out.extend(chars[i..].iter());
                break;
            }
        };
        out.extend(chars[i..head_start].iter());

        // Find the matching close paren, honoring nesting and string literals.
        let body_start = paren_open + 1;
        let mut depth = 1i32;
        let mut j = body_start;
        let mut in_str: Option<char> = None;
        let mut closed = false;
        while j < n {
            let ch = chars[j];
            if let Some(quote) = in_str {
                if ch == quote {
                    in_str = None;
                }
            } else if ch == '\'' || ch == '"' {
                in_str = Some(ch);
            } else if ch == '(' {
                depth += 1;
            } else if ch == ')' {
                depth -= 1;
                if depth == 0 {
                    closed = true;
                    break;
                }
            }
            j += 1;
        }

        if !closed {
            // Unbalanced — emit the remainder verbatim; the DSL parser will
            // produce the canonical error.
            out.extend(chars[head_start..].iter());
            break;
        }

        let term_text: String = chars[head_start..=j].iter().collect();

        let key_norm = strip_ws(&term_text);

        match wanted.get(&key_norm) {
            None => out.extend(chars[head_start..=j].iter()),
            Some(kind) => {
                let head_paren: String = chars[head_start..body_start].iter().collect();
                let inside: String = chars[body_start..j].iter().collect();
                let inside = inside.trim();
                if inside.is_empty() {
                    out.push_str(&format!("{head_paren}shape={kind})"));
                } else {
                    out.push_str(&format!("{head_paren}{inside}, shape={kind})"));
                }
                matched.insert(key_norm);
            }
        }

        i = j + 1;
    }

    let mut missing: Vec<String> = wanted
        .keys()
        .filter(|k| !matched.contains(*k))
        .map(|k| originals.get(k).cloned().unwrap_or_else(|| k.clone()))
        .collect();

    if !missing.is_empty() {
        missing.sort();
        return Err(format!(
            "shape constraints referenced smooth term(s) not found in formula: {}",
            missing.join(", ")
        ));
    }

    Ok(out)
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum BySmoothKind {
    Numeric,
    Level { level_bits: u64 },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SmoothBasisSpec {
    /// Row-gated wrapper used for mgcv-style ``by=`` smooths.
    ///
    /// ``ByNumeric`` multiplies the inner smooth by a numeric column.
    /// ``ByLevel`` keeps the inner smooth active only for rows whose encoded
    /// categorical value has the stored bit pattern.  Unordered factor-by
    /// smooths are represented as one independent ``ByLevel`` term per level.
    ///
    /// `kind` preserves the compact structural discriminator, while `by`
    /// carries the full row-gating spec used to build the local design.
    ByVariable {
        inner: Box<SmoothBasisSpec>,
        by_col: usize,
        kind: BySmoothKind,
        by: ByVariableSpec,
    },
    /// Sum-to-zero factor smooth (`bs="sz"`): with L levels, estimate L-1
    /// deviation coefficient blocks and use the final level as the negative
    /// sum of the others, enforcing coefficient-wise zero sums across levels.
    FactorSumToZero {
        inner: Box<SmoothBasisSpec>,
        by_col: usize,
        levels: Vec<u64>,
        /// Global-orthogonality column map `Z` captured at fit time when this
        /// term overlapped an owner smooth (`s(x) + s(g, x, bs=sz)`, #978):
        /// the hierarchical-ownership pass residualized this term's realized
        /// design as `X ← X·Z`, shrinking its coefficient block. `Z` depends
        /// on the *training-row* owner designs, so prediction cannot rederive
        /// it — it must be persisted and replayed
        /// (`apply_global_smooth_identifiability` consumes it verbatim).
        /// Chart convention: `Z` lives in the post-restack, post-joint-null-Q
        /// coordinates — the raw `sz` rebuild reapplies `Q` deterministically
        /// (#700), then `Z` applies on top. `None` for non-overlapping terms.
        #[serde(default)]
        frozen_global_orthogonality: Option<Array2<f64>>,
    },
    BSpline1D {
        feature_col: usize,
        spec: BSplineBasisSpec,
    },
    /// A smooth modulated by a `by=` variable. Numeric `by` scales one inner
    /// smooth; factor `by` replicates the inner smooth by level.
    BySmooth {
        smooth: Box<SmoothBasisSpec>,
        by_kind: ByVarKind,
    },
    /// Factor-smooth interaction families (`bs="fs"`, `bs="sz"`) and
    /// random slopes (`bs="re"`).
    FactorSmooth { spec: FactorSmoothSpec },
    ThinPlate {
        feature_cols: Vec<usize>,
        spec: ThinPlateBasisSpec,
        /// Per-column standard deviations used to standardize input dimensions
        /// before kernel evaluation when d > 1. `None` means no standardization
        /// (either d == 1 or explicitly disabled).
        #[serde(default)]
        input_scales: Option<Vec<f64>>,
    },
    Sphere {
        feature_cols: Vec<usize>,
        spec: SphericalSplineBasisSpec,
    },
    /// Constant-curvature (`M_κ`) geodesic-kernel smooth over κ-stereographic
    /// chart coordinates (#944): one construction interpolating
    /// S^d → ℝ^d → H^d through the spec's fixed κ. The Wahba S² smooth is the
    /// structural template; the geometry comes from
    /// `geometry::constant_curvature::ConstantCurvature`.
    ConstantCurvature {
        feature_cols: Vec<usize>,
        spec: ConstantCurvatureBasisSpec,
    },
    Matern {
        feature_cols: Vec<usize>,
        spec: MaternBasisSpec,
        #[serde(default)]
        input_scales: Option<Vec<f64>>,
    },
    /// Measure-jet spline smooth: multiscale local-jet-residual energy of the
    /// empirical measure (centers as μ-quadrature, masses as μ-weights — no
    /// graph, mesh, or neighbor set inside the statistical object). The
    /// feature columns are ambient coordinates of data concentrated near an
    /// unknown low-dimensional, possibly stratified set.
    MeasureJet {
        feature_cols: Vec<usize>,
        spec: MeasureJetBasisSpec,
        #[serde(default)]
        input_scales: Option<Vec<f64>>,
    },
    Duchon {
        feature_cols: Vec<usize>,
        spec: DuchonBasisSpec,
        #[serde(default)]
        input_scales: Option<Vec<f64>>,
    },
    Pca {
        feature_cols: Vec<usize>,
        basis_matrix: Array2<f64>,
        centered: bool,
        #[serde(default = "default_pca_smooth_penalty")]
        smooth_penalty: f64,
        #[serde(default)]
        center_mean: Option<Array1<f64>>,
        #[serde(default)]
        pca_basis_path: Option<PathBuf>,
        #[serde(default = "default_pca_chunk_size")]
        chunk_size: usize,
    },
    /// Tensor-product smooth built from 1D B-spline marginals.
    ///
    /// This is the `te()`-style construction used when axes have different units/scales
    /// (for example, space x time) and isotropic radial kernels are not appropriate.
    TensorBSpline {
        feature_cols: Vec<usize>,
        spec: TensorBSplineSpec,
    },
}

impl SmoothBasisSpec {
    /// Conservative lower bound on the number of sample rows needed for this
    /// smooth basis to have a well-posed REML fit.
    ///
    /// Each basis kind answers the question for itself, so the workflow does
    /// not have to know how many columns a B-spline, tensor product, PCA
    /// projection, or spatial kernel emits. The contract is a *lower bound*:
    /// returning too small a number is permitted (the inner solver will catch
    /// any genuine n-vs-rank failure that slips past); returning too large a
    /// number is a regression because it rejects legitimate fits.
    ///
    /// Rationale: B-spline / tensor / PCA bases have a closed-form column
    /// count, so we use the exact dimension. Radial bases (TPS, Matern,
    /// Duchon, Sphere) and factor smooths choose their column count from the
    /// data (`heuristic_centers`, `unique_count`); we fall back to a small
    /// constant floor because a fit on fewer than five rows cannot stabilise
    /// any radial smooth regardless of the configured kernel scale.
    pub fn min_sample_rows(&self) -> usize {
        // Floor used for data-driven bases whose column count is not known
        // from the spec alone. Five rows is the minimum at which the inner
        // pivot/QR + REML smoothing-parameter search has any chance of being
        // well-posed for a non-parametric smooth.
        const RADIAL_FLOOR: usize = 5;

        match self {
            Self::ByVariable { inner, .. } => inner.min_sample_rows(),
            Self::FactorSumToZero { inner, levels, .. } => {
                // L-1 independent deviation blocks each carrying the inner
                // basis dimension. Skip the levels-multiplier if it doesn't
                // bring more rows; we want the *lower bound* not the rank.
                let inner_min = inner.min_sample_rows();
                let lvls = levels.len().saturating_sub(1).max(1);
                inner_min.saturating_mul(lvls)
            }
            Self::BSpline1D { spec, .. } => bspline_basis_min_rows(spec),
            Self::BySmooth { smooth, .. } => smooth.min_sample_rows(),
            Self::FactorSmooth { spec } => {
                // Replicates the marginal once per level; without a known
                // level count we conservatively require at least the marginal
                // basis dimension.
                bspline_basis_min_rows(&spec.marginal)
            }
            Self::ThinPlate { .. }
            | Self::Sphere { .. }
            | Self::ConstantCurvature { .. }
            | Self::Matern { .. }
            | Self::MeasureJet { .. }
            | Self::Duchon { .. } => RADIAL_FLOOR,
            Self::Pca { basis_matrix, .. } => basis_matrix.ncols().max(1),
            Self::TensorBSpline { spec, .. } => {
                // A `te(...)` smooth is *penalized*: each margin carries a
                // difference (wiggliness) penalty and the tensor inherits a
                // Kronecker-sum penalty `S = Σ_i I ⊗ … ⊗ S_i ⊗ … ⊗ I`. The raw
                // column count is the *product* of the per-marginal column
                // counts, but that product is the lower bound for an
                // *unpenalized* tensor regression — it is the number of rows you
                // would need to identify every interaction column with no
                // regularization. The penalty regularizes all of those
                // interaction directions; only the combined penalty *null space*
                // (the tensor product of the per-margin polynomial trends, a
                // handful of columns) must be identified by the data, and the
                // smoothing-parameter search shrinks the rest. The effective
                // degrees of freedom of the fitted `te()` are therefore a small
                // fraction of the column product, which is exactly why mgcv
                // fits a default `te(x, y)` on a couple hundred rows.
                //
                // The honest *penalized* lower bound is the **sum** of the
                // per-marginal column counts, not their product: a row floor of
                // `Σ_i k_i` still guarantees enough data to identify each
                // margin's additive main-effect (the largest sub-block the
                // penalty cannot shrink to zero), while no longer conflating
                // unpenalized column-count identifiability with penalized
                // well-posedness. This accepts moderate-`n` penalized tensors
                // (e.g. a 20×20 default basis on n=200) yet still rejects a
                // genuinely undersized fit where `n < Σ_i k_i` and even the
                // additive part is rank-deficient.
                //
                // Binary / low-cardinality margins (#724): gam will accept a
                // `te(x, badh)` whose `badh ∈ {0, 1}` margin nominally requests
                // more basis columns than `badh` has unique values, where mgcv
                // refuses the unpenalized term as ill-posed ("badh has
                // insufficient unique values to support k knots"). This is
                // correct-by-design, *not* a degenerate fit: the marginal
                // wiggliness penalty on the `badh` axis has a null space that is
                // exactly its identifiable trend (the two cell means of a binary
                // covariate), and the Kronecker-sum penalty shrinks every tensor
                // column outside that null space toward zero. The resulting fit
                // is the well-posed "per-level `x` smooth + binary main effect"
                // that mgcv reaches only after manually collapsing the basis —
                // gam reaches it automatically because the penalty, not the raw
                // column count, sets the effective rank. A genuinely
                // rank-deficient design (penalty null space wider than the data
                // can support) is still caught downstream by the inner pivoted
                // factorization, which owns the exact n-vs-rank decision; this
                // pre-fit gate only refuses the grossly-undersized formula.
                let mut total: usize = 0;
                for marginal in &spec.marginalspecs {
                    let m = bspline_basis_min_rows(marginal);
                    total = total.saturating_add(m.max(1));
                }
                total.max(RADIAL_FLOOR)
            }
        }
    }

    /// Stable structural discriminant for warm-start cache keying (#869).
    ///
    /// Two smooths that produce different bases / penalty structures must map
    /// to different strings here so they cannot collide on the persistent
    /// warm-start `cache_key` (which is otherwise blind to topology: it hashes
    /// only the raw input column count, so e.g. `sphere` vs `torus` vs
    /// `euclidean` candidates fit on the *same* data would otherwise share one
    /// key and cross-contaminate each other's β/ρ seed). The string is the
    /// topology identity, not the fitted coefficients, so same-topology refits
    /// (the screen→full-refit cascade) still hit the same key and reuse work.
    pub fn structural_kind(&self) -> &'static str {
        match self {
            Self::ByVariable { .. } => "by_variable",
            Self::FactorSumToZero { .. } => "factor_sum_to_zero",
            Self::BSpline1D { .. } => "bspline_1d",
            Self::BySmooth { .. } => "by_smooth",
            Self::FactorSmooth { .. } => "factor_smooth",
            Self::ThinPlate { .. } => "thin_plate",
            Self::Sphere { .. } => "sphere",
            Self::ConstantCurvature { .. } => "constant_curvature",
            Self::Matern { .. } => "matern",
            Self::MeasureJet { .. } => "measurejet",
            Self::Duchon { .. } => "duchon",
            Self::Pca { .. } => "pca",
            Self::TensorBSpline { .. } => "tensor_bspline",
        }
    }

    /// Feature columns this basis consumes, used alongside [`structural_kind`]
    /// to disambiguate two same-kind smooths on different axes. Wrapper
    /// variants delegate to their inner basis.
    pub fn structural_feature_cols(&self) -> Vec<usize> {
        match self {
            Self::ByVariable { inner, .. } | Self::FactorSumToZero { inner, .. } => {
                inner.structural_feature_cols()
            }
            Self::BySmooth { smooth, .. } => smooth.structural_feature_cols(),
            Self::FactorSmooth { .. } => Vec::new(),
            Self::BSpline1D { feature_col, .. } => vec![*feature_col],
            Self::ThinPlate { feature_cols, .. }
            | Self::Sphere { feature_cols, .. }
            | Self::ConstantCurvature { feature_cols, .. }
            | Self::Matern { feature_cols, .. }
            | Self::MeasureJet { feature_cols, .. }
            | Self::Duchon { feature_cols, .. }
            | Self::Pca { feature_cols, .. }
            | Self::TensorBSpline { feature_cols, .. } => feature_cols.clone(),
        }
    }
}

/// Lower bound on the number of sample rows a 1D B-spline smooth needs for a
/// well-posed *penalized* REML fit. Used as the per-smooth row floor in
/// [`SmoothBasisSpec::min_sample_rows`].
///
/// For a *singly*-penalized smooth the floor is the full column count: the
/// wiggliness penalty leaves the order-`m` polynomial trend unpenalized, and
/// gam's original gate conservatively required enough rows for the whole basis.
/// That conservative floor is kept here unchanged.
///
/// A *double*-penalized smooth (mgcv `select=TRUE`) is different: it adds a
/// second penalty on the wiggliness penalty's null space, so even the
/// polynomial trend is shrinkable toward zero and *nothing* in the basis
/// requires unpenalized identification by the data — exactly the reasoning the
/// `TensorBSpline` arm of [`SmoothBasisSpec::min_sample_rows`] already applies
/// to a penalized tensor. Its honest floor is therefore a small stabilization
/// constant, not the column count. This is what lets mgcv (and now gam) fit
/// several `select=TRUE` smooths on a dataset whose row count is below the
/// summed basis width (e.g. the n≈30 `wine_gamair` fold, 5 `ps` smooths,
/// p≈51): the penalties, not the data, set the effective rank. The bounded
/// outer REML loop still terminates, and the genuine n-vs-rank decision is
/// owned downstream by the inner pivoted factorization. Without this, gam
/// rejected the fit outright (or, before the gate existed, the outer REML loop
/// wandered the flat overparameterized surface until the benchmark wall budget
/// killed it — #1089).
fn bspline_basis_min_rows(spec: &crate::terms::basis::BSplineBasisSpec) -> usize {
    use crate::terms::basis::BSplineKnotSpec;
    let columns = match &spec.knotspec {
        BSplineKnotSpec::Generate {
            num_internal_knots, ..
        } => *num_internal_knots + spec.degree + 1,
        BSplineKnotSpec::Automatic {
            num_internal_knots: Some(k),
            ..
        } => *k + spec.degree + 1,
        BSplineKnotSpec::Automatic {
            num_internal_knots: None,
            ..
        } => {
            // Knot count is data-derived (`default_internal_knot_count_for_data`).
            // A minimal cubic basis is `degree + 2` columns; below that the
            // basis cannot represent a non-parametric smooth.
            spec.degree + 2
        }
        BSplineKnotSpec::Provided(knots) => knots.len().saturating_sub(spec.degree + 1).max(1),
        BSplineKnotSpec::PeriodicUniform { num_basis, .. } => *num_basis,
    };
    let columns = columns.max(spec.degree + 2);

    if spec.double_penalty {
        // Fully shrinkable basis: only a small stabilization floor must be
        // identified by the data, capped by the actual column count.
        const DOUBLE_PENALTY_FLOOR: usize = 2;
        DOUBLE_PENALTY_FLOOR.min(columns).max(1)
    } else {
        columns
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ByVariableSpec {
    Numeric,
    Level { value_bits: u64, label: String },
}

/// Tensor-product B-spline smooth specification.
///
/// `marginalspecs[i]` is the 1D B-spline setup for `feature_cols[i]`.
/// The final penalty set is one Kronecker penalty per margin:
/// `S_i = I ⊗ ... ⊗ S_marginal_i ⊗ ... ⊗ I`, plus optional global ridge.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TensorMarginalSpec {
    BSpline(BSplineBasisSpec),
    Categorical {
        feature_col_offset: usize,
        drop_first_level: bool,
        center_for_identifiability: bool,
        frozen_levels: Option<Vec<u64>>,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ByVarKind {
    Numeric {
        feature_col: usize,
    },
    Factor {
        feature_col: usize,
        ordered: bool,
        frozen_levels: Option<Vec<u64>>,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FactorSmoothSpec {
    pub continuous_cols: Vec<usize>,
    pub group_col: usize,
    pub marginal: BSplineBasisSpec,
    pub flavour: FactorSmoothFlavour,
    pub group_frozen_levels: Option<Vec<u64>>,
    /// Fit-time global-orthogonality chart `Z` for this term (`s(x) + fs(x, g)`
    /// overlap residualization, #978), in the post-joint-null-`Q` coordinates
    /// (the raw rebuild recomputes any `Q` itself; `fs` penalties are
    /// typically full-rank so `Q` is absent). Training-row dependent, hence
    /// persisted; replayed verbatim by `apply_global_smooth_identifiability`.
    #[serde(default)]
    pub frozen_global_orthogonality: Option<Array2<f64>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum FactorSmoothFlavour {
    Fs { m_null_penalty_orders: Vec<usize> },
    Sz,
    Re,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TensorBSplineSpec {
    pub marginalspecs: Vec<BSplineBasisSpec>,
    #[serde(default)]
    pub periods: Vec<Option<f64>>,
    pub double_penalty: bool,
    #[serde(default)]
    pub identifiability: TensorBSplineIdentifiability,
}

#[derive(Debug, Default, Clone, Serialize, Deserialize)]
pub enum TensorBSplineIdentifiability {
    None,
    #[default]
    SumToZero,
    /// mgcv `ti(...)` semantics: a *tensor interaction* smooth that excludes the
    /// marginal main effects. A sum-to-zero constraint is applied to **each
    /// marginal basis independently** before forming the tensor product, so the
    /// resulting column space contains no function of a single variable alone —
    /// only the pure interaction survives. The realized identifiability
    /// transform is the Kronecker product `Z = Z₀ ⊗ Z₁ ⊗ … ⊗ Z_{d-1}` of the
    /// per-margin sum-to-zero null-space bases, which is exactly the
    /// reparameterization that turns the full-tensor design into the tensor
    /// product of the centered margins.
    MarginalSumToZero,
    FrozenTransform {
        transform: Array2<f64>,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SmoothTermSpec {
    pub name: String,
    pub basis: SmoothBasisSpec,
    pub shape: ShapeConstraint,
    /// Joint-null absorption rotation captured at fit time. `Some(Q)` means
    /// the fitted coefficient vector lives in `γ`-coordinates with
    /// `β_raw = Q · γ`; prediction must rotate the raw-basis design via
    /// `X_new = X_new_raw · Q` to match. `None` means either the smooth had
    /// no joint null space (penalty already full-rank) or rotation was
    /// suppressed (smooth carries shape constraints whose cone geometry
    /// would not survive an arbitrary orthogonal rotation). Persisted so
    /// `save → load → predict` is bit-equivalent to in-memory prediction.
    #[serde(default)]
    pub joint_null_rotation: Option<crate::terms::basis::JointNullRotation>,
}

#[derive(Debug, Clone)]
pub struct SmoothTerm {
    pub name: String,
    pub coeff_range: Range<usize>,
    pub shape: ShapeConstraint,
    pub penalties_local: Vec<Array2<f64>>,
    pub nullspace_dims: Vec<usize>,
    pub penaltyinfo_local: Vec<PenaltyInfo>,
    pub metadata: BasisMetadata,
    /// Optional term-local lower bounds for constrained coefficients.
    /// `-inf` means unconstrained.
    pub lower_bounds_local: Option<Array1<f64>>,
    /// Optional term-local inequality constraints in local coefficient coordinates.
    /// `A_local * beta_local >= b_local`.
    pub linear_constraints_local: Option<LinearInequalityConstraints>,
    /// Optional factored tensor-product representation preserved for operator-backed
    /// assembly in the main design builder.
    pub kronecker_factored: Option<KroneckerFactoredBasis>,
    /// Joint-null absorption rotation. `Some(Q)` records the orthonormal
    /// `(p_local × p_local)` matrix that was applied to this term's design
    /// and per-block penalties at construction time:
    /// `term_design ← X_raw · Q`, `penalties_local[k] ← Qᵀ · S_raw · Q`.
    /// The smooth's coefficient vector therefore lives in the rotated
    /// (`γ`) coordinate system, with `β_raw = Q · γ` recovering the raw
    /// pre-rotation parameterization. `None` means either no joint null
    /// space (penalty already full-rank) or rotation was suppressed —
    /// suppression fires when the smooth carries shape constraints
    /// (lower bounds or local linear inequalities) that would lose their
    /// cone geometry under a general orthogonal rotation.
    ///
    /// Prediction-side replay: callers building a new-data design `X_new_raw`
    /// from the *raw* basis must call [`SmoothTerm::apply_rotation_to_predict`]
    /// (or equivalent) to obtain `X_new = X_new_raw · Q` matching this
    /// term's coefficient system.
    ///
    /// Persistence replay: `freeze_term_collection_from_design` copies this
    /// rotation into `SmoothTermSpec`, which is serialized with fitted-model
    /// payloads and reused by the predict-time basis builder. Saved models
    /// therefore replay the same `X_new_raw · Q` transform as in-memory
    /// prediction.
    pub joint_null_rotation: Option<crate::terms::basis::JointNullRotation>,
    /// Global-orthogonality transform that `apply_global_smooth_identifiability`
    /// applied to this term's design but could NOT embed into `metadata`
    /// (factor-smooth kinds: `sz` metadata is per-marginal, `fs` metadata has
    /// no transform slot — #978). `freeze_term_collection_from_design` copies
    /// it onto the term's basis spec (`frozen_global_orthogonality`) so the
    /// predict-side rebuild replays it instead of emitting the unresidualized
    /// (wider) design that the fitted coefficients no longer match.
    /// Chart convention is per kind: post-`Q` `Z` for `sz` (the raw rebuild
    /// reapplies `Q` itself, #700), full `Q·Z` chart for `fs`.
    pub unabsorbed_global_orthogonality: Option<Array2<f64>>,
}

impl SmoothTerm {
    /// Apply the joint-null absorption rotation to a raw new-data design
    /// matrix, returning `X_new_raw · Q` when this term was rotated at
    /// fit time, or `X_new_raw` unchanged when no rotation was applied.
    ///
    /// Callers in the prediction path: after building the smooth's basis
    /// at new data via the *raw* basis builder (the same builder used at
    /// fit time, applied to `x_new` instead of the training rows), call
    /// this method on the resulting matrix before forming `X · β`. The
    /// fitted `β` lives in `γ`-coordinates if Q was applied; multiplying
    /// the un-rotated `X_new_raw` by `β` would give a wrong η.
    ///
    /// Returns an error if the raw design's column count does not match
    /// the rotation's `p_local`. The width invariant must hold: the raw
    /// basis builder MUST emit the same `p_local` columns that the
    /// fit-time builder did, and the rotation is `(p_local × p_local)`.
    pub fn apply_rotation_to_predict(
        &self,
        x_new_raw: Array2<f64>,
    ) -> Result<Array2<f64>, BasisError> {
        let Some(rot) = self.joint_null_rotation.as_ref() else {
            return Ok(x_new_raw);
        };
        let p_local = rot.rotation.nrows();
        if x_new_raw.ncols() != p_local {
            crate::bail_dim_basis!(
                "joint-null rotation replay for term '{}': raw design has {} columns, \
                 rotation expects {} (the raw basis builder must emit the same column \
                 count as at fit time)",
                self.name,
                x_new_raw.ncols(),
                p_local,
            );
        }
        Ok(crate::linalg::faer_ndarray::fast_ab(
            &x_new_raw,
            &rot.rotation,
        ))
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PenaltyBlockInfo {
    pub global_index: usize,
    pub termname: Option<String>,
    pub penalty: PenaltyInfo,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DroppedPenaltyBlockInfo {
    pub termname: Option<String>,
    pub penalty: PenaltyInfo,
}

#[derive(Debug, Clone)]
pub struct SmoothDesign {
    pub term_designs: Vec<DesignMatrix>,
    /// Per-term block-local penalties.  Each `col_range` is relative to the
    /// smooth block (i.e. indexing into the concatenation of `term_designs`).
    pub penalties: Vec<BlockwisePenalty>,
    pub nullspace_dims: Vec<usize>,
    pub penaltyinfo: Vec<PenaltyBlockInfo>,
    pub dropped_penaltyinfo: Vec<DroppedPenaltyBlockInfo>,
    pub terms: Vec<SmoothTerm>,
    /// Optional smooth-block lower bounds in smooth coefficient coordinates.
    /// Length equals `total_smooth_cols()` when present.
    pub coefficient_lower_bounds: Option<Array1<f64>>,
    /// Optional smooth-block inequality constraints:
    /// `A_smooth * beta_smooth >= b`.
    pub linear_constraints: Option<LinearInequalityConstraints>,
}

impl SmoothDesign {
    pub fn total_smooth_cols(&self) -> usize {
        self.term_designs.iter().map(DesignMatrix::ncols).sum()
    }
    pub fn nrows(&self) -> usize {
        self.term_designs.first().map_or(0, DesignMatrix::nrows)
    }
}

#[derive(Debug, Clone)]
pub struct RawSmoothDesign {
    pub term_designs: Vec<DesignMatrix>,
    /// Per-term block-local penalties.  Each `col_range` is relative to the
    /// smooth block (i.e. indexing into the concatenation of `term_designs`).
    pub penalties: Vec<BlockwisePenalty>,
    pub nullspace_dims: Vec<usize>,
    pub penaltyinfo: Vec<PenaltyBlockInfo>,
    pub dropped_penaltyinfo: Vec<DroppedPenaltyBlockInfo>,
    pub terms: Vec<SmoothTerm>,
    pub coefficient_lower_bounds: Option<Array1<f64>>,
    pub linear_constraints: Option<LinearInequalityConstraints>,
}

impl RawSmoothDesign {
    pub fn total_smooth_cols(&self) -> usize {
        self.term_designs.iter().map(DesignMatrix::ncols).sum()
    }
    pub fn nrows(&self) -> usize {
        self.term_designs.first().map_or(0, DesignMatrix::nrows)
    }
}

impl From<RawSmoothDesign> for SmoothDesign {
    fn from(value: RawSmoothDesign) -> Self {
        Self {
            term_designs: value.term_designs,
            penalties: value.penalties,
            nullspace_dims: value.nullspace_dims,
            penaltyinfo: value.penaltyinfo,
            dropped_penaltyinfo: value.dropped_penaltyinfo,
            terms: value.terms,
            coefficient_lower_bounds: value.coefficient_lower_bounds,
            linear_constraints: value.linear_constraints,
        }
    }
}

#[derive(Debug, Default, Clone, Serialize, Deserialize)]
pub enum BoundedCoefficientPriorSpec {
    #[default]
    None,
    Uniform,
    Beta {
        a: f64,
        b: f64,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub enum LinearCoefficientGeometry {
    #[default]
    Unconstrained,
    Bounded {
        min: f64,
        max: f64,
        #[serde(default)]
        prior: BoundedCoefficientPriorSpec,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LinearTermSpec {
    pub name: String,
    /// Primary feature column index. For Wilkinson-Rogers `:` interaction
    /// terms (`a:b[:c...]`) this is the first column in `feature_cols`; the
    /// realized design column is the elementwise product across every entry
    /// of `feature_cols`. Plain (non-interaction) linear terms set
    /// `feature_cols == vec![feature_col]`.
    pub feature_col: usize,
    /// Full list of columns whose elementwise product yields this term's
    /// design column. `len() >= 1`; `len() == 1` is a plain linear effect.
    #[serde(default)]
    pub feature_cols: Vec<usize>,
    /// Optional ridge (`S = I`, REML-selected `λ`) on this linear coefficient.
    /// A parametric linear term carries no wiggliness, so it is **unpenalized by
    /// default** — gam reports the MLE, matching mgcv/glm/survreg/VGAM (which
    /// penalize parametric terms only under an explicit `paraPen`). Set `true`
    /// to opt into an explicit shrinkage ridge (a zero-mean Gaussian prior
    /// `β ~ N(0, λ⁻¹)`); doing so adds one outer REML smoothing coordinate.
    #[serde(default = "default_linear_term_double_penalty")]
    pub double_penalty: bool,
    #[serde(default)]
    pub coefficient_geometry: LinearCoefficientGeometry,
    #[serde(default)]
    pub coefficient_min: Option<f64>,
    #[serde(default)]
    pub coefficient_max: Option<f64>,
}

impl LinearTermSpec {
    /// Return the effective list of feature columns. Backfills from
    /// `feature_col` for legacy specs that predate the multi-column field.
    pub fn effective_feature_cols(&self) -> Vec<usize> {
        if self.feature_cols.is_empty() {
            vec![self.feature_col]
        } else {
            self.feature_cols.clone()
        }
    }

    /// True when this term is a Wilkinson-Rogers `:` interaction (multi-col).
    pub fn is_interaction(&self) -> bool {
        self.feature_cols.len() > 1
    }
}

const fn default_linear_term_double_penalty() -> bool {
    // Parametric/linear terms are unpenalized by default — a single linear
    // coefficient has no roughness for a smoothing penalty to control, so the
    // historical `S = I`, REML-selected `λ` shrank every linear coefficient off
    // the MLE and injected a spurious outer smoothing coordinate (#749). Mature
    // tools (mgcv/glm/survreg/VGAM) leave parametric terms unpenalized; gam now
    // matches that and reports the MLE. An explicit `double_penalty = true`
    // still opts a term into a ridge.
    false
}

const fn default_pca_smooth_penalty() -> f64 {
    1.0
}

const fn default_pca_chunk_size() -> usize {
    4096
}

/// Random-effects term specification.
///
/// The selected feature column is interpreted as a categorical grouping variable.
/// The term contributes a one-hot dummy block with an identity penalty on group
/// coefficients, equivalent to i.i.d. Gaussian random effects.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RandomEffectTermSpec {
    pub name: String,
    pub feature_col: usize,
    /// If true, drop the lexicographically first group level to use treatment coding.
    /// If false, keep all levels (full one-hot block, still identifiable under ridge).
    pub drop_first_level: bool,
    /// If true, add a ridge penalty and estimate this block as a random effect.
    /// If false, leave the one-hot/treatment-coded block unpenalized so it is a
    /// fixed categorical main effect.  The default preserves older saved models.
    #[serde(default = "default_random_effect_penalized")]
    pub penalized: bool,
    /// Optional fixed kept-level set (sorted by f64 bit pattern) captured at fit time.
    /// When present, prediction uses exactly these columns to avoid design drift.
    #[serde(default)]
    pub frozen_levels: Option<Vec<u64>>,
}

fn default_random_effect_penalized() -> bool {
    true
}

fn validate_measure_jet_positive_vec_len(
    label: &str,
    term_name: &str,
    field: &str,
    values: &[f64],
    expected: usize,
) -> Result<(), String> {
    if values.len() != expected {
        return Err(SmoothError::invalid_config(format!(
            "{label} term '{term_name}' frozen MeasureJet {field} has length {}, expected {expected}",
            values.len()
        ))
        .into());
    }
    if values
        .iter()
        .any(|value| !(value.is_finite() && *value > 0.0))
    {
        return Err(SmoothError::invalid_config(format!(
            "{label} term '{term_name}' frozen MeasureJet {field} values must be positive and finite"
        ))
        .into());
    }
    Ok(())
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TermCollectionSpec {
    pub linear_terms: Vec<LinearTermSpec>,
    pub random_effect_terms: Vec<RandomEffectTermSpec>,
    pub smooth_terms: Vec<SmoothTermSpec>,
}

fn validate_smooth_basis_frozen(
    basis: &SmoothBasisSpec,
    label: &str,
    term_name: &str,
) -> Result<(), String> {
    match basis {
        SmoothBasisSpec::ByVariable { inner, .. }
        | SmoothBasisSpec::FactorSumToZero { inner, .. } => {
            validate_smooth_basis_frozen(inner, label, term_name)
        }
        SmoothBasisSpec::BSpline1D { spec, .. } => {
            if !matches!(
                spec.knotspec,
                BSplineKnotSpec::Provided(_) | BSplineKnotSpec::PeriodicUniform { .. }
            ) {
                return Err(format!(
                    "{label} term '{term_name}' is not frozen: BSpline knotspec must be Provided or PeriodicUniform"
                ));
            }
            Ok(())
        }
        SmoothBasisSpec::ThinPlate { spec, .. } => {
            if !matches!(spec.center_strategy, CenterStrategy::UserProvided(_)) {
                return Err(format!(
                    "{label} term '{term_name}' is not frozen: ThinPlate centers must be UserProvided"
                ));
            }
            if matches!(
                spec.identifiability,
                SpatialIdentifiability::OrthogonalToParametric
            ) {
                return Err(format!(
                    "{label} term '{term_name}' is not frozen: ThinPlate identifiability must be FrozenTransform or None"
                ));
            }
            Ok(())
        }
        _ => Ok(()),
    }
}

impl TermCollectionSpec {
    /// Write this collection's topology identity into a warm-start cache
    /// fingerprint (#869).
    ///
    /// The persistent warm-start `cache_key` hashes only family + raw input
    /// dimensions, so two fits on the same data that differ *only* in their
    /// smooth topology (the `s(..., type=AUTO)` candidate enumeration: sphere
    /// vs torus vs euclidean vs duchon) collide on one key and seed each other
    /// with geometrically incompatible β/ρ. Folding the per-term structural
    /// kind + feature columns + linear/random-effect counts into the shape hash
    /// gives each candidate its own key, so the screen→full-refit reuse of one
    /// candidate is preserved while cross-candidate contamination is removed.
    /// Only the structural identity is hashed (not fitted coefficients or
    /// frozen knot values), so a refit of the *same* topology still hits.
    pub fn write_structural_shape_hash(&self, h: &mut crate::warm_start::Fingerprinter) {
        h.write_str("term-collection");
        h.write_usize(self.linear_terms.len());
        for linear in &self.linear_terms {
            h.write_str(&linear.name);
        }
        h.write_usize(self.random_effect_terms.len());
        h.write_usize(self.smooth_terms.len());
        for smooth in &self.smooth_terms {
            h.write_str(&smooth.name);
            h.write_str(smooth.basis.structural_kind());
            for col in smooth.basis.structural_feature_cols() {
                h.write_usize(col);
            }
        }
    }

    /// Validate that a term collection spec represents a fully frozen model
    /// (i.e. all knots/centers are pre-computed, identifiability transforms are
    /// baked in, and random-effect levels are fixed).
    pub fn validate_frozen(&self, label: &str) -> Result<(), String> {
        for linear in &self.linear_terms {
            if let (Some(min), Some(max)) = (linear.coefficient_min, linear.coefficient_max)
                && (!min.is_finite() || !max.is_finite() || min > max)
            {
                return Err(SmoothError::invalid_config(format!(
                    "{label} linear term '{}' has invalid coefficient constraint [{min}, {max}]",
                    linear.name
                ))
                .into());
            }
            if let Some(min) = linear.coefficient_min
                && !min.is_finite()
            {
                return Err(SmoothError::invalid_config(format!(
                    "{label} linear term '{}' has non-finite coefficient minimum {min}",
                    linear.name
                ))
                .into());
            }
            if let Some(max) = linear.coefficient_max
                && !max.is_finite()
            {
                return Err(SmoothError::invalid_config(format!(
                    "{label} linear term '{}' has non-finite coefficient maximum {max}",
                    linear.name
                ))
                .into());
            }
            if let LinearCoefficientGeometry::Bounded { min, max, prior } =
                &linear.coefficient_geometry
            {
                if !min.is_finite() || !max.is_finite() || min >= max {
                    return Err(SmoothError::invalid_config(format!(
                        "{label} bounded term '{}' has invalid bounds [{min}, {max}]",
                        linear.name
                    ))
                    .into());
                }
                match prior {
                    BoundedCoefficientPriorSpec::None | BoundedCoefficientPriorSpec::Uniform => {}
                    BoundedCoefficientPriorSpec::Beta { a, b } => {
                        if !a.is_finite() || !b.is_finite() || *a < 1.0 || *b < 1.0 {
                            return Err(SmoothError::invalid_config(format!(
                                "{label} bounded term '{}' has invalid Beta prior ({a}, {b})",
                                linear.name
                            ))
                            .into());
                        }
                    }
                }
            }
        }
        for st in &self.smooth_terms {
            match &st.basis {
                SmoothBasisSpec::ByVariable { inner, .. } => {
                    validate_smooth_basis_frozen(inner, label, &st.name)?;
                    let nested = SmoothTermSpec {
                        name: st.name.clone(),
                        basis: (**inner).clone(),
                        shape: st.shape,
                        joint_null_rotation: None,
                    };
                    TermCollectionSpec {
                        linear_terms: Vec::new(),
                        random_effect_terms: Vec::new(),
                        smooth_terms: vec![nested],
                    }
                    .validate_frozen(label)?;
                }
                SmoothBasisSpec::FactorSumToZero { inner, levels, .. } => {
                    if levels.len() < 2 {
                        return Err(format!(
                            "{label} term '{}' has invalid frozen sz levels",
                            st.name
                        ));
                    }
                    validate_smooth_basis_frozen(inner, label, &st.name)?;
                }
                SmoothBasisSpec::BSpline1D { spec, .. } => {
                    if !matches!(
                        spec.knotspec,
                        BSplineKnotSpec::Provided(_) | BSplineKnotSpec::PeriodicUniform { .. }
                    ) {
                        return Err(SmoothError::invalid_config(format!(
                            "{label} term '{}' is not frozen: BSpline knotspec must be Provided or PeriodicUniform",
                            st.name
                        ))
                        .into());
                    }
                }
                SmoothBasisSpec::ThinPlate { spec, .. } => {
                    if !matches!(spec.center_strategy, CenterStrategy::UserProvided(_)) {
                        return Err(SmoothError::invalid_config(format!(
                            "{label} term '{}' is not frozen: ThinPlate centers must be UserProvided",
                            st.name
                        ))
                        .into());
                    }
                    if matches!(
                        spec.identifiability,
                        SpatialIdentifiability::OrthogonalToParametric
                    ) {
                        return Err(SmoothError::invalid_config(format!(
                            "{label} term '{}' is not frozen: ThinPlate identifiability must be FrozenTransform or None",
                            st.name
                        ))
                        .into());
                    }
                }
                SmoothBasisSpec::Sphere { spec, .. } => {
                    if !matches!(spec.center_strategy, CenterStrategy::UserProvided(_)) {
                        return Err(SmoothError::invalid_config(format!(
                            "{label} term '{}' is not frozen: Sphere centers must be UserProvided",
                            st.name
                        ))
                        .into());
                    }
                    if matches!(spec.method, crate::basis::SphereMethod::Harmonic)
                        && spec.max_degree.is_none_or(|d| d == 0)
                    {
                        return Err(format!(
                            "{label} term '{}' is not frozen: sphere max_degree must be positive",
                            st.name
                        ));
                    }
                }
                SmoothBasisSpec::ConstantCurvature { spec, .. } => {
                    if !matches!(spec.center_strategy, CenterStrategy::UserProvided(_)) {
                        return Err(SmoothError::invalid_config(format!(
                            "{label} term '{}' is not frozen: ConstantCurvature centers must be UserProvided",
                            st.name
                        ))
                        .into());
                    }
                    if !(spec.length_scale.is_finite() && spec.length_scale > 0.0) {
                        return Err(SmoothError::invalid_config(format!(
                            "{label} term '{}' is not frozen: ConstantCurvature length_scale must be the realized positive value",
                            st.name
                        ))
                        .into());
                    }
                }
                SmoothBasisSpec::MeasureJet { spec, .. } => {
                    let centers = match &spec.center_strategy {
                        CenterStrategy::UserProvided(centers) => centers,
                        _ => {
                            return Err(SmoothError::invalid_config(format!(
                                "{label} term '{}' is not frozen: MeasureJet centers must be UserProvided",
                                st.name
                            ))
                            .into());
                        }
                    };
                    if centers.nrows() == 0 {
                        return Err(SmoothError::invalid_config(format!(
                            "{label} term '{}' is not frozen: MeasureJet centers are empty",
                            st.name
                        ))
                        .into());
                    }
                    if !(spec.length_scale.is_finite() && spec.length_scale > 0.0) {
                        return Err(SmoothError::invalid_config(format!(
                            "{label} term '{}' is not frozen: MeasureJet length_scale must be the realized positive value",
                            st.name
                        ))
                        .into());
                    }
                    // Exact replay needs the fit-data penalty quadrature and
                    // normalization payload (`BasisMetadata::MeasureJet`).
                    let frozen = spec.frozen_quadrature.as_ref().ok_or_else(|| {
                        SmoothError::invalid_config(format!(
                            "{label} term '{}' is not frozen: MeasureJet frozen_quadrature payload is missing",
                            st.name
                        ))
                    })?;
                    if frozen.masses.len() != centers.nrows() {
                        return Err(SmoothError::invalid_config(format!(
                            "{label} term '{}' frozen MeasureJet has {} masses for {} centers",
                            st.name,
                            frozen.masses.len(),
                            centers.nrows()
                        ))
                        .into());
                    }
                    let total_mass = frozen.masses.sum();
                    if frozen
                        .masses
                        .iter()
                        .any(|mass| !(mass.is_finite() && *mass >= 0.0))
                        || !(total_mass.is_finite() && total_mass > 0.0)
                    {
                        return Err(SmoothError::invalid_config(format!(
                            "{label} term '{}' frozen MeasureJet masses must be finite, nonnegative, and have positive total mass",
                            st.name
                        ))
                        .into());
                    }
                    let n_levels = frozen.eps_band.len();
                    if n_levels == 0
                        || frozen
                            .eps_band
                            .iter()
                            .any(|eps| !(eps.is_finite() && *eps > 0.0))
                    {
                        return Err(SmoothError::invalid_config(format!(
                            "{label} term '{}' frozen MeasureJet eps_band must be nonempty, finite, and positive",
                            st.name
                        ))
                        .into());
                    }
                    for (idx, pair) in frozen.eps_band.windows(2).enumerate() {
                        if pair[1] <= pair[0] {
                            return Err(SmoothError::invalid_config(format!(
                                "{label} term '{}' frozen MeasureJet eps_band is not strictly ascending at {idx}: {} then {}",
                                st.name,
                                pair[0],
                                pair[1]
                            ))
                            .into());
                        }
                    }
                    validate_measure_jet_positive_vec_len(
                        label,
                        &st.name,
                        "support_means",
                        &frozen.support_means,
                        n_levels,
                    )?;
                    // Mode predicate MUST match the builder's
                    // (`measure_jet_multiscale_mode`): per-level/multiscale is the
                    // explicit `spec.multiscale` opt-in (#1116). In single-scale
                    // mode the builder emits a single FUSED penalty (empty
                    // per-level scales + `fused_penalty_normalization_scale:
                    // Some`); only the multiscale opt-in carries `n_levels`
                    // per-level scales.
                    let per_level = crate::basis::measure_jet_multiscale_mode(spec);
                    if per_level {
                        validate_measure_jet_positive_vec_len(
                            label,
                            &st.name,
                            "penalty_normalization_scales",
                            &frozen.penalty_normalization_scales,
                            n_levels,
                        )?;
                        validate_measure_jet_positive_vec_len(
                            label,
                            &st.name,
                            "raw_penalty_normalization_scales",
                            &frozen.raw_penalty_normalization_scales,
                            n_levels,
                        )?;
                        if frozen.fused_penalty_normalization_scale.is_some() {
                            return Err(SmoothError::invalid_config(format!(
                                "{label} term '{}' per-level MeasureJet must not carry a fused penalty normalization scale",
                                st.name
                            ))
                            .into());
                        }
                    } else {
                        if !frozen.penalty_normalization_scales.is_empty()
                            || !frozen.raw_penalty_normalization_scales.is_empty()
                        {
                            return Err(SmoothError::invalid_config(format!(
                                "{label} term '{}' fused MeasureJet must not carry per-level penalty normalization scales",
                                st.name
                            ))
                            .into());
                        }
                        match frozen.fused_penalty_normalization_scale {
                            Some(scale) if scale.is_finite() && scale > 0.0 => {}
                            Some(scale) => {
                                return Err(SmoothError::invalid_config(format!(
                                    "{label} term '{}' fused MeasureJet penalty normalization scale must be positive and finite, got {scale}",
                                    st.name
                                ))
                                .into());
                            }
                            None => {
                                return Err(SmoothError::invalid_config(format!(
                                    "{label} term '{}' fused MeasureJet is missing its penalty normalization scale",
                                    st.name
                                ))
                                .into());
                            }
                        }
                    }
                }
                SmoothBasisSpec::Matern { spec, .. } => {
                    if !matches!(spec.center_strategy, CenterStrategy::UserProvided(_)) {
                        return Err(SmoothError::invalid_config(format!(
                            "{label} term '{}' is not frozen: Matern centers must be UserProvided",
                            st.name
                        ))
                        .into());
                    }
                }
                SmoothBasisSpec::Duchon { spec, .. } => {
                    if !matches!(spec.center_strategy, CenterStrategy::UserProvided(_)) {
                        return Err(SmoothError::invalid_config(format!(
                            "{label} term '{}' is not frozen: Duchon centers must be UserProvided",
                            st.name
                        ))
                        .into());
                    }
                    if matches!(
                        spec.identifiability,
                        SpatialIdentifiability::OrthogonalToParametric
                    ) {
                        return Err(SmoothError::invalid_config(format!(
                            "{label} term '{}' is not frozen: Duchon identifiability must be FrozenTransform or None",
                            st.name
                        ))
                        .into());
                    }
                }
                SmoothBasisSpec::Pca {
                    centered,
                    center_mean,
                    pca_basis_path,
                    ..
                } => {
                    if *centered && center_mean.is_none() && pca_basis_path.is_none() {
                        return Err(SmoothError::invalid_config(format!(
                            "{label} term '{}' is not frozen: centered Pca missing center_mean",
                            st.name
                        ))
                        .into());
                    }
                }
                SmoothBasisSpec::BySmooth { smooth, by_kind } => {
                    if let SmoothBasisSpec::BySmooth { .. } = smooth.as_ref() {
                        return Err(format!("{label} term '{}' has nested by-smooths", st.name));
                    }
                    match by_kind {
                        ByVarKind::Numeric { .. } => {}
                        ByVarKind::Factor { frozen_levels, .. } if frozen_levels.is_none() => {
                            return Err(format!(
                                "{label} term '{}' is not frozen: by-factor levels missing",
                                st.name
                            ));
                        }
                        ByVarKind::Factor { .. } => {}
                    }
                    let nested = TermCollectionSpec {
                        linear_terms: vec![],
                        random_effect_terms: vec![],
                        smooth_terms: vec![SmoothTermSpec {
                            name: st.name.clone(),
                            basis: (**smooth).clone(),
                            shape: st.shape,
                            joint_null_rotation: None,
                        }],
                    };
                    nested.validate_frozen(label)?;
                }
                SmoothBasisSpec::FactorSmooth { spec } => {
                    if spec.group_frozen_levels.is_none() {
                        return Err(format!(
                            "{label} term '{}' is not frozen: factor-smooth levels missing",
                            st.name
                        ));
                    }
                    if !matches!(
                        spec.marginal.knotspec,
                        BSplineKnotSpec::Provided(_) | BSplineKnotSpec::PeriodicUniform { .. }
                    ) {
                        return Err(format!(
                            "{label} term '{}' is not frozen: factor-smooth marginal knots missing",
                            st.name
                        ));
                    }
                }
                SmoothBasisSpec::TensorBSpline { spec, .. } => {
                    for (dim, marginal) in spec.marginalspecs.iter().enumerate() {
                        if !matches!(
                            marginal.knotspec,
                            BSplineKnotSpec::Provided(_) | BSplineKnotSpec::PeriodicUniform { .. }
                        ) {
                            return Err(SmoothError::invalid_config(format!(
                                "{label} term '{}' dim {} is not frozen: tensor marginal knotspec must be Provided or PeriodicUniform",
                                st.name, dim
                            ))
                            .into());
                        }
                    }
                    if matches!(
                        spec.identifiability,
                        TensorBSplineIdentifiability::SumToZero
                            | TensorBSplineIdentifiability::MarginalSumToZero
                    ) {
                        return Err(SmoothError::invalid_config(format!(
                            "{label} term '{}' is not frozen: tensor identifiability must be FrozenTransform or None",
                            st.name
                        ))
                        .into());
                    }
                }
            }
        }

        for rt in &self.random_effect_terms {
            if rt.frozen_levels.is_none() {
                return Err(SmoothError::invalid_config(format!(
                    "{label} random-effect term '{}' is not frozen: missing frozen_levels",
                    rt.name
                ))
                .into());
            }
        }

        Ok(())
    }

    /// Re-resolve every stored feature-column index through `remap`, returning a
    /// spec that addresses a different column layout.
    ///
    /// A frozen `TermCollectionSpec` stores feature columns as *absolute indices
    /// into the training table*. To replay it on a fresh dataset whose columns
    /// sit at different positions — the common case at prediction time, where
    /// the response column is unknown and may be absent entirely — every index
    /// must be re-resolved against the new layout. `remap` receives each
    /// training-table index and returns its position in the runtime table;
    /// callers typically implement it as "look the name up in the training
    /// headers, then resolve that name against the prediction dataset".
    ///
    /// This is the single authority on *which* fields carry a column index
    /// across every basis variant (linear, random-effect, the `by=` column of
    /// `ByVariable`/`FactorSumToZero`/`BySmooth`, the continuous and group
    /// columns of a `FactorSmooth`, and the multi-axis `feature_cols` of every
    /// spatial/tensor basis), so a predict-time realignment cannot silently miss
    /// one and dereference a stale training index.
    pub fn remap_feature_columns<E, F>(&self, mut remap: F) -> Result<TermCollectionSpec, E>
    where
        F: FnMut(usize) -> Result<usize, E>,
    {
        let mut out = self.clone();
        for lt in &mut out.linear_terms {
            lt.feature_col = remap(lt.feature_col)?;
            // Also remap the full interaction-factor list. The design builder
            // (`build_term_collection_design_inner`) materializes the column from
            // `effective_feature_cols()` — which returns `feature_cols` whenever
            // it is non-empty (i.e. essentially always, including a plain linear
            // term where `feature_cols == [feature_col]`). Remapping only the
            // singular `feature_col` left these at their saved *training* indices
            // at predict time, so a parametric `Surv(...) ~ x` (and any `:`
            // interaction) bailed with "feature column N out of bounds" once the
            // response/time columns shift the runtime layout (issue #898).
            for fc in lt.feature_cols.iter_mut() {
                *fc = remap(*fc)?;
            }
        }
        for rt in &mut out.random_effect_terms {
            rt.feature_col = remap(rt.feature_col)?;
        }
        for st in &mut out.smooth_terms {
            remap_smooth_basis_feature_columns(&mut st.basis, &mut remap)?;
        }
        Ok(out)
    }
}

/// Walk a `SmoothBasisSpec` tree, re-resolving every column index through
/// `remap`. Shared by all predict-time column realignment (see
/// [`TermCollectionSpec::remap_feature_columns`]); kept exhaustive so a newly
/// added index-bearing variant fails to compile until it is handled here.
fn remap_smooth_basis_feature_columns<E, F>(
    basis: &mut SmoothBasisSpec,
    remap: &mut F,
) -> Result<(), E>
where
    F: FnMut(usize) -> Result<usize, E>,
{
    match basis {
        SmoothBasisSpec::ByVariable { inner, by_col, .. }
        | SmoothBasisSpec::FactorSumToZero { inner, by_col, .. } => {
            *by_col = remap(*by_col)?;
            remap_smooth_basis_feature_columns(inner, remap)?;
        }
        SmoothBasisSpec::BSpline1D { feature_col, .. } => {
            *feature_col = remap(*feature_col)?;
        }
        SmoothBasisSpec::BySmooth { smooth, by_kind } => {
            let by_feature_col = match by_kind {
                ByVarKind::Numeric { feature_col } | ByVarKind::Factor { feature_col, .. } => {
                    feature_col
                }
            };
            *by_feature_col = remap(*by_feature_col)?;
            remap_smooth_basis_feature_columns(smooth, remap)?;
        }
        SmoothBasisSpec::FactorSmooth { spec } => {
            for fc in spec.continuous_cols.iter_mut() {
                *fc = remap(*fc)?;
            }
            spec.group_col = remap(spec.group_col)?;
        }
        SmoothBasisSpec::ThinPlate { feature_cols, .. }
        | SmoothBasisSpec::Sphere { feature_cols, .. }
        | SmoothBasisSpec::ConstantCurvature { feature_cols, .. }
        | SmoothBasisSpec::Matern { feature_cols, .. }
        | SmoothBasisSpec::MeasureJet { feature_cols, .. }
        | SmoothBasisSpec::Duchon { feature_cols, .. }
        | SmoothBasisSpec::Pca { feature_cols, .. }
        | SmoothBasisSpec::TensorBSpline { feature_cols, .. } => {
            for fc in feature_cols.iter_mut() {
                *fc = remap(*fc)?;
            }
        }
    }
    Ok(())
}

#[derive(Debug, Clone)]
pub enum PenaltyStructureHint {
    Ridge(f64),
    Kronecker(Vec<Array2<f64>>),
}

/// A penalty matrix stored at its natural block size together with the
/// column range it occupies in the global coefficient vector.
///
/// Instead of embedding every penalty into a full `p_total × p_total` dense
/// matrix filled with zeros, we keep the compact local matrix and reconstruct
/// the global view only when a downstream consumer explicitly requires it.
#[derive(Clone)]
pub struct BlockwisePenalty {
    /// Column range in the global coefficient vector that this penalty covers.
    pub col_range: Range<usize>,
    /// The local penalty matrix — dimensions `block_p × block_p` where
    /// `block_p = col_range.len()`.
    pub local: Array2<f64>,
    /// Optional nonzero centering vector for this coefficient block.
    pub prior_mean: crate::solver::estimate::CoefficientPriorMean,
    /// Optional structural hint so downstream spectral/logdet code can stay
    /// block-local or factorized without reverse-engineering the matrix.
    pub structure_hint: Option<PenaltyStructureHint>,
    /// Optional operator-form handle bit-equivalent to `local`. Populated when
    /// the originating closed-form factory emitted an op-form penalty so exact
    /// operator algebra can use matvec instead of materializing the dense
    /// `block_p × block_p` Gram. `None` for ordinary dense penalties.
    pub op: Option<std::sync::Arc<dyn crate::terms::analytic_penalties::PenaltyOp>>,
}

impl std::fmt::Debug for BlockwisePenalty {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("BlockwisePenalty")
            .field("col_range", &self.col_range)
            .field(
                "local",
                &format_args!("{}×{}", self.local.nrows(), self.local.ncols()),
            )
            .field("prior_mean", &self.prior_mean)
            .field("structure_hint", &self.structure_hint)
            .field("op", &self.op.as_ref().map(|o| o.dim()))
            .finish()
    }
}

impl BlockwisePenalty {
    /// Create a new blockwise penalty.
    pub fn new(col_range: Range<usize>, local: Array2<f64>) -> Self {
        assert_eq!(col_range.len(), local.nrows());
        assert_eq!(col_range.len(), local.ncols());
        Self {
            col_range,
            local,
            prior_mean: crate::solver::estimate::CoefficientPriorMean::Zero,
            structure_hint: None,
            op: None,
        }
    }

    pub fn with_prior_mean(
        mut self,
        prior_mean: crate::solver::estimate::CoefficientPriorMean,
    ) -> Self {
        self.prior_mean = prior_mean;
        self
    }

    /// Attach an op-form penalty handle bit-equivalent to `local`.
    pub fn with_op(
        mut self,
        op: Option<std::sync::Arc<dyn crate::terms::analytic_penalties::PenaltyOp>>,
    ) -> Self {
        self.op = op;
        self
    }

    pub fn ridge(col_range: Range<usize>, scale: f64) -> Self {
        let block_size = col_range.len();
        let mut local = Array2::<f64>::zeros((block_size, block_size));
        for i in 0..block_size {
            local[[i, i]] = scale;
        }
        Self {
            col_range,
            local,
            prior_mean: crate::solver::estimate::CoefficientPriorMean::Zero,
            structure_hint: Some(PenaltyStructureHint::Ridge(scale)),
            op: None,
        }
    }

    pub fn kronecker(
        col_range: Range<usize>,
        local: Array2<f64>,
        factors: Vec<Array2<f64>>,
    ) -> Self {
        assert_eq!(col_range.len(), local.nrows());
        assert_eq!(col_range.len(), local.ncols());
        Self {
            col_range,
            local,
            prior_mean: crate::solver::estimate::CoefficientPriorMean::Zero,
            structure_hint: Some(PenaltyStructureHint::Kronecker(factors)),
            op: None,
        }
    }

    /// Expand this blockwise penalty into a full `p_total × p_total` dense
    /// matrix (mostly zeros). Use sparingly — the whole point of blockwise
    /// storage is to avoid this allocation.
    pub fn to_global(&self, p_total: usize) -> Array2<f64> {
        let mut g = Array2::<f64>::zeros((p_total, p_total));
        let r = &self.col_range;
        assert!(
            r.end <= p_total && self.local.nrows() == r.len() && self.local.ncols() == r.len(),
            "BlockwisePenalty::to_global shape invariant violated: \
             col_range={}..{}, local={}x{}, p_total={}",
            r.start,
            r.end,
            self.local.nrows(),
            self.local.ncols(),
            p_total,
        );
        g.slice_mut(s![r.start..r.end, r.start..r.end])
            .assign(&self.local);
        g
    }

    /// Convert into a blockwise [`crate::custom_family::PenaltyMatrix`] without
    /// expanding to full dimensions.
    pub(crate) fn to_penalty_matrix(
        &self,
        total_dim: usize,
    ) -> crate::custom_family::PenaltyMatrix {
        crate::custom_family::PenaltyMatrix::Blockwise {
            local: self.local.clone(),
            col_range: self.col_range.clone(),
            total_dim,
        }
    }

    /// The block size of this penalty.
    #[inline]
    pub fn block_size(&self) -> usize {
        self.col_range.len()
    }
}

/// Compute `Σ_k λ_k S_k` directly from blockwise penalties, accumulating
/// into a pre-allocated `p_total × p_total` output without ever materializing
/// individual global matrices.
pub fn weighted_blockwise_penalty_sum(
    penalties: &[BlockwisePenalty],
    lambdas: &[f64],
    p_total: usize,
) -> Array2<f64> {
    assert_eq!(penalties.len(), lambdas.len());
    // Smoothing parameters λ_k must be non-negative and finite. A negative
    // λ would flip the sign of the corresponding block S_k, turning the
    // total penalty matrix indefinite and silently corrupting every
    // downstream Cholesky / PIRLS / REML / pseudo-logdet computation that
    // assumes S_λ ⪰ 0. Catch this at the boundary rather than after it
    // has propagated.
    for (idx, &lam) in lambdas.iter().enumerate() {
        assert!(
            lam.is_finite() && lam >= 0.0,
            "weighted_blockwise_penalty_sum: lambdas[{idx}] = {lam} is invalid (must be finite and non-negative; negative smoothing parameters violate S_λ ⪰ 0)",
        );
    }
    // Block column ranges must also fit inside the declared total parameter
    // dimension; an out-of-bounds slice would otherwise panic from ndarray
    // with a far less informative message.
    for (idx, bp) in penalties.iter().enumerate() {
        let r = &bp.col_range;
        assert!(
            r.end <= p_total,
            "weighted_blockwise_penalty_sum: penalties[{idx}] col_range {:?} exceeds p_total = {p_total}",
            r,
        );
    }
    let mut out = Array2::<f64>::zeros((p_total, p_total));
    for (bp, &lam) in penalties.iter().zip(lambdas.iter()) {
        let r = &bp.col_range;
        let mut slice = out.slice_mut(s![r.start..r.end, r.start..r.end]);
        slice.scaled_add(lam, &bp.local);
    }
    out
}

// ---------------------------------------------------------------------------
// KroneckerPenaltySystem — factored tensor-product penalty representation
// ---------------------------------------------------------------------------

/// Factored representation of tensor-product penalties with precomputed
/// marginal eigensystems for O(∏q_j) logdet and penalty operations.
#[derive(Debug, Clone)]
pub struct KroneckerPenaltySystem {
    /// Marginal penalty matrices: `marginal_penalties[k]` is `(q_k, q_k)`.
    pub marginal_penalties: Vec<Array2<f64>>,
    /// Precomputed eigensystems: `(eigenvalues, eigenvectors)` per marginal.
    pub marginal_eigensystems: Vec<(Array1<f64>, Array2<f64>)>,
    /// Marginal basis dimensions.
    pub marginal_dims: Vec<usize>,
    /// Whether a global ridge (double) penalty is present.
    pub has_double_penalty: bool,
}

impl KroneckerPenaltySystem {
    pub fn new(
        marginal_penalties: Vec<Array2<f64>>,
        marginal_dims: Vec<usize>,
        has_double_penalty: bool,
    ) -> Result<Self, BasisError> {
        if marginal_penalties.len() != marginal_dims.len() {
            crate::bail_dim_basis!(
                "KroneckerPenaltySystem: {} penalties vs {} dims",
                marginal_penalties.len(),
                marginal_dims.len()
            );
        }
        let eigensystems =
            kronecker_marginal_eigensystems(&marginal_penalties, "KroneckerPenaltySystem")
                .map_err(|e| BasisError::InvalidInput(e.to_string()))?;
        Ok(Self {
            marginal_penalties,
            marginal_eigensystems: eigensystems,
            marginal_dims,
            has_double_penalty,
        })
    }

    pub fn p_total(&self) -> usize {
        self.marginal_dims.iter().copied().product()
    }

    pub fn ndim(&self) -> usize {
        self.marginal_dims.len()
    }

    pub fn num_penalties(&self) -> usize {
        self.marginal_dims.len() + if self.has_double_penalty { 1 } else { 0 }
    }

    /// Compute `log|S|₊` and its first/second derivatives w.r.t. `ρ_k = log(λ_k)`.
    ///
    /// Iterates over the ∏q_j multi-index grid. Cost: O(d · ∏q_j), no O(p²) storage.
    pub fn logdet_and_derivatives(
        &self,
        lambdas: &[f64],
        ridge: f64,
    ) -> (f64, Array1<f64>, Array2<f64>) {
        let n_pen = self.num_penalties();
        assert_eq!(lambdas.len(), n_pen, "lambda count mismatch");
        let marginal_evals: Vec<_> = self
            .marginal_eigensystems
            .iter()
            .map(|(evals, _)| evals.view())
            .collect();
        kronecker_logdet_and_derivatives(
            &marginal_evals,
            &self.marginal_dims,
            lambdas,
            self.has_double_penalty,
            ridge,
        )
    }

    pub fn logdet_rank_and_derivatives(
        &self,
        lambdas: &[f64],
        ridge: f64,
    ) -> (f64, usize, Array1<f64>, Array2<f64>) {
        let n_pen = self.num_penalties();
        assert_eq!(lambdas.len(), n_pen, "lambda count mismatch");
        let d = self.marginal_dims.len();
        let mut logdet = 0.0;
        let mut rank = 0usize;
        let mut grad = Array1::<f64>::zeros(n_pen);
        let mut hess = Array2::<f64>::zeros((n_pen, n_pen));
        // Positivity floor for a penalized eigenvalue `σ`: below this the mode
        // is treated as an unpenalized (null-space) direction and excluded from
        // both the rank count and the pseudo-log-determinant.
        const EIGENVALUE_POSITIVITY_FLOOR: f64 = 1e-12;
        // Floor on the *structural* eigenvalue sum (λ-independent) used to decide
        // whether a mode lives in the penalty range space and so should receive
        // the stabilizing ridge; a structurally-null mode gets no ridge.
        const STRUCTURAL_ZERO_FLOOR: f64 = 1e-12;
        let mut multi_idx = vec![0usize; d];
        loop {
            let mut sigma = 0.0;
            let mut structural_sigma = 0.0;
            for k in 0..d {
                let marginal_eigenvalue = self.marginal_eigensystems[k].0[multi_idx[k]];
                structural_sigma += marginal_eigenvalue;
                sigma += lambdas[k] * marginal_eigenvalue;
            }
            if self.has_double_penalty {
                structural_sigma += 1.0;
                sigma += lambdas[d];
            }
            if structural_sigma > STRUCTURAL_ZERO_FLOOR {
                sigma += ridge;
            }

            if sigma > EIGENVALUE_POSITIVITY_FLOOR {
                rank += 1;
                logdet += sigma.ln();
                let inv_sigma = 1.0 / sigma;
                let inv_sigma2 = inv_sigma * inv_sigma;
                for k in 0..n_pen {
                    let ck = if k < d {
                        lambdas[k] * self.marginal_eigensystems[k].0[multi_idx[k]]
                    } else {
                        lambdas[d]
                    };
                    grad[k] += ck * inv_sigma;
                    hess[[k, k]] += ck * inv_sigma - ck * ck * inv_sigma2;
                    for l in (k + 1)..n_pen {
                        let cl = if l < d {
                            lambdas[l] * self.marginal_eigensystems[l].0[multi_idx[l]]
                        } else {
                            lambdas[d]
                        };
                        let off = -ck * cl * inv_sigma2;
                        hess[[k, l]] += off;
                        hess[[l, k]] += off;
                    }
                }
            }

            let mut carry = true;
            for dim in (0..d).rev() {
                if carry {
                    multi_idx[dim] += 1;
                    if multi_idx[dim] < self.marginal_dims[dim] {
                        carry = false;
                    } else {
                        multi_idx[dim] = 0;
                    }
                }
            }
            if carry {
                break;
            }
        }
        (logdet, rank, grad, hess)
    }
}

#[derive(Clone, Debug)]
pub struct TermCollectionDesign {
    /// The full design matrix.
    ///
    /// Prefer a true sparse matrix when every block is sparse-compatible.
    /// If the collection already contains intrinsically sparse blocks, preserve
    /// that storage and let PIRLS decide later whether the penalized system is
    /// sparse-native eligible. Purely dense materialized blocks still fall back
    /// to the lazy block operator when sparse storage would just re-encode a
    /// dense matrix.
    pub design: DesignMatrix,
    pub penalties: Vec<BlockwisePenalty>,
    pub nullspace_dims: Vec<usize>,
    pub penaltyinfo: Vec<PenaltyBlockInfo>,
    pub dropped_penaltyinfo: Vec<DroppedPenaltyBlockInfo>,
    /// Optional global coefficient lower bounds for constrained fitting.
    /// Length equals `design.ncols()` when present. Unconstrained entries are `-inf`.
    pub coefficient_lower_bounds: Option<Array1<f64>>,
    /// Optional global inequality constraints:
    /// `A * beta >= b`.
    pub linear_constraints: Option<LinearInequalityConstraints>,
    pub intercept_range: Range<usize>,
    pub linear_ranges: Vec<(String, Range<usize>)>,
    pub random_effect_ranges: Vec<(String, Range<usize>)>,
    pub random_effect_levels: Vec<(String, Vec<u64>)>,
    pub smooth: SmoothDesign,
}

impl TermCollectionDesign {
    /// Convert blockwise penalties to `PenaltyMatrix::Blockwise` without
    /// expanding to `p_total × p_total`. This is the preferred path for
    /// family modules that accept `Vec<PenaltyMatrix>`.
    pub fn penalties_as_penalty_matrix(&self) -> Vec<crate::custom_family::PenaltyMatrix> {
        let p = self.design.ncols();
        self.penalties
            .iter()
            .map(|bp| bp.to_penalty_matrix(p))
            .collect()
    }

    /// Number of penalty blocks.
    #[inline]
    pub fn num_penalties(&self) -> usize {
        self.penalties.len()
    }

    /// Resolve coefficient groups against this design's global coefficient
    /// layout and append their penalties after the existing term penalties.
    pub fn realize_coefficient_groups(
        &self,
        groups: &[CoefficientGroupSpec],
        base_prior: &crate::types::RhoPrior,
    ) -> Result<RealizedCoefficientGroups, BasisError> {
        realize_coefficient_groups(self, groups, base_prior)
    }

    /// Extract a `KroneckerPenaltySystem` if exactly one smooth term has
    /// Kronecker structure and it accounts for all penalties.
    ///
    /// Returns `None` if no Kronecker structure is present or if the model
    /// has multiple smooth terms with mixed structure.
    pub fn kronecker_penalty_system(&self) -> Option<KroneckerPenaltySystem> {
        let kron_terms: Vec<&KroneckerFactoredBasis> = self
            .smooth
            .terms
            .iter()
            .filter_map(|t| t.kronecker_factored.as_ref())
            .collect();
        if kron_terms.len() != 1 {
            return None; // 0 or multiple Kronecker terms — not supported yet
        }
        let kron = kron_terms[0];
        // Only use the Kronecker path when the model is purely this tensor term
        // (no other smooth terms with separate penalties).
        let has_non_kron_smooth_terms = self
            .smooth
            .terms
            .iter()
            .any(|t| t.kronecker_factored.is_none());
        if has_non_kron_smooth_terms {
            return None; // mixed model — fall back to standard path
        }
        KroneckerPenaltySystem::new(
            kron.marginal_penalties.clone(),
            kron.marginal_dims.clone(),
            kron.has_double_penalty,
        )
        .ok()
    }
}

#[derive(Clone)]
pub struct FittedTermCollection {
    pub fit: UnifiedFitResult,
    pub design: TermCollectionDesign,
    pub adaptive_diagnostics: Option<AdaptiveRegularizationDiagnostics>,
}

#[derive(Clone)]
pub struct FittedTermCollectionWithSpec {
    pub fit: UnifiedFitResult,
    pub design: TermCollectionDesign,
    pub resolvedspec: TermCollectionSpec,
    pub adaptive_diagnostics: Option<AdaptiveRegularizationDiagnostics>,
}

#[derive(Clone)]
pub struct StandardLatentCoordConfig {
    pub values: std::sync::Arc<crate::terms::latent::LatentCoordValues>,
    pub term_index: crate::types::SmoothTermIdx,
    pub feature_cols: Vec<usize>,
    pub manifold: crate::terms::latent::LatentManifold,
    pub manifold_auto: bool,
    pub retraction_registry: crate::solver::latent_cache::LatentRetractionRegistry,
    pub analytic_penalties: Option<std::sync::Arc<crate::terms::AnalyticPenaltyRegistry>>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AdaptiveSpatialMap {
    pub termname: String,
    pub feature_cols: Vec<usize>,
    pub collocation_points: Array2<f64>,
    pub inv_magweight: Array1<f64>,
    pub invgradweight: Array1<f64>,
    pub inv_lapweight: Array1<f64>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AdaptiveRegularizationDiagnostics {
    pub epsilon_0: f64,
    pub epsilon_g: f64,
    pub epsilon_c: f64,
    pub epsilon_outer_iterations: usize,
    pub mm_iterations: usize,
    pub converged: bool,
    pub maps: Vec<AdaptiveSpatialMap>,
}

#[derive(Debug, Clone)]
struct LinearColumnConditioning {
    col_idx: usize,
    mean: f64,
    scale: f64,
}

#[derive(Debug, Clone, Default)]
struct LinearFitConditioning {
    intercept_idx: usize,
    columns: Vec<LinearColumnConditioning>,
}

#[derive(Clone)]
pub(crate) struct SpatialPsiDerivative {
    // These are derivatives with respect to psi = log(kappa), not log(length_scale).
    pub penalty_index: usize,
    pub penalty_indices: Vec<usize>,
    pub global_range: Range<usize>,
    pub total_p: usize,
    pub x_psi_local: Array2<f64>,
    pub s_psi_components_local: Vec<Array2<f64>>,
    pub x_psi_psi_local: Array2<f64>,
    pub s_psi_psi_components_local: Vec<Array2<f64>>,
    pub aniso_group_id: Option<usize>,
    /// Pre-computed cross-derivative design matrices for other axes
    /// in the same aniso group: Vec of (axis_offset_in_group, matrix).
    pub aniso_cross_designs: Option<Vec<(usize, Array2<f64>)>>,
    /// On-demand cross-penalty second derivatives ∂²S_m/∂ψ_a∂ψ_b for axes in
    /// the same anisotropy group. The input is the other axis offset in the
    /// group, and the output is one local penalty matrix per active penalty.
    pub aniso_cross_penalty_provider: Option<
        std::sync::Arc<
            dyn Fn(usize) -> Result<Vec<Array2<f64>>, EstimationError> + Send + Sync + 'static,
        >,
    >,
    /// Optional implicit design-derivative operator (shared across all axes
    /// in the same aniso group). When present, `x_psi_local` and
    /// `x_psi_psi_local` may be zero-sized, and design-derivative matvecs
    /// should go through this operator using `implicit_axis` as the axis index.
    pub implicit_operator: Option<std::sync::Arc<crate::terms::basis::ImplicitDesignPsiDerivative>>,
    /// Which axis in the implicit operator this entry corresponds to.
    pub implicit_axis: usize,
}

#[derive(Debug, Clone)]
pub(crate) struct SpatialLogKappaCoords {
    /// Flattened ψ values. For isotropic terms, one entry per term.
    /// For anisotropic terms, d entries per term (one ψ_a per axis).
    values: Array1<f64>,
    /// Dimensionality of each term: 1 for isotropic, d for anisotropic.
    dims_per_term: Vec<usize>,
}

/// Which end of the ψ bound the shared `aniso_bounds_from_data` helper is
/// computing. The lower end uses `-max_length_scale.ln()` as the pure-Duchon
/// fallback and the `.0` element of `spatial_term_psi_bounds`; the upper end
/// uses `-min_length_scale.ln()` and `.1`. Everything else is identical.
#[derive(Clone, Copy)]
enum AnisoBoundEnd {
    Lower,
    Upper,
}

impl SpatialLogKappaCoords {
    /// Construct from an explicit dims layout plus values.
    pub(crate) fn new_with_dims(values: Array1<f64>, dims_per_term: Vec<usize>) -> Self {
        assert_eq!(
            values.len(),
            dims_per_term.iter().sum::<usize>(),
            "SpatialLogKappaCoords: values length {} != sum of dims_per_term {}",
            values.len(),
            dims_per_term.iter().sum::<usize>(),
        );
        Self {
            values,
            dims_per_term,
        }
    }

    /// Isotropic initialization (backward-compatible path).
    pub(crate) fn from_length_scales(
        spec: &TermCollectionSpec,
        term_indices: &[usize],
        options: &SpatialLengthScaleOptimizationOptions,
    ) -> Self {
        let mut out = Array1::<f64>::zeros(term_indices.len());
        for (slot, &term_idx) in term_indices.iter().enumerate() {
            // Constant-curvature: the single ψ slot is the raw signed κ, seeded
            // from the spec (default κ = 0). The −ln(length_scale) convention is
            // log-κ semantics and must not touch the raw-κ coordinate; the κ
            // window projection happens later via `clamp_to_bounds`. Mirrors the
            // aniso constructor's κ branch.
            if let Some(cc) = constant_curvature_term_spec(spec, term_idx) {
                out[slot] = cc.kappa;
                continue;
            }
            let length_scale = get_spatial_length_scale(spec, term_idx)
                .unwrap_or(options.min_length_scale)
                .clamp(options.min_length_scale, options.max_length_scale);
            out[slot] = -length_scale.ln();
        }
        Self {
            values: out,
            dims_per_term: vec![1; term_indices.len()],
        }
    }

    /// Anisotropic-aware initialization.
    ///
    /// Initialization strategy (per math team recommendation): standardize the
    /// knot cloud axiswise, then run the existing isotropic κ initializer in
    /// the standardized space. This reuses the trusted isotropic initializer
    /// and gives initial η_a = −ln(σ_a) + mean(ln(σ_a)), which satisfies
    /// Ση_a = 0 by construction.
    ///
    /// For each term, checks whether it has `aniso_log_scales` set on its basis spec.
    /// - If isotropic (no aniso_log_scales, or 1-D): 1 entry = −ln(length_scale).
    /// - If anisotropic with a scalar length scale: d entries, one ψ_a per axis.
    ///   Initialized as ψ_a = −ln(length_scale) + η_a  where η_a are the existing
    ///   aniso_log_scales (which sum to zero). Multi-dimensional terms without
    ///   explicit anisotropy stay scalar here so the seed dimensionality matches
    ///   `spatial_dims_per_term`.
    /// - If pure Duchon anisotropic: d - 1 free entries store the leading η_a
    ///   values directly; the final axis is reconstructed to keep Ση_a = 0.
    pub(crate) fn from_length_scales_aniso(
        spec: &TermCollectionSpec,
        term_indices: &[usize],
        options: &SpatialLengthScaleOptimizationOptions,
    ) -> Self {
        let mut vals = Vec::new();
        let mut dims = Vec::new();
        for &term_idx in term_indices {
            // Measure-jet: dial coordinates seeded directly from the term's
            // realized (α, τ[, s]); the −ln(length_scale) convention below is
            // κ-semantics and never applies to dials.
            if let Some(mj) = measure_jet_term_spec(spec, term_idx) {
                let seed = measure_jet_psi_seed(mj);
                dims.push(seed.len());
                vals.extend(seed);
                continue;
            }
            // Constant-curvature: one signed κ slot seeded from the spec's κ
            // (clamped feasible). The −ln(length_scale) convention below is
            // log-κ semantics and must not touch the raw-κ coordinate. Bounds
            // are unavailable here (no data view), so this is the raw spec κ;
            // `reseed_from_data` / `clamp_to_bounds` later project it feasible.
            if let Some(cc) = constant_curvature_term_spec(spec, term_idx) {
                vals.push(cc.kappa);
                dims.push(1);
                continue;
            }
            let length_scale = get_spatial_length_scale(spec, term_idx)
                .unwrap_or(options.min_length_scale)
                .clamp(options.min_length_scale, options.max_length_scale);
            let psi_bar = -length_scale.ln(); // global scale = −ln(length_scale)

            if spatial_term_uses_per_axis_psi(spec, term_idx) {
                // Per-axis anisotropy is enrolled in the joint outer vector:
                // ψ_a = ψ̄ + η_a, one slot per axis. The hyper_dirs builder
                // produces matching per-axis derivatives in
                // `try_build_spatial_term_log_kappa_aniso_derivativeinfos`.
                let d = get_spatial_feature_dim(spec, term_idx).unwrap_or(1);
                let eta_raw = get_spatial_aniso_log_scales(spec, term_idx)
                    .expect("predicate guarantees aniso_log_scales is Some");
                let eta = center_aniso_log_scales(&eta_raw);
                for &eta_a in &eta {
                    vals.push(psi_bar + eta_a);
                }
                dims.push(d);
            } else {
                // Isotropic enrollment — either a 1-D term, a multi-D term
                // without explicit anisotropy, or a basis (e.g. Duchon) whose
                // η is a fixed geometry parameter rather than a REML hyper
                // axis. Exactly one ψ̄ slot, matching the single
                // `SpatialPsiDerivative` produced by
                // `try_build_spatial_term_log_kappa_derivativeinfo`.
                vals.push(psi_bar);
                dims.push(1);
            }
        }
        Self {
            values: Array1::from_vec(vals),
            dims_per_term: dims,
        }
    }

    /// Isotropic lower bounds derived from per-term data geometry.
    /// Each entry gets the ψ_lo bound returned by `spatial_term_psi_bounds`
    /// for the corresponding term, intersected with the options window.
    pub(crate) fn lower_bounds_from_data(
        data: ArrayView2<'_, f64>,
        spec: &TermCollectionSpec,
        term_indices: &[usize],
        options: &SpatialLengthScaleOptimizationOptions,
    ) -> Self {
        let mut values = Array1::<f64>::zeros(term_indices.len());
        for (slot, &term_idx) in term_indices.iter().enumerate() {
            values[slot] = spatial_term_psi_bounds(data, spec, term_idx, options).0;
        }
        Self {
            values,
            dims_per_term: vec![1; term_indices.len()],
        }
    }

    /// Isotropic upper bounds derived from per-term data geometry.
    pub(crate) fn upper_bounds_from_data(
        data: ArrayView2<'_, f64>,
        spec: &TermCollectionSpec,
        term_indices: &[usize],
        options: &SpatialLengthScaleOptimizationOptions,
    ) -> Self {
        let mut values = Array1::<f64>::zeros(term_indices.len());
        for (slot, &term_idx) in term_indices.iter().enumerate() {
            values[slot] = spatial_term_psi_bounds(data, spec, term_idx, options).1;
        }
        Self {
            values,
            dims_per_term: vec![1; term_indices.len()],
        }
    }

    /// Anisotropic-aware lower bounds derived from per-term data geometry.
    /// For hybrid anisotropic terms the scalar ψ_lo bound applies to the
    /// mean `ψ̄`, not directly to every raw axis coordinate `ψ_a = ψ̄ + η_a`.
    /// Shift each axis by the current centered `η_a` so projecting/clamping
    /// the seed moves only the global scale direction and does not silently
    /// shrink anisotropy that is already consistent with the current
    /// `length_scale`.
    ///
    /// Pure Duchon anisotropy is structurally different: its stored
    /// coordinates are (d-1) free η_a values representing log axis-scale
    /// ratios, NOT log-κ. For those terms the κ-range geometry bound is
    /// over-restrictive (η_a = ±5 is normal, but that corresponds to 7+
    /// orders of magnitude in κ-space and would be rejected by the data
    /// window). Fall back to the options window `[-ln(max_ls), -ln(min_ls)]`
    /// for those coordinates — that's the same bound the pre-data-geometry
    /// code used, which is calibrated to allow legitimate anisotropy.
    pub(crate) fn lower_bounds_aniso_from_data(
        data: ArrayView2<'_, f64>,
        spec: &TermCollectionSpec,
        term_indices: &[usize],
        dims_per_term: &[usize],
        options: &SpatialLengthScaleOptimizationOptions,
    ) -> Self {
        Self::aniso_bounds_from_data(
            data,
            spec,
            term_indices,
            dims_per_term,
            options,
            AnisoBoundEnd::Lower,
        )
    }

    /// Anisotropic-aware upper bounds derived from per-term data geometry.
    /// See `lower_bounds_aniso_from_data` for the hybrid-aniso offsetting and
    /// pure-Duchon dispatch rationale.
    pub(crate) fn upper_bounds_aniso_from_data(
        data: ArrayView2<'_, f64>,
        spec: &TermCollectionSpec,
        term_indices: &[usize],
        dims_per_term: &[usize],
        options: &SpatialLengthScaleOptimizationOptions,
    ) -> Self {
        Self::aniso_bounds_from_data(
            data,
            spec,
            term_indices,
            dims_per_term,
            options,
            AnisoBoundEnd::Upper,
        )
    }

    /// Shared implementation for the lower/upper aniso bounds. The bound end
    /// only changes which options scale (`max_length_scale` vs
    /// `min_length_scale`) becomes the pure-Duchon fallback bound and which
    /// element of the `(lo, hi)` data-geometry tuple is consumed; the
    /// per-term cursor walk and aniso-offset handling are identical.
    fn aniso_bounds_from_data(
        data: ArrayView2<'_, f64>,
        spec: &TermCollectionSpec,
        term_indices: &[usize],
        dims_per_term: &[usize],
        options: &SpatialLengthScaleOptimizationOptions,
        end: AnisoBoundEnd,
    ) -> Self {
        assert_eq!(term_indices.len(), dims_per_term.len());
        let total: usize = dims_per_term.iter().sum();
        let mut values = Array1::<f64>::zeros(total);
        let mut cursor = 0;
        for (slot, &term_idx) in term_indices.iter().enumerate() {
            let d = dims_per_term[slot];
            // Measure-jet: per-coordinate dial boxes, never κ-window geometry
            // (which would reject legitimate dial values outright).
            if let Some(mj) = measure_jet_term_spec(spec, term_idx) {
                let bounds = measure_jet_psi_bound_values(mj, matches!(end, AnisoBoundEnd::Upper));
                for (offset, bound) in bounds.into_iter().enumerate() {
                    if offset < d {
                        values[cursor + offset] = bound;
                    }
                }
                cursor += d;
                continue;
            }
            // Constant-curvature: the single signed-κ box from the data chart
            // window (symmetric about κ = 0), never a κ = log-scale window.
            if constant_curvature_term_spec(spec, term_idx).is_some() {
                let (lo, hi) = constant_curvature_kappa_bounds(data, spec, term_idx);
                if d >= 1 {
                    values[cursor] = match end {
                        AnisoBoundEnd::Lower => lo,
                        AnisoBoundEnd::Upper => hi,
                    };
                }
                cursor += d;
                continue;
            }
            let psi_bound = {
                let (lo, hi) = spatial_term_psi_bounds(data, spec, term_idx, options);
                match end {
                    AnisoBoundEnd::Lower => lo,
                    AnisoBoundEnd::Upper => hi,
                }
            };
            let axis_offsets = if d <= 1 {
                vec![0.0; d]
            } else {
                get_spatial_aniso_log_scales(spec, term_idx)
                    .filter(|eta| eta.len() == d)
                    .map(|eta| center_aniso_log_scales(&eta))
                    .unwrap_or_else(|| vec![0.0; d])
            };
            for offset in 0..d {
                values[cursor + offset] = psi_bound + axis_offsets[offset];
            }
            cursor += d;
        }
        Self {
            values,
            dims_per_term: dims_per_term.to_vec(),
        }
    }

    /// Rewrite any ψ entries whose originating term lacks an explicit
    /// `length_scale` so they sit at the midpoint of the per-term data-derived
    /// ψ window. Used so the outer optimizer starts inside the physically
    /// meaningful region instead of at an arbitrary `options.max_length_scale`
    /// derived seed. For terms with an explicit length_scale, the user's
    /// choice is respected. Anisotropy offsets η_a (those stored by
    /// `from_length_scales_aniso`) are preserved: we re-center around the new
    /// ψ̄, keeping Ση_a = 0.
    pub(crate) fn reseed_from_data(
        mut self,
        data: ArrayView2<'_, f64>,
        spec: &TermCollectionSpec,
        term_indices: &[usize],
        options: &SpatialLengthScaleOptimizationOptions,
    ) -> Self {
        assert_eq!(term_indices.len(), self.dims_per_term.len());
        let mut cursor = 0;
        for (slot, &term_idx) in term_indices.iter().enumerate() {
            let d = self.dims_per_term[slot];
            // Measure-jet dials are seeded from the realized spec and must
            // not be recentered into a κ data window.
            if measure_jet_term_spec(spec, term_idx).is_some() {
                cursor += d;
                continue;
            }
            // Constant-curvature κ is seeded from the spec (the user's curvature
            // hint, default κ = 0); `clamp_to_bounds` projects it feasible. It
            // is not a log-scale, so the log-κ recenter below never applies.
            if constant_curvature_term_spec(spec, term_idx).is_some() {
                cursor += d;
                continue;
            }
            let Some(psi_bar_new) = spatial_term_psi_seed(data, spec, term_idx, options) else {
                cursor += d;
                continue;
            };
            if d == 0 {
                continue;
            }
            let current: Vec<f64> = self.values.slice(s![cursor..cursor + d]).to_vec();
            let psi_bar_old = current.iter().sum::<f64>() / d as f64;
            for (offset, &old_value) in current.iter().enumerate() {
                self.values[cursor + offset] = psi_bar_new + (old_value - psi_bar_old);
            }
            cursor += d;
        }
        self
    }

    /// Project ψ values into `[lower, upper]` element-wise. Used after
    /// `from_length_scales*` + `reseed_from_data` when a user-supplied
    /// `spec.length_scale` falls outside the data-derived ψ window set by
    /// `{lower,upper}_bounds*_from_data`. BFGS requires theta0 ∈ [lower,
    /// upper]; projecting is the unique closest feasible seed. The user's
    /// length_scale was always a hint for the outer optimizer (the optimizer
    /// is authoritative for κ), not a hard constraint — so clipping preserves
    /// their intent as far as the geometry allows. Emits `log::info!` when
    /// any coordinate moves, so the outside-window case is diagnostically
    /// visible (not silent).
    pub(crate) fn clamp_to_bounds(
        mut self,
        lower: &SpatialLogKappaCoords,
        upper: &SpatialLogKappaCoords,
    ) -> Self {
        assert_eq!(self.values.len(), lower.values.len());
        assert_eq!(self.values.len(), upper.values.len());
        let mut n_projected = 0usize;
        let mut worst_delta = 0.0_f64;
        for idx in 0..self.values.len() {
            let lo = lower.values[idx];
            let hi = upper.values[idx];
            if !(lo.is_finite() && hi.is_finite()) {
                continue;
            }
            let v = self.values[idx];
            if v < lo {
                worst_delta = worst_delta.max(lo - v);
                self.values[idx] = lo;
                n_projected += 1;
            } else if v > hi {
                worst_delta = worst_delta.max(v - hi);
                self.values[idx] = hi;
                n_projected += 1;
            }
        }
        if n_projected > 0 {
            log::info!(
                "[spatial-kappa] projected {n_projected}/{} ψ seed coords into data-derived bounds \
                 (worst excess={worst_delta:.3} log units); user length_scale falls outside \
                 [{KERNEL_RANGE_MIN_DIAMETER_FRACTION}/r_max, {KERNEL_RANGE_MAX_SPACING_MULTIPLE}/r_min] geometry window",
                self.values.len()
            );
        }
        self
    }

    /// Reconstruct from theta tail with known dimensionality layout.
    pub(crate) fn from_theta_tail_with_dims(
        theta: &Array1<f64>,
        start: usize,
        dims_per_term: Vec<usize>,
    ) -> Self {
        let total: usize = dims_per_term.iter().sum();
        Self {
            values: theta.slice(s![start..start + total]).to_owned(),
            dims_per_term,
        }
    }

    /// Total number of ψ values in the flat array (= sum of dims_per_term).
    pub(crate) fn len(&self) -> usize {
        self.values.len()
    }

    /// Dimensionality layout: how many ψ values each term contributes.
    pub(crate) fn dims_per_term(&self) -> &[usize] {
        &self.dims_per_term
    }

    /// Get the offset into the flat array for logical term i.
    fn term_offset(&self, term_idx: usize) -> usize {
        self.dims_per_term[..term_idx].iter().sum()
    }

    /// Get the slice of ψ values for logical term i.
    fn term_slice(&self, term_idx: usize) -> &[f64] {
        let offset = self.term_offset(term_idx);
        let d = self.dims_per_term[term_idx];
        &self.values.as_slice().unwrap()[offset..offset + d]
    }

    pub(crate) fn as_array(&self) -> &Array1<f64> {
        &self.values
    }

    /// Split at a logical-term boundary. `mid` is the number of terms in the
    /// first half (not a flat-array index).
    pub(crate) fn split_at(&self, mid: usize) -> (Self, Self) {
        let flat_mid: usize = self.dims_per_term[..mid].iter().sum();
        (
            Self {
                values: self.values.slice(s![0..flat_mid]).to_owned(),
                dims_per_term: self.dims_per_term[..mid].to_vec(),
            },
            Self {
                values: self.values.slice(s![flat_mid..]).to_owned(),
                dims_per_term: self.dims_per_term[mid..].to_vec(),
            },
        )
    }

    /// Apply optimized ψ values back to the spec.
    ///
    /// For isotropic terms (dims=1): sets scalar length_scale = exp(−ψ).
    /// For anisotropic terms (dims=d): hybrid/isotropic families set
    /// length_scale = exp(−ψ̄) with centered η_a = ψ_a − ψ̄, while pure Duchon
    /// writes only centered η_a and leaves length_scale = None.
    pub(crate) fn apply_tospec(
        &self,
        spec: &TermCollectionSpec,
        term_indices: &[usize],
    ) -> Result<TermCollectionSpec, EstimationError> {
        if term_indices.len() != self.dims_per_term.len() {
            crate::bail_invalid_estim!(
                "SpatialLogKappaCoords::apply_tospec: term count mismatch: \
                 term_indices={} dims_per_term={}",
                term_indices.len(),
                self.dims_per_term.len()
            );
        }
        let mut updated = spec.clone();
        for (slot, &term_idx) in term_indices.iter().enumerate() {
            let psi = self.term_slice(slot);
            let d = self.dims_per_term[slot];
            // Measure-jet: write the dial coordinates straight back; the
            // κ-translation below would misread them as log-scales.
            if measure_jet_term_spec(&updated, term_idx).is_some() {
                set_measure_jet_psi_dials(&mut updated, term_idx, psi)?;
                continue;
            }
            // Constant-curvature: write the optimized signed κ straight back;
            // the −exp(ψ) length-scale translation below is log-κ semantics and
            // would misread the raw curvature.
            if constant_curvature_term_spec(&updated, term_idx).is_some() {
                set_constant_curvature_kappa(&mut updated, term_idx, psi)?;
                continue;
            }
            let (next_length_scale, next_aniso) = spatial_term_psi_to_length_scale_and_aniso(psi);
            if (d == 1 || next_length_scale.is_some())
                && let Some(length_scale) = next_length_scale
            {
                set_spatial_length_scale(&mut updated, term_idx, length_scale)?;
            }
            if let Some(eta) = next_aniso {
                set_spatial_aniso_log_scales(&mut updated, term_idx, eta)?;
            }
        }
        Ok(updated)
    }
}

pub(crate) fn center_aniso_log_scales(eta: &[f64]) -> Vec<f64> {
    if eta.len() <= 1 {
        return eta.to_vec();
    }
    let mean = eta.iter().sum::<f64>() / eta.len() as f64;
    eta.iter()
        .map(|&v| {
            let centered = v - mean;
            if centered.abs() <= 1e-15 {
                0.0
            } else {
                centered
            }
        })
        .collect()
}

fn spatial_term_supports_hyper_optimization(spec: &TermCollectionSpec, term_idx: usize) -> bool {
    // Ordinary penalized thin-plate regression splines do not have an
    // identifiable kernel scale once REML is already learning the smoothing
    // penalty. Treat the resolved length scale as fixed geometry; enrolling a
    // scalar TPS kappa axis creates the flat ρ/κ valleys reported in #718,
    // #721, #731, and #732.
    if let Some(term) = spec.smooth_terms.get(term_idx)
        && let SmoothBasisSpec::ThinPlate { .. } = &term.basis
    {
        return false;
    }

    // Duchon anisotropy η is a FIXED, geometry-derived basis parameter, NOT a
    // REML hyper axis: the metric is estimated once from the knot-cloud spread
    // (`auto_seed_aniso_contrasts`, applied on every Duchon basis build) and the
    // Hilbert-scale λ's carry all learned smoothness. So a pure Duchon (no κ)
    // contributes no outer optimization axis even when `scale_dims` is on —
    // "standardize the geometry, then learn the smoothness." Only an explicit
    // kernel length scale κ (the Matérn / hybrid path) is optimized here.
    //
    // ISOTROPIC Matérn: the *default* `matern(x1, x2)` is isotropic
    // (`scale_dims=false` → `aniso_log_scales = None`). It contributes exactly
    // ONE κ optimization axis — its scalar log-κ. The shared GAMLSS /
    // location-scale exact-joint ψ engine and the spatial-κ joint outer solver
    // both require an isotropic Matérn block to expose this single isotropic κ
    // axis (#822/#851); without it the per-block ψ-derivative lists are empty
    // and the joint-ψ hooks degenerate to `None`. The isotropic κ is the lone
    // kernel hyper axis here, mirroring the per-axis ψ ARD that the anisotropic
    // path exposes (just collapsed to one dimension).
    //
    // ANISOTROPIC Matérn (`scale_dims=true` → `aniso_log_scales = Some`) keeps
    // its per-axis kernel-η ARD: the d-dimensional ψ search is the *point* of
    // the anisotropic request ("Matérn keeps its kernel-η ARD").
    //
    // Either way a Matérn term always enrolls a κ/ψ axis (1 isotropic, or d
    // anisotropic), so `spatial_dims_per_term` reports the correct count.
    if let Some(term) = spec.smooth_terms.get(term_idx)
        && let SmoothBasisSpec::Matern { .. } = &term.basis
    {
        return true;
    }

    // Measure-jet geometry dials are outer ψ coordinates; enrollment is
    // owned by `measure_jet_enrolls_psi`.
    if let Some(mj) = measure_jet_term_spec(spec, term_idx) {
        return measure_jet_enrolls_psi(mj);
    }

    // Constant-curvature smooths always enroll their single signed curvature κ
    // as an outer ψ-coordinate (#944 stage 3): κ̂ is the headline estimand, so
    // unlike a fixed-ℓ kernel it is fitted by default, not gated on a
    // user-supplied scale. The coordinate is raw κ (interior κ = 0), and its
    // exact design/penalty κ-derivatives come from
    // `build_constant_curvature_basis_kappa_derivatives`.
    if constant_curvature_term_spec(spec, term_idx).is_some() {
        return true;
    }

    get_spatial_length_scale(spec, term_idx).is_some()
}

/// The measure-jet term's spec, when `term_idx` is a measure-jet smooth.
/// Single accessor for every dial-plumbing dispatch below.
fn measure_jet_term_spec(
    spec: &TermCollectionSpec,
    term_idx: usize,
) -> Option<&crate::basis::MeasureJetBasisSpec> {
    spec.smooth_terms
        .get(term_idx)
        .and_then(|term| match &term.basis {
            SmoothBasisSpec::MeasureJet { spec, .. } => Some(spec),
            _ => None,
        })
}

/// Single source for measure-jet outer-ψ enrollment: the lnτ dial is
/// undefined in the τ = 0 pseudo-inverse oracle mode (see
/// `build_measure_jet_basis_psi_derivatives`), so only a positive ridge
/// enrolls the dial group. `spatial_term_supports_hyper_optimization` and
/// `spatial_term_uses_per_axis_psi` both defer here so the θ-layout
/// sources cannot disagree.
fn measure_jet_enrolls_psi(mj: &crate::basis::MeasureJetBasisSpec) -> bool {
    // Two independent enrollment sources (#1116):
    //   * the design-moving representer length-scale ℓ (`learn_length_scale`),
    //     available in EVERY mode (it is the cure for the over-smoothing on
    //     low-intrinsic-dimension manifolds — matérn's log_kappa analog);
    //   * the multiscale penalty dials (s, α, lnτ): the per-scale spectral
    //     split's (α, lnτ) ride the explicit `multiscale` opt-in, and the lnτ
    //     channel additionally needs a positive ridge (τ = 0 is the
    //     pseudo-inverse oracle mode where lnτ is undefined).
    // A term enrolls if EITHER source is active.
    measure_jet_learns_length_scale(mj)
        || (mj.tau0 > 0.0 && crate::basis::measure_jet_multiscale_mode(mj))
}

/// Whether the design-moving ℓ dial is enrolled for this term. ℓ is learnable
/// in every mode; an explicitly-pinned positive `length_scale` with
/// `learn_length_scale = false` freezes it (matches a user-fixed kernel scale).
fn measure_jet_learns_length_scale(mj: &crate::basis::MeasureJetBasisSpec) -> bool {
    mj.learn_length_scale
}

/// Measure-jet ψ dial boxes. The dials are NOT log-kernel-scales, so the
/// κ-window machinery never applies: `α` spans density-weighted (0) through
/// past-Coifman–Lafon (>1) normalization, and `lnτ` covers the ridge from
/// numerically-exact-projection to heavy noise-floor damping. (The energy
/// order `s` is the pinned explicit value or absorbed by the REML-learned
/// per-scale amplitudes — see `measure_jet_penalty_psi_dim` — so it carries no
/// dial box.)
const MEASURE_JET_PSI_ALPHA_BOUNDS: (f64, f64) = (-1.0, 3.0);

const MEASURE_JET_PSI_LN_TAU_BOUNDS: (f64, f64) = (-18.420680743952367, 4.605170185988092);

/// Log-ℓ box for the design-moving representer length-scale dial (#1116). An
/// ABSOLUTE window in the data coordinate scale (ln of ℓ ∈ [1e-3, 1e2]): wide
/// enough to let REML shrink ℓ ~orders below the auto ambient-spacing seed
/// (the fix for the 1-D-curve-in-3-D over-smoothing) or grow it for very smooth
/// surfaces, while keeping the Gaussian kernel numerically well-scaled. Absolute
/// (not seed-relative) so the bound producer needs no data view, matching the
/// other dial boxes. `ln(1e-3) = -6.9077…`, `ln(1e2) = 4.6051…`.
const MEASURE_JET_PSI_LN_LENGTH_SCALE_BOUNDS: (f64, f64) = (-6.907755278982137, 4.605170185988092);

/// Number of multiscale PENALTY dials (excluding the design-moving ℓ):
/// multiscale (per-scale spectral) mode carries (α, lnτ) = 2 — the order is
/// either the pinned explicit `s` or absorbed by the REML-learned per-scale
/// amplitudes, so it is NOT a dial; single-scale (the default) carries none.
/// MUST agree with the penalty-coordinate layout of
/// `build_measure_jet_basis_psi_derivatives` (its `per_level` branch always
/// emits exactly the (α, lnτ) coordinate pair).
fn measure_jet_penalty_psi_dim(mj: &crate::basis::MeasureJetBasisSpec) -> usize {
    if crate::basis::measure_jet_multiscale_mode(mj) {
        2
    } else {
        0
    }
}

/// ψ dimension of a measure-jet term. The design-moving ℓ dial (when enrolled)
/// is coordinate 0; the multiscale penalty dials follow. MUST agree with the
/// coordinate layout of `build_measure_jet_basis_psi_derivatives` (ℓ first).
fn measure_jet_psi_dim(mj: &crate::basis::MeasureJetBasisSpec) -> usize {
    usize::from(measure_jet_learns_length_scale(mj)) + measure_jet_penalty_psi_dim(mj)
}

/// Seed ψ from the term's realized dials, in producer coordinate order: ℓ first
/// (when enrolled), then the multiscale penalty dials. The ℓ seed is the
/// realized representer range `ln(length_scale)` (the resolved spec carries the
/// concrete auto value after the design build/freeze).
fn measure_jet_psi_seed(mj: &crate::basis::MeasureJetBasisSpec) -> Vec<f64> {
    let mut seed = Vec::with_capacity(measure_jet_psi_dim(mj));
    if measure_jet_learns_length_scale(mj) {
        // length_scale > 0 after resolution; the 0.0 sentinel (pre-resolution)
        // falls back to the centre of the log-ℓ box so the optimizer still
        // starts feasible and the first data-aware reseed corrects it.
        let ell = if mj.length_scale > 0.0 {
            mj.length_scale
        } else {
            1.0
        };
        seed.push(ell.ln());
    }
    if measure_jet_penalty_psi_dim(mj) > 0 {
        // Multiscale penalty dials, producer order: (α, lnτ).
        let ln_tau = mj.tau0.max(f64::MIN_POSITIVE).ln();
        seed.extend_from_slice(&[mj.alpha, ln_tau]);
    }
    seed
}

/// One end of the per-coordinate dial boxes, in producer coordinate order
/// (ℓ first when enrolled, then the multiscale penalty dials).
fn measure_jet_psi_bound_values(mj: &crate::basis::MeasureJetBasisSpec, upper: bool) -> Vec<f64> {
    let pick = |b: (f64, f64)| if upper { b.1 } else { b.0 };
    let mut bounds = Vec::with_capacity(measure_jet_psi_dim(mj));
    if measure_jet_learns_length_scale(mj) {
        bounds.push(pick(MEASURE_JET_PSI_LN_LENGTH_SCALE_BOUNDS));
    }
    if measure_jet_penalty_psi_dim(mj) > 0 {
        // Multiscale penalty dials, producer order: (α, lnτ).
        bounds.push(pick(MEASURE_JET_PSI_ALPHA_BOUNDS));
        bounds.push(pick(MEASURE_JET_PSI_LN_TAU_BOUNDS));
    }
    bounds
}

/// Write optimized ψ dials back into a measure-jet spec. Returns `true` when
/// any dial actually moved. The geometry (centers, masses, band, ℓ, z) is
/// ψ-FIXED by contract — only the dials change, so frozen-quadrature
/// rebuilds reproduce the identical penalty layout at the new dials.
fn apply_measure_jet_psi(
    mj: &mut crate::basis::MeasureJetBasisSpec,
    psi: &[f64],
) -> Result<bool, EstimationError> {
    if psi.len() != measure_jet_psi_dim(mj) {
        crate::bail_invalid_estim!(
            "measure-jet ψ write-back dimension mismatch: got {} values for a {}-dial term",
            psi.len(),
            measure_jet_psi_dim(mj)
        );
    }
    let mut changed = false;
    // Coordinate 0 (when enrolled) is the design-moving ln(ℓ); the multiscale
    // penalty dials follow. Same order as `measure_jet_psi_seed` and the
    // producer (`build_measure_jet_basis_psi_derivatives`).
    let mut cursor = 0usize;
    if measure_jet_learns_length_scale(mj) {
        let next_ell = psi[cursor].exp();
        cursor += 1;
        if !(next_ell.is_finite() && next_ell > 0.0) {
            crate::bail_invalid_estim!(
                "measure-jet ψ write-back produced a non-finite/non-positive length_scale (ℓ={next_ell})"
            );
        }
        if next_ell != mj.length_scale {
            mj.length_scale = next_ell;
            changed = true;
        }
    }
    if measure_jet_penalty_psi_dim(mj) > 0 {
        // Multiscale penalty dials, producer order: (α, lnτ). The order `s` is
        // not a dial (pinned explicit or absorbed by the per-scale amplitudes).
        let next_alpha = psi[cursor];
        let next_tau = psi[cursor + 1].exp();
        if !(next_alpha.is_finite() && next_tau.is_finite() && next_tau > 0.0) {
            crate::bail_invalid_estim!(
                "measure-jet ψ write-back produced non-finite dials (alpha={next_alpha}, tau={next_tau})"
            );
        }
        if next_alpha != mj.alpha {
            mj.alpha = next_alpha;
            changed = true;
        }
        if next_tau != mj.tau0 {
            mj.tau0 = next_tau;
            changed = true;
        }
    }
    Ok(changed)
}

/// Collection-level measure-jet dial write-back (the `apply_tospec` /
/// realizer-side entry). Returns whether anything moved.
fn set_measure_jet_psi_dials(
    spec: &mut TermCollectionSpec,
    term_idx: usize,
    psi: &[f64],
) -> Result<bool, EstimationError> {
    let Some(term) = spec.smooth_terms.get_mut(term_idx) else {
        crate::bail_invalid_estim!("measure-jet ψ write-back: term index {term_idx} out of range");
    };
    set_single_term_measure_jet_psi_dials(term, psi)
}

/// Single-term dial write-back: the shared match+apply core, also used
/// directly on the cached per-trial build spec (whose caller has already
/// change-checked at the collection level and rebuilds regardless of the
/// moved flag).
fn set_single_term_measure_jet_psi_dials(
    term: &mut SmoothTermSpec,
    psi: &[f64],
) -> Result<bool, EstimationError> {
    let SmoothBasisSpec::MeasureJet { spec: mj, .. } = &mut term.basis else {
        crate::bail_invalid_estim!("measure-jet ψ write-back targeted a non-measure-jet term");
    };
    apply_measure_jet_psi(mj, psi)
}

/// The constant-curvature smooth's spec, when `term_idx` is one. Single
/// accessor for every κ-ψ dispatch below, mirroring `measure_jet_term_spec`.
fn constant_curvature_term_spec(
    spec: &TermCollectionSpec,
    term_idx: usize,
) -> Option<&crate::basis::ConstantCurvatureBasisSpec> {
    spec.smooth_terms
        .get(term_idx)
        .and_then(|term| match &term.basis {
            SmoothBasisSpec::ConstantCurvature { spec, .. } => Some(spec),
            _ => None,
        })
}

/// Hard positive cap on |κ| relative to the data's inverse squared chart
/// radius. The κ-stereographic chart is valid for `1 + κ‖x‖² > 0`; at
/// `|κ| = 1/R²` (R² = max squared chart radius) the gauge `1 + κ‖x‖²` reaches
/// the chart edge for the farthest data point, so the optimizer is boxed to a
/// safe fraction of that scale on both sides. κ = 0 (flat) is the centre of
/// the window, an interior point of the `S^d ← ℝ^d → H^d` family — exactly the
/// reachability the raw-κ (not log-κ) coordinate exists to preserve.
const CONSTANT_CURVATURE_KAPPA_CHART_FRACTION: f64 = 0.5;

/// Floor on the data's squared chart radius used to scale the κ window, so a
/// degenerate (near-origin) point cloud still yields a finite, usable bracket
/// rather than an unbounded one.
const CONSTANT_CURVATURE_MIN_CHART_RADIUS2: f64 = 1e-8;

/// `(κ_min, κ_max)` outer-optimization window for a constant-curvature term,
/// derived from the data's maximum squared chart radius `R²` so the κ-jets
/// never leave the κ-stereographic chart. Symmetric about κ = 0:
/// `±CONSTANT_CURVATURE_KAPPA_CHART_FRACTION / R²`.
fn constant_curvature_kappa_bounds(
    data: ArrayView2<'_, f64>,
    spec: &TermCollectionSpec,
    term_idx: usize,
) -> (f64, f64) {
    let feature_cols = match spec.smooth_terms.get(term_idx).map(|t| &t.basis) {
        Some(SmoothBasisSpec::ConstantCurvature { feature_cols, .. }) => feature_cols,
        _ => return (-1.0, 1.0),
    };
    let mut max_r2 = CONSTANT_CURVATURE_MIN_CHART_RADIUS2;
    for row in data.outer_iter() {
        let mut r2 = 0.0_f64;
        for &c in feature_cols.iter() {
            if let Some(&v) = row.get(c)
                && v.is_finite()
            {
                r2 += v * v;
            }
        }
        if r2 > max_r2 {
            max_r2 = r2;
        }
    }
    let half = CONSTANT_CURVATURE_KAPPA_CHART_FRACTION / max_r2;
    (-half, half)
}

/// Write the optimized κ back into a constant-curvature term spec. Returns
/// `true` when κ moved. Centers, ℓ, and the constraint transform `z` are
/// κ-FIXED by the basis κ-contract, so only `kappa` changes.
fn set_constant_curvature_kappa(
    spec: &mut TermCollectionSpec,
    term_idx: usize,
    psi: &[f64],
) -> Result<bool, EstimationError> {
    let Some(term) = spec.smooth_terms.get_mut(term_idx) else {
        crate::bail_invalid_estim!(
            "constant-curvature κ write-back: term index {term_idx} out of range"
        );
    };
    set_single_term_constant_curvature_kappa(term, psi)
}

/// Single-term κ write-back: the shared validate+apply core, also used directly
/// on the cached per-trial build spec in the incremental realizer (whose caller
/// has already change-checked at the collection level and rebuilds regardless
/// of the moved flag). Mirrors [`set_single_term_measure_jet_psi_dials`].
fn set_single_term_constant_curvature_kappa(
    term: &mut SmoothTermSpec,
    psi: &[f64],
) -> Result<bool, EstimationError> {
    if psi.len() != 1 {
        crate::bail_invalid_estim!(
            "constant-curvature κ write-back expects exactly one value, got {}",
            psi.len()
        );
    }
    let next_kappa = psi[0];
    if !next_kappa.is_finite() {
        crate::bail_invalid_estim!(
            "constant-curvature κ write-back produced a non-finite κ = {next_kappa}"
        );
    }
    let SmoothBasisSpec::ConstantCurvature { spec: cc, .. } = &mut term.basis else {
        crate::bail_invalid_estim!(
            "constant-curvature κ write-back targeted a non-constant-curvature term"
        );
    };
    if cc.kappa != next_kappa {
        cc.kappa = next_kappa;
        Ok(true)
    } else {
        Ok(false)
    }
}

/// Returns `true` when a spatial term has NO outer optimization axes — i.e.
/// the user provided an explicit `length_scale` and the term does not enroll
/// REML-side per-axis ψ contrasts, so both the scalar κ and any fixed geometry
/// anisotropy are anchored.
///
/// This is the per-term predicate that distinguishes "fixed kernel scale"
/// from "optimize the kernel scale" within the family entry points that
/// want to honor an explicit user-supplied scale (e.g. Bernoulli
/// marginal-slope, where the joint-spatial outer solver otherwise spends
/// ~80 iters stalled on the user's chosen ρ at high gradient).
pub fn spatial_term_has_locked_kappa(spec: &TermCollectionSpec, term_idx: usize) -> bool {
    get_spatial_length_scale(spec, term_idx).is_some()
        && !spatial_term_uses_per_axis_psi(spec, term_idx)
}

/// Per-term data-derived ψ = log κ bounds.
///
/// Uses the same safe operating range documented in
/// [`crate::basis::build_matern_basis`] / [`crate::basis::build_duchon_basis`]:
///   κ ∈ [2 / r_max, 1e2 / r_min]
/// where (r_min, r_max) are pairwise-distance extrema of the term's resolved
/// centers (post-fit) or the standardized feature data columns (pre-fit).
/// Lower edge of the data-derived kernel-range window, as a fraction of the
/// maximum pairwise distance `r_max`: length scales below `2/r_max` resolve
/// structure finer than the closest center pair, so the kernel range floor is
/// set at twice the maximum spacing.
const KERNEL_RANGE_MIN_DIAMETER_FRACTION: f64 = 2.0;

/// Upper edge of the data-derived kernel-range window, as a multiple of the
/// minimum pairwise distance `r_min`: beyond `100/r_min` the radial columns go
/// nearly collinear with the polynomial nullspace, so the kernel range is
/// capped here to keep the basis geometry well-conditioned.
const KERNEL_RANGE_MAX_SPACING_MULTIPLE: f64 = 1e2;

/// Returns ψ-space bounds (ψ_lo = ln(κ_lo), ψ_hi = ln(κ_hi)).
///
/// When geometry is unavailable (e.g., fewer than 2 distinct points), falls
/// back to the scalar `options.min_length_scale` / `options.max_length_scale`
/// window so the outer optimizer never sees NaN bounds.
///
/// The returned window is intersected with the options window so user-set
/// `min_length_scale` / `max_length_scale` remain hard limits.
fn spatial_term_psi_bounds(
    data: ArrayView2<'_, f64>,
    spec: &TermCollectionSpec,
    term_idx: usize,
    options: &SpatialLengthScaleOptimizationOptions,
) -> (f64, f64) {
    let fallback = (
        -options.max_length_scale.ln(),
        -options.min_length_scale.ln(),
    );
    // Constant-curvature: the ψ coordinate is the raw signed κ, so its window is
    // the chart-feasible κ bracket, NOT a log-ℓ window. Mirrors the aniso bounds
    // path's `constant_curvature_kappa_bounds` branch so the isotropic
    // (non-aniso) seed clamp projects κ into the right interval.
    if constant_curvature_term_spec(spec, term_idx).is_some() {
        return constant_curvature_kappa_bounds(data, spec, term_idx);
    }
    let Some(term) = spec.smooth_terms.get(term_idx) else {
        return fallback;
    };
    // Prefer resolved centers (post-fit) since they live in the same standardized
    // space the kernel actually sees. Centers are capped at `default_num_centers`
    // (<=2000), so exact pairwise bounds are cheap (<4M ops). If centers are
    // not yet UserProvided, fall back to the standardized feature data columns
    // with the capped-sample path (O(K²·d), K=1024) — the sample is
    // conservative for κ bounds (see `pairwise_distance_bounds_sampled`
    // docs): it never excludes a feasible κ the exact method would include.
    //
    // Under anisotropy the kernel metric is y-space (y_a = exp(η_a) x_a),
    // so r_min/r_max must be y-space distances. This matters only when the
    // spec already carries calibrated η_a at setup time (e.g., warm-start
    // or refit paths); for fresh optimization η_a starts at 0 and y = x.
    let aniso = get_spatial_aniso_log_scales(spec, term_idx);
    let r_bounds = match spatial_term_center_strategy(term) {
        Some(CenterStrategy::UserProvided(centers)) if centers.nrows() >= 2 => {
            match aniso.as_deref() {
                Some(eta) if eta.len() == centers.ncols() => {
                    let y = points_in_aniso_y_space(centers.view(), eta);
                    pairwise_distance_bounds(y.view())
                }
                _ => pairwise_distance_bounds(centers.view()),
            }
        }
        _ => standardized_spatial_term_data(data, term)
            .ok()
            .and_then(|x| match aniso.as_deref() {
                Some(eta) if eta.len() == x.ncols() => {
                    let y = points_in_aniso_y_space(x.view(), eta);
                    pairwise_distance_bounds_sampled(y.view())
                }
                _ => pairwise_distance_bounds_sampled(x.view()),
            }),
    };
    let Some((r_min, r_max)) = r_bounds else {
        return fallback;
    };
    // Length scales substantially larger than the data diameter make radial
    // TPS/Matern columns nearly collinear with their polynomial nullspace.
    // The nullspace already carries constant/linear low-frequency structure,
    // so cap the kernel range at the diameter scale instead of letting the
    // optimizer enter a numerically degenerate basis geometry.
    let psi_lo_data = (KERNEL_RANGE_MIN_DIAMETER_FRACTION / r_max).ln();
    let psi_hi_data = (KERNEL_RANGE_MAX_SPACING_MULTIPLE / r_min).ln();
    // Intersect with the options window so min/max_length_scale remain hard caps.
    let psi_lo = psi_lo_data.max(fallback.0);
    let psi_hi = psi_hi_data.min(fallback.1);
    if psi_lo >= psi_hi {
        // Degenerate intersection — fall back to the options window to keep the
        // outer optimizer from collapsing to a point.
        return fallback;
    }
    (psi_lo, psi_hi)
}

/// Data-derived ψ seed for a spatial term when the user has not set an
/// explicit length_scale on its basis spec. Uses the geometric mean of the
/// data-informed kappa range (i.e., the midpoint of the ψ window).
fn spatial_term_psi_seed(
    data: ArrayView2<'_, f64>,
    spec: &TermCollectionSpec,
    term_idx: usize,
    options: &SpatialLengthScaleOptimizationOptions,
) -> Option<f64> {
    if get_spatial_length_scale(spec, term_idx).is_some() {
        return None; // user/spec-provided length_scale wins
    }
    let (psi_lo, psi_hi) = spatial_term_psi_bounds(data, spec, term_idx, options);
    Some(0.5 * (psi_lo + psi_hi))
}

fn spatial_term_psi_to_length_scale_and_aniso(psi: &[f64]) -> (Option<f64>, Option<Vec<f64>>) {
    if psi.len() <= 1 {
        (Some((-psi.first().copied().unwrap_or(0.0)).exp()), None)
    } else {
        let psi_bar = psi.iter().sum::<f64>() / psi.len() as f64;
        (
            Some((-psi_bar).exp()),
            Some(psi.iter().map(|&value| value - psi_bar).collect()),
        )
    }
}

/// Get the `aniso_log_scales` from a spatial term, if present.
pub fn get_spatial_aniso_log_scales(
    spec: &TermCollectionSpec,
    term_idx: usize,
) -> Option<Vec<f64>> {
    spec.smooth_terms
        .get(term_idx)
        .and_then(|term| match &term.basis {
            SmoothBasisSpec::Matern { spec, .. } => spec.aniso_log_scales.clone(),
            SmoothBasisSpec::Duchon { spec, .. } => spec.aniso_log_scales.clone(),
            _ => None,
        })
}

/// Get the number of feature columns (spatial dimensionality) for a spatial term.
fn get_spatial_feature_dim(spec: &TermCollectionSpec, term_idx: usize) -> Option<usize> {
    spec.smooth_terms
        .get(term_idx)
        .and_then(|term| match &term.basis {
            SmoothBasisSpec::ThinPlate { feature_cols, .. } => Some(feature_cols.len()),
            SmoothBasisSpec::Matern { feature_cols, .. } => Some(feature_cols.len()),
            SmoothBasisSpec::Duchon { feature_cols, .. } => Some(feature_cols.len()),
            _ => None,
        })
}

/// Log the learned per-axis spatial anisotropy for all spatial terms that
/// have `aniso_log_scales` set after optimization.
///
/// For scalar-scale families this reports eta, effective per-axis length
/// scales, and per-axis kappa values. For pure Duchon it reports the centered
/// eta contrasts only.
pub fn log_spatial_aniso_scales(spec: &TermCollectionSpec) {
    for (term_idx, term) in spec.smooth_terms.iter().enumerate() {
        let (aniso, length_scale) = match &term.basis {
            SmoothBasisSpec::Matern { spec, .. } => {
                (spec.aniso_log_scales.as_ref(), Some(spec.length_scale))
            }
            SmoothBasisSpec::Duchon { spec, .. } => {
                (spec.aniso_log_scales.as_ref(), spec.length_scale)
            }
            _ => (None, None),
        };
        let Some(eta) = aniso else { continue };
        if eta.is_empty() {
            continue;
        }
        let mut lines = match length_scale {
            Some(ls) => format!(
                "[spatial-kappa] term {} (\"{}\"): anisotropic length scales optimized (global length_scale={:.4})",
                term_idx, term.name, ls
            ),
            None => format!(
                "[spatial-kappa] term {} (\"{}\"): pure Duchon shape anisotropy optimized",
                term_idx, term.name
            ),
        };
        for (a, &eta_a) in eta.iter().enumerate() {
            if let Some(ls) = length_scale {
                let length_a = ls * (-eta_a).exp();
                let kappa_a = (1.0 / ls) * eta_a.exp();
                lines.push_str(&format!(
                    "\n  axis {}: eta={:+.4}, length={:.4}, kappa={:.4}",
                    a, eta_a, length_a, kappa_a
                ));
            } else {
                lines.push_str(&format!("\n  axis {}: eta={:+.4}", a, eta_a));
            }
        }
        log::info!("{}", lines);
    }
}

/// Set `aniso_log_scales` on a spatial term's basis spec.
fn set_spatial_aniso_log_scales(
    spec: &mut TermCollectionSpec,
    term_idx: usize,
    eta: Vec<f64>,
) -> Result<(), EstimationError> {
    let eta = center_aniso_log_scales(&eta);
    let Some(term) = spec.smooth_terms.get_mut(term_idx) else {
        crate::bail_invalid_estim!("spatial aniso_log_scales term index {term_idx} out of range");
    };
    match &mut term.basis {
        SmoothBasisSpec::Matern { spec, .. } => {
            spec.aniso_log_scales = Some(eta);
            Ok(())
        }
        SmoothBasisSpec::Duchon { spec, .. } => {
            spec.aniso_log_scales = Some(eta);
            Ok(())
        }
        _ => Err(EstimationError::InvalidInput(format!(
            "term '{}' does not support aniso_log_scales",
            term.name
        ))),
    }
}

/// Sync knot-cloud-derived anisotropy contrasts from basis metadata back into
/// the mutable spec so the optimizer starts from the correct eta values.
///
/// Call this after building the smooth design but before initializing the
/// optimizer's psi coordinates. For each spatial term whose metadata contains
/// computed `aniso_log_scales`, this writes them into the spec.
pub(crate) fn sync_aniso_contrasts_from_metadata(
    spec: &mut TermCollectionSpec,
    design: &SmoothDesign,
) {
    for (term_idx, term) in design.terms.iter().enumerate() {
        let meta_aniso = match &term.metadata {
            BasisMetadata::Matern {
                aniso_log_scales, ..
            } => aniso_log_scales.clone(),
            BasisMetadata::Duchon {
                aniso_log_scales, ..
            } => aniso_log_scales.clone(),
            _ => None,
        };
        if let Some(eta) = meta_aniso
            && eta.len() > 1
        {
            set_spatial_aniso_log_scales(spec, term_idx, eta).ok();
        }
    }
}

#[derive(Debug, Clone)]
pub struct SpatialLengthScaleOptimizationOptions {
    /// Enable outer-loop optimization over spatial κ (= 1 / length_scale)
    /// for supported radial-kernel smooths.
    /// This applies to ThinPlate, Matérn, and Duchon terms.
    pub enabled: bool,
    /// Maximum number of outer iterations in the exact joint [rho, psi] solve.
    pub max_outer_iter: usize,
    /// Relative improvement threshold for terminating the outer solve.
    pub rel_tol: f64,
    /// Initial log(length_scale) perturbation used for seed construction.
    pub log_step: f64,
    /// Minimum allowed length_scale during κ search.
    pub min_length_scale: f64,
    /// Maximum allowed length_scale during κ search.
    pub max_length_scale: f64,
    /// Automatic geometry-initializer threshold for large-scale spatial fits.
    ///
    /// When n exceeds twice this value, the fitter uses a spatially stratified
    /// subsample only to seed κ/anisotropy geometry: centers are resolved,
    /// axis contrasts are initialized from center/data spread, and one or two
    /// cheap ψ reseeding updates are applied. It never runs PIRLS, REML, ARC,
    /// BFGS, or any recursive optimizer on the pilot.
    ///
    /// The final coefficients, smoothing parameters, and spatial geometry are
    /// always optimized on the full dataset.
    ///
    /// Set to 0 to skip the pilot geometry initializer.
    pub pilot_subsample_threshold: usize,
}

impl Default for SpatialLengthScaleOptimizationOptions {
    fn default() -> Self {
        Self {
            enabled: true,
            max_outer_iter: 80,
            rel_tol: 1e-4,
            log_step: std::f64::consts::LN_2,
            min_length_scale: 1e-3,
            max_length_scale: 1e3,
            pilot_subsample_threshold: 10_000,
        }
    }
}

impl SpatialLengthScaleOptimizationOptions {
    /// Validate the struct's invariants. Callers that construct these options
    /// from external input (CLI, config, Python API) should call this before
    /// passing the options into the fitter. Returns `Err` with a descriptive
    /// message when an invariant is violated; the fitter then panics or
    /// returns `EstimationError` at its own boundary.
    ///
    /// Invariants:
    ///   * `min_length_scale > 0`, finite
    ///   * `max_length_scale > 0`, finite
    ///   * `min_length_scale < max_length_scale`
    ///   * `rel_tol > 0`, finite
    ///   * `log_step > 0`, finite
    ///
    /// These invariants are what the downstream κ-bound and ψ-window code
    /// assumes (`-log(max_ls)` must be finite, `(min,max)` must not be
    /// inverted, etc.). Without validation, invalid options produce silent
    /// NaN-propagation inside the outer optimizer.
    pub fn validate(&self) -> Result<(), String> {
        if !self.min_length_scale.is_finite() || self.min_length_scale <= 0.0 {
            return Err(SmoothError::invalid_config(format!(
                "SpatialLengthScaleOptimizationOptions::min_length_scale must be > 0 and finite, got {}",
                self.min_length_scale
            ))
            .into());
        }
        if !self.max_length_scale.is_finite() || self.max_length_scale <= 0.0 {
            return Err(SmoothError::invalid_config(format!(
                "SpatialLengthScaleOptimizationOptions::max_length_scale must be > 0 and finite, got {}",
                self.max_length_scale
            ))
            .into());
        }
        if self.min_length_scale >= self.max_length_scale {
            return Err(SmoothError::invalid_config(format!(
                "SpatialLengthScaleOptimizationOptions requires min_length_scale < max_length_scale, got min={} max={}",
                self.min_length_scale, self.max_length_scale
            ))
            .into());
        }
        if !self.rel_tol.is_finite() || self.rel_tol <= 0.0 {
            return Err(SmoothError::invalid_config(format!(
                "SpatialLengthScaleOptimizationOptions::rel_tol must be > 0 and finite, got {}",
                self.rel_tol
            ))
            .into());
        }
        if !self.log_step.is_finite() || self.log_step <= 0.0 {
            return Err(SmoothError::invalid_config(format!(
                "SpatialLengthScaleOptimizationOptions::log_step must be > 0 and finite, got {}",
                self.log_step
            ))
            .into());
        }
        Ok(())
    }
}

#[derive(Debug, Clone)]
struct RandomEffectBlock {
    name: String,
    /// O(n) group-label vector: group_ids[i] = column index in [0, num_groups).
    /// `None` if the observation's level is not in the kept set.
    group_ids: Vec<Option<usize>>,
    num_groups: usize,
    kept_levels: Vec<u64>,
}

const BLOCK_SPARSE_ZERO_EPS: f64 = 1e-12;

const BLOCK_SPARSE_MAX_DENSITY: f64 = 0.20;

fn blocks_have_intrinsic_sparse_structure(blocks: &[DesignBlock]) -> bool {
    blocks
        .iter()
        .any(|block| matches!(block, DesignBlock::Sparse(_) | DesignBlock::RandomEffect(_)))
}

fn sparse_compatible_block_nnz(block: &DesignBlock) -> Option<usize> {
    match block {
        DesignBlock::Intercept(n) => Some(*n),
        DesignBlock::RandomEffect(op) => {
            Some(op.group_ids.iter().filter(|gid| gid.is_some()).count())
        }
        DesignBlock::Sparse(sparse) => Some(sparse.val().len()),
        DesignBlock::Dense(dense) => dense.as_dense_ref().map(|matrix| {
            matrix
                .iter()
                .filter(|&&value| value.abs() > BLOCK_SPARSE_ZERO_EPS)
                .count()
        }),
    }
}

fn try_build_sparse_design_from_blocks(
    blocks: &[DesignBlock],
) -> Result<Option<DesignMatrix>, BasisError> {
    if blocks.is_empty() {
        return Ok(None);
    }
    let nrows = blocks[0].nrows();
    let ncols: usize = blocks.iter().map(DesignBlock::ncols).sum();
    if nrows == 0 || ncols == 0 || ncols <= 32 {
        return Ok(None);
    }

    let preserve_sparse_storage = blocks_have_intrinsic_sparse_structure(blocks);
    let sparse_nnz_limit = if preserve_sparse_storage {
        usize::MAX
    } else {
        let total_cells = nrows.saturating_mul(ncols);
        ((total_cells as f64) * BLOCK_SPARSE_MAX_DENSITY).floor() as usize
    };
    let mut nnz = 0usize;
    for block in blocks {
        let block_nnz = if let Some(block_nnz) = sparse_compatible_block_nnz(block) {
            block_nnz
        } else {
            return Ok(None);
        };
        nnz = nnz.saturating_add(block_nnz);
        if nnz > sparse_nnz_limit {
            return Ok(None);
        }
    }

    let mut triplets = Vec::<Triplet<usize, usize, f64>>::with_capacity(nnz);
    let mut col_offset = 0usize;
    for block in blocks {
        match block {
            DesignBlock::Intercept(n) => {
                for row in 0..*n {
                    triplets.push(Triplet::new(row, col_offset, 1.0));
                }
            }
            DesignBlock::RandomEffect(op) => {
                for (row, group_id) in op.group_ids.iter().enumerate() {
                    if let Some(group) = group_id {
                        triplets.push(Triplet::new(row, col_offset + group, 1.0));
                    }
                }
            }
            DesignBlock::Sparse(sparse) => {
                let (symbolic, values) = sparse.parts();
                let col_ptr = symbolic.col_ptr();
                let row_idx = symbolic.row_idx();
                for col in 0..sparse.ncols() {
                    for idx in col_ptr[col]..col_ptr[col + 1] {
                        let value = values[idx];
                        if value.abs() > BLOCK_SPARSE_ZERO_EPS {
                            triplets.push(Triplet::new(row_idx[idx], col_offset + col, value));
                        }
                    }
                }
            }
            DesignBlock::Dense(dense) => {
                let matrix = dense.as_dense_ref().ok_or_else(|| {
                    BasisError::InvalidInput(
                        "sparse-compatible block assembly requires materialized dense blocks"
                            .to_string(),
                    )
                })?;
                for row in 0..matrix.nrows() {
                    for col in 0..matrix.ncols() {
                        let value = matrix[[row, col]];
                        if value.abs() > BLOCK_SPARSE_ZERO_EPS {
                            triplets.push(Triplet::new(row, col_offset + col, value));
                        }
                    }
                }
            }
        }
        col_offset += block.ncols();
    }

    let sparse = SparseColMat::try_new_from_triplets(nrows, ncols, &triplets).map_err(|_| {
        BasisError::SparseCreation("failed to assemble sparse term-collection design".to_string())
    })?;
    Ok(Some(DesignMatrix::Sparse(
        crate::matrix::SparseDesignMatrix::new(sparse),
    )))
}

fn assemble_term_collection_design_matrix(
    blocks: Vec<DesignBlock>,
) -> Result<DesignMatrix, BasisError> {
    if let Some(sparse) = try_build_sparse_design_from_blocks(&blocks)? {
        return Ok(sparse);
    }
    let block_op = BlockDesignOperator::new(blocks).map_err(|e| {
        BasisError::InvalidInput(format!("failed to build block design operator: {e}"))
    })?;
    Ok(DesignMatrix::Dense(crate::matrix::DenseDesignMatrix::from(
        Arc::new(block_op),
    )))
}

fn select_columns(data: ArrayView2<'_, f64>, cols: &[usize]) -> Result<Array2<f64>, BasisError> {
    let n = data.nrows();
    let p = data.ncols();
    for &c in cols {
        if c >= p {
            crate::bail_dim_basis!("feature column {c} is out of bounds for data with {p} columns");
        }
    }
    let mut out = Array2::<f64>::zeros((n, cols.len()));
    for (j, &c) in cols.iter().enumerate() {
        out.column_mut(j).assign(&data.column(c));
    }
    Ok(out)
}

fn nonfinite_value_label(value: f64) -> &'static str {
    if value.is_nan() {
        "NaN"
    } else if value.is_sign_positive() {
        "+Inf"
    } else {
        "-Inf"
    }
}

fn validate_term_feature_column_finite(
    data: ArrayView2<'_, f64>,
    term_kind: &str,
    term_name: &str,
    feature_col: usize,
) -> Result<(), BasisError> {
    let p = data.ncols();
    if feature_col >= p {
        crate::bail_dim_basis!(
            "{term_kind} term '{term_name}' feature column {feature_col} out of bounds for {p} columns"
        );
    }
    for (row, &value) in data.column(feature_col).iter().enumerate() {
        if !value.is_finite() {
            crate::bail_invalid_basis!(
                "{term_kind} term '{term_name}' feature column {feature_col} row {row} contains non-finite value {}",
                nonfinite_value_label(value)
            );
        }
    }
    Ok(())
}

fn validate_smooth_terms_finite_inputs(
    data: ArrayView2<'_, f64>,
    terms: &[SmoothTermSpec],
) -> Result<(), BasisError> {
    for term in terms {
        for feature_col in smooth_term_feature_cols(term) {
            validate_term_feature_column_finite(data, "smooth", &term.name, feature_col)?;
        }
    }
    Ok(())
}

fn validate_term_collection_finite_inputs(
    data: ArrayView2<'_, f64>,
    spec: &TermCollectionSpec,
) -> Result<(), BasisError> {
    for term in &spec.linear_terms {
        validate_term_feature_column_finite(data, "linear", &term.name, term.feature_col)?;
    }
    for term in &spec.random_effect_terms {
        validate_term_feature_column_finite(data, "random-effect", &term.name, term.feature_col)?;
    }
    validate_smooth_terms_finite_inputs(data, &spec.smooth_terms)
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
struct JointSpatialCenterGroupKey {
    feature_cols: Vec<usize>,
    strategy_kind: CenterStrategyKind,
    strategy_aux: usize,
    requested_num_centers: usize,
    input_scale_bits: Option<Vec<u64>>,
}

fn spatial_term_min_center_count(term: &SmoothTermSpec) -> usize {
    match &term.basis {
        SmoothBasisSpec::ThinPlate { feature_cols, .. } => feature_cols.len() + 1,
        SmoothBasisSpec::Duchon {
            feature_cols, spec, ..
        } => match spec.nullspace_order {
            crate::basis::DuchonNullspaceOrder::Zero => 1,
            crate::basis::DuchonNullspaceOrder::Linear => feature_cols.len() + 1,
            crate::basis::DuchonNullspaceOrder::Degree(degree) => {
                crate::basis::duchon_nullspace_dimension(feature_cols.len(), degree)
            }
        },
        SmoothBasisSpec::Matern { .. } => 1,
        _ => 1,
    }
}

fn spatial_term_group_key(term: &SmoothTermSpec) -> Option<JointSpatialCenterGroupKey> {
    let (feature_cols, strategy, input_scales) = match &term.basis {
        SmoothBasisSpec::ThinPlate {
            feature_cols,
            spec,
            input_scales,
        } => (feature_cols, &spec.center_strategy, input_scales.as_ref()),
        SmoothBasisSpec::Matern {
            feature_cols,
            spec,
            input_scales,
        } => (feature_cols, &spec.center_strategy, input_scales.as_ref()),
        SmoothBasisSpec::Duchon {
            feature_cols,
            spec,
            input_scales,
        } => (feature_cols, &spec.center_strategy, input_scales.as_ref()),
        _ => return None,
    };
    let strategy_kind = center_strategy_kind(strategy);
    let strategy_aux = match strategy {
        CenterStrategy::Auto(inner) => match inner.as_ref() {
            CenterStrategy::KMeans { max_iter, .. } => *max_iter,
            CenterStrategy::UniformGrid { points_per_dim } => *points_per_dim,
            _ => 0,
        },
        CenterStrategy::KMeans { max_iter, .. } => *max_iter,
        CenterStrategy::UniformGrid { points_per_dim } => *points_per_dim,
        _ => 0,
    };
    Some(JointSpatialCenterGroupKey {
        feature_cols: feature_cols.clone(),
        strategy_kind,
        strategy_aux,
        requested_num_centers: center_strategy_num_centers(strategy)?,
        input_scale_bits: input_scales
            .map(|values| values.iter().map(|value| value.to_bits()).collect()),
    })
}

fn spatial_term_center_strategy(term: &SmoothTermSpec) -> Option<&CenterStrategy> {
    match &term.basis {
        SmoothBasisSpec::ThinPlate { spec, .. } => Some(&spec.center_strategy),
        SmoothBasisSpec::Matern { spec, .. } => Some(&spec.center_strategy),
        SmoothBasisSpec::Duchon { spec, .. } => Some(&spec.center_strategy),
        _ => None,
    }
}

fn set_spatial_term_centers(
    term: &mut SmoothTermSpec,
    centers: Array2<f64>,
) -> Result<(), BasisError> {
    match &mut term.basis {
        SmoothBasisSpec::ThinPlate { spec, .. } => {
            spec.center_strategy = CenterStrategy::UserProvided(centers);
            Ok(())
        }
        SmoothBasisSpec::Matern { spec, .. } => {
            spec.center_strategy = CenterStrategy::UserProvided(centers);
            Ok(())
        }
        SmoothBasisSpec::Duchon { spec, .. } => {
            spec.center_strategy = CenterStrategy::UserProvided(centers);
            Ok(())
        }
        _ => Err(BasisError::InvalidInput(format!(
            "term '{}' does not support spatial center planning",
            term.name
        ))),
    }
}

fn standardized_spatial_term_data(
    data: ArrayView2<'_, f64>,
    term: &SmoothTermSpec,
) -> Result<Array2<f64>, BasisError> {
    let (feature_cols, input_scales) = match &term.basis {
        SmoothBasisSpec::ThinPlate {
            feature_cols,
            input_scales,
            ..
        }
        | SmoothBasisSpec::Matern {
            feature_cols,
            input_scales,
            ..
        }
        | SmoothBasisSpec::Duchon {
            feature_cols,
            input_scales,
            ..
        } => (feature_cols, input_scales.as_ref()),
        _ => {
            crate::bail_invalid_basis!("term '{}' is not a spatial smooth", term.name);
        }
    };
    let mut x = select_columns(data, feature_cols)?;
    if let Some(scales) = input_scales {
        apply_input_standardization(&mut x, scales);
    } else if let Some(scales) = compute_spatial_input_scales(x.view()) {
        apply_input_standardization(&mut x, &scales);
    }
    Ok(x)
}

fn plan_joint_spatial_centers_for_term_blocks(
    data: ArrayView2<'_, f64>,
    term_blocks: &[Vec<SmoothTermSpec>],
) -> Result<Vec<Vec<SmoothTermSpec>>, BasisError> {
    let mut planned_blocks = term_blocks.to_vec();
    let n = data.nrows();
    let mut groups: BTreeMap<JointSpatialCenterGroupKey, Vec<(usize, usize)>> = BTreeMap::new();

    for (block_idx, terms) in planned_blocks.iter().enumerate() {
        for (term_idx, term) in terms.iter().enumerate() {
            let Some(strategy) = spatial_term_center_strategy(term) else {
                continue;
            };
            if !center_strategy_is_auto(strategy) {
                continue;
            }
            let Some(group_key) = spatial_term_group_key(term) else {
                continue;
            };
            if !matches!(
                group_key.strategy_kind,
                CenterStrategyKind::EqualMass
                    | CenterStrategyKind::EqualMassCovarRepresentative
                    | CenterStrategyKind::FarthestPoint
                    | CenterStrategyKind::KMeans
            ) {
                continue;
            }
            if center_strategy_num_centers(strategy).is_none() {
                continue;
            }
            groups
                .entry(group_key)
                .or_default()
                .push((block_idx, term_idx));
        }
    }

    for (group_key, members) in groups {
        if members.len() < 2 {
            continue;
        }
        let min_required = members
            .iter()
            .map(|&(block_idx, term_idx)| {
                spatial_term_min_center_count(&planned_blocks[block_idx][term_idx])
            })
            .max()
            .unwrap_or(1);
        let joint_centers = group_key
            .requested_num_centers
            .max(min_required)
            .min(n.max(1));
        let (first_block_idx, first_term_idx) = members[0];
        let prototype = &planned_blocks[first_block_idx][first_term_idx];
        let standardized = standardized_spatial_term_data(data, prototype)?;
        let strategy = spatial_term_center_strategy(prototype).ok_or_else(|| {
            BasisError::InvalidInput(format!(
                "term '{}' lost its spatial center strategy during joint planning",
                prototype.name
            ))
        })?;
        let joint_strategy = center_strategy_with_num_centers(strategy, joint_centers)?;
        let shared_centers = select_centers_by_strategy(standardized.view(), &joint_strategy)?;
        log::info!(
            "sharing {} spatial centers across {} smooth terms over columns {:?} (requested {} centers)",
            shared_centers.nrows(),
            members.len(),
            group_key.feature_cols,
            group_key.requested_num_centers,
        );
        for (block_idx, term_idx) in members {
            set_spatial_term_centers(
                &mut planned_blocks[block_idx][term_idx],
                shared_centers.clone(),
            )?;
        }
    }

    // Sentinel auto-init: Matern and thin-plate builders write length_scale =
    // 0.0 when the user didn't pass `length_scale=...`. Replace those with a
    // data-driven initialization here so REML starts in a regime where it can
    // escape; the hard-coded 1.0 default was a basin from which ν ≥ 5/2 Matern
    // could not recover on high-frequency truths, silently collapsing the fit
    // to a near-constant prediction.
    for block in planned_blocks.iter_mut() {
        for term in block.iter_mut() {
            auto_init_length_scale_in_place(data, term);
        }
    }

    Ok(planned_blocks)
}

/// Compute a data-driven initial length scale from the per-axis range of the
/// feature columns. The heuristic `max_range / sqrt(n)` puts the kernel on
/// the wiggly side of REML's basin so the optimizer can grow it back if the
/// signal is smooth, but is small enough that high-frequency truths remain
/// reachable for smoother kernels (ν ≥ 5/2). Clamped to a tiny positive
/// floor so degenerate constant-input columns can't produce 0.
fn auto_initial_length_scale(data: ArrayView2<'_, f64>, feature_cols: &[usize]) -> f64 {
    /// Tiny positive floor for the auto length scale, guarding against a zero
    /// kernel range when every feature column is (near-)constant.
    const LENGTH_SCALE_FLOOR: f64 = 1e-6;
    let n = data.nrows();
    if n == 0 || feature_cols.is_empty() {
        return 1.0;
    }
    let mut max_range = 0.0_f64;
    for &c in feature_cols {
        if c >= data.ncols() {
            continue;
        }
        let col = data.column(c);
        let mut lo = f64::INFINITY;
        let mut hi = f64::NEG_INFINITY;
        for &v in col.iter() {
            if v.is_finite() {
                if v < lo {
                    lo = v;
                }
                if v > hi {
                    hi = v;
                }
            }
        }
        if hi > lo {
            let r = hi - lo;
            if r > max_range {
                max_range = r;
            }
        }
    }
    if !max_range.is_finite() || max_range <= 0.0 {
        return 1.0;
    }
    let init = max_range / (n as f64).sqrt();
    init.max(LENGTH_SCALE_FLOOR).min(max_range)
}

/// Walk a term and, if it is a Matern or thin-plate smooth whose length_scale
/// was left at the auto sentinel (`0.0`), overwrite it with
/// [`auto_initial_length_scale`].
fn auto_init_length_scale_in_place(data: ArrayView2<'_, f64>, term: &mut SmoothTermSpec) {
    auto_init_length_scale_in_basis(data, &mut term.basis);
}

/// Replace the `0.0` auto-init length-scale sentinel with a data-derived value
/// for any Matern / thin-plate kernel reachable from this basis — including the
/// inner kernel of a `by=`/factor-smooth wrapper.
///
/// `by=<factor>` and the sum-to-zero factor smooth wrap a spatial kernel inside
/// `SmoothBasisSpec::ByVariable` / `SmoothBasisSpec::FactorSumToZero` /
/// `SmoothBasisSpec::BySmooth`, so the wrapper variant is what the planner sees.
/// Without recursing into the wrapped basis the inner ThinPlate/Matern keeps the
/// `0.0` sentinel (the post-`1605b3a6e` builder default), which makes the kernel
/// distance divide by `length_scale² = 0`, producing a non-finite design at both
/// fit and predict time. Recurse so the inner kernel is initialized identically
/// to a top-level one.
fn auto_init_length_scale_in_basis(data: ArrayView2<'_, f64>, basis: &mut SmoothBasisSpec) {
    match basis {
        SmoothBasisSpec::Matern {
            feature_cols, spec, ..
        } => {
            if spec.length_scale == 0.0 {
                spec.length_scale = auto_initial_length_scale(data, feature_cols);
            }
        }
        SmoothBasisSpec::ThinPlate {
            feature_cols, spec, ..
        } => {
            if spec.length_scale == 0.0 {
                spec.length_scale = auto_initial_length_scale(data, feature_cols);
            }
        }
        SmoothBasisSpec::ByVariable { inner, .. }
        | SmoothBasisSpec::FactorSumToZero { inner, .. } => {
            auto_init_length_scale_in_basis(data, inner);
        }
        SmoothBasisSpec::BySmooth { smooth, .. } => {
            auto_init_length_scale_in_basis(data, smooth);
        }
        _ => {}
    }
}

impl LinearFitConditioning {
    fn from_columns(design: &TermCollectionDesign, selected_cols: &[usize]) -> Self {
        const SCALE_EPS: f64 = 1e-12;
        let n = design.design.nrows();
        let p = design.design.ncols();
        let mut columns = Vec::with_capacity(selected_cols.len());
        if n == 0 || selected_cols.is_empty() {
            return Self {
                intercept_idx: design.intercept_range.start,
                columns,
            };
        }
        let chunk_rows = crate::linalg::utils::row_chunk_for_byte_budget(n, p);
        // Two-pass mean/variance so operator-backed designs don't need to
        // materialize the full dense matrix. Pass 1 accumulates per-column
        // sums; pass 2 accumulates the sum of squared deviations from the
        // pass-1 mean. This matches the original `Σ (x − mean)² / n` formula
        // without the catastrophic cancellation of `E[X²] − E[X]²`.
        let mut sums = vec![0.0_f64; selected_cols.len()];
        for start in (0..n).step_by(chunk_rows) {
            let end = (start + chunk_rows).min(n);
            let chunk = design
                .design
                .try_row_chunk(start..end)
                .expect("LinearFitConditioning::from_columns row chunk failed");
            for (k, &col_idx) in selected_cols.iter().enumerate() {
                let column = chunk.column(col_idx);
                for &v in column.iter() {
                    sums[k] += v;
                }
            }
        }
        let inv_n = 1.0_f64 / n as f64;
        let means: Vec<f64> = sums.iter().map(|&s| s * inv_n).collect();
        let mut sq_devs = vec![0.0_f64; selected_cols.len()];
        for start in (0..n).step_by(chunk_rows) {
            let end = (start + chunk_rows).min(n);
            let chunk = design
                .design
                .try_row_chunk(start..end)
                .expect("LinearFitConditioning::from_columns row chunk failed");
            for (k, &col_idx) in selected_cols.iter().enumerate() {
                let mean_k = means[k];
                let column = chunk.column(col_idx);
                for &v in column.iter() {
                    let d = v - mean_k;
                    sq_devs[k] += d * d;
                }
            }
        }
        for (k, &col_idx) in selected_cols.iter().enumerate() {
            let mean = means[k];
            let var = sq_devs[k] * inv_n;
            let (mean, scale) = if var.is_finite() && var > SCALE_EPS * SCALE_EPS {
                (mean, var.sqrt())
            } else {
                // Leave nearly-constant columns untouched; centering them would collapse
                // the design column to ~0 and change the model rather than just condition it.
                (0.0, 1.0)
            };
            columns.push(LinearColumnConditioning {
                col_idx,
                mean,
                scale,
            });
        }
        Self {
            intercept_idx: design.intercept_range.start,
            columns,
        }
    }

    fn apply_to_design(&self, design: &Array2<f64>) -> Array2<f64> {
        let mut out = design.clone();
        for col in &self.columns {
            {
                let mut dst = out.column_mut(col.col_idx);
                dst -= col.mean;
            }
            if col.scale != 1.0 {
                out.column_mut(col.col_idx).mapv_inplace(|v| v / col.scale);
            }
        }
        out
    }

    fn transform_matrix_columnswith_a(&self, mat: &Array2<f64>) -> Array2<f64> {
        let mut out = mat.clone();
        let intercept = self.intercept_idx;
        for col in &self.columns {
            let intercept_col = out.column(intercept).to_owned();
            let mut target = out.column_mut(col.col_idx);
            target -= &(intercept_col * col.mean);
            if col.scale != 1.0 {
                target.mapv_inplace(|v| v / col.scale);
            }
        }
        out
    }

    fn transform_matrixrowswith_a_transpose(&self, mat: &Array2<f64>) -> Array2<f64> {
        let mut out = mat.clone();
        let intercept = self.intercept_idx;
        for col in &self.columns {
            let interceptrow = out.row(intercept).to_owned();
            let mut target = out.row_mut(col.col_idx);
            target -= &(interceptrow * col.mean);
            if col.scale != 1.0 {
                target.mapv_inplace(|v| v / col.scale);
            }
        }
        out
    }

    /// Left-multiply `mat_internal` by `M⁻ᵀ` where `M⁻¹[intercept, j] = mean_j`
    /// and `M⁻¹[j, j] = scale_j` for each conditioned column. Used together
    /// with [`Self::right_multiply_by_m_inv`] to back-transform an internal
    /// penalized Hessian to the original coefficient basis.
    fn left_multiply_by_m_inv_transpose(&self, mat_internal: &Array2<f64>) -> Array2<f64> {
        let mut out = mat_internal.clone();
        let intercept = self.intercept_idx;
        let interceptrow_snapshot = mat_internal.row(intercept).to_owned();
        for col in &self.columns {
            if col.scale != 1.0 {
                out.row_mut(col.col_idx).mapv_inplace(|v| v * col.scale);
            }
            if col.mean != 0.0 {
                let mut target = out.row_mut(col.col_idx);
                target += &(&interceptrow_snapshot * col.mean);
            }
        }
        out
    }

    /// Right-multiply `mat_internal` by `M⁻¹`. Mirror of
    /// [`Self::left_multiply_by_m_inv_transpose`] on columns.
    fn right_multiply_by_m_inv(&self, mat_internal: &Array2<f64>) -> Array2<f64> {
        let mut out = mat_internal.clone();
        let intercept = self.intercept_idx;
        let intercept_col_snapshot = mat_internal.column(intercept).to_owned();
        for col in &self.columns {
            if col.scale != 1.0 {
                out.column_mut(col.col_idx).mapv_inplace(|v| v * col.scale);
            }
            if col.mean != 0.0 {
                let mut target = out.column_mut(col.col_idx);
                target += &(&intercept_col_snapshot * col.mean);
            }
        }
        out
    }

    /// Transform blockwise penalties through the conditioning.
    ///
    /// For block-local penalties whose `col_range` does not overlap with any
    /// conditioning column, the transform is identity (the conditioning only
    /// affects unpenalized linear columns). In that common case the penalty
    /// passes through unchanged, avoiding O(p²) materialization entirely.
    fn transform_blockwise_penalties_to_internal(
        &self,
        penalties: &[BlockwisePenalty],
        p: usize,
    ) -> Vec<PenaltySpec> {
        let conditioning_cols: std::collections::HashSet<usize> =
            self.columns.iter().map(|c| c.col_idx).collect();
        penalties
            .iter()
            .map(|bp| {
                let overlaps =
                    (bp.col_range.start..bp.col_range.end).any(|j| conditioning_cols.contains(&j));
                if overlaps {
                    // Rare: penalty block overlaps conditioning columns.
                    // Fall back to dense transform.
                    let global = bp.to_global(p);
                    let right = self.transform_matrix_columnswith_a(&global);
                    let transformed = self.transform_matrixrowswith_a_transpose(&right);
                    PenaltySpec::Dense(transformed)
                } else {
                    // Common: smooth penalty block doesn't touch linear columns.
                    // The conditioning is identity on this block.
                    PenaltySpec::from_blockwise(bp.clone())
                }
            })
            .collect()
    }

    fn backtransform_beta(&self, beta_internal: &Array1<f64>) -> Array1<f64> {
        let mut beta = beta_internal.clone();
        let intercept = self.intercept_idx;
        for col in &self.columns {
            beta[intercept] -= beta_internal[col.col_idx] * col.mean / col.scale;
            beta[col.col_idx] = beta_internal[col.col_idx] / col.scale;
        }
        beta
    }

    /// `H_orig = M⁻ᵀ · H_int · M⁻¹`, derived from
    /// `L_int(β_int) = L_orig(M · β_int)` via the chain rule.
    fn transform_penalized_hessian_to_original(&self, h_internal: &Array2<f64>) -> Array2<f64> {
        let right = self.right_multiply_by_m_inv(h_internal);
        self.left_multiply_by_m_inv_transpose(&right)
    }

    fn internal_bounds_for(&self, col_idx: usize, min: f64, max: f64) -> (f64, f64) {
        if let Some(col) = self.columns.iter().find(|c| c.col_idx == col_idx) {
            (min * col.scale, max * col.scale)
        } else {
            (min, max)
        }
    }
}

fn freeze_raw_spatial_metadata(metadata: BasisMetadata, raw_cols: usize) -> BasisMetadata {
    match metadata {
        BasisMetadata::ThinPlate {
            centers,
            length_scale,
            periodic,
            identifiability_transform: None,
            input_scales,
            radial_reparam,
        } => BasisMetadata::ThinPlate {
            centers,
            length_scale,
            periodic,
            identifiability_transform: Some(Array2::eye(raw_cols)),
            input_scales,
            radial_reparam,
        },
        BasisMetadata::Duchon {
            centers,
            length_scale,
            periodic,
            power,
            nullspace_order,
            identifiability_transform: None,
            input_scales,
            aniso_log_scales,
            operator_collocation_points,
        } => BasisMetadata::Duchon {
            centers,
            length_scale,
            periodic,
            power,
            nullspace_order,
            identifiability_transform: Some(Array2::eye(raw_cols)),
            input_scales,
            aniso_log_scales,
            operator_collocation_points,
        },
        other => other,
    }
}

fn matern_operator_penalty_triplet_from_metadata(
    metadata: &BasisMetadata,
) -> Result<(Vec<Array2<f64>>, Vec<usize>, Vec<PenaltyInfo>), BasisError> {
    let BasisMetadata::Matern {
        centers,
        length_scale,
        periodic,
        nu,
        include_intercept,
        identifiability_transform,
        aniso_log_scales,
        input_scales,
        ..
    } = metadata
    else {
        crate::bail_invalid_basis!("Matérn operator penalties require Matérn metadata");
    };
    // The metadata records `length_scale` in *original* (un-standardized) data
    // coordinates, while `centers` live in the *standardized* coordinate frame
    // (per-axis division by `input_scales`). The realized design built the
    // kernel against those standardized centers using the σ_geom-compensated
    // effective length scale `length_scale / σ_geom`. The collocation operators
    // here are evaluated on the same standardized centers, so they must use the
    // SAME effective length scale — otherwise the penalty regularizes a
    // different RKHS range than the design lives in, leaving rough coefficient
    // directions effectively unpenalized. That mismatch is benign in 1-D
    // (no standardization) but produces a catastrophic out-of-sample blow-up in
    // d ≥ 2 where σ_geom ≠ 1 (#706).
    let penalty_length_scale = match input_scales.as_deref() {
        Some(scales) => compensate_length_scale_for_standardization(*length_scale, scales),
        None => *length_scale,
    };
    let penalty_centers = crate::basis::expand_periodic_centers(centers, periodic.as_deref())?;
    let ops = build_matern_collocation_operator_matrices(
        penalty_centers.view(),
        None,
        penalty_length_scale,
        *nu,
        *include_intercept,
        identifiability_transform.as_ref().map(|z| z.view()),
        aniso_log_scales.as_deref(),
    )?;
    // Gate the operator dials on the Matérn-ν RKHS Sobolev order m = ν + d/2:
    // mass (j=0) is always on, tension (j=1) is on for m > 1, stiffness (j=2)
    // is on for m > 2. The threshold is strict so the roughest kernel ν=1/2 in
    // d=1 (m=1, the exponential/OU H¹ process) sheds both higher operators —
    // its kernel already encodes the H¹ control, so adding an extra tension
    // dial over-smooths the oscillation it is meant to track (#707). The
    // matching gate lives at `DuchonOperatorPenaltySpec::matern_for_smoothness`.
    const ORDER_EPS: f64 = 1e-9;
    let d = penalty_centers.ncols();
    let m = nu.half_integer_value() + 0.5 * d as f64;
    let mut candidates = Vec::with_capacity(3);
    for (raw, source, min_order) in [
        (ops.d0.t().dot(&ops.d0), PenaltySource::OperatorMass, 0.0),
        (ops.d1.t().dot(&ops.d1), PenaltySource::OperatorTension, 1.0),
        (
            ops.d2.t().dot(&ops.d2),
            PenaltySource::OperatorStiffness,
            2.0,
        ),
    ] {
        if min_order > 0.0 && m <= min_order + ORDER_EPS {
            continue;
        }
        let sym = (&raw + &raw.t()) * 0.5;
        let (matrix, normalization_scale) = normalize_penalty_in_constrained_space(&sym);
        candidates.push(PenaltyCandidate {
            matrix,
            nullspace_dim_hint: 0,
            source,
            normalization_scale,
            kronecker_factors: None,
            op: None,
        });
    }
    filter_active_penalty_candidates(candidates)
}

fn normalize_penalty_in_constrained_space(matrix: &Array2<f64>) -> (Array2<f64>, f64) {
    // Constrained-space normalization:
    //   c = ||S_con||_F,  S_tilde = S_con / c.
    // This is the only normalization coherent with a REML objective that is
    // evaluated entirely in constrained coordinates.
    let matrix = (matrix + &matrix.t().to_owned()) * 0.5;
    // Clamp noise-floor negative eigenvalues so β'Sβ is non-negative as a contract, not just in exact arithmetic.
    let matrix = crate::terms::basis::project_penalty_to_psd_cone(&matrix);
    let c = matrix.iter().map(|v| v * v).sum::<f64>().sqrt();
    if c.is_finite() && c > 0.0 {
        (matrix.mapv(|v| v / c), c)
    } else {
        (matrix, 1.0)
    }
}

fn build_periodic_fourier_margin(
    x: ArrayView1<'_, f64>,
    period: f64,
    requested_cols: usize,
    penalty_order: usize,
) -> Result<(Array2<f64>, Array2<f64>, Array1<f64>), BasisError> {
    if !period.is_finite() || period <= 0.0 {
        crate::bail_invalid_basis!(
            "periodic tensor margin requires finite positive period, got {period}"
        );
    }
    let q = requested_cols.max(3);
    let harmonics = q / 2;
    let has_nyquist_cos = q.is_multiple_of(2);
    let mut basis = Array2::<f64>::zeros((x.len(), q));
    basis.column_mut(0).fill(1.0);
    for (i, &xi) in x.iter().enumerate() {
        let angle = 2.0 * std::f64::consts::PI * xi / period;
        let mut col = 1usize;
        for h in 1..=harmonics {
            if col >= q {
                break;
            }
            basis[[i, col]] = (h as f64 * angle).cos();
            col += 1;
            if col >= q {
                break;
            }
            basis[[i, col]] = (h as f64 * angle).sin();
            col += 1;
        }
        if has_nyquist_cos && q > 1 {
            basis[[i, q - 1]] = (harmonics as f64 * angle).cos();
        }
    }
    let mut penalty = Array2::<f64>::zeros((q, q));
    let order = penalty_order.max(1) as i32;
    let mut col = 1usize;
    for h in 1..=harmonics {
        let w = (h as f64).powi(2 * order);
        if col < q {
            penalty[[col, col]] = w;
            col += 1;
        }
        if col < q {
            penalty[[col, col]] = w;
            col += 1;
        }
    }
    if has_nyquist_cos && q > 1 {
        penalty[[q - 1, q - 1]] = (harmonics as f64).powi(2 * order);
    }
    // The Fourier margin has no B-spline knots; this vector is a placeholder
    // that downstream code (the tensor freeze) treats as carrying the periodic
    // control-site count in its *length* and the domain start in `[0]`. Its
    // length MUST equal the basis column count `q` (= `basis.ncols()`): the
    // freeze records `num_basis = knots.len()` and the predict-time rebuild
    // re-derives `q` columns from it, so a `q + 1`-length vector reconstructs
    // one extra column per periodic axis and breaks the frozen identifiability
    // transform (issue #498).
    let knots = Array1::linspace(0.0, period, q);
    Ok((basis, penalty, knots))
}

fn tensor_product_design_from_sparse_marginals(
    marginal_sparse: &[&SparseColMat<usize, f64>],
) -> Result<SparseColMat<usize, f64>, BasisError> {
    if marginal_sparse.is_empty() {
        crate::bail_invalid_basis!("TensorBSpline requires at least one marginal basis");
    }
    let n = marginal_sparse[0].nrows();
    for (i, m) in marginal_sparse.iter().enumerate().skip(1) {
        if m.nrows() != n {
            crate::bail_dim_basis!(
                "tensor sparse marginal row mismatch at dim {i}: expected {n}, got {}",
                m.nrows()
            );
        }
    }
    let dims: Vec<usize> = marginal_sparse.iter().map(|m| m.ncols()).collect();
    let total_cols = dims.iter().try_fold(1usize, |acc, &q| {
        acc.checked_mul(q)
            .ok_or_else(|| BasisError::DimensionMismatch("tensor basis too large".to_string()))
    })?;
    let mut strides = vec![1usize; dims.len()];
    for d in (0..dims.len().saturating_sub(1)).rev() {
        strides[d] = strides[d + 1]
            .checked_mul(dims[d + 1])
            .ok_or_else(|| BasisError::DimensionMismatch("tensor basis too large".to_string()))?;
    }

    use faer::sparse::SparseRowMat;
    let csrs: Vec<SparseRowMat<usize, f64>> = marginal_sparse
        .iter()
        .enumerate()
        .map(|(d, m)| {
            m.as_ref().to_row_major().map_err(|e| {
                BasisError::SparseCreation(format!(
                    "tensor sparse marginal {d} CSR conversion failed: {e:?}"
                ))
            })
        })
        .collect::<Result<Vec<_>, _>>()?;
    let row_ptrs: Vec<&[usize]> = csrs.iter().map(|c| c.symbolic().row_ptr()).collect();
    let col_idxs: Vec<&[usize]> = csrs.iter().map(|c| c.symbolic().col_idx()).collect();
    let vals: Vec<&[f64]> = csrs.iter().map(|c| c.val()).collect();

    use rayon::prelude::*;
    const CHUNK: usize = 1024;
    let num_chunks = n.div_ceil(CHUNK);
    let per_chunk: Vec<Vec<Triplet<usize, usize, f64>>> = (0..num_chunks)
        .into_par_iter()
        .map(|chunk_idx| {
            let row_start = chunk_idx * CHUNK;
            let row_end = (row_start + CHUNK).min(n);
            let mut chunk_triplets = Vec::<Triplet<usize, usize, f64>>::new();
            let mut cur_cols = Vec::<usize>::with_capacity(64);
            let mut cur_vals = Vec::<f64>::with_capacity(64);
            let mut next_cols = Vec::<usize>::with_capacity(64);
            let mut next_vals = Vec::<f64>::with_capacity(64);
            for i in row_start..row_end {
                cur_cols.clear();
                cur_vals.clear();
                cur_cols.push(0);
                cur_vals.push(1.0);
                let mut row_is_zero = false;
                for d in 0..dims.len() {
                    let row_start_d = row_ptrs[d][i];
                    let row_end_d = row_ptrs[d][i + 1];
                    if row_start_d == row_end_d {
                        row_is_zero = true;
                        break;
                    }
                    let stride = strides[d];
                    next_cols.clear();
                    next_vals.clear();
                    next_cols.reserve(cur_cols.len() * (row_end_d - row_start_d));
                    next_vals.reserve(cur_vals.len() * (row_end_d - row_start_d));
                    for (&prev_col, &prev_val) in cur_cols.iter().zip(cur_vals.iter()) {
                        for ptr in row_start_d..row_end_d {
                            let cj = col_idxs[d][ptr];
                            let vj = vals[d][ptr];
                            next_cols.push(prev_col + cj * stride);
                            next_vals.push(prev_val * vj);
                        }
                    }
                    std::mem::swap(&mut cur_cols, &mut next_cols);
                    std::mem::swap(&mut cur_vals, &mut next_vals);
                }
                if row_is_zero {
                    continue;
                }
                for (&col, &val) in cur_cols.iter().zip(cur_vals.iter()) {
                    chunk_triplets.push(Triplet::new(i, col, val));
                }
            }
            chunk_triplets
        })
        .collect();
    let total_nnz: usize = per_chunk.iter().map(Vec::len).sum();
    let mut triplets = Vec::<Triplet<usize, usize, f64>>::with_capacity(total_nnz);
    for chunk in per_chunk {
        triplets.extend(chunk);
    }
    SparseColMat::try_new_from_triplets(n, total_cols, &triplets).map_err(|e| {
        BasisError::SparseCreation(format!(
            "failed to assemble sparse tensor product design: {e:?}"
        ))
    })
}

fn build_tensor_bspline_basis(
    data: ArrayView2<'_, f64>,
    feature_cols: &[usize],
    spec: &TensorBSplineSpec,
) -> Result<BasisBuildResult, BasisError> {
    if feature_cols.is_empty() {
        crate::bail_invalid_basis!("TensorBSpline requires at least one feature column");
    }
    if feature_cols.len() != spec.marginalspecs.len() {
        crate::bail_dim_basis!(
            "TensorBSpline feature/spec mismatch: feature_cols={}, marginalspecs={}",
            feature_cols.len(),
            spec.marginalspecs.len()
        );
    }
    if !spec.periods.is_empty() && spec.periods.len() != feature_cols.len() {
        crate::bail_dim_basis!(
            "TensorBSpline periods length {} does not match feature count {}",
            spec.periods.len(),
            feature_cols.len()
        );
    }
    let p = data.ncols();
    for &c in feature_cols {
        if c >= p {
            crate::bail_dim_basis!(
                "tensor feature column {c} is out of bounds for data with {p} columns"
            );
        }
    }

    let mut marginal_knots = Vec::<Array1<f64>>::with_capacity(feature_cols.len());
    let mut marginal_degrees = Vec::<usize>::with_capacity(feature_cols.len());
    let mut marginalnum_basis = Vec::<usize>::with_capacity(feature_cols.len());
    let mut marginal_penalties = Vec::<Array2<f64>>::with_capacity(feature_cols.len());
    let mut marginal_designs = Vec::<Array2<f64>>::with_capacity(feature_cols.len());
    // Per-margin effective period: either user-set via `spec.periods` (forcing
    // the Fourier path) or implied by a `PeriodicUniform` marginal knotspec
    // (which the 1D B-spline builder already realizes as a periodic basis).
    // Captured here so freeze→reload round-trips both routes back to a
    // `PeriodicUniform` marginal knotspec; otherwise a `PeriodicUniform`
    // margin specified without `spec.periods` would freeze as a plain
    // `Provided(knots)` open spline and lose its wrap-around at predict time.
    let mut marginal_effective_periods = Vec::<Option<f64>>::with_capacity(feature_cols.len());
    // Per-marginal sparse representation, populated when the 1D builder returned
    // a `DesignMatrix::Sparse`. Used to assemble the Khatri-Rao tensor product
    // sparsely (only ∏(degree+1) nonzeros per row) instead of densifying to
    // shape (n, ∏ q_j) up front. When any marginal lacks a sparse form (e.g.
    // periodic B-splines currently realize a dense Array2), we fall back to the
    // existing dense Khatri-Rao path.
    let mut marginal_sparse =
        Vec::<Option<SparseColMat<usize, f64>>>::with_capacity(feature_cols.len());

    // Reuse the robust 1D builder to ensure the same knot validation and
    // marginal difference-penalty construction as standalone smooth terms.
    for (dim, (&col, marginalspec)) in feature_cols
        .iter()
        .zip(spec.marginalspecs.iter())
        .enumerate()
    {
        // Tensor basis uses raw marginal knot-product columns. Applying 1D
        // identifiability constraints here would change marginal penalty sizes
        // without changing the tensor design construction, causing dimension
        // mismatch. Keep marginal builders unconstrained at this stage.
        if let Some(period) = spec.periods.get(dim).and_then(|p| *p) {
            let requested_cols = match marginalspec.knotspec {
                BSplineKnotSpec::Generate {
                    num_internal_knots, ..
                } => num_internal_knots + marginalspec.degree + 1,
                BSplineKnotSpec::Provided(ref knots) => {
                    knots.len().saturating_sub(marginalspec.degree + 1)
                }
                BSplineKnotSpec::Automatic {
                    num_internal_knots, ..
                } => {
                    // Fallback internal-knot count when an automatic marginal has
                    // not yet resolved its knot count at periodic-margin build
                    // time; matches the modest default used for a 1-D `s()`.
                    const DEFAULT_AUTOMATIC_INTERNAL_KNOTS: usize = 8;
                    num_internal_knots.unwrap_or(DEFAULT_AUTOMATIC_INTERNAL_KNOTS)
                        + marginalspec.degree
                        + 1
                }
                BSplineKnotSpec::PeriodicUniform { num_basis, .. } => num_basis,
            };
            let (basis, penalty, knots) = build_periodic_fourier_margin(
                data.column(col),
                period,
                requested_cols,
                marginalspec.penalty_order,
            )?;
            marginal_knots.push(knots);
            marginal_degrees.push(marginalspec.degree);
            marginalnum_basis.push(basis.ncols());
            marginal_designs.push(basis);
            marginal_penalties.push(penalty);
            // Periodic Fourier margins are realized densely; no sparse form
            // is available, so record `None` and force the dense fall-back
            // for the tensor product if any dimension is periodic.
            marginal_sparse.push(None);
            marginal_effective_periods.push(Some(period));
        } else {
            let mut marginal_unconstrained = marginalspec.clone();
            marginal_unconstrained.identifiability = BSplineIdentifiability::None;
            let built = build_bspline_basis_1d(data.column(col), &marginal_unconstrained)?;
            let knots = match built.metadata {
                BasisMetadata::BSpline1D { knots, .. } => knots,
                _ => {
                    crate::bail_invalid_basis!(
                        "internal TensorBSpline error at dim {dim}: expected BSpline1D metadata"
                    );
                }
            };
            marginal_knots.push(knots);
            marginal_degrees.push(marginalspec.degree);
            marginalnum_basis.push(built.design.ncols());
            // Capture the sparse representation of this marginal (when the
            // 1D builder produced one) before densifying for the dense
            // marginal cache used by `tensor_product_design_from_marginals`
            // and `TensorProductDesignOperator`.
            let sparse_view: Option<SparseColMat<usize, f64>> =
                built.design.as_sparse().map(|sd| {
                    let inner: &SparseColMat<usize, f64> = sd;
                    inner.clone()
                });
            marginal_sparse.push(sparse_view);
            marginal_designs.push(built.design.to_dense());
            marginal_penalties.push(
                built
                    .penalties
                    .first()
                    .ok_or_else(|| {
                        BasisError::InvalidInput(format!(
                            "internal TensorBSpline error at dim {dim}: missing marginal penalty"
                        ))
                    })?
                    .clone(),
            );
            built.nullspace_dims.first().ok_or_else(|| {
                BasisError::InvalidInput(format!(
                    "internal TensorBSpline error at dim {dim}: missing marginal nullspace dim"
                ))
            })?;
            // A `PeriodicUniform` marginal knotspec implies the margin is
            // wrap-around: the 1D builder already realized it as a periodic
            // basis, so the tensor product inherits that periodicity. Record
            // the period derived from the knotspec's data range so freeze
            // restores `PeriodicUniform` on the marginal — otherwise the
            // round-trip downgrades it to `Provided(knots)` (an open spline)
            // and predict-time wraps disappear.
            let implied_period = match marginalspec.knotspec {
                BSplineKnotSpec::PeriodicUniform { data_range, .. } => {
                    Some(data_range.1 - data_range.0)
                }
                _ => None,
            };
            marginal_effective_periods.push(implied_period);
        }
    }

    let total_cols: usize = marginalnum_basis.iter().product();
    let mut dense_design = (!matches!(spec.identifiability, TensorBSplineIdentifiability::None))
        .then(|| tensor_product_design_from_marginals(&marginal_designs))
        .transpose()?;
    let mut candidates = Vec::<PenaltyCandidate>::with_capacity(
        marginal_penalties.len() + if spec.double_penalty { 1 } else { 0 },
    );

    // Tensor-product smoothing parameters are one-per-margin.  Therefore the
    // physical penalty attached to a margin must be normalized in that margin's
    // own working coordinates before it is embedded in the full tensor product.
    // Normalizing only the already-Kroneckered matrix would fold arbitrary
    // dimension-dependent identity factors into the margin's lambda and would
    // make anisotropic REML/LAML smoothing depend on the other margins' basis
    // sizes rather than on the marginal roughness operator itself.
    let normalized_marginal_penalties: Vec<(Array2<f64>, f64)> = marginal_penalties
        .iter()
        .map(normalize_penalty_in_constrained_space)
        .collect();
    let mut kronecker_marginal_penalties =
        Vec::<Array2<f64>>::with_capacity(normalized_marginal_penalties.len());

    // Accumulate the Kronecker-sum of the per-margin penalties, `Σ_dim S_dim`,
    // whose null space is exactly the *joint* null space of all marginal
    // penalties — the tensor of the marginal polynomial null spaces, i.e. the
    // bilinear (low-order) directions that no marginal roughness operator
    // touches. The tensor double penalty (below) shrinks only this joint null,
    // never the already-penalized interaction range.
    let mut marginal_kron_sum = Array2::<f64>::zeros((total_cols, total_cols));

    for dim in 0..normalized_marginal_penalties.len() {
        let mut s_dim = Array2::<f64>::eye(1);
        let mut factors = Vec::<Array2<f64>>::with_capacity(marginalnum_basis.len());
        for (j, &qj) in marginalnum_basis.iter().enumerate() {
            let factor = if j == dim {
                normalized_marginal_penalties[j].0.clone()
            } else {
                Array2::<f64>::eye(qj)
            };
            factors.push(factor.clone());
            s_dim = kronecker_product(&s_dim, &factor);
        }
        if dim == kronecker_marginal_penalties.len() {
            kronecker_marginal_penalties.push(normalized_marginal_penalties[dim].0.clone());
        }
        marginal_kron_sum += &s_dim;

        candidates.push(PenaltyCandidate {
            matrix: s_dim,
            nullspace_dim_hint: 0,
            source: PenaltySource::TensorMarginal { dim },
            normalization_scale: normalized_marginal_penalties[dim].1,
            kronecker_factors: Some(factors),
            op: None,
        });
    }

    if spec.double_penalty {
        // mgcv `select=TRUE` semantics, mirrored from the 1D double penalty:
        // the extra penalty shrinks ONLY the directions left unpenalized by the
        // primary (marginal) penalties — here the joint null space of the
        // Kronecker-sum `Σ_dim S_dim` (the bilinear tensor null `Z₀ ⊗ Z₁`).
        // A full identity ridge `I` over *all* tensor coefficients (the prior
        // behavior) instead penalizes the already-penalized interaction range
        // as well, and REML/LAML then places positive weight on it and
        // systematically over-smooths the recovered surface; mgcv's `te`/`ti`
        // carry no such global ridge. Penalizing the null subspace alone keeps
        // the interaction range governed solely by the per-margin λ's.
        if let Some(shrink) =
            crate::terms::basis::build_nullspace_shrinkage_penalty(&marginal_kron_sum)?
        {
            let (matrix, normalization_scale) =
                normalize_penalty_in_constrained_space(&shrink.sym_penalty);
            candidates.push(PenaltyCandidate {
                matrix,
                nullspace_dim_hint: 0,
                source: PenaltySource::TensorGlobalRidge,
                normalization_scale,
                kronecker_factors: None,
                op: None,
            });
        }
    }

    let z_opt = match &spec.identifiability {
        TensorBSplineIdentifiability::None => None,
        TensorBSplineIdentifiability::SumToZero => {
            if total_cols < 2 {
                crate::bail_invalid_basis!(
                    "TensorBSpline requires at least 2 basis coefficients to enforce sum-to-zero identifiability"
                );
            }
            let dense_design_ref = dense_design.as_ref().ok_or_else(|| {
                BasisError::InvalidInput(
                    "tensor sum-to-zero identifiability requires a realized basis".to_string(),
                )
            })?;
            let (_, z) = apply_sum_to_zero_constraint(dense_design_ref.view(), None)?;
            Some(z)
        }
        TensorBSplineIdentifiability::MarginalSumToZero => {
            // `ti(...)`: drop the marginal main effects by centering every
            // margin independently, then form the tensor product of the
            // centered margins. Concretely, each margin `j` is reparameterized
            // by its own sum-to-zero null basis `Z_j` (so the constant — i.e.
            // the marginal intercept — is removed from that axis), and the
            // combined reparameterization is the Kronecker product
            // `Z = Z₀ ⊗ Z₁ ⊗ … ⊗ Z_{d-1}`. Applying `Z` to the full-tensor
            // design `B = B₀ ⊗ … ⊗ B_{d-1}` yields `B Z = (B₀ Z₀) ⊗ … ⊗
            // (B_{d-1} Z_{d-1})`, the tensor product of the centered margins,
            // which by construction contains no pure main effect.
            if marginal_designs.len() < 2 {
                crate::bail_invalid_basis!(
                    "tensor interaction (ti) identifiability requires at least 2 margins"
                );
            }
            let mut z = Array2::<f64>::eye(1);
            for (dim, marginal) in marginal_designs.iter().enumerate() {
                if marginal.ncols() < 2 {
                    crate::bail_invalid_basis!(
                        "tensor interaction (ti) margin {dim} has fewer than 2 basis functions; \
                         cannot remove its marginal main effect"
                    );
                }
                let (_, z_dim) = apply_sum_to_zero_constraint(marginal.view(), None)?;
                z = kronecker_product(&z, &z_dim);
            }
            Some(z)
        }
        TensorBSplineIdentifiability::FrozenTransform { transform } => {
            if transform.nrows() != total_cols {
                crate::bail_dim_basis!(
                    "frozen tensor identifiability transform mismatch: design has {} columns but transform has {} rows",
                    total_cols,
                    transform.nrows()
                );
            }
            Some(transform.clone())
        }
    };

    if let Some(z) = z_opt.as_ref() {
        let dense = dense_design.as_mut().ok_or_else(|| {
            BasisError::InvalidInput(
                "tensor identifiability transform requires a realized basis".to_string(),
            )
        })?;
        *dense = dense.dot(z);
        candidates = candidates
            .into_iter()
            .map(|candidate| -> Result<PenaltyCandidate, BasisError> {
                let zt_s = fast_atb(z, &candidate.matrix);
                let matrix = fast_ab(&zt_s, z);
                let (matrix, c_new) = normalize_penalty_in_constrained_space(&matrix);
                let preserve_margin_scale =
                    matches!(&candidate.source, PenaltySource::TensorMarginal { .. });
                let (matrix, normalization_scale) = if preserve_margin_scale {
                    (matrix.mapv(|v| v * c_new), candidate.normalization_scale)
                } else {
                    (matrix, candidate.normalization_scale * c_new)
                };
                Ok(PenaltyCandidate {
                    nullspace_dim_hint: candidate.nullspace_dim_hint,
                    matrix,
                    source: candidate.source,
                    normalization_scale,
                    // Z^T S Z is no longer a Kronecker product of the original
                    // marginal factors, so the Kronecker fast path in construction.rs
                    // must not be taken. Clearing kronecker_factors forces the generic
                    // block-local eigendecomposition path, which operates on the
                    // transformed matrix and is correct.
                    kronecker_factors: None,
                    op: candidate.op.clone(),
                })
            })
            .collect::<Result<Vec<_>, _>>()?;
    }

    let (penalties, nullspace_dims, penaltyinfo, null_eigenvectors, ops) =
        filter_active_penalty_candidates_with_ops(candidates)?;
    let identifiability_is_none =
        matches!(spec.identifiability, TensorBSplineIdentifiability::None);
    // All marginals expose a sparse representation iff each `marginal_sparse`
    // slot is `Some(...)`. Currently this is true when every marginal is a
    // free-boundary, non-periodic 1D B-spline returned as
    // `DesignMatrix::Sparse` from `build_bspline_basis_1d`. Periodic B-splines
    // and other dense-only marginals leave a `None` and trigger the fall-back
    // path. Identifiability transforms (`SumToZero`, `FrozenTransform`) make
    // the tensor design dense in general, so we also gate on that.
    let all_marginals_sparse = marginal_sparse.iter().all(Option::is_some);
    let design = if let Some(dense_design) = dense_design {
        DesignMatrix::Dense(crate::matrix::DenseDesignMatrix::from(dense_design))
    } else if identifiability_is_none && all_marginals_sparse {
        // Sparse Khatri-Rao path: assemble the (n, ∏ q_j) tensor product
        // directly as a SparseColMat, preserving the ∏(degree_j+1) nonzero
        // structure per row instead of densifying to ∏ q_j columns. This is
        // mathematically identical to `tensor_product_design_from_marginals`
        // applied to the corresponding dense marginals.
        let sparse_marginals: Vec<&SparseColMat<usize, f64>> = marginal_sparse
            .iter()
            .map(|m| m.as_ref().expect("all_marginals_sparse just verified"))
            .collect();
        let sparse_design = tensor_product_design_from_sparse_marginals(&sparse_marginals)?;
        DesignMatrix::Sparse(crate::matrix::SparseDesignMatrix::new(sparse_design))
    } else {
        let marginals: Vec<Arc<Array2<f64>>> = marginal_designs
            .iter()
            .map(|m| Arc::new(m.clone()))
            .collect();
        let op = TensorProductDesignOperator::new(marginals).map_err(|e| {
            BasisError::InvalidInput(format!("TensorProductDesignOperator build failed: {e}"))
        })?;
        DesignMatrix::Dense(crate::matrix::DenseDesignMatrix::from(Arc::new(op)))
    };

    Ok(BasisBuildResult {
        design,
        penalties,
        nullspace_dims,
        penaltyinfo,
        ops,
        null_eigenvectors,
        joint_null_rotation: None,
        metadata: BasisMetadata::TensorBSpline {
            feature_cols: feature_cols.to_vec(),
            knots: marginal_knots,
            degrees: marginal_degrees,
            // Prefer the per-margin effective period derived in the loop —
            // it captures both the explicit `spec.periods` route and the
            // implied period from a `PeriodicUniform` marginal knotspec.
            // Falling back to `spec.periods` when populated keeps any
            // user-supplied explicit period authoritative even if the
            // marginal knotspec carried no periodicity hint.
            periods: marginal_effective_periods,
            identifiability_transform: z_opt,
        },
        kronecker_factored: if matches!(spec.identifiability, TensorBSplineIdentifiability::None) {
            Some(KroneckerFactoredBasis {
                marginal_designs,
                marginal_penalties: kronecker_marginal_penalties,
                marginal_dims: marginalnum_basis.clone(),
                has_double_penalty: spec.double_penalty,
            })
        } else {
            None
        },
    })
}

fn tensor_product_design_from_marginals(
    marginal_designs: &[Array2<f64>],
) -> Result<Array2<f64>, BasisError> {
    if marginal_designs.is_empty() {
        crate::bail_invalid_basis!("TensorBSpline requires at least one marginal basis");
    }
    let n = marginal_designs[0].nrows();
    for (i, b) in marginal_designs.iter().enumerate().skip(1) {
        if b.nrows() != n {
            crate::bail_dim_basis!(
                "tensor marginal row mismatch at dim {i}: expected {n}, got {}",
                b.nrows()
            );
        }
    }
    let total_cols = marginal_designs.iter().try_fold(1usize, |acc, b| {
        acc.checked_mul(b.ncols())
            .ok_or_else(|| BasisError::DimensionMismatch("tensor basis too large".to_string()))
    })?;
    // Tensor-product Khatri-Rao: design[i, j] = Π_d marginal_d[i, j_d]
    // where j is the multi-index (j_1, ..., j_D) flattened. Independent
    // across rows; parallelize row chunks and fill the pre-allocated
    // contiguous Array2 in place (no Vec-flatten-collect intermediate,
    // which doubled the peak memory at large-scale N).
    use ndarray::parallel::prelude::*;
    use rayon::iter::{IntoParallelIterator, ParallelIterator};
    let mut design = Array2::<f64>::zeros((n, total_cols));
    design
        .axis_chunks_iter_mut(ndarray::Axis(0), 1024)
        .into_par_iter()
        .enumerate()
        .for_each(|(chunk_idx, mut block)| {
            let row_offset = chunk_idx * 1024;
            // Scratch buffers reused across rows in this chunk.
            let mut cur = Vec::<f64>::with_capacity(total_cols);
            let mut next = Vec::<f64>::with_capacity(total_cols);
            for (local_i, mut out_row) in block.outer_iter_mut().enumerate() {
                let i = row_offset + local_i;
                cur.clear();
                cur.push(1.0);
                for b in marginal_designs {
                    let q = b.ncols();
                    next.clear();
                    next.resize(cur.len() * q, 0.0);
                    // Hoist the row view out of the inner `col` loop so the
                    // q reads per `a_idx` reuse a single contiguous slice
                    // instead of recomputing `b[[i, col]]` strides per cell.
                    let b_row = b.row(i);
                    let b_slice = b_row
                        .as_slice()
                        .expect("Array2 row from outer_iter is contiguous");
                    for (a_idx, &aval) in cur.iter().enumerate() {
                        let off = a_idx * q;
                        let dst = &mut next[off..off + q];
                        for col in 0..q {
                            dst[col] = aval * b_slice[col];
                        }
                    }
                    std::mem::swap(&mut cur, &mut next);
                }
                // `out_row` is a row of the contiguous C-major `design`
                // Array2, so it is backed by a contiguous slice. Use a
                // bulk slice copy instead of an element-by-element write
                // loop.
                let out_slice = out_row
                    .as_slice_mut()
                    .expect("design row is contiguous in C-major Array2");
                out_slice.copy_from_slice(&cur);
            }
        });
    Ok(design)
}

fn build_random_effect_block(
    data: ArrayView2<'_, f64>,
    spec: &RandomEffectTermSpec,
) -> Result<RandomEffectBlock, BasisError> {
    let n = data.nrows();
    let p = data.ncols();
    if spec.feature_col >= p {
        crate::bail_dim_basis!(
            "random-effect term '{}' feature column {} out of bounds for {} columns",
            spec.name,
            spec.feature_col,
            p
        );
    }

    let col = data.column(spec.feature_col);
    if col.iter().any(|v| !v.is_finite()) {
        crate::bail_invalid_basis!(
            "random-effect term '{}' contains non-finite group values",
            spec.name
        );
    }

    let kept_levels: Vec<u64> = if let Some(levels) = spec.frozen_levels.as_ref() {
        if levels.is_empty() {
            crate::bail_invalid_basis!(
                "random-effect term '{}' has empty frozen_levels",
                spec.name
            );
        }
        levels.clone()
    } else {
        let mut levels_set = BTreeSet::<u64>::new();
        for &v in col {
            levels_set.insert(v.to_bits());
        }
        if levels_set.is_empty() {
            crate::bail_invalid_basis!("random-effect term '{}' has no observed levels", spec.name);
        }
        let levels: Vec<u64> = levels_set.into_iter().collect();
        let start_idx = if spec.drop_first_level && levels.len() > 1 {
            1usize
        } else {
            0usize
        };
        levels[start_idx..].to_vec()
    };

    if kept_levels.is_empty() {
        crate::bail_invalid_basis!(
            "random-effect term '{}' drops all levels; keep at least one level",
            spec.name
        );
    }

    let q = kept_levels.len();
    let mut level_to_col = BTreeMap::<u64, usize>::new();
    for (idx, &bits) in kept_levels.iter().enumerate() {
        if level_to_col.insert(bits, idx).is_some() {
            crate::bail_invalid_basis!(
                "random-effect term '{}' has duplicate frozen level bits {bits}",
                spec.name
            );
        }
    }
    let mut group_ids = Vec::with_capacity(n);
    for &v in col {
        let bits = v.to_bits();
        group_ids.push(level_to_col.get(&bits).copied());
    }

    Ok(RandomEffectBlock {
        name: spec.name.clone(),
        group_ids,
        num_groups: q,
        kept_levels,
    })
}

impl SmoothDesign {
    /// Map an unconstrained term coefficient vector to its constrained shape space.
    /// This is useful for nonlinear fits that optimize unconstrained parameters.
    pub fn map_term_coefficients(
        unconstrained: &Array1<f64>,
        shape: ShapeConstraint,
    ) -> Result<Array1<f64>, BasisError> {
        if unconstrained.is_empty() {
            crate::bail_invalid_basis!("unconstrained coefficient vector cannot be empty");
        }
        let mapped = match shape {
            ShapeConstraint::None => unconstrained.clone(),
            ShapeConstraint::MonotoneIncreasing => cumulative_exp(unconstrained, 1.0),
            ShapeConstraint::MonotoneDecreasing => cumulative_exp(unconstrained, -1.0),
            ShapeConstraint::Convex => second_cumulative_exp(unconstrained, 1.0),
            ShapeConstraint::Concave => second_cumulative_exp(unconstrained, -1.0),
        };
        Ok(mapped)
    }
}

struct LocalSmoothTermBuild {
    dim: usize,
    design: DesignMatrix,
    penalties: Vec<Array2<f64>>,
    ops: Vec<Option<std::sync::Arc<dyn crate::terms::analytic_penalties::PenaltyOp>>>,
    nullspaces: Vec<usize>,
    /// Per-active-penalty null-space eigenvector matrices, parallel to
    /// `penalties` / `ops` / `nullspaces`. `Some(U_null)` when
    /// `nullspaces[k] > 0`, with `U_null` orthonormal columns spanning
    /// `null(penalties[k])` in this smooth's local coordinate system; `None`
    /// when the active block is full-rank. Stage 1 plumbing; Stage 2
    /// consumes this to absorb the smooth's null space into the parametric
    /// block at `TermCollectionDesign` construction.
    null_eigenvectors: Vec<Option<Array2<f64>>>,
    /// Joint-null absorption rotation for this smooth. `Some(rotation)`
    /// records `Q = [U_range | U_null]` spanning `null(Σ_k penalties[k])`,
    /// the joint null across all active penalty blocks on this smooth.
    /// `None` means the joint penalty is full-rank (joint nullity = 0) or
    /// there are no penalties. Stage-2 commit A: plumbing only — populated
    /// by commit B, applied by commit D.
    joint_null_rotation: Option<crate::terms::basis::JointNullRotation>,
    penaltyinfo: Vec<PenaltyInfo>,
    pre_dropped_penaltyinfo: Vec<PenaltyInfo>,
    metadata: BasisMetadata,
    linear_constraints: Option<LinearInequalityConstraints>,
    box_reparam: bool,
    kronecker_factored: Option<KroneckerFactoredBasis>,
}

#[derive(Clone)]
struct PcaScoresMemmapDesignOperator {
    mmap: Arc<memmap2::Mmap>,
    data_offset: usize,
    nrows: usize,
    ncols: usize,
    chunk_size: usize,
}

impl PcaScoresMemmapDesignOperator {
    fn open(path: PathBuf, chunk_size: usize) -> Result<Self, BasisError> {
        let file = File::open(&path).map_err(|err| {
            BasisError::InvalidInput(format!(
                "failed to open lazy Pca .npy scores '{}': {err}",
                path.display()
            ))
        })?;
        // The .npy scores file is read-only training-cache data; this
        // module never mutates it. The error path below converts mmap
        // failure to a typed `BasisError::InvalidInput`.
        // SAFETY: `memmap2::Mmap::map` requires no concurrent writers; the
        // contract is held by this module's read-only access pattern.
        let mmap = unsafe {
            memmap2::Mmap::map(&file).map_err(|err| {
                BasisError::InvalidInput(format!(
                    "failed to memmap lazy Pca .npy scores '{}': {err}",
                    path.display()
                ))
            })?
        };
        let (data_offset, nrows, ncols) = parse_f64_2d_npy_header(&mmap, &path)?;
        let expected = data_offset
            .checked_add(nrows.saturating_mul(ncols).saturating_mul(8))
            .ok_or_else(|| {
                BasisError::InvalidInput(format!(
                    "lazy Pca .npy scores '{}' shape is too large",
                    path.display()
                ))
            })?;
        if mmap.len() < expected {
            crate::bail_invalid_basis!(
                "lazy Pca .npy scores '{}' is truncated: header expects {} bytes, file has {}",
                path.display(),
                expected,
                mmap.len()
            );
        }
        Ok(Self {
            mmap: Arc::new(mmap),
            data_offset,
            nrows,
            ncols,
            chunk_size: chunk_size.max(1),
        })
    }

    fn value(&self, row: usize, col: usize) -> f64 {
        let offset = self.data_offset + (row * self.ncols + col) * 8;
        let mut bytes = [0_u8; 8];
        bytes.copy_from_slice(&self.mmap[offset..offset + 8]);
        f64::from_le_bytes(bytes)
    }

    fn chunk_rows(&self) -> usize {
        self.chunk_size.min(self.nrows.max(1))
    }
}

impl LinearOperator for PcaScoresMemmapDesignOperator {
    fn nrows(&self) -> usize {
        self.nrows
    }

    fn ncols(&self) -> usize {
        self.ncols
    }

    fn apply(&self, vector: &Array1<f64>) -> Array1<f64> {
        assert_eq!(
            vector.len(),
            self.ncols,
            "lazy Pca apply vector length mismatch"
        );
        let mut out = Array1::<f64>::zeros(self.nrows);
        for start in (0..self.nrows).step_by(self.chunk_rows()) {
            let end = (start + self.chunk_rows()).min(self.nrows);
            for row in start..end {
                let mut acc = 0.0;
                for col in 0..self.ncols {
                    acc += self.value(row, col) * vector[col];
                }
                out[row] = acc;
            }
        }
        out
    }

    fn apply_transpose(&self, vector: &Array1<f64>) -> Array1<f64> {
        assert_eq!(
            vector.len(),
            self.nrows,
            "lazy Pca apply_transpose vector length mismatch"
        );
        let mut out = Array1::<f64>::zeros(self.ncols);
        for start in (0..self.nrows).step_by(self.chunk_rows()) {
            let end = (start + self.chunk_rows()).min(self.nrows);
            for row in start..end {
                let scale = vector[row];
                if scale == 0.0 {
                    continue;
                }
                for col in 0..self.ncols {
                    out[col] += scale * self.value(row, col);
                }
            }
        }
        out
    }

    fn diag_xtw_x(&self, weights: &Array1<f64>) -> Result<Array2<f64>, String> {
        if weights.len() != self.nrows {
            return Err(format!(
                "lazy Pca diag_xtw_x weight length mismatch: weights={}, nrows={}",
                weights.len(),
                self.nrows
            ));
        }
        let mut gram = Array2::<f64>::zeros((self.ncols, self.ncols));
        for start in (0..self.nrows).step_by(self.chunk_rows()) {
            let end = (start + self.chunk_rows()).min(self.nrows);
            for row in start..end {
                let w = weights[row];
                if w == 0.0 {
                    continue;
                }
                for a in 0..self.ncols {
                    let xa = self.value(row, a);
                    if xa == 0.0 {
                        continue;
                    }
                    for b in a..self.ncols {
                        gram[[a, b]] += w * xa * self.value(row, b);
                    }
                }
            }
        }
        for a in 0..self.ncols {
            for b in 0..a {
                gram[[a, b]] = gram[[b, a]];
            }
        }
        Ok(gram)
    }

    fn apply_weighted_normal(
        &self,
        weights: &Array1<f64>,
        vector: &Array1<f64>,
        penalty: Option<&Array2<f64>>,
        ridge: f64,
    ) -> Array1<f64> {
        assert_eq!(
            weights.len(),
            self.nrows,
            "lazy Pca weighted-normal weight mismatch"
        );
        assert_eq!(
            vector.len(),
            self.ncols,
            "lazy Pca weighted-normal vector mismatch"
        );
        let mut out = Array1::<f64>::zeros(self.ncols);
        for start in (0..self.nrows).step_by(self.chunk_rows()) {
            let end = (start + self.chunk_rows()).min(self.nrows);
            for row in start..end {
                let w = weights[row].max(0.0);
                if w == 0.0 {
                    continue;
                }
                let mut row_dot = 0.0;
                for col in 0..self.ncols {
                    row_dot += self.value(row, col) * vector[col];
                }
                if row_dot == 0.0 {
                    continue;
                }
                let scaled = w * row_dot;
                for col in 0..self.ncols {
                    out[col] += scaled * self.value(row, col);
                }
            }
        }
        if let Some(pen) = penalty {
            out += &pen.dot(vector);
        }
        if ridge > 0.0 {
            out += &vector.mapv(|x| ridge * x);
        }
        out
    }
}

impl DenseDesignOperator for PcaScoresMemmapDesignOperator {
    fn compute_xtwy(&self, weights: &Array1<f64>, y: &Array1<f64>) -> Result<Array1<f64>, String> {
        if weights.len() != self.nrows || y.len() != self.nrows {
            return Err(format!(
                "lazy Pca compute_xtwy dimension mismatch: weights={}, y={}, nrows={}",
                weights.len(),
                y.len(),
                self.nrows
            ));
        }
        let mut out = Array1::<f64>::zeros(self.ncols);
        for start in (0..self.nrows).step_by(self.chunk_rows()) {
            let end = (start + self.chunk_rows()).min(self.nrows);
            for row in start..end {
                let scale = weights[row] * y[row];
                if scale == 0.0 {
                    continue;
                }
                for col in 0..self.ncols {
                    out[col] += scale * self.value(row, col);
                }
            }
        }
        Ok(out)
    }

    fn row_chunk_into(
        &self,
        rows: Range<usize>,
        mut out: ArrayViewMut2<'_, f64>,
    ) -> Result<(), MatrixMaterializationError> {
        if rows.end > self.nrows || rows.start > rows.end {
            return Err(MatrixMaterializationError::MissingRowChunk {
                context: "lazy Pca row range out of bounds",
            });
        }
        if out.nrows() != rows.end - rows.start || out.ncols() != self.ncols {
            return Err(MatrixMaterializationError::MissingRowChunk {
                context: "lazy Pca row_chunk_into shape mismatch",
            });
        }
        for (local, row) in (rows.start..rows.end).enumerate() {
            for col in 0..self.ncols {
                out[[local, col]] = self.value(row, col);
            }
        }
        Ok(())
    }

    fn to_dense(&self) -> Array2<f64> {
        let mut out = Array2::<f64>::zeros((self.nrows, self.ncols));
        self.row_chunk_into(0..self.nrows, out.view_mut())
            .expect("lazy Pca full materialization failed");
        out
    }
}

fn parse_f64_2d_npy_header(
    bytes: &[u8],
    path: &PathBuf,
) -> Result<(usize, usize, usize), BasisError> {
    if bytes.len() < 10 || &bytes[0..6] != b"\x93NUMPY" {
        crate::bail_invalid_basis!("lazy Pca scores '{}' is not a .npy file", path.display());
    }
    let major = bytes[6];
    let header_len = match major {
        1 => u16::from_le_bytes([bytes[8], bytes[9]]) as usize,
        2 | 3 => {
            if bytes.len() < 12 {
                crate::bail_invalid_basis!(
                    "lazy Pca scores '{}' has a truncated .npy header",
                    path.display()
                );
            }
            u32::from_le_bytes([bytes[8], bytes[9], bytes[10], bytes[11]]) as usize
        }
        other => {
            crate::bail_invalid_basis!(
                "lazy Pca scores '{}' uses unsupported .npy version {}",
                path.display(),
                other
            );
        }
    };
    let header_start = if major == 1 { 10 } else { 12 };
    let data_offset = header_start + header_len;
    if bytes.len() < data_offset {
        crate::bail_invalid_basis!(
            "lazy Pca scores '{}' has a truncated .npy header",
            path.display()
        );
    }
    let header = std::str::from_utf8(&bytes[header_start..data_offset]).map_err(|err| {
        BasisError::InvalidInput(format!(
            "lazy Pca scores '{}' has a non-UTF8 .npy header: {err}",
            path.display()
        ))
    })?;
    if !(header.contains("'descr': '<f8'")
        || header.contains("\"descr\": \"<f8\"")
        || header.contains("'descr': '|f8'")
        || header.contains("\"descr\": \"|f8\""))
    {
        crate::bail_invalid_basis!(
            "lazy Pca scores '{}' must be float64 little-endian .npy",
            path.display()
        );
    }
    if header.contains("True") {
        crate::bail_invalid_basis!(
            "lazy Pca scores '{}' must be C-contiguous, not Fortran-ordered",
            path.display()
        );
    }
    let shape_pos = header.find("shape").ok_or_else(|| {
        BasisError::InvalidInput(format!(
            "lazy Pca scores '{}' .npy header is missing shape",
            path.display()
        ))
    })?;
    let open = header[shape_pos..].find('(').ok_or_else(|| {
        BasisError::InvalidInput(format!(
            "lazy Pca scores '{}' .npy header has malformed shape",
            path.display()
        ))
    })? + shape_pos;
    let close = header[open..].find(')').ok_or_else(|| {
        BasisError::InvalidInput(format!(
            "lazy Pca scores '{}' .npy header has malformed shape",
            path.display()
        ))
    })? + open;
    let dims = header[open + 1..close]
        .split(',')
        .map(str::trim)
        .filter(|part| !part.is_empty())
        .map(|part| part.parse::<usize>())
        .collect::<Result<Vec<_>, _>>()
        .map_err(|err| {
            BasisError::InvalidInput(format!(
                "lazy Pca scores '{}' .npy shape is not integral: {err}",
                path.display()
            ))
        })?;
    if dims.len() != 2 {
        crate::bail_invalid_basis!(
            "lazy Pca scores '{}' must have shape (N, K), got {:?}",
            path.display(),
            dims
        );
    }
    Ok((data_offset, dims[0], dims[1]))
}

fn pca_center_mean(x: ArrayView2<'_, f64>) -> Result<Array1<f64>, BasisError> {
    if x.nrows() == 0 {
        crate::bail_invalid_basis!("Pca basis requires at least one row to compute center mean");
    }
    let mut mean = Array1::<f64>::zeros(x.ncols());
    for row in x.rows() {
        mean += &row;
    }
    mean.mapv_inplace(|v| v / x.nrows() as f64);
    Ok(mean)
}

fn build_pca_smooth_basis(
    data: ArrayView2<'_, f64>,
    feature_cols: &[usize],
    basis_matrix: &Array2<f64>,
    centered: bool,
    smooth_penalty: f64,
    center_mean: Option<&Array1<f64>>,
    pca_basis_path: Option<&PathBuf>,
    chunk_size: usize,
) -> Result<BasisBuildResult, BasisError> {
    if let Some(path) = pca_basis_path {
        let op = PcaScoresMemmapDesignOperator::open(path.clone(), chunk_size)?;
        if op.nrows != data.nrows() {
            crate::bail_dim_basis!(
                "lazy Pca scores row mismatch: .npy has {}, data has {}",
                op.nrows,
                data.nrows()
            );
        }
        let k = op.ncols;
        let mut penalty = Array2::<f64>::eye(k);
        penalty.mapv_inplace(|v| v * smooth_penalty);
        let (penalties, nullspace_dims, penaltyinfo, null_eigenvectors, ops) =
            filter_active_penalty_candidates_with_ops(vec![PenaltyCandidate {
                matrix: penalty,
                nullspace_dim_hint: 0,
                source: PenaltySource::Other("PcaRidge".to_string()),
                normalization_scale: 1.0,
                kronecker_factors: None,
                op: None,
            }])?;
        return Ok(BasisBuildResult {
            design: DesignMatrix::Dense(crate::matrix::DenseDesignMatrix::from(Arc::new(op))),
            penalties,
            nullspace_dims,
            penaltyinfo,
            ops,
            null_eigenvectors,
            joint_null_rotation: None,
            metadata: BasisMetadata::Pca {
                feature_cols: feature_cols.to_vec(),
                basis_matrix: basis_matrix.clone(),
                centered,
                smooth_penalty,
                center_mean: center_mean.cloned(),
                pca_basis_path: Some(path.clone()),
                chunk_size: chunk_size.max(1),
            },
            kronecker_factored: None,
        });
    }
    if basis_matrix.nrows() != feature_cols.len() {
        crate::bail_dim_basis!(
            "Pca basis row mismatch: basis rows={}, feature columns={}",
            basis_matrix.nrows(),
            feature_cols.len()
        );
    }
    let mut x = select_columns(data, feature_cols)?;
    let mean = if centered {
        match center_mean {
            Some(mean) => mean.clone(),
            None => pca_center_mean(x.view())?,
        }
    } else {
        Array1::<f64>::zeros(feature_cols.len())
    };
    if centered {
        for mut row in x.rows_mut() {
            row -= &mean;
        }
    }
    let design = fast_ab(&x, basis_matrix);
    let k = basis_matrix.ncols();
    let mut penalty = Array2::<f64>::eye(k);
    penalty.mapv_inplace(|v| v * smooth_penalty);
    let (penalties, nullspace_dims, penaltyinfo, null_eigenvectors, ops) =
        filter_active_penalty_candidates_with_ops(vec![PenaltyCandidate {
            matrix: penalty,
            nullspace_dim_hint: 0,
            source: PenaltySource::Other("PcaRidge".to_string()),
            normalization_scale: 1.0,
            kronecker_factors: None,
            op: None,
        }])?;
    Ok(BasisBuildResult {
        design: DesignMatrix::Dense(crate::matrix::DenseDesignMatrix::from(design)),
        penalties,
        nullspace_dims,
        penaltyinfo,
        ops,
        null_eigenvectors,
        joint_null_rotation: None,
        metadata: BasisMetadata::Pca {
            feature_cols: feature_cols.to_vec(),
            basis_matrix: basis_matrix.clone(),
            centered,
            smooth_penalty,
            center_mean: centered.then_some(mean),
            pca_basis_path: None,
            chunk_size: chunk_size.max(1),
        },
        kronecker_factored: None,
    })
}

fn apply_by_variable_to_local_build(
    mut built: LocalSmoothTermBuild,
    data: ArrayView2<'_, f64>,
    by_col: usize,
    by: &ByVariableSpec,
    term_name: &str,
) -> Result<LocalSmoothTermBuild, BasisError> {
    if by_col >= data.ncols() {
        crate::bail_dim_basis!(
            "by-variable smooth term '{term_name}' references column {by_col}, but data has {} columns",
            data.ncols()
        );
    }
    let weights = match by {
        ByVariableSpec::Numeric => data.column(by_col).to_owned(),
        ByVariableSpec::Level { value_bits, .. } => data.column(by_col).mapv(|value| {
            if value.to_bits() == *value_bits {
                1.0
            } else {
                0.0
            }
        }),
    };
    if weights.iter().any(|value| !value.is_finite()) {
        crate::bail_invalid_basis!(
            "by-variable smooth term '{term_name}' has non-finite by-column values"
        );
    }

    let mut dense = built
        .design
        .try_to_dense_by_chunks("by-variable smooth row gating")
        .map_err(BasisError::InvalidInput)?;
    for (mut row, &weight) in dense.rows_mut().into_iter().zip(weights.iter()) {
        row.mapv_inplace(|value| value * weight);
    }
    built.design = DesignMatrix::Dense(crate::matrix::DenseDesignMatrix::from(dense));
    built.kronecker_factored = None;
    Ok(built)
}

fn ensure_by_variable_specs_match(
    kind: &BySmoothKind,
    by: &ByVariableSpec,
    term_name: &str,
) -> Result<(), BasisError> {
    match (kind, by) {
        (BySmoothKind::Numeric, ByVariableSpec::Numeric) => Ok(()),
        (BySmoothKind::Level { level_bits }, ByVariableSpec::Level { value_bits, .. })
            if level_bits == value_bits =>
        {
            Ok(())
        }
        _ => Err(BasisError::InvalidInput(format!(
            "by-variable smooth term '{term_name}' has inconsistent by-variable specifications"
        ))),
    }
}

/// Build a factor-smooth interaction basis (`bs="fs"`/`"sz"`/`"re"`).
///
/// A factor smooth replicates a shared marginal smooth in the continuous
/// covariate(s) once per level of a grouping factor, coupling all level blocks
/// through a *single* set of smoothing parameters (one per marginal penalty).
/// This is mgcv's `smooth.construct.fs.smooth.spec` realization and the
/// random-effect interpretation of a smooth: the per-level deviations are an
/// exchangeable family whose joint wiggliness/shrinkage is governed by the
/// shared λ, so the construction scales to many levels with a fixed parameter
/// count.
///
/// Flavours:
/// * `Fs` — full random factor-smooth. The marginal carries its wiggliness
///   penalty *and* a null-space ridge (double penalty), so the replicated
///   design is a proper full-rank random effect: each level's curve is shrunk
///   toward zero (intercept + linear trend included), recovering the mgcv
///   `bs="fs"` penalty structure `I_L ⊗ S_j` for every marginal penalty `S_j`.
/// * `Sz` — sum-to-zero factor smooth. Delegates to the existing
///   [`SmoothBasisSpec::FactorSumToZero`] construction (`L-1` deviation blocks,
///   coefficient-wise zero sum across levels).
/// * `Re` — pure random effect / random slope (`bs="re"`). A degree-1 marginal
///   gives the per-level `[1, x]` span; the penalty is the identity over each
///   level block (iid Gaussian coefficients), matching mgcv's `bs="re"` ridge.
///
/// The grouping levels are resolved once at fit time (sorted unique bit
/// patterns of the factor column) and frozen into the returned metadata so the
/// predict-time rebuild evaluates every row against its own level's block.
fn build_factor_smooth(
    data: ArrayView2<'_, f64>,
    spec: &FactorSmoothSpec,
    term_name: &str,
    workspace: &mut crate::basis::BasisWorkspace,
) -> Result<LocalSmoothTermBuild, BasisError> {
    if spec.continuous_cols.len() != 1 {
        crate::bail_invalid_basis!(
            "factor smooth term '{}' currently supports exactly one continuous covariate; found {}",
            term_name,
            spec.continuous_cols.len()
        );
    }
    let feature_col = spec.continuous_cols[0];
    let group_col = spec.group_col;
    if feature_col >= data.ncols() || group_col >= data.ncols() {
        crate::bail_dim_basis!(
            "factor smooth term '{}' references columns ({}, {}) out of bounds for {} columns",
            term_name,
            feature_col,
            group_col,
            data.ncols()
        );
    }

    // `Sz` is exactly the existing sum-to-zero factor smooth: reuse it verbatim
    // so there is a single source of truth for the zero-sum construction.
    if matches!(spec.flavour, FactorSmoothFlavour::Sz) {
        let levels = resolve_factor_smooth_levels(data, group_col, spec, term_name)?;
        let inner = SmoothBasisSpec::BSpline1D {
            feature_col,
            spec: factor_smooth_marginal_for_replay(&spec.marginal),
        };
        let sz_term = SmoothTermSpec {
            name: term_name.to_string(),
            basis: SmoothBasisSpec::FactorSumToZero {
                inner: Box::new(inner),
                by_col: group_col,
                levels,
                frozen_global_orthogonality: None,
            },
            shape: ShapeConstraint::None,
            joint_null_rotation: None,
        };
        return build_single_local_smooth_term(data, &sz_term, workspace);
    }

    let levels = resolve_factor_smooth_levels(data, group_col, spec, term_name)?;
    let n_levels = levels.len();
    if n_levels < 2 {
        crate::bail_invalid_basis!(
            "factor smooth term '{}' requires at least two grouping levels; found {}",
            term_name,
            n_levels
        );
    }

    // `Fs` (order ≥ 1, the default) is the random-effect flavour: it penalizes
    // each null-space dimension of the marginal wiggliness penalty separately
    // below (mgcv's `bs="fs"` construction). That replaces the marginal's single
    // *combined* double penalty, so disable the latter here to avoid penalizing
    // the null space twice (once combined, once per dimension). The explicit
    // `m=0` opt-out keeps the legacy combined double penalty and adds no
    // per-dimension penalties.
    let use_per_dim_null = matches!(
        &spec.flavour,
        FactorSmoothFlavour::Fs { m_null_penalty_orders }
            if m_null_penalty_orders.iter().copied().max().unwrap_or(0) >= 1
    );

    // Build the shared marginal design + penalties from the 1-D B-spline.
    // `Re` forces a degree-1 marginal (linear span) and replaces the marginal
    // wiggliness with an identity ridge below; `Fs` keeps the user's marginal
    // (cubic by default) and, under the per-dimension null path, gets its null
    // space penalized one dimension at a time after replication.
    let mut marginal_spec = factor_smooth_marginal_for_replay(&spec.marginal);
    if use_per_dim_null {
        marginal_spec.double_penalty = false;
    }
    let inner_term = SmoothTermSpec {
        name: format!("{term_name}::marginal"),
        basis: SmoothBasisSpec::BSpline1D {
            feature_col,
            spec: marginal_spec,
        },
        shape: ShapeConstraint::None,
        joint_null_rotation: None,
    };
    let inner = build_single_local_smooth_term(data, &inner_term, workspace)?;
    let base = inner
        .design
        .try_to_dense_by_chunks("factor smooth marginal")
        .map_err(BasisError::InvalidInput)?;
    let n = base.nrows();
    let p = base.ncols();
    let q = p * n_levels;

    // Block-diagonal replicated design: row i contributes its marginal row to
    // the column block owned by its grouping level, zeros elsewhere.
    let mut dense = Array2::<f64>::zeros((n, q));
    for i in 0..n {
        let bits = data[[i, group_col]].to_bits();
        let level_idx = levels.iter().position(|b| *b == bits).ok_or_else(|| {
            BasisError::InvalidInput(format!(
                "factor smooth term '{term_name}' saw an unseen grouping level at row {}",
                i + 1
            ))
        })?;
        let start = level_idx * p;
        dense
            .slice_mut(s![i, start..start + p])
            .assign(&base.row(i));
    }

    // Penalties: replicate each marginal penalty into a block-diagonal
    // `I_L ⊗ S_j` so every level shares the same smoothing parameter λ_j (one
    // λ per marginal penalty), the defining feature of a factor smooth. For
    // `Re` the marginal penalty is replaced by an identity ridge so each
    // per-level coefficient is an iid Gaussian random effect.
    let marginal_penalties: Vec<Array2<f64>> = if matches!(spec.flavour, FactorSmoothFlavour::Re) {
        vec![Array2::<f64>::eye(p)]
    } else {
        inner.penalties.clone()
    };
    let marginal_penaltyinfo: Vec<PenaltyInfo> = if matches!(spec.flavour, FactorSmoothFlavour::Re)
    {
        vec![PenaltyInfo {
            source: PenaltySource::Primary,
            original_index: 0,
            active: true,
            effective_rank: p,
            dropped_reason: None,
            nullspace_dim_hint: 0,
            normalization_scale: 1.0,
            kronecker_factors: None,
        }]
    } else {
        inner.penaltyinfo.clone()
    };
    if marginal_penalties.len() != marginal_penaltyinfo.len() {
        crate::bail_invalid_basis!(
            "internal factor-smooth penalty metadata mismatch for term '{}': penalties={}, infos={}",
            term_name,
            marginal_penalties.len(),
            marginal_penaltyinfo.len()
        );
    }

    let mut penalties = Vec::<Array2<f64>>::with_capacity(marginal_penalties.len());
    let mut penaltyinfo = Vec::<PenaltyInfo>::with_capacity(marginal_penalties.len());
    for (penalty_pos, s_inner) in marginal_penalties.iter().enumerate() {
        let mut s_big = Array2::<f64>::zeros((q, q));
        for level in 0..n_levels {
            let start = level * p;
            s_big
                .slice_mut(s![start..start + p, start..start + p])
                .assign(s_inner);
        }
        let (s_big, factor_smooth_scale) = normalize_penalty_in_constrained_space(&s_big);
        let mut info = marginal_penaltyinfo[penalty_pos].clone();
        info.original_index = penalty_pos;
        info.normalization_scale *= factor_smooth_scale;
        info.nullspace_dim_hint = info.nullspace_dim_hint.saturating_mul(n_levels);
        info.kronecker_factors = None;
        penalties.push(s_big);
        penaltyinfo.push(info);
    }

    let mut nullspaces: Vec<usize> = if matches!(spec.flavour, FactorSmoothFlavour::Re) {
        vec![0]
    } else {
        inner
            .nullspaces
            .iter()
            .map(|ns| ns.saturating_mul(n_levels))
            .collect()
    };

    // `Fs` is the random-effect flavour of a smooth: the per-group curve is an
    // exchangeable Gaussian *function*, so EVERY coefficient — including the
    // {const, linear} null space of the marginal wiggliness penalty — must be
    // shrinkable toward zero under its own shared variance. The wiggliness
    // penalty `S_wiggle` shapes curvature but leaves the per-group intercept and
    // slope (its null space) completely UNPENALIZED. With the null space free,
    // each group fits its own intercept and slope with NO partial pooling, so
    // the held-out per-subject forecast inherits the full no-pooling variance
    // and curves away from the true per-group line (gam#712 real arm, gam#713;
    // gam#903 sleepstudy forecast ran ~74% over the lme4 BLUP bar).
    //
    // mgcv's `bs="fs"` fixes this by penalizing each null-space dimension
    // SEPARATELY (`smooth.construct.fs.smooth.spec` adds one rank-1 penalty per
    // null coordinate), each replicated block-diagonally across levels under a
    // single shared smoothing parameter — so REML fits a distinct
    // random-intercept variance and random-slope variance, the partial pooling
    // that makes the forecast track lme4's correlated random-effect BLUP. A
    // single *combined* null penalty (one λ for intercept+slope together) cannot
    // express the typically very different intercept and slope variances, which
    // is the residual forecast gap. We mirror mgcv exactly: for each orthonormal
    // null direction `z_k` of the marginal wiggliness penalty add
    // `I_L ⊗ (z_k z_kᵀ)` as its own penalty. The marginal's combined double
    // penalty was disabled above, so the null space is penalized once, per
    // dimension. With linear data REML drives the curvature λ up and degrades
    // `fs` to a linear random slope (edf → ≈2/group); with genuine curvature the
    // wiggliness λ stays small and the wiggle survives (data-adaptive, not a
    // cap). Gated by `m_null_penalty_orders`: order ≥ 1 (default) enables the
    // per-dimension null penalties; `m=0` keeps the legacy combined double
    // penalty and adds nothing here.
    if use_per_dim_null
        && let Some(Some(z)) = inner.null_eigenvectors.first()
        && z.nrows() == p
    {
        for k in 0..z.ncols() {
            // Rank-1 marginal penalty `z_k z_kᵀ`, replicated block-diagonally
            // across levels into `I_L ⊗ (z_k z_kᵀ)`. Its own λ is one shared
            // variance for this null component (intercept or slope) across all
            // groups — the random-effect structure of mgcv `fs`.
            let zk = z.column(k);
            let mut p_k = Array2::<f64>::zeros((p, p));
            for a in 0..p {
                for b in 0..p {
                    p_k[[a, b]] = zk[a] * zk[b];
                }
            }
            let mut s_null = Array2::<f64>::zeros((q, q));
            for level in 0..n_levels {
                let start = level * p;
                s_null
                    .slice_mut(s![start..start + p, start..start + p])
                    .assign(&p_k);
            }
            let (s_null, null_scale) = normalize_penalty_in_constrained_space(&s_null);
            let null_block = crate::terms::basis::analyze_penalty_block_with_op(&s_null, None)?;
            if null_block.rank > 0 {
                let original_index = penalties.len();
                penalties.push(null_block.sym_penalty);
                nullspaces.push(null_block.nullity);
                penaltyinfo.push(PenaltyInfo {
                    source: PenaltySource::Primary,
                    original_index,
                    active: true,
                    effective_rank: null_block.rank,
                    dropped_reason: None,
                    nullspace_dim_hint: null_block.nullity,
                    normalization_scale: null_scale,
                    kronecker_factors: None,
                });
            }
        }
    }
    let null_eigenvectors = crate::terms::basis::recompute_null_eigenvectors(&penalties)?;
    let joint_null_rotation = crate::terms::basis::compute_joint_null_rotation(&penalties)?;

    // Metadata: carry the marginal knot geometry + frozen levels so prediction
    // reconstructs an identical replicated design.
    let (knots, degree, periodic) = match &inner.metadata {
        BasisMetadata::BSpline1D {
            knots,
            periodic,
            degree,
            ..
        } => (
            knots.clone(),
            degree.unwrap_or(spec.marginal.degree),
            *periodic,
        ),
        other => {
            crate::bail_invalid_basis!(
                "factor smooth term '{}' produced an unexpected marginal metadata variant {:?}",
                term_name,
                other
            );
        }
    };
    let flavour_tag = match &spec.flavour {
        FactorSmoothFlavour::Fs { .. } => "fs",
        FactorSmoothFlavour::Sz => "sz",
        FactorSmoothFlavour::Re => "re",
    }
    .to_string();
    let metadata = BasisMetadata::FactorSmooth {
        continuous_cols: spec.continuous_cols.clone(),
        group_col,
        knots,
        degree,
        periodic,
        group_levels: levels,
        flavour: flavour_tag,
    };

    let ops = vec![None; penalties.len()];
    Ok(LocalSmoothTermBuild {
        dim: q,
        design: DesignMatrix::Dense(crate::matrix::DenseDesignMatrix::from(dense)),
        penalties,
        ops,
        nullspaces,
        null_eigenvectors,
        joint_null_rotation,
        penaltyinfo,
        pre_dropped_penaltyinfo: Vec::new(),
        metadata,
        linear_constraints: None,
        box_reparam: false,
        kronecker_factored: None,
    })
}

/// Resolve the grouping levels for a factor smooth: replay the frozen level
/// list when present (predict path), otherwise discover the sorted unique bit
/// patterns of the factor column (fit path).
fn resolve_factor_smooth_levels(
    data: ArrayView2<'_, f64>,
    group_col: usize,
    spec: &FactorSmoothSpec,
    term_name: &str,
) -> Result<Vec<u64>, BasisError> {
    if let Some(frozen) = &spec.group_frozen_levels {
        if frozen.is_empty() {
            crate::bail_invalid_basis!(
                "factor smooth term '{}' has an empty frozen level list",
                term_name
            );
        }
        return Ok(frozen.clone());
    }
    let mut bits: Vec<u64> = data.column(group_col).iter().map(|v| v.to_bits()).collect();
    bits.sort_by(|a, b| {
        f64::from_bits(*a)
            .partial_cmp(&f64::from_bits(*b))
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    bits.dedup();
    Ok(bits)
}

/// Marginal B-spline spec for a factor-smooth block. The marginal always builds
/// without an identifiability constraint (the per-level replication, not a
/// sum-to-zero side constraint, provides identifiability against the parametric
/// block). At predict time the marginal's knot geometry has already been pinned
/// into `marginal.knotspec` by the metadata replay, so the spec is used
/// verbatim aside from clearing the identifiability transform.
fn factor_smooth_marginal_for_replay(marginal: &BSplineBasisSpec) -> BSplineBasisSpec {
    let mut m = marginal.clone();
    m.identifiability = BSplineIdentifiability::None;
    m
}

fn build_single_local_smooth_term(
    data: ArrayView2<'_, f64>,
    term: &SmoothTermSpec,
    workspace: &mut crate::basis::BasisWorkspace,
) -> Result<LocalSmoothTermBuild, BasisError> {
    if term.shape != ShapeConstraint::None && !shape_supports_basis(term) {
        crate::bail_invalid_basis!(
            "ShapeConstraint::{:?} is unsupported for term '{}'",
            term.shape,
            term.name
        );
    }
    if let SmoothBasisSpec::ByVariable {
        inner,
        by_col,
        kind,
        by,
    } = &term.basis
    {
        ensure_by_variable_specs_match(kind, by, &term.name)?;
        let inner_term = SmoothTermSpec {
            name: term.name.clone(),
            basis: (**inner).clone(),
            shape: term.shape,
            joint_null_rotation: None,
        };
        let built = build_single_local_smooth_term(data, &inner_term, workspace)?;
        return apply_by_variable_to_local_build(built, data, *by_col, by, &term.name);
    }

    let mut shape_axis_col: Option<usize> = None;
    let mut built: BasisBuildResult = match &term.basis {
        SmoothBasisSpec::FactorSumToZero {
            inner,
            by_col,
            levels,
            ..
        } => {
            if *by_col >= data.ncols() {
                crate::bail_dim_basis!(
                    "term '{}' by column {} out of bounds for {} columns",
                    term.name,
                    by_col,
                    data.ncols()
                );
            }
            if levels.len() < 2 {
                crate::bail_invalid_basis!(
                    "sum-to-zero factor smooth term '{}' requires at least two levels",
                    term.name
                );
            }
            if term.shape != ShapeConstraint::None {
                crate::bail_invalid_basis!(
                    "ShapeConstraint::{:?} is unsupported for sum-to-zero factor smooth term '{}'",
                    term.shape,
                    term.name
                );
            }
            let inner_term = SmoothTermSpec {
                name: format!("{}::inner", term.name),
                basis: (**inner).clone(),
                shape: ShapeConstraint::None,
                joint_null_rotation: None,
            };
            let mut inner_built = build_single_local_smooth_term(data, &inner_term, workspace)?;
            let base = inner_built
                .design
                .try_to_dense_by_chunks("sum-to-zero factor smooth")
                .map_err(BasisError::InvalidInput)?;
            let n = base.nrows();
            let p = base.ncols();
            let l_minus_one = levels.len() - 1;
            let mut dense = Array2::<f64>::zeros((n, p * l_minus_one));
            for i in 0..n {
                let bits = data[[i, *by_col]].to_bits();
                let level_idx = levels.iter().position(|b| *b == bits).ok_or_else(|| {
                    BasisError::InvalidInput(format!(
                        "sum-to-zero factor smooth term '{}' saw an unseen level at row {}",
                        term.name,
                        i + 1
                    ))
                })?;
                if level_idx < l_minus_one {
                    let start = level_idx * p;
                    dense
                        .slice_mut(s![i, start..start + p])
                        .assign(&base.row(i));
                } else {
                    for level in 0..l_minus_one {
                        let start = level * p;
                        dense
                            .slice_mut(s![i, start..start + p])
                            .assign(&base.row(i).mapv(|v| -v));
                    }
                }
            }
            let mut penalties = Vec::<Array2<f64>>::with_capacity(inner_built.penalties.len());
            let active_penalty_indices = inner_built
                .penaltyinfo
                .iter()
                .enumerate()
                .filter_map(|(idx, info)| info.active.then_some(idx))
                .collect::<Vec<_>>();
            if active_penalty_indices.len() != inner_built.penalties.len() {
                crate::bail_invalid_basis!(
                    "internal sz penalty metadata mismatch: activeinfos={}, penalties={}",
                    active_penalty_indices.len(),
                    inner_built.penalties.len()
                );
            }
            for (penalty_pos, s_inner) in inner_built.penalties.iter().enumerate() {
                let mut s_big = Array2::<f64>::zeros((p * l_minus_one, p * l_minus_one));
                for a in 0..l_minus_one {
                    for b in 0..l_minus_one {
                        let factor = if a == b { 2.0 } else { 1.0 };
                        let mut block = s_big.slice_mut(s![a * p..(a + 1) * p, b * p..(b + 1) * p]);
                        block.assign(&s_inner.mapv(|v| v * factor));
                    }
                }
                let (s_big, factor_smooth_scale) = normalize_penalty_in_constrained_space(&s_big);
                let info_idx = active_penalty_indices[penalty_pos];
                inner_built.penaltyinfo[info_idx].normalization_scale *= factor_smooth_scale;
                penalties.push(s_big);
            }
            inner_built.dim = p * l_minus_one;
            inner_built.design = DesignMatrix::Dense(crate::matrix::DenseDesignMatrix::from(dense));
            inner_built.penalties = penalties;
            inner_built.ops = vec![None; inner_built.penalties.len()];
            inner_built.nullspaces = inner_built
                .nullspaces
                .iter()
                .map(|ns| ns.saturating_mul(l_minus_one))
                .collect();
            // Invariant: `null_eigenvectors[k]` must mirror `penalties[k]`'s
            // spectral null space. We just rebuilt `inner_built.penalties` from
            // Kronecker-like `S_big` blocks, so the previously-plumbed
            // `null_eigenvectors` (still parallel to the OLD per-level penalty)
            // is stale. Recompute from the rebuilt penalties to restore the
            // invariant; ditto for the joint-null absorption rotation.
            inner_built.null_eigenvectors =
                crate::terms::basis::recompute_null_eigenvectors(&inner_built.penalties)?;
            inner_built.joint_null_rotation =
                crate::terms::basis::compute_joint_null_rotation(&inner_built.penalties)?;
            inner_built.kronecker_factored = None;
            return Ok(inner_built);
        }
        SmoothBasisSpec::BSpline1D { feature_col, spec } => {
            if *feature_col >= data.ncols() {
                crate::bail_dim_basis!(
                    "term '{}' feature column {} out of bounds for {} columns",
                    term.name,
                    feature_col,
                    data.ncols()
                );
            }
            let mut spec_local = spec.clone();
            if term.shape != ShapeConstraint::None {
                // Shape-constrained B-splines are anchored by construction.
                // Sum-to-zero side constraints conflict with monotonic/convex cones.
                spec_local.identifiability = BSplineIdentifiability::None;
            }
            // Boundary conditions are emitted by the smooth-level paired
            // linear-constraint path (`bspline_boundary_linear_constraints`),
            // which supports non-zero anchors and composes them with the frozen
            // identifiability transform. Clear them here so the basis builder
            // does not also bake them into the basis null space — the legacy
            // basis-level path rejected non-zero anchors and dropped columns
            // before the frozen transform, conflicting with the constraint path.
            spec_local.boundary_conditions = BSplineBoundaryConditions::default();
            build_bspline_basis_1d(data.column(*feature_col), &spec_local)?
        }
        SmoothBasisSpec::ThinPlate {
            feature_cols,
            spec,
            input_scales,
        } => {
            if term.shape != ShapeConstraint::None {
                if feature_cols.len() != 1 {
                    crate::bail_invalid_basis!(
                        "ShapeConstraint::{:?} for term '{}' on ThinPlate basis requires exactly 1 feature axis; found {}",
                        term.shape,
                        term.name,
                        feature_cols.len()
                    );
                }
                shape_axis_col = Some(feature_cols[0]);
            }
            let mut x = select_columns(data, feature_cols)?;
            // Auto-standardize multivariate inputs: use stored scales (prediction)
            // or compute fresh ones (training). Same standardization-vs-
            // length-scale compensation as Matérn / hybrid Duchon: divide
            // the user's L by σ_geom so kernel(‖x_std − c_std‖/L_eff)
            // matches the original-coord kernel for uniform σ.
            let (scales, length_scale_eff) = if let Some(s) = input_scales {
                apply_input_standardization(&mut x, s);
                (
                    Some(s.clone()),
                    compensate_length_scale_for_standardization(spec.length_scale, s),
                )
            } else if let Some(s) = compute_spatial_input_scales(x.view()) {
                apply_input_standardization(&mut x, &s);
                let l_eff = compensate_length_scale_for_standardization(spec.length_scale, &s);
                (Some(s), l_eff)
            } else {
                (None, spec.length_scale)
            };
            let mut spec_local = spec.clone();
            spec_local.length_scale = length_scale_eff;
            if matches!(
                spec_local.identifiability,
                SpatialIdentifiability::OrthogonalToParametric
            ) {
                spec_local.identifiability = SpatialIdentifiability::None;
            }
            let mut result = build_thin_plate_basis(x.view(), &spec_local).map_err(|err| {
                rewrite_thin_plate_knots_error(err, &term.name, feature_cols.len(), spec)
            })?;
            // Inject input scales into metadata; also restore the user's
            // original length_scale (not the σ_geom-compensated one) so a
            // metadata-driven rebuild that re-applies compensation does not
            // double-divide. The build may auto-promote to Duchon when
            // canonical TPS is infeasible (k < polynomial-nullspace size);
            // in that case patch the Duchon metadata variant so predict-time
            // round-trips through the same standardized data path.
            match &mut result.metadata {
                BasisMetadata::ThinPlate {
                    input_scales: ms,
                    length_scale,
                    ..
                } => {
                    *ms = scales;
                    *length_scale = spec.length_scale;
                }
                BasisMetadata::Duchon {
                    input_scales: ms,
                    length_scale,
                    ..
                } => {
                    *ms = scales;
                    // The ThinPlate auto-promotion path delegates to
                    // `build_duchon_basis` with `Some(spec_local.length_scale)`,
                    // which is the σ_geom-compensated value. The metadata
                    // therefore records the compensated kernel range, but the
                    // freeze→replay round trip plugs that value back into a
                    // user-facing `DuchonBasisSpec.length_scale` whose builder
                    // applies the σ_geom compensation a second time. Restore
                    // the user-facing scale here so replay re-compensates
                    // exactly once and reproduces the realized fit-time basis.
                    *length_scale = Some(spec.length_scale);
                }
                _ => {}
            }
            result
        }
        SmoothBasisSpec::Sphere { feature_cols, spec } => {
            if term.shape != ShapeConstraint::None {
                crate::bail_invalid_basis!(
                    "ShapeConstraint::{:?} for term '{}' is not supported on spherical splines",
                    term.shape,
                    term.name
                );
            }
            let x = select_columns(data, feature_cols)?;
            build_spherical_spline_basis(x.view(), spec)?
        }
        SmoothBasisSpec::ConstantCurvature { feature_cols, spec } => {
            if term.shape != ShapeConstraint::None {
                crate::bail_invalid_basis!(
                    "ShapeConstraint::{:?} for term '{}' is not supported on constant-curvature smooths",
                    term.shape,
                    term.name
                );
            }
            // Chart coordinates are consumed verbatim: NO auto-standardization.
            // Rescaling axes would change the chart gauge `1 + κ‖x‖²` and
            // silently redefine which curvature κ refers to (the same point
            // cloud at a different chart scale has a different κ̂); the user's
            // coordinates ARE the geometry here, exactly as for the sphere
            // smooth's (lat, lon).
            let x = select_columns(data, feature_cols)?;
            build_constant_curvature_basis(x.view(), spec)?
        }
        SmoothBasisSpec::MeasureJet {
            feature_cols,
            spec,
            input_scales,
        } => {
            if term.shape != ShapeConstraint::None {
                crate::bail_invalid_basis!(
                    "ShapeConstraint::{:?} for term '{}' is not supported on measure-jet smooths",
                    term.shape,
                    term.name
                );
            }
            let mut x = select_columns(data, feature_cols)?;
            // Matern-style per-axis standardization; the realized σ vector is
            // persisted into the metadata for predict-time replay.
            //
            // Length-scale round-trip contract (owning statement; the freeze
            // and frozen-validation arms reference it): `input_scales: Some`
            // marks the REPLAY path — the frozen length_scale is already the
            // realized post-standardization value and passes through
            // verbatim. Fresh path: an explicit user length_scale is in
            // ORIGINAL coordinates and gets the σ_geom compensation; the 0.0
            // auto sentinel passes through (auto-derivation runs inside the
            // builder, post-standardization).
            let (scales, length_scale_eff) = if let Some(s) = input_scales {
                apply_input_standardization(&mut x, s);
                (Some(s.clone()), spec.length_scale)
            } else if let Some(s) = compute_spatial_input_scales(x.view()) {
                apply_input_standardization(&mut x, &s);
                let l_eff = if spec.length_scale > 0.0 {
                    compensate_length_scale_for_standardization(spec.length_scale, &s)
                } else {
                    spec.length_scale
                };
                (Some(s), l_eff)
            } else {
                (None, spec.length_scale)
            };
            let mut spec_local = spec.clone();
            spec_local.length_scale = length_scale_eff;
            let mut result = build_measure_jet_basis(x.view(), &spec_local)?;
            if let BasisMetadata::MeasureJet {
                input_scales: ms, ..
            } = &mut result.metadata
            {
                *ms = scales;
            }
            result
        }
        SmoothBasisSpec::Matern {
            feature_cols,
            spec,
            input_scales,
        } => {
            if term.shape != ShapeConstraint::None {
                if feature_cols.len() != 1 {
                    crate::bail_invalid_basis!(
                        "ShapeConstraint::{:?} for term '{}' on Matern basis requires exactly 1 feature axis; found {}",
                        term.shape,
                        term.name,
                        feature_cols.len()
                    );
                }
                shape_axis_col = Some(feature_cols[0]);
            }
            let mut x = select_columns(data, feature_cols)?;
            // Auto-standardization (per-axis division by σ_a) reinterprets
            // the user's `length_scale` from original data coordinates
            // into post-standardization coordinates: for uniform σ_a = σ,
            // `kernel(‖x_std − c_std‖/L)` equals `kernel(‖x − c‖/(σ·L))`,
            // so the effective kernel range shrinks by σ. To keep
            // `length_scale` consistently expressed in *original* data
            // coordinates regardless of axis variances, we standardize
            // and divide L by σ_geom = (∏σ_a)^(1/d). For uniform σ this
            // recovers the user's kernel exactly; for anisotropic data
            // the resulting per-axis effective scales σ_a / σ_geom are
            // the standard Mahalanobis preconditioning and preserve the
            // geometric-mean kernel range. Storing the σ vector in
            // metadata.input_scales makes the same transformation
            // replayable at predict time.
            let (scales, length_scale_eff) = if let Some(s) = input_scales {
                apply_input_standardization(&mut x, s);
                (
                    Some(s.clone()),
                    compensate_length_scale_for_standardization(spec.length_scale, s),
                )
            } else if let Some(s) = compute_spatial_input_scales(x.view()) {
                apply_input_standardization(&mut x, &s);
                let l_eff = compensate_length_scale_for_standardization(spec.length_scale, &s);
                (Some(s), l_eff)
            } else {
                (None, spec.length_scale)
            };
            let mut spec_local = spec.clone();
            spec_local.length_scale = length_scale_eff;
            let mut result = build_matern_basiswithworkspace(x.view(), &spec_local, workspace)?;
            if let BasisMetadata::Matern {
                input_scales,
                length_scale,
                ..
            } = &mut result.metadata
            {
                *input_scales = scales;
                *length_scale = spec.length_scale;
            }
            result
        }
        SmoothBasisSpec::Duchon {
            feature_cols,
            spec,
            input_scales,
        } => {
            if term.shape != ShapeConstraint::None {
                if feature_cols.len() != 1 {
                    crate::bail_invalid_basis!(
                        "ShapeConstraint::{:?} for term '{}' on Duchon basis requires exactly 1 feature axis; found {}",
                        term.shape,
                        term.name,
                        feature_cols.len()
                    );
                }
                shape_axis_col = Some(feature_cols[0]);
            }
            let mut x = select_columns(data, feature_cols)?;
            // Hybrid Duchon (length_scale=Some) is governed by the same
            // standardization-vs-length-scale equivalence as Matérn: the
            // user's `length_scale` is interpreted in original data
            // coordinates, but auto-standardization (per-axis division by
            // σ_a) reinterprets it as σ_geom · L. Pre-multiply by 1/σ_geom
            // so kernel(‖x_std − c_std‖/L_eff) reproduces the user's
            // original-coord kernel exactly for uniform σ_a, and reduces
            // to standard Mahalanobis preconditioning for anisotropic σ.
            // Pure Duchon (length_scale=None) is scale-free and needs no
            // compensation.
            let (scales, length_scale_eff) = if let Some(s) = input_scales {
                apply_input_standardization(&mut x, s);
                (
                    Some(s.clone()),
                    compensate_optional_length_scale_for_standardization(spec.length_scale, s),
                )
            } else if let Some(s) = compute_spatial_input_scales(x.view()) {
                apply_input_standardization(&mut x, &s);
                let l_eff =
                    compensate_optional_length_scale_for_standardization(spec.length_scale, &s);
                (Some(s), l_eff)
            } else {
                (None, spec.length_scale)
            };
            let mut spec_local = spec.clone();
            spec_local.length_scale = length_scale_eff;
            if matches!(
                spec_local.identifiability,
                SpatialIdentifiability::OrthogonalToParametric
            ) {
                spec_local.identifiability = SpatialIdentifiability::None;
            }
            let mut result = build_duchon_basiswithworkspace(x.view(), &spec_local, workspace)?;
            if let BasisMetadata::Duchon {
                input_scales,
                length_scale,
                ..
            } = &mut result.metadata
            {
                *input_scales = scales;
                *length_scale = spec.length_scale;
            }
            result
        }
        SmoothBasisSpec::Pca {
            feature_cols,
            basis_matrix,
            centered,
            smooth_penalty,
            center_mean,
            pca_basis_path,
            chunk_size,
        } => {
            if term.shape != ShapeConstraint::None {
                crate::bail_invalid_basis!(
                    "ShapeConstraint::{:?} for term '{}' is not supported on Pca basis",
                    term.shape,
                    term.name
                );
            }
            build_pca_smooth_basis(
                data,
                feature_cols,
                basis_matrix,
                *centered,
                *smooth_penalty,
                center_mean.as_ref(),
                pca_basis_path.as_ref(),
                *chunk_size,
            )?
        }
        SmoothBasisSpec::TensorBSpline { feature_cols, spec } => {
            build_tensor_bspline_basis(data, feature_cols, spec)?
        }
        SmoothBasisSpec::ByVariable { .. } => {
            crate::bail_invalid_basis!(
                "internal: ByVariable smooths must return before inner basis dispatch"
            );
        }
        SmoothBasisSpec::BySmooth { .. } => {
            crate::bail_invalid_basis!("internal: BySmooth smooths must be lowered to ByVariable before inner basis dispatch"
                    .to_string(),);
        }
        SmoothBasisSpec::FactorSmooth { spec } => {
            if term.shape != ShapeConstraint::None {
                crate::bail_invalid_basis!(
                    "ShapeConstraint::{:?} is unsupported for factor smooth term '{}'",
                    term.shape,
                    term.name
                );
            }
            return build_factor_smooth(data, spec, &term.name, workspace);
        }
    };

    if let SmoothBasisSpec::Matern { .. } = &term.basis {
        let (penalties, nullspace_dims, penaltyinfo) =
            matern_operator_penalty_triplet_from_metadata(&built.metadata)?;
        built.penalties = penalties;
        built.nullspace_dims = nullspace_dims;
        built.penaltyinfo = penaltyinfo;
    }

    let p_local = built.design.ncols();
    let mut metadata = built.metadata.clone();
    // Extract factored Kronecker representation before consuming fields.
    // Invalidate it if shape transforms will be applied (they break structure).
    let kron_factored = if term.shape == ShapeConstraint::None {
        built.kronecker_factored
    } else {
        None
    };
    let mut design_t = built.design;
    let mut penalties_t: Vec<Array2<f64>> = built.penalties;
    // Ops vector parallel to `penalties_t`. Survives unchanged through the
    // identity path; nulled element-wise when `T^T S T` reparametrization
    // is applied (operator no longer bit-equivalent to the transformed
    // matrix); wrapped in `ScaledPenaltyOp` after Frobenius normalization.
    let mut ops_t: Vec<Option<std::sync::Arc<dyn crate::terms::analytic_penalties::PenaltyOp>>> =
        built.ops;
    if matches!(
        spatial_identifiability_policy(term),
        Some(SpatialIdentifiability::OrthogonalToParametric)
    ) {
        metadata = freeze_raw_spatial_metadata(metadata, design_t.ncols());
    }

    let active_penaltyinfo_t = built
        .penaltyinfo
        .iter()
        .filter(|info| info.active)
        .cloned()
        .collect::<Vec<_>>();
    let pre_dropped_penaltyinfo_t = built
        .penaltyinfo
        .iter()
        .filter(|info| !info.active)
        .cloned()
        .collect::<Vec<_>>();
    let use_box_reparam =
        term.shape != ShapeConstraint::None && shape_uses_box_reparameterization(&term.basis);
    let mut coefficient_transform_for_constraints: Option<Array2<f64>> = None;
    if let Some((order, sign)) = shape_order_and_sign(term.shape)
        && use_box_reparam
    {
        // Order 1 (monotone): the plain first-difference cone θ_{i+1}−θ_i ≥ 0 is
        // the control-polygon monotonicity criterion, which is independent of
        // Greville-abscissa spacing (it only fixes the *sign* of consecutive
        // control-point gaps), so the integer-difference transform is exact.
        //
        // Order 2 (convex/concave): the plain second-difference cone is only
        // correct for evenly spaced Greville abscissae. gam's B-splines are
        // clamped (and may use quantile knots), so the abscissae are not
        // uniform and the geometrically-correct cone is the second *divided*
        // difference. Build the Greville-scaled transform so γ_{≥2} ≥ 0
        // certifies convexity of the function, not of the raw coefficient
        // index. Periodic B-splines use uniform interior knots (uniform
        // abscissae), where the divided differences coincide with the integer
        // differences up to scale, so the plain path stays exact there.
        let t = if order == 2 {
            let bspline_meta = match &metadata {
                BasisMetadata::BSpline1D {
                    knots,
                    degree,
                    periodic,
                    ..
                } if periodic.is_none() => Some((knots.clone(), degree.unwrap_or(0))),
                _ => None,
            };
            match bspline_meta {
                Some((knots, degree)) if degree >= 1 => {
                    let greville = crate::basis::compute_greville_abscissae(&knots, degree)?;
                    if greville.len() != p_local {
                        crate::bail_invalid_basis!(
                            "shape-constraint Greville abscissae count {} does not match basis dim {} for term '{}'",
                            greville.len(),
                            p_local,
                            term.name
                        );
                    }
                    convex_divided_difference_transform_matrix(&greville, sign)?
                }
                _ => cumulative_sum_transform_matrix(p_local, order, sign),
            }
        } else {
            cumulative_sum_transform_matrix(p_local, order, sign)
        };
        coefficient_transform_for_constraints = Some(t.clone());
        // Coefficient-side transform: wrap the design in an operator that
        // applies T on the coefficient side, preserving sparsity/operator
        // structure of the inner design.
        let inner_dense = match design_t {
            DesignMatrix::Dense(d) => d,
            DesignMatrix::Sparse(sp) => crate::matrix::DenseDesignMatrix::from(
                sp.try_to_dense_arc("shape-constrained coefficient transform")
                    .map_err(BasisError::InvalidInput)?,
            ),
        };
        let coeff_op = crate::matrix::CoefficientTransformOperator::new(inner_dense, t.clone())
            .map_err(|e| BasisError::InvalidInput(format!("CoefficientTransformOperator: {e}")))?;
        design_t = DesignMatrix::Dense(crate::matrix::DenseDesignMatrix::from(Arc::new(coeff_op)));
        if penalties_t.len() != active_penaltyinfo_t.len() {
            crate::bail_invalid_basis!(
                "internal box-reparam penalty/info mismatch for term '{}': penalties={}, infos={}",
                term.name,
                penalties_t.len(),
                active_penaltyinfo_t.len()
            );
        }
        // Wiggliness penalties undergo the exact congruence `S → TᵀST` (PSD
        // preserving). The double-penalty *nullspace shrinkage* ridge must NOT:
        // it is a unit-eigenvalue projector `ZZᵀ` onto null(S_wiggle) in the
        // β (B-spline coefficient) coordinates, and the congruence
        // `Tᵀ(ZZᵀ)T = (TᵀZ)(TᵀZ)ᵀ` is no longer a projector — its eigenvalues
        // blow up by the conditioning of the cumulative-sum `T` (cond(T) grows
        // with the basis dim), concentrating an enormous penalty on the leading
        // γ₀ "level" coordinate. REML then drives the shared λ to its ceiling
        // and the smooth collapses to a flat constant (#509, the over-smoothing
        // face). The principled fix keeps mgcv's double-penalty semantics in the
        // *reparametrized* space: rebuild the ridge as the unit-eigenvalue
        // nullspace projector of the transformed wiggliness penalty `TᵀST`, so
        // the double penalty shrinks exactly the unpenalized polynomial
        // directions of the γ-space smooth with eigenvalue 1, identical in
        // conditioning to the unconstrained fit.
        let transformed_wiggliness = penalties_t
            .iter()
            .zip(active_penaltyinfo_t.iter())
            .find(|(_, info)| !matches!(info.source, PenaltySource::DoublePenaltyNullspace))
            .map(|(s_local, _)| {
                let tt_s = fast_atb(&t, s_local);
                fast_ab(&tt_s, &t)
            });
        let mut rebuilt = Vec::with_capacity(penalties_t.len());
        for (s_local, info) in penalties_t.iter().zip(active_penaltyinfo_t.iter()) {
            if matches!(info.source, PenaltySource::DoublePenaltyNullspace) {
                let s_wiggle_t = transformed_wiggliness.as_ref().ok_or_else(|| {
                    BasisError::InvalidInput(format!(
                        "box-reparam term '{}' has a double-penalty ridge but no primary wiggliness penalty to derive its nullspace from",
                        term.name
                    ))
                })?;
                let ridge = crate::terms::basis::build_nullspace_shrinkage_penalty(s_wiggle_t)?
                    .map(|shrink| shrink.sym_penalty)
                    .unwrap_or_else(|| Array2::<f64>::zeros((p_local, p_local)));
                rebuilt.push(ridge);
            } else {
                let tt_s = fast_atb(&t, s_local);
                rebuilt.push(fast_ab(&tt_s, &t));
            }
        }
        penalties_t = rebuilt;
        // T^T S T (and the rebuilt γ-space ridge) invalidate op-form
        // bit-equivalence; drop ops here.
        ops_t = vec![None; penalties_t.len()];
    }
    if penalties_t.len() != active_penaltyinfo_t.len() {
        crate::bail_invalid_basis!(
            "internal penalty metadata mismatch for term '{}': active penalties={}, active infos={}",
            term.name,
            penalties_t.len(),
            active_penaltyinfo_t.len()
        );
    }
    if ops_t.len() != penalties_t.len() {
        ops_t = vec![None; penalties_t.len()];
    }
    let penalty_candidates = penalties_t
        .into_iter()
        .zip(active_penaltyinfo_t.into_iter())
        .zip(ops_t.into_iter())
        .map(
            |((matrix, info), op_in)| -> Result<PenaltyCandidate, BasisError> {
                let (matrix, c_new) = normalize_penalty_in_constrained_space(&matrix);
                let preserve_margin_scale =
                    matches!(&info.source, PenaltySource::TensorMarginal { .. });
                let (matrix, normalization_scale, op_scale, kronecker_scale) =
                    if preserve_margin_scale {
                        (
                            matrix.mapv(|v| v * c_new),
                            info.normalization_scale,
                            1.0,
                            1.0,
                        )
                    } else {
                        (
                            matrix,
                            info.normalization_scale * c_new,
                            1.0 / c_new,
                            1.0 / c_new,
                        )
                    };
                // Frobenius rescale: wrap inner op in `ScaledPenaltyOp(1/c_new)`
                // so `op.as_dense() == matrix` post-normalization.
                let scaled_op = if op_scale > 0.0 && op_scale.is_finite() {
                    op_in.map(|op| {
                        std::sync::Arc::new(crate::terms::analytic_penalties::ScaledPenaltyOp::new(
                            op, op_scale,
                        ))
                            as std::sync::Arc<dyn crate::terms::analytic_penalties::PenaltyOp>
                    })
                } else {
                    None
                };
                let kronecker_factors = info.kronecker_factors.map(|mut factors| {
                    if let Some(first) = factors.first_mut() {
                        first.mapv_inplace(|v| v * kronecker_scale);
                    }
                    factors
                });
                Ok(PenaltyCandidate {
                    nullspace_dim_hint: info.nullspace_dim_hint,
                    matrix,
                    source: info.source,
                    normalization_scale,
                    kronecker_factors,
                    op: scaled_op,
                })
            },
        )
        .collect::<Result<Vec<_>, _>>()?;
    let (penalties_t, nullspaces_t, penaltyinfo_t, null_eigenvectors_t, ops_t) =
        crate::terms::basis::filter_active_penalty_candidates_with_ops(penalty_candidates)?;
    let shape_linear_constraints = if term.shape != ShapeConstraint::None && !use_box_reparam {
        let axis = shape_axis_col.ok_or_else(|| {
            BasisError::InvalidInput(format!(
                "internal shape-constraint axis missing for term '{}'",
                term.name
            ))
        })?;
        let (x_shape_eval, design_shape_eval) =
            build_shape_constraint_design_1d(data, term, &metadata, axis)?;
        build_shape_linear_constraints_1d(
            x_shape_eval.view(),
            design_shape_eval.view(),
            term.shape,
        )?
    } else {
        None
    };
    let boundary_linear_constraints = match &term.basis {
        SmoothBasisSpec::BSpline1D { spec, .. } => bspline_boundary_linear_constraints(
            spec.boundary_conditions,
            &metadata,
            spec.degree,
            coefficient_transform_for_constraints.as_ref(),
        )?,
        _ => None,
    };
    let linear_constraints_local =
        merge_linear_constraints_global(shape_linear_constraints, boundary_linear_constraints);

    // Joint-null absorption rotation. Fresh fit specs compute Q from the final
    // per-smooth penalty set (after all in-smooth reparameterizations have
    // already been applied). Frozen specs already carry the complete realized
    // coefficient chart in their `FrozenTransform`; recomputing Q there would
    // rotate an already-frozen chart a second time and desynchronize value
    // rebuilds from derivative operators.
    //
    // Kronecker-factored smooths (tensor B-splines under `TensorBSplineIdentifiability::None`)
    // carry their joint penalty as `Σ_d S_d` with `S_d = I ⊗ … ⊗ S_d^{1D} ⊗ … ⊗ I`.
    // The joint null space is the tensor of marginal nulls and is handled directly
    // by the REML runtime's `kronecker_penalty_system` path (see
    // `runtime.rs:8334-8344`). Applying a dense (p × p) Q here would densify
    // `X_raw = mx ⊗ my` into `X_raw · Q`, destroying the Kronecker product
    // structure that the runtime relies on for fast log-det/derivative
    // assembly — and the rotation block at the wrapper site also unconditionally
    // wipes `kronecker_factored`, leaving the runtime to fall back to the
    // dense per-block log-det. Skip the rotation for Kronecker-factored terms
    // so the factored representation survives end-to-end.
    let joint_null_rotation = match term.joint_null_rotation.clone() {
        Some(persisted) => Some(persisted),
        None if smooth_has_frozen_identifiability(term) => None,
        None if kron_factored.is_some() => None,
        None => crate::terms::basis::compute_joint_null_rotation(&penalties_t)?,
    };

    Ok(LocalSmoothTermBuild {
        dim: p_local,
        design: design_t,
        penalties: penalties_t,
        ops: ops_t,
        nullspaces: nullspaces_t,
        null_eigenvectors: null_eigenvectors_t,
        joint_null_rotation,
        penaltyinfo: penaltyinfo_t,
        pre_dropped_penaltyinfo: pre_dropped_penaltyinfo_t,
        metadata,
        linear_constraints: linear_constraints_local,
        box_reparam: use_box_reparam,
        kronecker_factored: kron_factored,
    })
}

pub fn build_smooth_design(
    data: ArrayView2<'_, f64>,
    terms: &[SmoothTermSpec],
) -> Result<RawSmoothDesign, BasisError> {
    let mut ws = crate::basis::BasisWorkspace::new();
    build_smooth_design_withworkspace(data, terms, &mut ws)
}

/// Like `build_smooth_design`, but honors the caller workspace policy while
/// building each planned smooth term with an independent per-term workspace.
///
/// Independent workspaces avoid shared mutable distance-cache state during the
/// parallel term build; the final design, penalties, and metadata are assembled
/// in the original smooth-term order.
pub fn build_smooth_design_withworkspace(
    data: ArrayView2<'_, f64>,
    terms: &[SmoothTermSpec],
    workspace: &mut crate::basis::BasisWorkspace,
) -> Result<RawSmoothDesign, BasisError> {
    validate_smooth_terms_finite_inputs(data, terms)?;
    build_smooth_design_withworkspace_unvalidated(data, terms, workspace)
}

fn build_smooth_design_withworkspace_unvalidated(
    data: ArrayView2<'_, f64>,
    terms: &[SmoothTermSpec],
    workspace: &mut crate::basis::BasisWorkspace,
) -> Result<RawSmoothDesign, BasisError> {
    let mut planned_blocks = plan_joint_spatial_centers_for_term_blocks(data, &[terms.to_vec()])?;
    let planned_terms = planned_blocks.pop().ok_or_else(|| {
        BasisError::InvalidInput(
            "joint spatial center planner returned no smooth blocks".to_string(),
        )
    })?;
    let policy = workspace.policy().clone();
    let local_builds: Vec<LocalSmoothTermBuild> = {
        use rayon::iter::{IntoParallelIterator, ParallelIterator};
        planned_terms
            .into_par_iter()
            .map(|term| {
                let mut term_workspace = crate::basis::BasisWorkspace::with_policy(policy.clone());
                build_single_local_smooth_term(data, &term, &mut term_workspace)
            })
            .collect::<Result<Vec<_>, _>>()?
    };

    let total_p: usize = local_builds.iter().map(|built| built.dim).sum();

    let mut local_designs: Vec<DesignMatrix> = Vec::with_capacity(local_builds.len());
    let mut terms_out = Vec::<SmoothTerm>::with_capacity(terms.len());
    let mut penalties_global = Vec::<BlockwisePenalty>::new();
    let mut nullspace_dims_global = Vec::<usize>::new();
    let mut penaltyinfo_global = Vec::<PenaltyBlockInfo>::new();
    let mut dropped_penaltyinfo_global = Vec::<DroppedPenaltyBlockInfo>::new();
    let mut coefficient_lower_bounds = Array1::<f64>::from_elem(total_p, f64::NEG_INFINITY);
    let mut any_bounds = false;
    // Each linear-constraint row only touches the current term's column slice.
    // Track `(col_start, col_end, local_row_values)` and assemble the final
    // dense `Array2` in one pass, avoiding per-row `Array1::zeros(total_p)`
    // allocation plus a row-by-row copy at the end.
    let mut linear_constraintsrows: Vec<(usize, usize, Array1<f64>)> = Vec::new();
    let mut linear_constraints_b: Vec<f64> = Vec::new();

    let mut col_start = 0usize;
    for (term, mut built) in terms.iter().zip(local_builds.into_iter()) {
        let p_local = built.dim;
        let col_end = col_start + p_local;
        let lb_local = if built.box_reparam {
            shape_lower_bounds_local(term.shape, p_local)
        } else {
            None
        };

        // Stage-2 joint-null absorption rotation. Fired *before* the
        // penalty / design / global aggregation loops below so that every
        // subsequent reference to `built.penalties`, `built.design`, and
        // `built.ops` sees the post-rotation values.
        //
        // The math: when the smooth's joint penalty `Σ_k S_k` has a
        // non-trivial null space, eigh selects `Q = [U_range | U_null]`
        // with null columns at the tail. Setting `β_raw = Q · γ` and
        // applying:
        //     design        ← X · Q
        //     penalties[k]  ← Qᵀ · S_k · Q   (block-diag, zero null tail)
        // yields a model whose fitted γ is invariant to the rotation
        // (since likelihood depends only on `X · β_raw = X · Q · γ`), but
        // whose penalty is full-rank on the range columns. The large-scale
        // failing case (cert refusal in the joint-Newton inner solve)
        // resolves because `H_pen = H_loglik + S` becomes full rank on
        // the smooth's range columns.
        //
        // Rotation is suppressed when the smooth carries coordinate-wise
        // shape constraints (`lb_local` or `built.linear_constraints`):
        // those encode a cone in the original coordinate system and a
        // general orthogonal rotation breaks the cone geometry. Smooths
        // with shape constraints typically have full-rank joint penalty
        // (their structural shape comes from the cone, not from null
        // directions in the penalty), so suppression is rarely a loss.
        //
        // `applied_rotation` carries the Q that was applied (or `None`
        // if no rotation fired). It is persisted onto `SmoothTerm` below
        // so prediction-side `X_new_raw · Q` replay can reproduce the
        // exact rotation. Persistence through the saved-model artifact
        // is a follow-up — see the doc on `SmoothTerm.joint_null_rotation`.
        let applied_rotation: Option<crate::terms::basis::JointNullRotation> = match (
            built.joint_null_rotation.take(),
            lb_local.is_some(),
            built.linear_constraints.is_some(),
        ) {
            (Some(rot), false, false) => {
                let q = &rot.rotation;
                let dense = built
                    .design
                    .try_to_dense_by_chunks("joint-null absorption rotation")
                    .map_err(BasisError::InvalidInput)?;
                let rotated = crate::linalg::faer_ndarray::fast_ab(&dense, q);
                built.design = DesignMatrix::Dense(crate::matrix::DenseDesignMatrix::from(rotated));
                built.penalties = built
                    .penalties
                    .into_iter()
                    .map(|s_local| {
                        let qt_s = crate::linalg::faer_ndarray::fast_atb(q, &s_local);
                        crate::linalg::faer_ndarray::fast_ab(&qt_s, q)
                    })
                    .collect();
                built.ops = vec![None; built.penalties.len()];
                built.kronecker_factored = None;
                Some(rot)
            }
            (Some(_), _, _) => None,
            (None, _, _) => None,
        };

        let activeinfos = built
            .penaltyinfo
            .iter()
            .filter(|info| info.active)
            .collect::<Vec<_>>();
        if activeinfos.len() != built.penalties.len() {
            crate::bail_invalid_basis!(
                "internal penalty info mismatch for term '{}': activeinfos={}, penalties={}",
                term.name,
                activeinfos.len(),
                built.penalties.len()
            );
        }
        for (((s_local, &ns), info), op_local) in built
            .penalties
            .iter()
            .zip(built.nullspaces.iter())
            .zip(activeinfos.into_iter())
            .zip(built.ops.iter())
        {
            let global_index = penalties_global.len();
            penalties_global.push(
                BlockwisePenalty::new(col_start..col_end, s_local.clone())
                    .with_op(op_local.clone()),
            );
            nullspace_dims_global.push(ns);
            let mut penalty = info.clone();
            penalty.nullspace_dim_hint = ns;
            penaltyinfo_global.push(PenaltyBlockInfo {
                global_index,
                termname: Some(term.name.clone()),
                penalty,
            });
        }
        for info in built.penaltyinfo.iter().filter(|info| !info.active) {
            dropped_penaltyinfo_global.push(DroppedPenaltyBlockInfo {
                termname: Some(term.name.clone()),
                penalty: info.clone(),
            });
        }
        for info in &built.pre_dropped_penaltyinfo {
            dropped_penaltyinfo_global.push(DroppedPenaltyBlockInfo {
                termname: Some(term.name.clone()),
                penalty: info.clone(),
            });
        }

        if let Some(lin_local) = &built.linear_constraints {
            for r in 0..lin_local.a.nrows() {
                linear_constraintsrows.push((col_start, col_end, lin_local.a.row(r).to_owned()));
                linear_constraints_b.push(lin_local.b[r]);
            }
        }
        if let Some(lb_local) = &lb_local {
            coefficient_lower_bounds
                .slice_mut(s![col_start..col_end])
                .assign(lb_local);
            any_bounds = true;
        }

        // Move the per-term design out of `built` rather than cloning it.
        local_designs.push(built.design);

        terms_out.push(SmoothTerm {
            name: term.name.clone(),
            coeff_range: col_start..col_end,
            shape: term.shape,
            penalties_local: built.penalties,
            nullspace_dims: built.nullspaces,
            penaltyinfo_local: built.penaltyinfo,
            metadata: built.metadata,
            lower_bounds_local: lb_local,
            linear_constraints_local: built.linear_constraints,
            kronecker_factored: built.kronecker_factored.take(),
            joint_null_rotation: applied_rotation,
            unabsorbed_global_orthogonality: None,
        });

        col_start = col_end;
    }

    assert_eq!(
        penalties_global.len(),
        nullspace_dims_global.len(),
        "global smooth penalty/nullspace bookkeeping diverged"
    );
    assert_eq!(
        penalties_global.len(),
        penaltyinfo_global.len(),
        "global smooth penalty metadata bookkeeping diverged"
    );

    Ok(RawSmoothDesign {
        term_designs: local_designs,
        penalties: penalties_global,
        nullspace_dims: nullspace_dims_global,
        penaltyinfo: penaltyinfo_global,
        dropped_penaltyinfo: dropped_penaltyinfo_global,
        terms: terms_out,
        coefficient_lower_bounds: if any_bounds {
            Some(coefficient_lower_bounds)
        } else {
            None
        },
        linear_constraints: if linear_constraintsrows.is_empty() {
            None
        } else {
            let mut a = Array2::<f64>::zeros((linear_constraintsrows.len(), total_p));
            for (i, (cs, ce, values)) in linear_constraintsrows.iter().enumerate() {
                a.row_mut(i).slice_mut(s![*cs..*ce]).assign(values);
            }
            Some(LinearInequalityConstraints {
                a,
                b: Array1::from_vec(linear_constraints_b),
            })
        },
    })
}
