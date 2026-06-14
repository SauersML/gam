//! Smooth-term *specification* types: the user-facing config/spec enums and
//! structs (shape constraints, basis specs, term/collection specs, penalty and
//! coefficient-group specs) plus their parsing and inherent value methods. This
//! is the vocabulary every other smooth submodule builds on; it owns no design
//! assembly, optimization, or fitting logic.

use super::*;

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
    pub fn write_structural_shape_hash(&self, h: &mut crate::cache::Fingerprinter) {
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
    pub op: Option<std::sync::Arc<dyn crate::terms::penalty_op::PenaltyOp>>,
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
        op: Option<std::sync::Arc<dyn crate::terms::penalty_op::PenaltyOp>>,
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
pub enum CoefficientSelector {
    /// Explicit global coefficient indices in the realized design matrix.
    GlobalColumns(Vec<usize>),
    /// A half-open global coefficient range.
    GlobalRange(Range<usize>),
    LinearTerm(String),
    RandomEffectTerm(String),
    SmoothTerm(String),
    /// Selected basis columns within one smooth term.
    SmoothTermColumns {
        term: String,
        columns: Vec<usize>,
    },
}


#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CoefficientGroupPrior {
    Flat,
    NormalLogPrecision {
        mean: f64,
        sd: f64,
    },
    GammaPrecision {
        shape: f64,
        rate: f64,
    },
    /// Penalized-complexity prior calibrated by `P(exp(-ρ/2) > upper) =
    /// tail_prob`; see [`crate::types::RhoPrior::PenalizedComplexity`].
    PenalizedComplexity {
        upper: f64,
        tail_prob: f64,
    },
}


impl CoefficientGroupPrior {
    fn to_rho_prior(&self) -> crate::types::RhoPrior {
        match *self {
            Self::Flat => crate::types::RhoPrior::Flat,
            Self::NormalLogPrecision { mean, sd } => crate::types::RhoPrior::Normal { mean, sd },
            Self::GammaPrecision { shape, rate } => {
                crate::types::RhoPrior::GammaPrecision { shape, rate }
            }
            Self::PenalizedComplexity { upper, tail_prob } => {
                crate::types::RhoPrior::PenalizedComplexity { upper, tail_prob }
            }
        }
    }

    fn validate(&self, context: &str) -> Result<(), BasisError> {
        match *self {
            Self::Flat => Ok(()),
            Self::NormalLogPrecision { mean, sd } => {
                if !mean.is_finite() {
                    crate::bail_invalid_basis!(
                        "{context} Normal log-precision prior requires finite mean, got {mean}"
                    );
                }
                if !sd.is_finite() || sd <= 0.0 {
                    crate::bail_invalid_basis!(
                        "{context} Normal log-precision prior requires sd > 0, got {sd}"
                    );
                }
                Ok(())
            }
            Self::GammaPrecision { shape, rate } => {
                validate_gamma_precision_prior(context, shape, rate)
            }
            Self::PenalizedComplexity { upper, tail_prob } => {
                validate_penalized_complexity_prior(context, upper, tail_prob)
            }
        }
    }
}


#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CoefficientGroupSpec {
    pub name: String,
    pub selectors: Vec<CoefficientSelector>,
    pub parent: Option<String>,
    pub prior: Option<CoefficientGroupPrior>,
    #[serde(skip, default)]
    pub prior_mean: crate::solver::estimate::CoefficientPriorMean,
}


#[derive(Debug, Clone)]
pub struct RealizedCoefficientGroups {
    pub penalty_specs: Vec<PenaltySpec>,
    pub nullspace_dims: Vec<usize>,
    pub rho_prior: crate::types::RhoPrior,
    pub group_column_indices: Vec<(String, Vec<usize>)>,
}


#[derive(Debug, Clone)]
pub struct PenaltyBlockGammaPriorMetadata<'a> {
    pub label: String,
    pub global_index: usize,
    pub termname: Option<&'a str>,
    pub source: String,
    pub effective_rank: usize,
    pub nullspace_dim_hint: usize,
}


fn penalty_block_label_candidates(info: &PenaltyBlockInfo) -> Vec<String> {
    let mut labels = Vec::<String>::new();
    labels.push(format!("penalty:{}", info.global_index));
    labels.push(info.global_index.to_string());
    if let Some(termname) = info.termname.as_ref() {
        labels.push(termname.clone());
        labels.push(format!("{termname}:{}", info.penalty.original_index));
    }
    if let PenaltySource::Other(label) = &info.penalty.source {
        labels.push(label.clone());
    }
    labels.push(format!("{:?}", info.penalty.source));
    labels.sort();
    labels.dedup();
    labels
}


fn penalty_block_metadata(info: &PenaltyBlockInfo) -> PenaltyBlockGammaPriorMetadata<'_> {
    PenaltyBlockGammaPriorMetadata {
        label: info
            .termname
            .clone()
            .unwrap_or_else(|| format!("penalty:{}", info.global_index)),
        global_index: info.global_index,
        termname: info.termname.as_deref(),
        source: format!("{:?}", info.penalty.source),
        effective_rank: info.penalty.effective_rank,
        nullspace_dim_hint: info.penalty.nullspace_dim_hint,
    }
}


fn validate_gamma_precision_prior(label: &str, shape: f64, rate: f64) -> Result<(), BasisError> {
    if !shape.is_finite() || shape <= 0.0 {
        crate::bail_invalid_basis!(
            "Gamma precision hyperprior for penalty block '{label}' requires shape > 0, got {shape}"
        );
    }
    if !rate.is_finite() || rate < 0.0 {
        crate::bail_invalid_basis!(
            "Gamma precision hyperprior for penalty block '{label}' requires rate >= 0, got {rate}"
        );
    }
    Ok::<(), _>(())
}


fn validate_penalized_complexity_prior(
    label: &str,
    upper: f64,
    tail_prob: f64,
) -> Result<(), BasisError> {
    if !upper.is_finite() || upper <= 0.0 {
        crate::bail_invalid_basis!(
            "Penalized-complexity hyperprior for '{label}' requires upper > 0, got {upper}"
        );
    }
    if !tail_prob.is_finite() || tail_prob <= 0.0 || tail_prob >= 1.0 {
        crate::bail_invalid_basis!(
            "Penalized-complexity hyperprior for '{label}' requires tail probability in (0, 1), got {tail_prob}"
        );
    }
    Ok::<(), _>(())
}


fn realize_penalty_block_gamma_priors<F>(
    design: &TermCollectionDesign,
    mut callback: F,
) -> Result<crate::types::RhoPrior, BasisError>
where
    F: FnMut(&PenaltyBlockGammaPriorMetadata<'_>) -> Option<(f64, f64)>,
{
    let mut priors = Vec::<crate::types::RhoPrior>::with_capacity(design.penaltyinfo.len());
    for info in &design.penaltyinfo {
        let metadata = penalty_block_metadata(info);
        if let Some((shape, rate)) = callback(&metadata) {
            validate_gamma_precision_prior(&metadata.label, shape, rate)?;
            priors.push(crate::types::RhoPrior::GammaPrecision { shape, rate });
        } else {
            priors.push(crate::types::RhoPrior::Flat);
        }
    }
    Ok(crate::types::RhoPrior::Independent(priors))
}


fn realize_keyed_penalty_block_gamma_priors(
    design: &TermCollectionDesign,
    priors: &[(String, f64, f64)],
) -> Result<crate::types::RhoPrior, BasisError> {
    let mut keyed = BTreeMap::<String, (f64, f64)>::new();
    for (label, shape, rate) in priors {
        if keyed.insert(label.clone(), (*shape, *rate)).is_some() {
            crate::bail_invalid_basis!(
                "duplicate Gamma precision hyperprior for penalty block label '{label}'"
            );
        }
    }
    let mut consumed = BTreeSet::<String>::new();
    let prior = realize_penalty_block_gamma_priors(design, |metadata| {
        let info = design
            .penaltyinfo
            .iter()
            .find(|info| info.global_index == metadata.global_index)
            .expect("metadata global index should match penaltyinfo");
        for label in penalty_block_label_candidates(info) {
            if let Some(value) = keyed.get(&label) {
                consumed.insert(label);
                return Some(*value);
            }
        }
        None
    })?;
    let unknown: Vec<String> = keyed
        .keys()
        .filter(|label| !consumed.contains(*label))
        .cloned()
        .collect();
    if !unknown.is_empty() {
        let available = design
            .penaltyinfo
            .iter()
            .flat_map(penalty_block_label_candidates)
            .collect::<BTreeSet<_>>()
            .into_iter()
            .collect::<Vec<_>>()
            .join(", ");
        crate::bail_invalid_basis!(
            "unknown Gamma precision hyperprior penalty block label(s): {}; available labels: {available}",
            unknown.join(", ")
        );
    }
    Ok(prior)
}


fn validate_rho_prior_coordinate(
    prior: &crate::types::RhoPrior,
    context: &str,
) -> Result<(), BasisError> {
    match prior {
        crate::types::RhoPrior::Flat => Ok(()),
        crate::types::RhoPrior::Normal { mean, sd } => {
            if !mean.is_finite() {
                crate::bail_invalid_basis!(
                    "{context} Normal log-precision prior requires finite mean, got {mean}"
                );
            }
            if !sd.is_finite() || *sd <= 0.0 {
                crate::bail_invalid_basis!(
                    "{context} Normal log-precision prior requires sd > 0, got {sd}"
                );
            }
            Ok(())
        }
        crate::types::RhoPrior::GammaPrecision { shape, rate } => {
            validate_gamma_precision_prior(context, *shape, *rate)
        }
        crate::types::RhoPrior::PenalizedComplexity { upper, tail_prob } => {
            validate_penalized_complexity_prior(context, *upper, *tail_prob)
        }
        crate::types::RhoPrior::Independent(_) => Err(BasisError::InvalidInput(format!(
            "{context} must be a scalar rho prior, not a nested Independent prior"
        ))),
    }
}


fn expand_base_rho_prior(
    base_prior: &crate::types::RhoPrior,
    base_count: usize,
    context: &str,
) -> Result<Vec<crate::types::RhoPrior>, BasisError> {
    match base_prior {
        crate::types::RhoPrior::Independent(priors) => {
            if priors.len() != base_count {
                crate::bail_invalid_basis!(
                    "{context} base Independent rho prior length mismatch: got {}, expected {base_count}",
                    priors.len()
                );
            }
            for (idx, prior) in priors.iter().enumerate() {
                validate_rho_prior_coordinate(prior, &format!("{context} base prior {idx}"))?;
            }
            Ok(priors.clone())
        }
        prior => {
            validate_rho_prior_coordinate(prior, context)?;
            Ok((0..base_count).map(|_| prior.clone()).collect())
        }
    }
}


fn combine_group_rho_prior(
    base_prior: &crate::types::RhoPrior,
    base_count: usize,
    groups: &[CoefficientGroupSpec],
) -> Result<crate::types::RhoPrior, BasisError> {
    let mut priors = Vec::with_capacity(base_count + groups.len());
    priors.extend(expand_base_rho_prior(
        base_prior,
        base_count,
        "coefficient groups",
    )?);
    for group in groups {
        let context = format!("coefficient group '{}'", group.name);
        let prior = match group.prior.as_ref() {
            Some(prior) => {
                prior.validate(&context)?;
                prior.to_rho_prior()
            }
            None => {
                validate_rho_prior_coordinate(base_prior, &context)?;
                base_prior.clone()
            }
        };
        priors.push(prior);
    }
    Ok(crate::types::RhoPrior::Independent(priors))
}


fn insert_range(
    cols: &mut BTreeSet<usize>,
    range: Range<usize>,
    p: usize,
    context: &str,
) -> Result<(), BasisError> {
    if range.end > p {
        crate::bail_dim_basis!(
            "{context} coefficient range {}..{} exceeds design width {p}",
            range.start,
            range.end
        );
    }
    cols.extend(range);
    Ok(())
}


fn resolve_group_columns(
    design: &TermCollectionDesign,
    group: &CoefficientGroupSpec,
) -> Result<BTreeSet<usize>, BasisError> {
    let p = design.design.ncols();
    let mut cols = BTreeSet::<usize>::new();
    for selector in &group.selectors {
        match selector {
            CoefficientSelector::GlobalColumns(indices) => {
                for &idx in indices {
                    if idx >= p {
                        crate::bail_dim_basis!(
                            "coefficient group '{}' references global column {idx}, but design width is {p}",
                            group.name
                        );
                    }
                    cols.insert(idx);
                }
            }
            CoefficientSelector::GlobalRange(range) => insert_range(
                &mut cols,
                range.clone(),
                p,
                &format!("coefficient group '{}'", group.name),
            )?,
            CoefficientSelector::LinearTerm(name) => {
                let (_, range) = design
                    .linear_ranges
                    .iter()
                    .find(|(term, _)| term == name)
                    .ok_or_else(|| {
                        BasisError::InvalidInput(format!(
                            "coefficient group '{}' references unknown linear term '{name}'",
                            group.name
                        ))
                    })?;
                insert_range(&mut cols, range.clone(), p, &group.name)?;
            }
            CoefficientSelector::RandomEffectTerm(name) => {
                let (_, range) = design
                    .random_effect_ranges
                    .iter()
                    .find(|(term, _)| term == name)
                    .ok_or_else(|| {
                        BasisError::InvalidInput(format!(
                            "coefficient group '{}' references unknown random-effect term '{name}'",
                            group.name
                        ))
                    })?;
                insert_range(&mut cols, range.clone(), p, &group.name)?;
            }
            CoefficientSelector::SmoothTerm(name) => {
                let term = design
                    .smooth
                    .terms
                    .iter()
                    .find(|term| &term.name == name)
                    .ok_or_else(|| {
                        BasisError::InvalidInput(format!(
                            "coefficient group '{}' references unknown smooth term '{name}'",
                            group.name
                        ))
                    })?;
                let start = p - design.smooth.total_smooth_cols() + term.coeff_range.start;
                insert_range(
                    &mut cols,
                    start..(start + term.coeff_range.len()),
                    p,
                    &group.name,
                )?;
            }
            CoefficientSelector::SmoothTermColumns { term, columns } => {
                let smooth_term = design
                    .smooth
                    .terms
                    .iter()
                    .find(|smooth_term| &smooth_term.name == term)
                    .ok_or_else(|| {
                        BasisError::InvalidInput(format!(
                            "coefficient group '{}' references unknown smooth term '{term}'",
                            group.name
                        ))
                    })?;
                let smooth_start = p - design.smooth.total_smooth_cols();
                for &local_col in columns {
                    if local_col >= smooth_term.coeff_range.len() {
                        crate::bail_dim_basis!(
                            "coefficient group '{}' references smooth term '{term}' local column {local_col}, but the term has {} columns",
                            group.name,
                            smooth_term.coeff_range.len()
                        );
                    }
                    cols.insert(smooth_start + smooth_term.coeff_range.start + local_col);
                }
            }
        }
    }
    if cols.is_empty() {
        crate::bail_invalid_basis!(
            "coefficient group '{}' contains no coefficients",
            group.name
        );
    }
    Ok(cols)
}


fn realize_coefficient_groups(
    design: &TermCollectionDesign,
    groups: &[CoefficientGroupSpec],
    base_prior: &crate::types::RhoPrior,
) -> Result<RealizedCoefficientGroups, BasisError> {
    use crate::terms::coefficient_group_resolver::{ResolvedGroup, ResolvedGroupHierarchy};

    let p = design.design.ncols();
    // Carrier-specific validation and selector resolution. The standard-term
    // carrier is columns of the realized design matrix; `resolve_group_columns`
    // turns each declared selector into a `BTreeSet<usize>` and rejects empty
    // selector lists. The prior is validated here because its diagnostic
    // context uses the standard-term label.
    for group in groups {
        if group.selectors.is_empty() {
            crate::bail_invalid_basis!("coefficient group '{}' contains no selectors", group.name);
        }
        if let Some(prior) = group.prior.as_ref() {
            prior.validate(&format!("coefficient group '{}'", group.name))?;
        }
    }

    let resolved_groups = groups
        .iter()
        .map(|group| {
            Ok(ResolvedGroup {
                label: group.name.clone(),
                parent: group.parent.clone(),
                coordinates: resolve_group_columns(design, group)?,
            })
        })
        .collect::<Result<Vec<_>, BasisError>>()?;
    // Carrier-agnostic policy: unique non-empty labels, acyclic hierarchy,
    // child ⊆ parent, interior == union of children.
    let hierarchy =
        ResolvedGroupHierarchy::build(resolved_groups).map_err(BasisError::InvalidInput)?;

    let mut penalty_specs: Vec<PenaltySpec> = design
        .penalties
        .iter()
        .map(PenaltySpec::from_blockwise_ref)
        .collect();
    let mut nullspace_dims = design.nullspace_dims.clone();
    let mut group_column_indices = Vec::<(String, Vec<usize>)>::with_capacity(groups.len());
    for (group, resolved) in groups.iter().zip(hierarchy.groups()) {
        let cols = &resolved.coordinates;
        let mut penalty = Array2::<f64>::zeros((p, p));
        let penalty_components = hierarchy.concatenated_penalty_components(&group.name);
        let active_cols = penalty_components
            .iter()
            .flat_map(|component| component.iter().copied())
            .collect::<BTreeSet<_>>();
        let local_mean = group
            .prior_mean
            .evaluate(
                active_cols.len(),
                &format!("coefficient group '{}'", group.name),
            )
            .map_err(|err| BasisError::InvalidInput(err.to_string()))?;
        let mut prior_mean = Array1::<f64>::zeros(p);
        // Hierarchical Gamma precision update.
        //
        // For a leaf group,
        //
        //   p(beta_g | lambda_g) p(lambda_g)
        //     ∝ lambda_g^{|g|/2}
        //       exp[-lambda_g (beta_g - mu_g)' S_g (beta_g - mu_g) / 2]
        //       lambda_g^{a_g-1} exp[-b_g lambda_g],
        //
        // so fixed-beta MAP gives
        //
        //   lambda_g* = (a_g + |g|/2 - 1)
        //               / (b_g + (beta_g - mu_g)' S_g (beta_g - mu_g) / 2).
        //
        // Interior nodes use the same identity with beta_g formed by
        // concatenating child beta vectors.  Equivalently, |g| and the
        // quadratic are sums over the recursively expanded child factors.  In
        // the standard term-collection path there is one rho coordinate per
        // group, so we materialize that summed child penalty into the group's
        // dense S_g.  Leaves reduce to the ordinary identity penalty.
        for component in &penalty_components {
            for &col in component {
                penalty[[col, col]] += 1.0;
            }
        }
        for (mean_idx, &col) in active_cols.iter().enumerate() {
            prior_mean[col] = local_mean[mean_idx];
        }
        penalty_specs.push(PenaltySpec::DenseWithMean {
            matrix: penalty,
            prior_mean: crate::solver::estimate::CoefficientPriorMean::constant(prior_mean),
        });
        nullspace_dims.push(p.saturating_sub(active_cols.len()));
        group_column_indices.push((group.name.clone(), cols.iter().copied().collect()));
    }

    Ok(RealizedCoefficientGroups {
        penalty_specs,
        nullspace_dims,
        rho_prior: combine_group_rho_prior(base_prior, design.penalties.len(), groups)?,
        group_column_indices,
    })
}


#[derive(Clone)]
