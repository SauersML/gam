//! The one-dimensional position basis a single-λ Gaussian REML position fit
//! builds (#2899 P4).
//!
//! `gaussian_reml_fit_positions` and its backward and batched forms take the
//! sample positions `t`, a basis kind, an optional basis size or explicit
//! knots/centers, an optional penalty, an optional order and the periodic
//! domain. [`resolve_position_basis`] turns that request into the exact basis
//! state the fit builds and a caller replays at predict time: the kind, the
//! effective order, the knot or center vector, the single-λ penalty, and the
//! wrap period the basis and penalty share. The bindings forward the request
//! here unchanged, so no front door resolves a basis of its own.
//!
//! An omitted basis size takes the formula front door's univariate default for
//! the same kind on the same data: the open B-spline internal-knot heuristic,
//! the cyclic basis dimension, and the 1-D Duchon center count.

use super::*;
use crate::term_builder::{
    DEFAULT_BSPLINE_DEGREE, DEFAULT_PENALTY_ORDER, default_cyclic_basis_dim,
    default_duchon_center_count, heuristic_knots_for_column, univariate_spline_basis_dim,
};

/// Where a position basis's knots or centers come from.
#[derive(Clone, Debug)]
pub enum PositionBasisLocations {
    /// The formula front door's default basis size for the kind on `t`.
    Default,
    /// A basis size: the internal knots of an open B-spline, the functions of
    /// a cyclic B-spline, or the centers of a Duchon basis.
    Count(usize),
    /// An explicit knot vector, cyclic grid or center vector, used as given.
    Given(Array1<f64>),
}

/// The single-λ penalty a position fit uses.
#[derive(Clone, Debug)]
pub enum PositionPenaltyRequest {
    /// The kind's canonical penalty.
    Canonical,
    /// The canonical penalty named explicitly (`"roughness"` for a B-spline,
    /// `"function-norm"` for a Duchon basis, and their aliases). A name the kind
    /// does not own is refused.
    Named(String),
    /// An explicit penalty matrix, used as given.
    Given(Array2<f64>),
}

/// The two position-basis engines.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PositionBasisKind {
    BSpline,
    /// The Duchon spline. The 1-D thin-plate spline is its `m = 2` member, the
    /// cubic smoothing spline.
    Duchon,
}

impl PositionBasisKind {
    /// The name the position design builders dispatch on.
    pub fn engine_name(self) -> &'static str {
        match self {
            Self::BSpline => "bspline",
            Self::Duchon => "duchon",
        }
    }

    /// Read a caller's basis kind. Case, `_` and `-` are ignored, so
    /// `"B-Spline"`, `"duchon_spline"` and `"thin_plate"` all resolve.
    pub fn parse(basis_kind: &str) -> Result<Self, String> {
        let normalized = basis_kind
            .trim()
            .to_ascii_lowercase()
            .replace(['_', '-'], "");
        match normalized.as_str() {
            "bspline" | "spline" => Ok(Self::BSpline),
            "duchon" | "duchonspline" | "thinplate" | "thinplatespline" | "tps" => {
                Ok(Self::Duchon)
            }
            "duchonmultipenalty" | "duchontripleoperator" => Err(
                "basis_kind='duchon_multipenalty' is the multi-λ amplitude/slope/curvature \
                 smoother and has no single-λ position-helper representation. Fit it through the \
                 formula API (gamfit.smooth.Duchon / basis_kind='duchon')."
                    .to_string(),
            ),
            _ => Err(format!(
                "basis_kind must be 'bspline', 'duchon' or 'thinplate'; got {basis_kind:?}"
            )),
        }
    }

    /// The order an omitted `basis_order` takes: the formula front door's cubic
    /// B-spline degree, and for Duchon the order `m` of the cubic default's
    /// polynomial null space.
    fn default_order(self) -> usize {
        match self {
            Self::BSpline => DEFAULT_BSPLINE_DEGREE,
            Self::Duchon => duchon_p_from_nullspace_order(duchon_cubic_default(1).0),
        }
    }
}

/// The basis state a position fit builds and a caller replays.
#[derive(Clone, Debug)]
pub struct ResolvedPositionBasis {
    /// The caller's spelling of the kind, or the default kind's name.
    pub display_kind: String,
    pub kind: PositionBasisKind,
    /// The B-spline degree or the Duchon order `m`. An auto-placed open knot
    /// vector lowers the requested degree when `t` is too short for it (#340);
    /// this is the order the knots were built for.
    pub order: usize,
    /// The knot vector, cyclic grid or center vector.
    pub locations: Array1<f64>,
    pub penalty: Array2<f64>,
    /// The domain wrap the basis and penalty share, `None` for an open basis.
    pub period: Option<f64>,
}

/// Resolve a position fit's basis request on the sample positions `t`.
///
/// - The kind defaults to a B-spline. The order defaults to
///   [`PositionBasisKind::default_order`] and must be at least 1.
/// - Open B-spline: an explicit knot vector is used as given; otherwise the
///   internal-knot count (the request's, or the formula default on `t`) is
///   placed at quantiles by [`auto_knot_vector_1d_quantile`], which may lower
///   the degree for a short `t`.
/// - Cyclic B-spline: the explicit period is required. A basis size `K` (the
///   request's, or the formula's cyclic default) becomes the `K + 1`-point
///   uniform grid over `[min t, min t + period]`, one cyclic control per
///   interval, and needs `K >= degree + 1`.
/// - Duchon: an explicit center vector is used as given; otherwise the center
///   count (the request's, at least 2, or the formula's 1-D Duchon default on
///   `t`) is placed by equal mass. A periodic Duchon basis without a period
///   wraps at the center span plus one mean center spacing, so the two end
///   centers sit one spacing apart across the wrap (a span-sized period gave a
///   non-PSD Gram, gam#580).
/// - Penalty: an explicit matrix is used as given; otherwise the kind's
///   canonical single-λ penalty is built on the resolved locations: the exact
///   second-derivative roughness of the (open or cyclic) B-spline, or the
///   function-norm Gram of the cubic Duchon basis at the shared period.
pub fn resolve_position_basis(
    t: ArrayView1<'_, f64>,
    basis_kind: Option<&str>,
    locations: PositionBasisLocations,
    penalty: PositionPenaltyRequest,
    basis_order: Option<usize>,
    periodic: bool,
    period: Option<f64>,
) -> Result<ResolvedPositionBasis, String> {
    finite_nonempty("t", t)?;
    let kind = basis_kind.map_or(Ok(PositionBasisKind::BSpline), PositionBasisKind::parse)?;
    let display_kind = basis_kind.map_or_else(|| kind.engine_name().to_string(), str::to_string);
    let order = basis_order.unwrap_or_else(|| kind.default_order());
    if order < 1 {
        return Err("basis_order must be at least 1".to_string());
    }
    if let Some(period) = period {
        if !periodic {
            return Err("period is only valid when periodic=true".to_string());
        }
        if !(period.is_finite() && period > 0.0) {
            return Err(format!("period must be finite and positive, got {period}"));
        }
    }
    if let PositionBasisLocations::Given(given) = &locations {
        finite_nonempty("knots_or_centers", given.view())?;
    }
    let (locations, order, period) = match kind {
        PositionBasisKind::BSpline => {
            let (knots, degree) = bspline_locations(t, locations, order, periodic, period)?;
            (knots, degree, period)
        }
        PositionBasisKind::Duchon => {
            let centers = duchon_centers(t, locations)?;
            let period = if periodic {
                Some(match period {
                    Some(period) => period,
                    None => duchon_wrap_period(centers.view())?,
                })
            } else {
                None
            };
            validate_position_period("duchon", centers.view(), periodic, period)?;
            (centers, order, period)
        }
    };
    let penalty = position_penalty(kind, penalty, locations.view(), order, period)?;
    Ok(ResolvedPositionBasis {
        display_kind,
        kind,
        order,
        locations,
        penalty,
        period,
    })
}

fn finite_nonempty(name: &str, values: ArrayView1<'_, f64>) -> Result<(), String> {
    if values.is_empty() {
        return Err(format!("{name} cannot be empty"));
    }
    if values.iter().any(|value| !value.is_finite()) {
        return Err(format!("{name} must contain only finite values"));
    }
    Ok(())
}

/// The knots (or cyclic grid) and effective degree of a B-spline position basis.
fn bspline_locations(
    t: ArrayView1<'_, f64>,
    request: PositionBasisLocations,
    degree: usize,
    periodic: bool,
    period: Option<f64>,
) -> Result<(Array1<f64>, usize), String> {
    if !periodic {
        let internal_knots = match request {
            PositionBasisLocations::Given(knots) => return Ok((knots, degree)),
            PositionBasisLocations::Count(count) => count,
            PositionBasisLocations::Default => heuristic_knots_for_column(t),
        };
        let auto = auto_knot_vector_1d_quantile(t, internal_knots, degree)
            .map_err(|err| err.to_string())?;
        return Ok((auto.knots, auto.degree));
    }
    let num_basis = match request {
        PositionBasisLocations::Given(grid) => {
            if period.is_none() {
                return Err(
                    "periodic B-spline position fits require a finite positive period".to_string(),
                );
            }
            return Ok((grid, degree));
        }
        PositionBasisLocations::Count(count) => count,
        PositionBasisLocations::Default => {
            default_cyclic_basis_dim(heuristic_knots_for_column(t), degree)
        }
    };
    if num_basis < degree + 1 {
        return Err(format!(
            "periodic B-spline position basis count must be at least degree + 1 \
             (got {num_basis} for degree {degree})"
        ));
    }
    let Some(period) = period else {
        return Err("periodic B-spline position fits require a finite positive period".to_string());
    };
    let origin = t.iter().copied().fold(f64::INFINITY, f64::min);
    let end = origin + period;
    // `i · step + origin`, closing exactly on `end`: the evaluation order a
    // uniform `linspace(origin, end, num_basis + 1)` uses.
    let step = (end - origin) / num_basis as f64;
    let grid = Array1::from_iter((0..=num_basis).map(|i| {
        if i == num_basis {
            end
        } else {
            i as f64 * step + origin
        }
    }));
    Ok((grid, degree))
}

/// The centers of a Duchon position basis.
fn duchon_centers(
    t: ArrayView1<'_, f64>,
    request: PositionBasisLocations,
) -> Result<Array1<f64>, String> {
    let count = match request {
        PositionBasisLocations::Given(centers) => return Ok(centers),
        PositionBasisLocations::Count(count) => {
            if count < 2 {
                return Err(format!(
                    "knots_or_centers: integer count must be >= 2, got {count}"
                ));
            }
            count
        }
        PositionBasisLocations::Default => default_univariate_duchon_center_count(t),
    };
    auto_centers_1d_equal_mass(t, count).map_err(|err| err.to_string())
}

/// The center count the formula front door gives `duchon(x)` on the column `t`
/// with the cubic default null space: its default Duchon count, floored at the
/// basis dimension of the default open `s(x)` on the same data (#1867).
fn default_univariate_duchon_center_count(t: ArrayView1<'_, f64>) -> usize {
    let n = t.len();
    let (nullspace_order, _) = duchon_cubic_default(1);
    let polynomial_cols =
        duchon_nullspace_dimension(1, duchon_p_from_nullspace_order(nullspace_order) - 1);
    default_duchon_center_count(
        n,
        1,
        default_num_centers(n, 1),
        polynomial_cols,
        univariate_spline_basis_dim(t),
    )
}

/// The wrap of a periodic Duchon basis given no period: the center span plus
/// one mean center spacing.
fn duchon_wrap_period(centers: ArrayView1<'_, f64>) -> Result<f64, String> {
    if centers.len() < 2 {
        return Err(format!(
            "a periodic Duchon position basis given no period needs at least two centers to \
             derive its wrap; got {}",
            centers.len()
        ));
    }
    let low = centers.iter().copied().fold(f64::INFINITY, f64::min);
    let high = centers.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let span = high - low;
    Ok(span + span / (centers.len() - 1) as f64)
}

/// Check a position basis's period against its knots or centers.
///
/// The period is the domain WRAP, not the sample or center span: centers on a
/// half-open grid `[start, start + period)` span only `period − one spacing`,
/// so the only constraint is that every location fits inside one period
/// (gam#580). A periodic B-spline needs an explicit period; a periodic Duchon
/// basis derives one when none is given; an open basis takes none.
pub fn validate_position_period(
    label: &str,
    knots_or_centers: ArrayView1<'_, f64>,
    periodic: bool,
    period: Option<f64>,
) -> Result<(), String> {
    if periodic {
        let left = knots_or_centers
            .iter()
            .fold(f64::INFINITY, |a, &b| a.min(b));
        let right = knots_or_centers
            .iter()
            .fold(f64::NEG_INFINITY, |a, &b| a.max(b));
        if !left.is_finite() || !right.is_finite() || left >= right {
            return Err(format!(
                "{label} periodic support must have increasing finite endpoints"
            ));
        }
        let implied = right - left;
        if let Some(period) = period {
            if !period.is_finite() || period <= 0.0 {
                return Err(format!(
                    "{label} period must be finite and positive; got {period}"
                ));
            }
            if period < implied - 1.0e-10 * implied.max(1.0) {
                return Err(format!(
                    "{label} explicit period ({period}) is smaller than the center span \
                     ({implied}); every center must lie within a single period"
                ));
            }
        } else if label != "duchon" {
            return Err(format!(
                "{label} periodic position basis requires an explicit period"
            ));
        }
    } else if period.is_some() {
        return Err(format!("{label} period is only valid when periodic=true"));
    }
    Ok(())
}

/// The single-λ penalty of a resolved position basis.
fn position_penalty(
    kind: PositionBasisKind,
    request: PositionPenaltyRequest,
    locations: ArrayView1<'_, f64>,
    order: usize,
    period: Option<f64>,
) -> Result<Array2<f64>, String> {
    let name = match request {
        PositionPenaltyRequest::Given(matrix) => {
            if matrix.is_empty() {
                return Err("penalty cannot be empty".to_string());
            }
            if matrix.iter().any(|value| !value.is_finite()) {
                return Err("penalty must contain only finite values".to_string());
            }
            return Ok(matrix);
        }
        PositionPenaltyRequest::Canonical => None,
        PositionPenaltyRequest::Named(name) => Some(name),
    };
    let normalized = name
        .as_deref()
        .map(|name| name.trim().to_ascii_lowercase().replace('_', "-"));
    match kind {
        PositionBasisKind::Duchon => {
            match normalized.as_deref() {
                None
                | Some(
                    "function-norm" | "functionnorm" | "rkhs" | "smoothness" | "bending-energy"
                    | "bendingenergy",
                ) => {}
                Some("triple-operator" | "tripleoperator" | "operator") => {
                    return Err(
                        "the triple-operator (amplitude + slope + curvature) penalty has THREE \
                         independent REML smoothing parameters and cannot be collapsed into the \
                         single-λ position helper. Fit it through the formula API \
                         (gamfit.smooth.Duchon / basis_kind='duchon'), which routes each \
                         operator to its own λ."
                            .to_string(),
                    );
                }
                Some(_) => {
                    return Err(format!(
                        "unsupported Duchon penalty {:?}",
                        name.as_deref().unwrap_or_default()
                    ));
                }
            }
            let periodic = period.is_some();
            let (nullspace_order, power) = duchon_cubic_default_with_periodicity(1, periodic);
            let centers = locations.insert_axis(Axis(1));
            duchon_function_norm_penalty(
                centers,
                None,
                nullspace_order,
                power,
                &[periodic],
                period,
            )
            .map_err(|err| err.to_string())
        }
        PositionBasisKind::BSpline => {
            match normalized.as_deref() {
                None | Some("roughness" | "bending-energy" | "bendingenergy") => {}
                Some(_) => {
                    return Err(format!(
                        "unsupported B-spline penalty {:?}",
                        name.as_deref().unwrap_or_default()
                    ));
                }
            }
            match period {
                Some(period) => {
                    if locations.len() < 2 {
                        return Err(
                            "a periodic B-spline grid must contain at least start and end"
                                .to_string(),
                        );
                    }
                    cyclic_bspline_derivative_penalty_matrix(
                        order,
                        locations.len() - 1,
                        period,
                        DEFAULT_PENALTY_ORDER,
                    )
                    .map_err(|err| err.to_string())
                }
                None => {
                    if locations.len() <= order + 1 {
                        return Err(format!(
                            "knot vector is too short for degree={order}: got {} knots",
                            locations.len()
                        ));
                    }
                    bspline_derivative_penalty_matrix(locations, order, DEFAULT_PENALTY_ORDER)
                        .map_err(|err| format!("failed to build smoothness penalty: {err}"))
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn positions() -> Array1<f64> {
        Array1::from_iter((0..40).map(|i| 0.5 + 0.45 * (i as f64 * 1.7).sin()))
    }

    /// #2899 P4: an omitted basis size takes the formula front door's default for
    /// the same kind on the same data, not a separate binding default.
    #[test]
    fn omitted_sizes_take_the_formula_defaults_2899() {
        let t = positions();
        let open = resolve_position_basis(
            t.view(),
            None,
            PositionBasisLocations::Default,
            PositionPenaltyRequest::Canonical,
            None,
            false,
            None,
        )
        .expect("default open B-spline");
        let internal = heuristic_knots_for_column(t.view());
        assert_eq!(open.kind, PositionBasisKind::BSpline);
        assert_eq!(open.display_kind, "bspline");
        assert_eq!(open.order, DEFAULT_BSPLINE_DEGREE);
        assert_eq!(open.locations.len(), internal + 2 * (DEFAULT_BSPLINE_DEGREE + 1));
        assert_eq!(open.penalty.nrows(), internal + DEFAULT_BSPLINE_DEGREE + 1);

        let cyclic = resolve_position_basis(
            t.view(),
            Some("bspline"),
            PositionBasisLocations::Default,
            PositionPenaltyRequest::Canonical,
            None,
            true,
            Some(1.0),
        )
        .expect("default cyclic B-spline");
        let num_basis = default_cyclic_basis_dim(internal, DEFAULT_BSPLINE_DEGREE);
        assert_eq!(cyclic.locations.len(), num_basis + 1);
        assert_eq!(cyclic.penalty.nrows(), num_basis);
        let origin = t.iter().copied().fold(f64::INFINITY, f64::min);
        assert_eq!(cyclic.locations[0], origin);
        assert_eq!(cyclic.locations[num_basis], origin + 1.0);

        let duchon = resolve_position_basis(
            t.view(),
            Some("Duchon"),
            PositionBasisLocations::Default,
            PositionPenaltyRequest::Canonical,
            None,
            false,
            None,
        )
        .expect("default Duchon");
        assert_eq!(duchon.kind, PositionBasisKind::Duchon);
        assert_eq!(duchon.display_kind, "Duchon");
        assert_eq!(duchon.order, 2);
        assert_eq!(duchon.locations.len(), default_univariate_duchon_center_count(t.view()));
        assert!(
            duchon.locations.len() >= univariate_spline_basis_dim(t.view()),
            "the 1-D Duchon default is floored at the open s(x) dimension (#1867)"
        );
    }

    /// #2899 P4: the 1-D thin-plate spline is Duchon `m = 2`, and a periodic
    /// Duchon basis without a period shares ONE derived wrap between its basis
    /// and its penalty (the binding used to build the thin-plate penalty at no
    /// period at all).
    #[test]
    fn thin_plate_is_duchon_and_shares_the_derived_wrap_2899() {
        let t = positions();
        let request = |kind| {
            resolve_position_basis(
                t.view(),
                Some(kind),
                PositionBasisLocations::Count(8),
                PositionPenaltyRequest::Canonical,
                None,
                true,
                None,
            )
        };
        let duchon = request("duchon").expect("periodic Duchon");
        let thin_plate = request("thin_plate").expect("periodic thin-plate");
        assert_eq!(thin_plate.kind, PositionBasisKind::Duchon);
        assert_eq!(thin_plate.order, duchon.order);
        assert_eq!(thin_plate.locations, duchon.locations);
        assert_eq!(thin_plate.penalty, duchon.penalty);
        let span = duchon.locations[7] - duchon.locations[0];
        assert_eq!(duchon.period, Some(span + span / 7.0));
    }

    /// #2899 P4: requests the fit cannot honour are refused by name.
    #[test]
    fn unfit_requests_are_refused_by_name_2899() {
        let t = positions();
        let refusal = |kind: &str,
                       locations: PositionBasisLocations,
                       penalty: PositionPenaltyRequest,
                       periodic: bool,
                       period: Option<f64>| {
            resolve_position_basis(
                t.view(),
                Some(kind),
                locations,
                penalty,
                None,
                periodic,
                period,
            )
            .expect_err("the request must be refused")
        };
        assert!(
            refusal(
                "bspline",
                PositionBasisLocations::Count(6),
                PositionPenaltyRequest::Canonical,
                true,
                None,
            )
            .contains("require a finite positive period")
        );
        assert!(
            refusal(
                "bspline",
                PositionBasisLocations::Count(3),
                PositionPenaltyRequest::Canonical,
                true,
                Some(1.0),
            )
            .contains("at least degree + 1")
        );
        assert!(
            refusal(
                "duchon",
                PositionBasisLocations::Default,
                PositionPenaltyRequest::Named("triple-operator".to_string()),
                false,
                None,
            )
            .contains("THREE independent REML smoothing parameters")
        );
        assert!(
            refusal(
                "duchon_multipenalty",
                PositionBasisLocations::Default,
                PositionPenaltyRequest::Canonical,
                false,
                None,
            )
            .contains("multi-λ")
        );
        assert!(
            refusal(
                "bspline",
                PositionBasisLocations::Default,
                PositionPenaltyRequest::Named("function-norm".to_string()),
                false,
                None,
            )
            .contains("unsupported B-spline penalty")
        );
        assert!(
            refusal(
                "bspline",
                PositionBasisLocations::Default,
                PositionPenaltyRequest::Canonical,
                false,
                Some(1.0),
            )
            .contains("only valid when periodic=true")
        );
    }
}
