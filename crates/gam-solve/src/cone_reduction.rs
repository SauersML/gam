//! Geometry of a cone-truncated Laplace posterior whose reduced precision is
//! INDEFINITE.
//!
//! # The object
//!
//! After the `#2442` reparameterization a location-scale fit with an indefinite
//! ambient Hessian leaves
//!
//! ```text
//! π(w) ∝ exp(−½ wᵀ M w) · 1{w ≥ ℓ},     In(M) = (n−1, 0, 1)
//! ```
//!
//! `M` is not a precision matrix: it has exactly one negative eigenvalue, so
//! `M⁻¹` is not a covariance and this law is NOT a truncated Gaussian. It is
//! normalizable exactly when `M` is strictly copositive on the nonnegative
//! orthant, because along every feasible ray `d ≥ 0` the exponent grows like
//! `½ t² dᵀMd` and copositivity is the statement that `dᵀMd > 0` there.
//!
//! # Why the origin has to move
//!
//! Everything downstream is expressed as an offset from the CONSTRAINED MODE,
//! not from `w = 0` or from `w = ℓ`. That is not presentation. On the fixture
//! this module was built against, the ambient centre lies outside the feasible
//! set and the integrand's peak over that set is `exp(−513.82)`; carried in the
//! reduction's natural origin, every downstream conditional sits about thirty
//! posterior standard deviations outside the feasible region, and a cubature
//! asked for such a probability returns a number that climbs monotonically with
//! its node count instead of converging (measured: +74 log units from `2¹⁰` to
//! `2¹⁸` nodes, still climbing). Re-centred, the same quantities are ordinary.
//!
//! # Everything here is exact
//!
//! Both searches are finite face enumerations rather than iterative solves:
//!
//! * `min wᵀMw` over the simplex is attained at a stationary point in the
//!   relative interior of some face, so enumerating all `2ⁿ − 1` supports plus
//!   the vertices decides copositivity exactly;
//! * the constrained minimiser of `½xᵀMx + cᵀx` over `x ≥ 0` satisfies, on its
//!   free set `F`, `M_FF x_F = −c_F` with `x_F ≥ 0`, `(Mx + c)_A ≥ 0` on the
//!   active set, and `M_FF ⪰ 0`; enumerating supports and keeping the feasible
//!   KKT point of least value is therefore exact.
//!
//! Exactness is worth the `2ⁿ` because `n` is the number of RETAINED constraint
//! rows — six on the motivating fixture — and because the multiplier vector
//! `g = Mx* + c` is consumed downstream, where an optimiser's tolerance would
//! become the quadrature's error floor.

use ndarray::{Array1, Array2, ArrayView2};
use serde::{Deserialize, Serialize};

/// Inertia `(positive, zero, negative)` of a symmetric matrix.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Inertia {
    pub positive: usize,
    pub zero: usize,
    pub negative: usize,
}

/// Symmetric inertia by `LDLᵀ` with symmetric pivoting.
///
/// Sylvester's law of inertia makes the pivot signs the inertia, so this needs
/// no eigensolver. Diagonal pivoting keeps it well posed for the indefinite
/// case, which is the case this module exists for.
pub(crate) fn symmetric_inertia(matrix: ArrayView2<'_, f64>, tolerance: f64) -> Result<Inertia, String> {
    let n = matrix.nrows();
    if matrix.ncols() != n {
        return Err(format!(
            "inertia needs a square matrix, got {}x{}",
            matrix.nrows(),
            matrix.ncols()
        ));
    }
    let mut work = matrix.to_owned();
    let scale = work
        .iter()
        .fold(0.0f64, |worst, value| worst.max(value.abs()))
        .max(1.0);
    let floor = tolerance * scale;
    let mut remaining: Vec<usize> = (0..n).collect();
    let mut inertia = Inertia {
        positive: 0,
        zero: 0,
        negative: 0,
    };
    while !remaining.is_empty() {
        // Pivot on the largest-magnitude remaining diagonal entry.
        let (position, &pivot_index) = remaining
            .iter()
            .enumerate()
            .max_by(|left, right| {
                work[[*left.1, *left.1]]
                    .abs()
                    .partial_cmp(&work[[*right.1, *right.1]].abs())
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .ok_or_else(|| "inertia pivot selection found no candidate".to_string())?;
        let pivot = work[[pivot_index, pivot_index]];
        if !pivot.is_finite() {
            return Err(format!("inertia pivot {pivot_index} is not finite"));
        }
        if pivot.abs() <= floor {
            // The whole remaining block is numerically zero on its diagonal. A
            // nonzero off-diagonal here would be a 2x2 block; refuse rather
            // than guess, since callers of this module treat the inertia as a
            // certificate.
            for &i in &remaining {
                for &j in &remaining {
                    if i != j && work[[i, j]].abs() > floor {
                        return Err(format!(
                            "inertia needs a 2x2 pivot at ({i},{j}); the matrix is not \
                             diagonally pivotable at tolerance {tolerance:.3e}"
                        ));
                    }
                }
            }
            inertia.zero += remaining.len();
            break;
        }
        if pivot > 0.0 {
            inertia.positive += 1;
        } else {
            inertia.negative += 1;
        }
        remaining.remove(position);
        let rest = remaining.clone();
        for &i in &rest {
            let factor = work[[i, pivot_index]] / pivot;
            if factor == 0.0 {
                continue;
            }
            for &j in &rest {
                work[[i, j]] -= factor * work[[pivot_index, j]];
            }
        }
        for &i in &rest {
            work[[i, pivot_index]] = 0.0;
            work[[pivot_index, i]] = 0.0;
        }
    }
    Ok(inertia)
}

/// What a cone-truncated posterior's properness was decided against.
///
/// Every field is a measured quantity rather than a summary, because the point
/// of this type is that a decline can name what it declined on.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ConeProperness {
    /// The reduced precision `M` on the recession cone's normal coordinates
    /// `w = Ad`: `wᵀMw` is the STATIONARY value of `dᵀHd` on `{d : Ad = w}`, which
    /// is its minimum exactly when `H` is positive definite on `null(A)`.
    pub reduced: Array2<f64>,
    /// `In(H)` — the ambient precision's inertia. A constrained mode is not
    /// obliged to make this all-positive.
    pub ambient_inertia: Inertia,
    /// `In(M)`.
    pub reduced_inertia: Inertia,
    /// `In(ZᵀHZ)` for `Z` a basis of `null(A)`, obtained from Haynsworth
    /// additivity `In(H) = In(ZᵀHZ) + In(M)` rather than by forming `Z`.
    /// `null(A)` is the recession cone's LINEALITY space — both `±d` are
    /// feasible there — so a negative direction here is impropriety. A zero is a
    /// pivot at or below the inertia tolerance, which cannot separate a true null
    /// from small positive curvature, so it leaves properness undecided.
    pub lineality_inertia: Inertia,
    /// `min wᵀMw` over the unit simplex. `Some(v)` with `v > 0` is a proof that
    /// the cone-truncated posterior is proper; `Some(v)` with `v <= 0` is a
    /// proof that it is improper. `None` means the face is too wide for the
    /// exact `2^q` enumeration, so properness is undecided — never assumed.
    pub copositive_minimum: Option<f64>,
    /// The simplex point attaining [`Self::copositive_minimum`]. Where that minimum
    /// is non-positive, its support names the constraint rows whose normal
    /// coordinates carry the feasible direction of non-positive curvature (#979).
    /// Absent from certificates persisted before it existed.
    #[serde(default)]
    pub copositive_minimizer: Option<Array1<f64>>,
}

impl ConeProperness {
    /// `Some(true)`/`Some(false)` when properness is PROVED either way, `None`
    /// when it is undecided: the face is too wide for the exact enumeration, or
    /// `In(ZᵀHZ)` has a direction at or below the inertia tolerance and no
    /// feasible direction already proves the posterior improper. Undecided is
    /// deliberately not folded into either answer.
    pub fn is_proper(&self) -> Option<bool> {
        if self.lineality_inertia.negative > 0 {
            return Some(false);
        }
        if self.copositive_minimum.is_some_and(|minimum| minimum <= 0.0) {
            return Some(false);
        }
        if self.lineality_inertia.zero > 0 {
            return None;
        }
        self.copositive_minimum.map(|minimum| minimum > 0.0)
    }

    /// One line naming every quantity the verdict was decided against, for a
    /// refusal or a decline to carry.
    pub fn summary(&self) -> String {
        let verdict = match self.is_proper() {
            Some(true) => "PROPER".to_string(),
            Some(false) => "IMPROPER".to_string(),
            None if self.lineality_inertia.zero > 0 => format!(
                "UNDECIDED ({} direction(s) of null(A) sit at or below the inertia tolerance, \
                 which cannot separate a true null from small positive curvature)",
                self.lineality_inertia.zero
            ),
            None => format!(
                "UNDECIDED (the exact enumeration is out of range at q = {})",
                self.reduced.nrows()
            ),
        };
        let copositive = match self.copositive_minimum {
            Some(minimum) => format!("{minimum:.6e}"),
            None => "not enumerated".to_string(),
        };
        // Where the simplex minimum proves impropriety, name the constraint rows the
        // feasible direction of non-positive curvature moves along (#979).
        let support = match (self.copositive_minimum, self.copositive_minimizer.as_ref()) {
            (Some(minimum), Some(point)) if minimum <= 0.0 => {
                let rows: Vec<usize> = (0..point.len()).filter(|&row| point[row] > 0.0).collect();
                format!(", attained along constraint row(s) {rows:?}")
            }
            _ => String::new(),
        };
        format!(
            "cone-truncated posterior is {verdict}: In(H) = ({}, {}, {}), \
             In(M) = ({}, {}, {}), In(ZᵀHZ) = ({}, {}, {}) on null(A), \
             min wᵀMw over the simplex = {copositive}{support}",
            self.ambient_inertia.positive,
            self.ambient_inertia.zero,
            self.ambient_inertia.negative,
            self.reduced_inertia.positive,
            self.reduced_inertia.zero,
            self.reduced_inertia.negative,
            self.lineality_inertia.positive,
            self.lineality_inertia.zero,
            self.lineality_inertia.negative,
        )
    }
}

/// The reduced precision `M` on the recession cone's normal coordinates.
///
/// For a feasible set `{d : Ad ≥ b}` the recession cone is `{d : Ad ≥ 0}`, and
/// splitting `d = Zt + Nw` with `Z` a basis of `null(A)` and `w = Ad` leaves the
/// `w`-marginal precision as the Schur complement
/// `M = NᵀHN − NᵀHZ(ZᵀHZ)⁻¹ZᵀHN`, from the defining variational identity
///
/// ```text
/// wᵀMw = stat{ dᵀHd : Ad = w }
/// ```
///
/// — the stationary value, which is the MINIMUM exactly when `ZᵀHZ ≻ 0` and is
/// the algebraic Schur complement either way, so this route does not presuppose
/// the condition the certificate above it goes on to test. The reason this
/// module exists is that `H` is INDEFINITE, so `Σ = H⁻¹` may not be a covariance
/// and `M = (AH⁻¹Aᵀ)⁻¹` — the identity that holds when `H ≻ 0` — cannot be
/// evaluated by inverting `H`. Only `ZᵀHZ` is inverted, and it has to be
/// nonsingular for `M` to exist at all.
///
/// The reduction exists exactly when `A` has full row rank and `H` is
/// nonsingular on `null(A)`. Each condition is decided in its own units, and a
/// failure refuses by naming which of the two the face broke:
///
/// * the thin SVD `A = UΣV₁ᵀ` must resolve all `q` singular values above its
///   backward-error band ([`factor_singular_band`]), or the rows are dependent;
/// * every eigenvalue of `ZᵀHZ`, for an orthonormal basis `Z` of `null(A)`, must
///   lie outside the ambient precision's spectral rounding band
///   ([`symmetric_spectrum_rounding_band`] of `H`), the error that forming `ZᵀHZ`
///   through a computed `Z` can leave, since `Z`'s rounding leaks `H`'s largest
///   curvature into it. An eigenvalue inside the band is a lineality direction
///   without resolved curvature, which is itself impropriety.
///
/// With `d = V₁u + Zt` the constraint values are `w = UΣu`, so from the
/// eigenpairs `(Λ, W)` of `ZᵀHZ`
///
/// ```text
/// M = (Σ⁻¹Uᵀ)ᵀ (V₁ᵀHV₁ − (V₁ᵀHZW) Λ⁻¹ (V₁ᵀHZW)ᵀ) (Σ⁻¹Uᵀ),
/// ```
///
/// symmetric by construction.
///
/// gam#3008: this used to eliminate the saddle system `[[H, Aᵀ],[A, 0]]` by
/// Gaussian elimination against a pivot floor of `1e-12·max|entry|`. The
/// trailing pivots of that elimination are entries of `−AH⁻¹Aᵀ`, in units of
/// `H⁻¹`, while the floor was in units of `H`. A survival time block with
/// curvature near `4.3e6` put the floor at `4.3e-6` and its own pivots near
/// `1/4.3e6`, so a nonsingular face was refused as singular and a certified
/// fit's posterior moments were declined.
pub(crate) fn reduced_cone_precision(
    hessian: ArrayView2<'_, f64>,
    constraints: ArrayView2<'_, f64>,
) -> Result<Array2<f64>, String> {
    use gam_linalg::faer_ndarray::{FaerEigh, FaerSvd};
    use gam_linalg::roundoff::{factor_singular_band, symmetric_spectrum_rounding_band};

    let p = hessian.nrows();
    if hessian.ncols() != p {
        return Err(format!(
            "cone reduction needs a square ambient precision, got {}x{}",
            hessian.nrows(),
            hessian.ncols()
        ));
    }
    let q = constraints.nrows();
    if constraints.ncols() != p {
        return Err(format!(
            "cone reduction: the ambient precision is {p}x{p} but the constraint rows have \
             {} columns",
            constraints.ncols()
        ));
    }
    if q == 0 {
        return Err(
            "cone reduction needs at least one inequality row; with none the recession cone \
             is all of R^p and properness is just positive definiteness of H"
                .to_string(),
        );
    }
    if q > p {
        return Err(format!(
            "cone reduction: {q} constraint rows in {p} dimensions cannot be independent, so \
             the reduction's coordinates are not well defined; canonicalize the face to an \
             independent row basis first"
        ));
    }
    if hessian
        .iter()
        .chain(constraints.iter())
        .any(|value| !value.is_finite())
    {
        return Err(
            "cone reduction: the ambient precision or the constraint rows carry a non-finite \
             entry, so neither can be trusted"
                .to_string(),
        );
    }
    let mut symmetric = hessian.to_owned();
    for row in 0..p {
        for column in (row + 1)..p {
            let averaged = 0.5 * (symmetric[[row, column]] + symmetric[[column, row]]);
            symmetric[[row, column]] = averaged;
            symmetric[[column, row]] = averaged;
        }
    }

    let (left_vectors, singular, right_transposed) = constraints
        .svd(true, true)
        .map_err(|error| format!("cone reduction: the constraint rows' SVD failed: {error}"))?;
    let (Some(left_vectors), Some(right_transposed)) = (left_vectors, right_transposed) else {
        return Err(
            "cone reduction: the constraint rows' SVD returned no singular vectors".to_string(),
        );
    };
    let sigma_max = singular
        .iter()
        .fold(0.0f64, |worst, value| worst.max(value.abs()));
    let rank_band = factor_singular_band(q, p, sigma_max);
    let resolved = singular.iter().filter(|&&value| value > rank_band).count();
    if resolved < q {
        return Err(format!(
            "cone reduction: the {q} constraint rows are dependent: only {resolved} of their \
             singular values {singular:?} are resolved above the SVD's rounding band \
             {rank_band:.3e}, so the reduction's coordinates are not well defined; canonicalize \
             the face to an independent row basis first"
        ));
    }
    // `V₁`, `p × q`: an orthonormal basis of the row space.
    let row_space = right_transposed.t().to_owned();
    let row_curvature = row_space.t().dot(&symmetric).dot(&row_space);
    let mut schur = Array2::<f64>::zeros((q, q));
    for row in 0..q {
        for column in 0..q {
            schur[[row, column]] =
                0.5 * (row_curvature[[row, column]] + row_curvature[[column, row]]);
        }
    }

    let lineality_dimension = p - q;
    if lineality_dimension > 0 {
        // `I − V₁V₁ᵀ` is the orthogonal projector onto `null(A)`: its spectrum is
        // `q` zeros and `p − q` ones, so the eigenvectors of its `p − q` largest
        // eigenvalues are an orthonormal basis `Z` of the lineality space.
        let projector = Array2::<f64>::eye(p) - row_space.dot(&row_space.t());
        let (projector_values, projector_vectors) = projector
            .eigh(faer::Side::Lower)
            .map_err(|error| {
                format!(
                    "cone reduction: the null-space projector's eigendecomposition failed: \
                     {error}"
                )
            })?;
        let mut order: Vec<usize> = (0..p).collect();
        order.sort_by(|&first, &second| {
            projector_values[first].total_cmp(&projector_values[second])
        });
        let mut lineality_basis = Array2::<f64>::zeros((p, lineality_dimension));
        for (target, &source) in order[q..].iter().enumerate() {
            lineality_basis
                .column_mut(target)
                .assign(&projector_vectors.column(source));
        }
        let (ambient_values, _) = symmetric.eigh(faer::Side::Lower).map_err(|error| {
            format!("cone reduction: the ambient precision's eigendecomposition failed: {error}")
        })?;
        let ambient_spectrum = ambient_values
            .as_slice()
            .ok_or_else(|| "cone reduction: the ambient spectrum is not contiguous".to_string())?;
        let lineality_band = symmetric_spectrum_rounding_band(ambient_spectrum);
        let lineality_curvature = lineality_basis.t().dot(&symmetric).dot(&lineality_basis);
        let (lineality_values, lineality_vectors) = lineality_curvature
            .eigh(faer::Side::Lower)
            .map_err(|error| {
                format!("cone reduction: ZᵀHZ's eigendecomposition failed: {error}")
            })?;
        if let Some(unresolved) = lineality_values
            .iter()
            .copied()
            .find(|value| value.abs() <= lineality_band)
        {
            return Err(format!(
                "cone reduction: H is singular on null(A): ZᵀHZ carries the eigenvalue \
                 {unresolved:.3e}, inside the ambient precision's rounding band \
                 {lineality_band:.3e}. null(A) is the recession cone's lineality space, so a \
                 direction there without resolved curvature is itself impropriety"
            ));
        }
        let coupling = row_space
            .t()
            .dot(&symmetric)
            .dot(&lineality_basis)
            .dot(&lineality_vectors);
        for row in 0..q {
            for column in 0..q {
                let mut correction = 0.0;
                for k in 0..lineality_dimension {
                    correction +=
                        coupling[[row, k]] * coupling[[column, k]] / lineality_values[k];
                }
                schur[[row, column]] -= correction;
            }
        }
    }

    // `w = UΣu`, so `u = Σ⁻¹Uᵀw` and `M = (Σ⁻¹Uᵀ)ᵀ S (Σ⁻¹Uᵀ)`.
    let mut lift = left_vectors.t().to_owned();
    for (index, mut row) in lift.rows_mut().into_iter().enumerate() {
        row /= singular[index];
    }
    let mut reduced = lift.t().dot(&schur).dot(&lift);
    for row in 0..q {
        for column in (row + 1)..q {
            let averaged = 0.5 * (reduced[[row, column]] + reduced[[column, row]]);
            reduced[[row, column]] = averaged;
            reduced[[column, row]] = averaged;
        }
    }
    Ok(reduced)
}

/// Decide whether a cone-truncated Laplace posterior is proper, exactly.
///
/// The feasible set is `{d : Ad ≥ b}`, so `exp(−½dᵀHd − …)` is normalizable over
/// it exactly when `dᵀHd > 0` for every nonzero `d` in the recession cone
/// `{Ad ≥ 0}` — strict copositivity of `H` on that cone, NOT `H ≻ 0`. In the
/// `d = Zt + Nw` coordinates that separates into two conditions, and this
/// returns both:
///
/// * `ZᵀHZ ≻ 0`, i.e. properness along the cone's lineality space `null(A)`,
///   where both `±d` are feasible so there is nothing for a constraint to do;
/// * `M` strictly copositive on `{w ≥ 0}`, decided exactly by face enumeration.
///
/// `In(ZᵀHZ)` comes from Haynsworth additivity — `In(H) = In(ZᵀHZ) + In(M)` —
/// so no null-space basis is ever formed.
pub(crate) fn cone_properness_certificate(
    hessian: ArrayView2<'_, f64>,
    constraints: ArrayView2<'_, f64>,
    tolerance: f64,
) -> Result<ConeProperness, String> {
    let reduced = reduced_cone_precision(hessian, constraints)?;
    let ambient_inertia = symmetric_inertia(hessian, tolerance)
        .map_err(|error| format!("ambient precision inertia: {error}"))?;
    let reduced_inertia = symmetric_inertia(reduced.view(), tolerance)
        .map_err(|error| format!("reduced precision inertia: {error}"))?;
    let (positive, zero, negative) = (
        ambient_inertia.positive.checked_sub(reduced_inertia.positive),
        ambient_inertia.zero.checked_sub(reduced_inertia.zero),
        ambient_inertia.negative.checked_sub(reduced_inertia.negative),
    );
    let (Some(positive), Some(zero), Some(negative)) = (positive, zero, negative) else {
        return Err(format!(
            "Haynsworth additivity In(H) = In(ZᵀHZ) + In(M) is violated: In(H) = ({}, {}, {}) \
             cannot contain In(M) = ({}, {}, {}). One of the two inertias is wrong, so the \
             lineality verdict has no basis",
            ambient_inertia.positive,
            ambient_inertia.zero,
            ambient_inertia.negative,
            reduced_inertia.positive,
            reduced_inertia.zero,
            reduced_inertia.negative,
        ));
    };
    let lineality_inertia = Inertia {
        positive,
        zero,
        negative,
    };
    let expected = hessian.nrows() - reduced.nrows();
    let realized = positive + zero + negative;
    if realized != expected {
        return Err(format!(
            "the lineality inertia has {realized} directions where null(A) has {expected}; \
             In(H) − In(M) is not an inertia of the right dimension"
        ));
    }
    // Only enumerate when the answer would be exact. `copositive_simplex_minimum`
    // owns that range, and an out-of-range face reports UNDECIDED rather than
    // borrowing a cheaper sufficient condition and calling it a proof.
    let (copositive_minimum, copositive_minimizer) =
        match copositive_simplex_minimum(reduced.view()) {
            Ok((minimum, point)) => (Some(minimum), Some(point)),
            Err(out_of_range) => {
                log::debug!("[cone properness] exact copositivity not enumerated: {out_of_range}");
                (None, None)
            }
        };
    Ok(ConeProperness {
        reduced,
        ambient_inertia,
        reduced_inertia,
        lineality_inertia,
        copositive_minimum,
        copositive_minimizer,
    })
}

/// Solve `A y = b` for a small dense `A` by Gaussian elimination with partial
/// pivoting. Returns `None` when a pivot falls below the floor, which the
/// callers read as "this face is degenerate, skip it" rather than as an error —
/// a singular face carries no isolated stationary point to compare.
///
/// The name records where it is used, not a requirement: the elimination is a
/// general LU with row pivoting.
fn symmetric_solve(a: &Array2<f64>, b: &Array1<f64>, floor: f64) -> Option<Array1<f64>> {
    let n = a.nrows();
    let mut work = a.clone();
    let mut rhs = b.clone();
    for column in 0..n {
        let mut pivot_row = column;
        let mut best = work[[column, column]].abs();
        for row in (column + 1)..n {
            let candidate = work[[row, column]].abs();
            if candidate > best {
                best = candidate;
                pivot_row = row;
            }
        }
        if !best.is_finite() || best <= floor {
            return None;
        }
        if pivot_row != column {
            for j in 0..n {
                let swap = work[[column, j]];
                work[[column, j]] = work[[pivot_row, j]];
                work[[pivot_row, j]] = swap;
            }
            rhs.swap(column, pivot_row);
        }
        let pivot = work[[column, column]];
        for row in (column + 1)..n {
            let factor = work[[row, column]] / pivot;
            if factor == 0.0 {
                continue;
            }
            for j in column..n {
                work[[row, j]] -= factor * work[[column, j]];
            }
            rhs[row] -= factor * rhs[column];
        }
    }
    let mut solution = Array1::<f64>::zeros(n);
    for row in (0..n).rev() {
        let mut total = rhs[row];
        for column in (row + 1)..n {
            total -= work[[row, column]] * solution[column];
        }
        solution[row] = total / work[[row, row]];
    }
    if solution.iter().any(|value| !value.is_finite()) {
        return None;
    }
    Some(solution)
}

/// Exact minimum of `wᵀMw` over the unit simplex `{w ≥ 0, 1ᵀw = 1}`.
///
/// Strictly positive iff `M` is strictly copositive, which is exactly the
/// condition for `exp(−½wᵀMw)` to be normalizable on a shifted orthant. On the
/// face with support `S` the stationary value is `1/(1ᵀM_SS⁻¹1)`, so enumerating
/// all `2ⁿ − 1` supports and the vertices `M_jj` decides it — no nonconvex QP,
/// and a non-positive answer is a PROOF of impropriety rather than an
/// inconclusive bound.
pub(crate) fn copositive_simplex_minimum(
    matrix: ArrayView2<'_, f64>,
) -> Result<(f64, Array1<f64>), String> {
    let n = matrix.nrows();
    if matrix.ncols() != n {
        return Err(format!(
            "copositivity needs a square matrix, got {}x{}",
            matrix.nrows(),
            matrix.ncols()
        ));
    }
    if n == 0 || n > 20 {
        return Err(format!(
            "exact copositivity enumerates 2^n faces and is meant for a retained \
             constraint face; n = {n} is out of range"
        ));
    }
    let owned = matrix.to_owned();
    let scale = owned
        .iter()
        .fold(0.0f64, |worst, value| worst.max(value.abs()))
        .max(1.0);
    let floor = 1e-12 * scale;
    let mut best = f64::INFINITY;
    let mut best_point = Array1::<f64>::zeros(n);
    for mask in 1u32..(1u32 << n) {
        let support: Vec<usize> = (0..n).filter(|j| mask & (1 << j) != 0).collect();
        let size = support.len();
        let mut block = Array2::<f64>::zeros((size, size));
        for (i, &row) in support.iter().enumerate() {
            for (j, &column) in support.iter().enumerate() {
                block[[i, j]] = owned[[row, column]];
            }
        }
        let ones = Array1::<f64>::ones(size);
        let Some(solution) = symmetric_solve(&block, &ones, floor) else {
            continue;
        };
        let total: f64 = solution.sum();
        if !total.is_finite() || total.abs() <= floor {
            continue;
        }
        let weights = &solution / total;
        if weights.iter().any(|value| *value <= 0.0) {
            continue;
        }
        let value = weights.dot(&block.dot(&weights));
        if value.is_finite() && value < best {
            best = value;
            best_point = Array1::zeros(n);
            for (i, &row) in support.iter().enumerate() {
                best_point[row] = weights[i];
            }
        }
    }
    for j in 0..n {
        if owned[[j, j]] < best {
            best = owned[[j, j]];
            best_point = Array1::zeros(n);
            best_point[j] = 1.0;
        }
    }
    if !best.is_finite() {
        return Err("copositivity enumeration produced no finite face value".to_string());
    }
    Ok((best, best_point))
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    /// The reduced precision `M` of the refusing location-scale fixture on
    /// #2529, taken from the probe dump at `e23e674633b` (`p = 9`, `m = 6`,
    /// blocks `MU ⊕ LOG_SIGMA ⊕ WIGGLE`). Every constant asserted against it
    /// below was produced twice by two lanes on two independent methods.
    const FIXTURE_M: [[f64; 6]; 6] = [
        [2144.265169679624, 1715.134178122592, 1747.5745584612605, 935.098928788, -2.7864165543774675, -0.20985105745649374],
        [1715.134178122592, 2085.4964662263064, 1875.9766836439958, 759.8968208234021, -39.68741458861115, -0.2501447114808116],
        [1747.5745584612605, 1875.9766836439958, 1822.1414523216163, 1123.054947333127, 109.2026621607369, -0.17598452900630168],
        [935.098928788, 759.8968208234021, 1123.054947333127, 938.363436676176, 106.59121181068619, -4.890252117554146],
        [-2.7864165543774675, -39.68741458861115, 109.2026621607369, 106.59121181068619, 23.64370794528972, -21.728482419069984],
        [-0.20985105745649374, -0.2501447114808116, -0.17598452900630168, -4.890252117554146, -21.728482419069984, 57.945174065326796],
    ];
    const FIXTURE_ELL: [f64; 6] = [
        0.41517285129090653,
        -1.8692500719946608,
        2.765160237666297,
        -3.8165670131467633,
        6.59422728766729,
        4.190338688011645,
    ];

    fn fixture() -> (Array2<f64>, Array1<f64>) {
        let mut matrix = Array2::<f64>::zeros((6, 6));
        for (i, row) in FIXTURE_M.iter().enumerate() {
            for (j, value) in row.iter().enumerate() {
                matrix[[i, j]] = *value;
            }
        }
        (matrix, Array1::from_vec(FIXTURE_ELL.to_vec()))
    }

    #[test]
    fn inertia_counts_pivot_signs_rather_than_solving_an_eigenproblem() {
        // Diagonal: the inertia is read straight off.
        let diagonal = array![[3.0, 0.0, 0.0], [0.0, -2.0, 0.0], [0.0, 0.0, 5.0]];
        assert_eq!(
            symmetric_inertia(diagonal.view(), 1e-12).expect("diagonal inertia"),
            Inertia { positive: 2, zero: 0, negative: 1 }
        );
        // A congruence transform must leave the inertia alone — that is
        // Sylvester's law, and it is the whole reason pivot signs are a
        // certificate. `C A Cᵀ` with `C` invertible.
        let c = array![[1.0, 2.0, 0.0], [0.0, 1.0, 3.0], [4.0, 0.0, 1.0]];
        let congruent = c.dot(&diagonal).dot(&c.t());
        assert_eq!(
            symmetric_inertia(congruent.view(), 1e-12).expect("congruent inertia"),
            Inertia { positive: 2, zero: 0, negative: 1 },
            "congruence preserves inertia"
        );
    }

    /// `H⁻¹Aᵀ` one column at a time, for the tests that need an independent
    /// route to `W = AΣAᵀ`. Only ever called on a positive definite `H`.
    fn ambient_solve_against_rows(hessian: &Array2<f64>, constraints: &Array2<f64>) -> Array2<f64> {
        let p = hessian.nrows();
        let q = constraints.nrows();
        let scale = hessian
            .iter()
            .fold(0.0f64, |worst, value| worst.max(value.abs()))
            .max(1.0);
        let mut lifted = Array2::<f64>::zeros((p, q));
        for row in 0..q {
            let rhs = constraints.row(row).to_owned();
            let solution =
                symmetric_solve(hessian, &rhs, 1e-12 * scale).expect("a PD ambient solve");
            for i in 0..p {
                lifted[[i, row]] = solution[i];
            }
        }
        lifted
    }

    #[test]
    fn the_reduced_precision_inverts_the_constraint_normal_covariance_when_the_ambient_is_pd() {
        // `M = (A H⁻¹ Aᵀ)⁻¹` is the identity the #2417 decomposition uses, and it
        // holds only when `H ≻ 0`. So it is exactly the right independent check
        // on the saddle route, which never forms `H⁻¹`: on a PD ambient the two
        // must agree, and the saddle route is then used on ambients where the
        // identity's right-hand side does not exist at all.
        let hessian = array![
            [7.0, 1.0, 0.5, 0.0],
            [1.0, 5.0, -1.0, 0.25],
            [0.5, -1.0, 6.0, 1.5],
            [0.0, 0.25, 1.5, 4.0],
        ];
        let constraints = array![[1.0, 0.0, -1.0, 0.0], [0.0, 2.0, 1.0, -0.5]];
        let reduced = reduced_cone_precision(hessian.view(), constraints.view())
            .expect("the saddle reduction on a PD ambient");
        let lifted = ambient_solve_against_rows(&hessian, &constraints);
        let normal_covariance = constraints.dot(&lifted);
        let product = normal_covariance.dot(&reduced);
        for i in 0..2 {
            for j in 0..2 {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!(
                    (product[[i, j]] - expected).abs() < 1e-10,
                    "(A H⁻¹ Aᵀ) M should be the identity, entry ({i},{j}) was {:.6e}",
                    product[[i, j]]
                );
            }
        }
    }

    #[test]
    fn the_live_reduction_reproduces_the_fixture_reduced_precision_and_its_minimum() {
        // Until now `M` existed here only as 36 constants dumped by a Python
        // probe. This builds an ambient `H` whose reduction IS that matrix and
        // checks that the production route recovers it — so the published
        // copositivity minimum becomes a property of the code, not of a paste.
        //
        // With `A = [I_6 | 0]` the normal coordinates are the first six, the
        // lineality space is the last three, and the reduction is the ordinary
        // Schur complement `H₁₁ − H₁₂H₂₂⁻¹H₂₁`. Choosing `H₂₂ ≻ 0` (its min
        // eigenvalue echoes the measured `+51.4` on `null(A)`) and a nonzero
        // coupling `H₁₂` makes the reduction do real work rather than copy a
        // block.
        let (target, _) = fixture();
        let lineality = array![[51.4, 3.0, -1.0], [3.0, 60.0, 2.0], [-1.0, 2.0, 70.0]];
        let mut coupling = Array2::<f64>::zeros((6, 3));
        for i in 0..6 {
            for j in 0..3 {
                coupling[[i, j]] = ((i + 1) as f64) * 0.5 - ((j + 1) as f64) * 1.25;
            }
        }
        // `H₁₁ = M + H₁₂H₂₂⁻¹H₂₁` reverses the Schur complement exactly.
        let mut lineality_solve = Array2::<f64>::zeros((3, 6));
        for column in 0..6 {
            let rhs = coupling.row(column).to_owned();
            let solution = symmetric_solve(&lineality, &rhs, 1e-12 * 70.0)
                .expect("the PD lineality block is invertible");
            for i in 0..3 {
                lineality_solve[[i, column]] = solution[i];
            }
        }
        let correction = coupling.dot(&lineality_solve);
        let mut hessian = Array2::<f64>::zeros((9, 9));
        hessian
            .slice_mut(ndarray::s![0..6, 0..6])
            .assign(&(&target + &correction));
        hessian.slice_mut(ndarray::s![0..6, 6..9]).assign(&coupling);
        hessian
            .slice_mut(ndarray::s![6..9, 0..6])
            .assign(&coupling.t());
        hessian.slice_mut(ndarray::s![6..9, 6..9]).assign(&lineality);
        let mut constraints = Array2::<f64>::zeros((6, 9));
        for j in 0..6 {
            constraints[[j, j]] = 1.0;
        }

        let certificate = cone_properness_certificate(hessian.view(), constraints.view(), 1e-12)
            .expect("a certificate on an indefinite ambient with a PD lineality block");
        let scale = target
            .iter()
            .fold(0.0f64, |worst, value| worst.max(value.abs()));
        for i in 0..6 {
            for j in 0..6 {
                assert!(
                    (certificate.reduced[[i, j]] - target[[i, j]]).abs() < 1e-8 * scale,
                    "recovered M[{i},{j}] = {:.9e}, expected {:.9e}",
                    certificate.reduced[[i, j]],
                    target[[i, j]]
                );
            }
        }
        assert_eq!(
            certificate.reduced_inertia,
            Inertia {
                positive: 5,
                zero: 0,
                negative: 1
            },
            "In(M) = (5,0,1) survives the round trip through the ambient"
        );
        // The whole point of the Haynsworth route: `null(A)` never gets a basis,
        // yet its inertia comes out right. The ambient built here is indefinite,
        // so this is not the PD case in disguise.
        assert_eq!(
            certificate.lineality_inertia,
            Inertia {
                positive: 3,
                zero: 0,
                negative: 0
            },
            "H is PD on null(A), which is what licenses marginalizing the tangent"
        );
        assert_eq!(certificate.ambient_inertia.negative, 1);
        let minimum = certificate
            .copositive_minimum
            .expect("q = 6 is inside the exact enumeration range");
        assert!(
            (minimum - 6.683215003061817).abs() < 1e-6,
            "the live reduction's copositivity minimum was {minimum:.12e}, expected \
             6.683215003061817"
        );
        assert_eq!(
            certificate.is_proper(),
            Some(true),
            "a copositive M with a PD lineality block is a PROOF of properness"
        );
        let summary = certificate.summary();
        assert!(
            summary.contains("PROPER") && summary.contains("min wᵀMw"),
            "the summary must name the quantity it decided on, got: {summary}"
        );
    }

    #[test]
    fn a_negative_direction_inside_null_a_is_reported_as_impropriety() {
        // The constraint touches only the first coordinate, so `null(A)` carries
        // the other two — and a negative curvature there is a direction along
        // which BOTH `±d` are feasible. No inequality can make that proper, and
        // copositivity of `M` cannot see it, so the lineality inertia has to be
        // the thing that decides.
        let hessian = array![[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, 1.0]];
        let constraints = array![[1.0, 0.0, 0.0]];
        let certificate = cone_properness_certificate(hessian.view(), constraints.view(), 1e-12)
            .expect("a certificate on a lineality-improper ambient");
        assert_eq!(
            certificate.lineality_inertia.negative, 1,
            "the negative direction lands in null(A), not in the normal coordinates"
        );
        assert_eq!(
            certificate.copositive_minimum,
            Some(1.0),
            "M is the 1x1 block [1], so copositivity alone would have said PROPER"
        );
        assert_eq!(
            certificate.is_proper(),
            Some(false),
            "impropriety along the cone's lineality space outranks a copositive M"
        );
        assert!(certificate.summary().contains("IMPROPER"));
        assert!(
            !certificate.summary().contains("attained along"),
            "a direction inside null(A) loads no constraint row, got: {}",
            certificate.summary()
        );
    }

    #[test]
    fn a_nonpositive_simplex_minimum_names_the_constraint_rows_it_is_attained_along() {
        // `A = I` on three coordinates leaves no lineality space, so `M = H` and
        // copositivity alone decides. The leading block has eigenvalues 4 and −2,
        // and its −2 eigenvector lies inside the orthant: `w = (½, ½, 0)` gives
        // `wᵀMw = −1`, the simplex minimum. The third row carries only positive
        // curvature, so the named support must leave it out (#979).
        let hessian = array![[1.0, -3.0, 0.0], [-3.0, 1.0, 0.0], [0.0, 0.0, 2.0]];
        let constraints = Array2::<f64>::eye(3);
        let certificate = cone_properness_certificate(hessian.view(), constraints.view(), 1e-12)
            .expect("a certificate on an orthant-improper ambient");
        assert_eq!(
            certificate.lineality_inertia,
            Inertia { positive: 0, zero: 0, negative: 0 },
            "A = I leaves no lineality space, so the lineality branch cannot decide"
        );
        assert_eq!(
            certificate.ambient_inertia,
            Inertia { positive: 2, zero: 0, negative: 1 }
        );
        let minimum = certificate
            .copositive_minimum
            .expect("q = 3 is inside the exact enumeration range");
        assert!(
            (minimum + 1.0).abs() < 1e-12,
            "min wᵀMw over the simplex was {minimum:.15e}, expected −1"
        );
        let point = certificate
            .copositive_minimizer
            .as_ref()
            .expect("the minimizer accompanies an enumerated minimum");
        assert!(
            (point[0] - 0.5).abs() < 1e-12 && (point[1] - 0.5).abs() < 1e-12 && point[2] == 0.0,
            "the minimizer is (½, ½, 0), got {point:?}"
        );
        assert_eq!(certificate.is_proper(), Some(false));
        let summary = certificate.summary();
        assert!(
            summary.contains("IMPROPER")
                && summary.contains("attained along constraint row(s) [0, 1]"),
            "got: {summary}"
        );
    }

    #[test]
    fn a_lineality_direction_below_the_inertia_tolerance_leaves_properness_undecided() {
        // `symmetric_inertia` counts a pivot at or below `tolerance · max|entry|` as zero, which
        // cannot separate a true null of `ZᵀHZ` from small positive curvature. Here `H` is PD
        // with its smallest lineality direction at 1e-10 of the largest entry, so the truncated
        // posterior is proper. At a √EPS tolerance the certificate cannot resolve that direction
        // and must decline to answer; a tolerance that resolves it proves PROPER.
        let hessian = array![[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1e-10]];
        let constraints = array![[1.0, 0.0, 0.0]];
        let coarse =
            cone_properness_certificate(hessian.view(), constraints.view(), f64::EPSILON.sqrt())
                .expect("a certificate on an ill-conditioned proper cone");
        assert_eq!(
            coarse.lineality_inertia,
            Inertia {
                positive: 1,
                zero: 1,
                negative: 0
            },
            "the 1e-10 direction sits below the √EPS inertia floor"
        );
        assert_eq!(coarse.copositive_minimum, Some(1.0), "M is the 1x1 block [1]");
        assert_eq!(
            coarse.is_proper(),
            None,
            "a sub-tolerance lineality direction is not a proof of impropriety"
        );
        assert!(
            coarse.summary().contains("UNDECIDED") && coarse.summary().contains("inertia tolerance"),
            "got: {}",
            coarse.summary()
        );
        let resolved = cone_properness_certificate(hessian.view(), constraints.view(), 1e-12)
            .expect("a certificate that resolves the small direction");
        assert_eq!(
            resolved.lineality_inertia,
            Inertia {
                positive: 2,
                zero: 0,
                negative: 0
            },
            "resolved, the small direction is positive curvature"
        );
        assert_eq!(
            resolved.is_proper(),
            Some(true),
            "the same cone is provably proper once the direction is resolved"
        );
    }

    #[test]
    fn dependent_constraint_rows_are_refused_by_name_rather_than_reduced() {
        // Two copies of one row leave `A` rank-deficient. The reduction has no
        // coordinates in that case, and the refusal has to say so — a silently
        // pseudo-inverted `M` would be a matrix built on neither of the two
        // conditions the certificate reports.
        let hessian = array![[4.0, 0.0, 0.0], [0.0, 3.0, 0.0], [0.0, 0.0, 2.0]];
        let constraints = array![[1.0, 1.0, 0.0], [1.0, 1.0, 0.0]];
        let message = reduced_cone_precision(hessian.view(), constraints.view())
            .expect_err("dependent rows have no reduction");
        assert!(
            message.contains("constraint rows are dependent") && !message.contains("null(A)"),
            "the refusal must name the rank condition the face broke, and only it, got: {message}"
        );
        // More rows than dimensions cannot be independent at all, and that is
        // decidable without a solve.
        let wide = Array2::<f64>::ones((4, 3));
        let message = reduced_cone_precision(hessian.view(), wide.view())
            .expect_err("q > p has no independent reduction");
        assert!(
            message.contains("cannot be independent"),
            "got: {message}"
        );
    }

    #[test]
    fn a_flat_lineality_direction_is_refused_as_impropriety_by_name() {
        // `A` touches only the first coordinate and `H` carries no curvature along
        // the second, which lies in `null(A)`: `±d` are both feasible there and the
        // integrand is flat, so the reduction does not exist, and the refusal names
        // the lineality condition rather than the rank one.
        let hessian = array![[4.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 2.0]];
        let constraints = array![[1.0, 0.0, 0.0]];
        let message = reduced_cone_precision(hessian.view(), constraints.view())
            .expect_err("a flat lineality direction has no reduction");
        assert!(
            message.contains("H is singular on null(A)")
                && message.contains("lineality")
                && !message.contains("dependent"),
            "the refusal must name the lineality condition, and only it, got: {message}"
        );
    }

    /// gam#3008: a coordinate face whose constrained block carries a survival time
    /// block's curvature (`4.29e6`, the #3008 gout fit's scale), coupled to the
    /// free block strongly enough that the ambient is indefinite. `null(A)` is the
    /// free block, which is positive definite, so the reduction exists and is
    /// `M = τI − C F⁻¹ Cᵀ` in real arithmetic, with one negative direction. The
    /// saddle elimination this replaced refused it: its trailing pivots are entries
    /// of `−A H⁻¹ Aᵀ`, near `1/4.29e6`, below its floor `1e-12·4.29e6`.
    #[test]
    fn a_large_curvature_face_reduces_in_its_own_units_3008() {
        let tau = 4.29e6;
        let free = [2.0, 4.0];
        let coupling = array![[3000.0, 0.0], [0.0, 1000.0], [500.0, 0.0]];
        let mut hessian = Array2::<f64>::zeros((5, 5));
        for row in 0..3 {
            hessian[[row, row]] = tau;
            for k in 0..2 {
                hessian[[row, 3 + k]] = coupling[[row, k]];
                hessian[[3 + k, row]] = coupling[[row, k]];
            }
        }
        for k in 0..2 {
            hessian[[3 + k, 3 + k]] = free[k];
        }
        let mut constraints = Array2::<f64>::zeros((3, 5));
        for row in 0..3 {
            constraints[[row, row]] = 1.0;
        }
        let mut expected = Array2::<f64>::zeros((3, 3));
        for row in 0..3 {
            for column in 0..3 {
                let correction: f64 = (0..2)
                    .map(|k| coupling[[row, k]] * coupling[[column, k]] / free[k])
                    .sum();
                expected[[row, column]] = (if row == column { tau } else { 0.0 }) - correction;
            }
        }
        assert!(
            expected[[0, 0]] < 0.0,
            "the fixture must put a negative direction on the face, as #3008's indefinite \
             ambient does: M[0,0] = {:.6e}",
            expected[[0, 0]]
        );

        let reduced = reduced_cone_precision(hessian.view(), constraints.view())
            .expect("a full-rank face with a positive definite lineality block reduces");
        // The route forms `ZᵀHZ` and `V₁ᵀHZ` through computed bases, so each can
        // carry the ambient spectral rounding `5·ε·‖H‖₂` (bounded here by
        // Gershgorin). To first order that moves `C F⁻¹ Cᵀ` by the band times
        // `1 + 2‖C‖‖F⁻¹‖ + ‖C‖²‖F⁻¹‖²`.
        let gershgorin = (0..5)
            .map(|row| hessian.row(row).iter().map(|value| value.abs()).sum::<f64>())
            .fold(0.0f64, f64::max);
        let ambient_band = 5.0 * f64::EPSILON * gershgorin;
        let coupling_max = coupling.iter().fold(0.0f64, |worst, value| worst.max(value.abs()));
        let free_inverse_max = free.iter().map(|value| value.recip()).fold(0.0f64, f64::max);
        let amplification = 1.0
            + 2.0 * coupling_max * free_inverse_max
            + (coupling_max * free_inverse_max).powi(2);
        let band = ambient_band * amplification;
        for row in 0..3 {
            for column in 0..3 {
                let gap = (reduced[[row, column]] - expected[[row, column]]).abs();
                assert!(
                    gap <= band,
                    "M[{row},{column}] = {:.12e}, expected {:.12e}: gap {gap:.3e} above the \
                     rounding band {band:.3e}",
                    reduced[[row, column]],
                    expected[[row, column]]
                );
            }
        }
    }

    #[test]
    fn a_face_too_wide_for_the_exact_enumeration_reports_undecided_rather_than_proper() {
        // `copositive_simplex_minimum` is exact because it enumerates `2^q`
        // faces, and it owns the range where that is affordable. Past it the
        // certificate must decline to answer: a diagonally dominant `M` here is
        // OBVIOUSLY copositive, and reporting PROPER from that would be a
        // sufficient condition wearing a proof's clothes.
        let width = 21usize;
        let mut hessian = Array2::<f64>::eye(width);
        for j in 0..width {
            hessian[[j, j]] = 2.0 + (j as f64);
        }
        let constraints = Array2::<f64>::eye(width);
        let certificate = cone_properness_certificate(hessian.view(), constraints.view(), 1e-12)
            .expect("a certificate on a wide face");
        assert_eq!(
            certificate.copositive_minimum, None,
            "q = {width} is outside the exact range"
        );
        assert_eq!(
            certificate.is_proper(),
            None,
            "undecided must not collapse into either verdict"
        );
        assert!(
            certificate.summary().contains("UNDECIDED"),
            "got: {}",
            certificate.summary()
        );
    }

    #[test]
    fn the_fixture_reduced_precision_has_exactly_one_negative_direction() {
        let (matrix, _) = fixture();
        assert_eq!(
            symmetric_inertia(matrix.view(), 1e-12).expect("fixture inertia"),
            Inertia { positive: 5, zero: 0, negative: 1 },
            "In(M) = (5,0,1) is what makes this a cone problem rather than a truncated Gaussian"
        );
    }

    #[test]
    fn copositivity_is_decided_exactly_and_matches_the_published_minimum() {
        let (matrix, _) = fixture();
        let (minimum, point) =
            copositive_simplex_minimum(matrix.view()).expect("simplex minimum");
        // Published on #2529 step 1 by the constrained-posterior lane
        // (face enumeration cross-checked by 4000-start projected gradient to
        // 3.02e-13 relative); reproduced here by an independent enumeration.
        assert!(
            (minimum - 6.683215003061817).abs() < 1e-9,
            "min wᵀMw over the simplex was {minimum:.15e}, expected 6.683215003061817"
        );
        assert!(minimum > 0.0, "strict copositivity ⇒ the cone-truncated law is proper");
        let total: f64 = point.sum();
        assert!((total - 1.0).abs() < 1e-9, "the argmin lies on the simplex, sum was {total}");
        assert!(point.iter().all(|value| *value >= 0.0), "the argmin is nonnegative");
    }

}
