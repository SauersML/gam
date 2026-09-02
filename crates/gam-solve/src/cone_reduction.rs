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

/// The constrained mode of `½xᵀMx + cᵀx` over the nonnegative orthant.
#[derive(Clone, Debug)]
pub struct ConeMode {
    /// The minimiser `x*`.
    pub point: Array1<f64>,
    /// `φ* = ½x*ᵀMx* + cᵀx*`.
    pub value: f64,
    /// `g = Mx* + c`. Zero on the free set and non-negative on the active set —
    /// these are the KKT multipliers, and the downstream quadrature integrates
    /// `exp(−½dᵀMd − gᵀd)` over `{d ≥ −x*}`.
    pub gradient: Array1<f64>,
    /// Indices with `x*_j > 0`.
    pub free: Vec<usize>,
}

/// Symmetric inertia by `LDLᵀ` with symmetric pivoting.
///
/// Sylvester's law of inertia makes the pivot signs the inertia, so this needs
/// no eigensolver. Diagonal pivoting keeps it well posed for the indefinite
/// case, which is the case this module exists for.
pub fn symmetric_inertia(matrix: ArrayView2<'_, f64>, tolerance: f64) -> Result<Inertia, String> {
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
    /// feasible there — so anything but all-positive here is impropriety.
    pub lineality_inertia: Inertia,
    /// `min wᵀMw` over the unit simplex. `Some(v)` with `v > 0` is a proof that
    /// the cone-truncated posterior is proper; `Some(v)` with `v <= 0` is a
    /// proof that it is improper. `None` means the face is too wide for the
    /// exact `2^q` enumeration, so properness is undecided — never assumed.
    pub copositive_minimum: Option<f64>,
}

impl ConeProperness {
    /// `Some(true)`/`Some(false)` when properness is PROVED either way, `None`
    /// when it is undecided. Undecided is deliberately not folded into either
    /// answer.
    pub fn is_proper(&self) -> Option<bool> {
        if self.lineality_inertia.negative > 0 || self.lineality_inertia.zero > 0 {
            return Some(false);
        }
        self.copositive_minimum.map(|minimum| minimum > 0.0)
    }

    /// One line naming every quantity the verdict was decided against, for a
    /// refusal or a decline to carry.
    pub fn summary(&self) -> String {
        let verdict = match self.is_proper() {
            Some(true) => "PROPER".to_string(),
            Some(false) => "IMPROPER".to_string(),
            None => format!(
                "UNDECIDED (the exact enumeration is out of range at q = {})",
                self.reduced.nrows()
            ),
        };
        let copositive = match self.copositive_minimum {
            Some(minimum) => format!("{minimum:.6e}"),
            None => "not enumerated".to_string(),
        };
        format!(
            "cone-truncated posterior is {verdict}: In(H) = ({}, {}, {}), \
             In(M) = ({}, {}, {}), In(ZᵀHZ) = ({}, {}, {}) on null(A), \
             min wᵀMw over the simplex = {copositive}",
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
/// `M = NᵀHN − NᵀHZ(ZᵀHZ)⁻¹ZᵀHN`. This computes it WITHOUT forming `Z`, `N`, or
/// `H⁻¹`, from the defining variational identity
///
/// ```text
/// wᵀMw = stat{ dᵀHd : Ad = w }
/// ```
///
/// — the stationary value, which is the MINIMUM exactly when `ZᵀHZ ≻ 0` and is
/// the algebraic Schur complement either way, so this route does not presuppose
/// the condition the certificate above it goes on to test. Its stationarity
/// system is the symmetric saddle point
///
/// ```text
/// [ H  Aᵀ ] [ d ]   [ 0 ]
/// [ A  0  ] [ ν ] = [ w ],        M w = −ν
/// ```
///
/// so one solve per constraint row gives `M` exactly. That matters here: the
/// reason this module exists is that `H` is INDEFINITE, so `Σ = H⁻¹` may not be
/// a covariance and `M = (AH⁻¹Aᵀ)⁻¹` — the identity that holds when `H ≻ 0` —
/// cannot be evaluated by inverting anything. The saddle system is indefinite by
/// construction and needs no positive definiteness anywhere.
///
/// The system is nonsingular exactly when `A` has full row rank and `H` is
/// nonsingular on `null(A)`; a failed pivot therefore refuses by naming which of
/// those two the face broke, rather than returning a matrix built on neither.
pub fn reduced_cone_precision(
    hessian: ArrayView2<'_, f64>,
    constraints: ArrayView2<'_, f64>,
) -> Result<Array2<f64>, String> {
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
    let size = p + q;
    let mut saddle = Array2::<f64>::zeros((size, size));
    saddle
        .slice_mut(ndarray::s![0..p, 0..p])
        .assign(&hessian);
    saddle
        .slice_mut(ndarray::s![0..p, p..size])
        .assign(&constraints.t());
    saddle
        .slice_mut(ndarray::s![p..size, 0..p])
        .assign(&constraints);
    if saddle.iter().any(|value| !value.is_finite()) {
        return Err(
            "cone reduction: the saddle system carries a non-finite entry, so neither the \
             ambient precision nor the constraint rows can be trusted"
                .to_string(),
        );
    }
    let scale = saddle
        .iter()
        .fold(0.0f64, |worst, value| worst.max(value.abs()))
        .max(1.0);
    let floor = 1e-12 * scale;
    let mut reduced = Array2::<f64>::zeros((q, q));
    for column in 0..q {
        let mut rhs = Array1::<f64>::zeros(size);
        rhs[p + column] = 1.0;
        let Some(solution) = symmetric_solve(&saddle, &rhs, floor) else {
            return Err(format!(
                "cone reduction: the saddle system [[H, Aᵀ],[A, 0]] is singular at pivot floor \
                 {floor:.3e} while eliminating constraint row {column}. Either the {q} \
                 constraint rows are dependent, or H is singular on null(A) — and the second \
                 case is itself impropriety, since null(A) is the recession cone's lineality \
                 space"
            ));
        };
        for row in 0..q {
            reduced[[row, column]] = -solution[p + row];
        }
    }
    // `M` is symmetric in exact arithmetic (it is a Schur complement of a
    // symmetric matrix); the elimination is not symmetry preserving, so the
    // asymmetry it leaves is measured and then removed rather than assumed
    // absent.
    let mut worst_asymmetry = 0.0f64;
    for row in 0..q {
        for column in 0..q {
            let gap = (reduced[[row, column]] - reduced[[column, row]]).abs();
            worst_asymmetry = worst_asymmetry.max(gap);
        }
    }
    let reduced_scale = reduced
        .iter()
        .fold(0.0f64, |worst, value| worst.max(value.abs()))
        .max(1.0);
    if worst_asymmetry > 1e-6 * reduced_scale {
        return Err(format!(
            "cone reduction: the reduced precision came back asymmetric by \
             {worst_asymmetry:.3e} against a scale of {reduced_scale:.3e}, which a Schur \
             complement of a symmetric matrix cannot be — the saddle solve lost the face's \
             conditioning"
        ));
    }
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
pub fn cone_properness_certificate(
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
    let copositive_minimum = copositive_simplex_minimum(reduced.view())
        .ok()
        .map(|(minimum, _)| minimum);
    Ok(ConeProperness {
        reduced,
        ambient_inertia,
        reduced_inertia,
        lineality_inertia,
        copositive_minimum,
    })
}

/// Solve `A y = b` for a small dense `A` by Gaussian elimination with partial
/// pivoting. Returns `None` when a pivot falls below the floor, which the
/// callers read as "this face is degenerate, skip it" rather than as an error —
/// a singular face carries no isolated stationary point to compare.
///
/// The name records where it is used, not a requirement: the elimination is a
/// general LU with row pivoting, and `reduced_cone_precision` deliberately feeds
/// it an indefinite symmetric saddle matrix.
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
pub fn copositive_simplex_minimum(
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
    }

    #[test]
    fn dependent_constraint_rows_are_refused_by_name_rather_than_reduced() {
        // Two copies of one row make the saddle system singular. The reduction
        // has no coordinates in that case, and the refusal has to say so — a
        // silently pseudo-inverted `M` would be a matrix built on neither of the
        // two conditions the certificate reports.
        let hessian = array![[4.0, 0.0, 0.0], [0.0, 3.0, 0.0], [0.0, 0.0, 2.0]];
        let constraints = array![[1.0, 1.0, 0.0], [1.0, 1.0, 0.0]];
        let message = reduced_cone_precision(hessian.view(), constraints.view())
            .expect_err("dependent rows have no reduction");
        assert!(
            message.contains("dependent") && message.contains("lineality"),
            "the refusal must name both readings of a singular saddle, got: {message}"
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
