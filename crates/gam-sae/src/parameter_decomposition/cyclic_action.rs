//! Structured-edit coordinates from a declared single-cycle action on the rows of one
//! use-site table (#2951, trained modular-addition benchmark, S2).
//!
//! # Scope
//!
//! The action is DECLARED: a successor map over the table's rows whose moved rows form one
//! odd cycle. An undeclared or non-cyclic operator is refused here; invariant subspaces of a
//! general operator belong to `schur`, and plane recovery from a frozen matrix to [`super::spectral`].
//!
//! # The native reference
//!
//! A table `E` (`rows × d`, e.g. a token embedding) and a declared cycle `r_0 → r_1 → … →
//! r_{p−1} → r_0` define, for each shift `s`, the row-permutation edit that puts row `r_{a+s}`
//! where row `r_a` was. Executing that edit is executing the shifted tokens, so it is the exact
//! native reference. Rows outside the cycle stay fixed.
//!
//! # Closed-form invariant planes
//!
//! For odd `p` the characters `1` and `D(ω_k a) = (cos ω_k a, sin ω_k a)`, `ω_k = 2πk/p`,
//! `k = 1..m`, `m = (p − 1)/2`, are an orthogonal basis of `R^p` with squared norms `p` and
//! `p/2`. With `e_a` the row at cycle position `a`,
//!
//! ```text
//! e_a     = c_0 + Σ_k U_k D(ω_k a),   c_0 = (1/p) Σ_a e_a,   U_k = (2/p) Σ_a e_a D(ω_k a)ᵀ,
//! e_{a+s} = c_0 + Σ_k U_k R_{ω_k s} D(ω_k a).
//! ```
//!
//! Stacking the rows, `Eᵀ = B Φ` with `B = [c_0, U_1 … U_m]` (`d × p`) and `Φ` the square
//! character matrix, whose singular values are `√p` and `√(p/2)`. So `σ_p(E)` lies between
//! `√(p/2) σ_p(B)` and `√p σ_p(B)`: `B` has full column rank exactly when `E` has full row rank.
//!
//! # The frequency edit
//!
//! The shift edit of a frequency subset `S` turns each cycled row's own components in the planes of
//! `S`, with the characters of its cycle position as coordinates ([`frequency_edit`]):
//!
//! ```text
//! e_a  ↦  e_a + Σ_{k∈S} U_k (D(ω_k (a + s)) − D(ω_k a)).
//! ```
//!
//! It is exact for the planes it names, and leaves every other plane and every row outside the cycle
//! fixed, as the native reference does. On the full set it is the shift. It solves no inverse, so it
//! needs no condition on the basis.
//!
//! The coordinates are characters, not least-squares coordinates of the row on `B_S = [c_0, U_S]`.
//! An edit that turns those, `I + U_S (R_{S,s} − I) X_U` with `X B_S = I`, agrees with the shift on
//! the full set but, for a proper subset whose planes are skewed against the others, also turns
//! whatever of the other planes projects onto `B_S`, and it moves the rows outside the cycle. On the
//! trained modular-addition table (`p = 113`, pos0, every test pair at every shift `2..112`), the six
//! most powerful planes miss the shifted reference's argmax in 10 of 992,118 rows under the frequency
//! edit and in 40,687 under the least-squares edit, whose miss is not monotone in the plane count
//! (mpd-lead's torch measurement `freq_edit_diag`, acn112, 09-19).
//!
//! # What reproducing the shift does not show
//!
//! `B` has `p` columns, so for `p ≤ d` it is generically of full column rank, and a linear edit
//! reproducing the shift exists for any table, a random one included (mpd-verify's theorem on
//! #2951). Reproducing the shift is no evidence of a mechanism. What separates a table that
//! carries one is the code of a proper subset at fidelity, which [`plane_program_code`] prices.
//!
//! # Evidence status
//!
//! The plane coefficients are sums of rows times library trigonometric values, whose accuracy
//! has no derivation on main, so this module returns NO status for them and claims no band on
//! them. The one status it returns is on the executed artifact: [`ShiftResidual::residual_status`]
//! is a `UniformBound` over the stated finite family (every cycled row and column at one shift),
//! because each entry's evaluation band is derived ([`evaluation_band`]) and every entry is
//! evaluated. It certifies the binary64 edited table computed here, and nothing an external
//! executor recomputes at another precision.
//!
//! # Code
//!
//! A plane program sends its frequency subset in the enumerative subset code and the `2d|S|`
//! basis reals as one [`LatticeCode`] at a declared precision. The decoder rebuilds the basis, and
//! every angle from `p`, `k` and `s`, so no angle is transmitted. Distortion belongs to the decoded
//! artifact, measured where it executes.

use super::codec::{BitString, subset_code_len_bits};
use super::precision::{DeclaredPrecision, LatticeCode};
use super::receipts::evaluation_band;
use super::supports::{EvidenceStatus, EvidenceStatusError};
use ndarray::{Array1, Array2, ArrayView2};
use std::f64::consts::PI;
use std::fmt;

/// A refused cyclic-action computation.
#[derive(Debug)]
pub enum CyclicActionError {
    /// The table has no rows or no columns.
    EmptyTable,
    /// A table, mean or basis entry is not finite.
    NonFinite { row: usize, col: usize },
    /// A successor points outside the table.
    RowOutOfRange { row: usize, rows: usize },
    /// Two rows share a successor, so the declaration is not a permutation.
    NotPermutation { image: usize },
    /// The moved rows form several cycles: not a declared single-cycle action. A general
    /// operator's invariant subspaces belong to `schur`.
    NotSingleCycle { cycles: usize },
    /// A cycle needs at least three rows to carry a plane.
    CycleTooShort { length: usize },
    /// An even cycle has the character `(−1)^a`, a one-dimensional half-turn space with no plane.
    EvenCycle { length: usize },
    /// A shape does not match.
    ShapeMismatch { what: &'static str, expected: usize, found: usize },
    /// A frequency subset must be non-empty, strictly ascending, and inside `1..=m`.
    InvalidFrequencies { frequencies: Vec<usize>, planes: usize },
    /// A code failure from `codec` or `precision`.
    Code(String),
    /// An evidence status was refused by its constructor.
    Evidence(EvidenceStatusError),
}

impl fmt::Display for CyclicActionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyTable => write!(f, "the table has no rows or no columns"),
            Self::NonFinite { row, col } => write!(f, "entry ({row}, {col}) is not finite"),
            Self::RowOutOfRange { row, rows } => {
                write!(f, "successor {row} is outside a table of {rows} rows")
            }
            Self::NotPermutation { image } => {
                write!(f, "row {image} is the successor of two rows; the declaration is not a permutation")
            }
            Self::NotSingleCycle { cycles } => write!(
                f,
                "the moved rows form {cycles} cycles, not one declared cycle; route a general operator to schur"
            ),
            Self::CycleTooShort { length } => {
                write!(f, "a cycle of {length} rows carries no plane; at least 3 are needed")
            }
            Self::EvenCycle { length } => write!(
                f,
                "a cycle of even length {length} has a one-dimensional half-turn character with no plane"
            ),
            Self::ShapeMismatch { what, expected, found } => {
                write!(f, "{what}: expected {expected}, found {found}")
            }
            Self::InvalidFrequencies { frequencies, planes } => write!(
                f,
                "frequencies {frequencies:?} must be non-empty, strictly ascending and within 1..={planes}"
            ),
            Self::Code(message) => write!(f, "code failed: {message}"),
            Self::Evidence(error) => write!(f, "evidence status refused: {error:?}"),
        }
    }
}

impl std::error::Error for CyclicActionError {}

fn up(value: f64) -> f64 {
    value.next_up()
}

/// A declared odd cycle of table rows: position `a` holds row `rows[a]`, and position 0 is the
/// smallest moved row (a phase convention the edit does not depend on).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RowCycle {
    rows: Vec<usize>,
    table_rows: usize,
}

impl RowCycle {
    /// Validate a successor map over every table row; fixed rows map to themselves.
    pub fn from_successor(successor: &[usize]) -> Result<Self, CyclicActionError> {
        let table_rows = successor.len();
        let mut seen = vec![false; table_rows];
        for &image in successor {
            if image >= table_rows {
                return Err(CyclicActionError::RowOutOfRange { row: image, rows: table_rows });
            }
            if seen[image] {
                return Err(CyclicActionError::NotPermutation { image });
            }
            seen[image] = true;
        }
        let mut visited = vec![false; table_rows];
        let mut cycles = 0;
        let mut start = None;
        for row in 0..table_rows {
            if successor[row] == row || visited[row] {
                continue;
            }
            cycles += 1;
            if start.is_none() {
                start = Some(row);
            }
            let mut current = row;
            while !visited[current] {
                visited[current] = true;
                current = successor[current];
            }
        }
        if cycles > 1 {
            return Err(CyclicActionError::NotSingleCycle { cycles });
        }
        let mut rows = Vec::new();
        if let Some(first) = start {
            let mut current = first;
            loop {
                rows.push(current);
                current = successor[current];
                if current == first {
                    break;
                }
            }
        }
        let length = rows.len();
        if length < 3 {
            return Err(CyclicActionError::CycleTooShort { length });
        }
        if length % 2 == 0 {
            return Err(CyclicActionError::EvenCycle { length });
        }
        Ok(Self { rows, table_rows })
    }

    /// The cycle length `p`.
    pub fn length(&self) -> usize {
        self.rows.len()
    }

    /// The table row at each cycle position.
    pub fn rows(&self) -> &[usize] {
        &self.rows
    }

    /// The number of table rows the declaration covers.
    pub fn table_rows(&self) -> usize {
        self.table_rows
    }

    /// The number of planes, `m = (p − 1)/2`.
    pub fn plane_count(&self) -> usize {
        (self.rows.len() - 1) / 2
    }
}

fn check_table(table: ArrayView2<'_, f64>, cycle: &RowCycle) -> Result<(), CyclicActionError> {
    if table.nrows() == 0 || table.ncols() == 0 {
        return Err(CyclicActionError::EmptyTable);
    }
    if table.nrows() != cycle.table_rows() {
        return Err(CyclicActionError::ShapeMismatch {
            what: "table rows against the declared successor map",
            expected: cycle.table_rows(),
            found: table.nrows(),
        });
    }
    if let Some(((row, col), _)) = table.indexed_iter().find(|(_, value)| !value.is_finite()) {
        return Err(CyclicActionError::NonFinite { row, col });
    }
    Ok(())
}

fn check_frequencies(frequencies: &[usize], planes: usize) -> Result<(), CyclicActionError> {
    let ascending = frequencies.windows(2).all(|pair| pair[0] < pair[1]);
    let inside = frequencies.iter().all(|&frequency| (1..=planes).contains(&frequency));
    if frequencies.is_empty() || !ascending || !inside {
        return Err(CyclicActionError::InvalidFrequencies {
            frequencies: frequencies.to_vec(),
            planes,
        });
    }
    Ok(())
}

/// The angle `ω_k s = 2π (k s mod p)/p`, reduced before the division so it lies in `[0, 2π)`.
fn plane_angle(frequency: usize, shift: usize, length: usize) -> f64 {
    2.0 * PI * ((frequency % length) * (shift % length) % length) as f64 / length as f64
}

/// The closed-form planes of a table under a declared odd cycle.
#[derive(Clone, Debug, PartialEq)]
pub struct CyclicPlanes {
    cycle: RowCycle,
    /// `c_0`, the mean of the cycled rows (`d`).
    pub mean: Array1<f64>,
    /// `[u_1c, u_1s, u_2c, u_2s, …]`, `d × 2m`.
    pub planes: Array2<f64>,
    /// `‖u_kc‖² + ‖u_ks‖²` for `k = 1..m`, each plane's share of the cycled rows' variation up to
    /// the common factor `p/2` (Parseval).
    pub power: Vec<f64>,
}

impl CyclicPlanes {
    /// The declared cycle.
    pub fn cycle(&self) -> &RowCycle {
        &self.cycle
    }

    /// `U_S` for a frequency subset, `d × 2|S|`.
    pub fn basis(&self, frequencies: &[usize]) -> Result<Array2<f64>, CyclicActionError> {
        check_frequencies(frequencies, self.cycle.plane_count())?;
        let mut basis = Array2::<f64>::zeros((self.planes.nrows(), 2 * frequencies.len()));
        for (plane, &frequency) in frequencies.iter().enumerate() {
            let source = 2 * (frequency - 1);
            basis.column_mut(2 * plane).assign(&self.planes.column(source));
            basis.column_mut(2 * plane + 1).assign(&self.planes.column(source + 1));
        }
        Ok(basis)
    }
}

/// The closed-form planes of `table` under `cycle`.
pub fn cyclic_planes(
    table: ArrayView2<'_, f64>,
    cycle: &RowCycle,
) -> Result<CyclicPlanes, CyclicActionError> {
    check_table(table, cycle)?;
    let length = cycle.length();
    let plane_count = cycle.plane_count();
    let width = table.ncols();
    let inverse_length = 1.0 / length as f64;
    let scale = 2.0 * inverse_length;
    let mut mean = Array1::<f64>::zeros(width);
    let mut planes = Array2::<f64>::zeros((width, 2 * plane_count));
    for (position, &row) in cycle.rows().iter().enumerate() {
        let values = table.row(row);
        mean.scaled_add(inverse_length, &values);
        for frequency in 1..=plane_count {
            let angle = plane_angle(frequency, position, length);
            let column = 2 * (frequency - 1);
            planes.column_mut(column).scaled_add(scale * angle.cos(), &values);
            planes.column_mut(column + 1).scaled_add(scale * angle.sin(), &values);
        }
    }
    let power = (0..plane_count)
        .map(|plane| {
            planes.column(2 * plane).iter().map(|value| value * value).sum::<f64>()
                + planes.column(2 * plane + 1).iter().map(|value| value * value).sum::<f64>()
        })
        .collect();
    Ok(CyclicPlanes { cycle: cycle.clone(), mean, planes, power })
}

/// The frequency edit of one plane subset at one shift. Each cycled row's own components in the planes of
/// `frequencies` turn by `ω_k s`, and nothing else in the table moves. Row `r_a` becomes
/// `e_a + Σ_{k∈S} U_k (D(ω_k (a + s)) − D(ω_k a))`, so the edited table is `E + left · rightᵀ` with
/// `right = U_S` and row `r_a` of `left` holding `D(ω_k (a + s)) − D(ω_k a)` for each `k ∈ S`. A row
/// outside the cycle has a zero `left` row, so it stays exactly fixed, as the native reference keeps it.
#[derive(Clone, Debug, PartialEq)]
pub struct CyclicFrequencyEdit {
    pub frequencies: Vec<usize>,
    pub shift: usize,
    pub left: Array2<f64>,
    pub right: Array2<f64>,
}

/// The frequency edit of `frequencies` at `shift` under `cycle`, from a plane basis `U_S` (`d × 2|S|`),
/// the closed form or a decoded artifact. Every coordinate is a character of the row's cycle position,
/// so no inverse is solved and no basis condition is needed.
pub fn frequency_edit(
    cycle: &RowCycle,
    basis: ArrayView2<'_, f64>,
    frequencies: &[usize],
    shift: usize,
) -> Result<CyclicFrequencyEdit, CyclicActionError> {
    check_frequencies(frequencies, cycle.plane_count())?;
    let plane_columns = 2 * frequencies.len();
    if basis.ncols() != plane_columns {
        return Err(CyclicActionError::ShapeMismatch {
            what: "basis columns against 2|S|",
            expected: plane_columns,
            found: basis.ncols(),
        });
    }
    if let Some(((row, col), _)) = basis.indexed_iter().find(|(_, value)| !value.is_finite()) {
        return Err(CyclicActionError::NonFinite { row, col });
    }
    let length = cycle.length();
    let mut left = Array2::<f64>::zeros((cycle.table_rows(), plane_columns));
    for (position, &row) in cycle.rows().iter().enumerate() {
        let target = (position + shift % length) % length;
        for (plane, &frequency) in frequencies.iter().enumerate() {
            let (sin_at, cos_at) = plane_angle(frequency, position, length).sin_cos();
            let (sin_to, cos_to) = plane_angle(frequency, target, length).sin_cos();
            left[[row, 2 * plane]] = cos_to - cos_at;
            left[[row, 2 * plane + 1]] = sin_to - sin_at;
        }
    }
    Ok(CyclicFrequencyEdit {
        frequencies: frequencies.to_vec(),
        shift,
        left,
        right: basis.to_owned(),
    })
}

/// The finite family a [`ShiftResidual`] covers: every cycled row and every column at one shift.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ShiftRegion {
    pub shift: usize,
    pub cycled_rows: usize,
    pub columns: usize,
}

/// How the executed edited table `E + L Rᵀ` compares with the row permutation of its shift.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ShiftResidual {
    pub region: ShiftRegion,
    /// The largest computed `|(E + L Rᵀ)_{r_a j} − E_{r_{a+s} j}|` over cycled rows.
    pub max_abs_residual: f64,
    /// The largest evaluation band among those entries.
    pub max_residual_band: f64,
    /// The largest computed `|(L Rᵀ)_{r j}|` over rows outside the cycle, which the native
    /// reference leaves unchanged; 0 when every row is cycled.
    pub max_abs_fixed_row_change: f64,
    /// The largest evaluation band among those entries.
    pub max_fixed_row_band: f64,
}

impl ShiftResidual {
    /// An upper bound on the exact residual of the computed factors over cycled rows.
    pub fn residual_upper_bound(&self) -> f64 {
        up(self.max_abs_residual + self.max_residual_band)
    }

    /// An upper bound on the exact change the computed factors make to rows outside the cycle.
    pub fn fixed_row_upper_bound(&self) -> f64 {
        up(self.max_abs_fixed_row_change + self.max_fixed_row_band)
    }

    /// `|residual entry| ≤ upper` over the enumerated family, including numerical error.
    pub fn residual_status(&self) -> Result<EvidenceStatus<(), ShiftRegion>, CyclicActionError> {
        EvidenceStatus::uniform_bound(self.residual_upper_bound(), self.max_residual_band, self.region)
            .map_err(CyclicActionError::Evidence)
    }
}

/// Compare the edited table of `edit` with the row permutation of its shift under `cycle`.
///
/// A residual entry `e_{r_a j} + Σ_t l_{r_a t} r_{jt} − e_{r_{a+s} j}` has `2|S|` product terms,
/// each rounding once and passing at most `2|S| − 1` additions among the products, one where
/// they meet `e_{r_a j}` and one subtraction: `k = 2|S| + 2`. An entry of a fixed row is the
/// product sum alone: `k = 2|S|`.
pub fn shift_residual(
    table: ArrayView2<'_, f64>,
    cycle: &RowCycle,
    edit: &CyclicFrequencyEdit,
) -> Result<ShiftResidual, CyclicActionError> {
    check_table(table, cycle)?;
    let (rows, width) = table.dim();
    let (left, right, shift) = (edit.left.view(), edit.right.view(), edit.shift);
    let components = left.ncols();
    for (what, expected, found) in [
        ("edit left rows against table rows", rows, left.nrows()),
        ("edit right rows against table width", width, right.nrows()),
        ("edit right components against left components", components, right.ncols()),
    ] {
        if expected != found {
            return Err(CyclicActionError::ShapeMismatch { what, expected, found });
        }
    }
    let length = cycle.length();
    let mut target = vec![None; rows];
    for (position, &row) in cycle.rows().iter().enumerate() {
        target[row] = Some(cycle.rows()[(position + shift % length) % length]);
    }
    let mut out = ShiftResidual {
        region: ShiftRegion { shift, cycled_rows: length, columns: width },
        max_abs_residual: 0.0,
        max_residual_band: 0.0,
        max_abs_fixed_row_change: 0.0,
        max_fixed_row_band: 0.0,
    };
    for row in 0..rows {
        for column in 0..width {
            let mut change = 0.0;
            let mut absolute_sum = 0.0;
            for component in 0..components {
                let term = left[[row, component]] * right[[column, component]];
                change += term;
                absolute_sum = up(absolute_sum + up(term.abs()));
            }
            match target[row] {
                Some(shifted) => {
                    let residual = table[[row, column]] + change - table[[shifted, column]];
                    let bound_sum = up(up(absolute_sum + table[[row, column]].abs())
                        + table[[shifted, column]].abs());
                    let band = evaluation_band(components + 2, bound_sum, components as f64);
                    out.max_abs_residual = out.max_abs_residual.max(residual.abs());
                    out.max_residual_band = out.max_residual_band.max(band);
                }
                None => {
                    let band = evaluation_band(components, absolute_sum, components as f64);
                    out.max_abs_fixed_row_change = out.max_abs_fixed_row_change.max(change.abs());
                    out.max_fixed_row_band = out.max_fixed_row_band.max(band);
                }
            }
        }
    }
    Ok(out)
}

/// The integer-bit code of a plane program: the frequency subset and the basis reals.
#[derive(Clone, Debug, PartialEq)]
pub struct PlaneProgramCode {
    /// `L(S) = L_int(|S| + 1) + ⌈log₂ C(m, |S|)⌉`.
    pub subset_bits: u64,
    /// The `2d|S|` basis reals, column-major, at the declared precision.
    pub basis: LatticeCode,
    /// The length of `basis` written as one message.
    pub basis_bits: u64,
}

impl PlaneProgramCode {
    /// The exact program length.
    pub fn total_bits(&self) -> u64 {
        self.subset_bits + self.basis_bits
    }
}

/// Price the plane program of `frequencies` over `planes` at a declared precision.
pub fn plane_program_code(
    planes: &CyclicPlanes,
    frequencies: &[usize],
    precision: DeclaredPrecision,
) -> Result<PlaneProgramCode, CyclicActionError> {
    let basis = planes.basis(frequencies)?;
    let subset_bits = subset_code_len_bits(planes.cycle().plane_count(), frequencies.len())
        .map_err(|error| CyclicActionError::Code(format!("{error:?}")))?;
    let values: Vec<f64> = basis.t().iter().copied().collect();
    let code = LatticeCode::encode(&values, precision).map_err(CyclicActionError::Code)?;
    let mut message = BitString::new();
    code.write(&mut message).map_err(CyclicActionError::Code)?;
    Ok(PlaneProgramCode { subset_bits, basis: code, basis_bits: message.len_bits() })
}

#[cfg(test)]
mod tests {
    use super::*;
    use gam_linalg::faer_ndarray::{FaerArrayView, FaerQr, FaerSvd, col_piv_qr_solve_lstsq};
    use gam_linalg::roundoff::factor_singular_band;
    use ndarray::{ArrayView1, s};
    use rand::rngs::StdRng;
    use rand::{RngExt, SeedableRng};

    const WIDTH: usize = 8;
    const LENGTH: usize = 7;
    /// Cycle positions `0..7` in table-row order, starting at the smallest moved row; row 3 is
    /// fixed.
    const CYCLE: [usize; LENGTH] = [0, 5, 2, 7, 1, 6, 4];
    const FIXED_ROW: usize = 3;
    const ALL: [usize; 3] = [1, 2, 3];

    fn successor() -> Vec<usize> {
        let mut map: Vec<usize> = (0..LENGTH + 1).collect();
        for (position, &row) in CYCLE.iter().enumerate() {
            map[row] = CYCLE[(position + 1) % LENGTH];
        }
        map
    }

    fn declared_cycle() -> RowCycle {
        RowCycle::from_successor(&successor()).expect("a declared 7-cycle with one fixed row")
    }

    /// `[1, cos ω_k a, sin ω_k a, …]` for `frequencies`, in the convention the module uses.
    fn characters(frequencies: &[usize], position: usize) -> Array1<f64> {
        let mut out = Array1::<f64>::zeros(1 + 2 * frequencies.len());
        out[0] = 1.0;
        for (plane, &frequency) in frequencies.iter().enumerate() {
            let (sin, cos) = plane_angle(frequency, position, LENGTH).sin_cos();
            out[1 + 2 * plane] = cos;
            out[2 + 2 * plane] = sin;
        }
        out
    }

    /// Cycled rows `F φ_a` for planted factors `F = [c_0, U_k …]`, plus a random fixed row, and an
    /// upper bound on the Frobenius norm of the planting's own evaluation error. With `orthogonal`
    /// the factor columns are orthonormal; with `duplicate` the last plane copies the first.
    fn plant(
        frequencies: &[usize],
        orthogonal: bool,
        duplicate: bool,
        seed: u64,
    ) -> (Array2<f64>, f64) {
        let mut rng = StdRng::seed_from_u64(seed);
        let columns = 1 + 2 * frequencies.len();
        let draw = Array2::<f64>::from_shape_fn((WIDTH, WIDTH), |_| rng.random_range(-1.0..1.0));
        let mut factors = if orthogonal {
            let (q, _) = draw.qr().expect("Householder QR of a uniform draw");
            q.slice(s![.., ..columns]).to_owned()
        } else {
            draw.slice(s![.., ..columns]).to_owned()
        };
        if duplicate {
            let first = factors.slice(s![.., 1..3]).to_owned();
            factors.slice_mut(s![.., columns - 2..]).assign(&first);
        }
        let mut table = Array2::<f64>::zeros((LENGTH + 1, WIDTH));
        let mut planting_squares = 0.0;
        for (position, &row) in CYCLE.iter().enumerate() {
            let phi = characters(frequencies, position);
            table.row_mut(row).assign(&factors.dot(&phi));
            for j in 0..WIDTH {
                let absolute_sum = (0..columns)
                    .fold(0.0, |acc, t| up(acc + up((factors[[j, t]] * phi[t]).abs())));
                let band = evaluation_band(columns, absolute_sum, columns as f64);
                planting_squares = up(planting_squares + up(band * band));
            }
        }
        for column in 0..WIDTH {
            table[[FIXED_ROW, column]] = rng.random_range(-1.0..1.0);
        }
        (table, up(planting_squares.sqrt()))
    }

    /// An upper bound on `max_i Σ_j |m_ij|`.
    fn inf_norm_upper(matrix: ArrayView2<'_, f64>) -> f64 {
        matrix.outer_iter().fold(0.0_f64, |acc, row| {
            acc.max(row.iter().fold(0.0, |sum, value| up(sum + value.abs())))
        })
    }

    /// A lower bound on `min_{b ≠ c} max_j |e_bj − e_cj|` over the cycled rows.
    fn minimum_row_spacing(table: &Array2<f64>) -> f64 {
        let mut spacing = f64::INFINITY;
        for (index, &b) in CYCLE.iter().enumerate() {
            for &c in &CYCLE[index + 1..] {
                let separation = (0..WIDTH).fold(0.0_f64, |acc, j| {
                    let difference = (table[[b, j]] - table[[c, j]]).abs();
                    let band = evaluation_band(1, up(table[[b, j]].abs() + table[[c, j]].abs()), 0.0);
                    acc.max((difference - band).next_down())
                });
                spacing = spacing.min(separation);
            }
        }
        spacing
    }

    /// `r_E ≥ max_a ‖e_a − B φ_a‖_∞`, an a posteriori bound on how far the cycled rows lie from their
    /// characters on `full = B = [c_0, U]`, with columns planted at `basis_frequencies`.
    fn reconstruction_defect(table: &Array2<f64>, full: ArrayView2<'_, f64>, basis_frequencies: &[usize]) -> f64 {
        let columns = full.ncols();
        let mut reconstruction = 0.0_f64;
        for (position, &row) in CYCLE.iter().enumerate() {
            let phi = characters(basis_frequencies, position);
            for j in 0..WIDTH {
                let mut value = table[[row, j]];
                let mut absolute_sum = table[[row, j]].abs();
                for t in 0..columns {
                    let term = full[[j, t]] * phi[t];
                    value -= term;
                    absolute_sum = up(absolute_sum + up(term.abs()));
                }
                let upper = up(value.abs() + evaluation_band(columns + 1, absolute_sum, columns as f64));
                reconstruction = reconstruction.max(upper);
            }
        }
        reconstruction
    }

    fn full_basis(mean: ArrayView1<'_, f64>, planes: ArrayView2<'_, f64>) -> Array2<f64> {
        let mut full = Array2::<f64>::zeros((WIDTH, 1 + planes.ncols()));
        full.column_mut(0).assign(&mean);
        full.slice_mut(s![.., 1..]).assign(&planes);
        full
    }

    /// Singular values relating the cycled rows `E` (`p × d`) to `B = [c_0, U_1 … U_m]` (`d × p`).
    struct RankRelation {
        sigma_e: f64,
        band_e: f64,
        sigma_b: f64,
        band_b: f64,
        phi_min: f64,
        phi_max: f64,
        band_phi: f64,
        /// `≥ ‖E − (B Φ)ᵀ‖_F`.
        delta: f64,
    }

    fn rank_relation(table: &Array2<f64>) -> RankRelation {
        let cycle = declared_cycle();
        let planes = cyclic_planes(table.view(), &cycle).expect("closed-form planes");
        let full = full_basis(planes.mean.view(), planes.basis(&ALL).expect("every plane").view());
        let mut cycled = Array2::<f64>::zeros((LENGTH, WIDTH));
        let mut phi = Array2::<f64>::zeros((LENGTH, LENGTH));
        let mut delta_squares = 0.0;
        for (position, &row) in CYCLE.iter().enumerate() {
            cycled.row_mut(position).assign(&table.row(row));
            let at = characters(&ALL, position);
            phi.column_mut(position).assign(&at);
            for j in 0..WIDTH {
                let mut value = table[[row, j]];
                let mut absolute_sum = table[[row, j]].abs();
                for t in 0..LENGTH {
                    let term = full[[j, t]] * at[t];
                    value -= term;
                    absolute_sum = up(absolute_sum + up(term.abs()));
                }
                let upper = up(value.abs() + evaluation_band(LENGTH + 1, absolute_sum, LENGTH as f64));
                delta_squares = up(delta_squares + up(upper * upper));
            }
        }
        let extremes = |matrix: &Array2<f64>| {
            let (_, values, _) = matrix.view().svd(false, false).expect("a thin SVD");
            let largest = values.iter().fold(0.0_f64, |acc, &value| acc.max(value));
            let smallest = values.iter().fold(f64::INFINITY, |acc, &value| acc.min(value));
            (smallest, largest, factor_singular_band(matrix.nrows(), matrix.ncols(), largest))
        };
        let (sigma_e, _, band_e) = extremes(&cycled);
        let (sigma_b, _, band_b) = extremes(&full);
        let (phi_min, phi_max, band_phi) = extremes(&phi);
        RankRelation {
            sigma_e,
            band_e,
            sigma_b,
            band_b,
            phi_min,
            phi_max,
            band_phi,
            delta: up(delta_squares.sqrt()),
        }
    }

    #[test]
    fn the_basis_is_resolved_exactly_when_the_cycled_rows_are() {
        // `Eᵀ = B Φ + Δᵀ` with `Φ` square, so `σ_p(B) σ_p(Φ) − ‖Δ‖ ≤ σ_p(E) ≤ σ_p(B) σ_1(Φ) + ‖Δ‖`,
        // each computed singular value within its backward-error band.
        for (duplicate, seed) in [(false, 41), (true, 41)] {
            let (table, planting) = plant(&ALL, false, duplicate, seed);
            let r = rank_relation(&table);
            let upper_side = up(up(up(up(r.sigma_b + r.band_b) * up(r.phi_max + r.band_phi)) + r.delta) + r.band_e);
            assert!(r.sigma_e <= upper_side, "duplicate={duplicate}: σ_p(E) {:e} above {upper_side:e}", r.sigma_e);
            let lower_side = ((r.sigma_b - r.band_b).next_down() * (r.phi_min - r.band_phi).next_down()
                - r.delta
                - r.band_e)
                .next_down();
            assert!(r.sigma_e >= lower_side, "duplicate={duplicate}: σ_p(E) {:e} below {lower_side:e}", r.sigma_e);
            if duplicate {
                // The planted factors repeat a plane, so the exact `σ_p(E)` of the planted rows is 0 and
                // the computed one is within the planting error and its band; the relation then pins
                // `σ_p(B)` to the same scale.
                assert!(r.sigma_e <= up(planting + r.band_e), "σ_p(E) {:e} is not deficient", r.sigma_e);
                let pinned = up(up(up(r.sigma_e + r.delta) + r.band_e) / (r.phi_min - r.band_phi).next_down()) + r.band_b;
                assert!(r.sigma_b <= up(pinned), "σ_p(B) {:e} above its pinned scale {pinned:e}", r.sigma_b);
            } else {
                let floor = ((r.sigma_e - r.delta - r.band_e).next_down() / up(r.phi_max + r.band_phi)
                    - r.band_b)
                    .next_down();
                assert!(floor > r.band_b, "σ_p(B) floor {floor:e} is not resolved above {:e}", r.band_b);
                assert!(r.sigma_b >= floor);
            }
        }
    }

    /// The largest `evaluation_band(1, |entry|, 0)` over a matrix: one rounding per entry, a
    /// difference of two characters, which carries no underflow allowance.
    fn one_rounding(matrix: ArrayView2<'_, f64>) -> f64 {
        matrix.iter().fold(0.0_f64, |acc, value| acc.max(evaluation_band(1, value.abs(), 0.0)))
    }

    /// The derived bound on the frequency edit's exact residual against `e_{a+s}`: `e_a = B φ_a + δ_a`, and
    /// each computed coordinate is `φ_{a+s} − φ_a` from the same characters with one rounding `η`, so the
    /// exact residual of the computed factors is `δ_a − δ_{a+s} + U η`, at most `2 r_E + ‖U‖ max|η|`.
    fn every_plane_bound(table: &Array2<f64>, full: ArrayView2<'_, f64>, edit: &CyclicFrequencyEdit) -> f64 {
        let reconstruction = reconstruction_defect(table, full, &ALL);
        up(up(2.0 * reconstruction) + up(inf_norm_upper(edit.right.view()) * one_rounding(edit.left.view())))
    }

    #[test]
    fn the_frequency_edit_of_every_plane_reproduces_every_shift_within_a_derived_bound() {
        let table = plant(&ALL, false, false, 2951).0;
        let cycle = declared_cycle();
        let planes = cyclic_planes(table.view(), &cycle).expect("closed-form planes");
        let basis = planes.basis(&ALL).expect("every plane");
        let full = full_basis(planes.mean.view(), basis.view());
        let spacing = minimum_row_spacing(&table);
        for shift in 1..LENGTH {
            let edit = frequency_edit(&cycle, basis.view(), &ALL, shift).expect("a declared cycle");
            let residual = shift_residual(table.view(), &cycle, &edit).expect("matching shapes");
            let bound = every_plane_bound(&table, full.view(), &edit);
            assert!(
                residual.max_abs_residual <= up(bound + residual.max_residual_band),
                "shift {shift}: residual {:e} exceeds its derived bound {bound:e}",
                residual.max_abs_residual
            );
            // Magnitude floor: a bound at or above the row spacing could not tell P^s from another
            // permutation, so agreement within it would say nothing.
            assert!(
                up(bound + residual.max_residual_band) < spacing,
                "shift {shift}: derived bound {bound:e} does not resolve the row spacing {spacing:e}"
            );
            assert_eq!(residual.region, ShiftRegion { shift, cycled_rows: LENGTH, columns: WIDTH });
            assert!(matches!(
                residual.residual_status().expect("a finite bound"),
                EvidenceStatus::UniformBound { upper, .. } if upper >= residual.max_abs_residual
            ));
            // The row outside the cycle is not touched at all.
            assert!(edit.left.row(FIXED_ROW).iter().all(|&value| value == 0.0));
            assert_eq!(residual.max_abs_fixed_row_change, 0.0);
        }
    }

    #[test]
    fn a_plane_edited_with_another_frequencys_characters_misses_its_shift() {
        // One planted plane at frequency 1. The right label reproduces shift 1 within the derived bound.
        let table = plant(&[1], false, false, 17).0;
        let cycle = declared_cycle();
        let planes = cyclic_planes(table.view(), &cycle).expect("closed-form planes");
        let basis = planes.basis(&[1]).expect("plane 1");
        let full = full_basis(planes.mean.view(), basis.view());
        let reconstruction = reconstruction_defect(&table, full.view(), &[1]);
        let correct = frequency_edit(&cycle, basis.view(), &[1], 1).expect("a declared cycle");
        let at_one = shift_residual(table.view(), &cycle, &correct).expect("matching shapes");
        let bound = up(up(2.0 * reconstruction) + up(inf_norm_upper(basis.view()) * one_rounding(correct.left.view())));
        assert!(
            at_one.max_abs_residual <= up(bound + at_one.max_residual_band),
            "the right label misses shift 1 by {:e}, beyond its derived bound {bound:e}",
            at_one.max_abs_residual
        );
        // Plane 1 moved along frequency 2's characters is not the shift: the same bound refutes it, with its
        // own coordinate rounding added, at a held-out entry.
        let wrong = frequency_edit(&cycle, basis.view(), &[2], 1).expect("a declared cycle");
        let missed = shift_residual(table.view(), &cycle, &wrong).expect("matching shapes");
        let wrong_bound =
            up(up(2.0 * reconstruction) + up(inf_norm_upper(basis.view()) * one_rounding(wrong.left.view())));
        assert!(
            (missed.max_abs_residual - missed.max_residual_band).next_down() > wrong_bound,
            "the wrong label misses shift 1 by only {:e}, inside the derived bound {wrong_bound:e}",
            missed.max_abs_residual
        );
    }

    #[test]
    fn undeclared_and_malformed_declarations_are_refused_with_typed_errors() {
        assert!(matches!(
            RowCycle::from_successor(&[1, 2, 0, 4, 5, 3, 6]),
            Err(CyclicActionError::NotSingleCycle { cycles: 2 })
        ));
        assert!(matches!(
            RowCycle::from_successor(&[1, 2, 3, 0, 4]),
            Err(CyclicActionError::EvenCycle { length: 4 })
        ));
        assert!(matches!(
            RowCycle::from_successor(&[1, 1, 0]),
            Err(CyclicActionError::NotPermutation { image: 1 })
        ));
        assert!(matches!(
            RowCycle::from_successor(&[0, 1, 2]),
            Err(CyclicActionError::CycleTooShort { length: 0 })
        ));
        assert!(matches!(
            RowCycle::from_successor(&[3, 0, 1]),
            Err(CyclicActionError::RowOutOfRange { row: 3, rows: 3 })
        ));
        // Positive control: one odd cycle with a fixed row, starting at its smallest moved row.
        let cycle = declared_cycle();
        assert_eq!(cycle.rows(), &CYCLE[..]);
        assert_eq!(cycle.plane_count(), 3);

        let table = plant(&ALL, false, false, 5).0;
        let planes = cyclic_planes(table.view(), &cycle).expect("closed-form planes");
        let intact = planes.basis(&[1, 2]).expect("two planes");
        let accepted = frequency_edit(&cycle, intact.view(), &[1, 2], 1).expect("a declared cycle");
        assert_eq!(accepted.left.dim(), (LENGTH + 1, 4));
        // A basis of the wrong width for its subset, and a non-finite basis entry, are refused typed.
        assert!(matches!(
            frequency_edit(&cycle, intact.view(), &[1], 1),
            Err(CyclicActionError::ShapeMismatch { what: "basis columns against 2|S|", expected: 2, found: 4 })
        ));
        let mut poisoned = intact.clone();
        poisoned[[5, 3]] = f64::NAN;
        assert!(matches!(
            frequency_edit(&cycle, poisoned.view(), &[1, 2], 1),
            Err(CyclicActionError::NonFinite { row: 5, col: 3 })
        ));
        assert!(matches!(
            frequency_edit(&cycle, intact.view(), &[2, 1], 1),
            Err(CyclicActionError::InvalidFrequencies { .. })
        ));
        assert!(matches!(planes.basis(&[2, 1]), Err(CyclicActionError::InvalidFrequencies { .. })));
        assert!(matches!(planes.basis(&[4]), Err(CyclicActionError::InvalidFrequencies { .. })));
    }

    /// The superseded least-squares edit of `subset` at `shift`, rebuilt here as a control: each row's
    /// coordinates `X_U e_a` on `B_S = [c_0, U_S]`, `X B_S = I` from the rank-aware QR owner, turned by
    /// `R_{S,s} − I`. Its left factor, `rows × 2|S|`, multiplies `U_Sᵀ` like the frequency edit's.
    fn least_squares_left(
        table: &Array2<f64>,
        mean: ArrayView1<'_, f64>,
        basis: ArrayView2<'_, f64>,
        subset: &[usize],
        shift: usize,
    ) -> Array2<f64> {
        let full = full_basis(mean, basis);
        let identity = Array2::<f64>::eye(WIDTH);
        let solved = col_piv_qr_solve_lstsq(
            FaerArrayView::new(&full).as_ref(),
            FaerArrayView::new(&identity).as_ref(),
        );
        let plane_columns = basis.ncols();
        let coordinates_of = Array2::from_shape_fn((plane_columns, WIDTH), |(r, c)| solved[(1 + r, c)]);
        let mut block = Array2::<f64>::zeros((plane_columns, plane_columns));
        for (plane, &frequency) in subset.iter().enumerate() {
            let (sin, cos) = plane_angle(frequency, shift, LENGTH).sin_cos();
            let at = 2 * plane;
            block[[at, at]] = cos - 1.0;
            block[[at, at + 1]] = -sin;
            block[[at + 1, at]] = sin;
            block[[at + 1, at + 1]] = cos - 1.0;
        }
        table.dot(&coordinates_of.t()).dot(&block.t())
    }

    #[test]
    fn a_frequency_edit_turns_only_its_planes_where_the_least_squares_edit_turns_the_others_too() {
        // Three mutually skewed planes; the subset is the middle one.
        let table = plant(&ALL, false, false, 2951).0;
        let cycle = declared_cycle();
        let planes = cyclic_planes(table.view(), &cycle).expect("closed-form planes");
        let every = planes.basis(&ALL).expect("every plane");
        let full = full_basis(planes.mean.view(), every.view());
        let reconstruction = reconstruction_defect(&table, full.view(), &ALL);
        let subset = [2];
        let basis = planes.basis(&subset).expect("plane 2");
        let norm_basis = inf_norm_upper(basis.view());
        for shift in 1..LENGTH {
            let edit = frequency_edit(&cycle, basis.view(), &subset, shift).expect("a declared cycle");
            let least_squares = least_squares_left(&table, planes.mean.view(), basis.view(), &subset, shift);
            // The target turns plane 2 of every cycled row and keeps planes 1 and 3: `B φ'_a`, with φ'_a
            // the characters at `a` except plane 2's at `a + s`. The frequency edit's exact row is
            // `B φ'_a + δ_a + U_2 η`.
            let slack = up(reconstruction + up(norm_basis * one_rounding(edit.left.view())));
            let (mut worst, mut least_squares_miss) = (f64::NEG_INFINITY, f64::NEG_INFINITY);
            for (position, &row) in CYCLE.iter().enumerate() {
                let mut turned = characters(&ALL, position);
                let ahead = characters(&ALL, (position + shift) % LENGTH);
                turned[3] = ahead[3];
                turned[4] = ahead[4];
                for j in 0..WIDTH {
                    let (mut target, mut target_sum) = (0.0, 0.0);
                    for t in 0..full.ncols() {
                        let term = full[[j, t]] * turned[t];
                        target += term;
                        target_sum = up(target_sum + up(term.abs()));
                    }
                    let target_band = evaluation_band(full.ncols(), target_sum, full.ncols() as f64);
                    for (factors, miss) in [(&edit.left, &mut worst), (&least_squares, &mut least_squares_miss)] {
                        let (mut edited, mut edited_sum) = (table[[row, j]], table[[row, j]].abs());
                        for t in 0..basis.ncols() {
                            let term = factors[[row, t]] * basis[[j, t]];
                            edited += term;
                            edited_sum = up(edited_sum + up(term.abs()));
                        }
                        let band = up(up(evaluation_band(basis.ncols() + 1, edited_sum, basis.ncols() as f64) + target_band)
                            + evaluation_band(1, up(edited.abs() + target.abs()), 0.0));
                        *miss = miss.max((edited - target).abs() - band);
                    }
                }
            }
            assert!(worst <= slack, "shift {shift}: the frequency edit misses its target by {worst:e}, beyond {slack:e}");
            // Positive control: the least-squares edit turns the other planes' content that projects onto
            // plane 2's span too, and misses the same target by more than the same slack.
            assert!(
                least_squares_miss.next_down() > slack,
                "shift {shift}: the least-squares edit's miss {least_squares_miss:e} is inside the slack {slack:e}"
            );
            // And it moves the row outside the cycle, which the frequency edit leaves exactly fixed.
            assert!(edit.left.row(FIXED_ROW).iter().all(|&value| value == 0.0));
            let fixed_change =
                least_squares.row(FIXED_ROW).dot(&basis.t()).iter().fold(0.0_f64, |acc, v| acc.max(v.abs()));
            assert!(fixed_change > 0.0, "shift {shift}: the least-squares edit left the fixed row unchanged");
        }
    }

    #[test]
    fn a_plane_program_sends_its_subset_and_decodes_every_basis_real_within_the_declared_step() {
        let table = plant(&ALL, false, false, 7).0;
        let cycle = declared_cycle();
        let planes = cyclic_planes(table.view(), &cycle).expect("closed-form planes");
        let precision = DeclaredPrecision::new(20).expect("20 fraction bits");
        let one = plane_program_code(&planes, &[2], precision).expect("one plane");
        let three = plane_program_code(&planes, &ALL, precision).expect("every plane");
        let mut message = BitString::new();
        three.basis.write(&mut message).expect("one message");
        assert_eq!(message.len_bits(), three.basis_bits);
        let mut reader = message.reader();
        let decoded = LatticeCode::read(&mut reader).expect("the message decodes");
        reader.finish().expect("nothing follows the message");
        let basis = planes.basis(&ALL).expect("every plane");
        assert_eq!(decoded.indices().len(), 2 * WIDTH * ALL.len());
        for (&index, value) in decoded.indices().iter().zip(basis.t().iter()) {
            assert!((index as f64 * precision.step() - value).abs() <= precision.worst_case_error());
        }
        assert!(three.total_bits() > one.total_bits());
        assert_eq!(three.total_bits(), three.subset_bits + three.basis_bits);
    }
}
