//! Matrix-free structured edits of a native linear use site (#2951).
//!
//! A linear use site with native weight `W*` (`d_out × d_in`, the torch `Linear`
//! layout) executes under the residual anchor `Θ(m) = m_Δ W* + Σ_k s_k u_k v_kᵀ`.
//! The rank-one terms carry the masks: `s_k = (m_c − m_Δ)·coefficient` for the
//! component or field basis that owns term `k`. [`apply_anchored_linear`] applies
//! `Θ(m)` to the site's input rows without forming the `d_out × d_in` edit.
//! [`edit_frobenius_contractions`] and [`edit_factor_cotangents`] return the P16
//! pullback pieces of `G = Σ_r g_r h_rᵀ` (`g` the output cotangent rows, `h` the
//! input rows) without forming `G`.
//!
//! [`FactoredEdit`] owns an edit's factors; [`FactorView`] borrows them from
//! whoever stores them, and every kernel reads the borrowed form.
//!
//! The input rows must be the site's CURRENT intervened input: the activation the
//! edited upstream network produced. An edit applied to a clean input cached
//! before an upstream edit answers a different experiment.
//!
//! Rows are observations. Every kernel streams rows in tiles sized by the library
//! row-chunk rule ([`byte_balanced_row_chunk`]) and reserves its result and tile
//! scratch on the process [`MemoryGovernor`] before allocating, so a request that
//! cannot fit is a typed refusal rather than an out-of-memory abort.

use faer::Accum;
use faer::linalg::matmul::matmul;
use gam_linalg::faer_ndarray::{
    FaerArrayView, array2_to_matmut, fast_ab_into, matmul_parallelism,
};
use gam_runtime::resource::{
    Governed, MemoryGovernor, MemoryReservation, MemoryReservationError, byte_balanced_row_chunk,
    dense_f64_bytes,
};
use ndarray::{Array1, Array2, ArrayView1, ArrayView2, Axis, ShapeBuilder, s};

/// Refusal from a matrix-free edit kernel.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ApplyError {
    /// An operand's shape disagrees with the site. A vector's shape is `(len, 1)`.
    Shape {
        operand: &'static str,
        expected: (usize, usize),
        found: (usize, usize),
    },
    /// An edit factor holds a non-finite entry.
    NonFinite { operand: &'static str },
    /// The byte footprint of the request does not fit in `usize`.
    SizeOverflow { context: &'static str },
    /// The process memory governor refused the request's footprint.
    Memory(MemoryReservationError),
}

impl std::fmt::Display for ApplyError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Shape {
                operand,
                expected,
                found,
            } => write!(
                f,
                "structured edit refused: {operand} has shape {found:?}, the site needs {expected:?}"
            ),
            Self::NonFinite { operand } => {
                write!(f, "structured edit refused: {operand} holds a non-finite entry")
            }
            Self::SizeOverflow { context } => {
                write!(f, "{context}: the byte footprint overflows usize")
            }
            Self::Memory(err) => write!(f, "structured edit refused by the memory governor: {err}"),
        }
    }
}

impl std::error::Error for ApplyError {}

impl From<MemoryReservationError> for ApplyError {
    fn from(err: MemoryReservationError) -> Self {
        Self::Memory(err)
    }
}

/// A structured edit `Σ_k s_k u_k v_kᵀ`, owning its factors.
#[derive(Clone, Debug)]
pub struct FactoredEdit {
    left: Array2<f64>,
    right: Array2<f64>,
}

impl FactoredEdit {
    /// `left` is `d_out × R` with columns `u_k`; `right` is `d_in × R` with
    /// columns `v_k`. A rank-`r` component or field basis is `r` terms. Refuses
    /// mismatched ranks and non-finite entries.
    pub fn new(left: Array2<f64>, right: Array2<f64>) -> Result<Self, ApplyError> {
        FactorView::new(left.view(), right.view())?;
        Ok(Self { left, right })
    }

    /// The borrowed form the kernels read.
    pub fn view(&self) -> FactorView<'_> {
        FactorView {
            left: self.left.view(),
            right: self.right.view(),
        }
    }

    pub fn left(&self) -> ArrayView2<'_, f64> {
        self.left.view()
    }

    pub fn right(&self) -> ArrayView2<'_, f64> {
        self.right.view()
    }

    pub fn output_dim(&self) -> usize {
        self.left.nrows()
    }

    pub fn input_dim(&self) -> usize {
        self.right.nrows()
    }

    pub fn term_count(&self) -> usize {
        self.left.ncols()
    }
}

/// The factors of a structured edit, borrowed from whoever stores them.
#[derive(Clone, Copy, Debug)]
pub struct FactorView<'a> {
    left: ArrayView2<'a, f64>,
    right: ArrayView2<'a, f64>,
}

impl<'a> FactorView<'a> {
    /// The contract of [`FactoredEdit::new`], without taking ownership.
    pub fn new(left: ArrayView2<'a, f64>, right: ArrayView2<'a, f64>) -> Result<Self, ApplyError> {
        expect_dim(
            "right factor",
            (right.nrows(), left.ncols()),
            right.dim(),
        )?;
        expect_finite("left factor", left)?;
        expect_finite("right factor", right)?;
        Ok(Self { left, right })
    }

    pub fn left(&self) -> ArrayView2<'a, f64> {
        self.left
    }

    pub fn right(&self) -> ArrayView2<'a, f64> {
        self.right
    }

    pub fn output_dim(&self) -> usize {
        self.left.nrows()
    }

    pub fn input_dim(&self) -> usize {
        self.right.nrows()
    }

    pub fn term_count(&self) -> usize {
        self.left.ncols()
    }
}

/// The bytes one kernel call holds: its result, and the scratch of one row tile.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct EditFootprint {
    pub tile_rows: usize,
    pub result_bytes: usize,
    pub scratch_bytes: usize,
}

impl EditFootprint {
    /// [`apply_anchored_linear`] over `n_rows` rows, with `active_terms` of the
    /// edit's `terms` carrying a nonzero scale. A tile holds the native product
    /// and the factor contribution (`tile × d_out` each), one product fallback of
    /// the same size, and the term projections (`tile × active`). A partial mask
    /// also copies the active factor columns and scales.
    pub fn anchored_linear(
        n_rows: usize,
        input_dim: usize,
        output_dim: usize,
        terms: usize,
        active_terms: usize,
    ) -> Result<Self, ApplyError> {
        const CONTEXT: &str = "anchored linear apply";
        let tile_cols = checked_sum(&[output_dim, output_dim, active_terms], CONTEXT)?;
        let tile_rows = byte_balanced_row_chunk(tile_cols, n_rows);
        let result_bytes = f64_bytes(n_rows, output_dim, CONTEXT)?;
        let tile_bytes = f64_bytes(tile_rows, tile_cols, CONTEXT)?;
        let copy_bytes = if active_terms == terms {
            0
        } else {
            let copy_rows = checked_sum(&[output_dim, input_dim, 1], CONTEXT)?;
            f64_bytes(copy_rows, active_terms, CONTEXT)?
        };
        Ok(Self {
            tile_rows,
            result_bytes,
            scratch_bytes: checked_sum(&[tile_bytes, copy_bytes], CONTEXT)?,
        })
    }

    /// [`edit_frobenius_contractions`] over `n_rows` rows: the two `tile × R`
    /// projections and one length-`R` column sum.
    pub fn frobenius_contractions(n_rows: usize, terms: usize) -> Result<Self, ApplyError> {
        const CONTEXT: &str = "edit frobenius contractions";
        let tile_cols = checked_sum(&[terms, terms], CONTEXT)?;
        let tile_rows = byte_balanced_row_chunk(tile_cols, n_rows);
        let result_bytes = f64_bytes(terms, 1, CONTEXT)?;
        let tile_bytes = f64_bytes(tile_rows, tile_cols, CONTEXT)?;
        Ok(Self {
            tile_rows,
            result_bytes,
            scratch_bytes: checked_sum(&[tile_bytes, result_bytes], CONTEXT)?,
        })
    }

    /// [`edit_factor_cotangents`] over `n_rows` rows: the two `tile × R`
    /// projections. Each tile's product adds straight into the result.
    pub fn factor_cotangents(
        n_rows: usize,
        input_dim: usize,
        output_dim: usize,
        terms: usize,
    ) -> Result<Self, ApplyError> {
        const CONTEXT: &str = "edit factor cotangents";
        let tile_cols = checked_sum(&[terms, terms], CONTEXT)?;
        let tile_rows = byte_balanced_row_chunk(tile_cols, n_rows);
        let result_rows = checked_sum(&[output_dim, input_dim], CONTEXT)?;
        let result_bytes = f64_bytes(result_rows, terms, CONTEXT)?;
        let tile_bytes = f64_bytes(tile_rows, tile_cols, CONTEXT)?;
        Ok(Self {
            tile_rows,
            result_bytes,
            scratch_bytes: tile_bytes,
        })
    }

    /// Charge the footprint to the process memory governor before allocating.
    pub fn reserve(&self, context: &str) -> Result<EditReservation, ApplyError> {
        let governor = MemoryGovernor::global();
        let result = governor.try_reserve(self.result_bytes, context)?;
        let scratch = governor.try_reserve(self.scratch_bytes, context)?;
        Ok(EditReservation { result, scratch })
    }
}

/// Live governor charges of one kernel call. The result charge moves into the
/// returned value; the scratch charge is released when the call returns.
#[derive(Debug)]
pub struct EditReservation {
    result: MemoryReservation,
    scratch: MemoryReservation,
}

impl EditReservation {
    pub fn result_bytes(&self) -> usize {
        self.result.bytes()
    }

    pub fn scratch_bytes(&self) -> usize {
        self.scratch.bytes()
    }
}

/// The all-on path of a linear use site: `h W*ᵀ` over the input rows, executed
/// on the original tensor. [`apply_anchored_linear`] with `m_Δ = 1` and every
/// term scale zero executes exactly this, bit for bit, and never reads the edit
/// factors.
pub fn native_linear(
    native: ArrayView2<'_, f64>,
    input: ArrayView2<'_, f64>,
) -> Result<Governed<Array2<f64>>, ApplyError> {
    expect_dim("input rows", (input.nrows(), native.ncols()), input.dim())?;
    let footprint =
        EditFootprint::anchored_linear(input.nrows(), native.ncols(), native.nrows(), 0, 0)?;
    let reservation = footprint.reserve("native linear apply")?;
    let output = execute_anchored(native, 1.0, None, input, footprint.tile_rows);
    Ok(reservation.result.bind(output))
}

/// `Θ(m) h = m_Δ W* h + Σ_k s_k u_k (v_kᵀ h)` for every input row, with `anchor`
/// = `m_Δ` and `term_scales` = `s`. Terms with a zero scale are never read, so
/// the all-on setting executes [`native_linear`].
pub fn apply_anchored_linear(
    native: ArrayView2<'_, f64>,
    anchor: f64,
    edit: FactorView<'_>,
    term_scales: ArrayView1<'_, f64>,
    input: ArrayView2<'_, f64>,
) -> Result<Governed<Array2<f64>>, ApplyError> {
    let terms = edit.term_count();
    expect_dim("input rows", (input.nrows(), native.ncols()), input.dim())?;
    expect_dim("left factor", (native.nrows(), terms), edit.left.dim())?;
    expect_dim("right factor", (native.ncols(), terms), edit.right.dim())?;
    expect_dim("term scales", (terms, 1), (term_scales.len(), 1))?;
    let active: Vec<usize> = term_scales
        .iter()
        .enumerate()
        .filter_map(|(term, scale)| (*scale != 0.0).then_some(term))
        .collect();
    let footprint = EditFootprint::anchored_linear(
        input.nrows(),
        native.ncols(),
        native.nrows(),
        terms,
        active.len(),
    )?;
    let reservation = footprint.reserve("anchored linear apply")?;
    let output = if active.is_empty() {
        execute_anchored(native, anchor, None, input, footprint.tile_rows)
    } else if active.len() == terms {
        let all_terms = ActiveTerms {
            left: edit.left,
            right: edit.right,
            scales: term_scales,
        };
        execute_anchored(native, anchor, Some(all_terms), input, footprint.tile_rows)
    } else {
        let left = edit.left.select(Axis(1), &active);
        let right = edit.right.select(Axis(1), &active);
        let scales = term_scales.select(Axis(0), &active);
        let active_terms = ActiveTerms {
            left: left.view(),
            right: right.view(),
            scales: scales.view(),
        };
        execute_anchored(native, anchor, Some(active_terms), input, footprint.tile_rows)
    };
    Ok(reservation.result.bind(output))
}

/// `⟨G, u_k v_kᵀ⟩_F = Σ_r (g_rᵀ u_k)(v_kᵀ h_r)` for every term `k`, with `G` never
/// formed. `∂ℓ/∂s_k` is this contraction; a field basis's Frobenius pairing
/// `⟨G, B_j⟩_F` is the sum over its terms (P16 labels and coefficients). A tied
/// tensor sums the per-use contractions.
pub fn edit_frobenius_contractions(
    edit: FactorView<'_>,
    cotangent: ArrayView2<'_, f64>,
    input: ArrayView2<'_, f64>,
) -> Result<Governed<Array1<f64>>, ApplyError> {
    expect_pullback_rows(edit, cotangent, input)?;
    let footprint = EditFootprint::frobenius_contractions(input.nrows(), edit.term_count())?;
    let reservation = footprint.reserve("edit frobenius contractions")?;
    let sums = contractions_tiled(edit, cotangent, input, footprint.tile_rows);
    Ok(reservation.result.bind(sums))
}

/// `G V` and `Gᵀ U` of `G = Σ_r g_r h_rᵀ`, with `G` never formed.
#[derive(Clone, Debug)]
pub struct FactorCotangents {
    /// `G V`, `d_out × R`. P16: `∂ℓ/∂u_k = s_k (G V)_k`.
    pub left: Array2<f64>,
    /// `Gᵀ U`, `d_in × R`. P16: `∂ℓ/∂v_k = s_k (Gᵀ U)_k`.
    pub right: Array2<f64>,
}

/// The low-rank factor pullbacks of P16 before their term scales. The
/// contraction of term `k` is `u_kᵀ (G V)_k`. A tied tensor sums the per-use
/// cotangents.
pub fn edit_factor_cotangents(
    edit: FactorView<'_>,
    cotangent: ArrayView2<'_, f64>,
    input: ArrayView2<'_, f64>,
) -> Result<Governed<FactorCotangents>, ApplyError> {
    expect_pullback_rows(edit, cotangent, input)?;
    let footprint = EditFootprint::factor_cotangents(
        input.nrows(),
        edit.input_dim(),
        edit.output_dim(),
        edit.term_count(),
    )?;
    let reservation = footprint.reserve("edit factor cotangents")?;
    let cotangents = cotangents_tiled(edit, cotangent, input, footprint.tile_rows);
    Ok(reservation.result.bind(cotangents))
}

/// The terms an apply reads: the nonzero-scale columns of the edit.
struct ActiveTerms<'a> {
    left: ArrayView2<'a, f64>,
    right: ArrayView2<'a, f64>,
    scales: ArrayView1<'a, f64>,
}

fn execute_anchored(
    native: ArrayView2<'_, f64>,
    anchor: f64,
    terms: Option<ActiveTerms<'_>>,
    input: ArrayView2<'_, f64>,
    tile_rows: usize,
) -> Array2<f64> {
    let (n_rows, output_dim) = (input.nrows(), native.nrows());
    let native_transpose = native.t();
    let mut output = Array2::<f64>::zeros((n_rows, output_dim));
    let mut start = 0;
    while start < n_rows {
        let end = n_rows.min(start + tile_rows);
        let rows = input.slice(s![start..end, ..]);
        let mut tile = output.slice_mut(s![start..end, ..]);
        if anchor != 0.0 {
            let mut native_part = Array2::<f64>::zeros((end - start, output_dim));
            fast_ab_into(&rows, &native_transpose, &mut native_part);
            if anchor != 1.0 {
                native_part *= anchor;
            }
            tile.assign(&native_part);
        }
        if let Some(terms) = &terms {
            let mut projections = Array2::<f64>::zeros((end - start, terms.scales.len()));
            fast_ab_into(&rows, &terms.right, &mut projections);
            projections *= &terms.scales;
            let mut contribution = Array2::<f64>::zeros((end - start, output_dim));
            fast_ab_into(&projections, &terms.left.t(), &mut contribution);
            tile += &contribution;
        }
        start = end;
    }
    output
}

fn contractions_tiled(
    edit: FactorView<'_>,
    cotangent: ArrayView2<'_, f64>,
    input: ArrayView2<'_, f64>,
    tile_rows: usize,
) -> Array1<f64> {
    let (n_rows, terms) = (input.nrows(), edit.term_count());
    let mut sums = Array1::<f64>::zeros(terms);
    let mut start = 0;
    while start < n_rows {
        let end = n_rows.min(start + tile_rows);
        let mut output_projections = Array2::<f64>::zeros((end - start, terms));
        fast_ab_into(
            &cotangent.slice(s![start..end, ..]),
            &edit.left,
            &mut output_projections,
        );
        let mut input_projections = Array2::<f64>::zeros((end - start, terms));
        fast_ab_into(
            &input.slice(s![start..end, ..]),
            &edit.right,
            &mut input_projections,
        );
        output_projections *= &input_projections;
        sums += &output_projections.sum_axis(Axis(0));
        start = end;
    }
    sums
}

fn cotangents_tiled(
    edit: FactorView<'_>,
    cotangent: ArrayView2<'_, f64>,
    input: ArrayView2<'_, f64>,
    tile_rows: usize,
) -> FactorCotangents {
    let (n_rows, terms) = (input.nrows(), edit.term_count());
    let mut left = Array2::<f64>::zeros((edit.output_dim(), terms));
    let mut right = Array2::<f64>::zeros((edit.input_dim(), terms));
    let mut start = 0;
    while start < n_rows {
        let end = n_rows.min(start + tile_rows);
        let output_rows = cotangent.slice(s![start..end, ..]);
        let input_rows = input.slice(s![start..end, ..]);
        // Column-major projections make `gᵀ (h V)` a product of column-major operands
        // (the transpose of row-major rows is column-major). faer multiplies those at full
        // speed and adds the product into the result with no temporary: at 4096 terms and
        // LLM width, 2.6-3.1 s against 8.2 s for a row-major temporary (#2951, job 1181471).
        let mut input_projections = Array2::<f64>::zeros((end - start, terms).f());
        fast_ab_into(&input_rows, &edit.right, &mut input_projections);
        accumulate_transposed_product(&mut left, output_rows, input_projections.view());
        let mut output_projections = Array2::<f64>::zeros((end - start, terms).f());
        fast_ab_into(&output_rows, &edit.left, &mut output_projections);
        accumulate_transposed_product(&mut right, input_rows, output_projections.view());
        start = end;
    }
    FactorCotangents { left, right }
}

/// `target += rowsᵀ projections`.
fn accumulate_transposed_product(
    target: &mut Array2<f64>,
    rows: ArrayView2<'_, f64>,
    projections: ArrayView2<'_, f64>,
) {
    let par = matmul_parallelism(target.nrows(), target.ncols(), rows.nrows());
    let lhs = FaerArrayView::new(&rows);
    let rhs = FaerArrayView::new(&projections);
    let mut dst = array2_to_matmut(target);
    matmul(
        dst.as_mut(),
        Accum::Add,
        lhs.as_ref().transpose(),
        rhs.as_ref(),
        1.0,
        par,
    );
}

fn expect_pullback_rows(
    edit: FactorView<'_>,
    cotangent: ArrayView2<'_, f64>,
    input: ArrayView2<'_, f64>,
) -> Result<(), ApplyError> {
    expect_dim(
        "cotangent rows",
        (input.nrows(), edit.output_dim()),
        cotangent.dim(),
    )?;
    expect_dim("input rows", (cotangent.nrows(), edit.input_dim()), input.dim())
}

fn expect_dim(
    operand: &'static str,
    expected: (usize, usize),
    found: (usize, usize),
) -> Result<(), ApplyError> {
    if expected == found {
        Ok(())
    } else {
        Err(ApplyError::Shape {
            operand,
            expected,
            found,
        })
    }
}

fn expect_finite(operand: &'static str, values: ArrayView2<'_, f64>) -> Result<(), ApplyError> {
    if values.iter().all(|value| value.is_finite()) {
        Ok(())
    } else {
        Err(ApplyError::NonFinite { operand })
    }
}

fn f64_bytes(rows: usize, cols: usize, context: &'static str) -> Result<usize, ApplyError> {
    dense_f64_bytes(rows, cols).ok_or(ApplyError::SizeOverflow { context })
}

fn checked_sum(parts: &[usize], context: &'static str) -> Result<usize, ApplyError> {
    parts
        .iter()
        .try_fold(0usize, |total, &part| total.checked_add(part))
        .ok_or(ApplyError::SizeOverflow { context })
}

#[cfg(test)]
mod tests {
    use super::*;
    use gam_linalg::roundoff::{accumulation_band, accumulation_growth};
    use ndarray::Zip;

    /// `γ_k·A` for an exact absolute sum `A`, from its computed value `Â`. `Â` is an
    /// accumulation of nonnegative terms through the same `k` operations, so
    /// `A ≤ Â/(1 − γ_k)`.
    fn rounded_band(operations: usize, computed_absolute: f64) -> f64 {
        accumulation_band(operations, computed_absolute) / (1.0 - accumulation_growth(operations))
    }

    fn fixture(rows: usize, cols: usize, phase: f64) -> Array2<f64> {
        Array2::from_shape_fn((rows, cols), |(i, j)| {
            ((i as f64 + 1.0) * (0.37 + phase) + (j as f64 + 1.0) * (0.61 - 0.5 * phase)).sin()
                + 0.25 * ((i * (j + 3)) as f64 * 0.113 + phase).cos()
        })
    }

    /// Two floating-point routes to one exact value, each within `band` of it,
    /// differ by at most `2·band` entrywise. A NaN is never within.
    fn within_band(a: &Array2<f64>, b: &Array2<f64>, band: &Array2<f64>) -> bool {
        a.dim() == b.dim()
            && Zip::from(a)
                .and(b)
                .and(band)
                .all(|&x, &y, &e| (x - y).abs() <= 2.0 * e)
    }

    /// Output entry `(r, o)` of `Θ(m) h` unfolds into `N = d_in·(R + 1)` products
    /// of at most four factors (`m_Δ W h` or `s u v h`). Each product rounds at most
    /// three times, then at most `N − 1` additions, so `k = N + 2` over their
    /// absolute sum, whichever summation order a kernel takes.
    fn anchored_band(
        native: &Array2<f64>,
        anchor: f64,
        left: &Array2<f64>,
        right: &Array2<f64>,
        scales: &Array1<f64>,
        input: &Array2<f64>,
    ) -> Array2<f64> {
        let absolute_edit = native.mapv(f64::abs) * anchor.abs()
            + left
                .mapv(f64::abs)
                .dot(&Array2::from_diag(&scales.mapv(f64::abs)))
                .dot(&right.mapv(f64::abs).t());
        let absolute = input.mapv(f64::abs).dot(&absolute_edit.t());
        let products = input.ncols() * (scales.len() + 1);
        absolute.mapv(|sum| rounded_band(products + 2, sum))
    }

    #[test]
    fn anchored_apply_equals_direct_edited_tensor_execution_under_every_mask_kind() {
        let (n_rows, input_dim, output_dim, terms) = (37, 48, 40, 6);
        let native = fixture(output_dim, input_dim, 0.11);
        let left = fixture(output_dim, terms, 0.29);
        let right = fixture(input_dim, terms, 0.47);
        let input = fixture(n_rows, input_dim, 0.83);
        let edit = FactoredEdit::new(left.clone(), right.clone()).expect("factor shapes agree");
        let cases: [(&str, f64, [f64; 6]); 4] = [
            ("binary deletion", 1.0, [0.0, -1.0, 0.0, -1.0, 0.0, 0.0]),
            ("continuous", 1.0, [-0.25, -0.5, -0.75, -0.125, -0.625, -0.875]),
            ("signed, partial anchor", 0.5, [1.7, -1.3, 0.9, -2.1, 0.4, -0.6]),
            ("residual deleted", 0.0, [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]),
        ];
        for (label, anchor, scales) in cases {
            let scales = Array1::from(scales.to_vec());
            let edited = &native * anchor + left.dot(&Array2::from_diag(&scales)).dot(&right.t());
            let direct = input.dot(&edited.t());
            let band = anchored_band(&native, anchor, &left, &right, &scales, &input);
            let factored =
                apply_anchored_linear(native.view(), anchor, edit.view(), scales.view(), input.view())
                    .expect("a small edit is admitted");
            assert!(
                within_band(&factored, &direct, &band),
                "{label}: factored apply leaves the roundoff band of direct execution"
            );
            let tiled = execute_anchored(
                native.view(),
                anchor,
                Some(ActiveTerms {
                    left: left.view(),
                    right: right.view(),
                    scales: scales.view(),
                }),
                input.view(),
                5,
            );
            assert!(
                within_band(&tiled, &direct, &band),
                "{label}: five-row tiles leave the roundoff band of direct execution"
            );
        }
        // Positive control: a 1e-6 relative change in one scale is a different
        // edit, and the band check must see it.
        let scales = Array1::from(vec![1.7, -1.3, 0.9, -2.1, 0.4, -0.6]);
        let mut perturbed = scales.clone();
        perturbed[1] *= 1.0 + 1e-6;
        let edited = &native * 0.5 + left.dot(&Array2::from_diag(&scales)).dot(&right.t());
        let direct = input.dot(&edited.t());
        let band = anchored_band(&native, 0.5, &left, &right, &scales, &input);
        let wrong =
            apply_anchored_linear(native.view(), 0.5, edit.view(), perturbed.view(), input.view())
                .expect("a small edit is admitted");
        assert!(
            !within_band(&wrong, &direct, &band),
            "the band check cannot see a 1e-6 relative change in one term scale"
        );
    }

    #[test]
    fn all_on_path_executes_the_native_tensor_without_reading_the_factors() {
        let (n_rows, input_dim, output_dim, terms) = (33, 40, 36, 4);
        let native = fixture(output_dim, input_dim, 0.21);
        // Inputs in [1, 2.25] and right factors of 1e307 make every projection
        // `v_kᵀ h ≥ 40·1e307` overflow to +inf in any summation order, so a route
        // that reads a zero-scale term produces `inf·0 = NaN`.
        let input = fixture(n_rows, input_dim, 0.57).mapv(|entry| entry.abs() + 1.0);
        let left = fixture(output_dim, terms, 0.33);
        let right = Array2::from_elem((input_dim, terms), 1e307);
        let edit = FactoredEdit::new(left.clone(), right.clone()).expect("finite factors");
        let zeros = Array1::<f64>::zeros(terms);
        let native_output =
            native_linear(native.view(), input.view()).expect("a small apply is admitted");
        let all_on = apply_anchored_linear(native.view(), 1.0, edit.view(), zeros.view(), input.view())
            .expect("a small apply is admitted");
        assert!(
            native_output.iter().all(|value| value.is_finite()),
            "the native fixture must be finite for the bit comparison to mean anything"
        );
        assert!(
            Zip::from(&*all_on)
                .and(&*native_output)
                .all(|a, b| a.to_bits() == b.to_bits()),
            "the all-on apply is not bit-identical to native execution"
        );
        // Positive control: routing the same zero scales through the factored
        // arithmetic reads the factors and poisons the output.
        let footprint = EditFootprint::anchored_linear(n_rows, input_dim, output_dim, terms, terms)
            .expect("small footprint");
        let through_factors = execute_anchored(
            native.view(),
            1.0,
            Some(ActiveTerms {
                left: left.view(),
                right: right.view(),
                scales: zeros.view(),
            }),
            input.view(),
            footprint.tile_rows,
        );
        assert!(
            !Zip::from(&through_factors)
                .and(&*native_output)
                .all(|a, b| a.to_bits() == b.to_bits()),
            "the bit check cannot see a route that reads zero-scale factors"
        );
    }

    /// Contraction `k` unfolds into `N = n·d_out·d_in` four-factor products
    /// `g u v h`, so `k = N + 2` over `Σ_r (|g_r|ᵀ|u_k|)(|v_k|ᵀ|h_r|)`.
    fn contraction_band(
        left: &Array2<f64>,
        right: &Array2<f64>,
        cotangent: &Array2<f64>,
        input: &Array2<f64>,
    ) -> Array1<f64> {
        let absolute = (cotangent.mapv(f64::abs).dot(&left.mapv(f64::abs))
            * input.mapv(f64::abs).dot(&right.mapv(f64::abs)))
        .sum_axis(Axis(0));
        let products = input.nrows() * left.nrows() * right.nrows();
        absolute.mapv(|sum| rounded_band(products + 2, sum))
    }

    fn vector_within_band(a: &Array1<f64>, b: &Array1<f64>, band: &Array1<f64>) -> bool {
        a.len() == b.len()
            && Zip::from(a)
                .and(b)
                .and(band)
                .all(|&x, &y, &e| (x - y).abs() <= 2.0 * e)
    }

    #[test]
    fn frobenius_contractions_equal_the_dense_gradient_pairing() {
        let (n_rows, input_dim, output_dim, terms) = (29, 44, 38, 5);
        let left = fixture(output_dim, terms, 0.17);
        let right = fixture(input_dim, terms, 0.53);
        let cotangent = fixture(n_rows, output_dim, 0.91);
        let input = fixture(n_rows, input_dim, 0.39);
        let edit = FactorView::new(left.view(), right.view()).expect("factor shapes agree");
        let gradient = cotangent.t().dot(&input);
        let dense = Array1::from_shape_fn(terms, |k| left.column(k).dot(&gradient.dot(&right.column(k))));
        let band = contraction_band(&left, &right, &cotangent, &input);
        let contractions = edit_frobenius_contractions(edit, cotangent.view(), input.view())
            .expect("small contractions are admitted");
        assert!(
            vector_within_band(&contractions, &dense, &band),
            "contractions leave the roundoff band of the dense pairing"
        );
        let tiled = contractions_tiled(edit, cotangent.view(), input.view(), 4);
        assert!(
            vector_within_band(&tiled, &dense, &band),
            "four-row tiles leave the roundoff band of the dense pairing"
        );
        // Positive control: dropping the last row is a different G.
        let truncated = contractions_tiled(
            edit,
            cotangent.slice(s![..n_rows - 1, ..]),
            input.slice(s![..n_rows - 1, ..]),
            4,
        );
        assert!(
            !vector_within_band(&truncated, &dense, &band),
            "the band check cannot see a missing row"
        );
    }

    #[test]
    fn factor_cotangents_equal_dense_products_and_carry_the_contractions() {
        let (n_rows, input_dim, output_dim, terms) = (31, 42, 46, 5);
        let left = fixture(output_dim, terms, 0.23);
        let right = fixture(input_dim, terms, 0.61);
        let cotangent = fixture(n_rows, output_dim, 0.77);
        let input = fixture(n_rows, input_dim, 0.19);
        let edit = FactorView::new(left.view(), right.view()).expect("factor shapes agree");
        let gradient = cotangent.t().dot(&input);
        let dense_left = gradient.dot(&right);
        let dense_right = gradient.t().dot(&left);
        // (G V)_{ik} unfolds into n·d_in three-factor products g h v, (Gᵀ U)_{jk}
        // into n·d_out products h g u: k = N + 1 for each.
        let left_band = cotangent
            .mapv(f64::abs)
            .t()
            .dot(&input.mapv(f64::abs).dot(&right.mapv(f64::abs)))
            .mapv(|sum| rounded_band(n_rows * input_dim + 1, sum));
        let right_band = input
            .mapv(f64::abs)
            .t()
            .dot(&cotangent.mapv(f64::abs).dot(&left.mapv(f64::abs)))
            .mapv(|sum| rounded_band(n_rows * output_dim + 1, sum));
        let cotangents = edit_factor_cotangents(edit, cotangent.view(), input.view())
            .expect("small cotangents are admitted");
        assert!(
            within_band(&cotangents.left, &dense_left, &left_band),
            "G V leaves the roundoff band of the dense product"
        );
        assert!(
            within_band(&cotangents.right, &dense_right, &right_band),
            "Gᵀ U leaves the roundoff band of the dense product"
        );
        let tiled = cotangents_tiled(edit, cotangent.view(), input.view(), 6);
        assert!(
            within_band(&tiled.left, &dense_left, &left_band)
                && within_band(&tiled.right, &dense_right, &right_band),
            "six-row tiles leave the roundoff band of the dense products"
        );
        let carried = Array1::from_shape_fn(terms, |k| left.column(k).dot(&cotangents.left.column(k)));
        let pairing_band = contraction_band(&left, &right, &cotangent, &input);
        let contractions = contractions_tiled(edit, cotangent.view(), input.view(), 6);
        assert!(
            vector_within_band(&carried, &contractions, &pairing_band),
            "u_kᵀ (G V)_k disagrees with the contraction of term k"
        );
        // Positive control: dropping the last row is a different G.
        let truncated = cotangents_tiled(
            edit,
            cotangent.slice(s![..n_rows - 1, ..]),
            input.slice(s![..n_rows - 1, ..]),
            6,
        );
        assert!(
            !within_band(&truncated.left, &dense_left, &left_band),
            "the band check cannot see a missing row"
        );
    }

    #[test]
    fn an_unfittable_request_is_refused_before_allocation() {
        let small = EditFootprint::anchored_linear(64, 48, 40, 6, 3).expect("small footprint");
        let admitted = small.reserve("apply admission test");
        assert!(
            admitted.is_ok(),
            "a {small:?} footprint must be admitted (positive control)"
        );
        // 2^40 rows at LLM width is 2^55 bytes of output: no host holds it.
        let huge = EditFootprint::anchored_linear(1 << 40, 4096, 4096, 64, 64)
            .expect("the footprint itself fits usize");
        assert!(
            matches!(
                huge.reserve("apply admission test"),
                Err(ApplyError::Memory(MemoryReservationError::BudgetExceeded { .. }))
            ),
            "a {huge:?} footprint was not refused by the memory governor"
        );
        assert!(
            matches!(
                EditFootprint::anchored_linear(usize::MAX, 4096, 4096, 64, 64),
                Err(ApplyError::SizeOverflow { .. })
            ),
            "an overflowing footprint was not refused"
        );
    }

    #[test]
    fn a_misshapen_or_non_finite_operand_is_refused() {
        let native = fixture(36, 40, 0.3);
        let input = fixture(9, 40, 0.4);
        let edit = FactoredEdit::new(fixture(36, 3, 0.5), fixture(40, 3, 0.6))
            .expect("factor shapes agree");
        let scales = Array1::from(vec![0.5, -0.5, 1.0]);
        assert!(
            apply_anchored_linear(native.view(), 1.0, edit.view(), scales.view(), input.view()).is_ok(),
            "a well-shaped apply must be admitted (positive control)"
        );
        let short = Array1::from(vec![0.5, -0.5]);
        assert_eq!(
            apply_anchored_linear(native.view(), 1.0, edit.view(), short.view(), input.view()).err(),
            Some(ApplyError::Shape {
                operand: "term scales",
                expected: (3, 1),
                found: (2, 1),
            })
        );
        assert!(
            matches!(
                FactoredEdit::new(fixture(36, 3, 0.5), fixture(40, 2, 0.6)),
                Err(ApplyError::Shape { operand: "right factor", .. })
            ),
            "mismatched factor ranks were not refused"
        );
        let mut poisoned = fixture(40, 3, 0.6);
        poisoned[[7, 2]] = f64::NAN;
        assert_eq!(
            FactoredEdit::new(fixture(36, 3, 0.5), poisoned).err(),
            Some(ApplyError::NonFinite { operand: "right factor" })
        );
        let infinite = fixture(36, 3, 0.5).mapv(|entry| entry / 0.0);
        assert!(
            matches!(
                FactorView::new(infinite.view(), fixture(40, 3, 0.6).view()),
                Err(ApplyError::NonFinite { operand: "left factor" })
            ),
            "a non-finite left factor was not refused"
        );
    }
}
