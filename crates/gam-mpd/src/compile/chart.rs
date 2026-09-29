//! Fixed-rank charts of a low-rank joint operator, and their lift to the native factors.
//!
//! A joint operator `Z = W₁ W₂` (`m × n`, inner dimension `r`, both factors of rank `r`) is
//! an attention head's value/output product `O_h V` or a query/key product. It lives on the
//! manifold of rank-`r` matrices. Choose `r` pivot rows `I` and columns `J` with an
//! invertible pivot block; in that order
//!
//! ```text
//! Z = [[A, B], [C, D]],   D = C A⁻¹ B.
//! ```
//!
//! `(A, B, C)` are the chart's free coordinates; the dependent block `D` is computed from
//! them, never fitted. An explanation that edits the operator edits `(A, B, C)`, and the
//! dependent block follows.
//!
//! # Pivots
//!
//! `I` is the row order Gaussian elimination with partial pivoting picks on `W₁` and `J`
//! the one it picks on `W₂ᵀ`, so `P = W₁[I]` and `Q = W₂[:, J]` are nonsingular when the
//! factors have rank `r`; `A = P Q`. Both ranks are read above the SVD band, and `A` must
//! have its smallest singular value resolved above its band; a chart is refused otherwise.
//!
//! # Lift to the native factors
//!
//! The factorization `Z = [I; C A⁻¹] [A, B]` (rows `I` then the rest, columns `J` then the
//! rest) is a `GL(r)` pass-through ([`LinearPassthrough`]) of the same product as
//! `(W₂, W₁)`, and the gauge element `S = P⁻¹` maps it to the native factors exactly:
//! `P⁻¹ [A, B] = W₂` and `[I; C A⁻¹] P = W₁` (as `C A⁻¹ P = W₁[Iᶜ] Q (P Q)⁻¹ P = W₁[Iᶜ]`).
//! An edited chart `(A′, B′, C′)` lifts through the same element:
//!
//! ```text
//! W₁′ = [I; C′ A′⁻¹] P,   W₂′ = P⁻¹ [A′, B′],
//! ```
//!
//! which keeps the native pivot rows `W₁[I] = P` exactly and so moves the native tensors
//! only as far as the chart edit demands. The native edits are `ΔW₁ = W₁′ − W₁` (zero on the
//! pivot rows) and `ΔW₂ = W₂′ − W₂`, each of rank at most `r`, in each tensor's stored
//! orientation. An unedited chart compiles to the native plan, `ρ(0) = θ`.
//!
//! # Certificate
//!
//! The lifted product is compared with the chart's target `[I; C′A′⁻¹][A′, B′]` by
//! [`LinearPassthrough::operator_difference`], an algebraic supremum over every input. The
//! target's `C` block is `C′ A′⁻¹ A′`, which differs from `C′` by the rounding of the
//! solve; that defect is evaluated and added. The control is
//! [`ControlRealization::ExactlyRealized`]: the lift is an identity over all inputs.
//!
//! # The dependent block's band
//!
//! `D̂ = fl(C X̂)` with `X̂` the solve of `A X = B`. With the evaluated residual
//! `R = A X̂ − B`, `X̂ − A⁻¹B = A⁻¹R`, so `‖X̂ − X‖_F ≤ ‖R‖_F / σ_min(A)`, with `σ_min(A)`
//! taken below its SVD band. The coordinates `A, B, C` are themselves rounded products of
//! the stored factors (`γ_r |P||Q|` entrywise); their first-order effect on `C A⁻¹ B` is
//! added.

use gam_linalg::roundoff::accumulation_growth;
use gam_math::roundoff::inflated;
use ndarray::{Array2, ArrayView2, Axis, concatenate};

use super::super::apply::FactoredEdit;
use super::super::dense::{solve, svd};
use super::super::gauge::{AllInputs, LinearPassthrough};
use super::super::lift::{TensorId, TensorRegistry};
use super::super::supports::{EvidenceStatus, ExactBasis};
use super::linear::frobenius;
use super::{
    CompileError, CompiledControl, CompiledParameterEdit, ControlRealization, NativeEditPlan, require_finite,
    require_shape,
};

/// A fixed-rank chart of `Z = W₁ W₂` at pivots `I`, `J`.
#[derive(Clone, Debug)]
pub struct FixedRankChart {
    write: Array2<f64>,
    read: Array2<f64>,
    pivot_rows: Vec<usize>,
    free_rows: Vec<usize>,
    pivot_cols: Vec<usize>,
    free_cols: Vec<usize>,
    a: Array2<f64>,
    b: Array2<f64>,
    c: Array2<f64>,
    /// Entrywise formation bands of `A`, `B`, `C` against the exact products.
    bands: [Array2<f64>; 3],
}

/// The dependent block and a bound on its Frobenius distance from the exact `C A⁻¹ B`.
#[derive(Clone, Debug, PartialEq)]
pub struct DependentBlock {
    pub values: Array2<f64>,
    pub frobenius_band: f64,
}

/// A stored native factor: its storage and whether it is stored transposed.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FactorBinding {
    pub storage: TensorId,
    pub stored_transposed: bool,
}

/// New chart coordinates; an absent block is unchanged.
#[derive(Clone, Debug, Default)]
pub struct ChartSetting<'a> {
    pub a: Option<ArrayView2<'a, f64>>,
    pub b: Option<ArrayView2<'a, f64>>,
    pub c: Option<ArrayView2<'a, f64>>,
}

/// What [`compile_chart_edit`] produced.
#[derive(Clone, Debug)]
pub struct ChartEditReport {
    pub compiled: CompiledControl<Vec<f64>, AllInputs>,
    /// The lifted native factors `W₁′` (`m × r`) and `W₂′` (`r × n`).
    pub lifted_write: Array2<f64>,
    pub lifted_read: Array2<f64>,
    /// The new dependent block `C′ A′⁻¹ B′`.
    pub dependent: DependentBlock,
}

/// Row order chosen by Gaussian elimination with partial pivoting on `matrix` (`rows × r`):
/// the `r` pivot rows, then the rest in increasing order.
fn partial_pivot_rows(matrix: ArrayView2<'_, f64>) -> Result<(Vec<usize>, Vec<usize>), CompileError> {
    let (rows, rank) = matrix.dim();
    let mut work = matrix.to_owned();
    let mut order: Vec<usize> = (0..rows).collect();
    for k in 0..rank {
        let pivot = (k..rows).fold(k, |best, row| if work[[row, k]].abs() > work[[best, k]].abs() { row } else { best });
        if work[[pivot, k]] == 0.0 {
            return Err(CompileError::InvalidDeclaration {
                what: "chart factor",
                reason: format!("elimination met a zero pivot at column {k}"),
            });
        }
        if pivot != k {
            for col in 0..rank {
                work.swap([pivot, col], [k, col]);
            }
            order.swap(pivot, k);
        }
        for row in k + 1..rows {
            let factor = work[[row, k]] / work[[k, k]];
            for col in k..rank {
                work[[row, col]] -= factor * work[[k, col]];
            }
        }
    }
    let pivots = order[..rank].to_vec();
    let mut rest = order[rank..].to_vec();
    rest.sort_unstable();
    Ok((pivots, rest))
}

/// Refuses a factor whose rank, read above its SVD band, is below its inner dimension.
fn require_full_rank(what: &'static str, matrix: ArrayView2<'_, f64>, rank: usize) -> Result<(), CompileError> {
    let decomposed = svd(matrix, false)?;
    let resolved = decomposed.singular_values.iter().filter(|&&value| value > decomposed.band).count();
    if resolved < rank {
        return Err(CompileError::InvalidDeclaration {
            what,
            reason: format!("resolved rank {resolved} is below the chart rank {rank}"),
        });
    }
    Ok(())
}

/// `σ_min` of a square block taken below its band, refused when not resolved from zero.
fn smallest_resolved(what: &'static str, block: ArrayView2<'_, f64>) -> Result<f64, CompileError> {
    let decomposed = svd(block, false)?;
    let smallest = decomposed.singular_values.iter().copied().fold(f64::INFINITY, f64::min);
    let lower = smallest - decomposed.band;
    if !(lower > 0.0) {
        return Err(CompileError::InvalidDeclaration {
            what,
            reason: format!("smallest singular value {smallest} is not resolved above its band {}", decomposed.band),
        });
    }
    Ok(lower.next_down())
}

fn product_band(left: ArrayView2<'_, f64>, right: ArrayView2<'_, f64>) -> Array2<f64> {
    left.mapv(f64::abs).dot(&right.mapv(f64::abs)) * accumulation_growth(left.ncols())
}

impl FixedRankChart {
    /// The chart of `write · read` (`W₁` `m × r`, `W₂` `r × n`).
    pub fn from_factors(write: ArrayView2<'_, f64>, read: ArrayView2<'_, f64>) -> Result<Self, CompileError> {
        let rank = write.ncols();
        require_shape("chart read factor", (rank, read.ncols()), read.dim())?;
        require_finite("chart write factor", write.iter().copied())?;
        require_finite("chart read factor", read.iter().copied())?;
        if rank == 0 || write.nrows() < rank || read.ncols() < rank {
            return Err(CompileError::InvalidDeclaration {
                what: "chart shape",
                reason: format!("a rank-{rank} chart needs at least {rank} rows and columns"),
            });
        }
        require_full_rank("chart write factor", write, rank)?;
        require_full_rank("chart read factor", read, rank)?;
        let (pivot_rows, free_rows) = partial_pivot_rows(write)?;
        let (pivot_cols, free_cols) = partial_pivot_rows(read.t())?;
        let p = write.select(Axis(0), &pivot_rows);
        let q = read.select(Axis(1), &pivot_cols);
        let read_free = read.select(Axis(1), &free_cols);
        let write_free = write.select(Axis(0), &free_rows);
        let a = p.dot(&q);
        smallest_resolved("chart pivot block", a.view())?;
        let b = p.dot(&read_free);
        let c = write_free.dot(&q);
        let bands = [
            product_band(p.view(), q.view()),
            product_band(p.view(), read_free.view()),
            product_band(write_free.view(), q.view()),
        ];
        Ok(Self {
            write: write.to_owned(),
            read: read.to_owned(),
            pivot_rows,
            free_rows,
            pivot_cols,
            free_cols,
            a,
            b,
            c,
            bands,
        })
    }

    pub fn rank(&self) -> usize {
        self.a.nrows()
    }

    pub fn pivot_rows(&self) -> &[usize] {
        &self.pivot_rows
    }

    pub fn free_rows(&self) -> &[usize] {
        &self.free_rows
    }

    pub fn pivot_cols(&self) -> &[usize] {
        &self.pivot_cols
    }

    pub fn free_cols(&self) -> &[usize] {
        &self.free_cols
    }

    /// The free coordinates `(A, B, C)`.
    pub fn coordinates(&self) -> (ArrayView2<'_, f64>, ArrayView2<'_, f64>, ArrayView2<'_, f64>) {
        (self.a.view(), self.b.view(), self.c.view())
    }

    /// `D = C A⁻¹ B`, computed from the coordinates.
    pub fn dependent(&self) -> Result<DependentBlock, CompileError> {
        dependent_block(self.a.view(), self.b.view(), self.c.view(), &self.bands)
    }

    /// `W₁[Iᶜ] W₂[:, Jᶜ]` formed directly from the native factors, and its Frobenius band.
    pub fn native_dependent(&self) -> (Array2<f64>, f64) {
        let write_free = self.write.select(Axis(0), &self.free_rows);
        let read_free = self.read.select(Axis(1), &self.free_cols);
        let values = write_free.dot(&read_free);
        let band = frobenius(product_band(write_free.view(), read_free.view()).view());
        (values, band)
    }
}

/// `C A⁻¹ B` with its Frobenius band; `formation` holds entrywise bands of `A`, `B`, `C`
/// against the operator they chart.
fn dependent_block(
    a: ArrayView2<'_, f64>,
    b: ArrayView2<'_, f64>,
    c: ArrayView2<'_, f64>,
    formation: &[Array2<f64>; 3],
) -> Result<DependentBlock, CompileError> {
    let rank = a.nrows();
    let sigma = smallest_resolved("chart pivot block", a)?;
    let solved = solve(a, b)?;
    let residual = a.dot(&solved) - b;
    let residual_band = frobenius(product_band(a, solved.view()).view());
    let residual_upper = inflated(frobenius(residual.view()) + residual_band, 2);
    let solve_error = inflated(residual_upper / sigma, 1);
    let values = c.dot(&solved);
    let c_norm = frobenius(c);
    let solved_norm = frobenius(solved.view());
    let product = frobenius(product_band(c, solved.view()).view());
    // First order in the coordinates' own formation: δC X + C A⁻¹ (δB − δA X).
    let (delta_a, delta_b, delta_c) = (frobenius(formation[0].view()), frobenius(formation[1].view()), frobenius(formation[2].view()));
    let coordinates = delta_c * solved_norm + c_norm / sigma * (delta_b + delta_a * solved_norm);
    let frobenius_band = inflated(c_norm * solve_error + product + coordinates, rank + 4);
    Ok(DependentBlock { values, frobenius_band })
}

/// `ΔW` held as `left rightᵀ` in the stored orientation, for a dense change `delta` of a
/// factor (`rows × cols` in the factor's own orientation).
fn dense_edit(delta: Array2<f64>, stored_transposed: bool) -> Result<FactoredEdit, CompileError> {
    let (rows, cols) = delta.dim();
    let edit = if rows <= cols {
        // delta = I · deltaᵀᵀ: left the identity on the short side.
        let identity = Array2::<f64>::eye(rows);
        if stored_transposed {
            FactoredEdit::new(delta.reversed_axes(), identity)?
        } else {
            FactoredEdit::new(identity, delta.reversed_axes())?
        }
    } else {
        let identity = Array2::<f64>::eye(cols);
        if stored_transposed {
            FactoredEdit::new(identity, delta)?
        } else {
            FactoredEdit::new(delta, identity)?
        }
    };
    Ok(edit)
}

fn place(rows: &[usize], blocks: &[&Array2<f64>], total: usize, width: usize) -> Array2<f64> {
    let mut out = Array2::<f64>::zeros((total, width));
    let mut position = 0;
    for block in blocks {
        for row in block.rows() {
            out.row_mut(rows[position]).assign(&row);
            position += 1;
        }
    }
    out
}

/// Compiles new chart coordinates into edits of the native factors.
pub fn compile_chart_edit(
    registry: &TensorRegistry,
    chart: &FixedRankChart,
    setting: &ChartSetting<'_>,
    write_binding: &FactorBinding,
    read_binding: &FactorBinding,
    control: &str,
) -> Result<ChartEditReport, CompileError> {
    let rank = chart.rank();
    let (m, n) = (chart.write.nrows(), chart.read.ncols());
    for (what, given, current) in [
        ("chart A", setting.a, &chart.a),
        ("chart B", setting.b, &chart.b),
        ("chart C", setting.c, &chart.c),
    ] {
        if let Some(given) = given {
            require_shape(what, current.dim(), given.dim())?;
            require_finite(what, given.iter().copied())?;
        }
    }
    let a = setting.a.map_or_else(|| chart.a.clone(), |value| value.to_owned());
    let b = setting.b.map_or_else(|| chart.b.clone(), |value| value.to_owned());
    let c = setting.c.map_or_else(|| chart.c.clone(), |value| value.to_owned());
    let exact_coordinates = [
        Array2::<f64>::zeros(a.dim()),
        Array2::<f64>::zeros(b.dim()),
        Array2::<f64>::zeros(c.dim()),
    ];
    let unchanged = a == chart.a && b == chart.b && c == chart.c;
    let formation = if setting.a.is_none() && setting.b.is_none() && setting.c.is_none() {
        chart.bands.clone()
    } else {
        // Declared coordinates are exact data; unchanged blocks keep their formation band.
        [
            if setting.a.is_some() { exact_coordinates[0].clone() } else { chart.bands[0].clone() },
            if setting.b.is_some() { exact_coordinates[1].clone() } else { chart.bands[1].clone() },
            if setting.c.is_some() { exact_coordinates[2].clone() } else { chart.bands[2].clone() },
        ]
    };
    let dependent = dependent_block(a.view(), b.view(), c.view(), &formation)?;
    let p = chart.write.select(Axis(0), &chart.pivot_rows);
    // [I; C′A′⁻¹] in pivot-then-free row order, and [A′, B′] in pivot-then-free columns.
    let c_over_a = solve(a.t(), c.t())?.reversed_axes();
    let identity = Array2::<f64>::eye(rank);
    let target_write_ordered = concatenate(Axis(0), &[identity.view(), c_over_a.view()]).map_err(|error| {
        CompileError::InvalidDeclaration {
            what: "chart write factor",
            reason: error.to_string(),
        }
    })?;
    let row_order: Vec<usize> = chart.pivot_rows.iter().chain(&chart.free_rows).copied().collect();
    let col_order: Vec<usize> = chart.pivot_cols.iter().chain(&chart.free_cols).copied().collect();
    let target_write = place(&row_order, &[&target_write_ordered], m, rank);
    let target_read = place(&col_order, &[&a.t().to_owned(), &b.t().to_owned()], n, rank).reversed_axes();
    let target = LinearPassthrough::new(target_read.clone(), target_write.clone())?;
    // The gauge element S = P⁻¹: W₂′ = P⁻¹ [A′, B′], W₁′ = [I; C′A′⁻¹] P.
    let lifted_read = solve(p.view(), target_read.view())?;
    let lifted_write = target_write.dot(&p);
    let lifted = LinearPassthrough::new(lifted_read.clone(), lifted_write.clone())?;
    let difference = lifted.operator_difference(&target)?;
    // The target's C block is C′A′⁻¹A′; its distance from C′ bounds what the target misses.
    let recovered_c = c_over_a.dot(&a);
    let c_defect = frobenius((&recovered_c - &c).view())
        + frobenius(product_band(c_over_a.view(), a.view()).view());
    let (value, numerical_error, witness) = match difference {
        EvidenceStatus::Exact {
            value,
            numerical_error,
            witness,
            ..
        } => (value, numerical_error, witness),
        _ => {
            return Err(CompileError::InvalidDeclaration {
                what: "chart certificate",
                reason: "the pass-through difference is not an exact status".to_string(),
            });
        }
    };
    let residual = EvidenceStatus::exact(
        value,
        inflated(numerical_error + c_defect, 1),
        ExactBasis::Algebraic,
        witness.map(|vector| vector.to_vec()),
        AllInputs { width: n },
    )?;
    let plan = if unchanged {
        NativeEditPlan::native()
    } else {
        let write_delta = &lifted_write - &chart.write;
        let read_delta = &lifted_read - &chart.read;
        let mut edits = Vec::new();
        if write_delta.iter().any(|value| *value != 0.0) {
            edits.push(CompiledParameterEdit {
                storage: write_binding.storage.clone(),
                delta: dense_edit(write_delta, write_binding.stored_transposed)?,
            });
        }
        if read_delta.iter().any(|value| *value != 0.0) {
            edits.push(CompiledParameterEdit {
                storage: read_binding.storage.clone(),
                delta: dense_edit(read_delta, read_binding.stored_transposed)?,
            });
        }
        NativeEditPlan::new(registry, edits)?
    };
    Ok(ChartEditReport {
        compiled: CompiledControl::new(
            control.to_string(),
            Some(plan),
            ControlRealization::exactly_realized(residual)?,
        )?,
        lifted_write,
        lifted_read,
        dependent,
    })
}
