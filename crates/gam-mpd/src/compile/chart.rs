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
//! # Factors wider than the chart
//!
//! A pair whose inner dimension `h` exceeds the product's rank `r` (a head whose value/output
//! product is numerically low rank) is first reduced to the chart rank through the `GL(h)`
//! gauge. With the thin QRs `W₁ = Q₁R₁`, `W₂ᵀ = Q₂R₂` and the core SVD
//! `R₁R₂ᵀ = U S Vᵀ`, `r` counts the core's singular values above its band (the SVD band plus
//! both QR backward bands carried through the other factor). When `R₁` is invertible,
//!
//! ```text
//! T = R₁⁻¹ [U_r S_r^{1/2}, U_t],   T⁻¹ = [S_r^{-1/2} U_rᵀ; U_tᵀ] R₁,
//! W₁ T = Q₁ [U_r S_r^{1/2}, U_t],  T⁻¹ W₂ = [S_r^{1/2} V_rᵀ; S_t V_tᵀ] Q₂ᵀ,
//! ```
//!
//! so the gauge pair splits into the rank-`r` part `(L_r, R_r)` the chart is built on and a
//! tail `L_t R_t = Q₁ U_t S_t V_tᵀ Q₂ᵀ` of norm `σ_{r+1}`, below the band. When only `R₂` is
//! invertible the mirrored element `T = R₂ᵀ [V_r S_r^{-1/2}, V_t]` does the same. The chart
//! edits `(L_r, R_r)` and keeps the tail, and the lift maps back:
//! `W₁′ = [L_r′, L_t] T⁻¹`, `W₂′ = T [R_r′; R_t]`. The **reduction band** bounds what the
//! chart does not see: `‖L_t‖_F ‖R_t‖_F` plus `‖T T⁻¹ − I‖_F ‖W₁‖_F ‖W₂‖_F` for the gauge's own
//! rounding; it enters the dependent block's band and the certificate.
//!
//! # The dependent block's band
//!
//! `D̂ = fl(C X̂)` with `X̂` the solve of `A X = B`. With the evaluated residual
//! `R = A X̂ − B`, `X̂ − A⁻¹B = A⁻¹R`, so `‖X̂ − X‖_F ≤ ‖R‖_F / σ_min(A)`, with `σ_min(A)`
//! taken below its SVD band. The coordinates `A, B, C` are themselves rounded products of
//! the stored factors (`γ_r |P||Q|` entrywise); their first-order effect on `C A⁻¹ B` is
//! added.

use gam_linalg::roundoff::{accumulation_growth, householder_qr_backward_band};
use gam_linalg::utils::frobenius_norm;
use gam_math::roundoff::inflated;
use ndarray::{Array2, ArrayView2, Axis, concatenate};

use super::super::apply::FactoredEdit;
use gam_linalg::decompose::{QrMode, qr, solve, svd};
use super::super::gauge::{AllInputs, LinearPassthrough};
use super::super::lift::{TensorId, TensorRegistry};
use super::super::supports::{EvidenceStatus, ExactBasis};
use super::{
    CompileError, CompiledControl, CompiledParameterEdit, ControlRealization, NativeEditPlan, require_finite,
    require_shape,
};

/// The `GL(h)` reduction of a factor pair wider than its chart rank.
#[derive(Clone, Debug)]
struct Reduction {
    gauge: Array2<f64>,
    inverse: Array2<f64>,
    write_tail: Array2<f64>,
    read_tail: Array2<f64>,
}

/// A fixed-rank chart of `Z = W₁ W₂` at pivots `I`, `J`.
#[derive(Clone, Debug)]
pub struct FixedRankChart {
    /// The native factors (`m × h`, `h × n`).
    native_write: Array2<f64>,
    native_read: Array2<f64>,
    /// The rank-`r` write factor the chart is built on: the native one when `h = r`.
    write: Array2<f64>,
    reduction: Option<Reduction>,
    reduction_band: f64,
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
    /// The lifted native factors `W₁′` (`m × h`) and `W₂′` (`h × n`).
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

/// The rank-`r` gauge pair of `(W₁, W₂)`, the reduction, and its band (module docs).
fn reduce(
    write: ArrayView2<'_, f64>,
    read: ArrayView2<'_, f64>,
) -> Result<(Array2<f64>, Array2<f64>, Option<Reduction>, f64), CompileError> {
    let inner = write.ncols();
    let first = qr(write, QrMode::Economic)?;
    let second = qr(read.t(), QrMode::Economic)?;
    let (r1, r2) = (first.r, second.r);
    let core = r1.dot(&r2.t());
    let decomposed = svd(core.view(), false)?;
    let (norm1, norm2) = (frobenius_norm(r1.view()), frobenius_norm(r2.view()));
    let band = inflated(
        decomposed.band
            + householder_qr_backward_band(write.nrows(), inner, norm1) * norm2
            + norm1 * householder_qr_backward_band(read.ncols(), inner, norm2)
            + frobenius_norm(product_band(r1.view(), r2.t()).view()),
        3,
    );
    let rank = decomposed.singular_values.iter().filter(|&&value| value > band).count();
    if rank == 0 {
        return Err(CompileError::InvalidDeclaration {
            what: "chart factors",
            reason: "their product is not resolved from zero".to_string(),
        });
    }
    if rank == inner && r1.nrows() == inner && r2.nrows() == inner {
        return Ok((write.to_owned(), read.to_owned(), None, 0.0));
    }
    let sigma = &decomposed.singular_values;
    let u = &decomposed.u;
    let v = decomposed.vt.t().to_owned();
    let k = sigma.len();
    let square = |r: &Array2<f64>| r.nrows() == inner && smallest_resolved("inner triangle", r.view()).is_ok();
    let scaled = |basis: &Array2<f64>, power: f64| {
        let mut out = basis.clone();
        for (index, mut column) in out.columns_mut().into_iter().enumerate() {
            if index < rank {
                let factor = sigma[index].powf(power);
                column.mapv_inplace(|value| value * factor);
            }
        }
        out
    };
    if k < inner {
        return Err(CompileError::InvalidDeclaration {
            what: "chart factors",
            reason: format!("the inner dimension {inner} exceeds a factor's outer dimension"),
        });
    }
    let (gauge, inverse) = if square(&r1) {
        // T = R₁⁻¹ [U_r S_r^{1/2}, U_t],  T⁻¹ = [S_r^{-1/2} U_rᵀ; U_tᵀ] R₁.
        (solve(r1.view(), scaled(u, 0.5).view())?, scaled(u, -0.5).t().dot(&r1))
    } else if square(&r2) {
        // T = R₂ᵀ [V_r S_r^{-1/2}, V_t],  T⁻¹ = (R₂⁻¹ [V_r S_r^{1/2}, V_t])ᵀ.
        (r2.t().dot(&scaled(&v, -0.5)), solve(r2.view(), scaled(&v, 0.5).view())?.reversed_axes())
    } else {
        return Err(CompileError::InvalidDeclaration {
            what: "chart factors",
            reason: "neither factor resolves its full inner dimension, so no GL(h) element reduces the pair".to_string(),
        });
    };
    let gauge_write = write.dot(&gauge);
    let gauge_read = inverse.dot(&read);
    let (write_rank, write_tail) = (
        gauge_write.slice(ndarray::s![.., ..rank]).to_owned(),
        gauge_write.slice(ndarray::s![.., rank..]).to_owned(),
    );
    let (read_rank, read_tail) = (
        gauge_read.slice(ndarray::s![..rank, ..]).to_owned(),
        gauge_read.slice(ndarray::s![rank.., ..]).to_owned(),
    );
    let mut identity_defect = gauge.dot(&inverse);
    for index in 0..inner {
        identity_defect[[index, index]] -= 1.0;
    }
    let reduction_band = inflated(
        frobenius_norm(write_tail.view()) * frobenius_norm(read_tail.view())
            + (frobenius_norm(identity_defect.view()) + accumulation_growth(inner) * frobenius_norm(gauge.view()) * frobenius_norm(inverse.view()))
                * frobenius_norm(write)
                * frobenius_norm(read),
        4,
    );
    Ok((
        write_rank,
        read_rank,
        Some(Reduction {
            gauge,
            inverse,
            write_tail,
            read_tail,
        }),
        reduction_band,
    ))
}

fn product_band(left: ArrayView2<'_, f64>, right: ArrayView2<'_, f64>) -> Array2<f64> {
    left.mapv(f64::abs).dot(&right.mapv(f64::abs)) * accumulation_growth(left.ncols())
}

impl FixedRankChart {
    /// The chart of `write · read` (`W₁` `m × h`, `W₂` `h × n`), reduced to the product's
    /// resolved rank when `h` exceeds it.
    pub fn from_factors(write: ArrayView2<'_, f64>, read: ArrayView2<'_, f64>) -> Result<Self, CompileError> {
        let inner = write.ncols();
        require_shape("chart read factor", (inner, read.ncols()), read.dim())?;
        require_finite("chart write factor", write.iter().copied())?;
        require_finite("chart read factor", read.iter().copied())?;
        if inner == 0 {
            return Err(CompileError::InvalidDeclaration {
                what: "chart shape",
                reason: "a chart needs a positive inner dimension".to_string(),
            });
        }
        let (reduced_write, reduced_read, reduction, reduction_band) = reduce(write, read)?;
        let rank = reduced_write.ncols();
        if write.nrows() < rank || read.ncols() < rank {
            return Err(CompileError::InvalidDeclaration {
                what: "chart shape",
                reason: format!("a rank-{rank} chart needs at least {rank} rows and columns"),
            });
        }
        require_full_rank("chart write factor", reduced_write.view(), rank)?;
        require_full_rank("chart read factor", reduced_read.view(), rank)?;
        let (pivot_rows, free_rows) = partial_pivot_rows(reduced_write.view())?;
        let (pivot_cols, free_cols) = partial_pivot_rows(reduced_read.t())?;
        let p = reduced_write.select(Axis(0), &pivot_rows);
        let q = reduced_read.select(Axis(1), &pivot_cols);
        let read_free = reduced_read.select(Axis(1), &free_cols);
        let write_free = reduced_write.select(Axis(0), &free_rows);
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
            native_write: write.to_owned(),
            native_read: read.to_owned(),
            write: reduced_write,
            reduction,
            reduction_band,
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

    /// A bound on `‖W₁W₂ − L_r R_r‖₂`: what the rank-`r` chart does not see (zero when the
    /// factors' inner dimension is the chart rank).
    pub fn reduction_band(&self) -> f64 {
        self.reduction_band
    }

    /// The native inner dimension `h`.
    pub fn inner_dimension(&self) -> usize {
        self.native_write.ncols()
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
        let mut block = dependent_block(self.a.view(), self.b.view(), self.c.view(), &self.bands)?;
        block.frobenius_band = inflated(block.frobenius_band + self.reduction_band, 1);
        Ok(block)
    }

    /// `W₁[Iᶜ] W₂[:, Jᶜ]` formed directly from the native factors, and its Frobenius band.
    pub fn native_dependent(&self) -> (Array2<f64>, f64) {
        let write_free = self.native_write.select(Axis(0), &self.free_rows);
        let read_free = self.native_read.select(Axis(1), &self.free_cols);
        let values = write_free.dot(&read_free);
        let band = frobenius_norm(product_band(write_free.view(), read_free.view()).view());
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
    let residual_band = frobenius_norm(product_band(a, solved.view()).view());
    let residual_upper = inflated(frobenius_norm(residual.view()) + residual_band, 2);
    let solve_error = inflated(residual_upper / sigma, 1);
    let values = c.dot(&solved);
    let c_norm = frobenius_norm(c);
    let solved_norm = frobenius_norm(solved.view());
    let product = frobenius_norm(product_band(c, solved.view()).view());
    // First order in the coordinates' own formation: δC X + C A⁻¹ (δB − δA X).
    let (delta_a, delta_b, delta_c) = (frobenius_norm(formation[0].view()), frobenius_norm(formation[1].view()), frobenius_norm(formation[2].view()));
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
    let (m, n) = (chart.native_write.nrows(), chart.native_read.ncols());
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
    let mut dependent = dependent;
    dependent.frobenius_band = inflated(dependent.frobenius_band + chart.reduction_band, 1);
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
    let reduced_read = solve(p.view(), target_read.view())?;
    let reduced_write = target_write.dot(&p);
    // Back through the reduction's gauge element: W₁′ = [L_r′, L_t] T⁻¹, W₂′ = T [R_r′; R_t].
    let (lifted_write, lifted_read) = match &chart.reduction {
        None => (reduced_write, reduced_read),
        Some(reduction) => {
            let join = |what: &'static str, error: ndarray::ShapeError| CompileError::InvalidDeclaration {
                what,
                reason: error.to_string(),
            };
            let write = concatenate(Axis(1), &[reduced_write.view(), reduction.write_tail.view()])
                .map_err(|error| join("lifted write", error))?;
            let read = concatenate(Axis(0), &[reduced_read.view(), reduction.read_tail.view()])
                .map_err(|error| join("lifted read", error))?;
            (write.dot(&reduction.inverse), reduction.gauge.dot(&read))
        }
    };
    let lifted = LinearPassthrough::new(lifted_read.clone(), lifted_write.clone())?;
    let difference = lifted.operator_difference(&target)?;
    // The target's C block is C′A′⁻¹A′; its distance from C′ bounds what the target misses.
    let recovered_c = c_over_a.dot(&a);
    let c_defect = frobenius_norm((&recovered_c - &c).view())
        + frobenius_norm(product_band(c_over_a.view(), a.view()).view());
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
        inflated(numerical_error + c_defect + chart.reduction_band, 1),
        ExactBasis::Algebraic,
        witness.map(|vector| vector.to_vec()),
        AllInputs { width: n },
    )?;
    let plan = if unchanged {
        NativeEditPlan::native()
    } else {
        let write_delta = &lifted_write - &chart.native_write;
        let read_delta = &lifted_read - &chart.native_read;
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
