//! Frame starts for library_vpd (#2951): `M`'s maps cut exactly into components through an
//! overcomplete frame of each read space, a start that needs no VPD (Qwen3 has none) and that can
//! hold more components than a map has rank.
//!
//! A frame of a read space of width `d` is `C = 4d` atoms `f_i` (the rows of `F`, `C × d`) that
//! span it; its canonical dual is `g_i = (FᵀF)⁻¹ f_i`, so `Σ_i f_i g_iᵀ = FᵀF (FᵀF)⁻¹ = I` and every
//! map `W` reading the space is exactly `Σ_i (W f_i) g_iᵀ`: slice `i` reads `g_iᵀ x`, `x`'s
//! coefficient on atom `i`, and writes `W f_i`. All components on is `M`, whatever the frame.
//!
//! The toy gate (`bench/toys_2951`) found that neuron and head starts cannot recover features held
//! in superposition (TMS: 10 hidden units, 40 features; the fitter cannot split a component), so a
//! map whose read is not followed by an elementwise nonlinearity on the sliced axis (attention's q,
//! k and v reading the stream, o reading the heads' outputs, an MLP with the identity law) is cut
//! through a frame, while an MLP whose law is elementwise nonlinear keeps its neuron groups (the
//! law privileges the neuron axis): per neuron its c_fc row and down_proj column.
//!
//! Three frames (`FrameKind`):
//! * `Tight`: `C` standard normal atoms made a Parseval frame, `F ← F (FᵀF)^{-1/2}`, so its dual
//!   is itself; no data.
//! * `Dictionary`: a one-sparse dictionary of `M`'s activations at the read on fitting rows
//!   (spherical k-means: each row is assigned the atom of largest `|f·x|` among unit atoms, each
//!   atom becomes the leading eigenvector of its rows' second moment; started at the tight frame,
//!   repeated until no assignment changes). The atoms are directions the activations take one at a
//!   time; no ground truth is used. Directions the activations never take are completed by the
//!   unresolved eigenvectors of the atoms' Gram, so the frame spans its space.
//!
//! * `Sparse`: a k-sparse dictionary (orthogonal matching pursuit, each row's atom count the one of
//!   shortest two-part code, so `k` is the data's own sparsity; atoms by the method of optimal
//!   directions) with the dual of least ℓ1 norm of its coefficients on the fitting rows among all
//!   duals, so the reads are sparse on the data and every map is still cut exactly.
//!
//! Each component is one atom's slices on the maps reading its space, with one own gate at that
//! read, started at `τ = 0`: on wherever it reads anything, which is everywhere its output is
//! nonzero, so the start is exact and no gate starts saturated. The direction arm gates the same
//! components by `gᵀx − τ` with `g` the component's own read direction (its first read slice, the
//! sign that makes its mean coefficient on the fitting rows positive), `c = 0` and `τ = 0`: on
//! where the coefficient is positive, so it is exact only where the coefficients have one sign.
//! Every gate's width is its read's spread on the fitting rows (library_vpd's learned width, as
//! vpd_start sets it); a direction is written in those units, width 1.

use crate::{
    artifact::Artifact,
    explanation_battery::KINDS,
    operator_program::{FamilyInputs, Law, Node, OperatorProgram},
    run_check::LayerNodes,
};
use gam_linalg::{decompose::eigh, faer_ndarray::fast_ata, roundoff::SymmetricAssembly};
use ndarray::{Array2, Axis, concatenate};
use rand::{RngExt, SeedableRng, rngs::StdRng};
use serde_json::{Value, json};
use std::path::Path;

fn error(e: impl std::fmt::Display) -> String {
    format!("library frame: {e}")
}

/// How a read space's frame is made.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FrameKind {
    Tight,
    Dictionary,
    /// A k-sparse dictionary with its sparse dual ([`Frame::sparse_dictionary`],
    /// [`Frame::sparse_dual`]).
    Sparse,
}

/// A frame of a read space: its atoms as rows, `C × d`.
#[derive(Clone, Debug)]
pub struct Frame {
    pub atoms: Array2<f64>,
}

/// The frame's atoms per dimension of the space.
pub const REDUNDANCY: usize = 4;

impl Frame {
    /// `count` standard normal atoms in `dim` dimensions made Parseval, `FᵀF = I`.
    pub fn random_tight(count: usize, dim: usize, seed: u64) -> Result<Self, String> {
        if count < dim || dim == 0 {
            return Err(error(format!("{count} atoms cannot span {dim} dimensions")));
        }
        let mut rng = StdRng::seed_from_u64(seed);
        let draws = Array2::from_shape_fn((count, dim), |_| {
            let (u, angle) = (1.0 - rng.random::<f64>(), std::f64::consts::TAU * rng.random::<f64>());
            (-2.0 * u.ln()).sqrt() * angle.cos()
        });
        let gram = fast_ata(&draws);
        let root = eigh(gram.view(), SymmetricAssembly::Mirrored, None)
            .map_err(|e| error(format!("{e:?}")))?
            .psd_map(0.0, |v| v.powf(-0.5))
            .map_err(|e| error(format!("{e:?}")))?;
        Ok(Self { atoms: draws.dot(&root) })
    }

    /// A one-sparse dictionary of `rows` (one per row, `d` columns), started at `start`: spherical
    /// k-means of the rows with nonzero norm (module note). Atoms are unit vectors.
    pub fn dictionary(rows: &Array2<f64>, start: Frame) -> Result<Self, String> {
        let dim = start.atoms.ncols();
        if rows.ncols() != dim {
            return Err(error(format!("rows of {} columns for a frame of {dim}", rows.ncols())));
        }
        let live: Vec<usize> = (0..rows.nrows()).filter(|&r| rows.row(r).iter().any(|v| *v != 0.0)).collect();
        let x = rows.select(Axis(0), &live);
        let mut atoms = start.atoms;
        for mut atom in atoms.rows_mut() {
            let norm = atom.dot(&atom).sqrt();
            atom.mapv_inplace(|v| v / norm);
        }
        let mut assigned = vec![usize::MAX; x.nrows()];
        // Assignments only ever change to a strictly better atom, and the atoms' fit only improves,
        // so the loop ends; the pass bound guards a tie cycle.
        for _ in 0..1000 {
            let scores = x.dot(&atoms.t());
            let next: Vec<usize> = scores
                .rows()
                .into_iter()
                .map(|r| (0..r.len()).max_by(|&a, &b| r[a].abs().total_cmp(&r[b].abs())).unwrap_or(0))
                .collect();
            if next == assigned {
                break;
            }
            assigned = next;
            for (i, mut atom) in atoms.rows_mut().into_iter().enumerate() {
                let mine: Vec<usize> = (0..x.nrows()).filter(|&r| assigned[r] == i).collect();
                if mine.is_empty() {
                    continue;
                }
                let moment = fast_ata(&x.select(Axis(0), &mine));
                let top = eigh(moment.view(), SymmetricAssembly::Mirrored, Some((dim - 1, dim))).map_err(|e| error(format!("{e:?}")))?;
                atom.assign(&top.vectors.column(0));
            }
        }
        // Directions the rows never take (a toy's stream slots that hold zeros at this read) have
        // no atom; they are completed by the unresolved eigenvectors of the atoms' Gram, so the
        // frame spans the space and every map is still cut exactly (their components read nothing
        // on the fitting rows).
        Self::dictionary_completed(atoms)
    }

    /// The canonical dual, `C × d`: rows `g_i = (FᵀF)⁻¹ f_i`. Refused for atoms that do not span.
    pub fn dual(&self) -> Result<Array2<f64>, String> {
        let gram = fast_ata(&self.atoms);
        let decomposition = eigh(gram.view(), SymmetricAssembly::Mirrored, None).map_err(|e| error(format!("{e:?}")))?;
        let largest = decomposition.values.iter().fold(0.0_f64, |m, v| m.max(*v));
        if decomposition.values.iter().any(|v| *v <= decomposition.band.max(largest * f64::EPSILON * gram.nrows() as f64)) {
            return Err(error("the atoms do not span their space"));
        }
        let inverse = decomposition.psd_map(0.0, |v| 1.0 / v).map_err(|e| error(format!("{e:?}")))?;
        Ok(self.atoms.dot(&inverse))
    }

    /// The frame of `kind` for a space of `rows` (`M`'s activations there, one per row).
    pub fn of(kind: FrameKind, rows: &Array2<f64>, seed: u64) -> Result<Self, String> {
        let dim = rows.ncols();
        match kind {
            FrameKind::Tight => Self::random_tight(REDUNDANCY * dim, dim, seed),
            FrameKind::Dictionary => Self::dictionary(rows, Self::random_tight(REDUNDANCY * dim, dim, seed)?),
            FrameKind::Sparse => Ok(Self::sparse_dictionary(rows, REDUNDANCY * dim, seed)?.0),
        }
    }

    /// A k-sparse dictionary of `count` atoms for `rows` (one per row) and the rows' mean atom count.
    /// The atoms start as `count` distinct nonzero rows (a seeded draw; where the rows are sparse in
    /// some basis, many of them are single basis vectors, which a random start never finds). Each row
    /// is coded by orthogonal matching pursuit over the unit atoms, its atom count the one of
    /// shortest two-part code: per atom its index and coefficient, `log₂ C + ½ log₂ N` bits (`N` the
    /// rows), and the residual at its own mean square per dimension, `(d/2) log₂(‖r‖²/d)`, down to
    /// the rounding of the row (`‖r‖² ≥ d (ε‖x‖)²`), so the count is the data's own sparsity,
    /// measured, not set. The atoms then take the least squares fit to the codes (the method of
    /// optimal directions), kept only while the rows' total code length falls. Directions the rows
    /// never take are completed as in [`Frame::dictionary`].
    pub fn sparse_dictionary(rows: &Array2<f64>, count: usize, seed: u64) -> Result<(Self, f64), String> {
        let dim = rows.ncols();
        let live: Vec<usize> = (0..rows.nrows()).filter(|&r| rows.row(r).iter().any(|v| *v != 0.0)).collect();
        let x = rows.select(Axis(0), &live);
        let unit = |atoms: &mut Array2<f64>| {
            for mut atom in atoms.rows_mut() {
                let norm = atom.dot(&atom).sqrt();
                if norm > 0.0 {
                    atom.mapv_inplace(|v| v / norm);
                }
            }
        };
        // `count` distinct rows by a seeded shuffle (fewer when the rows hold fewer).
        let mut rng = StdRng::seed_from_u64(seed);
        let mut order: Vec<usize> = (0..x.nrows()).collect();
        for i in (1..order.len()).rev() {
            order.swap(i, rng.random_range(0..=i));
        }
        let mut picked: Vec<usize> = Vec::with_capacity(count);
        for &r in &order {
            if picked.len() == count {
                break;
            }
            if !picked.iter().any(|&q| x.row(q) == x.row(r)) {
                picked.push(r);
            }
        }
        if picked.is_empty() {
            return Ok((Self::random_tight(count.max(dim), dim, seed)?, 0.0));
        }
        let mut atoms = x.select(Axis(0), &picked);
        unit(&mut atoms);
        let count = atoms.nrows();
        let bits_per_atom = (count as f64).log2() + 0.5 * (x.nrows().max(1) as f64).log2();
        let code = |atoms: &Array2<f64>| -> Result<(Vec<(Vec<usize>, Vec<f64>)>, f64), String> {
            let codes: Vec<(Vec<usize>, Vec<f64>, f64)> = (0..x.nrows()).map(|r| pursuit(x.row(r), atoms, bits_per_atom)).collect::<Result<_, _>>()?;
            let bits = codes.iter().map(|c| c.2).sum();
            Ok((codes.into_iter().map(|(s, a, _)| (s, a)).collect(), bits))
        };
        let (mut codes, mut bits) = code(&atoms)?;
        for _ in 0..100 {
            // The method of optimal directions: atoms = (AᵀA)⁺ Aᵀ X over the codes A.
            let mut ata = Array2::<f64>::zeros((count, count));
            let mut atx = Array2::<f64>::zeros((count, dim));
            for (r, (support, coefficients)) in codes.iter().enumerate() {
                for (a, &i) in support.iter().enumerate() {
                    for (b, &j) in support.iter().enumerate() {
                        ata[[i, j]] += coefficients[a] * coefficients[b];
                    }
                    atx.row_mut(i).scaled_add(coefficients[a], &x.row(r));
                }
            }
            let used: Vec<usize> = (0..count).filter(|&i| ata[[i, i]] > 0.0).collect();
            if used.is_empty() {
                break;
            }
            let sub = ata.select(Axis(0), &used).select(Axis(1), &used);
            let inverse = eigh(sub.view(), SymmetricAssembly::Mirrored, None).map_err(|e| error(format!("{e:?}")))?.psd_map(0.0, |v| 1.0 / v).map_err(|e| error(format!("{e:?}")))?;
            let fitted = inverse.dot(&atx.select(Axis(0), &used));
            let mut next = atoms.clone();
            for (k, &i) in used.iter().enumerate() {
                next.row_mut(i).assign(&fitted.row(k));
            }
            unit(&mut next);
            let (next_codes, next_bits) = code(&next)?;
            if next_bits >= bits {
                break;
            }
            (atoms, codes, bits) = (next, next_codes, next_bits);
        }
        let mean_k = codes.iter().map(|c| c.0.len()).sum::<usize>() as f64 / codes.len().max(1) as f64;
        let frame = Self::dictionary_completed(atoms)?;
        Ok((frame, mean_k))
    }

    /// `atoms` completed by the unresolved eigenvectors of their Gram, so they span their space.
    fn dictionary_completed(mut atoms: Array2<f64>) -> Result<Self, String> {
        let dim = atoms.ncols();
        let gram = fast_ata(&atoms);
        let decomposition = eigh(gram.view(), SymmetricAssembly::Mirrored, None).map_err(|e| error(format!("{e:?}")))?;
        let largest = decomposition.values.iter().fold(0.0_f64, |m, v| m.max(*v));
        let floor = decomposition.band.max(largest * f64::EPSILON * dim as f64);
        let missing: Vec<usize> = (0..dim).filter(|&k| decomposition.values[k] <= floor).collect();
        if !missing.is_empty() {
            let complement = decomposition.vectors.select(Axis(1), &missing).t().to_owned();
            atoms = concatenate(Axis(0), &[atoms.view(), complement.view()]).map_err(error)?;
        }
        let frame = Self { atoms };
        frame.dual()?;
        Ok(frame)
    }

    /// The dual of least ℓ1 norm of its coefficients on `rows`: among the duals `G` (`C × d`,
    /// `FᵀG = I`, so `Σ_i f_i g_iᵀ = I` and every map is still cut exactly), `G = G₀ + P W` with
    /// `G₀` the canonical dual and `P = I − G₀Fᵀ` the projector onto the coefficient vectors no
    /// atom combination synthesizes, the one minimizing `Σ_rows ‖G x‖₁`, by the alternating
    /// direction method of multipliers (its penalty, the canonical coefficients' mean magnitude,
    /// sets the speed, not the solution; on TMS's inputs it reaches a relative residual of 3e-7 in
    /// 3,000 passes with 7 nonzero coefficients a row against the canonical dual's 320), to a
    /// relative residual of 1e-7 or 3,000 passes. Returns the dual and its coefficients' mean
    /// count of nonzeros per row (the ADMM's sparse iterate).
    pub fn sparse_dual(&self, rows: &Array2<f64>) -> Result<(Array2<f64>, f64), String> {
        let g0 = self.dual()?;
        let count = self.atoms.nrows();
        let p = Array2::<f64>::eye(count) - g0.dot(&self.atoms.t());
        let x = rows;
        let m0 = x.dot(&g0.t());
        let xtx_inverse = eigh(fast_ata(x).view(), SymmetricAssembly::Mirrored, None).map_err(|e| error(format!("{e:?}")))?.psd_map(0.0, |v| 1.0 / v).map_err(|e| error(format!("{e:?}")))?;
        let scale = m0.iter().map(|v| v.abs()).sum::<f64>() / m0.len().max(1) as f64;
        let threshold = if scale > 0.0 { scale } else { 1.0 };
        let (mut a, mut u) = (m0.clone(), Array2::<f64>::zeros(m0.dim()));
        let mut y = Array2::<f64>::zeros((x.ncols(), count));
        let norm = m0.iter().map(|v| v * v).sum::<f64>().sqrt().max(f64::MIN_POSITIVE);
        for _ in 0..3000 {
            let b = &a - &m0 - &u;
            y = xtx_inverse.dot(&x.t().dot(&b)).dot(&p);
            let t = &m0 + &x.dot(&y);
            let previous = a.clone();
            a = (&t + &u).mapv(|v| v.signum() * (v.abs() - threshold).max(0.0));
            u = &u + &t - &a;
            let primal = (&t - &a).iter().map(|v| v * v).sum::<f64>().sqrt();
            let change = (&a - &previous).iter().map(|v| v * v).sum::<f64>().sqrt();
            if primal <= 1e-7 * norm && change <= 1e-7 * norm {
                break;
            }
        }
        let dual = &g0 + &y.t();
        let nonzeros = a.iter().filter(|v| **v != 0.0).count() as f64 / a.nrows().max(1) as f64;
        Ok((dual, nonzeros))
    }
}

/// Orthogonal matching pursuit of `x` over the unit `atoms` (rows), its atom count the one of
/// shortest two-part code ([`Frame::sparse_dictionary`]): the support, its coefficients and the
/// code length in bits.
fn pursuit(x: ndarray::ArrayView1<f64>, atoms: &Array2<f64>, bits_per_atom: f64) -> Result<(Vec<usize>, Vec<f64>, f64), String> {
    let dim = x.len();
    let energy = x.dot(&x);
    if energy == 0.0 {
        return Ok((Vec::new(), Vec::new(), 0.0));
    }
    let floor = dim as f64 * (f64::EPSILON * energy.sqrt()).powi(2);
    let code_bits = |k: usize, residual: f64| k as f64 * bits_per_atom + 0.5 * dim as f64 * (residual.max(floor) / dim as f64).log2();
    let (mut support, mut residual) = (Vec::new(), x.to_owned());
    let mut best = (code_bits(0, energy), Vec::new(), Vec::new());
    while support.len() < dim.min(atoms.nrows()) {
        let scores = atoms.dot(&residual);
        let next = (0..scores.len()).filter(|i| !support.contains(i)).max_by(|&a, &b| scores[a].abs().total_cmp(&scores[b].abs()));
        let Some(next) = next else { break };
        support.push(next);
        let chosen = atoms.select(Axis(0), &support);
        let gram = chosen.dot(&chosen.t());
        let rhs = chosen.dot(&x).insert_axis(Axis(1));
        // The least squares coefficients on the support; a pseudo-inverse, since a duplicated atom
        // (atoms drawn from equal-direction rows) makes the support's Gram singular.
        let solved = gam_linalg::decompose::pseudo_inverse_solve(gram.view(), rhs.view()).map_err(|e| error(format!("{e:?}")))?;
        let coefficients = solved.column(0).to_vec();
        residual = &x - &chosen.t().dot(&solved.column(0));
        let left = residual.dot(&residual);
        let bits = code_bits(support.len(), left);
        if bits < best.0 {
            best = (bits, support.clone(), coefficients);
        }
        if left <= floor {
            break;
        }
    }
    Ok((best.1, best.2, best.0))
}

/// The gates a component of the shared arm may move to, its own first (library_vpd's gate sharing;
/// as descent's share arm, d6ad2537cb): a component of the attention's or the MLP's input its
/// stage's, a down component the MLP's input components'.
const CANDIDATES: usize = 8;

/// Per component of a stage (the columns of `magnitudes`, its read's magnitude on each fitting row),
/// its candidate gates for the shared arm: the `CANDIDATES − 1` other components of the stage whose
/// read magnitudes are most correlated with its own over the rows (the components likeliest to fire
/// together), as indices from `first`, the stage's first component in the arm.
fn candidates(magnitudes: &Array2<f64>, first: usize) -> Vec<Vec<usize>> {
    let z = standardized(magnitudes);
    let correlation = fast_ata(&z);
    (0..z.ncols())
        .map(|b| {
            let mut others: Vec<usize> = (0..z.ncols()).filter(|j| *j != b).collect();
            others.sort_by(|i, j| correlation[[b, *j]].total_cmp(&correlation[[b, *i]]).then(i.cmp(j)));
            others.truncate(CANDIDATES - 1);
            others.into_iter().map(|j| first + j).collect()
        })
        .collect()
}

/// Per own-gated down component (the columns of `down`, its read's magnitude on each fitting
/// row), its candidates among the MLP's input components (the columns of `up`, from `first`): the
/// `CANDIDATES − 1` whose read magnitudes are most correlated with its own, whose gates it may
/// follow (library_vpd's cross-stage sharing, a part spanning c_fc and down slices).
fn following(down: &Array2<f64>, up: &Array2<f64>, first: usize) -> Vec<Vec<usize>> {
    let (zd, zu) = (standardized(down), standardized(up));
    let correlation = zd.t().dot(&zu);
    (0..zd.ncols())
        .map(|b| {
            let mut others: Vec<usize> = (0..zu.ncols()).collect();
            others.sort_by(|i, j| correlation[[b, *j]].total_cmp(&correlation[[b, *i]]).then(i.cmp(j)));
            others.truncate(CANDIDATES - 1);
            others.into_iter().map(|j| first + j).collect()
        })
        .collect()
}

/// `magnitudes` with each column centred and scaled to unit variance over the rows (zero where a
/// column is constant).
fn standardized(magnitudes: &Array2<f64>) -> Array2<f64> {
    let n = magnitudes.nrows().max(1) as f64;
    let mut z = magnitudes.clone();
    for mut column in z.columns_mut() {
        let mean = column.sum() / n;
        column.mapv_inplace(|v| v - mean);
        let sd = (column.dot(&column) / n).sqrt();
        column.mapv_inplace(|v| if sd > 0.0 { v / sd } else { 0.0 });
    }
    z
}

/// The frame of `kind` for a space of `rows` and its dual: the sparse dual for a sparse frame, the
/// canonical dual otherwise.
fn framed(kind: FrameKind, rows: &Array2<f64>, seed: u64) -> Result<(Frame, Array2<f64>), String> {
    let frame = Frame::of(kind, rows, seed)?;
    let dual = match kind {
        FrameKind::Sparse => frame.sparse_dual(rows)?.0,
        _ => frame.dual()?,
    };
    Ok((frame, dual))
}

/// One component of a start file, as library_vpd reads it: its read, `τ = 0`, its gate's width
/// and its slices.
fn component(read: Value, width: f64, slices: &[[usize; 2]]) -> Value {
    json!({"read": read, "tau": 0.0, "width": width, "slices": slices})
}

/// The spread (standard deviation) of `values`; a read that is the same on every fitting row
/// (a zero map's, or a completing atom's that the rows never take) has no scale to set, and its
/// gate is given width 1.
fn spread(values: impl Iterator<Item = f64>) -> f64 {
    let (mut n, mut sum, mut squares) = (0.0, 0.0, 0.0);
    for v in values {
        n += 1.0;
        sum += v;
        squares += v * v;
    }
    let variance = if n > 0.0 { (squares / n - (sum / n).powi(2)).max(0.0) } else { 0.0 };
    if variance > 0.0 { variance.sqrt() } else { 1.0 }
}

/// An own gate reading `copies` slices that each read `g` on `rows`: its width, the spread of
/// `‖V_bᵀx‖ = √copies |gᵀx|`.
fn own_width(rows: &Array2<f64>, g: ndarray::ArrayView1<f64>, copies: f64) -> f64 {
    spread(rows.dot(&g).iter().map(|v| copies.sqrt() * v.abs()))
}

/// A direction gate reading `g` on `rows`, signed so that its mean on the rows is not negative and
/// written in units of its spread there (`g` over the spread, width 1), with no constant.
fn direction_read(rows: &Array2<f64>, g: ndarray::ArrayView1<f64>, site: usize) -> Value {
    let values = rows.dot(&g);
    let sign = if values.sum() >= 0.0 { 1.0 } else { -1.0 };
    let scale = spread(values.iter().copied());
    let mut coefficients: Vec<f64> = g.iter().map(|v| sign * v / scale).collect();
    coefficients.push(0.0);
    json!({"direction": {"site": site, "coefficients": coefficients}})
}

/// The matrix of the native operator named `name`.
fn operator(native: &OperatorProgram, name: &str) -> Result<Array2<f64>, String> {
    let found: Vec<&std::sync::Arc<crate::operator_program::Operator>> = native.operators.iter().filter(|op| op.name == name).collect();
    match found[..] {
        [op] => Ok(op.matrix()),
        _ => Err(error(format!("no unique native operator {name}"))),
    }
}

/// A site's slices, written `U` (`C × out`) and read `V` (`in × C`), and per slice its component.
struct Site {
    u: Array2<f64>,
    v: Array2<f64>,
}

impl Site {
    /// `W` cut through the frame with dual `dual`: slice `i` writes `W f_i` and reads `g_i`.
    fn framed(w: &Array2<f64>, frame: &Frame, dual: &Array2<f64>) -> Self {
        Self {
            u: frame.atoms.dot(&w.t()),
            v: dual.t().to_owned(),
        }
    }

    /// A zero map: one zero slice.
    fn zero(rows: usize, cols: usize) -> Self {
        Self {
            u: Array2::zeros((1, rows)),
            v: Array2::zeros((cols, 1)),
        }
    }
}

/// The frame start of the split native program `native` with its `layers`, fitted on `fitting`
/// (rows of `M`'s input): per kind of frame its decomposition in `dir/{tight,dictionary}/` (the
/// layout explanation_battery::load_factors reads: config.sites `h.{l}.{kind}` in `M`'s order,
/// `{site}.U` `C × out`, `{site}.V` `in × C`) and its arms in `dir/{tight,dictionary}/start.json`
/// (`frame_own`, `frame_direction`; module note). Returns a summary.
pub fn frame_start(native: &OperatorProgram, layers: &[LayerNodes], fitting: &FamilyInputs, seed: u64, dir: &Path) -> Result<Value, String> {
    let artifact = Artifact::native(native)?;
    let trace = artifact.execute(fitting)?;
    let value = |node: usize| -> Result<Array2<f64>, String> { Ok(trace.values[artifact.place(node).ok_or_else(|| error(format!("no node {node}")))?].clone()) };
    let mut summary = Vec::new();
    for (kind, name) in [(FrameKind::Tight, "tight"), (FrameKind::Dictionary, "dictionary"), (FrameKind::Sparse, "sparse")] {
        let out = dir.join(name);
        std::fs::create_dir_all(&out).map_err(error)?;
        let (mut sites, mut files) = (Vec::new(), serde_json::Map::new());
        let (mut own, mut direction) = (Vec::new(), Vec::new());
        // The shared arm's candidates: per shareable stage (the attention's input, the MLP's input,
        // and the down atoms following the MLP's input) its components' candidates, by index in `own`.
        let mut shared: Vec<(usize, Vec<usize>)> = Vec::new();
        for (l, layer) in layers.iter().enumerate() {
            let site = |k: usize| KINDS.len() * l + k;
            let heads = layer.reads.len();
            let weight = |part: &str| operator(native, &format!("blocks.{l}.{part}"));
            let stack = |parts: Vec<Array2<f64>>, axis: usize| -> Result<Array2<f64>, String> {
                let views: Vec<_> = parts.iter().map(|p| p.view()).collect();
                concatenate(Axis(axis), &views).map_err(error)
            };
            let wq = stack((0..heads).map(|h| weight(&format!("q{h}"))).collect::<Result<_, _>>()?, 0)?;
            let kv = layer.keys.len();
            let wk = stack((0..kv).map(|g| weight(&format!("k{g}"))).collect::<Result<_, _>>()?, 0)?;
            let wv = stack((0..kv).map(|g| weight(&format!("v{g}"))).collect::<Result<_, _>>()?, 0)?;
            let wo = stack((0..heads).map(|h| weight(&format!("o{h}"))).collect::<Result<_, _>>()?, 1)?;
            let (up, down) = (weight("c_fc")?, weight("down_proj")?);
            let mut layer_sites: Vec<Site> = Vec::with_capacity(KINDS.len());
            // The attention's input: one frame for q, k and v; a component per atom.
            let x = value(layer.normed_stream)?;
            let zero_attention = [&wq, &wk, &wv].iter().all(|w| w.iter().all(|v| *v == 0.0));
            let (attention_frame, attention_dual) = if zero_attention {
                for w in [&wq, &wk, &wv] {
                    layer_sites.push(Site::zero(w.nrows(), w.ncols()));
                }
                own.push(component(json!({"own": [site(0), 0]}), 1.0, &[[site(0), 0], [site(1), 0], [site(2), 0]]));
                direction.push(component(
                    json!({"direction": {"site": site(0), "coefficients": vec![0.0; x.ncols() + 1]}}),
                    1.0,
                    &[[site(0), 0], [site(1), 0], [site(2), 0]],
                ));
                (None, None)
            } else {
                let (frame, dual) = framed(kind, &x, seed ^ (l as u64 * 6 + 1))?;
                for w in [&wq, &wk, &wv] {
                    layer_sites.push(Site::framed(w, &frame, &dual));
                }
                (Some(frame), Some(dual))
            };
            // o reads the heads' outputs, concatenated.
            let heads_out = stack(layer.reads.iter().map(|&r| value(r)).collect::<Result<_, _>>()?, 1)?;
            let zero_o = wo.iter().all(|v| *v == 0.0);
            let o_frame = if zero_o {
                layer_sites.push(Site::zero(wo.nrows(), wo.ncols()));
                None
            } else {
                let (frame, dual) = framed(kind, &heads_out, seed ^ (l as u64 * 6 + 2))?;
                layer_sites.push(Site::framed(&wo, &frame, &dual));
                Some((frame, dual))
            };
            // The MLP: framed under the identity law, neuron groups under a nonlinear one.
            let Node::Pointwise { laws, .. } = &native.nodes[layer.active] else {
                return Err(error(format!("layer {l}: the MLP activation is not one pointwise law")));
            };
            let linear = laws.iter().all(|law| *law == Law::Identity);
            let h2 = value(layer.normed)?;
            let zero_mlp = up.iter().all(|v| *v == 0.0) && down.iter().all(|v| *v == 0.0);
            let mlp = if zero_mlp {
                layer_sites.push(Site::zero(up.nrows(), up.ncols()));
                layer_sites.push(Site::zero(down.nrows(), down.ncols()));
                None
            } else if linear {
                let (up_frame, up_dual) = framed(kind, &h2, seed ^ (l as u64 * 6 + 3))?;
                layer_sites.push(Site::framed(&up, &up_frame, &up_dual));
                let hidden = value(layer.active)?;
                let (down_frame, down_dual) = framed(kind, &hidden, seed ^ (l as u64 * 6 + 4))?;
                layer_sites.push(Site::framed(&down, &down_frame, &down_dual));
                Some((up_frame.atoms.nrows(), down_frame.atoms.nrows(), Some((up_dual, down_dual))))
            } else {
                // Per neuron its c_fc row and down_proj column.
                let m = up.nrows();
                layer_sites.push(Site {
                    u: Array2::eye(m),
                    v: up.t().to_owned(),
                });
                layer_sites.push(Site {
                    u: down.t().to_owned(),
                    v: Array2::eye(m),
                });
                Some((m, m, None))
            };
            // The components, per site group, own and direction; each gate's width is its read's
            // spread on the fitting rows (a direction is written in those units, width 1).
            if let (Some(frame), Some(dual)) = (&attention_frame, &attention_dual) {
                let first = own.len();
                shared.extend(candidates(&x.dot(&dual.t()).mapv(f64::abs), first).into_iter().enumerate().map(|(i, c)| (first + i, c)));
                for i in 0..frame.atoms.nrows() {
                    let slices = [[site(0), i], [site(1), i], [site(2), i]];
                    own.push(component(json!({"own": [site(0), i]}), own_width(&x, dual.row(i), 3.0), &slices));
                    direction.push(component(direction_read(&x, dual.row(i), site(0)), 1.0, &slices));
                }
            }
            if let Some((frame, dual)) = &o_frame {
                for i in 0..frame.atoms.nrows() {
                    own.push(component(json!({"own": [site(3), i]}), own_width(&heads_out, dual.row(i), 1.0), &[[site(3), i]]));
                    direction.push(component(direction_read(&heads_out, dual.row(i), site(3)), 1.0, &[[site(3), i]]));
                }
            }
            // A zero MLP has no component (library_vpd adds zero for it).
            if let Some((ups, downs, Some((up_dual, down_dual)))) = &mlp {
                let hidden = value(layer.active)?;
                let first = own.len();
                let up_magnitudes = h2.dot(&up_dual.t()).mapv(f64::abs);
                shared.extend(candidates(&up_magnitudes, first).into_iter().enumerate().map(|(i, c)| (first + i, c)));
                // The down atoms, after the c_fc atoms: each may follow a c_fc atom's gate.
                let first_down = first + *ups;
                shared.extend(following(&hidden.dot(&down_dual.t()).mapv(f64::abs), &up_magnitudes, first).into_iter().enumerate().map(|(i, c)| (first_down + i, c)));
                for i in 0..*ups {
                    own.push(component(json!({"own": [site(4), i]}), own_width(&h2, up_dual.row(i), 1.0), &[[site(4), i]]));
                    direction.push(component(direction_read(&h2, up_dual.row(i), site(4)), 1.0, &[[site(4), i]]));
                }
                for i in 0..*downs {
                    own.push(component(json!({"own": [site(5), i]}), own_width(&hidden, down_dual.row(i), 1.0), &[[site(5), i]]));
                    direction.push(component(direction_read(&hidden, down_dual.row(i), site(5)), 1.0, &[[site(5), i]]));
                }
            } else if let Some((m, _, None)) = &mlp {
                let first = own.len();
                shared.extend(candidates(&h2.dot(&up.t()).mapv(f64::abs), first).into_iter().enumerate().map(|(i, c)| (first + i, c)));
                for n in 0..*m {
                    let slices = [[site(4), n], [site(5), n]];
                    own.push(component(json!({"own": [site(4), n]}), own_width(&h2, up.row(n), 1.0), &slices));
                    direction.push(component(direction_read(&h2, up.row(n), site(4)), 1.0, &slices));
                }
            }
            for (k, s) in layer_sites.iter().enumerate() {
                let name = format!("h.{l}.{}", crate::library_vpd::EXPORT_NAMES[k]);
                for (suffix, t) in [("U", &s.u), ("V", &s.v)] {
                    let bytes: Vec<u8> = t.iter().flat_map(|v| v.to_le_bytes()).collect();
                    std::fs::write(out.join(format!("{name}.{suffix}.f64")), bytes).map_err(error)?;
                    files.insert(format!("{name}.{suffix}"), json!({"shape": [t.nrows(), t.ncols()]}));
                }
                sites.push(name);
            }
        }
        let record = json!({"config": {"sites": sites, "frame": name, "redundancy": REDUNDANCY, "seed": seed}, "files": files});
        std::fs::write(out.join("export.json"), record.to_string()).map_err(error)?;
        // The shared arm: the own arm with candidates where components can share gates (library_vpd
        // builds those stages shared; the o stage keeps own gates).
        let mut own_shared = own.clone();
        for (b, c) in shared {
            own_shared[b]["candidates"] = json!(c);
        }
        let arms = json!([{"arm": "frame_own", "components": own}, {"arm": "frame_direction", "components": direction}, {"arm": "frame_own_shared", "components": own_shared}]);
        std::fs::write(out.join("start.json"), arms.to_string()).map_err(error)?;
        summary.push(json!({"frame": name, "components": own.len(), "sites": sites.len(), "fitting_rows": fitting.rows}));
    }
    summary.push(unit_start(native, layers, &value, &dir.join("heads_neurons"))?);
    Ok(json!(summary))
}

/// A head's or neuron's slices of one map: `U` rows (writes) and `V` columns (reads), as for a
/// [`Site`], with the unit owning each.
struct Units {
    writes: Vec<Vec<f64>>,
    reads: Vec<Vec<f64>>,
    owner: Vec<usize>,
}

impl Units {
    fn new() -> Self {
        Self {
            writes: Vec::new(),
            reads: Vec::new(),
            owner: Vec::new(),
        }
    }

    /// The block `rows × cols` of `W` that unit `unit` owns (zero elsewhere), cut by its exact SVD
    /// (the resolved singular values).
    fn cut(&mut self, w: &Array2<f64>, rows: std::ops::Range<usize>, cols: std::ops::Range<usize>, unit: usize) -> Result<(), String> {
        let block = w.slice(ndarray::s![rows.clone(), cols.clone()]);
        if block.iter().all(|v| *v == 0.0) {
            return Ok(());
        }
        let svd = gam_linalg::decompose::svd(block, false).map_err(|e| error(format!("{e:?}")))?;
        for (j, sigma) in svd.singular_values.iter().enumerate().filter(|(_, s)| **s > svd.band) {
            let mut write = vec![0.0; w.nrows()];
            let mut read = vec![0.0; w.ncols()];
            for (r, row) in rows.clone().enumerate() {
                write[row] = svd.u[[r, j]] * sigma;
            }
            for (c, col) in cols.clone().enumerate() {
                read[col] = svd.vt[[j, c]];
            }
            self.writes.push(write);
            self.reads.push(read);
            self.owner.push(unit);
        }
        Ok(())
    }

    /// The site: a zero map's one zero slice when no unit holds a slice.
    fn site(&self, rows: usize, cols: usize) -> Site {
        if self.writes.is_empty() {
            return Site::zero(rows, cols);
        }
        let u = Array2::from_shape_fn((self.writes.len(), rows), |(i, r)| self.writes[i][r]);
        let v = Array2::from_shape_fn((cols, self.reads.len()), |(c, i)| self.reads[i][c]);
        Site { u, v }
    }

    fn of(&self, unit: usize) -> Vec<usize> {
        (0..self.owner.len()).filter(|&i| self.owner[i] == unit).collect()
    }
}

/// The per-unit start, the baseline the frame starts are compared to, in `dir`: per head one
/// component (its q, k and v rows and its o columns, each head's block cut by its exact SVD), per
/// MLP neuron one (its c_fc row and down_proj column), arms `per_slice_own` (every slice its own
/// component, an o or down_proj slice gated on its own read), `grouped_own` and
/// `grouped_direction` (the grouped components gated by their first read slice, in units of its
/// spread). Gates start at `τ = 0`, widths the reads' spreads on the fitting rows, as for the
/// frames. Heads must own their keys and values (no grouped-query sharing).
fn unit_start(native: &OperatorProgram, layers: &[LayerNodes], value: &dyn Fn(usize) -> Result<Array2<f64>, String>, dir: &Path) -> Result<Value, String> {
    std::fs::create_dir_all(dir).map_err(error)?;
    let (mut sites, mut files) = (Vec::new(), serde_json::Map::new());
    let (mut per_slice, mut grouped, mut direction) = (Vec::new(), Vec::new(), Vec::new());
    for (l, layer) in layers.iter().enumerate() {
        let site = |k: usize| KINDS.len() * l + k;
        let heads = layer.reads.len();
        if layer.keys.len() != heads {
            return Err(error(format!(
                "layer {l}: {} key heads for {heads} heads (the per-unit start needs a key and value per head)",
                layer.keys.len()
            )));
        }
        let weight = |part: &str| operator(native, &format!("blocks.{l}.{part}"));
        let stack = |parts: Vec<Array2<f64>>, axis: usize| -> Result<Array2<f64>, String> {
            let views: Vec<_> = parts.iter().map(|p| p.view()).collect();
            concatenate(Axis(axis), &views).map_err(error)
        };
        let wq = stack((0..heads).map(|h| weight(&format!("q{h}"))).collect::<Result<_, _>>()?, 0)?;
        let wk = stack((0..heads).map(|h| weight(&format!("k{h}"))).collect::<Result<_, _>>()?, 0)?;
        let wv = stack((0..heads).map(|h| weight(&format!("v{h}"))).collect::<Result<_, _>>()?, 0)?;
        let wo = stack((0..heads).map(|h| weight(&format!("o{h}"))).collect::<Result<_, _>>()?, 1)?;
        let (up, down) = (weight("c_fc")?, weight("down_proj")?);
        let hd = wq.nrows() / heads;
        let mut units: Vec<Units> = (0..KINDS.len()).map(|_| Units::new()).collect();
        for h in 0..heads {
            for (k, w) in [&wq, &wk, &wv].into_iter().enumerate() {
                units[k].cut(w, h * hd..(h + 1) * hd, 0..w.ncols(), h)?;
            }
            units[3].cut(&wo, 0..wo.nrows(), h * hd..(h + 1) * hd, h)?;
        }
        for n in 0..up.nrows() {
            units[4].cut(&up, n..n + 1, 0..up.ncols(), n)?;
            units[5].cut(&down, 0..down.nrows(), n..n + 1, n)?;
        }
        let shapes = [
            (wq.nrows(), wq.ncols()),
            (wk.nrows(), wk.ncols()),
            (wv.nrows(), wv.ncols()),
            (wo.nrows(), wo.ncols()),
            (up.nrows(), up.ncols()),
            (down.nrows(), down.ncols()),
        ];
        let layer_sites: Vec<Site> = units.iter().zip(shapes).map(|(u, (r, c))| u.site(r, c)).collect();
        let (x, heads_out, h2, hidden) = (
            value(layer.normed_stream)?,
            stack(layer.reads.iter().map(|&r| value(r)).collect::<Result<_, _>>()?, 1)?,
            value(layer.normed)?,
            value(layer.active)?,
        );
        let read_of = |k: usize, i: usize| layer_sites[k].v.column(i).to_owned();
        // The attention: per head its slices; a zero attention one component of its zero slices.
        let attention_units: Vec<usize> = if units[..3].iter().all(|u| u.writes.is_empty()) { Vec::new() } else { (0..heads).collect() };
        if attention_units.is_empty() {
            let slices = [[site(0), 0], [site(1), 0], [site(2), 0]];
            for arm in [&mut per_slice, &mut grouped] {
                arm.push(component(json!({"own": [site(0), 0]}), 1.0, &slices));
            }
            direction.push(component(json!({"direction": {"site": site(0), "coefficients": vec![0.0; x.ncols() + 1]}}), 1.0, &slices));
        }
        for &h in &attention_units {
            let mut slices = Vec::new();
            let mut reads = Vec::new();
            for k in 0..3 {
                for i in units[k].of(h) {
                    slices.push([site(k), i]);
                    reads.push(read_of(k, i));
                    per_slice.push(component(json!({"own": [site(k), i]}), own_width(&x, read_of(k, i).view(), 1.0), &[[site(k), i]]));
                }
            }
            for i in units[3].of(h) {
                slices.push([site(3), i]);
                per_slice.push(component(json!({"own": [site(3), i]}), own_width(&heads_out, read_of(3, i).view(), 1.0), &[[site(3), i]]));
            }
            let Some(first) = slices.first().copied() else { continue };
            let stacked = concatenate(Axis(1), &reads.iter().map(|r| r.view().insert_axis(Axis(1))).collect::<Vec<_>>()).map_err(error)?;
            let norms = x.dot(&stacked).map_axis(Axis(1), |r| r.dot(&r).sqrt());
            grouped.push(component(json!({"own": first}), spread(norms.iter().copied()), &slices));
            direction.push(component(direction_read(&x, read_of(0, first[1]).view(), first[0]), 1.0, &slices));
        }
        // The MLP: per neuron its c_fc row and down_proj column.
        for n in 0..up.nrows() {
            let (ins, outs) = (units[4].of(n), units[5].of(n));
            for &i in &ins {
                per_slice.push(component(json!({"own": [site(4), i]}), own_width(&h2, read_of(4, i).view(), 1.0), &[[site(4), i]]));
            }
            for &i in &outs {
                per_slice.push(component(json!({"own": [site(5), i]}), own_width(&hidden, read_of(5, i).view(), 1.0), &[[site(5), i]]));
            }
            let slices: Vec<[usize; 2]> = ins.iter().map(|&i| [site(4), i]).chain(outs.iter().map(|&i| [site(5), i])).collect();
            if let Some(&i) = ins.first() {
                grouped.push(component(json!({"own": [site(4), i]}), own_width(&h2, read_of(4, i).view(), 1.0), &slices));
                direction.push(component(direction_read(&h2, read_of(4, i).view(), site(4)), 1.0, &slices));
            }
        }
        for (k, s) in layer_sites.iter().enumerate() {
            let name = format!("h.{l}.{}", crate::library_vpd::EXPORT_NAMES[k]);
            for (suffix, t) in [("U", &s.u), ("V", &s.v)] {
                let bytes: Vec<u8> = t.iter().flat_map(|v| v.to_le_bytes()).collect();
                std::fs::write(dir.join(format!("{name}.{suffix}.f64")), bytes).map_err(error)?;
                files.insert(format!("{name}.{suffix}"), json!({"shape": [t.nrows(), t.ncols()]}));
            }
            sites.push(name);
        }
    }
    std::fs::write(dir.join("export.json"), json!({"config": {"sites": sites, "frame": "heads_neurons"}, "files": files}).to_string()).map_err(error)?;
    let arms = json!([{"arm": "per_slice_own", "components": per_slice}, {"arm": "grouped_own", "components": grouped}, {"arm": "grouped_direction", "components": direction}]);
    std::fs::write(dir.join("start.json"), arms.to_string()).map_err(error)?;
    Ok(json!({"frame": "heads_neurons", "components": grouped.len(), "slices": per_slice.len(), "sites": sites.len()}))
}
