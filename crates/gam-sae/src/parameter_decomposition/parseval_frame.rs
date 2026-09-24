//! Parseval-frame decompositions of a weight: overcomplete, exactly faithful by the
//! geometry of a Stiefel manifold, with rank-`m` pieces chosen per token (#2951).
//!
//! # The object
//!
//! Take one side of a weight `W ∈ ℝ^{q×p}`: its output space (`d = q`) or its input
//! space (`d = p`). A frame of `K` atoms `O_k ∈ ℝ^{d×m}` is stacked as the rows of
//! `X = [O_1ᵀ; … ; O_Kᵀ] ∈ ℝ^{Km × d}`. When `X` has orthonormal columns,
//! `Xᵀ X = I_d`, the frame is Parseval: `Σ_k O_k O_kᵀ = I_d` exactly, for any `K m ≥ d`.
//! The pieces
//!
//! ```text
//! output side:  B_k = O_k O_kᵀ W        input side:  B_k = W O_k O_kᵀ
//! ```
//!
//! are rank `m` and sum to `W` with no remainder: faithfulness is a property of the
//! point `X ∈ St(Km, d)`, not a penalty to be traded against anything. `K m > d` is an
//! overcomplete frame, the room superposition needs. The frame is an ordinary
//! component basis of the residual anchor ([`ParsevalFrame::anchor`]): basis block `k`
//! is the factored piece `B_k`, the coefficients are the identity, and the anchor's
//! residual `W − Σ_k B_k` vanishes up to the frame's orthonormality defect
//! ([`ParsevalFrame::faithfulness`]).
//!
//! # Per-token pieces
//!
//! A row `z` uses the `L` pieces that write the most output energy on it:
//! `e_k(z) = ‖B_k z‖²`, which is `‖O_kᵀ W z‖²` on the output side and
//! `c_kᵀ (O_kᵀ Wᵀ W O_k) c_k`, `c_k = O_kᵀ z`, on the input side (an `m × m` Gram per
//! atom, formed once). The selection is the anchor mask with `m_Δ = 0` and `m_k = 1` on
//! the `L` chosen pieces; [`ParsevalFrame::restrict_rows`] executes it row by row
//! without forming a piece, and the tests check it against the anchor itself.
//!
//! # Geometry
//!
//! The frame is moved on `St(Km, d)` by projecting a Euclidean cotangent `G` to the
//! tangent space, `G − X sym(Xᵀ G)` ([`tangent_project`]), and retracting a step with
//! the QR factor whose `R` has a positive diagonal ([`qr_retract`]). The sign
//! normalization is load-bearing: a column of `Q` flipped by the factorization conjugates
//! every piece by that coordinate's sign, `P_k ↦ D P_k D`, so an unnormalized QR is not
//! a retraction of this decomposition. The step size and the cotangent belong to the
//! caller (the objective is the model's own divergence, executed outside this crate).
use gam_linalg::faer_ndarray::{FaerLinalgError, FaerQr, FaerSvd};
use ndarray::{Array2, ArrayView2, Axis, s};
use std::fmt;

use super::apply::{ApplyError, FactoredEdit};
use super::lift::{AnchorMask, ComponentCoefficients, LiftError, ResidualAnchor, TensorId, TensorRegistry};

/// Which space of the weight the frame resolves.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FrameSide {
    /// Atoms live in the output space; pieces are `O_k O_kᵀ W`.
    Output,
    /// Atoms live in the input space; pieces are `W O_k O_kᵀ`.
    Input,
}

/// Why a frame could not be built or applied.
#[derive(Debug)]
pub enum FrameError {
    /// The dictionary alternation increased its residual, which exact alternation cannot do.
    NotMonotone { residual_trace: Vec<f64> },
    /// The SVD owner failed.
    Svd(String),
    /// `X` is not `Km × d` for the declared atom dimension and side.
    Shape { rows: usize, cols: usize, atom_dim: usize, side_dim: usize },
    /// `X` is not Parseval within its roundoff band.
    NotParseval { defect: f64, band: f64 },
    /// A frame or weight entry is not finite.
    NonFinite,
    /// More pieces requested per row than the frame holds.
    TooManyActive { active: usize, atoms: usize },
    /// The QR owner failed.
    Qr(FaerLinalgError),
    /// The factored-edit owner refused the pieces.
    Apply(ApplyError),
    /// The residual anchor refused the pieces.
    Lift(LiftError),
}

impl fmt::Display for FrameError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Shape { rows, cols, atom_dim, side_dim } => write!(
                f,
                "parseval_frame: frame is {rows} x {cols}, expected (K*{atom_dim}) x {side_dim}"
            ),
            Self::NotParseval { defect, band } => write!(
                f,
                "parseval_frame: ||X^T X - I|| = {defect:.3e} exceeds the roundoff band {band:.3e}"
            ),
            Self::NonFinite => write!(f, "parseval_frame: non-finite entry"),
            Self::TooManyActive { active, atoms } => {
                write!(f, "parseval_frame: {active} active pieces requested of {atoms}")
            }
            Self::Qr(error) => write!(f, "parseval_frame: QR failed: {error}"),
            Self::NotMonotone { residual_trace } => write!(
                f,
                "parseval_frame: dictionary alternation increased its residual: {residual_trace:?}"
            ),
            Self::Svd(error) => write!(f, "parseval_frame: SVD failed: {error}"),
            Self::Apply(error) => write!(f, "parseval_frame: {error}"),
            Self::Lift(error) => write!(f, "parseval_frame: {error}"),
        }
    }
}

impl std::error::Error for FrameError {}

impl From<FaerLinalgError> for FrameError {
    fn from(error: FaerLinalgError) -> Self {
        Self::Qr(error)
    }
}

impl From<ApplyError> for FrameError {
    fn from(error: ApplyError) -> Self {
        Self::Apply(error)
    }
}

impl From<LiftError> for FrameError {
    fn from(error: LiftError) -> Self {
        Self::Lift(error)
    }
}

/// A Parseval frame on one side of a weight.
#[derive(Clone, Debug)]
pub struct ParsevalFrame {
    side: FrameSide,
    atom_dim: usize,
    /// `Km × d`, orthonormal columns.
    frame: Array2<f64>,
    /// `‖Xᵀ X − I‖_F`, measured at construction.
    defect: f64,
}

/// The normalized orthonormality band of a `n × d` factor: `(n + d)·d·ε`.
fn parseval_band(rows: usize, cols: usize) -> f64 {
    ((rows + cols) * cols) as f64 * f64::EPSILON
}

fn gram_defect(frame: ArrayView2<'_, f64>) -> f64 {
    let d = frame.ncols();
    let mut g = frame.t().dot(&frame);
    for i in 0..d {
        g[[i, i]] -= 1.0;
    }
    g.iter().map(|v| v * v).sum::<f64>().sqrt()
}

impl ParsevalFrame {
    /// A frame on `side` of a weight whose side has dimension `side_dim`. Refuses a
    /// frame that is not Parseval within its roundoff band.
    pub fn new(
        side: FrameSide,
        atom_dim: usize,
        side_dim: usize,
        frame: Array2<f64>,
    ) -> Result<Self, FrameError> {
        let (rows, cols) = frame.dim();
        if atom_dim == 0 || cols != side_dim || rows % atom_dim != 0 || rows < cols {
            return Err(FrameError::Shape { rows, cols, atom_dim, side_dim });
        }
        if !frame.iter().all(|v| v.is_finite()) {
            return Err(FrameError::NonFinite);
        }
        let defect = gram_defect(frame.view());
        let band = parseval_band(rows, cols);
        if defect > band {
            return Err(FrameError::NotParseval { defect, band });
        }
        Ok(Self { side, atom_dim, frame, defect })
    }

    /// The side the frame resolves.
    #[must_use]
    pub fn side(&self) -> FrameSide {
        self.side
    }

    /// Atoms `K`.
    #[must_use]
    pub fn atoms(&self) -> usize {
        self.frame.nrows() / self.atom_dim
    }

    /// `m`.
    #[must_use]
    pub fn atom_dim(&self) -> usize {
        self.atom_dim
    }

    /// `X`.
    #[must_use]
    pub fn frame(&self) -> ArrayView2<'_, f64> {
        self.frame.view()
    }

    /// Atom `k` as `d × m` columns.
    fn atom(&self, k: usize) -> ArrayView2<'_, f64> {
        self.frame.slice(s![k * self.atom_dim..(k + 1) * self.atom_dim, ..]).reversed_axes()
    }

    /// The pieces of `weight` as one factored edit: `K` blocks of rank `m`.
    pub fn pieces(&self, weight: ArrayView2<'_, f64>) -> Result<FactoredEdit, FrameError> {
        let (q, p) = weight.dim();
        let km = self.frame.nrows();
        let (left, right) = match self.side {
            FrameSide::Output => {
                if q != self.frame.ncols() {
                    return Err(FrameError::Shape {
                        rows: km,
                        cols: self.frame.ncols(),
                        atom_dim: self.atom_dim,
                        side_dim: q,
                    });
                }
                let left = self.frame.t().to_owned(); // q × Km, columns O_k
                let right = weight.t().dot(&left); // p × Km, columns Wᵀ O_k
                (left, right)
            }
            FrameSide::Input => {
                if p != self.frame.ncols() {
                    return Err(FrameError::Shape {
                        rows: km,
                        cols: self.frame.ncols(),
                        atom_dim: self.atom_dim,
                        side_dim: p,
                    });
                }
                let right = self.frame.t().to_owned(); // p × Km, columns O_k
                let left = weight.dot(&right); // q × Km, columns W O_k
                (left, right)
            }
        };
        Ok(FactoredEdit::new(left, right)?)
    }

    /// The frame's pieces anchored on the registered teacher tensor: `K` components of
    /// rank `m`, identity coefficients.
    pub fn anchor<'a>(
        &self,
        registry: &TensorRegistry,
        storage: TensorId,
        native: ArrayView2<'a, f64>,
    ) -> Result<ResidualAnchor<'a>, FrameError> {
        let pieces = self.pieces(native)?;
        Ok(ResidualAnchor::new(
            registry,
            storage,
            native,
            pieces,
            vec![self.atom_dim; self.atoms()],
            ComponentCoefficients::Basis,
        )?)
    }

    /// `‖W − Σ_k B_k‖_F ≤ ‖Xᵀ X − I‖_F ‖W‖_F`: the anchor residual of an exact frame is
    /// bounded by the measured orthonormality defect (`Σ_k O_k O_kᵀ = Xᵀ X`).
    #[must_use]
    pub fn faithfulness(&self, weight: ArrayView2<'_, f64>) -> f64 {
        let norm = weight.iter().map(|v| v * v).sum::<f64>().sqrt();
        self.defect * norm
    }

    /// Each row's `L` highest-energy pieces, as anchor masks (`m_Δ = 0`).
    pub fn select(
        &self,
        weight: ArrayView2<'_, f64>,
        inputs: ArrayView2<'_, f64>,
        active: usize,
    ) -> Result<Vec<AnchorMask>, FrameError> {
        let energy = self.energies(weight, inputs);
        let k = self.atoms();
        if active > k {
            return Err(FrameError::TooManyActive { active, atoms: k });
        }
        Ok(energy
            .outer_iter()
            .map(|e| {
                let mut order: Vec<usize> = (0..k).collect();
                order.sort_by(|&a, &b| e[b].total_cmp(&e[a]).then(a.cmp(&b)));
                let mut components = vec![0.0; k];
                for &c in order.iter().take(active) {
                    components[c] = 1.0;
                }
                AnchorMask { residual: 0.0, components }
            })
            .collect())
    }

    /// `n × K` piece energies `‖B_k z‖²`.
    fn energies(&self, weight: ArrayView2<'_, f64>, inputs: ArrayView2<'_, f64>) -> Array2<f64> {
        let (n, k, m) = (inputs.nrows(), self.atoms(), self.atom_dim);
        let coords = match self.side {
            FrameSide::Output => inputs.dot(&weight.t()).dot(&self.frame.t()), // n × Km
            FrameSide::Input => inputs.dot(&self.frame.t()),
        };
        let mut energy = Array2::<f64>::zeros((n, k));
        match self.side {
            FrameSide::Output => {
                for (mut row, c) in energy.outer_iter_mut().zip(coords.outer_iter()) {
                    for a in 0..k {
                        row[a] = c.slice(s![a * m..(a + 1) * m]).iter().map(|v| v * v).sum();
                    }
                }
            }
            FrameSide::Input => {
                let grams: Vec<Array2<f64>> = (0..k)
                    .map(|a| {
                        let wo = weight.dot(&self.atom(a));
                        wo.t().dot(&wo)
                    })
                    .collect();
                for (mut row, c) in energy.outer_iter_mut().zip(coords.outer_iter()) {
                    for a in 0..k {
                        let ca = c.slice(s![a * m..(a + 1) * m]);
                        row[a] = ca.dot(&grams[a].dot(&ca));
                    }
                }
            }
        }
        energy
    }

    /// Every row through its own `L` pieces, `Σ_{k ∈ A(z)} B_k z`, without forming a
    /// piece. Rows are inputs `z` of the weight (`n × p`); the result is `n × q`.
    pub fn restrict_rows(
        &self,
        weight: ArrayView2<'_, f64>,
        inputs: ArrayView2<'_, f64>,
        active: usize,
    ) -> Result<Array2<f64>, FrameError> {
        let masks = self.select(weight, inputs, active)?;
        let mut out = Array2::<f64>::zeros((inputs.nrows(), weight.nrows()));
        for ((mut row, z), mask) in out.outer_iter_mut().zip(inputs.outer_iter()).zip(&masks) {
            for (k, &on) in mask.components.iter().enumerate() {
                if on == 0.0 {
                    continue;
                }
                let o = self.atom(k);
                match self.side {
                    FrameSide::Output => {
                        let y = weight.dot(&z);
                        row += &o.dot(&o.t().dot(&y));
                    }
                    FrameSide::Input => {
                        let c = o.t().dot(&z);
                        row += &weight.dot(&o.dot(&c));
                    }
                }
            }
        }
        Ok(out)
    }
}

/// `G − X sym(Xᵀ G)`: the component of a Euclidean cotangent tangent to `St(n, d)` at `X`.
#[must_use]
pub fn tangent_project(frame: ArrayView2<'_, f64>, cotangent: ArrayView2<'_, f64>) -> Array2<f64> {
    let a = frame.t().dot(&cotangent);
    let sym = (&a + &a.t()) * 0.5;
    &cotangent - &frame.dot(&sym)
}

/// The QR retraction `R_X(ξ) = qf(X + ξ)` with `R` normalized to a positive diagonal.
pub fn qr_retract(frame: ArrayView2<'_, f64>, step: ArrayView2<'_, f64>) -> Result<Array2<f64>, FrameError> {
    let moved = &frame + &step;
    let (q, r) = moved.qr()?;
    let d = frame.ncols();
    let mut q = q.slice(s![.., ..d]).to_owned();
    for (j, mut col) in q.axis_iter_mut(Axis(1)).enumerate() {
        if r[[j, j]] < 0.0 {
            col.mapv_inplace(|v| -v);
        }
    }
    Ok(q)
}

/// Orthogonal dictionary learning of a complete frame on a bank of rows `D` (`n × d`):
///
/// # What sparsity identifies
///
/// The fit identifies a subspace only up to the rotations that preserve every row's
/// selection. If every row's energy lies in the same union `U` of `r` atom groups, any
/// orthogonal `Q` acting inside `U` (and as the identity off it) maps an optimal frame to
/// another optimal frame with identical codes' energies, identical selections and the same
/// residual, so the individual planes inside `U` are a gauge (P1), not an output: the
/// grokked modular-addition embedding, whose tokens all use all five key frequency planes,
/// is the measured case (#2951). What splits `U` into planes is a symmetry the rows carry
/// (the invariant planes of a shift operator, `spectral`/`schur`), not sparsity.
///
/// alternate (a) each row coded by its `active` highest-energy atom groups of the current
/// frame, `S = mask ⊙ (D Xᵀ)`, and (b) the orthogonal Procrustes update
/// `X = polar(Sᵀ D)`, the orthogonal matrix nearest `Sᵀ D`, which minimizes
/// `‖D − S X‖_F` over `O(d)` for the fixed codes. Step (a) is the exact minimizer over
/// `active`-sparse codes for a fixed orthogonal frame (Parseval: the error is the dropped
/// energy), so the residual never increases; an increase beyond roundoff is refused.
/// Stops at a fixed point of the selection. Returns the frame and the residual trace.
pub fn orthogonal_dictionary_fit(
    bank: ArrayView2<'_, f64>,
    start: ArrayView2<'_, f64>,
    atom_dim: usize,
    active: usize,
) -> Result<(Array2<f64>, Vec<f64>), FrameError> {
    let (n, d) = bank.dim();
    if start.dim() != (d, d) || atom_dim == 0 || d % atom_dim != 0 {
        return Err(FrameError::Shape { rows: start.nrows(), cols: start.ncols(), atom_dim, side_dim: d });
    }
    let k = d / atom_dim;
    if active > k {
        return Err(FrameError::TooManyActive { active, atoms: k });
    }
    let mut frame = start.to_owned();
    let mut trace = Vec::new();
    let mut previous: Option<Vec<Vec<usize>>> = None;
    let energy_total: f64 = bank.iter().map(|v| v * v).sum();
    loop {
        let coords = bank.dot(&frame.t()); // n × d
        let mut codes = Array2::<f64>::zeros((n, d));
        let mut selection = Vec::with_capacity(n);
        let mut kept = 0.0;
        for (i, row) in coords.outer_iter().enumerate() {
            let mut energy: Vec<(usize, f64)> = (0..k)
                .map(|a| (a, row.slice(s![a * atom_dim..(a + 1) * atom_dim]).iter().map(|v| v * v).sum()))
                .collect();
            energy.sort_by(|x, y| y.1.total_cmp(&x.1).then(x.0.cmp(&y.0)));
            let mut chosen: Vec<usize> = energy.iter().take(active).map(|e| e.0).collect();
            chosen.sort_unstable();
            for &a in &chosen {
                let (lo, hi) = (a * atom_dim, (a + 1) * atom_dim);
                codes.slice_mut(s![i, lo..hi]).assign(&row.slice(s![lo..hi]));
            }
            kept += energy.iter().take(active).map(|e| e.1).sum::<f64>();
            selection.push(chosen);
        }
        let residual = (energy_total - kept).max(0.0);
        if let Some(&last) = trace.last() {
            let band = 64.0 * f64::EPSILON * energy_total * (n.max(d) as f64);
            if residual > last + band {
                trace.push(residual);
                return Err(FrameError::NotMonotone { residual_trace: trace });
            }
        }
        trace.push(residual);
        if previous.as_ref() == Some(&selection) {
            return Ok((frame, trace));
        }
        previous = Some(selection);
        let target = codes.t().dot(&bank); // d × d
        let (u, _, vt) = target.svd(true, true).map_err(|e| FrameError::Svd(e.to_string()))?;
        frame = u.expect("requested").dot(&vt.expect("requested"));
    }
}

/// A Parseval frame with the standard basis of `d` as its first `d` rows (atoms group
/// consecutive coordinates `m` at a time) and, when `overcomplete`, a rotated copy
/// `R` below it, the whole scaled by `1/√2`: `Xᵀ X = (I + Rᵀ R)/2 = I` exactly for
/// orthogonal `R`. On a weight's neuron side its first atoms are neuron groups.
pub fn neuron_frame(d: usize, rotation: Option<ArrayView2<'_, f64>>) -> Array2<f64> {
    match rotation {
        None => Array2::eye(d),
        Some(r) => {
            let mut x = Array2::<f64>::zeros((2 * d, d));
            x.slice_mut(s![..d, ..]).assign(&Array2::<f64>::eye(d));
            x.slice_mut(s![d.., ..]).assign(&r);
            x / std::f64::consts::SQRT_2
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parameter_decomposition::lift::TieOrientation;
    use rand::rngs::StdRng;
    use rand::{RngExt, SeedableRng};

    fn gaussian(rng: &mut StdRng, rows: usize, cols: usize) -> Array2<f64> {
        Array2::from_shape_simple_fn((rows, cols), || {
            let u: f64 = rng.random_range(1e-12..1.0);
            let v: f64 = rng.random_range(0.0..1.0);
            (-2.0 * u.ln()).sqrt() * (std::f64::consts::TAU * v).cos()
        })
    }

    fn stiefel(rng: &mut StdRng, n: usize, d: usize) -> Array2<f64> {
        qr_retract(gaussian(rng, n, d).view(), Array2::<f64>::zeros((n, d)).view()).unwrap()
    }

    /// The pieces of an overcomplete Parseval frame sum to the weight on both sides.
    #[test]
    fn pieces_sum_to_the_weight() {
        let mut rng = StdRng::seed_from_u64(1);
        let (q, p, m) = (6, 5, 2);
        let weight = gaussian(&mut rng, q, p);
        for (side, d) in [(FrameSide::Output, q), (FrameSide::Input, p)] {
            let km = 2 * d + (2 * d) % m;
            let frame = ParsevalFrame::new(side, m, d, stiefel(&mut rng, km, d)).unwrap();
            let pieces = frame.pieces(weight.view()).unwrap();
            let sum = pieces.left().dot(&pieces.right().t());
            let err = (&sum - &weight).iter().map(|v| v * v).sum::<f64>().sqrt();
            assert!(err <= frame.faithfulness(weight.view()) + 1e-13, "{side:?}: {err:e}");
        }
    }

    /// Row-by-row restriction equals the residual anchor under each row's own mask.
    #[test]
    fn restriction_is_the_anchor_under_the_selection_mask() {
        let mut rng = StdRng::seed_from_u64(2);
        let (q, p, m, n, active) = (6, 8, 2, 12, 3);
        let weight = gaussian(&mut rng, q, p);
        let inputs = gaussian(&mut rng, n, p);
        let mut registry = TensorRegistry::default();
        registry.register_storage(TensorId("w".into()), weight.view().into_dyn()).unwrap();
        for (side, d) in [(FrameSide::Output, q), (FrameSide::Input, p)] {
            let frame = ParsevalFrame::new(side, m, d, stiefel(&mut rng, 2 * d, d)).unwrap();
            let anchor = frame.anchor(&registry, TensorId("w".into()), weight.view()).unwrap();
            let fast = frame.restrict_rows(weight.view(), inputs.view(), active).unwrap();
            let masks = frame.select(weight.view(), inputs.view(), active).unwrap();
            for (i, mask) in masks.iter().enumerate() {
                let row = inputs.slice(s![i..i + 1, ..]);
                let via = anchor.apply(mask, row, TieOrientation::Identity).unwrap();
                let gap = (&via.row(0) - &fast.row(i)).iter().map(|v| v.abs()).fold(0.0, f64::max);
                assert!(gap <= 1e-12, "{side:?} row {i}: {gap:e}");
            }
        }
    }

    /// All pieces on is the teacher, bit for bit, through the anchor's native path.
    #[test]
    fn all_on_is_the_teacher() {
        let mut rng = StdRng::seed_from_u64(3);
        let weight = gaussian(&mut rng, 5, 4);
        let inputs = gaussian(&mut rng, 7, 4);
        let mut registry = TensorRegistry::default();
        registry.register_storage(TensorId("w".into()), weight.view().into_dyn()).unwrap();
        let frame = ParsevalFrame::new(FrameSide::Output, 1, 5, stiefel(&mut rng, 10, 5)).unwrap();
        let anchor = frame.anchor(&registry, TensorId("w".into()), weight.view()).unwrap();
        let on = anchor.apply(&AnchorMask::all_on(10), inputs.view(), TieOrientation::Identity).unwrap();
        let native = anchor.native_apply(inputs.view(), TieOrientation::Identity).unwrap();
        assert_eq!(*on, *native);
    }

    /// The sign-normalized QR retraction stays Parseval, is the identity at a zero step,
    /// and moves continuously; an unnormalized factor would flip coordinate signs.
    #[test]
    fn retraction_stays_parseval_and_is_continuous() {
        let mut rng = StdRng::seed_from_u64(4);
        let x = stiefel(&mut rng, 12, 5);
        let same = qr_retract(x.view(), Array2::<f64>::zeros((12, 5)).view()).unwrap();
        let gap = (&same - &x).iter().map(|v| v.abs()).fold(0.0, f64::max);
        assert!(gap <= 1e-12, "zero step moved the frame by {gap:e}");
        let g = gaussian(&mut rng, 12, 5);
        let xi = tangent_project(x.view(), g.view()) * 1e-6;
        let y = qr_retract(x.view(), xi.view()).unwrap();
        assert!(gram_defect(y.view()) <= parseval_band(12, 5));
        let moved = (&y - &x).iter().map(|v| v.abs()).fold(0.0, f64::max);
        assert!(moved <= 1e-5, "a 1e-6 step moved the frame by {moved:e}");
    }

    /// Rows exactly `active`-sparse in a planted rotation: the alternation's residual never
    /// increases, stops at a fixed point, and ends no worse than the planted frame's own
    /// residual when started from it; from the identity it strictly improves.
    #[test]
    fn orthogonal_dictionary_fit_is_monotone_and_finds_sparse_rotation() {
        let mut rng = StdRng::seed_from_u64(6);
        let (n, d, m, active) = (400, 8, 2, 1);
        let planted = stiefel(&mut rng, d, d);
        let mut bank = Array2::<f64>::zeros((n, d));
        for i in 0..n {
            let a = i % (d / m);
            let c = gaussian(&mut rng, 1, m);
            let mut code = Array2::<f64>::zeros((1, d));
            code.slice_mut(s![0, a * m..(a + 1) * m]).assign(&c.row(0));
            bank.row_mut(i).assign(&code.dot(&planted).row(0));
        }
        let energy: f64 = bank.iter().map(|v| v * v).sum();
        let (_, from_planted) = orthogonal_dictionary_fit(bank.view(), planted.view(), m, active).unwrap();
        // the residual is total minus kept energy, so its roundoff floor is that subtraction's
        let floor = (n * d) as f64 * f64::EPSILON * energy;
        assert!(*from_planted.last().unwrap() <= floor, "{from_planted:?} vs floor {floor:e}");
        let (_, trace) = orthogonal_dictionary_fit(bank.view(), Array2::<f64>::eye(d).view(), m, active).unwrap();
        for w in trace.windows(2) {
            assert!(w[1] <= w[0] + 1e-9 * energy, "{trace:?}");
        }
        assert!(trace.last().unwrap() < trace.first().unwrap(), "{trace:?}");
    }

    /// Every row uses the same two planes: the fit's residual vanishes, but a rotation of
    /// the frame inside the union of those planes is exactly as optimal, so the planes are
    /// a gauge of the fit, not an output of it.
    #[test]
    fn a_union_every_row_uses_is_identified_but_its_planes_are_not() {
        let mut rng = StdRng::seed_from_u64(8);
        let (n, d, m, active) = (300, 6, 2, 2);
        let planted = stiefel(&mut rng, d, d);
        let mut bank = Array2::<f64>::zeros((n, d));
        for i in 0..n {
            let c = gaussian(&mut rng, 1, 2 * m);
            let mut code = Array2::<f64>::zeros((1, d));
            code.slice_mut(s![0, ..2 * m]).assign(&c.row(0));
            bank.row_mut(i).assign(&code.dot(&planted).row(0));
        }
        let energy: f64 = bank.iter().map(|v| v * v).sum();
        let floor = (n * d) as f64 * f64::EPSILON * energy;
        let (_, from_planted) = orthogonal_dictionary_fit(bank.view(), planted.view(), m, active).unwrap();
        assert!(*from_planted.last().unwrap() <= floor);
        // rotate inside the union of the first two planes (rows 0..4 of the frame)
        let theta: f64 = 0.7;
        let mut mix = Array2::<f64>::eye(d);
        mix[[0, 0]] = theta.cos();
        mix[[0, 2]] = -theta.sin();
        mix[[2, 0]] = theta.sin();
        mix[[2, 2]] = theta.cos();
        let rotated = mix.dot(&planted);
        let (_, from_rotated) = orthogonal_dictionary_fit(bank.view(), rotated.view(), m, active).unwrap();
        assert!(*from_rotated.last().unwrap() <= floor, "{from_rotated:?}");
        // and the rotated frame's first plane is not the planted first plane
        let a = planted.slice(s![0..2, ..]).to_owned();
        let b = rotated.slice(s![0..2, ..]).to_owned();
        let overlap = a.dot(&b.t());
        let det = overlap[[0, 0]] * overlap[[1, 1]] - overlap[[0, 1]] * overlap[[1, 0]];
        assert!(det.abs() < 0.9, "planes coincide: {det}");
    }

    /// The neuron frame is Parseval with and without its rotated copy.
    #[test]
    fn neuron_frame_is_parseval() {
        let mut rng = StdRng::seed_from_u64(5);
        let r = stiefel(&mut rng, 6, 6);
        for x in [neuron_frame(6, None), neuron_frame(6, Some(r.view()))] {
            assert!(gram_defect(x.view()) <= parseval_band(x.nrows(), 6), "{}", gram_defect(x.view()));
        }
    }
}
