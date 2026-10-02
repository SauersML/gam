//! The starting library of one linear map (#2951): its Fisher-whitened singular pieces, rank-1
//! pieces whose sum is the map.
//!
//! A site `y = W x` is written as `W = Σ_c u_c v_cᵀ` on the centred input `x − μ` (with the bias
//! `W μ` always on); the masked program (`crate::masked`) lists per input which pieces are on.
//! Dropping a set of pieces on input `x` costs, to second order in the output Fisher `B` of the
//! site's written value, `½ ‖B^{1/2} Σ_{c dropped} u_c (v_c · (x − μ))‖²` nats. In whitened
//! coordinates (`ξ = A^{-1/2}(x − μ)` with `A` the input covariance, `M = B^{1/2} W A^{1/2}`) the
//! singular pieces of `M` make that cost a sum of the pieces' own terms: each piece is one
//! singular direction, so the pieces are orthogonal in the metric both sides of the site use.
//! What the supports of `A` and `B` leave out is appended as the singular pieces of the remainder,
//! so all pieces on is the map itself.

use super::dense::{eigh, svd};
use gam_linalg::roundoff::{SymmetricAssembly, accumulation_growth};
use ndarray::{Array1, Array2, Axis, concatenate};

/// One site's data: the map, the input's second moment and mean, and the output Fisher.
pub struct Site {
    /// `d_out × d_in`.
    pub w: Array2<f64>,
    /// `E[x xᵀ]` (`d_in × d_in`).
    pub second_moment: Array2<f64>,
    pub mean: Array1<f64>,
    /// `E[g gᵀ]` at the output (`d_out × d_out`).
    pub fisher: Array2<f64>,
}

/// A library in the program's coordinates: `v` is `d_in × C`, `u` is `C × d_out`, and `v u = Wᵀ`
/// on the centred input.
pub struct Library {
    pub v: Array2<f64>,
    pub u: Array2<f64>,
    /// Pieces appended to make the library exact.
    pub exactness_pieces: usize,
}

/// `(M^{1/2}, M^{-1/2})` of a symmetric positive semidefinite matrix over the eigenvalues above
/// its decomposition's band.
fn roots(m: &Array2<f64>) -> Result<(Array2<f64>, Array2<f64>), String> {
    let mut sym = m.clone();
    let n = sym.nrows();
    for i in 0..n {
        for j in (i + 1)..n {
            let v = 0.5 * (sym[[i, j]] + sym[[j, i]]);
            sym[[i, j]] = v;
            sym[[j, i]] = v;
        }
    }
    let decomposed = eigh(sym.view(), SymmetricAssembly::Mirrored, None).map_err(|e| format!("{e:?}"))?;
    let (mut half, mut inverse) = (decomposed.vectors.clone(), decomposed.vectors.clone());
    for (k, l) in decomposed.values.iter().enumerate() {
        let (h, i) = if *l > decomposed.band { (l.sqrt(), 1.0 / l.sqrt()) } else { (0.0, 0.0) };
        half.column_mut(k).mapv_inplace(|x| x * h);
        inverse.column_mut(k).mapv_inplace(|x| x * i);
    }
    Ok((half.dot(&decomposed.vectors.t()), inverse.dot(&decomposed.vectors.t())))
}

/// The whitened problem of one site.
struct Whitened {
    /// `A^{-1/2}` and `B^{-1/2}`.
    a_inverse: Array2<f64>,
    b_inverse: Array2<f64>,
    /// `M = B^{1/2} W A^{1/2}`.
    m: Array2<f64>,
}

impl Whitened {
    fn new(site: &Site) -> Result<Self, String> {
        let covariance = &site.second_moment - &outer(&site.mean, &site.mean);
        let (a_half, a_inverse) = roots(&covariance)?;
        let (b_half, b_inverse) = roots(&site.fisher)?;
        let m = b_half.dot(&site.w).dot(&a_half);
        Ok(Self { a_inverse, b_inverse, m })
    }
}

fn outer(a: &Array1<f64>, b: &Array1<f64>) -> Array2<f64> {
    let (n, m) = (a.len(), b.len());
    Array2::from_shape_fn((n, m), |(i, j)| a[i] * b[j])
}

/// The rank-1 pieces `(u: r × d_out, v: r × d_in)` of `residual` (`d_out × d_in`): its singular
/// values above the band, `√σ` on each side.
fn singular_pieces(residual: &Array2<f64>) -> Result<(Array2<f64>, Array2<f64>), String> {
    singular_pieces_above(residual, 0.0)
}

/// [`singular_pieces`] keeping only the singular values above `floor` as well.
fn singular_pieces_above(residual: &Array2<f64>, floor: f64) -> Result<(Array2<f64>, Array2<f64>), String> {
    let decomposed = svd(residual.view(), false).map_err(|e| format!("{e:?}"))?;
    let kept: Vec<usize> =
        (0..decomposed.singular_values.len()).filter(|&i| decomposed.singular_values[i] > decomposed.band.max(floor)).collect();
    let (d_out, d_in) = residual.dim();
    let mut u = Array2::<f64>::zeros((kept.len(), d_out));
    let mut v = Array2::<f64>::zeros((kept.len(), d_in));
    for (row, &i) in kept.iter().enumerate() {
        let root = decomposed.singular_values[i].sqrt();
        u.row_mut(row).assign(&(&decomposed.u.column(i) * root));
        v.row_mut(row).assign(&(&decomposed.vt.row(i) * root));
    }
    Ok((u, v))
}

impl Library {
    /// The largest entry of `Wᵀ − v u`, against the largest entry of `W`.
    pub fn exactness(&self, w: &Array2<f64>) -> f64 {
        let error = &w.t() - &self.v.dot(&self.u);
        let largest = w.iter().fold(0.0_f64, |m, x| m.max(x.abs()));
        error.iter().fold(0.0_f64, |m, x| m.max(x.abs())) / largest.max(f64::MIN_POSITIVE)
    }
}

/// The Fisher-whitened singular pieces of a site, exact: `M = B^{1/2} W A^{1/2} = P S Qᵀ`, piece
/// `c` is `u_c = B^{-1/2} p_c √s_c`, `v_c = A^{-1/2} q_c √s_c` (so dropping a set costs, to second
/// order in the global Fisher, the sum of the pieces' own `s_c (q_c · ξ)²/2`), with what the
/// supports of `A` and `B` leave appended beyond the product's rounding band.
pub fn fisher_svd(site: &Site) -> Result<Library, String> {
    let whitened = Whitened::new(site)?;
    let (u_white, v_white) = singular_pieces(&whitened.m)?;
    let mut u = u_white.dot(&whitened.b_inverse);
    let mut v = whitened.a_inverse.dot(&v_white.t());
    let (d_out, d_in) = site.w.dim();
    let left = &site.w - &u.t().dot(&v.t());
    let band = u.mapv(f64::abs).t().dot(&v.mapv(f64::abs).t()) * accumulation_growth(u.nrows());
    let within = left.iter().zip(band.iter()).all(|(r, b)| r.abs() <= *b);
    let band_norm = band.iter().map(|b| b * b).sum::<f64>().sqrt();
    let (u_left, v_left) = if within { (Array2::zeros((0, d_out)), Array2::zeros((0, d_in))) } else { singular_pieces_above(&left, band_norm)? };
    let extra = u_left.nrows();
    if extra > 0 {
        u = concatenate(Axis(0), &[u.view(), u_left.view()]).map_err(|e| e.to_string())?;
        v = concatenate(Axis(1), &[v.view(), v_left.t()]).map_err(|e| e.to_string())?;
    }
    Ok(Library { v, u, exactness_pieces: extra })
}
