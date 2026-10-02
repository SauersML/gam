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
    complete(&site.w, u_white.dot(&whitened.b_inverse), whitened.a_inverse.dot(&v_white.t()))
}

/// The library `(v: d_in × C, u: C × d_out)` with the singular pieces of what it leaves of `w`
/// beyond its product's rounding band appended, so all pieces on is the map.
fn complete(w: &Array2<f64>, mut u: Array2<f64>, mut v: Array2<f64>) -> Result<Library, String> {
    let (d_out, d_in) = w.dim();
    let left = w - &u.t().dot(&v.t());
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

/// What a site's Fisher-SVD needs, measured on its narrow side only, for a site one of whose sides
/// is too wide for its own `d × d` statistic (a language model's MLP): with `d_in ≤ d_out`, the
/// reads' covariance `A` and the Fisher pulled back through the map, `WᵀBW = E[(Wᵀg)(Wᵀg)ᵀ]`; with
/// `d_out < d_in`, the output Fisher `B` and the covariance of the written value, `W A Wᵀ`.
pub enum Narrow {
    Reads { covariance: Array2<f64>, pulled_fisher: Array2<f64> },
    Writes { fisher: Array2<f64>, written_covariance: Array2<f64> },
}

/// The pieces of [`fisher_svd`] from a site's narrow statistics, and each piece's singular value
/// `s_c` (its own second-order weight `u_cᵀ B u_c`; zero for an appended exactness piece). `M` is
/// never formed: with `Reads`, `MᵀM = A^{1/2} WᵀBW A^{1/2} = Q S² Qᵀ`, `v_c = A^{-1/2} q_c √s_c` and
/// `u_c = B^{-1/2} p_c √s_c = W A^{1/2} q_c / √s_c`; with `Writes`, `MMᵀ = B^{1/2} WAWᵀ B^{1/2} =
/// P S² Pᵀ`, `u_c = B^{-1/2} p_c √s_c` and `v_c = Wᵀ B^{1/2} p_c / √s_c`. Pieces are in decreasing
/// `s_c`; eigenvalues within the decomposition's band are left to the exactness pieces.
pub fn fisher_svd_narrow(w: &Array2<f64>, narrow: &Narrow) -> Result<(Library, Vec<f64>), String> {
    let (whitening, metric) = match narrow {
        Narrow::Reads { covariance, pulled_fisher } => (covariance, pulled_fisher),
        Narrow::Writes { fisher, written_covariance } => (fisher, written_covariance),
    };
    let (half, inverse) = roots(whitening)?;
    let mut gram = half.dot(metric).dot(&half);
    for i in 0..gram.nrows() {
        for j in (i + 1)..gram.ncols() {
            let mean = 0.5 * (gram[[i, j]] + gram[[j, i]]);
            gram[[i, j]] = mean;
            gram[[j, i]] = mean;
        }
    }
    let decomposed = eigh(gram.view(), SymmetricAssembly::Mirrored, None).map_err(|e| format!("{e:?}"))?;
    let mut kept: Vec<usize> = (0..decomposed.values.len()).filter(|&i| decomposed.values[i] > decomposed.band).collect();
    kept.sort_by(|a, b| decomposed.values[*b].total_cmp(&decomposed.values[*a]));
    let singular: Vec<f64> = kept.iter().map(|&i| decomposed.values[i].sqrt()).collect();
    let (d_out, d_in) = w.dim();
    let mut u = Array2::<f64>::zeros((kept.len(), d_out));
    let mut v = Array2::<f64>::zeros((d_in, kept.len()));
    for (c, (&i, s)) in kept.iter().zip(&singular).enumerate() {
        let direction = decomposed.vectors.column(i);
        let root = s.sqrt();
        match narrow {
            Narrow::Reads { .. } => {
                let whitened = half.dot(&direction);
                v.column_mut(c).assign(&(inverse.dot(&direction) * root));
                u.row_mut(c).assign(&(w.dot(&whitened) / root));
            }
            Narrow::Writes { .. } => {
                let whitened = half.dot(&direction);
                u.row_mut(c).assign(&(inverse.dot(&direction) * root));
                v.column_mut(c).assign(&(w.t().dot(&whitened) / root));
            }
        }
    }
    let library = complete(w, u, v)?;
    let mut weights = singular;
    weights.resize(library.u.nrows(), 0.0);
    Ok((library, weights))
}

/// One side of a site in the model's own unit basis ([`unit_pieces`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Units {
    /// Piece `j` is written unit `j`: `u_j = e_j`, `v_j = W[j, :]`.
    Written,
    /// Piece `j` is read unit `j`: `u_j = W[:, j]`, `v_j = e_j`.
    Read,
}

/// A site's pieces in the model's own unit basis on one side (an MLP's neurons): exactly the map,
/// one piece per unit.
pub fn unit_pieces(w: &Array2<f64>, side: Units) -> Library {
    let (d_out, d_in) = w.dim();
    match side {
        Units::Written => Library { v: w.t().to_owned(), u: Array2::eye(d_out), exactness_pieces: 0 },
        Units::Read => Library { v: Array2::eye(d_in), u: w.t().to_owned(), exactness_pieces: 0 },
    }
}

/// Per-input samples of a site: its reads `x_t` (rows × d_in) and the gradients `g_t` of `−log q_y`
/// at its written value, `y` drawn from the model's own output (rows × d_out); single precision,
/// since they only propose a library.
pub struct Attributions {
    pub reads: Array2<f32>,
    pub gradients: Array2<f32>,
}

/// What [`attribution_dictionary`] did.
#[derive(Clone, Debug)]
pub struct DictionaryReport {
    pub seeded: usize,
    pub kept: usize,
    pub iterations: usize,
    /// The share of the inputs' attribution energy their atoms carry, `Σ_t max_c (a_c·ĝ_t)²(b_c·x̂_t)² / Σ_t |ĝ_t|²|x̂_t|²`.
    pub captured: f64,
}

/// A site's starting library from what single inputs use of it (#2951): an overcomplete rank-one
/// dictionary of the inputs' attributions, completed to the map.
///
/// On input `t` a site `y = W x̃` (`x̃ = x − μ`) is used, to first order in the code's KL, through the
/// rank-one attribution `g_t x̃_tᵀ`: a piece `u vᵀ` contributes `(g_t·u)(v·x̃_t)` to `g_tᵀ W x̃_t`,
/// and dropping it on `t` costs `½ E[(g_t·u)²](v·x̃_t)²` nats. In the whitened coordinates of
/// [`fisher_svd`] (`ĝ = B^{-1/2} g`, `x̂ = A^{-1/2} x̃`, `M = B^{1/2} W A^{1/2}`, a piece `a bᵀ` with
/// `a = B^{1/2} u`, `b = A^{1/2} v`) an input is coded by the atom that carries most of its
/// attribution, `max_c (a_c·ĝ_t)²(b_c·x̂_t)²` over unit atoms. The atoms are fitted as rank-one
/// k-means: seeded from inputs drawn in proportion to their own attribution `(ĝ_tᵀ M x̂_t)²`, each
/// input assigned to its best atom, and each atom moved by one alternating power step on its inputs
/// (`a ← Σ (b·x̂)² (a·ĝ) ĝ`, then `b ← Σ (a·ĝ)² (b·x̂) x̂`; each step raises its own objective), until
/// a round gains less than one input's mean energy; `atoms` atoms are seeded. Their scales `λ` are the least squares of `M` on
/// them (the Fisher metric). An atom stays when the KL bits it carries for its inputs, `Σ n (λ a·ĝ
/// b·x̂)² / (2 ln 2)`, exceed their listing bits at its firing rate plus `bits_per_piece` (its library
/// bits, spread over the samples' share of the training inputs); the scales are refitted on the
/// atoms that stay, and the Fisher-whitened singular pieces of what they leave of `M` are appended,
/// so all pieces on is the map.
pub fn attribution_dictionary(
    site: &Site,
    samples: &Attributions,
    atoms: usize,
    observations: f64,
    bits_per_piece: f64,
    seed: u64,
) -> Result<(Library, DictionaryReport), String> {
    let whitened = Whitened::new(site)?;
    let (d_out, d_in) = site.w.dim();
    let rows = samples.reads.nrows();
    if rows == 0 || samples.gradients.nrows() != rows {
        return Err(format!("{rows} reads and {} gradients", samples.gradients.nrows()));
    }
    // The whitened samples (single precision; chunks are widened for products).
    let chunk = 2048;
    let mut x_hat = Array2::<f32>::zeros((rows, d_in));
    let mut g_hat = Array2::<f32>::zeros((rows, d_out));
    for start in (0..rows).step_by(chunk) {
        let end = (start + chunk).min(rows);
        let x = samples.reads.slice(ndarray::s![start..end, ..]).mapv(f64::from) - &site.mean;
        x_hat.slice_mut(ndarray::s![start..end, ..]).assign(&x.dot(&whitened.a_inverse).mapv(|v| v as f32));
        let g = samples.gradients.slice(ndarray::s![start..end, ..]).mapv(f64::from);
        g_hat.slice_mut(ndarray::s![start..end, ..]).assign(&g.dot(&whitened.b_inverse).mapv(|v| v as f32));
    }
    let widen = |m: &Array2<f32>, start: usize, end: usize| m.slice(ndarray::s![start..end, ..]).mapv(f64::from);
    // Each input's own attribution `ĝᵀ M x̂` and energy `|ĝ|²|x̂|²`.
    let mut own = Array1::<f64>::zeros(rows);
    let mut energy = Array1::<f64>::zeros(rows);
    for start in (0..rows).step_by(chunk) {
        let end = (start + chunk).min(rows);
        let (g, x) = (widen(&g_hat, start, end), widen(&x_hat, start, end));
        let mx = x.dot(&whitened.m.t());
        for r in 0..end - start {
            own[start + r] = g.row(r).dot(&mx.row(r));
            energy[start + r] = g.row(r).dot(&g.row(r)) * x.row(r).dot(&x.row(r));
        }
    }
    let total_energy = energy.sum();
    // Seeds: inputs drawn in proportion to their own squared attribution (deterministic stream).
    let atoms = atoms.min(rows);
    let weights: Vec<f64> = own.iter().map(|s| s * s).collect();
    let total_weight: f64 = weights.iter().sum();
    let mut state = seed | 1;
    let mut uniform = || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        (state >> 11) as f64 / (1u64 << 53) as f64
    };
    let mut cumulative = Vec::with_capacity(rows);
    let mut running = 0.0;
    for w in &weights {
        running += w;
        cumulative.push(running);
    }
    let mut a = Array2::<f64>::zeros((atoms, d_out));
    let mut b = Array2::<f64>::zeros((atoms, d_in));
    for c in 0..atoms {
        let t = if total_weight > 0.0 { cumulative.partition_point(|x| *x < uniform() * total_weight).min(rows - 1) } else { c };
        let (g, x) = (g_hat.row(t).mapv(f64::from), x_hat.row(t).mapv(f64::from));
        let (gn, xn) = (g.dot(&g).sqrt(), x.dot(&x).sqrt());
        if gn > 0.0 && xn > 0.0 {
            a.row_mut(c).assign(&(g / gn));
            b.row_mut(c).assign(&(x / xn));
        }
    }
    // Rank-one k-means.
    let mut assigned = vec![usize::MAX; rows];
    let mut iterations = 0;
    let mut previous = 0.0;
    let mut captured;
    loop {
        iterations += 1;
        // Assign: each input to the atom carrying most of its attribution.
        let mut changed = 0usize;
        captured = 0.0;
        let mut p_of = vec![0.0; rows];
        let mut q_of = vec![0.0; rows];
        for start in (0..rows).step_by(chunk) {
            let end = (start + chunk).min(rows);
            let p = widen(&g_hat, start, end).dot(&a.t());
            let q = widen(&x_hat, start, end).dot(&b.t());
            for r in 0..end - start {
                let (mut best, mut value) = (0usize, -1.0);
                for c in 0..atoms {
                    let s = p[[r, c]] * q[[r, c]];
                    if s * s > value {
                        best = c;
                        value = s * s;
                    }
                }
                if assigned[start + r] != best {
                    changed += 1;
                    assigned[start + r] = best;
                }
                captured += value;
                p_of[start + r] = p[[r, best]];
                q_of[start + r] = q[[r, best]];
            }
        }
        // Done when no input moves, or a round gains less than one input's mean energy.
        if changed == 0 || (iterations > 1 && captured - previous < total_energy / rows as f64) {
            break;
        }
        previous = captured;
        // Move: one alternating power step per atom on its inputs.
        let mut next_a = Array2::<f64>::zeros((atoms, d_out));
        for t in 0..rows {
            let weight = q_of[t] * q_of[t] * p_of[t];
            next_a.row_mut(assigned[t]).scaled_add(weight, &g_hat.row(t).mapv(f64::from));
        }
        for c in 0..atoms {
            let norm = next_a.row(c).dot(&next_a.row(c)).sqrt();
            if norm > 0.0 {
                a.row_mut(c).assign(&(&next_a.row(c) / norm));
            }
        }
        let mut next_b = Array2::<f64>::zeros((atoms, d_in));
        for t in 0..rows {
            let c = assigned[t];
            let p = a.row(c).dot(&g_hat.row(t).mapv(f64::from));
            next_b.row_mut(c).scaled_add(p * p * q_of[t], &x_hat.row(t).mapv(f64::from));
        }
        for c in 0..atoms {
            let norm = next_b.row(c).dot(&next_b.row(c)).sqrt();
            if norm > 0.0 {
                b.row_mut(c).assign(&(&next_b.row(c) / norm));
            }
        }
    }
    // Scales, then the atoms that pay for themselves, then scales again.
    let scales = |a: &Array2<f64>, b: &Array2<f64>| -> Result<Array1<f64>, String> {
        let gram = &a.dot(&a.t()) * &b.dot(&b.t());
        let rhs = Array1::from_iter((0..a.nrows()).map(|c| a.row(c).dot(&whitened.m.dot(&b.row(c)))));
        let decomposed = eigh(gram.view(), SymmetricAssembly::Mirrored, None).map_err(|e| format!("{e:?}"))?;
        let projected = decomposed.vectors.t().dot(&rhs);
        let solved = Array1::from_iter(projected.iter().zip(decomposed.values.iter()).map(|(p, l)| if *l > decomposed.band { p / l } else { 0.0 }));
        Ok(decomposed.vectors.dot(&solved))
    };
    let lambda = scales(&a, &b)?;
    let mut carried = vec![0.0; atoms];
    let mut count = vec![0.0; atoms];
    for t in 0..rows {
        let c = assigned[t];
        let s = lambda[c] * p_of_atom(&a, &g_hat, c, t) * p_of_atom(&b, &x_hat, c, t);
        carried[c] += observations * s * s / (2.0 * std::f64::consts::LN_2);
        count[c] += 1.0;
    }
    let keep: Vec<usize> = (0..atoms)
        .filter(|&c| {
            let listing = if count[c] > 0.0 { count[c] * (rows as f64 / count[c]).log2() } else { 0.0 };
            count[c] > 0.0 && carried[c] - listing > bits_per_piece
        })
        .collect();
    let (a, b) = (a.select(Axis(0), &keep), b.select(Axis(0), &keep));
    let lambda = scales(&a, &b)?;
    // Back to the program's coordinates: `u = B^{-1/2} a √|λ| sign λ`, `v = A^{-1/2} b √|λ|`.
    let mut u = Array2::<f64>::zeros((keep.len(), d_out));
    let mut v = Array2::<f64>::zeros((d_in, keep.len()));
    for c in 0..keep.len() {
        let root = lambda[c].abs().sqrt();
        let sign = lambda[c].signum();
        u.row_mut(c).assign(&(whitened.b_inverse.dot(&a.row(c)) * (root * sign)));
        v.column_mut(c).assign(&(whitened.a_inverse.dot(&b.row(c)) * root));
    }
    let fitted = (&a * &lambda.view().insert_axis(Axis(1))).t().dot(&b);
    // What the atoms leave of `M`, as Fisher-whitened singular pieces.
    let (u_left, v_left) = singular_pieces(&(&whitened.m - &fitted))?;
    let u = concatenate(Axis(0), &[u.view(), u_left.dot(&whitened.b_inverse).view()]).map_err(|e| e.to_string())?;
    let v = concatenate(Axis(1), &[v.view(), whitened.a_inverse.dot(&v_left.t()).view()]).map_err(|e| e.to_string())?;
    let library = complete(&site.w, u, v)?;
    Ok((library, DictionaryReport { seeded: atoms, kept: keep.len(), iterations, captured: captured / total_energy.max(f64::MIN_POSITIVE) }))
}

/// Unit row `c` of `directions` against sample row `t`, widened.
fn p_of_atom(directions: &Array2<f64>, samples: &Array2<f32>, c: usize, t: usize) -> f64 {
    directions.row(c).iter().zip(samples.row(t).iter()).map(|(d, s)| d * f64::from(*s)).sum()
}
