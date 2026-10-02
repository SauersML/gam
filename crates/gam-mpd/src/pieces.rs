//! Per-input pieces of one linear map (#2951): an overcomplete library of rank-1 pieces whose sum is
//! the map, fitted so that each input is explained by few of them.
//!
//! # The code
//!
//! A site `y = W x` is written as `W = Σ_c u_c v_cᵀ` (on the centred input `x − μ`, with the bias
//! `W μ` always on). On input `x` the program lists the pieces `A_x` that are on and computes
//! `Σ_{c ∈ A_x} u_c (v_c · (x − μ))`; a piece that is off is dropped. Whether a piece is on is
//! decided by the model's own computation on `x`: its contribution `u_c (v_c · (x − μ))`. The
//! code of one input is
//!
//! ```text
//! L(A_x) + n · KL / ln 2,     KL ≈ ½ ‖B^{1/2} Σ_{c ∉ A_x} u_c (v_c · (x − μ))‖²,
//! ```
//!
//! with `B` the sampled-label Fisher at the site's output (the second-order KL of dropping a
//! contribution) and `L(A_x)` the listing code of the set: each piece costs
//! `log₂(T / count_c)` bits at its firing frequency over the fit inputs (smoothed by a half
//! count, so an unused piece has a finite cost), and a set of `k` is one of its `k!` orders.
//!
//! # The fit
//!
//! In whitened coordinates (`ξ = A^{-1/2}(x − μ)` with `A` the input covariance, the map
//! `M = B^{1/2} W A^{1/2}`, the target `y_x = M ξ_x`) a piece is `(ũ_c, ṽ_c)` contributing
//! `a_c(x) ũ_c`, `a_c = ṽ_c · ξ`. The fit alternates two exact steps until an iteration saves
//! less than one bit per input:
//!
//! * **selection** — each input's set grows greedily by the piece whose KL bits saved,
//!   `n (a_c s_c − ½ a_c² ‖ũ_c‖²)/ln 2` with `s_c = r · ũ_c` against the current residual `r`,
//!   exceed its listing cost; it stops when none does (the derived stopping rule);
//! * **update** — every piece is refitted on the inputs that list it, against their residual
//!   with its own contribution added back: `ũ_c` by least squares, and `ṽ_c` along the
//!   residual-weighted mean of those inputs, scaled by least squares.
//!
//! The library starts from `K = 2 min(d_in, d_out)` prototypes seeded by k-means++ over the
//! inputs (each reproduces its seed input exactly), and ends exact: the residual
//! `W − Σ u_c v_cᵀ` is appended as the rank-1 pieces of its singular value decomposition, so all
//! pieces on is the map itself.

use super::dense::{eigh, svd};
use gam_linalg::roundoff::{SymmetricAssembly, accumulation_growth};
use ndarray::{Array1, Array2, Axis, concatenate};

/// One site's data: the map, the input's second moment and mean, the output Fisher, and inputs.
pub struct Site {
    /// `d_out × d_in`.
    pub w: Array2<f64>,
    /// `E[x xᵀ]` (`d_in × d_in`).
    pub second_moment: Array2<f64>,
    pub mean: Array1<f64>,
    /// `E[g gᵀ]` at the output (`d_out × d_out`).
    pub fisher: Array2<f64>,
}

/// A fitted library in the program's coordinates: `v` is `d_in × C`, `u` is `C × d_out`, and
/// `v u = Wᵀ` on the centred input.
pub struct Library {
    pub v: Array2<f64>,
    pub u: Array2<f64>,
    /// Pieces appended to make the library exact.
    pub exactness_pieces: usize,
    /// Each piece's listing bits at its firing frequency on the fit inputs.
    pub costs: Vec<f64>,
    /// Per fit iteration, the mean code per input (listing bits, KL bits).
    pub history: Vec<(f64, f64)>,
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
pub struct Whitened {
    /// `A^{1/2}`, `A^{-1/2}`, `B^{1/2}`, `B^{-1/2}`.
    a_half: Array2<f64>,
    a_inverse: Array2<f64>,
    b_half: Array2<f64>,
    b_inverse: Array2<f64>,
    /// `M = B^{1/2} W A^{1/2}`.
    m: Array2<f64>,
    mean: Array1<f64>,
}

impl Whitened {
    pub fn new(site: &Site) -> Result<Self, String> {
        let covariance = &site.second_moment - &outer(&site.mean, &site.mean);
        let (a_half, a_inverse) = roots(&covariance)?;
        let (b_half, b_inverse) = roots(&site.fisher)?;
        let m = b_half.dot(&site.w).dot(&a_half);
        Ok(Self { a_half, a_inverse, b_half, b_inverse, m, mean: site.mean.clone() })
    }

    /// `ξ = A^{-1/2}(x − μ)` for inputs `x` (rows).
    pub fn inputs(&self, x: &Array2<f64>) -> Array2<f64> {
        (x - &self.mean).dot(&self.a_inverse)
    }
}

fn outer(a: &Array1<f64>, b: &Array1<f64>) -> Array2<f64> {
    let (n, m) = (a.len(), b.len());
    Array2::from_shape_fn((n, m), |(i, j)| a[i] * b[j])
}

/// A deterministic generator for seeding.
struct XorShift(u64);

impl XorShift {
    fn next(&mut self) -> f64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        (self.0 >> 11) as f64 / (1u64 << 53) as f64
    }
}

/// Each input's set (module note, "selection"): greedy, with `costs[c]` the listing bits of piece
/// `c`, at `observations` per input. Returns the sets and the final residuals `y − Σ_A a ũ`.
fn select(
    targets: &Array2<f64>,
    activations: &Array2<f64>,
    u: &Array2<f64>,
    gram: &Array2<f64>,
    costs: &[f64],
    observations: f64,
) -> (Vec<Vec<usize>>, Array2<f64>) {
    let k = u.nrows();
    let scale = observations / std::f64::consts::LN_2;
    let projections = targets.dot(&u.t());
    let mut sets = Vec::with_capacity(targets.nrows());
    let mut residuals = targets.clone();
    for t in 0..targets.nrows() {
        let a = activations.row(t);
        let mut s = projections.row(t).to_owned();
        let mut chosen: Vec<usize> = Vec::new();
        let mut taken = vec![false; k];
        loop {
            let credit = ((chosen.len() + 1) as f64).log2();
            let mut best = (0.0_f64, usize::MAX);
            for c in 0..k {
                if taken[c] {
                    continue;
                }
                let saved = scale * (a[c] * s[c] - 0.5 * a[c] * a[c] * gram[[c, c]]);
                let net = saved - (costs[c] - credit).max(0.0);
                if net > best.0 {
                    best = (net, c);
                }
            }
            if best.1 == usize::MAX {
                break;
            }
            let c = best.1;
            taken[c] = true;
            chosen.push(c);
            let ac = a[c];
            for d in 0..k {
                s[d] -= ac * gram[[c, d]];
            }
            residuals.row_mut(t).scaled_add(-ac, &u.row(c));
        }
        sets.push(chosen);
    }
    (sets, residuals)
}

/// The listing bits and KL bits per input of a selection.
fn code(sets: &[Vec<usize>], residuals: &Array2<f64>, costs: &[f64], observations: f64) -> (f64, f64) {
    let n = sets.len() as f64;
    let mut listing = 0.0;
    for set in sets {
        listing += set.iter().map(|&c| costs[c]).sum::<f64>();
        listing -= (1..=set.len()).map(|k| (k as f64).log2()).sum::<f64>();
    }
    let kl: f64 = residuals.iter().map(|v| v * v).sum::<f64>() * 0.5;
    (listing / n, observations * kl / std::f64::consts::LN_2 / n)
}

/// Each piece's firing count on the inputs, plus half a count.
fn counts(sets: &[Vec<usize>], k: usize) -> Vec<f64> {
    let mut counts = vec![0.5_f64; k];
    for set in sets {
        for &c in set {
            counts[c] += 1.0;
        }
    }
    counts
}

fn costs_of(counts: &[f64]) -> Vec<f64> {
    let total: f64 = counts.iter().sum();
    counts.iter().map(|c| (total / c).log2()).collect()
}

/// Fit a library to `site` on the inputs `x` (rows), at `observations` per input (module note).
pub fn fit(site: &Site, x: &Array2<f64>, observations: f64) -> Result<(Library, Whitened), String> {
    let whitened = Whitened::new(site)?;
    let xi = whitened.inputs(x);
    let targets = xi.dot(&whitened.m.t());
    let (d_out, d_in) = site.w.dim();
    let k = 2 * d_in.min(d_out);
    // k-means++ seeds over the inputs, by their targets' energy.
    let mut rng = XorShift(0x9E37_79B9_7F4A_7C15);
    let energy: Vec<f64> = targets.outer_iter().map(|y| y.dot(&y)).collect();
    let mut distance = energy.clone();
    let mut seeds: Vec<usize> = Vec::with_capacity(k);
    let mut u = Array2::<f64>::zeros((k, d_out));
    let mut v = Array2::<f64>::zeros((k, d_in));
    for c in 0..k {
        let total: f64 = distance.iter().sum();
        if !(total > 0.0) {
            break;
        }
        let mut pick = rng.next() * total;
        let mut chosen = distance.len() - 1;
        for (t, d) in distance.iter().enumerate() {
            if pick < *d {
                chosen = t;
                break;
            }
            pick -= d;
        }
        seeds.push(chosen);
        let norm = xi.row(chosen).dot(&xi.row(chosen));
        if norm > 0.0 {
            v.row_mut(c).assign(&(&xi.row(chosen) / norm));
            u.row_mut(c).assign(&targets.row(chosen));
        }
        // Distance to the nearest seed's prediction of each input.
        let a = xi.dot(&v.row(c));
        for t in 0..targets.nrows() {
            let predicted = &u.row(c) * a[t];
            let err = &targets.row(t) - &predicted;
            distance[t] = distance[t].min(err.dot(&err));
        }
    }
    let mut history = Vec::new();
    let mut fired = vec![0.5_f64; k];
    let mut costs = costs_of(&fired);
    let mut previous = f64::INFINITY;
    loop {
        let gram = u.dot(&u.t());
        let activations = xi.dot(&v.t());
        let (sets, residuals) = select(&targets, &activations, &u, &gram, &costs, observations);
        let (listing, kl) = code(&sets, &residuals, &costs, observations);
        history.push((listing, kl));
        let total = listing + kl;
        if previous - total < 1.0 {
            break;
        }
        previous = total;
        // Update every piece on the inputs that list it.
        let mut members: Vec<Vec<usize>> = vec![Vec::new(); k];
        for (t, set) in sets.iter().enumerate() {
            for &c in set {
                members[c].push(t);
            }
        }
        let (old_u, old_v) = (u.clone(), v.clone());
        for c in 0..k {
            if members[c].is_empty() {
                continue;
            }
            // Residual with the piece's own contribution added back, and its activation.
            let (mut numerator, mut denominator) = (Array1::<f64>::zeros(d_out), 0.0);
            for &t in &members[c] {
                let a = activations[[t, c]];
                let e = &residuals.row(t) + &(&old_u.row(c) * a);
                numerator.scaled_add(a, &e);
                denominator += a * a;
            }
            if !(denominator > 0.0) {
                continue;
            }
            let new_u = numerator / denominator;
            let norm = new_u.dot(&new_u);
            if !(norm > 0.0) {
                continue;
            }
            // The activation each member asks for, and the direction that gives it.
            let mut direction = Array1::<f64>::zeros(d_in);
            let mut wanted = Vec::with_capacity(members[c].len());
            for &t in &members[c] {
                let a = activations[[t, c]];
                let e = &residuals.row(t) + &(&old_u.row(c) * a);
                let beta = new_u.dot(&e) / norm;
                direction.scaled_add(beta, &xi.row(t));
                wanted.push(beta);
            }
            let projected: Vec<f64> = members[c].iter().map(|&t| xi.row(t).dot(&direction)).collect();
            let (num, den) = projected.iter().zip(&wanted).fold((0.0, 0.0), |(n, d), (p, w)| (n + p * w, d + p * p));
            if den > 0.0 {
                v.row_mut(c).assign(&(direction * (num / den)));
            } else {
                v.row_mut(c).assign(&old_v.row(c));
            }
            u.row_mut(c).assign(&new_u);
        }
        fired = counts(&sets, k);
        costs = costs_of(&fired);
    }
    // Exactness: the residual map's singular pieces appended, first in the whitened metric, then
    // whatever the supports of `A` and `B` leave, in the program's own coordinates.
    let residual = &whitened.m - &u.t().dot(&v);
    let (u_extra, v_extra) = singular_pieces(&residual)?;
    let u_all = concatenate(Axis(0), &[u.view(), u_extra.view()]).map_err(|e| e.to_string())?;
    let v_all = concatenate(Axis(0), &[v.view(), v_extra.view()]).map_err(|e| e.to_string())?;
    // Back to the program's coordinates: u_c = B^{-1/2} ũ_c, v_c = A^{-1/2} ṽ_c.
    let mut u_program = u_all.dot(&whitened.b_inverse);
    let mut v_program = whitened.a_inverse.dot(&v_all.t());
    let left = &site.w - &u_program.t().dot(&v_program.t());
    // What the supports leave is kept only beyond the product's own rounding band,
    // `γ_C |u|ᵀ|v|ᵀ` entrywise; its Frobenius norm bounds the band's singular values.
    let band = u_program.mapv(f64::abs).t().dot(&v_program.mapv(f64::abs).t()) * accumulation_growth(u_program.nrows());
    let within = left.iter().zip(band.iter()).all(|(r, b)| r.abs() <= *b);
    let band_norm = band.iter().map(|b| b * b).sum::<f64>().sqrt();
    let (u_left, v_left) = if within { (Array2::zeros((0, d_out)), Array2::zeros((0, d_in))) } else { singular_pieces_above(&left, band_norm)? };
    if u_left.nrows() > 0 {
        u_program = concatenate(Axis(0), &[u_program.view(), u_left.view()]).map_err(|e| e.to_string())?;
        v_program = concatenate(Axis(1), &[v_program.view(), v_left.t()]).map_err(|e| e.to_string())?;
    }
    let extra = u_extra.nrows() + u_left.nrows();
    // The appended pieces fired on no fit input: half a count each.
    fired.extend(std::iter::repeat_n(0.5, extra));
    let costs = costs_of(&fired);
    Ok((Library { v: v_program, u: u_program, exactness_pieces: extra, costs, history }, whitened))
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

/// The sets of `library`'s pieces on inputs `x` (rows) at `observations` per input, with the
/// listing costs of the fit's own frequencies.
pub fn sets(library: &Library, whitened: &Whitened, x: &Array2<f64>, observations: f64) -> Vec<Vec<usize>> {
    // ũ_c = B^{1/2} u_c and ṽ_c = A^{1/2} v_c, on the supports.
    let u = library.u.dot(&whitened.b_half);
    let v = library.v.t().dot(&whitened.a_half);
    let gram = u.dot(&u.t());
    let xi = whitened.inputs(x);
    select(&xi.dot(&whitened.m.t()), &xi.dot(&v.t()), &u, &gram, &library.costs, observations).0
}

impl Library {
    /// The largest entry of `Wᵀ − v u`, against the largest entry of `W`.
    pub fn exactness(&self, w: &Array2<f64>) -> f64 {
        let error = &w.t() - &self.v.dot(&self.u);
        let largest = w.iter().fold(0.0_f64, |m, x| m.max(x.abs()));
        error.iter().fold(0.0_f64, |m, x| m.max(x.abs())) / largest.max(f64::MIN_POSITIVE)
    }
}
