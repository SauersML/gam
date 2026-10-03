//! A site's library fitted on its own inputs (#2951).
//!
//! The masked program (`super::masked`) codes each input by the subcomponents that run on it,
//! `Σ_{c on} bits(c) + n KL / ln 2`, under the box claim: an off subcomponent's gate may be anywhere
//! in `[0, 1]`. Its libraries can come from VPD, from the map's own singular pieces, or from this
//! module: a library fitted on a site's inputs to that one total, with the KL taken to second
//! order at the site.
//!
//! # The site's code
//!
//! A site `y = W x` with library `y ≈ Σ_c u_c (v_c · x)` and per-input on-sets `m_t` leaves on input
//! `t` the residual `r_t = y_t − Σ_{c on} u_c a_tc` (`a_tc = v_c · x_t`, the read uncentred as the
//! masked program reads it), and the off subcomponents write `S_t = Σ_{c off} u_c a_tc` at full
//! gate. Input `t`'s KL is `½ eᵀ F_t e` in its own output Fisher `F_t`, taken as the site's mean
//! Fisher `F` scaled by the input's sensitivity `s_t = tr F_t / tr F` (both measured from
//! sampled-label gradients of the model's own output). The box claim's error is its worst point;
//! two of them are closed forms in `F`: the corner (every off gate at 0), `‖r‖²`, and the
//! expectation over every off gate uniform, `‖r − ½S‖² + 1/12 Σ_{c off} a_c² u_cᵀF u_c`. Input `t`
//! is charged the larger:
//!
//! ```text
//! code_t = Σ_{c on} bits(c) + n s_t / (2 ln 2) · max(‖r_t‖²_F, ‖r_t − ½S_t‖²_F + 1/12 Σ_{c off} a_tc² u_cᵀF u_c).
//! ```
//!
//! `bits(c)` is the subcomponent's description ([`super::blocks::Describe`]), paid on every input
//! it runs on. Nothing else enters: no library is amortised, and a fit that leans on its off
//! subcomponents at half gate pays for it at the corner.
//!
//! # The fit
//!
//! Alternating exact steps of that one total:
//!
//! * **Sets.** Each input's on-set by single flips, a flip kept when it lowers the input's code
//!   (both points tracked exactly through `K = U F Uᵀ`), swept until no flip pays.
//! * **Writes.** With the sets and each input's worst point fixed the code is a quadratic in `U`
//!   whose metric `F` factors out, `tr F (UᵀQU − 2UᵀR)`, with `Q = Σ_t s_t z̃_t z̃_tᵀ + 1/12
//!   diag(Σ_t s_t ν_t ⊙ a_t²)`, `R = Σ_t s_t z̃_t y_tᵀ`, `z̃ = μ ⊙ a`, `μ` one on, one half off at an
//!   input charged its expectation (zero at one charged its corner), `ν` its off indicator there.
//!   Every subcomponent on is the map exactly on the reads' span (all but `10⁻⁶` of their second
//!   moment, as the masked program restricts its reads), `Uᵀ V E√Λ = W E√Λ`, so an input the fit
//!   never saw is still carried by the off subcomponents it leaves. With `V E√Λ = P S Gᵀ` (full
//!   `P`) every such `U` is `U₀ + N Z`, `U₀ = P₁ S⁻¹ Gᵀ (W E√Λ)ᵀ` and `N` the columns of `P` past
//!   its rank, and the code fixes `Z` by `(NᵀQN) Z = Nᵀ(R − Q U₀)`; the step toward that minimiser
//!   is halved until the code (each input at its worse point) falls.
//! * **Reads.** With `U` and the sets fixed, each input's two points are quadratics in `V`. The
//!   step is the charged points' preconditioned negative gradient (left by each subcomponent's own
//!   curvature, right by the inputs' sensitivity-weighted second moment) restricted to the moves
//!   that keep the map, `Uᵀ D = 0`, its length the one of least code among a geometric ladder
//!   around the quadratic's minimiser: along a line every input's scalars are quadratics in the
//!   length, so one pass measures the whole ladder.
//! * **Reseeding.** A subcomponent that runs on no input is replaced by one reading the input of
//!   largest error (its direction in the reads' inverse second moment), the writes solved again;
//!   kept when the sets selected with it code the inputs in fewer bits.
//!
//! The fit stops when a round (sets, writes, reads) saves less than one bit per input, or at
//! `rounds`. Every product over inputs runs in single precision (the fit only proposes a library;
//! the masked program's exact forward codes it), and pseudo-inverses drop eigenvalues within the
//! single-precision band of their sums, `√T 2⁻²⁴` of the largest.

use super::blocks::Describe;
use super::dense::{eigh, svd};
use super::derivatives::vjp;
use super::device::proposing;
use super::masked::{Library, Site, Target, read_values, sampled_label_cotangent};
use super::operator_program::{FamilyInputs, OperatorProgram};
use faer::linalg::matmul::matmul;
use faer::{Accum, MatMut, MatRef};
use gam_linalg::faer_ndarray::{fast_atb, matmul_parallelism};
use gam_linalg::roundoff::SymmetricAssembly;
use ndarray::{Array1, Array2, ArrayView2, Axis, s};
use rayon::prelude::*;
use std::f64::consts::LN_2;

/// What a site's fit reads of the model: its inputs' reads and sensitivities, its mean written
/// Fisher and the reads' second moment.
pub struct Samples {
    /// The reads, inputs × d_in (single precision).
    pub reads: Array2<f32>,
    /// Each input's `tr F_t / tr F` (module note).
    pub sensitivity: Array1<f64>,
    /// `F = E[g gᵀ]` at the written value (d_out × d_out).
    pub fisher: Array2<f64>,
    /// `E[x xᵀ]` of the uncentred reads (d_in × d_in).
    pub second_moment: Array2<f64>,
}

/// Every site's [`Samples`] on the native program over `batches`, `draws` sampled-label reverse
/// passes per batch (labels drawn from the program's own output).
pub fn samples(program: &OperatorProgram, sites: &[Site], batches: impl IntoIterator<Item = FamilyInputs>, draws: usize, seed: u64) -> Result<Vec<Samples>, String> {
    if draws == 0 {
        return Err("samples need at least one draw".to_string());
    }
    let mut reads: Vec<Vec<Array2<f32>>> = vec![Vec::new(); sites.len()];
    let mut norms: Vec<Vec<f64>> = vec![Vec::new(); sites.len()];
    let mut fishers: Vec<Option<Array2<f64>>> = vec![None; sites.len()];
    let mut moments: Vec<Option<Array2<f64>>> = vec![None; sites.len()];
    let no_rows = Target { logits: Array2::zeros((0, 0)), scored: None };
    let mut drawn = 0u64;
    for inputs in batches {
        let trace = program.execute(&inputs, false).map_err(|e| e.to_string())?;
        let first = norms[0].len();
        for (k, site) in sites.iter().enumerate() {
            let x = read_values(&trace, site)?;
            let outer = fast_atb(&x, &x);
            moments[k] = Some(match moments[k].take() {
                Some(m) => m + outer,
                None => outer,
            });
            reads[k].push(x.mapv(|v| v as f32));
            norms[k].extend(std::iter::repeat_n(0.0, inputs.rows));
        }
        for _ in 0..draws {
            drawn += 1;
            let cotangent = sampled_label_cotangent(&trace.values[program.output], &no_rows, seed.wrapping_add(drawn.wrapping_mul(0x9E37_79B9)));
            let back = proposing(|| vjp(program, &inputs, &trace, cotangent)).map_err(|e| e.to_string())?;
            for (k, site) in sites.iter().enumerate() {
                let written: Vec<Array2<f64>> =
                    site.writes.iter().map(|n| back[*n].clone().unwrap_or_else(|| Array2::zeros(trace.values[*n].dim()))).collect();
                let views: Vec<_> = written.iter().map(|w| w.view()).collect();
                let g = ndarray::concatenate(Axis(1), &views).map_err(|e| e.to_string())?;
                for (r, row) in g.outer_iter().enumerate() {
                    norms[k][first + r] += row.dot(&row) / draws as f64;
                }
                let outer = proposing(|| super::device::product_atb(&g, &g)).map_err(|e| e.to_string())?;
                fishers[k] = Some(match fishers[k].take() {
                    Some(f) => f + outer,
                    None => outer,
                });
            }
        }
    }
    let rows = norms.first().map_or(0, Vec::len);
    if rows == 0 {
        return Err("samples need inputs".to_string());
    }
    reads
        .into_iter()
        .zip(norms)
        .zip(fishers.into_iter().zip(moments))
        .map(|((parts, norms), (fisher, moment))| {
            let views: Vec<_> = parts.iter().map(|p| p.view()).collect();
            let reads = ndarray::concatenate(Axis(0), &views).map_err(|e| e.to_string())?;
            let fisher = fisher.ok_or("no Fisher")? / (rows * draws) as f64;
            let mean_norm = fisher.diag().sum();
            let sensitivity = Array1::from_iter(norms.iter().map(|n| if mean_norm > 0.0 { n / mean_norm } else { 1.0 }));
            Ok(Samples { reads, sensitivity, fisher, second_moment: moment.ok_or("no moment")? / rows as f64 })
        })
        .collect()
}

/// One round of [`fit`]: its code per input after the sets (description and error bits), the
/// mean on-set size, the subcomponents reseeded, and the rung of the reads' chosen length (0: none).
#[derive(Clone, Debug)]
pub struct Round {
    pub round: usize,
    pub code: f64,
    pub description: f64,
    pub error: f64,
    pub l0: f64,
    pub corner_share: f64,
    pub reseeded: usize,
    pub read_steps: usize,
}

/// The share of the reads' second moment left out of their span (the masked driver's).
const LEFT_OUT: f64 = 1e-6;

/// Rows per chunk of the products over inputs.
const CHUNK: usize = 2048;

/// `out (+)= α op(a) op(b)` in single precision; every operand row-major and contiguous.
fn gemm(out: &mut Array2<f32>, add: bool, a: ArrayView2<'_, f32>, ta: bool, b: ArrayView2<'_, f32>, tb: bool, alpha: f32) {
    let (ar, ac) = a.dim();
    let (br, bc) = b.dim();
    let lhs = MatRef::from_row_major_slice(a.to_slice().expect("a contiguous operand"), ar, ac);
    let rhs = MatRef::from_row_major_slice(b.to_slice().expect("a contiguous operand"), br, bc);
    let lhs = if ta { lhs.transpose() } else { lhs };
    let rhs = if tb { rhs.transpose() } else { rhs };
    let (m, n) = out.dim();
    let par = matmul_parallelism(m, n, lhs.ncols());
    let dst = MatMut::from_row_major_slice_mut(out.as_slice_mut().expect("a contiguous result"), m, n);
    matmul(dst, if add { Accum::Add } else { Accum::Replace }, lhs, rhs, alpha, par);
}

fn product(a: ArrayView2<'_, f32>, ta: bool, b: ArrayView2<'_, f32>, tb: bool) -> Array2<f32> {
    let m = if ta { a.ncols() } else { a.nrows() };
    let n = if tb { b.nrows() } else { b.ncols() };
    let mut out = Array2::zeros((m, n));
    gemm(&mut out, false, a, ta, b, tb, 1.0);
    out
}

fn single(m: &Array2<f64>) -> Array2<f32> {
    m.mapv(|v| v as f32)
}

/// The pseudo-inverse of a symmetric positive semidefinite matrix summed in single precision over
/// `terms` inputs: eigenvalues within `√terms · 2⁻²⁴` of `scale` (its largest when `None`, else the
/// matrix it was projected from) are dropped; with `terms` zero, a matrix formed in float64, only
/// its decomposition's band.
fn pseudo_inverse(m: &Array2<f64>, terms: usize) -> Result<Array2<f64>, String> {
    pseudo_inverse_of(m, terms, None)
}

fn pseudo_inverse_of(m: &Array2<f64>, terms: usize, scale: Option<f64>) -> Result<Array2<f64>, String> {
    let mut sym = m.clone();
    let n = sym.nrows();
    for i in 0..n {
        for j in (i + 1)..n {
            let v = 0.5 * (sym[[i, j]] + sym[[j, i]]);
            sym[[i, j]] = v;
            sym[[j, i]] = v;
        }
    }
    let d = eigh(sym.view(), SymmetricAssembly::Mirrored, None).map_err(|e| format!("{e:?}"))?;
    let largest = scale.unwrap_or_else(|| d.values.iter().fold(0.0_f64, |m, l| m.max(*l)));
    let floor = (largest * (terms as f64).sqrt() * f64::from(f32::EPSILON) / 2.0).max(d.band);
    let mut scaled = d.vectors.clone();
    for (k, l) in d.values.iter().enumerate() {
        let inverse = if *l > floor { 1.0 / l } else { 0.0 };
        scaled.column_mut(k).mapv_inplace(|x| x * inverse);
    }
    Ok(scaled.dot(&d.vectors.t()))
}

/// The per-input state of a fit: the sets and which point each input is charged.
struct Fitting<'a> {
    x: &'a Array2<f32>,
    /// The site's map, `d_out × d_in`: every subcomponent on is exactly it on the reads' span.
    w: Array2<f64>,
    /// The reads' span (all but [`LEFT_OUT`] of their second moment), each direction scaled by
    /// its root second moment: `d_in × r`.
    span: Array2<f64>,
    /// `E Λ⁻¹ Eᵀ` on that span: a read seeded from an input is its direction in it.
    seeding: Array2<f64>,
    /// `y = x Wᵀ`, inputs × d_out.
    y: Array2<f32>,
    /// `yᵀ F y` per input.
    yfy: Vec<f64>,
    s: &'a Array1<f64>,
    fisher: &'a Array2<f64>,
    /// `n / (2 ln 2)`.
    scale: f64,
    pieces: usize,
    /// Inputs × pieces, 1 where on.
    masks: Vec<u8>,
    /// Whether each input is charged its corner (else the expectation).
    corner: Vec<bool>,
}

/// One input's code under its sets (module note), from its scalars.
fn worst(rr: f64, rs: f64, ss: f64, off: f64) -> (f64, bool) {
    let expected = rr - rs + 0.25 * ss + off / 12.0;
    if rr >= expected { (rr, true) } else { (expected, false) }
}

impl Fitting<'_> {
    fn rows(&self) -> usize {
        self.x.nrows()
    }

    /// Each input's gates on its own write and its weights: `μ` (1 on, ½ off where the expectation
    /// is charged, 0 off at a corner) and `ν` (off where the expectation is charged).
    fn gates(&self, t: usize, c: usize) -> (f32, f32) {
        let on = self.masks[t * self.pieces + c] == 1;
        match (on, self.corner[t]) {
            (true, _) => (1.0, 0.0),
            (false, true) => (0.0, 0.0),
            (false, false) => (0.5, 1.0),
        }
    }

    /// The code of `(v, u)`: with `flip`, every input's single flips swept first until none lowers
    /// its code; with `keep`, each input's worst point is recorded (both leave the state as it was
    /// otherwise, so a trial library is measured by neither). Returns the total description and
    /// error bits and each input's error bits.
    fn code(&mut self, v: &Array2<f64>, u: &Array2<f64>, bits: &Array1<f64>, flip: bool, keep: bool) -> (f64, f64, Vec<f64>) {
        let c_total = self.pieces;
        let uf = u.dot(self.fisher);
        let k = uf.dot(&u.t());
        let k32 = single(&k);
        let (v32, uf32) = (single(v), single(&uf));
        let diag: Vec<f64> = (0..c_total).map(|c| k[[c, c]]).collect();
        let mut description = 0.0;
        let mut error = 0.0;
        let mut errors = vec![0.0; self.rows()];
        for start in (0..self.rows()).step_by(CHUNK) {
            let end = (start + CHUNK).min(self.rows());
            let a = product(self.x.slice(s![start..end, ..]), false, v32.view(), true);
            let g0 = product(self.y.slice(s![start..end, ..]), false, uf32.view(), true);
            let mut z_on = a.clone();
            let mut z_off = a.clone();
            for r in 0..end - start {
                for c in 0..c_total {
                    if self.masks[(start + r) * c_total + c] == 1 {
                        z_off[[r, c]] = 0.0;
                    } else {
                        z_on[[r, c]] = 0.0;
                    }
                }
            }
            let p_all = product(z_on.view(), false, k32.view(), false);
            let q_all = product(z_off.view(), false, k32.view(), false);
            let masks = &mut self.masks[start * c_total..end * c_total];
            let corner = &mut self.corner[start..end];
            let (s, yfy, scale) = (&self.s, &self.yfy, self.scale);
            let results: Vec<(f64, f64)> = masks
                .par_chunks_mut(c_total)
                .zip(corner.par_iter_mut())
                .enumerate()
                .map(|(r, (m, corner))| {
                    let t = start + r;
                    let a: Vec<f64> = a.row(r).iter().map(|v| f64::from(*v)).collect();
                    let g0: Vec<f64> = g0.row(r).iter().map(|v| f64::from(*v)).collect();
                    let mut p: Vec<f64> = p_all.row(r).iter().map(|v| f64::from(*v)).collect();
                    let mut q: Vec<f64> = q_all.row(r).iter().map(|v| f64::from(*v)).collect();
                    let (mut zg_on, mut zg_off, mut zp_on, mut zq_on, mut zq_off, mut off) = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
                    for c in 0..c_total {
                        if m[c] == 1 {
                            zg_on += a[c] * g0[c];
                            zp_on += a[c] * p[c];
                            zq_on += a[c] * q[c];
                        } else {
                            zg_off += a[c] * g0[c];
                            zq_off += a[c] * q[c];
                            off += a[c] * a[c] * diag[c];
                        }
                    }
                    // ‖r‖², r·S, ‖S‖² and the off subcomponents' own energy, tracked exactly.
                    let mut rr = yfy[t] - 2.0 * zg_on + zp_on;
                    let mut rs = zg_off - zq_on;
                    let mut ss = zq_off;
                    let weight = scale * s[t];
                    let mut current = worst(rr, rs, ss, off).0;
                    for _ in 0..if flip { c_total } else { 0 } {
                        let mut flipped = false;
                        for c in 0..c_total {
                            let sigma = if m[c] == 1 { -1.0 } else { 1.0 };
                            let (ac, kc) = (a[c], diag[c]);
                            let sa = sigma * ac;
                            let d_rr = -2.0 * sa * g0[c] + 2.0 * sa * p[c] + ac * ac * kc;
                            let d_rs = -sa * g0[c] - sa * q[c] + sa * p[c] + ac * ac * kc;
                            let d_ss = -2.0 * sa * q[c] + ac * ac * kc;
                            let d_off = -sigma * ac * ac * kc;
                            let next = worst(rr + d_rr, rs + d_rs, ss + d_ss, off + d_off).0;
                            if sigma * bits[c] + weight * (next - current) < 0.0 {
                                rr += d_rr;
                                rs += d_rs;
                                ss += d_ss;
                                off += d_off;
                                current = next;
                                let row = k.row(c);
                                for (j, kj) in row.iter().enumerate() {
                                    p[j] += sa * kj;
                                    q[j] -= sa * kj;
                                }
                                m[c] = u8::from(sigma > 0.0);
                                flipped = true;
                            }
                        }
                        if !flipped {
                            break;
                        }
                    }
                    if keep {
                        *corner = worst(rr, rs, ss, off).1;
                    }
                    let listed: f64 = (0..c_total).filter(|c| m[*c] == 1).map(|c| bits[c]).sum();
                    (listed, weight * current)
                })
                .collect();
            for (r, (listed, err)) in results.into_iter().enumerate() {
                description += listed;
                error += err;
                errors[start + r] = err;
            }
        }
        (description, error, errors)
    }

    /// The writes in closed form (module note): `U = Q⁺ R`.
    fn writes(&self, v: &Array2<f64>) -> Result<Array2<f64>, String> {
        let c_total = self.pieces;
        let v32 = single(v);
        let mut q32 = Array2::<f32>::zeros((c_total, c_total));
        let mut r32 = Array2::<f32>::zeros((c_total, self.y.ncols()));
        let mut own = vec![0.0f64; c_total];
        for start in (0..self.rows()).step_by(CHUNK) {
            let end = (start + CHUNK).min(self.rows());
            let a = product(self.x.slice(s![start..end, ..]), false, v32.view(), true);
            let mut z = a.clone();
            let mut zs = a.clone();
            for r in 0..end - start {
                let t = start + r;
                let st = self.s[t] as f32;
                for c in 0..c_total {
                    let (mu, nu) = self.gates(t, c);
                    z[[r, c]] *= mu;
                    zs[[r, c]] *= mu * st;
                    if nu > 0.0 {
                        own[c] += self.s[t] * f64::from(a[[r, c]]) * f64::from(a[[r, c]]);
                    }
                }
            }
            gemm(&mut q32, true, zs.view(), true, z.view(), false, 1.0);
            gemm(&mut r32, true, zs.view(), true, self.y.slice(s![start..end, ..]), false, 1.0);
        }
        let mut q = q32.mapv(f64::from);
        for c in 0..c_total {
            q[[c, c]] += own[c] / 12.0;
        }
        // The minimiser under `Uᵀ V = W`: with `V = P S Gᵀ` (full `P`), every exact `U` is
        // `U₀ + N Z`, `U₀ = P₁ S⁻¹ Gᵀ Wᵀ` and `N` the columns of `P` past `V`'s rank; the code then
        // fixes `Z` by `(NᵀQN) Z = Nᵀ(R − Q U₀)`.
        // On the reads' span (`x = E√Λ ξ`) the constraint is `Uᵀ (V E√Λ) = W E√Λ`.
        let decomposed = svd(v.dot(&self.span).view(), true).map_err(|e| format!("{e:?}"))?;
        let rank = decomposed.singular_values.iter().filter(|s| **s > decomposed.band).count();
        let p1 = decomposed.u.slice(s![.., ..rank]);
        let inverse = Array1::from_iter(decomposed.singular_values.iter().take(rank).map(|s| 1.0 / s));
        let particular = (&p1 * &inverse).dot(&decomposed.vt.slice(s![..rank, ..])).dot(&self.w.dot(&self.span).t());
        let null = decomposed.u.slice(s![.., rank..]);
        let rhs = r32.mapv(f64::from) - q.dot(&particular);
        // `NᵀQN` is resolved only as far as `Q` itself is: its band is `Q`'s (bounded by its trace).
        let z = pseudo_inverse_of(&null.t().dot(&q).dot(&null), self.rows(), Some(q.diag().sum()))?.dot(&null.t().dot(&rhs));
        Ok(particular + null.dot(&z))
    }

    /// Per input, `μ` and `ν ⊙ κ` over a chunk.
    fn chunk_gates(&self, start: usize, end: usize, kappa: &[f64]) -> (Array2<f32>, Array2<f32>) {
        let c_total = self.pieces;
        let mut mu = Array2::<f32>::zeros((end - start, c_total));
        let mut nk = Array2::<f32>::zeros((end - start, c_total));
        for r in 0..end - start {
            for c in 0..c_total {
                let (m, n) = self.gates(start + r, c);
                mu[[r, c]] = m;
                nk[[r, c]] = n * kappa[c] as f32;
            }
        }
        (mu, nk)
    }

    /// The gradient of the code (module note) in `V` at `v`, `U` fixed.
    fn read_gradient(&self, v: &Array2<f64>, u: &Array2<f64>, k32: &Array2<f32>, kappa: &[f64]) -> Array2<f64> {
        let v32 = single(v);
        let uf32 = single(&u.dot(self.fisher));
        let mut gradient = Array2::<f32>::zeros(v.dim());
        for start in (0..self.rows()).step_by(CHUNK) {
            let end = (start + CHUNK).min(self.rows());
            let a = product(self.x.slice(s![start..end, ..]), false, v32.view(), true);
            let g0 = product(self.y.slice(s![start..end, ..]), false, uf32.view(), true);
            let (mu, nk) = self.chunk_gates(start, end, kappa);
            let z = &a * &mu;
            let kz = product(z.view(), false, k32.view(), false);
            // With h = U F e = g0 − K z, the gradient in a is w (−2 μ ⊙ h + ν κ a / 6).
            let mut ga = Array2::<f32>::zeros(a.dim());
            for r in 0..end - start {
                let w = self.scale * self.s[start + r];
                for c in 0..self.pieces {
                    let h = f64::from(g0[[r, c]]) - f64::from(kz[[r, c]]);
                    let own = f64::from(nk[[r, c]]) * f64::from(a[[r, c]]) / 6.0;
                    ga[[r, c]] = (w * (-2.0 * f64::from(mu[[r, c]]) * h + own)) as f32;
                }
            }
            gemm(&mut gradient, true, ga.view(), true, self.x.slice(s![start..end, ..]), false, 1.0);
        }
        gradient.mapv(f64::from)
    }

    /// `dᵀ H d` of the reads' quadratic along `d` (C × d_in).
    fn read_curvature(&self, d: &Array2<f64>, k32: &Array2<f32>, kappa: &[f64]) -> f64 {
        let d32 = single(d);
        let mut quadratic = 0.0;
        for start in (0..self.rows()).step_by(CHUNK) {
            let end = (start + CHUNK).min(self.rows());
            let da = product(self.x.slice(s![start..end, ..]), false, d32.view(), true);
            let (mu, nk) = self.chunk_gates(start, end, kappa);
            let dz = &da * &mu;
            let kdz = product(dz.view(), false, k32.view(), false);
            for r in 0..end - start {
                let mut q = 0.0f64;
                for c in 0..self.pieces {
                    let dac = f64::from(da[[r, c]]);
                    q += f64::from(dz[[r, c]]) * f64::from(kdz[[r, c]]) + f64::from(nk[[r, c]]) * dac * dac / 12.0;
                }
                quadratic += 2.0 * self.scale * self.s[start + r] * q;
            }
        }
        quadratic
    }

    /// The reads' step (module note): the preconditioned negative gradient of the charged points'
    /// quadratic in `V`, and the length minimising that quadratic along it (`None` when it is flat).
    fn read_step(&self, v: &Array2<f64>, u: &Array2<f64>, right: &Array2<f64>) -> Result<Option<(Array2<f64>, f64)>, String> {
        let k = u.dot(self.fisher).dot(&u.t());
        let k32 = single(&k);
        let kappa: Vec<f64> = (0..self.pieces).map(|c| k[[c, c]]).collect();
        // Each subcomponent's own curvature, the left preconditioner.
        let mut left = vec![0.0f64; self.pieces];
        for t in 0..self.rows() {
            for (c, l) in left.iter_mut().enumerate() {
                let (mu, nu) = self.gates(t, c);
                *l += 2.0 * self.scale * self.s[t] * (f64::from(mu * mu) * kappa[c] + f64::from(nu) * kappa[c] / 12.0);
            }
        }
        let gradient = self.read_gradient(v, u, &k32, &kappa);
        let mut direction = -gradient.dot(right);
        for (c, mut row) in direction.outer_iter_mut().enumerate() {
            let l = left[c];
            row.mapv_inplace(|x| if l > 0.0 { x / l } else { 0.0 });
        }
        // Only the moves that keep every subcomponent on the map, `Uᵀ D = 0`.
        let along = u.t().dot(&direction);
        let direction = &direction - &u.dot(&pseudo_inverse(&u.t().dot(u), 0)?.dot(&along));
        let slope = -(&gradient * &direction).sum();
        let quadratic = self.read_curvature(&direction, &k32, &kappa);
        Ok((slope > 0.0 && quadratic > 0.0).then(|| (direction, slope / quadratic)))
    }

    /// The error bits of `v + η d` for every `η` of `lengths`, the sets held: each input's four
    /// scalars (`‖r‖²`, `r·S`, `‖S‖²`, the off energy) are quadratics in `η`, measured in one pass.
    fn read_profile(&self, v: &Array2<f64>, u: &Array2<f64>, d: &Array2<f64>, lengths: &[f64]) -> Vec<f64> {
        let c_total = self.pieces;
        let uf = u.dot(self.fisher);
        let k = uf.dot(&u.t());
        let k32 = single(&k);
        let (v32, d32, uf32) = (single(v), single(d), single(&uf));
        let mut totals = vec![0.0; lengths.len()];
        for start in (0..self.rows()).step_by(CHUNK) {
            let end = (start + CHUNK).min(self.rows());
            let a = product(self.x.slice(s![start..end, ..]), false, v32.view(), true);
            let da = product(self.x.slice(s![start..end, ..]), false, d32.view(), true);
            let g0 = product(self.y.slice(s![start..end, ..]), false, uf32.view(), true);
            let (mut z_on, mut z_off, mut dz_on, mut dz_off) = (a.clone(), a.clone(), da.clone(), da.clone());
            for r in 0..end - start {
                for c in 0..c_total {
                    if self.masks[(start + r) * c_total + c] == 1 {
                        z_off[[r, c]] = 0.0;
                        dz_off[[r, c]] = 0.0;
                    } else {
                        z_on[[r, c]] = 0.0;
                        dz_on[[r, c]] = 0.0;
                    }
                }
            }
            let pk = product(z_on.view(), false, k32.view(), false);
            let qk = product(z_off.view(), false, k32.view(), false);
            let dpk = product(dz_on.view(), false, k32.view(), false);
            let dqk = product(dz_off.view(), false, k32.view(), false);
            let rows: Vec<Vec<f64>> = (0..end - start)
                .into_par_iter()
                .map(|r| {
                    let t = start + r;
                    let dot = |x: &Array2<f32>, y: &Array2<f32>| -> f64 { x.row(r).iter().zip(y.row(r).iter()).map(|(p, q)| f64::from(*p) * f64::from(*q)).sum() };
                    let rr = [self.yfy[t] - 2.0 * dot(&z_on, &g0) + dot(&z_on, &pk), -2.0 * dot(&dz_on, &g0) + 2.0 * dot(&dz_on, &pk), dot(&dz_on, &dpk)];
                    let rs = [dot(&z_off, &g0) - dot(&z_on, &qk), dot(&dz_off, &g0) - dot(&dz_on, &qk) - dot(&z_on, &dqk), -dot(&dz_on, &dqk)];
                    let ss = [dot(&z_off, &qk), 2.0 * dot(&dz_off, &qk), dot(&dz_off, &dqk)];
                    let mut off = [0.0; 3];
                    for c in 0..c_total {
                        if self.masks[t * c_total + c] == 0 {
                            let (ac, dc) = (f64::from(a[[r, c]]), f64::from(da[[r, c]]));
                            off[0] += k[[c, c]] * ac * ac;
                            off[1] += 2.0 * k[[c, c]] * ac * dc;
                            off[2] += k[[c, c]] * dc * dc;
                        }
                    }
                    let at = |q: &[f64; 3], eta: f64| q[0] + eta * (q[1] + eta * q[2]);
                    let weight = self.scale * self.s[t];
                    lengths.iter().map(|&eta| weight * worst(at(&rr, eta), at(&rs, eta), at(&ss, eta), at(&off, eta)).0).collect()
                })
                .collect();
            for row in rows {
                for (total, value) in totals.iter_mut().zip(row) {
                    *total += value;
                }
            }
        }
        totals
    }
}

/// What [`fit`] fits to: the code's `n`, the library's size, the most rounds, and the seed of its
/// starting reads.
#[derive(Clone, Copy, Debug)]
pub struct Settings {
    pub observations: f64,
    pub pieces: usize,
    pub rounds: usize,
    pub seed: u64,
}

impl<'a> Fitting<'a> {
    /// The state of a fit of `pieces` subcomponents on `samples`, every subcomponent on.
    fn new(site: usize, w: &Array2<f64>, samples: &'a Samples, observations: f64, pieces: usize) -> Result<Self, String> {
        let (d_out, d_in) = w.dim();
        let x = &samples.reads;
        let rows = x.nrows();
        if x.ncols() != d_in || samples.sensitivity.len() != rows || samples.fisher.dim() != (d_out, d_out) || rows == 0 || pieces == 0 {
            return Err(format!("site {site}: samples of {rows} inputs do not fit its {d_out}×{d_in} map"));
        }
        let y = product(x.view(), false, single(w).view(), true);
        let yf = product(y.view(), false, single(&samples.fisher).view(), false);
        let yfy: Vec<f64> = (0..rows).map(|t| y.row(t).iter().zip(yf.row(t).iter()).map(|(a, b)| f64::from(*a) * f64::from(*b)).sum()).collect();
        // The reads' span, as the masked program restricts its reads (`Masked::project_reads`).
        let moment = eigh(samples.second_moment.view(), SymmetricAssembly::Mirrored, None).map_err(|e| format!("{e:?}"))?;
        let total: f64 = moment.values.iter().map(|l| l.max(0.0)).sum();
        let mut order: Vec<usize> = (0..moment.values.len()).collect();
        order.sort_by(|a, b| moment.values[*b].total_cmp(&moment.values[*a]));
        let mut kept = Vec::new();
        let mut held = 0.0;
        for &i in &order {
            if held >= (1.0 - LEFT_OUT) * total || moment.values[i] <= moment.band {
                break;
            }
            held += moment.values[i];
            kept.push(i);
        }
        let basis = moment.vectors.select(Axis(1), &kept);
        let roots = Array1::from_iter(kept.iter().map(|&i| moment.values[i].sqrt()));
        let span = &basis * &roots;
        let seeding = (&basis / &roots.mapv(|r| r * r)).dot(&basis.t());
        Ok(Self {
            x,
            w: w.clone(),
            span,
            seeding,
            y,
            yfy,
            s: &samples.sensitivity,
            fisher: &samples.fisher,
            scale: observations / (2.0 * LN_2),
            pieces,
            masks: vec![1; rows * pieces],
            corner: vec![true; rows],
        })
    }

    /// The round's report of the state as last coded.
    fn report(&self, round: usize, description: f64, error: f64) -> Round {
        let rows = self.rows() as f64;
        let on = self.masks.iter().filter(|m| **m == 1).count() as f64;
        let corner_share = self.corner.iter().filter(|c| **c).count() as f64 / rows;
        Round { round, code: (description + error) / rows, description: description / rows, error: error / rows, l0: on / rows, corner_share, reseeded: 0, read_steps: 0 }
    }
}

/// Every subcomponent's description bits.
fn description_bits(describe: &dyn Describe, site: usize, v: &Array2<f64>, u: &Array2<f64>) -> Result<Array1<f64>, String> {
    Ok((0..v.nrows())
        .into_par_iter()
        .map(|c| describe.bits(site, u.slice(s![c..c + 1, ..]), v.slice(s![c..c + 1, ..])))
        .collect::<Result<Vec<f64>, String>>()?
        .into())
}

/// A given library's code on `samples` (module note): every input's sets selected from all on.
pub fn measure(site: usize, w: &Array2<f64>, samples: &Samples, describe: &dyn Describe, observations: f64, library: &Library) -> Result<Round, String> {
    if library.mean.iter().any(|m| *m != 0.0) {
        return Err("a measured library reads the uncentred input".to_string());
    }
    let mut fitting = Fitting::new(site, w, samples, observations, library.v.nrows())?;
    let bits = description_bits(describe, site, &library.v, &library.u)?;
    let (description, error, _) = fitting.code(&library.v, &library.u, &bits, true, true);
    Ok(fitting.report(0, description, error))
}

/// A library of `settings.pieces` subcomponents for site `site` (`w` its `d_out × d_in` map, `site`
/// its index in `describe`), fitted on `samples` to the site's code (module note); `progress` sees
/// every round with the library it measured. Its first reads are `start` (rows of `d_in`, at most
/// `pieces`; a site reading a layer of units starts from the units themselves), the rest seeded
/// from inputs. The library reads the uncentred input (`mean` zero).
pub fn fit(
    site: usize,
    w: &Array2<f64>,
    samples: &Samples,
    describe: &dyn Describe,
    settings: Settings,
    start: Option<&Array2<f64>>,
    mut progress: impl FnMut(&Round, &Library),
) -> Result<Library, String> {
    let Settings { observations, pieces, rounds, seed } = settings;
    let d_in = w.ncols();
    let x = &samples.reads;
    let rows = x.nrows();
    let mut fitting = Fitting::new(site, w, samples, observations, pieces)?;
    // The reads' inverse second moment on their span (seeding) and the sensitivity-weighted one
    // (the reads' right preconditioner).
    let seeding = fitting.seeding.clone();

    let weighted = {
        let scaled = Array2::from_shape_fn(x.dim(), |(t, i)| (f64::from(x[[t, i]]) * samples.sensitivity[t].sqrt()) as f32);
        product(scaled.view(), true, scaled.view(), false).mapv(f64::from) / samples.sensitivity.sum().max(f64::MIN_POSITIVE)
    };
    let right = pseudo_inverse(&weighted, rows)?;
    // A read seeded from input `t`: its direction in the inverse second moment, unit variance.
    let seed_read = |t: usize| -> Array1<f64> {
        let xt = x.row(t).mapv(f64::from);
        let v = seeding.dot(&xt);
        let variance = v.dot(&samples.second_moment.dot(&v));
        if variance > 0.0 { v / variance.sqrt() } else { v }
    };
    let mut state = seed | 1;
    let mut draw = |n: usize| {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        (state % n as u64) as usize
    };
    let mut v = Array2::<f64>::zeros((pieces, d_in));
    let given = start.map_or(0, |s| s.nrows());
    if given > pieces || start.is_some_and(|s| s.ncols() != d_in) {
        return Err(format!("site {site}: {given} starting reads for {pieces} subcomponents of {d_in} reads"));
    }
    if let Some(s) = start {
        v.slice_mut(s![..given, ..]).assign(s);
    }
    for c in given..pieces {
        v.row_mut(c).assign(&seed_read(draw(rows)));
    }
    // The reads span the inputs' span, so every subcomponent on can be the map.
    let span = fitting.span.clone();
    let unwhiten = seeding.dot(&span);
    let cover = |v: &mut Array2<f64>, rows: &[usize]| -> Result<(), String> {
        // The directions (in the whitened reads) the reads leave out replace the reads of `rows`.
        let covered = svd(v.dot(&span).view(), false).map_err(|e| format!("{e:?}"))?;
        let rank = covered.singular_values.iter().filter(|s| **s > covered.band).count();
        for (&c, i) in rows.iter().zip(rank..covered.vt.nrows()) {
            v.row_mut(c).assign(&unwhiten.dot(&covered.vt.row(i)));
        }
        Ok(())
    };
    cover(&mut v, &(given..pieces).rev().collect::<Vec<_>>())?;
    // Every subcomponent on: the writes that make the library the map on these inputs.
    let mut u = fitting.writes(&v)?;
    let mut bits = description_bits(describe, site, &v, &u)?;
    let (description, error, mut errors) = fitting.code(&v, &u, &bits, true, true);
    let mut current = description + error;
    let mut report = fitting.report(0, description, error);
    for round in 0..rounds {
        // The writes: their closed form under the charged points, halved until the code falls.
        let target = fitting.writes(&v)?;
        let step = &target - &u;
        let before = fitting.corner.clone();
        let mut eta = 1.0;
        let mut moved = false;
        while eta > f64::EPSILON {
            let trial = &u + &(&step * eta);
            let (d, e, _) = fitting.code(&v, &trial, &bits, false, true);
            if d + e < current {
                (u, current, moved) = (trial, d + e, true);
                break;
            }
            eta *= 0.5;
        }
        if !moved {
            fitting.corner = before;
        }
        // The reads: along the preconditioned step, the length of least code among the quadratic's
        // minimiser times every power of √2 from 2⁻³² to 2⁴, all measured in one pass.
        if let Some((direction, alpha)) = fitting.read_step(&v, &u, &right)? {
            let mut lengths = vec![0.0];
            lengths.extend((-64..=8).map(|k| alpha * 2f64.powf(f64::from(k) / 2.0)));
            let profile = fitting.read_profile(&v, &u, &direction, &lengths);
            let (best, value) = profile.iter().enumerate().fold((0, profile[0]), |b, (i, p)| if *p < b.1 { (i, *p) } else { b });
            report.read_steps = best;
            if best > 0 {
                // The sets' description is unchanged; only the error moved.
                current += value - profile[0];
                v.scaled_add(lengths[best], &direction);
            }
        }
        // Reseeding: every subcomponent on nowhere, from the inputs of largest error in turn, kept
        // when the sets selected with it code the inputs in fewer bits.
        let on: Vec<bool> = (0..pieces).map(|c| (0..rows).any(|t| fitting.masks[t * pieces + c] == 1)).collect();
        let dead: Vec<usize> = (0..pieces).filter(|c| !on[*c]).collect();
        if !dead.is_empty() {
            let saved = (v.clone(), u.clone(), fitting.masks.clone(), fitting.corner.clone());
            let mut order: Vec<usize> = (0..rows).collect();
            order.sort_by(|a, b| errors[*b].total_cmp(&errors[*a]));
            let mut reseeded = 0;
            for (&c, &t) in dead.iter().zip(&order) {
                v.row_mut(c).assign(&seed_read(t));
                reseeded += 1;
            }
            cover(&mut v, &dead)?;
            // The writes again, so every subcomponent on stays the map.
            u = fitting.writes(&v)?;
            bits = description_bits(describe, site, &v, &u)?;
            let (d, e, _) = fitting.code(&v, &u, &bits, true, true);
            if d + e < current {
                report.reseeded = reseeded;
            } else {
                (v, u, fitting.masks, fitting.corner) = saved;
            }
        }
        progress(&report, &Library { v: v.clone(), u: u.clone(), mean: Array1::zeros(d_in) });
        // The next round's sets, under the descriptions as the steps left them.
        bits = description_bits(describe, site, &v, &u)?;
        let (description, error, next_errors) = fitting.code(&v, &u, &bits, true, true);
        let previous = report.code;
        report = fitting.report(round + 1, description, error);
        current = description + error;
        errors = next_errors;
        if previous - report.code < 1.0 {
            break;
        }
    }
    progress(&report, &Library { v: v.clone(), u: u.clone(), mean: Array1::zeros(d_in) });
    Ok(Library { v, u, mean: Array1::zeros(d_in) })
}
