//! A site's library fitted on its own inputs (#2951).
//!
//! The masked program (`super::masked`) codes each input by the subcomponents that run on it,
//! `Σ_{c on} bits(c) + n KL / ln 2`. Its libraries can come from VPD, from the map's own singular
//! pieces, or from this module: a library fitted on a site's inputs to that one total, each input
//! judged against the site's real computation on it.
//!
//! # The site's code
//!
//! A site `y = W x` with library `y = Σ_c u_c (v_c · x)` runs on input `t` with its real read `x_t`
//! (the hybrid state: the model's input with every earlier site already replaced by its own
//! explanation), so every subcomponent's real contribution `z_tc = u_c a_tc` (`a_tc = v_c · x_t`,
//! the read uncentred as the masked program reads it) is known. What the explanation claims at the
//! site is sufficiency: the subcomponents on carry the site's real output, and what they leave,
//!
//! ```text
//! r_t = W x_t − Σ_{c on} z_tc = D_t + Σ_{c off} z_tc        (D_t = W x_t − Σ_c z_tc),
//! ```
//!
//! is the off subcomponents' real vector sum. To second order its KL is `½ ‖r_t‖²_F`, so input `t`
//! is charged
//!
//! ```text
//! code_t = Σ_{c on} bits(c) + n / (2 ln 2) · ‖W x_t − Σ_{c on} u_c a_tc‖²_F.
//! ```
//!
//! * Only the real sum counts. Thousands of near-orthogonal interference terms of size `ε` sum to
//!   about `√N ε`, while charging each off subcomponent at its own size (`Σ_off ‖z_tc‖`, the box
//!   over every off gate to first order) grows as `N ε` and would keep most of any library on.
//! * Splitting a subcomponent into `q` copies of `z/q` changes nothing: they sum to `z`. The
//!   expectation over independent uniform off gates is not used: `q` copies cut its spread by `1/q`
//!   while every point of the box stays where it was, so it rewards refining for its own sake.
//! * Subcomponents that cancel within the site are not penalised by the error, only by their
//!   description: netting out takes precision, and `bits(c)`
//!   ([`super::blocks::Describe`]) prices magnitude and precision on every input a subcomponent
//!   runs on. Nothing else enters: no library is amortised.
//! * `F` is the site's mean written Fisher at the hybrid state, the same for every input, so the
//!   sets depend on the site's input alone: a run-time rule can choose them. It approximates each
//!   input's own Fisher; `E[x xᵀ ⊗ F_x] ≠ E[x xᵀ] ⊗ E[F_x]` in general.
//! * Each site is judged at its own input; that bounds nothing end to end. Sites composed on hybrid
//!   states, each checked whether or not the others are replaced (a box over the sites, not the
//!   subcomponents), are what the end-to-end code measures.
//!
//! # The fit
//!
//! Alternating steps that each lower that one total:
//!
//! * **Sets.** Each input's on-set: the subcomponents each worth their own bits alone, `n a_tc²
//!   u_cᵀFu_c / (2 ln 2) > bits(c)`, when they code the input in fewer bits than its current sets,
//!   then single flips swept until none lowers its code. A flip of `c` changes the error by `∓2 a_tc
//!   (U F e_t)_c + a_tc² (U F Uᵀ)_cc`, so each is `O(C)` with `U F e_t` kept up to date.
//! * **Writes.** With the sets and reads fixed the error is a quadratic in `U` whose metric `F`
//!   factors out, `tr F (UᵀQU − 2UᵀR)`, `Q = Σ_t z̃_t z̃_tᵀ`, `R = Σ_t z̃_t y_tᵀ`, `z̃_t` the reads on.
//!   Every subcomponent on is the map exactly, on every read direction (in the metric `E√Λ` of the
//!   reads' second moment, floored at `10⁻⁶` of its mean so directions no input reached still
//!   count), `Uᵀ V E√Λ = W E√Λ`: with `V E√Λ = P S Gᵀ` (full `P`) every such `U` is `U₀ + N Z`,
//!   `U₀ = P₁ S⁻¹ Gᵀ (W E√Λ)ᵀ`, `N` the columns of `P` past its rank, and `(NᵀQN) Z = Nᵀ(R − QU₀)`.
//! * **Reads.** With `U` and the sets fixed the error is a quadratic in `V`: one step along its
//!   preconditioned negative gradient (left by each subcomponent's own curvature, right by the
//!   reads' second moment) restricted to the moves that keep the map, `Uᵀ D = 0`, at its exact
//!   minimiser along that line.
//! * **Growth.** A subcomponent that runs nowhere or writes nothing takes half of the one whose
//!   inputs carry the most error, split along that one's inputs (at most a chunk of them, evenly
//!   strided; the halves sum to it); kept when the sets selected with them code the inputs in
//!   fewer bits.
//!
//! # Blocks
//!
//! A fitted library can be gated in blocks ([`blocks`]): column runs that one gate runs or drops
//! whole, block `c`'s real contribution on input `t` `U_cᵀ a_tc` and its description `bits(c)` of
//! the whole block. Merging two blocks that run together pays one description where both run; it
//! costs the whole block where only one is needed. The same code decides.
//!
//! The fit stops when a round saves less than one bit per input, or at `rounds`. Every product over
//! inputs runs in single precision (the fit only proposes a library; the masked program's exact
//! forward codes it), and pseudo-inverses drop eigenvalues within the single-precision band of their
//! sums, `√T 2⁻²⁴` of the largest.

use super::blocks::Describe;
use super::dense::{QrMode, eigh, qr, svd};
use super::derivatives::vjp;
use super::device::proposing;
use super::masked::{Library, Site, Target, read_values, sampled_label_cotangent};
use super::operator_program::{FamilyInputs, OperatorProgram};
use faer::linalg::matmul::matmul;
use faer::{Accum, MatMut, MatRef};
use gam_linalg::faer_ndarray::{fast_ab, fast_abt, fast_atb, matmul_parallelism};
use gam_linalg::roundoff::SymmetricAssembly;
use ndarray::{Array1, Array2, ArrayView2, Axis, s};
use rayon::prelude::*;
use std::f64::consts::LN_2;

/// What a site's fit reads of the model: its inputs' reads and sensitivities, its mean written
/// Fisher and the reads' second moment.
pub struct Samples {
    /// The reads, inputs × d_in (single precision).
    pub reads: Array2<f32>,
    /// Each input's `tr F_t / tr F`, its own Fisher's trace against the mean's.
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

/// `m` in single precision, row-major whatever `m`'s layout.
fn single(m: &Array2<f64>) -> Array2<f32> {
    m.mapv(|v| v as f32).as_standard_layout().into_owned()
}

/// The pseudo-inverse of a symmetric positive semidefinite matrix summed in single precision over
/// `terms` inputs: eigenvalues within `√terms · 2⁻²⁴` of the largest are dropped; with `terms`
/// zero, a matrix formed in float64, only its decomposition's band.
fn pseudo_inverse(m: &Array2<f64>, terms: usize) -> Result<Array2<f64>, String> {
    pseudo_inverse_within(m, terms, 0.0)
}

/// [`pseudo_inverse`] of a matrix projected from one whose own scale is `scale` (its trace, say):
/// its eigenvalues are resolved only within that one's band, `√terms 2⁻²⁴ · max(scale, largest)`.
fn pseudo_inverse_within(m: &Array2<f64>, terms: usize, scale: f64) -> Result<Array2<f64>, String> {
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
    let largest = d.values.iter().fold(scale, |m, l| m.max(*l));
    let floor = (largest * (terms as f64).sqrt() * f64::from(f32::EPSILON) / 2.0).max(d.band);
    let mut scaled = d.vectors.clone();
    for (k, l) in d.values.iter().enumerate() {
        let inverse = if *l > floor { 1.0 / l } else { 0.0 };
        scaled.column_mut(k).mapv_inplace(|x| x * inverse);
    }
    Ok(scaled.dot(&d.vectors.t()))
}

/// The per-input state of a fit: the sets.
struct Fitting<'a> {
    x: &'a Array2<f32>,
    /// The site's map, `d_out × d_in`: every subcomponent on is exactly it.
    w: Array2<f64>,
    /// `y = x Wᵀ`, inputs × d_out.
    y: Array2<f32>,
    /// `yᵀ F y` per input.
    yfy: Vec<f64>,
    fisher: &'a Array2<f64>,
    /// `n / (2 ln 2)`.
    scale: f64,
    pieces: usize,
    /// Inputs × pieces, 1 where on.
    masks: Vec<u8>,
}

/// The products every pass over the inputs needs: `V`, `U F` and `K = U F Uᵀ` in single precision,
/// and `K` in double for each input's own updates.
struct Operands {
    v32: Array2<f32>,
    uf32: Array2<f32>,
    k32: Array2<f32>,
    k: Array2<f64>,
}

/// What the inputs `start..end` of a chunk need under their sets: every read `a`, the site's own
/// output in the Fisher, `g = U F y`, the written error's `h = U F e` (`e = y − Uᵀ z`, `z` the reads
/// on), and each input's error `eᵀ F e`.
struct Residuals {
    a: Array2<f32>,
    g: Array2<f32>,
    h: Array2<f32>,
    error: Vec<f64>,
}

/// Geometry used only to fit the library, never to score it or choose its on-sets. In particular,
/// evaluating a learned switching rule must not eigendecompose the read covariance again.
struct ReadGeometry {
    span: Array2<f64>,
    unwhiten: Array2<f64>,
    seeding: Array2<f64>,
}

impl ReadGeometry {
    fn new(moment: &Array2<f64>) -> Result<Self, String> {
        let moment = eigh(moment.view(), SymmetricAssembly::Mirrored, None).map_err(|e| format!("{e:?}"))?;
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
        let seeding = (&basis / &roots.mapv(|r| r * r)).dot(&basis.t());
        // Keep every direction in the all-on constraint, including those no input reached.
        let floor = LEFT_OUT * total / moment.values.len().max(1) as f64;
        let all_roots = moment.values.mapv(|l| l.max(floor).sqrt());
        let span = &moment.vectors * &all_roots;
        let unwhiten = &moment.vectors / &all_roots;
        Ok(Self { span, unwhiten, seeding })
    }
}

/// Column `j`'s block, and every block's first column, of a partition into runs of `ranks`.
fn partition(ranks: &[usize]) -> (Vec<usize>, Vec<usize>) {
    let mut starts = vec![0];
    let mut block_of = Vec::new();
    for (b, r) in ranks.iter().enumerate() {
        starts.push(starts[b] + r);
        block_of.extend(std::iter::repeat_n(b, *r));
    }
    (block_of, starts)
}

/// What every input's sets are chosen against: `K = U F Uᵀ`, the blocks' first columns `starts`
/// (column runs), their description `bits`, and `weight` bits per unit of error.
struct Gram<'b> {
    k: &'b Array2<f64>,
    starts: &'b [usize],
    bits: &'b [f64],
    weight: f64,
}

/// One input as its sets see it: its reads `a`, `g = U F y`, its `yᵀ F y`, and for its current sets
/// `h = U F e` (kept up to date) and `error = eᵀ F e`.
struct Input {
    a: Vec<f64>,
    g: Vec<f64>,
    yfy: f64,
    h: Vec<f64>,
    error: f64,
}

/// One input's sets (module note, "Sets") `m` over the blocks of `gram`: with `flip`, the blocks each
/// worth their own bits alone when they code the input in fewer bits than its current sets, then
/// single flips swept until none lowers its code. Returns the input's description and error bits.
fn select(gram: &Gram<'_>, input: &mut Input, m: &mut [u8], flip: bool) -> (f64, f64) {
    let Gram { k, starts, bits, weight } = *gram;
    let Input { a, g, yfy, h, error } = input;
    let (yfy, mut error) = (*yfy, *error);
    let blocks = bits.len();
    let columns = |b: usize| starts[b]..starts[b + 1];
    // `aᵀ K a` over block `b`'s columns.
    let own = |b: usize| -> f64 { columns(b).map(|i| a[i] * columns(b).map(|j| k[[i, j]] * a[j]).sum::<f64>()).sum() };
    if flip {
        let listed: f64 = (0..blocks).filter(|b| m[*b] == 1).map(|b| bits[b]).sum();
        let alone: Vec<usize> = (0..blocks).filter(|&b| weight * own(b) > bits[b]).collect();
        let cols: Vec<usize> = alone.iter().flat_map(|&b| columns(b)).collect();
        let alone_error = yfy - 2.0 * cols.iter().map(|&i| a[i] * g[i]).sum::<f64>()
            + cols.iter().map(|&i| a[i] * cols.iter().map(|&j| k[[i, j]] * a[j]).sum::<f64>()).sum::<f64>();
        let alone_bits: f64 = alone.iter().map(|&b| bits[b]).sum();
        if alone_bits + weight * alone_error.max(0.0) < listed + weight * error.max(0.0) {
            m.fill(0);
            for &b in &alone {
                m[b] = 1;
            }
            h.copy_from_slice(g.as_slice());
            // `K` is symmetric: column `i` is row `i`.
            for &i in &cols {
                for (hj, kij) in h.iter_mut().zip(k.row(i)) {
                    *hj -= kij * a[i];
                }
            }
            error = alone_error;
        }
        for _ in 0..blocks.max(1) {
            let mut flipped = false;
            for b in 0..blocks {
                // Running `b` writes `Uᵀ a_b` more; dropping it, that much less.
                let sigma = if m[b] == 1 { -1.0 } else { 1.0 };
                let along: f64 = columns(b).map(|i| a[i] * h[i]).sum();
                let delta = -2.0 * sigma * along + own(b);
                if sigma * bits[b] + weight * delta < 0.0 {
                    error += delta;
                    for i in columns(b) {
                        for (hj, kij) in h.iter_mut().zip(k.row(i)) {
                            *hj -= sigma * kij * a[i];
                        }
                    }
                    m[b] = u8::from(sigma > 0.0);
                    flipped = true;
                }
            }
            if !flipped {
                break;
            }
        }
    }
    input.error = error;
    let listed: f64 = (0..blocks).filter(|b| m[*b] == 1).map(|b| bits[b]).sum();
    (listed, weight * error.max(0.0))
}

impl<'a> Fitting<'a> {
    /// The state of a fit of `pieces` subcomponents on `samples`, every subcomponent on.
    fn new(site: usize, w: &Array2<f64>, samples: &'a Samples, observations: f64, pieces: usize) -> Result<Self, String> {
        let (d_out, d_in) = w.dim();
        let x = &samples.reads;
        let rows = x.nrows();
        if x.ncols() != d_in || samples.fisher.dim() != (d_out, d_out) || rows == 0 || pieces == 0 {
            return Err(format!("site {site}: samples of {rows} inputs do not fit its {d_out}×{d_in} map"));
        }
        let y = product(x.view(), false, single(w).view(), true);
        let yf = product(y.view(), false, single(&samples.fisher).view(), false);
        let yfy: Vec<f64> = (0..rows).map(|t| y.row(t).iter().zip(yf.row(t).iter()).map(|(a, b)| f64::from(*a) * f64::from(*b)).sum()).collect();
        Ok(Self { x, w: w.clone(), y, yfy, fisher: &samples.fisher, scale: observations / (2.0 * LN_2), pieces, masks: vec![1; rows * pieces] })
    }

    fn rows(&self) -> usize {
        self.x.nrows()
    }

    fn operands(&self, v: &Array2<f64>, u: &Array2<f64>) -> Operands {
        let uf = u.dot(self.fisher);
        let k = uf.dot(&u.t());
        Operands { v32: single(v), uf32: single(&uf), k32: single(&k), k }
    }

    /// The inputs `start..end` under the column gates `on` (rows × columns, 1 where a column's block
    /// runs).
    fn residuals(&self, start: usize, end: usize, ops: &Operands, on: &Array2<f32>) -> Residuals {
        let a = product(self.x.slice(s![start..end, ..]), false, ops.v32.view(), true);
        let g = product(self.y.slice(s![start..end, ..]), false, ops.uf32.view(), true);
        let z = &a * on;
        let p = product(z.view(), false, ops.k32.view(), false);
        let h = &g - &p;
        let error = (0..end - start)
            .into_par_iter()
            .map(|r| {
                let (zg, zp): (f64, f64) = z.row(r).iter().zip(g.row(r)).zip(p.row(r)).fold((0.0, 0.0), |(x, y), ((zi, gi), pi)| {
                    (x + f64::from(*zi) * f64::from(*gi), y + f64::from(*zi) * f64::from(*pi))
                });
                self.yfy[start + r] - 2.0 * zg + zp
            })
            .collect();
        Residuals { a, g, h, error }
    }

    /// The column gates of the inputs `start..end` from block sets `masks` (inputs × blocks).
    fn gates(&self, start: usize, end: usize, masks: &[u8], block_of: &[usize]) -> Array2<f32> {
        let blocks = masks.len() / self.rows();
        Array2::from_shape_fn((end - start, self.pieces), |(r, j)| f32::from(masks[(start + r) * blocks + block_of[j]]))
    }

    /// The code of `(v, u)` gated in blocks of `ranks` (column runs, module note "Blocks"): `masks`
    /// (inputs × blocks) are the sets, selected first with `flip`. Returns the total description and
    /// error bits.
    fn code_blocks(&self, v: &Array2<f64>, u: &Array2<f64>, ranks: &[usize], bits: &[f64], masks: &mut [u8], flip: bool) -> (f64, f64) {
        let blocks = ranks.len();
        let (block_of, starts) = partition(ranks);
        let ops = self.operands(v, u);
        let (mut description, mut error) = (0.0, 0.0);
        for start in (0..self.rows()).step_by(CHUNK) {
            let end = (start + CHUNK).min(self.rows());
            let on = self.gates(start, end, masks, &block_of);
            let res = self.residuals(start, end, &ops, &on);
            let gram = Gram { k: &ops.k, starts: &starts, bits, weight: self.scale };
            let results: Vec<(f64, f64)> = masks[start * blocks..end * blocks]
                .par_chunks_mut(blocks)
                .enumerate()
                .map(|(r, m)| {
                    let widen = |row: ndarray::ArrayView1<'_, f32>| row.iter().map(|x| f64::from(*x)).collect::<Vec<f64>>();
                    let mut input = Input { a: widen(res.a.row(r)), g: widen(res.g.row(r)), yfy: self.yfy[start + r], h: widen(res.h.row(r)), error: res.error[r] };
                    select(&gram, &mut input, m, flip)
                })
                .collect();
            for (listed, err) in results {
                description += listed;
                error += err;
            }
        }
        (description, error)
    }

    /// The code of `(v, u)`, every subcomponent its own gate (module note); with `flip`, every
    /// input's sets selected first (module note, "Sets"). Returns the total description and error bits.
    fn code(&mut self, v: &Array2<f64>, u: &Array2<f64>, bits: &Array1<f64>, flip: bool) -> (f64, f64) {
        let mut masks = std::mem::take(&mut self.masks);
        let bits = bits.to_vec();
        let out = self.code_blocks(v, u, &vec![1; self.pieces], &bits, &mut masks, flip);
        self.masks = masks;
        out
    }

    /// Every column's real size on every input, `|a_tc| ‖u_c‖_F` (inputs × columns).
    fn usage(&self, v: &Array2<f64>, u: &Array2<f64>) -> Array2<f32> {
        let ops = self.operands(v, u);
        let sizes: Vec<f32> = (0..self.pieces).map(|c| ops.k[[c, c]].max(0.0).sqrt() as f32).collect();
        let mut out = product(self.x.view(), false, ops.v32.view(), true);
        for mut row in out.outer_iter_mut() {
            for (x, s) in row.iter_mut().zip(&sizes) {
                *x = x.abs() * s;
            }
        }
        out
    }

    /// The writes (module note, "Writes"): the closed-form minimiser of the error under the sets,
    /// with every subcomponent on the map on every read direction (`span`).
    fn writes(&self, v: &Array2<f64>, span: &Array2<f64>) -> Result<Array2<f64>, String> {
        let c_total = self.pieces;
        let v32 = single(v);
        let block_of: Vec<usize> = (0..c_total).collect();
        let mut q32 = Array2::<f32>::zeros((c_total, c_total));
        let mut r32 = Array2::<f32>::zeros((c_total, self.y.ncols()));
        for start in (0..self.rows()).step_by(CHUNK) {
            let end = (start + CHUNK).min(self.rows());
            let a = product(self.x.slice(s![start..end, ..]), false, v32.view(), true);
            let z = &a * &self.gates(start, end, &self.masks, &block_of);
            gemm(&mut q32, true, z.view(), true, z.view(), false, 1.0);
            gemm(&mut r32, true, z.view(), true, self.y.slice(s![start..end, ..]), false, 1.0);
        }
        let q = q32.mapv(f64::from);
        // On every read direction (`x = E√Λ ξ`) the constraint is `Uᵀ (V E√Λ) = W E√Λ`.
        let decomposed = svd(v.dot(span).view(), true).map_err(|e| format!("{e:?}"))?;
        let rank = decomposed.singular_values.iter().filter(|s| **s > decomposed.band).count();
        let p1 = decomposed.u.slice(s![.., ..rank]);
        let inverse = Array1::from_iter(decomposed.singular_values.iter().take(rank).map(|s| 1.0 / s));
        let particular = (&p1 * &inverse).dot(&decomposed.vt.slice(s![..rank, ..])).dot(&self.w.dot(span).t());
        let null = decomposed.u.slice(s![.., rank..]).to_owned();
        // `NᵀQN` is resolved only as far as `Q` itself is: its band is `Q`'s, bounded by its trace.
        let reduced = null.t().dot(&q).dot(&null);
        let z = pseudo_inverse_within(&reduced, self.rows(), q.diag().sum())?.dot(&null.t().dot(&(r32.mapv(f64::from) - q.dot(&particular))));
        Ok(particular + null.dot(&z))
    }

    /// The reads' step (module note, "Reads"): its direction and its exact length, or `None` when no
    /// move keeping the map lowers the error.
    fn read_step(&self, v: &Array2<f64>, u: &Array2<f64>, right: &Array2<f64>) -> Result<Option<(Array2<f64>, f64)>, String> {
        let c_total = self.pieces;
        let ops = self.operands(v, u);
        let block_of: Vec<usize> = (0..c_total).collect();
        let mut gradient = Array2::<f32>::zeros(v.dim());
        let mut left = vec![0.0f64; c_total];
        for start in (0..self.rows()).step_by(CHUNK) {
            let end = (start + CHUNK).min(self.rows());
            let on = self.gates(start, end, &self.masks, &block_of);
            let res = self.residuals(start, end, &ops, &on);
            // `∂ error / ∂ a_tc = −2 w h_tc` where `c` runs, nothing where it is off.
            let ga = (&on * &res.h) * (-2.0 * self.scale as f32);
            for (c, l) in left.iter_mut().enumerate() {
                *l += 2.0 * self.scale * ops.k[[c, c]] * on.column(c).iter().map(|x| f64::from(*x)).sum::<f64>();
            }
            gemm(&mut gradient, true, ga.view(), true, self.x.slice(s![start..end, ..]), false, 1.0);
        }
        let gradient = gradient.mapv(f64::from);
        let mut direction = -gradient.dot(right);
        for (c, mut row) in direction.outer_iter_mut().enumerate() {
            let l = left[c];
            row.mapv_inplace(|x| if l > 0.0 { x / l } else { 0.0 });
        }
        // Only the moves that keep every subcomponent on the map, `Uᵀ D = 0`.
        let along = u.t().dot(&direction);
        let direction = &direction - &u.dot(&pseudo_inverse(&u.t().dot(u), 0)?.dot(&along));
        let slope = -(&gradient * &direction).sum();
        // The error is exactly quadratic along the line: `w ‖Uᵀ (on ⊙ δa)‖²_F` per input.
        let d32 = single(&direction);
        let mut curvature = 0.0;
        for start in (0..self.rows()).step_by(CHUNK) {
            let end = (start + CHUNK).min(self.rows());
            let dz = product(self.x.slice(s![start..end, ..]), false, d32.view(), true) * &self.gates(start, end, &self.masks, &block_of);
            let dk = product(dz.view(), false, ops.k32.view(), false);
            curvature += 2.0 * self.scale * dz.iter().zip(dk.iter()).map(|(a, b)| f64::from(*a) * f64::from(*b)).sum::<f64>();
        }
        Ok((slope > 0.0 && curvature > 0.0).then(|| (direction, slope / curvature)))
    }

    /// Per subcomponent, the error bits of the inputs it runs on.
    fn carried(&self, v: &Array2<f64>, u: &Array2<f64>) -> Vec<f64> {
        let c_total = self.pieces;
        let ops = self.operands(v, u);
        let block_of: Vec<usize> = (0..c_total).collect();
        let mut carried = vec![0.0; c_total];
        for start in (0..self.rows()).step_by(CHUNK) {
            let end = (start + CHUNK).min(self.rows());
            let on = self.gates(start, end, &self.masks, &block_of);
            let res = self.residuals(start, end, &ops, &on);
            for (r, error) in res.error.iter().enumerate() {
                for (c, total) in carried.iter_mut().enumerate() {
                    if on[[r, c]] > 0.0 {
                        *total += self.scale * error.max(0.0);
                    }
                }
            }
        }
        carried
    }

    /// The round's report of the state as last coded.
    fn report(&self, round: usize, description: f64, error: f64) -> Round {
        let rows = self.rows() as f64;
        let on = self.masks.iter().filter(|m| **m == 1).count() as f64;
        Round { round, code: (description + error) / rows, description: description / rows, error: error / rows, l0: on / rows, reseeded: 0, read_steps: 0 }
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

/// Every subcomponent's description bits.
fn description_bits(describe: &dyn Describe, site: usize, v: &Array2<f64>, u: &Array2<f64>) -> Result<Array1<f64>, String> {
    Ok((0..v.nrows())
        .into_par_iter()
        .map(|c| gam_linalg::faer_ndarray::with_nested_parallel(|| describe.bits_at(site, c, u.slice(s![c..c + 1, ..]), v.slice(s![c..c + 1, ..]))))
        .collect::<Result<Vec<f64>, String>>()?
        .into())
}

/// Every subcomponent's price for selecting the sets inside a fit: its cheap price
/// ([`Describe::cheap`]) on the description's own scale, the median of `exact / cheap` over an evenly
/// strided sample of `√C` subcomponents (each priced exactly), whose own exact prices stand. A
/// description with no cheap price is priced exactly throughout. The fit's final decode prices
/// every subcomponent that runs exactly ([`decoded_prices`]).
fn prices(describe: &dyn Describe, site: usize, v: &Array2<f64>, u: &Array2<f64>) -> Result<Array1<f64>, String> {
    let c_total = v.nrows();
    let cheap: Option<Vec<f64>> = (0..c_total)
        .into_par_iter()
        .map(|c| describe.cheap(site, u.slice(s![c..c + 1, ..]), v.slice(s![c..c + 1, ..])))
        .collect::<Result<Vec<Option<f64>>, String>>()?
        .into_iter()
        .collect();
    let Some(cheap) = cheap else { return description_bits(describe, site, v, u) };
    let stride = ((c_total as f64).sqrt() as usize).max(1);
    let sample: Vec<usize> = (0..c_total).step_by(stride).collect();
    let sampled: Vec<f64> = sample
        .par_iter()
        .map(|&c| gam_linalg::faer_ndarray::with_nested_parallel(|| describe.bits_at(site, c, u.slice(s![c..c + 1, ..]), v.slice(s![c..c + 1, ..]))))
        .collect::<Result<_, String>>()?;
    let scale = median_ratio(sample.iter().zip(&sampled).map(|(c, e)| (*e, cheap[*c])));
    let mut bits = Array1::from_iter(cheap.iter().map(|b| b * scale));
    for (c, e) in sample.iter().zip(&sampled) {
        bits[*c] = *e;
    }
    Ok(bits)
}

/// The median of `exact / cheap` over the pairs with a positive cheap price (1 with none).
fn median_ratio(pairs: impl Iterator<Item = (f64, f64)>) -> f64 {
    let mut ratios: Vec<f64> = pairs.filter(|(_, cheap)| *cheap > 0.0).map(|(exact, cheap)| exact / cheap).collect();
    if ratios.is_empty() {
        return 1.0;
    }
    ratios.sort_by(f64::total_cmp);
    ratios[ratios.len() / 2]
}

/// The exact price of every subcomponent some input runs, the cheap one elsewhere: the final
/// decode's description of the sets selected.
fn decoded_prices(describe: &dyn Describe, site: usize, v: &Array2<f64>, u: &Array2<f64>, fitting: &Fitting, bits: &Array1<f64>) -> Result<Array1<f64>, String> {
    let c_total = v.nrows();
    let rows = fitting.masks.len() / c_total.max(1);
    let on: Vec<usize> = (0..c_total).filter(|c| (0..rows).any(|t| fitting.masks[t * c_total + c] == 1)).collect();
    let exact: Vec<f64> = on
        .par_iter()
        .map(|c| gam_linalg::faer_ndarray::with_nested_parallel(|| describe.bits_at(site, *c, u.slice(s![*c..*c + 1, ..]), v.slice(s![*c..*c + 1, ..]))))
        .collect::<Result<_, String>>()?;
    let mut out = bits.clone();
    for (c, e) in on.iter().zip(exact) {
        out[*c] = e;
    }
    Ok(out)
}

/// A given library's code on `samples` (module note), every input's sets selected, and those sets
/// (per input, the subcomponents on, ascending).
pub fn measure(site: usize, w: &Array2<f64>, samples: &Samples, describe: &dyn Describe, observations: f64, library: &Library) -> Result<(Round, Vec<Vec<u32>>), String> {
    if library.mean.iter().any(|m| *m != 0.0) {
        return Err("a measured library reads the uncentred input".to_string());
    }
    let mut fitting = Fitting::new(site, w, samples, observations, library.v.nrows())?;
    let bits = prices(describe, site, &library.v, &library.u)?;
    fitting.code(&library.v, &library.u, &bits, true);
    let bits = decoded_prices(describe, site, &library.v, &library.u, &fitting, &bits)?;
    let (description, error) = fitting.code(&library.v, &library.u, &bits, false);
    let pieces = library.v.nrows();
    let sets = fitting.masks.chunks(pieces).map(|m| (0..pieces as u32).filter(|c| m[*c as usize] == 1).collect()).collect();
    Ok((fitting.report(0, description, error), sets))
}

/// The code of given sets on `samples` (per input, the subcomponents on, ascending) under the
/// library's own terms, the ones [`measure`] selects its sets by: a switching function's sets, or
/// any other rule's, priced exactly as the selection's.
pub fn code_of(site: usize, w: &Array2<f64>, samples: &Samples, describe: &dyn Describe, observations: f64, library: &Library, sets: &[Vec<u32>]) -> Result<Round, String> {
    if library.mean.iter().any(|m| *m != 0.0) {
        return Err("a measured library reads the uncentred input".to_string());
    }
    let pieces = library.v.nrows();
    let mut fitting = Fitting::new(site, w, samples, observations, pieces)?;
    if sets.len() != fitting.rows() || sets.iter().flatten().any(|c| *c as usize >= pieces) {
        return Err(format!("site {site}: sets of {} inputs over {pieces} subcomponents do not fit {} inputs", sets.len(), fitting.rows()));
    }
    fitting.masks.fill(0);
    for (t, set) in sets.iter().enumerate() {
        for c in set {
            fitting.masks[t * pieces + *c as usize] = 1;
        }
    }
    let bits = decoded_prices(describe, site, &library.v, &library.u, &fitting, &Array1::zeros(pieces))?;
    let (description, error) = fitting.code(&library.v, &library.u, &bits, false);
    Ok(fitting.report(0, description, error))
}

/// A library of `settings.pieces` subcomponents for site `site` (`w` its `d_out × d_in` map, `site`
/// its index in `describe`), fitted on `samples` to the site's code (module note); `progress` sees
/// every round with the library it measured. Its first subcomponents are `start` (at most
/// `pieces`; a site reading a layer of units starts from the units themselves), the rest read
/// inputs; when `start` carries its writes and is the map, the rest start writing nothing, else
/// the first writes are the smallest that make every subcomponent on the map. The library reads
/// the uncentred input (`mean` zero).
pub fn fit(
    site: usize,
    w: &Array2<f64>,
    samples: &Samples,
    describe: &dyn Describe,
    settings: Settings,
    start: Option<&Library>,
    mut progress: impl FnMut(&Round, &Library),
) -> Result<Library, String> {
    let Settings { observations, pieces, rounds, seed } = settings;
    let d_in = w.ncols();
    let x = &samples.reads;
    let rows = x.nrows();
    let mut fitting = Fitting::new(site, w, samples, observations, pieces)?;
    let geometry = ReadGeometry::new(&samples.second_moment)?;
    // The reads' inverse second moment on their span (seeding) and in full (the reads' right
    // preconditioner).
    let seeding = &geometry.seeding;
    let right = pseudo_inverse(&samples.second_moment, rows)?;
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
    let given = start.map_or(0, |s| s.v.nrows());
    if given > pieces || start.is_some_and(|s| s.v.ncols() != d_in) {
        return Err(format!("site {site}: {given} starting reads for {pieces} subcomponents of {d_in} reads"));
    }
    if let Some(s) = start {
        v.slice_mut(s![..given, ..]).assign(&s.v);
    }
    for c in given..pieces {
        v.row_mut(c).assign(&seed_read(draw(rows)));
    }
    // The reads span every read direction, so every subcomponent on can be the map.
    let span = &geometry.span;
    let unwhiten = &geometry.unwhiten;
    let cover = |v: &mut Array2<f64>, rows: &[usize]| -> Result<(), String> {
        // The directions (in the whitened reads) the reads leave out replace the reads of `rows`.
        let covered = svd(v.dot(span).view(), false).map_err(|e| format!("{e:?}"))?;
        let rank = covered.singular_values.iter().filter(|s| **s > covered.band).count();
        for (&c, i) in rows.iter().zip(rank..covered.vt.nrows()) {
            v.row_mut(c).assign(&unwhiten.dot(&covered.vt.row(i)));
        }
        Ok(())
    };
    cover(&mut v, &(given..pieces).rev().collect::<Vec<_>>())?;
    // The starting writes when they are the map, else the smallest that make every subcomponent on
    // the map.
    let tolerance = 1e-9 * w.iter().fold(0.0_f64, |m, x| m.max(x.abs()));
    let mut u = match start.filter(|s| s.u.nrows() == given && s.u.ncols() == w.nrows()) {
        Some(s) if (&s.u.t().dot(&s.v) - w).iter().all(|e| e.abs() <= tolerance) => {
            let mut u = Array2::<f64>::zeros((pieces, w.nrows()));
            u.slice_mut(s![..given, ..]).assign(&s.u);
            u
        }
        // With every subcomponent on, the error is what all on leaves, the same for every exact `U`:
        // the writes' closed form is then its particular solution.
        _ => fitting.writes(&v, span)?,
    };
    let mut bits = prices(describe, site, &v, &u)?;
    let (description, error) = fitting.code(&v, &u, &bits, true);
    let mut current = description + error;
    let mut report = fitting.report(0, description, error);
    for round in 0..rounds {
        // The writes: the error's minimiser under the sets, kept when the code falls.
        let trial = fitting.writes(&v, span)?;
        let (d, e) = fitting.code(&v, &trial, &bits, false);
        if d + e < current {
            (u, current) = (trial, d + e);
        }
        // The reads: one step at its exact length, kept when the code falls.
        if let Some((direction, length)) = fitting.read_step(&v, &u, &right)? {
            let trial = &v + &(&direction * length);
            let (d, e) = fitting.code(&trial, &u, &bits, false);
            if d + e < current {
                (v, current) = (trial, d + e);
                report.read_steps = 1;
            }
        }
        // Growth: every subcomponent that runs nowhere or writes nothing takes half of the one whose
        // inputs carry the most error, split along its inputs (`super::masked::split`:
        // the halves sum to it, so every subcomponent on stays the map); kept when the sets
        // selected with them code the inputs in fewer bits.
        let sizes: Vec<f64> = u.outer_iter().map(|r| r.dot(&r).sqrt()).collect();
        let largest = sizes.iter().fold(0.0_f64, |m, s| m.max(*s));
        let idle: Vec<usize> = (0..pieces).filter(|&c| sizes[c] <= f64::EPSILON * largest || (0..rows).all(|t| fitting.masks[t * pieces + c] == 0)).collect();
        if !idle.is_empty() {
            let saved = (v.clone(), u.clone(), fitting.masks.clone());
            let carried = fitting.carried(&v, &u);
            let mut parents: Vec<usize> = (0..pieces).filter(|c| !idle.contains(c)).collect();
            parents.sort_by(|a, b| carried[*b].total_cmp(&carried[*a]));
            // Each split from at most `CHUNK` of its parent's inputs (evenly strided), all in parallel.
            let halves: Vec<(usize, usize, Option<Library>)> = idle
                .iter()
                .zip(&parents)
                .collect::<Vec<_>>()
                .into_par_iter()
                .map(|(&slot, &parent)| {
                    let members: Vec<usize> = (0..rows).filter(|t| fitting.masks[t * pieces + parent] == 1).collect();
                    if members.len() < 2 {
                        return (slot, parent, None);
                    }
                    let stride = members.len().div_ceil(CHUNK);
                    let kept: Vec<usize> = members.iter().step_by(stride).copied().collect();
                    let reads = Array2::from_shape_fn((kept.len(), d_in), |(i, j)| f64::from(x[[kept[i], j]]));
                    let one = Library { v: v.slice(s![parent..parent + 1, ..]).to_owned(), u: u.slice(s![parent..parent + 1, ..]).to_owned(), mean: Array1::zeros(d_in) };
                    let (split, _, _) = super::masked::split(&one, &reads, &Array2::ones((kept.len(), 1)));
                    (slot, parent, (split.v.nrows() == 2).then_some(split))
                })
                .collect();
            let mut grown = 0;
            for (slot, parent, split) in halves {
                if let Some(split) = split {
                    v.row_mut(parent).assign(&split.v.row(0));
                    v.row_mut(slot).assign(&split.v.row(1));
                    let write = u.row(parent).to_owned();
                    u.row_mut(slot).assign(&write);
                    grown += 1;
                }
            }
            bits = prices(describe, site, &v, &u)?;
            let (d, e) = fitting.code(&v, &u, &bits, true);
            if grown > 0 && d + e < current {
                report.reseeded = grown;
            } else {
                (v, u, fitting.masks) = saved;
            }
        }
        progress(&report, &Library { v: v.clone(), u: u.clone(), mean: Array1::zeros(d_in) });
        // The next round's sets, under the descriptions as the steps left them.
        bits = prices(describe, site, &v, &u)?;
        let (description, error) = fitting.code(&v, &u, &bits, true);
        let previous = report.code;
        report = fitting.report(round + 1, description, error);
        current = description + error;
        if previous - report.code < 1.0 {
            break;
        }
    }
    // The final decode: every subcomponent some input runs at its description's own price.
    let bits = decoded_prices(describe, site, &v, &u, &fitting, &bits)?;
    let (description, error) = fitting.code(&v, &u, &bits, false);
    report = Round { reseeded: report.reseeded, read_steps: report.read_steps, ..fitting.report(report.round, description, error) };
    progress(&report, &Library { v: v.clone(), u: u.clone(), mean: Array1::zeros(d_in) });
    Ok(Library { v, u, mean: Array1::zeros(d_in) })
}

/// [`measure`] of a library gated in blocks of `ranks` (column runs): its code and every input's
/// blocks on, selected from all on.
pub fn measure_blocks(
    site: usize,
    w: &Array2<f64>,
    samples: &Samples,
    describe: &dyn Describe,
    observations: f64,
    library: &Library,
    ranks: &[usize],
) -> Result<(Round, Vec<Vec<u32>>), String> {
    if library.mean.iter().any(|m| *m != 0.0) {
        return Err("a measured library reads the uncentred input".to_string());
    }
    if ranks.contains(&0) || ranks.iter().sum::<usize>() != library.v.nrows() {
        return Err(format!("site {site}: blocks {ranks:?} do not partition {} subcomponents", library.v.nrows()));
    }
    let fitting = Fitting::new(site, w, samples, observations, library.v.nrows())?;
    let rows = fitting.rows();
    let blocks = ranks.len();
    let mut start = 0;
    let mut bits = Vec::with_capacity(blocks);
    for r in ranks {
        bits.push(describe.bits(site, library.u.slice(s![start..start + r, ..]), library.v.slice(s![start..start + r, ..]))?);
        start += r;
    }
    let mut masks = vec![1u8; rows * blocks];
    let (description, error) = fitting.code_blocks(&library.v, &library.u, ranks, &bits, &mut masks, true);
    let on = masks.iter().filter(|m| **m == 1).count() as f64;
    let report = Round { round: 0, code: (description + error) / rows as f64, description: description / rows as f64, error: error / rows as f64, l0: on / rows as f64, reseeded: 0, read_steps: 0 };
    Ok((report, masks.chunks(blocks).map(|m| (0..blocks as u32).filter(|c| m[*c as usize] == 1).collect()).collect()))
}

/// A site's selection at run time: [`measure_blocks`]'s sets (from all on, in the metric `fisher`
/// at `n = observations`), each input's from its read alone, with what that forms on every call
/// held once: `V`, `U F W`, `Wᵀ F W` and `K = U F Uᵀ`, so an input costs `O(C (d_in + C))`.
#[derive(Debug)]
pub struct Selector {
    v: Array2<f64>,
    ufw: Array2<f64>,
    wfw: Array2<f64>,
    k: Array2<f64>,
    starts: Vec<usize>,
    bits: Vec<f64>,
    weight: f64,
}

impl Selector {
    /// The selection of `library` (on `w`, `d_out × d_in`) in blocks of `ranks`, block `b` priced
    /// at `bits[b]`.
    pub fn new(w: &Array2<f64>, fisher: &Array2<f64>, library: &Library, ranks: &[usize], bits: &[f64], observations: f64) -> Result<Self, String> {
        let (d_out, d_in) = w.dim();
        let columns = library.v.nrows();
        if library.v.ncols() != d_in || library.u.dim() != (columns, d_out) || fisher.dim() != (d_out, d_out) {
            return Err(format!("a {:?}, {:?} library in a {:?} metric for a {d_out}×{d_in} map", library.u.dim(), library.v.dim(), fisher.dim()));
        }
        if library.mean.iter().any(|m| *m != 0.0) {
            return Err("a selected library reads the uncentred input".to_string());
        }
        if ranks.contains(&0) || ranks.iter().sum::<usize>() != columns || bits.len() != ranks.len() {
            return Err(format!("{} bits for blocks {ranks:?} of {columns} columns", bits.len()));
        }
        let uf = fast_ab(&library.u, fisher);
        let (_, starts) = partition(ranks);
        Ok(Self {
            v: library.v.clone(),
            ufw: fast_ab(&uf, w),
            wfw: fast_ab(&fast_atb(w, fisher), w),
            k: fast_abt(&uf, &library.u),
            starts,
            bits: bits.to_vec(),
            weight: observations / (2.0 * LN_2),
        })
    }

    /// Each input's blocks on (inputs × blocks, 0 or 1) from its read (`reads`, inputs × d_in).
    pub fn select(&self, reads: &Array2<f64>) -> Array2<f64> {
        let a = fast_abt(reads, &self.v);
        let g = fast_abt(reads, &self.ufw);
        // `K` is symmetric: row `t` of `a K` is `K a_t`, the reads all on.
        let p = fast_ab(&a, &self.k);
        let xq = fast_ab(reads, &self.wfw);
        let gram = Gram { k: &self.k, starts: &self.starts, bits: &self.bits, weight: self.weight };
        let blocks = self.bits.len();
        let sets: Vec<Vec<u8>> = (0..reads.nrows())
            .into_par_iter()
            .map(|t| {
                let (at, gt, pt) = (a.row(t), g.row(t), p.row(t));
                let yfy = reads.row(t).dot(&xq.row(t));
                let mut input = Input { a: at.to_vec(), g: gt.to_vec(), yfy, h: (&gt - &pt).to_vec(), error: yfy - 2.0 * at.dot(&gt) + at.dot(&pt) };
                let mut m = vec![1u8; blocks];
                select(&gram, &mut input, &mut m, true);
                m
            })
            .collect();
        Array2::from_shape_fn((reads.nrows(), blocks), |(t, b)| f64::from(sets[t][b]))
    }
}

/// Block `b`'s factors `(u, v)`, its columns `starts[b]..starts[b + 1]`.
fn block_factors<'a>(library: &'a Library, starts: &[usize], b: usize) -> (ArrayView2<'a, f64>, ArrayView2<'a, f64>) {
    (library.u.slice(s![starts[b]..starts[b + 1], ..]), library.v.slice(s![starts[b]..starts[b + 1], ..]))
}

/// A block's columns bisected by their use (`usage`: inputs × columns, each column's real size on
/// each input): the columns centred and scaled to unit norm over the inputs, split by the sign of
/// their correlation matrix's eigenvector of second largest eigenvalue.
fn bisect(usage: ArrayView2<'_, f32>) -> Result<(Vec<usize>, Vec<usize>), String> {
    let k = usage.ncols();
    let mut profiles = usage.mapv(f64::from);
    for mut column in profiles.columns_mut() {
        let mean = column.mean().unwrap_or(0.0);
        column.mapv_inplace(|x| x - mean);
        let norm = column.dot(&column).sqrt();
        if norm > 0.0 {
            column.mapv_inplace(|x| x / norm);
        }
    }
    let correlation = profiles.t().dot(&profiles);
    let symmetric = (&correlation + &correlation.t()) * 0.5;
    let d = eigh(symmetric.view(), SymmetricAssembly::Mirrored, None).map_err(|e| format!("{e:?}"))?;
    // Eigenvalues ascend: the second largest is at `k − 2`.
    let second = d.vectors.column(k - 2);
    Ok(((0..k).filter(|&j| second[j] >= 0.0).collect(), (0..k).filter(|&j| second[j] < 0.0).collect()))
}

/// A library gated in blocks ([`blocks`]): its columns, their partition into blocks (column runs),
/// and per input the blocks on.
#[derive(Clone, Debug)]
pub struct Blocked {
    pub library: Library,
    pub ranks: Vec<usize>,
    pub sets: Vec<Vec<u32>>,
}

/// The blocks of `library` (module note, "Blocks") chosen by the site's code, moving both ways
/// from the partition `start` (column runs; every column its own block is the fine start, the
/// whole site one block the coarse one). Inside the search a block is priced by its cheap price
/// ([`Describe::cheap`]) on the description's own scale (the median `exact / cheap` over `√B` of the
/// starting blocks), exactly where the description has no cheap price:
///
/// * **Merges.** Each block proposes its most co-firing partner (largest Jaccard overlap of the
///   inputs it runs on, every pair counted by one product of the sets); a round's disjoint merges,
///   best first by the description they save where both run (`bits(a) + bits(b) − bits(a ∪ b)` per
///   input running both), stand when the code with every input's sets selected again falls,
///   halved until they do; a single refused merge is set aside and ends the merges.
/// * **Splits.** Each block of rank ≥ 2 proposes its bisection by how its columns are used: their
///   real sizes over the inputs, centred and scaled, split by the sign of the correlation matrix's
///   second eigenvector (the leading one is their common scale); a round's splits stand together
///   the same way, and a single refused split is set aside and ends the splits.
///
/// Rounds of both until neither changes the partition; then every block is priced exactly and
/// every input's sets are selected again under those prices. Columns only move, so the map is
/// unchanged.
pub fn blocks(
    site: usize,
    w: &Array2<f64>,
    samples: &Samples,
    describe: &dyn Describe,
    observations: f64,
    library: &Library,
    start: &[usize],
) -> Result<(Blocked, Round), String> {
    if library.mean.iter().any(|m| *m != 0.0) {
        return Err("a blocked library reads the uncentred input".to_string());
    }
    let columns = library.v.nrows();
    if start.contains(&0) || start.iter().sum::<usize>() != columns {
        return Err(format!("site {site}: blocks {start:?} do not partition {columns} subcomponents"));
    }
    let fitting = Fitting::new(site, w, samples, observations, columns)?;
    let rows = fitting.rows();
    let starts_of = |ranks: &[usize]| -> Vec<usize> { partition(ranks).1 };
    let exact = |(u, v): (ArrayView2<'_, f64>, ArrayView2<'_, f64>)| gam_linalg::faer_ndarray::with_nested_parallel(|| describe.bits(site, u, v));
    let mut current = library.clone();
    let mut ranks = start.to_vec();
    let starts = starts_of(&ranks);
    // The cheap price's scale, from an evenly strided sample of the starting blocks.
    let stride = ((ranks.len() as f64).sqrt() as usize).max(1);
    let sample: Vec<(f64, Option<f64>)> = (0..ranks.len())
        .step_by(stride)
        .collect::<Vec<_>>()
        .into_par_iter()
        .map(|b| {
            let (u, v) = block_factors(&current, &starts, b);
            Ok((exact((u, v))?, describe.cheap(site, u, v)?))
        })
        .collect::<Result<_, String>>()?;
    let scale = median_ratio(sample.iter().filter_map(|(e, c)| c.map(|c| (*e, c))));
    let price = |(u, v): (ArrayView2<'_, f64>, ArrayView2<'_, f64>)| -> Result<f64, String> {
        match describe.cheap(site, u, v)? {
            Some(cheap) => Ok(cheap * scale),
            None => exact((u, v)),
        }
    };
    let mut bits: Vec<f64> = (0..ranks.len()).into_par_iter().map(|b| price(block_factors(&current, &starts, b))).collect::<Result<_, String>>()?;
    let mut masks = vec![1u8; rows * ranks.len()];
    let (d, e) = fitting.code_blocks(&current.v, &current.u, &ranks, &bits, &mut masks, true);
    let mut total = d + e;
    // A refused merge or split, by its blocks' first reads (bit patterns) and ranks, so it is not
    // proposed again.
    let mut refused: std::collections::BTreeSet<(Vec<u64>, Vec<u64>)> = std::collections::BTreeSet::new();
    let mut refused_splits: std::collections::BTreeSet<(Vec<u64>, usize)> = std::collections::BTreeSet::new();
    let id = |library: &Library, start: usize| -> Vec<u64> { library.v.row(start).iter().map(|x| x.to_bits()).collect() };
    // `library` with the blocks of `groups` (each a list of block indices, its columns in that
    // order, every block in exactly one group) made one block each: the library, ranks and masks
    // (a group runs where any of its blocks ran).
    let regroup = |library: &Library, ranks: &[usize], masks: &[u8], groups: &[Vec<usize>]| -> (Library, Vec<usize>, Vec<u8>) {
        let blocks = ranks.len();
        let starts = starts_of(ranks);
        let order: Vec<usize> = groups.iter().flat_map(|g| g.iter().flat_map(|&b| starts[b]..starts[b + 1])).collect();
        let merged = Library { v: library.v.select(Axis(0), &order), u: library.u.select(Axis(0), &order), mean: library.mean.clone() };
        let new_ranks: Vec<usize> = groups.iter().map(|g| g.iter().map(|b| ranks[*b]).sum()).collect();
        let width = groups.len();
        let mut new_masks = vec![0u8; rows * width];
        new_masks.par_chunks_mut(width).enumerate().for_each(|(t, row)| {
            for (j, g) in groups.iter().enumerate() {
                row[j] = g.iter().map(|b| masks[t * blocks + b]).max().unwrap_or(0);
            }
        });
        (merged, new_ranks, new_masks)
    };
    loop {
        let mut changed = false;
        // Merges.
        loop {
            let blocks = ranks.len();
            let starts = starts_of(&ranks);
            let gates = Array2::from_shape_fn((rows, blocks), |(t, b)| f32::from(masks[t * blocks + b]));
            let shared = product(gates.view(), true, gates.view(), false);
            drop(gates);
            let partners: Vec<Option<usize>> = (0..blocks)
                .into_par_iter()
                .map(|a| {
                    let fired = f64::from(shared[[a, a]]);
                    let mut best: Option<(f64, usize)> = None;
                    for (b, n) in shared.row(a).iter().enumerate() {
                        let n = f64::from(*n);
                        if b == a || n <= 0.0 {
                            continue;
                        }
                        let jaccard = n / (fired + f64::from(shared[[b, b]]) - n);
                        if best.is_none_or(|(j, _)| jaccard > j) {
                            best = Some((jaccard, b));
                        }
                    }
                    best.map(|(_, b)| b)
                })
                .collect();
            let mut pairs: Vec<(usize, usize)> = partners.iter().enumerate().filter_map(|(a, b)| b.map(|b| (a.min(b), a.max(b)))).collect();
            pairs.sort_unstable();
            pairs.dedup();
            pairs.retain(|(a, b)| !refused.contains(&(id(&current, starts[*a]), id(&current, starts[*b]))));
            let mut candidates: Vec<(f64, usize, usize, f64)> = pairs
                .into_par_iter()
                .map(|(a, b)| {
                    let (ua, va) = block_factors(&current, &starts, a);
                    let (ub, vb) = block_factors(&current, &starts, b);
                    let u = ndarray::concatenate(Axis(0), &[ua, ub]).map_err(|e| e.to_string())?;
                    let v = ndarray::concatenate(Axis(0), &[va, vb]).map_err(|e| e.to_string())?;
                    let merged = price((u.view(), v.view()))?;
                    Ok((f64::from(shared[[a, b]]) * (bits[a] + bits[b] - merged), a, b, merged))
                })
                .collect::<Result<Vec<_>, String>>()?;
            candidates.retain(|c| c.0 > 0.0);
            candidates.sort_by(|x, y| y.0.total_cmp(&x.0));
            let mut used = std::collections::BTreeSet::new();
            candidates.retain(|(_, a, b, _)| {
                let free = !used.contains(a) && !used.contains(b);
                if free {
                    used.extend([*a, *b]);
                }
                free
            });
            let mut take = candidates.len();
            let mut kept = false;
            while take > 0 {
                let partner: std::collections::BTreeMap<usize, (usize, f64)> = candidates[..take].iter().map(|(_, a, b, m)| (*a, (*b, *m))).collect();
                let absorbed: std::collections::BTreeSet<usize> = candidates[..take].iter().map(|c| c.2).collect();
                let mut groups = Vec::new();
                let mut trial_bits = Vec::new();
                for a in (0..blocks).filter(|a| !absorbed.contains(a)) {
                    match partner.get(&a) {
                        Some(&(b, merged)) => {
                            groups.push(vec![a, b]);
                            trial_bits.push(merged);
                        }
                        None => {
                            groups.push(vec![a]);
                            trial_bits.push(bits[a]);
                        }
                    }
                }
                let (trial, trial_ranks, mut trial_masks) = regroup(&current, &ranks, &masks, &groups);
                let (d, e) = fitting.code_blocks(&trial.v, &trial.u, &trial_ranks, &trial_bits, &mut trial_masks, true);
                log::info!("site {site}: {take} merges, code {:.1} -> {:.1} bits per input", total / rows as f64, (d + e) / rows as f64);
                if d + e < total {
                    (current, ranks, bits, masks, total) = (trial, trial_ranks, trial_bits, trial_masks, d + e);
                    kept = true;
                    break;
                }
                if take == 1 {
                    let (_, a, b, _) = candidates[0];
                    refused.insert((id(&current, starts[a]), id(&current, starts[b])));
                }
                take /= 2;
            }
            if !kept {
                break;
            }
            changed = true;
        }
        // Splits: every block of rank ≥ 2 bisected by its columns' use, a round's together.
        let usage = fitting.usage(&current.v, &current.u);
        let starts = starts_of(&ranks);
        let proposed: Vec<(usize, Option<(Vec<usize>, Vec<usize>, f64, f64)>)> = (0..ranks.len())
            .filter(|&c| ranks[c] >= 2 && !refused_splits.contains(&(id(&current, starts[c]), ranks[c])))
            .collect::<Vec<_>>()
            .into_par_iter()
            .map(|c| {
                let (first, second) = bisect(usage.slice(s![.., starts[c]..starts[c + 1]]))?;
                if first.is_empty() || second.is_empty() {
                    return Ok((c, None));
                }
                let half = |cols: &[usize]| -> Result<f64, String> {
                    let order: Vec<usize> = cols.iter().map(|j| starts[c] + j).collect();
                    price((current.u.select(Axis(0), &order).view(), current.v.select(Axis(0), &order).view()))
                };
                let (a, b) = (half(&first)?, half(&second)?);
                Ok((c, Some((first, second, a, b))))
            })
            .collect::<Result<_, String>>()?;
        drop(usage);
        let mut splits = Vec::new();
        for (c, proposal) in proposed {
            match proposal {
                Some(p) => splits.push((c, p)),
                None => {
                    refused_splits.insert((id(&current, starts[c]), ranks[c]));
                }
            }
        }
        let mut take = splits.len();
        while take > 0 {
            let chosen: std::collections::BTreeMap<usize, &(Vec<usize>, Vec<usize>, f64, f64)> = splits[..take].iter().map(|(c, p)| (*c, p)).collect();
            // Every block a run of columns in the new order; a split block's halves both on where it was.
            let (mut order, mut trial_ranks, mut trial_bits, mut sources) = (Vec::new(), Vec::new(), Vec::new(), Vec::new());
            for c in 0..ranks.len() {
                match chosen.get(&c) {
                    Some((first, second, a, b)) => {
                        for (half, bits_half) in [(first, a), (second, b)] {
                            order.extend(half.iter().map(|j| starts[c] + j));
                            trial_ranks.push(half.len());
                            trial_bits.push(*bits_half);
                            sources.push(c);
                        }
                    }
                    None => {
                        order.extend(starts[c]..starts[c + 1]);
                        trial_ranks.push(ranks[c]);
                        trial_bits.push(bits[c]);
                        sources.push(c);
                    }
                }
            }
            let trial = Library { v: current.v.select(Axis(0), &order), u: current.u.select(Axis(0), &order), mean: current.mean.clone() };
            let (blocks, width) = (ranks.len(), trial_ranks.len());
            let mut trial_masks = vec![0u8; rows * width];
            trial_masks.par_chunks_mut(width).enumerate().for_each(|(t, row)| {
                for (j, source) in sources.iter().enumerate() {
                    row[j] = masks[t * blocks + source];
                }
            });
            let (d, e) = fitting.code_blocks(&trial.v, &trial.u, &trial_ranks, &trial_bits, &mut trial_masks, true);
            log::info!("site {site}: {take} splits, code {:.1} -> {:.1} bits per input", total / rows as f64, (d + e) / rows as f64);
            if d + e < total {
                (current, ranks, bits, masks, total) = (trial, trial_ranks, trial_bits, trial_masks, d + e);
                changed = true;
                break;
            }
            if take == 1 {
                let c = splits[0].0;
                refused_splits.insert((id(&current, starts[c]), ranks[c]));
            }
            take /= 2;
        }
        if !changed {
            break;
        }
    }
    // Every block at its exact price, every input's sets selected again under them.
    let starts = starts_of(&ranks);
    let bits: Vec<f64> = (0..ranks.len()).into_par_iter().map(|b| exact(block_factors(&current, &starts, b))).collect::<Result<_, String>>()?;
    let blocks = ranks.len();
    let (description, error) = fitting.code_blocks(&current.v, &current.u, &ranks, &bits, &mut masks, true);
    let report = fitting.report(0, description, error);
    let on = masks.iter().filter(|m| **m == 1).count() as f64;
    let report = Round { l0: on / rows as f64, ..report };
    let sets = masks.chunks(blocks).map(|m| (0..blocks as u32).filter(|c| m[*c as usize] == 1).collect()).collect();
    Ok((Blocked { library: current, ranks, sets }, report))
}

/// `(M^{1/2}, M^{+1/2})` of a symmetric positive semidefinite matrix summed from `terms` single-
/// precision products, over its eigenvalues beyond their band (as [`pseudo_inverse`] drops them).
fn roots(m: &Array2<f64>, terms: usize) -> Result<(Array2<f64>, Array2<f64>), String> {
    let symmetric = (m + &m.t()) * 0.5;
    let d = eigh(symmetric.view(), SymmetricAssembly::Mirrored, None).map_err(|e| format!("{e:?}"))?;
    let largest = d.values.iter().fold(0.0_f64, |m, l| m.max(*l));
    let floor = (largest * (terms as f64).sqrt() * f64::from(f32::EPSILON) / 2.0).max(d.band);
    let (mut half, mut inverse) = (d.vectors.clone(), d.vectors.clone());
    for (k, l) in d.values.iter().enumerate() {
        let (h, i) = if *l > floor { (l.sqrt(), 1.0 / l.sqrt()) } else { (0.0, 0.0) };
        half.column_mut(k).mapv_inplace(|x| x * h);
        inverse.column_mut(k).mapv_inplace(|x| x * i);
    }
    Ok((half.dot(&d.vectors.t()), inverse.dot(&d.vectors.t())))
}

/// The empirical variational-Bayes estimate of a singular value `gamma` of an `l × m` matrix
/// observed under Gaussian noise of variance `sigma2` (Nakajima et al., the fully observed matrix
/// factorization's global analytic solution): zero below `σ(√l + √m)`, where no non-zero local
/// solution exists, else `γ/2 (1 − (l+m)σ²/γ² + √((1 − (l+m)σ²/γ²)² − 4 l m σ⁴/γ⁴))`.
fn evb_singular(gamma: f64, l: f64, m: f64, sigma2: f64) -> f64 {
    if gamma <= sigma2.sqrt() * (l.sqrt() + m.sqrt()) {
        return 0.0;
    }
    let a = 1.0 - (l + m) * sigma2 / (gamma * gamma);
    let b = a * a - 4.0 * l * m * sigma2 * sigma2 / gamma.powi(4);
    if b < 0.0 { 0.0 } else { 0.5 * gamma * (a + b.sqrt()) }
}

/// One group's refit by evidence (module note, "Blocks"): on its inputs `members`, with targets
/// `targets` (members × d_out, what they need of it) and reads `reads` (members × d_in), each
/// weighted by its sensitivity `weights`, the least-squares map in the written Fisher's root
/// `f_half` and the reads' own span: with `√s X = Q R` (thin), the target `Y = F^{1/2} (√s T)ᵀ Q`
/// observed with noise of variance `1/n`; its singular values take their empirical
/// variational-Bayes estimates ([`evb_singular`]) and its rank is the one whose kept directions save
/// the most KL bits, `n/(2 ln 2) Σ_j (γ_j² − (γ_j − γ̂_j)²)`, over their description on every member
/// (up to that net's first peak). Returns the group's rows `(u, v)`, `None` at rank zero.
fn evidence_group(
    site: usize,
    describe: &dyn Describe,
    observations: f64,
    (f_half, f_inverse): (&Array2<f64>, &Array2<f64>),
    targets: &Array2<f32>,
    reads: &Array2<f32>,
    weights: &[f32],
) -> Result<Option<(Array2<f64>, Array2<f64>)>, String> {
    let members = reads.nrows();
    let root = |m: &Array2<f32>| {
        let mut out = m.mapv(f64::from);
        for (mut row, w) in out.outer_iter_mut().zip(weights) {
            row.mapv_inplace(|x| x * f64::from(*w).sqrt());
        }
        out
    };
    let (x, t) = (root(reads), root(targets));
    let decomposed = qr(x.view(), QrMode::Economic).map_err(|e| format!("{e:?}"))?;
    let q = decomposed.q.ok_or("a thin QR without Q")?;
    let r = decomposed.r;
    let y = f_half.dot(&t.t().dot(&q));
    let d = svd(y.view(), false).map_err(|e| format!("{e:?}"))?;
    let (l, m) = (y.nrows().min(y.ncols()) as f64, y.nrows().max(y.ncols()) as f64);
    let shrunk: Vec<f64> = d.singular_values.iter().map(|g| evb_singular(*g, l, m, 1.0 / observations)).collect();
    // `V` from `R V = Q₂ √Γ̂` (least norm), `U = (F^{1/2})⁺ P √Γ̂`.
    let r_inverse = pseudo_inverse(&r.t().dot(&r), members)?.dot(&r.t());
    let factors = |rank: usize| -> (Array2<f64>, Array2<f64>) {
        let roots = Array1::from_iter((0..rank).map(|i| shrunk[i].sqrt()));
        let u = (f_inverse.dot(&d.u.slice(s![.., ..rank])) * &roots).t().to_owned();
        let v = (r_inverse.dot(&d.vt.slice(s![..rank, ..]).t()) * &roots).t().to_owned();
        (u, v)
    };
    let scale = observations / (2.0 * LN_2);
    let (mut best, mut gain) = ((0.0, 0usize), 0.0);
    for j in 0..shrunk.len() {
        if shrunk[j] <= 0.0 {
            break;
        }
        gain += d.singular_values[j].powi(2) - (d.singular_values[j] - shrunk[j]).powi(2);
        let (u, v) = factors(j + 1);
        let net = scale * gain - members as f64 * describe.bits(site, u.view(), v.view())?;
        if net > best.0 {
            best = (net, j + 1);
        } else if best.1 > 0 {
            break;
        }
    }
    Ok((best.1 > 0).then(|| factors(best.1)))
}

/// A site's library as blocks whose number and ranks follow from its code by evidence (module
/// note, "Blocks"): up to `capacity` groups, each a block of any rank, from `start` (each group's
/// columns, a partition of `library`'s; the rest empty), alternating
///
/// * **Gates.** Every input's groups on, selected by its code ([`select`] over the groups' real
///   sizes and descriptions).
/// * **Groups.** Each group in turn refitted by evidence ([`evidence_group`]) on the inputs it runs
///   on, to what they need of it: the map less every other group on there (backfitting). Unused
///   directions are exactly zero, so a group's rank falls out of its refit, and a group of rank
///   zero is gone.
/// * **Growth.** Every empty group is fitted by evidence to the inputs of largest error, an equal
///   share of the inputs each, their targets what the groups on there leave of the map.
///
/// The decomposition of least code over the rounds is kept; the rounds stop when one lowers the
/// code by less than one bit per input, or at `rounds`.
pub fn ard(
    site: usize,
    w: &Array2<f64>,
    samples: &Samples,
    describe: &dyn Describe,
    observations: f64,
    (library, start, capacity): (&Library, &[usize], usize),
    rounds: usize,
) -> Result<(Blocked, Round), String> {
    if start.contains(&0) || start.iter().sum::<usize>() != library.v.nrows() || start.len() > capacity {
        return Err(format!("site {site}: groups {start:?} do not partition {} subcomponents within {capacity}", library.v.nrows()));
    }
    let (d_out, d_in) = w.dim();
    let x = &samples.reads;
    let rows = x.nrows();
    let (f_half, f_inverse) = roots(&samples.fisher, rows)?;
    let mut groups: Vec<(Array2<f64>, Array2<f64>)> = Vec::new();
    let mut at = 0;
    for r in start {
        groups.push((library.u.slice(s![at..at + r, ..]).to_owned(), library.v.slice(s![at..at + r, ..]).to_owned()));
        at += r;
    }
    let assemble = |groups: &[(Array2<f64>, Array2<f64>)]| -> Result<(Library, Vec<usize>), String> {
        let us: Vec<_> = groups.iter().map(|g| g.0.view()).collect();
        let vs: Vec<_> = groups.iter().map(|g| g.1.view()).collect();
        let library = Library {
            u: ndarray::concatenate(Axis(0), &us).map_err(|e| e.to_string())?,
            v: ndarray::concatenate(Axis(0), &vs).map_err(|e| e.to_string())?,
            mean: Array1::zeros(d_in),
        };
        Ok((library, groups.iter().map(|g| g.0.nrows()).collect()))
    };
    let contribution = |g: &(Array2<f64>, Array2<f64>)| -> Array2<f32> {
        let a = product(x.view(), false, single(&g.1).view(), true);
        product(a.view(), false, single(&g.0).view(), false)
    };
    let y32 = product(x.view(), false, single(w).view(), true);
    let fit_on = |members: &[usize], targets: Array2<f32>| -> Result<Option<(Array2<f64>, Array2<f64>)>, String> {
        let reads = x.select(Axis(0), members);
        let weights: Vec<f32> = members.iter().map(|t| samples.sensitivity[*t] as f32).collect();
        evidence_group(site, describe, observations, (&f_half, &f_inverse), &targets, &reads, &weights)
    };
    let mut masks = vec![1u8; rows * groups.len()];
    let mut best: Option<(Blocked, Round)> = None;
    let mut previous = f64::INFINITY;
    for round in 0..rounds.max(1) {
        // Gates.
        let (current, ranks) = assemble(&groups)?;
        let fitting = Fitting::new(site, w, samples, observations, current.v.nrows())?;
        let bits: Vec<f64> = groups.par_iter().map(|(u, v)| describe.bits(site, u.view(), v.view())).collect::<Result<_, String>>()?;
        let (description, error) = fitting.code_blocks(&current.v, &current.u, &ranks, &bits, &mut masks, true);
        let count = groups.len();
        let on = masks.iter().filter(|m| **m == 1).count() as f64;
        let report = Round { round, code: (description + error) / rows as f64, description: description / rows as f64, error: error / rows as f64, l0: on / rows as f64, reseeded: 0, read_steps: 0 };
        let mut histogram = std::collections::BTreeMap::<usize, usize>::new();
        for r in &ranks {
            *histogram.entry(*r).or_default() += 1;
        }
        log::info!("site {site} round {round}: {count} groups of ranks {histogram:?}, code {:.1} bits per input (description {:.1}, error {:.1}), {:.2} on", report.code, report.description, report.error, report.l0);
        if best.as_ref().is_none_or(|(_, b)| report.code < b.code) {
            let sets = masks.chunks(count).map(|m| (0..count as u32).filter(|c| m[*c as usize] == 1).collect()).collect();
            best = Some((Blocked { library: current, ranks, sets }, report.clone()));
        }
        if previous - report.code < 1.0 {
            break;
        }
        previous = report.code;
        // Groups, in turn, against every other group on (backfitting).
        let mut on_sum = Array2::<f32>::zeros((rows, d_out));
        let contributions: Vec<Array2<f32>> = groups.iter().map(contribution).collect();
        for (k, z) in contributions.iter().enumerate() {
            for t in (0..rows).filter(|t| masks[t * count + k] == 1) {
                let mut row = on_sum.row_mut(t);
                row += &z.row(t);
            }
        }
        let mut next: Vec<Option<(Array2<f64>, Array2<f64>)>> = Vec::with_capacity(count);
        for (k, z) in contributions.iter().enumerate() {
            let members: Vec<usize> = (0..rows).filter(|t| masks[t * count + k] == 1).collect();
            if members.is_empty() {
                next.push(None);
                continue;
            }
            let targets = y32.select(Axis(0), &members) - &on_sum.select(Axis(0), &members) + &z.select(Axis(0), &members);
            let fresh = fit_on(&members, targets)?;
            let fresh_z = fresh.as_ref().map(contribution);
            for &t in &members {
                let mut row = on_sum.row_mut(t);
                row -= &z.row(t);
                if let Some(f) = &fresh_z {
                    row += &f.row(t);
                }
            }
            next.push(fresh);
        }
        let kept: Vec<usize> = (0..count).filter(|k| next[*k].is_some()).collect();
        let mut grown: Vec<(Array2<f64>, Array2<f64>)> = next.into_iter().flatten().collect();
        // Growth: the empty groups from the inputs of largest error, an equal share each.
        let empty = capacity.saturating_sub(grown.len());
        if empty > 0 {
            let left = &y32 - &on_sum;
            let lf = product(left.view(), false, single(&samples.fisher).view(), false);
            let mut errors: Vec<(f64, usize)> = (0..rows)
                .map(|t| (samples.sensitivity[t] * left.row(t).iter().zip(lf.row(t).iter()).map(|(a, b)| f64::from(*a) * f64::from(*b)).sum::<f64>(), t))
                .collect();
            errors.sort_by(|a, b| b.0.total_cmp(&a.0));
            let share = (rows / capacity).max(1);
            for chunk in errors.chunks(share).take(empty) {
                let members: Vec<usize> = chunk.iter().map(|(_, t)| *t).collect();
                if let Some(g) = fit_on(&members, left.select(Axis(0), &members))? {
                    grown.push(g);
                }
            }
        }
        if grown.is_empty() {
            break;
        }
        // The gates of the groups kept carry over; a grown group starts on everywhere.
        let width = grown.len();
        let mut carried = vec![1u8; rows * width];
        for t in 0..rows {
            for (j, &k) in kept.iter().enumerate() {
                carried[t * width + j] = masks[t * count + k];
            }
        }
        groups = grown;
        masks = carried;
    }
    best.ok_or_else(|| "no round".to_string())
}
