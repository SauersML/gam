//! A site's library fitted on its own inputs (#2951).
//!
//! The masked program (`super::masked`) codes each input by the subcomponents that run on it,
//! `Σ_{c on} bits(c) + n KL / ln 2`, under the box claim: an off subcomponent's gate may be anywhere
//! in `[0, 1]`. Its libraries can come from VPD, from the map's own singular pieces, or from this
//! module: a library fitted on a site's inputs to that one total, each input judged against the
//! site's real computation on it.
//!
//! # The site's code
//!
//! A site `y = W x` with library `y ≈ Σ_c u_c (v_c · x)` runs on input `t` with its real read `x_t`,
//! so every subcomponent's real contribution `z_tc = u_c a_tc` (`a_tc = v_c · x_t`, the read
//! uncentred as the masked program reads it) is known, and so is what all on leaves of the map,
//! `D_t = W x_t − Σ_c z_tc` (zero for a library whose subcomponents sum to the map). With the off
//! subcomponents' gates anywhere in `[0, 1]` the site's output misses `D_t + Σ_{c off} (1 − m_c)
//! z_tc`, so by the triangle inequality every point of the box is within
//!
//! ```text
//! B_t = ‖D_t‖_{F_t} + Σ_{c off} ‖z_tc‖_{F_t}
//! ```
//!
//! of the real output, in the input's own output Fisher `F_t = mean_k g_tk g_tkᵀ`, its sampled-label
//! gradients at the written value (labels drawn from the model's own output; without them, the
//! site's mean Fisher `F` scaled by the input's sensitivity `s_t = tr F_t / tr F`). Its KL is at
//! most `½ B_t²` to second order, so input `t` is charged
//!
//! ```text
//! code_t = Σ_{c on} bits(c) + n / (2 ln 2) · (‖D_t‖_{F_t} + Σ_{c off} |a_tc| ‖u_c‖_{F_t})².
//! ```
//!
//! It is a certified upper bound on the claim at the site, needs no adversary, and counts every
//! off subcomponent at its own real size: two that cancel are each large, so they cannot hide
//! among the off ones, and splitting a subcomponent into copies changes nothing.
//! `bits(c)` is the subcomponent's description ([`super::blocks::Describe`]), paid on every input
//! it runs on. Nothing else enters: no library is amortised.
//!
//! # The fit
//!
//! Alternating steps that each lower that one total:
//!
//! * **Sets.** Each input's on-set from its subcomponents ranked by real size per description bit,
//!   `|a_tc| ‖u_c‖_{F_t} / bits(c)`: the best prefix of that ranking when it codes the input in fewer
//!   bits than its current sets, then single flips swept until none lowers the input's code.
//! * **Writes.** With the sets and reads fixed the error is convex in `U`. With `L_t` the current
//!   `B_t`, `(Σ_c α_c)² ≤ Σ_c α_c² L/α_c` majorises it, tight at the current writes. Each input's
//!   metric is taken at its ratio to `F` for the current write, `‖u'‖²_{F_t} ≈ (‖u_c‖_{F_t} /
//!   ‖u_c‖_F)² ‖u'‖²_F`, so the majoriser is `Σ_c ω_c ‖u_c‖²_F`, `ω_c = Σ_{t: c off} n L_t |a_tc|
//!   ‖u_c‖_{F_t} / (2 ln 2 ‖u_c‖²_F)`, and its minimiser is a proposal the code decides.
//!   Every subcomponent on is the map exactly, on every read direction (in the metric `E√Λ` of the
//!   reads' second moment, floored at `10⁻⁶` of its mean so directions no input reached still
//!   count), `Uᵀ V E√Λ = W E√Λ`: with `V E√Λ = P S Gᵀ` (full `P`) every such `U` is `U₀ + N Z`,
//!   `U₀ = P₁ S⁻¹ Gᵀ (W E√Λ)ᵀ`, `N` the columns of `P` past its rank, and `(NᵀΩN) Z = −NᵀΩU₀`.
//! * **Reads.** With `U` and the sets fixed the error is convex in `V`: one step along its
//!   preconditioned negative subgradient (left by each subcomponent's own curvature `Σ_t n
//!   ‖u_c‖²_{F_t}`, right by the inputs' sensitivity-weighted second moment) restricted to the moves
//!   that keep the map, `Uᵀ D = 0`, its length the one of least code on a geometric ladder around
//!   the step's quadratic estimate; every input's `B_t` along the line is measured in one pass.
//! * **Growth.** A subcomponent that runs nowhere or writes nothing takes half of the one that
//!   carries the most error where it is off, split along that one's inputs (at most a chunk of
//!   them, evenly strided; the halves sum to it);
//!   kept when the sets selected with them code the inputs in fewer bits.
//!
//! # Blocks
//!
//! A fitted library can be gated in blocks ([`blocks`]): column runs that one gate runs or drops
//! whole, block `c`'s real contribution on input `t` `U_c a_tc` of size `‖U_c a_tc‖_{F_t}` in the
//! bound above and its description `bits(c)` of the whole block. Merging two blocks that run
//! together pays one description where both run, and where both are off counts `‖z_a + z_b‖`, at
//! most `‖z_a‖ + ‖z_b‖`; it costs the whole block where only one is needed. The same code decides.
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
    /// Per draw, every input's sampled-label gradient at the written value (inputs × d_out,
    /// single precision): input `t`'s own Fisher is `F_t = mean_k g_tk g_tkᵀ`.
    pub gradients: Vec<Array2<f32>>,
}

/// Every site's [`Samples`] on the native program over `batches`, `draws` sampled-label reverse
/// passes per batch (labels drawn from the program's own output).
pub fn samples(program: &OperatorProgram, sites: &[Site], batches: impl IntoIterator<Item = FamilyInputs>, draws: usize, seed: u64) -> Result<Vec<Samples>, String> {
    if draws == 0 {
        return Err("samples need at least one draw".to_string());
    }
    let mut reads: Vec<Vec<Array2<f32>>> = vec![Vec::new(); sites.len()];
    let mut norms: Vec<Vec<f64>> = vec![Vec::new(); sites.len()];
    // Per site and draw, every batch's gradients.
    let mut drawn_gradients: Vec<Vec<Vec<Array2<f32>>>> = vec![vec![Vec::new(); draws]; sites.len()];
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
        for draw in 0..draws {
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
                drawn_gradients[k][draw].push(g.mapv(|v| v as f32));
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
        .zip(drawn_gradients)
        .map(|(((parts, norms), (fisher, moment)), per_draw)| {
            let stack = |parts: &[Array2<f32>]| -> Result<Array2<f32>, String> {
                let views: Vec<_> = parts.iter().map(|p| p.view()).collect();
                ndarray::concatenate(Axis(0), &views).map_err(|e| e.to_string())
            };
            let reads = stack(&parts)?;
            let gradients = per_draw.iter().map(|parts| stack(parts)).collect::<Result<Vec<_>, _>>()?;
            let fisher = fisher.ok_or("no Fisher")? / (rows * draws) as f64;
            let mean_norm = fisher.diag().sum();
            let sensitivity = Array1::from_iter(norms.iter().map(|n| if mean_norm > 0.0 { n / mean_norm } else { 1.0 }));
            Ok(Samples { reads, sensitivity, fisher, second_moment: moment.ok_or("no moment")? / rows as f64, gradients })
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
    let largest = d.values.iter().fold(0.0_f64, |m, l| m.max(*l));
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
    /// Per draw, every input's gradient (`Samples::gradients`) and its `g_tk · y_t`.
    gradients: &'a [Array2<f32>],
    gy: Vec<Vec<f64>>,
    s: &'a Array1<f64>,
    fisher: &'a Array2<f64>,
    /// `n / (2 ln 2)`.
    scale: f64,
    pieces: usize,
    /// Inputs × pieces, 1 where on.
    masks: Vec<u8>,
}

/// What the inputs' code needs of the library: their reads `a`, every write's size in each input's
/// own Fisher `q_tc = ‖u_c‖_{F_t}`, and what all on leaves, `‖D_t‖_{F_t}`.
struct Chunk {
    a: Array2<f32>,
    q: Array2<f32>,
    left: Vec<f64>,
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
            if held >= (1.0 - LEFT_OUT) * total || moment.values[i] <= moment.band { break; }
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

impl Chunk {
    /// Input `r`'s real contribution sizes `|a_tc| q_tc`.
    fn sizes(&self, r: usize) -> Vec<f64> {
        self.a.row(r).iter().zip(self.q.row(r).iter()).map(|(a, q)| f64::from(*a).abs() * f64::from(*q)).collect()
    }
}

/// One input's sets (module note, "Sets") over subcomponents of real sizes `size` and description
/// `bits`, with `left` what all on leaves: with `flip` the best prefix of the ranking by size per bit
/// when it codes the input in fewer bits than its current sets `m`, then single flips swept until
/// none lowers its code. Returns its description and error bits.
fn select(size: &[f64], bits: &[f64], left: f64, weight: f64, m: &mut [u8], flip: bool) -> (f64, f64) {
    let c_total = size.len();
    if flip {
        let mut order: Vec<usize> = (0..c_total).collect();
        let ratio = |c: usize| match (bits[c] > 0.0, size[c] > 0.0) {
            (true, _) => size[c] / bits[c],
            (false, true) => f64::INFINITY,
            (false, false) => 0.0,
        };
        order.sort_by(|a, b| ratio(*b).total_cmp(&ratio(*a)));
        let all: f64 = size.iter().sum();
        let (mut listed, mut bound) = (0.0, left + all);
        let (mut best, mut best_code) = (0, weight * bound * bound);
        for (k, &c) in order.iter().enumerate() {
            listed += bits[c];
            bound -= size[c];
            let code = listed + weight * bound * bound;
            if code < best_code {
                (best, best_code) = (k + 1, code);
            }
        }
        // The prefix replaces the input's current sets only when it codes it in fewer bits.
        let held = left + (0..c_total).filter(|c| m[*c] == 0).map(|c| size[c]).sum::<f64>();
        let held_code = (0..c_total).filter(|c| m[*c] == 1).map(|c| bits[c]).sum::<f64>() + weight * held * held;
        if best_code < held_code {
            m.fill(0);
            for &c in &order[..best] {
                m[c] = 1;
            }
        }
        let mut bound = left + (0..c_total).filter(|c| m[*c] == 0).map(|c| size[c]).sum::<f64>();
        for _ in 0..c_total {
            let mut flipped = false;
            for c in 0..c_total {
                let next = if m[c] == 1 { bound + size[c] } else { (bound - size[c]).max(0.0) };
                let delta = if m[c] == 1 { -bits[c] } else { bits[c] } + weight * (next * next - bound * bound);
                if delta < 0.0 {
                    m[c] = 1 - m[c];
                    bound = next;
                    flipped = true;
                }
            }
            if !flipped {
                break;
            }
        }
    }
    let bound = left + (0..c_total).filter(|c| m[*c] == 0).map(|c| size[c]).sum::<f64>();
    let listed: f64 = (0..c_total).filter(|c| m[*c] == 1).map(|c| bits[c]).sum();
    (listed, weight * bound * bound)
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
        let gy = samples
            .gradients
            .iter()
            .map(|g| (0..rows).map(|t| g.row(t).iter().zip(y.row(t).iter()).map(|(a, b)| f64::from(*a) * f64::from(*b)).sum()).collect())
            .collect();
        if samples.gradients.iter().any(|g| g.dim() != (rows, d_out)) {
            return Err(format!("site {site}: gradients do not fit its {rows} inputs of {d_out} writes"));
        }
        Ok(Self {
            gradients: &samples.gradients,
            gy,
            x,
            w: w.clone(),
            y,
            yfy,
            s: &samples.sensitivity,
            fisher: &samples.fisher,
            scale: observations / (2.0 * LN_2),
            pieces,
            masks: vec![1; rows * pieces],
        })
    }

    fn rows(&self) -> usize {
        self.x.nrows()
    }

    /// What the code of the inputs `start..end` needs (`Chunk`): in each input's own Fisher
    /// `F_t = mean_k g_tk g_tkᵀ` when the samples carry gradients, else in `s_t F`.
    fn chunk(&self, start: usize, end: usize, v32: &Array2<f32>, u32: &Array2<f32>, k32: &Array2<f32>, uf32: &Array2<f32>, sizes: &[f64]) -> Chunk {
        let rows = end - start;
        let a = product(self.x.slice(s![start..end, ..]), false, v32.view(), true);
        if self.gradients.is_empty() {
            let g0 = product(self.y.slice(s![start..end, ..]), false, uf32.view(), true);
            let ak = product(a.view(), false, k32.view(), false);
            let left = (0..rows)
                .into_par_iter()
                .map(|r| {
                    let (mut ag, mut aka) = (0.0f64, 0.0f64);
                    for c in 0..self.pieces {
                        let ac = f64::from(a[[r, c]]);
                        ag += ac * f64::from(g0[[r, c]]);
                        aka += ac * f64::from(ak[[r, c]]);
                    }
                    (self.s[start + r] * (self.yfy[start + r] - 2.0 * ag + aka)).max(0.0).sqrt()
                })
                .collect();
            let q = Array2::from_shape_fn((rows, self.pieces), |(r, c)| (self.s[start + r].sqrt() * sizes[c]) as f32);
            return Chunk { a, q, left };
        }
        let draws = self.gradients.len() as f64;
        let mut q2 = Array2::<f64>::zeros((rows, self.pieces));
        let mut left = vec![0.0f64; rows];
        for (g, gy) in self.gradients.iter().zip(&self.gy) {
            // `g_tk · u_c` for every input and write; the inputs accumulate in parallel.
            let p = product(g.slice(s![start..end, ..]), false, u32.view(), true);
            q2.axis_iter_mut(Axis(0)).into_par_iter().zip(left.par_iter_mut()).enumerate().for_each(|(r, (mut q2_row, left_r))| {
                let mut gd = gy[start + r];
                for c in 0..self.pieces {
                    let pc = f64::from(p[[r, c]]);
                    q2_row[c] += pc * pc / draws;
                    gd -= f64::from(a[[r, c]]) * pc;
                }
                *left_r += gd * gd / draws;
            });
        }
        Chunk { a, q: q2.mapv(|x| x.sqrt() as f32), left: left.into_iter().map(f64::sqrt).collect() }
    }

    /// The products every pass over the inputs needs: `V`, `U`, `K = U F Uᵀ` and `U F` in single
    /// precision, and every `‖u_c‖_F`.
    fn operands(&self, v: &Array2<f64>, u: &Array2<f64>) -> (Array2<f32>, Array2<f32>, Array2<f32>, Array2<f32>, Vec<f64>) {
        let uf = u.dot(self.fisher);
        let k = uf.dot(&u.t());
        let sizes = (0..u.nrows()).map(|c| k[[c, c]].max(0.0).sqrt()).collect();
        (single(v), single(u), single(&k), single(&uf), sizes)
    }

    /// The code of `(v, u)` (module note); with `flip`, every input's sets selected first (module
    /// note, "Sets"). Returns the total description and error bits.
    fn code(&mut self, v: &Array2<f64>, u: &Array2<f64>, bits: &Array1<f64>, flip: bool) -> (f64, f64) {
        let c_total = self.pieces;
        let (v32, u32, k32, uf32, sizes) = self.operands(v, u);
        let bits = bits.to_vec();
        let mut description = 0.0;
        let mut error = 0.0;
        for start in (0..self.rows()).step_by(CHUNK) {
            let end = (start + CHUNK).min(self.rows());
            let chunk = self.chunk(start, end, &v32, &u32, &k32, &uf32, &sizes);
            let masks = &mut self.masks[start * c_total..end * c_total];
            let weight = self.scale;
            let results: Vec<(f64, f64)> =
                masks.par_chunks_mut(c_total).enumerate().map(|(r, m)| select(&chunk.sizes(r), &bits, chunk.left[r], weight, m, flip)).collect();
            for (listed, err) in results {
                description += listed;
                error += err;
            }
        }
        (description, error)
    }

    /// The code of `(v, u)` gated in blocks of `ranks` (column runs, module note "Blocks"): block
    /// `c`'s real contribution on input `t` is `U_c a_tc`, of size `‖U_c a_tc‖_{F_t}`, and one gate
    /// runs or drops it whole. `masks` (inputs × blocks) are the sets, selected first with `flip`.
    fn code_blocks(&self, v: &Array2<f64>, u: &Array2<f64>, ranks: &[usize], bits: &[f64], masks: &mut [u8], flip: bool) -> (f64, f64) {
        let blocks = ranks.len();
        let starts: Vec<usize> = std::iter::once(0).chain(ranks.iter().scan(0, |a, r| {
            *a += r;
            Some(*a)
        })).collect();
        let (v32, u32, k32, uf32, sizes) = self.operands(v, u);
        let (mut description, mut error) = (0.0, 0.0);
        for start in (0..self.rows()).step_by(CHUNK) {
            let end = (start + CHUNK).min(self.rows());
            let rows = end - start;
            let chunk = self.chunk(start, end, &v32, &u32, &k32, &uf32, &sizes);
            // Each block's size: in `F_t = mean_k g_tk g_tkᵀ` the mean of `(Σ_{j ∈ c} a_tj g_tk·u_j)²`,
            // else `s_t aᵀ K_c a`.
            let mut size2 = Array2::<f64>::zeros((rows, blocks));
            if self.gradients.is_empty() {
                size2.axis_iter_mut(Axis(0)).into_par_iter().enumerate().for_each(|(r, mut row)| {
                    for c in 0..blocks {
                        let mut total = 0.0;
                        for i in starts[c]..starts[c + 1] {
                            for j in starts[c]..starts[c + 1] {
                                total += f64::from(chunk.a[[r, i]]) * f64::from(k32[[i, j]]) * f64::from(chunk.a[[r, j]]);
                            }
                        }
                        row[c] = self.s[start + r] * total.max(0.0);
                    }
                });
            } else {
                let draws = self.gradients.len() as f64;
                for g in self.gradients {
                    let p = product(g.slice(s![start..end, ..]), false, u32.view(), true);
                    size2.axis_iter_mut(Axis(0)).into_par_iter().enumerate().for_each(|(r, mut row)| {
                        for c in 0..blocks {
                            let along: f64 = (starts[c]..starts[c + 1]).map(|j| f64::from(chunk.a[[r, j]]) * f64::from(p[[r, j]])).sum();
                            row[c] += along * along / draws;
                        }
                    });
                }
            }
            let weight = self.scale;
            let results: Vec<(f64, f64)> = masks[start * blocks..end * blocks]
                .par_chunks_mut(blocks)
                .enumerate()
                .map(|(r, m)| {
                    let size: Vec<f64> = size2.row(r).iter().map(|x| x.sqrt()).collect();
                    select(&size, bits, chunk.left[r], weight, m, flip)
                })
                .collect();
            for (listed, err) in results {
                description += listed;
                error += err;
            }
        }
        (description, error)
    }

    /// Every column's real size on every input, `|a_tc| ‖u_c‖_{F_t}` (inputs × columns).
    fn usage(&self, v: &Array2<f64>, u: &Array2<f64>) -> Array2<f32> {
        let (v32, u32, k32, uf32, sizes) = self.operands(v, u);
        let mut out = Array2::<f32>::zeros((self.rows(), self.pieces));
        for start in (0..self.rows()).step_by(CHUNK) {
            let end = (start + CHUNK).min(self.rows());
            let chunk = self.chunk(start, end, &v32, &u32, &k32, &uf32, &sizes);
            for r in 0..end - start {
                for (c, size) in chunk.sizes(r).into_iter().enumerate() {
                    out[[start + r, c]] = size as f32;
                }
            }
        }
        out
    }

    /// The writes minimising `Σ_c ω_c ‖u_c‖²_F` under every subcomponent on being the map (module
    /// note, "Writes").
    fn writes_weighted(&self, v: &Array2<f64>, omega: &Array1<f64>, span: &Array2<f64>) -> Result<Array2<f64>, String> {
        // On every read direction (`x = E√Λ ξ`) the constraint is `Uᵀ (V E√Λ) = W E√Λ`.
        let decomposed = svd(v.dot(span).view(), true).map_err(|e| format!("{e:?}"))?;
        let rank = decomposed.singular_values.iter().filter(|s| **s > decomposed.band).count();
        let p1 = decomposed.u.slice(s![.., ..rank]);
        let inverse = Array1::from_iter(decomposed.singular_values.iter().take(rank).map(|s| 1.0 / s));
        let particular = (&p1 * &inverse).dot(&decomposed.vt.slice(s![..rank, ..])).dot(&self.w.dot(span).t());
        let null = decomposed.u.slice(s![.., rank..]).to_owned();
        let weighted = &null * &omega.view().insert_axis(Axis(1));
        let z = pseudo_inverse(&null.t().dot(&weighted), 0)?.dot(&weighted.t().dot(&particular));
        Ok(particular - null.dot(&z))
    }

    /// The writes' majorise-minimise step at `(v, u)` (module note, "Writes").
    fn writes(&self, v: &Array2<f64>, u: &Array2<f64>, span: &Array2<f64>) -> Result<Array2<f64>, String> {
        let c_total = self.pieces;
        let (v32, u32, k32, uf32, sizes) = self.operands(v, u);
        let mut omega = Array1::<f64>::zeros(c_total);
        // A subcomponent writing nothing is held near nothing (its majoriser's weight is bounded at
        // the double-precision resolution of the largest write).
        let floor = sizes.iter().fold(0.0_f64, |m, s| m.max(*s)) * f64::EPSILON.sqrt();
        for start in (0..self.rows()).step_by(CHUNK) {
            let end = (start + CHUNK).min(self.rows());
            let chunk = self.chunk(start, end, &v32, &u32, &k32, &uf32, &sizes);
            for r in 0..end - start {
                let t = start + r;
                let m = &self.masks[t * c_total..(t + 1) * c_total];
                let size = chunk.sizes(r);
                let bound = chunk.left[r] + (0..c_total).filter(|c| m[*c] == 0).map(|c| size[c]).sum::<f64>();
                for c in (0..c_total).filter(|c| m[*c] == 0) {
                    // Each input's own metric taken at its ratio to `F` for the current write.
                    omega[c] += self.scale * bound * size[c] / sizes[c].max(floor).powi(2);
                }
            }
        }
        self.writes_weighted(v, &omega, span)
    }

    /// The reads' step (module note, "Reads"): its direction and the quadratic estimate of its
    /// length, or `None` when no move keeping the map lowers the code.
    fn read_step(&self, v: &Array2<f64>, u: &Array2<f64>, right: &Array2<f64>) -> Result<Option<(Array2<f64>, f64)>, String> {
        let c_total = self.pieces;
        let (v32, u32, k32, uf32, sizes) = self.operands(v, u);
        let mut gradient = Array2::<f32>::zeros(v.dim());
        let mut left = vec![0.0f64; c_total];
        for start in (0..self.rows()).step_by(CHUNK) {
            let end = (start + CHUNK).min(self.rows());
            let chunk = self.chunk(start, end, &v32, &u32, &k32, &uf32, &sizes);
            // `∂ code / ∂ a_tc = 2 n s_t B_t ‖u_c‖_F sign(a_tc)` for every off subcomponent.
            let mut ga = Array2::<f32>::zeros(chunk.a.dim());
            for r in 0..end - start {
                let t = start + r;
                let m = &self.masks[t * c_total..(t + 1) * c_total];
                let size = chunk.sizes(r);
                let bound = chunk.left[r] + (0..c_total).filter(|c| m[*c] == 0).map(|c| size[c]).sum::<f64>();
                for c in (0..c_total).filter(|c| m[*c] == 0) {
                    let q = f64::from(chunk.q[[r, c]]);
                    ga[[r, c]] = (2.0 * self.scale * bound * q * f64::from(chunk.a[[r, c]]).signum()) as f32;
                    left[c] += 2.0 * self.scale * q * q;
                }
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
        // The quadratic estimate: the kinks of `|a|` ignored, `B_t` moves by `Σ_off ‖u_c‖ sign(a) δa`.
        let d32 = single(&direction);
        let mut curvature = 0.0;
        for start in (0..self.rows()).step_by(CHUNK) {
            let end = (start + CHUNK).min(self.rows());
            let chunk = self.chunk(start, end, &v32, &u32, &k32, &uf32, &sizes);
            let da = product(self.x.slice(s![start..end, ..]), false, d32.view(), true);
            for r in 0..end - start {
                let t = start + r;
                let m = &self.masks[t * c_total..(t + 1) * c_total];
                let rate: f64 = (0..c_total)
                    .filter(|c| m[*c] == 0)
                    .map(|c| f64::from(chunk.q[[r, c]]) * f64::from(chunk.a[[r, c]]).signum() * f64::from(da[[r, c]]))
                    .sum();
                curvature += 2.0 * self.scale * rate * rate;
            }
        }
        Ok((slope > 0.0 && curvature > 0.0).then(|| (direction, slope / curvature)))
    }

    /// The error bits of `v + η d` for every `η` of `lengths`, the sets held, in one pass.
    fn read_profile(&self, v: &Array2<f64>, u: &Array2<f64>, d: &Array2<f64>, lengths: &[f64]) -> Vec<f64> {
        let c_total = self.pieces;
        let (v32, u32, k32, uf32, sizes) = self.operands(v, u);
        let d32 = single(d);
        let mut totals = vec![0.0; lengths.len()];
        for start in (0..self.rows()).step_by(CHUNK) {
            let end = (start + CHUNK).min(self.rows());
            // `Uᵀ d = 0`, so what all on leaves does not move along the line.
            let chunk = self.chunk(start, end, &v32, &u32, &k32, &uf32, &sizes);
            let da = product(self.x.slice(s![start..end, ..]), false, d32.view(), true);
            let rows: Vec<Vec<f64>> = (0..end - start)
                .into_par_iter()
                .map(|r| {
                    let t = start + r;
                    let m = &self.masks[t * c_total..(t + 1) * c_total];
                    lengths
                        .iter()
                        .map(|&eta| {
                            let bound = chunk.left[r]
                                + (0..c_total)
                                    .filter(|c| m[*c] == 0)
                                    .map(|c| (f64::from(chunk.a[[r, c]]) + eta * f64::from(da[[r, c]])).abs() * f64::from(chunk.q[[r, c]]))
                                    .sum::<f64>();
                            self.scale * bound * bound
                        })
                        .collect()
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

    /// Per subcomponent, the error bits it carries where it is off, `Σ_{t: c off} n s_t B_t |a_tc|
    /// ‖u_c‖_F / (2 ln 2)` (its share of each input's bound times the bound).
    fn carried(&self, v: &Array2<f64>, u: &Array2<f64>) -> Vec<f64> {
        let c_total = self.pieces;
        let (v32, u32, k32, uf32, sizes) = self.operands(v, u);
        let mut carried = vec![0.0; c_total];
        for start in (0..self.rows()).step_by(CHUNK) {
            let end = (start + CHUNK).min(self.rows());
            let chunk = self.chunk(start, end, &v32, &u32, &k32, &uf32, &sizes);
            for r in 0..end - start {
                let t = start + r;
                let m = &self.masks[t * c_total..(t + 1) * c_total];
                let size = chunk.sizes(r);
                let bound = chunk.left[r] + (0..c_total).filter(|c| m[*c] == 0).map(|c| size[c]).sum::<f64>();
                for c in (0..c_total).filter(|c| m[*c] == 0) {
                    carried[c] += self.scale * bound * size[c];
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

/// A given library's code on `samples` (module note), every input's sets selected, and those sets
/// (per input, the subcomponents on, ascending).
pub fn measure(site: usize, w: &Array2<f64>, samples: &Samples, describe: &dyn Describe, observations: f64, library: &Library) -> Result<(Round, Vec<Vec<u32>>), String> {
    if library.mean.iter().any(|m| *m != 0.0) {
        return Err("a measured library reads the uncentred input".to_string());
    }
    let mut fitting = Fitting::new(site, w, samples, observations, library.v.nrows())?;
    let bits = description_bits(describe, site, &library.v, &library.u)?;
    let (description, error) = fitting.code(&library.v, &library.u, &bits, true);
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
    let bits = description_bits(describe, site, &library.v, &library.u)?;
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
    // The reads' inverse second moment on their span (seeding) and the sensitivity-weighted one
    // (the reads' right preconditioner).
    let seeding = &geometry.seeding;
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
        _ => fitting.writes_weighted(&v, &Array1::ones(pieces), span)?,
    };
    let mut bits = description_bits(describe, site, &v, &u)?;
    let (description, error) = fitting.code(&v, &u, &bits, true);
    let mut current = description + error;
    let mut report = fitting.report(0, description, error);
    for round in 0..rounds {
        // The writes: the majoriser's minimiser, kept when the code falls.
        let trial = fitting.writes(&v, &u, span)?;
        let (d, e) = fitting.code(&v, &trial, &bits, false);
        if d + e < current {
            (u, current) = (trial, d + e);
        }
        // The reads: along the preconditioned step, the length of least code among the quadratic
        // estimate times every power of √2 from 2⁻³² to 2⁴, all measured in one pass.
        if let Some((direction, alpha)) = fitting.read_step(&v, &u, &right)? {
            let mut lengths = vec![0.0];
            lengths.extend((-64..=8).map(|k| alpha * 2f64.powf(f64::from(k) / 2.0)));
            let profile = fitting.read_profile(&v, &u, &direction, &lengths);
            let (best, value) = profile.iter().enumerate().fold((0, profile[0]), |b, (i, p)| if *p < b.1 { (i, *p) } else { b });
            if best > 0 {
                // The sets' description is unchanged; only the error moved.
                report.read_steps = best;
                current += value - profile[0];
                v.scaled_add(lengths[best], &direction);
            }
        }
        // Growth: every subcomponent that runs nowhere or writes nothing takes half of one that
        // carries the most error where it is off, split along its inputs (`super::masked::split`:
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
            bits = description_bits(describe, site, &v, &u)?;
            let (d, e) = fitting.code(&v, &u, &bits, true);
            if grown > 0 && d + e < current {
                report.reseeded = grown;
            } else {
                (v, u, fitting.masks) = saved;
            }
        }
        progress(&report, &Library { v: v.clone(), u: u.clone(), mean: Array1::zeros(d_in) });
        // The next round's sets, under the descriptions as the steps left them.
        bits = description_bits(describe, site, &v, &u)?;
        let (description, error) = fitting.code(&v, &u, &bits, true);
        let previous = report.code;
        report = fitting.report(round + 1, description, error);
        current = description + error;
        if previous - report.code < 1.0 {
            break;
        }
    }
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
/// whole site one block the coarse one):
///
/// * **Merges.** Each block proposes its most co-firing partner (largest Jaccard overlap of the
///   inputs it runs on); a round's disjoint merges, best first by the description they save where
///   both run (`bits(a) + bits(b) − bits(a ∪ b)` per input running both), stand when the code with
///   every input's sets selected again falls, halved until they do or a single refused merge is
///   set aside.
/// * **Splits.** Each block of rank ≥ 2 proposes its bisection by how its columns are used: their
///   real sizes over the inputs, centred and scaled, split by the sign of the correlation matrix's
///   second eigenvector (the leading one is their common scale); kept when the code falls.
///
/// Rounds of both until neither changes the partition. Columns only move, so the map is unchanged.
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
    let block_bits = |library: &Library, ranks: &[usize]| -> Result<Vec<f64>, String> {
        let starts: Vec<usize> = std::iter::once(0).chain(ranks.iter().scan(0, |a, r| {
            *a += r;
            Some(*a)
        })).collect();
        (0..ranks.len())
            .into_par_iter()
            .map(|c| describe.bits(site, library.u.slice(s![starts[c]..starts[c + 1], ..]), library.v.slice(s![starts[c]..starts[c + 1], ..])))
            .collect()
    };
    // The state: library, ranks, per block its columns' ids (to undo merges) and bits, masks.
    let mut current = library.clone();
    let mut ranks = start.to_vec();
    let mut bits = block_bits(&current, &ranks)?;
    let mut masks = vec![1u8; rows * ranks.len()];
    let (d, e) = fitting.code_blocks(&current.v, &current.u, &ranks, &bits, &mut masks, true);
    let mut total = d + e;
    // A refused merge or split, by its blocks' first reads (bit patterns) and ranks, so it is not
    // proposed again.
    let mut refused: std::collections::BTreeSet<(Vec<u64>, Vec<u64>)> = std::collections::BTreeSet::new();
    let mut refused_splits: std::collections::BTreeSet<(Vec<u64>, usize)> = std::collections::BTreeSet::new();
    let id = |library: &Library, start: usize| -> Vec<u64> { library.v.row(start).iter().map(|x| x.to_bits()).collect() };
    let starts_of = |ranks: &[usize]| -> Vec<usize> {
        std::iter::once(0)
            .chain(ranks.iter().scan(0, |a, r| {
                *a += r;
                Some(*a)
            }))
            .collect()
    };
    // `library` with blocks `a < b` merged (b's columns moved after a's) and the merged masks.
    let merge = |library: &Library, ranks: &[usize], masks: &[u8], pairs: &[(usize, usize)]| -> (Library, Vec<usize>, Vec<u8>) {
        let blocks = ranks.len();
        let starts = starts_of(ranks);
        let partner: std::collections::BTreeMap<usize, usize> = pairs.iter().copied().collect();
        let absorbed: std::collections::BTreeSet<usize> = pairs.iter().map(|p| p.1).collect();
        let (mut order, mut new_ranks, mut sources) = (Vec::new(), Vec::new(), Vec::new());
        for c in (0..blocks).filter(|c| !absorbed.contains(c)) {
            order.extend(starts[c]..starts[c + 1]);
            let mut rank = ranks[c];
            let mut source = vec![c];
            if let Some(&b) = partner.get(&c) {
                order.extend(starts[b]..starts[b + 1]);
                rank += ranks[b];
                source.push(b);
            }
            new_ranks.push(rank);
            sources.push(source);
        }
        let merged = Library { v: library.v.select(Axis(0), &order), u: library.u.select(Axis(0), &order), mean: library.mean.clone() };
        let width = new_ranks.len();
        let mut new_masks = vec![0u8; rows * width];
        for t in 0..rows {
            for (j, source) in sources.iter().enumerate() {
                new_masks[t * width + j] = source.iter().map(|c| masks[t * blocks + c]).max().unwrap_or(0);
            }
        }
        (merged, new_ranks, new_masks)
    };
    loop {
        let mut changed = false;
        loop {
            let blocks = ranks.len();
            // Co-firing counts within the current sets.
            let mut fired = vec![0.0f64; blocks];
            let mut shared: std::collections::BTreeMap<(usize, usize), f64> = std::collections::BTreeMap::new();
            for t in 0..rows {
                let on: Vec<usize> = (0..blocks).filter(|c| masks[t * blocks + c] == 1).collect();
                for (i, &a) in on.iter().enumerate() {
                    fired[a] += 1.0;
                    for &b in &on[i + 1..] {
                        *shared.entry((a, b)).or_insert(0.0) += 1.0;
                    }
                }
            }
            let mut best: Vec<Option<(f64, usize)>> = vec![None; blocks];
            for (&(a, b), &n) in &shared {
                let jaccard = n / (fired[a] + fired[b] - n);
                for (me, other) in [(a, b), (b, a)] {
                    if best[me].is_none_or(|(j, _)| jaccard > j) {
                        best[me] = Some((jaccard, other));
                    }
                }
            }
            let starts = starts_of(&ranks);
            let mut candidates: Vec<(f64, usize, usize)> = Vec::new();
            let mut seen = std::collections::BTreeSet::new();
            for (a, partner) in best.iter().enumerate() {
                let Some((_, b)) = partner else { continue };
                let (a, b) = (a.min(*b), a.max(*b));
                if !seen.insert((a, b)) {
                    continue;
                }
                if refused.contains(&(id(&current, starts[a]), id(&current, starts[b]))) {
                    continue;
                }
                let u = ndarray::concatenate(Axis(0), &[current.u.slice(s![starts[a]..starts[a + 1], ..]), current.u.slice(s![starts[b]..starts[b + 1], ..])]).map_err(|e| e.to_string())?;
                let v = ndarray::concatenate(Axis(0), &[current.v.slice(s![starts[a]..starts[a + 1], ..]), current.v.slice(s![starts[b]..starts[b + 1], ..])]).map_err(|e| e.to_string())?;
                let merged_bits = describe.bits(site, u.view(), v.view())?;
                let saving = shared[&(a, b)] * (bits[a] + bits[b] - merged_bits);
                if saving > 0.0 {
                    candidates.push((saving, a, b));
                }
            }
            candidates.sort_by(|x, y| y.0.total_cmp(&x.0));
            let mut used = std::collections::BTreeSet::new();
            candidates.retain(|(_, a, b)| {
                let free = !used.contains(a) && !used.contains(b);
                if free {
                    used.extend([*a, *b]);
                }
                free
            });
            let mut take = candidates.len();
            let mut kept = false;
            while take > 0 {
                let pairs: Vec<(usize, usize)> = candidates[..take].iter().map(|(_, a, b)| (*a, *b)).collect();
                let (trial, trial_ranks, mut trial_masks) = merge(&current, &ranks, &masks, &pairs);
                let trial_bits = block_bits(&trial, &trial_ranks)?;
                let (d, e) = fitting.code_blocks(&trial.v, &trial.u, &trial_ranks, &trial_bits, &mut trial_masks, true);
                log::info!("site {site}: {take} merges, code {:.1} -> {:.1} bits per input", total / rows as f64, (d + e) / rows as f64);
                if d + e < total {
                    (current, ranks, bits, masks, total) = (trial, trial_ranks, trial_bits, trial_masks, d + e);
                    kept = true;
                    break;
                }
                if take == 1 {
                    let (_, a, b) = candidates.remove(0);
                    refused.insert((id(&current, starts[a]), id(&current, starts[b])));
                    take = candidates.len();
                } else {
                    take /= 2;
                }
            }
            if !kept {
                break;
            }
            changed = true;
        }
        // Splits: every block of rank ≥ 2 bisected by its columns' use, one at a time.
        let mut usage = fitting.usage(&current.v, &current.u);
        let mut c = 0;
        while c < ranks.len() {
            let starts = starts_of(&ranks);
            let rank = ranks[c];
            let key = (id(&current, starts[c]), rank);
            if rank < 2 || refused_splits.contains(&key) {
                c += 1;
                continue;
            }
            let (first, second) = bisect(usage.slice(s![.., starts[c]..starts[c + 1]]))?;
            if first.is_empty() || second.is_empty() {
                refused_splits.insert(key);
                c += 1;
                continue;
            }
            // The block's columns reordered, its first half then its second, each a block on where it
            // was.
            let order: Vec<usize> = (0..starts[c]).chain(first.iter().chain(&second).map(|j| starts[c] + j)).chain(starts[c + 1]..columns).collect();
            let trial = Library { v: current.v.select(Axis(0), &order), u: current.u.select(Axis(0), &order), mean: current.mean.clone() };
            let mut trial_ranks = ranks[..c].to_vec();
            trial_ranks.extend([first.len(), second.len()]);
            trial_ranks.extend_from_slice(&ranks[c + 1..]);
            let (blocks, width) = (ranks.len(), trial_ranks.len());
            let mut trial_masks = vec![0u8; rows * width];
            for t in 0..rows {
                for j in 0..width {
                    let source = if j <= c { j } else { j - 1 };
                    trial_masks[t * width + j] = masks[t * blocks + source];
                }
            }
            let trial_bits = block_bits(&trial, &trial_ranks)?;
            let (d, e) = fitting.code_blocks(&trial.v, &trial.u, &trial_ranks, &trial_bits, &mut trial_masks, true);
            log::info!("site {site}: split of a rank-{rank} block into {} + {}, code {:.1} -> {:.1} bits per input", first.len(), second.len(), total / rows as f64, (d + e) / rows as f64);
            if d + e < total {
                (current, ranks, bits, masks, total) = (trial, trial_ranks, trial_bits, trial_masks, d + e);
                usage = fitting.usage(&current.v, &current.u);
                changed = true;
            } else {
                refused_splits.insert(key);
                c += 1;
            }
        }
        if !changed {
            break;
        }
    }
    let blocks = ranks.len();
    let (description, error) = fitting.code_blocks(&current.v, &current.u, &ranks, &bits, &mut masks, false);
    let on = masks.iter().filter(|m| **m == 1).count() as f64;
    let report = Round { round: 0, code: (description + error) / rows as f64, description: description / rows as f64, error: error / rows as f64, l0: on / rows as f64, reseeded: 0, read_steps: 0 };
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
