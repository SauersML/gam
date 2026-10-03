//! A masked program's library trained on a device (#2951): every product of a step runs where the
//! library lives, and nothing of a step crosses to the host but a few numbers per row.
//!
//! # A step
//!
//! On a batch of sequences, every site's sets are its own code's (`super::site_fit`, module note):
//! with the model's real read `x_t`, every subcomponent's read `a_tc = v_c · x_t` and its write's size
//! in the input's own output Fisher, `‖u_c‖_{F_t}` with `F_t = mean_k g_tk g_tkᵀ` from `draws`
//! sampled-label gradients of the model's own output at the site's written value, each input's set
//! minimises `Σ_{c on} bits(c) + n / (2 ln 2) · (‖D_t‖_{F_t} + Σ_{c off} |a_tc| ‖u_c‖_{F_t})²`
//! (`D_t = (W − Uᵀ V) x_t`, what all on leaves of the map). The library then descends
//!
//! ```text
//! Σ_t KL_t + Σ_sites ½ (Σ_{c off} |z_tc| ‖u_c‖_{F_t})²
//! ```
//!
//! at those sets: the masked forward's KL to the model, and the box claim's charge
//! (`super::masked::box_upper`) on the masked forward's own reads `z_tc` (every point of the box is
//! within it of the sets' point, which the KL measures exactly). Its gradient is exact: the charge's
//! cotangent enters the reverse pass at each site's reads, and its dependence on `U` through the
//! metric is closed-form. Adam moves both factors of every site.
//!
//! # The map
//!
//! Every subcomponent on is the map the library started at, `Uᵀ V = S` (a parameter
//! decomposition sums to its model). After every update `V ← V + G (S − Uᵀ V)` with `G = U (Uᵀ
//! U)⁺`, then `U ← U + (S − Uᵀ V) H` with `H = (V Vᵀ)⁺ V` on what that leaves (nothing when `U`
//! spans every written direction; when it cannot, the reads do), both maps from the last
//! [`Trainer::sync`]. At a sync, in float64, the sum is restored exactly; between syncs, in float32,
//! the residual left is only what the factors moved since times what that step moved.
//!
//! # Precision
//!
//! A step's products run in the trainer's arithmetic (TF32 on a device whose tensor cores make
//! that worth it); [`Trainer::evaluate`] runs the same pass in float64, the number to decide on.
//! The map's correction always runs in float64: it is a small difference of large products.
//! On the Apple GPU, which has no float64 (`gam_gpu::tensor`), every product asked in float64
//! here runs in f32 (the library is held in f32 there), and [`Trainer::evaluate`] is refused: its
//! decision runs on a float64 device.

use super::blocks::Describe;
use super::dense::{eigh, svd};
use super::device_program::DeviceProgram;
use super::masked::{Library, Masked, Site};
use super::operator_program::{FamilyInputs, Node, OperatorProgram, SlotValues};
use gam_gpu::tensor::{Arithmetic, Device, Op, Tensor};
use gam_linalg::faer_ndarray::fast_atb;
use gam_linalg::roundoff::SymmetricAssembly;
use ndarray::{Array1, Array2, Axis, s};
use rayon::prelude::*;
use std::collections::BTreeMap;
use std::f64::consts::LN_2;

fn error(e: impl std::fmt::Display) -> String {
    format!("device train: {e}")
}

/// Float64 on `device`, or f32 where it has none (module note, "Precision").
fn exact(device: &Device) -> Arithmetic {
    if device.float64() { Arithmetic::F64 } else { Arithmetic::F32 }
}

/// How a library is trained.
#[derive(Clone, Copy, Debug)]
pub struct Settings {
    /// `n` of the code.
    pub observations: f64,
    /// Adam's step, as a share of each factor's root mean square entry.
    pub rate: f64,
    pub betas: (f64, f64),
    pub epsilon: f64,
    /// Sampled-label gradients per input for its Fisher.
    pub draws: usize,
    pub seed: u64,
}

/// A pass over a batch: its rows and their totals (KL and charge in nats, the sets' sizes and
/// description bits).
#[derive(Clone, Copy, Debug, Default)]
pub struct Tally {
    pub rows: usize,
    pub kl: f64,
    pub charge: f64,
    pub l0: f64,
    pub description: f64,
}

impl Tally {
    pub fn add(&mut self, other: &Self) {
        self.rows += other.rows;
        self.kl += other.kl;
        self.charge += other.charge;
        self.l0 += other.l0;
        self.description += other.description;
    }

    /// Bits per input under the box claim: `Σ_{c on} bits(c) + n (KL + charge) / ln 2`.
    #[must_use]
    pub fn code(&self, observations: f64) -> f64 {
        (self.description + observations * (self.kl + self.charge) / LN_2) / self.rows.max(1) as f64
    }
}

/// Mirrors `masked`'s label sampler.
struct Uniforms(u64);

impl Uniforms {
    fn next(&mut self) -> f64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        (self.0 >> 11) as f64 / (1u64 << 53) as f64
    }

    fn rows(&mut self, draws: usize, rows: usize) -> Vec<Vec<f64>> {
        (0..draws).map(|_| (0..rows).map(|_| self.next()).collect()).collect()
    }
}

/// A block of a site's library: the node it reads (or writes) in the model and in the masked
/// program, its operator (`V_j`, pieces × d_j; or `U_iᵀ`, d_i × pieces), Adam's moments, its
/// summed gradient and its rate.
struct Block {
    native: usize,
    node: usize,
    op: usize,
    moments: (Tensor, Tensor),
    gradient: Tensor,
    rate: f64,
}

/// One site under training: its blocks, its nodes, and what its step needs.
struct Trained {
    reads: Vec<Block>,
    writes: Vec<Block>,
    z: usize,
    masked: usize,
    slot: usize,
    mask: usize,
    pieces: usize,
    /// Per subcomponent its description bits (1 × pieces), and `[1, bits]` per subcomponent
    /// (pieces × 2) to total a set's size and bits.
    bits: Tensor,
    listing: Tensor,
    /// `S_ij`, the map every subcomponent on is held to, per written and read block.
    map: Vec<Vec<Tensor>>,
    /// `W − S = A Bᵀ` past its decomposition's band, `A` per written block (d_i × r) and `B` per
    /// read block (d_j × r), when any.
    gap: Option<(Vec<Tensor>, Vec<Tensor>)>,
    /// The corrections' maps from the last sync: `G = U (Uᵀ U)⁺` per written block (pieces × d_i),
    /// and `H = (V Vᵀ)⁺ V` per read block (d_j × pieces), in the held layouts.
    projectors: (Vec<Tensor>, Vec<Tensor>),
}

/// A masked program's library on a device, trained as the module note says.
pub struct Trainer {
    device: Device,
    native: DeviceProgram,
    program: DeviceProgram,
    masked: Masked,
    sites: Vec<Trained>,
    settings: Settings,
    arithmetic: Arithmetic,
    steps: u64,
    uniforms: Uniforms,
}

fn ones(device: &Device, rows: usize, cols: usize) -> Result<Tensor, String> {
    let mut out = device.zeros(rows, cols).map_err(error)?;
    let row = device.upload_vec(1, cols, vec![1.0; cols]).map_err(error)?;
    device.add_row(&mut out, 1.0, &row).map_err(error)?;
    Ok(out)
}

/// `α op(a) op(b)` into a fresh tensor.
fn product(device: &Device, a: &Tensor, ta: Op, b: &Tensor, tb: Op, alpha: f64, arithmetic: Arithmetic) -> Result<Tensor, String> {
    let rows = if ta == Op::N { a.rows() } else { a.cols() };
    let cols = if tb == Op::N { b.cols() } else { b.rows() };
    let mut out = device.zeros(rows, cols).map_err(error)?;
    device.gemm(&mut out, alpha, a, ta, b, tb, 0.0, arithmetic).map_err(error)?;
    Ok(out)
}

/// `Σ_i op(a_i) op(b_i)` into a fresh tensor (at least one term).
fn products<'a>(device: &Device, terms: impl IntoIterator<Item = (&'a Tensor, &'a Tensor)>, (ta, tb): (Op, Op), alpha: f64, arithmetic: Arithmetic) -> Result<Tensor, String> {
    let mut out: Option<Tensor> = None;
    for (a, b) in terms {
        match &mut out {
            None => out = Some(product(device, a, ta, b, tb, alpha, arithmetic)?),
            Some(total) => device.gemm(total, alpha, a, ta, b, tb, 1.0, arithmetic).map_err(error)?,
        }
    }
    out.ok_or_else(|| error("an empty sum of products"))
}

/// The matrix of blocks `blocks[i][j]` (each downloaded).
fn assemble(device: &Device, blocks: &[Vec<Tensor>]) -> Result<Array2<f64>, String> {
    let rows: Vec<Array2<f64>> = blocks
        .iter()
        .map(|row| {
            let parts: Vec<Array2<f64>> = row.iter().map(|b| device.download(b).map_err(error)).collect::<Result<_, _>>()?;
            ndarray::concatenate(Axis(1), &parts.iter().map(|p| p.view()).collect::<Vec<_>>()).map_err(error)
        })
        .collect::<Result<_, _>>()?;
    ndarray::concatenate(Axis(0), &rows.iter().map(|p| p.view()).collect::<Vec<_>>()).map_err(error)
}

/// Per site the model's statistics a description is priced in (`super::pieces::Site`): its map,
/// the reads' second moment and the written Fisher from `draws` sampled-label gradients per input,
/// over `batches`, computed on `device` in float64.
pub fn statistics(device: &Device, model: &OperatorProgram, sites: &[Site], batches: &[FamilyInputs], draws: usize, seed: u64) -> Result<Vec<super::pieces::Site>, String> {
    if draws == 0 || batches.is_empty() {
        return Err(error("statistics need draws and inputs"));
    }
    let mut native = DeviceProgram::compile(device, model)?;
    native.set_arithmetic(exact(device));
    let mut uniforms = Uniforms(seed | 1);
    // Per site, the blocks of `Σ xᵀx` over its read nodes and of `Σ gᵀg` over its written ones.
    let mut moments: Vec<Vec<Vec<Option<Tensor>>>> = sites.iter().map(|s| s.reads.iter().map(|_| s.reads.iter().map(|_| None).collect()).collect()).collect();
    let mut fishers: Vec<Vec<Vec<Option<Tensor>>>> = sites.iter().map(|s| s.writes.iter().map(|_| s.writes.iter().map(|_| None).collect()).collect()).collect();
    let writes: Vec<usize> = sites.iter().flat_map(|s| s.writes.iter().copied()).collect();
    let mut rows = 0;
    let accumulate = |slot: &mut Option<Tensor>, x: &Tensor, y: &Tensor| -> Result<(), String> {
        match slot {
            Some(total) => device.gemm(total, 1.0, x, Op::T, y, Op::N, 1.0, exact(device)).map_err(error),
            None => {
                *slot = Some(product(device, x, Op::T, y, Op::N, 1.0, exact(device))?);
                Ok(())
            }
        }
    };
    for batch in batches {
        let trace = native.forward(batch)?;
        rows += batch.rows;
        for (k, site) in sites.iter().enumerate() {
            for (j, a) in site.reads.iter().enumerate() {
                for (l, b) in site.reads.iter().enumerate() {
                    accumulate(&mut moments[k][j][l], trace.value(*a)?, trace.value(*b)?)?;
                }
            }
        }
        for seed in native.sampled_many(&trace, &uniforms.rows(draws, batch.rows), None)? {
            let back = native.vjp(&trace, seed, &writes, exact(device))?;
            for (k, site) in sites.iter().enumerate() {
                for (i, a) in site.writes.iter().enumerate() {
                    for (l, b) in site.writes.iter().enumerate() {
                        if let (Some(ga), Some(gb)) = (back.get(a), back.get(b)) {
                            accumulate(&mut fishers[k][i][l], ga, gb)?;
                        }
                    }
                }
            }
        }
    }
    let widths = native.widths();
    let mut out = Vec::new();
    for (k, site) in sites.iter().enumerate() {
        let w = super::masked::matrix(model, site)?;
        let filled = |blocks: &mut Vec<Vec<Option<Tensor>>>, nodes: &[usize]| -> Result<Vec<Vec<Tensor>>, String> {
            blocks
                .iter_mut()
                .zip(nodes)
                .map(|(row, a)| row.iter_mut().zip(nodes).map(|(b, c)| b.take().map_or_else(|| device.zeros(widths[*a], widths[*c]).map_err(error), Ok)).collect())
                .collect()
        };
        let second_moment = assemble(device, &filled(&mut moments[k], &site.reads)?)? / rows as f64;
        let fisher = assemble(device, &filled(&mut fishers[k], &site.writes)?)? / (rows * draws) as f64;
        out.push(super::pieces::Site { mean: Array1::zeros(w.ncols()), w, second_moment, fisher });
    }
    Ok(out)
}

/// Per subcomponent its description bits under `describe`.
fn description_bits(describe: &dyn Describe, site: usize, library: &Library) -> Result<Vec<f64>, String> {
    (0..library.v.nrows())
        .into_par_iter()
        .map(|c| describe.bits(site, library.u.slice(s![c..c + 1, ..]), library.v.slice(s![c..c + 1, ..])))
        .collect()
}

/// Splits `m`'s rows (or, with `columns`, its columns) at the widths of `blocks`.
fn split(m: &Array2<f64>, widths: &[usize], columns: bool) -> Vec<Array2<f64>> {
    let mut start = 0;
    widths
        .iter()
        .map(|w| {
            let part = if columns { m.slice(s![.., start..start + w]).to_owned() } else { m.slice(s![start..start + w, ..]).to_owned() };
            start += w;
            part
        })
        .collect()
}

impl Trainer {
    /// `masked` (its sites `sites` of `model`) trained on `device`, its steps' products in
    /// `arithmetic`, every subcomponent described by `describe`.
    pub fn new(device: &Device, model: &OperatorProgram, sites: &[Site], masked: Masked, describe: &dyn Describe, settings: Settings, arithmetic: Arithmetic) -> Result<Self, String> {
        if sites.len() != masked.sites.len() || settings.draws == 0 {
            return Err(error("one model site per masked site, and at least one draw"));
        }
        let mut native = DeviceProgram::compile(device, model)?;
        let mut program = DeviceProgram::compile(device, &masked.program)?;
        native.set_arithmetic(arithmetic);
        program.set_arithmetic(arithmetic);
        let widths = native.widths();
        let mut trained = Vec::new();
        for (k, (site, inner)) in sites.iter().zip(&masked.sites).enumerate() {
            if site.reads.len() != inner.reads.len() || site.writes.len() != inner.writes.len() {
                return Err(error(format!("{}: the masked site is not the model's", site.name)));
            }
            let Node::Hadamard { right, .. } = &masked.program.nodes[masked.masked[k]] else {
                return Err(error(format!("{}: its masked node is not its mask's product", site.name)));
            };
            let library = masked.library(k)?;
            let w = masked.w(k)?;
            let (read_widths, write_widths): (Vec<usize>, Vec<usize>) = (site.reads.iter().map(|n| widths[*n]).collect(), site.writes.iter().map(|n| widths[*n]).collect());
            let map = fast_atb(&library.u, &library.v);
            let gap = {
                let decomposed = svd((&w - &map).view(), false).map_err(|e| format!("{e:?}"))?;
                let kept: Vec<usize> = (0..decomposed.singular_values.len()).filter(|&j| decomposed.singular_values[j] > decomposed.band).collect();
                if kept.is_empty() {
                    None
                } else {
                    let a = decomposed.u.select(Axis(1), &kept) * &decomposed.singular_values.select(Axis(0), &kept);
                    let b = decomposed.vt.select(Axis(0), &kept).t().to_owned();
                    let up = |parts: Vec<Array2<f64>>| parts.iter().map(|p| device.upload(p.view()).map_err(error)).collect::<Result<Vec<_>, _>>();
                    Some((up(split(&a, &write_widths, false))?, up(split(&b, &read_widths, false))?))
                }
            };
            let map = split(&map, &write_widths, false)
                .iter()
                .map(|row| split(row, &read_widths, true).iter().map(|b| device.upload(b.view()).map_err(error)).collect::<Result<Vec<_>, _>>())
                .collect::<Result<Vec<_>, _>>()?;
            let rms = |m: &Array2<f64>| (m.iter().map(|x| x * x).sum::<f64>() / m.len().max(1) as f64).sqrt();
            let block = |native: usize, node: usize, op: usize, rate: f64| -> Result<Block, String> {
                let held = program.dense(op)?;
                let zeros = || device.zeros(held.rows(), held.cols()).map_err(error);
                Ok(Block { native, node, op, moments: (zeros()?, zeros()?), gradient: zeros()?, rate })
            };
            let reads = (0..site.reads.len()).map(|j| block(site.reads[j], inner.reads[j], masked.v_ops(k)[j], settings.rate * rms(&library.v))).collect::<Result<Vec<_>, _>>()?;
            let writes = (0..site.writes.len()).map(|i| block(site.writes[i], inner.writes[i], masked.u_ops(k)[i], settings.rate * rms(&library.u))).collect::<Result<Vec<_>, _>>()?;
            let pieces = library.v.nrows();
            trained.push(Trained {
                reads,
                writes,
                z: masked.z[k],
                masked: masked.masked[k],
                slot: masked.slots[k],
                mask: *right,
                pieces,
                bits: device.zeros(1, pieces).map_err(error)?,
                listing: device.zeros(pieces, 2).map_err(error)?,
                map,
                gap,
                projectors: (Vec::new(), Vec::new()),
            });
        }
        let mut trainer = Self { device: device.clone(), native, program, masked, sites: trained, settings, arithmetic, steps: 0, uniforms: Uniforms(settings.seed | 1) };
        trainer.sync(describe)?;
        Ok(trainer)
    }

    /// The masked program as of the last [`Self::sync`].
    #[must_use]
    pub fn masked(&self) -> &Masked {
        &self.masked
    }

    /// Updates taken.
    #[must_use]
    pub fn steps(&self) -> u64 {
        self.steps
    }

    /// A training pass on `inputs` (whole sequences): its sets, KL and charge, and their gradient
    /// added to the update's.
    pub fn train(&mut self, inputs: &FamilyInputs) -> Result<Tally, String> {
        self.pass(inputs, true)
    }

    /// The same pass in float64, without a gradient, its labels drawn from the settings' seed
    /// alone: the same inputs always meet the same draws, so two libraries compare on one footing.
    pub fn evaluate(&mut self, inputs: &FamilyInputs) -> Result<Tally, String> {
        if !self.device.float64() {
            return Err(error(format!("{} has no float64: evaluate on a float64 device", self.device.name())));
        }
        let (arithmetic, state) = (self.arithmetic, self.uniforms.0);
        self.set_arithmetic(Arithmetic::F64);
        self.uniforms = Uniforms(self.settings.seed.rotate_left(32) | 1);
        let tally = self.pass(inputs, false);
        self.set_arithmetic(arithmetic);
        self.uniforms = Uniforms(state);
        tally
    }

    fn set_arithmetic(&mut self, arithmetic: Arithmetic) {
        self.arithmetic = arithmetic;
        self.native.set_arithmetic(arithmetic);
        self.program.set_arithmetic(arithmetic);
    }

    fn pass(&mut self, inputs: &FamilyInputs, learn: bool) -> Result<Tally, String> {
        let d = self.device.clone();
        let (rows, arithmetic, draws) = (inputs.rows, self.arithmetic, self.settings.draws);
        let trace = self.native.forward(inputs)?;
        let target = self.native.logits_on_device(&trace)?;
        // Every written block's sampled-label gradients, per site, per draw.
        let writes: Vec<usize> = self.sites.iter().flat_map(|s| s.writes.iter().map(|b| b.native)).collect();
        let mut gradients: Vec<Vec<Vec<Tensor>>> = self.sites.iter().map(|_| Vec::new()).collect();
        for seed in self.native.sampled_many(&trace, &self.uniforms.rows(draws, rows), None)? {
            let back = self.native.vjp(&trace, seed, &writes, arithmetic)?;
            for (k, site) in self.sites.iter().enumerate() {
                let drawn = site
                    .writes
                    .iter()
                    .map(|b| match back.get(&b.native) {
                        Some(g) => d.copy(g).map_err(error),
                        None => d.zeros(rows, self.program.dense(b.op)?.rows()).map_err(error),
                    })
                    .collect::<Result<Vec<_>, _>>()?;
                gradients[k].push(drawn);
            }
        }
        // Each site's sets under its own code, and its writes' sizes in each input's metric.
        let mut tally = Tally { rows, ..Tally::default() };
        let mut masks = BTreeMap::new();
        let mut owns = Vec::new();
        let weight = d.upload_vec(rows, 1, vec![self.settings.observations / (2.0 * LN_2); rows]).map_err(error)?;
        let root = (draws as f64).sqrt().recip();
        for (k, site) in self.sites.iter().enumerate() {
            let program = &self.program;
            let reads: Vec<(&Tensor, &Tensor)> = site.reads.iter().map(|b| Ok((trace.value(b.native)?, program.dense(b.op)?))).collect::<Result<_, String>>()?;
            let a = products(&d, reads.iter().copied(), (Op::N, Op::T), 1.0, arithmetic)?;
            let mut own = d.zeros(rows, site.pieces).map_err(error)?;
            for drawn in &gradients[k] {
                let terms: Vec<(&Tensor, &Tensor)> = drawn.iter().zip(&site.writes).map(|(g, b)| Ok((g, program.dense(b.op)?))).collect::<Result<_, String>>()?;
                let p = products(&d, terms, (Op::N, Op::N), root, arithmetic)?;
                d.hadamard(&mut own, &p, &p, true).map_err(error)?;
            }
            let left = match &site.gap {
                None => d.zeros(rows, 1).map_err(error)?,
                Some((a_gap, b_gap)) => {
                    let through = products(&d, reads.iter().map(|(x, _)| *x).zip(b_gap), (Op::N, Op::N), 1.0, arithmetic)?;
                    let mut squares = vec![0.0; rows];
                    for drawn in &gradients[k] {
                        let mut along = vec![0.0; rows];
                        for (g, a_i) in drawn.iter().zip(a_gap) {
                            let missed = product(&d, &through, Op::N, a_i, Op::T, 1.0, arithmetic)?;
                            let mut both = d.zeros(rows, missed.cols()).map_err(error)?;
                            d.hadamard(&mut both, g, &missed, false).map_err(error)?;
                            let summed = d.download(&product(&d, &both, Op::N, &ones(&d, missed.cols(), 1)?, Op::N, 1.0, exact(&d))?).map_err(error)?;
                            for (total, v) in along.iter_mut().zip(summed.iter()) {
                                *total += v;
                            }
                        }
                        for (q, v) in squares.iter_mut().zip(&along) {
                            *q += v * v / draws as f64;
                        }
                    }
                    d.upload_vec(rows, 1, squares.into_iter().map(f64::sqrt).collect()).map_err(error)?
                }
            };
            let mut mask = ones(&d, rows, site.pieces)?;
            d.select_sets((&a, &own), &site.bits, &left, &weight, &mut mask).map_err(error)?;
            let listed = d.download(&product(&d, &mask, Op::N, &site.listing, Op::N, 1.0, exact(&d))?).map_err(error)?;
            tally.l0 += listed.column(0).sum();
            tally.description += listed.column(1).sum();
            masks.insert(site.slot, mask);
            owns.push(own);
        }
        drop(trace);
        // The masked forward at the sets, its KL, and the box claim's charge on its reads.
        let mut family = inputs.clone();
        let slots = self.sites.iter().map(|s| s.slot + 1).max().unwrap_or(0);
        while family.slots.len() < slots {
            family.slots.push(SlotValues::Raw(Array2::zeros((0, 0))));
        }
        let masked_trace = self.program.forward_given(&family, masks)?;
        let (kl, hidden) = if learn {
            let (kl, cot) = self.program.kl(&masked_trace, &target, None)?;
            (kl, Some(cot))
        } else {
            (self.program.score_only(&masked_trace, &target, None)?, None)
        };
        drop(target);
        tally.kl = kl.sum();
        let mut seeds = BTreeMap::new();
        let mut coefficients = Vec::new();
        for (site, own) in self.sites.iter().zip(owns) {
            let z = masked_trace.value(site.z)?;
            let mut cot = d.zeros(z.rows(), z.cols()).map_err(error)?;
            let mut coefficient = d.zeros(z.rows(), z.cols()).map_err(error)?;
            tally.charge += d.box_charge(z, masked_trace.value(site.mask)?, &own, &mut cot, &mut coefficient).map_err(error)?.iter().sum::<f64>();
            seeds.insert(site.z, cot);
            coefficients.push(coefficient);
        }
        let Some(hidden) = hidden else { return Ok(tally) };
        let keep: Vec<usize> = self.sites.iter().flat_map(|s| std::iter::once(s.z).chain(s.writes.iter().map(|b| b.node))).collect();
        let back = self.program.vjp_seeded(&masked_trace, hidden, seeds, &keep, arithmetic)?;
        let program = &self.program;
        for (k, site) in self.sites.iter_mut().enumerate() {
            if let Some(cot_z) = back.get(&site.z) {
                for b in &mut site.reads {
                    d.gemm(&mut b.gradient, 1.0, cot_z, Op::T, masked_trace.value(b.node)?, Op::N, 1.0, arithmetic).map_err(error)?;
                }
            }
            let gated = masked_trace.value(site.masked)?;
            for b in &mut site.writes {
                if let Some(g) = back.get(&b.node) {
                    d.gemm(&mut b.gradient, 1.0, g, Op::T, gated, Op::N, 1.0, arithmetic).map_err(error)?;
                }
            }
            // The charge through each write's size: ∂‖u_c‖_{F_t}/∂u_c = mean_k (g_tk · u_c) g_tk / ‖u_c‖_{F_t}.
            for drawn in &gradients[k] {
                let terms: Vec<(&Tensor, &Tensor)> = drawn.iter().zip(&site.writes).map(|(g, b)| Ok((g, program.dense(b.op)?))).collect::<Result<_, String>>()?;
                let p = products(&d, terms, (Op::N, Op::N), 1.0, arithmetic)?;
                let mut weighted = d.zeros(p.rows(), p.cols()).map_err(error)?;
                d.hadamard(&mut weighted, &coefficients[k], &p, false).map_err(error)?;
                for (g, b) in drawn.iter().zip(&mut site.writes) {
                    d.gemm(&mut b.gradient, 1.0 / draws as f64, g, Op::T, &weighted, Op::N, 1.0, arithmetic).map_err(error)?;
                }
            }
        }
        Ok(tally)
    }

    /// The gradients summed since the last update, every site's read blocks then written blocks
    /// (to sum replicas' passes: data parallel over devices).
    pub fn gradients(&self) -> Result<Vec<Array2<f64>>, String> {
        self.sites.iter().flat_map(|s| s.reads.iter().chain(&s.writes)).map(|b| self.device.download(&b.gradient).map_err(error)).collect()
    }

    /// Replace the summed gradients by `gradients` (as [`Self::gradients`] orders them).
    pub fn set_gradients(&mut self, gradients: &[Array2<f64>]) -> Result<(), String> {
        let blocks: Vec<&mut Block> = self.sites.iter_mut().flat_map(|s| s.reads.iter_mut().chain(s.writes.iter_mut())).collect();
        if blocks.len() != gradients.len() {
            return Err(error(format!("{} gradients for {} blocks", gradients.len(), blocks.len())));
        }
        for (b, g) in blocks.into_iter().zip(gradients) {
            if g.dim() != b.gradient.dim() {
                return Err(error(format!("a {:?} gradient for a {:?} block", g.dim(), b.gradient.dim())));
            }
            b.gradient = self.device.upload(g.view()).map_err(error)?;
        }
        Ok(())
    }

    /// One Adam update from the gradients the training passes since the last summed, then each
    /// site's map restored (module note, "The map").
    pub fn update(&mut self) -> Result<(), String> {
        self.steps += 1;
        let d = self.device.clone();
        let Settings { betas: (beta1, beta2), epsilon, .. } = self.settings;
        for k in 0..self.sites.len() {
            let site = &mut self.sites[k];
            for b in site.reads.iter_mut().chain(site.writes.iter_mut()) {
                let (m, v) = &mut b.moments;
                d.adam(self.program.dense_mut(b.op)?, (m, v), &b.gradient, b.rate, (beta1, beta2, epsilon), self.steps).map_err(error)?;
                b.gradient = d.zeros(b.gradient.rows(), b.gradient.cols()).map_err(error)?;
            }
            self.keep_map(k, Arithmetic::F32)?;
        }
        Ok(())
    }

    /// The residuals `S_ij − U_iᵀ V_j` of site `k`.
    fn residuals(&self, k: usize, arithmetic: Arithmetic) -> Result<Vec<Vec<Tensor>>, String> {
        let (d, site) = (&self.device, &self.sites[k]);
        site.writes
            .iter()
            .enumerate()
            .map(|(i, write)| {
                site.reads
                    .iter()
                    .enumerate()
                    .map(|(j, read)| {
                        let mut residual = d.copy(&site.map[i][j]).map_err(error)?;
                        d.gemm(&mut residual, -1.0, self.program.dense(write.op)?, Op::N, self.program.dense(read.op)?, Op::N, 1.0, arithmetic).map_err(error)?;
                        Ok(residual)
                    })
                    .collect()
            })
            .collect()
    }

    /// Site `k`'s map restored (module note, "The map"): `V_j ← V_j + Σ_i G_i R_ij`, then `U_i ←
    /// U_i + Σ_j R'_ij H_j` on what that leaves (nothing, when `U` spans every written direction).
    fn keep_map(&mut self, k: usize, arithmetic: Arithmetic) -> Result<(), String> {
        let d = self.device.clone();
        let residuals = self.residuals(k, arithmetic)?;
        let site = &self.sites[k];
        let corrections = (0..site.reads.len())
            .map(|j| products(&d, site.projectors.0.iter().zip(residuals.iter().map(|row| &row[j])), (Op::N, Op::N), 1.0, arithmetic))
            .collect::<Result<Vec<_>, _>>()?;
        for (read, correction) in site.reads.iter().zip(corrections) {
            d.axpy(self.program.dense_mut(read.op)?, 1.0, &correction).map_err(error)?;
        }
        let residuals = self.residuals(k, arithmetic)?;
        let site = &self.sites[k];
        let corrections = residuals.iter().map(|row| products(&d, row.iter().zip(&site.projectors.1), (Op::N, Op::N), 1.0, arithmetic)).collect::<Result<Vec<_>, _>>()?;
        for (write, correction) in site.writes.iter().zip(corrections) {
            d.axpy(self.program.dense_mut(write.op)?, 1.0, &correction).map_err(error)?;
        }
        Ok(())
    }

    /// Every site's projector from its current `U`, its map restored exactly, its library written
    /// into the masked program, and its description bits priced again under `describe`.
    pub fn sync(&mut self, describe: &dyn Describe) -> Result<&Masked, String> {
        let d = self.device.clone();
        for k in 0..self.sites.len() {
            let site = &self.sites[k];
            let us: Vec<&Tensor> = site.writes.iter().map(|b| self.program.dense(b.op)).collect::<Result<_, _>>()?;
            let vs: Vec<&Tensor> = site.reads.iter().map(|b| self.program.dense(b.op)).collect::<Result<_, _>>()?;
            // G_i = Σ_l U_lᵀ (U Uᵀ)⁺_li and H_j = Σ_l (Vᵀ V)⁺_jl V_lᵀ, from the Grams' pseudo-inverses.
            let pseudo_inverse = |blocks: Vec<Vec<Tensor>>| -> Result<Array2<f64>, String> {
                let gram = assemble(&d, &blocks)?;
                let decomposed = eigh(gram.view(), SymmetricAssembly::Mirrored, None).map_err(|e| format!("{e:?}"))?;
                let mut scaled = decomposed.vectors.clone();
                for (j, value) in decomposed.values.iter().enumerate() {
                    let inverse = if *value > decomposed.band { value.recip() } else { 0.0 };
                    scaled.column_mut(j).mapv_inplace(|x| x * inverse);
                }
                Ok(scaled.dot(&decomposed.vectors.t()))
            };
            let grams = |held: &[&Tensor], ta: Op, tb: Op| -> Result<Vec<Vec<Tensor>>, String> {
                held.iter().map(|a| held.iter().map(|b| product(&d, a, ta, b, tb, 1.0, exact(&d))).collect()).collect()
            };
            let (u_widths, v_widths): (Vec<usize>, Vec<usize>) = (us.iter().map(|t| t.rows()).collect(), vs.iter().map(|t| t.cols()).collect());
            let inverse = pseudo_inverse(grams(&us, Op::N, Op::T)?)?;
            let mut g = Vec::new();
            for column in split(&inverse, &u_widths, true) {
                let parts = split(&column, &u_widths, false).iter().map(|p| d.upload(p.view()).map_err(error)).collect::<Result<Vec<_>, _>>()?;
                g.push(products(&d, us.iter().copied().zip(&parts), (Op::T, Op::N), 1.0, exact(&d))?);
            }
            let inverse = pseudo_inverse(grams(&vs, Op::T, Op::N)?)?;
            let mut h = Vec::new();
            for row in split(&inverse, &v_widths, false) {
                let parts = split(&row, &v_widths, true).iter().map(|p| d.upload(p.view()).map_err(error)).collect::<Result<Vec<_>, _>>()?;
                h.push(products(&d, parts.iter().zip(vs.iter().copied()), (Op::N, Op::T), 1.0, exact(&d))?);
            }
            self.sites[k].projectors = (g, h);
            self.keep_map(k, exact(&d))?;
            let site = &self.sites[k];
            let download = |op: usize| -> Result<Array2<f64>, String> { d.download(self.program.dense(op)?).map_err(error) };
            let v_parts: Vec<Array2<f64>> = site.reads.iter().map(|b| download(b.op)).collect::<Result<_, _>>()?;
            let u_parts: Vec<Array2<f64>> = site.writes.iter().map(|b| download(b.op).map(|m| m.t().to_owned())).collect::<Result<_, _>>()?;
            let v = ndarray::concatenate(Axis(1), &v_parts.iter().map(|p| p.view()).collect::<Vec<_>>()).map_err(error)?;
            let u = ndarray::concatenate(Axis(1), &u_parts.iter().map(|p| p.view()).collect::<Vec<_>>()).map_err(error)?;
            let library = Library { mean: Array1::zeros(v.ncols()), v, u };
            let bits = description_bits(describe, k, &library)?;
            let pieces = bits.len();
            self.masked.set_library(k, library)?;
            let listing: Vec<f64> = bits.iter().flat_map(|b| [1.0, *b]).collect();
            self.sites[k].listing = d.upload_vec(pieces, 2, listing).map_err(error)?;
            self.sites[k].bits = d.upload_vec(1, pieces, bits).map_err(error)?;
        }
        Ok(&self.masked)
    }
}
