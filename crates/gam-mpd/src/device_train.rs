//! A masked program's library trained on a device (#2951): every product of a step runs where the
//! library lives, and nothing of a step crosses to the host but a few numbers per row.
//!
//! # A step
//!
//! On a batch of sequences the explanation runs autonomously, the evaluator's corner run
//! (`super::explanation::Explanation::execute`): the masked program executes in order, and at each
//! site, once its read `x_t` is computed (what the sites before it, replaced, handed on), the site's
//! own code chooses its subcomponents on from that read alone (`super::site_fit::Selector`, the
//! certified `super::sparse_code::Coder`): per input `Σ_{c on} bits(c) + n/(2 ln 2) ‖W x_t − Σ_{c
//! on} u_c (v_c · x_t)‖²_F̄`, `F̄` the site's mean written Fisher. At those sets the library descends
//! the code of the run,
//!
//! ```text
//! Σ_t Σ_{c on} bits(c) + n KL_t / ln 2,
//! ```
//!
//! `KL_t` the corner run's KL to the model at input `t`: the same discrete sets the evaluator scores,
//! no expected masks, no box over the off gates and no adversary. The sets are a step's decisions,
//! so the gradient is the KL's at them, exact through the masked forward; Adam moves both factors of
//! every site. A step's coder reads `U F̄` from the factors as they are; its Gram `U F̄ Uᵀ` and the
//! bits are the last [`Trainer::sync`]'s.
//!
//! # The map
//!
//! Every subcomponent on is the map the library started at, `U V = S` in the held layouts (`U`
//! d_out × pieces, `V` pieces × d_in; a parameter decomposition sums to its model). An update
//! moves the map by `ΔU V + U ΔV` (exactly, with `V` after the step and `U` before it), and the
//! least change of both factors that undoes `R` to first order is `δU = Λ Vᵀ`, `δV = Uᵀ Λ` with
//! `U Uᵀ Λ + Λ Vᵀ V = R`: a Sylvester equation, solved in the Grams' eigenbases by dividing by
//! `λ_U + λ_V`, which no direction makes small unless both factors lack it (then nothing can
//! restore it to first order). The bases are the last [`Trainer::sync`]'s; a sync decomposes the
//! Grams afresh and retracts `S − U V` in float64 while it shrinks, so the map is exact there and
//! between syncs it drifts only by what the stale bases and the step's second order leave.
//!
//! # Precision
//!
//! A step's products run in the trainer's arithmetic (TF32 on a device whose tensor cores make
//! that worth it); [`Trainer::evaluate`] runs the same pass in float64, the number to decide on.
//! A step's retraction runs in f32 on the step's own small products; a sync's runs in float64 on
//! `S − U V`, a small difference of large products.
//! On the Apple GPU, which has no float64 (`gam_gpu::tensor`), every product asked in float64
//! here runs in f32 (the library is held in f32 there), and [`Trainer::evaluate`] is refused: its
//! decision runs on a float64 device.

use super::blocks::Describe;
use super::core_device::DeviceCoder;
use gam_linalg::decompose::eigh;
use super::device_program::DeviceProgram;
use super::masked::{Library, Masked, Site};
use super::operator_program::{FamilyInputs, Node, OperatorProgram, SlotValues};
use gam_gpu::tensor::{Arithmetic, Device, Op, Tensor};
use gam_linalg::faer_ndarray::{fast_ab, fast_abt, fast_atb};
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
}

/// A pass over a batch: its rows and their totals (the corner run's KL in nats, the sets' sizes and
/// description bits).
#[derive(Clone, Copy, Debug, Default)]
pub struct Tally {
    pub rows: usize,
    pub kl: f64,
    pub l0: f64,
    pub description: f64,
}

impl Tally {
    pub fn add(&mut self, other: &Self) {
        self.rows += other.rows;
        self.kl += other.kl;
        self.l0 += other.l0;
        self.description += other.description;
    }

    /// Bits per input of the corner run: `Σ_{c on} bits(c) + n KL / ln 2`.
    #[must_use]
    pub fn code(&self, observations: f64) -> f64 {
        (self.description + observations * self.kl / LN_2) / self.rows.max(1) as f64
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

/// A block of a site's library: the node it reads (or writes) in the masked program, its operator
/// (`V_j`, pieces × d_j; or `U_iᵀ`, d_i × pieces), Adam's moments, its summed gradient and its
/// rate.
struct Block {
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
    /// Per subcomponent its description bits (1 × pieces), and `[1, bits]` per subcomponent
    /// (pieces × 2) to total a set's size and bits.
    bits: Tensor,
    listing: Tensor,
    /// `S_ij`, the map every subcomponent on is held to, per written and read block.
    map: Vec<Vec<Tensor>>,
    /// What the site's code reads (module note, "A step"): the site's map `W` per read block
    /// (`d_out × d_j`), its mean written Fisher `F̄` whole and per written block's rows
    /// (`d_i × d_out`), on the host for the Gram, and a column of ones (`d_out × 1`).
    w: Vec<Tensor>,
    fisher: Tensor,
    fisher_rows: Vec<Tensor>,
    fisher_host: Array2<f64>,
    ones: Tensor,
    /// The code as of the last sync.
    coder: Option<DeviceCoder>,
    /// The Grams' eigenbases from the last sync (module note, "The map").
    bases: Bases,
}

/// The eigendecompositions `U Uᵀ = Q_U Λ_U Q_Uᵀ` and `Vᵀ V = Q_V Λ_V Q_Vᵀ` of a site's held factors
/// (`U` stacked over its written blocks, `V` side by side over its read blocks): `Q_U`'s rows per
/// written block, `Λ_U` a column, `Q_V`'s rows per read block, `Λ_V` a row, and the sum of their
/// bands, below which `λ_U + λ_V` is no direction.
struct Bases {
    u: Vec<Tensor>,
    u_values: Tensor,
    v: Vec<Tensor>,
    v_values: Tensor,
    floor: f64,
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
        .map(|c| gam_linalg::faer_ndarray::with_nested_parallel(|| describe.bits(site, library.u.slice(s![c..c + 1, ..]), library.v.slice(s![c..c + 1, ..]))))
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
    /// `arithmetic`, every subcomponent described by `describe`, each site's code in its mean
    /// written Fisher `fishers[k]` (d_out × d_out).
    pub fn new(
        device: &Device,
        (model, sites): (&OperatorProgram, &[Site]),
        masked: Masked,
        (describe, fishers): (&dyn Describe, &[Array2<f64>]),
        settings: Settings,
        arithmetic: Arithmetic,
    ) -> Result<Self, String> {
        if sites.len() != masked.sites.len() || fishers.len() != sites.len() {
            return Err(error("one model site and one Fisher per masked site"));
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
            if !matches!(masked.program.nodes[masked.masked[k]], Node::Hadamard { .. }) {
                return Err(error(format!("{}: its masked node is not its mask's product", site.name)));
            }
            let library = masked.library(k)?;
            let w = masked.w(k)?;
            let fisher = &fishers[k];
            if fisher.dim() != (w.nrows(), w.nrows()) {
                return Err(error(format!("{}: a {:?} Fisher for {} writes", site.name, fisher.dim(), w.nrows())));
            }
            let (read_widths, write_widths): (Vec<usize>, Vec<usize>) = (site.reads.iter().map(|n| widths[*n]).collect(), site.writes.iter().map(|n| widths[*n]).collect());
            let map = split(&fast_atb(&library.u, &library.v), &write_widths, false)
                .iter()
                .map(|row| split(row, &read_widths, true).iter().map(|b| device.upload(b.view()).map_err(error)).collect::<Result<Vec<_>, _>>())
                .collect::<Result<Vec<_>, _>>()?;
            let upload = |parts: Vec<Array2<f64>>| parts.iter().map(|p| device.upload(p.view()).map_err(error)).collect::<Result<Vec<_>, _>>();
            let rms = |m: &Array2<f64>| (m.iter().map(|x| x * x).sum::<f64>() / m.len().max(1) as f64).sqrt();
            let block = |node: usize, op: usize, rate: f64| -> Result<Block, String> {
                let held = program.dense(op)?;
                let zeros = || device.zeros(held.rows(), held.cols()).map_err(error);
                Ok(Block { node, op, moments: (zeros()?, zeros()?), gradient: zeros()?, rate })
            };
            let reads = (0..site.reads.len()).map(|j| block(inner.reads[j], masked.v_ops(k)[j], settings.rate * rms(&library.v))).collect::<Result<Vec<_>, _>>()?;
            let writes = (0..site.writes.len()).map(|i| block(inner.writes[i], masked.u_ops(k)[i], settings.rate * rms(&library.u))).collect::<Result<Vec<_>, _>>()?;
            let pieces = library.v.nrows();
            trained.push(Trained {
                reads,
                writes,
                z: masked.z[k],
                masked: masked.masked[k],
                slot: masked.slots[k],
                bits: device.zeros(1, pieces).map_err(error)?,
                listing: device.zeros(pieces, 2).map_err(error)?,
                map,
                w: upload(split(&w, &read_widths, true))?,
                fisher: device.upload(fisher.view()).map_err(error)?,
                fisher_rows: upload(split(fisher, &write_widths, false))?,
                fisher_host: fisher.clone(),
                ones: device.upload_vec(w.nrows(), 1, vec![1.0; w.nrows()]).map_err(error)?,
                coder: None,
                bases: Bases { u: Vec::new(), u_values: device.zeros(0, 1).map_err(error)?, v: Vec::new(), v_values: device.zeros(1, 0).map_err(error)?, floor: 0.0 },
            });
        }
        let mut trainer = Self { device: device.clone(), native, program, masked, sites: trained, settings, arithmetic, steps: 0 };
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

    /// Every site's subcomponents' description bits as of the last [`Self::sync`].
    pub fn bits(&self) -> Result<Vec<Vec<f64>>, String> {
        self.sites.iter().map(|s| Ok(self.device.download(&s.bits).map_err(error)?.row(0).to_vec())).collect()
    }

    /// A training pass on `inputs` (whole sequences): the corner run's sets and KL, and the KL's
    /// gradient at those sets added to the update's.
    pub fn train(&mut self, inputs: &FamilyInputs) -> Result<Tally, String> {
        self.pass(inputs, true)
    }

    /// The same pass in float64, without a gradient.
    pub fn evaluate(&mut self, inputs: &FamilyInputs) -> Result<Tally, String> {
        if !self.device.float64() {
            return Err(error(format!("{} has no float64: evaluate on a float64 device", self.device.name())));
        }
        let arithmetic = self.arithmetic;
        self.set_arithmetic(Arithmetic::F64);
        let tally = self.pass(inputs, false);
        self.set_arithmetic(arithmetic);
        tally
    }

    fn set_arithmetic(&mut self, arithmetic: Arithmetic) {
        self.arithmetic = arithmetic;
        self.native.set_arithmetic(arithmetic);
        self.program.set_arithmetic(arithmetic);
    }

    fn pass(&mut self, inputs: &FamilyInputs, learn: bool) -> Result<Tally, String> {
        let d = self.device.clone();
        let (rows, arithmetic) = (inputs.rows, self.arithmetic);
        // Each phase's wall time, at debug level (the device synchronised at its end).
        let timed = log::log_enabled!(log::Level::Debug);
        let mut clock = std::time::Instant::now();
        let mut lap = |phase: &str| -> Result<(), String> {
            if timed {
                d.synchronize().map_err(error)?;
                log::debug!("device train: {phase} {:.1} ms on {rows} rows", 1e3 * clock.elapsed().as_secs_f64());
                clock = std::time::Instant::now();
            }
            Ok(())
        };
        let target = {
            let trace = self.native.forward(inputs)?;
            self.native.logits_on_device(&trace)?
        };
        lap("model forward")?;
        // Each site's `U F̄` from its factors as they are.
        let ufs: Vec<Tensor> = self
            .sites
            .iter()
            .map(|site| {
                let terms: Vec<(&Tensor, &Tensor)> = site.writes.iter().zip(&site.fisher_rows).map(|(b, f)| Ok((self.program.dense(b.op)?, f))).collect::<Result<_, String>>()?;
                products(&d, terms, (Op::T, Op::N), 1.0, arithmetic)
            })
            .collect::<Result<_, _>>()?;
        // The corner run: every site's sets chosen from the read its own program computed.
        let mut family = inputs.clone();
        let slots = self.sites.iter().map(|s| s.slot + 1).max().unwrap_or(0);
        while family.slots.len() < slots {
            family.slots.push(SlotValues::Raw(Array2::zeros((0, 0))));
        }
        let gates = self.masked.gates();
        let mut listed = vec![(0.0, 0.0); self.sites.len()];
        let (sites, masked) = (&self.sites, &self.masked);
        let masked_trace = self.program.forward_gated(&family, BTreeMap::new(), &gates, |z_node, trace| {
            let k = masked.z.iter().position(|n| *n == z_node).ok_or_else(|| error("an unknown gated amplitude"))?;
            let site = &sites[k];
            let coder = site.coder.as_ref().ok_or_else(|| error("a site without its code"))?;
            let d_out = site.fisher.rows();
            // The site's output `y = W x`, `U F̄ y` and `yᵀ F̄ y` from its read.
            let mut y = d.zeros(rows, d_out).map_err(error)?;
            for (b, w) in site.reads.iter().zip(&site.w) {
                d.gemm(&mut y, 1.0, trace.value(b.node)?, Op::N, w, Op::T, 1.0, arithmetic).map_err(error)?;
            }
            let weights = product(&d, &y, Op::N, &ufs[k], Op::T, 1.0, arithmetic)?;
            let yf = product(&d, &y, Op::N, &site.fisher, Op::N, 1.0, arithmetic)?;
            let mut both = d.zeros(rows, d_out).map_err(error)?;
            d.hadamard(&mut both, &y, &yf, false).map_err(error)?;
            let yfy = product(&d, &both, Op::N, &site.ones, Op::N, 1.0, arithmetic)?;
            let (on, _, _) = coder.code(&d, (trace.value(z_node)?, &weights, &yfy))?;
            let on = coder.columns(&d, on)?;
            let counted = d.download(&product(&d, &on, Op::N, &site.listing, Op::N, 1.0, exact(&d))?).map_err(error)?;
            listed[k] = (counted.column(0).sum(), counted.column(1).sum());
            Ok(on)
        })?;
        drop(ufs);
        let mut tally = Tally { rows, ..Tally::default() };
        for (l0, bits) in listed {
            tally.l0 += l0;
            tally.description += bits;
        }
        lap("corner run")?;
        if !learn {
            tally.kl = self.program.score_only(&masked_trace, &target, None)?.sum();
            return Ok(tally);
        }
        let (kl, hidden) = self.program.kl(&masked_trace, &target, None)?;
        drop(target);
        tally.kl = kl.sum();
        let keep: Vec<usize> = self.sites.iter().flat_map(|s| std::iter::once(s.z).chain(s.writes.iter().map(|b| b.node))).collect();
        let back = self.program.vjp_seeded(&masked_trace, hidden, BTreeMap::new(), &keep, arithmetic)?;
        for site in &mut self.sites {
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
        }
        lap("reverse and gradients")?;
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
    /// site's map restored to first order (module note, "The map").
    pub fn update(&mut self) -> Result<(), String> {
        self.steps += 1;
        let d = self.device.clone();
        let Settings { betas: (beta1, beta2), epsilon, .. } = self.settings;
        for k in 0..self.sites.len() {
            let site = &mut self.sites[k];
            let mut steps = Vec::new();
            for b in site.reads.iter_mut().chain(site.writes.iter_mut()) {
                let before = d.copy(self.program.dense(b.op)?).map_err(error)?;
                let (m, v) = &mut b.moments;
                let held = self.program.dense_mut(b.op)?;
                d.adam(held, (m, v), &b.gradient, b.rate, (beta1, beta2, epsilon), self.steps).map_err(error)?;
                let mut step = d.copy(held).map_err(error)?;
                d.axpy(&mut step, -1.0, &before).map_err(error)?;
                steps.push((before, step));
                b.gradient = d.zeros(b.gradient.rows(), b.gradient.cols()).map_err(error)?;
            }
            // What the step moved the map by, `ΔU_i V_j + U_i ΔV_j` (V after, U before): small
            // products, so single precision holds it to its own relative precision.
            let reads = site.reads.len();
            let (read_steps, write_steps) = steps.split_at(reads);
            let mut residuals = Vec::new();
            for (u_before, u_step) in write_steps {
                let mut row = Vec::new();
                for (read, (_, v_step)) in site.reads.iter().zip(read_steps) {
                    let mut r = product(&d, u_step, Op::N, self.program.dense(read.op)?, Op::N, -1.0, Arithmetic::F32)?;
                    d.gemm(&mut r, -1.0, u_before, Op::N, v_step, Op::N, 1.0, Arithmetic::F32).map_err(error)?;
                    row.push(r);
                }
                residuals.push(row);
            }
            drop(steps);
            self.retract(k, &residuals, Arithmetic::F32)?;
        }
        Ok(())
    }

    /// The residuals `S_ij − U_i V_j` of site `k`.
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

    /// Site `k`'s least change `(δU, δV)` (in the sum of both factors' squared entries) with `δU V
    /// + U δV = R`, `R` per written and read block: `δU = Λ Vᵀ`, `δV = Uᵀ Λ` with `U Uᵀ Λ + Λ Vᵀ V =
    /// R`, solved in the bases ([`Bases`]): `Λ = Q_U ((Q_Uᵀ R Q_V) ⊘ (λ_U + λ_V)) Q_Vᵀ`.
    fn retract(&mut self, k: usize, residuals: &[Vec<Tensor>], arithmetic: Arithmetic) -> Result<(), String> {
        let d = self.device.clone();
        let site = &self.sites[k];
        let bases = &site.bases;
        let mut core: Option<Tensor> = None;
        for (i, row) in residuals.iter().enumerate() {
            for (j, r) in row.iter().enumerate() {
                let right = product(&d, r, Op::N, &bases.v[j], Op::N, 1.0, arithmetic)?;
                match &mut core {
                    None => core = Some(product(&d, &bases.u[i], Op::T, &right, Op::N, 1.0, arithmetic)?),
                    Some(total) => d.gemm(total, 1.0, &bases.u[i], Op::T, &right, Op::N, 1.0, arithmetic).map_err(error)?,
                }
            }
        }
        let mut core = core.ok_or_else(|| error("a site without blocks"))?;
        d.divide_sums(&mut core, &bases.u_values, &bases.v_values, bases.floor).map_err(error)?;
        let us: Vec<&Tensor> = site.writes.iter().map(|b| self.program.dense(b.op)).collect::<Result<_, _>>()?;
        let vs: Vec<&Tensor> = site.reads.iter().map(|b| self.program.dense(b.op)).collect::<Result<_, _>>()?;
        // `Uᵀ Q_U` (pieces × d_out) and `Q_Vᵀ Vᵀ` (d_in × pieces).
        let u_turned = products(&d, us.iter().copied().zip(&bases.u), (Op::T, Op::N), 1.0, arithmetic)?;
        let v_turned = products(&d, bases.v.iter().zip(vs.iter().copied()), (Op::T, Op::T), 1.0, arithmetic)?;
        let core_v = product(&d, &core, Op::N, &v_turned, Op::N, 1.0, arithmetic)?;
        let u_core = product(&d, &u_turned, Op::N, &core, Op::N, 1.0, arithmetic)?;
        let write_moves = bases.u.iter().map(|q| product(&d, q, Op::N, &core_v, Op::N, 1.0, arithmetic)).collect::<Result<Vec<_>, _>>()?;
        let read_moves = bases.v.iter().map(|q| product(&d, &u_core, Op::N, q, Op::T, 1.0, arithmetic)).collect::<Result<Vec<_>, _>>()?;
        let ops: Vec<usize> = site.writes.iter().chain(&site.reads).map(|b| b.op).collect();
        for (op, change) in ops.into_iter().zip(write_moves.iter().chain(&read_moves)) {
            d.axpy(self.program.dense_mut(op)?, 1.0, change).map_err(error)?;
        }
        Ok(())
    }

    /// The largest entry of site `k`'s residuals.
    fn largest_residual(&self, k: usize, arithmetic: Arithmetic) -> Result<f64, String> {
        let mut largest = 0.0_f64;
        for row in self.residuals(k, arithmetic)? {
            for r in row {
                largest = self.device.download(&r).map_err(error)?.iter().fold(largest, |m, x| m.max(x.abs()));
            }
        }
        Ok(largest)
    }

    /// Every site's bases from its current factors, its map restored exactly (retractions while the
    /// residual shrinks), its library written into the masked program, its description bits priced
    /// again under `describe`, and its code (module note, "A step") formed from them.
    pub fn sync(&mut self, describe: &dyn Describe) -> Result<&Masked, String> {
        let d = self.device.clone();
        let exact = exact(&d);
        for k in 0..self.sites.len() {
            let site = &self.sites[k];
            let us: Vec<&Tensor> = site.writes.iter().map(|b| self.program.dense(b.op)).collect::<Result<_, _>>()?;
            let vs: Vec<&Tensor> = site.reads.iter().map(|b| self.program.dense(b.op)).collect::<Result<_, _>>()?;
            let decompose = |held: &[&Tensor], (ta, tb): (Op, Op)| -> Result<gam_linalg::decompose::Eigh, String> {
                let blocks: Vec<Vec<Tensor>> = held.iter().map(|a| held.iter().map(|b| product(&d, a, ta, b, tb, 1.0, exact)).collect()).collect::<Result<_, _>>()?;
                eigh(assemble(&d, &blocks)?.view(), SymmetricAssembly::Mirrored, None).map_err(|e| format!("{e:?}"))
            };
            let (u_gram, v_gram) = (decompose(&us, (Op::N, Op::T))?, decompose(&vs, (Op::T, Op::N))?);
            let (u_widths, v_widths): (Vec<usize>, Vec<usize>) = (us.iter().map(|t| t.rows()).collect(), vs.iter().map(|t| t.cols()).collect());
            let upload = |parts: Vec<Array2<f64>>| parts.iter().map(|p| d.upload(p.view()).map_err(error)).collect::<Result<Vec<_>, _>>();
            let bases = Bases {
                u: upload(split(&u_gram.vectors, &u_widths, false))?,
                u_values: d.upload_vec(u_gram.values.len(), 1, u_gram.values.to_vec()).map_err(error)?,
                v: upload(split(&v_gram.vectors, &v_widths, false))?,
                v_values: d.upload_vec(1, v_gram.values.len(), v_gram.values.to_vec()).map_err(error)?,
                floor: u_gram.band + v_gram.band,
            };
            self.sites[k].bases = bases;
            let mut largest = self.largest_residual(k, exact)?;
            while largest > 0.0 {
                let residuals = self.residuals(k, exact)?;
                self.retract(k, &residuals, exact)?;
                let next = self.largest_residual(k, exact)?;
                if next >= largest {
                    break;
                }
                largest = next;
            }
            let site = &self.sites[k];
            let download = |op: usize| -> Result<Array2<f64>, String> { d.download(self.program.dense(op)?).map_err(error) };
            let v_parts: Vec<Array2<f64>> = site.reads.iter().map(|b| download(b.op)).collect::<Result<_, _>>()?;
            let u_parts: Vec<Array2<f64>> = site.writes.iter().map(|b| download(b.op).map(|m| m.t().to_owned())).collect::<Result<_, _>>()?;
            let v = ndarray::concatenate(Axis(1), &v_parts.iter().map(|p| p.view()).collect::<Vec<_>>()).map_err(error)?;
            let u = ndarray::concatenate(Axis(1), &u_parts.iter().map(|p| p.view()).collect::<Vec<_>>()).map_err(error)?;
            let library = Library { mean: Array1::zeros(v.ncols()), v, u };
            let bits = description_bits(describe, k, &library)?;
            let pieces = bits.len();
            let gram = fast_abt(&fast_ab(&library.u, &self.sites[k].fisher_host), &library.u);
            let coder = super::sparse_code::Coder::new(gram, &vec![1; pieces], &bits, self.settings.observations, super::site_fit::NODES)?;
            self.sites[k].coder = Some(DeviceCoder::new(&d, coder)?);
            self.masked.set_library(k, library)?;
            let listing: Vec<f64> = bits.iter().flat_map(|b| [1.0, *b]).collect();
            self.sites[k].listing = d.upload_vec(pieces, 2, listing).map_err(error)?;
            self.sites[k].bits = d.upload_vec(1, pieces, bits).map_err(error)?;
        }
        Ok(&self.masked)
    }
}
