//! The explanation's core path on a device (#2951): the passages' targets, every replacement's
//! forward with its selection running where the program runs, and its KL, so an end-to-end
//! evaluation (`super::explanation::replaced`, `super::explanation::site_switch`) costs device
//! forwards, not host ones.
//!
//! # What runs where
//!
//! [`Evaluator`] lowers the model once ([`DeviceProgram`]) and every replaced program after it
//! shares the model's weights ([`DeviceProgram::compile_sharing`]). Passages run in chunks of whole
//! passages (one strided-batched attention per chunk), each chunk's target the model's own logits
//! held on the device. A replacement's forward ([`OnDevice`]) is the CPU's term for term:
//!
//! * [`Given`] sets (VPD's published ones) are uploaded once per passage and site and copied into
//!   each forward's mask slots;
//! * an [`Explanation`] chooses its blocks inside the forward (`DeviceProgram::forward_gated`): at
//!   each replaced site the read the explanation's own program computed gives `z = V x` (the
//!   program's amplitude node), the site's output `y = W x`, `U F y` and `yᵀ F y` as device
//!   products, and the site's selection codes every input from those
//!   (`super::site_fit::Selector::select_products`); its blocks on go back as the mask.
//!
//! # Precision
//!
//! On a float64 device the forward, the products the selection reads and the KL are float64, the
//! same quantities as the CPU's up to the summation order of their reductions. An [`Evaluator`]
//! may run its products in f32 or TF32 ([`Evaluator::set_arithmetic`]) to search (the site-switch
//! claim's subsets); what is reported is re-run in float64. The Apple GPU has no float64: there
//! every product is f32 (`gam_gpu::tensor`), and its numbers carry that precision.

use super::device_program::{DeviceProgram, DeviceTrace};
use super::explanation::{Explanation, Fitted, Given, Passage, Replacement};
use super::masked::{Masked, Site, Target};
use super::site_fit::Samples;
use super::operator_program::{FamilyInputs, OperatorProgram};
use gam_gpu::tensor::{Arithmetic, Device, Op, Tensor};
use ndarray::{Array1, Array2, Axis, s};
use std::collections::BTreeMap;
use std::sync::{Arc, Mutex};

fn error(e: impl std::fmt::Display) -> String {
    format!("core device: {e}")
}

/// Which devices the core path may run on.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Choice {
    /// Only a float64 device (a CUDA accelerator): the default, so what the CPU would report is
    /// what the device reports.
    Float64,
    /// Any device, the Apple GPU's f32 included (module note, "Precision").
    Any,
    /// The CPU.
    Off,
}

impl Choice {
    /// `f64`, `any` or `off`.
    pub fn parse(raw: &str) -> Result<Self, String> {
        match raw {
            "f64" => Ok(Self::Float64),
            "any" => Ok(Self::Any),
            "off" => Ok(Self::Off),
            other => Err(format!("device {other}: neither f64, any nor off")),
        }
    }
}

static CHOICE: std::sync::OnceLock<Choice> = std::sync::OnceLock::new();

/// Set the process's [`Choice`] before the core path first asks for its device (the first call
/// wins; the default is [`Choice::Float64`]).
pub fn choose(choice: Choice) {
    if CHOICE.set(choice).is_err() && CHOICE.get() != Some(&choice) {
        log::warn!("core device: already chose {:?}, {choice:?} ignored", CHOICE.get());
    }
}

/// The process's device for the core path under its [`Choice`] and `gam_gpu`'s global policy:
/// `None` runs the CPU path.
pub fn device() -> Result<Option<Device>, String> {
    match CHOICE.get().copied().unwrap_or(Choice::Float64) {
        Choice::Float64 => super::masked_device::device(),
        Choice::Any => super::masked_device::training_device(),
        Choice::Off => Ok(None),
    }
}

/// The bytes of device memory a chunk's forward may take: a quarter of what is free, and on a
/// device that does not say, 4 GiB.
fn chunk_budget(device: &Device) -> Result<usize, String> {
    Ok(device.memory().map_err(error)?.map_or(4 << 30, |(free, _)| free / 4))
}

/// The passages' bases each with the model's logits on it, every forward on `device` (float64
/// where it has it): `super::explanation::Passage::new` for many passages at once.
pub fn passages(device: &Device, model: &OperatorProgram, bases: Vec<FamilyInputs>) -> Result<Vec<Passage>, String> {
    let native = DeviceProgram::compile(device, model)?;
    let rows = bases.iter().map(|b| b.rows).max().unwrap_or(1).max(1);
    let per_chunk = (chunk_budget(device)? / (rows * (native.bytes_per_row() + 8 * native.classes())).max(1)).max(1);
    let mut out = Vec::with_capacity(bases.len());
    let mut bases = bases.into_iter().peekable();
    while bases.peek().is_some() {
        let chunk: Vec<FamilyInputs> = bases.by_ref().take(per_chunk).collect();
        let family = append(&chunk)?;
        let trace = native.forward(&family)?;
        let mut at = 0;
        for base in chunk {
            let logits = native.logits(&trace, at, base.rows)?;
            at += base.rows;
            out.push(Passage { base, target: Target::every_row(logits) });
        }
    }
    Ok(out)
}

/// `families` one after another (sequences kept apart).
fn append(families: &[FamilyInputs]) -> Result<FamilyInputs, String> {
    let (first, rest) = families.split_first().ok_or("core device: no rows")?;
    rest.iter().try_fold(first.clone(), |all, f| all.append(f).map_err(|e| e.to_string()))
}

/// Some passages run as one forward: their indices and their rows one after another.
pub struct Chunk<'c> {
    pub passages: &'c [usize],
    pub base: &'c FamilyInputs,
}

/// How a replacement's forward chooses its blocks on a device (module note).
pub trait OnDevice {
    /// One forward of `program` (lowered from `masked`, [`Replacement::masked`] of `members`) on
    /// `chunk`: the trace and, with `keep`, every member's blocks on (rows × blocks; else they may
    /// be left empty).
    fn forward_on(
        &self,
        evaluator: &Evaluator<'_>,
        lowered: (&DeviceProgram, &Masked),
        members: &[usize],
        chunk: Chunk<'_>,
        keep: bool,
    ) -> Result<(DeviceTrace, Vec<Array2<f64>>), String>;
}

/// What a replaced site's selection reads, held on the device: `W` in one block per read node
/// (`d_out × d_j`), the written Fisher `F`, `U F` (columns × d_out) and a column of ones.
struct Resident {
    w: Vec<Tensor>,
    f: Tensor,
    uf: Tensor,
    ones: Tensor,
    coder: DeviceCoder,
}

impl Resident {
    fn new(device: &Device, fitted: &Fitted, widths: &[usize]) -> Result<Self, String> {
        let mut w = Vec::new();
        let mut at = 0;
        for width in widths {
            w.push(device.upload(fitted.w.slice(s![.., at..at + width])).map_err(error)?);
            at += width;
        }
        let uf = fitted.library.u.dot(&fitted.fisher);
        let d_out = fitted.w.nrows();
        Ok(Self {
            w,
            f: device.upload(fitted.fisher.view()).map_err(error)?,
            uf: device.upload(uf.view()).map_err(error)?,
            ones: device.upload_vec(d_out, 1, vec![1.0; d_out]).map_err(error)?,
            coder: DeviceCoder::new(device, fitted.coder().clone())?,
        })
    }
}

/// Passages on a device, evaluated under any replacement (module note).
pub struct Evaluator<'a> {
    model: &'a OperatorProgram,
    passages: &'a [Passage],
    native: DeviceProgram,
    arithmetic: Mutex<Arithmetic>,
    /// Each passage's target on the device.
    targets: Vec<Tensor>,
    /// Given sets uploaded, per (replacement, site, passage), expanded to their columns.
    given: Mutex<BTreeMap<(usize, usize, usize), Arc<Tensor>>>,
    selections: Selections,
}

impl<'a> Evaluator<'a> {
    /// `passages` of `model` on `device`, their targets uploaded, every product float64 (where
    /// the device has it).
    pub fn new(device: &Device, model: &'a OperatorProgram, passages: &'a [Passage]) -> Result<Self, String> {
        let native = DeviceProgram::compile(device, model)?;
        let targets = passages.iter().map(|p| device.upload(p.target.logits.view()).map_err(error)).collect::<Result<_, _>>()?;
        Ok(Self { model, passages, native, arithmetic: Mutex::new(Arithmetic::F64), targets, given: Mutex::new(BTreeMap::new()), selections: Selections::default() })
    }

    /// Run the forwards' products in `arithmetic` from now on (module note, "Precision").
    pub fn set_arithmetic(&self, arithmetic: Arithmetic) {
        *self.arithmetic.lock().unwrap_or_else(std::sync::PoisonError::into_inner) = arithmetic;
    }

    /// The products' arithmetic (f32 on a device without float64).
    #[must_use]
    pub fn arithmetic(&self) -> Arithmetic {
        if self.native.device().float64() { *self.arithmetic.lock().unwrap_or_else(std::sync::PoisonError::into_inner) } else { Arithmetic::F32 }
    }

    /// The device.
    #[must_use]
    pub fn device(&self) -> &Device {
        self.native.device()
    }

    /// Per passage of `which`, `KL(model ‖ replacement)` per row with the sites `members` of
    /// `replacement` replaced, and every member's blocks on: `super::explanation::replaced`.
    pub fn replaced(&self, replacement: &dyn Replacement, members: &[usize], which: &[usize]) -> Result<Vec<(Array1<f64>, Vec<Array2<f64>>)>, String> {
        self.run(replacement, members, which, true)
    }

    /// [`Evaluator::replaced`]'s KL alone (the blocks on stay on the device).
    pub fn kls(&self, replacement: &dyn Replacement, members: &[usize], which: &[usize]) -> Result<Vec<Array1<f64>>, String> {
        Ok(self.run(replacement, members, which, false)?.into_iter().map(|(kl, _)| kl).collect())
    }

    fn run(&self, replacement: &dyn Replacement, members: &[usize], which: &[usize], keep: bool) -> Result<Vec<(Array1<f64>, Vec<Array2<f64>>)>, String> {
        let masked = replacement.masked(self.model, members)?;
        let mut program = DeviceProgram::compile_sharing(&self.native, &masked.program)?;
        program.set_arithmetic(self.arithmetic());
        let d = self.device();
        let rows = which.iter().map(|p| self.passages[*p].base.rows).max().unwrap_or(1).max(1);
        let per_chunk = (chunk_budget(d)? / (rows * (program.bytes_per_row() + 8 * program.classes())).max(1)).max(1);
        let mut out = Vec::with_capacity(which.len());
        for chunk in which.chunks(per_chunk) {
            let bases: Vec<FamilyInputs> = chunk.iter().map(|p| self.passages[*p].base.clone()).collect();
            let base = append(&bases)?;
            let mut target = d.zeros(base.rows, program.classes()).map_err(error)?;
            let mut at = 0;
            for &p in chunk {
                d.set_rows(&mut target, at, &self.targets[p]).map_err(error)?;
                at += self.passages[p].base.rows;
            }
            let (trace, masks) = replacement.forward_on(self, (&program, &masked), members, Chunk { passages: chunk, base: &base }, keep)?;
            let kl = program.score_only(&trace, &target, None)?;
            drop((trace, target));
            let mut at = 0;
            for &p in chunk {
                let n = self.passages[p].base.rows;
                let rows = |m: &Array2<f64>| if m.nrows() == base.rows { m.slice(s![at..at + n, ..]).to_owned() } else { Array2::zeros((0, m.ncols())) };
                out.push((kl.slice(s![at..at + n]).to_owned(), masks.iter().map(rows).collect()));
                at += n;
            }
        }
        Ok(out)
    }

    /// Given sets of site `site` of `owner` on passage `passage`, expanded to columns, uploaded once.
    fn given_mask(&self, owner: usize, site: usize, passage: usize, upload: impl FnOnce() -> Result<Tensor, String>) -> Result<Arc<Tensor>, String> {
        let key = (owner, site, passage);
        if let Some(t) = self.given.lock().map_err(|_| "core device: a poisoned cache".to_string())?.get(&key) {
            return Ok(Arc::clone(t));
        }
        let t = Arc::new(upload()?);
        self.given.lock().map_err(|_| "core device: a poisoned cache".to_string())?.insert(key, Arc::clone(&t));
        Ok(t)
    }
}

impl OnDevice for Given {
    fn forward_on(&self, evaluator: &Evaluator<'_>, (program, masked): (&DeviceProgram, &Masked), members: &[usize], chunk: Chunk<'_>, keep: bool) -> Result<(DeviceTrace, Vec<Array2<f64>>), String> {
        let (d, Chunk { passages, base }) = (evaluator.device(), chunk);
        let owner = std::ptr::from_ref(self) as usize;
        let mut slots = BTreeMap::new();
        let mut chosen = Vec::with_capacity(members.len());
        for (k, &m) in members.iter().enumerate() {
            let mut column = d.zeros(base.rows, masked.pieces(k)).map_err(error)?;
            let mut blocks = Vec::with_capacity(passages.len());
            let mut at = 0;
            for &p in passages {
                let given = &self.masks.get(p).ok_or_else(|| format!("no given sets for passage {p}"))?[m];
                let part = evaluator.given_mask(owner, m, p, || d.upload(masked.expand(k, given).view()).map_err(error))?;
                d.set_rows(&mut column, at, &part).map_err(error)?;
                at += given.nrows();
                blocks.push(given.view());
            }
            if at != base.rows {
                return Err(format!("given sets cover {at} of {} rows", base.rows));
            }
            slots.insert(masked.slots[k], column);
            chosen.push(if keep { ndarray::concatenate(Axis(0), &blocks).map_err(|e| e.to_string())? } else { Array2::zeros((0, masked.blocks(k))) });
        }
        Ok((program.forward_given(base, slots)?, chosen))
    }
}

impl OnDevice for Explanation {
    fn forward_on(&self, evaluator: &Evaluator<'_>, lowered: (&DeviceProgram, &Masked), members: &[usize], chunk: Chunk<'_>, keep: bool) -> Result<(DeviceTrace, Vec<Array2<f64>>), String> {
        evaluator.selections.forward(lowered, (self, members), chunk.base, evaluator.arithmetic(), keep)
    }
}

/// The explanation's fitted sites' selections held on a device, by the fitted site's address.
#[derive(Default)]
pub struct Selections {
    resident: Mutex<BTreeMap<usize, Arc<Resident>>>,
}

impl Selections {
    fn resident(&self, device: &Device, fitted: &Fitted, widths: &[usize]) -> Result<Arc<Resident>, String> {
        let key = std::ptr::from_ref(fitted) as usize;
        let mut held = self.resident.lock().map_err(|_| "core device: a poisoned cache".to_string())?;
        if let Some(r) = held.get(&key) {
            return Ok(Arc::clone(r));
        }
        let r = Arc::new(Resident::new(device, fitted, widths)?);
        held.insert(key, Arc::clone(&r));
        Ok(r)
    }

    /// `explanation`'s autonomous forward of `program` (lowered from `masked`, its sites `members`
    /// replaced) on `base`: `Explanation::execute` on the device, its products in `arithmetic`.
    /// Returns the trace and, with `keep`, every member's blocks on (else empty).
    pub fn forward(
        &self,
        (program, masked): (&DeviceProgram, &Masked),
        (explanation, members): (&Explanation, &[usize]),
        base: &FamilyInputs,
        arithmetic: Arithmetic,
        keep: bool,
    ) -> Result<(DeviceTrace, Vec<Array2<f64>>), String> {
        let d = program.device();
        let gates = masked.gates();
        if gates.len() != members.len() || masked.sites.len() != members.len() {
            return Err("a masked site without its gate".to_string());
        }
        let mut chosen: Vec<Array2<f64>> = (0..members.len()).map(|k| Array2::zeros((0, masked.blocks(k)))).collect();
        let trace = program.forward_gated(base, BTreeMap::new(), &gates, |z_node, trace| {
            let k = masked.z.iter().position(|n| *n == z_node).ok_or("an unknown gated amplitude")?;
            let fitted = &explanation.sites[members[k]];
            let reads = &masked.sites[k].reads;
            let widths: Vec<usize> = reads.iter().map(|n| trace.value(*n).map(Tensor::cols)).collect::<Result<_, _>>()?;
            let resident = self.resident(d, fitted, &widths)?;
            // The site's output `y = W x`, `U F y` and `yᵀ F y` from the read its program computed.
            let d_out = fitted.w.nrows();
            let mut y = d.zeros(base.rows, d_out).map_err(error)?;
            for (n, w) in reads.iter().zip(&resident.w) {
                d.gemm(&mut y, 1.0, trace.value(*n)?, Op::N, w, Op::T, 1.0, arithmetic).map_err(error)?;
            }
            let mut weights = d.zeros(base.rows, fitted.library.u.nrows()).map_err(error)?;
            d.gemm(&mut weights, 1.0, &y, Op::N, &resident.uf, Op::T, 0.0, arithmetic).map_err(error)?;
            let mut yf = d.zeros(base.rows, d_out).map_err(error)?;
            d.gemm(&mut yf, 1.0, &y, Op::N, &resident.f, Op::N, 0.0, arithmetic).map_err(error)?;
            let mut product = d.zeros(base.rows, d_out).map_err(error)?;
            d.hadamard(&mut product, &y, &yf, false).map_err(error)?;
            drop((y, yf));
            let mut yfy = d.zeros(base.rows, 1).map_err(error)?;
            d.gemm(&mut yfy, 1.0, &product, Op::N, &resident.ones, Op::N, 0.0, arithmetic).map_err(error)?;
            drop(product);
            let (on, _, _) = resident.coder.code(d, (trace.value(z_node)?, &weights, &yfy))?;
            if keep {
                chosen[k] = d.download(&on).map_err(error)?;
            }
            resident.coder.columns(d, on)
        })?;
        Ok((trace, chosen))
    }
}

/// Mirrors the CPU's label sampler (`masked::sampled_label_cotangent`), so the device draws the
/// CPU's labels.
struct Uniforms(u64);

impl Uniforms {
    fn next(&mut self) -> f64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        (self.0 >> 11) as f64 / (1u64 << 53) as f64
    }
}

/// `site_fit::samples` of `sites` (in `program`'s nodes) on `batches`, `program` the model with
/// `explanation`'s fitted sites replaced (`masked`, every one of them, each running its own
/// selection) or the model itself (`masked` `None`): every forward, sampled label and reverse
/// pass, and the second moments and Fishers they sum to, on `device` in float64 (where it has
/// it). The labels are the CPU's (the same seeds, in the same order), so the statistics are the
/// CPU's up to the summation order of the products.
pub fn samples(
    device: &Device,
    program: &OperatorProgram,
    (masked, explanation): (Option<&Masked>, &Explanation),
    sites: &[Site],
    batches: &[FamilyInputs],
    draws: usize,
    seed: u64,
) -> Result<Vec<Samples>, String> {
    if draws == 0 {
        return Err("samples need at least one draw".to_string());
    }
    let members: Vec<usize> = (0..explanation.sites.len()).collect();
    if masked.is_some_and(|m| m.sites.len() != members.len()) || (masked.is_none() && !members.is_empty()) {
        return Err("samples: the hybrid program does not replace every fitted site".to_string());
    }
    let mapped = sites;
    let lowered = DeviceProgram::compile(device, program)?;
    let arithmetic = if device.float64() { Arithmetic::F64 } else { Arithmetic::F32 };
    let d = device;
    let widths = lowered.widths();
    let selections = Selections::default();
    let keep: Vec<usize> = mapped.iter().flat_map(|s| s.writes.iter().copied()).collect();
    // Per site: its reads per batch, each read's norm, and the blocks `x_iᵀ x_j`, `g_iᵀ g_j`.
    let mut reads: Vec<Vec<Array2<f32>>> = vec![Vec::new(); mapped.len()];
    let mut norms: Vec<Vec<f64>> = vec![Vec::new(); mapped.len()];
    let blocks = |nodes: &[usize]| -> Result<Vec<Vec<Tensor>>, String> {
        nodes.iter().map(|i| nodes.iter().map(|j| d.zeros(widths[*i], widths[*j]).map_err(error)).collect()).collect()
    };
    let mut moments: Vec<Vec<Vec<Tensor>>> = mapped.iter().map(|s| blocks(&s.reads)).collect::<Result<_, _>>()?;
    let mut fishers: Vec<Vec<Vec<Tensor>>> = mapped.iter().map(|s| blocks(&s.writes)).collect::<Result<_, _>>()?;
    let ones: BTreeMap<usize, Tensor> =
        keep.iter().map(|w| Ok((widths[*w], d.upload_vec(widths[*w], 1, vec![1.0; widths[*w]]).map_err(error)?))).collect::<Result<_, String>>()?;
    // Batches run together, as many as a chunk holds (a forward and its reverse passes resident);
    // each batch's labels are still its own draws, the CPU's.
    let per_row = 3 * lowered.bytes_per_row() + 8 * lowered.classes();
    let budget = chunk_budget(d)?;
    let mut chunks: Vec<Vec<usize>> = Vec::new();
    for (b, batch) in batches.iter().enumerate() {
        match chunks.last_mut() {
            Some(chunk) if (chunk.iter().map(|i| batches[*i].rows).sum::<usize>() + batch.rows) * per_row <= budget => chunk.push(b),
            _ => chunks.push(vec![b]),
        }
    }
    for chunk in chunks {
        let parts: Vec<FamilyInputs> = chunk.iter().map(|b| batches[*b].clone()).collect();
        let batch = &append(&parts)?;
        let rows = batch.rows;
        let trace = match masked {
            None => lowered.forward(batch)?,
            Some(m) => selections.forward((&lowered, m), (explanation, &members), batch, arithmetic, false)?.0,
        };
        for (k, site) in mapped.iter().enumerate() {
            let parts: Vec<Array2<f64>> = site.reads.iter().map(|n| d.download(trace.value(*n)?).map_err(error)).collect::<Result<_, _>>()?;
            let views: Vec<_> = parts.iter().map(|p| p.view()).collect();
            reads[k].push(ndarray::concatenate(Axis(1), &views).map_err(|e| e.to_string())?.mapv(|v| v as f32));
            norms[k].extend(std::iter::repeat_n(0.0, rows));
            for (i, a) in site.reads.iter().enumerate() {
                for (j, b) in site.reads.iter().enumerate().skip(i) {
                    d.gemm(&mut moments[k][i][j], 1.0, trace.value(*a)?, Op::T, trace.value(*b)?, Op::N, 1.0, arithmetic).map_err(error)?;
                }
            }
        }
        // Batch `b`'s draw `j` is the CPU's `b draws + j + 1`th.
        let uniforms: Vec<Vec<f64>> = (0..draws)
            .map(|j| {
                chunk
                    .iter()
                    .flat_map(|&b| {
                        let drawn = (b * draws + j + 1) as u64;
                        let mut rng = Uniforms(seed.wrapping_add(drawn.wrapping_mul(0x9E37_79B9)) | 1);
                        (0..batches[b].rows).map(move |_| rng.next())
                    })
                    .collect()
            })
            .collect();
        for g_hidden in lowered.sampled_many(&trace, &uniforms, None)? {
            let back = lowered.vjp(&trace, g_hidden, &keep, arithmetic)?;
            for (k, site) in mapped.iter().enumerate() {
                let first = norms[k].len() - rows;
                let mut squares = d.zeros(rows, 1).map_err(error)?;
                for (i, a) in site.writes.iter().enumerate() {
                    let Some(ga) = back.get(a) else { continue };
                    let mut product = d.zeros(rows, widths[*a]).map_err(error)?;
                    d.hadamard(&mut product, ga, ga, false).map_err(error)?;
                    d.gemm(&mut squares, 1.0, &product, Op::N, &ones[&widths[*a]], Op::N, 1.0, arithmetic).map_err(error)?;
                    for (j, b) in site.writes.iter().enumerate().skip(i) {
                        let Some(gb) = back.get(b) else { continue };
                        d.gemm(&mut fishers[k][i][j], 1.0, ga, Op::T, gb, Op::N, 1.0, arithmetic).map_err(error)?;
                    }
                }
                for (n, s) in norms[k][first..].iter_mut().zip(d.download(&squares).map_err(error)?.column(0)) {
                    *n += s / draws as f64;
                }
            }
        }
    }
    let rows = norms.first().map_or(0, Vec::len);
    if rows == 0 {
        return Err("samples need inputs".to_string());
    }
    // The blocks assembled whole (the lower ones the upper's transposes).
    let whole = |blocks: &[Vec<Tensor>], nodes: &[usize]| -> Result<Array2<f64>, String> {
        let offsets: Vec<usize> = std::iter::once(0)
            .chain(nodes.iter().scan(0, |a, n| {
                *a += widths[*n];
                Some(*a)
            }))
            .collect();
        let mut out = Array2::<f64>::zeros((offsets[nodes.len()], offsets[nodes.len()]));
        for i in 0..nodes.len() {
            for j in i..nodes.len() {
                let values = d.download(&blocks[i][j]).map_err(error)?;
                out.slice_mut(s![offsets[i]..offsets[i + 1], offsets[j]..offsets[j + 1]]).assign(&values);
                if i != j {
                    out.slice_mut(s![offsets[j]..offsets[j + 1], offsets[i]..offsets[i + 1]]).assign(&values.t());
                }
            }
        }
        Ok(out)
    };
    mapped
        .iter()
        .enumerate()
        .map(|(k, site)| {
            let views: Vec<_> = reads[k].iter().map(|p| p.view()).collect();
            let reads = ndarray::concatenate(Axis(0), &views).map_err(|e| e.to_string())?;
            let fisher = whole(&fishers[k], &site.writes)? / (rows * draws) as f64;
            let mean_norm = fisher.diag().sum();
            let sensitivity = Array1::from_iter(norms[k].iter().map(|n| if mean_norm > 0.0 { n / mean_norm } else { 1.0 }));
            Ok(Samples { reads, sensitivity, fisher, second_moment: whole(&moments[k], &site.reads)? / rows as f64 })
        })
        .collect()
}

/// A site's sparse code held on a device (`super::sparse_code::Coder` there): its pieces' Gram in
/// the metric, its blocks and their bits, and for blocks wider than a column the map from blocks to
/// columns (blocks × columns).
pub struct DeviceCoder {
    coder: super::sparse_code::Coder,
    gram: Tensor,
    starts: gam_gpu::tensor::Indices,
    bits: Tensor,
    expansion: Option<Tensor>,
}

impl DeviceCoder {
    /// `coder` on `device`.
    pub fn new(device: &Device, coder: super::sparse_code::Coder) -> Result<Self, String> {
        let starts: Vec<u32> = coder.starts().iter().map(|s| u32::try_from(*s).map_err(|e| e.to_string())).collect::<Result<_, _>>()?;
        let (blocks, columns) = (coder.blocks(), *coder.starts().last().unwrap_or(&0));
        let expansion = (blocks != columns)
            .then(|| {
                let map = Array2::from_shape_fn((blocks, columns), |(b, c)| f64::from(u8::from(starts[b] as usize <= c && c < starts[b + 1] as usize)));
                device.upload(map.view()).map_err(error)
            })
            .transpose()?;
        Ok(Self {
            gram: device.upload(coder.gram().view()).map_err(error)?,
            starts: device.upload_indices(&starts).map_err(error)?,
            bits: device.upload_vec(1, blocks, coder.bits().to_vec()).map_err(error)?,
            expansion,
            coder,
        })
    }

    /// Every row's code from its products on the device (`z`, `weights` rows × columns, `yfy` rows
    /// × 1): its blocks on (rows × blocks, on the device) and per row its code and a lower bound on
    /// its best. The device's kernel where it has one (`Device::code_rows`), else the CPU's coder
    /// on the products brought back.
    pub fn code(&self, device: &Device, (z, weights, yfy): (&Tensor, &Tensor, &Tensor)) -> Result<(Tensor, Vec<f64>, Vec<f64>), String> {
        let rows = z.rows();
        let mut on = device.zeros(rows, self.coder.blocks()).map_err(error)?;
        match device.code_rows((z, weights, yfy), (&self.gram, &self.starts, &self.bits), None, (self.coder.kappa(), self.coder.nodes(), 1e-12), &mut on) {
            Ok((upper, lower)) => Ok((on, upper, lower)),
            Err(gam_gpu::GpuError::NoDeviceKernel { .. }) => {
                let (z, weights) = (device.download(z).map_err(error)?, device.download(weights).map_err(error)?);
                let yfy = device.download(yfy).map_err(error)?.column(0).to_vec();
                let coded = self.coder.code_rows(z.view(), weights.view(), &yfy, None)?;
                let blocks = self.coder.blocks();
                let sets = Array2::from_shape_fn((rows, blocks), |(r, b)| f64::from(u8::from(coded[r].0[b])));
                let (upper, lower) = coded.iter().map(|(_, u, l)| (*u, *l)).unzip();
                Ok((device.upload(sets.view()).map_err(error)?, upper, lower))
            }
            Err(e) => Err(error(e)),
        }
    }

    /// Blocks on (rows × blocks) as the masked program's column gates (rows × columns).
    pub fn columns(&self, device: &Device, on: Tensor) -> Result<Tensor, String> {
        let Some(expansion) = &self.expansion else { return Ok(on) };
        let mut out = device.zeros(on.rows(), expansion.cols()).map_err(error)?;
        let exact = if device.float64() { Arithmetic::F64 } else { Arithmetic::F32 };
        device.gemm(&mut out, 1.0, &on, Op::N, expansion, Op::N, 0.0, exact).map_err(error)?;
        Ok(out)
    }
}
