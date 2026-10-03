//! The masked fit's hot path on a device (#2951): the masked forward and its KL, the mask
//! gradients, the sampled-label Fishers, the pieces' gradients and the step's curvature, each the
//! same quantity as its CPU twin in `masked` (`forward`, `mask_gradients`, `fisher`, `gradients`,
//! the quadratic of `step_pieces`), computed by a [`DeviceProgram`] of the masked program.
//!
//! The forward and its KL, which decide (a selection keeps or refuses flips by them, a step is
//! accepted by them), are float64 throughout. Everything else proposes: its products run in the
//! `proposal` arithmetic the [`Accelerated`] was made with (f32 or TF32 on a device whose tensor
//! cores make that worth it), and a wrong proposal costs only the float64 test that refuses it.
//!
//! A family of many sequences runs as one batch: every node is one product over all their rows,
//! and their attention one strided-batched product, so a device is filled by sequences, not by
//! one sequence's 512 rows.

use super::device_program::{DeviceProgram, DeviceTrace};
use super::masked::{BoxGradients, Masked, Target};
use super::operator_program::{FamilyInputs, Node};
use gam_gpu::tensor::{Arithmetic, ColumnBlocks, Device, Op, Tensor};
use ndarray::{Array1, Array2, Axis, s};
use std::collections::BTreeMap;
use std::sync::OnceLock;

fn error(e: impl std::fmt::Display) -> String {
    format!("device: {e}")
}

/// The process's accelerator under `gam_gpu`'s global policy, resolved once: a CUDA device, or
/// `None`, where the masked fit runs on the CPU (`masked`, module note, "Devices").
pub fn device() -> Result<Option<Device>, String> {
    static DEVICE: OnceLock<Result<Option<Device>, String>> = OnceLock::new();
    DEVICE.get_or_init(|| Device::accelerator(gam_gpu::global_policy()).map_err(error)).clone()
}

/// A masked program lowered onto a device (module note).
pub struct Accelerated {
    program: DeviceProgram,
    proposal: Arithmetic,
    blocks: Vec<ColumnBlocks>,
    ranks: Vec<Vec<usize>>,
}

/// A target's logits held on the device, and its scored rows.
pub struct DeviceTarget {
    logits: Tensor,
    scored: Option<Vec<bool>>,
}

/// One masked forward on the device: the KL per row (float64, on the host), the trace, and the
/// KL's cotangent at the head's hidden node.
pub struct State {
    pub kl: Array1<f64>,
    pub trace: DeviceTrace,
    cotangent: Option<Tensor>,
}

/// Mirrors `masked`'s label sampler (`XorShift`), so the device draws the CPU's labels.
struct Uniforms(u64);

impl Uniforms {
    fn next(&mut self) -> f64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        (self.0 >> 11) as f64 / (1u64 << 53) as f64
    }
}

impl Accelerated {
    /// `masked` on `device`, its proposals' products in `proposal`; refused for a program the
    /// device cannot run (a frozen head after it, a node without a device rule).
    pub fn new(device: &Device, masked: &Masked, proposal: Arithmetic) -> Result<Self, String> {
        if masked.head.is_some() {
            return Err("device: a masked window with a frozen head after it".to_string());
        }
        let ranks: Vec<_> = (0..masked.sites.len()).map(|k| masked.ranks(k).to_vec()).collect();
        let blocks = ranks.iter().map(|r| device.column_blocks(r).map_err(error)).collect::<Result<_, _>>()?;
        Ok(Self { program: DeviceProgram::compile(device, &masked.program)?, proposal, blocks, ranks })
    }

    /// The device program.
    #[must_use]
    pub fn program(&self) -> &DeviceProgram {
        &self.program
    }

    /// Re-upload the operators `masked` now holds new copies of (after a step).
    pub fn refresh(&mut self, masked: &Masked) -> Result<(), String> {
        self.program.refresh(&masked.program)?;
        let ranks: Vec<_> = (0..masked.sites.len()).map(|k| masked.ranks(k).to_vec()).collect();
        if ranks != self.ranks {
            self.blocks = ranks.iter().map(|r| self.program.device().column_blocks(r).map_err(error)).collect::<Result<_, _>>()?;
            self.ranks = ranks;
        }
        Ok(())
    }

    /// `target` on the device.
    pub fn target(&self, target: &Target) -> Result<DeviceTarget, String> {
        Ok(DeviceTarget { logits: self.program.device().upload(target.logits.view()).map_err(error)?, scored: target.scored.clone() })
    }

    /// The masked forward on `family` (its masks in their slots, as `Masked::family` sets them).
    pub fn forward(&self, family: &FamilyInputs, target: &DeviceTarget) -> Result<State, String> {
        self.score_and_gradient(family, target)
    }

    /// Forward and KL with the hidden cotangent needed to propose an optimization step.
    pub fn score_and_gradient(&self, family: &FamilyInputs, target: &DeviceTarget) -> Result<State, String> {
        let trace = self.program.forward(family)?;
        let (kl, cotangent) = self.program.kl(&trace, &target.logits, target.scored.as_deref())?;
        Ok(State { kl, trace, cotangent: Some(cotangent) })
    }

    /// Retain the candidate forward for reuse, but defer its head gradient until it is needed.
    pub fn score_state(&self, family: &FamilyInputs, target: &DeviceTarget) -> Result<State, String> {
        let trace = self.program.forward(family)?;
        let kl = self.program.score_only(&trace, &target.logits, target.scored.as_deref())?;
        Ok(State { kl, trace, cotangent: None })
    }

    pub fn prepare_gradient(&self, state: &mut State, target: &DeviceTarget) -> Result<(), String> {
        if state.cotangent.is_none() {
            state.cotangent = Some(self.program.kl(&state.trace, &target.logits, target.scored.as_deref())?.1);
        }
        Ok(())
    }

    /// Evaluate a candidate without constructing or pulling back the KL cotangent.
    pub fn score_only(&self, family: &FamilyInputs, target: &DeviceTarget) -> Result<Array1<f64>, String> {
        let trace = self.program.forward(family)?;
        self.program.score_only(&trace, &target.logits, target.scored.as_deref())
    }

    fn cotangents(&self, trace: &DeviceTrace, seed: Tensor, keep: &[usize]) -> Result<BTreeMap<usize, Tensor>, String> {
        self.program.vjp(trace, seed, keep, self.proposal)
    }

    /// Per site `∂KL/∂m` (rows × B): `masked::mask_gradients` of the state's KL.
    pub fn mask_gradients(&self, masked: &Masked, state: &State) -> Result<Vec<Array2<f64>>, String> {
        let d = self.program.device();
        let seed = d.copy(state.cotangent.as_ref().ok_or("device: prepare the state's gradient first")?).map_err(error)?;
        let back = self.cotangents(&state.trace, seed, &masked.masked)?;
        let mut out = Vec::new();
        for k in 0..masked.sites.len() {
            let z = state.trace.value(masked.z[k])?;
            out.push(match back.get(&masked.masked[k]) {
                Some(c) => {
                    let g = d.block_products(c, z, &self.blocks[k]).map_err(error)?;
                    d.download(&g).map_err(error)?
                }
                None => Array2::zeros((z.rows(), masked.blocks(k))),
            });
        }
        Ok(out)
    }

    /// `masked::fisher`: per site the Fisher diagonal of every mask entry and, when `written`, the
    /// written nodes' Fisher, from `samples` sampled-label reverse passes seeded as the CPU's.
    pub fn fisher(
        &self,
        masked: &Masked,
        state: &State,
        target: &DeviceTarget,
        samples: usize,
        seed: u64,
        written: bool,
    ) -> Result<Vec<(Array2<f64>, Option<Array2<f64>>)>, String> {
        let d = self.program.device();
        let rows = state.trace.rows;
        if samples == 0 || target.scored.as_ref().is_some_and(|s| s.len() != rows) {
            return Err("device: Fisher needs samples and one scored flag per row".to_string());
        }
        let mut keep = masked.masked.clone();
        if written {
            keep.extend(masked.sites.iter().flat_map(|s| s.writes.iter().copied()));
        }
        let mut h: Vec<Tensor> = Vec::new();
        for k in 0..masked.sites.len() {
            let z = state.trace.value(masked.z[k])?;
            h.push(d.zeros(z.rows(), masked.blocks(k)).map_err(error)?);
        }
        // Per site, the blocks `g_iᵀ g_j` of the written nodes' Fisher.
        let mut blocks: Vec<Vec<Vec<Option<Tensor>>>> =
            masked.sites.iter().map(|s| (0..s.writes.len()).map(|_| (0..s.writes.len()).map(|_| None).collect()).collect()).collect();
        let mut rng = Uniforms(seed | 1);
        let scored = |r: usize| target.scored.as_ref().is_none_or(|s| s[r]);
        // Preserve the old sample-major RNG order, but share the expensive vocabulary work.
        // At most eight seeds and about 64 MiB at once (or one seed if it exceeds that).
        let seed_bytes = rows.saturating_mul(state.trace.value(self.program.hidden())?.cols()).saturating_mul(8).max(1);
        let batch = ((64 * 1024 * 1024) / seed_bytes).clamp(1, 8);
        for start in (0..samples).step_by(batch) {
            let uniforms: Vec<Vec<f64>> = (start..samples.min(start + batch))
                .map(|_| (0..rows).map(|r| if scored(r) { rng.next() } else { 0.0 }).collect()).collect();
            let seeds = self.program.sampled_many(&state.trace, &uniforms, target.scored.as_deref())?;
            for g_hidden in seeds {
                let back = self.cotangents(&state.trace, g_hidden, &keep)?;
                for (k, hk) in h.iter_mut().enumerate() {
                    if let Some(c) = back.get(&masked.masked[k]) {
                        let z = state.trace.value(masked.z[k])?;
                        let g = d.block_products(c, z, &self.blocks[k]).map_err(error)?;
                        d.hadamard(hk, &g, &g, true).map_err(error)?;
                    }
                    if !written {
                        continue;
                    }
                    let writes = &masked.sites[k].writes;
                    for (i, wi) in writes.iter().enumerate() {
                        for (j, wj) in writes.iter().enumerate() {
                            let (Some(gi), Some(gj)) = (back.get(wi), back.get(wj)) else { continue };
                            let block = &mut blocks[k][i][j];
                            if block.is_none() {
                                *block = Some(d.zeros(gi.cols(), gj.cols()).map_err(error)?);
                            }
                            let target = block.as_mut().ok_or("device: fisher block")?;
                            d.gemm(target, 1.0, gi, Op::T, gj, Op::N, 1.0, self.proposal).map_err(error)?;
                        }
                    }
                }
            }
        }
        let scored_rows = (0..rows).filter(|r| scored(*r)).count().max(1) as f64;
        let mut out = Vec::new();
        for (k, hk) in h.iter().enumerate() {
            let diagonal = d.download(hk).map_err(error)? / samples as f64;
            let fisher = if written {
                let widths: Vec<usize> = masked.sites[k].writes.iter().map(|w| state.trace.value(*w).map(|t| t.cols())).collect::<Result<_, _>>()?;
                let offsets: Vec<usize> = std::iter::once(0).chain(widths.iter().scan(0, |a, w| {
                    *a += w;
                    Some(*a)
                })).collect();
                let total = offsets[widths.len()];
                let mut f = Array2::<f64>::zeros((total, total));
                for (i, row) in blocks[k].iter().enumerate() {
                    for (j, block) in row.iter().enumerate() {
                        if let Some(b) = block {
                            f.slice_mut(s![offsets[i]..offsets[i + 1], offsets[j]..offsets[j + 1]]).assign(&d.download(b).map_err(error)?);
                        }
                    }
                }
                Some(f / (samples as f64 * scored_rows))
            } else {
                None
            };
            out.push((diagonal, fisher));
        }
        Ok(out)
    }

    /// `masked::gradients` of the state's KL: per site `(∂KL/∂m, ∂KL/∂V, ∂KL/∂U)`.
    pub fn gradients(&self, masked: &Masked, state: &State, masks: &[Array2<f64>]) -> Result<Vec<(Array2<f64>, Array2<f64>, Array2<f64>)>, String> {
        let d = self.program.device();
        let mut keep = masked.masked.clone();
        keep.extend(masked.sites.iter().flat_map(|s| s.writes.iter().copied()));
        let seed = d.copy(state.cotangent.as_ref().ok_or("device: prepare the state's gradient first")?).map_err(error)?;
        let back = self.cotangents(&state.trace, seed, &keep)?;
        let mut out = Vec::new();
        for (k, site) in masked.sites.iter().enumerate() {
            let (pieces, rows) = (masked.pieces(k), state.trace.rows);
            let z = state.trace.value(masked.z[k])?;
            let Node::Hadamard { right: mask_node, .. } = &masked.program.nodes[masked.masked[k]] else {
                return Err("device: a site's masked node is not its mask's product".to_string());
            };
            let mask = state.trace.value(*mask_node)?;
            if mask.dim() != (rows, pieces) || masks[k].dim() != (rows, masked.blocks(k)) {
                return Err("device: the state's masks are not the given ones".to_string());
            }
            let Some(cot_masked) = back.get(&masked.masked[k]) else {
                let width = |nodes: &[usize]| nodes.iter().map(|n| state.trace.value(*n).map(|t| t.cols())).sum::<Result<usize, _>>();
                let (d_in, d_out) = (width(&site.reads)?, width(&site.writes)?);
                out.push((Array2::zeros((rows, masked.blocks(k))), Array2::zeros((pieces, d_in)), Array2::zeros((pieces, d_out))));
                continue;
            };
            let mask_gradient = d.block_products(cot_masked, z, &self.blocks[k]).map_err(error)?;
            let mut cot_z = d.zeros(rows, pieces).map_err(error)?;
            d.hadamard(&mut cot_z, cot_masked, mask, false).map_err(error)?;
            // ∂KL/∂V = cot_zᵀ x, one block per read node (the reads are uncentred, `crate::masked`).
            let mut v_blocks = Vec::new();
            for read in &site.reads {
                let x = state.trace.value(*read)?;
                let mut block = d.zeros(pieces, x.cols()).map_err(error)?;
                d.gemm(&mut block, 1.0, &cot_z, Op::T, x, Op::N, 0.0, self.proposal).map_err(error)?;
                v_blocks.push(d.download(&block).map_err(error)?);
            }
            // ∂KL/∂U = z̃ᵀ g_written, one block per written node.
            let zm = state.trace.value(masked.masked[k])?;
            let mut u_blocks = Vec::new();
            for w in &site.writes {
                let width = state.trace.value(*w)?.cols();
                let block = match back.get(w) {
                    Some(g) => {
                        let mut block = d.zeros(pieces, width).map_err(error)?;
                        d.gemm(&mut block, 1.0, zm, Op::T, g, Op::N, 0.0, self.proposal).map_err(error)?;
                        d.download(&block).map_err(error)?
                    }
                    None => Array2::zeros((pieces, width)),
                };
                u_blocks.push(block);
            }
            let join = |blocks: &[Array2<f64>]| -> Result<Array2<f64>, String> {
                let views: Vec<_> = blocks.iter().map(|b| b.view()).collect();
                ndarray::concatenate(Axis(1), &views).map_err(|e| e.to_string())
            };
            out.push((d.download(&mask_gradient).map_err(error)?, join(&v_blocks)?, join(&u_blocks)?));
        }
        Ok(out)
    }

    /// Per site the reads' second moment `xᵀx / rows` of the state's forward (the pieces' `V`
    /// preconditioner, as `step_pieces` measures it).
    pub fn covariances(&self, masked: &Masked, state: &State) -> Result<Vec<Array2<f64>>, String> {
        let d = self.program.device();
        let rows = state.trace.rows as f64;
        let mut out = Vec::new();
        for site in &masked.sites {
            let mut centred: Vec<&Tensor> = Vec::new();
            for read in &site.reads {
                centred.push(state.trace.value(*read)?);
            }
            let width: usize = centred.iter().map(|c| c.cols()).sum();
            let mut covariance = Array2::<f64>::zeros((width, width));
            let (mut oi, mut blocks) = (0, Vec::new());
            for ci in &centred {
                blocks.push(oi);
                oi += ci.cols();
            }
            for (i, ci) in centred.iter().enumerate() {
                for (j, cj) in centred.iter().enumerate().skip(i) {
                    let mut block = d.zeros(ci.cols(), cj.cols()).map_err(error)?;
                    d.gemm(&mut block, 1.0 / rows, ci, Op::T, cj, Op::N, 0.0, self.proposal).map_err(error)?;
                    let values = d.download(&block).map_err(error)?;
                    covariance
                        .slice_mut(s![blocks[i]..blocks[i] + ci.cols(), blocks[j]..blocks[j] + cj.cols()])
                        .assign(&values);
                    if i != j {
                        covariance.slice_mut(s![blocks[j]..blocks[j] + cj.cols(), blocks[i]..blocks[i] + ci.cols()]).assign(&values.t());
                    }
                }
            }
            out.push(covariance);
        }
        Ok(out)
    }

    /// `masked::box_excess` of the state: per input what the box claim adds to the masks' own KL,
    /// in float64 (it decides a step's backtracking), and with `gradients` its gradients in every
    /// site's `V` and `U` in the proposal arithmetic. Refused for a site gated in blocks.
    pub fn box_excess(
        &self,
        masked: &Masked,
        state: &State,
        masks: &[Array2<f64>],
        fishers: &[Array2<f64>],
        gradients: bool,
    ) -> Result<(Array1<f64>, Option<BoxGradients>), String> {
        let d = self.program.device();
        let rows = state.trace.rows;
        if let Some(k) = (0..masked.sites.len()).find(|k| !masked.is_rank_one(*k)) {
            return Err(format!("device: the box claim of {}, gated in blocks", masked.sites[k].name));
        }
        let keep: Vec<usize> = masked.sites.iter().flat_map(|s| s.writes.iter().copied()).collect();
        let seed = d.copy(state.cotangent.as_ref().ok_or("device: prepare the state's gradient first")?).map_err(error)?;
        // The excess decides, so the gradient it reads is float64 too.
        let back = self.program.vjp(&state.trace, seed, &keep, Arithmetic::F64)?;
        let ones = |n: usize, m: usize| d.upload_vec(n, m, vec![1.0; n * m]).map_err(error);
        let ones_rows = ones(rows, 1)?;
        let mut excess = d.zeros(rows, 1).map_err(error)?;
        let mut out = Vec::new();
        for (k, site) in masked.sites.iter().enumerate() {
            let pieces = masked.pieces(k);
            let library = masked.library(k)?;
            let d_out = library.u.ncols();
            let u = d.upload(library.u.view()).map_err(error)?;
            let f = d.upload(fishers[k].view()).map_err(error)?;
            // Each piece's own weight `u_c F u_cᵀ`, as a column and as a row.
            let mut uf = d.zeros(pieces, d_out).map_err(error)?;
            d.gemm(&mut uf, 1.0, &u, Op::N, &f, Op::N, 0.0, Arithmetic::F64).map_err(error)?;
            let mut ufu = d.zeros(pieces, d_out).map_err(error)?;
            d.hadamard(&mut ufu, &uf, &u, false).map_err(error)?;
            let mut own_weight = d.zeros(pieces, 1).map_err(error)?;
            d.gemm(&mut own_weight, 1.0, &ufu, Op::N, &ones(d_out, 1)?, Op::N, 0.0, Arithmetic::F64).map_err(error)?;
            drop(ufu);
            // The off pieces' coordinates `a = z ⊙ (1 − m)` and their output `S = a U`.
            let off = d.upload(masks[k].mapv(|m| 1.0 - m).view()).map_err(error)?;
            let mut a = d.zeros(rows, pieces).map_err(error)?;
            d.hadamard(&mut a, state.trace.value(masked.z[k])?, &off, false).map_err(error)?;
            let mut s_out = d.zeros(rows, d_out).map_err(error)?;
            d.gemm(&mut s_out, 1.0, &a, Op::N, &u, Op::N, 0.0, Arithmetic::F64).map_err(error)?;
            // The KL's gradient at the written values, joined in the site's column order.
            let mut written = Vec::new();
            for w in &site.writes {
                let width = state.trace.value(*w)?.cols();
                written.push(match back.get(w) {
                    Some(g) => d.download(g).map_err(error)?,
                    None => Array2::zeros((rows, width)),
                });
            }
            let views: Vec<_> = written.iter().map(|w| w.view()).collect();
            let g = d.upload(ndarray::concatenate(Axis(1), &views).map_err(|e| e.to_string())?.view()).map_err(error)?;
            drop(written);
            let mut sf = d.zeros(rows, d_out).map_err(error)?;
            d.gemm(&mut sf, 1.0, &s_out, Op::N, &f, Op::N, 0.0, Arithmetic::F64).map_err(error)?;
            // ½ gᵀS + ⅛ SᵀF S + 1/24 Σ_off a_c² u_c F u_cᵀ, per input.
            let ones_out = ones(d_out, 1)?;
            let mut product = d.zeros(rows, d_out).map_err(error)?;
            d.hadamard(&mut product, &g, &s_out, false).map_err(error)?;
            d.gemm(&mut excess, 0.5, &product, Op::N, &ones_out, Op::N, 1.0, Arithmetic::F64).map_err(error)?;
            d.hadamard(&mut product, &sf, &s_out, false).map_err(error)?;
            d.gemm(&mut excess, 0.125, &product, Op::N, &ones_out, Op::N, 1.0, Arithmetic::F64).map_err(error)?;
            drop((product, s_out));
            let mut aa = d.zeros(rows, pieces).map_err(error)?;
            d.hadamard(&mut aa, &a, &a, false).map_err(error)?;
            d.gemm(&mut excess, 1.0 / 24.0, &aa, Op::N, &own_weight, Op::N, 1.0, Arithmetic::F64).map_err(error)?;
            if !gradients {
                continue;
            }
            // ∂/∂S = ½ g + ¼ S F; ∂/∂a = (∂/∂S) Uᵀ + a ⊙ (u_c F u_cᵀ)/12.
            let mut g_s = d.zeros(rows, d_out).map_err(error)?;
            d.axpy(&mut g_s, 0.5, &g).map_err(error)?;
            d.axpy(&mut g_s, 0.25, &sf).map_err(error)?;
            drop((g, sf));
            let mut g_a = d.zeros(rows, pieces).map_err(error)?;
            d.gemm(&mut g_a, 1.0, &g_s, Op::N, &u, Op::T, 0.0, self.proposal).map_err(error)?;
            let own_row = d.upload(d.download(&own_weight).map_err(error)?.t()).map_err(error)?;
            let mut own_a = d.zeros(rows, pieces).map_err(error)?;
            d.scale_columns(&mut own_a, &a, &own_row, false).map_err(error)?;
            d.axpy(&mut g_a, 1.0 / 12.0, &own_a).map_err(error)?;
            drop(own_a);
            // ∂/∂U = aᵀ ∂/∂S + (Σ_inputs a_c²)/12 · u_c F per piece.
            let mut u_gradient = d.zeros(pieces, d_out).map_err(error)?;
            d.gemm(&mut u_gradient, 1.0, &a, Op::T, &g_s, Op::N, 0.0, self.proposal).map_err(error)?;
            let mut weight = d.zeros(pieces, 1).map_err(error)?;
            d.gemm(&mut weight, 1.0 / 12.0, &aa, Op::T, &ones_rows, Op::N, 0.0, Arithmetic::F64).map_err(error)?;
            let mut spread = d.zeros(pieces, d_out).map_err(error)?;
            d.gemm(&mut spread, 1.0, &weight, Op::N, &ones(1, d_out)?, Op::N, 0.0, Arithmetic::F64).map_err(error)?;
            d.hadamard(&mut u_gradient, &spread, &uf, true).map_err(error)?;
            drop((spread, uf, aa, a));
            // ∂/∂V = ((∂/∂a) ⊙ (1 − m))ᵀ x, one block per read node.
            let mut g_off = d.zeros(rows, pieces).map_err(error)?;
            d.hadamard(&mut g_off, &g_a, &off, false).map_err(error)?;
            let mut v_blocks = Vec::new();
            for read in &site.reads {
                let x = state.trace.value(*read)?;
                let mut block = d.zeros(pieces, x.cols()).map_err(error)?;
                d.gemm(&mut block, 1.0, &g_off, Op::T, x, Op::N, 0.0, self.proposal).map_err(error)?;
                v_blocks.push(d.download(&block).map_err(error)?);
            }
            let views: Vec<_> = v_blocks.iter().map(|b| b.view()).collect();
            let v_gradient = ndarray::concatenate(Axis(1), &views).map_err(|e| e.to_string())?;
            out.push((v_gradient, d.download(&u_gradient).map_err(error)?));
        }
        let excess = d.download(&excess).map_err(error)?.column(0).to_owned();
        Ok((excess, gradients.then_some(out)))
    }

    /// `step_pieces`' curvature along a direction: `Σ_rows` of the output Fisher's quadratic form
    /// on the logits' tangent when each operator in `tangents` moves along its entry.
    pub fn quadratic(&self, state: &State, target: &DeviceTarget, tangents: &BTreeMap<usize, Array2<f64>>) -> Result<f64, String> {
        match self.program.jvp(&state.trace, tangents, self.proposal)? {
            Some(t) => self.program.quadratic(&state.trace, &t, target.scored.as_deref(), self.proposal),
            None => Ok(0.0),
        }
    }
}
