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
use super::masked::{Masked, Target};
use super::operator_program::{FamilyInputs, Node};
use gam_gpu::tensor::{Arithmetic, Device, Op, Tensor};
use ndarray::{Array1, Array2, Axis, s};
use std::collections::BTreeMap;

fn error(e: impl std::fmt::Display) -> String {
    format!("device: {e}")
}

/// A masked program lowered onto a device (module note).
pub struct Accelerated {
    program: DeviceProgram,
    proposal: Arithmetic,
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
    cotangent: Tensor,
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
        Ok(Self { program: DeviceProgram::compile(device, &masked.program)?, proposal })
    }

    /// The device program.
    #[must_use]
    pub fn program(&self) -> &DeviceProgram {
        &self.program
    }

    /// Re-upload the operators `masked` now holds new copies of (after a step).
    pub fn refresh(&mut self, masked: &Masked) -> Result<(), String> {
        self.program.refresh(&masked.program)
    }

    /// `target` on the device.
    pub fn target(&self, target: &Target) -> Result<DeviceTarget, String> {
        Ok(DeviceTarget { logits: self.program.device().upload(target.logits.view()).map_err(error)?, scored: target.scored.clone() })
    }

    /// The masked forward on `family` (its masks in their slots, as `Masked::family` sets them).
    pub fn forward(&self, family: &FamilyInputs, target: &DeviceTarget) -> Result<State, String> {
        let trace = self.program.forward(family)?;
        let (kl, cotangent) = self.program.kl(&trace, &target.logits, target.scored.as_deref())?;
        Ok(State { kl, trace, cotangent })
    }

    fn cotangents(&self, trace: &DeviceTrace, seed: Tensor, keep: &[usize]) -> Result<BTreeMap<usize, Tensor>, String> {
        self.program.vjp(trace, seed, keep, self.proposal)
    }

    /// Per site `∂KL/∂m` (rows × C): `masked::mask_gradients` of the state's KL.
    pub fn mask_gradients(&self, masked: &Masked, state: &State) -> Result<Vec<Array2<f64>>, String> {
        let d = self.program.device();
        let seed = d.copy(&state.cotangent).map_err(error)?;
        let back = self.cotangents(&state.trace, seed, &masked.masked)?;
        let mut out = Vec::new();
        for k in 0..masked.sites.len() {
            let z = state.trace.value(masked.z[k])?;
            out.push(match back.get(&masked.masked[k]) {
                Some(c) => {
                    let mut g = d.zeros(z.rows(), z.cols()).map_err(error)?;
                    d.hadamard(&mut g, c, z, false).map_err(error)?;
                    d.download(&g).map_err(error)?
                }
                None => Array2::zeros(z.dim()),
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
        let mut keep = masked.masked.clone();
        if written {
            keep.extend(masked.sites.iter().flat_map(|s| s.writes.iter().copied()));
        }
        let mut h: Vec<Tensor> = Vec::new();
        for k in 0..masked.sites.len() {
            let z = state.trace.value(masked.z[k])?;
            h.push(d.zeros(z.rows(), z.cols()).map_err(error)?);
        }
        // Per site, the blocks `g_iᵀ g_j` of the written nodes' Fisher.
        let mut blocks: Vec<Vec<Vec<Option<Tensor>>>> =
            masked.sites.iter().map(|s| (0..s.writes.len()).map(|_| (0..s.writes.len()).map(|_| None).collect()).collect()).collect();
        let mut rng = Uniforms(seed | 1);
        let scored = |r: usize| target.scored.as_ref().is_none_or(|s| s[r]);
        for _ in 0..samples {
            let uniforms: Vec<f64> = (0..rows).map(|r| if scored(r) { rng.next() } else { 0.0 }).collect();
            let g_hidden = self.program.sampled(&state.trace, &uniforms, target.scored.as_deref(), self.proposal)?;
            let back = self.cotangents(&state.trace, g_hidden, &keep)?;
            for (k, hk) in h.iter_mut().enumerate() {
                if let Some(c) = back.get(&masked.masked[k]) {
                    let z = state.trace.value(masked.z[k])?;
                    let mut g = d.zeros(z.rows(), z.cols()).map_err(error)?;
                    d.hadamard(&mut g, c, z, false).map_err(error)?;
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
        let seed = d.copy(&state.cotangent).map_err(error)?;
        let back = self.cotangents(&state.trace, seed, &keep)?;
        let mut out = Vec::new();
        for (k, site) in masked.sites.iter().enumerate() {
            let (pieces, rows) = (masked.pieces(k), state.trace.rows);
            let z = state.trace.value(masked.z[k])?;
            let Node::Hadamard { right: mask_node, .. } = &masked.program.nodes[masked.masked[k]] else {
                return Err("device: a site's masked node is not its mask's product".to_string());
            };
            let mask = state.trace.value(*mask_node)?;
            if mask.dim() != masks[k].dim() {
                return Err("device: the state's masks are not the given ones".to_string());
            }
            let Some(cot_masked) = back.get(&masked.masked[k]) else {
                let (d_in, d_out) = (masked.mean(k).len(), site.writes.iter().map(|w| state.trace.value(*w).map(|t| t.cols())).sum::<Result<usize, _>>()?);
                out.push((Array2::zeros((rows, pieces)), Array2::zeros((pieces, d_in)), Array2::zeros((pieces, d_out))));
                continue;
            };
            let mut mask_gradient = d.zeros(rows, pieces).map_err(error)?;
            d.hadamard(&mut mask_gradient, cot_masked, z, false).map_err(error)?;
            let mut cot_z = d.zeros(rows, pieces).map_err(error)?;
            d.hadamard(&mut cot_z, cot_masked, mask, false).map_err(error)?;
            // ∂KL/∂V = cot_zᵀ (x − μ), one block per read node.
            let mut v_blocks = Vec::new();
            let mut offset = 0;
            for read in &site.reads {
                let x = state.trace.value(*read)?;
                let mean = masked.mean(k).slice(s![offset..offset + x.cols()]).to_owned();
                offset += x.cols();
                let mut centred = d.copy(x).map_err(error)?;
                d.add_row(&mut centred, -1.0, &d.upload_vec(1, mean.len(), mean.to_vec()).map_err(error)?).map_err(error)?;
                let mut block = d.zeros(pieces, x.cols()).map_err(error)?;
                d.gemm(&mut block, 1.0, &cot_z, Op::T, &centred, Op::N, 0.0, self.proposal).map_err(error)?;
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

    /// Per site the read covariance `(x − μ)ᵀ(x − μ) / rows` of the state's forward (the pieces'
    /// `V` preconditioner, as `step_pieces` measures it).
    pub fn covariances(&self, masked: &Masked, state: &State) -> Result<Vec<Array2<f64>>, String> {
        let d = self.program.device();
        let rows = state.trace.rows as f64;
        let mut out = Vec::new();
        for (k, site) in masked.sites.iter().enumerate() {
            let library_mean = masked.mean(k);
            let mut centred: Vec<Tensor> = Vec::new();
            let mut offset = 0;
            for read in &site.reads {
                let x = state.trace.value(*read)?;
                let mean = library_mean.slice(s![offset..offset + x.cols()]).to_vec();
                offset += x.cols();
                let mut c = d.copy(x).map_err(error)?;
                d.add_row(&mut c, -1.0, &d.upload_vec(1, mean.len(), mean).map_err(error)?).map_err(error)?;
                centred.push(c);
            }
            let width = library_mean.len();
            let mut covariance = Array2::<f64>::zeros((width, width));
            let (mut oi, mut blocks) = (0, Vec::new());
            for ci in &centred {
                blocks.push(oi);
                oi += ci.cols();
            }
            for (i, ci) in centred.iter().enumerate() {
                for (j, cj) in centred.iter().enumerate() {
                    let mut block = d.zeros(ci.cols(), cj.cols()).map_err(error)?;
                    d.gemm(&mut block, 1.0 / rows, ci, Op::T, cj, Op::N, 0.0, self.proposal).map_err(error)?;
                    covariance
                        .slice_mut(s![blocks[i]..blocks[i] + ci.cols(), blocks[j]..blocks[j] + cj.cols()])
                        .assign(&d.download(&block).map_err(error)?);
                }
            }
            out.push(covariance);
        }
        Ok(out)
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
