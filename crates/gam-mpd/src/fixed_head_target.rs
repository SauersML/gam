//! Training-only fixed, bias-free dense-head sufficient statistics.
//! Full vocabulary normalization is retained. Vendor floating-point exp/log/GEMM
//! are operational arithmetic, not acceptance certificates. Hidden is AFTER final norm.
//! Algebraically sufficient statistics do not promise bitwise full-logit reduction parity:
//! projection/reduction ordering and cancellation in logZ-mu.h+c can change rounding.
use crate::{
    device_program::DeviceProgram,
    operator_program::{FamilyInputs, Node, OperatorBody, OperatorProgram},
};
use gam_gpu::tensor::{Arithmetic, ColumnBlocks, Device, Op, Storage, Tensor};
use ndarray::Array2;
use std::sync::{Arc, Mutex};
fn error(e: impl std::fmt::Display) -> String {
    e.to_string()
}

pub(crate) struct Head {
    pub hidden: usize,
    pub operator: usize,
    pub embedding: Array2<f64>, // classes x hidden
}
impl Head {
    pub fn of(source: &OperatorProgram) -> Result<Self, String> {
        source.interfaces().map_err(error)?;
        let logits = match &source.nodes[source.output] {
            Node::Readout { input, basis }
                if matches!(
                    source.bases[*basis],
                    crate::operator_program::Basis::Indicator { .. }
                ) =>
            {
                *input
            }
            Node::Readout { .. } => return Err("fixed head requires indicator readout".into()),
            _ => source.output,
        };
        let (hidden, operator, transposed) = match &source.nodes[logits] {
            Node::Transposed { input, operator } => (*input, *operator, true),
            Node::Affine { terms, bias: None } if terms.len() == 1 => {
                (terms[0].0, terms[0].1, false)
            }
            _ => {
                return Err(
                    "compact targets require an unedited bias-free single dense head".into(),
                );
            }
        };
        let op = &source.operators[operator];
        let OperatorBody::Dense {
            values, present, ..
        } = &op.body
        else {
            return Err("fixed head must be dense".into());
        };
        if present.iter().any(|v| !*v) || values.iter().any(|v| !v.is_finite()) {
            return Err("fixed head must be all-present and finite".into());
        }
        // No executed computation may follow hidden except this fixed linear head/readout.
        if logits != hidden + 1 || source.output != logits && source.output != logits + 1 {
            return Err(
                "fixed head requires a final contiguous hidden/head/readout boundary".into(),
            );
        }
        let embedding = if transposed {
            values.t().to_owned()
        } else {
            values.clone()
        };
        Ok(Self {
            hidden,
            operator,
            embedding,
        })
    }
    pub fn same(&self, other: &Self) -> bool {
        self.embedding.dim() == other.embedding.dim()
            && self
                .embedding
                .iter()
                .zip(other.embedding.iter())
                .all(|(a, b)| a.to_bits() == b.to_bits())
    }
    pub fn prefix(&self, source: &OperatorProgram) -> OperatorProgram {
        let mut prefix = source.clone();
        prefix.nodes.truncate(self.hidden + 1);
        prefix.output = self.hidden;
        prefix
    }
}
/// Immutable labels projected through the SAME frozen head. No teacher states feed candidate forward.
#[derive(Clone)]
pub struct Target {
    pub(crate) mu: Arc<Tensor>,
    pub(crate) entropy: Vec<f64>,
    pub(crate) head: Arc<Head>,
    pub(crate) scored: Option<Vec<bool>>,
}
impl Target {
    /// Share exactly identical immutable head storage across independently generated labels.
    /// The projected labels and entropy remain unchanged; incompatible heads are refused.
    pub fn with_shared_head(&self, reference: &Self) -> Result<Self, String> {
        if !self.head.same(&reference.head) {
            return Err("compact targets have incompatible immutable heads".into());
        }
        let mut target = self.clone();
        target.head = reference.head.clone();
        Ok(target)
    }
    pub fn rows(&self) -> usize {
        self.mu.rows()
    }
    pub fn width(&self) -> usize {
        self.mu.cols()
    }
    pub fn numeric_bytes(&self) -> usize {
        self.rows().saturating_mul(self.width()).saturating_mul(8)
    }
}
/// Reusable native teacher prefix and fixed head, with bounded row-tiled vocabulary scratch.
pub struct Teacher {
    device: Device,
    prefix: DeviceProgram,
    head: Arc<Head>,
    embedding: Tensor,
    tile_rows: usize,
    numeric_bytes: usize,
    retained_target_bytes: Mutex<usize>,
    attention: Vec<(usize, bool)>,
    law_code_bytes: usize,
}
impl Teacher {
    pub fn new(
        device: &Device,
        source: &OperatorProgram,
        tile_rows: usize,
        numeric_bytes: usize,
    ) -> Result<Self, String> {
        if tile_rows == 0 || numeric_bytes == 0 {
            return Err("positive head tile and numeric budget required".into());
        }
        let (source, _) = crate::artifact_device::mapped_inlined(source)?;
        let head = Arc::new(Head::of(&source)?);
        let prefix_source = head.prefix(&source);
        let mut attention = Vec::new();
        let mut law_code_bytes = 0usize;
        for node in &prefix_source.nodes {
            match node {
                Node::Attend { query, rotary, .. } => attention.push((
                    prefix_source.node_interface(*query).map_err(error)?.width(),
                    rotary.is_some(),
                )),
                Node::Pointwise { input, .. } => {
                    law_code_bytes = law_code_bytes
                        .checked_add(
                            prefix_source
                                .node_interface(*input)
                                .map_err(error)?
                                .width()
                                .checked_mul(4)
                                .ok_or("law code size overflow")?,
                        )
                        .ok_or("law code bytes overflow")?
                }
                _ => continue,
            }
        }
        let prefix = DeviceProgram::compile_values_bounded(device, &prefix_source, numeric_bytes)?;
        if head
            .embedding
            .len()
            .checked_mul(8)
            .is_none_or(|n| n > numeric_bytes)
        {
            return Err("fixed head exceeds teacher numeric budget".into());
        }
        let embedding = device.upload(head.embedding.view()).map_err(error)?;
        Ok(Self {
            device: device.clone(),
            prefix,
            head,
            embedding,
            tile_rows,
            numeric_bytes,
            retained_target_bytes: Mutex::new(0),
            attention,
            law_code_bytes,
        })
    }
    /// Budget conservatively charges every target returned by this Teacher, even if dropped.
    /// Calls serialize to preserve the shared numeric lease. Host metadata and CUDA context,
    /// allocator/library workspace remain outside this explicit numeric-buffer plan.
    pub fn target(&self, inputs: &FamilyInputs, scored: Option<&[bool]>) -> Result<Target, String> {
        if inputs.rows == 0
            || scored.is_some_and(|x| x.len() != inputs.rows || !x.iter().any(|v| *v))
        {
            return Err("invalid compact teacher row domain".into());
        }
        let rows = inputs.rows;
        let width = self.head.embedding.ncols();
        let classes = self.head.embedding.nrows();
        let planned = self
            .prefix
            .operator_numeric_bytes()?
            .checked_add(
                self.head
                    .embedding
                    .len()
                    .checked_mul(8)
                    .ok_or("head bytes overflow")?,
            )
            .and_then(|v| {
                v.checked_add(
                    self.prefix
                        .bytes_per_row()
                        .checked_mul(rows)?
                        .checked_mul(4)?,
                )
            })
            .and_then(|v| v.checked_add(rows.checked_mul(width)?.checked_mul(8)?))
            .and_then(|v| {
                v.checked_add(
                    self.tile_rows
                        .min(rows)
                        .checked_mul(classes)?
                        .checked_mul(16)?,
                )
            })
            .ok_or("compact teacher plan overflow")?;
        let mut attention_peak = 0usize;
        let mut indices = self.law_code_bytes;
        for (query_width, rotary) in &self.attention {
            let scratch = rows
                .checked_mul(rows)
                .and_then(|n| n.checked_mul(8 * 12))
                .and_then(|n| n.checked_add(rows.checked_mul(*query_width)?.checked_mul(8 * 16)?))
                .ok_or("attention workspace overflow")?;
            attention_peak = attention_peak.max(scratch);
            if *rotary {
                indices = indices
                    .checked_add(
                        rows.checked_mul(*query_width)
                            .and_then(|n| n.checked_mul(8 * 2))
                            .ok_or("rotation indices overflow")?,
                    )
                    .ok_or("indices bytes overflow")?;
            }
        }
        for slot in &inputs.slots {
            if let crate::operator_program::SlotValues::Tokens(tokens) = slot {
                indices = indices
                    .checked_add(
                        tokens
                            .len()
                            .checked_mul(4)
                            .ok_or("token indices overflow")?,
                    )
                    .ok_or("indices bytes overflow")?;
            }
        }
        if scored.is_some() {
            indices = indices
                .checked_add(rows.checked_mul(4).ok_or("scored flags overflow")?)
                .ok_or("indices bytes overflow")?;
        }
        let planned = planned
            .checked_add(attention_peak)
            .and_then(|v| v.checked_add(indices))
            .ok_or("teacher attention/indices plan overflow")?;
        let mut retained = self
            .retained_target_bytes
            .lock()
            .map_err(|e| e.to_string())?;
        let planned = planned
            .checked_add(*retained)
            .ok_or("retained teacher target bytes overflow")?;
        if planned > self.numeric_bytes {
            return Err(format!(
                "compact teacher numeric plan {planned} exceeds {}",
                self.numeric_bytes
            ));
        }
        let trace = self.prefix.forward(inputs)?;
        let mut mu = self.device.zeros(rows, width).map_err(error)?;
        let mut entropy = Vec::with_capacity(rows);
        for start in (0..rows).step_by(self.tile_rows) {
            let n = self.tile_rows.min(rows - start);
            let hidden = self
                .device
                .rows_of(trace.value(self.prefix.hidden())?, start, n)
                .map_err(error)?;
            let mut probabilities = self.device.zeros(n, classes).map_err(error)?;
            self.device
                .gemm(
                    &mut probabilities,
                    1.,
                    &hidden,
                    Op::N,
                    &self.embedding,
                    Op::T,
                    0.,
                    Arithmetic::F64,
                )
                .map_err(error)?;
            let flags = scored
                .map(|s| {
                    self.device.upload_indices(
                        &s[start..start + n]
                            .iter()
                            .map(|v| u32::from(*v))
                            .collect::<Vec<_>>(),
                    )
                })
                .transpose()
                .map_err(error)?;
            let stats = self
                .device
                .softmax_stats_rows(&mut probabilities, flags.as_ref())
                .map_err(error)?;
            let mut projected = self.device.zeros(n, width).map_err(error)?;
            self.device
                .gemm(
                    &mut projected,
                    1.,
                    &probabilities,
                    Op::N,
                    &self.embedding,
                    Op::N,
                    0.,
                    Arithmetic::F64,
                )
                .map_err(error)?;
            self.device
                .set_rows(&mut mu, start, &projected)
                .map_err(error)?;
            entropy.extend(stats.into_iter().map(|s| s[1]));
        }
        *retained = retained
            .checked_add(
                rows.checked_mul(width)
                    .and_then(|n| n.checked_mul(8))
                    .ok_or("target bytes overflow")?,
            )
            .ok_or("retained target bytes overflow")?;
        Ok(Target {
            mu: Arc::new(mu),
            entropy,
            head: self.head.clone(),
            scored: scored.map(<[bool]>::to_vec),
        })
    }
}

pub(crate) struct ResidentHead {
    pub embedding: Tensor,
    pub ones: Tensor,
    pub tile_rows: usize,
    /// The hidden width as one column block (row dots).
    width: ColumnBlocks,
}
impl ResidentHead {
    pub fn new(d: &Device, head: &Head, tile_rows: usize) -> Result<Self, String> {
        Ok(Self {
            embedding: d.upload(head.embedding.view()).map_err(error)?,
            ones: d
                .upload(Array2::ones((head.embedding.ncols(), 1)).view())
                .map_err(error)?,
            tile_rows,
            width: d
                .column_blocks(&[head.embedding.ncols()])
                .map_err(error)?,
        })
    }
    /// Per-row compact KL; with `gradient`, the hidden seed. Logit and seed products run in
    /// `arithmetic`.
    pub fn score(
        &self,
        d: &Device,
        hidden: &Tensor,
        target: &Target,
        gradient: bool,
        arithmetic: Arithmetic,
    ) -> Result<(Vec<f64>, Option<Tensor>), String> {
        if hidden.storage() == Storage::F32 {
            return self.swept(d, hidden, target, gradient, arithmetic);
        }
        let mut losses = Vec::with_capacity(hidden.rows());
        let mut seed = if gradient {
            Some(d.zeros(hidden.rows(), hidden.cols()).map_err(error)?)
        } else {
            None
        };
        for start in (0..hidden.rows()).step_by(self.tile_rows) {
            let n = self.tile_rows.min(hidden.rows() - start);
            let h = d.rows_of(hidden, start, n).map_err(error)?;
            let mu = d.rows_of(&target.mu, start, n).map_err(error)?;
            let mut q = d.zeros(n, self.embedding.rows()).map_err(error)?;
            d.gemm(
                &mut q,
                1.,
                &h,
                Op::N,
                &self.embedding,
                Op::T,
                0.,
                arithmetic,
            )
            .map_err(error)?;
            let flags = target
                .scored
                .as_ref()
                .map(|s| {
                    d.upload_indices(
                        &s[start..start + n]
                            .iter()
                            .map(|v| u32::from(*v))
                            .collect::<Vec<_>>(),
                    )
                })
                .transpose()
                .map_err(error)?;
            let stats = d
                .softmax_stats_rows(&mut q, flags.as_ref())
                .map_err(error)?;
            let mut products = d.zeros(n, h.cols()).map_err(error)?;
            d.hadamard(&mut products, &h, &mu, false).map_err(error)?;
            // mu.h stays F64: logZ - mu.h + c cancels, so it must not add product rounding.
            let mut dots = d.zeros(n, 1).map_err(error)?;
            d.gemm(
                &mut dots,
                1.,
                &products,
                Op::N,
                &self.ones,
                Op::N,
                0.,
                Arithmetic::F64,
            )
            .map_err(error)?;
            let dots = d.download(&dots).map_err(error)?;
            for r in 0..n {
                let loss = if target.scored.as_ref().is_some_and(|s| !s[start + r]) {
                    0.
                } else {
                    stats[r][0] - dots[(r, 0)] + target.entropy[start + r]
                };
                if !loss.is_finite() {
                    return Err("nonfinite compact head KL".into());
                }
                losses.push(loss);
            }
            if let Some(seed) = &mut seed {
                let mut projected = d.zeros(n, h.cols()).map_err(error)?;
                d.gemm(
                    &mut projected,
                    1.,
                    &q,
                    Op::N,
                    &self.embedding,
                    Op::N,
                    0.,
                    arithmetic,
                )
                .map_err(error)?;
                d.axpy(&mut projected, -1., &mu).map_err(error)?;
                // Target creation masks BOTH p and q, so unscored projected seeds are zero.
                d.set_rows(seed, start, &projected).map_err(error)?;
            }
        }
        Ok((losses, seed))
    }
    /// [`Self::score`] in f32 storage: the vocabulary is swept without forming the logits
    /// ([`Device::head_log_partition`]), which also returns the seed's `E^T q`.
    fn swept(
        &self,
        d: &Device,
        hidden: &Tensor,
        target: &Target,
        gradient: bool,
        arithmetic: Arithmetic,
    ) -> Result<(Vec<f64>, Option<Tensor>), String> {
        let flags = target
            .scored
            .as_ref()
            .map(|s| d.upload_indices(&s.iter().map(|v| u32::from(*v)).collect::<Vec<_>>()))
            .transpose()
            .map_err(error)?;
        let mut seed = if gradient {
            Some(d.zeros(hidden.rows(), hidden.cols()).map_err(error)?)
        } else {
            None
        };
        let partitions = d
            .head_log_partition(
                hidden,
                &self.embedding,
                false,
                flags.as_ref(),
                seed.as_mut(),
                arithmetic,
            )
            .map_err(error)?;
        let dots = d
            .download(
                &d.block_products(hidden, &target.mu, &self.width)
                    .map_err(error)?,
            )
            .map_err(error)?;
        let losses = (0..hidden.rows())
            .map(|r| {
                let loss = if target.scored.as_ref().is_some_and(|s| !s[r]) {
                    0.
                } else {
                    partitions[r] - dots[(r, 0)] + target.entropy[r]
                };
                if loss.is_finite() {
                    Ok(loss)
                } else {
                    Err("nonfinite compact head KL".to_string())
                }
            })
            .collect::<Result<Vec<_>, String>>()?;
        if let Some(seed) = &mut seed {
            // Unscored rows of both E^T q and mu are zero.
            d.axpy(seed, -1., &target.mu).map_err(error)?;
        }
        Ok((losses, seed))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn target_head_sharing_preserves_labels_and_rejects_incompatible_head() {
        let device = Device::host();
        let first = Target {
            mu: Arc::new(device.zeros(2, 2).expect("mu")),
            entropy: vec![0., 0.],
            head: Arc::new(Head {
                hidden: 0,
                operator: 0,
                embedding: Array2::eye(2),
            }),
            scored: None,
        };
        let independent = Target {
            head: Arc::new(Head {
                hidden: 4,
                operator: 3,
                embedding: Array2::eye(2),
            }),
            ..first.clone()
        };
        assert!(!Arc::ptr_eq(&first.head, &independent.head));
        let shared = independent
            .with_shared_head(&first)
            .expect("identical numeric head");
        assert!(Arc::ptr_eq(&first.head, &shared.head));
        assert!(Arc::ptr_eq(&independent.mu, &shared.mu));
        assert_eq!(shared.entropy, independent.entropy);
        let incompatible = Target {
            head: Arc::new(Head {
                hidden: 0,
                operator: 0,
                embedding: Array2::zeros((2, 2)),
            }),
            ..first.clone()
        };
        assert!(incompatible.with_shared_head(&first).is_err());
    }
}
