//! Opt-in head-only f64 CUDA logits; normalization and acceptance metrics remain CPU.
use crate::counterfactual::Decoder;
use gam_gpu::tensor::{Arithmetic, Device, Op, Tensor};
use ndarray::{Array2, Axis};

/// Numeric-buffer budgets. Excludes allocator metadata, CUDA library workspaces and context.
#[derive(Clone, Copy, Debug)]
pub struct Budget {
    pub resident_bytes: usize,
    pub workspace_bytes: usize,
}

/// Immutable native head only: no blocks or candidate parameters are uploaded.
pub struct Resident {
    device: Device,
    embedding: Tensor,
    gain: Tensor,
    eps: f64,
    width: usize,
    vocab: usize,
    tile_rows: usize,
    resident_bytes: usize,
}

fn bytes(elements: usize) -> Result<usize, String> {
    elements
        .checked_mul(8)
        .ok_or_else(|| "readout byte count overflow".into())
}

/// Conservative numeric buffers: four logits tiles (including caller's paired
/// returned tile), six residual-width tiles, and one reduction scalar per row.
/// Both host and device buffers count. Library-private GEMM workspace is excluded.
pub fn tile_bytes(width: usize, vocab: usize) -> Result<usize, String> {
    let n = vocab
        .checked_mul(4)
        .and_then(|n| width.checked_mul(6).and_then(|d| n.checked_add(d)))
        .and_then(|n| n.checked_add(1))
        .ok_or("readout tile count overflow")?;
    bytes(n)
}

pub fn normalize_logits(mut logits: Array2<f64>) -> Result<Array2<f64>, String> {
    if logits.ncols() == 0 || logits.iter().any(|x| !x.is_finite()) {
        return Err("readout logits must be finite and nonempty".into());
    }
    for mut row in logits.outer_iter_mut() {
        let max = row.fold(f64::NEG_INFINITY, |m, v| m.max(*v));
        let total = row.iter().map(|v| (v - max).exp()).sum::<f64>().ln() + max;
        if !total.is_finite() {
            return Err("readout log normalization is nonfinite".into());
        }
        row.mapv_inplace(|v| v - total);
    }
    if logits.iter().any(|x| !x.is_finite()) {
        return Err("normalized readout is nonfinite".into());
    }
    Ok(logits)
}

impl Resident {
    pub fn new(device: Device, decoder: &Decoder, budget: Budget) -> Result<Self, String> {
        if !cfg!(target_os = "linux") || device.is_host() || !device.float64() {
            return Err("native readout requires CUDA f64, without fallback".into());
        }
        Self::from_head(
            device,
            decoder.embedding(),
            decoder.final_gain(),
            decoder.eps(),
            budget,
        )
    }
    fn from_head(
        device: Device,
        embedding: &Array2<f64>,
        gain: &ndarray::Array1<f64>,
        eps: f64,
        budget: Budget,
    ) -> Result<Self, String> {
        let (vocab, width) = embedding.dim();
        if width == 0
            || vocab == 0
            || gain.len() != width
            || !eps.is_finite()
            || eps <= 0.0
            || embedding.iter().chain(gain.iter()).any(|x| !x.is_finite())
        {
            return Err("invalid native head shape or finite parameters".into());
        }
        let resident_bytes = bytes(
            embedding
                .len()
                .checked_add(width)
                .ok_or("readout resident count overflow")?,
        )?;
        if resident_bytes > budget.resident_bytes {
            return Err(format!(
                "native readout resident numeric bytes {resident_bytes} exceed budget {}",
                budget.resident_bytes
            ));
        }
        let tile_rows = budget.workspace_bytes / tile_bytes(width, vocab)?;
        if tile_rows == 0 {
            return Err("native readout workspace cannot hold one paired tile row".into());
        }
        let embedding = device.upload(embedding.view()).map_err(|e| e.to_string())?;
        let gain = device
            .upload(gain.view().insert_axis(Axis(0)))
            .map_err(|e| e.to_string())?;
        Ok(Self {
            device,
            embedding,
            gain,
            eps,
            width,
            vocab,
            tile_rows,
            resident_bytes,
        })
    }
    pub fn tile_rows(&self) -> usize {
        self.tile_rows
    }
    pub fn resident_bytes(&self) -> usize {
        self.resident_bytes
    }
    pub fn log_probs(&self, residual: &Array2<f64>) -> Result<Array2<f64>, String> {
        if residual.ncols() != self.width
            || residual.nrows() > self.tile_rows
            || residual.iter().any(|x| !x.is_finite())
        {
            return Err(
                "native readout residual shape, finite values or tile budget violated".into(),
            );
        }
        let d = &self.device;
        let x = d.upload(residual.view()).map_err(|e| e.to_string())?;
        let norm = d.rms_norm(&x, self.eps).map_err(|e| e.to_string())?;
        let mut scaled = d.zeros(x.rows(), self.width).map_err(|e| e.to_string())?;
        d.scale_columns(&mut scaled, &norm, &self.gain, false)
            .map_err(|e| e.to_string())?;
        drop(x);
        drop(norm);
        let mut out = d
            .zeros(residual.nrows(), self.vocab)
            .map_err(|e| e.to_string())?;
        d.gemm(
            &mut out,
            1.0,
            &scaled,
            Op::N,
            &self.embedding,
            Op::T,
            0.0,
            Arithmetic::F64,
        )
        .map_err(|e| e.to_string())?;
        let logits = d.download(&out).map_err(|e| e.to_string())?;
        normalize_logits(logits)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn normalization_extremes_and_finite_errors() {
        let p = normalize_logits(ndarray::array![
            [1000., 999., -1000.],
            [-1000., -1001., -3000.]
        ])
        .unwrap();
        for i in 0..3 {
            assert!((p[[0, i]] - p[[1, i]]).abs() < 1e-12);
        }
        assert!(normalize_logits(ndarray::array![[f64::INFINITY]]).is_err());
        assert!(normalize_logits(ndarray::array![[f64::NAN]]).is_err());
        assert!(tile_bytes(usize::MAX, 1).is_err());
    }
    #[test]
    fn head_only_host_structure_and_budget() {
        let e = ndarray::array![[1., 2.], [3., 4.], [5., 6.]];
        let g = ndarray::array![1., 1.];
        let b = Budget {
            resident_bytes: 64,
            workspace_bytes: tile_bytes(2, 3).unwrap() * 2,
        };
        let h = Resident::from_head(Device::host(), &e, &g, 1e-6, b).unwrap();
        assert_eq!(h.resident_bytes(), 64);
        assert_eq!(h.tile_rows(), 2);
        assert!(h.log_probs(&Array2::zeros((3, 2))).is_err());
        assert!(
            Resident::from_head(
                Device::host(),
                &e,
                &g,
                1e-6,
                Budget {
                    resident_bytes: 63,
                    ..b
                }
            )
            .is_err()
        );
        let p = h.log_probs(&ndarray::array![[1., -1.]]).unwrap();
        assert!(p.iter().all(|x| x.is_finite()));
    }
}
