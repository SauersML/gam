//! Weight-sourced frame catalog for in-frame curved charts.
//!
//! The catalog is a mechanism-support object: each frame is the column image of a
//! model component matrix, so it says where an activation can live before any
//! corpus rows are inspected. Data enters later only through occupancy and
//! in-frame coordinates.

use ndarray::{Array1, Array2, ArrayView2};

use crate::frames::{GrassmannFrame, SAE_FRAME_RANK_CUTOFF};
use gam_linalg::faer_ndarray::{FaerSvd, fast_ab};

/// Model component whose output image supplies a chart frame.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum WeightFrameSource {
    /// Per-head attention OV map, with matrix image `range(W_O W_V)`.
    AttentionHeadOv { layer: usize, head: usize },
    /// MLP down projection, with matrix image `range(W_down)`.
    MlpDownProjection { layer: usize },
}

/// Rank-selection policy for weight-sourced component images.
#[derive(Clone, Debug)]
pub struct WeightFrameCatalogConfig {
    /// Lower clamp on the component image rank.
    pub frame_rank_min: usize,
    /// Upper clamp on the component image rank.
    pub frame_rank_max: usize,
    /// Relative singular-value cutoff selecting the numerical column rank.
    pub rank_cutoff: f64,
}

impl Default for WeightFrameCatalogConfig {
    fn default() -> Self {
        Self {
            frame_rank_min: 1,
            frame_rank_max: 32,
            rank_cutoff: SAE_FRAME_RANK_CUTOFF,
        }
    }
}

/// Owned component matrix whose columns are residual-stream output vectors.
#[derive(Clone, Debug)]
pub struct WeightFrameMatrix {
    pub source: WeightFrameSource,
    pub matrix: Array2<f64>,
}

impl WeightFrameMatrix {
    /// Build the per-head OV component matrix `W_O W_V`.
    ///
    /// `w_o` is shaped `(d_model, d_head)` and `w_v` is shaped
    /// `(d_head, d_model)`; the resulting square map has column image in the
    /// residual stream.
    pub fn attention_head_ov(
        layer: usize,
        head: usize,
        w_o: ArrayView2<'_, f64>,
        w_v: ArrayView2<'_, f64>,
    ) -> Result<Self, String> {
        if w_o.ncols() != w_v.nrows() {
            return Err(format!(
                "WeightFrameMatrix::attention_head_ov: W_O cols {} must equal W_V rows {}",
                w_o.ncols(),
                w_v.nrows()
            ));
        }
        Ok(Self {
            source: WeightFrameSource::AttentionHeadOv { layer, head },
            matrix: fast_ab(&w_o.to_owned(), &w_v.to_owned()),
        })
    }

    /// Build an MLP down-projection component matrix `W_down`.
    ///
    /// `w_down` is shaped `(d_model, d_mlp)` so its columns are residual-stream
    /// output vectors.
    pub fn mlp_down_projection(layer: usize, w_down: ArrayView2<'_, f64>) -> Self {
        Self {
            source: WeightFrameSource::MlpDownProjection { layer },
            matrix: w_down.to_owned(),
        }
    }
}

/// One frame in the mechanism-support catalog.
#[derive(Clone, Debug)]
pub struct WeightFrameCatalogEntry {
    pub source: WeightFrameSource,
    pub frame: GrassmannFrame,
    pub singular_values: Array1<f64>,
    pub matrix_rows: usize,
    pub matrix_cols: usize,
}

/// Source-tagged collection of component image frames sharing one ambient width.
#[derive(Clone, Debug)]
pub struct WeightFrameCatalog {
    output_dim: usize,
    entries: Vec<WeightFrameCatalogEntry>,
}

impl WeightFrameCatalog {
    pub fn new(entries: Vec<WeightFrameCatalogEntry>) -> Result<Self, String> {
        let Some(first) = entries.first() else {
            return Err("WeightFrameCatalog::new: catalog must contain at least one frame".into());
        };
        let output_dim = first.frame.output_dim();
        for (idx, entry) in entries.iter().enumerate() {
            if entry.frame.output_dim() != output_dim {
                return Err(format!(
                    "WeightFrameCatalog::new: entry {idx} output dim {} != catalog dim {output_dim}",
                    entry.frame.output_dim()
                ));
            }
        }
        Ok(Self {
            output_dim,
            entries,
        })
    }

    pub fn output_dim(&self) -> usize {
        self.output_dim
    }

    pub fn entries(&self) -> &[WeightFrameCatalogEntry] {
        &self.entries
    }

    pub fn entry(&self, index: usize) -> Option<&WeightFrameCatalogEntry> {
        self.entries.get(index)
    }
}

/// SVD every supplied component matrix into a source-tagged ambient frame.
pub fn frame_catalog_from_weight_matrices(
    components: &[WeightFrameMatrix],
    config: &WeightFrameCatalogConfig,
) -> Result<WeightFrameCatalog, String> {
    if config.frame_rank_min == 0 || config.frame_rank_max < config.frame_rank_min {
        return Err(
            "frame_catalog_from_weight_matrices: require 1 <= frame_rank_min <= frame_rank_max"
                .to_string(),
        );
    }
    let mut entries = Vec::with_capacity(components.len());
    for component in components {
        entries.push(component_image_frame(component, config)?);
    }
    WeightFrameCatalog::new(entries)
}

fn component_image_frame(
    component: &WeightFrameMatrix,
    config: &WeightFrameCatalogConfig,
) -> Result<WeightFrameCatalogEntry, String> {
    let (p, cols) = component.matrix.dim();
    if p == 0 || cols == 0 {
        return Err("component_image_frame: component matrix must be non-empty".to_string());
    }
    let (u_opt, sv, _vt) = component
        .matrix
        .svd(true, false)
        .map_err(|e| format!("component_image_frame: SVD failed: {e}"))?;
    let u = u_opt.ok_or_else(|| "component_image_frame: SVD returned no left factor".to_string())?;
    let max_sv = sv.iter().copied().fold(0.0_f64, f64::max);
    if !(max_sv > 0.0) {
        return Err("component_image_frame: zero component image is not chartable".to_string());
    }
    let tol = config.rank_cutoff * max_sv;
    let numerical_rank = sv.iter().filter(|&&v| v > tol).count();
    let available = u.ncols().min(p);
    let rank = numerical_rank
        .max(config.frame_rank_min)
        .min(config.frame_rank_max)
        .min(available);
    if rank == 0 {
        return Err("component_image_frame: selected rank is zero".to_string());
    }
    let mut frame = Array2::<f64>::zeros((p, rank));
    for col in 0..rank {
        for row in 0..p {
            frame[[row, col]] = u[[row, col]];
        }
    }
    let mut gauge = Array1::<f64>::zeros(rank);
    for i in 0..rank {
        gauge[i] = sv[i];
    }
    Ok(WeightFrameCatalogEntry {
        source: component.source.clone(),
        frame: GrassmannFrame::from_oriented(frame, gauge),
        singular_values: sv,
        matrix_rows: p,
        matrix_cols: cols,
    })
}
