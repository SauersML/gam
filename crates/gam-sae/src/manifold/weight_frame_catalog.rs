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

