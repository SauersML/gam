#[derive(Clone, Debug)]
pub struct ResourcePolicy {
    pub max_single_materialization_bytes: usize,
    pub max_operator_cache_bytes: usize,
    pub max_spatial_distance_cache_bytes: usize,
    pub max_owned_data_cache_bytes: usize,
    pub row_chunk_target_bytes: usize,
    pub derivative_storage_mode: DerivativeStorageMode,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DerivativeStorageMode {
    /// Production exact-math: operator-backed, no dense fallback.
    AnalyticOperatorRequired,
    /// Allow dense materialization if under the single-materialization budget.
    MaterializeIfSmall,
    /// Dense materialization only permitted for diagnostic code paths.
    DiagnosticsOnly,
}

#[derive(Clone, Debug)]
pub struct MaterializationPolicy {
    pub max_single_dense_bytes: usize,
    pub max_cached_dense_bytes: usize,
    pub row_chunk_target_bytes: usize,
    pub allow_operator_materialization: bool,
    pub allow_diagnostic_materialization: bool,
}

#[derive(Debug, thiserror::Error)]
pub enum MatrixMaterializationError {
    #[error(
        "{context}: dense materialization of {nrows}x{ncols} requires {bytes} bytes (limit {limit_bytes})"
    )]
    TooLarge {
        context: &'static str,
        nrows: usize,
        ncols: usize,
        bytes: usize,
        limit_bytes: usize,
    },

    #[error("{context}: operator does not implement chunked row access")]
    MissingRowChunk { context: &'static str },

    #[error("{context}: materialization forbidden by policy (mode={mode:?})")]
    Forbidden {
        context: &'static str,
        mode: DerivativeStorageMode,
    },
}

pub trait ResidentBytes {
    fn resident_bytes(&self) -> usize;
}

impl ResourcePolicy {
    /// Conservative default suitable for general-purpose use.
    pub fn default_library() -> Self {
        Self {
            max_single_materialization_bytes: 256 * 1024 * 1024, // 256 MB
            max_operator_cache_bytes: 1024 * 1024 * 1024,        // 1 GB
            max_spatial_distance_cache_bytes: 512 * 1024 * 1024, // 512 MB
            max_owned_data_cache_bytes: 512 * 1024 * 1024,       // 512 MB
            row_chunk_target_bytes: 8 * 1024 * 1024,             // 8 MB per chunk
            derivative_storage_mode: DerivativeStorageMode::AnalyticOperatorRequired,
        }
    }

    /// Permissive mode for small-data usage and tests.
    pub fn permissive_small_data() -> Self {
        Self {
            max_single_materialization_bytes: 2 * 1024 * 1024 * 1024, // 2 GB
            max_operator_cache_bytes: 2 * 1024 * 1024 * 1024,
            max_spatial_distance_cache_bytes: 1024 * 1024 * 1024,
            max_owned_data_cache_bytes: 1024 * 1024 * 1024,
            row_chunk_target_bytes: 64 * 1024 * 1024,
            derivative_storage_mode: DerivativeStorageMode::MaterializeIfSmall,
        }
    }

    pub fn material_policy(&self) -> MaterializationPolicy {
        MaterializationPolicy {
            max_single_dense_bytes: self.max_single_materialization_bytes,
            max_cached_dense_bytes: self.max_operator_cache_bytes,
            row_chunk_target_bytes: self.row_chunk_target_bytes,
            allow_operator_materialization: matches!(
                self.derivative_storage_mode,
                DerivativeStorageMode::MaterializeIfSmall
            ),
            allow_diagnostic_materialization: !matches!(
                self.derivative_storage_mode,
                DerivativeStorageMode::AnalyticOperatorRequired
            ),
        }
    }
}

/// Returns how many rows to stream per chunk so that each chunk uses approximately
/// `target_bytes` given a row width of `cols` f64 entries.
pub fn rows_for_target_bytes(target_bytes: usize, cols: usize) -> usize {
    let bytes_per_row = cols.saturating_mul(std::mem::size_of::<f64>()).max(1);
    (target_bytes / bytes_per_row).max(1)
}
