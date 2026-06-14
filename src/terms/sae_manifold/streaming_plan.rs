use super::*;

const SAE_BYTES_PER_F64: usize = 8;

const SAE_HOST_IN_CORE_FALLBACK_BYTES: usize = 2 * 1024 * 1024 * 1024;

const SAE_HOST_MEMORY_BUDGET_FRACTION_NUMERATOR: usize = 3;

const SAE_HOST_MEMORY_BUDGET_FRACTION_DENOMINATOR: usize = 5;

const SAE_CPU_L2_CACHE_BYTES: usize = 1024 * 1024;

const SAE_CHUNK_CACHE_MULTIPLE: usize = 8;

const SAE_MIN_STREAMING_CHUNK_ROWS: usize = 256;

const SAE_MATRIX_FREE_VECTOR_WORKSPACE_MULTIPLIER: usize = 32;
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SaeStreamingPlan {
    pub streaming: bool,
    pub chunk_size: usize,
    pub estimated_full_batch_bytes: usize,
    pub estimated_dense_schur_bytes: usize,
    pub estimated_row_cross_bytes: usize,
    pub estimated_direct_peak_bytes: usize,
    pub estimated_matrix_free_peak_bytes: usize,
    pub in_core_budget_bytes: usize,
    pub host_available_bytes: usize,
    pub direct_admitted: bool,
    pub matrix_free_admitted: bool,
}

fn sae_streaming_plan_from_budget(
    n_obs: usize,
    total_basis: usize,
    k_atoms: usize,
    d_max: usize,
    border_dim: usize,
    in_core_budget_bytes: usize,
    chunk_window_bytes: usize,
    host_available_bytes: usize,
) -> SaeStreamingPlan {
    let per_row_words = total_basis
        .saturating_mul(1 + d_max)
        .saturating_add(k_atoms)
        .max(1);
    let per_row_bytes = per_row_words.saturating_mul(SAE_BYTES_PER_F64);
    let full_batch_bytes = n_obs.saturating_mul(per_row_bytes);
    let dense_schur_bytes = border_dim
        .saturating_mul(border_dim)
        .saturating_mul(SAE_BYTES_PER_F64);
    let row_block_dim = k_atoms.saturating_mul(1usize.saturating_add(d_max));
    let row_cross_bytes = n_obs
        .saturating_mul(row_block_dim)
        .saturating_mul(border_dim)
        .saturating_mul(SAE_BYTES_PER_F64);
    let direct_peak_bytes = full_batch_bytes
        .saturating_add(row_cross_bytes)
        .saturating_add(dense_schur_bytes);
    let matrix_free_peak_bytes = chunk_window_bytes
        .min(full_batch_bytes.max(per_row_bytes))
        .saturating_add(row_cross_bytes)
        .saturating_add(
            border_dim
                .saturating_mul(SAE_BYTES_PER_F64)
                .saturating_mul(SAE_MATRIX_FREE_VECTOR_WORKSPACE_MULTIPLIER),
        );
    let direct_admitted = direct_peak_bytes <= in_core_budget_bytes;
    let matrix_free_admitted = matrix_free_peak_bytes <= in_core_budget_bytes;
    let rows_per_chunk = (chunk_window_bytes / per_row_bytes).max(SAE_MIN_STREAMING_CHUNK_ROWS);
    SaeStreamingPlan {
        streaming: !direct_admitted,
        chunk_size: if direct_admitted {
            n_obs.max(1)
        } else {
            rows_per_chunk.min(n_obs).max(1)
        },
        estimated_full_batch_bytes: full_batch_bytes,
        estimated_dense_schur_bytes: dense_schur_bytes,
        estimated_row_cross_bytes: row_cross_bytes,
        estimated_direct_peak_bytes: direct_peak_bytes,
        estimated_matrix_free_peak_bytes: matrix_free_peak_bytes,
        in_core_budget_bytes,
        host_available_bytes,
        direct_admitted,
        matrix_free_admitted,
    }
}

pub fn sae_streaming_plan_for_shape(
    n_obs: usize,
    total_basis: usize,
    k_atoms: usize,
    d_max: usize,
    border_dim: usize,
) -> SaeStreamingPlan {
    let (budget, chunk_window, host_available) = match crate::gpu::runtime::GpuRuntime::global() {
        Some(rt) => {
            let aggregate_budget: usize = rt
                .device_ordinals()
                .iter()
                .map(|&ord| rt.memory_budget_for(ord))
                .sum();
            let per_device_budget = aggregate_budget / rt.device_count().max(1);
            let window =
                (per_device_budget / 16).max(SAE_CPU_L2_CACHE_BYTES * SAE_CHUNK_CACHE_MULTIPLE);
            let host_available = sae_host_available_memory_bytes();
            (
                (aggregate_budget / 4).min(host_available),
                window,
                host_available,
            )
        }
        None => {
            let (budget, host_available) = sae_host_in_core_budget_bytes();
            (
                budget,
                SAE_CPU_L2_CACHE_BYTES * SAE_CHUNK_CACHE_MULTIPLE,
                host_available,
            )
        }
    };
    sae_streaming_plan_from_budget(
        n_obs,
        total_basis,
        k_atoms,
        d_max,
        border_dim,
        budget,
        chunk_window,
        host_available,
    )
}

impl SaeStreamingPlan {
    pub(crate) fn admitted_or_error(
        self,
        n: usize,
        p: usize,
        k_atoms: usize,
    ) -> Result<Self, String> {
        if self.direct_admitted || self.matrix_free_admitted {
            Ok(self)
        } else {
            Err(format!(
                "SaeManifoldTerm::streaming_plan: predicted working set {} bytes exceeds budget {} bytes; shape n={n},p={p},K={k_atoms}",
                self.estimated_matrix_free_peak_bytes, self.in_core_budget_bytes
            ))
        }
    }

    pub(crate) fn solve_options_for_border_dim(self, border_dim: usize) -> ArrowSolveOptions {
        if self.direct_admitted {
            ArrowSolveOptions::automatic(border_dim)
        } else {
            ArrowSolveOptions::inexact_pcg()
        }
    }

    pub(crate) fn direct_logdet_admitted(self) -> bool {
        self.direct_admitted
    }
}

fn sae_host_available_memory_bytes() -> usize {
    let mut sys = sysinfo::System::new();
    sys.refresh_memory();
    let available = sys.available_memory() as usize;
    if available == 0 {
        SAE_HOST_IN_CORE_FALLBACK_BYTES
    } else {
        available
    }
}

fn sae_host_in_core_budget_bytes() -> (usize, usize) {
    let available = sae_host_available_memory_bytes();
    let fraction = available.saturating_mul(SAE_HOST_MEMORY_BUDGET_FRACTION_NUMERATOR)
        / SAE_HOST_MEMORY_BUDGET_FRACTION_DENOMINATOR;
    (fraction.max(SAE_HOST_IN_CORE_FALLBACK_BYTES), available)
}
