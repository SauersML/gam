use crate::custom_family::{ParameterBlockSpec, PenaltyMatrix};
use crate::model_types::PenaltySpec;
use gam_linalg::matrix::DesignMatrix;
use ndarray::Array1;

const DEFAULT_GAUGE_PRIORITY: u8 = 100;

/// Generic block input for high-level built-in family APIs.
#[derive(Clone)]
pub struct ParameterBlockInput {
    pub design: DesignMatrix,
    pub offset: Array1<f64>,
    pub penalties: Vec<PenaltySpec>,
    /// Structural nullspace dimension per penalty (same length as `penalties`).
    /// Empty means "use eigenvalue-based rank detection."
    pub nullspace_dims: Vec<usize>,
    pub initial_log_lambdas: Option<Array1<f64>>,
    pub initial_beta: Option<Array1<f64>>,
}

impl ParameterBlockInput {
    pub fn intospec(self, name: &str) -> Result<ParameterBlockSpec, String> {
        self.intospec_with_gauge_priority(name, DEFAULT_GAUGE_PRIORITY)
    }

    pub fn intospec_with_gauge_priority(
        self,
        name: &str,
        gauge_priority: u8,
    ) -> Result<ParameterBlockSpec, String> {
        let p = self.design.ncols();
        let n = self.design.nrows();
        if self.offset.len() != n {
            return Err(format!(
                "block '{name}' offset length mismatch: got {}, expected {n}",
                self.offset.len()
            ));
        }
        if let Some(beta0) = &self.initial_beta
            && beta0.len() != p
        {
            return Err(format!(
                "block '{name}' initial_beta length mismatch: got {}, expected {p}",
                beta0.len()
            ));
        }
        for (k, s) in self.penalties.iter().enumerate() {
            match s {
                PenaltySpec::Block {
                    local, col_range, ..
                } => {
                    if col_range.end > p
                        || local.nrows() != col_range.len()
                        || local.ncols() != col_range.len()
                    {
                        return Err(format!(
                            "block '{name}' penalty {k} block shape mismatch: col_range={}..{}, local={}x{}, total_dim={p}",
                            col_range.start,
                            col_range.end,
                            local.nrows(),
                            local.ncols()
                        ));
                    }
                }
                PenaltySpec::Dense(m) | PenaltySpec::DenseWithMean { matrix: m, .. } => {
                    let (r, c) = m.dim();
                    if r != p || c != p {
                        return Err(format!(
                            "block '{name}' penalty {k} must be {p}x{p}, got {r}x{c}"
                        ));
                    }
                }
            }
        }
        let k = self.penalties.len();
        let initial_log_lambdas = self
            .initial_log_lambdas
            .unwrap_or_else(|| Array1::<f64>::zeros(k));
        if initial_log_lambdas.len() != k {
            return Err(format!(
                "block '{name}' initial_log_lambdas length mismatch: got {}, expected {k}",
                initial_log_lambdas.len()
            ));
        }
        Ok(ParameterBlockSpec {
            name: name.to_string(),
            design: self.design,
            offset: self.offset,
            penalties: {
                self.penalties
                    .into_iter()
                    .map(|spec| match spec {
                        PenaltySpec::Block {
                            local, col_range, ..
                        } => PenaltyMatrix::Blockwise {
                            local,
                            col_range,
                            total_dim: p,
                        },
                        PenaltySpec::Dense(m) | PenaltySpec::DenseWithMean { matrix: m, .. } => {
                            PenaltyMatrix::Dense(m)
                        }
                    })
                    .collect()
            },
            nullspace_dims: self.nullspace_dims,
            initial_log_lambdas,
            initial_beta: self.initial_beta,
            gauge_priority,
            jacobian_callback: None,
            stacked_design: None,
            stacked_offset: None,
        })
    }
}
