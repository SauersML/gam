//! The known block before any declared law is absorbed, its one absorption owner, and its readers from a GPT-NeoX MLP's
//! and a Qwen3 MLP's torch parameters (#2946).
//!
//! A dense block `F(h) = Σ_j u_j σ(b_j + w_jᵀ h) + b_out` acts on the post-norm activation `h ∈ ℝ^D`. A declared law
//! `h = h₀ + L Z`, `Z ~ N(0, I_d)`, absorbs into it as readers `W L` and biases `b + W h₀` ([`UnabsorbedBlock::absorb`]),
//! once per context in [`super::context::ContextBlocks::new`]. The output bias shifts every output by one constant: no
//! variance, discarded error or frame gradient reads it, but the retained response at a point carries it.
//!
//! # Torch layout
//!
//! [`UnabsorbedBlock::from_torch_parameters`] reads a `GPTNeoXMLP`'s `named_parameters()`, with names relative to the
//! module and `nn.Linear` weights shaped `out × in`, together with the config's `hidden_act`:
//! - `dense_h_to_4h.weight` and `dense_h_to_4h.bias` give the readers and biases;
//! - `dense_4h_to_h.weight` and `dense_4h_to_h.bias` give the writers and the output bias.
//!
//! A bias the module does not hold is zero: that is the layer's definition (`bias=False`), not a default. A name outside
//! the layout is refused, never ignored. `hidden_act` is parsed by its one owner,
//! [`GaussianActivation::from_hidden_act`], which refuses the tanh approximations of GELU and every unknown tag. This
//! reader keeps only the layout's own constraint: a dense block has kernels for ReLU and the exact GELU, while SiLU
//! gates a gated layout.
//!
//! # Gated layout
//!
//! A SwiGLU block `F(h) = Σ_j u_j (a_jᵀ h + c_j) s(w_jᵀ h + b_j) + b_out` absorbs the same law as gate readers `W L`,
//! gate biases `b + W h₀`, up readers `A L` and up biases `c + A h₀` ([`UnabsorbedGatedBlock::absorb`]).
//! [`UnabsorbedGatedBlock::from_torch_parameters`] reads a `Qwen3MLP`: `gate_proj.weight` gives `W`, `up_proj.weight`
//! gives `A` and `down_proj.weight` gives `U`. The layer holds no biases, so `b`, `c` and `b_out` are zero by its
//! definition, and `hidden_act` must name the SiLU gate.

use std::collections::BTreeMap;
use std::fmt;

use gam_linalg::faer_ndarray::fast_ab;
use gam_math::gaussian_activation::{GaussianActivation, GaussianActivationError};
use ndarray::{Array1, Array2, ArrayD, ArrayView1, ArrayView2, Ix1, Ix2};

use super::subspace::{self, KnownBlock, KnownGatedBlock, ResponseError};

/// The parameter names of a `GPTNeoXMLP`.
const GPT_NEOX_PARAMETERS: [&str; 4] =
    ["dense_h_to_4h.weight", "dense_h_to_4h.bias", "dense_4h_to_h.weight", "dense_4h_to_h.bias"];

/// The parameter names of a `Qwen3MLP`, whose three projections carry no bias.
const QWEN3_PARAMETERS: [&str; 3] = ["gate_proj.weight", "up_proj.weight", "down_proj.weight"];

/// A dense known block before any declared law is absorbed: `F(h) = Σ_j u_j σ(b_j + w_jᵀ h) + b_out` on `h ∈ ℝ^D`.
#[derive(Clone, Debug)]
pub struct UnabsorbedBlock {
    /// `W`, `h × D`.
    pub readers: Array2<f64>,
    /// `b`, length `h`.
    pub biases: Array1<f64>,
    /// `U`, `p × h`.
    pub writers: Array2<f64>,
    /// `b_out`, length `p`: zeros for a layer without an output bias.
    pub output_bias: Array1<f64>,
    /// `M`, `p × p`, symmetric positive definite.
    pub metric: Array2<f64>,
    pub activation: GaussianActivation,
}

/// Why a torch MLP's parameters, a GPT-NeoX or a Qwen3 layout, were refused.
#[derive(Clone, Debug, PartialEq)]
pub enum TorchLayoutError {
    /// `hidden_act` names no activation with a Gaussian kernel.
    Activation(GaussianActivationError),
    /// `hidden_act` names an activation the layout does not carry: a dense block has ReLU or the exact GELU, and a
    /// gated block the SiLU gate.
    ActivationOutsideLayout { hidden_act: String },
    /// A parameter outside the layout being read.
    UnexpectedParameter { name: String },
    MissingParameter { name: &'static str },
    /// A weight is not a matrix or a bias is not a vector.
    WrongRank {
        name: &'static str,
        expected: usize,
        shape: Vec<usize>,
    },
    DimensionMismatch {
        context: &'static str,
        expected: usize,
        got: usize,
    },
    NonFinite { name: String },
}

impl fmt::Display for TorchLayoutError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Activation(error) => write!(f, "torch MLP: {error}"),
            Self::ActivationOutsideLayout { hidden_act } => write!(
                f,
                "torch MLP: hidden_act {hidden_act} does not fit the layout: a dense GPT-NeoX block reads relu or the exact gelu, a gated Qwen3 block the silu gate"
            ),
            Self::UnexpectedParameter { name } => {
                write!(f, "torch MLP: parameter {name} is not in the layout (GPT-NeoX dense_h_to_4h, dense_4h_to_h; Qwen3 gate_proj, up_proj, down_proj)")
            }
            Self::MissingParameter { name } => write!(f, "torch MLP: missing parameter {name}"),
            Self::WrongRank {
                name,
                expected,
                shape,
            } => write!(f, "torch MLP: {name} must have {expected} axes, got shape {shape:?}"),
            Self::DimensionMismatch {
                context,
                expected,
                got,
            } => write!(f, "torch MLP: {context}: expected {expected}, got {got}"),
            Self::NonFinite { name } => write!(f, "torch MLP: parameter {name} holds a non-finite entry"),
        }
    }
}

impl std::error::Error for TorchLayoutError {}

impl UnabsorbedBlock {
    /// The block under the declared law `h = h₀ + L Z`, `Z ~ N(0, I_d)`, with `baseline = h₀` (`D`) and `loading = L`
    /// (`D × d`): readers `W L` (`h × d`) and biases `b + W h₀`, with the writers, output bias, metric and activation
    /// unchanged.
    pub fn absorb(
        &self,
        baseline: ArrayView1<'_, f64>,
        loading: ArrayView2<'_, f64>,
    ) -> Result<KnownBlock, ResponseError> {
        let ambient = self.readers.ncols();
        subspace::require_length("declared law baseline", ambient, baseline.len())?;
        subspace::require_length("declared law loading rows", ambient, loading.nrows())?;
        let readers = fast_ab(&self.readers, &loading);
        let biases = &self.biases + &self.readers.dot(&baseline);
        KnownBlock::new(
            readers,
            biases,
            self.writers.clone(),
            self.output_bias.clone(),
            self.metric.view(),
            self.activation,
        )
    }

    /// Read a `GPTNeoXMLP`'s parameters, keyed by their `named_parameters()` names, its config's `hidden_act`, and the
    /// output metric `M` (`p × p`). The arrays move into the block; nothing is copied.
    pub fn from_torch_parameters(
        mut parameters: BTreeMap<String, ArrayD<f64>>,
        hidden_act: &str,
        metric: Array2<f64>,
    ) -> Result<Self, TorchLayoutError> {
        if let Some(name) = parameters.keys().find(|name| !GPT_NEOX_PARAMETERS.contains(&name.as_str())) {
            return Err(TorchLayoutError::UnexpectedParameter { name: name.clone() });
        }
        let activation = match GaussianActivation::from_hidden_act(hidden_act).map_err(TorchLayoutError::Activation)? {
            activation @ (GaussianActivation::Relu | GaussianActivation::ExactGelu) => activation,
            GaussianActivation::Silu => {
                return Err(TorchLayoutError::ActivationOutsideLayout {
                    hidden_act: hidden_act.to_string(),
                });
            }
        };
        let nonfinite = parameters
            .iter()
            .find_map(|(name, value)| (!value.iter().all(|entry| entry.is_finite())).then(|| name.clone()));
        if let Some(name) = nonfinite {
            return Err(TorchLayoutError::NonFinite { name });
        }
        let readers = take_matrix(&mut parameters, "dense_h_to_4h.weight")?;
        let writers = take_matrix(&mut parameters, "dense_4h_to_h.weight")?;
        require_length("dense_4h_to_h.weight columns", readers.nrows(), writers.ncols())?;
        let biases = take_bias(&mut parameters, "dense_h_to_4h.bias", readers.nrows())?;
        let output_bias = take_bias(&mut parameters, "dense_4h_to_h.bias", writers.nrows())?;
        require_length("output metric rows", writers.nrows(), metric.nrows())?;
        require_length("output metric columns", writers.nrows(), metric.ncols())?;
        Ok(Self {
            readers,
            biases,
            writers,
            output_bias,
            metric,
            activation,
        })
    }
}

/// A SwiGLU known block before any declared law is absorbed:
/// `F(h) = Σ_j u_j (a_jᵀ h + c_j) s(w_jᵀ h + b_j) + b_out` on `h ∈ ℝ^D`, with the SiLU gate `s(t) = t σ(t)`.
#[derive(Clone, Debug)]
pub struct UnabsorbedGatedBlock {
    /// `W`, `h × D`: the gate readers.
    pub gate_readers: Array2<f64>,
    /// `b`, length `h`.
    pub gate_biases: Array1<f64>,
    /// `A`, `h × D`: the up-projection readers.
    pub up_readers: Array2<f64>,
    /// `c`, length `h`.
    pub up_biases: Array1<f64>,
    /// `U`, `p × h`.
    pub writers: Array2<f64>,
    /// `b_out`, length `p`: zeros for a layer without an output bias.
    pub output_bias: Array1<f64>,
    /// `M`, `p × p`, symmetric positive definite.
    pub metric: Array2<f64>,
}

impl UnabsorbedGatedBlock {
    /// The block under the declared law `h = h₀ + L Z`, `Z ~ N(0, I_d)`, with `baseline = h₀` (`D`) and `loading = L`
    /// (`D × d`): gate readers `W L` and biases `b + W h₀`, up readers `A L` and biases `c + A h₀`, with the writers,
    /// output bias and metric unchanged.
    pub fn absorb(
        &self,
        baseline: ArrayView1<'_, f64>,
        loading: ArrayView2<'_, f64>,
    ) -> Result<KnownGatedBlock, ResponseError> {
        let ambient = self.gate_readers.ncols();
        subspace::require_length("declared law baseline", ambient, baseline.len())?;
        subspace::require_length("declared law loading rows", ambient, loading.nrows())?;
        KnownGatedBlock::new(
            fast_ab(&self.gate_readers, &loading),
            &self.gate_biases + &self.gate_readers.dot(&baseline),
            fast_ab(&self.up_readers, &loading),
            &self.up_biases + &self.up_readers.dot(&baseline),
            self.writers.clone(),
            self.output_bias.clone(),
            self.metric.view(),
        )
    }

    /// Read a `Qwen3MLP`'s parameters, keyed by their `named_parameters()` names: `gate_proj.weight` and
    /// `up_proj.weight` (`h × D`) and `down_proj.weight` (`D × h`), with no biases (`bias=False`), so every bias is
    /// zero by the layer's definition. `hidden_act` must name the SiLU gate. The arrays move into the block.
    pub fn from_torch_parameters(
        mut parameters: BTreeMap<String, ArrayD<f64>>,
        hidden_act: &str,
        metric: Array2<f64>,
    ) -> Result<Self, TorchLayoutError> {
        if let Some(name) = parameters.keys().find(|name| !QWEN3_PARAMETERS.contains(&name.as_str())) {
            return Err(TorchLayoutError::UnexpectedParameter { name: name.clone() });
        }
        match GaussianActivation::from_hidden_act(hidden_act).map_err(TorchLayoutError::Activation)? {
            GaussianActivation::Silu => {}
            GaussianActivation::Relu | GaussianActivation::ExactGelu => {
                return Err(TorchLayoutError::ActivationOutsideLayout {
                    hidden_act: hidden_act.to_string(),
                });
            }
        }
        let nonfinite = parameters
            .iter()
            .find_map(|(name, value)| (!value.iter().all(|entry| entry.is_finite())).then(|| name.clone()));
        if let Some(name) = nonfinite {
            return Err(TorchLayoutError::NonFinite { name });
        }
        let gate_readers = take_matrix(&mut parameters, "gate_proj.weight")?;
        let up_readers = take_matrix(&mut parameters, "up_proj.weight")?;
        let writers = take_matrix(&mut parameters, "down_proj.weight")?;
        require_length("up_proj.weight rows", gate_readers.nrows(), up_readers.nrows())?;
        require_length("up_proj.weight columns", gate_readers.ncols(), up_readers.ncols())?;
        require_length("down_proj.weight columns", gate_readers.nrows(), writers.ncols())?;
        require_length("output metric rows", writers.nrows(), metric.nrows())?;
        require_length("output metric columns", writers.nrows(), metric.ncols())?;
        let width = gate_readers.nrows();
        let output_dim = writers.nrows();
        Ok(Self {
            gate_readers,
            gate_biases: Array1::zeros(width),
            up_readers,
            up_biases: Array1::zeros(width),
            writers,
            output_bias: Array1::zeros(output_dim),
            metric,
        })
    }
}

fn take_matrix(
    parameters: &mut BTreeMap<String, ArrayD<f64>>,
    name: &'static str,
) -> Result<Array2<f64>, TorchLayoutError> {
    let value = parameters.remove(name).ok_or(TorchLayoutError::MissingParameter { name })?;
    let shape = value.shape().to_vec();
    value.into_dimensionality::<Ix2>().ok().ok_or(TorchLayoutError::WrongRank {
        name,
        expected: 2,
        shape,
    })
}

/// The bias `name` of length `length`, or zeros when the layer holds none.
fn take_bias(
    parameters: &mut BTreeMap<String, ArrayD<f64>>,
    name: &'static str,
    length: usize,
) -> Result<Array1<f64>, TorchLayoutError> {
    let Some(value) = parameters.remove(name) else {
        return Ok(Array1::zeros(length));
    };
    let shape = value.shape().to_vec();
    let bias = value.into_dimensionality::<Ix1>().ok().ok_or(TorchLayoutError::WrongRank {
        name,
        expected: 1,
        shape,
    })?;
    require_length(name, length, bias.len())?;
    Ok(bias)
}

fn require_length(context: &'static str, expected: usize, got: usize) -> Result<(), TorchLayoutError> {
    if expected == got {
        Ok(())
    } else {
        Err(TorchLayoutError::DimensionMismatch {
            context,
            expected,
            got,
        })
    }
}

#[cfg(test)]
#[path = "raw_block_tests.rs"]
mod raw_block_tests;
