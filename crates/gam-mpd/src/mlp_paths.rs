//! Path programs of residual ReLU MLP stacks and their two-part code (#2951).
//!
//! # The model
//!
//! A [`ResidualMlp`] reads `x ∈ R^{n_in}` through a fixed embedding `E` (`n_in × d`), runs `L`
//! blocks `r ← r + relu(r W_inᵀ) W_outᵀ` on the residual stream and reads `y = r U` out
//! (`U`: `d × n_out`). It is the residual-MLP toy of APD and SPD (Braun et al. 2025, Bushnaq et
//! al. 2025) and the MLP half of a transformer layer stack.
//!
//! # The exact rewrite
//!
//! Every residual state is a sum of sources, `r_l = x E + Σ_{k<l} a_k W_out,kᵀ` with
//! `a_k = relu(h_k)`, so
//!
//! ```text
//! h_l = x (E W_in,lᵀ) + Σ_{k<l} a_k (W_out,kᵀ W_in,lᵀ),      y = x (E U) + Σ_k a_k (W_out,kᵀ U).
//! ```
//!
//! A [`PathProgram`] holds, per block, the read operator from every earlier source (the input
//! coordinates, then the earlier blocks' neurons) and the write operator to the outputs. The
//! `d`-wide stream is gone: the identity is exact in exact arithmetic, and
//! [`PathProgram::from_model`]'s rounding is measured against [`ResidualMlp::forward`] by the
//! caller. The direct operator `E U` is made of the fixed read-in and read-out only; it is
//! declared, like `E` and `U`, and never sent.
//!
//! # Components
//!
//! A component is a source with its rows: an input coordinate with its read row into every block,
//! or a neuron with its read rows into the later blocks and its write row. On an input where its
//! source is zero (an input coordinate that is off, a neuron whose pre-activation is not positive)
//! a component contributes exactly zero, so which components act on an input is exact, not
//! estimated, and ablating any set of inactive components leaves the output bit-identical.
//!
//! # Gauge
//!
//! ReLU is positively homogeneous: scaling a neuron's read column by `c > 0` and its write row and
//! outgoing read rows by `1/c` leaves the function unchanged. [`PathProgram::canonical_gauge`]
//! fixes `c` so every live write row has unit norm; the read entries into a neuron then move the
//! output by comparable amounts per unit, which is what one precision per row assumes.
//!
//! # The two-part code
//!
//! The behaviour is the model's output on a sample of `M` inputs, read as a Gaussian with the
//! variance `σ²` the behaviour declares (the model's own residual variance on its task), explained
//! `n` times ([`GaussianBehaviour`]). `KL(N(y, σ²I) ‖ N(ŷ, σ²I)) = |y − ŷ|²/(2σ²)`, so
//!
//! ```text
//! L_total(P) = L(P) + (n/M) Σ_rows |y_row − ŷ_row|² / (2σ² ln 2).
//! ```
//!
//! `L(P)` ([`PathProgram::code_bits`]) is one message: the block count and widths in the prefix
//! code; then per row (every read row of every block, then every write row) the present entries
//! as a subset in the enumerative code and, when any is present, the row's dyadic precision
//! (`precision::DeclaredPrecision`) as a signed prefix integer and each entry's lattice index as a
//! signed prefix integer. An absent entry is a zero: pruning is the lattice index `0` taken out
//! of the subset.
//!
//! # The fit
//!
//! [`fit`] sweeps Gauss–Seidel over rows, last block first, writes before reads. Each row is solved
//! against the current residual of the whole program with its Gauss–Newton block `H` and gradient
//! `g` (in bits): exact for a write row, whose output is linear in it; linearised at the current
//! gates for a read row. The row objective `½ΔᵀHΔ + gᵀΔ + L(row)` is minimised over the lattice by
//! exact coordinate descent (each entry takes the best of `0`, and the two lattice points around
//! its unconstrained optimum, by its exact change of code length). The precision is solved, not
//! declared: `δ² = 12 k/(ln 2 · tr_k H)` balances the index bits of `k` entries against their
//! expected rounding cost `tr_k H δ²/24`, and the row keeps the best of the dyadic steps around it
//! by its exact row objective. After each row the affected inputs are re-executed exactly, so later
//! rows fit the true residual, rounding included. A sweep that shortens the measured total by less
//! than one bit, the code's own unit, ends the fit; a sweep that lengthens it is undone.

use crate::codec::{CodecError, prefix_integer_len_bits, signed_prefix_integer_len_bits, subset_code_len_bits};
use crate::precision::DeclaredPrecision;
use ndarray::{Array1, Array2, ArrayView1, Axis};
use std::f64::consts::LN_2;
use std::fmt;

/// One input: its nonzero coordinates.
pub type SparseRow = Vec<(usize, f64)>;

/// One residual block `r ← r + relu(r W_inᵀ) W_outᵀ`.
#[derive(Clone, Debug, PartialEq)]
pub struct MlpBlock {
    /// `width × d`.
    pub w_in: Array2<f64>,
    /// `d × width`.
    pub w_out: Array2<f64>,
}

/// A residual stack of ReLU MLP blocks between a fixed read-in and read-out.
#[derive(Clone, Debug, PartialEq)]
pub struct ResidualMlp {
    /// `n_in × d`.
    pub embed: Array2<f64>,
    /// `d × n_out`.
    pub unembed: Array2<f64>,
    pub blocks: Vec<MlpBlock>,
}

/// A refused model, program or code.
#[derive(Debug, Clone, PartialEq)]
pub enum PathError {
    Shape(String),
    Code(CodecError),
    Precision(String),
    /// A row holds reals that are not on a declared lattice, so it has no code.
    Unquantized { block: usize, write: bool, row: usize },
    Behaviour(String),
}

impl fmt::Display for PathError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Shape(message) => write!(f, "mlp paths shape: {message}"),
            Self::Code(error) => write!(f, "mlp paths code: {error:?}"),
            Self::Precision(message) => write!(f, "mlp paths precision: {message}"),
            Self::Unquantized { block, write, row } => {
                let kind = if *write { "write" } else { "read" };
                write!(f, "mlp paths: {kind} row {row} of block {block} is not on a lattice")
            }
            Self::Behaviour(message) => write!(f, "mlp paths behaviour: {message}"),
        }
    }
}

impl std::error::Error for PathError {}

impl From<CodecError> for PathError {
    fn from(error: CodecError) -> Self {
        Self::Code(error)
    }
}

impl ResidualMlp {
    pub fn validate(&self) -> Result<(), PathError> {
        let d = self.embed.ncols();
        if self.unembed.nrows() != d {
            return Err(PathError::Shape(format!("unembed has {} rows for a {d}-wide stream", self.unembed.nrows())));
        }
        for (l, block) in self.blocks.iter().enumerate() {
            let width = block.w_in.nrows();
            if block.w_in.ncols() != d || block.w_out.nrows() != d || block.w_out.ncols() != width {
                return Err(PathError::Shape(format!(
                    "block {l}: w_in {:?}, w_out {:?} on a {d}-wide stream",
                    block.w_in.dim(),
                    block.w_out.dim()
                )));
            }
        }
        Ok(())
    }

    /// The outputs on `inputs` through the `d`-wide stream: the reference the rewrite is held to.
    pub fn forward(&self, inputs: &[SparseRow]) -> Result<Array2<f64>, PathError> {
        self.validate()?;
        let d = self.embed.ncols();
        let mut stream = Array2::<f64>::zeros((inputs.len(), d));
        for (row, input) in inputs.iter().enumerate() {
            for &(i, v) in input {
                if i >= self.embed.nrows() {
                    return Err(PathError::Shape(format!("input coordinate {i} beyond {}", self.embed.nrows())));
                }
                stream.row_mut(row).scaled_add(v, &self.embed.row(i));
            }
        }
        for block in &self.blocks {
            let hidden = stream.dot(&block.w_in.t()).mapv(|t| t.max(0.0));
            stream += &hidden.dot(&block.w_out.t());
        }
        Ok(stream.dot(&self.unembed))
    }
}

/// One block of a path program.
#[derive(Clone, Debug, PartialEq)]
pub struct PathBlock {
    /// Rows: the sources before this block (inputs, then earlier blocks' neurons); columns: this
    /// block's neurons.
    pub read: Array2<f64>,
    /// Rows: this block's neurons; columns: the outputs.
    pub write: Array2<f64>,
    /// Each read row's lattice, `None` while its reals are unquantized.
    pub read_precision: Vec<Option<DeclaredPrecision>>,
    pub write_precision: Vec<Option<DeclaredPrecision>>,
}

impl PathBlock {
    pub fn width(&self) -> usize {
        self.read.ncols()
    }
}

/// A residual MLP stack over its sources (see the module note).
#[derive(Clone, Debug, PartialEq)]
pub struct PathProgram {
    pub inputs: usize,
    pub outputs: usize,
    /// `E U`, declared.
    pub direct: Array2<f64>,
    pub blocks: Vec<PathBlock>,
}

/// One input executed: every block's pre-activations and activations, and the output.
#[derive(Clone, Debug, PartialEq)]
pub struct RowTrace {
    pub pre: Vec<Array1<f64>>,
    pub act: Vec<Array1<f64>>,
    pub out: Array1<f64>,
}

impl PathProgram {
    /// The exact rewrite of `model` (module note).
    pub fn from_model(model: &ResidualMlp) -> Result<Self, PathError> {
        model.validate()?;
        let direct = model.embed.dot(&model.unembed);
        let mut blocks: Vec<PathBlock> = Vec::with_capacity(model.blocks.len());
        for (l, block) in model.blocks.iter().enumerate() {
            let mut parts = vec![model.embed.dot(&block.w_in.t())];
            for earlier in &model.blocks[..l] {
                parts.push(earlier.w_out.t().dot(&block.w_in.t()));
            }
            let views: Vec<_> = parts.iter().map(|p| p.view()).collect();
            let read = ndarray::concatenate(Axis(0), &views).map_err(|e| PathError::Shape(e.to_string()))?;
            let write = block.w_out.t().dot(&model.unembed);
            blocks.push(PathBlock {
                read_precision: vec![None; read.nrows()],
                write_precision: vec![None; write.nrows()],
                read,
                write,
            });
        }
        Ok(Self { inputs: model.embed.nrows(), outputs: model.unembed.ncols(), direct, blocks })
    }

    /// The source index of block `l`'s neuron `0`.
    pub fn source_offset(&self, l: usize) -> usize {
        self.inputs + self.blocks[..l].iter().map(PathBlock::width).sum::<usize>()
    }

    /// Every source's `(block, neuron)`, `None` for an input coordinate.
    pub fn source_neuron(&self, source: usize) -> Option<(usize, usize)> {
        let mut offset = self.inputs;
        for (l, block) in self.blocks.iter().enumerate() {
            if source < offset + block.width() && source >= offset {
                return Some((l, source - offset));
            }
            offset += block.width();
        }
        None
    }

    /// Execute one input.
    pub fn trace_row(&self, input: &[(usize, f64)]) -> RowTrace {
        self.trace_row_ablated(input, None)
    }

    /// Execute one input with the components whose source is marked in `ablated` removed: their
    /// read rows and, for a neuron, its write row contribute nothing.
    pub fn trace_row_ablated(&self, input: &[(usize, f64)], ablated: Option<&[bool]>) -> RowTrace {
        let off = |source: usize| ablated.is_some_and(|mask| mask[source]);
        let mut pre = Vec::with_capacity(self.blocks.len());
        let mut act: Vec<Array1<f64>> = Vec::with_capacity(self.blocks.len());
        let mut out = Array1::<f64>::zeros(self.outputs);
        for &(i, v) in input {
            out.scaled_add(v, &self.direct.row(i));
        }
        for (l, block) in self.blocks.iter().enumerate() {
            let mut h = Array1::<f64>::zeros(block.width());
            for &(i, v) in input {
                if !off(i) {
                    h.scaled_add(v, &block.read.row(i));
                }
            }
            let mut offset = self.inputs;
            for earlier in &act {
                for (j, &a) in earlier.iter().enumerate() {
                    if a > 0.0 && !off(offset + j) {
                        h.scaled_add(a, &block.read.row(offset + j));
                    }
                }
                offset += earlier.len();
            }
            let a = h.mapv(|t| t.max(0.0));
            let own = self.inputs + self.blocks[..l].iter().map(PathBlock::width).sum::<usize>();
            for (j, &value) in a.iter().enumerate() {
                if value > 0.0 && !off(own + j) {
                    out.scaled_add(value, &block.write.row(j));
                }
            }
            pre.push(h);
            act.push(a);
        }
        RowTrace { pre, act, out }
    }

    pub fn forward(&self, inputs: &[SparseRow]) -> Array2<f64> {
        let mut out = Array2::<f64>::zeros((inputs.len(), self.outputs));
        for (row, input) in inputs.iter().enumerate() {
            out.row_mut(row).assign(&self.trace_row(input).out);
        }
        out
    }

    /// Fix the ReLU gauge: every live write row to unit norm (module note).
    pub fn canonical_gauge(&mut self) {
        for l in 0..self.blocks.len() {
            let offset = self.source_offset(l);
            for j in 0..self.blocks[l].width() {
                let norm = self.blocks[l].write.row(j).dot(&self.blocks[l].write.row(j)).sqrt();
                if !(norm > 0.0 && norm.is_finite()) {
                    continue;
                }
                self.blocks[l].write.row_mut(j).mapv_inplace(|w| w / norm);
                self.blocks[l].read.column_mut(j).mapv_inplace(|w| w * norm);
                for later in &mut self.blocks[l + 1..] {
                    later.read.row_mut(offset + j).mapv_inplace(|w| w / norm);
                }
            }
        }
    }

    /// The number of present (nonzero) reals.
    pub fn real_count(&self) -> usize {
        self.blocks
            .iter()
            .map(|b| b.read.iter().chain(b.write.iter()).filter(|v| **v != 0.0).count())
            .sum()
    }

    /// The decoded message length (module note).
    pub fn code_bits(&self) -> Result<u64, PathError> {
        let mut bits = prefix_integer_len_bits(self.blocks.len() as u64 + 1)?;
        for block in &self.blocks {
            bits += prefix_integer_len_bits(block.width() as u64 + 1)?;
        }
        for (l, block) in self.blocks.iter().enumerate() {
            for (row, values) in block.read.outer_iter().enumerate() {
                bits += row_code_bits(values, block.read_precision[row])
                    .map_err(|e| e.at(l, false, row))?;
            }
            for (row, values) in block.write.outer_iter().enumerate() {
                bits += row_code_bits(values, block.write_precision[row])
                    .map_err(|e| e.at(l, true, row))?;
            }
        }
        Ok(bits)
    }
}

/// A row's code failure before its place is known.
enum RowCodeError {
    Unquantized,
    Code(CodecError),
    OffLattice(f64),
}

impl RowCodeError {
    fn at(self, block: usize, write: bool, row: usize) -> PathError {
        match self {
            Self::Unquantized => PathError::Unquantized { block, write, row },
            Self::Code(error) => PathError::Code(error),
            Self::OffLattice(value) => PathError::Precision(format!(
                "{} row {row} of block {block} holds {value}, off its lattice",
                if write { "write" } else { "read" }
            )),
        }
    }
}

fn row_code_bits(values: ArrayView1<'_, f64>, precision: Option<DeclaredPrecision>) -> Result<u64, RowCodeError> {
    let present = values.iter().filter(|v| **v != 0.0).count();
    let mut bits = subset_code_len_bits(values.len(), present).map_err(RowCodeError::Code)?;
    if present == 0 {
        return Ok(bits);
    }
    let precision = precision.ok_or(RowCodeError::Unquantized)?;
    bits += signed_prefix_integer_len_bits(i64::from(precision.fraction_bits())).map_err(RowCodeError::Code)?;
    let step = precision.step();
    for &value in values.iter().filter(|v| **v != 0.0) {
        let index = value / step;
        if index != index.round() || index.abs() > 2f64.powi(53) {
            return Err(RowCodeError::OffLattice(value));
        }
        bits += signed_prefix_integer_len_bits(index as i64).map_err(RowCodeError::Code)?;
    }
    Ok(bits)
}

/// The behaviour a path program explains (module note).
#[derive(Clone, Debug)]
pub struct GaussianBehaviour {
    pub inputs: Vec<SparseRow>,
    /// The model's outputs on `inputs`.
    pub outputs: Array2<f64>,
    /// `σ²` per output coordinate.
    pub variance: f64,
    /// `n`: how many draws of the input law the code explains.
    pub observations: f64,
}

impl GaussianBehaviour {
    fn validate(&self, outputs: usize) -> Result<(), PathError> {
        if self.inputs.is_empty() || self.outputs.dim() != (self.inputs.len(), outputs) {
            return Err(PathError::Behaviour(format!(
                "{} inputs with outputs {:?} for {outputs} output coordinates",
                self.inputs.len(),
                self.outputs.dim()
            )));
        }
        if !(self.variance > 0.0 && self.variance.is_finite() && self.observations > 0.0 && self.observations.is_finite()) {
            return Err(PathError::Behaviour(format!(
                "variance {} and observations {} must be positive and finite",
                self.variance, self.observations
            )));
        }
        Ok(())
    }

    /// `κ`: the data bits are `κ/2 Σ_rows |y − ŷ|²`.
    fn curvature(&self) -> f64 {
        self.observations / (self.inputs.len() as f64 * self.variance * LN_2)
    }

    /// The data bits of `program` on this behaviour.
    pub fn data_bits(&self, program: &PathProgram) -> Result<f64, PathError> {
        self.validate(program.outputs)?;
        let residual = program.forward(&self.inputs) - &self.outputs;
        Ok(0.5 * self.curvature() * residual.iter().map(|e| e * e).sum::<f64>())
    }
}

/// The result of [`fit`].
#[derive(Clone, Debug)]
pub struct PathFit {
    pub program: PathProgram,
    pub code_bits: u64,
    pub data_bits: f64,
    /// `(code bits, data bits)` after each kept sweep.
    pub sweeps: Vec<(u64, f64)>,
}

impl PathFit {
    pub fn total_bits(&self) -> f64 {
        self.code_bits as f64 + self.data_bits
    }
}

/// The fit's working state: the program, every input's trace and residual.
struct State<'a> {
    program: PathProgram,
    behaviour: &'a GaussianBehaviour,
    traces: Vec<RowTrace>,
    /// `ŷ − y` per input.
    residual: Array2<f64>,
    /// The inputs where each input coordinate is nonzero.
    by_input: Vec<Vec<usize>>,
    kappa: f64,
}

impl<'a> State<'a> {
    fn new(program: PathProgram, behaviour: &'a GaussianBehaviour) -> Result<Self, PathError> {
        let mut by_input = vec![Vec::new(); program.inputs];
        for (row, input) in behaviour.inputs.iter().enumerate() {
            for &(i, v) in input {
                if i >= program.inputs {
                    return Err(PathError::Behaviour(format!("input coordinate {i} beyond {}", program.inputs)));
                }
                if v != 0.0 {
                    by_input[i].push(row);
                }
            }
        }
        let mut state = Self {
            traces: Vec::with_capacity(behaviour.inputs.len()),
            residual: Array2::zeros(behaviour.outputs.dim()),
            kappa: behaviour.curvature(),
            program,
            behaviour,
            by_input,
        };
        for row in 0..behaviour.inputs.len() {
            let trace = state.program.trace_row(&behaviour.inputs[row]);
            state.residual.row_mut(row).assign(&(&trace.out - &behaviour.outputs.row(row)));
            state.traces.push(trace);
        }
        Ok(state)
    }

    fn retrace(&mut self, rows: &[usize]) {
        for &row in rows {
            let trace = self.program.trace_row(&self.behaviour.inputs[row]);
            self.residual.row_mut(row).assign(&(&trace.out - &self.behaviour.outputs.row(row)));
            self.traces[row] = trace;
        }
    }

    fn data_bits(&self) -> f64 {
        0.5 * self.kappa * self.residual.iter().map(|e| e * e).sum::<f64>()
    }

    /// The value of source `source` on input `row`.
    fn source_value(&self, row: usize, source: usize) -> f64 {
        match self.program.source_neuron(source) {
            Some((l, j)) => self.traces[row].act[l][j],
            None => self.behaviour.inputs[row].iter().filter(|(i, _)| *i == source).map(|(_, v)| *v).sum(),
        }
    }

    /// The inputs on which `source` is nonzero.
    fn source_rows(&self, source: usize) -> Vec<usize> {
        match self.program.source_neuron(source) {
            Some((l, j)) => (0..self.traces.len()).filter(|&row| self.traces[row].act[l][j] > 0.0).collect(),
            None => self.by_input[source].clone(),
        }
    }

    /// `∂y/∂a` for every neuron of block `l` on input `row` (`width_l × n_out`).
    fn output_jacobian(&self, row: usize, l: usize) -> Array2<f64> {
        let program = &self.program;
        let trace = &self.traces[row];
        let depth = program.blocks.len();
        let mut jac: Vec<Array2<f64>> = vec![Array2::zeros((0, 0)); depth];
        for m in (l..depth).rev() {
            let mut g = program.blocks[m].write.clone();
            let offset = program.source_offset(m);
            for later in m + 1..depth {
                let read = &program.blocks[later].read;
                for (k, &pre) in trace.pre[later].iter().enumerate() {
                    if pre > 0.0 {
                        let downstream = jac[later].row(k).to_owned();
                        for j in 0..g.nrows() {
                            let w = read[[offset + j, k]];
                            if w != 0.0 {
                                g.row_mut(j).scaled_add(w, &downstream);
                            }
                        }
                    }
                }
            }
            jac[m] = g;
        }
        std::mem::take(&mut jac[l])
    }

    fn fit_write_row(&mut self, l: usize, j: usize) -> Result<(), PathError> {
        let outputs = self.program.outputs;
        let mut curvature = 0.0;
        let mut gradient = Array1::<f64>::zeros(outputs);
        let mut rows = Vec::new();
        for (row, trace) in self.traces.iter().enumerate() {
            let a = trace.act[l][j];
            if a > 0.0 {
                curvature += a * a;
                gradient.scaled_add(a, &self.residual.row(row));
                rows.push(row);
            }
        }
        let theta = self.program.blocks[l].write.row(j).to_owned();
        let hessian = RowHessian::Diagonal(Array1::from_elem(outputs, self.kappa * curvature));
        let (values, precision) = solve_row(&theta, &hessian, &(gradient * self.kappa))?;
        let delta = &values - &theta;
        self.program.blocks[l].write.row_mut(j).assign(&values);
        self.program.blocks[l].write_precision[j] = precision;
        if l + 1 == self.program.blocks.len() {
            // The last block's writes move the outputs linearly and nothing else.
            for &row in &rows {
                let a = self.traces[row].act[l][j];
                self.traces[row].out.scaled_add(a, &delta);
                self.residual.row_mut(row).scaled_add(a, &delta);
            }
        } else {
            self.retrace(&rows);
        }
        Ok(())
    }

    fn fit_read_row(&mut self, l: usize, source: usize) -> Result<(), PathError> {
        let width = self.program.blocks[l].width();
        let rows = self.source_rows(source);
        let mut hessian = Array2::<f64>::zeros((width, width));
        let mut gradient = Array1::<f64>::zeros(width);
        for &row in &rows {
            let v = self.source_value(row, source);
            let jac = self.output_jacobian(row, l);
            let live: Vec<usize> = (0..width).filter(|&k| self.traces[row].pre[l][k] > 0.0).collect();
            for (x, &k) in live.iter().enumerate() {
                gradient[k] += v * jac.row(k).dot(&self.residual.row(row));
                for &k2 in &live[x..] {
                    let h = v * v * jac.row(k).dot(&jac.row(k2));
                    hessian[[k, k2]] += h;
                    if k2 != k {
                        hessian[[k2, k]] += h;
                    }
                }
            }
        }
        hessian *= self.kappa;
        gradient *= self.kappa;
        let theta = self.program.blocks[l].read.row(source).to_owned();
        let (values, precision) = solve_row(&theta, &RowHessian::Dense(hessian), &gradient)?;
        self.program.blocks[l].read.row_mut(source).assign(&values);
        self.program.blocks[l].read_precision[source] = precision;
        self.retrace(&rows);
        Ok(())
    }

    fn sweep(&mut self) -> Result<(), PathError> {
        for l in (0..self.program.blocks.len()).rev() {
            for j in 0..self.program.blocks[l].width() {
                self.fit_write_row(l, j)?;
            }
            for source in 0..self.program.source_offset(l) {
                self.fit_read_row(l, source)?;
            }
        }
        Ok(())
    }
}

enum RowHessian {
    Diagonal(Array1<f64>),
    Dense(Array2<f64>),
}

impl RowHessian {
    fn diag(&self, e: usize) -> f64 {
        match self {
            Self::Diagonal(d) => d[e],
            Self::Dense(h) => h[[e, e]],
        }
    }

    /// `gradient += H[:, e] · step`.
    fn move_gradient(&self, gradient: &mut Array1<f64>, e: usize, step: f64) {
        match self {
            Self::Diagonal(d) => gradient[e] += d[e] * step,
            Self::Dense(h) => gradient.scaled_add(step, &h.column(e)),
        }
    }
}

/// The row objective's code part: subset, precision and indices.
fn row_bits(values: &Array1<f64>, precision: DeclaredPrecision) -> Result<f64, PathError> {
    row_code_bits(values.view(), Some(precision))
        .map(|bits| bits as f64)
        .map_err(|e| e.at(0, false, 0))
}

/// Minimise `½ΔᵀHΔ + gᵀΔ + L(row)` over lattice rows `θ + Δ` (module note). Returns the row and
/// its precision (`None` for an empty row).
fn solve_row(
    theta: &Array1<f64>,
    hessian: &RowHessian,
    gradient: &Array1<f64>,
) -> Result<(Array1<f64>, Option<DeclaredPrecision>), PathError> {
    let n = theta.len();
    let empty = Array1::<f64>::zeros(n);
    // The empty row: data change `½θᵀHθ − gᵀθ`.
    let empty_objective = {
        let mut g = gradient.clone();
        let mut value = 0.0;
        for e in 0..n {
            if theta[e] != 0.0 {
                let step = -theta[e];
                value += step * g[e] + 0.5 * hessian.diag(e) * step * step;
                hessian.move_gradient(&mut g, e, step);
            }
        }
        value + subset_code_len_bits(n, 0)? as f64
    };
    let live: Vec<usize> = (0..n).filter(|&e| hessian.diag(e) > 0.0).collect();
    if live.is_empty() {
        return Ok((empty, None));
    }
    let trace: f64 = live.iter().map(|&e| hessian.diag(e)).sum();
    let solved = (12.0 * live.len() as f64 / (LN_2 * trace)).sqrt();
    let centre = (-solved.log2()).round() as i32;
    let mut best: Option<(f64, Array1<f64>, DeclaredPrecision)> = None;
    let consider = |fraction_bits: i32, best: &mut Option<(f64, Array1<f64>, DeclaredPrecision)>| -> Result<bool, PathError> {
        let precision = DeclaredPrecision::new(fraction_bits).map_err(PathError::Precision)?;
        let (objective, values) = lattice_descent(theta, hessian, gradient, precision)?;
        let better = best.as_ref().is_none_or(|(b, _, _)| objective < *b);
        if better {
            *best = Some((objective, values, precision));
        }
        Ok(better)
    };
    consider(centre, &mut best)?;
    // Walk each way while the exact row objective falls.
    for direction in [1, -1] {
        let mut p = centre + direction;
        while consider(p, &mut best)? {
            p += direction;
        }
    }
    let (objective, values, precision) = best.expect("the centre was considered");
    if empty_objective <= objective || values.iter().all(|v| *v == 0.0) {
        return Ok((empty, None));
    }
    Ok((values, Some(precision)))
}

/// Exact coordinate descent on the lattice of `precision`, from the rounded row; returns the row
/// objective and the row.
fn lattice_descent(
    theta: &Array1<f64>,
    hessian: &RowHessian,
    gradient: &Array1<f64>,
    precision: DeclaredPrecision,
) -> Result<(f64, Array1<f64>), PathError> {
    let n = theta.len();
    let step = precision.step();
    let mut values = theta.mapv(|t| (t / step).round() * step);
    // `grad = g + H(values − θ)`, and the data part of the objective.
    let mut grad = gradient.clone();
    let mut data = 0.0;
    for e in 0..n {
        let move_e = values[e] - theta[e];
        if move_e != 0.0 {
            data += move_e * grad[e] + 0.5 * hessian.diag(e) * move_e * move_e;
            hessian.move_gradient(&mut grad, e, move_e);
        }
    }
    let mut code = row_bits(&values, precision)?;
    loop {
        let mut changed = false;
        for e in 0..n {
            let h = hessian.diag(e);
            let current = values[e];
            let mut candidates = vec![0.0];
            if h > 0.0 {
                let optimum = current - grad[e] / h;
                let low = (optimum / step).floor() * step;
                candidates.push(low);
                candidates.push(low + step);
            }
            let mut choice = (0.0, 0.0, current);
            for &candidate in &candidates {
                if candidate == current {
                    continue;
                }
                let delta = candidate - current;
                let data_change = delta * grad[e] + 0.5 * h * delta * delta;
                values[e] = candidate;
                let code_change = row_bits(&values, precision)? - code;
                values[e] = current;
                if data_change + code_change < choice.0 + choice.1 {
                    choice = (data_change, code_change, candidate);
                }
            }
            if choice.2 != current {
                let delta = choice.2 - current;
                values[e] = choice.2;
                data += choice.0;
                code += choice.1;
                hessian.move_gradient(&mut grad, e, delta);
                changed = true;
            }
        }
        if !changed {
            break;
        }
    }
    Ok((data + code, values))
}

/// Fit the shortest two-part code of `behaviour` from `reference` (module note).
pub fn fit(reference: &PathProgram, behaviour: &GaussianBehaviour) -> Result<PathFit, PathError> {
    behaviour.validate(reference.outputs)?;
    let mut program = reference.clone();
    program.canonical_gauge();
    let mut state = State::new(program, behaviour)?;
    let mut kept: Option<(PathProgram, u64, f64)> = None;
    let mut sweeps = Vec::new();
    loop {
        state.sweep()?;
        let code = state.program.code_bits()?;
        let data = state.data_bits();
        let total = code as f64 + data;
        let previous = kept.as_ref().map(|(_, c, d)| *c as f64 + *d);
        if previous.is_some_and(|previous| total > previous) {
            break;
        }
        sweeps.push((code, data));
        kept = Some((state.program.clone(), code, data));
        if previous.is_some_and(|previous| previous - total < 1.0) {
            break;
        }
    }
    let (program, code_bits, data_bits) = kept.expect("one sweep is always kept");
    Ok(PathFit { program, code_bits, data_bits, sweeps })
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::rngs::StdRng;
    use rand::{RngExt, SeedableRng};

    fn random_model(rng: &mut StdRng, n: usize, d: usize, widths: &[usize]) -> ResidualMlp {
        let mut matrix = |r: usize, c: usize, scale: f64| Array2::from_shape_fn((r, c), |_| scale * (rng.random::<f64>() * 2.0 - 1.0));
        let embed = matrix(n, d, 0.3);
        let unembed = embed.t().to_owned();
        let blocks = widths
            .iter()
            .map(|&w| MlpBlock { w_in: matrix(w, d, 0.3), w_out: matrix(d, w, 0.3) })
            .collect();
        ResidualMlp { embed, unembed, blocks }
    }

    fn sparse_inputs(rng: &mut StdRng, rows: usize, n: usize, p: f64) -> Vec<SparseRow> {
        (0..rows)
            .map(|_| (0..n).filter_map(|i| (rng.random::<f64>() < p).then(|| (i, rng.random::<f64>() * 2.0 - 1.0))).collect())
            .collect()
    }

    #[test]
    fn rewrite_and_gauge_preserve_the_function() {
        let mut rng = StdRng::seed_from_u64(7);
        let model = random_model(&mut rng, 12, 40, &[6, 5, 4]);
        let inputs = sparse_inputs(&mut rng, 200, 12, 0.2);
        let reference = model.forward(&inputs).unwrap();
        let mut program = PathProgram::from_model(&model).unwrap();
        let scale = reference.iter().fold(0.0_f64, |m, v| m.max(v.abs()));
        let gap = (&program.forward(&inputs) - &reference).iter().fold(0.0_f64, |m, v| m.max(v.abs()));
        assert!(gap <= 1e-12 * scale.max(1.0), "rewrite gap {gap}");
        program.canonical_gauge();
        let gap = (&program.forward(&inputs) - &reference).iter().fold(0.0_f64, |m, v| m.max(v.abs()));
        assert!(gap <= 1e-12 * scale.max(1.0), "gauge gap {gap}");
        for block in &program.blocks {
            for row in block.write.outer_iter() {
                assert!((row.dot(&row).sqrt() - 1.0).abs() < 1e-12);
            }
        }
    }

    #[test]
    fn inactive_components_contribute_nothing() {
        let mut rng = StdRng::seed_from_u64(3);
        let model = random_model(&mut rng, 10, 30, &[8, 8]);
        let program = PathProgram::from_model(&model).unwrap();
        let input: SparseRow = vec![(4, 0.7)];
        let trace = program.trace_row(&input);
        let mut ablated = program.clone();
        // Zero every source row that is inactive on this input.
        for l in 0..ablated.blocks.len() {
            for source in 0..ablated.source_offset(l) {
                let active = match program.source_neuron(source) {
                    Some((k, j)) => trace.act[k][j] > 0.0,
                    None => source == 4,
                };
                if !active {
                    ablated.blocks[l].read.row_mut(source).fill(0.0);
                }
            }
            for j in 0..ablated.blocks[l].width() {
                if trace.act[l][j] <= 0.0 {
                    ablated.blocks[l].write.row_mut(j).fill(0.0);
                }
            }
        }
        assert_eq!(ablated.trace_row(&input).out, trace.out);
    }

    #[test]
    fn fit_codes_every_row_and_shortens_with_less_behaviour() {
        let mut rng = StdRng::seed_from_u64(11);
        let model = random_model(&mut rng, 10, 30, &[6, 5]);
        let inputs = sparse_inputs(&mut rng, 400, 10, 0.15);
        let outputs = model.forward(&inputs).unwrap();
        let reference = PathProgram::from_model(&model).unwrap();
        let mut totals = Vec::new();
        for observations in [1e3, 1e7] {
            let behaviour = GaussianBehaviour { inputs: inputs.clone(), outputs: outputs.clone(), variance: 1e-3, observations };
            let fitted = fit(&reference, &behaviour).unwrap();
            assert_eq!(fitted.code_bits, fitted.program.code_bits().unwrap());
            let data = behaviour.data_bits(&fitted.program).unwrap();
            assert!((data - fitted.data_bits).abs() <= 1e-6 * data.max(1.0), "{data} vs {}", fitted.data_bits);
            totals.push((fitted.code_bits, fitted.program.real_count()));
        }
        // More behaviour buys a longer, finer program.
        assert!(totals[0].0 < totals[1].0, "{totals:?}");
        assert!(totals[0].1 <= totals[1].1, "{totals:?}");
    }
}
