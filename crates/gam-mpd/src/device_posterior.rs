//! The library fit's posterior resident on the device (#2951): the means `μ`, log standard
//! deviations `s`, the gradient's momentum, the curvature estimate and the gradient's second
//! moment (which measures the momentum's noise, [`Device::posterior_ivon`]) of every trainable operator
//! stay on the device from step to step, so a step moves no parameter between host and device.
//!
//! A step writes the weight sample `θ = μ + exp(s) ε` into the explanation's program
//! ([`Device::reparameterize`], `ε` regenerated from its counter), runs the experiments, and takes
//! the improved variational online Newton step (IVON, [`Device::posterior_ivon`]) with the data
//! term's curvature in the Gauss–Newton approximation, from the gradient the program's reverse
//! pass left on the device and a draw of the Gauss–Newton factor (`interchange::Factor`, a second
//! reverse pass from labels drawn from the explanation's own predictions), which also sums each
//! prior group's new moments; the groups' variances and divergences follow from those sums
//! ([`Device::group_divergence`]). The posterior, the curvature and the gradient's second moment are held in the fitting storage
//! (f32 on CUDA and the Apple GPU, float64 on the host); the momentum in bfloat16 where the masters
//! are f32 on CUDA (rounded once as it is stored, its update computed in f32), else in the fitting
//! storage; the group sums in float64 where the backend holds it (CUDA, the host). The objective is
//! `library_mdl`'s (module note there): `KL(q_G ‖ p_G) = ½ (|G| ln v_G − Σ 2s)` at the
//! empirical-Bayes variance `v_G`, whose prior precision per token `1 / (N v_G)` is IVON's weight
//! decay.
//!
//! The posterior's mean is the Polyak average `μ̄` of IVON's iterate `μ` over about one epoch
//! (`μ̄ ← μ̄ + w (μ − μ̄)`, `w = max(1/t, 1 − β₂)`, `t` the steps since the posterior was last set,
//! `1 / (1 − β₂)` the epoch's batches): the steps move the iterate, sampled at `μ + σ ε`, while
//! `v_G`, `KL(q_G ‖ p_G)`, the reported and checkpointed posterior and every evaluation take `μ̄`.
//! The iterate's optimizer noise is not part of `q`: where the data's curvature is small,
//! `σ² ≈ v_G`, and charging the iterate's random walk within `±σ` to `v_G = mean(μ² + σ²)` grew
//! `v_G` without bound (vpd4l at `N = 2^16`: `Σ v_G` 356 → 1.4e4 over 12 epochs).

use crate::{
    device_program::DeviceProgram,
    library_mdl::{Curvature, Explanation, Posterior, Rotation, Side},
};
use gam_gpu::{
    gpu_error::GpuError,
    tensor::{Arithmetic, Device, GroupMap, Op, PosteriorStep, Storage, Tensor},
};
use ndarray::Array2;
use std::collections::BTreeMap;

fn error(e: impl std::fmt::Display) -> String {
    format!("device posterior: {e}")
}

/// IVON's settings of a step ([`DevicePosterior::step`], [`Device::posterior_ivon`]): the mean's
/// step `α` as a fraction of the Newton step, the gradient momentum's decay `β₁` and the curvature
/// estimate's decay `β₂`.
#[derive(Clone, Copy, Debug)]
pub struct Ivon {
    pub rate: f64,
    pub beta1: f64,
    pub beta2: f64,
}

/// A posterior's host arrays ([`DevicePosterior::from_parts`]).
pub struct Parts<'a> {
    pub operators: &'a [usize],
    pub mean: &'a [Array2<f64>],
    pub log_sd: &'a [Array2<f64>],
    pub groups: &'a [Vec<u32>],
    pub count: usize,
    /// Per operator the rotation of its noise, if any (`library_mdl::Rotation`): its posterior
    /// lives along the rotated axes on the device.
    pub rotations: &'a [Option<Rotation>],
}

/// The per-group constants of a posterior's code length and one row per step for its values
/// ([`DevicePosterior::code_length`]).
pub struct CodeLength {
    weight: Tensor,
    constant: Tensor,
    initial: Tensor,
    rows: Tensor,
}

/// The posterior of a library explanation's trainable operators, on the device.
pub struct DevicePosterior {
    /// The device holding the group sums (float64 where the backend holds it), and the one holding
    /// the posterior, its samples and gradients.
    wide: Device,
    fitting: Device,
    /// Per trainable operator (`Explanation::trainable` order): its id, `μ`, `s`, IVON's state
    /// (the gradient's momentum, the curvature estimate, the gradient's second moment) and its
    /// entries' groups (one id per
    /// row or column where the groups are rows or columns).
    operators: Vec<usize>,
    mean: Vec<Tensor>,
    log_sd: Vec<Tensor>,
    moments: Vec<[Tensor; 3]>,
    groups: Vec<GroupMap>,
    /// Per operator the posterior's mean `μ̄`, the Polyak average of the iterate `mean` over the
    /// `averaged` steps since the posterior was last set, up to one epoch (module note).
    average: Vec<Tensor>,
    averaged: u64,
    /// The A/B arms of the 2^16 comparison (`PosteriorStep::trust`, `PosteriorStep::split`, and
    /// steps taken at the iterate itself, no weight noise); the A/B's outcome deletes them.
    trust: f64,
    split: bool,
    deterministic: bool,
    /// The line arm: the iterate moves to the Gauss–Newton minimum of `F` along IVON's full
    /// direction (`DevicePosterior::line_step`), with the joint curvature's ratio to the diagonal
    /// one averaged over `ratio_steps` steps (up to one epoch).
    line: bool,
    ratio: f64,
    ratio_steps: u64,
    /// Per group `(n, Σ μ² + σ², Σ 2s)` being summed, its variance and its divergence in nats.
    sums: Tensor,
    variance: Tensor,
    divergence: Tensor,
    /// The training tokens `N` (the data term's weight) and the steps taken.
    tokens: f64,
    steps: u64,
    /// Per operator its rotation (on the host), and on the device each distinct rotation matrix and
    /// per operator its side and matrix: the operator's `μ`, gradient and IVON state are held along
    /// the rotated axes, so IVON's step stays entrywise there.
    rotations: Vec<Option<Rotation>>,
    matrices: Vec<Tensor>,
    placed: Vec<Option<(Side, usize)>>,
    /// Per operator, the host `μ` and `s` its device values were last set from
    /// ([`DevicePosterior::set_values`]) while no step or restore has changed them since: a later
    /// `set_values` sends only the operators whose values differ in some bit (a removal trial
    /// changes a few operators of many).
    uploaded: Vec<Option<(Array2<f64>, Array2<f64>)>>,
}

/// Whether two host arrays hold the same values bit for bit (`-0.0` differs from `0.0`).
fn same_bits(a: &Array2<f64>, b: &Array2<f64>) -> bool {
    a.dim() == b.dim() && a.iter().zip(b.iter()).all(|(x, y)| x.to_bits() == y.to_bits())
}

/// Each trainable operator's entries' groups, row-major.
fn membership(explanation: &Explanation, shapes: &[(usize, usize)]) -> Result<Vec<Vec<u32>>, String> {
    let position: BTreeMap<usize, usize> = explanation.trainable.iter().enumerate().map(|(i, op)| (*op, i)).collect();
    let mut out: Vec<Vec<u32>> = shapes.iter().map(|(r, c)| vec![u32::MAX; r * c]).collect();
    let count = u32::try_from(explanation.groups.len()).map_err(|_| error("too many prior groups"))?;
    for (g, group) in (0..count).zip(&explanation.groups) {
        for cell in &group.cells {
            let i = *position.get(&cell.operator).ok_or_else(|| error(format!("{}: operator {} is not trainable", group.name, cell.operator)))?;
            let (rows, cols) = shapes[i];
            for &row in &cell.rows {
                for col in cell.cols.clone() {
                    if row >= rows || col >= cols {
                        return Err(error(format!("{}: entry ({row}, {col}) outside its operator", group.name)));
                    }
                    out[i][row * cols + col] = g;
                }
            }
        }
    }
    if out.iter().flatten().any(|g| *g == u32::MAX) {
        return Err(error("a trainable entry is in no prior group"));
    }
    Ok(out)
}

impl DevicePosterior {
    /// `posterior` of `explanation` for `tokens` training tokens on `fitting` (the device whose
    /// storage the explanation's program runs in), with IVON's state per operator (`moments`: the
    /// gradient's momentum, the curvature estimate and the gradient's second moment; when `None`,
    /// the moments zero and the curvature at which the posterior's standard deviations are IVON's)
    /// after `steps` steps.
    pub fn new(fitting: &Device, explanation: &Explanation, posterior: &Posterior, tokens: f64, moments: Option<&[[Array2<f64>; 3]]>, steps: u64) -> Result<Self, String> {
        let shapes: Vec<(usize, usize)> = posterior.mean.iter().map(Array2::dim).collect();
        if shapes.len() != explanation.trainable.len() {
            return Err(error("one posterior array per trainable operator required"));
        }
        let groups = membership(explanation, &shapes)?;
        let parts = Parts { operators: &explanation.trainable, mean: &posterior.mean, log_sd: &posterior.log_sd, groups: &groups, count: explanation.groups.len(), rotations: &posterior.rotations };
        Self::from_parts(fitting, &parts, tokens, moments, steps)
    }

    /// The posterior of the trainable operators `parts.operators` of a program, each entry in group
    /// `parts.groups[i][entry]` (row-major) of `parts.count`, for `tokens` training tokens on
    /// `fitting`, with IVON's state `moments` (see [`DevicePosterior::new`]) after `steps` steps.
    pub fn from_parts(fitting: &Device, parts: &Parts<'_>, tokens: f64, moments: Option<&[[Array2<f64>; 3]]>, steps: u64) -> Result<Self, String> {
        let wide = match fitting.with_storage(Storage::F64) {
            Ok(wide) => wide,
            Err(GpuError::NoDeviceKernel { .. }) => fitting.clone(),
            Err(e) => return Err(error(e)),
        };
        let master = fitting.clone();
        // The momentum: bfloat16 beside f32 masters where the backend stores it (CUDA).
        let narrow = if fitting.storage() == Storage::F32 {
            match fitting.with_storage(Storage::Bf16) {
                Ok(narrow) => narrow,
                Err(GpuError::NoDeviceKernel { .. }) => fitting.clone(),
                Err(e) => return Err(error(e)),
            }
        } else {
            fitting.clone()
        };
        let shapes: Vec<(usize, usize)> = parts.mean.iter().map(Array2::dim).collect();
        let sizes_agree = parts.log_sd.iter().map(Array2::dim).eq(shapes.iter().copied()) && parts.groups.iter().map(Vec::len).eq(shapes.iter().map(|(r, c)| r * c));
        if shapes.len() != parts.operators.len() || !sizes_agree || moments.is_some_and(|m| m.len() != shapes.len()) || parts.rotations.len() != shapes.len() {
            return Err(error("one posterior array, group list and rotation per trainable operator required"));
        }
        if parts.groups.iter().flatten().any(|g| *g as usize >= parts.count) {
            return Err(error("a group id beyond the groups"));
        }
        if !(tokens.is_finite() && tokens > 0.0) {
            return Err(error("positive training tokens required"));
        }
        // Without a state, the curvature `h = 1 / (N σ²) − δ` at which IVON's standard deviation
        // `1 / √(N (h + δ))` is the posterior's, `δ = 1 / (N v_G)` at each group's variance (zero
        // where the posterior is wider than the prior: `h` is nonnegative, `Device::posterior_ivon`).
        let start = match moments {
            Some(_) => None,
            None => {
                let mut sums = vec![(0.0, 0.0); parts.count];
                for ((mean, log_sd), groups) in parts.mean.iter().zip(parts.log_sd).zip(parts.groups) {
                    for ((mu, s), g) in mean.iter().zip(log_sd.iter()).zip(groups) {
                        if *s != f64::NEG_INFINITY {
                            sums[*g as usize].0 += 1.0;
                            sums[*g as usize].1 += mu * mu + (2.0 * s).exp();
                        }
                    }
                }
                let curvature = |log_sd: &Array2<f64>, groups: &[u32]| -> Array2<f64> {
                    let (rows, cols) = log_sd.dim();
                    Array2::from_shape_fn((rows, cols), |(r, c)| {
                        let (s, (n, second)) = (log_sd[[r, c]], sums[groups[r * cols + c] as usize]);
                        if s == f64::NEG_INFINITY { 0.0 } else { (1.0 / (tokens * (2.0 * s).exp()) - n / (tokens * second)).max(0.0) }
                    })
                };
                Some(parts.log_sd.iter().zip(parts.groups).map(|(log_sd, groups)| [Array2::zeros(log_sd.dim()), curvature(log_sd, groups), Array2::zeros(log_sd.dim())]).collect::<Vec<_>>())
            }
        };
        let moments = moments.or(start.as_deref());
        let up = |m: &Array2<f64>| master.upload(m.view()).map_err(error);
        let moment = |m: &Array2<f64>| narrow.upload(m.view()).map_err(error);
        let mut out = Self {
            sums: wide.zeros(parts.count, 3).map_err(error)?,
            variance: wide.zeros(parts.count, 1).map_err(error)?,
            divergence: wide.zeros(parts.count, 1).map_err(error)?,
            mean: parts.mean.iter().zip(parts.rotations).map(|(m, r)| up(&r.as_ref().map_or_else(|| m.clone(), |r| r.undo(m)))).collect::<Result<_, _>>()?,
            log_sd: parts.log_sd.iter().map(up).collect::<Result<_, _>>()?,
            moments: moments.ok_or_else(|| error("no posterior state"))?.iter().map(|m| Ok([moment(&m[0])?, up(&m[1])?, up(&m[2])?])).collect::<Result<_, String>>()?,
            groups: parts.groups.iter().zip(&shapes).map(|(ids, shape)| master.group_map(ids, *shape).map_err(error)).collect::<Result<_, _>>()?,
            average: Vec::new(),
            averaged: 0,
            trust: 1.0,
            split: false,
            deterministic: false,
            line: false,
            ratio: 1.0,
            ratio_steps: 0,
            operators: parts.operators.to_vec(),
            fitting: fitting.clone(),
            wide,
            tokens,
            steps,
            rotations: parts.rotations.to_vec(),
            matrices: Vec::new(),
            placed: Vec::new(),
            uploaded: Vec::new(),
        };
        let (placed, matrices) = crate::library_mdl::rotation_layout(parts.rotations);
        out.matrices = matrices.iter().map(|m| up(m)).collect::<Result<_, _>>()?;
        out.placed = placed;
        out.average = out.mean.iter().map(|m| out.fitting.copy(m).map_err(error)).collect::<Result<_, _>>()?;
        out.refresh()?;
        Ok(out)
    }

    /// Sets the A/B arm: the mean's move clamped to `±trust σ`, the data momentum filtered alone
    /// (`split`), and steps at the iterate without weight noise (`deterministic`).
    pub fn set_arm(&mut self, trust: f64, split: bool, deterministic: bool, line: bool) {
        (self.trust, self.split, self.deterministic, self.line) = (trust, split, deterministic, line);
    }

    /// The groups' variances and divergences from the posterior as it stands, at its mean `μ̄`.
    fn refresh(&mut self) -> Result<(), String> {
        for ((mean, log_sd), groups) in self.average.iter().zip(&self.log_sd).zip(&self.groups) {
            self.fitting.group_moments((mean, log_sd), groups, &mut self.sums).map_err(error)?;
        }
        self.wide.group_divergence(&mut self.sums, &mut self.variance, &mut self.divergence).map_err(error)
    }

    /// The iterate set to the posterior's mean, the average restarted there, and the groups'
    /// variances and divergences from it.
    fn restart(&mut self) -> Result<(), String> {
        self.average = self.mean.iter().map(|m| self.fitting.copy(m).map_err(error)).collect::<Result<_, _>>()?;
        self.averaged = 0;
        self.refresh()
    }

    /// The products' arithmetic in the fitting storage.
    fn arithmetic(&self) -> Arithmetic {
        if self.fitting.storage() == Storage::F64 { Arithmetic::F64 } else { Arithmetic::F32 }
    }

    /// Operator `i`'s array `x` along its rotated axes, along its own axes (`None` without a
    /// rotation).
    fn applied(&self, i: usize, x: &Tensor) -> Result<Option<Tensor>, String> {
        let Some((side, at)) = self.placed[i] else { return Ok(None) };
        let r = &self.matrices[at];
        let mut out = self.fitting.zeros(x.rows(), x.cols()).map_err(error)?;
        match side {
            Side::Columns => self.fitting.gemm(&mut out, 1.0, x, Op::N, r, Op::T, 0.0, self.arithmetic()),
            Side::Rows => self.fitting.gemm(&mut out, 1.0, r, Op::N, x, Op::N, 0.0, self.arithmetic()),
        }
        .map_err(error)?;
        Ok(Some(out))
    }

    /// Operator `i`'s array `x` along its own axes (a gradient), along its rotated axes (`None`
    /// without a rotation).
    fn undone(&self, i: usize, x: &Tensor) -> Result<Option<Tensor>, String> {
        let Some((side, at)) = self.placed[i] else { return Ok(None) };
        let r = &self.matrices[at];
        let mut out = self.fitting.zeros(x.rows(), x.cols()).map_err(error)?;
        match side {
            Side::Columns => self.fitting.gemm(&mut out, 1.0, x, Op::N, r, Op::N, 0.0, self.arithmetic()),
            Side::Rows => self.fitting.gemm(&mut out, 1.0, r, Op::T, x, Op::N, 0.0, self.arithmetic()),
        }
        .map_err(error)?;
        Ok(Some(out))
    }

    /// Operator `i`'s posterior means `μ̄` along its own axes, on the host.
    fn host_mean(&self, i: usize) -> Result<Array2<f64>, String> {
        let mean = self.fitting.download(&self.average[i]).map_err(error)?;
        Ok(match &self.rotations[i] {
            Some(r) => r.apply(&mean),
            None => mean,
        })
    }

    /// Per operator, the rotation of its noise.
    #[must_use]
    pub fn rotations(&self) -> &[Option<Rotation>] {
        &self.rotations
    }

    /// Trainable operator `op`'s array `x` along its own axes (a gradient or a Gauss–Newton
    /// factor), along its rotated axes; `None` without a rotation.
    pub fn along_rotated_axes(&self, op: usize, x: &Tensor) -> Result<Option<Tensor>, String> {
        let (i, _, _) = self.entries(op)?;
        self.undone(i, x)
    }

    /// The line arm's step from the iterate `before`: IVON's full direction `d = ĝ / (h + δ)` (the
    /// step the kernel just took at rate 1 without a clamp, `ĝ` the filtered full gradient), and
    /// the iterate moved to `before − η d` with `η` the minimum along `d` of `F`'s Gauss–Newton
    /// model: slope `ĝ · d = Σ (h + δ) d²` over curvature `d^T (G + Δ) d`, `Δ` the prior precisions
    /// `δ`. The data curvature along `d` is `ρ Σ h d²`, `ρ` the average over the steps (up to one
    /// epoch, `w = max(1/t, 1 − β₂)`) of one draw's `c (u · d)² / Σ h d²` (`u` the step's draws of
    /// the Gauss–Newton factor `draws`, `c` the factor turning its square into curvature per token,
    /// so `E[c (u · d)²] = d^T G d` including the entries' correlations that the diagonal `h` omits).
    fn line_step(&mut self, before: &[Tensor], (sums, directions): (&[Tensor], &[Tensor]), square: f64, beta2: f64) -> Result<(), String> {
        let variances = self.variances()?;
        let column = |t: &Tensor| -> Result<Vec<f64>, String> { Ok(self.wide.download(t).map_err(error)?.column(1).to_vec()) };
        let (along, square_sums, diagonal) = (column(&sums[0])?.iter().sum::<f64>(), column(&sums[1])?, column(&sums[2])?.iter().sum::<f64>());
        let prior: f64 = square_sums.iter().zip(&variances).filter(|(_, v)| **v > 0.0).map(|(dd, v)| dd / (self.tokens * v)).sum();
        if diagonal > 0.0 {
            self.ratio_steps += 1;
            let w = (1.0 / self.ratio_steps as f64).max(1.0 - beta2);
            self.ratio += w * (square * along * along / diagonal - self.ratio);
        }
        let (slope, curvature) = (diagonal + prior, self.ratio * diagonal + prior);
        let eta = if curvature > 0.0 && slope.is_finite() { slope / curvature } else { 0.0 };
        log::debug!("line step {}: η {eta:.4e}, ρ {:.4e}", self.steps, self.ratio);
        for ((mean, start), d) in self.mean.iter_mut().zip(before).zip(directions) {
            *mean = self.fitting.copy(start).map_err(error)?;
            self.fitting.axpy(mean, -eta, d).map_err(error)?;
        }
        Ok(())
    }

    /// Operator `i`'s direction `d = before − μ` after the kernel's full step, with its terms added
    /// into `sums` (per group `u · d`, `Σ d²` and `Σ h d²` in column 1; `DevicePosterior::line_step`).
    fn line_terms(&self, i: usize, before: &Tensor, draw: &Tensor, sums: &mut [Tensor]) -> Result<Tensor, String> {
        let mut d = self.fitting.copy(before).map_err(error)?;
        self.fitting.axpy(&mut d, -1.0, &self.mean[i]).map_err(error)?;
        let mut weighted = self.fitting.empty(d.rows(), d.cols()).map_err(error)?;
        self.fitting.hadamard(&mut weighted, &self.moments[i][1], &d, false).map_err(error)?;
        let [along, square_sums, diagonal] = sums else { return Err(error("three sums")) };
        self.fitting.group_curvature((draw, &d, &self.log_sd[i]), &self.groups[i], along).map_err(error)?;
        self.fitting.group_curvature((&d, &d, &self.log_sd[i]), &self.groups[i], square_sums).map_err(error)?;
        self.fitting.group_curvature((&weighted, &d, &self.log_sd[i]), &self.groups[i], diagonal).map_err(error)?;
        Ok(d)
    }

    /// The steps taken.
    #[must_use]
    pub fn steps(&self) -> u64 {
        self.steps
    }

    /// Writes the posterior means into `program`'s trainable operators (rounded to its storage).
    /// Trainable operator `op`'s position, and its means and log standard deviations.
    fn entries(&self, op: usize) -> Result<(usize, &Tensor, &Tensor), String> {
        let i = self.operators.iter().position(|o| *o == op).ok_or_else(|| error(format!("operator {op} is not trainable")))?;
        Ok((i, &self.average[i], &self.log_sd[i]))
    }

    /// Trainable operator `op`'s weight sample of `key` (the draws [`DevicePosterior::sample_into`]
    /// writes) into the block of `out` at `(row, col)`, a stacked operand such as a fused group's:
    /// written in place, no copy of the operator ([`Device::reparameterize_block`]).
    /// An operator with a rotation is sampled along its rotated axes, turned to its own, and written
    /// whole.
    pub fn sample_block(&self, op: usize, out: &mut Tensor, at: (usize, usize), key: u64) -> Result<(), String> {
        self.block_of(op, out, at, key, &self.average)
    }

    /// [`DevicePosterior::sample_block`] around `means` (the posterior's `μ̄` or the iterate).
    fn block_of(&self, op: usize, out: &mut Tensor, at: (usize, usize), key: u64, means: &[Tensor]) -> Result<(), String> {
        let (i, _, log_sd) = self.entries(op)?;
        let mean = &means[i];
        if self.placed[i].is_none() {
            return self.fitting.reparameterize_block(out, at, (mean, log_sd), (key, i as u64)).map_err(error);
        }
        let mut rotated = self.fitting.empty(mean.rows(), mean.cols()).map_err(error)?;
        self.fitting.reparameterize(&mut rotated, (mean, log_sd), (key, i as u64)).map_err(error)?;
        let theta = self.applied(i, &rotated)?.ok_or_else(|| error("a rotation lost"))?;
        self.write_block(out, at, theta)
    }

    /// `value` (in the fitting storage) written into the block of `out` at `(row, col)`, in `out`'s
    /// storage.
    fn write_block(&self, out: &mut Tensor, (row, col): (usize, usize), value: Tensor) -> Result<(), String> {
        let narrow = |value: Tensor| if out.storage() == Storage::Bf16 { self.fitting.bf16_copy(&value).map_err(error) } else { Ok(value) };
        if col == 0 && value.cols() == out.cols() {
            let value = narrow(value)?;
            self.fitting.set_rows(out, row, &value).map_err(error)
        } else if row == 0 && value.rows() == out.rows() {
            let value = narrow(value)?;
            self.fitting.set_columns(out, col, &value).map_err(error)
        } else {
            // A block of deviation zero is its mean (`exp(−∞) = 0`).
            let row_of = self.fitting.upload_vec(1, value.cols(), vec![f64::NEG_INFINITY; value.cols()]).map_err(error)?;
            let fixed = self.fitting.broadcast_rows(&row_of, value.rows()).map_err(error)?;
            self.fitting.reparameterize_block(out, (row, col), (&value, &fixed), (0, 0)).map_err(error)
        }
    }

    /// Writes the posterior means into `program`'s trainable operators and their stacks.
    pub fn mean_into(&self, program: &mut DeviceProgram) -> Result<(), String> {
        self.means_into(program, &self.average)
    }

    /// Writes `means` (the posterior's `μ̄` or the iterate) into `program`'s trainable operators
    /// and their stacks.
    fn means_into(&self, program: &mut DeviceProgram, means: &[Tensor]) -> Result<(), String> {
        for (i, &op) in self.operators.iter().enumerate() {
            // A program holding the operator in bfloat16 (`DeviceProgram::hold_bf16`) gets it so.
            let bf16 = program.dense(op).is_ok_and(|held| held.storage() == Storage::Bf16);
            let rotated = self.applied(i, &means[i])?;
            let mean = rotated.as_ref().unwrap_or(&means[i]);
            let value = if bf16 { self.fitting.bf16_copy(mean) } else { self.fitting.copy(mean) };
            program.replace_dense_parameter(op, value.map_err(error)?)?;
        }
        program.refresh_fused()
    }

    /// Writes the rounded posterior means (each `μ` to the nearest multiple of `2^⌊log2 σ⌋`,
    /// `Posterior::rounded`) into `program`'s trainable operators (rounded to its storage). An
    /// operator with a rotation is rounded along its own axes, each mean to its marginal
    /// deviation's grid: the decoder reads the operator's own values.
    pub fn rounded_into(&self, program: &mut DeviceProgram) -> Result<(), String> {
        for (i, &op) in self.operators.iter().enumerate() {
            let mut rounded = self.fitting.empty(self.average[i].rows(), self.average[i].cols()).map_err(error)?;
            match &self.rotations[i] {
                Some(rotation) => {
                    let mean = self.applied(i, &self.average[i])?.ok_or_else(|| error("a rotation lost"))?;
                    let variances = rotation.marginal(&self.fitting.download(&self.log_sd[i]).map_err(error)?.mapv(|s| (2.0 * s).exp()));
                    let log_sd = self.fitting.upload(variances.mapv(|v| 0.5 * v.ln()).view()).map_err(error)?;
                    self.fitting.round_to_deviation(&mut rounded, (&mean, &log_sd)).map_err(error)?;
                }
                None => self.fitting.round_to_deviation(&mut rounded, (&self.average[i], &self.log_sd[i])).map_err(error)?,
            }
            let bf16 = program.dense(op).is_ok_and(|held| held.storage() == Storage::Bf16);
            let value = if bf16 { self.fitting.bf16_copy(&rounded).map_err(error)? } else { rounded };
            program.replace_dense_parameter(op, value)?;
        }
        program.refresh_fused()
    }

    /// Writes the weight sample of `key` into `program`'s trainable operators (operator `i` in
    /// `Explanation::trainable` order draws stream `i`); in place where the program holds the
    /// operator in one role, replaced where it also holds a column copy (a bias). A fused group's
    /// stack gets the same draws written straight into its blocks
    /// (`DeviceProgram::refresh_fused_with`), not restacked from the operators.
    pub fn sample_into(&self, program: &mut DeviceProgram, key: u64) -> Result<(), String> {
        self.sample_of(program, key, &self.average)
    }

    /// The point a step's gradient is taken at, into `program`: the iterate's weight sample of
    /// `key` (`μ + σ ε`, the draws of [`DevicePosterior::sample_into`] around the iterate rather
    /// than `μ̄`), or the iterate itself in the deterministic arm.
    pub fn iterate_into(&self, program: &mut DeviceProgram, key: u64) -> Result<(), String> {
        if self.deterministic { self.means_into(program, &self.mean) } else { self.sample_of(program, key, &self.mean) }
    }

    /// [`DevicePosterior::sample_into`] around `means`.
    fn sample_of(&self, program: &mut DeviceProgram, key: u64, means: &[Tensor]) -> Result<(), String> {
        for (i, &op) in self.operators.iter().enumerate() {
            let parts = (&means[i], &self.log_sd[i]);
            if self.placed[i].is_some() {
                // The sample along the rotated axes, then along the operator's own.
                let mut rotated = self.fitting.zeros(means[i].rows(), means[i].cols()).map_err(error)?;
                self.fitting.reparameterize(&mut rotated, parts, (key, i as u64)).map_err(error)?;
                let theta = self.applied(i, &rotated)?.ok_or_else(|| error("a rotation lost"))?;
                match program.dense_mut(op) {
                    Ok(held) => *held = theta,
                    Err(_) => program.replace_dense_parameter(op, theta)?,
                }
            } else if let Ok(theta) = program.dense_mut(op) {
                self.fitting.reparameterize(theta, parts, (key, i as u64)).map_err(error)?;
            } else {
                let (rows, cols) = (means[i].rows(), means[i].cols());
                let mut theta = self.fitting.zeros(rows, cols).map_err(error)?;
                self.fitting.reparameterize(&mut theta, parts, (key, i as u64)).map_err(error)?;
                program.replace_dense_parameter(op, theta)?;
            }
        }
        program.refresh_fused_with(&mut |op, stack, at| {
            if !self.operators.contains(&op) {
                return Ok(false);
            }
            self.block_of(op, stack, at, key, means)?;
            Ok(true)
        })
    }

    /// One IVON step: `gradients` holds per trainable operator (by id) the gradient of the batch's
    /// data term at the sample, which `scale` turns into an unbiased estimate of the collection's
    /// gradient per token in nats (`B / N` for one of `B` batches of a collection of `N` scored
    /// tokens, times the conversion from bits); `factor` holds per operator a draw of the
    /// Gauss–Newton factor and the factor (`B / N`) turning its square into the curvature estimate
    /// per token. An operator the batch does not reach has neither, and its step takes the
    /// prior's alone.
    pub fn step(&mut self, gradients: &BTreeMap<usize, Tensor>, scale: f64, factor: (&BTreeMap<usize, Tensor>, f64), ivon: &Ivon) -> Result<(), String> {
        self.uploaded.clear();
        self.steps += 1;
        // The line arm takes IVON's full direction (rate 1, no clamp) from the iterate kept here.
        let before = if self.line { Some(self.mean.iter().map(|m| self.fitting.copy(m).map_err(error)).collect::<Result<Vec<_>, _>>()?) } else { None };
        let (rate, trust) = if self.line { (1.0, f64::INFINITY) } else { (ivon.rate, self.trust) };
        let mut line = if self.line { Some(((0..3).map(|_| self.wide.zeros(self.group_count(), 3).map_err(error)).collect::<Result<Vec<_>, _>>()?, Vec::new())) } else { None };
        for (i, &op) in self.operators.iter().enumerate() {
            let zero = |given: Option<&Tensor>| -> Result<Option<Tensor>, String> {
                match given {
                    Some(_) => Ok(None),
                    None => self.fitting.zeros(self.mean[i].rows(), self.mean[i].cols()).map(Some).map_err(error),
                }
            };
            let (missing_gradient, missing_factor) = (zero(gradients.get(&op))?, zero(factor.0.get(&op))?);
            let gradient = gradients.get(&op).or(missing_gradient.as_ref()).ok_or_else(|| error("no gradient"))?;
            let draw = factor.0.get(&op).or(missing_factor.as_ref()).ok_or_else(|| error("no Gauss–Newton factor"))?;
            // Along the rotated axes, where the posterior is held.
            let (rotated_gradient, rotated_draw) = (self.undone(i, gradient)?, self.undone(i, draw)?);
            let (gradient, draw) = (rotated_gradient.as_ref().unwrap_or(gradient), rotated_draw.as_ref().unwrap_or(draw));
            let step = PosteriorStep { gradient_scale: scale, factor_scale: factor.1, tokens: self.tokens, rate, beta1: ivon.beta1, beta2: ivon.beta2, step: self.steps, trust, split: self.split };
            let [momentum, curvature, power] = &mut self.moments[i];
            self.fitting
                .posterior_ivon((&mut self.mean[i], &mut self.log_sd[i]), [momentum, curvature, power], (gradient, draw), (&self.groups[i], &self.variance), &mut self.sums, &step)
                .map_err(error)?;
            if let (Some(before), Some((sums, directions))) = (before.as_ref(), line.as_mut()) {
                directions.push(self.line_terms(i, &before[i], draw, sums)?);
            }
        }
        if let (Some(before), Some((sums, directions))) = (before, line) {
            self.line_step(&before, (&sums[..], &directions[..]), factor.1, ivon.beta2)?;
        }
        // The posterior's mean: the iterate's Polyak average, uniform over the steps since the
        // posterior was set and then over about one epoch (module note).
        self.averaged += 1;
        let weight = (1.0 / self.averaged as f64).max(1.0 - ivon.beta2);
        for (average, mean) in self.average.iter_mut().zip(&self.mean) {
            let mut difference = self.fitting.copy(mean).map_err(error)?;
            self.fitting.axpy(&mut difference, -1.0, average).map_err(error)?;
            self.fitting.axpy(average, weight, &difference).map_err(error)?;
        }
        // The groups' variances and divergences at `μ̄`, not at the iterate the step summed.
        self.sums = self.wide.zeros(self.sums.rows(), 3).map_err(error)?;
        self.refresh()
    }

    /// The number of prior groups.
    #[must_use]
    pub fn group_count(&self) -> usize {
        self.variance.rows()
    }

    /// Adds one batch to `curvature` (`library_mdl`'s module note): with `θ` the weight sample
    /// [`DevicePosterior::sample_into`] draws under `key`, `g` the batch's data gradient at `θ` (per
    /// trainable operator by id, on the device), times `nats`, and `u` a draw of the Gauss–Newton
    /// factor at `θ`, per group `g_G · θ_G` and `(u_G · θ_G)²` over the group's live entries,
    /// summed on the device and read as one row per group; an operator neither reaches adds
    /// nothing.
    pub fn add_removal(&self, (g, nats): (&BTreeMap<usize, Tensor>, f64), u: &BTreeMap<usize, Tensor>, key: u64, curvature: &mut Curvature) -> Result<(), String> {
        let groups = self.group_count();
        let (mut slopes, mut forms) = (self.wide.zeros(groups, 3).map_err(error)?, self.wide.zeros(groups, 3).map_err(error)?);
        for (i, op) in self.operators.iter().enumerate() {
            let (gradient, draw) = (g.get(op), u.get(op));
            if gradient.is_none() && draw.is_none() {
                continue;
            }
            // The sample along the rotated axes, where the means and deviations are, as
            // `sample_into` draws it; `x · θ` is the same along either axes.
            let mut theta = self.fitting.zeros(self.mean[i].rows(), self.mean[i].cols()).map_err(error)?;
            self.fitting.reparameterize(&mut theta, (&self.average[i], &self.log_sd[i]), (key, i as u64)).map_err(error)?;
            for (x, sums) in [(gradient, &mut slopes), (draw, &mut forms)] {
                let Some(x) = x else { continue };
                let rotated = self.undone(i, x)?;
                self.fitting.group_curvature((rotated.as_ref().unwrap_or(x), &theta, &self.log_sd[i]), &self.groups[i], sums).map_err(error)?;
            }
        }
        let slope: Vec<f64> = self.wide.download(&slopes).map_err(error)?.column(1).iter().map(|s| nats * s).collect();
        let form: Vec<f64> = self.wide.download(&forms).map_err(error)?.column(1).iter().map(|d| d * d).collect();
        curvature.add_batch(&slope, &form)
    }

    /// Rows for `steps` steps' code lengths ([`DevicePosterior::code_length_into`]) of a posterior
    /// whose groups are `active`, of `sizes` entries, their variances' scales sent against the
    /// starting variances `initial`.
    pub fn code_length(&self, active: &[bool], sizes: &[f64], initial: &[f64], steps: usize) -> Result<CodeLength, String> {
        let column = |values: Vec<f64>| self.wide.upload_vec(values.len(), 1, values).map_err(error);
        Ok(CodeLength {
            weight: column(active.iter().map(|a| f64::from(u8::from(*a))).collect())?,
            constant: column(sizes.iter().map(|n| 0.5 * n.ln()).collect())?,
            initial: column(initial.to_vec())?,
            rows: self.wide.zeros(steps, 3).map_err(error)?,
        })
    }

    /// Adds the code length in nats of the groups' posteriors as they stand into row `step` of
    /// `code`: over the active groups `G`, `KL(q_G ‖ p_G) + ½ ln |G|` and `ln 2` times the bits of
    /// `v_G`'s scale against `v⁰_G` (`library_mdl`'s description without its subset code), summed
    /// on the device ([`Device::group_code_length`]).
    pub fn code_length_into(&self, code: &mut CodeLength, step: usize) -> Result<(), String> {
        self.wide.group_code_length((&self.divergence, &self.variance), (&code.weight, &code.constant, &code.initial), &mut code.rows, step).map_err(error)
    }

    /// Each step's code length in nats, read at once.
    pub fn code_lengths(&self, code: &CodeLength) -> Result<Vec<f64>, String> {
        Ok(self.wide.download(&code.rows).map_err(error)?.column(1).to_vec())
    }

    /// Per group, its empirical-Bayes variance `v_G` at the posterior as it stands.
    pub fn variances(&self) -> Result<Vec<f64>, String> {
        Ok(self.wide.download(&self.variance).map_err(error)?.into_iter().collect())
    }

    /// Per group, `KL(q_G ‖ p_G)` in nats at the posterior as it stands (zero for a removed group).
    pub fn divergences(&self) -> Result<Vec<f64>, String> {
        Ok(self.wide.download(&self.divergence).map_err(error)?.into_iter().collect())
    }

    /// The posterior's means and log standard deviations into `posterior`, and IVON's state per
    /// operator.
    pub fn download(&self, posterior: &mut Posterior) -> Result<Vec<[Array2<f64>; 3]>, String> {
        self.values_into(posterior)?;
        let down = |t: &Tensor| self.fitting.download(t).map_err(error);
        self.moments.iter().map(|m| Ok([down(&m[0])?, down(&m[1])?, down(&m[2])?])).collect()
    }

    /// The posterior's means and log standard deviations into `posterior`; IVON's state stays on
    /// the device.
    pub fn values_into(&self, posterior: &mut Posterior) -> Result<(), String> {
        for i in 0..self.mean.len() {
            posterior.mean[i] = self.host_mean(i)?;
            posterior.log_sd[i] = self.fitting.download(&self.log_sd[i]).map_err(error)?;
        }
        Ok(())
    }

    /// `posterior`'s means and log standard deviations onto the device, IVON's state kept, and
    /// the groups' variances and divergences with them (after a removal on the host, whose removed
    /// entries, `μ = 0` and `s = −∞`, every later step leaves alone). An operator whose values
    /// are bit for bit those it was last set from, with no step since, is not sent again.
    pub fn set_values(&mut self, posterior: &Posterior) -> Result<(), String> {
        if posterior.mean.len() != self.mean.len() {
            return Err(error("one posterior array per trainable operator required"));
        }
        self.uploaded.resize_with(self.mean.len(), || None);
        for i in 0..self.mean.len() {
            if self.uploaded[i].as_ref().is_some_and(|(m, s)| same_bits(m, &posterior.mean[i]) && same_bits(s, &posterior.log_sd[i])) {
                continue;
            }
            let mean = self.rotations[i].as_ref().map_or_else(|| posterior.mean[i].clone(), |r| r.undo(&posterior.mean[i]));
            self.mean[i] = self.fitting.upload(mean.view()).map_err(error)?;
            self.log_sd[i] = self.fitting.upload(posterior.log_sd[i].view()).map_err(error)?;
            self.uploaded[i] = Some((posterior.mean[i].clone(), posterior.log_sd[i].clone()));
        }
        self.restart()
    }

    /// Trainable operator `i`'s `μ̄` and `s` on the host.
    pub fn values(&self, i: usize) -> Result<(Array2<f64>, Array2<f64>), String> {
        Ok((self.host_mean(i)?, self.fitting.download(&self.log_sd[i]).map_err(error)?))
    }

    /// The storage of the means, the log standard deviations, the gradient's momentum, the
    /// curvature estimate and the gradient's second moment (every operator's alike), in which a
    /// checkpoint keeps them.
    #[must_use]
    pub fn storages(&self) -> [Storage; 6] {
        match (self.mean.first(), self.log_sd.first(), self.moments.first()) {
            (Some(mean), Some(log_sd), Some([momentum, curvature, power])) => [mean.storage(), log_sd.storage(), momentum.storage(), curvature.storage(), power.storage(), mean.storage()],
            _ => [self.fitting.storage(); 6],
        }
    }

    /// Trainable operator `i`'s iterate `μ` on the host as the device holds it (along its rotated
    /// axes), whose Polyak average is the posterior's mean.
    pub fn iterate(&self, i: usize) -> Result<Array2<f64>, String> {
        self.fitting.download(self.mean.get(i).ok_or_else(|| error("no such trainable operator"))?).map_err(error)
    }

    /// The steps the posterior's mean averages the iterate over.
    #[must_use]
    pub fn averaged(&self) -> u64 {
        self.averaged
    }

    /// The line arm's ratio of the joint curvature to the diagonal one, and the steps it averages.
    #[must_use]
    pub fn line_ratio(&self) -> (f64, u64) {
        (self.ratio, self.ratio_steps)
    }

    /// The line arm's ratio and its steps, restored from a checkpoint.
    pub fn set_line_ratio(&mut self, (ratio, steps): (f64, u64)) {
        (self.ratio, self.ratio_steps) = (ratio, steps);
    }

    /// Trainable operator `i`'s state on the host as the device holds it, one operator at a time
    /// (a checkpoint streams them rather than holding every operator's state at once): `μ`, `s`
    /// and IVON's state, all along the operator's rotated axes where it has a rotation.
    pub fn operator(&self, i: usize) -> Result<(Array2<f64>, Array2<f64>, [Array2<f64>; 3]), String> {
        let down = |t: &Tensor| self.fitting.download(t).map_err(error);
        let m = self.moments.get(i).ok_or_else(|| error("no such trainable operator"))?;
        Ok((down(&self.average[i])?, down(&self.log_sd[i])?, [down(&m[0])?, down(&m[1])?, down(&m[2])?]))
    }

    /// The means as the device holds them (along each operator's rotated axes), restored exactly
    /// from a checkpoint ([`DevicePosterior::operator`]).
    pub fn restore_means(&mut self, held: &[Array2<f64>]) -> Result<(), String> {
        if held.len() != self.mean.len() || held.iter().zip(&self.mean).any(|(h, m)| h.dim() != (m.rows(), m.cols())) {
            return Err(error("one mean per trainable operator, of its shape, required"));
        }
        self.uploaded.clear();
        for (mean, values) in self.mean.iter_mut().zip(held) {
            *mean = self.fitting.upload(values.view()).map_err(error)?;
        }
        self.restart()
    }

    /// The posterior's mean `held`, the iterate `iterate` it averages and the steps `averaged` it
    /// averages over, restored exactly from a checkpoint ([`DevicePosterior::operator`],
    /// [`DevicePosterior::iterate`]): the fit goes on as if it had not stopped.
    pub fn restore(&mut self, held: &[Array2<f64>], iterate: &[Array2<f64>], averaged: u64) -> Result<(), String> {
        let fits = |arrays: &[Array2<f64>]| arrays.len() == self.mean.len() && arrays.iter().zip(&self.mean).all(|(a, m)| a.dim() == (m.rows(), m.cols()));
        if !fits(held) || !fits(iterate) {
            return Err(error("one mean and one iterate per trainable operator, of its shape, required"));
        }
        for ((mean, average), (values, iterate)) in self.mean.iter_mut().zip(self.average.iter_mut()).zip(held.iter().zip(iterate)) {
            *mean = self.fitting.upload(iterate.view()).map_err(error)?;
            *average = self.fitting.upload(values.view()).map_err(error)?;
        }
        self.averaged = averaged;
        // The groups' variances and divergences at `μ̄`, as the last step left them.
        self.sums = self.wide.zeros(self.sums.rows(), 3).map_err(error)?;
        self.refresh()
    }

}

#[cfg(test)]
mod tests {
    use super::*;
    use gam_gpu::tensor::posterior_normal;

    /// One prior group of 64 entries at `N = 2^16`: 32 without data curvature whose gradient is
    /// pure noise of standard deviation `s` per step, starting at `μ = 0.05`, and 32 with
    /// curvature `h` and data term `½ h (θ − a)²`. Over 3000 steps (`β₂ = 1 − 1/64`) the group's
    /// variance is the empirical-Bayes value at the posterior mean `μ̄` (not at the iterate),
    /// stays within 10% of its start (the iterate-charged variance grew 35% per epoch on vpd4l), and the curvature-free entries' `μ̄` go to zero against the prior
    /// scale `√v_G`.
    #[test]
    fn the_variance_is_charged_at_the_averaged_mean_and_stays_bounded() {
        const R: usize = 64;
        let device = Device::host();
        let tokens = 65_536.0;
        let free = |i: usize| i < R / 2;
        let a = Array2::from_shape_fn((1, R), |(_, i)| if free(i) { 0.05 } else { 0.1 * f64::from(posterior_normal(5, 0, i as u64)) });
        let v0 = a.iter().map(|x| x * x).sum::<f64>() / R as f64;
        let h = 1.0 / (tokens * v0);
        let log_sd = Array2::from_shape_fn((1, R), |(_, i)| if free(i) { 0.5 * v0.ln() } else { -0.5 * (tokens * (h + 1.0 / (tokens * v0))).ln() });
        let groups = vec![vec![0u32; R]];
        let parts = Parts { operators: &[0], mean: std::slice::from_ref(&a), log_sd: std::slice::from_ref(&log_sd), groups: &groups, count: 1, rotations: &[None] };
        let mut posterior = DevicePosterior::from_parts(&device, &parts, tokens, None, 0).unwrap();
        let start = posterior.variances().unwrap()[0];
        let ivon = Ivon { rate: 0.1, beta1: 0.9, beta2: 1.0 - 1.0 / 64.0 };
        let s = 1e-4;
        let factor = device.upload(Array2::from_shape_fn((1, R), |(_, i)| if free(i) { 0.0 } else { h.sqrt() }).view()).unwrap();
        let mut largest: f64 = 0.0;
        for t in 1..=3000u64 {
            let (mean, log_sd) = posterior.values(0).unwrap();
            let gradient = Array2::from_shape_fn((1, R), |(_, i)| {
                let noise = s * f64::from(posterior_normal(12, t, i as u64));
                if free(i) { noise } else { h * (mean[(0, i)] + log_sd[(0, i)].exp() * f64::from(posterior_normal(11, t, i as u64)) - a[(0, i)]) + noise }
            });
            let gradients = BTreeMap::from([(0, device.upload(gradient.view()).unwrap())]);
            let factors = BTreeMap::from([(0, device.copy(&factor).unwrap())]);
            posterior.step(&gradients, 1.0, (&factors, 1.0), &ivon).unwrap();
            largest = largest.max(posterior.variances().unwrap()[0]);
        }
        let (mean, log_sd) = posterior.values(0).unwrap();
        let v = posterior.variances().unwrap()[0];
        let charged = mean.iter().zip(&log_sd).map(|(m, s)| m * m + (2.0 * s).exp()).sum::<f64>() / R as f64;
        assert!((v - charged).abs() <= 1e-9 * v, "v_G {v} against mean(μ̄² + σ²) {charged}");
        let iterate = device.download(&posterior.mean[0]).unwrap();
        assert!(iterate.iter().zip(&mean).any(|(x, y)| x != y), "the iterate is its own average");
        assert!(largest <= 1.1 * start, "v_G rose from {start} to {largest}");
        let spread = ((0..R / 2).map(|i| mean[(0, i)] * mean[(0, i)]).sum::<f64>() / (R / 2) as f64).sqrt();
        assert!(spread <= 0.1 * v.sqrt(), "curvature-free μ̄ at rms {spread} against √v_G {}", v.sqrt());
    }
}
