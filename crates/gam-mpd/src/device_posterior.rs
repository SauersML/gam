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
    library_mdl::{Curvature, Explanation, Posterior, Shared},
};
use gam_gpu::{
    gpu_error::GpuError,
    tensor::{Device, GroupMap, PosteriorStep, Storage, Tensor},
};
use ndarray::Array2;
use std::{borrow::Borrow, collections::BTreeMap};

/// The batches whose removal sums wait on the device before they are read
/// ([`DevicePosterior::add_removal`]): about 45 MB of sums on vpd4l's 29,184 groups.
pub const REMOVAL_READS: usize = 64;

/// Draws of the step's curvature ratio averaged before its first move: the relative
/// standard error of a mean of `K` single-draw ratios is `√(2 / K)`, one half at `K = 8` (and the
/// chance that the mean is below a tenth of its expectation 0.08%, from 25% at one draw). At
/// `N = 2^24` (4096 batches per epoch) the eight steps are 0.2% of the first epoch.
const RATIO_DRAWS: u64 = 8;

fn error(e: impl std::fmt::Display) -> String {
    format!("device posterior: {e}")
}

/// IVON's settings of a step ([`DevicePosterior::step`], [`Device::posterior_ivon`]): the gradient
/// momentum's decay `β₁` and the curvature estimate's decay `β₂`. The step's length along IVON's
/// direction is the Gauss–Newton minimum with the epoch-averaged curvature ratio
/// ([`DevicePosterior::step`]).
#[derive(Clone, Copy, Debug)]
pub struct Ivon {
    pub beta1: f64,
    pub beta2: f64,
}

/// IVON's state a device posterior starts from ([`DevicePosterior::new`]); none: the momentum and
/// second moment zero and the curvature at which IVON's standard deviations are the posterior's.
pub enum State<'a> {
    /// Per operator the gradient's momentum, the curvature estimate and the gradient's second
    /// moment (a checkpoint's).
    Saved(&'a [[Array2<f64>; 3]]),
    /// The momentum, the curvature and the second moment zero, made on the device: a start whose
    /// deviations and curvature are then set one operator at a time ([`DevicePosterior::set_start`],
    /// the Laplace start).
    Zero,
}

impl State<'_> {
    /// The operators it holds a state for, when it holds one per operator.
    fn len(&self) -> Option<usize> {
        match self {
            Self::Saved(moments) => Some(moments.len()),
            Self::Zero => None,
        }
    }
}

/// A posterior's host arrays ([`DevicePosterior::from_parts`]).
pub struct Parts<'a, A: Borrow<Array2<f64>> = Array2<f64>> {
    pub operators: &'a [usize],
    pub mean: &'a [A],
    pub log_sd: &'a [A],
    pub groups: &'a [Vec<u32>],
    pub count: usize,
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
    /// The step's length along IVON's direction ([`DevicePosterior::step`]): the epoch's average
    /// `rho` of one Gauss–Newton draw's curvature along `d` over the diagonal model's, over
    /// `rho_steps` draws; the last step's `η`; and whether the means are held (a pass that sets
    /// the deviations only, [`DevicePosterior::hold_means`]).
    rho: f64,
    rho_steps: u64,
    last_eta: f64,
    hold: bool,
    /// Per group `(n, Σ μ² + σ², Σ 2s)` being summed, its variance and its divergence in nats.
    sums: Tensor,
    variance: Tensor,
    divergence: Tensor,
    /// The training tokens `N` (the data term's weight) and the steps taken.
    tokens: f64,
    steps: u64,
    /// The momentum's weights `(W, W2)` (`PosteriorStep::weights`), kept whatever `β₁` each step
    /// used.
    momentum_weights: (f64, f64),
    /// Per operator, the host `μ` and `s` its device values were last set from
    /// ([`DevicePosterior::set_values`]) while no step or restore has changed them since, held as
    /// the posterior's own shared arrays (no copy): a later `set_values` sends only the operators
    /// whose arrays are not these ([`Shared::same`]; a removal trial shares the operators it does
    /// not change, and a write to a shared array copies it).
    uploaded: Vec<Option<(Shared, Shared)>>,
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
    pub fn new(fitting: &Device, explanation: &Explanation, posterior: &Posterior, tokens: f64, moments: Option<State<'_>>, steps: u64) -> Result<Self, String> {
        let shapes: Vec<(usize, usize)> = posterior.mean.iter().map(|m| m.dim()).collect();
        if shapes.len() != explanation.trainable.len() {
            return Err(error("one posterior array per trainable operator required"));
        }
        let groups = membership(explanation, &shapes)?;
        let parts = Parts { operators: &explanation.trainable, mean: &posterior.mean, log_sd: &posterior.log_sd, groups: &groups, count: explanation.groups.len() };
        Self::from_parts(fitting, &parts, tokens, moments, steps)
    }

    /// The posterior of the trainable operators `parts.operators` of a program, each entry in group
    /// `parts.groups[i][entry]` (row-major) of `parts.count`, for `tokens` training tokens on
    /// `fitting`, with IVON's state `moments` (see [`DevicePosterior::new`]) after `steps` steps.
    pub fn from_parts<A: Borrow<Array2<f64>>>(fitting: &Device, parts: &Parts<'_, A>, tokens: f64, moments: Option<State<'_>>, steps: u64) -> Result<Self, String> {
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
        let shapes: Vec<(usize, usize)> = parts.mean.iter().map(|m| m.borrow().dim()).collect();
        let sizes_agree = parts.log_sd.iter().map(|s| s.borrow().dim()).eq(shapes.iter().copied()) && parts.groups.iter().map(Vec::len).eq(shapes.iter().map(|(r, c)| r * c));
        if shapes.len() != parts.operators.len() || !sizes_agree || moments.as_ref().and_then(State::len).is_some_and(|n| n != shapes.len()) {
            return Err(error("one posterior array and group list per trainable operator required"));
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
        // Each operator's state is made and sent one operator at a time, its zeros on the device.
        let mut sums = vec![(0.0, 0.0); parts.count];
        if moments.is_none() {
            for ((mean, log_sd), groups) in parts.mean.iter().zip(parts.log_sd).zip(parts.groups) {
                for ((mu, s), g) in mean.borrow().iter().zip(log_sd.borrow().iter()).zip(groups) {
                    if *s != f64::NEG_INFINITY {
                        sums[*g as usize].0 += 1.0;
                        sums[*g as usize].1 += mu * mu + (2.0 * s).exp();
                    }
                }
            }
        }
        let start = |log_sd: &Array2<f64>, groups: &[u32]| -> Array2<f64> {
            let (rows, cols) = log_sd.dim();
            Array2::from_shape_fn((rows, cols), |(r, c)| {
                let (s, (n, second)) = (log_sd[[r, c]], sums[groups[r * cols + c] as usize]);
                if s == f64::NEG_INFINITY { 0.0 } else { (1.0 / (tokens * (2.0 * s).exp()) - n / (tokens * second)).max(0.0) }
            })
        };
        let up = |m: &Array2<f64>| master.upload(m.view()).map_err(error);
        let moment = |m: &Array2<f64>| narrow.upload(m.view()).map_err(error);
        let state = |i: usize| -> Result<[Tensor; 3], String> {
            let (rows, cols) = shapes[i];
            let zero = |d: &Device| d.zeros(rows, cols).map_err(error);
            Ok(match &moments {
                Some(State::Saved(m)) => [moment(&m[i][0])?, up(&m[i][1])?, up(&m[i][2])?],
                Some(State::Zero) => [zero(&narrow)?, zero(&master)?, zero(&master)?],
                None => [zero(&narrow)?, up(&start(parts.log_sd[i].borrow(), &parts.groups[i]))?, zero(&master)?],
            })
        };
        let mut out = Self {
            sums: wide.zeros(parts.count, 3).map_err(error)?,
            variance: wide.zeros(parts.count, 1).map_err(error)?,
            divergence: wide.zeros(parts.count, 1).map_err(error)?,
            mean: parts.mean.iter().map(|m| up(m.borrow())).collect::<Result<_, _>>()?,
            log_sd: parts.log_sd.iter().map(|s| up(s.borrow())).collect::<Result<_, _>>()?,
            moments: (0..shapes.len()).map(state).collect::<Result<_, String>>()?,
            groups: parts.groups.iter().zip(&shapes).map(|(ids, shape)| master.group_map(ids, *shape).map_err(error)).collect::<Result<_, _>>()?,
            average: Vec::new(),
            averaged: 0,
            rho: 1.0,
            rho_steps: 0,
            last_eta: 0.0,
            hold: false,
            operators: parts.operators.to_vec(),
            fitting: fitting.clone(),
            wide,
            tokens,
            steps,
            // A posterior resumed after `steps` steps of the fixed `β₁ = 0.9` fits used before the
            // rule (`library_mdl::momentum_decay`).
            momentum_weights: PosteriorStep::constant_weights(0.9, steps),
            uploaded: Vec::new(),
        };
        out.average = out.mean.iter().map(|m| out.fitting.copy(m).map_err(error)).collect::<Result<_, _>>()?;
        out.refresh()?;
        Ok(out)
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

    /// Operator `i`'s posterior means `μ̄` on the host.
    fn host_mean(&self, i: usize) -> Result<Array2<f64>, String> {
        self.fitting.download(&self.average[i]).map_err(error)
    }

    /// Holds the means (`η = 0` every step) while `hold`: a pass that sets the deviations and the
    /// curvature only, such as a pricing pass.
    pub fn hold_means(&mut self, hold: bool) {
        self.hold = hold;
    }

    /// The last step's `η` and the curvature ratio's average `ρ̄` with its draws.
    #[must_use]
    pub fn step_state(&self) -> (f64, f64, u64) {
        (self.last_eta, self.rho, self.rho_steps)
    }

    /// Operator `i`'s direction `d = before − μ` after the kernel's full step, with its terms added
    /// into `sums` (groups × 5, per group: `g · d`, `Σ d²`, `μ · d` at the step's start, `u · d`,
    /// `Σ h d²`), in one pass over the operator.
    fn line_terms(&self, i: usize, before: &Tensor, (gradient, draw): (&Tensor, &Tensor), sums: &mut Tensor) -> Result<Tensor, String> {
        let mut d = self.fitting.empty(before.rows(), before.cols()).map_err(error)?;
        self.fitting
            .group_line_terms((gradient, draw, &self.moments[i][1]), (before, &self.mean[i], &self.log_sd[i]), &self.groups[i], (&mut d, sums))
            .map_err(error)?;
        Ok(d)
    }

    /// The posterior's mean, the iterate's Polyak average (uniform over the steps since the
    /// posterior was set, then over about one epoch; module note), and the groups' variances and
    /// divergences at it.
    fn average_and_refresh(&mut self, beta2: f64) -> Result<(), String> {
        self.averaged += 1;
        let weight = (1.0 / self.averaged as f64).max(1.0 - beta2);
        for (average, mean) in self.average.iter_mut().zip(&self.mean) {
            self.fitting.move_toward(average, weight, mean).map_err(error)?;
        }
        self.sums = self.wide.zeros(self.sums.rows(), 3).map_err(error)?;
        self.refresh()
    }

    /// The steps taken.
    #[must_use]
    pub fn steps(&self) -> u64 {
        self.steps
    }

    /// The momentum's weights `(W, W2)`: its bias correction `W` and effective number of
    /// gradients `W² / W2`.
    #[must_use]
    pub fn momentum_weights(&self) -> (f64, f64) {
        self.momentum_weights
    }

    /// The momentum's weights `(W, W2)`, restored from a checkpoint with IVON's state.
    pub fn set_momentum_weights(&mut self, weights: (f64, f64)) {
        self.momentum_weights = weights;
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
    pub fn sample_block(&self, op: usize, out: &mut Tensor, at: (usize, usize), key: u64) -> Result<(), String> {
        self.block_of(op, out, at, key, &self.average)
    }

    /// [`DevicePosterior::sample_block`] around `means` (the posterior's `μ̄` or the iterate).
    fn block_of(&self, op: usize, out: &mut Tensor, at: (usize, usize), key: u64, means: &[Tensor]) -> Result<(), String> {
        let (i, _, log_sd) = self.entries(op)?;
        self.fitting.reparameterize_block(out, at, (&means[i], log_sd), (key, i as u64)).map_err(error)
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
            let value = if bf16 { self.fitting.bf16_copy(&means[i]) } else { self.fitting.copy(&means[i]) };
            program.replace_dense_parameter(op, value.map_err(error)?)?;
        }
        program.refresh_fused()
    }

    /// Writes the rounded posterior means (each `μ` to the nearest multiple of `2^⌊log2 σ⌋`,
    /// `Posterior::rounded`) into `program`'s trainable operators (rounded to its storage).
    pub fn rounded_into(&self, program: &mut DeviceProgram) -> Result<(), String> {
        for (i, &op) in self.operators.iter().enumerate() {
            let mut rounded = self.fitting.empty(self.average[i].rows(), self.average[i].cols()).map_err(error)?;
            self.fitting.round_to_deviation(&mut rounded, (&self.average[i], &self.log_sd[i])).map_err(error)?;
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
    /// than `μ̄`).
    pub fn iterate_into(&self, program: &mut DeviceProgram, key: u64) -> Result<(), String> {
        self.sample_of(program, key, &self.mean)
    }

    /// [`DevicePosterior::sample_into`] around `means`.
    ///
    /// The operators held in place and the stacks' blocks are gathered and written together
    /// ([`Device::run_samples`]); an operator replaced (a bias's column copy) is written first.
    fn sample_of(&self, program: &mut DeviceProgram, key: u64, means: &[Tensor]) -> Result<(), String> {
        let mut samples = self.fitting.samples(key);
        for (i, &op) in self.operators.iter().enumerate() {
            let parts = (&means[i], &self.log_sd[i]);
            if let Ok(theta) = program.dense_mut(op) {
                // SAFETY: the operator's copy and the masters stay in place until run_samples below,
                // and nothing enqueued before it reads the copy (a replacement copies only its own
                // value, the restacking only operators it does not write through the callback).
                unsafe { self.fitting.add_sample(&mut samples, theta, (0, 0), parts, i as u64) }.map_err(error)?;
            } else {
                let (rows, cols) = (means[i].rows(), means[i].cols());
                let mut theta = self.fitting.zeros(rows, cols).map_err(error)?;
                self.fitting.reparameterize(&mut theta, parts, (key, i as u64)).map_err(error)?;
                program.replace_dense_parameter(op, theta)?;
            }
        }
        program.refresh_fused_with(&mut |op, stack, at| {
            let Some(i) = self.operators.iter().position(|o| *o == op) else {
                return Ok(false);
            };
            // SAFETY: the stack (the group's own, copied first when shared) and the masters stay in
            // place until run_samples below, and the restacking reads no block it writes here.
            unsafe { self.fitting.add_sample(&mut samples, stack, at, (&means[i], &self.log_sd[i]), i as u64) }.map_err(error)?;
            Ok(true)
        })?;
        self.fitting.run_samples(samples).map_err(error)
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
        self.momentum_weights = PosteriorStep::weights_after(self.momentum_weights, ivon.beta1);
        // The kernel takes IVON's full direction from the iterate kept here.
        let before: Vec<Tensor> = self.mean.iter().map(|m| self.fitting.copy(m).map_err(error)).collect::<Result<_, _>>()?;
        let mut sums = self.wide.zeros(self.group_count(), 5).map_err(error)?;
        let mut directions = Vec::with_capacity(before.len());
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
            let step = PosteriorStep { gradient_scale: scale, factor_scale: factor.1, tokens: self.tokens, beta1: ivon.beta1, beta2: ivon.beta2, weights: self.momentum_weights };
            let [momentum, curvature, power] = &mut self.moments[i];
            self.fitting
                .posterior_ivon((&mut self.mean[i], &mut self.log_sd[i]), [momentum, curvature, power], (gradient, draw), (&self.groups[i], &self.variance), &mut self.sums, &step)
                .map_err(error)?;
            directions.push(self.line_terms(i, &before[i], (gradient, draw), &mut sums)?);
        }
        {
            // The step's length along `d`: the minimum along `d` of `F`'s Gauss–Newton model,
            // `η = (Σ h d² + Σ δ d²) / (ρ̄ Σ h d² + Σ δ d²)` (the filtered gradient's slope
            // `ĝ · d = Σ (h + δ) d²` over the curvature along `d`), with the data curvature
            // `ρ̄ Σ h d²`: `ρ̄` the average over the steps (uniform, then over about one epoch,
            // `w = max(1/t, 1 − β₂)`) of one draw's `c (u · d)² / Σ h d²` (`u` the step's
            // sampled-label factor, `c` the factor turning its square into curvature per token:
            // `E[c (u · d)²] = dᵀ G d`, with the entries' joint terms the diagonal `h` omits). One
            // draw is a single χ²₁-like sample whatever the batch's tokens; the epoch's average is
            // what the step uses, and the iterate stays until `RATIO_DRAWS` draws are averaged. At
            // 2^24 a step measured on the next batch instead (one batch's noisy measurement, three
            // forward passes) reached held-out F 2.230, 2.154, 2.100 after epochs 1–3 against this
            // rule's 2.202, 2.096, 2.015 at two thirds of the time; on the one-block case at equal
            // tokens it ended at 2.633 against 2.866 bits per token but took 2.6 times the time.
            let variances = self.variances()?;
            let terms = self.wide.download(&sums).map_err(error)?;
            let column = |k: usize| terms.column(k).to_vec();
            let precision = |g: usize| if variances[g] > 0.0 { 1.0 / (self.tokens * variances[g]) } else { 0.0 };
            let weighted = |values: Vec<f64>| -> f64 { values.iter().enumerate().map(|(g, x)| precision(g) * x).sum() };
            let along_u: f64 = column(3).iter().sum();
            let draw_curvature = factor.1 * along_u * along_u;
            let diagonal: f64 = column(4).iter().sum();
            let prior_curvature = weighted(column(1));
            if diagonal > 0.0 {
                self.rho_steps += 1;
                let w = (1.0 / self.rho_steps as f64).max(1.0 - ivon.beta2);
                self.rho += w * (draw_curvature / diagonal - self.rho);
            }
            let (slope, curvature) = (diagonal + prior_curvature, self.rho * diagonal + prior_curvature);
            let eta = if self.hold || self.rho_steps < RATIO_DRAWS || !(curvature > 0.0 && slope.is_finite()) { 0.0 } else { slope / curvature };
            self.last_eta = eta;
            for ((mean, start), d) in self.mean.iter_mut().zip(&before).zip(&directions) {
                self.fitting.axpy_from(mean, start, -eta, d).map_err(error)?;
            }
        }
        self.average_and_refresh(ivon.beta2)
    }

    /// The number of prior groups.
    #[must_use]
    pub fn group_count(&self) -> usize {
        self.variance.rows()
    }

    /// Adds one batch to `curvature` (`library_mdl`'s module note): with `θ` the weight sample
    /// [`DevicePosterior::sample_into`] draws under `key`, `g` the batch's data gradient at `θ` (per
    /// trainable operator by id, on the device), times `nats`, and `u` a draw of the Gauss–Newton
    /// factor at `θ`, per group `g_G · θ_G` and `u_G · θ_G` over the group's live entries,
    /// summed on the device and read as one row per group; an operator neither reaches adds
    /// nothing. The sums wait on the device in `pending` and are read [`REMOVAL_READS`] batches at
    /// a time ([`DevicePosterior::read_removal`]), so the device sums one batch while the host
    /// prepares the next.
    pub fn add_removal(&self, (g, nats): (&BTreeMap<usize, Tensor>, f64), u: &BTreeMap<usize, Tensor>, key: u64, curvature: &mut Curvature, pending: &mut Vec<(Tensor, Tensor, f64)>) -> Result<(), String> {
        let groups = self.group_count();
        let (mut slopes, mut forms) = (self.wide.zeros(groups, 3).map_err(error)?, self.wide.zeros(groups, 3).map_err(error)?);
        for (i, op) in self.operators.iter().enumerate() {
            let (gradient, draw) = (g.get(op), u.get(op));
            if gradient.is_none() && draw.is_none() {
                continue;
            }
            // The sample as `sample_into` draws it.
            let mut theta = self.fitting.zeros(self.mean[i].rows(), self.mean[i].cols()).map_err(error)?;
            self.fitting.reparameterize(&mut theta, (&self.average[i], &self.log_sd[i]), (key, i as u64)).map_err(error)?;
            for (x, sums) in [(gradient, &mut slopes), (draw, &mut forms)] {
                let Some(x) = x else { continue };
                self.fitting.group_curvature((x, &theta, &self.log_sd[i]), &self.groups[i], sums).map_err(error)?;
            }
        }
        pending.push((slopes, forms, nats));
        if pending.len() >= REMOVAL_READS {
            self.read_removal(pending, curvature)?;
        }
        Ok(())
    }

    /// The batches' sums waiting in `pending` ([`DevicePosterior::add_removal`]) read into
    /// `curvature` in the order they were made.
    pub fn read_removal(&self, pending: &mut Vec<(Tensor, Tensor, f64)>, curvature: &mut Curvature) -> Result<(), String> {
        for (slopes, forms, nats) in pending.drain(..) {
            let slope: Vec<f64> = self.wide.download(&slopes).map_err(error)?.column(1).iter().map(|s| nats * s).collect();
            let dot: Vec<f64> = self.wide.download(&forms).map_err(error)?.column(1).to_vec();
            curvature.add_batch(&slope, &dot)?;
        }
        Ok(())
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
            posterior.mean[i] = self.host_mean(i)?.into();
            posterior.log_sd[i] = self.fitting.download(&self.log_sd[i]).map_err(error)?.into();
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
            if self.uploaded[i].as_ref().is_some_and(|(m, s)| m.same(&posterior.mean[i]) && s.same(&posterior.log_sd[i])) {
                continue;
            }
            self.mean[i] = self.fitting.upload(posterior.mean[i].view()).map_err(error)?;
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

    /// Trainable operator `i`'s iterate `μ` on the host, whose Polyak average is the posterior's
    /// mean.
    pub fn iterate(&self, i: usize) -> Result<Array2<f64>, String> {
        self.fitting.download(self.mean.get(i).ok_or_else(|| error("no such trainable operator"))?).map_err(error)
    }

    /// The steps the posterior's mean averages the iterate over.
    #[must_use]
    pub fn averaged(&self) -> u64 {
        self.averaged
    }

    /// The step's curvature-ratio average `ρ̄` and the draws it averages.
    #[must_use]
    pub fn line_ratio(&self) -> (f64, u64) {
        (self.rho, self.rho_steps)
    }

    /// The step's curvature-ratio average and its draws, restored from a checkpoint.
    pub fn set_line_ratio(&mut self, (rho, steps): (f64, u64)) {
        (self.rho, self.rho_steps) = (rho, steps);
    }

    /// Trainable operator `i`'s state on the host as the device holds it, one operator at a time
    /// (a checkpoint streams them rather than holding every operator's state at once): `μ`, `s`
    /// and IVON's state.
    pub fn operator(&self, i: usize) -> Result<(Array2<f64>, Array2<f64>, [Array2<f64>; 3]), String> {
        let down = |t: &Tensor| self.fitting.download(t).map_err(error);
        let m = self.moments.get(i).ok_or_else(|| error("no such trainable operator"))?;
        Ok((down(&self.average[i])?, down(&self.log_sd[i])?, [down(&m[0])?, down(&m[1])?, down(&m[2])?]))
    }

    /// The means as the device holds them, restored exactly from a checkpoint
    /// ([`DevicePosterior::operator`]).
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

    /// Operator `i`'s deviations `log_sd` (along its rotated axes) and IVON's curvature
    /// `curvature`, set one operator at a time after a start from [`State::Zero`], so no more than
    /// one operator's start is on the host; [`DevicePosterior::settle`] then takes the groups'
    /// variances and divergences at them.
    pub fn set_start(&mut self, i: usize, log_sd: &Array2<f64>, curvature: &Array2<f64>) -> Result<(), String> {
        let shape = self.log_sd.get(i).map(|t| (t.rows(), t.cols())).ok_or_else(|| error("no such trainable operator"))?;
        if log_sd.dim() != shape || curvature.dim() != shape {
            return Err(error("a start of another shape"));
        }
        self.log_sd[i] = self.fitting.upload(log_sd.view()).map_err(error)?;
        self.moments[i][1] = self.fitting.upload(curvature.view()).map_err(error)?;
        Ok(())
    }

    /// The groups' variances and divergences at the posterior the starts set
    /// ([`DevicePosterior::set_start`]).
    pub fn settle(&mut self) -> Result<(), String> {
        self.sums = self.wide.zeros(self.sums.rows(), 3).map_err(error)?;
        self.refresh()
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
        let parts = Parts { operators: &[0], mean: std::slice::from_ref(&a), log_sd: std::slice::from_ref(&log_sd), groups: &groups, count: 1 };
        let mut posterior = DevicePosterior::from_parts(&device, &parts, tokens, None, 0).unwrap();
        let start = posterior.variances().unwrap()[0];
        let ivon = Ivon { beta1: 0.9, beta2: 1.0 - 1.0 / 64.0 };
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
