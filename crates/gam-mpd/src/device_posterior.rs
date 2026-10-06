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
    library_mdl::{Curvature, Explanation, Posterior},
};
use gam_gpu::{
    gpu_error::GpuError,
    tensor::{Arithmetic, Device, GroupMap, Op, PosteriorStep, Storage, Tensor},
};
use ndarray::Array2;
use std::collections::BTreeMap;

/// A line step measured on its batch ([`DevicePosterior::finish_line`]): the step's direction
/// `d` per operator and the iterate `before` it starts from, and along `d` the data term's slope
/// `Σ g d` (per token, nats), the prior's slope `Σ δ μ d` and curvature `Σ δ d²`, and for the
/// record one Gauss–Newton draw's `c (u · d)²` and the diagonal's `Σ h d²`.
struct LineState {
    before: Vec<Tensor>,
    directions: Vec<Tensor>,
    slope_data: f64,
    slope_prior: f64,
    prior_curvature: f64,
    draw_curvature: f64,
    diagonal: f64,
}

/// What a measured line step found: its step `η`, the trial it measured, and the data term's
/// curvature along `d` measured from the trial, one Gauss–Newton draw's, and the diagonal's.
#[derive(Clone, Copy, Debug)]
pub struct LineReport {
    pub eta: f64,
    pub trial: f64,
    pub measured: f64,
    pub draw: f64,
    pub diagonal: f64,
    /// The data term's slope down `d` from the step's own gradient, and as measured.
    pub own_slope: f64,
    pub slope: f64,
}

/// The batches whose removal sums wait on the device before they are read
/// ([`DevicePosterior::add_removal`]): about 45 MB of sums on vpd4l's 29,184 groups.
pub const REMOVAL_READS: usize = 64;

fn error(e: impl std::fmt::Display) -> String {
    format!("device posterior: {e}")
}

/// IVON's settings of a step ([`DevicePosterior::step`], [`Device::posterior_ivon`]): the gradient
/// momentum's decay `β₁` and the curvature estimate's decay `β₂`. The step's length along IVON's
/// direction is measured ([`DevicePosterior::finish_line`]).
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
    /// Per operator the curvature estimate, the momentum and the second moment zero (a Laplace
    /// start): the zeros are made on the device, not sent from the host.
    Curvature(&'a [Array2<f64>]),
}

impl State<'_> {
    /// The operators it holds a state for.
    fn len(&self) -> usize {
        match self {
            Self::Saved(moments) => moments.len(),
            Self::Curvature(curvature) => curvature.len(),
        }
    }
}

/// A posterior's host arrays ([`DevicePosterior::from_parts`]).
pub struct Parts<'a> {
    pub operators: &'a [usize],
    pub mean: &'a [Array2<f64>],
    pub log_sd: &'a [Array2<f64>],
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
    /// The line step: the iterate moves to the minimum along IVON's full direction of `F`, measured
    /// at trial steps (`DevicePosterior::finish_line`); `ratio` is the next trial step and
    /// `ratio_steps` the line steps taken, `line_state` the step awaiting its measurement.
    ratio: f64,
    ratio_steps: u64,
    line_state: Option<LineState>,
    /// Per group `(n, Σ μ² + σ², Σ 2s)` being summed, its variance and its divergence in nats.
    sums: Tensor,
    variance: Tensor,
    divergence: Tensor,
    /// The training tokens `N` (the data term's weight) and the steps taken.
    tokens: f64,
    steps: u64,
    /// Per operator, the host `μ` and `s` its device values were last set from
    /// ([`DevicePosterior::set_values`]) while no step or restore has changed them since: a later
    /// `set_values` sends only the operators whose values differ in some bit (a removal trial
    /// changes a few operators of many).
    uploaded: Vec<Option<(Array2<f64>, Array2<f64>)>>,
    /// Per operator its prior groups (sorted, once each), and the step's preconditioner along its
    /// input axis where it has one ([`DevicePosterior::set_directions`]).
    members: Vec<Vec<u32>>,
    directions: Vec<Option<Direction>>,
}

/// The step's preconditioner along one operator's input axis (EKFAC's input side): `U` the
/// eigenvectors of the input's second moment `Σ x xᵀ` (`library_mdl::input_factors`), and
/// IVON's state along those axes (momentum, curvature, gradient second moment) for a second pass
/// of the step's kernel there. The diagonal curvature `h` along the operator's own axes cannot see
/// entries that move together: an MLP's output map, whose input is the functions' activations,
/// shifts every function's output column by nearly one residual vector along the
/// mean-activation direction, which the diagonal underprices about 1500× (toygate, one MLP
/// block); the input's second moment has that direction as an eigenvector, so the curvature per
/// entry along `U` prices it. The pass uses one prior precision for the whole operator, the mean
/// `δ̄` of its groups' `1 / (N v_G)` (the groups' precisions are not diagonal along `U`): the
/// step's length along the resulting direction is measured on the exact `F`
/// ([`DevicePosterior::finish_line`]), so the preconditioner need only be positive definite.
struct Direction {
    /// `U_A` (columns × columns) and, where the output factor is known, `U_G` (rows × rows).
    input: Tensor,
    output: Option<Tensor>,
    moments: [Tensor; 3],
    groups: GroupMap,
    variance: Tensor,
    sums: Tensor,
    /// 1 at the operator's live entries and 0 at removed ones (`s = −∞`): the direction mixes the
    /// operator's axes, and a removed entry must stay at zero.
    live: Tensor,
    /// The step's buffers (rows × columns), allocated once: the turned gradient, draw and iterate,
    /// the iterate before the pass, the deviations the pass writes, and two products' scratch.
    gradient: Tensor,
    draw: Tensor,
    iterate: Tensor,
    start: Tensor,
    deviations: Tensor,
    scratch: Tensor,
    moved: Tensor,
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
    pub fn new(fitting: &Device, explanation: &Explanation, posterior: &Posterior, tokens: f64, moments: Option<State<'_>>, steps: u64) -> Result<Self, String> {
        let shapes: Vec<(usize, usize)> = posterior.mean.iter().map(Array2::dim).collect();
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
    pub fn from_parts(fitting: &Device, parts: &Parts<'_>, tokens: f64, moments: Option<State<'_>>, steps: u64) -> Result<Self, String> {
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
        if shapes.len() != parts.operators.len() || !sizes_agree || moments.as_ref().is_some_and(|m| m.len() != shapes.len()) {
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
                for ((mu, s), g) in mean.iter().zip(log_sd.iter()).zip(groups) {
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
                Some(State::Curvature(h)) => [zero(&narrow)?, up(&h[i])?, zero(&master)?],
                None => [zero(&narrow)?, up(&start(&parts.log_sd[i], &parts.groups[i]))?, zero(&master)?],
            })
        };
        let mut out = Self {
            sums: wide.zeros(parts.count, 3).map_err(error)?,
            variance: wide.zeros(parts.count, 1).map_err(error)?,
            divergence: wide.zeros(parts.count, 1).map_err(error)?,
            mean: parts.mean.iter().map(up).collect::<Result<_, _>>()?,
            log_sd: parts.log_sd.iter().map(up).collect::<Result<_, _>>()?,
            moments: (0..shapes.len()).map(state).collect::<Result<_, String>>()?,
            groups: parts.groups.iter().zip(&shapes).map(|(ids, shape)| master.group_map(ids, *shape).map_err(error)).collect::<Result<_, _>>()?,
            average: Vec::new(),
            averaged: 0,
            ratio: 1.0,
            ratio_steps: 0,
            line_state: None,
            operators: parts.operators.to_vec(),
            fitting: fitting.clone(),
            wide,
            tokens,
            steps,
            uploaded: Vec::new(),
            members: parts
                .groups
                .iter()
                .map(|ids| {
                    let mut unique = ids.clone();
                    unique.sort_unstable();
                    unique.dedup();
                    unique
                })
                .collect(),
            directions: Vec::new(),
        };
        out.average = out.mean.iter().map(|m| out.fitting.copy(m).map_err(error)).collect::<Result<_, _>>()?;
        out.refresh()?;
        Ok(out)
    }

    /// Trainable operator `i`'s iterate and its log standard deviations, on the host: where a
    /// step's gradient is taken (`DevicePosterior::iterate_into`), for a caller that measures a
    /// line step on the host.
    pub fn iterate_values(&self, i: usize) -> Result<(Array2<f64>, Array2<f64>), String> {
        let mean = self.fitting.download(&self.mean[i]).map_err(error)?;
        Ok((mean, self.fitting.download(&self.log_sd[i]).map_err(error)?))
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
        self.line_state = None;
        self.refresh_live()?;
        self.refresh()
    }

    /// The products' arithmetic in the fitting storage.
    fn arithmetic(&self) -> Arithmetic {
        if self.fitting.storage() == Storage::F64 { Arithmetic::F64 } else { Arithmetic::F32 }
    }

    /// Sets the step's preconditioner from `inputs` (per operator the eigenvectors `U_A` of its
    /// input's second moment `A`, its columns' axis, and their eigenvalues `λ_A` per token) and
    /// `outputs` (per operator the eigenvectors `U_G` and eigenvalues `λ_G` of its output-side
    /// Gauss–Newton factor `G`, its rows' axis, where known): see [`Direction`]. The curvature in
    /// the product basis starts at the Kronecker model's diagonal there, `h̃_rk = c λ_G,r λ_A,k`
    /// (`λ_G = 1` and `U_G = I` without an output factor), with `c` matching the diagonal
    /// curvature's total, `Σ h̃ = Σ h` (the diagonal `h_rj ≈ G_rr A_jj` sums to `tr G tr A`). The
    /// momentum and the gradient's second moment start at zero. A resumed fit sets them afresh.
    pub fn set_directions(&mut self, inputs: &[Option<std::sync::Arc<crate::library_mdl::InputFactor>>], outputs: &[Option<std::sync::Arc<crate::library_mdl::InputFactor>>]) -> Result<(), String> {
        if inputs.len() != self.mean.len() || outputs.len() != self.mean.len() {
            return Err(error("one input and one output factor per trainable operator required"));
        }
        self.directions = Vec::with_capacity(inputs.len());
        for (i, (input, output)) in inputs.iter().zip(outputs).enumerate() {
            let (rows, cols) = (self.mean[i].rows(), self.mean[i].cols());
            let Some(input) = input.as_ref().filter(|f| f.vectors.dim() == (cols, cols) && f.values.len() == cols && f.values.iter().sum::<f64>() > 0.0) else {
                self.directions.push(None);
                continue;
            };
            let output = output.as_ref().filter(|f| f.vectors.dim() == (rows, rows) && f.values.len() == rows && f.values.iter().sum::<f64>() > 0.0);
            let output_values: Vec<f64> = output.map_or_else(|| vec![1.0; rows], |f| f.values.clone());
            let total: f64 = self.fitting.download(&self.moments[i][1]).map_err(error)?.sum();
            let scale = total / (output_values.iter().sum::<f64>() * input.values.iter().sum::<f64>());
            let curvature = self.fitting.upload(Array2::from_shape_fn((rows, cols), |(r, k)| scale * output_values[r] * input.values[k]).view()).map_err(error)?;
            let zeros = || self.fitting.zeros(rows, cols).map_err(error);
            self.directions.push(Some(Direction {
                input: self.fitting.upload(input.vectors.view()).map_err(error)?,
                output: output.map(|f| self.fitting.upload(f.vectors.view()).map_err(error)).transpose()?,
                moments: [zeros()?, curvature, zeros()?],
                groups: self.fitting.group_map(&vec![0; rows * cols], (rows, cols)).map_err(error)?,
                variance: self.wide.zeros(1, 1).map_err(error)?,
                sums: self.wide.zeros(1, 3).map_err(error)?,
                live: zeros()?,
                gradient: zeros()?,
                draw: zeros()?,
                iterate: zeros()?,
                start: zeros()?,
                deviations: zeros()?,
                scratch: zeros()?,
                moved: zeros()?,
            }));
        }
        self.refresh_live()
    }

    /// Each preconditioned operator's live-entry mask from its log standard deviations.
    fn refresh_live(&mut self) -> Result<(), String> {
        for (i, direction) in self.directions.iter_mut().enumerate() {
            let Some(direction) = direction else { continue };
            let s = self.fitting.download(&self.log_sd[i]).map_err(error)?;
            direction.live = self.fitting.upload(s.mapv(|x| if x == f64::NEG_INFINITY { 0.0 } else { 1.0 }).view()).map_err(error)?;
        }
        Ok(())
    }

    /// Operator `i`'s step direction preconditioned in the product basis of its factors, written
    /// as the iterate `before − d` the line step reads its direction from: the step's kernel run a
    /// second time on the turned gradient, Gauss–Newton draw and iterate (`U_Gᵀ x U_A`) with
    /// IVON's state there and the operator's mean prior precision `δ̄`, its move `d̃` turned back
    /// (`d = U_G d̃ U_Aᵀ`) and zero at removed entries ([`Direction`]). The buffers are the
    /// direction's own: a step allocates nothing here.
    fn precondition(&mut self, i: usize, before: &Tensor, (gradient, draw): (&Tensor, &Tensor), step: &PosteriorStep, variances: &[f64]) -> Result<(), String> {
        let arithmetic = self.arithmetic();
        let Some(direction) = self.directions.get_mut(i).and_then(Option::as_mut) else { return Ok(()) };
        let precision: Vec<f64> = self.members[i].iter().map(|&g| variances[g as usize]).filter(|v| *v > 0.0).map(|v| 1.0 / (self.tokens * v)).collect();
        if precision.is_empty() {
            return Ok(());
        }
        let mean_precision = precision.iter().sum::<f64>() / precision.len() as f64;
        direction.variance = self.wide.upload(Array2::from_elem((1, 1), 1.0 / (self.tokens * mean_precision)).view()).map_err(error)?;
        let d = &self.fitting;
        let Direction { input, output, moments, groups, variance, sums, live, gradient: along_gradient, draw: along_draw, iterate, start, deviations, scratch, moved } = direction;
        // `out ← U_Gᵀ x U_A` (or `x U_A`).
        let turn = |out: &mut Tensor, x: &Tensor, scratch: &mut Tensor| -> Result<(), String> {
            match output.as_ref() {
                Some(g) => {
                    d.gemm(scratch, 1.0, x, Op::N, input, Op::N, 0.0, arithmetic).map_err(error)?;
                    d.gemm(out, 1.0, g, Op::T, scratch, Op::N, 0.0, arithmetic).map_err(error)
                }
                None => d.gemm(out, 1.0, x, Op::N, input, Op::N, 0.0, arithmetic).map_err(error),
            }
        };
        turn(along_gradient, gradient, scratch)?;
        turn(along_draw, draw, scratch)?;
        turn(iterate, before, scratch)?;
        d.set_rows(start, 0, iterate).map_err(error)?;
        let [momentum, curvature, power] = moments;
        d.posterior_ivon((&mut *iterate, &mut *deviations), [momentum, curvature, power], (&*along_gradient, &*along_draw), (&*groups, &*variance), &mut *sums, step).map_err(error)?;
        // `d̃ = start − iterate`, then `d = U_G d̃ U_Aᵀ` (or `d̃ U_Aᵀ`), masked.
        d.axpy(start, -1.0, iterate).map_err(error)?;
        match output.as_ref() {
            Some(g) => {
                d.gemm(scratch, 1.0, start, Op::N, input, Op::T, 0.0, arithmetic).map_err(error)?;
                d.gemm(moved, 1.0, g, Op::N, scratch, Op::N, 0.0, arithmetic).map_err(error)?;
            }
            None => d.gemm(moved, 1.0, start, Op::N, input, Op::T, 0.0, arithmetic).map_err(error)?,
        }
        d.hadamard(scratch, moved, live, false).map_err(error)?;
        d.set_rows(&mut self.mean[i], 0, before).map_err(error)?;
        d.axpy(&mut self.mean[i], -1.0, scratch).map_err(error)
    }

    /// Operator `i`'s posterior means `μ̄` on the host.
    fn host_mean(&self, i: usize) -> Result<Array2<f64>, String> {
        self.fitting.download(&self.average[i]).map_err(error)
    }

    /// The trial step at which the iterate sits while a line step awaits its measurement.
    #[must_use]
    pub fn line_trial(&self) -> Option<f64> {
        self.line_state.as_ref().map(|_| self.ratio)
    }

    /// Puts a pending line step's iterate at `before − η d` (`η = 0`: where the step started), for
    /// its measurement at the step's draws and the deviations the step set.
    pub fn place_line(&mut self, eta: f64) -> Result<(), String> {
        let state = self.line_state.as_ref().ok_or_else(|| error("no line step awaits a measurement"))?;
        for ((mean, start), d) in self.mean.iter_mut().zip(&state.before).zip(&state.directions) {
            *mean = self.fitting.copy(start).map_err(error)?;
            self.fitting.axpy(mean, -eta, d).map_err(error)?;
        }
        Ok(())
    }

    /// Ends a line step from the data term per token in nats of a batch other than the step's
    /// (its own draws) at the iterate `before − η d` for `η = 0, η₀, 2η₀`, all at the deviations
    /// the step set (`DevicePosterior::place_line`). The step's direction came from its own batch,
    /// whose noise it follows, so its own batch measures a descent that the others do not have
    /// (vpd4l, one MLP block at 0.78M tokens, measured on the step's batch: the training data term
    /// rose 2.35 → 3.28 → 12.5 → 15.6 bits per token over steps 0–3); another batch measures `F`
    /// along `d` without that bias. The data term along `d` is the parabola through the three
    /// values, and with the prior's exact slope `Σ δ μ d` and curvature `Σ δ d²` the measured `F` is
    /// least at `η = (−D′(0) + Σ δ μ d) / (D″ + Σ δ d²)`, trusted up to twice the measured range;
    /// where the measured curvature is not positive the step is the measured point of least `F`.
    /// The next trial is this step (half the trial after none). No rate: the step's length is
    /// measured on the objective, including the entries' joint moves that the diagonal curvature
    /// `h` omits.
    pub fn finish_line(&mut self, [zero, one, two]: [f64; 3], beta2: f64) -> Result<LineReport, String> {
        let state = self.line_state.take().ok_or_else(|| error("no line step awaits a measurement"))?;
        let trial = self.ratio;
        let c = (zero - 2.0 * one + two) / (2.0 * trial * trial);
        let descent = (3.0 * zero - 4.0 * one + two) / (2.0 * trial);
        let (slope, curvature) = (descent + state.slope_prior, 2.0 * c + state.prior_curvature);
        let objective = |eta: f64, data: f64| data - eta * state.slope_prior + 0.5 * eta * eta * state.prior_curvature;
        let eta = if !(slope.is_finite() && curvature.is_finite()) {
            0.0
        } else if curvature > 0.0 {
            (slope / curvature).clamp(0.0, 4.0 * trial)
        } else {
            [(0.0, zero), (trial, one), (2.0 * trial, two)].into_iter().map(|(eta, data)| (eta, objective(eta, data))).fold((0.0, f64::INFINITY), |best, x| if x.1 < best.1 { x } else { best }).0
        };
        for ((mean, start), d) in self.mean.iter_mut().zip(&state.before).zip(&state.directions) {
            *mean = self.fitting.copy(start).map_err(error)?;
            self.fitting.axpy(mean, -eta, d).map_err(error)?;
        }
        self.ratio = if eta > 0.0 { eta } else { 0.5 * trial };
        self.ratio_steps += 1;
        self.average_and_refresh(beta2)?;
        Ok(LineReport { eta, trial, measured: 2.0 * c, draw: state.draw_curvature, diagonal: state.diagonal, own_slope: state.slope_data, slope: descent })
    }

    /// Ends a pending line step at `η` along its direction without a measurement: `η = 0` for a
    /// caller that moves no mean (a pass that only sets the deviations), or a fraction a test
    /// fixes. A fit measures its steps ([`DevicePosterior::finish_line`]).
    pub fn settle_line(&mut self, eta: f64, beta2: f64) -> Result<(), String> {
        let state = self.line_state.take().ok_or_else(|| error("no line step awaits a measurement"))?;
        for ((mean, start), d) in self.mean.iter_mut().zip(&state.before).zip(&state.directions) {
            *mean = self.fitting.copy(start).map_err(error)?;
            self.fitting.axpy(mean, -eta, d).map_err(error)?;
        }
        self.average_and_refresh(beta2)
    }

    /// Operator `i`'s direction `d = before − μ` after the kernel's full step, with its terms added
    /// into `sums` (column 1 per group: `g · d`, `Σ d²`, `μ · d`, `u · d`, `Σ h d²`).
    fn line_terms(&self, i: usize, before: &Tensor, (gradient, draw): (&Tensor, &Tensor), sums: &mut [Tensor]) -> Result<Tensor, String> {
        let mut d = self.fitting.copy(before).map_err(error)?;
        self.fitting.axpy(&mut d, -1.0, &self.mean[i]).map_err(error)?;
        let mut weighted = self.fitting.empty(d.rows(), d.cols()).map_err(error)?;
        self.fitting.hadamard(&mut weighted, &self.moments[i][1], &d, false).map_err(error)?;
        let [along_g, square_sums, along_mean, along_u, diagonal] = sums else { return Err(error("five sums")) };
        let s = &self.log_sd[i];
        self.fitting.group_curvature((gradient, &d, s), &self.groups[i], along_g).map_err(error)?;
        self.fitting.group_curvature((&d, &d, s), &self.groups[i], square_sums).map_err(error)?;
        self.fitting.group_curvature((before, &d, s), &self.groups[i], along_mean).map_err(error)?;
        self.fitting.group_curvature((draw, &d, s), &self.groups[i], along_u).map_err(error)?;
        self.fitting.group_curvature((&weighted, &d, s), &self.groups[i], diagonal).map_err(error)?;
        Ok(d)
    }

    /// The posterior's mean, the iterate's Polyak average (uniform over the steps since the
    /// posterior was set, then over about one epoch; module note), and the groups' variances and
    /// divergences at it.
    fn average_and_refresh(&mut self, beta2: f64) -> Result<(), String> {
        self.averaged += 1;
        let weight = (1.0 / self.averaged as f64).max(1.0 - beta2);
        for (average, mean) in self.average.iter_mut().zip(&self.mean) {
            let mut difference = self.fitting.copy(mean).map_err(error)?;
            self.fitting.axpy(&mut difference, -1.0, average).map_err(error)?;
            self.fitting.axpy(average, weight, &difference).map_err(error)?;
        }
        self.sums = self.wide.zeros(self.sums.rows(), 3).map_err(error)?;
        self.refresh()
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
    fn sample_of(&self, program: &mut DeviceProgram, key: u64, means: &[Tensor]) -> Result<(), String> {
        for (i, &op) in self.operators.iter().enumerate() {
            let parts = (&means[i], &self.log_sd[i]);
            if let Ok(theta) = program.dense_mut(op) {
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
        if self.line_state.is_some() {
            return Err(error("a line step awaits its measurement"));
        }
        self.uploaded.clear();
        self.steps += 1;
        // The kernel takes IVON's full direction from the iterate kept here.
        let before: Vec<Tensor> = self.mean.iter().map(|m| self.fitting.copy(m).map_err(error)).collect::<Result<_, _>>()?;
        let mut sums: Vec<Tensor> = (0..5).map(|_| self.wide.zeros(self.group_count(), 3).map_err(error)).collect::<Result<_, _>>()?;
        let mut directions = Vec::with_capacity(before.len());
        // The groups' prior precisions as the step's kernel reads them, for the preconditioner.
        let variances = if self.directions.iter().any(Option::is_some) { self.variances()? } else { Vec::new() };
        let operators = self.operators.clone();
        for (i, &op) in operators.iter().enumerate() {
            let zero = |given: Option<&Tensor>| -> Result<Option<Tensor>, String> {
                match given {
                    Some(_) => Ok(None),
                    None => self.fitting.zeros(self.mean[i].rows(), self.mean[i].cols()).map(Some).map_err(error),
                }
            };
            let (missing_gradient, missing_factor) = (zero(gradients.get(&op))?, zero(factor.0.get(&op))?);
            let gradient = gradients.get(&op).or(missing_gradient.as_ref()).ok_or_else(|| error("no gradient"))?;
            let draw = factor.0.get(&op).or(missing_factor.as_ref()).ok_or_else(|| error("no Gauss–Newton factor"))?;
            let step = PosteriorStep { gradient_scale: scale, factor_scale: factor.1, tokens: self.tokens, beta1: ivon.beta1, beta2: ivon.beta2, step: self.steps };
            let [momentum, curvature, power] = &mut self.moments[i];
            self.fitting
                .posterior_ivon((&mut self.mean[i], &mut self.log_sd[i]), [momentum, curvature, power], (gradient, draw), (&self.groups[i], &self.variance), &mut self.sums, &step)
                .map_err(error)?;
            // Along the operator's input factor, where it has one, the direction is the
            // preconditioned move; σ and h stay as the kernel set them along the own axes.
            self.precondition(i, &before[i], (gradient, draw), &step, &variances)?;
            directions.push(self.line_terms(i, &before[i], (gradient, draw), &mut sums)?);
        }
        {
            // The step's terms along `d`; the iterate waits at the trial step for its measurement
            // (`DevicePosterior::finish_line`), which also averages and refreshes.
            let variances = self.variances()?;
            let column = |t: &Tensor| -> Result<Vec<f64>, String> { Ok(self.wide.download(t).map_err(error)?.column(1).to_vec()) };
            let precision = |g: usize| if variances[g] > 0.0 { 1.0 / (self.tokens * variances[g]) } else { 0.0 };
            let weighted = |values: Vec<f64>| -> f64 { values.iter().enumerate().map(|(g, x)| precision(g) * x).sum() };
            let along_u: f64 = column(&sums[3])?.iter().sum();
            let state = LineState {
                slope_data: scale * column(&sums[0])?.iter().sum::<f64>(),
                prior_curvature: weighted(column(&sums[1])?),
                slope_prior: weighted(column(&sums[2])?),
                draw_curvature: factor.1 * along_u * along_u,
                diagonal: column(&sums[4])?.iter().sum(),
                before,
                directions,
            };
            for ((mean, start), d) in self.mean.iter_mut().zip(&state.before).zip(&state.directions) {
                *mean = self.fitting.copy(start).map_err(error)?;
                self.fitting.axpy(mean, -self.ratio, d).map_err(error)?;
            }
            self.line_state = Some(state);
        }
        Ok(())
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

    /// The line arm's next trial step and the line steps taken.
    #[must_use]
    pub fn line_ratio(&self) -> (f64, u64) {
        (self.ratio, self.ratio_steps)
    }

    /// The line arm's next trial step and its steps, restored from a checkpoint.
    pub fn set_line_ratio(&mut self, (ratio, steps): (f64, u64)) {
        (self.ratio, self.ratio_steps) = (ratio, steps);
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
    /// With the identity as an operator's input factor, the preconditioned direction is IVON's
    /// own: the kernel's second pass along `U = I`, with the one group's precision and the same
    /// starting state, repeats the first. With a rotation it differs.
    #[test]
    fn an_identity_input_factor_leaves_the_step_direction() {
        const R: usize = 4;
        const C: usize = 6;
        let device = Device::host();
        let mean = Array2::from_shape_fn((R, C), |(r, c)| 0.1 * (r as f64 - c as f64));
        let log_sd = Array2::from_elem((R, C), -3.0);
        let groups = vec![vec![0u32; R * C]];
        let parts = Parts { operators: &[0], mean: std::slice::from_ref(&mean), log_sd: std::slice::from_ref(&log_sd), groups: &groups, count: 1 };
        let ivon = Ivon { beta1: 0.9, beta2: 0.75 };
        let turned = {
            let (c, s) = (0.6_f64, 0.8_f64);
            let mut u = Array2::<f64>::eye(C);
            u[[0, 0]] = c;
            u[[0, 1]] = -s;
            u[[1, 0]] = s;
            u[[1, 1]] = c;
            u
        };
        let run = |factor: Option<Array2<f64>>| -> Vec<Array2<f64>> {
            let mut posterior = DevicePosterior::from_parts(&device, &parts, 500.0, None, 0).unwrap();
            if let Some(u) = factor {
                // Eigenvalues whose Kronecker start equals the diagonal's: every entry of the
                // starting curvature is the same, so `g_r λ_k = h_rk` along any `U`.
                let values = vec![1.0; C];
                posterior.set_directions(&[Some(std::sync::Arc::new(crate::library_mdl::InputFactor { vectors: u, values }))], &[None]).unwrap();
            }
            let mut iterates = Vec::new();
            for t in 0..3u64 {
                let gradient = Array2::from_shape_fn((R, C), |(r, c)| f64::from(posterior_normal(21, t, (r * C + c) as u64)));
                let draw = Array2::from_shape_fn((R, C), |(r, c)| f64::from(posterior_normal(22, t, (r * C + c) as u64)));
                let gradients = BTreeMap::from([(0, device.upload(gradient.view()).unwrap())]);
                let draws = BTreeMap::from([(0, device.upload(draw.view()).unwrap())]);
                posterior.step(&gradients, 0.01, (&draws, 0.01), &ivon).unwrap();
                iterates.push(device.download(&posterior.mean[0]).unwrap());
                posterior.settle_line(0.5, ivon.beta2).unwrap();
            }
            iterates
        };
        let (own, identity, rotated) = (run(None), run(Some(Array2::eye(C))), run(Some(turned)));
        for (t, (a, b)) in own.iter().zip(&identity).enumerate() {
            let gap = a.iter().zip(b).fold(0.0_f64, |m, (x, y)| m.max((x - y).abs()));
            assert!(gap <= 1e-12, "step {t}: the identity factor moved the direction by {gap}");
        }
        let moved = own.iter().zip(&rotated).skip(1).any(|(a, b)| a.iter().zip(b).any(|(x, y)| (x - y).abs() > 1e-9));
        assert!(moved, "a rotated input factor left the direction unchanged");
    }

    /// A line step on a quadratic data term `D(μ) = ½ a Σ (μ − t)²` (gradient `a (μ − t)` per
    /// token, measured exactly at the iterate) lands on the minimum along its direction of
    /// `D + Σ δ μ² / 2`: the parabola through the two measurements is exact.
    #[test]
    fn a_line_step_on_a_quadratic_lands_on_the_minimum_along_its_direction() {
        const R: usize = 16;
        let device = Device::host();
        let tokens = 1000.0;
        let a = 3.0;
        let target = Array2::from_shape_fn((1, R), |(_, i)| 0.2 + 0.05 * i as f64);
        let mean = Array2::from_elem((1, R), 0.1);
        let log_sd = Array2::from_elem((1, R), -4.0);
        let groups = vec![vec![0u32; R]];
        let parts = Parts { operators: &[0], mean: std::slice::from_ref(&mean), log_sd: std::slice::from_ref(&log_sd), groups: &groups, count: 1 };
        let mut posterior = DevicePosterior::from_parts(&device, &parts, tokens, None, 0).unwrap();
        let ivon = Ivon { beta1: 0.9, beta2: 0.75 };
        let data = |mu: &Array2<f64>| 0.5 * a * mu.iter().zip(&target).map(|(m, t)| (m - t) * (m - t)).sum::<f64>();
        let factor = device.upload(Array2::from_elem((1, R), a.sqrt()).view()).unwrap();
        for step in 0..2 {
            let mu = device.download(&posterior.mean[0]).unwrap();
            let gradient = Array2::from_shape_fn((1, R), |(_, i)| a * (mu[(0, i)] - target[(0, i)]));
            let gradients = BTreeMap::from([(0, device.upload(gradient.view()).unwrap())]);
            let factors = BTreeMap::from([(0, device.copy(&factor).unwrap())]);
            posterior.step(&gradients, 1.0, (&factors, 1.0), &ivon).unwrap();
            let trial = posterior.line_trial().expect("a pending line step");
            let delta = 1.0 / (tokens * posterior.variances().unwrap()[0]);
            let state = posterior.line_state.as_ref().unwrap();
            let (start, d) = (device.download(&state.before[0]).unwrap(), device.download(&state.directions[0]).unwrap());
            let mut values = [0.0; 3];
            for (k, value) in values.iter_mut().enumerate() {
                posterior.place_line(trial * k as f64).unwrap();
                *value = data(&device.download(&posterior.mean[0]).unwrap());
            }
            let report = posterior.finish_line(values, ivon.beta2).unwrap();
            let slope: f64 = start.iter().zip(&d).zip(&target).map(|((m, x), t)| (a * (m - t) + delta * m) * x).sum();
            let curvature: f64 = d.iter().map(|x| (a + delta) * x * x).sum();
            if step == 0 {
                // The first step has one gradient, no measured spread, no direction.
                assert_eq!(report.eta, 0.0);
                continue;
            }
            let exact = (slope / curvature).clamp(0.0, 4.0 * trial);
            assert!(curvature > 0.0 && exact > 0.0, "no descent along d: slope {slope}, curvature {curvature}");
            assert!((report.eta - exact).abs() <= 1e-9 * exact, "η {} against the minimum {exact}", report.eta);
            let landed = device.download(&posterior.mean[0]).unwrap();
            for ((x, m), dx) in landed.iter().zip(&start).zip(&d) {
                assert!((x - (m - exact * dx)).abs() <= 1e-12, "the iterate is not at the step");
            }
        }
    }

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
            // A tenth of IVON's direction, as a line step whose measurement the test does not model.
            posterior.settle_line(0.1, ivon.beta2).unwrap();
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
