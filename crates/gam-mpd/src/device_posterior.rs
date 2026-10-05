//! The library fit's posterior resident on the device (#2951): the means `μ`, log standard
//! deviations `s`, the gradient's momentum and the curvature estimate of every trainable operator
//! stay on the device from step to step, so a step moves no parameter between host and device.
//!
//! A step writes the weight sample `θ = μ + exp(s) ε` into the explanation's program
//! ([`Device::reparameterize`], `ε` regenerated from its counter), runs the experiments, and takes
//! the improved variational online Newton step (IVON, [`Device::posterior_ivon`]) with the data
//! term's curvature in the Gauss–Newton approximation, from the gradient the program's reverse
//! pass left on the device and a draw of the Gauss–Newton factor (`interchange::Factor`, a second
//! reverse pass from labels drawn from the explanation's own predictions), which also sums each
//! prior group's new moments; the groups' variances and divergences follow from those sums
//! ([`Device::group_divergence`]). The posterior and the curvature are held in the fitting storage
//! (f32 on CUDA and the Apple GPU, float64 on the host); the momentum in bfloat16 where the masters
//! are f32 on CUDA (rounded once as it is stored, its update computed in f32), else in the fitting
//! storage; the group sums in float64 where the backend holds it (CUDA, the host). The objective is
//! `library_mdl`'s (module note there): `KL(q_G ‖ p_G) = ½ (|G| ln v_G − Σ 2s)` at the
//! empirical-Bayes variance `v_G`, whose prior precision per token `1 / (N v_G)` is IVON's weight
//! decay.

use crate::{
    device_program::DeviceProgram,
    library_mdl::{Explanation, Posterior},
};
use gam_gpu::{
    gpu_error::GpuError,
    tensor::{Device, Indices, PosteriorStep, Storage, Tensor},
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
}

/// The posterior of a library explanation's trainable operators, on the device.
pub struct DevicePosterior {
    /// The device holding the group sums (float64 where the backend holds it), and the one holding
    /// the posterior, its samples and gradients.
    wide: Device,
    fitting: Device,
    /// Per trainable operator (`Explanation::trainable` order): its id, `μ`, `s`, IVON's state
    /// (the gradient's momentum, then the curvature estimate) and each entry's group.
    operators: Vec<usize>,
    mean: Vec<Tensor>,
    log_sd: Vec<Tensor>,
    moments: Vec<[Tensor; 2]>,
    groups: Vec<Indices>,
    /// Per group `(n, Σ μ² + σ², Σ 2s)` being summed, its variance and its divergence in nats.
    sums: Tensor,
    variance: Tensor,
    divergence: Tensor,
    /// The training tokens `N` (the data term's weight) and the steps taken.
    tokens: f64,
    steps: u64,
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
    /// gradient's momentum and the curvature estimate; when `None`, momentum zero and the
    /// curvature at which the posterior's standard deviations are IVON's) after `steps` steps.
    pub fn new(fitting: &Device, explanation: &Explanation, posterior: &Posterior, tokens: f64, moments: Option<&[[Array2<f64>; 2]]>, steps: u64) -> Result<Self, String> {
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
    pub fn from_parts(fitting: &Device, parts: &Parts<'_>, tokens: f64, moments: Option<&[[Array2<f64>; 2]]>, steps: u64) -> Result<Self, String> {
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
        if shapes.len() != parts.operators.len() || !sizes_agree || moments.is_some_and(|m| m.len() != shapes.len()) {
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
                Some(parts.log_sd.iter().zip(parts.groups).map(|(log_sd, groups)| [Array2::zeros(log_sd.dim()), curvature(log_sd, groups)]).collect::<Vec<_>>())
            }
        };
        let moments = moments.or(start.as_deref());
        let up = |m: &Array2<f64>| master.upload(m.view()).map_err(error);
        let moment = |m: &Array2<f64>| narrow.upload(m.view()).map_err(error);
        let mut out = Self {
            sums: wide.zeros(parts.count, 3).map_err(error)?,
            variance: wide.zeros(parts.count, 1).map_err(error)?,
            divergence: wide.zeros(parts.count, 1).map_err(error)?,
            mean: parts.mean.iter().map(up).collect::<Result<_, _>>()?,
            log_sd: parts.log_sd.iter().map(up).collect::<Result<_, _>>()?,
            moments: moments.ok_or_else(|| error("no posterior state"))?.iter().map(|m| Ok([moment(&m[0])?, up(&m[1])?])).collect::<Result<_, String>>()?,
            groups: parts.groups.iter().map(|ids| master.upload_indices(ids).map_err(error)).collect::<Result<_, _>>()?,
            operators: parts.operators.to_vec(),
            fitting: fitting.clone(),
            wide,
            tokens,
            steps,
        };
        out.refresh()?;
        Ok(out)
    }

    /// The groups' variances and divergences from the posterior as it stands.
    fn refresh(&mut self) -> Result<(), String> {
        for ((mean, log_sd), groups) in self.mean.iter().zip(&self.log_sd).zip(&self.groups) {
            self.fitting.group_moments((mean, log_sd), groups, &mut self.sums).map_err(error)?;
        }
        self.wide.group_divergence(&mut self.sums, &mut self.variance, &mut self.divergence).map_err(error)
    }

    /// The steps taken.
    #[must_use]
    pub fn steps(&self) -> u64 {
        self.steps
    }

    /// Writes the posterior means into `program`'s trainable operators (rounded to its storage).
    pub fn mean_into(&self, program: &mut DeviceProgram) -> Result<(), String> {
        for (i, &op) in self.operators.iter().enumerate() {
            // A program holding the operator in bfloat16 (`DeviceProgram::hold_bf16`) gets it so.
            let bf16 = program.dense(op).is_ok_and(|held| held.storage() == Storage::Bf16);
            let value = if bf16 { self.fitting.bf16_copy(&self.mean[i]) } else { self.fitting.copy(&self.mean[i]) };
            program.replace_dense_parameter(op, value.map_err(error)?)?;
        }
        program.refresh_fused()
    }

    /// Writes the weight sample of `key` into `program`'s trainable operators (operator `i` in
    /// `Explanation::trainable` order draws stream `i`); in place where the program holds the
    /// operator in one role, replaced where it also holds a column copy (a bias).
    pub fn sample_into(&self, program: &mut DeviceProgram, key: u64) -> Result<(), String> {
        for (i, &op) in self.operators.iter().enumerate() {
            let parts = (&self.mean[i], &self.log_sd[i]);
            if let Ok(theta) = program.dense_mut(op) {
                self.fitting.reparameterize(theta, parts, (key, i as u64)).map_err(error)?;
            } else {
                let (rows, cols) = (self.mean[i].rows(), self.mean[i].cols());
                let mut theta = self.fitting.zeros(rows, cols).map_err(error)?;
                self.fitting.reparameterize(&mut theta, parts, (key, i as u64)).map_err(error)?;
                program.replace_dense_parameter(op, theta)?;
            }
        }
        program.refresh_fused()
    }

    /// One IVON step: `gradients` holds per trainable operator (by id) the gradient of the batch's
    /// data term at the sample, which `scale` turns into an unbiased estimate of the collection's
    /// gradient per token in nats (`B / N` for one of `B` batches of a collection of `N` scored
    /// tokens, times the conversion from bits); `factor` holds per operator a draw of the
    /// Gauss–Newton factor and the factor (`B / N`) turning its square into the curvature estimate
    /// per token. An operator the batch does not reach has neither, and its step takes the
    /// prior's alone.
    pub fn step(&mut self, gradients: &BTreeMap<usize, Tensor>, scale: f64, factor: (&BTreeMap<usize, Tensor>, f64), ivon: &Ivon) -> Result<(), String> {
        self.steps += 1;
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
            let step = PosteriorStep { gradient_scale: scale, factor_scale: factor.1, tokens: self.tokens, rate: ivon.rate, beta1: ivon.beta1, beta2: ivon.beta2, step: self.steps };
            let [momentum, curvature] = &mut self.moments[i];
            self.fitting
                .posterior_ivon((&mut self.mean[i], &mut self.log_sd[i]), [momentum, curvature], (gradient, draw), (&self.groups[i], &self.variance), &mut self.sums, &step)
                .map_err(error)?;
        }
        self.wide.group_divergence(&mut self.sums, &mut self.variance, &mut self.divergence).map_err(error)
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
    pub fn download(&self, posterior: &mut Posterior) -> Result<Vec<[Array2<f64>; 2]>, String> {
        self.values_into(posterior)?;
        let down = |t: &Tensor| self.fitting.download(t).map_err(error);
        self.moments.iter().map(|m| Ok([down(&m[0])?, down(&m[1])?])).collect()
    }

    /// The posterior's means and log standard deviations into `posterior`; IVON's state stays on
    /// the device.
    pub fn values_into(&self, posterior: &mut Posterior) -> Result<(), String> {
        let down = |t: &Tensor| self.fitting.download(t).map_err(error);
        for i in 0..self.mean.len() {
            posterior.mean[i] = down(&self.mean[i])?;
            posterior.log_sd[i] = down(&self.log_sd[i])?;
        }
        Ok(())
    }

    /// `posterior`'s means and log standard deviations onto the device, IVON's state kept, and
    /// the groups' variances and divergences with them (after a removal on the host, whose removed
    /// entries, `μ = 0` and `s = −∞`, every later step leaves alone).
    pub fn set_values(&mut self, posterior: &Posterior) -> Result<(), String> {
        if posterior.mean.len() != self.mean.len() {
            return Err(error("one posterior array per trainable operator required"));
        }
        for i in 0..self.mean.len() {
            self.mean[i] = self.fitting.upload(posterior.mean[i].view()).map_err(error)?;
            self.log_sd[i] = self.fitting.upload(posterior.log_sd[i].view()).map_err(error)?;
        }
        self.refresh()
    }

    /// Trainable operator `i`'s `μ` and `s` on the host.
    pub fn values(&self, i: usize) -> Result<(Array2<f64>, Array2<f64>), String> {
        Ok((self.fitting.download(&self.mean[i]).map_err(error)?, self.fitting.download(&self.log_sd[i]).map_err(error)?))
    }

    /// Trainable operator `i`'s `μ`, `s` and IVON's state on the host, one operator at a time (a
    /// checkpoint streams them rather than holding every operator's state at once).
    pub fn operator(&self, i: usize) -> Result<(Array2<f64>, Array2<f64>, [Array2<f64>; 2]), String> {
        let down = |t: &Tensor| self.fitting.download(t).map_err(error);
        let m = self.moments.get(i).ok_or_else(|| error("no such trainable operator"))?;
        Ok((down(&self.mean[i])?, down(&self.log_sd[i])?, [down(&m[0])?, down(&m[1])?]))
    }

}
