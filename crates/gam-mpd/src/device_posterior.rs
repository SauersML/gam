//! The library fit's posterior resident on the device (#2951): the means `μ`, log standard
//! deviations `s` and Adam's moments of every trainable operator stay on the device from step to
//! step, so a step moves no parameter between host and device.
//!
//! A step writes the weight sample `θ = μ + exp(s) ε` into the explanation's program
//! ([`Device::reparameterize`], `ε` regenerated from its counter), runs the experiments, and takes
//! Adam's step from the gradient the program's reverse pass left on the device
//! ([`Device::posterior_adam`]), which also sums each prior group's new moments; the groups'
//! variances and divergences follow from those sums ([`Device::group_divergence`]). The posterior
//! and its moments are held in the fitting storage (f32 on CUDA and the Apple GPU, float64 on the
//! host), the group sums in float64 where the backend holds it (CUDA, the host). The objective and its derivatives are those of
//! `library_mdl` (module note there): `KL(q_G ‖ p_G) = ½ (|G| ln v_G − Σ 2s)` at the empirical-Bayes
//! variance `v_G`.

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

/// Adam's settings of a step ([`DevicePosterior::step`]).
#[derive(Clone, Copy, Debug)]
pub struct Adam {
    pub mean_rate: f64,
    pub log_sd_rate: f64,
    pub beta1: f64,
    pub beta2: f64,
    pub epsilon: f64,
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
    /// Per trainable operator (`Explanation::trainable` order): its id, `μ`, `s`, Adam's moments
    /// (`μ`'s first and second, then `s`'s) and each entry's group.
    operators: Vec<usize>,
    mean: Vec<Tensor>,
    log_sd: Vec<Tensor>,
    moments: Vec<[Tensor; 4]>,
    groups: Vec<Indices>,
    /// Per group `(n, Σ μ² + σ², Σ 2s)` being summed, its variance and its divergence in nats.
    sums: Tensor,
    variance: Tensor,
    divergence: Tensor,
    /// Adam's steps taken.
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
    /// `posterior` of `explanation` on `fitting` (the device whose storage the explanation's
    /// program runs in), with Adam's `moments` per operator (`μ`'s first and second, then `s`'s;
    /// zero when `None`) after `steps` steps.
    pub fn new(fitting: &Device, explanation: &Explanation, posterior: &Posterior, moments: Option<&[[Array2<f64>; 4]]>, steps: u64) -> Result<Self, String> {
        let shapes: Vec<(usize, usize)> = posterior.mean.iter().map(Array2::dim).collect();
        if shapes.len() != explanation.trainable.len() {
            return Err(error("one posterior array per trainable operator required"));
        }
        let groups = membership(explanation, &shapes)?;
        let parts = Parts { operators: &explanation.trainable, mean: &posterior.mean, log_sd: &posterior.log_sd, groups: &groups, count: explanation.groups.len() };
        Self::from_parts(fitting, &parts, moments, steps)
    }

    /// The posterior of the trainable operators `parts.operators` of a program, each entry in group
    /// `parts.groups[i][entry]` (row-major) of `parts.count`, on `fitting`, with Adam's `moments`
    /// (zero when `None`) after `steps` steps.
    pub fn from_parts(fitting: &Device, parts: &Parts<'_>, moments: Option<&[[Array2<f64>; 4]]>, steps: u64) -> Result<Self, String> {
        let wide = match fitting.with_storage(Storage::F64) {
            Ok(wide) => wide,
            Err(GpuError::NoDeviceKernel { .. }) => fitting.clone(),
            Err(e) => return Err(error(e)),
        };
        let master = fitting.clone();
        let shapes: Vec<(usize, usize)> = parts.mean.iter().map(Array2::dim).collect();
        let sizes_agree = parts.log_sd.iter().map(Array2::dim).eq(shapes.iter().copied()) && parts.groups.iter().map(Vec::len).eq(shapes.iter().map(|(r, c)| r * c));
        if shapes.len() != parts.operators.len() || !sizes_agree || moments.is_some_and(|m| m.len() != shapes.len()) {
            return Err(error("one posterior array and group list per trainable operator required"));
        }
        if parts.groups.iter().flatten().any(|g| *g as usize >= parts.count) {
            return Err(error("a group id beyond the groups"));
        }
        let up = |m: &Array2<f64>| master.upload(m.view()).map_err(error);
        let mut out = Self {
            sums: wide.zeros(parts.count, 3).map_err(error)?,
            variance: wide.zeros(parts.count, 1).map_err(error)?,
            divergence: wide.zeros(parts.count, 1).map_err(error)?,
            mean: parts.mean.iter().map(up).collect::<Result<_, _>>()?,
            log_sd: parts.log_sd.iter().map(up).collect::<Result<_, _>>()?,
            moments: match moments {
                Some(given) => given.iter().map(|m| Ok([up(&m[0])?, up(&m[1])?, up(&m[2])?, up(&m[3])?])).collect::<Result<_, String>>()?,
                None => shapes
                    .iter()
                    .map(|(r, c)| Ok([master.zeros(*r, *c).map_err(error)?, master.zeros(*r, *c).map_err(error)?, master.zeros(*r, *c).map_err(error)?, master.zeros(*r, *c).map_err(error)?]))
                    .collect::<Result<_, String>>()?,
            },
            groups: parts.groups.iter().map(|ids| master.upload_indices(ids).map_err(error)).collect::<Result<_, _>>()?,
            operators: parts.operators.to_vec(),
            fitting: fitting.clone(),
            wide,
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

    /// Adam's steps taken.
    #[must_use]
    pub fn steps(&self) -> u64 {
        self.steps
    }

    /// Writes the posterior means into `program`'s trainable operators (rounded to its storage).
    pub fn mean_into(&self, program: &mut DeviceProgram) -> Result<(), String> {
        for (i, &op) in self.operators.iter().enumerate() {
            program.replace_dense_parameter(op, self.fitting.copy(&self.mean[i]).map_err(error)?)?;
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

    /// One Adam step from the gradient of the data term at the sample of `key`: `gradients` holds
    /// per trainable operator (by id) the gradient of the batch's data term, which `scale` weighs
    /// (the training tokens over the batch's, and the conversion to nats); an operator the batch
    /// does not reach has none, and its step takes the prior's derivatives alone.
    pub fn step(&mut self, gradients: &BTreeMap<usize, Tensor>, scale: f64, adam: &Adam, key: u64) -> Result<(), String> {
        self.steps += 1;
        for (i, &op) in self.operators.iter().enumerate() {
            let zero;
            let gradient = match gradients.get(&op) {
                Some(g) => g,
                None => {
                    zero = self.fitting.zeros(self.mean[i].rows(), self.mean[i].cols()).map_err(error)?;
                    &zero
                }
            };
            let step = PosteriorStep {
                gradient_scale: scale,
                mean_rate: adam.mean_rate,
                log_sd_rate: adam.log_sd_rate,
                beta1: adam.beta1,
                beta2: adam.beta2,
                epsilon: adam.epsilon,
                step: self.steps,
                key,
                stream: i as u64,
            };
            let [m0, m1, m2, m3] = &mut self.moments[i];
            self.fitting
                .posterior_adam((&mut self.mean[i], &mut self.log_sd[i]), [m0, m1, m2, m3], gradient, (&self.groups[i], &self.variance), &mut self.sums, &step)
                .map_err(error)?;
        }
        self.wide.group_divergence(&mut self.sums, &mut self.variance, &mut self.divergence).map_err(error)
    }

    /// Per group, `KL(q_G ‖ p_G)` in nats at the posterior as it stands (zero for a removed group).
    pub fn divergences(&self) -> Result<Vec<f64>, String> {
        Ok(self.wide.download(&self.divergence).map_err(error)?.into_iter().collect())
    }

    /// The posterior's means and log standard deviations into `posterior`, and Adam's moments per
    /// operator.
    pub fn download(&self, posterior: &mut Posterior) -> Result<Vec<[Array2<f64>; 4]>, String> {
        self.values_into(posterior)?;
        let down = |t: &Tensor| self.fitting.download(t).map_err(error);
        self.moments.iter().map(|m| Ok([down(&m[0])?, down(&m[1])?, down(&m[2])?, down(&m[3])?])).collect()
    }

    /// The posterior's means and log standard deviations into `posterior`; Adam's moments stay on
    /// the device.
    pub fn values_into(&self, posterior: &mut Posterior) -> Result<(), String> {
        let down = |t: &Tensor| self.fitting.download(t).map_err(error);
        for i in 0..self.mean.len() {
            posterior.mean[i] = down(&self.mean[i])?;
            posterior.log_sd[i] = down(&self.log_sd[i])?;
        }
        Ok(())
    }

    /// `posterior`'s means and log standard deviations onto the device, Adam's moments kept, and
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

    /// Trainable operator `i`'s `μ`, `s` and Adam's moments on the host, one operator at a time
    /// (a checkpoint streams them rather than holding every operator's moments at once).
    pub fn operator(&self, i: usize) -> Result<(Array2<f64>, Array2<f64>, [Array2<f64>; 4]), String> {
        let down = |t: &Tensor| self.fitting.download(t).map_err(error);
        let m = self.moments.get(i).ok_or_else(|| error("no such trainable operator"))?;
        Ok((down(&self.mean[i])?, down(&self.log_sd[i])?, [down(&m[0])?, down(&m[1])?, down(&m[2])?, down(&m[3])?]))
    }

}
