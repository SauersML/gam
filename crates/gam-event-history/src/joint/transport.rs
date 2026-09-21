//! Fixed OU innovation coordinates of the joint state law.
//!
//! At fixed coefficients the state law of one history is the affine-Gaussian
//! [`StateLaw`] in standardized innovation coordinates. Per signature `k`:
//!
//! ```text
//! x_0 = m_0(g) + e_0
//! x_n = phi (x_(n-1) + j_(n-1)) + (1 - phi) m_n(g) + sqrt(1 - phi^2) e_n,    phi = exp(-r dt).
//! ```
//!
//! Here `m(g)` are the entry and drive regressions with genetic interactions,
//! affine in the missing scores, and `j_(n-1)` is the summed learned jump of the
//! marks fired at node `n - 1`. The OU density and its transport Jacobian cancel,
//! so `1/sqrt(V)` never appears, and nearly static and static signatures stay
//! representable.
//!
//! The state law's coefficient derivatives are contracted with posterior
//! innovation moments, and the transport's coefficient derivatives with node
//! forces, by reverse recursions. Neither divides by `V`, and no
//! nodes-by-coefficients Jacobian is formed. The transport routes are generic
//! over the scalar, so a rounding-bound tracker can instrument the same
//! arithmetic production evaluates in `f64`. Rate factors are formed once in
//! `f64` from the coefficient values and enter as constants.
use super::emission;
use super::law::{JointHistory, JointLikelihood, invalid, numerical};
use super::precision::{Cholesky, StateLaw, Step};
use crate::EventHistoryError;
use crate::scalar::{exp, ln, sqrt};
use gam_math::nested_dual::JetField;
use ndarray::{Array1, Array2};

/// `log((exp(x) - 1) / x)` for `x >= 0`, over any scalar. Its series
/// `x/2 + sum_k B_2k x^(2k) / (2k (2k)!)` alternates with decreasing terms for
/// `x < 2 pi`, so stopping after `x^2/24` omits at most `x^4 / 2880`. Below
/// `x = (1440 eps)^(1/3)` that omission is at most `eps x / 2`, within the one
/// rounding the series' final sum already charges, and the series is taken.
/// Above it the closed form `x + log(-expm1(-x)) - log(x)`, which never
/// overflows, is taken.
fn log_exprel<S: JetField>(x: &S) -> S {
    if x.value() <= (1440.0 * f64::EPSILON).cbrt() {
        x.scale(0.5).add(&x.mul(x).scale(1.0 / 24.0))
    } else {
        x.add(&ln(&emission::expm1(&x.neg()).neg())).sub(&ln(x))
    }
}

impl JointLikelihood {
    fn missing_genes(h: &JointHistory) -> Vec<usize> {
        h.genetics
            .iter()
            .enumerate()
            .filter_map(|(g, value)| value.is_none().then_some(g))
            .collect()
    }

    /// Every genetic score of the history, with the missing ones taken from
    /// the latent coordinates.
    fn gene_values<S: JetField>(h: &JointHistory, genes: &[S], like: &S) -> Vec<S> {
        let mut missing = genes.iter();
        h.genetics
            .iter()
            .map(|value| match value {
                Some(observed) => like.constant_like(*observed),
                None => missing
                    .next()
                    .cloned()
                    .unwrap_or_else(|| like.constant_like(f64::NAN)),
            })
            .collect()
    }

    /// Offset and missing-score loadings of one axis of a genetic-interaction
    /// regression `sum_j f_j (theta_j0 + sum_g theta_jg g)`, with the observed
    /// scores folded into the offset.
    fn regression_parts<S: JetField>(
        &self,
        theta: &[S],
        start: usize,
        columns: &[f64],
        h: &JointHistory,
        missing: &[usize],
        axis: usize,
    ) -> (S, Vec<S>) {
        let width = h.genetics.len() + 1;
        let zero = theta[0].constant_like(0.0);
        let mut offset = zero.clone();
        let mut loadings = vec![zero; missing.len()];
        for (j, &feature) in columns.iter().enumerate() {
            let base = start + (axis * columns.len() + j) * width;
            offset = offset.add(&theta[base].scale(feature));
            for (g, value) in h.genetics.iter().enumerate() {
                if let Some(value) = value {
                    offset = offset.add(&theta[base + g + 1].scale(feature).scale(*value));
                }
            }
            for (a, &g) in missing.iter().enumerate() {
                loadings[a] = loadings[a].add(&theta[base + g + 1].scale(feature));
            }
        }
        (offset, loadings)
    }

    /// Adds `constant` times the derivative of one regression axis's offset and
    /// `genes[a]` times the derivative of its missing-score loading `a` into the
    /// coefficient vector.
    fn regression_force<S: JetField>(
        &self,
        coefficients: &mut [S],
        start: usize,
        columns: &[f64],
        h: &JointHistory,
        missing: &[usize],
        axis: usize,
        constant: &S,
        genes: &[S],
    ) {
        let width = h.genetics.len() + 1;
        for (j, &feature) in columns.iter().enumerate() {
            let base = start + (axis * columns.len() + j) * width;
            coefficients[base] = coefficients[base].add(&constant.scale(feature));
            for (g, value) in h.genetics.iter().enumerate() {
                if let Some(value) = value {
                    coefficients[base + g + 1] =
                        coefficients[base + g + 1].add(&constant.scale(feature).scale(*value));
                }
            }
            for (a, &g) in missing.iter().enumerate() {
                coefficients[base + g + 1] = coefficients[base + g + 1].add(&genes[a].scale(feature));
            }
        }
    }

    /// Adds `force` times the derivative of the summed jump `j(fired)` on one
    /// axis: each fired mark with a jump contributes its count.
    fn jump_force<S: JetField>(&self, coefficients: &mut [S], fired: &[usize], axis: usize, force: &S) {
        for (d, range) in self.layout.jumps.iter().enumerate() {
            let count = fired.iter().filter(|&&e| e == d).count();
            if let Some(range) = range.as_ref().filter(|_| count > 0) {
                let index = range.start + axis;
                coefficients[index] = coefficients[index].add(&force.scale(count as f64));
            }
        }
    }

    /// `phi = exp(-r dt)`, `1 - phi` and `sqrt(1 - phi^2)` of one gap at the raw
    /// rate `r = softplus(raw)`, over any scalar: a running-error scalar charges
    /// each factor's rounding chain, and a jet carries its derivatives. Tied
    /// times give the identity transition exactly. The chain's charges for `exp`,
    /// `expm1` and `ln` are cited from the runtime libm; the charge for
    /// `emission::softplus` is measured until that function is composed over
    /// cited operations, so a band resting on these factors is labelled measured.
    fn transition<S: JetField>(raw: &S, earlier: f64, later: f64) -> Result<(S, S, S), EventHistoryError> {
        if later == earlier {
            return Ok((raw.constant_like(1.0), raw.constant_like(0.0), raw.constant_like(0.0)));
        }
        let decay = emission::softplus(raw).mul(&Self::gap(raw, earlier, later)).neg();
        let retained = exp(&decay);
        let weight = emission::expm1(&decay).neg();
        let variance = emission::expm1(&decay.scale(2.0)).neg();
        if !(variance.value() >= 0.0 && variance.value().is_finite() && weight.value().is_finite()) {
            return Err(numerical("joint OU transition is not representable"));
        }
        // A rate that underflows to zero leaves a static axis; its scale slope is
        // carried by the log-domain form of `rate_slopes`.
        let scale = if variance.value() == 0.0 {
            raw.constant_like(0.0)
        } else {
            sqrt(&variance)
        };
        Ok((retained, weight, scale))
    }

    /// `phi` and `1 - phi` of one gap at a raw rate.
    fn retention<S: JetField>(raw: &S, earlier: f64, later: f64) -> Result<(S, S), EventHistoryError> {
        let factors = Self::transition(raw, earlier, later)?;
        Ok((factors.0, factors.1))
    }

    /// The gap `later - earlier` between two node times, formed over S from its
    /// exact operands, so a running-error scalar charges its rounding.
    fn gap<S: JetField>(like: &S, earlier: f64, later: f64) -> S {
        like.constant_like(later).sub(&like.constant_like(earlier))
    }

    /// The complete state law of one history at fixed coefficients.
    pub(super) fn state_law(
        &self,
        theta: &[f64],
        h: &JointHistory,
    ) -> Result<StateLaw, EventHistoryError> {
        self.validate_parameters(theta)?;
        self.validate_history(h)?;
        let k = self.spec.signatures;
        let missing = Self::missing_genes(h);
        let m = missing.len();
        let precision = &self.spec.genetic_precision;
        let gene_precision =
            Array2::from_shape_fn((m, m), |(a, b)| precision[[missing[a], missing[b]]]);
        // Conditional mean mu_m - Lambda_mm^-1 Lambda_mo (o - mu_o).
        let mut information = Array1::<f64>::zeros(m);
        for (a, &ga) in missing.iter().enumerate() {
            for (gb, value) in h.genetics.iter().enumerate() {
                information[a] += precision[[ga, gb]]
                    * match value {
                        Some(observed) => self.spec.genetic_mean[gb] - observed,
                        None => self.spec.genetic_mean[gb],
                    };
            }
        }
        let gene_mean = Cholesky::new(&gene_precision)
            .ok_or_else(|| numerical("conditional genetic precision is not positive definite"))?
            .solve_vector(information.view());
        let entry_features = self.entry_features(h);
        let mut entry = Step {
            offset: Array1::zeros(k),
            genes: Array2::zeros((k, m)),
            innovation: Array2::eye(k),
        };
        for axis in 0..k {
            let (offset, loadings) = self.regression_parts(
                theta,
                self.layout.entry.start,
                &entry_features,
                h,
                &missing,
                axis,
            );
            entry.offset[axis] = offset;
            entry.genes.row_mut(axis).assign(&Array1::from(loadings));
        }
        let gaps = h.times.len() - 1;
        let mut transitions = Vec::with_capacity(gaps);
        let mut steps = Vec::with_capacity(gaps);
        for n in 1..h.times.len() {            let columns = h.drive_design.row(n - 1).to_vec();
            let mut phi = Array2::<f64>::zeros((k, k));
            let mut step = Step {
                offset: Array1::zeros(k),
                genes: Array2::zeros((k, m)),
                innovation: Array2::zeros((k, k)),
            };
            for axis in 0..k {
                let (retained, weight, scale) =
                    Self::transition(&theta[self.layout.rates.start + axis], h.times[n - 1], h.times[n])?;
                let (drive, loadings) = self.regression_parts(
                    theta,
                    self.layout.drive.start,
                    &columns,
                    h,
                    &missing,
                    axis,
                );
                phi[[axis, axis]] = retained;
                step.offset[axis] =
                    retained * self.jump(theta, &h.events[n - 1], axis) + weight * drive;
                step.genes
                    .row_mut(axis)
                    .assign(&(Array1::from(loadings) * weight));
                step.innovation[[axis, axis]] = scale;
            }
            transitions.push(phi);
            steps.push(step);
        }
        let law = StateLaw {
            gene_mean,
            gene_precision,
            entry,
            transitions,
            steps,
        };
        law.validate()?;
        Ok(law)
    }

    /// `log v` for `v = 1 - exp(-2 r dt)` and `r = softplus(raw)`, formed from
    /// `log r` so that it stays finite when `r` itself underflows:
    /// `log v = log(2 r dt) + log_exprel(2 r dt) - 2 r dt`.
    fn log_innovation_variance<S: JetField>(raw: &S, earlier: f64, later: f64) -> S {
        let twice = Self::gap(raw, earlier, later).scale(2.0);
        let exponent = emission::softplus(raw).mul(&twice);
        emission::log_softplus(raw)
            .add(&ln(&twice))
            .add(&log_exprel(&exponent))
            .sub(&exponent)
    }

    /// Raw-rate slopes of `phi = exp(-r dt)` and of `L = sqrt(1 - phi^2)`, over
    /// any scalar. `dL/draw = dt phi^2 sigmoid(raw) / L` is formed in the log
    /// domain; it tends to zero as the rate does, where `L` alone would
    /// underflow.
    fn rate_slopes<S: JetField>(raw: &S, earlier: f64, later: f64) -> (S, S) {
        // Tied event times give an identity transition with no innovation,
        // whatever the rate.
        if later == earlier {
            return (raw.constant_like(0.0), raw.constant_like(0.0));
        }
        let gap = Self::gap(raw, earlier, later);
        let log_gap = ln(&gap);
        let rate = emission::softplus(raw);
        let log_sigmoid = emission::softplus(&raw.neg()).neg();
        let dphi = exp(&log_gap.add(&log_sigmoid).sub(&rate.mul(&gap))).neg();
        let dscale = exp(
            &log_gap
                .add(&log_sigmoid)
                .sub(&rate.mul(&gap.scale(2.0)))
                .sub(&Self::log_innovation_variance(raw, earlier, later).scale(0.5)),
        );
        (dphi, dscale)
    }

    /// Raw-rate curvatures of `phi` and `L`, with `r' = sigmoid(raw)` and
    /// `r'' = sigmoid(raw) sigmoid(-raw)`:
    ///
    /// ```text
    /// phi''  = dt phi r' (dt r' - sigmoid(-raw))
    /// L''    = -2 dt^2 phi^2 r'^2 / L + dt phi^2 r'' / L - (L')^2 / L.
    /// ```
    ///
    /// Each term is formed in the log domain and stays finite as the rate
    /// vanishes.
    fn rate_curvatures(raw: f64, earlier: f64, later: f64) -> (f64, f64) {
        if later == earlier {
            return (0.0, 0.0);
        }
        let dt = later - earlier;
        let rate = emission::softplus(&raw);
        let log_sigmoid = -emission::softplus(&(-raw));
        let log_complement = -emission::softplus(&raw);
        let log_dt = dt.ln();
        let log_phi = -rate * dt;
        let half_log_variance = 0.5 * Self::log_innovation_variance(&raw, earlier, later);
        let curvature_phi = (log_dt + log_phi + log_sigmoid).exp()
            * (dt * log_sigmoid.exp() - log_complement.exp());
        let log_slope = log_dt + 2.0 * log_phi + log_sigmoid - half_log_variance;
        let curvature_scale = -2.0 * (2.0 * log_dt + 2.0 * log_phi + 2.0 * log_sigmoid
            - half_log_variance)
            .exp()
            + (log_dt + 2.0 * log_phi + log_sigmoid + log_complement - half_log_variance).exp()
            - (2.0 * log_slope - half_log_variance).exp();
        (curvature_phi, curvature_scale)
    }

    fn check_transport<S: JetField>(
        &self,
        h: &JointHistory,
        theta: &[S],
        genes: &[S],
        innovations: &[S],
    ) -> Result<(), EventHistoryError> {
        self.validate_parameters(theta)?;
        self.validate_history(h)?;
        if genes.len() != Self::missing_genes(h).len()
            || innovations.len() != h.times.len() * self.spec.signatures
            || genes
                .iter()
                .chain(innovations)
                .any(|v| !v.value().is_finite())
        {
            return Err(invalid(
                "joint transport coordinates have invalid dimensions or non-finite values",
            ));
        }
        Ok(())
    }

    fn check_nodes<S: JetField>(&self, h: &JointHistory, values: &[S]) -> Result<(), EventHistoryError> {
        if values.len() != h.times.len() * self.spec.signatures
            || values.iter().any(|v| !v.value().is_finite())
        {
            return Err(invalid(
                "joint node forces have invalid dimensions or non-finite values",
            ));
        }
        Ok(())
    }

    fn check_direction<S: JetField>(&self, direction: &[S]) -> Result<(), EventHistoryError> {
        if direction.len() != self.layout.width || direction.iter().any(|v| !v.value().is_finite()) {
            return Err(invalid(
                "joint coefficient direction has invalid dimensions or non-finite values",
            ));
        }
        Ok(())
    }

    /// Node-major states reached by genetic scores and standardized
    /// innovations. `affine = false` gives the linear part `J u`: offsets,
    /// observed scores and jumps drop.
    pub(super) fn transport_states<S: JetField>(
        &self,
        theta: &[S],
        h: &JointHistory,
        genes: &[S],
        innovations: &[S],
        affine: bool,
    ) -> Result<Vec<S>, EventHistoryError> {
        self.check_transport(h, theta, genes, innovations)?;
        let k = self.spec.signatures;
        let missing = Self::missing_genes(h);
        let features = self.entry_features(h);
        let zero = theta[0].constant_like(0.0);
        let mut states: Vec<S> = Vec::with_capacity(h.times.len() * k);
        let regression = |start: usize, columns: &[f64], axis: usize| {
            let (offset, loadings) = self.regression_parts(theta, start, columns, h, &missing, axis);
            let loaded = loadings
                .iter()
                .zip(genes)
                .fold(zero.clone(), |sum, (l, g)| sum.add(&l.mul(g)));
            if affine { offset.add(&loaded) } else { loaded }
        };
        for axis in 0..k {
            states.push(regression(self.layout.entry.start, &features, axis).add(&innovations[axis]));
        }
        for n in 1..h.times.len() {            let columns = h.drive_design.row(n - 1).to_vec();
            for axis in 0..k {
                // The rate factors are formed over S: a running-error scalar
                // charges their rounding chains.
                let (retained, weight, scale) =
                    Self::transition(&theta[self.layout.rates.start + axis], h.times[n - 1], h.times[n])?;
                let mut after = states[(n - 1) * k + axis].clone();
                if affine {
                    after = after.add(&self.jump(theta, &h.events[n - 1], axis));
                }
                let next = after
                    .mul(&retained)
                    .add(&regression(self.layout.drive.start, &columns, axis).mul(&weight))
                    .add(&innovations[n * k + axis].mul(&scale));
                states.push(next);
            }
        }
        Ok(states)
    }

    /// `(d x / d theta) direction` at fixed genetic scores and standardized
    /// innovations: the path the same latent coordinates follow when the
    /// coefficients move.
    pub(super) fn transport_tangent<S: JetField>(
        &self,
        theta: &[S],
        h: &JointHistory,
        genes: &[S],
        innovations: &[S],
        direction: &[S],
    ) -> Result<Vec<S>, EventHistoryError> {
        self.check_direction(direction)?;
        let states = self.transport_states(theta, h, genes, innovations, true)?;
        let k = self.spec.signatures;
        let values = Self::gene_values(h, genes, &theta[0]);
        let features = self.entry_features(h);
        let mut tangent: Vec<S> = Vec::with_capacity(h.times.len() * k);
        for axis in 0..k {
            tangent.push(self.mean(direction, self.layout.entry.start, &features, &values, axis));
        }
        for n in 1..h.times.len() {            let columns = h.drive_design.row(n - 1).to_vec();
            for axis in 0..k {
                let raw = theta[self.layout.rates.start + axis].value();
                let (retained, weight) = Self::retention(&raw, h.times[n - 1], h.times[n])?;
                let (dphi, dscale) = Self::rate_slopes(&raw, h.times[n - 1], h.times[n]);
                let lever = states[(n - 1) * k + axis]
                    .add(&self.jump(theta, &h.events[n - 1], axis))
                    .sub(&self.mean(theta, self.layout.drive.start, &columns, &values, axis));
                let next = direction[self.layout.rates.start + axis]
                    .mul(&lever.scale(dphi).add(&innovations[n * k + axis].scale(dscale)))
                    .add(
                        &tangent[(n - 1) * k + axis]
                            .add(&self.jump(direction, &h.events[n - 1], axis))
                            .scale(retained),
                    )
                    .add(
                        &self
                            .mean(direction, self.layout.drive.start, &columns, &values, axis)
                            .scale(weight),
                    );
                tangent.push(next);
            }
        }
        if tangent.iter().any(|v| !v.value().is_finite()) {
            return Err(numerical("non-finite joint transport tangent"));
        }
        Ok(tangent)
    }

    /// `(d x / d theta)' forces` at fixed genetic scores and standardized
    /// innovations, by one reverse recursion. `affine = false` differentiates
    /// the linear part of the transport, `J(theta) u`.
    pub(super) fn transport_coefficient_adjoint<S: JetField>(
        &self,
        theta: &[S],
        h: &JointHistory,
        genes: &[S],
        innovations: &[S],
        forces: &[S],
        affine: bool,
    ) -> Result<Vec<S>, EventHistoryError> {
        self.check_nodes(h, forces)?;
        let states = self.transport_states(theta, h, genes, innovations, affine)?;
        let k = self.spec.signatures;
        let missing = Self::missing_genes(h);
        let features = self.entry_features(h);
        let zero = theta[0].constant_like(0.0);
        let mut coefficients = vec![zero.clone(); self.layout.width];
        let mut carried = vec![zero.clone(); k];
        for n in (0..h.times.len()).rev() {
            for axis in 0..k {
                carried[axis] = carried[axis].add(&forces[n * k + axis]);
                let adjoint = carried[axis].clone();
                let constant = if affine { adjoint.clone() } else { zero.clone() };
                let gene_forces: Vec<S> = genes.iter().map(|g| g.mul(&adjoint)).collect();
                if n == 0 {
                    self.regression_force(
                        &mut coefficients,
                        self.layout.entry.start,
                        &features,
                        h,
                        &missing,
                        axis,
                        &constant,
                        &gene_forces,
                    );
                    continue;
                }                let columns = h.drive_design.row(n - 1).to_vec();
                // The rate factors are formed over S: a running-error scalar
                // charges their rounding chains.
                let rate = &theta[self.layout.rates.start + axis];
                let (retained, weight) = Self::retention(rate, h.times[n - 1], h.times[n])?;
                let (dphi, dscale) = Self::rate_slopes(rate, h.times[n - 1], h.times[n]);
                let (offset, loadings) =
                    self.regression_parts(theta, self.layout.drive.start, &columns, h, &missing, axis);
                let mut drive = loadings
                    .iter()
                    .zip(genes)
                    .fold(zero.clone(), |sum, (l, g)| sum.add(&l.mul(g)));
                let mut lever = states[(n - 1) * k + axis].clone();
                if affine {
                    drive = drive.add(&offset);
                    lever = lever.add(&self.jump(theta, &h.events[n - 1], axis));
                }
                let index = self.layout.rates.start + axis;
                coefficients[index] = coefficients[index].add(
                    &adjoint.mul(
                        &lever
                            .sub(&drive)
                            .mul(&dphi)
                            .add(&innovations[n * k + axis].mul(&dscale)),
                    ),
                );
                if affine {
                    self.jump_force(
                        &mut coefficients,
                        &h.events[n - 1],
                        axis,
                        &adjoint.mul(&retained),
                    );
                }
                let weighted: Vec<S> = gene_forces.iter().map(|f| f.mul(&weight)).collect();
                self.regression_force(
                    &mut coefficients,
                    self.layout.drive.start,
                    &columns,
                    h,
                    &missing,
                    axis,
                    &constant.mul(&weight),
                    &weighted,
                );
                carried[axis] = adjoint.mul(&retained);
            }
        }
        if coefficients.iter().any(|v| !v.value().is_finite()) {
            return Err(numerical("non-finite joint transport coefficient adjoint"));
        }
        Ok(coefficients)
    }

    /// `(d^2 x / d theta^2 [direction])' forces` at fixed genetic scores,
    /// innovations and forces: the directional derivative, along `direction`,
    /// of [`Self::transport_coefficient_adjoint`]. With `a_n` the carried
    /// adjoint and `da_n` its tangent, `a_(n-1) = f_(n-1) + phi_n a_n` and
    /// `da_(n-1) = phi_n da_n + phi_n' v a_n`. Each contribution of the
    /// first-order recursion is differentiated through its rate, state, drive
    /// and jump factors.
    pub(super) fn transport_second_adjoint<S: JetField>(
        &self,
        theta: &[S],
        h: &JointHistory,
        genes: &[S],
        innovations: &[S],
        forces: &[S],
        direction: &[S],
    ) -> Result<Vec<S>, EventHistoryError> {
        self.check_nodes(h, forces)?;
        let states = self.transport_states(theta, h, genes, innovations, true)?;
        let tangent = self.transport_tangent(theta, h, genes, innovations, direction)?;
        let k = self.spec.signatures;
        let missing = Self::missing_genes(h);
        let values = Self::gene_values(h, genes, &theta[0]);
        let features = self.entry_features(h);
        let zero = theta[0].constant_like(0.0);
        let mut coefficients = vec![zero.clone(); self.layout.width];
        let mut carried = vec![zero.clone(); k];
        let mut carried_tangent = vec![zero.clone(); k];
        for n in (0..h.times.len()).rev() {
            for axis in 0..k {
                carried[axis] = carried[axis].add(&forces[n * k + axis]);
                let adjoint = carried[axis].clone();
                let adjoint_tangent = carried_tangent[axis].clone();
                let gene_forces: Vec<S> = genes.iter().map(|g| g.mul(&adjoint_tangent)).collect();
                if n == 0 {
                    self.regression_force(
                        &mut coefficients,
                        self.layout.entry.start,
                        &features,
                        h,
                        &missing,
                        axis,
                        &adjoint_tangent,
                        &gene_forces,
                    );
                    continue;
                }                let columns = h.drive_design.row(n - 1).to_vec();
                let raw = theta[self.layout.rates.start + axis].value();
                let slope = &direction[self.layout.rates.start + axis];
                let (retained, weight) = Self::retention(&raw, h.times[n - 1], h.times[n])?;
                let (dphi, dscale) = Self::rate_slopes(&raw, h.times[n - 1], h.times[n]);
                let (curvature_phi, curvature_scale) = Self::rate_curvatures(raw, h.times[n - 1], h.times[n]);
                let lever = states[(n - 1) * k + axis]
                    .add(&self.jump(theta, &h.events[n - 1], axis))
                    .sub(&self.mean(theta, self.layout.drive.start, &columns, &values, axis));
                let lever_tangent = tangent[(n - 1) * k + axis]
                    .add(&self.jump(direction, &h.events[n - 1], axis))
                    .sub(&self.mean(direction, self.layout.drive.start, &columns, &values, axis));
                let innovation = &innovations[n * k + axis];
                let index = self.layout.rates.start + axis;
                let rate = adjoint_tangent
                    .mul(&lever.scale(dphi).add(&innovation.scale(dscale)))
                    .add(&adjoint.mul(&slope.mul(&lever.scale(curvature_phi))))
                    .add(&adjoint.mul(&lever_tangent.scale(dphi)))
                    .add(&adjoint.mul(&slope.mul(&innovation.scale(curvature_scale))));
                coefficients[index] = coefficients[index].add(&rate);
                let jumped = adjoint_tangent
                    .scale(retained)
                    .add(&adjoint.mul(slope).scale(dphi));
                self.jump_force(&mut coefficients, &h.events[n - 1], axis, &jumped);
                // d(1 - phi) = -phi' v.
                let weight_tangent = adjoint_tangent
                    .scale(weight)
                    .sub(&adjoint.mul(slope).scale(dphi));
                let weighted: Vec<S> = genes.iter().map(|g| g.mul(&weight_tangent)).collect();
                self.regression_force(
                    &mut coefficients,
                    self.layout.drive.start,
                    &columns,
                    h,
                    &missing,
                    axis,
                    &weight_tangent,
                    &weighted,
                );
                carried[axis] = adjoint.scale(retained);
                carried_tangent[axis] = adjoint_tangent
                    .scale(retained)
                    .add(&slope.mul(&adjoint).scale(dphi));
            }
        }
        if coefficients.iter().any(|v| !v.value().is_finite()) {
            return Err(numerical("non-finite joint transport second adjoint"));
        }
        Ok(coefficients)
    }

    /// Transpose of the linear transport `J u`: node forces to genetic and
    /// innovation forces, by one reverse recursion.
    pub(super) fn transport_latent_adjoint<S: JetField>(
        &self,
        theta: &[S],
        h: &JointHistory,
        forces: &[S],
    ) -> Result<(Vec<S>, Vec<S>), EventHistoryError> {
        self.check_nodes(h, forces)?;
        let k = self.spec.signatures;
        let missing = Self::missing_genes(h);
        let features = self.entry_features(h);
        let zero = theta[0].constant_like(0.0);
        let mut genes = vec![zero.clone(); missing.len()];
        let mut innovations = vec![zero.clone(); h.times.len() * k];
        let mut carried = vec![zero.clone(); k];
        for n in (0..h.times.len()).rev() {
            for axis in 0..k {
                carried[axis] = carried[axis].add(&forces[n * k + axis]);
                let adjoint = carried[axis].clone();
                if n == 0 {
                    let loadings = self
                        .regression_parts(theta, self.layout.entry.start, &features, h, &missing, axis)
                        .1;
                    for (a, loading) in loadings.iter().enumerate() {
                        genes[a] = genes[a].add(&loading.mul(&adjoint));
                    }
                    innovations[axis] = adjoint;
                    continue;
                }                let columns = h.drive_design.row(n - 1).to_vec();
                let (retained, weight, scale) =
                    Self::transition(&theta[self.layout.rates.start + axis].value(), h.times[n - 1], h.times[n])?;
                let loadings = self
                    .regression_parts(theta, self.layout.drive.start, &columns, h, &missing, axis)
                    .1;
                for (a, loading) in loadings.iter().enumerate() {
                    genes[a] = genes[a].add(&loading.mul(&adjoint).scale(weight));
                }
                innovations[n * k + axis] = adjoint.scale(scale);
                carried[axis] = adjoint.scale(retained);
            }
        }
        Ok((genes, innovations))
    }

    /// `(dJ/dtheta [direction])' forces`: the latent-coordinate transpose of the
    /// linear part of the transport tangent. That part is
    /// `phi' v (x_(n-1) - D_n g) + L' v e_n + phi dx_(n-1) + (1 - phi) D_n[v] g`,
    /// so its transpose carries `b_n = f_n + phi_(n+1) b_(n+1)` backwards. It
    /// sends `b_n phi' v` through the transport's own transpose, and adds the
    /// direct genetic and innovation forces.
    pub(super) fn tangent_latent_adjoint<S: JetField>(
        &self,
        theta: &[S],
        h: &JointHistory,
        direction: &[S],
        forces: &[S],
    ) -> Result<(Vec<S>, Vec<S>), EventHistoryError> {
        self.validate_parameters(theta)?;
        self.validate_history(h)?;
        self.check_direction(direction)?;
        self.check_nodes(h, forces)?;
        let k = self.spec.signatures;
        let missing = Self::missing_genes(h);
        let features = self.entry_features(h);
        let zero = theta[0].constant_like(0.0);
        let mut gene_forces = vec![zero.clone(); missing.len()];
        let mut innovation_forces = vec![zero.clone(); h.times.len() * k];
        let mut node_forces = vec![zero.clone(); h.times.len() * k];
        let mut carried = vec![zero.clone(); k];
        for n in (0..h.times.len()).rev() {
            for axis in 0..k {
                carried[axis] = carried[axis].add(&forces[n * k + axis]);
                let adjoint = carried[axis].clone();
                if n == 0 {
                    let loadings = self
                        .regression_parts(direction, self.layout.entry.start, &features, h, &missing, axis)
                        .1;
                    for (a, loading) in loadings.iter().enumerate() {
                        gene_forces[a] = gene_forces[a].add(&loading.mul(&adjoint));
                    }
                    continue;
                }                let columns = h.drive_design.row(n - 1).to_vec();
                let raw = theta[self.layout.rates.start + axis].value();
                let slope = &direction[self.layout.rates.start + axis];
                let (retained, weight) = Self::retention(&raw, h.times[n - 1], h.times[n])?;
                let (dphi, dscale) = Self::rate_slopes(&raw, h.times[n - 1], h.times[n]);
                let loadings = self
                    .regression_parts(theta, self.layout.drive.start, &columns, h, &missing, axis)
                    .1;
                let loadings_tangent = self
                    .regression_parts(direction, self.layout.drive.start, &columns, h, &missing, axis)
                    .1;
                let lever = adjoint.mul(slope).scale(dphi);
                let index = (n - 1) * k + axis;
                node_forces[index] = node_forces[index].add(&lever);
                for a in 0..missing.len() {
                    gene_forces[a] = gene_forces[a]
                        .sub(&lever.mul(&loadings[a]))
                        .add(&adjoint.mul(&loadings_tangent[a]).scale(weight));
                }
                innovation_forces[n * k + axis] =
                    innovation_forces[n * k + axis].add(&adjoint.mul(slope).scale(dscale));
                carried[axis] = adjoint.scale(retained);
            }
        }
        let (transported_genes, transported_innovations) =
            self.transport_latent_adjoint(theta, h, &node_forces)?;
        let genes: Vec<S> = gene_forces
            .iter()
            .zip(&transported_genes)
            .map(|(a, b)| a.add(b))
            .collect();
        let innovations: Vec<S> = innovation_forces
            .iter()
            .zip(&transported_innovations)
            .map(|(a, b)| a.add(b))
            .collect();
        if genes.iter().chain(&innovations).any(|v| !v.value().is_finite()) {
            return Err(numerical("non-finite joint tangent latent adjoint"));
        }
        Ok((genes, innovations))
    }
}

#[cfg(test)]
mod test_support {
    use super::*;

    impl JointLikelihood {
        /// Node-major states reached by genetic scores and standardized
        /// innovations, over any jet in the coefficients and the latent
        /// coordinates, with every rate factor formed from the jet: the
        /// independent route against which the analytic transport sensitivities
        /// are checked. It forms `ln V`, so it is an oracle only away from static
        /// transitions.
        pub(crate) fn transport_path<S: JetField>(
            &self,
            theta: &[S],
            h: &JointHistory,
            genes: &[S],
            innovations: &[S],
        ) -> Vec<S> {
            let k = self.spec.signatures;
            let values = Self::gene_values(h, genes, &theta[0]);
            let features = self.entry_features(h);
            let mut states: Vec<S> = Vec::with_capacity(h.times.len() * k);
            for axis in 0..k {
                let mean = self.mean(theta, self.layout.entry.start, &features, &values, axis);
                states.push(mean.add(&innovations[axis]));
            }
            for n in 1..h.times.len() {                let columns = h.drive_design.row(n - 1).to_vec();
                for axis in 0..k {
                    let rate = &theta[self.layout.rates.start + axis];
                    let decay = emission::softplus(rate).mul(&Self::gap(rate, h.times[n - 1], h.times[n])).neg();
                    let retained = exp(&decay);
                    let weight = emission::expm1(&decay).neg();
                    let scale = exp(&ln(&emission::expm1(&decay.scale(2.0)).neg()).scale(0.5));
                    let drive = self.mean(theta, self.layout.drive.start, &columns, &values, axis);
                    let after =
                        states[(n - 1) * k + axis].add(&self.jump(theta, &h.events[n - 1], axis));
                    states.push(
                        retained
                            .mul(&after)
                            .add(&weight.mul(&drive))
                            .add(&scale.mul(&innovations[n * k + axis])),
                    );
                }
            }
            states
        }
    }
}

#[cfg(test)]
mod tests {
    use super::super::law::{CompensatorPoint, JointSpecification};
    use super::*;
    use crate::MarkKind;
    use crate::scalar::Rows;
    use crate::test_support::{Bound, agrees};

    fn exact(values: &[f64]) -> Vec<Bound> {
        values.iter().map(|&v| Bound::exact(v)).collect()
    }

    /// Two signatures, one missing and one observed score, a recurrent and a
    /// once-only mark, tied events at node 4, and one cell point per gap.
    fn fixture() -> (JointLikelihood, JointHistory, Vec<f64>, Vec<f64>, Vec<f64>) {
        let model = JointLikelihood::new(JointSpecification {
            signatures: 2,
            marks: vec![MarkKind::Recurrent, MarkKind::Once],
            baseline_columns: 1,
            population_columns: 1,
            drive_columns: 2,
            entry_columns: 1,
            baseline_penalties: vec![],
            drive_penalties: vec![],
            population_penalties: vec![],
            measurements: vec![],
            genetic_mean: vec![0.1, -0.2],
            genetic_precision: ndarray::arr2(&[[1.5, -0.4], [-0.4, 1.2]]),
        })
        .unwrap();
        let times = vec![0.0, 0.2, 0.3, 0.5, 0.8, 1.0];
        let mut points = vec![CompensatorPoint {
            node: 0,
            time: times[0],
            weight: 0.0,
        }];
        for n in 1..times.len() {
            points.push(CompensatorPoint {
                node: n,
                time: times[n],
                weight: 0.0,
            });
            points.push(CompensatorPoint {
                node: n,
                time: times[n],
                weight: times[n] - times[n - 1],
            });
        }
        let rows = points.len();
        let h = JointHistory {
            drive_design: Array2::from_shape_fn((5, 2), |(n, b)| if b == 0 { 1.0 } else { times[n] }),
            times,
            points,
            events: vec![vec![], vec![], vec![0], vec![], vec![1, 0], vec![]],
            initially_at_risk: vec![true, true],
            baseline_design: Array2::ones((rows, 1)),
            population_design: Array2::ones((rows, 1)),
            entry_design: vec![0.6],
            genetics: vec![None, Some(0.6)],
            measurements: vec![],
        };
        let theta: Vec<f64> = (0..model.layout.width)
            .map(|j| 0.4 * (1.3 * j as f64).sin())
            .collect();
        let genes = vec![0.35];
        let innovations: Vec<f64> = (0..12).map(|i| 0.7 * ((2 * (i / 2) + 3 * (i % 2)) as f64).cos()).collect();
        (model, h, theta, genes, innovations)
    }

    fn inner(a: &[Bound], b: &[Bound]) -> Bound {
        a.iter()
            .zip(b)
            .fold(Bound::exact(0.0), |sum, (x, y)| sum.add(&x.mul(y)))
    }

    #[test]
    fn transport_tangent_and_its_adjoint_differentiate_the_innovation_path() {
        let (model, h, mut theta, genes, innovations) = fixture();
        let direction: Vec<f64> = (0..theta.len()).map(|j| (0.7 * j as f64).cos()).collect();
        let forces: Vec<f64> = (0..12).map(|i| 0.3 * ((i / 2 + 5 * (i % 2)) as f64).sin()).collect();
        for raw in [0.4, -40.0] {
            for axis in 0..2 {
                theta[model.layout.rates.start + axis] = raw + 0.3 * axis as f64;
            }
            let bounded = exact(&theta);
            let states = model
                .transport_states(&bounded, &h, &exact(&genes), &exact(&innovations), true)
                .unwrap();
            let tangent = model
                .transport_tangent(
                    &bounded,
                    &h,
                    &exact(&genes),
                    &exact(&innovations),
                    &exact(&direction),
                )
                .unwrap();
            let seeded: Vec<Rows<Bound, 1>> = theta
                .iter()
                .zip(&direction)
                .map(|(&v, &d)| Rows::seed(Bound::exact(v), [d]))
                .collect();
            let lift = |values: &[f64]| -> Vec<Rows<Bound, 1>> {
                values.iter().map(|&v| Rows::seed(Bound::exact(v), [0.0])).collect()
            };
            let jets = model.transport_path(&seeded, &h, &lift(&genes), &lift(&innovations));
            for (i, jet) in jets.iter().enumerate() {
                agrees(&states[i], &jet.base, &format!("raw {raw} path {i}"));
                agrees(&tangent[i], &jet.rows[0], &format!("raw {raw} tangent {i}"));
            }
            let adjoint = model
                .transport_coefficient_adjoint(
                    &bounded,
                    &h,
                    &exact(&genes),
                    &exact(&innovations),
                    &exact(&forces),
                    true,
                )
                .unwrap();
            agrees(
                &inner(&adjoint, &exact(&direction)),
                &inner(&exact(&forces), &tangent),
                &format!("raw {raw} duality"),
            );
        }
        // A rate that underflows to zero keeps a vanishing phi slope and a
        // finite, positive innovation-scale slope, and tied times move nothing.
        let (dphi, dscale) = JointLikelihood::rate_slopes(&-800.0_f64, 0.0, 0.25);
        assert_eq!(dphi, 0.0);
        assert!(dscale.is_finite() && dscale > 0.0);
        let (curvature_phi, curvature_scale) = JointLikelihood::rate_curvatures(-800.0, 0.0, 0.25);
        assert!(curvature_phi.is_finite() && curvature_scale.is_finite());
        assert_eq!(JointLikelihood::rate_slopes(&0.4_f64, 0.3, 0.3), (0.0, 0.0));
        assert_eq!(JointLikelihood::rate_curvatures(0.4, 0.3, 0.3), (0.0, 0.0));
    }

    #[test]
    fn second_and_latent_transport_adjoints_are_transposes_of_their_derivatives() {
        let (model, h, theta, genes, innovations) = fixture();
        let width = theta.len();
        let v: Vec<f64> = (0..width).map(|j| (0.7 * j as f64).cos()).collect();
        let w: Vec<f64> = (0..width).map(|j| (1.1 * j as f64 + 0.3).sin()).collect();
        let forces: Vec<f64> = (0..12).map(|i| 0.3 * ((i / 2 + 5 * (i % 2)) as f64).sin()).collect();
        let bounded = exact(&theta);

        // Second derivative of the path along (v, w) by nested jets.
        type Two = Rows<Rows<Bound, 1>, 1>;
        let two = |value: f64, a: f64, b: f64| -> Two { Rows::seed(Rows::seed(Bound::exact(value), [a]), [b]) };
        let seeded: Vec<Two> = theta
            .iter()
            .zip(v.iter().zip(&w))
            .map(|(&value, (&a, &b))| two(value, a, b))
            .collect();
        let gene_jets: Vec<Two> = genes.iter().map(|&g| two(g, 0.0, 0.0)).collect();
        let innovation_jets: Vec<Two> = innovations.iter().map(|&e| two(e, 0.0, 0.0)).collect();
        let jets = model.transport_path(&seeded, &h, &gene_jets, &innovation_jets);
        let direct = forces
            .iter()
            .zip(&jets)
            .fold(Bound::exact(0.0), |sum, (&f, jet)| {
                sum.add(&jet.rows[0].rows[0].scale(f))
            });
        let second = model
            .transport_second_adjoint(
                &bounded,
                &h,
                &exact(&genes),
                &exact(&innovations),
                &exact(&forces),
                &exact(&v),
            )
            .unwrap();
        agrees(&inner(&second, &exact(&w)), &direct, "second adjoint");

        // The tangent is affine in the latent coordinates; its linear part is the
        // difference of two tangents, and both transposes must reproduce it.
        let du_genes = vec![-0.8];
        let du_innovations: Vec<f64> = (0..12).map(|i| 0.5 * ((3 * (i / 2) + i % 2) as f64).sin()).collect();
        let moved = model
            .transport_tangent(&bounded, &h, &exact(&du_genes), &exact(&du_innovations), &exact(&v))
            .unwrap();
        let anchored = model
            .transport_tangent(&bounded, &h, &exact(&[0.0]), &exact(&[0.0; 12]), &exact(&v))
            .unwrap();
        let difference: Vec<Bound> = moved.iter().zip(&anchored).map(|(a, b)| a.sub(b)).collect();
        let linear = inner(&exact(&forces), &difference);
        let (gene_forces, innovation_forces) = model
            .tangent_latent_adjoint(&bounded, &h, &exact(&v), &exact(&forces))
            .unwrap();
        let latent = inner(&gene_forces, &exact(&du_genes)).add(&inner(
            &innovation_forces,
            &exact(&du_innovations),
        ));
        agrees(&latent, &linear, "latent adjoint");
        let linear_adjoint = model
            .transport_coefficient_adjoint(
                &bounded,
                &h,
                &exact(&du_genes),
                &exact(&du_innovations),
                &exact(&forces),
                false,
            )
            .unwrap();
        agrees(&inner(&linear_adjoint, &exact(&v)), &linear, "linear coefficient adjoint");

        // The latent transpose of the transport matches the linear transport.
        let (transported_genes, transported_innovations) =
            model.transport_latent_adjoint(&bounded, &h, &exact(&forces)).unwrap();
        let states = model
            .transport_states(&bounded, &h, &exact(&du_genes), &exact(&du_innovations), false)
            .unwrap();
        agrees(
            &inner(&transported_genes, &exact(&du_genes))
                .add(&inner(&transported_innovations, &exact(&du_innovations))),
            &inner(&exact(&forces), &states),
            "transport transpose",
        );
    }
}
