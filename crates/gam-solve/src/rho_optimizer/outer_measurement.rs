//! An outer result's first-order measurement: the value and gradient an
//! evaluation returned, and the ρ it was evaluated at (#2953).

use super::*;

/// One first-order measurement of the outer criterion: the value and gradient
/// an evaluation returned, and the ρ it was evaluated at.
///
/// The three form one record, so no writer can move the point without the
/// gradient. #2953: the dominance continuation moved an incumbent's `rho` and
/// `final_value` to its retry's point and kept the gradient of the point the
/// retry started from, and the gradient-reproducibility floor read that pair
/// as two measurements at one ρ.
///
/// A record also keeps every EARLIER measurement it replaced at the same ρ
/// (bit for bit), so the instrument's demonstrated same-point spread survives
/// the certificate's own re-measurement. Without them the mint's only prior
/// was screening's evaluation, taken from the same inner warm start and
/// therefore bit-identical, and the mint refused on a spread it had discarded:
/// the prostate binomial-logit fit screened at |Pg| = 2.010e-9 against a
/// measured same-ρ spread of 1.949e-9 and the mint of that exact ρ then judged
/// the same |Pg| against the 2.500e-10 rounding band.
#[derive(Clone, Debug)]
pub struct OuterFirstOrderMeasurement {
    rho: Array1<f64>,
    value: f64,
    gradient: Array1<f64>,
    earlier_at_rho: Vec<(f64, Array1<f64>)>,
}

impl OuterFirstOrderMeasurement {
    pub fn new(rho: Array1<f64>, value: f64, gradient: Array1<f64>) -> Self {
        Self {
            rho,
            value,
            gradient,
            earlier_at_rho: Vec::new(),
        }
    }

    /// This measurement's value and gradient followed by every earlier
    /// measurement it replaced at the same ρ, newest first.
    pub fn same_rho_measurements(&self) -> impl Iterator<Item = (f64, &Array1<f64>)> {
        std::iter::once((self.value, &self.gradient)).chain(
            self.earlier_at_rho
                .iter()
                .rev()
                .map(|(value, gradient)| (*value, gradient)),
        )
    }

    /// Re-express every coordinate vector the record holds (its ρ, its
    /// gradient and each earlier same-ρ gradient) through `to_coordinates`.
    pub fn map_coordinates(self, to_coordinates: impl Fn(Array1<f64>) -> Array1<f64>) -> Self {
        Self {
            rho: to_coordinates(self.rho),
            value: self.value,
            gradient: to_coordinates(self.gradient),
            earlier_at_rho: self
                .earlier_at_rho
                .into_iter()
                .map(|(value, gradient)| (value, to_coordinates(gradient)))
                .collect(),
        }
    }

    pub fn rho(&self) -> &Array1<f64> {
        &self.rho
    }

    pub fn value(&self) -> f64 {
        self.value
    }

    pub fn gradient(&self) -> &Array1<f64> {
        &self.gradient
    }

    pub fn into_gradient(self) -> Array1<f64> {
        self.gradient
    }

    pub fn into_parts(self) -> (Array1<f64>, f64, Array1<f64>) {
        (self.rho, self.value, self.gradient)
    }

    /// Whether this measurement was taken at exactly `rho`, bit for bit in
    /// every coordinate.
    pub fn is_at(&self, rho: &Array1<f64>) -> bool {
        self.rho.len() == rho.len()
            && self
                .rho
                .iter()
                .zip(rho.iter())
                .all(|(measured, point)| measured.to_bits() == point.to_bits())
    }
}

impl OuterResult {
    /// The gradient of [`Self::final_measurement`], when the solver measured one.
    pub fn final_gradient(&self) -> Option<&Array1<f64>> {
        self.final_measurement
            .as_ref()
            .map(OuterFirstOrderMeasurement::gradient)
    }

    /// Install a value and gradient evaluated at this result's own ρ.
    pub(crate) fn record_measurement_at_rho(&mut self, value: f64, gradient: Array1<f64>) {
        self.final_measurement = Some(OuterFirstOrderMeasurement::new(
            self.rho.clone(),
            value,
            gradient,
        ));
    }

    /// Install a value and gradient evaluated at this result's own ρ, keeping
    /// `prior` and its own earlier measurements when `prior` was taken at this
    /// same ρ (see [`OuterFirstOrderMeasurement`]).
    pub(crate) fn record_measurement_at_rho_after(
        &mut self,
        prior: Option<&OuterFirstOrderMeasurement>,
        value: f64,
        gradient: Array1<f64>,
    ) {
        let mut measurement = OuterFirstOrderMeasurement::new(self.rho.clone(), value, gradient);
        if let Some(prior) = prior
            && prior.is_at(&self.rho)
        {
            measurement.earlier_at_rho = prior.earlier_at_rho.clone();
            measurement
                .earlier_at_rho
                .push((prior.value, prior.gradient.clone()));
        }
        self.final_measurement = Some(measurement);
    }
}
