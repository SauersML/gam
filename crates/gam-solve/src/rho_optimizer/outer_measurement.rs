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
#[derive(Clone, Debug)]
pub struct OuterFirstOrderMeasurement {
    rho: Array1<f64>,
    value: f64,
    gradient: Array1<f64>,
}

impl OuterFirstOrderMeasurement {
    pub fn new(rho: Array1<f64>, value: f64, gradient: Array1<f64>) -> Self {
        Self {
            rho,
            value,
            gradient,
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
}
