//! Observation channels of the latent state: the support each declared family
//! accepts.
use super::law::{MeasurementFamily, invalid};
use crate::EventHistoryError;

pub(super) fn validate_value(family: &MeasurementFamily, y: f64) -> Result<(), EventHistoryError> {
    let valid = y.is_finite()
        && match family {
            MeasurementFamily::StudentT => true,
            MeasurementFamily::Probit { categories } => {
                y >= 0.0 && y.fract() == 0.0 && y < *categories as f64
            }
            MeasurementFamily::NegativeBinomial => y >= 0.0 && y.fract() == 0.0,
        };
    if valid {
        Ok(())
    } else {
        Err(invalid("measurement is outside its declared support"))
    }
}
