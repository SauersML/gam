#![deny(dead_code)]
#![deny(unused_imports)]
#![allow(non_snake_case)]

pub mod basis;
pub mod construction;
pub mod data;

#[cfg(feature = "survival-data")]
pub mod survival_data;

pub mod calibrator;
pub mod estimate;
pub mod faer_ndarray;
pub mod hull;
pub mod model;
pub mod pirls;
pub mod survival;
// No global functions here; settings are scoped to owning modules.
