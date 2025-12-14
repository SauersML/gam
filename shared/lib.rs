#![feature(portable_simd)]
#![deny(unused_variables)]
#![deny(dead_code)]
#![deny(unused_imports)]
#![deny(clippy::no_effect_underscore_binding)]

pub mod files;

pub mod shared {
    pub use super::files;
}

pub mod adapt_plink2;

#[path = "../score/mod.rs"]
pub mod score;

#[path = "../terms/mod.rs"]
pub mod terms;

pub mod batch {
    pub use crate::score::batch::*;
}

pub use score::{complex, decide, download, io, kernel, pipeline, prepare, reformat, types};

#[path = "../map/mod.rs"]
pub mod map;

#[path = "../calibrate/mod.rs"]
pub mod calibrate;
