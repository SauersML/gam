//! Regression for the lazy anisotropic gap in gam#1818.
//!
//! A cold dense Duchon build and the lazy/operator build represent the same
//! function and therefore must use the same data-metric radial chart.  The old
//! lazy implementation computed `V` only in its isotropic branch; anisotropic
//! builds stayed in the raw constrained-kernel frame, so their penalty spectrum
//! and REML geometry depended on which memory route happened to be selected.

