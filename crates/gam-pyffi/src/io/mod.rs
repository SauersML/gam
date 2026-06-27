//! Payload rendering and I/O helpers: fitted-model summary HTML/repr rendering,
//! competing-risks prediction JSON decoding, and survival-surface interpolation
//! plus CSV streaming.
//!
//! Re-exported at the crate root under their flat names so the entrypoint
//! fragments keep resolving `summary_render::…`, `competing_risks_decode::…`,
//! and `survival_surface_io::…`.

pub(crate) mod summary_render;

pub(crate) mod competing_risks_decode;

pub(crate) mod survival_surface_io;
