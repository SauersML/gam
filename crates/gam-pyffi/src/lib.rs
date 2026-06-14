// Split from the original oversized module; keep included in order.
include!("lib_parts/part_000.rs");


mod benchmark_scores;

mod competing_risks_decode;

mod inference_instruments;

mod manifold_pyclasses;

mod python_literal;

mod sklearn_metadata;

mod summary_render;

mod survival_surface_io;

include!("lib_parts/part_001.rs");
include!("lib_parts/part_002.rs");
