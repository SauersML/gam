// Split from the original oversized module; keep included in order.
include!("smooth/part_000.rs");


mod bspline_boundary;

mod coefficient_transforms;

mod error;

mod input_standardization;

mod shape_constraints;

include!("smooth/part_001.rs");
include!("smooth/part_002.rs");
include!("smooth/part_003.rs");
include!("smooth/part_004.rs");
