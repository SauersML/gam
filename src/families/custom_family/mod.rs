// Split from the original oversized module; keep included in order.
include!("imports.rs");

mod coefficient_groups;

mod persistent_cache;

include!("family_trait_and_blocks.rs");
include!("blockwise_solve.rs");
include!("joint_newton_outer.rs");
include!("psi_hyper_and_jeffreys.rs");
include!("tests.rs");
