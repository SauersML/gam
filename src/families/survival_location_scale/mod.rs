// Split from the original oversized module; keep included in order.
include!("imports.rs");

mod numeric_guards;

include!("types_and_specs.rs");
include!("family_impl.rs");
include!("tests.rs");
