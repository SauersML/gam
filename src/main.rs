#![deny(unused_variables)]

// Split from the original oversized module; keep included in order.
include!("main_parts/part_000.rs");
include!("main_parts/part_001.rs");

#[cfg(test)]
#[path = "../tests/src_modules/cli_tests.rs"]
mod cli_tests;
