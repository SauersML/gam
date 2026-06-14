#![deny(unused_variables)]

// Split from the original oversized module; keep included in order.
include!("main_cli_definitions.rs");
include!("main_fit_and_prediction_io.rs");

#[cfg(test)]
#[path = "../tests/src_modules/cli_tests.rs"]
mod cli_tests;
