//! Deflation/PD-region log-det-trace regression tests (#1026/#1590),
//! split verbatim out of `tests.rs` to keep that tracked file under the #780
//! 10k-line gate. Declared as a sibling `#[cfg(test)] mod` in `mod.rs`; shared
//! `gamma_fd_tiny_fixture` / `fixed_state_logdet_sample` are sourced from the sibling
//! `tests` module.

