pub mod fast_channel;
pub mod jet_algebra;
pub mod jet_partitions;
pub mod jet_scalar;
pub mod jet_tower;
pub mod nested_dual;
// One paired, interleaved, order-randomised timing harness for every "does A
// beat B" gate in the workspace. Fifteen separate harnesses in three
// methodologies were doing this, and only two interleaved the arms — which is
// the property that decides whether a gate can tell a regression from a busy
// machine (#932, #2470).
/// The speed-gate harness (#932). It ships in this library because the
/// integration tests of other crates measure with it, and a `#[cfg(test)]`
/// item cannot cross a crate boundary; no product artifact links it, and the
/// `paired_timing_report` example is the reachability root that says so to a
/// dead-code sweep keyed on artifact symbol tables (one such sweep deleted
/// the harness, every gate that called it and every hand opponent those
/// gates raced).
pub mod paired_timing;
pub mod probability;
pub mod quadrature;
pub mod score_opt;
pub mod serial_dependence;
pub mod special;

#[cfg(test)]
mod jet_gamma_oracle_tests;
#[cfg(test)]
mod jet_poisson_oracle_tests;
pub mod quantile;
