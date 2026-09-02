pub mod fast_channel;
pub mod jet_algebra;
pub mod jet_partitions;
pub mod jet_scalar;
pub mod jet_tower;
pub mod nested_dual;
pub mod order2_graph;
// One paired, interleaved, order-randomised timing harness for every "does A
// beat B" gate in the workspace. Fifteen separate harnesses in three
// methodologies were doing this, and only two interleaved the arms — which is
// the property that decides whether a gate can tell a regression from a busy
// machine (#932, #2470).
pub mod paired_timing;
pub mod probability;
pub mod quadrature;
pub mod score_opt;
pub mod special;

#[cfg(test)]
mod jet_gamma_oracle_tests;
#[cfg(test)]
mod jet_poisson_oracle_tests;
pub mod quantile;
