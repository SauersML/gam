//! Exact finite-response calculations for a known MLP block under a declared Gaussian intervention law (#2946).
//!
//! A known block `F(z) = Σ_j u_j σ(b_j + w_jᵀ z)` is read under the declared experiment `h = h₀ + L Z`,
//! `Z ~ N(0, I_d)`, taken after the pre-MLP norm, with `W h₀ + b` absorbed into the biases and `W L` into the readers.
//! It is a declared intervention: nothing here claims natural activations are Gaussian.

pub mod subspace;
pub mod reader_gram;
pub mod context;
pub mod raw_block;
pub mod executed_transport;
pub mod reuse;
pub mod state_blocks;
pub mod compose;
pub mod compile;
pub mod hermite;
pub mod interaction;
pub mod tiles;
