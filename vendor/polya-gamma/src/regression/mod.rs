#![cfg(feature = "regression")]
//! Bayesian regression models using Gibbs sampling with Polya-Gamma augmentation.
//!
//! This module provides implementations of various Bayesian regression models that use
//! Polya-Gamma augmentation for efficient Gibbs sampling. The models in this module
//! are designed to handle binary and count data through data augmentation techniques
//! that make the conditional distributions conjugate.
//!
//! # Available Models
//! - [`GibbsLogit`]: Bayesian logistic regression for binary outcome data
//! - [`GibbsNegativeBinomial`]: Bayesian negative binomial regression for overdispersed count data
//!
//! # Examples
//! See the examples directory for complete usage examples of each model.
pub use logit::GibbsLogit;
pub use negative_binomial::GibbsNegativeBinomial;

mod logit;
mod negative_binomial;
