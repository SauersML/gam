//! Unit gates for the #2629 floor classifier.
//!
//! Every case here is a SYNTHETIC ladder built from a known law, so the
//! classifier is tested against ground truth rather than against a fixture's
//! behaviour. The two families it must separate are:
//!
//! * `g(ρ) = w·a·tanh(a·ρ̃) + c·e^{−ρ}` — the criterion carries the barrier;
//! * `g(ρ) = c·e^{−ρ}` — it does not.
//!
//! The measured constants from the shipped fixtures are used as the `c` values
//! (`+87.5` from #2450's Matérn/Gaussian ladder, `−22.8` from #2629's
//! SAS/binomial one) so the synthetic ladders sit exactly where the real ones do.

