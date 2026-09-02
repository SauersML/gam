//! #1031 acceptance arm (2), through the LIVE backend: `fit_pair_surface`
//! — THE first-class pair-component estimator the #975 ANOVA carve consumes —
//! must recover a known 2-D surface and match-or-beat `mgcv`'s tensor-product
//! `te()`/`ti()` on truth recovery.
//!
//! Why this is distinct from `quality_vs_mgcv_grid_spline_2d_truth_recovery`.
//! That sibling test drives the bare `fit_grid_spline_2d` engine on a perfect
//! lattice. THIS test drives the production entry `fit_pair_surface` on
//! SCATTERED (non-lattice) raw coordinates — the shape the carve actually sees
//! — exercising the full live backend: the axis-rescaling-invariant metric
//! `a_i = L_i²`, the cube-root `K` sizing rule, the exact-REML λ-selection with
//! the dense-ridge degeneracy fallback, and the carve-facing posterior via the
//! consumer's own `predict`. We additionally assert the live path resolved to
//! the exact grid backend (`PairSurfaceBackend::GridExact`), certifying the
//! grid engine is the one actually reached on a well-posed scattered pair — not
//! the dense fallback.
//!
//! Why `te()` is the right yardstick AND why match-or-beat is principled. The
//! grid engine penalizes the FULL anisotropic biharmonic energy
//!   `J(f) = ∫∫ a₁²·f_{x1x1}² + 2·a₁a₂·f_{x1x2}² + a₂²·f_{x2x2}²`,
//! INCLUDING the mixed `f_{x1x2}²` term that `te()`'s Kronecker-marginal
//! penalty drops. On a truth carrying a genuine cross-derivative component the
//! biharmonic penalty is better matched to the signal, so "match-or-beat mgcv
//! on truth recovery" is principled, not a tolerance artifact. We never assert
//! the two fitted surfaces are close to each other (different posteriors — the
//! whole reason the grid is its own estimator, not a `te()` back-end).
//!
//! DGP (self-constructed truth, #904). A non-separable surface with a real
//! mixed-derivative term,
//!   `f(x1,x2) = sin(2.4·x1)·cos(2.1·x2) + 0.8·x1·x2 + 0.5·sin(3.0·x1·x2)`,
//! over `[0,1]²` at SCATTERED low-discrepancy (golden-ratio / √2) abscissae
//! (n=2000), plus a fixed, RNG-free zero-mean noise stream so gam and mgcv see
//! IDENTICAL rows.
//!
//! OBJECTIVE METRIC. RMSE of each engine's fitted surface against the NOISELESS
//! truth `f` at the training rows. PRIMARY: the live backend recovers `f` to a
//! small fraction of the signal range. MATCH-OR-BEAT: its recovery RMSE is no
//! worse than mgcv's by more than 10%.

