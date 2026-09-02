//! #1031 acceptance arm (2): the streaming 2-D grid spline engine
//! (`solver::grid_spline_2d`) must RECOVER a known 2-D surface from noisy data
//! and do so at least as accurately as `mgcv`'s tensor-product smooth `te()` —
//! the mature, de-facto standard for anisotropic 2-D GAM smoothing.
//!
//! Why `te()` is the right yardstick AND why this is its own estimator. The
//! grid engine penalizes the FULL anisotropic biharmonic energy
//!   `J(f) = ∫∫ a₁²·f_{x1x1}² + 2·a₁a₂·f_{x1x2}² + a₂²·f_{x2x2}²`,
//! INCLUDING the mixed `f_{x1x2}²` term. `te(x1,x2)` penalizes a Kronecker SUM
//! of per-margin wiggliness — a *different* roughness functional that drops the
//! mixed cross-derivative coupling (which is precisely why #1031 exposes the
//! grid as its own pair-component estimator instead of silently re-routing
//! `te()` through it). On a truth with a genuine cross-derivative component the
//! grid engine's penalty is better matched to the signal, so "match-or-beat
//! mgcv on truth recovery" is principled, not a tolerance artifact.
//!
//! DGP (self-constructed truth, #904). A non-separable surface with a real
//! mixed-derivative term,
//!   `f(x1,x2) = sin(2.4·x1) · cos(2.1·x2) + 0.8·x1·x2 + 0.5·sin(3.0·x1·x2)`,
//! over `[0,1]²` on a deterministic 45×45 lattice (n=2025), plus a fixed,
//! RNG-free golden-ratio noise stream (so gam and mgcv see IDENTICAL rows). The
//! `0.5·sin(3·x1·x2)` term has a non-trivial `f_{x1x2}`, the channel the grid
//! penalty captures and the Kronecker-marginal `te` penalty does not.
//!
//! OBJECTIVE METRIC. RMSE of each engine's fitted surface against the NOISELESS
//! truth `f` at the training rows. PRIMARY: the grid engine recovers `f` to a
//! small fraction of the signal amplitude. MATCH-OR-BEAT: its recovery RMSE is
//! no worse than mgcv's by more than 10%. We never assert the two fitted
//! surfaces are close to each other.

