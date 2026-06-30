# Issue #1575 — binomial (logit) REML fit ~100-160x slower than mgcv

## Status of prior work (from Discussions thread, all landed on main)
- Inner-PIRLS tolerance relaxation: PROVEN INERT, reverted.
- PSIS rho-uncertainty diagnostic: made opt-in (−33 solves/fit).
- Redundant parsimony 2nd-seed: waived for sharp well-penalized optima.
- Multi-slot outer-eval LRU cache: landed.
- Firth design-factor hoist out of inner PIRLS loop: landed (bit-identical).
- Firth Hessian direction reuse + eye-cache: landed (~37% of Firth-Hessian cost, bit-identical).
- Seed-grid full-n solves: tracked under #1033 (rearchitecture).

## THE dominant bottleneck (measured)
Firth/Jeffreys default-ON for binomial n<=20000 (FIRTH_MAX_OBSERVATIONS).
Firth-ON path ~250-480x slower than Firth-OFF at IDENTICAL solve counts.
~261s of ~267s spent in outer REML LAML-Hessian Firth directional derivatives:
  FirthDenseOperator::hphisecond_direction_apply / D^2H_phi contractions
  in crates/gam-solve/src/reml/gradient_hessian.rs.

## My plan
1. Reproduce / measure the Firth outer-Hessian cost in isolation (Rust bench).
2. Attack the per-(i,j) mixed D^2J2 contraction inside hphisecond_direction_apply
   — restructure the O(k^2) heavy reduced-Gram applies, bit-identical.
3. Verify bit-identical vs oracle, measure wall-clock.
