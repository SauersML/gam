# Issue #1082 — quality tests exceed 360s budget

## Current state (run 28399514029, 2026-06-29)
Of the original ~17 timeouts, prior fleet work resolved all but ONE:
- **TIMEOUT**: `survival::quality_competing_risks_truth_recovery::joint_competing_risks_transformation_recovers_true_cause_specific_cifs` (360s)

## Near-budget risks (PASS but slow — flake risk)
- pymc_hmc_binomial_penalized_vs_unpenalized (192s)
- fit_quality_stress::hifreq_tensor_k10 (185s)
- pymc_nuts_binomial_logit (126s)
- synthetic_multinomial_deviance_identity (83s, REF_ERROR)

## Plan
1. Root-cause the competing-risks CIF timeout (profile the hot loop). Magic-by-default: make the loop converge / cut redundant work, NO budget bumps, NO accuracy loss.
2. Harden the near-budget PASS tests if the slowness is gam-side.
