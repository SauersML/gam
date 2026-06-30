# Issue #1625 — SAE row-jet oracle convergence

## Status: issue CLOSED on GitHub with 2 fixes on main (c65fc4e, a5f099e)

## Plan
1. Reproduce CURRENT state of the two oracle tests on main.
2. The issue noted ADJACENT failures (4 dense-FD logdet-trace oracles) still red.
3. Verify whether closure is genuine; if tests are still red, fix the true root cause.
4. Expand: harden inner (t,β) Newton convergence on marginal fixtures.

## Tests in question
- manifold::tests_row_jet_and_outer_objective_780::sae_row_jet_program_matches_production_row_jets_on_converged_cache
- manifold::tests_row_jet_and_outer_objective_780::batch4_jet_lanes_match_scalar_hand_row_jets

## ROOT CAUSE — ibp_map adjoint (verified by derivation)

H column-k block: H_{(i,k),(j,k)} = d_k·J_ik·J_jk + δ_ij·w·s_k·c_ik
  where d_k=w·s'_k, J=z_jac, c=logit_curvature, s=score, s'=score_derivative, s''=score_second_derivative.
  (the J² self-term lives in the rank-one, the assembled diagonal is the c-diagonal w·s_k·c_ik.)

Full ∂log|H|/∂ℓ_wk = tr(H⁻¹ ∂H/∂ℓ_wk) = P1+P2+P3+P4:
  P1 = dd_k·J_wk·(uᵀGu)                      [rank-one coeff M-deriv; dd_k=w·s''_k]
  P2 = 2·d_k·c_wk·x_w                         [rank-one J explicit-deriv; x=H⁻¹u]
  P3 = w·s'_k·J_wk·Σ_i G_ii·c_ik              [c-diagonal M-deriv]
  P4 = G_ww·w·s_k·(dc/dz)_wk·J_wk             [c-diagonal explicit-z deriv]

Current code:
  INLINE  = diag(P2) + P4         (G_ww·local_logit_third)
  PASS1   = diag(P1) + P3         (m_channel = w(s''J²+s'c))
  PASS2   = 0.5·P1 + 0.5·P2       (the ½ on term A and the missing ×2 on term B)
  SUM     = diagP1+diagP2+0.5P1+0.5P2+P3+P4   ≠ P1+P2+P3+P4.  BUG.

FIX: PASS2 must add ONLY the off-diagonal remainder:
  term A: dd_k·J_wk·(uᵀGu − Σ_i G_ii·J_ik²)
  term B: 2·d_k·c_wk·(x_w − G_ww·J_wk)
All quantities available (inv_diag = G_ii per site in ibp_logit_sites).
