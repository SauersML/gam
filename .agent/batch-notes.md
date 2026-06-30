# #159 / #160: sae_manifold_fit kwarg-alias contract

Both issues were closed citing commit 7b0a3455 and a `_ASSIGNMENT_ALIASES`
map + `n_atoms` conflict detection that DO NOT exist in the tree. This branch
implements the documented contract for real:

- `assignment` / `assignment_prior` resolve through one validator
  (`_canonical_public_assignment`) with an expanded alias table
  (ibp/ibp-map -> ibp_map, gated/jump_relu -> jumprelu). Conflicting values
  raise an eager ValueError naming both.
- `K` / `n_atoms` aliases; conflicting values raise an eager ValueError.

Tests: tests/test_sae_manifold_kwarg_aliases.py (14 cases).
