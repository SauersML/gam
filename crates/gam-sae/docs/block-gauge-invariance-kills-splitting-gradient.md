# The block gauge invariance kills the splitting gradient

## Claim

The group-l2 block gate removes the feature-splitting descent path that SASA
identifies for l1-SAE training. The reason is structural: the block objective
prices a selected block by the norm of its within-block code, not by a preferred
coordinate system inside the block. Once the gate only sees `||z_g||_2`, every
change of basis `z_g -> R z_g` with `R in O(b)` is a gauge transformation rather
than a new sparse representation with a different penalty.

## Splitting-Dynamics Theorem

Fix a block `g` with an orthonormal decoder frame `D_g in R^{b x P}` and a tied
within-block code `z_g = x D_g^T`. The block gate and group penalty are

```text
gate_g = ||z_g||_2
penalty_g = lambda ||z_g||_2 .
```

For any orthogonal matrix `R in O(b)`, replacing the frame by `R D_g` changes the
coordinate vector to `R z_g`, leaves the reconstruction unchanged, and leaves the
penalty unchanged:

```text
(R z_g)^T (R D_g) = z_g^T D_g
||R z_g||_2 = ||z_g||_2 .
```

Thus the within-block coordinates carry amplitude and direction, but they do not
define separate priced features. The penalty is constant on the entire
within-block gauge orbit.

SASA's l1 descent mechanism depends on the opposite property. When a coherent
feature is represented by several near-collinear one-dimensional atoms, the
training objective can trade reconstruction-neutral rotations against l1 penalty
anisotropy. The l1 ball has coordinate corners, so changing the atom/code
factorization inside an almost fixed reconstruction can expose a lower-penalty
path. That anisotropy is the energy source for feature splitting.

The group-l2 block gate deletes that source. Inside one block the penalty ball is
round, so reconstruction-neutral rotations have zero first-order penalty
gradient. There is no distinguished coordinate axis for the optimizer to split
toward.

## Why splitting is strictly worse at equal reconstruction

Take a block code `z in R^b` with at least two nonzero coordinates. The same
reconstruction can be written either as one selected rank-`b` block or as `b`
selected singleton atoms using the same decoder rows:

```text
one block:       x_hat = sum_r z_r D_r
b singletons:   x_hat = sum_r z_r D_r
```

The fit is identical, but the price is not. The block pays

```text
lambda ||z||_2 + tau log2 choose(G, k),
```

while the singleton split pays

```text
lambda ||z||_1 + tau log2 choose(G b, k b)
```

for the corresponding atom-level support. For a genuinely multidimensional
activation, `||z||_1 > ||z||_2`; and the atom-level support catalogue is larger
than the block catalogue. Therefore the split representation is strictly more
expensive at equal reconstruction. This reverses the l1-SAE splitting incentive:
splitting is not a descent direction, it is an MDL and group-lasso cost increase.

## Relation to implementation

The sparse dictionary block lane enforces the theorem at the primitive surface:

- `block_gates` routes only by the group-l2 norm of the block projection.
- `row_loss` reconstructs through the selected block projectors, so basis changes
  inside a selected block do not change fit.
- `block_tests::gauge_invariant_selection_and_loss_under_block_rotation` pins
  the practical gauge invariance of routing and loss.
- `block_tests::splitting_dynamics_theorem_group_l2_kills_splitting_gradient`
  pins the theorem's two numeric claims: `O(b)` rotations leave the group-l2
  penalty invariant to `1e-12`, and singleton splitting strictly increases the
  group-lasso plus selection cost at equal reconstruction.

## Related work

- SASA: Seyed Arshan Dalili and Mehrdad Mahdavi, "Subspace-Aware Sparse
  Autoencoders for Effective Mechanistic Interpretability",
  [arXiv:2606.06333](https://arxiv.org/abs/2606.06333).
  SASA proves that l1-SAE training has descent paths that actively prefer feature
  splitting, and proposes learned decoder subspaces with block sparsity to
  consolidate multidimensional features.
- Geometric Wall: Eslam Zaher, Maciej Trzaskowski, Quan Nguyen, and Fred Roosta,
  "The Geometric Wall: Manifold Structure Predicts Layerwise Sparse Autoencoder
  Scaling Laws", [arXiv:2605.09887](https://arxiv.org/abs/2605.09887). The
  geometric-wall view is complementary:
  layerwise SAE scaling failures reflect manifold curvature and intrinsic
  dimension, while the block-gate theorem explains why pricing whole subspaces
  removes a coordinate-level splitting incentive inside each selected block.
