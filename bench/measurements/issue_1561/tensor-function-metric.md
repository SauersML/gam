# Tensor roughness still depends on coefficient coordinates

Source audit, 2026-09-08, main at `010538f68`.

`SPEC.md` requires penalties on the final function. The current
`build_tensor_bspline_basis` constructs a margin's roughness block as
`S_x ⊗ I_z`. The other margin's function Gram is computed only when double
penalization is enabled, and is used for the joint-null ridge, not for this
roughness block. Both committed main and the shared working tree have this
construction. The earlier null-ridge correction therefore does not resolve
the primary tensor roughness metric.

For `f(x,z) = b_x(x)^T C b_z(z)`, integration of the squared derivative in x
requires `S_x ⊗ G_z`, where `G_z = integral b_z(z)b_z(z)^T dz`. In d dimensions,
each differentiated margin contributes its derivative Gram and every other
margin contributes its function Gram. Replacing those Grams by identities is
equivalent only in a basis orthonormal under the declared function measure.
The raw knot-product bases used here are not orthonormal under that measure.

## Exact counterexample

Take the linear spline margin `b(t) = (1-t, t)` on `[0,1]`, with first-derivative
penalty. Before scalar normalization,

```
S = [[ 1, -1],       G = [[1/3, 1/6],
     [-1,  1]]            [1/6, 1/3]]
```

The x-major coefficient vectors for `f=x` and `f=xz` are `(0,0,1,1)` and
`(0,0,0,1)`, respectively.

| Surface | Current x block, S ⊗ I | Integrated squared x derivative, S ⊗ G |
| --- | ---: | ---: |
| x | 2 | 1 |
| xz | 1 | 1/3 |

The respective ratios are 2 and 3. A scalar normalization or change of the
margin's smoothing parameter cannot correct both. Equivalently, under an
invertible marginal basis change `b -> b A`, the function Gram transforms as
`G -> A^T G A`; reconstructing an identity instead does not preserve the
quadratic form of the same surface. This is an algebraic source finding, not
a numerical experiment or proof that correcting it will close the RMSE gaps.

## Required implementation scope

The canonical penalty construction and its factored runtime must change
together. The current `KroneckerInvariantStructure` diagonalizes marginal
roughness matrices with ordinary Euclidean eigenvectors. Carrying marginal
function Grams permits a generalized eigensystem `S U = G U D`,
`U^T G U = I`, retaining the separable sum and cached per-margin work. Every
consumer that currently treats U's inverse as its transpose must be audited.
Coefficient/covariance transport, penalty determinants, frozen predictions,
and the null-function ridge must describe the same transformed problem.

The `t2` construction also uses Euclidean range/null projectors and requires
a function-metric audit. Replacing only the canonical `te` matrices while
leaving a factored solver on `S ⊗ I` would make execution paths fit different
models. Disabling the fast path is not a completed performance solution.

Validation should compare tensor energies against independently integrated
separable functions, exercise nonorthogonal marginal coordinate changes, and
check factored/canonical solves and frozen prediction parity. Then replay the
unchanged Gaussian and Poisson fixtures and the complete quality manifest.
The energy identity alone establishes neither better recovery nor the issue's
required full-suite significance result.

## Execution availability

At this audit, direct MSI acn112 and acn116 checks failed. Read-only login-node
SLURM queries reported the compute partitions as `inval`/`unk`/`drain`; acn116
reported `IDLE+DRAIN+INVALID_REG`. No diagnostic, build, or numerical run was
started locally or on a login node, and no queued or foreign job was modified.
The exact source finding remains available for implementation and verification
when a compute node is usable. Issue #1561 remains open.
