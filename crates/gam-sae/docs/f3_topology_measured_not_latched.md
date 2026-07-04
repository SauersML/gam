# Topology as a measured property, not a latched race winner (reviewer F3)

## The problem

The SAE-manifold pipeline adjudicates an atom's topology with a **race**: it
fits a fixed library of typed candidates (`Periodic`/circle, `Torus`,
`Sphere`, `Cylinder`, `EuclideanPatch`, `Duchon`, `Linear`, `Poincare`) and
crowns the best-scoring one. This is model selection among a fixed menu, and it
*always* returns a winner — even when every candidate is misspecified. A set of
seven discrete clusters raced against the circle/line/patch library still comes
back "circle" (or "patch"), with nothing in the certificate to say the winner
is wrong.

Two complementary instruments turn topology back into something **measured**.

## 1. Persistent-homology audit (landed, `persistence.rs`)

For each accepted atom we read its assigned rows' in-atom residual-space
positions — the decoded image points `g_k(t_{ik}) = Φ_k(t_{ik}) B_k` of the
rows the atom owns (hard argmax assignment) — and compute their exact
Vietoris–Rips persistent homology up to H₁ (H₀ = connected components, H₁ =
loops). The measured diagram is confronted with the raced type's prediction:

| raced type | predicts |
| --- | --- |
| Periodic / Torus / Cylinder | 1 component, ≥1 persistent loop |
| Sphere / Duchon / Euclidean / Linear / Poincare | 1 component, no loop |

A disagreement raises the first-class `AtomTopologyPersistence::contested`
certificate flag, which the probe planner reads to re-adjudicate rather than
trust the winner. The three canonical cases:

- **clean circle** → one dominant H₁ bar, one component → agrees with a raced
  circle;
- **7-cluster ring forced through a circle fit** → seven persistent H₀ bars →
  `contested` (disagrees with *every* connected candidate in the library);
- **straight line** → no loop, one component → clean against a line, and
  `contested` against a circle (a predicted loop that isn't there).

### No magic constants (SPEC.md)

The filtration values *are* the exact pairwise distances — no scale grid, no
bucketing. "How many components / is there a loop" is decided by a
**dominant-gap** test on the bar lengths: the finite H₀ merge scales of a
connected manifold are all the local point spacing (a narrow band), whereas `c`
genuine clusters produce `c − 1` merges at the inter-cluster gap, orders of
magnitude larger. A single log-gap that outweighs *all* the others combined is
accepted as the cluster cut; on the roughly uniform spacings of one connected
manifold no such dominant gap exists, so the count is `1`. The only compute-side
number is the farthest-point subsample cap `PERSISTENCE_MAX_POINTS` — a budget
on the `O(m³)` triangle enumeration, above the covering number of any modest
atom, mirroring the in-tree `SHAPE_BAND_MAX_POINTS` band ceiling.

## 2. Atlas-first inversion (landed as a demo, `atlas_nerve`)

The inversion **measures-then-imposes**. Cover the atom's points with
overlapping typed-free local charts (Duchon patches, glued in production by the
in-tree `chart_transfer` pulled-back operators
`A_kj = (JₖᵀJₖ)⁻¹ JₖᵀJ_F J_j`), then read the topology from the **nerve** of the
cover — the graph whose vertices are charts and whose edges join overlapping
charts. The nerve's first Betti number `b₁ = E − V + C` is the loop count:

- `b₁ = 1, C = 1` → the charts close into a cycle → `S¹` (circle);
- `b₁ = 0, C = 1` → the charts form a path → an arc.

`atlas_nerve` demonstrates the recovery on synthetic data: the nerve of a circle
cover returns `S¹` and the nerve of an arc cover returns a path graph. The chart
count is data-derived (`⌈√n⌉` farthest-point landmarks); the nerve edges are the
parameter-free **witness-complex** 1-skeleton (each point witnesses an edge
between its two nearest charts — the Voronoi adjacency of the atlas), so there is
no overlap radius to tune. The recovered topology is then what a subsequent typed
refit *imposes*, closing the assume-then-race loop into measure-then-impose.

In production the geometric overlap is replaced by the `chart_transfer`
transport certificate (an edge where the pulled-back operator between two charts
is near-isometric), which is why the gluing algebra already lives in-tree; the
synthetic demo isolates the nerve-topology step that sits on top of it.
