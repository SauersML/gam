# Atlas #2280: the Swiss-roll hole localized

The September 7 MSI diagnostic identifies a representative of the erroneous
Swiss-roll `H1` class. It does **not** establish a repaired atlas or revive the
previously refuted angular-injectivity and ambient-turn hypotheses.

The deterministic `swiss_roll(80, 16)` fixture still reads:

```text
charts=72 b0=1 b1=1 b2=0 chi=0 orientation=trivial
open_tri=0 unsigned_tri=315 signed_b0=1 signed_b1=1 dropped_centers=0
```

An independently reconstructed nerve uses **every nonempty patch intersection**,
including overlaps too small to fit a transition. It enumerates all simplex
orders with actual common-membership intersections, not a clique expansion of
pairwise edges. Its Betti signature remains `(1, 1, 0)`. Thus the transition
minimum-overlap gate omits real intersections, but those omissions do **not**
explain this fixture's erroneous hole.

The diagnostic reduces graph cycles modulo all triangular boundaries with exact
`GF(2)` elimination. It checks that the resulting representatives have dimension
equal to the independent boundary-rank `b1`; spanning-tree chords alone are not
treated as manifold cycles. Its surviving cycle is:

```text
patch 0 -> patch 45 -> patch 18 -> patch 48 -> patch 0
```

| Patch | Center row | Intrinsic grid coordinates `(t_index, h_index)` |
| --- | ---: | --- |
| 0 | 0 | (0, 0) |
| 45 | 13 | (0, 13) |
| 18 | 143 | (8, 15) |
| 48 | 131 | (8, 3) |

These patches belong to the **inner end of the roll**, not different windings.
The two diagonals omitted by the transition list have distinct singleton
intersections:

| Patch pair | Shared rows | Intrinsic grid coordinates |
| --- | --- | --- |
| (0, 18) | `[88]` | (5, 8) |
| (45, 48) | `[87]` | (5, 7) |

The full membership nerve restores these edges and their existing triangles,
yet its hole remains. On these four vertices, common row 88 supplies triangle
`(0,18,45)` and row 87 supplies `(0,45,48)`. The other two triples have no common
row. The cycle therefore points to the realized cover and its sampled
intersections. Whether a continuous chart-domain intersection is being missed,
or the actual domains leave a gap, still needs geometric evidence; the present
measurement does not decide between them.

The Möbius control `mobius_strip(40,10)` retains `(b0,b1,b2)=(1,1,0)` and twisted
orientation in both nerve constructions, despite 28 omitted small-overlap
edges. Its representative follows the band. The torus control is authored but
was not executed in the first fail-fast batch.

The initial focused run executed two tests: one passed and the Swiss-roll
acceptance failed in 0.154 seconds. Twelve other selected checks were not run
because the runner stopped at the failure; they are not passing evidence.
Log: `bench/measurements/issue_triage_20260907/atlas-trace-focused.log`.
Source: `crates/gam-sae/src/manifold/tests_nerve_membership_2280.rs`, registered
under `atlas_topology`. The executable was built from the current MSI working
source, including preexisting edits, with 16 code-generation units and eight
assigned CPUs in 79 seconds. Subsequent execution can reuse its captured
nextest binary metadata without rebuilding.

The new acceptance requires the known contractible Swiss sheet to have `b1=0`
and requires topology to equal the complete membership nerve. The expected
answer has not been changed to make the currently failing implementation pass.
