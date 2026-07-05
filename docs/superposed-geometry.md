# Superposed Geometry: what the code already is

This is the engine-to-theory map for the manifold-SAE fit
(`gamfit.sae_manifold_fit`, the `gam-sae` crate). The structure memo
("Superposed Geometry", Part VI) argues that the machinery already shipped in
this crate is not a heuristic pipeline that happens to find shapes — it is a
concrete fragment of a theory of *identifiable structure under superposition*.
Each mechanism the fit runs is an operational form of a theory object. This
document is the dictionary between the two, kept tight and honest: where a claim
is proven it says so, where it is a schema or a conjecture it says that too.

Nothing here changes behavior. It records what the existing code *is*, so the
next reader inherits the frame instead of re-deriving it. The in-code comments
in `structure_harvest.rs`, `manifold/isa_seed.rs`, `identifiability.rs`,
`chart_canonicalization.rs`, and `manifold/certificate.rs` carry the same
mapping at each mechanism.

---

## The one-line thesis

**Superposition ambiguity is a flatness disease, and curvature is the cure.**

A dictionary of *flat* (linear) co-firing directions is generically
**non-identifiable**: any invertible recombination `GL(d)` of a co-active linear
subspace reconstructs the data identically, so nothing in the loss prefers one
labeling over another. The gauge groupoid — the set of transformations that
leave the reconstruction unchanged — is huge. **Curved** atoms are generically
**rigid**: by jet transversality the second-order osculation of two generic
embeddings is infinite-codimension, so a curved atom's gauge groupoid collapses
to `Diff(M) × Sym(F)` (reparameterize the chart, permute identical atoms — and
nothing else). Circles are therefore the optimizer's **equilibrium response** to
superposition, not curiosities: given the freedom to be flat or curved, the fit
becomes curved exactly where curvature is what buys identifiability.

Everything below is a corollary of that thesis or an instrument that measures it.

---

## The four slogans

| Slogan | What it means | Where it lives in the engine |
| --- | --- | --- |
| **Curvature is identifiability** | Flat co-firing subspaces admit any `GL(d)` gauge (unidentifiable); curvature collapses the gauge to `Diff × Sym` (rigid). The topology race chooses curvature *because* it breaks the flat gauge. | `structure_harvest.rs` birth/topology race (`topology_candidates_for_dim`, `fit_topology_candidate`, `race_birth_topology`, `born_atom`); `manifold/certificate.rs` curvature-budget verdict (`curved_dictionary_global_optimality_verdict`, `SAE_CERT_CURVATURE_CONSTANT`, `atom_curvature_bound`). |
| **Persistence is bits** | The description length an atom earns is its persistent (noise-robust) signal, not its momentary reconstruction gain. Occupancy of a topological feature is what pays. | Persistence-homology topology audit (`AtomTopologyPersistence`), the RLCT-½-per-dimension accounting behind the `rank_eff = 0` veto in `identifiability.rs`, and the birth/death e-value gates. |
| **Binding is transport** | Comparing two fits/layers means transporting one to the other's canonical gauge. Two representations are "the same structure" iff they agree up to the residual stabilizer. | `chart_canonicalization.rs` (arc-length / isometry-flow slices, `recompose_decoder_exact_ls`, `transport_ladder`); Prop H residual-stabilizer verdicts in `identifiability.rs`. |
| **Symmetry is charge** | A residual gauge freedom is a conserved quantity: the symmetry group the fit is identified *up to* is a physical property of the model, reported not assumed. | `identifiability.rs` `GeneratorFamily` / `VerdictProvenance`; `residual_gauge` certificate. |

---

## Part VI table: engine mechanism → theory object

| Engine mechanism (file) | Theory object | Proof status |
| --- | --- | --- |
| Realized-rank / Marchenko–Pastur-edge test on the border-block Jacobian, per atom (`identifiability.rs` `residual_gauge`, RRQR generator pinning) | **Empirical Terracini certificate (Theorem A).** Border-block Jacobian rank `= Σ_k (d_k+1)` is the Terracini tangent dimension of the join/secant variety of the atom manifolds; the MP edge tests whether those tangent directions are *independent* (signal above the noise edge). Identifiability stops being an assumption and becomes a certificate the fit carries. | exact (given the rank model); MP-edge separation is standard RMT |
| `rank_eff = 0` veto of a degenerate-tangent generator/atom (`identifiability.rs`) | **Degenerate-tangent exclusion of Theorem A + RLCT-½ necessity.** A null atom's real-log-canonical-threshold contribution is ½ per genuine dimension, so a rank-0 atom is asymptotically cheap and buys no identifiability — the veto is a validity condition, not a heuristic. | exact (exclusion); RLCT-½ is standard singular-learning theory |
| ISA κ-contrast births (`manifold/isa_seed.rs` `certify_plane`, the `(κ−2)²` objective; consumed in `structure_harvest.rs` birth channel) | **Measure-level identifiability of support-invisible atoms (Proposition 1).** A centered circle's cone `ℝ₊·Y` *is* the plane `P∖{0}` — support-indistinguishable from a 2-plane — so it is identifiable *only* through its radial law. `κ = E[r⁴]/E[r²]²` (= 1 dense circle, = 2 Gaussian plane, = 1/q gated) is the lowest-order separating statistic. This is why the κ producer must exist *alongside* the support/rank test: they see complementary halves — **measure vs support** — and neither subsumes the other. | exact (the κ anchors are closed-form population values); fourth-order necessity proven by the Davis–Kahan blend argument |
| Birth/topology race: line vs circle vs torus/sphere/cylinder/patch (`structure_harvest.rs`) | **Curvature is identifiability, made operational.** The race replaces an unidentifiable flat co-firing blob with a rigid curved atom whenever curvature earns its keep, adjudicated by TK-normalized REML so cross-topology evidence is commensurable. | schema (the race is the mechanism; that the curved winner is the identifiable one follows from the flatness thesis) |
| Chart canonicalization: arc-length (`d=1`), isometry-flow (`d=2` torus/patch), conformal-boost (sphere) (`chart_canonicalization.rs`) | **A slice of the gauge groupoid; residual = the atom's continuous linear stabilizer (Prop H).** The intrinsic smoothness penalty makes every reparameterization equal-cost, so the fit lives on a full `Diff(M)` orbit; canonicalization picks an exact, image-frozen representative (a gauge *retraction* leaving data-fit and smoothness invariant). The residual gauge is the finite isometry group of the reference manifold — exactly the uniqueness condition for canonical layer transport. | exact (image-frozen retraction, honest-refuse gate); Prop H residual group is exact per topology |
| `residual_gauge` generator enumeration + `VerdictProvenance` (`identifiability.rs`) | **Symmetry is charge.** Each generator's pinned/unpinned verdict slices the gauge groupoid at this fit; the surviving unpinned generators are the model's isotropy subgroup — the conserved residual freedom, reported per generator so partial flatness stays visible. | exact (curvature-flatness test in the fit metric) |
| REML race vs persistence/topology audit on the same atom ("measured, not latched") (`identifiability.rs`, `AtomTopologyPersistence.contested`) | **One statistic in two notations (Corollary F).** Asymptotically the marginal-likelihood evidence and the persistence audit must agree, so a finite-n disagreement is a *misspecification detector*, not a conflict to resolve by fiat. | standard (asymptotic agreement); the detector reading is a schema |
| Curved-dictionary global-optimality verdict (`manifold/certificate.rs`) | **Curvature-side rigidity certificate.** Certifies the basin stationary point unique up to the residual gauge when incoherence `μ̂` is small and curvature `κ̂` is present and controlled (`C_κ κ̂ < 1`). Conservative by construction: `CertifiedGlobal` is never wrong, `Uncertified` is honest "cannot decide". | conjecture (the threshold constants are conservative but not sharp); conservatism direction is exact |
| Khemakhem iVAE 2k+1-distinct-auxiliary precondition (`identifiability.rs` `ConditionalPriorIvae::new`) | The classical global identifiability gate the per-generator residual-gauge certificate generalizes (one yes/no gate → per-generator Prop H stabilizer). | exact (Khemakhem et al. Theorem 1) |

---

## The learnability trichotomy

Structure is learned in three strictly ordered stages. Each is a different
question, and the instruments that answer them are different — which is why the
engine carries several certificates that look redundant but are not.

1. **Existence** — *is there any structure above noise here?* Answered by the
   Marchenko–Pastur edge (`isa_seed.rs`) and the birth e-value gate: a direction
   above the MP edge is real signal, below it is a fluctuation.

2. **Dimension** — *how many independent tangent directions does it have?*
   Answered by the empirical Terracini rank (`identifiability.rs`): the
   border-block Jacobian rank `= Σ_k(d_k+1)`, with the `rank_eff = 0` veto
   excluding degenerate tangents.

3. **Topology** — *what shape is it (circle? line? torus?)* Answered by the κ
   measure test (`isa_seed.rs`), the topology race (`structure_harvest.rs`), and
   the persistence audit — the measure-level and occupancy-level instruments,
   because topology is invisible to support and dimension alone.

**Fidelity cannot buy topology, only occupancy.** Reconstruction EV is a
fidelity statistic: it rises monotonically as an atom absorbs residual variance,
and it is *blind* to whether the coordinate honestly covers the shape. The
planted-circle failure that motivated chart canonicalization compressed a full
loop into ~1 radian of coordinate span at EV 0.9979 — image-perfect,
chart-dishonest. So a high-fidelity fit tells you an atom *occupies* a feature;
it cannot tell you the feature's *topology*. Topology is certified only by the
measure (κ), the persistence, and the arc-length occupancy of the chart — never
by EV. This is the deepest reason the engine refuses to read structure off
reconstruction loss alone.

---

## Honesty ledger: proof status of the theory objects

The memo's discipline is that every claim carries its epistemic status. This
crate inherits it — a certificate never over-claims, and this ledger records how
much weight each mapping above can bear.

- **Exact** — proven from the definitions, no regularity conditions to hope for.
  Proposition 1's κ anchors (closed-form population moments); the Terracini
  tangent-dimension identity; the image-frozen chart retraction and its
  honest-refuse gate; the Prop H residual stabilizers per topology; the RLCT-½
  degenerate-tangent exclusion; the conservatism *direction* of the
  global-optimality verdict (a certified verdict is never wrong).

- **Standard** — true by established results in RMT / singular-learning theory /
  asymptotic statistics, imported rather than re-proven. Marchenko–Pastur edge
  separation; RLCT-½-per-dimension accounting; the asymptotic agreement of the
  REML evidence and the persistence audit (Corollary F's premise).

- **Schema** — a correct mechanism whose full theorem is stated but whose
  constants/generality are not pinned. The topology race as the operational form
  of "curvature is identifiability" (the race is exact; that its curved winner is
  *the* identifiable representative is the flatness thesis applied); the
  measured-not-latched doctrine as a misspecification detector.

- **Conjecture** — believed, conservative in the safe direction, not sharp. The
  curved-dictionary global-optimality *threshold* (`μ̂ ≤ c0 · a_floor² ·
  (1 − 1/SNR) · (1 − C_κ κ̂) / K`): the constants shrink the certified region
  relative to the true sharp threshold, so it is safe but not tight, and the
  sharp constant is open.

---

## Why the certificates are not redundant

A recurring question is why the engine carries a support/rank test *and* a
measure (κ) test *and* a curvature certificate *and* a chart-canonicalization
slice. The trichotomy answers it: they attack different stages and see different
halves of identifiability.

- **Support/rank (Terracini)** sees whether a tangent direction *exists and is
  independent*. It is blind to a centered circle, whose support is the whole
  plane.
- **Measure (κ)** sees the *radial law on that support*. It is the only test
  that separates a circle from the plane it lives in.
- **Curvature certificate** sees whether the *global basin is rigid* — whether
  the flat gauge has actually been broken enough to pin the fit uniquely.
- **Chart canonicalization** removes the *remaining continuous freedom* (the
  `Diff(M)` orbit) so that two fits become comparable at all (binding =
  transport).

Remove any one and a whole class of structure becomes either invisible or
gauge-ambiguous. That is the sense in which the code is a *fragment of the
theory*: the theory says exactly this set of instruments is necessary, and the
crate ships exactly this set.
