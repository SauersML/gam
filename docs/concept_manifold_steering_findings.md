# Concept Manifold Steering Findings

## 1. Executive Summary

- Match the latent dimension to the supervised concept rank before doing
  anything clever.
  In the cogito color study, HSV is rank three, and `auto_exp_53` found that
  `d=3` is both BIC-optimal and CV-optimal.
  At `d=3`, held-out hue stayed at `R^2=0.687`, mean CV reached `R^2=0.680`,
  and extra dimensions above three were unused by reduced-rank ridge.

- Use a joint supervised gauge fix, not a sequential residual search.
  `auto_exp_38` recovered HSV with
  `R^2(hue,sat,val)=(0.700,0.657,0.719)` at `d=3`.
  `auto_exp_51` then showed that fitting HSV first, projecting it out, and
  asking ARD plus Circle to find a residual hue ring fails:
  the best residual cyclic correlation was only `|rho|=0.013`, far below the
  `|rho|>=0.30` target and the oracle plane.

- Cyclic topology is not enough when the concept is sub-dominant in variance.
  `auto_exp_47` found that unsupervised Circle on raw principal components
  failed (`|rho|=0.041`, circular rho `-0.021`), even though an oracle
  `PC2+PC4` plane had circular correlation `-0.720`.
  `auto_exp_49` and `auto_exp_50` showed why:
  reconstruction-driven ARD over PCs follows variance mechanics, not semantic
  cyclic identity.

- Steering directions must be semantic anchor offsets or supervised semantic
  axes, not raw local tangents.
  `auto_exp_44` proved that tangents causally affect generation at moderate
  strength, shifting warm-language ratios by `+0.165` at red and `+0.317` at
  blue for `alpha=2`.
  It also falsified the tangent-as-semantics story:
  `t1_at_blue` pushed language warmer, not cooler, and `alpha=5` collapsed.

- The recipe generalizes beyond HSV, but the evidence is still single-model
  and single-layer.
  `auto_exp_54` applied the same gauge-fix recipe to name-semantic targets and
  got CV `R^2=0.763` for modifier count, `0.733` for monoword, and `0.620` for
  template sigma.
  `auto_exp_93` then showed that the modifier-count name axis is nearly
  orthogonal to hue: `rho=+0.029` against the HSV hue axis and `rho=-0.108`
  against raw hue.

## 2. Background

This document summarizes the cogito color-manifold investigation as a recipe
for finding, validating, and steering concept manifolds in model activations.

The concrete setting was cogito layer 40 on xkcd color-name prompts.

The data object was a collection of hidden-state responses.

The basic unit for the color investigation was a per-color centroid:
many prompt templates for the same color name were averaged into one row.

That centroiding choice matters.

`auto_exp_48` compared raw per-prompt fitting to centroid fitting.

On raw per-prompt rows, held-out `R^2` was much lower:
`0.264` for hue, `0.121` for saturation, and `0.184` for value.

Aggregating raw predictions back to color centroids improved those numbers but
still lagged the centroid fit by roughly `0.350` mean `R^2`.

The recipe below therefore treats nuisance prompt variation as noise unless a
separate experiment shows it is signal for the concept of interest.

A concept manifold, in this framing, is not just a low-dimensional embedding.

It is a low-dimensional, coordinate-bearing subspace that satisfies four
conditions.

First, the coordinates should predict the concept labels or concept-derived
auxiliary variables.

Second, the coordinates should be identifiable enough that repeated fits do not
rotate the concept into arbitrary axes.

Third, the coordinates should support causal interventions in the model.

Fourth, failures of topology, sparsity, or steering should be tested against
nulls, not promoted into geometry stories.

The main lesson is that concept identity and representation variance are
different objects.

The top-variance directions in an activation space can carry useful structure,
but they are not guaranteed to carry the concept the investigator cares about.

Hue was present in cogito layer 40, but not as the top unsupervised cyclic
plane.

Name semantics were present too, but they were not the same as perceptual hue.

The final picture is a decomposition:
perceptual color structure and name-semantic structure coexist, and the recipe
must fix the gauge before it asks the model to reveal either one cleanly.

### Concept Manifolds

A concept manifold is a local chart of a model's internal representation that
tracks a human-interpretable factor.

For HSV color, the chart has at least three degrees of freedom:
hue, saturation, and value.

For color-name syntax, the chart can instead track variables such as whether a
name is one word, how many modifiers precede the base color word, or how
template-sensitive the representation is.

The chart is not the same as a visual plot.

A t-SNE, PCA, or other projection can make a cloud look structured while still
failing to identify the coordinate system needed for steering.

The chart is also not the same as topology.

A Circle latent can be the right topology for hue, but `auto_exp_47` showed
that unsupervised Circle does not find hue if reconstruction loss is dominated
by saturation and value.

In this investigation, the useful chart was produced by a supervised
auxiliary-conditional fit.

The auxiliary variables supplied the gauge:
they broke the rotation symmetry of the latent block.

Once that happened, free structure could emerge in the remaining block, but
only under the right conditions.

### Steering

Steering is the causal test of a concept manifold.

A coordinate system that predicts labels but cannot move generation may still
be descriptive rather than mechanistic.

The cogito steering test in `auto_exp_44` used live interventions at layer 40.

The experiment applied tangent-probe directions from a color manifold to
color-agnostic prompts.

It then scored generated text by the warm-language ratio:
warm words divided by warm plus cool words.

The causal result was real at moderate strength.

For `t1_at_red`, the warm ratio moved from `0.627` at `alpha=0` to `0.792` at
`alpha=2`, a shift of `+0.165`.

For `t1_at_blue`, the warm ratio moved from `0.517` at `alpha=0` to `0.833` at
`alpha=2`, a shift of `+0.317`.

The semantic result was negative.

The blue tangent did not push toward cool language.

It pushed warm language too.

At `alpha=5`, both completed tangent families collapsed to a warm ratio around
`0.417`, with looser and less coherent text.

The operational conclusion is narrow and important:
local tangents are derivatives of the fitted chart, not global semantic
directions.

For steering, the direction should be an anchor offset, such as concept anchor
minus manifold center, or a supervised semantic axis.

Raw tangent columns need projection into a globally fitted semantic frame
before they should be treated as directions like "blue" or "cool".

## 3. The Validated Recipe

This section is the cookbook form of the investigation.

It is written as a recipe for applying the same process to another concept.

The examples use color, but the claims are about the workflow.

### Step 1: Choose `d` To Match Concept Rank

Start by deciding the rank of the supervised target.

Do not pick `d` because it is convenient.

Do not overparameterize and hope ARD will clean up the chart.

Do not use a topology prior before knowing whether the target is actually
recoverable in the chosen dimension.

For HSV, the target rank is three.

`auto_exp_53` swept `d in {2,3,4,5,6}` with reduced-rank ridge on cogito layer
40 centroids.

The result was sharp.

At `d=2`, hue and saturation were partly recovered, but value was crushed:
in-sample value `R^2=0.174`, mean CV `R^2=0.470`.

At `d=3`, the full HSV block appeared:
in-sample `R^2=(0.700,0.658,0.720)`.

Held-out hue was `R^2=0.687`.

Mean held-out `R^2` was `0.680`.

BIC was best at `d=3`, with `BIC=-2383.0`.

For `d=4`, `d=5`, and `d=6`, the in-sample and CV scores were essentially
unchanged.

The active-axis count stayed at three.

The extra columns were dead under reduced-rank ridge and were penalized by BIC.

The transferable rule is:
when the concept target has known rank `r`, start with `d=r`.

Underparameterization drops the weakest channel.

Overparameterization may be harmless for prediction if the solver self-truncates,
but it reintroduces unnecessary gauge degrees of freedom for downstream
analysis.

### Step 2: Fit A Joint HSV-Style Gauge Fix

Use an auxiliary-conditional or iVAE-style mean gauge to make the latent axes
line up with the supervised concept variables.

In the color study, the winning setting was the HSV-supervised gauge fix in
`auto_exp_38`.

It used `d_aux=6` overall:
three supervised HSV axes and three unconstrained free axes.

The supervised axes recovered:
`R^2(hue)=0.700`,
`R^2(sat)=0.657`,
and `R^2(val)=0.719`.

The important point is not only that HSV was predicted.

The supervised block also fixed the gauge well enough that the free block could
be interpreted.

The free axes had maximum absolute correlations with name features:
`0.463`, `0.667`, and `0.249`.

One free axis captured modifier count around `0.67` and monoword around `0.63`.

Another mixed monoword, modifier count, and template sigma.

The free axes were also mostly orthogonal to HSV, with maximum free-axis HSV
correlation about `0.13`.

This is the companion-block pattern:
a supervised block can break the symmetry that otherwise prevents the free
block from settling into interpretable directions.

`auto_exp_40` checked whether this was only a PCA-prefix artifact.

It implemented a Bishop-1999 PPCA-with-ARD coordinate-descent fit on the
HSV-orthogonal free block.

Across `w_ard in {0.01,0.1,1.0,10.0}`, the name-active count remained two.

The best free correlation stayed around `0.674` to `0.676` with modifier count.

The third free axis never became a strong name-semantic axis.

The method-robust conclusion is that the unsupervised post-HSV free block has
two clear name-active axes in this data regime.

Full supervision can pull a third name axis out of the noise, but unsupervised
ARD does not.

Do not rewrite this as "ARD solved the problem".

ARD was diagnostic after the gauge was fixed.

The gauge fix was the load-bearing move.

### Step 3: For Sub-Dominant Cyclic Concepts, Require Supervised Aux

Hue is cyclic, so it is tempting to start with a Circle latent.

The investigation tested that temptation directly.

`auto_exp_47` compared Euclidean and Circle recovery of hue from raw principal
components.

The unsupervised Circle fit failed:
`|rho|=0.041`, circular rho `-0.021`, full circular MSE `0.0684`.

The top-1 Euclidean baseline was also weak:
`|rho|=0.108`, circular rho `+0.134`.

But the oracle row changed the interpretation.

There existed a `PC2+PC4` plane with circular correlation `-0.720` to true hue.

Its full circular MSE was `0.0294`, less than half the unsupervised values.

So hue-shaped structure was present.

The unsupervised objective simply did not select it.

`auto_exp_49` asked whether ARD over PCs plus Circle could find the oracle
plane without labels.

It could not.

ARD plus Circle got `|rho|=0.032` and circular rho `-0.041`.

Its top alpha PCs were `[9,7,4]`, missing the `PC2+PC4` pair.

ARD plus Euclidean also failed and concentrated on high-index PCs
`[11,14,12]`.

The failure mode was variance inversion:
under a reconstruction-only objective, the ARD weights reward mechanics that
do not correspond to semantic cyclic structure.

`auto_exp_50` partially fixed the scaling symmetry but not the concept problem.

A simplex-normalized alpha variant moved one top alpha onto `PC2` and improved
`|rho|` to `0.170`.

A fixed-sigma variant reached `|rho|=0.195`.

Neither discovered the `PC2+PC4` pair.

Neither reached the `|rho|>=0.30` target.

The transferable rule is:
if a cyclic concept lives in a sub-dominant variance plane, topology and ARD
are not substitutes for supervision.

Use an auxiliary signal that names the cycle.

Then compose Circle or other topology only after the semantic gauge exists.

### Step 4: Use Anchor-Offset Directions For Steering

The steering result should be treated as both success and warning.

`auto_exp_44` proved that concept-manifold probes can causally move text.

At `alpha=2`, the warm-language ratio moved materially for both red and blue
tangent probes.

That is a positive causal result.

The same experiment falsified a common interpretation of tangent vectors.

The blue tangent moved generation warmer, not cooler.

Therefore local tangent identity is not the same as global concept identity.

For steering another concept, prefer these directions in this order.

First, use a supervised semantic axis if the fit has one.

For HSV, that means steering along the recovered hue, saturation, or value
axis, not along an arbitrary tangent column.

Second, use anchor offsets:
anchor representation minus the manifold center.

For color, this would mean an offset for red, blue, green, or another named
anchor, rather than a local derivative at that anchor.

Third, if using tangents, project them into the supervised semantic basis and
interpret only the projected component.

Do not label a tangent column with the anchor name and assume that the column
means the anchor.

The `alpha=5` collapse also gives a practical steering rule.

Estimate a stability budget.

Do not assume larger strength gives cleaner steering.

The cogito tangent probes had a moderate-strength causal window and an
over-strength collapse regime.

### Step 5: Rank Curvature To Predict Steering Stability

`auto_exp_52` computed per-anchor curvature for the HSV-supervised `U_3d`
manifold from `auto_exp_38`.

The forward map was a Nadaraya-Watson Gaussian-kernel smoother over the 949
anchors in standardized PC space.

Curvature was measured as the Frobenius norm of the second derivative.

The curvature range was `14.1` to `132`, with median `34.1`.

The high-curvature anchors included:
indian red at about `1.3e2`,
diarrhea at about `1.2e2`,
bland at about `1.1e2`,
lightblue at about `1.1e2`,
and drab at about `1.0e2`.

The low-curvature anchors included:
dark lilac at about `14`,
yellow orange at about `14`,
pinkish grey at about `15`,
pastel yellow at about `16`,
and light tan at about `16`.

The geometric interpretation was that high curvature tends to occur near the
boundary of the convex hull of the supervised chart, where local neighborhoods
are one-sided.

The steering interpretation is:
high-curvature anchors should destabilize at smaller intervention strengths,
because local quadratic error pushes the activation off the learned manifold
faster.

This is not yet a completed steering experiment.

It is a validated ranking to use in the next steering study.

For a new concept, compute curvature after the supervised gauge fit.

Then stratify steering tests by low-, median-, and high-curvature anchors.

If high-curvature anchors fail earlier, the curvature score is doing useful
operational work.

## 4. Negative Results

The most important outputs of this investigation are the failed prescriptions.

They prevent the cookbook from becoming folklore.

### Sphere-Wins Was A Lift Artifact

`auto_exp_41` ran a free-block topology sweep after the HSV-supervised and
PPCA-ARD setup.

Sphere appeared to win dramatically.

Its BIC was `-423.85`.

Euclidean was `5420.54`.

Circle was `5516.85`.

Cylinder was `6338.31`.

The apparent Sphere-over-Euclidean BIC margin was `5844.40`.

Cross-validation also favored Sphere in the raw comparison:
Sphere mean log likelihood per point was `-0.2752`, compared with Euclidean
`-2.8446`.

This could have become a false discovery:
"the free block is spherical."

`auto_exp_42` ran the necessary null controls.

IID Gaussian covariance nulls reproduced much of the effect:
BIC gap about `4736.5 +/- 232.5`.

IID identity nulls did the same:
BIC gap about `4738.6 +/- 251.1`.

The shuffled null nearly matched the real effect:
BIC gap `5835.9 +/- 4.3`, ratio `0.999` versus real.

Real data had BIC gap `5844.40` and CV gap `2.5694` nats per point.

The verdict was lift artifact.

The scorer's stereographic lift and vMF tails created a Sphere advantage that
null data could reproduce.

Cookbook rule:
never accept a topology win without a null that passes through the identical
scoring path.

### Tangent-As-Color Was Falsified

The `auto_exp_44` steering experiment is easy to misread.

The causal intervention worked.

But the semantic interpretation failed.

At `alpha=2`, red tangent steering warmed language by `+0.165`.

At `alpha=2`, blue tangent steering warmed language by `+0.317`.

The blue tangent therefore did not mean "blue" or "cool".

It meant a local derivative direction at the blue anchor, expressed in the
chart's tangent basis.

At `alpha=5`, both completed tangent probes collapsed to around `0.417` warm
ratio.

Cookbook rule:
do not use raw tangent columns as named concept steering vectors.

Use anchor offsets or supervised semantic axes.

### Sequential Gauge-Fix-Then-Residual Search Failed

`auto_exp_51` tested a natural sequential recipe:
fit the HSV gauge,
orthogonalize the representation against that supervised block,
then run ARD plus Circle on the residual to find a hue ring.

This is exactly the kind of method that sounds principled before it is tested.

It failed.

Bare Circle on the full `K=16` PC space had `|rho|=0.041`.

Bare Circle on the HSV residual fell to `|rho|=0.005`.

Raw ARD plus Circle on the residual reached only `|rho|=0.013`.

Normalized-alpha ARD plus Circle on the residual reached only `|rho|=0.008`.

The target was `|rho|>=0.30`.

The oracle `PC2+PC4` plane remained far better.

The lesson is not "always project out the supervised subspace."

The lesson is the opposite for this recipe:
orthogonalize-then-search can destroy or hide the weaker structure you wanted
the second stage to discover.

If two concept blocks interact in the raw representation, compose the
constraints in a joint fit rather than relying on a lossy residual pipeline.

### ARD Alone Did Not Become A Concept Finder

ARD is useful after the gauge is fixed.

It is not a magic semantic search engine.

`auto_exp_40` found that real PPCA-ARD on the free block did not prune axes and
did not recover a third name-semantic axis.

The alphas stayed small across the sweep.

The best free correlation stayed around modifier count.

The method confirmed a two-axis free-block result rather than discovering
hidden semantics.

`auto_exp_49` and `auto_exp_50` extended the warning to cyclic concepts.

ARD over PCs under reconstruction objectives followed variance and scaling
mechanics.

It did not discover the hue plane.

Cookbook rule:
ARD can help select among already-gauged alternatives.

It cannot by itself define the semantic coordinate system.

### IBP-Gumbel Found Partial Group Differences, Not Clean Classes

`auto_exp_45` used per-row IBP-Gumbel indicators and checked active atoms
across warm, cool, and neutral color groups.

The warm group had more atoms:
`maj_atoms=4`, mean per row `4.13`, typical set `[0,3,4,6]`.

Cool had `maj_atoms=2`, mean per row `3.31`, typical set `[3,6]`.

Neutral had `maj_atoms=2`, mean per row `3.61`, typical set `[3,4]`.

This is a partial differentiation signal.

It is not a clean class decomposition.

Warm versus cool and warm versus neutral had Jaccard `0.500`.

Cool versus neutral had Jaccard `0.333`.

The recorded verdict was not fully differentiated under those settings.

Cookbook rule:
use IBP-Gumbel active sets as an exploratory diagnostic for mechanism count,
not as proof that class semantics have separated.

## 5. Generalization Claim

The recipe is not color-specific.

The strongest evidence is `auto_exp_54`.

That experiment reused the HSV-style gauge-fix recipe on non-HSV
name-semantic targets:
modifier count,
monoword,
and template sigma.

The same cogito layer 40 centroids and `K=16` PC basis were used.

The result was better than HSV hue on two name targets.

For modifier count, `d=1` CV `R^2=0.763`.

For monoword, `d=1` CV `R^2=0.733`.

For template sigma, `d=1` CV `R^2=0.620`.

The joint `d=3` name-semantic block had mean CV `R^2=0.706`.

This supports a transferable claim:
if a concept can be expressed as a stable auxiliary target over rows, the
gauge-fix recipe can recover it even when the concept is not perceptual color.

`auto_exp_93` adds the separation claim.

The recovered name-semantic modifier-count axis correlated strongly with name
features:
`rho=-0.820` with modifier count,
`rho=+0.762` with monoword,
`rho=+0.654` with template sigma,
and `rho=-0.820` with token count.

The same axis was nearly orthogonal to perceptual hue:
`rho=+0.029` versus the HSV-gauge hue axis,
and `rho=-0.108` versus raw hue.

Axis-wise correlations with perceptual features were small for the main
modifier-count axis, with `|rho|<=0.13` against hue, saturation, value, and
RGB channels.

That means name-semantics is not merely a proxy for color.

It is a separate recoverable subspace in the same layer.

The caveat is large.

This is one model family, one layer, one dataset, and one concept family.

The validated artifact is a recipe, not a law.

Layer-specificity remains unresolved because `auto_exp_43` could not run:
the probed server only had layer 40 hooked, and layer 20 was unavailable.

Cross-model transfer is also unresolved.

The right next tests require new harvests.

## 6. How To Apply This To Your Concept

This section turns the findings into an operational checklist.

### Define The Concept Target

Start by writing down what would count as a supervised signal.

A good signal is available for each row or anchor.

It should be concept-specific.

It should be cheap enough to compute for all examples.

It should not be a post-hoc label invented after inspecting the latent plot.

For perceptual color, the signals were HSV.

For color-name semantics, the signals were modifier count, monoword, and
template sigma.

For sentiment, possible signals could be human sentiment labels, lexicon
scores, or contrastive prompt labels.

For syntax, possible signals could be parse depth, tense, part-of-speech
patterns, or clause count.

For time, possible signals could be hour of day, day of week, season, or phase.

For entity attributes, possible signals could be geography, profession,
organization type, or relation role.

Prefer direct signals over proxies.

A hue label is better than asking a Circle prior to infer hue.

A modifier-count label is better than asking sparsity to infer name syntax.

### Choose The Row Unit

Decide whether the row should be a raw prompt, a grouped centroid, or a
conditioned residual.

The color investigation used centroids because raw prompt variation was
nuisance noise for HSV.

`auto_exp_48` quantified the cost of not centroiding:
raw held-out `R^2` was `0.264` for hue, `0.121` for saturation, and `0.184`
for value.

Centroid CV stayed near the original `auto_exp_38` values:
about `0.684` for hue, `0.644` for saturation, and `0.696` for value.

For your concept, do the same check if templates or contexts vary.

If context variation is not the concept, average it out.

If context variation is the concept, keep it and supervise it directly.

### Pick The Rank

Use the target rank as the initial latent dimension.

Scalar concept:
start with `d=1`.

Three-channel concept:
start with `d=3`.

Cyclic scalar represented as sine and cosine:
start with a two-coordinate circle-aware representation or a rank-two
auxiliary target, but still supervise the phase.

Composite concept:
split it into blocks and give each block the rank implied by its target.

Then run a dimension sweep like `auto_exp_53`.

Look for the smallest `d` that saturates CV and BIC.

If higher dimensions do not improve held-out score and the active-axis count
does not grow, keep the matched dimension.

### Compose The Fit Jointly

Use a joint objective that sees the supervised block and any free block
together.

The winning color recipe did not first remove HSV and then search a residual.

It fit the supervised HSV block while allowing a free block to exist in the
same representation.

That is why free name-semantic structure could emerge in `auto_exp_38`.

When adding more primitives, add them as simultaneous constraints when
possible.

Use block orthogonality to prevent obvious leakage between supervised and free
blocks.

Use ARD to assess which free axes are active after the gauge exists.

Use topology only after you know the semantic plane is being selected.

Avoid a pipeline where every stage destroys information needed by the next.

`auto_exp_51` is the warning example.

### Validate Against Nulls

Every impressive geometry claim needs a null.

For topology, use a null that runs through the identical scorer.

The Sphere result in `auto_exp_41` looked overwhelming until `auto_exp_42`
showed that shuffled or Gaussian nulls reproduced it.

For supervised recovery, use shuffled labels.

Earlier validation in the same thread found that label permutation drove all
six auxiliary `R^2` scores into the null range, approximately `-0.035` to
`-0.018`.

For steering, use control prompts and both direction signs.

A direction that changes generation is not necessarily a direction that changes
generation in the intended semantic way.

### Choose Steering Directions

Use the supervised semantic axis if the target is a scalar or small vector.

For a named anchor, use anchor minus center.

For a contrast, use anchor A minus anchor B, or the fitted semantic axis for
the contrast label.

For a tangent, first project it into the supervised semantic basis.

Then name the projected direction by its measured semantic effect, not by its
base point.

Run a small strength sweep before a large intervention.

The cogito tangent probes had useful motion at `alpha=2` and collapse at
`alpha=5`.

The same pattern can occur in other concept manifolds.

### Rank Steering Risk

Compute curvature or a related local stability score.

Treat high-curvature anchors as risky.

Treat interior, low-curvature anchors as safer starting points.

Test steering at low, medium, and high curvature.

If a high-curvature anchor fails at lower alpha, reduce strength or use a more
global semantic axis.

### What To Skip

Skip unsupervised Circle as the first move for a buried cyclic concept.

`auto_exp_47` showed that it finds the reconstruction-favored plane, not the
hue plane.

Skip ARD-over-PCs as a concept discovery mechanism when the spectrum is fat.

`auto_exp_49` and `auto_exp_50` showed that it can amplify the wrong PCs.

Skip topology claims without null controls.

`auto_exp_42` debunked the Sphere win.

Skip tangent steering when you need semantic identity.

`auto_exp_44` showed that `t1_at_blue` can push warm language.

Skip sequential orthogonalize-then-search when the second-stage structure is
weak.

`auto_exp_51` showed that the residual search failed badly.

Skip treating IBP active-set differences as class proof unless the active sets
separate cleanly.

`auto_exp_45` showed partial atom-count differences but overlapping typical
sets.

## 7. Primitives Table

| Primitive | What it does in this pipeline |
| --- | --- |
| `IvaeRidgeMeanGauge` | Applies the auxiliary conditional gauge: it uses row-level concept signals, such as HSV or modifier count, to pin latent coordinates and break rotational ambiguity. |
| `ARDPenalty` | Performs dimension pressure or free-axis diagnostics after a gauge exists; useful for checking active axes, not sufficient as a semantic concept finder by itself. |
| `BlockOrthogonalityPenalty` | Keeps supervised and free latent blocks from trivially leaking into each other while preserving within-block structure for discovery. |
| `IBPAssignmentPenalty` | Encourages sparse per-row atom usage; in this investigation it exposed partial warm/cool/neutral active-set differences without proving clean class separation. |
| `GumbelTemperatureSchedule` | Anneals relaxed discrete assignments for IBP-style atom indicators so per-row active sets can be inspected before hard discreteness is trusted. |
| `LatentCoord` | Represents the per-row latent chart whose axes become the candidate concept manifold coordinates. |
| `Retraction` (`Circle`/`Sphere`) | Imposes manifold topology on latent coordinates; useful only after null controls and, for sub-dominant concepts, after semantic gauge selection. |
| `TopologyAutoSelector` | Compares candidate topologies such as Euclidean, Circle, Sphere, or Cylinder; must be paired with null controls because `auto_exp_41/42` exposed a lift artifact. |
| `PcaBasis` (lazy) | Supplies the activation-space basis for centroids or rows; useful for denoising and efficient fitting, but PC variance order is not concept order. |
| `MatérnBasis` (streaming) | Provides scalable smooth basis machinery for manifold forward maps or local geometry estimation when row count or ambient dimension makes dense bases expensive. |
| `MechanismSparsityPenalty` | Encourages small mechanism sets in SAE-style or composed models; should be treated as a structural prior, not as a replacement for supervised concept labels. |

## 8. Minimal Cookbook

This is the shortest version of the recipe that still preserves the findings.

1. Gather rows or centroids for the model layer.

2. Define auxiliary concept signals before looking at the final latent chart.

3. Choose latent dimension equal to the target rank.

4. Fit a joint supervised gauge using those auxiliary signals.

5. Add free blocks only when the supervised gauge is present.

6. Use ARD to inspect the free block after the gauge, not before.

7. If the concept is cyclic, supervise the cycle before trusting Circle.

8. Compare topology with null controls that use the identical scorer.

9. Validate prediction with held-out groups, not only in-sample scores.

10. Validate steering with a strength sweep and semantic scoring.

11. Use anchor offsets or supervised axes for steering.

12. Use curvature ranking to choose safer and riskier anchors for intervention.

13. Report failed prescriptions alongside successes.

## 9. Worked Transfer Example

Suppose the target concept is time-of-day in a language model.

The naive plan is to fit a Circle to raw activation PCs and call the recovered
angle "time".

The cogito investigation says this is unsafe.

The better plan is to build supervised auxiliaries:
hour encoded as sine and cosine,
maybe a day/night binary,
and maybe a lexical template-nuisance score if prompt forms vary.

The concept rank is at least two for the cyclic hour encoding.

Start with `d=2` for the phase block.

If adding brightness or activity level, make it a separate supervised block.

Fit the phase block jointly with any nuisance or semantic blocks.

Do not first project out the strongest time proxy and then ask Circle to find
the residual phase.

After fitting, test whether the recovered phase predicts held-out hours.

Then test Circle topology against nulls.

For steering, use a morning anchor offset, an evening anchor offset, or the
supervised phase axis.

Do not use a raw tangent at "midnight" and label it midnight.

Compute curvature over anchors.

Expect edge cases such as dawn, dusk, or ambiguous phrases to have higher
steering risk.

## 10. Interpretation Boundaries

The investigation supports a recipe, not universal geometry.

The successful claims are empirical and tied to cogito layer 40.

The HSV recovery numbers are strong:
`auto_exp_38` got `R^2=(0.700,0.657,0.719)`.

The dimension-selection result is strong:
`auto_exp_53` selected `d=3` by both BIC and CV.

The non-HSV generalization is strong within the same setting:
`auto_exp_54` got `R^2=0.763`, `0.733`, and `0.620` on name-semantic targets.

The orthogonality result is strong for the main modifier-count axis:
`auto_exp_93` found `rho=+0.029` against the HSV hue axis.

The steering evidence is causal but incomplete:
`auto_exp_44` tested live interventions, but only red and blue tangent probes
completed before the time budget.

The anchor-offset steering hypothesis remains the correct next experiment, not
a completed positive result.

The layer-specificity question remains open.

The cross-model transfer question remains open.

The topology story is especially constrained:
Sphere was debunked by null controls,
and Circle failed without supervision even though a hue plane existed.

## 11. Open Questions

Layer-specificity is unresolved.

`auto_exp_43` could not compare layer 20 and layer 40 because the server only
had layer 40 hooked.

The next harvest should include multiple hooked layers in the same model.

The direct question is whether the perceptual and name-semantic split appears
earlier, later, or only at layer 40.

Cross-model transfer is unresolved.

The recipe should be repeated on another model family with the same color
prompts, same centroiding policy, and same held-out scoring.

The direct question is whether the `d=3` HSV gauge and name-semantic
generalization survive architecture and training differences.

Anchor-offset steering needs the blocked experiment.

`auto_exp_46` was prepared but blocked by server availability.

It should compare anchor red and anchor blue against tangent red and tangent
blue across a small alpha sweep.

The decisive criterion is whether anchor red warms and anchor blue cools in the
expected directions.

Curvature needs a causal validation.

`auto_exp_52` produced a ranking.

The next study should test whether high-curvature anchors fail or collapse at
lower alphas than low-curvature anchors.

Topology selection needs better calibrated evidence metrics.

`auto_exp_41/42` showed that a raw BIC win can be a scorer artifact.

Future topology selection should include null-calibrated margins by default.

Sub-dominant cyclic discovery needs a supervised-plus-topology composition.

The oracle `PC2+PC4` result in `auto_exp_47` proves the hue plane exists.

The failure of `auto_exp_49/50/51` proves the tested unsupervised and
sequential searches do not find it.

The next positive test should couple the supervised hue gauge and Circle
retraction in a single joint objective.

## 12. Final Recipe Claim

The most transferable claim is this:

Concept manifolds in LLM activations should be treated as supervised,
gauge-fixed charts first and geometric objects second.

The concept label or auxiliary signal selects the chart.

Topology describes the chart only after the semantic identity has been fixed.

Sparsity and ARD help inspect or regularize the chart only after the gauge is
identifiable.

Steering should use semantic axes or anchor offsets from that chart.

Local tangents, topology wins, and free-block axes are all hypotheses until
they survive nulls, held-out prediction, and causal steering tests.

