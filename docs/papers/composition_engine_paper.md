# A Penalized-Latent-Variable Engine for LLM Concept Manifold Inference: gamfit, Cogito Color Representations, and Steering

## Abstract

Interpretability work on large language models often alternates between two incompatible views of representations: discrete feature dictionaries and continuous geometry. This paper argues that both are special cases of a composition engine for penalized latent-variable inference. In `gamfit`, model interpretation is expressed as a joint penalized likelihood over decoder coefficients β, design-moving latent coordinates ψ or per-row `t`, and evidence-selected penalty hyperparameters ρ. The primitive vocabulary, including `LatentCoord`, `IvaeRidgeMeanGauge`, `ARDPenalty`, `BlockOrthogonalityPenalty`, `IBPAssignmentPenalty`, `GumbelTemperatureSchedule`, and Riemannian retractions, is the language in which inductive biases are stated and compared. We study cogito-L40 hidden states for 949 xkcd color names across 28 prompt templates. Three empirical findings stand out. First, supervised gauge fixing recovers an HSV color chart with `R²(hue,sat,val)=(0.700,0.657,0.719)`, and `auto_exp_53` selects `d=3` by both BIC and cross-validation. Second, the same recipe generalizes to name semantics, with `auto_exp_54` reaching `R²=0.763` for modifier count and `0.733` for monoword. Third, `auto_exp_44` shows causal steering at layer 40, moving the warm-word ratio for `t1_at_red` from `0.627` to `0.792` at `α=2`. The negative result is equally central: unsupervised topology selection can be wrong, as the apparent Sphere win in `auto_exp_41` was refuted by null controls in `auto_exp_42`. We close with a steering pipeline based on supervised semantic axes, anchor offsets, and curvature-ranked intervention strength.

## 1. Introduction

LLM interpretability needs a statistical object that is richer than a probe and more constrained than an arbitrary embedding. A probe asks whether information is linearly or nonlinearly decodable from activations, but it often leaves the coordinate system non-identifiable. An embedding displays geometry, but it can mistake variance structure for concept structure. Sparse autoencoders provide a powerful dictionary-learning view, but a dictionary atom is not the only kind of interpretable unit: hue is cyclic, saturation and value are continuous, and name morphology can occupy a different low-dimensional subspace from perceptual color. The central claim of this paper is that concept-manifold inference should be formulated as penalized latent-variable inference with explicit, composable inductive biases.

The reason is identifiability. A latent chart is not meaningful just because it reconstructs activations. For a smooth decoder `f(t)`, many smooth reparameterizations of `t` can be absorbed into the decoder coefficients without changing the likelihood. This is the same gauge problem that appears in Gaussian process latent-variable models, nonlinear ICA, and neural manifold learning. The practical consequence is that a recovered "hue axis" or "semantic direction" can rotate, scale, split, or disappear unless the model includes a gauge-breaking assumption. In LLM settings, that assumption is often available: prompt metadata, synthetic labels, known concept coordinates, intervention labels, or contrastive anchors can provide auxiliary variables. The inference problem is then not "find a pretty manifold"; it is "compose the weakest adequate set of priors that makes the concept chart identifiable, predictive, and causally testable."

`gamfit` is useful in this setting because it exposes the primitive vocabulary of those priors. A standard generalized additive model is only one member of a broader methodspace. The same engine can treat smooth coefficients, per-row latent coordinates, sparse mechanism assignments, topology constraints, and smoothness strengths as pieces of one penalized likelihood. The analyst chooses a likelihood, a decoder basis, a latent coordinate block, and analytic penalties. REML or Laplace evidence then selects the strength of penalization where the implementation supports it. This is not merely an API convenience. It makes interpretability claims auditable. The paper can say which assumption identified the chart: `IvaeRidgeMeanGauge` supplied the auxiliary conditional gauge, `BlockOrthogonalityPenalty` separated supervised and free blocks, `ARDPenalty` applied axis pressure after the gauge was fixed, and a Circle retraction was tested only where cyclic topology was actually being evaluated.

The empirical testbed is a color-manifold investigation in cogito layer 40. The dataset consists of hidden-state responses for 949 xkcd color names under 28 prompt templates, harvested through a custom probe server. Color is a good test because it contains multiple structures that are easy to confuse. HSV gives a three-dimensional perceptual target with a cyclic hue component. Color names also contain linguistic structure: some are monowords, some are compounds, some contain modifiers, and some are template-sensitive. A method that only finds the largest variance directions can mix these factors; a method that only asserts topology can find a ring that is not hue.

The results support a narrow but useful thesis. When a low-dimensional concept lives in the top principal-component subspace and a meaningful auxiliary target is available, a joint penalized latent fit can recover an identifiable concept chart and use it for steering. `auto_exp_38` recovered HSV with `R²(hue)=0.700`, `R²(sat)=0.657`, and `R²(val)=0.719`. `auto_exp_53` showed that `d=3` is the correct rank for HSV, with held-out hue `R²=0.687`, mean held-out `R²=0.680`, and best BIC at `-2383.0`. `auto_exp_54` showed that the recipe is not color-specific, recovering modifier count with `R²=0.763`, monoword with `R²=0.733`, and template sigma with `R²=0.620`. `auto_exp_93` separated the main name-semantic axis from perceptual hue, with `ρ=+0.029` against the HSV-gauge hue axis and `ρ=-0.108` against raw hue. Finally, `auto_exp_44` showed that chart-derived interventions can move generation, but also showed that local tangents should not be naively named as global concepts.

The failures are part of the contribution. `auto_exp_47` found that unsupervised Circle on raw PCs failed to recover hue, even though an oracle `PC2+PC4` plane had circular correlation `-0.720`. `auto_exp_41` appeared to show that a Sphere topology won, but `auto_exp_42` showed that shuffled and Gaussian null controls reproduced the advantage. `auto_exp_49`, `auto_exp_50`, and `auto_exp_51` showed that sequentially gauge-fixing HSV and then searching residuals with ARD plus Circle is lossy. `auto_exp_55` showed that `TopologyAutoSelector` can misclassify Circle or Sphere structure as Torus when cosine-plus-sine parameterizations overfit. The message is not that priors solve interpretability automatically. The message is that a composition engine makes successes and failures precise enough to be cumulative.

## 2. Related Work

Gaussian-process factor analysis (GPFA) is the closest neural-data ancestor of the present framing. Yu, Cunningham, Santhanam, Ryu, Shenoy, and Sahani introduced GPFA to extract low-dimensional single-trial neural trajectories by unifying smoothing and dimensionality reduction in a probabilistic model [Yu et al., 2009]. GPFA matters here because it treats latent trajectories as inferred objects with temporal structure, not as post-hoc projections. The cogito setting differs in that rows are color-name centroids rather than time bins, and the key structure is concept geometry rather than temporal smoothness. The shared principle is that a population representation should be inferred through a generative or penalized model whose assumptions are explicit.

The nonlinear identifiability literature explains why auxiliary variables are not optional decoration. Khemakhem, Kingma, Monti, and Hyvarinen's iVAE framework connects variational autoencoders to nonlinear ICA and shows how auxiliary variables can make latent factors identifiable under conditional exponential-family assumptions [Khemakhem et al., 2020]. This paper uses that lesson in a pragmatic, penalized-likelihood form. `IvaeRidgeMeanGauge` is not a full iVAE; it is a row-local auxiliary mean gauge that makes a latent chart estimable inside a GAM-style engine. The philosophical debt is direct: without an auxiliary signal or another gauge-breaking prior, a nonlinear latent coordinate system is generally not unique.

Gaussian process latent-variable models provide the broader manifold-learning background. Lawrence introduced the GPLVM as a probabilistic nonlinear analogue of PCA [Lawrence, 2004]. Manifold GPLVMs extend that idea to non-Euclidean latent spaces. Jensen, Kao, Tripodi, and Hennequin's mGPLVM explicitly places latent states on manifolds such as rings, tori, spheres, and rotation groups, and uses cross-validated comparisons to distinguish candidate topologies in neural data [Jensen et al., 2020]. The cogito experiments borrow the topology question but change the burden of proof. In `auto_exp_41/42`, a topology scorer selected Sphere for reasons that null controls reproduced. That failure is a warning that topology evidence must be routed through identical null pipelines before it becomes an interpretability claim.

Sparse autoencoders form the dominant recent language-model feature-discovery baseline. Bricken et al. decomposed transformer activations with dictionary learning and argued that sparse features can be more monosemantic units of analysis than individual neurons [Bricken et al., 2023]. Cunningham, Ewart, Riggs, Huben, and Sharkey found highly interpretable features in language models using sparse autoencoders [Cunningham et al., 2023]. DeepMind work on gated sparse autoencoders and Gemma Scope scaled and refined the SAE toolkit, including shrinkage-aware variants and released suites of SAEs over Gemma 2 layers [Rajamanoharan et al., 2024; Lieberum et al., 2024]. Our framing is complementary rather than adversarial. SAE atoms are one kind of primitive in the composition engine; cyclic charts, auxiliary-gauged continuous coordinates, block-orthogonal residuals, and sparse mechanism assignments are others.

The computational substrate resembles bundle adjustment. Ceres Solver and g2o organize large nonlinear least-squares problems with sparse block structure, where global parameters and per-observation variables interact through Schur complements [Agarwal and Mierle, 2012; Kummerle et al., 2011]. `gamfit` uses the same structural idea for penalized latent inference: β is global, each row has local latent coordinates, and the Hessian has an arrow or Schur form. Finally, mgcv's REML treatment of generalized additive models is the evidence-selection template [Wood, 2017]. Smoothness and penalty strengths are not arbitrary knobs; they are hyperparameters selected by marginal likelihood approximations. The composition engine generalizes that discipline from smooth terms to concept-manifold primitives.

## 3. Methods

The composition engine represents an interpretability model as a three-tier penalized likelihood. The first tier, β, contains decoder or smooth coefficients. These are the parameters solved by the penalized Newton or PIRLS inner loop. The second tier contains design-moving state: kernel-shape coordinates ψ for terms such as anisotropic spatial smooths, and per-row latent coordinates `t_i` for concept charts. The third tier, ρ, contains penalty strengths, smoothness parameters, sparsity weights, and other structural hyperparameters selected by REML or Laplace approximate marginal likelihood. In a standard GAM, only β and smoothness ρ are visible. In concept-manifold inference, `t` becomes first class.

The objective can be read as a sentence with slots. The likelihood states what the rows are asked to reconstruct or predict. The decoder basis states what functions of the latent chart are allowed. The analytic penalties state which symmetries are broken, which axes are sparse, which blocks are separated, and which topologies are legal. The evidence approximation scores the whole sentence rather than one piece in isolation. This is the reason the paper treats primitive calls as part of the scientific method: changing `ARDPenalty` from a post-gauge diagnostic into the only source of structure is not a small hyperparameter change. It is a different identifiability claim.

```text
                 evidence / REML / LAML
                         selects ρ
                            |
                            v
        penalties Sρ(β,t,ψ) + likelihood ℓ(y | β,t,ψ)
                            |
             +--------------+--------------+
             |                             |
        β tier: decoder              ψ/t tier: design
        coefficients                 moving coordinates
```

This separation matters because each tier has a different statistical role. β explains observations given a design. The latent tier moves the design itself, which means its derivatives must propagate through the fitted β optimum. ρ decides how much each inductive bias is trusted. The implicit-function-theorem correction through the inner optimum is what lets a change in latent coordinates or penalty strength be evaluated as a change in the fitted penalized model rather than as a frozen-design perturbation. In the color experiments, the observed object is an activation-derived response matrix or PC representation, the row-level latent `t_i` is the proposed concept chart for color `i`, and the auxiliary variables are HSV or name-semantic labels.

Computationally, per-row latent variables create an arrow-structured system. β is shared by all rows, while each `t_i` affects only row `i`'s design contribution and row-local analytic penalties. If the Hessian is ordered as global β followed by row blocks, the matrix has a dense global block, many small row blocks, and off-diagonal couplings only through β. This is the same algebraic pattern exploited by bundle adjustment, where cameras and points are eliminated by a Schur complement. For concept manifolds, it permits row-local latent updates, row-local Hessian-vector products, and scalable evidence calculations without materializing a dense `(N*d)²` latent Hessian.

```text
H =
  [ Hββ   Hβt1  Hβt2  ... HβtN ]
  [ Ht1β  Ht1   0         0    ]
  [ Ht2β  0     Ht2       0    ]
  [ ...                    ... ]
  [ HtNβ  0     0      ... HtN ]

Schur view: eliminate row blocks Hti, solve the global β system,
then recover each latent update independently.
```

The primitive vocabulary is the main modeling interface. `LatentCoord(n,d,...)` creates a per-row coordinate chart with dimension `d`, initialization, optional topology, and optional retraction. A Euclidean latent coordinate is appropriate when the concept is locally linear. A Circle retraction is appropriate only when a cyclic coordinate has already been selected or supervised. Sphere, Torus, Cylinder, and product retractions are available for hypotheses that genuinely live on those manifolds. The Riemannian retraction updates embedded points by projecting a Euclidean tangent step back onto the constraint, so optimization respects the manifold rather than repairing it after the fact.

Analytic penalties express the assumptions that make the chart identifiable. `IvaeRidgeMeanGauge` is the central primitive in the cogito results. It uses auxiliary variables to define a row-local ridge mean gauge, shrinking latent coordinates toward an aux-conditioned chart and breaking the rotation symmetry that would otherwise make the axes arbitrary. `ARDPenalty` applies axis-wise evidence pressure and can prune or diagnose inactive dimensions, but only after a gauge exists. `BlockOrthogonalityPenalty` penalizes cross-products between configured groups of latent axes, which is useful when a supervised block and a free block are fit jointly. `IBPAssignmentPenalty` defines sparse per-row atom use for mechanism discovery; paired with `GumbelTemperatureSchedule`, it gives a smooth relaxation whose active sets can be inspected as the temperature falls. `MechanismSparsityPenalty` and related sparsity primitives encourage small active mechanism sets, while smoothness penalties regularize decoder functions over the latent chart.

The distinction between row-local and cross-row penalties is operational. Row-local penalties such as ARD-style diagonal shrinkage and IBP assignment costs preserve the arrow-Schur structure. Cross-row structure, such as graph total variation or block orthogonality computed over the entire latent matrix, can still be useful, but it changes the Hessian algebra and has to be routed through analytic Hessian-vector products or stochastic trace estimation. This is why the method section emphasizes architecture rather than just statistics. The valid composition is the one whose derivatives, normalizers, and null comparisons all pass through the same computational path.

| Primitive | Tier | Intended role in concept inference |
| --- | --- | --- |
| `LatentCoord` | ψ/t | Per-row concept coordinates, optionally with manifold retraction. |
| `IvaeRidgeMeanGauge` | analytic penalty on t | Auxiliary conditional gauge for identifiability. |
| `ARDPenalty` | analytic penalty on t or blocks | Axis pressure after gauge fixing, not a concept finder alone. |
| `BlockOrthogonalityPenalty` | analytic penalty on t | Separates supervised and free blocks in a joint fit. |
| `IBPAssignmentPenalty` | analytic penalty on assignments | Sparse mechanism-count prior for SAE-manifold variants. |
| `GumbelTemperatureSchedule` | ρ schedule | Anneals relaxed discrete assignments. |
| Riemannian retraction | latent update rule | Keeps Circle, Sphere, Torus, or product coordinates on-manifold. |

The validated recipe has five steps. First, choose the latent dimension to match the supervised target rank. In HSV, that means `d=3`, not because three is aesthetically convenient but because hue, saturation, and value are the target channels. Second, fit a joint supervised gauge, using an iVAE-style auxiliary prior rather than asking reconstruction loss to invent semantics. Third, if the target is cyclic and sub-dominant in variance, require supervised auxiliary information before trusting Circle or Torus topology. Fourth, use supervised semantic axes or anchor offsets for steering, not raw local tangent columns named after their base point. Fifth, rank curvature or another local stability score before intervention, because high-curvature anchors are expected to leave the learned manifold at smaller steering strengths.

| Step | Operation | Evidence in the cogito study |
| --- | --- | --- |
| 1 | Match `d` to concept rank | `auto_exp_53`: HSV selects `d=3`, BIC `-2383.0`, mean held-out `R²=0.680`. |
| 2 | Fit a joint auxiliary gauge | `auto_exp_38`: HSV `R²=(0.700,0.657,0.719)`. |
| 3 | Supervise buried cycles | `auto_exp_47`: unsupervised Circle `|ρ|=0.041`, oracle `PC2+PC4` circ-corr `-0.720`. |
| 4 | Steer with axes or anchors | `auto_exp_44`: `t1_at_red`, `α=2`, warm ratio `0.627 -> 0.792`. |
| 5 | Rank local stability | `auto_exp_52`: curvature range `14.1` to `132`, median `34.1`. |

This recipe is deliberately conservative. It does not say that every interpretable concept requires supervision, nor that unsupervised methods are useless. It says that claims about named human concepts require a gauge, and that a gauge is an empirical assumption that should be encoded as a primitive and tested. When the primitive fails, the failure is informative. ARD without a gauge is a rotationally ambiguous shrinkage device. Topology selection without null controls is a scorer. Tangent steering without a semantic basis is a local differential operation, not a named concept intervention.

## 4. Experiments: Cogito Color Manifold

The empirical study uses cogito-L40 hidden states for the xkcd color-name set. The harvest contains 949 colors and 28 prompt templates per color. Activations were collected with a custom probe server exposing `/v1/encode`, with `max_model_len=8192`, and layer 40 was the only probed layer available for the main run. The core row unit is the per-color centroid: template responses for the same color name are averaged before fitting. `auto_exp_48` validated this choice by showing that raw per-prompt fitting had much lower held-out `R²`, namely `0.264` for hue, `0.121` for saturation, and `0.184` for value. Centroiding treats prompt variation as nuisance variation for the perceptual color task. The main PC representation uses `K=16`, which is small enough for controlled latent experiments and large enough to contain the relevant color and name-semantic signals.

The supervision was defined before the final latent fits were interpreted. HSV came from the color target and was treated as a three-channel auxiliary variable. Name-semantic supervision came from surface properties of the color names and prompt responses, including modifier count, monoword status, token count, and template sigma. This distinction prevents a common failure mode in representation analysis: discovering a latent axis first and inventing its label afterward. In these experiments, the labels were either external perceptual coordinates or predeclared name features, and the fit was judged by held-out prediction, axis correlations, null controls, and steering behavior.

The first result is identifiability through supervised gauge fixing. In `auto_exp_38`, an HSV-supervised gauge was fit with a companion free block. The supervised coordinates recovered `R²(hue)=0.700`, `R²(sat)=0.657`, and `R²(val)=0.719`. This is the key positive result because it shows that color structure in cogito-L40 is not merely visible in a plot; it is recoverable as a coordinate-bearing chart. The free block also became interpretable after the supervised block fixed the gauge. Its maximum absolute correlations with name features were `0.463`, `0.667`, and `0.249`; one axis tracked modifier count around `0.67` and monoword around `0.63`, while free-axis correlations with HSV stayed small, with maximum around `0.13`. This companion-block behavior is exactly what the composition engine is meant to express: a supervised block can identify the coordinate frame while a free block captures residual structure without being forced to duplicate the supervised target.

`auto_exp_53` validated the dimension choice rather than assuming it. The experiment swept `d in {2,3,4,5,6}` with reduced-rank ridge on cogito layer 40 centroids. At `d=2`, hue and saturation were partly recovered, but value was crushed, with in-sample value `R²=0.174` and mean CV `R²=0.470`. At `d=3`, the full HSV block appeared, with in-sample `R²=(0.700,0.658,0.720)`, held-out hue `R²=0.687`, mean held-out `R²=0.680`, and BIC `-2383.0`. Increasing to `d=4`, `d=5`, or `d=6` did not improve the in-sample or CV scores; the active-axis count remained three. The result is a clean rank-selection story: match the latent dimension to the known target rank, then use evidence and cross-validation to check that extra axes are inactive.

This result also clarifies the role of BIC and CV in the paper. BIC is not used as a decorative score after choosing the model by eye. It penalizes unnecessary latent degrees of freedom, while grouped cross-validation checks that the recovered chart predicts held-out color identities rather than only interpolating within the training centroids. Their agreement at `d=3` is the important point. If BIC had preferred a smaller chart while CV improved with larger `d`, the result would have been a modeling conflict. Instead, both criteria selected the rank implied by the concept.

The second result is generalization beyond HSV. `auto_exp_54` reused the same gauge-fix recipe for non-HSV name-semantic targets on the same cogito-L40 centroids and `K=16` PC basis. For modifier count, a scalar `d=1` target reached CV `R²=0.763`. For monoword, `d=1` reached CV `R²=0.733`. For template sigma, `d=1` reached CV `R²=0.620`. A joint `d=3` name-semantic block had mean CV `R²=0.706`. These values are not weaker curiosities in the residual of color; two exceed the HSV hue CV level. They show that the recipe recovers stable auxiliary concepts when those concepts are linguistic rather than perceptual.

`auto_exp_93` sharpened the generalization claim by testing separation from hue. The recovered modifier-count name axis had strong correlations with name features: `ρ=-0.820` with modifier count, `ρ=+0.762` with monoword, `ρ=+0.654` with template sigma, and `ρ=-0.820` with token count. Against perceptual color, the same axis was nearly orthogonal: `ρ=+0.029` against the HSV-gauge hue axis and `ρ=-0.108` against raw hue. Axis-wise correlations with hue, saturation, value, and RGB channels stayed within `|ρ|<=0.13` for the main modifier-count axis. This matters because a color-name dataset could easily confound name form with color category. The experiment says that, at least in layer 40, name semantics is a separable recoverable subspace rather than a proxy for perceptual hue.

The third result is causal steering. `auto_exp_44` applied live interventions at layer 40 using tangent-probe directions from the fitted color manifold and scored generated text by warm-language ratio, defined as warm words divided by warm plus cool words. For `t1_at_red`, the warm ratio moved from `0.627` at `α=0` to `0.792` at `α=2`, a shift of `+0.165`. For `t1_at_blue`, the warm ratio moved from `0.517` at `α=0` to `0.833` at `α=2`, a larger shift of `+0.317`. This is a real causal result: perturbing the representation at the probed layer changed generated language in a semantically scored direction.

The steering result also contains a negative semantic result. The blue tangent did not make generations cooler; it made them warmer. At `α=5`, both completed tangent families collapsed to a warm ratio around `0.417`, with weaker coherence. The operational conclusion is that local tangent vectors are derivatives of a fitted chart, not automatically global semantic directions. A steering pipeline should therefore use a supervised semantic axis when available, use anchor offsets such as red-anchor minus manifold center for named anchors, and project any tangent vector into the supervised semantic basis before assigning it a name. The composition engine gives the coordinate frame required for that projection, but the tangent itself is not the interpretation.

The fourth result is geometric localization of hue. `auto_exp_47` found that unsupervised Circle on raw principal components failed to recover hue: `|ρ|=0.041`, circular rho `-0.021`, and full circular MSE `0.0684`. A top-1 Euclidean baseline was also weak, with `|ρ|=0.108` and circular rho `+0.134`. Yet an oracle `PC2+PC4` plane had circular correlation `-0.720` with true hue and full circular MSE `0.0294`, less than half the unsupervised values. Follow-up geometry runs `auto_82` and `auto_86` locate the hue ring in the `PC2 x PC4` plane rather than in the top unsupervised cyclic plane. The important distinction is between existence and selection. Hue geometry exists in the representation, but a reconstruction-driven unsupervised objective does not necessarily select it.

The `PC2 x PC4` finding is the strongest argument against a simplistic "no hue manifold" reading of the negative unsupervised results. The ring is present, and the circular correlation of `-0.720` is large enough to matter. The failure is selection under the wrong objective. This is exactly the situation in which a composition engine is preferable to a single unsupervised geometry routine: it lets the analyst add the missing semantic gauge and then re-test geometry under a model that is actually aimed at the named concept.

Three negative results define the boundary of the method. First, the Sphere-wins claim was refuted. `auto_exp_41` ran a free-block topology sweep after the HSV-supervised and PPCA-ARD setup and appeared to show a dramatic Sphere advantage: Sphere BIC `-423.85`, Euclidean `5420.54`, Circle `5516.85`, Cylinder `6338.31`, with a Sphere-over-Euclidean BIC margin of `5844.40`. Cross-validation also favored Sphere in the raw comparison, with mean log likelihood per point `-0.2752` compared with Euclidean `-2.8446`. `auto_exp_42` then ran null controls through the identical scoring path. IID Gaussian covariance nulls reproduced a BIC gap of about `4736.5 +/- 232.5`; IID identity nulls gave `4738.6 +/- 251.1`; a shuffled null nearly matched the real result with BIC gap `5835.9 +/- 4.3`, ratio `0.999` versus real. The apparent Sphere geometry was a stereographic lift and vMF-tail artifact, not a property of cogito color semantics.

Second, sequential "gauge-fix-then-residual ARD" was lossy. `auto_exp_49` asked whether ARD over PCs plus Circle could find the oracle hue plane without labels. It could not: ARD plus Circle reached `|ρ|=0.032` and circular rho `-0.041`, with top alpha PCs `[9,7,4]`, missing the `PC2+PC4` pair. `auto_exp_50` partially fixed scaling symmetry, improving `|ρ|` to `0.170` for a simplex-normalized alpha variant and `0.195` for fixed sigma, but neither reached the `|ρ|>=0.30` target or found the oracle pair. `auto_exp_51` then tested the natural sequential recipe: fit HSV, orthogonalize the representation against it, and run ARD plus Circle on the residual. Bare Circle on the full `K=16` PC space had `|ρ|=0.041`; bare Circle on the HSV residual fell to `|ρ|=0.005`; raw ARD plus Circle on the residual reached only `|ρ|=0.013`; normalized-alpha ARD plus Circle reached `|ρ|=0.008`. Joint block-coordinate fits are required when concept blocks interact in the raw representation.

Third, unsupervised topology auto-selection overfit in `auto_exp_55`. The `TopologyAutoSelector` misclassified Circle and Sphere cases as Torus because cosine-plus-sine parameterizations overfit the reconstruction objective. This failure is consistent with the Sphere null-control result. Topology selection is useful only when the candidate scorer is calibrated against nulls and when the semantic plane has already been selected. In a fat residual spectrum, recon-only topology search can explain variance in a geometrically elaborate way without identifying the named concept.

## 5. Reproducibility

The following configuration summarizes the runs as a reproducible recipe rather than as a hidden notebook convention. The data source is cogito-L40, harvested with the custom probe server endpoint `/v1/encode`, `max_model_len=8192`, 949 xkcd color names, and 28 templates per color. The row unit for the reported positive results is the per-color centroid. The activation representation is the standardized `K=16` PC basis unless a specific experiment, such as oracle `PC2+PC4`, states otherwise.

```python
# Shared row object
Y = cogito_l40_centroid_pcs(colors=949, templates=28, k_pc=16)
hsv = aux_hsv(colors)                         # hue, saturation, value
name = aux_name_semantics(colors)             # modifier_count, monoword, template_sigma

# auto_exp_38 and auto_exp_53: HSV gauge and dimension sweep
t_hsv = gamfit.LatentCoord(n=949, d=3, init="pca", name="t_hsv")
fit_hsv = gamfit.gaussian_reml_fit_latent(
    t_hsv,
    Y,
    penalties=[
        gamfit.IvaeRidgeMeanGauge(target="t_hsv", aux=hsv, weight="reml"),
        gamfit.ARDPenalty(target="t_hsv", latent_dim=3),
    ],
)

# Companion-block version used for HSV plus free structure in auto_exp_38
t_joint = gamfit.LatentCoord(n=949, d=6, init="pca", name="t_joint")
fit_joint = gamfit.gaussian_reml_fit_latent(
    t_joint,
    Y,
    penalties=[
        gamfit.IvaeRidgeMeanGauge(target="t_joint[0:3]", aux=hsv, weight="reml"),
        gamfit.BlockOrthogonalityPenalty(target="t_joint", groups=[[0, 1, 2], [3, 4, 5]], weight="reml"),
        gamfit.ARDPenalty(target="t_joint[3:6]", latent_dim=3),
    ],
)

# auto_exp_54 and auto_exp_93: non-HSV name-semantic gauges
t_name = gamfit.LatentCoord(n=949, d=3, init="pca", name="t_name")
fit_name = gamfit.gaussian_reml_fit_latent(
    t_name,
    Y,
    penalties=[
        gamfit.IvaeRidgeMeanGauge(target="t_name", aux=name, weight="reml"),
        gamfit.ARDPenalty(target="t_name", latent_dim=3),
    ],
)

# auto_exp_47, 49, 50, 51, 55: topology tests and negative controls
t_circle = gamfit.LatentCoord(n=949, d=2, init="pca", name="t_circle", retraction="circle")
circle_fit = gamfit.gaussian_reml_fit_latent(
    t_circle,
    Y,
    penalties=[gamfit.ARDPenalty(target="t_circle", latent_dim=2)],
)
topology_rank = gamfit.TopologyAutoSelector.for_latent("t_circle").select(Y)

# auto_exp_44: steering
direction = tangent_probe(fit_joint, anchor="red", tangent="t1")
generate_with_layer40_intervention(direction=direction, alpha=2, score="warm_ratio")
```

The code above is intentionally phrased in primitive calls because the paper's main reproducibility claim is compositional. `auto_exp_38` is not just "a fit"; it is `LatentCoord` plus an auxiliary `IvaeRidgeMeanGauge`, optionally with a block-orthogonal free component. `auto_exp_53` is the same fit under a dimension sweep with BIC and grouped CV. `auto_exp_54` swaps HSV auxiliaries for name-semantic auxiliaries. `auto_exp_47`, `49`, `50`, `51`, and `55` are the negative topology and ARD controls. `auto_exp_44` uses the fitted chart to define a layer-40 intervention and then scores text with the same warm-language metric across α.

## 6. Discussion

The composition engine wins when the statistical and computational assumptions line up. The concept should have a low-dimensional representation inside the chosen activation subspace, the row unit should suppress nuisance variation rather than average away the target, and some auxiliary signal should be available to fix the gauge. The cogito color results satisfy these conditions. HSV is rank three, the per-color centroid is a cleaner row unit than the raw prompt, and HSV labels provide a direct auxiliary gauge. Name semantics also satisfy the recipe: modifier count, monoword, and template sigma are cheap row-level labels, and `auto_exp_54` shows that they recover with high cross-validated `R²`.

The engine also wins when negative results matter. A less structured workflow could have reported the Sphere win from `auto_exp_41` as a discovery. A composition-engine workflow forces the scorer, the topology primitive, and the null control to be named. Once `auto_exp_42` ran shuffled and Gaussian nulls through the same path, the Sphere claim collapsed. Similarly, ARD is demoted from a magic concept finder to an axis-pressure primitive whose interpretation depends on the surrounding gauge. This is a healthier scientific loop: encode the assumption, run the fit, test a null, and report the failed prescription.

The method fails when reconstruction is asked to stand in for semantics. `auto_exp_47` is the cleanest example. Hue exists in `PC2 x PC4`, but unsupervised Circle selects the wrong structure because the dominant reconstruction objective is not the same as the named cyclic concept. A fat residual spectrum worsens this failure because many planes can absorb variance. `auto_exp_49` and `auto_exp_50` show that adding ARD over PCs does not repair the problem; it can follow scale mechanics and high-index variance instead. `auto_exp_51` shows that a sequential residual pipeline can be actively destructive. Orthogonalizing out a supervised block before searching for a weak cyclic residual may remove information needed to identify the cycle.

The steering results should be interpreted with the same restraint. `auto_exp_44` proves causal influence at moderate strength, not a complete semantic control system. The red tangent increased warm-language ratio, but the blue tangent also increased warm-language ratio. The lesson is that a tangent at an anchor is a local derivative in the fitted chart, and local derivatives do not automatically inherit the anchor's name. For practical steering, the pipeline should prioritize supervised semantic axes, anchor offsets, and projected tangent components. It should also include a strength sweep, because `α=5` produced collapse where `α=2` produced useful motion.

Future work has three natural directions. The first is SAE-manifold training. Sparse autoencoders already provide scalable feature dictionaries, but many concepts are better represented as low-dimensional manifolds attached to sparse mechanisms. A joint SAE-manifold model would let assignments choose mechanisms while per-mechanism `LatentCoord` blocks model cyclic or continuous variation. `IBPAssignmentPenalty`, `GumbelTemperatureSchedule`, `MechanismSparsityPenalty`, and block ARD are already the right primitive vocabulary; the next step is to train these models at the scale and rigor of modern SAE pipelines.

The second direction is transfer. The current evidence is single-model and single-layer. `auto_exp_43` could not compare layer 20 and layer 40 because only layer 40 was hooked in the server. Cross-model transfer is also unresolved. A stronger claim would harvest the same 949-color, 28-template dataset across layers and model families, fit the same auxiliary-gauged charts, and compare whether hue, saturation, value, and name semantics occupy homologous subspaces. The `auto_exp_93` orthogonality result gives a concrete transfer target: does the modifier-count axis remain nearly orthogonal to hue across models?

The third direction is scale. The arrow-Schur structure suggests a path to `K=1M` row-level fits with GPU streaming, but the implementation has to treat row-local latent blocks, global β updates, and REML trace terms as first-class sparse operations. The bundle-adjustment analogy is useful here. Large-scale visual reconstruction became practical because sparse Schur algebra, analytic derivatives, and robust solver engineering were treated as the core problem rather than as implementation details. Concept-manifold inference for LLMs will need the same seriousness: streamed basis construction, row-block Hessian-vector products, stochastic log-determinants, and explicit retraction-aware updates.

The broader conclusion is modest. The paper does not show that every LLM concept is a clean manifold, or that supervised gauges are always available, or that topology selection can be automated without judgment. It shows that `gamfit` can express a family of interpretable latent-variable models in one penalized-likelihood language, and that this language was strong enough to recover HSV, recover separable name semantics, falsify tempting topology stories, and produce a causal steering signal in cogito-L40. That is enough to make the composition engine a useful research instrument.

A useful next paper would be less about adding primitives and more about closing the empirical loop. It should pre-register the concept labels, harvest multiple layers and models, run the same null controls for every topology claim, and reserve steering prompts until after the chart has been selected. That is the standard required for turning a successful recipe into a reusable interpretability protocol.

## References

Agarwal, S., Mierle, K., and The Ceres Solver Team. 2012. Ceres Solver. http://ceres-solver.org/

Bricken, T., Templeton, A., Batson, J., Chen, B., Jermyn, A., Conerly, T., Turner, N., Anil, C., Denison, C., Askell, A., Lasenby, R., Wu, Y., Kravec, S., Schiefer, N., Maxwell, T., Joseph, N., Hatfield-Dodds, Z., Tamkin, A., Nguyen, K., McLean, B., Burke, J. E., Hume, T., Carter, S., Henighan, T., and Olah, C. 2023. Towards Monosemanticity: Decomposing Language Models With Dictionary Learning. Transformer Circuits Thread. https://transformer-circuits.pub/2023/monosemantic-features/

Cunningham, H., Ewart, A., Riggs, L., Huben, R., and Sharkey, L. 2023. Sparse Autoencoders Find Highly Interpretable Features in Language Models. arXiv:2309.08600. https://arxiv.org/abs/2309.08600

Jensen, K. T., Kao, T.-C., Tripodi, M., and Hennequin, G. 2020. Manifold GPLVMs for discovering non-Euclidean latent structure in neural data. NeurIPS 33. https://proceedings.neurips.cc/paper/2020/hash/fedc604da8b0f9af74b6cfc0fab2163c-Abstract.html

Khemakhem, I., Kingma, D. P., Monti, R. P., and Hyvarinen, A. 2020. Variational Autoencoders and Nonlinear ICA: A Unifying Framework. AISTATS, PMLR 108:2207-2217. https://proceedings.mlr.press/v108/khemakhem20a.html

Kummerle, R., Grisetti, G., Strasdat, H., Konolige, K., and Burgard, W. 2011. g2o: A general framework for graph optimization. IEEE International Conference on Robotics and Automation, 3607-3613. https://doi.org/10.1109/ICRA.2011.5979949

Lawrence, N. D. 2004. Gaussian Process Models for Visualisation of High Dimensional Data. NeurIPS 16. https://inverseprobability.com/publications/lawrence-gplvm03.html

Lieberum, T., Rajamanoharan, S., Conmy, A., Smith, L., Sonnerat, N., Varma, V., Kramar, J., Dragan, A., Shah, R., and Nanda, N. 2024. Gemma Scope: Open Sparse Autoencoders Everywhere All At Once on Gemma 2. arXiv:2408.05147. https://arxiv.org/abs/2408.05147

Rajamanoharan, S., Conmy, A., Smith, L., Lieberum, T., Varma, V., Kramar, J., Shah, R., and Nanda, N. 2024. Improving Dictionary Learning with Gated Sparse Autoencoders. arXiv. https://deepmind.google/research/publications/88147/

Wood, S. N. 2017. Generalized Additive Models: An Introduction with R. Second edition. CRC Press. https://research-information.bris.ac.uk/en/publications/generalized-additive-models-an-introduction-with-r/

Yu, B. M., Cunningham, J. P., Santhanam, G., Ryu, S. I., Shenoy, K. V., and Sahani, M. 2009. Gaussian-process factor analysis for low-dimensional single-trial analysis of neural population activity. Journal of Neurophysiology 102:614-635. https://pubmed.ncbi.nlm.nih.gov/19357332/
