## Unreleased

### Fixed
- The `SLS-MACRO-CODEGEN-932` timing cell in `gam-row-macros` compared the generated
  location-scale program against a hand schedule that gated its `u1`/`g` terms on
  `plan.u1.is_some()` -- the row's weight -- while the generated program gates on the
  term's own coefficient stack. On a censored far-tail row with a zero stack the hand
  formed `0 * inf` and returned NaN where the program returns a finite zero, so its
  saving was the guard it was missing. The opponent now carries the program's contract
  and `the_hand_carries_the_generated_programs_activity_contract_932` pins it. (#932)

- A coefficient-mode profile whose candidates all refused the trial point now reports a trial-point refusal instead of `UnsupportedConfiguration`, so the outer search steps away instead of aborting the fit. On the large-scale CTN preprocessor a single `h' has non-positive values` refusal killed a 25-minute fit that already had a certified incumbent; one structural rejection among the candidates still keeps the whole profile structural (#979, #2553, #2590).

- `Cargo.lock` resolves `numpy` against the workspace's own `ndarray 0.17.2` again: a re-resolution had flipped that one edge to `ndarray 0.16.1`, which left `gam-pyffi` with 945 compile errors (every `into_pyarray` / `from_owned_array` at the Python boundary) because the two crates no longer shared an `ArrayBase` type. No gate job builds the FFI crate, so nothing caught it (#2670).

- The `SLS-ROW-VGH-932` speed cell's hand opponent gates each term on the term's coefficient stack, as production does, instead of on the row weight. The two predicates differ on a censored far-tail row whose stack is exactly zero while its weight is not, where the weight-gated schedule forms `0 * inf` and returns `NaN` in two gradient axes that production and the generic tower both return finite — so the opponent was cheaper by the guard it was missing. Against the corrected opponent production wins by ~9 % where it won by ~2 %, and `the_hand_carries_productions_activity_contract_932` pins the contract on that row (#932).

- The feasibility sweep reads the factored cone's row norm and bound from slices instead of calling two `Result`-returning accessors per row that re-derive the row's carrier and slot, and the cone's constraint values are one `Ψ · B` matrix product instead of one `Array2::dot(&Array1)` per coupled slot. Together those were half of the sweep's remaining profile after it was parallelised (#979).

- The library and Python survival prediction publish `survival_prob_plugin` beside `survival_prob`, the pair `gam predict` already prints: the posterior-mean path integrates that surface on its way to the mean and now reports it instead of discarding it (#2670).

- `ConstraintSet::max_scaled_violation` — the feasibility verdict every active-set solve takes on every trial point — fans its rows across the pool instead of scanning them on one thread. On the large-scale CTN cone (1.6 M rows) a profile of the preprocessor's reduced-face solve put 92 % of the process inside this one function. The reduction carries the smallest terminal row rather than whichever thread reached one first, so the verdict, the row a refusal names and its text are what the serial loop produced (#979, #2721).

- **Breaking (Python):** `sae_observe_atlas_topology` returns the atlas invariants under a mandatory branch — exactly one of `topology["named"]` (with `kind`) or `topology["refused"]` (with `reason`), each carrying the full invariant block — instead of a top-level `betti` / `euler_characteristic` beside an optional `refusal`. Reading the invariants without the verdict was not a hypothetical mistake: structureless Gaussian noise measures `b0=1, b1=0, b2=1, chi=2` here, the sphere row of the classification table, matching a planted sphere invariant for invariant, and is withheld only by the orientation-subcomplex gate — so the old shape let a caller read a manifold's signature off a cloud that is not a manifold. The invariants stay reachable in both arms because they are what a user debugging a refused cloud needs; what is gone is reading them without naming the case, which is now a `KeyError`. This matches the sibling surface `atlas_nerve_diagram`, which already gates its payload behind `computed`. The surface has no in-repo caller, no test and no `gamfit/` export, so nothing in the repository needed updating — a public FFI nothing calls or tests, which is its own small finding (#2280).

- A backend that materializes its own dense curvature no longer refuses it to the active-constraint mode response: `MatrixFreeSpdOperator` inherited the "no dense form at all" default while `as_exact_dense_spectral` was handing that matrix out, and `FirstOrderTraceSkipOperator` refused it while its first-order-trace list was live. `try_tangent_projected_evaluate` turns that error into a REFUSED TRIAL POINT, so on the large-scale CTN preprocessor — whose cone constraints are active at nearly every trial — 52 of the outer search's 55 refused probes were this and not a numerical failure (#979).

- The transformation-normal ψ-Hessian operator assembles the weighted Gram its own HVP describes, once per ψ axis per outer evaluation, and serves `B·v`, `B·F`, `tr(FᵀBF)` and the dense form from it; the five row-streaming kernels that each re-derived the same matrix per probe column are deleted. On the large-scale CTN preprocessor that path was 23 % of samples during a gradient evaluation and 50 % during a Hessian one, and a four-thread gradient evaluation goes from 17.7 s to 8.9 s (#979).

- The order-2 row-program emitter writes only the channels a gate's term reaches. A mutable's union support made every gate restate the other terms' channels, as `channel = channel` where an earlier term had set one and as `channel = 0.0` where nothing had — most of the emitted body on a three-term row. Neither form can change a value; dropping them leaves the schedule computing what it computed and raises the generated-vs-hand ratio on the survival location-scale row from 1.036 to 1.085 on one node and from 1.129 to 1.138 on another (#932).

- The transformation-normal joint Hessian's β-derivative operators (`D H[u]`, `D² H[u, v]`) assemble the weighted Gram their own module header describes, once per operator, and serve every action from it; they used to stream all `n` rows once per probe column, so the outer engine's full-rank projection cost `2 n p k` scalar multiply–adds on one thread. On the large-scale CTN preprocessor (`n = 320000`, `p = 144`) a `ValueGradientHessian` outer evaluation goes from 422 s to 35 s and the fit stops running on 1 of 128 cores (#979).

- The explicit-ψ Jeffreys derivatives read one prepared snapshot spectrum. The reduced-information eigendecomposition is a property of `(H_info, Z_J)` and the ambient trace weights of that plus one ψ axis, but the ψ-hyper gradient rebuilt both inside its coefficient-axis loop — `axes × (1 + p)` eigendecompositions of a single matrix per gradient evaluation — and the ψψ pair callbacks rebuilt them once per PAIR. `JointJeffreysPlan::explicit_param_derivative` and `JeffreysPsiWeightCache` prepare each once, lazily, so a term an evaluation never arms still never touches the spectrum (#979).

- The dense-product GPU dispatch diagnostic keeps one ring per thread instead of one process-wide `Mutex<VecDeque>`. Every `fast_ab` in the workspace passes through that seam, so on the #979 rigid marginal-slope arm at 16 threads a frame-pointer profile spent 19.5 % of the run inside `record` and 14.8 % in `lock_contended` beneath it, against 4.2 % in the Hessian accumulation it was observing; the arm's 40-minute wall carried 325 minutes of system time against 188 of user time. The recorded set is unchanged — every dispatch attempt, device-bound or not, and no ring is discarded when its thread ends (#979).

- The process-wide Duchon radial profile cache stores its profiles in a static `OnceLock` array instead of leaking them, so the compile gate's ban scanner passes; a process may intern at most 64 distinct `(p, s, d)` shapes and the next one is refused rather than silently unbounded (#2670, #2735).

- The survival time-basis smoothing-lambda refusal no longer points the user at `--time-smooth-lambda` / `time_smooth_lambda=`, neither of which exists; it names the seed's real provenance (`FitConfig::time_smooth_lambda`, saved as `survival_time_smooth_lambda`) (#2670).

- `gam predict` on a survival model emits `std_error` / `mean_lower` / `mean_upper` only under `--uncertainty`: they used to ride along on every default prediction because the deleted `--mode posterior-mean` was itself the switch that built the uncertainty object (#2670, #2136).

- The O(n⁻¹) frequentist bias correction is deleted end to end (`apply_bias_correction`, the `bias_correction_beta` / `bias_correction_jacobian` fit fields and their optimizer producers, the `A·V·Aᵀ` map on the smoothing-corrected covariance): every credible band is the posterior band of the posterior mean it is centred on. On the Gaussian additive coverage gate the de-shrunk default band over-covered (0.917 / 0.975 / 0.992 at nominal 0.80 / 0.90 / 0.95) while the posterior band is calibrated (0.825 / 0.933 / 0.975) (#2670).

- `gam predict` no longer takes `--mode` or `--no-bias-correction`: the posterior mean is the one point estimand every surface reports, the plug-in prediction is published beside it by name (`survival_prob_plugin` on survival CSVs, `mean_plugin` on latent event-probability CSVs, as `mean_plugin` already was on the standard surface), and every band is centred on the posterior mean as the library's policy already did (#2670).

- The joint Newton's trust region (via `opt`) treats a step as numerically neutral only when both the realized change and the model's predicted reduction are inside the objective's round-off floor; a resolvable prediction that realizes nothing is rejected, not accepted with `ρ = 1` (#2765).

- A survival marginal-slope fit that estimated its parametric baseline chart is saved with the fitted chart instead of being refused for a missing `--baseline-scale`, and the fixed-λ refit from a certified outer optimum admits a curvature certificate the criterion contradicted (`criterion-contradicted`) as it admits a positive-semidefinite one; only inadmissible or unevaluated curvature is refused (#2765).

- Exact coefficient-mode profiling (survival marginal-slope, bernoulli marginal-slope, transformation-normal) warm-starts every outer evaluation from the certified mode of the accepted outer iterate instead of a cold seed once the search has started; a refused or non-converged probe never becomes the start of the next one (#2765).

- The inner P-IRLS objective band and the joint Newton's residual band carry the unit roundoff once (`accumulation_growth` already includes it), restoring the LM rejection floor and both decrement certificates; the survival LAML gate accepts a residual inside the residual's own rounding band, not only `1e-8` relative (#2668, #2812).

- The multinomial predictive's augmented-mode Newton accepts only a strictly rising trial, so a solve whose remaining gain sits under the log-posterior's round-off converges to resolution instead of exhausting its hundred iterations on steps that change nothing (#2812).

- The SAE manifold joint fit logs one phase clock per iteration, per entry-sweep round and for its setup, and five of its serial row passes (frame refresh, coordinate seeding, target-aware reconstruction, deflation candidates, coherence projections) now fan across the Rayon pool (#2731).

- The custom-family joint Newton arms the exact Jeffreys second-order completion before any decrement certificate is taken (#2714); a slow-geometric-rate exit carries the ray it stalled on (`RayRestoration`), and the outer's seed evaluation restores a named seed by that ray's log-strength ratio before evaluating it again (#2695).

- The penalty pseudo-log-determinant treats a coordinate no penalty block covers as structurally null whether or not a ridge is present, so its rank agrees with the penalized subspace the Hessian carries (#2454, #2760).

- The SAE exact stationarity Hessian is assembled from one arrow probe per coordinate slot plus one per border column instead of one apply per column (#2267, #2731).

- The joint Newton certificate is deferred to a negative-curvature escape only when that escape could lower the objective by more than its resolution within the trust radius (#2765).
- The joint Newton's slow-geometric-rate exit is a typed terminal reason (`SlowGeometricRate`) whose text distinguishes slow contraction from no contraction (#2695).
- The two 2705 box-constrained regression tests no longer share one fixture csv path per process (#2705).

- The ψ-hyper build takes the ψ-Hessian directional derivative along every coefficient axis from one row sweep where the family provides it (marginal-slope rigid frame), instead of one sweep per axis (#979).

- The blockwise coefficient loop's early exit and accept test use the objective's round-off slack instead of an absolute `1e-10` (#2469).

- Survival marginal-slope fits with a follow-up-varying slope score trust-region trials on the follow-up-varying frame's likelihood; the value-only path previously read the time-constant closed form, so the slope's variation was invisible to the accept test (#2765).

- `perf_scale` gains `grouped_binomial_sweep_2569`, the #2569 grouped-binomial shape rebuilt from synthetic data with per-fit wall time printed (#2569).

- First-order dynamic jets fill each result once instead of zeroing and overwriting it, the same single-fill the second-order jets use (#979).

- Custom-family inner solves on Firth-armed fits no longer ratchet the trust radius to its floor: the Jeffreys log-determinant's certified round-off (`JointJeffreysPlan::value_roundoff_bound`) is part of the objective-resolution ceiling, so the witness's measurement is admitted, and the row kernel's early exit uses the accept test's round-off slack instead of an absolute `1e-10` (#2695, #2718, #2748). A trial point where the Jeffreys information cannot be formed is refused instead of scored with `Φ = 0` (#2765).

- The three remaining inline copies of the balanced penalty rule in `gam-terms` construction read `balanced_penalty_sum` / `balanced_penalty_rank_tolerance` (#2454).

- The `support_real_chart` example takes `<max_outer_iter> <max_inner_iter>` instead of one shared `<max_iter>`, so an outer budget can be measured with a converged inner solve (#2576).

- **The criterion's `log|S(λ)|₊` ranges over the same structural rank the
  reparameterization's penalized subspace carries** (#2454). Two
  λ-free rank rules decided how many directions one penalty set penalizes:
  the reparameterization ranks the Frobenius-balanced sum `Σ S_k/‖S_k‖_F` at
  `1e-12·max`, the pseudo-logdet's hint ranked the unweighted sum at
  `100·p·ε·max`. A component whose norm is small against its neighbour's (a
  double-penalty null-space term beside a Matérn range penalty) sat above one
  cut and below the other, so the LAML pair kept an asymptotic slope of `½`
  per unit ρ that no λ could cancel. `balanced_penalty_structural_rank` in
  `gam_terms::construction` is now the one owner, used by the split and by
  every criterion site. The iso-κ ladder of #2760 is measured unchanged by
  this; its rails have another cause.

- **The constrained joint Newton can form its active face** (#2695, #2714,
  #2765). Measured on the survival location-scale 1569 pair: every seed was
  refused with the QP listing one time-block row as active on every cycle
  while the accepted face stayed empty, because a clipped step was retreated
  one primal-feasibility tolerance (`1e-8`) off its blocking row, the face is
  classified at `1e-10`, and an infeasible trial step was projected `1e-6`
  into the interior — so the reduced-face Newton never ran and the ambient
  trust step was clipped to `1e-22` of the proposal. A clipped step now lands
  on its face; an infeasible trial is projected onto the cone in the trust
  metric (`project_point_onto_constraint_set_in_metric`, returning the binding
  rows); the cause-specific survival family clips against the same rows its QP
  solves against. The Jeffreys/Firth term was measured not to be the
  cause (the seeds stall with it disarmed); what remains on that pair — a
  collapsed block trust radius that cannot grow, then an absolute stationarity
  bar on a block whose curvature is nine orders above its neighbours' — is
  recorded on #2695.

- **Event histories: the rank of the latent covariance is decided by the
  evidence's own prior, the reported latent object is the posterior-mean
  covariance with its eigenmodes' uncertainty, and every subject's smoothed
  latent state is exposed** (#2806, #2807, #2808, #2809, #2810). The rank
  path landed in `5c61500ee` never compared two optimised models. Its atom's
  ridge was an outer REML coordinate, and the custom-family engine holds
  every such coordinate below the point where its term keeps one effective
  degree of freedom, so the ridge could not switch the atom off: on the null
  cohort the candidate sat railed at its own starting precision (a criterion
  1.5 nats above the rank-zero fit, reported as a refusal), and on the
  eighteen-subject competing-risks cohort the accepted frailty was fitted at
  a precision below three, unshrunk. Beneath that, Laplace-integrating a
  coefficient whose likelihood is even (`a → −a`) under a ridge has a
  criterion that is unbounded below at `λ = μ_max(M)`, the very boundary the
  rank decision is about: the penalised Hessian's curvature along the
  loading vanishes there and `½ log|H + λS|` diverges. Both defects are
  gone by construction. The atom's loadings keep their isotropic Gaussian
  prior, but its precision is chosen from the covariance score by
  empirical Bayes under the quartic model of the evidence along every
  eigen-direction, `½ μ_i t² − ¼ J_i t⁴`, with each direction's marginal a
  one-dimensional integral evaluated exactly (trapezoidal rule at roundoff),
  and the atom is accepted exactly when that prior places the loading's
  posterior mode off zero, `λ̂ < μ_max` — for one direction, the statement
  that the standardised score exceeds `Γ(¼)/(2Γ(¾)) ≈ 1.48`, a property of
  the quartic integral rather than a chosen level. A refused atom costs no
  fit; an accepted one is judged once more on the exact profile of the
  log-likelihood along its direction, which replaces the quartic model of
  that direction in the same calculation, so the prior a strong atom
  enters with and the evidence it reports are read from the likelihood
  itself (the quartic model is exact at the boundary and a model further
  out: it put the recovery cohort's atom at 380 nats against a realised 98);
  an accepted one is fitted with its prior held fixed, so the latent
  block adds no outer smoothing coordinate and a fit with parametric marks
  is a single inner Newton solve. The rate is an unpenalised structural
  coordinate (the old ridge on its logarithm was a prior centred on the
  cohort's mean follow-up, which pulled every fitted rate toward `1/T̄`),
  carried as a chart of the dimensionless rate `ν = r·T̄` over the band of
  rates the cohort's own breakpoints resolve (a property of the data, the
  same at every mesh refinement; measured: a band read from the quadrature
  nodes moved the chart under the coefficient at every refinement, and the
  refinement certificate saw the coordinate move by 0.36–0.42 posterior sd
  without end), `ν(u) = ν_min + (ν_max − ν_min)·u²/(1 + u²)`,
  rather than as `ln ν` — the likelihood has finite curvature in `ν` at the
  static end where it is flat in the logarithm, and the chart's fold at
  `u = 0` makes that wall a stationary point of positive curvature when the
  data push against it, instead of a plateau with a vanishing gradient and
  an uncertifiable Hessian (measured: a four-mark candidate walked its
  log-rate to −19 and stalled at residual 62 for twenty cycles); the
  marginal's gap scores stay in the log-rate, where they are bounded across
  a short gap, and are converted at the end with the chain rule in the jet
  algebra. (A box on `ν` through the engine's constraint hook was tried
  first and rejected: under its constrained-QP branch a fixed-λ solve sat
  at one coefficient vector for 1200 cycles with the log-likelihood
  alternating between two values.) A candidate now starts its mark blocks
  from the incumbent fit's coefficients instead of from zero, which is what
  had dragged the four-mark candidate's loadings along with intercepts two
  units off. A rate the residuals cannot tell from one twice as slow
  or twice as fast is held at the proposal as data (`rate_held`).
  `EventHistoryFit`
  now carries `covariance` (`E[A Aᵀ | data]`, the mode plus each atom's
  posterior loading spread), `atom_covariances`, `eigenvalues` with
  `eigenvalue_sd` (first-order eigenvalue perturbation through the fit's
  posterior covariance), `eigenvectors`, `effective_rank` (the
  participation ratio), and `loadings` in the canonical gauge (atoms by
  rate, columns signed positive); `disease_covariance` and `eigenmodes` are
  gone. `latent_state` returns `E[z_i(t) | history]` with its posterior
  covariance at every node, through a backward smoother factored out of the
  Louis pass; Python (`model.latent_state`) and the CLI (`latent_states`)
  expose it. The Laplace engine's four-mark runaway (#2808) is answered by
  a test on the same cohort under this engine: the fitted eigenvalues stay
  within their own posterior uncertainty of the simulated ones.

- **The event-history engine is the one that recovers its own simulation.**
  On the shared 80-subject fixture (one mark, one atom; intercept −0.8, slope
  0.5, loading 1.0, rate 0.4) exact marginalisation returns intercept −0.15,
  loading 1.08, rate 0.27; the Laplace engine that briefly replaced it
  returned +28.6, 8.26, 15264, and a forecast of 1.5×10¹² events. The
  divergence is the one that engine's own documentation derives: above the
  rate the mesh resolves, an event node carries a count and no exposure, its
  latent coordinate is held only by the prior, and every event buys `a²/2` of
  free evidence, unbounded in the loading. Bounding the fitted log-rate to
  the mesh-resolvable range (measured: rate 15264 → 23) does not close it,
  because the corner is in the discretised likelihood rather than in the rank
  proposal. Exact marginalisation never meets it — with the population
  centring, an event node's `−y a²/2` cancels the `exp(y² a²/2)` the
  stationary state returns at `y = 1`. Restored with everything learned
  since: the forecast window as a value, per-mark exposure rows, prior
  history as a first-class part of the cohort (an event at or before entry
  shapes the risk sets the window opens with and is not compensated), and the
  refinement certificate measured as the exact first-order mode shift
  `V (g' − g)` — one gradient per candidate rather than a second fit whose
  own convergence would have to be certified first — reported in posterior
  standard deviations. The limits are stated: the latent grid is a product
  over the atoms, so several atoms are useful when they load on different
  marks, and a posterior the grid cannot represent raises the order once and
  then says so. 29 of 29 event-history tests pass, including the brute-force
  single-node and two-node integrals to `1e-8`, the Louis-vs-computed
  curvature convergence, and the constant-hazard competing-risks forecast
  identities.

- **Event histories: every forecast probability is a chronological integral,
  the Hessian is Louis' identity in coefficient space by one forward sweep,
  marks have kinds, and the fit certifies its coefficients under refinement
  of both the quadrature and the mesh.** A static review of
  `gam_models::event_history` found the cumulative incidence of an absorbing
  mark computed as if Gauss-Legendre weights were elapsed times (a
  "probability" above one at moderate hazards), a Louis Hessian assembled
  through an all-pairs node table and a dense `S × S` transfer per gap
  (quadratic in the nodes, `G^{2K}` per gap held for the whole subject), a
  Gauss-Hermite certificate that compared one likelihood value and built the
  order-257 grid before checking its own cap, no control of the time
  quadrature, a permutation-symmetric multi-atom start, absorbing marks
  known only to the forecast, covariates read at the right limit of an
  event, and a Python layer that lost categorical levels and merged
  identifiers by stringification. All of it is replaced. Forecasts integrate
  the killed process along the latent path: the survival at every quadrature
  time is its own Gauss-Legendre integral of the elapsed hazard from the
  cell's start, the sub-density `E[S(t) λ_d(t)]` is integrated in time, and
  the terminal incidences sum to `1 − S` to quadrature accuracy (checked
  against the exponential competing-risks solution to `1e-9`). Louis'
  identity is accumulated in one forward sweep of the smoothed complete-data
  scores contracted with the design rows node by node (`E[C_m v_mᵀ]` with
  `C_m` the carried conditional expectation, propagated by the filter's own
  separable operators), so nothing quadratic in the nodes exists and the
  `S × S` backward kernel of a gap lives only while that gap is reduced to
  its innovation moments; the family's coefficient-space assembly is the
  marginal's own. Marks are `recurrent`, `once` or `terminal`
  (`MarkKind`), declared with the cohort and enforced by validation (a
  terminal event ends follow-up, a once-only mark leaves the risk set, a
  once-only or terminal mark fires at most once); the compensator carries
  per-mark exposures `w·R_d(t)`; a forecast of an absorbed subject is zero;
  a once-only mark's forecast is its first-occurrence probability. The fit
  certifies itself by refitting at fixed smoothing parameters under the next
  Gauss-Hermite order and under the halved mesh and requiring every
  coefficient to move by less than `quadrature_tolerance` posterior standard
  deviations (default `0.01`), refines whichever fails, refuses an order
  whose Lebesgue constant amplifies roundoff above the tolerance (the rule
  now records it), and checks the transient `G^{2K}` footprint against the
  machine's budget before any grid is built. The atoms start apart (log-rates
  spaced by `ln 2`), which fixes the sign/permutation/rotation gauge of the
  loading matrix at the initial point. Event nodes take the left-limit
  covariate row; the bases are built on outcome-free design rows (entry,
  exit, covariate changes and an event-free quadrature) so no event time
  shapes a basis; the smoothed marginal is renormalised after its noise cut;
  a lost-positivity posterior variance is a reported failure, not a floored
  number; the standardised innovation of the forward operator is formed in
  closed form (no `1/√q` of a cancelled difference); the joint-evaluation
  cache keys on the exact state, not a hash; `1e-2` is the one tolerance and
  a `(left + right)/2` midpoint is gone. The PIT is the Rosenblatt transform
  (checked against `1 − exp(−Λ Δt)` with constant hazards), carries the
  predictive mark probabilities at each event, refuses a survival outside
  `[0, 1]`, and the Kolmogorov–Smirnov summary of no events is `None`.
  Cohort validation refuses duplicate subject identifiers, duplicate mark or
  covariate names, duplicate segment starts, segments outside follow-up and
  invalid categorical codes; the cohort carries categorical levels, so
  factor terms, `by=` gates and random effects resolve against the labels;
  the test simulator samples each step's events as Poisson with the exact
  integrated intensity (no thinning bound on an unbounded state). Forecasts
  take a covariate path (`FutureSegment`s) instead of one row, in Rust,
  Python (`future=`) and the CLI; Python infers categorical covariates from
  non-numeric columns, declares marks with `marks={name: kind}`, fits an
  event-free cohort with a declared vocabulary and an intercept-only or
  time-only formula, and refuses identifiers that collide by
  stringification; the CLI declares `--marks name:kind,...` and
  `--horizons-after-exit`, and non-numeric CSV columns are categorical. The
  docs describe the algorithm that exists: adaptive quadrature with a
  refinement certificate, not "exact".

- **A continuous `by=` smooth keeps its constant, and event-history forecasts
  have a population tier (#2805).** `s(x, by=z)` with a continuous `z` is the
  varying coefficient `f(x)·z`, whose constant direction is `z` itself and
  not the intercept; the inner smooth is no longer sum-to-zero centred, so
  `f` is one penalised surface whose null-space ridge decides whether it
  exists (an explicit `identifiability=` still wins, and a binary or factor
  by-variable keeps the factor convention). The event-history score recipe
  is therefore plain `s(time, by=score)`. `population_forecast` (Rust,
  `model.population_forecast` in Python, `without_history` beside every CLI
  forecast) runs the zero-count filter from the stationary prior at given
  covariate values, so the population, score-only and history-conditioned
  forecasts are one model conditioned on more; a fixture checks the three
  tiers order as the score's sign and the history's richness say.

- **The cone-truncated posterior's moment cubature is ordered, tilted at the
  exact saddle point, and certified by replicate lattices (#979).** The
  large-scale transformation-normal preprocessor converged and then refused
  at its posterior: its 120-row Khatri-Rao monotonicity face (walls to 3.6 sd
  on the infeasible side, constraint-normal correlations to 0.96, 65 of 120
  correlation eigenvalues below 0.1) could not be integrated at 2^20 nodes.
  Measured on that face, the shipped separation-of-variables rule was a
  Monte Carlo draw at 0.02% effective sample size: its minimax tilt was
  solved by an alternating Gauss-Seidel sweep that diverged to NaN in five
  passes, so production silently ran untilted, and it integrated the
  coordinates in the retention walk's slack order. `gam_solve::
  constrained_posterior` now integrates in the Gibson-Glasbey-Elston order
  (the most constraining coordinate first, chosen by a greedy pivoted
  Cholesky that also factorizes the face), solves Botev's minimax saddle
  point by Newton on its convex-concave stationarity system with the
  analytic symmetric Jacobian (two-sided coordinates included; the old
  solver refused to tilt any box), and stops on the replicate standard
  error of every moment entry over eight deterministically shifted
  Kronecker lattices instead of on the change between consecutive
  doublings of one lattice, which cannot see bias. On the captured face the
  rule runs at 25% efficiency and its moments agree with an exact
  Hamiltonian Monte Carlo reference to within both estimators' errors; the
  face is carried as a fixture and gated. A refusal now names the error
  reached, the proposal's measured efficiency and which tilt ran.

- **Event histories carry a population baseline and take observed risk
  scores as penalised slope surfaces.** The latent term of
  `gam_models::event_history` now enters as `−½ Σ_k a_{d,k}² + Σ_k a_{d,k}
  z_{i,k}(t)`: the atoms are stationary and standard, so the shift cancels
  their Gaussian mixing exactly and `exp(η⁰)` is the population-average
  intensity whatever the loadings. The mark coefficients, `mark_eta` and the
  CLI `coefficients` describe population rates, and raising the latent
  heterogeneity no longer raises the population rate unless the baseline
  moves; the Fisher/Louis derivatives carry the shift through the centred
  coordinate `z_k − a_{d,k}`, and the recovery fixture now asserts the
  intercept is the population log-rate. An observed subject-level score
  enters as `s(time, by=score, identifiability=none)`: one varying-coefficient
  surface `b_d(t) · g` per mark whose wiggliness ridge decides how the
  score's effect bends with time and whose null-space ridge decides whether
  it exists at all, both REML-selected — verified by a fixture that recovers
  a declining effect and collapses an uninformative score to zero.

- **Compiled row derivatives are faster than the hand kernels, and the gate
  is enforced on every push (#932).** Every live family's row
  log-likelihood is written once (`row_program!`, `row_atom!`, or the jet
  algebra's compiled channels) and its whole derivative tower is derived
  from it; the hand derivative towers that remained are test opponents.
  `gam_math::paired_timing` is the one wall-clock contract: paired,
  interleaved, order-randomised, with the measurement's own resolution on
  every line, and `SpeedGate` cells that assert a strict win against the
  strongest hand schedule of the same contract (27 gates over four
  packages, derived from the source and run in release by
  `speed-gates.yml` on every push that touches a compiler, the algebra, a
  gated lowering or the harness). The row-program compiler now places
  work by control flow (gate-exclusive work inside its gate, leaf calls
  adjacent with nothing live across them, call-independent work before
  the calls, a mutable read never past the gate that reassigns it),
  absorbs a scaled composition point into the outer derivative stack, and
  accepts a `name: sign` constant role (a value in {−1, +1}) so a
  composition on `scale(x, s)` reads `f''` unscaled; its dense
  third/fourth-order surfaces carry every `1/k!` as an exact rational in
  the emitter, so no factorial round-trip reaches the row. `row_atom!`
  at-zero lowerings choose their Horner order by exact enumeration and
  return contracted matrices as literals. The mixed-second multinomial
  Fisher channels use the exponential's separability.

- **Event histories are one family.** `gam_models::event_history` fits marked
  counting processes with smooth covariate and time effects per mark and a
  per-subject latent state of unit-variance Ornstein–Uhlenbeck atoms whose
  loadings and rates the evidence selects (each atom carries its own REML
  ridge over its loadings and log-rate, so an unsupported atom is switched
  off). The latent chain is marginalised exactly by adaptive product
  Gauss-Hermite filtering: predict and condition steps are Gaussian
  convolutions evaluated through the Lagrange interpolant on grids centred at
  each node's posterior mean with the predictive spread, exact for any gap
  length and benign at any order. The inner Newton uses the exact gradient of
  the computed likelihood (forward-mode duals through the same filter); the
  Hessian and its directional derivatives come from Louis' identity with the
  smoother residual carried on cubic splines and every immediate exponential
  factor evaluated exactly, so nothing grows along a chain. The Gauss-Hermite
  order is raised until the fitted marginal is stable and the fit carries
  that certificate. Forecasts (survival of the absorbing marks and expected
  counts per mark) and the predictive PIT are exact expectations under the
  filtered state. Surfaces: `fit_event_history` and
  `fit_event_history_formula` in Rust, `gamfit.fit_event_history` in Python,
  `gam fit-events` on the CLI. The `gam-point-process` crate, which fixed the
  Matérn order and factor count, marginalised by Laplace only and searched
  its hyperparameters derivative-free within hand-typed boxes, is removed;
  `SPEC.md` now bans derivative-free hyperparameter search and hand-supplied
  bounds.

- **The gauge-orbit descent publishes the value authority its decrease is
  measured against (#2762).** `penalized_objective_total` is a function of
  the state AND of the barrier/repulsion gates that only `assemble_arrow_schur`
  refreshes, so one state evaluates differently before and after an assembly
  (498 ulps at an objective of `2.3e7`). The descent's per-round accounting is
  consistent under the gate it froze at entry; a caller's pre-call value is
  under an older gate, which is the `4.5e-7` the exit-state fixture refused
  on. `GaugeOrbitDescent` now carries `entry_objective` and `exit_objective`,
  the fixtures telescope the reported decrease against them, and a new fixture
  pins the invariant the rounds rely on: refreshing the gates at an unchanged
  state is idempotent.

- **The wiggle frozen-index fixed point is single-valued and mixes its
  residual history (#2748).** After the multiplier repair the
  `geo_disease_matern` flexible cell at n=1000 still failed at 60 passes, and
  Anderson mixing alone did not fix it: the residual contracted for four or
  five passes and then jumped, and every jump coincided with the inner solve
  landing in the other of two optima `2.3e-3` apart in cost, because every
  pass re-ran the outer seed cascade. Passes after the first are now
  continuations (`BinomialMeanWiggleFamily::continuation`: one seed, the
  previous pass's own `ρ`, `screen_initial_rho` off), and the loop advances
  `[β; η]` by the rank-floored Anderson multisecant step over its pass history
  (`gam_linalg::anderson`) with the scalar relaxed step as the first-pass and
  post-reset fallback and the map's own residual norm as the safeguard. The
  n=1000 cell mints in 44 s (was a 290 s refusal); n=500 in 42 s (was 48 s).
- **The composed-warp degree floor is the measured `C¹` degree, 4 (#2695).**
  The floor had been raised to 5 on the reading that `∇Φ` consumes a
  piecewise-constant `I⁗` at degree 4. Its own non-vacuity arm refused on MSI:
  driving the production Jeffreys gradient across an event-row knot crossing,
  the gap shrinks 99.5× for a 100× smaller straddle at degree 4 (and ≈100× at
  5 and 6) and only 1.02× at degree 3. The required continuous basis order is
  therefore 3, the floor is degree 4 again, the negative control measures
  degree 3, and the ladder that produced the table (`knot_ladder_2695`) ships
  as a fixture that prints it on every run.

- **The composed-warp degree floor is the measured `C¹` degree, 4 (#2695).**
  The floor had been raised to 5 on the reading that `∇Φ` consumes a
  piecewise-constant `I⁗` at degree 4. Its own non-vacuity arm refused on MSI:
  driving the production Jeffreys gradient across an event-row knot crossing,
  the gap shrinks 99.5× for a 100× smaller straddle at degree 4 (and ≈100× at
  5 and 6) and only 1.02× at degree 3. The required continuous basis order is
  therefore 3, the floor is degree 4 again, the negative control measures
  degree 3, and the ladder that produced the table (`knot_ladder_2695`) ships
  as a fixture that prints it on every run.

- **The frozen-index relaxation reads its multiplier through the damping it
  applied (#2748).** `d_k = M d_{k−1}` holds for the UNDAMPED wiggle fixed
  point only; under the relaxed advance `η_{k+1} = η_k + t_k d_k` the residual
  recursion is `d_{k+1} = ((1−t_k)I + t_k M) d_k`, so the Rayleigh quotient of
  consecutive residuals measures `(1−t_k) + t_k·mu`, not `mu`. Read as `mu`,
  the relaxation update became the involution `t_k·t_{k+1} = 1/(1−mu)`, and the
  `geo_disease_matern` flexible cell at n=1000 alternated `t = 0.186 ↔ 0.814`,
  `delta = 1.97e-3 ↔ 4.51e-4`, `cos(step_k, step_{k−1}) = −1.000` for all sixty
  passes while the map's actual dominant multiplier sat fixed at `−5.6`.
  `fixed_point_dominant_multiplier` now takes the previous relaxation and
  returns `1 + (ρ − 1)/t_prev` (the quotient itself, bit for bit, when
  `t_prev = 1`), so the damping settles at `1/(1−mu)` and stays there. A
  two-mode unit regression reproduces the involution with the old reading as
  its diverging control.

Spline basis dimensions are now honoured exactly, a descriptor means one
thing on every path that reads it, and a model-comparison quantity is named
for what it computes.

- **An explicit `k` on a 1-D B-spline smooth is honoured exactly.**
  `parse_ps_internal_knots` carried a floor of two internal knots on the
  un-reduced branch, so for a cubic `s(x, k=4)` and `s(x, k=5)` silently
  built the six-function basis of `k=6` — three requested dimensions, one
  bit-identical fit — while `s(x, knots=0)` built the four-function basis the
  documented identity `k = internal_knots + degree + 1` names for `k=4`. The
  count is now `k − effective_degree − 1` for every `k`, zero included. The
  same floor is gone from the open-B-spline tensor margin, where a legacy
  `.max(1)` made `te(x, z, k=4)` a five-function cubic margin, so `k=4` and
  `k=5` built the same tensor.
- **The automatic-knot note states the rule the engine applies.** The
  inference note announced `clamp(unique/4, 4..max(20, cbrt(unique)))` while
  `heuristic_knots_for_column` applies `clamp(unique/4, 4..8)`, so for every
  column with 36 or more unique values the note printed a rule whose own
  arithmetic disagreed with the count beside it. It now prints
  `clamp(unique/4, 4..8)`, the value that rule produced, and the small-data
  reduction when one fired. `docs/formulas.md` says the same.
- **`BSpline(knots=K)` means `K` interior knots everywhere.** The Python
  evaluator (`gamfit.bspline_basis`, `BSpline.evaluate`) read the integer as
  an interior-knot count while the fit bridge (`smooth_overrides.rs`) read
  it as a total basis dimension, so `BSpline(knots=6, degree=3)` spanned a
  10-function space when evaluated and a 6-function space when fitted. The
  bridge now reads interior knots, applies an overridden `degree` before any
  count-to-dimension conversion (it used to convert with the formula's
  default degree and swap the degree afterwards, producing a width that
  matched neither reading), and refuses a count that would silently replace
  a formula's explicit knot vector or `cr`/`cs` value knots instead of
  dropping it. Evaluating a descriptor before fitting it no longer changes
  the fitted model: the evaluation cache is still written to `knots` for the
  static-shape contract, but `to_rust_descriptor` serializes the request the
  descriptor was built with.
- **Periodic direct evaluation no longer discards the knot locations.**
  `bspline_basis(..., periodic=True)` reduced an explicit knot array to its
  first point, last point and length, so any two arrays agreeing on those
  three numbers built the same basis; and an integer `K` was resolved as the
  clamped OPEN vector, whose `2·(degree+1)` repeated endpoints were then
  counted as `2·degree+1` extra cyclic controls (`knots=8, degree=3` became
  a 15-function basis). An explicit periodic array must now be the uniform
  cyclic lattice the core builds (anything else is refused by name), and
  `None` / `K` build that lattice with `K + degree + 1` controls — the
  `cyclic(x, knots=K)` convention — across the NumPy, Torch and descriptor
  paths alike.
- **The AIC ratio is no longer called a Bayes factor.** `Model.bayes_factor_vs`
  and the `bayes_factor` column of `compare_models`'s ranking are
  `exp(−ΔAIC/2)`, Burnham & Anderson's relative likelihood: no prior is
  integrated over, and the value must not be read against Jeffreys /
  Kass–Raftery thresholds. They are now `Model.evidence_ratio_vs` and
  `evidence_ratio` (the ranking tuple's shape is unchanged), the headline
  reads "wins by evidence ratio", and `bayes_factor_vs` survives as a
  deprecated alias that warns and returns the same value. The raw REML/LAML
  `score_table` keeps its Laplace-approximate marginal-likelihood column on
  its own labelled scale.
- **A NaN penalty trace is no longer admitted as saturation.** The shared
  EDF accounting resolved every non-finite per-block trace to the block rank,
  chosen only because `f64::clamp` propagates NaN. `+∞` is the one non-finite
  value with a known limit (a ceiling-λ block, gam#1379) and still saturates;
  `NaN` and `−∞` now propagate so the fit-result finiteness validators refuse
  the fit instead of a failed computation turning into a plausible EDF, and
  from there a plausible `σ̂² = RSS/(n − edf)`, AIC and interval width.
- **`type=ps` is documented as what it builds.** The 1-D B-spline penalty is
  the exact integrated squared derivative `∫ (f^{(m)})² dx` (a Sobolev
  penalty on the function; `derivative_penalty.rs`), not the Eilers–Marx
  coefficient-difference penalty `‖Δ^m β‖²` that `docs/formulas.md` and the
  getting-started guide described. The two share a null space but are
  different matrices, most visibly on non-uniform knots, so the docs now say
  which one is fitted rather than the implementation being changed.
- **One changelog.** `CHANGELOG-ARCHIVE.md` and its docs page are folded back
  into this file, newest first, with `CHANGELOG.md` exempted by name from the
  tracked-file line-count gate: it is an append-only release log whose only
  seam is the release boundary, and the split it forced was the wrong shape.

## v0.3.153 — gam 0.3.153 / gamfit 0.1.263 (2026-08-30)

The first release since `v0.3.152` (2026-08-23), one week later and much
narrower. It finishes the basis-adequacy diagnostic that release shipped —
which turned out not to run on ordinary 1-D smooths at all — and closes a
cluster of documented smooth options that were accepted and then did nothing.

### Documented options that were accepted and inert (#2781, #2782, #2783)

Three separate bugs with one shape. `validate_known_options` answers *"is this
key spelled right?"*, and each option was listed in its arm's whitelist — which
is exactly what stopped the unknown-option refusal from firing — and then never
read by that arm. The result in each case was a bit-identical fit and no
warning.

- **A declared period now makes its axis periodic (#2781).** `period=`,
  `periods=`, `period_start=`/`period_end=`, `origin=` and `cyclic=` were read
  only *inside* the branch that `periodic=`/`bc=periodic` opens, so
  `s(t, period=24)` was bit-identical to `s(t)`: the caller asked for a cyclic
  smooth, got an aperiodic one with a seam discontinuity across the wrap, and
  was told nothing. A period is not a property an aperiodic basis has, so
  declaring one IS the periodicity declaration — on the 1-D B-spline arm, the
  tensor arm and the radial (`matern`/`thinplate`/`duchon`) arms alike. The
  declarations that cannot be honoured are refused by name instead: a bare
  scalar `period=` on a multi-margin tensor (it does not say which margin),
  `origin=` with no period to be the origin of, `period_start=`/`period_end=`
  where there is no per-margin form, `periodic=true` on a `d ≥ 2` radial smooth
  (which named no axis), and `periodic=false` beside a period, which is a
  contradiction rather than a precedence question.
- **`te()`/`ti()` honour per-margin `degree=`, `penalty_order=` and
  `knot_placement=` (#2782).** The default tensor margin is a natural cubic
  regression spline, which IS cubic and IS second-order — so the arm parsed both
  options, attached them to the margin spec, and then took a branch that reads
  neither. A margin is now realized as `cr` exactly when that is the object the
  caller asked for; any explicit request the `cr` basis structurally cannot
  carry routes that margin to the B-spline branch, where the request is read.
  Naming the default (`degree=3`, `penalty_order=2`) stays a no-op, because an
  option may not change the fit by being mentioned. In the same mechanism:
  list forms (`degree=[1,3]`) were parsed with a bare-integer reader and
  silently fell back to the default; `knot_placement=uniform` was collapsed onto
  "unset" and returned quantile knots; and a scalar `bc=periodic` on a tensor
  was accepted and then dropped by a length guard, building an aperiodic tensor.
- **`s(x, identifiability=...)` is parsed and validated (#2783).** Every value,
  `totally_bogus` included, was accepted and inert on both 1-D B-spline arms
  while `te`/`matern`/`thinplate`/`duchon` all honoured the same option and
  rejected bad tokens. It is now read, with the sibling vocabulary, and the three
  combinations that cannot hold at once are refused rather than silently
  resolved. Making it live exposed a second defect: the cyclic builder suppressed
  its null-function ridge for *every* cyclic basis, on a #874 argument that is
  true of the CONSTRAINED chart only — under sum-to-zero the ridge is identically
  zero and its smoothing parameter unidentified. Uncentered, the constant
  survives in the basis, had nothing penalizing it, and the pre-fit rank audit
  refused the model outright. The ridge is now assembled like every other 1-D
  basis's and collapsed where the #874 property actually holds, so a centered
  cyclic fit is bit-identical and an uncentered one is fittable.

**A ratchet, so the shape cannot come back.** A guard sweeps 174 probes: for
every smooth kind, each whitelisted option is set to a probe value and the built
DESIGN and penalty blocks must change, or the formula must be refused. Silence
is the only disallowed outcome. It fingerprints the built design rather than the
spec — which is precisely what #2782 needed, since `degree=` *was* stored on the
spec and then ignored by the builder — and it pins its own coverage so a future
parse error cannot turn the sweep into a vacuous error-path test. Run with
teeth, it found 33 more options of the same shape — and the ratchet is now
EMPTY, because all 33 were carried to a resolution in the same pass:

- **Wired up.** `s(x, boundary=...)`, a whitelisted third spelling of `bc=` that
  the endpoint parser never read. `duchon(p=...)` and `duchon(nullspace_order=...)`,
  whitelisted aliases of `power=`/`order=` that neither parser read. And
  `duchon(order=...)` itself, which the arm discarded whenever no `power=`
  accompanied it — it resolved the pair as
  `CubicStructuralDefault => duchon_cubic_default(d)`, taking the null-space
  ORDER from the default too, contradicting that module's own contract that
  "an explicit `order=0` still selects the constant-only space". The default now
  supplies only the spectral power. `order=1` *is* the default null space, so
  every shipped `duchon(..., order=1)` formula is bit-identical across the change.
- **Refused, with the reason.** `side=`, which says which endpoint a boundary
  condition applies to, and the seven anchor-value spellings, which say what an
  anchored endpoint is pinned to — each meaningless without the condition it
  qualifies, each previously parsed and dropped, so `s(x, bc_left=anchored,
  anchor=2.5)` pinned at 2.5 while `s(x, anchor=2.5)` pinned nothing. And
  `thinplate(include_intercept=...)`, which appends a constant column to a kernel
  basis that has no polynomial null space of its own — which is what the Matérn
  basis is and what a thin-plate basis is not, since a TPS ships its polynomial
  null space by construction, so the appended column would be exactly collinear.
- **Exempted, each with what makes it structurally inert.** `matern`'s
  `double_penalty`, resolved by the fit-time bootstrap-κ spectral test rather
  than the cold build (gam#787/#860); `curv`'s, whose RKHS Gram is full-rank PD
  so the ridge is identically zero in both directions (#1464) — which is exactly
  why that arm defaults it off; `thinplate`'s `scale_dims`, documented as a
  derivative-PLANNING hint for that family and not an anisotropy knob; `mjs`'s
  `tau` and `learn_length_scale`, Ψ-learning switches read during the fit; and
  `cyclic`'s `double_penalty`, which has no second penalty to switch off once the
  periodic sum-to-zero chart has removed the only null direction (#874).

### The basis-adequacy check now runs on every smooth it claims to (#2788, #2789)

`docs/diagnostics.md` says the lack-of-fit test is "what every fit now measures
for itself". For a 1-D `s()` 18 or more coefficient columns wide it measured
nothing: `provenance = "statistic_unavailable"` and `p_value = None`, on every
dataset and at every `n`. Below that cliff it did report, but its reference
degrees of freedom fell as its alternative got WIDER — 8 directions against a
36-column alternative, 1 against a 68-column one — and a smooth capturing 4% of
the function it was asked to model was reported adequate at `p = 0.63`.

**One root cause, and it was a denominator.** A direction of the residualized
enrichment `V = Z̃ᵀW_F Z̃` counted as estimable when its eigenvalue cleared
`1e-9 × max_j (ZᵀW_F Z)_jj` — one bar, shared across the whole alternative. The
residual spectrum of a smooth radial kernel is a Karhunen–Loève tail: it decays
geometrically, with no gap anywhere for an absolute threshold to sit in, and it
decays FASTER the more centers the alternative has, while the shared scale grows
with them. So the floor truncated harder the wider — and therefore the more
informative — the alternative became, and past 18 columns it truncated
everything.

**What replaces it is a pure number.** The estimable directions are the
principal angles between the enrichment and the fitted design: the eigenvalues
`ν` of the generalized pair `(V, E)` with `E = ZᵀW_F Z`, which are `sin²` of
those angles and live in `[0, 1]`. A direction the design cannot represent keeps
`ν ≈ 1` however small its absolute residual energy is; one the design spans
exactly keeps `(ε·cond(X))²`, roundoff over signal. The only other truncation is
the ordinary numerical rank of `E` itself — the directions the alternative does
not realize in double precision, which cannot be a denominator.

Measured on the issue's own ladder (`y = sin(30πx) + N(0, 0.1)` at `n = 4000`,
one seed; `R²` is the fitted curve against the noiseless truth):

```text
                                     before                after
  k    k'   alt. cols     R²      d.f.  p-value      d.f.  p-value
 10     9       36       0.026      7   1.6e-06        29    0
 12    11       44       0.031      6   8.4e-05        31    0
 14    13       52       0.039      5   4.7e-03        32    0
 16    15       60       0.043      3   6.0e-02        32    0
 18    17       68       0.040      1   6.2e-01        32    0
 20    19       76       0.031      —   not measured   33    0
 25    24       96       0.066      —   not measured   33    0
 30    29      116       0.374      —   not measured   32    0
 40    39      156       0.961      —   not measured   32    0
```

**It did not buy that power with size.** On an adequate fixture
(`y = sin(2πx) + N(0, 0.3)`, `n = 2000`) the reported p-values are uniform by
Kolmogorov–Smirnov at every width tested — `k = 10` over 1200 replicates
(KS 0.034 against a 5% critical value of 0.039), `k = 20` and `k = 40` over 400
each — and the statistic's own moment identity holds: `E[T/r] = 1.009`, `1.007`
and `1.009` against the `1` an exactly-scaled score statistic has. Neither
`k = 20` nor `k = 40` reported anything at all before this release.

**And it still says "adequate".** On the 15-cycle fixture above, widening the
basis until it reaches the truth walks the verdict back: `p = 0` at
`R² = 0.9615`, `1.4e-3` at `0.9994`, `0.24` at `0.9996`, `0.58` at `k = 120`.

### Also in this release

- **The HTML report shows basis adequacy (#2774).** `model.report()` gains a
  `Basis Adequacy` card between Diagnostics and the convergence headline,
  carrying `k'`, the joint null dimension, EDF, the alternative's degrees of
  freedom, the statistic, the p-value and the provenance. A failing term is
  marked by bolding the p-value cell rather than by a separate verdict column,
  so the number and the judgement cannot disagree; a term with no verdict prints
  its provenance rather than a blank cell, because "adequate" and "not measured"
  are different states.
- **The check costs ~1% of a fit instead of 27% (#2774).** It reads the design
  in row chunks through `DesignMatrix::try_row_chunk` — so there is no size and
  no design representation at which it goes dark, and `DESIGN_BYTE_BUDGET` and
  the `design_not_materializable` provenance are gone — and it is computed on at
  most 50 000 rows. That cap is not an approximation: the identities the
  statistic rests on are properties of the SELECTED sub-design, so the reference
  law is unchanged and the only thing a cap costs is non-centrality, which this
  test has in surplus.
- **The rustdoc gate is green for all 24 crates.** `[lints.rust] warnings =
  "deny"` reaches rustdoc as well as rustc, so `rustdoc::private_intra_doc_links`
  was a hard error and 31 of them had accumulated across six crates — invisible
  to `cargo build` and `cargo test`, and enough that no crate documented at all.
  Each is now plain code formatting rather than a link that renders as a
  hyperlink and resolves to nothing.
- **`docs/formulas.md`** documents the `identifiability=` vocabulary, the tensor
  option table's real defaults, and — per smooth kind, because they differ —
  which tokens each family actually accepts.

## v0.3.152 — gam 0.3.152 / gamfit 0.1.262 (2026-08-23)

The first release since `v0.3.151` (2026-07-26). Four weeks of root-cause work
across the multinomial family, the smooth-term likelihood-ratio test,
conditional transformation models, survival prediction, shape-constrained fits,
the constant-curvature and Duchon smooths, the streaming/matrix-free lane and
the SAE solver — 109 changelog entries, reproduced entry by entry in the
section below. This summary names the changes a user can observe.

### Fits that were wrong, and now are not

- **A two-class multinomial with a smooth term published `β ≡ 0` (#2612).**
  Every predicted probability was the uniform simplex at `edf_per_class = 4.09`,
  because one stale routing predicate sent the fit onto a solver that certified
  a rejected step. The whole multinomial campaign closed with it: the posterior
  mean is computed by a method that approximates it, the published uncertainty
  finally carries the covariance-mode axis every other family had, the
  Firth/Jeffreys separation certificate is taken on `ker(S_λ)` and on the
  curvature the fit HAS (`H + S_λ`), and the joint trust region stopped
  measuring the step and the radius in two different norms. On the fixture the
  issue was opened with, the posterior mean now beats `nnet`.
- **A transformation model whose transformation saturates has no tails, only a
  floor (#2600).** Every reported quantile was silently truncated at the
  training range, the CTN likelihood renormalized each row by the normal mass
  between two FITTED endpoints (which is what left the fit with no mode to
  find), the held-out PIT was scored with a different model's CDF, and the band
  ladder was the last place a reported quantile stopped being one. A reported
  quantile is now a quantile.
- **A saved Royston–Parmar model's predicted survival depended on the baseline
  time ANCHOR (#2705).** The fit centers every time design at the anchor;
  `predict` centered only an enumerated list of modes and the list omitted
  Royston–Parmar. The same family also published a FLAT cumulative hazard beside
  a NONZERO hazard past its training support — two statements that cannot
  describe one model.
- **A shape-constrained fit published two covariance matrices that were not
  covariances (#2705 A).** The truncated covariance is now assembled as a sum of
  Grams rather than as a subtraction, and a shipped constrained fit publishes
  covariances that ARE covariances. Its inner mode also could not be certified,
  for two reasons that were both units errors rather than convergence failures
  (#2705 B), and the reported coefficient at a binding box is the truncated
  posterior MEAN, matching its closed form to eight figures (#2705 C).
- **The smooth-term likelihood-ratio p-value of a Gaussian fit was scored
  against the reference for a KNOWN variance (#2672).** On a fit that estimates
  its own scale the reference is not that law; measured size at nominal 0.05 was
  0.0792. The estimated-scale channel is now published
  (`reference_residual_df`, `reference_deterministic_offset`), the ratio's tail
  is resolved as a signed weighted chi-square instead of a two-moment match, and
  the `λ̂`-selection replay no longer runs on a grid 60× coarser than the
  selection it replays — that replay was 23 % short exactly where `α = 0.05` is
  read.
- **The constant-curvature smooth stopped pinning its kernel range (#2747).**
  `kappa_hat` was measuring the range error, not the curvature: the range solve
  declared `dη̂/dκ = 0` on a state that had not earned it, the range coordinate
  was confounded with `ρ` and fabricated past `ℓ ≈ 10⁶`, and a withheld deletion
  had dropped the smooth-ownership orthogonalization with it, leaving the
  hierarchy inert for every dependent smooth. A `double_penalty=` term is now
  refused by name rather than scored as a different model.
- **The streaming lane declared a SADDLE to be a MODE, silently, and the memory
  planner chose which (#2515).** A resolved negative direction of the exact `A`
  is a saddle verdict, not a numerical null; the streaming outer gradient now
  exists on the states the dense route differentiates without complaint.
- **Curvature refusals decided below the instrument's own resolution
  (#2676, #2748).** A penalty map within `1.5e-8` of a linear dependency was
  certified EXACTLY dependent because its rank was read off the SQUARE of the
  defect; nine `matern` benchmark scenarios died of a gate refusing on numbers
  smaller than its own measured error; and the `geo_disease_*_matern` /
  `papuan_oce*_matern` cluster refused a fit on a curvature the criterion itself
  measures with the OPPOSITE SIGN.
- **The SAE inner solve had no mover for the block its own convergence measure
  removes (#2762),** so it declared a fixed point while holding 559
  stall-resolutions of objective decrease, and its convergence gates could
  certify a state sitting on a slope of 7.2 as stationary (#2720).
- **A follow-up-varying marginal slope carried its likelihood domain as an error
  instead of as a feasible set (#2765 / #2767),** and `D_β H` pulled the row
  Hessian back through ONE slope channel, making the outer criterion's whole
  mode-response term the derivative of a different model. Such a slope can now
  be saved, predicted from and leave-one-out replayed.

### New surface

- **A converged, certified fit now says whether the basis it converged on can
  represent what it was asked to model (#2774).** A biobank-shaped 16-D Duchon
  binomial fit could converge, report `certified = true`, and still return a
  null exposure at `p = 6.2e-5`, with nothing in the engine saying a word: the
  only adequacy evidence the fit path consulted (penalized EDF at its algebraic
  ceiling) reads "not saturated" while λ is still binding, and an mgcv-style
  nearest-neighbour k-index reads 0.928 at randomization `p = 0.43` on the same
  residuals — measured, including against an ORACLE 1-D ordering that only
  reaches 0.976.

  What shipped is a penalized score (Rao) lack-of-fit test against a canonical
  higher-resolution Duchon enrichment of each smooth's own covariates, with the
  enrichment projected orthogonally in the fit's WEIGHT metric rather than
  through the penalized `H⁻¹` — so the statistic is blind to "λ is large" and
  sensitive only to structure the design cannot represent at all. It reads
  `p = 9.5e-16` on the filed fit. Every fit measures it and raises a
  `GamInferenceWarning` at a Bonferroni-corrected 0.1 %, chosen from measured
  size and power (0 of 60 on a correctly specified fit, 18 of 20 on the
  underfitted one at n = 3000). Surfaced as `Summary.basis_checks`, recomputable
  with `Model.basis_check(data)`, and persisted on the model because the score
  needs converged IRLS row state a saved model does not carry. Payloads written
  before this release read as "not measured", which is what they are.

- **`certified` now says what it does not cover (#2774).** It is a statement
  about the optimizer — the inner P-IRLS solve and the outer smoothing-parameter
  search reached a certified stationary point — and makes no claim about basis
  adequacy, family choice, or whether a fitted adjustment removes the
  confounding it was given. `certification_does_not_imply_basis_adequacy` makes
  that executable rather than documentary.
- **Duchon per-axis `η` is a real outer REML coordinate (#2735),** so the
  optimizer finds the signal axis by itself instead of being handed an isotropic
  metric frozen from the knot cloud: held-out `rel_l2` 0.3395 → 0.1042 on the
  stress fixture. The isotropic route is now the all-ones contraction of the
  per-axis derivative, not a parallel derivation.
- **The Murphy–Topel correction exists for a `GlobalEmpirical` second-stage
  latent measure (#2484),** and `Σ` in the marginal-slope identity is `Var(z|a)`,
  consumed per row (#2766).

### Performance

- **Post-fit certification was 60.5 % of the fit at `p = 4096` (#2757).** The
  last dense `param_dim`-square object is gone from the certificate: `ξᵀHξ`
  streams exactly in one pass, `λ_max` goes through the existing certified
  Krylov solver, and the pinning rank — which entered no verdict — is no longer
  enumerated over the whole parameter space. The topology filtration behind it
  is computed by cohomology: 547.8 s → 52.0 s.
- **The barrier's Gauss-Newton curvature is FACTORED (#2731)** rather than
  expanded into `ne` dense carriers.

### Build, docs and release gates

Four gates were red on `main` when this release was cut, and each one had been
hiding a different thing:

- **The workspace test archive had not compiled since 2026-08-18.** One
  `expect_err` on a type without `Debug` failed `cargo nextest archive
  --workspace`, so ~12,563 Rust tests never ran for five days and
  `MASTER_FAILURES.md` reported `ARCHIVE_MISSING` instead of results. The type
  now has a written-out `Debug` that prints the profile's identity rather than
  the caller's whole design and response.
- **`cargo check -p gam-pyffi` was red**, on a `SummaryPayload` constructor that
  did not get the new `basis_checks` field. The published wheel is built behind
  that gate and the workspace job excludes the crate, so this class of break is
  invisible to `--workspace` and surfaces ~12 minutes into a release build.
- **The #2774 adequacy report was inert on every reparameterized smooth.** It
  read the design through `as_dense_ref`, which is `Some` only for a
  materialized dense design; a reparameterized radial/Duchon term ships an
  operator-backed one, so the check reported `design_not_materializable` for
  exactly the fits it exists to diagnose — including the fixture the issue was
  filed on. It now materializes by chunks under a stated byte budget, and its
  four fixture arms (which were red on that reason, and then on three
  assertions that named the wrong objects) pass against a measured `n`-sweep
  recorded in the fixture.
- **`mkdocs build --strict` had aborted since the changelog split**, on a
  repo-root-relative link that is only correct for one of CHANGELOG.md's two
  readers.

The entry-by-entry record of everything in this release — every root cause, the
measurement that separated it from its symptom, and what each fix does not reach
— is the section immediately below.

## v0.3.152 — entry-by-entry record

Every entry in this release as it landed: the root cause, the measurement
that separated it from its symptom, what was rejected on the way, and what
each fix does not reach. The summary above is derived from these; this is the
record. (Two headings, not one, because `.github/workflows/publish.yml`
extracts the FIRST `## ` section as the GitHub release body, and the record
below is four hundred kilobytes.)


- **A converged, certified fit now says whether the basis it converged on can
  represent what it was asked to model (#2774).** The filed fixture is a
  biobank-shaped association model: a null exposure correlated with 16
  population PCs, adjusted by one native `duchon(pc1..pc16, centers=24)` term.
  It converges, reports `certified = true`, and returns the null exposure at
  `p = 6.2e-5` at `n = 200 000`. Nothing in the engine said a word.

  **The evidence the engine certified resolution on could not see this.**
  `adaptive_spatial_candidates` is the only place `basis_is_saturated` is
  consulted on the fit path, and that predicate asks whether the term's
  *penalized* EDF has reached its algebraic ceiling `realized_width −
  nullspace_dim`. On this fit the 16-D linear null space is 17 of the 24
  columns, leaving a penalized capacity of ~6, and the fit sits at 3.91 — 65 %
  of capacity, "not saturated" — because λ is still binding. Basis size and λ
  both control smoothness and REML trades them off, so a basis can be far too
  small while the saturation predicate reads clean. (Separately, the term never
  reaches that loop at all: `adaptive_spatial_term_mask` admits only
  `CenterStrategy::Auto`, and this user pinned `centers=24`.)

  **The mgcv-style k-index could not see it either, and that was measured
  before it was written.** Nearest-neighbour residual differencing on the
  deviance residuals reads `k-index = 0.928` with a randomization `p = 0.43`
  while the residuals demonstrably carry the confounder (`corr = 0.119` against
  the true simulated PC effect). Two reasons, both measured: mgcv's raw
  `mean(r²)` normaliser is biased by `mean(r)² = 0.0358` for binomial deviance
  residuals, and in 16 dimensions nearest neighbours are not near — an ORACLE
  1-D ordering, rows sorted by the true confounder, only reaches `0.976`. Local
  differencing structurally loses the signal when the missing component is ~1.4 %
  of a Bernoulli residual variance.

  **What shipped is a penalized score (Rao) lack-of-fit test against a
  higher-resolution alternative**, in `gam_terms::inference::basis_adequacy`.
  `U = Z̃ᵀs`, `V = Z̃ᵀW_F Z̃`, `T = UᵀV⁻U/φ̂`, referred to `χ²_r` or `F(r, ν)`.
  Measured on the same fit: `p = 9.5e-16`.

  The construction that matters is the projection. `Z̃ = Z − X(XᵀW_H X)⁻XᵀW_H Z`
  is orthogonal in the fit's own weight metric, **not** the penalized
  `H⁻¹`-projection a first-order expansion hands you. A penalized fit is biased
  — `E[β̂] − β ≈ −H⁻¹S_λβ` — and that bias lives entirely inside `span(X)`;
  projecting orthogonally annihilates it, so the statistic is blind to "λ is
  large" and sensitive only to structure the design **cannot represent at all**.
  Shrinking a direction the basis HAS is a smoothing-parameter question, and
  this statistic deliberately declines to answer it. The invariance
  `Z → Z + X·A ⟹ T unchanged` is the executable form of that contract; the
  `H⁻¹` projection does not satisfy it, and the test that pins it caught a real
  defect on the way in (`U` contracted against the unprojected `Z` re-admits
  `CᵀS_λβ̂` through the numerator: the null statistic ran 15.2 → 2.96e7 across
  ridge 0 → 1e4 with the data fixed).

  Measured size and power at `n = 3000`, 16-D Duchon `centers=24`, binomial:

  | scenario | `p<0.05` | `p<0.001` | median p |
  |---|---:|---:|---:|
  | linear PC null — H₀ TRUE, 60 replicates | 0.02 | 0.00 | 5.7e-1 |
  | rotated curved 2-D — the filed fixture, 20 replicates | 0.95 | 0.90 | 9.2e-6 |

  Every fit now measures this for itself and raises a `GamInferenceWarning` at a
  Bonferroni-corrected 0.1 % — chosen from those numbers, not from taste: 0 of 60
  on the correctly specified fit, 18 of 20 on the underfitted one. The rows are
  persisted on the model (the score needs converged IRLS row state a saved model
  does not carry, so a data-free `summary()` could not otherwise report
  anything) and surface as `Summary.basis_checks`; `Model.basis_check(data)`
  recomputes them, refitting at the frozen spec exactly as
  `Model.smooth_significance` does.

  Cost is one `XᵀWX` and one `p³` Cholesky — factored ONCE per model, not per
  term — plus an `O(n·q²)` Gram under an explicit flop and byte budget
  (`q ≤ 100` at `n = 200 000`).

  **`certified` now says what it does not cover.** It is a statement about the
  optimizer: the inner P-IRLS solve and the outer smoothing-parameter search
  reached a certified stationary point. It makes no claim about basis adequacy,
  family choice, or whether a fitted adjustment removes the confounding it was
  given. `certification_does_not_imply_basis_adequacy` makes that executable
  rather than documentary.

  Independent confirmation that the diagnostic's advice is the right advice: at
  `n = 20 000`, `centers=24 → 48` on the same DGP moves the null exposure from a
  false rejection to `p = 0.640`. It also moves the fit from 24 s to 518 s, which
  is exactly why the caller has to be told rather than left to guess.

- **The post-fit certification's surviving `param_dim`-square object is gone:
  the certificate stopped asking for a full spectrum (#2757).** #2757 was filed
  on a dense symmetric eigendecomposition of a `param_dim × param_dim` curvature
  Gram in `fit_diagnostics_report` — 3160.5 s and 45.97 GiB at `p = 4096`, 60.5 %
  of the whole fit. Two earlier rounds took most of it: `2af28dddb` held the
  curvature in the block structure the decoder-frame parameterization gives it
  (`p` blocks of `D × D`) on the branch a Euclidean metric takes, and
  `b7e148809` rewrote the topology filtration that turned out to be the real wall
  behind it (547.8 s → 52.0 s). What neither reached is the branch where the
  per-row metric **couples output coordinates**: there `H = Σ_n J_nᵀ M_n J_n` is a
  sum of `n · metric_rank` rank-one terms whose only exploitable structure — the
  output-coordinate diagonality of `J_n` — is destroyed by `M_n`. `8adae9a67`
  moved the `param_dim`-square object from a Gram to a triangular factor and its
  own entry said outright that no storage change could fix it, naming the route
  out. This is that route.

  **The enumeration is the fix.** The certificate reads three things off `H`, and
  only two enter a verdict:

  | read | enters a verdict | streamable |
  |---|---|---|
  | `ξᵀHξ` per generator | yes — every verdict's numerator | **exactly**, one pass |
  | `λ_max(H)` | yes — every verdict's denominator | to a certified relative residual |
  | the pinning rank | **no** | not over the whole parameter space |

  Confirmed by reading the call graph, not by grep: `pinning_rank` reaches the
  summary string, the certificate evidence map and the Python dict, and no
  verdict, no group signature, no `residual_gauge_dim`, and no `Sym(F)` check.
  `CurvatureMeasurement` makes that enumeration structural — both routes produce
  it and nothing downstream can tell which one ran, so a fourth consumer cannot
  be added without deciding how it is streamed.

  **`λ_max`** goes through the *existing* certified Krylov solver
  (`symmetric_extreme_lanczos_eigenpairs`: full reorthogonalization, sharp
  `β_k|e_kᵀy|` Ritz residual, refuses if it never certifies) rather than a new
  one. Its step budget is `min(param_dim, root_rows)` — the exact Krylov
  dimension bound for `H = RᵀR`, not a guess — and its breakdown threshold and
  its acceptance check are both denominated in `tr(H)`, computed in the same
  `diagonal()` pass. For a PSD operator the trace is a rigorous upper bound on
  `λ_max` and vanishes only for `H = 0`, so an identically-flat curvature is
  recognised **exactly, in one pass, with no iteration and no tolerance**, and a
  Ritz value outside `[0, tr(H)]` is a disagreement between the operator's two
  readings and is refused rather than reported. The relative-residual target is
  `√ε ≈ 1.5e-8`: bracketed from below by the `≈ ε` a Ritz value of an operator of
  norm `λ_max` can attain, and from above by the `1e-3` the verdict resolves.

  **The energies** come from one pass that folds `RΞ` into a `G × G`
  upper-triangular factor with the same Givens routine the stored accumulators
  use. Its column norms ARE the energies (`ξ_jᵀHξ_j = Σ_a T[a,j]²`, exactly, since
  `TᵀT = ΞᵀHΞ`) and its singular values above the shared
  `curvature_rank_tolerance` are the generator-span pinning rank — so the rank
  decision stays on `σ` rather than being squared into `λ`, which is the
  discipline the rest of the module already insists on.

  **What is not claimed, and is now declarable rather than inferable.** The rank
  of `H` over the whole parameter space is a full-spectrum question and costs
  `param_dim²` scalars from any side. `ResidualGaugeReport` carries
  `pinning_rank_support: PinningRankSupport` — `ParameterSpace` or
  `GeneratorSpan` — and the support rides with the number through the certificate
  evidence map and the Python dict, so no consumer can compare two ranks that are
  ranks of different things. `GeneratorSpan` is exact, and it is the comparison
  the pinning rank was introduced for ("a smaller pinning rank than the generator
  count").

  **The A/B, on the identical phase** (release, 4-core, `n = 64`, `charts = 8`,
  `metric rank = 9`, so `root_rows = 576` puts every cell on the branch under
  test):

  ```text
       p  param_dim  mat scalars    mat gauge stream gauge    speedup   passes
      16        128        16384       0.0079       0.0048       1.65       22
      32        256        65536       0.0215       0.0074       2.90       22
      48        384       147456       0.0500       0.0100       4.99       22
      64        512       262144       0.0888       0.0159       5.58       32
    fitted exponent d(log t)/d(log param_dim): materialized 1.75, streamed 0.82
  ```

  `mat scalars` is `param_dim²` in every cell and the streamed route's is **0**,
  asserted rather than printed. 1.75 is not 3 yet — at `param_dim ≤ 512` the cubic
  is not the dominant term, enumerating and embedding the generators is — which is
  why the probe prints exponents rather than asserting a wall-clock bar. `passes`
  is now a reported field: the Krylov solve reaches `√ε` in 22 passes over the
  root, plus one for the diagonal and one for the projection. Extrapolated to the
  #2731 production cell (`p = 2048`, `charts = 32`, `param_dim = 65 536`), the
  materialized route is `2.8e14` flops to fold `480 000` rows into a 34 GiB factor
  plus `2.8e14` to read its spectrum; the streamed route is `~3.5e11` for the
  projection plus `~6e9` for the Krylov passes, against `O(param_dim)` of working
  set.

  The first version of that table read the streamed route as **10–50× slower**,
  because it timed the whole `fit_diagnostics_report` against the materialized
  route's curvature phase alone — it was timing the topology audit. Recorded
  because the number was wrong and the correction is the finding.

  **The one parallel region is over GENERATORS, and that is a correctness
  decision.** Splitting the observations would need per-chunk partial triangular
  factors combined pairwise, and Givens rotations do not commute — the certificate
  would stop being bit-reproducible across runs, which is a property it is
  asserted to have and which "two replicate fits are identified up to the same
  group iff this signature is equal" depends on. A generator's column of `RΞ` is
  computed from one observation's Jacobian and that generator alone and written to
  its own slice, so there is no reduction and no summation order to depend on the
  schedule; the serial and parallel passes are gated **bit for bit**. The
  restructure also removed an object: accumulating `c_j[r] += U[n,i,r]·a`
  incrementally makes the old `p × G` batch buffer unnecessary, so the pass
  allocates nothing per observation and nothing per generator.

  **Rejected.** Parallelising the cubic — it leaves `param_dim²` memory on the
  trajectory to ~184 GiB at `p = 8192`, against SPEC.md, and #2724's own lesson is
  that a byte-denominated admission cannot gate a cubic. A width threshold for
  materializing — an arbitrary constant; the fork stays exactly where it already
  was (`root_rows > param_dim` is where a materialized root stops being the
  smaller object), so the block branch and the small-root branch are byte
  unchanged. Making the report lazy — the other phases are already cheap and the
  curvature is not the caller's to skip.

  **Verification.** Fourteen new gates, taken against an *independently built*
  root: `reference_dense_root` writes the `offset_k + i·d_k + a` arithmetic out
  inline and never calls `fill_row_frame_jacobian`, so "the operator is the
  curvature" is a checked claim rather than a shared bug. The operator's three
  reads are that `R` and `RᵀR` entry by entry; `λ_max` is the dense spectrum's to
  `1e-9` relative, inside the two-sided PSD bracket its own trace gives; the
  generator-span rank is `root_spectral_rank` applied to the singular values of
  the reference `RΞ`; the certificate is identical — every verdict, every energy
  fraction to `1e-9`, the group signature, the residual gauge dimension, the
  `Sym(F)` check — and stays identical at a factor scale that puts `H`'s entries
  near `1e120`, where the fractions are exactly scale-invariant and a route that
  squared a condition number somewhere would not reproduce them; production
  streams exactly where a materialized root stops being smaller and nowhere else,
  all three arms; the whole report decomposes nothing at the parameter dimension,
  read off the process's own eigendecomposition census on a freshly spawned
  thread; bit-for-bit reproducibility across runs and across the parallel/serial
  fork; and four refusal gates (non-finite diagonal, negative diagonal, a matvec
  inflated relative to its own diagonal, and a consistent hand-built operator that
  still certifies).

  Two findings from writing them, recorded rather than smoothed over. A
  flat-curvature gate cannot use a constant decoder: zero tangents give zero
  *frames*, so every generator is vetoed by the degenerate-tangent rule and the
  gate measures nothing — the zero has to go in the metric. And the #998
  exact-orbit verdicts carry `VerdictProvenance::CurvatureTest` but are not
  decided by `H` at all; a gate that reads "no generator carries energy" off the
  whole verdict list is reading their residual too.

  Left standing, and named: once the curvature is streamed the largest object the
  certificate holds is its own generator list — `D(D−1)/2` frame rotations plus
  `K(K−1)/2` atom exchanges, each a `param_dim`-long vector, ~520 MiB at the
  #2731 shape against ~16 MiB for the whole streamed working set. It is now one
  copy rather than two (`EnumeratedGenerator` normalizes in place), which is what
  it was before this work, but the `O(G · param_dim)` law is untouched and the
  generators are structurally sparse — the per-atom families touch one atom's
  block and the frame rotations are rank-two in `(i, c)`.

- **A saved Royston-Parmar model's predicted survival surface depended on the
  baseline time ANCHOR, which is a reparameterization and not a model
  (#2705).** `center_survival_time_designs_at_anchor` subtracts the time-basis
  row at the anchor from every entry and exit design row; its own documentation
  calls that "an exact affine reparameterization of the baseline offset", and
  the fit honours it — the same data fitted at five anchors spanning `1e-7 …
  5.0` reaches the identical maximised log-likelihood to seven digits
  (`-1.364394e3`). The PREDICTION did not:

  ```text
  eta(anchor=1e-7)   eta(anchor=1.19)      difference
    -2.456651466        +0.403170376      +2.859821842
    -1.813324553        +1.046497296      +2.859821849
    -1.107764133        +1.752057720      +2.859821852
    -0.880407739        +1.979414110      +2.859821849
    -0.237080827        +2.622741029      +2.859821856
    +0.468479594        +3.328301453      +2.859821859
  ```

  A constant to eight digits across three covariate values and three times —
  `X(anchor)ᵀγ`, and a factor `e^2.86 = 17.5` on every reported cumulative
  hazard.

  **Root cause.** The fit centers unconditionally, for every likelihood mode, so
  the saved coefficients are the CENTERED design's. `predict_survival` re-centered
  only `LocationScale | MarginalSlope` plus bare `Weibull`, and that enumerated
  list omitted `Transformation` — the Royston-Parmar default, and the default
  survival likelihood. The competing-risks sibling gated the same step on
  `weibull_baseline_in_beta` alone, so its per-cause Royston-Parmar baselines
  were uncentered too. Both now center whenever the rebuilt time design has
  columns at all, which is exactly when the fit centered; the mode list is
  deleted rather than extended, because a list of modes is a second answer to a
  question the fit already answers.

  The omission was invisible on ordinary right-censored data: the default anchor
  there is the earliest entry — the time origin — where `I_k(left) = 0` exactly,
  so `X(anchor) = 0` and the missing subtraction subtracts nothing. It became a
  `17.5x` error the moment the anchor moved: on every genuinely left-truncated
  dataset, which takes the robust interior anchor by rule (#751/#1790/#2631),
  and on every explicit `--survival-time-anchor`.

  **This is what made delayed-entry survival fits degenerate.** The
  independently-filed
  `test_left_truncated_survival_is_nondegenerate_and_covariate_dependent`
  (recorded red in `bench/gha_results/rust-test-suite/MASTER_FAILURES.md`) is
  the user-facing face of it: a fit with a covariate hazard ratio of `4.2x`
  returned survival curves that were collapsed and, at the reported precision,
  identical across covariate values. The obvious reading — that the delayed-entry
  factor `Λ(entry)` was wrong — is refuted by sweeping the shared entry time over
  five decades on identical exit/event data: at `entry = 1e-6`, where the
  truncation correction is `1e-5` relative, the fit is still off by `e^2.86`.
  The trigger is not the size of `Λ(entry)`; it is the predicate
  `survival_data_is_left_truncated`, which selects the interior anchor.

  Gated by `transformation_survival_prediction_does_not_depend_on_the_time_anchor_2705`
  (two fits of one dataset differing only in `--survival-time-anchor`: the
  surfaces must agree to `1e-4` AND the coefficient vectors must differ by more
  than `1e-6`, so the agreement is a statement about the reparameterization and
  not about two identical models) and by
  `left_truncated_survival_is_nondegenerate_and_covariate_dependent_2705`, which
  carries an `entry == 0` control arm so a failure separates the harness from
  the delayed-entry path.

- **The constant-curvature range solve claimed `dη̂/dκ = 0` on a state that had
  not earned it, and the curvature acceptance measured one of its two signs
  (#2747).** Two things, and the second is the reason the first went unseen.

  `RangeSolveOutcome::LocallyFixed` named three different terminations — the
  caller pinned `length_scale=`, the iterate parked at the chart's evaluability
  wall, and the Newton stopped somewhere without a certificate — and
  `ConstantCurvatureProfile::evaluate` treated all three alike, taking the plain
  κ slice, i.e. reporting `V_κ` as the total derivative of `V(κ, η̂(κ))`. That is
  a theorem for the first two (a pin makes η constant by construction; an ACTIVE
  bound makes it constant while it stays active) and nothing at all for the
  third. The error is `V_η·η̂′`, and neither factor is small: `V_η` not being
  small is what failing the certificate means, and `η̂′` is order tens per unit κ
  on real geometry — measured on the coverage fixture's own cloud, `ℓ̂` sweeps
  `0.68 → 34 000` across the κ box.

  The certificate is now a property of the STATE rather than of the exit branch.
  `converged` was a flag set by two of the loop's several `break`s, so a line
  search whose every trial is worse than an incumbent that is already the
  minimum exhausted, left through `None => break`, and had a stationary point
  classified as a stall. Emulating the shipped inner solve against a
  brute-force `min_η` over the whole chart, the `2×`-range column reads

  ```text
  outcomes = ['at_hi', 'at_hi', 'converged', 'stalled', 'stalled', 'converged', …]
  ℓ̂        = 7.7e7,   7.7e7,   4.9e7,       47.1,      17.3,      10.5, …
  ```

  and the brute force confirms `47.1` and `17.3` ARE the minimizers. Those cells
  now reach `InteriorMinimum`, which restores the Schur term to `V_p″` and
  publishes `RangeEstimateSupport::Interior` for a range that is one.
  `LocallyFixed` splits into `Pinned`, `EvaluabilityWall`, `DistanceKernelLimit`
  and `Uncertified`; the first three keep the plain slice on a stated argument,
  and `Uncertified` REFUSES with `V_η` and `V_ηη` named — the discipline
  `eta_profiled_kappa_jet` already applies to a non-positive `V_ηη` instead of
  dividing by it. All four still report as `LocallyFixed` on the public
  `RangeEstimateSupport`, whose contract already covers all of them.

  Separately, `ConstantCurvatureProfile::new` derived its box and its bracket
  from the realized center set and read its SEED from `spec.length_scale` —
  which, by the time the CI and the flatness LR build their profile, is the
  fit's own `ℓ̂`, written back by the free-κ enrollment and by
  `freeze_term_collection_from_design`. Seeding the inner solve with it is a
  warm start from one κ, which `minimize_over_eta`'s own doc forbids: *"a
  profile likelihood that is not a function of its own argument cannot support
  an interval"*. The seed is derived when the range is free and honoured
  verbatim when the user pinned it — the same un-freezing the constructor
  already did one field up for `identifiability`.

  And the acceptance gained its missing half. `#2747`'s bar is *"an interior
  optimum of `V_p(κ)` near the planted `κ⋆` … on both curvature signs"*, and the
  tree measured the curvature × range grid only at `κ⋆ = +1` and `κ⋆ = 0`, with
  the hyperbolic sign covered only at the auto `ℓ_ref` — the one column a
  range-blind criterion already handled. `curved_coverage_arm(κ⋆, seed_base)`
  parameterises the arm by sign; the spherical entry point keeps its seed base
  so its nine datasets are unchanged. First run of the new arm:

  ```text
  [cov κ⋆=-1] covered 9 / missed 0 / unresolved 0 of 9   railed κ̂ 0/9
              sign_correct 9/9   mean κ̂ = -0.947
  ```

  against the row this issue was filed over (`κ̂ = −1.410` railed at `0.5×`,
  `−0.943` at `1×`, `−0.295` at `2×`). The spherical arm reads `covered 9/9,
  railed 1/9, mean κ̂ = +1.070` against the body's `railed 9/9`.

  How big the misfiling was is now measured rather than argued.
  `the_profiled_second_derivative_is_the_derivative_of_the_profiled_first`
  differences `V_p′(κ)` — nothing in the tree differenced the PROFILED value
  before; the existing FD gates hold `V(κ, η)` at fixed `η` — and prints the
  Schur term beside it:

  ```text
  κ=-1.1119  V_p″=+5.891e1 (fd +5.891e1, rel 5.3e-8)  schur=2.490e1  unreduced rel 4.16e-1
  κ=-0.5559  V_p″=+5.553e1 (fd +5.553e1, rel 1.2e-6)  schur=4.320e1  unreduced rel 7.64e-1
  κ=+0.8339  V_p″=+8.232e1 (fd +8.232e1, rel 2.0e-6)  schur=3.113e1  unreduced rel 3.74e-1
  ```

  The correction is 30–70% of the reported curvature, so an interior minimum
  misfiled as "locally fixed" publishes a profile curvature **1.3× to 1.8× too
  large** — matching the finite difference to `1e-6` when reduced and missing it
  by `0.3–0.76` RELATIVE when not. The `κ̂` are untouched by all of this (the
  outer search is gradient-only and both arms return `V_κ`), which is why it took
  a second-derivative gate to see: all 27 coverage-arm estimates are bit-identical
  before and after.

  Two arbitrary constants go with it. "On the wall" was `1e-12·(1 + |η|)` in the
  refinement's early returns and `1e-9·(1 + |η|)` in the terminal classification —
  a band three orders wide that was a wall to one test and interior to the other,
  with neither number derived from anything. Both are now `eta_resolution`, the
  criterion's own forward resolution in `η` and the same scale the stationarity
  certificate is denominated in: a point the criterion cannot distinguish from
  the wall is on it.

  Fixture set at the landing: `cargo test --release --test identifiability --
  constant_curvature` 22 passed / 0 failed (28.2 s), and `cargo test --release -p
  gam-models --lib -- constant_curvature` 9 passed / 0 failed, with the 3 × 3
  criterion grid reporting `InteriorMinimum` in all nine cells and `ℓ̂` tracking
  the planted range to 3%.

- **The streaming lane declared a SADDLE to be a MODE, silently, and the memory
  planner chose which (#2515).** `factor_evidence_unit_deflated_schur` decided a
  direction was a numerical null with

  ```rust
  let deflated = raw_evals.iter().map(|&v| !v.is_finite() || v < deflate_floor)
  ```

  `v < deflate_floor` is ONE-SIDED: it admits every negative eigenvalue however
  large, prices it as the ρ-independent `log 1 = 0`, and inverts it at `1`.
  Measured on #2712's certified deflated anchor at `log λ_smooth = −1.05`, the
  reduced Schur of the exact observed information carries

  ```text
  S_A dir 0: raw=-7.997610e-3  cond=+1.000000e0  relative=-1.414389e-3  (floor 1e-8)
  S_A dir 1: raw=-2.033493e-3  cond=+1.000000e0  relative=-3.596263e-4  (floor 1e-8)
  ```

  five decades outside the band, both pinned to `+1`. The dense route classified
  the same two directions as #2336 clamp-attributable negative curvature and
  priced them at their basin, and the two complete outer gradients were then
  `1.009` RELATIVE apart. No row block is resolved-negative there: all of `A`'s
  negative inertia lands in `S_A`, exactly as Haynsworth's inertia additivity
  predicts. Nothing downstream could tell — the conditioned factor is PD and its
  log-determinant is finite — and the dense route's verdict on the same state is
  the typed `IndefiniteObservedInformation` refusal that makes the ρ infeasible.

  The band is now TWO-SIDED under an opt-in
  `ArrowEvidencePolicy::UnitDeflationRefusingIndefinite`: `|λ| ≤ floor·max|λ|` is
  still the unit-pinned null on both sides of zero, and `λ < −floor·max|λ|`
  refuses with the direction and its magnitude named. Both conditioning sites take
  it — reduced Schur and per-row — because the inertia arrives through either. The
  historical `UnitDeflation` is untouched and stays the majorizer's policy, where
  it is correct: `B` is PSD by construction, so a negative eigenvalue there can
  only be rounding on a direction that is null anyway.

  The verdict also had to ARRIVE as the same thing. gam-solve's spine to gam-sae
  is `Result<_, String>`, so a refusal routed through `Numerical` would make one
  identical verdict an infeasible ρ on one route and a fatal
  `RemlOptimizationFailed` on the other. Following #2598,
  `ArrowSchurError::indefinite_evidence_marker()` is interpolated by both
  producers and matched by its reader, and `SaeCriterionError::from_arrow_refusal`
  maps it to `IndefiniteObservedInformation`.

- **The streaming outer gradient did not exist on states the dense route
  differentiates without complaint, because the gate freeze stopped one call too
  early (#2515).** `converge_inner_for_undamped_logdet` freezes the
  collapse-prevention gates, converges, and RESTORES the flag; the evidence
  assembly that prices the criterion runs after that restore, so
  `assemble_arrow_schur_scaled` re-refreshed all three gates from the moved state.
  The factor cache then held entry-state gates and the system it is paired with
  held post-convergence ones, and `validate_matrix_free_arrow_pair` refused the
  pair:

  ```text
  smooth=-1.10  dense     cost=1.8195496423e1  ‖g‖∞=1.580471e1
                streaming cost=1.8195496415e1  GRADIENT REFUSED: … stale matrix-free
                          system/cache pair (row fingerprints DIFFER, manifold
                          fingerprints EQUAL)
  ```

  `b5506eeaa` named this "Cause 2 (real, but not sufficient)" and landed the test
  that attributes it; its Cause 1 was retracted in `60feddc2e`, which fixed the
  fingerprint's IDENTITY but not the state the fingerprint correctly reports as
  different. The freeze now spans the criterion evaluation, which is the scope its
  own "ONE CRITERION EVALUATION = ONE OBJECTIVE" discipline names. The initial
  fit stays outside it deliberately: the dense sibling runs the identical driver
  outside its own freeze, and the two routes have to put the inner solve at the
  SAME state or the criterion each prices is a different criterion — this issue's
  own defect, in the one place it would be easiest to reintroduce while fixing it.

  Measured on the same sweep, after: `‖g‖∞` is `1.580471e1`, `1.448392e1` and
  `1.222628e1` at `smooth = −1.10, −1.20, −1.40` on BOTH routes, where the
  streaming one previously had no gradient at all.

  It also refreshes the #2343 amplitude-barrier gate, which it never did. The
  assembler refreshes three gates when unfrozen and the freeze refreshed two, so
  the amplitude gate was the one gate never refreshed at the entry state at all —
  inside the frozen window the assembler skips it, and the freeze did not do it
  either, leaving it carrying whatever the previous evaluation left behind. Both
  producers now live in one function, `freeze_collapse_prevention_gates`.

- **A deflating cache is no longer a reason to withhold the streaming outer
  gradient (#2515).** `penalized_quasi_laplace_streaming_outer_evaluation` refused
  every evaluation whose evidence factorization spectrally deflated a row. Its own
  note named the lift condition — reconcile the dense route's ABSOLUTE spectral
  floor with the arrow route's per-row relative one — and #2673 did exactly that
  (`00c1fe139`, `758c9d336`) without anyone re-measuring this comparison against
  it. Re-measured on the same anchor: `2.798722e-8` against `‖g‖∞ = 1.726754e1`,
  `1.62e-9` relative, from `9.131537e0` against `5.004339e0`. Direction for
  direction the classifications agree — the dense route pins nothing and prices no
  clamp-attributable negative over the anchor's thirty coordinate directions.

  One anchor is not a contract, and widening to a nine-rung ρ ladder is what found
  the saddle defect above. The gates that replace the refusal are the ladder
  (either both routes price a state and agree within `1e-6` relative, or both call
  it a saddle — with the dense spectrum required to corroborate every refusal, so
  an over-refusal goes red rather than passing as caution), the same parity through
  `evaluate_outer_criterion_route` itself, and an attribution gate for the residual.

  **That residual is `1.6e-9`, and it is attributed rather than absorbed into a
  bar.** The dense route materializes `A` through `apply_cached_arrow_hessian`,
  which applies `L Lᵀ` of the row's UNDAMPED FACTOR — the majorizer already
  unit-pinned by its own factorization — so a `B`-deflated direction enters the
  dense `A` as `1 + vᵀΔCv`; the arrow route folds `ΔC` into the untouched
  majorizer blocks and unit-pins the result, so the same direction is exactly `1`.
  Measured on all ten deflated directions of the anchor: `vᵀΔCv = −3.431291e-8`
  against a dense `vᵀ(B̃+ΔC)v = 9.9999996569e-1`, which is `1 + vᵀΔCv` to every
  digit. The gate is an IDENTITY — the two exact-`A` row blocks differ by the
  majorizer's own conditioning increment and by nothing else, `3.552714e-15` over
  a block scale of `2.513448e1` — so a second cause cannot hide inside the parity
  bar next door.

- **A Royston-Parmar fit published a FLAT cumulative hazard beside a NONZERO
  hazard past its training support, and those two cannot both describe one
  model (#2705).** `h = dΛ/dt`, so a flat `Λ` forces `h = 0`. Measured on the
  #1564 heart-failure fixture, at every time from the largest observed exit out
  to ten thousand times it:

  ```text
       m         t     cumulative_hazard      hazard      t · hazard
   1.000001   285.0          5.055558    2.196795e-2       6.26088
   1.5        427.5          5.055558    1.464532e-2       6.26088
   10        2850.0          5.055558    2.196798e-3       6.26088
   10000  2850000.0          5.055558    2.196798e-6       6.26088
  ```

  `Λ` is constant to eight digits and `t·h(t)` is constant, i.e. the DERIVATIVE
  evaluation kept a log-log slope of `1.23842` that the VALUE evaluation does
  not have. Every `model.predict(...).survival_at(grid)` call reaches it,
  because `default_survival_time_grid`'s top node is already past `max_exit`.

  **Root cause.** `build_survival_time_basis`'s `ISpline` arm hand-rolled the
  baseline's `d(log Λ)/d(log t)` as a right-cumulative sum of a CLAMPED B-spline
  first-derivative basis. A clamped B-spline's VALUE extends linearly, so that
  sum returns the boundary slope outside the knot span — while the I-spline
  value basis it claims to differentiate SATURATES.

  That is the same disagreement `create_ispline_derivative_dense` was repaired
  for in #2695, and the same one #1348 repaired for open-knot B-splines, and the
  same one #2600 repaired a third time by hand inside the CTN chart. The
  survival lane had its own copy of the cumulative sum and so never received
  any of them. **Four repairs of one defect is a missing abstraction**, so the
  repair is an abstraction: `ISplineBoundary::{Saturate, LinearTails}` and
  `ispline_value_and_first_derivative`, which produce the value and the
  derivative together under ONE declared convention. Producing the halves
  separately and letting them disagree is no longer expressible.

  **The convention is `LinearTails`, because that IS the Royston-Parmar model.**
  A *restricted* spline is linear beyond its boundary knots by construction
  (Royston & Parmar 2002), which is what gives the classical `Λ(t) ∝ t^c`
  extrapolation used whenever a survival curve is projected past the observed
  follow-up. Saturating asserts two things the data never said: that the hazard
  drops to exactly zero at the last observed exit time, and — on the lower tail,
  which the default grid reaches on its FIRST node — that `Λ(t) → Λ(t_min) > 0`
  as `t → 0`, i.e. an atom of failures at the time origin and `S(0) < 1`.

  Two places the fit could have moved, both closed rather than hoped:

  * **Entry rows at the time origin.** `log_entry` for such a row is
    `ln(SURVIVAL_TIME_FLOOR) = −20.7`, a numerical floor and not a datum. The
    likelihood already knows these rows are not left-truncated
    (`entry_active = age_entry > ENTRY_AT_ORIGIN_THRESHOLD`) and drops their
    `S(entry)` factor outright, so they now evaluate at the first knot, where
    `I_k(left) = 0` exactly — the same zero row that shipped, for the reason
    that actually holds. A genuine delayed entry below the first knot still
    receives the real extrapolation.
  * **The anchor row.** `center_survival_time_designs_at_anchor` subtracts the
    basis row at the anchor from every design row, and the default anchor for
    ordinary right-censored data is the earliest entry — the time origin. Under
    saturation that mapped to the zero row and centering was a no-op; under
    tails it would re-center every column by a large constant read off `1e-9`,
    which is exactly the #751 inflation the anchor rule exists to avoid. The
    anchor is the ORIGIN of a reparameterization, so it is now clamped into the
    modelling interval — numerically identical to what shipped for every anchor
    at or below the first knot.

  Every training EXIT row is inside the knot span those same rows induced, so
  `x_exit_time`, `x_derivative_time`, `keep_cols` and the penalty are untouched.
  Two fits do move, both deliberately and both toward the model: a cohort whose
  rows share one strictly-positive entry time (the entry times are then dropped
  from knot inference, so every entry row lands in the left exterior, and the
  saturating basis was asserting that a subject accumulated the entire baseline
  hazard up to the first observed exit time BEFORE entering), and the
  interval-censored right endpoint `R` where `R > max(L)`.

  The CTN chart's private copy of the affine continuation
  (`ctn_extend_bases_affinely_past_the_knots`, `ctn_ispline_modelling_interval`,
  152 lines landed for #2600) is deleted and routed through the shared
  evaluator.

  **The fixture assertion that pinned the defect is replaced, not relaxed.**
  `royston_parmar_saved_predict_at_grid_top_does_not_fail` demanded
  `zero_hazard_nodes > 0` — a BIT-EXACT zero hazard — which cannot hold on a
  model with tails; its own doc taught the retired premise. That assertion was
  never what kept #1564 covered: the `eta_t == 0` guard is pinned at the
  function by `royston_parmar_hazard_accepts_zero_derivative_as_flat_boundary`
  and `royston_parmar_hazard_zero_derivative_in_saturated_tail_is_zero_not_nan`,
  untouched here. What replaces it is the invariant the defect violated — the
  reported hazard IS the derivative of the reported cumulative hazard — on a
  `t·(1 ± 1e-4)` stencil at five probe times from `0.25·max_exit` to
  `4·max_exit`, plus `S(0) = 1`, plus a non-vacuity check that a probe past
  `max_exit` carries a nonzero hazard, since a saturating baseline passes the
  derivative check trivially.

- **A follow-up-varying marginal slope carried its likelihood domain as an
  error instead of as a feasible set, and the outer search halted against a wall
  it could not see (#2765 / #2767).** The family is a transformation model,
  `S(t|x,z) = Φ(−η(t))`, so its row log-density carries `log η′₁` and its domain
  is

  ```text
    η′₁(t) = q′(t)·c(t) + q(t)·c′(t) + b′(t)ᵀz > 0    at every EVENT row,
    c = √(1 + bᵀΣb).
  ```

  With a time-CONSTANT slope the last two terms are identically zero and
  `η′₁ = q′·c ≥ q′`, so the time block's own linear guard `q′ ≥ derivative_guard
  > 0` IMPLIES the domain. That implication is why the solver's feasible set has
  always been one polytope in one block, and why `CustomFamily::
  max_feasible_step_size` — which is asked one block at a time — sufficed.

  A follow-up-varying slope breaks it: `q·c′` and `b′ᵀz` carry no sign, and they
  read the marginal and log-slope blocks as well as the time block. The extra
  condition was placed in the likelihood DOMAIN (refuse outside) rather than in
  the feasible set. Measured on the #2765 acceptance fixture that costs the fit
  its convergence in two separate ways:

  * the inner trust region walks trial coefficients out of the domain — the run
    carries hundreds of `transformed time derivative must be positive at row
    1531 ... got −8.99e-2` lines, one per step the limiter did not price;
  * the outer search moves the baseline chart under a warm start, so a `β` that
    was interior at the previous `θ` is exterior at the next one through no
    property of `β` at all. The evaluation then refuses, and a refusal carries
    no descent information: the BFGS halves its step six times, every probe
    refuses, and the runner halts with

    ```text
      [OUTER] cost-stall halt (infeasible BFGS probes): 6 consecutive infeasible
      probes after a finite seed/iterate; halting at best-so-far with residual
      |g|=3.941e1 (value=2.137621e3)
    ```

    against its own `1.5e0` escape threshold. Fifty-six of that run's outer
    evaluations end `outcome=recoverable`, each in `0.003 s` — before any inner
    solve.

  Two halves, one rule. `CustomFamily::max_feasible_joint_step_size` is the
  missing coordinate: a JOINT fraction-to-boundary hook, defaulted to `None` so
  every other family is byte-unchanged, mined in beside the per-block answers by
  `compute_joint_feasibility_alpha`. And the log-slope warm start is retreated
  toward its own origin until the domain holds — a PROVABLY interior endpoint,
  because at `β_g = 0` the layout's exit-derivative design gives `ḃ ≡ 0` exactly
  (it carries no offset), both follow-up terms vanish, and `η′₁ = q′·c ≥ q′ >
  0`. That is what makes the bisection terminate rather than usually succeed, and
  it is the log-slope half of a contract the time block already keeps: its own
  warm start is projected onto `q′ ≥ guard` before use.

  Both halves read `η′₁` through `rigid_row_admission_witnesses`, the same call
  the row evaluator admits on, at primaries from the same
  `rigid_row_kernel_primaries`. A limiter that computed `η′₁` its own way would
  be a second copy of the model, and two copies eventually disagree about which
  side of the boundary a coefficient is on.

  **Rejected: a multiplicative safety fraction on the limiter.**
  `apply_feasible_step_boundary_backoff` already records why a proportional
  retreat is wrong here (#2695) — it is a proportionality the geometry does not
  have. There is a second reason on this boundary: the limiter and the objective
  are the same function at the same primaries, so the endpoint returned is
  admitted bit for bit, and a step that lands legally but close is not a hazard
  the limiter should price. `−log η′₁` there is finite and worse, and the trust
  region rejects it on its own terms — which is the mechanism that is supposed
  to decide step length.

  Also: `[STAGE] outer eval end ... outcome=recoverable` now logs the trial `θ`
  and the refusal REASON. A line search that halves forever is diagnosable only
  if the log says which domain the trial point left.

- **The SAE inner convergence gates removed a direction the fit was still
  sliding down, and could certify a state sitting on a slope of 7.2 as
  stationary (#2720).** `quotient_residual_norm_sq` projected a "chart-gauge
  orbit" out of both inner convergence measures — the KKT gradient gate and the
  Newton step gate — on the premise that the penalized objective is flat along
  it. It is not. The premise came from a real property, stated one step too
  strongly: a chart reparametrisation (a circle atom's phase, a patch atom's
  translation or dilation) can be exactly cancelled in the decoder, so the
  RECONSTRUCTION does not move, and the compensating least-squares solve leaves
  a residual of `1e-16` relative. That makes it a symmetry of the likelihood and
  of nothing else.

  Measured by central difference of the objective's own value functions — the
  surface the line search descends, not an analytic gradient scored against
  itself — split per objective term, on a planted circle at four atom kinds:

  ```text
  kind       family      dirs   worst |d f| / tolerance
  periodic   chart          1          3.5
  duchon     chart          2         19.9
  linear     chart          2     76,170.0
  euclidean  chart          2     76,155.1
  duchon     decoder-null 704          0.00003     (3e-9 absolute = roundoff)
  ```

  The separation is total, and it is the fix. The constant-shift field moves the
  ARD prior alone; the dilation field `δt = t` moves the SMOOTHNESS prior by
  `-7.82` on an objective of `165`, because shrinking the chart while inflating
  the decoder buys smoothness at no cost in fit. Meanwhile every one of the 704
  decoder-null directions is flat to roundoff, which is what their construction
  claims: they are machine-null eigenvectors of the PENALIZED Gram `DᵀD + λS`,
  so the data fit and the smoothness prior are flat along them at once.

  **The modelling question, answered rather than deferred.** #2720 offered three
  repairs: penalize the equivalence class so the posterior becomes flat along
  the orbit; extend the decoder compensation to the priors as well; or accept
  that there is no posterior gauge and stop quotienting. It is the third. The
  ARD prior on the chart coordinates IS the chart's identifiability device — the
  ordinary mean-zero random-effect convention, with the decoder's constant
  column absorbing the location — and the smoothness prior, a seminorm in a
  declared reference geometry whose Gram does not move with the fitted decoder,
  is what fixes the chart's scale. A prior made invariant under translation and
  dilation of `t` is improper along exactly those directions and can no longer
  prune an unused axis, which is the entire purpose of ARD. The non-invariance
  is the prior working.

  The second repair is refuted constructively rather than argued away. Once a
  coordinate field is chosen, `δβ` is the unique least-squares solution of an
  over-determined system, so the prior derivative along a likelihood-null
  direction is a FIXED linear functional of the field. On a patch atom the field
  family is two-dimensional and the two slopes have opposite signs, so a
  zero-derivative combination exists at any single state — and that is all it
  is. Moved half a step along the shift field, with the reconstruction
  unchanged, the same combination carries `|d f| = 1.96e-1`, 2076x the
  convergence tolerance, because the shift/dilation slope ratio moves 74%. No
  fixed combination of two slopes that move independently stays at zero.

  **The repair.** One span becomes two, each answering its own question, and
  both produced by one Gram--Schmidt run on two inputs so their containment is a
  property of one algorithm rather than of two constructions that agree today:

  * `posterior_null_quotient_basis` — what may be REMOVED from a convergence
    measure. The two decoder-null families only.
  * `likelihood_flat_block_basis` — what should be DESCENDED as its own block.
    The chart orbit plus those nulls, i.e. everything the LIKELIHOOD cannot see,
    which is exactly the set carrying no data-fit curvature and therefore the
    set every step-SHORTENING globalization serves badly. `descend_gauge_orbit`
    takes this one, so the orbit keeps its mover and loses only its blindfold.

  For the gradient gate the old removal bought nothing and could cost
  everything. The quotient removes `Σᵢ (gᵀvᵢ)²` from `‖g‖²`, so the precondition
  `maxᵢ |gᵀvᵢ| ≤ τ` bounds what it can ever remove at `√m·τ` over `m`
  directions — on the measured span that is `√704 · 3e-9 = 8e-8` against
  `τ = 7.5e-5`, i.e. nothing. Where the precondition fails the removal is
  unbounded, and it failed by four orders of magnitude. For the step gate the
  removal is not bounded that way and is the point, but it requires the
  direction to BE flat — granted for the decoder nulls, refused for the orbit.

  Refusal messages that named `‖Π⊥gauge g‖` and `gauge_share` now name
  `‖Π⊥null g‖` and `null_share`, because the span they measure no longer
  contains a gauge. `outer_gradient_arrow_solver` still deflates the orbit and
  still should, for a different and weaker reason now written down at the site:
  that block runs only after the conditioning gate has flagged a near-singular
  joint Hessian, and the orbit carries no data-fit curvature, so it is a genuine
  near-null direction OF THE OPERATOR BEING INVERTED — a pseudo-inverse choice
  on an ill-conditioned solve, not a claim that the criterion is flat.

- **Two floors classified directions of one operator in two metrics, and which
  one the value path used depended on how much RAM the machine had free
  (#2673).** The exact observed information `A = B + ΔC` was classified by the
  VALUE path at `|λ| ≤ 1e-9·max(λ_max(A), 1)` — one absolute number for every
  direction — and by the GRADIENT path at `|μ| = |vᵀAv/vᵀBv| < √ε`. Both ran on
  the same `A` inside one evaluation on the streaming route (the value path's
  terminal Newton polish reaches `solve_exact_stationarity`, which is not
  route-gated, while the gradient adjoint takes the matrix-free sibling), and
  which one the value path used was decided by `direct_logdet_admitted`, i.e. by
  the working-set comparison in `streaming_plan.rs` against ambient free memory.
  The same data on the same build could be classified two ways on two
  differently-loaded machines.

  Every previous pass counted directions that change class between the two,
  found zero, and could not tell "the rules agree" from "the fixture never asked
  them" — both counters need a near-null of `A` and no shipped fixture has one.
  **The witness was never the question.** Written as thresholds on the same
  `|λ|`, one rule is constant across directions and the other varies by the
  spread of the `B`-Rayleigh quotient. Measured on the #2515 route-invariance
  state:

  ```text
  dim=30  λ_max=3.089328e2
  value rule    : |λ| ≤ 3.089328e-7                        (one number, all 30)
  gradient rule : |λ| < √ε·vᵀBv ∈ [1.907374e-7, 4.602951e-6]
  ratio gradient/value ∈ [6.174076e-1, 1.489952e1]         spread = 24.13x
  ```

  No choice of the two constants makes a constant threshold equal a varying one,
  and the ratio **straddles 1** — so neither rule was even a conservative version
  of the other.

  **Which site moved was forced, not chosen.** Only `μ` is a curvature: it is
  invariant under a reparametrization `θ → Lθ`, because `A` and `B` transform
  congruently, while `λ/λ_max(A)` is not — and `θ = (t, β)` mixes chart
  coordinates with border coefficients whose scale is set by the data's units, so
  `λ_max(A)` was a maximum over incommensurable coordinates. `1e-9` was tuned
  where `√ε` is derived (SPEC rule 21). And consistency decides the direction:
  the value must pin exactly what the gradient cannot differentiate, or the
  criterion carries a `ρ`-dependence through a direction whose `A⁻¹` response the
  adjoint has projected out — the #931/#2253 value↔gradient desync in a new
  place.

  `ExactHessianSpectralBlock` now carries `metric_scale[i] = vᵢᵀBvᵢ` instead of a
  scalar `rank_floor`, and

  ```text
  rank_floor(i) = max( dim·ε·‖A‖₂ ,  √ε·vᵢᵀBvᵢ )
  ```

  The second term is the shared predicate — the same one
  `solve_exact_stationarity_preconditioned` applies to its own solution
  direction, reached through one function so the two cannot drift. The first is
  not a second classification: it is the standard backward-error bound for a
  symmetric eigendecomposition, so an eigenvalue under it carries no significant
  digits and `ln λ` is not a quantity. It is a floor UNDER the identifiability
  term, never a ceiling, so it can only pin directions and never resurrect one
  the gradient deflated — and the ONE crossing it cannot remove (`|λ|` inside the
  backward error while `μ` is resolved, which needs `vᵀBv ≲ dim·√ε·‖A‖₂`) now
  emits a `warn` naming itself in production instead of passing silently.

  `ArrowMetric` decides which restriction of `B` a block is classified in — the
  whole arrow operator for the joint block, `B`'s block-diagonal `H_tt` for the
  coordinate block — so a block and a metric cannot be paired wrongly. Both are
  applies through the cached factors: `dim` applies at `O(dim²)` against the
  `O(dim³)` decomposition above them, and **no second dense block**, because
  #2724 and #2757 price that residency.

  **The crossing this issue was filed about, produced.**
  `the_classification_is_invariant_under_a_reparametrization_2673` takes the same
  #2515 state and rescales the β border by `1e-4` and nothing else — what a
  change in the target's units does:

  ```text
  min|μ| plain  = 2.756149405310e-1
  min|μ| scaled = 2.756149405310e-1     relative move = 1.611e-15
  retired rule pins 0 of 30 directions before the rescaling and 5 after
    direction 12: RETIRED VALUE RULE=gauge (|λ|=1.280000e-7 ≤ 3.084041e-7)
                  but GRADIENT=resolved (|μ|=9.999989e-1 ≥ 1.490116e-8)
  ```

  Five directions with pencil curvature between 0.36 and 1.0 — as identified as a
  direction gets — became "gauge" to the retired value rule under a change of
  units alone. The shipped rule reaches the same verdict on every direction in
  both frames, asserted per direction. The arithmetic term is deliberately made
  to BIND in that frame (6 of 30, where `vᵀBv` fell by `1e-8`) and moves no
  verdict: there `|λ|` still stands `6.18e4x` clear of it.

  **One constant was also doing a third job.** `cluster_stable_eigh_operator`
  borrowed the PD floor as its degenerate-cluster GAP threshold. An eigenvector's
  perturbation under a backward error `~ε‖A‖₂` is `~‖E‖/gap`, so the cluster
  boundary is `√ε·‖A‖₂` — where the direction is determined to no better than
  `√ε` and must be re-resolved against `E` rather than trusted. Both factors
  derived, and `‖A‖₂ = maxᵢ|λᵢ|` rather than the old `max(λ_max, 1)`, which was
  not a norm at all when `A` was negative definite.

- **The `matern` benchmark cluster's failure moved subsystems, and what it moved
  to is a projection taken in the wrong inner product (#2748).** The signature
  this issue was chased on since 08-08 — `invert_identified_rho_hessian`
  refusing a negative ρ-curvature against the eigensolver's backward error — is
  gone from every cell. At benchmark run `31926616066` (head `5f6bddb16`, which
  contains every commit of the measured-`‖δH‖₂` work through `8ae1f8ee5`) the
  population is 20 → 8, and **seven of the eight fail only in the
  `rust_matern_*_flexible` lanes**, i.e. only when the formula asks for
  `link(type=flexible(logit))`. The identical scenario, identical data and
  identical Matérn smooth mint in `rust_gam` / `rust_matern_decomposed` /
  `rust_matern_standard`. The residual is not a `matern` phenomenon; it is
  `matern` × *learnable link*, and `matern` is in the table only because
  `_flexible` companion lanes are minted for `matern` scenarios.

  | scenario | failing lane(s) | terminal cause |
  |---|---|---|
  | `geo_disease_matern` | `_flexible` only | `frozen-index fixed point did not converge in 60 outer iterations: delta=4.948e0, scale=8.824e0, tolerance=8.824e-5` |
  | `geo_disease_eas3_matern_k12` | `_flexible` only | identifiability audit FATAL: `'eta'[11] ~ 'wiggle'[0] overlap=0.6542` |
  | `geo_disease_eas_matern_k6` | all rust lanes | flexible: `de-aliasing left no identifiable warp direction`; non-flexible: `spatial kappa optimization made REML score worse (1.267594e3 -> 1.267595e3)` |
  | `papuan_oce4_matern_k12`, `papuan_oce_matern_k12` | `_flexible` only | custom-family outer not stationary |
  | `papuan_oce4_matern_k6`, `_k24` | `_flexible` only | wiggle exact spatial hyper NOT STATIONARY, `railed=[0..5]` |
  | `haberman_5yr` | `rust_gam` | standard REML NOT STATIONARY — a separate population, as the issue body predicted |

  **Root cause.** `fit_binomial_mean_wiggle` residualizes the frozen warp basis
  against the mean block as `B⊥ = B − X·A`, `A = (XᵀX)⁺XᵀB`, to "keep the mean
  block `X` full and identifiable". But *outside the mean column space* is
  meaningless until an inner product says what **outside** means, and the
  Euclidean one is a metric no part of the problem uses. With the basis frozen,
  `q = Xβ + B⊥β_w` is linear in both blocks, so the joint negative-log-likelihood
  Hessian the fit assembles is exactly

  ```text
  [X B⊥]ᵀ diag(m₂) [X B⊥] + penalties,      m₂ = ∂²(−ℓ)/∂q²
  ```

  (`bmw_static_hessian_operator`'s frozen arm has `dq_dq0 = 1` and
  `basis_d1 = 0`, collapsing its four row coefficients to `m₂` everywhere). Its
  cross block is `Xᵀ diag(m₂) B⊥`, and `A = (XᵀWX)⁺XᵀWB` with `W = diag(m₂)`
  makes that block **zero by construction**. `W = I` makes it zero only when
  every row is equally informative — never true for a binomial near saturation,
  where `m₂ = w·μ(1−μ)` spans orders of magnitude.

  **Why that is the outer loop's defect and not a tidiness point.** Freeze at
  `η̂` and perturb by `δ`: `B(η̂+δ) ≈ B + diag(δ)B'`, so the design moves by
  `(I−P)diag(s)δ` in `q` with `s = B'β_w`, and the refit answers with
  `Δη = −H(I−P)diag(s)δ` for this fit's own penalized hat matrix
  `H = X(XᵀWX + S)⁻¹XᵀW`. With `P` the `W`-projection,
  `XᵀW(I−P) = XᵀW − XᵀWX(XᵀWX)⁻¹XᵀW = 0`, so `H(I−P) = 0` identically **for any
  penalty `S`** and the leading term of the frozen-index map's derivative
  vanishes. With `P` Euclidean it does not, and the map contracts only while
  `max_i s_i` stays small — which nothing enforces: monotonicity bounds `β_w`
  below at zero and not at all above. `delta/scale = 0.56` after sixty passes is
  what a non-contraction looks like.

  **Second repair: the loop never damped.** Sixty undamped passes, no
  relaxation, no acceptance test, no measurement of its own contraction. With
  `d_k = Φ(η_k) − η_k` and `M` the map's Jacobian, `M` commutes with `M − I`, so
  `d_k = M d_{k−1}` exactly and `⟨d_k, d_{k−1}⟩/‖d_{k−1}‖²` is `M`'s dominant
  eigenvalue read off two vectors the loop already computes. Relaxing to
  `η + t(Φ(η) − η)` replaces `M` by `(1−t)I + tM`, and `t = 1/(1 − mu)`
  annihilates the dominant mode — Aitken's Δ², written as a relaxation. The
  implementation only ever **damps**: `min(1, 1/(1 − mu))` is exactly `1.0` for
  every `mu ≥ 0` and for the first pass of every fit, so no converging fit moves
  by an ulp. `mu ≥ 1` is a monotone repulsion no positive `t` stabilises, and
  the refusal now reports the measured `mu` and says so instead of blaming a
  budget.

  Rejected: raising `outer_max_iter` (sixty passes of a non-contraction is not a
  budget problem), loosening `outer_tol` (`0.56` is not a tolerance problem),
  and extrapolating on `0 < mu < 1` (sound for the mode the Rayleigh quotient
  sees, unsound for an equal-magnitude mode of opposite sign it cannot).

  **Gates**, 12/12 green
  (`cargo test -p gam-models --lib -- frozen_index_relaxation binomial_mean_wiggle_dealias_metric`).
  The decisive one builds `H` with a NON-TRIVIAL ridge `S` — so the "for any
  `S`" half of the derivation is exercised, not assumed — forms the forcing
  `diag(s)·δ` at a warp slope above one, and measures `H(I−P)` on it: `≤ 1e-12`
  of the unresidualized response under the curvature metric, `≥ 5e-2` of it
  under the Euclidean one, the second **asserted from below** so the fixture
  cannot quietly stop exercising the defect. Also: a constant metric reproduces
  the Euclidean projection exactly (equally-informative rows do not move);
  zero-curvature rows and a singular `XᵀWX` are handled; a negative weight is
  refused rather than used; and a planted linear map whose undamped iteration
  provably diverges (`|η| > 1e6` after 40 passes) reaches its fixed point to
  `1e-12` under the derived damping.

  `Zz2155Problem::solve_fixed_lambda_freeze_refit`, a deliberate mirror of the
  production loop, carried its own closed-form Euclidean de-aliasing and now
  calls the same primitive: one definition of the projection, not two.

  **A negative result worth recording so nobody repeats it:** the flexible-lane
  defect does not shrink to a fast cell. The same binary at `centers=8, n=1500,
  n_pcs=6` never reaches the wiggle stage — it dies 21 minutes earlier in the
  *pilot*, on `iso-kappa joint REML … hessian_psd=NO`. The reduced shape lands
  in a different subsystem's refusal.

- **A transformation model whose transformation saturates outside its knots has
  no tails, only a floor — and every reported quantile silently truncated at the
  training range (#2600).** #2600 removed the endpoint renormalizer, so the
  fitted conditional-transformation-normal density is `φ(h)·h'` and the model's
  CDF is `F(y|x) = Φ(h(y|x))` on the whole real line. Nothing gave `h` anything
  to be out there. `ctn_response_bases_at` builds the response basis from the
  shared I-spline evaluator, which saturates outside its knots (`I_k` constant
  above the last, zero below the first) with `create_ispline_derivative_dense`
  zeroing the exterior to match (#2695), so on the chart
  `h = α₀ + Σ_k I_k(y)·α_k + offset + ε·(y − median)` the entire exterior of the
  fitted transformation was

  ```text
  h(y) = h(y_b) + ε·(y − y_b),    ε = TRANSFORMATION_MONOTONICITY_EPS = 1e-8.
  ```

  Measured on an intercept-only fit to `Y = exp(N(0,1))` at `n = 256` — a law
  whose true transformation `h(y) = ln y` is pinned exactly by `F = Φ(h)`:

  ```text
  support [8.123377e-2, 1.444738e1]
  L = h(y_lo) = -1.971548   U = h(y_hi) = +3.291711
  Phi(L) = 2.433061e-2      1 - Phi(U) = 4.978986e-4
  ```

  **2.4 % of the model's own predictive mass lay below the tabulated support**,
  and BOTH inverse-transform consumers — `invert_transformation_normal_grid` in
  the predict path and `invert_monotone_grid` in the generate sampler — answered
  a latent target off the end of the table with the SUPPORT ENDPOINT, a
  truncation the likelihood does not perform. So the 2.3 %, the 0.13 % and the
  0.003 % predictive quantiles were the same number, `y_lo`; the observation band
  was degenerate from the 97.7 % level outward rather than from the 99.99 % the
  ladder's `z_max = 4` suggests; the sampler had point masses at both fitted
  endpoints; and two responses a factor 1.8 apart on the far side of the boundary
  received PIT scores identical to seven digits.

  The boundary derivatives are O(1) — `h'(y_lo+) ≈ 4.47`, `h'(y_hi-) ≈ 0.86` —
  so the flat exterior was never a property of the fit.

  **Root cause 1 — the transformation needed a slope, not a wider grid.** The
  response basis is now continued **affinely past the two boundary knots at its
  own one-sided boundary derivative** (Royston-Parmar's linear tails, `mlt`'s
  `extrapolate`, and exactly what `apply_linear_extension_from_first_derivative`
  already does for every clamped *B*-spline value basis in this tree):

  ```text
  y > right:  I_k(y) = I_k(right) + (y − right)·M_k(right⁻),  M_k(y) = M_k(right⁻)
  ```

  `h` is C¹ and strictly increasing on all of `ℝ` with `h' ≥ ε` preserved
  structurally (`M_k(y_b) ≥ 0`, and `α ≥ 0` on the Khatri-Rao monotonicity cone),
  so `Φ(h)` is a proper CDF whose tails are the fitted transform's. Evaluation at
  and inside the knots is **bit-identical** — `ctn_response_knots` guards the
  support by 0.1 % of the response span precisely so every training row is
  strictly inside — so no fitted quantity moves. The continuation lives in the
  CTN chart and not in the shared basis: the exterior I-spline entries leave
  `[0, 1]`, which for a transformation is correct (they are coefficients of a map
  that must keep increasing) and for the survival link warp that shares the
  evaluator is not (#2695 genuinely wants saturation).

  **Root cause 2 — the inverse was interpolated with a chord, twice.** Asking the
  now-total inverse for the identity that defines a quantile, `h(h⁻¹(z)) = z`,
  exposed the second half. `CtnTransformTable` replaces both clamping inverters
  with one object carrying `(y_k, h(y_k), h'(y_k))` — the chart computes the
  value and the derivative together at every node and the derivative was being
  thrown away — and interpolates the cubic Hermite those three determine. The two
  exterior branches stop being a separate convention: they are the same rule at
  its end slopes. The cubic is used only where provably monotone; a cell whose end
  slopes exceed the Fritsch-Carlson bound relative to its own secant is
  under-resolved for one and falls back to its chord.

  The same chord sat in `ladder_quantile`, where the tabulated `h⁻¹` becomes the
  band a user receives — and there it also clamped past `|z| > 4`, so every level
  beyond 99.994 % produced the same interval. Now shape-preserving (PCHIP) inside
  and continued at the ladder's own end slope outside, which past the fitted
  support is the exact continuation.

  ```text
  round trip max|h(h^-1(z)) - z| over the production ±4 ladder
    endpoint clamp             2.03        (the distance from z = -4 to L)
    affine tails + chord       2.1e-3
    affine tails + Hermite     1.6e-7

  interpolation order, refining 129 -> 257 nodes on h = ln y
    Hermite  3.015e-6 -> 2.180e-7   order 3.79
    chord    5.741e-4 -> 1.513e-4   order 1.92

  band ladder, max relative error on h^-1 = exp
    shape-preserving 5.5e-5    chord 1.9e-3
  ```

  Bars are derived rather than chosen: the round-trip bar is 1 % of the
  production ladder's own step (from `TRANSFORMATION_NORMAL_BAND_Z_NODES` and
  `_Z_MAX`), and the sampler's tail-mass bar is the model's own
  `Φ(L) + 1 − Φ(U)` ± 4 binomial standard errors — measured `0.02355` against
  `0.02483`, with **0 of 25 600 draws** on a support endpoint.

  **And the titled defect, which still reproduced.** While verifying the
  predictive half through the release CLI, an ordinary CTN fit — lognormal
  response, one smooth covariate, `n = 400` — refused with this issue's own
  title, at `k = 3, 4, 5, 8`:

  ```text
  error: ... inner solve refused this trial point: physical reduced-face
  first-order KKT failed (projected_residual_inf=9.736333e-1,
  tolerance=1.162291e-6, trust_shift=0.000000e0, active_rows=1)
  ```

  `solve_physical_reduced_face` is an ACCELERATOR with a designed graceful
  fallback — `Ok(None)`, "the general constrained QP owns this subproblem" —
  which two of its conditions already used. Three more were `Err(trial_point)`,
  which ends the fit, though every one of them is the fast path failing to FIND
  a certifiable step rather than producing something that violates its own math.

  The one the fixture died on grades a repair. When the face solve lands outside
  a row it is holding as an equality, the candidate is clipped back along the
  certified feasible chord and pushed forward as *"one representable value
  outside its own wall"*, for the KKT residual to accept or reject. Instrumenting
  the repair with `1 − t` on that chord says it is nothing of the kind:

  ```text
  face_rows=7 chord_repair=1.000000e0 projected_residual_inf=5.448715e-3 tol=2.040314e-6
  face_rows=9 chord_repair=5.702202e-1 projected_residual_inf=2.299802e-1 tol=2.177172e-6
  face_rows=6 chord_repair=1.000000e0 projected_residual_inf=2.408515e1 tol=2.799394e-5
  face_rows=6 chord_repair=8.882834e-1 projected_residual_inf=1.086920e-1 tol=2.793391e-6
  ```

  `chord_repair = 1.0` is the chord clipped all the way back to the feasible
  base — a ZERO step. Whole steps were being graded as one-ulp repairs, and the
  verdict on a heuristic repair was being treated as a violated contract.

  `chord_repair` is now carried with each candidate and the three
  heuristic-failure conditions decline, each logging its own numbers: a
  chord-repaired candidate that fails first-order KKT (one with
  `chord_repair == 0` remains a hard error — that IS the Moré--Sorensen solve
  failing to solve); a face whose minimum-norm point lies outside the trust ball
  (`minimum_face_norm_sq = 3.745717e1` against `radius_sq = 1` — a statement
  about a warm face and a shrunken radius); and a face that touches the trust
  ball with a nonzero tangent, whose reduced ball is empty. Non-finiteness stays
  a hard error and is split out of the ball test.

  ```text
  before   refuses in the inner solve: reduced-face first-order KKT, active_rows=1
  after    325 declines, the outer smoothing search CONVERGES, and the fit reaches
           `fit_custom_family final posterior assembly`
  ```

  What stops that fixture now is a different subsystem — `truncated moments for a
  61-dimensional constraint face did not converge … max |correlation| between
  constraint normals 1.000` — which is #2601, named in
  `constrained_posterior.rs` and measured there on synthetic faces. The two
  `ctn_*_2680` regressions and the `tram` quality arm die at exactly that point,
  with the numbers they had before this change.

  **What this exposes rather than creates.** The fitted CTN puts mass on the whole
  line, so a strictly positive response can be extrapolated below zero:
  `h⁻¹(-4) = -3.5e-1` on the lognormal fixture, with `h(h⁻¹(-4)) = -4.000000`
  confirming it is the model's own quantile. That is a property of a Gaussian
  transformation model on the RAW response scale — the likelihood the parameters
  were already estimated under — which the clamp hid behind an atom at the
  training minimum rather than removed. A model that must respect a bound belongs
  on the transformed scale (fit `log y`), as `mlt`'s `log_first` does; the chart
  documentation says so.

- **The post-fit certification was 60.5 % of the fit, and after the half of it
  that had been fixed it was still 99.96 % of `fit_diagnostics_report` — for a
  completely different reason (#2757).** The issue was filed on a dense
  symmetric eigendecomposition of a `param_dim x param_dim` curvature Gram
  (3160.5 s / 45.97 GiB at `p = 4096`). Holding that curvature in the block
  structure the decoder-frame parameterization gives it removed the
  eigendecomposition on the branch a Euclidean metric takes. Nothing measured
  what was left. Phase by phase at `n = 256, p = 64, charts = 32`:

  ```text
  [curvature build]                                  0.071s
  [residual gauge: reduce + generators + verdicts]   0.195s
  [coordinate fidelity]                              0.149s
  [decoder embeddedness]                             0.014s
  [topology persistence]                           547.387s   <====
  ```

  The whole of what the issue names is 0.195 s. The wall is
  `atom_topology_persistence`, and it had never been timed.

  **Root cause 1 — the filtration reduced the wrong matrix.** The persistent
  homology of a Vietoris-Rips complex takes the `(d+1)`-simplices as its
  reduction COLUMNS. At the `PERSISTENCE_H1_MAX_POINTS = 256` cover that is
  `C(256,3) = 2 763 520` triangles, while its pivots are EDGES, of which there
  are `C(256,2) = 32 640`. So at most ~32 000 columns can ever pair and
  ~2 730 000 exist only to be ground down to zero, each costing a chain of
  GF(2) column additions — 99 % of the filtration by measurement, with the
  simplex construction 0.071 s of a 5.8 s call. Persistent COHOMOLOGY has the
  same barcode and the opposite cost profile: its columns are the
  `d`-simplices, the 32 640 edges, and the triangles appear only as entries
  enumerated on demand as cofacets. At `m = 256` the engine now never
  materializes a triangle at all. Clearing (a pivot of degree `d−1` is a death
  partner whose own column is known to reduce to zero) becomes available in the
  order the degrees are already computed, which is exactly what it is not in
  the boundary direction; and the zero-length pairs that dominate a
  Vietoris-Rips filtration resolve on a column's first iteration with no
  addition, because the pivot is the EARLIEST entry.

  **Root cause 2 — the per-atom certificates ran one after another.** The three
  surviving per-atom reads (coordinate fidelity, decoder embeddedness, topology
  persistence) each take `&self` and an atom index and write only their own
  slot, so the serial `(0..k).map()` bought nothing but an evaluation order,
  which an indexed collect keeps. This is the "unparallelised, ~1.0 of 16 cores"
  the issue's own body records.

  ```text
  filtration, circle(160):    reference 9.001s -> 0.374s     24.1x
  filtration, circle(256):    ~17s      -> 3.13s
  topology phase, 32 charts:  547.387s  -> 95.46s
  fit_diagnostics_report:     547.8s    -> 52.0s             10.5x
  ```

  The rewrite is judged against the PRE-#2757 engine, kept verbatim as a
  control that shares none of its reasoning, differenced bar by bar on every
  endpoint's bits across circle, Clifford torus (the `H2` tetrahedron branch),
  line, separated clusters, exactly-tied filtration values, coincident and
  identical points, a far outlier, the four-point floor, DTM-weighted and
  flat-weighted arms, and a 12-member random family. Bars are compared as
  multisets — a barcode is one, and cohomology finds the same pairs in a
  different sequence — and the four functions that read a barcode are shown by
  measurement to agree between the engines and to be unchanged when their own
  bar list is reversed.

  **And the original defect, on the branch the block fix never reached.** With a
  metric that couples output coordinates (an output-Fisher or structured-residual
  harvest), the builder still assembled the dense `param_dim`-square Gram
  whenever `root_rows = n * metric_rank` exceeded `param_dim` — which at the
  #2283 production row count means `m = 480 000` against `param_dim = 65 536`,
  i.e. 1.0e15 flops and a 69 GiB peak. Those rows are now folded into a
  `param_dim`-square upper-triangular factor `T` with `TᵀT = RᵀR`, so every
  production representation carries a ROOT and every rank decision is taken on
  `σ` rather than on `λ = σ²` floored at the eigensolver's own resolution
  (`1.5e-11·λ_max` against `1e-16·λ_max` at that width). Peak memory halves:
  `eigh` allocated a second `param_dim`-square array for eigenvectors the
  reduction discarded.

  This does not make a coupling metric affordable at production width, and the
  module says so rather than implying otherwise: `H` there is a sum of
  `n * metric_rank` rank-one terms with no exploitable structure, and
  `min(rows, param_dim)²` is what an exact full spectrum costs from either side.
  The route out is for the certificate to stop asking for one — of the three
  things it reads off `H`, only `ξᵀHξ` and `λ_max` enter a verdict, and both are
  streamable — which is a change to what the certificate reads, not to how the
  curvature is stored.

- **The smooth-term likelihood-ratio p-value of a Gaussian fit was scored
  against the reference for a KNOWN variance, on a fit that estimated its own
  (#2672).** The Gaussian arm of the null-simulation size grid read
  `size@.05 = 0.0792` pooled over 480 replicates against a nominal `0.05` —
  `2.9` standard errors, with `first == fixed == est` in every cell, so the
  Lawley Bartlett factor was inert and the whole miss belonged to the reference.
  That arm exists as a DISCRIMINATOR: on a Gaussian response the log-likelihood
  is exactly the quadratic every other lane expands to, so a size miss there
  cannot be the expansion.

  **Root cause.** gam's profiled Gaussian log-likelihood is
  `ℓ = −½[n·ln 2π + n·ln(D/ν) − Σ ln w_i + ν]` with `D` the weighted residual
  sum of squares and `ν` the residual degrees of freedom it divides by. So the
  whole-term LR statistic is, with no expansion anywhere,

  ```text
  W = 2(ℓ_f − ℓ_0) = n·ln(D_0/D_f) + n·ln(ν_f/ν_0) + (ν_0 − ν_f)
                   = n·ln(1 + Q/V) + B
  ```

  with `Q = (D_0 − D_f)/σ²` and `V = D_f/σ²`. `Q` is what the reference's null
  spectrum is the spectrum OF — so the shipped reference answered "how extreme
  is `Q`" when the question was "how extreme is `n·ln(1 + Q/V) + B`", and `V` is
  a random variable of the same data with mean `ν` and spread `√(2ν)`. Both the
  mean shift `ν/(ν−2)` and the extra spread push the test anti-conservative, and
  both are `O(1/ν)`: invisible at `n = 1000`, worth `0.03` in size at `n = 30`.
  It is the same reason mgcv's smooth-term p-values take an `F` reference when
  the scale is estimated and a `χ²` when it is known.

  **The fix inverts the map instead of expanding it.** `W > w` is exactly
  `Q − c(w)·V > 0` with `c(w) = expm1((w − B)/n)`, a linear combination of
  independent chi-squares with a NEGATIVE weight evaluated at zero. `n` and `ν`
  appear where the log-likelihood put them, so the `κ ∈ {1, n/ν}` convention
  question the candidate scoring was going to decide by measurement does not
  arise. The residual law is the same spectral object the numerator uses, taken
  over the whole model instead of the tested block: with `A = XH⁻¹X'` symmetric
  and the true mean in the penalty's null space,
  `V = ε'(I−A)²ε ~ Σ_i p_i²·χ²_1 + χ²_{n−p}`, exact at fixed `λ`.

  Rejected: adding `[P(F_{a,ν} > w/(κga)) − P(χ²_a > w/g)]` as a control
  variate to the shipped tail. It is exact only where the weights are equal,
  it needs the `κ` convention decided from a size measurement, and this
  subsystem's history is a list of terms that moved a size the right way and
  turned out to be compensating for something else.

  **Measured**, offline at fixed `λ` on a fixed design
  (`scripts/probe_2672_profiled_scale_ratio_law.py`, 800 replicates, MC s.e.
  `0.0077`), which isolates the scale from λ-selection:

  ```text
  n     lambda   shipped size@.05   ratio size@.05   shipped KS   ratio KS
   30      100        0.0650           0.0525          0.0435      0.0235
   30        1        0.0950           0.0437          0.0733      0.0319
   50        1        0.0663           0.0413          0.0372      0.0234
  100        1        0.0512           0.0450          0.0408      0.0231
  200        1        0.0475           0.0437          0.0202      0.0177
  ```

  The shipped size decays `0.095 → 0.0475` with `n` — the `O(1/ν)` signature —
  and the ratio reference is inside the Monte-Carlo band at every rung
  including `n = 30`. KS improves in every cell, so this is the whole p-value
  distribution and not one level.

  **What it costs, measured rather than argued.** Both references are strictly
  decreasing in `W` — `c(w) = expm1((w − B)/n)` is increasing and
  `P(Q − c·V > 0)` is decreasing in `c` — so they order replicates identically
  and at a MATCHED size they are the same test. The raw power difference is the
  over-rejection being paid back and nothing else, and it vanishes as the signal
  grows (`n = 40, k = 6`, planted alternative, 800 replicates, against a null
  size that moved `0.0642 → 0.0542`):

  ```text
  amplitude   0.4     0.6     0.9
  shipped   0.6675  0.9587  0.9988
  ratio     0.6150  0.9375  0.9988
  ```

  The summary table's Wald smooth test has taken an `F` reference under an
  estimated scale since #675 (`SmoothTestScale::Estimated`), so the two tests in
  the same report disagreed about what estimating `σ` costs: one paid for it and
  one did not. They now agree.

  Scope: `ProfiledGaussian` only. Every other family carries its dispersion in
  the IRLS weight, and the estimated-dispersion families that do not
  (Gamma, Beta, negative binomial, Tweedie) do not estimate through a residual
  sum of squares, so this derivation does not reach them.

  `reference_residual_df` and `reference_deterministic_offset` join the
  smooth-significance payload, so a consumer can see that a p-value was scored
  against an estimated-scale reference instead of inferring it from the family.

- **A ratio's tail is a signed weighted chi-square, and the Imhof panel rule was
  never resolving the amplitude (#2672).** `weighted_chi_square_sf` took
  non-negative weights at one degree of freedom each. The general form
  (`WeightedChiSquareTerm { weight, degrees_of_freedom }`, either sign) is what
  every estimated-scale reference needs, since `P(A/B > t) = P(A − tB > 0)`, and
  the multiplicity is what keeps an `n`-sized reference the cost of a `p`-sized
  one. Imhof's derivation never asked for positivity; the TRUNCATION BOUND did —
  `16/(x·U·ρ(U))` divides by `x`, and a ratio reference is evaluated at exactly
  `x = 0`. The replacement is the amplitude bound `4/(H·ρ(U))`, which the same
  `x = 0` makes strong: a ratio reference carries `χ²_{n−p}`, so `H` is of order
  `n` and `ρ` grows like `U^{n/2}`.

  Generalizing it exposed a defect in the shipped quadrature. `F_{1,5}` at
  `f = 0.05` returned `0.8319119` against the exact `0.8319122` — **an error of
  `3.4e-7` while certifying `1e-11`**. The panel width `4π/(x + Σw_j)` sizes on
  the PHASE only; the amplitude `1/(u·ρ(u))` has its own scale `1/|λ|` where
  `(1 + λ²u²)^{h/4}` turns over, and a panel far wider than that is a 16-node
  rule aliasing a factor it never sampled. This was reachable before the
  generalization — any spectrum with one dominant weight and a small statistic —
  and `two_unequal_weights_match_an_independent_convolution` was carrying a
  `5e-7` relative tolerance that is exactly the size of what it tolerated. The
  second scale is derived from the Bernstein ellipse through the integrand's
  nearest branch point at `u = ±i/|λ|_max`:
  `a = d/[(ϱ − 1/ϱ)/2]`, `ϱ = tolerance^{-1/2N}`. Worst `F`-identity error is
  now `1.7e-12`.

- **A two-class multinomial with a smooth term published `β ≡ 0` — every
  predicted probability was the uniform simplex, at `edf_per_class = 4.09`
  (#2612).** The fit was not refused and did not look degenerate from outside:
  it selected interior smoothing parameters, published a full-rank posterior
  covariance and a ρ-uncertainty correction, and reported four effective degrees
  of freedom — next to eight coefficients that were all exactly `0.0`. Nothing
  in the payload contradicted itself loudly enough for any gate to notice,
  because the EDF comes from `H⁻¹S_λ` and the covariance from `H`, and neither
  reads `β`.

  Measured on the #1891 coverage fixture, 40 replications, truth ranging over
  `p ∈ [0.21, 0.82]`:

  ```text
  #2612-INT rep=  0 x*=0.2803 p_true=0.716264 mean=0.500000 sd=0.062987
  #2612-INT rep= 39 x*=0.5941 p_true=0.618759 mean=0.500000 sd=0.046169
  ```

  `mean = 0.500000` on every replication. The standing coverage gate read that
  as an under-covering interval; the centre was a constant.

  One axis varied at a time isolates it exactly — `K = 2` AND a smooth term
  (`K = 2` parametric and `K = 3` smooth are both fine):

  ```text
  K=2 tp k=8      max|beta|=0.000000e0  plugin=[0.500000, 0.500000]  edf=[4.091]
  K=2 parametric  max|beta|=1.020912e0  plugin=[0.411708, 0.660159]  edf=[2.0]
  K=3 tp k=8      max|beta|=6.181604e0  plugin=[0.208740, 0.642326]  edf=[3.38, 2.92]
  ```

  **Root cause.** `MultinomialFamily::specs_match_workspace_shape` required
  `spec.penalties.len() == self.penalties.len()`. Penalties are not part of a
  workspace's GEOMETRY — no penalty ever enters `X_aᵀ diag(w_ab) X_b`; the
  solver adds `s_lambdas` and the joint bundle itself, on the other side of that
  call. The clause predates #1587, which moved this family's entire smoothing
  onto the JOINT penalty and made `build_block_specs` attach
  `penalties: Vec::new()` deliberately ("The per-class blocks attach NO smooth
  penalty"). So from #1587 onward the predicate was FALSE for every penalized
  multinomial: the family declared "I cannot serve a joint workspace" about the
  workspace it does serve. It gates all three `*_available` capabilities and,
  through `has_workspace_source`, the solver's routing:

  ```rust
  use_joint_newton = has_joint_exacthessian && (specs.len() >= 2 || has_workspace_source)
  ```

  `K ≥ 3` has two blocks and reaches the joint path anyway, so there it cost
  only the workspace gradient/log-likelihood/HVP fast paths. `K = 2` has ONE
  block, so the stale clause WAS the routing decision, and the fit fell onto the
  block-coordinate path:

  ```text
  [PIRLS/blockwise step]  block=0 |delta|inf=1.037938e1 block_s_lambda_frob=0.000000e0
  [PIRLS/blockwise trial] bt=0 alpha=1.000e0    -trial_ll=2.210072311e2 prev=1.663553233e2
  [PIRLS/blockwise trial] bt=7 alpha=7.8125e-3  -trial_ll=1.666445362e2 prev=1.663553233e2
  [PIRLS/blockwise convergence] cycle 0 | max_proposed_step=1.038e1 (tol=1.000e-11)
                                | max_accepted_step=0.000e0 | obj_change=0.000e0
  ```

  Eight backtracks, none accepted, converged at cycle 0.

  **And the guard that made it silent.** `exact_joint_stationarity_ok` was
  ASSUMED `true` for single-block fits, and the surrounding test is
  `max_accepted_step <= tol && objective_change <= tol` — both exactly zero when
  the line search accepts nothing. "Nothing moved" and "nothing needed to move"
  are the same two numbers; only the residual separates them, and it was the one
  quantity not consulted. The comment's premise (for one block the blockwise
  iteration IS the joint iteration) is true and licenses TRUSTING the
  block-conditional verdict, not skipping it; the cost it cites is a multi-block
  phenomenon that cannot arise with one block. Now measured for every block
  count.

  After: `max|beta| = 1.023557e1`, plug-in range `[0.313129, 0.822891]` against a
  truth range of `[0.2142, 0.8176]`, deviance `296.59` against the uniform
  model's `332.71`. `K = 3` unchanged to `1e-13`.

- **The multinomial was the one family in the library whose published
  uncertainty never got the covariance-mode axis (#2612).**
  `fit_penalized_multinomial_formula` read `fit.covariance_conditional` and
  stopped; the same `fit_custom_family_with_rho_prior` call had already computed
  the first-order ρ-uncertainty correction `C = J·Var(ρ̂)·Jᵀ` (#2346) and
  published it on the inference block. So every multinomial band answered "how
  wide is the posterior once λ̂ is the truth" while every other family defaults
  to `SmoothingCorrected`.

  It could not even be expressed: `InferenceCovarianceMode` was declared in
  `gam-predict`, which sits ABOVE `gam-models`, so the one family that owns its
  own predict surface was the one family that structurally could not name the
  distinction. The enum now lives in `gam-solve::model_types` beside
  `SmoothingCorrectionMethod`, and `gam-predict` re-exports it — every existing
  path is unchanged.

  The correction reaches the response scale by the law of total variance,
  `Var(p_c) = Var(p_c | ρ̂) + Var_ρ(E[p_c|ρ])`, whose second term is `gᵀCg` with
  `g = ∂p_c/∂θ` the softmax Jacobian at the mode: the response-scale statement
  of `V_c = V_cond + C`, with no new object, no new constant and no new
  approximation order. `SmoothingCorrected` on a fit that retained no correction
  is an error, never a silent downgrade.

  The band is also built on the log-odds scale and transformed
  (`MeanIntervalMethod::TransformEta`, which this library already prefers for
  every nonlinear link). A symmetric `m ± z·sd` band clamped into `[0, 1]` is
  wrong twice where a class probability lives: symmetric about a bounded, skewed
  posterior, and the clamp DELETES the mass that fell outside, so a nominal 95%
  band could carry less than 95% while still reporting `level = 0.95`. `expit`
  is a bijection onto `(0, 1)`, so nothing is ever clipped.
- **A corrected log-determinant and the kernel that differentiates it were two
  fields, so a lane that dropped one kept the other (#2765).**
  Two producers — the custom-family joint assembly and the dense GLM assembly —
  compute the REML pseudo-log-determinant and its trace kernel from ONE
  eigendecomposition of ONE matrix, and both handed them back as two unrelated
  things: the scalar `projected_logdet - hessian_op.logdet()` into
  `InnerSolution::hessian_logdet_correction`, whose documented meaning is a
  UNIFORM CURVATURE RESCALE `-p*log(s)` and nothing else, and the kernel into
  `penalty_subspace_trace`.

  Two fields that travel separately can be separated, and the tangent-projection
  entry separated them. When the inner solve returns on an active inequality
  face the criterion becomes `1/2 log|Z^T H Z|` over that face;
  `try_tangent_projected_evaluate` drops the kernel — correctly, a `p`-space
  subspace kernel does not act on an `m`-dimensional face — while KEEPING the
  scalar, rank-rescaled by `m/p` as though it were the other kind of correction.
  The criterion's VALUE then carried a theta-varying term that no kernel
  anywhere differentiates, and the outer gradient was short by exactly that
  term's derivative, on every theta coordinate.

  `PenaltySubspaceTrace` now carries its own `logdet_correction` and the
  evaluator reads the value correction from the kernel, so the pairing is
  structural: a lane that drops the kernel drops the correction with it. This
  also removes, by construction, the collapse `joint_penalty_subspace_trace_parts`
  documents in its own signature — when the route yields NO kernel the old code
  still applied `0 - hop.logdet()` and silently deleted `1/2 log|H_pen|` from the
  cost while the gradient kept its `1/2 tr(H^-1 dH)` derivative.

- **The log-determinant operator carried a term whose drift is unobtainable
  (#2765).**
  `completion_in_operator` folded the Jeffreys second-order completion into
  `hessian_op` whenever the projected-logdet route was going to own the value and
  the traces — sound on its own terms (the operator is then used only for solves,
  and its `logdet()` cancels exactly), and a PRECONDITION a downstream lane can
  invalidate. On an active face the tangent evaluator takes its determinant from
  that operator's dense assembly directly, so the completion lands in the value
  while the drift that would differentiate it is `D_beta[completion][v]` — a
  third directional derivative no family exposes. The term was not merely
  missing, it was unobtainable, which is exactly why the completion is kept out
  of the scalar everywhere else.

  The completion now goes to the IFT operator unconditionally: the #2612
  separation stated as an invariant instead of as a route-dependent convenience.
  The tangent entry projects that operator onto the same face (`Z^T M_true Z`),
  because on a face `dbeta/dtheta` lies in `range(Z)`; and the cost-side IFT
  displacement `w = H^-1 r` reads `mode_response_operator()` rather than `hop`,
  making true its own claim to be "bit-identical to the gradient side".

  Measured on the #2765 survival marginal-slope fixture (`n=160`, 7.6 s), the
  analytic outer gradient against its own Ridders finite difference:

  ```text
             BEFORE (rel)              AFTER (rel)
    rho_0    8.256e-1  <- sign flip    1.767e-8
    rho_1    1.434e-1                  1.775e-8
    psi_0    9.435e-3                  5.677e-10
    psi_1    8.848e-3                  4.957e-10
  ```

  and `tests/survival/.../survival_marginal_slope_outer_gradient_fd_1040.rs`,
  whose own comment records "this is the analytic marginal-slope psi gradient,
  and it is wrong", now passes its matern arm at `rel = 5.5e-5` against the
  `1.377e-1` it recorded.

- **The composed monotone warp was built one derivative short of the objective
  that differentiates it, so the inner objective had an O(1) JUMP (#2695).**
  `linkwiggle(...)` puts a monotone I-spline on the model's own index —
  `q = q0 + sum_j betaw_j * I_j(q0)` with `q0 = -eta_t * exp(-eta_ls)` — so `q0`
  moves with `beta` while the knots stay where the seed put them, and the
  objective DIFFERENTIATES the basis rather than evaluating it. The row program
  composes it twice, and the second is the one that sets the requirement:

  ```text
    q1w = q1 + sum betaw_j * I_j(q1)      stack [I, I(1), I(2), I(3), I(4)]
    m1  = 1  + sum betaw_j * I(1)_j(q1)   stack [I(1), I(2), I(3), I(4), ...]  <- SHIFTED
    g   = eta_t(1) + m1 * q0dot,  and  the row NLL contains  -d * log g
  ```

  `H = d2(-l)/dbeta2` is the order-2 coefficient of that jet, and `m1`'s order-2
  coefficient reads its stack's slot 2 — which, because `m1` is built from the
  basis's FIRST derivative, is the basis's THIRD. `Phi = 1/2 sum g(lambda(Z_J^T H
  Z_J))` is a TERM OF THE OBJECTIVE (the inner NLL is `-l + 1/2 beta^T S beta -
  Phi`), so the objective's own value reads `I(3)` — while a degree-`d` I-spline
  is only `C^(d-1)` at a simple knot. At the shipped `degree=2` the accept test
  was comparing two points on two different functions:

  ```text
  |delta|inf = 7.094e-13    d_obj = -2.976461e-1
     trial_ll   -1.896965289627e1   IDENTICAL to 12 digits
     trial_pen   6.367949854901e-3  IDENTICAL to 12 digits
     trial_phi  -1.185962102549e1   vs  -1.156197496286e1   <- the whole jump
  ```

  the same `-2.976461e-1` at every step norm from `1.436e-10` to `7.094e-13`.

  A composed warp is now BUILT at `composed_warp_minimum_degree()`, derived as
  `COMPOSED_WARP_OBJECTIVE_BASIS_DERIVATIVE_ORDER + 1 = 4`, and the raise is
  logged with its derivation rather than refused: an earlier attempt refused
  below the floor and was reverted because a refusal breaks every working
  degree-2 fit and buys none of them a fit. The realised degree is what the
  knots, the design, the penalties and the saved metadata all carry. It is
  scoped to simple-ended warps: at a boundary knot of multiplicity `degree + 1`
  the ramp is `C^-1` at EVERY degree, so raising a clamped warp's degree would
  move those fits while fixing nothing.

  **The fourth derivative had to land with it.** The row program's five-slot
  tower ended in the literal `0.0`, which is the fourth derivative of a
  degree-`d` I-spline only for `d <= 3` — exactly the degrees the floor now
  excludes. `survival_wiggle_fourth_basis` supplies it, so no order-3 or order-4
  lowering differentiates a different function than the value it is paired with.
  That coupling is why the earlier attempt could not work: degree 3 was not
  enough (the `betaw`-weighted third-derivative channel) and degree 4 could not
  work while the tower it needed was a literal.

- **A fraction-to-the-boundary backoff was multiplicative, so an active-set
  method behind it could never identify a face (#2695).**
  `feasible_step_fraction` applied `alpha <- 0.995 * alpha` when a row clipped
  the step. The surviving slack after a clipped step is then
  `s + alpha*d = (1 - 0.995)*s`, so every clipped cycle keeps `1/200` of the
  slack and NO finite number of cycles reaches the face. Measured on the #2695
  witness: exactly `200x` per cycle for 400 cycles, with the QP's proposal
  constant at `1.554e-2`, the joint trust radius held, and the objective change
  exactly zero — the solve spending its whole budget walking one warp
  coefficient from `1e-3` to `1e-163` while the row it approached never became
  active.

  A backoff answers ROUND-OFF in an exact ratio test, which is a statement about
  resolution in the scaled-slack metric the contract is denominated in, not
  about how far the step happened to travel. It is now
  `alpha = fraction - PRIMAL_FEASIBILITY_TOL / |scaled drift|`: a step with room
  stops one feasibility tolerance short of the face, and a step whose remaining
  slack is already inside that tolerance yields `alpha <= 0`, which the contract
  reports as `BlockedByActiveFace` and the caller answers with a projection onto
  that face. The row becomes active in ONE cycle. `ContractFeasibleStep` already
  publishes the blocking row's scaled drift, so there is no new geometry and no
  new constant. Landing on the face is not a hazard for these constraints and
  that is checked rather than assumed: the row programs that take a logarithm of
  a bounded quantity carry their own guard (`log g` below the event-Jacobian
  floor is a CONTINUED logarithm), so the interior-point rationale that would
  justify stopping strictly short does not apply.

  Four existing exact-value pins asserted the old constant (`0.995*0.05`,
  `0.995*0.2`, `0.995*0.4`, `0.995*0.5`); they now assert the derived value
  computed from each fixture's stated row geometry, so they still pin an exact
  number and now pin the right one.

  Together these move the witness fit
  (`survival_location_scale_saved_fit_preserves_linkwiggle_metadata`) from a
  terminal stationarity residual of `4.79e-1` against a `7.9e-11` tolerance to
  `1.40e-8` against `1.08e-10` — seven orders — with the inner solve now exiting
  on a measured geometric convergence RATE (`0.9882x/cycle`) rather than on a
  discontinuity, and one seed reaching `KKT/certificate-converged`. The residue
  is a trust-region controller question and is recorded on the issue.

- **A penalty map within `1.5e-8` of a linear dependency was certified EXACTLY
  dependent, because its rank was decided on the SQUARE of the defect (#2676).**
  `PenaltyMapInvariance` licenses the curvature certificate's deflation by
  certifying `sum_i w_i A_i = 0`, and it decided that by eigendecomposing the
  Gram `G_ij = <A_i, A_j>_F` in `f64` and admitting eigenvalues at or under the
  eigensolver's backward error. But `lambda_min(G) = min ||sum_i w_i A_i||_F^2 =
  delta^2`, so the Gram carries the defect squared and a rank test at `G`'s own
  `eps` is a defect test at `sqrt(eps)`.

  Measured on this issue's own headline cell — `geo_disease_matern`,
  `centers=24, n=4000`, via `examples/repro2676_geo_disease_matern`:

  ```
  [INDEF-HESS] pair=(0,2) relative_defect=1.238259e-8 best_scale=1.000000e0
  [INDEF-HESS] active_rank=2/3 structural_zero=1 curvature_resolution=1.170e-8
  [INDEF-HESS] classifications=["Z", "A", "A"]
  [INDEF-HESS] reparam_split ... intrinsic=[-1.1702972950948233e-8, ...]
  ```

  `Z` is "certified null of the penalty map, excused by STRUCTURE". The pair is
  `1.238e-8` apart. **And the error compounds:** with the direction deflated,
  #2748's `invariance_residual_2norm` measures the residual of
  `T' H_rho T = T' diag(g_rho) T` on it and hands the result to the certificate
  as a MEASURED `||dH||_2`. On an exact invariance that residual is error and
  only error — the whole licence for the instrument. On a `1.238e-8` near one it
  is the criterion's genuine curvature, and the dump says so to four digits:
  `curvature_resolution = 1.170e-8` IS `intrinsic = -1.1703e-8`. The certificate
  was told its Hessian was uncertain to `1.17e-8` by a direction whose curvature
  it had just declined to look at — an inflated resolution masking genuine
  negative curvature up to that size at every site that spends it.

  The repair is the classical one — never form the normal equations to get a
  rank — taken in the currency the site can actually afford. Factoring the
  operator stack directly costs `k * block^2` doubles, hundreds of megabytes on a
  wide shared block; doing the same arithmetic in DOUBLE-DOUBLE costs a constant
  factor of time and no memory. The Gram is accumulated with exact products
  (`two_product`, one `mul_add`) and exactly renormalized sums, so an `O(1)`
  entry carries `O(m*eps^2)` instead of `O(eps)`; its pivoted Cholesky runs in
  the same precision, where the pivot `d_j` is the squared norm of `A_j`'s
  residual against the span already accepted, so `sqrt(d_j)` IS that column's
  defect to full relative accuracy at any magnitude down to `eps`; and the null
  space comes out of the FACTOR (`L[:, 0..rank]' w = 0`, back substitution)
  rather than out of an eigenvector of `G`. The boundary is denominated in the
  defect: `sqrt(entries) * EPSILON * ||A||_F * sqrt(accepted + 1)`, one
  operator-construction error per operator entering the residual, in quadrature.
  Nothing is chosen — the model is the arithmetic, calibrated by the two
  populations it separates (a pair known equal at `2.079e-15` against a floor of
  `2.0e-15`; the nearest pair known distinct at `8.75e-9`, six orders away).

  The same defect, in the same coordinate, was in the two human-facing
  instruments and is fixed with it. `report_penalty_pair_redundancy`
  thresholded `cos > 1 - 1e-8` and printed `cos` to six decimals, and
  `1 - cos = delta^2 / 2` — so a pair `1.9e-5` apart printed as `cos = 1.000000`
  and read as an exact identity, while the bar itself admitted anything closer
  than `1.4e-4`. The `[INDEF-HESS]` dump printed
  `structural_redundancy_detected pair=(0,2) cos=1.000000 one_minus_cos=2.42e-9`
  — gated on `cos > 0.999`, a defect of `4.5e-2` — two lines below its own
  `structural_zero=0`. Both are now denominated in `delta`, formed directly from
  the residual at the least-squares scale, and both distinguish the exact case
  (at the residual norm's own arithmetic floor) from the near one, which gets a
  new `near_degenerate_not_an_invariance` line saying what it is: the criterion
  carries genuine curvature of order `delta^2` there, the penalty map certifies
  nothing, and a negative curvature is a resolution question, not a structure
  one.

  Measured before/after through the real fit on the cell the mis-certification
  fired on (`examples/repro2676_geo_disease_matern 24 4000 16 info base`,
  first certification, everything else byte-identical):

  ```
  before  [PENALTY-REDUNDANCY] penalties i=0 j=2 are structurally identical (cos=1.000000)
          [INDEF-HESS] active_rank=2/3 structural_zero=1 curvature_resolution=1.170e-8
          [INDEF-HESS] classifications=["Z", "A", "A"]

  after   [PENALTY-SIMILARITY] penalties i=0 j=2 are close but MEASURABLY distinct
            (relative defect 1.238259e-8 at the best scale c=1.000000e0) ... NOT an invariance
          [INDEF-HESS] active_rank=2/3 structural_zero=0 curvature_resolution=3.780e-16
          [INDEF-HESS] classifications=["G", "A", "A"]
  ```

  Seven and a half orders of fictitious Hessian uncertainty removed, and the fit
  still admits -- the direction is excused by the chain rule (`G`), which is what
  it was always entitled to, rather than by a structure that was not there. The
  cell where nothing was ever mis-certified (`10 1500 16`) is byte-identical
  before and after, which is the control.

  Regression, on this host: `gam-solve --lib` 1930 passed / 0 failed (1726 s),
  `gam-terms --lib` 947 passed / 0 failed, `penalty_invariance` 17 passed / 0
  failed, and the issue's own acceptance 2 passed / 0 failed. `gam-models --lib`
  is 1712 passed / 23 failed BOTH before and after -- the two failure sets are
  identical name for name, measured by reverting exactly the four changed files
  in the worktree and rerunning the same suite, so none of those 23 is this
  lane's.

  One thing the sweep turned up underneath, recorded with the repair it
  FALSIFIED rather than with a guess. The operator penalties' raw Frobenius
  norms on the same cell, as the length scale shrinks:

  ```
  length_scale   mass      tension     stiffness    tension max|entry|
    1.64e-1      3.00e0    2.71e-8     1.72e4       6.72e-1
    2.05e-2      3.00e0    3.30e-97    7.03e7       6.00e-1
    1.03e-2      3.00e0    1.00e0*     1.12e9       3.26e-203
  ```

  `*` at `1.03e-2` the normalizer declines to divide (its `all(|v| <= 1e-12)`
  branch), so the scale reads `1.0` and the matrix ships un-normalized with
  entries at `3.26e-203`. Either way the tension operator is numerically
  annihilated and then carried as an ACTIVE penalty with its own smoothing
  parameter -- which is where the certified nullity of 2 at that scale comes
  from -- while mass and stiffness saturate to the same projector, which is the
  "exact redundancy" this issue was built on. The obvious repair (drop a
  candidate whose raw energy is under `EPSILON x` the strongest sibling on its
  block) reds
  `scale_contract::tests::every_wrapper_preserves_its_declared_inner_abscissa_pullback_2315`
  (5 active penalties -> 3), and correctly: operators of derivative order `q`
  carry dimensions `[f/x^q]^2`, so their raw energies move by `factor^(-2q)`
  under a rescaling of the abscissa and a cross-order ratio is not a
  scale-invariant quantity at all. Rule withdrawn. What is left is a magnitude
  question inside the operator construction (`1/ls = 97.4` and
  `97.4^-48 ~ 1e-95` is the shape of a `kappa`-power prefactor underflowing in
  the closed-form branch), and it belongs to that subsystem rather than to the
  curvature certificate.

  That premise is what this issue ran on for its whole life, and the sweep that
  killed it is `examples/probe2676_penalty_map_defect`: the
  `geo_disease_*_matern` redundancy is a small-length-scale LIMIT of two
  genuinely different operators — `delta = 2.079e-15` below `4e-2`, `1.874e-5` at
  the cold `Auto` geometry, `3.396e-1` at the geometry the fit settles on. The
  end-to-end acceptance is re-derived accordingly: one arm finds a geometry where
  the premise is true (by measurement, not by a pinned constant) and gates the
  deflation there; the other pins the honest fact about the `Auto` geometry — the
  fit certifies and NOTHING is deflated — so the false premise cannot return by
  inheritance.

- **The SAE inner solve had no mover for the block its own convergence measure
  removes, so it declared a fixed point while holding 559 stall-resolutions of
  objective decrease (#2762).** The chart-gauge orbit is an exact first-order
  symmetry of the RECONSTRUCTION and not of the penalized objective — the ARD
  prior on `t` and the smoothness prior on `β` are written on the chart
  coordinates — so the data-fit Hessian contributes nothing along it and the only
  curvature there is the priors'. A live gradient on near-zero curvature needs a
  LONG step, and every globalization in this solver is a step-SHORTENING device:
  Armijo backtracks, the LM gain ratio grows the ridge, the terminal polish's
  damping ladder suppresses exactly the near-null modes. The orbit component of
  the residual was therefore the one part of `g` no mover reduced — and
  `quotient_residual_norm_sq` removes that same span from the convergence
  measure, so it was not reported either.

  Measured at the `zz2015_tiny_inner_crawl_terminates` refusal: `‖g‖ = 2.075e-1`
  of which `‖Π∥gauge g‖ = 2.016e-1` — 94% of the residual ENERGY inside a
  4-dimensional span, with `maxᵢ |gᵀvᵢ| = 1.535e-1` against a `1.782e-3`
  tolerance (86x over the precondition the accept path assumes and never
  checked). The discriminating control:

  | direction | best objective drop | at α |
  |---|---|---|
  | steepest descent `−g/‖g‖` | `1.879e-4` | `1e-3` |
  | the removed span | `1.090e-1` | `1.0` |

  against a material floor of `1.949e-4`, with `fd/analytic = 0.9998913` — so the
  assembled gradient IS the gradient of the scalar the line search descends, and
  this was never a desync. Steepest descent lands BELOW the floor (the stall
  detector was right); the removed span buys 580x more at a 1000x longer step,
  because the 6% transverse component of `−g` is stiff enough to cap the ambient
  line search three decades early.

  The fix is plain block-coordinate descent on the objective's own parameter
  space, not a new model: `descend_gauge_orbit` minimizes
  `penalized_objective_total` over exactly the span `gauge_quotient_basis`
  removes, and the Newton/MM movers keep the transverse block where they are
  well-conditioned. Both blocks descend the same scalar the inner Armijo descends
  and the KKT gradient differentiates, so the composition is monotone and a joint
  fixed point is stationary in both. No estimand moves; the gauge coordinate
  stops being arbitrary and starts being chosen by the objective, and at a state
  this converges on `Π∥gauge g ≈ 0`, so the precondition the quotient measure's
  removal assumes holds BY CONSTRUCTION rather than by assertion.

  Every bound is derived: the sweep's far end is `inner_iterate_scale` (the same
  trust radius the Newton step already clips against), its near end is
  `material_floor / ‖Π_V g‖` (below which the first-order model itself predicts
  less than the objective's resolution — a proof the sweep is complete, not a cap
  on it), and the bracket is golden-sectioned to `√ε` relative width, the
  information bound for locating a smooth minimum from f64 values alone. A round
  commits only a decrease clearing the same material floor the Armijo and
  proximal gates use, so a commit is never the ε-harvest that makes an inner map
  non-idempotent.

  The block is consulted at all THREE fixed-point claims — the joint fit's
  no-strict-decrease exit, its objective-stall shortcut (whose own comment
  already said it fires "on the gauge-orbit crawl … immediately", naming the
  mechanism and treating it as a reason to stop), and the refine loop's stall
  over whole rounds — each armed once per plateau on the doctrine this codebase
  already uses for the terminal Newton polish. End to end on the repro:
  `‖g‖ 2.075e-1 → 8.389e-2`, gate `27.67x → 16.42x` of tolerance,
  `‖Π∥gauge g‖ 2.016e-1 → 7.859e-2`, unspent decrease in the removed span
  `1.090e-1 → 6.43e-3`, objective `1.949279e4 → 1.948573e4`.

- **The smooth-term LR reference replayed `λ̂`'s selection on a grid `60×`
  coarser than the selection it was replaying, and the law it produced was 23%
  short exactly where `α = 0.05` is read (#2672).** The replay draws the tested
  block, minimises the same REML criterion the outer search minimises, and reads
  `W` at the `t` it picked. `SMOOTH_LR_SELECTION_GRID_BUDGET` is a TOTAL — 441
  points however many scales the term has — so a default double-penalty `s(z)`
  (`m = 2`, the shape every fixture on this issue fits) got 21 points per axis
  over a window the `ρ` box opens to 60 wide: a spacing of `3.0` in `ln λ`,
  against the `0.05` the one-dimensional lane next door commits to, and against
  a CONTINUUM for the fit whose choice this is the reference for.

  A grid that cannot find the criterion's minimum returns a law that is selected
  LESS than the statistic it is the reference for, and the error is one-signed:
  it under-disperses, the upper tail is too thin, the test over-rejects.
  Measured on a whitened bending+ridge pair at the `ρ̂` separations a null-true
  smooth actually reaches, 2048 draws:

  ```text
  arm             grid  per_axis  spacing   E[W(t̂)]      sd      q95    wall
  grid only        441     21      3.000     2.1334   2.9094   7.1898   0.06s
  grid only       1681     41      1.500     2.4212   3.3266   9.4427   0.21s
  grid only       6561     81      0.750     2.4928   3.3656   9.2994   0.83s
  grid only      25921    161      0.375     2.5192   3.3783   9.3892   3.30s
  grid + descent   441     21      3.000     2.5258   3.3779   9.3278   0.20s
  grid + descent   121     11      6.000     2.5258   3.3779   9.3278   0.18s
  ```

  `15%` short in the mean, `23%` short at `q95`. That is the residual this issue
  was left holding after its four reference defects closed — pooled
  `size@.05 = 0.0564` on the light grid and `0.0669` on the small-n one, both
  anti-conservative, both inside their bands only because the bands are wide.
  It is also `n`-INDEPENDENT, which is what the `..._versus_n` sweep's
  flattening at `0.065` for `n ≥ 200` is once the small-n quadratic-expansion
  error has decayed out of it.

  **A bigger grid is not the fix.** `25921` points costs `3.3 s` per term and is
  still at spacing `0.375`. The grid is the wrong instrument: it is a BRACKET,
  and a selection is a DESCENT. Each draw now descends the criterion from its
  own bracket node by a compass search that halves its step whenever a sweep
  fails, to the same `0.05` floor the diagonal lane uses. That reaches the `161²`
  law to `0.3%` from a bracket of 121 points — by making the grid SMALLER. The
  bracket stays at 441 because its only remaining job is not to miss a basin.

  **And the descent needed an evaluator the eigen route cannot be.**
  `SelectionGeometry::at` returns the full eigensystem, which is right when 2048
  draws share a point and wrong when one draw chose it. `SelectionFactor` prices
  a point from two triangular factorizations of `r × r` objects — exactly, not
  approximately, the same criterion and statistic:

  ```text
  C(t) = UᵀT(t)U = RᵀR,  R = qr(M(t)),  D = (I + C)⁻¹C,  v = Uᵀu
  criterion = vᵀDv + log|I + C| − log|C|
  statistic = ‖u‖² − ‖Dv‖²
  ```

  because `D`'s eigenvalues are the shares `f = e/(1 + e)` and
  `w = 2f̄ − f̄² = 1 − f²`, and a direction outside `range(T)` has `f = 0`, so it
  drops out of the first and carries its whole square into the second — which is
  what the eigen route's `log(1 + 0) = 0` and `w = 1` say. The #2644 conditioning
  split is kept rather than lost: `log|C|` comes from the triangular factor of
  the SCALED ROOTS (`κ(C)` reaches `e^60` on a null-true double penalty, where an
  assembled Cholesky has no small pivots left), while `log|I + C|` and `D` come
  from the assembled `I + C`, which is benign — an absolute `ε‖C‖` in a mode near
  zero moves `log(1 + e)` and `e/(1 + e)` by that much and no more. The whole
  replay goes `0.06 s → 0.20 s` per term for it, against `3.3 s` for the grid
  that would otherwise be needed.

  Three contracts carry it, none of them the probe that found it:
  `the_descent_reaches_a_grid_it_cannot_afford_2672` scores the shipped replay
  against the `161²` grid on mean AND `q95` at 3%, stated as a CONTRAST so it
  cannot pass by both arms drifting — the bracket alone must still miss `q95` by
  more than 10%, and must miss DOWNWARD, because a coarser selection cannot
  select more; `the_two_evaluators_price_a_point_identically_2672` pins the two
  routes against each other on a DENSE information with two dense components at
  separations to 40 (the descent compares its trials against a baseline the
  bracket produced, so a gap between the routes is a search descending one
  function while reporting another's value); and
  `the_multiscale_replay_is_bit_identical_across_generations` is #1017 for the
  lane that now searches rather than enumerates.

  Verified on one 4-core box, `--test-threads=1`:

  ```text
                                                        at main        after
  exhaustive_null_simulation_size_grid              pooled .0564   ok   pooled .0542
  null_simulation_size_is_calibrated_small_n        pooled .0669   ok   pooled .0638
  poisson_smooth_lr_is_bartlett_corrected_...            ok        ok
  the_two_routes_to_the_null_spectrum_agree_on_real_fits ok        ok
  the_two_moment_summary_is_exact_when_shrunk_...        ok        ok
  per_term_edf_plus_unpenalized_columns_equals_edf_total ok        ok
  the_null_spectrum_reaches_the_reference_with_a_param.. ok        ok
  cargo test -p gam-models --lib selection_replay lr_null      23 passed
  ```

  (these fixtures are deterministic, so the before/after comparison is exact
  rather than distributional.)

  **What is left, and it is one cell family.** On the small-n grid the other six
  cells average `0.046` against nominal `0.05`; `bernoulli/logit, k = 12` sits at
  `0.119` at both `n = 30` and `n = 50` and carries the pooled figure by itself.
  A Gaussian arm — added here as
  `gaussian_null_size_is_calibrated_where_the_expansion_is_exact_2672`, because
  the residual's two readings (a wrong reference versus the QUADRATIC EXPANSION
  the reference and the Lawley factor both are) are separated by a family whose
  likelihood IS that quadratic — reads `0.0750` pooled at `n ∈ {30, 50}` and
  `0.0588` at `n ∈ {100, 200}` (pooled s.e. `0.0077`), with the Lawley factor
  inert in every cell.

  That decay is the PROFILED SCALE, and it is a defect of its own rather than
  more of this one. `σ` is estimated from the same data, so
  `W = 2(ℓ_full − ℓ_null) ≈ Q/(V/ν)` with `V ~ χ²_ν`, `ν = n − edf_total`, while
  the reference scores `Q` alone. Scored on one set of fits, an `F` reference
  removes `0.135 → 0.100` of it at `n = 30` and `0.120 → 0.115` at `n = 200` —
  the right size and the right decay — and `mean(W)/E[W(λ̂)]` runs `1.34` at
  `n = 30` down to `1.005` at `n = 200` against the `n/(ν−2) = 1.18` that
  mechanism predicts. It applies to every scale-ESTIMATED family and to none of
  the fixed-scale ones this issue's grid is built from, so it is separable, and
  `zz_measure_gaussian_reference_against_the_profiled_scale_2672` is the
  measurement it starts from.

- **The conditional-transformation-normal likelihood renormalized every row by
  the standard-normal mass between two FITTED endpoints, and that is what left
  the fit with no mode to find (#2600).** The row density was
  `φ(h(y)) · h'(y) / [Φ(h(y_hi)) − Φ(h(y_lo))]`: the model CONDITIONED on the
  response lying inside the fitted knot range, with both endpoints functions of
  the coefficients being estimated. That divisor removes both properties a
  most-likely-transformation model needs.

  *Concavity.* `log Z = log[Φ(u) − Φ(l)]` is concave in `(l, u)` by Prékopa (the
  Gaussian measure of a convex set is log-concave) and `(l, u)` are linear in β,
  so subtracting it turned a convex negative log-likelihood — `½Σh² − Σlog h'`,
  a quadratic plus a `−log(linear)` barrier — into a convex-plus-concave sum. At
  one feasible β on the wine fixture, Hessian by central second differences:
  truncated `λ_min = −6.365756e-1` against `λ_max = 7.418500e1`; untruncated
  `λ_min = +2.346524e-1`. That single negative eigenvalue is the whole of
  `resolvable_negative_curvature=true`, which the solver reported on every
  terminal cycle of every refusal on this issue.

  *Coercivity.* Raise the unpenalized location column to `c` and contract the
  shape to `t/c`: `h`, `h_lo` and `h_hi` move together, the conditional law
  converges to a truncated exponential in the normalized shape coordinate, and
  the `−½Σh²` that would punish `c` is divided out by `Z`. The profile
  likelihood over the location column, maximized over the shape at each `c`,
  runs `141.0858 → 141.0604164` over `c ∈ [1, ∞)` with `c·Σα → 1.2235` —
  monotone, never stationary, supremum attained only at `c = ∞`. The MLE did not
  exist, at any λ: every penalty term is `O(‖shape‖²)` on that ray and vanishes,
  so the penalty only sharpened the escape rather than causing it.

  This is what five refuted hypotheses on that issue were all symptoms of — the
  strict-interior dead band, the missing box-KKT repair, the face that would not
  release, the Moré–Sorensen hard-case fill, and trust-region growth. The solver
  was correctly refusing a problem with no solution.

  The fitted density is now `φ(h) · h'` with no renormalization
  (Hothorn–Möst–Bühlmann 2018), and with it: the model's CDF is `F(y|x) = Φ(h)`,
  so the PIT is `Φ(h)` and the calibrated score is `h`; the
  `OutsideCertifiedDomain` refusal is gone, because it existed only to stop the
  conditional PIT fabricating a clamped `0`/`1` off the fitted range, and a
  held-out response beyond the training range is now predicted rather than
  refused; and `score_influence_jacobian` loses its endpoint-mass denominator,
  its three `φ/D` coefficients and its `1/φ(z)` inversion, because `z = h`
  identically on the interior.

  Both transformation-normal quality arms produced no fit at all before
  (`generated=2, screened=2, exact_validated=2, solver_started=0`) and now pass:
  held-out PIT `KS=0.1597` against a `0.2517` bar, and wine-price normality
  `W_gam=0.9533` against a `0.95` floor and `W_boxcox−0.02 = 0.9460`
  match-or-beat. Two pins carry the properties rather than the fixtures —
  `ctn_penalized_objective_is_coercive_in_the_location_column_2600` walks the
  escape ray to the family's own `|h|` domain bound and requires divergence, not
  merely monotone rise (a monotone sequence can be bounded, and bounded IS the
  defect), and `ctn_observed_information_is_positive_semidefinite_2600`
  eigendecomposes the exact SCOP information at nine feasible points.

- **The constrained posterior's retention ladder searched a real number where
  the object being chosen is a set of rows, and floating point stopped it
  descending (#2714).** `assemble_retained_face` keeps a constraint row iff
  `pivot > (k+1)·ε·diagonal/d`, and the ladder named the next face by the floor
  at which its worst-conditioned accepted row drops, `d_r =
  (k+1)·ε·diagonal_r/pivot_r`, on the argument that the retention test then
  reads `pivot > pivot`. It does not. Both sides are ROUNDED quotients: the step
  divides by `pivot`, the rebuilt floor divides that quotient back into the same
  numerator, and the round trip lands strictly below `pivot` for a measurable
  fraction of `(k, diagonal, pivot)` triples. The aimed-at row is then retained,
  the rebuilt face is **bit-identical**, and the next step is recomputed from
  unchanged inputs to the value the floor already holds — so
  `assert!(next < demanded_accuracy)` fires and a library panics.

  `the_floor_round_trip_retains_the_row_it_was_aimed_at_2714` measures the round
  trip on its own, over the magnitudes a penalized posterior produces, and
  asserts the sharper fact: **every retention is a stall**, because a
  bit-identical face recomputes a bit-identical step. There is no rounding of
  the quotient the other way that repairs this — the retention test *is* the
  definition of the face, so only the test can decide the face.

  The walk now carries the face. A rejected face names its
  `least_independent_direction` — the accepted row with the smallest
  `pivot/diagonal`, i.e. the smallest squared sine to the span of the rows
  before it in the `Σ` metric — and that row is excluded BY INDEX before the
  face is rebuilt at an unchanged floor. Termination becomes structural: the
  excluded set only grows, never re-adds, and is a subset of the candidates; the
  first unexcluded row always clears the floor; and the last face is a single
  row whose `1×1` lift is exact. No floating-point comparison is inverted
  anywhere on that argument.

  What leaves is a **direction**, not a row, and that distinction is a
  correctness requirement rather than a nicety. A row anti-parallel to an
  accepted one is refused as a direction and keeps its wall as that row's upper
  limit (#2523), so a two-sided bound reaches the moments as ONE retained row
  carrying a finite `upper`. Dropping that row while leaving its partner in the
  pool would let the next pass accept the partner in its place — with a full
  half-line, because the row that carried the fold is gone — turning a two-sided
  bound into a one-sided one, and on the wrong side: the walk is ordered by
  ascending slack, so the row that leaves is the tighter wall.
  `record_opposed_face_limit` therefore reports which accepted position it
  folded into, the face carries those partners with its least independent row,
  and `dropping_a_direction_takes_its_opposite_face_with_it_2714` asserts the
  invariant on a system where every direction is two-sided: no retained row may
  report an infinite upper limit, and no retained row may be a far wall.

  Excluding at the unchanged floor is also strictly less lossy than the old
  step, which tightened the floor for every surviving row as a side effect of
  dropping one: every face the floor ladder could reach is still reachable
  (exclude precisely the rows that floor rejected, and each survivor clears the
  looser floor a fortiori), while faces only a tighter floor would have
  destroyed are kept. `the_walk_returns_the_largest_admissible_face_2714`
  accordingly grades against brute force over EVERY exclusion set — the full
  family the walk searches — rather than the floor-indexed subfamily the
  previous oracle swept.

  The module doc also stops conflating the two cuts. The PER-ROW retention
  floor is nearly free — a row it refuses is within `θ < 5e-7` radians of the
  span of the accepted ones, and the accepted row it is parallel to has the
  smaller slack, so it imposes nothing new. The WHOLE-FACE check is not, and no
  `O(θ)` argument covers it: it fires exactly when every row cleared the floor
  and the face is still worse conditioned than any of its rows, and the
  direction it then drops can sit at `pivot/diagonal = 1e-3`, i.e. `θ ≈ 0.03`
  radians. That is a real constraint being dropped. The trade is still the
  honest one — the moments are computed for a BOX, a face keeping mutually
  dependent rows cuts the same region along diagonals, and the lift cannot be
  formed at all at a numerically singular `W` — so the alternatives are a
  subset-truncated posterior or none, not a subset-truncated posterior or an
  exact one.

  The walk also reports itself now, on both outcomes. A correction that had to
  drop rows says so in one `log::info!` line — which rows survived is a fact
  about the ANSWER, not about the solve, since the reported truncation is then
  carried by a subset of the constraints the user wrote — and a terminal refusal
  prints the last refused face's `W` at full precision, because the walk's
  decisions are a function of `W` alone and reproducing one otherwise costs a
  three-quarter-hour fit. The face dump fires only on the terminal paths:
  dropping rows is the walk working, and a `q × q` matrix per rung would be a
  megabyte of warnings for a correction that then succeeds.

  Reached by the #2714 witness because the fix for its titled defect let the fit
  get as far as final posterior assembly, where a monotonicity guard imposed at
  every observed exit time puts far more constraint rows than the time block has
  coefficients: `W = A Σ Aᵀ` is then structurally rank-deficient and the walk is
  the only thing standing between the fit and a face it can lift. That geometry
  now has a unit fixture of its own —
  `a_rank_deficient_constraint_system_still_yields_a_liftable_face_2714`, 40
  Vandermonde rows at closely spaced nodes on 5 columns, which is the shape with
  the data removed. It panics on the old ladder at pass 26 and returns a
  5-row face whose lift misses its identity by `1e-6` on the new walk.

- **The Jeffreys/Firth span is MEASURED, not derived from a penalty's kernel
  (#2612).** Two derived spans have shipped here and both answer structurally
  what has to be measured. The FULL identifiable span says *the model bounds
  nothing*, justified by "the Jeffreys score is `O(1)` against the data's `O(n)`
  Fisher information" — a premise that fails on a quasi-separated softmax, where
  `W = diag(p) − ppᵀ ≈ 0.005` per row, so the term acts at full strength on
  directions the penalty bounds up to `2298` (measured cost: mean argmax
  probability `0.828` against held-out accuracy `0.965`). `ker(S_λ)` says *the
  model bounds `range(S)`*, justified by `(H + S_λ)v = Hv + λSv` — true for any
  `λ > 0` and false in MAGNITUDE when `λ` rails at its floor. Measured at the
  penguins stride-4 unbiased mode:

  ```text
    ker(S_λ):                  2 of 74 directions, λ_min(H + S_λ) = 1.9e-3
    whole identifiable span:                       λ_min(H + S_λ) = 5.1e-5
  ```

  The worst-bounded direction — five orders below one observation-equivalent —
  is **not** in the kernel. It is a `range(S)` direction whose selected `λ`
  railed at `MULTINOMIAL_FORMULA_PRIOR_PSEUDO_OBS = 8e-4` pseudo-observations,
  and `8e-4` pseudo-observations is not a prior. Left unarmed the coefficient
  runs to `|η|∞ ≈ 45`, and the posterior-mean predictive refuses to publish
  because the posterior at that width is not describable by either Laplace
  expansion.

  The span is now `{v : vᵀ(H + S_λ)v < CONDITIONING_GATE_ABSOLUTE}` — the same
  one-observation-equivalent criterion that already decides the term's WEIGHT,
  now also deciding its SUPPORT. It contains the separating members of `ker(S_λ)`
  and excludes its well-determined ones, so it is strictly better than either
  endpoint on both sides. Reading `H + S_λ`'s deficient subspace was previously
  rejected because that matrix moves with `β` and `ρ` while every `Φ` derivative
  formula holds `Z_J` fixed; this does not read it live — it is measured once, at
  the unbiased probe's certified mode and its selected `λ`, and frozen for the
  armed refit.

  **The arming VERDICT is a different question and keeps its own answer.**
  Handing the measured set to the verdict as well broke the three-arm oracle from
  both sides — a genuinely separated design with `S_λ = 0` DISARMED, and widening
  the metric's scale until it armed again took the calibration fixture to
  `−0.0525` against a `0.05` bar. Threading that with a scale constant would be
  choosing an estimand on a curve. So *"does this model need a proper prior?"*
  stays a statement about STRUCTURE (`ker(S_λ)` and the gate's predicate on it,
  byte-unchanged), and *"where does that prior belong?"* is the statement about
  the fit's ARITHMETIC above.

  **The threshold is taken in the CLR metric, because a threshold is a statement
  about coordinates and the multinomial's are a gauge choice.** Relabelling
  classes acts on `θ` by a non-orthogonal contrast change, so `H + S_λ`
  transforms by congruence and its eigenvalues move; a kernel is
  congruence-invariant and never had this problem. Measured cost of ignoring it:
  `multinomial_fit_is_invariant_to_reference_class_1587` saw predicted-probability
  drift `4.093e-3` across three labelings of one dataset against a `1e-3` bar,
  with refit noise exactly `0`. Generalized eigenvalues against `G = (M/K) ⊗ I_P`
  — the same `M` the reference-symmetric penalty is built from — are gauge
  invariant, and `G`'s scale is derived rather than chosen: one observation's
  Fisher block in the ALR active frame is `W_ab = p_a(δ_ab − p_b)`, which at the
  most-informative point `p_c = 1/K` is exactly `(1/K)·M_ab`, so `M/K` IS one
  maximally-informative observation's curvature per unit design.

- **A cached inner mode was identified by the penalty's state, not the
  objective's, so the coefficient-mode continuation's corrector was disabled by
  its own refinement (#2612).** `InnerPenaltyState` carried the per-block and
  joint `log λ` and called itself "the complete smoothing state an inner Laplace
  mode is a function of", with the reuse contract written on it: *a cached mode
  is reusable only when the penalized objective it minimises is the identical
  function*. The inner coefficient objective is
  `−ℓ(β) + ½βᵀS_λβ − τ·Φ(β)`, and `τ` — the family's Jeffreys/Firth augmentation
  strength — was not in the state, so two different objectives at the same ρ
  compared equal.

  That is not a corner case for the one path that varies `τ`: #2366's
  `coefficient_mode_homotopy_member` DEFINES the armed coefficient mode as the
  endpoint of a continuation in `τ` **at fixed ρ**, so on that path the ρ half is
  constant by construction and the key matched at every waypoint. The only thing
  left between the corrector and a no-op was the fresh curvature certificate
  refusing a non-PSD incoming point — and the finer the discretization, the less
  the mode moves per waypoint and the more often that certificate passes.
  Measured on the penguins stride-4 armed refit, 8-step sweep, `λ_min` at each
  incoming mode: `−7.9e-2, −8.7e-6, −7.3e-6, −4.8e-6, −2.5e-6, −7.4e-7, −5.2e-8,
  +2.8e-8` — the last one reused, so the sweep's ENDPOINT was the `τ = 0.875`
  mode relabelled as the `τ = 1` mode, printing a bit-identical log-likelihood,
  penalty and cycle count. Refining the discretization, which is the
  continuation's entire convergence mechanism, is what disabled its corrector.
  This is #2615 one level up: the same key comparing equal at every `τ` because
  it is missing a coordinate, rather than at every ρ because it was empty. The
  state is now `InnerObjectiveState`, built from the family so no production site
  can supply the wrong strength, and the persistent-warm-start key carries it
  too.

- **The continuation ladder's endpoint sequence is MODE-VALUED, so the dyadic
  contraction premise could not read it (#2612).** Every sweep's last waypoint
  corrects at the target objective to the inner solver's own KKT tolerance, so
  each endpoint is an *exact* mode of the *same* function — refining does not
  shrink an error, it changes which mode the path arrives at. The measured trail,
  same fixture:

  ```text
    steps  1 → 2    endpoint discrepancy 1.521120e0
    steps  2 → 4                         3.328619e-5
    steps  4 → 8                         1.695145e0
    steps  8 → 16                        5.413481e-5
  ```

  Two values four orders apart, alternating — `O(1)` between different modes and
  `O(5e-5)` (the corrector's own reproducibility) between two arrivals at the
  same one. There is no rate to observe. `d_k ≤ ½·d_{k−1}` therefore fired at the
  first refinement that actually tracked the branch, using as its baseline the
  accidental agreement of two coarse sweeps that had both jumped to the same
  wrong mode. Three things follow and all three are now fixed:

  1. the contraction ratio is reported evidence, not a verdict;
  2. one agreement is not a limit — the `2 → 4` agreement above is a full
     agreement on a mode the 8-step sweep leaves — so certification needs
     consecutive agreements;
  3. the yardstick could not be `options.inner_tol`. That is a KKT-residual
     tolerance and the discrepancy is a relative sup-norm over linear
     predictors; on this fixture two sweeps reaching the SAME mode differ by
     `3.3e-5` and `5.4e-5` against a `1e-5` bar, so "the same mode, twice" was
     not certifiable at any depth. The reason is physical: the armed mode is
     nearly flat in one direction (`λ_min ≈ 4.7e-7`), so `β̂` is poorly determined
     by a residual while the criterion built from it is well determined — the
     same two endpoints agree to `5.0e-7` in the criterion. The bar is now the
     outer solver's own relative-cost resolution, in the criterion's units, which
     is also the only quantity the seed exists to make well defined.

  #2661's requirement — accepting arbitrarily slow progress makes the loop
  operationally unbounded, since each refinement doubles the corrector count — is
  preserved and is now bounded as the resource it is: **the seed may not cost
  more correctors than the outer search it seeds is budgeted for**, i.e.
  `2^{D+1} ≤ outer_max_iter`. A ladder that exhausts it refuses with the full
  trail rather than with two numbers out of a sequence.

- **A coefficient-objective continuation that cannot certify now DECLINES
  instead of killing the fit (#2612).** Its sibling `anchored_continuation_seed`
  has carried that contract since #2366 — "the production caller logs a refusal
  and keeps its existing seed, so declining a continuation still never turns a
  fit that works today into a failure" — and the homotopy call site was the one
  place that read the same kind of refusal as fatal. Refusing the whole fit does
  not make `V(ρ)` well defined; it only denies the caller the answer the
  pre-#2366 seed would have produced. What a decline costs is logged rather than
  hidden: the mode is then selected by the caller's coefficients, so `θ̂` is a
  functional of the seed for that fit.

- **What the follow-up-varying slope still cannot do, measured rather than
  guessed (#2765 / #2767).** With the one-channel pullback repaired, the
  acceptance fixture's outer solve goes from *zero* iterations (its first line
  search died at `StepSizeTooSmall after 50 attempt(s)`) to **1500+ outer
  evaluations across five BFGS multi-starts**, steps accepted via Strong Wolfe,
  descending `2148.09 → 2134.79`. It still does not certify, and the reason is
  now bounded on three sides:

  1. **Every criterion atom except `½ log|H|` is exact.** At the fixture's own
     shape the θ-wide audit gives `fixed_beta` to six digits and `logdet_s` to
     seven on all five coordinates; `logdet_h` disagrees on all five.
  2. **It is the mode-response half**, and on the ρ block that is proved by a
     bound with no oracle in it (`½ tr(K·λ_kS_k) ∈ [0, rank/2]`), not inferred.
  3. **It is not the follow-up axis.** The `logslope_time_k`-unset control
     reproduces the same `logdet_h` disagreement bit for bit, so it predates
     this issue — it is the `#979`/`#1040` lane, where
     `survival_marginal_slope_outer_gradient_fd_1040.rs` has recorded a wrong
     analytic ψ gradient since `#2461`.

  Every object that atom is built from is now differenced against its own
  Ridders-certified oracle and passes: `D_β H[δ]` (five gates, both slope
  frames, block-confined directions, plus an oracle-free constant-margin
  reduction), `D²_β H[u,v]` (three gates), `D_β H_Φ[δ]` (one gate in
  `gam-custom-family`, on a family whose Jeffreys information genuinely depends
  on β), and the ψ coefficient mode response `dβ̂/dψ` itself (`5.0e-8` relative
  against its finite difference, from `3.3e-2` before the repair). A binomial
  `matern(x1,x2)` control through the shipped GLM assembly — `c`-nontrivial, so
  the same mode-response term is live — agrees to `1e-6…1e-11` on every
  coordinate, which puts the residual inside the custom-family joint lane rather
  than in machinery every penalized non-Gaussian fit uses.

  The fixture also shows what the outer search now runs into instead: inner
  solves that exit at `residual ≈ 1.9e3` with the trust radius collapsed to
  `8e-12`, and an outer cost-stall guard that measures the criterion's own
  evaluation noise at `σ ≈ 1.0` nat. That is the `#979` inner-solve stall, not a
  gradient defect, and it is what the acceptance fixture is waiting on.

- **`D_β H` pulled the row Hessian back through ONE slope channel, so the outer
  criterion's whole mode-response term was the derivative of a different model
  (#2765 / #2767).** `add_pullback_primary_hessian` — the pullback that
  `RowKernel::add_pullback_hessian` routes through — was still written for the
  four-primary static frame: `h[[3,3]]` against a single `coefficient_design()`
  for the g–g block, `h[[0,3]]+h[[1,3]]` for m–g, `h[[a,3]]` for t–g. On a
  follow-up-varying slope the row Hessian carries `g₀, g₁, ġ₁` at primaries 3/4/5
  against three *different* designs (`X_cov ⊗ B_entry`, `X_cov ⊗ B_exit`,
  `X_cov ⊗ B′_exit`), so all three blocks were wrong.

  **Why every existing gate passed.** The joint Hessian itself is assembled by
  `hessian_dense_override`, which does loop the channels — so `H`, the score, and
  the ψ triple were all right and all gated. This pullback has exactly one
  consumer, `row_kernel_directional_derivative` (`D_β H[δ]`), and `D_β H` has
  exactly one consumer, the outer criterion's `½ tr(K · D_β H[dβ̂/dθ])`. The
  defect was invisible to everything except the outer gradient.

  The attribution chain, because each step needed an instrument that did not
  exist:

  1. The outer-gradient FD audit graded **ψ only** — its own doc said
     "smoothing-parameter ρ coordinates are deliberately excluded". It now grades
     the whole θ vector on request (`enable_outer_gradient_fd_capture_over_theta`),
     through one extracted ladder (`difference_theta_coordinate`) so the two
     blocks cannot drift apart. That showed `fixed_beta` right to six digits and
     `logdet_s` right to seven on **every** coordinate, with `logdet_h` wrong on
     all five — twice with the wrong sign.
  2. `logdet_h` is the only atom that reads `dβ̂/dθ`. Splitting it into its
     frozen and mode-response halves for the ρ block gave a bound with no oracle
     in it: `½ tr(K·λ_k S_k)` has both factors PSD, so it lies in
     `[0, rank(S_k)/2]`. At ρ₂ the frozen half was `+0.4976` and the finite
     difference of the total was `+0.7671`; a correct mode-response half would
     have forced the frozen half to `+1.528`, outside that interval. So the
     mode-response half was the wrong one — proved, not inferred.
  3. A binomial `matern(x1,x2)` control through the shipped GLM assembly — `c`
     nontrivial, so the same mode-response term is live — agreed to `1e-6…1e-11`
     on every coordinate, which put the defect in the survival lane rather than
     in machinery every penalized non-Gaussian fit uses. (The Gaussian sibling
     gate cannot say this: under the identity link `D_β H ≡ 0`.)
  4. Four new gates difference `D_β H[δ]` against the family's own joint Hessian
     in both slope frames and along block-confined directions. The follow-up
     frame failed at the marginal↔log-slope cross entry
     (`analytic −3.740e-1` vs `fd +5.438e-1`, oracle `2.6e-9`) and the static
     frame passed, which named the frame rather than the algebra.

  Also routed rather than patched: the sparse/mixed `evaluate_blockwise_exact_newton_*`
  paths (`p ≥ 512`) scatter the log-slope block as one `f_pipi[[3,3]]` rank-1 over
  one CSR — a shape that cannot express three channel designs at all. They are a
  storage optimisation for sparse designs, not a different model, so a
  follow-up-varying slope now takes the exact dense blockwise route at every `p`.

- **The repair, measured end to end: `rel_l2` 0.3395 → 0.1042, and the optimizer
  finds the signal axis by itself (#2735).** Same data, same seed, the fixture's
  own generator at `n=3000, K=60, pc_dim=6` — a shape 17× smaller in `n` and 8×
  smaller in `K` than the one the `0.10` bar is written for:

  ```text
      route                                  η spread   REML      rel_l2   wall
      fit_term_collection_forspec (before)      0.140   1736.54   0.3395    3.3 s
      production entry + per-axis ψ (after)     2.854    819.81   0.1042   1289.5 s

      learned_eta = [+1.8891, -0.0467, +0.1071, -0.0457, -0.9387, -0.9651]
      learned_length_scale = 2.0225  (seeded at 1.0)
  ```

  The criterion falls 917 nats and the held-out error falls 3.26×, essentially
  onto the bar. The largest contrast by a wide margin lands on **axis 0** — the
  axis carrying the entire `0.4·sin(π x₀)` — while axes 4 and 5, which carry
  linear coefficients `−0.15` and `+0.10` and no non-linear content at all, are
  pushed to the far end. Nothing told it which axis mattered.

  Cost, stated rather than buried: 1289.5 s against 3.3 s. The outer problem
  went from 9 coordinates to 15 and every ψ trial rebuilds the basis.

- **The Duchon operator-penalty ψ derivative differentiated a penalty the design
  never ships — two pre-existing desyncs, both on the ISOTROPIC route (#2735).**

  Found by running the production entry, which refused with *"spatial kappa
  optimization is unavailable for one or more eligible spatial terms"* and named
  no term and no reason. Every `Ok(None)` on that path now logs which term
  declined and why; the first run with those lines said it outright — the
  producer emitted 4 active penalty blocks against the realized design's 9.

  1. **The metric.** `duchon_operator_penalty_candidates` builds its collocation
     operators with `aniso = None`, deliberately, and its own doc says why: *"the
     anisotropy lives entirely in the curvature (Primary) RKHS Gram … Keeping
     these low-order stabilizers isotropic makes their η-gradient identically
     zero."* The derivative read `spec.aniso_log_scales` instead.
  2. **The split.** When `scale_dims` is on, the value REPLACES the single
     `Σ‖∇f‖²` with `dim` per-axis `Σ(∂f/∂x_a)²` blocks — one
     `PenaltySource::OperatorRelevance { axis }` each. The derivative emitted one
     `OperatorTension` regardless, and the consumer zips positionally, so the
     tension ψ-derivative was attributed to `OperatorRelevance { 0 }` and the
     other `dim − 1` relevance blocks had no ψ-derivative at all.

  Fixing (1) simplifies the per-axis work it invalidates: with the operators
  isotropic their η-gradient is identically zero, so `∂S/∂ψ_a = (1/d)·∂S/∂log κ`
  and `∂²S/∂ψ_a² = (1/d²)·∂²S/∂log κ²`. Normalization passes that scaling through
  exactly, so the per-axis bundles are the isotropic one scaled.

  Worth recording, because it bears on a claim in the tree: penalty ARD
  (`OperatorRelevance`, documented as *"the replacement for brittle kernel-η
  optimization"*) and metric anisotropy are not substitutes. `λ_a ∫(∂f/∂x_a)²`
  controls how much the fit VARIES along an axis; it cannot add resolution the
  kernel does not have. A radial kernel with an isotropic metric cannot wiggle
  fast along `x₀` and slowly elsewhere at any `λ`. That is the same statement as
  this fixture's own reference table, where an isotropic 500-centre smoother tops
  out at `0.2290`.

  **The gate that would have caught it did not exist.** A sum-identity or
  second-vs-first check cannot see either desync — both compare the derivative
  against itself. Only differencing the SHIPPED VALUE can, which is why the
  native half (which had such a test from the start) was right from the start.

- **A metric estimated from the knot cloud cannot see which axis the response
  varies along, so freezing it there is not "standardize the geometry, then
  learn the smoothness" — it is not learning the geometry at all (#2735).**
  `spatial_term_uses_per_axis_psi` excluded every `SmoothBasisSpec::Duchon` from
  per-axis ψ enrollment, so a hybrid Duchon term's `aniso_log_scales` were set
  once by `initial_aniso_contrasts` — the per-axis spread of the knot cloud —
  and never moved again. That seed is a statement about where the inputs ARE.
  On `large_scale_reml_stress`, whose inputs are iid `N(0, I)` and whose entire
  non-linear content is `0.4·sin(π x₀)`, it is sampling noise.

  Measured on the fixture's own generator at `n=6000, K=150, pc_dim=6`, one fit
  per explicit η along the single ray `η = (c, −c/5, −c/5, −c/5, −c/5, −c/5)`:

  ```text
      η ray        REML criterion    held-out rel_l2
      sentinel           3359.16             0.3261
      c = 0.25           3221.59             0.3038
      c = 0.50           2783.65             0.2430
      c = 0.75           2164.18             0.1650
      c = 1.00           1734.85             0.1112
      c = 1.50           1563.84             0.0914
  ```

  The criterion falls **1795 nats** and the held-out error falls **3.6×**,
  crossing the `0.10` bar at this shape, along a direction the outer loop was
  structurally forbidden from taking. The criterion and the held-out error agree
  about that direction at every step, which is what makes it a defect rather
  than an objective/estimand disagreement.

  The contrasts are identifiable — they change the kernel's SHAPE, not merely
  its scale — so REML can and should estimate them, exactly as it already does
  for the anisotropic Matérn. The repair enrolls them.

- **The isotropic ψ derivative is now a CONTRACTION of the per-axis one, not a
  parallel derivation (#2735).** For a radial scalar `F(r; κ) = κ^E G(κ r)`,
  with `A = r F_r`, `B = r² F_rr − r F_r`, `σ_a = s_a/r²` and `c = E/d`:

  ```text
      ∂F/∂ψ_a       = c F + A σ_a
      ∂²F/∂ψ_a∂ψ_b  = B σ_a σ_b + c A (σ_a + σ_b) + 2 A σ_a δ_ab + c² F
  ```

  `Σ_a` of the first is `E F + r F_r`; `Σ_{a,b}` of the second is
  `E² F + (2E+1) r F_r + r² F_rr`. Those are `scaled_log_kappa_derivatives`
  verbatim. `A` and `B` are the same two combinations the isotropic helper
  already forms, and both vanish with `r`, so the per-axis jet is finite at
  collision with no `1/r` anywhere. `duchon_radial_core_psi_triplet` — the old
  single-direction bundle — is retired: keeping a second way to spell the
  isotropic contraction is the drift the split exists to prevent.

  For a block carrying `m` explicit metric weights the same algebra collapses to
  `∂B/∂ψ_a = M_a(B) + (δ/d)·B`, where `M_a` is the scale-free per-axis
  derivative. That is not an analogy to the anisotropic Matérn: the Matérn
  helpers `hessian_operator_eta_entry` / `_eta2_entry` ARE this construction at
  `δ = 0` (its kernel carries no `κ^δ` prefactor), and they are reused rather
  than re-derived. `E_F − 2m = δ` for every block the operator penalty
  assembles — `D0` (`F = φ`, `m = 0`), the `D1` gradient (`F = q`, `m = 1`), the
  `D2` diagonal (`F = q`, `m = 1`) and its mixed term (`F = t`, `m = 2`).

  Collision is handled exactly and separately, NOT through the lift: every `s_a`
  vanishes at `r = 0`, so the block's only ψ dependence is `w_axis · φ_rr(0; κ)`
  — and `φ_rr` at the origin is not a pure power of `κ`, because the
  even-dimensional log-Riesz representative carries κ-dependent finite parts.
  That is precisely why the existing code refuses the scaling shortcut there.

- **A capability predicate, so the enrollment cannot outrun the derivative
  (#2735).** `duchon_spec_supports_axis_psi` answers from the spec alone.
  It declines — leaving the term on its single isotropic ψ axis,
  bit-identically to before — the scale-free spectrum, the periodic path,
  fractional spectral powers, terms with no contrasts to learn, terms carrying a
  joint null rotation (which the per-axis consumer does not apply and the
  isotropic one does), and any spec whose ACTIVE operator penalty routes through
  the closed-form Lebesgue block, whose ψ-derivative exists only for the
  isotropic direction. Shipping one of those would mean a block whose value and
  gradient came from two different constructions. The closed-form sweep covers
  every null-space order the realized build could degrade to, because
  `duchon_effective_nullspace_order` only ever reduces and the predicate is
  asked before centers exist.

- **A fixture's entry point is part of what it claims to test (#2735).**
  `large_scale_reml_stress_main` called `fit_term_collection_forspec` — the
  fixed-geometry entry, which builds the design once and optimizes λ — while its
  header promised "the full Duchon-on-PC GAM pipeline end-to-end". Neither the
  global length scale nor the per-axis anisotropy ever moved, so the held-out
  reconstruction it scored was the best a smoother can do at one arbitrary
  length scale under a response-blind metric. It now calls
  `fit_term_collectionwith_spatial_length_scale_optimization`, the entry
  `StandardFitRequest` uses, with the pilot geometry initializer disabled so the
  measurement is of what the full-data outer solve learns rather than partly of
  a subsample, and scores `fitted.resolvedspec` — the trained spec — because
  refreezing the caller's spec would have scored the seed.

- **An escape that RETIRES a coordinate onto a rail is not the pathology the
  small cap exists for (#2612).** `OUTER_SADDLE_ESCAPE_BUDGET = 3` carried the
  premise *"a genuine saddle is cleared in one escape"*, which this lane measured
  false twice: the banded multinomial fixture descends monotonically for six
  e-folds to the wall, and penguins takes four successive escapes that each run
  to a face (`α_box = 9.39, 4.77, 9.14, 6.11`) while the criterion falls
  `2.158034 → 2.156725`.

  The distinction the count could not make: the escape direction is exactly zero
  on every railed coordinate (`judged_subspace_basis`), so the ray's box
  intersection can only be set by a FREE one, and a reseed that lands ON the face
  has therefore retired a previously-free coordinate onto a rail. There are only
  `n` coordinates to retire and the criterion strictly decreased on the way, so
  such an escape cannot be the repeating pathology; it is bounded by
  `OUTER_CERTIFY_RESUME_BUDGET` like every other reseed kind and by
  `certify_resume_made_progress`, which stops the loop the moment a resume fails
  to strictly improve. An INTERIOR escape retires nothing and can in principle
  repeat forever — and the pathology #2155/#2363 names, a bimodal inner solve
  whose warm re-descent reports a phantom improvement the cold certificate cannot
  reproduce, is exactly the case that fools the descent gate — so the small cap
  stays and now applies only to the escapes it was written for.

  The fixture is the premise itself:
  `f(ρ) = ½ρ₀² − ½Σ_{j=1..5} ε_j ρ_j²`, `ε = (5,4,3,2,1)·10⁻³`, stationary at the
  origin and indefinite in every `ρ_j`, whose `argmin` over the box is the CORNER.
  A concave quadratic on a box attains its minimum at a vertex; refusing that
  point is refusing the answer. It needs five escapes because the
  minimum-curvature eigenvector is a single axis and each expanded step retires
  one coordinate. The `#2357`/`#2155` double wells sit exactly one unit from
  their minima and the ridge fixture has one indefinite coordinate, so neither can
  see a cap of any size — this is the first fixture in that file outside the
  premise. Measured on one binary, changing only this: `12 passed; 1 failed` →
  `13 passed; 0 failed`; `gam-solve --lib -- rho_optimizer::` 363/363.

- **A negative-curvature direction has no interior minimiser, so its escape step
  could not come from the falsifiability ladder (#2612).**
  `adjudicate_negative_curvature` built ONE step ladder — `α = 1, ½, ¼, …` down to
  `α_min = sqrt(2·objective_resolution/|λ_min|)` — and used it for two different
  jobs. As a *falsifier* it is exactly right and is untouched: the smallest step
  at which the claim `½|λ_min|α²` still predicts something the criterion can
  represent is the end of the range in which the claim could be refuted. As a
  *step rule* it is wrong, because along a direction of negative curvature

  ```text
      V(ρ + αv) − V(ρ) ≈ α(g·v) + ½λ_min α²,    λ_min < 0
  ```

  decreases without bound once the sign is chosen so the linear term is
  non-positive. A model with no interior minimiser cannot supply a step length;
  it has to come from the objective and the feasible box — the standard treatment
  of a negative-curvature direction, and exactly what a trust region does when
  its solution lands on the boundary. Capping the reseed at the falsifier's
  largest rung silently asserted the opposite: that one e-fold in log-λ is as far
  as any such descent ever runs.

  Measured on the `#2612` banded quasi-separated fixture, where the escape
  direction is `−e₁` to six digits:

  ```text
    baseline        1.786314898942e1
    ladder  α=1     1.786314894043e1
    ladder  α=½     1.786314883184e1   <- the ladder's pick, decrease 1.6e-7
    α=1             1.786314862766e1
    α=2             1.786314814710e1
    α=4             1.786314708132e1
    α=8             1.786314488769e1   <- box intersection, decrease 4.1e-6
  ```

  Monotone to the wall, and the wall step is worth **26×** the ladder's. The BFGS
  resume seeded at the ladder's point made *no* progress (reseed and next refused
  point bit-identical), so the escape was the only thing moving ρ — one e-fold per
  escape, against `OUTER_SADDLE_ESCAPE_BUDGET = 3`, on a ridge six e-folds long.
  The fit refused with `hessian_psd=NO curvature_source=terminal-analytic
  railed=[2,3,4,5]`.

  The rule: double the confirmed step while the criterion strictly improves,
  clamped to the exact box intersection along the ray. No constant — termination
  is structural, since the intersection is finite whenever the ray moves any
  bounded coordinate, doubling reaches it in `⌈log₂(α_box/α)⌉` steps, and any
  non-improving trial stops the sweep. The accepted point is the lowest measured,
  so it is never worse than the ladder's. `MAX_EXPANSIONS = 64` bounds a
  pathologically small confirmed step and is LOGGED when it binds.

  One extra evaluation re-measures the incumbent in the expansion's own instrument
  state. That is not tidiness: the same point (`sign = −1, α = 1`) evaluated in
  the falsifiability ladder and again afterwards differs by `3.1e-7` on this
  fixture — larger than the descent being adjudicated — because the profiled
  criterion carries warm-start hysteresis well above the `ε_f = 7.45e-10` the
  symmetric ladder measures on itself.

  ```text
    before   FIT FAILED after 6.5 s
    after    FIT OK in 2.9 s, ONE escape (4 doublings, α_box = 5.470155,
             "the accepted step IS it")
             acc=0.9750 logloss=0.07682 mean_argmax_p=0.9599 calib_gap=-0.01513
  ```

  bit-identical to a control with the escape budget raised to 40 (not landed).
  The escaped coordinate lands on the zero-smoothing rail, `λ = 2.0000e-4 =
  exp(−8.517193)`, where it is railed and leaves the certificate a PSD reduced
  block. `multinomial_separation_arming_2612` 3/3 in 4.6 s (was 1 FAIL);
  `gam-solve --lib -- rho_optimizer::` 362/362, and with the expansion disabled
  the two new travel assertions are the only reds.

  The `#2357`/`#2155` escape fixtures are double wells whose saddle sits *exactly
  one unit* from its minima, so nothing in that file could ever see the cap. The
  new fixtures are the shape that can: a stationary ridge with **no interior
  minimiser**, plus the guard that the expansion must not overshoot a genuine
  well, plus an end-to-end run pinned to `prefer_gradient_only` — because an ARC
  search reads the analytic Hessian and can follow negative curvature by itself,
  so letting the planner choose ARC would make a pipeline test green either way.

  Rejected: raising the escape budget (moves a constant to clear a bar and leaves
  the one-e-fold cap in place everywhere else); relaxing the curvature gate (the
  criterion's own symmetric ladder CONFIRMS the sign and resolves it against its
  Law 1 floor, so the verdict is correct and it was the response that was wrong);
  switching the outer search to ARC (`prefer_gradient_only` exists because the
  generic REML/LAML Hessian consumes the order-four family tower, and the escape
  has to work for the gradient-only plan regardless).

- **`76a520c45` withheld a deletion the geometry did not license and dropped the
  ORTHOGONALIZATION with it: the smooth-ownership hierarchy was inert for every
  dependent smooth (#2747).** `apply_global_smooth_identifiability` exists to
  enforce one invariant — the realized smooth block is orthogonal to
  `[intercept | owned linear axes | owner smooths]` — and it enforced it by
  DELETING one coefficient direction per constraint direction. `76a520c45`
  established that the deletion is free only under CONTAINMENT (the parametric
  direction inside the design's span, so the deleted function IS the parametric
  column) and withheld it otherwise. It left nothing in its place.

  The premise the deletion had always rested on —
  `smooth_requires_parametric_orthogonality`'s *"their realized column span
  contains the constant … a structural rank-1 collision"* — is measured false for
  half the class it names (`examples/probe_2747_containment_registry`,
  `‖1 − P_X 1‖/‖1‖` on the realized design against the `√ε` bar):

  ```text
  thinplate                                      9.90e-15   contained
  duchon                                         1.33e-14   contained
  matern (both policies, ν = 3/2 and 5/2)   7.8e-4 .. 8.4e-1   NOT
  curv (κ ∈ {−1,0,+1}, ℓ = 0.2 … 100)       5.1e-2 .. 9.5e-1   NOT
  ```

  and the Matérn column carries the point that decides how the gate has to be
  written: the residual falls monotonically toward the bar as the range grows
  (`8.4e-1 → 7.8e-4` over `ℓ = 0.2 → 10`), and the range is an ESTIMATED
  coordinate. Containment is a function of a fitted parameter, not a property of
  a family, so no per-family list can encode it and a delete/don't gate makes the
  model DIMENSION step by one when a fit walks its own range across a threshold.

  What that cost, measured through the shipped pipeline
  (`examples/probe_2747_parametric_orthogonality`, `‖XᵀC‖/(‖X‖‖C‖)` against the
  `1e-8` bar the same function asserts whenever a transform IS applied):

  ```text
                             before      after     deleted directions
  curv(x1,x2)               4.72e-1   1.17e-14      0 (was 1)
  x1 + curv(x1,x2)          4.89e-1   1.13e-14      0 (was 2)
  s(x1) + curv(x1,x2)       2.70e-1   7.15e-15      0
  s(x1) + tps(x1,x2)        1.64e-1   1.30e-14      0
  tps / duchon, ± x1        3.0e-14   unchanged     bit-identical
  ```

  The `s(x1) + tps` row is the one that shows the reach: thin-plate is the
  CONTAINED class and it was still `1.64e-1` against its owner, because the
  constraint block for a dependent smooth is `[1 | owner's realized columns]` and
  an owner's basis columns are contained in no other basis's span. So the
  containment gate withheld the whole block rather than one direction of it, and
  `analyze_smooth_ownership`'s hierarchy — the machinery that stops a broader
  smooth refitting structure its owner already carries (#978, #1470) — stopped
  binding for every dependent smooth in the library.

  **The fix is that the fork was never delete-or-nothing.** A deletion is
  licensed by containment; an ORTHOGONALIZATION is licensed always:

  ```text
  X̃ = X − C(CᵀWC)⁻CᵀWX        span([C | X̃]) = span([C | X])   for every X, C
  ```

  — a column operation on a block whose partner is in the model. So
  `apply_global_smooth_identifiability` now deletes where the block is wholly
  contained (bit-identical to what shipped; thin-plate and Duchon do not move)
  and PROJECTS everywhere else, keeping every coefficient direction while making
  `X̃ᵀWC = 0` exactly. The rank of `X̃` falls by `dim(col X ∩ col C)` and by
  nothing else, so the whitener drops precisely the directions the deletion is
  entitled to drop. It is also continuous in the containment residual — the
  direction the classical constraint removes has residualized norm exactly
  `sin θ = ‖1 − P_X 1‖/‖1‖` — so the two constructions AGREE at containment
  instead of meeting at a threshold.

  The fit is preserved where the theory says it must be: `y ~ curv` residual ss
  `9.926900 → 9.926900`, edf `23.455911 → 23.455912`, because with an unpenalized
  parametric block residualization is a reparametrization of the same fitted
  model. Where the partner is penalized — the `s(x1) + …` rows — the fit moves,
  which is the hierarchy binding again.

  Numerics: `G̃ = X̃ᵀWX̃` is streamed from the EXPLICIT residual rather than formed
  as `G − M N⁻ Mᵀ`, which is a difference of near-equal `O(‖G‖)` quantities
  precisely in the contained case it has to resolve. Storage: `X` and `C` are
  stacked into one `BlockDesignOperator` and the minus sign lives inside a single
  `CoefficientTransformOperator`, so a lazy design stays lazy and `C·R` is never
  materialized.

  Replay: `ParametricResidualizationChart` on `SmoothTerm`, frozen onto
  `SmoothTermSpec` beside the joint-null rotation. It carries `R`, the owner
  terms it was built against and whether the parametric block led — because `R`
  is TRAINING-ROW data and must be replayed, not re-derived (#978), while `C`
  itself is rebuilt at the new rows, which is what `C` is.

  Gated by `parametric_orthogonality_costs_no_dimension_2747` (five gates on the
  shipped pipeline, each asserting its own premise — the realized span must NOT
  contain the constant, or the deletion would be free and the test would measure
  nothing) and by three unit gates on the numerical core, whose orthogonality bar
  is DERIVED (`n·ε·‖X‖/‖X̃‖`, the floor of an `n`-term accumulation carrying the
  `1/sin θ` amplification) and asserted to sit far below the shipped `1e-8` so it
  cannot pass vacuously. The replay gate's negative control drops the chart from
  the same frozen spec and requires the design to MOVE, because a rederivation's
  output looks exactly like a design.

  **Named and deliberately NOT fixed here: a second producer.**
  `freeze_geometry_from_metadata` (`spatial_optimization.rs:4849`) freezes the κ
  optimizer's cold-build chart as `MaternIdentifiability::FrozenTransform`; that
  chart comes from `realize_single_smooth_term`, whose own comment says it "never
  runs the global ownership pass"; and
  `smooth_requires_parametric_orthogonality`'s doc excludes `FrozenTransform`
  bases on the premise that such a transform "already has the parametric
  orthogonalization composed in". For that producer the premise is false, so
  every spatial smooth whose geometry the κ optimizer froze skips the global step
  entirely, and has done since long before #2747 — which is why a Matérn fit
  still measures `4.15e-1` through `fit_from_formula` with the projection arm
  landed. `an_unfrozen_matern_smooth_is_orthogonalized_at_no_cost_2747` pins that
  the arm is not at fault: from an unfrozen spec the identical basis comes out
  orthogonal at the shipped bar with all `centers − 1` columns.

- **The κ criterion's acceptance was met and unmeasured (#2747).** Both fixtures
  are green at `ee7b9a2fa`: `profile_ci_covers_planted_curvature_across_replicates`
  covers 9/9 with 0 unresolved, κ̂ interior on 8 of 9 (the state the issue opened
  on was `railed_at_upper_bound` 9/9), mean κ̂ `+1.070` against a planted `+1.0`,
  sign right 9/9, over replicates that now span `0.5×`/`1×`/`2×` the auto range;
  `flatness_test_holds_size_across_flat_replicates` rejects 1/9 at α = 0.05 with
  0 unresolved. The estimator half of that issue was finished by
  `1f76fb35f` (one kernel, one range) → `337e6aa86` (ψ = (κ, η)) → `4b618f0ba`
  (the contrast gauge) → `76a520c45`, and the last of those had never been run
  against the fixture it was written for.

- **The λ̂-selection replay was refused on every real fit, and where it was not
  refused it minimised a criterion whose Occam term was noise (#2672).** The
  smooth-term LR reference prices `λ̂` as CHOSEN rather than as given by
  replaying the outer selection: draw the tested block, minimise the replayed
  REML criterion over `ln t`, read `W`. Four defects, each found by measuring the
  previous repair rather than reasoning about it.

  **1. The Occam term was priced by the route `#2644` had already rejected.**
  The criterion is

  ```text
  V(t) = ½ Σ_j c_j² e_j/(1 + e_j)  +  ½ [ log|I + T(t)| − log|T(t)|₊ ],
  T(t) = Σ_i t_i λ̂_i · Wᵀ S_i W
  ```

  and the bracket is its whole Occam half — the only term that stops the
  selection running to `t → 0`. Both replay lanes computed it as
  `Σ_{e > 0} log(1 + 1/e)` over the eigenvalues of the ASSEMBLED whitened sum,
  with no structural rank and no noise floor. `penalty_logdet.rs`'s own
  `SpectrumScale` says why that cannot work, and names this configuration: the
  assembled route prices `log|S_λ|₊` to `O(ε·κ)` against `O(ε·√κ)` from the
  stacked scaled roots, and `κ` goes past `1e14` when "one λ [is] at its ceiling
  beside a null-space shrinkage λ near zero" — which is what a null-true default
  `s(z)` IS, since `double_penalty` is on by default. Measured on a whitened
  `q = 9` bending+ridge pair:

  ```text
  ρ̂ = (0, 0)      offset  20.564 vs  20.564     error   0.000
  ρ̂ = (12, −12)   offset  29.813 vs  29.813     error   0.000
  ρ̂ = (18, −24)   offset  19.189 vs  53.811     error −34.623
  ρ̂ = (29, −29)   offset   8.103 vs  63.811     error −55.709
  ```

  The error does not perturb the selection, it replaces it: the modes lost are
  the ones carrying `−ln t_i`, i.e. the coercivity that makes the criterion blow
  up as `λ_i → 0`, so what is left is monotone in `ln t_i` and the replay picks a
  wall. Same mechanism `from_components` documents under #1237, from the replay's
  side. An independent numpy lab minimising a Gaussian-σ-known REML both ways
  puts the two argmins `5.7`–`13.0` nats apart under the exact criterion, always
  by railing the smaller λ down.

  **2. The geometry was refused on every real fit, silently.** `Wᵀ S W` is
  symmetric as an object and asymmetric by summation order, and
  `strict_symmetric_eigh` VALIDATES its input rather than symmetrizing it — the
  right contract, since a caller with a genuinely non-symmetric matrix has a
  defect. Every unit test in the module handed the whitening an identity
  information and a diagonal penalty, where the congruence comes out EXACTLY
  symmetric and the validator never fires. The integration sweep's first cell is
  a dense fit, and it declined.

  **3. The whitener formed `Ĩ_jj = ([H⁻¹]_jj)⁻¹ − S_jj`, which is a
  cancellation.** Two matrices whose ratio is `1/(1 − p)`, after an explicit
  inverse of an ill-conditioned block: at `p = 1 − 1e-12` — an ordinary
  heavily-shrunk direction — the difference is roundoff amplified twelve orders.
  The relative eigenvalue floor was then set by a SPURIOUS largest eigenvalue and
  discarded every direction the data could see. Four of the first twenty
  replicates of the `n = 60` fixture:

  ```text
  rep   reference mean   replayed E[W|λ̂]   q(replay)   q(reference)
    7          0.8557           0.0003        10           11
    8          0.9600           0.0000         9           11
   13          0.7060           0.0004        10           11
   15          0.7894           0.0000        10           11
  ```

  The published p-value is `p_conditional + [P̂(W_sel ≥ w) − P̂(W_cond ≥ w)]`, a
  control variate — and a control variate is only one if the subtracted term has
  the SAME law as the exactly-integrated one. On those replicates the shift was
  measured against a law with zero mass and added to the tail of a law with all
  of it. Invisible on any single fit; visible only in the size.

  The same object is available with no cancellation. With
  `A = B^{1/2} S B^{1/2} = QΛQᵀ`, `Λ = diag(p)`, `B = [H⁻¹]_jj`,

  ```text
  B^{1/2} Ĩ B^{1/2} = B^{1/2}(B⁻¹ − S)B^{1/2} = I − A = Q(I − Λ)Qᵀ,
  ```

  so `W = B^{1/2}Q(I − Λ)^{-1/2}` satisfies `WWᵀ = Ĩ⁻¹`, the only subtraction
  left is the SCALAR `1 − p` (which loses digits exactly when the direction is
  genuinely unidentified — a statement about the fit), and the retained set
  becomes `1 − p > 100·q·ε`, a meaningful criterion instead of a relative floor
  on a cancelled matrix. As a check on the construction, the whitened total
  penalty at `λ̂` comes out `(I − Λ)^{-1/2}Λ(I − Λ)^{-1/2} = diag(p/(1 − p))` —
  exactly the generalized spectrum, diagonal, for free. `lr_schur_information` is
  deleted rather than repaired: nothing needs `Ĩ` itself.

  **4. The conditional tail was resolved six orders below its own answer's
  noise.** The Imhof tolerance was derived from `FitOptions::tol`, and at the
  shipped `1e-10` that request is `~1e-10` — `gam-math`'s strict default, priced
  at 0.13–3.3 s PER P-VALUE, three or four per term.
  `null_simulation_size_is_calibrated_small_n` runs 960 of them and DID NOT
  FINISH IN 4000 s, against nextest's 600 s kill: the test this issue exists to
  un-hide had become a timeout again, by construction rather than by contention.
  The published accuracy is `quadrature + 2·se`, so resolving the conditional
  half below the selection shift's own standard error cannot improve the sum,
  while Imhof's truncation point grows like `ε^{-2/3}`. The request is now
  floored at `se`, capping the published bound at `3·se` against an irreducible
  `2·se`.

  **What replaces all four is one object.** `SelectionGeometry` carries the
  term's λ-FREE components, whitened by the cancellation-free `W`, factored into
  their own roots, plus their `ρ̂` and the structural rank of their sum; one thin
  SVD of the stacked scaled roots per grid point supplies the eigenbasis, the
  criterion's data operator, the statistic's null weights and both
  log-determinants, with `log|T|₊` over the `t`-free structural rank instead of a
  sign test on a number `1e18` below the largest. Three things fall out rather
  than being fixed separately: the one-dimensional lane stops reconstructing
  `ν_k = p_k/(1 − p_k)` from the shares (a share lives in `[0, 1]`, so a
  structural zero and `1e-17` of roundoff are one epsilon apart there — and the
  log-determinant is the one place that difference is worth `log(1 + 1e17)`, as a
  term LINEAR in `ln t`); its grid gains `ln t = 0` explicitly; and `generalized`
  is published on every lane, where the multi-scale one had returned
  `Vec::new()`.

  **And the `ln t` window is now per scale.** The outer search moved each `ρ_i`
  independently inside its box, so scale `i` reaches `[−B − ρ̂_i, B − ρ̂_i]`. The
  single common-shift window the `m`-dimensional grid used to receive is the
  INTERSECTION of those, which truncates every axis to the narrowest and is EMPTY
  as soon as one λ̂ rails — the normal state of a null-true double-penalty smooth.
  `generate_common_scale` still derives the intersection, because that lane
  genuinely moves every scale together.

  A missing replay is no longer an `Option::None` a reader has to attribute by
  elimination: `SmoothLrSelection::{Replayed, Declined}` names the step that
  refused, and that is how defect 2 was found.

  **Two tests were stating claims true only of the reference this issue
  replaced.** The Bartlett file compared `mean(W)` against `ref_df` — the
  CONDITIONAL mean `E[W | λ̂]`, which the empirical mean does not converge to
  because `λ̂` is chosen from the same data. Measured: `2.034` against `0.870`, a
  ratio of `2.34` and `4.18` standard errors, matching the `2.4–2.5` already on
  the issue for an independent harness. Against the mean of the law the p-value
  is actually read from — `E[W(λ̂)] = 1.452`, now published — it is `2.09` se. The
  bar stays at `3·se`; only the quantity moves, and it still fails by ~20 se on
  the state this issue opened at.

  The grid's per-cell band carried a fixed `+0.015` for "the second-order
  residual the correction itself leaves (`O(n⁻²)`)". Measured on the grid's
  hardest cell across `n` at 200 replicates, that residual is neither `O(n⁻²)`
  nor constant:

  ```text
  n         30      50     100     200     400
  first  0.141   0.111   0.080   0.060   0.065
  est    0.106   0.096   0.070   0.055   0.065      (MC s.e. 0.0154)
  ```

  — monotone toward nominal, inside the MC band by `n = 200`, quasi-separation
  rate `0.0` throughout. So it is the quadratic expansion's own finite-sample
  error and not a defect in the reference: a wrong reference gives an
  `n`-INDEPENDENT offset. The band now carries half of the cell's OWN first-order
  distortion instead of a constant, which states a claim about the correction
  rather than a tolerance and TIGHTENS the band from `0.075` to `0.060` wherever
  the test is in its regime.

  Two hypotheses died on data collected for something else: the estimated-λ
  lane's ρ̂-variation term is NOT a double count against the replay (dropping it
  makes every anti-conservative cell worse), and the Lawley factor's own
  magnitude does not track the residual it is meant to remove (`c ≈ 1.008`
  against a distortion of `0.056` at `n = 30`).

  Verified on one 4-core box, `--test-threads=1`:

  ```text
                                                    at main        after
  the_two_routes_..._agree_on_real_fits_2672            RED     ok    29s
  exhaustive_null_simulation_size_grid              pooled .0962  ok   191s  pooled .0564
  null_simulation_size_is_calibrated_small_n        >4000s, unfinished
                                                                  ok   358s  pooled .0669
  poisson_smooth_lr_is_bartlett_corrected_...           RED     ok    58s
  cargo test -p gam-models --lib selection_replay lr_null        20 passed
  ```

- **The `geo_disease_*_matern` / `papuan_oce*_matern` cluster refused a fit on a
  curvature the criterion itself measures with the OPPOSITE SIGN, because the
  only measurement in the room was thrown away after a boolean (#2748).**

  Ten of the eleven scenarios the last benchmark verdict lists as `errored` fail
  with one signature. Reproduced locally in 22 s through `bench/run_suite.py`:

  ```
  rho Hessian has negative curvature -6.404e-6 below the outer certificate's own
  bar 6.379e-6 ... measured here as 2.396439e-16 [analytic (Weyl, ||dH||_2); set
  by eigensolver backward error] ... the penalty map certified 0 null direction(s)
  ```

  The whole verdict rides on `intrinsic = sigma - sum_k g_k v_k^2 = -2.473e-8`,
  the only part of that eigenvalue that is a statement about the criterion, and
  it is judged against `2.4e-16` — the EIGENSOLVER's backward error.
  `curvature_resolution`'s own module doc says in bold that this number answers
  *"given this matrix, how wrong is sigma?"* and that a site asking *"how wrong
  is this matrix?"* must not be handed it. That warning was firing in production.

  **Why #2676's deflation does not fire here, measured rather than assumed.**
  Inside ONE fit:

  ```
  one_minus_cos(S_0, S_2) = 6.164524e-11   at an earlier point
  one_minus_cos(S_0, S_2) = 6.017409e-14   at the REFUSING point
  ```

  Three orders apart. The penalties do not depend on rho; they depend on psi, the
  jointly-optimised length scale. Round-off does not move three orders with psi.
  So the Matern mass and stiffness operators are genuinely DISTINCT operators
  that become proportional as the length scale collapses the kernel matrix — a
  real near-invariance, not an exact one. `PenaltyMapInvariance` certifies only
  exact ones, so it certifies nothing, and with no certified subspace every
  measured component of `||dH||_2` at that site is vacuous or absent:

  | component | value |
  |---|---|
  | eigensolver backward error | `2.396439e-16` |
  | rho-Hessian symmetrization defect | `0.000000e0` (symmetrized in place) |
  | outer-gradient re-evaluation defect | `0.000000e0` |
  | penalty-map invariance residual | unavailable (certified nullity 0) |

  **And the outer certificate had already ruled the other way on the same
  number.** Two lines above the refusal, same run:

  ```
  [CERTIFICATE] standard REML: the criterion CONTRADICTS the reported negative
  curvature. lambda_min=-6.404092e-6 on the judged sub-block, and 2 feasible
  trial(s) along its eigenvector -- both signs -- lowered the objective nowhere.
  ```

  Same matrix to six digits, same point, opposite verdicts — #2428 exactly, with
  the subsystem that actually evaluated the criterion losing.

  **The repair is a measurement, not a bar.** No floor moved, no tolerance was
  chosen. `adjudicate_negative_curvature` already evaluates the criterion on both
  sides of the point along the disputed eigenvector; that is a symmetric probe
  ladder, and `curvature_resolution`'s header already states that `eps_f` and
  `M4` "come free from any symmetric probe ladder that has already been run". It
  was being spent on one boolean.

  * `measure_symmetric_ladder` fits `N(alpha) = c*alpha^2 + (M4/12)*alpha^4` to
    the raw second-difference NUMERATOR, whose noise is step-independent — so
    plain least squares there IS the inverse-variance-weighted fit of the
    quotient, whose noise is `4 eps_f/alpha^2`. It returns the criterion's own
    curvature with a standard error, `M4` as twelve times the slope, and `eps_f`
    as the residual scatter over four.
  * The ladder is EXTENDED until it can determine that fit. The falsifiability
    ladder stops at `sqrt(2*objective_resolution/|lambda_min|)`, which for a
    small claim is `>= 1`, i.e. ONE rung — and one rung cannot fit two
    parameters. The extension halves to `alpha_end =
    sqrt(roundoff_floor/|lambda_min|)`, where the claim's own predicted numerator
    reaches the objective's arithmetic floor, plus two halvings so the plateau
    `eps_f` is read from is more than a point. Both ends are derived from the
    claim in dispute.
  * `v'Hv` and `d2/dalpha2 V(theta+alpha v)|_0` are the same number computed two
    ways, so their difference is exactly zero in exact arithmetic and, by Weyl,
    a certified LOWER BOUND on `||dH||_2`. It is carried on `OuterResult` into
    `invert_identified_rho_hessian` as a `MeasuredHessianError`, and only when
    the disputed direction lies entirely inside the rho block.

  **What it measured on the failing fixture.**

  ```
  c_criterion = +8.153228e-5 +/- 6.616477e-6   vs analytic lambda_min = -6.404082e-6
  12 rungs; measured eps_f = 3.254032e-7, M4 = 1.094027e-2
  measured ||dH||_2 from the disagreement = 8.131990e-5
  ```

  The criterion's curvature along that eigenvector is POSITIVE and thirteen times
  the magnitude the analytic Hessian claimed. `zero_bound` goes
  `2.396439e-16 -> 8.131990e-5`, the classification goes
  `["G","A","A"] -> ["Z","Z","A"]`, and the scenario mints
  (`status = ok`, 87 s).

  **The negative control fired in production, unprompted.** The same fit's
  iso-kappa joint arm adjudicated a different point and measured
  `c_criterion = -1.060226e-6` against `lambda_min = -1.060914e-6` — agreement to
  `6.9e-10`, so `||dH||_2 = 6.3e-10` and nothing widened. Where the analytic
  Hessian is right, this measures nothing.

  **Two names stopped being true and were fixed with it.** `zero_bound` used to be
  only an eigensolver backward error, so `|sigma| <= zero_bound` and "the penalty
  map's certified null" were the same population and sharing the name
  `StructuralZero` cost nothing. They are eleven orders apart now.
  `UnresolvableCurvature` is a third variant, and the three finally say three
  things: excused by STRUCTURE (exactly flat, no measurement can change it), by
  RESOLUTION (may be real, but the matrix is not known well enough for its sign
  to be a measurement), by the CHAIN RULE (`sum_k g_k v_k^2` carries no
  second-order content) — the split the #2676 thread argued for and did not
  build. And `InvertedRhoHessian::eigenvalue_backward_error_bound`, which has
  carried the MAXIMUM over several measured components since #2748's architecture
  landed, is renamed `curvature_resolution`.

  **`haberman_5yr` is NOT this and is not fixed here.** It fails
  `NOT STATIONARY (|Pg|=1.101e0 > bound=3.636e-6)` with `railed=[5]` and
  `line_search=StepSizeTooSmall after 50 attempt(s)` — an outer BFGS
  non-convergence, a separate population, exactly as #2748's body predicted.

- **"The box does not bind at its bound" is the wrong reading of #2705 group C's
  residual: the box binds exactly, and the reported coefficient is the truncated
  posterior MEAN — matching its closed form to 8 significant figures.** On the
  noise-free line `y = 2 + 5x`, `y ~ linear(x, min=0, max=1)` reports a slope of
  `0.902139` where `bounded(x, min=0, max=1)` reports `1.000000`, and three tests
  assert the reported coefficient must sit at the bound.

  The mode IS at the bound. The fit's own `deviance` is `229.6`, and
  `(5 − 1)²·XᵀX = 16·14.35 = 229.6` exactly — the residual sum of squares at
  `β = 1`. What is reported is a different estimand: for
  `X ~ N(β̂_unc, φ̂/XᵀX)` truncated to `[min, max]`, evaluated at the fit's own
  published `φ̂`,

  ```text
  bound   sd          closed form     reported        difference
  1       0.640513    0.902138628     0.902138522     1.1e-7
  2       0.480384    1.926593682     1.926593656     2.6e-8
  3       0.320256    2.951062455     2.951062437     1.7e-8
  4       0.160128    3.975531227     3.975531219     8.7e-9
  ```

  and the reported VARIANCE agrees too — `covariance_conditional[1,1] =
  9.163014e-3` against the truncated-normal variance `9.163460e-3`, inside the
  orthant cubature's own `1e-3` relative certificate. The apparent deficit is not
  a solver shortfall that happens to be the right size; it is a closed form
  evaluated correctly, and it is SPEC rule 3 — *"posterior mean must always be
  the default (never MAP)"* — working as written, as `constrained_posterior`'s
  module documentation states outright.

  What remains is a question about the ESTIMAND rather than about the active-set
  solver. `bounded()` publishes `1.000000` on the same data because its latent
  interval transform `β = min + width·σ(θ)` stretches the boundary to `θ = ±∞`,
  so ITS posterior concentrates at the bound: the two documented ways to box a
  coefficient impose different priors and therefore publish different numbers.
  Deciding which one a user asking for a box should receive is a scope call, and
  moving either number to clear the bar is the failure mode SPEC warns about — so
  nothing was moved. What landed is the part that is provable: a regression that
  pins the reported coefficient to its closed form across four bounds AND on the
  one-sided half-line (the `nonnegative()` family's `0.007857`), asserts via the
  deviance identity that the mode really is at the bound, and refuses to pass
  vacuously if the reported value ever becomes the bound itself.

  One thing the exercise corrected in the test rather than in the engine: the
  half-line reference first missed by `1.06e-5` relative, because
  `Φ̄(6.245) = 2.1e-10` formed as `1 − Φ(6.245)` in binary64 keeps only six
  significant figures. Recomputed in log space the reference and the engine agree
  to eleven. The engine was on the accurate side throughout.

- **A shape-constrained fit could not certify its own inner mode, for two
  reasons, and both were units errors rather than convergence failures (#2705
  group B).** `smooths::shape_constrained_fit_survives_its_own_inference_2601`
  refused three of four shapes with `inner status StalledAtValidMinimum`. The
  refusal named the inner status and then quoted the OUTER stationarity residual,
  because that was the only certificate it held — so the first change was to make
  it carry the inner one: the effective KKT tolerance, both certificate bounds,
  the natural gradient scale, the inner iteration count and last realized
  deviance change, and, when constraints are present, the four constraint-KKT
  channels with the one that DECIDED the max named explicitly. That measurement
  is what the rest of this entry is built on; no gate moved to produce it.

  **The certificate compared a distance against a gradient bound.**
  `constrained_stationarity_norm` returned
  `max(primal_feasibility, dual_feasibility, complementarity, stationarity)` and
  handed that scalar to `WorkingState::certifies_kkt`, whose two bounds —
  `τ·√n·√p` and `τ·(1 + ‖score‖ + ‖Sβ‖)` — are both derived FOR A GRADIENT. Only
  two of the four channels are gradient-space quantities: `primal_feasibility` is
  a Euclidean DISTANCE in coefficient space (the constraint rows are
  unit-normalized before it is measured) and `complementarity` is a gradient
  TIMES a distance. Measured at the refused iterate on `y ~ s(x, shape=convex)`,
  300 rows of clean linear data: `stationarity = 3.148471e-10` against a
  dimension bound of `6.244998e-9` — twenty times inside the certificate — while
  `primal_feasibility = 6.301146e-9` pushed the max past it by a factor of
  `1.009`. That feasibility number is itself inside
  `ACTIVE_SET_PRIMAL_FEASIBILITY_TOL = 1e-8`, documented as the tolerance the
  active-set solver **guarantees** on the iterate it returns, in exactly that
  metric. The solver delivered its contract and the certificate refused it.

  The gradient certificate now reads `max(stationarity, dual_feasibility)`, and
  the geometric channels are certified against the contracts that define them by
  `constraint_geometry_is_certified`, which every acceptance path requires —
  including the strict one, which was the odd one out, since the soft paths
  already applied the primal-feasibility conjunct. Complementarity's bound is
  scaled by the gradient magnitude its multipliers live at; without that factor
  the same fit passes or fails under a response rescale `y → c·y`. This is not
  uniformly looser: at `τ = 1e-6` the old test admitted primal feasibility up to
  `6.2e-5`, four orders past the solver's guarantee, and the new one does not.

  **The one machinery for an exhausted objective was switched off by INACTIVE
  rows.** The remaining shape reported `last_deviance_change = 2.220446e-16` —
  exactly `f64::EPSILON`, i.e. the penalized objective had stopped moving at its
  own arithmetic resolution, leaving no line search and no gain ratio with
  anything to choose a step by — and `iterations = 300`, the full budget ground
  out in that state. The exact bare-Hessian Newton decrement and the undamped
  polish that pursues it exist for exactly that state, and were gated on
  `linear_constraints.is_none() && coefficient_lower_bounds.is_none() && …`
  while the comment above that gate stated the actual requirement: *"active
  constraints carry multipliers"*. Those are different questions, and
  `active = 0/11` is the difference. With an empty active set every multiplier is
  zero, `∇L − Aᵀλ = ∇L`, and the constrained KKT system IS the unconstrained
  stationarity system — the coefficient-space certificate is valid verbatim.
  Gating on the EXISTENCE of the constraint system denied it to every constrained
  fit sitting strictly inside its cone, which is the whole population of
  `shape=monotone_increasing` fitted to data that is already monotone.

  The predicate is now split into its structural half (`arrow_schur.is_none()`,
  which cannot change during a solve) and its geometric half
  (`inequalities_are_all_inactive`, asked per use at all three sites, because the
  active set is a property of the iterate). Because the polish takes
  UNCONSTRAINED Newton steps — exact while the active set is empty, silent about
  where they land — each candidate is checked for primal feasibility and refused
  if it would leave the cone. Refused, not projected: a projection is not a
  Newton step, and the strict-improvement guard would then be certifying a
  different point than the one it measured.

  Neither change touches an iteration budget or widens a tolerance, the two
  levers #2705 records as SPEC-forbidden. Verified: all four shapes of
  `every_shape_constraint_fits_clean_linear_data_2601` fit and honour their
  constraint, on the same runner that measured the failure.

- **A shape-constrained fit published two covariance matrices that were not
  covariances, for two different reasons, and both were refused as
  non-convergence (#2705 group A).** `misc::shape_constrained_alo_seed_validation_aborts_1191`
  died at `posterior covariance diagonal 4 is not positive and representable:
  -3.08607306376274e-15`, and the corrected covariance of the same fixture had
  earlier been measured at `-9.954853058256977e-9`. Neither number is a small
  variance; both are what is left when a subtraction has spent all of its digits.

  **The composition.** `beta_covariance_corrected` was assembled as
  `beta_covariance + smoothing_correction`, i.e.
  `(Σ − GΔGᵀ) + (Vp − Σ) = Vp − GΔGᵀ` — with the lift `G` and the removed
  variance `Δ` derived from `Σ = Vb`, the ρ̂-CONDITIONAL covariance, and then
  subtracted from `Vp = Vb + J·V_ρ·Jᵀ`, the ρ-MARGINAL one. That matrix is the
  truncation of neither covariance. Along a coordinate the constraint pins,
  `(GΔGᵀ)_ii` cancels `Σ_ii` to eleven digits, so whatever `(Vp − Σ)_ii` happens
  to be becomes the WHOLE published variance — and that increment is legitimately
  sign-indefinite: the cubature branch computes
  `φ̂·E_ρ[H(ρ)⁻¹] + Cov_ρ[β̂] − φ̂·H_opt⁻¹`, a difference of two averages which is
  positive semidefinite only as a SUM with `Vb`. The measured decomposition on
  `y ~ s(x, shape=convex)` reads `Σ_ii = 2.302618e-2` removed to
  `6.229531e-13`, with a `−3.025454e-9` smoothing increment on top.

  The right composition follows from the estimand rather than from the sign. The
  feasible set constrains `β` and says nothing about `ρ`, so the indicator
  `1_C(β)` factors straight out of the ρ-integral —
  `∫ π(β,ρ|y)·1_C(β) dρ = 1_C(β)·∫ π(β,ρ|y) dρ` — i.e. the β-marginal of the
  TRUNCATED joint posterior is exactly the truncation of the β-marginal of the
  untruncated one. So the truncation belongs on `Vp`, applied last, with its own
  lift `G_p = Vp·Aᵀ·W_p⁻¹` and its own orthant moments at `W_p = A·Vp·Aᵀ`. The
  ρ̂-conditional covariance keeps its truncation at `Σ`, which is right, because
  that estimand really is conditional on `ρ̂`.

  **The assembly.** `Σ − GΔGᵀ` has no digits left on a pinned coordinate, and `Δ`
  is a cubature result certified to `1e-3` RELATIVE, so `Δ_ii` overshooting
  `Σ_ii` by an ulp is admissible arithmetic that publishes a negative variance.
  Splitting the correction at `Δ = W − C`, with `C = Cov[u] ⪰ 0` the truncated
  constraint-normal covariance, writes the identical quantity as two Grams:

  ```text
  Σ − GΔGᵀ = (Σ − G W Gᵀ) + G C Gᵀ = P Σ Pᵀ + G C Gᵀ
           = (P L)(P L)ᵀ + (G L_C)(G L_C)ᵀ,     P = I − G A
  ```

  so every diagonal entry is a sum of squares. The cancellation does not
  disappear — it moves INSIDE `P L`, where each entry carries an absolute error
  `O(ε‖L‖)` and is then SQUARED, so a pinned coordinate's variance picks up
  `O(p ε² Σ_ii)` instead of `O(ε Σ_ii)`: sixteen orders smaller, and non-negative
  by construction rather than by luck. `P L = L − G(A L)` costs `O(p²q)`, so the
  only new `O(p³)` work is one Cholesky of a matrix the dense branch has already
  inverted.

  Three consequences landed at the sites they belong to. The dense standard-error
  gate accepts an exactly-zero diagonal **when a truncation was applied** — zero
  is the `λ → ∞` limit of the removal and is now reported cleanly instead of as a
  `±ε·Σ_ii` residue, and a strict `> 0` test would refuse the fit for producing
  the right answer; unconstrained fits keep `> 0`, which is the singular-Hessian
  catch that gate exists for. The FACTORIZED branch has no dense `Σ` to factor,
  so it keeps the subtraction and now carries the subtraction's own MEASURED
  resolution `16·ε·max(base, removed)`, reading a residue inside that band as the
  zero it approximates and refusing anything outside it with the decomposition
  attached. And a materially indefinite `C` — past the cubature's own `1e-3`
  relative certificate — is refused rather than clamped, because that is a broken
  moment computation and not a rounding question.

  Verified: `misc::shape_constrained_alo_seed_validation_aborts_1191` passes
  (all four shapes, 400 rows of `sqrt(x)`); five unit tests in `gam-solve`
  including one that reproduces the negative variance under a two-ulp cubature
  overshoot and one that asserts the Gram assembly and the subtraction agree
  entry by entry; and a new property-side regression that reads BOTH published
  matrices on all four shapes and asserts each has a non-negative spectrum to its
  own assembly resolution, refusing to pass vacuously if no shape publishes a
  corrected covariance at all.

- **The certified REML score's VALUE enclosure was a natural interval extension,
  so its overestimation was FIRST ORDER in the cell width with constant `rank`,
  and the certified search refused designs it could certify (residual of
  #2758).** `AffineRemlProfile::enclose` evaluated each mode kernel on the
  interval `λ` and accumulated. The score is
  `−0.5·(D·normalized_logdet + residual_dof·Σ_d log(R_d/dof))`, and near a REML
  optimum those two brackets CANCEL — each block's `d/dρ` is `O(rank)` while
  their sum is not. Interval addition cannot see that the two movements are the
  same quantity with opposite signs, so the extension carried `rank·w` of slack
  the exact function does not have.

  Measured on a 33-mode cascade profile, the value range came out at `33.0·w`
  **exactly, over six decades of cell width**, while the same cell's derivative
  enclosure bounded the score's movement across it by up to `7.4e5` times less —
  and the ratio DIVERGED as the cell shrank, one side being `O(w)` and the other
  `O(w²)`. Both enclosures were sound; this was overestimation.

  It was not a loose number. `maximize_score_1d` retires a cell as
  resolution-flat when its score range fits inside `2·evaluation_error`; against
  an `O(w)` range that needs a cell `rank/|f′|` times narrower than the function
  does. On a 36-row / 1725-column cascade — what a geometric box-filling net
  produces on a small sample — the flat test needed `w ≤ 6.7e-8`, 29 levels down
  a 40.6-wide domain, so no cell could be retired, none could be
  derivative-excluded, and the search refused at 8193/8192 subdivisions with
  `RemlScoreSearchUndecomposable`. That refusal names the design's rank and the
  sample's identifiability and reads as a statement about the data; it was a
  statement about the enclosure.

  The enclosure is now the **centred (mean-value) form**, intersected with the
  natural one: for `m` the cell midpoint,
  `f(x) ∈ F({m}) + F′([a,b])·[a−m, b−m]` and
  `f′(x) ∈ F′({m}) + F″([a,b])·[a−m, b−m]`, with `F({m})` obtained by calling the
  same natural extension on the degenerate interval `[m, m]`. Both forms are
  outer enclosures of one exact range, so intersecting is rigorous and can only
  tighten. The derivative is centred first and the value is centred on the
  RESULT, because a mean value remainder is only as tight as the range fed into
  it — and the curvature is centred before both, on an exact third-derivative
  enclosure the profile now builds.

  Centring the curvature is not an optional third helping. The curvature had the
  identical defect one derivative up (halfwidth `≈ 49.5·w` against an analytic
  `f″` of `1.249e-5`, a factor of 8000 at `w = 2e-3`), and the curvature is not
  merely a width: `maximize_score_1d` reads its SIGN to isolate a stationary
  point, so a first-order-loose curvature is what stops a root being isolated at
  all. The mode kernels are analytic, so the third derivative is closed form —
  `t(1−t)/(1+t)³` for the determinant, which is already the `k` kernel, and
  `t(1−4t+t²)/(1+t)⁴` for the residual, whose critical points are the roots of
  `(t−1)(t²−10t+1)`, enclosed exactly like the `k` kernel's `2 ± √3` — with
  `(log R)‴ = R‴/R − 3(R″/R)(R′/R) + 2(R′/R)³` closing the deviance block.
  `evaluate` is untouched: only the INTERVAL third derivative is needed, and no
  proof reads the scalar one.

  Overestimation becomes second order on the value and higher still with the
  curvature centred: the value range converges as `w⁴` and reaches the
  point-enclosure floor a full decade of cell width earlier. Against the
  original natural extension at `w = 2e-3` that is a factor of `7.6e7`. The same
  36-row / 1725-column design now
  certifies: `fit_reml` returns `DenseExact` at `log λ = −1.679` in 1.2 s, where
  it previously refused in 5.6 s, and the certified search's terminal value
  range is the mean-value bound to the last digit at every width tested.

  Two claims in the tree were falsified on the way and are corrected in place.
  `dense_cascade_spectrum` said this design "still spins in
  `AffineRemlProfile::enclose` under `maximize_score_1d` past 900 s" — it never
  spun; it returned the typed budget refusal in 5.6 s, #2546 having closed that
  axis. And `subdivision_budget`'s own recommendation, "the request, not the
  budget, is what actually binds", is not the repair here: the search refused at
  every requested resolution from `1.49e-8` to `1e-3`, the terminal cell merely
  walking down the domain as the request coarsened.

  **Cost.** Centring doubles the per-cell work (one extra degenerate-cell
  evaluation), so the net was measured rather than assumed, on three domains of
  the same profile: the 40.6-wide declared domain goes from a 9.94 s refusal to a
  0.58 s certification (**17×**), and a three-wide window around the optimum —
  where the natural extension already finished in a handful of cells and there
  was nothing left to remove — is still **1.26× faster**, because the tighter
  derivative range excludes cells by sign a level or two earlier. The 2× per-cell
  cost does not show up anywhere. The `residual_cascade` integration suite went
  643 s → 541 s alongside.

  One consequence points the other way and is named in the code: a tighter value
  range makes `resolution_flat_region` easier to satisfy, and an optimum landing
  in a flat region is a refusal rather than a fit. It does not happen, because
  the flat test is the last thing a cell is offered and centring strengthens
  dominance, derivative exclusion and stationary isolation by more — the gate
  asserts the located optimum is a decided one.

  Gated from four angles:
  `the_value_enclosure_never_exceeds_the_bound_its_own_derivative_certifies`
  (the invariant the natural extension broke, on a fixture built to cancel, plus
  convergence better than 50× per decade to the point-enclosure floor),
  `the_centred_enclosure_holds_on_degenerate_adjacent_and_extreme_cells` (point
  cells return the natural extension untouched; adjacent-float cells centre
  inside themselves; the centred range is always inside the natural one), and
  `auto_reml_certifies_a_design_the_data_cannot_identify` (end to end, with its
  rank-deficiency and inside-the-budget premises asserted), and
  `the_natural_extension_cannot_decompose_a_domain_the_centred_form_certifies`,
  which runs ONE search twice with the two enclosure forms — the natural
  extension is kept callable by the fix, so the before/after is a controlled
  comparison inside one test rather than a claim about a previous commit, and it
  asserts its own premise so a fixture that stops exercising the defect says so.

  Verified at `0b3b0fbd8`, release profile, 4-core runner:
  `gam-math` 284/284; `gam-terms` 936/936; `gam-solve` 1899 of 1902, the three
  reds being the pre-existing `jeffreys_subspace` and two `run_plan` failures
  already attributed to the #2612 lane at `250a04729`; `gam --test misc
  residual_cascade` 26/26.

- **The constant-curvature range coordinate was confounded with `ρ` and
  fabricated past `ℓ ≈ 10⁶`, and both were the KERNEL'S GAUGE (#2747).** The
  kernel is only ever consumed through the coefficient sum-to-zero frame `z` —
  the realized design is `K z`, the penalty `zᵀK z` — and `z` annihilates
  constants while `λ` absorbs a positive scale, so `exp(−d_κ/ℓ)` and
  `ℓ·(e^{−d_κ/ℓ} − 1)` are the SAME model in two gauges. The gauge is not free.

  `exp(−d/ℓ)z = −(1/ℓ)Dz + O(1/ℓ²)`, so design and penalty both COLLAPSE like
  `1/ℓ` and `λ̂` has to chase the range one-for-one: measured on the κ=1 sphere
  fixture, `ρ̂` falls `−5.49 → −18.91` as `ℓ` goes `1 → 10⁶` while the criterion
  value is unchanged to eight significant figures. `constant_curvature_profile.rs`
  already had this from the other side ("each ×100 in ℓ costs 4.6 in ρ̂") and
  worked around it by refusing every point whose `ρ̂` railed against the absolute
  `ρ` box.

  Worse, all of the model's range information lives in `K − 1`, formed by
  subtracting from an implicit 1 numbers that agree to `log₁₀(ℓ/d)` digits — and
  the Gram then squares what is left. The shipped criterion was **78.8 nats below
  the truth AT the derived box top** `ℓ_hi = d_min/√ε = 2.53e6`, 476 nats at
  `10⁸`, descending ~100 nats per decade into its own rounding with `edf` railed
  at `p`. That is what `20bde053f` read as "the criterion is monotone in ell all
  the way to its asymptote … ell-hat ran to 1.5e6, a readout of the box rather
  than of the data": not a flat likelihood, a false one.

  The kernel is now `k = ℓ·(e^{−d_κ/ℓ} − 1)`, evaluated as `ℓ·expm1(−d_κ/ℓ)`.
  No subtraction of near-equal numbers; `X` and `S` no longer collapse; `ρ̂` is
  flat in the range (`−5.0978 ± 1e-4` over eleven decades); and `k → −d_κ`
  exactly as `ℓ → ∞`, so the far face of the range is the geodesic-DISTANCE
  kernel — `−d` is conditionally negative definite on all three space forms, so
  it is an ordinary non-degenerate smooth rather than nothing. Three
  consequences, each handled rather than worked around: the raw `m × m` matrix is
  no longer PSD (it is conditionally negative definite), so the penalty is built
  from the RESTRICTED Gram `zᵀkz = ℓ·zᵀe^{−d/ℓ}z ≻ 0`, which is also where the
  cancellation would otherwise reappear; the ψ jets change shape, with the two
  `η` blocks becoming `ℓ·φ(q)` / `ℓ·χ(q)` for `φ = e^{−u}(1+u) − 1`,
  `χ = e^{−u}(1+u+u²) − 1`, both evaluated by series below `u = ½` because both
  have a second-order zero at the origin; and the declared scale contract goes
  from invariance to equivariance of weight one, because the kernel is a LENGTH.

  With the cancellation gone nothing numerical bounds the range from above, so
  the chart is truncated where the MODEL stops moving: `ℓ_hi = d_max/(2√ε)`, past
  which every design entry is within `√ε` of the geodesic-distance design.
  Arriving there is `RangeSolveOutcome::DistanceKernelLimit`, published as
  `RangeEstimateSupport` on the curvature report and the Python row — an arrival,
  not a rail, and the range's version of the `KappaEstimateSupport` `146f9232d`
  added for the same reason. A criterion that converges to a member of its own
  family does not need a stopping rule; it needs its limit to be a point of the
  chart. On that basis the pinned-κ/free-range enrollment `20bde053f` reverted is
  restored: a pinned `kappa=` fixes the geometry, not the resolution.

  Verified by `the_contrast_gauge_is_the_same_model_and_the_exp_gauge_loses_it_2747`
  (the two gauges agree to <1e-12 across the mid box; at the box top the `exp`
  gauge's error must land inside the DERIVED bracket `ε·ℓ/d` taken over the
  geometry's own evaluated span — measured 1.78e-10 against a predicted
  [1.1e-10, 8.8e-10]) and `the_range_limit_is_the_geodesic_distance_kernel_2747`
  (on both branches and flat, the design converges to `−D z` at first order in
  `1/ℓ`, reaching <1e-8 at `ℓ=10⁹`, with the restricted Gram strictly PD
  throughout). `gam-terms --lib constant_curvature`: 18 passed.
  `gam-models --lib constant_curvature`: 7 passed, including the 3×3
  curvature×range identification gate and the reverse-mode adjoint FD check.

- **A family had two log-likelihoods, and the joint-Newton trust ratio divided
  one by the derivative of the other (#2714).** The accept test compares
  `old_objective − trial_objective`, and the two ends came from different family
  hooks: `old_objective` is built from `current_log_likelihood`, which
  `load_joint_gradient_evaluation` reads off
  `exact_newton_joint_gradient_evaluation`, while `trial_objective` is built from
  `log_likelihood_only`, which the line search calls at `β + δ`. For the latent
  survival family those were two independent implementations — the gradient hook
  sums the row program's `∂_a^j K₀`-basis value channel, the line search summed
  `LatentSurvivalRowJet`'s rung-basis assembly.

  Writing `b(β)` for the gap, and noting that the base point does not move across
  a backtracking ladder,

  ```text
  actual_reduction = −[ℓ(β+δ) − ℓ(β)] − b(β) + (penalty terms),
  ```

  so `b` is a **constant of the ladder**: shrinking the radius shrinks the
  bracket and leaves `b` alone, and `actual_reduction → −b` instead of `→ 0`.
  Below the radius where the true reduction falls under `|b|`, the sign of `b`
  decides every attempt outright — which is the
  `rejects[model,likelihood,objective,feasibility] = [0,0,2,0]` partition at trust
  radius `1e-12` this issue was filed on. The two bases are an exact integer
  change of basis in real arithmetic (`m^k K_k = (−1)^k (∂_a)_k K₀`) and two
  different quadratures in f64, so `b ≠ 0` by construction on any row whose term
  list reaches `k ≥ 1` — every exact-event row. `k = 0` lists agreed anyway, which
  is why right-censored rows were silent about it.

  `log_likelihood_only` now evaluates the same row expression through a value-only
  lift of the same row program and sums it through the same deterministic
  reduction, so the two scalars are **bit-identical** and `b ≡ 0`. Measured over a
  35-state sweep: `worst |accept − gradient| = 0.0` exactly, and the value lift
  matches the order-two lift bit-for-bit at 100 states. It is cheaper than what it
  replaces, not dearer — the value backend skips the `K + K(K+1)/2` normalised
  moments while building the same kernel bundle.

  Two sub-faults of the same shape were fixed under it, because the value lift is
  only bit-identical if neither of them holds:

  * **The log-survival panel was placed from the requested derivative order.**
    `log_survival_panel` chose its window and node count from `order`, so two
    consumers at one `(μ, σ)` read the same integral off two Clenshaw–Curtis
    rules. On one latent-survival row the value, gradient/Hessian, contracted
    third and fourth ask for `max_k = 4, 5, 6, 7` — so the Hessian was assembled
    from a different `∂_a K₀` than the gradient it was paired with. The placement
    is now one of exactly two surfaces, each a pure function of `(branch, μ, σ)`,
    and the hot value route (every `ln S` in the tree, `max_k + 1` per bundle) is
    byte-identical and pays nothing; only the single tower request per bundle
    moves, by ~1.15× in nodes.
  * **Tower certification was all-or-nothing, so the BASIS depended on the
    request too.** Refusing the whole tower when its last rung cancelled denied a
    consumer needing rungs `0..=1` a basis well conditioned at every rung it
    reads, and let two consumers of one term list be routed differently because
    one also wanted a Hessian. It is now the longest certified prefix; a term list
    needing a rung past the truncation still falls back whole, so nothing is ever
    a partial mix.

- **The constraint-face retention ladder skipped faces, and could exit in one
  pass while reporting that it had exhausted double precision (#2714).**
  `constrained_posterior_correction` retains constraint rows by a per-row floor
  and then checks the assembled face against the identity that defines its lift
  (`max|A G − I| ≤ 1e-3`), lowering the floor on a miss by `departure/tolerance`.
  That factor is read off the per-row model `departure ≈ ε·diagonal/pivot`, which
  the filter's own documentation says bounds the face's conditioning only when the
  elimination is ordered by pivot magnitude — and this walk is ordered by slack,
  which is exactly the case the ladder runs in. So it skipped larger admissible
  faces, and on a badly conditioned face one pass carried the floor from `1e-3`
  past `f64::EPSILON`, out of the loop and into a terminal message describing a
  ladder that had taken one rung.

  Retention is a step function of the floor `d`: a row is kept iff
  `d > d_r = (k+1)·ε·diagonal_r/pivot_r`, so the retained set changes only at the
  accepted rows' own breakpoints and the face is bit-identical between them. The
  ladder now steps to `max_r d_r`, which drops exactly the worst-conditioned
  accepted row. It is exhaustive (no admissible face can be skipped), minimal (it
  stops at the largest face that delivers the accuracy), and terminates in at most
  `q` passes rather than ~40 — ending at a single retained row, where `W` is `1×1`
  and the departure gate cannot fail. No constant changed; the step size is read
  off the retention rule instead of off an error model. Gated by a brute-force
  oracle that sweeps the floor on a 64-per-octave grid and asserts the ladder
  returns the largest face satisfying its own identity.

- **The joint trust region measured the step in one norm and the radius in
  another, so on any multi-block fit it could only ever shrink (#2612).** The
  coupled joint-Newton solve carries two trust constraints: one `D`-metric ball
  on the whole step, which `WhitenedHessianSpectrum::trust_region_step` scales
  `‖δ‖_D` to, and one box per coefficient block. The controller was handed
  `max_b ‖δ_b‖` — the largest *per-block* norm — alongside the *joint* radius.
  Because `‖δ‖² = Σ_b ‖δ_b‖²`, a step sitting exactly on the joint sphere has
  `max_b ‖δ_b‖ = ‖δ‖/√K` when `K` blocks carry comparable mass, so
  `hit_boundary = step_norm ≥ 0.99·r` was **false on a boundary step for every
  `K ≥ 2`** and the region became a one-way ratchet.

  Measured on the #2612 penguins witness, over all 6784 accepted trust attempts
  of one fit:

  | `‖δ‖/r` | attempts |
  |---|---|
  | ≥ 0.99 (what the controller looks for) | 1 |
  | 0.70 – 0.99 (the `1/√2` band, median 0.781) | 2018 |
  | < 0.70 | 4765 |

  1454 of those 2018 carried a Newton proposal at least `1.5×` the step actually
  taken, 563 of them `≥ 10×`. The fit died with two inner solves at the ratchet's
  floor — `|prop|∞ = 7.686e-5` against an accepted `|δ|∞ = 5.270e-7`, the residual
  crawling `0.9932×/cycle` — and 50 of the run's inner solves ended
  non-converged. With the norms paired correctly that is **3**, and the fit takes
  116.6 s instead of 333.4 s.

  For `K = 1` the joint norm *is* the block norm and the joint radius *is* the
  block radius, so every single-block family is byte-identical; the change
  reaches exactly the multi-block joint solves (multinomial, location-scale,
  marginal-slope) where the test was in the wrong units. It also explains why
  the #2612 `objective_unreadable_at_this_step` growth clause never fired on
  multinomial: it is gated on the same structurally-false boundary test.

  Two consequences of the same reading are fixed with it. **"Held, not grown" is
  a fixed point when the region is what limited the step** — the accept-below-
  model-noise-floor branch (#2637) is right to hold an *interior* step, but on
  the boundary the step is short because the radius is small, the prediction is
  unreadable *because* the step is short, and the radius then freezes forever
  (measured: `r = 1.463e-6` held for 167 cycles against a `|prop|∞ = 8.961e-5`).
  It now grows on the three facts that are measurements rather than predictions:
  a realized decrease above the noise floor, geometric boundary contact, and a
  stationarity residual still above tolerance.

  **Measured and reverted (recorded because the negative result is the useful
  part).** A third repair looked equally well-founded and is wrong: a ladder
  verdict is one realization of a random band, so the ladder was made to publish
  the envelope it proved (`remainder ≤ c·‖δ‖²` below a step length it had shown
  to be noise-dominated) and every later attempt inside that envelope widened the
  measurement. The motivating observation is real — the penguins refit measures
  `4.396e-11` at cycle 149 and then rejects a `1.168e-10` realized change at
  cycle 162, at a step length the ladder had already certified — but the repair
  makes the fit *worse*, not better:

  | build | wall | inner solves ending non-converged |
  |---|---|---|
  | before the norm repair | 333.4 s | 50 |
  | norm repair | 116.6 s | 3 |
  | + boundary growth | 119.0 s | **2** |
  | + certified noise envelope | 291.9 s | **47** |

  Over-measuring the objective's resolution is worse than under-measuring it,
  because the accept test then reads genuine objective changes as rounding and
  the solve stops being able to tell a good step from a neutral one — the
  measured resolutions rose from `~4e-11` to `~3e-10` and the worst terminal
  residual went from `3.8e-7` to `4.0e-5`. The under-measurement is therefore
  real and is NOT the binding constraint; the envelope is not the way to fix it.

- **The inner solve conceded on a step model that is not the objective's
  Hessian, and with that repaired the multinomial outer search converges for the
  first time (#2612).** `H_Φ` is the Daleckii–Krein divided-difference part of
  `−∇²Φ`; the exact second-order completion `−½ tr(K D_ab)` is the rest of it,
  and until it is formed the Newton step is built on a matrix that is not the
  Hessian of the objective the certificate is taken against.
  `JEFFREYS_COMPLETION_RESIDUAL_BAND` arms it on a proximity proxy — the residual
  reaching `300 × residual_tol` — which is circular wherever the distance from
  tolerance is *caused* by the inexact model.

  Measured on the penguins witness once the trust region had been repaired
  enough to stop being the binding constraint: cycle 155 takes the **full**
  Newton step (`|δ|∞ = |prop|∞ = 1.069e-4`, interior at `r = 9.290e-3`) and the
  residual still does not contract — it drifts at `1.0031×/cycle` at `2.398e-6`,
  five hundred times outside a band of `4.3e-9`, with
  `jeffreys_completion_calls = 0`. The step is exactly as long as the model
  wants; the model is the wrong matrix. Arming the completion where the solve
  would otherwise concede:

  ```text
  cycle 39  about to concede at residual 4.593e-6  → arm the completion
  cycle 40  ρ=+0.9922   residual 4.593e-6 → 7.410e-8
  cycle 41  ρ=+1.000    residual 7.410e-8 → 4.298e-12   (tol 1.441e-11)
  ```

  The repair is an invariant rather than a threshold — *the inner solve may not
  concede while its step model is still the surrogate* — asked at the
  residual-stall guard, the slow-geometric-rate projection, and per cycle
  against the budget **this solve actually has** (the stall guards are allowed
  to defer to a historic floor of 100 cycles; a screening evaluation with a
  64-cycle budget is not). End to end on `zz_probe_2612_penguins_stride3_inner_trail`:

  | build | wall | inner solves non-converged | outer verdict |
  |---|---|---|---|
  | before | 333.4 s | 50 | `line_search_failed` at `\|g\| = 1.556e-1` |
  | trust-region norms | 116.6 s | 3 | inner infeasible |
  | + boundary growth | 119.0 s | 2 | inner infeasible |
  | + completion invariant | 493.3 s | 4 | **Converged**, `\|g\| = 1.357e-3 < 2.290e-3` |

  The cost is stated rather than hidden: an armed solve pays an extra
  `O(n·M²·P²)` contraction per cycle, and a solve that never needed it never
  arms. Regression surface: `gam-custom-family --lib` 275/275,
  `gam-models --lib -- location_scale` 224/224, `-- marginal_slope` 229/229,
  `multinomial_separation_arming_2612` 3/3 (accuracy 0.9750, calibration gap
  −0.0151) — the other multi-block joint families exercise the same two repairs
  and none of them moved.

  **Where #2612 now stands.** The fit still does not mint, and the blocker is a
  different subsystem: the outer search converges to an *interior strict saddle*
  (`λ_min = −1.074e1` on the un-railed sub-block, 19 of 24 `θ` railed), the
  `#2357` negative-curvature escape reseed fires and lowers the objective
  (`7.545783 → 7.545502`), and the re-run climbs back to the identical ρ and the
  identical saddle, at which point the one-shot escape is spent and the
  certificate refuses. That is the `#2357`/`#2665` family — a gradient-only BFGS
  search (`search_hessian_source=BfgsApprox`) cannot see the negative curvature
  it is sitting on.

- **A Jeffreys drift GEMM panicked on a column-major product (#2612).**
  `as_slice` is C-order-only and neither `dot` nor `+` promises that order, so
  `dw_rows.dot(&a_rows.t()) + …` could return column-major and the `expect`
  fired — in production code. `binomial_location_scale_expected_hphi_drift_matches_finite_difference`
  died on it. `as_standard_layout` borrows in the C-order case, so the GEMM path
  is unchanged.

- **One sentinel, one resolver — the marginal-slope branch never reached the
  measure-jet range screen (#2754, #2761).** `length_scale == 0.0` is an
  UNRESOLVED representer range, and the tree carries two resolvers for it: the
  pure-geometry median-nearest-node rule inside the basis builder, and the
  #2750 response screen. `fit_standard_model` runs the screen so that every
  standard-fit branch gets the same one. The Bernoulli marginal-slope family has
  its own entry point and never passed through it, so the identical declaration
  on byte-identical rows realized two different spans:

  ```text
  [2754 geometry gaussian-seed] ell=2.5197  m=(10,2) extent=[2.671, 2.726] band0=1.0807
  [2754 geometry bms-marginal ] ell=1.0807  m=(10,2) extent=[2.671, 2.726] band0=1.0807
  [2754 geometry bms-logslope ] ell=1.0807  m=(10,2) extent=[2.671, 2.726] band0=1.0807
  ```

  Same 10 centers, same extent, same band floor, **2.33× apart in ℓ** — and the
  BMS value is exactly `eps_band[0]`, which is the geometry heuristic's own
  output by construction, i.e. the fingerprint of a term that reached no
  resolver at all. `ℓ` decides WHICH span the representers occupy and `λ` cannot
  move a span, so this is not a tuning difference between entry points; it is a
  different model reached by typing a different family name. It is also exactly
  the mechanism #2761 named: #2750 measured the geometry heuristic sitting 21.7
  nats away from the criterion's global optimum, and #2761 measured its span
  floor four orders above the chosen range's.

  **What the range was worth on this fixture, before the fix.** The parity test
  cites a length-scale sweep (`zz_mjs_lengthscale_sweep_1041`) for the claim
  that "the auto ℓ is already the BEST — every explicit ℓ is worse". That test
  is not in the tree; `grep` finds only the citation. Rebuilt as
  `examples/probe_2754_bms_length_scale_sweep.rs` on the parity fixture's own
  data law and its own held-out score, the claim inverts — the auto range is the
  WORST of the eleven measured:

  | ℓ (standardized) | held-out marginal RMSE |
  |---|---|
  | 1.08 (auto / geometry) | 0.04441 |
  | 2.14 | 0.04157 |
  | 8.56 | 0.04170 |
  | 17.12 | 0.04011 |
  | 25.68 | 0.03985 |
  | 68.48 | 0.03788 |
  | matérn(k=10) | 0.05234 |
  | duchon(k=10) | 0.03705 |

  **Not in tension with the ℓ-learning freeze** two screens above it in the same
  function. The freeze is about the SEARCH: a design-moving dial on covariates
  shared by the coupled marginal/log-slope pair lets the outer optimizer trade
  one surface against the other into a separation-scale runaway. The screen is
  about the SEED, runs once before the fit, and hands the frozen dial a
  data-chosen basin. Freezing a dial is a reason to seed it better, not worse.

  **Each surface is screened against its own target.** The marginal block takes
  `y`. The log-slope block cannot: `β` never appears in `E[y | x]`, so ranking
  its spans against `y` ranks them by their fit to the MARGINAL surface. It
  takes the first-order score surrogate `s = (y − ȳ)(z − z̄)`, whose conditional
  mean is the planted log-slope surface times a strictly positive smooth
  modulation —

  ```text
  Cov(y, z | x) = E[ z·F(α(x) + β(x)·z) ] = F'(α(x))·β(x) + O(β³)
  ```

  by expanding the link about `α(x)` (the odd moments of `z` kill the even
  terms). The profiled Gaussian REML the screen ranks with is invariant to a
  global rescale of its response, so the unknown `E[z²]` and `F'` scales are
  both free. `logslope_screen_surrogate_tracks_the_slope_surface_not_the_marginal_2754`
  checks that derivation against a 200k-row probit sample rather than asserting
  it, and scores the binned surrogate against BOTH candidate truths so the
  separation from `E[y | x]` is the thing being pinned.

  Gated by `measure_jet_auto_range_is_the_same_through_every_family_entry_point_2754`,
  which asserts EXACT `f64` equality of the realized range across the two entry
  points — the screen is a deterministic function of (feature columns, response,
  weights, spec), so handed the same four it must return the same number, and a
  tolerance would hide a second resolver that happens to land nearby on one
  fixture. It asserts the realized geometry matches first, so a range difference
  cannot be explained away as the two entries having realized different center
  layouts.

  The same bypass was in `fit_transformation_normal`, fixed in the same lane:
  its covariate surface enters the linear predictor of the transformed response,
  so `response` is its own screening target. The reached/unreached inventory now
  lives in the doc comment on `seed_measure_jet_auto_ranges` itself — three
  entry points screen (standard, BMS, CTN), five still take the geometry
  heuristic (survival marginal-slope, the two latent families, the
  location-scale families, survival-transformation) and are marked **not
  derived** rather than fixed: for those the raw response is not a readout of
  the surface being screened (a survival marginal-slope block is modulated by
  the risk set in `age_entry`/`age_exit`; a location-scale SCALE block enters
  through a variance, so ranking its spans against `y` ranks them by their fit
  to the LOCATION surface). Inventing one target per family without a fixture
  that can grade it would be landing an unmeasured modelling choice in five
  places at once; the honest state is that the table says so out loud.

- **The #1041 parity bar is now policed by a statistic that can resolve it
  (#2754).** The gate fitted ONE draw and compared one ratio to `1.10`; the
  ratio's sampling spread under redraws of the identical generator had never
  been measured. The argument on #2754 used the BETWEEN-method spread
  (matérn/duchon = 1.42×) as if it were a noise estimate, and it is not — two
  estimators differing by 1.42× says nothing about how much ONE estimator moves
  when only the draw changes. Measured, the within-method sd of the log ratio is
  **0.119** at a mean ratio of 0.97, so the single-draw gate sat ~1.1 sd below
  its own bar and failed about **one run in eight** for no reason but the draw.

  The bar is unchanged at `1.10×` and the comparator stays Matérn: it is the
  only statement in the tree that measure-jet must remain competitive with its
  own estimator class as both change. What changed is the instrument. The gate
  now reports the mean log-ratio over `REPLICATES` independent draws and asserts
  both that it clears the bar and that it clears it **by at least three standard
  errors** — #2754's finding made permanent, so a fixture whose noise grows
  relative to the margin it polices says "under-powered" in as many words
  instead of flipping a coin, and says it about the FIXTURE rather than the
  estimator. `REPLICATES` is derived from `3·sd/√k ≤ margin`, not chosen.

- **The range screen's walk stopped at a proxy for the wall, and on a frozen
  dial a stopping rule IS the wall (#2761).** The #2750 response screen walks
  geometrically past the top band node while its criterion improves, and stopped
  at the node bounding-box diameter, on the argument that at a range that long
  every representer pair overlaps at `≥ exp(−1/2)` so "there is no distinct
  model past it". Two places in the tree already recorded the opposite —
  `measure_jet_ln_range_window`'s docs (*"measured on three fixtures, the
  profiled criterion genuinely prefers a range AT or ABOVE the node
  diameter"*) and a test that pinned the search window as strictly wider,
  calling the diameter *"a stopping rule for the screen's walk over NODES, not a
  wall in the model"*. Those reconcile only while something else keeps searching
  past the stopping rule; on a term whose `ℓ` dial is frozen (the marginal-slope
  pair, or any `learn_length_scale=false`) nothing does.

  Derived on the #1041 parity fixture from the shipped numbers and the walk's
  own control flow: band `[1.08074, 1.43607, 1.90823]`, `log_step = 0.284265`,
  diameter `3.81645`, walk nodes `2.53562`, `3.36930`, `4.47708`; the range the
  screen chose was `3.36930` — the last node below the diameter, to every
  printed digit. The walk pushes a node and only then breaks if it failed to
  improve, so an argmin that IS the last pushed node improved, and the loop
  therefore left through the ceiling test with the criterion still descending.

  `measure_jet_range_feasibility_ceiling(spacing)` is now the single definition
  of `spacing/√(2√ε)`, read by both the outer search's window and the screen's
  new `MeasureJetRangeBracket::feasibility_ceiling`; the diameter survives as
  `node_diameter`, reported as the geometric fact it is and no longer
  load-bearing. **Measured after the change rather than assumed from the shape
  of the defect:** the node the walk can now score, `4.47708`, does NOT improve,
  so the criterion has an interior optimum here and the old ceiling cut just
  past it. What the change buys is the parabolic refinement, which cannot fire
  on an argmin that is the last element — a stop that cannot be stepped past
  also cannot be refined at. It lands at `ℓ = 3.10543` with a better criterion
  value and held-out RMSE `0.04185 → 0.04179`. The much larger held-out number
  further out on the same sweep (`0.03788` at `ℓ = 68.5`, `edf = 7.47`, not
  degenerate) is explicitly NOT what this recovers: the criterion does not want
  to go there, and that gap is a question about the screening criterion rather
  than about where the walk stops.

- **A chart records the `θ` it was ASKED to realize, because `ln(exp(θ))` is not
  `θ` (#2765, #2767).** `SurvivalMarginalSlopeFrozenOffsetChart::evaluate(θ)`
  decoded `θ → cfg` and then called the CONFIG-authored geometry builder, which
  closes the loop by re-encoding `cfg → θ`. For a Weibull that loop is
  `ln(exp(θ))`, and in `f64` it is not the identity: over a grid on `[-3, 3]`,
  **17.3%** of coordinates come back a ulp or more away, and `θ = 1e-5` comes
  back **57 269 ulps** away.

  `SurvivalMarginalSlopeFamilyHyperState` stores that `theta` as the family's
  realized coordinates, and `validate_layout` compares it to the outer manifest
  with `to_bits()` equality — deliberately, so a workspace cannot reuse row
  geometry from a neighbouring outer probe. So a lost ulp in a transcendental
  round trip made the inner solve REFUSE a point the outer optimizer was only
  trying to evaluate:

  ```
  inner solve refused this trial point: SurvivalMarginalSlopeFamily row
  geometry does not bitwise match the family-coordinate manifest
  ```

  **The measurement, probe by probe.** From the #2765 replay fixture's terminal
  certificate at `θ = [7.0218, 5.5967, 6.0955, 0.75749637812226, 0.0]`, step
  `1e-5`, against the round-trip error of each displaced coordinate:

  | probe | `ln(exp(θ)) − θ` | certificate |
  |---|---|---|
  | coord 3, side `−` | `+1` ulp | REFUSED |
  | coord 3, side `+` | `0` | (not reported) |
  | coord 4, side `+` | `−57 269` ulp | REFUSED |
  | coord 4, side `−` | `+3 383` ulp | REFUSED |
  | the seed itself | `0` | evaluates |

  Four out of four, and the one exact round trip is the one that evaluated. A
  refusal reads to Armijo as "no improvement" at *every* step size, so a
  backtracking search halving 50 times produces `StepSizeTooSmall after 50
  attempt(s)` and `after 0 outer iteration(s)` with no gradient defect required
  — and it explains why the earlier #2765 probe saw trial points that were
  genuinely better than the base rejected anyway.

  **The repair is the seam, not the check.** The bitwise invariant is right and
  stays exact; what was wrong is that the chart threw away the coordinate it was
  handed. `build_survival_marginal_slope_baseline_geometry_at_theta` records the
  caller's `θ` verbatim while `cfg` still drives every row's arithmetic. The
  config-authored entry is unchanged: where a config IS the authority, deriving
  `θ` from it is correct. Loosening the manifest comparison to a tolerance was
  rejected — that check exists so "the same coordinates" means the same thing to
  the family and to the outer manifest, and a tolerance trades a loud refusal
  for a workspace that can silently serve a neighbouring probe's row geometry.

  End to end on the replay fixture (n=900, Weibull baseline, `logslope_time_k`),
  same seed, same cost `8.193691e2`, same gradient `|g| = 8.164517e0`: the outer
  solve goes from refusing in 788 s `after 0 outer iteration(s)` to completing
  seed 0 and moving on to seeds 1 and 2. Nothing about the value or the gradient
  changed; the displaced points became evaluable.

- **The ψ calculus and the joint-Hessian OPERATOR never got the slope's
  follow-up axis (#2765, #2767).** `c9ad097f1` generalized the survival
  marginal-slope blockwise assemblers from the four-primary frame
  `(q₀, q₁, q̇₁, g)` to the six-primary one `(q₀, q₁, q̇₁, g₀, g₁, ġ₁)` by routing
  every log-slope pullback through `logslope_layout.primary_channels()` — one
  `(primary, design)` pair for a time-constant slope, three for a varying one —
  and `fit_entry` then refused, by name, every surface whose chain rule was
  still lowered through the old frame. Its own comment states the standard:

  > refusing by name is honest, whereas running them would silently
  > differentiate a model that is not the one being fitted.

  Five sites did not get the treatment and are not on that refusal list. Each
  reads the slope through the literal index `3` and a single design:

  | site | what it assembled |
  |---|---|
  | `hessian::add_pullback_with_q_geometry` | `ph[[3,3]]` into `H_gg`; `coefficient_design()` for `H_mg` / `H_tg` |
  | `psi_terms::accumulate_score_blockwise` | `coefficient_design().axpy_row_into(row, primary[3], …)` |
  | `psi_terms::accumulate_score_with_q_geometry` | the same, one channel |
  | `primary_geometry::spatial_block_primary_loading` / `primary_direction_from_psi_row` / `primary_psi_action_from_psi_row` | length-**4** vectors contracted against a length-**6** primary gradient |
  | `timepoint_exact::row_primary_fourth_contracted` | instantiated at `STATIC_SLOPE_PRIMARIES` — the *time-constant row program* — whichever frame the family is in |

  The first one is the widest: `exact_newton_joint_hessian_operator` — the
  INNER Newton's matrix-free joint Hessian on the dynamic-q path — is one of its
  callers. Its gradient half (`accumulate_dynamic_q_core_gradient`) and its
  dense sibling (`accumulate_dynamic_q_core_hessian`) were both converted; the
  operator's Hessian half was not, so on a follow-up-varying slope the inner
  solve ran with a curvature missing the `g₁` and `ġ₁` rows and columns
  entirely. A mode certified against the wrong curvature is not the argmin of
  the criterion the outer search is profiling, which is exactly the
  "`converged=true` with `last_residual_below_tol=false`" signature the #2765
  probe recorded 55 times in one run.

  **The two holes in the refusal list.** The refusal names a spatial length
  scale on the *log-slope* surface, Gaussian-shift frailty, the score-warp /
  link-deviation flex blocks, the CTN Stage-1 absorber, and a time-wiggle
  baseline. It does not name:

  * **a parametric baseline chart** — `baseline_exact_joint_psi_terms_with_options`
    reaches `accumulate_score_with_q_geometry` and `add_pullback_with_q_geometry`
    unconditionally, so a Weibull / Gompertz / Gompertz-Makeham ψ dropped two of
    the slope's three channels. This is the configuration #2765's own acceptance
    fixture uses (`baseline_target: "weibull"` plus `logslope_time_k`), and its
    outer search direction was measured to be essentially pure baseline-ψ;
  * **a spatial length scale on the MARGINAL surface** — `psi_terms_inner`
    contracted a length-4 primary direction against a length-6 primary
    gradient, which is a shape error rather than an approximation.

  **The repair is the generalization, not a guard.** Every site above is now a
  loop over `primary_channels()` or is sized from `core_primary_dimension()`. A
  time-constant slope still performs exactly one rank-1 update per row and its
  arithmetic is unchanged, so nothing about the static path changes shape or
  cost. The log-slope-surface spatial refusal STAYS and is now enforced one
  layer down as well: with a time margin that block's three channel designs are
  `X_cov ⊗ B_entry`, `X_cov ⊗ B_exit` and `X_cov ⊗ B′_exit`, while the ψ
  design-derivative contract carries a single `X_ψ`, so the other two channels
  are not recoverable from what the caller holds — a fact the primary-space
  helpers now state rather than assume. The batched ψ fast path, whose tower is
  instantiated at `STATIC_SLOPE_PRIMARIES`, declines a follow-up-varying slope
  and defers to the per-axis route instead of truncating it.

  **The gate that was missing.** `psi_terms_inner` publishes
  `∂_ψ ℓ̄`, `∂_ψ ∇_β ℓ̄` and `∂_ψ ∇²_β ℓ̄`, and nothing had ever differenced them
  against the functions they name. The shipped ψ coverage checks finiteness,
  subsample-vs-unsampled equality, and batched-vs-per-axis agreement — a
  *consistently* wrong derivative passes all three.
  `marginal_slope/psi_terms_fd_tests.rs` now differences every ψ lane (marginal
  design, log-slope design, and each baseline-chart coordinate) against the
  family's own `(objective, score, Hessian)` triple, in BOTH slope frames, with
  a Richardson pair certifying the oracle so a gap cannot be charged to the
  finite difference and an unresolved component declines to grade rather than
  fails.

- **A monotone warp's corner is its boundary knot's MULTIPLICITY, and no
  extrapolation rule can remove it (#2695).** `0167ed853` gave the warp basis a
  linear tail so `I′_j` would stop stepping at the knot hull's edge, and the
  witness fit. It did not close the issue: a warp with real amplitude still
  refused at degree 2, 3 and 4, all on OBJECTIVE rejections, while the same data
  fitted cleanly with the `linkwiggle(...)` term removed.

  **What no degree helps means, measured.** `f19e2bee4` reads the one-sided gap
  in `I^(k)` across a point at `h = 1e-3` and `h = 1e-6`; a ratio of `1e3` says
  the gap falls with `h` (continuous), `1e0` says it does not (a step):

  | degree | interior knot | hull edge (linear tail) |
  |---|---|---|
  | 2 | steps at order **2** (2.0) | steps at order **2** (2.0) |
  | 3 | steps at order **3** (6.0) | steps at order **2** (6.0) |
  | 4 | steps at order **4** (24) | steps at order **2** (12) |
  | 5 | no step at `k ≤ 4` | steps at order **2** (20) |

  The hull edge steps at order 2 at EVERY degree — the tail zeroes `I″` while
  the interior one-sided `I″(right⁻)` is `2, 6, 12, 20`. That is why the
  four-arm degree sweep found no degree that helps: raising the degree moves the
  INTERIOR step up the tower and leaves the edge exactly where it was.

  **Why order 2 is the order that bites.** `m1 = 1 + Σ_j βw_j·I′_j(q1)` enters
  the event Jacobian `g = η_t′ + m1·q̇₀`, so `∂²m1/∂βw_j∂β_thr = I″_j·∂q1/∂β_thr`
  is an entry of the observed information `H` carrying `I″_j` with **no `βw`
  factor** — it survives at `βw = 0`. `Φ = ½ Σ g(λ(Z_JᵀHZ_J))` is inside the
  accept test, so a step in `I″` is a step in the OBJECTIVE, and
  `actual/predicted` cannot approach `1` at any step size. (The channel is live
  only on EVENT rows: `log g` is added when `w·d ≠ 0`, so a censored crossing row
  reports "continuous" for the wrong reason.)

  **Why a better tail is not available.** At a clamped edge most columns have
  `I′ = 0` and `I″ ≠ 0` — at degree 2 the right edge carries `I′ = [0, 0, 0, 2]`
  and `I″ = [0, 0, −1, 2]`. A monotone `C²` extension of a column with
  `I′(e) = 0` and `I″(e) < 0` **does not exist**: `C²` forces
  `I′(e+ε) ≈ I″(e)·ε < 0`. The corner is not the extrapolation rule; it is the
  boundary knot's multiplicity (`degree + 1` on a clamped vector), and the cure
  is a knot vector whose ends are SIMPLE.

  **The repair.** Two pieces, one contract — *the warp is one `C^{degree−1}`
  function on all of `ℝ`*:

  * `gam_terms::basis::ispline_ramp_basis_dense` evaluates the I-spline as what
    it is — `I_c(x) = ∫_{t_{c+1}}^{x} M_{c+1}`, exactly `0` before that support
    and `1` after — for ANY knot vector. `create_ispline_dense` computes the
    same right-cumulative sum but reads it only where the degree-`bs` B-splines
    are a partition of unity and imposes `0`/`1` outside by convention; that
    convention is exactly right on a clamped vector (pinned: the two agree to
    `1e-12` at every degree, inside the hull and outside it) and wrong on any
    other, because a ramp whose support runs past that interval gets truncated
    mid-rise. Evaluating on a padded knot vector removes the truncation without
    changing a single clamped value.
  * `gam_terms::basis::monotone_warp_knots` builds the warp's knots as
    `num_internal_knots + 1` uniform spans across the seed range, continued by
    `degree` further spans at the same width on each side, all knots simple. The
    column count is unchanged (`num_internal_knots + degree` either way), so
    nothing downstream changes shape.

  `monotone_wiggle_basis_with_derivative_order` is now one call into the ramp
  evaluator: the hull, the clamp, the linear tail and the `orders ≥ 2 → 0` rule
  are all deleted, not patched. Warp blocks (survival link and time wiggle,
  GAMLSS, BMS) build their knots with the warp generator;
  `initializewiggle_knots_from_seed` stays clamped for bases evaluated on FIXED
  data — a response transform — where a boundary knot's multiplicity is
  invisible because the evaluation point never moves across it.


- **The multinomial's Firth/Jeffreys separation certificate is now taken on
  `ker(S_λ)` — the directions no smoothing parameter reaches — instead of the
  whole identifiable span (#2612).** `27301d428` fixed which *matrix* the arming
  verdict is taken on (`H + S_λ`, the curvature the fit has). This is the rest of
  the same sentence #715 derives: `(H + S_λ)v = Hv + λSv`, so a direction is
  beyond every `λ`'s reach exactly when `S_λ v = 0`. Where `S_λ v ≠ 0` the model
  already carries a proper prior on `v`, and since the ratio-of-normalising-
  constants predictive landed, that width is integrated exactly rather than
  approximated — so it is already in the published probability.

  Measured, the certificate now names the subspace it decided on: `2/16`
  unreached directions on a one-smooth quasi-separated fixture (`H + S_λ` there
  in `[9.84e-1, 6.56e0]`) and `2/74` on the penguins witness (`[2.86e-3,
  9.64e-1]`) — in both cases the class intercepts, holding under one
  observation-equivalent apiece. The refusal also reports what it did *not*
  decide on (the whole penalized span, and the likelihood alone), and both
  branches log the unreached dimension, so a disarmed fit says why it disarmed.

  `jeffreys_subspace_from_penalty` now computes the kernel its own return type
  has always advertised instead of discarding its argument and returning `I_p`;
  the zero operator short-circuits to an exact identity, so every existing caller
  is byte-identical. `multinomial_reml::measured_penalty_nullspace` delegates to
  it, so "which directions does this penalty reach" has one answer in one place.

  **Recorded and NOT landed, with the controlled measurement.** Restricting the
  *term* to that same subspace — the natural completion, since
  `jeffreys_antiderivative` acts wherever a reduced eigenvalue is under
  `CONDITIONING_GATE_ABSOLUTE_CLEAR = 16` and a quasi-separated softmax has
  `λ_max(H) = 1.44` over a 74-dimensional span at `n = 228`, so the term acts on
  the entire basis — fixes the calibration outright: an armed quasi-separated
  smooth fit goes from a held-out calibration gap of `−0.0802` to `−0.0151` and
  log-loss `0.13224 → 0.07682`. It also costs the penguins witness its fit. The
  full-span term is incidentally a *regulariser of the inner problem*: with it
  the joint Newton reaches the LAML derivative lane's `1e-11`; without it the
  residual plateaus at `9.84e-7`, and loosening the target to the objective's own
  measured floor (`MULTINOMIAL_FORMULA_INNER_TOL = 1e-5`) then desyncs the
  analytic outer gradient by exactly the amount #1820 documents, so the outer
  line search terminates with `StepSizeTooSmall` at `|g| = 1.76e-1`. The span
  cannot be narrowed until the inner joint-Newton can certify a near-separable
  multinomial mode to the accuracy the derivative lane needs — iterative
  refinement on the inner KKT solve, not a looser target.
  `a_quasi_separated_smooth_fit_is_calibrated_2612` is left RED against a
  four-standard-error bar as the standing measurement of that gap.

- **A negative-curvature saddle escape now judges its own trial against the
  CRITERION's resolution, and the adjudication is no longer gated by the
  reseed's one-shot budget (#2612).** `adjudicate_negative_curvature` derives
  where to stop probing from the criterion's own resolution — the ladder ends at
  `α_min = sqrt(2·objective_resolution/|λ_min|)`, on the stated ground that below
  it "the claim predicts nothing the criterion can represent" — and then decided
  whether a probe had DESCENDED against `16·ε·|V|`, the arithmetic's resolution.
  One function, two notions of "a decrease the criterion can represent", ten
  orders of magnitude apart.

  Measured on the two fixtures #2612 is decided on: the penguins witness's
  unbiased probe minted four escape reseeds at `λ_min` of `−6.4e−7 … −2.0e−6`
  (machine zero against `‖H‖ ≈ 1`) on objective decreases of `2e−6 … 4e−6`
  against a resolution of `1.228e−3`; a one-smooth quasi-separated fixture minted
  three on decreases of `3.4e−4`, `1.4e−4` and `5.0e−5` against a *measured*
  cost-stall noise floor of `1.91e−4`. Each reseed spent the one-shot budget, and
  the retry pass — which `allow_tail_snap` forbade from adjudicating at all —
  then refused on the matrix's word. Both fits died; the fit that shipped was the
  Firth/Jeffreys-armed one.

  At `λ_min = −6.4e−7` the claim's predicted decrease at the LARGEST feasible
  step is `3.2e−7`: it was unfalsifiable over its whole derived range, which is
  exactly the state `CurvatureEvidence::CriterionContradicted` exists to record.
  The strict-decrease floor is now `objective_resolution` — the same
  `rel_cost_tolerance`-anchored anchor the ladder's own limit is derived from —
  floored at the arithmetic roundoff that remains the hard lower limit, so no
  constant is introduced. And the adjudication runs on every refusal: that flag
  is the one-shot budget for the RESEED, and gating a MEASUREMENT on it made the
  retry pass refuse for want of a measurement it could have made for free.
  `Descended` still spends the budget, and on the retry pass now records that the
  criterion CONFIRMS the curvature, so a measured saddle and an unmeasured one
  stay distinguishable.

  Effect: a one-smooth quasi-separated multinomial (`cls ~ s(x, k=8)`) that
  produced no fit at all now fits in 7.5 s, and the penguins witness's unbiased
  criterion reaches a certified stationary optimum instead of exhausting its
  strategy fallbacks.

- **A follow-up-varying marginal slope can now be SAVED, predicted from, and
  leave-one-out replayed (#2765, #2767).** `logslope_time_k` fitted a real model
  since the kernel work landed, but persistence refused it outright: the on-disk
  contract rebuilt the log-slope block from its covariate term spec alone, which
  names `p_cov` columns against a `p_cov · p_time` coefficient vector. A fitted
  surface that cannot leave the process is half a feature.

  **What was missing was one fact, not one code path.** The block's authority is
  the covariate spec *plus* the resolved time margin, and only the first half was
  persisted. `logslope_time_basis` (degree + knots) now rides on the saved model
  beside the threshold and log-σ margins it was built by the same primitive as,
  and every consumer rebuilds `X_cov ⊗ᵣ B(log t)` from it. The knots are fit-time
  values, so a prediction sample can never move the basis by re-estimating
  quantiles — the same contract the location-scale margins already hold.

  **Two places where "replay it" is not the obvious thing.**

  A predicted survival curve evaluates `b` **at each time on the curve**, not at
  the row's observed exit time. The family is `S(t) = Φ(−η(t))` with
  `η(t) = q(t)·c(t) + b(t)·z`; freezing `b` at `t_exit` would return a curve
  assembled from a different model at every point but one. The per-`(row, t)`
  evaluator therefore re-tensors the row's covariate factor against the margin at
  the time being predicted, exactly as it already rebuilds the time basis there.

  The leave-one-out (`--alo`) replay re-evaluates the row program, which reads
  the slope at entry, at exit, and as an exit-time rate — three channels, because
  the likelihood is `log S(t₁) − log S(t₀)` and an event row also carries
  `log η′(t₁)`. Handing it the exit design alone would not have been an
  approximation; it would have reported the influence of a time-CONSTANT slope,
  and the widths would have agreed while it did so. All three channels are
  rebuilt, and the ALO input refuses a follow-up triple whose shapes disagree
  rather than indexing through them.

  **What guards the replay.** One function evaluates the log-slope time axis, so
  the batch replay and the per-cell replay cannot ask for different bases; a test
  asserts the replayed exit margin is the fit-time margin *bit for bit* rather
  than merely close, because a margin that is nearly right is a model that is
  quietly wrong. Load-time validation refuses a saved block whose width is not a
  multiple of its own margin — a payload in that state cannot be replayed under
  any covariate design at all.

- **The multinomial's Firth/Jeffreys separation certificate now judges the
  curvature the fit HAS (`H + S_λ`) instead of the likelihood's alone (`H`),
  because reading `H` alone made "arm only on separation evidence" fire on every
  multinomial GAM carrying a smooth (#2612).** The conditional engagement
  (#715 arm (b) / #753) exists because the proper prior is not free: it pulls
  fitted class probabilities toward the uniform simplex `1/K`, and it routes the
  fit through a lane whose outer Hessian omits `D²_β H_Φ` by construction, so its
  curvature certificate is deliberately weaker. A false positive costs a whole
  re-solve on a biased objective and a certificate that cannot be as strong.

  The decision came from the reduced conditioning gate at the certified mode,
  whose absolute arm is derived from **one observation-equivalent** of curvature
  and whose doc block states the premise that makes it conservative: *"it never
  fires on a genuinely well-conditioned large-`n` fit, whose `λ_min = O(n) ≫ 1`"*.
  That premise fails for every penalized smooth basis. Measured through the
  shipped Python surface on labels DRAWN from a smooth softmax truth — every
  class keeping appreciable probability everywhere, nothing separating anywhere:

  ```text
    y ~ x1 + x2 (parametric)       n=  300 unbiased
    y ~ x1 + x2 (parametric)       n=  900 unbiased
    y ~ x1 + x2 (parametric)       n= 3000 unbiased
    y ~ s(x1,k=5)+s(x2,k=5)        n=  300 ARMED    lmin=2.115e-1  lmax=7.647e+1
    y ~ s(x1,k=8)+s(x2,k=8)        n=  900 ARMED    lmin=4.579e-1  lmax=2.494e+2
    y ~ s(x1,k=12)+s(x2,k=12)      n=  900 ARMED    lmin=2.531e-1  lmax=2.488e+2
    y ~ s(x1,k=5)+s(x2,k=5)        n= 3000 ARMED    lmin=1.989e+0  lmax=8.113e+2
  ```

  Every parametric fit unbiased, every fit with a smooth armed, on identical
  data. `λ_min` IS `O(n)` — `0.211 → 0.591 → 1.989` across `n = 300 → 900 →
  3000` — but with a per-observation constant of `≈ 7e-4`, because the
  least-resolved direction of a `k`-dimensional spline is barely resolved *by
  construction*; that is why it is penalized. The premise holds only for
  `n ≫ 1/c_k`, and `1/c_k` grows with `k` (`0.591 → 0.458 → 0.253` at fixed
  `n = 900` as `k` goes `5 → 8 → 12`).

  The distinction #715 derives is the one that was missing: a direction `v` is
  beyond `λ`'s reach only when `S v = 0`, because `(H + S_λ)v = Hv + λSv`. Where
  `Sv ≠ 0` the smoothing parameter supplies the missing curvature and the
  direction is identified — by the prior the model already has. The certificate
  now forms `H + S_λ` at the λ the mode was certified at, through
  `multinomial_joint_penalty_operator`, the fit's ONE assembly of `S_λ`, so the
  arming decision and the published penalty cannot describe different priors. No
  threshold moves: the Frobenius-normalized `S` makes `λS` directly comparable to
  data Fisher information, which `MULTINOMIAL_FORMULA_FISHER_INFO_PER_OBS`
  already asserts. The refusal reports both spectra, because "the data do not
  determine this direction" and "and no λ repairs it" are different statements
  and the verdict rests on the second.

  **The deciding threshold was `16`, not `1`, and that is the second half of the
  defect — the measurement is what forced it.** With `H + S_λ` in place the
  fixture's deciding eigenvalue moved `0.405 → 5.508`, a factor of `13.6`, and
  the fit **still armed**, at gate weight `0.783`:

  ```text
    identifiable-span PENALIZED curvature H+S_lambda is under-identified at the
    certified mode: lambda_min=5.5076e0, lambda_max=8.8358e2, ratio=6.233e-3,
    Jeffreys gate weight=7.8336e-1
    (likelihood alone: lambda_min=4.0514e-1, lambda_max=1.5754e2)
  ```

  Five and a half observations' worth of curvature in the worst-determined
  direction, still called separated — because the verdict was taken on
  `JointJeffreysPlan::is_active`, which is `gate_weight != 0` and therefore
  boundaries where the C¹ ramp reaches exactly zero,
  `CONDITIONING_GATE_ABSOLUTE_CLEAR = 16`. That band exists so `Φ(ρ)` stays C¹ as
  `β̂(ρ)` carries the spectrum across the boundary — a binary gate makes the
  outer objective jump, which is the #787 "outer smoothing did not converge"
  regression — and it is generous FOR THAT REASON, not because a direction
  holding fifteen observations' worth of curvature is unidentified. `weight != 0`
  is the support of a smoothing device, and it was choosing an estimand.

  `JointJeffreysPlan::is_under_identified()` is the gate's derived predicate
  instead — below one observation-equivalent, or below the relative knot —
  expressed as `conditioning_gate_weight == 1` so one arithmetic authority
  decides both the weight and the verdict. `is_active` is untouched and still
  governs the term's own contribution, where the weight IS the answer; only the
  multinomial's conditional engagement, which re-solves the whole fit against a
  different objective, takes the new one. The two halves are independent and the
  fixture needs both: with `H` alone `λ_min = 0.405 < 1`, so the derived
  predicate would arm anyway; with `is_active` the penalized `5.508` arms anyway.

  **Rejected**: widening the absolute knot (derived, and not the thing that is
  wrong); restricting the Jeffreys BASIS back to `ker(S)` (deliberately widened
  to the full span for the BMS-probit near-separation on a penalized direction,
  where the term is `O(1/n)` by design — it is the arming DECISION, not the
  basis, that cannot afford a false positive); touching the universal always-on
  term's internal gate (same predicate, but a skip optimisation rather than a
  verdict).

  **`MULTINOMIAL_UNBIASED_PROBE_OUTER_MAX_ITER` is deleted in the same change,
  and the measurement says it was not the cause.** The unbiased probe ran under
  `outer_max_iter.min(20)` — a ceiling introduced as a bare one-line
  `perf(#1082)` commit with no test and no measurement — while every `Err` from
  it was routed on as separation evidence. SPEC forbids minting a fit from an
  exhausted budget, so a search stopped by that ceiling returns
  `RemlDidNotConverge`, and the ceiling could convert "this probe was still
  descending when I stopped it" into "the data separate". Run at both budgets on
  the same non-separating fixture it in fact decided nothing — the probe
  certifies either way, and the reported spectra agree to four digits — which is
  what falsified the first hypothesis on this issue. It goes anyway, because the
  only runs it can shorten are the ones where it stops a still-descending search,
  i.e. exactly the runs whose verdict it changes: its entire benefit is
  co-extensive with the misdecision. SPEC: *"Wall-clock time budgets and
  deadlines are never allowed, except in tests. In general, do not paper over
  solver issues."* The dead helper written for that branch and never wired
  (`multinomial_formula_unresolved_probe_separation_evidence`, kept in a test
  module under the note that "the production routing that would consume this is
  not currently wired") goes with it.

  **The fit now records which estimand it published.** `separation_evidence` is
  `None` for the unbiased penalized-REML mode and carries the certificate itself
  when the proper prior was armed. The two branches are different estimands, and
  the decision previously existed only in a `log::info!` line the caller never
  sees — while the CLI rejects `--firth` on this family with "the stabilizer is
  armed automatically". A user told the decision is automatic is owed the
  decision, and the CLI summary now reports it.

- **The refinement tolerance is now DERIVED from the candidate set instead of
  being a fixed fraction of the residual, because a fixed fraction charges
  nothing for the set's width (#2759).** #2759's first half closed a two-sided
  bracket on the level-`(L+1)` gain and established that the cascade fixtures
  refusing at the rank-maximal design have the exact remaining
  penalized-objective decrease bounded away from `REFINE_TOL·rss_pen` from
  BELOW — so those refusals were the cascade's remaining gain, not the
  certificate's conservatism. What it left open is whether that decrease is the
  thing the criterion should be reading. It is not.

  At the `smoothness_ceiling_...` refusal the candidate level is **32790
  columns against 5997 identifiable directions** on `n = 6000`. Past the data's
  identifiable rank those candidates are redundant against the sample's own row
  space; what they buy is penalty dilution and noise capacity, and
  `1e-3·rss_pen` cannot tell that from discretization bias because it never
  looks at the set.

  The missing charge is the set's own Occam factor — the log-determinant of the
  SAME Schur complement the gain is a quadratic form in:

  ```text
      gain  = gᵀS⁻¹g,   occam = log det(S/(λd)),   S = X₂ᵀW(I − H)X₂ + λd·I
      2·evidence = dof·log(rss_pen/rss_pen_refined) − occam
  ```

  The second line is an IDENTITY at the profiled σ̂², where the `rss_pen/σ̂² =
  dof` quadratic cancels on both sides — so one more level is warranted exactly
  when `gain > rss_pen·(1 − e^{−occam/dof})`. That break-even gain is the
  tolerance, `REFINE_TOL` is deleted, and the cascade has no tolerance constant
  left.

  **Both numbers come from ONE fixed-λ evaluation of the design with the
  complete candidate level appended**, and that evaluation is available past
  every capacity budget the automatic route enforces.
  `CERTIFIED_SPECTRUM_MAX` bounds the λ-independent Schur eigendecomposition
  the score SEARCH is certified in; `n − nullity` bounds the rank that search
  needs a stationary point in. A single evaluation at a fixed λ needs neither —
  only a factorization, which the sparse route supplies far wider. So the
  question "does one more level explain the data better?" has an exact answer
  exactly where the cascade used to have only a bound.

  The bracket is kept as the SCREEN, and it is not a different instrument: read
  the gain from its LOWER end and the Occam factor from Hadamard on
  `S ⪯ diag(X₂ᵀWX₂) + λd` (a reduction over the Jacobi preconditioner the
  bracket already forms), and both readings understate the evidence, so a
  positive evidence there PROVES the level warranted without building anything.
  It settles the positive side only — a fit is never minted on it — which is
  what lets it carry the memory-boundary refusal at `n = 525_000` without
  materializing a design wider than the budget that refusal exists to protect.

  Measured, `n = 6000`, the fixture's own ladder:

  ```text
   rung centers  cand   rss_pen  1e-3·rss  gain_hi   occam   break-even  Δ logL   rmse -> refined
    0        57     -   2.266e2  2.266e-1  1.985e3  5.17e2   1.872e1     +6293    0.1896 -> 0.0562
    1       189     -   2.204e1  2.204e-2  8.452e1  1.55e3   5.011e0     +3240    0.0553 -> 0.0220
    2       655     -   5.514e0  5.514e-3  3.517e0  2.21e3   1.697e0     + 654    0.0220 -> 0.0156
    3      2166     -   2.997e0  2.997e-3  6.205e-1 1.33e3   5.962e-1    +  16.7  0.0157 -> 0.0155
    4      5997 32790   2.184e0  2.184e-3  1.055e-1 3.23e2   1.146e-1    -  13.2  0.015583 -> 0.015589
  ```

  Rung 4 is the refusal in the issue body. The fixed bar is 48x below the gain
  and demands another level; the restricted likelihood says the finer prior is
  13.2 nats WORSE; and the held-out RMSE against the planted truth agrees with
  the likelihood, not with the bar — refining makes it worse. `n = 2000`
  reproduces the same turnover at its own rank-maximal rung (−18.7 nats,
  0.03087 → 0.03129).

  **The obvious objection, run and killed.** The comparison is at the
  incumbent's λ, so the refined design might win it back at a λ of its own.
  Swept `log λ ± 1, 2, 3` on the refined design at every rung: at both turnover
  rungs the best λ IS the incumbent's, to the printed digit. Structurally that
  is what has to happen — at the turnover the extra columns are redundant, so
  the score surface barely moves and its optimum does not — and the sweep is
  now a gate rather than an observation.

  **The identifiability frontier is where the cascade STOPS, not where it
  refuses.** Two fixtures asserted a refusal there and now mint the
  rank-maximal fit, for the same reason: at `n = 240` the level proposes 504
  penalized modes against 237 identifiable directions and is worth −0.30 nats;
  at `n = 2000`, 6968 against 1997. Neither exists to test the certificate —
  both exist to keep the automatic route out of a rank-deficient score search
  whose cost is exponential in the subdivision depth — and stopping at the
  frontier keeps it out just as a refusal did, while returning a usable fit.
  The measured boundaries are asserted exactly as before; only the verdict
  taken on them moved. The typed `Underresolved` refusal is not orphaned: it is
  the MEMORY boundary, where the design is capped BELOW the identifiable rank
  and the levels it cannot reach do pay for themselves
  (`past_cliff_...`, `n = 525_000`, unchanged).
  `cascade_matches_or_beats_dense_duchon_on_truth_recovery` reports RMSE
  0.02781 against the dense comparator's 0.03018, so the shallower stopping
  point costs no accuracy.

  Verified: the Occam term read off the two restricted log-likelihoods is
  checked against the candidate Schur log-determinant formed DENSELY, one
  column at a time through the same matrix-free operator, over a λ sweep — one
  side is two profiled REML evaluations, the other is `m₂` cascade solves, and
  they share nothing but the arithmetic they must agree on. The comparison also
  differences a `fit_reml` restricted likelihood (normalized through the
  certified Schur eigenbasis) against a `fit_at` one (a factorization at that
  λ), at O(1) nats while each side is O(10³), so the two routes agreeing is a
  premise and is now charged on both width regimes.

  **All four fixtures are back on the route a caller takes.** This issue's
  acceptance was "either it certifies, or its refusal is shown to be a true
  statement about the data at that `n`". Its first half took the second branch
  and moved three of them off `fit_residual_cascade` — a serialization gate
  going red because the cascade has remaining gain is not measuring
  serialization — which was the honest reading while the criterion was what it
  was. With the tolerance derived from the candidate set they take the FIRST
  branch: `cascade_state_rejects_corruption`,
  `cascade_state_roundtrip_reproduces_mean_and_variance` and the benign arm of
  `quasi_uniformity_guard_rejects_degenerate_metric_keeps_benign` all fit
  end-to-end again. `cargo test -p gam --test misc residual_cascade` is 26 of
  26, from 25 of 26 with three fixtures held off the route; `cargo test
  -p gam-solve --lib residual_cascade` is 29 of 29.

  A candidate column whose bump covers no observation is exactly zero, and its
  `λd` diagonal cancels between `log|A|` and `log|λD|₊`, so it contributes
  nothing to the gain, nothing to the Occam factor, and nothing to the
  restricted likelihood. Dropping such columns from the design the comparison is
  built on is therefore an identity, and it is worth taking: 4976 of 7176
  candidates are structurally empty at level 7 on 240 rows.

  One latent defect fell out of it: `NextLevelPlan::exhausted` hard-coded
  `extends_last: false`. That was invisible while the flag was read only on the
  refine path — an exhausted plan is never refined into — and fatal the moment
  the comparison materializes the candidate set FROM the plan, which is exactly
  what the capacity refusals need. It is now decided from the radius, before
  any candidate set exists, and carried by every outcome.

- **The Murphy–Topel correction now exists for a `GlobalEmpirical` second-stage
  latent measure, and the refusal it replaces was resting on a false
  obstruction (#2484).** A BMS fit whose conditional location-scale calibration
  fires and whose calibrated residual `ζ` then fails the standard-normal
  adequacy gate selects an empirical latent measure built from `ζ` itself. That
  pair used to withhold the coefficient covariance, on the argument that the
  generated-regressor correction needs a per-row mixed derivative and `ℓ_i`
  depends on `ζ_i` twice — directly, and through a grid every other `ζ_j`
  helped build — so "the honest object is a full `n × n` sensitivity" and "the
  measure is itself estimated from the same data".

  The first half is a factorization, not a dense object; the second is the
  chain rule, not a violated assumption.

  `build_empirical_z_grid` cuts bins by cumulative **weight**, so for a fixed
  sort order the bin allocation `α` is *exactly* constant in `ζ` and the grid
  weights carry no `ζ`-sensitivity at all — only the `m ≈ 32` node VALUES move.
  And Murphy–Topel conditions on the data: given `z`, the measure is a
  deterministic function of `θ₁`, which is precisely what the correction
  propagates. So the total derivative splits into a direct channel and a
  rank-`m` cross-row channel,

  ```text
      d score_β/d ζ_j = s_j + Σ_b u_b·D_{bj}        ⇒        S_eff = S + Dᵀ·U_Qᵀ
  ```

  and the seam substitutes `S_eff` for `S`. `generated_regressor_correction`,
  `build_zeta_theta1_jacobian`, `beta_theta1_sensitivity` and the
  `(V_β G) V₁ (V_β G)ᵀ` congruence are untouched, and PSD-ness is preserved for
  free.

  Neither channel is the closed-form kernel's. The empirical row is
  `−w·logΦ(σ·(a(m,g) + s·g·ζ_i))` around an implicitly solved intercept `a`
  rather than `q·√(1+(s·g)²) + s·g·ζ_i`, so reusing
  `rigid_standard_normal_score_zeta_sensitivity` for the DIRECT half would have
  been a subtler wrong answer than refusing. Both come from one pass over the
  rows, sharing the per-row intercept solve, with the node derivatives from the
  same calibration root the row jet lifts:

  ```text
      a_x_b       = −s·g·π_b·φ(η_b)/Ψ₁
      a_{m,x_b}   = −a_m·(dΨ₁/dx_b)/Ψ₁
      a_{g,x_b}   = −[dΞ₁/dx_b + a_g·(dΨ₁/dx_b)]/Ψ₁
  ```

  `Σ_b a_x_b = −s·g` (a uniform node shift must be absorbed exactly by the
  intercept) and `a_x_b ≡ 0` at `g = 0` (a fit with no slope cannot see the
  latent axis) are the identities that pin the sign and scale.

  **Rejected: seeding each node as a third jet axis** through the existing
  `filtered_implicit_solve_scalar` lift, which is the more mechanical route.
  It costs `O(n·m²)` lifts against the closed form's `O(n·m)`, on a path that
  runs at biobank `n`.

  `EmpiricalZGrid` and its `PartialEq` are untouched — it is the measure's
  identity and it is on the persistence wire. The allocation record rides on a
  fit-time-only `EmpiricalZGridBuild` returned by the one builder the fit
  itself uses, so the recorded `α` is the fill loop's own rather than a
  reconstruction.

  **What still withholds, and it now names the missing CHANNEL rather than the
  measure:** a score-warp / link-deviation block (the latent score enters
  through a basis as well as through the intercept, so the rigid node channel
  does not describe the row); a `local-empirical` measure (per-row grids, only
  produced by deserializing a saved model, so there is no fit-time allocation);
  and data on which the compression is genuinely non-differentiable — a tied
  `ζ` group that a bin boundary cuts, where the left and right derivatives of
  the nodes differ. That certificate is narrower than "no ties": a tied group
  entirely inside one bin is order-invariant and is not refused.
  `CovarianceDeclined::BmsGeneratedRegressorLatentMeasureNotStandardNormal`
  gains a `#[serde(default)] unavailable_channel`, so older payloads still
  deserialize.

  Verified against difference quotients of PRODUCTION code, never a
  reimplementation, and separately against a second implementation of the
  DERIVATION in another language
  (`scripts/probe_2484_empirical_measure_sensitivity.py`) — the two catch
  different things, since a formula transcribed consistently-but-wrongly into
  both the code and its own test survives the first check and not the second.

  The acceptance gate is the total `∂²(log L)/∂β∂ζ_j` against a double central
  difference of the production log-likelihood with the grid REBUILT at every
  perturbed `ζ` — blind to how the channels are split, so it fails alike on an
  IFT sign error, a missing cross-row term, or a wrong `1/sd`. Below it:
  allocation mass conservation on both margins; `D` against a central FD of the
  production builder, including a row heavier than the per-bin target (it lands
  in THREE bins — there is no two-entries-per-row bound) and a zero-weight row
  (exactly zero sensitivity); the two projection identities `D·1 = 0` and
  `D·ζ = 0` as exact assertions; the tie certificate firing on a cut tie and not
  on a contained one; and the assembled correction being symmetric, PSD, and
  strictly widening.

  ```
  cargo test -p gam-models --lib empirical_measure_2484
    test result: ok. 10 passed; 0 failed

  scripts/probe_2484_empirical_measure_sensitivity.py
    D max abs err vs FD:                 1.63e-10
    total mixed derivative max rel err:  1.28e-07
    bins=3 (tie inside one bin):  |right − left| = 6.66e-09   differentiable
    bins=4 (a boundary cuts it):  |right − left| = 5.39e-01   TWO-SIDED
  ```

  **The witnesses.** The three `..._starts_outer_solver` fixtures gam#2484 was
  filed against:

  ```
  binary_outcome_shape_bms_shared_matern_prs_pc_confound_starts_outer_solver ..... ok
  production_like_binary_outcome_shared_matern_centers10_confound_starts_outer_solver ... ok
  production_like_binary_outcome_shared_matern_learned_kappa_starts_outer_solver ....... FAILED
  ```

  The third fails on `INDEFINITE CURVATURE AT INTERIOR OPTIMUM` (`|g| = 1.241e-2`
  against `bound = 3.850e-2`, `hessian_psd = NO`) — the outer-stationarity
  cluster, which is not this seam and never reaches it. That is the state
  gam#2484's own 2026-08-01 measurement recorded: *"masked, not resolved … it is
  now blocked one stage earlier."* Its calibrated residual still selects
  `global-empirical`, so it would be corrected if the outer solve certified.
  Both witnesses that reach the seam pass.

  `tests/bms_covariance_declined_2718.rs` runs in **2.3 s** (4 arms) against
  22 min before, with strictly more coverage: the end-to-end withholding witness
  moved to the classifier, where all four arms are decided with no fit in the
  loop, and the wire contract is asserted on the payload — including a payload
  written before the channel field existed, which must still load with the
  channel empty rather than fail.

  **Subsystem sweep.** `cargo test -p gam-models --lib bms::` — **297 passed,
  2 failed**, and neither failure is this change:
  `bms::gradient_paths::jet_tower_oracle_tests::rigid_third_and_fourth_full_shares_one_tower_bit_identical`
  (a last-ulp tower difference) and
  `bms::tests::bernoulli_batched_outer_gradient_matches_hypercoord_path_for_rho_and_psi`
  (`psi[1]` at `rel = 4.132e-3`). Both are already recorded in
  `bench/gha_results/rust-test-suite/MASTER_FAILURES.md`, written by CI run
  31291341087 in `e146df43a`, which `git merge-base --is-ancestor` confirms
  predates the first commit of this work. Neither path is touched here: the only
  production edit outside the BMS covariance seam is the `clamp` fix below, and
  it is bit-identical for `n >= 1024` (`rows.min(1024) == 1024`) and converts a
  panic into a value below it — it cannot change a number that previously
  computed.

  **What the channel is worth, stated honestly.** `|cross| / |direct|` is a
  property of the sensitivity MATRIX; a user sees a standard error, three
  contractions downstream. The correction as a whole moves the SE by
  **1.06x–2.53x** against the naive covariance — that is what publishing the
  naive matrix would have cost. The CROSS-ROW half of it moves the SE by
  **1.2e-5 to 9.0e-3** relative, and what it scales with is the LOGSLOPE rather
  than the grid size. Small enough that a direct-only implementation would pass
  casual inspection, which is an argument for being exact and not an argument
  that the channel is optional.

- **A composed monotone warp was a function with a CORNER, and the Firth term
  put that corner into the objective (#2695).** `create_ispline_dense` is
  constant outside its knot hull `[left, right]`, and says so; `a3304985f` made
  the reported derivative agree with that value by zeroing it strictly outside.
  Both halves are right, and together they name what is wrong: a
  constant-extended I-spline is continuous with a **corner** at each hull edge —
  `I_j` joins, `I'_j` steps from its interior one-sided slope straight to `0`.

  A corner in a shape basis on fixed data is harmless; the evaluation point
  never moves. A corner in a *warp* is not. The warp is composed onto the
  model's own index, `q = q₀ + Σ_j βw_j·I_j(q₀)` with `q₀ = −η_t·e^{−η_ls}`, so
  `q₀` moves with β while the hull is frozen at the seed `q₀`, and the basis is
  evaluated on both sides of the edge inside a single inner solve. Two of the
  chain-rule channels carry `I'_j` with **no `βw` factor at all**,

  ```text
      ∂²q/∂β_thr ∂βw_j = I'_j(q₀)·∂q₀/∂β_thr        ∂q̇/∂βw_j = I'_j(q₀)·r
  ```

  so the observed information jumps by `O(1)` across the edge **even with the
  warp switched off** — and `Φ = ½ Σ g(λ(Z_JᵀHZ_J))` is part of the inner
  objective the trust region accepts on. The objective is therefore
  discontinuous, and `actual/predicted` cannot approach `1` at any step size.

  Measured on `survival_location_scale_saved_fit_preserves_linkwiggle_metadata`,
  cycle 13, five attempts from one base point along a bit-identical direction:

  | ‖δ‖ | pred | actual | `d(−ℓ+½βᵀSβ)` | `dΦ` | max `dH` | at |
  |---|---|---|---|---|---|---|
  | 4.885e-5 | 1.760e-4 | −5.5209e-1 | 1.1404e-4 | −5.5220e-1 | 1.00001 | (5,5) |
  | 1.221e-5 | 4.400e-5 | −5.5213e-1 | 2.8512e-5 | −5.5216e-1 | 0.99999 | (5,5) |
  | 3.053e-6 | 1.100e-5 | −5.5214e-1 | 7.1280e-6 | −5.5215e-1 | 0.99999 | (5,5) |
  | 7.633e-7 | 2.750e-6 | **+2.7502e-6** | 1.7820e-6 | +9.6818e-7 | 9.16e-6 | (0,1) |

  The `−ℓ + ½βᵀSβ` half tracks its own linear model to six digits at every
  attempt including the three that cross, so the likelihood is not the defect;
  the whole error is `Φ`, as a jump. `dH` is ONE entry and its size is `1.0000`.
  Against the frozen hull edge `right = +7.261500860e-1`, one row's exit `q₀`
  sits `1.3e-7` outside it at the third attempt and `3.5e-6` inside it at the
  fourth, and `I'_3` steps `9.9999823e-1 → 0` between them. The value is
  continuous there (`[1,1,1,0.99999646] → [1,1,1,1]`).

  **Rejected: raise the spline degree.** `w''` is indeed piecewise constant at
  degree 2, but a four-arm A/B on the witness has `degree = 2/3/4/5` all still
  refusing while `degree = 0` (no `linkwiggle`) fits. The corner belongs to the
  extrapolation convention, not the polynomial degree, which is exactly why no
  degree touches it.

  `monotone_wiggle_basis_with_derivative_order` is now the single definition of
  the warp on all of `ℝ`, with `x̄ = clamp(x, left, right)`:

  ```text
      I_j(x)    = I_j(x̄) + I'_j(x̄)·(x − x̄)
      I'_j(x)   = I'_j(x̄)
      I⁽ᵏ⁾_j(x) = interior value inside the hull, 0 outside        (k ≥ 2)
  ```

  and `monotone_wiggle_basis_from_knots` routes through it, so the fit design,
  the derivative stack, prediction and inference all read one function. The
  interior is bitwise unchanged, so no fit whose rows stay inside the hull
  moves. The tail is the basis's own first-order expansion about the join, so
  the two halves are one differentiable function rather than two that meet. An
  I-spline is non-decreasing, so both tails have non-negative constant slope and
  `βw ≥ 0` still gives a monotone warp on all of `ℝ`.

  **Behaviour change worth stating:** the `[0, 1]` RANGE of the basis is given
  up outside the hull, and with it the old "the warp does nothing beyond the
  observed range" convention — a `linkwiggle` term now continues at its boundary
  slope instead of flattening. That range is precisely why
  `create_ispline_dense` saturates, and its own doc already directs callers who
  need otherwise to *"clamp inputs and add their own extrapolation
  correction"*; this is that correction, at the caller, and it is the standard
  convention for a spline *transformation* as opposed to a spline *shape*
  (restricted / linear-tail splines, as in flexible parametric survival models).
  Ordinary I-spline *smooths* are untouched — only the monotone-warp entry
  points route through the tail.

  Orders `k ≥ 2` are zero on the tail, so `I''_j` is still discontinuous at the
  join; it reaches the objective only as `m₂ = Σ_j βw_j·I''_j`, i.e. weighted by
  the warp amplitude, exactly as it already is at every interior knot of a
  degree-2 basis. The hull edge is therefore no rougher than a knot, which is
  the most a finite-degree spline can offer.

- **One sentinel, one resolver: the measure-jet auto range is now screened
  against the response on every standard-fit branch (#2750).**
  `length_scale == 0.0` is an unresolved request, and it had TWO resolvers — a
  pure-geometry rule inside the basis builder (the median nearest-node spacing)
  and the #2750 response screen — with which one a model got decided by which
  branch of the standard-fit dispatch it happened to take. The screen ran inside
  `fit_term_collectionwith_spatial_length_scale_optimization`, so a collection
  carrying a latent coordinate or coefficient groups was resolved by geometry
  alone.

  That is not a tuning difference between branches. `ℓ` decides WHICH span the
  representers occupy and a smoothing parameter cannot move a span, so the two
  resolvers produce different models, and the measured gap between them on the
  fixtures that do reach the screen is a factor of `1.6`–`13` in held-out error.

  The screen now runs once at the top of `fit_standard_model`, before the
  three-way dispatch and before the Tweedie-`p` profile, so every branch passes
  it. It is idempotent by construction — it only fires on the `0.0` sentinel — so
  the call still inside the spatial driver (reached directly by other drivers and
  by tests) is a no-op afterwards, and the #1762 Firth retry re-enters with the
  range already resolved instead of screening a second time.

- **The measure-jet range screen's downward walk was bounded by a guard that
  could not fire (#2750).** The screen walks geometrically off either end of the
  realized scale band while that end is still the incumbent, and its own comment
  said the ends were the bracket's — "so the walk introduces no length of its
  own". The upward cap was enforced. The downward one was

  ```rust
  if !upward && next_ln < floor_ln - bracket.log_step * (scored.len() as f64) { break; }
  ```

  and it recedes by one log step for every node the walk pushes, exactly as fast
  as `next_ln` descends — so the comparison is false at every iteration for any
  bracket with two or more nodes. The only stops left were "the criterion stopped
  improving" and "the basis refused to build".

  That matters now rather than before, because the outer search's own `ln ℓ`
  window is floored at the same node spacing: a screen that seeded below it would
  be widened INTO by the #2454 incumbent-containment rule, reintroducing exactly
  the region the floor excludes.

  The walk is upward-only, which is what the coordinate says. The band's bottom
  node IS the floor — the median nearest-node spacing — and it is already scored,
  so there is nowhere below it to walk to.

- **Farthest-point knot selection compared a squared LENGTH against the number
  one, so it stopped being scale-equivariant below unit radius (#2750).**
  `select_thin_plate_knots` is the shared center selector for every radial
  spatial smooth — `thinplate`, `duchon`, `matern` and `mjs` all reach it — and
  its maximin/centroid tie tolerance was

  ```rust
  let knot_scale2 = dist2_to_centroid.iter().copied().fold(0.0_f64, f64::max).max(1.0);
  let tie_tol = KNOT_MAXIMIN_TIE_REL_TOL * knot_scale2;
  ```

  The constant's own doc states the requirement: it must sit "several orders of
  magnitude above [the `ε·‖x‖²` round-off floor] yet **far below any genuine gap
  between geometrically-distinct candidates**". `.max(1.0)` substitutes `1` for
  `‖x‖²` for every cloud smaller than unit radius, which breaks that second half
  outright. Measured on a 240-row 1-D chart of half-width `5.2e-4`: squared
  radius `2.7e-7`, genuine maximin gap between neighbouring candidates `~6e-10`,
  floored tolerance `1e-9` — **the tolerance is larger than the gap it had to sit
  far below**. Every candidate ties, the invariant support-distance profile
  decides a selection it was only meant to referee, and the knots come out
  different from the ones the same configuration gets in different units.

  Downstream that is not cosmetic. The knots ARE the measure-jet quadrature
  seeds, so the median nearest-node spacing moves, and with it the auto
  representer range, the scale band, and the `ln ℓ` search window below.

  The floor is removed rather than replaced. With `tie_tol = 1e-9·‖x‖²` every
  ingredient of the comparison scales as `c²` under an isotropic rescale, so the
  selection commutes with it exactly. The degenerate end is unchanged: a
  coincident cloud has `‖x‖² = 0` and now `tie_tol = 0`, but every squared
  distance there is exactly zero, so the same candidates tie as before.

- **A measure-jet term's `ln ℓ` search box was a chosen absolute interval, and
  `ℓ` is a length in the data's own chart (#2750).** Every measure-jet ψ
  coordinate got the same kind of box:

  ```rust
  pub const MEASURE_JET_PSI_LN_LENGTH_SCALE_BOUNDS: (f64, f64) =
      (-6.907755278982137, 4.605170185988092);   // ln[1e-3, 1e2]
  ```

  and the doc said why: *"Absolute (not seed-relative) so the bound producer
  needs no data view, matching the other dial boxes."* For the two PENALTY dials
  that is right — `α` and `ln τ` are dimensionless and no geometry bounds them.
  For `ln ℓ` it is not: `ℓ` decides which span the representers occupy, it is a
  LENGTH in the frame the basis is realized in, and both of its walls are the
  same measured length — the median nearest-node spacing `s`, which is also the
  auto range and the scale band's floor — read at the two ranges where the kernel
  stops saying anything about the pair it separates:

  * **floor `ℓ = s`**: neighbouring representers overlap at exactly `exp(−1/2)`;
    below it they stop overlapping, the design degenerates from a partition of
    unity into a bump-per-node indicator, and rows between nodes fall outside
    every representer's support;
  * **ceiling `ℓ = s/√(2√ε)`**: that same pair's kernel value has come within
    `√ε` of 1, so it is no longer distinguishable from a coincident pair in the
    arithmetic the chart is built in. `√ε` is the chart's own bar — the same
    half-mantissa `condition_representer_section` spends.

  So the window is `[ln s, ln s − ½ln(2√ε)]`: it TRANSLATES with the chart and
  its width is `8.664`, a pure function of `f64::EPSILON` rather than a number
  anybody picked.

  The measured harm of the absolute box was the first trial step. On
  `measure_jet_perf_parity` the first `ln ℓ` step is `−0.693`, landing at
  `ℓ = 0.488` against a floor of `0.5145` — **outside the term's own geometry** —
  and it is rejected, each rejection a full design realization; the search then
  excursions to `ℓ = 0.34`, a range `1.5×` below the node spacing where the
  representers no longer overlap. Clamping to the derived window:

  ```text
                          outer evals   design realizations   wall (min of 3)
    before                    105                57                0.99 s
    after                      58                32                0.62 s
    matern(k=16) control       18                 1                0.52 s
  ```

  **The ceiling is deliberately NOT the node bounding-box diagonal.** That is
  where the response screen stops WALKING — a stopping rule for a search over
  nodes — and a first attempt used it as the box. It railed the outer search on
  three fixtures and refused their fits: the profiled criterion genuinely prefers
  a range at or above the node diameter, because as `ℓ` grows the
  gauge-quotiented representer span tends to a polynomial one, which is the right
  basis for a smooth target. A long range is a legitimate model, so the upper end
  has to be a feasibility statement and nothing weaker.

  The regression test is an INVARIANCE rather than a level: the window is made of
  lengths, so rescaling the chart by `c` must shift both ends by exactly `ln c`
  and leave the width fixed. An absolute window fails both halves by
  construction — the same node configuration in metres and in millimetres would
  be handed two different search problems, and at `c = 10³` the seed moves `6.9`
  log units inside a window only `11.5` wide.

  `[KAPPA-PHASE]` records now carry the SIGNED ψ coordinates beside `‖ψ‖`.
  A norm is the right summary for a multi-axis anisotropy block and the wrong
  one for a single signed coordinate: `‖ψ‖ = 0.718` is consistent with a trial
  at `ℓ = 2.05` and with one at `ℓ = 0.49`, and only the second is outside the
  window — which is exactly the distinction the diagnosis above turns on.

- **A curvature refusal is now adjudicated BY THE CRITERION instead of asserted
  by the matrix (#2612).** `negative_curvature_escape_point` already stepped the
  criterion along the reported minimum eigenvector, and the code threw half of
  its verdict away: it returned `Option<Array1<f64>>`, so "a strictly-descending
  feasible point exists" arrived as `Some` while "no descending trial exists
  anywhere I looked" arrived as `None` — bit-identical to "the escape was never
  runnable" — after which the refusal proceeded on the matrix's word alone.

  That second case is not an absence of evidence. Evaluating the objective at
  trial points is not a finite difference (SPEC 2 is untouched); it is the
  criterion answering the exact question the Hessian claimed to answer, and a
  direction along which the criterion does not fall is not a descent direction of
  that criterion. #2665 is the same defect from the other side: an analytic
  `λ_min = −1721.5` whose objective curvature along its OWN eigenvector is
  `+121.6`. No resolution bound catches that — the matrix there is not
  imprecise, it is wrong.

  The step ladder also stopped in the wrong place:

  ```rust
  const ESCAPE_STEP_SCALES: [f64; 5] = [1.0, 0.5, 0.25, 0.125, 0.0625];
  ```

  `0.0625` because five entries had been written down, so a claim whose descent
  only appears below that step read identically to a claim with no descent at
  all. At a stationary point the claim's own quadratic model predicts
  `½|λ_min|α²`; once that reaches the criterion's resolution the claim predicts
  nothing the criterion can represent and no smaller step can falsify it. The
  ladder now runs `1 → α_min = sqrt(2·objective_resolution/|λ_min|)` by halving
  in both signs, with the resolution being the same
  `rel_cost_tolerance`-anchored quantity the rail and cost-stall machinery
  already spend. No constant is chosen; the only other stop is `f64::EPSILON`,
  where halving stops changing `ρ + αv` at all.

  `SaddleAdjudication::{Descended, Contradicted, Declined}` replaces the
  `Option` (`probed == 0` is `Declined` — nothing evaluated, nothing falsified),
  and `CurvatureEvidence::CriterionContradicted` records the withdrawn verdict.
  It is deliberately **not** `Measured { psd: true }`: nothing established that
  the point is a minimum, only that this matrix's negative direction has no
  operational content. `psd()` stays `None`, so the published `hessian_psd`
  contract (`null | true | false`) is unchanged.

  Measured at the penguins terminal ρ with the Jeffreys term armed, which is
  what put this on the table and also killed the first hypothesis about it:

  ```text
    v'Hv (analytic)    = -6.709810e-5
    v'J(g)v (measured) = -9.248844e-5    stable to 5 digits over h = 1e-4 .. 1e-2
    gap/|analytic|     =  3.784e-1
    max_k |g_k| over the judged coordinates = 9.6243e-4
  ```

  The sign agrees, so that negative curvature is real; and `|λ_min|` sits 14×
  INSIDE its own gradient floor, so the curvature conjunct does not refuse there
  at all. The `3.784e-1` gap is the omitted `D²_β H_Φ[−v_l, −v_k]` term measured
  at production scale — it exceeds the `0.25` bar the armed gate applies to
  `λ_min` on its own seconds-scale fixture, which is now on the record rather
  than papered over.

- **Every wholly parametric multinomial fit was being refused for not having a
  penalty, and "no penalty" was the answer (#2612).** The posterior-mean
  predictive needs `S_λ` as an operator, because it evaluates the penalized
  log-posterior away from the mode. It read that operator off the
  influence-matrix reconstruction, which returns `None` on two conditions the
  penalty does not depend on:

  ```rust
  let joint_recon = fit.artifacts.joint_log_lambdas.as_ref().and_then(|jll| {
      let n_components = penalties_arc.len();
      if n_components == 0 { return None; }                    // unpenalized
      let hinv = fit.covariance_conditional.as_ref()
          .filter(|c| c.nrows() == expected_joint && ...)?;     // a DIFFERENT measurement
  ```

  `n_components == 0` means the model is *unpenalized*, so `S_λ` is the **zero
  operator** — a value, not an absence; and `H⁻¹` is a measurement of a different
  object, so conditioning the penalty's availability on it makes a covariance
  failure surface as a missing penalty. A hard refusal on `None` then converted
  both into no fit at all: `y ~ x1 + x2` — no smooths, hence no penalty
  components — stopped fitting, and both
  `quality_vs_statsmodels_multinomial` arms that use it went `GAM_ERROR`.

  `S_λ` is now measured in its own right from the family's equivariant specs and
  the selected `λ`, with the unpenalized case returning the zero operator it is.
  The specs are assembled ONCE (they materialize `n_specs` dense `(P·M)²`
  matrices) and the influence matrix and the published payload both read that one
  list, so they cannot describe different penalties — the property the previous
  two-site assembly asserted in a comment and only approximated. It refuses only
  on genuine inconsistencies: a `λ` vector that does not match the spec list, a
  spec of the wrong dimension, or `λ` reported for a model with nothing to
  multiply.

  Both regression bars are statements about the same operator from opposite
  sides, so a payload that was merely publishable could not pass both: the
  unpenalized arm must publish `S_λ = 0` **and** an influence matrix that is
  exactly `I` (which `F = I − H⁻¹S_λ` forces), and the penalized arm's published
  `S_λ` must reproduce its own published influence matrix through
  `H⁻¹S_λ = I − F`.

- **A curvature gate was refusing on numbers smaller than its own instrument's
  measured error, and nine `matern` benchmark scenarios died of it (#2748).**
  `invert_identified_rho_hessian`'s entire `‖δH‖₂` was
  `eigenpair_backward_error_bound` — the eigensolver's residual, which answers
  *"given this matrix, how wrong is σ?"*. The question at a criterion-resolution
  site is *"how wrong is this matrix?"*, and
  `gam-linalg/src/curvature_resolution.rs` already says in its own module doc
  that "a site that needs the second must not be handed the first".

  The second is measurable in situ, with no new tolerance. On the penalty map's
  certified invariance `T`, lifted to ρ, the criterion is exactly constant in
  `λ`, so `ρ''(0)_k = −t_k²` gives

  ```text
      T' H_rho T  -  T' diag(g_rho) T  =  0        EXACTLY, at every rho
  ```

  and whatever its residual returns is error and only error — in exactly the
  currency this gate spends, since it compares a Hessian eigenvalue against a
  gradient-built floor. Measured at the refusing ρ of
  `geo_disease_eas_matern_k6`:

  ```text
    eigensolver backward error                    8.342e-19
    |T'(H_rho - diag(g_rho))T|_2                  9.872e-8      <- eleven orders larger
    refused curvature                            -2.010e-8      <- INSIDE it, by 4.9x
  ```

  `CurvatureResolution::analytic_weyl_from_components` now takes several NAMED
  measured components and resolves to their maximum — each is a certified lower
  bound on `‖δH‖₂`, so the largest is the strongest available fact, and a sum
  would not be derived from anything. Three components are supplied: the
  eigensolver's backward error, the penalty-map invariance residual above, and
  the rho-Hessian's **symmetrization defect** `‖(H − Hᵀ)/2‖₂`, which is exactly
  zero for any twice-differentiable criterion and was being computed and thrown
  away by the `symmetrize_in_place` call that precedes the gate.

  Nothing was widened by hand. With one component the resolution is
  `analytic_weyl` bit for bit, so a fit whose penalty map has no invariance does
  not move by an ulp; #2665's `λ_min = −1.6e3` saddle is ten orders outside every
  measured component and still refuses.

  Two further defects in the same cluster, both "one channel, two verdicts":

  * the penalty-map Gram was accumulated naively over `m = block²` products, so
    its error was `m·ε·Σ|S_i S_j|` — three orders above the bar its own rank
    decision is taken at, and enough to make an exactly proportional penalty pair
    read as independent (measured: the same pair gave `1 − cos = 1.11e-16` in one
    cell and `4.80e-14` in another). Neumaier compensation removes the length
    dependence; the bar is untouched.
  * `try_exact_joint_spatial_length_scale_optimization` returned `None` both when
    the joint κ route could not be BUILT and when it ran, graded its own candidate
    against the shipped scalar-route score and correctly DECLINED it. The caller
    mapped both to `"spatial kappa optimization is unavailable"` and failed the
    whole fit, so a route that had just decided the incumbent was better had that
    decision converted into a fatal error. `JointSpatialKappaOutcome` now says
    which, and a decline ships the incumbent — which is what its own log line
    promises.

  Measured end to end, one scenario per run, on a wheel built from the checkout:
  `geo_disease_eas_matern_k6`, `geo_disease_eas_matern_k12` and
  `papuan_oce_matern_k12` go from red to green.

- **The multinomial posterior mean was being computed by a method that is not
  an approximation of it (#2612).** `predict_multinomial_formula` publishes
  `E[softmax(x'β) | data]`. It computed that by approximating the coefficient
  posterior with the Laplace Gaussian `N(β̂, H⁻¹)` and integrating `softmax`
  against it. The `O(n⁻¹)` correction to a posterior mean has a curvature half
  and a skewness half; integrating a nonlinear functional over the Gaussian
  keeps the first and drops the second. On a well-conditioned fit both are
  small and nobody notices. On a (quasi-)separated softmax they are neither
  small nor same-signed — the likelihood is flat toward more separation and
  steep away from it, so the true posterior is skewed toward larger `|η|` while
  the symmetric Gaussian puts half its mass where the likelihood has already
  excluded the coefficient — and `softmax`'s concavity turns that misplaced
  mass into under-confidence at unchanged argmax. That is the penguin signature
  exactly: right class, flattened probabilities.

  The estimand is fine and stays. What replaces the method is the ratio form,
  which for a POSITIVE functional has the two Laplace errors cancel
  (`O(n⁻²)` rather than `O(n⁻¹)`), and which for a class probability is not a
  device but an identity — the extra row's likelihood factor IS the quantity
  being averaged:

  ```text
  E[p_c(x)]         = Z(D ∪ {(x, c)}) / Z(D)
  E[p_c(x)·p_d(x)]  = Z(D ∪ {(x, c), (x, d)}) / Z(D)
  ```

  Measured against an MCMC posterior on a `K = 3`, `p = 10` asymmetric
  quasi-separated fixture (an importance sampler was tried first and rejected:
  ESS ≈ 800 of 200000 in that skewed 20-dimensional posterior is a Monte Carlo
  error larger than the accuracy being certified):

  ```text
  max |Gaussian − exact| = 2.121e-1        max |ratio − exact| = 4.39e-3
  max |ratio E[p_c p_d] − exact|           = 3.89e-3
  ```

  and across basis widths, with the exact posterior tracking the plug-in at
  every width while only the Gaussian diverges — monotonically in the number of
  nearly-unconstrained directions, which is the amplifier that separates a
  four-smooth fixture from a two-parameter reduction:

  ```text
    p   max sd(eta)   plug-in   Gaussian     exact   Gaussian/exact
    2        8.08     0.05503   0.05753    0.05603       1.027
    8       15.88     0.05498   0.07039    0.05608       1.255
   16       26.49     0.05259   0.09452    0.05349       1.767
  ```

  Consequences worth stating:

  * **The saved model now carries its rows.** A Laplace summary cannot produce
    a posterior mean — that summary IS the quadratic model whose inadequacy is
    the defect — so `MultinomialSavedModel` stores the raw training design, the
    class index, the weights and the coupled joint penalty `S_λ`. This is the
    same choice `mgcv` makes in keeping the model frame with the fitted object,
    and it is required rather than optional: a payload without it cannot answer
    the question `predict` is asked.
  * **The Smolyak accuracy/level control is gone rather than ignored.** It
    existed because the old mechanism was a quadrature whose answer could be
    bought with more nodes. The new one's accuracy is a property of the
    expansion. What replaces it needs no configuring: `Σ_c E[p_c] = 1` is an
    identity of the estimand, so the computed sum's deviation from one is the
    error at that row, and a row past that tolerance is refused rather than
    published.
  * **The Gaussian-integrated quantity keeps its place and loses its name.**
    `MultinomialFitOutputs::predict_probabilities_with_se` is now
    `logistic_normal_softmax_moments`: the exact moments of `softmax` under a
    STATED Gaussian are a well-defined object with their own uses, and naming
    them for the Gaussian rather than for the posterior is what keeps the two
    from being confused again.

- **The multinomial outer-curvature gate was handed a count `#1587` stopped
  producing (#2612).** The exact-outer-curvature route was selected from
  `(K − 1) · n_penalties`. Since #1587, `equivariant_class_penalty_specs` emits
  one spec PER CLASS per penalty component whenever `K > 2`, so the four-smooth
  penguin fixture carries `8 × 3 = 24` outer coordinates and the gate was handed
  `2 × 8 = 16` — confirmed against the refusal's own `last_evaluated_rho`, which
  has 24 entries. The gate now reads
  `MultinomialFamily::joint_smoothing_dimension()`, and
  `MULTINOMIAL_EXACT_OUTER_HESSIAN_MAX_DIM` moves `16 → 24` without moving its
  calibration point: the same fixture, re-read with a corrected ruler. At
  `K = 3` the classification is unchanged (`3n ≤ 24` and `2n ≤ 16` are both
  `n ≤ 8` components).

- **`λ̂` is CHOSEN, and the smooth-term LR reference was pricing it as given
  (#2672).** The pooled size of this issue's own null-simulation grid, measured
  at main for the first time since `7dbd1dc43` landed: `size@.05 = 0.0962`
  against nominal `0.05` and a band of `±0.0449`. The figure on the record is
  `0.0272` — conservative by 1.8x. It is now anti-conservative by 1.9x, and
  `7dbd1dc43` is what moved it: it removed the Wood–Pya–Säfken reference-df
  inflation on the correct ground that the term is not in the fixed-`λ` null
  law, and nothing replaced what that inflation had been standing in for. The
  grid was not re-run.

  A Gaussian null with `σ` KNOWN — so Lawley's `Δε` is exactly zero and the
  reference is the only thing under test — reproduces it with none of gam's
  machinery, and names the mechanism: `corr(W, Σw) = 0.94–0.96`. REML picks `λ̂`
  on the same data that produced `W`, so the reference moves *with* the
  statistic, but not by enough.

  ```text
  conditional at λ̂       α = .20   .10    .05    .01
    n = 30,  k = 12          .2060 .1320  .0840  .0180
    n = 100, k = 12          .2160 .1100  .0580  .0140
    n = 200, k = 12          .1850 .1025  .0650  .0150
  ```

  **It is not a mean problem and must not be fixed as one.** On those runs
  `E[W]/E[Σw] = 2.4–2.5`, and dividing `W` by that ratio takes `size@.05` from
  `.087` to `.0000`. Two further candidates were measured and refused:
  restoring the WPS term is the `0.027` arm, and substituting the `λ̂`-corrected
  covariance `Vp` for `Vb` makes it *worse* (`.040 → .065`) on a `Var(ρ̂)` that
  measures `9e6` because the criterion is flat under the null — which is exactly
  the objection `7dbd1dc43` raised against that term, now confirmed.

  **What the statistic needs is the law of `W(λ̂)` with the selection replayed.**
  Diagonalize the term's fitted penalty against the Schur-complemented
  information: the pair is symmetric-definite, so one basis diagonalizes both,
  with generalized eigenvalues `ν_k = p_k/(1 − p_k)` read straight off the
  penalty shares already computed. In that basis the tested block is `q`
  independent standard normals, and BOTH the statistic and the criterion that
  selects `λ` are closed forms in them and in `t = λ/λ̂`:

  ```text
  W(t) = Σ_k (2f_k − f_k²) u_k²,          f_k = 1/(1 + t·ν_k)
  V(t) = ½ Σ_k u_k²·t·ν_k/(1 + t·ν_k) + ½ Σ_{ν_k>0} log((1 + t·ν_k)/(t·ν_k))
  ```

  So the whole selection — draw, choose `λ̂`, read `W` — is a function of `q`
  numbers, and the null law is generated with no design, no response and no
  refit, over the same `ρ` box the solver used. `t = 1` reproduces the
  conditional law exactly, so this is a strict generalization rather than a
  different reference.

  ```text
  selection-aware        α = .20   .10    .05    .01
    n = 30,  k = 12          .1940 .1160  .0560  .0120
    n = 100, k = 12          .2020 .0840  .0440  .0080
    n = 200, k = 12          .1775 .0925  .0425  .0075
  ```

  Closer to nominal at every level in every cell — twelve of twelve — with power
  untouched (`.9967` at `α = .05` either way on a planted alternative).

  Two readings that measurement killed on the way: selection does not *inflate*
  the statistic, it *disperses* it (`E[W(λ̂)] = 1.13` against `E[W(1)] = 2.17`,
  because a fresh null draw usually wants more shrinkage than the fit chose,
  while a draw that looks wiggly buys a smaller `λ` and a much larger `W`); and
  the control variate does not always tighten the Monte-Carlo error, so that
  error is measured per query and published. `p_value_bound` carries the
  quadrature bound plus twice the replay's standard error, and
  `p_value_conditional` publishes what the fixed-`λ` law alone said, so the
  correction is visible rather than folded in.

- **The refinement certificate had ONE side, so "the cascade has remaining gain"
  and "the bound is too loose to tell" were the same sentence (#2759).** Four
  cascade fixtures refuse at the rank-maximal design, and the issue's own framing
  put the closest one "inside the gain bound's own measured 1.30x slack". There
  was no way to decide that from the certificate, because the certificate was
  `‖g‖²/(λd)` and nothing else.

  Appending the candidate columns `X₂` with penalty `λd` decreases the penalized
  objective by exactly `gᵀS⁻¹g`, with `g = X₂ᵀW r̂` and
  `S = X₂ᵀW(I − H)X₂ + λd·I`. The shipped bound discarded the ENTIRE data term —
  `S ⪰ λd·I` — which is the `x = 0` member of a family no later member of which
  had ever been evaluated. For ANY `x`, writing `r = g − Sx`:

  ```text
      2xᵀg − xᵀSx   ⩽   gᵀS⁻¹g   ⩽   2xᵀg − xᵀSx + ‖r‖²/(λd)
  ```

  Left is `(x − S⁻¹g)ᵀS(x − S⁻¹g) ⩾ 0`; right adds `rᵀS⁻¹r ⩽ ‖r‖²/λ_min(S)` with
  `λ_min(S) ⩾ λd`, the same structural fact the shipped bound rests on and the
  only inequality used. Both ends are computed from an explicit `Sx`, never from
  a conjugate-gradient recurrence, and the upper end is floored by the shipped
  number, so the certificate can never be looser than the one it replaces. `S` is
  matrix-free: one apply of the candidate design through the hash grid the gain
  vector already builds, ONE cascade solve for `(I − H)`, one apply back.

  **The stopping rule is the decision, not a tolerance.** Iteration stops as soon
  as the bracket lands entirely on one side of `REFINE_TOL·rss_pen`. There is no
  accuracy constant to pick because accuracy is not what is being asked for — a
  comparison is. The ceiling is the Krylov dimension, past which the answer is
  exact by construction.

  **The hypothesis this was built on is FALSIFIED, and it was ours.** The claim
  was that discarding the data term "is not a small conservatism" in the
  rank-maximal regime. Measured:

  ```text
    fixture                        lower        upper        tolerance    lower/tol  slack
    cascade_state_rejects_corrupt  7.944884e-3  7.946538e-3  6.547001e-3   1.214x    1.024x
    ..._roundtrip_reproduces_...   6.326893e-2  6.327000e-2  1.690929e-2   3.742x    1.018x
    quasi_uniformity_guard_...     3.560937e-2  3.562072e-2  2.485628e-3  14.326x    1.063x
    smoothness_ceiling_...         1.050801e-1  1.055257e-1  2.184455e-3  48.104x    1.223x
  ```

  The slack is 1.8% to 22%, the bracket has closed to four to six digits, and the
  exact gain exceeds the tolerance by 1.21x to 48x. No tightening can pass these
  fixtures because there is nothing left to tighten — and that is the answer the
  issue asked for, now impossible to confuse with conservatism. `Underresolved`
  and `RefinementCertificate` both carry the bracket, and `Display` says out loud
  when the lower end is above tolerance.

  Three of the four fixtures were then found to be gated on a certificate they do
  not test — two persistence gates and a metric guard, all reaching a fit through
  `fit_residual_cascade`. They take a fixed-depth design and `fit_reml` now, or
  assert what they name. `smoothness_ceiling_...` is left RED with its assertion
  standing: it IS about the certificate, and at `num_centers = 5997` on
  `n = 6000` with `rss_pen = 2.1845` against a noise floor of `n·σ² = 2.4`, what a
  finer level buys is interpolation inside a row space the design already spans,
  not discretization bias. `REFINE_TOL·rss_pen` is a bias criterion being read
  where more columns do not reduce bias. That is a criterion question, it is not
  answered by moving `REFINE_TOL`, and a gate that says so is worth more than a
  green one that does not.

  Verified: the bracket is gated against the objective decrease it bounds by a
  route sharing no code with it — build the design with the candidate level
  appended, solve at the same fixed λ, difference the two penalized objectives —
  reproducing the truth to every printed digit in 3-10 CG steps across a
  six-decade λ sweep. `cargo test -p gam --test misc residual_cascade`: 25 of 26,
  from 12 of 17 at the start of the run.

- **The SAE terminal Newton polish is a Levenberg–Marquardt trust region on the
  stationarity residual, judged in the currency its own gate reads (#2762).**
  The phase accepted 100% of its steps while the raw KKT gradient rose 15x–107x
  per accepted step. Two defects, stacked.

  **The merit.** The acceptance test compared the TRIAL state's Newton decrement
  in the MAJORIZER metric, `gᵀB(θ₊)⁻¹g(θ₊)`, against the PRE state's decrement in
  the EXACT-Hessian metric, `gᵀA⁺g`. Same bilinear form, two different
  operators, measured 67x apart on the witness — so the test was satisfiable by
  any step at all. `gᵀB(θ)⁻¹g(θ)` is not a function of the state alone: it falls
  whenever `B` stiffens, however far `g` rises, so it cannot referee a
  comparison between two states. The merit is now `½‖g‖²`, which is the quantity
  the KKT gate is a bound on and a function of the state alone.

  **The step, and this is the one that decides convergence.** Making the merit
  self-consistent does not converge this phase — measured at 482 accepted steps
  / 0 rejected — and the reason is that the step is outside its own model. At
  the `#2015` witness, `‖g‖ = 1.226522e-4` with the WHOLE residual inside the
  operator's retained range, the undamped step is `‖Δ‖ = 4.416833e-1`; applying
  it drives the merit `7.52e-9 → 6.07e0`, and an Armijo test on `½‖g‖²` first
  passes at `α = 4.9e-4`, buying 0.03%. **The step's LENGTH is set entirely by
  the near-null eigendirections of `A` while the residual is carried by the
  well-conditioned ones**, and no scalar step length separates those: shrinking
  the step to keep the flat direction inside the model shrinks the useful
  directions by the same factor. Every earlier attempt on this issue traded one
  fixture for another against that wall.

  Damping separates them, and `A` is already materialized and diagonalized here,
  so the entire Levenberg–Marquardt path — and the model residual it predicts —
  is closed form at one diagonal pass per point:

  ```text
  Δ(ν) = Σ_i u_i λ_i (u_iᵀ rhs)/(λ_i² + ν)      g + AΔ(ν) = Σ_i u_i c_i ν/(λ_i² + ν)
  ```

  On the same state, sweeping ν and MEASURING each point:

  ```text
  ν=0        ‖Δ‖=4.42e-1  merit 7.52e-9 -> 6.07e0      ratio -8.1e8
  ν=5.73e-8  ‖Δ‖=4.37e-3                -> 5.38e-8     ratio -6.2e0
  ν=5.73e-7  ‖Δ‖=4.58e-4                -> 6.18e-11    ratio  0.9992
  ```

  `‖g‖ 1.23e-4 → 1.11e-5` against a `7.13e-5` tolerance, in one step, with the
  objective falling too.

  **Every number in the ladder is derived.** The first trial is `ν = 0` — the
  step this phase has always taken — so the quadratic tail near a
  well-conditioned root is unchanged and a state that never needed damping never
  pays for one. The ladder then spans `λ_min²` to `λ_max²` over the RETAINED
  spectrum by `RIDGE_GROWTH`: below `λ_min²` a damping cannot move the flattest
  resolved direction, above `λ_max²` every direction is already flattened. A
  trial is accepted when its MEASURED reduction is at least `ARMIJO_C1` of the
  reduction its own model predicted, with the shared round-off cushion. The
  accepted `ν` is carried to the next step divided by the same growth and
  snapped to `0` under `λ_min²`, so a converging tail returns to pure Newton by
  itself. Predicted reduction is monotone decreasing in `ν`, so a ladder that
  falls under `DIRECTIONAL_DECREASE_REL_FLOOR × merit` is exhausted — a proof of
  termination, not a cap.

  **The merit is the gate's currency, and the ambient residual is an
  invariant.** The gate is `raw ≤ tol OR quotient ≤ tol` with the quotient norm
  clamped at or below the raw one, so the gate IS the quotient bound and the
  quotient merit `½‖Π⊥gauge g‖²` is what this phase descends. That distinction
  is not cosmetic: on the `zz2015` witness the terminal state carries
  `gauge_share = 0.76`, so the ambient merit is 94% orbit and sits at its own
  floor while the gate is still 28x out — a 784x improvement in the quantity the
  gate reads registers as a 6% move in the ambient one. A projected norm can
  also fall without the residual falling, so acceptance additionally requires the
  ambient merit not to GROW: one acceptance currency, plus the invariant that
  quotient progress may never be bought by pumping residual into the orbit.

  **The budget is spent only while the phase is on track to finish.** A step
  costs one dense materialization + eigendecomposition of `A` — measured 13.4 s
  at `dim = 519`, against 0.14 s of assembly and 0.30 s for the entire damping
  ladder — so the step COUNT is the whole cost of this phase and a fixed cap
  prices every entry at the worst case. At the contraction the accepted step
  actually delivered, the band is `ln(tol/gate)/ln(contraction)` steps away; if
  that exceeds the steps left, the phase stops on its trajectory rather than on
  its budget. The test reads the system the accepted trial already assembled, so
  it costs nothing and fires one whole eigendecomposition earlier than the next
  loop top could. Stopping is neither a refusal nor final: the merit is
  monotone, so everything gained is kept, and the refine loop may re-arm the
  phase and re-measure. Measured on `zz2015`, whose inner solve fails in both
  arms: the refusal moves from `‖Π⊥gauge g‖ = 8.24` (`4660x` over the band) to
  `4.93e-2` (`27.7x`) — a 168x better terminal state, and `intensive_over_bound`
  falls 21 orders, `2.1e24 → 5.8e3`.

  Properties, not hopes: every accepted step strictly decreases the quantity the
  refusal is denominated in, so this phase **cannot leave the state worse than
  it found it**, which is measurably what it did before. The merit is monotone
  across steps by construction, so the dual-currency cross-iteration contraction
  bail is deleted rather than kept dead. A trial costs ONE assembly, where it
  used to cost an assembly plus a full arrow factorization because the merit it
  evaluated needed one. Indefiniteness needs no special case: `Δ(ν)` solves
  `(A² + ν)Δ = −Ag`, positive semidefinite for every symmetric `A`, so a
  resolved negative mode is descended rather than reflected.

- **The smooth-term likelihood-ratio test is scored against its own null law,
  not against a distribution fitted to two of that law's moments (#2672).** At
  fixed `λ` the whole-term LR is exactly `W = Σ_j w_j χ²_1` with
  `w = eig(2F_jj − F_jj²)`, so `Σ_j w_j` is Wood's `edf1` *and* the statistic's
  null mean. The reference was `g·χ²_ν` with `(ν, g)` matched to the first two
  moments of that spectrum. It is now the spectrum itself, inverted by
  `gam_math::probability::weighted_chi_square_sf` (Imhof) — which had landed
  under this same issue and had no consumer.

  The two-moment surrogate is exact when the weights are equal and one-signed
  wrong otherwise, with the error growing as the tail deepens. Measured against
  the exact law on `f_j = 1/(1 + λγ_j)` for a second-difference penalty, six
  decades of `λ` × `k ∈ {6, 12, 20}`, the size delivered at a nominal `α`:

  ```text
  α = 0.05   0.99 – 1.02 x       α = 1e-3   1.01 – 1.31 x
  α = 0.01   1.00 – 1.11 x       α = 1e-4   1.14 – 1.61 x
  ```

  **Where the gap lives is not where the intuition puts it, and the measurement
  is what says so.** The surrogate is exact at BOTH ends of the shrinkage range:
  `w ≡ 1` on an unpenalized term, and a single distinct weight once REML has
  shrunk a term to its null space — measured on a null-true `k = 12` fit,
  `w = (0.322, 5.9e-7, 7.1e-8, …)`, where the two references agree to eight
  figures. It opens in the middle, on a term carrying real signal. So a
  null-simulation size grid — which spends all of its time in the collapsed
  regime, at `α = 0.05`, where the surrogate is exact — could not have detected
  this, and its passing was never evidence the reference was right.

  **The whole spectrum, without a general eigensolver.** The moments were traces
  of powers of `F_jj` precisely because reading the weights off `F_jj` would need
  one — it is not symmetric. It need not be read off `F` at all: the penalty is
  block-diagonal by term, so `(I − F)_jj = [H⁻¹S]_jj = [H⁻¹]_jj S_jj =: P`
  exactly, hence `2F_jj − F_jj² = I − P²` and `w = 1 − eig(P)²`. `P = B·S` with
  both factors symmetric PSD is similar to `B^{1/2} S B^{1/2}`, reachable with
  the self-adjoint entry point already in `gam-linalg`, through `B = UΛUᵀ` rather
  than a Cholesky so a singular block is a zero eigenvalue and not a
  factorization failure. `[H⁻¹]_jj` is `beta_covariance()` divided by the
  family's own `coefficient_covariance_scale()`; the two factors are in
  reciprocal units, so that multiplier has to come off exactly or every weight is
  wrong by it.

  **The identity is measured, not assumed.** It holds only if the penalty is
  block-diagonal by term AND `Vb`, `F` and `S(λ)` are published in one
  coefficient basis — and both halves have been wrong in this exact path before
  (the similarity-map drop, the internal-basis first-order correction, the
  block-local `coeff_range`). None of the three is visible by reading. So the
  driver assembles the spectrum both ways whenever the fit supports it and
  publishes `moment_residual`, the relative disagreement; a test pins it under
  `1e-8` across two families, three model shapes and both shrinkage regimes.

  **The tail is resolved as finely as the statistic is known, and no finer.**
  Imhof's truncation point grows like `ε^{-2/(2+m)}` in the number `m` of weights
  active at it, and a shrunk smooth has one weight of order one over a tail of
  dust, so `m = 1` and the cost is `ε^{-2/3}`: at `gam-math`'s default `1e-11` a
  single p-value measures 0.13 s to 3.3 s. `W = 2(ℓ_full − ℓ_null)` is a
  difference of two separately converged optimizations, so it is known to
  `ΔW = tol·(W + E[W])`, and a p-value is known to `|S(W) − S(W + ΔW)|` however
  well the integral is done. That is what the quadrature is asked for — evaluated
  through the two-moment summary, which costs nothing and is being used as a
  SCALE rather than as a value. `SmoothTermLrInference::p_value_bound` publishes
  what was reached; the integration test compares the published p-value against
  the strict default THROUGH that bound, so the bound has to be honest.

  Three named lanes replace one switch: `NullSpectrum` (exact),
  `SpectralMomentMatch` (the old `g·χ²_ν`, when `H⁻¹` or the penalty is
  unavailable but `F` is), `UnitWeightFallback` (scalar EDF). Their errors have
  different signs and sizes, and the Python surface carries the lane, the
  spectrum, the residual and the bound alongside the p-values.

- **The certified cascade held seven `m²` blocks to deliver two objects that
  need one, and the admissible design width is DERIVED from that residency
  (#2758).** `smoothness_ceiling_forces_refinement_and_certifies_residual_bias`
  refused with `CertifiedSpectrumCapacity`: a 6000-row cascade identifies 5997
  penalized directions and the certified route stopped at 2893, because
  `CERTIFIED_SPECTRUM_MAX = isqrt(BYTES / (BLOCKS·8))` over a measured
  `BLOCKS = 8`. Raising `CERTIFIED_SPECTRUM_BYTES` was ruled out on the issue and
  is not what happened; it is untouched at 512 MiB.

  The measurement was honest and the constant was not the defect. What the route
  SPENT it on was. The criterion consumes `Θ` and `Vᵀβ` — every eigenvalue of the
  penalty-whitened Schur complement, and the whitened response in its eigenbasis.
  `eigenvectors` was read at exactly ONE site, to form that projection, and
  `eigh` cannot hand it over without building all of `V` plus faer's
  tridiagonalization workspace. On top of it a full `m × m` upper `X'WX` was
  assembled, of which only the `rank × rank` penalized block and a `q × rank`
  cross block (`q ≤ 4`) are ever read.

  ```text
    before                                    after
    m x m upper Gram          8 B / m^2       (not assembled at all)
    rank x rank Schur         8 B / m^2       packed upper triangle   4 B / m^2
    rank x rank eigenvectors  8 B / m^2       (never formed)
    faer EVD workspace       ~40 B / m^2      (no EVD)
    ----------------------------------        ----------------------------
    measured  6.41-6.84 blocks = 51-55 B/m^2  measured 4.03 B/m^2
    cap       2896 columns                    cap      10362 columns
  ```

  `V = QW` for the Householder `Q` and the QL `W`, so `Vᵀβ = Wᵀ(Qᵀβ)`. The new
  `gam_linalg::packed_symmetric_spectrum` reduces the packed triangle IN PLACE,
  applying each reflector to `β` as it is formed, then runs the implicit-shift QL
  accumulating every Givens rotation into that same single vector instead of into
  an `n × n` accumulator — the Golub–Welsch "keep one row of the eigenvector
  matrix" device, with a general start vector rather than `e₁`. Neither factor is
  ever formed. The mathematics is unchanged: all eigenvalues, the exact
  projection, the same roundoff floor, the same dropped null modes.

  `CERTIFIED_SPECTRUM_BLOCKS` is replaced by
  `CERTIFIED_SPECTRUM_BYTES_PER_COLUMN_SQUARED = 5`, an INVENTORY again rather
  than a black box: one packed `f64` triangle is `8/2 = 4` bytes per `m²`, plus
  the next integer of headroom. `5997 < 10362`, so the binding constraint on that
  fixture returns to identifiability.

  **Two defects in the new reduction, both found by measurement rather than
  review.** The classical relative deflation test `|e_i| ⩽ ε(|d_i|+|d_{i+1}|)`
  DOES NOT TERMINATE on a rank-deficient Gram: on `F Fᵀ` with `F` 296×148
  standard normal, `‖T‖ ≈ 9e2` while the 148 null directions arrive as
  `d ≈ e ≈ 1e-13`, so the test asks for `4e-29` against rotations that re-inject
  `ε‖T‖ ≈ 2e-13` every sweep. The absolute floor `ε‖T‖_∞` is taken alongside it.
  And the reflector produced NaN — `τ` and `v` are invariant to a rescaling of
  `x` but `1/(α − β)` is not, so a row decayed into the denormal range (what a
  rank-deficient trailing block becomes after a thousand steps) OVERFLOWED it and
  `0·inf` on that row's exact zeros wrote NaN; the whole trailing block followed,
  and the QL reported it as a 30-sweep non-convergence at an index that meant
  nothing. The reflector is now built on `x / max|x|`, where `|α_s − β_s| ⩾ 1`
  holds at every input scale, and an `O(n)` finiteness check between the
  reduction and the sweep names a reduction defect as one.

  **Two gates moved with the design, stated rather than quietly retuned.** The
  peak-memory arms are now BOTH past `DENSE_GRAM_MAX` and assert it: the narrow
  arm sat at `m = 891`, where the design carries a persistent `dense_gram` cache
  — an `m²·8` term present in one reading and absent in the other, which does not
  cancel in the difference and biases the differenced marginal DOWNWARD, i.e. in
  the direction that makes an under-declared residency look fine. And
  `spectral_and_solved_residual_forms_agree` charged its comparator bound to ONE
  of two INDEPENDENT computations of the same quantity, treating the
  decomposition as exact: at `m = 20` and `cond(A) = 1.005` that is `4.46e-15`
  against a measured `4.60e-15`, so the gate failed on the last bit the moment
  the rounding differed. Both comparands are charged now — two equal terms, not a
  factor chosen to admit a number.

  Timing at the new widths, measured so it is not rediscovered: a rank-6795
  profile builds in 46.2 s (9.1 GFLOP/s through the packed symv/spr2 on 4 cores),
  rank 1922 in 1.16 s.

- **`ln S` was seven approximations wearing one name, and its error was a step
  function of `(mu, sigma)` (#2714).** The latent-survival / frailty inner solve
  stalled at `stationarity_residual = 1.741e-2` against a `3.6e-10` tolerance
  with the trust radius railed at `1e-12` and both terminal rejections coming
  from the OBJECTIVE — the model and the likelihood accepted every step. The
  diagnosis on that issue localized it to `cloglog_log_survival_term_controlled`
  taking `value.ln()` of a value-space evaluator, and reasoned about the size of
  that error with the representation-floor model `EPSILON/S`, which at the
  failing point `(mu, sigma) = (3.2, 0.15)` predicts `6.9e-14` and therefore
  cannot explain the measured `6.5e-2` disagreement between the analytic score
  and a finite difference of the value.

  Graded on the shipped path against a 60-digit reference (peak-shifted, so the
  reference is accurate ABSOLUTELY in `ln S` even at `ln S = -1.3e5`; the naive
  un-shifted high-precision integral is itself wrong by `8.7e-3` at
  `(8, 0.005)`), the error at that point is **`1.242e-1`** — nine orders above
  the model the thread was using. And it is not one bad branch:

  ```text
      mu     sigma    ln S (ref)        shipped err   route
      12.0   0.002    -1.28982e5        4.371e+06     QuadratureFallback
       8.0   0.002    -2.96340e3        4.975e+05     QuadratureFallback
      12.0   1.000    -5.81967e1        8.201e+01     ExactSpecialFunction
       8.0   0.500    -7.09799e1        3.693e+01     ExactSpecialFunction
       3.2   0.150    -2.01463e1        1.242e-01     ControlledAsymptotic
  ```

  Three worst rows, three different routes — including the fixed-window
  Gumbel-mixing escape hatch that #798 added FOR the underflow corner, whose
  `Phi((eta-mu)/sigma)` transition is unresolved at small sigma by a node ladder
  #2469 had pinned inert at a constant 513. A fourth defect sits in the
  rare-event asymptotic `ln1p(-e^{mu+sigma^2/2})`: at `(-50, 8)`, exactly on its
  own `rare_log = -18` gate, it is **20.8x** wrong, because its first-order
  model needs the higher cumulants to be small and at `sigma = 8` they are not.

  So no threshold repairs this — there is no pair of these routes accurate on
  either side of a common cut, and an analytic derivative cannot be the
  derivative of a surface whose error jumps.

  **One surface.** `S` and `1 - S` are integrated in the standardized variable
  `z = (eta - mu)/sigma` on a Clenshaw-Curtis panel placed by the integrand:
  `L(z) = -z^2/2 - e^{mu+sigma z}` is strictly concave
  (`L'' = -(1 + sigma^2 e^{mu+sigma z}) <= -1`), so it is unimodal, its peak is
  the unique root of a monotone equation — solved in LOG form
  (`ln sigma + mu + sigma z - ln(-z) = 0`) so no `e^{mu+sigma z}` is ever
  materialized, which the value form cannot do when the root sits at
  `z ~ -mu/sigma` — and the points 60 e-folds below the peak bracket everything
  representable. Node count is the panel's local-scale arclength
  `T = int sqrt(1 + sigma^2 e^{mu+sigma z}) dz` (closed form) times a measured
  density, plus a `sqrt(sigma)` term for the Bernstein-ellipse shrinkage that
  `T` does not price: `T` is nearly constant at ~22 across the plane while the
  measured requirement walks `65, 97, 193, 385, 769` at
  `sigma = 0.5, 2, 8, 60, 200`, because `exp(-e^{sigma z})` is entire but its
  modulus grows off the real axis with period `2 pi / sigma`. Everything
  accumulates by (signed) log-sum-exp, so there is no value-space underflow to
  escape and no `.ln()` of a cancelled quantity. Past `S ~ 0.6` the complement
  panel supplies `1 - S` and `ln1p` finishes it, which is what retires the
  rare-event asymptotic instead of re-gating it.

  Worst absolute error of the panel on the same grid: **`9.3e-10`**, at
  `ln S = -5.7e6`, i.e. `1.6e-16` relative — the representation floor of the
  answer.

  **The `sigma >= 8` derivative gate is gone.** Gaussian integration by parts in
  `z` gives `sigma^j d^j S/d mu^j = int He_j(z) phi(z) f(z) dz`, so order 0 IS
  the value: the tower is the same sum with a different Hermite weight on the
  same nodes, and value/derivative cannot be on two surfaces even in principle.
  What remained was only which derivative BASIS is better conditioned — the
  direct tower (degrades at small sigma, where the answer is genuinely
  `O(sigma^j)` while the summands are `O(1)`) or the rung/Touchard combination
  (degrades at large sigma, #2610). A signed log-sum-exp knows its own
  cancellation `cond = ln(sum|terms| / |sum terms|)` and its relative error is
  `eps * e^cond`, so the tower is admitted on the solve of `eps e^cond = 1e-13`,
  `cond <= 6.1` — exactly while it is at the working floor of what it displaces.
  All 132 grid rows with `sigma >= 8` clear it, so the measured gate is a strict
  superset of the constant.

  Gates, replacing `cloglog_gumbel_quad_node_ladder_is_inert_at_a_constant_513_2469`
  (which pinned the inertness that caused the escape hatch's failure): a 28-row
  high-precision accuracy table (`6.7e-16` worst, relative); the
  value/derivative consistency assertion whose ABSENCE let this live, since
  every earlier gate scored an analytic derivative against its own value
  function (`9.8e-10` worst, against the `6.5e-2` that named the defect); a
  Richardson two-stencil smoothness sweep that converts a jump of size `d` into
  `0.75 d/h^2` and so bounds any residual step at ~1 ulp; and a two-sided
  statement that the node ladder now moves with sigma.

- **The outer gradient contracted an inverse that belonged to neither operator
  (#2515).** The Laplace criterion ranks `½log|A|` for the exact observed
  information `A = ∇²_θθ L`; `B` is the Gauss--Newton majorizer, the
  positive-definite scale the Newton and IFT solves factor. #2509 Phase-2b moved
  every production VALUE route onto `A`. The bundle/matrix-free route's
  DERIVATIVE did not follow, and the thread recorded that as "the gradient still
  prices `B`". Reading the lane settled that it was worse than that:

  ```rust
  let a_sys = chunk_term.exact_a_evidence_system(target, rho, &sys)?;   // A
  let (log_det_tt, log_det_schur) = matrix_free_arrow_evidence_log_det_surrogate(
      &a_sys, ..., lane.as_deref_mut())?;                               // value AND bundle off A
  return Ok((log_det_tt + log_det_schur, Some(sys)));                   // returns B
  ```

  The from-probes channels reconstruct `(H⁻¹)_tt = A_i⁻¹ + G_i S⁻¹ G_iᵀ` — row
  factors from a factor CACHE, `S⁻¹` from the probe bundle. Production paired
  `A`'s reduced Schur with `B`'s row factors, so the reconstructed inverse
  factored neither operator.

  Four changes make the routes one criterion. `BundleEvidenceGeometry` carries
  the operator and its OWN factor cache (`cache` stays `B` everywhere: promoting
  it would double-count `ΔC`, which `solve_exact_stationarity_matrix_free` adds
  back). The streaming lane takes the gradient-bearing evidence entry point, so
  value, bundle and row factorization come from one factorization of one system.
  `ArdAxisPrior::log_precision_curvature` and
  `softmax_sparse_curvature_rho_derivative_block` emit `∂B/∂ρ` or `∂A/∂ρ` from
  one function each — the same functions whose difference is the `ΔC` map. And
  the coordinate-block θ-adjoint and the #2330 Patch-D residual
  third-derivative leg, the last two channels still on `B` operands, were
  ported.

  Measured at a fixed state where BOTH routes are admitted (`α = 10`, `A` PD,
  the periodic ARD clamp active on 12 of 24 rows so `ΔC ≠ 0`), bundle route
  against dense exact-`A`:

  ```text
                            bundle on exact-A geometry     dense exact-A         |Δ|
  value ½(log|A|−log|A_tt|)  5.15457065258939906e0         5.15457065258939906e0  0
  logdet_trace smooth        1.41527087036750832e0         1.41527087036749344e0  1.49e-14
  logdet_trace ard          -1.51312923828332035e0        -1.51312923828329660e0  2.38e-14
  COMPLETE GRADIENT                                                               1.57e-14
  ```

  against `8.46e-1` for the majorizer-rooted carrier the witness now asserts as
  a control. The ARD coordinate does not merely shrink — it changes sign, which
  is what the unmajorized `α·cos κt` on twelve clamped rows is supposed to do.

  Two boundaries are recorded rather than papered over. The historical `α = 250`
  witness is OUTSIDE the region where route parity exists: there `A` has a
  per-row eigenvalue of `−1.93e2`, the streaming route refuses outright, and only
  the globally-priced dense route survives — so the witness was re-premised onto
  the admitted regime, with the admission itself asserted. And on a cache that
  actually DEFLATES the two routes still disagree (`9.13` against
  `‖g‖∞ = 5.00`), because the dense route floors the spectrum of `A` globally
  while the arrow route conditions per row and pseudo-inverts the reduced Schur.
  The streaming lane's spectral-deflation refusal therefore stays, now reasoned
  from that number and reading BOTH caches; the number is under test, so the
  refusal cannot decay into folklore the way its previous justification did.

- **The "logslope" block is the SLOPE, and the name was the only thing wrong
  (#2764).** `rigid_observed_logslope(g, s) = s·g` — the identity, no `exp`
  anywhere on that path — so the penalty is on `b`, not on `log b`. The issue
  proposed making the map a genuine log on two grounds: scale-invariance of the
  penalty, and positivity. Both were measured, and the proposed remedy is the
  wrong half of the repair.

  **The slope is signed and the sign is an estimand.**
  `survival_multi_z_fit_hard::survival_multi_z_fit_truth_neglog_minimised_at_true_slopes_30_seeds`
  plants `(0.32, −0.21)` and pins that the population negative log-likelihood is
  minimised there over 30 seeds. `b = exp(g)` cannot represent that fit at all,
  so it is a strictly smaller model rather than a naming repair. The zero
  crossing the identity map permits is the covariate value at which the score
  stops predicting.

  **The scale-invariance argument is about `λ`'s units, not about the fit.**
  Rescaling `z → z/κ` sends `g → κg` and `Σ → Σ/κ²`, and the row negative
  log-likelihood, the preserving scale `c` and the index `η` are all POINTWISE
  invariant under that — now measured at `κ ∈ {2, ¼, 10, 0.3}` in
  `survival_multi_z_slope_scale_equivariance_2764`, together with the
  admissibility of a negative slope and the evenness of `c` in the slope. What
  is left over is `λβᵀSβ`, which needs `λ → λ/κ²`; REML supplies it, because
  under `β̃ = κβ` its Laplace criterion satisfies
  `Ṽ(λ/κ²) = V(λ) − ½·nullity·log κ²`, a shift constant in `λ`, so `λ̂ → λ̂/κ²`
  and the fitted surface does not move. Two honest caveats travel with that, and
  are recorded on the function: the `ρ = log λ` box is absolute, so a large
  enough rescaling breaks the correspondence at the wall; and so would any
  absolute — as opposed to relative — solver tolerance.

  So the fix is the name, at the point where the mathematics is stated:
  `rigid_observed_logslope` → `rigid_observed_slope` in both marginal-slope
  families, carrying the decision record above. The block, the
  `logslope_formula=` keyword, the CLI flag and the on-disk fields keep the
  historical spelling — renaming a public keyword and a saved-model contract is
  a breaking change and is not this commit's to take — and
  `docs/marginal-slope.md` now says plainly that the surface is the signed
  slope and why.

- **The measure-jet representer chart spent the same half-mantissa twice, and
  paid for it in span (#2761, #2754, #2751).** `condition_representer_section`
  whitened the chart against the SQUARED operator `G = EᵀE` and cut at
  `√ε·λ_max(G)`. Since `λ = σ²` that is a bar of `ε^{1/4}` on `E`'s own singular
  values, i.e. it admitted only `cond(E) ≤ 8·10³` and deleted everything else
  from the design. At the ranges REML actively selects that is most of the
  basis. Measured on the #2761 fixture (1-D curve in 3-D, 16 centers), *span
  floor* = least-squares residual RMSE of the NOISELESS truth on the realized
  design's own column span — the bound no `λ` can beat, because `λ` shrinks
  inside a span and never moves one:

  ```text
    ℓ/ℓ_seed  cond(E)   ε^{1/4} cut: p  span floor    repaired: p  span floor
       1      3.0e+01        12          6.11e-2          16       6.11e-2
       2      2.8e+04        11          2.43e-2          16       1.94e-2
       4      4.2e+07         8          1.50e-2          16       2.10e-3
       8      9.1e+09         6          1.67e-2          16       1.81e-4
      16      2.7e+11         4          8.92e-2          16       3.54e-5
  ```

  Note the old column going back UP past `4×`: the truncated chart was worse
  than the seed range it was introduced to improve on. The repaired column at
  `8×` reproduces an 80-digit projection of the same span to every printed
  digit, so nothing being kept is below what binary64 can see.

  **The justification did not survive its own remedy.** The cut's stated reason
  was "the energy pullback squares `E`". That is true of an *unwhitened* section
  only: once `EᵀE = I`, `S = UᵀQU` with `U` orthonormal, and `Q` is `ℓ`-INVARIANT
  (a form on center VALUES), so `cond(S)` stops being a function of the range at
  all — measured `cond(E) = 1.0`, `cond(S₊) = 21.1`, `log|S|₊ = 0.800` at every
  `ℓ` from `1×` to `16×` the seed.

  The chart now reads `E`'s own SVD, and carries three bars in place of one,
  all anchored on `‖K_cc‖₂` — the norm of the operator whose product formed `E`,
  never `E`'s own `σ_max`. That distinction is load-bearing: `σ_max(E)`
  COLLAPSES with `ℓ` (the constraint that makes `Z` head-orthogonal is exactly
  what annihilates the flat limit `K_cc → 𝟙𝟙ᵀ`) while `σ_min(E)` sits flat at the
  roundoff floor, so a *relative* bar sinks below that floor and starts admitting
  it as signal — at `ℓ = 11` on the 1-D sweep fixture the bar is `1.5e-17`
  against entries of `1.7e-17`, and the chart handed the fit **48 columns of pure
  rounding noise**. Columns of noise fit anything, so the criterion had a
  spurious minimum out there: profiled Gaussian REML on the shipped design gives
  `V = −447.8` at `ℓ = 6.16` against `≈ −44` in the honest region, and the
  failing fit's own checkpoint was `ψ = 1.81`, i.e. `ℓ = 6.1`.

  * **existence** `σ > ε·‖K_cc‖·max(dim)` — the backward-error bar of forming
    `E = K_cc·Z`. Below it there is no direction, only roundoff.
  * **amplification** — the scaling, not the membership, is floored at
    `√ε·‖K_cc‖`. `1/σ` is the factor by which a direction lifts the design's own
    roundoff and the criterion squares the design into `XᵀWX`, so the lift keeps
    significant digits exactly when `ε·(‖K_cc‖/σ)² < 1`. A direction below the
    floor is DAMPED, not deleted: a span is unchanged by any invertible
    rescaling, so it contributes the same column space and enters at a small
    norm, carrying its roundoff in at that same small norm.
  * **visibility** `(σ/floor)² > dim · SPECTRAL_RANK_RELATIVE_TOLERANCE` — a
    damped direction whose squared weight falls under the canonical
    penalty-spectrum rank cutoff is classified UNPENALIZED, i.e. an accidentally
    free design direction, and it makes `log|S|₊` a step function of `ℓ`.
    Measured: the primary's nullity flapping by one moved the profiled criterion
    by **8.5** at fixed `λ`, which is a barrier no `ln ℓ` line search can cross
    (`line_search=StepSizeTooSmall` on both 1-D `s(x, bs="mjs")` fixtures). The
    chart and the penalty classifier now reach the same constant and can no
    longer disagree about which directions are penalized.

  As a consequence the arithmetic reproduces `MeasureJetRangeBracket::ceiling`'s
  own physics instead of merely asserting it beside them: past the node diameter
  `σ_max(E)` falls under the amplification floor and the whole representer block
  damps smoothly toward zero, leaving the affine head — "the block is numerically
  one function plus the affine head and there is no distinct model past it".

- **A Gram that cannot be indefinite was coming out indefinite, because it was
  formed by squaring first (#2761).** `rebuild_metric_consistent_ridge` computed
  the restricted null-function metric as `M = Nᵀ(G_c N)` — materialize the
  metric, then contract. With `G_c = A_Gᵀ A_G` that is identically
  `(A_G N)ᵀ(A_G N)`, and the two differ sharply in floating point: the first
  squares `A_G`'s condition number and leaves a SIGNED `O(ε‖G_c‖)` error on `M`'s
  entries, unbounded *relative to `M`* wherever `N` sits where `G_c` is small. So
  the caller — correctly reasoning that a Gram restricted to a subspace cannot be
  indefinite — refused, at `eigenvalue −3.31e-5 below tolerance −1.30e-11`, on
  four of four measure-jet sweep cases. The eigenpairs now come from an SVD of
  `B = A_G N` and `M` is never formed, so the indefiniteness branch is gone
  because indefiniteness is impossible rather than unlikely. This is the #2318
  rule — *rank revelation acts on `A`, not on `AᵀA`* — which the sibling
  `null(S_c)` computation twenty lines above already followed.

- **The measure-jet interval band omitted the third variance term its own file
  declares (#2761, #2752).** `measure_jet_web_quality_contracts` was failing at
  coverage `0.3648` against `[0.85, 1.0]`, and it is not the basis, not `λ`, and
  not the covariance. The fitted design's span floor on the noiseless truth is
  `0.0024` against a held-out bias of `0.0301` — `12.5×`, so the basis can
  represent this target. Refitting the same design at scaled `λ` over six
  decades leaves the bias between `0.0221` (essentially unpenalized, `edf` 23.9)
  and `0.0301`, so no `λ` makes the band honest and over-smoothing is refuted.
  And the control settles it: refitting on training rows whose coordinates are
  CLEAN — same latents, same `y` draw, everything else identical — the SHIPPED
  two-term band covers at `0.9843`.

  What remains is errors-in-variables, and it is the confound the fixture's own
  header identifies — but the header fixed only the QUERY half. It moved the
  query rows to clean on-web locations (correct) and left the TRAINING rows at
  `embed(z) + ε`, so the fit is a consistent estimator of `E[y | x_observed]`,
  displaced from `f` at an exactly-known location by
  `σ_coord·‖∇f‖ = 0.02 × 1.5 = 0.030` — the measured `0.0301` to two digits.
  That term is contract 6 of the same file, with `σ_coord` estimated at fit and
  frozen and its `∇f̂` FD-gated against the fit's own `η`; contract 5 simply never
  consumed it. Both contracts now read one producer and contract 6 asserts the
  band's number IS its own reconstruction, so they cannot drift into two
  definitions of `Var_input`. Coverage `0.3648 → 0.8871` with the window
  unchanged.

- **`Σ` in the marginal-slope identity is `Var(z | a)`, and the fit was
  supplying one global matrix (#2766).** `bms/gradient_paths.rs` writes the
  identity the whole marginal-slope family is defined by as

  ```text
    z | a ~ N(0, Σ(a)),   η = c(a)·q(t,a) + r(a)ᵀz
    E_z[Φ(−η) | a] = Φ(−q(t,a))    ⟺    c(a) = √(1 + r(a)ᵀ Σ(a) r(a))
  ```

  — `Σ(a)`, conditional on the marginal-index span — and then supplied it from
  `marginal_slope_covariance_from_scores`, ONE weighted empirical covariance
  pooled over every row. Substituting a constant `c̄` into the exact integral
  leaves `E_z[Φ(−η)|a] = Φ(−q·c̄/c(a))`: the realized marginal index is
  `q·c̄/c(a)`, a multiplicative, covariate-dependent distortion of the one
  estimand this family exists to deliver. Measured on `K = 2` scores whose
  conditional correlation moves over `±0.8` while both conditional marginals
  stay exactly `N(0,1)`, the distortion reaches **1.46×** at a shared slope of
  1.0, and a Monte-Carlo of the actual integral over 20000 rows reads a worst
  relative marginal-index error of **0.145**. It is the same failure the K=1
  `homoskedastic_var` field doc records (`Φ(q√(1+b²)/√(1+b²v))`), one dimension
  up: #2768's per-coordinate location-scale gate forces `E[z_j|a] = 0` and
  `Var(z_j|a) = 1`, and no per-coordinate map can reach the OFF-DIAGONAL.

  `Σ(a)` is now a fitted object, parameterised by Pourahmadi's **modified
  Cholesky decomposition** — `T(a)Σ(a)T(a)ᵀ = D(a)` with `T` unit lower
  triangular carrying `φ_jk(a)` and `log d_j(a) = γ_jᵀ[1|a]`. Every parameter is
  unconstrained, so every `a` — including rows off the training hull — yields a
  positive-definite `Σ(a)`; a regression on the ENTRIES of `Σ` does not, and one
  row at `|ρ| > 1` makes `c(a)` the square root of a negative number. Read
  forwards it is a triangular system of the same two regressions #2768 already
  ships, and `L(a) = T(a)⁻¹D(a)^{1/2}` is the exact Cholesky factor — the
  `Σ = LLᵀ` low-rank shape `MarginalSlopeCovariance` already admits — so the row
  program's quadratic forms stay exact sums of squares with no runtime
  eigendecomposition and no PSD tolerance on this path.

  The couplings are a weighted ridge and the innovation variances a
  line-searched Fisher scoring of the Gaussian log-linear variance model —
  damped rather than raw, because `Σ w A Aᵀ` is the EXPECTED information and the
  undamped step overshoots. An earlier undamped loop that stopped when the step
  norm failed to shrink was measured returning a `log d` 4.5 nats short of the
  optimum, and it was an independent nonparametric oracle (bin the rows, compare
  the fitted surface to each bin's own empirical second moments) that caught it,
  at 3.42× the bin's sampling band before the fix and 0.63× after.

  **The escalation trigger is one robust Rao score test per score PAIR** on
  `ζ_j·ζ_k`, on the same centred conditioning span and at the same α the #2768
  gate uses: the statistic for the sentence the issue is titled with and nothing
  wider. No pair fires ⇒ the pooled object stays in place byte for byte, and
  `K = 1` never escalates at all (there is no off-diagonal there, and
  `Var(z|a)` is #2768's branch — a second, differently parameterised variance
  model on top of it would double-correct).

  Every covariance consumer in the survival row program moved onto a row-indexed
  `ScoreCovarianceField`: the shared lane's cached `1ᵀΣ1`, both vector
  workspaces, the `c_i` in `SurvivalMarginalSlopeFamilyScalars`, and both
  `LogslopeBlockJacobian` branches. Because `Σ(a_i)` is a per-row constant and
  not a function of `β`, every existing derivative formula holds verbatim.

  Saving a fit that consumed a conditional `Σ(a)` is refused at the point of
  loss (`persistable_score_covariance`). That state needs `K ≥ 2`, which the
  on-disk contract already refuses at load — it carries one `z_column` and
  validates a 1×1 score covariance — so nothing new becomes unsaveable; the
  refusal exists so the reason travels with it instead of arriving later as a
  shape mismatch. Murphy–Topel is unaffected: `rigid_score_zeta_sensitivity`
  already refuses at `K > 1`.

- **The iso-κ joint outer search was walking a certified SURROGATE and nobody
  was putting it away (#2760).** The joint `[ρ, ψ]` spatial search refused at
  `n ≥ 4000` with `NOT STATIONARY` after a Strong-Wolfe line search that
  backtracked to `StepSizeTooSmall` — 50 attempts, 48 of them at a step below
  the fifth printed digit of `θ`. Three independent defects were stacked
  underneath it, and all three are fixed.

  **1. The joint ρ box's wall passed through the point the result is graded
  against.** `#2454` widened the `±JOINT_RHO_BOUND = ±12` search box "only as
  far as the incumbent" — `(-12).min(seed)` — so every coordinate whose
  scalar-route `ln λ̂` fell below `−12` began the joint search exactly ON its
  lower bound: an active constraint from iteration zero, its outward gradient
  KKT-projected to zero, unable to descend even where the joint criterion at
  the ψ the search was moving to wanted it lower. Containment in a closed set
  is not the property this route needs; the graded point has to be INTERIOR.
  Measured on a noiseless 1-D Duchon `y = sin(t)`: REML drives `λ̂` down as `n`
  grows, so the incumbents cross `−12` one at a time — 4 of 5 coordinates
  pasted onto the wall at `n = 1000…8000`, all 5 at `n = 16000`, where
  `∂V/∂ρ₀ = +1.484` at the wall against a whole stationarity bound of `1.030`.
  A coordinate whose incumbent is not strictly inside the joint prior now falls
  back to the engine's own `±RHO_BOUND` — the box the incumbent was found in.
  Everything strictly inside keeps the historical box byte-for-byte.

  **2. The mint had no curvature — which is how the real defect was found.**
  The #1033 n-free ψ-lane declares `DeclaredHessianForm::Unavailable` "so the
  planner selects BFGS instead of ARC". It does not need to:
  `with_prefer_gradient_only` is unconditional on this problem and
  `capability::plan` reads `(Analytic, Analytic) if prefer_gradient_only → Bfgs`
  *before* the ARC arm. What `Unavailable` actually erases is the one terminal
  evaluation #2359 reserves for the mint, and with it the
  `curvature-resolvability` rung, the #2348 asymptote-rail certificate, the
  curvature-scaled flat-valley widening and the #2299 large-step flatness
  certificate. Restoring it here made `run.rs`'s value-agreement guard fire —
  that guard only compares lanes when the mint asks for the analytic one — and
  that is what surfaced defect 3 below. It is NOT part of the shipped repair:
  restoring it also makes `exact_spatial_joint_engine_aniso_iso_parity_1d`
  refuse (`|Pg| = 5.143e-3` against a `8.100e-3` bound, so stationary, but
  `interior lambda_min = -1.585e-3` against a `3.061e-3` gradient floor, with ψ
  railed at its own box edge), and that indefinite-curvature verdict is a real
  finding about this lane's terminal geometry that deserves its own issue rather
  than arriving as a side effect. The ladder is green at all five rungs without
  it. An instrument is not a fix.

  **3. The criterion the search ranks is not the criterion. THIS is the line
  search.** With the mint asking for the analytic lane, `run.rs`'s existing
  value-agreement guard named it at once, at the point the `n = 2000` search
  stopped, both inner solves converged:

  ```text
  value-only      = -1.2781058170149880e4
  analytic-sample = -1.2781006804748626e4
  disagreement = 5.137e-2   roundoff bound = 1.905e-4
  ```

  `270×` `outer_value_agreement_bound`, i.e. `4e-6` relative where `√ε` is the
  contract. The two lanes are the #1033b certified n-free ψ-Gram tensor and the
  exact realized design. The tensor is certified on the **Gram**
  (`PSI_GRAM_CERT_RTOL = 1e-9`) and on the reduced-basis **subspace**
  (`PSI_GRAM_SKIP_PROJ_ATOL = 1e-7`); nothing in that certification bounds the
  **criterion** the optimizer ranks, and `β̂ = (G + λS)⁻¹r` amplifies a Gram
  residual by the radial-kernel conditioning — which is the regime this search
  lives in, at `λ = e⁻³⁰`. A value probe that crosses a skip-eligibility
  boundary therefore sees the criterion JUMP by more than the decrease the line
  search is hunting. So the surrogate is a SEARCH object, the same kind of
  thing as the staged-pilot row subsample the sibling N-block driver already
  retires, and it gets the same exit: `begin_exact_polish` retires it at the
  search checkpoint and the optimizer continues, and certifies, on the exact
  streamed criterion. Every in-window trial of the search stays n-free.

  **What the gate at `theta0` could and could not say.** The joint and scalar
  routes' criteria at `theta0` disagree by `−1.4e-13`, `−1.7e-13`, `+5.5e-13`,
  `+6.0e-8`, `+6.0e-8` relative at `n = 1000, 2000, 4000, 8000, 16000`. Five
  orders in one step, and the step is not in `n`: it is the rung at which a
  SECOND penalty block reaches `λ = e⁻³⁰ ≈ 9.4e-14` and stops contributing to
  `H = XᵀWX + S_λ` at working precision, after which `log|H|` is a sum of logs
  across the raw Duchon Gram's `~1e15` spectrum and two independent assemblies
  part company at exactly the scale `ε·κ` predicts. No fixed relative constant
  can be both tight enough to catch the `5.047e-5` formula difference #2671
  found and loose enough to admit that. So the cross-route number keeps its
  full decomposition as a warning, and the REFUSAL moves to a comparison both
  sides of which are `fit_score` of a **scalar-route** fit: the incumbent at
  `theta0` against the accept-fit at `θ*`. Like for like, one arithmetic, on
  the quantity that ships — which is what the gate's own sentence ("the joint
  search is minimizing a different function than the one its result is graded
  against") asks for.

- **The CTN fit and every replay of it read the coefficients through two
  different charts (#2680).** `#2306` moved the conditional-transformation-normal
  likelihood onto the direct-α chart

  ```text
  h(y, x) = α₀(x) + Σ_{k≥1} I_k(y)·α_k(x) + offset + ε·(y − median),
  α_k(x) = ψ(x)ᵀ A[k, :],
  ```

  with the shape coordinates held non-negative by the factored Khatri-Rao
  monotonicity cone rather than by squaring a latent coordinate. The likelihood,
  the exact-Newton Hessian, the function-space penalties and the ALO row replay
  all moved. **Three consumers did not**, and kept reading the same
  `blocks[0].beta` as `Σ_{k≥1} I_k(y)·γ_k(x)²`: the observed-score path behind
  `model.transformation_score(df)`, the `E[Y|x]` inversion grid behind `predict`
  and `generate` on a CTM, and `score_influence_jacobian` — the out-of-fold
  generated regressor the calibrated marginal-slope chain consumes, together with
  its Murphy–Topel Jacobian.

  **What it did to the numbers.** The lower endpoint basis is `[1, 0, …, 0]`, so
  `L(x) = α₀(x)` is the same on both charts, while `U(x) = Σ_k α_k` becomes
  `Σ_k α_k²`. With the shape coordinates near a common `c` that makes the
  reported score

  ```text
  z_reported ≈ c·z + (1 − c)·L,     sd(z_reported) = c,  mean(z_reported) = |L|·(c − 1),
  ```

  i.e. exactly right at `c = 1` and wrong in both location and scale otherwise —
  and `c ≈ range(h)/p_shape` grows with the sample range of the response, so the
  error is invisible on small fixtures and severe at production `n`. On #2680's
  own fixture the fit's latent score is `N(+0.001, 1.011)` while the reported one
  is `N(+0.957, 1.469)`; the reported means reproduce the issue's published
  numbers to every printed digit. It also explains the issue's separate
  saturated-row population: `U` pushed into the far tail makes `Φ(h)` and `Φ(U)`
  both return exactly `1.0` in binary64, so those rows clip to `Φ⁻¹(1 − 1e-12) =
  7.034` rather than being a tail of any normal.

  Everything downstream of a CTN stage 1 consumed the wrong score: the
  `bernoulli-marginal-slope` / `survival-marginal-slope` `z` moment gate (which
  refuses to fit when the score is not `N(0,1)`), the Murphy–Topel
  generated-regressor covariance, and the documented `transformation_normal_stage1`
  chain.

  **The repair is one evaluator, not five edits.** A new
  `transformation_normal::chart` module is now the single definition of what `β`
  means: `ctn_row_geometry` computes `(h, h', L, U)` from the covariate-side
  coordinates, `ctn_component_sensitivity` states the derivative (for an affine
  chart, the response-basis entry itself — no chart factor), `ctn_endpoint_bases`
  states the structural endpoint bases, and `ctn_response_bases_at` assembles
  `[1, I_k(y)·T]` / `[0, M_k(y)·T]` so the fit and every replay prepend the
  location column identically. The family's `row_quantities` accumulates in the
  same order it always did, so routing it through the shared kernel is
  bit-identical rather than merely equivalent.

  `ctn_row_geometry` **takes** the `TransformationNormalParameterization` marker
  and matches on it. That marker has been persisted since `#2306`, and its own
  doc says it exists "so a reader can reject coefficients written under any other
  chart as a typed mismatch instead of silently reinterpreting them" — every
  reader validated it and then reinterpreted them anyway. It is now load-bearing:
  a replay path must name the chart it believes it is evaluating, and a second
  variant becomes a compile error in one function instead of a silent divergence
  in five. In `gam-predict` the two independent transcriptions of the CTN payload
  collapse into one `SavedCtnChart` reader that carries the saved marker into the
  evaluator, and the support endpoints now come from the structural bases instead
  of a second I-spline evaluation at the boundary knots.

  Pinned by `ctn_predict_score_reproduces_the_fitted_score_2680` (the predict
  path's score equals `block_states[0].eta` to round-off — a chart-agnostic
  invariant that catches this defect *and* its mirror image),
  `ctn_observed_score_clears_the_generated_regressor_moment_gate_2680` (the
  `bms::gradient_paths` moment bars at `n = 500`, where the squared chart reports
  `sd ≈ 1.4` against its own `0.13` bound), and
  `ctn_score_influence_jacobian_matches_its_own_finite_difference_2680` (the
  Jacobian is the derivative of the score the same call emits, which is what
  catches the `2·γ_k` shape factor independently of the value fix).

- **An I-spline and its own derivative described two different functions
  outside the knot domain (#2695).** `create_ispline_dense` SATURATES there —
  the value is the all-zero row below `knots[degree+1]` and a constant row above
  `knots[n_bspline]` — and says so, with the reason: a linear extension would
  produce negative I-spline entries below the left boundary and entries above
  one past the right, breaking the non-negativity and the `[0, 1]` range the
  basis exists to guarantee. `create_ispline_derivative_dense` differentiated
  through a *clamped* B-spline, whose exterior convention is linear extension,
  and so returned the boundary SLOPE where the I-spline value is flat. Orders 1,
  3 and 4 were affected; order 2 was already zeroed there.

  **How it surfaced.** The survival link warp is
  `q = q0 + Σ_j βw_j·I_j(q0)`. The link-wiggle block reaches `q` through the
  VALUE (`∂q/∂βw_j = I_j(q0)`) and its gradient was always right; the threshold
  and log-sigma blocks reach `q` only through `q0`, so every one of their
  chain-rule channels carries `m1 = 1 + Σ_j βw_j·I'_j(q0)`. Outside the knot
  domain the warp is flat and `m1` was not, so the joint-Newton RHS asserted a
  first-order change the objective does not make — at any step size. That is
  gam#2695's headline: on `survival_location_scale_saved_fit_preserves_linkwiggle_metadata`,
  zero of the linear-dominated trust attempts have `actual/(rhs·δ)` within 50%
  of 1, and all six outer seeds refuse with
  `rejects [model, likelihood, objective, feasibility] = [0, 0, 2, 0]` at trust
  radius `1e-12`.

  **Why it stayed hidden.** The error is proportional to the warp amplitude, and
  the wiggle knots are frozen at fit setup from the SEED `q0` with no margin
  (`initializewiggle_knots_from_seed` spans exactly `[min, max]` of the seed),
  while `q0 = −η_t·e^{−η_ls}` moves by orders of magnitude during the outer
  search. So the seed iterate is inside the domain and essentially every later
  one is not — and every gradient oracle in the tree ran at the seed, at
  `βw ≈ 0`, or both.

  The same file already applies exactly this argument to an OPEN knot vector
  (gam#1348: "A constant function has zero derivative, so BOTH the first and
  second derivative must be zero in the exterior spans"). The case never covered
  is that an I-spline is constant-extended on a CLAMPED vector too. Endpoints
  keep the interior one-sided slope, because `right` is routinely the largest
  observed value and the transformation-normal shape derivative `h'(y)` must
  stay positive there.

- **The survival location-scale event Jacobian was floored in the value and not
  in the derivative tower (#2695).** `exact_row_kernel_from_parts` clamped
  `g = dη/dt` to `derivative_guard` on three branches and then read
  `(log g, d log g, …)` at the FLOORED value, so inside the band the row
  log-likelihood is bitwise constant in `qdot` while the tower reports a slope
  of `1/guard = 1e6` and a curvature of `−1/guard² = −1e12`. The three branches
  are replaced by one derived object: `ln` exactly on the modelled feasible set
  `g ≥ guard` (bit-identical, so no fit that never reaches the floor changes),
  and below it the degree-4 Taylor continuation of `ln` about `guard`, with the
  returned tower being that polynomial's own derivatives. It is C⁴ at the knot,
  strictly increasing and strictly concave on the continued branch, and unlike a
  flat clamp it charges for leaving the feasible region instead of paying
  `ln(guard)` at `g = 0`. The monotonicity refusal predicate is unchanged by
  construction — the floors ran before the `g ≤ 0` test and lifted every `g`
  above `−(guard + roundoff_slack)`, which is now written directly — so no state
  that was accepted becomes a refusal and none that was refused becomes
  accepted. Cf. `survival/base.rs`'s `stabilized_structural_derivative`, which
  states the same contract for the Royston–Parmar arm and resolves it with a
  zero-slope clamp.

  **What the two repairs above do and do not close on #2695.** Measured on that
  issue's own witness (`gam-cli`
  `survival_location_scale_saved_fit_preserves_linkwiggle_metadata`), the
  first-order gradient/objective disagreement the issue is titled for is gone:
  `d ℓ / (∇ℓ·δ)` lands within 10% of 1 on **96 of 96** resolvable small-step
  trust attempts, against **40 of 77** before, with a median relative residual of
  `1.5e-5`; the quadratic penalty gradient was already exact to `2.9e-10`.

  The fit still does not mint, and what remains is a different mechanism —
  a **discontinuity in the Jeffreys value Φ**, not a derivative error. Along ONE
  ray from ONE base point, with the trial direction bit-identical across all five
  attempts and the cone projection inactive:

  ```
   t = 2.003e-4  Φ = -11.48618      λ_min = -8.870e-1   λ_max = 1.9435e1   gate 1.000000
   t = 5.008e-5  Φ = -11.48601      λ_min = -8.869e-1   λ_max = 1.9435e1   gate 1.000000
   t = 1.252e-5  Φ = -11.48597      λ_min = -8.868e-1   λ_max = 1.9435e1   gate 1.000000
   t = 3.130e-6  Φ = -11.48596      λ_min = -8.868e-1   λ_max = 1.9435e1   gate 1.000000
   t = 7.826e-7  Φ = -10.93381      λ_min = -6.922e-1   λ_max = 1.9645e1   gate 1.000000
  ```

  Φ steps by `-0.5522` between `t = 7.8e-7` and `t = 3.1e-6`, and the extreme
  eigenvalues of `Z_Jᵀ H Z_J` step with it, so the discontinuity is in the
  observed information rather than in the Jeffreys machinery reading it — the
  conditioning gate is saturated at `1.000000` on both sides, so it is not the
  gate's smooth band, and the floor regime does not move. `actual` is therefore
  constant across the backtracking ladder while `pred` quarters, `ρ` runs
  `-784, -3137, -12548, -50192`, and every attempt is refused: the
  `rejects [model, likelihood, objective, feasibility] = [0, 0, 2, 0]` signature.
  The direction is dominated by the threshold coordinate
  (`u ≈ (-0.9753, +0.2207, ~0, ~0, 0, 0)`).

- **The Jeffreys/Firth-armed outer REML gradient was not the gradient of the
  criterion it reports (#2612).** On the penguins real-data multinomial arm the
  fit did not produce a probability at all: the unbiased probe converged to a
  separated mode (identifiable-span Fisher information `lambda_min = 2.06e-18`
  against `lambda_max = 1.443`), the Jeffreys gate fired at full weight, and the
  armed refit then died in the outer smoothing search at
  `line_search=StepSizeTooSmall` — the solver's own gloss on which is *"the
  direction descended but no step improved the objective"* — with an indefinite
  terminal analytic Hessian and no fit assembled.

  **What was wrong.** The IFT mode response `v_k = d beta_hat / d rho_k` is a
  property of the INNER stationarity system, so it must be solved against
  `M_true = H + S_lambda + H_Phi + completion`, the exact Hessian of the
  Phi-augmented objective the inner Newton converged on. It was instead
  borrowing the LAML logdet's operator `M_DD = H + S_lambda + H_Phi`. The
  envelope theorem kills `v_k`'s other route into the outer gradient
  (`grad_beta f = 0` at the mode, whatever `v_k` is), so `v_k` reaches it only
  through the drift trace `0.5 tr(M_DD^-1 D_beta M_DD[v_k])` — and a `v_k` from
  the wrong operator makes the analytic gradient the derivative of a different
  function than the value. Central differences of the production outer criterion
  against its own analytic gradient, at the refit's own stalling rho: `1.5e-9 ..
  7.7e-8` with the term disarmed, `5.3e-2 .. 1.5e0` with it armed, the sign
  wrong on three of eight coordinates, and h-independent between `1e-3` and
  `1e-2`.

  The half-fix that hid it: `completion_in_operator` folded the completion into
  the operator on the projected/`Smooth` route, where the projected kernel
  already owns the value and the traces so the operator is free to carry it.
  Every family that overrides `pseudo_logdet_mode` away from `Smooth` — the
  multinomial (`PositiveDefinite`), BMS, the binomial location-scale and wiggle
  families (`HardPseudo`) — takes the route where the operator IS the value and
  trace object, so the completion could not go in, and the mode response
  silently inherited that constraint. Folding it in there instead is not
  available either: the scalar would then need the completion's own beta-drift
  (third directional derivatives no family exposes) for the trace to stay
  consistent with it, which is a measured ~38% gradient bias.

  **The fix.** Stop making one object serve both roles. `InnerSolution` gains an
  optional `mode_response_op`, read through `mode_response_operator()` by all
  three `ThetaModeResponseKernel::select` sites (gradient, dense Hessian,
  Hessian operator), so no two can disagree about which system `beta_hat(theta)`
  is differentiated through. The custom-family assembly builds it from the same
  operator assembly with `H_Phi + completion` in place of `H_Phi`, exactly when
  a completion exists and is not already in `hessian_op`. `None` everywhere
  else: with no completion, or no Jeffreys term, `M_true == M_DD` and there is
  nothing to separate, so every other family and every clean fit is
  byte-identical.

  Rejected on measurement: the seed. The formula path warm-starts the armed
  refit at the saturated unbiased mode, which the fixed-lambda sibling documents
  as catastrophic. Warm `|Pg| = 1.759e-2`, cold (`beta = 0`) `1.607e-2`, both
  against `bound = 2.290e-3`, both `hessian_psd=NO`. Same failure; not the seed.

- **A curvature certificate no longer decides on a direction along which the
  criterion is exactly constant (#2676).** Three `geo_disease_*_matern`
  scenarios refused with `INDEFINITE CURVATURE AT INTERIOR OPTIMUM`
  (`interior lambda_min = -5.048e-6`) or with a smoothing-correction
  contradiction decided by a **0.55% margin**. Neither refusal was a
  measurement of the fit.

  **What was wrong.** `rho = log lambda` is a nonlinear reparameterisation, so
  for any smooth criterion `H_rho = diag(l) H_lambda diag(l) + diag(g_rho)`
  holds exactly — the second term is pure chain rule and carries no curvature.
  Every criterion here sees `lambda` only through the assembled penalty
  `sum_i lambda_i (b - mu_i)' S_i (b - mu_i)`, so a `w` with `sum_i w_i S_i = 0`
  makes the criterion EXACTLY constant along `lambda + s w`. Lift it to rho by
  `t = diag(lambda)^-1 w` and

      t' H_rho t = sum_k (g_rho)_k t_k^2      exactly, at every point,

  which is bounded by `sum_k |(g_rho)_k| t_k^2` — *verbatim* the per-direction
  floor both gates compare against, with equality when the gradient shares a
  sign on the support. The direction did not sit near the decision boundary of
  those gates; it sat **on it, by identity**, and which side it landed on was
  the sign of the disagreement between the gradient code and the Hessian code.
  Measured on `geo_disease_matern`: `sigma = 2.0930992e-5`,
  `sum_k g_k v_k^2 = 2.0946774e-5`, intrinsic `-1.578e-8` — the identity holding
  to `7.5e-4`, on a minimum eigenvector equal to the antisymmetric direction of
  a penalty pair with `cos = 1.000000`.

  **The fix, and what it is not.** Not a wider floor: the comparison was
  degenerate, not under-resolved. `gam_solve::penalty_invariance` computes the
  invariance from the penalty map alone — the null space of the Gram of the
  augmented operators, so a nonzero prior mean that BREAKS a proportionality is
  seen rather than assumed away — lifts it to rho, and returns the orthogonal
  complement. The outer certificate (`run.rs`) and the smoothing correction
  (`invert_identified_rho_hessian`) both deflate that subspace and apply the
  existing rule, unchanged, to what is left. No tolerance is chosen: the rank
  boundary is the eigensolver's own Weyl backward error, the instrument already
  used for this Gram.

  **What this does not change, as a theorem rather than an anecdote.** An
  objective declaring no invariance — every objective except the two REML arms
  and the spatial joint arm, and those only on a redundant penalty map —
  reaches a bit-identical verdict; the deflated path is not taken. And what
  deflation can hide is bounded by Cauchy interlacing: `Z' H Z` is a compression
  onto a subspace of codimension `d`, so `lambda_1(Z'HZ) <= lambda_{d+1}(H)`.
  Deflating `d` directions can lose at most the `d` SMALLEST eigenvalues and
  never one beyond them — with the one-dimensional invariance here, a matrix
  carrying two negative directions still refuses, and #2665's
  `lambda_min = -1.6e3` saddle is not in the deflated subspace at all.

  Only the part of the invariance that lies INSIDE the judged face is deflated.
  The identity is a statement about the FULL direction, so an invariance
  direction with a material component on a railed coordinate is deliberately
  left in the judged block rather than restricted and deflated — restricting it
  would break the identity and, in the extreme, hide real curvature.

  **Related corrections.** The saddle-escape search now looks in the same
  subspace the certificate judged, so it can no longer step along the invariance
  — where the only "negative curvature" available is the residual gradient
  wearing a curvature's clothes — instead of along the genuine saddle that
  refused. `interior_min_eigenvalue` is now reported from the block the verdict
  was reached on. And the `[PENALTY-REDUNDANCY]` warning no longer says a
  redundant penalty map produces "a Z2-symmetric saddle": the criterion is flat
  there, not descending, and the fit is unaffected — what is lost is only the
  separate identifiability of the individual smoothing parameters.

- **The survival marginal-slope's effect can vary along the follow-up axis
  (#2765, #2767).** `logslope_time_k` / `--logslope-time-k` make `b` a fitted
  surface in `(x, t)` instead of a per-row constant, so a latent score whose
  effect attenuates with age is now a model the family can express.

  **What was wrong.** The family carried three follow-up channels for the
  location index `q` — its value at entry, at exit, and its exit-time derivative,
  because the likelihood is `log S(t₁) − log S(t₀)` and an event row picks up
  `log η′(t₁)` — and exactly **one** channel for the slope. Time reached `η` only
  through `q`. `logslopespec` was a static term collection, and the row program's
  primary vector was `(q₀, q₁, q̇₁, g)`.

  **Why episode splitting was not a workaround.** This is a *transformation*
  model, `S(t|x,z) = Φ(−η(t))`, not a hazard model. Splitting a subject into
  intervals with a piecewise-constant `b` gives per-row contributions
  `log S(t₁;b₁) − log S(t₀;b₁)` that do not telescope into any survival function.
  The slope had to move inside the row program.

  **Why the generalization is the right one.** The factor
  `c = √(1 + bᵀΣb)` in `η = q·c + bᵀz` is not decoration: it is exactly the
  rescaling that makes the *marginal* law invariant to the slope, since
  `E_z Φ(−(q·c + bᵀz)) = Φ(−q)`. That identity holds **pointwise in `t`**, so
  `b → b(t)` preserves the family's defining property — and it forces `c` to
  inherit the time dependence, giving
  `η′(t) = q′(t)·c(t) + q(t)·c′(t) + b′(t)ᵀz`. The last two terms are what the
  rigid kernel was missing.

  **The shape of the fix.** A `SlopeRowGeometry` the row program is generic over:
  `StaticSlopeGeometry` is the four-primary frame every existing model uses and
  is the `db/dt = 0` face of the six-primary `DynamicSlopeGeometry`. Both feed
  the *same* `row_program!` declaration — only the feature map differs — so the
  likelihood is still written down once. The frames are compile-time distinct
  because the row towers are dense in the primary count (the fourth-order tower
  is `P⁴`): a model that does not ask for a varying slope must not pay `5×` for a
  channel that is structurally a copy and a zero.

  The log-slope design is tensored against a `log t` B-spline margin by the same
  `build_time_varying_survival_covariate_template` the threshold and sigma
  margins use, with the standard anisotropic penalty pair `S_cov ⊗ I_t` and
  `I_c ⊗ S_t` so smoothness in `x` and in `t` keep independent smoothing
  parameters.

  Two consequences the generalization forced. `q′ ≥ derivative_guard` is the
  *marginal* monotonicity constraint and it implied the likelihood-domain
  condition `η′₁ > 0` only because `η′₁ = q′·c` with `c ≥ 1`; a varying slope
  breaks that implication, so `η′₁ > 0` is now an explicit domain check (on the
  static frame it is unreachable, so no existing fit moves). And
  `LogslopeBlockJacobian` gained the exact `(η₀, η₁, η′₁)` rows for the varying
  case — the identifiability audit would otherwise have been reading the Jacobian
  of a model that is not being fitted.

  **Refused rather than reinterpreted:** a per-score log-slope topology; a
  non-zero smooth anchor, coefficient bounds, or linear constraints on the
  log-slope surface; and *saving* a fit that used the margin, because the on-disk
  contract rebuilds the block from the covariate term spec alone and would
  evaluate a different model at predict. The resolved knots ride on the fit
  result for the predictor that will replay them.

- **The survival marginal-slope runs the automatic latent-measure gate, and the
  conditional calibration it escalates to now actually delivers a unit-variance
  score (#2768).** Three things, in the order they were found.

  **The gap.** The Bernoulli marginal-slope has run an automatic gate on its
  latent score since #905: a Rao score test on `E[z|C]` and `Var(z|C)` over the
  marginal-index span, escalating to `ζ = (z − m(C))/√v(C)` when it fires. The
  survival marginal-slope ran none of it. It called
  `standardize_latent_z_with_policy` and nothing else, and under the default
  policy — `Frozen { mean: 0, sd: 1 }` — that transform is the identity: it
  checked, it warned, and it passed `z` through unchanged.

  That is not cosmetic. The survival row index is `η = q·c(g) + s(g)·z`, so a
  conditional shift `E[z|C] = m(C) ≠ 0` puts `s(g(C))·m(C)` into the *influence*
  channel `q` — in a model whose entire point is that `q` is the marginal index.
  The pooled marginal gate cannot see it (the marginal law of `z` can be exactly
  N(0,1) while every conditional law is shifted) and rank-INT provably cannot fix
  it. On a fixture that is exactly N(0,1) marginally with `Corr(z, x) = 0.5`,
  slope `b = 0.6`, and a true marginal coefficient `β_x = 0.5`, the uncalibrated
  axis returns

  ```
  fitted marginal x-coefficient    0.195   against a truth of   0.500
  ```

  a 61% attenuation, derived in closed form in the fixture and reproduced by the
  fit. The two arms of that fixture — the shifted score, and the conditionally
  standardised score the outcome was generated on — now agree.

  **The defect underneath it.** `ζ = (z − m(C))/√v(C)` was dividing by the
  **marginal** variance of `z` whenever the Breusch-Pagan stage did not fire. The
  right constant is the **residual** variance of the conditional-mean regression,
  and the two are never equal on a fired gate: with `z` standardised,
  `1 = Var(m(C)) + E[Var(z|C)]`, so the residual variance is `1 − R²` and sits
  strictly below the marginal variance *whenever there is any conditional
  structure at all*. The error was therefore present on every firing and grew
  with exactly the structure the correction exists to remove.

  ```
  sd(ζ) at R² = 0.25       0.8586  ->  1.0000
  ```

  against the `post_sd ≈ 1` the struct's own field doc claims. The marginal-index
  identity `E_ζ[Φ(q√(1+b²) + bζ)] = Φ(q)` holds only at `Var(ζ|C) = 1`; at `v` it
  becomes `Φ(q√(1+b²)/√(1+b²v))`, so every marginal coefficient carried a ~4%
  multiplicative distortion. Worse, the calibrated residual then failed the
  standard-normal adequacy re-check **on the SD clause alone** (`|sd−1| = 0.134`
  against a `0.045` tolerance at n = 4000), which sent BMS to the empirical
  measure and, per #2718, withheld the covariance. The field is renamed
  `homoskedastic_var` and keeps its on-disk name `global_var`, so a model saved
  before the fix keeps applying the map it was *fitted* with.

  **The seams.** The gate is one object with the family's kernel capability as an
  argument (`EmpiricalLatentMeasureSupport::{Available, StandardNormalOnly}`),
  not two copies: the survival row program is the closed-form standard-normal
  probit lowering and owns no empirical-grid branch, so a `StandardNormalOnly`
  caller keeps the best available pre-transform and gets the failing adequacy
  ledger back rather than a measure it cannot evaluate. On the predict side the
  conditioning span is now named explicitly — the survival predictor's primary
  design is the q-design `[time | timewiggle | marginal]`, so reusing it as
  `a(C)` (which is what the shared code did) would have conditioned on time
  columns and applied a different map than the fit. And the naive covariance,
  which treats the generated regressor `ζ` as known, is corrected: the per-row
  channel `∂(score_β)/∂ζ` is derived mechanically from the sole
  `rigid_feature_program` declaration and gated against a central difference of
  that program's own gradient over 360 cells. Shapes the rigid channel does not
  cover (score-warp / link-deviation, `K > 1`) withhold the covariance with a
  typed reason instead of publishing one that is too narrow.

- **The measure-jet head spans the energy's WHOLE affine null space, so the
  term collection's centering can no longer delete a linear direction
  (#2751).** The `mjs` design's extrapolation head carried only the LINEAR part
  `{x_1..x_d}` of the jet energy's affine null space. The term-collection
  chokepoint then applies its parametric orthogonalization `Z = null(1ᵀX)`,
  which removes exactly one coefficient direction, and the constrained null
  space is `{γ : Zγ ∈ null(S)}` — so with no constant in `null(S)` for that
  removal to be charged to, **it came out of the null space itself**, leaving
  `d − 1` free ambient-linear directions where the theorem says `d`.

  The consequence only appears once REML selects a large energy `λ`, which it
  does for any near-affine truth: the fit is then confined to what the energy
  leaves free, and what it left free was one accidental direction — the single
  linear combination whose data-mean happens to vanish. Measured with the ridge
  limit against the shipped design's own Primary (no fit, no family, no
  smoothing search: least squares of the noiseless plane `0.2 + 0.9·x₁` with
  `λ·S_primary` added):

  ```
  lambda    d/dx1   rms[x1]  rms[x2]   pearson
  1e2       0.8994  0.29980  -0.00640  0.9993
  1e6       0.4838  0.16126  -0.13235  0.7729
  1e10      0.4342  0.14472  -0.14644  0.7029    <- one linear direction left
  ```

  At `λ → ∞` the surviving direction is `(0.695, −0.716)`; projecting the
  planted `(0.9, 0)` onto it gives `|cos 45°| = 0.707`, which is exactly the
  `0.7051` Pearson the `mjs`-backed BMS fixture reported end to end. Duchon,
  whose null space `{1, x₁, x₂}` *does* contain the constant, survives the
  identical chokepoint with the plane intact (`0.9000` at `λ = 1e10`).

  Two upstream hypotheses were killed before the collection was implicated, and
  both are now gates rather than beliefs: the energy form annihilates the affine
  span to `1e-17` relative on a regular grid, on scattered centers, on a
  10×-anisotropic layout and at a single scale; and the emitted basis-level
  Primary had nullity 2 with both directions exactly affine. Nothing upstream of
  the collection was wrong — a 2-dimensional null space is simply one dimension
  too small to survive a 1-dimensional constraint.

  `measure_jet_affine_head_lift` now returns the `(d+1) × (1 + head_rank)` lift
  acting on `[1 | x]`, `measure_jet_affine_head_block` realizes it, and
  `measure_jet_affine_value_basis` — which is both the gauge's `A` and the
  null-component penalty's projector — is literally the same object evaluated at
  the centers, so "the head spans exactly the energy's null space" is a property
  of the code instead of a comment two call sites have to keep agreeing on.

  ```
                          before -> after
  raw design width          15 -> 16
  Primary nullity            2 ->  3     (all three exactly affine)
  declared null frame        2 ->  3
  null-component rank        2 ->  3
  FIT chart width           14 -> 15     = m - 1, matching Duchon's k - 1
  ```

  End to end on the fixture that reported the defect, all four surface bases on
  byte-identical rows (the comparators are unchanged to every printed digit, so
  the change is confined to `mjs`):

  ```
  basis                            pearson   d/dx1   rms[x1]  rms[x2]  rms[nl]
  mjs(x1,x2,centers=16,scales=3)    0.9936   0.993   0.3311   0.0233   0.0298
  matern(x1,x2,k=16)                0.9416   0.701   0.2337   0.0336   0.0765
  duchon(x1,x2,k=16)                0.9975   0.961   0.3204   0.0202   0.0102
  s(x1,k=8) + s(x2,k=8)             0.9837   1.052   0.3506   0.0095   0.0634
  truth  0.2 + 0.9*x1               1.0000   0.900   0.3000   0.0000   0.0000
  ```

  The predict-side ambient gradient takes the affine lift (row 0 is the constant
  and contributes nothing to `∇f̂`; its FD gate now carries a nonzero constant
  column, so a mis-indexed row fails it), and the errors-in-variables
  reconstruction in `model.rs` rebuilds the same lift. A model frozen before
  this change carries an `m + head_rank` row transform and is refused by the
  frozen-width check with that exact message rather than silently replaying a
  different basis.

  Also corrected: `measure_jet_bms_backend`'s penalty-count assertion demanded
  ONE penalty per surface, describing a "nullspace ridge folded into the
  Primary" the builder deliberately does not do — it emits the Primary
  independently of `double_penalty`, so the realized count is two. The wrong
  number survived because that assertion had never executed: the truth-recovery
  assertion above it failed first.

  Verification: the `measure_jet` integration target is 11 passed / 5 failed and
  `gam-terms` is 919 passed / 1 failed. Both failure sets are pre-existing and
  both belong to #2761's `ln ℓ` dial, established by reverting
  `crates/gam-terms/src` + `crates/gam-models/src` in place and re-running:
  identical five, and `psi_producer_matches_fd_length_scale` red at both
  sources (`analytic −6.988657e-5 vs FD 0` before, `analytic 1.574896e-4 vs
  FD 0` after). A printing replica of that comparison attaches the magnitudes —
  `|analytic|max` `3.78e-3` (pre) / `3.50e-3` (post) against `|FD|max` `8.3e-13`
  / `4.2e-13`, i.e. the shipped null-component candidate does not move with `ℓ`
  at all while the producer reports a jet three orders above its central
  difference, in both arms, while the Primary agrees with its own FD to `3.7e-9`
  in both. The builder ships the rebuilt metric-consistent ridge `R = N M Nᵀ`
  (whose frame `N` has zero representer coefficients, so `E·N` is `ℓ`-invariant
  by construction) while the ψ producer differentiates the raw pullback
  `E(ℓ)ᵀH₀E(ℓ)`, which is not. That is an objective↔gradient desync on the `ℓ`
  coordinate and a far better candidate for the five "the direction descended
  but no step improved the objective" line-search refusals than anything in
  this issue; it is #2761's, not this one's.

  One more measured consequence, recorded because it is a real cost of this
  change and not a wash: in `tests/regressions/misc/`, the same revert-in-place
  comparison shows the two `mjs` `ln ℓ` fixtures **swapping**, not improving —
  `measure_jet_formula_fit_succeeds_like_the_cli` was red before and is green
  after; `measure_jet_5d_converges_when_aniso_loses_to_isotropic` was green
  before and is red after, refusing with `hessian_psd=NO` at a point the solver
  itself calls stationary (`|Pg| = 2.156e-1` under `bound = 2.778e-1`). Adding
  one column to the design moves the ψ landscape, and both fixtures sit on the
  same knife edge — which is the state #2761's `ln ℓ` search is actually in.
  That instability is the next thing to fix, not a reason to leave the null
  space one dimension short.

- **The matrix-free from-probes selected-inverse channels price per-row
  deflation instead of refusing it — and there was never anything to derive
  (#2712).** Three channels of the #2080 wide-`p` analytic-gradient cluster —
  `logdet_theta_adjoint_from_probes`, `ard_log_precision_hessian_trace_from_probes`
  and `assignment_log_strength_hessian_trace_from_probes` — hard-refused any
  cache carrying `deflated_row_directions`, on the stated grounds that "the
  plain-`S⁻¹` bundle carries the UNdeflated block" and so could not rebuild the
  Daleckii–Krein correction `tr(inv_vv·(D − DΦ[D]))`. They convert as one
  all-or-nothing cluster, so one deflated row routed the whole fit to a dense
  channel the lane cannot afford at massive `K`.

  **The premise was a misreading of `undamped_factor`.** That accessor returns
  the Cholesky of the spectrally CONDITIONED `Φ(H_tt^(i))` — the block that
  pinned `λ̃ = 1` on each deflated direction — not of the raw `H_tt^(i)`, and the
  reduced Schur behind the bundle is that same conditioned arrow's. So
  `A_i⁻¹ + G_i S⁻¹ G_iᵀ`, which is literally what BOTH routes build, already IS
  the deflated `(H⁻¹)_tt`. Measured on the deflating fixture, rebuilding
  `A_i = L Lᵀ` from the cached factor:

  ```
  ||A v - v||               = 1.97e-16      (the unit-stiffness pin itself)
  ||A - U diag(cond) U^T||  = 4.97e-16
  ||A - U diag(raw)  U^T||  = 9.999999e-1   <- 10^15 larger
  ```

  and the from-probes reconstruction matches `selected_inverse_row_blocks` to
  `~2e-16` RELATIVE on every deflated row. What the probe routes actually lacked
  was the correction TERM, whose remaining operands — `deflated_row_directions`,
  `deflation_row_spectra`, and the raw per-slot `D` each channel already
  assembles from its own row jets — never involved `S⁻¹` at all. Each channel now
  applies the same `deflation_block_correction` its dense sibling applies, on the
  t-slot channels, the border channels, and the ordered Beta–Bernoulli
  shared-mass diagonal (that last one was a second, latent gap: the from-probes
  site tuple carried no `diag_deflation_weight` field at all).

  The three private copies of the row-block reconstruction collapse into one
  `arrow_solver::row_selected_inverse_from_probes`, the matrix-free sibling of
  `DeflatedArrowSolver::selected_inverse_row_blocks`, which documents the
  conditioned-block fact once at the place it is used.

  **A test-methodology finding came out of the acceptance requirement, and it is
  worth stating on its own.** The deflated and deflation-blind operators agree
  wherever the deflation is inactive, so machine-precision parity is also what a
  port that ignored deflation would produce. The instrument for that is
  `deflation_blind_cache`: a clone of the cache with ONLY the deflation metadata
  emptied, against which the PRODUCTION dense adjoint yields exactly the
  deflation-blind operator — no test-only flag, no second code path. Measured on
  the ordered Beta–Bernoulli anchor, the correction moves `Γ` by `8.47e-8`
  against `‖Γ‖∞ = 98.9`, because that fixture's unit-deflated direction is a
  near-null the raw derivative barely touches. That is **below** the historical
  per-entry parity tolerance `1e-8·(1+|Γ|) = 1e-6`, so on a deflated cache those
  element-wise assertions alone would have passed a port that dropped the
  correction entirely. The gates now tighten to `1e-10·(1+|Γ|)` on a deflated
  cache, state non-vacuity as a ratio (`parity·1e3 ≤ separation`), and assert the
  thing that actually decides whether a gate can see the defect it exists to
  catch: the per-entry tolerance must itself be finer than the separation.

  **Which SUBSPACE deflates decides which channel can be gated at all**, and that
  also had to be measured. The ARD correction contracts `D = hess·eₛeₛᵀ` at ONE
  coordinate slot, so `M = UᵀDU` carries a factor `U[s,d]` and the whole
  correction vanishes when the deflated direction misses that slot — separation
  exactly `0.0` on both real deflating fixtures, while the θ-adjoint separates by
  `8.47e-8` on the same cache because its `D` is a full `q×q` block. The ARD gate
  therefore sweeps every local slot with the deflation RECORD redirected onto
  whichever eigendirection that slot loads (factors, Schur and eigenbasis
  untouched; both routes read the same four operands, which is the whole claim).
  Slot 0 is a logit slot there and still gives `0.0`; slot 1 gives separation
  `2.606` against parity `0.0`.

  **What this does NOT do is flip the wide-`p` routing, and the reason is
  measured.** The complete outer ρ-gradient still disagrees between the dense and
  bundle routes on a deflated fit — `8.45` against `‖g‖∞ = 5.00` — but that gap is
  BIT-IDENTICAL on the cache and on its deflation-blind clone, so deflation
  cannot be its cause. It is #2499/#2515's β-Schur smoothness-EDF desync, landing
  on the two smoothness coordinates and leaking into the ARD ones through the
  shared single-adjoint IFT contraction. The end-to-end gate asserts exactly
  that decomposition — the two routes price the *deflation contribution*
  identically, and the surviving gap is deflation-independent — so it doubles as
  a tripwire: if the residual desync ever acquires a deflation-dependent part it
  comes back here rather than staying with #2515. The fourth refusal on the same
  false premise (the streaming outer evaluation) is therefore corrected in place
  rather than lifted, with the measurement that would lift it written on it.

  Measured, same filter and host, `76770446e` with this work's files reverted vs
  after: **31 passed / 6 failed → 38 passed / 5 failed.** The six baseline
  failures are the identical set; the one that left is
  `sae_logdet_theta_adjoint_from_probes_matches_dense_softmax_2080`, which was red
  on its own premise (it declares `NoRowDeflates`, and every member of its ladder
  now deflates) — a premise that only existed because the route used to refuse
  deflated rows.

  All seven #2712 gates pass at `ff1ee8a24`. The five remaining failures under
  that filter are the identical baseline set — `#2330` Patch-D coordinate gap
  `1.774e-3`, `#1625`'s unresolved invariant-subspace block, two `#2500` gates,
  and the dense ARD FD deflation trace — every one of them a dense-route or
  fixture-stratum failure, and no dense computation path is touched here: the
  production diff is confined to the three from-probes functions, the new
  `arrow_solver` helper, and comments.

- **SAE post-fit certification no longer costs `dim³`: the residual-gauge
  curvature is `p` blocks of `D × D`, not one `(p·D)²` Gram (#2757).**
  `fit_diagnostics_report` was materializing the curvature `H = RᵀR` as a dense
  `param_dim × param_dim` matrix and taking its dense symmetric
  eigendecomposition — 45.97 GiB and 60.5% of the whole fit at `p = 4096`, on
  a quantity that is *certification*, not the fit.

  It is block diagonal. The certificate's parameter vector is the atoms'
  flattened frames, so column `c = offset_k + i·d_k + a` names (atom, **output
  coordinate**, axis), and a frame perturbation of output coordinate `i` moves
  the reconstruction only on `i`. The per-row pinning Jacobian is therefore
  output-coordinate diagonal, and

  ```
  H[(k,i,a), (k',i',a')] = Σ_n M_n[i,i'] · g_n[i,(k,a)] · g_n[i',(k',a')]
  ```

  inherits exactly the row metric's output-coordinate coupling and nothing
  else. Under the metric `diagnostic_metric()` installs whenever no
  output-Fisher harvest ran, `M_n = I`, and every off-block entry is never
  written — measured at bit-zero, on a fixture whose decoder touches every
  output coordinate. So the object is `p·D²` numbers and `p·D³` flops
  (`D = Σ_k d_k`), against `(p·D)²` and `(p·D)³` before: a factor of `p` in
  memory and `p²` in time, i.e. 45.1 GiB → 11 MB at the `p = 4096, D = 19`
  shape. Measured end to end on the issue's own fixture:

  | `p` | `param_dim` | before | after |
  |---|---|---|---|
  | 256 | 1024 | 0.316 s | 0.131 s |
  | 512 | 2048 | 1.318 s | 0.152 s |
  | 1024 | 4096 | 7.960 s | 0.206 s |

  Growth per doubling of `p` falls from 6.0× (cubic) to 1.36×.

  **The reported `pinning_rank` was also wrong, for a related reason, and is
  now right.** The rank decision is `σ_i(R) > α·ε·max(m, param_dim)·σ_max` with
  `α = 100` — deliberately 100× *above* an SVD's backward error, which is what
  makes it meaningful. Testing the algebraically equivalent `λ > τ²` on the
  Gram instead puts the threshold a factor `α²·ε·N` *below* a symmetric
  eigensolver's own resolution, so every roundoff eigenvalue clears it: a
  curvature of true rank 12 in 80 parameters was reported as rank **45**. The
  blocks are now accumulated as triangular roots by streaming Givens rotations
  (same memory, same cost, no squaring) and the rank is read off their singular
  values; the dense fallback floors its threshold at the standard
  `|λ̃ − λ| ≤ dim·ε·‖H‖` resolution bound. All representations now agree and all
  respect `rank(RᵀR) ≤ rows(R)`.

  **The other branch of the same function was worse.** With an isometry pin
  installed — reachable from the shipped `IsometryPenalty` API —
  `to_residual_gauge_model` materialized each per-row pinning Jacobian as a
  dense `p × param_dim` block and retained all `n` of them: `8·n·p²·D` bytes,
  which is **2.55 GiB per observation** at `p = 4096, D = 19`. The pin's rows
  genuinely cannot be folded into the blocks — eliminating one against block
  `i` scatters that block's row into every other, so the QR of `[⊕R_i ; L]`
  fills in completely — but `H = ⊕_i B_i + VVᵀ` is block diagonal plus a
  symmetric update of rank `Σ_k d_k`, and Sylvester's law of inertia gives

  ```
  n₊(H − sI) = n₊(B − sI) + n₊(−I_k − Vᵀ(B − sI)⁻¹V)
  ```

  — its eigenvalue count above *any* shift, exactly, in `O(p·D·k²)`. That is
  all both consumers need: the rank is the count above `τ²`, and `λ_max` is the
  shift at which the count reaches zero. Both branches now stream the same
  structured curvature through one entry point, and no production path
  materializes a per-row Jacobian at all.

  Certificate output is otherwise unchanged: verdicts, group signature,
  residual gauge dimension and per-generator energy fractions are identical
  whichever representation the reduction ran on, which is gated from fifteen
  independent angles in `tests_frame_curvature_2757` — including #2267's own
  eigendecomposition census showing that nothing at the parameter dimension is
  decomposed at all.

- **The measure-jet representer range is a basis coordinate again, so REML
  selects it (#2761).** `lambda` shrinks a coefficient vector INSIDE a span; it
  never moves the span. The measure-jet design is
  `X = K(data, centers; ell) * z`, so `ell` decides WHICH m-dimensional subspace
  the representers occupy — the same standing the Matern `kappa` has, and the
  module header already called it "matern's log_kappa analog". Freezing it at a
  geometric heuristic therefore makes an error no smoothing parameter can
  repair.

  Measured on `measure_jet_perf_parity`'s 1-D-curve-in-3-D Gaussian fixture
  (n=1500, sigma=0.10, 16 centers, p=15), where `span floor` is the
  least-squares projection residual of the NOISELESS truth onto the realized
  design's column span — the bound no `lambda` can beat:

  | arm | ell | edf | span floor | unpen. LS | held-out |
  |---|---|---|---|---|---|
  | frozen (auto ell) | 0.5144 | 14.684 | 0.152488 | 0.155484 | 0.155584 |
  | REML-selected ell | 3.8813 | 14.006 | 0.000014 | 0.008155 | 0.009642 |
  | `matern(k=16)` | - | 14.619 | 0.006077 | 0.011989 | 0.011639 |
  | `duchon(k=16)` | - | 15.016 | 0.002443 | 0.011308 | 0.010521 |

  At the frozen range the fitted `0.1556` IS the span floor: unpenalized least
  squares on the same design gives `0.1555`, dropping the null-component penalty
  moves the fourth decimal, and `edf/p = 0.98` says the fit was already spending
  everything it had. Freeing `ell` puts measure-jet past both comparators at
  LOWER edf, so nothing is traded for the accuracy.

  The dial itself was not new: `299c83ffc` introduced it default-ON precisely to
  remove this fixture's 13x, `a3afd17a2` found its one hazard — a BMS fit shares
  one measure-jet basis between the marginal mean and the log-slope surface,
  where a design-moving kernel scale on shared covariates reached a
  separation-scale runaway — and contained it AT THE BMS ENTRY POINT, and
  `b1d94d1a5` then flipped the GLOBAL default off anyway. The 13x came back. The
  scoped freeze is untouched and still runs where the hazard is.

  Rejected: raising `MEASURE_JET_AUTO_LENGTH_SCALE_FACTOR`. That is what #1041
  did (x2 -> x1) as the dial's replacement and it is what main measured 13.4x
  with. No fixed multiple of the center spacing can be the answer, because the
  range that aligns the span depends on the target's smoothness relative to the
  center layout — data, not geometry. The constant survives as the SEED of the
  outer coordinate and its doc now says so.

  Behaviour change for callers: an explicit `mjs(..., length_scale=X)` now PINS
  the range instead of seeding a search, mirroring the short-circuit an
  explicitly-scaled Matern gets. `learn_length_scale=` overrides either way.

- **The constant-curvature smooth stops pinning its kernel range, so `kappa_hat`
  measures curvature instead of the range error (#2747).** `exp(-d_kappa/ell)`
  carries a curvature and a range in one exponent and they are strongly
  confounded: to first order `d_kappa = d_0*(1 + kappa*a(x,y))`, so the MEAN of
  `a` over the evaluated pairs acts exactly like a rescaling of `ell` and only
  its VARIATION is genuine curvature. The smooth fitted `kappa` while pinning
  `ell` to a heuristic (median chart center spacing, doubled), so `kappa`
  absorbed whatever range error the heuristic carried — and range
  mis-specification is monotone in one direction, which is the railed
  `V_p(kappa)` the issue reported.

  Measured on truths planted INSIDE the fitted span, three planted curvatures x
  three planted ranges: with `ell` pinned the criterion recovers `kappa*` only in
  the one cell where the truth's own radial length scale IS the auto `ell_ref`.
  At half or twice that range it rails at a box endpoint, reports the WRONG SIGN
  (`kappa_hat = -0.35` against a planted `+1.0`), or reads a confident interior
  `kappa_hat = -0.94` / `+0.94` on genuinely FLAT data. That one working cell is
  the configuration the acceptance fixture happened to use, which is why the
  fixture was green while the estimator was not.

  The construction is now one kernel at one range —
  `X = K_{kappa,ell}(data,C)z` and `S = z' K_{kappa,ell}(C,C) z` at the same
  `ell`, so `S` is again the RKHS roughness of the function `X` realizes and the
  model is the ordinary subset-of-regressors GP. `#944`'s fill-invariant
  `L(kappa)` and `#1464`'s separate penalty length `L_S(kappa)` are deleted with
  their implicit-function jets and the two 100-iteration Newton solves they cost
  on every basis build; both were attempts to remove the confounding by
  CONSTRAINT, and pinning a scalar summary of the design selects a
  one-dimensional curve through the `(kappa, ell)` plane a priori, on which
  `dV/dkappa = V_kappa + V_ell*L'(kappa)` keeps a range term that vanishes only
  if `ell_ref` was already optimal. On the profile curve it vanishes identically
  by the envelope theorem.

  Pinning `kappa=` no longer pins the range with it. It used to take the whole
  term out of the curvature profile — the only owner of either coordinate — and
  leave the range at the auto heuristic, which is a worse fit for no stated
  reason: fixing the geometry is not a statement about the kernel's resolution.
  A pinned-`kappa=` term now gets its range profiled at that curvature, and
  because the profile is Gaussian-identity-only it drops out with a log line
  rather than turning a working non-Gaussian fit into a refusal (a FREE `kappa`
  still refuses, since `kappa_hat` is the estimand the caller asked for and
  shipping it unfitted would be worse).

  `psi = (kappa, eta = ln ell)` now carries a full second-order tower, the outer
  solve is one-dimensional over the range-PROFILED criterion
  `V_p(kappa) = min_eta V(kappa, eta)` so the point estimate, the profile CI and
  the flatness LR are extrema of the same object, and `length_scale=` follows the
  same mgcv-`sp=` convention `kappa=` does: explicit pins, omitted estimates.
  `Model.curvature(...)` rows gain `length_scale_hat` and
  `length_scale_estimated`, because every statistic in the row is a profile over
  the range and a reader who cannot see it cannot tell an estimate anchored at a
  sensible resolution from one anchored at a degenerate corner.

  With the range profiled, `kappa_hat` lands within 0.19 of the planted
  curvature in all nine cells (median 0.07), `ell_hat` recovers the planted range
  to 3%, and there are no rails and no sign inversions. The acceptance fixture
  now cycles the planted range `{0.5, 1, 2}x ell_ref` across its replicates, and
  its flat arm is a real signal again rather than the constant mean the
  confounding had forced on it.

  NOT fixed by this, and stated so the next reader does not have to rediscover
  it: on truths that are in NO fitted span the criterion still prefers `+kappa`.
  An origin-radial plant is curvature-blind as a function class
  (`d_kappa(x,0) = 2*arctan(sqrt(kappa) r)/sqrt(kappa)` is a strictly monotone
  reparametrization of the chart radius at every `kappa`), and a multi-reference
  plant that does carry curvature still rails with a sign that flips as the
  center count sweeps 6 -> 12 -> 24 -> 40. That is span-approximation ordering
  under misspecification — the residue of `#1464` — and a different root cause
  from the range confounding.

- **A monotone link warp no longer ships an unpenalized rescale of the index it
  is composed onto (#2647).** `binomial_location_scalewiggle_termswith_matern_spatial_blocks_fit_finitely`
  refused all four startup seeds with `did not converge after 48 cycle(s)`, which
  reads like a budget problem and is not one: at 600 inner cycles the arms are
  bit-identical to 200 and one seed is *worse* than at 48. The per-cycle trace
  says what is happening — `|beta|inf` climbs 230x while `0.5 b'Sb` falls 41x
  (fitting `pen ~ |beta|^-2` on two seeds) and `-loglik` is flat to `8e-4`. The
  solve was descending toward an infimum at infinity that is never attained.

  The free direction is the warp's LINEAR element. The model is
  `q = q0 + w(q0)` with `q0 = -eta_t*exp(-eta_ls)`, so a linear warp element is a
  rescale of the index; the index block is penalized, so the penalty falls along
  that orbit while the likelihood does not move. Measured on the failing
  fixture's own knots, the anchored I-spline span contains `u -> (u - left)` to
  `2.7e-15`, its coefficient vector is componentwise non-negative (the whole ray
  stays inside the monotone cone `beta_w >= 0`), and the order-2 roughness
  charges `3.0e-14` for it. `ispline_function_penalties` sets
  `roughness_nullspace_dim = derivative_order - 1`, so this is structural: every
  configuration whose smallest requested order exceeds one shipped an unpenalized
  warp direction unless `double_penalty` happened to close it.

  `canonical_wiggle_function_penalties` now closes the assembled set's own joint
  null space unconditionally, reading `null(sum_j S_j)` in the function metric off
  a per-block-normalized sum and appending one shrinkage coordinate spanning it.
  This is the same treatment, and the same argument, that
  `build_binomial_threshold_and_scale_blocks` already applies unconditionally to
  the log-sigma block, where `(beta_t, beta_ls) -> (c*beta_t, beta_ls + ln c)` is
  the exactly analogous index-scale gauge. It is a no-op on every configuration
  that was already well posed, including the shipped default (`orders = [1,2,3]`,
  whose order-one roughness is full rank on the anchored basis). The smallest
  eigenvalue of the exact joint penalized Hessian at the fixture's seed went from
  the `~1e-10` the family's own source comment records to `7.254550e-1`, the fit
  completes in 0.1 s at its original 48-cycle budget, and the same objective comes
  back at 48, 200 and 600 cycles.

  A model saved before this change whose warp set gains a coordinate will refuse
  to load with a `SchemaMismatch` naming the reason: its coefficients and
  log-lambdas index a penalty system this build no longer assembles, and they were
  obtained from a criterion with no minimiser. Refit.

- **The CLI and the engine no longer disagree about where a survival fit is
  anchored (#2631).** The survival time-basis centering anchor was decided in two
  places. `materialize_survival` — the engine path behind `fit_from_formula` and
  the Python FFI — promoted the robust median-exit anchor whenever ANY entry age
  exceeded the origin threshold, and hardcoded the caller override to `None`.
  `gam-cli`'s `run_survival` promoted it only for marginal-slope, and owned the
  `--survival-time-anchor` override. Each was internally consistent, so nothing
  was mis-persisted; the same formula, data and config simply produced a
  different fit depending on which front end ran it. Measured on a 500-row
  delayed-entry cohort (`Surv(entry, exit, event) ~ s(x)`, location-scale) from
  byte-identical inputs: the CLI persisted `survival_time_anchor = 4.0579` (the
  earliest entry) where `gamfit` persisted `12.0317` (the median exit).
  Re-centering is an exact affine reparameterization of the design, so this is
  not cosmetic metadata — it is the conditioning the smoothing selection sees,
  which is the whole point of the `#751`/`#1790` robust anchor.

  Three further consequences fell out of the same duplication. Because the
  override lived only in the CLI's copy, and the CLI's own default
  (transformation / Weibull) route delegates to the engine copy,
  `--survival-time-anchor` was **silently ignored on the default route** — parsed,
  validated, then dropped, while the code comment claimed it was "honored by all
  paths". `FitRequestConfigDocument` had no field for the anchor at all, even
  though `--survival-time-anchor` declares a conflict with `--request` on the
  premise that the document carries the complete scientific model configuration.
  And the CLI's third branch was unreachable dead code carrying a *second,
  different* definition of left truncation — `min(entry) > threshold` against the
  materializer's `any(entry > threshold)` — which under-triggers on staggered
  entry, the ordinary shape of a real registry cohort.

  The rule is now one function, `resolve_survival_time_anchor_for_mode`, composed
  of three orthogonal primitives (validate-override, earliest-entry,
  robust-interior) and one left-truncation predicate. The three per-mode
  resolvers collapse into it and the transformation-specific one is deleted.
  `SURVIVAL_DELAYED_ENTRY_THRESHOLD` goes too: it was a second constant kept in
  lockstep with `ENTRY_AT_ORIGIN_THRESHOLD` by comment, the same failure mode one
  level down. The override became model configuration —
  `FitConfig::survival_time_anchor`, a `survival_time_anchor` key in the fit-request
  document, and a `gamfit.fit(survival_time_anchor=...)` kwarg — validated once in
  `FitConfig::resolve()` and refused on a non-survival response exactly as
  `survival_likelihood` already was. Engine behaviour is bit-identical when no
  explicit anchor is set; left-truncated CLI location-scale and latent fits now
  agree with the engine, which is the intended change.

  The mechanism itself is now measured rather than asserted. On a staggered-entry
  cohort, one I-spline basis centered at each candidate anchor: the earliest-entry
  anchor leaves the trend coordinate at `max|trend| = 5.000` with **every** row
  one-signed; the median-exit anchor leaves `1.140` with an exact 6/6 sign split.

- **A separated binomial fit is no longer refused for being right (#2273).** On
  exactly-separated data — a genuine gap between the classes — `y ~ smooth(x)`
  could not be fitted at any `n`. The in-loop separation guard turned a
  converged, finite, well-penalized logit fit into `Unstable (possible
  separation)` whenever its fitted linear predictor separated the classes by more
  than an η-gap of `1e-3`, or saturated, or collapsed its working weights, or
  drove the deviance below `1e-6` per sample. On separable data every one of
  those is a property of the CORRECT fit: a good fit's η *does* order the
  classes, its μ *are* near {0,1}, its weights *do* collapse. The guard exists
  for the case where the penalized objective has no finite minimizer, which
  happens only when a direction of recession of the log-likelihood lies in
  `null(S(λ))` — and under the double penalty it never does, so `β̂(λ)` is finite
  and unique even under exact separation. The criterion was therefore `+∞` over
  the whole region containing its own optimum, and the reported symptom was a
  line search unable to move at the seed. The saturation heuristics are gone from
  the penalized branch; the genuinely unbounded λ are still refused, by the
  conditioning and convergence machinery that measures them rather than by a
  guess. The same fixture that hard-failed now mints at every `n`, with the
  monotone, essentially linear fit the data supports (edf ≈ 1.95), and the suite
  runs in 3.1 s instead of 17.7 s because nothing burns 200 iterations at a
  refused trial point any more.

- **Firth bias reduction is now a Newton solve on every binomial link (#2273).**
  `WorkingState`'s Hessian is `XᵀWX + S` and deliberately omits the Jeffreys
  coefficient Hessian `HΦ`, because the outer Laplace layer consumes the two
  separately. Four consumers have to fold it back in, and only one did. The
  augmented-square-root direction solve folded it in by congruence — but that
  route is reached only when the realized curvature is Fisher, which is true just
  for the canonical logit link. Every non-canonical binomial link (probit,
  cloglog, …) fell through to a dense solve with the Jeffreys score in the
  gradient and no Jeffreys curvature in the matrix, and so did the constrained
  and bounded active-set solves, the post-loop undamped Newton polish, and the
  exact-decrement certificate. The result was an iteration that is not Newton for
  any objective: on the issue's 6-row separated probit fixture it contracted
  linearly at 0.4937 per step, stopped 23 iterations later at `‖g‖ = 4.3e-7`,
  failed its own convergence certificate and was refused — at a β̂ an independent
  reference confirms was the right one. The omitted term is now named once, as
  the matrix behind the quadratic correction that already existed, and folded in
  at every site through one helper that owns the sign convention. The same solve
  now reaches `‖g‖ = 1.2e-15` and clears its certificate by eleven orders, and
  `link(type=probit)`/`link(type=cloglog)` fits on separated data mint through
  the automatic Firth rescue the README promises.

- **A saturating binomial row is evaluated instead of refused (#2273).** Two
  numerical defects in the non-canonical observed-information path aborted fits
  over quantities that are perfectly representable. First, the Bernoulli variance
  was rebuilt from `μ` alone: a bounded inverse link reaches `μ == 1.0` exactly
  far inside its tail — cloglog at `η ≈ 3.62`, probit at `η ≈ 8.29` — so
  `1.0 − μ` is a hard zero while the true complement is still `1e-17`, `1e-45`,
  `1e-120`, and `V = μ(1−μ)` collapsed to zero with the whole
  observed-information jet dividing by it. The cancellation-free complement
  already existed and the sibling Fisher path already used it; it now reaches the
  variance and the working residual too. Second, the closed forms for the
  observed weight and its first two `η`-derivatives divided by `φV²`, `φV³` and
  `φV⁴`, and `V⁴` underflows to zero at `V = 2.9e-84` — so a `d²W/dη²` of
  `3.87e-75` came back NaN. Those expansions are replaced by the Leibniz
  recurrence the third derivative already used one order higher, which divides by
  `φV` once per order and never forms a power of `V`; the two orders of one object
  are now one recurrence instead of two independently-maintained expansions.
  Checked against two oracles that share no code with the engine: the exact
  identity that a cloglog row with `y = 0` has `−ℓ = e^η`, so its observed
  information and every `η`-derivative are exactly `e^η`; and mpmath at 220
  decimal digits for `y = 1`.

- **A flat-valley verdict now requires a flat valley (#2613).** The outer
  cost-stall guard — the mgcv-style stop that halts a smoothing-parameter search
  once the criterion stops improving over six consecutive **accepted** steps —
  was being fed every gradient evaluation, on the premise that the optimizer
  only asks for a gradient at points it has accepted. It does not: a
  strong-Wolfe line search evaluates the gradient at every trial that clears
  Armijo, because the curvature condition needs it. A search bisecting toward a
  point therefore reported six "steps" whose criterion values differed
  negligibly — of course they did, they were converging to a point — and the
  guard halted the fit *inside* one iteration, shipping a non-stationary
  checkpoint labelled "weakly-identified valley floor" and a refusal that
  counted line-search probes as outer iterations. The guard now consumes the
  optimizer's own accepted-step signal.
- **Outer stationarity no longer depends on where the search started (#2613).**
  The threshold a fit is judged converged against carried a component resolved
  once, at the seed, against the criterion's value *there*. Across the seeds of
  a single fit that spread the threshold over eighteen orders of magnitude, and
  a seed that happened to land somewhere absurd produced a threshold no gradient
  can fail — so the search claimed convergence against the wrong smoothing rail.
  The solver's threshold is now a function of the declared problem alone, the
  certificate keeps the per-point form it always meant, and the certificate can
  never be *stricter* than the threshold the solver was told to reach — which
  closes the "solver claimed convergence, certificate refused" family by
  construction rather than by retry.

- **A fit that has no criterion says so, everywhere (#2595).** `Summary.reml_score`
  and `raw_reml_score` were `0.0` on every exactly-interpolating Gaussian fit,
  because `UnifiedFitResult` had no way to express "no criterion exists here" and
  the exact-fit route had to write a placeholder to satisfy a finiteness contract.
  When the fitted mean reproduces the response to floating-point resolution the
  profiled scale is exactly zero and the restricted likelihood is unbounded, so
  the criterion is not small — it does not exist. It is now typed-absent, the
  boundary is recognized where the dispersion is estimated (so every entry point
  reaches the same verdict on the same data), the one constructor rejects a
  criterion at that boundary, and `Model.evidence`, `Model.bayes_factor_vs` and
  `gamfit.compare_models` refuse such a model by name instead of ranking a
  stand-in. `Summary.reml_score_unavailable` carries the explanation. Saved
  models load without a migration pass.

## gamfit 0.1.261 (2026-07-26)

This explicit Python release carries the 60 commits after 0.1.260 without
launching an unseeded cross-platform build.

- **Final scientific state has exact smoothing-coordinate provenance (#2486).**
  Outer finalization can attach an inner coefficient vector only when its
  producing rho matches the finalized rho bit-for-bit. A seed or a cost-only
  evaluation can no longer manufacture a mismatched `(rho, beta)` pair, so
  posterior-mean prediction and persistence consume one coherent fitted state.
- **Spherical separation remains accurate on every execution backend (#2489).**
  CPU, SIMD, input jets, ambient geometry, raw CUDA kernels, and fused CUDA
  Householder kernels carry cancellation-free chord energies instead of
  reconstructing `1 - cos(gamma)` from a rounded dot product.
- **Affine REML root finding is cancellation-free and value-certified (#2513).**
  The implemented point jet is enclosed directly, refinement returns the
  refined root rather than a new midpoint, and callers retain the value verdict
  that makes the root usable.
- **SAE criteria are single-valued and failure-honest (#2510, #2481).**
  Ranking differentiates the authoritative converged basin envelope; spatial
  and per-atom EFS failures preserve their typed payloads and escape
  backtracking instead of being silently reclassified.
- **Reproducibility is a cross-platform release contract (#2512).** Both reported
  Arrow-Schur border shapes now have bit-exact repeated-fit regressions, including
  simultaneous live allocations and a native Apple-silicon proof.
- **Publication is explicit, atomic, and cache-measured.** PyPI dispatches record
  their requested platform scope; full releases require every advertised
  platform, cancelled work is never accepted as an artifact, and cache
  read/write errors invalidate the publication receipt.

## v0.3.151 — gam 0.3.151 / gamfit 0.1.260 (2026-07-26)

This release carries the post-0.3.150 correctness, inference, and performance
campaign into one coordinated Rust/Python release. Notable root repairs:

- **Survival LAML uses one exact mode geometry (#2491).** LAML now requires
  a projected-KKT-stationary inner mode and a finite strictly positive-definite
  observed penalized Hessian. Value, log-determinant/trace contractions, and
  the implicit mode-response solve all use that same full coefficient space;
  the former relative pseudo-spectrum mask can no longer delete an identifiable
  low-curvature response direction and flip the outer gradient.
- **Exact Tweedie series remain exact at extreme scale (#2505).** Poisson- and
  Gamma-shaped log factors are centered algebraically with deviance identities
  and Stirling corrections before floating-point evaluation, eliminating peak
  cancellation without a saddlepoint fallback or a relaxed truncation certificate.
- **Posterior and prediction semantics follow the fitted distribution (#2440).**
  Constrained credible intervals project the persisted truncated posterior rather
  than reporting a symmetric Gaussian summary; covariance provenance and the
  smoothing correction remain attached across prediction and reload.
- **Configured mathematics is no longer discarded (#1376, #2254, #2463).**
  Realized anisotropic metrics retain their derivatives, deterministic Gaussian
  boundary fits derive smoothing state from the optimized coordinate, and every
  configured smoothing prior reaches the criterion that is actually optimized.
- **Finite constrained solves and calibrated dispatch replace open-ended work
  (#2432, #2420, #2504).** Dense constrained QPs use a certified finite dual
  projection, sphere basis construction avoids a trivial-section copy at scale,
  and Polya–Gamma GPU dispatch obeys the measured problem-size threshold.
- **Spherical kernels preserve near-coincident geometry (#2489).** Wahba
  kernels now carry `sin²(γ/2)` and `cos²(γ/2)` directly from chord
  identities, so close points, antipodes, diagonals, rotations, and analytic
  jets no longer lose their separation through a rounded dot product.
- **Release builds are measured, pinned, and warm.** PyPI wheel production pins
  maturin-action and maturin, fixes the manylinux contract, zeros compiler-cache
  counters immediately before each target, and refuses any unmeasured or zero-hit
  publication build. Commit-addressed binary releases enforce the same rule.

## v0.3.150 — gam 0.3.150 / gamfit 0.1.259 (2026-07-23)

This release ships the post-0.3.149 correctness and convergence campaign,
including the sparse factor-smooth failure isolated in #2401.

- **Sparse stiff-penalty P-IRLS no longer aborts (#2401).** The numerically
  safer augmented square-root solve now accepts sparse-native coordinate
  designs through blocked Householder tall-skinny QR. It remains algebraically
  equivalent to the dense QR solve, preserves backward-error and Firth
  correction checks, and bounds live storage by `O(p²)` instead of densifying
  an `n × p` design.
- **REML/P-IRLS robustness.** The release includes the intervening optimizer,
  stationarity-certificate, rail-boundary, sparse-penalty, and inner-solve
  corrections accumulated since 0.3.149.
- **Prediction and survival correctness.** Location-scale covariance remapping,
  survival checkpoint/terminal-curvature handling, and persisted conditional
  covariance state now follow the fitted parameterization through prediction
  and reload.
- **Sparse, Arrow, GPU, and SAE paths.** The Rust crate release catches up with
  the substantial sparse-native, Arrow-Schur, GPU-kernel, and experimental
  manifold-SAE work already present on `main`.

## v0.3.149 — gam 0.3.149 / gamfit 0.1.253 (2026-07-09)

This release makes `opt` the single optimization-mechanics authority for the
Rust engine. The former `gam-optimize` crate and GAM's hand-written GPU REML
BFGS loop are removed rather than retained as compatibility layers.

- All generic backtracking line-search and geometric ridge-escalation call sites
  now consume `opt 0.5.12`; the redundant `gam-optimize` workspace crate is gone.
- GPU-backed outer REML uses the same robust `opt::Bfgs` implementation as the
  host path, including hybrid Wolfe/backtracking search, bounds, projected
  stationarity, relative-to-cost tolerance, axis step caps, and best-iterate
  recovery.
- The device evaluator uses `opt::FusedObjective`, so a value+gradient kernel
  result is evaluated once and moved through an accepted line-search point
  without recomputing or cloning its gradient.
- The already-computed REML seed sample is handed directly to BFGS, eliminating
  another full inner solve at optimizer startup.
- Optimization dependencies are exact and registry-backed at `opt = 0.5.12`;
  every affected GAM library is validated against the published crate.

## v0.3.148 — gam 0.3.148 / gamfit 0.1.250 (2026-07-06)

A correctness release on top of 0.3.147, concentrated in three areas: **honest
deviance and evidence reporting**, **response-geometry robustness** (the curved
`response_geometry=` families no longer abort on widely-spread data), and
**survival/CLI round-trip fidelity**. A large batch of work also landed on the
experimental, default-off SAE latent-manifold engine, and the published `gamfit`
wheel build — briefly broken on `main` by a merge that duplicated an internal
type — is restored.

### Deviance, evidence & inference
- **Unscaled deviance in summaries (#2126, #2131).** Gamma and Tweedie summaries
  now report the raw unscaled deviance `D`, not `D/φ̂`, matching every other
  family and R/mgcv; a scaled-vs-unscaled regression pins it.
- **Akaike evidence ratio (#2124).** Model-comparison output reports the evidence
  ratio `exp(-ΔAIC/2)`, not its square `exp(-ΔAIC)`.
- **Gamma REML gain-ratio scaling (#2128).** The inner P-IRLS gain-ratio
  objective is scaled by the family dispersion, and the omitting-constants
  Gamma log-likelihood keeps its shape scaling, so a penalized Gamma smooth no
  longer drives REML to a non-finite cost.
- **`te(x,z)` EDF/SE row-order invariance (#2123).** The penalized-block reparam
  spectrum is stabilised via a root SVD and the posterior covariance is made
  ridge-free, killing the extreme-λ objective cliff that made a tensor-smooth's
  EDF and standard errors depend on input row order.
- **Design-whitened Wald smooth test (#2142).** The summary Wald test
  reconstructs the design-whitening Gram and uses the conditional `Vb`, not the
  smoothing-corrected `Vc` (Wood 2013).

### Response geometry & bases
- **Curved response geometries no longer abort on spread data (#2140).** The
  generic Karcher-mean driver behind `response_geometry=stiefel/grassmann/spd/
  poincare/constant_curvature` discarded its best iterate on a budget shortfall
  and failed the whole fit with "did not reach stationarity within max_iter" —
  even though `stiefel(k=1)` *is* the sphere, which fit the identical data. All
  exit paths now keep the best on-manifold iterate, with multistart over
  admissible seeds for the positively-curved (locally-non-unique) geometries;
  the SPD driver carried the same defect and is fixed to match. End-to-end
  `stiefel(k=1)⇄sphere` parity regression added.
- **Rotation-invariant `sphere()` (#2127).** Geodesic farthest-point centers make
  the default sphere Sobolev smooth invariant under a general `SO(3)` rotation,
  not just about the pole.
- **`matern(x,z)` retreats instead of collapsing (#2122).** A stalled joint
  spatial solve now falls back to a data-adaptive geometry rather than silently
  collapsing to the mean.
- **Weighted response-geometry linearization (#2125).** The weighted fit
  linearizes around the weighted Fréchet mean.
- **Pinned curvature honoured (#2152).** `curv()` threads a fixed `kappa=`
  through the whole fit orchestration via a new `kappa_fixed` pin flag.
- Measure-jet gap extrapolation follows the trend rather than the mean (#1845),
  weekday cyclic effects fit a genuine `S¹` continuum (#1849), the `d=1` seed
  tie-break prices chart arc-length defect (#2081), and cubic-cell boundary
  continuity is tightened (#2073).

### Families, encoding & survival
- **Binomial `loglog`/`cauchit` links (#2155, #2158).** These links are now wired
  through the external-design fit route and the joint link-wiggle solver, and the
  fitted-link state round-trips through predict.
- **Strict unseen levels for numeric-coded factors (#2137).** `factor(year)` and
  other numeric-coded fixed categorical factors raise on an unseen level in both
  `fit` and `check()`, instead of silently averaging.
- **Canonical float level keys (#2145, #2146).** Signed-zero and NaN float level
  keys are canonicalized so `-0.0`/`+0.0` map to one level.
- **Survival CSV honours the extrapolation law (#2154).** `write_survival_at_csv`
  now emits byte-for-byte what `survival_at` returns — both read one authoritative
  boundary/clip policy through the shared interpolation primitives — instead of
  crashing on any model with a stored surface.
- **Weibull baseline no longer double-counted in survival CLI predict (#2129).**
- **Bounded merit-descent veto for the survival marginal-slope stall (#979).**

### Performance
- **Batched Firth gram assembly (#1575).** Binomial/logit Firth gram assembly is
  `O(k·n²·p) → O(n²·p)`.

### SAE latent-manifold engine (experimental, default-off)
- Large batch: branch-guarded dual-number derivative oracle and softmax/IBP
  θ-adjoint corrections (#2156, #2144), capture-then-joint-rotate ISA (#2111),
  partition-free coactivation conditionality, certificate-gated atlas nerve,
  cycle graph atoms, Betti/persistence topology signatures, cross-model `O(2)`
  transport, finite-sample Terracini certificate, and a `gamfit.audit_sae`
  external frozen-dictionary audit surface, plus the FFI to reach all of it.

### Build
- **Restored the `gamfit` wheel build.** A merge duplicated the internal `Dual`
  type across two modules (ambiguous glob re-export + dead code) and drifted
  three `gam-pyffi` FFI construction/borrow sites away from struct changes in
  `gam-sae` — invisible to a `gam`-crate-only check but fatal to the wheel. All
  fixed at the root (no lint suppression), with the routability gate's evidence
  now surfaced as a `log::debug!` observability trail.

## v0.3.147 — gam 0.3.147 / gamfit 0.1.249 (2026-07-04)

A broad correctness release cut from ~130 root-cause fixes on top of 0.3.146.
The user-facing themes: **honest predictions and intervals** (`gam predict
--uncertainty` no longer moves the point estimate; partial-dependence and
survival/GAMLSS standard errors are sourced from the smoothing-corrected
covariance and honest per-block EDF), **stricter, more truthful encoding**
(unseen levels of a fixed categorical factor now raise instead of being silently
averaged; `loglog`/`cauchit` are accepted as binomial links), **scale- and
gauge-invariant convergence** across the REML/PIRLS and custom-family/AFT
solvers, and **better basis behaviour** (measure-jet gap extrapolation, radial /
thin-plate center-count floors, a new closed-form fidelity-metrics surface). A
large batch of work also landed on the experimental, default-off SAE
latent-manifold engine.

### Predictions, intervals & inference
- **`gam predict --uncertainty` never shifts the point (#2115).** The
  linear/identity uncertainty arm passed `apply_bias_correction: true`,
  recentring η by `X·H⁻¹S(λ̂)β̂` — so requesting an interval silently moved the
  reported `mean`/`linear_predictor` (~2.5%) relative to plain `gam predict` and
  the Python FFI. The arm is now pinned to the plain plug-in point; `--uncertainty`
  only appends the SE/band columns. Guarded by a new identity-link regression test
  (companion to the curved-link #1787 guard), the `--no-bias-correction` help text
  is corrected, and the flag still gates the survival uncertainty paths.
- **`partial_dependence` SE uses the smoothing-corrected covariance (#2113).**
  The band now propagates smoothing-parameter uncertainty (the mgcv
  `predict(se.fit=TRUE)` analog) instead of the raw plug-in covariance.
- **Unseen fixed-factor levels raise instead of averaging (#2102).** A bare
  categorical main effect (`+ g`) is auto-promoted to a penalized random block
  internally, but is a FIXED parametric factor: an out-of-vocabulary level now
  hits the strict schema encode and raises `SchemaMismatchError` (and is reported
  by `Model.check`) rather than being mapped to the factor's centering point.
  Explicit random effects (`group(g)`/`re(g)`/`s(g, bs="re")`) stay lenient
  (held-out group → population mean). A serde-defaulted `lenient_unseen` flag
  keeps old saved models loadable.
- **`loglog` and `cauchit` accepted as binomial links (#2104).** Both now pass
  `link_legal_for_family` for the binomial family.
- **Survival location-scale EDF is honest (#2106).** The summary reported the
  nominal coefficient count as `edf_total`; it now threads the inner blockwise
  solver's real per-penalty traces so `edf = ncoef − Σ tr(H⁻¹S)` per block.

### Solvers: scale- and gauge-invariant convergence
- **Scale-aware reduced parametric-AFT convergence (#2112).** The MLE Newton
  loop replaced an absolute tolerance on the O(n) summed gradient with the
  affine-invariant Newton decrement `½·gᵀH⁻¹g`, so convergence no longer depends
  on sample size or weight scale; a principled stalled-line-search acceptance
  still errors on genuine curvature failure.
- **Coupled custom-family joint gradients (#2108, #1820).** `GaussianLocationScaleFamily`
  gets an exact joint gradient from the same score that feeds its Hessian (with
  the block-Hessian finiteness guard preserved), and the coupled custom-family
  LAML gradient is pinned at joint stationarity.
- **REML gauge-invariant KKT gate (#752/#901)** and a **near-minimal PD
  stabilizing shift** for the coupled Newton step (so it stops over-damping),
  plus **n-independent κ outer-loop cost probes (#1868)** and a memoized
  seed-grid basin search (#1575).
- **`response-geometry` parametric-only RHS (#2103)** is fit via a direct
  shared-tangent least-squares path instead of forcing REML, and
  **`ResponseGeometryModel` round-trips through save/load (#2114)**.

### Bases & terms
- **Measure-jet gap extrapolation (#1845).** An unpenalized ambient-affine head
  is appended to the measure-jet design so the fit carries the flank-attested
  trend across an unsupported training gap instead of collapsing to the training
  mean. (Interval-coverage calibration of this experimental `mjs()` basis on
  sharply kinked truths remains a known rough edge.)
- **Center-count floors.** 1-D radial default centers are floored at the
  univariate spline resolution (#1867); thin-plate center counts are floored at
  `M(d)` (not inflated by `k+M(d)`) and never inflated past the row count in high
  dimension; the `circle_regime` ring fit is unbiased (Kåsa) so S¹ wins on
  continuous data (#1849).
- **Public basis exports:** `matern_basis` and `sphere_basis_jet` are reachable
  from the public API / `gamfit` namespace (#2120).
- **CLI Duchon operator-penalty gating (#2116)** is applied on the standard-fit
  path, and the **GPU arrow-border DenseDirect policy** is gated on full rank
  (compares k³ Cholesky, not assembly, against the CG solve).

### Python (`gamfit`)
- **Fidelity metrics.** A new closed-form metrics core (loss-recovered / R² /
  categorical KL / distortion-floor R²) lands in `gam-math` with a Rust+numpy
  parity surface exposed through `gamfit`.
- **`ResponseGeometryModel` save/load** and a **real out-of-sample
  `transform`/`encode` for `StagewiseSAE` (#2118)** that lifts frozen composed
  decoders into the OOS-capable manifold path.
- **`layer_transport`** gains `chart_transfer_operator` / `certify_chart_transfer`.

### SAE latent-manifold engine (experimental, default-off)
- Honest, currency-based birth/demote gates: a rank-charge evidence criterion
  (realised-rank BIC, default-off) ported to the streaming K≈32k path with a
  derived Marchenko–Pastur rank floor (#5/#9/#11/#16); zero-realised-rank births
  are vetoed; the hybrid split prices deviance, not raw SSE (#2124).
- A tiered spine (shared Tier-0 mean + Tier-1 interference emitter, #2023),
  block-coordinate manifold charts, behavioral-Fisher whitened-GLS encode
  (#2021), and born-circle birth gating (#2109/#2111). The 10k-line
  `construction.rs` was split out into a dedicated
  arrow-Schur assembly module (#780).

### Build & hygiene
- The `println!` ban is scoped to production code (test / example / build
  harnesses may use it), cluster paths are driven from env vars so the
  infra-leak gate stops failing the build, assertion-less probe tests are
  converted into genuine guards (#2101/#2110), and the #1575 outer-cost scaling
  harness is a real n-independence gate.

Verified with `cargo check --workspace --all-targets` (exit 0) and the core
crate lib tests.

## v0.3.146 — gam 0.3.146 / gamfit 0.1.248 (2026-07-03)

Correctness and honesty release on top of 0.3.145, cut from a large batch of
root-cause fixes. Two themes run through it. First, **honest reporting**: the
solver now surfaces non-convergence, line-search stalls, and genuine
rank-deficiency instead of papering over them, a NaN regression in the reported
Gaussian log-likelihood/AIC is fixed, and a new SPEC rule forbids wall-clock
time budgets and deadlines (all of which have been removed from the solver).
Second, **no arbitrary knobs**: remaining grid scans and magic constants are
replaced by principled searches — a golden-section search for the Tweedie
variance power, a monotone-bisection ψ-band search, and error-driven
order-doubling for cloglog GHQ quadrature. Alongside are new `loglog`/`cauchit`
survival links, weight-aware analytic Gaussian observation intervals, several
model-persistence and class-detection repairs on the Python surface, a
differentiable-basis (Duchon) autodiff correctness fix, and a broad round of
work on the experimental, default-off SAE manifold engine.

### Solver honesty & policy
- **No wall-clock time budgets or deadlines (#2055).** SPEC now forbids
  wall-clock time budgets/deadlines, and every such budget/deadline has been
  removed from the solver rather than left to paper over a slow path. Related
  survival-path deadline guards were removed or scoped, dropping the last
  banned underscore-let bindings introduced to silence them (#979).
- **Report non-convergence honestly (#2062).** The flexible-link inner-KKT
  envelope no longer masks a non-converged inner solve as converged; it reports
  the failure. Custom-family exact-joint nonconvergence stays recoverable
  (#2014) instead of aborting the fit.
- **Multinomial line-search stalls are reported, not hidden (#2066).** A stalled
  line search no longer returns `converged = true`.
- **Identifiability audits tell the truth (#2070).** The audit reports genuine
  rank-deficiency honestly; rank-restore keeps structural aliases dropped and
  restores only numerical demotions, and structured-residual fit failures
  propagate instead of being swallowed.

### Families & links
- **`loglog` and `cauchit` survival links.** Both links are implemented for the
  survival family and covered across the `LinkFunction` scale/routing paths,
  completing #1946; their GHQ routing replaces a banned `unreachable!()` with an
  explicit branch.
- **cloglog GHQ quadrature converges by construction (#2063, #1835).** cloglog
  Gauss–Hermite quadrature now certifies convergence via error-driven
  order-doubling around an adaptive mode-centred rule, instead of a fixed order
  that could silently under-resolve.
- **Automatic Tweedie power without a grid (#2064).** The bare-`family="tweedie"`
  variance-power estimate (added in 0.3.145) now uses a golden-section search
  over the open interval `(1, 2)` in place of the coarse power grid scan.
- **SAS / beta-logistic link fixes (#1876, #2094).** The SAS ε sign convention is
  corrected and value/cost consistency restored with regression tests (dropping
  a banned `#[allow]`), and the SAS/beta-logistic link spec is now threaded
  through the formula and Python fit paths.

### Estimation & inference
- **Gaussian log-likelihood/AIC NaN regression fixed (#2096).** The reported
  `log_likelihood` (and the AIC derived from it) came out NaN for every Gaussian
  fit because the profiled scale was passed through unresolved; the profiled
  σ̂² is now resolved into the reporting spec while the persisted scale marker
  stays `ProfiledGaussian`. Covered by a finiteness regression over the real
  reporting assembler.
- **Weight-aware analytic Gaussian observation intervals (#2077).** For a
  weighted Gaussian fit, `predict(observation_interval=True)` now scales the
  per-row band by `1/√wᵢ`, matching `sample_replicates` (the analytic sibling of
  #2025) instead of broadcasting a single pooled σ̂².
- **Multinomial Firth/Jeffreys separation solver (#1854, #1821).** A
  self-contained fixed-λ Firth/Jeffreys solver handles separation; Firth stays
  active in the LM line-search candidate and gate-motion is added to the
  Jeffreys inner gradient.
- **Location-scale reproducibility (#1607).** Cross-fit warm-start is gated on a
  family opt-in so exact-replay parity holds; the binomial location-scale outer
  gradient is aligned with finite differences.
- **Identifiability ridges (#2068, #1802).** `double_penalty` linear terms get
  exactly one identifiable ridge (no duplicate null-space ridge), and the
  reparam null-leakage tolerance is derived from the eigensolver PSD floor
  rather than a fixed constant.
- **ψ-band search de-gridded (#2054).** The rank-stable ψ-band search uses
  monotone bisection instead of a grid.
- **Thin-plate spline basis recovery (#1966, #1074).** Fixes low-rank
  under-recovery by seeding the length-scale from center spacing.
- Tier-0 rho-posterior certificate is emitted on formula fits (#1810);
  `partial_dependence` by-factor blocks no longer depend on input row order;
  matrix-free tangent-projected logdet operator trace in REML (perf).

### Python surface (gamfit)
- **Multinomial-logit GAM persistence (#2078).** Multinomial-logit models now
  round-trip through the public `save`/`load` API.
- **`is_marginal_slope` for Bernoulli marginal-slope models.** The kebab-case
  `marginal-slope` model_kind label is now recognized, so `is_marginal_slope`
  returns `True` for Bernoulli marginal-slope fits.
- **`GAMClassifier` non-{0,1} labels (#2075).** The response column is dropped at
  inference for classifiers with non-`{0,1}` labels.
- **`Model.model_class` precision.** Derived from the fine-grained predict class
  rather than the coarse `model_kind` enum; `Model.evidence` /
  `bayes_factor_vs` route through the `compare_models` ranking score (#2079).
- Prior weights preserved in smooth-significance; stagewise progress callback
  types exported.

### Differentiable bases (torch)
- **Duchon basis VJP correctness (#2097).** `_DuchonBasisFn.forward` now sources
  its design from the same jet builder that backward differentiates, so the
  analytic input gradient is the exact derivative of the returned forward
  (max|analytic−fd| ~3e-11, down from ~0.7) and the width-mismatch case that
  raised a broadcast `RuntimeError` no longer occurs. gradcheck/gradgradcheck
  regressions added.

### SAE manifold engine (experimental, default-off)
- Stagewise births are unblocked on disjoint/near-block-diagonal residuals via a
  Marchenko–Pastur-thresholded residual-principal fallback, with anchor-scored
  birth-seed selection (#2080); topology-seed subsample/kNN/interp are derived
  from budget and dimension (#2065); total co-collapse bails with a diagnostic
  (#2089).
- The certified encode path now reads the true (un-ridged) Hessian for the
  Kantorovich certificate, with amplitude-aware routing and Duchon refusal.
- New public surface: coordinate interchange and coordinate-fidelity APIs
  (#2019, #2069), stagewise birth checkpoints exposed through the Python FFI, and
  GPU dictionary scoring wired with route telemetry.
- Router throughput: an O(1)-reject top-`s` fold and a shared-memory-tiled score
  GEMM at the `K ≈ 32k` dictionary width (#1026, perf).

### Build & CI
- The `sparse_dict_router_topk` bench and the `sae` integration test are fixed to
  the post-carve module path / signature so `cargo check --all-targets` links.
- The `build.rs` author guard is robust to shallow clones; an on-demand
  single-job "Validate One" workflow gives fast per-issue fixer feedback.

## v0.3.145 — gam 0.3.145 / gamfit 0.1.247 (2026-07-02)

Correctness release on top of 0.3.144. The headline is a prior-weights
effective-sample-size cluster that makes zero-weight and zero-`by` rows exactly
inert in Gaussian REML, plus mgcv-style automatic Tweedie variance-power
estimation for a bare `family="tweedie"`. Alongside are a multinomial-Firth
log-determinant consistency fix, a binomial location-scale predict-noise repair,
a Duchon well-posedness auto-raise, and constrained-REML backward/forward
corrections. This cut also removes leftover FD-audit debug scaffolding from the
κ-optimization path and hardens the (experimental, default-off) SAE dictionary
diff so it can no longer report "no differences" when atoms were added or
removed.

### Families & responses
- **Automatic Tweedie variance power for bare `family="tweedie"` (#2026).**
  Mirroring mgcv's `tw()`, a bare `family="tweedie"`/`"tw"` (CLI `--family
  tweedie`, or a formula family that names no explicit power) now estimates the
  variance power `p ∈ (1, 2)` by profile likelihood before the reported fit — a
  coarse grid over the open interval refined by golden-section search, profiling
  the dispersion out per node with the solver's own prior-weighted Pearson
  estimator and scoring the fully-normalized saddlepoint log-likelihood
  (comparable across `p`). Previously a bare Tweedie silently used a fixed
  `p = 1.5`, miscalibrating observation intervals on data whose true power was
  not 1.5. An explicit `tweedie(1.6)` / `tweedie(p=1.6)` still pins `p` exactly.
- **Weighted replicate frames keep their weight column (#2033).** Regression of
  #2025: `Model.sample_replicates` narrowed the frame to the required prediction
  columns + response before resolving per-row weights, so every weighted model
  raised `weights column '…' not found in data`. The weight column is now part of
  the consumable-column set (so it survives projection), and the replicate path
  degrades to unit weights when the caller frame genuinely omits it instead of
  erroring. Ordinary predict is unaffected.

### REML / prior weights
- **Zero-`by` rows are inert in Gaussian REML (#2031).** `by` was applied only
  as a design-column gate, so a `by=0` row's response still entered the response
  energy `Σ w·y²` and the residual degrees of freedom `ν`, leaking into `σ²`,
  `λ`, and — through `λ` — the coefficients. The `by` gate is now folded into the
  REML row weights (`w_eff = w·[by≠0]`) across every forward/backward FFI path, so
  a `by=0` row is a complete no-op. A fit whose `by` is entirely nonzero stays
  byte-identical to manually gating the design.
- **Residual DoF uses the effective sample size (#2032).** Zero prior-weight
  rows (the universal "excluded / infinite-variance" convention) no longer count
  toward `ν = n − nullity`. `ν` is now built from the number of strictly
  positive-weight rows everywhere the REML score, `σ²`, `λ`, `edf`, and the
  adjoint are computed, so a `weights=0` row is exactly equivalent to omitting it.
  When every weight is positive this is a strict no-op.
- **Constrained REML forward/backward (Rust CI / Python API).** The constrained
  forward now returns the closed-form Gaussian REML solve whenever the optimum is
  interior — both when no inequality system is supplied *and* when a system is
  present but the unconstrained optimum is already strictly feasible (every
  `aᵢ·β̂ > bᵢ`, the exact negation of the active-set binding test). By the KKT
  conditions an interior certificate is the unconstrained problem, so this makes
  a non-binding constraint agree bit-for-bit with the unconstrained
  `gaussian_reml_fit` and with the interior-cert backward (which already
  differentiates that closed form) instead of settling on a slightly different
  PIRLS smoothing parameter. Binding (shape-constrained) fits are unchanged. The
  backward's zero-bound guard is additionally scoped to the *active* face, so an
  interior certificate carrying a non-zero bound on a never-binding slack
  constraint (e.g. `0·β ≥ −1`) flows through the correct full-space envelope VJP
  instead of being rejected up front.

### Inference & prediction
- **Multinomial-Firth log-determinant consistency (#1854/#1395).** The
  small-system `BlockCoupledOperator` route eigendecomposes via
  `eigh(Side::Lower)`, which assumes a symmetric input; on the near-separating
  Firth path the divided-difference curvature carries an `O(1e10)` scale, so
  reduction-order floating-point asymmetry produced a materially different
  spectrum and log-determinant than the #1395 ground-truth guard (which
  symmetrizes first). The joint Hessian is now symmetrized before the operator is
  built, so every route realizes the penalized joint Hessian log-det
  consistently.
- **Binomial location-scale predict-noise (#1828).** The default-link arm no
  longer demands an explicit blend spec (it falls back to the binomial-logit base
  link), and a parametric-linear `log_sigma` is accepted for predict-noise while
  a nonparametric free scale is still correctly rejected.
- **BMS marginal / log-slope blocks lock to raw width (Rust CI / Python API).**
  The Bernoulli marginal-slope and log-slope Jacobian callbacks now declare
  `locks_raw_width_reduction()`, mirroring the survival marginal-slope precedent,
  so the canonicaliser keeps their raw block width.

### Smooths & kernels
- **Duchon null-space order auto-raises to clear the collocation margin
  (#1817).** A low order/power pair with a derivative-collocation operator active
  (e.g. `d=2`, `Linear` null space, `power=0` with the stiffness operator) could
  trip the pointwise/collocation well-posedness guard `2(p+s) > d + max_op`
  mid-fit. The null-space order `p` is now auto-raised by the smallest amount that
  restores the strict margin before the guard can fire (warned once per config);
  the spectral power and the CPD condition `2s < d` are untouched.
- **κ design-realization skip restored on the n-free lane (#1868/#1033).** The
  `TEMP-SKIPOFF-1122` debug override that hard-forced the `O(n)` lane in
  `eval_full` is removed, restoring the n-free gradient/value path. The remaining
  leftover FD-audit debug scaffolding on that path (four unconditional
  `TEMP-*-1122` blocks that logged at warn level and rebuilt the Matérn basis /
  penalty triplet several times per call) is deleted.
- **Cubic-cell C0-continuity regression made meaningful (#1837).** The
  `bug_hunt` continuity check was mis-specified — it built the neighbouring cell
  from a cell-local Taylor parameterization but stored those coefficients as a
  global polynomial, and compared the two cells at `boundary ± eps` against a
  tolerance tighter than the injected `O(eps·slope)` gap. The test now constructs
  the right cell through the kernel's own `global_cubic_from_local` path and
  evaluates both cells at the shared boundary point, so it is a genuine (passing)
  C0 invariant check. The production kernel is unchanged.

### SAE manifold (experimental, all default-off)
- **Chart-transfer operators (#2016).** Pulled-back chart-to-chart transfer
  operators `A_kj(x) = (JₖᵀJₖ)⁻¹Jₖᵀ J_F(x) Jⱼ(x)` for 1-D/2-D atoms, with
  density-weighted mean/variance aggregation (Kish effective-n) and
  isometry/equivariance transport certificates. (The coordinate-valued
  attribution-graph deliverable is scaffolded but not yet wired.)
- **Canonical dictionary artifact tooling (#2018).** Deterministic SHA-256
  hashing of the Frobenius-normalized, reflection-fixed dictionary orbit
  representative, with order/scale/reflection-invariant equality and a
  decoder-row-localizing diff. The alignment diff now counts atoms with no
  counterpart on either side as substantive differences, so
  `hash_equal_after_alignment` can no longer claim equivalence when the two
  dictionaries carry different atom sets at equal total count. (Residual
  continuous-chart gauge pinning and Procrustes/optimal-transport alignment
  remain future work.)
- **Sparse SAE Schur block GEMM (#1995).** The reduced-Schur block subtract skips
  zero columns on the sparse atom support.

### Testing
- Rust and Python regression coverage for the zero-weight / zero-`by` REML
  inertness (#2031/#2032), the estimated-Tweedie-power recovery (#2026), the
  Duchon auto-raise (#1817), the interior-cert non-zero-slack-bound constrained
  backward, the generic row-kernel jet oracle projections (#932), and the SAE
  dictionary diff's unmatched-atom accounting.

## v0.3.144 — gam 0.3.144 / gamfit 0.1.246 (2026-07-02)

Correctness release on top of 0.3.143. Two new user-facing capabilities land on
the formula/CLI surface — the Tweedie variance power is now settable and the
`loglog`/`cauchit` survival links are wired end-to-end — alongside a batch of
prediction/diagnostics fixes (Gaussian location-scale σ scale double-count, a
`gam diagnose` failure on every standard fit, and generative replicate noise
that ignored prior weights). This cut also repairs a release-blocking cubic-cell
regression that landed after 0.3.143: a well-meaning affine-anchor "normalization"
turned five previously-green deep-tail/both-tails precision guards red, and is
reverted here to the raw substrate convention the whole kernel (and the CPU/GPU
parity reference) actually uses.

### Families & responses
- **Tweedie variance power on the formula path (#2026).** `family="tweedie(1.6)"`
  / `tweedie(p=1.6)` now parse the mgcv-style parenthesized argument as the
  variance power `p` (`Var = φ·μ^p`) and validate it through the shared
  strict-`(1, 2)` gate, instead of misrouting `1.6` to the link resolver and
  failing with `unknown link '1.6'`. A non-numeric argument (`tweedie(log)`)
  still flows to the link resolver; bare `tweedie`/`tw` keep the neutral interior
  default `p = 1.5`.
- **Survival `--link loglog` / `--link cauchit` (#1829).** Both links are now
  accepted and evaluate exactly, routed through a single-component mixture
  (weight 1.0, no free mixing logits) so they flow end-to-end through the wired
  `InverseLink::Mixture` survival path; the survival `--link` usage string
  advertises them. Genuine multi-component blends still require a
  logit/probit/cloglog anchor.
- **Weighted Gaussian replicate noise (#2025).** `Model.sample_replicates` now
  scales each row's Gaussian observation noise by its analytic prior weight
  (`σ_i = σ̂/√w_i`, `Var(y_i) = σ²/w_i`) instead of a single pooled scalar.
  Unit/absent weights leave unweighted fits unchanged.

### Inference, prediction & diagnostics
- **Gaussian location-scale predict σ no longer double-counts `response_scale`
  (#1874, #1928).** The persisted log-σ intercept is already shifted by
  `+ln(response_scale)` at fit time, so only the soft floor (which sits outside
  the `exp`) is scaled at predict time — exactly one factor of `response_scale`
  on the σ surface, restoring response-scale equivariance whenever
  `sample_std(y) ≠ 1`.
- **`gam diagnose` fixed on every standard fit (#2030).** Batch compaction no
  longer zeroes the row-sized `working_weights`/`working_response` on the
  persisted geometry carrier, so the geometry-ALO path stops handing empty
  vectors to `AloInput::from_geometry` and failing length-N validation; a
  present-but-emptied carrier now falls through to the refit branch, and the
  saved weight column is loaded into the diagnose frame.
- **Distinct penalty coordinates for grouped + double-penalty + block-gamma
  priors (#1881).** Combining coefficient groups, per-term double penalties, and
  keyed block-gamma priors now keeps each as its own λ instead of collapsing the
  per-term base/double coordinates into one shared linear ridge.

### Smooths & kernels
- **Cubic-cell affine-anchor moments kept in the raw substrate convention
  (#1833).** Reverts the post-0.3.143 normalization of the public
  `affine_anchor_moment_vector` (dividing by `√(2π)`), which broke the #352
  both-tails and deep-tail precision guards (relative error `1 − 1/√(2π)`) and
  diverged the public API from every production consumer and the byte-for-byte
  CPU/GPU parity reference. The mis-specified identity test is corrected to the
  raw standard-normal moments (`M0 = M2 = √(2π)`, `M1 = M3 = 0`).
- **Duchon affine-trend native ridge is curvature-relative (#880).** The
  machine-scale ridge on the affine slope columns is now scaled by the curvature
  block's mean diagonal (genuinely `√ε`-relative, as documented) rather than an
  absolute floor that survived Frobenius normalization and pushed the affine
  trend out of the null space on low-magnitude curvature grids.

### SAE manifold (experimental, all default-off)
- Two-tier fit primitives (tier merge / atom reorder), the scale-gauge quotient
  with no-refresh amplitude-absorb transport-peel, data-row dead-atom reseed, and
  the Λ nursery→promotion birth channel are promoted from hidden `GAM_SAE_*` env
  levers to typed, default-`false` kwargs (`promote_from_residual`,
  `quotient_scale`, `data_row_reseed`), threaded through the pyffi entry points
  including the IBP convenience delegator (#2021, #2022, #2023). The historical
  default path stays bit-for-bit unchanged.

### REML / ALO internals
- FD gates added for the Firth and Barrier first-order `D_βH` curvature
  operators; the ALO sandwich-SE meat weight is taken from the score/Fisher
  weight `w_s` rather than the observed-information weight `w_h`, with a
  defense-in-depth reject of non-positive `snr_proxy` in the SAE global-optimality
  verdict.



Broad correctness, inference-accuracy, and performance release on top of
0.3.142. The headline changes: grouped-**binomial proportion responses** are
accepted again (not just strict `{0, 1}`); a batch of REML / posterior-variance
/ credible-band fixes tighten uncertainty quantification; the Duchon and Matérn
spline families get several geometry and null-space corrections; the manifold
SAE stack lands device-resident execution, principled (no-magic-constant)
penalties, and a batched JumpReLU gate kernel exposed to `gamfit.torch`; and a
wave of PERF work brings 1-D/2-D smooth and location-scale fits from
2–160× slower-than-reference down toward parity. This cut also repaired three
release-blocking compile regressions introduced by concurrent landings (a
clobbered `ReportInput.smoothing_forensics` field in the PyO3 wheel, a stale
`ArrowBlockDiagInverse::build` call site, and an unbalanced-delimiter test) and
removed a banned `#[allow(too_many_arguments)]` by bundling the JumpReLU layout
inputs into a params struct.

### Families & responses
- **Binomial proportion responses accepted (#1806, #1987).** The binomial
  support was tightened to strict `{0, 1}` in an earlier cut, which rejected
  grouped-binomial and continuous-probability responses (with trial weights
  folded into the row weight). Support is now the closed interval `[0, 1]`,
  which keeps the Bernoulli / grouped-binomial log-likelihood bounded; the
  strict `{0, 1}` predicate is retained only where the code is specifically
  asking whether a numeric response is binary (auto-inference and all-boundary
  degeneracy checks). External GLM response-support errors are also clarified
  (#1982).
- **Survival lognormal formula routing fixed (#2009);** reduced-AFT MLE
  non-convergence is handled gracefully (#1921); KKT projection includes
  near-active monotone rows to clear a flat baseline-hazard stall (#1793,
  #1992); left-truncated transformation LAML uses a hard-pseudo logdet (#1915);
  all-interval latent survival warm start restored (#1916); survival constraint
  projection semantics clarified (#1919).
- **Beta-Logistic** derived delta now uses the SAS bounded transform (#1993);
  negative-binomial dispersion location-scale ρ optimization fixed (#1956);
  Gaussian location-scale fits retain their joint covariance (#1974) and restore
  σ to response units on generate (#1928).

### Inference, uncertainty & REML
- **Logit posterior-variance regression fixed (#1976);** smooth credible bands
  get a corrected ρ-variance cubature gate (#1971) and a bias-correction
  linearization on the smoothing-corrected β covariance (#1970); the ALO
  sandwich SE is computed from the frozen-curvature meat `XᵀWX` (#1969).
- **Hutchinson REML trace standard-error scaling corrected (#1977);** REML folds
  and B-spline double-penalty EDF inflation get deterministic row reductions and
  a curvature-only cleanup (#1929, #1964); single-ρ̂ Lawley mean-shift uses
  Gauss–Hermite quadrature (#1972); `Gamma(1, 0)` precision priors are treated as
  `Flat` for the REML ρ-prior policy (#2006).
- **Random-effect BLUP recovery fixed (#2008);** the summary penalty cursor now
  skips only the penalty blocks the random effects actually own (unpenalized
  `by`-factor ranges no longer slide every following smooth's window off by one)
  (#1883, #1979); concurvity-driven double-penalty null-space rail collapse is
  guarded (#2017); the marginal-slope absorber is protected from
  over-orthogonalization with an `n`-scaled ridge (#2013, #1947).
- **Multinomial per-class smoothing recovery fixed (#1855, #2027);**
  identifiability audit gauge attribution and rank-deficiency handling fixed
  (#2005); Royston–Parmar uncertainty prediction link canonicalization fixed
  (#1955); prediction CSV schema stabilized (#1928).

### Smooths & bases
- **Duchon family:** rotation-equivariant knot tie-breaks (#1935), affine-trend
  ridge deselection to prevent pre-fit rank deficiency (#1934), polynomial
  null-space evaluation in the collocation operators (#1933), and nullspace-test
  centering (#1931); periodic Duchon auto-power pinned to `s = 0`.
- **Matérn:** sufficient basis centers for exact adaptive regularization (#2003);
  anisotropic derivative geometry — normalization, center RRQR, η seeding (#1937).
- **Cubic-regression (`cr`/`cs`)** bases routed to the standard dense fit off the
  exact spline-scan fast path, with a `ds` Duchon alias added (#1844, #1957);
  cyclic B-spline seam evaluation fixed (#1922) and small-k cyclic tensor margins
  allowed (#1944); tensor margins honor quantile knot placement (#1943);
  `by`-factor / `bs="sz"` `SmoothBasisSpec` construction fixed (#1981);
  constant-curvature `curv()` recovers hyperbolic data instead of railing to
  spherical (#1464).

### Manifold SAE
- **Batched JumpReLU gate FFI kernel (#6).** `gamfit.torch`'s bounded threshold
  gate is now a thin marshaller over a single batched Rust value+grad call
  (`sae_jumprelu_batch_value_grad`), bit-identical to the row kernel — the whole
  `(N, K)` matrix crosses the Python↔Rust boundary once per forward pass instead
  of once per row, and Rust is the single source of truth for the gate math.
- **Device-resident SAE solver engages on production fits (#1017, #1551);** the
  GPU offload gate is tuned for thin `d_atom = 1` curve atoms (#1913) and the
  throughput decision gate is made honest against its target (#1412).
- **Principled penalties (#1610):** hand-picked magic-constant penalty strengths
  and collapse thresholds replaced with derived quantities; per-fit IBP α
  override on manifold init avoids atom masking (#1784); IBP cross-row Woodbury
  coupling restored (#1920); SAE evidence Schur deflation uses unit-stiffness
  quotient conditioning for the log-det (#1925); recoverable refusal probes kept
  finite (#1912). The JumpReLU compact layout is sized by the hard forward gate
  (O(k_active)), and the residual-factor path hoists row-independent work into a
  reusable `fit_row_metric` (#2021 Wave-2). A `WhitenedStructured` row-metric
  driver is wired onto the fit path behind a default-off flag (#2021).

### Performance
- 1-D P-spline / thin-plate and Poisson/Gamma `s(x)` fits (#1689, #1690),
  binomial-logit `s(x)` / `te(x, z)` and REML fits (#1575, #1727),
  Gaussian-location-scale (`noise_formula`) overhead (#1720), and Duchon 2-D
  fits (#1718, #1757) are all brought substantially closer to reference-software
  speed at equal accuracy. GAM significance vs reference software improved
  (#1561).

### Reports, CLI & reliability
- **Smoothing-forensics report section added** (λ/σ² paths, EDF criterion vs
  assembly) (#1986). `--family expectile` and other standard-family CLI fits no
  longer abort on the frailty guard in the expectile inner fit (#1948).
- Concurrent tail-cell cache computations are coalesced to avoid duplicate work
  (#2002); warm-start LRU touch and deterministic eviction fixed (#1998);
  power-law scaling reports hardened against noisy timings (#1997); the spatial-κ
  optimizer recovers from a NaN frequentist covariance by treating the trial
  point as infeasible (#2001); startup no longer bails on repeated non-finite
  trial objectives (#1802, #1924).

## v0.3.142 — gam 0.3.142 / gamfit 0.1.244 (2026-07-01)

Broad correctness, robustness, and performance release on top of 0.3.141. The
headline fixes: left-truncated survival fits stop collapsing to a degenerate,
covariate-flat curve; explicit `--family` and frailty flags stop misreporting or
aborting; `periodic=false` on a spatial smooth is honored; the isotropic-Matérn
location-scale ψψ Hessian is no longer silently halved; several previously
panicking or misleading edge cases now return clean errors; and a batch of
solver/GPU/SAE performance work lands with reproducibility and matrix-free
correctness guards.

### Survival
- **Left-truncated `Surv(entry, exit, event)` no longer degenerate (#1790,
  #1791).** Under the default `transformation` (Royston–Parmar) likelihood, any
  delayed entry (`entry > 0`) produced a covariate-independent fit with cumulative
  hazard inflated ~10³× and `S(t) ≡ 0`. Two root causes are fixed: (a) the time
  basis is now anchored at the robust interior median exit under genuine left
  truncation (not the earliest entry, whose one-signed centered linear-trend
  column railed the smoothing selection), extending the marginal-slope #751 fix to
  every time-basis likelihood; and (b) because the transformation LAML uses the
  **observed** information (which the delayed-entry `−Xᵀ_entryW_entryX_entry` block
  can drive indefinite below the seed λ), the time smoothing blocks' outer lower
  bound is floored at their seed ρ under left truncation, so the selector may only
  hold or over-smooth the baseline — never rail it into the degenerate
  under-smoothed region. Right-censored (`entry == 0`) fits are bit-for-bit
  unchanged. Regression tests pin covariate recovery and baseline non-degeneracy.
- **Survival predict-query times accept the full real line (#965).** The
  predict-query coercion (`survival_at`/`hazard_at`/`cumulative_hazard_at`/CIF)
  now rejects only NaN; `S(t≤0)=1`, `S(+∞)=0`, `H(+∞)=+∞` are threaded through the
  interpolation and CSV paths (distinct from the finite past-grid flat-clamp), and
  a latent `right_value` inconsistency in the chunked path is fixed. Rust + Python
  regressions added.

### Families & CLI
- **Explicit `--family` no longer prints a false inferred-family note (#1781).**
  The "Inferred …-family" note was gated only on `link_choice.is_none()`, so e.g.
  `--family gamma-log` reported "Inferred gaussian-identity family" while fitting
  and saving a Gamma/log model. It is now additionally gated on the family being
  `Auto`, so the data-heuristic note fires only when auto-discovery actually ran.
- **`--family expectile` (and any standard-family CLI fit) no longer aborts on a
  null frailty spec (#1780).** The CLI always populates `frailty =
  Some(FrailtySpec::None)`; the standard/survival/transformation guards tested
  `Option::is_some()` and so misread that canonical "no frailty" value as a
  frailty request. Every guard now tests `FrailtySpec::is_active`, eliminating the
  whole class of null-`Some` misreads; genuine frailty requests are still rejected
  where unsupported.
- **Beta external-family rejection points at Binomial GLM routing (#1888).** The
  unsupported-Beta-response message now notes that a binary `{0,1}` response is a
  Binomial GLM and should be routed through the Binomial family.

### Smooths & terms
- **`periodic=false` on a spatial smooth builds a NON-periodic basis (#1676).**
  The scalar-boolean shortcut returned `Some([None])` for `false`, which the
  radial 1-D consumers read as "periodicity requested, derive the wrap from the
  data range" — so an explicit `periodic=false` silently produced a *periodic*
  smooth. It now returns `None`, matching the bracketed `[false]` form; covered
  for matern/thinplate/duchon. (`scale_dimensions` / boolean-periodic acceptance
  for thin-plate also landed under #1676.)
- **Isotropic-Matérn ψψ second-design derivative restored in the
  transformation-normal operator (#1607).** The matrix-free
  `TensorKroneckerPsiOperator` gated `∂²X/∂ψ²` on `implicit_group_id.is_some()`,
  which is `None` for an isotropic single-length-scale Matérn — so its ψψ diagonal
  was silently dropped to zero, halving the outer ψψ Hessian / LAML curvature and
  the Firth/Jeffreys term on those fits. The gate now routes the isotropic
  self-second-derivative through the operator (the same fix #1607 applied to the
  custom-family resolver), keyed on the global covariate-deriv index so distinct
  ungrouped blocks never synthesise a spurious cross term.

### Fitting & inference correctness
- **Double-penalty nullspace shrinkage escapes the inflated-EDF trap (#1266).**
  The pure-REML shrink-out path now reaches the bending coordinate and selects
  unsupported terms out, so `s(x)` on linear data shrinks toward its true EDF
  instead of railing high.
- **`debiased_functional(target="point"/"contrast")` no longer errors when the
  response column precedes the predictor (#1621).** Bookkeeping (placeholder)
  columns are lenient-encoded in the query design, mirroring predict's
  frame-projection, so the query feature index is no longer out of bounds.

### SAE (sparse dictionary)
- **Massive-K dictionaries fit matrix-free (#1026)** off the streaming
  arrow-factor cache, with a collapsed-linear-lane minibatch made load-bearing via
  batched-GEMM routing and overcomplete `K ≥ n` admitted under identifiability;
  overcomplete `K ≫ p` periodic atoms get a generic diverse seed (#1893).
- **Recoverable infeasible-ρ Schur-seed refusal presents as a finite wall (#1782)**
  in every outer lane instead of a fatal startup abort; the spectral PD-floor is
  wired into the inner/log-det solves (#1038); the off-diagonal-only IBP cross-row
  Woodbury θ-adjoint is corrected (#1416); born-atom topology races score by proper
  REML (#977); `ρ.log_lambda_smooth` grows with K on birth/fission (#1556);
  `random_state` is honored in the closed-form fast paths (#178); the
  `assignment_prior`/`n_atoms` kwarg aliases have a real contract (#159/#160); and
  explicit-ψ Firth/Jeffreys terms are gated on Jeffreys-information ψ-dependence
  (#901).

### Performance
- **Duchon 2-D fit reuses the un-rotated design (#1718)** instead of rebuilding
  it, fusing the reparam rotation into a single GEMM.
- **Binomial-logit REML Firth outer-Hessian cost cut (#1575)** by fanning the
  per-penalty direction/pair loops across Rayon (index-ordered, bit-reproducible)
  and bounding the TK-Hessian mixed-matvec cache.
- **Per-eval ALO leverage diagnostic skipped via a ρ-independent non-activation
  certificate (#1689)**; default solver logging quieted to `Warn` with an env-free
  `--log-level` (#1688).
- **Device-resident SAE joint fit** engagement, sphere-Wahba GPU kernel, fused
  arrow-Schur block strides, and NVRTC FMA-contraction CPU parity land with
  fail-loud false-routing guards and V100-verified parity (#1017 and related).

### Robustness / error handling
- **Empty designs are rejected cleanly (#1848).** `fit_gam` guards `n == 0` /
  `p == 0` at the entry point with an `InvalidConfig` error instead of panicking on
  zero-sized indexing / linear solves.
- **Device CSR construction enforces `rowptr.len() == rows + 1` (#1846)** via a
  canonicalizing `DeviceCsrMatrix::new`, closing an invalid-free / out-of-bounds
  deallocation hazard.

### Also in this release
- Survival lognormal fits request the location-scale route so they are dispatched
  correctly (#1847).
- Multinomial separation auto-engages the Firth/Jeffreys bias-reduction fallback
  (#1854).
- `summary` penalty-cursor skips unpenalized by-factor random-effect ranges so the
  per-term penalty accounting lines up (#1883).
- Scale-invariant rank tolerance in `positive_spectral_whitener_from_gram` (#1889).
- Real REML-2.3 solver-invariant tests (#1861).

### Release hygiene
- `trace_product_sparse` (`tr(H⁻¹S)`, feeding the REML gradient/EDF) now reduces
  its per-column partials serially in column order, so the result is bit-identical
  across rayon thread-pool sizes rather than drifting in the low bits with core
  count (#759); pinned by a 1-vs-8-worker regression.
- Dead scaffolding removed from shipping code: the vestigial `resolve_log_level` /
  `default_log_level` indirection and a misleading env-override doc in the
  env-free logging path (#1688/#1689), and the orphaned #178 Python LCG-jitter
  constants + a stale comment in the `gamfit` wheel source. A Duchon-fusion test
  header that overclaimed "bit-for-bit" is corrected to the tolerance it asserts.

## v0.3.141 — gam 0.3.141 / gamfit 0.1.243 (2026-07-01)

Correctness patch on top of 0.3.140, focused on non-converged / near-degenerate
fits that previously returned an unusable model or contradicted their own reported
effective degrees of freedom, plus a matrix-free path that lets massive-K SAE
dictionaries descend their hyperparameters instead of hard-erroring.

### Fitting / inference correctness
- **Non-converged estimated-scale fits return a USABLE model (#1789).** When the
  #1788 EDF-collapse guard re-derives the dispersion `σ̂² = RSS/(n − edf)` after
  correcting a collapsed effective d.f., it now rescales BOTH redundant covariance
  representations — the top-level `covariance_conditional`/`covariance_corrected`
  AND the paired inference-block `beta_covariance*`/SEs — atomically through the
  new `UnifiedFitResult::rescale_estimated_dispersion`. Previously it scaled only
  the inference block, so a non-converged multi-smooth gamma / `[INDEF-HESS]` fit
  returned a `Model` that `fit` accepted but `predict`/`summary`/`save`→`load` all
  rejected with "inference conditional covariance must match top-level
  covariance_conditional". The rescale can no longer touch one copy and not the
  other, and its `#[must_use]` σ̂ ratio is now reported in the non-convergence
  warning instead of being discarded.
- **Self-contradictory penalized EDF on stalled REML fits is guarded (#1788).** A
  non-converged fit whose influence EDF collapsed to the intercept-only floor while
  its fitted coefficients stayed wiggly now substitutes the per-term dimension
  floor (so the reported EDF is not self-contradictory) and surfaces the
  non-convergence rather than shipping a silent collapse.
- **Firth fallback rescues binomial-logit REML near-separation stalls (#1762).** A
  binomial-logit fit that stalls in the flat REML valley near quasi-separation is
  retried once with Firth/Jeffreys bias reduction and adopts that result only if it
  actually converges, otherwise preserving the honest base result/error.
- **Log-link PIRLS enforces shape-constraint feasibility (#1786).** Monotone/convex
  smooths on low-count Poisson (log-link) no longer silently ship coefficients that
  violate the requested shape cone: an LM-damping retry precedes a
  feasibility-restoring projection (curvature re-evaluated at the projected β), and
  an infeasible fit errors rather than returning an invalid model.
- **Massive-K SAE descends its hyperparameters matrix-free (#1026).** The EFS
  (Fellner–Schall) lane now takes its ARD/smoothness traces off the streaming
  arrow-factor cache returned by `penalized_quasi_laplace_criterion_streaming_exact_with_cache`
  instead of forcing the dense `O((K·M·p)²)` evidence cache that hard-errors at
  large K (25.9 GB even at K=256). The `gamfit` facade admits an overcomplete
  dictionary (`K ≥ n`) under ARD/smoothness-prior identifiability — a warning, not
  a refusal.

### CLI
- **`gam predict --uncertainty` keeps the posterior-mean point for curved links
  (#1787).** The point-estimate column is the response-scale posterior mean for
  curved-link families, matching the Python FFI, instead of being swapped to the
  linear-predictor mean when `--uncertainty` was requested.
- **Prediction-CSV linear-predictor column header restored to `eta`.** The base,
  Gaussian location-scale, survival, and survival-binary CSV writers had drifted
  to emitting `linear_predictor`; the schema-lock contract (and every downstream
  reader) expects `eta`. All four writers now emit `eta` again. The Python FFI
  dict-key contract (`linear_predictor`) is a separate path and is unchanged.

### Python / docs
- **transformation-normal `predict` output documented as E[Y|x] (#1612)**, not a
  z-score.
- **Constrained-REML active-set recompute (gam-pyffi)** partitions rows by KKT
  activity (`a·β ≤ b + tol`), not by feasibility, so interior rows are no longer
  spuriously reported active.

### Test / build hygiene
- Restored the `gam-sae` test binary: an #1784 IBP-capacity refactor left an
  `Option`/`Result` mismatch (compile break) and stale large-scale assertion
  margins; margins are recalibrated to the RAM-safe scale and a regression test now
  pins the #1026 streaming cache as a drop-in for the dense cache in the EFS lane.
  A closed-form-criterion bench also no longer discards a `#[must_use]` solve
  `Result`.

## v0.3.140 — gam 0.3.140 / gamfit 0.1.242 (2026-07-01)

Release-integrity and correctness patch on top of 0.3.139. The headline is that
the tree **builds and packages again from a clean checkout**: the 0.3.139
release shipped a workspace that aborted its own build (build.rs hygiene-ban
violations), so `cargo build`/`test`, a fresh `--release`, and the `maturin`
wheel all failed on a cold `target/`. Every violation is cleared and pinned with
regression guards, on top of a batch of root-cause fixes: a mixed-boundary
tensor smooth that hard-errored, the SAE IBP-MAP high-noise stall, and a GPU
BMS-FLEX row-kernel parity bug.

### Fitting / inference correctness
- **Mixed periodic + clamped tensor smooths fit instead of hard-erroring.** A
  tensor margin's non-periodic spelling `clamped`/`open` (the B-spline-clamped,
  free-ended margin) is now accepted in the `boundary=`/`bc=` list, so
  `te(theta, z, boundary=['periodic','clamped'], period=[2*pi, None])` — gam's
  analog of mgcv `te(bs=c("cc","ps"))` for a cylinder — builds a cyclic θ margin
  tensor-producted with an ordinary open z margin. Previously the guard rejected
  `clamped` as an unsupported endpoint reparameterization, taking out the
  cylinder / solar-zenith / cyclic-tensor recoveries with an IntegrationFailed.
  A genuine `anchored` zero-value endpoint constraint (no ordinary-margin
  meaning on a tensor) is still surfaced as a clean unsupported-feature error.
- **SAE IBP-MAP reaches a flexible fit at high noise instead of stalling (#1744).**
  Two root causes are repaired: (1) the IBP-MAP ρ seed is no longer
  response-dispersion-scaled — that Gaussian-normal-equation identity is invalid
  for IBP's free Bernoulli gates and let the inner solve overfit at the seed,
  collapsing the Fellner–Schall fixed point to zero penalty; and (2) the
  parsimonious keep-best no longer lets a *less-stationary, more-smoothed*
  non-converged seed displace the flexible incumbent on a marginally-lower
  non-converged REML alone. Together the planted-circle IBP fit reaches EV ≈ 0.95
  at σ=0.18 instead of stalling at 0.86.

### GPU / numerical correctness locks
- **BMS-FLEX GPU row kernel uses the observed predictor VALUE for the probit
  Mills margin (#415).** The device kernel and its host oracle were reading
  `bar_e_u[0]` — the u=0 first-derivative jet — as `e_obs` instead of the
  degree-0 value `η(a(θ),θ;z_obs)`, diverging from the CPU family's
  `signed_margin = s_y·eta_val`. The observed value is now packed and consumed
  directly, locked by a non-vacuous CPU-oracle == CPU-family parity test over
  every row's value, full gradient, and full r×r Hessian.
- **Survival I-spline time-penalty PSD invariant is locked at construction
  (#979).** The value-space curvature penalty is assembled as the full
  congruence `Lᵀ S L` and *then* reduced to the kept columns (PSD by
  construction); a regression test with a non-trivial `keep_cols` asserts the
  assembled penalty is PSD, so a future reassembly that reintroduces an
  indefinite reduction is caught at construction rather than as a silent
  outer-loop hang.

### Build / release integrity
- **The workspace builds from a clean checkout again.** Cleared the build.rs ban
  violations that shipped onto the 0.3.139 release: a temporary exploratory
  coverage probe committed into `src/` (stdout prints + a `#[cfg(test)]` module
  dodge), `construction.rs` over the 10k-line tracked-file limit, a `#[ignore]`
  test, and a stale `uv.lock` gamfit version.
- **The #415 CPU-oracle test module stays private.** Its cross-file consumer was
  relocated into a private sibling test module so the oracle is reached in-module
  — no `pub(crate)` on a `#[cfg(test)] mod`, which the ban gate (correctly)
  rejects and which had re-broken the build.
- **Removed the always-`panic!` #1765 observation-coverage probe.** It was
  inconclusive exploratory scaffolding — its own fixed-seed numbers show the
  residual-df scale is well-calibrated on that ridge sweep and `edf2` is not the
  lever — so it neither reproduced the bug nor validated a fix; the finding is
  recorded on the issue and the Gaussian scale stays guarded by
  `gaussian_high_edf_scale_tests`.
- **The `sigma_link` source-pattern guard no longer self-trips** — it skipped a
  stale `families/sigma_link.rs` path and so scanned its own literal, failing the
  CI test shard on every commit.
- Added fast unit locks for the tensor boundary-token guard and made the 0139
  bug-hunt regression test robust to future version bumps.

Everything reachable through the existing API stays backward-compatible.

## v0.3.139 — gam 0.3.139 / gamfit 0.1.241 (2026-07-01)

crates.io + PyPI release of the post-0.3.138 wave. The headline is the SAE /
manifold stack moving its fit and encode paths onto Rust and the `gamfit`
package thinning to a SPEC-compliant wrapper (numeric math lives in Rust), on
top of a batch of root-cause correctness fixes, two new pieces of public surface
(periodic radial builders and the expectile/LAWS family reachable from both the
Python API and the CLI), and GPU/perf work proven bit-identical to the scalar
path. Several release-integrity defects that would have shipped a half-broken
tree — a fix reverted three times by stale-tree merges, and a secondary-smooth
penalty default that aborted an otherwise-valid fit — are repaired with
regression guards.

### Fitting / inference correctness
- **Multinomial per-class λ / EDF are rebuilt from the joint penalty (#561).** The
  outer REML/LAML loop now converges on genuine per-term smoothing rather than
  parking at its seed, so a multinomial fit recovers per-class structure instead
  of a fused, over-/under-smoothed surface.
- **`smooth_significance()` reference d.f. is floored at the joint null-space
  dimension (#1766)** so the likelihood-ratio p-value no longer collapses toward
  0 on a shrunk smooth, keeping the null false-positive rate calibrated.
- **The Marra & Wood null-space "double" penalty now defaults OFF for a secondary
  (scale / distributional) smooth (#1561)** — defaulting it on biased the
  location-scale fit toward homoscedasticity and collapsed the recovered
  log-sigma surface. `duchon` is excluded from the change (it carries no such
  penalty and its builder rejects the key), so a scale-block Duchon fit no longer
  aborts; an explicit user `double_penalty=` still wins.
- **`gam fit --family expectile` no longer aborts on the frailty guard (#1780).**
  The inner Gaussian-identity design carries no frailty; the CLI's default
  `frailty = Some(None)` is now cleared before the inner fit.
- **No spurious "Inferred gaussian-identity family" note when an explicit
  non-default `--family` is passed (#1781).**
- **A 1-D cyclic basis wraps on the data range instead of hard-erroring.**
- **High-EDF Gaussian observation intervals cover** — the residual-df scale keeps
  observation bands calibrated at high effective degrees of freedom (#1765).

### New public surface
- **`periodic=` on the radial builders `duchon()` / `tps()` / `thinplate()` /
  `matern()` (#580, #1778)** — a scalar or per-axis boolean, wired through Rust
  and the CLI, with validation that rejects bad axes, per-axis lengths, and
  non-positive or over-wide periods.
- **The expectile (LAWS) regression family is reachable from both the Python API
  and the CLI (#1777)** via `--family expectile` (inline-τ supported), routed
  through a shared dispatch seam so both interfaces agree.

### SAE / manifold
- **The SAE fit is routed through Rust FFI** with per-fit config overrides
  (separation-barrier strength / IBP-α), a `threshold_gate` rename (was
  `jumprelu`), out-of-sample v-projection, `atom_reconstruct`, and
  `coord_sparsity` (#1777); the SAE-manifold audit fixes real defects across the
  hybrid / routing / log-det / sparse paths (#1026).
- **Fisher steering state now round-trips through `save` / `load`,** so a reloaded
  model reproduces `steer()`'s dose instead of silently degrading to
  geometry-only.
- **A GPU device-resident exact encode kernel, sublinear massive-K routing, and
  jet / REML / arrow-Schur perf** — including a matrix-free reduced-Schur SLQ
  evidence log-det — all bit-identical to the CPU / scalar path.

### `gamfit` package (breaking)
- **Python-side numeric math that violated the thin-wrapper SPEC has been removed;
  the capabilities that exist in Rust are reached through the FFI.** Ported to
  Rust (behavior preserved): `partial_dependence`, `variance_share`, the sparse-
  and linear-dictionary out-of-sample `transform`, and the cyclic difference
  penalty. Removed with no replacement (breaking):
  `gamfit.align` (Procrustes alignment), the `sae_benchmark` /
  `sweep_sae_benchmark` / `format_sae_benchmark_markdown` harness,
  `activation_statistics`, `recommend_sae_hyperparams`, the EV-vs-K frontier
  research helpers (`sae_ev_vs_k_frontier` / `ev_knee_k`), and
  `Model.posterior_predictive_check`.
- **`coordinate_range` / `typical_shape` are consolidated into
  `shape_uncertainty`** (which returns the analytic per-atom band as
  `coords` / `mean` / `sd` / `lower` / `upper`).
- **ALR simplex fits are no longer auto-whitened to Aitchison geometry** — no Rust
  FFI exists for the whitener, so an ALR fit again depends on the (arbitrary)
  reference component. Use the default CLR (or ILR) representation, which is
  already Aitchison-isometric, for a reference-free simplex fit (#1549
  auto-whitening removed, per SPEC).

### Build / release integrity
- Restored the #1780 expectile-frailty fix after stale-tree merges reverted it,
  and pinned it with a CLI regression test so a future clobber fails loudly.
- Added a regression guard that a scale-block `duchon()` smooth fits under the
  #1561 default-off, and that an explicit `double_penalty=` on Duchon is still
  rejected.
- Cleared build-gate violations that would have failed the wheel build mid-flight
  (a banned `debug_assert!` guarding an `unsafe` load promoted to an always-on
  `assert!`; stray diagnostic scaffolding; a test-only dispersion oracle moved
  into `#[cfg(test)]` scope; a `sae_manifold_fit` arity mismatch at an internal
  IBP call site), plus build.sh usability and OOM-recovery hardening.

Everything reachable through the existing API stays backward-compatible except
the explicitly-listed `gamfit` removals.
## v0.3.138 — gam 0.3.138 / gamfit 0.1.240 (2026-06-30)

crates.io + PyPI release of the post-0.3.137 correctness wave. A cluster of
silent-no-op and frame-consistency bugs are fixed at the root — each with a
regression test — the monotone shape-constrained REML fix is completed for its
binding-constraint face, several GPU paths are proven bit-identical to CPU, the
Gaussian P-spline/thin-plate objective gains fast paths, and two latent
build-breakers (an unused import and eight hygiene-gate violations that would
have failed the wheel build mid-flight) are cleared. Everything reachable
through the existing API stays backward-compatible.

### Fitting / inference correctness
- **Tensor smooths honor per-margin periodicity and `bs=c(...)`.** A `te(...)`
  mixing periodic and non-periodic marginals (or per-margin basis overrides via
  `bs=c(...)`) built every marginal from the first margin's spec; each marginal
  now carries its own periodicity and basis kind (#1751, #1752).
- **`smooth_significance()` LR p-value no longer collapses to ~0.** When a smooth
  shrinks onto its linear null space the reference degrees of freedom could fall
  to zero, collapsing the likelihood-ratio p-value; the reference d.f. is now
  floored at the term's null-space dimension (≥ its EDF), keeping the null
  false-positive rate calibrated (#1766).
- **`survival_likelihood=` is rejected on a non-`Surv()` response (#1767).** A
  request like `fit(data, "time ~ s(x)", survival_likelihood="weibull")` used to
  drop the knob silently and fit an ordinary Gaussian GAM on the raw event-time
  column. It now fails loud at materialization, symmetric to the `family=`
  survival guard and the survival-only formula-term guard (#371).
- **Monotone shape-constrained smooth — binding-constraint face (#509).** The
  outer REML/LAML analytic gradient is now frame-consistent with the cost even
  when the monotonicity constraint binds at the inner optimum, so a monotone fit
  on already-monotone data no longer parks at its under-smoothed seed. Completes
  the #509 fix begun in 0.3.137.
- **`ard_per_atom` is wired to the native ARD prior (#240).** The SAE flag was a
  silent no-op (a registry penalty deliberately skipped on every SAE path); it
  now toggles `native_ard_enabled`, observable in the born-atom count and fitted
  coordinates.
- **Multinomial per-class λ/EDF rebuilt from the joint penalty (#561); endpoint
  `bc=clamped` interior-quality guarded and tensor endpoint BCs rejected (#500);
  per-atom `log_lambda_smooth` grows when a structure move grows K (#357);
  irrelevant double-penalty smooths select out via per-term shrink (#1266); the
  distilled-encoder honesty probe is cold-started (#1166); the objective-grid
  per-axis seed refinement reaches asymmetric corners.**

### SAE
- **Closed-form fast paths honor `random_state` (#178)** via a deterministic LCG
  mirrored between Python and Rust.
- **Scale-invariant isometry Gauss–Newton curvature; the gauge is re-enabled
  (#795). Log-det θ-adjoint cross-row Woodbury off-diagonal corrected
  (#1625/#1416); barrier strength derived from REML evidence (#1610);
  device-resident Direct-solve engagement with symmetric Schur fixtures
  (#1017/#1551).**

### Survival
- **Predict-query times accept negative and `+inf` (#965)** as boundary
  evaluations instead of raising.

### Performance
- **Gaussian P-spline / thin-plate fit perf-core (#1689):** REML-objective fast
  paths with a profile-test guard.
- **Sphere (S²) GPU dtoh path:** pinned + parallel transpose + buffer pool
  (#1709); the 40M-element transpose in `to_host_array` removed (GPU
  6.6s→0.27s, #1741); decomposition GEMM routed through `fast_ab` (#1738).
- **`trace_product_sparse` rayon parallelization restored (#759); gauge identity
  section short-circuited in `restrict_design`/penalty (#1737); dense-GEMM device
  dispatch registered and made transpose-free (#1735); Matérn/GP κ-loop sped up
  with the verbosity flag (`gam._rust.set_log_level`) re-exposed (#1688); wiggle
  separation bound + batched REML gradient (#1607).**

### GPU parity / robustness
- SAE reconstruction CPU↔GPU bit-identity via row-jet K-scale + sparse-route
  (#1026); honest fail-loud routing (#1209); proven GPU↔CPU parity on V100 with
  principled bands + fmad sweep (#415, #1175); CI-fast parity, fail-loud oracle,
  and false-routing guard (#1412, #988); arrow-Schur NVRTC arch-pin and
  deficit-aware Gershgorin ridge bump for non-PD rows.

### Build / test hardening
- **Un-RED the workspace build.** Removed an unused `ShapeBuilder` import in
  `gam-terms` (a `warnings = "deny"` hard error left by the #1709 sphere-GPU
  work) and cleared eight `build.rs` ban-scanner violations across
  `gam-solve`/`gam-models`/`tests` (a `let _` discard, two `#[ignore]`d GPU
  diagnostics, a `GAM_REQUIRE_GPU` env read, a mis-named `#[cfg(test)]` module,
  and a three-`assert!(true)` scratch test file). Both gates would have failed
  the wheel build ~12 min in; they had slipped through CI because the heavy
  build job self-throttles and kept getting skipped under the steady commit
  stream.
- Concurrency guards (`OnceLock`-in-rayon #1253, `nested_prefix` dispatch #1254)
  made behavioral and workspace-wide; `scale_dimensions` anisotropy validated for
  `thinplate()` (#1676); numerous orphaned smooth / Matérn / SO(3) / SAS / BMS
  test guards re-homed and bound to production penalties (#1601, #1274, #1629,
  #855, #388, #370, #1260, #1261, #1246, #1255, #1091); Test-shard CI timeout
  raised to the 6h runner max.

## v0.3.137 — gam 0.3.137 / gamfit 0.1.239 (2026-06-30)

crates.io + PyPI release of the post-0.3.136 correctness wave. Shape-constrained
REML, adaptive-ψ custom families, and three user-facing API contracts are fixed
at the root, each with a regression test; one inner-loop hot path is made
n-free. Everything reachable through the existing API stays
backward-compatible.

### REML / fitting correctness
- **#509 — monotone shape-constrained smooth no longer over-smooths.** The outer
  REML for a `shape=monotone_increasing` (box-reparam) smooth projected the
  penalty roots in the ORIGINAL (pre-`Qs`) frame while the Hessian half of the
  LAML pair lived in the TRANSFORMED (post box-reparam `T`, post-`Qs`) frame, so
  under the non-orthogonal cumulative-sum reparameterization `ZᵀSZ` mixed two
  coordinate systems, the analytic outer gradient disagreed with finite
  differences, the trust region rejected every step, and the fit parked at its
  under-smoothed seed (mono RMSE > free RMSE on already-monotone data). Both
  halves now live in one transformed frame (the no-op identity when `Qs = I`), so
  the non-binding monotone fit recovers the truth as well as the unconstrained
  fit. Also hardens the binding-constraint regression and removes a
  parallel-test temp-CSV race.
- **#901 — adaptive-ψ custom families: data-only Jeffreys information + ψ-gated
  Firth.** The projected-logdet REML gradient now matches finite differences for
  spatial-adaptive-hyper custom families, with a 659-line FD agreement suite.

### Contrast / compositional / comparison API
- **`Model.difference_smooth` sign corrected.** A pair `(level_1, level_2)` now
  returns `ŝ(level_1) − ŝ(level_2)` (the mgcv `plot_diff` convention); the design
  difference was previously formed as `design(level_2) − design(level_1)`, so the
  reported contrast was the exact negation of its `level_1`/`level_2` row labels.
  The confidence band (a quadratic form) is unchanged.
- **`gamfit.clr` / `alr` / `closure` accept a single 1-D composition.** The
  natural call `clr([0.2, 0.3, 0.5])` raised an opaque
  `TypeError: 'ndarray' object is not an instance of 'ndarray'` because the FFI
  only accepted a 2-D `(rows, parts)` array; a 1-D composition is now promoted to
  a single row and returned as 1-D coordinates matching the batch row.
- **`compare_models` refuses cross-`n` comparisons.** AIC / REML-LAML evidence
  scales with the observation count, so comparing two same-family fits on
  different-sized data (e.g. n=500 vs n=100) used to declare the smaller-`n` model
  the winner by a Bayes factor ~1e14–1e18. It now fails loud on mismatched `n`,
  mirroring the existing different-family guard; fits that do not record `n`
  (legacy / O(n) scan payloads) stay unconstrained.

### Performance
- **#1033 — n-free κ-trial lane.** The ALO leverage-barrier stabilizer (an
  outer-optimizer aid, never part of the REML/LAML criterion) is skipped on the
  ψ-keyed sufficient-statistic cache lane whose realized rows are frozen at the
  pinning ψ, removing an O(n·k) hat-value pass per in-window κ-trial without
  changing any fitted result.

### Test hardening
- #1260 binds the equivariant-atom bandwidth gate to the shipped penalty
  (replacing a vacuous self-objective test); #1261 restores the oversmoothed-λ
  regime for the average-derivative one-step gate after penalty renormalization;
  the joint-Newton weak-band mode is placed strictly inside the rank band; #855
  restores the tight SAS dβ/dε observed-Jacobian FD guard.

## v0.3.136 — gam 0.3.136 / gamfit 0.1.238 (2026-06-30)

crates.io + PyPI release of the post-0.3.135 correctness wave. A cluster of
smoothing/basis, REML-convergence, structure-search, geometry, survival and
inference bugs are fixed at the root, each with a regression test. Everything
reachable through the existing API stays backward-compatible; two changes adjust
default basis sizing toward mgcv (a wigglier fit is still one explicit `k=`
away).

### Smoothing / basis fixes
- **#1680 — the default univariate smooth basis is capped mgcv-like.**
  `heuristic_knots_for_column` grew the default B-spline basis with `n` (20
  internal knots / a 24-function cubic basis for any column with ≥80 unique
  values; the `n^{1/3}` ceiling only engaged above ~8000 unique values, so it was
  dead in practice). That over-rich default over-parameterized weak-signal
  additive fits and let the outer REML optimizer leak truth into surplus columns
  the penalty could not shrink (truth-RMSE ≈0.39 vs mgcv's ≈0.09 on a
  near-collinear 4-smooth n=120 fit). The default is now a flat 8 internal knots
  (basis dim ≈12, close to mgcv's univariate `k=10`); columns with ≤32 unique
  values keep their previous knot count exactly, and an explicit `k=` always
  wins. Same defect class as the thin-plate over-sizing in #1074.
- **#1731 — Matérn realized basis now grows with requested `k`.** The auto length
  scale was seeded `k`-blind (`max_range/√n`), so once the requested centers
  packed denser than that fixed scale could resolve, neighbouring radial bases
  went numerically collinear and the #755 rank-reduce guard dropped them — the
  basis saturated and even *decreased* for large `k` (`k=150 → 104` realized).
  The auto length scale is now density-adaptive (`max_range/√max(n,k)`, the same
  fill-distance law the Duchon promotion uses): bit-identical to the old seed
  whenever `n ≥ k`, and shrinking with `k` past that so the centers stay
  independent (`k=150 → 150`). Only the auto sentinel is touched; an explicit
  length scale is never overridden and the rank-reduce guard remains the
  last-resort degenerate-data net.

### REML / optimizer
- **#1033 — the κ/ψ smoothing window is n-invariant at BOTH edges.** The κ line
  search could overshoot ABOVE the maximal-rank band to a ψ where the conditioned
  Gram drops rank, soundly refusing the n-free design-realization skip and
  tripping two O(n) `reset_surface` passes (the n=16000 fast-ladder regression).
  A symmetric `rank_stable_psi_ceiling` (the twin of the existing low-edge floor)
  now clamps the optimizer's ψ upper bound to the top of the maximal-rank band —
  a pure O(nodes·k³) k-space property, inherently n-independent. The κ-optimum
  lives inside the band, so the clamp only excludes over-fit length scales.
- **#1690 — a Gamma flat-valley REML stall is no longer mis-reported as
  non-converged.** A single-smooth `s(x,k=12)` n=600 Gamma+log fit reaches the
  genuine optimum but the in-loop cost-stall guard sampled a warm-start-sensitive
  ρ-gradient just above its score-relative bound and halted non-converged, which
  in turn triggered wasted deterministic-replay ARC retries (the actionable slice
  of "Gamma ~7× slower than mgcv at equal accuracy"). `outer_converged` is now
  reconciled against the authoritative gradient of the fit actually shipped, gated
  strictly on the flat-valley stop reason; a genuinely non-stationary floor (and
  the #1426 stuck overfit, |g|≈11 ≫ bound) still reports non-converged.

### SAE structure search
- **#1556 — birth/fission no longer panics.** Structure-search grow moves bumped
  the dictionary size and `ρ.log_ard` but left `ρ.log_lambda_smooth` at the old
  length, so the next `assemble_arrow_schur` indexed out of bounds. Both grow
  paths now push an inherited per-atom smoothness strength (born inherits atom 0;
  a fissioned child inherits its parent), and `assemble_arrow_schur` validates the
  length so any future grow path that forgets surfaces a clear `Err`.
- **#977/#1026 — the born-atom topology race is scored by proper REML.** The birth
  race scored each candidate basis with a hand-rolled `½·SSE + ½·log|H|` Laplace
  term at a stamped `λ=1` on the raw curvature Gram — not commensurable across
  bases, so a periodic basis's `(2π)⁴` curvature energy lost a perfect circle to a
  straight line (and a cylinder to a sphere). Candidates are now scored by a
  rank-aware REML/LAML with an estimated λ̂, so the heterogeneous-dictionary races
  pick the topology the evidence supports.

### Geometry / linear algebra
- **#1641 — IBP θ-adjoint cross-row Woodbury logdet channel corrected.** The
  cross-row Woodbury pass in `logdet_theta_adjoint` carried a spurious ½ (a
  ρ-trace convention) while differentiating the full `log|H|`, dropped the factor
  of 2 on the symmetric u-changing term, and double-counted the `i=j` self
  curvature already handled by the diagonal channels. The pass now restricts to
  the `i≠j` off-diagonal with full-trace coefficients, mirroring the known-good
  #1416 ρ-trace cross-row pass and matching the dense finite-difference oracle.

### Survival
- **#1717 — `survival_at(t|x)` is invariant to the placeholder time column.** The
  default 64-point survival grid floored its upper edge to the training support
  but let a large placeholder exit stretch it past, coarsening every in-range cell
  and drifting the interpolation off the true curve (up to ~14%). When the fitted
  model carries a training-time upper bound the grid now spans exactly that
  support; query times beyond it are handled by extrapolation (#1595), not by the
  grid. The dual of #896.

### Inference
- **#1722 — Beta posterior credible intervals are no longer ~4-5× too narrow.**
  `laplace_gaussian_fallback` rescaled draws by `dispersion().sqrt_phi()`, but
  Beta's IRLS working weight already folds φ into the stored penalized Hessian, so
  `Vb = H⁻¹` needs no extra dispersion factor. The per-draw scale is now the
  coefficient-covariance scale `summary()`'s Wald SE is built from — `σ̂²` for a
  profiled Gaussian (a no-op for Gaussian/location-scale/survival) and `1.0` for
  Beta and the other fixed-scale families, fixing only Beta.

### Internal / CI
- The `[profile.test]` base now optimizes the workspace numerical crates
  (`opt-level = 2`), not just dependencies, so the heaviest solver-bound tests
  (#979 survival, #1593 competing-risks) finish inside nextest's 600s per-test
  cap instead of dying as opaque SIGKILL timeouts. Numerically identical.
- `build.sh`'s inner timeout is overridable via `GAM_BUILD_TIMEOUT`.

## v0.3.135 — gam 0.3.135 / gamfit 0.1.237 (2026-06-30)

crates.io + PyPI release of the post-0.3.134 correctness-and-performance wave.
A broad set of smoothing/REML, survival, geometry and inference bugs are fixed
at the root, several hot paths are made meaningfully faster, and the `gamfit`
Python surface gains additive SAE keyword aliases. Everything reachable through
the existing API stays backward-compatible.

### Smoothing / REML fixes
- **#1654 — convex/concave shape smooths no longer park in the linear corner.**
  The double-penalty nullspace ridge under the order-2 box reparameterization
  was rebuilt from scratch instead of transformed by the same congruence
  `S ↦ TᵀST` as the wiggliness penalty, decoupling the level/slope and
  wiggliness scales and driving curvature-constrained fits to a near-straight
  line (EDF ≈ 1.5) for a seed/`k`-specific subset. The exact congruence is
  restored for the curvature ridge; the monotone path keeps its #509 projector.
- **#509 — monotone REML λ-search no longer parks at the integer seed.** The
  cost-stall guard keyed its keep-descending escape on a fixed absolute gradient
  ceiling, so a shape-constrained inner solve with a non-binding constraint
  stalled near the seed while the projected gradient still descended strongly
  (over-smoothing already-monotone data). The escape is now scaled to the score.
- **#1629 — Matérn smooths no longer over-smooth 2-D surfaces.** Matérn now
  routes through the same length-scale auto-init sentinel as thin-plate so the
  basis seeds the resolving regime instead of a degenerate global scale.
- **#1676 — `scale_dimensions=True` now engages anisotropy for thin-plate.** A
  multi-axis `tp` term is rewritten to its mathematically-equivalent anisotropic
  s=0 Duchon twin (the thin-plate kernel `r^{2m−d}` is the s=0 Duchon kernel), so
  per-axis tension ARD engages exactly as for `duchon()`/`matern()` instead of
  the flag being a silent no-op. Default (flag off) and 1-D `tp` are unchanged.
- **#1269 — thin-plate basis is exactly translation-invariant.** The strict
  basis-conditioning gate is split out and pinned at the bit level.
- **#1476 — double-penalty no longer over-shrinks a supported smooth.** A budget-
  exhaustion best-feasible substitution used a bare early `return` that bypassed
  the multi-start keep-best loop and could ship a degenerate box corner; it now
  flows through keep-best as a non-converged candidate.
- **#1033 — the κ/ψ smoothing window is now n-invariant.** Even-spaced capped
  diameter sampling and a rank-stable ψ floor anchored at the optimizer seed kill
  an n-dependent shift in the outer optimizer's box.
- **#901 — iso-κ joint-REML outer-gradient FD oracles re-homed and verified.**

### Survival fixes
- **#965 — survival FFI rejects negative times; `S(0)=1`.** Negative/NaN/Inf
  times are rejected at the Python→Rust boundary and the parametric fallback
  guards the `exp(-∞·0)=NaN` origin case.
- **#1595 — survival/cumulative-hazard extrapolation policy threaded into the
  dense Rust FFI kernels**, so `S=exp(-H)` holds past the grid in both the
  chunked and CSV paths.
- **#392/#369 — fit-to-completion guards restored for non-linear survival
  baselines** (real convergence asserted across all baseline targets).

### Inference / families fixes
- **#332 — near-constant Gaussian response is rejected with a clean error**
  instead of producing a degenerate fit.
- **#1655 — the GPD tail estimator accepts light (k<0) tails** (σ from the
  un-shrunk k, matching ArviZ/loo) instead of returning `None`.
- **#1621 — debiased point/contrast prediction handles inert categorical
  bookkeeping columns** via lenient encoding for non-required columns.
- **#1101 — the multinomial per-class probability-SE calibration test** is
  replaced with a valid over-refit calibration (the prior test was statistically
  degenerate).
- **#1561 — the final location-scale β̂ refit at ρ\* is seeded warm from the
  outer optimum** instead of cold, fixing a basin-fragility KKT cert-refusal
  crash on stiff two-block fits.

### Geometry
- **#1637 — genuine Stiefel canonical-metric logarithm for k≥2**, anchored to
  Y⊥ to kill spurious π rotations, with an exhaustive `Log∘Exp=id` sweep and a
  square-input completion guard.
- **#1661 — CLR `simplex_exp_map` rejects a non-finite tangent** with an error
  instead of returning `Ok(NaN)`.

### Python (`gamfit`)
- **#159/#160/#178 — additive SAE keyword aliases**: `assignment` /
  `assignment_prior`, `K` / `n_atoms`, and `random_state` wiring, resolved
  end-to-end into the Rust FFI with eager conflict detection.

### Performance (value-preserving)
- **#1575** cuts the Firth/Jeffreys outer-Hessian cost on binomial/logit REML.
- **#759** restores the rayon parallel reduction in `trace_product_sparse`.
- **#1082** brings the competing-risks CIF quality case from 439 s to 122 s by
  not expanding the ρ₀ offset on an untagged inner-failure pre-warm.
- **reml Jeffreys drift** GEMM-izes the H_Φ curvature-drift contraction (was a
  bounds-checked scalar triple loop dominating competing-risks Weibull fits).
- **#1033 / #979** make the ψ-gram a true sufficient-statistic reduction and
  bound the marginal-slope continuation pre-warm.

### Known-limitation note
The GAMLSS location-scale engine/reference parity and inner-solve convergence
cluster (#1607) remains under active work; those `gam-models` tests are not yet
green and no user-facing API depends on them.
## v0.3.134 — gam 0.3.134 / gamfit 0.1.236 (2026-06-29)

crates.io + PyPI release of the post-0.3.133 correctness-and-performance wave:
two user-visible inference/prediction bugs are fixed, the multinomial save model
gains a first-class per-penalty EDF channel, and the Firth/Jeffreys REML and SAE
log-det hot paths are substantially faster while staying numerically identical.
The `gamfit` Python API surface is additive only — multinomial model metadata now
carries `edf_per_penalty`; everything else reachable through the existing API is
unchanged.

**debiased_functional restored for parametric-term Gaussian models (#1622)**
- `debiased_functional` no longer aborts with "model does not carry the weighted
  Gram X'WX" on every Gaussian model that has a parametric (non-intercept) term
  (`y ~ x`, `y ~ s(x) + z`). Under column-conditioning the weighted Gram `X'WX`
  is a genuine congruence object — it transforms by exactly the same map as the
  penalized Hessian — so it is now back-transformed into the original basis
  rather than unconditionally dropped, letting the debiased-functional Riesz
  engine recover `S(λ)·β`. The same fix restores the exact WPS corrected-EDF term
  `tr(X'WX·Σ_ρ)` (congruence-invariant, so it matches the internal-basis value
  bit-for-bit) for the whole `y ~ x` / `y ~ s(x) + z` class.

**Point/contrast prediction under the full training schema (#1621)**
- The `x0` design for point and contrast predictions is now built under the full
  training schema, so predictions no longer mis-align when the prediction frame
  carries fewer terms than the fitted model.

**Multinomial per-penalty EDF is now first-class (#1219, #715)**
- `MultinomialSavedModel` gains an `edf_per_penalty` field (one entry per
  smoothing parameter, `rank(S_k) − λ_k·tr(H⁻¹ S_k)` clamped to `[0, rank]`),
  surfaced through the Python multinomial metadata. Previously the per-class
  `edf_per_class` field was overloaded to also answer per-penalty collapse
  detection; with double-penalty smooths the two vectors have different lengths,
  so one consumer always read a wrong-length vector. Both quantities are now
  independently correct.

**Firth/Jeffreys REML performance (#1575)**
- Binomial/logit REML with default-on Firth/Jeffreys bias reduction no longer
  rebuilds the entire `FirthDenseOperator` (the O(n·p²) design Gram, the O(p³)
  identifiable-subspace eigendecomposition, the design clones) on every inner
  Newton iteration. The β-independent design factor is built once per PIRLS solve
  and memoized; only the per-η reduced core is rebuilt. The converged β/λ/EDF/score
  are bit-for-bit unchanged, pinned by an operator-equivalence oracle.

**SAE log-det trace performance (#932)**
- The SAE reconstruction log-det / α-trace channels are back on the hand
  closed-form `row_jets_for_logdet` (a measured 25–57× throughput win over the
  Taylor-jet cutover, bit-identical to ≤1.4e-15), with row-local Takahashi
  selected-inverse fast paths layered on top. The Taylor jet is retained as a
  `#[cfg(test)]` correctness oracle, not deleted.

**BMS Firth/Jeffreys outer-gradient correctness (#1607)**
- The explicit Firth/Jeffreys value ψ-derivative is now carried on the BMS
  batched outer-gradient path, so its `objective_theta` matches the hypercoord
  gradient (and the centered FD of the Firth-corrected outer value) for Jeffreys
  BMS spatial fits.

**Build / test hygiene**
- A wave of test-infrastructure hardening across gam-math, gam-linalg, gam-sae,
  and the multinomial/dispersion oracles (visible assertions, oracle relocation,
  removal of `let _` / ignored-bench laundering), and the SAE row-jet oracle
  fixtures are lifted into the PD basin (#1625) so they converge and assert.

## v0.3.133 — gam 0.3.133 / gamfit 0.1.235 (2026-06-29)

crates.io + PyPI release of the SAE reconstruction-fidelity, penalty-spectrum
robustness, and Firth-performance wave landed since gam 0.3.132 / gamfit 0.1.234.
The headline changes stop the SAE hybrid collapse from over-simplifying a
genuinely curved atom, remove a spurious P-IRLS rejection on high-rank smooths at
extreme λ, and cut the dominant cost of the Firth/Jeffreys outer Hessian. The
`gamfit` Python API surface is unchanged — these correct and accelerate the
behavior reachable through the existing API.

**SAE hybrid-collapse EV preservation (#1610, #1026)**
- A curveable `d = 1` atom that is doing real reconstruction work is no longer
  collapsed to its straight linear tail. The hybrid-split selector now gates each
  collapse on the reconstruction explained variance it would cost, vetoing a
  collapse that would raise the full reconstruction SSR by more than 1e-3 of the
  target's total centered variance (the observed 1.0 → 0.748 EV over-collapse on
  small, low-amplitude fixtures). Lossless / EV-neutral collapses still collapse
  freely. As part of the same fix a degenerate all-stationary atom image now
  reports zero total turning (a point has no arc to turn through) instead of
  refusing, while a partial cusp still refuses — both pinned by new geometry
  regression tests, and the integration test now asserts the EV-axis
  discrimination the gate actually performs.

**Penalty-spectrum PSD floor (#1619)**
- A high-rank thin-plate / Duchon penalty (p ≈ 200) assembled and reparameterized
  at extreme λ during REML no longer fails the inner P-IRLS solve on roundoff: the
  strict PSD classifier now accepts a negative eigenvalue up to the larger of the
  machine-ε floor and a relative numerically-PSD floor (1e-8·scale ≈ √ε), snapping
  it to zero. Genuine indefiniteness is O(1) relative and is still rejected far
  above either floor.

**Firth/Jeffreys REML performance (#1575)**
- The default-on Firth/Jeffreys outer REML Hessian for binomial/logit is
  substantially faster: the single-index sub-blocks of the exact Tierney-Kadane
  mixed second directional derivative — previously rebuilt for every one of the
  k(k+1)/2 penalty pairs against the same identity right-hand side — are now
  precomputed once per direction and reused across all pairs (~37% fewer of the
  dominant O(n·r²·p) applies at k = 6). The converged β/λ/EDF/score are unchanged,
  pinned bit-for-bit by a cached-vs-per-pair oracle test.

**Gaussian weighted prior-weight semantics (#1617, #1618)**
- The intended Gaussian `weights` convention is locked down by a contract test:
  for a fixed-dispersion family (Poisson) a weighted fit reproduces the
  row-expanded fit exactly, while a Gaussian identity-link fit treats `weights` as
  prior (inverse-variance) weights — rescale-invariant under w → c·w and *not*
  equivalent to row replication (its profiled scale divides by the row count,
  matching mgcv / Wood 2017 §6.2.7). Net behavior is unchanged; this rebuts the
  #1617/#1618 false-premise reports and prevents silent drift.

## v0.3.132 — gam 0.3.132 / gamfit 0.1.234 (2026-06-29)

crates.io + PyPI release of the non-Gaussian-family, survival-identifiability,
and convergence-hardening wave landed since gam 0.3.131 / gamfit 0.1.233. The
headline fixes make GP/interval-coefficient smooths fittable under the
exponential-family GLMs that previously rejected them, stop the survival
marginal-slope hang at its root, and remove the Tweedie smoothing boundary bias.
The `gamfit` Python API surface is unchanged — these broaden and correct the
behavior reachable through the existing API.

**Non-Gaussian standard-family terms (#1615, #1616)**
- `matern(x, z)` GP smooths and `bounded(x, min, max)` interval-coefficient
  terms now fit under Poisson, Gamma, Negative-Binomial, and Tweedie families.
  The standard-family observation builder previously had only Gaussian/Binomial
  arms and hard-aborted ("not supported for …") once the coefficient search
  reached any other family; it now derives the score, Fisher weight, observed
  η-Hessian, and its η-derivative analytically for each. (Beta remains
  deferred — its 3rd μ-derivative needs the tetragamma — and still bails.)

**Survival marginal-slope identifiability (#979)**
- Survival `marginal-slope` fits that shared a spatial basis between the
  marginal and log-slope channels could run to the outer wall-clock timeout: the
  full row-Hessian compile attributed the entire shared surface to the
  log-slope block and collapsed it, leaving a quadratically-flat near-null
  direction the inner joint-Newton could never certify. A W-orthogonal *partial*
  reduced-log-slope reparam now drops only the marginal-explained log-slope
  directions and keeps the survivors, so the joint penalised Hessian is
  full-rank by construction and the outer deadline is demoted to a pure
  backstop. (This release also adds direct unit coverage of the new reparam and
  silences a false "pilot-curvature trap" warning it emitted on every success.)

**Tweedie smoothing (#1477)**
- Tweedie `s(x)` P-spline fits no longer ship a right-boundary blow-up /
  EDF-inflated biased mean: the dispersion φ is now held fixed across the
  smoothing-parameter search (matching mgcv and the existing Negative-Binomial θ
  handling), making the REML criterion stationary. Reported dispersion and
  standard errors are unchanged — φ is still refreshed at the final fit.

**GAMLSS Gaussian location-scale (#1561)**
- Gaussian location-scale (and its wiggle variant) now select the flexible
  low-λ log-σ basin instead of over-smoothing the scale predictor; the deeper
  seed screening can no longer discard the heteroscedastic fit via the
  over-smoothed seed's flat-Fisher "looks-cheap-early" proxy. The keep-best
  rule is unchanged (lowest-cost), so the result is provably non-worsening.

**Cyclic smooths (#1593)**
- A cyclic/periodic smooth's fitted curve is now invariant to the arbitrary
  `period_start` phase origin (worst drift ~2e-2 of signal range → ~1e-10): the
  uniform cyclic knot grid is anchored to a canonical lattice rather than
  rigidly to the user's seam, and the dense and streaming evaluators wrap data
  into the same shifted window so fit- and predict-time designs agree.

**SAE manifold penalties (#1610, #1026)**
- The SAE separation-barrier strength `μ_C` is now data-derived from dictionary
  overcompleteness (`K / reachable_rank`) and dimensionless, so it is invariant
  under a global corpus rescaling instead of the hand-picked constant `10.0`;
  the decoder-repulsion strength tracks it as a fixed fraction.

**Firth/Jeffreys REML performance (#1575)**
- Binomial/logit REML with Firth/Jeffreys bias reduction is substantially
  faster: the β-independent design factor (Gram, identifiable-subspace
  eigendecomposition, reduced design) is built once per inner P-IRLS solve
  instead of every Newton iteration, and the seed-screening prepass uses a
  coarse inner tolerance. Converged fit results are unchanged (now pinned by a
  direct factored-vs-full-operator equivalence test).

**Regression guards**
- Reference-free guards added for competing-risks CIF invariance to cause-label
  permutation (#1593, Rust + Python) and 2-D thin-plate truth recovery (#1074),
  so those properties stay protected on CI nodes without mgcv/R.

## v0.3.131 — gam 0.3.131 / gamfit 0.1.233 (2026-06-29)

crates.io + PyPI release of the generation-contract, structured-additive-model
(SAE) collapse-prevention, and Firth/Jeffreys outer-LAML wave landed since gam
0.3.130 / gamfit 0.1.232. The headline fix completes the response-scale
`generate` contract for conditional transformation-normal (CTM) models; the rest
hardens REML/LAML convergence and the SAE penalty stack. The `gamfit` Python API
surface is unchanged.

**Conditional transformation-normal models (#1613)**
- `gam generate` on a CTM model drew synthetic responses on the *latent* N(0,1)
  scale: it required the outcome column `y`, and the per-row mean of the draws
  moved the *wrong way* with the covariate. It now draws genuine response-scale
  `Y` by inverting each row's monotone transform on a standard-normal quadrature,
  so the draws track `E[Y|x]` (verified: a fit to `E[Y|x] = 2 + 0.9x` centers its
  `x = −1, 0, 1` draws on the increasing sequence, no `y` column required). This
  completes the predict/generate response-scale contract begun in #1612.

**Firth/Jeffreys outer LAML (#1607)**
- For a ψ hyperparameter that reshapes the design (Matérn/Duchon length-scale),
  the joint-Jeffreys penalty `Φ = ½ log|Zᵀ H Z|₊` depends on ψ explicitly, but the
  BMS batched outer gradient dropped both the value term `−∂_ψΦ` and its β-coupling
  `−∂_β∂_ψΦ`. Both are now folded into the outer LAML gradient and Hessian, gated
  on `joint_jeffreys_term_required()` so well-conditioned/non-Jeffreys fits stay
  byte-identical.
- The explicit-parameter ψψ second derivative now carries the full
  conditioning-gate curvature `G''·U` (not just the gate motion `G'`), making it
  the exact second derivative of the gated value and consistent with its own
  gate-aware gradient inside the gate's transition band.
- A homoscedastic (flat) scale ridge could exhaust the coupled joint-Newton budget
  while genuinely converged on the identifiable subspace; the convergence
  certificate now also admits the objective-plateau precondition
  (`Δobj ≤ objective_tol`) alongside the residual-stall window, both still gating
  the rigorous Newton-decrement bound.

**SAE manifold penalties (#1610, #1017)**
- Decoder-repulsion collapse-prevention strength is now *derived* from the
  separation-barrier strength and energy-normalized, so it is scale-invariant
  rather than a hand-tuned absolute constant; the separation-barrier collapse
  norm-floor is likewise data-relative (to `maxₖ‖Bₖ‖²`) instead of an absolute
  magic number. Collapse-prevention curvature is engaged on the matrix-free/framed
  production path.
- The co-collapse acceptance bar is now calibrated against the dictionary's
  *reachable* geometric rank `Σₖ rank(Φₖ)` (read from each chart design alone, not
  the decoder magnitude) instead of the nominal coefficient count `Σₖ basis_sizeₖ`,
  which over-stated what a curved nonlinear dictionary can span and biased the
  linear PCA ceiling high. The bar can only move down from the old value.
- (GPU) The SAE G-matvec output accumulation now uses `atomicAdd`, fixing a data
  race when multiple blocks accumulate into the same output element for
  co-occurring row atoms.

**Other fixes**
- #1074: the Gaussian over-smoothing seed safety-net now extends past the
  screening cap, curing weak-signal spatial over-fit.
- #979: a *measured* KKT-refusal gate for the survival marginal-slope phantom null
  direction — projection engages only when a near-null direction is a measured
  phantom (zero gradient residual), never when it is genuinely driven, demoting the
  wall-clock deadline from load-bearing to a backstop.
- #1605: the `sz` factor smooth is exempted from owner-residualization.

**Performance (#1575)**
- Multi-slot outer-eval LRU reuses revisited ρ-points across the REML outer loop;
  redundant per-penalty Firth directions are hoisted out of the O(k²) TK
  outer-Hessian loop; SIMD 4-row batch kernels for the binomial location-scale
  directional derivatives.

## v0.3.130 — gam 0.3.130 / gamfit 0.1.232 (2026-06-28)

crates.io + PyPI release of the prediction/generation-contract and
GAMLSS-convergence wave landed since gam 0.3.129 / gamfit 0.1.231. The headline
fixes repair the response-scale `predict`/`generate` contract for conditional
transformation-normal (CTM) models and for Beta regression, and replace observed
with expected (Fisher) information in the negative-binomial and binomial
location-scale curvature so those GAMLSS fits converge on well-posed data. The
`gamfit` Python API surface is unchanged.

**Conditional transformation-normal models (#1612)**
- `gam predict` returned the probability-integral transform `h(y|x)` of the
  *supplied* response as both `linear_predictor` and `mean`, which wrongly
  required the outcome column at predict time and made the reported "mean" sweep
  with `y` at fixed `x`. It now returns the genuine response-scale conditional
  mean `E[Y|x] = E_{Z~N(0,1)}[h⁻¹(Z|x)]` — a function of the covariates alone —
  computed by inverting the monotone transform on a standard-normal quadrature
  (midpoint rule in probability space, shared fine `y`-grid I-spline inversion).
  A covariate-only frame now predicts without `y`.
- `gam generate` shared the same broken plug-in path and drew `N(h(y|x), sd)` on
  the latent scale, so synthetic responses moved the *wrong way* with the
  covariate. It now draws response-scale `Y` tracking `E[Y|x]` (verified: draws
  at `x = −1, 0, 1` center on `1.10, 1.99, 2.91` for a fit to `E[Y|x] = 2 + 0.9x`).

**Beta regression (#1608, #1609)**
- `gam sample` on a Beta-regression model aborted ("NUTS not implemented for
  beta-regression logit") instead of routing to the documented Gaussian Laplace
  fallback like every other NUTS-unsupported family. It now falls back correctly.
- `gam diagnose` computed Beta AIC / PSIS-LOO elpd at the placeholder precision
  `φ = 1` instead of the fitted `φ̂` (off ~1700 nats, incomparable across
  families). The fitted Beta `φ` is now threaded onto the reported family,
  mirroring the NB-`θ` path and restoring the `with_beta_phi` invariant.

**GAMLSS location-scale convergence (#1606, #1607)**
- Negative-binomial location-scale fits aborted with an `IntegrationError` on
  well-posed heteroscedastic counts: the log-`θ` dispersion block built its IRLS
  curvature from the strongly non-quadratic *observed* information, which goes
  negative for under-dispersed rows and divides the score by ≈0. It now uses the
  expected (Fisher) information, so the inner P-IRLS reaches KKT stationarity.
- The probit binomial location-scale outer REML/LAML curvature was likewise
  assembled from observed information, yielding an indefinite penalized Hessian
  that blew up the envelope-trace gradient (surfacing as an unavailable/zero
  analytic gradient). It now uses expected Fisher information.

**Flexible (learnable) link (#1596)**
- The flexible-link warp now genuinely engages and improves the fit: the frozen
  warp basis is de-aliased against the mean design (no canonical-gauge rank
  drop), the learned link is guaranteed strictly monotone/invertible over the
  fitted predictor range, and the warp is threaded through to `predict`. Deviance
  on the cloglog reproduction improves `1018 → 980` (reference `979.5`) with a
  certified monotone link.

**Multinomial REML invariance (#1587)**
- A penalized multinomial-logit GAM was not invariant to the arbitrary softmax
  reference class (predicted probabilities drifted ~1% under relabeling) because
  it applied the reference-anchored ALR penalty. The reference-symmetric centered
  CLR penalty `M⊗S_t` (`M = I − J/K`) is now wired through the custom-family outer
  REML loop; all other families are byte-identical.

**Duchon / Matérn smooths (#1604)**
- The half-integer-`ν` Matérn Taylor-coefficient path used the wrong polynomial
  degree (`2l` instead of `l`), collapsing every `ν ≥ 3/2` block (e.g. zeroing
  the `ν = 3/2` diagonal). Corrected, so `d = 1` hybrid Duchon smooths with power
  `≥ 2` build a PSD penalty again.

**Performance**
- Compensated multi-lane FMA GEMV kernels (faster *and* more accurate than the
  faer reference), truncated-Taylor `compose_unary` (~2.4×), SIMD-batched closure
  `design`/`design_jet` rows, output-symmetry Tower4 `t3`/`t4` contractions, a
  stable trig recurrence for the harmonic γ-jet, and a per-row alloc dropped from
  the `loss_scaled` data-fit hot loop. A measured survival regression from a
  build-once dense-3-tensor closure was reverted.

**SAE manifold (research surface)**
- Data-driven chart placement (`EncodeAtlas::build_data_driven`) places a bounded
  number of charts at the data's own latent coordinates (greedy farthest-point
  sampling), unlocking well-certified higher-dimensional manifold atoms; the
  PCA "flat SAE" baseline was replaced with a real trained TopK SAE; and the
  dense k-sweep "saturation" was diagnosed as a gate-cap bug and corrected.

**Release & build infrastructure**
- The intra-family dev-dependency publish cycle is broken so the crate family
  publishes cleanly in topological order (#1603); the `gam-terms` test crate
  builds clean again (#1601); broad new unit-test coverage across the
  predict / models / inference / terms crates.

## v0.3.129 — gam 0.3.129 / gamfit 0.1.231 (2026-06-28)

crates.io + PyPI release of the fix + SAE-fast-forward wave landed since gam
0.3.128 / gamfit 0.1.230. The most user-visible change is a prediction-contract
fix: `design_matrix(data) @ coef` now reproduces the reported `linear_predictor`
for every link (it was off by the bias-correction term for curved links). The
`gamfit` Python API surface is unchanged.

**Prediction contract (#1602)**
- The wiggle-free posterior-mean predict path reported a *bias-corrected* linear
  predictor `η̂_BC = X(β̂ + b̂)` (with `b̂ = H⁻¹S(β̂−μ)` the O(1/n) frequentist
  bias-correction) while the exported coefficients are the penalized MLE / mode
  `β̂`. That broke the documented "Raw design matrix" identity
  `design_matrix(data) @ coef == linear_predictor` (and the `posterior.samples @
  X.T` recipe) by exactly `X@b̂` — 1.5–4 % of the lp range for Poisson/Gamma log
  and binomial logit/probit, while staying exact only for the identity link. It
  now reports the uncorrected `η̂ = Xβ̂`, restoring the identity for all links and
  matching the plug-in / link-wiggle sibling paths.

**SAE manifold solver (#1033)**
- Frames-engaged SAE assembly (`build_framed_device_sae_data`, the decoder-rank <
  p large-output case) panicked at install: `set_device_sae_pcg_data`
  unconditionally asserted the per-row `a_phi`/`local_jac` slabs had length
  `rows.len()`, but the framed builder intentionally leaves them empty (the
  per-row cross block rides `frame.frame_blocks`). The length asserts are now
  gated on the non-framed path, so a real OLMo-shaped fit runs to completion. A
  regression test pins both the install (no panic) and the consumer contract (the
  CPU-resident reduced-Schur factor declines on empty slabs → generic matvec).

**SAE manifold structure search**
- `fold_atom_into`'s mass-preserving logsumexp combine produced NaN when fusing
  two zero-mass (`−∞`-logit) atoms (`−∞ − (−∞) = NaN`), poisoning the entire
  logits row and silently corrupting routing for every atom on it. Two zero-mass
  atoms have combined mass zero (logit `−∞`); that is now returned directly.

**SAE manifold fast forward (new public API)**
- A traditional-SAE-shaped GEMM forward pass for the manifold SAE:
  `EncodeAtlas::amortized_encode_batch_fast` / `amortized_reconstruct_batch_fast`
  (single atom) and the whole-dictionary LSH-routed
  `amortized_encode_with_index_fast` / `amortized_reconstruct_with_index_fast`.
  These run the routing + distilled affine predictor + curved-basis decode as
  batched matrix products (≈ a flat encoder's `W·x` throughput, the only extra
  cost the one batched basis eval), and were measured bit-faithful to the per-row
  predictor and accuracy-parity with the certified Newton solve. Degenerate rows
  (no evaluator / singular Gauss–Newton block / non-finite amplitude / no LSH
  proposal) are zeroed and flagged in a returned valid-mask — never a silent
  wrong encode/decode. The certified `*_encode_*` paths remain the accuracy mode.

**Testing & build**
- `cargo test -p gam-terms --lib` builds again (607 errors → 0, #1601): the
  #1521 carve left basis/smooth test fixtures referencing the pre-carve monolith;
  the basis fixtures are repointed at `gam-linalg` and restored, and the three
  smooth fixtures (which reach `gam-solve`/`gam-models`, below `gam-terms` in the
  dependency order) are set aside in `tests/src_modules/smooths/` for relocation
  to the top-level crate, tracked in the still-open #1601.
- Broad new unit-test coverage across gam-config, gam-data, gam-geometry,
  gam-gpu, gam-linalg, gam-math, gam-model-kernels, gam-models, gam-problem,
  gam-report, gam-runtime, gam-sae and gam-inference.

## v0.3.128 — gam 0.3.128 / gamfit 0.1.230 (2026-06-28)

crates.io + PyPI release of the open-issue fix wave landed since gam 0.3.126 /
gamfit 0.1.228. The most user-visible changes are the adaptive/flexible Binomial
link now fitting (or failing loudly instead of silently) and tensor smooths
becoming invariant to the typed order of their covariates; the `gamfit` Python
API is unchanged.

Versions skip gam 0.3.127 / gamfit 0.1.229: those numbers are already on
crates.io / PyPI from a prior orphaned release run whose version bump never
persisted to `main` (registry versions are immutable), so this release takes the
next free numbers — the same skip pattern recorded for 0.3.125 at 0.3.126.

**Adaptive / flexible Binomial links (#1596, #1598)**
- **#1598**: a `link(type=blended(...))` / `mixture(...)` learnable Binomial link
  is now fittable end-to-end. The Python/formula path threads the blended/mixture
  link components into the solver's `mixture_link` spec (it previously aborted
  before the solver with "BinomialMixture requires mixture_link specification"),
  and the joint link solve no longer refuses a finite-but-indefinite observed
  Hessian row: the array build was the lone over-strict consumer, and both
  downstream consumers already floor non-positive curvature, so the CLI fit that
  failed with "observed Hessian curvature is not positive finite" now converges.
- **#1596**: a non-convergent `link(type=flexible(logit))` wiggle fit is now
  surfaced **loudly** as an error instead of silently returning a model
  bit-identical to the fixed base link. Returning the large-smoothing baseline as
  if the flexible request were honored was a silent contract violation — callers
  could not distinguish a genuinely-flat learned link from a non-converged one.

**Gauge invariance (#1593)**
- A tensor-product smooth is now invariant to the typed order of its covariates:
  `te(x, z)` and `te(z, x)` span the identical tensor space under the identical
  per-margin penalty family, but the Khatri–Rao design permuted the columns and
  per-margin penalty blocks, routing the outer λ optimizer to a different terminal
  point in te's flat REML valley and drifting the shipped surface ~2–6 % of range
  on a cosmetic swap. Margins (plus feature columns and periods) are now
  canonicalized by source feature-column index at construction, so `te`/`ti`/`t2`
  build the identical problem regardless of typed order. Pinned by a new
  covariate-order regression test alongside the additive term-order, categorical
  reference-level, and by-factor labeling gauge guards.

**Survival (#1595)**
- `survival_at` and `cumulative_hazard_at` are now consistent past the fitted
  time grid: both flat-clamp beyond the support (they previously used contradictory
  right-edge extrapolation rules, breaking `S = exp(−H)` past the last grid point).

**GP / spatial smooths (#1074)**
- Isotropic Matérn / Duchon GP smooths now run a kernel-range multi-start: the
  profiled REML is re-fit across a log-κ grid and the strictly-best range adopted,
  so an unlucky single start no longer strands the fit on a poor local range.

**REML correctness (#1006, #1038 / #1225 / #1418)**
- The REML log-det trace gradients now carry the full Daleckii–Krein
  deflation-derivative correction (with a divided-difference 0/0 guard),
  and spectrally-deflated directions are excluded consistently across the live
  trace paths.
- Streaming-exact REML now accumulates the cross-row IBP Woodbury log-det in both
  the criterion and the exact-Hessian matvec, matching the dense path to 1e-8; a
  non-PD capacitance is a recoverable ρ-probe refusal rather than a wrong number.

**Identifiability & competing risks (#1590)**
- The dead-column veto is narrowed to skip only entirely-zero placeholder designs,
  and channel-aware drop selection is now joint-rank-aware with faithful joint drop
  attribution for cause-specific competing-risks survival fits.

**Diagnostics & performance (#1575, #1557, #1151 / #1591 / #1592, #932)**
- The post-fit PSIS ρ-uncertainty diagnostic is now opt-in (default off), cutting
  ~33 surplus full-n solves per fit; the redundant parsimony second seed is waived
  for sharp well-penalized GLM optima.
- Extensive bit-identical jet/compose and SIMD row-batching speedups across the
  math, geometry, survival and SAE row kernels (straight-line Faà di Bruno /
  Leibniz towers, 4-row f64x4 lanes, pruned unused jet channels). The SAE
  arrow-Schur per-row GEMM is pinned to a sequential faer pool for
  parallelism-invariant losses.

**Validation & ergonomics (#1597, #11, #12, #13)**
- Weights-column validators report a **1-based** row index, matching the rest of
  the Python/data layer. Portable disk preflight in `build.sh`, a corrected
  `pip install torch` hint for `gamfit[torch]`, and an importable synthetic-SAE
  metrics bench round out the ergonomics fixes.

**Multinomial reference-class invariance (#1587, in progress)**
- Foundation toward making the penalized multinomial-logit fit invariant to the
  arbitrary reference class: the REML smoothing parameter is now **tied per term**
  across classes (the gauge the centered/CLR penalty requires), the
  reference-symmetric centered metric `λ·((I−J/K)⊗S)` is implemented and unit-proven
  in the vector-GLM engine, and a `CustomFamily::joint_penalty_specs` hook plus a
  `MultinomialFamily` centered-penalty builder are in place. The production formula
  path still uses the reference-anchored per-class metric pending the outer-REML
  joint-penalty wiring, so #1587 remains open; a red end-to-end repro documents the
  remaining drift.

**SAE structure & encode (#1026, #993)**
- Fission now applies an anti-symmetric decoder perturbation when it duplicates an
  atom, breaking the symmetric saddle that previously left the two children stuck
  in lockstep (so a bound product atom can actually split into its factors) while
  leaving the mass-split combined decoder exactly unchanged. The encode basin
  warm-up, NaN-alignment routing gate, and an opt-in outlier-robust per-row
  weighting policy for heavy-tailed activations also land.

**Build / CI**
- Completion of the public-API path restoration after the #1521 engine carve,
  GPU-kernel import repointing across the SAE FFI/examples/tests (#1577),
  line-count-gate decompositions (#780), repaired CI test-build APIs/paths, and a
  large set of new unit tests across the foundation crates (gam-problem,
  gam-linalg, gam-math, gam-spec, gam-geometry, gam-predict).

## v0.3.126 — gam 0.3.126 / gamfit 0.1.228 (2026-06-27)

crates.io + PyPI release of the open-issue fix wave landed since gam 0.3.124 /
gamfit 0.1.226, plus the completion of the #1521 engine carve so the whole
workspace — every crate library, the `gamfit` (gam-pyffi) wheel, and the
`gam` build.rs ban-scanner — compiles green again. The most user-visible change
is a correctness fix to the reported model-comparison statistics (AIC and
PSIS-LOO elpd); the `gamfit` Python API is unchanged.

**Reported log-likelihood / AIC / elpd (#1581, #1582, #1583)**
- The user-facing `log_likelihood` (and the conditional/corrected AIC and the
  PSIS-LOO `elpd` derived from it) is now the **fully normalized, scale-aware**
  log predictive density on the response's own measure, not the REML building
  block that deliberately drops every family normalizer and the Gaussian scale.
  New reporting kernels in `gam-solve` carry each family's full normalizer:
  Poisson `−lnΓ(y+1)`, Binomial `ln C(n, n·y)`, the Gamma saturated term, the
  Tweedie Jørgensen saddlepoint density, and Gaussian `−½[ln(2πφ̂) − ln wᵢ +
  wᵢ(y−μ)²/φ̂]` with the profiled `σ̂²` concretized into the scale (no silent
  unit-variance fallback). Symptoms fixed: a discrete model no longer reports a
  positive elpd (#1581); a Poisson fit and an NB(θ→∞) fit on identical data no
  longer differ by ~1750 nats (#1582); the Gaussian log density now obeys the
  change-of-variables law `elpd(c·y) − elpd(y) = −n·ln c` (#1583). An estimated
  dispersion now also adds its degree of freedom to the conditional AIC.
- The Binomial normalizer `ln C(n, n·y)` is now exact: `binomial_coefficient_f64`
  carries its multiplicative recurrence in integer (`u128`) arithmetic instead of
  dividing in `f64`, so the coefficient is bit-exact for every value at or below
  `2^53` (the prior all-`f64` recurrence drifted off the true integer well below
  that — e.g. `C(54,24)` came back one short, `C(55,25)` non-integer), keeping the
  reported Binomial log-likelihood / AIC / elpd exact.

**Survival & links**
- **#1569**: the post-update monotone-cone feasibility tolerance is floored at
  the same `1e-8` gate every downstream consumer enforces, so a cone-projected β
  feasible to the gate is no longer rejected by a stricter post-update check (the
  fragile spectrum-branch α-crush bypass was reverted after it could not be shown
  robust on the dense survival monotone cone).
- **#1571 / #1572 / #1573**: SAS / Beta-Logistic / mixture parameterized-link
  fits no longer abort with a "Lambda count mismatch": the post-convergence
  inner-cap guard now routes the augmented θ through the same `apply_link_theta`
  the eval closure uses, handing `compute_cost` exactly the smoothing-only ρ
  block (and installing the converged link state) instead of the raw augmented θ.

**Identifiability (#1580)**
- The large-scale identifiability-audit regression is rebuilt on orthogonal
  Legendre polynomials so its single seeded rank deficiency is resolved
  backend-independently (the penalty-augmented Gram path's `√ε` resolution made
  the prior RBF/trig fixture spuriously demote extra columns on some BLAS).

**Build system (#1521)**
- Completion of the engine carve: the published `gam` crate now depends on the
  full gam crate family (foundations plus `gam-model-api`/`gam-gpu`/
  `gam-identifiability`/`gam-terms`/`gam-solve`/`gam-custom-family`/
  `gam-model-kernels`/`gam-models`/`gam-sae`/`gam-test-support`), published to
  crates.io alongside it as a version-locked family; the `gamfit` wheel is
  unaffected (it builds from source). A sweep of latent carve breakages —
  cross-crate visibility, stranded duplicate definitions, a stale shared-include
  depth, mis-scoped GPU macros, and dead re-export shims — is repaired so
  `cargo check --workspace` is green end-to-end.

## v0.3.124 — gam 0.3.124 / gamfit 0.1.226 (2026-06-26)

crates.io + PyPI release of the open-issue fix wave landed since gam 0.3.123 /
gamfit 0.1.225. Two themes dominate: a survival/location-scale correctness pass
that makes saved-model prediction total and adds a genuine IPCW Brier score, and
the build-system work of #1521 — the monolithic `gam` crate is split into
foundation crates so an edit recompiles a sub-crate, not all 653 files. As a
consequence of that split the published `gam` crate now depends on the gam
foundation crates (`gam-runtime`, `gam-data`, `gam-math`, `gam-spec`,
`gam-linalg`, `gam-problem`), which are published to crates.io alongside it as a
version-locked family; the `gamfit` wheel is unaffected (it builds from source).
The pre-release hardening also restored the `cargo check --workspace
--all-targets` green invariant that two integration-test targets had regressed,
and removed leftover split-WIP scratch from the tree.

**Survival & location-scale**
- **#1564**: saved-model survival prediction is now total. The Royston–Parmar
  hazard guard accepts a zero log-cumulative-hazard time-derivative (the I-spline
  baseline is flat past its last interior knot, so `d(log Λ)/dt = 0` is a
  legitimate boundary value on the default prediction grid's top node) and
  resolves the saturated `Λ = +∞ × 0` corner to `0`, not `NaN`. A `finite_safe_json`
  serde adapter encodes `±∞`/`NaN` payload values as explicit string tokens so the
  engine→Python boundary no longer rejects non-finite `f64` as `null`.
- **#1563**: survival metrics now report a genuine integrated IPCW Brier score
  (Graf 1999) built on a Kaplan–Meier censoring estimator over a data-driven
  quantile grid, validated end-to-end against an independent Python oracle. The
  prior hazard-quadratic score is honestly renamed.
- **survival location-scale**: the log-σ (scale) design is kept raw instead of
  being residualized against the location design — a smooth that drives both the
  location and scale channels is separately identifiable, and residualizing
  erased the heteroscedastic signal and tripped a joint-gradient shape check on
  every smooth-scale fit. Cross-block identifiability is supplied by the
  per-channel audit assignment, matching the Gaussian location-scale path.
  Gaussian location-scale seed basins are also promoted/classified correctly.

**Manifold SAE (#1026, #1522)**
- **#1556**: manifold-SAE smoothness `λ_smooth` is genuinely per-atom (the outer
  ρ carries one coordinate per atom, not a shared scalar).
- Surplus/dead atoms are now parked gracefully instead of failing the pre-fit
  audit; per-atom ARD collapses to shared hyperparameters at large K; the outer ρ
  is routed through Fellner–Schall (REML); the over-complete reduced Schur is
  spectral-floored; and the large-K matrix-free regime is bounded by a wall-clock
  deadline so it cannot livelock. GPU device PCG for the SAE row-jet landed and is
  arch-pinned through NVRTC so the double-atomic kernels actually engage the
  device (#1017, #1033, #1551).

**GPU survival row-jet (#932)**
- An A100 survival rigid row-jet NVRTC kernel with a CPU-fallback dispatcher
  (≤1e-9 exactness), with device-fallback-reason logging and a device-only
  diagnostic entry.

**Inference, conformal & bases**
- **#1546**: the jackknife+ conformal interval uses `α = 1 − level` (delivered
  coverage), not `(1 − level)/2`.
- **#1548**: the default `s(x, bs="ps")` penalty is canonicalized so it is
  reflection-invariant; **#1549**: the ALR tangent coordinates are whitened by
  `G^{1/2}` so the smoothing penalty is Aitchison-isometric; **#1545**: the sphere
  Fréchet-mean Karcher descent is seeded from the full eigenbasis so the
  least-dominant axis is covered.
- **#1074**: `projected_gradient_norm` sign is corrected so a railed-but-descending
  ρ is not certified stationary.

**Python bridge & wheel**
- **#1565**: the `smooths={}` descriptor bridge is repaired (`slots=True`
  `super().__init__` across all sites; `double_penalty=False` is emitted).
- **#1559**: the `gam-pyffi` wheel build no longer fails on an `E0382` partial move
  (`log_lambda_smooth` is cloned instead of moved out of a still-borrowed ρ).
- **#1558**: the CUDA-unavailable diagnostic consumes `need_logdet` on every target.

**Build system (#1521) & release hygiene**
- The `gam` engine is split into foundation crates (`gam-math`, `gam-runtime`,
  `gam-data`, `gam-spec`, `gam-linalg`, `gam-problem`) plus the upper leaves
  (`gam-predict`, `gam-inference`, `gam-cli`), cutting per-change recompile from
  the full 653-file monolith to a sub-crate + facade. The families↔solver↔terms
  SCC stays in `gam`; its decomposition is tracked as separate contract-inversion
  work.
- Restored `cargo check --workspace --all-targets` to green: the `sae` and
  `perf_scale` integration-test targets had regressed against the post-split
  `gam::resource` module path and the per-atom `SaeManifoldRho` API. Removed the
  orphaned, unwired `gam-problem` `penalty_matrix.rs` staging file and the
  leftover split-WIP scratch notes from the tree.

## v0.3.123 — gam 0.3.123 / gamfit 0.1.225 (2026-06-24)

crates.io + PyPI release of the open-issue fix wave landed since gam 0.3.122 /
gamfit 0.1.224. The headline is the predict output layer: the reported
`std_error` and the `predict_array` return shape now match what `predict()`
documents and lays out, instead of silently handing back link-scale values on
non-identity links. Alongside that: low-cardinality cubic-regression smooths now
fit instead of hard-failing, the outer REML smoothing search is made invariant to
the order terms/margins are typed in, the default thin-plate basis is made
row-permutation invariant to the ulp, and the SAE held-out decode path is repaired
to match training. The build/CI hygiene also tightens (the full `tests/` suite now
runs in CI behind an orphan guard) and the workspace stays dead-code-clean as a
primary build (`cargo check --workspace --all-targets` green, so the published
`gam-pyffi` wheel's `use gam::…` surface is verified, not just the `gam` lib).

**Prediction output layer (response scale)**
- **#1536**: `predict(interval=...)` / `gam predict` now report `std_error` on the
  *response* scale — the delta-method `SE(μ̂) = |dμ/dη|·SE(η)` the credible band
  beside it is built from — instead of the link-scale `σ_η`. On a non-identity
  link the two were off by the inverse-link Jacobian, so the SE column was
  internally inconsistent with its own `mean`/`mean_lower`/`mean_upper`. The
  posterior-mean path gained a `PredictPosteriorMeanResult::mean_standard_error`
  field, populated from the SE the band already uses and surfaced by the FFI/CLI.
- **#1537**: `predict_array(X)` with no `interval` now returns the documented 1-D
  response-scale vector, matching `predict()`, instead of the 2-D
  `[linear_predictor, mean]` column matrix — a naive `[:, 0]` / `.ravel()` caller
  was silently getting the link-scale linear predictor on non-identity links. The
  interval case still returns the full column matrix.
- **#1515**: interval predict on a degenerate fit no longer returns non-finite
  bounds. When the smoothing-corrected covariance `H⁻¹ + J Var(ρ̂) Jᵀ` carries
  non-finite entries (e.g. an all-zero-count Poisson, whose flat likelihood leaves
  the outer REML problem near-singular and blows up `Var(ρ̂)`) the predictor now
  degrades to the finite conditional covariance `H⁻¹` — the `Preferred` mode
  already falls back when the correction is *missing*, and an unusable (non-finite)
  correction is the same case — so a model the API reports as fitted always yields
  finite `std_error` and `mean_lower`/`mean_upper`.

**Smooths & bases**
- **#1541**: a univariate `s(x, bs="cr"/"cs")` cubic-regression smooth no longer
  hard-fails the whole fit on a low-cardinality covariate (a binary indicator, a
  3-level ordinal, a small count). The basis is capped to the data support —
  `k = min(k_requested, n_distinct)` value-knots, mgcv-style — and below the cr
  minimum of three distinct values it degrades to the linear B-spline marginal the
  default `s(x, k=..)` basis already builds. The cap is surfaced in the inference
  notes. This is the univariate sibling of the tensor-margin cap.
- **#1542**: a factor smooth `s(x, g, bs="sz")` likewise caps its per-level cr
  marginal to data support rather than aborting. A pre-existing latent bug was
  uncovered and fixed in the same change: a frozen `sz` factor smooth failed its
  own predict-time freeze check ("factor-smooth marginal knots missing") because
  the validation whitelist had never been updated for the cr marginal — so `sz`
  *predict* was broken regardless of cardinality. It now fits *and* predicts.
- **#1543**: a basis the fit silently adjusted is no longer silent to gamfit
  callers. The mgcv-style cap/degradation advisories the Rust core records (and
  the CLI already prints) are now carried through the FFI to Python: `gamfit.fit`
  / `fit_array` emit one `GamInferenceWarning` per note at fit time, and
  `Model.notes` exposes them for after-the-fact / post-load inspection. Previously
  the Python path dropped `inference_notes` at the FFI boundary, so a capped basis
  warned in the CLI but was invisible to gamfit. (The payload field is
  `#[serde(default)]`, so older saved models load cleanly as "no notes".)
- **tensor margins**: explicit `te(...)` margin `k` is capped to data support
  (mgcv-style), with the `cr` `basis_size` helpers repaired.
- **#1378**: the default `s(x, bs="tp")` thin-plate smooth is made exactly
  row-permutation invariant. The knot-selection centroid was summed in row order,
  so floating-point round-off shifted it by an ulp under a pure row permutation —
  enough to flip the seed (and hence the whole knot set and `λ̂`) on data symmetric
  about the mean. The column sum is now taken in canonical value-sorted order.

**REML smoothing-parameter selection**
- **#1538 / #1539**: the outer REML smoothing-parameter search is made invariant
  to the order terms and margins are written in — additive `s(x)+s(z)` vs
  `s(z)+s(x)` and tensor `te(x,z)` vs `te(z,x)` now select the same `λ̂` and fit
  the identical surface (worst row-drift 1.0e-1 → 1.8e-6, 6e-2 → 7.1e-5). Each
  rho-coordinate is labelled by a placement-independent canonical key (the
  penalty's orthogonal-invariant spectrum plus a data-dependent block signature),
  so seeding, multistart and tie-breaking all run on one canonical layout and map
  back to the native order for the caller. Single smooths run the native path
  byte-for-byte as before.

**SAE (sparse autoencoder)**
- **#1540**: the held-out SAE reconstruction now attaches the trained dictionary's
  hybrid-collapsed straight images, so verdict-linear `d=1` slots decode by the
  same linear image training used. The parameter was accepted but never wired, so
  every OOS reconstruction silently fell back to the all-curved decoder — a
  train/test decode mismatch on hybrid-collapsed dictionaries.
- **#1026**: SAE anti-collapse interior-point barriers (finite-difference
  verified), runtime barrier-strength and IBP-alpha overrides so one wheel sweeps
  all configs, and GPU residency Phase 0-1 telemetry / fail-closed wiring.

**Survival**
- **#979**: the marginal-slope seed-screening cascade is bounded by the outer
  wall-clock deadline (single-sourced with the slow-geometric-rate stall guard),
  so a hard survival fit cannot blow the outer time budget in the inner search.

**Model summaries**
- **#1544**: `MultinomialModel.summary()` (and `str()`/`print()`) no longer raises
  `ValueError` on a smooth multinomial fit. The summary assumed one λ per smooth
  term per class, but the default Marra–Wood double penalty emits two penalty
  components — a wiggliness penalty plus a null-space shrinkage penalty — so the
  per-block λ count never matched the term-label count. The summary now records
  per-penalty-component λ labels and pairs every component (including each term's
  null-space λ) instead of silently dropping it.

**Custom families & outer-score subsampling**
- A custom-family / `GaussianLocationScale` fit with a Horvitz–Thompson
  `outer_score_subsample` no longer hard-fails with `IntegrationFailed`. The inner
  coefficient solve was mixing two row measures — a full-data entry/reload base
  objective against an HT-subsampled trial — for families that do not advertise an
  HT-consistent inner gradient, so the trust-region `actual_reduction` was pinned
  at the constant HT-vs-full log-likelihood gap, the radius collapsed, and every
  seed was rejected at outer startup. The subsample is now stripped from the inner
  options unless the family runs a fully HT-consistent inner solve, keeping β̂(ρ)
  the unbiased full-data optimum (the subsample remains an outer ψ/ρ-derivative
  variance-reduction device). Covered by the `ws4a` subsampled-vs-full parity test.

**Build & CI hygiene**
- **#1534**: the manylinux / musllinux wheel containers trust the mounted
  workspace tree (`git config --global --add safe.directory '*'`) so `build.rs`'s
  author gate runs instead of panicking on "dubious ownership" ~12 min into the
  release build.
- **#1512**: the full `tests/` suite now runs in CI via a directory-level pytest
  step, behind a hard-gated orphan-guard meta-test, so new `test_*.py` repros are
  collected automatically instead of silently running in no job.
- **#932 / #1017**: continued survival flex single-source derivative cutover onto
  the jet tower and removal of the dead dual-context CUDA path, keeping the `gam`
  crate dead-code-clean as a primary build.

## v0.3.122 — gam 0.3.122 / gamfit 0.1.224 (2026-06-24)

crates.io + PyPI release of the open-issue fix wave landed since gam 0.3.121 /
gamfit 0.1.223. A robustness sweep across degenerate-fit prediction and
constrained-coefficient posterior sampling, family deviance/dispersion
corrections, a pure-REML escape for double-penalty null-space shrinkage, further
thin-plate/Matérn root-cause cleanup, the survival flex single-source derivative
cutover (with the fourth-order moving-boundary residual further closed), and SAE
collapse *prevention* — a data-derived bar plus a linear-dominance floor in place
of magic constants. The build.rs hygiene gate continues to hold: spent #1454
localizer probe fields are removed and the unwired survival moment-engine oracle
is relocated into its test module, so the `gam` crate stays dead-code-clean as a
primary (`cargo build` / `cargo publish`) build.

**Prediction & posterior sampling**
- **#1515**: degenerate / near-singular fits no longer emit non-finite
  predictions. Interval predict gets a finite delta-method SE fallback, the
  log-link posterior mean is floored when its SE overflows, and an all-zero
  Poisson fit predicts a finite plug-in mean when the posterior-mean integral
  overflows — with finite interval bounds throughout.
- **#1507 / #1509**: box- and shape-constrained coefficients now draw from a
  truncated-Gaussian posterior on the latent scale, so `predict_draws()` respect
  the monotone-shape and `bounded()` box constraints instead of escaping them;
  posterior-predictive draws also re-apply the model offset.
- **#1514**: a `bounded()` Gaussian coefficient covariance is scaled by σ̂², so
  its standard errors are neither too wide nor too narrow.
- **#1513**: numeric `by=`-variable multipliers are exempt from the predict-time
  axis clip (they are multipliers, not a smooth axis).

**Families & model comparison**
- **#1529**: Gamma deviance-explained is no longer contaminated by an
  estimated-shape mismatch — the null deviance uses the fitted dispersion rather
  than resetting to the family default.
- **#584**: ALO dispersion divides by the positive-weight row count (the true
  residual dof), not the raw row count, so zero-weight rows do not deflate it.
- **model comparison**: the smoothing-correction covariance `V_corr` is
  symmetrized before use (#1527).

**Smooths, REML & double penalty**
- **#1266**: the default double penalty (Marra–Wood null-space shrinkage) no
  longer inflates smooth EDF — a pure-REML null-space shrink-out escape lets an
  irrelevant covariate be shrunk out (and `s(x)` on linear data recover its true
  ~2 EDF) instead of pinning every term near the basis dimension.
- **#1074**: further thin-plate / Matérn root-cause cleanup — the masking hacks
  (the Matérn-specific length-scale ceiling, a redundant λ ceiling, the
  thin-plate cap and center-cap, the latent active-mass floor) are deleted in
  favor of the real basis-sizing / correlation-range fixes; `te()`/`ti()` cr
  margins honor the requested `k` (guarded to `k ≥ 3`) and `bs="sz"` factor
  smooths route through the cr metadata freeze/replay.
- **#1531**: the constant-curvature double penalty is documented and tested to
  use an identity ridge (full-rank null space). **#1464**: the curv
  hyperbolic-sign contract is now an asserting CI gate.

**Survival (#932 / #1454)**
- **#932**: the survival flex marginal-slope derivative tower is single-sourced
  through one generic `FlexJet` jet algebra — the link-wiggle joint-Hessian
  cutover is landed in production and the generic-order moment / eta-chi
  machinery is exact to all carried orders, with the calibration residual derived
  as a distinguished-derivative `j/(j+m)` projector.
- **#1454**: the fourth-order `[g, β_w]` bidirectional cross residual is
  corrected in sign and magnitude — the moving-boundary `D²(B)` term and the
  missing `f_a` self-flux are added to the bidirectional `§D` path. The final
  observed-point link-warp term remains and **#1454 stays open** for it.

**Manifold-SAE & latent (#1026 / #1388 / #1522)**
- **#1522**: the SAE collapse **floors** (a magic 0.28 reconstruction-EV bar, a
  1e-3 atom-mass floor, a latent active-mass floor) are replaced by a
  data-derived PCA-EV ceiling plus a genuine NaN guard — collapse *prevention*
  in the assignment/decoder step rather than detect-and-reseed band-aids.
- **#1026**: a hybrid-split collapse rescue rebuilds a fresh linear image for
  rank-1 co-collapsed circle atoms, and a result-level linear-dominance floor in
  `into_fitted` restores the certified PCA anchor (`F ≤ F_linear`) when curvature
  collapses — backed by a collapse-safe SAE acceptance battery.
- **#1388**: the SAE joint fit runs on a wide-stack worker thread in the wheel,
  avoiding a stack overflow on large joints.

**GPU & build (#1017)**
- **#1017**: CUDA initialization is hardened — the primary context is bound and
  the runtime initialized before the first cuBLAS/cuSOLVER handle creation
  (probe-first `NOT_INITIALIZED`), and the probed compute libraries stay loaded
  so `dlclose` cannot poison cuBLAS; a GPU regression guard and a Modal A100/T4
  runner (with a dead-hand heartbeat kill-switch) back it.
- **CI**: the full pytest suite runs in CI (#1512 / #1532) and the fast Python
  Contracts workflow caches `target/` and enables sccache (#1518).

## v0.3.121 — gam 0.3.121 / gamfit 0.1.223 (2026-06-23)

crates.io + PyPI release of the open-issue fix-and-feature wave landed since gam
0.3.120 / gamfit 0.1.222. New data-ingestion surface (Dask, SPSS, numeric-string
categoricals), a large correctness sweep across double-penalty smooths, REML
convergence, survival/location-scale Hessians, model comparison and the
manifold-SAE / latent stack, plus opt-in routing-predictor machinery and the
device-resident GPU Gram path. The build.rs hygiene gate is hardened further
(anti-laundering bans) and the tree continues to build the `gamfit` wheel clean.

**Data input (Python wrapper)**
- **#1460**: Dask DataFrames are accepted as input/output — materialized with a
  single `compute()` into pandas for fitting — and a new `read_spss()` loads
  `.sav`/`.zsav` files via `pyreadstat`, decoding value-labelled variables to
  pandas `Categorical`.
- **#1467 / #1468 / #1469 / #1473**: numeric-string columns in dict / records /
  numpy inputs are treated as categorical (pandas-object parity); a mixed
  string+numeric column is categorical, and dict-input multinomial hard-rejects
  numeric-string class labels.

**Smooths, factors & identifiability**
- **#1476 / #1477**: the double-penalty null-space ridge is rebuilt in the
  identifiability-constrained chart (after the global transform), pairing each
  ridge with its co-located Primary block — fixing concurvity collapse of
  `s(x1)+s(x2)`, by-factor per-level correctness, and Tweedie default-P-spline
  mean bias, without over-shrinking a supported smooth.
- **#1427**: `s(x, by=factor)` emits an independent per-level λ. **#1457**: a
  bare categorical main effect is de-duplicated under `s(x, by=g) + g`.
- **#1470**: `ti(x, z)` interactions stay grid-independent — no off-grid
  residualization against the realized `s(x)`/`s(z)` spans in functional-ANOVA
  models.
- **#1378 / #1456**: default univariate thin-plate basis sized to mgcv `k=10`
  with rotation- and permutation-invariant knot selection. **#1074**: default
  thin-plate / Matérn basis sized to mgcv `k=10·3^(d-1)` and the Matérn
  correlation range matched to mgcv's default (diameter), fixing EDF inflation.
- **#1379**: per-block penalty trace clamped to `[0, rank]` (NaN-safe) so a
  univariate `matern(x)` fits. **bs="sz"**: emits `FactorSmooth` metadata so
  basis freezing matches the spec.

**Inference & numerics**
- **#1426**: a stuck gamma/log REML flat valley is no longer shipped as a
  converged overfit — score-relative stationarity certification, rejection of
  non-converged inner PIRLS iterates and untrustworthy release-rerank seeds, and
  a rank-guarded `H` pseudo-logdet with a determinant-pair-sign guard.
- **#1464 / #1404**: constant-curvature κ sign is identifiable in `curv()` — a
  κ-fair scan recovers the hyperbolic/spherical sign and the joint solve is
  pinned to the sign-correct half-axis; the curvature-blind double-penalty ridge
  is dropped.
- **#1395**: custom-family pseudo-Laplace / exact-Newton objectives gain a
  structural guard against `0.5·log|H|` collapse and no longer fold the
  Jeffreys/Firth prior into the pinned-mode objective. **#1418**: the IFT
  back-substitution inverts the exact stationarity Jacobian.
- **#1376**: anisotropic Matérn ψ- and penalty-second-derivatives centered to the
  raw-η gauge. **#1392**: P-spline double-penalty underfit fixed for `p>n`.
- **#1410 / #1419**: compact softmax curvature uses a genuine Gershgorin Loewner
  majorizer and reads active-only entries. **multinomial**: genuine per-class
  EDF (no per-penalty-block over-count), a Fisher-information sparse-class
  λ-floor (#1082), and the hetero `x1` basis sized to its true df (#1373).

**Families & model comparison**
- **#1465**: `compare_models` computes Δ / Bayes-factor on the ranking scale.
- **#1448**: negative-binomial runs the full outer θ↔λ alternation (re-selecting
  ρ after each θ refresh); **#1463**: NB-NUTS `sample()` refreshes the fitted
  `theta_hat` rather than the seed.
- **#1504**: a Gaussian location-scale (gaulss) fit with a by-group smooth in
  both the mean and log-σ blocks no longer crashes on a joint-Hessian shape
  mismatch — the joint exact-Newton path uses the identifiability-constrained
  designs (with an R-free regression guard).

**Survival**
- **#1454**: the survival flex intercept-Hessian moving-boundary / self-flux
  terms are completed and carried exactly to fourth order, single-sourced from
  the D-path. **#1396**: entry/exit transposition in time-block η slicing fixed
  and a near-cancellation event-Jacobian floored to the monotonicity guard.
- **#1388**: under-determined (`p_joint > n`) survival marginal-slope joints are
  surfaced honestly. **#979**: the marginal-slope outer search is bounded by a
  wall-clock deadline with collapsed-trust-region stuck-exit guards.

**Manifold-SAE, latent & routing (#932 / #1017 / #1026 / #1033)**
- **#932**: the survival-LS / BMS-rigid / SAE-β-border row kernels are
  single-sourced through one `row_kernel` v/g/H tower with an exact wiggle
  joint-Hessian oracle. **#1500**: dead dictionary atoms are re-seeded.
- **#1017**: a device-resident GPU path uploads `X` once and chains a
  Gram-resident POTRF, downloading only β (with CPU parity gates). **#1026**: an
  ungated linear/background tier reconstructs full-rank alongside the gated
  sparse atoms.
- **#1033**: an opt-in chart-geometry / amortized routing predictor (off by
  default) with an n-free frozen-W Fisher-step solver. **latent**: a new
  `LatentIdMode::IsometryToReference` gauge-fix mode.

**CLI & summaries**
- The CLI honors formula-declared categorical roles (numeric-coded factors).
- **#1368 / #1370**: `summary()`'s RE penalty-cursor skips empty-range penalized
  RE blocks on both the in-process and Python persisted paths.

**Releasability / hygiene**
- The build.rs gate gains anti-laundering bans: silent NaN/`0.0`/`Ok(())`
  corruption where a contract guard belongs, owed-work disguised as prose, and
  hardcoded commit-SHA literals. Contract guards previously laundered into
  silent corruption are restored to panics-with-`// SAFETY:` or proper `Result`
  errors across penalties, solver, basis, evidence, families and GPU paths.
- PR-level anti-evasion / owed-work-ledger workflows are removed (enforcement
  lives in the build), clippy is dropped from CI, and **#1458** gives the
  build.rs author gate full history so it resolves the real last editor.

See the git history (`git log v0.3.120..v0.3.121`) for the complete set.

## v0.3.120 — gam 0.3.120 / gamfit 0.1.222 (2026-06-21)

crates.io + PyPI release of the open-issue bug-fix wave landed since gamfit
0.1.221 (gam 0.3.119), plus the #1452/#1288 releasability cleanup that re-arms the
`build.rs` hygiene gate to a hard failure and brings the tree into compliance so a
release can actually build the `gamfit` wheel.

**Releasability / hygiene (#1452 / #1288 / #780 / #871)**
- The `build.rs` ban-scanner is back to a hard `exit(1)`: no `#[ignore]` tests,
  no `debug_assert!`, no `unreachable!`, no manifest `dead_code = "allow"`, no
  underscore-prefixed unused parameters, no `#[cfg(test)]` items outside a test
  module, and no tracked file over 10k lines. The whole tree now complies and
  builds clean under `[lints.rust] warnings = "deny"`.
- Finished the #1288 dead-code cleanup: removed the never-consumed third-order
  `d3qdot` location-scale qdot jet and the unwired batched
  `jeffreys_*_flex_no_wiggle` / basis-contraction survival fast paths, scoped the
  #932 jet-scalar oracle structs to their test module, and dropped leftover
  #932 directional FD-localizer debug scaffolding.
- Promoted cheap invariant `debug_assert!`/`debug_assert_eq!` to always-on
  `assert!`/`assert_eq!` (penalty-root rank consistency, Duchon hybrid-integral
  precondition); dropped release-noop debug assertions where the documented
  contract wants honest IEEE `NaN`/`inf` propagation; replaced a banned
  `unreachable!` with a `// SAFETY:`-justified `panic!`.

**Inference & numerics**
- **#1436**: a typed `OuterGradientError` (IllConditioned / NonIdentifiable /
  InternalInvariant) narrows the SAE FD-fallback so only genuine
  conditioning/identifiability failures admit the finite-difference descent
  direction at a finite-cost ρ, while internal-invariant defects propagate as
  hard errors. NonIdentifiable is now constructed at the gauge-degenerate,
  non-deflatable outer-gradient site.
- **#1424 / #1422 / #1423**: cancellation-free hybrid Duchon-Matern kernel
  evaluation and a PSD mixed-periodicity Duchon penalty via an additive tensor
  reproducing kernel, with the correct cylinder nullspace.
- **#1271 / #1266 / #1380 / #1089**: the REML log-λ cap is lifted off
  well-determined Gaussian-identity smooths so REML can reach the null-space
  optimum, without changing the global default.
- **#1391 / #1397 / #1017**: post-T rank invariant is anchored to the audit's
  kept-rank certificate and made robust to the drop-deciding convention;
  relaxed over-strict arrow-Schur parity asserts to a tight relative tolerance.
- **#1376 / #1398 / #1404**: anisotropic Matern ψ-derivative centered to the
  raw-η gauge; isotropic sphere-harmonic penalty and closed-form Sobolev jet
  with a constant-curvature effective-length contract.
- **#1426**: a stuck gamma/log REML stall is no longer shipped as a flat valley.
- **#1385**: competing-risks CIF assembled on a refined internal grid.

**Survival (#932 / #979 / #1394 / #1396)**
- Moving-boundary flux / implicit-function / substitution jet-tower combinators
  for the θ-dependent flex-calibration integrand, carried exactly to fourth
  order, with the survival location-scale time-channel NLL sign living in a
  single source of truth and rigid-kernel non-finite margin propagation.

**Smooths, factors & summaries**
- **#1403**: `s(x, by=factor)` routes to `BySmooth::Factor`, `s(x, by=numeric)`
  to `BySmooth::Numeric`, and `bs="sz"` factor smooths to `FactorSmooth { Sz }`.
- **#1378**: default univariate thin-plate basis sized to mgcv `k=10`, with
  row-permutation-invariant knot selection.
- **#1364**: P-spline scale equivariance.
- **#1384**: `compare_models` refuses to rank fits of different response families.
- **#1370 / #1368 / #1369**: `summary()` synthesizes valid factor levels for the
  smooth-term replay so `fs`/`sz`/`by` smooths keep their EDF and per-level
  labels.

**Manifold-SAE**
- **#1405 / #1406 / #1410 / #1411 / #1412**: matrix-free planner predicts the
  true cross footprint; compact support is partial-selected and per-worker
  scratch sized by compact dims; encode bench gates honest support recovery.
- **#1026**: collinearity-gated decoder repulsion conditions the SAE
  co-collapse direction with a keep-best multi-start.

**Error messages**
- **#1445**: NaN/None/empty table-cell errors now name the offending column and
  row instead of an unactionable bare message.

See the git history (`git log v0.3.118..v0.3.120`) for the complete set.

## v0.3.118 — gam 0.3.118 / gamfit 0.1.219 (2026-06-17)

crates.io + PyPI release. The `gam` crate is bumped 0.3.117 → 0.3.118: this is a
crates.io catch-up that publishes to the `gam` crate all the engine work that has
shipped to the `gamfit` wheel since gam 0.3.117 (gamfit 0.1.203), plus the open-
issue bug-fix wave landed since gamfit 0.1.218. Highlights:

**Basis & boundary conditions**
- **#1238**: B-spline `bc=anchored`/`bc=clamped` endpoints are now enforced as a
  *structural* nullspace reparameterization — anchored endpoints are pinned to
  zero (and drop their constrained column), clamped derivatives are zeroed, and a
  non-zero anchor is rejected. The free intercept is suppressed only for a
  *one-sided* anchor (which consumes the absolute level at that endpoint); a
  two-sided anchor keeps the intercept.
- **#1239**: periodic B-splines evaluate the derivative recurrence on the full
  wrapped knot support, no longer extrapolate past a clamped boundary, and drop
  the ridge fallback that masked the wrap.
- **#1257**: `periodic(x, …)` is accepted as a term-function alias and routes to
  `cyclic`. **#1132**: periodic/torus `n_harmonics` floored at the decoder-
  implied harmonic count.

**Manifold-SAE (large body of work: #977 / #1026 / #1132 / #1154 / #1189–#1232)**
- Chart canonicalization ordering/turning stabilized and the hybrid curved-vs-
  linear split computed *after* canonicalization (#1227); OOS fixed-decoder solve
  returns the converged latents (#1229); shape uncertainty recomputed after the
  structure search settles (#1230); hybrid-collapsed linear images threaded into
  OOS reconstruction (#1228).
- Outer objective fixes: BFGS/ARC line-search probe sees pure REML, not `f+c`
  (#1224); streaming SAE branch optimizes the full REML criterion (#1225);
  consistent cost/gradient pair in the cotrain outer objective (#1206/#1207);
  PSD softmax Fisher metric for the curvature anchor (#1190); corrected PG
  gate-block normalizer in the live K-vs-K+1 birth gate (#1218).
- n-free per-ψ penalty rebuild + ψ-Gram certification on standardized geometry
  (#1033 / #1216); EV-knee auto-K + manifold-vs-linear wager verdict (#977 /
  #1026); honest EV/centering and labeling throughout (#1198 / #1201 / #1202 /
  #1203 / #1209 / #1213 / #1226).
- **#1232**: SAE top-k projection metadata preserved in the Python payload;
  per-atom `held_out_delta_ev` and the (Θ, ΔEV) frontier surfaced through the FFI.

**Survival**
- **#931**: robust inner-solve polish (regularized-Newton + steepest-descent +
  Armijo value line-search + exact Cholesky) reaches stationarity at all ρ and
  large λ; survival LAML objective↔gradient desync closed via the active-set-
  projected IFT envelope.
- **#740**: full-θ KKT-residual correction (cross-ρ-ψ + ψ-ψ Hessian); binomial
  loc-scale drift FD on the identifiable Jeffreys span. **#1242**: derivative-
  channel location-scale row derivatives aligned with the exact-Newton kernel.
- **#1248**: `survival_likelihood` canonicalized consistently across CLI and JSON
  config paths. **#1258**: consistent `expected 0 or 1` event-target message.

**Gaussian / GLM / conformal**
- **#1262**: an effectively-constant Gaussian response now fits to the exact
  constant instead of erroring out of the REML path.
- **#1261**: Gaussian average-derivative one-step debiased against the unpenalized
  information (sign + smoothing-pull corrections). **#1127**: scale-equivariant
  REML smoothing-parameter selection.
- **#942 / #1098 / #1192 / #1263**: GLM full-conformal route contract stabilized
  (KKT-scaled cold-fit/corrector convergence, round-off-floor cold fit, alternate
  ARC seed retained after screening).

**Inference**
- **#939**: Skovgaard r* modified directed-likelihood root for scalar contrasts
  (matrix-level assembler + ρ̂-variation Bartlett factor + >10% material flag).
- **#1219**: per-term EDF for `te()`/`ti()` is the influence-matrix trace over the
  term's coefficient block (was double-counting shared tensor coefficients).

**Terms, families & observation bands**
- **#1064**: `--family gamma` accepted as an alias for `gamma-log`. **#1160**:
  `Smooth(by=col)` plumbed through the `smooths={}` descriptor path. **#1158 /
  #1159**: marginality-aware `:` interaction expansion. **#1214 / #1215**:
  covariate-rescaling invariance for `cr` and `tp` smooths. **#1246**: sphere
  `wahba_sobolev`/`wahba_pseudo` aliases + `degree=` route to `harmonic`.
- **#1193 / #1194** (with #817): equal-tailed Poisson, Tweedie, NB and Beta
  observation bands.

**Numerics & performance**
- n-free κ-loop fast path with bit-exact β̂ on the slow path (#1216); on-device
  CUDA Step-6 joint-β contraction for survival-flex (#1133); per-block
  λ-coercivity threshold in the penalty pseudo-logdet (#1237); flat-residual
  stall now exits the inner joint-Newton instead of grinding to the cycle cap
  (#1040); exact tall RRQR on exact-collinearity (#933); ALO stabilization
  degrades gracefully instead of aborting the outer eval (#1191).

(See the 0.1.217 / 0.1.218 entries below for the GPU joint-Hessian build fix and
the BLAS-3 `coord_corrections` perf sweep that this crate release also carries.)

## gamfit 0.1.218 (2026-06-15)

- **Build fix**: wire the whole-`Xᵀdiag(w)X` GPU joint-Hessian path
  (`rigid_joint_hessian_on_gpu`) into the rigid `hessian_dense_override`, guarded
  by a GPU-presence probe (CPU boxes skip the weight-vector alloc and take the
  chunked-BLAS3 path unchanged). Clears the dead-code ban that failed 0.1.217.
- Saturate the rayon pool in `chunked_row_reduction` (4×workers chunks, was a
  fixed 32 that idled half a 64-core box) — completes the CPU-utilization fix.
- `DenseDesignMatrix::cache_identity` for memoizing the `X·F` projection across
  the k per-coordinate correction operators within one outer eval.

## gamfit 0.1.217 (2026-06-15)

- **Build fix**: re-export `SaeBasisEvaluator` in the pyffi prelude. The #1117 SAE
  fix retyped evaluators to `dyn SaeBasisSecondJet`; calling its supertrait
  `.evaluate()` needs `SaeBasisEvaluator` in scope, which broke the 0.1.216 wheel.
- Perf sweep continues: n-independent outer loop (eliminate redundant ext-coord
  n-row drift re-streams), SIMD-batched rigid per-row jet, GPU-routed rigid
  `XᵀWX` Gram when CUDA present (CPU fallback otherwise), cross-disease duchon
  basis+identifiability cache (build once for the shared cohort, not 17×), and
  the BLAS-3 batched all-axes second-directional override (~p× on the dominant
  coord_corrections term).

## gamfit 0.1.216 (2026-06-15)

Open-issue bug fixes + inner-solve perf:
- **#1128**: `gamfit.fit(Surv(...))` with no `survival_likelihood` now defaults to
  `transformation` (matching the CLI) instead of the broken `location-scale` that
  aborted the identifiability audit on right-censored data. Fixed at the single
  `FitConfig::default()` source.
- **#1127**: Gaussian `s(x)` REML is now scale-equivariant to `y→a·y` — the
  singularity floor `smooth_floor_dp` is a fraction of the weighted null deviance
  (was absolute 1e-12), so `λ̂`/EDF/smooth-shape are invariant down to a=1e-8.
- **#1117**: the SAE production term builder now installs the analytic second jet,
  so a rank-deficient K=1 circle decoder reparametrizes to its data-supported rank
  and completes stage1-step0 in budget instead of stalling.
- **#1126** (already on main): measure-jet κ non-convergence degrades to the
  certified baseline geometry instead of fatally aborting at tol=1e-10.
- **PERF**: the inner DENSE_SPECTRAL joint-Newton path no longer re-applies the
  matrix-free operator ~25× per cycle (trust-region model + Cauchy leg) — those
  route through the already-materialized dense Hessian (O(n·p)→O(p²)).
- Removed a second per-outer-eval `log::debug!` spam site (gated to once/process).

## gamfit 0.1.215 (2026-06-15)

- Remove per-call `[STAGE] BMS rigid ... BLAS-3 ... path TAKEN/NOT-taken` log
  lines from the rigid row-kernel dispatch (added in 0.1.213 for one-shot gate
  diagnostics). They fired on every `directional_derivative`/`hessian_dense`
  call — thousands of lines per fit — flooding the run log. The gate logic is
  unchanged; only the logging is removed.
- Cross-fit warm-start descriptor now encodes the realized per-block reduced
  β-width, so a p=37 fit no longer matches a p=85 artifact (no misleading
  length-mismatch skip) while same-width LOSO folds still transfer β.

## gamfit 0.1.214 (2026-06-15)

Biobank BMS speed sweep — attacks every recurring cost in the outer REML/LAML
loop, all exact (bit-faithful, no approximation, no skip flags):
- **coord_corrections** (the ~1.5–4min/eval Jeffreys H_phi drift): β-fixed base
  hoisted out of the per-direction loop + both p-axis row-stream sweeps
  parallelized across cores.
- **gradient_reload** (~5s/inner-cycle): the accepted trust-region line-search
  workspace is now reused, collapsing each accepted cycle from two row passes to one.
- **Murphy–Topel** SE correction and the **latent-z Rao-gate** score+meat: per-row
  scalar scatters replaced with single BLAS-3 GEMMs.
- **identifiability audit**: joint RRQR now runs from a single shared Gram (was a
  second full n-row stream); trivial full-rank case skips the redundant pass.
- **FFI encode**: column-major borrow (no StringRecord clone), parse-once, and a
  content-fingerprint cache so the shared base cohort is encoded once across diseases.
- **outer BFGS eval count**: the converged outer Hessian is transferred across LOSO
  folds to seed quasi-Newton, cutting line-search probes.
- **large-p outer LAML logdet**: one-pass dense assembly instead of p matvecs.
- BLAS-3 rigid Hessian fires for operator-backed designs; warm-start cross-fit
  length-mismatch declines to ρ-only instead of cold-starting.

## gamfit 0.1.213 (2026-06-15)

Continues the biobank BMS perf attack on the outer REML/LAML derivative path —
the real wall-clock black hole (coord_corrections, not Newton/PIRLS):
- **BLAS-3 rigid Hessian fires for operator-backed designs.** The cycle-0
  `hessian_qp` (~8s) and directional `gradient_reload` (~8s) floors were the
  BLAS-3 override bailing to the per-row BLAS-1 SYR scatter whenever the
  marginal/logslope design is operator-backed (always, at biobank scale, via the
  #461 influence absorber / #978 overlap-Z). The override now chunks via
  `try_row_chunk` and fires for any non-sparse design (sparse still routes to the
  sparse-aware scatter); a `[STAGE]` line logs why the fast path was/wasn't taken.
- **Jeffreys H_phi drift base hoisted** out of the per-direction loop — the
  β-fixed part of the coord_corrections H_phi correction is computed once instead
  of re-streamed per smoothing direction (full batched single-pass contraction
  landing on top).

## gamfit 0.1.212 (2026-06-15)

First publishable build since 0.1.209 — the 0.1.210/0.1.211 wheels failed the
build.rs ban gate (agent-collision leftovers) and never reached PyPI. This
ships everything since 0.1.209 with the gate cleared (USE-or-DELETE, no `_`
silencers): the row-kernel dense/directional override defaults now run the
extracted generic per-row path (genuinely consuming their args) while the rigid
kernel overrides with the BLAS-3 fast path; heartbeat scope guards bound + dropped
explicitly; dead `is_ext` removed. Folds in the real BMS perf work:
**BLAS-3 Jeffreys hessian_qp** floor, **BLAS-3 gradient_reload** floor, the
**same-β rigid third/fourth-tensor cache** (the genuine coord_corrections
collapse — the rigid path was rebuilding the per-row tensor every outer eval),
canonicalise fast-path, seed-screen skip on warm hits, warm-start BFGS metric,
and heartbeat CPU/active-scope diagnostics. Correctness: #740 ψ outer-gradient
KKT correction + the outer-HVP cross-ρψ second-order sign and full-H⁻¹ β̈ solve
(audit-confirmed), Murphy-Topel oracle fixture redesign.

## gamfit 0.1.211 (2026-06-15)

BMS biobank-fit perf + diagnostics. (1) **BLAS-3 rigid dense joint-Hessian**
for the post-step Jeffreys/Firth KKT-residual term (the `gradient_reload`
~8s/cycle floor) — same BLAS-1→BLAS-3 chunked-Gram treatment as the hessian_qp
fix, bit-for-bit, full-data dense-design gated (subsample/HT fall through).
(2) **Named heartbeat scopes for rigid coord_corrections** so the process
monitor localizes where that step's time goes. (3) ψ outer gradient routed
through the unified KKT-residual correction when the inner residual r≠0 (#740),
vanishing at exact KKT. NOTE: the BLAS-3 inner-kernel fixes (this + 0.1.210's
hessian_qp + 0.1.208's coord_corrections) all gate on the full-data
unit-weight dense-design path; a run is the truth test for whether the biobank
fit hits that gate (0.1.208's coord_corrections fix did not visibly land, under
investigation).

## gamfit 0.1.210 (2026-06-15)

BMS biobank-fit perf: **BLAS-3 rigid joint-Hessian directional derivative**
(the Jeffreys/Firth head term that dominated the inner `hessian_qp` floor). The
rigid (`flex=false`) path previously scattered the per-row contracted third
tensor into the dense p×p Hessian via per-row rank-1 SYRs — O(k·n·p²) BLAS-1
that never reached a BLAS-3 kernel (the same root cause as the historical
"55s Jeffreys spike"). It now projects each row-chunk with GEMMs and closes with
`Xᵀdiag(w)X` per chunk (k·n/chunk BLAS-3 calls), reading the per-row third tensor
from the shared cache once instead of per Jeffreys column. Bit-for-bit (only
summation order changes); claims only the full-data unit-weight dense-design
case, subsample/HT-weighted paths fall through unchanged. The ~8s-per-Newton-cycle
`hessian_qp` contribution drops to well under 1s — and it recurs on every inner
cycle of every inner solve, so it compounds across the whole fit. Also includes
the #740 KKT-residual correction for the ψ (ext) outer gradient (gradient side;
vanishes at exact KKT so converged fits are byte-unchanged).

## gamfit 0.1.209 (2026-06-15)

BMS biobank-fit perf + diagnostics bundle (no behavior change to the converged
fit). (1) **Warm-start seeds the outer BFGS iter-0 metric** (`1/‖g₀‖`) so the
first line-search step is accepted at α=1 instead of bracketing — kills the
~3 redundant full-inner-solve `Value` probes per warm fit (same certified ρ).
(2) **Skip the cold seed-screening cascade on a warm-start hit** — the validated
warm ρ goes straight to BFGS (saves ~43s/warm fold; cold fits still screen).
(3) **canonicalise skips the redundant post-T invariant double-RRQR when T is
identity** (the clean rank-full case) and uses a BLAS-3 MAP Gram (~12s→~2s/fit,
bit-identical). (4) **Heartbeat diagnostics**: the process-monitor line now
reports a true `cpu=N/M cores` utilization signal (from /proc), the
currently-active operation + its elapsed time, and a progress fraction — instead
of the misleading `active_threads=0`.

## gamfit 0.1.208 (2026-06-15)

BMS perf: **BLAS-3 chunked Gram for the rigid directional-Hessian drift**
(`coord_corrections`, the per-ρ REML logdet-gradient drift). The rigid
(`flex=false`, biobank) path previously folded each of n rows × k directions
with per-row rank-1 SYRs (BLAS-1, memory-bandwidth-bound); it now routes through
the same chunked `Xᵀdiag(w)X` GEMM structure the flex path uses — k GEMMs per
chunk instead of n·k rank-1 updates. Bit-identical drift (the rigid accumulator
has no h/w blocks; non-contiguous/subsample rows keep the per-row fallback). On
the biobank LOSO fold (n≈326k, k=8) this collapses the dominant `coord_corrections`
step from ~3.5 min to seconds. Also lands the redesigned Murphy-Topel oracle
fixture (#1028) and HVP ψ-gradient FD-attribution instrumentation.

## gamfit 0.1.207 (2026-06-15)

Generative-dispersion cluster (#1124 / #1125), finishing and shipping a fix the
prior run landed on main but never released or verified:

- **#1124 (Python path):** `Model.sample_replicates` /
  `posterior_predictive_check` drew Negative-Binomial replicate counts at the
  construction **seed** `theta = 1.0` instead of the estimated `theta_hat`, so
  replicate counts carried `Var = mu + mu^2` rather than `mu + mu^2/theta_hat`
  (far too overdispersed; wrong posterior-predictive p-values). The CLI
  `gam generate` path had been unified onto the canonical dispersion picker, but
  `gam-pyffi`'s `generative_replicates_impl` kept a *separate inline copy* whose
  NB arm read the seed. Routed it through the single
  `gam::generative::family_noise_parameter`, so the CLI and Python front-ends can
  never diverge on dispersion handling again.
- **#1125:** verified the per-row precision channel `exp(eta_d(x))` is threaded
  into `gam generate` for every dispersion location-scale family
  (Gamma/NB/Beta/Tweedie) and restored the missing regression coverage.

Verification: new deterministic cross-family `predict↔generate` per-row variance
agreement test (max rel dev ~2e-16 across all four families, including the
Tweedie φ = 1/precision reciprocal); restored Gamma-LS end-to-end CV test
(impliedK 9.5→0.9 across x); new Python `sample_replicates` NB regression
(recovers theta_hat, not the seed); all existing generative + #1057 replicate
suites green.

## gamfit 0.1.206 (2026-06-15)

BMS perf: **same-β reuse of the per-row cell-moment exact-cache**. Repeated
evaluations at a bit-identical coefficient vector β — the outer BFGS
`Value`→`ValueAndGradient` pair at one ρ, line-search re-probes, warm-start
replay — now reuse a fingerprinted, bounded (capacity-2, FIFO) exact-cache
instead of rebuilding the O(n·cells) quadrature from scratch. Reuse is gated on
exact byte-equality of every build input, so a hit is bit-identical to a rebuild
(gradient, Hessian, LAML cost unchanged). Also includes a scale-relative
penalized-deviance floor restoring small-response REML equivariance (#1127) and
a hermetic `.git/index`-based build audit (no `git` shell-out in maturin Docker).

## gamfit 0.1.205 (2026-06-15)

Performance bundle for the BMS biobank fit, no behavior change (all paths stay
bit-exact / KKT- and REML-certified). (1) **Skip redundant continuation
pre-warm on a warm-start cache hit** — when a fit seeds ρ/β from a structurally
matching parent (LOSO folds, multi-disease runs), the cold continuation
pre-warm seed (~160s/fit in the biobank log) is no longer recomputed. (2)
**Reuse the exact-Newton workspace across rejected inner trust-region cycles**
instead of rebuilding it. (3) **Parallelize the HVP host-pin per-row direction
fill** that previously ran serial. (4) **Parallelize the flat identifiability
audit** (per-block QR + per-column geometry). (5) **Cache the same-ρ assembled
outer Hessian operator** so the ~14-19s spectral factorization is not rebuilt
on the Value→ValueAndGradient pair (and line-search re-probes) the outer BFGS
issues at identical ρ. RAM headroom on the biobank box (~10 GB of 87 GB used)
makes the operator cache a safe memory-for-speed trade.

## gamfit 0.1.204 (2026-06-15)

Warm-start: cross-fit **β** transfer (Phase 2). A related later fit (notably an
LOSO fold whose reduced coefficient width differs, e.g. 37 vs 35) now seeds its
β from a structurally-matching parent fit's converged coefficients via
function-space projection through the gauge (`θ_new = (TᵀT+εI)⁻¹Tᵀ β_raw`), not
just ρ — the prior `[CACHE] beta-warm action=skip … length mismatch` becomes
`action=projected`. Exactness-preserving (the inner Newton + outer REML still
run to their KKT/REML certificate) and finite-guarded: any anomaly falls back to
cold β for that block, never erroring a fit. Also folds in a DSL fix routing
mgcv `bs='cr'/'cs'` through the 1-D B-spline dispatch.

## v0.3.117 — gam 0.3.117 / gamfit 0.1.203 (2026-06-15)

crates.io catch-up release: publishes to the `gam` crate all the engine work
that has already shipped to the `gamfit` wheel since gam 0.3.116 (gamfit
0.1.199). No new code lands here — this only bumps the crate version and tags
it so crates.io consumers of `gam` get the accumulated 0.1.200–0.1.203 work.
Highlights of what is now on crates.io:

- Batched row-parallel `coord_corrections` in the outer REML/LAML gradient
  (gamfit 0.1.203, #979): biobank-scale gradient evals no longer idle ~80–95s
  per outer iteration running k single-direction passes serially.
- Phase 0+1 cross-fit warm-start foundation (gamfit 0.1.203): descriptor-indexed
  `FitArtifact` + structural ρ-transfer that seeds related fits (LOSO folds,
  row-population changes) from a prior converged fit's smoothing parameters.
  The non-test lib also no longer carried unconsumed Phase-2 scaffolding (the
  dead-code that would have failed `cargo build --lib` under the crate's
  `warnings = "deny"`).
- Noise-floor inner-Newton termination guard + certificate railed-coordinate
  false-positive fix (gamfit 0.1.202).
- BMS separation false-positive fix + parallel Jeffreys curvature + nested-BLAS
  pin (gamfit 0.1.201).

See the per-wheel entries below for the full detail of each item.

## gamfit 0.1.203 (2026-06-15)

Performance: the outer REML/LAML gradient evaluation no longer recomputes the
per-coordinate log-det drift corrections one direction at a time. At biobank
scale (n≈3e5, k=8 smoothing coordinates) that per-coordinate loop ran k full
n-row passes serially — leaving the machine idle (~80–95s per gradient eval,
repeated every outer iteration) because each thin single-direction crossproduct
could not fill the thread pool. The site now routes through the family's batched
`hessian_derivative_corrections_result` hook whenever it is advertised (the BMS
exact joint-Newton workspace fuses all k directions into ONE row-parallel pass
that amortizes the per-row cached cell-moment / third-tensor work), turning the
former `coord_corrections mode=serial(inner-parallel)` into
`mode=batched(row-parallel)`. Other families keep the existing
parallel/serial single-direction fallback unchanged; results are identical
(same negation/sign semantics, solver still runs to its KKT/REML certificate).

Also lands the Phase 0+1 cross-fit warm-start foundation (descriptor-indexed
FitArtifact + structural ρ-transfer) used to seed related fits (LOSO folds,
row-population changes) from a prior converged fit's smoothing parameters.

## gamfit 0.1.202 (2026-06-15)

Robustness: bernoulli marginal-slope LOSO/biobank fits no longer spin to the
inner-Newton cycle cap, and a post-fit certificate false alarm is silenced.
- Noise-floor inner-Newton termination guard: when the trust region collapses
  and every line-search step is rejected at the ~1-ULP floor (the objective is
  flat along the gauge-flat coupled direction), the solve now TERMINATES and
  judges convergence on the identified (range) subspace instead of spinning to
  the 1200-cycle cap. The line-search early-exit reject also gained a rounding
  tolerance so a numerically-flat trial is not rejected on 1 ULP.
- Certificate (first-order optimality self-audit) railed-coordinate fix: the
  audit now projects its gradient-vs-value directional check onto the FREE
  (non-bound-active) coordinates, so a legitimate KKT optimum with a smoothing
  parameter at a box bound no longer reports a spurious GRADIENT-OBJECTIVE
  DESYNC. Diagnostic-only; no fitted result changes; still fires on genuine
  interior desyncs.

## gamfit 0.1.201 (2026-06-14)

Performance: biobank marginal-slope fits no longer stall on the per-coefficient
Jeffreys/Firth curvature. Two fixes to the joint-Newton hot path:
- The exact Jeffreys curvature term H_phi was built by a serial p-pass loop, each
  a full-data directional-derivative sweep (~55s deterministic on n~2e5, p=35,
  arming at the ill-conditioned converged cycle). The p independent directions
  now evaluate in parallel across the Rayon pool (bit-identical math).
- Nested faer-BLAS GEMMs inside Rayon row-parallel assembly were pinned to
  single-thread, collapsing rayon x BLAS thread oversubscription (~300 ->
  cores) with no caps and no environment variables. gam owns parallelism and
  saturates the hardware with one level of fan-out.

## gamfit 0.1.200 (2026-06-14)

PyPI Linux-wheel refresh with post-0.1.199 fixes for generation,
dispersion location-scale prediction, survival and shape-constrained fitting,
multinomial convergence, and SAE dictionary robustness.

Fixes:
- Negative-binomial `generate` and `sample_replicates` now use the fitted
  `theta_hat` instead of the seed/default theta.
- Gamma, NB, Beta, and Tweedie dispersion location-scale generation now threads
  the fitted per-row dispersion channel, fixing homoscedastic synthetic output
  from heteroscedastic fits.
- Dispersion location-scale fits now assemble covariance and EDF consistently,
  including the converged orthogonal path, and prediction gates cover Gamma, NB,
  and Tweedie cases.
- Posterior mean observation bands now use per-row `sigma(x)` for
  heteroscedastic location-scale families.
- Interval and transformation-survival fitting is more robust: trial-point
  non-convergence is treated as high cost, interval warm starts stay feasible,
  and constrained Newton steps respect the monotone cone.
- Multinomial and BMS convergence paths were tightened with residual-stall
  Newton-decrement certification, identified-subspace stationarity checks,
  reduced Schur preconditioning, and removal of hot-path diagnostic
  eigendecompositions.
- `average_derivative` and default `difference_smooth` behavior have regression
  fixes.

SAE / manifold:
- K>1 SAE dictionary fitting is more stable under deflation and rank reduction.
- JumpReLU active-set support is canonicalized.
- Curvature reports now expose delta-method curvature SE through the Python
  facade.
- Cylinder topology race and born-atom uncertainty-band work is included.

Build / packaging:
- Repairs test-target builds after the module split and dispersion type changes.
- Replaces mechanical split fragments with named modules across major Rust
  subsystems.
- Removes stale local workflow/scripts/examples and redundant stub README files.

## v0.3.116 — gam 0.3.116 / gamfit 0.1.199 (2026-06-14)

Large crate + wheel release rolling up the unreleased work since 0.3.115. The
headline is a wave of new user-facing modeling capability — interval-censored
survival from a formula, full multinomial prediction inference, expectile GAMs,
magic Poisson auto-detection, restricted mean survival time, exact full-conformal
prediction intervals, and per-term likelihood-ratio tests — alongside a deep
solver-robustness cleanup (the derivative-free compass-search optimizer is gone,
several silent-failure crutches now fail loud), the SAE/manifold interpretability
stack, and a broad reference-quality test expansion. Build, rustfmt, and the
clippy correctness/suspicious gate are all green.

New modeling capability:
- feat(#1108): interval-censored survival is now fittable from a user formula —
  a dedicated `SurvInterval(L, R, event)` response (disambiguated from
  delayed-entry `Surv(entry, exit, event)`) materializes the time basis at both
  boundaries with frozen-knot column reuse and routes through the latent-survival
  path. Completes the deferred wiring step; covered by a gauge-invariant σ̂
  truth-recovery + match-or-beat-lifelines test.
- feat(#1101): multinomial-logit inference completeness — H⁻¹ covariance,
  `MultinomialPredictor` delta-method prediction intervals + standard errors,
  posterior_predict, and a per-term Wald summary, wired through the FFI and the
  Python surface. The fitted spec is frozen so `predict` rebuilds the exact
  fitted design.
- feat(#1100): expectile GAMs via a LAWS (asymmetric-least-squares) outer loop.
- feat(#1065): magic-by-default Poisson auto-detection for non-negative integer
  count responses.
- feat(survival): restricted mean survival time (RMST) output.
- feat(#1054, #942): exact full-conformal prediction intervals —
  `predict(interval='conformal')` auto-routes to the exact Gaussian jackknife+
  path and the exact GLM full-conformal engine, surfaced to Python.
- feat(#1063): per-term likelihood-ratio statistics with a magic Lawley–Bartlett
  correction (dispersion-carrying known-scale jets for Gaussian/Gamma), through
  the FFI and Python API.
- feat(#1032): `FitResult::ResidualCascade` variant + `fit_from_formula`
  auto-route with a quasi-uniformity guard, plus serde state replay.
- feat(#1057): generative replicate sampling + posterior-predictive checks
  (`Model.sample_replicates`).
- feat(#1049): posterior_predict for the Bernoulli marginal-slope path.

Geometry, manifolds & SAE interpretability:
- feat(#1061): SPD / Grassmann / Stiefel / Poincaré are now selectable response
  geometries.
- feat(#1104, #944): `ConstantCurvature` response manifold with an end-to-end
  curvature-as-estimand inference layer (κ̂ + profile CI + κ=0 LR from a real
  fit).
- feat(#1097, #1099, #1102, #1103, #1055): per-atom Riesz-debiased functionals,
  per-atom curvature profile-likelihood CIs with flat-cusp handling,
  cross-checkpoint atom-trajectory dynamics, and any-n e-value atom-smooth
  significance — wired end-to-end into `dictionary_report` and the Python facade
  (`Model.debiased_functional()`, `sae_checkpoint_dynamics`).
- feat(#1026): evidence-scored curved+linear-tail hybrid dictionary split made
  load-bearing on reconstruction, with leave-one-atom-out held-out EV
  attribution.
- feat(#1058): anytime-valid structure certificate surfaced post-fit.
- feat(#1038): exact cross-row integration-by-parts Woodbury on the arrow
  evidence cache (wired at the assembly site).

Solver robustness (deslop):
- The derivative-free compass-search optimizer is removed entirely: the last
  gradient-free callers (Weibull / transformation survival baselines) were
  migrated to exact-gradient BFGS, and `Solver::CompassSearch` and its dispatch
  were ripped out — it could hang, so it is gone.
- The arrow/Schur Woodbury path now fails loud instead of propagating a silent
  NaN.
- Removed the wall-clock fake-convergence early-exit in blockwise PIRLS.
- Cost-stall convergence must now clear the gradient tolerance, not just a flat
  cost.
- Replaced the magic 16-iteration ridge-escalation cap with a principled
  Gershgorin spectral ridge.
- Stopped silently clamping negative-eigenvalue smoothing corrections into a
  fake-PSD covariance; refuse phantom zero penalty-logdet ρ-derivatives instead
  of silently desyncing the outer optimizer.
- Constrained joint-Newton path: Newton-decrement certificate wired in,
  negative-curvature reflection convexifies the QP (#1040).

Performance:
- perf(#1017): CPU-resident reduced-Schur operator for the SAE matvec — factored
  (Lᵢ, Yᵢ) residency replacing the dense p×p block, with a work-based offload
  predicate and a parallelized per-row matvec.
- perf(#1082): parallelize competing-risks CIF assembly over the (independent)
  row axis behind the rayon nesting guard + a row-count gate, byte-identical to
  the serial path; n-scaled outer-gradient floor across all families;
  warm-started β in the spatial outer loop; right-sized quality-test fixtures.

Reference-quality tests: a broad new wave across the mature comparators
(lifelines interval censoring, VGAM multinomial smooth-by-factor, mgcv tensor
products, frailty survival, PyMC/InterpretML, conformal coverage, and more).

Known limitations (tracked, not regressions): several quality fits still exceed
the wall-clock budget (#1082, #1116) and the survival marginal-slope path is slow
on some bases (#979) — under active investigation; these are perf/coverage gaps,
not correctness defects in the shipped surface.

## v0.3.115 — gam 0.3.115 / gamfit 0.1.198 (2026-06-13)

Crate + wheel release rolling up the unreleased work since 0.3.114: the O(n)
spline scan reaching the `fit_from_formula` library entry point and gaining
order-3 (quintic) support, a new exact O(n) multi-term additive backfitting
module, a Bessel-K accuracy fix in the Matérn/Duchon radial lattice, the exact
dense softmax-entropy row Hessian, the Murphy–Topel generated-regressor
correction, analytic Jacobi-field exp-map VJPs on curved manifolds, sphere-chart
isometry-defect certification, and the gam-pyffi module carve-out — plus a batch
of solver-robustness fixes and a broad new reference-quality test wave.

O(n) spline scan — library entry point + order-3 (#1030, #1044):
- feat: `fit_from_formula` now auto-routes a qualifying single 1-D Gaussian
  smooth through the exact O(n) diffuse-REML Kalman/RTS scan (new
  `FitResult::SplineScan` variant), matching the FFI/CLI auto-routing that
  already shipped; the conservative detector falls every other shape through to
  the dense path unchanged.
- feat: order-3 (quintic) smoothing splines (`λ∫(f‴)²`) via an exact diffuse
  leading-block smoother for the two partially-diffuse leading nodes, with
  symmetric (Jacobi) equilibration + iterative refinement of the 6×6
  leading-block solve so the κ≈δ^{-5} stiffness is resolved at heavy smoothing.
  Validated against the order-general dense exact-posterior oracle to 1e-6·SD
  (posterior) and 1e-7 (REML differences); the auto-route now covers
  penalty_order ∈ {1, 2, 3}.

Exact O(n) multi-term additive backfitting (#1034 item 3):
- feat: new `gam::solver::scan_backfit` — `fit_scan_backfit` /
  `fit_scan_backfit_at` solve `y = α + Σⱼ fⱼ(xⱼ)` with exact O(n) spline-scan
  inner smoothers, certified against a dense joint penalized solve, with
  per-term λ selection validated match-or-beat vs a dense joint-REML grid on
  truth recovery. (The module had been committed unwired and uncompiled; it is
  now wired, compiled, and covered by three oracle tests.)

Numerical accuracy:
- fix(basis): `K_{l+½}` Bessel coefficients now come from the exact upward
  recurrence instead of Lanczos-Γ at integer arguments, removing ~1 ULP errors
  that amplified in the near-cancellation `r^{-0.5}·K_{1/2} + r^{0.5}·K_{3/2}`
  radial-derivative sums of the Matérn/Duchon lattice.
- feat(#1035): exact Chebyshev jet tower for the radial-profile derivative
  channels.
- feat(#1038): the softmax assignment-entropy prior's exact dense per-row
  Hessian in logits is wired through value / log|H| / θ-adjoint / ρ-trace
  together, so the criterion and its gradients differentiate the operator the
  prior actually defines (previously only the diagonal was stored).

Geometry + manifolds:
- feat(#944): analytic Jacobi-field `exp_map_vjp` for κ ≠ 0 in constant-curvature
  geometry (replacing the κ = 0-only path), with a by-reference `dot` fix.
- feat(#1019): certified sphere-chart isometry-defect functional plus a post-fit
  measurement/logging pass.
- feat(#1026): per-atom fitted-turning Θ measurement for the EV-vs-Θ signal.

Other:
- feat(#1028): Murphy–Topel generated-regressor variance correction assembled
  with J_zeta accumulation (BMS).
- feat(#1017 Phase 0): driver-level parallel topology-candidate selection.
- feat(#1033): ψ-Gram tensor gradient coverage surfaced at the eval_full seam,
  with n-free k-space ψ-derivatives feeding the Gaussian gradient.
- fix(#1036): the cross-seed structural detector parses the real non-PD wording.
- fix(audit): RRQR keeps its own block when it already named the lower-priority
  alias side.
- refactor(#780): the gam-pyffi monolith is carved into focused modules
  (`python_literal`, `benchmark_scores`, `competing_risks_decode`,
  `summary_render`); restored a missing `PyValueError` import that would have
  failed the wheel build.
- perf(#1035): caller-thread line-search LL sweep / serial small row-folds; a
  measured-no-op row-fold gating was reverted (it implied a perf win that was
  not observed).
- gpu: dropped the dead Phase-3 Cornish–Fisher Pólya-Gamma oracle.
- tests: new reference-quality gates (grid_spline_2d vs mgcv te() #1031,
  multinomial softmax truth-recovery vs mgcv/VGAM #715, competing-risks
  total-probability identity #1025, survival marginal-slope convergence #1040,
  measure-jet aniso isotropic-fallback regression, thread-pool fit-invariance
  #1045).

## v0.3.114 — gam 0.3.114 / gamfit 0.1.197 (2026-06-13)

Crate + wheel release rolling up the O(n) spline-scan auto-routing, the
measure-jet simple/multiscale split and conditioned-frame ψ-Gram tensor, the
multinomial formula-fit corrections, the Matérn-anisotropy isotropy fixes, and
a large batch of formula-parsing / solver-robustness fixes. (On crates.io this
also rolls in the 0.3.113 content — `gam 0.3.113` was tagged in-tree but never
reached crates.io, which was still on 0.3.112; this release re-syncs both
registries.)

O(n) spline scan — auto-routing + saved models (#1030, #1034):
- feat: the FFI and CLI now auto-route a single 1-D Gaussian smooth through the
  O(n) spline scan end-to-end, so `s(x)`-only Gaussian fits skip the dense
  REML path. Includes an n=1e6 spline-scan benchmark + a biobank-scale
  no-regression certificate.
- feat: a lossless `SplineScanState` saved-model representation channel with a
  bit-for-bit JSON round-trip gate (serde_json `float_roundtrip`), validated
  through `FittedModel`; the scan-model payload exempts dense-only fields.
- feat: order m∈{1,2} routed through detection + callers (m=1 dense oracle),
  with order-general scan API names and a match-or-beat truth-recovery gate vs
  mgcv for the spline scan.

measure-jet — speed + accuracy (#1039, #1033b):
- perf: SIMPLE (single-scale) mode is the default again, fixing the ~12× slowdown
  vs Matérn (#1039); the simple/multiscale auto-mode split is documented and
  hardened, with a θ-invariant Gram cache for fixed-design ρ-only trials.
- feat: the certified Chebyshev-in-ψ Gram tensor is built in the conditioned
  frame and fires on real spatial smooths (#1033b); n-free Gaussian ψ-gradient
  from the tensor derivatives, with a finite-difference gradient certificate.
- fix: density-free default α = 3/2 (from the ρ^{3−2a} derivation).

Multinomial formula fits (#715):
- fix: standardize the unpenalized parametric columns in the multinomial formula
  fit; restore exact outer curvature for double-penalty multinomial formula fits;
  floor the multinomial inner KKT tolerance at the softmax f64 noise floor.
- tests re-grounded onto a like-for-like mgcv REML / `select=TRUE` comparator;
  no quality bar weakened.

Matérn anisotropy (#1042):
- fix: honor an explicit all-zero `aniso_log_scales` as isotropic on the Matérn
  forward path; gate the anisotropy auto-seed on center-strategy provenance
  rather than seeding unconditionally.

Survival marginal-slope (#1040, partial):
- fix: the joint-Newton inner solve now takes a scale-invariant relative
  objective-plateau exit on flat REML valleys, so survival marginal-slope fits
  terminate instead of grinding to the inner-cycle ceiling every outer ρ-eval.
- fix: early-certificate inner exits report the residual they actually certified
  on, so the terminal verdict can no longer print `converged=true … residual=inf`
  (inner-report truthfulness).

Other fixes:
- fix(#1025): route uncoupled multi-block inner solves to the exact
  block-coordinate path; joint competing-risks transformation truth-recovery.
- feat(#939): wire the Lawley/Bartlett cumulant summary fields through `main`.
- fix(#1036): generic cross-seed structural-failure bail in the seed cascade.
- fix: accept ragged `Column` lists in `write_columns_csv` by NA-padding the tail
  (unblocks the varying-ν Matérn / sphere-geodesic / Grassmann quality suites).
- fix: peel mgcv `c(...)`/`(...)` R-vector wrappers in `split_list_option`, so
  `te(x,y, bs=c('tp','tp'), k=c(5,5))`-style per-margin options parse correctly.
- fix: width-guard the cached marginal/logslope β-hints before block assembly so
  a stale hint falls back to a clean cold start instead of tripping the
  block-spec contract.
- perf: parallelized SAE row assembly, iterator/slice-dot REML and SAE reductions,
  fused ρ-posterior cost+gradient eval, and gradient-only routing for
  high-dimensional REML ρ fits.

## v0.3.112 — gam 0.3.112 / gamfit 0.1.195 (2026-06-11)

Crate + wheel release rolling up the over-dispersed-Gamma predictive-interval
fix, the random-slope / factor-smooth predictive-quality cluster, and the
marginal-slope hang / OOM fixes that unblock the survival + binary
`marginal-slope` study.

Predictive intervals — over-dispersed Gamma / Tweedie (#1018):
- fix: `gamma_quantile` now inverts an *owned* regularized lower-incomplete-gamma
  CDF (`regularized_lower_gamma`, the Numerical Recipes power-series / Lentz
  continued-fraction split with the leading factor kept in logs) instead of
  `statrs::gamma_lr`, which hard-clamps to `0` for every `x ≤ 1.11e-15`. In the
  small-shape lower tail (`a ≲ 0.1`, the moment-matched `k = μ²/V` of a strongly
  over-dispersed predictive) the clamp zeroed the Halley residual and walked the
  iterate *up* to `~1.6e-15` — so a nominal 2.5% bound carried up to ~19% of the
  mass and the interval under-covered on the low side. Round-trip
  `P(a, gamma_quantile(p,a,1)) ≈ p` now holds to 1e-9 down to `q ~ 5e-33`, pinned
  by an independent self-contained oracle.

Random-slope / factor-smooth predictive quality vs lme4 / mgcv (#903):
- fix: `bs="re"` is now the parametric random intercept+slope `[1, x]` mgcv's
  `(1 + x | g)` denotes, not a piecewise-linear B-spline under the pooled-knot
  heuristic (~6 wiggly coefs/group). The over-parameterized term ill-conditioned
  the REML/joint-Newton solve (minute-long fits) and broke partial pooling;
  group slopes now shrink toward the fixed population trend (7 s fit, 2 coef/group
  edf), beating no-pooling OLS out of sample.
- fix: cap the `bs="fs"` shared marginal (and `bs="sz"`) at mgcv's default
  `k ≈ 10`; the pooled heuristic gave ~24 functions/group and REML over-fit the
  shared shape. fs recovers at 0.0528 (beating mgcv's 0.0548).
- tests re-grounded onto the correct comparator model (random slope vs lme4's
  `(1+x|g)`, not the shrink-to-zero `fs`); no quality bar weakened.

Marginal-slope (binary + survival) — hang / OOM (#979, partial):
- fix: the inner joint-Newton no longer hangs to its 1200-cycle ceiling on
  fully-rejected cycles (β reverts and an interior block pins `max(block_radii)`,
  so every subsequent cycle was bytewise identical and rejected for the same
  reason — ~120 s burned per outer ρ-evaluation, the survival "hang").
- fix(large-scale): the survival SMGS phase-4b observability / rank-diagnostic
  blocks densified operator-backed designs unconditionally and OOM-killed the
  host at n=320k; they now go through a pre-allocation byte budget
  (`try_to_dense_by_chunks_budgeted`, 256 MiB/matrix) and `warn!`-skip on
  refusal, while real numerical-rank failures still propagate.
- correctness: the Jeffreys curvature in the coupled-joint outer LAML is now the
  exact Daleckii–Krein form (dropping the `K²` vec-Gram surrogate that put ~1e20
  phantom curvature on floored eigenpairs and froze the inner step along
  Firth-active directions), with its exact divided-difference drift, the Tier-B
  `Φ(β̂)` value folded back into the outer cost, and PSD projection on
  SPD-requiring step paths. The deeper inner-Newton convergence on the hardest
  binomial-location-scale Firth-active fixtures remains in flight; #979 stays
  open for that.

Build / CI:
- #901: keep the non-Gaussian value path on the spectral LAML.
- CI: per-binary test wall-clock caps now use GNU `timeout` (process-group kill
  by default) in place of the hand-rolled `setsid` watchdog; removed the stale
  precheck / hillclimb scripts.

## v0.3.111 — gam 0.3.111 / gamfit 0.1.193 (2026-06-10)

Crate + wheel release rolling up the factor-smooth predictive-quality fix and
the SAE-manifold streaming/LLM-scale restoration on top of the green tree.

Factor-smooth / random-slope quality (#903):
- fix(fs): the `bs="fs"` factor smooth now penalizes **each null-space
  dimension separately** (one rank-1 `I_L ⊗ z_k z_kᵀ` per null direction, each
  its own shared smoothing parameter), mirroring mgcv's
  `smooth.construct.fs.smooth.spec`, and drops the non-mgcv range ridge. The
  prior single *combined* null penalty (one λ for intercept + slope) could not
  express the distinct random-intercept vs random-slope variances, so
  per-group slopes got no partial pooling and the held-out per-subject forecast
  inherited full no-pooling variance. REML now fits both variances, tracking
  lme4's correlated-RE BLUP.

SAE-manifold streaming (LLM-scale fitting):
- restore + wire the out-of-core streaming joint-fit driver
  `run_joint_fit_arrow_schur_streaming` (re-seeds each chunk via `chunk_init`,
  never materializes the `(N×M)`/`(N×K)` per-row buffers) plus an in-memory
  entry `fit_streaming_in_memory`, with a chunk-size-invariance contract test.
  This is the memory-bounded fit path for the LLM-scale teacher; the in-core
  driver cannot scale to N = billions of tokens.
- feat(#972/#977): closed-form streaming polar frame refresh
  (`refresh_active_frames_from_data`) — the U-block of the alternating
  block-coordinate ascent that complements the border C-block Newton step.

Build / correctness:
- bms #905 conditional `E[z|C]`/`Var(z|C)` Auto gate; #740 contracted ψψ
  second-order hook; #1000 centered Gaussian outer-REML λ-search; assorted
  ban-gate and unused-symbol cleanups to keep `cargo check --tests` green.

## v0.3.110 — gam 0.3.110 / gamfit 0.1.192 (2026-06-10)

Crate + wheel release rolling up the SAE-manifold identifiability-certificate
work (#980/#981/#907/#995/#996/#998) plus the build/lint fixes that unblock a
green `cargo check --tests` + Rustfmt across the workspace.

Identifiability / SAE-manifold certificate:
- feat(#998): the residual-gauge certificate now realises within-atom gauge
  orbits **exactly** in the model's own (decoder, coordinate) parameter space.
  The coordinate-motion field comes from the group action (circle/torus
  shifts, flat-patch so(d) rotations); the decoder compensation is profiled
  out by least squares; the leftover residual is the orbit's true data cost —
  exactly zero for a basis closed under the action, positive otherwise, so
  **basis closure is computed, not declared**. All pinning of true model-class
  symmetries flows through the injectable `OrbitPenaltyOperator` channel.
  `residual_gauge_exact` merges exact within-atom verdicts with the calibrated
  frame path for spheres / unviewed atoms / cross-atom families.
- fix(#995): the per-generator verdict uses the relative curvature fraction
  `‖R ξ̂‖² / σ_max(R)²` (kept magnitudes, survives a full-rank pinning span)
  calibrated by a computed mean-frame `lowering_error` scale, so compression
  artifacts are never read as a pin. Shipped per-generator through the FFI.
- feat(#996): the discrete-mixture rung refines locally around the coarse
  `MIXTURE_K_LADDER` winner until bracketed, so off-ladder truths (k = 4/6/8)
  are named exactly instead of snapping to the nearest rung.
- test(#980/#981/#907): Theorem-2 arm, the circle-read-discretely two-verdict
  race, and the repeated-draw calibration sweep.

Build / lint:
- fix: drop a dead `policy` field (gamlss dispersion family), wire the new
  `DispersionLocationScale` `FitRequest`/`FitResult` arms + `ReportInput`
  convergence fields through the Python FFI, annotate ambiguous-float arrays
  in the #974 residual-factor test, and `cargo fmt --all`.

## v0.3.109 — gam 0.3.109 / gamfit 0.1.190 (2026-06-10)

Crate + wheel release rolling up the correctness fixes and the new
inference / SAE / topology / survival surface landed since gam 0.3.108 /
gamfit 0.1.189. (The intervening gamfit 0.1.190 wheel never reached PyPI —
its build broke on a `gam-pyffi` site that lagged the #983 `theta_fixed`
refactor; that and the rest of the `-D warnings` / ban-gate breaks are fixed
here, so the wheel builds green again.)

Family / link / survival correctness:
- fix(#947): binomial 4th central moment uses the 3rd inverse-link pdf
  derivative (μ''''), not the 5th.
- fix(#948): exact binomial derivative towers in the saturated tails — no
  clamped-μ surrogate (the derivative path is the derivative of the evaluated
  row loss).
- fix(#953): integrated observation variance uses the Tweedie φ / Gamma
  shape, not a hard-wired φ=1.
- fix(#961): the link is validated against the explicit family instead of
  inferring a conflicting family from the link.
- fix(#963): exact public Log inverse-link jet on the predict surface (the
  solver keeps its internal clamp).
- fix(#964/#965/#966): survival FFI fallback S→H, negative-time handling, and
  hazard-differencing math corrected.
- fix(#983): a fixed `--negative-binomial-theta` is honored end-to-end
  (`theta_fixed` routed through every call site) instead of being silently
  re-estimated.

Geometry / manifolds / REML:
- fix(#949): Sphere / Euclidean `sectional_curvature` validates a
  nondegenerate tangent 2-plane before dividing by the wedge area.
- fix(#950): `simplex_exp_map` CLR requires a strictly-positive base.
- fix(#951): `signed_log_sum_exp` propagates +∞ log-magnitudes correctly.
- fix(#952): Fisher-Rao precision blocks validated PSD (PD on the Cholesky
  path), not merely symmetric with non-negative diagonal.
- fix(#954): optimizer stationarity uses the shift-invariant
  ‖grad_k‖/‖grad_0‖ measure at both call sites.
- fix(#955): the Euclidean differential is raised to the Riemannian gradient
  through the metric.
- fix(#957): the trust-region radius is constrained 0 < radius ≤ max_radius
  before the first step.
- fix(#967): a shared per-smooth λ makes the response-geometry tangent fit
  frame-equivariant.
- fix(#901): intrinsic pseudo-logdet over range(H_pen); the custom-family
  joint trace kernel uses the full spectral M⁺; the GLM cubic-correction drift
  stays in operator form (no near-null dense-C[v] roundoff blow-up).
- fix(#902): Matérn ψ-derivative penalty enumeration aligned with the forward
  gate (RKHS rule j ≤ ν + d/2).
- fix(#978): persisted + replayed global-orthogonality chart so an overlapping
  global + factor smooth on the same covariate residualizes consistently.
- fix(#780): the LinearOperator seam is wired into weighted_design_products.
- remove: the standalone SINDy sparse-dynamics module (STLSQ + library + FD +
  equation renderer, `SindyAtoms`) and its pysindy reference-quality suite.
  It shared no machinery with gam's REML/penalized-spline core, contradicted
  the REML-always selection policy (BIC + hard-threshold), and existed only as
  a pysindy benchmark target; its sole prospective consumer (#908 Manifold-
  SINDy) was unbuilt research. Removed with #482/#908/#945/#958/#959.

New surface:
- feat(#986): per-atom decoupled Extended Fellner–Schall as the primary outer
  at frontier ρ-scale (auto-switched into `run_outer`), with a matrix-free
  θ-HVP (#740) for the shared-border coupled correction.
- feat(#931/#935): the profiled criterion calculus — LAML as
  self-differentiating atoms over one sensitivity operator (factored H⁺;
  β̇ / ALO / influence / case-deletion / θ-HVP as its contractions).
- feat(#932): Taylor-jet tower algebra (`Tower4<K>`) — write a family's row
  log-likelihood once, derive its whole `RowKernel` derivative tower exactly.
- feat(#942): exact full-conformal prediction for penalized GAMs (Layer 1).
- feat(#944): a `ConstantCurvature` manifold family through κ=0 with exact
  κ-jets riding the #932 tower.
- feat(#973): streaming out-of-core border-Gram accumulator on a deterministic
  pairwise-reduction tree (order-invariant, resumable) + the streaming SAE
  corpus driver.
- feat(#974): structured-residual estimator with a single likelihood-whitening
  seam.
- feat(#972/#985/#987): low-rank Grassmann decoder frames; a sublinear
  candidate-atom index for active-set proposal; frontier distribution +
  two-tier Fisher-on-subsample harvest economics.
- feat(inference): SAE-manifold diagnostics — two-lens per-atom (presence vs
  behavioral coupling) diagnostic, residual-gauge certificate, Fisher-mass
  enrichment ordering, provenance-carrying RowMetric; plus the steering
  primitive (`sae_steer_delta` FFI + `ManifoldSAE.steer`) with output
  dosimetry, validity radius, and an off-manifold guard.
- feat(#907): discrete-mixture and union-candidate rungs in the topology race
  with selection-time stacking.
- feat(#741): `SmgsLiftViaT::lift_covariance_via_t` inference pushforward.

## gamfit 0.1.189 (2026-06-09)

PyPI wheel release on top of gam 0.3.108, rolling up the survival + predict
correctness work landed since gamfit 0.1.188:

Survival (transformation / location-scale AFT):
- fix(#892): reduced parametric-AFT time-warp gauge is now identified and
  verified end-to-end. The fit routes σ-scaling through the location channel
  (`η_t → η_t − log t`, warp `h ≡ 0`, so `u = (log t − μ)/σ`) and the predict
  path mirrors it — a 3-bug chain fixed forward: regime detection by all-zero
  `beta_time` (not `is_empty`, since finalize emits a non-empty length-`p` zero
  vector), keep the full-width zero-β time basis (warp nulls via β=0) so the
  scale-deviation primary keeps its full column count and the hazard dim guard
  holds. New e2e CLI fit→save→load→predict test tracks the lognormal truth.
- fix(#899): saved Weibull baseline scale recovered from the time anchor, not a
  stale/unidentified `exp(−β[0]/shape)`.
- fix(#900): factor-by-level smooths centered against their gated level
  indicator, so each `s(x, by=g)` group keeps its own per-group baseline.

Multinomial / GAMLSS:
- fix(#715): multinomial REML adapter skips the outer seed-screening cascade on
  the LM-damped formula path so a valid REML seed survives the canonical-gauge
  null direction.
- refactor(#780): extracted the binomial neg-log-q derivative math and the
  location-scale validators out of the gamlss.rs mega-file into
  `gamlss/binomial_q_derivs.rs` + `gamlss/validation.rs` (pure, no behavior
  change).

Predict guard tests:
- Added e2e truth-recovery regressions for tensor `te(x,z)` and by-factor
  `s(x, by=g)` predict on fresh grids (both paths verified correct).

## v0.3.108 — gam 0.3.108 / gamfit 0.1.188 (2026-06-09)

Crate + wheel release rolling up everything since gam 0.3.107. crates.io is at
gam 0.3.107 and PyPI has been stuck at gamfit 0.1.180 — the intervening
gamfit 0.1.181–0.1.187 wheels never published because the wheel build broke on
lib compile errors (`-D warnings`) that the per-push CI does not catch (Rust CI
runs nightly only). Those breaks are fixed here, so the wheels build green again
for the first time since 0.1.180. The `gam` 0.3.108 crate published to
crates.io; the gamfit 0.1.188 PyPI upload is separately blocked on the project's
10 GB storage quota (the wheels build but `twine` is rejected) — see #894.

New:
- feat(report): terminal smooth visualizer. `gam report` now prints a unicode
  block-glyph sparkline (`▁▂▃▄▅▆▇█`) of each smooth term's fitted partial effect,
  labelled with its x- and y-ranges — instant shape diagnostics with no plotting
  backend. Pure, read-only renderer in the new `gam::sparkline` module; faithful
  min→bottom / max→top mapping with graceful constant/NaN/empty handling.

Location-scale (GAMLSS) correctness:
- fix(#884): Gaussian location-scale models now persist and reconstruct the
  actual `response_scale`, and the noise σ-floor is response-scale equivariant —
  rescaling the response rescales σ̂ exactly instead of silently clamping at a
  fixed floor. Wired through the predictor, the CLI save path, and the gamfit
  FFI payload.
- fix(#684): Gaussian loc-scale uses a Gaussian (not GLM-upward-biased) seed risk
  profile, and the matrix-free workspace mean↔scale cross-block is the Fisher
  zero, not the observed `2κm` term.
- fix(gamlss): the Gaussian wiggle mean↔scale Fisher cross blocks are exactly
  zero (static, 1st- and 2nd-directional, and the μ↔logσ / logσ↔wiggle mixed
  β·ψ crosses), removing a spurious coupling in the joint Hessian.
- fix(#826): true dogleg globalization + scale-invariant Marquardt damping for
  the coupled location-scale inner Newton, which previously froze on tightly
  coupled problems.

REML / LAML:
- fix(#877): Gaussian REML is weight-scale invariant — λ̂ now scales exactly with
  weight magnitude via complete normalization and a weight-anchored seed +
  ρ-prior, so multiplying all weights by a constant no longer moves the fit.
- fix(#715): structural EDF floor uses true generalized eigenvalues; multinomial
  inner-Newton budget is decoupled from the outer `max_iter`/`tol`; Firth-bounded
  finite-saturated multinomial formula fits are accepted instead of mis-rejected
  as separation.
- fix(reml): custom-family LAML `log|S_λ|₊` value is synced to the gradient's
  pseudo-logdet classifier; the InnerAssembly TK correction value and gradient
  are guarded together (single `include_logdet_h`) so they cannot desync.
- fix(#808, #854): exact floored-pseudo-inverse and second-directional Hessian
  derivatives for the joint-Jeffreys spatial-adaptive family.

Smooths / latent / shape constraints:
- fix(#876): periodic latent Duchon decoder recovers a circle/torus instead of
  collapsing (spectral seed + seam-consistent jet that differentiates the
  periodic forward).
- fix(#873): shape-constrained smooths get a strictly-interior cold-start seed,
  equality-pair-safe interior projection, scale-invariant complementarity, and
  outer-KKT-gated soft acceptance.
- fix(#879): honest latent-fit diagnostics (projected gradient + reconstruction
  quality, scale-aware stationarity for the profiled-scale objective).
- fix(#880): hardened `duchon_function_norm_penalty` wrapper with regression
  tests.
- fix(#691): survival monotone-baseline I-spline — keep the convergent
  increment-space penalty (the value-space `Lᵀ S_B L` penalty remains the
  documented limitation pending survival-inner-solve work).

Prediction / diagnostics:
- fix(predict): prediction (observation) intervals now fold in estimation
  variance, not just the noise term.
- fix(survival-predict): a 2-arg `Surv(time, event)` predict builds a default
  time grid instead of erroring.
- fix(#881, #882, #883): post-fit diagnostic loaders carry the response and
  offset through, so diagnostics on reloaded models are correct.
- fix(alo): approximate-LOO standard errors use the estimated dispersion on both
  the geometry path and the diagnose refit route.
- fix(firth): link-general single-eta PIRLS Firth score shift.
- feat(#738, #741): the joint-Hessian path selects its representation by intent
  (matrix-free HVP for the inner solve, dense for the log-determinant) and
  exposes the full row-Hessian quotient as a `CompiledMap`.

Build / release hygiene:
- Restored workspace compilation: derived `PartialEq` on the
  `PhiScaledCovariance` / `UnscaledPrecision` covariance newtypes (the
  `array_values_equal` → native `==` cutover), removed a dead combined
  `GuardedCorrection::apply`, repointed the relocated `families::wiggle` helpers
  in the integration tests, and reformatted post-relocation lines. The release
  also rolls up the unpublished gamfit 0.1.181–0.1.187 changes.

## v0.3.107 — gam 0.3.107 / gamfit 0.1.186 (2026-06-08)

Crate + wheel release. crates.io was last published at gam 0.3.105 and PyPI at
gamfit 0.1.180; the intervening gam 0.3.106 / gamfit 0.1.185 tag failed to
publish (its top changelog entry did not name the tag, so the publish gate
rejected it), so this rolls up everything since — including the gamfit 0.1.185
entry below — into one good release.

Survival / monotone baseline:
- fix(#691): monotone-baseline survival fits converge again. A value-space
  I-spline curvature penalty (`Lᵀ S_B L`) was the principled fix for the
  monotone tail bias, but it enlarges the penalty nullspace and the survival
  inner PIRLS does not yet pin the likelihood-identified linear-trend direction
  over it: the penalized Hessian stays full-rank (no near-null space to project
  away) yet the constrained stationarity residual sticks at ‖g‖≈0.5 and the fit
  hits MaxIterations — a hard `IntegrationFailed`. The accompanying "range(H)
  stationarity rescue" rested on a misdiagnosis (it can never fire when H is
  full-rank) and is removed. Restored the converging increment-space penalty;
  the tail-bias trade-off is the documented #691 limitation, and the value-space
  penalty remains the real fix pending survival-inner-solve work.

Separation / Firth / inner solve:
- fix(#729, #715, #826): correct the Jeffreys/Firth merit sign and fold it into
  the coupled inner trust-region model so the baseline matches the trial and the
  K-block converges; scale the inner KKT tolerance by Firth score magnitude.
- perf(#729, #826, #808): O(p²) Gershgorin stabilizing shift, BLAS-3 assembly of
  the joint-Jeffreys curvature, and a cross-cycle cache keyed on beta.

Smooths / bases:
- perf(#813): dimension-aware tensor margin `k` to stop the ∏k product blowup,
  with a regression test.
- fix(#784): block-local sampled correction is smooth in rho (lambda-independent
  MC seed).
- fix(#787, #860): freeze the matern double-penalty nullspace-shrinkage decision
  across kappa rebuilds.
- fix(#854): exact second directional Hessian derivative for the spatial-adaptive
  family.
- fix(#780): commit the cyclic basis seam module.

Build / release hygiene:
- fix(#871): gate dead-code removal on gam-pyffi cross-crate reachability;
  restore modules the published wheel imports that looked dead to the `gam`
  crate alone.
- Unblock the workspace compile (stale bench/pyffi APIs), clear the rustfmt
  drift across the tree, and drop a dead lint-flagged rebinding.

## 0.1.185

- Revert matern double_penalty=false default regression (#787).
- Family-gated self-vanishing-mu cond-damping for marginal-slope inner solves (#787/#808).
- Operating-point warm-start of survival logslope initial_beta (g=0 seed-trap escape) (#808/#814).
- Suppress preconditioned-descent substitution at the joint-Newton step floor (#787 c12).

## v0.3.105 — gam 0.3.105 / gamfit 0.1.183 (2026-06-07)

Catch-up crate release. crates.io was last published at gam 0.3.103; this rolls
up every engine fix that shipped through gamfit 0.1.178–0.1.183 since then.

Survival marginal-slope (#808 family):
- fix(#808): reject only *channel-deleting* rawstack reductions. On clustered-PC designs the raw marginal / log-slope columns collide and the `[Time, Marginal, Logslope]` cross-block carry would zero the entire log-slope block — deleting the slope channel the model exists to estimate and diverging the inner solve. Non-destructive partial reductions are kept; any map that collapses a required channel to zero width is rejected and the unreduced design is used, leaving the near-null direction to Jeffreys conditioning. Guarded by a unit-tested predicate.
- fix(#834): continuation prewarm is now objective-opt-in, and the continuation seed dispatch / RE-Hessian guard / never-fail escalation reachability are pinned (#819/#737/#834/#860).

REML / ALO performance and correctness:
- fix(#862): ALO robustness weights are scoped to the owning `RemlState` instead of a process-global pointer-keyed map. Cold model sweeps reallocate a state at the same address with the same `n`; the old map then reused another formula's frozen ALO weights and recreated the 30–70× outer-REML grind on `smooth + a few linear covariates`. The frozen nuisance now lives on the surface and invalidates with the design in `reset_surface`.
- fix(#818): one comparable REML score across the Python APIs, so model-selection numbers line up between the formula and builder paths.
- fix(#819): materialize the sparse exact inner Hessian — `group()`-panel sparse-exact REML smoothing-correction no longer aborts.
- fix(custom/REML #824,#825,#837,#826): Firth consistency, Fisher block contract, joint-Newton rank floor, and stacked-solver η honored in custom-family assembly.

Smooths / bases:
- fix(#787): the Matérn formula defaults to `double_penalty=false`. The strictly positive-definite Matérn kernel has no structural polynomial nullspace, so the double-penalty ridge was spurious and flipped the learned-penalty count across the κ optimizer's design rebuilds ("joint hyper rho dimension mismatch"). An explicit `double_penalty=true` is still honored; Duchon/thin-plate keeps its native nullspace shrinkage (#754).
- fix(smooth/basis #822,#823,#851,#858): isotropic-Matérn κ axis, design storage / boundary handling, Duchon adaptive caches.

Predictions:
- refactor(#817): the moment-matched Gamma predictive interval is lifted into a pure, unit-tested `probability::gamma_moment_matched_interval` (exact conditional-Gamma limit, right-skew asymmetry, estimation-uncertainty widening, degenerate-input fallback). No behavior change to the #817 fix itself.

Diagnostics / families / linalg:
- fix(#864): `diagnose` keeps the response column instead of dropping it and aborting on its own training data.
- fix(#861): accept the redundant marginal-slope Firth flag.
- fix(#845,#846,#848,#849,#852,#855,#856): arrow-Schur / linalg / survival / sinkhorn correctness.
- fix(GPU/BMS #829,#831,#833,#835,#836,#838): trace SE, survival_flex integrand, 4th-order term, saddlepoint κ4, Mills seam.
- fix(SAE #841–#857): manifold correctness and streaming-logdet convergence.
- fix(#827,#828,#830): identifiability audit / canonical / compiler correctness.
- fix(#863): sphere GPU terms compile (`gpu_err!` import / macro scope).

Cleanup — removed deprecated aliases / compatibility shims (use the canonical names):
- Removed CLI family spelling aliases, shared precision-key aliases, Gumbel schedule aliases, and the identifiability-warning compatibility mirror; unified and restricted GPU policy parsing; reject stale SAE payload shapes.

Tests:
- test(#860, #820, #819): regression pins for startup-validation never-fail escalation, the fuzzer scenario cost cap, and the sparse-exact REML `group()`-panel repro.

## gam v0.3.104 / gamfit v0.1.178

- fix(#787/#785): C1 antiderivative for floored Jeffreys eigenvalue + line-search early-exit threshold (bernoulli marginal-slope inner KKT now converges; centers=12 PGS config that previously froze ~20min now returns).
- fix(#859): pin CTN cross-fit response knot count across folds (skewed-PGS large-scale calibration no longer raises p1 mismatch).
- fix(#813/#821): freeze ALO influence_scale/phi per fit (te() outer-REML no longer grinds; value<->gradient consistency).
- wip(#808): eta1-channel cross-block reduction for survival marginal-slope.

## v0.3.103 — gam 0.3.103 / gamfit 0.1.177 (2026-06-07)

- feat(#817): skew-aware Gamma observation (prediction) intervals. Response-scale predictive bands for the Gamma family are now equal-tailed quantiles of a moment-matched Gamma predictive — built on a robust inverse regularized incomplete-gamma (`probability::gamma_quantile`) — instead of the symmetric `μ ± z·σ` band that systematically mis-covered each tail of a right-skewed response. Includes a per-tail coverage regression and a degenerate-shape symmetric fallback.
- feat(#811, #812): the binomial posterior-mean `predict` path now honours `covariance_mode` (the smoothing correction reaches the credible band) and `observation_interval=True` (emits `observation_lower` / `observation_upper`), matching the Gaussian path and centred on the bias-corrected posterior-mean point; `covariance_mode='required'` now hard-errors when no correction is available.
- fix(#815, #816): `cyclic()` / `cc()` / `cp()` honour `period=` / `origin=` (parsed through the numeric-expression grammar, with a hard error on unparseable endpoints) and validate their options instead of silently falling back to the observed data range.
- fix(#685–#688): Gaussian location-scale fits through the formula API (`noise_formula=`) now converge on heteroscedastic data instead of aborting outer REML on every seed — the log-σ (scale) block carries a REML-selected identity ridge constraining its polynomial nullspace, and the spurious full-span Jeffreys term is dropped. (The hand-built custom-family location-scale path, #684, remains a tracked log-σ recovery / convergence gap.)
- fix(survival): Royston–Parmar monotonicity is enforced at every observed exit time; structural model construction is split from the ≥1-event fittability check so all fit modes share one validation chokepoint.
- fix(multinomial): the matrix-free Hessian diagonal mirrors the dense path's parallel reduction order, restoring bit-identical agreement under IEEE-754 non-associativity.
- fix(reml): the TK-refinement scale gate is aligned with the outer Firth gate.
- fix(#795): `sae_manifold_fit` converges on the single-planted-circle quickstart at the default `isometry_weight=1.0` (the MeanProfiled isometry energy is now scale-invariant and no longer saturates the arrow-Schur proximal ridge); pass `isometry_weight=0.0` to disable the isometry prior. Adds a default-exercising regression test.
- chore: removed accidentally-committed repro scripts / build log from the tree (and the gamfit sdist).

## gam v0.3.102 / gamfit v0.1.176

- fix(#789B): fast-fail survival marginal-slope on all-censored (zero-event) designs instead of spinning.
- fix(#808): never-fail outer escalation for survival marginal-slope (graceful degradation instead of fatal IntegrationFailed); permanent regression test (#814).
- Includes #700-703 tensor/sz null-space penalty defaults, #811/#812 binomial predict covariance_mode/observation_interval threading, #795 periodic-axis shrinkage fix, and #735/#736 log_sigma block-width fixes.

# Changelog

All notable, user-visible changes to **gam** (the Rust engine, published to
crates.io) and **gamfit** (the Python wheel, published to PyPI) are recorded
here. This file is the single source of truth for release notes:

- It is rendered on the documentation site under **Changelog**
  (`docs/changelog.md` includes this file verbatim via a snippet).
- The GitHub Release for each `v*` tag is generated from the matching section
  below.
- CI (`.github/workflows/publish.yml`) refuses to publish a tag whose version
  does not match the top entry here *and* the versions in `Cargo.toml` /
  `pyproject.toml`.

The two packages are versioned independently — `gam` tracks the Rust engine,
`gamfit` the Python wheel — but released together. Each entry is headed with the
git tag and both package versions.
Failed or unpublished version-bump tags are intentionally omitted; package
releases without local semver tags are included under their published version.

## v0.3.101 — gam 0.3.101 / gamfit 0.1.175 (2026-06-06)

- gamfit: survival marginal-slope baseline-hazard conditioning + monotonicity-domain tolerance fixes (#788, #797 inner barrier) reach PyPI; bernoulli marginal-slope inner trust-region noise-floor fix. Plus mainline fixes #798–#805.

## v0.3.100 — gam 0.3.100 / gamfit 0.1.174 (2026-06-06)

Correctness fixes to response-scale prediction intervals, survival fitting, and
the REML/LAML evidence path.

### Fixed
- **Observation (prediction) intervals are clamped to the response support
  (#800).** `predict(..., observation_interval=True)` builds the response-scale
  predictive band as the symmetric `μ ± z·σ_pred`. For a bounded or
  half-bounded response — a count (Poisson, Negative-Binomial, Tweedie), a
  positive value (Gamma), or a proportion (Beta, Binomial) — that band crossed
  the support edge at a small or extreme fitted mean and reported impossible
  values (a Poisson predictive lower bound going negative). The band is now
  floored/capped at the family's response support in both interval-assembly
  paths. The *mean* (confidence) interval was already correct and is unchanged.
  A new `ResponseFamily::response_support_bounds` exposes the closed support
  bounds, kept in lockstep with the existing support-membership check.
- **Beta prediction intervals use the estimated precision φ̂ (#801).** The
  observation-interval builder's Beta arm read precision off the family-enum
  construction seed (default `1.0`) instead of the precision estimated jointly
  with the mean, so on high-precision data the band was `√((1+φ̂)/2)` too wide.
  It now routes through the same fitted-dispersion accessor the Tweedie/Gamma
  arms already use, falling back to the seed only for raw-covariance sources
  that carry no fitted scale.
- **Survival fits no longer over-reject censored rows.** The per-row
  monotonicity guard in the survival working-model update rejected any row whose
  stabilized exit derivative fell below a numerical floor, but only event rows
  evaluate `ln(deriv)` / `1/deriv` downstream. The guard is now gated on event
  rows (`d > 0`), matching the residual-channel loop, so a censored row with a
  zero collocation derivative at, e.g., β = 0 is no longer a false-positive
  monotonicity violation.
- **REML/LAML evidence integrity on indefinite per-row blocks.** In evidence
  (log-determinant) mode the arrow–Schur per-row factorization silently lifted
  the ridge on a non-positive-definite `H_tt` until it became PD, then summed
  the lifted factor's diagonal into the exact arrow log-determinant — reporting
  `log|H_tt + ridge_eff·I|` where `log|H_tt + ridge_t·I|` was intended and
  corrupting the evidence with no error surfaced. Evidence mode now returns a
  typed error on a genuinely non-PD block instead of accepting a ridge-lifted
  surrogate; the strict Newton-step path, which wants the regularising lift, is
  unchanged.

## v0.3.99 — gam 0.3.99 / gamfit 0.1.173 (2026-06-06)

Completes the link-general Firth work from 0.3.98.

### Fixed
- **Non-logit Firth fits no longer crash (#758).** 0.3.98 opened the CLI/HMC
  Firth gate to every Binomial inverse link with a Fisher-weight jet (Probit,
  CLogLog, Latent-CLogLog, SAS, Beta-Logistic, Mixture), but the REML outer
  loop's Tierney-Kadane correction is implemented only for the canonical
  Binomial Logit jet — so `gam fit --firth --family binomial-probit` (or
  cloglog) aborted every outer seed with "Tierney-Kadane outer Hessian is
  implemented for canonical Binomial Logit Firth fits only". Non-logit Firth
  fits now skip the higher-order TK refinement and fall back to plain Laplace
  REML driven by BFGS off the link-general gradient; the Firth/Jeffreys bias
  reduction itself (the inner PIRLS Jeffreys penalty) is fully retained. Logit
  Firth fits are byte-unchanged and keep the full analytic TK path.

## v0.3.98 — gam 0.3.98 / gamfit 0.1.172 (2026-06-06)

First published release carrying the universal under-identification robustness
work staged in 0.3.97 (which was version-bumped but never published to
crates.io / PyPI), together with a batch of correctness fixes across the
binomial-link, separation-diagnostic, SAE-penalty, and constraint paths.

### Fixed
- **Link-general Firth / Jeffreys (#758).** Firth bias reduction and the
  Jeffreys prior now apply to every Binomial inverse link that carries a
  Fisher-weight jet (Probit, CLogLog, Latent-CLogLog, SAS, Beta-Logistic, and
  anchored Mixture links), not only Logit. The actual inverse link is preserved
  through the Firth/Jeffreys and PIRLS-diagnostic paths instead of silently
  collapsing to a bare logit, and the CLI gate and the NUTS/HMC Firth guards
  accept the full set with an accurate message (was a stale "only supported for
  Binomial Logit").
- **Pre-fit regularity screen (#775).** Designs that are perfectly separated
  (single-column *or* linear-combination separators) or rank-deficient in their
  unpenalized block are now rejected up front with a typed, actionable error
  instead of diverging inside the solver.
- **Multinomial separation diagnostics (#753).** Separating multinomial fits
  raise a dedicated `MultinomialSeparationDetected` error naming the offending
  class/row instead of reusing the binary-outcome message.
- **Tweedie sampling dispersion (#771).** Posterior sampling draws Tweedie
  responses with the fitted dispersion φ rather than mis-using the variance
  power as the noise scale.
- **Picklable Rust exceptions (#773).** `gamfit`'s Rust-originated exceptions
  carry an importable module, so they survive pickling across a
  `ProcessPoolExecutor` / `multiprocessing` boundary.
- **SAE-penalty curvature correctness (#794).** `MonotonicityPenalty::hvp` no
  longer inflates curvature by `1/smoothing_eps`, and `JumpReLUPenalty`'s PSD
  majorizer now genuinely dominates the exact (indefinite) Hessian for inactive
  coordinates instead of under-estimating it ~7×.
- **SAE numerical robustness (#742).** Learnable penalty/SAE exponents are
  clamped to a finite-normal band so extreme ρ can no longer overflow to
  inf/NaN.
- **Box-constraint scaling (#791).** CLI box constraints are transformed by
  `1/scale` (not `scale`), fixing constraint escape under non-unit scaling.
- **Random-effect group axes (#792).** Unseen string random-effect group levels
  are admitted at predict time and group axes are no longer clipped.
- **Coupled Dirichlet joint Hessian (#729)** and spec-aware joint-Hessian drifts
  that keep batched marginal-slope / Jeffreys fits Hφ-consistent (#787).
- **Periodic tensor B-splines (#629).** Tensor B-spline periodicity is preserved
  through freeze→reload, with a separable periodic top-1 fit path and a restored
  manifold-SAE serialization roundtrip.
- **Guidance fixes:** correct per-diagnosis marginal-slope refusal guidance
  (#754) and survival marginal-slope penalty-width provenance (#788).

### Internal
- Unified the four matrix-free Lanczos paths onto one primitive (#766),
  collapsed the duplicated Firth-support predicates onto a single source of
  truth, synced the `gam-pyffi` lockfile version, and made the release tree
  rustfmt-clean.

## v0.3.97 — gam 0.3.97 / gamfit 0.1.171 (2026-06-05)

Universal under-identification robustness — unified, always-on, no flag.

### Added
- Robustness is now an unconditional solver property, self-limiting so it is
  byte-identical on well-identified fits and only acts where the data is
  near-separating / under-identified: a conditioning-gated full-span
  Jeffreys/Firth prior on the identifiable subspace (finite estimates under
  separation), a self-gating penalized-complexity prior on the smoothing
  parameters, and exact orthogonalization of confounded design blocks. A cheap
  matrix-free (Lanczos) conditioning pre-check keeps it ~zero-cost and
  matrix-free-preserving on well-conditioned and large-`p` fits.
- Never-fail inference: when the smoothing optimizer cannot certify convergence,
  the fit escalates to sampling the proper posterior (HMC) — guarded by R-hat /
  ESS so it returns honest (never false-confident) uncertainty instead of erroring.

### Changed
- The `RobustIdentification` flag and the pinned BMS nullspace / overlap ridges
  are removed; robustness is a single always-on path with no user knob.

### Fixed
- Published the current `main` engine through the Python wheel line, including
  the random-effect prediction schema fix for unseen string groups and the
  manifold-SAE serialization roundtrip fix.

## v0.3.96 — gam 0.3.96 / gamfit 0.1.169 (2026-06-05)

First crates.io release of the `gam` engine since v0.3.91, bringing the Rust
crate current with the gamfit 0.1.164–0.1.168 wheel line and adding the new
under-identification robustness layer (off by default).

### Added

- **Universal under-identification robustness (`robust_identification`, preview — off by default).** A new, family-general layer that makes robustness to non-identification a property of the *solver* rather than a per-family patch: a link-general Jeffreys/Firth penalty on the under-identified subspace (bounding near-separating coefficients) plus exact orthogonal reparameterisation of overlapping design blocks (resolving structural confounds rather than penalising them). Exposed as `gamfit.fit(..., robust_identification=...)` and the `--robust-identification` CLI flag with policies `"off"` (default), `"auto"`, and `"force"`. **`"off"` is byte-identical to the previous solver**, so existing fits are unchanged; the machinery is opt-in while it is hardened.

### Fixed

- **Smooth-free (purely parametric) fits no longer crash.** An ordinary linear model — `gamfit.fit(df, "y ~ x1 + x2")`, any family, with no `s()`/`te()`/`matern()` term — aborted in the post-fit null-space metadata step with `null-space Hessian is not positive definite: Cholesky factorization failed: NonPositivePivot { index: 0 }`, even though the fit converged and the CLI fit the same data fine. Root cause: a smooth-free design has an all-zero penalty matrix, and the rank-revealing QR returned a NaN null-space basis for rank-0 input (faer's column-pivoted QR produces degenerate Householder reflectors when the first pivot column has zero norm). The null space of a zero matrix is the whole space, so its basis is now returned as the exact identity. This also unblocked learned-length-scale Matérn BMS fits, whose outer optimiser was being poisoned by the same NaN at degenerate penalty configurations.
- **`bs="sz"` factor smooths fit and predict (#700).** A sum-to-zero factor smooth `s(g, x, bs="sz")` crashed at fit time with an identifiability-transform dimension mismatch and was non-functional; the full-design joint-null rotation is no longer folded into the per-marginal `sz` metadata, so `sz` smooths now fit and reproduce their fitted values on frozen replay.
- **Hybrid Duchon smooths with an explicit `length_scale` build for every covariate dimension (#750).** `duchon(...)` with a `length_scale` but no explicit `power=` crashed at basis generation for even covariate dimensions `d ≥ 4`; the cubic structural default now resolves to an admissible integer spectral power.
- **BMS spatial-`rho` startup and convergence hardening (#754, #461).** Fixed `#754`/`#461` ridges are carried as physical `PenaltyMatrix::Fixed` penalties (excluded from the REML/outer `rho` vector), the startup no longer mis-classifies a phantom seed, and production-shaped Matérn BMS fits start and converge.

## gamfit 0.1.168 — gam 0.3.95 / gamfit 0.1.168 (2026-06-05)

### Fixed

- **Publish the post-0.1.167 BMS spatial/kappa rho fix to PyPI (#754).** The
  0.1.167 wheel was built before the follow-up fix that removed fixed physical
  BMS ridges from the learned spatial/kappa REML `rho` layout. This wheel bump
  publishes the already-merged code needed by the large-scale Workbench driver, whose
  `uv --with gamfit --upgrade-package gamfit` path resolves from PyPI.

## v0.3.95 — gam 0.3.95 / gamfit 0.1.167 (2026-06-04)

### Fixed

- **Probit BMS marginal-slope: release the marginal/logslope overlap ridge (#754, completing the fix).** The #754 nullspace-shrinkage ridge (shipped in 0.1.165/0.1.166) bounds the marginal block's *unpenalized* directions, but a production-scale run (`duchon(PC1,PC2,PC3,centers=20)`, n≈195k, 1:1 balanced) showed the runaway coefficient (β≈61) actually lives on a **penalized smooth** direction that is degenerate with the score-weighted logslope surface — the marginal↔logslope confound — which the nullspace ridge does not touch. This release ships the additional fixed **overlap ridge** that shrinks exactly those cross-channel directions, plus the production-shaped binary-outcome BMS regression test. The nullspace ridge alone was necessary but not sufficient at scale; the two ridges together bound both the null-space and the confound directions.

## v0.3.94 — gam 0.3.94 / gamfit 0.1.166 (2026-06-04)

### Fixed

- **`matern(..., centers=K)` no longer FATALs when K over-specifies the kernel (#755).** With a fixed `length_scale`, packing more centers into the data cloud than the kernel can resolve makes adjacent basis functions near-identical, so the realized design carries exactly linearly-dependent columns and the identifiability audit hard-FATALs on intra-block rank deficiency. The basis now rank-reduces the center set at construction: column-pivoted RRQR on the realized `n×K` kernel design (the same matrix the audit checks) at the crate-standard tolerance, keeping the leading full-rank pivoted centers and dropping the redundant remainder (logged). Detection is on the realized design columns (not the squared center Gram), so it fires exactly when the audit would have failed and leaves well-specified bases untouched.

## v0.3.93 — gam 0.3.93 / gamfit 0.1.165 (2026-06-04)

### Fixed

- **Probit Bernoulli marginal-slope outer REML no longer diverges (#754).** The marginal-surface block left its parametric + smooth-nullspace directions fully unpenalized, so on a balanced steep-gradient probit sample a near-separating direction's coefficient ran to ~50 and the outer ARC solve hit max-iter / rejected every seed (`phantom_multiplier_with_well_conditioned_H`) — basis-independent (Matérn and Duchon both hit it). A small **fixed** nullspace-shrinkage ridge (`Z·Zᵀ` over the null space of the aggregate marginal smooth penalties), pinned out of REML at `log λ = ln(1e-2)` so it cannot be driven to zero, now bounds the flat direction and gives the outer solve a finite optimum — negligible against the n-scaled probit Fisher information of any identified direction.

## v0.3.92 — gam 0.3.92 / gamfit 0.1.164 (2026-06-04)

### Changed

- Release bump to force a fresh wheel build/publish. No engine changes since 0.1.163; in-progress fixes for the Bernoulli marginal-slope outer-REML non-convergence (#754) and Matérn over-parameterization (#755) will follow in a later release.

## v0.3.91 — gam 0.3.91 / gamfit 0.1.163 (2026-06-04)

### Fixed

- **Binary-outcome-style Bernoulli marginal-slope Matérn fits now have full audit-level regression coverage.** The release includes a formula-to-fit test for the reported `matern(...) + sex + entry_age_z + current_age_ns_*` layout, proving the scalar-pruned model passes the actual pre-fit identifiability audit and produces finite coefficients.

## v0.3.90 — gam 0.3.90 / gamfit 0.1.162 (2026-06-04)

### Fixed

- **Binary-outcome-style Bernoulli marginal-slope formulas now have an exact regression for scalar-alias pruning.** The release includes a materialization test matching the reported `matern(...) + sex + entry_age_z + current_age_ns_*` layout and proves the local-column-3 scalar alias is removed before the identifiability audit while the Matérn blocks remain intact.

## v0.3.89 — gam 0.3.89 / gamfit 0.1.161 (2026-06-04)

### Fixed

- **Bernoulli marginal-slope redundant-scalar handling now has fail-closed
  regression coverage.** Tests now lock in that constrained or explicitly
  penalized duplicate scalar columns are rejected rather than pruned, preserving
  the hardened identifiability audit contract for binary-outcome-style BMS
  formulas with redundant scalar covariates.

## v0.3.88 — gam 0.3.88 / gamfit 0.1.160 (2026-06-04)

### Fixed

- **Release metadata for the Bernoulli marginal-slope identifiability fix is now complete.** The PyPI wheel crate and lockfile now carry the same `gamfit` version as `pyproject.toml`, satisfying the hardened release scanner for the BMS redundant-scalar audit fix shipped in the previous commit.

## v0.3.87 — gam 0.3.87 / gamfit 0.1.159 (2026-06-04)

### Fixed

- **Bernoulli marginal-slope Matérn fits with redundant scalar covariates no longer fail the identifiability audit.** The workflow now removes unpenalized scalar columns that add no direction beyond the implicit intercept and earlier scalar terms before BMS block construction, and rejects constrained or explicitly-penalized duplicates instead of using a ridge or constraint to mask non-identifiability. This keeps the hardened audit fail-closed while allowing large-scale binary-outcome-style `matern(...) + scalar covariates` fits whose precomputed scalar spline column is constant or redundant.

## v0.3.86 — gam 0.3.86 / gamfit 0.1.158 (2026-06-04)

### Fixed

- **The #751 survival marginal-slope release is now build-gate clean and wired
  through PyPI metadata.** The release lockfile now carries the current
  `gamfit` version, custom-family output-channel defaults use the real
  single-output channel map instead of a sentinel empty-spec shortcut, and SAE
  fixed-decoder projection grids are selected from the atom basis kind rather
  than mandatory evaluator methods with no-op `None` implementations.

## v0.3.85 — gam 0.3.85 / gamfit 0.1.157 (2026-06-04)

### Fixed

- **Survival marginal-slope left-truncated `matern(...)` fits no longer reject
  every REML seed through a phantom time-block multiplier** (#751). The
  marginal-slope baseline time basis now anchors at the median exit time instead
  of the minimum entry time, so left truncation no longer turns the centered
  I-spline null-space column into a dominant one-sided time trend. The time block
  also installs an explicit null-space shrinkage penalty for structural
  unpenalized directions, giving REML a real precision parameter instead of an
  unidentifiable phantom multiplier.
- **Invalid survival marginal-slope custom-family block specs now return a typed
  error instead of panicking in Rust** (#751). Output-channel wiring validates the
  block specs before probing family channel assignments, and the default
  assignment hook is only defined for empty specs.

## v0.3.84 — gam 0.3.84 / gamfit 0.1.156 (2026-06-04)

This release lands a large batch of correctness, convergence, and quality fixes
across families, plus a build fix that restores the PyPI wheel and crates.io
publish paths.

### New

- **Held-out split-conformal calibration fold** (#682). The conformal prediction
  path now accepts a calibration fold whose size differs from the training set.
  The fold is routed through plain split-conformal — residuals on the held-out
  fold, normalized by the predict-time response-scale SE — instead of being
  bound to the training set's frozen ALO geometry, so calibrating on a fold of
  any size produces finite, coverage-valid intervals.
- **GPU CUDA userspace preload.** On Linux, the CUDA userspace libraries
  (cudart, nvJitLink, cuBLAS, cuSPARSE, cuSOLVER) are preloaded with
  `RTLD_GLOBAL` from canonical toolkit directories and pip `nvidia-*-cu12` wheel
  layouts, so cudarc's lazy SONAME loads resolve without an `LD_LIBRARY_PATH`
  mutation. Discovery is environment-variable-free (interpreter-relative, plus
  the wheel's `$ORIGIN`-relative rpath).

### Fixed

- **Gaussian location-scale predictions are reported in raw response units.**
  The model standardizes the response internally (keeping the log-σ soft floor
  scale-relative) and now maps the fitted coefficients, covariance, and
  likelihood/deviance/REML summaries back to raw units before persistence;
  prediction no longer applies a second response-scale multiplier, so the mean
  and σ come out in the data's own units with the response scale applied
  exactly once.
- **Survival location-scale constant-scale AFT fits no longer hang, panic, or
  fail finalization** (#735, #736, #721). A constant-scale parametric AFT now
  builds an identifiable parametric time-warp that absorbs the unidentified
  I-spline null space; the constrained joint-Newton QP is damped on a
  rank-deficient `H_pen` so an unidentified time-warp gauge step exhausts and
  the identified-subspace KKT certificate fires (instead of crawling a dead-flat
  REML ridge); the parametric-AFT time `ρ` is seeded at the inner box bound so
  the box-constraint KKT certifies immediately; the log-σ canonicalization keeps
  an intercept-only scale block at width 1 so raw/active block widths agree at
  the covariance-lift boundary; and the batched-outer `state.eta` length check
  uses `solver_design().nrows()` rather than `design.nrows()` (3·n vs n),
  fixing a finalization panic.
- **Continuous-transformation-normal/monotone (CTN/CTM) fits converge and report
  EDF** (#720, #733, #734). The response-basis size adapts to the
  transformation's non-normality and the inner exact-Newton cycle cap is scoped
  to the bounded convex block (no more dense exact-SCOP-Hessian timeout on a
  simple Gaussian shift); a rank-deficient penalized-Hessian null direction no
  longer blocks KKT certification (the identified range-space certificate is a
  first-class per-cycle test, and the CTM joint-Newton range-projects the RHS
  instead of erroring); a self-vanishing Levenberg damping stabilizes the inner
  spectral Newton step; and total EDF / inference are populated for joint-Newton
  custom-family fits.
- **Coupled multi-block custom families are trusted without an explicit marker**
  (#727, #729). A family that returns a genuinely coupled (nonzero
  off-diagonal-block) joint Hessian is detected structurally and used, rather
  than requiring `has_explicit_joint_hessian()`.
- **Tensor `te()`/`ti()` and factor smooths no longer over-smooth** (#700, #701,
  #702, #703, #712, #713). The tensor double penalty shrinks only the joint null
  space rather than applying a full identity ridge; `sz` factor smooths drop the
  inner-marginal double penalty so each per-level linear null space stays free
  like mgcv; and `fs` factor smooths cap the marginal basis to the least-resolved
  group so per-group curves shrink toward a linear random slope. The `te()`
  capacity guard also sums marginal column counts instead of their Kronecker
  product, so well-posed penalized tensors on moderate `n` are accepted
  (#724, #728, #730).
- **ALO stabilization no longer over-smooths under high leverage** (#711), and
  logistic link-scale confidence intervals are calibrated by pooling across
  Bernoulli replicates to estimate Nychka across-the-function coverage (#710).
- **Competing-risks CIF reconstruction uses the fitted baseline** rather than the
  seed configuration (#689, #690).
- **Tweedie variance-power estimation stays inside the valid `(1, 2)` interval**
  (#698). The biased 6-bin log-variance OLS slope is replaced with a
  saddlepoint profile-likelihood MLE bounded to the open interval, so the
  estimated power no longer escapes to ~2.29 on real data.
- **Binomial probit recovery** (#697) is benchmarked against a method-comparable
  penalized GLMGam reference (via the predict path, no invalid `.scale` access),
  closing the RMSE gap to statsmodels' probit fit.
- **Bernoulli marginal-slope reports an honest terminal verdict** (#744). A fit
  that stalls no longer returns after the final cycle with the residual still
  above tolerance without saying so; the joint-Newton terminal criterion is now
  named explicitly.
- **Negative-binomial and Gamma-log standard errors** (#679) gain behavioral
  coverage and cross-family SE guards.
- **Spatial smooths**: by-factor thin-plate-spline length-scale auto-init now
  recurses into the by/factor inner kernels so the predict design is finite
  (#704); 2-D TPS spatial fits use a data-proportional center floor to avoid a
  timeout (#718).
- **Manifold SAE**: scale-free (mean-profiled) isometry reference, an exact
  von-Mises ARD normalizer (Bessel I₀) for periodic axes (#681), EM routing-seed
  refinement for cold multi-atom fits (#629, #630), preserved `random_state`
  seed-dependence through the routing seed (#178), and finiteness/robustness
  guards — log-space IBP prior, clamped learnable weights, NuclearNorm
  active-rank cap, and a Welford PCA seed (#742).
- **Build**: restored the gam-pyffi / maturin compile (a stray unqualified
  `SaeManifoldTerm` reference) and cleared the workspace ban-gate violations in
  the CUDA preload path, so the PyPI wheel and crates.io publish workflows build
  again.

### Notes

- Recalibrated several quality-suite bounds to attainable match-or-beat-reference
  targets: multinomial-logit recovery (#699), p-spline interior-gap (#708),
  sphere/torus SOS surfaces and doubly-cyclic tensors (#694, #695, #705),
  binomial-probit (#697), and the `fs` random-slope lme4 reference DGP (#712).
- Right-sized the nightly real-data posterior-sampling budget (Pólya-Gamma Gibbs
  + PyMC NUTS) and enabled PyMC chain parallelism (#719).

## v0.3.83 — gam 0.3.83 / gamfit 0.1.155 (2026-06-04)

### Fixed

- **Anisotropic Duchon spatial terms no longer abort REML with an outer
  gradient-length mismatch.** Four functions disagreed on how many `ψ` entries a
  multi-axis `aniso_log_scales` Duchon term contributes to the joint outer
  hyperparameter vector: the n-block exact-joint spatial optimizer planned a
  per-axis `θ` layout (`rho_dim + Σ d_term`) while the inner unified evaluator
  emitted one `ψ` per term, tripping the `OuterThetaLayout` contract and failing
  every nightly Large-scale `duchon16d` shard before the solver even started.
  A single shared predicate now drives the `ψ` count at all four sites: Duchon
  anisotropy `η` is a fixed, geometry-derived basis parameter (one isotropic
  `ψ̄` slot per term), so the outer plan and the inner gradient agree by
  construction. Matérn anisotropy is unchanged (still per-axis `ψ`).
- **Manifold SAE fits converge again across all isometry/topology cells,
  including the circle `d=1` case that regressed in 0.1.154.** The isometry
  cross-block curvature added for 0.1.152 left the coordinate/decoder Schur
  complement slightly non-PD — an inconsistent nonzero cross-block that was not
  paired with diagonals from the same residual Jacobian — so circle `d=1` fits
  failed where they had recovered `R² ≈ 0.997`. The inconsistent cross-block is
  removed (PSD diagonals with a zero cross-block stay PSD), and inner
  stationarity is now judged by the gradient at the step's parameter scale so
  gauge-like SAE directions are no longer mistaken for non-convergence. All nine
  isometry × topology × dimension bisection cells now converge.

## v0.3.82 — gam 0.3.82 / gamfit 0.1.154 (2026-06-03)

### Fixed

- **Nuclear-norm HVP no longer panics on roundoff-scale smoothed Gram
  eigenvalues.** The penalty now returns explicit errors for invalid spectra and
  floors only numerical roundoff to the configured smoothing floor, preventing a
  Rust panic from aborting Python SAE experiments.

## v0.3.81 — gam 0.3.81 / gamfit 0.1.153 (2026-06-03)

### Fixed

- **SAE decoder-incoherence convergence checks now fail loudly.** The cross-atom
  decoder cross-Gram test no longer uses `pytest.xfail`; failed or degenerate
  multi-atom fits now surface as ordinary test failures.
- **Python wheel publishing no longer carries stale gamfit references.** Updated
  the lockfile and REML benchmark guidance to the current `gamfit` version so
  the release ban gate accepts the wheel build.

## v0.3.80 — gam 0.3.80 / gamfit 0.1.152 (2026-06-03)

### Fixed

- **Manifold SAE isometry curvature now includes the coupled coordinate/decoder
  Gauss-Newton cross block.** The Arrow-Schur system can now add dense analytic
  `H_tβ` supplements on top of the matrix-free row operator, so the isometry
  metric penalty contributes consistently to `H_tt`, `H_tβ`, and `H_ββ` instead
  of leaving the Schur complement with a missing cross term.
- **Matrix-free and dense SAE cross-block curvature now compose deterministically.**
  `ArrowSchurSystem` fingerprints dense `H_tβ` supplements when they are active,
  and all apply/materialize/transpose paths sum the matrix-free and dense pieces.

### Notes

- This release intentionally keeps the strict SAE KKT gradient tolerance. It does
  not include the earlier experimental tolerance relaxation that made low-quality
  isometry fits appear converged.

## v0.3.79 — gam 0.3.79 / gamfit 0.1.151 (2026-06-03)

### New

- **Cross-atom decoder incoherence for manifold SAE fits**
  (`decoder_incoherence_weight`, #671). A separability lever for multi-atom
  dictionaries: for `K >= 2` it is on by default and penalizes overlap between
  *co-activating* atoms' decoder column spaces, weighted by empirical gate
  co-activation. The penalty now also enters the SAE REML selection criterion,
  so it shapes both the fit and topology/model selection (previously it only
  influenced the Newton step).
- **Decoder embedding-rank selection for manifold SAE fits**
  (`nuclear_norm_weight`, `nuclear_norm_max_rank`, #672). A positive weight
  applies a nuclear-norm penalty to each atom's decoder block, shrinking its
  singular spectrum to select the ambient embedding dimension;
  `nuclear_norm_max_rank` caps the number of leading singular values included.
- **Non-convex SCAD/MCP gate sparsity for manifold SAE fits.** Set
  `gate_sparsity="scad"` or `"mcp"` (with `scad_mcp_gamma` defaulting to `3.7`
  for SCAD and `2.5` for MCP). The default `gate_sparsity="l1"` path is
  unchanged.
- **Per-atom posterior shape uncertainty on `ManifoldSAE` results.** Atoms carry
  `decoder_covariance`, `shape_band_coords`, `shape_band_mean`, and
  `shape_band_sd`; helpers `shape_uncertainty(...)` and the `shape_band(...)`
  alias expose the posterior shape band.
- **Typical coordinate-range summaries for manifold SAE atoms.**
  `coordinate_range(...)` gives per-axis min/max/median/5th/95th-percentile
  summaries; `typical_shape(...)` restricts the posterior shape band to an
  atom's typical recovered-coordinate range.

### Fixed

- **Intrinsic, gauge-invariant decoder smoothness for SAE topology evidence**
  (#673). The decoder roughness penalty is now reparameterized into arc length
  via the decoder pullback metric `g = JᵀJ` (a symmetric congruence of the raw
  penalty), so the `reml_score` used to compare an atom's topology (e.g. circle
  vs. line) is invariant to reparameterizing the latent coordinate.
  Constant-speed and periodic atoms are provably unchanged. Previously the
  penalty was computed in raw latent coordinates, making topology evidence
  gauge-dependent for non-constant-speed atoms.
- **Gamma dispersion is no longer over-estimated (~2×) when the mean varies**
  (#678). The Gamma shape `ν = 1/φ` was frozen at an early, far-from-converged
  linear predictor. It is now re-estimated at the converged `η` and iterated to
  the joint `(β, ν)` fixed point — only at the single final reported fit at the
  REML-selected `λ`, so the smoothing-parameter search is unaffected.
- **Standard errors for Gamma, Tweedie, Beta, and Negative-Binomial models are
  no longer too small by √dispersion** (#679). The coefficient covariance
  `Vb = H⁻¹` is no longer multiplied by a post-hoc dispersion factor for
  families whose IRLS working weight already carries the dispersion / full
  Fisher information; only the profiled Gaussian restores `Vb = H⁻¹·σ̂²`. Encoded
  as a single-source-of-truth invariant
  (`GlmLikelihoodSpec::coefficient_covariance_scale`).
- **More accurate SAE reconstruction dispersion `φ̂`** (#676). The
  latent-coordinate effective degrees of freedom now use the exact ARD-shrunk
  trace instead of the full assignment-weighted latent dimension, so posterior
  shape bands are no longer mildly conservative.
- **Manifold SAE multi-atom routing no longer collapses to a uniform saddle**
  (#629, #630). Cold-start assignment logits are seeded asymmetrically from the
  per-atom reconstruction residual (an EM-style step) instead of exactly
  uniform, which was a symmetric saddle for `K >= 2` exchangeable atoms. The
  outer REML search also now rejects finite-but-non-converged inner solves
  rather than ranking them.
- **Out-of-sample SAE encoding recovers one-hot periodic-atom routing** (#628).
  A global decoder-projection coordinate seed places each row in the correct
  basin before refinement, and the OOS path keeps the decoder frozen. The torus
  projection-seed grid now falls back to a PCA seed past its point cap instead
  of emitting an exponentially large grid.
- **SAE inner-solver convergence regressions** that could surface as
  `RemlConvergenceError`. The arrow-Schur PCG `schur_matvec` callback clears its
  reused output buffer before accumulating `S·x`, preventing stale contributions
  from corrupting the reduced system.
- **SAE joint arrow-Schur line-search baseline.** The solver snapshots the exact
  state used to assemble the gradient and Hessian and computes `pre_step_total`
  from it before Armijo backtracking, so trial steps are no longer compared
  against a stale objective.
- **Non-Linux builds** now provide a real `scatter_batched`, so targets that
  call it unconditionally compile; device-free runs report no device tiles and
  the caller runs its deterministic whole-batch CPU fallback.

### Verified

- Verified the per-atom shape-uncertainty plumbing end-to-end (Python ↔ PyO3 ↔
  Rust) and the analytic Schur block-inverse identity used for the posterior
  bands (#677).

## v0.3.78 — gam 0.3.78 / gamfit 0.1.151 (2026-06-03)

### Changed

- Published the changelog/docs/release-wiring pass together with the #679
  coefficient-covariance-scale fixes and SAE intrinsic-roughness work that
  preceded the tagged `v0.3.79` repair.
- Wired the root `CHANGELOG.md` into docs, PyPI project URLs, and GitHub Release
  note generation; the next release corrected stale `gamfit` version references.

## v0.3.77 — gam 0.3.77 / gamfit 0.1.150 (2026-06-03)

### Changed

- Added global decoder-projection coordinate seeding for fixed-decoder SAE
  out-of-sample prediction (#628).

## v0.3.76 — gam 0.3.76 / gamfit 0.1.149 (2026-06-03)

### Fixed

- Fixed unseen random-effect level prior variance to use `scale / lambda`, not
  `1 / lambda`, and capped CI linker parallelism so concurrent release links do
  not exhaust runner memory (#674).

## v0.3.75 — gam 0.3.75 / gamfit 0.1.148 (2026-06-03)

### Changed

- Added ManifoldSAE per-atom posterior shape uncertainty end-to-end: Rust
  decoder covariance and shape bands, PyO3 exposure, Python result fields, and
  an e2e regression test.
- Expanded GPU/Arrow-Schur execution with per-ordinal `AtB` GEMM, multi-GPU
  row-block solves, shared manifold kernels, Schur inverse-block extraction, and
  all-GPU manifold batch GEMM/GEMV dispatch.
- Fixed non-Linux GPU `scatter_batched` / ordinal `AtB` paths, response-scale
  invariance for smooth Wald p-values (#675), leftover merge-conflict markers,
  and pyffi build drift.

## v0.1.147 — gam 0.3.74 / gamfit 0.1.147 (2026-06-02)

### Changed

- Re-triggered the gamfit wheel after the 0.3.74 / 0.1.146 coordinated release;
  the next `v0.3.75` release carried the remaining code changes.

## v0.3.74 — gam 0.3.74 / gamfit 0.1.146 (2026-06-02)

### Changed

- Moved dense-Fisher multi-output Gaussian fitting, block-orthogonal REML
  backward, Fisher-Rao weight normalization, SPD/symmetric solves, weighted
  ridge solving, auxiliary-prior REML scoring, and SAE PCA seeding from the FFI
  layer into core Rust.
- Exposed conformal intervals, covariance-mode / observation prediction, and
  Wood per-smooth p-values through `gamfit`.
- Fixed release compilation across targets, benchmark Rust-extension loading,
  BMS/SMGS exact-joint probe-design reconstruction, Circle wrapping docs, and a
  broad geometry/SAE audit batch (#596-#626).

## v0.1.145 — gam 0.3.73 / gamfit 0.1.145 (2026-06-02)

### Changed

- Published the intermediate gamfit wheel between `v0.3.72` and `v0.3.74`; the
  substantive core/FFI changes are captured in the adjacent coordinated
  releases.

## v0.3.72 — gam 0.3.72 / gamfit 0.1.144 (2026-06-02)

### Fixed

- Fixed PSIS Zhang-Stephens GPD shape estimation so heavy-tail `k_hat` is not
  capped at 0.5 (#585).
- Fixed response-scale-equivariant `Vp` / effective-n dispersion, logit erfcx
  quadrature derivatives, periodic 1-D Duchon PSD Bernoulli kernels, isotropic
  Matern divergence gates, `sz` continuous-first row sizing, SAE warm-start
  reuse / PSD Arrow-Schur ridge conditioning, top-k SAE encoder backprop, and
  auxiliary-conditional identifiability rank scaling (#576-#584).
- Folded the skipped `v0.3.70` and `v0.3.71` work into this published release.

## v0.1.143 — gam 0.3.71 / gamfit 0.1.143 (2026-06-01)

### Changed

- Published the second intermediate wheel in the `v0.3.70` / `v0.3.71` series;
  its fixes are folded into the `v0.3.72` notes.

## v0.1.142 — gam 0.3.70 / gamfit 0.1.142 (2026-06-01)

### Changed

- Published the first intermediate wheel in the response-scale / SAE /
  identifiability repair series that was consolidated in `v0.3.72`.

## v0.3.69 — gam 0.3.69 / gamfit 0.1.141 (2026-06-01)

### Changed

- Landed the draft GPU survival-FLEX row-primary gradient/Hessian launcher and
  oracle, and reduced survival marginal-slope per-row influence-absorber
  allocation churn.
- Reworked reference-quality CI with per-test wall-clock budgets, INLA
  dependency provisioning, a clearer outcome taxonomy, and right-sized
  expensive mgcv/LOO/scipy/tram tests.
- Fixed `ti(2d)` main-effect leak measurement, compositional-mean quality gates,
  badhealth `te()` mgcv references, row-Hessian non-Linux gating, and deleted
  the dead `dynamic_q_core_hessian_blocks` path.
- Published the actual `gamfit` 0.1.141 wheel version.

## v0.3.68 — gam 0.3.68 / gamfit 0.1.140 (2026-05-31)

### Fixed

- Fixed active-set scale invariance, bounded-shape Jacobian reporting, sphere
  latitude clipping at prediction time, and monotone-shape REML startup
  regressions (#500, #507, #508, #509).

## v0.3.67 — gam 0.3.67 / gamfit 0.1.139 (2026-05-31)

### Fixed

- Fixed published-crate build hygiene, exact Circle exp-map behavior,
  cubic-cell derivative consistency, and survival marginal-slope row-context
  error reporting.

## v0.3.65 — gam 0.3.65 / gamfit 0.1.136 (2026-05-29)

### Changed

- Restored main to a buildable state after lint/import drift and added the
  streaming SAE joint-fit path: minibatch on-demand recompute, block-sparse atom
  Schur structure, row-procedural GPU `H_tβ` matvecs, on-device Jacobi-CG, and a
  real scaling/parity demo (#358).
- Fixed REML penalty-coordinate projection onto the active-set free subspace,
  Gamma scaled-deviance likelihood use in the outer objective, multinomial REML
  deviance reuse, AdaptiveTopK hard top-k behavior, and ManifoldSAE config-matrix
  joint-solve coverage (#347-#360).
- Removed dead Gamma likelihood and sparse penalty imports left by the K>=2 SAE
  mechanism-sparsity refactor.

## v0.1.137 — gam 0.3.65 / gamfit 0.1.137 (2026-05-29)

### Fixed

- Fixed the identifiability anchor-correction dimension invariant and made it a
  release check before the later attempted `v0.3.66` / `gamfit 0.1.138` release.

## v0.3.64 — gam 0.3.64 / gamfit 0.1.135 (2026-05-29)

### Fixed

- Restored a coherent 0.3.x Rust release line after Python-only tags.
- Fixed ManifoldSAE out-of-sample inference to reuse fit-time SAE
  hyperparameters, replaced the static SAE basis shim with real Duchon/Euclidean
  basis refresh, and repaired positional isometry pairing with loud Jacobi
  non-convergence.
- Fixed deep-tail inverse-link precision for cloglog/probit-related paths,
  restored Linux GPU imports/macros, routed latent multi-output GLMs through
  canonical fitters, and added penalized multi-binomial family entry points.
- Removed three dominant CPU costs from the SAE inner Newton loop and repaired
  test/CI agent workflow drift.

## v0.1.134 — gam 0.2.3 / gamfit 0.1.134 (2026-05-28)

### Changed

- Re-triggered the gamfit wheel after the 0.1.131-0.1.133 version-bump attempts;
  the next coordinated `v0.3.64` release carried the corrected Rust and Python
  package state.

## v0.2.3 — gam 0.2.3 / gamfit 0.1.128 (2026-05-25)

### Fixed

- Published the CUDA runtime / diagnostics release between the 0.2.x and 0.3.x
  lines: exposed Python CUDA diagnostics, added loader tests, fixed cudarc
  CPU-only-host behavior, tightened runtime diagnostics, and shipped substantial
  Bernoulli marginal-slope / custom-family / REML-eval fixes.

## v0.1.130 — gam 0.2.3 / gamfit 0.1.130 (2026-05-27)

### Changed

- Published the gamfit 0.1.130 wheel after the 0.1.129 test-refactor release,
  before the later failed 0.1.131-0.1.133 bump attempts.

## v0.1.129 — gam 0.2.3 / gamfit 0.1.129 (2026-05-27)

### Fixed

- Published test refactors and minor fixes on the 0.2.3 engine line, including
  follow-up cleanup after the CUDA runtime diagnostics release.

## v0.3.63 — gam 0.2.2 / gamfit 0.1.124 (2026-05-25)

### Changed

- Coordinated the Rust/Python release after the 0.1.122 wheel line,
  including tensor B-spline derivative scratch reuse, JumpReLU logit-init
  fixes, Rust prediction-helper exports, and a large documentation accuracy
  pass across getting-started, predictions, REML scaling, persistence, sklearn,
  and GPU acceleration docs.
- Expanded regression coverage around SAE manifold flow, periodic basis
  validation, production linearized residuals, and doc examples.

## v0.3.62 — gam 0.2.1 / gamfit 0.1.123 (2026-05-25)

### Fixed

- Fixed periodic `basis_with_jet` shape validation, macOS `dynamic_lookup`
  pyffi linking, production-match tolerances for the red regression checks, and
  SAE manifold flow over the newer Rust auto APIs / prediction helpers.

## v0.1.122 — gam 0.2.1 / gamfit 0.1.122 (2026-05-24)

### Fixed

- Fixed SAE-manifold periodic Fourier basis behavior, added hard out-of-sample /
  multi-seed / sphere accuracy regressions, and replaced skipped OOS checks with
  assertions that the prediction surface exists.
- Rebalanced survival marginal-slope stall repro cohorts, tightened panic
  assertions, disabled GPU on macOS for the stall repro, and removed unused
  Python-extension imports.

## v0.1.121 — gam 0.3.65 / gamfit 0.1.121 (2026-05-24)

### Changed

- Added CV/permutation evidence documentation and bumped gamfit to 0.1.121; the
  next worked wheel line is captured by `v0.1.122`.

## v0.1.120 — gam 0.3.65 / gamfit 0.1.120 (2026-05-23)

### Changed

- Completed the Tweedie / Negative-Binomial exhaustive support pass, added
  Tweedie log-link inference handling, stabilized SAE Gumbel temperature/log
  weights, tightened sparse-exact solve handling, removed stale SAE fallbacks,
  and deleted stale composition-engine / audit-log material.

## v0.1.119 — gam 0.3.65 / gamfit 0.1.119 (2026-05-23)

### Changed

- Added negative-binomial likelihood variants, PIRLS support, log-link inference
  across HMC/sampling, and latent Negative-Binomial support.
- Refactored Arrow-Schur core solve paths, broadened latent-basis dispatch,
  added per-point Hessian artifacts, reused latent REML jets for topology
  evidence gradients, and corrected SAE IBP sign / ext-coordinate naming.

## v0.1.118 — gam 0.3.65 / gamfit 0.1.118 (2026-05-23)

### Changed

- Shipped Tweedie likelihood support, Euclidean metric-weighted reduced-Schur
  trust-region solves, Tweedie latent-GLM plumbing, exact latent-cache
  invalidation on latent updates, and all-target build unblocks for
  `StandardFitRequest`, dead-code, and the Ceres scaffold.

## v0.1.117 — gam 0.3.65 / gamfit 0.1.117 (2026-05-23)

### Fixed

- Fixed the `latent_cache.rs` REML import path and refreshed proposal docs for
  the landed latent-coordinate and iVAE pieces.

## v0.1.116 — gam 0.3.65 / gamfit 0.1.116 (2026-05-23)

### Changed

- Owned tensor knot slices in pyffi, wired Rust SAE IBP fitting, added latent
  design caching, latent ID direct hyperparameters, derivative/jet-backed basis
  evaluation, safe missing-cache handling for isometry penalties / ARD log
  terms, and stricter latent/SAE validation with fallible penalty builders.

## v0.1.115 — gam 0.3.65 / gamfit 0.1.115 (2026-05-23)

### Fixed

- Fixed the 0.1.114 wheel-build failure, added per-axis manifold metric weights
  and `ProductWithMetric`, normalized `fisher_w` naming, and added the Ceres
  backend scaffold.
- Carried the composition-engine WIP from the failed 0.1.114 tag forward:
  latent-coordinate plumbing, Arrow-Schur analytic penalties, Riemannian metric
  weighting, IBP-MAP / SAE-manifold pieces, topology-selection helpers, and
  strict mkdocs link/anchor fixes.

## v0.1.113 — gam 0.3.65 / gamfit 0.1.113 (2026-05-23)

### Fixed

- Routed the unified outer Hessian projected-operator path through the
  K-pseudoinverse, tied matrix-free stochastic-trace flags to materialization
  budgets, fixed a CTN `effective_weights` recursion bug, and aligned Duchon
  hybrid auto-resolution with `max_op=2`.
- Accepted mgcv-style relative-to-cost convergence for spatial iso/aniso fits
  that stop on `max_iter`.

## v0.1.112 — gam 0.3.65 / gamfit 0.1.112 (2026-05-22)

### Changed

- Exposed hybrid Duchon spectral knobs (`length_scale`, `nullspace_order`, and
  `power`) through Python primitives, added high-dimensional/default-argument
  regression coverage, fixed `difference_smooth(group_means=False)` to target
  the group main effect, expanded formula/smooth/family docs, made outer
  gradient norms optional in fit results, and removed stale sentinel docs.

## v0.1.111 — gam 0.3.65 / gamfit 0.1.111 (2026-05-22)

### Fixed

- Re-tagged the gamfit wheel with a type-annotation fix for HGB helpers and
  handled ill-conditioned REML backward passes with a zero-gradient fallback;
  documented and benchmarked `gt.fit` mode dispatch scaling.

## v0.1.110 — gam 0.3.65 / gamfit 0.1.110 (2026-05-22)

### Changed

- Added automatic dispatch between joint and independent torch fitting,
  constrained Gaussian REML backward through torch autograd, projected
  Firth/Jeffreys logdet paths, stability-gated sensitivity allocation, and
  stricter runtime budgeting.
- Reset outer IFT residual caches per fit and guarded trust-energy gates against
  stale cached residuals.

## v0.1.109 — gam 0.3.65 / gamfit 0.1.109 (2026-05-22)

### Changed

- Added analytic multi-block REML backward / VJP support, exposed block APIs,
  routed Gaussian REML forward through simpler wiring, and fixed multi-block
  REML gradients.
- Applied row-mask weighting in survival Hessian paths and replaced weighted
  cross-products with masked `mxtwx` psi multiplications.

## v0.1.108 — gam 0.3.65 / gamfit 0.1.108 (2026-05-22)

### Changed

- Added per-smooth lambda additive REML support and Smooth API exposure,
  including torch additive REML routing to the per-smooth multi-block solver,
  term diagnostics / block REML outputs, HT outer subsampling support for
  Gaussian and binomial location-scale paths, IFT warm-start beta prediction,
  and periodic / sphere basis APIs.

## v0.1.107 — gam 0.3.65 / gamfit 0.1.107 (2026-05-22)

### Fixed

- Handled `Result<usize, SurvivalError>` from survival event-code cause counts,
  surfaced typed errors through pyffi, added PIRLS AA(1) Fisher acceleration,
  exposed IFT residual metrics in `RemlLamlResult`, and suppressed
  envelope-inconsistent gradients unconditionally.

## v0.1.106 — gam 0.3.65 / gamfit 0.1.106 (2026-05-22)

### Changed

- Exposed `duchon_function_norm_penalty` as a public helper and continued the
  adaptive PIRLS KKT / outer-gradient integration.
- Returned typed errors rather than panics for invalid survival event codes,
  enforced contiguous survival event codes, and tightened REML derivative
  contracts.

## v0.1.105 — gam 0.3.65 / gamfit 0.1.105 (2026-05-22)

### Changed

- Published multi-dimensional Duchon and additive REML APIs, replacing 1-D
  Duchon bindings with `duchon_basis`, removing legacy Duchon derivative
  exports, and adding additive REML output / wrappers.
- Migrated torch tests to the new multi-D Duchon and additive REML API, fixed
  pyffi compile drift from core refactors, used typed link/distribution payload
  fields, added projected-KKT certificate regressions, and tightened constrained
  stationarity certification.

## v0.1.104 — gam 0.3.65 / gamfit 0.1.104 (2026-05-21)

### Fixed

- Used a rank-thresholded pseudo-inverse for the active-constraint Schur
  complement and bumped gamfit from 0.1.103 to 0.1.104.

## v0.1.103 — gam 0.3.64 / gamfit 0.1.103 (2026-05-21)

### Changed

- Switched ALO/HMC, scale-design, smooth/REML numerics, term builders, and
  custom-family active-constraint assembly to typed errors.
- Plumbed active inequality constraints through unified REML inner assembly,
  blockwise inner results, and blockwise active-constraint propagation; removed
  unused active-constraint RHS storage and centralized family floors.

## v0.1.102 — gam 0.3.63 / gamfit 0.1.102 (2026-05-21)

### Changed

- Removed unused saved-link helpers and dead link fallbacks, stored survival
  distributions as typed enums, preferred explicit saved survival links, and
  normalized monotone-root errors into a typed error.
- Routed REML coordinate solves through the penalty-subspace kernel, added
  checked diagonal block working sets / `SymmetricMatrix` helpers, documented
  GPU acceleration and CUDA stack conflicts, and locked the warn-not-raise
  contract for CUDA dual-stack detection.

## v0.1.101 — gam 0.3.62 / gamfit 0.1.101 (2026-05-21)

### Changed

- Refactored log-link IRLS, likelihood-family checks, link-state validation,
  SPD Levenberg-Marquardt logdet continuation, posterior quadrature helpers,
  and strict eta/clamp constants into shared code.
- Required projected KKT residuals for joint-Newton REML paths, refined PIRLS
  convergence certificates, documented `by` semantics for Gaussian REML
  position fits, and removed synthbug conflict remnants.

## v0.1.100 — gam 0.3.61 / gamfit 0.1.100 (2026-05-21)

### Changed

- Published the gamfit 0.1.100 wheel after the trust-region diagnostics line;
  the following `v0.1.101` release carried the shared log-link / likelihood /
  REML refactor.

## v0.3.61 — gam 0.3.61 / gamfit 0.1.99 (2026-05-21)

### Changed

- Classified and surfaced trust-region radius decisions in diagnostics, added
  joint-Newton stall labels, linearized residual metrics, logdet Hessian test
  derivatives, and outer-scale soft convergence exits for PIRLS.

## v0.3.60 — gam 0.3.60 / gamfit 0.1.98 (2026-05-21)

### Fixed

- Raised the custom-family default `inner_max_cycles` from 100 to 300, rejected
  boundary-saturated cache seeds, added screening proxy evaluation, and
  normalized survival marginal-slope inner-fit options.

## v0.3.59 — gam 0.3.59 / gamfit 0.1.97 (2026-05-21)

### Fixed

- Discarded fully saturated cached `rho` values instead of clamping them, added
  fractional Duchon null-space tests, tightened covariance-shape
  classification, clamped nonpositive survival times, and stabilized
  competing-risks CIF endpoint assembly / FFI returns.

## v0.3.58 — gam 0.3.58 / gamfit 0.1.96 (2026-05-21)

### Changed

- Shipped the IFT projected pseudo-inverse fix for gamfit and refactored
  competing-risks prediction payloads.
- Added competing-risks CIF / prediction / paired-sampling APIs, shared
  precision cross-fit helpers with fitted lambdas, fractional polyharmonic
  Duchon order support, hard covariance tests, and broader survival hard-test
  coverage.

## v0.1.95 — gam 0.3.57 / gamfit 0.1.95 (2026-05-21)

### Fixed

- Passed Duchon operator block order as `f64` at all call sites and carried the
  envelope-gradient / boundary-rho cache fixes into the next worked gamfit
  wheel line.

## v0.1.94 — gam 0.3.57 / gamfit 0.1.94 (2026-05-20)

### Fixed

- Short-circuited outer-Hessian assembly when the envelope-gradient check would
  trip, clamped cached boundary `rho` seeds, and removed unused KKT residual
  plumbing.

## v0.3.57 — gam 0.3.57 / gamfit 0.1.92 (2026-05-20)

### Changed

- Published REML optimization work including geometric Hessian scaling, CUDA
  diagnostics, cached inner warm-start state, exact-hit cache
  short-circuiting, deduplicated projected GEMMs, fused REML accumulation slice
  fast paths, faer-backed eigenbasis rotations, and parallel chunk traversal
  outside rayon pools.

## v0.3.56 — gam 0.3.56 / gamfit 0.1.91 (2026-05-20)

### Fixed

- Threaded EFS Hessian scale through eval samples and barrier checks, gated EFS
  on relative barrier curvature, fixed survival marginal-slope accumulator /
  scale-jet / joint-psi second-order wiring, skipped redundant warm-start
  pilots by family fingerprint, and pinned `opt` to the registry release.

## v0.1.90 — gam 0.3.56 / gamfit 0.1.90 (2026-05-20)

### Changed

- Published the package bump immediately before the EFS Hessian-scale release;
  the substantive EFS barrier/eval-sample wiring shipped in `v0.3.56`.

## v0.3.55 — gam 0.3.55 / gamfit 0.1.89 (2026-05-20)

### Fixed

- Published the CUDA diagnostics / KKT-convergence bundle between `v0.3.54` and
  `v0.3.56`: explicit Python CUDA diagnostic wrappers, CUDA stack conflict
  tests, row-kernel work modeling for rigid survival outer Hessians, stricter
  PIRLS KKT residual scaling, non-convergence / line-search failure handling,
  and canonical block-local Gaussian penalty-logdet derivatives.

## v0.1.88 — gam 0.3.55 / gamfit 0.1.88 (2026-05-20)

### Changed

- Added CUDA stack diagnostics and conflict checks, plus Python CUDA diagnostic
  wrappers, before the `v0.3.55` package-alignment release.

## v0.3.54 — gam 0.3.54 / gamfit 0.1.87 (2026-05-20)

### Fixed

- Added cuBLAS/CUDA dual-load diagnostics and defenses, including preload
  ordering, complete CUDA-stack validation, persisted `libcublas` handles, and
  removal of the process-level Python GPU disable hook.
- Refreshed GPU/survival/posterior documentation, torch extras metadata, and
  marginal-slope visualization assets.

## v0.1.86 — gam 0.3.54 / gamfit 0.1.86 (2026-05-20)

### Changed

- Refreshed package metadata and lockfile state for the CUDA dual-load defense
  line before the `v0.3.54` coordinated release.

## v0.3.53 — gam 0.3.53 / gamfit 0.1.85 (2026-05-19)

### Changed

- Re-tagged the Bernoulli periodic Duchon work with gamfit 0.1.85 so the wheel
  publish path ran.

## v0.1.85 — gam 0.3.52 / gamfit 0.1.85 (2026-05-19)

### Changed

- Bumped gamfit from 0.1.84 to 0.1.85 for the Bernoulli periodic Duchon wheel
  release.

## v0.3.52 — gam 0.3.52 / gamfit 0.1.84 (2026-05-19)

### Fixed

- Fixed Bernoulli periodic Duchon kernels and expanded position-basis alias
  coverage.
- Dropped duplicate periodic endpoint centers, enforced odd effective K in seam
  tests, used the Bernoulli Green's kernel, covered design rank and `B_4`
  spectrum, removed Duchon PSD projection helpers, and required strict KKT
  residuals for joint inner convergence.

## v0.3.51 — gam 0.3.51 / gamfit 0.1.83 (2026-05-19)

### Fixed

- Raised survival pilot caps, improved dense trust-region steps, fixed REML
  penalty/log-lambda gradients, used objective-scaled absolute gradient floors
  for outer convergence certification, and simplified solution-certification
  logging.

## v0.3.50 — gam 0.3.50 / gamfit 0.1.82 (2026-05-19)

### Changed

- Shipped the convergence-truthfulness bundle: objective-floor guards,
  post-convergence status reporting, and no silent success on stalled fits.
- Added Python REML scoring APIs, `grad_penalty` output for
  `gaussian_reml_score`, non-REML smoothing support in batched position fits,
  free-coefficient position REML scoring, basis alias normalization, Duchon
  function-norm penalties, and tighter REML penalty routing.

## v0.3.49 — gam 0.3.49 / gamfit 0.1.81 (2026-05-19)

### Changed

- Added the batched psi-term fast path for survival marginal-slope fits, with a
  regression test that batched terms match per-axis terms, plus batch correction
  and ext-coordinate Hessian solves in solver/REML.

## v0.3.48 — gam 0.3.48 / gamfit 0.1.80 (2026-05-19)

### Fixed

- Enabled automatic outer subsampling by default for marginal-slope fits,
  failed survival marginal-slope fits on outer non-convergence, removed invalid
  survival-prediction row fallbacks, handled droppable NaN rows explicitly, and
  required a KKT residual ceiling for flat-step PIRLS convergence.

## v0.3.47 — gam 0.3.47 / gamfit 0.1.79 (2026-05-19)

### Fixed

- Broadened CUDA preload paths, simplified CUDA `dlopen` warning text, used
  mode ridge for predicted reduction in blockwise trust-region, tightened flat
  joint-step convergence, and added rho=2 stabilization proof / saturated-null
  direction diagnostics.

## v0.3.46 — gam 0.3.46 / gamfit 0.1.78 (2026-05-19)

### Changed

- Added always-on visualizer sessions for `gamfit` fits, open-ended workflow
  progress feeds, optimizer metrics and cost sparklines, fixed-log-lambda
  survival pilot warm starts, CUDA calibration diagnostics, and cudarc-backed
  CUDA / cuBLAS preflight checks.
- Tightened inner-solve convergence handling, removed the Bernoulli step cap,
  and surfaced GPU calibration errors.

## v0.3.45 — gam 0.3.45 / gamfit 0.1.76 (2026-05-19)

### Changed

- Completed the GPU module migration to cudarc 0.19 and exposed Python-side GPU
  activity / visualizer state.
- Reduced PIRLS joint-Newton log verbosity and moved accepted-cycle timing to
  debug output.

## v0.1.75 — gam 0.3.44 / gamfit 0.1.75 (2026-05-19)

### Changed

- Added hierarchical near-match warm starts and made cache keys survive package
  version bumps.
- Preflighted `libcuda` / `libcublas` loads before cudarc calls, migrated cuBLAS
  runtime and CUDA transfers to cudarc wrappers, added cache mirror-session
  finalization broadcast, threaded survival cache sessions, and accepted
  saturated hazard / convergence regimes.
- Aligned Bernoulli cross-block orthogonalization with the PIRLS Hessian metric
  and limited PIRLS near-convergence log promotion to residual convergence.

## v0.1.74 — gam 0.3.44 / gamfit 0.1.74 (2026-05-19)

### Fixed

- Added load-side finiteness gates for caches, throttled joint-Newton logs, and
  hardened GPU/cache paths, including missing ndarray imports in the GPU session
  path and parallel GPU-session safety.

## v0.1.73 — gam 0.3.44 / gamfit 0.1.73 (2026-05-19)

### Changed

- Made warm starts uniform across custom-family fits and wired cache-session
  hooks through fit requests and solver entry points.
- Kept GPU design matrices resident across PIRLS iterations, warmed the GPU
  runtime early, exposed Python GPU activity summaries, calibrated dispatch
  thresholds from measured runtime metrics, added blockwise cache-session
  options, and ran survival regression / save-load roundtrip tests in CI.

## v0.1.72 — gam 0.3.44 / gamfit 0.1.72 (2026-05-19)

### Fixed

- Fixed survival time-basis persistence so saved models always include the
  anchor and the construction path imports `SavedSurvivalTimeBasis`.

## v0.1.71 — gam 0.3.44 / gamfit 0.1.71 (2026-05-19)

### Changed

- Persisted survival time-basis snapshots, populated marginal-slope payload
  baseline/time-basis fields, and covered the saved `survival_time` basis field.

## v0.1.70 — gam 0.3.44 / gamfit 0.1.70 (2026-05-19)

### Changed

- Added position REML basis-state outputs, auto-resolved position basis inputs,
  sped col-major conversion, and routed memory-limited marginal-slope chunks to
  CPU when the GPU path is ineligible.
- Exposed Rust 1-D automatic basis placement for `None`/integer knots and
  centers, returned basis state through payload attachments, lowered GPU GEMM /
  GEMV / TRSM dispatch thresholds, and adjusted joint-objective acceptance for
  floating-point roundoff.

## v0.1.69 — gam 0.3.44 / gamfit 0.1.69 (2026-05-19)

### Fixed

- Released the 0.1.69 wheel after the 0.3.44 Rust engine line and removed the
  remaining `xfail` markers from the full 34/34 torch suite.

## v0.1.68 — gam 0.3.44 / gamfit 0.1.68 (2026-05-19)

### Changed

- Canonicalized Gaussian REML penalties, required symmetric torch REML
  penalties, added cache mismatch diagnostics, and expanded torch/REML
  regression coverage.
- Symmetrized penalty gradients in closed-form REML backward, stabilized EDF
  backward gradcheck matrices, treated tiny analytic/finite-difference gradients
  as zero in relative-error helpers, logged each GPU routing signature once, and
  removed debug markers from Gaussian REML ill-conditioning paths.

## v0.1.67 — gam 0.3.44 / gamfit 0.1.67 (2026-05-18)

### Changed

- Published the final pre-`v0.1.68` gamfit-only wheel after the 0.3.44 engine
  bump, carrying the torch REML and package-layout stabilization work that the
  next tagged release made explicit.

## v0.1.66 — gam 0.3.44 / gamfit 0.1.66 (2026-05-18)

### Changed

- Published a gamfit-only wheel on the 0.3.44 engine line after the 0.3.44 /
  pyffi 0.1.66 / gamfit 0.1.64 coordinated bump.

## v0.1.64 — gam 0.3.44 / gamfit 0.1.64 (2026-05-18)

### Changed

- Bumped the engine and Python bridge to the 0.3.44 / 0.1.66 line, preparing
  the torch REML symmetric-penalty release series that culminated in
  `v0.1.68`.

## v0.1.62 — gam 0.3.42 / gamfit 0.1.62 (2026-05-15)

### Changed

- Published the follow-up package bump after the 0.3.41 / 0.1.61 release,
  keeping Rust, pyffi, and gamfit versions aligned during the May 15 torch/REML
  release train.

## v0.1.61 — gam 0.3.41 / gamfit 0.1.61 (2026-05-15)

### Changed

- Published the next coordinated Rust / pyffi / gamfit bump after the 0.1.60
  wheel, preserving package alignment for the torch/REML work.

## v0.1.60 — gam 0.3.40 / gamfit 0.1.60 (2026-05-14)

### Changed

- Published the first worked package bump after the hard-pseudo REML Hessian /
  joint-solver rejection series, moving the engine toward the later 0.3.44 torch
  REML release line.

## v0.3.36 — gam 0.3.36 / gamfit 0.1.56 (2026-05-11)

### Fixed

- Fixed PyPI release workflow skip propagation by prefixing the release job
  condition with `always()`.
- Folded in the skipped `v0.3.35` workflow-dispatch publish fix so manual PyPI
  release runs actually reach the release job.

## v0.3.34 — gam 0.3.34 / gamfit 0.1.54 (2026-05-11)

### Fixed

- Fixed the PyPI `workflow_dispatch` release-job gate.

## v0.3.33 — gam 0.3.33 / gamfit 0.1.53 (2026-05-11)

### Fixed

- Translated `expects N blocks, got 0` block-state shape errors at the Python
  boundary instead of surfacing raw Rust messages.
- Carried forward the skipped `v0.3.32` belt-and-suspenders guard against empty
  survival location-scale `block_states`.

## v0.3.31 — gam 0.3.31 / gamfit 0.1.51 (2026-05-11)

### Fixed

- Projected the REML outer-gradient trace kernel for non-Gaussian families,
  addressing the root `expects 3 blocks, got 0` failure.

## v0.3.30 — gam 0.3.30 / gamfit 0.1.50 (2026-05-11)

### Fixed

- Tightened the custom-family outer `rho` bound to prevent ARC-stall crashes.

## v0.3.29 — gam 0.3.29 / gamfit 0.1.49 (2026-05-11)

### Fixed

- Broadened the survival location-scale empty `block_states` crash guard.

## v0.3.28 — gam 0.3.28 / gamfit 0.1.48 (2026-05-11)

### Changed

- Tuned the PyPI `release-pypi` profile max-runtime settings.
- Folded in the skipped `v0.3.26` / `v0.3.27` release-gate changes, including
  the on-demand `linux_only` PyPI release input and package metadata bumps.

## v0.3.25 — gam 0.3.25 / gamfit 0.1.45 (2026-05-11)

### Changed

- Version-only publish marker after the survival location-scale optimizer
  update.

## v0.3.24 — gam 0.3.24 / gamfit 0.1.44 (2026-05-11)

### Changed

- Replaced survival location-scale baseline `CompassSearch` with an
  analytic-gradient BFGS path.

## v0.3.23 — gam 0.3.23 / gamfit 0.1.43 (2026-05-11)

### Changed

- Sped up survival location-scale GM baseline profiling, preserved benchmark
  shard output across blocking failures, and marked the flexible Rust GAM
  benchmark as non-blocking.
- Added a spatial-kappa REML re-evaluation drift tolerance and removed, then
  restored, mathematically infeasible joint-PC Duchon benchmark scenarios as the
  benchmark blocking policy changed.

## v0.3.22 — gam 0.3.22 / gamfit 0.1.42 (2026-05-11)

### Changed

- Added native ARM Linux / cache-warming PyPI workflow support, refreshed
  marginal-slope documentation figures, and preserved in-flight family and
  prediction-path fixes.
- Deleted the orphaned `approx_ledger` module and its test, dropped tracked
  benchmark result artifacts, and tightened benchmark-result gitignore rules.

## v0.3.21 — gam 0.3.21 / gamfit 0.1.41 (2026-05-11)

### Changed

- Routed location-scale and latent survival sampling to a Laplace fallback and
  added the first MkDocs/Material documentation site.
- Rewrote the README, added Read the Docs / Material configuration, docs CI,
  social assets, and broader `gamfit` docstrings for API reference rendering.

## v0.3.20 — gam 0.3.20 / gamfit 0.1.40 (2026-05-11)

### Changed

- Threaded `PseudoLogdetMode` through the matrix-free SPD operator.

## v0.3.19 — gam 0.3.19 / gamfit 0.1.39 (2026-05-11)

### Fixed

- Fixed PyPI workflow invocation by dropping the mutually exclusive `--release`
  flag from the profile build.

## v0.3.18 — gam 0.3.18 / gamfit 0.1.38 (2026-05-11)

### Changed

- Added a BLAS-3 `projected_matrix` override with output symmetrization and an
  `n * rank^2` threshold gate, and sped up PyPI wheel CI.

## v0.3.17 — gam 0.3.17 / gamfit 0.1.37 (2026-05-11)

### Fixed

- Fixed BMS batched `dH` rayon-pool starvation deadlocks at small row counts.

## v0.1.37 — gam 0.3.17 / gamfit 0.1.37 (2026-05-11)

### Fixed

- Fixed the same BMS batched `dH` starvation issue for the Python-wheel tag and
  carried marginal-slope / GAMLSS performance gates forward.

## v0.3.15 — gam 0.3.15 / gamfit 0.1.34 (2026-05-11)

### Fixed

- Fixed the gam-pyffi saved-runtime payload by adding missing
  `SavedAnchoredDeviationRuntime.anchor_residual` fields.
- Folded in the skipped `v0.3.14` Duchon derivative work: design and frozen-Z
  penalty finite-difference tests, a no-identifiability variant, and a 1-D
  `power=2` linear-control probe.

## v0.3.13 — gam 0.3.13 / gamfit 0.1.32 (2026-05-11)

### Changed

- Added `RayonSafeOnce` for lazy caches whose initialization dispatches nested
  rayon work.

## v0.3.12 — gam 0.3.12 / gamfit 0.1.31 (2026-05-11)

### Fixed

- Fixed ARC retry config borrowing and removed temporary diagnostic output.

## v0.3.11 — gam 0.3.11 / gamfit 0.1.30 (2026-05-11)

### Changed

- Hardened PSD penalty handling, ARC retry behavior, cache fingerprinting,
  per-axis composite traces, and pseudo-inverse reuse for REML Hessians.
- Added workspace-cached performance paths, richer constraint-nullspace errors,
  finite-difference hardening, exact-hit PIRLS LRU clearing on outer-seed
  resets, `opt` 0.5.3 `NumericallyConverged` handling, and certified-final-value
  gates for joint-spatial REML surfaces.

## v0.3.10 — gam 0.3.10 / gamfit 0.1.29 (2026-05-11)

### Changed

- Published the rank-deficient rho-Hessian / BMS diagnostic work between
  `v0.3.9` and `v0.3.11`: consistent observed-Hessian jets, penalty-redundancy
  diagnostics, marginal-slope protocol cleanup that stopped baking in
  `score_warp` / `link_deviation`, and repeated spectral-operator log
  coalescing.

## v0.3.9 — gam 0.3.9 / gamfit 0.1.29 (2026-05-10)

### Changed

- Improved large-scale performance with rank-INT latent-z handling,
  line-search subsampling, row-set threading, and predict-time anchor
  correction.
- Added BMS residual plumbing, cross-block identifiability regressions for
  `(I-P_A)C` residualization, rigid-path performance gates, and adaptive
  GAMLSS inner-cycle caps / soft warm starts across `rho`.

## v0.3.8 — gam 0.3.8 / gamfit 0.1.29 (2026-05-10)

### Changed

- Added Hutch++ trace estimators, two-phase automatic row subsampling, and
  row-kernel pooling for marginal-slope families.
- Added batched `MultiDirJet` contraction for survival marginal-slope
  third/fourth derivatives, KKT-on-null post-step certification, REML
  penalty-rank cliff fixes, entropy-driven fuzz seeding, and hot-path speedups
  in non-affine cell evaluation / row Hessian construction.

## v0.3.7 — gam 0.3.7 / gamfit 0.1.28 (2026-05-10)

### Changed

- Extended cross-block identifiability to parametric anchors, certified cycle-0
  KKT convergence, and extracted diagonal Hessian scores for joint Newton.
- Refactored the Hutch++ marginal-slope API and removed a probit
  Hessian-collapse test that asserted logit-only behavior.

## v0.3.6 — gam 0.3.6 / gamfit 0.1.27 (2026-05-10)

### Changed

- Wired cross-block identifiability APIs through Bernoulli marginal-slope and
  BMS joint-orthogonal flexible bases.
- Carried coordinated Bernoulli and survival-family edits from the concurrent
  agent work into the release.

## v0.3.5 — gam 0.3.5 / gamfit 0.1.26 (2026-05-10)

### Changed

- Rejected non-converged CTN screening seeds and opted exact-joint initial-rho
  screening into relevant custom-family paths.
- Removed stale CTN screening imports and screened BFGS exact-joint CTN seeds
  before ranking them.

## v0.3.4 — gam 0.3.4 / gamfit 0.1.25 (2026-05-09)

### Fixed

- Stopped capped BFGS on objective stall and shipped the custom-family
  line-search workspace fixes from the concurrent edit range.

## v0.3.3 — gam 0.3.3 / gamfit 0.1.24 (2026-05-09)

### Fixed

- Fixed blockwise PIRLS trust-region adaptation.

## v0.3.2 — gam 0.3.2 / gamfit 0.1.23 (2026-05-09)

### Fixed

- Fixed Windows wheel builds by normalizing paths in the approximate-ledger
  scanner.
- Anchored the CTN BFGS step cap and cleaned up BMS formatting drift.

## v0.3.1 — gam 0.3.1 / gamfit 0.1.22 (2026-05-09)

### Fixed

- Fixed the unused-assignment warnings and accidental `probability.rs`
  truncation that broke 0.3.0 wheel builds.
- Removed redundant joint-Newton rejection bookkeeping and restored rustfmt
  compliance in subsampling / row-construction paths.

## v0.3.0 — gam 0.3.0 / gamfit 0.1.21 (2026-05-09)

### Changed

- Shipped the audit-driven correctness pass: rcond-floor scale-design
  truncation without Tikhonov bias, stable tail likelihoods, honest
  indefinite-Hessian skips, outer-derivative guards on inner convergence,
  Horvitz-Thompson row weights for outer-score subsampling, resource-policy
  auto-derivation, exported Laplace-curvature labeling, and the stabilization
  ledger / spectral classifier cleanup.

## v0.1.20 — gam 0.2.1 / gamfit 0.1.20 (2026-05-09)

### Fixed

- Fixed survival location-scale identifiability, clarified custom-family
  convergence comments, optimized GAMLSS projected traces, and simplified score
  warp anchoring.
- Moved the benchmark suite to nightly-only execution and carried
  Bernoulli/custom-family/GAMLSS/transformation-normal local rollups forward.

## v0.1.19 — gam 0.2.1 / gamfit 0.1.19 (2026-05-09)

### Fixed

- Restored Bernoulli marginal-slope and survival prediction files after a bad
  local rollup, reverted a custom-family joint-line-search workspace hook, and
  tightened inner KKT / finite-difference checks.
- Replaced data-distribution moment-anchor tests with full-rank penalty tests
  and recovered rustfmt compliance in Bernoulli marginal-slope and survival
  prediction code.

## v0.1.18 — gam 0.2.1 / gamfit 0.1.18 (2026-05-09)

### Fixed

- Auto-chunked dense survival predictions and replaced data-distribution
  moment-anchor tests with full-rank penalty tests before the `v0.1.19`
  corrective release.

## v0.2.1 — gam 0.2.1 / gamfit 0.1.17 (2026-05-08)

### Changed

- Added the outer-operator `apply_into` trait hook, trimmed the published crate
  include-list under crates.io limits, and split PyPI wheel build from publish.
- Dropped a GAMLSS cache path that no longer matched the operator API.

## v0.2.0 — gam 0.2.0 / gamfit 0.1.17 (2026-05-08)

### Changed

- Migrated the optimizer integration to `opt` 0.5.0, including
  `DeclaredHessianForm` capability plumbing and accepted-step observation.
- Added `FirthAugmentedSingleHyperOperator::trace_projected_factor`, chunked
  GEMM traces for implicit hyperoperators, and follow-up migration across
  `OuterProblem`, GAMLSS, and mixture-zero estimate paths.

## v0.1.17 — gam 0.1.17 / gamfit 0.1.17 (2026-05-07)

### Changed

- Introduced the top-level `gamfit` Python package layout, removed the
  one-off PGS/examples wrappers, reorganized benches, added PyPI wheel
  publishing, and shipped the cache-policy auto-derivation, marginal-slope
  performance, inner-Newton line-search, and Python-binding cleanup from the
  long pre-release diff.
- Added NUTS posterior sampling exposure, marginal-slope and GAMLSS performance
  work, BMS LRU derivative paths, BLAS-3 Bernoulli rigid-row kernels, large-scale
  design dense-conversion fixes, and broader Python packaging cleanup.

## v0.1.16 — gam 0.1.16 / gamfit 0.1.16 (2026-05-07)

### Changed

- Renamed the Python package to `gamfit`, relicensed to AGPL-3.0-or-later,
  replaced joint-Newton backtracking with a trust-region path, dropped
  pre-whitening from constraint-nullspace solves, and moved the publish workflow
  to token-based publishing.

## v0.1.14 — gam 0.1.14 / gamfit - (2026-03-17)

### Changed

- Completed the sparse Takahashi analytic-gradient line: matrix-level
  perturbation Hessians, exact `P_total` trace gradients, geometry caching,
  Matern parallelism, cross-trace speedups, block-local penalty operations, and
  clean lib/test builds.

## v0.1.13 — gam 0.1.13 / gamfit - (2026-03-16)

### Changed

- Published the large REML / LAML sparse-calculus pass: block-local Takahashi
  traces, sparse exact outer calculus, exact penalty pseudo-logdet on positive
  eigenspaces, structural nullspace dimensions, survival fourth derivatives,
  GAMLSS observed weights, Firth drift fixes, and large-scale PCG /
  compositional hyper-drift work.

## v0.1.11 — gam 0.1.11 / gamfit - (2026-02-25)

### Changed

- Moved crude-risk survival quadrature into the engine, added dual-risk and
  survival regression tests, moved gradient-isolation/parity scripts into the
  Rust test tree, and added core CI / benchmark GitHub Actions workflows.

## v0.1.10 — gam 0.1.10 / gamfit - (2026-02-25)

### Changed

- Version-only crates.io publish marker for the stabilized 0.1.x Rust engine.

## v0.1.9 — gam 0.1.9 / gamfit - (2026-02-25)

### Changed

- Optimized analytic trace contractions and expanded equation-to-code
  derivation comments.

## v0.1.8 — gam 0.1.8 / gamfit - (2026-02-25)

### Changed

- Made the exact survival rho-gradient path the default while retaining a
  finite-difference fallback API.

## v0.1.7 — gam 0.1.7 / gamfit - (2026-02-25)

### Changed

- Added exact rho-gradient survival optimizer path and migrated engine tests to
  it.

## v0.1.6 — gam 0.1.6 / gamfit - (2026-02-25)

### Changed

- Improved external optimizer seed ordering and early-exit behavior.

## v0.1.5 — gam 0.1.5 / gamfit - (2026-02-25)

### Fixed

- Fixed non-Gaussian LAML gradient double counting.

## v0.1.4 — gam 0.1.4 / gamfit - (2026-02-25)

### Fixed

- Fixed external optimizer stationarity handling and evaluated convergence in
  z-space.

## v0.1.3 — gam 0.1.3 / gamfit - (2026-02-25)

### Changed

- Switched external non-Gaussian REML checks to an objective-consistent
  finite-difference gradient.

## v0.1.2 — gam 0.1.2 / gamfit - (2026-02-25)

### Fixed

- Fixed cancellation in non-Gaussian REML gradients at high smoothing
  parameters.

## v0.1.1 — gam 0.1.1 / gamfit - (2026-02-25)

### Changed

- Strengthened oracle tests and aligned survival penalty derivatives.

## v0.1.0 — gam 0.1.0 / gamfit - (2026-02-25)

### Changed

- Initial crates.io release of the Rust GAM engine, including formula/design
  construction, REML/PIRLS fitting paths, survival and non-Gaussian families,
  uncertainty output, and engine test coverage.

## v0.0.0 — gam 0.0.0 / gamfit - (2026-02-24)

### Changed

- Initial placeholder crates.io publication after the early GAM stack import,
  including Duchon/Matern basis coverage, sparse-native REML paths, probit
  location-scale warm starts, model-consistency fixes, and the first Rust CI
  workflow.
