## Confirmed at source level — and a second layer the grid trace exposes

I reproduced this analysis against the **live tree** (not an archive) and the decisive mechanism checks out exactly. Credit to the reporter: the "three-rejected-probes abort" framing is correct and sharper than the original issue text. But the seed-grid lines in the attached verbose log show the stall-guard bug is **necessary-but-not-sufficient** — there is a second, deeper defect underneath it. Both need to land for the EDF to come back.

### Part 1 — the stall-guard bug is real (verified in `src/solver/rho_optimizer/bridges.rs`)

The cost-stall guard increments its no-improvement streak on **strictly-worse, rejected** ARC trial probes:

- `bridges.rs:308-334` — `improvement = best_value - value; floor = rel_tol*(1+|best|); if improvement <= floor || kkt_stationary_at_bound { no_improve_streak += 1 }`. For a worse trial (`value` = 891.5 vs `best` = 529.2) `improvement = -362.4 ≤ floor (5.3e-5)`, so a probe that is *hundreds of units worse* counts as a "flat" step. The condition was written for **accepted** steps where `value ≈ best` (small positive improvement = genuinely flat); it does not distinguish "barely improved" from "much worse / rejected".
- `bridges.rs:1269-1271` (the guard's own doc-comment) — *"`opt::Arc` evaluates the (value, gradient, Hessian) triple at every trial point — accepted or rejected — through `eval_hessian`, so observing here counts ARC's outer descent."* So the guard is wired to the **Hessian oracle** (`observe_cost_stall` at `:1282-1303`, called from `eval_hessian` at `:1488`), i.e. it sees every probe the trust region rejects.
- `bridges.rs:285` — `observe()`'s contract literally says *"Fold one **accepted-iterate**"*, but the wiring feeds it rejected trials. That is the bug in one line: **documented contract (accepted) ≠ actual call site (every oracle eval)**.

With `ARC_COST_STALL_WINDOW = 3` (`:178`), three consecutive worse probes (891.5 → 901.7 → 907.6, all ≫ incumbent 529.2) fill the window and trip `FlatValleyStall`. ARC's trust region was doing exactly what it should — probe, reject, shrink — and the guard aborted it before it could shrink into the descent. The `Outer iterations: 4` in the summary is `observe_seed` (+1) plus the three rejected probes.

The reporter is also right that this is then **shipped silently**: `run_plan.rs:1050-1073` rebuilds `Ok(result)` with `exit.converged = false` and `CostStallFlatValley`, and the existing `OuterAcceptObserver` (`bridges.rs:1538-1552`) proves the accepted-step signal exists — it's used for the inner-cap schedule — but `StepInfo` (`:1543-1550`) carries only `iter/step_norm/predicted/actual_decrease`, **not** `(x, value, gradient)`, so the stall guard can't currently consume accepted steps even if we wanted it to. The clean fix does require extending that payload (or a local accepted-state bridge), as the reporter notes.

This bug is **family-agnostic** and high-severity on its own: any ARC outer run whose trust region rejects `window` probes in a row before improving will abort non-stationary and ship. Fixing it is unambiguously correct.

### Part 2 — but the grid trace says the objective itself prefers the overfit here

The original report concluded the objective is fine and this is purely a termination bug. The attached log's `[SEED-GRID]` lines (which that repro didn't have) contradict that. REML cost vs ρ (both penalty coords swept together):

| ρ | −11 | −8 | −5 | **−2** | +1 | +4 | +7 |
|---|---|---|---|---|---|---|---|
| cost | 645.5 | 604.7 | 565.9 | **529.2** | 681.3 | 882.0 | 886.7 |

The minimum is **interior**, near ρ≈−2 (λ≈0.13), with cost rising on **both** sides — toward λ→0 (645.5) *and* toward more smoothing (681.3, 882.0). So this is not a monotone λ→0 ridge at all. But ρ≈−2 already yields **EDF 23.95** (near-full-basis). In other words: gam's REML criterion for this fit has its genuine interior optimum in the high-EDF region. A perfectly-converged optimizer would still land near full basis. mgcv's REML on the same data minimizes at EDF≈8.

The tell is in the same log: **`rho_dim=2 penalty_rank=23`** and `k=2`. A default 1-D `s(x)` here is gam's **double-penalty** B-spline (a primary bending penalty **plus** a `DoublePenaltyNullspace` ZZᵀ term), so REML is selecting over **(λ_bend, λ_null)**, while mgcv's default `s(x)` penalizes only the range space. This is the exact mechanism of the still-open **#1266** (double-penalty EDF inflation: `ps` EDF 4.96 vs mgcv 2.10). Gamma/log appears to be where that nullspace-coordinate mis-scaling blows all the way up to the full basis. So:

- **Fix A (stall guard)** stops the silent ship and the spurious abort, and is required regardless.
- **Fix B (objective)** is what actually recovers EDF≈8: the team must confirm whether gam's Gamma + default double-penalty B-spline REML is optimizing the *same* criterion mgcv does. If the interior REML minimum genuinely sits at EDF≈24, Fix A alone just converts a silent non-converged overfit into a *certified* overfit — still wrong. This is almost certainly #1266 surfacing under `family=gamma, link=log`.

### Recommended path

1. **Stall guard (Fix A):** stop counting rejected probes. Either (a) only fold **accepted** iterates into the streak — extend `StepInfo`/observer to carry `(x, value, projected_g)` and drive the guard from `on_step_accepted`; or, as a minimal hotfix, (b) reject any `observe()` whose `value` regresses the incumbent by more than `floor` (the symmetric partner of the existing `observe_constrained_stationary` regress-guard at `:405-413`) so a worse probe resets rather than increments. Keep the genuine near-separable-multinomial behavior (#1082/#1237) — there every probe *beats* best, so it resets correctly and is unaffected.
2. **Certify on stationarity, not cost-flatness:** a large projected gradient (`|g|=10.75 ≫ 1e-3`) must never be returned as an ordinary fit. Fail-closed by default; surface `outer_converged`/`outer_gradient_norm` (already on `UnifiedFitResult`) through `summary()`/report/`predict()`.
3. **Discriminate A vs B before declaring victory:** evaluate gam's REML on a fine ρ grid and at mgcv's selected λ, with the double-penalty nullspace coordinate on vs off, and compare to mgcv's REML value at matched λ. If gam's interior optimum is at the wrong EDF, route the fix through #1266 (nullspace-penalty normalization), not the optimizer.

The regression test must assert the **recovered EDF/RMSE** (EDF≈8, RMSE≈0.094 on seeds 770013/900005), not merely "no longer stalls" — otherwise Fix A alone will pass a still-wrong fit.

*Verified against current `main`/HEAD source; the grid-trace analysis (Part 2) is drawn from the attached log and is the part I'd most want a second set of eyes on before we commit to A-only vs A+B.*
