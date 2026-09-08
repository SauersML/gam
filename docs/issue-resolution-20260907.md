# Open issue resolution ledger — 2026-09-07

All 35 open issues are in scope. Historical reports are starting evidence; an issue is closed only after its acceptance conditions are measured on the relevant repaired source. No full-suite pass has been established in this session.

Validation uses the existing MSI repository and warm build target, bounded CPU concurrency, and shared disk storage. No local builds or test execution.

## REML and model quality

Coordinator: reml_numerics.

| Issue | Problem | Current session status |
| --- | --- | --- |
| [#2830](https://github.com/SauersML/gam/issues/2830) | Block Gaussian REML: an SPD breakdown at ONE trial lambda is escalated to a whole-fit abort — 11 of 120 1e-6 nudges of y hard-error while the 109 survivors move the fit by 3.9e-07 | Open; acceptance evidence and current implementation under review |
| [#2831](https://github.com/SauersML/gam/issues/2831) | Constrained Gaussian REML unconditionally probes rho = +/-30 (lambda = 1.1e13) and dies there: 14 of 16 binding box constraints abort against a rank-deficient penalty, all 16 solve with a 4e-13 relative ridge | Fix implemented; remote verification in progress |
| [#2834](https://github.com/SauersML/gam/issues/2834) | y ~ s(x) + s(x, g, bs='fs') aborts on HALF of ordinary Gaussian datasets: the terminal lane-agreement audit fires at 35x-2180x its roundoff bound, always with a lambda railed at RHO_BOUND=30 | Open; acceptance evidence and current implementation under review |
| [#2835](https://github.com/SauersML/gam/issues/2835) | firth=True makes 11 of 30 ordinary two-smooth binomial GAMs unfittable: the railed rho-gradient is exactly -rank(S_k)/2 (-1.5/-3.0/-4.0/-5.0 for k=5/8/10/12), i.e. the +lambda*tr(H^-1 S_k) half is missing | Fix implemented; remote verification in progress |
| [#2817](https://github.com/SauersML/gam/issues/2817) | standard REML outer search never converges: every seed exhausts its 200-iteration budget against a band 29x tighter than the certificate that then accepts the fit, and the whole 3-seed sweep is retried | Open; acceptance evidence and current implementation under review |
| [#2735](https://github.com/SauersML/gam/issues/2735) | large_scale_reml_stress_main runs 2005 s and still never evaluates its held-out bar — the rho-REML terminates on the fixture's own MAIN_MAX_ITER = 40 with /Pg/ = 7.812e-1 against a 5.000e-1 stationarity bound, so no fit is minted | Open; acceptance evidence and current implementation under review |
| [#1561](https://github.com/SauersML/gam/issues/1561) | Bug: GAM should be but is not significantly better than reference software | Open; acceptance evidence and current implementation under review |
| [#2469](https://github.com/SauersML/gam/issues/2469) | Delete all arbitrary constants in the entire repository | Open; acceptance evidence and current implementation under review |

## Latent derivatives, survival and predictions

Coordinator: latent_adjoint.

| Issue | Problem | Current session status |
| --- | --- | --- |
| [#2833](https://github.com/SauersML/gam/issues/2833) | gaussian_reml_fit_latent_backward's grad_t is not the gradient of its own reml_score (cosine -0.43): the forward fits duchon_basis(t) @ V(t) with a batch-global V the adjoint does not differentiate -- Rust-side sibling of #2097 | Fix implemented; remote verification in progress |
| [#2765](https://github.com/SauersML/gam/issues/2765) | The slope can't vary | Open; acceptance evidence and current implementation under review |
| [#2767](https://github.com/SauersML/gam/issues/2767) | let b vary along the follow-up axis | Open; acceptance evidence and current implementation under review |
| [#2748](https://github.com/SauersML/gam/issues/2748) | a flexible-link BINOMIAL fit saves itself as family 'gaussian' with an Identity link, so every response-scale prediction is the linear predictor (posterior_mean bit-identical to linear_predictor_plugin on every held-out row) — 12 of the 20 benchmark scenarios error inside budget at brier_score | Open; acceptance evidence and current implementation under review |
| [#2714](https://github.com/SauersML/gam/issues/2714) | latent survival coxph frailty: the fit now reaches the inner solve and stalls there — 28 cycles, stationarity_residual=1.741513e-2 vs tol 3.59e-10, trust radius railed at 1e-12, both rejections are OBJECTIVE rejections | Open; acceptance evidence and current implementation under review |
| [#2705](https://github.com/SauersML/gam/issues/2705) | gam::regressions: 27 failures + 1 timeout at c1bceb7c6, grouped into 8 terminal causes (2 are not defects) | Open; acceptance evidence and current implementation under review |
| [#2695](https://github.com/SauersML/gam/issues/2695) | survival location-scale: the FEASIBILITY half is fixed by #2719 (24 rejects/seed -> 0, all six seeds move); what remains is objective rejection at a FIRST-ORDER gradient/objective disagreement -- rho is a ratio of linear terms and 0 of 75 linear-dominated attempts land near 1 | Open; acceptance evidence and current implementation under review |
| [#979](https://github.com/SauersML/gam/issues/979) | marginal-slope (binary + survival) bug: severe slowdown / survival hang in gamfit 0.1.189 | Open; acceptance evidence and current implementation under review |

## Release, test integrity and library surface

Coordinator: release_gate.

| Issue | Problem | Current session status |
| --- | --- | --- |
| [#2832](https://github.com/SauersML/gam/issues/2832) | The PyPI wheel gate refuses every release: `release-pypi` objects are shared with no other workflow, so the cache is always cold and `hits == 0` is unsatisfiable | Closed; published as `17fd4cc50`, all 9 cache parser tests pass on MSI; workflow YAML and embedded shell syntax pass |
| [#2829](https://github.com/SauersML/gam/issues/2829) | 1,080 deleted `pub` library functions are still absent, including `pub fn fit_gam` — the reachability sweep's universe was two binaries, and SPEC.md requires the Rust library be a product surface too | Open; acceptance evidence and current implementation under review |
| [#2818](https://github.com/SauersML/gam/issues/2818) | 227k lines and 2,321 tests were deleted by a reachability sweep whose criterion is vacuously true for test code — 148 issue-pinned regressions are unpinned, 9 on open issues | Open; acceptance evidence and current implementation under review |

## SAE numerical correctness, performance and research

Coordinator: root.

| Issue | Problem | Current session status |
| --- | --- | --- |
| [#2828](https://github.com/SauersML/gam/issues/2828) | Exact-A: the dense A is not dg/dtheta — v'Av off by 3.55x and /A.v/ 20x small, bit-identical across shas (last live item of #2820), plus the ungated null-space policy split between the dense and matrix-free adjoint solves | Open; acceptance evidence and current implementation under review |
| [#2826](https://github.com/SauersML/gam/issues/2826) | The block lane's GPU sits idle at the median (p50 = 0%, mean 1-16%) because the per-row state update and the CPU fallback GEMM are both serial | Open; acceptance evidence and current implementation under review |
| [#2825](https://github.com/SauersML/gam/issues/2825) | The block lane sets γ from an epoch's accumulators BEFORE refreshing the frames, so the alternation cycles and the EV plateau can certify a non-fixed point | Open; acceptance evidence and current implementation under review |
| [#2822](https://github.com/SauersML/gam/issues/2822) | gam-sae is broadly red: 69 failures in the lib suite and 89 in the integration binary, reproducing in isolation, invisible to every per-crate gate | Open; acceptance evidence and current implementation under review |
| [#2731](https://github.com/SauersML/gam/issues/2731) | PERF: the curved tier at the #2283 production shape (p=2048) reaches its refine fixed point in 30 min but cannot produce a criterion — streaming_exact_arrow_log_det fails at both charts=8 (292 s) and charts=32 (1947 s), in two different reduced-Schur paths | Open; acceptance evidence and current implementation under review |
| [#2576](https://github.com/SauersML/gam/issues/2576) | Overcomplete support lane: the evidence log/S/ costs ~50 s and its shifted CG converges in 192 iterations against a 20,000 cap — the cost is the inner fixed point (60 cycles / 608 s without recurring) and a deflation that removes -9.4% of the pilot variance and refuses at every error-bar target <= 1e-4 | Open; acceptance evidence and current implementation under review |
| [#2502](https://github.com/SauersML/gam/issues/2502) | Unsupervisedly learn an overcomplete manifold dictionary on a modern LLM | Open; acceptance evidence and current implementation under review |
| [#2333](https://github.com/SauersML/gam/issues/2333) | Route logdet_theta_adjoint_for_block through the Trace row-jet seam (whitening pre-fold needed) | Open; acceptance evidence and current implementation under review |
| [#2283](https://github.com/SauersML/gam/issues/2283) | Authoritative Eq-4 bits-at-R² @ 32K: theorem-faithful all-Rust hybrid row (split from #2233 Task 3) | Open; acceptance evidence and current implementation under review |
| [#2280](https://github.com/SauersML/gam/issues/2280) | Atlas-first manifold discovery: local charts + transition holonomy replace global-linear seeds and the fixed topology menu | Open; acceptance evidence and current implementation under review |
| [#2267](https://github.com/SauersML/gam/issues/2267) | Shipped manifold-SAE example times out past 900s at its own documented 635-row scale | Open; acceptance evidence and current implementation under review |
| [#2263](https://github.com/SauersML/gam/issues/2263) | Steering dosimetry calibration (items 1-3); structure_certificate blocked by #2548, NOT unusable — item 4 corrected in body | Open; acceptance evidence and current implementation under review |
| [#2234](https://github.com/SauersML/gam/issues/2234) | On-manifold causal steering: chart-coordinate interventions (dose in radians), collateral-damage curves, crosscoder transport test | Open; acceptance evidence and current implementation under review |
| [#2228](https://github.com/SauersML/gam/issues/2228) | SAC stagewise/manifold fit: RemlConvergenceError (inner solve stalls at fixed ρ) at LLM width p=4096 and on clean synthetic circles | Open; acceptance evidence and current implementation under review |
| [#2080](https://github.com/SauersML/gam/issues/2080) | Wide-p outer REML: bounded-but-slow (cubic width cost, #2087-amplified) + K>=2 entangled co-collapse (#2023) | Open; acceptance evidence and current implementation under review |
| [#2023](https://github.com/SauersML/gam/issues/2023) | Tiered decomposition as the architecture: sparse_dict linear bulk (Tier 1, K~10⁴) + evidence-K curved atoms (Tier 2) on the whitened residual, with a residual-factor ↔ linear ↔ curved migration ledger replacing PC reseeding | Open; acceptance evidence and current implementation under review |

## Shared blockers and observations

- The initial checkout predates 37 commits on remote main and contains extensive preexisting changes. Publication must preserve both those edits and subsequent upstream repairs.
- Latest GitHub Rust/Python builds fail before suite execution on unused SAE Hessian methods. Those methods are already wired or removed in the current checkout, which still needs compilation verification.
- Python Contracts continued after wheel failure, emitting missing-dependency errors and no inventory. Its steps now depend on successful setup while independent contracts continue after another contract fails; missing JUnit is explicitly reported as unmeasured.
- Initial remote attempt inherited unrelated OpenBLAS linker flags. Clearing those flags reuses the existing warm fingerprints. The next check found incomplete in-flight constrained-REML source and two outdated exact-A test initializers; the complete source and corrected carriers are being checked.
