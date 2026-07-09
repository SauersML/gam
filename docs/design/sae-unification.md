# SAE fit-path unification — one engine, one dictionary, one ledger

Status: RATIFIED (user, 2026-07-09). Implementation in increments; each keeps the
workspace green and pairs every deletion with its call-site re-point.

Thesis: ONE dictionary where a linear atom = Euclidean d=1 atom (atom.rs Linear
already shares EuclideanPatchEvaluator max_degree=1); ONE engine (inner
arrow-Schur Newton + outer REML evidence + birth/death ledger); dense-vs-streaming
is MEMORY-LAYOUT ADMISSION of the same engine (front_door + streaming_plan);
tiered is the warm-start/alternation SCHEDULE (residual tier = round-0 warm start
only; cofit alternation to joint stationarity is the fit); sparse_dict's
alternating LSQ survives only as the fixed-support linear-atom fast kernel inside
the inner solve. Public entry: sae_manifold_fit only. fit_tiered and
sparse_dictionary_fit public APIs are DELETED (no back-compat shims).

Unified state: per-row active sets (indices[N,s], gates[N,s], coords on support)
— the SaeTopKCurvedBudget shape. Dense certification (K<=P) = full-support
specialization. Gate liveness (Newton state vs read-only) and d_k are FIELDS of
the state, not lanes.

Linear fast kernel plug points: (1) code/gate solve -> solve_row_codes s x s
active-set ridge when all support atoms are Linear d=1 with read-only gates;
(2) decoder refresh -> MOD sparse normal equations (update.rs) as the d=1
specialization of the framed curved refresh; (3) routing -> TileScorer
top_s_online (never materialize N x K); (4) ONE SHARED REML variance component
for the linear block selected by fit_at_fixed_rho/efs_step (code_ridge/
decoder_ridge become initial rho, then REML-selected — kills the no-REML lane).

One SaeMigrationLedger: moves Birth (residual->linear->curved), Death (reverse),
Refuse; single evidence currency (REML delta + rank charge; tiered curved_charge
dl_bits unit); pc_reseed_events == 0 global invariant (births only from residual
factor pool). Subsumes structure_harvest proposals, tiered MigrationLedger, and
sparse_dict dead-atom revival (revival must pay evidence, not just RSS).

Increments (each green):
1. SaeAssignmentState internal type; dense SaeAssignment becomes a full-support
   materialization. No public change. Deletes: nothing.
2. Linear fast kernel routed through run_fixed_decoder_arrow_schur; shared-rho
   REML for the linear block; fit_sparse_dictionary becomes thin wrapper with a
   TEMPORARY bit-parity gate vs old update::run. Deletes: update.rs::run
   alternation loop body.
3. Unified SaeMigrationLedger. Deletes: tiered/fit.rs:71-249 ledger types.
4. fit_tiered -> seed policy + alternation cadence of the unified engine;
   re-point tests/test_examples_compose_tiers.py + FFI sae_manifold_fit_tiered
   callers in the same increment. Deletes: fit_tiered, TieredFitConfig/Report,
   FFI sae_manifold_fit_tiered, _sae_spectral.py wrapper.
5. Single public entry: _sae_manifold.py:3186 sparse_codes branch runs the
   unified linear schedule internally; re-point driver_1026_arms.py,
   test_sparse_dictionary_held_out_ev_1026.py, e2e_32k_real.py. Deletes:
   sparse_dictionary_fit / block_sparse_dictionary_fit public APIs (python+FFI),
   fit_sparse_dictionary public export.
6. Remove parity shim + dead public re-exports; front_door docs describe one
   engine. torch lane untouched (declared interop).

Risk pins (must stay green): streaming-plan laziness tests (no eager GPU probe;
never-OOM admission before any N x K alloc); front-door no-silent-substitution
tests (K>P topk stays CurvedStreaming); test_sparse_dictionary_held_out_ev_1026
through the new entry (REML must not regress linear EV, no magic ridge);
tiered_driver_runs_and_never_pc_reseeds + promotion tests; fixed_decoder lean-vs-
full parity 1407; cofit round-0 semantics tests.

Full seam-by-seam analysis with file:line cites lives in the session ledger
(2026-07-09 unification design run).
