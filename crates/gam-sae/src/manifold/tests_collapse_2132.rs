//! #2132 — the multi-atom manifold dictionary co-collapses on planted CIRCLE
//! MIXTURES: held-out reconstruction EV is LOW, DECREASES with K, and sits below
//! a trivial linear-PCA baseline on the exact curved structure the engine is
//! meant to capture. This is the issue's OWN acceptance bar, expressed as an
//! in-tree Rust test on HEAD source (no stale wheel, no MSI round-trip — the
//! owner's re-measure was blocked only by a day-stale `.so`).
//!
//! Data = SINGLE-ACTIVE planted circle mixture: each token lies on EXACTLY ONE of
//! `C` unit circles, each embedded in its own disjoint ambient 2-plane
//! `(dims 2c, 2c+1)`. The union of the C circles spans a 2C-dim subspace, so a
//! rank-2C linear PCA reconstructs every point exactly (up to noise) — the ceiling
//! the curved dictionary must MATCH. Circle membership is assigned by a hash of
//! the row index, NOT `row % k`, so the round-robin seed routing in `build_term`
//! is NOT oracle-aligned to the planted labels (no unfair advantage).
//!
//! The held-out EV is measured on the PRODUCTION out-of-sample path
//! (`sae_manifold_predict_oos` math): frozen fitted decoders, coordinates seeded
//! by the decoder-grid projection, softmax routing logits seeded from per-atom
//! projection residuals (the exact `seed_oos_softmax_logits_from_projection_residuals`
//! step the FFI runs — the earlier `tests_zoo_micro_local::cold_oos_ev` helper left
//! logits UNIFORM, understating the real predict path), then the fixed-decoder
//! arrow-Schur coordinate solve under the fit's own terminal ρ*.
//!
//! Acceptance (genuine red if the co-collapse persists):
//!   (a) held-out EV at K=2C does NOT drop below K=C (no co-collapse-in-K);
//!   (b) held-out EV at BOTH K=C and K=2C is at least the rank-2C held-out PCA
//!       baseline (curved reaches the linear reconstruction ceiling on its own
//!       curved structure).
//!
//! zz_measure discipline: eprintln every number; the asserts are the issue's bar.

use super::tests::global_ev;
use super::tests_startup_validation_1782::{Topo, objective_and_seed};
use super::{SaeManifoldRho, SaeManifoldTerm};
use crate::assignment::{AssignmentMode, SaeAssignment};
use gam_linalg::faer_ndarray::FaerSvd;
// `eval_efs` is a method of the `OuterObjective` trait (impl for
// SaeManifoldOuterObjective); the trait must be in scope to call it on the
// production objective, the same import the sibling #1782 seed-eval test uses.
use gam_solve::rho_optimizer::OuterObjective;
use ndarray::{Array2, ArrayView2};

/// splitmix64 mixer — deterministic, reproducible across threads / devices; no
/// RNG crate dependency (mirrors `pca_seed::splitmix_unit`).
fn splitmix_u64(mut z: u64) -> u64 {
    z = z.wrapping_add(0x9E3779B97F4A7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
    z ^ (z >> 31)
}

/// splitmix64 → `[0, 1)`.
fn splitmix01(z: u64) -> f64 {
    (splitmix_u64(z) >> 11) as f64 / (1u64 << 53) as f64
}

/// Single-active planted circle mixture: `n` rows, ambient dim `p`, `c` circles.
/// Row `r` belongs to circle `hash(r) % c` (NOT `r % k`, so the round-robin seed
/// routing is not oracle-aligned). Circle `j` lives in the disjoint 2-plane
/// spanned by ambient axes `(2j, 2j+1)`; a point is `(cosθ)e_{2j} + (sinθ)e_{2j+1}`
/// plus isotropic Gaussian-ish noise. `phase_key` decorrelates train vs test
/// angles so the two splits are independent draws from the same C circles.
fn planted_circle_mixture(n: usize, p: usize, c: usize, sigma: f64, phase_key: u64) -> Array2<f64> {
    assert!(p >= 2 * c, "need p >= 2C for disjoint circle planes");
    let mut z = Array2::<f64>::zeros((n, p));
    for row in 0..n {
        let circle = (splitmix_u64(row as u64 ^ 0x1234_5678_9abc_def0) % c as u64) as usize;
        let theta = std::f64::consts::TAU
            * splitmix01((row as u64).wrapping_mul(0x100000001b3) ^ phase_key);
        let (cos, sin) = (theta.cos(), theta.sin());
        z[[row, 2 * circle]] = cos;
        z[[row, 2 * circle + 1]] = sin;
        for col in 0..p {
            // Box–Muller-free: two independent uniforms averaged approximate a
            // zero-mean symmetric perturbation; scale by sigma. Deterministic.
            let u =
                splitmix01((row as u64) << 20 ^ (col as u64).wrapping_mul(0x9E3779B1) ^ phase_key);
            z[[row, col]] += sigma * (u - 0.5) * 2.0;
        }
    }
    z
}

/// Held-out PCA baseline: fit the top-`rank` right singular vectors + column mean
/// on TRAIN, project TEST (train-centered) onto them, reconstruct, and score EV
/// around the TEST mean (the issue's raw `ev(test, reconstruct(test))` convention).
fn pca_heldout_ev(train: ArrayView2<'_, f64>, test: ArrayView2<'_, f64>, rank: usize) -> f64 {
    let p = train.ncols();
    let mut mean = vec![0.0_f64; p];
    for col in 0..p {
        let mut acc = 0.0;
        for row in 0..train.nrows() {
            acc += train[[row, col]];
        }
        mean[col] = acc / train.nrows() as f64;
    }
    let mut centered = train.to_owned();
    for row in 0..centered.nrows() {
        for col in 0..p {
            centered[[row, col]] -= mean[col];
        }
    }
    let (_u, _s, vt_opt) = centered.svd(false, true).expect("PCA baseline SVD");
    let vt = vt_opt.expect("PCA baseline Vt");
    let r = rank.min(vt.nrows());
    // Reconstruct each test row: mean + Σ_{i<r} (x_c · v_i) v_i.
    let mut recon = Array2::<f64>::zeros(test.dim());
    for row in 0..test.nrows() {
        let mut xc = vec![0.0_f64; p];
        for col in 0..p {
            xc[col] = test[[row, col]] - mean[col];
        }
        for i in 0..r {
            let mut coeff = 0.0;
            for col in 0..p {
                coeff += xc[col] * vt[[i, col]];
            }
            for col in 0..p {
                recon[[row, col]] += coeff * vt[[i, col]];
            }
        }
        for col in 0..p {
            recon[[row, col]] += mean[col];
        }
    }
    global_ev(test, recon.view())
}

/// Faithful production out-of-sample held-out EV against a FITTED dictionary —
/// the `sae_manifold_predict_oos` math: cold coords seeded by decoder-grid
/// projection, softmax routing logits seeded from per-atom projection residuals,
/// then the fixed-decoder arrow-Schur coordinate solve under the fit's ρ*.
pub(crate) fn oos_heldout_ev(
    fitted_term: &SaeManifoldTerm,
    rho: &SaeManifoldRho,
    x: ArrayView2<'_, f64>,
) -> f64 {
    let n = x.nrows();
    let k = fitted_term.k_atoms();
    let p = x.ncols();
    let coords_blocks: Vec<Array2<f64>> = (0..k)
        .map(|atom| {
            let d = fitted_term.assignment.coords[atom].as_matrix().ncols();
            Array2::<f64>::zeros((n, d))
        })
        .collect();
    let manifolds: Vec<_> = (0..k)
        .map(|atom| fitted_term.assignment.coords[atom].manifold().clone())
        .collect();
    let assignment = SaeAssignment::from_blocks_with_mode_and_manifolds(
        Array2::<f64>::zeros((n, k)),
        coords_blocks,
        manifolds,
        fitted_term.assignment.mode.clone(),
    )
    .expect("OOS assignment");
    let mut term = SaeManifoldTerm::new(fitted_term.atoms.clone(), assignment).expect("OOS term");
    term.seed_coords_by_decoder_projection(x)
        .expect("decoder-projection seed");
    // Seed softmax routing logits from per-atom projection residuals — the exact
    // production step (`seed_oos_softmax_logits_from_projection_residuals`) the
    // uniform-logit `cold_oos_ev` harness omitted. Under uniform logits every OOS
    // row is a softmax blend of ALL K atoms (a near-mean reconstruction), which
    // understates the real predict path; residual-seeded logits route each row to
    // the atom that reconstructs it best before the coordinate solve.
    let tau = fitted_term.assignment.mode.temperature().max(1.0e-6);
    let mut logits = Array2::<f64>::zeros((n, k));
    let mut decoded = vec![0.0_f64; p];
    for row in 0..n {
        for atom in 0..k {
            term.atoms[atom].fill_decoded_row(row, &mut decoded);
            let mut err = 0.0_f64;
            for col in 0..p {
                let diff = x[[row, col]] - decoded[col];
                err += diff * diff;
            }
            logits[[row, atom]] = -err / tau;
        }
        let reference = logits[[row, k - 1]];
        for atom in 0..k {
            logits[[row, atom]] -= reference;
        }
    }
    term.assignment.logits.assign(&logits);
    let mut rho_oos = rho.clone();
    term.run_fixed_decoder_arrow_schur(x, &mut rho_oos, None, 24, 1.0, 1.0e-6)
        .expect("fixed-decoder OOS solve");
    let fitted = term.try_fitted().expect("OOS fitted");
    global_ev(x, fitted.view())
}

/// Run the full production outer cascade (`OuterProblem::run`, the FFI entry) for
/// a K-atom circle dictionary at the single-PCA-seed budget, returning the fitted
/// term + terminal ρ* and the native (in-sample) train EV.
fn fit_circle_dictionary(
    train: ArrayView2<'_, f64>,
    k: usize,
) -> (SaeManifoldTerm, SaeManifoldRho, f64) {
    let (mut objective, seed) =
        objective_and_seed(train, k, Topo::Circle, AssignmentMode::softmax(1.0));
    let n_params = seed.len();
    let result = gam_solve::rho_optimizer::OuterProblem::new(n_params)
        .with_initial_rho(seed)
        .with_max_iter(12)
        .with_seed_config(gam_problem::SeedConfig {
            max_seeds: 1,
            seed_budget: 1,
            ..Default::default()
        })
        .run(&mut objective, "SAE manifold")
        .expect("circle dictionary fit must not abort");
    objective
        .certify_outer_result(&result)
        .expect("circle dictionary outer result must certify the installed state");
    let fitted = objective.into_fitted().expect("outer fit was evaluated");
    let native_ev = global_ev(train, fitted.term.fitted().view());
    (fitted.term, fitted.rho, native_ev)
}

#[test]
fn zz_collapse_2132_heldout_ev_nondecreasing_and_beats_pca() {
    const C: usize = 4;
    const P: usize = 24;
    const SIGMA: f64 = 0.05;
    let train = planted_circle_mixture(1000, P, C, SIGMA, 0xA11CE);
    let test = planted_circle_mixture(600, P, C, SIGMA, 0xB0B);

    // rank-2C linear PCA ceiling on the identical held-out split — the union of
    // C disjoint-plane circles spans exactly 2C dims, so this reconstructs every
    // test point up to the planted noise. This is the bar the curved dictionary
    // must MATCH on its own curved structure.
    let pca_2c = pca_heldout_ev(train.view(), test.view(), 2 * C);
    eprintln!(
        "[collapse-2132] rank-{} held-out PCA EV = {pca_2c:.4}",
        2 * C
    );

    let (term_c, rho_c, native_c) = fit_circle_dictionary(train.view(), C);
    let ev_c = oos_heldout_ev(&term_c, &rho_c, test.view());
    eprintln!("[collapse-2132] K=C={C}: native_train_ev={native_c:.4} heldout_ev={ev_c:.4}");

    let (term_2c, rho_2c, native_2c) = fit_circle_dictionary(train.view(), 2 * C);
    let ev_2c = oos_heldout_ev(&term_2c, &rho_2c, test.view());
    eprintln!(
        "[collapse-2132] K=2C={}: native_train_ev={native_2c:.4} heldout_ev={ev_2c:.4}",
        2 * C
    );

    eprintln!(
        "[collapse-2132] SUMMARY: pca(rank-{})={pca_2c:.4} | curved K=C={ev_c:.4} K=2C={ev_2c:.4} \
         | dEV(K)={:.4}",
        2 * C,
        ev_2c - ev_c
    );

    assert!(
        ev_c.is_finite() && ev_2c.is_finite() && pca_2c.is_finite(),
        "held-out EVs must be finite (pca={pca_2c} K=C={ev_c} K=2C={ev_2c})"
    );

    // (a) No co-collapse in K: doubling the dictionary must not DEGRADE held-out
    // reconstruction (the issue's 0.26 -> 0.11 signature). Small numerical slack.
    assert!(
        ev_2c >= ev_c - 0.05,
        "#2132 co-collapse-in-K: held-out EV DROPPED from {ev_c:.4} (K={C}) to {ev_2c:.4} \
         (K={}) — more atoms made reconstruction worse",
        2 * C
    );

    // (b) The curved dictionary reaches the linear reconstruction ceiling on its
    // own curved structure at BOTH K. A slack of 0.05 absorbs noise/estimation;
    // the issue's failure was curved 0.23-0.26 vs PCA 0.55-0.73 — a ~0.3 gap, far
    // outside this slack.
    assert!(
        ev_c >= pca_2c - 0.05,
        "#2132: K={C} curved held-out EV {ev_c:.4} is below the rank-{} PCA ceiling {pca_2c:.4}",
        2 * C
    );
    assert!(
        ev_2c >= pca_2c - 0.05,
        "#2132: K={} curved held-out EV {ev_2c:.4} is below the rank-{} PCA ceiling {pca_2c:.4}",
        2 * C,
        2 * C
    );
}

/// #2132/#2228 — CHEAP fixed-ρ discriminator for the inner quotient stall.
///
/// The full EV-vs-K close driver runs THREE outer ρ searches (K=C, C+2, 2C), each
/// a 12-iteration BFGS over repeated inner solves — hours of walltime, and the
/// 13281964 trace showed it never even reached the sweep: it refused at the FIRST
/// fit's SEED evaluation with the SAE inner quotient stall (‖g‖=0.174 ≫ tol 3.6e-4,
/// ½λ²/scale=3.1e-6, terminal-polish bail → "refusing to rank an off-optimum Laplace
/// criterion"). This probe reproduces exactly that seed-evaluation refusal for a
/// FRACTION of the cost: a SINGLE `eval_efs(seed)` inner solve per small matched-K
/// (K=C) circle mixture — no outer loop at all. Seconds, not hours.
///
/// Run with `RUST_LOG=debug --nocapture` to surface the terminal-polish arbiter
/// lines ("terminal Newton bail: all backtracks rejected" vs "terminal Newton step
/// committed" vs "quotient solver refused"/"GMRES bail") that discriminate the root:
/// merit-rejects-valid-indefinite-Newton-step (the raw-‖g‖-only polish merit) vs an
/// objective↔gradient/pencil desync vs a preconditioner defect.
///
/// Acceptance: every seed evaluation must TERMINATE with a finite criterion — a
/// converged inner solve or a best-incumbent, never the off-optimum refusal that
/// blocks the whole #2132/#2228 SAE acceptance lane.
#[test]
fn manifold_circle_mixture_seed_eval_terminates_2132() {
    // The engine's `log::debug!` arbiter lines in `terminal_exact_newton_polish`
    // (bail / step-committed / quotient-solver-refused) are SILENTLY DROPPED by the
    // test harness unless a logger is installed — so a bare `--nocapture` run would
    // show the refusal but not WHY. Forward every record to stderr so the discriminator
    // lines surface without needing RUST_LOG (max level is set to Debug directly).
    struct ForwardingTestLogger;
    impl log::Log for ForwardingTestLogger {
        fn enabled(&self, _: &log::Metadata<'_>) -> bool {
            true
        }
        fn log(&self, record: &log::Record<'_>) {
            eprintln!("[{}] {}", record.level(), record.args());
        }
        fn flush(&self) {}
    }
    static FORWARDING_TEST_LOGGER: ForwardingTestLogger = ForwardingTestLogger;
    // Ignore the error when another test already installed a global logger.
    if log::set_logger(&FORWARDING_TEST_LOGGER).is_ok() {
        log::set_max_level(log::LevelFilter::Debug);
    }

    // (C, P, K, tag): matched-K planted circle mixtures FIRST (fast, known — K=C,
    // P=2C+2 keeps the C planes disjoint), then the #2267 OVER-COMPLETE lane last:
    // K=8 atoms on a C-circle mixture at P=32 (the curved-K=8 dense-softmax arm that
    // HANGS >40min on 508×32 in the #2267 timing runs, per implPERF). Matched arms
    // run first so their results are captured before any over-complete hang; the
    // START line before each eval names the culprit config if the job is
    // walltime-killed (a HANG here = the single inner solve itself has an unbounded
    // loop, distinct from an outer-search churn).
    const CONFIGS: [(usize, usize, usize, &str); 5] = [
        (2, 6, 2, "matched"),
        (3, 8, 3, "matched"),
        (4, 10, 4, "matched"),
        (2, 32, 8, "overcomplete_2267"),
        (4, 32, 8, "overcomplete_2267"),
    ];
    const N: usize = 508;
    const SIGMA: f64 = 0.05;

    let mut refusals: Vec<String> = Vec::new();
    for (c, p, k, tag) in CONFIGS {
        let train = planted_circle_mixture(N, p, c, SIGMA, 0xA11CE ^ c as u64);
        // Both routings the close driver / repros exercise: soft (softmax logits are
        // free Newton params — the #2267 default) AND hard TopK (support-sparse — no
        // logit is a Newton param, membership is the compact active-set whose
        // oscillation is the exchange-churn signature shared with the CTN joint lane).
        // top_k=1 mirrors the driver. Covering both means whichever mechanism the
        // driver hit — the indefinite-Hessian geometry stall OR top-k support
        // ping-pong — reproduces here, and the forwarded arbiter lines discriminate.
        for mode_idx in 0..2 {
            let (mode_label, mode) = match mode_idx {
                0 => ("softmax", AssignmentMode::softmax(1.0)),
                _ => ("topk1", AssignmentMode::top_k_support(1)),
            };
            eprintln!("[#2132 seed-eval] START C={c} K={k} P={p} routing={mode_label} tag={tag}");
            let (mut objective, seed) = objective_and_seed(train.view(), k, Topo::Circle, mode);
            match objective.eval_efs(&seed) {
                Ok(eval) => {
                    let steps_finite = eval.steps.iter().all(|v| v.is_finite());
                    eprintln!(
                        "[#2132 seed-eval] C={c} K={k} P={p} routing={mode_label} tag={tag}: \
                         cost={:.6e} n_steps={} steps_finite={steps_finite} cost_finite={}",
                        eval.cost,
                        eval.steps.len(),
                        eval.cost.is_finite()
                    );
                    if !eval.cost.is_finite() {
                        refusals.push(format!(
                            "C={c}/K={k}/{mode_label}: non-finite seed cost {:.6e}",
                            eval.cost
                        ));
                    }
                }
                Err(err) => {
                    eprintln!(
                        "[#2132 seed-eval] C={c} K={k} P={p} routing={mode_label} tag={tag}: REFUSED — {err}"
                    );
                    refusals.push(format!("C={c}/K={k}/{mode_label}: {err}"));
                }
            }
        }
    }
    assert!(
        refusals.is_empty(),
        "#2132/#2228: inner seed evaluation refused / went non-finite on a clean planted \
         circle mixture (the seed-evaluation stall the close driver never gets past): {refusals:?}"
    );
}
