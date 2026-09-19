//! Fit the overcomplete support lane on a real chart to an inner-certified
//! fixed point at fixed smoothing, then dump REAL fitted-atom artifacts for
//! plotting: per-atom topology + usage, Rust-decoded curve samples along each
//! selected atom, and the coordinates/rows of the real tokens on it.
//!
//! ```text
//! cargo run -p gam-sae --release --example support_fit_dump -- \
//!     chart.bin <rows> <cols> <k_atoms> <top_k> <out_dir>
//! ```

use gam_sae::front_door::{SaeFitLane, admit_topk_manifold};
use gam_sae::manifold::{
    SaeSupportSeedRequest, SaeSupportTermSeedRequest, build_sae_support_seed,
    build_sae_support_term_seed, resolve_support_auto_atoms, sae_support_effective_atom_dims,
};
use ndarray::{Array2, Axis};
use std::io::Write;
use std::time::Instant;

fn write_f64s(path: &str, values: &[f64]) -> Result<(), String> {
    let mut bytes = Vec::with_capacity(values.len() * 8);
    for value in values {
        bytes.extend_from_slice(&value.to_le_bytes());
    }
    std::fs::write(path, bytes).map_err(|error| format!("{path}: {error}"))
}

/// FNV-1a over raw bytes, with the published 64-bit offset basis and prime.
///
/// Identifies which array an arm actually read. Two charts of identical shape
/// are indistinguishable by every count- or shape-based check, and this
/// campaign compared two such charts as though they were one.
fn chart_digest(bytes: &[u8]) -> String {
    const FNV_OFFSET_BASIS: u64 = 0xcbf2_9ce4_8422_2325;
    const FNV_PRIME: u64 = 0x0000_0100_0000_01b3;
    let mut hash = FNV_OFFSET_BASIS;
    for &byte in bytes {
        hash ^= u64::from(byte);
        hash = hash.wrapping_mul(FNV_PRIME);
    }
    format!("{hash:016x}")
}

/// FNV digest of this executable's own bytes.
///
/// An arm's result line already carries the chart it read, because two charts
/// of identical size and different content once produced an EV that looked
/// like a finding. The same hazard exists one level up: two arms of an A/B can
/// share every argument and still differ, because one was built before a fix
/// and the other after. That happened on this lane -- a routing fix changed
/// what `exact` does during training, and a seed-0/seed-1 pair straddled it.
///
/// Reading `current_exe` costs a few milliseconds once per run and makes the
/// binary as checkable as the data. On failure the arm still runs and reports
/// `build=unknown`: a missing provenance field must not cost a measurement.
fn build_digest() -> String {
    let Ok(path) = std::env::current_exe() else {
        return "unknown".to_string();
    };
    match std::fs::read(path) {
        Ok(bytes) => chart_digest(&bytes),
        Err(_) => "unknown".to_string(),
    }
}

fn main() -> Result<(), String> {
    env_logger::init();
    let args: Vec<String> = std::env::args().collect();
    if !matches!(args.len(), 7 | 9 | 10 | 11 | 12 | 13 | 14 | 15 | 16 | 17 | 18 | 19 | 20) {
        return Err("usage: support_fit_dump <f64-le.bin> <rows> <cols> <k> <top_k> <out_dir> [test.bin test_rows] [reserved] [seed]".into());
    }
    // Seed for BOTH the support cold start and the term seed. A single fit
    // cannot support a gap claim, so this has to be varied and reported.
    let seed_arg: u64 = if args.len() >= 11 {
        args[10].parse().map_err(|e| format!("seed: {e}"))?
    } else {
        0
    };
    // Which model this run fits must be legible from the command line. A
    // rebuild once changed it silently, and the resulting runs were reported as
    // a fixed-smoothing confirmation while actually alternating REML.
    // Fixed smoothing is removed. It is not a selection: it pins every atom at
    // lambda = 1, and measured at K=11010 that leaves 1223 atoms with zero
    // effective df where per-atom REML leaves 569, for a dictionary that
    // differs by 39% in norm. Passing `fixed` is now an ERROR rather than a
    // silent downgrade -- a run that asked for one model and got another is
    // how this campaign ended up reporting fixed-smoothing numbers from an
    // alternating fit, which is why the mode is echoed on the line above.
    if args.get(11).map(String::as_str) == Some("fixed") {
        return Err(
            "fixed smoothing was removed: it pins lambda = 1 for every atom, \
             which is not a selection. Pass `reml`."
                .into(),
        );
    }
    let reml_arg: bool = true;
    // Coordinate-prior precision. Fixed at 1.0 for the whole campaign because
    // nothing selected it; `0` removes the prior entirely, which is the arm the
    // containment argument needs.
    let alpha_arg: f64 = if args.len() >= 13 {
        args[12].parse().map_err(|e| format!("alpha: {e}"))?
    } else {
        1.0
    };
    // Smoothing strength. Same unselected-penalty asymmetry as `alpha`: this
    // charges `lambda * tr(B' S B)` on the decoder while the TopK SAE baseline
    // carries no decoder penalty at all.
    let lambda_arg: f64 = if args.len() >= 14 {
        args[13].parse().map_err(|e| format!("lambda: {e}"))?
    } else {
        1.0
    };
    println!("alpha: {alpha_arg}  lambda: {lambda_arg}");
    println!("mode: REML per-atom smoothing alternation");
    let rows: usize = args[2].parse().map_err(|e| format!("rows: {e}"))?;
    let cols: usize = args[3].parse().map_err(|e| format!("cols: {e}"))?;
    let k_atoms: usize = args[4].parse().map_err(|e| format!("k: {e}"))?;
    let top_k: usize = args[5].parse().map_err(|e| format!("top_k: {e}"))?;
    let out_dir = &args[6];
    std::fs::create_dir_all(out_dir).map_err(|e| format!("{out_dir}: {e}"))?;

    let bytes = std::fs::read(&args[1]).map_err(|e| format!("{}: {e}", args[1]))?;
    if bytes.len() != rows * cols * 8 {
        return Err(format!("chart holds {} bytes != rows*cols*8", bytes.len()));
    }
    println!(
        "train chart: {} rows={rows} cols={cols} digest={}",
        args[1],
        chart_digest(&bytes)
    );
    let data: Vec<f64> = bytes
        .chunks_exact(8)
        .map(|c| f64::from_le_bytes(c.try_into().expect("8-byte chunk")))
        .collect();
    let target = Array2::from_shape_vec((rows, cols), data).map_err(|e| e.to_string())?;

    // "auto" is the round-robin linear/euclidean/periodic portfolio. Pinning
    // every atom to "linear" reduces this model to the TopK SAE it contains,
    // which is how the optimizer is measured separately from the manifold
    // hypothesis.
    let topology_arg = if args.len() >= 10 && args[9] != "0" {
        args[9].clone()
    } else {
        "auto".to_string()
    };
    let mut atom_basis = vec![topology_arg.clone(); k_atoms];
    if topology_arg == "auto" {
        resolve_support_auto_atoms(&mut atom_basis);
    } else if topology_arg == "earned" {
        // No atom is curved because of its index. Every atom starts on the
        // simplest chart that can carry a coordinate; the occupancy census
        // and the image-degeneracy test (both derived, no constants) remove
        // closure that the data does not support, and the remaining curved
        // capacity has to be earned by the fit rather than granted by a
        // modulus. This is the control the whole portfolio question needs:
        // if forced curvature beats earned curvature, the forcing was
        // carrying information; if not, it was noise with a shape.
        for basis in atom_basis.iter_mut() {
            *basis = "euclidean".to_string();
        }
    } else if topology_arg == "rich" {
        // Every topology the lane can fit, weighted by what today's censuses
        // and steering tables earned: linear stays the routing base, the
        // strong steerers (periodic, euclidean) and the mega-atom producer
        // (embedded sphere) carry the curved share, and the torus enters at
        // one atom in eight -- its 49-function basis is exactly the capacity
        // a working shrinkage ladder exists to police.
        for (atom, basis) in atom_basis.iter_mut().enumerate() {
            *basis = match atom % 8 {
                0 | 1 => "linear",
                2 => "euclidean",
                3 | 4 => "periodic",
                5 | 6 => "sphere",
                7 | _ => "torus",
            }
            .to_string();
        }
    } else if topology_arg == "mixed" {
        // `auto` is a 1-D-only portfolio (linear/euclidean/periodic), which is
        // why every atom this harness has ever fitted is a CURVE. `mixed` spans
        // both intrinsic dimensions so the dictionary can actually contain
        // surfaces: the two closed 2-manifolds below need only a coordinate
        // pair, so they stay comparable to the 1-D atoms at the same `top_k`.
        for (atom, basis) in atom_basis.iter_mut().enumerate() {
            *basis = match atom % 5 {
                0 => "linear",
                1 => "euclidean",
                2 => "periodic",
                // sphere-only 2-D rung. `torus` carries a 7x7 = 49-function
                // harmonic basis (6,272 decoder params/atom, 24.5x a linear
                // atom) against ~500 rows, so its per-atom gram is near
                // rank-deficient — a natural source of the indefinite Hessian
                // that blocks d>=2. `sphere` is 7 basis functions, comparable to
                // the curved 1-D atoms' 3.
                3 | _ => "sphere",
            }
            .to_string();
        }
    }
    println!("topology: {topology_arg}");
    // Intrinsic dimension comes from each atom's OWN topology, not from one
    // constant for the whole dictionary -- pinning this to 1 is what made every
    // fitted atom a curve regardless of the topology requested.
    fn atom_dim_for_basis(basis: &str) -> usize {
        match basis {
            // Public dim names the INTRINSIC dimension (2 for a sphere); the
            // seed pipeline maps it to the embedded 3-wide ambient chart
            // itself. Passing 3 here is refused by the same contract.
            "sphere" => 2,
            "torus" | "projective_plane" | "klein_bottle" => 2,
            _ => 1,
        }
    }
    // Decode-grid helper for the embedded sphere: a (lat, lon) lattice mapped
    // to ambient unit vectors, so the dump samples ON the manifold. `side x
    // side` rows, 3 columns.
    /// Near-uniform sphere sampling (Fibonacci lattice): the census probe
    /// must weight the manifold by AREA, and a (lat, lon) grid does not --
    /// it crowds points at its coordinate poles even though the embedded
    /// sphere itself has none.
    fn sphere_fibonacci(count: usize) -> Vec<f64> {
        let golden = (1.0 + 5.0_f64.sqrt()) / 2.0;
        let mut out = Vec::with_capacity(count * 3);
        for i in 0..count {
            let z = 1.0 - 2.0 * (i as f64 + 0.5) / count as f64;
            let radius = (1.0 - z * z).max(0.0).sqrt();
            let phi = 2.0 * std::f64::consts::PI * (i as f64 / golden).fract();
            out.push(radius * phi.cos());
            out.push(radius * phi.sin());
            out.push(z);
        }
        out
    }

    fn sphere_grid(side: usize) -> Vec<f64> {
        let mut out = Vec::with_capacity(side * side * 3);
        for i in 0..side {
            // open at the poles to avoid duplicated rows
            let lat = -std::f64::consts::FRAC_PI_2
                + std::f64::consts::PI * (i as f64 + 0.5) / side as f64;
            for j in 0..side {
                let lon = 2.0 * std::f64::consts::PI * j as f64 / (side - 1) as f64;
                out.push(lat.cos() * lon.cos());
                out.push(lat.cos() * lon.sin());
                out.push(lat.sin());
            }
        }
        out
    }
    let atom_dim: Vec<usize> = atom_basis
        .iter()
        .map(|basis| atom_dim_for_basis(basis))
        .collect();
    {
        let mut two_d = 0usize;
        for dim in &atom_dim {
            if *dim >= 2 {
                two_d += 1;
            }
        }
        println!("intrinsic dims: {two_d} of {k_atoms} atoms are 2-D");
    }
    let effective = sae_support_effective_atom_dims(&atom_basis, &atom_dim)?;
    let d_max = effective.iter().copied().max().unwrap_or(1);
    let admission = admit_topk_manifold(rows, cols, k_atoms, d_max, top_k)?;
    if admission.lane != SaeFitLane::CurvedStreaming {
        return Err(format!("expected CurvedStreaming; got {:?}", admission.lane));
    }
    let mean = target.mean_axis(Axis(0)).ok_or("empty target")?;
    let centered = &target - &mean;

    let seed = build_sae_support_seed(SaeSupportSeedRequest {
        target: centered.view(),
        atom_basis: &atom_basis,
        atom_dim: &atom_dim,
        support_k: top_k,
        random_state: seed_arg,
        admission,
    })?;
    let retained = seed.retained_atom_indices.clone();
    let retained_basis: Vec<String> =
        retained.iter().map(|&atom| atom_basis[atom].clone()).collect();
    let mut term_seed = build_sae_support_term_seed(SaeSupportTermSeedRequest {
        assignment: seed.assignment,
        atom_basis: retained_basis.clone(),
        atom_dim: retained.iter().map(|&atom| atom_dim[atom]).collect(),
        output_dim: cols,
        random_state: seed_arg,
    })?;
    let k_ret = term_seed.term.k_atoms();
    // Ambient-sphere axes are EXEMPT from the per-axis Gaussian prior:
    // ||u|| = 1 makes its energy constant on the manifold, so alpha there is
    // pure normal-direction noise. alpha == 0.0 is the typed exemption the
    // MacKay update respects.
    let ard: Vec<Vec<f64>> = (0..k_ret)
        .map(|atom| {
            let axis_alpha = if retained_basis[atom] == "sphere" {
                0.0
            } else {
                alpha_arg
            };
            vec![axis_alpha; term_seed.term.assignment.atom_coord_dim(atom)]
        })
        .collect();
    let lambda: Vec<f64> = vec![lambda_arg; k_ret];
    println!("seeded: retained {k_ret} of {k_atoms}");

    // Which tolerance the initial fit actually certified at, and how many
    // cycles it consumed getting there. The inner fixed point takes no cycle
    // budget (#2576); these are the dose, and they differ by up to 120 cycles
    // across arms this issue compares directly.
    let certified_tolerance = std::cell::Cell::new(1.0e-4_f64);
    let escalation_cycles = std::cell::Cell::new(0_usize);
    // "exact_train" arms the exact affine ranking BEFORE the initial solve, so
    // the training-time support updates inside `solve_fixed_point` use it too.
    // "exact" (armed later, beside pricing) affects held-out scoring only, and
    // the two are separate tokens so the effects stay separable.
    if args.iter().any(|arg| arg == "exact_train") {
        term_seed.term.set_exact_affine_ranking(true);
        println!("exact affine ranking: armed BEFORE the initial solve");
    }
    let t0 = Instant::now();
    // Per-atom REML by Fellner-Schall, alternated with the inner fit. The inner
    // certificate is a statement at fixed smoothing, so lambda moves only out
    // here, between certified fits. The loop stops on lambda's own relative
    // movement -- once smoothing shifts by less than the inner tolerance, the
    // fit it is selected from cannot resolve the difference.
    let mut lambda = lambda;
    let mut edf_prev = term_seed.term.effective_curvature_df(&lambda)?;
    let mut previous_move = f64::INFINITY;
    // Per-quantity stall (#2502): alpha freezes when ITS OWN movement stops
    // decreasing, so lambda's continued progress cannot drag the coordinate
    // prior through extra drift rounds. Same derived rule, per quantity.
    let mut previous_ard_move = f64::INFINITY;
    let mut ard_frozen = false;
    // A refused fit still holds a usable model: the objective converges long before
    // the KKT test does (measured on the all-linear arm -- objective flat to 2e-5
    // relative over the last 170 of 2000 cycles, while `raw KKT rel` sat at 2.7e-4
    // against a 1e-4 request and `max_change` stayed pinned at 8.578e-1, a parameter
    // move the objective does not see). Returning `Err` there throws the whole fit
    // away; three hours of compute produced no artifact at all. Report the miss
    // loudly, then re-solve at the tolerance the iterate actually reached so the
    // caller gets a real report and the atoms can be dumped.
    let mut report = match term_seed.term.solve_fixed_point(
        centered.view(),
        &lambda,
        &ard,
        1.0e-4,
        1.0,
    ) {
        Ok(report) => report,
        Err(error) => {
            eprintln!("[fit] NOT CONVERGED at 1e-4: {error}");
            eprintln!("[fit] accepting the stalled iterate; downstream numbers carry this caveat");
            term_seed
                .term
                // Escalate the tolerance until one closes. Both limbs are
                // relative to the objective scale, so a fit still moving fast fails
                // even a loose request: measured, a 2-cycle toy moved 4.07e5 against
                // a 2.3e4 threshold at 1e-2. Report which tolerance certified, so
                // the caveat travels with the numbers.
                .solve_fixed_point(centered.view(), &lambda, &ard, 1.0e-2, 1.0)
                .inspect(|report| {
                    certified_tolerance.set(1.0e-2);
                    escalation_cycles.set(escalation_cycles.get() + report.iterations);
                })
                .or_else(|first| {
                    let mut last = first;
                    // The last rung certifies nothing and says so. Without it a
                    // converged fit whose certificate limb is 1e20 -- #2517 on a
                    // flat coordinate direction -- is discarded entirely, while
                    // the held-out path in this same file scores the identical
                    // situation and labels it.
                    for tolerance in [1.0e-1_f64, 1.0, 1.0e1, 1.0e2, 1.0e3, 1.0e4, f64::MAX] {
                        match term_seed.term.solve_fixed_point(
                            centered.view(),
                            &lambda,
                            &ard,
                            tolerance,
                            1.0,
                        ) {
                            Ok(report) => {
                                if tolerance == f64::MAX {
                                    eprintln!(
                                        "[fit] ACCEPTED UNCERTIFIED: no rung certified; \
                                         downstream numbers carry this caveat"
                                    );
                                } else {
                                    eprintln!(
                                        "[fit] certified only at tolerance {tolerance:.0e}"
                                    );
                                }
                                certified_tolerance.set(tolerance);
                                escalation_cycles.set(
                                    escalation_cycles.get() + report.iterations,
                                );
                                return Ok(report);
                            }
                            Err(error) => last = error,
                        }
                    }
                    Err(last)
                })?
        }
    };
    // Optional decoder-strategy arg: "fista" runs the accelerated parallel
    // decoder update (6 majorized passes/cycle) instead of the colour-class
    // sweep -- a typed knob for the A/B, never an environment variable.
    // Optional topology arg: "unroll" arms the occupancy census between
    // REML rounds (#2502). Opt-in so an A/B pair shares this exact binary.
    let unroll_arg = args.iter().any(|arg| arg == "unroll");
    if unroll_arg {
        println!("topology unroll: armed");
    }
    // Optional admission-pricing arg: "price" charges every routing decision
    // the amortized description length of the atom's parameters, at the noise
    // floor certified by the initial solve (#2502: unpriced SSE admission is
    // why held-out EV degrades monotonically with the d>=2 atom share).
    // Every knob is matched anywhere in the argument list, so an unknown
    // trailing token can only be a typo -- and a typo that is ignored
    // disarms a mechanism while the arm still reports a number. Reject it.
    const KNOB_TOKENS: [&str; 13] =
        ["unroll", "pool", "joint", "price", "usage", "vark", "fista", "exact",
         "refine3", "refine9", "exact_train", "none", "peel"];
    for arg in args.iter().skip(15) {
        if !KNOB_TOKENS.contains(&arg.as_str()) {
            return Err(format!(
                "unrecognised trailing token {arg:?}; known knobs: {KNOB_TOKENS:?}"
            ));
        }
    }
    // "refineN" multiplies the routing grid's per-atom width. Measured with the
    // coordinate solve removed, tripling it is worth up to +0.0131 on a
    // euclidean dictionary; the point of the knob is to find out what it is
    // worth with the solve in place.
    for (token, factor) in [("refine3", 3usize), ("refine9", 9usize)] {
        if args.iter().any(|arg| arg == token) {
            term_seed.term.set_grid_refinement(factor);
            println!("routing grid refinement: {factor}x");
        }
    }
    // "exact" ranks affine atoms at their closed-form optimal coordinate during
    // greedy selection instead of at the best grid point. Measured worth 0.0103
    // held-out on a linear dictionary; opt-in so the A/B shares this binary.
    if args.iter().any(|arg| arg == "exact") {
        term_seed.term.set_exact_affine_ranking(true);
        println!("exact affine ranking: armed");
    }
    let price_arg = args.iter().any(|arg| arg == "price");
    if price_arg {
        let sigma2 = 2.0 * report.objective / ((rows * cols) as f64);
        term_seed.term.set_admission_dof_pricing(Some(sigma2));
        println!("admission pricing: armed (sigma2 = {sigma2:.6e})");
    }
    // Optional support arg: "vark" makes L0 per-token under priced
    // admission -- the router admits atoms while the priced gain is
    // positive, so capacity follows token complexity instead of a constant.
    // "usage" arms the usage amortization on top of pricing; the two terms
    // are reported apart because they disagree by portfolio.
    if args.iter().any(|arg| arg == "usage") {
        term_seed.term.set_admission_usage_amortization(true);
        println!("admission usage amortization: armed");
    }
    let vark_arg = args.iter().any(|arg| arg == "vark");
    if vark_arg {
        if !price_arg {
            return Err("vark requires price".into());
        }
        term_seed.term.set_variable_priced_support(true);
        println!("variable priced L0: armed");
    }
    if args.iter().any(|arg| arg == "fista") {
        term_seed.term.set_decoder_fista_passes(Some(6));
        println!("decoder strategy: FISTA (6 passes/cycle)");
    }
    let pool_arg = args.iter().any(|arg| arg == "pool");
    if pool_arg {
        println!("pooled smoothing: armed");
    }
    let joint_arg = args.iter().any(|arg| arg == "joint");
    if joint_arg {
        println!("joint (lambda, alpha) fixed point: armed");
    }
    let mut ard = ard;
    while reml_arg {
        // "joint" solves (lambda, alpha) to mutual self-consistency at a
        // frozen fit instead of taking one step of each per refit; the
        // alternation's feedback is what carried alpha to 134 at 1M rows.
        let (updated, joint_ard) = if joint_arg {
            let (lam, ard_next, sweeps) = term_seed.term.joint_hyperparameter_fixed_point(
                centered.view(),
                &lambda,
                &ard,
                1.0e-3,
            )?;
            println!("joint hyperparameter fixed point: {sweeps} sweeps");
            (lam, Some(ard_next))
        } else {
            (
                term_seed
                    .term
                    .fellner_schall_smoothing(centered.view(), &lambda)?,
                None,
            )
        };
        // The alpha update is LIVE again: the runaway that disabled it was the
        // crude fixed point's constant numerator (`n / (sum t^2 + tr H^-1)`
        // drives alpha -> infinity whenever the tangent is weak, measured
        // median 1 -> 38 -> 78). The update now iterates MacKay's gamma form,
        // whose numerator is the well-determined count and erases itself as
        // alpha grows, so the coupled alternation has a finite attractor.
        let updated_ard = match joint_ard {
            Some(joint) => joint,
            None => term_seed.term.mackay_ard_precisions(&ard)?,
        };
        let ard_move = updated_ard
            .iter()
            .zip(ard.iter())
            .flat_map(|(new_atom, old_atom)| new_atom.iter().zip(old_atom.iter()))
            .map(|(new, old)| (new.ln() - old.ln()).abs())
            .fold(0.0_f64, f64::max);
        let lambda_move = updated
            .iter()
            .zip(lambda.iter())
            .map(|(new, old)| (new.ln() - old.ln()).abs())
            .fold(0.0_f64, f64::max);
        // Convergence is measured in the effective degrees of freedom, which is
        // BOUNDED by the basis size, not in log lambda, which is not. When an
        // atom's curvature is unsupported the Fellner-Schall update correctly
        // sends its lambda to infinity -- the atom is shrinking onto the penalty
        // null space -- so `max |d log lambda|` never settles even though the
        // FITTED FUNCTION has stopped moving. Measured: log-lambda movement
        // stalls at 0.25 while lambda_max climbs 12 -> 94 without converging.
        // `tau_k - M0_k` saturates at that limit, so it is the quantity that
        // actually says whether the fit has stopped changing.
        // "pool" replaces K independent smoothing estimates with a shared
        // scale plus the deviation each atom's own effective df supports --
        // the repair the overcompleteness result calls for.
        let updated = if pool_arg {
            let edf_for_pool = term_seed.term.effective_curvature_df(&updated)?;
            term_seed.term.pooled_smoothing(&updated, &edf_for_pool)?
        } else {
            updated
        };
        let edf_now = term_seed.term.effective_curvature_df(&updated)?;
        let edf_move = edf_now
            .iter()
            .zip(edf_prev.iter())
            .map(|(new, old)| (new - old).abs())
            .fold(0.0_f64, f64::max);
        edf_prev = edf_now;
        let move_size = edf_move.max(ard_move);
        let quantiles = {
            let mut sorted = updated.clone();
            sorted.sort_by(|a, b| a.partial_cmp(b).expect("finite lambda"));
            (
                sorted[0],
                sorted[sorted.len() / 2],
                sorted[sorted.len() - 1],
            )
        };
        let ard_flat = {
            let mut flat: Vec<f64> = updated_ard.iter().flatten().copied().collect();
            flat.sort_by(|a, b| a.partial_cmp(b).expect("finite alpha"));
            flat
        };
        println!(
            "REML round: max |d edf| = {move_size:.4e} (lambda {:.4e}, alpha {ard_move:.4e}); \
             lambda min/med/max = {:.3e}/{:.3e}/{:.3e}; alpha min/med/max = {:.3e}/{:.3e}/{:.3e}",
            lambda_move,
            quantiles.0,
            quantiles.1,
            quantiles.2,
            ard_flat[0],
            ard_flat[ard_flat.len() / 2],
            ard_flat[ard_flat.len() - 1]
        );
        // Two ways this loop is finished, and the second is the one that fires.
        //
        // The inner fit certifies to a RELATIVE tolerance, so successive edf
        // readings carry a floor of that solve's own resolution -- measured, the
        // movement falls 1.07 -> 0.013 and then oscillates in [0.012, 0.015]
        // indefinitely while lambda_max stabilises at ~112 and the median at
        // ~2.16. The fit has converged; the absolute threshold is simply finer
        // than the measurement. Stopping when the movement is no longer
        // DECREASING detects that floor without naming it, so no stall count and
        // no floor constant enter the criterion.
        // Stop when the round's improvement is no longer resolvable against
        // the inner solve's own relative tolerance: below that, "better" is
        // indistinguishable from the certificate the fit was measured with.
        let improvement = previous_move - move_size;
        if !(move_size > 1.0e-4)
            || improvement <= previous_move.min(move_size) * 1.0e-4
        {
            break;
        }
        previous_move = move_size;
        lambda = updated;
        // #2502 occupancy-earned topology, once: unroll loops whose routed
        // tokens occupy at most half the circle, then let the next inner
        // solve refit the freed capacity on the arc the data owns. The edf
        // baseline is reset because converted atoms lawfully jump.
        if unroll_arg {
            // To its own fixed point: each conversion changes routing enough
            // to expose the next marginal loop, so a single pass leaves a
            // drip that never ends and keeps resetting the stall baseline.
            // Conversions are irreversible, so this terminates.
            let mut total = 0usize;
            loop {
                let unrolled = term_seed.term.convert_underoccupied_loops(2502)?;
                if unrolled.is_empty() {
                    break;
                }
                total += unrolled.len();
            }
            if total > 0 {
                // No stall-baseline reset: the conversion's edf jump is real
                // movement and the relative rule above absorbs it. Resetting
                // made a one-atom-per-round drip immortal.
                println!("topology reroute: unrolled {total} under-occupied loops");
            }
        }
        if !ard_frozen && ard_move >= previous_ard_move {
            ard_frozen = true;
            println!("alpha frozen at its own stall (move {ard_move:.4e})");
        }
        if !ard_frozen {
            previous_ard_move = ard_move;
            ard = updated_ard;
        }
        // The same acceptance ladder as the initial solve: a re-solve that
        // refuses still holds a usable iterate, and returning Err here is what
        // discarded a four-round REML arm at its round 4.
        report = match term_seed.term.solve_fixed_point(
            centered.view(),
            &lambda,
            &ard,
            1.0e-4,
            1.0,
        ) {
            Ok(report) => report,
            Err(error) => {
                eprintln!("[reml] round solve NOT CONVERGED at 1e-4: {error}");
                let mut accepted = None;
                // Each rung's refusal is carried, not dropped: if the whole
                // ladder fails, the reason the LOOSEST rung refused is the only
                // evidence about why this round is unrecoverable, and a silent
                // arm here would print "unrecoverable" with nothing behind it.
                // (Also: an empty error arm fails the root build script's ban
                // scanner, which blocks every wheel build in the repository.)
                let mut last_refusal = error;
                for tolerance in [1.0e-2_f64, 1.0e-1, 1.0, 1.0e1, 1.0e2, 1.0e3, 1.0e4] {
                    match term_seed.term.solve_fixed_point(
                        centered.view(),
                        &lambda,
                        &ard,
                        tolerance,
                        1.0,
                    ) {
                        Ok(report) => {
                            eprintln!("[reml] round certified only at tolerance {tolerance:.0e}");
                            accepted = Some(report);
                            break;
                        }
                        Err(rung_refusal) => {
                            eprintln!("[reml] rung {tolerance:.0e} refused: {rung_refusal}");
                            last_refusal = rung_refusal;
                        }
                    }
                }
                match accepted {
                    Some(report) => report,
                    None => {
                        eprintln!(
                            "[reml] round unrecoverable (loosest rung 1e4 refused: {last_refusal}); \
                             keeping the previous round's report and stopping the ladder"
                        );
                        break;
                    }
                }
            }
        };
    }
    // Evidence-supported curvature degrees of freedom per atom: the census that
    // says whether an atom's bend is paid for, rather than whether it exists.
    let edf = term_seed.term.effective_curvature_df(&lambda)?;
    let mut sorted_edf = edf.clone();
    sorted_edf.sort_by(|a, b| a.partial_cmp(b).expect("finite edf"));
    println!(
        "curvature edf: min {:.4} median {:.4} p90 {:.4} max {:.4}; atoms with edf<0.01: {}",
        sorted_edf[0],
        sorted_edf[sorted_edf.len() / 2],
        sorted_edf[(sorted_edf.len() * 9) / 10],
        sorted_edf[sorted_edf.len() - 1],
        edf.iter().filter(|value| **value < 0.01).count()
    );
    write_f64s(&format!("{out_dir}/curvature_edf.bin"), &edf)?;
    // The per-atom smoothing the fit ACTUALLY used -- without it a wiggly
    // curve in a gallery cannot be told apart from an under-smoothed one.
    write_f64s(&format!("{out_dir}/lambda.bin"), &lambda)?;
    {
        // Full fitted state, enough to rehydrate the model outside this
        // process (#2567): per-atom decoder blocks (concatenated row-major,
        // widths from atom_basis_size), the support grid, and every row's
        // compact coordinate block.
        let mut decoder_flat: Vec<f64> = Vec::new();
        let mut decoder_meta: Vec<f64> = Vec::new();
        for atom in 0..k_ret {
            let block = term_seed.term.atoms[atom].decoder_coefficients();
            decoder_meta.push(block.nrows() as f64);
            decoder_flat.extend(block.iter().copied());
        }
        write_f64s(&format!("{out_dir}/decoder_blocks.bin"), &decoder_flat)?;
        write_f64s(&format!("{out_dir}/decoder_rows.bin"), &decoder_meta)?;
        let mut support_flat: Vec<f64> = Vec::with_capacity(rows * top_k);
        let mut coords_flat: Vec<f64> = Vec::new();
        let mut coords_len: Vec<f64> = Vec::with_capacity(rows);
        for row in 0..rows {
            for &atom in term_seed.term.assignment.support_indices(row) {
                support_flat.push(atom as f64);
            }
            let rc = term_seed.term.assignment.coords_row(row);
            coords_len.push(rc.len() as f64);
            coords_flat.extend_from_slice(rc);
        }
        write_f64s(&format!("{out_dir}/support.bin"), &support_flat)?;
        write_f64s(&format!("{out_dir}/coords.bin"), &coords_flat)?;
        write_f64s(&format!("{out_dir}/coords_len.bin"), &coords_len)?;
    }
    println!(
        "inner CERTIFIED in {} cycles, {:.0}s, objective {:.4e}",
        report.iterations,
        t0.elapsed().as_secs_f64(),
        report.objective
    );
    let term = &term_seed.term;

    // reconstruction EV at the certified point
    let fitted = term.reconstruct()?;
    let res: f64 = centered
        .iter()
        .zip(fitted.iter())
        .map(|(t, f)| (t - f) * (t - f))
        .sum();
    let tot: f64 = centered.iter().map(|v| v * v).sum();
    println!("centered train EV = {:.4}", 1.0 - res / tot);

    if args.len() >= 9 {
        let te_rows: usize = args[8].parse().map_err(|e| format!("test rows: {e}"))?;
        let te_bytes = std::fs::read(&args[7]).map_err(|e| format!("{}: {e}", args[7]))?;
        if te_bytes.len() != te_rows * cols * 8 {
            println!("HELDOUT skipped: bad size");
        } else {
            let te_digest = chart_digest(&te_bytes);
            let te: Vec<f64> = te_bytes.chunks_exact(8)
                .map(|c| f64::from_le_bytes(c.try_into().expect("8"))).collect();
            let x_test = Array2::from_shape_vec((te_rows, cols), te).map_err(|e| format!("{e}"))?;
            let centered_test = &x_test - &mean;
            let mut te_term = term_seed.term.reroute_fixed_decoder_ard(centered_test.view(), top_k, 0, &ard)?;
            // A held-out solve that misses its KKT bound by a sliver still holds a
            // perfectly scoreable iterate -- refusing to SCORE it threw away the one
            // number a 45-minute REML arm existed to produce (measured: refused at
            // rel 1.2047e-4 against a 1e-4 bound). Mirror the fit-side ladder:
            // escalate the tolerance, and report which rung certified so the
            // caveat travels with the number; only an ERROR other than
            // non-recurrence stays fatal to the eval.
            let heldout = (|| {
                let mut last_error = String::new();
                for tolerance in [1.0e-4, 2.0e-4, 5.0e-4, 1.0e-3] {
                    match te_term.solve_coordinates_fixed_decoder(
                        centered_test.view(),
                        &ard,
                        tolerance,
                        1.0,
                    ) {
                        Ok(rep) => {
                            if tolerance > 1.0e-4 {
                                println!("HELDOUT certified only at tolerance {tolerance:.0e}");
                            }
                            return Ok(rep);
                        }
                        Err(error) => last_error = error,
                    }
                }
                Err(last_error)
            })();
            // Every rung refused: the final iterate is still a real point of
            // the model, and 12,000 rows of evaluation do not stop existing
            // because a certificate declined. Score it, labelled.
            let heldout = heldout.or_else(|error| {
                println!("HELDOUT stalled uncertified: {error}");
                Ok::<_, String>(gam_sae::manifold::SaeSupportCoordinateFixedPointReport {
                    iterations: 0,
                    objective: f64::NAN,
                    coordinate_l2: f64::NAN,
                    coordinate_max_abs: f64::NAN,
                    max_recurrence_change: f64::NAN,
                    recurred: false,
                })
            });
            match heldout {
                Ok(rep) => {
                    let recon = te_term.reconstruct()?;
                    // The held-out reconstruction itself, for the spliced
                    // delta-CE criterion: splice_paired.py substitutes exactly
                    // this matrix into the residual stream at the held-out
                    // positions. Chart-centred; the splice adds the mean back.
                    write_f64s(
                        &format!("{out_dir}/heldout_recon.bin"),
                        recon.as_slice().ok_or("recon not contiguous")?,
                    )?;
                    // The held-out routing itself, in the same layout as the
                    // training-side `support.bin` / `coords.bin`: one atom
                    // index and one coordinate block per routed slot, rows in
                    // order. Without these, the support-versus-scalar
                    // decomposition can only be computed on training rows.
                    let mut te_support: Vec<f64> = Vec::with_capacity(te_rows * top_k);
                    let mut te_coords: Vec<f64> = Vec::new();
                    let mut te_coords_len: Vec<f64> = Vec::with_capacity(te_rows);
                    for row in 0..te_rows {
                        for &atom in te_term.assignment.support_indices(row) {
                            te_support.push(atom as f64);
                        }
                        let rc = te_term.assignment.coords_row(row);
                        te_coords_len.push(rc.len() as f64);
                        te_coords.extend_from_slice(rc);
                    }
                    write_f64s(&format!("{out_dir}/heldout_support.bin"), &te_support)?;
                    write_f64s(&format!("{out_dir}/heldout_coords.bin"), &te_coords)?;
                    write_f64s(
                        &format!("{out_dir}/heldout_coords_len.bin"),
                        &te_coords_len,
                    )?;
                    let sse: f64 = centered_test.iter().zip(recon.iter()).map(|(x, r)| (x - r).powi(2)).sum();
                    let ss: f64 = centered_test.iter().map(|x| x * x).sum();
                    println!(
                        "HELDOUT rows={te_rows} recurred={} EV={:.4} chart={te_digest} \
                         build={} certified_at={:.0e} \
                         escalation_cycles={}",
                        rep.recurred,
                        1.0 - sse / ss,
                        build_digest(),
                        certified_tolerance.get(),
                        escalation_cycles.get()
                    );
                    // The held-out reconstruction itself. Chart EV alone cannot
                    // say whether the reconstructed directions are the ones the
                    // model computes with, so the ambient-space score and the
                    // cross-entropy splice both need this array, and neither can
                    // be recomputed from the per-atom dumps.
                    let mut recon_abs = recon.clone();
                    recon_abs += &mean;
                    write_f64s(
                        &format!("{out_dir}/heldout_recon.bin"),
                        recon_abs.as_slice().ok_or("heldout recon not contiguous")?,
                    )?;
                }
                Err(e) => println!("HELDOUT refused: {e}"),
            }
        }
    }

    // usage census + per-atom rows
    let mut usage = vec![0usize; k_ret];
    let mut atom_tokens: Vec<Vec<(usize, f64, f64)>> = vec![Vec::new(); k_ret]; // (row, t, value)
    // Full-width coordinates for d>=2 atoms: a sphere token's position is a
    // unit 3-vector, and `atom_tokens` keeps only axis 0 -- enough for the 1-D
    // rug plots, unplottable for a surface. Written as tokensN_<idx>.bin
    // (row, c_1..c_w) so surface figures can show WHERE the data lives on the
    // fitted manifold, not just what the manifold looks like.
    let mut atom_token_coords: Vec<Vec<(usize, Vec<f64>)>> = vec![Vec::new(); k_ret];
    for row in 0..rows {
        let support = term.assignment.support_indices(row);
        let values = term.assignment.gate_params(row);
        for (slot, (&atom, &value)) in support.iter().zip(values.iter()).enumerate() {
            if value != 0.0 {
                let atom = atom as usize;
                usage[atom] += 1;
                let coords = term.assignment.coords_for_slot(row, slot);
                atom_tokens[atom].push((row, coords[0], value));
                if coords.len() > 1 {
                    atom_token_coords[atom].push((row, coords.to_vec()));
                }
            }
        }
    }

    // pick the most-used atoms per topology kind (up to 4 each)
    let mut order: Vec<usize> = (0..k_ret).collect();
    order.sort_by_key(|&a| std::cmp::Reverse(usage[a]));
    let mut picked: Vec<usize> = Vec::new();
    // Quantile-sampled, not top-N: order each kind's atoms by usage and take 8
    // spread evenly across that order. Picking only the most-used atoms shows the
    // dictionary at its best and hides the tail — and the tail is where the
    // dead-coordinate atoms live. A figure meant to be read critically has to
    // sample the distribution it is describing.
    // "sphere" included: d>=2 only started fitting today, and a portfolio figure
    // that silently omits the surfaces shows only the part that already worked.
    for kind in ["linear", "euclidean", "periodic", "sphere", "torus"] {
        let mut of_kind: Vec<usize> = (0..k_ret)
            .filter(|&a| retained_basis[a] == kind && usage[a] >= 12)
            .collect();
        of_kind.sort_by(|&x, &y| usage[y].cmp(&usage[x]));
        if of_kind.is_empty() {
            continue;
        }
        let want = 8usize.min(of_kind.len());
        for slot in 0..want {
            // Even quantiles over the usage ordering, endpoints included.
            let pos = if want == 1 {
                0
            } else {
                slot * (of_kind.len() - 1) / (want - 1)
            };
            let atom = of_kind[pos];
            if !picked.contains(&atom) {
                picked.push(atom);
            }
        }
    }
    println!("picked atoms: {picked:?}");

    // K-wide census. Overcompleteness is a claim about the whole dictionary, so
    // it needs every atom, not the twelve that get plotted: usage, topology, the
    // arc length of the decoded curve (a collapsed atom has ~zero length and is
    // doing no manifold work), and the mean image so full pairwise coherence can
    // be checked against the Welch bound.
    {
        let kind_code = |kind: &str| -> f64 {
            match kind {
                "linear" => 0.0,
                "euclidean" => 1.0,
                "periodic" => 2.0,
                _ => 3.0,
            }
        };
        let mut census = Vec::with_capacity(k_ret * 4);
        let mut means = Vec::with_capacity(k_ret * cols);
        let probe: Vec<f64> = (0..33).map(|j| -1.0 + 2.0 * j as f64 / 32.0).collect();
        let periodic_probe: Vec<f64> = (0..33).map(|j| j as f64 / 32.0).collect();
        for atom in 0..k_ret {
            let is_periodic = retained_basis[atom] == "periodic";
            let grid = if is_periodic { &periodic_probe } else { &probe };
            // Decode at the atom's OWN latent dimension. A 2-D atom needs a
            // coordinate PAIR; passing a single column is rejected outright
            // ("coords width 1 != atom latent dim 2") and killed the whole dump.
            let dim = atom_dim_for_basis(&retained_basis[atom]);
            let coords = if retained_basis[atom] == "sphere" {
                // The ambient sphere decodes at width 3; its probe lattice
                // lives ON the sphere, not on a coordinate square.
                let side = 9usize;
                Array2::from_shape_vec((side * side, 3), sphere_fibonacci(side * side))
                    .map_err(|e| e.to_string())?
            } else if dim == 2 {
                let side = 9usize;
                let mut pairs = Vec::with_capacity(side * side * 2);
                for i in 0..side {
                    for j in 0..side {
                        pairs.push(grid[i * (grid.len() - 1) / (side - 1)]);
                        pairs.push(grid[j * (grid.len() - 1) / (side - 1)]);
                    }
                }
                Array2::from_shape_vec((side * side, 2), pairs).map_err(|e| e.to_string())?
            } else {
                Array2::from_shape_vec((grid.len(), 1), grid.clone())
                    .map_err(|e| e.to_string())?
            };
            let curve = term.decode_atom_at(atom, coords.view())?;
            let mut arc = 0.0_f64;
            for j in 1..curve.nrows() {
                let mut d = 0.0;
                for c in 0..cols {
                    let step = curve[[j, c]] - curve[[j - 1, c]];
                    d += step * step;
                }
                arc += d.sqrt();
            }
            for c in 0..cols {
                let mut acc = 0.0;
                for j in 0..curve.nrows() {
                    acc += curve[[j, c]];
                }
                means.push(acc / curve.nrows() as f64);
            }
            census.push(usage[atom] as f64);
            census.push(kind_code(&retained_basis[atom]));
            census.push(arc);
            census.push(term.atom_basis_size(atom) as f64);
        }
        write_f64s(&format!("{out_dir}/census.bin"), &census)?;
        write_f64s(&format!("{out_dir}/atom_means.bin"), &means)?;
        println!("CENSUS dumped for {k_ret} atoms");
    }

    // dump: per picked atom — kind, usage, curve samples over the coordinate
    // range actually used (Rust-decoded), token coords + chart rows projected later
    let mut manifest = String::from("[");
    for (idx, &a) in picked.iter().enumerate() {
        let toks = &atom_tokens[a];
        let kind = &retained_basis[a];
        let (lo, hi) = if kind == "periodic" || kind == "torus" {
            (0.0, 1.0)
        } else {
            let mut lo = f64::INFINITY;
            let mut hi = f64::NEG_INFINITY;
            for &(_, t, _) in toks {
                lo = lo.min(t);
                hi = hi.max(t);
            }
            let pad = 0.06 * (hi - lo).max(1e-9);
            (lo - pad, hi + pad)
        };
        // Divisor MUST match the last index, or the grid overshoots the range the
        // data actually covers: with 257 samples and /160.0 the last sample sat at
        // lo + 1.6*(hi - lo), i.e. 60% beyond `hi`, and every curve looked like it
        // extrapolated wildly. That was the dumper, not the model.
        let samples = 257usize;
        let grid: Vec<f64> = (0..samples)
            .map(|j| lo + (hi - lo) * j as f64 / (samples - 1) as f64)
            .collect();
        let dim = atom_dim_for_basis(kind);
        let grid_n = if dim == 2 { 33usize } else { samples };
        let coords = if kind == "sphere" {
            Array2::from_shape_vec((grid_n * grid_n, 3), sphere_grid(grid_n))
                .map_err(|e| e.to_string())?
        } else if dim == 2 {
            // Square grid over both axes -> a SURFACE, reshaped downstream via
            // `grid_n`. Both axes share the observed range because `atom_tokens`
            // carries one coordinate per token, not a pair.
            let mut pairs = Vec::with_capacity(grid_n * grid_n * 2);
            for i in 0..grid_n {
                for j in 0..grid_n {
                    pairs.push(lo + (hi - lo) * i as f64 / (grid_n - 1) as f64);
                    pairs.push(lo + (hi - lo) * j as f64 / (grid_n - 1) as f64);
                }
            }
            Array2::from_shape_vec((grid_n * grid_n, 2), pairs).map_err(|e| e.to_string())?
        } else {
            Array2::from_shape_vec((samples, 1), grid.clone()).map_err(|e| e.to_string())?
        };
        let curve = term.decode_atom_at(a, coords.view())?;
        write_f64s(
            &format!("{out_dir}/curve_{idx}.bin"),
            curve.as_slice().ok_or("curve not contiguous")?,
        )?;
        let mut tok_flat: Vec<f64> = Vec::with_capacity(toks.len() * 3);
        for &(row, t, value) in toks {
            tok_flat.push(row as f64);
            tok_flat.push(t);
            tok_flat.push(value);
        }
        write_f64s(&format!("{out_dir}/tokens_{idx}.bin"), &tok_flat)?;
        {
            let cap = 2000usize.min(toks.len());
            let mut partial: Vec<f64> = Vec::with_capacity(cap * (1 + cols));
            for &(row, _t, _v) in toks.iter().take(cap) {
                let support = term.assignment.support_indices(row);
                let mut own = vec![0.0_f64; cols];
                for (slot, &atom_id) in support.iter().enumerate() {
                    if atom_id as usize == a {
                        let coords_slice = term.assignment.coords_for_slot(row, slot);
                        let coords = Array2::from_shape_vec(
                            (1, coords_slice.len()),
                            coords_slice.to_vec(),
                        )
                        .map_err(|e| e.to_string())?;
                        let decoded = term.decode_atom_at(a, coords.view())?;
                        own.copy_from_slice(decoded.row(0).to_slice().ok_or("contig")?);
                        break;
                    }
                }
                partial.push(row as f64);
                for c in 0..cols {
                    partial.push(centered[[row, c]] - fitted[[row, c]] + own[c]);
                }
            }
            write_f64s(&format!("{out_dir}/partial_{idx}.bin"), &partial)?;
        }
        if !atom_token_coords[a].is_empty() {
            let width = atom_token_coords[a][0].1.len();
            let mut wide: Vec<f64> = Vec::with_capacity(atom_token_coords[a].len() * (width + 1));
            for (row, coords) in &atom_token_coords[a] {
                wide.push(*row as f64);
                wide.extend_from_slice(coords);
            }
            write_f64s(&format!("{out_dir}/tokens{width}_{idx}.bin"), &wide)?;
        }
        manifest.push_str(&format!(
            "{}{{\"idx\":{idx},\"atom\":{a},\"kind\":\"{kind}\",\"dim\":{dim},\"grid_n\":{grid_n},\"usage\":{},\"n_tokens\":{},\"grid_lo\":{lo},\"grid_hi\":{hi}}}",
            if idx == 0 { "" } else { "," },
            usage[a],
            toks.len()
        ));
    }
    manifest.push(']');
    let mut file = std::fs::File::create(format!("{out_dir}/manifest.json"))
        .map_err(|e| e.to_string())?;
    file.write_all(manifest.as_bytes()).map_err(|e| e.to_string())?;
    write_f64s(&format!("{out_dir}/mean.bin"), mean.as_slice().ok_or("mean")?)?;
    println!("DUMPED to {out_dir}");
    Ok(())
}
