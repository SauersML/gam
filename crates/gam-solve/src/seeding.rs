use ndarray::Array1;
use std::collections::HashSet;

pub use gam_problem::{SeedConfig, SeedRiskProfile};
use gam_problem::{clamp_seed_rho_to_bounds, normalize_seed_bounds};

fn add_seed_dedup(seeds: &mut Vec<Array1<f64>>, seen: &mut HashSet<Vec<u64>>, seed: Array1<f64>) {
    let key: Vec<u64> = seed.iter().map(|&v| v.to_bits()).collect();
    if seen.insert(key) {
        seeds.push(seed);
    }
}

fn safe_ln_pos(x: f64) -> Option<f64> {
    if x.is_finite() && x > 0.0 {
        Some(x.ln())
    } else {
        None
    }
}

fn spde_rho_triplet_from_log_tau_log_kappa_nu(
    log_tau: f64,
    log_kappa: f64,
    nu: f64,
    bounds: (f64, f64),
) -> Option<Array1<f64>> {
    if !(nu.is_finite() && nu > 1.0) {
        return None;
    }
    let logc0 = 0.0;
    let logc1 = safe_ln_pos(nu)?;
    let logc2 = safe_ln_pos(0.5 * nu * (nu - 1.0))?;
    let rho0 = clamp_seed_rho_to_bounds(log_tau + logc0 + 2.0 * nu * log_kappa, bounds);
    let rho1 = clamp_seed_rho_to_bounds(log_tau + logc1 + 2.0 * (nu - 1.0) * log_kappa, bounds);
    let rho2 = clamp_seed_rho_to_bounds(log_tau + logc2 + 2.0 * (nu - 2.0) * log_kappa, bounds);
    Some(Array1::from_vec(vec![rho0, rho1, rho2]))
}

fn add_spde_manifold_seeds(
    seeds: &mut Vec<Array1<f64>>,
    seen: &mut HashSet<Vec<u64>>,
    bounds: (f64, f64),
    heuristic_rhos: Option<&[f64]>,
    primary: &Array1<f64>,
) {
    if primary.len() != 3 {
        return;
    }
    // Broad default manifold grid in (log_tau, log_kappa, nu).
    let tau_anchors = [primary[2], 0.0, -2.0, 2.0];
    let log_kappa_grid = [-2.0, -1.0, 0.0, 1.0, 2.0];
    let nu_grid = [1.25, 1.5, 2.0, 2.5, 3.0, 4.0];
    for &tau in &tau_anchors {
        for &lk in &log_kappa_grid {
            for &nu in &nu_grid {
                if let Some(seed) = spde_rho_triplet_from_log_tau_log_kappa_nu(tau, lk, nu, bounds)
                {
                    add_seed_dedup(seeds, seen, seed);
                }
            }
        }
    }

    // Data-informed anchor: convert the rho seed to lambdas, then invert to
    // (nu, kappa^2, tau) when feasible.
    if let Some(vals) = heuristic_rhos
        && vals.len() == 3
    {
        let l0 = vals[0].exp();
        let l1 = vals[1].exp();
        let l2 = vals[2].exp();
        if l0.is_finite() && l1.is_finite() && l2.is_finite() && l0 > 1e-12 && l2 > 1e-12 {
            let r = (l1 * l1) / (l0 * l2);
            if r > 2.0 {
                let nu = r / (r - 2.0);
                let kappa2 = l1 / ((r - 2.0) * l2);
                if nu.is_finite() && nu > 1.0 && kappa2.is_finite() && kappa2 > 0.0 {
                    let log_kappa = 0.5 * kappa2.ln();
                    let c2 = 0.5 * nu * (nu - 1.0);
                    if c2.is_finite() && c2 > 0.0 {
                        let log_tau = (l2 / (c2 * kappa2.powf(nu - 2.0))).max(1e-12).ln();
                        let local_nu = [nu, (nu - 0.3).max(1.05), nu + 0.3];
                        let local_tau = [log_tau, log_tau - 1.0, log_tau + 1.0];
                        let local_kappa = [log_kappa, log_kappa - 0.5, log_kappa + 0.5];
                        for &t in &local_tau {
                            for &lk in &local_kappa {
                                for &n in &local_nu {
                                    if let Some(seed) =
                                        spde_rho_triplet_from_log_tau_log_kappa_nu(t, lk, n, bounds)
                                    {
                                        add_seed_dedup(seeds, seen, seed);
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

fn add_first_order_fallback_seeds(
    seeds: &mut Vec<Array1<f64>>,
    seen: &mut HashSet<Vec<u64>>,
    bounds: (f64, f64),
    heuristic_rhos: Option<&[f64]>,
) {
    // Degenerate λ2 -> 0 fallback (first-order mass+tension):
    // λ0 = τ κ^2, λ1 = τ, λ2 ≈ 0.
    let rho2_floor = bounds.0;
    let default_log_kappa = [-2.0, -1.0, 0.0, 1.0];
    let default_log_tau = [0.0, -2.0, 2.0];
    for &t in &default_log_tau {
        for &lk in &default_log_kappa {
            let rho0 = clamp_seed_rho_to_bounds(t + 2.0 * lk, bounds);
            let rho1 = clamp_seed_rho_to_bounds(t, bounds);
            add_seed_dedup(seeds, seen, Array1::from_vec(vec![rho0, rho1, rho2_floor]));
        }
    }
    if let Some(vals) = heuristic_rhos
        && vals.len() == 3
        && vals[0].is_finite()
        && vals[1].is_finite()
    {
        let l0 = vals[0].exp();
        let l1 = vals[1].exp();
        let kappa2 = l0 / l1;
        if kappa2.is_finite() && kappa2 > 0.0 {
            let lk = 0.5 * kappa2.ln();
            let t = vals[1];
            let rho0 = clamp_seed_rho_to_bounds(t + 2.0 * lk, bounds);
            let rho1 = clamp_seed_rho_to_bounds(t, bounds);
            add_seed_dedup(seeds, seen, Array1::from_vec(vec![rho0, rho1, rho2_floor]));
        }
    }
}

fn add_nu2_reverse_manifold_seeds(
    seeds: &mut Vec<Array1<f64>>,
    seen: &mut HashSet<Vec<u64>>,
    bounds: (f64, f64),
    primary: &Array1<f64>,
) {
    if primary.len() != 3 {
        return;
    }
    let ln_two = 2.0_f64.ln();
    let tau_anchors = [primary[2], 0.0, -2.0, 2.0];
    let log_kappa_grid = [-2.0, -1.0, 0.0, 1.0, 2.0];
    for &tau_rho in &tau_anchors {
        for &log_kappa in &log_kappa_grid {
            // Continuous-order reverse map at nu=2:
            // lambda0 = tau * kappa^4, lambda1 = tau * 2*kappa^2, lambda2 = tau.
            let rho2 = clamp_seed_rho_to_bounds(tau_rho, bounds);
            let rho1 = clamp_seed_rho_to_bounds(tau_rho + ln_two + 2.0 * log_kappa, bounds);
            let rho0 = clamp_seed_rho_to_bounds(tau_rho + 4.0 * log_kappa, bounds);
            add_seed_dedup(seeds, seen, Array1::from_vec(vec![rho0, rho1, rho2]));
        }
    }
}

fn halton(mut index: usize, base: usize) -> f64 {
    let mut f = 1.0_f64;
    let mut r = 0.0_f64;
    while index > 0 {
        f /= base as f64;
        r += f * (index % base) as f64;
        index /= base;
    }
    r
}

fn first_primes(n: usize) -> Vec<usize> {
    let mut primes = Vec::with_capacity(n);
    let mut x = 2usize;
    while primes.len() < n {
        let mut is_prime = true;
        let mut d = 2usize;
        while d * d <= x {
            if x.is_multiple_of(d) {
                is_prime = false;
                break;
            }
            d += 1;
        }
        if is_prime {
            primes.push(x);
        }
        x += 1;
    }
    primes
}

pub fn generate_rho_candidates(
    num_penalties: usize,
    heuristic_rhos: Option<&[f64]>,
    config: &SeedConfig,
) -> Vec<Array1<f64>> {
    let mut seeds = Vec::new();
    let mut seen: HashSet<Vec<u64>> = HashSet::new();

    let bounds = normalize_seed_bounds(config.bounds);
    let max_seeds = config.max_seeds.max(1);
    let risk_shift = config.risk_profile.anchor_rho_shift();

    if num_penalties == 0 {
        add_seed_dedup(&mut seeds, &mut seen, Array1::<f64>::zeros(0));
        return seeds;
    }

    // Prefer a full heuristic vector (length == k) as the primary anchor.
    // Values are already in the outer optimizer's rho/theta parameter space.
    let num_aux = config.num_auxiliary_trailing.min(num_penalties);
    let num_smoothing = num_penalties - num_aux;
    let aux_initial: Vec<f64> = if num_aux > 0 {
        heuristic_rhos
            .filter(|h| h.len() == num_penalties)
            .map(|h| {
                h[num_smoothing..]
                    .iter()
                    .copied()
                    .map(|v| clamp_seed_rho_to_bounds(v, bounds))
                    .collect()
            })
            .unwrap_or_else(|| vec![0.0; num_aux])
    } else {
        Vec::new()
    };
    let heuristic_rhovec: Option<Array1<f64>> = heuristic_rhos.and_then(|vals| {
        if vals.len() == num_penalties {
            Some(Array1::from_iter(
                vals[..num_smoothing]
                    .iter()
                    .copied()
                    .map(|v| clamp_seed_rho_to_bounds(v, bounds))
                    .chain(
                        vals[num_smoothing..]
                            .iter()
                            .copied()
                            .map(|v| clamp_seed_rho_to_bounds(v, bounds)),
                    ),
            ))
        } else {
            None
        }
    });

    let primary = heuristic_rhovec.clone().unwrap_or_else(|| {
        Array1::<f64>::from_elem(num_penalties, clamp_seed_rho_to_bounds(risk_shift, bounds))
    });
    add_seed_dedup(&mut seeds, &mut seen, primary.clone());
    // Always include neutral baseline independently of heuristic anchor.
    add_seed_dedup(&mut seeds, &mut seen, Array1::zeros(num_penalties));
    // Generalized and survival models can hit PIRLS separation at moderate
    // smoothing levels. Put an aggressively over-smoothed isotropic seed near
    // the front so startup validation can still find a stable basin.
    match config.risk_profile {
        SeedRiskProfile::Gaussian | SeedRiskProfile::GaussianLocationScale => {}
        SeedRiskProfile::GeneralizedLinear | SeedRiskProfile::Survival => {
            add_seed_dedup(
                &mut seeds,
                &mut seen,
                Array1::from_elem(num_penalties, bounds.1),
            );
        }
    }
    // For exactly three smoothing penalties (mass/tension/stiffness), inject
    // physically coherent manifold seeds in rho-space:
    // - general SPDE manifold over (log_tau, log_kappa, nu),
    // - nu=2 reverse-map seeds,
    // - first-order fallback seeds (lambda2 near lower bound).
    if num_smoothing == 3 {
        let smoothing_primary =
            Array1::from_vec(primary.iter().take(num_smoothing).copied().collect());
        let smoothing_heuristic_lambdas = heuristic_rhos.and_then(|vals| {
            if vals.len() >= num_smoothing {
                Some(&vals[..num_smoothing])
            } else {
                None
            }
        });
        let mut spde_prefix_seeds = Vec::new();
        let mut spde_prefix_seen: HashSet<Vec<u64>> = HashSet::new();
        // Guarantee a first-order fallback anchor regardless of later truncation.
        add_seed_dedup(
            &mut spde_prefix_seeds,
            &mut spde_prefix_seen,
            Array1::from_vec(vec![primary[0], primary[1], bounds.0]),
        );
        // Ensure a nu=2-consistent seed is always present before broader grids.
        add_nu2_reverse_manifold_seeds(
            &mut spde_prefix_seeds,
            &mut spde_prefix_seen,
            bounds,
            &smoothing_primary,
        );
        add_first_order_fallback_seeds(
            &mut spde_prefix_seeds,
            &mut spde_prefix_seen,
            bounds,
            smoothing_heuristic_lambdas,
        );
        add_spde_manifold_seeds(
            &mut spde_prefix_seeds,
            &mut spde_prefix_seen,
            bounds,
            smoothing_heuristic_lambdas,
            &smoothing_primary,
        );
        for prefix_seed in spde_prefix_seeds {
            let mut seed = Array1::<f64>::zeros(num_penalties);
            for i in 0..num_smoothing {
                seed[i] = prefix_seed[i];
            }
            for (i, &v) in aux_initial.iter().enumerate() {
                seed[num_smoothing + i] = v;
            }
            add_seed_dedup(&mut seeds, &mut seen, seed);
        }
    }

    // Broad symmetric baselines around the center to guarantee global coverage.
    for &center in config.risk_profile.baseline_centers() {
        add_seed_dedup(
            &mut seeds,
            &mut seen,
            Array1::from_elem(num_penalties, clamp_seed_rho_to_bounds(center, bounds)),
        );
    }

    let dims_to_touch = num_penalties.min(12);
    let step_base = if num_penalties <= 4 {
        2.0
    } else if num_penalties <= 12 {
        2.5
    } else {
        3.0
    };
    let high_dim_cluster_threshold = 10usize;

    if num_penalties >= high_dim_cluster_threshold {
        // High-dimensional path: probe relative scaling conflicts by clustering
        // penalties into low/high heuristic-magnitude groups.
        let mut sorted_idx: Vec<usize> = (0..num_penalties).collect();
        sorted_idx.sort_by(|&i, &j| primary[i].total_cmp(&primary[j]));

        let cluster_size = (num_penalties / 3).max(1);
        let small_end = cluster_size.min(num_penalties);
        let large_start = num_penalties.saturating_sub(cluster_size);
        let small_cluster = &sorted_idx[..small_end];
        let large_cluster = &sorted_idx[large_start..];

        let small_scale = step_base;
        let large_scale = step_base + 0.75;

        let mut conflict_a = primary.clone();
        for &i in large_cluster {
            conflict_a[i] = clamp_seed_rho_to_bounds(primary[i] + large_scale, bounds);
        }
        for &i in small_cluster {
            conflict_a[i] = clamp_seed_rho_to_bounds(primary[i] - small_scale, bounds);
        }
        add_seed_dedup(&mut seeds, &mut seen, conflict_a);

        let mut conflict_b = primary.clone();
        for &i in large_cluster {
            conflict_b[i] = clamp_seed_rho_to_bounds(primary[i] - large_scale, bounds);
        }
        for &i in small_cluster {
            conflict_b[i] = clamp_seed_rho_to_bounds(primary[i] + small_scale, bounds);
        }
        add_seed_dedup(&mut seeds, &mut seen, conflict_b);

        let mut heavy_up = primary.clone();
        for &i in large_cluster {
            heavy_up[i] = clamp_seed_rho_to_bounds(primary[i] + large_scale, bounds);
        }
        add_seed_dedup(&mut seeds, &mut seen, heavy_up);

        let mut light_down = primary.clone();
        for &i in small_cluster {
            light_down[i] = clamp_seed_rho_to_bounds(primary[i] - small_scale, bounds);
        }
        add_seed_dedup(&mut seeds, &mut seen, light_down);
    } else {
        // Low-dimensional path: coordinate and sparse pair probes are still cheap.
        for i in 0..dims_to_touch {
            let scale = step_base + 0.25 * primary[i].abs().min(8.0);
            for dir in [-1.0, 1.0] {
                let mut s = primary.clone();
                s[i] = clamp_seed_rho_to_bounds(primary[i] + dir * scale, bounds);
                add_seed_dedup(&mut seeds, &mut seen, s);
            }
        }

        let pair_dims = num_penalties.min(6);
        for i in 0..pair_dims {
            for j in (i + 1)..pair_dims {
                let mut s1 = primary.clone();
                s1[i] = clamp_seed_rho_to_bounds(primary[i] + step_base, bounds);
                s1[j] = clamp_seed_rho_to_bounds(primary[j] - step_base, bounds);
                add_seed_dedup(&mut seeds, &mut seen, s1);

                let mut s2 = primary.clone();
                s2[i] = clamp_seed_rho_to_bounds(primary[i] - step_base, bounds);
                s2[j] = clamp_seed_rho_to_bounds(primary[j] + step_base, bounds);
                add_seed_dedup(&mut seeds, &mut seen, s2);
            }
        }
    }

    // Global shrink/expand sweeps from the anchor to probe over/under-smoothing regimes.
    // The flexible (negative-shift) side MUST be probed as densely as the
    // over-smoothing side: the seed-screening proxy is a capped-inner-iteration
    // fit, and an over-smoothed seed converges trivially under that cap (its
    // coefficients collapse into the penalty null space, the LAML is locally
    // flat), so screening systematically ranks over-smoothed seeds first
    // (documented in `rank_seeds_with_screening`). For a GeneralizedLinear /
    // Survival model whose true optimum is flexible (e.g. a smooth Poisson
    // tensor surface that genuinely needs ~10 effective df), a seed grid that
    // only sweeps the over-smoothing side leaves the flexible basin unprobed,
    // so none of the few full-budget solves ever lands in it and the fit
    // over-smooths (#1082/#1373). Symmetric negative shifts give the flexible
    // basin a candidate; the keep-best multi-start then retains it only if it
    // actually scores better, so this can never worsen a fit — it only lets the
    // optimizer SEE the lower-λ basin. Over-smoothed seeds remain present (and
    // earlier in the list) so PIRLS-separation startup stability is unchanged.
    for &shift in config.risk_profile.global_shifts() {
        let swept = primary.mapv(|v| clamp_seed_rho_to_bounds(v + shift, bounds));
        add_seed_dedup(&mut seeds, &mut seen, swept);
    }

    // #1464 over-smoothing probe: an ABSOLUTE high-λ start on every smoothing
    // dimension (auxiliary dims left at the anchor's values; they are re-pinned
    // below). The global shift sweeps above reach only ≈ +4 from the anchor, so
    // a collapsing-kernel smooth whose true REML optimum is a large λ would never
    // be seeded into its over-smoothing basin. This puts a candidate IN it; the
    // keep-best multistart adopts it only when it scores strictly better, so it
    // can never worsen a fit. `None` (the default) skips this entirely.
    if let Some(probe_rho) = config.over_smoothing_probe_rho {
        let mut probe = primary.clone();
        for j in 0..num_smoothing {
            probe[j] = clamp_seed_rho_to_bounds(probe_rho, bounds);
        }
        add_seed_dedup(&mut seeds, &mut seen, probe);
    }

    // Low-discrepancy exploratory seeds around the anchor for basin discovery.
    // These are still deterministic and do not encode any solver-side bias.
    let exploratory = max_seeds.saturating_sub(seeds.len()).min(8);
    if exploratory > 0 {
        let primes = first_primes(num_penalties.max(1));
        let amp = config.risk_profile.exploratory_amplitude();
        for t in 0..exploratory {
            let mut s = primary.clone();
            for i in 0..num_penalties {
                let u = halton(t + 1, primes[i]); // (0,1)
                let centered = 2.0 * u - 1.0; // (-1,1)
                s[i] = clamp_seed_rho_to_bounds(primary[i] + amp * centered, bounds);
            }
            add_seed_dedup(&mut seeds, &mut seen, s);
        }
    }

    // Pin auxiliary trailing dimensions to their initial values in every seed.
    // Auxiliary params (e.g. SAS epsilon, log_delta) live in a different
    // parameter space than log-smoothing rho and must not be swept by the
    // smoothing seeding grid.  After pinning we re-dedup because seeds that
    // differed only in the (now-overwritten) auxiliary dimensions collapse.
    if num_aux > 0 {
        for seed in &mut seeds {
            for (i, &v) in aux_initial.iter().enumerate() {
                seed[num_smoothing + i] = v;
            }
        }
        let mut deduped = Vec::new();
        let mut seen2: HashSet<Vec<u64>> = HashSet::new();
        for seed in seeds {
            let key: Vec<u64> = seed.iter().map(|&v| v.to_bits()).collect();
            if seen2.insert(key) {
                deduped.push(seed);
            }
        }
        seeds = deduped;
    }

    if seeds.len() > max_seeds {
        seeds.truncate(max_seeds);
    }

    if seeds.is_empty() {
        seeds.push(Array1::<f64>::zeros(num_penalties));
    }

    seeds
}

/// Choose an initial log-smoothing vector by evaluating the same objective the
/// outer optimizer will minimize on a small deterministic grid around the
/// analytic/heuristic seed.
///
/// This is initialization, not a fallback: no candidate is accepted unless it
/// has a lower finite objective value under `eval_cost`, and the returned seed
/// is still optimized by the normal outer solver.
pub fn select_objective_seed_on_log_lambda_grid<F>(
    rho_seed: &Array1<f64>,
    bounds: (f64, f64),
    n_smooths: usize,
    nullspace_coords: &[usize],
    mut eval_cost: F,
) -> Array1<f64>
where
    F: FnMut(&Array1<f64>) -> Option<f64>,
{
    let k = rho_seed.len();
    if k == 0 || n_smooths == 0 || n_smooths > k {
        return rho_seed.clone();
    }
    let bnds = normalize_seed_bounds(bounds);
    let clamp_vec = |v: &Array1<f64>| -> Array1<f64> {
        let mut out = v.clone();
        for i in 0..n_smooths {
            out[i] = clamp_seed_rho_to_bounds(out[i], bnds);
        }
        out
    };

    let baseline_seed = clamp_vec(rho_seed);
    let baseline_cost = eval_cost(&baseline_seed);
    log::info!(
        "[SEED-GRID] baseline rho=[{}] cost={}",
        baseline_seed
            .iter()
            .map(|v| format!("{v:.2}"))
            .collect::<Vec<_>>()
            .join(","),
        baseline_cost
            .map(|c| format!("{c:.6e}"))
            .unwrap_or_else(|| "non-finite".to_string()),
    );

    let shifts: [f64; 9] = [-12.0, -9.0, -6.0, -3.0, 0.0, 3.0, 6.0, 9.0, 12.0];
    let mut best_seed = baseline_seed.clone();
    let mut best_cost: Option<f64> = baseline_cost.filter(|c| c.is_finite());

    for &delta in &shifts {
        if delta == 0.0 && best_cost.is_some() {
            continue;
        }
        let mut candidate = rho_seed.clone();
        for i in 0..n_smooths {
            candidate[i] = clamp_seed_rho_to_bounds(rho_seed[i] + delta, bnds);
        }
        let c_opt = eval_cost(&candidate);
        log::info!(
            "[SEED-GRID] shift={:+.1} rho=[{}] cost={}",
            delta,
            candidate
                .iter()
                .map(|v| format!("{v:.2}"))
                .collect::<Vec<_>>()
                .join(","),
            c_opt
                .map(|c| format!("{c:.6e}"))
                .unwrap_or_else(|| "non-finite".to_string()),
        );
        if let Some(c) = c_opt
            && c.is_finite()
            && best_cost.map(|b| c < b).unwrap_or(true)
        {
            best_cost = Some(c);
            best_seed = candidate;
        }
    }

    if n_smooths <= 6 {
        // Per-axis refinement around the best isotropic point. The ±3 steps
        // resolve a mild per-coordinate imbalance; the explicit saturation
        // target (`bnds.1`, the over-smoothing upper bound) reaches asymmetric
        // corners where selected penalty blocks are fully active while the
        // others stay at the refined anchor. Those corners are load-bearing for
        // double-penalty (Marra-Wood null-space shrinkage) smooths (#1266): an
        // unsupported term must be allowed to send BOTH its wiggliness and
        // null-space coordinates high, while a supported sibling term remains
        // free. The isotropic grid only moves all coordinates together, so it
        // cannot express "shrink s(z), keep s(x)". Probing these corners is
        // criterion-ranked — a candidate is adopted only when it strictly lowers
        // the true REML/LAML cost — so a genuinely better interior optimum or
        // supported smooth simply wins the comparison.
        let saturation = clamp_seed_rho_to_bounds(bnds.1, bnds);
        // Lower-saturation ("keep") corner, the symmetric dual of `saturation`.
        // The per-axis sweep above probes the over-smoothing/shrink-out corner
        // (`bnds.1`) so an unsupported double-penalty null-space coordinate can
        // rail its λ_null up and select the term out (#1266). The MISSING corner
        // is the opposite one: a SUPPORTED null space (a genuine linear/constant
        // trend the data buy) has its global REML optimum at a LOW λ_null "keep"
        // basin, separated from the high-λ_null annihilation shelf by a flat
        // valley. Without a keep-direction probe the grid can seed only the shelf
        // corner, leaving the outer optimizer to cross that flat valley to reach
        // the keep basin — a crossing whose success rode on sub-ULP gradient signs
        // that a covariate reflection x→−x flips, so the mirror fit stalled on the
        // shelf and annihilated the supported trend (#1548). Probing the keep
        // corner for EXACTLY the null-space coordinates (where un-shrinking is
        // safe — the wiggliness penalty stays active, so there is no λ→0
        // inner-cap overfit artifact) lets the grid seed the well-conditioned keep
        // basin directly. It is criterion-ranked like every other probe: an
        // unsupported term's keep corner is never cheaper than its shrink-out
        // corner, so #1266 is untouched.
        let keep_saturation = clamp_seed_rho_to_bounds(bnds.0, bnds);
        for axis in 0..n_smooths {
            let anchor = best_seed.clone();
            let mut targets = vec![
                clamp_seed_rho_to_bounds(anchor[axis] - 3.0, bnds),
                clamp_seed_rho_to_bounds(anchor[axis] + 3.0, bnds),
            ];
            if (anchor[axis] - saturation).abs() > 1e-9 {
                targets.push(saturation);
            }
            if nullspace_coords.contains(&axis) {
                // Step toward the keep basin (a moderate un-shrink) and the full
                // keep saturation, so the probe reaches the basin wherever it sits
                // between the anchor and λ_null → 0.
                targets.push(clamp_seed_rho_to_bounds(anchor[axis] - 6.0, bnds));
                if (anchor[axis] - keep_saturation).abs() > 1e-9 {
                    targets.push(keep_saturation);
                }
            }
            for target in targets {
                let mut candidate = anchor.clone();
                candidate[axis] = target;
                if let Some(c) = eval_cost(&candidate)
                    && c.is_finite()
                    && best_cost.map(|b| c < b).unwrap_or(true)
                {
                    best_cost = Some(c);
                    best_seed = candidate;
                }
            }
        }
        // Adjacent-pair over-smoothing corner: send one term's (mass, tension)
        // / null-space pair fully to the bound while the SIBLING terms stay
        // free. Probe this corner from BOTH the refined isotropic best AND the
        // baseline anchor. Anchoring only on `best_seed` cannot express "shrink
        // s(z), keep s(x) at its supported λ": the per-axis sweep above drives
        // every coordinate toward the dominant over-smoothing optimum, so the
        // kept siblings are already saturated and the genuine "keep the rest at
        // baseline" corner is unreachable (the unsupported pair gets railed but
        // the supported pair can never relax back below the ±3 refinement
        // reach). Including the baseline anchor lets the grid seed exactly the
        // asymmetric corner #1266 is about — one pair at the bound, the rest at
        // the supported baseline λ. Both anchors are criterion-ranked (adopted
        // only on a strict cost decrease), so this never displaces a better
        // interior optimum and leaves balanced fits byte-identical.
        for anchor in [best_seed.clone(), baseline_seed.clone()] {
            for start in 0..n_smooths.saturating_sub(1) {
                let mut candidate = anchor.clone();
                candidate[start] = saturation;
                candidate[start + 1] = saturation;
                if let Some(c) = eval_cost(&candidate)
                    && c.is_finite()
                    && best_cost.map(|b| c < b).unwrap_or(true)
                {
                    best_cost = Some(c);
                    best_seed = candidate;
                }
            }
        }
    }

    best_seed
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn uses_full_heuristicvector_as_primary_anchor() {
        let cfg = SeedConfig {
            risk_profile: SeedRiskProfile::Gaussian,
            ..SeedConfig::default()
        };
        let heur = [-2.0, 0.0, 2.0];
        let seeds = generate_rho_candidates(3, Some(&heur), &cfg);
        assert!(!seeds.is_empty());
        let first = &seeds[0];
        assert_eq!(first.len(), 3);
        assert!((first[0] - heur[0]).abs() < 1e-12);
        assert!((first[1] - heur[1]).abs() < 1e-12);
        assert!((first[2] - heur[2]).abs() < 1e-12);
    }

    #[test]
    fn high_dim_uses_cluster_conflict_probeswithout_exploding() {
        let cfg = SeedConfig {
            max_seeds: 18,
            risk_profile: SeedRiskProfile::GeneralizedLinear,
            ..SeedConfig::default()
        };
        let heur = [-6.0, -5.0, -4.0, 0.0, 2.0, 4.0, -3.0, 0.0, 3.0, 5.0];
        let seeds = generate_rho_candidates(10, Some(&heur), &cfg);
        assert!(seeds.len() <= 18);
        // Presence of at least one asymmetric cluster-conflict seed:
        // some coordinates increased while others decreased vs primary.
        let primary = &seeds[0];
        let has_conflict = seeds.iter().skip(1).any(|s| {
            let mut any_up = false;
            let mut any_down = false;
            for i in 0..s.len() {
                if s[i] > primary[i] {
                    any_up = true;
                } else if s[i] < primary[i] {
                    any_down = true;
                }
            }
            any_up && any_down
        });
        assert!(has_conflict);
    }

    #[test]
    fn includes_neutralzero_seed() {
        let cfg = SeedConfig::default();
        let seeds = generate_rho_candidates(5, None, &cfg);
        let haszero = seeds
            .iter()
            .any(|s| s.iter().all(|v| (*v - 0.0).abs() < 1e-12));
        assert!(haszero);
    }

    #[test]
    fn generalized_linear_seeds_include_early_stability_retreat_seed() {
        let cfg = SeedConfig {
            risk_profile: SeedRiskProfile::GeneralizedLinear,
            ..SeedConfig::default()
        };
        let seeds = generate_rho_candidates(3, None, &cfg);
        let retreat = Array1::from_elem(3, cfg.bounds.1);
        let retreat_idx = seeds
            .iter()
            .position(|seed| seed == retreat)
            .expect("generalized-linear seeds should include an upper-bound retreat seed");
        assert!(
            retreat_idx <= 2,
            "retreat seed should be available before broader exploratory seeds: {retreat_idx}"
        );
    }

    #[test]
    fn objective_grid_can_seed_adjacent_pair_oversmoothing_corner() {
        let base = Array1::zeros(4);
        let selected =
            select_objective_seed_on_log_lambda_grid(&base, (-12.0, 12.0), 4, &[], |rho| {
                let supported_cost = 0.1 * (rho[0].powi(2) + rho[1].powi(2));
                let unsupported_gap = (rho[2] - 12.0).powi(2) + (rho[3] - 12.0).powi(2);
                Some(supported_cost + unsupported_gap)
            });
        assert_eq!(selected.to_vec(), vec![0.0, 0.0, 12.0, 12.0]);
    }

    #[test]
    fn three_penalty_seeds_include_nu2_reverse_manifold_triplets() {
        let cfg = SeedConfig::default();
        let seeds = generate_rho_candidates(3, None, &cfg);
        let ln4 = 4.0_f64.ln();
        let has_nu2_manifold_seed = seeds
            .iter()
            .any(|s| s.len() == 3 && ((2.0 * s[1] - s[0] - s[2]) - ln4).abs() < 1e-8);
        assert!(has_nu2_manifold_seed);
    }

    #[test]
    fn three_penalty_seeds_include_general_spde_manifold_points() {
        let cfg = SeedConfig::default();
        let heur = [2.0, 10.0, 3.0];
        let seeds = generate_rho_candidates(3, Some(&heur), &cfg);
        let has_non_nu2 = seeds.iter().any(|s| {
            // For nu=2, 2*rho1-rho0-rho2 = ln(4).
            // General nu manifold should include points away from ln(4).
            s.len() == 3 && ((2.0 * s[1] - s[0] - s[2]) - 4.0_f64.ln()).abs() > 1e-3
        });
        assert!(has_non_nu2);
    }

    #[test]
    fn three_penalty_seeds_include_first_order_fallbackwith_rho2_floor() {
        let cfg = SeedConfig {
            bounds: (-12.0, 12.0),
            ..SeedConfig::default()
        };
        let seeds = generate_rho_candidates(3, None, &cfg);
        let has_floor = seeds
            .iter()
            .any(|s| s.len() == 3 && (s[2] - (-12.0)).abs() < 1e-12);
        assert!(has_floor);
    }

    #[test]
    fn auxiliary_trailing_dims_pinned_to_initial_values() {
        // Simulate SAS optimization: 2 smoothing dims + 2 auxiliary dims
        // (epsilon=0, log_delta=0).  The heuristic vector is in rho/theta
        // space for both smoothing and auxiliary dimensions.
        let cfg = SeedConfig {
            num_auxiliary_trailing: 2,
            risk_profile: SeedRiskProfile::GeneralizedLinear,
            ..SeedConfig::default()
        };
        let heur = [0.0, 10.0_f64.ln(), 0.0, 0.0]; // rhos + SAS initials
        let seeds = generate_rho_candidates(4, Some(&heur), &cfg);
        assert!(!seeds.is_empty());
        // EVERY seed must have the auxiliary dims pinned to 0.0.
        for (idx, seed) in seeds.iter().enumerate() {
            assert_eq!(seed.len(), 4);
            assert!(
                (seed[2] - 0.0).abs() < 1e-12 && (seed[3] - 0.0).abs() < 1e-12,
                "seed {} has auxiliary dims [{}, {}], expected [0, 0]",
                idx,
                seed[2],
                seed[3],
            );
        }
        // The smoothing dims should NOT all be zero (some seeds should vary them).
        let has_nonzero_smoothing = seeds
            .iter()
            .any(|s| s[0].abs() > 1e-12 || s[1].abs() > 1e-12);
        assert!(has_nonzero_smoothing);
    }

    #[test]
    fn auxiliary_dims_dedup_collapses_identical_seeds() {
        // With auxiliary pinning, seeds that differed only in aux dims
        // should collapse to a single seed.
        let cfg = SeedConfig {
            num_auxiliary_trailing: 1,
            max_seeds: 32,
            risk_profile: SeedRiskProfile::GeneralizedLinear,
            ..SeedConfig::default()
        };
        let seeds_with_aux = generate_rho_candidates(3, None, &cfg);
        let cfg_no_aux = SeedConfig {
            num_auxiliary_trailing: 0,
            max_seeds: 32,
            risk_profile: SeedRiskProfile::GeneralizedLinear,
            ..SeedConfig::default()
        };
        let seeds_without_aux = generate_rho_candidates(3, None, &cfg_no_aux);
        // Aux pinning causes many seeds to collapse, so fewer unique seeds.
        assert!(seeds_with_aux.len() <= seeds_without_aux.len());
    }

    #[test]
    fn objective_grid_seed_selects_lowest_finite_cost_candidate() {
        let base = Array1::from_vec(vec![0.0, 0.0]);
        let selected =
            select_objective_seed_on_log_lambda_grid(&base, (-12.0, 12.0), 2, &[], |rho| {
                Some((rho[0] - 6.0).powi(2) + (rho[1] - 6.0).powi(2))
            });

        assert!((selected[0] - 6.0).abs() < 1e-12);
        assert!((selected[1] - 6.0).abs() < 1e-12);
    }

    #[test]
    fn objective_grid_seed_keeps_baseline_when_no_candidate_improves_cost() {
        let base = Array1::from_vec(vec![1.0, -2.0]);
        let selected =
            select_objective_seed_on_log_lambda_grid(&base, (-12.0, 12.0), 2, &[], |rho| {
                if (rho[0] - 1.0).abs() < 1e-12 && (rho[1] + 2.0).abs() < 1e-12 {
                    Some(0.0)
                } else {
                    Some(1.0)
                }
            });

        assert_eq!(selected, base);
    }
}
