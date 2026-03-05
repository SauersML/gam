use ndarray::Array1;
use std::collections::HashSet;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SeedRiskProfile {
    Gaussian,
    GeneralizedLinear,
    Survival,
}

#[derive(Clone, Copy, Debug)]
pub struct SeedConfig {
    pub bounds: (f64, f64),
    pub max_seeds: usize,
    pub screening_budget: usize,
    pub screen_max_inner_iterations: usize,
    pub risk_profile: SeedRiskProfile,
}

impl Default for SeedConfig {
    fn default() -> Self {
        Self {
            bounds: (-12.0, 12.0),
            max_seeds: 16,
            screening_budget: 4,
            screen_max_inner_iterations: 5,
            risk_profile: SeedRiskProfile::GeneralizedLinear,
        }
    }
}

fn normalize_bounds(bounds: (f64, f64)) -> (f64, f64) {
    if bounds.0 <= bounds.1 {
        bounds
    } else {
        (bounds.1, bounds.0)
    }
}

fn clamp_to_bounds(value: f64, bounds: (f64, f64)) -> f64 {
    let (lo, hi) = normalize_bounds(bounds);
    value.clamp(lo, hi)
}

fn add_seed_dedup(seeds: &mut Vec<Array1<f64>>, seen: &mut HashSet<Vec<u64>>, seed: Array1<f64>) {
    let key: Vec<u64> = seed.iter().map(|&v| v.to_bits()).collect();
    if seen.insert(key) {
        seeds.push(seed);
    }
}

fn rho_from_lambda(lambda: f64, bounds: (f64, f64)) -> f64 {
    clamp_to_bounds(lambda.max(1e-12).ln(), bounds)
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
    let log_c0 = 0.0;
    let log_c1 = safe_ln_pos(nu)?;
    let log_c2 = safe_ln_pos(0.5 * nu * (nu - 1.0))?;
    let rho0 = clamp_to_bounds(log_tau + log_c0 + 2.0 * nu * log_kappa, bounds);
    let rho1 = clamp_to_bounds(log_tau + log_c1 + 2.0 * (nu - 1.0) * log_kappa, bounds);
    let rho2 = clamp_to_bounds(log_tau + log_c2 + 2.0 * (nu - 2.0) * log_kappa, bounds);
    Some(Array1::from_vec(vec![rho0, rho1, rho2]))
}

fn add_spde_manifold_seeds(
    seeds: &mut Vec<Array1<f64>>,
    seen: &mut HashSet<Vec<u64>>,
    bounds: (f64, f64),
    heuristic_lambdas: Option<&[f64]>,
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

    // Data-informed anchor: invert heuristic lambdas to (nu, kappa^2, tau) when feasible.
    if let Some(vals) = heuristic_lambdas
        && vals.len() == 3
    {
        let l0 = vals[0];
        let l1 = vals[1];
        let l2 = vals[2];
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
    heuristic_lambdas: Option<&[f64]>,
) {
    // Degenerate λ2 -> 0 fallback (first-order mass+tension):
    // λ0 = τ κ^2, λ1 = τ, λ2 ≈ 0.
    let rho2_floor = bounds.0;
    let default_log_kappa = [-2.0, -1.0, 0.0, 1.0];
    let default_log_tau = [0.0, -2.0, 2.0];
    for &t in &default_log_tau {
        for &lk in &default_log_kappa {
            let rho0 = clamp_to_bounds(t + 2.0 * lk, bounds);
            let rho1 = clamp_to_bounds(t, bounds);
            add_seed_dedup(seeds, seen, Array1::from_vec(vec![rho0, rho1, rho2_floor]));
        }
    }
    if let Some(vals) = heuristic_lambdas
        && vals.len() == 3
        && vals[0].is_finite()
        && vals[1].is_finite()
        && vals[0] > 1e-12
        && vals[1] > 1e-12
    {
        let kappa2 = vals[0] / vals[1];
        if kappa2.is_finite() && kappa2 > 0.0 {
            let lk = 0.5 * kappa2.ln();
            let t = vals[1].ln();
            let rho0 = clamp_to_bounds(t + 2.0 * lk, bounds);
            let rho1 = clamp_to_bounds(t, bounds);
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
            // Thread-2 reverse map at nu=2:
            // lambda0 = tau * kappa^4, lambda1 = tau * 2*kappa^2, lambda2 = tau.
            let rho2 = clamp_to_bounds(tau_rho, bounds);
            let rho1 = clamp_to_bounds(tau_rho + ln_two + 2.0 * log_kappa, bounds);
            let rho0 = clamp_to_bounds(tau_rho + 4.0 * log_kappa, bounds);
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
    heuristic_lambdas: Option<&[f64]>,
    config: &SeedConfig,
) -> Vec<Array1<f64>> {
    let mut seeds = Vec::new();
    let mut seen: HashSet<Vec<u64>> = HashSet::new();

    let bounds = normalize_bounds(config.bounds);
    let max_seeds = config.max_seeds.max(1);
    let risk_shift = match config.risk_profile {
        SeedRiskProfile::Gaussian => 0.0,
        SeedRiskProfile::GeneralizedLinear => 1.0,
        SeedRiskProfile::Survival => 2.0,
    };

    if num_penalties == 0 {
        add_seed_dedup(&mut seeds, &mut seen, Array1::<f64>::zeros(0));
        return seeds;
    }

    // Prefer a full heuristic vector (length == k) as the primary anchor.
    let heuristic_rho_vec: Option<Array1<f64>> = heuristic_lambdas.and_then(|vals| {
        if vals.len() == num_penalties {
            Some(Array1::from_iter(
                vals.iter().copied().map(|v| rho_from_lambda(v, bounds)),
            ))
        } else {
            None
        }
    });

    let primary = heuristic_rho_vec
        .clone()
        .unwrap_or_else(|| Array1::<f64>::zeros(num_penalties))
        .mapv(|v| clamp_to_bounds(v + risk_shift, bounds));
    add_seed_dedup(&mut seeds, &mut seen, primary.clone());
    // Always include neutral baseline independently of heuristic anchor.
    add_seed_dedup(&mut seeds, &mut seen, Array1::zeros(num_penalties));
    // For exactly three penalties (mass/tension/stiffness), inject
    // physically coherent manifold seeds in rho-space:
    // - general SPDE manifold over (log_tau, log_kappa, nu),
    // - backward-compatible nu=2 reverse-map seeds,
    // - first-order fallback seeds (lambda2 near lower bound).
    if num_penalties == 3 {
        add_spde_manifold_seeds(&mut seeds, &mut seen, bounds, heuristic_lambdas, &primary);
        add_nu2_reverse_manifold_seeds(&mut seeds, &mut seen, bounds, &primary);
        add_first_order_fallback_seeds(&mut seeds, &mut seen, bounds, heuristic_lambdas);
    }

    // Backward-compatible scalar heuristic support: treat each value as a symmetric λ seed.
    if let Some(vals) = heuristic_lambdas
        && vals.len() != num_penalties
    {
        for &lambda in vals {
            let rho = rho_from_lambda(lambda, bounds);
            add_seed_dedup(&mut seeds, &mut seen, Array1::from_elem(num_penalties, rho));
        }
    }

    // Broad symmetric baselines around the center to guarantee global coverage.
    let baseline_centers: &[f64] = match config.risk_profile {
        SeedRiskProfile::Gaussian => &[0.0, -3.0, 3.0, -6.0, 6.0],
        SeedRiskProfile::GeneralizedLinear => &[0.0, 2.0, 4.0, -2.0],
        SeedRiskProfile::Survival => &[0.0, 2.0, 4.0, 6.0],
    };
    for &center in baseline_centers {
        add_seed_dedup(
            &mut seeds,
            &mut seen,
            Array1::from_elem(num_penalties, clamp_to_bounds(center, bounds)),
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
            conflict_a[i] = clamp_to_bounds(primary[i] + large_scale, bounds);
        }
        for &i in small_cluster {
            conflict_a[i] = clamp_to_bounds(primary[i] - small_scale, bounds);
        }
        add_seed_dedup(&mut seeds, &mut seen, conflict_a);

        let mut conflict_b = primary.clone();
        for &i in large_cluster {
            conflict_b[i] = clamp_to_bounds(primary[i] - large_scale, bounds);
        }
        for &i in small_cluster {
            conflict_b[i] = clamp_to_bounds(primary[i] + small_scale, bounds);
        }
        add_seed_dedup(&mut seeds, &mut seen, conflict_b);

        let mut heavy_up = primary.clone();
        for &i in large_cluster {
            heavy_up[i] = clamp_to_bounds(primary[i] + large_scale, bounds);
        }
        add_seed_dedup(&mut seeds, &mut seen, heavy_up);

        let mut light_down = primary.clone();
        for &i in small_cluster {
            light_down[i] = clamp_to_bounds(primary[i] - small_scale, bounds);
        }
        add_seed_dedup(&mut seeds, &mut seen, light_down);
    } else {
        // Low-dimensional path: coordinate and sparse pair probes are still cheap.
        for i in 0..dims_to_touch {
            let scale = step_base + 0.25 * primary[i].abs().min(8.0);
            for dir in [-1.0, 1.0] {
                let mut s = primary.clone();
                s[i] = clamp_to_bounds(primary[i] + dir * scale, bounds);
                add_seed_dedup(&mut seeds, &mut seen, s);
            }
        }

        let pair_dims = num_penalties.min(6);
        for i in 0..pair_dims {
            for j in (i + 1)..pair_dims {
                let mut s1 = primary.clone();
                s1[i] = clamp_to_bounds(primary[i] + step_base, bounds);
                s1[j] = clamp_to_bounds(primary[j] - step_base, bounds);
                add_seed_dedup(&mut seeds, &mut seen, s1);

                let mut s2 = primary.clone();
                s2[i] = clamp_to_bounds(primary[i] - step_base, bounds);
                s2[j] = clamp_to_bounds(primary[j] + step_base, bounds);
                add_seed_dedup(&mut seeds, &mut seen, s2);
            }
        }
    }

    // Global shrink/expand sweeps from the anchor to probe over/under-smoothing regimes.
    let global_shifts: &[f64] = match config.risk_profile {
        SeedRiskProfile::Gaussian => &[-2.0, 2.0, -4.0, 4.0],
        SeedRiskProfile::GeneralizedLinear => &[0.0, 2.0, 4.0, -1.0],
        SeedRiskProfile::Survival => &[0.0, 2.0, 4.0, 6.0],
    };
    for &shift in global_shifts {
        let swept = primary.mapv(|v| clamp_to_bounds(v + shift, bounds));
        add_seed_dedup(&mut seeds, &mut seen, swept);
    }

    // Low-discrepancy exploratory seeds around the anchor for basin discovery.
    // These are still deterministic and do not encode any solver-side bias.
    let exploratory = max_seeds.saturating_sub(seeds.len()).min(8);
    if exploratory > 0 {
        let primes = first_primes(num_penalties.max(1));
        let amp = match config.risk_profile {
            SeedRiskProfile::Gaussian => 2.0,
            SeedRiskProfile::GeneralizedLinear => 2.5,
            SeedRiskProfile::Survival => 3.0,
        };
        for t in 0..exploratory {
            let mut s = primary.clone();
            for i in 0..num_penalties {
                let u = halton(t + 1, primes[i]); // (0,1)
                let centered = 2.0 * u - 1.0; // (-1,1)
                s[i] = clamp_to_bounds(primary[i] + amp * centered, bounds);
            }
            add_seed_dedup(&mut seeds, &mut seen, s);
        }
    }

    if seeds.len() > max_seeds {
        seeds.truncate(max_seeds);
    }

    if seeds.is_empty() {
        seeds.push(Array1::<f64>::zeros(num_penalties));
    }

    seeds
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn uses_full_heuristic_vector_as_primary_anchor() {
        let cfg = SeedConfig {
            risk_profile: SeedRiskProfile::Gaussian,
            ..SeedConfig::default()
        };
        let heur = [1e-2, 1.0, 1e2];
        let seeds = generate_rho_candidates(3, Some(&heur), &cfg);
        assert!(!seeds.is_empty());
        let first = &seeds[0];
        assert_eq!(first.len(), 3);
        assert!((first[0] - heur[0].ln()).abs() < 1e-12);
        assert!((first[1] - heur[1].ln()).abs() < 1e-12);
        assert!((first[2] - heur[2].ln()).abs() < 1e-12);
    }

    #[test]
    fn high_dim_uses_cluster_conflict_probes_without_exploding() {
        let cfg = SeedConfig {
            max_seeds: 18,
            risk_profile: SeedRiskProfile::GeneralizedLinear,
            ..SeedConfig::default()
        };
        let heur = [1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0, 1e-1, 1.0, 10.0, 100.0];
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
    fn includes_neutral_zero_seed() {
        let cfg = SeedConfig::default();
        let seeds = generate_rho_candidates(5, None, &cfg);
        let has_zero = seeds
            .iter()
            .any(|s| s.iter().all(|v| (*v - 0.0).abs() < 1e-12));
        assert!(has_zero);
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
    fn three_penalty_seeds_include_first_order_fallback_with_rho2_floor() {
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
}
