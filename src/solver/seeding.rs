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
        SeedRiskProfile::GeneralizedLinear => 1.5,
        SeedRiskProfile::Survival => 2.5,
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

    // Backward-compatible scalar heuristic support: treat each value as a symmetric λ seed.
    if let Some(vals) = heuristic_lambdas {
        if vals.len() != num_penalties {
            for &lambda in vals {
                let rho = rho_from_lambda(lambda, bounds);
                add_seed_dedup(&mut seeds, &mut seen, Array1::from_elem(num_penalties, rho));
            }
        }
    }

    // Broad symmetric baselines around the center to guarantee global coverage.
    let baseline_centers: &[f64] = match config.risk_profile {
        SeedRiskProfile::Gaussian => &[0.0, -3.0, 3.0, -6.0, 6.0],
        SeedRiskProfile::GeneralizedLinear => &[0.5, 2.0, 4.0, -1.0],
        SeedRiskProfile::Survival => &[2.0, 4.0, 6.0, 0.0],
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
}
