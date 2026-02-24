use ndarray::Array1;
use std::collections::HashSet;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SeedStrategy {
    Exhaustive,
    Light,
    Single,
}

#[derive(Clone, Copy, Debug)]
pub struct SeedConfig {
    pub strategy: SeedStrategy,
    pub bounds: (f64, f64),
}

impl Default for SeedConfig {
    fn default() -> Self {
        Self {
            strategy: SeedStrategy::Exhaustive,
            bounds: (-12.0, 12.0),
        }
    }
}

fn clamp_to_bounds(value: f64, bounds: (f64, f64)) -> f64 {
    let (lo, hi) = if bounds.0 <= bounds.1 {
        bounds
    } else {
        (bounds.1, bounds.0)
    };
    value.clamp(lo, hi)
}

fn base_values(strategy: SeedStrategy, bounds: (f64, f64)) -> Vec<f64> {
    match strategy {
        SeedStrategy::Single => vec![0.0],
        SeedStrategy::Light | SeedStrategy::Exhaustive => {
            let step = if strategy == SeedStrategy::Light {
                4.0
            } else {
                2.0
            };
            let (lo, hi) = if bounds.0 <= bounds.1 {
                bounds
            } else {
                (bounds.1, bounds.0)
            };
            let mut values = Vec::new();
            let mut v = hi;
            while v >= lo - 1e-9 {
                values.push(v);
                v -= step;
            }
            values
        }
    }
}

fn single_axis_values(strategy: SeedStrategy) -> Vec<f64> {
    match strategy {
        SeedStrategy::Single => Vec::new(),
        SeedStrategy::Light => vec![8.0, -8.0],
        SeedStrategy::Exhaustive => vec![12.0, 4.0, -4.0, -12.0],
    }
}

fn pairwise_templates(strategy: SeedStrategy) -> Vec<(f64, f64)> {
    match strategy {
        SeedStrategy::Exhaustive => vec![(12.0, 0.0), (8.0, -4.0), (6.0, -2.0)],
        SeedStrategy::Light | SeedStrategy::Single => Vec::new(),
    }
}

pub fn generate_rho_candidates(
    num_penalties: usize,
    heuristic_lambdas: Option<&[f64]>,
    config: &SeedConfig,
) -> Vec<Array1<f64>> {
    let mut seeds = Vec::new();

    if let Some(lambdas) = heuristic_lambdas {
        for &lambda in lambdas {
            let rho = clamp_to_bounds(lambda.max(1e-12).ln(), config.bounds);
            seeds.push(Array1::from_elem(num_penalties, rho));
        }
    }

    for val in base_values(config.strategy, config.bounds) {
        let clamped = clamp_to_bounds(val, config.bounds);
        seeds.push(Array1::from_elem(num_penalties, clamped));
    }

    if num_penalties > 0 {
        let axis_values: Vec<f64> = single_axis_values(config.strategy)
            .into_iter()
            .map(|v| clamp_to_bounds(v, config.bounds))
            .collect();
        for idx in 0..num_penalties {
            for &val in &axis_values {
                let mut seed = Array1::zeros(num_penalties);
                seed[idx] = val;
                seeds.push(seed);
            }
        }
    }

    if num_penalties >= 2 {
        let templates: Vec<(f64, f64)> = pairwise_templates(config.strategy)
            .into_iter()
            .map(|(hi, lo)| {
                (
                    clamp_to_bounds(hi, config.bounds),
                    clamp_to_bounds(lo, config.bounds),
                )
            })
            .collect();
        for i in 0..num_penalties {
            for j in (i + 1)..num_penalties {
                for &(hi, lo) in &templates {
                    let mut seed_ij = Array1::zeros(num_penalties);
                    seed_ij[i] = hi;
                    seed_ij[j] = lo;
                    seeds.push(seed_ij);

                    let mut seed_ji = Array1::zeros(num_penalties);
                    seed_ji[i] = lo;
                    seed_ji[j] = hi;
                    seeds.push(seed_ji);
                }
            }
        }
    }

    let mut seen: HashSet<Vec<u64>> = HashSet::new();
    let mut unique: Vec<Array1<f64>> = Vec::with_capacity(seeds.len());
    for s in seeds.into_iter() {
        let key: Vec<u64> = s.iter().map(|&v| v.to_bits()).collect();
        if seen.insert(key) {
            unique.push(s);
        }
    }

    unique
}
