//! Adversarial verbatim edits (#2951): the edit an explanation fails on worst, searched by ascent of
//! the gap `KL(M_e ‖ P_e)` over the edit's own parameters, the edit applied identically to `M` and
//! to `P` like every experiment.
//!
//! A search pushes one stream site at one row by one typical norm along a unit direction `θ`
//! ([`draw`] gives its sequence, site, row and start, from a seed alone, so every explanation faces
//! the same searches). [`ascend`] climbs the gap on the sphere of directions: each step estimates
//! the gradient there by central differences along random tangent directions `u` (`θ` turned by
//! ±0.05 rad toward `u`), then turns `θ` toward the estimate by the best of π/16, π/8, π/4 and π/2,
//! kept only where the gap rises. The caller scores candidates (each one experiment, run through
//! both models' own forward passes), so the search serves any explanation the caller can score.
//! It stops when the gain over the last three steps is below 1% of the gap, or after its steps.

use crate::interchange::SharedSite;
use rand::{RngExt, SeedableRng, rngs::StdRng};

/// The probes' turn, in radians.
const PROBE: f64 = 0.05;

/// One search's draw: its sequence (an index into the caller's rows), stream site, row (after the
/// first) and starting unit direction.
#[derive(Clone, Debug, PartialEq)]
pub struct Draw {
    pub sequence: usize,
    pub site: SharedSite,
    pub position: usize,
    pub start: Vec<f64>,
}

/// One search's outcome: the gap along the ascent (the start's first), the found direction, and
/// whether it stopped by saturation rather than by its step count.
#[derive(Clone, Debug, PartialEq)]
pub struct Found {
    pub path: Vec<f64>,
    pub direction: Vec<f64>,
    pub saturated: bool,
}

fn unit(v: Vec<f64>) -> Vec<f64> {
    let norm = v.iter().map(|x| x * x).sum::<f64>().sqrt();
    v.into_iter().map(|x| x / norm).collect()
}

fn normal(rng: &mut StdRng, width: usize) -> Vec<f64> {
    (0..width)
        .map(|_| {
            let (a, b): (f64, f64) = (rng.random::<f64>().max(f64::MIN_POSITIVE), rng.random());
            (-2.0 * a.ln()).sqrt() * (std::f64::consts::TAU * b).cos()
        })
        .collect()
}

fn turn(t: &[f64], u: &[f64], angle: f64) -> Vec<f64> {
    t.iter().zip(u).map(|(x, y)| angle.cos() * x + angle.sin() * y).collect()
}

/// `searches` draws over `sequences` rows of `length` tokens and a model of `blocks` blocks whose
/// stream is `width` wide, from `seed` alone.
pub fn draw(seed: u64, searches: usize, sequences: usize, length: usize, blocks: usize, width: usize) -> Result<Vec<Draw>, String> {
    if sequences == 0 || length < 2 || blocks == 0 || width == 0 {
        return Err("adversary: no rows, rows of one token, no blocks or no width".into());
    }
    let mut rng = StdRng::seed_from_u64(seed ^ 0x4144_5645_5253);
    Ok((0..searches)
        .map(|_| {
            let sequence = rng.random_range(0..sequences);
            let site = SharedSite::Stream(rng.random_range(0..blocks));
            let position = rng.random_range(1..length);
            Draw { sequence, site, position, start: unit(normal(&mut rng, width)) }
        })
        .collect())
}

/// The ascent of search `index` (its probes drawn from `seed` and `index` alone) from `start`, at
/// most `steps` steps of `probes` probes, `score` giving the gap of each candidate direction.
pub fn ascend(seed: u64, index: usize, start: &[f64], steps: usize, probes: usize, mut score: impl FnMut(Vec<Vec<f64>>) -> Result<Vec<f64>, String>) -> Result<Found, String> {
    let mut rng = StdRng::seed_from_u64(seed ^ 0x5052_4f42_4553 ^ (index as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15));
    let width = start.len();
    let mut theta = start.to_vec();
    let mut current = *score(vec![theta.clone()])?.first().ok_or("adversary: no score")?;
    let mut path = vec![current];
    for _ in 0..steps {
        let tangents: Vec<Vec<f64>> = (0..probes)
            .map(|_| {
                let u = normal(&mut rng, width);
                let along = u.iter().zip(&theta).map(|(a, b)| a * b).sum::<f64>();
                unit(u.iter().zip(&theta).map(|(a, b)| a - along * b).collect())
            })
            .collect();
        let f = score(tangents.iter().flat_map(|u| [turn(&theta, u, PROBE), turn(&theta, u, -PROBE)]).collect())?;
        let mut g = vec![0.0; width];
        for (j, u) in tangents.iter().enumerate() {
            let slope = (f[2 * j] - f[2 * j + 1]) / (2.0 * PROBE);
            g.iter_mut().zip(u).for_each(|(gk, uk)| *gk += slope * uk);
        }
        if g.iter().all(|v| *v == 0.0) {
            path.push(current);
            return Ok(Found { path, direction: theta, saturated: true });
        }
        let ascent = unit(g);
        let turns: Vec<Vec<f64>> = [16.0, 8.0, 4.0, 2.0].iter().map(|k| turn(&theta, &ascent, std::f64::consts::PI / k)).collect();
        let f = score(turns.clone())?;
        if let Some((best, value)) = f.iter().copied().enumerate().max_by(|a, b| a.1.total_cmp(&b.1))
            && value > current
        {
            theta = turns[best].clone();
            current = value;
        }
        path.push(current);
        if path.len() > 3 && current - path[path.len() - 4] < 0.01 * current.abs() {
            return Ok(Found { path, direction: theta, saturated: true });
        }
    }
    Ok(Found { path, direction: theta, saturated: false })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// On a gap with one maximum on the sphere, `f(θ) = 1 + θ·a` (`a` a unit vector), the ascent
    /// climbs from a random start to within 1% of the maximum 2, never falls, and its draws are the
    /// same for the same seed.
    #[test]
    fn the_ascent_climbs_a_smooth_gap_to_its_maximum() {
        let width = 32;
        let a = unit((0..width).map(|i| (i as f64 * 0.7).sin()).collect());
        let draws = draw(5, 2, 3, 10, 4, width).expect("draws");
        assert_eq!(draws, draw(5, 2, 3, 10, 4, width).expect("draws"));
        let found = ascend(5, 0, &draws[0].start, 60, 8, |candidates| Ok(candidates.iter().map(|t| 1.0 + t.iter().zip(&a).map(|(x, y)| x * y).sum::<f64>()).collect())).expect("the ascent");
        assert!(found.path.windows(2).all(|w| w[1] >= w[0]), "the gap never falls: {:?}", found.path);
        assert!(*found.path.last().expect("a path") > 0.99 * 2.0, "the ascent reaches {:?}", found.path.last());
    }
}
