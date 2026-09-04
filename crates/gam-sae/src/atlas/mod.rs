#![cfg(test)]
//! #2280 acceptance, the NULL arm: the atlas must not name a manifold in
//! structureless noise.
//!
//! Every generator in the shared zoo (`manifold::tests_topology_fixtures`) samples
//! a manifold, so `the_readout_never_misnames_a_known_manifold_2280` scores the
//! readout only on inputs where a correct name exists. That test therefore cannot
//! observe the misnaming that would matter most — asserting structure in data that
//! has none — because it is never shown any. This file supplies the arm whose
//! correct answer is NOTHING.
//!
//! It is the control the rest of the acceptance set rests on. A classifier that
//! recognizes a topology in noise has proved nothing about the topologies it got
//! right, and the atlas's one surviving promotion argument on this issue is
//! precisely "every error it makes is an abstention, never a misnaming". That
//! claim has never been measured against a null.
//!
//! # Why this is two-sided, and why the arms are MATCHED
//!
//! A one-sided null test (`assert!(verdict.is_none())`) is satisfied by any broken
//! harness — a wrong shape, a build that always errors, a helper that always
//! returns `None`. So every null arm here is paired with a planted arm carried
//! through the SAME [`verdict`] call at the SAME `(n, p, d)`, differing only in
//! whether the rows lie on a manifold. The pair fails if the readout goes blind
//! (planted arm loses its name) and fails if it hallucinates (null arm gains one).
//!
//! # What "structureless" has to mean here
//!
//! Isotropic noise filling `ℝ^p` is not a `d`-manifold only while `d < p`: noise in
//! `ℝ²` read at `d = 2` IS locally a solid patch of the plane, and an atlas calling
//! that a disk would be right. Every null arm below therefore keeps `d < p`, which
//! is also the regime the production consumer runs in — `atlas_prior_for_coords`
//! builds at the birth's chart rank on a wider ambient residual.
//!
//! The cloud's scale is irrelevant by construction: `LocalAtlasConfig` sizes a
//! patch by a nearest-row COUNT and picks centers by farthest-point traversal, so
//! the cover is scale-free and a unit-variance cloud is not a special case.

use crate::manifold::{
    GraphCompressionKind, LocalAtlas, LocalAtlasConfig, observe_atlas_topology,
    tests_topology_fixtures::{circle, cylinder_strip, sphere, trefoil_knot},
};
use ndarray::{Array2, ArrayView2};

/// SplitMix64, the reference finalizer. The three shift widths and two odd
/// multipliers are the published algorithm's, not tuning: they are what make the
/// generator equidistributed over its full period, and changing any of them
/// changes which generator this is rather than how well it is tuned.
fn splitmix64(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// One uniform in `(0, 1]` from the stream: the top 53 bits are the f64 mantissa,
/// shifted off zero so the Box–Muller logarithm below cannot see it.
fn next_unit(state: &mut u64) -> f64 {
    let bits = splitmix64(state) >> 11;
    (bits as f64 + 1.0) / ((1u64 << 53) as f64 + 1.0)
}

/// `n × p` of independent standard normals, deterministic in `seed`.
///
/// Box–Muller rather than a sum-of-uniforms approximation: the null has to be
/// genuinely structureless, and a truncated CLT surrogate has measurable
/// higher-moment structure of its own, which is exactly the thing under test.
fn structureless_cloud(n: usize, p: usize, seed: u64) -> Array2<f64> {
    let mut state = seed;
    let mut z = Array2::<f64>::zeros((n, p));
    for row in 0..n {
        for col in 0..p {
            let u1 = next_unit(&mut state);
            let u2 = next_unit(&mut state);
            z[[row, col]] = (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos();
        }
    }
    z
}

/// The production path end to end, and the ONLY way any arm below gets a verdict:
/// build the atlas at the builder's own derived configuration and read its
/// topology.
///
/// A build refusal, a readout error and a readout that names nothing are the same
/// outcome for this question — no manifold was asserted — so all three collapse to
/// `None`. Distinguishing them would make the null arm sensitive to WHICH honest
/// refusal fired, which is a property of the cover rather than of the claim being
/// tested; the refusal string is printed instead, so a change of mechanism is
/// visible without being asserted.
fn verdict(z: ArrayView2<'_, f64>, d: usize) -> (Option<GraphCompressionKind>, String) {
    let config = LocalAtlasConfig::balanced(z.nrows(), d);
    let atlas = match LocalAtlas::build(z, config) {
        Ok(atlas) => atlas,
        Err(error) => return (None, format!("build refused: {error:?}")),
    };
    match observe_atlas_topology(&atlas) {
        Ok(readout) => {
            let named = readout.observed_manifold();
            let why = match (named, readout.refusal()) {
                (Some(kind), _) => format!("named {kind:?}"),
                (None, Some(refusal)) => format!("refused: {refusal}"),
                (None, None) => "no verdict and no refusal".to_string(),
            };
            (named, why)
        }
        Err(error) => (None, format!("readout errored: {error}")),
    }
}

/// The matched pairs. Each row is one `(n, p, d)` cell carrying a planted cloud
/// whose name is known by construction and a structureless cloud of the identical
/// shape, so the two arms differ in nothing but whether a manifold is there.
fn matched_pairs() -> Vec<(&'static str, Array2<f64>, GraphCompressionKind, Array2<f64>, usize)> {
    vec![
        (
            "circle",
            circle(400, 2.0),
            GraphCompressionKind::Circle,
            structureless_cloud(400, 3, 0x2280_0001),
            1,
        ),
        (
            "trefoil",
            trefoil_knot(600, 1.0),
            GraphCompressionKind::Circle,
            structureless_cloud(600, 3, 0x2280_0002),
            1,
        ),
        (
            "sphere",
            sphere(900),
            GraphCompressionKind::Sphere,
            structureless_cloud(900, 3, 0x2280_0003),
            2,
        ),
        (
            "cylinder",
            cylinder_strip(40, 10),
            GraphCompressionKind::Cylinder,
            structureless_cloud(400, 3, 0x2280_0004),
            2,
        ),
    ]
}

/// The null arm and its matched positive control, in one table so neither can be
/// read without the other.
///
/// The diagnostic prints unconditionally, before any assertion, so a failure and a
/// pass are read off the same output.
#[test]
fn structureless_noise_earns_no_topology_and_the_planted_shapes_still_do_2280() {
    struct Row {
        label: &'static str,
        d: usize,
        n: usize,
        p: usize,
        expected: GraphCompressionKind,
        planted_named: Option<GraphCompressionKind>,
        planted_why: String,
        noise_named: Option<GraphCompressionKind>,
        noise_why: String,
    }

    let rows: Vec<Row> = matched_pairs()
        .into_iter()
        .map(|(label, planted, expected, noise, d)| {
            let (planted_named, planted_why) = verdict(planted.view(), d);
            let (noise_named, noise_why) = verdict(noise.view(), d);
            Row {
                label,
                d,
                n: planted.nrows(),
                p: planted.ncols(),
                expected,
                planted_named,
                planted_why,
                noise_named,
                noise_why,
            }
        })
        .collect();

    // Unconditional, before any assertion: the table reads the same on a pass and
    // on a failure, so a changed refusal mechanism is legible either way.
    for row in &rows {
        eprintln!(
            "[2280-null] {:>9} d={} n={} p={} | planted: {:<46} | noise: {}",
            row.label, row.d, row.n, row.p, row.planted_why, row.noise_why
        );
    }

    for row in &rows {
        assert_eq!(
            row.planted_named,
            Some(row.expected),
            "the {} positive control lost its name at d={}: {}. The null arm is \
             only meaningful while this harness can still recognize a manifold it \
             is shown.",
            row.label,
            row.d,
            row.planted_why
        );
        assert_eq!(
            row.noise_named, None,
            "structureless noise at the {} cell (d={}, same n={} and p={} as the \
             planted arm) was NAMED {:?}: {}. The atlas's only promotable property \
             on #2280 is that every error it makes is an abstention; asserting a \
             manifold in noise refutes it.",
            row.label, row.d, row.n, row.p, row.noise_named, row.noise_why
        );
    }
}

/// The null verdict is a deterministic function of the cloud, like every other
/// atlas readout (`atlas_is_bit_identical_run_to_run_2280`,
/// `the_readout_is_bit_identical_run_to_run_2280`). Re-deriving the cloud from the
/// seed rather than cloning it also pins the generator itself: a stateful or
/// address-dependent stream would diverge here.
#[test]
fn the_null_is_bit_identical_run_to_run_2280() {
    for d in [1usize, 2] {
        let first = structureless_cloud(500, 4, 0x2280_0005);
        let second = structureless_cloud(500, 4, 0x2280_0005);
        assert_eq!(first, second, "the cloud generator is not deterministic");
        let (a, why_a) = verdict(first.view(), d);
        let (b, why_b) = verdict(second.view(), d);
        eprintln!("[2280-null] determinism d={d}: {why_a} | {why_b}");
        assert_eq!(a, b, "the null verdict is not bit-identical at d={d}");
        assert_eq!(why_a, why_b, "the null refusal is not stable at d={d}");
        assert_eq!(a, None, "noise in R^4 read at d={d} was named {a:?}: {why_a}");
    }
}
