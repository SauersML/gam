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
    tests_topology_fixtures::{circle, cylinder_strip, mobius_strip, sphere, trefoil_knot},
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
/// tested.
///
/// The reported line is the readout's OWN `Display`, which already carries the
/// verdict-or-refusal plus every invariant it rests on, and the cover multiplicity
/// pair that `Display` omits. That matters most in the outcome this test would
/// hate: if noise is ever NAMED, the line reporting it must already contain the
/// invariants that produced the name, or the refutation needs a second run to be
/// diagnosable.
fn verdict(z: ArrayView2<'_, f64>, d: usize) -> (Option<GraphCompressionKind>, String) {
    let config = LocalAtlasConfig::balanced(z.nrows(), d);
    let atlas = match LocalAtlas::build(z, config) {
        Ok(atlas) => atlas,
        Err(error) => return (None, format!("build refused: {error:?}")),
    };
    match observe_atlas_topology(&atlas) {
        Ok(readout) => {
            let inv = readout.invariants();
            let line = format!(
                "{readout} max_mult={} mean_mult={:.3}",
                inv.max_cover_multiplicity, inv.mean_cover_multiplicity
            );
            (readout.observed_manifold(), line)
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
        // The holonomy cell. `cylinder_and_mobius_differ_only_by_the_holonomy_class_2280`
        // (`manifold/atlas_topology.rs`) already pins the Mobius band against its cylinder
        // at identical `(b0, b1, b2, chi)`, which is what makes the orientation class the
        // only thing separating them — that pair is not rebuilt here. What it does not
        // have is a NULL: it is positive-only. This row supplies one, so the single
        // measurement the readout makes most confidently is also the one whose matched
        // noise arm has to refuse.
        (
            "mobius",
            mobius_strip(40, 10),
            GraphCompressionKind::MobiusStrip,
            structureless_cloud(400, 3, 0x2280_0006),
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
            "[2280-null] {} d={} n={} p={}\n  planted: {}\n  noise:   {}",
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

// ---------------------------------------------------------------------------
// The roll sweep: is the verdict actually intrinsic?
//
// #2280's thesis is that "every input to the verdict is a transition between
// charts, and a transition is intrinsic". The trefoil row tests that for a
// 1-manifold and passes. This tests it for the case the issue's acceptance list
// names and the readout currently refuses: the swiss roll.
//
// The sweep holds the INTRINSIC manifold fixed — every arm is a flat rectangular
// sheet, `chi = 1`, `b1 = 0`, a disk — and varies only how tightly the ambient
// embedding winds it. If the verdict is intrinsic, every arm reads `Disk`. If the
// verdict tracks the winding, then the ambient embedding is reaching the readout,
// and the mechanism is the cover: `LocalAtlas` grows its patches by AMBIENT
// nearest rows, so on a tight roll a patch spans two sheets that are ambient-near
// and geodesically far. That is the same "genuine neighbour vs. ambient near-miss"
// distinction the trefoil-400 regression on this issue already exposed, and this
// sweep is built to say whether the swiss-roll refusal is one defect with it or
// two.
// ---------------------------------------------------------------------------

/// A flat `(t, h)` sheet wound into ambient 3-D across `turns` revolutions,
/// spanning a FIXED radial extent.
///
/// Holding the radial extent fixed while raising `turns` is what makes this a
/// control rather than a demonstration: the gap between successive windings
/// shrinks as `turns` grows, while the sheet stays intrinsically the same flat
/// rectangle. `turns = 1.5` reproduces the acceptance list's "swiss roll (1.5+
/// turns)"; the small-`turns` arms are gently curved sheets whose ambient and
/// geodesic metrics agree.
fn rolled_sheet(n_t: usize, n_h: usize, turns: f64) -> Array2<f64> {
    const INNER_RADIUS: f64 = 1.0;
    const OUTER_RADIUS: f64 = 4.0;
    const HEIGHT: f64 = 2.0;
    let n = n_t * n_h;
    let mut z = Array2::<f64>::zeros((n, 3));
    let mut row = 0usize;
    for it in 0..n_t {
        let s = (it as f64) / (n_t as f64 - 1.0);
        let radius = INNER_RADIUS + (OUTER_RADIUS - INNER_RADIUS) * s;
        let angle = std::f64::consts::TAU * turns * s;
        for ih in 0..n_h {
            z[[row, 0]] = radius * angle.cos();
            z[[row, 1]] = radius * angle.sin();
            z[[row, 2]] = HEIGHT * (ih as f64) / (n_h as f64 - 1.0);
            row += 1;
        }
    }
    z
}

/// Every arm of the roll sweep is intrinsically a disk, so the readout must
/// either say `Disk` or say nothing — and it must say `Disk` where the ambient
/// and geodesic metrics agree.
///
/// The gentle arm is the non-vacuity control: without it, "refuses on every arm"
/// would pass while telling us only that the harness is broken. The assertion at
/// the tight end is deliberately the weaker, honest one — no misnaming — because
/// the tight roll's verdict is the open question this issue records, and asserting
/// `Disk` there would be asserting the fix rather than measuring the defect.
#[test]
fn a_rolled_sheet_is_a_disk_at_every_winding_or_nothing_at_all_2280() {
    let mut rows = Vec::new();
    for turns in [0.25_f64, 0.5, 1.0, 1.5, 2.5] {
        let z = rolled_sheet(40, 12, turns);
        let (named, why) = verdict(z.view(), 2);
        rows.push((turns, named, why));
    }
    for (turns, _, why) in &rows {
        eprintln!("[2280-roll] turns={turns:>4} {why}");
    }

    let gentle = rows
        .first()
        .expect("the sweep has a gentle arm by construction");
    assert_eq!(
        gentle.1,
        Some(GraphCompressionKind::Disk),
        "a barely-curved sheet must read as a disk, or the sweep below measures a \
         broken harness rather than the winding: {}",
        gentle.2
    );

    for (turns, named, why) in &rows {
        assert!(
            matches!(named, None | Some(GraphCompressionKind::Disk)),
            "a rolled sheet is intrinsically a disk at every winding, so the readout \
             may name `Disk` or refuse — naming anything else is a misnaming, which is \
             the one property this readout has never violated. turns={turns}: {why}"
        );
    }
}
