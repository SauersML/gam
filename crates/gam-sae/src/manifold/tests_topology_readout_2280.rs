//! Recognition regression for the atlas topology readout (#2280).
//!
//! The readout is a DIAGNOSTIC that runs alongside the intrinsic-metric seed race
//! and names the coarse topology of an atom's output-energy cluster. Its
//! acceptance bar (set by the geometry owner, fable-mobius) is the numpy
//! falsification suite: it must correctly recognize the sphere, torus, Möbius band,
//! and swiss-roll sheet WITHOUT the false positives that killed the two earlier
//! designs (naïve nerve χ + holonomy; witness-triangulation homology) — see the
//! [`super::topology_readout`] module docs for the two falsification maps.
//!
//! Every fixture here is a DETERMINISTIC point cloud (a grid or a Fibonacci
//! lattice, no RNG), matching the numpy validation grids one-for-one, so the
//! recognition is reproducible run-to-run and the thresholds are the ones the numpy
//! suite validated stable across landmark density, noise, and seed.

use super::{atlas_topology_readout, AtlasTopology};
use ndarray::Array2;
use std::f64::consts::PI;

/// Fibonacci-lattice sphere S² ⊂ R³ (deterministic, near-uniform): χ = 2, closed,
/// orientable.
fn sphere(n: usize) -> Array2<f64> {
    let mut z = Array2::<f64>::zeros((n, 3));
    let golden = PI * (1.0 + 5.0_f64.sqrt());
    for i in 0..n {
        let y = 1.0 - 2.0 * (i as f64 + 0.5) / n as f64;
        let r = (1.0 - y * y).max(0.0).sqrt();
        let th = golden * i as f64;
        z[[i, 0]] = r * th.cos();
        z[[i, 1]] = y;
        z[[i, 2]] = r * th.sin();
    }
    z
}

/// Torus of revolution (R = 3, r = 1) as an na × nb angle grid: χ = 0, closed,
/// orientable, developable.
fn torus(na: usize, nb: usize) -> Array2<f64> {
    let (rr, r0) = (3.0_f64, 1.0_f64);
    let mut z = Array2::<f64>::zeros((na * nb, 3));
    for i in 0..na {
        for j in 0..nb {
            let a = 2.0 * PI * i as f64 / na as f64;
            let b = 2.0 * PI * j as f64 / nb as f64;
            let row = i * nb + j;
            z[[row, 0]] = (rr + r0 * b.cos()) * a.cos();
            z[[row, 1]] = (rr + r0 * b.cos()) * a.sin();
            z[[row, 2]] = r0 * b.sin();
        }
    }
    z
}

/// Flat disk (a slightly tilted plane) as a polar grid: χ = 1, bounded, orientable.
fn disk(nr: usize, na: usize) -> Array2<f64> {
    let mut pts: Vec<[f64; 3]> = vec![[0.0, 0.0, 0.0]];
    for i in 1..nr {
        let rr = i as f64 / nr as f64;
        for j in 0..na {
            let a = 2.0 * PI * j as f64 / na as f64;
            pts.push([rr * a.cos(), rr * a.sin(), 0.2 * rr * a.cos()]);
        }
    }
    to_array(&pts)
}

/// Swiss-roll SHEET (a flat 2-D sheet rolled up in 3-D) as a (t, h) grid: bounded,
/// orientable, developable — the #2240/#2280 fold. Topologically a disk/sheet.
fn swiss(nt: usize, nh: usize) -> Array2<f64> {
    let mut z = Array2::<f64>::zeros((nt * nh, 3));
    for i in 0..nt {
        let t = 1.2 * PI + (3.4 * PI - 1.2 * PI) * i as f64 / (nt - 1) as f64;
        for j in 0..nh {
            let h = 12.0 * j as f64 / (nh - 1) as f64;
            let row = i * nh + j;
            z[[row, 0]] = t * t.cos();
            z[[row, 1]] = h;
            z[[row, 2]] = t * t.sin();
        }
    }
    z
}

/// Möbius band as an (s, w) grid: NON-orientable, bounded. `s` wraps [0,1) with a
/// half-twist so the deck map (s, w) → (s+1, −w) identifies the ends.
fn mobius(ns: usize, nw: usize) -> Array2<f64> {
    let mut z = Array2::<f64>::zeros((ns * nw, 3));
    for i in 0..ns {
        let s = i as f64 / ns as f64;
        let th = 2.0 * PI * s;
        for j in 0..nw {
            let w = -0.4 + 0.8 * j as f64 / (nw - 1) as f64;
            let row = i * nw + j;
            z[[row, 0]] = (1.0 + w * (th / 2.0).cos()) * th.cos();
            z[[row, 1]] = (1.0 + w * (th / 2.0).cos()) * th.sin();
            z[[row, 2]] = w * (th / 2.0).sin();
        }
    }
    z
}

/// Open cylinder S¹ × [0,3] as an (s, z) grid: χ = 0, bounded (two boundary
/// circles), orientable. A control — not one of the four #2280 targets — that the
/// readout is documented to fold into `SheetOrDisk` (see the module note).
fn cylinder(ns: usize, nz: usize) -> Array2<f64> {
    let mut z = Array2::<f64>::zeros((ns * nz, 3));
    for i in 0..ns {
        let s = 2.0 * PI * i as f64 / ns as f64;
        for j in 0..nz {
            let zz = 3.0 * j as f64 / (nz - 1) as f64;
            let row = i * nz + j;
            z[[row, 0]] = s.cos();
            z[[row, 1]] = s.sin();
            z[[row, 2]] = zz;
        }
    }
    z
}

fn to_array(pts: &[[f64; 3]]) -> Array2<f64> {
    let mut z = Array2::<f64>::zeros((pts.len(), 3));
    for (r, p) in pts.iter().enumerate() {
        for c in 0..3 {
            z[[r, c]] = p[c];
        }
    }
    z
}

/// The four #2280 targets are each recognized correctly, and the cylinder control
/// folds into the documented `SheetOrDisk` bucket. This is the exact recognition
/// the numpy suite validated (sphere χ ≈ 2, torus χ ≈ 0, Möbius non-orientable,
/// swiss sheet bounded), lifted to the production Rust readout.
#[test]
fn atlas_readout_recognizes_the_four_2280_targets() {
    let sphere_r = atlas_topology_readout(sphere(900).view(), 2);
    assert_eq!(
        sphere_r.topology,
        AtlasTopology::Sphere,
        "sphere must read as a closed positively-curved surface (χ={:.2}, boundary={}, nonor={})",
        sphere_r.gb_chi, sphere_r.has_boundary, sphere_r.nonorientable
    );
    assert!(
        sphere_r.gb_chi >= 1.5 && !sphere_r.has_boundary && !sphere_r.nonorientable,
        "sphere invariants off: χ={:.2} boundary={} nonor={}",
        sphere_r.gb_chi, sphere_r.has_boundary, sphere_r.nonorientable
    );

    let torus_r = atlas_topology_readout(torus(40, 28).view(), 2);
    assert_eq!(
        torus_r.topology,
        AtlasTopology::Torus,
        "torus must read as a closed developable surface (χ={:.2}, boundary={}, nonor={})",
        torus_r.gb_chi, torus_r.has_boundary, torus_r.nonorientable
    );
    assert!(
        torus_r.gb_chi < 1.0 && !torus_r.has_boundary && !torus_r.nonorientable,
        "torus invariants off: χ={:.2} boundary={} nonor={}",
        torus_r.gb_chi, torus_r.has_boundary, torus_r.nonorientable
    );

    let mobius_r = atlas_topology_readout(mobius(90, 13).view(), 2);
    assert_eq!(
        mobius_r.topology,
        AtlasTopology::Mobius,
        "Möbius must read as non-orientable with a boundary (nonor={}, frustrated={}, boundary={})",
        mobius_r.nonorientable, mobius_r.frustrated, mobius_r.has_boundary
    );
    assert!(
        mobius_r.nonorientable && mobius_r.has_boundary,
        "Möbius invariants off: nonor={} boundary={}",
        mobius_r.nonorientable, mobius_r.has_boundary
    );

    let swiss_r = atlas_topology_readout(swiss(46, 12).view(), 2);
    assert_eq!(
        swiss_r.topology,
        AtlasTopology::SheetOrDisk,
        "swiss-roll sheet must read as an orientable bounded sheet (boundary={}, nonor={})",
        swiss_r.has_boundary, swiss_r.nonorientable
    );
    assert!(
        swiss_r.has_boundary && !swiss_r.nonorientable,
        "swiss invariants off: boundary={} nonor={}",
        swiss_r.has_boundary, swiss_r.nonorientable
    );
}

/// Non-target controls behave: a flat disk and an open cylinder are both orientable
/// bounded surfaces, so both fold into `SheetOrDisk` (the readout does not, and is
/// documented not to, separate disk from cylinder — that split has no robust local
/// invariant).
#[test]
fn atlas_readout_bounded_controls_fold_into_sheet() {
    for (name, z) in [("disk", disk(22, 40)), ("cylinder", cylinder(40, 26))] {
        let r = atlas_topology_readout(z.view(), 2);
        assert_eq!(
            r.topology,
            AtlasTopology::SheetOrDisk,
            "{name} must read as an orientable bounded sheet (boundary={}, nonor={}, χ={:.2})",
            r.has_boundary, r.nonorientable, r.gb_chi
        );
    }
}

/// Determinism doctrine: the readout is RNG-free, so the same cloud yields a
/// bit-identical verdict and identical invariants run-to-run.
#[test]
fn atlas_readout_is_deterministic() {
    let z = torus(36, 24);
    let a = atlas_topology_readout(z.view(), 2);
    let b = atlas_topology_readout(z.view(), 2);
    assert_eq!(a.topology, b.topology);
    assert_eq!(a.gb_chi.to_bits(), b.gb_chi.to_bits(), "χ must be bit-identical");
    assert_eq!(a.frac_boundary.to_bits(), b.frac_boundary.to_bits());
    assert_eq!(a.frustrated, b.frustrated);
    assert_eq!(a.nonorientable, b.nonorientable);
}

/// A degenerate cloud (too few observations for a cover) is reported
/// `Indeterminate` rather than mis-recognized.
#[test]
fn atlas_readout_too_small_is_indeterminate() {
    let z = sphere(4);
    let r = atlas_topology_readout(z.view(), 2);
    assert_eq!(r.topology, AtlasTopology::Indeterminate);
}
