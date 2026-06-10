//! #998 — the full-resolution certificate: exact gauge orbits in the model's
//! own (decoder, coordinate) parameter space.
//!
//! Three planted facts, each the acceptance criterion it is named after:
//!
//! 1. **Compensated orbits are exact data-nulls** for bases closed under the
//!    group action: the harmonic circle's U(1) phase shift must come back
//!    unpinned with its compensation residual at the numerical noise floor —
//!    no lowering-error calibration involved. And **closure is computed, not
//!    declared**: the same machinery on a flat patch must certify the so(2)
//!    rotation exactly for a linear basis (closed) while genuinely pinning it
//!    for a quadratic basis (not closed) — the data honestly pins what the
//!    model class is honestly not symmetric under.
//! 2. **The penalty channel inverts the #995 falsifier**: with exact-orbit
//!    realisation the verdict on a true model-class symmetry must come from
//!    the penalty root alone — installing an [`OrbitPenaltyOperator`] that
//!    costs the orbit pins it; removing the operator flips it unpinned while
//!    the data rows stay present throughout (they are a null either way).
//! 3. **Merging**: exact within-atom verdicts replace the frame-path ones for
//!    viewed atoms (no double reporting), while unviewed atoms keep the
//!    calibrated frame path.

use gam::inference::row_metric::RowMetric;
use gam::sae_identifiability::{
    AtomParameterView, AtomTopology, FittedAtom, FittedSaeManifold, GENERATOR_FLAT_ENERGY_TOL,
    GeneratorFamily, OrbitPenaltyOperator, residual_gauge_exact,
};
use ndarray::{Array1, Array2, Array3};

const N: usize = 48;

/// Harmonic circle view: Φ = [1, cos 2πt, sin 2πt], B fixed full-rank
/// (M = 3, p = 4), coordinates spread over the circle, unit activations.
/// The basis is closed under phase shifts: Φ(t+ε) = Φ(t)·R(ε).
fn circle_view() -> (FittedAtom, AtomParameterView) {
    let m = 3usize;
    let p = 4usize;
    let mut basis_values = Array2::<f64>::zeros((N, m));
    let mut basis_jacobian = Array3::<f64>::zeros((N, m, 1));
    let mut coords = Array2::<f64>::zeros((N, 1));
    for i in 0..N {
        let t = (i as f64 + 0.31) / N as f64;
        let ang = std::f64::consts::TAU * t;
        coords[[i, 0]] = t;
        basis_values[[i, 0]] = 1.0;
        basis_values[[i, 1]] = ang.cos();
        basis_values[[i, 2]] = ang.sin();
        basis_jacobian[[i, 1, 0]] = -std::f64::consts::TAU * ang.sin();
        basis_jacobian[[i, 2, 0]] = std::f64::consts::TAU * ang.cos();
    }
    let mut decoder = Array2::<f64>::zeros((m, p));
    for r in 0..m {
        for c in 0..p {
            decoder[[r, c]] = ((r * p + c) as f64 * 0.37 + 0.4).sin() + 0.1 * (r as f64 + 1.0);
        }
    }
    let atom = FittedAtom {
        name: "circle".to_string(),
        topology: AtomTopology::Circle,
        // The frame is irrelevant to the exact path; give the mean tangent's
        // shape so the frame-path families (none here: p ≥ 2 output rotations
        // are enumerated but harmless) stay well-formed.
        frame: Array2::<f64>::zeros((p, 1)),
        ard_variances: None,
        lowering_error: 0.9, // deliberately lossy — the exact path must ignore it
    };
    let view = AtomParameterView {
        basis_values,
        basis_jacobian,
        decoder,
        coords,
        activations: Array1::<f64>::ones(N),
    };
    (atom, view)
}

/// Flat-patch view over a 2-D grid with basis either LINEAR [t1, t2]
/// (closed under so(2)) or QUADRATIC [t1, t2, t1²] (not closed: rotating the
/// coordinates produces a t1·t2 term outside the span).
fn patch_view(quadratic: bool) -> (FittedAtom, AtomParameterView) {
    let m = if quadratic { 3 } else { 2 };
    let p = 3usize;
    let side = 7usize;
    let n = side * side;
    let mut basis_values = Array2::<f64>::zeros((n, m));
    let mut basis_jacobian = Array3::<f64>::zeros((n, m, 2));
    let mut coords = Array2::<f64>::zeros((n, 2));
    for gx in 0..side {
        for gy in 0..side {
            let row = gx * side + gy;
            let t1 = (gx as f64 - 3.0) / 3.0;
            let t2 = (gy as f64 - 3.0) / 3.0;
            coords[[row, 0]] = t1;
            coords[[row, 1]] = t2;
            basis_values[[row, 0]] = t1;
            basis_values[[row, 1]] = t2;
            basis_jacobian[[row, 0, 0]] = 1.0;
            basis_jacobian[[row, 1, 1]] = 1.0;
            if quadratic {
                basis_values[[row, 2]] = t1 * t1;
                basis_jacobian[[row, 2, 0]] = 2.0 * t1;
            }
        }
    }
    let mut decoder = Array2::<f64>::zeros((m, p));
    for r in 0..m {
        for c in 0..p {
            decoder[[r, c]] = if r == c { 1.0 } else { 0.2 * (r as f64 - c as f64) };
        }
    }
    let atom = FittedAtom {
        name: if quadratic { "quad-patch" } else { "lin-patch" }.to_string(),
        topology: AtomTopology::EuclideanPatch { latent_dim: 2 },
        frame: Array2::<f64>::zeros((p, 2)),
        ard_variances: None,
        lowering_error: 0.0,
    };
    let view = AtomParameterView {
        basis_values,
        basis_jacobian,
        decoder,
        coords,
        activations: Array1::<f64>::ones(n),
    };
    (atom, view)
}

fn single_atom_model(atom: FittedAtom) -> FittedSaeManifold {
    let param_dim = atom.frame.len();
    FittedSaeManifold {
        atoms: vec![atom],
        jacobian_rows: Vec::new(),
        isometry_penalty_root: Array2::<f64>::zeros((0, param_dim)),
        metric: RowMetric::euclidean(1, 1).expect("euclidean metric"),
    }
}

fn exact_isom_verdict(
    report: &gam::sae_identifiability::ResidualGaugeReport,
) -> &gam::sae_identifiability::GeneratorVerdict {
    report
        .generators
        .iter()
        .find(|g| g.family == GeneratorFamily::IsomAtom && g.description.contains("exact orbit"))
        .expect("an exact-orbit Isom generator must be present")
}

#[test]
fn closed_basis_circle_phase_shift_is_an_exact_data_null() {
    let (atom, view) = circle_view();
    let model = single_atom_model(atom);
    let report = residual_gauge_exact(&model, &[Some(view)], &[None]).expect("certificate");

    let phase = exact_isom_verdict(&report);
    assert!(
        phase.generator_norm > 0.0,
        "the phase orbit moves the fit before compensation"
    );
    assert!(
        phase.pinned_energy_fraction <= 1.0e-9,
        "harmonics are closed under shifts: the compensation residual must be \
         numerically zero, got {}",
        phase.pinned_energy_fraction
    );
    assert_eq!(
        phase.lowering_error_scale, 0.0,
        "the exact path involves no lowering-error calibration even though \
         the atom's frame compression is deliberately lossy (0.9)"
    );
    assert!(phase.unpinned, "{}", report.summary);
    // No frame-path Isom verdict may coexist for the viewed atom.
    let frame_isom = report
        .generators
        .iter()
        .filter(|g| {
            g.family == GeneratorFamily::IsomAtom && !g.description.contains("exact orbit")
        })
        .count();
    assert_eq!(
        frame_isom, 0,
        "the lossy frame-space lift must be replaced, not double-reported"
    );
}

#[test]
fn basis_closure_is_computed_not_declared() {
    // Linear patch basis: closed under so(2) ⇒ exact null ⇒ unpinned.
    let (atom_lin, view_lin) = patch_view(false);
    let report_lin = residual_gauge_exact(&single_atom_model(atom_lin), &[Some(view_lin)], &[None])
        .expect("linear-basis certificate");
    let rot_lin = exact_isom_verdict(&report_lin);
    assert!(
        rot_lin.pinned_energy_fraction <= 1.0e-9 && rot_lin.unpinned,
        "a linear chart basis is closed under rotation — exact null expected, \
         got fraction {}",
        rot_lin.pinned_energy_fraction
    );

    // Quadratic basis: rotating the chart produces a t1·t2 term outside the
    // span ⇒ the model class is genuinely NOT symmetric ⇒ the data pins the
    // rotation, honestly, with a substantial computed residual.
    let (atom_quad, view_quad) = patch_view(true);
    let report_quad =
        residual_gauge_exact(&single_atom_model(atom_quad), &[Some(view_quad)], &[None])
            .expect("quadratic-basis certificate");
    let rot_quad = exact_isom_verdict(&report_quad);
    assert!(
        !rot_quad.unpinned,
        "a basis NOT closed under the action means the rotation is not a \
         model-class symmetry — the data must pin it. {}",
        report_quad.summary
    );
    assert!(
        rot_quad.pinned_energy_fraction > 10.0 * GENERATOR_FLAT_ENERGY_TOL,
        "the computed closure residual must be substantial, got {}",
        rot_quad.pinned_energy_fraction
    );
}

#[test]
fn penalty_channel_alone_decides_true_symmetries() {
    // The #995 falsifier, inverted: the circle's phase shift is a data-null
    // either way; whether it is a residual freedom is decided ENTIRELY by the
    // penalty channel.
    let (atom, view) = circle_view();

    // Without a pin: unpinned (previous test). With a pin operator that costs
    // net phase drift (what an anchor/registration pin does), the SAME orbit
    // must come back pinned — fraction 1 here, since the operator's whole
    // stiffness lies along the drift.
    let pin = OrbitPenaltyOperator {
        apply: Box::new(|_delta_b, delta_t| {
            let n = delta_t.nrows() as f64;
            let drift = delta_t.column(0).sum() / n;
            ndarray::Array1::from(vec![10.0 * drift])
        }),
        stiffness_sq: 100.0,
    };
    let model = single_atom_model(atom);
    let report_pinned =
        residual_gauge_exact(&model, &[Some(view.clone())], &[Some(pin)]).expect("certificate");
    let phase_pinned = exact_isom_verdict(&report_pinned);
    assert!(
        !phase_pinned.unpinned,
        "a pin costing phase drift must pin the shift orbit. {}",
        report_pinned.summary
    );
    assert!(
        (phase_pinned.pinned_energy_fraction - 1.0).abs() < 1.0e-9,
        "the operator's whole stiffness lies along the drift: fraction must \
         be 1, got {}",
        phase_pinned.pinned_energy_fraction
    );

    // Remove the operator: the verdict flips unpinned with the data rows
    // untouched — the inversion of the #995 falsifier.
    let report_free = residual_gauge_exact(&model, &[Some(view)], &[None]).expect("certificate");
    let phase_free = exact_isom_verdict(&report_free);
    assert!(
        phase_free.unpinned && phase_free.pinned_energy_fraction <= 1.0e-9,
        "without the pin the exact data-null must be a certified freedom. {}",
        report_free.summary
    );
    assert_ne!(
        report_pinned.group_signature(),
        report_free.group_signature(),
        "pin on vs off must certify different groups"
    );
}
