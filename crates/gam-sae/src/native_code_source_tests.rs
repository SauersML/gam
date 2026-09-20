#![cfg(test)]
//! #2933 F10/F11/F12: the native code sources are measured in the output metric,
//! over transmitted codes only, and are invariant to representations that
//! preserve the decoded model.

use ndarray::{Array1, Array2, ArrayView2, array};

use super::{
    ActiveCodeSource, NativeGateModel, native_active_code_sources, native_gate_amplitude_code,
};
use crate::atom_codes::SparseAtomCodes;
use crate::description_length::{
    DictionaryCode, ManifoldFitDl, NativeDescriptionLengthRequest,
    native_manifold_description_length,
};
use crate::manifold::{
    SaeAtomBasisKind, SaeAtomGeometryPlan, SaeBasisResolution, SaeReferenceMetricPlan,
    reconstruct_persisted_atom_set,
};

fn linear_plan(latent_dim: usize) -> SaeAtomGeometryPlan {
    SaeAtomGeometryPlan::new(
        SaeAtomBasisKind::Linear,
        latent_dim,
        SaeBasisResolution::Polynomial { degree: 1 },
        SaeReferenceMetricPlan::EuclideanPolynomial,
    )
    .expect("linear plan")
}

fn periodic_plan() -> SaeAtomGeometryPlan {
    SaeAtomGeometryPlan::new(
        SaeAtomBasisKind::Periodic,
        1,
        SaeBasisResolution::PeriodicHarmonics { order: 2 },
        SaeReferenceMetricPlan::UnitCircle,
    )
    .expect("periodic plan")
}

fn sphere_plan() -> SaeAtomGeometryPlan {
    SaeAtomGeometryPlan::new(
        SaeAtomBasisKind::Sphere,
        3,
        SaeBasisResolution::AmbientSphereHarmonics { degree: 2 },
        SaeReferenceMetricPlan::RoundSphere,
    )
    .expect("sphere plan")
}

fn mobius_plan() -> SaeAtomGeometryPlan {
    SaeAtomGeometryPlan::new(
        SaeAtomBasisKind::Mobius,
        2,
        SaeBasisResolution::MobiusHarmonics {
            circle_order: 2,
            width_degree: 1,
        },
        SaeReferenceMetricPlan::MobiusQuotient,
    )
    .expect("mobius plan")
}

/// A deterministic dense decoder aligned with no axis.
fn dense_decoder(plan: &SaeAtomGeometryPlan, p_out: usize, seed: f64) -> Array2<f64> {
    let width = plan.basis_size().expect("plan width");
    Array2::from_shape_fn((width, p_out), |(m, c)| {
        (0.731 * (m + 1) as f64 + 1.37 * (c + 1) as f64 + seed).sin()
    })
}

fn support(assignments: &Array2<f64>) -> SparseAtomCodes {
    let (n, k) = assignments.dim();
    let mut codes = SparseAtomCodes::empty(n, k);
    for row in 0..n {
        for atom in 0..k {
            let gate = assignments[[row, atom]];
            if gate != 0.0 {
                codes.row_mut(row).assign(atom, gate);
            }
        }
    }
    codes
}

fn views(blocks: &[Array2<f64>]) -> Vec<ArrayView2<'_, f64>> {
    blocks.iter().map(|block| block.view()).collect()
}

/// The native ledger with a declared zero-width dictionary, so every bit and all
/// of the distortion belong to the codes and the residual. The fixtures' gates
/// vary independently per atom and are not simplex rows.
fn try_describe(
    assignments: &Array2<f64>,
    plans: &[SaeAtomGeometryPlan],
    decoders: &[Array2<f64>],
    coords: &[Array2<f64>],
    tier0_scale: Option<&Array1<f64>>,
    target: &Array2<f64>,
    fitted: &Array2<f64>,
) -> Result<ManifoldFitDl, String> {
    let decoder_views = views(decoders);
    let coord_views = views(coords);
    let dictionary = DictionaryCode::DeclaredPrecision {
        n_params: 0,
        bits_per_scalar: 0.0,
    };
    native_manifold_description_length(NativeDescriptionLengthRequest {
        assignments: assignments.view(),
        gate_model: NativeGateModel::Independent,
        geometry_plans: plans,
        decoder_blocks: &decoder_views,
        coords: &coord_views,
        tier0_scale: tier0_scale.map(|scale| scale.view()),
        target: target.view(),
        fitted: fitted.view(),
        dictionary: &dictionary,
    })
}

/// The ledger of the fit whose reconstruction is the decoded model and whose
/// target is that reconstruction plus `residual`.
fn describe(
    assignments: &Array2<f64>,
    plans: &[SaeAtomGeometryPlan],
    decoders: &[Array2<f64>],
    coords: &[Array2<f64>],
    tier0_scale: Option<&Array1<f64>>,
    residual: &Array2<f64>,
) -> ManifoldFitDl {
    let fitted = decoded(plans, decoders, coords, assignments);
    let target = &fitted + residual;
    try_describe(assignments, plans, decoders, coords, tier0_scale, &target, &fitted)
        .expect("native description length")
}

/// A deterministic dense `(n, p)` residual aligned with no channel.
fn planted_residual(n: usize, p: usize) -> Array2<f64> {
    Array2::from_shape_fn((n, p), |(i, c)| {
        0.3 * (2.3 * (i + 1) as f64 + 1.1 * (c + 1) as f64).sin()
    })
}

fn code_sources(
    assignments: &Array2<f64>,
    plans: &[SaeAtomGeometryPlan],
    decoders: &[Array2<f64>],
    coords: &[Array2<f64>],
    tier0_scale: Option<&Array1<f64>>,
) -> Result<Vec<ActiveCodeSource>, String> {
    native_active_code_sources(
        &support(assignments),
        plans,
        &views(decoders),
        &views(coords),
        tier0_scale.map(|scale| scale.view()),
    )
}

fn decoded(
    plans: &[SaeAtomGeometryPlan],
    decoders: &[Array2<f64>],
    coords: &[Array2<f64>],
    assignments: &Array2<f64>,
) -> Array2<f64> {
    reconstruct_persisted_atom_set(
        plans,
        &views(decoders),
        &views(coords),
        assignments.view(),
        decoders[0].ncols(),
    )
    .expect("decoded model")
}

fn max_abs_gap(left: &Array2<f64>, right: &Array2<f64>) -> f64 {
    left.iter()
        .zip(right.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0, f64::max)
}

fn relative_gap(left: f64, right: f64) -> f64 {
    (left - right).abs() / left.abs().max(right.abs())
}

fn assert_spectra_match(left: &[ActiveCodeSource], right: &[ActiveCodeSource], what: &str) {
    assert_eq!(left.len(), right.len(), "{what}: source count");
    for (atom, (a, b)) in left.iter().zip(right).enumerate() {
        assert_eq!(a.output_spectrum.len(), b.output_spectrum.len(), "{what}: atom {atom}");
        for (x, y) in a.output_spectrum.iter().zip(&b.output_spectrum) {
            assert!(
                (x - y).abs() <= 1.0e-10 * x.abs().max(y.abs()),
                "{what}: atom {atom} output spectrum {:?} vs {:?}",
                a.output_spectrum,
                b.output_spectrum
            );
        }
    }
}

#[test]
fn native_code_rate_is_the_output_metric_gaussian_rate_2933_f11() {
    // Audit check 3. Independent latent coordinates with variances in the ratio
    // (100, 1), decoded by diag(0.1, 1), so the decoded output covariance is
    // isotropic with spectrum (1, 1). The fit leaves a residual of energy 0.3 in the
    // channel the atom does not decode. Joint water filling of the code (1, 1) and
    // the residual (0.3) to the delivered distortion 0.3 sets the water level
    // 0.3 / 3 = 0.1, so the Gaussian output rate of the code is log2(10) bits per
    // token and the residual costs ½·log2(3). Water filling the latent spectrum
    // instead spends the budget on the insensitive coordinate.
    let coords = array![[10.0, 1.0], [-10.0, 1.0], [10.0, -1.0], [-10.0, -1.0]];
    let decoder = array![[0.0, 0.0, 0.0], [0.0, 0.1, 0.0], [0.0, 0.0, 1.0]];
    let assignments = Array2::from_elem((4, 1), 1.0);
    let amplitude = 0.3_f64.sqrt();
    let residual = Array2::from_shape_fn((4, 3), |(i, c)| {
        if c == 0 { amplitude * [1.0, -1.0, -1.0, 1.0][i] } else { 0.0 }
    });
    let dl = describe(
        &assignments,
        &[linear_plan(2)],
        &[decoder],
        &[coords],
        None,
        &residual,
    );
    let expected = 10.0_f64.log2();
    assert!(
        (dl.code_bits_per_token - expected).abs() < 1.0e-12,
        "output-metric Gaussian rate at distortion 0.3 is {expected}; got {}",
        dl.code_bits_per_token
    );
    assert!((dl.residual_bits_per_token - 0.5 * 3.0_f64.log2()).abs() < 1.0e-12);
    assert!((dl.distortion - 0.3).abs() < 1.0e-12, "distortion {}", dl.distortion);
    // TSS: the decoded channels have unit variance each, the residual channel 0.3.
    assert!((dl.ev - (1.0 - 0.3 / 2.3)).abs() < 1.0e-12, "ev {}", dl.ev);
}

#[test]
fn native_code_rate_is_invariant_to_compensated_coordinate_rescaling_2933_f11() {
    // A 2-D linear atom with correlated coordinates co-fires with a periodic atom
    // under varying gates and a standardized output metric. Rescaling and shifting
    // the linear coordinates, compensated in the decoder, decodes identically, so
    // every source spectrum and the code ledger must be unchanged.
    let n = 12;
    let t = Array2::from_shape_fn((n, 2), |(i, axis)| {
        let x = i as f64;
        if axis == 0 {
            3.0 * (0.7 * x).sin() + 0.2 * x
        } else {
            (1.3 * x).cos() + 0.5 * (0.7 * x).sin()
        }
    });
    let phases = Array2::from_shape_fn((n, 1), |(i, _)| 0.37 * i as f64 + 0.05 * (i * i) as f64);
    let assignments = Array2::from_shape_fn((n, 2), |(i, atom)| {
        0.3 + 0.6 * (0.9 * (i + 3 * atom) as f64).sin().abs()
    });
    let plans = [linear_plan(2), periodic_plan()];
    let linear_decoder = dense_decoder(&plans[0], 3, 0.1);
    let periodic_decoder = dense_decoder(&plans[1], 3, 2.0);
    let tier0_scale = array![1.0, 2.5, 0.4];
    let scale = [4.0, 0.25];
    let shift = [1.5, -2.0];
    let rescaled = Array2::from_shape_fn((n, 2), |(i, axis)| scale[axis] * t[[i, axis]] + shift[axis]);
    // γ(t) = b₀ + Σ_a t_a b_a = b₀' + Σ_a t'_a b'_a with t' = s·t + c:
    // b'_a = b_a / s_a and b₀' = b₀ − Σ_a c_a b_a / s_a.
    let mut compensated = linear_decoder.clone();
    for axis in 0..2 {
        for channel in 0..3 {
            let slope = linear_decoder[[axis + 1, channel]] / scale[axis];
            compensated[[axis + 1, channel]] = slope;
            compensated[[0, channel]] -= shift[axis] * slope;
        }
    }
    let original_decoders = [linear_decoder, periodic_decoder.clone()];
    let rescaled_decoders = [compensated, periodic_decoder];
    let original_coords = [t, phases.clone()];
    let rescaled_coords = [rescaled, phases];
    let decoded_gap = max_abs_gap(
        &decoded(&plans, &original_decoders, &original_coords, &assignments),
        &decoded(&plans, &rescaled_decoders, &rescaled_coords, &assignments),
    );
    assert!(
        decoded_gap < 1.0e-12,
        "fixture: the compensated chart must decode identically ({decoded_gap})"
    );

    let original = code_sources(
        &assignments,
        &plans,
        &original_decoders,
        &original_coords,
        Some(&tier0_scale),
    )
    .expect("original sources");
    let moved = code_sources(
        &assignments,
        &plans,
        &rescaled_decoders,
        &rescaled_coords,
        Some(&tier0_scale),
    )
    .expect("rescaled sources");
    // Positive control: the latent covariance itself moves with the rescaling.
    assert!(
        relative_gap(
            original[0].code_covariance[[0, 0]],
            moved[0].code_covariance[[0, 0]]
        ) > 0.5,
        "control: the rescaling must change the latent covariance"
    );
    assert_spectra_match(&original, &moved, "compensated rescaling");

    let base = describe(
        &assignments,
        &plans,
        &original_decoders,
        &original_coords,
        Some(&tier0_scale),
        &planted_residual(n, 3),
    );
    let other = describe(
        &assignments,
        &plans,
        &rescaled_decoders,
        &rescaled_coords,
        Some(&tier0_scale),
        &planted_residual(n, 3),
    );
    assert!(
        relative_gap(base.code_bits_per_token, other.code_bits_per_token) < 1.0e-10,
        "compensated rescaling changed the code ledger: {} vs {}",
        base.code_bits_per_token,
        other.code_bits_per_token
    );
}

#[test]
fn native_code_rate_is_invariant_to_shifted_angle_charts_2933_f11() {
    // A periodic atom whose phases straddle the chart's branch cut 0 ≡ 1, several
    // stored off by whole periods, co-fires with a line atom so the water-filling
    // allocation depends on the periodic spread. Shifting the angle chart and
    // rotating each harmonic pair of the decoder to compensate decodes identically.
    let phases = [0.93, 0.97, 0.01, 0.04, 1.08, -0.05, 0.02, 0.9, 0.12, 2.99];
    let n = phases.len();
    let angles = Array2::from_shape_fn((n, 1), |(i, _)| phases[i]);
    let offset = 0.43;
    let shifted = angles.mapv(|value| value + offset);
    let line = Array2::from_shape_fn((n, 1), |(i, _)| (1.7 * i as f64).sin());
    let assignments = Array2::from_shape_fn((n, 2), |(i, atom)| {
        if atom == 0 {
            0.8 + 0.1 * (i as f64).cos()
        } else {
            0.5
        }
    });
    let plans = [periodic_plan(), linear_plan(1)];
    let periodic_decoder = dense_decoder(&plans[0], 4, 0.3);
    // Columns are [1, sin 2πt, cos 2πt, sin 4πt, cos 4πt]. With t' = t + c,
    // sin(2πh(t' − c)) = sin(2πht')·C − cos(2πht')·S and
    // cos(2πh(t' − c)) = cos(2πht')·C + sin(2πht')·S.
    let mut rotated = periodic_decoder.clone();
    for h in 1..=2 {
        let (sine, cosine) = (2.0 * std::f64::consts::PI * h as f64 * offset).sin_cos();
        for channel in 0..4 {
            let s = periodic_decoder[[2 * h - 1, channel]];
            let c = periodic_decoder[[2 * h, channel]];
            rotated[[2 * h - 1, channel]] = cosine * s + sine * c;
            rotated[[2 * h, channel]] = -sine * s + cosine * c;
        }
    }
    let line_decoder = dense_decoder(&plans[1], 4, 1.1);
    let original_decoders = [periodic_decoder, line_decoder.clone()];
    let shifted_decoders = [rotated, line_decoder];
    let original_coords = [angles, line.clone()];
    let shifted_coords = [shifted, line];
    let decoded_gap = max_abs_gap(
        &decoded(&plans, &original_decoders, &original_coords, &assignments),
        &decoded(&plans, &shifted_decoders, &shifted_coords, &assignments),
    );
    assert!(
        decoded_gap < 1.0e-12,
        "fixture: the shifted chart must decode identically ({decoded_gap})"
    );

    let original = code_sources(&assignments, &plans, &original_decoders, &original_coords, None)
        .expect("original sources");
    let moved = code_sources(&assignments, &plans, &shifted_decoders, &shifted_coords, None)
        .expect("shifted sources");
    // Positive control: the phases cluster within ±0.12 of the cut, so the coded
    // spread is small although the stored values span [−0.05, 2.99].
    assert!(
        original[0].code_covariance[[0, 0]] < 0.01,
        "the angular code must be measured about its cluster, not across the cut: {}",
        original[0].code_covariance[[0, 0]]
    );
    assert_spectra_match(&original, &moved, "shifted angle chart");

    let base = describe(&assignments, &plans, &original_decoders, &original_coords, None, &planted_residual(n, 4));
    let other = describe(&assignments, &plans, &shifted_decoders, &shifted_coords, None, &planted_residual(n, 4));
    assert!(
        relative_gap(base.code_bits_per_token, other.code_bits_per_token) < 1.0e-10,
        "a shifted angle chart changed the code ledger: {} vs {}",
        base.code_bits_per_token,
        other.code_bits_per_token
    );
}

#[test]
fn native_code_source_ignores_coordinates_of_non_firing_rows_2933_f12() {
    // Audit check 5. Atom 0 fires on rows 0..4 only; its stored coordinates on rows
    // 4..8 reach no decoded product and are never transmitted.
    let n = 8;
    let assignments = Array2::from_shape_fn((n, 2), |(i, atom)| match atom {
        0 if i < 4 => [1.0, 0.8, 0.6, 0.9][i],
        0 => 0.0,
        _ => 0.7,
    });
    let quiet = array![[0.0], [1.0], [0.5], [-0.3], [0.0], [0.0], [0.0], [0.0]];
    let mut loud = quiet.clone();
    for (row, value) in [(4, 1000.0), (5, -1000.0), (6, 3.5e7), (7, -0.25)] {
        loud[[row, 0]] = value;
    }
    let phases = Array2::from_shape_fn((n, 1), |(i, _)| 0.11 * i as f64 + 0.03 * (i * i) as f64);
    let plans = [linear_plan(1), periodic_plan()];
    let decoders = [
        dense_decoder(&plans[0], 3, 0.4),
        dense_decoder(&plans[1], 3, 1.9),
    ];
    let quiet_coords = [quiet.clone(), phases.clone()];
    let loud_coords = [loud, phases.clone()];
    assert_eq!(
        decoded(&plans, &decoders, &quiet_coords, &assignments),
        decoded(&plans, &decoders, &loud_coords, &assignments),
        "fixture: non-firing coordinates must not reach the decoded model"
    );
    let quiet_sources =
        code_sources(&assignments, &plans, &decoders, &quiet_coords, None).expect("quiet sources");
    let loud_sources =
        code_sources(&assignments, &plans, &decoders, &loud_coords, None).expect("loud sources");
    assert_eq!(quiet_sources, loud_sources);
    // A non-finite stored coordinate on a non-firing row is equally never read.
    let mut unset = quiet;
    unset[[6, 0]] = f64::NAN;
    let unset_sources = code_sources(&assignments, &plans, &decoders, &[unset, phases], None)
        .expect("a non-firing NaN is never read");
    assert_eq!(unset_sources, quiet_sources);

    let quiet_dl = describe(&assignments, &plans, &decoders, &quiet_coords, None, &planted_residual(n, 3));
    let loud_dl = describe(&assignments, &plans, &decoders, &loud_coords, None, &planted_residual(n, 3));
    assert_eq!(quiet_dl.code_bits.to_bits(), loud_dl.code_bits.to_bits());
}

#[test]
fn native_code_source_is_unchanged_by_inactive_atoms_and_rows_where_it_is_off_2933_f12() {
    let n = 6;
    let assignments = Array2::from_shape_fn((n, 2), |(i, atom)| match atom {
        0 if i < 5 => 0.4 + 0.1 * i as f64,
        0 => 0.0,
        _ => 0.6,
    });
    let line = Array2::from_shape_fn((n, 1), |(i, _)| (0.9 * i as f64).sin());
    let phases = Array2::from_shape_fn((n, 1), |(i, _)| 0.21 * i as f64);
    let plans = [linear_plan(1), periodic_plan()];
    let decoders = [
        dense_decoder(&plans[0], 3, 0.7),
        dense_decoder(&plans[1], 3, 2.3),
    ];
    let coords = [line.clone(), phases.clone()];
    let base = code_sources(&assignments, &plans, &decoders, &coords, None).expect("base sources");
    let base_dl = describe(&assignments, &plans, &decoders, &coords, None, &planted_residual(n, 3));

    // A wholly inactive atom with arbitrary coordinates changes no other source
    // and no code bit.
    let widened = Array2::from_shape_fn((n, 3), |(i, atom)| {
        if atom < 2 { assignments[[i, atom]] } else { 0.0 }
    });
    let wide_plans = [linear_plan(1), periodic_plan(), linear_plan(2)];
    let wide_decoders = [
        decoders[0].clone(),
        decoders[1].clone(),
        dense_decoder(&wide_plans[2], 3, 5.0),
    ];
    let wide_coords = [
        line.clone(),
        phases,
        Array2::from_shape_fn((n, 2), |(i, axis)| 100.0 * (i + axis) as f64),
    ];
    let wide = code_sources(&widened, &wide_plans, &wide_decoders, &wide_coords, None)
        .expect("widened sources");
    assert_eq!(&wide[..2], &base[..]);
    assert_eq!(wide[2].firing_rows, 0);
    assert!(wide[2].output_spectrum.iter().all(|&value| value == 0.0));
    let wide_dl = describe(&widened, &wide_plans, &wide_decoders, &wide_coords, None, &planted_residual(n, 3));
    assert_eq!(
        wide_dl.code_bits_per_token.to_bits(),
        base_dl.code_bits_per_token.to_bits()
    );

    // Rows appended where atom 0 is off leave its conditional code unchanged;
    // only its firing probability moves.
    let extra = 4;
    let longer = Array2::from_shape_fn((n + extra, 2), |(i, atom)| {
        if i < n {
            assignments[[i, atom]]
        } else if atom == 0 {
            0.0
        } else {
            0.6
        }
    });
    let longer_coords = [
        Array2::from_shape_fn((n + extra, 1), |(i, _)| {
            if i < n { line[[i, 0]] } else { -50.0 * i as f64 }
        }),
        Array2::from_shape_fn((n + extra, 1), |(i, _)| 0.21 * i as f64),
    ];
    let longer_sources = code_sources(&longer, &plans, &decoders, &longer_coords, None)
        .expect("longer sources");
    assert_eq!(longer_sources[0].code_covariance, base[0].code_covariance);
    assert_eq!(
        longer_sources[0].mean_pullback_metric,
        base[0].mean_pullback_metric
    );
    assert_eq!(longer_sources[0].output_spectrum, base[0].output_spectrum);
    assert_eq!(longer_sources[0].firing_rows, 5);
    assert_eq!(
        longer_sources[0].firing_probability,
        5.0 / (n + extra) as f64
    );
}

#[test]
fn native_description_length_validates_its_input_domain_2933_f44() {
    // The persisted-artifact entry keeps the F44 domain: finite gates, coordinates,
    // target and reconstruction of one shape, a target with variance, and at least
    // one token. A reconstruction worse than the mean is a valid negative EV, and an
    // exact reconstruction is an infinite continuous rate.
    let plans = [linear_plan(1)];
    let decoders = [dense_decoder(&plans[0], 2, 0.6)];
    let assignments = array![[1.0], [1.0], [0.0], [1.0]];
    let coords = [array![[0.0], [1.0], [2.0], [3.0]]];
    let fitted = decoded(&plans, &decoders, &coords, &assignments);
    let target = &fitted + &planted_residual(4, 2);
    let check = |target: &Array2<f64>, fitted: &Array2<f64>| {
        try_describe(&assignments, &plans, &decoders, &coords, None, target, fitted)
    };
    assert!(check(&target, &fitted).is_ok());
    let reversed = &target + &(&target - &fitted).mapv(|v| 5.0 * v);
    let poor = check(&reversed, &fitted).expect("a reconstruction worse than the mean is valid");
    assert!(poor.ev < 0.0, "negative EV expected, got {}", poor.ev);
    let exact = check(&fitted, &fitted).expect("an exact reconstruction");
    assert_eq!(exact.distortion, 0.0);
    assert!(
        exact.code_bits.is_infinite(),
        "zero distortion is an infinite continuous rate, got {}",
        exact.code_bits
    );
    let wrong_shape = Array2::from_shape_fn((4, 3), |(i, c)| (i * 3 + c) as f64);
    let error = check(&wrong_shape, &wrong_shape).expect_err("decoders have two channels");
    assert!(error.contains("output channels"), "{error}");
    assert!(check(&target, &wrong_shape).is_err(), "target and fitted must share a shape");
    let constant = Array2::from_elem((4, 2), 1.5);
    let error = check(&constant, &fitted).expect_err("a constant target has no EV");
    assert!(error.contains("no variance"), "{error}");
    for bad in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        let mut broken = target.clone();
        broken[[2, 1]] = bad;
        let error = check(&broken, &fitted).expect_err("a nonfinite target must be rejected");
        assert!(error.contains("target[2, 1]"), "{error}");
        let gates = array![[1.0], [bad], [0.0], [1.0]];
        let error = try_describe(&gates, &plans, &decoders, &coords, None, &target, &fitted)
            .expect_err("a nonfinite gate must be rejected");
        assert!(error.contains("assignments[1, 0]"), "{error}");
        let stored = [array![[0.0], [1.0], [bad], [3.0]]];
        let error = try_describe(&assignments, &plans, &decoders, &stored, None, &target, &fitted)
            .expect_err("a nonfinite coordinate must be rejected");
        assert!(error.contains("coords[0][2, 0]"), "{error}");
    }
    let no_rows = Array2::<f64>::zeros((0, 1));
    let no_output = Array2::<f64>::zeros((0, 2));
    assert!(
        try_describe(&no_rows, &plans, &decoders, &[no_rows.clone()], None, &no_output, &no_output)
            .is_err(),
        "a report needs at least one token"
    );
}

#[test]
fn native_code_source_refuses_too_few_firings_instead_of_a_zero_spectrum_2933_f12() {
    // A 2-D code transmitted on two rows has at most a rank-one sample covariance:
    // too few firings to estimate the source, which must be reported, never priced
    // as a zero spectrum.
    let plans = [linear_plan(2)];
    let decoders = [dense_decoder(&plans[0], 3, 0.2)];
    let coords = [array![[1.0, 2.0], [-0.5, 0.25], [0.3, -1.1], [2.0, 0.0], [0.7, 0.7]]];
    let two = array![[1.0], [1.0], [0.0], [0.0], [0.0]];
    let refusal = code_sources(&two, &plans, &decoders, &coords, None)
        .expect_err("two firings of a 2-D code cannot give its covariance");
    assert!(refusal.contains("unavailable"), "{refusal}");
    let three = array![[1.0], [1.0], [1.0], [0.0], [0.0]];
    let sources = code_sources(&three, &plans, &decoders, &coords, None).expect("three firings");
    assert_eq!(sources[0].output_spectrum.len(), 2);
    assert!(sources[0].output_spectrum[1] > 0.0);
}

/// Descending eigenvalues of a symmetric 1×1 or 2×2 matrix.
fn small_eigenvalues(matrix: &Array2<f64>) -> Vec<f64> {
    if matrix.nrows() == 1 {
        return vec![matrix[[0, 0]]];
    }
    assert_eq!(matrix.dim(), (2, 2));
    let mean = 0.5 * (matrix[[0, 0]] + matrix[[1, 1]]);
    let radius = (0.25 * (matrix[[0, 0]] - matrix[[1, 1]]).powi(2) + matrix[[0, 1]].powi(2)).sqrt();
    vec![mean + radius, mean - radius]
}

/// `mean_i hᵀ G_i h` read off the decoded model: central second differences of the
/// squared output change in the metric `diag(σ⁻²)`, averaged over the rows.
/// `move_code(τ)` returns the coordinates with every row's code moved by `τ·h`.
fn decoded_quadratic_form(
    plan: &SaeAtomGeometryPlan,
    decoder: &Array2<f64>,
    gates: &Array2<f64>,
    tier0_scale: &Array1<f64>,
    move_code: &dyn Fn(f64) -> Array2<f64>,
) -> f64 {
    let epsilon = 1.0e-4;
    let plans = [plan.clone()];
    let decoders = [decoder.clone()];
    let at = |coords: Array2<f64>| decoded(&plans, &decoders, &[coords], gates);
    let center = at(move_code(0.0));
    let plus = at(move_code(epsilon));
    let minus = at(move_code(-epsilon));
    let mut total = 0.0;
    for row in 0..gates.nrows() {
        for channel in 0..decoder.ncols() {
            let weight = tier0_scale[channel].powi(-2);
            total += weight
                * ((plus[[row, channel]] - center[[row, channel]]).powi(2)
                    + (minus[[row, channel]] - center[[row, channel]]).powi(2));
        }
    }
    total / (2.0 * epsilon * epsilon * gates.nrows() as f64)
}

/// The decoded metric in a code basis, by polarization of the quadratic form.
fn decoded_metric(dim: usize, form: &dyn Fn(&[f64]) -> f64) -> Array2<f64> {
    Array2::from_shape_fn((dim, dim), |(i, j)| {
        let mut sum = vec![0.0; dim];
        let mut difference = vec![0.0; dim];
        sum[i] += 1.0;
        sum[j] += 1.0;
        difference[i] += 1.0;
        difference[j] -= 1.0;
        if i == j {
            form(&sum) / 4.0
        } else {
            (form(&sum) - form(&difference)) / 4.0
        }
    })
}

fn assert_metric_matches(source: &ActiveCodeSource, decoded: &Array2<f64>, what: &str) {
    let predicted = small_eigenvalues(&source.mean_pullback_metric);
    let observed = small_eigenvalues(decoded);
    for (p, o) in predicted.iter().zip(&observed) {
        assert!(
            (p - o).abs() <= 1.0e-6 * p.abs().max(o.abs()),
            "{what}: pullback metric eigenvalues {predicted:?}, decoded {observed:?}"
        );
    }
}

#[test]
fn native_pullback_metric_matches_decoded_perturbations_2933_f11() {
    // The mean pullback metric predicts the decoded squared output change of a
    // small code perturbation, with gates and the standardized output metric.
    // Check it against the actual decoded model for a flat chart, a periodic chart
    // and the ambient sphere coded on its tangent plane.
    let n = 7;
    let gates = Array2::from_shape_fn((n, 1), |(i, _)| 0.5 + 0.4 * (1.1 * i as f64).sin());
    let tier0_scale = array![0.8, 1.9, 0.3];

    let linear = linear_plan(2);
    let linear_decoder = dense_decoder(&linear, 3, 0.9);
    let flat = Array2::from_shape_fn((n, 2), |(i, axis)| (0.6 * i as f64 + axis as f64).cos());
    let source = code_sources(
        &gates,
        std::slice::from_ref(&linear),
        std::slice::from_ref(&linear_decoder),
        std::slice::from_ref(&flat),
        Some(&tier0_scale),
    )
    .expect("linear source");
    let metric = decoded_metric(2, &|h| {
        decoded_quadratic_form(&linear, &linear_decoder, &gates, &tier0_scale, &|tau| {
            Array2::from_shape_fn((n, 2), |(i, axis)| flat[[i, axis]] + tau * h[axis])
        })
    });
    assert_metric_matches(&source[0], &metric, "linear chart");

    let periodic = periodic_plan();
    let periodic_decoder = dense_decoder(&periodic, 3, 1.4);
    let angles = array![[0.95], [0.02], [0.31], [1.44], [-0.2], [0.66], [0.81]];
    let source = code_sources(
        &gates,
        std::slice::from_ref(&periodic),
        std::slice::from_ref(&periodic_decoder),
        std::slice::from_ref(&angles),
        Some(&tier0_scale),
    )
    .expect("periodic source");
    let metric = decoded_metric(1, &|h| {
        decoded_quadratic_form(&periodic, &periodic_decoder, &gates, &tier0_scale, &|tau| {
            angles.mapv(|value| value + tau * h[0])
        })
    });
    assert_metric_matches(&source[0], &metric, "periodic chart");

    let sphere = sphere_plan();
    let sphere_decoder = dense_decoder(&sphere, 3, 0.25);
    let units: Vec<[f64; 3]> = (0..n)
        .map(|i| {
            let x = i as f64;
            let raw = [
                1.0 + 0.3 * (1.3 * x).sin(),
                1.0 + 0.4 * (0.7 * x).cos(),
                1.0 - 0.35 * (2.1 * x).sin(),
            ];
            let norm = (raw[0] * raw[0] + raw[1] * raw[1] + raw[2] * raw[2]).sqrt();
            [raw[0] / norm, raw[1] / norm, raw[2] / norm]
        })
        .collect();
    let points = Array2::from_shape_fn((n, 3), |(i, axis)| units[i][axis]);
    let source = code_sources(
        &gates,
        std::slice::from_ref(&sphere),
        std::slice::from_ref(&sphere_decoder),
        std::slice::from_ref(&points),
        Some(&tier0_scale),
    )
    .expect("sphere source");
    assert_eq!(source[0].code_dim, 2, "S² carries two intrinsic code coordinates");
    assert_eq!(source[0].output_spectrum.len(), 2);
    // This test's own tangent frame and log/exp maps at the extrinsic mean.
    let dot = |a: &[f64; 3], b: &[f64; 3]| a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
    let mut mean = [0.0; 3];
    for unit in &units {
        for axis in 0..3 {
            mean[axis] += unit[axis];
        }
    }
    let mean_norm = dot(&mean, &mean).sqrt();
    let mean = [mean[0] / mean_norm, mean[1] / mean_norm, mean[2] / mean_norm];
    let seed = [1.0, 0.0, 0.0];
    let along = dot(&seed, &mean);
    let first = [
        seed[0] - along * mean[0],
        seed[1] - along * mean[1],
        seed[2] - along * mean[2],
    ];
    let first_norm = dot(&first, &first).sqrt();
    let first = [first[0] / first_norm, first[1] / first_norm, first[2] / first_norm];
    let second = [
        mean[1] * first[2] - mean[2] * first[1],
        mean[2] * first[0] - mean[0] * first[2],
        mean[0] * first[1] - mean[1] * first[0],
    ];
    let logs: Vec<[f64; 2]> = units
        .iter()
        .map(|unit| {
            let cosine = dot(&mean, unit);
            let radial = [
                unit[0] - cosine * mean[0],
                unit[1] - cosine * mean[1],
                unit[2] - cosine * mean[2],
            ];
            let sine = dot(&radial, &radial).sqrt();
            let scale = sine.atan2(cosine) / sine;
            [scale * dot(&radial, &first), scale * dot(&radial, &second)]
        })
        .collect();
    let exp = |code: [f64; 2]| -> [f64; 3] {
        let tangent = [
            code[0] * first[0] + code[1] * second[0],
            code[0] * first[1] + code[1] * second[1],
            code[0] * first[2] + code[1] * second[2],
        ];
        let angle = dot(&tangent, &tangent).sqrt();
        let (sine, cosine) = angle.sin_cos();
        [
            cosine * mean[0] + sine * tangent[0] / angle,
            cosine * mean[1] + sine * tangent[1] / angle,
            cosine * mean[2] + sine * tangent[2] / angle,
        ]
    };
    let metric = decoded_metric(2, &|h| {
        decoded_quadratic_form(&sphere, &sphere_decoder, &gates, &tier0_scale, &|tau| {
            Array2::from_shape_fn((n, 3), |(i, axis)| {
                exp([logs[i][0] + tau * h[0], logs[i][1] + tau * h[1]])[axis]
            })
        })
    });
    assert_metric_matches(&source[0], &metric, "ambient sphere tangent chart");
}

#[test]
fn native_code_source_is_invariant_to_deck_representatives_2933_f11() {
    // Moving rows to their deck twin preserves the decoded model of a quotient
    // atom, so its code source must not change: the ambient RP² (u ~ −u), the
    // Klein bottle ((θ, φ) ~ (θ + ½, −φ)) and the Möbius band ((s, w) ~ (s + 1, −w)).
    let n = 9;
    let twins = [1usize, 2, 5, 8];
    let gates = Array2::from_shape_fn((n, 1), |(i, _)| 0.6 + 0.3 * (0.8 * i as f64).cos());

    let check = |plan: SaeAtomGeometryPlan, original: Array2<f64>, twinned: Array2<f64>, what: &str| {
        let decoder = dense_decoder(&plan, 3, 0.55);
        let plans = [plan];
        let decoders = [decoder];
        let gap = max_abs_gap(
            &decoded(&plans, &decoders, std::slice::from_ref(&original), &gates),
            &decoded(&plans, &decoders, std::slice::from_ref(&twinned), &gates),
        );
        assert!(gap < 1.0e-12, "fixture {what}: deck twins must decode identically ({gap})");
        let left = code_sources(&gates, &plans, &decoders, &[original], None).expect(what);
        let right = code_sources(&gates, &plans, &decoders, &[twinned], None).expect(what);
        assert_spectra_match(&left, &right, what);
        assert!(
            max_abs_gap(&left[0].code_covariance, &right[0].code_covariance) < 1.0e-12,
            "{what}: code covariance {:?} vs {:?}",
            left[0].code_covariance,
            right[0].code_covariance
        );
    };

    let rp2 = Array2::from_shape_fn((n, 3), |(i, axis)| {
        let x = i as f64;
        let raw = [
            0.2 + 0.3 * (1.3 * x).sin(),
            1.0 + 0.4 * (0.7 * x).cos(),
            0.5 - 0.35 * (2.1 * x).sin(),
        ];
        raw[axis] / (raw[0] * raw[0] + raw[1] * raw[1] + raw[2] * raw[2]).sqrt()
    });
    let mut rp2_twinned = rp2.clone();
    for &row in &twins {
        for axis in 0..3 {
            rp2_twinned[[row, axis]] = -rp2[[row, axis]];
        }
    }
    check(
        SaeAtomGeometryPlan::projective_plane(1).expect("RP² plan"),
        rp2,
        rp2_twinned,
        "ambient RP²",
    );

    let klein = Array2::from_shape_fn((n, 2), |(i, axis)| {
        let x = i as f64;
        if axis == 0 { 0.13 * x + 0.02 * x * x } else { 0.3 * (1.7 * x).sin() + 0.1 * x }
    });
    let mut klein_twinned = klein.clone();
    for &row in &twins {
        klein_twinned[[row, 0]] = klein[[row, 0]] + 0.5;
        klein_twinned[[row, 1]] = -klein[[row, 1]];
    }
    check(
        SaeAtomGeometryPlan::klein_bottle(2).expect("Klein plan"),
        klein,
        klein_twinned,
        "Klein bottle",
    );

    let mobius = Array2::from_shape_fn((n, 2), |(i, axis)| {
        let x = i as f64;
        if axis == 0 { 0.23 * x + 0.01 * x * x } else { 0.8 * (0.9 * x).sin() }
    });
    let mut mobius_twinned = mobius.clone();
    for &row in &twins {
        mobius_twinned[[row, 0]] = mobius[[row, 0]] + 1.0;
        mobius_twinned[[row, 1]] = -mobius[[row, 1]];
    }
    check(mobius_plan(), mobius, mobius_twinned, "Möbius band");
}

#[test]
fn native_description_length_charges_gate_amplitudes_on_a_fixed_support_2933_f10() {
    // Audit check 4. One atom fires on every row with a constant chart coordinate,
    // and its decoder keeps only the constant basis column with value 3, so the
    // decoded curve is the constant 3 and the output is 3·a. The support and the
    // coordinate are fixed; only the amplitudes change. For a = 0.1, 0.2, …, 0.9 the
    // output has unbiased variance 9 · 0.075 = 0.675 while the coordinate code is
    // free. The fit leaves a residual of raw energy 0.135, so joint water filling of
    // the amplitude (0.675) and the residual (0.135) to the delivered distortion
    // 0.135 sets the water level 0.0675: the amplitude costs
    // ½log₂(0.675 / 0.0675) = ½log₂10 bits per token and the residual
    // ½log₂(0.135 / 0.0675) = ½ bit. A constant gate on the same support transmits
    // nothing, and its residual, whose whole energy is the delivered distortion,
    // is free.
    let n = 9;
    let plans = [periodic_plan()];
    let width = plans[0].basis_size().expect("plan width");
    let mut decoder = Array2::<f64>::zeros((width, 1));
    decoder[[0, 0]] = 3.0;
    let decoders = [decoder];
    let coords = [Array2::from_elem((n, 1), 0.25)];
    let varying = Array2::from_shape_fn((n, 1), |(i, _)| 0.1 + 0.1 * i as f64);
    let constant = Array2::from_elem((n, 1), 0.5);
    let output = decoded(&plans, &decoders, &coords, &varying);
    let mean = output.sum() / n as f64;
    let output_variance =
        output.iter().map(|value| (value - mean) * (value - mean)).sum::<f64>() / (n - 1) as f64;
    assert!(
        (output_variance - 0.675).abs() < 1.0e-12,
        "fixture: the decoded output varies ({output_variance})"
    );

    let amplitude = 0.135_f64.sqrt();
    let residual = Array2::from_shape_fn((n, 1), |(i, _)| {
        if i % 2 == 0 { amplitude } else { -amplitude }
    });
    let varying_dl = describe(&varying, &plans, &decoders, &coords, None, &residual);
    let constant_dl = describe(&constant, &plans, &decoders, &coords, None, &residual);
    assert_eq!(
        varying_dl.atom_code_bits_per_token,
        vec![0.0],
        "the constant coordinate is free"
    );
    let expected = 0.5 * 10.0_f64.log2();
    assert!(
        (varying_dl.gate_amplitude_bits_per_token - expected).abs() < 1.0e-12,
        "the amplitude costs {expected} bits per token, got {}",
        varying_dl.gate_amplitude_bits_per_token
    );
    assert!((varying_dl.code_bits_per_token - expected).abs() < 1.0e-12);
    assert_eq!(constant_dl.code_bits_per_token, 0.0);
    assert_eq!(
        varying_dl.selection_bits, constant_dl.selection_bits,
        "the support is unchanged"
    );
    assert!((varying_dl.residual_bits_per_token - 0.5).abs() < 1.0e-12);
    assert!(constant_dl.residual_bits_per_token.abs() < 1.0e-12);
    assert!(
        (varying_dl.total_bits - constant_dl.total_bits - n as f64 * (expected + 0.5)).abs()
            < 1.0e-9,
        "{} vs {}",
        varying_dl.total_bits,
        constant_dl.total_bits
    );
}

#[test]
fn native_gate_amplitude_code_prices_the_amplitude_given_the_support_2933_f10() {
    // The receiver already knows the support, so a gate's amplitude information is
    // its spread over the rows where it fires, paid only on those rows. Atom 0 fires
    // with gate 0.75 on rows 0..4 of 8. Its on/off column has variance 0.140625 over
    // all rows, but that pattern is the support, so its amplitude is free. Atom 1
    // fires on rows 1, 4 and 6 with gates 0.2, 0.5 and 0.8, whose unbiased variance is
    // 0.09. Its decoded curve is the constant (2, −1) with ‖γ‖² = 5, so its source
    // is 5 · 0.09 = 0.45 with weight 3/8.
    let n = 8;
    let plans = [linear_plan(1), linear_plan(1)];
    let width = plans[1].basis_size().expect("plan width");
    let mut constant_curve = Array2::<f64>::zeros((width, 2));
    constant_curve[[0, 0]] = 2.0;
    constant_curve[[0, 1]] = -1.0;
    let decoders = [dense_decoder(&plans[0], 2, 0.4), constant_curve];
    let coords = [
        Array2::from_shape_fn((n, 1), |(i, _)| (0.8 * i as f64).sin()),
        Array2::from_shape_fn((n, 1), |(i, _)| 0.3 * i as f64 - 1.0),
    ];
    let gates = Array2::from_shape_fn((n, 2), |(i, atom)| match (atom, i) {
        (0, 0..=3) => 0.75,
        (1, 1) => 0.2,
        (1, 4) => 0.5,
        (1, 6) => 0.8,
        _ => 0.0,
    });
    // Control: over all rows, atom 0's column does vary.
    let column_mean = gates.column(0).sum() / n as f64;
    let unconditional = gates
        .column(0)
        .iter()
        .map(|gate| (gate - column_mean) * (gate - column_mean))
        .sum::<f64>()
        / n as f64;
    assert_eq!(unconditional, 0.140625);

    let code_of = |gates: &Array2<f64>, coords: &[Array2<f64>]| {
        native_gate_amplitude_code(
            &support(gates),
            NativeGateModel::Independent,
            &plans,
            &views(&decoders),
            &views(coords),
            None,
        )
    };
    let code = code_of(&gates, &coords).expect("amplitude code");
    assert_eq!(code.representation_distortion, 0.0);
    assert_eq!(code.components.len(), 2);
    assert_eq!(code.components[0], (0.5, vec![0.0]), "the support carries atom 0");
    assert_eq!(code.components[1].0, 3.0 / 8.0);
    assert!(
        (code.components[1].1[0] - 0.45).abs() < 1.0e-12,
        "atom 1's conditional amplitude source: {:?}",
        code.components
    );
    assert!((code.decoded_variance() - 3.0 / 8.0 * 0.45).abs() < 1.0e-12);

    // Rows appended where neither atom fires leave each conditional source
    // unchanged; only the weights move.
    let extra = 4;
    let longer_gates = Array2::from_shape_fn((n + extra, 2), |(i, atom)| {
        if i < n { gates[[i, atom]] } else { 0.0 }
    });
    let longer_coords: Vec<Array2<f64>> = coords
        .iter()
        .map(|block| {
            Array2::from_shape_fn((n + extra, 1), |(i, _)| {
                if i < n { block[[i, 0]] } else { 40.0 * i as f64 }
            })
        })
        .collect();
    let longer = code_of(&longer_gates, &longer_coords).expect("longer amplitude code");
    assert_eq!(longer.components[0], (4.0 / 12.0, vec![0.0]));
    assert_eq!(longer.components[1].0, 3.0 / 12.0);
    assert_eq!(longer.components[1].1, code.components[1].1);

    // One firing cannot estimate an amplitude variance: unavailable, not free.
    let once = Array2::from_shape_fn((n, 2), |(i, atom)| {
        if atom == 1 && i != 4 { 0.0 } else { gates[[i, atom]] }
    });
    let refusal = code_of(&once, &coords).expect_err("one firing of an amplitude");
    assert!(refusal.contains("unavailable"), "{refusal}");
}

#[test]
fn native_gate_amplitude_code_codes_softmax_rows_on_reference_free_simplex_axes_2933_f10() {
    // Three atoms with softmax gates, whose rows sum to one. The simplex code carries
    // two amplitude axes, and they equal the spectrum computed in an explicit
    // orthonormal (Helmert) chart of the simplex, with each chart axis decoded atom by
    // atom. Relabelling the atoms, which changes any "reference" atom, leaves the
    // spectrum unchanged. Gates off the simplex are reconstructed on it, and the
    // departure is charged as decoded distortion.
    let n = 8;
    let plans = [periodic_plan(), periodic_plan(), periodic_plan()];
    let coords: Vec<Array2<f64>> = (0..3)
        .map(|atom| {
            Array2::from_shape_fn((n, 1), |(row, _)| {
                (0.13 * row as f64 + 0.29 * atom as f64).fract()
            })
        })
        .collect();
    let decoders: Vec<Array2<f64>> = (0..3)
        .map(|atom| dense_decoder(&plans[atom], 3, 0.5 + 1.7 * atom as f64))
        .collect();
    let logits = [
        [0.3, -1.2, 0.8],
        [1.5, 0.1, -0.4],
        [-0.6, 0.9, 0.2],
        [0.0, 0.0, 1.1],
        [2.0, -0.5, -1.0],
        [-1.3, 1.4, 0.6],
        [0.7, 0.7, -0.2],
        [-0.1, -0.9, 1.6],
    ];
    let gates = Array2::from_shape_fn((n, 3), |(row, atom)| {
        let total: f64 = logits[row].iter().map(|value: &f64| value.exp()).sum();
        logits[row][atom].exp() / total
    });
    let code_of = |gates: &Array2<f64>,
                   plans: &[SaeAtomGeometryPlan],
                   decoders: &[Array2<f64>],
                   coords: &[Array2<f64>]| {
        native_gate_amplitude_code(
            &support(gates),
            NativeGateModel::Simplex,
            plans,
            &views(decoders),
            &views(coords),
            None,
        )
        .expect("simplex amplitude code")
    };
    let code = code_of(&gates, &plans, &decoders, &coords);
    assert_eq!(code.components.len(), 1);
    let (weight, spectrum) = &code.components[0];
    assert_eq!(*weight, 1.0);
    assert_eq!(spectrum.len(), 3);
    assert!(spectrum[1] > 1.0e-4, "two free simplex axes: {spectrum:?}");
    assert!(
        spectrum[2] <= 1.0e-12 * spectrum[0],
        "the sum direction carries no information: {spectrum:?}"
    );
    assert!(code.representation_distortion <= 1.0e-24, "{}", code.representation_distortion);

    // Oracle: the Helmert chart z = H a, whose orthonormal rows are orthogonal to 1.
    let helmert = [
        [1.0 / 2.0_f64.sqrt(), -1.0 / 2.0_f64.sqrt(), 0.0],
        [1.0 / 6.0_f64.sqrt(), 1.0 / 6.0_f64.sqrt(), -2.0 / 6.0_f64.sqrt()],
    ];
    let unit = Array2::from_elem((n, 1), 1.0);
    let curves: Vec<Array2<f64>> = (0..3)
        .map(|atom| {
            decoded(
                std::slice::from_ref(&plans[atom]),
                std::slice::from_ref(&decoders[atom]),
                std::slice::from_ref(&coords[atom]),
                &unit,
            )
        })
        .collect();
    let chart = Array2::from_shape_fn((n, 2), |(row, axis)| {
        (0..3).map(|atom| helmert[axis][atom] * gates[[row, atom]]).sum::<f64>()
    });
    let chart_mean = [chart.column(0).sum() / n as f64, chart.column(1).sum() / n as f64];
    let mut sigma = [[0.0_f64; 2]; 2];
    let mut metric = [[0.0_f64; 2]; 2];
    for row in 0..n {
        let centered = [chart[[row, 0]] - chart_mean[0], chart[[row, 1]] - chart_mean[1]];
        // The output direction of chart axis u: Σ_k H[u][k]·γ_k(t_ik).
        let direction = |axis: usize| -> Vec<f64> {
            (0..3)
                .map(|channel| {
                    (0..3)
                        .map(|atom| helmert[axis][atom] * curves[atom][[row, channel]])
                        .sum::<f64>()
                })
                .collect()
        };
        let directions = [direction(0), direction(1)];
        for u in 0..2 {
            for v in 0..2 {
                sigma[u][v] += centered[u] * centered[v] / (n - 1) as f64;
                let dot: f64 = directions[u].iter().zip(&directions[v]).map(|(x, y)| x * y).sum();
                metric[u][v] += dot / n as f64;
            }
        }
    }
    // Eigenvalues of Σ_z·G_z, similar to G_z^{1/2} Σ_z G_z^{1/2}.
    let product = [
        [
            sigma[0][0] * metric[0][0] + sigma[0][1] * metric[1][0],
            sigma[0][0] * metric[0][1] + sigma[0][1] * metric[1][1],
        ],
        [
            sigma[1][0] * metric[0][0] + sigma[1][1] * metric[1][0],
            sigma[1][0] * metric[0][1] + sigma[1][1] * metric[1][1],
        ],
    ];
    let trace = product[0][0] + product[1][1];
    let det = product[0][0] * product[1][1] - product[0][1] * product[1][0];
    let disc = (trace * trace / 4.0 - det).max(0.0).sqrt();
    let oracle = [trace / 2.0 + disc, trace / 2.0 - disc];
    for axis in 0..2 {
        assert!(
            (spectrum[axis] - oracle[axis]).abs() <= 1.0e-9 * oracle[0],
            "simplex spectrum {spectrum:?} must match the Helmert-chart spectrum {oracle:?}"
        );
    }

    // Relabel the atoms: a cyclic shift changes every "reference" choice.
    let order = [2_usize, 0, 1];
    let shifted_plans: Vec<SaeAtomGeometryPlan> =
        order.iter().map(|&atom| plans[atom].clone()).collect();
    let shifted_decoders: Vec<Array2<f64>> =
        order.iter().map(|&atom| decoders[atom].clone()).collect();
    let shifted_coords: Vec<Array2<f64>> = order.iter().map(|&atom| coords[atom].clone()).collect();
    let shifted_gates = Array2::from_shape_fn((n, 3), |(row, column)| gates[[row, order[column]]]);
    let shifted = code_of(&shifted_gates, &shifted_plans, &shifted_decoders, &shifted_coords);
    for axis in 0..2 {
        assert!(
            (shifted.components[0].1[axis] - spectrum[axis]).abs() <= 1.0e-9 * spectrum[0],
            "relabelled atoms changed the simplex spectrum: {:?} vs {spectrum:?}",
            shifted.components[0].1
        );
    }

    // Every row scaled to sum 1.5: the receiver reconstructs the gates on the
    // simplex, so each gate is off by 0.5/3 and that decoded departure is charged.
    // The projected innovations scale by 1.5, so the spectrum scales by 2.25.
    let off_simplex = gates.mapv(|gate| 1.5 * gate);
    let off = code_of(&off_simplex, &plans, &decoders, &coords);
    let departure = decoded(&plans, &decoders, &coords, &Array2::from_elem((n, 3), 0.5 / 3.0));
    let expected = departure.iter().map(|value| value * value).sum::<f64>() / n as f64;
    assert!(expected > 1.0e-3, "fixture: the departure decodes to a visible output");
    assert!(
        (off.representation_distortion - expected).abs() <= 1.0e-12 * expected,
        "off-simplex gates must be charged {expected}, got {}",
        off.representation_distortion
    );
    for axis in 0..2 {
        assert!(
            (off.components[0].1[axis] - 2.25 * spectrum[axis]).abs() <= 1.0e-9 * spectrum[0],
            "{:?} vs 2.25 · {spectrum:?}",
            off.components[0].1
        );
    }
}

#[test]
fn native_gate_amplitude_code_charges_a_non_unit_topk_gate_as_decoded_distortion_2933_f10() {
    // Unit-support gates transmit no amplitude, and a gate of one costs nothing. A
    // transmitted gate of 0.5 decodes half the curve, and the receiver's unit gate is
    // off by 0.5·γ. With constant curves γ_0 = (2, 0) and γ_1 = (0, −1), the row
    // supports {0}, {1}, {0, 1} and {} give departures of squared norm 1, 0.25, 1.25
    // and 0, so the distortion is (1 + 0.25 + 1.25 + 0) / 4.
    let n = 4;
    let plans = [periodic_plan(), periodic_plan()];
    let width = plans[0].basis_size().expect("plan width");
    let mut decoder_0 = Array2::<f64>::zeros((width, 2));
    decoder_0[[0, 0]] = 2.0;
    let mut decoder_1 = Array2::<f64>::zeros((width, 2));
    decoder_1[[0, 1]] = -1.0;
    let decoders = [decoder_0, decoder_1];
    let coords = [Array2::from_elem((n, 1), 0.4), Array2::from_elem((n, 1), 0.7)];
    let unit = array![[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [0.0, 0.0]];
    let half = unit.mapv(|gate| 0.5 * gate);
    let code_of = |gates: &Array2<f64>| {
        native_gate_amplitude_code(
            &support(gates),
            NativeGateModel::UnitSupport,
            &plans,
            &views(&decoders),
            &views(&coords),
            None,
        )
        .expect("unit-support amplitude code")
    };
    let unit_code = code_of(&unit);
    assert!(unit_code.components.is_empty());
    assert_eq!(unit_code.representation_distortion, 0.0);
    let half_code = code_of(&half);
    assert!(half_code.components.is_empty());
    let expected = (1.0 + 0.25 + 1.25 + 0.0) / 4.0;
    assert!(
        (half_code.representation_distortion - expected).abs() < 1.0e-12,
        "non-unit gates must be charged {expected}, got {}",
        half_code.representation_distortion
    );
}
