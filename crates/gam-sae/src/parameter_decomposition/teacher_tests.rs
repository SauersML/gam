#![cfg(test)]
//! A13 (#2951): teacher controls scored at bounded code, and the cancelling pair.
//!
//! Take `p` states `x_i ∈ ℝ^d` and the shift `i → i + 1 mod p`. When `p ≤ d` and the
//! states are linearly independent, as generic states are, `X₊ X⁺` realizes the shift
//! exactly whether or not a rotation was planted. So held-out fidelity alone cannot reject
//! a random connection (mpd-modadd, #2951 comment 5716898972).
//!
//! What separates a planted rotation is code length at equal decoded fidelity (P11, P18). A
//! plane rotation sends one plane and one angle on `ℝ/2πℤ`; a generic operator sends `d²`
//! reals. Lengths are compared only between artifacts whose decoded distortion meets the
//! declared tolerance ([`code_saving_at_proven_fidelity`]).
//!
//! The declared inputs are the lattice precision, the angle resolution and the fidelity
//! tolerance. The regime `p > d`, where random states admit no linear realizer, needs a
//! derived lower bound on the least-squares residual and lands separately.
//!
//! # The cancelling pair (P10, A6 at block level)
//!
//! Appending `(+P, −P)` to a decomposition leaves the all-on tensor exact, so all-on fidelity
//! has no power against it. Deleting one member moves the tensor by `P`. The support of the
//! admissible zonotope in the direction `vec(P)` names that witness mask (P8), and executing the
//! block there refutes a fidelity claim with an `EvidenceStatus::Counterexample`. The uniform-mask
//! mean of the same pairing is zero: a stochastic average is reported beside the counterexample
//! and never as a bound.

use super::codec::{BitString, code_saving_at_proven_fidelity};
use super::moments::{GeneratorPart, MaskDomain, MaskMomentSystem, MomentBlock, MomentVector};
use super::precision::{
    DecodableArtifact, DeclaredPrecision, DecodedFidelity, FidelityVerdict, LatticeCode,
    PeriodicQuotient, QuotientCode, decode_then_evaluate,
};
use super::rewrite::{ComponentMask, ComponentMlp, ComponentRead, MlpMask, NativeMlp};
use super::supports::{EvidenceStatus, ExactBasis};
use gam_linalg::faer_ndarray::{FaerArrayView, col_piv_qr_solve_lstsq};
use gam_linalg::roundoff::UNIT_ROUNDOFF;
use gam_math::gaussian_activation::GaussianActivation;
use ndarray::{Array1, Array2, Axis, concatenate};
use rand::rngs::StdRng;
use rand::{RngExt, SeedableRng};
use std::f64::consts::TAU;

const WIDTH: usize = 8;
const STATES: usize = 6;

/// A plane rotation sent as its plane basis (`d × 2` reals on the declared lattice, row by
/// row) and its angle on `ℝ/2πℤ`.
struct PlaneRotationArtifact {
    basis: LatticeCode,
    angle: QuotientCode,
}

impl PlaneRotationArtifact {
    /// The written length of both messages.
    fn code_bits(&self) -> u64 {
        let mut message = BitString::new();
        self.basis.write(&mut message).expect("the plane basis writes");
        self.angle.write(&mut message).expect("the angle writes");
        message.len_bits()
    }
}

impl DecodableArtifact for PlaneRotationArtifact {
    type Decoded = Array2<f64>;

    /// `W = I + U (R(α) − I) Uᵀ`, rebuilt from the decoded plane and angle only.
    fn decode(&self) -> Result<Array2<f64>, String> {
        let basis = self.basis.decode()?;
        let angle = self.angle.decode()?;
        if basis.len() != 2 * WIDTH || angle.len() != 1 {
            return Err(format!(
                "a plane rotation decodes {} basis reals and {} angles, expected {} and 1",
                basis.len(),
                angle.len(),
                2 * WIDTH
            ));
        }
        let (sine, cosine) = angle[0].sin_cos();
        let block = [[cosine - 1.0, -sine], [sine, cosine - 1.0]];
        Ok(Array2::from_shape_fn((WIDTH, WIDTH), |(row, column)| {
            let mut entry = if row == column { 1.0 } else { 0.0 };
            for (a, block_row) in block.iter().enumerate() {
                for (b, value) in block_row.iter().enumerate() {
                    entry += basis[2 * row + a] * value * basis[2 * column + b];
                }
            }
            entry
        }))
    }
}

fn lattice_bits(code: &LatticeCode) -> u64 {
    let mut message = BitString::new();
    code.write(&mut message).expect("the lattice code writes");
    message.len_bits()
}

fn shifted_rows(states: &Array2<f64>) -> Array2<f64> {
    Array2::from_shape_fn(states.dim(), |(row, column)| {
        states[[(row + 1) % states.nrows(), column]]
    })
}

/// `x_i = (cos θ_i, sin θ_i, ½, …, ½)` with `θ_i = 2πi/p`, and the shifted rows. The shift is
/// a row permutation of the same floats, so the reference is exact.
fn planted_cycle_states() -> (Array2<f64>, Array2<f64>) {
    let states = Array2::from_shape_fn((STATES, WIDTH), |(row, column)| {
        let (sine, cosine) = (TAU * row as f64 / STATES as f64).sin_cos();
        match column {
            0 => cosine,
            1 => sine,
            _ => 0.5,
        }
    });
    let shifted = shifted_rows(&states);
    (states, shifted)
}

fn random_cycle_states(seed: u64) -> (Array2<f64>, Array2<f64>) {
    let mut rng = StdRng::seed_from_u64(seed);
    let states = Array2::from_shape_simple_fn((STATES, WIDTH), || rng.random_range(-1.0..1.0));
    let shifted = shifted_rows(&states);
    (states, shifted)
}

/// A realizer `T` of the shift with `X Tᵀ = Y`. It solves the square system
/// `(X Xᵀ) Z = Y` by the rank-aware QR owner, so `T = Zᵀ X` and `X Tᵀ = X Xᵀ Z = Y`.
fn least_squares_realizer(states: &Array2<f64>, shifted: &Array2<f64>) -> Array2<f64> {
    let gram = states.dot(&states.t());
    let solution = col_piv_qr_solve_lstsq(
        FaerArrayView::new(&gram).as_ref(),
        FaerArrayView::new(shifted).as_ref(),
    );
    let coefficients = Array2::from_shape_fn((STATES, WIDTH), |(row, column)| {
        solution[(row, column)]
    });
    coefficients.t().dot(states)
}

/// Decodes the artifact, applies the decoded operator to every state, and measures the largest
/// `|image − reference|` entry. Each entry is one rounded subtraction of two floats, so the
/// exact distortion exceeds the measured one by at most `u·distortion`, rounded up. Returns the
/// artifact's code length with its decoded fidelity, for [`code_saving_at_proven_fidelity`].
fn score<A: DecodableArtifact>(
    artifact: &A,
    code_bits: u64,
    operator: impl FnOnce(&A::Decoded) -> Array2<f64>,
    states: &Array2<f64>,
    reference: &Array2<f64>,
    tolerance: f64,
) -> (u64, DecodedFidelity<(), &'static str>) {
    let fidelity = decode_then_evaluate(
        artifact,
        |decoded| Ok(states.dot(&operator(decoded).t())),
        reference,
        |images: &Array2<f64>, native: &Array2<f64>| {
            let largest = images
                .iter()
                .zip(native.iter())
                .fold(0.0_f64, |largest, (image, target)| largest.max((image - target).abs()));
            EvidenceStatus::exact(
                largest,
                (UNIT_ROUNDOFF * largest).next_up(),
                ExactBasis::Exhaustive {
                    cardinality: STATES as u64,
                },
                None::<()>,
                "the declared cycle states",
            )
            .map_err(|error| format!("{error:?}"))
        },
        tolerance,
    )
    .expect("the artifact decodes and evaluates");
    (code_bits, fidelity)
}

/// On planted cycle states, a plane rotation and its dense generic realizer both meet the
/// declared tolerance, and the rotation's code is shorter. On random states the generic
/// realizer also meets the tolerance, so fidelity alone accepts a random connection, which is
/// the premise of correction C3. The planted rotation's artifact does not realize that
/// connection. Positive control: the code comparison refuses that pair, because lengths at
/// different fidelities are not a model comparison.
#[test]
fn a_planted_rotation_wins_on_code_where_fidelity_alone_accepts_a_random_connection() {
    let precision = DeclaredPrecision::new(20).expect("a declared dyadic precision");
    let angle_resolution = 20;
    let tolerance = 2.0_f64.powi(-10);
    let quotient = PeriodicQuotient::new(TAU).expect("the angle quotient");
    let as_operator = |values: &Vec<f64>| {
        Array2::from_shape_vec((WIDTH, WIDTH), values.clone()).expect("d × d reals")
    };

    let mut plane = vec![0.0; 2 * WIDTH];
    plane[0] = 1.0;
    plane[3] = 1.0;
    let rotation = PlaneRotationArtifact {
        basis: LatticeCode::encode(&plane, precision).expect("the plane basis encodes"),
        angle: QuotientCode::encode(&[TAU / STATES as f64], quotient, angle_resolution)
            .expect("the angle encodes"),
    };
    let rotation_bits = rotation.code_bits();
    let (sine, cosine) = (TAU / STATES as f64).sin_cos();
    let mut native = Array2::<f64>::eye(WIDTH);
    native[[0, 0]] = cosine;
    native[[0, 1]] = -sine;
    native[[1, 0]] = sine;
    native[[1, 1]] = cosine;
    let generic = LatticeCode::encode(native.as_slice().expect("standard layout"), precision)
        .expect("the dense operator encodes");

    let (planted, planted_shifted) = planted_cycle_states();
    let (planted_rotation_bits, planted_rotation) = score(
        &rotation,
        rotation_bits,
        |operator| operator.clone(),
        &planted,
        &planted_shifted,
        tolerance,
    );
    let (planted_generic_bits, planted_generic) = score(
        &generic,
        lattice_bits(&generic),
        as_operator,
        &planted,
        &planted_shifted,
        tolerance,
    );
    let saving = code_saving_at_proven_fidelity(
        (planted_generic_bits, &planted_generic),
        (planted_rotation_bits, &planted_rotation),
    )
    .expect("both planted artifacts meet the declared tolerance");
    assert!(
        saving > 0,
        "the plane rotation must be shorter: {planted_rotation:?} against {planted_generic:?}"
    );

    let (random, random_shifted) = random_cycle_states(2951);
    let realizer = least_squares_realizer(&random, &random_shifted);
    let random_code = LatticeCode::encode(realizer.as_slice().expect("standard layout"), precision)
        .expect("the realizer encodes");
    let (random_generic_bits, random_generic) = score(
        &random_code,
        lattice_bits(&random_code),
        as_operator,
        &random,
        &random_shifted,
        tolerance,
    );
    assert_eq!(
        random_generic.verdict(),
        FidelityVerdict::Meets,
        "a random connection's generic realizer must meet the tolerance: {random_generic:?}"
    );

    let (random_rotation_bits, random_rotation) = score(
        &rotation,
        rotation_bits,
        |operator| operator.clone(),
        &random,
        &random_shifted,
        tolerance,
    );
    assert!(
        code_saving_at_proven_fidelity(
            (random_generic_bits, &random_generic),
            (random_rotation_bits, &random_rotation),
        )
        .is_err(),
        "the planted rotation does not realize a random connection, so no code comparison stands: \
         {random_rotation:?}"
    );
}

const HIDDEN: usize = 8;
const ROWS: usize = 5;

fn uniform_matrix(rng: &mut StdRng, rows: usize, cols: usize) -> Array2<f64> {
    Array2::from_shape_simple_fn((rows, cols), || rng.random_range(-1.0..1.0))
}

/// The largest `|a − b|` entry, and one unit roundoff of it rounded up: each entry is one
/// rounded subtraction of two executed floats.
fn executed_distortion(left: &Array2<f64>, right: &Array2<f64>) -> (f64, f64) {
    let largest = left
        .iter()
        .zip(right.iter())
        .fold(0.0_f64, |largest, (a, b)| largest.max((a - b).abs()));
    (largest, (UNIT_ROUNDOFF * largest).next_up())
}

/// A residual GELU block with the cancelling pair `(+P, −P)`, `P = U Vᵀ` of rank two, appended
/// to its read-in weight as two mask groups.
///
/// The read stacks `I_d` over `Vᵀ` twice (`C = d + 4`, full column rank through `I_d`). The
/// candidate write stacks `W₁`, `+U` and `−U`, so `N R = W₁` exactly in reals. Each pair member
/// is one group whose two coordinates share one control (P1's `cI`).
///
/// The declared experiment:
/// - the mask domain is `[0, 1]` per member;
/// - the fidelity claim is `sup_m max |f_{Θ(m)}(x) − f_{θ*}(x)| ≤ ε` over the executed block;
/// - ε = 2⁻¹⁰.
///
/// The support in direction `vec(P)` names the witness that deletes one member and keeps the
/// other. The executed distortion there refutes the claim.
///
/// Positive controls:
/// - the all-on and both-deleted masks do not refute it, so all-on fidelity has no power here;
/// - the uniform-mask mean of `⟨vec(P), q⟩` is zero within its band, while the support is `‖P‖²`.
#[test]
fn a_cancelling_pair_is_refuted_at_its_support_witness_through_the_executed_block() {
    let mut rng = StdRng::seed_from_u64(2951);
    let read_in = uniform_matrix(&mut rng, HIDDEN, WIDTH);
    let write_out = uniform_matrix(&mut rng, WIDTH, HIDDEN);
    let bias_in = Array1::from_shape_simple_fn(HIDDEN, || rng.random_range(-1.0..1.0));
    let bias_out = Array1::from_shape_simple_fn(WIDTH, || rng.random_range(-1.0..1.0));
    let inputs = uniform_matrix(&mut rng, ROWS, WIDTH);
    let left = uniform_matrix(&mut rng, HIDDEN, 2);
    let right = uniform_matrix(&mut rng, WIDTH, 2);
    let tolerance = 2.0_f64.powi(-10);

    let native = NativeMlp::new(
        read_in.clone(),
        bias_in,
        write_out.clone(),
        bias_out,
        GaussianActivation::ExactGelu,
    )
    .expect("the block's shapes compose");
    let native_output = native.execute(inputs.view()).expect("the native block executes");

    let identity_read = Array2::<f64>::eye(WIDTH);
    let read = concatenate![Axis(0), identity_read, right.t(), right.t()];
    let candidate_write = concatenate![Axis(1), read_in, left, left.mapv(|value| -value)];
    let hidden_identity = Array2::<f64>::eye(HIDDEN);
    let program = ComponentMlp::new(
        native.clone(),
        ComponentRead {
            read: read.view(),
            candidate_write: candidate_write.view(),
        },
        ComponentRead {
            read: hidden_identity.view(),
            candidate_write: write_out.view(),
        },
    )
    .expect("both weights factor exactly");

    let pair = left.dot(&right.t());
    let generator = |sign: f64| {
        vec![GeneratorPart {
            block: 0,
            vector: Array1::from_iter(pair.iter().map(|value| sign * value)),
        }]
    };
    let system = MaskMomentSystem::new(
        vec![MomentBlock {
            dimension: HIDDEN * WIDTH,
        }],
        vec![generator(1.0), generator(-1.0)],
    )
    .expect("the moment system is well formed");
    let domain = MaskDomain::new(vec![(0.0, 1.0), (0.0, 1.0)]).expect("the declared mask domain");
    let kept = [false, false];
    let direction = MomentVector {
        blocks: vec![Array1::from_iter(pair.iter().copied())],
    };
    let support = system
        .support(&domain, &kept, &direction)
        .expect("the support evaluates");
    let witness = domain.mask_at(&support.witness).expect("the witness names a mask");

    let execute_at = |masks: &[f64]| {
        let mut component_masks = vec![1.0; WIDTH + 4];
        component_masks[WIDTH] = masks[0];
        component_masks[WIDTH + 1] = masks[0];
        component_masks[WIDTH + 2] = masks[1];
        component_masks[WIDTH + 3] = masks[1];
        let component_masks = Array1::from_vec(component_masks);
        program
            .execute(
                inputs.view(),
                MlpMask {
                    read_in: ComponentMask::Components(component_masks.view()),
                    write_out: ComponentMask::AllOn,
                },
            )
            .expect("the component block executes")
    };

    let (value, numerical_error) = executed_distortion(&execute_at(&witness), &native_output);
    let refutation =
        EvidenceStatus::<Vec<f64>, ()>::counterexample(value, numerical_error, tolerance, witness)
            .expect("the witness refutes the fidelity claim");
    assert!(
        matches!(refutation, EvidenceStatus::Counterexample { .. }),
        "the support witness must carry a counterexample at distortion {value:e}"
    );

    for control in [[1.0, 1.0], [0.0, 0.0]] {
        let (control_value, control_error) =
            executed_distortion(&execute_at(&control), &native_output);
        assert!(
            EvidenceStatus::<Vec<f64>, ()>::counterexample(
                control_value,
                control_error,
                tolerance,
                control.to_vec(),
            )
            .is_err(),
            "the mask {control:?} keeps the pair balanced and must not refute: {control_value:e}"
        );
    }

    let law = system
        .uniform_mask_law_moments(&domain, &kept, &direction)
        .expect("the uniform-mask moments evaluate");
    assert!(
        law.mean.abs() <= law.mean_band && support.value - support.band > law.mean_band,
        "the uniform-mask mean {law:?} must sit at zero while the support {support:?} does not"
    );
}
