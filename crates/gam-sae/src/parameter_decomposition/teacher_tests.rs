#![cfg(test)]
//! A13 (#2951): planted-rotation teacher controls, scored at bounded code.
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
//!
//! # The positive half: a planted rotation, recovered and intervened on (P1, P2, P3, P5)
//!
//! A residual block whose read-in weight rotates `K` planted planes has those planes and angles
//! recovered by [`recover_plane_rotations`] from the weight alone, within its derived bars. The
//! recovered planes then build an exact component program, and every intervention of a declared
//! finite family executes through it on held-out inputs against the teacher's own native edit.
//! Each entry's verdict is an [`EvidenceStatus`]: a [`EvidenceStatus::UniformBound`] from a
//! Lipschitz certificate, or a held-out [`EvidenceStatus::Counterexample`]. The same body runs at
//! `d = 8` and `d = 512`, for a ReLU and an exact-GELU teacher; `planted_rotation_fidelity` states
//! each derivation.

use super::codec::{BitString, code_saving_at_proven_fidelity};
use super::moments::{GeneratorPart, MaskDomain, MaskMomentSystem, MomentBlock, MomentVector};
use super::operators::{PlaneRotation, RotationPath};
use super::precision::{
    DecodableArtifact, DeclaredPrecision, DecodedFidelity, FidelityVerdict, LatticeCode,
    PeriodicQuotient, QuotientCode, decode_then_evaluate,
};
use super::receipts::{affine_stage_band, factored_edit_stage_band};
use super::rewrite::{ComponentMask, ComponentMlp, ComponentRead, ExactFactor, MlpMask, NativeMlp};
use super::spectral::{
    PlaneRotationRecovery, RotationAmbiguity, RotationCluster, RotationClusterKind,
    recover_plane_rotations,
};
use super::supports::{EvidenceStatus, ExactBasis, Extremum};
use super::test_support::projector_distance;
use gam_linalg::faer_ndarray::{FaerArrayView, FaerQr, col_piv_qr_solve_lstsq};
use gam_linalg::roundoff::{UNIT_ROUNDOFF, accumulation_growth};
use gam_math::gaussian_activation::{GaussianActivation, gaussian_hermite_coefficients};
use gam_math::roundoff::inflated;
use ndarray::{Array1, Array2, ArrayView1, ArrayView2, Axis, concatenate, s};
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

/// The planted angles, one per plane: distinct cosines away from `±1`, as in spectral's A3
/// fixture.
const PLANTED_ANGLES: [f64; 3] = [0.35, 1.2, 2.6];
/// Held-out rows per intervention. The recovery reads the weight alone, so no input is seen
/// before the interventions; these rows come from their own seed.
const HELD_OUT_ROWS: usize = 16;
/// The fidelity tolerance `ε = 2⁻¹⁰`, a declared input of the A13 experiment (see
/// `planted_rotation_fidelity`).
const FIDELITY_TOLERANCE: f64 = 1.0 / 1024.0;
const BASIS_SEED: u64 = 0x2951_a130;
const BLOCK_SEED: u64 = 0x2951_a131;
const FAMILY_SEED: u64 = 0x2951_a132;
const HELD_OUT_SEED: u64 = 0x2951_a133;

/// A verdict on one intervention, witnessed by a held-out `(row, output)` entry.
type FidelityStatus = EvidenceStatus<(usize, usize), String>;

fn up(value: f64) -> f64 {
    value.next_up()
}

/// `‖v‖₂` rounded up. Each square and partial sum is rounded to nearest and then stepped one
/// float up, which lies at or above the exact result of nonnegative operands, and the correctly
/// rounded root takes one more step.
fn norm_upper(values: impl IntoIterator<Item = f64>) -> f64 {
    let mut sum = 0.0_f64;
    for value in values {
        sum = up(sum + up(value * value));
    }
    up(sum.sqrt())
}

/// A residual block `h + W₂ σ(W₁ h + b₁) + b₂` whose read-in weight rotates `K` planted planes,
/// `W₁ = I + U (B − I) Uᵀ`, with `B = ⊕_k [[c_k, −s_k], [s_k, c_k]]` the `rho_so2` blocks of the
/// planted angles and `U` (`d × 2K`) the Householder factor of a seeded draw.
///
/// The planted truth is one [`PlaneRotation`] per plane, `(U_k, α_k)`, whose chord and angle
/// paths are the teacher's native edits.
///
/// # The declared error
///
/// Let `U_o` be the polar factor of `U` and `R_o = ⊕_k R(α̃_k)` with `α̃_k = atan2(s_k, c_k)`, so
/// that `B_k = r_k R(α̃_k)`. Then `O = I + U_o (R_o − I) U_oᵀ` is exactly orthogonal, with plane
/// `k` equal to `range(U_o,k)` and cosine `cos α̃_k`.
/// - `‖U − U_o‖₂ = max |σ_i − 1| ≤ max |σ_i² − 1| ≤ ‖UᵀU − I‖_F ≤ δ`: the owner's measured
///   defect, widened by its own rounding and by the Gram's, `γ_d ‖U‖_F²` (Cauchy–Schwarz).
/// - `‖B − R_o‖₂ = max |r_k − 1| ≤ max |r_k² − 1| ≤ η_R`: the computed `c² + s² − 1` widened by
///   `γ₂ (c² + s²)`. Subtracting one is exact (Sterbenz).
/// - `U (B − I) Uᵀ − U_o (R_o − I) U_oᵀ = (U − U_o)(B − I)Uᵀ + U_o (B − R_o) Uᵀ
///   + U_o (R_o − I)(U − U_o)ᵀ`, so `‖W₁ − O‖₂ ≤ δ(2 + η_R)(1 + δ) + η_R (1 + δ) + 2δ` on the
///   stored factors, plus the formation band of `W₁`.
/// - `‖U_o,k U_o,kᵀ − U_k U_kᵀ‖₂ ≤ δ(2 + δ)`, and `|cos α̃_k − c_k| = |c_k| |1 − r_k|/r_k ≤ η_R`.
struct PlantedTeacher {
    native: NativeMlp,
    basis: Array2<f64>,
    planes: Vec<PlaneRotation>,
    /// `c_k` as `rho_so2` rounds it.
    cosines: Vec<f64>,
    /// A bound on `‖W₁ − O‖₂`.
    declared_error: f64,
    /// `δ(2 + δ)`.
    plane_defect: f64,
    /// `η_R`.
    cosine_defect: f64,
    held_out: Array2<f64>,
}

impl PlantedTeacher {
    fn plane_basis(&self, plane: usize) -> ArrayView2<'_, f64> {
        self.basis.slice(s![.., 2 * plane..2 * plane + 2])
    }
}

fn plant_teacher(width: usize, activation: GaussianActivation) -> PlantedTeacher {
    let columns = 2 * PLANTED_ANGLES.len();
    let mut rng = StdRng::seed_from_u64(BASIS_SEED);
    let (basis, _) = uniform_matrix(&mut rng, width, columns)
        .qr()
        .expect("Householder QR of a uniform draw");
    let whole = PlaneRotation::new(basis.clone(), Array1::from_iter(PLANTED_ANGLES))
        .expect("the planted basis is within the owner's orthonormality floor");
    // The measured defect sums `(2K)²` squares and takes a root. The Gram behind it rounds each
    // inner product of `d` terms by at most `γ_d ‖u_i‖ ‖u_j‖`.
    let basis_squared = inflated(
        basis.iter().map(|value| value * value).sum::<f64>(),
        width * columns,
    );
    let defect = up(inflated(whole.orthonormality_defect(), columns * columns + 1)
        + up(up(accumulation_growth(width)) * basis_squared));

    let mut generator = Array2::<f64>::zeros((columns, columns));
    let mut cosines = Vec::with_capacity(PLANTED_ANGLES.len());
    let mut cosine_defect = 0.0_f64;
    for (plane, &angle) in PLANTED_ANGLES.iter().enumerate() {
        let (sine, cosine) = angle.sin_cos();
        let radius_squared = cosine * cosine + sine * sine;
        cosine_defect = cosine_defect.max(up(
            (radius_squared - 1.0).abs()
                + up(up(accumulation_growth(2)) * inflated(radius_squared, 2)),
        ));
        let offset = 2 * plane;
        generator[[offset, offset]] = cosine - 1.0;
        generator[[offset, offset + 1]] = -sine;
        generator[[offset + 1, offset]] = sine;
        generator[[offset + 1, offset + 1]] = cosine - 1.0;
        cosines.push(cosine);
    }
    // `W₁ = I + (U M) Uᵀ` with `M = B − I`. A term rounds once forming `c − 1`, `2K` times through
    // `U M`, `2K` times through the product with `Uᵀ` and once adding the identity, over the
    // magnitude `|U| |M| |U|ᵀ + I`. Computing that magnitude takes as many roundings of
    // nonnegative terms, and one more covers `|c − 1|` against its rounded value.
    let read_in = Array2::<f64>::eye(width) + &basis.dot(&generator).dot(&basis.t());
    let absolute_basis = basis.mapv(f64::abs);
    let magnitude = Array2::<f64>::eye(width)
        + &absolute_basis
            .dot(&generator.mapv(f64::abs))
            .dot(&absolute_basis.t());
    let roundings = 2 * columns + 2;
    let growth = up(accumulation_growth(roundings));
    let formation = norm_upper(
        magnitude
            .iter()
            .map(|&value| up(growth * inflated(value, roundings + 1))),
    );
    let spread = up(1.0 + defect);
    let turned = up(up(defect * up(2.0 + cosine_defect)) * spread);
    let scaled = up(cosine_defect * spread);
    let declared_error = up(up(up(turned + scaled) + up(2.0 * defect)) + formation);
    let plane_defect = up(defect * up(2.0 + defect));

    let planes = PLANTED_ANGLES
        .iter()
        .enumerate()
        .map(|(plane, &angle)| {
            PlaneRotation::new(
                basis.slice(s![.., 2 * plane..2 * plane + 2]).to_owned(),
                Array1::from_elem(1, angle),
            )
            .expect("a planted plane is within the owner's orthonormality floor")
        })
        .collect();
    let mut rng = StdRng::seed_from_u64(BLOCK_SEED);
    let write_out = uniform_matrix(&mut rng, width, width);
    let bias_in = Array1::from_shape_simple_fn(width, || rng.random_range(-1.0..1.0));
    let bias_out = Array1::from_shape_simple_fn(width, || rng.random_range(-1.0..1.0));
    let native = NativeMlp::new(read_in, bias_in, write_out, bias_out, activation)
        .expect("the teacher's shapes compose");
    let held_out = uniform_matrix(
        &mut StdRng::seed_from_u64(HELD_OUT_SEED),
        HELD_OUT_ROWS,
        width,
    );
    PlantedTeacher {
        native,
        basis,
        planes,
        cosines,
        declared_error,
        plane_defect,
        cosine_defect,
        held_out,
    }
}

/// Whose native edit an intervention is scored against.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Reference {
    /// The teacher's own planes and angles.
    Planted,
    /// Each plane's two columns swapped, so every edit turns the other way: a mis-rotated edit.
    MisRotated,
    /// Each plane edited with the next plane's planted angle: a displaced angle, the planes and
    /// angles connected wrongly.
    DisplacedAngle,
}

fn reference_planes(teacher: &PlantedTeacher, reference: Reference) -> Vec<PlaneRotation> {
    let count = teacher.planes.len();
    (0..count)
        .map(|plane| {
            let basis = teacher.plane_basis(plane);
            let (basis, angle) = match reference {
                Reference::Planted => return teacher.planes[plane].clone(),
                Reference::MisRotated => (
                    concatenate![Axis(1), basis.slice(s![.., 1..2]), basis.slice(s![.., 0..1])],
                    PLANTED_ANGLES[plane],
                ),
                Reference::DisplacedAngle => {
                    (basis.to_owned(), PLANTED_ANGLES[(plane + 1) % count])
                }
            };
            PlaneRotation::new(basis, Array1::from_elem(1, angle))
                .expect("a reference plane is within the owner's orthonormality floor")
        })
        .collect()
}

/// `W₁ + Σ_k [D_k(path_k) − D_k(Chord(1))]`, with `D_k` the matrix of plane `k`'s
/// [`PlaneRotation::apply_delta`]: the teacher's native edited read-in weight, as stored. A plane
/// on an identity path, `Chord(1)` or `Angle(1)`, computes the same delta twice and adds exactly
/// zero, so all-on is `W₁` bit for bit.
fn native_edit(
    read_in: ArrayView2<'_, f64>,
    planes: &[PlaneRotation],
    controls: &[RotationPath],
) -> Array2<f64> {
    let width = read_in.ncols();
    let mut edited = read_in.to_owned();
    let mut axis = Array1::<f64>::zeros(width);
    for (plane, &control) in planes.iter().zip(controls) {
        for column in 0..width {
            axis[column] = 1.0;
            let moved = plane
                .apply_delta(control, axis.view())
                .expect("the plane edit applies");
            let anchor = plane
                .apply_delta(RotationPath::Chord(1.0), axis.view())
                .expect("the plane edit applies");
            axis[column] = 0.0;
            for row in 0..width {
                edited[[row, column]] += moved[row] - anchor[row];
            }
        }
    }
    edited
}

/// A recovered plane: its cluster, the cluster basis `V` (`d × 2`), the angle `θ̂` and the
/// certified complex structure `Ĵ` in that basis.
struct RecoveredPlane {
    cluster: usize,
    basis: Array2<f64>,
    angle: f64,
    structure: Array2<f64>,
}

/// Whether the cluster's cosine interval meets `c ± η`, the enclosure of a true cosine, rounded
/// outward.
fn admits(cluster: &RotationCluster, cosine: f64, defect: f64) -> bool {
    cluster.cosine_interval.0 <= up(cosine + defect)
        && (cosine - defect).next_down() <= cluster.cosine_interval.1
}

/// For each planted plane, the one cluster that admits its cosine. It must hold one plane with a
/// certified orientation.
fn matched_planes(
    teacher: &PlantedTeacher,
    recovery: &PlaneRotationRecovery,
) -> Vec<RecoveredPlane> {
    teacher
        .cosines
        .iter()
        .enumerate()
        .map(|(plane, &cosine)| {
            let admitting: Vec<usize> = recovery
                .clusters
                .iter()
                .enumerate()
                .filter(|(_, cluster)| admits(cluster, cosine, teacher.cosine_defect))
                .map(|(index, _)| index)
                .collect();
            assert_eq!(
                admitting.len(),
                1,
                "planted plane {plane} (cosine {cosine}) must be admitted by exactly one cluster, \
                 got {admitting:?}"
            );
            let cluster = &recovery.clusters[admitting[0]];
            let RotationClusterKind::Rotation {
                planes: 1,
                angle,
                complex_structure: Some(structure),
            } = &cluster.kind
            else {
                panic!(
                    "planted plane {plane} must be recovered as one plane with a certified \
                     orientation, got {:?}",
                    cluster.kind
                );
            };
            RecoveredPlane {
                cluster: admitting[0],
                basis: cluster.basis.clone(),
                angle: *angle,
                structure: structure.clone(),
            }
        })
        .collect()
}

/// The exact component program of the recovered planes (P5).
///
/// The read stacks `I_d` over `V_kᵀ` twice per plane (`C = d + 4K`, full column rank through
/// `I_d`). The candidate write stacks `I_d` beside plane `k`'s cosine generator
/// `V_k (cos θ̂_k − 1)`, written `−2 sin²(θ̂_k/2) V_k` so nothing cancels, and its sine generator
/// `sin θ̂_k V_k Ĵ_k`. Under the masks `(a_k, b_k)` of the two coordinate pairs,
/// `N diag(m) R = I + Σ_k V_k [a_k (cos θ̂_k − 1) I + b_k sin θ̂_k Ĵ_k] V_kᵀ`. The exact factor
/// adds the recovery residual `(W₁ − N R) R⁺`, so all-on is `W₁` in exact arithmetic.
///
/// A plane's two coordinates alone cannot carry its angle path under diagonal masks:
/// `(R̂ − I) diag(m₁, m₂) = R̂(tθ̂) − I` only where `tan(tθ̂/2) = tan(θ̂/2)`. Two generators per
/// plane carry P3's `W = I + Σ_k [(cos α_k − 1) P_k + sin α_k J_k]` at every angle.
fn plane_program(native: &NativeMlp, planes: &[RecoveredPlane]) -> ComponentMlp {
    let width = native.width();
    let mut reads = vec![Array2::<f64>::eye(width)];
    let mut writes = vec![Array2::<f64>::eye(width)];
    for plane in planes {
        let half_sine = (0.5 * plane.angle).sin();
        reads.push(plane.basis.t().to_owned());
        reads.push(plane.basis.t().to_owned());
        writes.push(plane.basis.mapv(|value| -2.0 * half_sine * half_sine * value));
        writes.push(plane.basis.dot(&plane.structure) * plane.angle.sin());
    }
    let read = concatenate(
        Axis(0),
        &reads.iter().map(|matrix| matrix.view()).collect::<Vec<_>>(),
    )
    .expect("the plane reads stack");
    let candidate_write = concatenate(
        Axis(1),
        &writes.iter().map(|matrix| matrix.view()).collect::<Vec<_>>(),
    )
    .expect("the plane writes stack");
    let hidden_identity = Array2::<f64>::eye(native.hidden_width());
    ComponentMlp::new(
        native.clone(),
        ComponentRead {
            read: read.view(),
            candidate_write: candidate_write.view(),
        },
        ComponentRead {
            read: hidden_identity.view(),
            candidate_write: native.write_out(),
        },
    )
    .expect("both weights factor exactly")
}

/// The component mask of one intervention. A chord `m` on plane `k` sets `a_k = b_k = m`: one
/// control ties the plane's four coordinates, P1's `cI` on each generator block. The angle path
/// `t` needs `a_k (cos θ̂ − 1) = cos tθ̂ − 1` and `b_k sin θ̂ = sin tθ̂`, so
/// `a_k = sin²(tθ̂/2)/sin²(θ̂/2)` and `b_k = sin tθ̂/sin θ̂`, both exactly one at `t = 1`.
fn component_mask(
    width: usize,
    planes: &[RecoveredPlane],
    controls: &[RotationPath],
) -> Array1<f64> {
    let mut mask = Array1::<f64>::ones(width + 4 * planes.len());
    for (index, (plane, &control)) in planes.iter().zip(controls).enumerate() {
        let (cosine_part, sine_part) = match control {
            RotationPath::Chord(scale) => (scale, scale),
            RotationPath::Angle(turn) => {
                let ratio = (0.5 * turn * plane.angle).sin() / (0.5 * plane.angle).sin();
                (ratio * ratio, (turn * plane.angle).sin() / plane.angle.sin())
            }
        };
        let offset = width + 4 * index;
        mask.slice_mut(s![offset..offset + 2]).fill(cosine_part);
        mask.slice_mut(s![offset + 2..offset + 4]).fill(sine_part);
    }
    mask
}

/// The declared finite family over `K` planes, one path per plane:
/// - every binary chord mask in `{0, 1}^K`, which includes all-on, all-off and each plane removed
///   alone;
/// - interior chords: each plane alone at `½` (P2's midpoint), and one seeded mask in `[0, 1)^K`;
/// - angle paths: each plane alone at `t ∈ {−1, ½, 2}` (reversed, halved, doubled), and one seeded
///   `t ∈ [−1, 2)^K`.
///
/// A plane left alone is on `Chord(1)`, its identity path. At `K = 3` the family has 22 entries.
fn declared_family(planes: usize) -> Vec<(String, Vec<RotationPath>)> {
    let alone = |plane: usize, path: RotationPath| -> Vec<RotationPath> {
        (0..planes)
            .map(|other| if other == plane { path } else { RotationPath::Chord(1.0) })
            .collect()
    };
    let mut family: Vec<(String, Vec<RotationPath>)> = (0..1_usize << planes)
        .map(|bits| {
            let masks: Vec<f64> = (0..planes).map(|plane| ((bits >> plane) & 1) as f64).collect();
            (
                format!("binary chord {masks:?}"),
                masks.into_iter().map(RotationPath::Chord).collect(),
            )
        })
        .collect();
    let mut rng = StdRng::seed_from_u64(FAMILY_SEED);
    for plane in 0..planes {
        family.push((
            format!("chord 1/2 on plane {plane}"),
            alone(plane, RotationPath::Chord(0.5)),
        ));
    }
    let seeded: Vec<RotationPath> = (0..planes)
        .map(|_| RotationPath::Chord(rng.random_range(0.0..1.0)))
        .collect();
    family.push((format!("seeded chord {seeded:?}"), seeded));
    for plane in 0..planes {
        for turn in [-1.0, 0.5, 2.0] {
            family.push((
                format!("angle x{turn} on plane {plane}"),
                alone(plane, RotationPath::Angle(turn)),
            ));
        }
    }
    let seeded: Vec<RotationPath> = (0..planes)
        .map(|_| RotationPath::Angle(rng.random_range(-1.0..2.0)))
        .collect();
    family.push((format!("seeded angle {seeded:?}"), seeded));
    family
}

/// The three numbers of the certificate, each rounded up: `L_σ`, the largest row norm
/// `max_i ‖(W₂)_{i,:}‖₂` and the largest held-out norm `r = max ‖x‖₂`.
struct CertificateScales {
    lipschitz: f64,
    write_row_norm: f64,
    radius: f64,
}

fn certificate_scales(native: &NativeMlp, held_out: ArrayView2<'_, f64>) -> CertificateScales {
    let squared = native
        .activation()
        .slope_bound_squared()
        .expect("the activation owner bounds its slope");
    // The root is correctly rounded. It bounds `√squared` exactly when `root² ≥ squared`, whose
    // sign one fused multiply-add decides exactly. ReLU's squared bound is 1, whose root is 1.
    let root = squared.sqrt();
    let lipschitz = if root.mul_add(root, -squared) >= 0.0 {
        root
    } else {
        up(root)
    };
    CertificateScales {
        lipschitz,
        write_row_norm: largest_row_norm(native.write_out()),
        radius: largest_row_norm(held_out),
    }
}

fn largest_row_norm(rows: ArrayView2<'_, f64>) -> f64 {
    rows.outer_iter()
        .map(|row| norm_upper(row.iter().copied()))
        .fold(0.0_f64, f64::max)
}

/// Bounds on `‖U diag(m) R − W*‖₂ ≤ ‖U diag(m) R − W*‖_F` from the stored tensors: the measured
/// part, the Frobenius norm of the computed difference (the exact difference of two floats lies
/// within one step of the rounded one, and is zero when it is), and the formation band of
/// `U diag(m) R`. A term of the product rounds once scaling `u_jc m_c` and `C` times through the
/// product with `R`, `γ_{C+1} |U| |m| |R|` in all, and the computed magnitude takes as many
/// roundings of nonnegative terms. Both are rounded up.
fn read_in_gap(
    factor: &ExactFactor,
    mask: ArrayView1<'_, f64>,
    reference: ArrayView2<'_, f64>,
) -> (f64, f64) {
    let roundings = factor.components() + 1;
    let edited = (&factor.write() * &mask).dot(&factor.read());
    let magnitude = (&factor.write().mapv(f64::abs) * &mask.mapv(f64::abs))
        .dot(&factor.read().mapv(f64::abs));
    let measured = norm_upper(edited.iter().zip(reference.iter()).map(|(&value, &target)| {
        let difference = (value - target).abs();
        if difference == 0.0 { 0.0 } else { up(difference) }
    }));
    let growth = up(accumulation_growth(roundings));
    let band = norm_upper(
        magnitude
            .iter()
            .map(|&value| up(growth * inflated(value, roundings))),
    );
    (measured, band)
}

/// The block executed from its summed input, and per output entry a bound on its distance from
/// the exact block function of the same tensors, given `summed_band` on the summed input.
///
/// With `ẑ` the executed and `z` the exact summed input, `â = σ̂(ẑ)` and `a = σ(z)`:
/// `|â − a| ≤ |σ̂(ẑ) − σ(ẑ)| + L_σ |ẑ − z|`, the activation owner's value bound plus the
/// propagated band. The write `ŵ = fl(â W₂ᵀ + b₂)` is within [`affine_stage_band`] of
/// `â W₂ᵀ + b₂`, which is within `|â − a| |W₂|ᵀ` of `a W₂ᵀ + b₂` (`H` rounded steps over
/// nonnegative terms, then widened). The residual `ŷ = fl(ŵ + x)` rounds once, by at most
/// `γ₁ |ŷ|`. The owner's value is asserted equal to the executed activation bit for bit, so its
/// bound is about the value the block used.
fn executed_output(
    native: &NativeMlp,
    inputs: ArrayView2<'_, f64>,
    summed_input: ArrayView2<'_, f64>,
    summed_band: ArrayView2<'_, f64>,
    lipschitz: f64,
) -> (Array2<f64>, Array2<f64>) {
    let activation = native.activation();
    let activations = native
        .activate(summed_input)
        .expect("the native activation evaluates");
    let mut activation_error = Array2::<f64>::zeros(activations.raw_dim());
    for ((slot, &point), (&band, &executed)) in activation_error
        .iter_mut()
        .zip(summed_input.iter())
        .zip(summed_band.iter().zip(activations.iter()))
    {
        let mut value = [0.0];
        let mut bound = [0.0];
        gaussian_hermite_coefficients(activation, point, 0.0, &mut value, &mut bound)
            .expect("the activation owner bounds its value");
        assert_eq!(
            value[0].to_bits(),
            executed.to_bits(),
            "the owner's value bound must be about the executed activation at {point:e}"
        );
        *slot = up(bound[0] + up(lipschitz * band));
    }
    let output = native
        .write(inputs, activations.view())
        .expect("the native write composes");
    let written_band = affine_stage_band(
        native.write_out(),
        Some(native.bias_out()),
        activations.view(),
    )
    .expect("the write's band composes");
    let propagated = activation_error.dot(&native.write_out().mapv(f64::abs).t());
    let hidden = native.hidden_width();
    let residual = up(accumulation_growth(1));
    let band = Array2::from_shape_fn(output.raw_dim(), |entry| {
        up(up(written_band[entry] + inflated(propagated[entry], hidden))
            + up(residual * output[entry].abs()))
    });
    (output, band)
}

fn bitwise_equal(left: &Array2<f64>, right: &Array2<f64>) -> bool {
    left.dim() == right.dim()
        && left
            .iter()
            .zip(right.iter())
            .all(|(a, b)| a.to_bits() == b.to_bits())
}

/// One intervention's evidence against one reference.
struct EntryEvidence {
    label: String,
    /// The verdict: the certificate when it meets `ε`, else a held-out refutation, else
    /// unresolved between them.
    verdict: FidelityStatus,
    /// The measured distortion over the held-out family, exact up to its numerical error.
    held_out: FidelityStatus,
    /// The certified bound, whether or not it meets `ε`.
    upper: f64,
    /// The largest held-out `measured − error`, rounded down.
    lower: f64,
}

struct Experiment<'a> {
    teacher: &'a PlantedTeacher,
    program: &'a ComponentMlp,
    planes: &'a [RecoveredPlane],
    scales: CertificateScales,
    zero_weight: Array2<f64>,
}

impl Experiment<'_> {
    fn evidence(
        &self,
        reference: Reference,
        reference_planes: &[PlaneRotation],
        label: &str,
        controls: &[RotationPath],
    ) -> EntryEvidence {
        let native = &self.teacher.native;
        let inputs = self.teacher.held_out.view();
        let mask = component_mask(native.width(), self.planes, controls);
        let target = native_edit(native.read_in(), reference_planes, controls);
        let factor = self.program.read_in();

        let (measured, band) = read_in_gap(factor, mask.view(), target.view());
        let lift = up(self.scales.lipschitz * self.scales.write_row_norm);
        let upper = up(up(lift * up(measured + band)) * self.scales.radius);
        let band_error = up(up(lift * band) * self.scales.radius);

        let masked = ComponentMask::Components(mask.view());
        let summed = self
            .program
            .summed_input(inputs, masked)
            .expect("the component summed input composes");
        let summed_band = factored_edit_stage_band(
            self.zero_weight.view(),
            factor.write(),
            mask.view(),
            factor.read().t(),
            Some(native.bias_in()),
            inputs,
        )
        .expect("the component summed input's band composes");
        let (component, component_band) = executed_output(
            self.program.native(),
            inputs,
            summed.view(),
            summed_band.view(),
            self.scales.lipschitz,
        );
        let executed = self
            .program
            .execute(
                inputs,
                MlpMask {
                    read_in: masked,
                    write_out: ComponentMask::AllOn,
                },
            )
            .expect("the component block executes");
        assert!(
            bitwise_equal(&executed, &component),
            "{label}: the banded component output must be the executed one"
        );

        let edited = NativeMlp::new(
            target,
            native.bias_in().to_owned(),
            native.write_out().to_owned(),
            native.bias_out().to_owned(),
            native.activation(),
        )
        .expect("the edited block keeps the teacher's shapes");
        let edited_summed = edited
            .summed_input(inputs)
            .expect("the edited summed input composes");
        let edited_summed_band =
            affine_stage_band(edited.read_in(), Some(edited.bias_in()), inputs)
                .expect("the edited summed input's band composes");
        let (native_output, native_band) = executed_output(
            &edited,
            inputs,
            edited_summed.view(),
            edited_summed_band.view(),
            self.scales.lipschitz,
        );
        assert!(
            bitwise_equal(
                &edited.execute(inputs).expect("the edited block executes"),
                &native_output
            ),
            "{label}: the banded native output must be the executed one"
        );

        let subtraction = up(accumulation_growth(1));
        let mut largest = (0.0_f64, 0.0_f64, (0, 0));
        let mut refutation = (f64::NEG_INFINITY, 0.0_f64, 0.0_f64, (0, 0));
        for (entry, &value) in component.indexed_iter() {
            let distortion = (value - native_output[entry]).abs();
            let error = up(
                up(component_band[entry] + native_band[entry]) + up(subtraction * distortion),
            );
            if distortion > largest.0 {
                largest.0 = distortion;
                largest.2 = entry;
            }
            largest.1 = largest.1.max(error);
            let lower = (distortion - error).next_down();
            if lower > refutation.0 {
                refutation = (lower, distortion, error, entry);
            }
        }
        let held_out = EvidenceStatus::exact(
            largest.0,
            largest.1,
            ExactBasis::Exhaustive {
                cardinality: component.len() as u64,
            },
            Some(largest.2),
            format!("{reference:?} {label}: the {} held-out rows", inputs.nrows()),
        )
        .expect("a finite held-out distortion");

        let region = format!(
            "{reference:?} {label}: every input with ‖x‖₂ ≤ {:e}",
            self.scales.radius
        );
        let (lower, value, error, witness) = refutation;
        let verdict = if upper <= FIDELITY_TOLERANCE {
            EvidenceStatus::uniform_bound(upper, band_error, region).expect("a finite certificate")
        } else if let Ok(counterexample) =
            EvidenceStatus::counterexample(value, error, FIDELITY_TOLERANCE, witness)
        {
            counterexample
        } else {
            EvidenceStatus::unresolved(
                lower.max(0.0),
                upper,
                Extremum::Supremum,
                Some(witness),
                region,
            )
            .expect("the held-out lower end sits below the certificate")
        };
        EntryEvidence {
            label: label.to_owned(),
            verdict,
            held_out,
            upper,
            lower,
        }
    }
}

/// The two numbers between which the declared tolerance sits.
struct FidelityGap {
    /// The largest certified bound over the family against the planted edits.
    largest_certified: f64,
    /// Over the two controls, the smaller of each one's largest held-out `measured − error`.
    weakest_refutation: f64,
}

/// A13's positive half at width `d` for one activation (#2951 P1, P2, P3, P5).
///
/// # Recovery
///
/// [`recover_plane_rotations`] reads `W₁` at the teacher's declared error. Each planted plane
/// must be admitted by exactly one cluster, whose cosine interval meets `c_k ± η_R`, the
/// enclosure of the true cosine: that is the angle's bar. The cluster must hold one plane with a
/// certified orientation, and the only ambiguity is the winding convention. Its projector lies
/// within `projector_bar + δ(2 + δ)` of `U_k U_kᵀ` (one Davis–Kahan link to `O`, then the planted
/// factor's defect), plus the measurement band. Positive controls under the same bars: the next
/// planted plane is refused by the projector bar (a shuffled plane, at distance one), and its
/// cosine is not admitted (a displaced angle).
///
/// # The certificate
///
/// For read-in tensors `A`, `B` and the exact block function `F_A(x) = x + W₂ σ(A x + b₁) + b₂`,
/// `F_A − F_B = W₂ (σ(A x + b₁) − σ(B x + b₁))`. By Cauchy–Schwarz, then `σ` applied entrywise
/// with Lipschitz constant `L_σ`, each output entry is at most
/// `‖(W₂)_{i,:}‖₂ L_σ ‖(A − B) x‖₂ ≤ max_i ‖(W₂)_{i,:}‖₂ · L_σ · ‖A − B‖₂ · ‖x‖₂`. With
/// `A = U diag(m) R` the program's tensor and `B` the teacher's native edit, `‖A − B‖₂` is
/// bounded from the stored tensors (`read_in_gap`), and `‖x‖₂ ≤ r`, the largest held-out norm.
/// So the bound is a [`EvidenceStatus::UniformBound`] on the exact block distortion over every
/// input of norm at most `r`, the held-out rows among them. Execution rounding is not part of it.
/// For ReLU `L_σ = 1`, since `|max(a, 0) − max(b, 0)| ≤ |a − b|`: its `slope_bound_squared` is 1,
/// whose root is exactly 1. For the exact GELU `L_σ = sup|σ'| = σ'(√2) = 1.1289…`, the root of
/// the owner's upper bound on `σ'(√2)²`, rounded up.
///
/// # The held-out measurement
///
/// The program executes each intervention through its masks, and the teacher's edited block
/// executes natively, on the held-out rows. Each executed output lies within its band of the
/// exact block function (`executed_output`), so the measured distortion over the held-out
/// family is exact up to the two bands and the subtraction's rounding. Its lower end must not
/// exceed the certificate.
///
/// # Verdicts
///
/// An entry is a uniform bound when its certificate is at most `ε`. Otherwise it is a
/// counterexample when some held-out entry's distortion, minus its error, exceeds `ε`, and
/// unresolved between the two when none does.
/// - Against the planted edits every entry is certified, and so is its held-out measurement.
/// - A mis-rotated edit, whose planes turn the other way, and a displaced angle, whose planes
///   carry the next plane's angle (on a chord `m` the parameter gap is
///   `2√2 |sin(Δα/2)| |1 − m|`, C2), each carry at least one counterexample. All-on stays
///   certified against both: all-on fidelity has no power against either, as with the
///   cancelling pair.
///
/// # The declared tolerance
///
/// `ε = 2⁻¹⁰` is the one tolerance the A13 experiment declares, the same as its cancelling pair
/// and its random connections above: a pass and a refutation compare only at one tolerance. The
/// returned gap makes the value immaterial. Every certificate against the planted edits is at
/// most `largest_certified`, and each control holds a held-out distortion whose lower end is at
/// least `weakest_refutation`. The tests assert `largest_certified < ε < weakest_refutation`, so
/// every `ε` in that gap gives the same verdicts: every planted entry certified, each control
/// refuted.
///
/// # The activation
///
/// The body is activation-generic. `L_σ` is the owner's `slope_bound_squared`, and each executed
/// activation's value bound is the owner's `gaussian_hermite_coefficients` at scale zero, whose
/// value is `activate`'s bit for bit. ReLU's value bound is exactly zero, because `max` is exact
/// in IEEE-754. The exact GELU's value `t Φ(t)` (or `φ(t) t/λ` in the left tail) carries the
/// owner's bound with nothing truncated: `Φ` from the probability owner's proven table route,
/// `φ` from its certified density, `λ` from its derived tail ratios, and every product's
/// second-order term (#2946). So every status here is rigorous for both activations.
fn planted_rotation_fidelity(width: usize, activation: GaussianActivation) -> FidelityGap {
    let teacher = plant_teacher(width, activation);
    let count = PLANTED_ANGLES.len();
    let recovery = recover_plane_rotations(teacher.native.read_in(), teacher.declared_error)
        .expect("the planted read-in weight is within its declared error of an orthogonal matrix");
    let kinds: Vec<&RotationClusterKind> =
        recovery.clusters.iter().map(|cluster| &cluster.kind).collect();
    assert_eq!(
        recovery.ambiguities(),
        vec![RotationAmbiguity::Winding],
        "width {width}: distinct planted cosines leave only the winding convention: {kinds:?}"
    );
    assert_eq!(
        recovery.clusters.len(),
        count + 1,
        "width {width}: {count} planes and the fixed space: {kinds:?}"
    );

    let planes = matched_planes(&teacher, &recovery);
    for (plane, recovered) in planes.iter().enumerate() {
        let cluster = &recovery.clusters[recovered.cluster];
        let bar = up(cluster.projector_bar + teacher.plane_defect);
        let (distance, band) = projector_distance(recovered.basis.view(), teacher.plane_basis(plane));
        assert!(
            distance <= up(bar + band),
            "width {width}, plane {plane}: projector distance {distance:e} exceeds the bar {bar:e} \
             plus the band {band:e}"
        );
        let next = (plane + 1) % count;
        let (shuffled, shuffled_band) =
            projector_distance(recovered.basis.view(), teacher.plane_basis(next));
        assert!(
            (shuffled - shuffled_band).next_down() > bar,
            "width {width}, plane {plane}: the same bar {bar:e} must refuse the shuffled plane \
             {next} at distance {shuffled:e}"
        );
        assert!(
            !admits(cluster, teacher.cosines[next], teacher.cosine_defect),
            "width {width}, plane {plane}: the cosine interval {:?} must not admit plane {next}'s \
             cosine {}",
            cluster.cosine_interval,
            teacher.cosines[next]
        );
        assert!(
            planes[..plane]
                .iter()
                .all(|earlier| earlier.cluster != recovered.cluster),
            "width {width}: two planted planes matched one cluster"
        );
    }

    let program = plane_program(&teacher.native, &planes);
    let experiment = Experiment {
        teacher: &teacher,
        program: &program,
        planes: &planes,
        scales: certificate_scales(&teacher.native, teacher.held_out.view()),
        zero_weight: Array2::zeros((teacher.native.hidden_width(), width)),
    };
    let family = declared_family(count);
    let all_on = family
        .iter()
        .position(|(_, controls)| controls.iter().all(|&path| path == RotationPath::Chord(1.0)))
        .expect("the family holds all-on");

    let mut largest_certified = 0.0_f64;
    let mut weakest_refutation = f64::INFINITY;
    for reference in [
        Reference::Planted,
        Reference::MisRotated,
        Reference::DisplacedAngle,
    ] {
        let edits = reference_planes(&teacher, reference);
        let evidence: Vec<EntryEvidence> = family
            .iter()
            .map(|(label, controls)| experiment.evidence(reference, &edits, label, controls))
            .collect();
        for entry in &evidence {
            assert!(
                entry.lower <= entry.upper,
                "width {width}, {reference:?} {}: the held-out lower end {:e} contradicts the \
                 certificate {:e}",
                entry.label,
                entry.lower,
                entry.upper
            );
        }
        match reference {
            Reference::Planted => {
                for entry in &evidence {
                    assert!(
                        matches!(entry.verdict, EvidenceStatus::UniformBound { .. })
                            && entry.verdict.certifies_at_most(FIDELITY_TOLERANCE)
                            && entry.held_out.certifies_at_most(FIDELITY_TOLERANCE),
                        "width {width}, {}: the planted edit must be certified, got {:?} with \
                         held-out {:?}",
                        entry.label,
                        entry.verdict,
                        entry.held_out
                    );
                    largest_certified = largest_certified.max(entry.upper);
                }
            }
            Reference::MisRotated | Reference::DisplacedAngle => {
                assert!(
                    evidence[all_on].verdict.certifies_at_most(FIDELITY_TOLERANCE),
                    "width {width}, {reference:?}: all-on must stay certified, got {:?}",
                    evidence[all_on].verdict
                );
                let refuted = evidence
                    .iter()
                    .filter(|entry| matches!(entry.verdict, EvidenceStatus::Counterexample { .. }))
                    .count();
                assert!(
                    refuted > 0,
                    "width {width}, {reference:?}: the control must be refuted at a held-out \
                     witness"
                );
                weakest_refutation = weakest_refutation.min(
                    evidence
                        .iter()
                        .map(|entry| entry.lower)
                        .fold(f64::NEG_INFINITY, f64::max),
                );
            }
        }
    }
    FidelityGap {
        largest_certified,
        weakest_refutation,
    }
}

#[test]
fn a_planted_rotation_relu_teacher_is_recovered_and_certified_on_held_out_interventions_at_width_8()
{
    let gap = planted_rotation_fidelity(8, GaussianActivation::Relu);
    assert!(
        gap.largest_certified < FIDELITY_TOLERANCE && FIDELITY_TOLERANCE < gap.weakest_refutation,
        "the declared tolerance {FIDELITY_TOLERANCE:e} must sit between the largest certified \
         bound {:e} and the weakest refutation {:e}",
        gap.largest_certified,
        gap.weakest_refutation
    );
    println!(
        "width 8: largest certified {:e} < tolerance {FIDELITY_TOLERANCE:e} < weakest refutation {:e}",
        gap.largest_certified, gap.weakest_refutation
    );
}

#[test]
fn a_planted_rotation_relu_teacher_is_recovered_and_certified_on_held_out_interventions_at_width_512()
{
    let gap = planted_rotation_fidelity(512, GaussianActivation::Relu);
    assert!(
        gap.largest_certified < FIDELITY_TOLERANCE && FIDELITY_TOLERANCE < gap.weakest_refutation,
        "the declared tolerance {FIDELITY_TOLERANCE:e} must sit between the largest certified \
         bound {:e} and the weakest refutation {:e}",
        gap.largest_certified,
        gap.weakest_refutation
    );
    println!(
        "width 512: largest certified {:e} < tolerance {FIDELITY_TOLERANCE:e} < weakest refutation {:e}",
        gap.largest_certified, gap.weakest_refutation
    );
}

#[test]
fn a_planted_rotation_gelu_teacher_is_recovered_and_certified_on_held_out_interventions_at_width_8()
{
    let gap = planted_rotation_fidelity(8, GaussianActivation::ExactGelu);
    assert!(
        gap.largest_certified < FIDELITY_TOLERANCE && FIDELITY_TOLERANCE < gap.weakest_refutation,
        "the declared tolerance {FIDELITY_TOLERANCE:e} must sit between the largest certified \
         bound {:e} and the weakest refutation {:e}",
        gap.largest_certified,
        gap.weakest_refutation
    );
    println!(
        "GELU width 8: largest certified {:e} < tolerance {FIDELITY_TOLERANCE:e} < weakest refutation {:e}",
        gap.largest_certified, gap.weakest_refutation
    );
}

#[test]
fn a_planted_rotation_gelu_teacher_is_recovered_and_certified_on_held_out_interventions_at_width_512()
{
    let gap = planted_rotation_fidelity(512, GaussianActivation::ExactGelu);
    assert!(
        gap.largest_certified < FIDELITY_TOLERANCE && FIDELITY_TOLERANCE < gap.weakest_refutation,
        "the declared tolerance {FIDELITY_TOLERANCE:e} must sit between the largest certified \
         bound {:e} and the weakest refutation {:e}",
        gap.largest_certified,
        gap.weakest_refutation
    );
    println!(
        "GELU width 512: largest certified {:e} < tolerance {FIDELITY_TOLERANCE:e} < weakest refutation {:e}",
        gap.largest_certified, gap.weakest_refutation
    );
}
