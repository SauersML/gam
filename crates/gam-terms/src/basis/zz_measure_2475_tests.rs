//! Regression suite for the #2475 spherical-kernel fixes.
//!
//! The untruncated Sobolev kernel at `m = 1` has no finite coincident-point
//! value, so no public Gram-matrix constructor may reach its epsilon-floored
//! scalar evaluator. The explicit truncated kernel remains the honest way to
//! choose a finite resolution.
//!
//! The remaining tests preserve the independent spherical-jet regressions
//! discovered while tracing the original shared-floor defect.

use super::sphere_half_angle::HalfAngleSeparation;
use super::sphere_kernels::{
    wahba_sphere_kernel_sobolev, wahba_sphere_kernel_sobolev_derivative_dhav,
};
use super::sphere_spectral::{
    sobolev_s2_truncated_coefficients, sphere_truncated_spectral_derivative_eval,
};

/// This suite sweeps `cos γ` as its abscissa, because that is the coordinate
/// the #2475 floors were expressed in. The kernels now take the half-angle pair
/// `(u, v) = (sin²(γ/2), cos²(γ/2))` directly (#2489), so the sweep converts at
/// the boundary. Both halves are taken from the side where Sterbenz's lemma
/// makes them exact — `1 − cos γ` for `cos γ ∈ [0.5, 1]`, `1 + cos γ` for
/// `cos γ ∈ [−1, −0.5]` — which is the best available reading of a cosine, and
/// is bit-identical to what the shipped code did internally before the split.
fn sep_from_cos(cos_gamma: f64) -> HalfAngleSeparation {
    let cos_g = cos_gamma.clamp(-1.0, 1.0);
    HalfAngleSeparation {
        u: (1.0 - cos_g) * 0.5,
        v: (1.0 + cos_g) * 0.5,
    }
}

fn wahba_sphere_kernel_sobolev_closed_form(cos_gamma: f64, m: usize) -> f64 {
    wahba_sphere_kernel_sobolev(sep_from_cos(cos_gamma), m)
}

fn wahba_sphere_kernel_sobolev_derivative_dcos(cos_gamma: f64, m: usize) -> f64 {
    // `du/d(cos γ) = −1/2`.
    wahba_sphere_kernel_sobolev_derivative_dhav(sep_from_cos(cos_gamma), m) * (-0.5)
}

const FOUR_PI: f64 = 4.0 * std::f64::consts::PI;
/// The floor that used to be shared by all four sites.
const LEGACY_FLOOR: f64 = f64::EPSILON * 1.0e-4;
/// Truncated Sobolev diagonal `K_L(0)`, summed directly.
fn sobolev_truncated_diagonal(m: i32, l_max: usize) -> f64 {
    (1..=l_max)
        .map(|l| {
            let lf = l as f64;
            (2.0 * lf + 1.0) / (FOUR_PI * (lf * (lf + 1.0)).powi(m))
        })
        .sum()
}

// ---------------------------------------------------------------------------
// 2. NOT A DEFECT: Sobolev m = 2, 3 already land on their exact limits.
// ---------------------------------------------------------------------------

#[test]
fn zz_measure_2475_sobolev_m2_m3_coincident_are_already_exact() {
    const ZETA3: f64 = 1.202_056_903_159_594_3;
    let cases = [
        (2usize, 1.0 / FOUR_PI),
        (3usize, (2.0 * ZETA3 - 2.0) / FOUR_PI),
    ];
    println!(
        "\n{:>2} {:>24} {:>24} {:>10}",
        "m", "closed-form limit", "shipped at cos=1", "rel"
    );
    for (m, want) in cases {
        let got = wahba_sphere_kernel_sobolev_closed_form(1.0, m);
        let rel = (got - want).abs() / want.abs();
        println!("{m:>2} {want:>24.16e} {got:>24.16e} {rel:>10.2e}");
        assert!(
            rel < 1e-15,
            "Sobolev m={m} coincident value {got:.17e} should already equal its \
             exact limit {want:.17e} (rel {rel:.3e})"
        );
        // Cross-check the limit itself against the kernel's own spectral series.
        let spectral = sobolev_truncated_diagonal(m as i32, 200_000);
        assert!(
            (spectral - want).abs() / want.abs() < 1e-9,
            "m={m}: limit {want:.17e} vs spectral {spectral:.17e}"
        );
    }
}

// ---------------------------------------------------------------------------
// 4. DELETED: the Sobolev DERIVATIVE floors were unreachable behind the pole
//    guard, so they were dead arithmetic rather than a regularization.
// ---------------------------------------------------------------------------

#[test]
fn zz_measure_2475_sobolev_derivative_floors_were_unreachable() {
    // `wahba_sphere_kernel_sobolev_derivative_dcos` only enters the closed form
    // inside `|cos γ| <= 1 - POLE_LIMIT_THRESHOLD` (1e-10), which bounds
    // `u = (1 - cos γ)/2` into `[5e-11, 1 - 5e-11]`. Both bounds are nine
    // orders above the `EPSILON * 1e-4` floor the closed form used to apply
    // (a factor of 2.3e9), so neither `.max()` could ever change a value.
    const POLE_LIMIT_THRESHOLD: f64 = 1.0e-10;
    let u_min = (1.0 - (1.0 - POLE_LIMIT_THRESHOLD)) * 0.5;
    let u_max = (1.0 - (-(1.0 - POLE_LIMIT_THRESHOLD))) * 0.5;
    println!(
        "\n  closed form sees u in [{u_min:.3e}, {u_max:.6}]; the deleted floor was \
         {LEGACY_FLOOR:.3e}, smaller than the tighter end by {:.1e}x\n",
        u_min / LEGACY_FLOOR
    );
    assert!(u_min / LEGACY_FLOOR > 1e9);
    assert!((1.0 - u_max) / LEGACY_FLOOR > 1e9);

    // And the dispatcher stays finite right across the pole boundary and at the
    // poles themselves, where it hands off to the bounded spectral limit.
    for m in 1..=4 {
        for &x in &[
            -1.0,
            -1.0 + 0.5 * POLE_LIMIT_THRESHOLD,
            -1.0 + POLE_LIMIT_THRESHOLD,
            -0.5,
            0.0,
            0.5,
            1.0 - POLE_LIMIT_THRESHOLD,
            1.0 - 0.5 * POLE_LIMIT_THRESHOLD,
            1.0,
        ] {
            let v = wahba_sphere_kernel_sobolev_derivative_dcos(x, m);
            assert!(
                v.is_finite(),
                "Sobolev derivative m={m} at cos γ = {x} is not finite: {v}"
            );
        }
    }
}

// ---------------------------------------------------------------------------
// 5. The ANTIPODE is not a pole. The derivative's pole guard used to be
//    two-sided, and the spectral branch it routed `cos γ = -1` into does not
//    converge there at `m = 1`.
// ---------------------------------------------------------------------------

/// `1/(8π)` — the exact `dK/d(cos γ)` at the antipode for Sobolev `m = 1` and
/// `m = 2`, from `dK/du → -1/(4π)` and `du/d(cos γ) = -1/2`.
const ANTIPODAL_DERIVATIVE_M1_M2: f64 = 3.978_873_577_297_383_4e-2;

/// `(π²/6 - 1)/(8π)` — the same quantity at `m = 3`, where
/// `dK/du → (1 - π²/6)/(4π)`: the `-Li₂(u)/u` term tends to `-π²/6`, the
/// `-ln(u)/(1-u)` term to `+1`, and the `-ln(u)·ln(1-u)/u` term to `0` like
/// `v·ln v`.
const ANTIPODAL_DERIVATIVE_M3: f64 = 2.566_111_117_681_352_5e-2;

#[test]
fn zz_measure_2475_antipodal_sobolev_derivative_is_the_closed_form_limit() {
    // Values cross-checked against 50-digit `mpmath` differentiation of the
    // closed-form kernel: at cos γ = -1 it reports 0.039788735772973834 for
    // m ∈ {1,2} and 0.025661111176813525 for m = 3, and the approach from
    // cos γ = -1 + 1e-12 agrees to 13 significant figures, so the point is
    // interior and smooth rather than a limit being taken.
    println!(
        "\n{:>3} {:>26} {:>26} {:>11}",
        "m", "at cos γ = -1", "exact limit", "rel"
    );
    for (m, want) in [
        (1, ANTIPODAL_DERIVATIVE_M1_M2),
        (2, ANTIPODAL_DERIVATIVE_M1_M2),
        (3, ANTIPODAL_DERIVATIVE_M3),
    ] {
        let got = wahba_sphere_kernel_sobolev_derivative_dcos(-1.0, m);
        let rel = (got - want).abs() / want.abs();
        println!("{m:>3} {got:>26.15e} {want:>26.15e} {rel:>11.2e}");
        assert!(
            rel < 1e-14,
            "Sobolev m={m} antipodal derivative is {got:.15e}, exact limit is \
             {want:.15e} (rel {rel:.3e}). The spectral pole branch used to \
             return -162.9747 here at m=1 — the L-th partial sum of a DIVERGENT \
             alternating series, equal to -l_max/(8π)."
        );
    }
    println!();
}

#[test]
fn zz_measure_2475_antipodal_derivative_has_no_step_across_the_old_guard() {
    // The old guard switched representation at |cos γ| = 1 - 1e-10. Walking
    // through it from the interior must now be smooth: these kernels have no
    // feature at the antipode, so any step is an artefact of the routing.
    const T: f64 = 1.0e-10;
    for m in 1..=4 {
        let mut prev = wahba_sphere_kernel_sobolev_derivative_dcos(-1.0 + 10.0 * T, m);
        for &x in &[-1.0 + 2.0 * T, -1.0 + T, -1.0 + 0.5 * T, -1.0 + 1e-14, -1.0] {
            let got = wahba_sphere_kernel_sobolev_derivative_dcos(x, m);
            let step = (got - prev).abs() / prev.abs().max(1e-300);
            assert!(
                step < 1e-6,
                "Sobolev m={m} derivative steps by {step:.3e} relative at \
                 cos γ = {x} ({prev:.9e} -> {got:.9e}); the antipode is an \
                 interior point and must not carry a seam"
            );
            prev = got;
        }
    }
}

#[test]
fn zz_measure_2475_the_antipodal_spectral_sum_is_l_max_not_a_limit() {
    // Why the old branch could not be repaired by widening `l_max`: at m=1 the
    // differentiated Legendre terms GROW, so the sum has no limit to converge
    // to and its partial sum is exactly `-L/(8π)`. Reproduced here at three
    // truncation degrees — a genuine limit would not move.
    let eighth_pi = 1.0 / (8.0 * std::f64::consts::PI);
    println!("\n  m=1 antipodal spectral partial sums (the branch's own arithmetic):");
    for l_max in [256_usize, 1024, 4096] {
        let mut sum = 0.0_f64;
        for l in 1..=l_max {
            let ell = l as f64;
            let sign = if l % 2 == 0 { -1.0 } else { 1.0 };
            let p_l_prime = 0.5 * ell * (ell + 1.0) * sign;
            sum += (2.0 * ell + 1.0) / FOUR_PI * p_l_prime / (ell * (ell + 1.0));
        }
        let predicted = -(l_max as f64) * eighth_pi;
        println!("    l_max={l_max:>5}: sum={sum:>16.6} predicted -l_max/(8π)={predicted:>16.6}");
        assert!(
            (sum - predicted).abs() / predicted.abs() < 1e-12,
            "the m=1 antipodal spectral sum should be exactly -l_max/(8π)"
        );
    }
    println!(
        "\n    exact answer at the antipode: {ANTIPODAL_DERIVATIVE_M1_M2:.9e} \
         — the shipped l_max=4096 gave {:.6}\n",
        -4096.0 * eighth_pi
    );
}

/// `dK/d(cos γ)` from 40-digit `mpmath` — the analytic derivative of the
/// closed-form kernel, cross-checked against `mpmath.diff` of the kernel
/// itself to 1e-41 at four interior points, so the table is not circular with
/// the differentiation rule it encodes.
///
/// Spans both ends of the branch. The `cos γ = -1` rows are the ones the old
/// two-sided pole guard could not produce at all.
const SOBOLEV_DERIVATIVE_REFERENCE: [(usize, f64, f64); 33] = [
    (1, -1.0, 0.039788735772973834),
    (1, -0.9999999999, 0.039788735774963271),
    (1, -0.99999999, 0.039788735971917515),
    (1, -0.999, 0.039808640093020344),
    (1, -0.9, 0.041882879761025088),
    (1, -0.5, 0.053051647697298445),
    (1, 0.0, 0.079577471545947668),
    (1, 0.5, 0.15915494309189534),
    (1, 0.9, 0.79577471545947686),
    (1, 0.999, 79.577471545947597),
    (1, 0.99999999, 7957747.1146090032),
    (2, -1.0, 0.039788735772973834),
    (2, -0.9999999999, 0.039788735773968552),
    (2, -0.99999999, 0.039788735872445674),
    (2, -0.999, 0.039798686273888954),
    (2, -0.9, 0.040817906746232198),
    (2, -0.5, 0.045786023869621704),
    (2, 0.0, 0.055158900038162898),
    (2, 0.5, 0.073545200050883864),
    (2, 0.9, 0.12546989460948413),
    (2, 0.999, 0.30258159039406385),
    (2, 0.99999999, 0.76051505250115565),
    (3, -1.0, 0.025661111176813525),
    (3, -0.9999999999, 0.025661111177101862),
    (3, -0.99999999, 0.025661111205647242),
    (3, -0.999, 0.02566399516138079),
    (3, -0.9, 0.025955749592912204),
    (3, -0.5, 0.027281025216225471),
    (3, 0.0, 0.029407564933744697),
    (3, 0.5, 0.032525947199281328),
    (3, 0.9, 0.037107094521124394),
    (3, 0.999, 0.039718052462181445),
    (3, 0.99999999, 0.03978873392142211),
];

#[test]
fn zz_measure_2475_sobolev_derivative_matches_high_precision_reference() {
    // Before the one-sided guard and the `ln_1p` pairing, this table failed in
    // two distinct places:
    //   * `m=1, cos γ = -1` returned -162.9747 (the divergent spectral sum);
    //   * `m=3, cos γ = 0.99999999` carried 2.4e-6, because `ln(1-u)` was read
    //     off a number within 5e-11 of 1.
    let mut worst = 0.0_f64;
    let mut worst_at = (0usize, f64::NAN);
    println!(
        "\n{:>3} {:>16} {:>26} {:>26} {:>10}",
        "m", "cos γ", "reference", "shipped", "rel"
    );
    for (m, x, want) in SOBOLEV_DERIVATIVE_REFERENCE {
        let got = wahba_sphere_kernel_sobolev_derivative_dcos(x, m);
        let rel = (got - want).abs() / want.abs();
        if rel > worst {
            worst = rel;
            worst_at = (m, x);
        }
        println!("{m:>3} {x:>16} {want:>26.17e} {got:>26.17e} {rel:>10.2e}");
    }
    println!(
        "\n  worst: {worst:.3e} at m={}, cos γ = {}\n",
        worst_at.0, worst_at.1
    );
    assert!(
        worst < 1e-13,
        "Sobolev derivative is off by {worst:.3e} relative at m={}, cos γ = {}",
        worst_at.0,
        worst_at.1
    );
}

// ---------------------------------------------------------------------------
// 6. The Legendre DERIVATIVE recurrence. `P'_ℓ = ℓ(P_{ℓ-1} - x P_ℓ)/(1 - x²)`
//    is a removable 0/0 at the poles, so it carried a floor and a second
//    branch; it also decays like `ε/(1-x²)` long before either fires.
// ---------------------------------------------------------------------------

/// `d/d(cos γ) Σ_{ℓ=1}^{64} c_ℓ P_ℓ(cos γ)` for the truncated Sobolev `m = 2`
/// coefficients, from 40-digit `mpmath`. The `±1` rows use the closed pole
/// value `P'_ℓ(±1) = (±1)^{ℓ+1}·ℓ(ℓ+1)/2`; every other row differentiates the
/// Legendre polynomials themselves, so the table does not assume the identity
/// the code is being checked against.
const TRUNCATED_DERIVATIVE_REFERENCE: [(f64, f64); 15] = [
    (1.0, 0.33833024203025926),
    (-1.0, 0.039176601376466544),
    (0.9999999999, 0.338330237845485),
    (-0.9999999999, 0.039176601442087355),
    (0.99999999, 0.33832982355359548),
    (-0.99999999, 0.039176607938524592),
    (0.999999, 0.33828840165834044),
    (-0.999999, 0.039177257357422111),
    (0.9999, 0.33421830105231071),
    (-0.9999, 0.039239990051586801),
    (0.99, 0.2120348157213505),
    (-0.99, 0.039859419520289469),
    (0.5, 0.073546528978107252),
    (-0.5, 0.045788612629465944),
    (0.0, 0.055157001437210544),
];

#[test]
fn zz_measure_2475_truncated_spectral_derivative_holds_accuracy_into_the_poles() {
    // The quotient form this replaced was measured at 6.5e-8 relative at
    // |cos γ| = 1 - 1e-10 and 7.0e-10 at 1 - 1e-8 — i.e. it had already fallen
    // to eight digits by the point its own pole threshold took over, so the
    // loss sat entirely inside the branch nothing was checking. The bound here
    // is 1e-14, four orders tighter than the worst the old form reached.
    let coeffs = sobolev_s2_truncated_coefficients(64, 2);
    let mut worst = 0.0_f64;
    let mut worst_x = f64::NAN;
    println!(
        "\n{:>18} {:>24} {:>24} {:>10}",
        "cos γ", "reference", "shipped", "rel"
    );
    for (x, want) in TRUNCATED_DERIVATIVE_REFERENCE {
        let got = sphere_truncated_spectral_derivative_eval(x, &coeffs);
        let rel = (got - want).abs() / want.abs();
        if rel > worst {
            worst = rel;
            worst_x = x;
        }
        println!("{x:>18} {want:>24.17e} {got:>24.17e} {rel:>10.2e}");
    }
    println!("\n  worst: {worst:.3e} at cos γ = {worst_x}\n");
    assert!(
        worst < 1e-14,
        "truncated spectral derivative is off by {worst:.3e} relative at cos γ = {worst_x}"
    );
}

#[test]
fn zz_measure_2475_the_deleted_pole_branch_was_a_second_route_to_the_same_number() {
    // The justification for removing the `|cos γ| > 1 - 1e-10` branch: it summed
    // `Σ c_ℓ P'_ℓ(±1)` from the closed value `P'_ℓ(±1) = (±1)^{ℓ+1}·ℓ(ℓ+1)/2`,
    // which for a FINITE sum is exact — and the recurrence that now spans the
    // whole interval lands on the same value unaided. Two routes to one number
    // is what let the primary route's accuracy go unmeasured; this asserts they
    // agree so the deletion cannot be read as a change of answer.
    //
    // (Contrast the Sobolev m=1 case in test 5, where the analogous branch was
    // NOT a second route to the same number, because that sum is infinite and
    // its differentiated form diverges.)
    for m in 1..=3 {
        let coeffs = sobolev_s2_truncated_coefficients(64, m);
        for pole in [1.0_f64, -1.0] {
            let deleted_branch: f64 = (1..=64)
                .map(|ell| {
                    let lf = ell as f64;
                    let sign = if pole < 0.0 && ell % 2 == 0 {
                        -1.0
                    } else {
                        1.0
                    };
                    coeffs[ell] * 0.5 * lf * (lf + 1.0) * sign
                })
                .sum();
            let recurrence = sphere_truncated_spectral_derivative_eval(pole, &coeffs);
            let rel = (recurrence - deleted_branch).abs() / deleted_branch.abs();
            println!(
                "  m={m} cos γ={pole:>4}: recurrence {recurrence:>23.16e}  \
                 deleted branch {deleted_branch:>23.16e}  rel {rel:.2e}"
            );
            assert!(
                rel < 1e-14,
                "m={m}: the recurrence disagrees with the deleted pole branch at \
                 cos γ = {pole} by {rel:.3e} relative"
            );
        }
    }
}
