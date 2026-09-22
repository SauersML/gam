use approx::assert_abs_diff_eq;
use gam_sae::manifold::{
    SaeDecrementAgainstResolution, SaeInnerKktScaleBlock, SaeInnerKktScaleError,
    SaeInstalledInnerKktAudit, SaeManifoldTerm, SaeParameterSpaceKktAudit,
};
use gam_solve::arrow_schur::ArrowSchurSystem;

/// Row replication changes gradient-space L2 norms but must not change the
/// componentwise diagonal-scaled residual. Decoder gradient and curvature both
/// grow with the number of rows; coordinate blocks remain row-local.
#[test]
fn parameter_scale_is_intensive_under_row_replication() {
    fn fixture(rows: usize) -> ArrowSchurSystem {
        let mut system = ArrowSchurSystem::new(rows, 1, 2);
        for row in &mut system.rows {
            row.htt[[0, 0]] = 4.0;
            row.gt[0] = 0.08;
        }
        system.hbb[[0, 0]] = 10.0 * rows as f64;
        system.hbb[[1, 1]] = 20.0 * rows as f64;
        system.gb[0] = 0.3 * rows as f64;
        system.gb[1] = -0.2 * rows as f64;
        system
    }

    let one_row = fixture(1);
    let many_rows = fixture(64);
    let raw_norm_sq = |system: &ArrowSchurSystem| {
        system
            .rows
            .iter()
            .flat_map(|row| row.gt.iter())
            .chain(system.gb.iter())
            .map(|gradient| gradient * gradient)
            .sum::<f64>()
    };
    assert!(raw_norm_sq(&many_rows) > 1_000.0 * raw_norm_sq(&one_row));

    let one_scaled = SaeManifoldTerm::system_scaled_grad_max(&one_row)
        .expect("positive diagonal curvature");
    let many_scaled = SaeManifoldTerm::system_scaled_grad_max(&many_rows)
        .expect("positive diagonal curvature");
    assert_abs_diff_eq!(one_scaled, 0.03, epsilon = 1.0e-15);
    assert_abs_diff_eq!(many_scaled, one_scaled, epsilon = 1.0e-15);
}

/// A missing curvature scale may not be skipped: zero curvature with nonzero
/// gradient is a typed unresolved measurement, not a zero contribution to the max.
#[test]
fn parameter_scale_refuses_unscaled_gradient() {
    let mut system = ArrowSchurSystem::new(0, 0, 1);
    system.gb[0] = 1.0;
    let error = SaeManifoldTerm::system_scaled_grad_max(&system)
        .expect_err("nonzero gradient without curvature must be unresolved");
    assert!(matches!(
        error,
        SaeInnerKktScaleError::InvalidCurvature {
            block: SaeInnerKktScaleBlock::SharedDecoder,
            component: 0,
            gradient: 1.0,
            curvature: 0.0,
        }
    ));
}

/// #2933 F08 — a diagonal-scaled gradient is not a remaining Newton displacement,
/// so the installed-state audit may not certify on it. The shared decoder block
/// `H = c·[[1, 1−ε], [1−ε, 1]]` at the state `θ − θ* = (1, −1)` has gradient
/// `g = H·(1, −1) = cε·(1, −1)`. Its diagonal-scaled residual is `ε`, while the
/// displacement `H⁻¹g`, solved here in closed form, is `(1, −1)`. The curvature
/// scale `c` keeps the raw gradient-norm limb above its bound, so the diagonal limb
/// is the only one that could accept this state.
#[test]
fn audit_does_not_certify_on_a_diagonal_scaled_residual_2933_f08() {
    let epsilon = 1.0e-10;
    let curvature = 1.0e6;
    let mut system = ArrowSchurSystem::new(0, 0, 2);
    system.hbb[[0, 0]] = curvature;
    system.hbb[[0, 1]] = curvature * (1.0 - epsilon);
    system.hbb[[1, 0]] = curvature * (1.0 - epsilon);
    system.hbb[[1, 1]] = curvature;
    system.gb[0] = system.hbb[[0, 0]] - system.hbb[[0, 1]];
    system.gb[1] = system.hbb[[1, 0]] - system.hbb[[1, 1]];

    let determinant =
        system.hbb[[0, 0]] * system.hbb[[1, 1]] - system.hbb[[0, 1]] * system.hbb[[1, 0]];
    let displacement = [
        (system.hbb[[1, 1]] * system.gb[0] - system.hbb[[0, 1]] * system.gb[1]) / determinant,
        (system.hbb[[0, 0]] * system.gb[1] - system.hbb[[1, 0]] * system.gb[0]) / determinant,
    ];
    assert!(
        (displacement[0] - 1.0).abs() <= 1.0e-3 && (displacement[1] + 1.0).abs() <= 1.0e-3,
        "the fixture must sit one unit from its optimum: displacement {displacement:?}"
    );

    let scaled = SaeManifoldTerm::system_scaled_grad_max(&system)
        .expect("positive diagonal curvature");
    assert!(
        scaled <= 2.0 * epsilon,
        "the diagonal must see only ε: scaled residual {scaled:e}"
    );
    let raw_gradient_norm = system.gb.dot(&system.gb).sqrt();
    let stationarity_bound = 1.0e-5 * (1.0 + 2.0_f64.sqrt());
    assert!(
        raw_gradient_norm > stationarity_bound,
        "the raw limb must not be the one deciding: ‖g‖ {raw_gradient_norm:e} vs {stationarity_bound:e}"
    );

    let audit = SaeInstalledInnerKktAudit {
        raw_gradient_norm,
        quotient_gradient_norm: raw_gradient_norm,
        stationarity_bound,
        parameter_space: SaeParameterSpaceKktAudit::Resolved {
            scaled_gradient_max: scaled,
            stationarity_bound: 1.0e-5 * (1.0 + 1.0),
        },
        newton_decrement_relative: Err("not priced in this unit fixture".to_string()),
    };
    assert!(
        !audit.certifies(),
        "a state one unit from its optimum must not certify: displacement {displacement:?}, \
         diagonal-scaled residual {scaled:e}"
    );
}

/// #2263 — the native inner solve accepts a state on the affine-invariant
/// Newton decrement, so the zero-step audit must accept that currency too.
/// The numbers are the natively certified replay measured by guarded pool job
/// 541721 at 5e8436c44: ‖g‖ 55.6 along smoothing directions with curvature
/// ~5e12, against a KKT band of 5.3e-5, while the decrement was 1.34e-11 of
/// its scale. The perturbed decoder measured a decrement equal to its scale.
///
/// Since #3355 the audit's field is a [`SaeDecrementAgainstResolution`], ½λ² in
/// units of the criterion's own resolution `0.5/n_eff`, with the bar at one
/// unit, so the fixture states each decrement in those units through the only
/// constructor: `measure(value / n_eff, F, Some(n_eff))` measures exactly
/// `value`. The retired `|f| + 1` scale put the perturbed decoder at `1.0`
/// against a `1e-8` tolerance; in resolution units a decrement at the bar
/// CERTIFIES, so the descending state is placed at two units, the same point
/// the newtype's own `above` control uses.
#[test]
fn audit_accepts_the_native_newton_decrement_certificate() {
    const N_EFF: usize = 6_400;
    let in_resolution_units = |value: f64| {
        SaeDecrementAgainstResolution::measure(value / N_EFF as f64, 0.0, Some(N_EFF))
    };
    let stiff_optimum = SaeInstalledInnerKktAudit {
        raw_gradient_norm: 55.6,
        quotient_gradient_norm: 55.6,
        stationarity_bound: 5.3e-5,
        parameter_space: SaeParameterSpaceKktAudit::Resolved {
            scaled_gradient_max: 2.4e6,
            stationarity_bound: 2.0e-5,
        },
        newton_decrement_relative: Ok(in_resolution_units(1.34e-11)),
    };
    assert!(stiff_optimum.certifies());

    let descending = SaeInstalledInnerKktAudit {
        newton_decrement_relative: Ok(in_resolution_units(2.0)),
        ..stiff_optimum.clone()
    };
    assert!(!descending.certifies());

    let unpriced = SaeInstalledInnerKktAudit {
        newton_decrement_relative: Err("deflated evidence factorization failed".to_string()),
        ..stiff_optimum
    };
    assert!(!unpriced.certifies());
}
