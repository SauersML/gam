// #1561 dispersion-tower behavior tests, split from `tests.rs` to respect the
// crate-wide source-file length budget. Same conventions: `super::*` resolves
// to the parent `gamlss` module's flat re-exports; helpers exercised only here
// are imported at fn scope so a non-test `--lib` build never sees them.

use super::*;
use super::tests::assert_rel_close;
// `Order2::value()` resolves through the shared `JetField` supertrait.
use gam_math::nested_dual::JetField as _;
use ndarray::{Array1, Array2};

/// #1561 order-3 dispersion tower, part 1: the third-order tower's value,
/// gradient, and Hessian channels must reproduce the `Order2<2>` tower (same
/// expression structure, one order deeper) on every family arm.
#[test]
fn dispersion_order3_tower_matches_order2_through_second_order() {
    use super::test_support::{dispersion_eta_nll_order2, dispersion_eta_nll_order3};
    let cases = [
        (DispersionFamilyKind::NegativeBinomial, 6.0, 2.0, 3.0, 1.3),
        (DispersionFamilyKind::NegativeBinomial, 0.0, -1.5, 0.4, 0.7),
        (DispersionFamilyKind::Gamma, 0.2, -2.0, 0.8, 0.9),
        (DispersionFamilyKind::Gamma, 9.0, 1.7, 2.0, 1.1),
        (DispersionFamilyKind::Beta, 0.02, -3.0, 1.6, 0.8),
        (DispersionFamilyKind::Beta, 0.98, 3.0, -0.5, 1.4),
        (DispersionFamilyKind::Tweedie { p: 1.5 }, 4.2, 0.9, -0.6, 1.0),
        (DispersionFamilyKind::Tweedie { p: 1.5 }, 0.0, 0.3, 0.2, 1.2),
    ];
    for (kind, y, em, ed, w) in cases {
        let t2 = dispersion_eta_nll_order2(kind, y, em, ed, w);
        let t3 = dispersion_eta_nll_order3(kind, y, em, ed, w);
        let label = format!("{kind:?} y={y} em={em} ed={ed}");
        assert_rel_close(&format!("{label} value"), t3.v, t2.value(), 1e-12);
        let g2 = t2.g();
        let h2 = t2.h();
        for a in 0..2 {
            assert_rel_close(&format!("{label} grad[{a}]"), t3.g[a], g2[a], 1e-11);
            for b in 0..2 {
                assert_rel_close(
                    &format!("{label} hess[{a}][{b}]"),
                    t3.h[a][b],
                    h2[a][b],
                    1e-11,
                );
            }
        }
    }
}

/// #1561 order-3 dispersion tower, part 2: every `t3` channel must equal the
/// centered finite difference of the observed Hessian channels in both
/// predictor directions, on every family arm.
#[test]
fn dispersion_order3_third_channels_match_finite_difference() {
    use super::dispersion_family::dispersion_row_observed_hessian_weights;
    use super::test_support::dispersion_eta_nll_order3;
    let cases = [
        (DispersionFamilyKind::NegativeBinomial, 6.0, 2.0, 3.0, 1.3),
        (DispersionFamilyKind::Gamma, 9.0, 1.7, 2.0, 1.1),
        (DispersionFamilyKind::Beta, 0.3, -0.7, 1.6, 0.8),
        (DispersionFamilyKind::Beta, 0.92, 1.4, 0.9, 1.4),
        (DispersionFamilyKind::Tweedie { p: 1.5 }, 4.2, 0.9, -0.6, 1.0),
    ];
    let h = 1e-5;
    for (kind, y, em, ed, w) in cases {
        let tower = dispersion_eta_nll_order3(kind, y, em, ed, w);
        for (axis, name) in [(0usize, "d/d_eta_mu"), (1usize, "d/d_eta_d")] {
            let (emp, edp) = if axis == 0 { (em + h, ed) } else { (em, ed + h) };
            let (emm, edm) = if axis == 0 { (em - h, ed) } else { (em, ed - h) };
            let plus = dispersion_row_observed_hessian_weights(kind, y, emp, edp, w);
            let minus = dispersion_row_observed_hessian_weights(kind, y, emm, edm, w);
            let fd = [
                (plus.0 - minus.0) / (2.0 * h),
                (plus.1 - minus.1) / (2.0 * h),
                (plus.2 - minus.2) / (2.0 * h),
            ];
            let analytic = [
                tower.t3[0][0][axis],
                tower.t3[0][1][axis],
                tower.t3[1][1][axis],
            ];
            for (channel, (a, f)) in analytic.iter().zip(fd.iter()).enumerate() {
                let scale = a.abs().max(f.abs()).max(1e-6);
                assert!(
                    (a - f).abs() / scale < 5e-5,
                    "{kind:?} y={y} {name} t3 channel {channel}: analytic {a:.9e} vs FD {f:.9e}"
                );
            }
        }
    }
}

/// #1561 order-3 dispersion tower, part 3: the assembled joint-Hessian
/// directional derivative `D_beta H_L[u]` must equal the centered finite
/// difference of `exact_newton_joint_hessian_with_specs` along `u` — the
/// full-assembly gate (designs, offsets, block packing, cross block) whose
/// prior absence degraded the Firth/Jeffreys gradient to zero and desynced
/// the inner joint-Newton merit from its KKT residual.
#[test]
fn dispersion_joint_hessian_directional_matches_finite_difference() {
    use super::dispersion_family::DispersionGlmLocationScaleFamily;
    use crate::custom_family::CustomFamily as _;
    use gam_linalg::matrix::DesignMatrix;

    let n = 9usize;
    let p_mean = 3usize;
    let p_disp = 2usize;
    let x_mean = Array2::from_shape_fn((n, p_mean), |(i, j)| {
        let t = i as f64 / (n as f64 - 1.0);
        [1.0, t - 0.5, (t - 0.5) * (t - 0.5)][j]
    });
    let x_disp = Array2::from_shape_fn((n, p_disp), |(i, j)| {
        let t = i as f64 / (n as f64 - 1.0);
        [1.0, t - 0.5][j]
    });
    let offset_mean = Array1::from_shape_fn(n, |i| 0.01 * i as f64);
    let offset_disp = Array1::from_shape_fn(n, |i| -0.02 * i as f64);
    let make_specs = || {
        vec![
            ParameterBlockSpec {
                name: "mu".to_string(),
                design: DesignMatrix::Dense(gam_linalg::matrix::DenseDesignMatrix::from(
                    x_mean.clone(),
                )),
                offset: offset_mean.clone(),
                penalties: Vec::new(),
                nullspace_dims: Vec::new(),
                initial_log_lambdas: Array1::zeros(0),
                initial_beta: None,
                gauge_priority: 100,
                jacobian_callback: None,
                stacked_design: None,
                stacked_offset: None,
            },
            ParameterBlockSpec {
                name: "log_disp".to_string(),
                design: DesignMatrix::Dense(gam_linalg::matrix::DenseDesignMatrix::from(
                    x_disp.clone(),
                )),
                offset: offset_disp.clone(),
                penalties: Vec::new(),
                nullspace_dims: Vec::new(),
                initial_log_lambdas: Array1::zeros(0),
                initial_beta: None,
                gauge_priority: 100,
                jacobian_callback: None,
                stacked_design: None,
                stacked_offset: None,
            },
        ]
    };
    let beta_mu = Array1::from(vec![0.4, 0.8, -0.3]);
    let beta_d = Array1::from(vec![0.9, -0.6]);
    let u = Array1::from(vec![0.7, -1.1, 0.5, -0.4, 0.9]);
    let states_at = |scale: f64| {
        let bm = &beta_mu + &(u.slice(s![0..p_mean]).to_owned() * scale);
        let bd = &beta_d + &(u.slice(s![p_mean..p_mean + p_disp]).to_owned() * scale);
        vec![
            ParameterBlockState {
                eta: x_mean.dot(&bm) + &offset_mean,
                beta: bm,
            },
            ParameterBlockState {
                eta: x_disp.dot(&bd) + &offset_disp,
                beta: bd,
            },
        ]
    };
    let fixtures: [(DispersionFamilyKind, fn(usize) -> f64); 4] = [
        (DispersionFamilyKind::NegativeBinomial, |i| {
            [0.0, 1.0, 3.0, 6.0, 2.0, 0.0, 4.0, 1.0, 8.0][i]
        }),
        (DispersionFamilyKind::Gamma, |i| 0.5 + 0.9 * i as f64),
        (DispersionFamilyKind::Beta, |i| 0.08 + 0.09 * i as f64),
        (DispersionFamilyKind::Tweedie { p: 1.5 }, |i| {
            [0.0, 1.2, 3.4, 0.0, 2.7, 5.1, 0.9, 1.8, 4.4][i]
        }),
    ];
    let h = 1e-6;
    for (kind, y_fn) in fixtures {
        let family = DispersionGlmLocationScaleFamily {
            kind,
            y: Array1::from_shape_fn(n, y_fn),
            weights: Array1::from_elem(n, 1.0),
            jeffreys_armed: true,
        };
        let specs = make_specs();
        let analytic = family
            .exact_newton_joint_hessian_directional_derivative_with_specs(
                &states_at(0.0),
                &specs,
                &u,
            )
            .expect("directional derivative evaluates")
            .expect("directional derivative available");
        let h_plus = family
            .exact_newton_joint_hessian_with_specs(&states_at(h), &specs)
            .expect("H(beta+hu) evaluates")
            .expect("H(beta+hu) available");
        let h_minus = family
            .exact_newton_joint_hessian_with_specs(&states_at(-h), &specs)
            .expect("H(beta-hu) evaluates")
            .expect("H(beta-hu) available");
        let fd = (&h_plus - &h_minus) / (2.0 * h);
        let scale = fd
            .iter()
            .fold(0.0f64, |m, v| m.max(v.abs()))
            .max(analytic.iter().fold(0.0f64, |m, v| m.max(v.abs())))
            .max(1.0);
        let max_err = (&analytic - &fd)
            .iter()
            .fold(0.0f64, |m, v| m.max(v.abs()));
        assert!(
            max_err / scale < 5e-7,
            "{kind:?}: joint Hessian directional derivative disagrees with FD: \
             max abs err {max_err:.3e} at scale {scale:.3e}"
        );
    }
}

/// The four samples `[f(h), f(−h), f(h/2), f(−h/2)]` a Richardson difference reads.
fn richardson_stencil<T>(f: impl Fn(f64) -> T, h: f64) -> [T; 4] {
    [f(h), f(-h), f(0.5 * h), f(-0.5 * h)]
}

/// A centred difference at steps `h` and `h/2` from `samples`
/// (`[f(h), f(−h), f(h/2), f(−h/2)]`), Richardson-extrapolated, with an error bar
/// read off the two estimates: `|D(h/2) − D(h)|` is three times the leading
/// truncation error of `D(h/2)`, which bounds the extrapolated value's error, and
/// each difference also carries the rounding of its two samples, `ε·max|f| / step`.
fn richardson_difference(samples: [f64; 4], h: f64) -> (f64, f64) {
    let coarse = (samples[0] - samples[1]) / (2.0 * h);
    let fine = (samples[2] - samples[3]) / h;
    let magnitude = samples.iter().fold(0.0f64, |m, v| m.max(v.abs()));
    let rounding = f64::EPSILON * magnitude * (1.0 / h + 2.0 / h);
    ((4.0 * fine - coarse) / 3.0, (fine - coarse).abs() + rounding)
}

/// The contracted fourth row surface is the second directional derivative of
/// the observed Hessian channels: on every member (Tweedie on both of its row
/// programs), `D²W[u, v]` equals the Richardson difference of the contracted
/// third surface `DW[u]` along `v`, within the difference's own error bar.
#[test]
fn dispersion_fourth_surface_matches_the_difference_of_the_third() {
    use super::dispersion_family::{
        dispersion_row_observed_hessian_directional,
        dispersion_row_observed_hessian_second_directional,
    };
    let cases = [
        (DispersionFamilyKind::NegativeBinomial, 6.0, 2.0, 3.0, 1.3),
        (DispersionFamilyKind::NegativeBinomial, 0.0, -1.5, 0.4, 0.7),
        (DispersionFamilyKind::Gamma, 0.2, -2.0, 0.8, 0.9),
        (DispersionFamilyKind::Gamma, 9.0, 1.7, 2.0, 1.1),
        (DispersionFamilyKind::Beta, 0.3, -0.7, 1.6, 0.8),
        (DispersionFamilyKind::Beta, 0.92, 1.4, 0.9, 1.4),
        (DispersionFamilyKind::Tweedie { p: 1.5 }, 4.2, 0.9, -0.6, 1.0),
        (DispersionFamilyKind::Tweedie { p: 1.5 }, 0.0, 0.3, 0.2, 1.2),
    ];
    let directions = [([0.7, -1.1], [0.4, 0.9]), ([1.0, 0.0], [0.0, 1.0]), ([-0.3, 0.8], [-0.3, 0.8])];
    let h = 1e-3;
    for (kind, y, em, ed, w) in cases {
        for (u, v) in directions {
            let analytic = dispersion_row_observed_hessian_second_directional(kind, y, em, ed, w, u, v);
            let analytic = [analytic.0, analytic.1, analytic.2];
            let stencil = richardson_stencil(
                |t: f64| {
                    let drift = dispersion_row_observed_hessian_directional(
                        kind,
                        y,
                        em + t * v[0],
                        ed + t * v[1],
                        w,
                        u[0],
                        u[1],
                    );
                    [drift.0, drift.1, drift.2]
                },
                h,
            );
            for (channel, value) in analytic.iter().enumerate() {
                let samples = [
                    stencil[0][channel],
                    stencil[1][channel],
                    stencil[2][channel],
                    stencil[3][channel],
                ];
                let (difference, bar) = richardson_difference(samples, h);
                assert!(
                    (value - difference).abs() <= bar,
                    "{kind:?} y={y} em={em} ed={ed} u={u:?} v={v:?} channel {channel}: fourth \
                     surface {value:.12e} vs difference of the third {difference:.12e} \
                     (gap {:.3e} > bar {bar:.3e})",
                    (value - difference).abs()
                );
            }
        }
    }
}

/// The assembled `D²_β H_L[u, v]` equals the Richardson difference of the
/// assembled `D_β H_L[u]` along `v`, entry by entry within each entry's error
/// bar, on every member: designs, offsets, block packing and the mean/precision
/// cross block included. The trait default this override replaces is
/// block-diagonal and drops that cross block.
#[test]
fn dispersion_joint_hessian_second_directional_matches_the_difference_of_the_first() {
    use super::dispersion_family::DispersionGlmLocationScaleFamily;
    use crate::custom_family::CustomFamily as _;
    use gam_linalg::matrix::DesignMatrix;

    let n = 9usize;
    let p_mean = 3usize;
    let p_disp = 2usize;
    let x_mean = Array2::from_shape_fn((n, p_mean), |(i, j)| {
        let t = i as f64 / (n as f64 - 1.0);
        [1.0, t - 0.5, (t - 0.5) * (t - 0.5)][j]
    });
    let x_disp = Array2::from_shape_fn((n, p_disp), |(i, j)| {
        let t = i as f64 / (n as f64 - 1.0);
        [1.0, t - 0.5][j]
    });
    let offset_mean = Array1::from_shape_fn(n, |i| 0.01 * i as f64);
    let offset_disp = Array1::from_shape_fn(n, |i| -0.02 * i as f64);
    let spec = |name: &str, design: &Array2<f64>, offset: &Array1<f64>| ParameterBlockSpec {
        name: name.to_string(),
        design: DesignMatrix::Dense(gam_linalg::matrix::DenseDesignMatrix::from(design.clone())),
        offset: offset.clone(),
        penalties: Vec::new(),
        nullspace_dims: Vec::new(),
        initial_log_lambdas: Array1::zeros(0),
        initial_beta: None,
        gauge_priority: 100,
        jacobian_callback: None,
        stacked_design: None,
        stacked_offset: None,
    };
    let specs = vec![spec("mu", &x_mean, &offset_mean), spec("log_disp", &x_disp, &offset_disp)];
    let beta_mu = Array1::from(vec![0.4, 0.8, -0.3]);
    let beta_d = Array1::from(vec![0.9, -0.6]);
    let u = Array1::from(vec![0.7, -1.1, 0.5, -0.4, 0.9]);
    let v = Array1::from(vec![-0.2, 0.6, 1.3, 0.8, -0.5]);
    let states_at = |t: f64| {
        let bm = &beta_mu + &(v.slice(s![0..p_mean]).to_owned() * t);
        let bd = &beta_d + &(v.slice(s![p_mean..p_mean + p_disp]).to_owned() * t);
        vec![
            ParameterBlockState {
                eta: x_mean.dot(&bm) + &offset_mean,
                beta: bm,
            },
            ParameterBlockState {
                eta: x_disp.dot(&bd) + &offset_disp,
                beta: bd,
            },
        ]
    };
    let fixtures: [(DispersionFamilyKind, fn(usize) -> f64); 4] = [
        (DispersionFamilyKind::NegativeBinomial, |i| {
            [0.0, 1.0, 3.0, 6.0, 2.0, 0.0, 4.0, 1.0, 8.0][i]
        }),
        (DispersionFamilyKind::Gamma, |i| 0.5 + 0.9 * i as f64),
        (DispersionFamilyKind::Beta, |i| 0.08 + 0.09 * i as f64),
        (DispersionFamilyKind::Tweedie { p: 1.5 }, |i| {
            [0.0, 1.2, 3.4, 0.0, 2.7, 5.1, 0.9, 1.8, 4.4][i]
        }),
    ];
    let h = 1e-3;
    let total = p_mean + p_disp;
    for (kind, y_fn) in fixtures {
        let family = DispersionGlmLocationScaleFamily {
            kind,
            y: Array1::from_shape_fn(n, y_fn),
            weights: Array1::from_elem(n, 1.0),
            jeffreys_armed: true,
        };
        let analytic = family
            .exact_newton_joint_hessian_second_directional_derivative_with_specs(
                &states_at(0.0),
                &specs,
                &u,
                &v,
            )
            .expect("second directional derivative evaluates")
            .expect("second directional derivative available");
        let first_at = |t: f64| {
            family
                .exact_newton_joint_hessian_directional_derivative_with_specs(&states_at(t), &specs, &u)
                .expect("D_beta H[u] evaluates")
                .expect("D_beta H[u] available")
        };
        let stencil = richardson_stencil(first_at, h);
        for a in 0..total {
            for b in 0..total {
                let samples = [
                    stencil[0][[a, b]],
                    stencil[1][[a, b]],
                    stencil[2][[a, b]],
                    stencil[3][[a, b]],
                ];
                let (difference, bar) = richardson_difference(samples, h);
                let value = analytic[[a, b]];
                assert!(
                    (value - difference).abs() <= bar,
                    "{kind:?}: D2H[{a},{b}] {value:.12e} vs difference of D_beta H {difference:.12e} \
                     (gap {:.3e} > bar {bar:.3e})",
                    (value - difference).abs()
                );
            }
        }
    }
}
