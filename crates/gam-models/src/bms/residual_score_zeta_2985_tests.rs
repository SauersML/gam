//! gam#2985: the Murphy–Topel score/ζ sensitivity of a fit carrying a residual
//! repair block under the pooled joint `(z, r)` covariance, gated against a
//! central finite difference of the production β-score in which EVERYTHING
//! the fit builds from the calibrated score is rebuilt from the perturbed
//! score: the row's own `ζ_i`, the block's runtime (its pooled covariance and
//! declared unit score variance, through [`ResidualBlockRuntime::fit`]) and, on
//! a global-empirical law, the grid and its build record.

use super::hessian_paths::{new_cell_moment_cache_stats, new_cell_moment_lru_cache};
use super::residual_repair::{ResidualBlockRuntime, ResidualRepairSpec};
use super::residual_repair_kernel::ResidualDriveKernel;
use super::*;
use super::family::BernoulliMarginalSlopeFamily;
use crate::row_kernel::{RowKernel, RowSet, build_row_kernel_cache, row_kernel_gradient};
use gam_linalg::matrix::{DenseDesignMatrix, DesignMatrix};
use gam_problem::{InverseLink, StandardLink};
use ndarray::{Array1, Array2};
use std::sync::{Arc, Mutex};

/// A deterministic draw in `(0, 1)`.
fn unit(i: usize, salt: u64) -> f64 {
    let mut state = (i as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ salt;
    let draw = gam_linalg::utils::splitmix64(&mut state);
    ((draw >> 11) as f64 + 0.5) / (1u64 << 53) as f64
}

fn gauss(i: usize, salt: u64) -> f64 {
    (-2.0 * unit(i, salt).ln()).sqrt() * (std::f64::consts::TAU * unit(i, salt ^ 0xA5A5)).cos()
}

#[derive(Clone, Copy)]
enum Law {
    StandardNormal,
    GlobalEmpirical,
}

/// Rows whose `(z, r1, r2)` covariance does not move with the context, so the
/// block keeps the pooled law, with a skewed score for the empirical case.
struct Fixture {
    a: Array2<f64>,
    zeta: Array1<f64>,
    features: Array2<f64>,
    weights: Array1<f64>,
    y: Array1<f64>,
    marginal_x: Array2<f64>,
    slope_x: Array2<f64>,
    law: Law,
}

impl Fixture {
    fn new(n: usize, law: Law) -> Self {
        let a = Array2::from_shape_fn((n, 1), |(i, _)| 2.0 * unit(i, 1) - 1.0);
        let zeta = Array1::from_shape_fn(n, |i| {
            let g = gauss(i, 2);
            match law {
                Law::StandardNormal => g,
                Law::GlobalEmpirical => g + 0.35 * (g * g - 1.0),
            }
        });
        let features = Array2::from_shape_fn((n, 2), |(i, j)| {
            let first = 0.6 * zeta[i] + 0.8 * gauss(i, 3);
            if j == 0 { first } else { 0.5 * first - 0.4 * zeta[i] + 0.7 * gauss(i, 4) }
        });
        let marginal_x = Array2::from_shape_fn((n, 3), |(i, j)| a[[i, 0]].powi(j as i32));
        let slope_x = Array2::from_shape_fn((n, 2), |(i, j)| {
            if j == 0 { 0.5 + 0.3 * (2.1 * a[[i, 0]]).cos() } else { (3.3 * a[[i, 0]]).sin() }
        });
        Self {
            a,
            zeta,
            features,
            weights: Array1::from_shape_fn(n, |i| 0.6 + 0.4 * unit(i, 6)),
            y: Array1::from_shape_fn(n, |i| if unit(i, 5) < 0.4 { 1.0 } else { 0.0 }),
            marginal_x,
            slope_x,
            law,
        }
    }

    fn states(&self) -> Vec<ParameterBlockState> {
        let marginal_beta = Array1::from_vec(vec![-0.3, 0.2, -0.1]);
        let slope_beta = Array1::from_vec(vec![0.5, -0.2]);
        let residual_beta = Array1::from_vec(vec![0.31, -0.22]);
        vec![
            ParameterBlockState {
                eta: self.marginal_x.dot(&marginal_beta),
                beta: marginal_beta,
            },
            ParameterBlockState {
                eta: self.slope_x.dot(&slope_beta),
                beta: slope_beta,
            },
            ParameterBlockState {
                eta: self.features.dot(&residual_beta),
                beta: residual_beta,
            },
        ]
    }

    /// The fit's family and, on a finite law, its build record, everything
    /// built from `zeta` the way the fit builds it.
    fn family_at(
        &self,
        zeta: &Array1<f64>,
    ) -> (BernoulliMarginalSlopeFamily, Option<empirical_measure_sensitivity::EmpiricalZGridBuild>) {
        let spec = ResidualRepairSpec {
            columns: vec!["r1".to_string(), "r2".to_string()],
            features: self.features.clone(),
        };
        let runtime = ResidualBlockRuntime::fit(&spec, zeta.view(), self.weights.view(), self.a.view())
            .expect("residual runtime fits");
        assert!(
            !runtime.field.is_conditional(),
            "the fixture's joint covariance keeps the pooled law"
        );
        let (latent_measure, build) = match self.law {
            Law::StandardNormal => (LatentMeasureKind::StandardNormal, None),
            Law::GlobalEmpirical => {
                let (measure, build) = build_global_empirical_latent_measure(
                    zeta,
                    &self.weights,
                    DEFAULT_EMPIRICAL_LATENT_GRID_SIZE,
                )
                .expect("empirical measure builds");
                (measure, Some(build))
            }
        };
        let policy = gam_runtime::resource::ResourcePolicy::default_library();
        let family = BernoulliMarginalSlopeFamily {
            jeffreys_armed: false,
            residual: Some(Arc::new(runtime)),
            search: None,
            y: Arc::new(self.y.clone()),
            weights: Arc::new(self.weights.clone()),
            z: Arc::new(zeta.clone()),
            latent_measure,
            gaussian_frailty_sd: Some(0.5),
            base_link: InverseLink::Standard(StandardLink::Probit),
            marginal_design: DesignMatrix::Dense(DenseDesignMatrix::from(self.marginal_x.clone())),
            slope_design: DesignMatrix::Dense(DenseDesignMatrix::from(self.slope_x.clone())),
            score_warp: None,
            link_dev: None,
            policy: policy.clone(),
            cell_moment_lru: new_cell_moment_lru_cache(&policy),
            cell_moment_cache_stats: new_cell_moment_cache_stats(),
            intercept_warm_starts: None,
            auto_subsample_phase_counter: Arc::new(std::sync::atomic::AtomicUsize::new(0)),
            auto_subsample_last_rho: Arc::new(Mutex::new(None)),
        };
        (family, build)
    }

    /// The production β-score `∇_β log L` of the kernel over `family`.
    fn score(&self, family: BernoulliMarginalSlopeFamily) -> Array1<f64> {
        let kernel = ResidualDriveKernel::new(family, self.states()).expect("residual kernel");
        let cache = build_row_kernel_cache(&kernel, &RowSet::All).expect("row cache");
        -row_kernel_gradient(&kernel, &cache, &RowSet::All)
    }

    /// Rows spread over the sample whose nearest neighbour in `ζ` is far
    /// outside the difference step, so no step reorders the sample the grid is
    /// compressed from.
    fn probe_rows(&self, step: f64) -> Vec<usize> {
        let n = self.zeta.len();
        let separated = |j: usize| {
            (0..n).all(|i| i == j || (self.zeta[i] - self.zeta[j]).abs() > 8.0 * step)
        };
        (0..n).step_by(n / 12).filter(|&j| separated(j)).collect()
    }
}

fn assert_sensitivity_matches_fd(law: Law) {
    let fixture = Fixture::new(240, law);
    let (family, build) = fixture.family_at(&fixture.zeta);
    let analytic = ResidualDriveKernel::new(family, fixture.states())
        .expect("residual kernel")
        .score_zeta_sensitivity(build.as_ref())
        .expect("the pooled block has a score/ζ sensitivity");
    let p = analytic.ncols();
    assert_eq!(analytic.nrows(), fixture.zeta.len());
    let step = 1.0e-5;
    let rows = fixture.probe_rows(step);
    assert!(rows.len() >= 6, "the fixture keeps probe rows: {rows:?}");
    let mut cross_row_gap = 0.0_f64;
    for &j in &rows {
        let shifted = |sign: f64| {
            let mut zeta = fixture.zeta.clone();
            zeta[j] += sign * step;
            zeta
        };
        let (plus, _) = fixture.family_at(&shifted(1.0));
        let (minus, _) = fixture.family_at(&shifted(-1.0));
        let fd = (fixture.score(plus) - fixture.score(minus)) / (2.0 * step);
        // The row's own read alone: only `z_j` moves, the runtime and the law
        // stay at the fit's. The gap to the total is what the cross-row
        // channels carry.
        let (reference, _) = fixture.family_at(&fixture.zeta);
        let own_only = |sign: f64| {
            let mut family = reference.clone();
            family.z = Arc::new(shifted(sign));
            fixture.score(family)
        };
        let own = (own_only(1.0) - own_only(-1.0)) / (2.0 * step);
        for k in 0..p {
            let error = (analytic[[j, k]] - fd[k]).abs();
            assert!(
                error <= 1.0e-5 * (1.0 + fd[k].abs()),
                "gam#2985: d(∇_β log L)[{k}]/dζ[{j}] analytic {:.9e} vs central FD {:.9e} \
                 (|Δ| = {error:.3e})",
                analytic[[j, k]],
                fd[k]
            );
            cross_row_gap = cross_row_gap.max((fd[k] - own[k]).abs() / (1.0 + fd[k].abs()));
        }
    }
    // The gate discriminates: the cross-row channels are orders of magnitude
    // above its tolerance, so a sensitivity without them would fail it.
    assert!(
        cross_row_gap > 1.0e-3,
        "gam#2985: the cross-row channels move the sensitivity by only {cross_row_gap:.3e}"
    );
}

#[test]
fn residual_score_sensitivity_matches_fd_under_the_standard_normal_law_2985() {
    assert_sensitivity_matches_fd(Law::StandardNormal);
}

#[test]
fn residual_score_sensitivity_matches_fd_under_the_global_empirical_law_2985() {
    assert_sensitivity_matches_fd(Law::GlobalEmpirical);
}

#[test]
fn a_conditional_joint_covariance_has_no_residual_score_sensitivity_2985() {
    // Σ(a) moves with the context, so the block escalates; its M-estimate's
    // implicit derivative is not carried and the channel is refused by name.
    let n = 600;
    let fixture = Fixture::new(n, Law::StandardNormal);
    let features = Array2::from_shape_fn((n, 2), |(i, j)| {
        let rho = 0.8 * fixture.a[[i, 0]];
        let first = rho * fixture.zeta[i] + (1.0 - rho * rho).sqrt() * gauss(i, 3);
        if j == 0 { first } else { 0.5 * first + 0.8 * gauss(i, 4) }
    });
    let spec = ResidualRepairSpec {
        columns: vec!["r1".to_string(), "r2".to_string()],
        features: features.clone(),
    };
    let runtime =
        ResidualBlockRuntime::fit(&spec, fixture.zeta.view(), fixture.weights.view(), fixture.a.view())
            .expect("residual runtime fits");
    assert!(runtime.field.is_conditional());
    let (mut family, _) = fixture.family_at(&fixture.zeta);
    family.residual = Some(Arc::new(runtime));
    let mut states = fixture.states();
    states[2].eta = features.dot(&states[2].beta);
    let error = ResidualDriveKernel::new(family, states)
        .expect("residual kernel")
        .score_zeta_sensitivity(None)
        .expect_err("a conditional Σ(a) has no channel");
    assert!(error.contains("conditional Σ(a)"), "{error}");
}

#[test]
fn the_probe_rows_are_a_real_kernel_width_2985() {
    let fixture = Fixture::new(240, Law::StandardNormal);
    let (family, _) = fixture.family_at(&fixture.zeta);
    let kernel = ResidualDriveKernel::new(family, fixture.states()).expect("residual kernel");
    assert_eq!(kernel.n_coefficients(), 3 + 2 + 2);
}
