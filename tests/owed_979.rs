//! Owed-work regression gate for GitHub issue #979 — the rigid Jeffreys/Firth
//! all-axes information-derivative sweep must build its per-row geometry ONCE and
//! close all `p` coefficient axes from it, producing results IDENTICAL to the
//! prior per-axis sweep.
//!
//! ## The defect (now fixed)
//!
//! The inner-Newton Jeffreys/Firth term needs, each cycle, the directional
//! derivative of the joint information `∂J/∂β · e_a` for every canonical axis
//! `e_a` (`a = 0..p`). The original path asked the family for `Hdot[e_a]` `p`
//! separate times via the per-axis hook
//! `joint_jeffreys_information_directional_derivative_with_specs`. For a coupled
//! family whose per-axis derivative reconstructs a fresh row kernel each call
//! (the rigid Bernoulli/survival marginal-slope geometry), that rebuilds the
//! `O(n)` per-row tensor `p` times — the dominant cost on every cycle the
//! conditioning gate arms, the #979 RSS/wall hot path.
//!
//! ## The fix (commit `81ca742a7`)
//!
//! The `CustomFamily` trait gained a BATCHED all-axes hook
//! `joint_jeffreys_information_directional_derivative_all_axes_with_specs`
//! (`src/families/custom_family/family_trait.rs:1103`). The DEFAULT impl is
//! bit-for-bit the prior per-axis sweep (one `e_a` call per axis). A coupled
//! family whose information is a pure design-row Gram OVERRIDES it to build the
//! per-row geometry once and contract every axis off that single build (the rigid
//! marginal-slope override at
//! `src/families/survival/marginal_slope/custom_family_impl.rs:423`, routing
//! through the public `row_kernel_directional_derivative_all_axes` BLAS-3 path).
//!
//! ## The load-bearing contract this test pins
//!
//! A batched override is only a valid optimisation if it returns EXACTLY what the
//! per-axis sweep returns. This gate builds a minimal `CustomFamily` over the
//! PUBLIC `gam::custom_family` trait whose Jeffreys information is the analytic
//! design-row Gram `J(β) = Xᵀ diag(g(η)) X` (η = Xβ), with:
//!   * a per-axis hook that returns the exact directional derivative
//!     `∂J·e_a = Xᵀ diag(g'(η) ⊙ X·e_a) X`, and
//!   * a batched all-axes override that builds `η`, `g'(η)`, and the weighted
//!     design ONCE and closes all `p` axes from that single build.
//! It then asserts the override equals the per-axis sweep for every axis to
//! machine precision, and (independently) equals a hand-computed analytic
//! ground-truth derivative. A batched path that diverged from the per-axis
//! contract — the regression the #979 optimisation must never introduce — fails
//! the first assertion; a both-wrong batched+per-axis pair fails the second.

use std::sync::Arc;

use gam::families::custom_family::{
    CustomFamily, FamilyEvaluation, ParameterBlockSpec, ParameterBlockState,
};
use gam::linalg::matrix::{DenseDesignMatrix, DesignMatrix};
use ndarray::{Array1, Array2};

/// Smooth per-row weight `g(η)` and its derivative `g'(η)`. A logistic-style
/// `g(η) = σ(η)(1−σ(η))` mirrors the Bernoulli information weight whose per-row
/// rebuild was the #979 cost; any smooth `g` exercises the same batched-vs-per-
/// axis contract.
fn g_weight(eta: f64) -> f64 {
    let s = 1.0 / (1.0 + (-eta).exp());
    s * (1.0 - s)
}

/// `g'(η) = σ(1−σ)(1−2σ)`.
fn g_weight_deriv(eta: f64) -> f64 {
    let s = 1.0 / (1.0 + (-eta).exp());
    s * (1.0 - s) * (1.0 - 2.0 * s)
}

/// A minimal single-block `CustomFamily` whose Jeffreys information is the
/// analytic design-row Gram `J(β) = Xᵀ diag(g(η)) X`, η = X·β.
struct GramJeffreysFamily {
    x: Array2<f64>,
}

impl GramJeffreysFamily {
    fn eta(&self, beta: &Array1<f64>) -> Array1<f64> {
        self.x.dot(beta)
    }

    /// Analytic directional derivative `∂J·d = Xᵀ diag(g'(η) ⊙ (X·d)) X` — the
    /// independent ground truth both the per-axis and batched paths must match.
    fn analytic_directional(&self, beta: &Array1<f64>, d: &Array1<f64>) -> Array2<f64> {
        let eta = self.eta(beta);
        let xd = self.x.dot(d);
        let n = self.x.nrows();
        let p = self.x.ncols();
        let mut w = Array1::<f64>::zeros(n);
        for i in 0..n {
            w[i] = g_weight_deriv(eta[i]) * xd[i];
        }
        // Xᵀ diag(w) X
        let mut out = Array2::<f64>::zeros((p, p));
        for i in 0..n {
            let wi = w[i];
            if wi == 0.0 {
                continue;
            }
            for a in 0..p {
                let xia = self.x[[i, a]];
                for b in 0..p {
                    out[[a, b]] += wi * xia * self.x[[i, b]];
                }
            }
        }
        out
    }
}

impl CustomFamily for GramJeffreysFamily {
    fn evaluate(&self, _block_states: &[ParameterBlockState]) -> Result<FamilyEvaluation, String> {
        Ok(FamilyEvaluation {
            log_likelihood: 0.0,
            blockworking_sets: vec![],
        })
    }

    /// Per-axis hook: the exact directional derivative for the supplied direction.
    /// Deliberately recomputes η and the weighted geometry from scratch on EVERY
    /// call — the per-axis rebuild the #979 batched override exists to avoid. The
    /// batched override must reproduce this exactly.
    fn joint_jeffreys_information_directional_derivative_with_specs(
        &self,
        block_states: &[ParameterBlockState],
        _specs: &[ParameterBlockSpec],
        d_beta_flat: &Array1<f64>,
    ) -> Result<Option<Array2<f64>>, String> {
        let beta = &block_states[0].beta;
        Ok(Some(self.analytic_directional(beta, d_beta_flat)))
    }

    /// Batched all-axes override (#979): build η, g'(η), and the per-row geometry
    /// ONCE, then close all `p` canonical axes from that single build instead of
    /// `p` independent per-axis rebuilds. This must equal the per-axis sweep
    /// bit-for-bit.
    fn joint_jeffreys_information_directional_derivative_all_axes_with_specs(
        &self,
        block_states: &[ParameterBlockState],
        specs: &[ParameterBlockSpec],
    ) -> Result<Option<Vec<Array2<f64>>>, String>
    where
        Self: Sync,
    {
        let p = specs.iter().map(|s| s.design.ncols()).sum::<usize>();
        let beta = &block_states[0].beta;
        // ---- single build of the shared per-row geometry --------------------
        let eta = self.eta(beta);
        let n = self.x.nrows();
        let mut gprime = Array1::<f64>::zeros(n);
        for i in 0..n {
            gprime[i] = g_weight_deriv(eta[i]);
        }
        // ---- close every axis e_a from the single build ---------------------
        // ∂J·e_a = Xᵀ diag(g'(η) ⊙ X[:,a]) X
        let mut axes = Vec::with_capacity(p);
        for a in 0..p {
            let mut out = Array2::<f64>::zeros((p, p));
            for i in 0..n {
                let wi = gprime[i] * self.x[[i, a]];
                if wi == 0.0 {
                    continue;
                }
                for r in 0..p {
                    let xir = self.x[[i, r]];
                    for c in 0..p {
                        out[[r, c]] += wi * xir * self.x[[i, c]];
                    }
                }
            }
            axes.push(out);
        }
        Ok(Some(axes))
    }
}

fn single_block_spec(x: &Array2<f64>) -> ParameterBlockSpec {
    let n = x.nrows();
    ParameterBlockSpec {
        name: "gram".to_string(),
        design: DesignMatrix::Dense(DenseDesignMatrix::from(x.clone())),
        offset: Array1::<f64>::zeros(n),
        jacobian_callback: None,
        ..ParameterBlockSpec::defaults()
    }
}

/// A deterministic, well-conditioned design with `p` columns and `n` rows.
fn design(n: usize, p: usize) -> Array2<f64> {
    Array2::from_shape_fn((n, p), |(i, j)| {
        let t = (i as f64 + 1.0) / (n as f64);
        // distinct, non-collinear columns
        match j {
            0 => 1.0,
            1 => t,
            2 => (t * std::f64::consts::PI).sin(),
            _ => ((j as f64) * t).cos(),
        }
    })
}

/// #979: the batched all-axes Jeffreys derivative override must equal the
/// per-axis sweep — bit-close — for every canonical axis, AND match the
/// independent analytic ground truth. A batched path that diverged from the
/// per-axis contract is the exact regression the #979 build-once optimisation
/// must never introduce.
#[test]
fn jeffreys_all_axes_batched_equals_per_axis_sweep_979() {
    let n = 24usize;
    let p = 4usize;
    let x = design(n, p);
    let family = GramJeffreysFamily { x: x.clone() };
    let specs = vec![single_block_spec(&x)];
    let beta = Array1::from_vec(vec![0.30, -0.45, 0.20, 0.10]);
    let states = vec![ParameterBlockState {
        beta: beta.clone(),
        eta: family.eta(&beta),
    }];

    // Batched override: one build, all axes.
    let batched = family
        .joint_jeffreys_information_directional_derivative_all_axes_with_specs(&states, &specs)
        .expect("batched all-axes call ok")
        .expect("family exposes the exact derivative on every axis");
    assert_eq!(
        batched.len(),
        p,
        "the batched sweep must return one p×p derivative per coefficient axis"
    );

    for a in 0..p {
        // Per-axis sweep: one rebuild per axis (the pre-#979 path).
        let mut e_a = Array1::<f64>::zeros(p);
        e_a[a] = 1.0;
        let per_axis = family
            .joint_jeffreys_information_directional_derivative_with_specs(&states, &specs, &e_a)
            .expect("per-axis call ok")
            .expect("per-axis derivative present");

        // Independent analytic ground truth for axis a.
        let truth = family.analytic_directional(&beta, &e_a);

        assert_eq!(batched[a].dim(), (p, p));
        assert_eq!(per_axis.dim(), (p, p));

        let mut max_batched_vs_per_axis = 0.0_f64;
        let mut max_batched_vs_truth = 0.0_f64;
        for r in 0..p {
            for c in 0..p {
                max_batched_vs_per_axis =
                    max_batched_vs_per_axis.max((batched[a][[r, c]] - per_axis[[r, c]]).abs());
                max_batched_vs_truth =
                    max_batched_vs_truth.max((batched[a][[r, c]] - truth[[r, c]]).abs());
            }
        }
        // The build-once override must reproduce the per-axis sweep to machine
        // precision (same arithmetic, reorganised) — this is the #979 contract.
        assert!(
            max_batched_vs_per_axis < 1e-12,
            "axis {a}: #979 batched build-once all-axes derivative diverged from the \
             per-axis sweep by {max_batched_vs_per_axis:e}; the optimisation changed the \
             math, not just the schedule"
        );
        // And both must equal the independent analytic ∂J·e_a (guards against a
        // both-wrong batched+per-axis pair agreeing on a wrong value).
        assert!(
            max_batched_vs_truth < 1e-12,
            "axis {a}: batched derivative disagrees with the analytic \
             Xᵀdiag(g'(η)⊙X·e_a)X ground truth by {max_batched_vs_truth:e}"
        );
    }
}

/// #979 default-equivalence guard: a family that does NOT override the batched
/// hook must, by the trait default, produce exactly the per-axis sweep. This
/// pins that the default all-axes impl is the faithful per-axis fallback the
/// override is benchmarked against (so the override-vs-default substitution the
/// fix performs is sound). `DefaultSweepFamily` reuses the same Gram derivative
/// but leaves the batched hook at its trait default.
#[test]
fn jeffreys_all_axes_default_is_the_per_axis_sweep_979() {
    struct DefaultSweepFamily {
        inner: GramJeffreysFamily,
    }
    impl CustomFamily for DefaultSweepFamily {
        fn evaluate(
            &self,
            _block_states: &[ParameterBlockState],
        ) -> Result<FamilyEvaluation, String> {
            Ok(FamilyEvaluation {
                log_likelihood: 0.0,
                blockworking_sets: vec![],
            })
        }
        fn joint_jeffreys_information_directional_derivative_with_specs(
            &self,
            block_states: &[ParameterBlockState],
            specs: &[ParameterBlockSpec],
            d_beta_flat: &Array1<f64>,
        ) -> Result<Option<Array2<f64>>, String> {
            // Delegate to the exact per-axis derivative.
            self.inner
                .joint_jeffreys_information_directional_derivative_with_specs(
                    block_states,
                    specs,
                    d_beta_flat,
                )
        }
        // NOTE: no override of the all-axes hook — exercises the trait default.
    }

    let n = 18usize;
    let p = 3usize;
    let x = design(n, p);
    let family = DefaultSweepFamily {
        inner: GramJeffreysFamily { x: x.clone() },
    };
    let specs = vec![single_block_spec(&x)];
    let beta = Array1::from_vec(vec![0.2, -0.3, 0.15]);
    let eta = x.dot(&beta);
    let states = vec![ParameterBlockState {
        beta: beta.clone(),
        eta,
    }];

    let default_all_axes = family
        .joint_jeffreys_information_directional_derivative_all_axes_with_specs(&states, &specs)
        .expect("default all-axes call ok")
        .expect("default sweep present");
    assert_eq!(default_all_axes.len(), p);

    for a in 0..p {
        let mut e_a = Array1::<f64>::zeros(p);
        e_a[a] = 1.0;
        let per_axis = family
            .joint_jeffreys_information_directional_derivative_with_specs(&states, &specs, &e_a)
            .expect("per-axis call ok")
            .expect("per-axis derivative present");
        let mut max_gap = 0.0_f64;
        for r in 0..p {
            for c in 0..p {
                max_gap = max_gap.max((default_all_axes[a][[r, c]] - per_axis[[r, c]]).abs());
            }
        }
        assert!(
            max_gap < 1e-12,
            "axis {a}: the trait DEFAULT all-axes hook must be exactly the per-axis sweep; \
             gap {max_gap:e}"
        );
    }
}
