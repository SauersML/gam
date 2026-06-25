//! Owed-work objective evaluation for issue #975 — "the carving problem":
//! functional-ANOVA carving of a fitted product-manifold atom into additive
//! main effects (superposition) vs a genuine interaction (binding), plus a
//! STATISTICAL TEST for feature binding.
//!
//! The capability lives in `terms::structure::anova_atom`: `fit_pair_surface`
//! fits the penalized tensor surface `y_d(x1,x2) ≈ φ¹(x1)ᵀ C_d φ²(x2)` over a
//! scattered code sample with REML-selected smoothness, returning coefficients
//! AND their scale-included posterior covariance; `carve` performs the EXACT
//! ANOVA reparameterization (mean + main_a + main_b + interaction), measures
//! the continuous interaction-energy fraction (the "how bound" dial), and runs
//! a Wood-style Wald test of `f₁₂ ≡ 0` on the gauge-projected interaction block
//! — its `edge_p_value` is the feature-binding test. `fission_decision` turns
//! the test into a three-valued split/keep verdict.
//!
//! The in-module unit tests already pin the algebra (exact reparameterization,
//! a hand-planted additive surface fissions losslessly, a hand-planted rank-1
//! interaction refuses + the test rejects). What they do NOT exercise is the
//! capability the issue actually asks for: does the END-TO-END pipeline —
//! REML-fit a scattered surface, harvest its posterior, carve — correctly
//! CLASSIFY a panel of synthetic feature pairs with KNOWN additive-vs-
//! interaction structure as separable vs bound, at controlled Type-I error and
//! high power? This file is that objective evaluation.
//!
//! ## Design
//!
//! A panel of deterministically generated pairs (no RNG: a fixed
//! low-discrepancy abscissa cloud + a fixed RNG-free zero-mean noise stream so
//! the run is a pure function of its inputs). Half are SEPARABLE truths
//! `f(x1)+f(x2)` (additivity = superposition: the binding test must NOT reject
//! and the energy fraction must be tiny), half are BOUND truths carrying a
//! genuine, non-removable cross term (`f₁₂ ≢ 0`: the binding test must reject
//! and the energy fraction must be large). The classifier is the single
//! production rule the evidence ledger consumes:
//!
//!     bound  ⇔  carve(...).edge_p_value ≤ alpha
//!
//! and we require it to classify EVERY pair in the panel correctly (a confusion
//! matrix with zero off-diagonal), plus the continuous `interaction_fraction`
//! and the `fission_decision` to agree with the truth. This is the objective
//! the mandate names: "on synthetic data with known additive-vs-interaction
//! structure, the test must correctly classify bound vs separable feature
//! pairs."
//!
//! No `let _`, no `#[allow(...)]`, no env vars, no `#[cfg(feature=...)]`.

use gam::inference::smooth_test::SmoothTestScale;
use gam::terms::structure::anova_atom::{
    BindingNotion, CarveInput, FissionDecision, carve, fission_decision, fit_pair_surface,
};
use ndarray::Array2;

const N: usize = 1600;
const NOISE_AMP: f64 = 0.05;
const ALPHA: f64 = 0.01;

/// Deterministic low-discrepancy scattered cloud over `[0,1]²`: two coprime
/// irrational phases give a quasi-uniform but non-lattice point set — the
/// scattered shape the carve actually consumes, never a perfect grid.
fn abscissae() -> (Vec<f64>, Vec<f64>) {
    let g1 = 0.618_033_988_749_894_9_f64; // 1/φ
    let g2 = 0.414_213_562_373_095_f64; // √2 − 1
    let mut x1 = Vec::with_capacity(N);
    let mut x2 = Vec::with_capacity(N);
    for i in 0..N {
        x1.push(((i as f64 + 0.5) * g1).fract());
        x2.push(((i as f64 + 0.5) * g2).fract());
    }
    (x1, x2)
}

/// Fixed RNG-free zero-mean noise stream (low-discrepancy phase), so every
/// fit sees identical, reproducible rows.
fn noise(i: usize) -> f64 {
    let golden = 0.754_877_666_246_692_8_f64; // plastic-number fractional phase
    (((i + 1) as f64 * golden).fract() - 0.5) * 2.0 * NOISE_AMP
}

/// One panel entry: a name, a truth `f(x1,x2)`, and whether it is BOUND
/// (genuine interaction) or SEPARABLE (additive / superposition).
struct PanelCase {
    name: &'static str,
    truth: fn(f64, f64) -> f64,
    bound: bool,
}

/// The carve verdict for one panel case, run through the full public path.
struct Verdict {
    edge_p_value: f64,
    interaction_fraction: f64,
    decision: FissionDecision,
}

/// Fit the pair surface end-to-end, harvest its posterior, and carve the
/// representational interaction block. Returns the feature-binding p-value,
/// the continuous interaction-energy fraction, and the fission decision.
fn carve_case(case: &PanelCase) -> Verdict {
    let (x1, x2) = abscissae();
    let mut responses = Array2::<f64>::zeros((N, 1));
    for i in 0..N {
        responses[[i, 0]] = (case.truth)(x1[i], x2[i]) + noise(i);
    }

    // Producer: REML-fit the tensor surface + its scale-included posterior.
    let fit =
        fit_pair_surface(&x1, &x2, responses.view()).expect("pair surface must fit a smooth truth");

    // The joint cross-dimension covariance feeds the exact-rank joint Wald
    // (here D=1, so it coincides with the per-dimension test, but we route the
    // production path the evidence ledger uses).
    let joint_cov = fit.surface.joint_covariance();
    let input = CarveInput {
        phi_a: fit.phi_a.view(),
        phi_b: fit.phi_b.view(),
        coeffs: fit.surface.coeffs.as_slice(),
        coeff_covariance: Some(fit.surface.coeff_covariance.as_slice()),
        joint_coeff_covariance: Some(&joint_cov),
        kernel_a: None,
        kernel_b: None,
        edf: None,
        residual_df: fit.surface.residual_df,
        scale: SmoothTestScale::Estimated,
        notion: BindingNotion::Representational,
    };
    let report = carve(&input, ALPHA).expect("carve must run on a fitted surface");
    let decision = fission_decision(&report, None);
    Verdict {
        edge_p_value: report
            .edge_p_value
            .expect("a fitted surface with covariance must yield a binding p-value"),
        interaction_fraction: report.interaction_fraction,
        decision,
    }
}

// --- the panel -------------------------------------------------------------
// SEPARABLE truths: f(x1)+f(x2). The ANOVA interaction block is exactly zero
// in expectation; any nonzero estimate is noise, so the binding test must NOT
// reject (Type-I control) and the energy fraction must be small.
fn sep_sin_lin(x1: f64, x2: f64) -> f64 {
    (2.0 * std::f64::consts::PI * x1).sin() + 1.3 * x2
}
fn sep_quad_cos(x1: f64, x2: f64) -> f64 {
    3.0 * (x1 - 0.5) * (x1 - 0.5) + (3.0 * x2).cos()
}
fn sep_two_sines(x1: f64, x2: f64) -> f64 {
    (3.0 * x1).sin() - 0.7 * (4.0 * x2).sin()
}

// BOUND truths: a genuine, non-removable cross term (f₁₂ ≢ 0). The binding
// test must reject and the energy fraction must be large.
fn bound_product(x1: f64, x2: f64) -> f64 {
    // Pure bilinear interaction on centered axes: the canonical rank-1 binding.
    4.0 * (x1 - 0.5) * (x2 - 0.5)
}
fn bound_sin_of_product(x1: f64, x2: f64) -> f64 {
    // A curved interaction that no additive split can represent.
    (4.0 * x1 * x2).sin()
}
fn bound_mixed(x1: f64, x2: f64) -> f64 {
    // Additive part PLUS a real cross term: superposition + binding together;
    // the carve must still detect the binding even buried under main effects.
    (2.0 * x1).sin() + 1.1 * x2 + 1.5 * (x1 - 0.5) * (x2 - 0.5)
}

fn panel() -> Vec<PanelCase> {
    vec![
        PanelCase {
            name: "sep_sin_lin",
            truth: sep_sin_lin,
            bound: false,
        },
        PanelCase {
            name: "sep_quad_cos",
            truth: sep_quad_cos,
            bound: false,
        },
        PanelCase {
            name: "sep_two_sines",
            truth: sep_two_sines,
            bound: false,
        },
        PanelCase {
            name: "bound_product",
            truth: bound_product,
            bound: true,
        },
        PanelCase {
            name: "bound_sin_of_product",
            truth: bound_sin_of_product,
            bound: true,
        },
        PanelCase {
            name: "bound_mixed",
            truth: bound_mixed,
            bound: true,
        },
    ]
}

/// #975 objective evaluation: the feature-binding test, run end-to-end through
/// `fit_pair_surface` → `carve`, must classify EVERY synthetic pair in the
/// panel correctly (bound vs separable), with a clean confusion matrix, the
/// continuous interaction-energy dial on the correct side of the divide, and a
/// fission decision that splits the separable atoms and keeps the bound ones.
#[test]
fn carve_classifies_bound_vs_separable_feature_pairs_975() {
    let panel = panel();

    // For a readable failure, collect every verdict before asserting so the
    // panel-wide confusion matrix is visible in one place.
    let verdicts: Vec<(&PanelCase, Verdict)> = panel.iter().map(|c| (c, carve_case(c))).collect();

    let mut wrong: Vec<String> = Vec::new();
    let mut max_sep_fraction = 0.0_f64; // worst (largest) separable interaction energy
    let mut min_bound_fraction = f64::INFINITY; // worst (smallest) bound interaction energy

    for (case, v) in &verdicts {
        let predicted_bound = v.edge_p_value <= ALPHA;

        // (1) The binding TEST classifies correctly.
        if predicted_bound != case.bound {
            wrong.push(format!(
                "{}: truth bound={}, predicted bound={} (edge_p={:.3e}, alpha={ALPHA})",
                case.name, case.bound, predicted_bound, v.edge_p_value
            ));
        }

        // (2) The CONTINUOUS interaction-energy dial agrees with the truth: a
        // separable surface carries negligible interaction energy; a bound one
        // carries a substantial fraction. (A direct, test-independent check on
        // the ANOVA carve itself.)
        if case.bound {
            min_bound_fraction = min_bound_fraction.min(v.interaction_fraction);
        } else {
            max_sep_fraction = max_sep_fraction.max(v.interaction_fraction);
        }

        // (3) The three-valued FISSION DECISION matches the truth: separable
        // atoms are reconstruction-splittable (no computational arm supplied
        // here), bound atoms are KEPT.
        let split_ok = matches!(
            v.decision,
            FissionDecision::SplitReconstructionOnly | FissionDecision::SplitCertifiedJoint
        );
        let kept = matches!(v.decision, FissionDecision::Keep);
        if case.bound && !kept {
            wrong.push(format!(
                "{}: bound truth must KEEP the atom whole, got {:?}",
                case.name, v.decision
            ));
        }
        if !case.bound && !split_ok {
            wrong.push(format!(
                "{}: separable truth must permit a split, got {:?} \
                 (interaction_fraction={:.3e})",
                case.name, v.decision, v.interaction_fraction
            ));
        }
    }

    assert!(
        wrong.is_empty(),
        "#975 feature-binding classifier mislabeled {} of {} panel pairs:\n  {}",
        wrong.len(),
        verdicts.len(),
        wrong.join("\n  ")
    );

    // (4) The continuous dial SEPARATES the two classes with a clear margin —
    // every bound interaction-energy fraction exceeds every separable one (a
    // separating threshold exists), and by a real factor, not a hair.
    assert!(
        min_bound_fraction > max_sep_fraction * 4.0 && min_bound_fraction.is_finite(),
        "#975: the interaction-energy dial must separate bound from separable with margin \
         (max separable fraction = {max_sep_fraction:.3e}, min bound fraction = {min_bound_fraction:.3e})"
    );
}
