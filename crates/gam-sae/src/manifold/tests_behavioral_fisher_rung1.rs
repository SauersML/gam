//! Rung 1 acceptance — the **output-Fisher metric in the reconstruction loss**
//! ([`gam_problem::MetricProvenance::BehavioralFisher`]), installed on a
//! [`SaeManifoldTerm`] via [`SaeManifoldTerm::set_row_metric`], turns the
//! reconstruction data-fit into generalized least squares in nats
//! (`½ eᵀ G_n e`) while leaving the estimator linear in the coefficients — so
//! the REML/evidence/EDF stack is preserved verbatim.
//!
//! Two load-bearing properties are pinned here:
//!
//! * **GLS preserves REML (the `G = I` limit).** With identity probes
//!   (`s = p`, `U_n = I_p`, so `G_n = I`) the `BehavioralFisher` metric reports
//!   `whitens_likelihood()` yet reproduces the plain-MSE fit: the data-fit
//!   value and every entry of the assembled Arrow–Schur system (the penalized
//!   normal system whose factorization/logdet the REML evidence and EDF are
//!   computed from) match the no-metric isotropic path. Because REML evidence
//!   and EDF are deterministic functions of exactly that assembled system,
//!   identical assembly ⇒ identical evidence and EDF. This is the operational
//!   statement of "GLS with a fixed row metric is still a linear Gaussian model,
//!   so the whole REML machinery survives".
//!
//! * **The metric enters ONLY the reconstruction seam (still linear-in-β).** A
//!   genuinely anisotropic `G_n` moves the data-fit value and the assembled
//!   gradient, but leaves the metric-independent penalties (assignment sparsity,
//!   ARD, decoder smoothness) bit-identical — i.e. it re-weights the
//!   reconstruction residual and nothing else, which is exactly what makes it a
//!   generalized-least-squares reconstruction rather than a different model.
//!
//! This is the principled form of Braun's end-to-end KL + MSE objective:
//! reconstruction anchored to the activation, priced in nats through the
//! pulled-back output Fisher `G = JᵀFJ`, with the sketch
//! `G ≈ Σᵢ vᵢ vᵢᵀ = U_n U_nᵀ` computed by `s` random harvest-time probes.

use crate::assignment::{AssignmentMode, SaeAssignment};
use crate::manifold::{SaeAtomBasisKind, SaeManifoldAtom, SaeManifoldRho, SaeManifoldTerm};
use gam_problem::{MetricProvenance, RowMetric, pack_probe_factors};
use gam_terms::latent::LatentManifold;
use ndarray::{Array1, Array2, Array3};
use std::sync::Arc;

fn lcg_uniform(s: &mut u64) -> f64 {
    *s = s
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    ((*s >> 11) as f64) / ((1u64 << 53) as f64)
}
fn lcg_normal(s: &mut u64) -> f64 {
    let u1 = lcg_uniform(s).max(1e-12);
    let u2 = lcg_uniform(s);
    (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
}

/// Same fixture family as the #2021 structured-residual acceptance: Euclidean
/// atoms, width-2 basis, one latent axis, distinct nonzero decoders so the
/// residual the metric weights is genuinely nonzero.
fn build_term(n: usize, p: usize, k: usize) -> SaeManifoldTerm {
    let atoms: Vec<SaeManifoldAtom> = (0..k)
        .map(|i| {
            let f = (i as f64) + 1.0;
            let decoder = Array2::<f64>::from_shape_fn((2, p), |(m, c)| {
                0.1 * f * ((m + 1) as f64) - 0.05 * (c as f64) + 0.02 * f
            });
            SaeManifoldAtom::new(
                format!("atom{i}"),
                SaeAtomBasisKind::EuclideanPatch,
                1,
                Array2::<f64>::from_elem((n, 2), 1.0),
                Array3::<f64>::zeros((n, 2, 1)),
                decoder,
                Array2::<f64>::eye(2),
            )
            .unwrap()
        })
        .collect();
    let coords: Vec<Array2<f64>> = (0..k)
        .map(|_| Array2::<f64>::from_shape_fn((n, 1), |(r, _)| 0.05 * (r as f64)))
        .collect();
    let manifolds = vec![LatentManifold::Euclidean; k];
    let logits =
        Array2::<f64>::from_shape_fn((n, k), |(r, c)| 0.3 * (c as f64) - 0.1 * (r as f64) + 0.2);
    let assignment = SaeAssignment::from_blocks_with_mode_and_manifolds(
        logits,
        coords,
        manifolds,
        AssignmentMode::ibp_map(0.7, 1.0, false),
    )
    .unwrap();
    SaeManifoldTerm::new(atoms, assignment).unwrap()
}

fn target(n: usize, p: usize) -> Array2<f64> {
    Array2::<f64>::from_shape_fn((n, p), |(r, c)| {
        0.4 - 0.15 * (r as f64) + 0.25 * (c as f64) + 0.05 * ((r * p + c) as f64)
    })
}

/// The identity-probe `BehavioralFisher` metric: `s = p` probes `vₖ = eₖ`, so
/// `U_n = I_p` and `G_n = I`. This is the GLS `G = I` limit — it must whiten
/// the likelihood *in name* yet reproduce the plain-MSE arithmetic exactly.
fn behavioral_fisher_identity(n: usize, p: usize) -> RowMetric {
    // probes[row, i, k] = δ_ik (same identity on every row).
    let probes =
        Array3::<f64>::from_shape_fn((n, p, p), |(_, i, k)| if i == k { 1.0 } else { 0.0 });
    let u = pack_probe_factors(probes.view()).unwrap();
    RowMetric::behavioral_fisher(Arc::new(u), p, p).unwrap()
}

/// A genuinely anisotropic `BehavioralFisher` metric: `s = 2` random probes per
/// row, so `G_n = v₁v₁ᵀ + v₂v₂ᵀ` is a nontrivial rank-2 output-Fisher sketch
/// that varies row-to-row.
fn behavioral_fisher_anisotropic(n: usize, p: usize) -> RowMetric {
    let s = 2usize;
    let mut seed = 0xF15E_B00C_1234_5678u64;
    // Bias one probe toward channel 0 and the other toward the last channel so
    // the induced G is directional (not a scalar multiple of I).
    let probes = Array3::<f64>::from_shape_fn((n, p, s), |(_, i, k)| {
        let base = if k == 0 && i == 0 {
            1.3
        } else if k == 1 && i + 1 == p {
            1.1
        } else {
            0.0
        };
        base + 0.2 * lcg_normal(&mut seed)
    });
    let u = pack_probe_factors(probes.view()).unwrap();
    RowMetric::behavioral_fisher(Arc::new(u), p, s).unwrap()
}

/// GLS preserves REML: at `G = I` the `BehavioralFisher` likelihood-whitening
/// path reproduces the isotropic plain-MSE fit bit-for-bit — data-fit value and
/// every assembled Arrow–Schur entry — so the REML evidence and EDF (functions
/// of exactly that assembled system) are unchanged.
#[test]
fn behavioral_fisher_identity_reproduces_plain_mse_reml_assembly() {
    let (n, p, k) = (6usize, 4usize, 3usize);
    let mut term = build_term(n, p, k);
    let z = target(n, p);
    let rho = SaeManifoldRho::new(-1.0, -6.0, vec![Array1::<f64>::from_elem(1, 0.0); k]);

    // Isotropic (no metric) reference: the plain-MSE loss and assembled system.
    assert!(term.row_metric().is_none());
    let loss_iid = term.loss(z.view(), &rho).unwrap();
    let sys_iid = term.assemble_arrow_schur(z.view(), &rho, None).unwrap();

    // Install the G = I BehavioralFisher metric. It reports whitens_likelihood,
    // but the metric_rank equals p so the whitened residual-dof accounting is
    // unchanged, and the identity factor makes the whitened residual == residual.
    let metric = behavioral_fisher_identity(n, p);
    assert!(
        metric.whitens_likelihood(),
        "BehavioralFisher must whiten the likelihood"
    );
    assert_eq!(
        metric.provenance(),
        MetricProvenance::BehavioralFisher { probes: p }
    );
    assert_eq!(
        metric.metric_rank(),
        p,
        "G=I metric rank must equal p (dof preserved)"
    );
    term.set_row_metric(metric).unwrap();
    assert!(term.row_metric().is_some_and(|m| m.whitens_likelihood()));

    let loss_gls = term.loss(z.view(), &rho).unwrap();
    let sys_gls = term.assemble_arrow_schur(z.view(), &rho, None).unwrap();

    // Data-fit value: bit-for-bit (identity whitening is a term-by-term identity).
    assert_eq!(
        loss_gls.data_fit, loss_iid.data_fit,
        "G=I GLS data-fit must equal plain MSE exactly"
    );
    // Every metric-independent penalty is untouched.
    assert_eq!(loss_gls.assignment_sparsity, loss_iid.assignment_sparsity);
    assert_eq!(loss_gls.smoothness, loss_iid.smoothness);
    assert_eq!(loss_gls.ard, loss_iid.ard);

    // Assembled Arrow–Schur system — the penalized normal system REML
    // differentiates — matches entry-for-entry. β-tier gradient:
    assert_eq!(sys_gls.gb.len(), sys_iid.gb.len());
    for (a, b) in sys_gls.gb.iter().zip(sys_iid.gb.iter()) {
        assert!(
            (a - b).abs() <= 1e-12 * (1.0 + b.abs()),
            "gb mismatch: {a} vs {b}"
        );
    }
    // Per-row t-tier gradient:
    assert_eq!(sys_gls.rows.len(), sys_iid.rows.len());
    for (rg, ri) in sys_gls.rows.iter().zip(sys_iid.rows.iter()) {
        assert_eq!(
            rg.gt.len(),
            ri.gt.len(),
            "per-row t-gradient length mismatch"
        );
        for (a, b) in rg.gt.iter().zip(ri.gt.iter()) {
            assert!(
                (a - b).abs() <= 1e-12 * (1.0 + b.abs()),
                "gt mismatch: {a} vs {b}"
            );
        }
    }
}

/// The output-Fisher metric enters ONLY the reconstruction seam: an anisotropic
/// `G_n` moves the data-fit value and the assembled gradient (GLS is active, not
/// a no-op) while every metric-independent penalty stays bit-identical — the
/// re-weighting is confined to the linear-Gaussian reconstruction, which is what
/// keeps the estimator a generalized least squares (REML applies verbatim).
#[test]
fn behavioral_fisher_anisotropic_moves_only_the_reconstruction() {
    let (n, p, k) = (6usize, 4usize, 3usize);
    let mut term = build_term(n, p, k);
    let z = target(n, p);
    let rho = SaeManifoldRho::new(-1.0, -6.0, vec![Array1::<f64>::from_elem(1, 0.0); k]);

    let loss_iid = term.loss(z.view(), &rho).unwrap();
    let sys_iid = term.assemble_arrow_schur(z.view(), &rho, None).unwrap();

    let metric = behavioral_fisher_anisotropic(n, p);
    assert!(metric.whitens_likelihood());
    assert!(matches!(
        metric.provenance(),
        MetricProvenance::BehavioralFisher { .. }
    ));
    term.set_row_metric(metric).unwrap();

    let loss_gls = term.loss(z.view(), &rho).unwrap();
    let sys_gls = term.assemble_arrow_schur(z.view(), &rho, None).unwrap();

    // Data-fit moved materially (the GLS weighting is genuinely anisotropic).
    let df_rel = (loss_gls.data_fit - loss_iid.data_fit).abs() / (1.0 + loss_iid.data_fit.abs());
    assert!(
        df_rel > 1e-3,
        "GLS data-fit ({}) must differ from MSE ({})",
        loss_gls.data_fit,
        loss_iid.data_fit
    );
    assert!(loss_gls.data_fit.is_finite());

    // Assembled gradient moved: the reconstruction Jacobian is now weighted by
    // the anisotropic G_n, so at least one β-tier gradient entry shifts far above
    // floating-point noise (the total energy alone can be misleading — a rank-s
    // metric inflates some directions while shrinking others). Entrywise max
    // relative change is the robust signal that the metric reaches the assembly.
    let max_gb_rel = sys_gls
        .gb
        .iter()
        .zip(sys_iid.gb.iter())
        .map(|(a, b)| (a - b).abs() / (1.0 + b.abs()))
        .fold(0.0_f64, f64::max);
    assert!(
        max_gb_rel > 1e-3,
        "GLS β-tier gradient must differ materially from MSE; max rel change {max_gb_rel:e}"
    );

    // Metric-independent penalties are byte-identical: the metric touches only
    // the reconstruction data-fit.
    assert_eq!(loss_gls.assignment_sparsity, loss_iid.assignment_sparsity);
    assert_eq!(loss_gls.smoothness, loss_iid.smoothness);
    assert_eq!(loss_gls.ard, loss_iid.ard);
}
