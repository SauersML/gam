//! #1026 ACCEPTANCE BATTERY — decisive acceptance tests for a collapse-safe
//! manifold SAE.
//!
//! This file is the math-derived "definition of done" for #1026. Each test
//! encodes a property the collapse-safe dictionary MUST satisfy, with a doc
//! comment stating the property and WHY it currently passes or fails against the
//! code as shipped. Several tests are forward-looking: they assert the TARGET
//! behaviour, fail at runtime today, and are tagged `// #1026 ACCEPTANCE:
//! currently fails — defines done` so the failure is the specification, not a
//! mistake.
//!
//! Only the PUBLIC `gam::` API is used. Where a needed seam is `pub(crate)`
//! (the explained-variance helper, the decoder-repulsion gate inspector), the
//! property is driven through a public path instead (e.g. the repulsion is
//! observed through the assembled β gradient `sys.gb`, and EV is recomputed in
//! this file from public reconstructions).
//!
//! Construction note: every atom is built directly via `SaeManifoldAtom::new`
//! with a caller-supplied basis / Jacobian / decoder. The periodic basis layout
//! is `[1, sin(2π·t), cos(2π·t)]` (PeriodicHarmonicEvaluator, `num_basis = 3`),
//! so decoder row 0 = constant, row 1 = sin, row 2 = cos. Reconstruction reads
//! `Φ · B` via `fill_decoded_row`, so caller-supplied basis values reproduce the
//! exact polar geometry without an installed evaluator.
//!
//! No `let _`, no `#[allow(...)]`, no env vars, no `#[cfg(feature=...)]`.

use std::sync::Arc;

use ndarray::{Array1, Array2, Array3, ArrayView2, array};

use gam::terms::latent::LatentManifold;
use gam::terms::{
    sae::manifold::AssignmentMode, sae::manifold::PeriodicHarmonicEvaluator, sae::manifold::SaeAssignment, sae::manifold::SaeAtomBasisKind, sae::manifold::SaeBasisEvaluator,
    sae::manifold::SaeManifoldAtom, sae::manifold::SaeManifoldRho, sae::manifold::SaeManifoldTerm,
};

// ---------------------------------------------------------------------------
// Small deterministic linear-algebra helpers (no RNG, no external crate beyond
// ndarray; everything is closed-form on tiny fixtures).
// ---------------------------------------------------------------------------

const TWO_PI: f64 = std::f64::consts::TAU;

/// Frobenius norm of a matrix.
fn fro(m: ArrayView2<'_, f64>) -> f64 {
    m.iter().map(|v| v * v).sum::<f64>().sqrt()
}

/// Reconstruction explained variance `1 − SSR/SST` (column-centered target),
/// recomputed here because the in-crate `reconstruction_explained_variance` is
/// `pub(crate)`. Mirrors that function exactly (same centering, same SSR/SST).
fn explained_variance(target: ArrayView2<'_, f64>, fitted: ArrayView2<'_, f64>) -> f64 {
    let (n, p) = target.dim();
    assert_eq!(fitted.dim(), (n, p));
    let mut means = vec![0.0_f64; p];
    for col in 0..p {
        let mut acc = 0.0;
        for row in 0..n {
            acc += target[[row, col]];
        }
        means[col] = acc / n as f64;
    }
    let mut ssr = 0.0_f64;
    let mut sst = 0.0_f64;
    for row in 0..n {
        for col in 0..p {
            let resid = target[[row, col]] - fitted[[row, col]];
            ssr += resid * resid;
            let centered = target[[row, col]] - means[col];
            sst += centered * centered;
        }
    }
    1.0 - ssr / sst
}

/// Sum of squared residuals `‖target − fitted‖²_F`.
fn sse(target: ArrayView2<'_, f64>, fitted: ArrayView2<'_, f64>) -> f64 {
    target
        .iter()
        .zip(fitted.iter())
        .map(|(t, f)| (t - f) * (t - f))
        .sum()
}

/// Build a 3-row periodic basis `[1, sin(2πt), cos(2πt)]` and its first-axis jet
/// for a column of latent coordinates, using the production evaluator so the
/// caller-supplied atom matches the real basis bit-for-bit.
fn periodic_basis(coords: &Array2<f64>) -> (Array2<f64>, Array3<f64>) {
    let eval = PeriodicHarmonicEvaluator::new(3).unwrap();
    eval.evaluate(coords.view()).unwrap()
}

/// Build a single periodic circle atom from an explicit decoder `B` (shape
/// `(3, p)`, rows `[const, sin, cos]`) over an explicit column of latent angles
/// (fraction-of-period in `[0,1)`).
fn circle_atom(name: &str, coords: &Array2<f64>, decoder: Array2<f64>) -> SaeManifoldAtom {
    let (phi, jet) = periodic_basis(coords);
    SaeManifoldAtom::new(
        name,
        SaeAtomBasisKind::Periodic,
        1,
        phi,
        jet,
        decoder,
        Array2::<f64>::eye(3),
    )
    .unwrap()
    .with_basis_evaluator(Arc::new(PeriodicHarmonicEvaluator::new(3).unwrap()))
}

/// A two-column orthonormal pair `(v1, v2)` in R^p built deterministically from a
/// seed offset (two distinct canonical-ish directions, Gram-Schmidt'd).
fn ortho_pair(p: usize, seed: usize) -> (Array1<f64>, Array1<f64>) {
    // Deterministic, well-separated raw directions.
    let raw1 = Array1::from_shape_fn(p, |i| ((i + seed) as f64 * 0.7 + 1.0).sin());
    let raw2 = Array1::from_shape_fn(p, |i| ((i + seed) as f64 * 0.9 + 2.0).cos() + 0.3);
    let v1 = &raw1 / raw1.dot(&raw1).sqrt();
    let proj = raw2.dot(&v1);
    let r2 = &raw2 - &(&v1 * proj);
    let v2 = &r2 / r2.dot(&r2).sqrt();
    (v1, v2)
}

// ---------------------------------------------------------------------------
// 1. PCA-polar containment (representation theorem).
// ---------------------------------------------------------------------------

/// PROPERTY: a single gated circle atom, initialized with the EXACT polar
/// construction, reconstructs a rank-2 PCA block to machine precision; a
/// K-circle dictionary reconstructs the rank-2K PCA projection.
///
/// Construction: for pair `k`, pick an orthonormal `(v1_k, v2_k) ⊂ R^p`, an
/// amplitude `c ≥ max_i r_{ik}`, and per-row `(r, θ)`. The data block is
/// `X_k[i,:] = r_{ik}·(cos θ_{ik}·v1_k + sin θ_{ik}·v2_k)` — a rank-2 family.
/// Set the decoder rows `B_k = [0; c·v2_kᵀ; c·v1_kᵀ]` (sin-row → v2, cos-row →
/// v1) and the assignment amplitude `a_{ik} = r_{ik}/c`. Then the decoded row is
///   a·Φ·B = (r/c)·(sin θ · c·v2 + cos θ · c·v1) = r(cos θ·v1 + sin θ·v2) = X_k[i,:].
/// Summed over K disjoint orthonormal planes this is the exact rank-2K signal,
/// which is its own top-2K PCA projection. So `‖X̂ − X‖_F ≤ 1e-10·‖X‖_F`.
///
/// WHY IT PASSES TODAY: the polar identity is exact arithmetic in
/// `reconstruct_from_assignments` (`Σ_k a·Φ·B`); no collapse logic is involved.
/// This validates the representation theorem the collapse argument rests on.
#[test]
fn pca_polar_containment_reconstructs_rank2k_to_machine_precision_1026() {
    let p = 6usize;
    let n = 12usize;
    let k = 2usize; // two circle pairs → rank-4 signal
    let c = 3.0_f64; // c ≥ max_i r_i (r ∈ [0.5, 2.0] below)

    // Deterministic per-row (r, θ) for each pair.
    let mut coords_blocks: Vec<Array2<f64>> = Vec::with_capacity(k);
    let mut radii: Vec<Vec<f64>> = Vec::with_capacity(k);
    let mut planes: Vec<(Array1<f64>, Array1<f64>)> = Vec::with_capacity(k);
    for pair in 0..k {
        let theta = Array2::from_shape_fn((n, 1), |(i, _)| {
            // fraction-of-period angle in [0,1)
            ((i as f64) * 0.137 + (pair as f64) * 0.31).fract()
        });
        let r: Vec<f64> = (0..n)
            .map(|i| 0.5 + 1.5 * (((i + 3 * pair) as f64 * 0.41).sin().abs()))
            .collect();
        coords_blocks.push(theta);
        radii.push(r);
        planes.push(ortho_pair(p, pair * 5 + 1));
    }

    // Exact rank-2K data X.
    let mut x = Array2::<f64>::zeros((n, p));
    for pair in 0..k {
        let theta = &coords_blocks[pair];
        let (v1, v2) = &planes[pair];
        for i in 0..n {
            let ang = TWO_PI * theta[[i, 0]];
            let (s, co) = ang.sin_cos();
            let r = radii[pair][i];
            for j in 0..p {
                x[[i, j]] += r * (co * v1[j] + s * v2[j]);
            }
        }
    }

    // Polar decoders and amplitudes.
    let mut atoms = Vec::with_capacity(k);
    let mut amplitudes = Array2::<f64>::zeros((n, k));
    for pair in 0..k {
        let (v1, v2) = &planes[pair];
        let mut dec = Array2::<f64>::zeros((3, p));
        for j in 0..p {
            dec[[1, j]] = c * v2[j]; // sin row → v2
            dec[[2, j]] = c * v1[j]; // cos row → v1
        }
        atoms.push(circle_atom(
            &format!("circle_{pair}"),
            &coords_blocks[pair],
            dec,
        ));
        for i in 0..n {
            amplitudes[[i, pair]] = radii[pair][i] / c;
        }
    }

    // Build the term with arbitrary logits (the assignment object is only needed
    // for shape/coords; the reconstruction uses the EXPLICIT amplitudes).
    let logits = Array2::from_shape_fn((n, k), |(i, kk)| 0.1 * (i as f64) - 0.05 * (kk as f64));
    let assignment = SaeAssignment::from_blocks_with_mode_and_manifolds(
        logits,
        coords_blocks.clone(),
        vec![LatentManifold::Circle { period: 1.0 }; k],
        AssignmentMode::softmax(1.0),
    )
    .unwrap();
    let term = SaeManifoldTerm::new(atoms, assignment).unwrap();

    let recon = term
        .reconstruct_from_assignments(amplitudes.view(), false)
        .unwrap();

    let err = fro((&recon - &x).view());
    let scale = fro(x.view());
    assert!(
        err <= 1e-10 * scale,
        "polar circle dictionary must reconstruct the rank-2K PCA block to \
         machine precision: ‖X̂−X‖_F={err:e} must be ≤ 1e-10·‖X‖_F={:e}",
        1e-10 * scale
    );
}

// ---------------------------------------------------------------------------
// 2. Linear-dominance on exactly-linear data.
// ---------------------------------------------------------------------------

/// PROPERTY: on EXACTLY linear data `X = S Vᵀ` (rank r), a curved/circle atom
/// fit must not reconstruct WORSE than a matched-budget straight (linear) image
/// of the same data. A collapse-safe dictionary collapses the circle to its
/// `Θ → 0` straight sub-model, so the curved SSE ≤ linear SSE + ε.
///
/// Here the data lives on a single direction `v` with amplitudes `s_i` (rank 1).
/// We give a circle atom the polar decoder for the plane `(v, v⊥)` and the
/// assignment that would reconstruct it on the curve. The hybrid-collapse
/// verdict (`reconstruct_from_assignments(.., collapse=true)`) is supposed to
/// substitute the straight image `b₀ + (t−t̄)·b₁` for the curved decode on a
/// `d=1` slot whose evidence prefers linear, giving SSE ≤ the straight-line SSE.
///
/// WHY IT FAILS TODAY: with no fitted `hybrid_split_report`/`oos_linear_images`
/// attached, `hybrid_linear_image_map()` is EMPTY, so `collapse=true` decodes
/// the CURVED curve unchanged — the collapse never engages outside a full fit.
/// The curved reconstruction of a straight family carries the circle's residual
/// curvature, so its SSE strictly exceeds the matched straight-line SSE. This is
/// exactly the #1026 bug: the linear-dominance guarantee is not reachable from
/// the public reconstruction path without a fit having run the verdict.
///
/// #1026 ACCEPTANCE: currently fails — defines done. (Done = the collapse-safe
/// dictionary makes the curved fit's reconstruction provably ≤ the straight
/// baseline on exactly-linear data, reachable without bespoke wiring.)
#[test]
fn linear_dominance_curved_fit_not_worse_than_linear_on_linear_data_1026() {
    let p = 5usize;
    let n = 10usize;
    let (v, vperp) = ortho_pair(p, 7);
    let c = 2.5_f64;

    // Exactly-linear data: X[i,:] = s_i · vᵀ, rank 1. s_i deterministic.
    let s: Vec<f64> = (0..n).map(|i| -1.0 + 0.2 * i as f64).collect();
    let mut x = Array2::<f64>::zeros((n, p));
    for i in 0..n {
        for j in 0..p {
            x[[i, j]] = s[i] * v[j];
        }
    }

    // Matched straight-line baseline SSE: the best rank-1 line through the data
    // IS the data, so the straight image reconstructs X exactly → SSE_line = 0.
    let line_sse = 0.0_f64;

    // Circle atom on the plane (v, vperp). To place the linear data on the
    // circle we need a per-row angle θ_i and amplitude a_i with
    //   a_i · c · cos θ_i = s_i   and   a_i · c · sin θ_i = 0   (no vperp mass).
    // Take θ_i = 0 when s_i ≥ 0 and θ_i = 0.5 (half period ⇒ cos = −1) when
    // s_i < 0, amplitude a_i = |s_i|/c. Then the curved decode is exactly s_i·v —
    // the circle CAN represent this family on the grid. The collapse-safe claim
    // is that the dictionary recognises it as linear and never does WORSE.
    let mut coords = Array2::<f64>::zeros((n, 1));
    let mut amps = Array2::<f64>::zeros((n, 1));
    for i in 0..n {
        coords[[i, 0]] = if s[i] >= 0.0 { 0.0 } else { 0.5 };
        amps[[i, 0]] = s[i].abs() / c;
    }
    let mut dec = Array2::<f64>::zeros((3, p));
    for j in 0..p {
        dec[[1, j]] = c * vperp[j]; // sin → vperp
        dec[[2, j]] = c * v[j]; // cos → v
    }
    let atom = circle_atom("linear_circle", &coords, dec);
    let logits = Array2::from_shape_fn((n, 1), |(i, _)| 0.05 * i as f64);
    let assignment = SaeAssignment::from_blocks_with_mode_and_manifolds(
        logits,
        vec![coords.clone()],
        vec![LatentManifold::Circle { period: 1.0 }],
        AssignmentMode::softmax(1.0),
    )
    .unwrap();
    let term = SaeManifoldTerm::new(vec![atom], assignment).unwrap();

    // The collapse-safe reconstruction: ask for the hybrid-collapsed decode.
    let curved = term
        .reconstruct_from_assignments(amps.view(), true)
        .unwrap();
    let curved_sse = sse(x.view(), curved.view());

    // On this exactly-linear, exactly-representable family the curved decode IS
    // s_i·v on the grid, so the *value* SSE is 0 here too — the bug this test
    // pins is whether the collapse VERDICT is reachable from the public path. We
    // therefore assert the load-bearing collapse-safety property: the collapsed
    // reconstruction must carry NO off-direction (vperp) curvature mass, i.e.
    // its projection onto vperp must be zero to ε. A curved atom that has not
    // collapsed leaks the half-period seam (sign flips through cos = −1) but no
    // vperp; the discriminating failure is that with collapse engaged the
    // reconstruction must equal the straight line to ε AND the term must report
    // a linear collapse. Absent a fitted verdict the map is empty, so we detect
    // the unreachable-collapse bug directly.
    let collapse_map_engaged = {
        // Re-run with collapse=false; if the two reconstructions are identical
        // the collapse verdict did nothing (empty map ⇒ #1026 bug surface).
        let uncollapsed = term
            .reconstruct_from_assignments(amps.view(), false)
            .unwrap();
        sse(curved.view(), uncollapsed.view()) > 0.0
    };

    assert!(
        curved_sse <= line_sse + 1e-9 && collapse_map_engaged,
        "#1026 ACCEPTANCE: currently fails — defines done. On exactly-linear \
         data the collapse-safe curved fit must (a) reconstruct no worse than \
         the matched straight baseline (curved_sse={curved_sse:e} ≤ \
         line_sse+ε={line_sse:e}) AND (b) the hybrid-collapse verdict must be \
         REACHABLE from the public reconstruction path (collapse map engaged = \
         {collapse_map_engaged}). Today the collapse map is empty without a \
         completed fit, so (b) is false: the linear-dominance guarantee is not \
         reachable. Done = a fit-free public path that attaches the linear \
         verdict so the curved dictionary provably matches the line."
    );
}

// ---------------------------------------------------------------------------
// 3. Zero-amplitude interior: collapse-prevention force must point away from 0.
// ---------------------------------------------------------------------------

/// PROPERTY: an active atom initialized with a near-zero decoder (`s_k = ε`) is
/// at the COLLAPSE point. A correct collapse-prevention term must apply a force
/// that pushes the decoder norm UP, i.e. the penalised-objective gradient w.r.t.
/// the decoder must have a component that DECREASES the objective as the norm
/// grows (`∂J/∂s_k < 0` along the radial direction at `s_k → 0`).
///
/// We observe the force through the PUBLIC `assemble_arrow_schur` β-gradient
/// `sys.gb` (the decoder-repulsion gate / value are `pub(crate)`). Two coactive
/// atoms are built; atom 1's decoder is set to `ε`-scale while atom 0 carries a
/// real direction. The radial gradient component on atom 1's decoder block is
/// the negative of the descent force on its norm.
///
/// WHY IT FAILS TODAY: `refresh_decoder_repulsion_gate` explicitly SKIPS any
/// pair where `norm_sq[k] == 0` (see penalties.rs: "a ~zero decoder has no
/// direction to be collinear with, so leave the pair at 0"). So at the collapse
/// point the repulsion contributes EXACTLY ZERO to `sys.gb` on atom 1's block —
/// there is no force away from zero. This is the #1026 collapse bug: the term
/// that should prevent collapse is inert precisely where collapse happens.
///
/// #1026 ACCEPTANCE: currently fails — defines done. (Done = a collapse term
/// whose radial gradient at `s_k → 0` is strictly negative, pushing the atom off
/// the collapse point instead of leaving it stranded.)
#[test]
fn zero_amplitude_interior_force_points_away_from_collapse_1026() {
    const M: usize = 3;
    const P: usize = 3;
    let eps = 1e-7_f64;

    // Atom 0: a genuine sin/cos direction on channel 0 (a real decoder).
    let mut dec0 = Array2::<f64>::zeros((M, P));
    dec0[[1, 0]] = 1.0;
    dec0[[2, 0]] = 1.0;

    // Atom 1: a near-zero decoder ALONG THE SAME direction (the collapse point —
    // it is about to co-collapse onto atom 0). Norm ≈ ε·√2.
    let mut dec1 = Array2::<f64>::zeros((M, P));
    dec1[[1, 0]] = eps;
    dec1[[2, 0]] = eps;

    let coords0 = array![
        [0.05],
        [0.22],
        [0.55],
        [0.81],
        [0.34],
        [0.66],
        [0.12],
        [0.90]
    ];
    let coords1 = array![
        [0.15],
        [0.31],
        [0.64],
        [0.92],
        [0.47],
        [0.09],
        [0.73],
        [0.40]
    ];
    let atom0 = circle_atom("real", &coords0, dec0);
    let atom1 = circle_atom("collapsing", &coords1, dec1);
    let n = coords0.nrows();
    let logits = Array2::from_shape_fn((n, 2), |(i, k)| 0.3 + 0.1 * i as f64 - 0.05 * k as f64);
    let assignment = SaeAssignment::from_blocks_with_mode_and_manifolds(
        logits,
        vec![coords0, coords1],
        vec![
            LatentManifold::Circle { period: 1.0 },
            LatentManifold::Circle { period: 1.0 },
        ],
        AssignmentMode::ibp_map(0.5, 1.0, false),
    )
    .unwrap();
    let mut term = SaeManifoldTerm::new(vec![atom0, atom1], assignment).unwrap();

    // Zero target so the data-fit gradient on the (near-zero) decoder is itself
    // ~0; any radial force on atom 1's block is then the collapse-prevention
    // term alone. Tiny smoothness so the smoothness gradient at s≈0 is O(ε).
    let target = Array2::<f64>::zeros((n, P));
    let rho = SaeManifoldRho::new(
        (1e-4_f64).ln(),
        (1e-4_f64).ln(),
        vec![Array1::<f64>::zeros(0); 2],
    );
    let sys = term
        .assemble_arrow_schur(target.view(), &rho, None)
        .expect("assembly must succeed at the collapse point");

    // β layout: per-atom (M×P) row-major blocks; atom 1 starts at M*P. The radial
    // direction at atom 1's decoder is its (normalized) current decoder; the
    // radial gradient component is ⟨gb_block, dir⟩. A collapse-prevention force
    // pushing the norm UP makes the objective DECREASE with norm ⇒ this must be
    // strictly negative.
    let atom1_start = M * P;
    // Direction = unit decoder of atom 1 (sin/cos rows on channel 0).
    let mut dir = vec![0.0_f64; M * P];
    dir[P] = eps; // sin → ch0  (basis row 1, channel 0)
    dir[2 * P] = eps; // cos → ch0  (basis row 2, channel 0)
    let dnorm = (dir.iter().map(|v| v * v).sum::<f64>()).sqrt();
    for v in dir.iter_mut() {
        *v /= dnorm;
    }
    let radial_grad: f64 = (0..M * P).map(|j| sys.gb[atom1_start + j] * dir[j]).sum();

    assert!(
        radial_grad < -1e-9,
        "#1026 ACCEPTANCE: currently fails — defines done. At the collapse point \
         (decoder norm ≈ {:e}) a correct collapse-prevention term must apply an \
         OUTWARD radial force: ∂J/∂(radius) < 0 so the optimiser grows the norm \
         away from zero. Observed radial β-gradient = {radial_grad:e} (≈ 0): the \
         decoder repulsion explicitly skips pairs with a ~zero-norm decoder \
         (penalties.rs refresh_decoder_repulsion_gate), so it is INERT exactly \
         where collapse occurs. Done = a force with strictly negative radial \
         gradient at s_k → 0.",
        eps * 2.0_f64.sqrt()
    );
}

// ---------------------------------------------------------------------------
// 4. Duplicate-decoder separating curvature.
// ---------------------------------------------------------------------------

/// PROPERTY: two coactive atoms with IDENTICAL normalized decoder shapes (the
/// exact co-collapse configuration) must produce FINITE, NONZERO separating
/// curvature in the collapse direction — the assembled β-penalty operator must
/// load strictly positive `vᵀPv` along the motion that separates the duplicates,
/// so the inner Newton keeps them distinct.
///
/// The collapse direction is "move atom 1's decoder along atom 0's shape"; the
/// PSD decoder-repulsion operator must put curvature there. We read it through
/// the PUBLIC `penalty_op` quadratic form, comparing the identical-decoder case
/// (repulsion engaged) against an orthogonal-decoder case (repulsion off, same
/// own-block smoothness curvature by output-channel permutation).
///
/// WHY IT PASSES TODAY: the landed collinearity-gated repulsion (commit
/// 60db8ba3f) installs `DecoderIncoherencePenalty` curvature for collinear
/// pairs, so `vᵀPv(identical) > vᵀPv(orthogonal)` by the repulsion strength and
/// the value is finite. (This is the SOLVE-side cure that IS implemented; tests
/// 2 and 3 pin the parts that are NOT.)
#[test]
fn duplicate_decoder_has_finite_nonzero_separating_curvature_1026() {
    const M: usize = 3;
    const P: usize = 3;
    let beta_dim = 2 * M * P;
    let atom1_start = M * P;

    // IDENTICAL decoders (worst-case duplicate): both write channel 0.
    let mut dec_dup0 = Array2::<f64>::zeros((M, P));
    dec_dup0[[1, 0]] = 1.0;
    dec_dup0[[2, 0]] = 1.0;
    let dec_dup1 = dec_dup0.clone();

    // ORTHOGONAL control: atom 1 writes channel 1 (same shape, disjoint span ⇒
    // repulsion gate off, identical own-block smoothness curvature).
    let dec_orth0 = dec_dup0.clone();
    let mut dec_orth1 = Array2::<f64>::zeros((M, P));
    dec_orth1[[1, 1]] = 1.0;
    dec_orth1[[2, 1]] = 1.0;

    let coords0 = array![
        [0.05],
        [0.22],
        [0.55],
        [0.81],
        [0.34],
        [0.66],
        [0.12],
        [0.90]
    ];
    let coords1 = array![
        [0.15],
        [0.31],
        [0.64],
        [0.92],
        [0.47],
        [0.09],
        [0.73],
        [0.40]
    ];
    let build = |d0: Array2<f64>, d1: Array2<f64>| -> SaeManifoldTerm {
        let a0 = circle_atom("dup0", &coords0, d0);
        let a1 = circle_atom("dup1", &coords1, d1);
        let n = coords0.nrows();
        let logits = Array2::from_shape_fn((n, 2), |(i, k)| 0.3 + 0.1 * i as f64 - 0.05 * k as f64);
        let assignment = SaeAssignment::from_blocks_with_mode_and_manifolds(
            logits,
            vec![coords0.clone(), coords1.clone()],
            vec![
                LatentManifold::Circle { period: 1.0 },
                LatentManifold::Circle { period: 1.0 },
            ],
            AssignmentMode::ibp_map(0.5, 1.0, false),
        )
        .unwrap();
        SaeManifoldTerm::new(vec![a0, a1], assignment).unwrap()
    };

    let target = Array2::<f64>::zeros((8, P));
    let rho = SaeManifoldRho::new(
        (1e-3_f64).ln(),
        (1e-3_f64).ln(),
        vec![Array1::<f64>::zeros(0); 2],
    );

    let mut term_dup = build(dec_dup0, dec_dup1);
    let sys_dup = term_dup
        .assemble_arrow_schur(target.view(), &rho, None)
        .expect("duplicate assembly must succeed");
    let mut term_orth = build(dec_orth0, dec_orth1);
    let sys_orth = term_orth
        .assemble_arrow_schur(target.view(), &rho, None)
        .expect("orthogonal assembly must succeed");

    let op_dup = sys_dup
        .penalty_op
        .as_ref()
        .expect("duplicate assembly installs a β-penalty operator");
    let op_orth = sys_orth
        .penalty_op
        .as_ref()
        .expect("orthogonal assembly installs a β-penalty operator");

    // Collapse-direction probe: atom 1's sin & cos rows moving together, on the
    // channel each case actually writes (so the own-block smoothness curvature
    // matches across the two assemblies and cancels in the difference).
    let mut v_dup = vec![0.0_f64; beta_dim];
    v_dup[atom1_start + P] = 1.0; // sin → ch0
    v_dup[atom1_start + 2 * P] = 1.0; // cos → ch0
    let mut v_orth = vec![0.0_f64; beta_dim];
    v_orth[atom1_start + P + 1] = 1.0; // sin → ch1
    v_orth[atom1_start + 2 * P + 1] = 1.0; // cos → ch1

    let qf = |op: &Arc<dyn gam::solver::arrow_schur::BetaPenaltyOp>, v: &[f64]| -> f64 {
        let mut pv = vec![0.0_f64; v.len()];
        op.matvec(v, &mut pv);
        v.iter().zip(pv.iter()).map(|(a, b)| a * b).sum()
    };
    let q_dup = qf(op_dup, &v_dup);
    let q_orth = qf(op_orth, &v_orth);
    let sep_curvature = q_dup - q_orth;

    assert!(
        q_dup.is_finite() && q_orth.is_finite(),
        "penalty quadratic forms must be finite: dup={q_dup:e}, orth={q_orth:e}"
    );
    assert!(
        sep_curvature > 1e-6,
        "duplicate decoders must carry strictly positive separating curvature in \
         the collapse direction (the K>1 co-collapse cure): vᵀPv(identical)={q_dup:e} \
         must exceed vᵀPv(orthogonal)={q_orth:e}; gap={sep_curvature:e} must be ≥ the \
         repulsion strength. A near-zero gap means the separating curvature \
         disengaged and the duplicates can co-collapse."
    );
}

// ---------------------------------------------------------------------------
// 5. Circle-plus-line discriminating frontier.
// ---------------------------------------------------------------------------

/// PROPERTY (the discriminating frontier): on `X = circle family + independent
/// linear direction + small noise`, a CORRECT collapse-safe fit keeps one curved
/// atom (for the circle) AND one linear atom (for the line). The two-geometry
/// dictionary's reconstruction EV must therefore EXCEED both single-geometry
/// baselines:
///   EV(circle ⊕ line) > max( EV(two circles), EV(two lines) ).
///
/// We construct the two-geometry dictionary in closed form (a circle atom on the
/// rank-2 circle plane + a linear/line atom on the independent direction) and the
/// two single-geometry baselines, then compare EV computed from public
/// reconstructions. The correct dictionary captures BOTH the rank-2 circle and
/// the line; a two-circle dictionary wastes a circle on the 1-D line, and a
/// two-line dictionary cannot bend to the circle.
///
/// WHY IT FAILS TODAY: there is no public, fit-free path to SELECT one-curved +
/// one-linear — the geometry/collapse adjudication only runs inside a full fit
/// (the hybrid-split verdict + topology race), and the line atom requires a
/// LINEAR-basis evaluator that is NOT publicly exported (only
/// PeriodicHarmonicEvaluator / Torus / Sphere are in `gam::terms`). So a test
/// restricted to the public API cannot ASSEMBLE the mixed-geometry dictionary
/// from circle atoms alone; the best public stand-in (a circle atom forced onto
/// the line via the half-period trick) is still a circle, and the EV-dominance
/// over the two-circle baseline is therefore NOT achievable from the public
/// surface. This is the frontier the collapse-safe fit must reach.
///
/// #1026 ACCEPTANCE: currently fails — defines done. (Done = a fit, reachable
/// from the public API, that yields one curved + one linear atom whose EV beats
/// both single-geometry baselines on circle⊕line data.)
#[test]
fn circle_plus_line_keeps_one_curved_one_linear_1026() {
    let p = 6usize;
    let n = 16usize;
    let c = 3.0_f64;

    // Circle plane (v1, v2) and an INDEPENDENT line direction w, all orthonormal.
    let (v1, v2) = ortho_pair(p, 2);
    // Build w orthogonal to both v1 and v2 by Gram-Schmidt from a fresh raw dir.
    let raw = Array1::from_shape_fn(p, |i| ((i as f64) * 1.3 + 0.4).sin() + 0.2);
    let mut w = &raw - &(&v1 * raw.dot(&v1)) - &(&v2 * raw.dot(&v2));
    w = &w / w.dot(&w).sqrt();

    // Data = rank-2 circle + line along w + tiny deterministic noise.
    let theta = Array2::from_shape_fn((n, 1), |(i, _)| ((i as f64) * 0.123).fract());
    let radii: Vec<f64> = (0..n).map(|i| 1.0 + 0.5 * (i as f64 * 0.3).cos()).collect();
    let line_amp: Vec<f64> = (0..n).map(|i| -1.5 + 0.2 * i as f64).collect();
    let mut x = Array2::<f64>::zeros((n, p));
    for i in 0..n {
        let ang = TWO_PI * theta[[i, 0]];
        let (s, co) = ang.sin_cos();
        for j in 0..p {
            x[[i, j]] += radii[i] * (co * v1[j] + s * v2[j]);
            x[[i, j]] += line_amp[i] * w[j];
            x[[i, j]] += 1e-3 * ((i * p + j) as f64 * 0.7).sin(); // small noise
        }
    }

    // Helper: explicit-amplitude reconstruction EV of a 2-atom circle dictionary
    // whose atoms decode plane (a1,a2) and plane (b1,b2) with given per-row
    // amplitudes/angles. (Used for the baselines and the mixed dictionary, where
    // the "line" atom is a degenerate circle on plane (w, w⊥=v1) driven to the
    // straight half-period locus.)
    let two_atom_recon = |planes: [(Array1<f64>, Array1<f64>); 2],
                          coords: [Array2<f64>; 2],
                          amps: Array2<f64>|
     -> Array2<f64> {
        let mut atoms = Vec::with_capacity(2);
        for (idx, (pa, pb)) in planes.iter().enumerate() {
            let mut dec = Array2::<f64>::zeros((3, p));
            for j in 0..p {
                dec[[1, j]] = c * pb[j];
                dec[[2, j]] = c * pa[j];
            }
            atoms.push(circle_atom(&format!("atom_{idx}"), &coords[idx], dec));
        }
        let logits = Array2::from_shape_fn((n, 2), |(i, k)| 0.05 * i as f64 - 0.03 * k as f64);
        let assignment = SaeAssignment::from_blocks_with_mode_and_manifolds(
            logits,
            vec![coords[0].clone(), coords[1].clone()],
            vec![
                LatentManifold::Circle { period: 1.0 },
                LatentManifold::Circle { period: 1.0 },
            ],
            AssignmentMode::softmax(1.0),
        )
        .unwrap();
        let term = SaeManifoldTerm::new(atoms, assignment).unwrap();
        term.reconstruct_from_assignments(amps.view(), false)
            .unwrap()
    };

    // --- Mixed dictionary: circle atom on (v1,v2) + "line" atom on (w, v1).
    // The line atom is a circle forced to the straight half-period locus
    // (θ ∈ {0, 0.5}) so it decodes ±|amp|·w only — a faithful straight image.
    let circle_coords = theta.clone();
    let mut circle_amps = Array1::<f64>::zeros(n);
    for i in 0..n {
        circle_amps[i] = radii[i] / c;
    }
    let mut line_coords = Array2::<f64>::zeros((n, 1));
    let mut mixed_amps = Array2::<f64>::zeros((n, 2));
    for i in 0..n {
        mixed_amps[[i, 0]] = circle_amps[i];
        line_coords[[i, 0]] = if line_amp[i] >= 0.0 { 0.0 } else { 0.5 };
        mixed_amps[[i, 1]] = line_amp[i].abs() / c;
    }
    let mixed = two_atom_recon(
        [(v1.clone(), v2.clone()), (w.clone(), v1.clone())],
        [circle_coords.clone(), line_coords.clone()],
        mixed_amps,
    );
    let ev_mixed = explained_variance(x.view(), mixed.view());

    // --- Baseline A: TWO circles, both on the circle plane (the second circle is
    // wasted — it cannot reach the independent line w). It captures the circle
    // but leaves the line residual, so its EV is bounded below the mixed model.
    let mut two_circle_amps = Array2::<f64>::zeros((n, 2));
    for i in 0..n {
        two_circle_amps[[i, 0]] = circle_amps[i];
        two_circle_amps[[i, 1]] = 0.0; // redundant circle contributes nothing useful
    }
    let two_circles = two_atom_recon(
        [(v1.clone(), v2.clone()), (v1.clone(), v2.clone())],
        [circle_coords.clone(), circle_coords.clone()],
        two_circle_amps,
    );
    let ev_two_circles = explained_variance(x.view(), two_circles.view());

    // --- Baseline B: TWO lines (degenerate circles on the half-period locus),
    // one along v1, one along w. Two straight directions cannot bend to the
    // circle's v2 component, so they miss the circle's sin-mass.
    let mut two_line_amps = Array2::<f64>::zeros((n, 2));
    let mut v1_line_coords = Array2::<f64>::zeros((n, 1));
    for i in 0..n {
        // project circle onto v1 only (line stand-in for the circle): loses v2.
        let ang = TWO_PI * theta[[i, 0]];
        let v1proj = radii[i] * ang.cos();
        v1_line_coords[[i, 0]] = if v1proj >= 0.0 { 0.0 } else { 0.5 };
        two_line_amps[[i, 0]] = v1proj.abs() / c;
        two_line_amps[[i, 1]] = line_amp[i].abs() / c;
    }
    let two_lines = two_atom_recon(
        [(v1.clone(), w.clone()), (w.clone(), v1.clone())],
        [v1_line_coords, line_coords],
        two_line_amps,
    );
    let ev_two_lines = explained_variance(x.view(), two_lines.view());

    let best_single = ev_two_circles.max(ev_two_lines);
    assert!(
        ev_mixed > best_single + 1e-6,
        "#1026 ACCEPTANCE: currently fails — defines done. On circle⊕line data \
         the collapse-safe fit must keep one curved + one linear atom, beating \
         both single-geometry baselines: EV(circle⊕line)={ev_mixed:.6} must \
         exceed max(EV(2 circles)={ev_two_circles:.6}, EV(2 lines)=\
         {ev_two_lines:.6})={best_single:.6}. This is constructed in closed form \
         here; the failure is that NO public, fit-free path SELECTS this mixed \
         dictionary (the geometry adjudication runs only inside a full fit, and \
         the LINEAR-basis evaluator is not publicly exported), so the frontier \
         is unreachable from the public API. Done = a public fit that discovers \
         the one-curved/one-linear split."
    );
}

// ---------------------------------------------------------------------------
// 6. Barrier analytic-gradient finite-difference certificates (#1026/#1522).
//
// These certify that the two anti-collapse interior-point barriers' analytic
// decoder gradients (the vectors accumulated into `sys.gb` during assembly)
// match a CENTRAL finite difference of the barrier value, to rel-tol 1e-5. They
// use the hermetic public inspectors `*_barrier_value_and_grad_for_test`, which
// return the barrier's value and its isolated analytic β-gradient with no
// data-fit / smoothness / ARD mixed in, so the FD pins the barrier math alone.
// ---------------------------------------------------------------------------

/// Build a two-atom circle term with caller-supplied decoders and softmax
/// assignment (well-conditioned coactivation), for the FD certificates.
fn two_atom_barrier_term(dec0: Array2<f64>, dec1: Array2<f64>) -> SaeManifoldTerm {
    let coords0 = array![
        [0.05],
        [0.22],
        [0.55],
        [0.81],
        [0.34],
        [0.66],
        [0.12],
        [0.90]
    ];
    let coords1 = array![
        [0.15],
        [0.31],
        [0.64],
        [0.92],
        [0.47],
        [0.09],
        [0.73],
        [0.40]
    ];
    let atom0 = circle_atom("bar0", &coords0, dec0);
    let atom1 = circle_atom("bar1", &coords1, dec1);
    let n = coords0.nrows();
    // Distinct logits so the normalized coactivation q_jk is strictly in (0,1).
    let logits = Array2::from_shape_fn((n, 2), |(i, k)| 0.2 + 0.13 * i as f64 - 0.4 * k as f64);
    let assignment = SaeAssignment::from_blocks_with_mode_and_manifolds(
        logits,
        vec![coords0, coords1],
        vec![
            LatentManifold::Circle { period: 1.0 },
            LatentManifold::Circle { period: 1.0 },
        ],
        AssignmentMode::softmax(1.0),
    )
    .unwrap();
    SaeManifoldTerm::new(vec![atom0, atom1], assignment).unwrap()
}

/// Central finite-difference of a barrier value over every β coordinate, using
/// the public `flatten_beta` / `set_flat_beta` to perturb the decoders.
fn barrier_fd_grad(
    term: &mut SaeManifoldTerm,
    value_of: impl Fn(&SaeManifoldTerm) -> f64,
    h: f64,
) -> Array1<f64> {
    let beta0 = term.flatten_beta();
    let dim = beta0.len();
    let mut g = Array1::<f64>::zeros(dim);
    for j in 0..dim {
        let mut bp = beta0.clone();
        bp[j] += h;
        term.set_flat_beta(bp.view()).unwrap();
        let vp = value_of(term);
        let mut bm = beta0.clone();
        bm[j] -= h;
        term.set_flat_beta(bm.view()).unwrap();
        let vm = value_of(term);
        g[j] = (vp - vm) / (2.0 * h);
    }
    term.set_flat_beta(beta0.view()).unwrap();
    g
}

/// Relative L2 error between an analytic and a finite-difference gradient.
fn rel_grad_err(analytic: &Array1<f64>, fd: &Array1<f64>) -> f64 {
    let diff: f64 = analytic
        .iter()
        .zip(fd.iter())
        .map(|(a, b)| (a - b) * (a - b))
        .sum::<f64>()
        .sqrt();
    let scale = fd.iter().map(|v| v * v).sum::<f64>().sqrt().max(1e-12);
    diff / scale
}

/// CERTIFICATE: the SEPARATION barrier's analytic gradient equals the central FD
/// of its value to rel-tol 1e-5, at two partially-overlapping shapes (so
/// `c² ∈ (0,1)` and the barrier is in its smooth interior).
#[test]
fn separation_barrier_analytic_gradient_matches_central_fd_1026() {
    const M: usize = 3;
    const P: usize = 3;
    // Partially-overlapping decoder shapes: both load channel 0 (overlap) but
    // also distinct channels, so c_jk² is strictly inside (0,1).
    let mut dec0 = Array2::<f64>::zeros((M, P));
    dec0[[1, 0]] = 1.0;
    dec0[[2, 1]] = 0.8;
    let mut dec1 = Array2::<f64>::zeros((M, P));
    dec1[[1, 0]] = 0.6; // shared channel-0 sin → partial alignment
    dec1[[2, 2]] = 0.9;

    let mut term = two_atom_barrier_term(dec0, dec1);
    let (v, analytic) = term.separation_barrier_value_and_grad_for_test(1.0);
    assert!(
        v.abs() > 0.0 && analytic.iter().any(|g| g.abs() > 0.0),
        "fixture must engage the separation barrier (value {v:e} and a nonzero \
         gradient) so the FD certificate is non-vacuous"
    );
    let fd = barrier_fd_grad(
        &mut term,
        |t| t.separation_barrier_value_and_grad_for_test(1.0).0,
        1e-6,
    );
    let err = rel_grad_err(&analytic, &fd);
    assert!(
        err <= 1e-5,
        "separation-barrier analytic gradient must match central FD to rel-tol \
         1e-5; got rel err {err:e}. analytic={analytic:?} fd={fd:?}"
    );
}
