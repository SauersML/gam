//! #1939 OBJECTIVE-QUALITY acceptance bar — existence/intensity DECOUPLING in the
//! physical dictionary. The current representation carries intensity directly in
//! each fitted decoder. *Existence* (does the data support this atom at all) must
//! be identified separately from *intensity* (how large its contribution is).
//!
//! We plant that ground truth: two live circles on DISJOINT output subspaces whose
//! amplitudes differ by ~an order of magnitude, plus a DEAD atom slot with no
//! planted signal, and fit a K=3 dictionary. The objective is truth recovery —
//! the planted amplitude ratio and the dead/alive partition — NOT reproduction of
//! any reference tool's fitted parameters.
//!
//! Intensity lives in the decoder magnitude `‖B_k‖`. Existence is a marginal
//! (evidence) question, so it is decided where the fit decides it: the outer
//! REML/LAML selection of each atom's decoder precision `λ_k`, whose vanishing
//! face removes the atom (#4325 — see the test's doc for why a hand-fixed ρ cannot
//! decide it).

use super::tests::deterministic_circle_noise;
use super::tests_cocollapse_disjoint_2027::term_from_padded_blocks_with_mode;
use super::*;
use ndarray::Axis;

/// Two circles on disjoint output-column parities with UNEQUAL amplitudes, plus a
/// third (dead) subspace that carries no planted signal — returned UN-whitened so
/// the planted amplitudes survive into the target (a column-standardized target
/// would quotient exactly the intensity this test measures).
///
/// * circle A: even output columns, amplitude `amp_a` (the strong atom),
/// * circle B: odd output columns, amplitude `amp_b` (the weak-but-real atom),
/// * the remaining variance is small isotropic noise (no third circle).
fn two_unequal_circles_plus_dead(
    n: usize,
    p: usize,
    amp_a: f64,
    amp_b: f64,
    sigma: f64,
) -> Array2<f64> {
    let mut fa = Array2::<f64>::zeros((2, p));
    let mut fb = Array2::<f64>::zeros((2, p));
    for j in 0..p {
        if j % 2 == 0 {
            fa[[0, j]] = deterministic_circle_noise(j, 0);
            fa[[1, j]] = deterministic_circle_noise(j, 1);
        } else {
            fb[[0, j]] = deterministic_circle_noise(j, 2);
            fb[[1, j]] = deterministic_circle_noise(j, 3);
        }
    }
    // Orthonormalize each planted 2-frame so the circle's ambient radius is 1 and
    // the decoded amplitude is exactly `amp_*` (not tangled with frame scale).
    for f in [&mut fa, &mut fb] {
        for r in 0..2 {
            let nrm = (0..p).map(|j| f[[r, j]] * f[[r, j]]).sum::<f64>().sqrt();
            for j in 0..p {
                f[[r, j]] /= nrm.max(1.0e-300);
            }
        }
    }
    let mut z = Array2::<f64>::zeros((n, p));
    for row in 0..n {
        let ta = std::f64::consts::TAU * (row as f64) / (n as f64);
        // Use frequency 3 for the weak circle. Each fitted circle basis reaches
        // only harmonic 2, so the strong frequency-1 chart cannot also absorb the
        // weak signal as its second harmonic. A frequency-2 planting would make
        // decoder ownership non-identifiable before existence/intensity is tested.
        let tb = std::f64::consts::TAU * (3.0 * row as f64 + 0.37) / (n as f64);
        let (ca, sa) = (ta.cos(), ta.sin());
        let (cb, sb) = (tb.cos(), tb.sin());
        for j in 0..p {
            z[[row, j]] = amp_a * (ca * fa[[0, j]] + sa * fa[[1, j]])
                + amp_b * (cb * fb[[0, j]] + sb * fb[[1, j]])
                + sigma * deterministic_circle_noise(row, j + 7);
        }
    }
    z
}

/// Build a fresh K periodic term with the production PCA coordinates and joint
/// decoder-LSQ seed at the given atom count on the UN-whitened target.
fn kterm_periodic(target: &Array2<f64>, k: usize, m: usize) -> SaeManifoldTerm {
    let n = target.nrows();
    let p = target.ncols();
    let d = 1usize;
    let basis_kinds = vec![SaeAtomBasisKind::Periodic; k];
    let dims = vec![d; k];
    let seed = sae_pca_seed_initial_coords(target.view(), &basis_kinds, &dims)
        .expect("the planted target has full enough rank to seed k 1-D atoms");
    let evaluator = Arc::new(
        PeriodicHarmonicEvaluator::new(m)
            .expect("an odd harmonic count is a valid periodic basis size"),
    );

    let mut basis_values = Array3::<f64>::zeros((k, n, m));
    let mut basis_jacobian = Array4::<f64>::zeros((k, n, m, d));
    let mut penalties = Array3::<f64>::zeros((k, m, m));
    let mut coords_vec: Vec<Array2<f64>> = Vec::new();
    for atom in 0..k {
        let coords = seed.slice(s![atom, .., 0..d]).to_owned();
        let (phi, jet) = evaluator
            .evaluate(coords.view())
            .expect("fixture coords are already wrapped into the evaluator's unit period");
        basis_values.slice_mut(s![atom, .., ..]).assign(&phi);
        basis_jacobian.slice_mut(s![atom, .., .., ..]).assign(&jet);
        penalties
            .slice_mut(s![atom, .., ..])
            .assign(&Array2::<f64>::eye(m));
        coords_vec.push(coords);
    }
    let logits = Array2::<f64>::zeros((n, k));
    let basis_sizes = vec![m; k];
    let decoder = sae_decoder_lsq_init(
        basis_values.view(),
        &basis_sizes,
        target.view(),
        logits.view(),
        "ordered_beta_bernoulli",
        1.0,
        1.0,
        0.0,
        None,
    )
    .expect("least-squares decoder init: basis blocks and target share n rows");
    let mut evaluators: Vec<Option<Arc<dyn SaeBasisSecondJet>>> = Vec::new();
    for _ in 0..k {
        evaluators.push(Some(evaluator.clone()));
    }
    term_from_padded_blocks_with_mode(
        n,
        p,
        &basis_kinds,
        basis_values.view(),
        basis_jacobian.view(),
        &basis_sizes,
        &dims,
        decoder.view(),
        penalties.view(),
        logits.view(),
        &coords_vec,
        AssignmentMode::ordered_beta_bernoulli(1.0, 1.0, false),
        &evaluators,
    )
    .expect("fixture term: every atom's basis width matches its assignment block")
}

/// Per-atom RMS of the GATED decoded contribution `a_ik · (Φ_k B_k)_i` over all
/// rows and output columns — the atom's physical intensity as it enters the
/// reconstruction (existence × intensity combined into an output-energy readout).
fn per_atom_contribution_rms(term: &SaeManifoldTerm) -> Vec<f64> {
    let n = term.n_obs();
    let p = term.output_dim();
    let k = term.k_atoms();
    let mut sumsq = vec![0.0_f64; k];
    let mut buf = vec![0.0_f64; p];
    for row in 0..n {
        let weights = term
            .assignment
            .try_assignments_row(row)
            .expect("row index is below n_obs and the buffer is k_atoms wide");
        for atom in 0..k {
            let a_k = weights[atom];
            term.atoms[atom].fill_decoded_row(row, &mut buf);
            for &g in buf.iter() {
                let v = a_k * g;
                sumsq[atom] += v * v;
            }
        }
    }
    sumsq
        .into_iter()
        .map(|s| (s / (n as f64 * p as f64)).sqrt())
        .collect()
}

/// Even-column energy fraction of an atom's decoder — the subspace fingerprint
/// used to match a fitted atom to circle A (even, ~1.0) vs circle B (odd, ~0.0).
fn even_energy_fraction(atom: &SaeManifoldAtom, p: usize) -> f64 {
    let b = atom.decoder_coefficients();
    let mut e_even = 0.0_f64;
    let mut e_odd = 0.0_f64;
    for col in 0..b.nrows() {
        for out in 0..p {
            let v = b[[col, out]] * b[[col, out]];
            if out % 2 == 0 {
                e_even += v;
            } else {
                e_odd += v;
            }
        }
    }
    e_even / (e_even + e_odd).max(1.0e-300)
}

/// OBJECTIVE BAR (reachable) — a K=3 fit of two unequal circles + one dead slot
/// recovers the planted structure: the reconstruction is faithful, the two live
/// atoms land on the two planted (disjoint-parity) subspaces with intensities in
/// the planted ~8:1 ratio, and the dead slot is separately identified as
/// EXISTENCE-negative even though the WEAK live atom is also small in magnitude.
/// That last clause is the #1939 payoff: existence and intensity are decoupled, so
/// "small" (weak circle) is not confused with "absent" (dead slot).
///
/// WHERE EXISTENCE IS DECIDED (#4325). This bar used to fit at a hand-fixed
/// `ρ = (0, −6, 0)` and demand that the dead slot's leave-one-atom-out EV fall
/// below `0.25·loao(weak)`. The fixed-ρ objective does not imply that. At
/// `λ_smooth = e⁻⁶` every decoder prior `B_k ~ N(0, (λ_k S_k)⁻¹)`, `S_k = I`, is
/// essentially flat, so a third atom placed on the weak circle's subspace buys
/// reconstruction with its `m·p` decoder coefficients and its `n` chart
/// coordinates, and the only term pricing it at fixed ρ is the ordered
/// Beta–Bernoulli gate prior. The certified inner optima measured on #4325 show
/// the gain wins: total 194.261 with three live atoms against 194.380 with the
/// slot empty. No seed and no bound on that state can make a fixed-ρ fit
/// existence-negative, because the objective it certifies prefers the opposite.
///
/// Existence is a marginal question, and the fit prices it in the outer
/// REML/LAML criterion over the per-atom precisions `λ_k`. Given charts and
/// gates, each decoder block is linear-Gaussian, so for one output column the
/// criterion in `λ_k` (others held) is the Gaussian marginal likelihood with
/// covariance `C = φI + Σ_j Φ_j Φ_jᵀ/λ_j`. For a scalar basis direction with
/// sparsity `s = φ_kᵀ C₋ₖ⁻¹ φ_k` and quality `q = φ_kᵀ C₋ₖ⁻¹ z` (`C₋ₖ` the
/// covariance without atom k), the criterion is
/// `½[ln λ − ln(λ + s) + q²/(λ + s)]`, which is stationary at the finite
/// `λ = s²/(q² − s)` when `q² > s` and increases toward `λ → ∞` otherwise. A
/// surplus atom duplicating the weak circle after atom 1 has explained it sees
/// only noise in `q` (`q² ≲ s`), so its evidence optimum is the vanishing face.
/// The weak circle's own atom has `q² ≫ s` (its signal-to-noise ratio is
/// `amp_b²/σ² = 2500`) and keeps a finite `λ`. The production entry removes
/// an atom whose outer search reaches that face (`fit_outer_stage_to_boundary` →
/// `vanished_disposition` → restart at `K − 1`), so the bar runs the single fit
/// entry with its outer ρ search and asserts that exactly the two planted atoms
/// survive, with the weak one still existence-positive and in the planted
/// intensity ratio.
///
/// The fit runs on the centered target with its (zero-to-rounding) Tier-0 mean
/// installed, so the entry does not column-standardize: standardizing would
/// rescale each output column to unit RMS and erase the 8:1 intensity this bar
/// measures.
#[test]
fn existence_and_intensity_are_separately_identified_1939() {
    let n = 144usize;
    let p = 16usize;
    let m = 5usize; // [1, sin2πt, cos2πt, sin4πt, cos4πt]
    let amp_a = 8.0_f64;
    let amp_b = 1.0_f64;
    let mut target = two_unequal_circles_plus_dead(n, p, amp_a, amp_b, 0.02);
    let raw_mean = target
        .mean_axis(Axis(0))
        .expect("the fixture has n > 0 rows");
    for mut row in target.rows_mut() {
        row -= &raw_mean;
    }
    let centered_mean = target
        .mean_axis(Axis(0))
        .expect("the fixture has n > 0 rows");
    let mut term = kterm_periodic(&target, 3, m);
    term.set_tier0_mean(centered_mean)
        .expect("the centered target's mean has the output width and is finite");

    // The seed ρ only; the outer REML/LAML search selects every λ_k from here.
    let initial_rho = SaeManifoldRho::new(
        0.0,
        -6.0,
        vec![
            Array1::<f64>::zeros(1),
            Array1::<f64>::zeros(1),
            Array1::<f64>::zeros(1),
        ],
    );
    let outcome = run_sae_manifold_fit(SaeFitRequest {
        reconstruction_optimism_folds: None,
        base_term: term,
        target: target.clone(),
        registry: AnalyticPenaltyRegistry::new(),
        initial_rho,
        max_iter: 80,
        learning_rate: 0.05,
        ridge_ext_coord: 1.0e-3,
        ridge_beta: 1.0e-3,
        alpha: 1.0,
        isometry_pin_active: false,
        metric_provenance: "Euclidean",
        promote_from_residual: false,
        run_structure_search: false,
        run_outer_rho_search: true,
        structured_residual_passes: 0,
        cancel: None,
    })
    .unwrap_or_else(|error| panic!("the K=3 amplitude fit must converge its outer search: {error}"));
    let report = outcome
        .manifold_or_error()
        .unwrap_or_else(|error| panic!("two planted circles must keep a manifold fit: {error}"));
    let term = report.term;
    let rho = report.rho;
    let k_fit = term.k_atoms();

    let ev = term
        .dictionary_reconstruction_ev(target.view(), &rho)
        .expect("the fit converged, so dictionary EV is defined");
    let loao = term
        .per_atom_loao_explained_variance(target.view(), &rho)
        .expect("the fit converged, so per-atom LOAO EV is defined");
    let contrib = per_atom_contribution_rms(&term);
    let even_frac: Vec<f64> = term
        .atoms
        .iter()
        .map(|a| even_energy_fraction(a, p))
        .collect();
    eprintln!(
        "[#1939] K_fit={k_fit}, log_lambda_smooth={:?}, criterion={:.6}, EV={ev:.4}, \
         contrib_rms={contrib:?}, loao_ev={loao:?}, even_frac={even_frac:?}",
        rho.log_lambda_smooth, report.penalized_quasi_laplace_criterion
    );

    // Faithful reconstruction: two rank-2 circles dominate a 16-dim cloud, so an
    // honest dictionary explains most of the variance.
    assert!(
        ev > 0.80,
        "the two-circle target must be well reconstructed (EV={ev:.4})"
    );

    // (c) EXISTENCE identified SEPARATELY from intensity — the crux. The marginal
    // criterion removes the dead slot and keeps both planted circles.
    assert_eq!(
        k_fit, 2,
        "the outer criterion must remove exactly the dead slot and keep both planted circles; \
         it kept {k_fit} atoms with log λ_smooth {:?}, contributions {contrib:?}, LOAO {loao:?}",
        rho.log_lambda_smooth
    );

    let (strong, weak) = if contrib[0] >= contrib[1] { (0, 1) } else { (1, 0) };

    // (b) INTENSITY RECOVERY — the two live atoms carry the planted amplitude
    // ratio. The gate weight is common (uniform-ish ordered_beta_bernoulli gate), so the ratio of
    // decoded contribution RMS tracks amp_a/amp_b; allow a generous factor since
    // fit noise and the shared gate perturb the absolute scale but not the order.
    let planted_ratio = amp_a / amp_b;
    let recovered_ratio = contrib[strong] / contrib[weak].max(1.0e-300);
    eprintln!("[#1939] planted amp ratio={planted_ratio:.2}, recovered={recovered_ratio:.2}");
    assert!(
        recovered_ratio > 2.0 && recovered_ratio < 4.0 * planted_ratio,
        "the live intensity ratio {recovered_ratio:.2} must recover the planted order of \
         magnitude {planted_ratio:.2} (strong ≫ weak, not collapsed to parity)"
    );

    // The two live atoms occupy OPPOSITE planted subspaces (one even-dominant, one
    // odd-dominant) — they recovered distinct circles, not one shared basin.
    let (lo, hi) = if even_frac[strong] <= even_frac[weak] {
        (even_frac[strong], even_frac[weak])
    } else {
        (even_frac[weak], even_frac[strong])
    };
    assert!(
        lo < 0.5 && hi > 0.5,
        "the two live atoms must separate onto the two planted (disjoint-parity) circles; \
         even-fractions strong={:.3} weak={:.3}",
        even_frac[strong],
        even_frac[weak]
    );

    // The WEAK atom survived the same criterion that removed the dead slot, and it
    // carries real variance: removing it loses its circle's share of the target.
    let loao_weak = loao[weak].unwrap_or(0.0);
    eprintln!("[#1939] loao(weak live)={loao_weak:.4}");
    assert!(
        loao_weak > 0.01,
        "the WEAK live atom must explain real variance (existence-positive); \
         loao_weak={loao_weak:.4}"
    );
}
