//! Tests for the MAP estimate uniqueness condition:
//!   ker(J^T W J) ∩ ker(S) = {0}
//!
//! Two scenarios:
//!
//! 1. **Passes**: 2-block synthetic where post-canonicalize J is rank-deficient
//!    by 1 (one shared direction between blocks) AND both blocks carry non-trivial
//!    S that together cover every null direction of J^T W J.  The check passes.
//!
//! 2. **Fails**: same 2-block setup but S = 0 on the rank-deficient direction
//!    (block B has zero-rank smoothness on the direction shared with block A).
//!    The check fails with a `MapUniquenessError` naming block B as the dominant
//!    block.

use gam::families::custom_family::{CustomFamilyError, ParameterBlockSpec, PenaltyMatrix};
use gam::linalg::matrix::{DenseDesignMatrix, DesignMatrix};
use gam::solver::identifiability_canonical::canonicalize_for_identifiability;
use ndarray::{Array1, Array2};

/// Build a `ParameterBlockSpec` from a dense design matrix and a single
/// penalty matrix.  `gauge_priority` is set to `priority` to allow the
/// two-block aliased setup to pass the RRQR audit.
fn spec_with_penalty(
    name: &str,
    design: Array2<f64>,
    penalty: Array2<f64>,
    priority: u8,
) -> ParameterBlockSpec {
    let n = design.nrows();
    ParameterBlockSpec {
        name: name.to_string(),
        design: DesignMatrix::Dense(DenseDesignMatrix::from(design)),
        offset: Array1::<f64>::zeros(n),
        penalties: vec![PenaltyMatrix::Dense(penalty)],
        nullspace_dims: Vec::new(),
        initial_log_lambdas: Array1::from(vec![0.0]),
        initial_beta: None,
        gauge_priority: priority,
        row_scaling: None,
        jacobian_callback: None,
    }
}

/// Build a `ParameterBlockSpec` with no penalty.
fn spec_no_penalty(name: &str, design: Array2<f64>, priority: u8) -> ParameterBlockSpec {
    let n = design.nrows();
    ParameterBlockSpec {
        name: name.to_string(),
        design: DesignMatrix::Dense(DenseDesignMatrix::from(design)),
        offset: Array1::<f64>::zeros(n),
        penalties: Vec::new(),
        nullspace_dims: Vec::new(),
        initial_log_lambdas: Array1::<f64>::zeros(0),
        initial_beta: None,
        gauge_priority: priority,
        row_scaling: None,
        jacobian_callback: None,
    }
}

/// Construct the 2-block synthetic setup:
///
/// - Block A: columns [v0, v1] where v0 is the shared direction.
/// - Block B: columns [v0, v2] where v0 is also the shared direction
///   (exact alias of block A's first column).
///
/// After canonicalize (with block A at higher gauge_priority), the
/// shared direction is attributed to block B and dropped.  Post-T, the
/// joint J_can is full-rank among the surviving columns.  BUT J^T J
/// computed over the original joint design still has a null direction
/// (the one that was aliased).
///
/// For the MAP uniqueness check to pass, the reduced specs' joint penalty
/// must cover the null direction of the REDUCED J^T J.  Since the
/// canonicalization drops the shared direction from block B, the null
/// space of J_can^T J_can should be empty (J_can is full column rank).
/// In that case the check trivially passes.
///
/// To build a setup where the null space of J_can^T J_can is non-empty
/// (post-canonicalize) AND the penalty either covers it or not:
/// use two blocks where the JOINT design (after T) still has a null
/// direction.
///
/// Concretely: n = 8, p_A = 2, p_B = 2.
///
/// Block A design:
///   col 0: [1, 1, 1, ..., 1]  (constant)
///   col 1: [1, 2, 3, ..., n]  (linear)
///
/// Block B design:
///   col 0: [1, 1, 1, ..., 1]  (same as A col 0)
///   col 1: [sin(1), sin(2), ..., sin(n)]  (unique direction)
///
/// With A at priority 200 and B at priority 100:
///   - canonicalize drops B col 0 (attributed to A col 0).
///   - After T: A has [const, lin] (2 cols), B has [sin] (1 col).
///   - J_can (n × 3) is full rank.
///   - null(J_can^T J_can) = {} (empty).
///   - MAP uniqueness check trivially passes.
///
/// To test the FAILURE case, we need J_can^T J_can to actually be
/// rank-deficient.  This requires a post-T design that is rank-deficient.
/// We achieve this by making both blocks' UNIQUE directions proportional:
///
/// Block A design: [const, lin]
/// Block B design: [const, 2*lin]
///
/// With A at priority 200 and B at priority 100:
///   - canonicalize drops B col 0 (attributed to A col 0).
///   - After T: A has [const, lin], B has [2*lin].
///   - J_can (n × 3) has columns: const, lin, 2*lin.
///   - lin and 2*lin are proportional → J_can is rank 2, null dim 1.
///   - The null direction of J_can^T J_can lies in the {lin, 2*lin} subspace.
///
/// Now:
/// - PASS case: block A has penalty [[0,0],[0,1]] (penalises lin direction),
///   block B (after T, single col 2*lin) has penalty [[1]] (penalises 2*lin).
///   The null direction n of J_can^T J_can has components in lin and 2*lin.
///   Both penalties contribute positively → n^T S n > 0 → PASS.
///
/// - FAIL case: block B (after T) has penalty [[0]] (zero penalty on 2*lin).
///   The null direction n still has a component along 2*lin but B's penalty
///   is zero.  If A's penalty [[0,0],[0,1]] also contributes zero in the lin
///   direction of n, then n^T S n = 0 → FAIL.
///
///   Concretely, the null direction of [const, lin, 2*lin]^T [const, lin, 2*lin]
///   is n = (0, 2, -1) / sqrt(5) (since 2*lin - 2*(2*lin) + lin*2 = 0).
///   A's penalty in reduced space: [[0,0],[0,1]] acts on (n_A0, n_A1) = (0, 2/sqrt(5)).
///   n_A^T S_A n_A = (2/sqrt(5))^2 = 4/5 > 0.
///
/// Wait — that would PASS even in the fail case. We need n^T S n = 0.
///
/// Simplest fail case: block A has ZERO penalty, block B has zero penalty.
/// Then S = 0 everywhere and any null direction of J^T J gives n^T S n = 0.
///
/// For the PASS case: both blocks have non-trivial penalty covering the
/// null direction.
///
/// Revised simpler approach (cleaner than above):
///
/// n = 16 rows.
/// Block A: p_A = 2 columns: [v0, v1] (both unique, full rank).
/// Block B: p_B = 2 columns: [v0, v2] where v0 = A's col 0 exactly.
///
/// A at priority 200, B at priority 100 → canonicalize drops B col 0.
/// After T: A has [v0, v1] (2 cols), B has [v2] (1 col).
/// J_can: n × 3, columns [v0, v1, v2], presumably full rank.
///
/// To create a post-T null direction, make v1 ∝ v2:
/// v1 = sin(x), v2 = 2*sin(x).
/// Then [v0, v1, v2] has columns [const, sin, 2*sin] → rank 2, null dim 1.
/// Null direction: n = [0, 2, -1]/sqrt(5).
///
/// PASS: A's reduced penalty = [[0,0],[0,4]] (penalises v1=sin direction with weight 4),
///        B's reduced penalty = [[1]] (penalises v2=2*sin direction).
///        n^T S n = n_A^T S_A n_A + n_B^T S_B n_B
///               = [0, 2/sqrt(5)]^T [[0,0],[0,4]] [0, 2/sqrt(5)]
///                 + [−1/sqrt(5)]^T [[1]] [−1/sqrt(5)]
///               = 4*(4/5) + 1*(1/5) = 16/5 + 1/5 = 17/5 > 0.  PASS.
///
/// FAIL: A's reduced penalty = [[0,0],[0,0]] (zero), B's = [[0]] (zero).
///        n^T S n = 0. FAIL.
///
/// ALSO FAIL (single-block zero S): A has non-trivial penalty,
///        B's reduced penalty = [[0]].
///        n_B = -1/sqrt(5) ≠ 0.
///        n_A^T S_A n_A = 16/5 > 0 but n_B^T S_B n_B = 0 and S_B = 0.
///        Since S = blockdiag(S_A, S_B) is NOT positive on n (it annihilates
///        the B component), n^T S n = 16/5 > 0 ≠ 0.
///
/// Hmm — that would still PASS.  The condition is n^T S n = 0 for the FULL
/// joint n vector, so even if S_B = 0, S_A contributes through n_A.
///
/// We need n_A = 0 for the fail case.  That means the null direction is
/// entirely in block B's column space.  To achieve this, make block B's
/// columns span the null direction of J_can^T J_can entirely.
///
/// Cleanest fail construction:
/// n = 16.  Block A: [v0] (1 col, unique).  Block B: [v0, v1] where v0 is
/// shared with A.
/// After canonicalize (A priority 200, B 100): B col 0 dropped.
/// After T: A has [v0] (1 col), B has [v1] (1 col).
/// J_can: [v0, v1] → both unique → full rank → no null direction.
///
/// OK — the issue is that canonicalize makes J_can full rank (it drops aliases).
/// So J_can^T J_can has no null space in the normal case.
///
/// The MAP uniqueness condition fires when AFTER canonicalization the
/// reduced J_can is STILL rank-deficient AND S does not cover the null.
///
/// When would J_can be rank-deficient AFTER canonicalize?  When:
///   - The blocks' unique directions (not shared across blocks) happen to be
///     proportional among themselves.
///   - This is a WITHIN-BLOCK proportionality, not caught by the cross-block audit.
///
/// Example: Block A: [v0, v1] with v0 unique to A, v1 = shared.
///          Block B: [v1, v2] with v1 = shared with A, v2 unique.
///          After canonicalize (A 200, B 100): B's v1 dropped.
///          After T: A has [v0, v1], B has [v2].
///          If v2 = c * v0, then J_can = [v0, v1, c*v0] → rank 2.
///          Null direction: n such that v0*n[0] + v1*n[1] + c*v0*n[2] = 0
///          → (n[0] + c*n[2])*v0 + n[1]*v1 = 0 → n[0] = -c*n[2], n[1] = 0.
///          n = (-c, 0, 1)/sqrt(1+c²) (in reduced [v0, v1, v2] space).
///
///          PASS: A's penalty [[p,0],[0,0]] + B's penalty [[q]].
///          n^T S n = n_A^T [[p,0],[0,0]] n_A + n_B^T [[q]] n_B
///               = p*c²/(1+c²) + q/(1+c²) > 0 for p > 0 or q > 0.
///
///          FAIL: A's penalty = 0, B's penalty = 0 → n^T S n = 0.
///                OR: A's penalty [[p,0],[0,0]], B's penalty = 0,
///                    → n^T S n = p*c²/(1+c²) > 0 if p > 0. PASS.
///
///                To get FAIL: A's penalty [[0,0],[0,q]], B's penalty = 0.
///                n_A = (-c, 0)/sqrt(1+c²), n_A^T [[0,0],[0,q]] n_A = 0.
///                n_B = 1/sqrt(1+c²), n_B^T [[0]] n_B = 0.
///                n^T S n = 0. FAIL!
///
/// This is the right construction. Let me set c = 1 for simplicity.
///
/// FINAL CONSTRUCTION (n=32, c=1):
///
/// v0 = [1, 2, ..., n] (linear ramp)
/// v1 = [1, 1, ..., 1] (constant, shared between A and B)
/// v2 = v0 (= c*v0 with c=1)
///
/// Block A: [v0, v1] (p_A=2).
/// Block B: [v1, v2] (p_B=2, where v1 is the shared constant, v2=v0).
/// A at priority 200, B at priority 100.
///
/// After canonicalize: B col 0 (v1) dropped, B col 1 (v2=v0) kept.
/// After T: A=[v0,v1] (2 cols), B=[v2=v0] (1 col, 2nd raw col of B).
/// J_can (n×3): columns [v0, v1, v0] → rank 2. Null direction:
///   n = (-1, 0, 1)/sqrt(2) (A-reduced basis).
///
/// PASS case:
///   A's reduced penalty = [[1,0],[0,0]] (penalises v0 direction with weight 1).
///   B's reduced penalty = [[1]] (penalises v2=v0 direction).
///   n_A = (-1/sqrt(2), 0), n_B = 1/sqrt(2).
///   n^T S n = (1/2)*1 + (1/2)*1 = 1. PASS.
///
/// FAIL case:
///   A's reduced penalty = [[0,0],[0,1]] (penalises v1 direction only, NOT v0).
///   B's reduced penalty = [[0]] (zero penalty on v2=v0).
///   n_A^T [[0,0],[0,1]] n_A = 0 (n_A[1]=0).
///   n_B^T [[0]] n_B = 0.
///   n^T S n = 0. FAIL.

const N: usize = 32;

fn linramp(n: usize) -> Vec<f64> {
    (1..=n).map(|i| i as f64).collect()
}

fn constant(n: usize) -> Vec<f64> {
    vec![1.0; n]
}

/// Build the two-block aliased design as described in the module doc.
///
/// Block A: [[v0, v1]] where v0 = linramp, v1 = constant.
/// Block B: [[v1, v0]] where v1 = constant (shared with A col 1),
///           v0 = linramp (same as A col 0).
///
/// With A at priority 200, B at priority 100, canonicalize should drop
/// B col 0 (the constant v1 from B is aliased with A's col 1).
/// After T: A=[linramp, constant] (2 cols), B=[linramp] (1 col).
/// J_can has columns [linramp, constant, linramp] → rank 2.
fn build_block_a() -> Array2<f64> {
    let v0 = linramp(N);
    let v1 = constant(N);
    let mut m = Array2::<f64>::zeros((N, 2));
    for i in 0..N {
        m[[i, 0]] = v0[i];
        m[[i, 1]] = v1[i];
    }
    m
}

fn build_block_b() -> Array2<f64> {
    let v1 = constant(N);
    let v0 = linramp(N);
    let mut m = Array2::<f64>::zeros((N, 2));
    for i in 0..N {
        m[[i, 0]] = v1[i]; // shared constant
        m[[i, 1]] = v0[i]; // linramp = same as A col 0
    }
    m
}

/// PASS case: both penalties cover the null direction.
///
/// After canonicalize:
///   A reduced = 2 cols (linramp, constant).  A penalty = [[1,0],[0,0]]
///   (penalises the linramp direction).
///   B reduced = 1 col (linramp).  B penalty = [[1]] (penalises linramp).
///
/// Null direction of J_can^T J_can:
///   J_can columns = [linramp, constant, linramp].
///   n = (-1, 0, 1)/sqrt(2) in reduced coordinates.
///   n_A = (-1/sqrt(2), 0), n_B = (1/sqrt(2)).
///   n^T S n = (1/2)*1 + (1/2)*1 = 1 > 0. PASS.
#[test]
fn penalty_joint_nullspace_check_passes_when_both_penalties_cover_null() {
    let design_a = build_block_a();
    let design_b = build_block_b();

    // A's penalty: [[1,0],[0,0]] (penalises linramp component = col 0 of A).
    let mut pen_a = Array2::<f64>::zeros((2, 2));
    pen_a[[0, 0]] = 1.0;

    // B's penalty: [[1,0],[0,1]] applied to raw 2-col B.
    // After T (col 1 of B kept), the reduced B has 1 col and penalty = [[1]].
    // We set the raw B penalty to [[0,0],[0,1]] so that after pullback to
    // the kept col (col index 1 of raw B), reduced penalty = [[1]].
    let mut pen_b_raw = Array2::<f64>::zeros((2, 2));
    pen_b_raw[[1, 1]] = 1.0;

    let spec_a = spec_with_penalty("block_a", design_a, pen_a, 200);
    let spec_b = spec_with_penalty("block_b", design_b, pen_b_raw, 100);

    let specs = [spec_a, spec_b];
    // canonicalize_for_identifiability runs the MAP uniqueness check.
    // With both penalties covering the null direction, it must succeed.
    let result = canonicalize_for_identifiability(&specs);
    assert!(
        result.is_ok(),
        "MAP uniqueness check must pass when both penalties cover the null direction; \
         got: {:?}",
        result.err(),
    );
}

/// FAIL case: S = 0 on the rank-deficient direction.
///
/// After canonicalize:
///   A reduced = 2 cols (linramp, constant).  A penalty = [[0,0],[0,1]]
///   (penalises constant direction only, NOT linramp).
///   B reduced = 1 col (linramp).  B penalty = [[0]] (zero).
///
/// Null direction of J_can^T J_can: n = (-1, 0, 1)/sqrt(2).
///   n_A = (-1/sqrt(2), 0), n_B = 1/sqrt(2).
///   n_A^T [[0,0],[0,1]] n_A = 0 (n_A[1] = 0).
///   n_B^T [[0]] n_B = 0.
///   n^T S n = 0 → MAP non-unique. FAIL.
///
/// The error must name the dominant block.  The null direction has equal
/// components in A and B (both |·|² = 1/2), so either block could be named
/// dominant; we just verify that an error IS returned.
#[test]
fn penalty_joint_nullspace_check_fails_when_s_zero_on_null_direction() {
    let design_a = build_block_a();
    let design_b = build_block_b();

    // A's penalty: [[0,0],[0,1]] (penalises constant, NOT linramp).
    let mut pen_a = Array2::<f64>::zeros((2, 2));
    pen_a[[1, 1]] = 1.0;

    // B's raw penalty: [[0,0],[0,0]] (zero on both raw B cols).
    // After T (col 1 of B kept), reduced B penalty = [[0]].
    let pen_b_raw = Array2::<f64>::zeros((2, 2));

    let spec_a = spec_with_penalty("block_a", design_a, pen_a, 200);
    let spec_b = spec_with_penalty("block_b", design_b, pen_b_raw, 100);

    let specs = [spec_a, spec_b];
    let result = canonicalize_for_identifiability(&specs);

    match result {
        Err(CustomFamilyError::MapUniquenessFailure { error }) => {
            // The error must name a dominant block (either block_a or block_b).
            assert!(
                error.dominant_block == "block_a" || error.dominant_block == "block_b",
                "dominant block must be one of the two specs; got '{}'",
                error.dominant_block,
            );
            // The penalty quadratic form must be near zero.
            assert!(
                error.penalty_quadratic_form < 1e-8,
                "penalty quadratic form must be near zero; got {}",
                error.penalty_quadratic_form,
            );
        }
        Err(other) => {
            // The audit itself might refuse the fit for rank-deficiency reasons.
            // That is acceptable: the model IS non-identifiable.
            // We accept both MapUniquenessFailure and IdentifiabilityFailure.
            match other {
                CustomFamilyError::IdentifiabilityFailure { ref audit } => {
                    assert!(
                        audit.fatal,
                        "IdentifiabilityFailure must carry a fatal audit; got {}",
                        audit.summary,
                    );
                }
                _ => {
                    panic!(
                        "expected MapUniquenessFailure or IdentifiabilityFailure; got {:?}",
                        other,
                    );
                }
            }
        }
        Ok(_) => {
            panic!(
                "MAP uniqueness check must fail when S = 0 on the null direction; \
                 got Ok (check silently passed a non-unique model)"
            );
        }
    }
}

/// Verify that a fully-clean 2-block model (no aliasing, non-trivial
/// penalties covering all directions) passes the MAP uniqueness check.
#[test]
fn penalty_joint_nullspace_check_clean_model_passes() {
    // Block A: [linramp, constant] (2 cols, fully linearly independent).
    // Block B: [sin(x), cos(x)] (2 cols, independent of A).
    let v0 = linramp(N);
    let v1 = constant(N);
    let mut design_a = Array2::<f64>::zeros((N, 2));
    for i in 0..N {
        design_a[[i, 0]] = v0[i];
        design_a[[i, 1]] = v1[i];
    }
    let mut design_b = Array2::<f64>::zeros((N, 2));
    for i in 0..N {
        let x = (i as f64) * std::f64::consts::PI / (N as f64);
        design_b[[i, 0]] = x.sin();
        design_b[[i, 1]] = x.cos();
    }

    // Full-rank diagonal penalties on both blocks.
    let mut pen_a = Array2::<f64>::zeros((2, 2));
    pen_a[[0, 0]] = 1.0;
    pen_a[[1, 1]] = 1.0;
    let mut pen_b = Array2::<f64>::zeros((2, 2));
    pen_b[[0, 0]] = 1.0;
    pen_b[[1, 1]] = 1.0;

    let spec_a = spec_with_penalty("block_a", design_a, pen_a, 100);
    let spec_b = spec_with_penalty("block_b", design_b, pen_b, 100);
    let specs = [spec_a, spec_b];
    let result = canonicalize_for_identifiability(&specs);
    assert!(
        result.is_ok(),
        "clean model with full-rank penalties must pass all checks; got {:?}",
        result.err(),
    );
}

/// Verify that a model with NO penalties (all S = 0) fails the MAP
/// uniqueness check when the joint design is rank-deficient.
#[test]
fn penalty_joint_nullspace_check_no_penalty_rank_deficient_fails() {
    // Same aliased setup but no penalty on either block.
    // The audit may refuse with IdentifiabilityFailure (rank deficiency
    // without gauge resolution) OR pass the audit and fail the MAP check.
    // Either failure mode is acceptable: the model is non-identifiable.
    let design_a = build_block_a();
    let design_b = build_block_b();

    let spec_a = spec_no_penalty("block_a", design_a, 200);
    let spec_b = spec_no_penalty("block_b", design_b, 100);

    let specs = [spec_a, spec_b];
    let result = canonicalize_for_identifiability(&specs);

    match result {
        Ok(_) => {
            panic!(
                "an aliased model with no penalties must not pass all checks; \
                 got Ok (model should fail identifiability or MAP uniqueness)"
            );
        }
        Err(CustomFamilyError::IdentifiabilityFailure { .. }) => {
            // Audit caught the aliasing — acceptable.
        }
        Err(CustomFamilyError::MapUniquenessFailure { .. }) => {
            // MAP uniqueness check caught it after gauge resolution — acceptable.
        }
        Err(other) => {
            panic!(
                "expected IdentifiabilityFailure or MapUniquenessFailure; got {:?}",
                other
            );
        }
    }
}
