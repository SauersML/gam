//! #2627 — every `BetaPenaltyOp` declares a structural norm majorant, and the arrow
//! system turns it into guaranteed bounds.
//!
//! A ridge ladder can stop structurally only at a shift it can prove positive
//! definite, and a matrix-free border has no dense `H_ββ` to read that shift from.
//! Each operator therefore declares `M ≥ |P|, |P|ᵀ` from its own entries, and
//! `max_i (M·1)_i ≥ ‖P‖₂`. The pins here:
//!
//! * the declared bound covers the true spectral norm of every operator, on
//!   randomized fixtures and on planted ones where the bound is attained;
//! * on the planted fixtures it is TIGHT, so a vacuous declaration (say `+∞`) does
//!   not pass as coverage;
//! * a declaration that underestimates (the diagonal alone) fails the same check,
//!   for the reason the message names;
//! * rescaling the operator scales the bound: bit-exactly by `2^±20`, and within
//!   the derived rounding band at `10^±6`;
//! * a sum of operators on disjoint blocks is bounded by its largest block.

#![cfg(test)]

use super::*;
use gam_linalg::faer_ndarray::FaerEigh;
use gam_linalg::roundoff::{accumulation_growth, symmetric_spectrum_rounding_band};
use ndarray::s;

/// Deterministic entries in `[-1, 1)`.
fn uniform_entries(seed: u64, count: usize) -> Vec<f64> {
    let mut state = seed
        .wrapping_mul(0x9E37_79B9_7F4A_7C15)
        .wrapping_add(0x2545_F491_4F6C_DD1D);
    let mut entries = Vec::with_capacity(count);
    for index in 0..count {
        state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407 ^ index as u64);
        entries.push(((state >> 11) as f64) / ((1u64 << 53) as f64) * 2.0 - 1.0);
    }
    entries
}

fn random_matrix(seed: u64, rows: usize, cols: usize) -> Array2<f64> {
    Array2::from_shape_vec((rows, cols), uniform_entries(seed, rows * cols)).expect("shape")
}

fn random_symmetric(seed: u64, n: usize) -> Array2<f64> {
    let raw = random_matrix(seed, n, n);
    Array2::from_shape_fn((n, n), |(i, j)| if i <= j { raw[[i, j]] } else { raw[[j, i]] })
}

/// `‖P‖₂` from the symmetric dilation `[[0, P], [Pᵀ, 0]]`, whose eigenvalues are
/// `±σ_i(P)`, together with the eigensolver's rounding band `2K·ε·σ_max`.
fn spectral_norm_with_band(dense: &Array2<f64>) -> (f64, f64) {
    let k = dense.nrows();
    let mut dilation = Array2::<f64>::zeros((2 * k, 2 * k));
    dilation.slice_mut(s![..k, k..]).assign(dense);
    dilation.slice_mut(s![k.., ..k]).assign(&dense.t());
    let eigenvalues = dilation.eigh(Side::Lower).expect("dilation EVD").0;
    let reference = eigenvalues.iter().fold(0.0_f64, |acc, value| acc.max(value.abs()));
    (
        reference,
        symmetric_spectrum_rounding_band(eigenvalues.as_slice().expect("contiguous")),
    )
}

/// The declared `max_i r_i` with each row sum made guaranteed, and the depth.
fn declared_norm_bound(op: &dyn BetaPenaltyOp) -> (f64, usize) {
    let k = op.dim();
    let ones = vec![1.0_f64; k];
    let mut row_sums = vec![0.0_f64; k];
    let depth = op.accumulate_abs_majorant_matvec(&ones, &mut row_sums);
    let bound = row_sums
        .iter()
        .map(|&sum| guaranteed_norm_upper_bound(sum, depth))
        .fold(0.0_f64, f64::max);
    (bound, depth)
}

/// `Ok((declared, reference, band, depth))` when the declaration covers the
/// reference norm up to the reference's own rounding band.
fn check_declared_bound_covers_spectral_norm(
    name: &str,
    op: &dyn BetaPenaltyOp,
) -> Result<(f64, f64, f64, usize), String> {
    let (declared, depth) = declared_norm_bound(op);
    let (reference, band) = spectral_norm_with_band(&op.to_dense());
    if declared >= reference - band {
        Ok((declared, reference, band, depth))
    } else {
        Err(format!(
            "{name}: declared bound {declared:e} underestimates the spectral norm {reference:e} \
             (reference band {band:e})"
        ))
    }
}

fn carrier_runs(seed: u64, runs: &[(usize, usize)]) -> Vec<(usize, Vec<f64>)> {
    runs.iter()
        .enumerate()
        .map(|(index, &(start, len))| (start, uniform_entries(seed + index as u64, len)))
        .collect()
}

fn coupled_carrier_fixture(seed: u64, k: usize, scale: f64) -> CoupledCarrierPenaltyOp {
    CoupledCarrierPenaltyOp {
        k,
        coupling: random_symmetric(seed, 3) * scale,
        carriers: vec![
            carrier_runs(seed + 10, &[(0, 3), (6, 2)]),
            carrier_runs(seed + 20, &[(2, 4)]),
            carrier_runs(seed + 30, &[(8, 4)]),
        ],
    }
}

fn identity_right_kronecker_fixture(seed: u64, scale: f64) -> IdentityRightKroneckerPenaltyOp {
    IdentityRightKroneckerPenaltyOp {
        factor_a: random_symmetric(seed, 4) * scale,
        p: 3,
        global_offset: 1,
        k: 14,
    }
}

fn sparse_block_fixture(seed: u64) -> SparseBlockKroneckerPenaltyOp {
    let cross = random_matrix(seed + 1, 3, 3);
    SparseBlockKroneckerPenaltyOp {
        p: 2,
        dim_a: 6,
        k: 12,
        blocks: vec![
            SparseGBlock { row_off: 0, col_off: 0, data: random_symmetric(seed, 3) },
            SparseGBlock { row_off: 0, col_off: 3, data: cross.clone() },
            SparseGBlock { row_off: 3, col_off: 0, data: cross.t().to_owned() },
            SparseGBlock { row_off: 3, col_off: 3, data: random_symmetric(seed + 2, 3) },
            // A repeated placement: the majorant must sum it, not keep one copy.
            SparseGBlock { row_off: 0, col_off: 0, data: random_symmetric(seed + 3, 3) },
        ],
    }
}

fn factored_frame_fixture(seed: u64, scale: f64) -> FactoredFrameKroneckerOp {
    let cross_g = random_matrix(seed + 1, 3, 2);
    let cross_w = random_matrix(seed + 2, 2, 1);
    FactoredFrameKroneckerOp::new(
        vec![2, 1],
        vec![3, 2],
        vec![
            FactoredFrameGBlock {
                atom_i: 0,
                atom_j: 0,
                g: random_symmetric(seed, 3) * scale,
                w: random_symmetric(seed + 3, 2),
            },
            FactoredFrameGBlock {
                atom_i: 0,
                atom_j: 1,
                g: cross_g.clone() * scale,
                w: cross_w.clone(),
            },
            FactoredFrameGBlock {
                atom_i: 1,
                atom_j: 0,
                g: cross_g.t().to_owned() * scale,
                w: cross_w.t().to_owned(),
            },
            FactoredFrameGBlock {
                atom_i: 1,
                atom_j: 1,
                g: random_symmetric(seed + 4, 2) * scale,
                w: random_symmetric(seed + 5, 1),
            },
        ],
    )
    .expect("factored frame fixture")
}

fn matvec_diag_fixture(seed: u64) -> MatvecDiagPenaltyOp {
    let matrix = random_symmetric(seed, 7);
    let diagonal = matrix.diag().to_owned();
    let applied = matrix.clone();
    let matvec: SharedBetaMatvec =
        Arc::new(move |input: ArrayView1<'_, f64>, output: &mut Array1<f64>| {
            output.assign(&applied.dot(&input));
        });
    MatvecDiagPenaltyOp::new(7, matvec, diagonal)
}

/// Every gam-solve operator, randomized at `seed`.
fn randomized_operators(seed: u64) -> Vec<(String, Arc<dyn BetaPenaltyOp>)> {
    let lonely_cross = SparseBlockKroneckerPenaltyOp {
        p: 2,
        dim_a: 6,
        k: 12,
        blocks: vec![SparseGBlock {
            row_off: 0,
            col_off: 3,
            data: random_matrix(seed + 40, 3, 3),
        }],
    };
    let composite = CompositePenaltyOp {
        k: 14,
        ops: vec![
            Arc::new(DensePenaltyOp(random_symmetric(seed + 50, 14))),
            Arc::new(identity_right_kronecker_fixture(seed + 51, 1.0)),
            Arc::new(coupled_carrier_fixture(seed + 52, 14, 1.0)),
        ],
    };
    vec![
        ("dense".to_string(), Arc::new(DensePenaltyOp(random_symmetric(seed, 9)))),
        ("coupled carrier".to_string(), Arc::new(coupled_carrier_fixture(seed, 12, 1.0))),
        (
            "identity-right kronecker".to_string(),
            Arc::new(identity_right_kronecker_fixture(seed, 1.0)),
        ),
        ("sparse block kronecker".to_string(), Arc::new(sparse_block_fixture(seed))),
        (
            "sparse block kronecker, unpaired placement".to_string(),
            Arc::new(lonely_cross),
        ),
        ("factored frame kronecker".to_string(), Arc::new(factored_frame_fixture(seed, 1.0))),
        ("composite".to_string(), Arc::new(composite)),
        ("matvec-diagonal adapter".to_string(), Arc::new(matvec_diag_fixture(seed))),
    ]
}

/// Operators whose declared bound is attained: all-positive rank-one structure,
/// where every row sum of `M` equals `‖P‖₂`.
fn planted_operators() -> Vec<(String, Arc<dyn BetaPenaltyOp>, f64)> {
    vec![
        (
            "dense 11ᵀ".to_string(),
            Arc::new(DensePenaltyOp(Array2::<f64>::ones((8, 8)))),
            8.0,
        ),
        (
            "coupled carrier 2·11ᵀ".to_string(),
            Arc::new(CoupledCarrierPenaltyOp {
                k: 6,
                coupling: Array2::from_elem((1, 1), 2.0),
                carriers: vec![vec![(0, vec![1.0; 6])]],
            }),
            12.0,
        ),
        (
            "identity-right kronecker 11ᵀ ⊗ I₂".to_string(),
            Arc::new(IdentityRightKroneckerPenaltyOp {
                factor_a: Array2::<f64>::ones((4, 4)),
                p: 2,
                global_offset: 0,
                k: 8,
            }),
            4.0,
        ),
        (
            "sparse block kronecker 11ᵀ ⊗ I₂".to_string(),
            Arc::new(SparseBlockKroneckerPenaltyOp {
                p: 2,
                dim_a: 3,
                k: 6,
                blocks: vec![SparseGBlock {
                    row_off: 0,
                    col_off: 0,
                    data: Array2::<f64>::ones((3, 3)),
                }],
            }),
            3.0,
        ),
        (
            "factored frame kronecker 11ᵀ ⊗ 11ᵀ".to_string(),
            Arc::new(
                FactoredFrameKroneckerOp::new(
                    vec![2],
                    vec![2],
                    vec![FactoredFrameGBlock {
                        atom_i: 0,
                        atom_j: 0,
                        g: Array2::<f64>::ones((2, 2)),
                        w: Array2::<f64>::ones((2, 2)),
                    }],
                )
                .expect("planted frame"),
            ),
            4.0,
        ),
    ]
}

/// A declaration that reports only the diagonal of `|P|`: the mutant control.
struct DiagonalOnlyDeclaration(DensePenaltyOp);

impl BetaPenaltyOp for DiagonalOnlyDeclaration {
    fn dim(&self) -> usize {
        self.0.dim()
    }

    fn matvec(&self, x: &[f64], y: &mut [f64]) {
        self.0.matvec(x, y);
    }

    fn gradient(&self, beta: &[f64], out: &mut [f64]) {
        self.0.gradient(beta, out);
    }

    fn diagonal(&self, diag: &mut [f64]) {
        self.0.diagonal(diag);
    }

    fn block(&self, id: BetaBlockId, offsets: &[Range<usize>], out: &mut Array2<f64>) {
        self.0.block(id, offsets, out);
    }

    fn to_dense(&self) -> Array2<f64> {
        self.0.to_dense()
    }

    fn fingerprint(&self, hasher: &mut Fingerprinter) {
        self.0.fingerprint(hasher);
    }

    fn accumulate_abs_majorant_matvec(&self, x: &[f64], out: &mut [f64]) -> usize {
        for (index, target) in out.iter_mut().enumerate() {
            *target += self.0.0[[index, index]].abs() * x[index];
        }
        1
    }
}

/// The naive sum of ten copies of `fl(0.1)` rounds BELOW the exact sum of those ten
/// representable values (`fl(0.1) > 1/10`, so the exact sum exceeds 1, while the
/// rounded sum is `1 − 2⁻⁵³`). The inflation must lift it past the exact value,
/// which lies below the next representable number above 1.
#[test]
fn guaranteed_norm_upper_bound_covers_a_sum_that_rounds_below_its_exact_value_2627() {
    let mut naive = 0.0_f64;
    for term in [0.1_f64; 10] {
        naive += term;
    }
    assert!(
        naive < 1.0,
        "positive control: the naive sum {naive:e} must undercut 1, which the exact sum exceeds"
    );
    let guaranteed = guaranteed_norm_upper_bound(naive, 10);
    assert!(
        guaranteed >= 1.0_f64.next_up(),
        "the guaranteed bound {guaranteed:e} must cover the exact sum, which exceeds 1 by less \
         than one ulp"
    );
    assert_eq!(
        guaranteed_norm_upper_bound(1.0, usize::MAX / 2),
        f64::INFINITY,
        "an accumulation too deep for γ to exist carries no bound"
    );
}

#[test]
fn every_gam_solve_operator_declares_a_covering_norm_bound_2627() {
    let mut checked = 0usize;
    for seed in 0..5u64 {
        for (name, op) in randomized_operators(seed) {
            let result = check_declared_bound_covers_spectral_norm(&name, op.as_ref());
            assert!(result.is_ok(), "seed {seed}: {result:?}");
            checked += 1;
        }
    }
    assert_eq!(checked, 40, "every operator at every seed must be checked");
}

#[test]
fn planted_operators_attain_their_declared_bound_2627() {
    for (name, op, exact_norm) in planted_operators() {
        let result = check_declared_bound_covers_spectral_norm(&name, op.as_ref());
        assert!(result.is_ok(), "{result:?}");
        let (declared, reference, band, depth) = result.expect("checked above");
        assert!(
            (reference - exact_norm).abs() <= band,
            "{name}: the reference {reference:e} must resolve the planted norm {exact_norm}"
        );
        // `declared ≤ exact·(1 + γ_d)(1 + γ_{2d+3})(1 + u) ≤ exact·(1 + γ_{3d+5})`.
        assert!(
            declared <= (reference + band) * (1.0 + accumulation_growth(3 * depth + 5)),
            "{name}: declared {declared:e} is not tight against the attained norm {reference:e}"
        );
    }
}

#[test]
fn an_underestimating_declaration_fails_the_coverage_check_2627() {
    let planted = Array2::<f64>::ones((8, 8));
    let (reference, band) = spectral_norm_with_band(&planted);
    assert!(
        (reference - 8.0).abs() <= band,
        "premise: the reference must resolve ‖11ᵀ‖₂ = 8, got {reference:e} (band {band:e})"
    );
    let honest = DensePenaltyOp(planted.clone());
    let honest_result = check_declared_bound_covers_spectral_norm("honest dense 11ᵀ", &honest);
    assert!(honest_result.is_ok(), "control: {honest_result:?}");
    let mutant = DiagonalOnlyDeclaration(DensePenaltyOp(planted));
    let failure = check_declared_bound_covers_spectral_norm("diagonal-only 11ᵀ", &mutant)
        .expect_err("the diagonal-only declaration must fail coverage");
    assert!(
        failure.starts_with("diagonal-only 11ᵀ: declared bound")
            && failure.contains("underestimates the spectral norm"),
        "the mutant must fail for underestimating ‖P‖₂, got: {failure}"
    );
}

/// `2^±20` scales every entry, partial sum and product exactly, so the bound scales
/// bit-exactly. A decimal scale rounds each entry once, so the two bounds agree to
/// `γ_{2d+5}`: `(1 + θ_{d+1})(1 + θ_1) / ((1 + θ_d)(1 + θ_1)(1 + θ_1))`, and one more
/// for the division that forms the ratio.
#[test]
fn a_rescaled_operator_rescales_its_declared_bound_2627() {
    for &(scale, exact) in &[(2.0_f64.powi(20), true), (2.0_f64.powi(-20), true), (1e6, false), (1e-6, false)] {
        let pairs: Vec<(&str, Arc<dyn BetaPenaltyOp>, Arc<dyn BetaPenaltyOp>)> = vec![
            (
                "dense",
                Arc::new(DensePenaltyOp(random_symmetric(7, 9))),
                Arc::new(DensePenaltyOp(random_symmetric(7, 9) * scale)),
            ),
            (
                "coupled carrier",
                Arc::new(coupled_carrier_fixture(7, 12, 1.0)),
                Arc::new(coupled_carrier_fixture(7, 12, scale)),
            ),
            (
                "identity-right kronecker",
                Arc::new(identity_right_kronecker_fixture(7, 1.0)),
                Arc::new(identity_right_kronecker_fixture(7, scale)),
            ),
            (
                "factored frame kronecker",
                Arc::new(factored_frame_fixture(7, 1.0)),
                Arc::new(factored_frame_fixture(7, scale)),
            ),
        ];
        for (name, base, scaled) in pairs {
            let (base_bound, depth) = declared_norm_bound(base.as_ref());
            let (scaled_bound, scaled_depth) = declared_norm_bound(scaled.as_ref());
            assert_eq!(depth, scaled_depth, "{name}: a rescale must not change the depth");
            if exact {
                assert_eq!(
                    scaled_bound.to_bits(),
                    (scale * base_bound).to_bits(),
                    "{name}: scale {scale:e} must rescale the bound bit-exactly \
                     ({scaled_bound:e} vs {:e})",
                    scale * base_bound
                );
            } else {
                let ratio = scaled_bound / (scale * base_bound);
                assert!(
                    (ratio - 1.0).abs() <= accumulation_growth(2 * depth + 5),
                    "{name}: scale {scale:e} moved the bound by {:e}, past γ_(2d+5) = {:e}",
                    ratio - 1.0,
                    accumulation_growth(2 * depth + 5)
                );
            }
        }
    }
}

/// Two `11ᵀ` blocks on disjoint ranges: `‖P‖₂ = 4`. Summing per-operator scalar
/// norms would declare 8; the row-sum vectors add, so the declaration is 4.
#[test]
fn a_composite_of_disjoint_blocks_is_bounded_by_its_largest_block_2627() {
    let block = |offset: usize| -> Arc<dyn BetaPenaltyOp> {
        Arc::new(IdentityRightKroneckerPenaltyOp {
            factor_a: Array2::<f64>::ones((4, 4)),
            p: 1,
            global_offset: offset,
            k: 8,
        })
    };
    let composite = CompositePenaltyOp { k: 8, ops: vec![block(0), block(4)] };
    let result = check_declared_bound_covers_spectral_norm("disjoint composite", &composite);
    assert!(result.is_ok(), "{result:?}");
    let (declared, reference, band, depth) = result.expect("checked above");
    assert!((reference - 4.0).abs() <= band, "reference {reference:e} must resolve 4");
    assert!(
        declared <= 4.0 * (1.0 + accumulation_growth(3 * depth + 5)),
        "the disjoint composite declared {declared:e}; it must be the largest block's 4, not \
         the sum of the blocks' norms"
    );
}

/// The system accessor follows the same operator → dense dispatch as the solve: a
/// structured operator and a dense `hbb` holding the same matrix declare the same
/// majorant row sums at the same depth, and their guaranteed maximum covers `‖H_ββ‖₂`.
#[test]
fn the_shared_block_majorant_follows_the_installed_operator_2627() {
    let matrix = random_symmetric(11, 9);
    let mut structured =
        ArrowSchurSystem::new_with_per_row_dims_empty_hbb_and_htbeta_cols(Vec::new(), 9, 0);
    structured.set_penalty_op(Arc::new(DensePenaltyOp(matrix.clone())));
    let mut dense = ArrowSchurSystem::new(0, 1, 9);
    dense.hbb = matrix.clone();
    let ones = vec![1.0_f64; 9];
    let mut structured_rows = vec![0.0_f64; 9];
    let mut dense_rows = vec![0.0_f64; 9];
    let structured_depth = structured.shared_block_abs_majorant_matvec(&ones, &mut structured_rows);
    let dense_depth = dense.shared_block_abs_majorant_matvec(&ones, &mut dense_rows);
    assert_eq!(structured_rows, dense_rows, "one matrix must declare one row-sum vector");
    assert_eq!(structured_depth, dense_depth, "one matrix must declare one depth");
    let (reference, band) = spectral_norm_with_band(&matrix);
    let declared = structured_rows
        .iter()
        .map(|&sum| guaranteed_norm_upper_bound(sum, structured_depth))
        .fold(0.0_f64, f64::max);
    assert!(
        declared >= reference - band,
        "the system's declared {declared:e} must cover ‖H_ββ‖₂ = {reference:e}"
    );
}
