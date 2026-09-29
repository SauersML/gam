//! Physically null edits: how far a bounded edit can move the response.
//!
//! An edit direction `u` of parameter coordinates has size `uᵀ G u` under the edit Gram `G`
//! (a metric on edits, e.g. a Fisher or a Frobenius Gram of the edited tensors) and response
//! error `uᵀ K u` under the response-error Gram `K` (the Gram of the response Jacobian on
//! the declared inputs). The largest response error of an edit in the unit `G`-ball is
//!
//! ```text
//! sup { uᵀ K u : u ∈ range(G), uᵀ G u ≤ 1 } = λ_max(Λ^{-1/2} Uᵀ K U Λ^{-1/2}),
//! ```
//!
//! with `G = U Λ Uᵀ` over its resolved eigenvalues (above the eigendecomposition's band),
//! and it is attained at `u = U Λ^{-1/2} w` for the top eigenvector `w`. An edit is
//! physically null at a declared tolerance when this is below it.
//!
//! # Refusal
//!
//! A direction in `ker G` has no size, so if `K` is nonzero on it, the edit metric cannot
//! see an edit that moves the response: the supremum over the whole space is infinite and
//! no nullness claim stands. [`physically_null_supremum`] refuses then, with a
//! counterexample: the unit `u ∈ ker G` that maximizes `uᵀ K u`, whose value exceeds its own
//! band ([`null_quadratic_band`]).
//!
//! # Band
//!
//! The computed eigenpairs of `G` are exact for `G + δG` with `‖δG‖₂ ≤ β` (the spectrum
//! band). For `u` in the resolved range with `uᵀ G̃ u = 1`, `‖u‖² ≤ 1/λ_r` and so
//! `|uᵀ G u − 1| ≤ β/λ_r =: ρ`; the two balls differ by that relative factor, which moves the
//! supremum by at most `λ_max ρ/(1 − ρ)`. The reduced matrix's formation (`γ_{2n}` inner
//! products, the basis's orthonormality defect `ω` entering as `(1 + ω)`) and its own
//! spectrum band are added.

use gam_linalg::roundoff::{accumulation_growth, null_quadratic_band, orthonormality_defect_bound};
use gam_math::roundoff::inflated;
use ndarray::{Array1, Array2, ArrayView2, Axis};

use super::super::dense::{Triangle, eigh};
use super::super::supports::{EvidenceStatus, ExactBasis};
use super::linear::frobenius;
use super::{CompileError, require_finite, require_shape};

/// The ball a null-edit supremum is stated over: every `u` of the resolved range of `G`
/// with `uᵀ G u ≤ 1`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct EditBall {
    pub dimension: usize,
    pub range_rank: usize,
}

/// A witness edit direction and its two quadratic forms.
#[derive(Clone, Debug, PartialEq)]
pub struct EditDirection {
    pub direction: Vec<f64>,
    /// `uᵀ G u` and `uᵀ K u` as evaluated.
    pub edit_size: f64,
    pub response_error: f64,
}

/// The supremum, or the refusal.
#[derive(Clone, Debug, PartialEq)]
pub enum NullEditOutcome {
    /// `sup uᵀKu` over the ball, exact up to its numerical error, attained at the witness.
    Bounded(EvidenceStatus<EditDirection, EditBall>),
    /// `ker G ⊄ ker K`: a counterexample `u ∈ ker G` with `uᵀ K u` resolved above zero.
    Refused(EvidenceStatus<EditDirection, EditBall>),
}

/// The symmetric part `(M + Mᵀ)/2` of a declared Gram; a Gram formed as `JᵀJ` by a
/// blocked product is symmetric only up to rounding, and its symmetric part is the Gram.
fn symmetric(what: &'static str, matrix: ArrayView2<'_, f64>) -> Result<Array2<f64>, CompileError> {
    require_shape(what, (matrix.nrows(), matrix.nrows()), matrix.dim())?;
    require_finite(what, matrix.iter().copied())?;
    Ok((&matrix + &matrix.t()) * 0.5)
}

fn quadratic(matrix: ArrayView2<'_, f64>, vector: &Array1<f64>) -> f64 {
    vector.dot(&matrix.dot(vector))
}

/// `sup { uᵀKu : u ∈ range(G), uᵀGu ≤ 1 }`, refused when `ker G ⊄ ker K`.
pub fn physically_null_supremum(
    edit_gram: ArrayView2<'_, f64>,
    response_gram: ArrayView2<'_, f64>,
) -> Result<NullEditOutcome, CompileError> {
    let edit_gram = symmetric("edit Gram", edit_gram)?;
    let response_gram = symmetric("response-error Gram", response_gram)?;
    let (edit_gram, response_gram) = (edit_gram.view(), response_gram.view());
    let n = edit_gram.nrows();
    require_shape("response-error Gram", (n, n), response_gram.dim())?;
    if n == 0 {
        return Err(CompileError::InvalidDeclaration {
            what: "edit Gram",
            reason: "an edit space needs at least one coordinate".to_string(),
        });
    }
    let decomposed = eigh(edit_gram, Triangle::Lower, None)?;
    let beta = decomposed.band;
    let resolved: Vec<usize> = (0..n).filter(|&i| decomposed.values[i] > beta).collect();
    let kernel: Vec<usize> = (0..n).filter(|&i| decomposed.values[i] <= beta).collect();
    let ball = EditBall {
        dimension: n,
        range_rank: resolved.len(),
    };
    if !kernel.is_empty() {
        let null_basis = decomposed.vectors.select(Axis(1), &kernel);
        let reduced = null_basis.t().dot(&response_gram.dot(&null_basis));
        let symmetrized = (&reduced + &reduced.t()) * 0.5;
        let top = eigh(symmetrized.view(), Triangle::Lower, None)?;
        let last = kernel.len() - 1;
        let direction = null_basis.dot(&top.vectors.column(last));
        let length = direction.dot(&direction).sqrt();
        let direction = direction.mapv(|value| value / length);
        let response_error = quadratic(response_gram, &direction);
        let band = null_quadratic_band(response_gram, direction.view());
        if response_error - band > 0.0 {
            return Ok(NullEditOutcome::Refused(EvidenceStatus::counterexample(
                response_error,
                band,
                0.0,
                EditDirection {
                    edit_size: quadratic(edit_gram, &direction),
                    response_error,
                    direction: direction.to_vec(),
                },
            )?));
        }
    }
    if resolved.is_empty() {
        return Ok(NullEditOutcome::Bounded(EvidenceStatus::exact(
            0.0,
            0.0,
            ExactBasis::Algebraic,
            None,
            ball,
        )?));
    }
    let basis = decomposed.vectors.select(Axis(1), &resolved);
    let values: Vec<f64> = resolved.iter().map(|&i| decomposed.values[i]).collect();
    let root_inverse: Array1<f64> = values.iter().map(|value| 1.0 / value.sqrt()).collect();
    let mut scaled = basis.clone();
    for (mut column, &factor) in scaled.columns_mut().into_iter().zip(root_inverse.iter()) {
        column.mapv_inplace(|value| value * factor);
    }
    let reduced = scaled.t().dot(&response_gram.dot(&scaled));
    let reduced = (&reduced + &reduced.t()) * 0.5;
    let rank = resolved.len();
    let top = eigh(reduced.view(), Triangle::Lower, None)?;
    let supremum = top.values[rank - 1];
    let witness = scaled.dot(&top.vectors.column(rank - 1));
    let absolute = scaled.mapv(f64::abs);
    let formation = frobenius(
        (absolute.t().dot(&response_gram.mapv(f64::abs).dot(&absolute)) * accumulation_growth(2 * n + 3)).view(),
    );
    let mut gram = basis.t().dot(&basis);
    for index in 0..rank {
        gram[[index, index]] -= 1.0;
    }
    let omega = orthonormality_defect_bound(frobenius(gram.view()), n, rank);
    let smallest = values.iter().copied().fold(f64::INFINITY, f64::min);
    let rho = beta / (smallest - beta).max(f64::MIN_POSITIVE);
    let ball_error = if rho < 1.0 {
        supremum.abs() * rho / (1.0 - rho)
    } else {
        f64::INFINITY
    };
    let numerical_error = inflated(
        ball_error + formation + top.band + supremum.abs() * (2.0 * omega + omega * omega),
        4,
    );
    if !numerical_error.is_finite() {
        return Ok(NullEditOutcome::Bounded(EvidenceStatus::unresolved(
            (supremum - top.band - formation).max(0.0),
            f64::INFINITY,
            super::super::supports::Extremum::Supremum,
            None,
            ball,
        )?));
    }
    let witness_direction = EditDirection {
        edit_size: quadratic(edit_gram, &witness),
        response_error: quadratic(response_gram, &witness),
        direction: witness.to_vec(),
    };
    Ok(NullEditOutcome::Bounded(EvidenceStatus::exact(
        supremum,
        numerical_error,
        ExactBasis::Algebraic,
        Some(witness_direction),
        ball,
    )?))
}

