//! The linear pass-through gauge of native tensors (#2951).
//!
//! A read `A ∈ ℝ^{r×d_in}` followed by a write `B ∈ ℝ^{d_out×r}` with no primitive between
//! them executes `B A`. Examples are an attention head's value and output projections, or a
//! low-rank factor `P = U Vᵀ`. `(A, B) ↦ (S A, B S⁻¹)` leaves `B A` unchanged for every
//! `S ∈ GL(r)`, so two settings execute the same map iff `B′A′ = B A`, and at full rank that
//! holds iff they lie on one `GL(r)` orbit. [`LinearPassthrough::operator_difference`]
//! reports `sup_x ‖(B′A′ − B A) x‖/‖x‖` with its numerical error and a witness input; the
//! edit compiler's chart edits are certified through it.
//!
//! Every operator acts through its factors; nothing here forms a `d_out × d_in` product.

use gam_linalg::faer_ndarray::{FaerLinalgError, FaerQr, FaerSvd, fast_ab, fast_abt, fast_av};
use gam_linalg::roundoff::{accumulation_growth, factor_singular_band};
use gam_linalg::utils::frobenius_norm;
use ndarray::{Array1, Array2, ArrayView1, ArrayView2, Axis, concatenate, s};

use super::supports::{EvidenceStatus, EvidenceStatusError, ExactBasis};

/// Why a pass-through construction, gauge change or comparison was declined.
#[derive(Clone, Debug, PartialEq)]
pub enum GaugeRefusal {
    /// A dimension disagrees with the object it is combined with.
    DimensionMismatch {
        what: &'static str,
        expected: usize,
        found: usize,
    },
    /// An input carries a non-finite entry.
    NonFinite { what: &'static str },
    /// A tensor has no rows or no columns.
    EmptyTensor { what: &'static str },
    /// A gauge change has fewer singular values above its SVD's rounding band than
    /// its order.
    SingularGaugeChange { resolved: usize, order: usize },
    /// The linear-algebra backend failed to decompose a matrix.
    Decomposition { what: &'static str, detail: String },
    /// A reported number was refused by its evidence constructor.
    Evidence(EvidenceStatusError),
}

impl From<EvidenceStatusError> for GaugeRefusal {
    fn from(error: EvidenceStatusError) -> Self {
        Self::Evidence(error)
    }
}

/// The region of every nonzero input of a `width`-dimensional map. Over it an
/// operator difference `Δ` is reported as `sup_x ‖Δ x‖/‖x‖`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AllInputs {
    pub width: usize,
}

/// A read `A` (`r × d_in`) followed by a write `B` (`d_out × r`) with no primitive
/// between them, executing `x ↦ B A x`.
#[derive(Clone, Debug)]
pub struct LinearPassthrough {
    read: Array2<f64>,
    write: Array2<f64>,
}

impl LinearPassthrough {
    pub fn new(read: Array2<f64>, write: Array2<f64>) -> Result<Self, GaugeRefusal> {
        require_nonempty("pass-through read", read.view())?;
        require_nonempty("pass-through write", write.view())?;
        check_len("pass-through write columns", read.nrows(), write.ncols())?;
        require_finite_matrix("pass-through read", read.view())?;
        require_finite_matrix("pass-through write", write.view())?;
        Ok(Self { read, write })
    }

    /// The internal dimension `r`.
    pub fn order(&self) -> usize {
        self.read.nrows()
    }

    pub fn read(&self) -> ArrayView2<'_, f64> {
        self.read.view()
    }

    pub fn write(&self) -> ArrayView2<'_, f64> {
        self.write.view()
    }

    /// `B (A x)`, through the `r` internal coordinates.
    pub fn apply_to(&self, x: ArrayView1<'_, f64>) -> Result<Array1<f64>, GaugeRefusal> {
        check_len("pass-through input", self.read.ncols(), x.len())?;
        Ok(fast_av(&self.write, &fast_av(&self.read, &x)))
    }

    /// `(A, B) ↦ (S A, B S⁻¹)`. It refuses an `S` whose SVD resolves fewer than `r`
    /// singular values.
    pub fn apply(&self, gauge_change: ArrayView2<'_, f64>) -> Result<Self, GaugeRefusal> {
        let r = self.order();
        check_len("gauge change rows", r, gauge_change.nrows())?;
        check_len("gauge change columns", r, gauge_change.ncols())?;
        require_finite_matrix("gauge change", gauge_change)?;
        let inverse = invert_gauge_change(gauge_change)?;
        Ok(Self {
            read: fast_ab(&gauge_change, &self.read),
            write: fast_ab(&self.write, &inverse),
        })
    }

    /// `sup_x ‖(B′A′ − B A) x‖/‖x‖ = ‖B′A′ − B A‖₂` for `other = (A′, B′)`, as an
    /// exact algebraic extremum with its numerical error and a witness input.
    ///
    /// The difference is `L Rᵀ` with `L = [B′ | −B]` and `R = [A′ᵀ | Aᵀ]`, each with
    /// `k = r′ + r` columns. A factor with at least `k` rows is replaced by its thin
    /// QR `Q T`. Householder QR is backward stable: `T` is the exact triangle of
    /// `L + δL` for an exactly orthonormal `Q`, with `‖δL‖₂` inside
    /// `factor_singular_band(rows, k, ‖L‖_F)`. So `‖T_L T_Rᵀ‖₂` is the norm of the
    /// perturbed difference. A factor with fewer rows than `k` is used as it is,
    /// with no error.
    ///
    /// The numerical error, to first order, is the sum of:
    /// - `‖δL‖ ‖R‖_F + ‖L‖_F ‖δR‖`;
    /// - the rounding of the `k`-term core product, `γ_k ‖T_L‖_F ‖T_R‖_F`;
    /// - the core SVD's own band.
    ///
    /// The witness is the input along the top right singular vector.
    pub fn operator_difference(&self, other: &Self) -> Result<EvidenceStatus<Array1<f64>, AllInputs>, GaugeRefusal> {
        check_len("compared input width", self.read.ncols(), other.read.ncols())?;
        check_len("compared output width", self.write.nrows(), other.write.nrows())?;
        let negated = self.write.mapv(|entry| -entry);
        let left = concatenate(Axis(1), &[other.write.view(), negated.view()]).map_err(shape_failed)?;
        let right = concatenate(Axis(1), &[other.read.t(), self.read.t()]).map_err(shape_failed)?;
        let left_range = RangeFactor::new("difference write range", left.view())?;
        let right_range = RangeFactor::new("difference read range", right.view())?;
        let core = fast_abt(&left_range.core, &right_range.core);
        let decomposition = core
            .svd(false, true)
            .map_err(|failure| decomposition_failed("difference core", &failure))?;
        let sigma = decomposition.1;
        let Some(right_singular) = decomposition.2 else {
            return Err(GaugeRefusal::Decomposition {
                what: "difference core",
                detail: "right singular vectors were requested but not returned".to_string(),
            });
        };
        let top = (0..sigma.len()).fold(0, |best, index| if sigma[index] > sigma[best] { index } else { best });
        let direction = right_singular.row(top).to_owned();
        let witness = match &right_range.basis {
            Some(basis) => fast_av(basis, &direction),
            None => direction,
        };
        let left_norm = frobenius_norm(left.view());
        let right_norm = frobenius_norm(right.view());
        let numerical_error = left_range.backward_error * right_norm
            + left_norm * right_range.backward_error
            + accumulation_growth(left.ncols()) * frobenius_norm(left_range.core.view()) * frobenius_norm(right_range.core.view())
            + factor_singular_band(core.nrows(), core.ncols(), frobenius_norm(core.view()));
        Ok(EvidenceStatus::exact(
            sigma[top],
            numerical_error,
            ExactBasis::Algebraic,
            Some(witness),
            AllInputs {
                width: self.read.ncols(),
            },
        )?)
    }
}

/// A factor `M` (`rows × k`) reduced to a core with the same singular values: its
/// thin QR triangle when `rows ≥ k`, otherwise `M` itself.
struct RangeFactor {
    core: Array2<f64>,
    basis: Option<Array2<f64>>,
    /// The QR backward-error band on `‖δM‖₂`, zero when no QR was taken.
    backward_error: f64,
}

impl RangeFactor {
    fn new(what: &'static str, factor: ArrayView2<'_, f64>) -> Result<Self, GaugeRefusal> {
        let (rows, k) = factor.dim();
        if rows < k {
            return Ok(Self {
                core: factor.to_owned(),
                basis: None,
                backward_error: 0.0,
            });
        }
        let (q, t) = factor.qr().map_err(|failure| decomposition_failed(what, &failure))?;
        Ok(Self {
            core: t.slice(s![..k, ..k]).to_owned(),
            basis: Some(q.slice(s![.., ..k]).to_owned()),
            backward_error: factor_singular_band(rows, k, frobenius_norm(factor)),
        })
    }
}

fn check_len(what: &'static str, expected: usize, found: usize) -> Result<(), GaugeRefusal> {
    if expected == found {
        Ok(())
    } else {
        Err(GaugeRefusal::DimensionMismatch { what, expected, found })
    }
}

fn require_nonempty(what: &'static str, matrix: ArrayView2<'_, f64>) -> Result<(), GaugeRefusal> {
    if matrix.nrows() == 0 || matrix.ncols() == 0 {
        Err(GaugeRefusal::EmptyTensor { what })
    } else {
        Ok(())
    }
}

fn require_finite_matrix(what: &'static str, matrix: ArrayView2<'_, f64>) -> Result<(), GaugeRefusal> {
    if matrix.iter().all(|entry| entry.is_finite()) {
        Ok(())
    } else {
        Err(GaugeRefusal::NonFinite { what })
    }
}

fn decomposition_failed(what: &'static str, failure: &FaerLinalgError) -> GaugeRefusal {
    GaugeRefusal::Decomposition {
        what,
        detail: format!("{failure:?}"),
    }
}

fn shape_failed(failure: ndarray::ShapeError) -> GaugeRefusal {
    GaugeRefusal::Decomposition {
        what: "factor concatenation",
        detail: failure.to_string(),
    }
}

/// `S⁻¹ = V Σ⁻¹ Uᵀ` from the SVD `S = U Σ Vᵀ`. It refuses a singular value inside
/// the SVD's rounding band, where the inverse carries no correct digit.
fn invert_gauge_change(gauge_change: ArrayView2<'_, f64>) -> Result<Array2<f64>, GaugeRefusal> {
    let what = "gauge change";
    let order = gauge_change.nrows();
    let (left, sigma, right_t) = gauge_change
        .svd(true, true)
        .map_err(|failure| decomposition_failed(what, &failure))?;
    let sigma_max = sigma.iter().fold(0.0_f64, |largest, &value| largest.max(value));
    let band = factor_singular_band(order, order, sigma_max);
    let resolved = sigma.iter().filter(|&&value| value > band).count();
    if resolved < order {
        return Err(GaugeRefusal::SingularGaugeChange { resolved, order });
    }
    let (Some(left), Some(right_t)) = (left, right_t) else {
        return Err(GaugeRefusal::Decomposition {
            what,
            detail: "singular vectors were requested but not returned".to_string(),
        });
    };
    let mut right_scaled = right_t.t().to_owned();
    for (j, &value) in sigma.iter().enumerate() {
        right_scaled.column_mut(j).mapv_inplace(|entry| entry / value);
    }
    Ok(fast_abt(&right_scaled, &left))
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::rngs::StdRng;
    use rand::{RngExt, SeedableRng};

    fn uniform_matrix(rng: &mut StdRng, rows: usize, cols: usize) -> Array2<f64> {
        let mut out = Array2::<f64>::zeros((rows, cols));
        for entry in out.iter_mut() {
            *entry = rng.random_range(-1.0..1.0);
        }
        out
    }

    fn exact_parts(status: &EvidenceStatus<Array1<f64>, AllInputs>) -> (f64, f64, Array1<f64>) {
        match status {
            EvidenceStatus::Exact {
                value,
                numerical_error,
                witness: Some(witness),
                ..
            } => (*value, *numerical_error, witness.clone()),
            other => panic!("an operator difference is an exact extremum with a witness, got {other:?}"),
        }
    }

    /// First-order bound on `‖B Ŝ⁻¹ fl(S A) − B A‖₂` for the regauged setting.
    ///
    /// The computed inverse is the exact inverse of `S + E` with `‖E‖₂ ≤ r·ε·‖S‖₂`
    /// (the SVD backward error, [`factor_singular_band`]'s convention). It is formed
    /// from products of at most `2r` rounded terms, whose absolute magnitudes
    /// `κ_F = ‖S‖_F ‖Ŝ⁻¹‖_F` majorizes. So `‖Ŝ⁻¹ S − I‖ ≤ η = κ_F·r·(ε + γ_{2r})`.
    /// `fl(S A)` and `fl(B Ŝ⁻¹)` each round `r` terms per entry, on magnitudes
    /// majorized by `‖S‖_F ‖A‖_F` and `‖B‖_F ‖Ŝ⁻¹‖_F`, and each meets the other
    /// factor once. Together they give `‖A‖_F ‖B‖_F (η + 2 γ_r κ_F)`.
    fn regauge_band(teacher: &LinearPassthrough, gauge_change: &Array2<f64>) -> f64 {
        let r = teacher.order();
        let inverse = invert_gauge_change(gauge_change.view()).expect("the fixture's gauge change is invertible");
        let kappa = frobenius_norm(gauge_change.view()) * frobenius_norm(inverse.view());
        let eta = kappa * r as f64 * (f64::EPSILON + accumulation_growth(2 * r));
        frobenius_norm(teacher.read()) * frobenius_norm(teacher.write()) * (eta + 2.0 * accumulation_growth(r) * kappa)
    }

    /// A2's `GL(r)` half through the executed map: the regauged setting executes the
    /// teacher's map within its construction band. The one-sided change `(S A, B)`
    /// is the positive control, and it must be proven distinct: its lower bound
    /// exceeds the rounding of `fl(S A)`, `γ_r ‖S‖_F ‖A‖_F ‖B‖_F`, so the exact
    /// `B S A ≠ B A`. Its witness input must move the executed output by more than
    /// both executions' rounding.
    #[test]
    fn a_general_linear_change_keeps_the_executed_map_and_a_one_sided_change_moves_it() {
        let mut rng = StdRng::seed_from_u64(295_401);
        let (d_in, d_out, r) = (7, 6, 4);
        let teacher = LinearPassthrough::new(uniform_matrix(&mut rng, r, d_in), uniform_matrix(&mut rng, d_out, r))
            .expect("finite factors");
        let gauge_change = uniform_matrix(&mut rng, r, r);
        let regauged = teacher.apply(gauge_change.view()).expect("a random change is invertible");
        let band = regauge_band(&teacher, &gauge_change);
        let (value, numerical_error, witness) =
            exact_parts(&teacher.operator_difference(&regauged).expect("comparable settings"));
        assert!(
            value <= band + numerical_error,
            "regauged map moved by {value:.3e}, construction band {band:.3e} + evaluation error {numerical_error:.3e}"
        );
        assert_eq!(witness.len(), d_in);

        let one_sided = LinearPassthrough::new(fast_ab(&gauge_change, &teacher.read), teacher.write.clone())
            .expect("finite factors");
        let one_sided_band = accumulation_growth(r)
            * frobenius_norm(gauge_change.view())
            * frobenius_norm(teacher.read())
            * frobenius_norm(teacher.write());
        let difference = teacher.operator_difference(&one_sided).expect("comparable settings");
        let lower = difference.lower_bound().expect("an exact extremum bounds itself");
        assert!(
            lower > one_sided_band,
            "a one-sided change must be proven distinct: lower bound {lower:.3e}, rounding {one_sided_band:.3e}"
        );
        let (value, numerical_error, witness) = exact_parts(&difference);
        let executed = frobenius_norm(
            &(one_sided.apply_to(witness.view()).expect("apply") - &teacher.apply_to(witness.view()).expect("apply")),
        );
        let widest = d_in.max(d_out).max(r);
        let execution_band = accumulation_growth(2 * widest)
            * frobenius_norm(teacher.write())
            * (frobenius_norm(teacher.read()) + frobenius_norm(one_sided.read()))
            * frobenius_norm(&witness)
            + one_sided_band * frobenius_norm(&witness);
        assert!(
            executed > execution_band,
            "the witness moved the executed output by {executed:.3e}, band {execution_band:.3e} (sup {value:.3e} ± {numerical_error:.3e})"
        );
    }
}
