//! Matrix-valued parameter fields and their anchored pullbacks (#2951 P16).
//!
//! A parameter family is a matrix-valued GAM field over one of the crate's
//! existing latent bases ([`SaeBasisEvaluator`]),
//!
//! ```text
//! Gamma(z) = sum_j phi_j(z) B_j,
//! ```
//!
//! read at fixed instances `P_c = w_c Gamma(z_c)`. The labels `z_c` and scales
//! `w_c` belong to the family, not to the input: input dependence only selects
//! or masks instances. With `v_c = w_c phi(z_c)` an instance is `P_c = B v_c`,
//! which is how the residual anchor
//!
//! ```text
//! Theta(m) = m_Delta Theta_* + B sum_c (m_c - m_Delta) v_c
//! ```
//!
//! reads a family: `Theta(m) = m_Delta Theta_* + sum_j s_j B_j` with
//! `s_j = sum_c (m_c - m_Delta) v_cj` ([`ParameterFamily::anchor_basis_weights`]).
//!
//! # Scale gauge
//!
//! `(w, B) -> (s w, B / s)` leaves every instance unchanged, so a family's scale
//! is not identified. [`ParameterFamily::new`] moves `sum_c w_c` into the
//! coefficients and leaves `sum_c w_c = 1`. That slice is transversal to the
//! orbit, which moves `sum_c w_c` at rate `sum_c w_c = 1`.
//!
//! # Product form
//!
//! A coefficient is dense or the product `B_j = U_j V_j^T`. The product form is
//! how a field lives at network width; nothing here forms `U_j V_j^T`, and a
//! cotangent can stay a sum of per-row outer products.
//!
//! # Anchored pullbacks
//!
//! With `G = d loss / d Theta_m`:
//!
//! ```text
//! d loss / d B_j   = s_j G
//! d loss / d z^a_c = (m_c - m_Delta) w_c sum_j d_a phi_j(z_c) <G, B_j>_F
//! d loss / d w_c   = (m_c - m_Delta) sum_j phi_j(z_c) <G, B_j>_F
//! d loss / d U_j   = s_j G V_j,        d loss / d V_j = s_j G^T U_j
//! ```
//!
//! Every derivative carries a factor `m_c - m_Delta`, so at the all-on setting
//! each one is exactly zero: an all-on loss cannot train anything. Resolving a
//! tensor's uses and its tied sum belongs to `occurrence`, which emits a
//! [`ParameterCotangent`] of per-use terms; a pullback reads only the action of
//! `G` on the coefficients.
//!
//! The adversarial fidelity `R = sup_m loss(Theta(m))` runs over a mask domain
//! that does not depend on the family, while the moment zonotope
//! `sum_c [0, v_c]` does. Where the maximizing mask is unique, Danskin's theorem
//! makes `dR` the pullback at that fixed MASK witness. Holding the moment
//! `q = sum_c (m_c - m_Delta) v_c` fixed instead drops the label and weight
//! terms, because labels and weights enter `Theta` only through `q`.
//!
//! # Function-space penalty
//!
//! [`FieldPenalty`] charges `J(Gamma) = sum_ik Q_ik <Gamma(c_i), Gamma(c_k)>_F`,
//! where `Q` is the measure-jet energy of a declared latent measure. That is the
//! same quadratic as `sum_jl S_jl <B_j, B_l>_F` with `S = Phi_c^T Q Phi_c`, so the
//! penalty is on the function values, not on the coefficients.

use std::sync::Arc;

use gam_linalg::faer_ndarray::rrqr_nullspace_basis_with_cutoff;
use gam_linalg::matrix::symmetrize_in_place;
use gam_linalg::roundoff::accumulation_growth;
use gam_terms::basis::{
    MeasureJetBasisSpec, affine_function_nullspace_form, measure_jet_band, measure_jet_energy_form,
};
use ndarray::{Array1, Array2, Array3, ArrayView1, ArrayView2};

use crate::basis::SaeBasisEvaluator;

use super::apply::{FactorView, edit_factor_cotangents, edit_frobenius_contractions};

/// One basis coefficient `B_j` of a `rows x cols` field.
#[derive(Debug, Clone)]
pub enum FieldCoefficient {
    /// A materialized `rows x cols` coefficient.
    Dense(Array2<f64>),
    /// `B = left right^T`, with `left` `rows x r` and `right` `cols x r`.
    Factored { left: Array2<f64>, right: Array2<f64> },
}

impl FieldCoefficient {
    /// `(rows, cols)` of `B`, refusing a malformed product form or a non-finite
    /// entry.
    fn checked_shape(&self) -> Result<(usize, usize), String> {
        match self {
            FieldCoefficient::Dense(b) => {
                if !b.iter().all(|v| v.is_finite()) {
                    return Err("field coefficient has a non-finite entry".to_string());
                }
                Ok(b.dim())
            }
            FieldCoefficient::Factored { left, right } => {
                if left.ncols() != right.ncols() || left.ncols() == 0 {
                    return Err(format!(
                        "product-form coefficient needs one positive rank on both factors; got left {:?}, right {:?}",
                        left.dim(),
                        right.dim()
                    ));
                }
                if !left.iter().chain(right.iter()).all(|v| v.is_finite()) {
                    return Err("product-form coefficient has a non-finite entry".to_string());
                }
                Ok((left.nrows(), right.nrows()))
            }
        }
    }

    fn into_scaled(self, scale: f64) -> Self {
        match self {
            FieldCoefficient::Dense(b) => FieldCoefficient::Dense(b.mapv_into(|v| v * scale)),
            FieldCoefficient::Factored { left, right } => FieldCoefficient::Factored {
                left: left.mapv_into(|v| v * scale),
                right,
            },
        }
    }
}

/// One use's contribution to the cotangent `G = d loss / d Theta_m` of a
/// `rows x cols` tensor.
#[derive(Debug, Clone)]
pub enum CotangentTerm {
    /// A materialized `rows x cols` cotangent.
    Dense(Array2<f64>),
    /// `G = sum_i output_i input_i^T` for a linear use `y_i = Theta x_i`:
    /// `output` is `n x rows` (the cotangents of the `y_i`) and `input` is
    /// `n x cols` (the `x_i`).
    Outer {
        output: Array2<f64>,
        input: Array2<f64>,
    },
}

/// The cotangent of one tensor as the per-use terms `occurrence` resolved: `G` is
/// their sum, and the pullbacks read only its action.
#[derive(Debug, Clone)]
pub struct ParameterCotangent {
    rows: usize,
    cols: usize,
    terms: Vec<CotangentTerm>,
}

impl ParameterCotangent {
    /// The cotangent `G = sum_u G_u` of a `rows x cols` tensor from its per-use
    /// terms.
    pub fn from_terms(rows: usize, cols: usize, terms: Vec<CotangentTerm>) -> Result<Self, String> {
        for (index, term) in terms.iter().enumerate() {
            let (shaped, finite) = match term {
                CotangentTerm::Dense(g) => (g.dim() == (rows, cols), g.iter().all(|v| v.is_finite())),
                CotangentTerm::Outer { output, input } => (
                    output.ncols() == rows && input.ncols() == cols && output.nrows() == input.nrows(),
                    output.iter().chain(input.iter()).all(|v| v.is_finite()),
                ),
            };
            if !shaped {
                return Err(format!(
                    "cotangent term {index} does not describe a {rows} x {cols} tensor"
                ));
            }
            if !finite {
                return Err(format!("cotangent term {index} has a non-finite entry"));
            }
        }
        Ok(Self { rows, cols, terms })
    }

    /// `(rows, cols)` of the tensor this cotangent belongs to.
    pub fn shape(&self) -> (usize, usize) {
        (self.rows, self.cols)
    }

    /// `<G, B>_F`, never forming a product-form `B` or an outer-product `G`. An
    /// outer-product term against a product-form coefficient streams through
    /// `apply`'s governed contraction, which refuses rather than allocate past
    /// its memory admission.
    pub fn frobenius_inner(&self, coefficient: &FieldCoefficient) -> Result<f64, String> {
        let mut total = 0.0;
        for term in &self.terms {
            total += match (term, coefficient) {
                (CotangentTerm::Dense(g), FieldCoefficient::Dense(b)) => frobenius(g.view(), b.view()),
                (CotangentTerm::Dense(g), FieldCoefficient::Factored { left, right }) => {
                    frobenius(g.dot(right).view(), left.view())
                }
                (CotangentTerm::Outer { output, input }, FieldCoefficient::Dense(b)) => {
                    frobenius(output.view(), input.dot(&b.t()).view())
                }
                (CotangentTerm::Outer { output, input }, FieldCoefficient::Factored { left, right }) => {
                    let view = FactorView::new(left.view(), right.view()).map_err(|e| e.to_string())?;
                    edit_frobenius_contractions(view, output.view(), input.view())
                        .map_err(|e| e.to_string())?
                        .sum()
                }
            };
        }
        Ok(total)
    }

    /// `(G V, G^T U)` for a product-form coefficient `U V^T`, with outer-product
    /// terms streamed through `apply`.
    fn factor_cotangents(
        &self,
        left: ArrayView2<'_, f64>,
        right: ArrayView2<'_, f64>,
    ) -> Result<(Array2<f64>, Array2<f64>), String> {
        let mut g_v = Array2::<f64>::zeros((self.rows, right.ncols()));
        let mut g_t_u = Array2::<f64>::zeros((self.cols, left.ncols()));
        for term in &self.terms {
            match term {
                CotangentTerm::Dense(g) => {
                    g_v += &g.dot(&right);
                    g_t_u += &g.t().dot(&left);
                }
                CotangentTerm::Outer { output, input } => {
                    let view = FactorView::new(left, right).map_err(|e| e.to_string())?;
                    let pieces = edit_factor_cotangents(view, output.view(), input.view())
                        .map_err(|e| e.to_string())?;
                    g_v += &pieces.left;
                    g_t_u += &pieces.right;
                }
            }
        }
        Ok((g_v, g_t_u))
    }
}

fn frobenius(a: ArrayView2<'_, f64>, b: ArrayView2<'_, f64>) -> f64 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

/// `Gamma(z) = sum_j phi_j(z) B_j` over an existing latent basis.
#[derive(Debug, Clone)]
pub struct MatrixParameterField {
    basis: Arc<dyn SaeBasisEvaluator>,
    rows: usize,
    cols: usize,
    coefficients: Vec<FieldCoefficient>,
}

impl MatrixParameterField {
    /// A field with one coefficient per basis function. The basis width is
    /// checked where the basis is first evaluated ([`ParameterFamily::new`]).
    pub fn new(basis: Arc<dyn SaeBasisEvaluator>, coefficients: Vec<FieldCoefficient>) -> Result<Self, String> {
        let first = coefficients
            .first()
            .ok_or_else(|| "a parameter field needs at least one basis coefficient".to_string())?
            .checked_shape()?;
        for (j, coefficient) in coefficients.iter().enumerate() {
            let shape = coefficient.checked_shape()?;
            if shape != first {
                return Err(format!(
                    "field coefficient {j} has shape {shape:?}; coefficient 0 has {first:?}"
                ));
            }
        }
        Ok(Self {
            basis,
            rows: first.0,
            cols: first.1,
            coefficients,
        })
    }

    /// `(rows, cols)` of every `Gamma(z)`.
    pub fn shape(&self) -> (usize, usize) {
        (self.rows, self.cols)
    }

    pub fn coefficients(&self) -> &[FieldCoefficient] {
        &self.coefficients
    }

    fn check_penalty(&self, penalty: &FieldPenalty) -> Result<(), String> {
        let width = self.coefficients.len();
        if penalty.gram.dim() != (width, width) {
            return Err(format!(
                "penalty Gram {:?} does not match the field's {width} coefficients",
                penalty.gram.dim()
            ));
        }
        Ok(())
    }

    /// `J(B) = sum_jl S_jl <B_j, B_l>_F`.
    pub fn penalty(&self, penalty: &FieldPenalty) -> Result<f64, String> {
        self.check_penalty(penalty)?;
        let mut total = 0.0;
        for (j, a) in self.coefficients.iter().enumerate() {
            for (l, b) in self.coefficients.iter().enumerate() {
                total += penalty.gram[[j, l]] * coefficient_inner(a, b);
            }
        }
        Ok(total)
    }

    /// `dJ / dB_j = 2 sum_l S_jl B_l`. For a product-form coefficient this returns
    /// `(dJ / dB_j) V_j` and `(dJ / dB_j)^T U_j` without forming `dJ / dB_j`.
    pub fn penalty_gradient(&self, penalty: &FieldPenalty) -> Result<Vec<PenaltyGradient>, String> {
        self.check_penalty(penalty)?;
        Ok(self
            .coefficients
            .iter()
            .enumerate()
            .map(|(j, coefficient)| match coefficient {
                FieldCoefficient::Dense(..) => {
                    let mut gradient = Array2::<f64>::zeros((self.rows, self.cols));
                    for (l, other) in self.coefficients.iter().enumerate() {
                        let s = 2.0 * penalty.gram[[j, l]];
                        match other {
                            FieldCoefficient::Dense(x) => gradient.scaled_add(s, x),
                            FieldCoefficient::Factored { left, right } => {
                                gradient.scaled_add(s, &left.dot(&right.t()))
                            }
                        }
                    }
                    PenaltyGradient::Dense(gradient)
                }
                FieldCoefficient::Factored { left: u, right: v } => {
                    let mut d_left = Array2::<f64>::zeros(u.dim());
                    let mut d_right = Array2::<f64>::zeros(v.dim());
                    for (l, other) in self.coefficients.iter().enumerate() {
                        let s = 2.0 * penalty.gram[[j, l]];
                        match other {
                            FieldCoefficient::Dense(x) => {
                                d_left.scaled_add(s, &x.dot(v));
                                d_right.scaled_add(s, &x.t().dot(u));
                            }
                            FieldCoefficient::Factored { left, right } => {
                                d_left.scaled_add(s, &left.dot(&right.t().dot(v)));
                                d_right.scaled_add(s, &right.dot(&left.t().dot(u)));
                            }
                        }
                    }
                    PenaltyGradient::Factored {
                        left: d_left,
                        right: d_right,
                    }
                }
            })
            .collect())
    }
}

/// A field with its fixed instances `P_c = w_c Gamma(z_c)`, in the scale gauge
/// `sum_c w_c = 1`.
#[derive(Debug, Clone)]
pub struct ParameterFamily {
    field: MatrixParameterField,
    labels: Array2<f64>,
    weights: Array1<f64>,
    basis_values: Array2<f64>,
    basis_jets: Array3<f64>,
}

impl ParameterFamily {
    /// Instances at the `C x d` family labels with scales `weights`. The scale
    /// `sum_c w_c` moves into the coefficients, so the instances are unchanged
    /// up to rounding and the stored weights sum to one.
    pub fn new(field: MatrixParameterField, labels: Array2<f64>, weights: Array1<f64>) -> Result<Self, String> {
        let instances = labels.nrows();
        if instances == 0 || weights.len() != instances {
            return Err(format!(
                "a parameter family needs one weight per instance and at least one instance; got {instances} labels and {} weights",
                weights.len()
            ));
        }
        if !labels.iter().chain(weights.iter()).all(|v| v.is_finite()) {
            return Err("parameter family labels and weights must be finite".to_string());
        }
        // The weights are pre-formed terms, so their sum commits C - 1 rounded
        // additions (gam_linalg::roundoff, Higham Lemma 3.1). A total inside that
        // band has no determined sign, so no gauge representative exists.
        let total = weights.sum();
        let rounding = accumulation_growth(instances - 1) * weights.iter().map(|w| w.abs()).sum::<f64>();
        if !(total.abs() > rounding) {
            return Err(format!(
                "the scale gauge sum_c w_c = 1 cannot be fixed: the weights sum to {total:e}, inside its rounding {rounding:e}"
            ));
        }
        let MatrixParameterField {
            basis,
            rows,
            cols,
            coefficients,
        } = field;
        let (basis_values, basis_jets) = basis.evaluate(labels.view())?;
        let width = coefficients.len();
        if basis_values.dim() != (instances, width) || basis_jets.dim() != (instances, width, labels.ncols()) {
            return Err(format!(
                "basis at the family labels returned values {:?} and jets {:?}; the field has {width} coefficients over {} latent dimensions",
                basis_values.dim(),
                basis_jets.dim(),
                labels.ncols()
            ));
        }
        let coefficients = coefficients.into_iter().map(|b| b.into_scaled(total)).collect();
        Ok(Self {
            field: MatrixParameterField {
                basis,
                rows,
                cols,
                coefficients,
            },
            labels,
            weights: weights.mapv_into(|w| w / total),
            basis_values,
            basis_jets,
        })
    }

    pub fn field(&self) -> &MatrixParameterField {
        &self.field
    }

    /// The `C x d` family labels `z_c`.
    pub fn labels(&self) -> ArrayView2<'_, f64> {
        self.labels.view()
    }

    /// The gauge-fixed scales `w_c`.
    pub fn weights(&self) -> ArrayView1<'_, f64> {
        self.weights.view()
    }

    /// The `C x J` moments `v_c = w_c phi(z_c)`, so that `P_c = B v_c`.
    pub fn instance_moments(&self) -> Array2<f64> {
        let mut moments = self.basis_values.clone();
        for (mut row, &w) in moments.rows_mut().into_iter().zip(self.weights.iter()) {
            row.mapv_inplace(|v| v * w);
        }
        moments
    }

    /// `s_j = sum_c (m_c - m_Delta) w_c phi_j(z_c)`, so that the family's part of
    /// the anchored tensor is `sum_j s_j B_j`.
    pub fn anchor_basis_weights(
        &self,
        component_mask: ArrayView1<'_, f64>,
        residual_mask: f64,
    ) -> Result<Array1<f64>, String> {
        let gaps = self.mask_gaps(component_mask, residual_mask)?;
        Ok(self.basis_values.t().dot(&(&gaps * &self.weights)))
    }

    /// Pull `G = d loss / d Theta_m` back to the coefficients, labels, weights and
    /// product-form factors at the mask `(m_c, m_Delta)`. At an adversary's
    /// witness mask this is the Danskin derivative of the adversarial fidelity,
    /// but only where that witness is the unique maximizer. With ties the
    /// fidelity has only directional derivatives (a max over the tied witnesses),
    /// so the witness margin, the gap between the two largest objective values,
    /// belongs next to this derivative. At a local maximizer found by ascent it is
    /// the derivative of that lower witness's value, not of the supremum.
    pub fn pullback(
        &self,
        component_mask: ArrayView1<'_, f64>,
        residual_mask: f64,
        cotangent: &ParameterCotangent,
    ) -> Result<FamilyPullback, String> {
        if cotangent.shape() != self.field.shape() {
            return Err(format!(
                "cotangent shape {:?} does not match the field shape {:?}",
                cotangent.shape(),
                self.field.shape()
            ));
        }
        let gaps = self.mask_gaps(component_mask, residual_mask)?;
        let scaled_gaps = &gaps * &self.weights;
        let anchor_weights = self.basis_values.t().dot(&scaled_gaps);
        let inner = self
            .field
            .coefficients
            .iter()
            .map(|b| cotangent.frobenius_inner(b))
            .collect::<Result<Array1<f64>, String>>()?;
        let weights = &gaps * &self.basis_values.dot(&inner);
        let (instances, width, dimension) = self.basis_jets.dim();
        let mut labels = Array2::<f64>::zeros((instances, dimension));
        for c in 0..instances {
            for a in 0..dimension {
                let mut along = 0.0;
                for j in 0..width {
                    along += self.basis_jets[[c, j, a]] * inner[j];
                }
                labels[[c, a]] = scaled_gaps[c] * along;
            }
        }
        let mut coefficients = Vec::with_capacity(self.field.coefficients.len());
        for (coefficient, &s) in self.field.coefficients.iter().zip(anchor_weights.iter()) {
            coefficients.push(match coefficient {
                FieldCoefficient::Dense(..) => CoefficientPullback::Dense { weight: s },
                FieldCoefficient::Factored { left, right } => {
                    let (g_v, g_t_u) = cotangent.factor_cotangents(left.view(), right.view())?;
                    CoefficientPullback::Factored {
                        left: g_v.mapv_into(|v| s * v),
                        right: g_t_u.mapv_into(|v| s * v),
                    }
                }
            });
        }
        Ok(FamilyPullback {
            coefficients,
            labels,
            weights,
        })
    }

    fn mask_gaps(&self, component_mask: ArrayView1<'_, f64>, residual_mask: f64) -> Result<Array1<f64>, String> {
        if component_mask.len() != self.labels.nrows() {
            return Err(format!(
                "mask has {} components; the family has {} instances",
                component_mask.len(),
                self.labels.nrows()
            ));
        }
        if !residual_mask.is_finite() || !component_mask.iter().all(|m| m.is_finite()) {
            return Err("mask entries must be finite".to_string());
        }
        Ok(component_mask.mapv(|m| m - residual_mask))
    }
}

/// The derivative of a coefficient.
#[derive(Debug, Clone)]
pub enum CoefficientPullback {
    /// `d loss / d B_j = weight G`, with `G` the cotangent that was pulled back.
    Dense { weight: f64 },
    /// `d loss / d U_j = left` and `d loss / d V_j = right`.
    Factored { left: Array2<f64>, right: Array2<f64> },
}

/// The derivatives of one family at one mask.
#[derive(Debug, Clone)]
pub struct FamilyPullback {
    pub coefficients: Vec<CoefficientPullback>,
    /// `C x d`: `d loss / d z_c`.
    pub labels: Array2<f64>,
    /// `d loss / d w_c`.
    pub weights: Array1<f64>,
}

impl FamilyPullback {
    /// The weight derivative on the gauge slice `sum_c w_c = 1`: the component
    /// along the slice normal `1` is removed, so a step along it keeps the gauge.
    pub fn gauge_tangent_weights(&self) -> Array1<f64> {
        let mean = self.weights.sum() / self.weights.len() as f64;
        self.weights.mapv(|g| g - mean)
    }
}

/// `<A, B>_F` of two coefficients, never forming a product-form coefficient.
fn coefficient_inner(a: &FieldCoefficient, b: &FieldCoefficient) -> f64 {
    match (a, b) {
        (FieldCoefficient::Dense(x), FieldCoefficient::Dense(y)) => frobenius(x.view(), y.view()),
        (FieldCoefficient::Dense(x), FieldCoefficient::Factored { left, right })
        | (FieldCoefficient::Factored { left, right }, FieldCoefficient::Dense(x)) => {
            frobenius(x.dot(right).view(), left.view())
        }
        (
            FieldCoefficient::Factored { left: u1, right: v1 },
            FieldCoefficient::Factored { left: u2, right: v2 },
        ) => frobenius(u1.t().dot(u2).view(), v1.t().dot(v2).view()),
    }
}

/// The function-space roughness of a field,
///
/// ```text
/// J(Gamma) = sum_ik Q_ik <Gamma(c_i), Gamma(c_k)>_F = sum_jl S_jl <B_j, B_l>_F,
/// S = Phi_c^T Q Phi_c,
/// ```
///
/// with `Q` the measure-jet energy of a declared latent measure (centers `c_i`,
/// masses) and `Phi_c` the field's basis at the centers. The penalty acts on the
/// function values `Gamma(c_i)` entry by entry, and `S` is that same quadratic
/// written in the coefficients, not a stand-in for it. Reading Gamma only through
/// Frobenius products makes it invariant under orthogonal changes of the row and
/// column bases. The energy annihilates affine values, so an affine field pays
/// nothing.
///
/// `(w, B) -> (s w, B / s)` scales `J` by `1 / s^2`. A penalty read before the
/// scale gauge is fixed can therefore be driven to zero without changing any
/// instance, so it is read on a [`ParameterFamily`]'s gauge-fixed field.
#[derive(Debug, Clone)]
pub struct FieldPenalty {
    gram: Array2<f64>,
}

impl FieldPenalty {
    /// The measure-jet penalty of `basis` over the declared latent measure
    /// `(centers, masses)` under the declared `spec`. The smoothness order, the
    /// density exponent, the tau coordinate and the scale count all come from the
    /// owner's spec, so nothing here reads a default. The auto order
    /// (`order_s == 0.0`) is refused because its resolution belongs to the
    /// smooth's own realization; declare the order.
    ///
    /// A coefficient direction `v` with `Phi_c v = 0` is a field that vanishes at
    /// every center. No function-space penalty over this measure can see such a
    /// direction, so it would carry no prior, and a basis that has one is
    /// refused. The numerical rank of `Phi_c` is decided against
    /// `max(m, J) eps ||Phi_c||_F`; the Frobenius norm bounds the largest singular
    /// value from above.
    pub fn measure_jet(
        basis: &dyn SaeBasisEvaluator,
        centers: ArrayView2<'_, f64>,
        masses: ArrayView1<'_, f64>,
        spec: &MeasureJetBasisSpec,
    ) -> Result<Self, String> {
        if spec.order_s == 0.0 {
            return Err(
                "the measure-jet auto order (order_s == 0.0) is resolved only inside the smooth's realization; declare order_s in (0, 2)"
                    .to_string(),
            );
        }
        let phi = basis_at_centers(basis, centers)?;
        let band = measure_jet_band(centers, spec.num_scales).map_err(|e| e.to_string())?;
        let energy = measure_jet_energy_form(centers, masses, &band, spec.order_s, spec.alpha)
            .map_err(|e| e.to_string())?;
        let mut gram = phi.t().dot(&energy).dot(&phi);
        symmetrize_in_place(&mut gram);
        Ok(Self { gram })
    }

    /// The function-space null component of the measure-jet penalty,
    /// `S_null = Phi_c^T H_0 Phi_c`, where `H_0` is the owner's mass-metric
    /// projector onto affine center values (`affine_function_nullspace_form`).
    /// That projector spans exactly the energy's null space: `v^T S_null v` is the
    /// squared mass norm of the affine part of the field's center values. Next to
    /// [`FieldPenalty::measure_jet`], every coefficient direction the measure can
    /// see is charged by one of the two forms, each with its own smoothing
    /// parameter. The same centre-invisible refusal applies.
    pub fn measure_jet_affine_null(
        basis: &dyn SaeBasisEvaluator,
        centers: ArrayView2<'_, f64>,
        masses: ArrayView1<'_, f64>,
    ) -> Result<Self, String> {
        let phi = basis_at_centers(basis, centers)?;
        let null_form = affine_function_nullspace_form(centers, masses).map_err(|e| e.to_string())?;
        let mut gram = phi.t().dot(&null_form).dot(&phi);
        symmetrize_in_place(&mut gram);
        Ok(Self { gram })
    }

    /// `S`, one row and column per basis function.
    pub fn gram(&self) -> ArrayView2<'_, f64> {
        self.gram.view()
    }
}

/// `Phi_c`, the field basis at the declared centers, refusing a basis that has a
/// coefficient direction vanishing at every center.
fn basis_at_centers(basis: &dyn SaeBasisEvaluator, centers: ArrayView2<'_, f64>) -> Result<Array2<f64>, String> {
    let phi = basis.evaluate(centers)?.0;
    if phi.nrows() != centers.nrows() {
        return Err(format!(
            "basis returned {} rows at {} centers",
            phi.nrows(),
            centers.nrows()
        ));
    }
    let width = phi.ncols();
    let scale = phi.iter().map(|v| v * v).sum::<f64>().sqrt();
    let cutoff = centers.nrows().max(width) as f64 * f64::EPSILON * scale;
    let rank = rrqr_nullspace_basis_with_cutoff(&phi, cutoff)
        .map_err(|e| e.to_string())?
        .1;
    if rank < width {
        return Err(format!(
            "basis has {width} functions but rank {rank} at the {} declared centers (cutoff {cutoff:e}): {} coefficient directions vanish at every center, so no function-space penalty over this measure constrains them",
            centers.nrows(),
            width - rank
        ));
    }
    Ok(phi)
}

/// The penalty's derivative for one coefficient, in the coefficient's own form.
#[derive(Debug, Clone)]
pub enum PenaltyGradient {
    /// `dJ / dB_j`.
    Dense(Array2<f64>),
    /// `dJ / dU_j = left` and `dJ / dV_j = right`.
    Factored { left: Array2<f64>, right: Array2<f64> },
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::basis::EuclideanPatchEvaluator;
    use gam_math::probability::{normal_cdf, normal_pdf};
    use faer::Side;
    use gam_linalg::faer_ndarray::FaerEigh;
    use gam_linalg::roundoff::symmetric_spectrum_rounding_band;
    use gam_terms::basis::monomial_exponents;

    const ROWS: usize = 4;
    const COLS: usize = 3;
    const RANK: usize = 2;
    const LATENT: usize = 2;
    const DEGREE: usize = 2;

    /// Deterministic fixture entries in `[-1, 1)`.
    fn entries(seed: usize, count: usize) -> Vec<f64> {
        (0..count)
            .map(|k| {
                let x = ((seed * 7919 + k * 104_729 + 1) as f64).sin() * 43_758.545_3;
                (x - x.floor()) * 2.0 - 1.0
            })
            .collect()
    }

    fn matrix(seed: usize, rows: usize, cols: usize) -> Array2<f64> {
        Array2::from_shape_vec((rows, cols), entries(seed, rows * cols)).expect("fixture shape")
    }

    fn patch() -> EuclideanPatchEvaluator {
        EuclideanPatchEvaluator::new(LATENT, DEGREE).expect("quadratic patch basis")
    }

    fn dense(coefficient: &FieldCoefficient) -> Array2<f64> {
        match coefficient {
            FieldCoefficient::Dense(b) => b.clone(),
            FieldCoefficient::Factored { left, right } => left.dot(&right.t()),
        }
    }

    /// Entrywise magnitude of the terms that form `B`: `|B|` dense, and
    /// `sum_l |U_rl| |V_kl|` in the product form.
    fn dense_magnitude(coefficient: &FieldCoefficient) -> Array2<f64> {
        match coefficient {
            FieldCoefficient::Dense(b) => b.mapv(f64::abs),
            FieldCoefficient::Factored { left, right } => left.mapv(f64::abs).dot(&right.mapv(f64::abs).t()),
        }
    }

    fn family(coefficients: Vec<FieldCoefficient>, labels: Array2<f64>, weights: Array1<f64>) -> ParameterFamily {
        let basis: Arc<dyn SaeBasisEvaluator> = Arc::new(patch());
        let field = MatrixParameterField::new(basis, coefficients).expect("field");
        ParameterFamily::new(field, labels, weights).expect("family")
    }

    struct Edited {
        theta: Array2<f64>,
        magnitude: Array2<f64>,
    }

    /// `m_Delta Theta_* + sum_c (m_c - m_Delta) w_c Gamma(z_c)` from a direct
    /// evaluation of the field, independent of `anchor_basis_weights`, with the
    /// entrywise magnitude of the terms it sums.
    fn edited(family: &ParameterFamily, anchor: &Array2<f64>, mask: &[f64], residual: f64) -> Edited {
        let phi = patch().evaluate(family.labels()).expect("basis at labels").0;
        let mut theta = anchor * residual;
        let mut magnitude = anchor.mapv(f64::abs) * residual.abs();
        for (c, &m) in mask.iter().enumerate() {
            let scale = (m - residual) * family.weights()[c];
            for (j, coefficient) in family.field().coefficients().iter().enumerate() {
                theta = theta + dense(coefficient) * (scale * phi[[c, j]]);
                magnitude = magnitude + dense_magnitude(coefficient) * (scale * phi[[c, j]]).abs();
            }
        }
        Edited { theta, magnitude }
    }

    /// A linear use `y_i = Theta x_i` read out as `sum_r a_r gelu(y_ir + b_r)`,
    /// or compared with a teacher output.
    struct Use {
        inputs: Array2<f64>,
        readout: Array1<f64>,
        bias: Array1<f64>,
    }

    struct Reading {
        value: f64,
        rounding: f64,
        output: Array2<f64>,
    }

    fn gelu(x: f64) -> f64 {
        x * normal_cdf(x)
    }

    fn gelu_slope(x: f64) -> f64 {
        normal_cdf(x) + x * normal_pdf(x)
    }

    /// First-order rounding model: every entry of `Theta` sums
    /// `C J (RANK + 3) + 1` products, each preactivation sums `COLS` more, and
    /// the loss sums one term per output entry; each operation contributes
    /// `eps` times the magnitude of what it forms.
    fn operations(instances: usize, width: usize, reading: &Use) -> f64 {
        (instances * width * (RANK + 3) + 1 + COLS + reading.inputs.nrows() * ROWS) as f64
    }

    /// `sum_i sum_r a_r gelu(y_ir + b_r)` and its per-row cotangents `a_r gelu'`.
    fn read(edit: &Edited, reading: &Use, instances: usize, width: usize) -> Reading {
        let pre = reading.inputs.dot(&edit.theta.t()) + &reading.bias;
        let reach = reading.inputs.mapv(f64::abs).dot(&edit.magnitude.t()) + &reading.bias.mapv(f64::abs);
        let mut value = 0.0;
        let mut magnitude = 0.0;
        let mut output = Array2::<f64>::zeros(pre.dim());
        for ((i, r), &y) in pre.indexed_iter() {
            value += reading.readout[r] * gelu(y);
            magnitude += reading.readout[r].abs() * (gelu(y).abs() + reach[[i, r]]);
            output[[i, r]] = reading.readout[r] * gelu_slope(y);
        }
        Reading {
            value,
            rounding: operations(instances, width, reading) * f64::EPSILON * magnitude,
            output,
        }
    }

    /// `sum_i ||gelu(Theta x_i) - gelu(Theta_* x_i)||^2 / 2` and its per-row
    /// cotangents.
    fn deviation(edit: &Edited, anchor: &Array2<f64>, reading: &Use, instances: usize, width: usize) -> Reading {
        let pre = reading.inputs.dot(&edit.theta.t());
        let teacher = reading.inputs.dot(&anchor.t());
        let reach = reading.inputs.mapv(f64::abs).dot(&edit.magnitude.t());
        let mut value = 0.0;
        let mut magnitude = 0.0;
        let mut output = Array2::<f64>::zeros(pre.dim());
        for ((i, r), &y) in pre.indexed_iter() {
            let gap = gelu(y) - gelu(teacher[[i, r]]);
            value += 0.5 * gap * gap;
            magnitude += gap.abs() * (gelu(y).abs() + gelu(teacher[[i, r]]).abs() + reach[[i, r]]);
            output[[i, r]] = gap * gelu_slope(y);
        }
        Reading {
            value,
            rounding: operations(instances, width, reading) * f64::EPSILON * magnitude,
            output,
        }
    }

    struct Fixture {
        coefficients: Vec<FieldCoefficient>,
        labels: Array2<f64>,
        weights: Array1<f64>,
        anchor: Array2<f64>,
        uses: Vec<Use>,
    }

    /// A mixed field (even coefficients dense, odd ones in the product form) over
    /// the quadratic patch in two latent dimensions, three instances, a teacher
    /// tensor carrying a residual the field does not, and two tied uses.
    fn fixture() -> Fixture {
        let width = patch().basis_size();
        let coefficients: Vec<FieldCoefficient> = (0..width)
            .map(|j| {
                if j % 2 == 0 {
                    FieldCoefficient::Dense(matrix(10 + j, ROWS, COLS))
                } else {
                    FieldCoefficient::Factored {
                        left: matrix(20 + j, ROWS, RANK),
                        right: matrix(30 + j, COLS, RANK),
                    }
                }
            })
            .collect();
        let labels = matrix(40, 3, LATENT);
        let weights = Array1::from(vec![0.5, 0.25, 0.25]);
        let base = family(coefficients.clone(), labels.clone(), weights.clone());
        let instances = edited(&base, &Array2::<f64>::zeros((ROWS, COLS)), &[1.0, 1.0, 1.0], 0.0).theta;
        let anchor = instances + matrix(50, ROWS, COLS);
        let uses = vec![
            Use {
                inputs: matrix(60, 5, COLS),
                readout: Array1::from(entries(61, ROWS)),
                bias: Array1::from(entries(62, ROWS)),
            },
            Use {
                inputs: matrix(70, 4, COLS),
                readout: Array1::from(entries(71, ROWS)),
                bias: Array1::from(entries(72, ROWS)),
            },
        ];
        Fixture {
            coefficients,
            labels,
            weights,
            anchor,
            uses,
        }
    }

    /// Readout loss over both uses with its rounding bound.
    fn readout_loss(fx: &Fixture, family: &ParameterFamily, mask: &[f64], residual: f64) -> (f64, f64) {
        let edit = edited(family, &fx.anchor, mask, residual);
        let width = family.field().coefficients().len();
        fx.uses.iter().fold((0.0, 0.0), |(value, rounding), reading| {
            let part = read(&edit, reading, mask.len(), width);
            (value + part.value, rounding + part.rounding)
        })
    }

    /// The two uses' cotangents, the first as per-row outer products and the
    /// second materialized, together with the dense `G` they sum to.
    fn readout_cotangent(
        fx: &Fixture,
        family: &ParameterFamily,
        mask: &[f64],
        residual: f64,
    ) -> (ParameterCotangent, Array2<f64>) {
        let edit = edited(family, &fx.anchor, mask, residual);
        let width = family.field().coefficients().len();
        let first = read(&edit, &fx.uses[0], mask.len(), width).output;
        let second = read(&edit, &fx.uses[1], mask.len(), width).output.t().dot(&fx.uses[1].inputs);
        let dense_g = first.t().dot(&fx.uses[0].inputs) + &second;
        let cotangent = ParameterCotangent::from_terms(
            ROWS,
            COLS,
            vec![
                CotangentTerm::Outer {
                    output: first,
                    input: fx.uses[0].inputs.clone(),
                },
                CotangentTerm::Dense(second),
            ],
        )
        .expect("cotangent");
        (cotangent, dense_g)
    }

    /// Central differences `D(h) = f' + c h^2 + O(h^4)`: `|D(h) - D(h/2)|` is three
    /// times the leading truncation of `D(h/2)`, and the rounding of the two fine
    /// evaluations enters divided by the step. The step is a design choice; the
    /// tolerance comes from the halved-step check. Returns `(D(h/2), tolerance)`.
    fn checked_difference(loss: &dyn Fn(f64) -> (f64, f64), step: f64) -> (f64, f64) {
        let coarse = (loss(step).0 - loss(-step).0) / (2.0 * step);
        let half = 0.5 * step;
        let (plus, plus_rounding) = loss(half);
        let (minus, minus_rounding) = loss(-half);
        let fine = (plus - minus) / (2.0 * half);
        (fine, (coarse - fine).abs() + (plus_rounding + minus_rounding) / (2.0 * half))
    }

    const STEP: f64 = 1.0e-3;

    #[test]
    fn anchored_pullbacks_match_a_richardson_checked_difference_through_gelu() {
        let fx = fixture();
        let mask = [0.3, -0.7, 1.0];
        let residual = 0.6;
        let base = family(fx.coefficients.clone(), fx.labels.clone(), fx.weights.clone());
        let (cotangent, dense_g) = readout_cotangent(&fx, &base, &mask, residual);
        let pullback = base
            .pullback(ArrayView1::from(&mask[..]), residual, &cotangent)
            .expect("pullback");
        let anchor_weights = base
            .anchor_basis_weights(ArrayView1::from(&mask[..]), residual)
            .expect("anchor weights");
        let moved_coefficient = |j: usize, edit: &dyn Fn(&mut FieldCoefficient, f64), h: f64| {
            let mut coefficients = fx.coefficients.clone();
            edit(&mut coefficients[j], h);
            readout_loss(&fx, &family(coefficients, fx.labels.clone(), fx.weights.clone()), &mask, residual)
        };
        let mut checked = 0usize;
        for (j, coefficient) in fx.coefficients.iter().enumerate() {
            match (coefficient, &pullback.coefficients[j]) {
                (FieldCoefficient::Dense(b), CoefficientPullback::Dense { weight }) => {
                    assert_eq!(*weight, anchor_weights[j], "dense coefficient {j}: the weight is s_j");
                    assert_eq!(b.dim(), (ROWS, COLS));
                    for r in 0..ROWS {
                        for k in 0..COLS {
                            let loss = |h: f64| {
                                moved_coefficient(
                                    j,
                                    &|target: &mut FieldCoefficient, h: f64| {
                                        if let FieldCoefficient::Dense(m) = target {
                                            m[[r, k]] += h;
                                        }
                                    },
                                    h,
                                )
                            };
                            let analytic = weight * dense_g[[r, k]];
                            let (fine, tolerance) = checked_difference(&loss, STEP);
                            assert!(
                                (analytic - fine).abs() <= tolerance,
                                "d loss / d B_{j}[{r},{k}]: pullback {analytic} against difference {fine} (tolerance {tolerance:e})"
                            );
                            checked += 1;
                        }
                    }
                }
                (
                    FieldCoefficient::Factored { left, right },
                    CoefficientPullback::Factored {
                        left: d_left,
                        right: d_right,
                    },
                ) => {
                    assert_eq!(d_left.dim(), left.dim());
                    assert_eq!(d_right.dim(), right.dim());
                    for ((r, l), analytic) in d_left.indexed_iter() {
                        let loss = |h: f64| {
                            moved_coefficient(
                                j,
                                &|target: &mut FieldCoefficient, h: f64| {
                                    if let FieldCoefficient::Factored { left: u, .. } = target {
                                        u[[r, l]] += h;
                                    }
                                },
                                h,
                            )
                        };
                        let (fine, tolerance) = checked_difference(&loss, STEP);
                        assert!(
                            (analytic - fine).abs() <= tolerance,
                            "d loss / d U_{j}[{r},{l}]: pullback {analytic} against difference {fine} (tolerance {tolerance:e})"
                        );
                        checked += 1;
                    }
                    for ((k, l), analytic) in d_right.indexed_iter() {
                        let loss = |h: f64| {
                            moved_coefficient(
                                j,
                                &|target: &mut FieldCoefficient, h: f64| {
                                    if let FieldCoefficient::Factored { right: v, .. } = target {
                                        v[[k, l]] += h;
                                    }
                                },
                                h,
                            )
                        };
                        let (fine, tolerance) = checked_difference(&loss, STEP);
                        assert!(
                            (analytic - fine).abs() <= tolerance,
                            "d loss / d V_{j}[{k},{l}]: pullback {analytic} against difference {fine} (tolerance {tolerance:e})"
                        );
                        checked += 1;
                    }
                }
                (FieldCoefficient::Dense(..), CoefficientPullback::Factored { .. })
                | (FieldCoefficient::Factored { .. }, CoefficientPullback::Dense { .. }) => {}
            }
        }
        for c in 0..fx.weights.len() {
            let loss = |h: f64| {
                let mut weights = fx.weights.clone();
                weights[c] += h;
                readout_loss(&fx, &family(fx.coefficients.clone(), fx.labels.clone(), weights), &mask, residual)
            };
            let (fine, tolerance) = checked_difference(&loss, STEP);
            let analytic = pullback.weights[c];
            assert!(
                (analytic - fine).abs() <= tolerance,
                "d loss / d w_{c}: pullback {analytic} against difference {fine} (tolerance {tolerance:e})"
            );
            checked += 1;
        }
        // On the gauge slice: a move along e_0 - e_1 keeps sum_c w_c = 1 without
        // re-gauging, and the tangent derivative sums to zero within its rounding
        // (the mean and each subtraction round once per instance).
        let tangent = pullback.gauge_tangent_weights();
        let raw_scale = pullback.weights.iter().map(|g| g.abs()).sum::<f64>();
        let tangent_rounding = (2 * fx.weights.len()) as f64 * f64::EPSILON * raw_scale;
        assert!(
            tangent.sum().abs() <= tangent_rounding,
            "gauge tangent {tangent:?} sums to {} (rounding {tangent_rounding:e})",
            tangent.sum()
        );
        // Positive control: the raw weight derivative leaves the slice.
        assert!(
            pullback.weights.sum().abs() > tangent_rounding,
            "the raw weight derivative {:?} already sums to zero, so the projection is not exercised",
            pullback.weights
        );
        let along_slice = |h: f64| {
            let mut weights = fx.weights.clone();
            weights[0] += h;
            weights[1] -= h;
            readout_loss(&fx, &family(fx.coefficients.clone(), fx.labels.clone(), weights), &mask, residual)
        };
        let (fine, tolerance) = checked_difference(&along_slice, STEP);
        let analytic = tangent[0] - tangent[1];
        assert!(
            (analytic - fine).abs() <= tolerance,
            "slice derivative along e_0 - e_1: tangent {analytic} against difference {fine} (tolerance {tolerance:e})"
        );
        for c in 0..fx.labels.nrows() {
            let mut resolved = false;
            for a in 0..LATENT {
                let loss = |h: f64| {
                    let mut labels = fx.labels.clone();
                    labels[[c, a]] += h;
                    readout_loss(&fx, &family(fx.coefficients.clone(), labels, fx.weights.clone()), &mask, residual)
                };
                let (fine, tolerance) = checked_difference(&loss, STEP);
                let analytic = pullback.labels[[c, a]];
                assert!(
                    (analytic - fine).abs() <= tolerance,
                    "d loss / d z_{c}^{a}: pullback {analytic} against difference {fine} (tolerance {tolerance:e})"
                );
                // Negative control: a fixed-moment derivative has no label term,
                // and the difference refutes that zero.
                resolved |= fine.abs() > tolerance;
                checked += 1;
            }
            assert!(
                resolved,
                "instance {c}: no label derivative is resolved from zero, so the fixed-moment derivative is not refuted"
            );
        }
        let width = fx.coefficients.len();
        let factored = fx
            .coefficients
            .iter()
            .filter(|b| matches!(b, FieldCoefficient::Factored { .. }))
            .count();
        let expected = (width - factored) * ROWS * COLS + factored * (ROWS + COLS) * RANK + 3 + 3 * LATENT;
        assert_eq!(checked, expected, "every coordinate was differenced");
    }

    #[test]
    fn every_anchored_derivative_is_exactly_zero_at_all_on() {
        let fx = fixture();
        let base = family(fx.coefficients.clone(), fx.labels.clone(), fx.weights.clone());
        let all_on = [1.0, 1.0, 1.0];
        let (cotangent, dense_g) = readout_cotangent(&fx, &base, &all_on, 1.0);
        assert!(dense_g.iter().any(|&v| v != 0.0), "the all-on cotangent is not itself zero");
        let at_all_on = base
            .pullback(ArrayView1::from(&all_on[..]), 1.0, &cotangent)
            .expect("all-on pullback");
        let anchor_weights = base
            .anchor_basis_weights(ArrayView1::from(&all_on[..]), 1.0)
            .expect("all-on anchor weights");
        assert!(anchor_weights.iter().all(|&v| v == 0.0), "all-on anchor weights {anchor_weights:?}");
        assert!(at_all_on.labels.iter().all(|&v| v == 0.0), "all-on label derivative {:?}", at_all_on.labels);
        assert!(at_all_on.weights.iter().all(|&v| v == 0.0), "all-on weight derivative {:?}", at_all_on.weights);
        let moves = |derivative: &CoefficientPullback| match derivative {
            CoefficientPullback::Dense { weight } => *weight != 0.0,
            CoefficientPullback::Factored { left, right } => left.iter().chain(right.iter()).any(|&v| v != 0.0),
        };
        for (j, derivative) in at_all_on.coefficients.iter().enumerate() {
            assert!(!moves(derivative), "coefficient {j} derivative at all-on: {derivative:?}");
        }
        // Positive control: the same cotangent with instance 1 deleted moves that
        // instance's label and weight and the coefficients, and leaves the kept
        // instances' labels exactly still.
        let deleted = [1.0, 0.0, 1.0];
        let moved = base
            .pullback(ArrayView1::from(&deleted[..]), 1.0, &cotangent)
            .expect("pullback with instance 1 deleted");
        assert!(moved.labels.row(1).iter().any(|&v| v != 0.0), "deleted label {:?}", moved.labels);
        assert!(moved.weights[1] != 0.0, "deleted weight {:?}", moved.weights);
        assert!(moved.coefficients.iter().any(moves), "coefficients {:?}", moved.coefficients);
        assert!(
            moved.labels.row(0).iter().chain(moved.labels.row(2).iter()).all(|&v| v == 0.0),
            "kept labels {:?}",
            moved.labels
        );
    }

    /// The binary mask with bit `c` of `code` as instance `c`'s control.
    fn binary_mask(code: usize) -> [f64; 3] {
        [(code & 1) as f64, ((code >> 1) & 1) as f64, ((code >> 2) & 1) as f64]
    }

    struct Fidelity {
        value: f64,
        rounding: f64,
        witness: usize,
        runner_up: f64,
        runner_up_rounding: f64,
    }

    /// `R = max_m sum_i ||gelu(Theta(m) x_i) - gelu(Theta_* x_i)||^2 / 2`, exhaustive
    /// over the binary masks with the residual kept: the value, its rounding, the
    /// witness code, and the second-largest objective with its rounding.
    fn adversarial_fidelity(fx: &Fixture, family: &ParameterFamily) -> Fidelity {
        let width = family.field().coefficients().len();
        let mut best = Fidelity {
            value: f64::NEG_INFINITY,
            rounding: 0.0,
            witness: 0,
            runner_up: f64::NEG_INFINITY,
            runner_up_rounding: 0.0,
        };
        for code in 0..8 {
            let mask = binary_mask(code);
            let edit = edited(family, &fx.anchor, &mask, 1.0);
            let part = deviation(&edit, &fx.anchor, &fx.uses[0], mask.len(), width);
            if part.value > best.value {
                best.runner_up = best.value;
                best.runner_up_rounding = best.rounding;
                best.value = part.value;
                best.rounding = part.rounding;
                best.witness = code;
            } else if part.value > best.runner_up {
                best.runner_up = part.value;
                best.runner_up_rounding = part.rounding;
            }
        }
        best
    }

    #[test]
    fn danskin_derivative_is_the_pullback_at_the_mask_witness_not_at_the_moment() {
        let fx = fixture();
        let base = family(fx.coefficients.clone(), fx.labels.clone(), fx.weights.clone());
        let best = adversarial_fidelity(&fx, &base);
        let witness = best.witness;
        let mask = binary_mask(witness);
        assert!(
            best.value > 0.0 && mask.iter().any(|&m| m == 0.0),
            "the witness {mask:?} deletes an instance (R = {})",
            best.value
        );
        // Danskin needs a unique maximizer: the witness margin is resolved above
        // the rounding of the two objective values it separates.
        let margin = best.value - best.runner_up;
        assert!(
            margin > best.rounding + best.runner_up_rounding,
            "witness margin {margin:e} is not resolved above rounding {:e}",
            best.rounding + best.runner_up_rounding
        );
        let edit = edited(&base, &fx.anchor, &mask, 1.0);
        let output = deviation(&edit, &fx.anchor, &fx.uses[0], mask.len(), fx.coefficients.len()).output;
        let cotangent = ParameterCotangent::from_terms(
            ROWS,
            COLS,
            vec![CotangentTerm::Outer {
                output,
                input: fx.uses[0].inputs.clone(),
            }],
        )
        .expect("witness cotangent");
        let pullback = base
            .pullback(ArrayView1::from(&mask[..]), 1.0, &cotangent)
            .expect("witness pullback");
        let mut refuted = false;
        for c in 0..fx.labels.nrows() {
            for a in 0..LATENT {
                let loss = |h: f64| {
                    let mut labels = fx.labels.clone();
                    labels[[c, a]] += h;
                    let moved = adversarial_fidelity(&fx, &family(fx.coefficients.clone(), labels, fx.weights.clone()));
                    // Danskin's hypothesis on the stencil: one mask stays the maximizer.
                    assert_eq!(moved.witness, witness, "z_{c}^{a} moved by {h}: the witness changed");
                    (moved.value, moved.rounding)
                };
                let (fine, tolerance) = checked_difference(&loss, STEP);
                let analytic = pullback.labels[[c, a]];
                assert!(
                    (analytic - fine).abs() <= tolerance,
                    "dR / d z_{c}^{a}: witness pullback {analytic} against difference {fine} (tolerance {tolerance:e})"
                );
                if mask[c] == 0.0 {
                    refuted |= fine.abs() > tolerance;
                }
            }
        }
        // Negative control: the fixed-moment derivative of R has no label term.
        assert!(
            refuted,
            "no deleted instance's label derivative is resolved from zero, so the fixed-moment derivative is not refuted"
        );
    }

    #[test]
    fn scale_gauge_keeps_every_instance_and_refuses_an_unresolved_total() {
        let fx = fixture();
        let raw_weights = Array1::from(vec![1.5, 0.6, 0.4]);
        let gauged = family(fx.coefficients.clone(), fx.labels.clone(), raw_weights.clone());
        // Each w_c / t rounds once, and the sum of C terms is exact to
        // (C - 1) eps sum_c |w_c|; the rounding of t itself moves the sum by at most
        // the same amount.
        let instances = raw_weights.len();
        let sum_rounding = (3 * instances - 2) as f64
            * f64::EPSILON
            * gauged.weights().iter().map(|w| w.abs()).sum::<f64>();
        let weight_sum = gauged.weights().sum();
        assert!(
            (weight_sum - 1.0).abs() <= sum_rounding,
            "gauge-fixed weights sum to {weight_sum} (rounding {sum_rounding:e})"
        );
        let phi = patch().evaluate(fx.labels.view()).expect("basis at labels").0;
        let width = fx.coefficients.len();
        // Both sides sum J products of at most RANK + 3 factors; the gauge adds one
        // division and one multiplication.
        let operations = (2 * (width * (RANK + 3) + 2)) as f64;
        for c in 0..instances {
            let mut raw = Array2::<f64>::zeros((ROWS, COLS));
            let mut gauge_fixed = Array2::<f64>::zeros((ROWS, COLS));
            let mut magnitude = Array2::<f64>::zeros((ROWS, COLS));
            for j in 0..width {
                raw = raw + dense(&fx.coefficients[j]) * (raw_weights[c] * phi[[c, j]]);
                gauge_fixed =
                    gauge_fixed + dense(&gauged.field().coefficients()[j]) * (gauged.weights()[c] * phi[[c, j]]);
                magnitude = magnitude + dense_magnitude(&fx.coefficients[j]) * (raw_weights[c] * phi[[c, j]]).abs();
            }
            for ((r, k), &value) in raw.indexed_iter() {
                let tolerance = operations * f64::EPSILON * magnitude[[r, k]];
                assert!(
                    (value - gauge_fixed[[r, k]]).abs() <= tolerance,
                    "instance {c} entry [{r},{k}]: raw {value} against gauge-fixed {} (tolerance {tolerance:e})",
                    gauge_fixed[[r, k]]
                );
            }
        }
        let field = || MatrixParameterField::new(Arc::new(patch()), fx.coefficients.clone()).expect("field");
        let two_labels = fx.labels.slice(ndarray::s![0..2, ..]).to_owned();
        assert!(
            ParameterFamily::new(field(), two_labels, Array1::from(vec![0.5, -0.5])).is_err(),
            "a zero total is refused"
        );
        assert!(
            ParameterFamily::new(field(), fx.labels.clone(), Array1::from(vec![1.0, -1.0, 0.5 * f64::EPSILON]))
                .is_err(),
            "a total of eps/2 inside its rounding band 2 eps (2 + eps/2) is refused"
        );
        // Positive control: a total resolved above the same band is accepted.
        assert!(
            ParameterFamily::new(field(), fx.labels.clone(), Array1::from(vec![1.0, -1.0, 16.0 * f64::EPSILON]))
                .is_ok(),
            "a total of 16 eps above its rounding band 2 eps (2 + 16 eps) is accepted"
        );
    }

    /// The declared latent measure for the penalty tests: sixteen centers with
    /// equal mass, at an order inside the owner's admissible `(0, 2)` and its
    /// density-weighted exponent.
    const ORDER_S: f64 = 1.5;
    const ALPHA: f64 = 1.0;

    fn latent_measure() -> (Array2<f64>, Array1<f64>) {
        (matrix(80, 16, LATENT), Array1::from_elem(16, 1.0 / 16.0))
    }

    fn measure_jet_penalty() -> FieldPenalty {
        let (centers, masses) = latent_measure();
        FieldPenalty::measure_jet(&patch(), centers.view(), masses.view(), &declared_spec()).expect("measure-jet penalty")
    }

    /// The declared measure-jet spec: an explicit order in (0, 2), the
    /// density-weighted exponent, and the owner's remaining fields.
    fn declared_spec() -> MeasureJetBasisSpec {
        MeasureJetBasisSpec {
            order_s: ORDER_S,
            alpha: ALPHA,
            ..MeasureJetBasisSpec::default()
        }
    }

    fn energy() -> Array2<f64> {
        let (centers, masses) = latent_measure();
        let spec = declared_spec();
        let band = measure_jet_band(centers.view(), spec.num_scales).expect("band");
        measure_jet_energy_form(centers.view(), masses.view(), &band, spec.order_s, spec.alpha)
            .expect("energy")
    }

    /// `Gamma(c_i)` and the entrywise magnitude of its terms at every center.
    fn center_values(coefficients: &[FieldCoefficient]) -> (Vec<Array2<f64>>, Vec<Array2<f64>>) {
        let centers = latent_measure().0;
        let phi = patch().evaluate(centers.view()).expect("basis at centers").0;
        let mut values = Vec::new();
        let mut magnitudes = Vec::new();
        for i in 0..centers.nrows() {
            let mut value = Array2::<f64>::zeros((ROWS, COLS));
            let mut magnitude = Array2::<f64>::zeros((ROWS, COLS));
            for (j, coefficient) in coefficients.iter().enumerate() {
                value = value + dense(coefficient) * phi[[i, j]];
                magnitude = magnitude + dense_magnitude(coefficient) * phi[[i, j]].abs();
            }
            values.push(value);
            magnitudes.push(magnitude);
        }
        (values, magnitudes)
    }

    #[test]
    fn measure_jet_penalty_is_the_energy_of_the_field_values() {
        let fx = fixture();
        let field = MatrixParameterField::new(Arc::new(patch()), fx.coefficients.clone()).expect("field");
        let q = energy();
        let (values, magnitudes) = center_values(&fx.coefficients);
        let m = values.len();
        let width = fx.coefficients.len();
        let mut reference = 0.0;
        let mut diagonal_only = 0.0;
        let mut scale = 0.0;
        for i in 0..m {
            for k in 0..m {
                let inner = frobenius(values[i].view(), values[k].view());
                reference += q[[i, k]] * inner;
                if i == k {
                    diagonal_only += q[[i, k]] * inner;
                }
                scale += q[[i, k]].abs() * frobenius(magnitudes[i].view(), magnitudes[k].view());
            }
        }
        // First-order rounding: both sides sum m^2 J^2 ROWS COLS products of at
        // most RANK + 3 factors, each rounding once against the magnitude it forms.
        let tolerance = (m * m * width * width * ROWS * COLS * (RANK + 3)) as f64 * f64::EPSILON * scale;
        let value = field.penalty(&measure_jet_penalty()).expect("penalty value");
        assert!(
            (value - reference).abs() <= tolerance,
            "coefficient-form penalty {value} against the energy of the center values {reference} (tolerance {tolerance:e})"
        );
        assert!(reference > tolerance, "the fixture's roughness {reference} is resolved above {tolerance:e}");
        // Positive control: the diagonal of Q alone is refuted by the same tolerance.
        assert!(
            (diagonal_only - reference).abs() > tolerance,
            "diagonal-only energy {diagonal_only} is not separated from {reference} (tolerance {tolerance:e})"
        );
    }

    #[test]
    fn measure_jet_penalty_annihilates_an_affine_field_and_charges_a_curved_one() {
        let exponents = monomial_exponents(LATENT, DEGREE);
        let affine: Vec<FieldCoefficient> = exponents
            .iter()
            .enumerate()
            .map(|(j, exponent)| {
                if exponent.iter().sum::<usize>() <= 1 {
                    FieldCoefficient::Dense(matrix(90 + j, ROWS, COLS))
                } else {
                    FieldCoefficient::Dense(Array2::<f64>::zeros((ROWS, COLS)))
                }
            })
            .collect();
        let curved_column = exponents
            .iter()
            .position(|exponent| exponent.iter().sum::<usize>() == 2)
            .expect("the quadratic patch has a degree-two column");
        let mut curved = affine.clone();
        curved[curved_column] = FieldCoefficient::Dense(matrix(99, ROWS, COLS));
        let penalty = measure_jet_penalty();
        let q = energy();
        let q_norm = q.iter().map(|v| v * v).sum::<f64>().sqrt();
        // The energy annihilates affine center values exactly in exact arithmetic.
        // Its floor is the roundoff of m local blocks, eps ||Q||_F per unit center
        // value, against sum_i ||Gamma(c_i)||_F^2.
        let floor = |coefficients: &[FieldCoefficient]| {
            let values = center_values(coefficients).0;
            let squared: f64 = values.iter().map(|v| frobenius(v.view(), v.view())).sum();
            values.len() as f64 * f64::EPSILON * q_norm * squared
        };
        let affine_value = MatrixParameterField::new(Arc::new(patch()), affine.clone())
            .expect("affine field")
            .penalty(&penalty)
            .expect("affine penalty");
        let affine_floor = floor(&affine);
        assert!(
            affine_value.abs() <= affine_floor,
            "affine field pays {affine_value:e} above its roundoff floor {affine_floor:e}"
        );
        // Positive control: one degree-two coefficient is charged above the same floor.
        let curved_value = MatrixParameterField::new(Arc::new(patch()), curved.clone())
            .expect("curved field")
            .penalty(&penalty)
            .expect("curved penalty");
        let curved_floor = floor(&curved);
        assert!(
            curved_value > curved_floor,
            "curved field pays {curved_value:e}, not above its floor {curved_floor:e}"
        );
    }

    #[test]
    fn penalty_gradient_matches_a_richardson_checked_difference() {
        let fx = fixture();
        let penalty = measure_jet_penalty();
        let field = MatrixParameterField::new(Arc::new(patch()), fx.coefficients.clone()).expect("field");
        let gradient = field.penalty_gradient(&penalty).expect("penalty gradient");
        let width = fx.coefficients.len();
        let gram = penalty.gram();
        let mut scale = 0.0;
        for j in 0..width {
            for l in 0..width {
                scale += gram[[j, l]].abs()
                    * frobenius(dense_magnitude(&fx.coefficients[j]).view(), dense_magnitude(&fx.coefficients[l]).view());
            }
        }
        // The penalty sums J^2 inner products of ROWS COLS products of at most
        // RANK + 3 factors.
        let rounding = (width * width * ROWS * COLS * (RANK + 3)) as f64 * f64::EPSILON * scale;
        let penalty_at = |coefficients: Vec<FieldCoefficient>| {
            let value = MatrixParameterField::new(Arc::new(patch()), coefficients)
                .expect("moved field")
                .penalty(&penalty)
                .expect("moved penalty");
            (value, rounding)
        };
        let mut checked = 0usize;
        let mut half_refuted = false;
        for (j, coefficient) in fx.coefficients.iter().enumerate() {
            match (coefficient, &gradient[j]) {
                (FieldCoefficient::Dense(b), PenaltyGradient::Dense(d)) => {
                    assert_eq!(b.dim(), d.dim());
                    for ((r, k), analytic) in d.indexed_iter() {
                        let loss = |h: f64| {
                            let mut moved = fx.coefficients.clone();
                            if let FieldCoefficient::Dense(x) = &mut moved[j] {
                                x[[r, k]] += h;
                            }
                            penalty_at(moved)
                        };
                        let (fine, tolerance) = checked_difference(&loss, STEP);
                        assert!(
                            (analytic - fine).abs() <= tolerance,
                            "dJ / dB_{j}[{r},{k}]: gradient {analytic} against difference {fine} (tolerance {tolerance:e})"
                        );
                        half_refuted |= (0.5 * analytic - fine).abs() > tolerance;
                        checked += 1;
                    }
                }
                (
                    FieldCoefficient::Factored { left, right },
                    PenaltyGradient::Factored {
                        left: d_left,
                        right: d_right,
                    },
                ) => {
                    assert_eq!(left.dim(), d_left.dim());
                    assert_eq!(right.dim(), d_right.dim());
                    for ((r, l), analytic) in d_left.indexed_iter() {
                        let loss = |h: f64| {
                            let mut moved = fx.coefficients.clone();
                            if let FieldCoefficient::Factored { left: u, .. } = &mut moved[j] {
                                u[[r, l]] += h;
                            }
                            penalty_at(moved)
                        };
                        let (fine, tolerance) = checked_difference(&loss, STEP);
                        assert!(
                            (analytic - fine).abs() <= tolerance,
                            "dJ / dU_{j}[{r},{l}]: gradient {analytic} against difference {fine} (tolerance {tolerance:e})"
                        );
                        half_refuted |= (0.5 * analytic - fine).abs() > tolerance;
                        checked += 1;
                    }
                    for ((k, l), analytic) in d_right.indexed_iter() {
                        let loss = |h: f64| {
                            let mut moved = fx.coefficients.clone();
                            if let FieldCoefficient::Factored { right: v, .. } = &mut moved[j] {
                                v[[k, l]] += h;
                            }
                            penalty_at(moved)
                        };
                        let (fine, tolerance) = checked_difference(&loss, STEP);
                        assert!(
                            (analytic - fine).abs() <= tolerance,
                            "dJ / dV_{j}[{k},{l}]: gradient {analytic} against difference {fine} (tolerance {tolerance:e})"
                        );
                        half_refuted |= (0.5 * analytic - fine).abs() > tolerance;
                        checked += 1;
                    }
                }
                (FieldCoefficient::Dense(..), PenaltyGradient::Factored { .. })
                | (FieldCoefficient::Factored { .. }, PenaltyGradient::Dense(..)) => {}
            }
        }
        let factored = fx
            .coefficients
            .iter()
            .filter(|b| matches!(b, FieldCoefficient::Factored { .. }))
            .count();
        assert_eq!(
            checked,
            (width - factored) * ROWS * COLS + factored * (ROWS + COLS) * RANK,
            "every coefficient coordinate was differenced"
        );
        // Positive control: a gradient missing the factor 2 of the symmetric form
        // is refuted.
        assert!(half_refuted, "half the gradient is not refuted anywhere");
    }

    #[test]
    fn measure_jet_penalty_refuses_coefficient_directions_invisible_at_the_centers() {
        let basis = patch();
        let (centers, masses) = latent_measure();
        // Four centers for six quadratic functions: rank at most 4.
        let four = centers.slice(ndarray::s![0..4, ..]).to_owned();
        let four_masses = Array1::from_elem(4, 0.25);
        let narrow = FieldPenalty::measure_jet(&basis, four.view(), four_masses.view(), &declared_spec())
            .expect_err("six functions at four centers are refused");
        assert!(
            narrow.contains("rank 4") && narrow.contains("vanish at every center"),
            "refusal names the rank deficit: {narrow}"
        );
        // Sixteen collinear centers with z_2 = 2 z_1, which doubling represents
        // exactly: every column is exactly 1, z_1 or z_1^2 times a constant, so the
        // rank is 3 however many centers there are.
        let mut collinear = centers.clone();
        for i in 0..collinear.nrows() {
            collinear[[i, 1]] = 2.0 * collinear[[i, 0]];
        }
        let flat = FieldPenalty::measure_jet(&basis, collinear.view(), masses.view(), &declared_spec())
            .expect_err("a quadratic basis on a line is refused");
        assert!(
            flat.contains("rank 3") && flat.contains("vanish at every center"),
            "refusal names the rank deficit: {flat}"
        );
        // Positive control: the same sixteen centers in general position are accepted.
        assert!(
            FieldPenalty::measure_jet(&basis, centers.view(), masses.view(), &declared_spec()).is_ok(),
            "sixteen centers in general position see every coefficient direction"
        );
    }

    #[test]
    fn measure_jet_null_component_closes_exactly_the_energy_null_space() {
        let (centers, masses) = latent_measure();
        let energy_gram = measure_jet_penalty().gram().to_owned();
        let null_gram = FieldPenalty::measure_jet_affine_null(&patch(), centers.view(), masses.view())
            .expect("null-component penalty")
            .gram()
            .to_owned();
        let phi = patch().evaluate(centers.view()).expect("basis at centers").0;
        let phi_squared = phi.iter().map(|v| v * v).sum::<f64>();
        let q_norm = energy().iter().map(|v| v * v).sum::<f64>().sqrt();
        let null_form =
            affine_function_nullspace_form(centers.view(), masses.view()).expect("owner's affine null form");
        let null_norm = null_form.iter().map(|v| v * v).sum::<f64>().sqrt();
        // A Gram's unresolved band: the eigensolver's band of its own spectrum plus
        // the owner's construction floor, m eps ||form||_F per unit center value,
        // carried through ||Phi_c||_F^2. Each floor is denominated in the center-value
        // form (Q or H_0), never in the pulled-back Gram.
        let m = centers.nrows() as f64;
        let unresolved = |gram: &Array2<f64>, form_norm: f64| {
            let eigenvalues = gram.eigh(Side::Lower).expect("symmetric eigendecomposition").0.to_vec();
            let band = symmetric_spectrum_rounding_band(&eigenvalues) + m * f64::EPSILON * form_norm * phi_squared;
            (eigenvalues.iter().filter(|&&lambda| lambda <= band).count(), band)
        };
        let affine_columns = monomial_exponents(LATENT, DEGREE)
            .iter()
            .filter(|exponent| exponent.iter().sum::<usize>() <= 1)
            .count();
        // The energy annihilates exactly the affine functions 1, z_1, z_2.
        let (energy_null, energy_band) = unresolved(&energy_gram, q_norm);
        assert_eq!(
            energy_null, affine_columns,
            "energy Gram has {energy_null} eigenvalues inside its band {energy_band:e}; the affine span has {affine_columns}"
        );
        // The null component sees only the affine part, so it leaves J - 3 unresolved.
        let (null_null, null_band) = unresolved(&null_gram, null_norm);
        assert_eq!(
            null_null,
            energy_gram.nrows() - affine_columns,
            "null component has {null_null} eigenvalues inside {null_band:e}"
        );
        // Together the two forms charge every coefficient direction.
        let combined = &energy_gram + &null_gram;
        let (combined_null, combined_band) = unresolved(&combined, q_norm + null_norm);
        assert_eq!(
            combined_null, 0,
            "energy plus null component leaves {combined_null} eigenvalues inside {combined_band:e}"
        );
    }
}
