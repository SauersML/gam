//! Rules of a transformer's attention heads (#2951): one body, read through what the decoder
//! already holds, bound per head by a scale and the head's own subspace.
//!
//! Heads that implement one function in different subspaces share no weights: their query-key and
//! output-value maps disagree entrywise and under any change of each head's coordinates. What they
//! share is a relation through a body the decoder holds, the other heads' maps and the model's norm
//! gains. Two such relations, as the operator a rule derives from the operators decoded before it:
//!
//! * **match** ([`match_prediction`]): a query head's content rows (its rotary planes from `first` to
//!   the slowest, where position barely turns) read, as the current token, what its key rows read as
//!   the previous token through an earlier head's output-value circuit `M = O_s V_s diag(g_s)`. With
//!   `Z = K[R] diag(g) M` on the rows `R` and `Z Zᵀ = U Λ Uᵀ`, the query rows are
//!   `λ U_k Λ_k⁻¹ U_kᵀ Z diag(1/g)`, so `diag(g) Q[R]ᵀ K[R] diag(g) M = λ Π`, `Π` the projector onto
//!   `Z`'s first `k` directions: the score is `λ` wherever the current token is the previous one.
//! * **copy** ([`copy_prediction`]): an output head writes what its value head read, through the norm
//!   gains, `O = λ diag(g / g_f) V⁺`, so the logits rise for the token the head attended to.
//!
//! [`match_alignment`] and [`copy_alignment`] say how much of a head's own circuit a rule's body
//! accounts for, before anything is priced; [`match_scale`] and [`copy_scale`] are the bindings'
//! least-squares scales. [`with_operator`] gives the program with one operator's matrix replaced, so a
//! rule's derived operator runs in the model itself.

use super::dense::{eigh, svd};
use super::operator_program::{Operator, OperatorProgram, Provenance, exact_precision};
use gam_linalg::roundoff::SymmetricAssembly;
use ndarray::{Array1, Array2, Axis};
use std::sync::Arc;

/// The rows of a `width`-wide rotate-half head holding its rotary planes `first..width/2` (both
/// halves of each plane), the slowest last.
pub fn content_rows(width: usize, first: usize) -> Vec<usize> {
    let half = width / 2;
    (first..half).chain(first + half..width).collect()
}

/// `x` with each column `j` scaled by `scale[j]`.
fn scale_columns(x: &Array2<f64>, scale: &Array1<f64>) -> Array2<f64> {
    x * &scale.view().insert_axis(Axis(0))
}

/// The pseudo-inverse of `x` over its singular values beyond the decomposition's band.
fn pseudo_inverse(x: &Array2<f64>) -> Result<Array2<f64>, String> {
    let d = svd(x.view(), false).map_err(|e| format!("{e:?}"))?;
    let kept: Vec<usize> = (0..d.singular_values.len()).filter(|i| d.singular_values[*i] > d.band).collect();
    let inverse = Array1::from_iter(kept.iter().map(|i| 1.0 / d.singular_values[*i]));
    let scaled = &d.u.select(Axis(1), &kept).t() * &inverse.insert_axis(Axis(1));
    Ok(d.vt.select(Axis(0), &kept).t().dot(&scaled))
}

/// What the match rule reads a head's key rows through: `Z = K[R] diag(g) O_s V_s diag(g_s)`, `R`
/// the content rows, `O_s` (`d × width`) and `V_s` (`width × d`) the source head's output and value
/// operators, `g` and `g_s` the norm gains the two layers read through.
pub fn match_reading(key: &Array2<f64>, source_output: &Array2<f64>, source_value: &Array2<f64>, gain: &Array1<f64>, source_gain: &Array1<f64>, rows: &[usize]) -> Array2<f64> {
    let circuit = scale_columns(&source_output.dot(source_value), source_gain);
    scale_columns(&key.select(Axis(0), rows), gain).dot(&circuit)
}

/// The match rule's query rows (`|R| × d`, unscaled): `U_k Λ_k⁻¹ U_kᵀ Z diag(1/g)` over `Z Zᵀ`'s
/// `directions` leading eigenvectors beyond its band (all of them when `None`).
pub fn match_prediction(reading: &Array2<f64>, gain: &Array1<f64>, directions: Option<usize>) -> Result<Array2<f64>, String> {
    let e = eigh(reading.dot(&reading.t()).view(), SymmetricAssembly::Mirrored, None).map_err(|e| format!("{e:?}"))?;
    let mut order: Vec<usize> = (0..e.values.len()).filter(|i| e.values[*i] > e.band).collect();
    order.sort_by(|a, b| e.values[*b].total_cmp(&e.values[*a]));
    order.truncate(directions.unwrap_or(order.len()));
    let u = e.vectors.select(Axis(1), &order);
    let inverse = Array1::from_iter(order.iter().map(|i| 1.0 / e.values[*i]));
    let projector = (&u * &inverse.view().insert_axis(Axis(0))).dot(&u.t());
    Ok(scale_columns(&projector.dot(reading), &gain.mapv(|g| 1.0 / g)))
}

/// `T = diag(g) Q[R]ᵀ Z` for a head's query rows: the head's content scores between a query's
/// current token and a key's previous token, through the match rule's reading.
fn match_form(query: &Array2<f64>, reading: &Array2<f64>, gain: &Array1<f64>, rows: &[usize]) -> Array2<f64> {
    scale_columns(&query.select(Axis(0), rows), gain).t().dot(reading)
}

/// How much a head's content scores are the match rule's: `tr T / (‖T‖_F √r)`, `r` the numerical
/// rank of `T`; one for `T` a positive multiple of a rank-`r` orthogonal projector, about zero for a
/// form unrelated to the identity.
pub fn match_alignment(query: &Array2<f64>, reading: &Array2<f64>, gain: &Array1<f64>, rows: &[usize]) -> Result<f64, String> {
    let t = match_form(query, reading, gain, rows);
    let d = svd(t.view(), false).map_err(|e| format!("{e:?}"))?;
    let rank = d.singular_values.iter().filter(|s| **s > d.band).count().max(1);
    let norm = d.singular_values.iter().map(|s| s * s).sum::<f64>().sqrt();
    Ok(if norm > 0.0 { t.diag().sum() / (norm * (rank as f64).sqrt()) } else { 0.0 })
}

/// The match rule's scale for a head: the least-squares `λ` in `T ≈ λ Π` over the projector the
/// prediction realizes, `tr(T Π) / rank Π`, which is `Σ (Q[R] diag(g)) ∘ Z / rank Z` for the full
/// projector.
pub fn match_scale(query: &Array2<f64>, reading: &Array2<f64>, gain: &Array1<f64>, rows: &[usize], prediction: &Array2<f64>) -> f64 {
    // tr(T Π) = tr(diag(g) Q[R]ᵀ Z Π) with Π = diag(g) P̃ᵀ Z for the prediction P̃ (unscaled).
    let t = match_form(query, reading, gain, rows);
    let projector = match_form(prediction, reading, gain, &(0..prediction.nrows()).collect::<Vec<_>>());
    let rank = projector.diag().sum();
    if rank > 0.0 { (&t * &projector.t()).sum() / rank } else { 0.0 }
}

/// The copy rule's output head (`d × width`, unscaled): `diag(g / g_f) V⁺`, `V` the head's value
/// operator (`width × d`), `g` its layer's norm gain and `g_f` the final norm's.
pub fn copy_prediction(value: &Array2<f64>, gain: &Array1<f64>, final_gain: &Array1<f64>) -> Result<Array2<f64>, String> {
    let mut p = pseudo_inverse(value)?;
    for (i, mut row) in p.rows_mut().into_iter().enumerate() {
        row *= gain[i] / final_gain[i];
    }
    Ok(p)
}

/// The cosine between a head's output-value circuit `O V` and the copy rule's `P V`.
pub fn copy_alignment(output: &Array2<f64>, value: &Array2<f64>, prediction: &Array2<f64>) -> f64 {
    let (a, b) = (output.dot(value), prediction.dot(value));
    let norm = ((&a * &a).sum() * (&b * &b).sum()).sqrt();
    if norm > 0.0 { (&a * &b).sum() / norm } else { 0.0 }
}

/// The copy rule's scale for a head: the least-squares `λ` in `O V ≈ λ P V`.
pub fn copy_scale(output: &Array2<f64>, value: &Array2<f64>, prediction: &Array2<f64>) -> f64 {
    let (a, b) = (output.dot(value), prediction.dot(value));
    let own = (&b * &b).sum();
    if own > 0.0 { (&a * &b).sum() / own } else { 0.0 }
}

/// `program` with operator `name`'s matrix replaced by `values` (its interfaces kept, exact on the
/// lattice that holds the new values).
pub fn with_operator(program: &OperatorProgram, name: &str, values: Array2<f64>) -> Result<OperatorProgram, String> {
    let at = program.operators.iter().position(|o| o.name == name).ok_or(format!("no operator {name}"))?;
    let old = &program.operators[at];
    let precision = exact_precision(values.iter().copied()).map_err(|e| e.to_string())?;
    let replaced = Operator::dense(name, old.rows.clone(), old.cols.clone(), values, precision, Provenance::native(name)).map_err(|e| e.to_string())?;
    let mut out = program.clone();
    out.operators[at] = Arc::new(replaced);
    Ok(out)
}
