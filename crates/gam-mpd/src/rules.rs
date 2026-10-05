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
//! accounts for, before anything is priced, and [`match_energy`] how much of it a match on given
//! content planes explains; [`match_scale`] and [`copy_scale`] are the bindings'
//! least-squares scales.

use gam_linalg::decompose::{eigh, pseudo_inverse};
use gam_linalg::roundoff::SymmetricAssembly;
use ndarray::{Array1, Array2, Axis};

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

/// The copy rule's output head (`d × width`, unscaled): `diag(g / g_f) V⁺`, `V` the head's value
/// operator (`width × d`), `g` its layer's norm gain and `g_f` the final norm's.
pub fn copy_prediction(value: &Array2<f64>, gain: &Array1<f64>, final_gain: &Array1<f64>) -> Result<Array2<f64>, String> {
    let mut p = pseudo_inverse(value.view()).map_err(|e| e.to_string())?;
    for (i, mut row) in p.rows_mut().into_iter().enumerate() {
        row *= gain[i] / final_gain[i];
    }
    Ok(p)
}

