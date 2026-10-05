//! An MLP accounted for by explicit rules between amplitudes (#2951).
//!
//! An MLP `y = W_out σ(W_in h)` (σ the tanh GELU) is accounted for by `m` read directions (the
//! rows `v_c` of `reads`), which give its incoming amplitudes `a_c = v_c · h`; one rule per
//! outgoing amplitude, a small function of a few incoming amplitudes `a_S`,
//!
//! ```text
//!   b̂_j = β_j + ℓ_jᵀ a_{S_j} + Σ_u c_u σ(w_uᵀ a_{S_j} + d_u),
//! ```
//!
//! and `K` write directions (the rows `p_j` of `writes`): `ŷ = Σ_j b̂_j p_j + μ`. The hidden layer
//! appears nowhere. The dense map is the account whose reads are the rows of `W_in`, whose rules
//! are `σ(a_k)` one per neuron and whose writes are the columns of `W_out` ([`Account::neurons`]).
//! `proposals` turns an account into a rule of the artifact, priced and judged there.

use super::operator_program::Law;
use gam_linalg::faer_ndarray::{fast_ab, fast_abt};
use ndarray::{Array1, Array2, ArrayView2, Zip, s};
use serde::{Deserialize, Serialize};

/// Rows per pass over the inputs, so no pass holds more than a chunk's hidden layer.
const CHUNK: usize = 4096;

/// One nonlinear unit of a rule: `c · σ(wᵀa + d)`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Unit {
    pub w: Vec<f64>,
    pub d: f64,
    pub c: f64,
}

/// One outgoing amplitude's rule (module note).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Rule {
    /// The incoming amplitudes it reads, as rows of the account's reads.
    pub inputs: Vec<usize>,
    pub beta: f64,
    /// One per input.
    pub linear: Vec<f64>,
    /// Each unit's `w` has one weight per input.
    pub units: Vec<Unit>,
}

fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

impl Rule {
    /// Its value at incoming amplitudes `x` (in the order of `inputs`).
    pub fn value(&self, x: &[f64]) -> f64 {
        let mut b = self.beta + dot(&self.linear, x);
        for u in &self.units {
            b += u.c * Law::GeluTanh.apply(dot(&u.w, x) + u.d);
        }
        b
    }
}

/// An MLP's account (module note).
#[derive(Clone, Debug)]
pub struct Account {
    /// `m × d_in`: the read directions.
    pub reads: Array2<f64>,
    /// One per outgoing amplitude.
    pub rules: Vec<Rule>,
    /// `K × d_out`: the write directions.
    pub writes: Array2<f64>,
    /// `d_out`: the output offset `μ`.
    pub offset: Array1<f64>,
}

impl Account {
    /// The dense map as an account: neuron `k` reads row `k` of `w_in`, its rule is `σ(a_k)`, and it
    /// writes column `k` of `w_out`.
    pub fn neurons(w_in: &Array2<f64>, w_out: &Array2<f64>) -> Self {
        let rules = (0..w_in.nrows())
            .map(|k| Rule { inputs: vec![k], beta: 0.0, linear: vec![0.0], units: vec![Unit { w: vec![1.0], d: 0.0, c: 1.0 }] })
            .collect();
        Account { reads: w_in.clone(), rules, writes: w_out.t().to_owned(), offset: Array1::zeros(w_out.nrows()) }
    }

    /// The incoming amplitudes at inputs `h` (rows), inputs × m.
    pub fn amplitudes(&self, h: ArrayView2<f64>) -> Array2<f64> {
        fast_abt(&h, &self.reads)
    }

    /// The outgoing amplitudes the rules give at incoming amplitudes `a`, inputs × K.
    pub fn outgoing(&self, a: &Array2<f64>) -> Array2<f64> {
        let mut b = Array2::<f64>::zeros((a.nrows(), self.rules.len()));
        Zip::from(b.rows_mut()).and(a.rows()).par_for_each(|mut row, input| {
            let mut x = Vec::new();
            for (j, rule) in self.rules.iter().enumerate() {
                x.clear();
                x.extend(rule.inputs.iter().map(|c| input[*c]));
                row[j] = rule.value(&x);
            }
        });
        b
    }

    /// The account's output at inputs `h`.
    pub fn apply(&self, h: ArrayView2<f64>) -> Array2<f64> {
        let mut y = Array2::<f64>::zeros((h.nrows(), self.writes.ncols()));
        for start in (0..h.nrows()).step_by(CHUNK) {
            let end = (start + CHUNK).min(h.nrows());
            let b = self.outgoing(&self.amplitudes(h.slice(s![start..end, ..])));
            let mut part = fast_ab(&b, &self.writes);
            part += &self.offset;
            y.slice_mut(s![start..end, ..]).assign(&part);
        }
        y
    }
}
