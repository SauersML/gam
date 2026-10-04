//! An MLP accounted for by explicit rules between its subcomponents' amplitudes (#2951).
//!
//! # The account
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
//! appears nowhere. A library factors the MLP as `W_in = U Vᵀ`, `W_out = P Qᵀ`, so the map between
//! its amplitudes is `a ↦ Qᵀσ(U a)`; the rules replace that map, and with it the factors `U` and `Q`.
//! The dense map is the account whose reads are the rows of `W_in`, whose rules are `σ(a_k)` one per
//! neuron and whose writes are the columns of `W_out` ([`Account::neurons`]).
//!
//! # Its length
//!
//! Every independent real is a literal of [`LITERAL_BITS`]: the read and write directions, each
//! rule's coefficients and the output offset. A rule also names its inputs among the `m`
//! (`log₂ C(m, |S|)` bits) and its sizes (`|S|` and the number of units, prefix integers). What the
//! account leaves is the remainder `e = y − ŷ` at the MLP's own output, priced as the one total
//! prices any error there, `n/(2 ln 2) · E_t ‖e_t‖²_F` bits for `n` observations, `F` the output's
//! mean written Fisher. The plain factorization `U Vᵀ`, `P Qᵀ` and the dense map are accounts with
//! no remainder ([`plain_bits`], [`dense_bits`]).
//!
//! Rules that differ only by gauge share one body ([`share`]): an outgoing amplitude's scale is
//! its write direction's (`p_j b̂_j = (s p_j)(b̂_j / s)`), its constant is the offset's, and the
//! scale of an incoming amplitude only one rule reads is its read direction's. After those moves a
//! rule is its structure (inputs, units) and the shape left over; rules of one structure whose
//! shapes agree once rounded share one stored shape, each instance then naming its body.
//!
//! # The fit
//!
//! * **Rules** ([`grow`]). Each outgoing amplitude's rule is fitted to a target amplitude (a
//!   library's `Qᵀσ(W_in h)`) by `gates::fit_to`, its inputs grown one at a time, the one whose
//!   linear term the residual favours most, while the rule's bits plus its error on held-out
//!   inputs fall (in-sample error over `n` observations would pay for fitting noise). The error of
//!   amplitude `j` costs what the one total charges it at the written side, `n/2 · p_jᵀ F p_j`
//!   nats per unit² of mean squared error.
//! * **Writes** ([`refit_writes`]). Given the rules, the writes and offset minimise the remainder:
//!   least squares of `y` on the centred outgoing amplitudes (one Fisher for every input factors
//!   out), its Gram's eigenvalues below its rounding band (`λ_max N 2⁻⁵²`) dropped.
//! * **Pruning** ([`prune`]). An outgoing amplitude whose rule saves less remainder than its write
//!   direction and rule cost is dropped, its mean folded into the offset; then any incoming
//!   amplitude no rule reads. Each batch of drops (every amplitude predicted to pay, from the
//!   remainder's exact change with the others fixed) is kept when the refitted total on held-out
//!   inputs falls and halved otherwise. Every comparison of totals is made on held-out inputs: a
//!   fit's own inputs, their error extrapolated to `n` observations, would pay for fitting noise.
//! * **Refinement** ([`refine`]). With the structure fixed, every real of the account moves
//!   together to lower the remainder on the training inputs: L-BFGS on its exact gradient, started
//!   from the inverse of the Gauss–Newton matrix's diagonal (the reals differ in scale by orders of
//!   magnitude), Armijo backtracking, keeping the point that is best on held-out inputs.

use super::codec::prefix_integer_len_bits;
use gam_linalg::decompose::eigh;
use super::gates::{self, Feature, Switch, Targets, Unit, gelu, gelu_derivative};
use gam_linalg::faer_ndarray::{fast_ab, fast_abt, fast_atb};
use gam_linalg::roundoff::SymmetricAssembly;
use ndarray::{Array1, Array2, ArrayView2, Axis, Zip, s};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use statrs::function::gamma::ln_gamma;
use std::collections::BTreeMap;
use std::f64::consts::LN_2;

/// The bits of one independent real.
pub const LITERAL_BITS: f64 = 32.0;

/// Rows per pass over the inputs, so no pass holds more than a chunk's hidden layer.
const CHUNK: usize = 4096;

fn count_bits(value: usize) -> f64 {
    prefix_integer_len_bits(value as u64 + 1).map_or(f64::INFINITY, |b| b as f64)
}

fn log2_binomial(n: usize, k: usize) -> f64 {
    if k > n {
        return f64::INFINITY;
    }
    (ln_gamma(n as f64 + 1.0) - ln_gamma(k as f64 + 1.0) - ln_gamma((n - k) as f64 + 1.0)) / LN_2
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

impl Rule {
    fn from_switch(switch: &Switch) -> Self {
        Rule {
            inputs: switch.features.iter().map(|f| f.piece).collect(),
            beta: switch.beta,
            linear: switch.linear.clone(),
            units: switch.units.clone(),
        }
    }

    /// Its value at incoming amplitudes `x` (in the order of `inputs`).
    pub fn value(&self, x: &[f64]) -> f64 {
        let mut b = self.beta + dot(&self.linear, x);
        for u in &self.units {
            b += u.c * gelu(dot(&u.w, x) + u.d);
        }
        b
    }

    /// Its reals: `β`, one linear coefficient per input, and per unit its weights, offset and scale.
    pub fn literals(&self) -> usize {
        let d = self.inputs.len();
        1 + d + self.units.len() * (d + 2)
    }

    /// Its inputs named among `pool` incoming amplitudes, and its sizes.
    pub fn pointer_bits(&self, pool: usize) -> f64 {
        log2_binomial(pool, self.inputs.len()) + count_bits(self.inputs.len()) + count_bits(self.units.len())
    }

    fn pack(&self, out: &mut Vec<f64>) {
        out.push(self.beta);
        out.extend_from_slice(&self.linear);
        for u in &self.units {
            out.extend_from_slice(&u.w);
            out.push(u.d);
            out.push(u.c);
        }
    }

    fn unpack(&mut self, theta: &[f64]) {
        let d = self.inputs.len();
        self.beta = theta[0];
        self.linear.copy_from_slice(&theta[1..1 + d]);
        for (k, u) in self.units.iter_mut().enumerate() {
            let o = 1 + d + k * (d + 2);
            u.w.copy_from_slice(&theta[o..o + d]);
            u.d = theta[o + d];
            u.c = theta[o + d + 1];
        }
    }

    /// Adds `g ∂b̂/∂θ` to `grad` (this rule's slice, packed as [`Rule::pack`]) and `g ∂b̂/∂x` to `gx`.
    fn backward(&self, x: &[f64], g: f64, grad: &mut [f64], gx: &mut [f64]) {
        let d = self.inputs.len();
        grad[0] += g;
        for k in 0..d {
            grad[1 + k] += g * x[k];
            gx[k] += g * self.linear[k];
        }
        for (k, u) in self.units.iter().enumerate() {
            let o = 1 + d + k * (d + 2);
            let pre = dot(&u.w, x) + u.d;
            let gs = g * u.c * gelu_derivative(pre);
            for i in 0..d {
                grad[o + i] += gs * x[i];
                gx[i] += gs * u.w[i];
            }
            grad[o + d] += gs;
            grad[o + d + 1] += g * gelu(pre);
        }
    }

    /// The formula in terms of `a[i]`, coefficients to three significant figures.
    pub fn formula(&self) -> String {
        let name = |k: usize| format!("a{}", self.inputs[k]);
        let affine = |w: &[f64], d: f64| {
            let mut s = String::new();
            for (k, wk) in w.iter().enumerate() {
                if *wk != 0.0 {
                    s.push_str(&format!("{}{:.3}·{}", if s.is_empty() { "" } else { " + " }, wk, name(k)));
                }
            }
            if d != 0.0 || s.is_empty() {
                s.push_str(&format!("{}{:.3}", if s.is_empty() { "" } else { " + " }, d));
            }
            s
        };
        let mut terms = Vec::new();
        if self.linear.iter().any(|l| *l != 0.0) || self.beta != 0.0 {
            terms.push(affine(&self.linear, self.beta));
        }
        for u in &self.units {
            terms.push(format!("{:.3}·σ({})", u.c, affine(&u.w, u.d)));
        }
        if terms.is_empty() { "0".to_string() } else { terms.join(" + ") }
    }
}

fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
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

/// The MLP `y = W_out σ(W_in h)` on inputs `h` (rows), `w_in` hidden × d_in and `w_out` d_out × hidden.
pub fn native(w_in: &Array2<f64>, w_out: &Array2<f64>, h: ArrayView2<f64>) -> Array2<f64> {
    let mut y = Array2::<f64>::zeros((h.nrows(), w_out.nrows()));
    for start in (0..h.nrows()).step_by(CHUNK) {
        let end = (start + CHUNK).min(h.nrows());
        let hidden = fast_abt(&h.slice(s![start..end, ..]), w_in).mapv(gelu);
        y.slice_mut(s![start..end, ..]).assign(&fast_abt(&hidden, w_out));
    }
    y
}

/// The literal bits of the plain factorization of the MLP by `m` input and `k` output subcomponents.
pub fn plain_bits(d_in: usize, hidden: usize, d_out: usize, m: usize, k: usize) -> f64 {
    (m * (d_in + hidden) + k * (hidden + d_out)) as f64 * LITERAL_BITS
}

/// The literal bits of the dense map `W_in`, `W_out`.
pub fn dense_bits(d_in: usize, hidden: usize, d_out: usize) -> f64 {
    (hidden * (d_in + d_out)) as f64 * LITERAL_BITS
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

    pub fn literals(&self) -> usize {
        self.reads.len() + self.rules.iter().map(Rule::literals).sum::<usize>() + self.writes.len() + self.offset.len()
    }

    pub fn pointer_bits(&self) -> f64 {
        let m = self.reads.nrows();
        self.rules.iter().map(|r| r.pointer_bits(m)).sum::<f64>() + count_bits(m) + count_bits(self.rules.len())
    }

    /// Literals and pointers.
    pub fn structure_bits(&self) -> f64 {
        self.literals() as f64 * LITERAL_BITS + self.pointer_bits()
    }

    /// The remainder in nats per input to second order, `½ E_t ‖y_t − ŷ_t‖²_F`.
    pub fn remainder(&self, h: ArrayView2<f64>, y: ArrayView2<f64>, fisher: &Array2<f64>) -> f64 {
        let e = &y - &self.apply(h);
        let fe = fast_ab(&e, fisher);
        0.5 * (&e * &fe).sum() / h.nrows().max(1) as f64
    }

    /// Structure bits plus the remainder over `observations` inputs.
    pub fn total_bits(&self, h: ArrayView2<f64>, y: ArrayView2<f64>, fisher: &Array2<f64>, observations: f64) -> f64 {
        self.structure_bits() + observations * self.remainder(h, y, fisher) / LN_2
    }

    /// The account without outgoing amplitudes `dropped`, each one's mean (at incoming amplitudes
    /// `a`) folded into the offset, and without the incoming amplitudes no rule then reads.
    fn without(&self, dropped: &[usize], means: &Array1<f64>) -> Account {
        let mut gone = vec![false; self.rules.len()];
        for j in dropped {
            gone[*j] = true;
        }
        let mut offset = self.offset.clone();
        for j in dropped {
            offset.scaled_add(means[*j], &self.writes.row(*j));
        }
        let kept: Vec<usize> = (0..self.rules.len()).filter(|j| !gone[*j]).collect();
        let mut account = Account {
            reads: self.reads.clone(),
            rules: kept.iter().map(|j| self.rules[*j].clone()).collect(),
            writes: self.writes.select(Axis(0), &kept),
            offset,
        };
        account.drop_unread();
        account
    }

    /// Removes the read directions no rule reads.
    pub fn drop_unread(&mut self) {
        let mut used = vec![false; self.reads.nrows()];
        for r in &self.rules {
            for c in &r.inputs {
                used[*c] = true;
            }
        }
        let kept: Vec<usize> = (0..used.len()).filter(|c| used[*c]).collect();
        if kept.len() == used.len() {
            return;
        }
        let mut index = vec![usize::MAX; used.len()];
        for (new, old) in kept.iter().enumerate() {
            index[*old] = new;
        }
        self.reads = self.reads.select(Axis(0), &kept);
        for r in &mut self.rules {
            for c in &mut r.inputs {
                *c = index[*c];
            }
        }
    }

    fn parameters(&self) -> usize {
        self.literals()
    }

    fn pack(&self) -> Vec<f64> {
        let mut theta = Vec::with_capacity(self.parameters());
        theta.extend(self.reads.iter());
        for r in &self.rules {
            r.pack(&mut theta);
        }
        theta.extend(self.writes.iter());
        theta.extend(self.offset.iter());
        theta
    }

    fn unpack(&mut self, theta: &[f64]) {
        let mut o = self.reads.len();
        self.reads.iter_mut().zip(&theta[..o]).for_each(|(v, t)| *v = *t);
        for r in &mut self.rules {
            let n = r.literals();
            r.unpack(&theta[o..o + n]);
            o += n;
        }
        let w = self.writes.len();
        self.writes.iter_mut().zip(&theta[o..o + w]).for_each(|(v, t)| *v = *t);
        o += w;
        self.offset.iter_mut().zip(&theta[o..]).for_each(|(v, t)| *v = *t);
    }

    /// The remainder (nats per input) at inputs `h` with targets `y` and its gradient in the packed
    /// reals ([`Account::pack`]).
    fn remainder_gradient(&self, h: ArrayView2<f64>, y: ArrayView2<f64>, fisher: &Array2<f64>) -> (f64, Vec<f64>) {
        let rows = h.nrows();
        let offsets: Vec<usize> = self
            .rules
            .iter()
            .scan(0usize, |o, r| {
                let at = *o;
                *o += r.literals();
                Some(at)
            })
            .collect();
        let rule_reals: usize = self.rules.iter().map(Rule::literals).sum();
        let mut g_reads = Array2::<f64>::zeros(self.reads.dim());
        let mut g_rules = vec![0.0; rule_reals];
        let mut g_writes = Array2::<f64>::zeros(self.writes.dim());
        let mut g_offset = Array1::<f64>::zeros(self.offset.len());
        let mut loss = 0.0;
        for start in (0..rows).step_by(CHUNK) {
            let end = (start + CHUNK).min(rows);
            let hc = h.slice(s![start..end, ..]);
            let a = self.amplitudes(hc);
            let b = self.outgoing(&a);
            let mut e = y.slice(s![start..end, ..]).to_owned() - fast_ab(&b, &self.writes);
            e -= &self.offset;
            let fe = fast_ab(&e, fisher);
            loss += 0.5 * (&e * &fe).sum();
            // ∂/∂ŷ of ½ eᵀFe is −Fe.
            let gy = -fe;
            g_writes += &fast_atb(&b, &gy);
            g_offset += &gy.sum_axis(Axis(0));
            let gb = fast_abt(&gy, &self.writes);
            // Each worker takes a run of rows: its rows of ∂/∂a, its own sum of the rules' gradients.
            let threads = rayon::current_num_threads().max(1);
            let per = (end - start).div_ceil(threads);
            let mut ga = Array2::<f64>::zeros(a.dim());
            let parts: Vec<Vec<f64>> = ga
                .axis_chunks_iter_mut(Axis(0), per.max(1))
                .into_par_iter()
                .enumerate()
                .map(|(block, mut ga_block)| {
                    let mut grad = vec![0.0; rule_reals];
                    let (mut x, mut gx) = (Vec::new(), Vec::new());
                    for (i, mut ga_row) in ga_block.rows_mut().into_iter().enumerate() {
                        let t = block * per.max(1) + i;
                        for (j, rule) in self.rules.iter().enumerate() {
                            let g = gb[[t, j]];
                            if g == 0.0 || (rule.inputs.is_empty() && rule.units.is_empty()) {
                                grad[offsets[j]] += g;
                                continue;
                            }
                            x.clear();
                            x.extend(rule.inputs.iter().map(|c| a[[t, *c]]));
                            gx.clear();
                            gx.resize(x.len(), 0.0);
                            let n = rule.literals();
                            rule.backward(&x, g, &mut grad[offsets[j]..offsets[j] + n], &mut gx);
                            for (k, c) in rule.inputs.iter().enumerate() {
                                ga_row[*c] += gx[k];
                            }
                        }
                    }
                    grad
                })
                .collect();
            for part in parts {
                g_rules.iter_mut().zip(&part).for_each(|(g, p)| *g += p);
            }
            g_reads += &fast_atb(&ga, &hc);
        }
        let scale = 1.0 / rows.max(1) as f64;
        let mut grad = Vec::with_capacity(self.parameters());
        grad.extend(g_reads.iter().map(|g| g * scale));
        grad.extend(g_rules.iter().map(|g| g * scale));
        grad.extend(g_writes.iter().map(|g| g * scale));
        grad.extend(g_offset.iter().map(|g| g * scale));
        (loss * scale, grad)
    }
}

impl Account {
    /// The diagonal of the remainder's Gauss–Newton matrix in the packed reals, cross terms between
    /// outgoing amplitudes left out of the reads' entries: per write `E[b̂_j²] F_oo`, per offset
    /// entry `F_oo`, per rule real `p_jᵀ F p_j E[(∂b̂_j/∂θ)²]`, per read entry `E[h_i² Σ_j p_jᵀ F p_j
    /// (∂b̂_j/∂a_c)²]`.
    fn curvature(&self, h: ArrayView2<f64>, fisher: &Array2<f64>) -> Vec<f64> {
        let rows = h.nrows().max(1) as f64;
        let offsets: Vec<usize> = self
            .rules
            .iter()
            .scan(0usize, |o, r| {
                let at = *o;
                *o += r.literals();
                Some(at)
            })
            .collect();
        let rule_reals: usize = self.rules.iter().map(Rule::literals).sum();
        let pfp = fast_abt(&fast_ab(&self.writes, fisher), &self.writes).diag().to_owned();
        let mut d_reads = Array2::<f64>::zeros(self.reads.dim());
        let mut d_rules = vec![0.0; rule_reals];
        let mut squares = Array1::<f64>::zeros(self.rules.len());
        for start in (0..h.nrows()).step_by(CHUNK) {
            let end = (start + CHUNK).min(h.nrows());
            let hc = h.slice(s![start..end, ..]);
            let a = self.amplitudes(hc);
            let b = self.outgoing(&a);
            squares += &(&b * &b).sum_axis(Axis(0));
            let threads = rayon::current_num_threads().max(1);
            let per = (end - start).div_ceil(threads).max(1);
            let mut sens = Array2::<f64>::zeros(a.dim());
            let parts: Vec<Vec<f64>> = sens
                .axis_chunks_iter_mut(Axis(0), per)
                .into_par_iter()
                .enumerate()
                .map(|(block, mut sens_block)| {
                    let mut acc = vec![0.0; rule_reals];
                    let (mut x, mut gx, mut grad) = (Vec::new(), Vec::new(), Vec::new());
                    for (i, mut row) in sens_block.rows_mut().into_iter().enumerate() {
                        let t = block * per + i;
                        for (j, rule) in self.rules.iter().enumerate() {
                            x.clear();
                            x.extend(rule.inputs.iter().map(|c| a[[t, *c]]));
                            gx.clear();
                            gx.resize(x.len(), 0.0);
                            grad.clear();
                            grad.resize(rule.literals(), 0.0);
                            rule.backward(&x, 1.0, &mut grad, &mut gx);
                            for (k, g) in grad.iter().enumerate() {
                                acc[offsets[j] + k] += pfp[j] * g * g;
                            }
                            for (k, c) in rule.inputs.iter().enumerate() {
                                row[*c] += pfp[j] * gx[k] * gx[k];
                            }
                        }
                    }
                    acc
                })
                .collect();
            for part in parts {
                d_rules.iter_mut().zip(&part).for_each(|(d, p)| *d += p);
            }
            d_reads += &fast_atb(&sens, &hc.mapv(|v| v * v));
        }
        let mut diagonal = Vec::with_capacity(self.parameters());
        diagonal.extend(d_reads.iter().map(|d| d / rows));
        diagonal.extend(d_rules.iter().map(|d| d / rows));
        for j in 0..self.rules.len() {
            diagonal.extend(fisher.diag().iter().map(|f| squares[j] / rows * f));
        }
        diagonal.extend(fisher.diag().iter());
        diagonal
    }
}

/// Each outgoing amplitude's rule (module note): `fit` the incoming amplitudes (inputs × m) and
/// the outgoing ones (inputs × K) the rules are fitted on, `check` other inputs, the same two
/// matrices, on which each step's error is measured, and amplitude `j`'s mean squared error
/// costing `prices[j]` nats (`n/2 · p_jᵀ F p_j` for `n` observations).
pub fn grow(fit: (&Array2<f64>, &Array2<f64>), check: (&Array2<f64>, &Array2<f64>), prices: &[f64]) -> Vec<Rule> {
    let norms = fit.0.map_axis(Axis(0), |c| c.dot(&c));
    // The dearest amplitudes grow the largest rules: start them first.
    let mut order: Vec<usize> = (0..fit.1.ncols()).collect();
    order.sort_by(|p, q| prices[*q].total_cmp(&prices[*p]));
    let grown: Vec<(usize, Rule)> = order
        .into_par_iter()
        .with_max_len(1)
        .map(|j| {
            let target = fit.1.column(j).to_vec();
            let held = check.1.column(j).to_vec();
            (j, Rule::from_switch(&grow_one((fit.0, &target), (check.0, &held), &norms, prices[j])))
        })
        .collect();
    let mut rules = vec![Rule { inputs: Vec::new(), beta: 0.0, linear: Vec::new(), units: Vec::new() }; grown.len()];
    for (j, rule) in grown {
        rules[j] = rule;
    }
    rules
}

/// A switch's error bits on the check inputs at `price` nats per unit² of mean squared error.
fn held_out_bits(switch: &Switch, a: &Array2<f64>, target: &[f64], price: f64) -> f64 {
    let pieces: Vec<usize> = switch.features.iter().map(|f| f.piece).collect();
    let mut x = vec![0.0; pieces.len()];
    let squares: f64 = (0..a.nrows())
        .map(|t| {
            for (k, c) in pieces.iter().enumerate() {
                x[k] = a[[t, *c]];
            }
            let r = switch.logit(&x) - target[t];
            r * r
        })
        .sum();
    price * squares / a.nrows().max(1) as f64 / LN_2
}

/// One amplitude's rule, its inputs grown one at a time (each fitted on `fit`) while its own bits
/// plus its error on `check` fall.
fn grow_one(fit: (&Array2<f64>, &[f64]), check: (&Array2<f64>, &[f64]), norms: &Array1<f64>, price: f64) -> Switch {
    let (a, target) = fit;
    let y = Targets::Values { values: target, weight: price / a.nrows().max(1) as f64 };
    let pool = a.ncols();
    let mut chosen: Vec<usize> = Vec::new();
    let mut current = gates::constant(y);
    let mut current_total = current.function_bits + held_out_bits(&current, check.0, check.1, price);
    let mut x = Vec::new();
    loop {
        let residual: Array1<f64> = (0..a.nrows())
            .map(|t| {
                x.clear();
                x.extend(chosen.iter().map(|c| a[[t, *c]]));
                target[t] - current.logit(&x)
            })
            .collect();
        let score = a.t().dot(&residual);
        let Some(next) = (0..pool)
            .filter(|c| !chosen.contains(c) && norms[*c] > 0.0)
            .max_by(|p, q| (score[*p] * score[*p] / norms[*p]).total_cmp(&(score[*q] * score[*q] / norms[*q])))
        else {
            return current;
        };
        let mut trial_set = chosen.clone();
        trial_set.push(next);
        let features: Vec<Feature> = trial_set.iter().map(|c| Feature { site: 0, piece: *c, lag: 0, magnitude: false }).collect();
        let start = (!chosen.is_empty()).then_some(&current);
        let trial = gates::fit_to(a.select(Axis(1), &trial_set).view(), y, &features, log2_binomial(pool, trial_set.len()), start);
        let trial_total = trial.function_bits + held_out_bits(&trial, check.0, check.1, price);
        if trial_total >= current_total {
            return current;
        }
        (chosen, current, current_total) = (trial_set, trial, trial_total);
    }
}

/// Every outgoing amplitude's rule grown again ([`grow`]) on the reads as they now are, to the
/// target the rest of the account leaves it, `b̂_j + p_jᵀ F e / p_jᵀ F p_j` (the remainder's
/// minimiser in `b̂_j` alone), fitted on the inputs `rows` of `train` and checked on `check`; a
/// grown rule replaces the old one where its literals, pointers and error on the check inputs take
/// fewer bits. The writes are refitted on all of `train`, and the change is kept only when the
/// total over `observations` on `check` falls. Returns the number of rules replaced.
pub fn regrow(
    account: &mut Account,
    train: (ArrayView2<f64>, ArrayView2<f64>),
    check: (ArrayView2<f64>, ArrayView2<f64>),
    fisher: &Array2<f64>,
    observations: f64,
    rows: &[usize],
) -> Result<usize, String> {
    let before = account.total_bits(check.0, check.1, fisher, observations);
    let pfp = fast_abt(&fast_ab(&account.writes, fisher), &account.writes).diag().to_owned();
    // The incoming amplitudes and each rule's target at inputs `h` with MLP outputs `y`.
    let implied = |h: ArrayView2<f64>, y: ArrayView2<f64>| {
        let a = account.amplitudes(h);
        let b = account.outgoing(&a);
        let mut e = &y - &fast_ab(&b, &account.writes);
        e -= &account.offset;
        let pfe = fast_abt(&fast_ab(&e, fisher), &account.writes);
        let mut targets = b;
        for (j, mut column) in targets.columns_mut().into_iter().enumerate() {
            if pfp[j] > 0.0 {
                column.scaled_add(1.0 / pfp[j], &pfe.column(j));
            }
        }
        (a, targets)
    };
    let (hs, ys) = (train.0.select(Axis(0), rows), train.1.select(Axis(0), rows));
    let (a_fit, t_fit) = implied(hs.view(), ys.view());
    let (a_check, t_check) = implied(check.0, check.1);
    let prices: Vec<f64> = pfp.iter().map(|w| 0.5 * observations * w).collect();
    let grown = grow((&a_fit, &t_fit), (&a_check, &t_check), &prices);
    let m = account.reads.nrows();
    let cost = |rule: &Rule, j: usize| -> f64 {
        let mut x = Vec::new();
        let squares: f64 = (0..a_check.nrows())
            .map(|t| {
                x.clear();
                x.extend(rule.inputs.iter().map(|c| a_check[[t, *c]]));
                let r = rule.value(&x) - t_check[[t, j]];
                r * r
            })
            .sum();
        rule.literals() as f64 * LITERAL_BITS + rule.pointer_bits(m) + prices[j] * squares / a_check.nrows().max(1) as f64 / LN_2
    };
    let mut trial = account.clone();
    let mut replaced = 0;
    for (j, rule) in grown.into_iter().enumerate() {
        if cost(&rule, j) < cost(&account.rules[j], j) {
            trial.rules[j] = rule;
            replaced += 1;
        }
    }
    refit_writes(&mut trial, train.0, train.1)?;
    let after = trial.total_bits(check.0, check.1, fisher, observations);
    log::info!("regrow: {replaced} rules replaced, total {after:.0} vs {before:.0}");
    if after < before {
        *account = trial;
        Ok(replaced)
    } else {
        Ok(0)
    }
}

/// The writes and offset that minimise the remainder for the account's rules at inputs `h` with
/// targets `y` (module note).
pub fn refit_writes(account: &mut Account, h: ArrayView2<f64>, y: ArrayView2<f64>) -> Result<(), String> {
    let rows = h.nrows();
    let k = account.rules.len();
    if k == 0 {
        account.writes = Array2::zeros((0, y.ncols()));
        account.offset = y.mean_axis(Axis(0)).unwrap_or_else(|| Array1::zeros(y.ncols()));
        return Ok(());
    }
    let mut gram = Array2::<f64>::zeros((k, k));
    let mut cross = Array2::<f64>::zeros((k, y.ncols()));
    let mut b_sum = Array1::<f64>::zeros(k);
    let mut y_sum = Array1::<f64>::zeros(y.ncols());
    for start in (0..rows).step_by(CHUNK) {
        let end = (start + CHUNK).min(rows);
        let b = account.outgoing(&account.amplitudes(h.slice(s![start..end, ..])));
        let yc = y.slice(s![start..end, ..]);
        gram += &fast_atb(&b, &b);
        cross += &fast_atb(&b, &yc);
        b_sum += &b.sum_axis(Axis(0));
        y_sum += &yc.sum_axis(Axis(0));
    }
    let n = rows.max(1) as f64;
    let (b_mean, y_mean) = (b_sum / n, y_sum / n);
    // Centre: Σ (b − b̄)(b − b̄)ᵀ = Σ b bᵀ − n b̄ b̄ᵀ, and likewise the cross moment.
    for i in 0..k {
        for j in 0..k {
            gram[[i, j]] -= n * b_mean[i] * b_mean[j];
        }
        for o in 0..y.ncols() {
            cross[[i, o]] -= n * b_mean[i] * y_mean[o];
        }
    }
    let symmetric = (&gram + &gram.t()) * 0.5;
    let d = eigh(symmetric.view(), SymmetricAssembly::Mirrored, None).map_err(|e| format!("{e:?}"))?;
    let top = d.values.iter().fold(0.0f64, |m, v| m.max(*v));
    let floor = top * n * f64::EPSILON;
    let rotated = fast_atb(&d.vectors, &cross);
    let mut scaled = rotated;
    for (i, mut row) in scaled.rows_mut().into_iter().enumerate() {
        let l = d.values[i];
        row *= if l > floor { 1.0 / l } else { 0.0 };
    }
    account.writes = fast_ab(&d.vectors, &scaled);
    account.offset = &y_mean - &account.writes.t().dot(&b_mean);
    Ok(())
}

/// What [`prune`] removed.
#[derive(Clone, Copy, Debug, Default, Serialize)]
pub struct Pruned {
    pub outgoing: usize,
    pub incoming: usize,
}

/// Drops outgoing amplitudes, then incoming ones, while the total over `observations` on the
/// `check` inputs falls (module note); candidates are scored and the writes refitted on `train`.
pub fn prune(
    account: &mut Account,
    train: (ArrayView2<f64>, ArrayView2<f64>),
    check: (ArrayView2<f64>, ArrayView2<f64>),
    fisher: &Array2<f64>,
    observations: f64,
) -> Result<Pruned, String> {
    let (h, y) = train;
    let (k0, m0) = (account.rules.len(), account.reads.nrows());
    let rows = h.nrows().max(1) as f64;
    let mut total = account.total_bits(check.0, check.1, fisher, observations);
    loop {
        let k = account.rules.len();
        // Per outgoing amplitude: its mean, and the remainder's exact change were it alone dropped
        // (its centred contribution added back to the residual), `E[(b̂_j − b̄_j) p_jᵀF e] + ½
        // Var(b̂_j) p_jᵀ F p_j`.
        let mut means = Array1::<f64>::zeros(k);
        let mut squares = Array1::<f64>::zeros(k);
        let mut cross = Array1::<f64>::zeros(k);
        let mut pulls = Array1::<f64>::zeros(k);
        for start in (0..h.nrows()).step_by(CHUNK) {
            let end = (start + CHUNK).min(h.nrows());
            let b = account.outgoing(&account.amplitudes(h.slice(s![start..end, ..])));
            let mut e = y.slice(s![start..end, ..]).to_owned() - fast_ab(&b, &account.writes);
            e -= &account.offset;
            let pfe = fast_abt(&fast_ab(&e, fisher), &account.writes);
            means += &b.sum_axis(Axis(0));
            squares += &(&b * &b).sum_axis(Axis(0));
            cross += &(&b * &pfe).sum_axis(Axis(0));
            pulls += &pfe.sum_axis(Axis(0));
        }
        means /= rows;
        let pfp = fast_abt(&fast_ab(&account.writes, fisher), &account.writes).diag().to_owned();
        let m = account.reads.nrows();
        let mut readers = vec![0usize; m];
        for r in &account.rules {
            for c in &r.inputs {
                readers[*c] += 1;
            }
        }
        let mut candidates: Vec<(f64, usize)> = (0..k)
            .filter_map(|j| {
                let variance = (squares[j] / rows - means[j] * means[j]).max(0.0);
                let change = (cross[j] - means[j] * pulls[j]) / rows + 0.5 * variance * pfp[j];
                // Its write direction, its rule, and every read direction only it reads.
                let own_reads = account.rules[j].inputs.iter().filter(|c| readers[**c] == 1).count();
                let saved = (account.writes.ncols() + account.rules[j].literals() + own_reads * account.reads.ncols()) as f64 * LITERAL_BITS
                    + account.rules[j].pointer_bits(m);
                let net = saved - observations * change / LN_2;
                (net > 0.0).then_some((net, j))
            })
            .collect();
        if candidates.is_empty() {
            break;
        }
        candidates.sort_by(|p, q| q.0.total_cmp(&p.0));
        let mut batch = candidates.len();
        let accepted = loop {
            let dropped: Vec<usize> = candidates[..batch].iter().map(|(_, j)| *j).collect();
            let mut trial = account.without(&dropped, &means);
            refit_writes(&mut trial, h, y)?;
            let trial_total = trial.total_bits(check.0, check.1, fisher, observations);
            log::info!("prune: drop {batch} of {k} outgoing, total {trial_total:.0} vs {total:.0}");
            if trial_total < total {
                *account = trial;
                total = trial_total;
                break true;
            }
            if batch == 1 {
                break false;
            }
            batch = batch.div_ceil(2);
        };
        if !accepted {
            break;
        }
    }
    Ok(Pruned { outgoing: k0 - account.rules.len(), incoming: m0 - account.reads.nrows() })
}

/// Where [`refine`] stopped.
#[derive(Clone, Copy, Debug, Default, Serialize)]
pub struct Refined {
    pub iterations: usize,
    /// The step whose point was kept.
    pub kept: usize,
    /// The remainder (nats per input) on the training and held-out inputs at the kept point.
    pub train: f64,
    pub valid: f64,
}

/// Moves every real of the account to lower the remainder on `train` (inputs, targets), keeping
/// the point best on `valid`; stops after `iterations` steps or `patience` steps without a better
/// held-out remainder (module note).
pub fn refine(
    account: &mut Account,
    train: (ArrayView2<f64>, ArrayView2<f64>),
    valid: (ArrayView2<f64>, ArrayView2<f64>),
    fisher: &Array2<f64>,
    iterations: usize,
    patience: usize,
) -> Refined {
    const MEMORY: usize = 8;
    let mut theta = account.pack();
    let (mut f, mut g) = account.remainder_gradient(train.0, train.1, fisher);
    let mut best = (account.remainder(valid.0, valid.1, fisher), theta.clone(), f, 0usize);
    // The initial inverse Hessian: the Gauss–Newton diagonal's inverse, each entry floored at
    // √ε of the mean (a real the remainder barely sees still moves boundedly).
    let diagonal = account.curvature(train.0, fisher);
    let mean = diagonal.iter().sum::<f64>() / diagonal.len().max(1) as f64;
    let floor = mean * f64::EPSILON.sqrt();
    let inverse: Vec<f64> = diagonal.iter().map(|d| 1.0 / d.max(floor).max(f64::MIN_POSITIVE)).collect();
    let mut history: Vec<(Vec<f64>, Vec<f64>, f64)> = Vec::new();
    let mut since = 0;
    let mut done = 0;
    for it in 0..iterations {
        // Two-loop recursion for the quasi-Newton direction from the diagonal start.
        let mut q: Vec<f64> = g.iter().map(|v| -v).collect();
        let mut alphas = Vec::with_capacity(history.len());
        for (sv, yv, rho) in history.iter().rev() {
            let alpha = rho * dot(sv, &q);
            q.iter_mut().zip(yv).for_each(|(qi, yi)| *qi -= alpha * yi);
            alphas.push(alpha);
        }
        q.iter_mut().zip(&inverse).for_each(|(v, h)| *v *= h);
        for ((sv, yv, rho), alpha) in history.iter().zip(alphas.iter().rev()) {
            let beta = rho * dot(yv, &q);
            q.iter_mut().zip(sv).for_each(|(qi, si)| *qi += (alpha - beta) * si);
        }
        let mut slope = dot(&g, &q);
        if slope >= 0.0 {
            history.clear();
            q = g.iter().zip(&inverse).map(|(v, h)| -v * h).collect();
            slope = dot(&g, &q);
        }
        // Armijo backtracking from the full step.
        let mut step = 1.0;
        let mut next = None;
        for _ in 0..40 {
            let trial: Vec<f64> = theta.iter().zip(&q).map(|(t, d)| t + step * d).collect();
            let mut moved = account.clone();
            moved.unpack(&trial);
            let value = moved.remainder(train.0, train.1, fisher);
            if value.is_finite() && value <= f + 1e-4 * step * slope {
                next = Some((trial, moved));
                break;
            }
            step *= 0.5;
        }
        let Some((trial, moved)) = next else { break };
        let (f_new, g_new) = moved.remainder_gradient(train.0, train.1, fisher);
        let sv: Vec<f64> = trial.iter().zip(&theta).map(|(a, b)| a - b).collect();
        let yv: Vec<f64> = g_new.iter().zip(&g).map(|(a, b)| a - b).collect();
        let sy = dot(&sv, &yv);
        if sy > 0.0 {
            history.push((sv, yv, 1.0 / sy));
            if history.len() > MEMORY {
                history.remove(0);
            }
        }
        (theta, f, g) = (trial, f_new, g_new);
        *account = moved;
        done = it + 1;
        let held = account.remainder(valid.0, valid.1, fisher);
        log::info!("refine {done}: train {f:.6} held-out {held:.6} (step {step:.3e})");
        if held < best.0 {
            best = (held, theta.clone(), f, done);
            since = 0;
        } else {
            since += 1;
            if since >= patience {
                break;
            }
        }
    }
    account.unpack(&best.1);
    Refined { iterations: done, kept: best.3, train: best.2, valid: best.0 }
}

/// One stored rule body and the outgoing amplitudes that use it.
#[derive(Clone, Debug, Serialize)]
pub struct Body {
    pub inputs: usize,
    pub units: usize,
    /// The body's formula over inputs `x0, x1, …` (gauge removed).
    pub formula: String,
    pub instances: Vec<usize>,
}

/// The gauge moves of the module note, exact (the account's output is unchanged): every rule's
/// constant into the offset, its scale (its first unit's, else its first linear coefficient) into
/// its write direction, and the scale of an input only it reads into that input's read direction
/// (making its first unit's weight on it, else its linear coefficient, one). Units are ordered by
/// their offsets.
pub fn canonical(account: &mut Account) {
    let mut readers = vec![0usize; account.reads.nrows()];
    for r in &account.rules {
        for c in &r.inputs {
            readers[*c] += 1;
        }
    }
    for (j, rule) in account.rules.iter_mut().enumerate() {
        account.offset.scaled_add(rule.beta, &account.writes.row(j));
        rule.beta = 0.0;
        rule.units.sort_by(|p, q| p.d.total_cmp(&q.d));
        let scale = match (rule.units.first(), rule.linear.first()) {
            (Some(u), _) if u.c != 0.0 => u.c,
            (_, Some(l)) if *l != 0.0 && rule.units.is_empty() => *l,
            _ => 1.0,
        };
        rule.linear.iter_mut().for_each(|l| *l /= scale);
        rule.units.iter_mut().for_each(|u| u.c /= scale);
        account.writes.row_mut(j).mapv_inplace(|p| p * scale);
        for (k, c) in rule.inputs.clone().iter().enumerate() {
            if readers[*c] != 1 {
                continue;
            }
            let weight = match rule.units.first() {
                Some(u) => u.w[k],
                None => rule.linear[k],
            };
            if weight == 0.0 {
                continue;
            }
            account.reads.row_mut(*c).mapv_inplace(|v| v * weight);
            rule.linear[k] /= weight;
            rule.units.iter_mut().for_each(|u| u.w[k] /= weight);
        }
    }
}

/// A canonical rule's reals after `β` (which the offset holds), in pack order.
fn shape(rule: &Rule) -> Vec<f64> {
    let mut theta = Vec::new();
    rule.pack(&mut theta);
    theta[1..].to_vec()
}

/// Rules of one structure whose shapes agree after rounding to `2^-p` share a body; per structure,
/// the precision with the fewest total bits (shared shapes' literals once, each instance naming its
/// body, plus the remainder over `observations` with the shapes rounded), or no sharing. Applies
/// the choice and returns the bodies with more than one instance.
pub fn share(account: &mut Account, h: ArrayView2<f64>, y: ArrayView2<f64>, fisher: &Array2<f64>, observations: f64) -> Vec<Body> {
    let mut classes: BTreeMap<(usize, usize), Vec<usize>> = BTreeMap::new();
    for (j, r) in account.rules.iter().enumerate() {
        if !r.inputs.is_empty() {
            classes.entry((r.inputs.len(), r.units.len())).or_default().push(j);
        }
    }
    let mut bodies = Vec::new();
    for ((d, units), members) in classes {
        let reals = shape(&account.rules[members[0]]).len();
        let unshared = (members.len() * reals) as f64 * LITERAL_BITS;
        let base = observations * account.remainder(h, y, fisher) / LN_2;
        let mut best: Option<(f64, Account, Vec<Vec<usize>>)> = None;
        let largest = members.iter().flat_map(|j| shape(&account.rules[*j])).fold(0.0f64, |m, v| m.max(v.abs()));
        let top = -(largest.max(f64::MIN_POSITIVE).log2().ceil() as i32) - 1;
        for p in top..top + 24 {
            let unit = (-p as f64).exp2();
            let mut groups: BTreeMap<Vec<i64>, Vec<usize>> = BTreeMap::new();
            let mut trial = account.clone();
            for j in &members {
                let key: Vec<i64> = shape(&account.rules[*j]).iter().map(|v| (v / unit).round() as i64).collect();
                let mut theta = vec![account.rules[*j].beta];
                theta.extend(key.iter().map(|k| *k as f64 * unit));
                trial.rules[*j].unpack(&theta);
                groups.entry(key).or_default().push(*j);
            }
            if groups.len() == members.len() {
                break;
            }
            let shared = (groups.len() * reals) as f64 * LITERAL_BITS + members.len() as f64 * (groups.len() as f64).log2();
            let cost = shared + observations * trial.remainder(h, y, fisher) / LN_2 - base;
            if cost < unshared && best.as_ref().is_none_or(|b| cost < b.0) {
                best = Some((cost, trial, groups.into_values().collect()));
            }
        }
        if let Some((cost, trial, groups)) = best {
            log::info!("share ({d} inputs, {units} units): {} rules in {} bodies, {cost:.0} bits vs {unshared:.0}", members.len(), groups.len());
            *account = trial;
            for g in groups.into_iter().filter(|g| g.len() > 1) {
                let mut rule = account.rules[g[0]].clone();
                rule.inputs = (0..d).collect();
                let formula = rule.formula().replace('a', "x");
                bodies.push(Body { inputs: d, units, formula, instances: g });
            }
        }
    }
    bodies.sort_by(|p, q| q.instances.len().cmp(&p.instances.len()));
    bodies
}

/// The bits of the account's rules once bodies are shared: per body its shape once, per instance
/// its body's name among its structure's bodies; rules in no body pay their own shape.
pub fn shared_rule_bits(account: &Account, bodies: &[Body]) -> f64 {
    let mut in_body = vec![false; account.rules.len()];
    let mut bits = 0.0;
    let mut per_class: BTreeMap<(usize, usize), usize> = BTreeMap::new();
    for b in bodies {
        *per_class.entry((b.inputs, b.units)).or_default() += 1;
    }
    for b in bodies {
        let reals = shape(&account.rules[b.instances[0]]).len();
        bits += reals as f64 * LITERAL_BITS;
        let names = per_class[&(b.inputs, b.units)] as f64;
        for j in &b.instances {
            in_body[*j] = true;
            bits += names.log2().max(0.0);
        }
    }
    let m = account.reads.nrows();
    for (j, r) in account.rules.iter().enumerate() {
        bits += r.pointer_bits(m);
        if !in_body[j] {
            bits += r.literals() as f64 * LITERAL_BITS;
        }
    }
    bits
}
