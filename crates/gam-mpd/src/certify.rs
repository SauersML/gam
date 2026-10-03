//! A certified worst case for an explanation's box claim (#2951).
//!
//! # The claim
//!
//! An explanation of an input names the subcomponents that are on. Under the box claim
//! ([`super::masked::Claim::Box`]) every off subcomponent's gate may sit anywhere in `[0, 1]`, and
//! the explanation's error is the worst `KL(model ‖ masked)` over all of them at once. Sampling and
//! an adversary ([`adversary`]) only find settings that are bad, a lower bound. [`certify`] bounds it
//! from above: per input, a number no gate setting in the box exceeds, rounding included.
//!
//! # Affine forms
//!
//! Each node's value at an input is enclosed by an affine form
//!
//! ```text
//! x = c + Σ_s G_s ε_s + r ⊙ β,     every ε_s and β_j in [−1, 1],
//! ```
//!
//! a center, generators on shared symbols, and an entrywise radius. Each symbol is one exact
//! quantity of the forward, defined once, so every form that names it moves with it. That keeps the
//! correlations an interval loses: the residual stream and the branch that reads it, or one gate's
//! effect on a query and on a value. A symbol is either a radius coordinate of an earlier form,
//! `ε_(n, row, j) = β_j` of node `n` at that row (promoted when a linear map reads it, below), or the
//! gate of a block spanning several columns. A rank-one gate needs no symbol: it is read only once,
//! by its own column, so a radius holds it exactly. Read as `∃ ε, β`, a form is sound when the
//! node's exact value is one of its points at the symbols' exact values.
//!
//! # Operations
//!
//! * **Linear maps** (affine nodes, transposed reads) are exact on centers and generators. A read
//!   coordinate's radius either passes as `|A| r` or is promoted to its symbol, with generator
//!   `A_{:,j} r_j`. The form keeps the promoted coordinates and its own generators of largest written
//!   `ℓ₁` mass, at most `budget` per row, and every other one passes as radius. A promoted coordinate
//!   whose symbol the form already carries (the residual stream read again) always joins it.
//! * **Products** (`z ⊙ m` at a masked site, `x ⊙ s` in a norm, the scores `q·k`, the reads `α v`):
//!   `(c + aε + r_a β)(d + bε + r_b β′)` has center `cd + ½ Σ_s a_s b_s`, generators `d a + c b`, and
//!   radius `|c| r_b + |d| r_a + (‖a‖₁ + r_a)(‖b‖₁ + r_b) − ½ Σ_s |a_s b_s|`. The square of a shared
//!   symbol lies in `[0, 1]`, which gives the half.
//! * **Scalar curves** (ReLU, SiLU, both GELUs, `exp`, `t^{-1/2}`, `1/t`) on each coordinate's interval
//!   `[l, h]` become a line `λ t + μ` with a certified deviation `δ`; `λ` is the chord's slope. For a
//!   convex curve the deviation is read at the ends, and below the tangent at the point of slope `λ`.
//!   For SiLU and the GELUs it is read on a grid over `[l, h]`, widened by `M₂ h²/8`, with `M₂` a bound
//!   on the curve's second derivative. ReLU is the zonotope relaxation, read at `l`, `0` and `h`.
//! * **RMS norm** `x (mean x² + ε)^{-1/2}`: the square, the mean, the curve `t^{-1/2}` on `[ε, h]`
//!   (the mean of squares is nonnegative), and the product.
//! * **Attention** rotates each query and key, a rotation per plane. It forms the scores as
//!   products, and each weight in its stable form `α_j = 1 / Σ_k e^{s_k − s_j}`. The exponents are
//!   differences of scores, so whatever the scores share cancels exactly, and each weight's curve
//!   is the reciprocal on `[1, h]`. The read is `Σ_j α_j v_j`; the weights are convex, so each read
//!   coordinate also lies within the values' own range, which replaces a coordinate the product
//!   relaxation leaves wider.
//!
//! # The divergence
//!
//! The logits are a linear read of a hidden node `h`, or the output node itself. Every radius of `h`
//! is promoted, so for each class `i` the gap `z′_i − z′_t` against the reference's top class `t` is
//! enclosed by `(C_i − C_t) ± Σ_s |G_{s,i} − G_{s,t}|`, and a shift common to all logits cancels. The
//! divergence over that box of gaps is bounded by P15″ ([`kl_supremum_over_gap_box`]), against the
//! reference's computed logits and their radius.
//!
//! # Rounding
//!
//! Every form is computed in binary64, and its radius absorbs the rounding:
//! * a linear map with inner width `n` adds `γ_{n+k} |A| (|c| + Σ_s |G_s| + r)`;
//! * a product adds `γ` times its operands' magnitudes;
//! * a curve adds its enclosures' evaluation error ([`Law`]'s computed-value radius, [`certified_exp`]);
//! * every sum of nonnegative terms is rounded up.
//!
//! Operators with a low-rank body are refused, since their product is not stored. A form whose
//! radius overflows makes every bound infinite. With `MPD_CERTIFY_TRACE` set, every node's mean
//! half-width and each row's logit-gap widths go to stderr, to show where the relaxation widens.
//!
//! # Branching
//!
//! The relaxation is loosest where a wide gate drives a curved map. [`certify_branching`] splits the
//! box at the gate whose setting matters most for the worst row: the largest `|∂KL/∂g|` times its
//! width, at the box's center. It bounds each half, and the worst case is the largest bound over
//! the leaves.

use std::collections::HashMap;
use std::sync::Arc;

use gam_linalg::faer_ndarray::fast_abt;
use gam_linalg::roundoff::{UNIT_ROUNDOFF, accumulation_growth};
use gam_math::score_opt::certified_exp;
use ndarray::{Array1, Array2, ArrayView2, Axis, Zip, s};
use rayon::prelude::*;

use super::bounds::kl_supremum_over_gap_box;
use super::masked::{Masked, Target, forward, mask_gradients};
use super::operator_program::{Basis, FamilyInputs, Law, Node, Operator, OperatorBody, OperatorProgram, Rotary, SlotValues};

const U: f64 = UNIT_ROUNDOFF;

fn up(x: f64) -> f64 {
    x.next_up()
}

fn down(x: f64) -> f64 {
    x.next_down()
}

fn gamma(n: usize) -> f64 {
    accumulation_growth(n)
}

/// An upper bound on the exact `a − b` from its rounded difference.
fn above(a: f64, b: f64) -> f64 {
    let d = a - b;
    up(d + 2.0 * U * d.abs())
}

/// A lower bound on the exact `a − b` from its rounded difference.
fn below(a: f64, b: f64) -> f64 {
    let d = a - b;
    down(d - 2.0 * U * d.abs())
}

/// The symbol of radius coordinate `j` of node `node` at row `row`.
fn promoted(node: usize, row: usize, j: usize) -> u64 {
    ((node as u64 + 1) << 42) | ((row as u64) << 21) | j as u64
}

/// The symbol of block `block` of slot `slot`'s gate at row `row`.
fn gate(slot: usize, row: usize, block: usize) -> u64 {
    (1u64 << 63) | ((slot as u64) << 42) | ((row as u64) << 21) | block as u64
}

const FIELD: usize = 1 << 21;

/// The union of ascending symbol lists, and each list's places in it.
fn union(lists: &[&[u64]]) -> (Vec<u64>, Vec<Vec<usize>>) {
    let mut all: Vec<u64> = lists.iter().flat_map(|l| l.iter().copied()).collect();
    all.sort_unstable();
    all.dedup();
    let places = lists
        .iter()
        .map(|l| l.iter().map(|id| all.binary_search(id).unwrap_or_default()).collect())
        .collect();
    (all, places)
}

/// Fresh symbols, each defined once by the [`enclose`] that draws it.
static FRESH: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(3 << 62);

/// The share of a budget held by principal symbols ([`enclose`]).
fn principal_share(budget: usize) -> usize {
    budget / 8
}

/// Generators `d_i` (rows of `dropped`, each on its own symbol) enclosed by at most `k` fresh symbols
/// along principal directions and a radius: for any `Q`, `Σ_i ε_i d_i = Q (Σ_i ε_i Qᵀd_i) + Σ_i ε_i r_i`
/// with `r_i = d_i − Q Qᵀ d_i`, and the first sum's `j`-th coordinate lies within `s_j = Σ_i |(Qᵀ d_i)_j|`,
/// so it is `Q_{:,j} s_j` times one fresh symbol. `Q` is an orthonormalised power iterate of the
/// generators' span (a randomized range finder); only the identity matters for soundness. Returns the
/// fresh symbols, their generators and the radius `Σ_i |r_i|`, with the products' and sums' rounding.
fn enclose(dropped: &Array2<f64>, k: usize) -> (Vec<u64>, Array2<f64>, Array1<f64>) {
    let (m, width) = dropped.dim();
    let total = |a: &Array2<f64>| {
        let mut out = Array1::<f64>::zeros(a.ncols());
        for row in a.outer_iter() {
            Zip::from(&mut out).and(&row).for_each(|o, &v| *o += v.abs());
        }
        out
    };
    let k = k.min(m).min(width);
    if k == 0 {
        let grow = 1.0 + gamma(m + 1);
        return (Vec::new(), Array2::zeros((0, width)), total(dropped).mapv(|v| up(v * grow)));
    }
    let mut rng = SplitMix(0x2951_e1c1 ^ ((m as u64) << 32) ^ width as u64);
    let omega = Array2::from_shape_fn((m, k), |_| rng.next() - 0.5);
    let mut q = dropped.t().dot(&omega);
    q = dropped.t().dot(&dropped.dot(&q));
    // Modified Gram–Schmidt; a column without a new direction is left zero.
    for j in 0..k {
        for i in 0..j {
            let projection = q.column(i).dot(&q.column(j));
            let previous = q.column(i).to_owned();
            q.column_mut(j).scaled_add(-projection, &previous);
        }
        let norm = q.column(j).dot(&q.column(j)).sqrt();
        if norm > 0.0 && norm.is_finite() {
            q.column_mut(j).mapv_inplace(|v| v / norm);
        } else {
            q.column_mut(j).fill(0.0);
        }
    }
    let p = dropped.dot(&q);
    let reach = total(&p).mapv(|v| up(v * (1.0 + gamma(m + 1))));
    let residual = dropped - &p.dot(&q.t());
    let q_abs = q.mapv(f64::abs);
    let spread = q_abs.dot(&reach);
    let g = gamma(k + width + 4);
    let grow = 1.0 + gamma(m + 1);
    let radius = Zip::from(&total(&residual))
        .and(&total(dropped))
        .and(&spread)
        .map_collect(|&r, &d, &s| up(up(r * grow) + up(g * up(d + up(3.0 * s)))));
    let mut generators = Array2::<f64>::zeros((k, width));
    for j in 0..k {
        generators.row_mut(j).assign(&q.column(j).mapv(|v| v * reach[j]));
    }
    let fresh: Vec<u64> = (0..k).map(|_| FRESH.fetch_add(1, std::sync::atomic::Ordering::Relaxed)).collect();
    (fresh, generators, radius)
}

/// One input's affine form over a node's coordinates (module note, "Affine forms").
#[derive(Clone, Debug)]
struct Row {
    center: Array1<f64>,
    /// Ascending symbols, one generator row each.
    ids: Vec<u64>,
    coef: Array2<f64>,
    radius: Array1<f64>,
}

impl Row {
    fn exact(center: Array1<f64>, radius: Array1<f64>) -> Self {
        let width = center.len();
        Self { center, ids: Vec::new(), coef: Array2::zeros((0, width)), radius }
    }

    fn width(&self) -> usize {
        self.center.len()
    }

    fn finite(&self) -> bool {
        self.radius.iter().all(|r| r.is_finite()) && self.center.iter().all(|c| c.is_finite())
    }

    /// Per coordinate, an upper bound on `Σ_s |G_s|`.
    fn spread(&self) -> Array1<f64> {
        let mut out = Array1::<f64>::zeros(self.width());
        for g in self.coef.outer_iter() {
            Zip::from(&mut out).and(&g).for_each(|o, &v| *o += v.abs());
        }
        let grow = 1.0 + gamma(self.ids.len() + 1);
        out.mapv_inplace(|v| up(v * grow));
        out
    }

    /// Per coordinate, an upper bound on `|c| + Σ_s |G_s| + r`: what every computed term is below.
    fn magnitude(&self) -> Array1<f64> {
        let mut out = self.spread();
        Zip::from(&mut out).and(&self.center).and(&self.radius).for_each(|o, &c, &r| *o = up(up(*o + c.abs()) + r));
        out
    }

    /// Per coordinate, an interval holding every point of the form.
    fn bounds(&self) -> (Array1<f64>, Array1<f64>) {
        let mut reach = self.spread();
        Zip::from(&mut reach).and(&self.radius).for_each(|a, &r| *a = up(*a + r));
        let lo = Zip::from(&self.center).and(&reach).map_collect(|&c, &r| down(c - r));
        let hi = Zip::from(&self.center).and(&reach).map_collect(|&c, &r| up(c + r));
        (lo, hi)
    }

    /// Coordinate `c` as a one-coordinate form.
    fn column(&self, c: usize) -> Row {
        Row {
            center: Array1::from_elem(1, self.center[c]),
            ids: self.ids.clone(),
            coef: self.coef.slice(s![.., c..c + 1]).to_owned(),
            radius: Array1::from_elem(1, self.radius[c]),
        }
    }

    /// Keep at most `budget` symbols: those of largest `ℓ₁` generator, and the rest enclosed by
    /// [`enclose`] in a share of the budget.
    fn reduce(&mut self, budget: usize) {
        if self.ids.len() <= budget {
            return;
        }
        let principal = principal_share(budget);
        let held = budget - principal;
        let mass: Vec<f64> = self.coef.outer_iter().map(|g| g.iter().map(|v| v.abs()).sum()).collect();
        let mut order: Vec<usize> = (0..self.ids.len()).collect();
        order.select_nth_unstable_by(held, |a, b| mass[*b].total_cmp(&mass[*a]));
        let mut keep = order[..held].to_vec();
        keep.sort_unstable();
        let mut dropped: Vec<usize> = order[held..].to_vec();
        dropped.sort_unstable();
        let (fresh, generators, radius) = enclose(&self.coef.select(Axis(0), &dropped), principal);
        Zip::from(&mut self.radius).and(&radius).for_each(|r, &d| *r = up(*r + d));
        let mut rows: Vec<(u64, Array1<f64>)> = keep.iter().map(|&k| (self.ids[k], self.coef.row(k).to_owned())).collect();
        rows.extend(fresh.into_iter().zip(generators.outer_iter().map(|g| g.to_owned())));
        rows.sort_unstable_by_key(|r| r.0);
        let mut coef = Array2::<f64>::zeros((rows.len(), self.width()));
        for (k, (_, g)) in rows.iter().enumerate() {
            coef.row_mut(k).assign(g);
        }
        self.ids = rows.into_iter().map(|r| r.0).collect();
        self.coef = coef;
    }

    /// `k x` for an exact constant `k`, with `extra` added to the radius.
    fn scaled(&self, k: f64, extra: f64) -> Row {
        let magnitude = self.magnitude();
        Row {
            center: self.center.mapv(|c| k * c),
            ids: self.ids.clone(),
            coef: self.coef.mapv(|g| k * g),
            radius: Zip::from(&self.radius)
                .and(&magnitude)
                .map_collect(|&r, &m| up(up(k.abs() * r) + up(up(2.0 * U * k.abs() + extra) * m))),
        }
    }

    /// `x + k` for an exact constant `k`.
    fn shifted(&self, k: f64) -> Row {
        let mut out = self.clone();
        Zip::from(&mut out.center).and(&mut out.radius).for_each(|c, r| {
            *c += k;
            *r = up(*r + up(U * c.abs()));
        });
        out
    }
}

/// `Σ_t sign_t x_t` over forms of one width.
fn combine(terms: &[(&Row, f64)]) -> Row {
    let width = terms[0].0.width();
    let lists: Vec<&[u64]> = terms.iter().map(|(x, _)| x.ids.as_slice()).collect();
    let (ids, places) = union(&lists);
    let mut coef = Array2::<f64>::zeros((ids.len(), width));
    let mut center = Array1::<f64>::zeros(width);
    let mut radius = Array1::<f64>::zeros(width);
    let mut magnitude = Array1::<f64>::zeros(width);
    for ((x, sign), place) in terms.iter().zip(&places) {
        center.scaled_add(*sign, &x.center);
        for (s, &u) in place.iter().enumerate() {
            coef.row_mut(u).scaled_add(*sign, &x.coef.row(s));
        }
        Zip::from(&mut radius).and(&x.radius).for_each(|o, &r| *o = up(*o + r));
        let m = x.magnitude();
        Zip::from(&mut magnitude).and(&m).for_each(|o, &v| *o = up(*o + v));
    }
    let g = gamma(terms.len() + 2);
    Zip::from(&mut radius).and(&magnitude).for_each(|r, &m| *r = up(*r + up(g * m)));
    Row { center, ids, coef, radius }
}

/// The sum of a form's coordinates, as a one-coordinate form.
fn total(x: &Row) -> Row {
    let width = x.width();
    let magnitude = x.magnitude().sum();
    let radius = up(up(x.radius.sum() * (1.0 + gamma(width + 1))) + up(gamma(width + 2) * magnitude));
    Row {
        center: Array1::from_elem(1, x.center.sum()),
        ids: x.ids.clone(),
        coef: x.coef.sum_axis(Axis(1)).insert_axis(Axis(1)),
        radius: Array1::from_elem(1, radius),
    }
}

/// Forms side by side.
fn concat(parts: &[&Row]) -> Row {
    let width: usize = parts.iter().map(|p| p.width()).sum();
    let lists: Vec<&[u64]> = parts.iter().map(|x| x.ids.as_slice()).collect();
    let (ids, places) = union(&lists);
    let mut coef = Array2::<f64>::zeros((ids.len(), width));
    let mut center = Array1::<f64>::zeros(width);
    let mut radius = Array1::<f64>::zeros(width);
    let mut offset = 0;
    for (x, place) in parts.iter().zip(&places) {
        let w = x.width();
        center.slice_mut(s![offset..offset + w]).assign(&x.center);
        radius.slice_mut(s![offset..offset + w]).assign(&x.radius);
        for (s, &u) in place.iter().enumerate() {
            coef.slice_mut(s![u, offset..offset + w]).assign(&x.coef.row(s));
        }
        offset += w;
    }
    Row { center, ids, coef, radius }
}

/// The product of two forms coordinate by coordinate, a one-coordinate form broadcast against the
/// other (module note, "Operations").
fn product(l: &Row, r: &Row) -> Row {
    let (wl, wr) = (l.width(), r.width());
    let width = wl.max(wr);
    let li = |d: usize| if wl == 1 { 0 } else { d };
    let ri = |d: usize| if wr == 1 { 0 } else { d };
    let (ids, places) = union(&[&l.ids, &r.ids]);
    let (sl, sr) = (l.spread(), r.spread());
    let mut coef = Array2::<f64>::zeros((ids.len(), width));
    for (s, &u) in places[0].iter().enumerate() {
        let mut out = coef.row_mut(u);
        for d in 0..width {
            out[d] += r.center[ri(d)] * l.coef[[s, li(d)]];
        }
    }
    for (s, &u) in places[1].iter().enumerate() {
        let mut out = coef.row_mut(u);
        for d in 0..width {
            out[d] += l.center[li(d)] * r.coef[[s, ri(d)]];
        }
    }
    let mut shift = Array1::<f64>::zeros(width);
    let mut paired = Array1::<f64>::zeros(width);
    let (mut i, mut j) = (0, 0);
    while i < l.ids.len() && j < r.ids.len() {
        match l.ids[i].cmp(&r.ids[j]) {
            std::cmp::Ordering::Less => i += 1,
            std::cmp::Ordering::Greater => j += 1,
            std::cmp::Ordering::Equal => {
                for d in 0..width {
                    let p = l.coef[[i, li(d)]] * r.coef[[j, ri(d)]];
                    shift[d] += 0.5 * p;
                    paired[d] += 0.5 * p.abs();
                }
                i += 1;
                j += 1;
            }
        }
    }
    let g = gamma(ids.len() + 8);
    let mut center = Array1::<f64>::zeros(width);
    let mut radius = Array1::<f64>::zeros(width);
    for d in 0..width {
        let (cl, cr) = (l.center[li(d)], r.center[ri(d)]);
        let (al, ar) = (sl[li(d)], sr[ri(d)]);
        let (rl, rr) = (l.radius[li(d)], r.radius[ri(d)]);
        center[d] = cl * cr + shift[d];
        let quadratic = up(up(al + rl) * up(ar + rr));
        let tightened = above(quadratic, down(paired[d] * (1.0 - g))).max(0.0);
        let linear = up(up(cl.abs() * rr) + up(cr.abs() * rl));
        let magnitude = up(up(up(cl.abs() + al) + rl) * up(up(cr.abs() + ar) + rr));
        radius[d] = up(up(linear + tightened) + up(3.0 * g * magnitude));
    }
    Row { center, ids, coef, radius }
}

/// An operator read as the linear map `x ↦ M x` (written × read), with what its relaxation reads of it.
struct Map {
    operator: Arc<Operator>,
    /// The matrix of a body that stores none (the identity, a diagonal).
    formed: Option<Array2<f64>>,
    /// Whether `M` is the operator's transpose (a transposed read).
    transposed: bool,
    /// Per read coordinate, `Σ_i |m_ij|`, rounded up.
    column_l1: Array1<f64>,
}

impl Map {
    fn new(operator: Arc<Operator>, transposed: bool) -> Result<Self, String> {
        let formed = match &operator.body {
            OperatorBody::Dense { .. } => None,
            OperatorBody::LowRank { .. } => return Err(format!("operator {}: a low-rank body is not certified", operator.name)),
            _ => Some(operator.matrix()),
        };
        let mut map = Self { operator, formed, transposed, column_l1: Array1::zeros(0) };
        let m = map.m();
        let mut column_l1 = Array1::<f64>::zeros(m.ncols());
        for row in m.outer_iter() {
            Zip::from(&mut column_l1).and(&row).for_each(|o, &v| *o += v.abs());
        }
        let grow = 1.0 + gamma(m.nrows() + 1);
        column_l1.mapv_inplace(|v| up(v * grow));
        map.column_l1 = column_l1;
        Ok(map)
    }

    fn m(&self) -> ArrayView2<'_, f64> {
        let base = match (&self.formed, &self.operator.body) {
            (Some(formed), _) => formed.view(),
            (None, OperatorBody::Dense { values, .. }) => values.view(),
            (None, _) => unreachable!("only a dense body is read in place"),
        };
        if self.transposed { base.reversed_axes() } else { base }
    }

    /// `|M| v` for `v ≥ 0`, rounded up.
    fn absolute(&self, v: &Array1<f64>) -> Array1<f64> {
        let m = self.m();
        let (rows, cols) = m.dim();
        let nonzero: Vec<usize> = (0..cols).filter(|&j| v[j] != 0.0).collect();
        let mut out = Array1::<f64>::zeros(rows);
        if nonzero.len() * 8 < cols {
            for &j in &nonzero {
                let vj = v[j];
                Zip::from(&mut out).and(m.column(j)).for_each(|o, &a| *o += a.abs() * vj);
            }
        } else {
            for (o, row) in out.iter_mut().zip(m.outer_iter()) {
                *o = row.iter().zip(v.iter()).map(|(a, b)| a.abs() * b).sum();
            }
        }
        let grow = 1.0 + gamma(cols + 1);
        out.mapv_inplace(|x| up(x * grow));
        out
    }
}

/// `Σ_t M_t x_t + b` at row `row`, each term `(form, map, read node)` (module note, "Linear maps").
fn affine(terms: &[(&Row, &Map, usize)], bias: Option<&Array1<f64>>, row: usize, budget: usize) -> Row {
    let written = terms[0].1.m().nrows();
    let lists: Vec<&[u64]> = terms.iter().map(|(x, _, _)| x.ids.as_slice()).collect();
    let (ids, places) = union(&lists);
    let mut coef = Array2::<f64>::zeros((ids.len(), written));
    let mut center = bias.cloned().unwrap_or_else(|| Array1::zeros(written));
    // Promotion candidates: symbol → (written ℓ₁ mass, its (term, coordinate) members).
    let mut fresh: HashMap<u64, (f64, Vec<(usize, usize)>)> = HashMap::new();
    for (t, (x, map, node)) in terms.iter().enumerate() {
        let m = map.m();
        center += &m.dot(&x.center);
        if !x.ids.is_empty() {
            let g = fast_abt(&x.coef, &m);
            for (s, &u) in places[t].iter().enumerate() {
                coef.row_mut(u).scaled_add(1.0, &g.row(s));
            }
        }
        for j in 0..x.width() {
            let r = x.radius[j];
            if r > 0.0 {
                let id = promoted(*node, row, j);
                match ids.binary_search(&id) {
                    Ok(u) => coef.row_mut(u).scaled_add(r, &m.column(j)),
                    Err(_) => {
                        let entry = fresh.entry(id).or_insert((0.0, Vec::new()));
                        entry.0 += map.column_l1[j] * r;
                        entry.1.push((t, j));
                    }
                }
            }
        }
    }
    let mut fresh: Vec<(u64, f64, Vec<(usize, usize)>)> = fresh.into_iter().map(|(id, (mass, members))| (id, mass, members)).collect();
    fresh.sort_unstable_by_key(|f| f.0);
    let mass: Vec<f64> = coef.outer_iter().map(|g| g.iter().map(|v| v.abs()).sum()).collect();
    let candidates = ids.len() + fresh.len();
    let mut keep = vec![true; candidates];
    let principal = if candidates > budget { principal_share(budget) } else { 0 };
    let held = budget - principal;
    // The dropped candidates, by written mass: the heaviest go to the principal enclosure.
    let mut order: Vec<(f64, usize)> =
        mass.iter().copied().enumerate().map(|(i, m)| (m, i)).chain(fresh.iter().enumerate().map(|(i, f)| (f.1, ids.len() + i))).collect();
    if candidates > budget {
        order.sort_unstable_by(|a, b| b.0.total_cmp(&a.0));
        keep.iter_mut().for_each(|k| *k = false);
        for &(_, i) in &order[..held] {
            keep[i] = true;
        }
    }
    let enclosed: Vec<usize> = if candidates > budget { order[held..].iter().take(4 * budget.max(1)).map(|o| o.1).collect() } else { Vec::new() };
    let mut in_enclosure = vec![false; candidates];
    for &i in &enclosed {
        in_enclosure[i] = true;
    }
    let mut radius = Array1::<f64>::zeros(written);
    let mut dropped_rows: Vec<Array1<f64>> = Vec::new();
    let mut boxed = 0usize;
    for (u, g) in coef.outer_iter().enumerate().filter(|(u, _)| !keep[*u]) {
        if in_enclosure[u] {
            dropped_rows.push(g.to_owned());
        } else {
            Zip::from(&mut radius).and(&g).for_each(|o, &v| *o += v.abs());
            boxed += 1;
        }
    }
    let grow = 1.0 + gamma(boxed + 1);
    radius.mapv_inplace(|v| up(v * grow));
    // What passes through `|M_t|`: each term's rounding magnitude and its unpromoted radii.
    let g = gamma(terms.iter().map(|(x, _, _)| x.width()).max().unwrap_or(0) + terms.len() + 4);
    let mut passed: Vec<Array1<f64>> = terms.iter().map(|(x, _, _)| x.magnitude().mapv(|m| up(g * m))).collect();
    for (i, f) in fresh.iter().enumerate().filter(|(i, _)| !keep[ids.len() + i]) {
        if in_enclosure[ids.len() + i] {
            let mut generator = Array1::<f64>::zeros(written);
            for &(t, j) in &f.2 {
                generator.scaled_add(terms[t].0.radius[j], &terms[t].1.m().column(j));
            }
            dropped_rows.push(generator);
        } else {
            for &(t, j) in &f.2 {
                passed[t][j] = up(passed[t][j] + terms[t].0.radius[j]);
            }
        }
    }
    for ((_, map, _), v) in terms.iter().zip(&passed) {
        let through = map.absolute(v);
        Zip::from(&mut radius).and(&through).for_each(|r, &x| *r = up(*r + x));
    }
    if let Some(b) = bias {
        Zip::from(&mut radius).and(b).for_each(|r, &x| *r = up(*r + up(g * x.abs())));
    }
    let mut principal_rows: Vec<(u64, Array1<f64>)> = Vec::new();
    if !dropped_rows.is_empty() {
        let mut dropped = Array2::<f64>::zeros((dropped_rows.len(), written));
        for (i, r) in dropped_rows.iter().enumerate() {
            dropped.row_mut(i).assign(r);
        }
        let (ids_new, generators, extra) = enclose(&dropped, principal);
        Zip::from(&mut radius).and(&extra).for_each(|r, &x| *r = up(*r + x));
        principal_rows = ids_new.into_iter().zip(generators.outer_iter().map(|g| g.to_owned())).collect();
    }
    // The kept symbols, ascending.
    let mut rows: Vec<(u64, Array1<f64>)> = Vec::new();
    for (u, id) in ids.iter().enumerate() {
        if keep[u] {
            rows.push((*id, coef.row(u).to_owned()));
        }
    }
    for (f, _) in fresh.iter().zip(&keep[ids.len()..]).filter(|(_, k)| **k) {
        let mut generator = Array1::<f64>::zeros(written);
        for &(t, j) in &f.2 {
            generator.scaled_add(terms[t].0.radius[j], &terms[t].1.m().column(j));
        }
        rows.push((f.0, generator));
    }
    rows.extend(principal_rows);
    rows.sort_unstable_by_key(|r| r.0);
    let mut out = Array2::<f64>::zeros((rows.len(), written));
    for (s, (_, generator)) in rows.iter().enumerate() {
        out.row_mut(s).assign(generator);
    }
    Row { center, ids: rows.into_iter().map(|r| r.0).collect(), coef: out, radius }
}

/// A scalar curve a relaxation replaces by lines (module note, "Operations").
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) enum Curve {
    Law(Law),
    Exp,
    InverseSqrt,
    Reciprocal,
}

/// Bounds on the second derivative of the smooth laws over the reals: SiLU's is `σ(1 − σ)(2 + t(1 − 2σ))`, largest
/// in size at `0` (`½`); the exact GELU's is `φ(t)(2 − t²)`, largest at `0` (`2φ(0) ≈ 0.7979`), and the tanh GELU's
/// is `√(2/π) ≈ 0.7979` there (`second_derivative_bounds_hold` checks both on a grid).
pub(crate) const SILU_CURVATURE: f64 = 0.5001;
pub(crate) const GELU_CURVATURE: f64 = 0.8;

impl Curve {
    /// An enclosure of the exact curve at `x`.
    fn enclose(self, x: f64) -> Option<(f64, f64)> {
        match self {
            Self::Law(law) => {
                let v = law.apply(x);
                let e = law.radius(x, v, 0.0);
                (v.is_finite() && e.is_finite()).then(|| (down(v - e), up(v + e)))
            }
            Self::Exp => certified_exp(x).map(|i| (i.lo, i.hi)),
            Self::InverseSqrt => {
                let v = 1.0 / x.sqrt();
                (x > 0.0 && v.is_finite()).then(|| (down(v - 3.0 * U * v), up(v + 3.0 * U * v)))
            }
            Self::Reciprocal => {
                let v = 1.0 / x;
                (x > 0.0 && v.is_finite()).then(|| (down(v - 2.0 * U * v), up(v + 2.0 * U * v)))
            }
        }
    }
}

/// `(μ, δ)` with every value of `[g̲, ḡ]` within `δ` of `μ`.
fn centred(low: f64, high: f64) -> (f64, f64) {
    let mu = 0.5 * (low + high);
    (mu, above(high, mu).max(above(mu, low)))
}

/// `(λ, μ, δ)` with `|f(t) − λ t − μ| ≤ δ` on `[l, h]` (module note, "Operations"); `None` when an
/// enclosure fails.
pub(crate) fn linearize(curve: Curve, l: f64, h: f64) -> Option<(f64, f64, f64)> {
    if !(l.is_finite() && h.is_finite() && l <= h) {
        return None;
    }
    // `g(t) = f(t) − λ t` from an enclosure of `f(t)`, outward.
    let g_hi = |hi: f64, lambda: f64, t: f64| {
        let p = lambda * t;
        up(above(hi, p) + U * p.abs())
    };
    let g_lo = |lo: f64, lambda: f64, t: f64| {
        let p = lambda * t;
        down(below(lo, p) - U * p.abs())
    };
    let point = |t: f64| curve.enclose(t).map(|(lo, hi)| centred(lo, hi)).map(|(mu, delta)| (0.0, mu, delta));
    match curve {
        Curve::Law(Law::Identity) => Some((1.0, 0.0, 0.0)),
        Curve::Law(Law::Zero) => Some((0.0, 0.0, 0.0)),
        Curve::Law(Law::Relu) => {
            if h <= 0.0 {
                Some((0.0, 0.0, 0.0))
            } else if l >= 0.0 {
                Some((1.0, 0.0, 0.0))
            } else {
                // `relu(t) − λ t` is piecewise linear with its only kink at 0.
                let lambda = h / (h - l);
                let ends = [g_lo(0.0, lambda, l), g_lo(h, lambda, h), 0.0];
                let tops = [g_hi(0.0, lambda, l), g_hi(h, lambda, h), 0.0];
                let (mu, delta) = centred(ends.iter().copied().fold(f64::INFINITY, f64::min), tops.iter().copied().fold(f64::NEG_INFINITY, f64::max));
                Some((lambda, mu, delta))
            }
        }
        Curve::Law(law) => {
            if h == l {
                return point(l);
            }
            let curvature = if law == Law::Silu { SILU_CURVATURE } else { GELU_CURVATURE };
            let (fl, fh) = (curve.enclose(l)?, curve.enclose(h)?);
            let lambda = (0.5 * (fh.0 + fh.1) - 0.5 * (fl.0 + fl.1)) / (h - l);
            let steps = (((h - l) * 8.0).ceil() as usize).clamp(4, 256);
            let mut previous = l;
            let mut gap = 0.0_f64;
            let (mut low, mut high) = (f64::INFINITY, f64::NEG_INFINITY);
            for k in 0..=steps {
                let t = if k == steps { h } else { (l + (h - l) * (k as f64 / steps as f64)).clamp(l, h) };
                gap = gap.max(above(t, previous));
                previous = t;
                let (lo, hi) = curve.enclose(t)?;
                low = low.min(g_lo(lo, lambda, t));
                high = high.max(g_hi(hi, lambda, t));
            }
            let bend = up(up(up(curvature * up(gap * gap)) / 8.0) * (1.0 + 4.0 * U));
            let (mu, delta) = centred(down(low - bend), up(high + bend));
            Some((lambda, mu, delta))
        }
        Curve::Exp | Curve::InverseSqrt | Curve::Reciprocal => {
            let (fl, fh) = (curve.enclose(l)?, curve.enclose(h)?);
            if h == l {
                return point(l);
            }
            let lambda = (0.5 * (fh.0 + fh.1) - 0.5 * (fl.0 + fl.1)) / (h - l);
            let high = g_hi(fl.1, lambda, l).max(g_hi(fh.1, lambda, h));
            // The convex `g` lies above its tangent at `t`, the point of slope `λ`.
            let t = match curve {
                Curve::Exp => lambda.ln(),
                Curve::InverseSqrt => (-2.0 * lambda).powf(-2.0 / 3.0),
                _ => (-lambda).powf(-0.5),
            };
            let t = if t.is_finite() { t.clamp(l, h) } else { 0.5 * (l + h) };
            let (ft_lo, ft_hi) = curve.enclose(t)?;
            let (slope_lo, slope_hi) = match curve {
                Curve::Exp => (ft_lo, ft_hi),
                Curve::InverseSqrt => {
                    let v = -0.5 / (t * t.sqrt());
                    (v - 6.0 * U * v.abs(), v + 6.0 * U * v.abs())
                }
                _ => {
                    let v = -1.0 / (t * t);
                    (v - 4.0 * U * v.abs(), v + 4.0 * U * v.abs())
                }
            };
            let tilt = above(slope_hi, lambda).abs().max(below(slope_lo, lambda).abs());
            let tilt = up(tilt * (1.0 + 4.0 * U));
            let reach = above(t, l).max(above(h, t));
            let low = down(g_lo(ft_lo, lambda, t) - up(tilt * reach));
            if !(low.is_finite() && high.is_finite()) {
                return None;
            }
            let (mu, delta) = centred(low, high);
            Some((lambda, mu, delta))
        }
    }
}

/// Each coordinate's curve applied (`floor` the smallest value its argument takes exactly).
fn curved(x: &Row, curves: &[Curve], floor: f64) -> Row {
    let (lo, hi) = x.bounds();
    curved_within(x, curves, &lo.mapv(|l| l.max(floor)), &hi)
}

/// Each coordinate's curve applied on `[lo, hi]`, an interval its exact argument is known to lie in.
fn curved_within(x: &Row, curves: &[Curve], lo: &Array1<f64>, hi: &Array1<f64>) -> Row {
    let width = x.width();
    let magnitude = x.magnitude();
    let mut out = x.clone();
    for d in 0..width {
        let curve = curves[if curves.len() == 1 { 0 } else { d }];
        let line = if lo[d].is_nan() || hi[d].is_nan() { None } else { linearize(curve, lo[d], hi[d].max(lo[d])) };
        // A failed relaxation poisons the center, so no later maximum can hide it.
        let Some((lambda, mu, delta)) = line else {
            out.center[d] = f64::NAN;
            out.radius[d] = f64::INFINITY;
            continue;
        };
        out.center[d] = lambda * x.center[d] + mu;
        out.coef.column_mut(d).mapv_inplace(|g| lambda * g);
        let rounding = up(gamma(4) * up(up(lambda.abs() * magnitude[d]) + mu.abs()));
        out.radius[d] = up(up(up(lambda.abs() * x.radius[d]) + delta) + rounding);
    }
    out
}

/// `x (mean x² + ε)^{-1/2}`. The mean of squares is also bounded through the row's length,
/// `‖x‖ ∈ ‖c‖ ± (‖G‖₂ √S + ‖r‖)` (the symbols' corners have length `√S`), which a sum of
/// coordinatewise squares cannot see; the scale is the curve's line on the intersection of both
/// intervals, or that interval itself when it is narrower than the line.
fn rms_norm(x: &Row, epsilon: f64) -> Row {
    let n = x.width();
    let mean = total(&product(x, x)).scaled(1.0 / n as f64, 0.0).shifted(epsilon);
    let (lo, hi) = mean.bounds();
    let (mut l, mut h) = (lo[0].max(epsilon), hi[0]);
    let operator = if x.ids.is_empty() { Some(0.0) } else { super::operator_program::matrix_spectral_bound(x.coef.view()).ok() };
    if let Some(operator) = operator {
        let rows = (x.ids.len() as f64).sqrt().next_up();
        let r = up(x.radius.iter().map(|v| v * v).sum::<f64>() * (1.0 + gamma(n + 1))).sqrt().next_up();
        let rho = up(up(operator * rows) + r);
        let squares = x.center.iter().map(|v| v * v).sum::<f64>();
        let length_hi = up(squares * (1.0 + gamma(n + 1))).sqrt().next_up();
        let length_lo = down(squares * (1.0 - gamma(n + 1))).max(0.0).sqrt().next_down().max(0.0);
        let shortest = down(length_lo - rho).max(0.0);
        let longest = up(length_hi + rho);
        let k = n as f64;
        l = l.max(down(down(down(shortest * shortest) / k) + epsilon));
        h = h.min(up(up(up(longest * longest) / k) + epsilon));
    }
    let h = h.max(l);
    let line = curved_within(&mean, &[Curve::InverseSqrt], &Array1::from_elem(1, l), &Array1::from_elem(1, h));
    let scale = match (Curve::InverseSqrt.enclose(h), Curve::InverseSqrt.enclose(l)) {
        (Some((low, _)), Some((_, high))) => {
            let (mid, half) = centred(low, high);
            let reach = |row: &Row| up(row.spread()[0] + row.radius[0]);
            if half < reach(&line) || !line.finite() { Row::exact(Array1::from_elem(1, mid), Array1::from_elem(1, half)) } else { line }
        }
        _ => line,
    };
    product(x, &scale)
}

/// `x` rotated to `position`, plane by plane.
fn rotate(x: &Row, rotary: Rotary, position: u32) -> Row {
    let mut out = x.clone();
    let magnitude = x.magnitude();
    for (plane, (a, b)) in rotary.pairs().into_iter().enumerate() {
        let (c, s) = rotary.turn(plane, position);
        out.center[a] = c * x.center[a] - s * x.center[b];
        out.center[b] = s * x.center[a] + c * x.center[b];
        for k in 0..x.ids.len() {
            let (ga, gb) = (x.coef[[k, a]], x.coef[[k, b]]);
            out.coef[[k, a]] = c * ga - s * gb;
            out.coef[[k, b]] = s * ga + c * gb;
        }
        // libm's one ulp on each of the cosine and sine, and the products and sums.
        let rounding = up(4.0 * U * up(magnitude[a] + magnitude[b]));
        out.radius[a] = up(up(up(c.abs() * x.radius[a]) + up(s.abs() * x.radius[b])) + rounding);
        out.radius[b] = up(up(up(s.abs() * x.radius[a]) + up(c.abs() * x.radius[b])) + rounding);
    }
    out
}

/// The weights `α_j = 1 / Σ_k e^{s_k − s_j}` of one-coordinate scores.
fn softmax(scores: &[Row], budget: usize) -> Vec<Row> {
    if scores.len() == 1 {
        return vec![Row::exact(Array1::ones(1), Array1::zeros(1))];
    }
    (0..scores.len())
        .map(|j| {
            let mut terms: Vec<Row> = Vec::with_capacity(scores.len() - 1);
            for k in (0..scores.len()).filter(|&k| k != j) {
                let mut e = curved(&combine(&[(&scores[k], 1.0), (&scores[j], -1.0)]), &[Curve::Exp], f64::NEG_INFINITY);
                e.reduce(budget);
                terms.push(e);
            }
            let refs: Vec<(&Row, f64)> = terms.iter().map(|t| (t, 1.0)).collect();
            let mut sum = combine(&refs).shifted(1.0);
            sum.reduce(budget);
            let mut weight = curved(&sum, &[Curve::Reciprocal], 1.0);
            weight.reduce(budget);
            weight
        })
        .collect()
}

/// `Σ_j α_j p_j`; with `convex` (weights that sum to one and are nonnegative, a softmax's), each
/// coordinate also lies between the payloads' smallest lower and largest upper end, and a
/// coordinate whose form is wider than that interval is replaced by it.
fn mix(weights: &[Row], payloads: &[&Row], budget: usize, convex: bool) -> Row {
    let reads: Vec<Row> = weights.iter().zip(payloads).map(|(a, p)| product(a, p)).collect();
    let refs: Vec<(&Row, f64)> = reads.iter().map(|r| (r, 1.0)).collect();
    let mut out = combine(&refs);
    if convex {
        let width = out.width();
        let mut low = Array1::from_elem(width, f64::INFINITY);
        let mut high = Array1::from_elem(width, f64::NEG_INFINITY);
        for p in payloads {
            let (l, h) = p.bounds();
            Zip::from(&mut low).and(&l).for_each(|a, &b| *a = a.min(b));
            Zip::from(&mut high).and(&h).for_each(|a, &b| *a = a.max(b));
        }
        let (lo, hi) = out.bounds();
        for d in 0..width {
            if high[d] - low[d] < hi[d] - lo[d] {
                let (mid, half) = centred(low[d], high[d]);
                out.center[d] = mid;
                out.radius[d] = half;
                out.coef.column_mut(d).fill(0.0);
            }
        }
    }
    out.reduce(budget);
    out
}

/// `c q·k`.
fn score(q: &Row, k: &Row, c: f64) -> Row {
    total(&product(q, k)).scaled(c, 0.0)
}

/// Per input row, which inputs it may read: its sequence's rows at or before its position when
/// `causal`, every row of its sequence otherwise, in position order.
fn visible(inputs: &FamilyInputs, causal: bool) -> Result<Vec<Vec<usize>>, String> {
    let layout = inputs.layout.as_ref().ok_or("an attend node needs a sequence layout")?;
    let mut members: HashMap<u32, Vec<usize>> = HashMap::new();
    for row in 0..inputs.rows {
        members.entry(layout.sequence[row]).or_default().push(row);
    }
    for list in members.values_mut() {
        list.sort_by_key(|&r| layout.position[r]);
    }
    Ok((0..inputs.rows)
        .map(|row| {
            members[&layout.sequence[row]]
                .iter()
                .copied()
                .filter(|&other| !causal || layout.position[other] <= layout.position[row])
                .collect()
        })
        .collect())
}

/// A node's kind, for the relaxation's log.
fn kind(node: &Node) -> &'static str {
    match node {
        Node::Feature { .. } => "feature",
        Node::Raw { .. } => "raw",
        Node::Constant { .. } => "constant",
        Node::Affine { .. } => "affine",
        Node::Bilinear { .. } => "bilinear",
        Node::Softmax { .. } => "softmax",
        Node::Mix { .. } => "mix",
        Node::Pointwise { .. } => "pointwise",
        Node::Hadamard { .. } => "hadamard",
        Node::Readout { .. } => "readout",
        Node::Outer { .. } => "outer",
        Node::Concat { .. } => "concat",
        Node::Param { .. } => "param",
        Node::Call { .. } => "call",
        Node::Gain { .. } => "gain",
        Node::Attend { .. } => "attend",
        Node::RmsNorm { .. } => "rms norm",
        Node::Transposed { .. } => "transposed",
    }
}

/// One free raw slot: each entry anywhere in `[lower, upper]`, the columns of one block moving together.
#[derive(Clone, Debug)]
pub struct FreeSlot {
    pub slot: usize,
    pub lower: Array2<f64>,
    pub upper: Array2<f64>,
    /// Each column's block; the columns of one block share one value.
    pub blocks: Vec<usize>,
}

/// The logits as a dense read of a hidden node: `(hidden, class-major map, bias, skipped nodes)`.
type HeadRead = (usize, Map, Option<Array1<f64>>, Vec<usize>);

/// The program's logits when they are one dense read of a hidden node (module note, "The divergence").
fn head_of(program: &OperatorProgram) -> Result<Option<HeadRead>, String> {
    let output = program.output;
    let (logits, mut skip) = match &program.nodes[output] {
        Node::Readout { input, basis } if matches!(program.bases[*basis], Basis::Indicator { .. }) => (*input, vec![output, *input]),
        _ => (output, vec![output]),
    };
    let consumed_elsewhere = program.nodes.iter().enumerate().any(|(i, n)| i != output && n.arguments().contains(&logits));
    if consumed_elsewhere {
        return Ok(None);
    }
    let read = match &program.nodes[logits] {
        Node::Affine { terms, bias } if terms.len() == 1 => {
            let bias = bias.map(|b| program.operators[b].column(0));
            (terms[0].0, Map::new(program.operators[terms[0].1].clone(), false)?, bias)
        }
        Node::Transposed { input, operator } => (*input, Map::new(program.operators[*operator].clone(), true)?, None),
        _ => return Ok(None),
    };
    skip.dedup();
    Ok(Some((read.0, read.1, read.2, skip)))
}

/// Per class, an interval of the exact gap `z′_i − z′_top` of the logits read from the hidden form `x` by
/// `classes` (class-major) and `bias`, every radius of `x` promoted.
fn head_gaps(x: &Row, classes: ArrayView2<'_, f64>, bias: Option<&Array1<f64>>, top: usize) -> (Array1<f64>, Array1<f64>) {
    let width = x.width();
    let promoted: Vec<usize> = (0..width).filter(|&j| x.radius[j] > 0.0).collect();
    let mut generators = Array2::<f64>::zeros((x.ids.len() + promoted.len(), width));
    generators.slice_mut(s![..x.ids.len(), ..]).assign(&x.coef);
    for (k, &j) in promoted.iter().enumerate() {
        generators[[x.ids.len() + k, j]] = x.radius[j];
    }
    let symbols = generators.nrows();
    let magnitude = x.magnitude();
    let rounding = gamma(width + symbols + 4);
    let read = |i: usize| classes.row(i).dot(&x.center) + bias.map_or(0.0, |b| b[i]);
    let top_center = read(top);
    let top_generator = generators.dot(&classes.row(top));
    let top_magnitude = up(classes.row(top).iter().zip(magnitude.iter()).map(|(a, m)| a.abs() * m).sum::<f64>() * (1.0 + gamma(width + 1)));
    let count = classes.nrows();
    let mut lower = Array1::<f64>::zeros(count);
    let mut upper = Array1::<f64>::zeros(count);
    const TILE: usize = 2048;
    for start in (0..count).step_by(TILE) {
        let end = (start + TILE).min(count);
        let tile = classes.slice(s![start..end, ..]);
        let generated = fast_abt(&generators, &tile);
        let centers = tile.dot(&x.center);
        for (k, i) in (start..end).enumerate() {
            if i == top {
                continue;
            }
            let spread: f64 = generated.column(k).iter().zip(top_generator.iter()).map(|(a, b)| (a - b).abs()).sum();
            let spread = up(spread * (1.0 + gamma(symbols + 2)));
            let class_magnitude = up(tile.row(k).iter().zip(magnitude.iter()).map(|(a, m)| a.abs() * m).sum::<f64>() * (1.0 + gamma(width + 1)));
            let error = up(rounding * up(up(class_magnitude + top_magnitude) + bias.map_or(0.0, |b| b[i].abs() + b[top].abs())));
            let gap = (centers[k] + bias.map_or(0.0, |b| b[i])) - top_center;
            let reach = up(up(spread + error) + 2.0 * U * gap.abs());
            lower[i] = down(gap - reach);
            upper[i] = up(gap + reach);
        }
    }
    (lower, upper)
}

/// Per class, an interval of the gap `z′_i − z′_top` of a logits form.
fn form_gaps(x: &Row, top: usize) -> (Array1<f64>, Array1<f64>) {
    let count = x.width();
    let mut lower = Array1::<f64>::zeros(count);
    let mut upper = Array1::<f64>::zeros(count);
    let symbols = x.ids.len();
    for i in (0..count).filter(|&i| i != top) {
        let spread: f64 = (0..symbols).map(|s| (x.coef[[s, i]] - x.coef[[s, top]]).abs()).sum();
        let spread = up(spread * (1.0 + gamma(symbols + 2)));
        let gap = x.center[i] - x.center[top];
        let reach = up(up(up(spread + x.radius[i]) + x.radius[top]) + 2.0 * U * gap.abs());
        lower[i] = down(gap - reach);
        upper[i] = up(gap + reach);
    }
    (lower, upper)
}

/// Per input row of `inputs`, an upper bound on `KL(target ‖ program)` at every value of the free slots
/// (module note), each row's reference within `reference_radius` of the target's logits. At most
/// `budget` symbols are kept per row. An unscored row's bound is zero.
pub fn certify_program(
    program: &OperatorProgram,
    inputs: &FamilyInputs,
    free: &[FreeSlot],
    target: &Target,
    reference_radius: Option<&Array2<f64>>,
    budget: usize,
) -> Result<Array1<f64>, String> {
    let rows = inputs.rows;
    let nodes = program.nodes.len();
    if rows >= FIELD || nodes + 1 >= FIELD {
        return Err(format!("{rows} rows and {nodes} nodes exceed the symbol fields"));
    }
    if target.logits.nrows() != rows || reference_radius.is_some_and(|r| r.dim() != target.logits.dim()) {
        return Err("the target and its radius need one row per input".to_string());
    }
    let interfaces = program.interfaces().map_err(|e| e.to_string())?;
    if interfaces.iter().any(|i| i.width() >= FIELD) {
        return Err("a node is wider than the symbol field".to_string());
    }
    let head = head_of(program)?;
    let mut skip = vec![false; nodes];
    if let Some((_, _, _, skipped)) = &head {
        for &n in skipped {
            skip[n] = true;
        }
    }
    let kept = head.as_ref().map_or(program.output, |h| h.0);
    let mut last_use = vec![0usize; nodes];
    for (index, node) in program.nodes.iter().enumerate().filter(|(i, _)| !skip[*i]) {
        for a in node.arguments() {
            last_use[a] = last_use[a].max(index);
        }
    }
    let mut maps: HashMap<(usize, bool), Arc<Map>> = HashMap::new();
    let mut map_of = |op: usize, transposed: bool| -> Result<Arc<Map>, String> {
        if let Some(map) = maps.get(&(op, transposed)) {
            return Ok(map.clone());
        }
        let map = Arc::new(Map::new(program.operators[op].clone(), transposed)?);
        maps.insert((op, transposed), map.clone());
        Ok(map)
    };
    let ones = vec![1.0; program.declarations.parameters];
    // `MPD_CERTIFY_TRACE` set: every node's mean half-width and center on stderr.
    let trace = std::env::var_os("MPD_CERTIFY_TRACE").is_some();
    let mut forms: Vec<Option<Vec<Row>>> = vec![None; nodes];
    let unbounded = || Array1::from_iter((0..rows).map(|r| if target.scores(r) { f64::INFINITY } else { 0.0 }));
    for index in 0..nodes {
        if skip[index] {
            continue;
        }
        let get = |n: usize| -> Result<&Vec<Row>, String> { forms[n].as_ref().ok_or_else(|| format!("node {n} is not evaluated")) };
        let value: Vec<Row> = match &program.nodes[index] {
            Node::Feature { slot, basis } => {
                let SlotValues::Tokens(tokens) = &inputs.slots[*slot] else {
                    return Err(format!("slot {slot} holds no tokens"));
                };
                let banded = program.bases[*basis].evaluate(&program.declarations, tokens).map_err(|e| e.to_string())?;
                (0..rows).map(|r| Row::exact(banded.values.row(r).to_owned(), banded.bands.row(r).to_owned())).collect()
            }
            Node::Raw { slot } => match free.iter().find(|f| f.slot == *slot) {
                Some(f) => (0..rows).map(|r| free_row(f, r)).collect(),
                None => {
                    let SlotValues::Raw(values) = &inputs.slots[*slot] else {
                        return Err(format!("slot {slot} holds no raw rows"));
                    };
                    (0..rows).map(|r| Row::exact(values.row(r).to_owned(), Array1::zeros(values.ncols()))).collect()
                }
            },
            Node::Constant { operator } => {
                map_of(*operator, false)?;
                let column = program.operators[*operator].column(0);
                (0..rows).map(|_| Row::exact(column.clone(), Array1::zeros(column.len()))).collect()
            }
            Node::Affine { terms, bias } => {
                let term_maps: Vec<Arc<Map>> = terms.iter().map(|(_, op)| map_of(*op, false)).collect::<Result<_, _>>()?;
                if let Some(b) = bias {
                    map_of(*b, false)?;
                }
                let bias = bias.map(|b| program.operators[b].column(0));
                let inputs_of: Vec<&Vec<Row>> = terms.iter().map(|(n, _)| get(*n)).collect::<Result<_, _>>()?;
                (0..rows)
                    .into_par_iter()
                    .map(|r| {
                        let parts: Vec<(&Row, &Map, usize)> =
                            terms.iter().zip(&inputs_of).zip(&term_maps).map(|(((n, _), x), m)| (&x[r], &**m, *n)).collect();
                        affine(&parts, bias.as_ref(), r, budget)
                    })
                    .collect()
            }
            Node::Transposed { input, operator } => {
                let map = map_of(*operator, true)?;
                let x = get(*input)?;
                (0..rows).into_par_iter().map(|r| affine(&[(&x[r], &*map, *input)], None, r, budget)).collect()
            }
            Node::Gain { input, coefficient } => {
                let (c, magnitude, operations) = coefficient.evaluate(&ones).map_err(|e| e.to_string())?;
                let extra = up(gamma(operations) * magnitude);
                get(*input)?.iter().map(|x| x.scaled(c, extra)).collect()
            }
            Node::Pointwise { input, laws } => {
                let interface = &interfaces[*input];
                let mut curves = Vec::with_capacity(interface.width());
                for (g, law) in laws.iter().enumerate() {
                    curves.extend(std::iter::repeat_n(Curve::Law(*law), interface.range(g).len()));
                }
                get(*input)?.par_iter().map(|x| curved(x, &curves, f64::NEG_INFINITY)).collect()
            }
            Node::Hadamard { left, right } => {
                let (l, r) = (get(*left)?, get(*right)?);
                l.par_iter().zip(r.par_iter()).map(|(a, b)| product(a, b)).collect()
            }
            Node::Bilinear { left, right, scale } => {
                let (l, r) = (get(*left)?, get(*right)?);
                l.par_iter().zip(r.par_iter()).map(|(a, b)| score(a, b, scale.value())).collect()
            }
            Node::Softmax { scores } => {
                let parts: Vec<&Vec<Row>> = scores.iter().map(|n| get(*n)).collect::<Result<_, _>>()?;
                (0..rows)
                    .into_par_iter()
                    .map(|r| {
                        let row_scores: Vec<Row> = parts.iter().map(|p| p[r].clone()).collect();
                        let weights = softmax(&row_scores, budget);
                        concat(&weights.iter().collect::<Vec<_>>())
                    })
                    .collect()
            }
            Node::Mix { weights, payloads } => {
                // A softmax's columns are convex weights when the mix reads each of them once.
                let convex = matches!(&program.nodes[*weights], Node::Softmax { scores } if scores.len() == payloads.len())
                    && payloads.iter().enumerate().all(|(i, (c, _))| *c == i);
                let w = get(*weights)?;
                let parts: Vec<(usize, &Vec<Row>)> = payloads.iter().map(|(c, n)| get(*n).map(|p| (*c, p))).collect::<Result<_, _>>()?;
                (0..rows)
                    .into_par_iter()
                    .map(|r| {
                        let alphas: Vec<Row> = parts.iter().map(|(c, _)| w[r].column(*c)).collect();
                        let values: Vec<&Row> = parts.iter().map(|(_, p)| &p[r]).collect();
                        mix(&alphas, &values, budget, convex)
                    })
                    .collect()
            }
            Node::Readout { input, basis } => {
                if !matches!(program.bases[*basis], Basis::Indicator { .. }) {
                    return Err("only an indicator readout is certified".to_string());
                }
                get(*input)?.clone()
            }
            Node::Concat { parts } => {
                let parts: Vec<&Vec<Row>> = parts.iter().map(|n| get(*n)).collect::<Result<_, _>>()?;
                (0..rows).map(|r| concat(&parts.iter().map(|p| &p[r]).collect::<Vec<_>>())).collect()
            }
            Node::RmsNorm { input, epsilon } => get(*input)?.par_iter().map(|x| rms_norm(x, *epsilon)).collect(),
            Node::Attend { query, key, value, scale, rotary, causal } => {
                let (q, k, v) = (get(*query)?, get(*key)?, get(*value)?);
                let positions: Vec<u32> = inputs.layout.as_ref().map(|l| l.position.clone()).unwrap_or_else(|| vec![0; rows]);
                let turn = |x: &Vec<Row>| -> Vec<Row> {
                    x.par_iter().enumerate().map(|(r, f)| rotary.map_or_else(|| f.clone(), |t| rotate(f, t, positions[r]))).collect()
                };
                let (q, k) = (turn(q), turn(k));
                let reads = visible(inputs, *causal)?;
                let c = scale.value();
                (0..rows)
                    .into_par_iter()
                    .map(|r| {
                        let scores: Vec<Row> = reads[r].iter().map(|&j| score(&q[r], &k[j], c)).collect();
                        let weights = softmax(&scores, budget);
                        let values: Vec<&Row> = reads[r].iter().map(|&j| &v[j]).collect();
                        mix(&weights, &values, budget, true)
                    })
                    .collect()
            }
            Node::Outer { .. } | Node::Param { .. } | Node::Call { .. } => {
                return Err(format!("node {index}: outer products and rule calls are not certified"));
            }
        };
        let mut value = value;
        value.par_iter_mut().for_each(|x| x.reduce(budget));
        if trace {
            // Where the relaxation widens: each node's mean half-width against its mean center.
            let (mut reach, mut size, mut symbols) = (0.0, 0.0, 0);
            for x in &value {
                let spread = x.spread();
                reach += (&spread + &x.radius).mean().unwrap_or(0.0) / rows as f64;
                size += x.center.mapv(f64::abs).mean().unwrap_or(0.0) / rows as f64;
                symbols = symbols.max(x.ids.len());
            }
            eprintln!("certify: node {index} {}: half-width {reach:.3e}, |center| {size:.3e}, {symbols} symbols", kind(&program.nodes[index]));
        }
        if value.iter().any(|x| !x.finite()) {
            return Ok(unbounded());
        }
        forms[index] = Some(value);
        for a in program.nodes[index].arguments() {
            if last_use[a] == index && a != kept {
                forms[a] = None;
            }
        }
    }
    let hidden = forms[kept].take().ok_or("the logits' node is not evaluated")?;
    let zero = Array1::<f64>::zeros(target.logits.ncols());
    let bounds: Vec<f64> = (0..rows)
        .into_par_iter()
        .map(|r| -> Result<f64, String> {
            if !target.scores(r) {
                return Ok(0.0);
            }
            let reference = target.logits.row(r);
            let top = reference.iter().enumerate().fold(0, |best, (i, &v)| if v > reference[best] { i } else { best });
            let (lower, upper) = match &head {
                Some((_, classes, bias, _)) => head_gaps(&hidden[r], classes.m(), bias.as_ref(), top),
                None => form_gaps(&hidden[r], top),
            };
            if lower.len() != reference.len() {
                return Err(format!("{} logits against a reference of {}", lower.len(), reference.len()));
            }
            if trace {
                let width = Zip::from(&upper).and(&lower).fold(0.0_f64, |m, &h, &l| m.max(h - l));
                let mean = (&upper - &lower).mean().unwrap_or(0.0);
                eprintln!("certify: row {r} logit gaps: widest {width:.3e}, mean width {mean:.3e}");
            }
            let radius = reference_radius.map_or(zero.view(), |m| m.row(r));
            let status = kl_supremum_over_gap_box(reference, radius, top, lower.view(), upper.view()).map_err(|e| e.to_string())?;
            Ok(status.upper_bound().unwrap_or(f64::INFINITY))
        })
        .collect::<Result<_, _>>()?;
    Ok(Array1::from(bounds))
}

/// Row `r` of a free slot: a lone column's interval as its radius, a wider block's as its symbol.
fn free_row(slot: &FreeSlot, r: usize) -> Row {
    let width = slot.lower.ncols();
    let mut sizes: HashMap<usize, usize> = HashMap::new();
    for &b in &slot.blocks {
        *sizes.entry(b).or_default() += 1;
    }
    let mut center = Array1::<f64>::zeros(width);
    let mut radius = Array1::<f64>::zeros(width);
    let mut symbols: Vec<(u64, Vec<(usize, f64)>)> = Vec::new();
    let mut shared: HashMap<usize, usize> = HashMap::new();
    for c in 0..width {
        let (mid, half) = centred(slot.lower[[r, c]], slot.upper[[r, c]]);
        center[c] = mid;
        let block = slot.blocks[c];
        if sizes[&block] > 1 && half > 0.0 {
            let k = *shared.entry(block).or_insert_with(|| {
                symbols.push((gate(slot.slot, r, block), Vec::new()));
                symbols.len() - 1
            });
            symbols[k].1.push((c, half));
        } else {
            radius[c] = half;
        }
    }
    symbols.sort_unstable_by_key(|s| s.0);
    let mut coef = Array2::<f64>::zeros((symbols.len(), width));
    for (k, (_, columns)) in symbols.iter().enumerate() {
        for &(c, half) in columns {
            coef[[k, c]] = half;
        }
    }
    Row { center, ids: symbols.into_iter().map(|s| s.0).collect(), coef, radius }
}

/// Per site, each block's gate interval (rows × blocks).
#[derive(Clone, Debug)]
pub struct Gates {
    pub lower: Vec<Array2<f64>>,
    pub upper: Vec<Array2<f64>>,
}

impl Gates {
    /// The box claim at `masks`: an on gate (positive) fixed at its value, an off gate anywhere in `[0, 1]`.
    pub fn claim(masks: &[Array2<f64>]) -> Self {
        Self {
            lower: masks.iter().map(|m| m.mapv(|x| if x > 0.0 { x } else { 0.0 })).collect(),
            upper: masks.iter().map(|m| m.mapv(|x| if x > 0.0 { x } else { 1.0 })).collect(),
        }
    }

    /// Every gate at the middle of its interval.
    pub fn center(&self) -> Vec<Array2<f64>> {
        self.lower.iter().zip(&self.upper).map(|(l, u)| (l + u) * 0.5).collect()
    }
}

/// Per input, an upper bound on `KL(target ‖ masked)` over every gate setting in `gates` (module
/// note), the reference within `reference_radius` of the target's logits, at most `budget` symbols
/// per input. A program with a head is refused.
pub fn certify(
    masked: &Masked,
    base: &FamilyInputs,
    target: &Target,
    reference_radius: Option<&Array2<f64>>,
    gates: &Gates,
    budget: usize,
) -> Result<Array1<f64>, String> {
    if masked.head.is_some() {
        return Err("a program with a head is not certified".to_string());
    }
    let free: Vec<FreeSlot> = (0..masked.sites.len())
        .map(|k| FreeSlot {
            slot: masked.slots[k],
            lower: masked.expand(k, &gates.lower[k]),
            upper: masked.expand(k, &gates.upper[k]),
            blocks: masked.ranks(k).iter().enumerate().flat_map(|(b, r)| std::iter::repeat_n(b, *r)).collect(),
        })
        .collect();
    let inputs = masked.family(base, &gates.lower);
    certify_program(&masked.program, &inputs, &free, target, reference_radius, budget)
}

/// A branched certificate: per input the largest bound over the leaves, and how many leaves were bounded.
#[derive(Clone, Debug)]
pub struct Branched {
    pub kl: Array1<f64>,
    pub root: Array1<f64>,
    pub leaves: usize,
}

/// [`certify`] with the box split at most `leaves − 1` times (module note, "Branching"): each split
/// takes the leaf and input of the largest bound and halves the gate of largest `|∂KL/∂g|` times
/// width at that leaf's center, for that input's KL.
pub fn certify_branching(
    masked: &Masked,
    base: &FamilyInputs,
    target: &Target,
    reference_radius: Option<&Array2<f64>>,
    gates: Gates,
    budget: usize,
    leaves: usize,
) -> Result<Branched, String> {
    let root = certify(masked, base, target, reference_radius, &gates, budget)?;
    let mut open = vec![(gates, root.clone())];
    let mut spent = 1;
    while spent + 2 <= leaves {
        let (leaf, row, _) = open
            .iter()
            .enumerate()
            .flat_map(|(l, (_, b))| b.iter().enumerate().map(move |(r, v)| (l, r, *v)))
            .fold((0, 0, f64::NEG_INFINITY), |best, c| if c.2 > best.2 { c } else { best });
        let center = open[leaf].0.center();
        let family = masked.family(base, &center);
        let (_, trace, cotangent) = forward(masked, &family, target)?;
        let mut focused = Array2::<f64>::zeros(cotangent.dim());
        focused.row_mut(row).assign(&cotangent.row(row));
        let gradients = mask_gradients(masked, &family, &trace, focused)?;
        let (lower, upper) = (&open[leaf].0.lower, &open[leaf].0.upper);
        let mut pick: Option<(usize, usize, usize)> = None;
        let mut best = 0.0_f64;
        for (k, g) in gradients.iter().enumerate() {
            for ((r, b), &d) in g.indexed_iter() {
                let width = upper[k][[r, b]] - lower[k][[r, b]];
                let weight = d.abs() * width;
                if width > 0.0 && weight > best {
                    best = weight;
                    pick = Some((k, r, b));
                }
            }
        }
        let Some((k, r, b)) = pick else { break };
        let (gates, _) = open.swap_remove(leaf);
        let middle = 0.5 * (gates.lower[k][[r, b]] + gates.upper[k][[r, b]]);
        let mut left = gates.clone();
        left.upper[k][[r, b]] = middle;
        let mut right = gates;
        right.lower[k][[r, b]] = middle;
        for half in [left, right] {
            let bound = certify(masked, base, target, reference_radius, &half, budget)?;
            open.push((half, bound));
        }
        spent += 2;
    }
    let mut kl = Array1::<f64>::zeros(base.rows);
    for (_, bound) in &open {
        Zip::from(&mut kl).and(bound).for_each(|k, &b| *k = k.max(b));
    }
    Ok(Branched { kl, root, leaves: spent })
}

/// A deterministic generator for the adversary's starts.
struct SplitMix(u64);

impl SplitMix {
    fn next(&mut self) -> f64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^= z >> 31;
        (z >> 11) as f64 / (1u64 << 53) as f64
    }
}

/// The largest KL found per input inside `gates`: every point visited by `restarts` sign-ascent runs
/// of `steps` steps on the free gates (those whose interval is wider than a point). The runs start
/// from every gate at its lower end (for the box claim, the masks themselves), at its upper end, at
/// its middle, then at uniform draws; the step shrinks linearly from half of each gate's width to
/// `1/(2 steps)` of it. The ascent climbs the KL of input `focus` alone when given, the total
/// otherwise. A lower bound on the box's worst case.
pub fn adversary(
    masked: &Masked,
    base: &FamilyInputs,
    target: &Target,
    gates: &Gates,
    focus: Option<usize>,
    steps: usize,
    restarts: usize,
    seed: u64,
) -> Result<Array1<f64>, String> {
    let mut rng = SplitMix(seed);
    let mut best = Array1::<f64>::from_elem(base.rows, f64::NEG_INFINITY);
    for restart in 0..restarts.max(1) {
        let mut point: Vec<Array2<f64>> = gates
            .lower
            .iter()
            .zip(&gates.upper)
            .map(|(l, u)| {
                Zip::from(l).and(u).map_collect(|&l, &u| match restart {
                    0 => l,
                    1 => u,
                    2 => 0.5 * (l + u),
                    _ => l + (u - l) * rng.next(),
                })
            })
            .collect();
        for step in 0..=steps {
            let family = masked.family(base, &point);
            let (kl, trace, cotangent) = forward(masked, &family, target)?;
            Zip::from(&mut best).and(&kl).for_each(|b, &v| *b = b.max(v));
            if step == steps {
                break;
            }
            let cotangent = match focus {
                Some(row) => {
                    let mut focused = Array2::<f64>::zeros(cotangent.dim());
                    focused.row_mut(row).assign(&cotangent.row(row));
                    focused
                }
                None => cotangent,
            };
            let ascent = mask_gradients(masked, &family, &trace, cotangent)?;
            let rate = 0.5 - (0.5 - 0.5 / steps as f64) * step as f64 / steps.max(2).saturating_sub(1) as f64;
            for (((g, l), u), a) in point.iter_mut().zip(&gates.lower).zip(&gates.upper).zip(&ascent) {
                Zip::from(g).and(l).and(u).and(a).for_each(|g, &l, &u, &a| {
                    if u > l {
                        *g = (*g + rate * (u - l) * a.signum()).clamp(l, u);
                    }
                });
            }
        }
    }
    Ok(best)
}
