//! Per-input sparse coding of a site's real output by its subcomponents' real contributions, with
//! a certificate of how close each input's code is to its optimum (#2951).
//!
//! A site `y = W x` with library `W = Σ_c u_c v_cᵀ`, gated in blocks (contiguous runs of columns,
//! one gate per block `b` with columns `J_b`), runs on input `t` with its real read `x_t`; block
//! `b`'s real contribution is `Z_tb = Σ_{c ∈ J_b} z_tc u_c` with `z_tc = v_c · x_t`. Input `t`'s code
//! for the blocks `m ∈ {0, 1}^B` it runs is
//!
//! ```text
//! f_t(m) = Σ_b bits_b m_b + κ ‖y_t − Σ_b m_b Z_tb‖²_F,      κ = n / (2 ln 2),
//! ```
//!
//! the blocks' description bits plus the second-order KL bits of what they leave of the site's real
//! output, in the metric `F`: the site's mean output Fisher ([`Metric::Mean`], input-independent,
//! so a run-time selection can use it), or each input's own from its sampled-label gradients
//! ([`Metric::PerRow`], a training proxy). Expanded, `f_t(m) = κ yᵀFy + Σ_b ℓ_b m_b + κ mᵀQm` with
//! `ℓ_b = bits_b − 2κ Z_tbᵀ F y_t` and `Q_bb' = Z_tbᵀ F Z_tb'`, positive semidefinite.
//!
//! # The certificate
//!
//! The same expression on the box `[0, 1]^B` is convex and equals `f_t` at every vertex, so its
//! minimum is a lower bound on the best code; for any box point `m`, convexity gives the bound
//! `g(m) + Σ_b min(∂_b g · (0 − m_b), ∂_b g · (1 − m_b))` (the linearisation's minimum over the box),
//! which is exact at the relaxation's optimum. The relaxation is solved to optimality by a working
//! set of blocks (those any KKT condition violates are added, the set's subproblem solved by exact
//! coordinate minimisation), its optimum rounded and then improved by exact single flips, which
//! gives an upper bound: the input's code. An input whose bounds are more than a bit apart is
//! branched on its most fractional block, best bound first, within `nodes` nodes; each input
//! returns its blocks and both bounds, so a code within a bit of its optimum is certified so.

use gam_linalg::faer_ndarray::{fast_ab, fast_abt};
use ndarray::{Array1, Array2, ArrayView2, s};
use rayon::prelude::*;
use std::collections::HashMap;

/// The metric of an input's residual (module note).
pub enum Metric<'a> {
    /// The site's mean output Fisher, `d_out × d_out`.
    Mean(ArrayView2<'a, f64>),
    /// Per draw `k`, every input's gradient at the written value (inputs × d_out): input `t`'s
    /// metric is `F_t = mean_k g_tk g_tkᵀ`.
    PerRow(&'a [ArrayView2<'a, f64>]),
}

/// One site's coding problem over its inputs (module note).
pub struct Problem<'a> {
    /// The reads `x_t`, inputs × d_in.
    pub reads: ArrayView2<'a, f64>,
    /// The site's real outputs `y_t = W x_t`, inputs × d_out.
    pub targets: ArrayView2<'a, f64>,
    /// The library: `v` pieces × d_in, `u` pieces × d_out.
    pub v: ArrayView2<'a, f64>,
    pub u: ArrayView2<'a, f64>,
    /// The blocks' sizes (contiguous runs of pieces, in order); all ones for rank-one gates.
    pub ranks: &'a [usize],
    /// Each block's description bits.
    pub bits: &'a [f64],
    pub metric: Metric<'a>,
    /// `n` of the code's `n KL / ln 2`.
    pub observations: f64,
    /// Branch-and-bound nodes per input whose bounds are more than a bit apart.
    pub nodes: usize,
}

/// Every input's code (module note).
pub struct Coding {
    /// Per input, its blocks on (ascending).
    pub sets: Vec<Vec<u32>>,
    /// Per input, its code `f_t` at those blocks, and a lower bound on its best code.
    pub upper: Array1<f64>,
    pub lower: Array1<f64>,
    /// Per input, what its blocks leave of the site's real output, `y_t − Σ_on Z_tb`.
    pub residual: Array2<f64>,
}

/// Inputs coded at once (bounds the per-input products' memory).
const CHUNK: usize = 512;

/// A bound gap no larger than this many bits is certified close enough.
const BIT: f64 = 1.0;

/// One input's quadratic over its blocks (module note).
struct Row<'p> {
    /// `z_tc` per piece.
    z: Vec<f64>,
    /// `ℓ_b` per block.
    linear: Vec<f64>,
    /// `κ yᵀFy`.
    constant: f64,
    /// `Q_bb` per block.
    diagonal: Vec<f64>,
    kappa: f64,
    starts: &'p [usize],
    /// The pieces' Gram in the metric, `K = U F Uᵀ` (mean metric), or the input's per-draw
    /// projections `p_k = U g_tk` (draws × pieces).
    operator: Operator<'p>,
}

enum Operator<'p> {
    Mean(&'p Array2<f64>),
    PerRow(Array2<f64>),
}

/// A relaxed solution on a box: the point and its certified lower bound.
#[derive(Clone)]
struct Relaxed {
    m: Vec<f64>,
    lower: f64,
}

impl Row<'_> {
    fn blocks(&self) -> usize {
        self.starts.len() - 1
    }

    /// `Q e_b` over the blocks.
    fn column(&self, b: usize) -> Vec<f64> {
        let pieces = self.z.len();
        let mut t = vec![0.0; pieces];
        match &self.operator {
            Operator::Mean(k) => {
                for c in self.starts[b]..self.starts[b + 1] {
                    let zc = self.z[c];
                    if zc != 0.0 {
                        for (tc, kc) in t.iter_mut().zip(k.row(c).iter()) {
                            *tc += zc * kc;
                        }
                    }
                }
            }
            Operator::PerRow(p) => {
                let draws = p.nrows() as f64;
                for row in p.outer_iter() {
                    let along: f64 = (self.starts[b]..self.starts[b + 1]).map(|c| row[c] * self.z[c]).sum::<f64>() / draws;
                    for (tc, pc) in t.iter_mut().zip(row.iter()) {
                        *tc += along * pc;
                    }
                }
            }
        }
        (0..self.blocks()).map(|b2| (self.starts[b2]..self.starts[b2 + 1]).map(|c| self.z[c] * t[c]).sum()).collect()
    }

    fn value(&self, m: &[f64], qm: &[f64]) -> f64 {
        self.constant + m.iter().zip(&self.linear).map(|(a, l)| a * l).sum::<f64>() + self.kappa * m.iter().zip(qm).map(|(a, q)| a * q).sum::<f64>()
    }

    fn gradient(&self, b: usize, qm: &[f64]) -> f64 {
        self.linear[b] + 2.0 * self.kappa * qm[b]
    }

    /// The relaxation's optimum on the box `[lo, hi]` from `start` (module note).
    fn relax(&self, lo: &[f64], hi: &[f64], start: &[f64], columns: &mut HashMap<usize, Vec<f64>>) -> Relaxed {
        let blocks = self.blocks();
        let mut m: Vec<f64> = (0..blocks).map(|b| start[b].clamp(lo[b], hi[b])).collect();
        let mut qm = vec![0.0; blocks];
        for b in 0..blocks {
            if m[b] != 0.0 {
                let column = columns.entry(b).or_insert_with(|| self.column(b));
                for (q, c) in qm.iter_mut().zip(column.iter()) {
                    *q += m[b] * c;
                }
            }
        }
        let mut working: Vec<usize> = (0..blocks).filter(|b| m[*b] != 0.0 && lo[*b] < hi[*b]).collect();
        loop {
            // Exact coordinate minimisation over the working set until it is stationary.
            for _ in 0..1000 {
                let mut moved = 0.0f64;
                for &b in &working {
                    let g = self.gradient(b, &qm);
                    let curvature = 2.0 * self.kappa * self.diagonal[b];
                    let next = if curvature > 0.0 { (m[b] - g / curvature).clamp(lo[b], hi[b]) } else if g > 0.0 { lo[b] } else if g < 0.0 { hi[b] } else { m[b] };
                    let step = next - m[b];
                    if step != 0.0 {
                        m[b] = next;
                        let column = &columns[&b];
                        for (q, c) in qm.iter_mut().zip(column.iter()) {
                            *q += step * c;
                        }
                        moved = moved.max(step.abs());
                    }
                }
                if moved <= 1e-12 {
                    break;
                }
            }
            // Coordinates outside it that violate a KKT condition, most violating first; the set
            // at most doubles per round, so it stays near the solution's support.
            let mut violators: Vec<(f64, usize)> = (0..blocks)
                .filter(|b| lo[*b] < hi[*b] && !working.contains(b))
                .filter_map(|b| {
                    let g = self.gradient(b, &qm);
                    ((g < 0.0 && m[b] < hi[b]) || (g > 0.0 && m[b] > lo[b])).then_some((g.abs(), b))
                })
                .collect();
            if violators.is_empty() {
                break;
            }
            violators.sort_by(|a, b| b.0.total_cmp(&a.0));
            let room = working.len().max(16);
            working.extend(violators.iter().take(room).map(|(_, b)| *b));
            for &b in &working {
                columns.entry(b).or_insert_with(|| self.column(b));
            }
        }
        let value = self.value(&m, &qm);
        let slack: f64 = (0..blocks)
            .map(|b| {
                let g = self.gradient(b, &qm);
                (g * (lo[b] - m[b])).min(g * (hi[b] - m[b]))
            })
            .sum();
        Relaxed { m, lower: value + slack.min(0.0) }
    }

    /// `relaxed` rounded on the box `[lo, hi]`, then improved by exact single flips until none
    /// lowers the code: the blocks on and their code.
    fn round(&self, relaxed: &Relaxed, lo: &[f64], hi: &[f64], columns: &mut HashMap<usize, Vec<f64>>) -> (Vec<bool>, f64) {
        let blocks = self.blocks();
        let mut on: Vec<bool> = (0..blocks).map(|b| if lo[b] == hi[b] { hi[b] > 0.5 } else { relaxed.m[b] > 0.5 }).collect();
        let mut qm = vec![0.0; blocks];
        for b in (0..blocks).filter(|b| on[*b]) {
            let column = columns.entry(b).or_insert_with(|| self.column(b));
            for (q, c) in qm.iter_mut().zip(column.iter()) {
                *q += c;
            }
        }
        loop {
            let mut best: Option<(f64, usize)> = None;
            for b in (0..blocks).filter(|b| lo[*b] < hi[*b]) {
                let delta = if on[b] {
                    -self.linear[b] + self.kappa * (self.diagonal[b] - 2.0 * qm[b])
                } else {
                    self.linear[b] + self.kappa * (self.diagonal[b] + 2.0 * qm[b])
                };
                if delta < 0.0 && best.is_none_or(|(d, _)| delta < d) {
                    best = Some((delta, b));
                }
            }
            let Some((_, b)) = best else { break };
            let sign = if on[b] { -1.0 } else { 1.0 };
            on[b] = !on[b];
            let column = columns.entry(b).or_insert_with(|| self.column(b));
            for (q, c) in qm.iter_mut().zip(column.iter()) {
                *q += sign * c;
            }
        }
        let m: Vec<f64> = on.iter().map(|o| if *o { 1.0 } else { 0.0 }).collect();
        let value = self.value(&m, &qm);
        (on, value)
    }

    /// The input's code (module note): its blocks on, its code and a lower bound on its best.
    fn code(&self, start: &[f64], nodes: usize) -> (Vec<bool>, f64, f64) {
        let blocks = self.blocks();
        let mut columns: HashMap<usize, Vec<f64>> = HashMap::new();
        let (lo, hi) = (vec![0.0; blocks], vec![1.0; blocks]);
        let root = self.relax(&lo, &hi, start, &mut columns);
        let (mut best, mut upper) = self.round(&root, &lo, &hi, &mut columns);
        if upper - root.lower <= BIT || nodes == 0 {
            return (best, upper, root.lower.min(upper));
        }
        // Best bound first: open nodes as (box, relaxation); the least lower bound is the input's.
        let mut open: Vec<(Vec<f64>, Vec<f64>, Relaxed)> = vec![(lo, hi, root)];
        let mut floor = f64::INFINITY;
        let mut explored = 0;
        while let Some(index) = (0..open.len()).min_by(|a, b| open[*a].2.lower.total_cmp(&open[*b].2.lower)) {
            if open[index].2.lower >= upper - BIT || explored >= nodes {
                break;
            }
            let (lo, hi, node) = open.swap_remove(index);
            explored += 1;
            let Some(j) = (0..blocks).filter(|b| lo[*b] < hi[*b]).max_by(|a, b| (0.5 - (node.m[*a] - 0.5).abs()).total_cmp(&(0.5 - (node.m[*b] - 0.5).abs()))) else {
                floor = floor.min(node.lower);
                continue;
            };
            if node.m[j] == 0.0 || node.m[j] == 1.0 {
                // Integral: the relaxation's optimum is a vertex, its own code.
                floor = floor.min(node.lower);
                continue;
            }
            for fixed in [0.0, 1.0] {
                let (mut child_lo, mut child_hi) = (lo.clone(), hi.clone());
                child_lo[j] = fixed;
                child_hi[j] = fixed;
                let child = self.relax(&child_lo, &child_hi, &node.m, &mut columns);
                let (on, value) = self.round(&child, &child_lo, &child_hi, &mut columns);
                if value < upper {
                    (best, upper) = (on, value);
                }
                if child.lower < upper - BIT {
                    open.push((child_lo, child_hi, child));
                } else {
                    floor = floor.min(child.lower);
                }
            }
        }
        let lower = open.iter().map(|(_, _, r)| r.lower).fold(floor, f64::min).min(upper);
        (best, upper, lower)
    }
}

/// Code every input of `problem` (module note); `warm`, per input its blocks on to start from.
pub fn code_site(problem: &Problem<'_>, warm: Option<&[Vec<u32>]>) -> Result<Coding, String> {
    let (rows, d_in) = problem.reads.dim();
    let pieces = problem.v.nrows();
    let d_out = problem.u.ncols();
    let blocks = problem.ranks.len();
    if problem.v.ncols() != d_in || problem.u.nrows() != pieces || problem.targets.dim() != (rows, d_out) {
        return Err(format!("sparse code: reads {rows}×{d_in}, targets {:?}, library {:?} and {:?} disagree", problem.targets.dim(), problem.v.dim(), problem.u.dim()));
    }
    if problem.ranks.contains(&0) || problem.ranks.iter().sum::<usize>() != pieces || problem.bits.len() != blocks {
        return Err(format!("sparse code: blocks {:?} with {} bits do not partition {pieces} pieces", problem.ranks, problem.bits.len()));
    }
    if warm.is_some_and(|w| w.len() != rows) {
        return Err("sparse code: a warm start per input".to_string());
    }
    let starts: Vec<usize> = std::iter::once(0)
        .chain(problem.ranks.iter().scan(0, |a, r| {
            *a += r;
            Some(*a)
        }))
        .collect();
    let kappa = problem.observations / (2.0 * std::f64::consts::LN_2);
    let u = problem.u.to_owned();
    // The mean metric's pieces Gram `K = U F Uᵀ` and `F Uᵀ`, formed once.
    let mean = match &problem.metric {
        Metric::Mean(f) => {
            if f.dim() != (d_out, d_out) {
                return Err(format!("sparse code: a {:?} metric for {d_out} writes", f.dim()));
            }
            let fu = fast_abt(&f.to_owned(), &u);
            Some((fast_ab(&u, &fu), fu))
        }
        Metric::PerRow(draws) => {
            if draws.is_empty() || draws.iter().any(|g| g.dim() != (rows, d_out)) {
                return Err("sparse code: per-input gradients of every input's writes".to_string());
            }
            None
        }
    };
    let mut sets = Vec::with_capacity(rows);
    let (mut upper, mut lower) = (Vec::with_capacity(rows), Vec::with_capacity(rows));
    let mut residual = Array2::<f64>::zeros((rows, d_out));
    for first in (0..rows).step_by(CHUNK) {
        let last = (first + CHUNK).min(rows);
        let reads = problem.reads.slice(s![first..last, ..]).to_owned();
        let targets = problem.targets.slice(s![first..last, ..]).to_owned();
        let z = fast_abt(&reads, &problem.v.to_owned());
        // Per input: `U F y` and `yᵀ F y` (mean metric), or the per-draw projections.
        let (weights, constants, projections): (Array2<f64>, Vec<f64>, Vec<Array2<f64>>) = match (&problem.metric, &mean) {
            (Metric::Mean(f), Some((_, fu))) => {
                let weights = fast_ab(&targets, fu);
                let fy = fast_abt(&targets, &f.to_owned());
                let constants = targets.outer_iter().zip(fy.outer_iter()).map(|(y, g)| kappa * y.dot(&g)).collect();
                (weights, constants, Vec::new())
            }
            (Metric::PerRow(draws), _) => {
                let k = draws.len() as f64;
                let projections: Vec<Array2<f64>> = draws.iter().map(|g| fast_abt(&g.slice(s![first..last, ..]).to_owned(), &u)).collect();
                let along: Vec<Vec<f64>> =
                    draws.iter().map(|g| g.slice(s![first..last, ..]).outer_iter().zip(targets.outer_iter()).map(|(gr, y)| gr.dot(&y)).collect()).collect();
                let mut weights = Array2::<f64>::zeros((last - first, pieces));
                for (p, a) in projections.iter().zip(&along) {
                    for (mut w, (pr, ar)) in weights.outer_iter_mut().zip(p.outer_iter().zip(a)) {
                        w.scaled_add(ar / k, &pr);
                    }
                }
                let constants = (0..last - first).map(|r| kappa * along.iter().map(|a| a[r] * a[r]).sum::<f64>() / k).collect();
                (weights, constants, projections)
            }
            _ => return Err("sparse code: a mean metric without its Gram".to_string()),
        };
        let coded: Vec<(Vec<bool>, f64, f64)> = (0..last - first)
            .into_par_iter()
            .map(|r| {
                let zr = z.row(r).to_vec();
                let operator = match &mean {
                    Some((k, _)) => Operator::Mean(k),
                    None => Operator::PerRow(Array2::from_shape_fn((projections.len(), pieces), |(j, c)| projections[j][[r, c]])),
                };
                let linear: Vec<f64> = (0..blocks)
                    .map(|b| problem.bits[b] - 2.0 * kappa * (starts[b]..starts[b + 1]).map(|c| zr[c] * weights[[r, c]]).sum::<f64>())
                    .collect();
                let diagonal: Vec<f64> = (0..blocks)
                    .map(|b| {
                        let block = starts[b]..starts[b + 1];
                        match &operator {
                            Operator::Mean(k) => block.clone().map(|c| block.clone().map(|c2| zr[c] * zr[c2] * k[[c, c2]]).sum::<f64>()).sum::<f64>().max(0.0),
                            Operator::PerRow(p) => {
                                p.outer_iter().map(|row| block.clone().map(|c| row[c] * zr[c]).sum::<f64>().powi(2)).sum::<f64>() / p.nrows() as f64
                            }
                        }
                    })
                    .collect();
                let row = Row { z: zr, linear, constant: constants[r], diagonal, kappa, starts: &starts, operator };
                let start: Vec<f64> = match warm {
                    Some(w) => {
                        let mut m = vec![0.0; blocks];
                        for &b in &w[first + r] {
                            if let Some(slot) = m.get_mut(b as usize) {
                                *slot = 1.0;
                            }
                        }
                        m
                    }
                    None => vec![0.0; blocks],
                };
                row.code(&start, problem.nodes)
            })
            .collect();
        // What each input's blocks leave of its real output.
        let mut gated = z.clone();
        for (r, (on, _, _)) in coded.iter().enumerate() {
            for b in (0..blocks).filter(|b| !on[*b]) {
                gated.slice_mut(s![r, starts[b]..starts[b + 1]]).fill(0.0);
            }
        }
        residual.slice_mut(s![first..last, ..]).assign(&(&targets - &fast_ab(&gated, &u)));
        for (on, up, low) in coded {
            sets.push((0..blocks as u32).filter(|b| on[*b as usize]).collect());
            upper.push(up);
            lower.push(low);
        }
    }
    Ok(Coding { sets, upper: Array1::from(upper), lower: Array1::from(lower), residual })
}
