//! Shared factors fitted in the code's own metric (#2951).
//!
//! # The model
//!
//! A *component* is a set of dense operators read as affine terms, together with the nodes they
//! read, closed under "reads": every operator reading one of the nodes, and every node one of the
//! operators reads (a tied operator joins all its arguments). The component's operators `A_s`
//! are factored through one basis of the read space,
//!
//! ```text
//! A_s x  ≈  L_s (R x),      z = R x  (r factor coordinates, one shared node per read node),
//! ```
//!
//! so the basis `R` is sent once and bound to every reader, and each reader keeps only its
//! coefficients `L_s`. Units of a pointwise layer are rows of the operator that writes their
//! pre-activations (the read side) and columns of the operator that reads their activations
//! (the write side: there the units are the read space and `R` is the per-unit write
//! coefficients). A rule is a set of units with one support on the factor coordinates; its
//! members' coefficients are the per-member data.
//!
//! # The metric
//!
//! A change `δA` moves the data code `n Σ KL/ln 2` by `(n/2 ln 2) Σ_x δyᵀ H_x δy` to second order,
//! with `δy = δA x̃` (`x̃ = x − x̄`; the mean is folded into the bias exactly) and `H_x` the
//! Gauss–Newton curvature `Jᵀ(diag q − q qᵀ)J` of the logits at the reading node. Its Kronecker
//! factorisation `n N tr(δA Σ δAᵀ G)`, with `Σ` the read nodes' pooled covariance and `G` the
//! summed per-use curvature `(1/N) Σ_x J_xᵀ H_x J_x` (exact, from one forward tangent per output
//! coordinate), makes `Ã = √(nN) G^{1/2} A Σ^{1/2}` the matrix whose squared error is twice the
//! data bits in nats. Every operator of a component is stacked into one `Ã`; its singular value
//! decomposition gives the basis.
//!
//! # What the fit decides, and how
//!
//! * **Rank.** On `Ã` an entry quantised at the rate-distortion optimum has unit distortion, so
//!   a direction pays when its energy exceeds the factor entries it needs: rank `r*` keeps
//!   `σ_i² > M + d` (`M` stacked rows, `d` read dimensions). The neighbouring ranks `r*/2` and
//!   `2r*`, and the exact rank on the family (singular values above the band), are proposed
//!   too; the decoded code and the contract decide.
//! * **Precision.** Each factor's lattice is the step that minimises its precision bits plus the
//!   data bits it causes, `Δ² = 12 m / (n N tr G · tr(R Σ Rᵀ))` for `L` and
//!   `Δ² = 12 m / (n N tr Σ · Σ_s tr(L_sᵀ G_s L_s))` for `R`, both exact in the fitted factors:
//!   the basis is orthonormal in the whitened read space, so `tr(R Σ Rᵀ) = r`.
//! * **Gauge.** `L R = (L Qᵀ)(Q R)` for every rotation `Q` of the factor space: the factorisation
//!   is fixed by the rotation whose per-member coefficients `L` have the shortest code,
//!   `Σ log₂(1 + |v|/Δ)`, by Jacobi sweeps until a sweep saves less than a bit. The shared basis
//!   is sent once, and a rotation moves its length only through the rounding of its entries. A coefficient that rounds to
//!   zero on its lattice leaves its block absent. Units that read one plane of a shared basis come
//!   out reading one pair of coordinates: the sparsity is found, not declared.
//! * **Support.** Each rank is also proposed sparse: a coefficient is kept only when dropping it
//!   costs more data bits, `u²/(2 ln 2)` at its normalised size `u`, than coding it costs,
//!   `log₂(1 + |u|/√12) + 1`. The rules this leaves are the support's components.

//!
//! # When the rules are identified
//!
//! **Proposition.** Let the units' whitened reads be vectors `v_n ∈ ℝ^r`. Call an orthogonal
//! decomposition `ℝ^r = ⊕_k T_k` *compatible* when every `v_n` lies in one `T_k`. There is a
//! unique finest compatible decomposition, and the partition of the units by the co-support
//! components of any orthonormal coefficient basis that realises it is that decomposition's
//! partition; two such bases differ by an orthogonal map inside each `T_k`, a permutation and
//! signs, which is exactly the gauge [`super::identify::gauge`] reports for factor coordinates.
//!
//! *Proof.* If `⊕ T_k` and `⊕ U_l` are compatible, so is `⊕_{k,l} (T_k ∩ U_l)` restricted to its
//! non-zero terms together with the orthogonal complement of their sum: each `v_n` lies in some
//! `T_k` and some `U_l`, hence in `T_k ∩ U_l`; distinct intersections are orthogonal because
//! either their `T` or their `U` factors are. Compatible decompositions are therefore closed
//! under common refinement and a finest one exists and is unique. An orthonormal basis whose
//! co-support components are `C_1, …, C_K` yields the compatible decomposition
//! `T_k = span{q_i : i ∈ C_k}`; it realises the finest one when that is its decomposition, and
//! the units in `T_k` are exactly the units of component `C_k`. Two bases realising the same
//! decomposition span the same `T_k` blockwise, so they differ by an element of `∏_k O(T_k)`
//! up to the order and signs of their vectors. ∎
//!
//! The Jacobi rotation of the gauge searches for such a basis; the planted test checks that it
//! reaches the finest decomposition when the reads lie exactly on orthogonal planes.

use super::dense::{eigh, svd};
use super::engine::{EngineError, Edit, Exactness, Primitive, Proposal, SearchContext};
use super::derivatives::vjp;
use super::fit::ProposalKind;
use super::operator_program::{
    Interface, LabelKind, Node, Operator, OperatorBody, OperatorProgram, Provenance, Trace, remap_node,
};
use super::precision::DeclaredPrecision;
use gam_linalg::roundoff::SymmetricAssembly;
use ndarray::{Array1, Array2, Axis, s};
use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;

/// Shared factors fitted in the code's metric (module note).
///
/// The readers' curvature is the costly part of a fit (one reverse pass per sketched class), so it
/// is kept per component, keyed by the component's own operators: it is measured again only when
/// one of them changes. A change elsewhere leaves it a proposal metric; the contract certifies.
#[derive(Default)]
pub struct Factors {
    curvature: std::cell::RefCell<BTreeMap<Vec<usize>, BTreeMap<usize, Array2<f64>>>>,
}

fn refuse(message: impl Into<String>) -> EngineError {
    EngineError::Primitive(message.into())
}

/// A component: operators and the nodes they read (module note).
struct Component {
    operators: Vec<usize>,
    arguments: Vec<usize>,
    /// Every affine node holding one of the operators as a term.
    readers: Vec<usize>,
}

fn is_factor(interface: &Interface) -> bool {
    interface.groups().iter().any(|g| g.label.kind == LabelKind::Factor)
}

/// The components of `program` (module note): dense, fully present operators that are only ever
/// affine terms, outside rules, and not themselves factors.
fn components(program: &OperatorProgram) -> Vec<Component> {
    let count = program.operators.len();
    let mut eligible: Vec<bool> = program
        .operators
        .iter()
        .map(|op| {
            matches!(&op.body, OperatorBody::Dense { present, .. } if present.iter().all(|k| *k))
                && !is_factor(&op.rows)
                && !is_factor(&op.cols)
                && op.rows.width() >= 2
                && op.cols.width() >= 2
        })
        .collect();
    let mut used = vec![false; count];
    for node in &program.nodes {
        match node {
            Node::Affine { terms, bias } => {
                for (_, op) in terms {
                    used[*op] = true;
                }
                if let Some(b) = bias {
                    eligible[*b] = false;
                }
            }
            other => {
                for op in other.operators() {
                    eligible[op] = false;
                }
            }
        }
    }
    for rule in &program.rules {
        for node in &rule.nodes {
            for op in node.operators() {
                eligible[op] = false;
            }
        }
    }
    // Union-find over operators and argument nodes (offset by the operator count).
    let mut parent: Vec<usize> = (0..count + program.nodes.len()).collect();
    fn find(parent: &mut [usize], mut x: usize) -> usize {
        while parent[x] != x {
            parent[x] = parent[parent[x]];
            x = parent[x];
        }
        x
    }
    let mut readers: BTreeMap<usize, BTreeSet<usize>> = BTreeMap::new();
    for (index, node) in program.nodes.iter().enumerate() {
        let Node::Affine { terms, .. } = node else { continue };
        for (argument, op) in terms {
            if eligible[*op] && used[*op] {
                let (a, b) = (find(&mut parent, *op), find(&mut parent, count + argument));
                parent[a] = b;
                readers.entry(*op).or_default().insert(index);
            }
        }
    }
    let mut groups: BTreeMap<usize, Component> = BTreeMap::new();
    for op in readers.keys().copied().collect::<Vec<_>>() {
        let root = find(&mut parent, op);
        let entry = groups.entry(root).or_insert(Component { operators: Vec::new(), arguments: Vec::new(), readers: Vec::new() });
        entry.operators.push(op);
    }
    for (index, node) in program.nodes.iter().enumerate() {
        let Node::Affine { terms, .. } = node else { continue };
        for (argument, op) in terms {
            if readers.contains_key(op) {
                let root = find(&mut parent, *op);
                let component = groups.get_mut(&root).expect("every eligible operator has a component");
                if !component.arguments.contains(argument) {
                    component.arguments.push(*argument);
                }
                if !component.readers.contains(&index) {
                    component.readers.push(index);
                }
            }
        }
    }
    groups.into_values().collect()
}

/// `G_y = (1/N) Σ_x J_xᵀ (diag q − q qᵀ) J_x` at every node `y` of `nodes` (width × width), with
/// `q` the program's own distributions and `J_x` the logits' Jacobian in the node's value.
///
/// `diag q − q qᵀ = Σ_c w_c w_cᵀ` with `w_c = √q_c (e_c − q)`, so `G_y = (1/N) Σ_c C_cᵀ C_c` with
/// `C_c` the reverse-mode cotangent at
/// `y` of `w_c`: one reverse pass per class serves every node at once. When the classes outnumber
/// the widest node, the class axis is sketched: `w_k = Σ_c P_kc w_c` with `P` a Rademacher sketch
/// scaled so `E[Pᵀ P] = I`, an unbiased estimate (the fit only proposes; the contract certifies).
fn curvatures(program: &OperatorProgram, context: &SearchContext<'_>, nodes: &[usize]) -> Result<BTreeMap<usize, Array2<f64>>, EngineError> {
    let trace: &Trace = context.trace;
    let logits = &trace.values[program.output];
    let rows = logits.nrows();
    let readouts = context.contract.readouts.max(1);
    let classes = logits.ncols() / readouts;
    let widest = nodes.iter().map(|&n| trace.values[n].ncols()).max().unwrap_or(0);
    let mut out: BTreeMap<usize, Array2<f64>> =
        nodes.iter().map(|&n| (n, Array2::<f64>::zeros((trace.values[n].ncols(), trace.values[n].ncols())))).collect();
    if widest == 0 {
        return Ok(out);
    }
    let mut probabilities = Array2::<f64>::zeros((rows * readouts, classes));
    for row in 0..rows {
        for part in 0..readouts {
            let z = logits.slice(s![row, part * classes..(part + 1) * classes]);
            let m = z.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            let e: Vec<f64> = z.iter().map(|v| (v - m).exp()).collect();
            let sum: f64 = e.iter().sum();
            for (c, v) in e.iter().enumerate() {
                probabilities[[row * readouts + part, c]] = v / sum;
            }
        }
    }
    // Exact when the classes are fewer than the widest reader; otherwise a sketch of at most as
    // many probes as the family has distribution rows (each pass probes every row at once).
    let passes = classes.min(widest).min(rows * readouts);
    let sketch: Option<Array2<f64>> = (passes < classes).then(|| {
        let mut state = 0x9E37_79B9_7F4A_7C15_u64;
        let scale = 1.0 / (passes as f64).sqrt();
        Array2::from_shape_fn((passes, classes), |_| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            if state & 1 == 1 { scale } else { -scale }
        })
    });
    for k in 0..passes {
        let mut cotangent = Array2::<f64>::zeros(logits.dim());
        for row in 0..rows {
            for part in 0..readouts {
                let q = probabilities.row(row * readouts + part);
                // `Σ_c P_kc √q_c (e_c − q)`: the class part, then the shared `−q` part.
                let weights: Vec<f64> = (0..classes)
                    .map(|c| q[c].sqrt() * sketch.as_ref().map_or(if c == k { 1.0 } else { 0.0 }, |p| p[[k, c]]))
                    .collect();
                let total: f64 = weights.iter().sum();
                for c in 0..classes {
                    cotangent[[row, part * classes + c]] = weights[c] - total * q[c];
                }
            }
        }
        let back = vjp(program, &context.contract.family, trace, cotangent)?;
        for (node, g) in out.iter_mut() {
            if let Some(c) = &back[*node] {
                *g += &c.t().dot(c);
            }
        }
    }
    for g in out.values_mut() {
        *g /= rows as f64;
        symmetrize(g);
    }
    Ok(out)
}

fn symmetrize(m: &mut Array2<f64>) {
    let n = m.nrows();
    for i in 0..n {
        for j in (i + 1)..n {
            let v = 0.5 * (m[[i, j]] + m[[j, i]]);
            m[[i, j]] = v;
            m[[j, i]] = v;
        }
    }
}

/// The eigenpairs of a symmetric positive semidefinite matrix above its band: `(values, vectors)`.
fn support(m: &Array2<f64>) -> Result<(Array1<f64>, Array2<f64>), EngineError> {
    let decomposed = eigh(m.view(), SymmetricAssembly::Mirrored, None).map_err(|e| refuse(format!("{e:?}")))?;
    let keep: Vec<usize> = (0..decomposed.values.len()).filter(|&i| decomposed.values[i] > decomposed.band).collect();
    let values = Array1::from_iter(keep.iter().map(|&i| decomposed.values[i]));
    let vectors = decomposed.vectors.select(Axis(1), &keep);
    Ok((values, vectors))
}

/// `V diag(f(λ)) Vᵀ` over a support.
fn spectral(values: &Array1<f64>, vectors: &Array2<f64>, f: impl Fn(f64) -> f64) -> Array2<f64> {
    let mut scaled = vectors.clone();
    for (k, v) in values.iter().enumerate() {
        scaled.column_mut(k).mapv_inplace(|x| x * f(*v));
    }
    scaled.dot(&vectors.t())
}

/// The lattice whose step is nearest `step` in the log, within the reals' range.
fn lattice(step: f64, largest: f64) -> Option<DeclaredPrecision> {
    if !(step > 0.0 && step.is_finite()) {
        return None;
    }
    let bits = (-step.log2()).round().clamp(-1000.0, 1000.0) as i32;
    DeclaredPrecision::new(bits).ok().map(|p| p.within_range(largest))
}

/// Rotate the factor space of `left` (`M × r`, row blocks with their own steps) and `right`
/// (`r × d`, rotated along) to the shortest code of the per-member coefficients `left`
/// (module note, "Gauge").
pub fn fix_gauge(left: &mut Array2<f64>, left_steps: &[f64], right: &mut Array2<f64>) {
    let r = right.nrows();
    if r < 2 {
        return;
    }
    let pair_cost = |l: &Array2<f64>, _: &Array2<f64>, i: usize, j: usize, theta: f64| -> f64 {
        let (c, s) = (theta.cos(), theta.sin());
        let mut total = 0.0;
        for row in 0..l.nrows() {
            let (a, b) = (l[[row, i]], l[[row, j]]);
            let step = left_steps[row];
            total += (1.0 + (c * a + s * b).abs() / step).log2() + (1.0 + (-s * a + c * b).abs() / step).log2();
        }
        total
    };
    const GRID: usize = 64;
    loop {
        let mut saved = 0.0;
        for i in 0..r {
            for j in (i + 1)..r {
                let base = pair_cost(left, right, i, j, 0.0);
                let quarter = std::f64::consts::FRAC_PI_4;
                let mut best = (0.0, base);
                for k in 0..GRID {
                    let theta = -quarter + 2.0 * quarter * (k as f64 + 0.5) / GRID as f64;
                    let cost = pair_cost(left, right, i, j, theta);
                    if cost < best.1 {
                        best = (theta, cost);
                    }
                }
                // Golden-section refinement inside the best grid cell.
                let width = 2.0 * quarter / GRID as f64;
                let (mut lo, mut hi) = (best.0 - width, best.0 + width);
                let golden = 0.5 * (5.0_f64.sqrt() - 1.0);
                for _ in 0..24 {
                    let a = hi - golden * (hi - lo);
                    let b = lo + golden * (hi - lo);
                    if pair_cost(left, right, i, j, a) < pair_cost(left, right, i, j, b) {
                        hi = b;
                    } else {
                        lo = a;
                    }
                }
                let mid = 0.5 * (lo + hi);
                let refined = pair_cost(left, right, i, j, mid);
                if refined < best.1 {
                    best = (mid, refined);
                }
                if best.1 < base {
                    let (c, s) = (best.0.cos(), best.0.sin());
                    for row in 0..left.nrows() {
                        let (a, b) = (left[[row, i]], left[[row, j]]);
                        left[[row, i]] = c * a + s * b;
                        left[[row, j]] = -s * a + c * b;
                    }
                    for col in 0..right.ncols() {
                        let (a, b) = (right[[i, col]], right[[j, col]]);
                        right[[i, col]] = c * a + s * b;
                        right[[j, col]] = -s * a + c * b;
                    }
                    saved += base - best.1;
                }
            }
        }
        if saved < 1.0 {
            break;
        }
    }
}

/// A dense operator on `precision` whose blocks that round to zero are absent.
fn sparse_operator(
    name: String,
    rows: Interface,
    cols: Interface,
    values: Array2<f64>,
    precision: DeclaredPrecision,
    provenance: Provenance,
) -> Result<Operator, EngineError> {
    let rounded = Operator::dense(name.clone(), rows.clone(), cols.clone(), values, precision, provenance.clone())?;
    let matrix = rounded.matrix();
    let mut present = Array2::from_elem((rows.group_count(), cols.group_count()), false);
    for r in 0..rows.group_count() {
        for c in 0..cols.group_count() {
            present[[r, c]] = matrix.slice(s![rows.range(r), cols.range(c)]).iter().any(|v| *v != 0.0);
        }
    }
    Ok(Operator::blocks(name, rows, cols, matrix, present, precision, provenance)?)
}

/// What one component's fit measured, shared by its rank proposals.
struct Fit {
    /// The read nodes' pooled mean and covariance support `(λ, V)`.
    mean: Array1<f64>,
    covariance_trace: f64,
    unwhiten: Array2<f64>,
    /// Per operator: its curvature `G`'s `G^{-1/2}` and `tr G`, and its row range in the stack.
    inverse_roots: Vec<Array2<f64>>,
    curvature_traces: Vec<f64>,
    /// Per operator, the diagonal of its curvature `G`: each coefficient row's own weight.
    curvature_diagonals: Vec<Array1<f64>>,
    /// Per reader node, `tr G` of its own curvature.
    reader_traces: BTreeMap<usize, f64>,
    ranges: Vec<std::ops::Range<usize>>,
    /// The stack's decomposition.
    u: Array2<f64>,
    singular_values: Array1<f64>,
    vt: Array2<f64>,
    band: f64,
    scale: f64,
}

fn measure(
    program: &OperatorProgram,
    context: &SearchContext<'_>,
    component: &Component,
    cache: &std::cell::RefCell<BTreeMap<Vec<usize>, BTreeMap<usize, Array2<f64>>>>,
) -> Result<Option<Fit>, EngineError> {
    let trace = context.trace;
    let d = trace.values[component.arguments[0]].ncols();
    let rows = trace.values[component.arguments[0]].nrows();
    // Pooled mean and covariance of the read nodes.
    let mut mean = Array1::<f64>::zeros(d);
    for &a in &component.arguments {
        mean += &trace.values[a].sum_axis(Axis(0));
    }
    let pooled = (rows * component.arguments.len()) as f64;
    mean /= pooled;
    // The read covariance's support and its roots. With fewer pooled rows than read dimensions the
    // covariance has the rank of the rows: it is taken from the centred data's thin singular value
    // decomposition (`X = P S Qᵀ`, `Σ = Q S² Qᵀ / N`), never as a d × d matrix.
    let (lambda, v) = if component.arguments.len() * rows < d {
        let views: Vec<Array2<f64>> = component.arguments.iter().map(|&a| &trace.values[a] - &mean).collect();
        let stacked = ndarray::concatenate(Axis(0), &views.iter().map(|m| m.view()).collect::<Vec<_>>()).map_err(|e| refuse(e.to_string()))?;
        let decomposed = svd(stacked.view(), false).map_err(|e| refuse(format!("{e:?}")))?;
        let keep: Vec<usize> = (0..decomposed.singular_values.len()).filter(|&i| decomposed.singular_values[i] > decomposed.band).collect();
        let lambda = Array1::from_iter(keep.iter().map(|&i| decomposed.singular_values[i] * decomposed.singular_values[i] / pooled));
        let vectors = decomposed.vt.select(Axis(0), &keep).t().to_owned();
        (lambda, vectors)
    } else {
        let mut covariance = Array2::<f64>::zeros((d, d));
        for &a in &component.arguments {
            let centred = &trace.values[a] - &mean;
            covariance += &centred.t().dot(&centred);
        }
        covariance /= pooled;
        symmetrize(&mut covariance);
        support(&covariance)?
    };
    if lambda.is_empty() {
        return Ok(None);
    }
    let covariance_trace = lambda.sum();
    // Σ^{1/2} restricted to its support (d × d_eff) and its inverse (d_eff × d).
    let mut whiten = v.clone();
    let mut unwhiten = v.t().to_owned();
    for (k, l) in lambda.iter().enumerate() {
        whiten.column_mut(k).mapv_inplace(|x| x * l.sqrt());
        unwhiten.row_mut(k).mapv_inplace(|x| x / l.sqrt());
    }
    // Per reader node, its curvature; per operator, the sum over its uses.
    // The component's identity: its operators' allocations and its readers' positions.
    let key: Vec<usize> = component
        .operators
        .iter()
        .map(|op| std::sync::Arc::as_ptr(&program.operators[*op]) as usize)
        .chain(component.readers.iter().copied())
        .collect();
    let cached = cache.borrow().get(&key).cloned();
    let node_curvature = match cached {
        Some(curvature) => curvature,
        None => {
            let curvature = curvatures(program, context, &component.readers)?;
            cache.borrow_mut().insert(key, curvature.clone());
            curvature
        }
    };
    let scale = (context.contract.observations as f64 * rows as f64).sqrt();
    let mut blocks = Vec::new();
    let (mut inverse_roots, mut curvature_traces, mut ranges) = (Vec::new(), Vec::new(), Vec::new());
    let mut curvature_diagonals = Vec::new();
    let mut offset = 0;
    for &op in &component.operators {
        let m = program.operators[op].rows.width();
        let mut g = Array2::<f64>::zeros((m, m));
        for (index, node) in program.nodes.iter().enumerate() {
            if let Node::Affine { terms, .. } = node {
                let uses = terms.iter().filter(|(_, o)| *o == op).count();
                if uses > 0 {
                    g.scaled_add(uses as f64, &node_curvature[&index]);
                }
            }
        }
        symmetrize(&mut g);
        let (mu, w) = support(&g)?;
        let root = spectral(&mu, &w, f64::sqrt);
        inverse_roots.push(spectral(&mu, &w, |x| 1.0 / x.sqrt()));
        curvature_traces.push(mu.sum());
        curvature_diagonals.push(g.diag().to_owned());
        blocks.push(root.dot(&program.operators[op].matrix()).dot(&whiten) * scale);
        ranges.push(offset..offset + m);
        offset += m;
    }
    let views: Vec<_> = blocks.iter().map(|b| b.view()).collect();
    let stacked = ndarray::concatenate(Axis(0), &views).map_err(|e| refuse(e.to_string()))?;
    let decomposed = svd(stacked.view(), false).map_err(|e| refuse(format!("{e:?}")))?;
    Ok(Some(Fit {
        mean,
        covariance_trace,
        unwhiten,
        inverse_roots,
        curvature_traces,
        curvature_diagonals,
        reader_traces: node_curvature.iter().map(|(node, g)| (*node, g.diag().sum())).collect(),
        ranges,
        u: decomposed.u,
        singular_values: decomposed.singular_values,
        vt: decomposed.vt,
        band: decomposed.band,
        scale,
    }))
}

/// The program with `component` factored at rank `r` (module note).
fn factor(program: &OperatorProgram, component: &Component, fit: &Fit, r: usize, sparse: bool) -> Result<Option<OperatorProgram>, EngineError> {
    let observations_rows = fit.scale * fit.scale;
    // The basis is orthonormal in the whitened read space, where reads that use different
    // directions are orthogonal, so a rotation of the factor space can separate them; each
    // coefficient carries its direction's singular value. Mapped back to the program's coordinates.
    let mut right = fit.vt.slice(s![..r, ..]).dot(&fit.unwhiten);
    let total_rows = fit.ranges.last().map_or(0, |range| range.end);
    let mut left = Array2::<f64>::zeros((total_rows, r));
    for (s_index, range) in fit.ranges.iter().enumerate() {
        let mut block = fit.u.slice(s![range.clone(), ..r]).to_owned();
        for (k, sk) in fit.singular_values.iter().take(r).enumerate() {
            block.column_mut(k).mapv_inplace(|v| v * sk);
        }
        let mapped = fit.inverse_roots[s_index].dot(&block) / fit.scale;
        left.slice_mut(s![range.clone(), ..]).assign(&mapped);
    }
    // Steps (module note, "Precision"): tr(R Σ Rᵀ) = r for the orthonormal basis, and
    // Σ_s tr(L_sᵀ G_s L_s) = Σ σ_i² / (n N) over the kept directions.
    let energy: f64 = fit.singular_values.iter().take(r).map(|s| s * s).sum();
    if !(energy > 0.0) {
        return Ok(None);
    }
    let left_steps: Vec<f64> = fit
        .ranges
        .iter()
        .enumerate()
        .map(|(s_index, range)| {
            let m = (range.len() * r) as f64;
            (12.0 * m / (observations_rows * fit.curvature_traces[s_index] * r as f64)).sqrt()
        })
        .collect();
    let right_step = (12.0 * (right.len() as f64) / (fit.covariance_trace * energy)).sqrt();
    let row_steps: Vec<f64> = fit.ranges.iter().enumerate().flat_map(|(s_index, range)| std::iter::repeat_n(left_steps[s_index], range.len())).collect();
    fix_gauge(&mut left, &row_steps, &mut right);
    if sparse {
        // A coefficient is kept when dropping it costs more data bits, `u²/(2 ln 2)` at its
        // normalised size `u` (the factor coordinate has unit variance, and `G`'s diagonal weighs
        // the row), than coding it costs, `log₂(1 + |u|/√12) + 1` on its lattice.
        let weights: Vec<f64> = fit.curvature_diagonals.iter().flat_map(|d| d.iter().copied()).collect();
        for (row, weight) in weights.iter().enumerate() {
            let scale = (observations_rows * weight.max(0.0)).sqrt();
            for value in left.row_mut(row).iter_mut() {
                let u = *value * scale;
                if u * u / (2.0 * std::f64::consts::LN_2) <= (1.0 + u.abs() / 12.0_f64.sqrt()).log2() + 1.0 {
                    *value = 0.0;
                }
            }
        }
    }
    let factor_interface = Interface::uniform(r, 1, LabelKind::Factor, 0)?;
    let argument_interface = program.interfaces()?[component.arguments[0]].clone();
    let names: Vec<&str> = component.operators.iter().map(|op| program.operators[*op].name.as_str()).collect();
    let parts: Vec<&Provenance> = component.operators.iter().map(|op| &program.operators[*op].provenance).collect();
    let right_largest = right.iter().fold(0.0_f64, |m, v| m.max(v.abs()));
    let Some(right_precision) = lattice(right_step, right_largest) else { return Ok(None) };
    let mut candidate = program.clone();
    candidate.operators.push(Arc::new(sparse_operator(
        format!("basis of {}", names.join("+")),
        factor_interface.clone(),
        argument_interface,
        right.clone(),
        right_precision,
        Provenance::derived(&parts, "curvature-whitened shared basis".to_string()),
    )?));
    let basis = candidate.operators.len() - 1;
    let basis_matrix = candidate.operators[basis].matrix();
    let mut coefficient_of: BTreeMap<usize, usize> = BTreeMap::new();
    let mut residual_of: BTreeMap<usize, Array1<f64>> = BTreeMap::new();
    for (s_index, &op) in component.operators.iter().enumerate() {
        let old = &program.operators[op];
        let values = left.slice(s![fit.ranges[s_index].clone(), ..]).to_owned();
        let largest = values.iter().fold(0.0_f64, |m, v| m.max(v.abs()));
        let Some(precision) = lattice(left_steps[s_index], largest) else { return Ok(None) };
        let coefficients = sparse_operator(
            format!("{}·factors", old.name),
            old.rows.clone(),
            factor_interface.clone(),
            values,
            precision,
            Provenance::derived(&[&old.provenance], "coefficients on the shared basis".to_string()),
        )?;
        // The mean the factored operator misses, folded into its readers' biases.
        let missed = old.matrix().dot(&fit.mean) - coefficients.matrix().dot(&basis_matrix.dot(&fit.mean));
        candidate.operators.push(Arc::new(coefficients));
        coefficient_of.insert(op, candidate.operators.len() - 1);
        residual_of.insert(op, missed);
    }
    // Per reader, the mean correction its terms need.
    let mut correction: BTreeMap<usize, Array1<f64>> = BTreeMap::new();
    for &reader in &component.readers {
        let Node::Affine { terms, .. } = &program.nodes[reader] else { continue };
        for (_, op) in terms {
            if let Some(missed) = residual_of.get(op) {
                let entry = correction.entry(reader).or_insert_with(|| Array1::zeros(missed.len()));
                *entry += missed;
            }
        }
    }
    // New bias operators: one per (old bias, correction) class of readers.
    let mut bias_users: BTreeMap<usize, usize> = BTreeMap::new();
    for node in &program.nodes {
        if let Node::Affine { bias: Some(b), .. } = node {
            *bias_users.entry(*b).or_default() += 1;
        }
    }
    let interfaces = program.interfaces()?;
    let mut new_bias: BTreeMap<usize, usize> = BTreeMap::new();
    let mut made: Vec<(Option<usize>, Vec<u64>, usize)> = Vec::new();
    for (&reader, fix) in &correction {
        let Node::Affine { bias, .. } = &program.nodes[reader] else { continue };
        let key: Vec<u64> = fix.iter().map(|v| v.to_bits()).collect();
        if let Some((_, _, op)) = made.iter().find(|(b, k, _)| b == bias && *k == key) {
            new_bias.insert(reader, *op);
            continue;
        }
        let rows = interfaces[reader].clone();
        let (values, precision, provenance, name) = match bias {
            Some(b) => {
                let old = &program.operators[*b];
                let OperatorBody::Dense { precision, .. } = &old.body else { return Ok(None) };
                let mut values = old.matrix();
                values.column_mut(0).scaled_add(1.0, fix);
                (values, *precision, Provenance::derived(&[&old.provenance], "mean of the factored reads folded in".to_string()), old.name.clone())
            }
            None => {
                // The bias's own optimum: `Δ² = 12 m / (n N tr G)` at its reader.
                let largest = fix.iter().fold(0.0_f64, |m, v| m.max(v.abs()));
                let step = (12.0 * fix.len() as f64 / (observations_rows * fit.reader_traces[&reader])).sqrt();
                let Some(precision) = lattice(step, largest) else { return Ok(None) };
                (fix.clone().insert_axis(Axis(1)), precision, Provenance::derived(&parts, "mean of the factored reads".to_string()), format!("bias of node {reader}"))
            }
        };
        let largest = values.iter().fold(0.0_f64, |m, v| m.max(v.abs()));
        let precision = precision.within_range(largest);
        candidate.operators.push(Arc::new(Operator::dense(name, rows, Interface::constant(), values, precision, provenance)?));
        let op = candidate.operators.len() - 1;
        new_bias.insert(reader, op);
        made.push((*bias, key, op));
    }
    // Rebuild the nodes: each read node is followed by its factor coordinates.
    let operators_map: Vec<usize> = (0..candidate.operators.len()).collect();
    let bases_map: Vec<usize> = (0..candidate.bases.len()).collect();
    let rules_map: Vec<usize> = (0..candidate.rules.len()).collect();
    let mut map = vec![0usize; program.nodes.len()];
    let mut coordinates: BTreeMap<usize, usize> = BTreeMap::new();
    let mut nodes = Vec::with_capacity(program.nodes.len() + component.arguments.len());
    for (index, node) in program.nodes.iter().enumerate() {
        let mut rebuilt = node.clone();
        remap_node(&mut rebuilt, &map, &operators_map, &bases_map, &rules_map);
        if let (Node::Affine { terms: old_terms, .. }, Node::Affine { terms, bias }) = (node, &mut rebuilt) {
            for ((argument, op), term) in old_terms.iter().zip(terms.iter_mut()) {
                if let Some(&coefficients) = coefficient_of.get(op) {
                    *term = (coordinates[argument], coefficients);
                }
            }
            if let Some(&b) = new_bias.get(&index) {
                *bias = Some(b);
            }
        }
        nodes.push(rebuilt);
        map[index] = nodes.len() - 1;
        if component.arguments.contains(&index) {
            nodes.push(Node::Affine { terms: vec![(map[index], basis)], bias: None });
            coordinates.insert(index, nodes.len() - 1);
        }
    }
    candidate.nodes = nodes;
    candidate.output = map[program.output];
    candidate.prune();
    candidate.interfaces()?;
    Ok(Some(candidate))
}

impl Primitive for Factors {
    fn name(&self) -> &'static str {
        "factors"
    }

    fn propose(&self, context: &SearchContext<'_>) -> Result<Vec<Proposal>, EngineError> {
        let program = context.program;
        let mut out = Vec::new();
        for component in components(program) {
            // A reader at least as wide as the output's classes and wider than the family has a
            // curvature of more entries than the family can resolve, at one reverse pass per class:
            // such an operator (an unembedding) is left to the precision and restriction moves.
            let classes = context.trace.values[program.output].ncols() / context.contract.readouts.max(1);
            if component.readers.iter().any(|r| {
                let width = context.trace.values[*r].ncols();
                width >= classes && width > context.trace.values[*r].nrows()
            }) {
                continue;
            }
            let Some(fit) = measure(program, context, &component, &self.curvature)? else { continue };
            let stacked_rows = fit.ranges.last().map_or(0, |range| range.end);
            let d = fit.vt.ncols();
            let full = fit.singular_values.len();
            let reals: usize = component.operators.iter().map(|op| program.operators[*op].real_count()).sum();
            let threshold = (stacked_rows + d) as f64;
            let rd = fit.singular_values.iter().filter(|s| *s * *s > threshold).count().max(1);
            let exact = fit.singular_values.iter().filter(|s| **s > fit.band).count().max(1);
            let mut ranks: BTreeSet<usize> = [rd / 2, rd, 2 * rd, exact].into_iter().filter(|r| *r >= 1 && *r <= full).collect();
            ranks.retain(|r| r * (stacked_rows + program.operators[component.operators[0]].cols.width()) < reals);
            let names: Vec<&str> = component.operators.iter().map(|op| program.operators[*op].name.as_str()).collect();
            for r in ranks {
                for sparse in [false, true] {
                    let Some(candidate) = factor(program, &component, &fit, r, sparse)? else { continue };
                    // Every factor is rounded to its rate-distortion lattice: never exact.
                    out.push(Proposal {
                        primitive: "factors",
                        kind: ProposalKind::Share,
                        exactness: Exactness::Approximate,
                        description: format!(
                            "{}rank {r} factors of {names:?} (rate-distortion rank {rd})",
                            if sparse { "sparse " } else { "" }
                        ),
                        edit: Edit::Program(Box::new(candidate)),
                    });
                }
            }
        }
        Ok(out)
    }
}
