//! Coupled controls of a factorization `γ̂ = A φ(B h)`.
//!
//! The explanation's controls are the `k` coordinates of `φ(B h)`; `A` (`m × k`) writes them
//! into `m` native units (a SwiGLU block's hidden units, whose down-projection columns or
//! up-projection rows carry native per-unit gains). A per-control setting `α` asks for
//! `A diag(α) φ(B h)` at every input. It is set-type: `α_c` is control `c`'s gain relative to
//! the native control (`α = 1` is `θ`), never a factor on an earlier edit, so setting `α` and
//! then `β` is setting `β`.
//!
//! # Writer gains
//!
//! Native unit gains `s` give `diag(s) A φ(B h)`, so they realize `α` at every input iff
//!
//! ```text
//! diag(s) A = A diag(α)   ⇔   s_i = α_c for every stored A_ic ≠ 0.
//! ```
//!
//! On the bipartite support graph (units and controls, an edge per nonzero stored entry)
//! the equations force `α` constant along every path, so the **coupling classes** are the
//! graph's connected components: controls in one class move together or not at all. A
//! request that differs inside a class is witnessed by the shortest path joining two
//! controls of different requested gains; each edge on it is a stored nonzero entry. A
//! control with an all-zero column writes nothing and is its own class. The support is
//! the stored one, never thresholded: a dense fitted `A` has one class, and that is the
//! answer. The best writer-only gains in the least-squares sense,
//! `s_i = Σ_c A_ic² α_c / Σ_c A_ic²`, and the residual `‖diag(s) A − A diag(α)‖_F` are
//! reported beside the verdict; the residual operator is what the response then errs by,
//! `(diag(s) A − A diag(α)) φ(B h)`.
//!
//! # Reader rows and coordinated edits
//!
//! When `B`'s rows are native reads, row gains `t` give `A φ(diag(t) B h)`. This equals
//! `A diag(t) φ(B h)` for a linear branch (SwiGLU's up projection, bilinear in its read)
//! at every `t`, and for a positively homogeneous law (ReLU) at `t ≥ 0`; for the gate's
//! SiLU or GELU no `t ≠ 1` qualifies, so it is not a permitted reader. With both sites the
//! equations are `s_i t_c = α_c` on the support:
//!
//! * a linear reader realizes every `α` alone (`t = α`, `s = 1`);
//! * a positively homogeneous reader realizes `α ≥ 0` alone, and with writer gains every
//!   `α` whose signs are constant on the classes of the support restricted to controls
//!   with `α_c ≠ 0`: `t_c = |α_c|`, `s_i = sign α_c`. A sign change inside such a class is
//!   witnessed by a path, as above.
//!
//! The compiler takes the first feasible of writer alone, reader alone, and both, so a
//! realized control edits as few tensors as it can. Every realization is an algebraic
//! identity over all inputs: [`ControlRealization::ExactlyRealized`] with the computed
//! defect `‖diag(s) A diag(t) − A diag(α)‖_F` and its band.
//!
//! # Invariances
//!
//! Renaming controls permutes the classes; rescaling a control's column by a nonzero factor
//! (with `φ` compensating) keeps its support and so its class; a duplicated control shares
//! its original's support and joins its class.

use std::collections::VecDeque;

use gam_linalg::roundoff::accumulation_growth;
use gam_math::roundoff::inflated;
use ndarray::{Array2, ArrayView2};

use super::super::apply::FactoredEdit;
use super::super::lift::{TensorId, TensorRegistry};
use super::super::supports::{EvidenceStatus, ExactBasis};
use super::{
    CompileError, CompiledControl, CompiledParameterEdit, ControlRealization, DescriptiveReason, NativeEditPlan,
    require_finite, require_shape,
};

/// Which axis of a stored matrix carries the per-unit gain.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GainAxis {
    /// `ΔW = diag(g − 1) W`: the units are the stored rows (an up projection's rows).
    Rows,
    /// `ΔW = W diag(g − 1)`: the units are the stored columns (a down projection's columns).
    Columns,
}

/// A stored matrix whose rows or columns carry native gains.
#[derive(Clone, Debug)]
pub struct GainSite<'a> {
    pub storage: TensorId,
    pub weight: ArrayView2<'a, f64>,
    pub axis: GainAxis,
}

/// How the law between a native reader and the controls commutes with a row gain.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ReaderHomogeneity {
    /// `φ(t z) = t φ(z)` for every real `t` (a linear or bilinear branch).
    Linear,
    /// `φ(t z) = t φ(z)` for `t ≥ 0` (ReLU).
    PositivelyHomogeneous,
}

/// A native reader whose rows are the controls' reads.
#[derive(Clone, Debug)]
pub struct ReaderSite<'a> {
    pub site: GainSite<'a>,
    pub homogeneity: ReaderHomogeneity,
}

/// A coupled-controls problem.
#[derive(Clone, Debug)]
pub struct CoupledControlProblem<'a> {
    pub registry: &'a TensorRegistry,
    /// `A`, `m` native units × `k` controls, as stored (its exact support is read).
    pub writer: ArrayView2<'a, f64>,
    /// The requested per-control rescaling `α` (`k`).
    pub alpha: &'a [f64],
    /// Where unit gains `s` act, when permitted.
    pub writer_site: Option<GainSite<'a>>,
    /// Where control gains `t` act, when permitted.
    pub reader_site: Option<ReaderSite<'a>>,
}

/// One node of a support path.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SupportNode {
    Control(usize),
    Unit(usize),
}

/// A connected component of the support graph.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CouplingClass {
    pub controls: Vec<usize>,
    pub units: Vec<usize>,
}

/// Two controls joined through stored nonzero entries whose requested gains differ (in
/// value for the writer, in sign for a positively homogeneous reader).
#[derive(Clone, Debug, PartialEq)]
pub struct CouplingWitness {
    pub path: Vec<SupportNode>,
}

/// The family a coupled-controls realization is stated over: every input `h`, for the `k`
/// controls.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ControlFamily {
    pub controls: usize,
}

/// Which sites a realization uses.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SiteChoice {
    Writer,
    Reader,
    WriterAndReader,
}

/// What [`compile_coupled_controls`] found.
#[derive(Clone, Debug)]
pub struct CoupledControlReport {
    /// Connected components of the whole support.
    pub writer_classes: Vec<CouplingClass>,
    /// Components of the support restricted to controls with `α_c ≠ 0` (the sign classes a
    /// positively homogeneous reader leaves coupled).
    pub sign_classes: Vec<CouplingClass>,
    /// Least-squares writer-only gains and `‖diag(s) A − A diag(α)‖_F` with its band.
    pub best_writer_gains: Vec<f64>,
    pub writer_residual: (f64, f64),
    /// The sites the realization uses and its gains (`s` over units, `t` over controls).
    pub choice: Option<SiteChoice>,
    pub unit_gains: Option<Vec<f64>>,
    pub control_gains: Option<Vec<f64>>,
    pub compiled: CompiledControl<CouplingWitness, ControlFamily>,
}

/// Components of the bipartite support graph over the controls in `active` (all units).
fn components(writer: ArrayView2<'_, f64>, active: &[bool]) -> Vec<CouplingClass> {
    let (units, controls) = writer.dim();
    let mut seen_control = vec![false; controls];
    let mut seen_unit = vec![false; units];
    let mut classes = Vec::new();
    for start in 0..controls {
        if seen_control[start] || !active[start] {
            continue;
        }
        let mut class = CouplingClass {
            controls: Vec::new(),
            units: Vec::new(),
        };
        let mut queue = VecDeque::from([SupportNode::Control(start)]);
        seen_control[start] = true;
        while let Some(node) = queue.pop_front() {
            match node {
                SupportNode::Control(c) => {
                    class.controls.push(c);
                    for i in 0..units {
                        if writer[[i, c]] != 0.0 && !seen_unit[i] {
                            seen_unit[i] = true;
                            queue.push_back(SupportNode::Unit(i));
                        }
                    }
                }
                SupportNode::Unit(i) => {
                    class.units.push(i);
                    for c in 0..controls {
                        if writer[[i, c]] != 0.0 && active[c] && !seen_control[c] {
                            seen_control[c] = true;
                            queue.push_back(SupportNode::Control(c));
                        }
                    }
                }
            }
        }
        class.controls.sort_unstable();
        class.units.sort_unstable();
        classes.push(class);
    }
    classes
}

/// The shortest support path from control `from` to control `to` over `active` controls.
fn support_path(writer: ArrayView2<'_, f64>, active: &[bool], from: usize, to: usize) -> Vec<SupportNode> {
    let (units, controls) = writer.dim();
    let mut parent_control: Vec<Option<usize>> = vec![None; controls];
    let mut parent_unit: Vec<Option<usize>> = vec![None; units];
    let mut seen_control = vec![false; controls];
    let mut seen_unit = vec![false; units];
    seen_control[from] = true;
    let mut queue = VecDeque::from([SupportNode::Control(from)]);
    while let Some(node) = queue.pop_front() {
        match node {
            SupportNode::Control(c) => {
                if c == to {
                    break;
                }
                for i in 0..units {
                    if writer[[i, c]] != 0.0 && !seen_unit[i] {
                        seen_unit[i] = true;
                        parent_unit[i] = Some(c);
                        queue.push_back(SupportNode::Unit(i));
                    }
                }
            }
            SupportNode::Unit(i) => {
                for c in 0..controls {
                    if writer[[i, c]] != 0.0 && active[c] && !seen_control[c] {
                        seen_control[c] = true;
                        parent_control[c] = Some(i);
                        queue.push_back(SupportNode::Control(c));
                    }
                }
            }
        }
    }
    let mut path = vec![SupportNode::Control(to)];
    let mut control = to;
    while control != from {
        let Some(unit) = parent_control[control] else {
            break;
        };
        path.push(SupportNode::Unit(unit));
        let Some(previous) = parent_unit[unit] else {
            break;
        };
        path.push(SupportNode::Control(previous));
        control = previous;
    }
    path.reverse();
    path
}

/// `‖diag(s) A diag(t) − A diag(α)‖_F` and its band: each entry is two products per side
/// and a subtraction, so it errs by `γ_3` of its operands' magnitudes; the sum of squares
/// adds `γ_{mk+1}`.
fn defect(writer: ArrayView2<'_, f64>, unit_gains: &[f64], control_gains: &[f64], alpha: &[f64]) -> (f64, f64) {
    let (units, controls) = writer.dim();
    let mut squares = 0.0;
    let mut bound = 0.0;
    for i in 0..units {
        for c in 0..controls {
            let a = writer[[i, c]];
            let edited = unit_gains[i] * a * control_gains[c];
            let requested = a * alpha[c];
            let difference = edited - requested;
            squares += difference * difference;
            let entry_band = accumulation_growth(3) * (edited.abs() + requested.abs());
            let upper = difference.abs() + entry_band;
            bound += upper * upper - difference * difference;
        }
    }
    let value = squares.sqrt();
    let band = inflated((squares + bound).sqrt() - value + accumulation_growth(units * controls + 1) * value, 2);
    (value, band.max(0.0))
}

/// The native edit of gains `g` on `site` (`None` when every gain is exactly 1).
pub fn gain_edit(site: &GainSite<'_>, gains: &[f64]) -> Result<Option<CompiledParameterEdit>, CompileError> {
    require_finite("gains", gains.iter().copied())?;
    let (rows, cols) = site.weight.dim();
    let units = match site.axis {
        GainAxis::Rows => rows,
        GainAxis::Columns => cols,
    };
    require_shape("gains", (units, 1), (gains.len(), 1))?;
    let changed: Vec<usize> = (0..units).filter(|&unit| gains[unit] != 1.0).collect();
    if changed.is_empty() {
        return Ok(None);
    }
    let rank = changed.len();
    let (left, right) = match site.axis {
        GainAxis::Rows => {
            let mut left = Array2::<f64>::zeros((rows, rank));
            let mut right = Array2::<f64>::zeros((cols, rank));
            for (term, &unit) in changed.iter().enumerate() {
                left[[unit, term]] = gains[unit] - 1.0;
                right.column_mut(term).assign(&site.weight.row(unit));
            }
            (left, right)
        }
        GainAxis::Columns => {
            let mut left = Array2::<f64>::zeros((rows, rank));
            let mut right = Array2::<f64>::zeros((cols, rank));
            for (term, &unit) in changed.iter().enumerate() {
                let step = gains[unit] - 1.0;
                left.column_mut(term).assign(&site.weight.column(unit).mapv(|value| value * step));
                right[[unit, term]] = 1.0;
            }
            (left, right)
        }
    };
    Ok(Some(CompiledParameterEdit {
        storage: site.storage.clone(),
        delta: FactoredEdit::new(left, right)?,
    }))
}

/// Compiles the per-control rescaling `α` into native gains, or reports the coupling.
pub fn compile_coupled_controls(
    problem: &CoupledControlProblem<'_>,
    control: &str,
) -> Result<CoupledControlReport, CompileError> {
    let writer = problem.writer;
    let (units, controls) = writer.dim();
    require_finite("writer factor", writer.iter().copied())?;
    require_finite("requested rescaling", problem.alpha.iter().copied())?;
    require_shape("requested rescaling", (controls, 1), (problem.alpha.len(), 1))?;
    let alpha = problem.alpha;
    if let Some(site) = &problem.writer_site {
        let length = match site.axis {
            GainAxis::Rows => site.weight.nrows(),
            GainAxis::Columns => site.weight.ncols(),
        };
        require_shape("writer site units", (units, 1), (length, 1))?;
    }
    if let Some(reader) = &problem.reader_site {
        let length = match reader.site.axis {
            GainAxis::Rows => reader.site.weight.nrows(),
            GainAxis::Columns => reader.site.weight.ncols(),
        };
        require_shape("reader site controls", (controls, 1), (length, 1))?;
    }
    let all = vec![true; controls];
    let nonzero: Vec<bool> = alpha.iter().map(|value| *value != 0.0).collect();
    let writer_classes = components(writer, &all);
    let sign_classes = components(writer, &nonzero);

    let mut best_writer_gains = vec![1.0; units];
    for (i, gain) in best_writer_gains.iter_mut().enumerate() {
        let (weighted, mass) = (0..controls).fold((0.0, 0.0), |(w, m), c| {
            let square = writer[[i, c]] * writer[[i, c]];
            (w + square * alpha[c], m + square)
        });
        if mass > 0.0 {
            *gain = weighted / mass;
        }
    }
    let ones_controls = vec![1.0; controls];
    let writer_residual = defect(writer, &best_writer_gains, &ones_controls, alpha);

    // Writer alone: α constant on each class.
    let writer_conflict = writer_classes.iter().find_map(|class| {
        let first = class.controls[0];
        class
            .controls
            .iter()
            .find(|&&c| alpha[c] != alpha[first])
            .map(|&c| support_path(writer, &all, first, c))
    });
    let writer_gains = || {
        let mut gains = vec![1.0; units];
        for class in &writer_classes {
            for &unit in &class.units {
                gains[unit] = alpha[class.controls[0]];
            }
        }
        gains
    };
    // Positively homogeneous reader with writer signs: signs constant on sign classes.
    let sign_conflict = sign_classes.iter().find_map(|class| {
        let first = class.controls[0];
        class
            .controls
            .iter()
            .find(|&&c| (alpha[c] > 0.0) != (alpha[first] > 0.0))
            .map(|&c| support_path(writer, &nonzero, first, c))
    });

    let mut choice = None;
    let mut unit_gains = None;
    let mut control_gains = None;
    if problem.writer_site.is_some() && writer_conflict.is_none() {
        choice = Some(SiteChoice::Writer);
        unit_gains = Some(writer_gains());
        control_gains = Some(ones_controls.clone());
    } else if let Some(reader) = &problem.reader_site {
        let reader_alone = match reader.homogeneity {
            ReaderHomogeneity::Linear => true,
            ReaderHomogeneity::PositivelyHomogeneous => alpha.iter().all(|value| *value >= 0.0),
        };
        if reader_alone {
            choice = Some(SiteChoice::Reader);
            unit_gains = Some(vec![1.0; units]);
            control_gains = Some(alpha.to_vec());
        } else if problem.writer_site.is_some() && sign_conflict.is_none() {
            choice = Some(SiteChoice::WriterAndReader);
            let mut signs = vec![1.0; units];
            for class in &sign_classes {
                let sign = if alpha[class.controls[0]] < 0.0 { -1.0 } else { 1.0 };
                for &unit in &class.units {
                    signs[unit] = sign;
                }
            }
            unit_gains = Some(signs);
            control_gains = Some(alpha.iter().map(|value| value.abs()).collect());
        }
    }

    let family = ControlFamily { controls };
    let compiled = match (choice, &unit_gains, &control_gains) {
        (Some(chosen), Some(s), Some(t)) => {
            let (value, band) = defect(writer, s, t, alpha);
            let mut edits = Vec::new();
            if matches!(chosen, SiteChoice::Writer | SiteChoice::WriterAndReader)
                && let Some(site) = &problem.writer_site
            {
                edits.extend(gain_edit(site, s)?);
            }
            if matches!(chosen, SiteChoice::Reader | SiteChoice::WriterAndReader)
                && let Some(reader) = &problem.reader_site
            {
                edits.extend(gain_edit(&reader.site, t)?);
            }
            let plan = NativeEditPlan::new(problem.registry, edits)?;
            let residual = EvidenceStatus::exact(value, band, ExactBasis::Algebraic, None, family)?;
            CompiledControl::new(
                control.to_string(),
                Some(plan),
                ControlRealization::exactly_realized(residual)?,
            )?
        }
        _ => {
            let path = if problem.reader_site.is_some() && problem.writer_site.is_some() {
                sign_conflict.or(writer_conflict)
            } else {
                writer_conflict.or(sign_conflict)
            };
            let witness = match path {
                Some(path) if writer_residual.0 - writer_residual.1 > 0.0 => Some(EvidenceStatus::counterexample(
                    writer_residual.0,
                    writer_residual.1,
                    0.0,
                    CouplingWitness { path },
                )?),
                Some(_) | None => None,
            };
            CompiledControl::new(
                control.to_string(),
                None,
                ControlRealization::descriptive(DescriptiveReason::CoupledControls, witness)?,
            )?
        }
    };
    Ok(CoupledControlReport {
        writer_classes,
        sign_classes,
        best_writer_gains,
        writer_residual,
        choice,
        unit_gains,
        control_gains,
        compiled,
    })
}

