//! Direct support-sparse cold starts for overcomplete hard-TopK SAE fits.
//!
//! The dense seed's `K×N×D` PCA tensor and `N×K` routing matrix are not valid
//! representations when only `s = top_k` atoms exist in each row's local
//! problem. This module consumes the typed front-door admission and constructs
//! [`SaeAssignmentState`] directly with an `O(s)` row-local score window and
//! heterogeneous, unpadded coordinate blocks.

use crate::assignment_state::{SaeAssignmentAtomSpec, SaeAssignmentState};
use crate::front_door::{SaeFitAdmission, SaeFitLane};
use ndarray::{Array2, Array3, ArrayView2, s};
use rayon::prelude::*;

use super::{
    SaeAtomBasisKind, SaeManifoldAtom, SaeReferenceRoughness, sae_atom_basis_kind_from_str,
    sae_build_atom_plans, sae_build_padded_basis_stacks, sae_pick_duchon_center_indices,
};

pub struct SaeSupportSeedRequest<'a> {
    pub target: ArrayView2<'a, f64>,
    pub atom_basis: &'a [String],
    pub atom_dim: &'a [usize],
    pub support_k: usize,
    pub random_state: u64,
    /// The exact public-front-door decision. Consuming it here prevents a seed
    /// builder from silently re-deciding or discarding the admitted lane.
    pub admission: SaeFitAdmission,
}

pub struct SaeSupportSeedReport {
    pub assignment: SaeAssignmentState,
    pub atom_kinds: Vec<SaeAtomBasisKind>,
    pub effective_atom_dim: Vec<usize>,
    /// Original requested atom index for each retained, occupied atom. Atoms
    /// with zero support mass are structurally dead and never enter the fit.
    pub retained_atom_indices: Vec<usize>,
    /// Maximum atom-score cells retained simultaneously, independent of `K`.
    pub peak_score_cells: usize,
}

pub struct SaeSupportTermSeedRequest {
    pub assignment: SaeAssignmentState,
    pub atom_basis: Vec<String>,
    /// Public dimensions (periodic entries are harmonic resolution, matching
    /// the dense planner); the assignment carries effective chart dimensions.
    pub atom_dim: Vec<usize>,
    pub output_dim: usize,
    pub random_state: u64,
}

pub struct SaeSupportTermSeedReport {
    pub term: super::SaeSupportSparseTerm,
    pub atom_plans: Vec<super::SaeAtomBuildPlan>,
}

#[derive(Clone, Copy)]
struct RankedAtom {
    atom: usize,
    score: f64,
}

pub(super) fn splitmix64(value: u64) -> u64 {
    gam_linalg::utils::splitmix64_hash(value)
}

/// Bounded-work CountSketch projection. At small P each coordinate appears in
/// the cyclic hash permutation; at large P eight deterministic samples keep
/// routing cost independent of the ambient output width.
pub(super) fn projection(row: &[f64], atom: usize, axis: usize, random_state: u64) -> f64 {
    let width = row.len();
    let samples = width.min(8).max(1);
    let mut total = 0.0;
    for sample in 0..samples {
        let key = random_state
            ^ (atom as u64).wrapping_mul(0xd6e8_feb8_6659_fd93)
            ^ (axis as u64).wrapping_mul(0xa5a3_564e_27f8_864d)
            ^ (sample as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15);
        let hash = splitmix64(key);
        let column = (sample + hash as usize % width) % width;
        total += if hash >> 63 == 0 {
            -row[column]
        } else {
            row[column]
        };
    }
    total / (samples as f64).sqrt()
}

fn score(row: &[f64], atom: usize, random_state: u64) -> f64 {
    projection(row, atom, 0, random_state).hypot(projection(
        row,
        atom,
        usize::MAX / 2,
        random_state,
    ))
}

fn better(lhs: RankedAtom, rhs: RankedAtom) -> bool {
    lhs.score > rhs.score || (lhs.score == rhs.score && lhs.atom < rhs.atom)
}

/// Chart-sample count the atom planner is seeded with: the flat-basis kinds
/// size their basis from the seed's span so they need a spread of samples;
/// analytic kinds derive their basis from the geometry alone. ONE owner for
/// the convention -- the topology conversion reuses it.
pub(crate) fn planner_design_rows(kind: &SaeAtomBasisKind) -> usize {
    if matches!(
        kind,
        SaeAtomBasisKind::Duchon
            | SaeAtomBasisKind::Linear
            | SaeAtomBasisKind::EuclideanPatch
            | SaeAtomBasisKind::Poincare
    ) {
        32
    } else {
        1
    }
}

fn effective_atom(
    public_dim: usize,
    kind: &SaeAtomBasisKind,
    atom: usize,
) -> Result<(usize, SaeAssignmentAtomSpec), String> {
    if public_dim == 0 {
        return Err(format!(
            "build_sae_support_seed: atom_dim[{atom}] must be positive"
        ));
    }
    let latent_dim = match kind {
        // Public periodic dimension is harmonic resolution; its chart is 1-D.
        SaeAtomBasisKind::Periodic => 1,
        // `S²` and its antipodal quotient are 2-dimensional but carry THREE
        // coordinates: they admit no global 2-D chart, so the ambient unit
        // vector is the only pole-free parameterisation. The public dimension
        // stays 2 (it names the manifold's intrinsic dimension, which is what a
        // caller means by `atom_dim`) while the live chart is 3-wide -- the same
        // public-vs-effective split `Periodic` already uses in the arm above.
        SaeAtomBasisKind::Sphere | SaeAtomBasisKind::ProjectivePlane => {
            if public_dim != 2 {
                return Err(format!(
                    "build_sae_support_seed: atom {atom} basis requires atom_dim == 2; got {public_dim}"
                ));
            }
            3
        }
        SaeAtomBasisKind::Torus
        | SaeAtomBasisKind::KleinBottle
        | SaeAtomBasisKind::Mobius => {
            if public_dim != 2 {
                return Err(format!(
                    "build_sae_support_seed: atom {atom} basis requires atom_dim == 2; got {public_dim}"
                ));
            }
            2
        }
        SaeAtomBasisKind::Cylinder | SaeAtomBasisKind::FiniteSet => {
            return Err(format!(
                "build_sae_support_seed: atom {atom} uses a discovery-only basis that cannot seed a continuous TopK chart"
            ));
        }
        SaeAtomBasisKind::Precomputed(label) => {
            return Err(format!(
                "build_sae_support_seed: atom {atom} basis {label:?} has no analytic sparse-seed chart"
            ));
        }
        _ => public_dim,
    };
    Ok((
        latent_dim,
        SaeAssignmentAtomSpec {
            latent_dim,
            id_mode: gam_terms::latent::LatentIdMode::None,
            manifold: kind.latent_manifold(latent_dim),
            retraction: gam_problem::LatentRetractionRegistry::all_euclidean(),
            latent_id: splitmix64(atom as u64),
        },
    ))
}

/// Scalable `"auto"` policy for the overcomplete support lane. At K > P the
/// per-atom fit-entry evidence race (#2238) is statistically vacuous — a
/// hard-TopK dictionary sees ~N·s/K rows per atom — so `"auto"` seeds a
/// UNIFORM, cyclic-bias-free topology portfolio (linear / euclidean curve /
/// periodic, round-robin by atom index) and lets the support competition, the
/// LAML-selected final-function seminorm, and the coordinate ARD prior
/// adjudicate which kinds survive where. Deterministic and target-free, so
/// admission and seeding resolve identically.
pub fn resolve_support_auto_atoms(atom_basis: &mut [String]) {
    for (atom, basis) in atom_basis.iter_mut().enumerate() {
        if basis == "auto" {
            *basis = match atom % 3 {
                0 => "linear",
                1 => "euclidean",
                _ => "periodic",
            }
            .to_string();
        }
    }
}

fn resolve_support_atoms(
    atom_basis: &[String],
    atom_dim: &[usize],
) -> Result<
    (
        Vec<SaeAtomBasisKind>,
        Vec<usize>,
        Vec<SaeAssignmentAtomSpec>,
    ),
    String,
> {
    if atom_basis.len() != atom_dim.len() {
        return Err(format!(
            "support-sparse atom metadata lengths differ: basis={}, dims={}",
            atom_basis.len(),
            atom_dim.len()
        ));
    }
    let mut atom_kinds = Vec::with_capacity(atom_basis.len());
    let mut effective_atom_dim = Vec::with_capacity(atom_basis.len());
    let mut atom_specs = Vec::with_capacity(atom_basis.len());
    for atom in 0..atom_basis.len() {
        let kind = sae_atom_basis_kind_from_str(&atom_basis[atom])
            .map_err(|error| format!("support-sparse atom {atom}: {error}"))?;
        let (latent_dim, spec) = effective_atom(atom_dim[atom], &kind, atom)?;
        atom_kinds.push(kind);
        effective_atom_dim.push(latent_dim);
        atom_specs.push(spec);
    }
    Ok((atom_kinds, effective_atom_dim, atom_specs))
}

/// Resolve public atom dimensions to the actual heterogeneous chart widths
/// charged by support-sparse admission. In particular, a periodic atom's
/// public dimension selects harmonic resolution while its live chart is 1-D.
pub fn sae_support_effective_atom_dims(
    atom_basis: &[String],
    atom_dim: &[usize],
) -> Result<Vec<usize>, String> {
    resolve_support_atoms(atom_basis, atom_dim).map(|(_, dimensions, _)| dimensions)
}

pub(super) fn chart_coordinate(kind: &SaeAtomBasisKind, axis: usize, raw: f64) -> f64 {
    match kind {
        SaeAtomBasisKind::Periodic | SaeAtomBasisKind::Torus | SaeAtomBasisKind::KleinBottle => {
            0.5 + raw.atan() / std::f64::consts::PI
        }
        // The ambient sphere needs no per-axis squashing: its coordinate is a
        // DIRECTION, and the only constraint (unit norm) couples all three axes,
        // so no per-axis function can express it. The raw projection is passed
        // through as a direction and the manifold normalises the block -- see
        // the projection step in `build_sae_support_seed`.
        SaeAtomBasisKind::Sphere | SaeAtomBasisKind::ProjectivePlane => raw,
        SaeAtomBasisKind::Mobius if axis == 0 => 1.0 + 2.0 * raw.atan() / std::f64::consts::PI,
        SaeAtomBasisKind::Mobius => raw.tanh(),
        _ => raw,
    }
}

/// Construct the canonical overcomplete TopK cold start without allocating a
/// `K×N`, `N×K`, or padded `K×N×D` array.
pub fn build_sae_support_seed(
    request: SaeSupportSeedRequest<'_>,
) -> Result<SaeSupportSeedReport, String> {
    let (n_obs, output_dim) = request.target.dim();
    let k_atoms = request.atom_basis.len();
    if n_obs == 0 || output_dim == 0 || k_atoms == 0 {
        return Err(format!(
            "build_sae_support_seed requires positive N, P, and K; got N={n_obs}, P={output_dim}, K={k_atoms}"
        ));
    }
    if request.atom_dim.len() != k_atoms {
        return Err(format!(
            "build_sae_support_seed: atom_dim length {} must equal K={k_atoms}",
            request.atom_dim.len()
        ));
    }
    if k_atoms > u32::MAX as usize {
        return Err(format!(
            "build_sae_support_seed: K={k_atoms} exceeds the canonical u32 support-index range"
        ));
    }
    let admission = request.admission;
    if admission.lane != SaeFitLane::CurvedStreaming
        || admission.n_obs != n_obs
        || admission.output_dim != output_dim
        || admission.n_atoms != k_atoms
    {
        return Err(format!(
            "build_sae_support_seed: admission does not describe this overcomplete curved shape (lane={:?}, admitted N/P/K={}/{}/{}, requested N/P/K={n_obs}/{output_dim}/{k_atoms})",
            admission.lane, admission.n_obs, admission.output_dim, admission.n_atoms
        ));
    }
    let budget = admission.topk_budget.ok_or_else(|| {
        "build_sae_support_seed: curved admission is missing its TopK memory ledger".to_string()
    })?;
    if budget.support_k != request.support_k || !budget.streaming_admitted {
        return Err(format!(
            "build_sae_support_seed: admission ledger mismatch (ledger s={}, requested s={}, streaming_admitted={})",
            budget.support_k, request.support_k, budget.streaming_admitted
        ));
    }

    let (atom_kinds, effective_atom_dim, atom_specs) =
        resolve_support_atoms(request.atom_basis, request.atom_dim)?;
    let d_max = effective_atom_dim.iter().copied().max().unwrap_or(1);
    if d_max != budget.d_max {
        return Err(format!(
            "build_sae_support_seed: admission ledger d_max={} != effective chart d_max={d_max}",
            budget.d_max
        ));
    }

    let mut means = vec![0.0; output_dim];
    for row in request.target.rows() {
        for column in 0..output_dim {
            if !row[column].is_finite() {
                return Err("build_sae_support_seed: target contains a non-finite value".into());
            }
            means[column] += row[column];
        }
    }
    for mean in &mut means {
        *mean /= n_obs as f64;
    }

    // THIS is the seed's cost: `n_obs * k_atoms` scores -- 3.0e9 calls at
    // 250k rows and K=12148, measured at 725 s of wall clock before the first
    // fixed-point cycle, with no output while it ran.
    //
    // Rows are independent. `score` and `projection` read the row's own
    // centred values and `request.random_state`, never an earlier row, so the
    // result is order-independent and `into_par_iter` over a range collects in
    // row order -- the seed is unchanged element for element. The centring
    // scratch moves inside the closure because it is the only piece of
    // per-row mutable state the serial version hoisted out of the loop.
    let seeded: Vec<(Vec<u32>, Vec<f64>, Vec<f64>)> = (0..n_obs)
        .into_par_iter()
        .map(|row| {
        let mut centered = vec![0.0; output_dim];
        for column in 0..output_dim {
            centered[column] = request.target[[row, column]] - means[column];
        }
        let mut selected: Vec<RankedAtom> = Vec::with_capacity(request.support_k);
        for atom in 0..k_atoms {
            let candidate = RankedAtom {
                atom,
                score: score(&centered, atom, request.random_state),
            };
            if selected.len() < request.support_k {
                selected.push(candidate);
                continue;
            }
            let mut worst = 0;
            for slot in 1..selected.len() {
                if better(selected[worst], selected[slot]) {
                    worst = slot;
                }
            }
            if better(candidate, selected[worst]) {
                selected[worst] = candidate;
            }
        }
        selected.sort_by(|lhs, rhs| {
            if better(*lhs, *rhs) {
                std::cmp::Ordering::Less
            } else if better(*rhs, *lhs) {
                std::cmp::Ordering::Greater
            } else {
                std::cmp::Ordering::Equal
            }
        });
        let mut row_indices = Vec::with_capacity(request.support_k);
        let mut row_gates = Vec::with_capacity(request.support_k);
        let mut row_coords = Vec::with_capacity(
            selected
                .iter()
                .map(|entry| effective_atom_dim[entry.atom])
                .sum(),
        );
        for entry in selected {
            row_indices.push(entry.atom as u32);
            row_gates.push(entry.score);
            let block_start = row_coords.len();
            for axis in 0..effective_atom_dim[entry.atom] {
                let raw = projection(&centered, entry.atom, axis + 1, request.random_state);
                row_coords.push(chart_coordinate(&atom_kinds[entry.atom], axis, raw));
            }
            // A seed must LIE ON the manifold it seeds. `chart_coordinate` is
            // per-axis and so cannot express any constraint that couples axes --
            // the unit norm of an ambient sphere coordinate is not a property
            // any single axis has. The manifold is the sole authority for its
            // own point set, so the freshly-built block is projected onto it
            // before the seed is accepted. On Euclidean, circle and interval
            // charts `project_point` is the identity or the wrap/clamp those
            // charts already imply, so no existing seed moves.
            let block = ndarray::Array1::from_vec(row_coords[block_start..].to_vec());
            let projected = atom_specs[entry.atom].manifold.project_point(block.view());
            row_coords[block_start..].copy_from_slice(
                projected
                    .as_slice()
                    .expect("a projected coordinate block is contiguous"),
            );
        }
        (row_indices, row_gates, row_coords)
        })
        .collect();
    let mut indices = Vec::with_capacity(n_obs);
    let mut gates = Vec::with_capacity(n_obs);
    let mut coords = Vec::with_capacity(n_obs);
    for (row_indices, row_gates, row_coords) in seeded {
        indices.push(row_indices);
        gates.push(row_gates);
        coords.push(row_coords);
    }
    // A hard-support dictionary has no likelihood term for an atom that occurs
    // in zero rows. Keeping such an atom would add an unidentifiable decoder
    // block and a singular evidence direction. Remove it at the seed boundary
    // and remap supports once, in ascending original-atom order.
    let mut occupied = vec![false; k_atoms];
    for row in &indices {
        for &atom in row {
            occupied[atom as usize] = true;
        }
    }
    let retained_atom_indices = occupied
        .iter()
        .enumerate()
        .filter_map(|(atom, &used)| used.then_some(atom))
        .collect::<Vec<_>>();
    let mut remap = vec![usize::MAX; k_atoms];
    for (new, &old) in retained_atom_indices.iter().enumerate() {
        remap[old] = new;
    }
    for row in &mut indices {
        for atom in row {
            *atom = remap[*atom as usize] as u32;
        }
    }
    let atom_specs = retained_atom_indices
        .iter()
        .map(|&atom| atom_specs[atom].clone())
        .collect::<Vec<_>>();
    let atom_kinds = retained_atom_indices
        .iter()
        .map(|&atom| atom_kinds[atom].clone())
        .collect::<Vec<_>>();
    let effective_atom_dim = retained_atom_indices
        .iter()
        .map(|&atom| effective_atom_dim[atom])
        .collect::<Vec<_>>();
    let assignment = SaeAssignmentState::from_topk_support_heterogeneous(
        n_obs,
        retained_atom_indices.len(),
        request.support_k,
        atom_specs,
        indices,
        gates,
        coords,
    )?;
    Ok(SaeSupportSeedReport {
        assignment,
        atom_kinds,
        effective_atom_dim,
        retained_atom_indices,
        peak_score_cells: request.support_k,
    })
}

fn bounded_atom_chart_samples(
    assignment: &SaeAssignmentState,
    atom: usize,
    seed_width: usize,
    wanted: usize,
    random_state: u64,
) -> Array2<f64> {
    let effective_dim = assignment.atom_coord_dim(atom);
    let mut observed = Vec::<Vec<f64>>::new();
    for row in 0..assignment.n_obs() {
        if let Ok(slot) = assignment
            .support_indices(row)
            .binary_search(&(atom as u32))
        {
            observed.push(assignment.coords_for_slot(row, slot).to_vec());
        }
    }
    let mut means = vec![0.0; effective_dim];
    for sample in &observed {
        for axis in 0..effective_dim {
            means[axis] += sample[axis];
        }
    }
    if !observed.is_empty() {
        for mean in &mut means {
            *mean /= observed.len() as f64;
        }
    }
    let mut scales = vec![1.0; effective_dim];
    if observed.len() > 1 {
        for axis in 0..effective_dim {
            let variance = observed
                .iter()
                .map(|sample| (sample[axis] - means[axis]).powi(2))
                .sum::<f64>()
                / observed.len() as f64;
            if variance.is_finite() && variance > f64::EPSILON {
                scales[axis] = variance.sqrt();
            }
        }
    }
    let rows = wanted.max(1);
    let mut out = Array2::<f64>::zeros((rows, seed_width));
    let retained = observed.len().min(rows);
    let retained_indices = sae_pick_duchon_center_indices(
        observed.len(),
        retained,
        random_state.wrapping_add(atom as u64),
    );
    for (row, source) in retained_indices.into_iter().enumerate() {
        for axis in 0..effective_dim {
            out[[row, axis]] = observed[source][axis];
        }
    }
    for row in retained..rows {
        for axis in 0..seed_width {
            let hash = splitmix64(
                random_state
                    ^ (atom as u64).wrapping_mul(0xd6e8_feb8_6659_fd93)
                    ^ (row as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15)
                    ^ (axis as u64).wrapping_mul(0xa5a3_564e_27f8_864d),
            );
            let unit = ((hash >> 11) as f64) * (1.0 / ((1_u64 << 53) as f64));
            if axis < effective_dim {
                out[[row, axis]] = means[axis] + scales[axis] * (2.0 * unit - 1.0);
            } else {
                out[[row, axis]] = 2.0 * unit - 1.0;
            }
        }
    }
    out
}

/// Build analytic atom templates from a support seed one atom at a time. The
/// largest observation-indexed allocation is one `(1, M_k, d_k)` basis jet;
/// no K-wide observation tensor is constructed.
pub fn build_sae_support_term_seed(
    request: SaeSupportTermSeedRequest,
) -> Result<SaeSupportTermSeedReport, String> {
    let k_atoms = request.assignment.k_atoms();
    if request.atom_basis.len() != k_atoms || request.atom_dim.len() != k_atoms {
        return Err(format!(
            "build_sae_support_term_seed: metadata lengths basis={}, dims={} must equal K={k_atoms}",
            request.atom_basis.len(),
            request.atom_dim.len()
        ));
    }
    if request.output_dim == 0 {
        return Err("build_sae_support_term_seed: output_dim must be positive".into());
    }
    // Each atom's seed depends only on its own index: the planner is handed
    // `random_state.wrapping_add(atom)`, every input is read through a shared
    // reference, and nothing here observes an earlier iteration. So this is
    // order-independent and parallelises BIT-IDENTICALLY -- `into_par_iter`
    // over a range collects in index order, and the per-atom seed makes the
    // draw independent of which worker ran it.
    //
    // It was worth doing because the phase is not small: seeding K=12148 took
    // 725 s of wall clock (59.7 ms/atom) before the first fixed-point cycle,
    // with no output at all while it ran.
    let seeded = (0..k_atoms)
        .into_par_iter()
        .map(|atom| -> Result<(SaeManifoldAtom, super::SaeAtomBuildPlan), String> {
        let effective_dim = request.assignment.atom_coord_dim(atom);
        let public_dim = request.atom_dim[atom];
        let kind = sae_atom_basis_kind_from_str(&request.atom_basis[atom])
            .map_err(|error| format!("build_sae_support_seed: atom {atom}: {error}"))?;
        let design_rows = planner_design_rows(&kind);
        // `sae_build_atom_plans` interprets periodic public_dim as harmonic
        // order before reducing to a 1-D chart, so its temporary seed width
        // must cover both the public and effective dimensions.
        let seed_width = public_dim.max(effective_dim);
        let chart_samples = bounded_atom_chart_samples(
            &request.assignment,
            atom,
            seed_width,
            design_rows,
            request.random_state,
        );
        let mut plan_seed = Array3::<f64>::zeros((1, design_rows, seed_width));
        plan_seed.slice_mut(s![0, .., ..]).assign(&chart_samples);
        let dummy_target = Array2::<f64>::zeros((design_rows, 1));
        let mut plans = sae_build_atom_plans(
            dummy_target.view(),
            std::slice::from_ref(&request.atom_basis[atom]),
            std::slice::from_ref(&public_dim),
            plan_seed.view(),
            request.random_state.wrapping_add(atom as u64),
            &[None],
        )?;
        let plan = plans.pop().ok_or_else(|| {
            "build_sae_support_term_seed: atom planner returned no plan".to_string()
        })?;
        if plan.latent_dim() != effective_dim {
            return Err(format!(
                "build_sae_support_term_seed: atom {atom} plan latent dim {} != sparse state dim {effective_dim}",
                plan.latent_dim()
            ));
        }
        let mut probe_seed = Array3::<f64>::zeros((1, 1, effective_dim));
        for axis in 0..effective_dim {
            probe_seed[[0, 0, axis]] = chart_samples[[0, axis]];
        }
        let (phi_stack, jet_stack, penalty_stack, basis_sizes, _) =
            sae_build_padded_basis_stacks(std::slice::from_ref(&plan), probe_seed.view(), 1)?;
        let evaluator = plan.geometry.build_evaluator()?;
        let m = basis_sizes[0];
        let phi = phi_stack.slice(s![0, 0..1, 0..m]).to_owned();
        let jet = jet_stack
            .slice(s![0, 0..1, 0..m, 0..effective_dim])
            .to_owned();
        let reference = SaeReferenceRoughness::ProvidedFunctionGram(
            penalty_stack.slice(s![0, 0..m, 0..m]).to_owned(),
        );
        let atom_template = SaeManifoldAtom::new(
            format!("atom_{atom}"),
            kind,
            effective_dim,
            phi,
            jet,
            Array2::<f64>::zeros((m, request.output_dim)),
            reference,
        )?
        .with_basis_second_jet(evaluator)
        .with_geometry_plan(plan.geometry.clone())?;
            Ok((atom_template, plan))
        })
        .collect::<Result<Vec<_>, String>>()?;
    let (atoms, atom_plans): (Vec<_>, Vec<_>) = seeded.into_iter().unzip();
    let term = super::SaeSupportSparseTerm::new(atoms, request.assignment)?;
    Ok(SaeSupportTermSeedReport { term, atom_plans })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::front_door::admit_topk_manifold_with_budget;
    use ndarray::array;

    fn admitted(n: usize, p: usize, k: usize, d: usize, s: usize) -> SaeFitAdmission {
        admit_topk_manifold_with_budget(n, p, k, d, s, usize::MAX).expect("admitted")
    }

    /// The #2502 mixed portfolio, end to end through BOTH seed stages, in
    /// milliseconds. This is the micro-repro for the #2602 sphere seam: the
    /// sparse seed builds a 3-wide ambient state for a public-dim-2 sphere,
    /// and the term-seed planner must ask for the geometry by that width --
    /// requesting the public width selected the chart form and refused its
    /// own seed. A 250k-row fit was the only thing that exercised this path.
    #[test]
    fn mixed_portfolio_with_spheres_seeds_end_to_end() {
        let n = 24usize;
        let p = 6usize;
        let target = Array2::from_shape_fn((n, p), |(row, col)| {
            ((row * 7 + col * 3) as f64 * 0.37).sin()
        });
        let k = 10usize;
        let basis: Vec<String> = (0..k)
            .map(|atom| {
                match atom % 6 {
                    0 => "linear",
                    1 => "euclidean",
                    2 => "periodic",
                    3 => "torus",
                    _ => "sphere",
                }
                .to_string()
            })
            .collect();
        let dims: Vec<usize> = basis
            .iter()
            .map(|b| if b == "sphere" || b == "torus" { 2 } else { 1 })
            .collect();
        let seed = build_sae_support_seed(SaeSupportSeedRequest {
            target: target.view(),
            atom_basis: &basis,
            atom_dim: &dims,
            support_k: 3,
            random_state: 11,
            // admission is charged at the EFFECTIVE chart width (the ambient
            // sphere is 3-wide), exactly as the front door computes it via
            // sae_support_effective_atom_dims.
            admission: admitted(n, p, k, 3, 3),
        })
        .expect("sparse seed accepts the mixed portfolio");
        // Retention drops unrouted atoms; the term seed takes the RETAINED
        // metadata, exactly as the fitting harness maps it.
        let retained_basis: Vec<String> = seed
            .retained_atom_indices
            .iter()
            .map(|&atom| basis[atom].clone())
            .collect();
        let retained_dims: Vec<usize> = seed
            .retained_atom_indices
            .iter()
            .map(|&atom| dims[atom])
            .collect();
        let report = build_sae_support_term_seed(SaeSupportTermSeedRequest {
            assignment: seed.assignment,
            atom_basis: retained_basis.clone(),
            atom_dim: retained_dims,
            output_dim: p,
            random_state: 11,
        })
        .expect("term seed accepts the sparse state the seed built");
        assert!(
            retained_basis.iter().any(|b| b == "sphere"),
            "the retention must keep at least one sphere for this test to bite"
        );
        for (atom, plan) in report.atom_plans.iter().enumerate() {
            if retained_basis[atom] == "sphere" {
                assert_eq!(
                    plan.latent_dim(),
                    3,
                    "sphere atom {atom} must carry the ambient (pole-free) chart"
                );
            }
        }
    }

    /// #2502: a periodic atom whose routed tokens sit on ONE contiguous arc
    /// (wrapping the phase seam) is a bounded curve wearing a circle. The
    /// occupancy census must unroll exactly that atom to a Euclidean chart
    /// with in-range, order-preserving coordinates, and leave the
    /// fully-occupied loop alone. The assignment is constructed directly so
    /// the fixture does not depend on seed-time routing luck.
    #[test]
    fn under_occupied_loop_unrolls_to_euclidean() {
        let n = 24usize;
        let p = 5usize;
        let k = 2usize;
        let kind = sae_atom_basis_kind_from_str("periodic").expect("periodic kind");
        let specs: Vec<SaeAssignmentAtomSpec> = (0..k)
            .map(|atom| SaeAssignmentAtomSpec {
                latent_dim: 1,
                id_mode: gam_terms::latent::LatentIdMode::None,
                manifold: kind.latent_manifold(1),
                retraction: gam_problem::LatentRetractionRegistry::all_euclidean(),
                latent_id: atom as u64 + 1,
            })
            .collect();
        let period = match specs[0].manifold {
            gam_terms::latent::LatentManifold::Circle { period } => period,
            ref other => panic!("periodic kind must chart a circle; got {other:?}"),
        };
        let indices: Vec<Vec<u32>> = (0..n).map(|_| vec![0u32, 1u32]).collect();
        let gate_params = vec![vec![1.0_f64, 1.0]; n];
        // atom 0: one 15% arc THROUGH the seam; atom 1: the full circle.
        let coords: Vec<Vec<f64>> = (0..n)
            .map(|row| {
                let phase = row as f64 / n as f64;
                let arc = (0.925 + 0.15 * phase).rem_euclid(1.0) * period;
                vec![arc, phase * period]
            })
            .collect();
        let assignment = SaeAssignmentState::from_topk_support_heterogeneous(
            n, k, 2, specs, indices, gate_params, coords,
        )
        .expect("hand-built sparse state");
        let mut term = build_sae_support_term_seed(SaeSupportTermSeedRequest {
            assignment,
            atom_basis: vec!["periodic".to_string(); k],
            atom_dim: vec![1usize; k],
            output_dim: p,
            random_state: 11,
        })
        .expect("term seed")
        .term;
        let converted = term
            .convert_underoccupied_loops(3)
            .expect("census runs");
        assert_eq!(converted, vec![0], "exactly the arc-bound loop unrolls");
        assert_eq!(
            term.assignment.atom_axis_periods(0),
            vec![None],
            "the unrolled atom is Euclidean"
        );
        assert!(
            term.assignment.atom_axis_periods(1)[0].is_some(),
            "the fully-occupied loop keeps its topology"
        );
        let mut previous = f64::NEG_INFINITY;
        for row in 0..n {
            let t = term.assignment.coords_for_slot(row, 0)[0];
            assert!(
                (-1.0..=1.0).contains(&t),
                "row {row} unwrapped coordinate {t} must lie in the chart"
            );
            assert!(
                t >= previous,
                "unwrap must preserve arc order through the seam (row {row}: {t} < {previous})"
            );
            previous = t;
        }
    }

    /// #2502 DoF-priced admission: disarmed pricing must be bit-identical to
    /// the unpriced router, and an absurd price must push support toward the
    /// smaller-basis atoms (the A/B's "absurd arm" in unit form).
    #[test]
    fn admission_pricing_disarmed_is_identity_and_absurd_price_prefers_thrift() {
        let n = 24usize;
        let p = 5usize;
        let target = Array2::from_shape_fn((n, p), |(row, col)| {
            ((row * 3 + col * 5) as f64 * 0.41).sin()
        });
        let basis: Vec<String> = (0..8)
            .map(|atom| if atom % 2 == 0 { "linear" } else { "euclidean" }.to_string())
            .collect();
        let dims = vec![1usize; basis.len()];
        let seed = build_sae_support_seed(SaeSupportSeedRequest {
            target: target.view(),
            atom_basis: &basis,
            atom_dim: &dims,
            support_k: 2,
            random_state: 5,
            admission: admitted(n, p, basis.len(), 1, 2),
        })
        .expect("sparse seed");
        let retained_basis: Vec<String> = seed
            .retained_atom_indices
            .iter()
            .map(|&atom| basis[atom].clone())
            .collect();
        let retained_dims: Vec<usize> = seed
            .retained_atom_indices
            .iter()
            .map(|&atom| dims[atom])
            .collect();
        let term = build_sae_support_term_seed(SaeSupportTermSeedRequest {
            assignment: seed.assignment,
            atom_basis: retained_basis.clone(),
            atom_dim: retained_dims,
            output_dim: p,
            random_state: 9,
        })
        .expect("term seed")
        .term;
        let unpriced = term
            .reroute_fixed_decoder(target.view(), 2, 0)
            .expect("unpriced route");
        let mut disarmed = term.clone();
        disarmed.set_admission_dof_pricing(None);
        let disarmed = disarmed
            .reroute_fixed_decoder(target.view(), 2, 0)
            .expect("disarmed route");
        for row in 0..n {
            assert_eq!(
                unpriced.assignment.support_indices(row),
                disarmed.assignment.support_indices(row),
                "row {row}: disarmed pricing must not change routing"
            );
        }
        // Absurd price: wide-basis atoms must not GAIN support share.
        let mut priced = term.clone();
        priced.set_admission_dof_pricing(Some(1.0e6));
        let priced = priced
            .reroute_fixed_decoder(target.view(), 2, 0)
            .expect("priced route");
        let width = |basis_name: &str| retained_basis
            .iter()
            .filter(|b| b.as_str() == basis_name)
            .count();
        assert!(width("euclidean") > 0, "fixture needs wide atoms retained");
        let wide_share = |routed: &crate::manifold::SaeSupportSparseTerm| {
            let mut wide = 0usize;
            let mut total = 0usize;
            for row in 0..n {
                for &atom in routed.assignment.support_indices(row) {
                    total += 1;
                    if retained_basis[atom as usize] == "euclidean" {
                        wide += 1;
                    }
                }
            }
            wide as f64 / total.max(1) as f64
        };
        assert!(
            wide_share(&priced) <= wide_share(&unpriced) + 1.0e-12,
            "an absurd DoF price must not increase the wide-basis support share \
             (priced {:.3} vs unpriced {:.3})",
            wide_share(&priced),
            wide_share(&unpriced)
        );
    }

    #[test]
    fn heterogeneous_seed_is_unpadded_and_deterministic() {
        let target = array![[1.0, -2.0, 0.5], [0.5, 3.0, -0.25]];
        let basis = vec![
            "periodic".into(),
            "sphere".into(),
            "euclidean".into(),
            "mobius".into(),
        ];
        let dims = vec![3, 2, 3, 2];
        let build = || {
            build_sae_support_seed(SaeSupportSeedRequest {
                target: target.view(),
                atom_basis: &basis,
                atom_dim: &dims,
                support_k: 3,
                random_state: 19,
                admission: admitted(2, 3, 4, 3, 3),
            })
            .expect("seed")
        };
        let (first, second) = (build(), build());
        for row in 0..2 {
            assert_eq!(
                first.assignment.support_indices(row),
                second.assignment.support_indices(row)
            );
            assert_eq!(
                first.assignment.coords_row(row),
                second.assignment.coords_row(row)
            );
            let expected: usize = first
                .assignment
                .support_indices(row)
                .iter()
                .map(|&atom| first.effective_atom_dim[atom as usize])
                .sum();
            assert_eq!(first.assignment.coords_row(row).len(), expected);
        }
    }

    #[test]
    fn seed_refuses_discarded_admission() {
        let target = array![[1.0, 2.0]];
        let err = build_sae_support_seed(SaeSupportSeedRequest {
            target: target.view(),
            atom_basis: &["periodic".into()],
            atom_dim: &[1],
            support_k: 1,
            random_state: 0,
            admission: crate::front_door::admit_sae_fit(1, 2, 1).expect("dense"),
        })
        .err()
        .expect("refused");
        assert!(err.contains("admission does not describe"));
    }
}

/// Parts of a fitted support-sparse term, as recovered from a serialized model.
pub struct SaeSupportRehydrateRequest {
    pub atom_basis: Vec<String>,
    pub atom_dim: Vec<usize>,
    pub output_dim: usize,
    pub support_k: usize,
    pub random_state: u64,
    pub support_indices: Vec<Vec<u32>>,
    pub support_values: Vec<Vec<f64>>,
    pub coords: Vec<Vec<f64>>,
    pub decoder_blocks: Vec<Array2<f64>>,
}

/// Rebuild a fitted support-sparse term from its serialized parts (#2567).
///
/// The overcomplete lane could serialize a fit and never reopen it, so every
/// downstream analysis had to run inside the fitting process and a lost process
/// meant a lost fit. SPEC-20 requires work to survive walls via
/// checkpoint/resume; this is the inverse that makes that possible.
///
/// Atom construction is deliberately routed back through
/// [`build_sae_support_term_seed`] rather than reimplemented here: the basis
/// evaluators, chart plans and effective dimensions are exactly the parts that
/// must agree with the fitting path, so they are produced by the same code that
/// produces them during a fit. Only the fitted decoder coefficients are then
/// substituted, and the term is rebuilt through `SaeSupportSparseTerm::new`
/// so the support inversion is recomputed rather than carried across.
pub fn rehydrate_sae_support_term(
    request: SaeSupportRehydrateRequest,
) -> Result<super::SaeSupportSparseTerm, String> {
    let k_atoms = request.atom_basis.len();
    if k_atoms == 0 {
        return Err("rehydrate_sae_support_term: K must be positive".into());
    }
    if request.decoder_blocks.len() != k_atoms {
        return Err(format!(
            "rehydrate_sae_support_term: decoder_blocks length {} must equal K={k_atoms}",
            request.decoder_blocks.len()
        ));
    }
    let n_obs = request.support_indices.len();
    if n_obs == 0 {
        return Err("rehydrate_sae_support_term: N must be positive".into());
    }
    let (_atom_kinds, _effective_atom_dim, atom_specs) =
        resolve_support_atoms(&request.atom_basis, &request.atom_dim)?;
    let assignment = SaeAssignmentState::from_topk_support_heterogeneous(
        n_obs,
        k_atoms,
        request.support_k,
        atom_specs,
        request.support_indices,
        request.support_values,
        request.coords,
    )?;
    let seeded = build_sae_support_term_seed(SaeSupportTermSeedRequest {
        assignment,
        atom_basis: request.atom_basis,
        atom_dim: request.atom_dim,
        output_dim: request.output_dim,
        random_state: request.random_state,
    })?;
    let mut atoms = seeded.term.atoms.clone();
    let assignment = seeded.term.assignment.clone();
    for (atom, decoder) in request.decoder_blocks.into_iter().enumerate() {
        let planned = atoms[atom].decoder_coefficients().dim();
        if decoder.dim() != planned {
            return Err(format!(
                "rehydrate_sae_support_term: atom {atom} decoder shape {:?} != planned {:?}",
                decoder.dim(),
                planned
            ));
        }
        atoms[atom].set_decoder_coefficients(decoder)?;
    }
    super::SaeSupportSparseTerm::new(atoms, assignment)
}
