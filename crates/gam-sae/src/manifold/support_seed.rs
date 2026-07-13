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

use super::{SaeAtomBasisKind, sae_atom_basis_kind_from_str};

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

pub(super) fn splitmix64(mut value: u64) -> u64 {
    value = value.wrapping_add(0x9e37_79b9_7f4a_7c15);
    value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    value ^ (value >> 31)
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
        SaeAtomBasisKind::Sphere | SaeAtomBasisKind::Mobius => {
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

fn resolve_support_atoms(
    atom_basis: &[String],
    atom_dim: &[usize],
) -> Result<(Vec<SaeAtomBasisKind>, Vec<usize>, Vec<SaeAssignmentAtomSpec>), String> {
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
        let kind = sae_atom_basis_kind_from_str(&atom_basis[atom]);
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
        SaeAtomBasisKind::Periodic | SaeAtomBasisKind::Torus => {
            0.5 + raw.atan() / std::f64::consts::PI
        }
        SaeAtomBasisKind::Sphere if axis == 0 => raw.atan(),
        SaeAtomBasisKind::Sphere => std::f64::consts::PI + 2.0 * raw.atan(),
        SaeAtomBasisKind::Mobius if axis == 0 => {
            1.0 + 2.0 * raw.atan() / std::f64::consts::PI
        }
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

    let mut indices = Vec::with_capacity(n_obs);
    let mut gates = Vec::with_capacity(n_obs);
    let mut coords = Vec::with_capacity(n_obs);
    let mut centered = vec![0.0; output_dim];
    for row in 0..n_obs {
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
            for axis in 0..effective_atom_dim[entry.atom] {
                let raw = projection(&centered, entry.atom, axis + 1, request.random_state);
                row_coords.push(chart_coordinate(&atom_kinds[entry.atom], axis, raw));
            }
        }
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
        if let Ok(slot) = assignment.support_indices(row).binary_search(&(atom as u32)) {
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
    let mut atoms = Vec::with_capacity(k_atoms);
    let mut atom_plans = Vec::with_capacity(k_atoms);
    for atom in 0..k_atoms {
        let effective_dim = request.assignment.atom_coord_dim(atom);
        let public_dim = request.atom_dim[atom];
        let kind = sae_atom_basis_kind_from_str(&request.atom_basis[atom]);
        let design_rows = if matches!(
            kind,
            SaeAtomBasisKind::Duchon
                | SaeAtomBasisKind::Linear
                | SaeAtomBasisKind::EuclideanPatch
                | SaeAtomBasisKind::Poincare
        ) {
            32
        } else {
            1
        };
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
        let plan = plans
            .pop()
            .ok_or_else(|| "build_sae_support_term_seed: atom planner returned no plan".to_string())?;
        if plan.latent_dim != effective_dim {
            return Err(format!(
                "build_sae_support_term_seed: atom {atom} plan latent dim {} != sparse state dim {effective_dim}",
                plan.latent_dim
            ));
        }
        let mut probe_seed = Array3::<f64>::zeros((1, 1, effective_dim));
        for axis in 0..effective_dim {
            probe_seed[[0, 0, axis]] = chart_samples[[0, axis]];
        }
        let (phi_stack, jet_stack, penalty_stack, basis_sizes, coord_blocks) =
            sae_build_padded_basis_stacks(std::slice::from_ref(&plan), probe_seed.view(), 1)?;
        let evaluators = build_sae_basis_evaluators(
            std::slice::from_ref(&plan.kind),
            &basis_sizes,
            std::slice::from_ref(&effective_dim),
            &coord_blocks,
            std::slice::from_ref(&plan.duchon_centers),
        )?;
        let evaluator = evaluators
            .into_iter()
            .next()
            .flatten()
            .ok_or_else(|| format!("build_sae_support_term_seed: atom {atom} has no evaluator"))?;
        let m = basis_sizes[0];
        let phi = phi_stack.slice(s![0, 0..1, 0..m]).to_owned();
        let jet = jet_stack
            .slice(s![0, 0..1, 0..m, 0..effective_dim])
            .to_owned();
        let reference = if matches!(kind, SaeAtomBasisKind::Poincare) {
            SaeReferenceRoughness::PoincareConformalDirichlet {
                reference_coords: coord_blocks[0].clone(),
            }
        } else {
            SaeReferenceRoughness::ProvidedFunctionGram(
                penalty_stack.slice(s![0, 0..m, 0..m]).to_owned(),
            )
        };
        let atom_template = SaeManifoldAtom::new(
            format!("atom_{atom}"),
            kind,
            effective_dim,
            phi,
            jet,
            Array2::<f64>::zeros((m, request.output_dim)),
            reference,
        )?
        .with_basis_second_jet(evaluator);
        atoms.push(atom_template);
        atom_plans.push(plan);
    }
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

    #[test]
    fn k_10000_seed_retains_only_active_support() {
        let target = array![[1.0, -2.0], [0.5, 3.0], [-1.0, 0.25]];
        let k = 10_000;
        let basis = vec!["periodic".to_string(); k];
        let dims = vec![1; k];
        let report = build_sae_support_seed(SaeSupportSeedRequest {
            target: target.view(),
            atom_basis: &basis,
            atom_dim: &dims,
            support_k: 2,
            random_state: 7,
            admission: admitted(3, 2, k, 1, 2),
        })
        .expect("seed");
        assert_eq!(report.peak_score_cells, 2);
        assert_eq!(report.assignment.active_state_cells(), 3 * 2 * 3);
        assert!(report.assignment.materialize_dense().is_err());
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
            assert_eq!(first.assignment.support_indices(row), second.assignment.support_indices(row));
            assert_eq!(first.assignment.coords_row(row), second.assignment.coords_row(row));
            let expected: usize = first.assignment.support_indices(row).iter()
                .map(|&atom| first.effective_atom_dim[atom as usize]).sum();
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
