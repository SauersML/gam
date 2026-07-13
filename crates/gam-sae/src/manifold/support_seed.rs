//! Direct support-sparse cold starts for overcomplete hard-TopK SAE fits.
//!
//! The dense seed's `K×N×D` PCA tensor and `N×K` routing matrix are not valid
//! representations when only `s = top_k` atoms exist in each row's local
//! problem. This module consumes the typed front-door admission and constructs
//! [`SaeAssignmentState`] directly with an `O(s)` row-local score window and
//! heterogeneous, unpadded coordinate blocks.

use crate::assignment_state::{SaeAssignmentAtomSpec, SaeAssignmentState};
use crate::front_door::{SaeFitAdmission, SaeFitLane};
use ndarray::ArrayView2;

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
    /// Maximum atom-score cells retained simultaneously, independent of `K`.
    pub peak_score_cells: usize,
}

#[derive(Clone, Copy)]
struct RankedAtom {
    atom: usize,
    score: f64,
}

fn splitmix64(mut value: u64) -> u64 {
    value = value.wrapping_add(0x9e37_79b9_7f4a_7c15);
    value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    value ^ (value >> 31)
}

/// Bounded-work CountSketch projection. At small P each coordinate appears in
/// the cyclic hash permutation; at large P eight deterministic samples keep
/// routing cost independent of the ambient output width.
fn projection(row: &[f64], atom: usize, axis: usize, random_state: u64) -> f64 {
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

fn chart_coordinate(kind: &SaeAtomBasisKind, axis: usize, raw: f64) -> f64 {
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

    let mut atom_kinds = Vec::with_capacity(k_atoms);
    let mut effective_atom_dim = Vec::with_capacity(k_atoms);
    let mut atom_specs = Vec::with_capacity(k_atoms);
    for atom in 0..k_atoms {
        let kind = sae_atom_basis_kind_from_str(&request.atom_basis[atom]);
        let (latent_dim, spec) = effective_atom(request.atom_dim[atom], &kind, atom)?;
        atom_kinds.push(kind);
        effective_atom_dim.push(latent_dim);
        atom_specs.push(spec);
    }
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
    let assignment = SaeAssignmentState::from_topk_support_heterogeneous(
        n_obs,
        k_atoms,
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
        peak_score_cells: request.support_k,
    })
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
