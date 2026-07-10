//! Atom basis-width policy and analytic-evaluator rebuild (issue #2236).
//!
//! One home for the width conventions that tie a fitted atom's basis width to
//! its analytic evaluator — the Duchon nullspace knob, the periodic/torus
//! harmonic-width algebra, the Euclidean-patch degree recovery, the cylinder
//! and Möbius production layouts — plus [`build_sae_basis_evaluators`], which
//! rebuilds every atom's second-jet evaluator from that metadata so the inner
//! Newton loop can refresh `Phi_k`/`dPhi_k/dt` without Python in the loop.
//! Moved verbatim from `gam-pyffi` so the CLI, Rust users, and the binding
//! share one policy.

use std::sync::Arc;

use gam_terms::basis::monomial_exponents;
use ndarray::Array2;

use super::{
    CylinderHarmonicEvaluator, DuchonCoordinateEvaluator, EuclideanPatchEvaluator,
    MobiusHarmonicEvaluator, PeriodicHarmonicEvaluator, SaeAtomBasisKind, SaeBasisSecondJet,
    SphereChartEvaluator, TorusHarmonicEvaluator,
};

/// Default per-axis harmonic order for a torus atom (Φ has `(2H+1)^d`
/// columns). Three harmonics per axis gives a 7-column 1-D factor and a
/// 49-column tensor basis at `d=2`, which is the smallest expansion that
/// reliably resolves a non-trivial signal on `T^2` without exploding the
/// design.
pub const SAE_DEFAULT_TORUS_HARMONICS: usize = 3;
/// Sphere chart basis size (lat/lon ⇒ `[1, x, y, z, xy, yz, xz]`).
pub const SAE_SPHERE_BASIS_SIZE: usize = 7;

/// Duchon nullspace knob `m` for a SAE-manifold atom of latent dimension
/// `dim`, sized so the native reproducing-norm Gram (`PenaltySource::Primary`)
/// on the scale-free polyharmonic basis is well-posed in every dimension.
///
/// `m` maps to the polynomial-nullspace order via `duchon_nullspace_from_m`
/// (`m = 1 → Zero/p=1`, `m = 2 → Linear/p=2`, `m = k+1 → Degree(k)/p=k+1`), so
/// `p_order == m`. The pure-polyharmonic basis the `DuchonCoordinateEvaluator`
/// evaluates uses `s = 0`; sizing the null space by `2·m > dim + 2` keeps the
/// kernel CPD/null-space adequacy comfortable across dimensions, whose smallest
/// integer solution is `m = ⌊dim / 2⌋ + 2`.
///
/// For `dim == 1` this reproduces the historical `m = 2` (constant + linear
/// null space). For `dim ≥ 2` it grows the null space with the latent
/// dimension — the previous hard-coded `m = 2` left the resolved power/order
/// inconsistent with the power-0 evaluator. The seed build and every
/// [`DuchonCoordinateEvaluator`] refresh read this same derived `m`, so the
/// design `Φ`, its jet, and the penalty stay column-consistent (the issue-247
/// invariant).
pub fn sae_duchon_atom_m(dim: usize) -> usize {
    dim / 2 + 2
}
/// Maximum total monomial degree for a Euclidean tangent-patch SAE atom.
///
/// IMPORTANT (#1201): this is **degree 2**, so the `"euclidean"` atom basis is a
/// QUADRATIC patch `{1, t, t²}` at `d_atom = 1` (and `{1, t_a, t_b, t_a², t_a t_b,
/// t_b²}` at `d_atom = 2`), NOT a single straight decoder direction `γ(t) = t·b`.
/// Any comparison that calls the `"euclidean"` atom the "linear" baseline is
/// therefore comparing curved-vs-QUADRATIC, not curved-vs-linear — label such
/// comparisons honestly. (A genuinely linear secant baseline is available as the
/// per-atom hybrid-split LINEAR candidate, `crate::terms::sae::hybrid_split`,
/// which fits `b₀ + (t − t̄)·b₁` exactly; the `"euclidean"` OUTER fit path is the
/// quadratic patch.)
pub const SAE_EUCLIDEAN_PATCH_MAX_DEGREE: usize = 2;

/// Upper bound for RECOVERING a EuclideanPatch atom's monomial degree from its
/// trained decoder width (`sae_euclidean_degree_for_basis_size`). The seed patch
/// is degree 2 ([`SAE_EUCLIDEAN_PATCH_MAX_DEGREE`]), but a structure-search BIRTH
/// races a `d=1` line candidate at degree 3 (`gam::terms::sae::structure_harvest`
/// `topology_candidates_for_dim`: `EuclideanPatchEvaluator::new(1, 3)`, width
/// `M = 4`). The OOS rebuild and the inner-Newton basis refresh must recover that
/// born degree from the trained width, so the recovery search reaches degree 3
/// even though no SEED atom is built past degree 2. The per-`d` monomial widths
/// are strictly increasing in the degree (`d=1`: 1,2,3,4; `d=2`: 1,3,6,10), so a
/// width maps back to a unique degree with no collision.
pub const SAE_EUCLIDEAN_PATCH_RECOVERY_MAX_DEGREE: usize = 3;

/// Flat-line polynomial degree of a Cylinder `S¹ × ℝ` atom's line axis (axis 1).
/// Mirrors the Euclidean-patch degree so the cylinder's flat factor matches the
/// patch candidate it races against; `Ml = SAE_CYLINDER_LINE_DEGREE + 1`.
pub const SAE_CYLINDER_LINE_DEGREE: usize = 2;

/// Möbius production convention (#2240): circle harmonics on the DOUBLE-COVER
/// angle and the width monomial degree of the deck-invariant basis. Must match
/// the seed builder and the topology-race candidate (`MobiusHarmonicEvaluator::
/// new(3, 2)`, deck-invariant width 10).
pub const SAE_MOBIUS_CIRCLE_HARMONICS: usize = 3;
pub const SAE_MOBIUS_WIDTH_DEGREE: usize = 2;

/// Recover the cylinder evaluator's `(circle_harmonics H, line_degree D)` from
/// its fitted product basis width `m = (2H + 1)·(D + 1)` and the production
/// line-degree convention (`D = SAE_CYLINDER_LINE_DEGREE`). The line factor width
/// `Ml = D + 1` must divide `m`, and the recovered circle width `Mc = m / Ml`
/// must be odd (`= 2H + 1`); otherwise the basis width is inconsistent with a
/// cylinder product and the error is surfaced rather than guessed.
pub fn sae_cylinder_harmonics_degree(m: usize) -> Result<(usize, usize), String> {
    let d_line = SAE_CYLINDER_LINE_DEGREE;
    let ml = d_line + 1;
    if ml == 0 || m == 0 || m % ml != 0 {
        return Err(format!(
            "sae_cylinder_harmonics_degree: basis size {m} is not (2H+1)·{ml} for a cylinder \
             with line degree {d_line}"
        ));
    }
    let mc = m / ml;
    if mc < 3 || mc % 2 == 0 {
        return Err(format!(
            "sae_cylinder_harmonics_degree: recovered circle width {mc} (= {m}/{ml}) is not an \
             odd 2H+1 ≥ 3 for a cylinder"
        ));
    }
    Ok(((mc - 1) / 2, d_line))
}

pub const SAE_MAX_PERIODIC_HARMONICS: usize = 4096;

pub fn sae_periodic_basis_size(n_harmonics: usize) -> Result<usize, String> {
    if n_harmonics > SAE_MAX_PERIODIC_HARMONICS {
        return Err(format!(
            "sae_build_periodic_atom: n_harmonics={n_harmonics} exceeds dense limit {SAE_MAX_PERIODIC_HARMONICS}"
        ));
    }
    n_harmonics
        .checked_mul(2)
        .and_then(|twice| twice.checked_add(1))
        .ok_or_else(|| {
            format!("sae_build_periodic_atom: basis size overflows for n_harmonics={n_harmonics}")
        })
}


/// Compute the per-axis basis size `axis_m = (m)^(1/d)` for a torus atom and
/// verify that `m = axis_m^d` with `axis_m` odd (i.e. `2H+1`). Returns
/// `axis_m`.
pub fn sae_torus_axis_basis_size(m: usize, d: usize) -> Result<usize, String> {
    if d == 0 {
        return Err("sae_torus_axis_basis_size: d must be >= 1".to_string());
    }
    if m == 0 {
        return Err("sae_torus_axis_basis_size: m must be >= 1".to_string());
    }
    // Integer d-th root via search; m is small (`<= 4096^d` in practice).
    let mut axis_m: usize = 1;
    loop {
        let mut prod: usize = 1;
        let mut overflow = false;
        for _ in 0..d {
            match prod.checked_mul(axis_m) {
                Some(p) => prod = p,
                None => {
                    overflow = true;
                    break;
                }
            }
        }
        if overflow || prod > m {
            return Err(format!(
                "sae_torus_axis_basis_size: m={m} is not a perfect d-th power for d={d}"
            ));
        }
        if prod == m {
            if axis_m % 2 == 0 {
                return Err(format!(
                    "sae_torus_axis_basis_size: m={m} = {axis_m}^{d} but axis size must be odd (2H+1)"
                ));
            }
            return Ok(axis_m);
        }
        axis_m += 1;
    }
}

pub fn sae_euclidean_degree_for_basis_size(dim: usize, basis_size: usize) -> Result<usize, String> {
    // Recover up to the BORN ceiling (degree 3), not just the seed degree 2: a
    // structure-search birth grows a `d=1` EuclideanPatch line at degree 3
    // (`M = 4`), and its OOS rebuild / refresh must map that trained width back to
    // its degree. Widths are strictly increasing in the degree per `dim`, so the
    // recovery is unique.
    for degree in 0..=SAE_EUCLIDEAN_PATCH_RECOVERY_MAX_DEGREE {
        if monomial_exponents(dim, degree).len() == basis_size {
            return Ok(degree);
        }
    }
    Err(format!(
        "euclidean patch basis size {basis_size} is not a valid monomial width for latent_dim={dim} with max_degree<={SAE_EUCLIDEAN_PATCH_RECOVERY_MAX_DEGREE}"
    ))
}

/// Build per-atom Rust basis evaluators so the Newton loop can refresh
/// `Phi_k` and `dPhi_k/dt` between steps without bouncing back to Python.
///
/// Every supported kind has a concrete analytic evaluator: `Periodic`
/// (`latent_dim == 1`) → harmonic, `Sphere` (`latent_dim == 2`) → chart,
/// `Torus` → tensor harmonic, `Duchon` → radial+polynomial coordinate
/// evaluator (requires the atom's centers in `atom_centers`), and
/// `EuclideanPatch` → monomial patch. A `Precomputed` atom — or a kind whose
/// refresh metadata is missing (e.g. a Duchon atom with no centers) — has no
/// way to re-evaluate `Phi(t)` at updated coordinates, so construction errors
/// rather than freezing a stale snapshot.
pub fn build_sae_basis_evaluators(
    basis_kinds: &[SaeAtomBasisKind],
    basis_sizes: &[usize],
    atom_dim: &[usize],
    coord_blocks: &[Array2<f64>],
    atom_centers: &[Option<Array2<f64>>],
) -> Result<Vec<Option<Arc<dyn SaeBasisSecondJet>>>, String> {
    let k_atoms = basis_kinds.len();
    if atom_dim.len() != k_atoms
        || basis_sizes.len() != k_atoms
        || coord_blocks.len() != k_atoms
        || atom_centers.len() != k_atoms
    {
        return Err(format!(
            "build_sae_basis_evaluators: K-length metadata mismatch (kinds={k_atoms}, dims={}, sizes={}, coords={}, centers={})",
            atom_dim.len(),
            basis_sizes.len(),
            coord_blocks.len(),
            atom_centers.len()
        ));
    }
    let mut out: Vec<Option<Arc<dyn SaeBasisSecondJet>>> = Vec::with_capacity(k_atoms);
    for k in 0..k_atoms {
        let m = basis_sizes[k];
        let d = atom_dim[k];
        // Every production atom evaluator implements `SaeBasisSecondJet` (it
        // exposes the analytic basis Hessian). Returning the second-jet trait
        // object lets the term builder install it through
        // `with_basis_second_jet`, which is the slot the #1117 rank-revealing
        // reduction reads to reparametrize a rank-deficient decoder.
        let evaluator: Arc<dyn SaeBasisSecondJet> = match &basis_kinds[k] {
            SaeAtomBasisKind::Periodic if d == 1 && m % 2 == 1 => {
                Arc::new(PeriodicHarmonicEvaluator::new(m)?)
            }
            SaeAtomBasisKind::Sphere if d == 2 && m == SAE_SPHERE_BASIS_SIZE => {
                Arc::new(SphereChartEvaluator)
            }
            SaeAtomBasisKind::Torus if d >= 1 => {
                // Recover the per-axis harmonic count `H` from `m = (2H+1)^d`.
                let axis_m = sae_torus_axis_basis_size(m, d)?;
                let h = (axis_m - 1) / 2;
                Arc::new(TorusHarmonicEvaluator::new(d, h)?)
            }
            SaeAtomBasisKind::Duchon => {
                let centers = atom_centers[k].as_ref().ok_or_else(|| {
                    format!(
                        "build_sae_basis_evaluators: Duchon atom {k} cannot refresh its basis without centers; \
                         build the atom through the SAE auto path so its Duchon centers are threaded in"
                    )
                })?;
                // Same dimension-aware `m` the seed build
                // (`sae_build_duchon_atom`) used: both read `centers.ncols()`,
                // so the refreshed `Φ`/jet stays column-consistent with the
                // seed design and its native (Primary) penalty (issue-247 invariant).
                Arc::new(DuchonCoordinateEvaluator::new(
                    centers.clone(),
                    sae_duchon_atom_m(centers.ncols()),
                )?)
            }
            SaeAtomBasisKind::Cylinder if d == 2 => {
                // Cylinder `S¹ × ℝ`: recover the circle harmonic count `H` and
                // line degree `D` from the fitted product width
                // `m = (2H+1)·(D+1)` (the production line-degree convention), so
                // the refreshed `Φ`/jet stays column-consistent with the seed
                // design. An inconsistent width is surfaced, not guessed.
                let (h, d_line) = sae_cylinder_harmonics_degree(m)?;
                Arc::new(CylinderHarmonicEvaluator::new(h, d_line)?)
            }
            // #1221 — the genuinely-linear (affine) atom is the degree-1 monomial
            // patch `{1, t}`. It shares the `EuclideanPatchEvaluator`, built at
            // `max_degree = 1` recovered from its basis width `m = d + 1` so the
            // refreshed Φ/jet stays column-consistent with the seed design.
            SaeAtomBasisKind::Linear => Arc::new(EuclideanPatchEvaluator::new(
                d,
                sae_euclidean_degree_for_basis_size(d, m)?,
            )?),
            SaeAtomBasisKind::EuclideanPatch | SaeAtomBasisKind::Poincare => {
                // Recover the patch degree from the TRAINED width `m`, not the
                // seed default 2: a structure-search-born `d=1` EuclideanPatch line
                // is degree 3 (`m = 4`), so freezing degree 2 here would re-emit a
                // 3-column Φ that disagrees with the trained 4-row decoder and break
                // the inner-Newton latent refresh / OOS reconstruct on a born line.
                Arc::new(EuclideanPatchEvaluator::new(
                    d,
                    sae_euclidean_degree_for_basis_size(d, m)?,
                )?)
            }
            SaeAtomBasisKind::Mobius if d == 2 => {
                // Möbius production convention: H = 3 double-cover circle
                // harmonics, width degree 2 (deck-invariant width 10, see
                // `MobiusHarmonicEvaluator`). Verify the fitted width instead
                // of guessing an alternative layout.
                let evaluator = MobiusHarmonicEvaluator::new(
                    SAE_MOBIUS_CIRCLE_HARMONICS,
                    SAE_MOBIUS_WIDTH_DEGREE,
                )?;
                if evaluator.basis_size() != m {
                    return Err(format!(
                        "build_sae_basis_evaluators: Mobius atom {k} width {m} does not match \
                         the production deck-invariant layout ({} columns)",
                        evaluator.basis_size()
                    ));
                }
                Arc::new(evaluator)
            }
            SaeAtomBasisKind::Mobius => {
                return Err(format!(
                    "build_sae_basis_evaluators: Mobius atom {k} requires latent_dim == 2; got dim={d}, m={m}"
                ));
            }
            SaeAtomBasisKind::Cylinder => {
                return Err(format!(
                    "build_sae_basis_evaluators: Cylinder atom {k} requires latent_dim == 2; got dim={d}, m={m}"
                ));
            }
            SaeAtomBasisKind::FiniteSet => {
                // A finite-set atom's latent is CATEGORICAL (a discrete anchor
                // index), so it has no continuous Phi(t)/dPhi/dt for the inner
                // Newton latent update to refresh. The candidate is inert
                // scaffolding not enrolled in the topology race
                // (`structure_harvest::finite_set_race_enrolled` is false by
                // default), so it cannot reach here from a discovered dictionary;
                // surface loudly if the enrolment flag is flipped before the
                // first-class continuous-optimizer integration lands, rather than
                // mis-refreshing a categorical basis as a smooth one.
                return Err(format!(
                    "build_sae_basis_evaluators: atom {k} 'finite_set' is a discrete-anchor \
                     (categorical) candidate with no continuous Phi(t) refresh; it is not yet \
                     wired into the inner Newton latent-update path"
                ));
            }
            SaeAtomBasisKind::Precomputed(label) => {
                return Err(format!(
                    "build_sae_basis_evaluators: atom {k} basis {label:?} is precomputed and has no \
                     analytic refresh routine; the inner Newton latent update requires a basis kind \
                     that can re-evaluate Phi(t)/dPhi/dt at updated coordinates"
                ));
            }
            SaeAtomBasisKind::Periodic => {
                return Err(format!(
                    "build_sae_basis_evaluators: Periodic atom {k} requires latent_dim == 1 and odd basis size; got dim={d}, m={m}"
                ));
            }
            SaeAtomBasisKind::Sphere => {
                return Err(format!(
                    "build_sae_basis_evaluators: Sphere atom {k} requires latent_dim == 2 and basis size {SAE_SPHERE_BASIS_SIZE}; got dim={d}, m={m}"
                ));
            }
            SaeAtomBasisKind::Torus => {
                return Err(format!(
                    "build_sae_basis_evaluators: Torus atom {k} requires latent_dim >= 1; got dim={d}, m={m}"
                ));
            }
        };
        out.push(Some(evaluator));
    }
    Ok(out)
}

/// Canonical [`SaeAtomBasisKind`] for a Python-facing basis-kind spelling,
/// resolved through the shared alias schema ([`crate::atom_schema`]).
pub fn sae_atom_basis_kind_from_str(value: &str) -> SaeAtomBasisKind {
    let canonical = crate::atom_schema::canonical_basis_kind(value);
    match canonical.as_str() {
        "duchon" => SaeAtomBasisKind::Duchon,
        "periodic" => SaeAtomBasisKind::Periodic,
        "sphere" => SaeAtomBasisKind::Sphere,
        "torus" => SaeAtomBasisKind::Torus,
        // #1221 — the genuinely-linear (affine) atom: `γ(t) = b₀ + Σ t_a·b_a`,
        // the degree-1 monomial patch. This is the honest "linear" baseline,
        // distinct from `"euclidean"` / `"euclidean_patch"`, which is the degree-2
        // QUADRATIC patch `{1, t, t²}`. `"euclidean_quadratic_patch"` is accepted
        // as an explicit synonym so callers can name the quadratic patch honestly.
        "linear" => SaeAtomBasisKind::Linear,
        // #BSF — `"linear_block"` is a BSF block expressed AS a manifold-SAE atom:
        // γ_g(t) = t·D_g with an orthonormal decoder frame D_g and block-level
        // (norm-selection or separate-gate) gating. Mathematically it IS the
        // `Linear` degree-1 patch (the honest encoding of "BSF ⊂ ManifoldSAE" is a
        // frame + gating CONFIG on the linear atom, not a new basis type), so it
        // maps to `SaeAtomBasisKind::Linear` for construction/evidence; the
        // `"linear_block"` label is preserved by the gamfit facade so an artifact
        // fitted as linear_block round-trips as linear_block (not linear). A
        // first-class `LinearBlock` enum variant was DEFERRED deliberately: it
        // would force exhaustive-match edits across ~10 manifold/ files (a large,
        // then-unverifiable, collision-prone change) for a type-level distinction
        // the frame+gating config already carries. Do NOT "simplify" this alias
        // away — it is load-bearing for the executable BSF-subset claim.
        "linear_block" => SaeAtomBasisKind::Linear,
        "euclidean" => SaeAtomBasisKind::EuclideanPatch,
        "poincare" => SaeAtomBasisKind::Poincare,
        "cylinder" => SaeAtomBasisKind::Cylinder,
        "mobius" => SaeAtomBasisKind::Mobius,
        other => SaeAtomBasisKind::Precomputed(other.to_string()),
    }
}

/// The canonical lowercase string name of an atom basis kind — the inverse of
/// [`sae_atom_basis_kind_from_str`] for the round-trippable kinds, and the
/// string the python `from_payload` boundary reads under each atom's
/// `"basis_kind"` key and each plan's `"kind"` key. Derived from the FITTED
/// atom (not the user's input metadata) so a structure-search-grown / shrunk
/// dictionary serializes its DISCOVERED per-atom topology, not the input one.
pub fn sae_atom_basis_kind_name(kind: &SaeAtomBasisKind) -> String {
    match kind {
        SaeAtomBasisKind::Periodic => "periodic".to_string(),
        SaeAtomBasisKind::Duchon => "duchon".to_string(),
        SaeAtomBasisKind::Sphere => "sphere".to_string(),
        SaeAtomBasisKind::Torus => "torus".to_string(),
        // #1221 — the genuinely-linear atom round-trips under its own honest
        // name. The degree-2 patch keeps its established `"euclidean_patch"` wire
        // name (renaming it would break every consumer reading the serialized
        // dictionary); honesty for it is carried by the distinct `"linear"`
        // topology, the `"euclidean_quadratic_patch"` input synonym, and the
        // documentation that `"euclidean"` is quadratic, not linear.
        SaeAtomBasisKind::Linear => "linear".to_string(),
        SaeAtomBasisKind::EuclideanPatch => "euclidean_patch".to_string(),
        SaeAtomBasisKind::Poincare => "poincare".to_string(),
        SaeAtomBasisKind::Cylinder => "cylinder".to_string(),
        SaeAtomBasisKind::Mobius => "mobius".to_string(),
        // The finite-set (discrete-anchor) candidate is inert scaffolding that is
        // not enrolled in the topology race by default, so a discovered dictionary
        // never actually carries it (see
        // `structure_harvest::finite_set_race_enrolled`). Round-trip it under the
        // same `"finite_set"` token the gam-sae inference/harvest paths already
        // emit, so serialization stays consistent the moment it is enrolled.
        SaeAtomBasisKind::FiniteSet => "finite_set".to_string(),
        SaeAtomBasisKind::Precomputed(name) => name.clone(),
    }
}
