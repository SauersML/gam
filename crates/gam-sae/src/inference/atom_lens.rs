//! Two-score per-atom **lens** (#980, amended): an *additive* per-atom report on
//! a fitted [`SaeManifoldTerm`](crate::manifold::SaeManifoldTerm).
//!
//! # The amendment this file encodes
//!
//! The original #980 framing folded the output-Fisher metric into the SAE
//! *loss* — "replace the Euclidean reconstruction loss by a Fisher-pulled-back
//! loss". That is wrong: it makes the gauge drive the fit, which silently
//! suppresses any structure that is *represented but not currently used*, and it
//! couples the criterion to a quantity (the output-Fisher factors) that is
//! optional and may be absent. The corrected paradigm:
//!
//! * **The SAE fit stays on activations.** The reconstruction likelihood
//!   whitens through the [`RowMetric`](gam_problem::RowMetric)
//!   exactly as before; with the default Euclidean provenance that is the
//!   bit-for-bit isotropic path. The Fisher metric **never** replaces the loss.
//! * **The lens is an additive report.** It reads the *already-fitted* model and
//!   the (optional) `RowMetric`, and emits, per atom, two orthogonal scores plus
//!   their discrepancy. Nothing it computes feeds back into any loss, criterion,
//!   penalty, or optimizer state.
//!
//! # The two scores
//!
//! For each atom `k`:
//!
//! * **presence** (representational, activation-side, *Fisher-free*): how
//!   strongly the atom is encoded *in the activations*. Mean active mass on the
//!   rows where the atom is truly active, times an amplitude-weighted decoder
//!   norm. This is a pure reconstruction-side quantity: it does not touch the
//!   `RowMetric` at all, so it is identical whether or not output-Fisher factors
//!   were supplied. *Everything represented survives* — a loud-but-inert atom is
//!   just as present as a quiet load-bearing one.
//! * **coupling** (behavioral, *the only place Fisher enters*): the output-Fisher
//!   mass along the atom's decoder tangent `dg_k/dt`, averaged over the atom's
//!   active rows. This is computed through
//!   [`RowMetric::fisher_mass`](gam_problem::RowMetric::fisher_mass)
//!   — a *reported* score, never folded into a loss or criterion. Under a
//!   Euclidean / no-Fisher provenance the coupling is **not available** (`None`),
//!   degrading gracefully exactly as the harvest of the Fisher factors is
//!   optional. It is never an error.
//!
//! # The headline: discrepancy
//!
//! `discrepancy = normalized_presence − normalized_coupling`. A high value means
//! **high presence + low coupling**: the atom is strongly *represented* in the
//! activations yet carries almost no behavioral mass — "represented but not
//! currently used", i.e. *thinking it, not saying it*. That is the headline
//! safety number this lens exists to surface. The lens *reports* it; it does not
//! suppress the atom, because suppression would be the loss-replacement mistake
//! the amendment removes.

use ndarray::{ArrayView1, ArrayView2};

use gam_problem::{MetricProvenance, RowMetric};
use crate::manifold::SaeManifoldTerm;

/// Below this active mass a row is not "truly active" for an atom, so it
/// contributes to neither the presence average nor the coupling average. The
/// assignment masses are convex weights in `[0, 1]`; this floor excludes rows
/// where the atom is essentially off (numerical dust) from the per-atom
/// averages, so a globally-near-zero atom does not get a spuriously large
/// amplitude-per-active-row.
pub const SAE_TRUST_ACTIVE_MASS_FLOOR: f64 = 1e-6;

/// One atom's lens entry.
#[derive(Clone, Debug, PartialEq)]
pub struct AtomLensEntry {
    /// The atom's name (mirrors [`crate::manifold::SaeManifoldAtom::name`]).
    pub name: String,
    /// **presence** (representational, activation-side, Fisher-free): mean active
    /// mass on truly-active rows × amplitude-weighted decoder norm. Always
    /// available — it reads only the activation-side fit.
    pub presence: f64,
    /// **coupling** (behavioral): mean output-Fisher mass of the decoder tangent
    /// `dg_k/dt` over the atom's active rows. `None` under a Euclidean /
    /// no-Fisher provenance (the metric carries no behavioral information, so the
    /// score is *not available* — not zero, not an error).
    pub coupling: Option<f64>,
    /// **presence** normalized to `[0, 1]` across the report's atoms (divided by
    /// the max presence; `0` if every atom has zero presence).
    pub presence_normalized: f64,
    /// **coupling** normalized to `[0, 1]` across the report's atoms (divided by
    /// the max coupling). `None` whenever coupling itself is unavailable.
    pub coupling_normalized: Option<f64>,
    /// The headline: `presence_normalized − coupling_normalized`, the
    /// "represented but not currently used" discrepancy. High ⇒ thinking it, not
    /// saying it. `None` when coupling is unavailable (no behavioral axis to
    /// compare presence against).
    pub discrepancy: Option<f64>,
}

impl AtomLensEntry {
    /// Whether this atom reads as **represented but not currently used** —
    /// strong activation presence, weak behavioral coupling. Pure classification
    /// of the already-computed scores; it suppresses nothing.
    ///
    /// Returns `false` when coupling is unavailable (no behavioral axis exists to
    /// declare a discrepancy against).
    pub fn is_represented_not_used(&self) -> bool {
        match self.discrepancy {
            Some(d) => d >= REPRESENTED_NOT_USED_THRESHOLD,
            None => false,
        }
    }

    /// Whether this atom reads as **used** — its behavioral coupling is at least
    /// as strong as its representational presence (non-positive discrepancy).
    /// Returns `false` when coupling is unavailable.
    pub fn is_used(&self) -> bool {
        match self.discrepancy {
            Some(d) => d <= USED_THRESHOLD,
            None => false,
        }
    }
}

/// Discrepancy at or above this flags "represented but not currently used".
/// Presence and coupling are each normalized to `[0, 1]`, so the discrepancy
/// lives in `[-1, 1]`; a value this large means presence outruns coupling by a
/// wide, normalized margin.
const REPRESENTED_NOT_USED_THRESHOLD: f64 = 0.5;

/// Discrepancy at or below this flags "used" (coupling matches or exceeds
/// presence).
const USED_THRESHOLD: f64 = 0.0;

/// The full two-score lens over every atom of a fitted SAE-manifold term.
#[derive(Clone, Debug, PartialEq)]
pub struct AtomTwoLensReport {
    /// One entry per atom, in atom order.
    pub atoms: Vec<AtomLensEntry>,
    /// The provenance of the metric the coupling was read through (or would have
    /// been): `OutputFisher` / `WhitenedStructured` ⇒ coupling available;
    /// `Euclidean` (or no metric installed) ⇒ coupling unavailable. Echoed so a
    /// consumer can certify *why* a coupling is `None`.
    pub coupling_provenance: Option<MetricProvenance>,
}

impl AtomTwoLensReport {
    /// Whether the behavioral coupling axis is available at all (i.e. an
    /// output-Fisher / structured metric was installed). When `false`, every
    /// entry's `coupling`, `coupling_normalized`, and `discrepancy` are `None`.
    pub fn coupling_available(&self) -> bool {
        self.coupling_provenance
            .is_some_and(metric_carries_behavior)
    }
}

/// Does this provenance carry behavioral (output-Fisher) information? Euclidean
/// does not (it is the isotropic activation-only path); the factored
/// provenances do.
fn metric_carries_behavior(p: MetricProvenance) -> bool {
    match p {
        MetricProvenance::Euclidean => false,
        MetricProvenance::OutputFisher { .. }
        | MetricProvenance::OutputFisherDownstream { .. }
        | MetricProvenance::BehavioralFisher { .. }
        | MetricProvenance::WhitenedStructured { .. } => true,
    }
}

/// Build the two-score per-atom lens over a fitted [`SaeManifoldTerm`].
///
/// `model` is the fitted term (read only). `metric` is the per-row inner product
/// the coupling is measured through; pass the model's own installed metric
/// ([`SaeManifoldTerm::row_metric`]) or any metric whose row/output dimensions
/// match the term. When the metric's provenance is Euclidean (no behavioral
/// information), the coupling degrades to `None` for every atom — the lens stays
/// available, only its behavioral axis is absent.
///
/// This function is a *pure read*: it never mutates the model, never touches a
/// loss / criterion / penalty, and the only place the Fisher metric enters is the
/// [`RowMetric::fisher_mass`] call that produces the (reported) coupling score.
pub fn atom_two_lens(
    model: &SaeManifoldTerm,
    metric: &RowMetric,
    assignments_override: Option<ArrayView2<'_, f64>>,
) -> AtomTwoLensReport {
    let n = model.n_obs();
    let k = model.k_atoms();
    let provenance = metric.provenance();
    // Coupling is only meaningful when the metric carries behavioral
    // information *and* its dimensions match the term. A mismatched metric (or a
    // Euclidean one) degrades the behavioral axis to "not available" rather than
    // erroring — the lens is optional, mirroring the harvest being optional.
    let coupling_axis_available = metric_carries_behavior(provenance)
        && metric.n_rows() == n
        && metric.p_out() == model.output_dim();

    // Per-row assignment masses, computed once. When a hard top-k projection has
    // been applied (#1232), the caller supplies the projected matrix so the lens
    // matches the returned payload rather than the smooth optimization assignments.
    let assignments_owned;
    let assignments = match assignments_override {
        Some(view) => view,
        None => {
            assignments_owned = model.assignment.assignments();
            assignments_owned.view()
        }
    };

    let mut presence = vec![0.0_f64; k];
    let mut coupling_raw = vec![0.0_f64; k];
    let mut any_coupling = vec![false; k];

    for (atom_idx, atom) in model.atoms.iter().enumerate() {
        // Amplitude-weighted decoder norm: ‖B_k‖_F. The decoder coefficients
        // B_k ∈ ℝ^{M_k × p} are the linear map from basis activations to the
        // reconstruction output, so their Frobenius norm is the per-atom output
        // amplitude per unit of basis activation — the "how loud is this atom in
        // the reconstruction" factor of presence. Pure activation-side: no
        // metric is consulted.
        let decoder_norm = atom
            .decoder_coefficients
            .iter()
            .map(|&b| b * b)
            .sum::<f64>()
            .sqrt();

        let latent_dim = atom.latent_dim;

        let mut active_mass_sum = 0.0_f64;
        let mut active_row_count = 0.0_f64;
        let mut coupling_sum = 0.0_f64;

        for row in 0..n {
            let mass = assignments[[row, atom_idx]];
            if !(mass > SAE_TRUST_ACTIVE_MASS_FLOOR) {
                continue;
            }
            active_mass_sum += mass;
            active_row_count += 1.0;

            if coupling_axis_available {
                // Behavioral coupling on this active row: the output-Fisher mass
                // of the decoder tangent dg_k/dt summed over the atom's latent
                // axes, weighted by the active mass (so a barely-active row
                // contributes proportionally less behavioral evidence, matching
                // the presence weighting). This is the ONLY place the Fisher
                // metric enters; `fisher_mass` reads no loss / criterion.
                let mut row_tangent_mass = 0.0_f64;
                for axis in 0..latent_dim {
                    let dg = atom.decoded_derivative_row(row, axis);
                    let dg_view: ArrayView1<'_, f64> = dg.view();
                    row_tangent_mass += metric.fisher_mass(row, dg_view);
                }
                coupling_sum += mass * row_tangent_mass;
                any_coupling[atom_idx] = true;
            }
        }

        // Mean active mass on truly-active rows (0 if the atom is active nowhere).
        let mean_active_mass = if active_row_count > 0.0 {
            active_mass_sum / active_row_count
        } else {
            0.0
        };
        presence[atom_idx] = mean_active_mass * decoder_norm;

        // Mean behavioral coupling over the atom's active rows.
        if coupling_axis_available && active_row_count > 0.0 {
            coupling_raw[atom_idx] = coupling_sum / active_row_count;
        }
    }

    // Normalize presence across atoms (divide by the max; 0 when all zero).
    let presence_max = presence.iter().copied().fold(0.0_f64, f64::max);
    // Normalize coupling across atoms, only over atoms with an available score.
    let coupling_max = coupling_raw
        .iter()
        .zip(any_coupling.iter())
        .filter(|&(_, &has)| has)
        .map(|(&c, _)| c)
        .fold(0.0_f64, f64::max);

    let mut entries = Vec::with_capacity(k);
    for (atom_idx, atom) in model.atoms.iter().enumerate() {
        let p = presence[atom_idx];
        let presence_normalized = if presence_max > 0.0 {
            p / presence_max
        } else {
            0.0
        };

        let (coupling, coupling_normalized, discrepancy) =
            if coupling_axis_available && any_coupling[atom_idx] {
                let c = coupling_raw[atom_idx];
                let c_norm = if coupling_max > 0.0 {
                    c / coupling_max
                } else {
                    0.0
                };
                (Some(c), Some(c_norm), Some(presence_normalized - c_norm))
            } else {
                (None, None, None)
            };

        entries.push(AtomLensEntry {
            name: atom.name.clone(),
            presence: p,
            coupling,
            presence_normalized,
            coupling_normalized,
            discrepancy,
        });
    }

    AtomTwoLensReport {
        atoms: entries,
        coupling_provenance: Some(provenance),
    }
}
