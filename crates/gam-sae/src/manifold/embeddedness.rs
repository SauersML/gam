//! Certified embeddedness of a fitted `d = 1` periodic atom's decoded image
//! (#2518 item 2).
//!
//! # What was missing
//!
//! Every certificate this crate already ships about a `d = 1` circle atom is
//! *infinitesimal* or *reparameterization*-shaped:
//! [`crate::manifold::coordinate_fidelity::atom_coordinate_fidelity`] certifies
//! that the chart carries an honest arc-length coordinate,
//! [`crate::chart_canonicalization`] certifies that the canonicalizing flow is a
//! diffeomorphism *of the latent circle*, and reconstruction EV certifies fit.
//! An atom whose decoder wraps its ambient loop TWICE —
//! `m(t) = (cos 4πt, sin 4πt)` — passes all of them: the parameter circle maps
//! onto the image at constant speed, the arc-length chart is a perfectly good
//! degree-1 reparameterization, the isometry defect is zero, and the fit is
//! exact. But `m(t) = m(t + ½)` for every `t`, so the decoded image is a
//! 2-to-1 immersion, every encode fiber has two points, and the "unique
//! preimage inside a certified ball" claim the encode path makes is locally
//! true and globally worthless.
//!
//! Nothing in the tree could see that, because it is a GLOBAL property of the
//! decoder and every existing guard is local. This module supplies it.
//!
//! # The certificate
//!
//! The decoder of a periodic atom is a trig polynomial
//! (`crate::basis::PeriodicHarmonicEvaluator` layout — column `0` is the
//! constant, columns `2h-1`, `2h` are `sin 2πht`, `cos 2πht`):
//!
//! ```text
//!     m(t) = b₀ + Σ_{h=1}^{H} [ a_h sin(2πht) + b_h cos(2πht) ],   a_h, b_h ∈ ℝ^p.
//! ```
//!
//! `m` is an EMBEDDING iff it is an injective immersion, i.e. iff the
//! separation `m(u + s) − m(u)` vanishes for no `(u, s)` with `s ∈ (0, 1)` and
//! `‖m'‖` vanishes nowhere. Both halves are one positivity statement. Writing
//! the difference with the sum-to-product identities, in the CENTERED
//! coordinate `c = u + s/2`,
//!
//! ```text
//!     m(u+s) − m(u) = 2 sin(πs) · F(c, s),
//!     F(c, s) = Σ_h r_h(s) · [ a_h cos(2πhc) − b_h sin(2πhc) ],
//!     r_h(s)  = sin(πhs) / sin(πs).
//! ```
//!
//! The kernel `r_h` is exactly the Chebyshev polynomial of the second kind:
//! `r_h(s) = U_{h−1}(x)` with `x = cos(πs)`, so `F` is a POLYNOMIAL of degree
//! `H − 1` in `x` and a trig polynomial of degree `H` in `c` — no half-integer
//! frequencies, no division, no removable singularity left in the formula. The
//! factor `2 sin(πs)` carries the entire diagonal zero, and dividing it out is
//! what turns "the difference vanishes on the diagonal by construction" into a
//! statement with content. Define
//!
//! ```text
//!     G(c, x) = ‖F(c, x)‖²   on   c ∈ [0, 1),  x ∈ [−1, 1].
//! ```
//!
//! * `x ∈ (−1, 1)` ⟺ `s ∈ (0, 1)`, where `sin(πs) > 0`, so `G = 0` exactly at a
//!   genuine off-diagonal coincidence `m(u+s) = m(u)`;
//! * `x = 1` is `s → 0`, where `r_h → h` and `G = ‖m'(c)‖² / 4π²` — the
//!   IMMERSION condition, recovered by the same expression;
//! * `x = −1` is `s → 1` (the full wrap), where `G = ‖m'(c + ½)‖² / 4π²`.
//!
//! So **`G > 0` on the whole compact domain ⟺ `m` is an embedding**, and a
//! rigorous positive lower bound on `min G` is a certificate of embeddedness.
//!
//! # Why a grid minimum is not that bound, and what fixes it
//!
//! `G` is sampled on a finite grid, and between nodes it can dip below every
//! value the grid saw. Exactly as
//! [`crate::chart_canonicalization::TorusFlowBasis::certified_min_jacobian_det`]
//! does for the determinant guard, the sample is corrected to a bound by the
//! sup-norm of the gradient, and — for the same reason given there — the bound
//! is taken from the COEFFICIENTS rather than from Bernstein's inequality:
//! Bernstein charges the correction against `‖G‖_∞`, while the coefficient
//! bound is the actual `O(‖B‖²)` modulus of continuity.
//!
//! With `A_h = √(‖a_h‖² + ‖b_h‖²)` and the sharp Chebyshev bounds on `[−1, 1]`
//! (`|U_n| ≤ n + 1`, `|U_n′| ≤ n(n+1)(n+2)/3`, both attained at the endpoints):
//!
//! ```text
//!     ‖F‖   ≤ Σ_h h·A_h                        =: F̄
//!     ‖∂_c F‖ ≤ 2π Σ_h h²·A_h                  =: C̄
//!     ‖∂_x F‖ ≤ Σ_h (h−1)h(h+1)/3 · A_h        =: X̄
//!     |∂_c G| = |2⟨F, ∂_c F⟩| ≤ 2 F̄ C̄,   |∂_x G| ≤ 2 F̄ X̄
//! ```
//!
//! and every point of the domain lies within half a node spacing of a grid node
//! along each axis (the `c` grid wraps with the circle; the `x` grid includes
//! both endpoints), giving the rigorous
//!
//! ```text
//!     min G  ≥  min_grid G  −  (h_c/2)·2F̄C̄  −  (h_x/2)·2F̄X̄.
//! ```
//!
//! # The certificate is ONE-SIDED, and says so
//!
//! `certified_min > 0` PROVES the decoded image is embedded. `certified_min ≤ 0`
//! proves nothing: it means the grid plus its correction could not separate `G`
//! from zero, which happens both for a genuinely folded atom and for an embedded
//! one whose fold margin is finer than the grid resolves. Consumers must read
//! [`AtomEmbeddednessCertificate::embedded`] as "certified embedded", never its
//! negation as "certified folded".

use ndarray::ArrayView2;

use super::atom::{SaeAtomBasisKind, SaeManifoldAtom};
use super::term::SaeManifoldTerm;

/// Nodes on the centered-coordinate axis `c ∈ [0, 1)`.
///
/// The axis is a circle, so the grid wraps and the half-spacing `1/(2N)` is the
/// exact worst-case distance to a node — there is no edge effect to pad. The
/// count is a RESOLUTION, not a threshold: raising it shrinks the correction
/// term monotonically (`∝ 1/N`) and can only turn an uncertified atom into a
/// certified one, never the reverse. It is set so that a unit-scale first
/// harmonic (`Ā = √2`, the plain circle) certifies with better than 90 % of its
/// true margin, which is the regime every fitted circle atom lives in.
pub const EMBEDDEDNESS_CENTER_NODES: usize = 512;

/// Nodes on the separation axis `x = cos(πs) ∈ [−1, 1]`, endpoints included.
///
/// Odd so that `x = 0` (the half-turn `s = ½`, where a double-wrapped decoder
/// has its coincidence) is a node — the one point where an exactly folded atom
/// is caught by the sample rather than only by the correction. Like
/// [`EMBEDDEDNESS_CENTER_NODES`] this is a resolution and not a threshold.
pub(crate) const EMBEDDEDNESS_SEPARATION_NODES: usize = 257;

/// Per-atom certificate that a fitted `d = 1` periodic decoder's image is an
/// embedded circle — produced by [`certify_periodic_decoder_embeddedness`].
///
/// Read [`Self::embedded`] as *certified embedded*; see the module docs on why
/// its negation certifies nothing.
#[derive(Debug, Clone)]
pub struct AtomEmbeddednessCertificate {
    /// Number of non-constant harmonics `H` in the decoder basis.
    pub harmonics: usize,
    /// `min G` over the sample grid — a SAMPLE, not the continuum minimum.
    pub grid_min: f64,
    /// The inter-node correction `(h_c/2)·2F̄C̄ + (h_x/2)·2F̄X̄`, closed form in
    /// the decoder coefficients.
    pub grid_correction: f64,
    /// `grid_min − grid_correction`: a rigorous lower bound on `G` over the
    /// WHOLE domain. Strictly positive ⟺ the decoded image is embedded.
    pub certified_min: f64,
    /// `F̄²`, the closed-form sup bound on `G` — the natural scale to read
    /// [`Self::certified_min`] against, since both are homogeneous of degree 2
    /// in the decoder.
    pub scale: f64,
    /// `certified_min / scale`: the dimensionless certified margin, invariant
    /// under rescaling the decoder (and hence under the atom amplitude `z`).
    pub relative_margin: f64,
    /// `true` iff `certified_min > 0`, i.e. the image is CERTIFIED embedded.
    pub embedded: bool,
    /// Grid resolutions the bound was computed at, so a reported margin can be
    /// reproduced and a refusal attributed to resolution rather than geometry.
    pub center_nodes: usize,
    pub separation_nodes: usize,
}

/// Certify that the periodic decoder `B` (shape `(2H+1, p)`, in the
/// [`crate::basis::PeriodicHarmonicEvaluator`] column layout) has an embedded
/// image.
///
/// The atom amplitude is irrelevant: `G` is homogeneous of degree 2 in `B`, so
/// scaling the decoder scales `certified_min` and `scale` together and leaves
/// [`AtomEmbeddednessCertificate::relative_margin`] and the verdict unchanged.
pub fn certify_periodic_decoder_embeddedness(
    decoder: ArrayView2<'_, f64>,
) -> Result<AtomEmbeddednessCertificate, String> {
    let m = decoder.nrows();
    if m == 0 || m % 2 == 0 {
        return Err(format!(
            "certify_periodic_decoder_embeddedness: periodic decoder needs an odd \
             row count 2H+1; got {m}"
        ));
    }
    let harmonics = (m - 1) / 2;
    if harmonics == 0 {
        // A constant decoder: the image is a point, `F ≡ 0`, and nothing can be
        // certified. Report the honest zero bound rather than erroring — a
        // collapsed atom is a fit outcome, not a caller bug.
        return Ok(AtomEmbeddednessCertificate {
            harmonics: 0,
            grid_min: 0.0,
            grid_correction: 0.0,
            certified_min: 0.0,
            scale: 0.0,
            relative_margin: 0.0,
            embedded: false,
            center_nodes: EMBEDDEDNESS_CENTER_NODES,
            separation_nodes: EMBEDDEDNESS_SEPARATION_NODES,
        });
    }

    // Harmonic coefficient blocks: `a_h` is the sin row, `b_h` the cos row.
    let sin_row = |h: usize| decoder.row(2 * h - 1);
    let cos_row = |h: usize| decoder.row(2 * h);

    // The four coefficient Grams. Every grid node's `H × H` quadratic form is a
    // fixed combination of these, so the ambient width `p` is paid ONCE
    // (`O(H²p)`) instead of once per node — the whole grid then costs `O(H²)`
    // per node regardless of how wide the activations are.
    let mut g_aa = vec![0.0_f64; harmonics * harmonics];
    let mut g_ab = vec![0.0_f64; harmonics * harmonics];
    let mut g_bb = vec![0.0_f64; harmonics * harmonics];
    for h in 1..=harmonics {
        for k in 1..=harmonics {
            let idx = (h - 1) * harmonics + (k - 1);
            g_aa[idx] = sin_row(h).dot(&sin_row(k));
            g_ab[idx] = sin_row(h).dot(&cos_row(k));
            g_bb[idx] = cos_row(h).dot(&cos_row(k));
        }
    }

    // Coefficient sup bounds. `A_h = √(‖a_h‖² + ‖b_h‖²)` bounds
    // `‖a_h cos φ − b_h sin φ‖` for every phase by Cauchy–Schwarz.
    let mut f_bar = 0.0_f64;
    let mut c_bar = 0.0_f64;
    let mut x_bar = 0.0_f64;
    for h in 1..=harmonics {
        let idx = (h - 1) * harmonics + (h - 1);
        let a_h = (g_aa[idx] + g_bb[idx]).max(0.0).sqrt();
        let hf = h as f64;
        f_bar += hf * a_h;
        c_bar += std::f64::consts::TAU * hf * hf * a_h;
        // `|U_{h−1}′| ≤ (h−1)h(h+1)/3` on `[−1, 1]`, attained at the endpoints.
        x_bar += (hf - 1.0) * hf * (hf + 1.0) / 3.0 * a_h;
    }

    let center_nodes = EMBEDDEDNESS_CENTER_NODES;
    let separation_nodes = EMBEDDEDNESS_SEPARATION_NODES;
    let half_c = 0.5 / center_nodes as f64;
    // `x` runs over `[−1, 1]` with both endpoints as nodes, so the spacing is
    // `2/(N−1)` and the worst-case distance to a node is half of it.
    let half_x = if separation_nodes > 1 {
        1.0 / (separation_nodes - 1) as f64
    } else {
        1.0
    };
    let grid_correction = half_c * 2.0 * f_bar * c_bar + half_x * 2.0 * f_bar * x_bar;

    // Grid minimum of `G(c, x) = Σ_{h,k} U_{h−1}(x) U_{k−1}(x) ⟨v_h(c), v_k(c)⟩`
    // with `v_h(c) = a_h cos(2πhc) − b_h sin(2πhc)`.
    let mut cheb = vec![0.0_f64; harmonics];
    let mut quad = vec![0.0_f64; harmonics * harmonics];
    let mut cos_c = vec![0.0_f64; harmonics];
    let mut sin_c = vec![0.0_f64; harmonics];
    let mut grid_min = f64::INFINITY;
    for ci in 0..center_nodes {
        let c = ci as f64 / center_nodes as f64;
        for h in 1..=harmonics {
            let angle = std::f64::consts::TAU * h as f64 * c;
            cos_c[h - 1] = angle.cos();
            sin_c[h - 1] = angle.sin();
        }
        for h in 1..=harmonics {
            for k in 1..=harmonics {
                let idx = (h - 1) * harmonics + (k - 1);
                let idx_t = (k - 1) * harmonics + (h - 1);
                quad[idx] = cos_c[h - 1] * cos_c[k - 1] * g_aa[idx]
                    - cos_c[h - 1] * sin_c[k - 1] * g_ab[idx]
                    - sin_c[h - 1] * cos_c[k - 1] * g_ab[idx_t]
                    + sin_c[h - 1] * sin_c[k - 1] * g_bb[idx];
            }
        }
        for xi in 0..separation_nodes {
            let x = if separation_nodes > 1 {
                -1.0 + 2.0 * xi as f64 / (separation_nodes - 1) as f64
            } else {
                0.0
            };
            // Chebyshev U by its three-term recurrence: `U₀ = 1`, `U₁ = 2x`.
            for h in 1..=harmonics {
                cheb[h - 1] = match h {
                    1 => 1.0,
                    2 => 2.0 * x,
                    _ => 2.0 * x * cheb[h - 2] - cheb[h - 3],
                };
            }
            let mut value = 0.0_f64;
            for h in 0..harmonics {
                for k in 0..harmonics {
                    value += cheb[h] * cheb[k] * quad[h * harmonics + k];
                }
            }
            grid_min = grid_min.min(value);
        }
    }

    let certified_min = grid_min - grid_correction;
    let scale = f_bar * f_bar;
    let relative_margin = if scale > 0.0 {
        certified_min / scale
    } else {
        0.0
    };
    Ok(AtomEmbeddednessCertificate {
        harmonics,
        grid_min,
        grid_correction,
        certified_min,
        scale,
        relative_margin,
        embedded: certified_min > 0.0,
        center_nodes,
        separation_nodes,
    })
}

/// Build the embeddedness certificate for one fitted atom of `term`, or `None`
/// when the atom is not a `d = 1` periodic (trig-polynomial) decoder — the only
/// family whose separation function this module's algebra covers.
pub(crate) fn atom_decoder_embeddedness(
    term: &SaeManifoldTerm,
    atom_idx: usize,
) -> Result<Option<AtomEmbeddednessCertificate>, String> {
    let Some(atom) = term.atoms.get(atom_idx) else {
        return Err(format!(
            "atom_decoder_embeddedness: atom {atom_idx} is not in the term"
        ));
    };
    periodic_atom_embeddedness(atom)
}

/// Embeddedness certificate of one atom's decoded image.
///
/// The algebra reads the decoder in the
/// [`crate::basis::PeriodicHarmonicEvaluator`] row layout, and the only decoder
/// the atom holds in that layout is [`SaeManifoldAtom::full_width_decoder`].
/// After a #1117 rank reduction the stored
/// [`SaeManifoldAtom::decoder_coefficients`] is `B̃ = QᵀB` (`r × p`) in the
/// frozen eigenvector frame `Q`, whose rows are mixtures of harmonics, not
/// harmonics: reading it directly certifies a different curve when `r` is odd
/// and silently drops the certificate when `r` is even (#3514). The full-width
/// `Q·B̃` decodes identically on the standard `[1, sin, cos, …]` inner basis, so
/// it is exactly the curve the atom draws.
pub(crate) fn periodic_atom_embeddedness(
    atom: &SaeManifoldAtom,
) -> Result<Option<AtomEmbeddednessCertificate>, String> {
    if atom.latent_dim() != 1 || !matches!(atom.basis_kind(), SaeAtomBasisKind::Periodic) {
        return Ok(None);
    }
    let decoder = atom.full_width_decoder();
    if decoder.nrows() % 2 == 0 {
        // A periodic-tagged atom whose INNER basis is not `2H+1` columns cannot
        // be read in the harmonic layout; report "no certificate" rather than
        // guessing.
        return Ok(None);
    }
    certify_periodic_decoder_embeddedness(decoder.view()).map(Some)
}
