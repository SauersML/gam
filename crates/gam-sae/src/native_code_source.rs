//! Conditional active-code sources for the native manifold-SAE description
//! length (#2933 F10, F11, F12).
//!
//! A token is reconstructed as `f_i = μ + Σ_k a_ik γ_k(t_ik)` with
//! `γ_k(t) = Φ_k(t) B_k`. The native code transmits, for every atom `k` in a
//! row's support, that atom's chart coordinate and the free information in its
//! gate. This module builds one Gaussian rate–distortion source per atom for
//! that transmitted coordinate, and the gate-amplitude sources of
//! [`native_gate_amplitude_code`], all measured in the output metric in which
//! the fit's explained variance is defined.
//!
//! # Only transmitted coordinates (F12)
//!
//! A row on which atom `k` does not fire carries no coordinate for `k`: its
//! stored value never reaches a decoded product and is never sent. The source's
//! moments are therefore taken over exactly the firing rows `R_k`, and its
//! expected per-token cost is weighted by the firing probability
//! `p_k = |R_k| / N`. The covariance is the unbiased one, so it needs
//! `|R_k| ≥ d_k + 1` firings for a full-rank estimate. An atom that fires on
//! `1..=d_k` rows is refused: too few firings to estimate a covariance is not
//! evidence that the source is deterministic, and pricing it at a zero spectrum
//! would make it free. A covariance that is singular with enough firings is a
//! measured rank and is kept.
//!
//! # Output metric (F11)
//!
//! Quantize atom `k`'s code with a subtractively dithered quantizer. Its error
//! `n_k` is independent of the source, zero mean, with covariance `E_k`, and the
//! errors of different atoms are independent. To first order the decoded error
//! of row `i` is `Σ_{k∈S_i} a_ik J_k(t_ik) n_k`, with `J_k = B_kᵀ ∂Φ_k/∂t` pushed
//! forward to the code chart. In the output metric `M`,
//!
//! ```text
//!   E‖δf_i‖²_M = Σ_{k∈S_i} tr(G_ik E_k),   G_ik = a_ik² J_kᵀ M J_k .
//! ```
//!
//! The cross-atom terms `E[n_kᵀ a_ik a_il J_kᵀ M J_l n_l]` vanish even where
//! decoders overlap, because the errors are independent and zero mean. Averaged
//! over tokens the distortion is `Σ_k p_k tr(Ḡ_k E_k)`, where `Ḡ_k` is the mean
//! pullback metric over `R_k`. The substitution `w = Ḡ_k^{1/2} z` makes this
//! Euclidean distortion of a source with covariance `Ḡ_k^{1/2} Σ_k Ḡ_k^{1/2}`.
//! Its eigenvalues are [`ActiveCodeSource::output_spectrum`], in squared output
//! units, and a rescaling of the coordinates that the decoder compensates
//! leaves them unchanged. The remainder of the linearization is the decoder's
//! second-order term. Its cross term with the first-order error is an odd moment
//! of a symmetric error and vanishes, so the distortion error is
//! `E‖½ a_ik ∂²γ_k[n_k, n_k]‖²_M = O(tr(E_k)²)`.
//!
//! The Gaussian rate of a covariance upper-bounds the squared-error
//! rate–distortion function of every source with that covariance (maximum
//! entropy). This is a declared surrogate for the linearized source, not a
//! certificate for the nonlinear one.
//!
//! # Gate amplitudes (F10)
//!
//! The support fixes which gates are nonzero, not their values: with a constant
//! chart and a constant curve `γ = 3`, the output `3·a_i` still varies with the
//! gate, and neither the support code nor the coordinate code carries it. The
//! receiver already knows the support, so the amplitude information is the
//! spread of a gate over the rows where it fires. The gate family decides how
//! much of it is free:
//!
//! * [`NativeGateModel::Independent`]: atom `k` transmits `a_ik` on `R_k`.
//!   Quantize it with an independent dithered error of variance `e_k`. The decoded
//!   error is `e_k·‖γ_k(t_ik)‖²_M`, and its cross terms with the coordinate errors
//!   and other atoms vanish as above, so the per-token distortion is
//!   `p_k ḡ_k e_k` with `ḡ_k = mean_{i∈R_k} ‖γ_k(t_ik)‖²_M`. The source is the
//!   scalar `ḡ_k v_k` with weight `p_k`, where `v_k` is the unbiased variance of
//!   the gate over `R_k` (two firings are needed). Rescaling a gate against its
//!   decoder leaves `ḡ_k v_k` unchanged, and a gate that is constant where it
//!   fires is free.
//! * [`NativeGateModel::Simplex`]: a row's transmitted gates sum to one, so a
//!   row with support `S` carries `|S| − 1` free amplitudes. Its innovation
//!   `e_ik = a_ik − m_k` (`m_k` the mean over `R_k`) is projected by
//!   `P_S = I − 1_S 1_Sᵀ/|S|`. The discarded component is known to the receiver
//!   from the constraint, which names no reference atom. The source covariance is
//!   `Σ = Σ_i P_S e_i e_iᵀ P_S / (N − 1)`. The sensitivity is
//!   `Ḡ_jk = (1/N) Σ_i 1[j,k ∈ S_i] γ_j(t_ij)ᵀ M γ_k(t_ik)`, because the projected
//!   error couples atoms. The source is `eig(Ḡ^{1/2} Σ Ḡ^{1/2})` with weight one.
//!   A row whose gates do not sum to one is reconstructed on the simplex, and that
//!   departure `(Σ_S a_ik − 1)/|S|` per gate is charged exactly as decoded
//!   distortion.
//! * [`NativeGateModel::UnitSupport`]: a transmitted gate is one, so nothing is
//!   sent. A transmitted gate other than one is charged exactly as the decoded
//!   distortion of `(a_ik − 1)·γ_k(t_ik)`.
//!
//! # Code coordinates independent of the representation
//!
//! * A Euclidean axis is coded as stored.
//! * A circle axis of period `P` is coded at the representative in
//!   `[0, P)`, unwrapped at the cut that minimizes its sample variance over the
//!   firing rows. Between consecutive sorted representatives the unwrapped set
//!   is the same, so there are exactly `|R_k|` candidate cuts and the minimum is
//!   exact. Shifting the chart moves the points and the minimizing cut together,
//!   so the covariance does not change, and a stored coordinate off by whole
//!   periods is the same point. An exact tie between two cuts is resolved
//!   toward the first candidate in sorted order.
//! * The ambient sphere `S²` stores three numbers for two intrinsic dimensions.
//!   It is coded by the Riemannian log map at the extrinsic mean direction of the
//!   firing rows, in an orthonormal tangent basis. The output Jacobian is
//!   composed with the differential of the exponential map. Rotating the
//!   representation rotates the tangent basis, which leaves the spectrum
//!   unchanged.
//! * A quotient cover first selects a deck representative. The ambient `RP²`
//!   (`u ~ −u`) takes the hemisphere of the principal axis of `Σ u uᵀ`. The Klein
//!   bottle (`(θ, φ) ~ (θ + ½, −φ)`) and the Möbius band (`(s, w) ~ (s + 1, −w)`)
//!   code the base circle (period `½` and `1`) at its minimum-variance cut and
//!   flip the fibre coordinate on every base half-turn. Every basis column is
//!   deck invariant, so the basis is evaluated at the selected representative.
//!
//! Finite-anchor and caller-precomputed bases carry no analytic coordinate jet,
//! and the charted `RP²` has a pole in its `(lat, lon)` chart. None of these has
//! such a representation here, and each is refused.

use ndarray::{Array2, Array3, ArrayView1, ArrayView2, Axis};

use crate::atom_codes::SparseAtomCodes;
use crate::gpu_kernels::sae_encode_resident::jacobi_eigh;
use crate::manifold::{SaeAtomGeometryPlan, SaeBasisResolution};

/// One atom's conditional active-code source in the output metric.
#[derive(Clone, Debug, PartialEq)]
pub struct ActiveCodeSource {
    /// Rows on which the atom is transmitted, `|R_k|`.
    pub firing_rows: usize,
    /// `|R_k| / N`: the per-token weight of this source's distortion and rate.
    pub firing_probability: f64,
    /// Intrinsic code dimension `d_k` (two for the ambient sphere).
    pub code_dim: usize,
    /// Unbiased covariance `Σ_k` of the intrinsic code over the firing rows.
    pub code_covariance: Array2<f64>,
    /// Mean output pullback metric `Ḡ_k = mean_{i∈R_k} a_ik² J_kᵀ M J_k`.
    pub mean_pullback_metric: Array2<f64>,
    /// Eigenvalues of `Ḡ_k^{1/2} Σ_k Ḡ_k^{1/2}`, descending, in squared output
    /// units. All zero for an atom that never fires.
    pub output_spectrum: Vec<f64>,
}

/// Build every atom's [`ActiveCodeSource`] from the transmitted support `codes`
/// (the gate on a firing row is read from `codes`), the persisted geometry
/// plans, the `M_k × p` decoders and the `N × latent_dim` coordinate blocks.
///
/// `tier0_scale` is the per-channel standardization the fit's explained variance
/// was measured under. The output metric is `M = diag(σ⁻²)` in the decoder's
/// physical frame, or the identity when the fit did not standardize.
pub fn native_active_code_sources(
    codes: &SparseAtomCodes,
    geometry_plans: &[SaeAtomGeometryPlan],
    decoder_blocks: &[ArrayView2<'_, f64>],
    coords: &[ArrayView2<'_, f64>],
    tier0_scale: Option<ArrayView1<'_, f64>>,
) -> Result<Vec<ActiveCodeSource>, String> {
    let n_obs = codes.n_obs();
    let k_atoms = codes.k_atoms();
    if geometry_plans.len() != k_atoms || decoder_blocks.len() != k_atoms || coords.len() != k_atoms
    {
        return Err(format!(
            "native active-code sources: a support over {k_atoms} atoms needs as many geometry \
             plans, decoders and coordinate blocks; got {}, {} and {}",
            geometry_plans.len(),
            decoder_blocks.len(),
            coords.len()
        ));
    }
    if k_atoms == 0 {
        return Ok(Vec::new());
    }
    if n_obs == 0 {
        return Err("native active-code sources: the support has no rows".to_string());
    }
    let p_out = decoder_blocks[0].ncols();
    if p_out == 0 {
        return Err("native active-code sources: decoders have no output channels".to_string());
    }
    let metric = output_metric_weights(tier0_scale, p_out)?;
    let mut sources = Vec::with_capacity(k_atoms);
    for atom in 0..k_atoms {
        let plan = &geometry_plans[atom];
        let decoder = decoder_blocks[atom];
        let width = plan.basis_size()?;
        if decoder.dim() != (width, p_out) {
            return Err(format!(
                "native active-code sources: atom {atom} decoder shape {:?} must equal the \
                 plan-derived ({width}, {p_out})",
                decoder.dim()
            ));
        }
        if coords[atom].dim() != (n_obs, plan.latent_dim()) {
            return Err(format!(
                "native active-code sources: atom {atom} coordinates {:?} must be ({n_obs}, {})",
                coords[atom].dim(),
                plan.latent_dim()
            ));
        }
        let mut rows = Vec::new();
        let mut amplitudes = Vec::new();
        for row in 0..n_obs {
            let code = codes.row(row);
            if code.active_mask.get(atom) {
                let amplitude = code.weights[atom];
                if !amplitude.is_finite() {
                    return Err(format!(
                        "native active-code sources: atom {atom} has a non-finite gate on \
                         firing row {row}"
                    ));
                }
                rows.push(row);
                amplitudes.push(amplitude);
            }
        }
        let firing = FiringRows {
            atom,
            rows: &rows,
            amplitudes: &amplitudes,
            n_obs,
        };
        sources.push(atom_source(&firing, plan, decoder, coords[atom], &metric)?);
    }
    Ok(sources)
}

/// The transmitted rows of one atom and the gates they carry.
struct FiringRows<'a> {
    atom: usize,
    rows: &'a [usize],
    amplitudes: &'a [f64],
    n_obs: usize,
}

pub(crate) fn output_metric_weights(
    tier0_scale: Option<ArrayView1<'_, f64>>,
    p_out: usize,
) -> Result<Vec<f64>, String> {
    let Some(scale) = tier0_scale else {
        return Ok(vec![1.0; p_out]);
    };
    if scale.len() != p_out {
        return Err(format!(
            "native active-code sources: tier0_scale has {} channels, decoders have {p_out}",
            scale.len()
        ));
    }
    scale
        .iter()
        .map(|&sigma| {
            if sigma.is_finite() && sigma > 0.0 {
                Ok((sigma * sigma).recip())
            } else {
                Err(format!(
                    "native active-code sources: tier0_scale entries must be finite and \
                     positive; got {sigma}"
                ))
            }
        })
        .collect()
}

fn atom_source(
    firing: &FiringRows<'_>,
    plan: &SaeAtomGeometryPlan,
    decoder: ArrayView2<'_, f64>,
    coords: ArrayView2<'_, f64>,
    metric: &[f64],
) -> Result<ActiveCodeSource, String> {
    let atom = firing.atom;
    let code_dim = plan.intrinsic_dim();
    let n = firing.rows.len();
    if n == 0 {
        return Ok(ActiveCodeSource {
            firing_rows: 0,
            firing_probability: 0.0,
            code_dim,
            code_covariance: Array2::zeros((code_dim, code_dim)),
            mean_pullback_metric: Array2::zeros((code_dim, code_dim)),
            output_spectrum: vec![0.0; code_dim],
        });
    }
    if n <= code_dim {
        return Err(format!(
            "native active-code sources: atom {atom} is transmitted on {n} row(s), but its \
             {code_dim}-dimensional code covariance needs at least {} firings for a full-rank \
             unbiased estimate; its Gaussian rate is unavailable, not zero",
            code_dim + 1
        ));
    }
    let firing_coords = coords.select(Axis(0), firing.rows);
    if firing_coords.iter().any(|value| !value.is_finite()) {
        return Err(format!(
            "native active-code sources: atom {atom} has a non-finite coordinate on a firing row"
        ));
    }
    let chart = intrinsic_code_chart(atom, plan, firing_coords.view())?;
    let (_, jet) = plan
        .build_evaluator()?
        .evaluate(chart.evaluation_coords.view())?;
    let latent_dim = plan.latent_dim();
    let width = decoder.nrows();
    if jet.dim() != (n, width, latent_dim) {
        return Err(format!(
            "native active-code sources: atom {atom} basis jet {:?} must be ({n}, {width}, \
             {latent_dim})",
            jet.dim()
        ));
    }

    // Basis-space output metric `K = B M Bᵀ`, so a row's ambient pullback is
    // `∂Φᵀ K ∂Φ` without forming the `p × latent_dim` Jacobian.
    let mut basis_metric = Array2::<f64>::zeros((width, width));
    for a in 0..width {
        for b in a..width {
            let mut acc = 0.0;
            for (channel, &weight) in metric.iter().enumerate() {
                acc += decoder[[a, channel]] * weight * decoder[[b, channel]];
            }
            basis_metric[[a, b]] = acc;
            basis_metric[[b, a]] = acc;
        }
    }
    let mut metric_sum = Array2::<f64>::zeros((code_dim, code_dim));
    let mut k_jet = vec![0.0_f64; width * latent_dim];
    let mut ambient = vec![0.0_f64; latent_dim * latent_dim];
    for r in 0..n {
        for axis in 0..latent_dim {
            for m in 0..width {
                let mut acc = 0.0;
                for l in 0..width {
                    acc += basis_metric[[m, l]] * jet[[r, l, axis]];
                }
                k_jet[m * latent_dim + axis] = acc;
            }
        }
        for a in 0..latent_dim {
            for b in 0..latent_dim {
                let mut acc = 0.0;
                for m in 0..width {
                    acc += jet[[r, m, a]] * k_jet[m * latent_dim + b];
                }
                ambient[a * latent_dim + b] = acc;
            }
        }
        let gate_squared = firing.amplitudes[r] * firing.amplitudes[r];
        for i in 0..code_dim {
            for j in 0..code_dim {
                let mut acc = 0.0;
                for a in 0..latent_dim {
                    let push_a = chart.pushforward[[r, a, i]];
                    for b in 0..latent_dim {
                        acc += push_a * ambient[a * latent_dim + b] * chart.pushforward[[r, b, j]];
                    }
                }
                metric_sum[[i, j]] += gate_squared * acc;
            }
        }
    }
    let inv_n = (n as f64).recip();
    let mean_pullback_metric =
        Array2::from_shape_fn((code_dim, code_dim), |(i, j)| {
            0.5 * (metric_sum[[i, j]] + metric_sum[[j, i]]) * inv_n
        });
    let code_covariance = unbiased_covariance(chart.code.view());
    let output_spectrum =
        output_metric_spectrum(&format!("atom {atom}"), &code_covariance, &mean_pullback_metric)?;
    Ok(ActiveCodeSource {
        firing_rows: n,
        firing_probability: n as f64 / firing.n_obs as f64,
        code_dim,
        code_covariance,
        mean_pullback_metric,
        output_spectrum,
    })
}

fn unbiased_covariance(code: ArrayView2<'_, f64>) -> Array2<f64> {
    let (n, d) = code.dim();
    let mut mean = vec![0.0_f64; d];
    for row in code.rows() {
        for (slot, value) in mean.iter_mut().zip(row.iter()) {
            *slot += value;
        }
    }
    for slot in &mut mean {
        *slot /= n as f64;
    }
    let mut covariance = Array2::<f64>::zeros((d, d));
    for row in code.rows() {
        for a in 0..d {
            let va = row[a] - mean[a];
            for b in a..d {
                covariance[[a, b]] += va * (row[b] - mean[b]);
            }
        }
    }
    let inv = ((n - 1) as f64).recip();
    for a in 0..d {
        for b in a..d {
            let value = covariance[[a, b]] * inv;
            covariance[[a, b]] = value;
            covariance[[b, a]] = value;
        }
    }
    covariance
}

fn symmetric_eigen(
    source: &str,
    what: &str,
    matrix: &Array2<f64>,
) -> Result<(Vec<f64>, Vec<f64>), String> {
    let d = matrix.nrows();
    let flat: Vec<f64> = matrix.iter().copied().collect();
    let mut values = vec![0.0_f64; d];
    let mut vectors = vec![0.0_f64; d * d];
    if !jacobi_eigh(&flat, d, &mut values, &mut vectors) {
        return Err(format!(
            "native active-code sources: {source} {what} did not diagonalize within the \
             Jacobi sweep budget"
        ));
    }
    Ok((values, vectors))
}

/// Eigenvalues of `Ḡ^{1/2} Σ Ḡ^{1/2}`, computed as those of the similar
/// `Σ^{1/2} Ḡ Σ^{1/2}` so a measured-rank-deficient `Σ` needs no inverse.
pub(crate) fn output_metric_spectrum(
    source: &str,
    covariance: &Array2<f64>,
    metric: &Array2<f64>,
) -> Result<Vec<f64>, String> {
    let d = covariance.nrows();
    let (values, vectors) = symmetric_eigen(source, "code covariance", covariance)?;
    let root = Array2::from_shape_fn((d, d), |(r, c)| {
        (0..d)
            .map(|i| vectors[i * d + r] * values[i].max(0.0).sqrt() * vectors[i * d + c])
            .sum::<f64>()
    });
    let whitened = root.dot(metric).dot(&root);
    let symmetric =
        Array2::from_shape_fn((d, d), |(r, c)| 0.5 * (whitened[[r, c]] + whitened[[c, r]]));
    let (mut spectrum, _) = symmetric_eigen(source, "output-metric code covariance", &symmetric)?;
    for value in &mut spectrum {
        *value = value.max(0.0);
    }
    spectrum.sort_by(|left, right| right.total_cmp(left));
    Ok(spectrum)
}

/// Which continuous gate information a receiver still needs once it knows the
/// support (#2933 F10). The gate family, not the atom count, decides how many
/// free amplitudes a row carries.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NativeGateModel {
    /// Row-wise softmax responsibilities. A row's transmitted gates sum to one,
    /// so a row with support `S` carries `|S| − 1` free amplitudes.
    Simplex,
    /// Independent sigmoid gates (ordered Beta–Bernoulli, threshold gate): one
    /// free amplitude per transmitted gate.
    Independent,
    /// Hard top-k gates: a transmitted gate is one, so the support determines
    /// every amplitude.
    UnitSupport,
}

impl NativeGateModel {
    /// Resolve a public assignment token through the shared strict schema.
    pub fn from_assignment_tag(tag: &str) -> Result<Self, String> {
        match crate::atom_schema::canonical_assignment_kind(tag)? {
            "softmax" => Ok(Self::Simplex),
            "ordered_beta_bernoulli" | "threshold_gate" => Ok(Self::Independent),
            "topk" => Ok(Self::UnitSupport),
            canonical => Err(format!(
                "native gate amplitudes: the assignment schema returned token {canonical:?}, \
                 which has no gate model"
            )),
        }
    }
}

/// The gate-amplitude sources of the native message (#2933 F10).
#[derive(Clone, Debug, PartialEq)]
pub struct GateAmplitudeCode {
    /// Weighted output-metric spectra `(w, μ)`, in the metric of
    /// [`ActiveCodeSource::output_spectrum`]: `(p_k, [ḡ_k v_k])` per atom for
    /// independent gates, one `(1, eig(Ḡ^{1/2} Σ Ḡ^{1/2}))` for the simplex, and
    /// none for unit support.
    pub components: Vec<(f64, Vec<f64>)>,
    /// Per-token output distortion of the gates the receiver reconstructs from the
    /// gate model in place of the fitted gates. Zero for independent gates.
    pub representation_distortion: f64,
}

impl GateAmplitudeCode {
    /// Decoded output variance of the transmitted amplitudes, `Σ_c w_c Σ_u μ_cu`.
    pub fn decoded_variance(&self) -> f64 {
        self.components
            .iter()
            .map(|(weight, spectrum)| weight * spectrum.iter().sum::<f64>())
            .sum()
    }
}

/// Build the [`GateAmplitudeCode`] of the transmitted support `codes` under
/// `gate_model` (see the module section on gate amplitudes), from the persisted
/// geometry plans, the `M_k × p` decoders and the `N × latent_dim` coordinate
/// blocks, in the output metric of [`native_active_code_sources`].
pub fn native_gate_amplitude_code(
    codes: &SparseAtomCodes,
    gate_model: NativeGateModel,
    geometry_plans: &[SaeAtomGeometryPlan],
    decoder_blocks: &[ArrayView2<'_, f64>],
    coords: &[ArrayView2<'_, f64>],
    tier0_scale: Option<ArrayView1<'_, f64>>,
) -> Result<GateAmplitudeCode, String> {
    let n_obs = codes.n_obs();
    let k_atoms = codes.k_atoms();
    if geometry_plans.len() != k_atoms || decoder_blocks.len() != k_atoms || coords.len() != k_atoms
    {
        return Err(format!(
            "native gate amplitudes: a support over {k_atoms} atoms needs as many geometry \
             plans, decoders and coordinate blocks; got {}, {} and {}",
            geometry_plans.len(),
            decoder_blocks.len(),
            coords.len()
        ));
    }
    if k_atoms == 0 {
        return Ok(GateAmplitudeCode {
            components: Vec::new(),
            representation_distortion: 0.0,
        });
    }
    if n_obs == 0 {
        return Err("native gate amplitudes: the support has no rows".to_string());
    }
    let p_out = decoder_blocks[0].ncols();
    if p_out == 0 {
        return Err("native gate amplitudes: decoders have no output channels".to_string());
    }
    let metric = output_metric_weights(tier0_scale, p_out)?;
    let mut firing: Vec<(Vec<usize>, Vec<f64>)> = vec![(Vec::new(), Vec::new()); k_atoms];
    for row in 0..n_obs {
        let code = codes.row(row);
        for (atom, (rows, gates)) in firing.iter_mut().enumerate() {
            if code.active_mask.get(atom) {
                let gate = code.weights[atom];
                if !gate.is_finite() {
                    return Err(format!(
                        "native gate amplitudes: atom {atom} has a non-finite gate on firing row \
                         {row}"
                    ));
                }
                rows.push(row);
                gates.push(gate);
            }
        }
    }
    let basis_of = |atom: usize| {
        firing_basis(
            atom,
            &geometry_plans[atom],
            decoder_blocks[atom],
            coords[atom],
            &firing[atom].0,
            n_obs,
            p_out,
        )
    };
    let n = n_obs as f64;

    match gate_model {
        NativeGateModel::Independent => {
            let mut components = Vec::with_capacity(k_atoms);
            for (atom, (rows, gates)) in firing.iter().enumerate() {
                let firings = rows.len();
                if firings == 0 {
                    components.push((0.0, vec![0.0]));
                    continue;
                }
                if firings == 1 {
                    return Err(format!(
                        "native gate amplitudes: atom {atom} is transmitted on one row, but its \
                         amplitude variance needs at least two firings for an unbiased estimate; \
                         its Gaussian rate is unavailable, not zero"
                    ));
                }
                let count = firings as f64;
                let mean = gates.iter().sum::<f64>() / count;
                let variance = gates
                    .iter()
                    .map(|gate| (gate - mean) * (gate - mean))
                    .sum::<f64>()
                    / (count - 1.0);
                let curves = basis_of(atom)?.dot(&decoder_blocks[atom]);
                let sensitivity = squared_output_norms(&curves, &metric) / count;
                components.push((count / n, vec![sensitivity * variance]));
            }
            Ok(GateAmplitudeCode {
                components,
                representation_distortion: 0.0,
            })
        }
        NativeGateModel::UnitSupport => {
            let mut departure = Array2::<f64>::zeros((n_obs, p_out));
            for (atom, (rows, gates)) in firing.iter().enumerate() {
                if gates.iter().all(|&gate| gate == 1.0) {
                    continue;
                }
                let curves = basis_of(atom)?.dot(&decoder_blocks[atom]);
                for (slot, (&row, &gate)) in rows.iter().zip(gates).enumerate() {
                    for channel in 0..p_out {
                        departure[[row, channel]] += (gate - 1.0) * curves[[slot, channel]];
                    }
                }
            }
            Ok(GateAmplitudeCode {
                components: Vec::new(),
                representation_distortion: squared_output_norms(&departure, &metric) / n,
            })
        }
        NativeGateModel::Simplex => {
            let means: Vec<f64> = firing
                .iter()
                .map(|(rows, gates)| {
                    if rows.is_empty() {
                        0.0
                    } else {
                        gates.iter().sum::<f64>() / rows.len() as f64
                    }
                })
                .collect();
            let mut support_size = vec![0_usize; n_obs];
            let mut row_sum = vec![0.0_f64; n_obs];
            let mut innovation_sum = vec![0.0_f64; n_obs];
            for (atom, (rows, gates)) in firing.iter().enumerate() {
                for (&row, &gate) in rows.iter().zip(gates) {
                    support_size[row] += 1;
                    row_sum[row] += gate;
                    innovation_sum[row] += gate - means[atom];
                }
            }
            let mut departure = Array2::<f64>::zeros((n_obs, p_out));
            let mut innovations = Array2::<f64>::zeros((n_obs, k_atoms));
            let mut masked_basis = Vec::with_capacity(k_atoms);
            for (atom, (rows, gates)) in firing.iter().enumerate() {
                let basis = basis_of(atom)?;
                let curves = basis.dot(&decoder_blocks[atom]);
                let mut masked = Array2::<f64>::zeros((n_obs, basis.ncols()));
                for (slot, (&row, &gate)) in rows.iter().zip(gates).enumerate() {
                    let size = support_size[row] as f64;
                    let drift = (row_sum[row] - 1.0) / size;
                    for channel in 0..p_out {
                        departure[[row, channel]] += drift * curves[[slot, channel]];
                    }
                    innovations[[row, atom]] = gate - means[atom] - innovation_sum[row] / size;
                    masked.row_mut(row).assign(&basis.row(slot));
                }
                masked_basis.push(masked);
            }
            let representation_distortion = squared_output_norms(&departure, &metric) / n;
            if support_size.iter().all(|&size| size < 2) {
                return Ok(GateAmplitudeCode {
                    components: Vec::new(),
                    representation_distortion,
                });
            }
            if n_obs < 2 {
                return Err(
                    "native gate amplitudes: one row cannot estimate the simplex amplitude \
                     covariance; its Gaussian rate is unavailable, not zero"
                        .to_string(),
                );
            }
            let covariance = innovations.t().dot(&innovations) / (n - 1.0);
            let weights = ndarray::Array1::from(metric);
            let mut sensitivity = Array2::<f64>::zeros((k_atoms, k_atoms));
            for j in 0..k_atoms {
                let weighted_decoder = &decoder_blocks[j] * &weights;
                for k in j..k_atoms {
                    let basis_cross = masked_basis[j].t().dot(&masked_basis[k]);
                    let decoder_cross = weighted_decoder.dot(&decoder_blocks[k].t());
                    let value = (&basis_cross * &decoder_cross).sum() / n;
                    sensitivity[[j, k]] = value;
                    sensitivity[[k, j]] = value;
                }
            }
            let spectrum =
                output_metric_spectrum("the simplex gate amplitudes", &covariance, &sensitivity)?;
            Ok(GateAmplitudeCode {
                components: vec![(1.0, spectrum)],
                representation_distortion,
            })
        }
    }
}

/// The basis `Φ_k(t_ik)` of one atom on its firing rows, `(|R_k|, M_k)`, evaluated
/// at the stored coordinates exactly as the persisted reconstruction evaluates it.
fn firing_basis(
    atom: usize,
    plan: &SaeAtomGeometryPlan,
    decoder: ArrayView2<'_, f64>,
    coords: ArrayView2<'_, f64>,
    rows: &[usize],
    n_obs: usize,
    p_out: usize,
) -> Result<Array2<f64>, String> {
    let width = plan.basis_size()?;
    if decoder.dim() != (width, p_out) {
        return Err(format!(
            "native gate amplitudes: atom {atom} decoder shape {:?} must equal the plan-derived \
             ({width}, {p_out})",
            decoder.dim()
        ));
    }
    if coords.dim() != (n_obs, plan.latent_dim()) {
        return Err(format!(
            "native gate amplitudes: atom {atom} coordinates {:?} must be ({n_obs}, {})",
            coords.dim(),
            plan.latent_dim()
        ));
    }
    if rows.is_empty() {
        return Ok(Array2::zeros((0, width)));
    }
    let firing_coords = coords.select(Axis(0), rows);
    if firing_coords.iter().any(|value| !value.is_finite()) {
        return Err(format!(
            "native gate amplitudes: atom {atom} has a non-finite coordinate on a firing row"
        ));
    }
    let (basis, _) = plan.build_evaluator()?.evaluate(firing_coords.view())?;
    if basis.dim() != (rows.len(), width) {
        return Err(format!(
            "native gate amplitudes: atom {atom} basis {:?} must be ({}, {width})",
            basis.dim(),
            rows.len()
        ));
    }
    Ok(basis)
}

/// `Σ_i ‖values_i‖²_M` over the rows of `values` for the diagonal metric `metric`.
fn squared_output_norms(values: &Array2<f64>, metric: &[f64]) -> f64 {
    values
        .rows()
        .into_iter()
        .map(|row| {
            row.iter()
                .zip(metric)
                .map(|(value, weight)| weight * value * value)
                .sum::<f64>()
        })
        .sum()
}

/// Intrinsic code coordinates of the firing rows and the point each row's basis
/// is evaluated at.
struct CodeChart {
    /// Intrinsic code, `(n, d)`.
    code: Array2<f64>,
    /// Deck-selected, unwrapped representative evaluated per row, `(n, latent_dim)`.
    evaluation_coords: Array2<f64>,
    /// `∂(evaluation coordinate)/∂(code)` per row, `(n, latent_dim, d)`.
    pushforward: Array3<f64>,
}

fn flat_chart(code: Array2<f64>) -> CodeChart {
    let (n, d) = code.dim();
    let pushforward = Array3::from_shape_fn((n, d, d), |(_, a, i)| if a == i { 1.0 } else { 0.0 });
    CodeChart {
        evaluation_coords: code.clone(),
        code,
        pushforward,
    }
}

fn intrinsic_code_chart(
    atom: usize,
    plan: &SaeAtomGeometryPlan,
    coords: ArrayView2<'_, f64>,
) -> Result<CodeChart, String> {
    match plan.resolution() {
        SaeBasisResolution::DuchonCoordinates { .. } | SaeBasisResolution::Polynomial { .. } => {
            Ok(flat_chart(coords.to_owned()))
        }
        SaeBasisResolution::PeriodicHarmonics { .. } | SaeBasisResolution::TorusHarmonics { .. } => {
            let mut code = coords.to_owned();
            for axis in 0..code.ncols() {
                unwrap_circle_axis(&mut code, axis, 1.0);
            }
            Ok(flat_chart(code))
        }
        SaeBasisResolution::CylinderHarmonics { .. } => {
            let mut code = coords.to_owned();
            unwrap_circle_axis(&mut code, 0, 1.0);
            Ok(flat_chart(code))
        }
        SaeBasisResolution::MobiusHarmonics { .. } => Ok(twisted_chart(coords, 1.0, None)),
        SaeBasisResolution::KleinBottleHarmonics { .. } => Ok(twisted_chart(coords, 0.5, Some(1.0))),
        SaeBasisResolution::AmbientSphereHarmonics { .. } => sphere_log_chart(atom, coords, false),
        SaeBasisResolution::AmbientProjectivePlaneHarmonics { .. } => {
            sphere_log_chart(atom, coords, true)
        }
        SaeBasisResolution::ProjectivePlaneHarmonics { .. } => Err(format!(
            "native active-code sources: atom {atom} is a charted RP² whose (lat, lon) chart has \
             a pole; its code has no pole-free representation here (re-express it on the \
             ambient cover)"
        )),
        SaeBasisResolution::FiniteAnchors { .. } => Err(format!(
            "native active-code sources: atom {atom} is a finite-anchor atom; its categorical \
             anchor is not a continuous code"
        )),
        SaeBasisResolution::Precomputed { .. } => Err(format!(
            "native active-code sources: atom {atom} has a caller-precomputed basis with no \
             analytic coordinate jet"
        )),
    }
}

/// Representative of `value` in `[0, period)` and the whole periods removed.
fn canonical_phase(value: f64, period: f64) -> (f64, i64) {
    let turns = (value / period).floor();
    let mut phase = value - turns * period;
    let mut turns = turns as i64;
    if phase >= period {
        phase -= period;
        turns += 1;
    }
    if phase < 0.0 {
        phase += period;
        turns -= 1;
    }
    (phase, turns)
}

/// Which representatives in `[0, period)` move up one period when the circle is
/// unwrapped at the cut minimizing their sample variance.
fn minimum_variance_wraps(canonical: &[f64], period: f64) -> Vec<bool> {
    let n = canonical.len();
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&left, &right| canonical[left].total_cmp(&canonical[right]));
    let count = n as f64;
    let total: f64 = canonical.iter().sum();
    let total_squares: f64 = canonical.iter().map(|value| value * value).sum();
    let variance = |sum: f64, squares: f64| squares / count - (sum / count).powi(2);
    let mut best_wrapped = 0usize;
    let mut best_variance = variance(total, total_squares);
    let mut prefix = 0.0_f64;
    for wrapped in 1..n {
        prefix += canonical[order[wrapped - 1]];
        let shifted = wrapped as f64;
        let sum = total + shifted * period;
        let squares = total_squares + 2.0 * period * prefix + shifted * period * period;
        let candidate = variance(sum, squares);
        if candidate < best_variance {
            best_wrapped = wrapped;
            best_variance = candidate;
        }
    }
    let mut wraps = vec![false; n];
    for &row in &order[..best_wrapped] {
        wraps[row] = true;
    }
    wraps
}

pub(crate) fn unwrap_circle_values(values: &mut [f64], period: f64) {
    let canonical: Vec<f64> = values
        .iter()
        .map(|&value| canonical_phase(value, period).0)
        .collect();
    let wraps = minimum_variance_wraps(&canonical, period);
    for ((slot, &phase), &wrap) in values.iter_mut().zip(&canonical).zip(&wraps) {
        *slot = if wrap { phase + period } else { phase };
    }
}

fn unwrap_circle_axis(code: &mut Array2<f64>, axis: usize, period: f64) {
    let mut values: Vec<f64> = code.column(axis).to_vec();
    unwrap_circle_values(&mut values, period);
    for (slot, value) in code.column_mut(axis).iter_mut().zip(values) {
        *slot = value;
    }
}

/// Code a twisted product whose deck map is `(b, f) ~ (b + base_period, −f)`:
/// the base circle at its minimum-variance cut, with the fibre flipped on every
/// base half-turn, then the fibre unwrapped when it is itself a circle.
fn twisted_chart(
    coords: ArrayView2<'_, f64>,
    base_period: f64,
    fibre_period: Option<f64>,
) -> CodeChart {
    let n = coords.nrows();
    let mut base = Vec::with_capacity(n);
    let mut fibre = Vec::with_capacity(n);
    for row in 0..n {
        let (phase, turns) = canonical_phase(coords[[row, 0]], base_period);
        let sign = if turns.rem_euclid(2) == 0 { 1.0 } else { -1.0 };
        base.push(phase);
        fibre.push(sign * coords[[row, 1]]);
    }
    let wraps = minimum_variance_wraps(&base, base_period);
    for row in 0..n {
        if wraps[row] {
            base[row] += base_period;
            fibre[row] = -fibre[row];
        }
    }
    if let Some(period) = fibre_period {
        unwrap_circle_values(&mut fibre, period);
    }
    flat_chart(Array2::from_shape_fn((n, 2), |(row, axis)| {
        if axis == 0 { base[row] } else { fibre[row] }
    }))
}

fn dot3(left: &[f64; 3], right: &[f64; 3]) -> f64 {
    left[0] * right[0] + left[1] * right[1] + left[2] * right[2]
}

fn sphere_log_chart(
    atom: usize,
    coords: ArrayView2<'_, f64>,
    antipodal_quotient: bool,
) -> Result<CodeChart, String> {
    let n = coords.nrows();
    let mut units = Vec::with_capacity(n);
    for row in 0..n {
        let point = [coords[[row, 0]], coords[[row, 1]], coords[[row, 2]]];
        let norm = dot3(&point, &point).sqrt();
        if !(norm.is_finite() && norm > 0.0) {
            return Err(format!(
                "native active-code sources: atom {atom} firing row has a zero or non-finite \
                 sphere coordinate"
            ));
        }
        units.push([point[0] / norm, point[1] / norm, point[2] / norm]);
    }
    if antipodal_quotient {
        let mut second_moment = [0.0_f64; 9];
        for unit in &units {
            for a in 0..3 {
                for b in 0..3 {
                    second_moment[a * 3 + b] += unit[a] * unit[b];
                }
            }
        }
        let mut values = [0.0_f64; 3];
        let mut vectors = [0.0_f64; 9];
        if !jacobi_eigh(&second_moment, 3, &mut values, &mut vectors) {
            return Err(format!(
                "native active-code sources: atom {atom} RP² second moment did not diagonalize \
                 within the Jacobi sweep budget"
            ));
        }
        let top = (0..3)
            .max_by(|&left, &right| values[left].total_cmp(&values[right]))
            .unwrap_or(0);
        let axis = [vectors[top * 3], vectors[top * 3 + 1], vectors[top * 3 + 2]];
        for unit in &mut units {
            if dot3(unit, &axis) < 0.0 {
                *unit = [-unit[0], -unit[1], -unit[2]];
            }
        }
    }
    let mut sum = [0.0_f64; 3];
    for unit in &units {
        for a in 0..3 {
            sum[a] += unit[a];
        }
    }
    let length = dot3(&sum, &sum).sqrt();
    if !(length > 0.0) {
        return Err(format!(
            "native active-code sources: atom {atom} firing directions sum to zero, so the \
             extrinsic mean direction the log map is taken at is undefined"
        ));
    }
    let mean = [sum[0] / length, sum[1] / length, sum[2] / length];
    let pivot = (0..3)
        .min_by(|&left, &right| mean[left].abs().total_cmp(&mean[right].abs()))
        .unwrap_or(0);
    let mut first = [0.0_f64; 3];
    first[pivot] = 1.0;
    let along_mean = first[pivot] * mean[pivot];
    for a in 0..3 {
        first[a] -= along_mean * mean[a];
    }
    let first_norm = dot3(&first, &first).sqrt();
    let first = [first[0] / first_norm, first[1] / first_norm, first[2] / first_norm];
    let second = [
        mean[1] * first[2] - mean[2] * first[1],
        mean[2] * first[0] - mean[0] * first[2],
        mean[0] * first[1] - mean[1] * first[0],
    ];
    let mut code = Array2::<f64>::zeros((n, 2));
    let mut evaluation_coords = Array2::<f64>::zeros((n, 3));
    let mut pushforward = Array3::<f64>::zeros((n, 3, 2));
    for (row, unit) in units.iter().enumerate() {
        let cosine = dot3(&mean, unit).clamp(-1.0, 1.0);
        let radial = [
            unit[0] - cosine * mean[0],
            unit[1] - cosine * mean[1],
            unit[2] - cosine * mean[2],
        ];
        let sine = dot3(&radial, &radial).sqrt();
        let angle = sine.atan2(cosine);
        let direction = if sine > 0.0 {
            [radial[0] / sine, radial[1] / sine, radial[2] / sine]
        } else if cosine > 0.0 {
            first
        } else {
            return Err(format!(
                "native active-code sources: atom {atom} firing row is antipodal to the extrinsic \
                 mean direction, where the log map has no chart"
            ));
        };
        code[[row, 0]] = angle * dot3(&direction, &first);
        code[[row, 1]] = angle * dot3(&direction, &second);
        let sinc = if angle > 0.0 { angle.sin() / angle } else { 1.0 };
        for (j, basis) in [first, second].iter().enumerate() {
            let along = dot3(basis, &direction);
            for a in 0..3 {
                let geodesic = -angle.sin() * mean[a] + angle.cos() * direction[a];
                pushforward[[row, a, j]] =
                    along * geodesic + sinc * (basis[a] - along * direction[a]);
            }
        }
        for a in 0..3 {
            evaluation_coords[[row, a]] = unit[a];
        }
    }
    Ok(CodeChart {
        code,
        evaluation_coords,
        pushforward,
    })
}

#[cfg(test)]
#[path = "native_code_source_tests.rs"]
mod native_code_source_tests;
