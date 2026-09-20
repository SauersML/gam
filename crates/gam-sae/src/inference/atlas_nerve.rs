//! Certificate-gated atlas nerves for block/chart dictionaries.
//!
//! Single chart atoms expose exact within-chart graph readouts, but they cannot
//! see cross-chart topology such as an atlas covering a sphere or torus. This
//! module builds the nerve over dictionary charts: a chart is a vertex, a pair
//! becomes an edge only when the supports co-activate and the chart-transfer
//! certificate is valid, and higher simplices are admitted from mutual
//! co-activation. The resulting filtered clique complex is read by exact GF(2)
//! homology through H2.  Every order of simplex is counted for the Euler
//! characteristic; stopping at triangles or tetrahedra changes topology when
//! five or more charts share an overlap.

use crate::inference::atlas_holonomy::{
    AtlasEulerCharacteristic, AtlasHolonomyCertificate, AtlasHolonomyEdgeId,
};
use crate::manifold::{AtlasOrientability, BettiSignature, GraphCompressionKind};
use crate::null_battery::ClaimNullCalibration;
use std::cmp::Reverse;
use std::collections::{BTreeMap, BTreeSet, BinaryHeap, HashMap};

/// Which side of the chart-count sampling diagnostic the diagram sits on.
///
/// This says only whether the observed support count is at least the number of
/// charts.  It is useful for spotting an obviously under-sampled atlas, but is
/// neither a contractibility test nor a Nerve-theorem premise, and nothing here
/// establishes the good-cover precondition.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AtlasCoveringSide {
    BelowCoveringNumber,
    AtOrAboveCoveringNumber,
}

impl AtlasCoveringSide {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::BelowCoveringNumber => "below_covering_number",
            Self::AtOrAboveCoveringNumber => "at_or_above_covering_number",
        }
    }
}

/// One dictionary chart in the atlas nerve.
#[derive(Clone, Debug)]
pub struct AtlasChart {
    pub chart_idx: usize,
    row_count: usize,
    support_rows: Vec<usize>,
    support_weights: Vec<f64>,
    pub support_mass: f64,
    pub support_ess: f64,
}

impl AtlasChart {
    /// Construct a chart from its strictly-positive sparse row support.
    ///
    /// `support_rows` must be strictly increasing and every row must be below
    /// `row_count`. Storing only positive weights makes atlas memory scale with
    /// route nonzeros rather than `row_count × number_of_charts`.
    #[must_use = "atlas chart construction errors must be handled"]
    pub fn from_sparse_weights(
        chart_idx: usize,
        row_count: usize,
        support_rows: Vec<usize>,
        support_weights: Vec<f64>,
    ) -> Result<Self, String> {
        if support_rows.len() != support_weights.len() {
            return Err(format!(
                "atlas chart {chart_idx} has {} support rows but {} weights",
                support_rows.len(),
                support_weights.len()
            ));
        }
        let mut previous = None;
        let mut support_mass = 0.0_f64;
        let mut fisher_mass = 0.0_f64;
        for (position, (&row, &weight)) in
            support_rows.iter().zip(support_weights.iter()).enumerate()
        {
            if row >= row_count {
                return Err(format!(
                    "atlas chart {chart_idx} support row {row} at position {position} is outside 0..{row_count}"
                ));
            }
            if previous.is_some_and(|prior| prior >= row) {
                return Err(format!(
                    "atlas chart {chart_idx} support rows must be strictly increasing"
                ));
            }
            if !(weight.is_finite() && weight > 0.0) {
                return Err(format!(
                    "atlas chart {chart_idx} sparse weight at row {row} must be finite and positive, got {weight}"
                ));
            }
            previous = Some(row);
            support_mass += weight;
            fisher_mass += weight * weight;
        }
        if !(support_mass.is_finite() && fisher_mass.is_finite()) {
            return Err(format!(
                "atlas chart {chart_idx} support moments overflowed"
            ));
        }
        let support_ess = if fisher_mass > 0.0 {
            support_mass * support_mass / fisher_mass
        } else {
            0.0
        };
        Ok(Self {
            chart_idx,
            row_count,
            support_rows,
            support_weights,
            support_mass,
            support_ess,
        })
    }

    pub fn row_count(&self) -> usize {
        self.row_count
    }

    pub fn support_rows(&self) -> &[usize] {
        &self.support_rows
    }
}

/// Orientation class of a planar conformal transfer `x_b = c·Q·x_a`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PlanarTransferOrientation {
    /// `Q` is a rotation: the transfer lies in `span{I, G}` and commutes with
    /// the SO(2) generator `G = [[0, −1], [1, 0]]`.
    Preserving,
    /// `Q` is a reflection: the transfer lies in `span{F, H}`
    /// (`F = diag(1, −1)`, `H = [[0, 1], [1, 0]]`) and anti-commutes with `G`.
    Reversing,
}

impl PlanarTransferOrientation {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Preserving => "preserving",
            Self::Reversing => "reversing",
        }
    }
}

/// Calibrated test that the empirical transfer between two planar charts is a
/// scaled rotation or reflection.
///
/// The co-firing rows give paired codes `x_i` (chart a) and `y_i` (chart b).
/// The least-squares operator `A = (XᵀX)⁻¹XᵀY` (so `y_i ≈ Aᵀx_i`) decomposes
/// uniquely and Frobenius-orthogonally as
/// `A = a·I + b·G + p·F + q·H`, with
/// `(a, b) = ((A₀₀ + A₁₁)/2, (A₁₀ − A₀₁)/2)` the conformal-rotation part and
/// `(p, q) = ((A₀₀ − A₁₁)/2, (A₀₁ + A₁₀)/2)` the conformal-reflection part.
/// `A` is a scaled rotation exactly when `(p, q) = 0`, and a scaled reflection
/// exactly when `(a, b) = 0`. Both constraints leave the scale free, because
/// two independently learned chart blocks have unidentified relative radii.
///
/// The sampling covariance of `vec(A)` is the HC2 sandwich
/// `(I ⊗ S⁻¹) [Σᵢ Ωᵢ ⊗ xᵢxᵢᵀ] (I ⊗ S⁻¹)` with `S = XᵀX` and
/// `Ωᵢ = eᵢeᵢᵀ/(1 − hᵢ) + Q_{y,i} + AᵀQ_{x,i}A`. Here `eᵢ` is the residual and
/// `hᵢ = xᵢᵀS⁻¹xᵢ` the leverage; HC2 makes the meat unbiased under
/// homoskedastic errors. `Q_{·,i}` is the uniform-quantization variance
/// `ulp²/12` of the f32 codes the routes store. A code is known only to half an
/// f32 spacing, so this is the input's own resolution. It is negligible beside
/// any sampling noise, and it keeps the covariance of an exactly consistent pair
/// equal to the input resolution instead of `0/0`.
///
/// Each Wald statistic is referred to its χ² distribution:
/// `(p, q)` and `(a, b)` against χ²₂, and `vec(A) = 0` against χ²₄. The
/// conformality p-value is that of the union null (rotation or reflection),
/// `max(p_rot, p_ref)`. It is exact when the scale is resolved, because the
/// other class's p-value then vanishes.
#[derive(Clone, Debug)]
pub struct PlanarTransferTest {
    pub n_rows: usize,
    /// Least-squares operator with `X_a·A ≈ X_b`.
    pub operator: [[f64; 2]; 2],
    /// The conformal class with the larger p-value.
    pub orientation: PlanarTransferOrientation,
    /// Relative radius `c`: the norm of the in-class coordinates.
    pub scale: f64,
    /// Frobenius norm of the off-class component of `A`.
    pub conformal_defect: f64,
    /// Null root-mean-square of [`Self::conformal_defect`],
    /// `√(E‖defect‖²) = √(2·tr Cov(off-class coordinates))`.
    pub conformal_defect_se: f64,
    pub conformal_statistic: f64,
    pub conformal_p_value: f64,
    /// Wald statistic of `A = 0`: no linear transfer between the charts.
    pub existence_statistic: f64,
    pub existence_p_value: f64,
}

/// Uniform-quantization variance `ulp(v)²/12` of one stored f32 code.
fn f32_quantization_variance(value: f32) -> f64 {
    let magnitude = value.abs();
    let next = f32::from_bits(magnitude.to_bits() + 1);
    let spacing = f64::from(next) - f64::from(magnitude);
    spacing * spacing / 12.0
}

/// `vᵀ C⁻¹ v` for a symmetric positive-definite `C` through its Cholesky
/// factor; `None` when `C` is not numerically positive definite.
fn spd_quadratic_form<const N: usize>(cov: &[[f64; N]; N], v: &[f64; N]) -> Option<f64> {
    let mut l = [[0.0_f64; N]; N];
    for i in 0..N {
        for j in 0..=i {
            let mut sum = cov[i][j];
            for k in 0..j {
                sum -= l[i][k] * l[j][k];
            }
            if i == j {
                if !(sum > 0.0 && sum.is_finite()) {
                    return None;
                }
                l[i][i] = sum.sqrt();
            } else {
                l[i][j] = sum / l[j][j];
            }
        }
    }
    let mut z = [0.0_f64; N];
    for i in 0..N {
        let mut sum = v[i];
        for k in 0..i {
            sum -= l[i][k] * z[k];
        }
        z[i] = sum / l[i][i];
    }
    let form = z.iter().map(|value| value * value).sum::<f64>();
    form.is_finite().then_some(form)
}

impl PlanarTransferTest {
    /// Test the transfer carried by paired f32 codes, one row per co-firing
    /// observation. Needs at least three rows: `2n` responses fit four
    /// coefficients, so `n ≥ 3` leaves residual degrees of freedom.
    pub fn from_codes(x_a: &[[f32; 2]], x_b: &[[f32; 2]]) -> Result<Self, String> {
        let n = x_a.len();
        if x_b.len() != n {
            return Err(format!(
                "planar transfer test needs paired rows, got {n} and {}",
                x_b.len()
            ));
        }
        if n < 3 {
            return Err(format!(
                "planar transfer test needs at least three co-firing rows to leave residual degrees of freedom, got {n}"
            ));
        }
        if x_a
            .iter()
            .chain(x_b)
            .any(|row| !(row[0].is_finite() && row[1].is_finite()))
        {
            return Err("planar transfer test received a non-finite code".to_string());
        }
        let x = |i: usize| [f64::from(x_a[i][0]), f64::from(x_a[i][1])];
        let y = |i: usize| [f64::from(x_b[i][0]), f64::from(x_b[i][1])];
        let mut gram = [[0.0_f64; 2]; 2];
        let mut cross = [[0.0_f64; 2]; 2];
        for i in 0..n {
            let (xi, yi) = (x(i), y(i));
            for k in 0..2 {
                for m in 0..2 {
                    gram[k][m] += xi[k] * xi[m];
                    cross[k][m] += xi[k] * yi[m];
                }
            }
        }
        let det = gram[0][0] * gram[1][1] - gram[0][1] * gram[1][0];
        if !(det > 0.0 && det.is_finite()) {
            return Err(format!(
                "chart-a overlap codes span less than the plane (Gram determinant {det})"
            ));
        }
        let gram_inv = [
            [gram[1][1] / det, -gram[0][1] / det],
            [-gram[1][0] / det, gram[0][0] / det],
        ];
        // operator[k][j]: regressor coordinate k, response coordinate j.
        let mut operator = [[0.0_f64; 2]; 2];
        for k in 0..2 {
            for j in 0..2 {
                operator[k][j] = gram_inv[k][0] * cross[0][j] + gram_inv[k][1] * cross[1][j];
            }
        }
        // Meat indexed by θ = vec(A) with θ[2j + k] = A[k][j].
        let mut meat = [[0.0_f64; 4]; 4];
        for i in 0..n {
            let (xi, yi) = (x(i), y(i));
            let leverage = (0..2)
                .map(|k| (0..2).map(|m| xi[k] * gram_inv[k][m] * xi[m]).sum::<f64>())
                .sum::<f64>();
            let free = 1.0 - leverage;
            if !(free > 0.0) {
                return Err(format!(
                    "overlap row {i} has unit leverage, so its residual carries no noise information"
                ));
            }
            let residual: [f64; 2] =
                std::array::from_fn(|j| yi[j] - operator[0][j] * xi[0] - operator[1][j] * xi[1]);
            let qx = [
                f32_quantization_variance(x_a[i][0]),
                f32_quantization_variance(x_a[i][1]),
            ];
            let qy = [
                f32_quantization_variance(x_b[i][0]),
                f32_quantization_variance(x_b[i][1]),
            ];
            let mut omega = [[0.0_f64; 2]; 2];
            for j in 0..2 {
                for l in 0..2 {
                    omega[j][l] = residual[j] * residual[l] / free
                        + (0..2)
                            .map(|k| operator[k][j] * qx[k] * operator[k][l])
                            .sum::<f64>();
                }
                omega[j][j] += qy[j];
            }
            for j in 0..2 {
                for k in 0..2 {
                    for l in 0..2 {
                        for m in 0..2 {
                            meat[2 * j + k][2 * l + m] += omega[j][l] * xi[k] * xi[m];
                        }
                    }
                }
            }
        }
        let mut cov = [[0.0_f64; 4]; 4];
        for j in 0..2 {
            for k in 0..2 {
                for l in 0..2 {
                    for m in 0..2 {
                        cov[2 * j + k][2 * l + m] = (0..2)
                            .map(|kk| {
                                (0..2)
                                    .map(|mm| {
                                        gram_inv[k][kk]
                                            * meat[2 * j + kk][2 * l + mm]
                                            * gram_inv[mm][m]
                                    })
                                    .sum::<f64>()
                            })
                            .sum();
                    }
                }
            }
        }
        // φ = (a, b, p, q) = Lθ with θ = (A₀₀, A₁₀, A₀₁, A₁₁).
        let lmap = [
            [0.5, 0.0, 0.0, 0.5],
            [0.0, 0.5, -0.5, 0.0],
            [0.5, 0.0, 0.0, -0.5],
            [0.0, 0.5, 0.5, 0.0],
        ];
        let theta = [
            operator[0][0],
            operator[1][0],
            operator[0][1],
            operator[1][1],
        ];
        let phi: [f64; 4] = std::array::from_fn(|r| (0..4).map(|c| lmap[r][c] * theta[c]).sum());
        let mut phi_cov = [[0.0_f64; 4]; 4];
        for r in 0..4 {
            for s in 0..4 {
                phi_cov[r][s] = (0..4)
                    .map(|c| {
                        (0..4)
                            .map(|d| lmap[r][c] * cov[c][d] * lmap[s][d])
                            .sum::<f64>()
                    })
                    .sum();
            }
        }
        let block = |offset: usize| -> ([f64; 2], [[f64; 2]; 2]) {
            (
                [phi[offset], phi[offset + 1]],
                [
                    [phi_cov[offset][offset], phi_cov[offset][offset + 1]],
                    [phi_cov[offset + 1][offset], phi_cov[offset + 1][offset + 1]],
                ],
            )
        };
        let singular = || "planar transfer covariance is not positive definite".to_string();
        let (rotation, rotation_cov) = block(0);
        let (reflection, reflection_cov) = block(2);
        // Rotation null: the reflection coordinates vanish, and vice versa.
        let rotation_statistic =
            spd_quadratic_form(&reflection_cov, &reflection).ok_or_else(singular)?;
        let reflection_statistic =
            spd_quadratic_form(&rotation_cov, &rotation).ok_or_else(singular)?;
        let existence_statistic = spd_quadratic_form(&phi_cov, &phi).ok_or_else(singular)?;
        let rotation_p = gam_math::probability::chi_square_sf(rotation_statistic, 2.0);
        let reflection_p = gam_math::probability::chi_square_sf(reflection_statistic, 2.0);
        let norm = |v: [f64; 2]| v[0].hypot(v[1]);
        let (orientation, in_class, off_class, off_cov, statistic, p_value) =
            if rotation_p >= reflection_p {
                (
                    PlanarTransferOrientation::Preserving,
                    rotation,
                    reflection,
                    reflection_cov,
                    rotation_statistic,
                    rotation_p,
                )
            } else {
                (
                    PlanarTransferOrientation::Reversing,
                    reflection,
                    rotation,
                    rotation_cov,
                    reflection_statistic,
                    reflection_p,
                )
            };
        Ok(Self {
            n_rows: n,
            operator,
            orientation,
            scale: norm(in_class),
            conformal_defect: std::f64::consts::SQRT_2 * norm(off_class),
            conformal_defect_se: (2.0 * (off_cov[0][0] + off_cov[1][1])).sqrt(),
            conformal_statistic: statistic,
            conformal_p_value: p_value,
            existence_statistic,
            existence_p_value: gam_math::probability::chi_square_sf(existence_statistic, 4.0),
        })
    }
}

/// Transfer evidence for one connected chart-overlap component.
///
/// The gate is valid at a per-edge level `α_e` when the transfer exists
/// (`vec(A) = 0` is rejected, `p_exist ≤ α_e`) and is conformal (the union
/// rotation-or-reflection null is not rejected, `p_conf > α_e`). An edge can
/// then be wrongly admitted only when an absent transfer passes the existence
/// test, and wrongly refused only when a conformal transfer fails the
/// conformality test. Each happens with probability at most `α_e`. With no
/// level the gate carries its test but certifies nothing.
#[derive(Clone, Debug)]
pub struct AtlasTransferGate {
    edge: AtlasHolonomyEdgeId,
    pub test: PlanarTransferTest,
    pub level: Option<f64>,
    pub valid: bool,
}

impl AtlasTransferGate {
    pub fn from_test(
        edge: AtlasHolonomyEdgeId,
        test: PlanarTransferTest,
        level: Option<f64>,
    ) -> Self {
        let valid = level
            .is_some_and(|alpha| test.existence_p_value <= alpha && test.conformal_p_value > alpha);
        Self {
            edge,
            test,
            level,
            valid,
        }
    }

    #[must_use]
    pub fn edge(&self) -> AtlasHolonomyEdgeId {
        self.edge
    }
}

/// One admitted or rejected pair after both gates are evaluated.
#[derive(Clone, Debug)]
pub struct AtlasNerveEdge {
    pub a: usize,
    pub b: usize,
    /// Connected overlap-component identity inherited from the transfer gate.
    /// When no transfer evidence exists, the rejected aggregate audit row uses
    /// identity zero and can never enter the certified inventory.
    pub overlap: usize,
    pub coactivation_mass: f64,
    pub coactivation_threshold: f64,
    pub transfer_valid: bool,
    pub admitted: bool,
    pub filtration: f64,
}

/// Filtered nerve diagram and its exact Betti readout.
#[derive(Clone, Debug)]
pub struct AtlasNerveDiagram {
    pub betti: BettiSignature,
    pub null_calibration: Option<ClaimNullCalibration>,
    pub n_vertices: usize,
    pub n_edges: usize,
    pub n_triangles: usize,
    pub n_tetrahedra: usize,
    /// Number of admitted simplices by cardinality: index zero counts vertices,
    /// index one edges, and so on through the full nerve.
    pub simplex_counts: Vec<usize>,
    /// Full alternating simplex sum `N1 - N2 + N3 - ...`.
    pub euler_characteristic: i128,
    /// The authoritative holonomy proof, including statistical refusals and
    /// every scalar needed to audit a noisy decision.
    pub holonomy_certificate: Option<AtlasHolonomyCertificate>,
    pub sampled_support_size: usize,
    pub covering_side: AtlasCoveringSide,
    pub max_filtration: f64,
    pub edges: Vec<AtlasNerveEdge>,
    pub note: String,
}

impl AtlasNerveDiagram {
    /// Orientability only when the preserved authoritative certificate signed
    /// that claim.  Missing and statistically refused certificates both remain
    /// non-promotable, while the certificate itself retains the distinction.
    #[must_use]
    pub fn certified_orientability(&self) -> Option<AtlasOrientability> {
        self.holonomy_certificate
            .as_ref()
            .and_then(AtlasHolonomyCertificate::certified_orientability)
    }

    /// Independently certified integer curvature readout, when the noisy-PCA
    /// certificate carried a Gauss--Bonnet input and its rounding confidence
    /// met the allocated familywise level.
    #[must_use]
    pub fn certified_gauss_bonnet_euler_characteristic(&self) -> Option<AtlasEulerCharacteristic> {
        self.holonomy_certificate
            .as_ref()
            .and_then(AtlasHolonomyCertificate::certified_euler_characteristic)
    }
}

/// The closed-form classification of compact surfaces, read off the exact
/// `GF(2)` homology of the full nerve plus the orientation class.
///
/// `(χ, orientability, boundary)` is a COMPLETE invariant of a compact surface,
/// so this is a table lookup on measured invariants — not a search over a
/// candidate menu. It is the single surface table in the crate, and the observed
/// local-chart stack reaches it through `manifold::atlas_topology`.
///
/// It deliberately does NOT cover the one-manifolds. A circle and a cylinder are
/// homotopy equivalent, hence share `(b₀, b₁, b₂, χ) = (1, 1, 0, 0)` with a
/// trivial orientation class: the nerve alone cannot separate them, and only the
/// LOCAL CHART RANK `d` can. Callers that know `d` dispatch on it before
/// consulting this table.
pub(crate) fn surface_from_invariants(
    betti: BettiSignature,
    euler_characteristic: i128,
    orientability: AtlasOrientability,
) -> Option<(GraphCompressionKind, &'static str)> {
    match (
        betti.b0,
        betti.b1,
        betti.b2,
        euler_characteristic,
        orientability,
    ) {
        (1, 2, Some(1), 0, AtlasOrientability::Orientable) => {
            Some((GraphCompressionKind::Torus, "torus"))
        }
        (1, 1, Some(0), 0, AtlasOrientability::Orientable) => {
            Some((GraphCompressionKind::Cylinder, "cylinder"))
        }
        (1, 0, Some(1), 2, AtlasOrientability::Orientable) => {
            Some((GraphCompressionKind::Sphere, "sphere"))
        }
        // A contractible bounded surface: orientable, no handle (b₁ = 0), no
        // closed 2-cycle (b₂ = 0), χ = 1. This is the sheet a swiss roll glues to
        // — distinguished from the sphere purely by χ (1 vs 2) and the absent
        // b₂, exactly the robust simply-connected discriminant the geometry
        // review endorsed (#2280).
        (1, 0, Some(0), 1, AtlasOrientability::Orientable) => {
            Some((GraphCompressionKind::Disk, "disk"))
        }
        // The Möbius strip: the non-orientable counterpart of the cylinder (same
        // F₂ homology b₁ = 1, b₂ = 0, χ = 0), separated ONLY by the certified
        // orientation cocycle — the half-twist read as a discrete sign, the
        // orientability review's reliable core (#2280).
        (1, 1, Some(0), 0, AtlasOrientability::NonOrientable) => {
            Some((GraphCompressionKind::MobiusStrip, "mobius_strip"))
        }
        (1, 1, Some(1), 1, AtlasOrientability::NonOrientable) => {
            Some((GraphCompressionKind::ProjectivePlane, "projective_plane"))
        }
        (1, 2, Some(1), 0, AtlasOrientability::NonOrientable) => {
            Some((GraphCompressionKind::KleinBottle, "klein_bottle"))
        }
        _ => None,
    }
}

fn validate_charts(charts: &[AtlasChart]) -> Result<usize, String> {
    let Some(first) = charts.first() else {
        return Ok(0);
    };
    let n = first.row_count;
    for (pos, chart) in charts.iter().enumerate() {
        if chart.chart_idx != pos {
            return Err(format!(
                "atlas chart indices must be contiguous; chart at position {pos} has index {}",
                chart.chart_idx
            ));
        }
        if chart.row_count != n {
            return Err(format!(
                "atlas chart {} has {} rows but expected {n}",
                chart.chart_idx, chart.row_count
            ));
        }
        if chart.support_rows.len() != chart.support_weights.len() {
            return Err(format!(
                "atlas chart {} has mismatched sparse row and weight lengths",
                chart.chart_idx
            ));
        }
        let mut previous = None;
        for (&row, &weight) in chart.support_rows.iter().zip(&chart.support_weights) {
            if row >= n || previous.is_some_and(|prior| prior >= row) {
                return Err(format!(
                    "atlas chart {} has invalid sparse support ordering/domain",
                    chart.chart_idx
                ));
            }
            if !(weight.is_finite() && weight > 0.0) {
                return Err(format!(
                    "atlas chart {} has non-finite or non-positive sparse weight",
                    chart.chart_idx
                ));
            }
            previous = Some(row);
        }
    }
    Ok(n)
}

fn mutual_row_mass(charts: &[AtlasChart], simplex: &[usize]) -> (f64, f64) {
    if simplex.is_empty() {
        return (0.0, f64::INFINITY);
    }
    let mut positions = vec![0usize; simplex.len()];
    let mut total = 0.0_f64;
    let mut positive_count = 0usize;

    loop {
        if simplex
            .iter()
            .zip(&positions)
            .any(|(&chart_idx, &position)| position == charts[chart_idx].support_rows.len())
        {
            break;
        }
        let target_row = simplex
            .iter()
            .zip(&positions)
            .map(|(&chart_idx, &position)| charts[chart_idx].support_rows[position])
            .max()
            .expect("non-empty simplex");
        for (&chart_idx, position) in simplex.iter().zip(&mut positions) {
            while *position < charts[chart_idx].support_rows.len()
                && charts[chart_idx].support_rows[*position] < target_row
            {
                *position += 1;
            }
        }
        if simplex
            .iter()
            .zip(&positions)
            .any(|(&chart_idx, &position)| position == charts[chart_idx].support_rows.len())
        {
            break;
        }
        if simplex
            .iter()
            .zip(&positions)
            .all(|(&chart_idx, &position)| charts[chart_idx].support_rows[position] == target_row)
        {
            let row_mass = simplex
                .iter()
                .zip(&positions)
                .map(|(&chart_idx, &position)| charts[chart_idx].support_weights[position])
                .fold(f64::INFINITY, f64::min);
            total += row_mass;
            positive_count += 1;
            for position in &mut positions {
                *position += 1;
            }
        }
    }
    let threshold = if positive_count > 0 {
        total / positive_count as f64
    } else {
        f64::INFINITY
    };
    (total, threshold)
}

#[derive(Clone, Copy, Debug, Default)]
struct PairOverlap {
    mass: f64,
    positive_rows: usize,
}

/// Merge the per-chart sorted support lists by row and accumulate only chart
/// pairs that genuinely co-activate. Working memory is `O(charts + row_width)`;
/// the output is proportional to observed pair support, never all chart pairs.
fn sparse_pair_overlaps(charts: &[AtlasChart]) -> (BTreeMap<(usize, usize), PairOverlap>, usize) {
    let mut heap = BinaryHeap::<Reverse<(usize, usize, usize)>>::new();
    for (chart_idx, chart) in charts.iter().enumerate() {
        if let Some(&row) = chart.support_rows.first() {
            heap.push(Reverse((row, chart_idx, 0)));
        }
    }
    let mut overlaps = BTreeMap::<(usize, usize), PairOverlap>::new();
    let mut sampled_rows = 0usize;
    let mut live = Vec::<(usize, f64)>::new();
    while let Some(Reverse((row, _, _))) = heap.peek().copied() {
        live.clear();
        while heap
            .peek()
            .is_some_and(|Reverse((candidate, _, _))| *candidate == row)
        {
            let Reverse((_, chart_idx, position)) = heap.pop().expect("peeked atlas support entry");
            let chart = &charts[chart_idx];
            live.push((chart_idx, chart.support_weights[position]));
            let next = position + 1;
            if next < chart.support_rows.len() {
                heap.push(Reverse((chart.support_rows[next], chart_idx, next)));
            }
        }
        sampled_rows += 1;
        for left in 0..live.len() {
            for right in (left + 1)..live.len() {
                let (a, wa) = live[left];
                let (b, wb) = live[right];
                let pair = overlaps.entry((a, b)).or_default();
                pair.mass += wa.min(wb);
                pair.positive_rows += 1;
            }
        }
    }
    (overlaps, sampled_rows)
}

fn chart_distance(a: &AtlasChart, b: &AtlasChart, overlap: f64) -> f64 {
    let denom = a.support_mass.min(b.support_mass);
    if denom > 0.0 {
        (1.0 - overlap / denom).clamp(0.0, 1.0)
    } else {
        1.0
    }
}

fn dtm_radii(charts: &[AtlasChart], overlaps: &BTreeMap<(usize, usize), PairOverlap>) -> Vec<f64> {
    let n = charts.len();
    if n <= 1 {
        return vec![0.0; n];
    }
    let total_mass = charts.iter().map(|chart| chart.support_mass).sum::<f64>();
    if !(total_mass.is_finite() && total_mass > 0.0) {
        return vec![0.0; n];
    }
    let target_mass = total_mass / n as f64;
    let mut sparse_neighbors = vec![Vec::<(f64, f64)>::new(); n];
    for (&(a, b), overlap) in overlaps {
        let distance = chart_distance(&charts[a], &charts[b], overlap.mass);
        sparse_neighbors[a].push((distance, charts[b].support_mass));
        sparse_neighbors[b].push((distance, charts[a].support_mass));
    }
    let mut radii = vec![0.0_f64; n];
    for i in 0..n {
        let mut neighbors = std::mem::take(&mut sparse_neighbors[i]);
        if charts[i].support_mass > 0.0 {
            neighbors.push((0.0, charts[i].support_mass));
        }
        neighbors.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
        let mut mass = 0.0_f64;
        let mut moment = 0.0_f64;
        for (dist, weight) in neighbors {
            let take = (target_mass - mass).min(weight);
            if take > 0.0 {
                moment += take * dist * dist;
                mass += take;
            }
            if mass >= target_mass {
                break;
            }
        }
        // Every omitted chart has zero overlap and therefore distance exactly
        // one. Their individual masses need not be materialised or sorted.
        if mass < target_mass {
            moment += target_mass - mass;
            mass = target_mass;
        }
        if mass > 0.0 {
            radii[i] = (moment / mass).sqrt();
        }
    }
    radii
}

fn xor_sorted(a: &[usize], b: &[usize]) -> Vec<usize> {
    let mut out = Vec::with_capacity(a.len() + b.len());
    let mut i = 0usize;
    let mut j = 0usize;
    while i < a.len() || j < b.len() {
        if j == b.len() || (i < a.len() && a[i] < b[j]) {
            out.push(a[i]);
            i += 1;
        } else if i == a.len() || b[j] < a[i] {
            out.push(b[j]);
            j += 1;
        } else {
            i += 1;
            j += 1;
        }
    }
    out
}

fn gf2_rank(columns: Vec<Vec<usize>>) -> usize {
    let mut pivots: HashMap<usize, Vec<usize>> = HashMap::new();
    let mut rank = 0usize;
    for mut col in columns {
        col.sort_unstable();
        while let Some(&pivot) = col.last() {
            if let Some(existing) = pivots.get(&pivot) {
                col = xor_sorted(&col, existing);
            } else {
                pivots.insert(pivot, col);
                rank += 1;
                break;
            }
        }
    }
    rank
}

fn boundary_rank(lower: &[Vec<usize>], upper: &[Vec<usize>]) -> usize {
    if lower.is_empty() || upper.is_empty() {
        return 0;
    }
    let mut lower_index = HashMap::with_capacity(lower.len());
    for (idx, simplex) in lower.iter().enumerate() {
        lower_index.insert(simplex.clone(), idx);
    }
    let mut columns = Vec::with_capacity(upper.len());
    for simplex in upper {
        let mut column = Vec::with_capacity(simplex.len());
        for drop in 0..simplex.len() {
            let mut face = Vec::with_capacity(simplex.len() - 1);
            for (pos, &vertex) in simplex.iter().enumerate() {
                if pos != drop {
                    face.push(vertex);
                }
            }
            if let Some(&row) = lower_index.get(&face) {
                column.push(row);
            }
        }
        columns.push(column);
    }
    gf2_rank(columns)
}

/// Exact `GF(2)` Betti numbers `b0, b1, b2` of the retained low-dimensional
/// skeleton, from the ranks of the three boundary matrices.
pub(crate) fn compute_betti(
    vertices: &[Vec<usize>],
    edges: &[Vec<usize>],
    triangles: &[Vec<usize>],
    tetrahedra: &[Vec<usize>],
) -> BettiSignature {
    let rank_d1 = boundary_rank(vertices, edges);
    let rank_d2 = boundary_rank(edges, triangles);
    let rank_d3 = boundary_rank(triangles, tetrahedra);
    let b0 = vertices.len().saturating_sub(rank_d1);
    let b1 = edges.len().saturating_sub(rank_d1 + rank_d2);
    let b2 = triangles.len().saturating_sub(rank_d2 + rank_d3);
    BettiSignature {
        b0,
        b1,
        b2: Some(b2),
    }
}

/// Streamed inventory of an enumerated nerve: exact counts and alternating sum
/// at every cardinality, with the boundary-matrix inputs retained only through
/// dimension three (the homology this stack reads).
#[derive(Debug)]
pub(crate) struct SimplexInventory {
    pub(crate) counts: Vec<usize>,
    pub(crate) vertices: Vec<Vec<usize>>,
    pub(crate) edges: Vec<Vec<usize>>,
    pub(crate) triangles: Vec<Vec<usize>>,
    pub(crate) tetrahedra: Vec<Vec<usize>>,
    pub(crate) euler_characteristic: i128,
}

fn record_simplex(simplex: &[usize], inventory: &mut SimplexInventory) -> Result<(), String> {
    let cardinality = simplex.len();
    inventory.counts[cardinality - 1] = inventory.counts[cardinality - 1]
        .checked_add(1)
        .ok_or_else(|| "atlas nerve simplex count overflowed usize".to_string())?;
    let count = i128::try_from(inventory.counts[cardinality - 1])
        .map_err(|_| "atlas nerve simplex count overflowed i128".to_string())?;
    let previous = count - 1;
    inventory.euler_characteristic = if cardinality % 2 == 1 {
        inventory.euler_characteristic.checked_add(count - previous)
    } else {
        inventory.euler_characteristic.checked_sub(count - previous)
    }
    .ok_or_else(|| "atlas nerve Euler characteristic overflowed i128".to_string())?;

    let roster = match cardinality {
        1 => Some(&mut inventory.vertices),
        2 => Some(&mut inventory.edges),
        3 => Some(&mut inventory.triangles),
        4 => Some(&mut inventory.tetrahedra),
        // Simplices of dimension 4 and above are counted and folded into the
        // Euler characteristic above, but the inventory keeps explicit rosters
        // only through tetrahedra.
        _ => None,
    };
    if let Some(roster) = roster {
        roster.push(simplex.to_vec());
    }
    Ok(())
}

fn enumerate_simplices_from(
    nonempty_intersection: &dyn Fn(&[usize]) -> bool,
    adjacency: &[BTreeSet<usize>],
    prefix: &mut Vec<usize>,
    candidates: &[usize],
    inventory: &mut SimplexInventory,
) -> Result<(), String> {
    for (position, &vertex) in candidates.iter().enumerate() {
        prefix.push(vertex);
        let nonempty = prefix.len() == 1 || nonempty_intersection(prefix);
        if nonempty {
            record_simplex(prefix, inventory)?;
            let next: Vec<usize> = candidates[(position + 1)..]
                .iter()
                .copied()
                .filter(|candidate| adjacency[vertex].contains(candidate))
                .collect();
            enumerate_simplices_from(nonempty_intersection, adjacency, prefix, &next, inventory)?;
        }
        prefix.pop();
    }
    Ok(())
}

/// Enumerate the FULL nerve of a cover: every set of charts whose common
/// intersection is non-empty, at every cardinality, not a truncation at triples.
///
/// The cover enters only through `nonempty_intersection`, the predicate deciding
/// whether a candidate chart set co-fires, and through `adjacency`, the admitted
/// pairwise 1-skeleton the cliques are grown inside. Keeping the geometry behind
/// that predicate is what lets the weighted dictionary cover
/// ([`build_atlas_nerve`], where co-firing is a positive mutual row MASS) and the
/// local-chart cover (`manifold::atlas_topology`, where it is a non-empty
/// intersection of patch memberships) share one exact enumerator instead of two
/// drifting copies of the same recursion.
///
/// Truncating at triples is wrong for any data cover — 4-, 5-, and 6-way overlaps
/// are routinely non-empty and each carries its own sign in
/// `χ = Σ_k (−1)^k N_{k+1}` — so the alternating sum is streamed over ALL
/// cardinalities while only the dimension ≤ 3 boundary inputs are retained.
pub(crate) fn enumerate_full_nerve(
    chart_count: usize,
    nonempty_intersection: &dyn Fn(&[usize]) -> bool,
    adjacency: &[BTreeSet<usize>],
) -> Result<SimplexInventory, String> {
    let mut inventory = SimplexInventory {
        counts: vec![0; chart_count],
        vertices: Vec::with_capacity(chart_count),
        edges: Vec::new(),
        triangles: Vec::new(),
        tetrahedra: Vec::new(),
        euler_characteristic: 0,
    };
    let candidates: Vec<usize> = (0..chart_count).collect();
    enumerate_simplices_from(
        nonempty_intersection,
        adjacency,
        &mut Vec::new(),
        &candidates,
        &mut inventory,
    )?;
    Ok(inventory)
}

fn validate_holonomy_certificate(
    chart_count: usize,
    admitted_edges: &BTreeSet<AtlasHolonomyEdgeId>,
    certificate: Option<&AtlasHolonomyCertificate>,
) -> Result<(), String> {
    let Some(certificate) = certificate else {
        return Ok(());
    };
    if certificate.chart_count() != chart_count {
        return Err(format!(
            "holonomy certificate is for {} charts but the atlas has {chart_count}",
            certificate.chart_count(),
        ));
    }
    let inventory = certificate.edge_inventory();
    let certified_edges: BTreeSet<AtlasHolonomyEdgeId> = inventory.iter().copied().collect();
    if certified_edges.len() != inventory.len() {
        return Err("holonomy certificate contains duplicate overlap-edge identities".to_string());
    }
    if admitted_edges != &certified_edges {
        let missing: Vec<_> = admitted_edges
            .difference(&certified_edges)
            .copied()
            .collect();
        let surplus: Vec<_> = certified_edges
            .difference(&admitted_edges)
            .copied()
            .collect();
        return Err(format!(
            "holonomy certificate does not match the full admitted (a, b, overlap) inventory: missing={missing:?}, surplus={surplus:?}"
        ));
    }
    Ok(())
}

/// Build the certificate-gated atlas nerve and read Betti numbers through H2.
#[must_use = "atlas nerve construction errors must be handled"]
pub fn build_atlas_nerve(
    charts: &[AtlasChart],
    transfer_gates: &[AtlasTransferGate],
    holonomy_certificate: Option<AtlasHolonomyCertificate>,
) -> Result<AtlasNerveDiagram, String> {
    let row_count = validate_charts(charts)?;
    let n = charts.len();
    if n == 0 {
        validate_holonomy_certificate(0, &BTreeSet::new(), holonomy_certificate.as_ref())?;
        return Ok(AtlasNerveDiagram {
            betti: BettiSignature {
                b0: 0,
                b1: 0,
                b2: Some(0),
            },
            null_calibration: None,
            n_vertices: 0,
            n_edges: 0,
            n_triangles: 0,
            n_tetrahedra: 0,
            simplex_counts: Vec::new(),
            euler_characteristic: 0,
            holonomy_certificate,
            sampled_support_size: 0,
            covering_side: AtlasCoveringSide::BelowCoveringNumber,
            max_filtration: 0.0,
            edges: Vec::new(),
            note: "empty atlas nerve".to_string(),
        });
    }

    let mut gate_map = BTreeMap::<(usize, usize), BTreeMap<usize, &AtlasTransferGate>>::new();
    for gate in transfer_gates {
        let edge = gate.edge();
        if edge.b() >= n {
            return Err(format!(
                "transfer gate ({}, {}, overlap {}) is outside the {n}-chart atlas",
                edge.a(),
                edge.b(),
                edge.overlap()
            ));
        }
        if gate_map
            .entry((edge.a(), edge.b()))
            .or_default()
            .insert(edge.overlap(), gate)
            .is_some()
        {
            return Err(format!(
                "duplicate transfer gate for atlas edge ({}, {}, overlap {})",
                edge.a(),
                edge.b(),
                edge.overlap()
            ));
        }
    }

    let (overlaps, sampled) = sparse_pair_overlaps(charts);
    let dtm = dtm_radii(charts, &overlaps);

    let mut adjacency = vec![BTreeSet::<usize>::new(); n];
    let mut edge_reports = Vec::new();
    let mut max_filtration = 0.0_f64;
    for (&(a, b), overlap) in &overlaps {
        let threshold = overlap.mass / overlap.positive_rows as f64;
        let coactive = overlap.mass.is_finite() && overlap.mass >= threshold;
        let filtration = chart_distance(&charts[a], &charts[b], overlap.mass)
            .max(dtm[a])
            .max(dtm[b]);
        let mut record_component = |overlap_id: usize, transfer_valid: bool| {
            let admitted = coactive && transfer_valid;
            if admitted {
                adjacency[a].insert(b);
                adjacency[b].insert(a);
                max_filtration = max_filtration.max(filtration);
            }
            edge_reports.push(AtlasNerveEdge {
                a,
                b,
                overlap: overlap_id,
                coactivation_mass: overlap.mass,
                coactivation_threshold: threshold,
                transfer_valid,
                admitted,
                filtration,
            });
        };
        if let Some(component_gates) = gate_map.get(&(a, b)) {
            for (&overlap_id, gate) in component_gates {
                record_component(overlap_id, gate.valid);
            }
        } else {
            // Preserve the observed overlap in the audit report, but without a
            // transfer certificate it is deliberately not admitted.
            record_component(0, false);
        }
    }

    // Enumerate each clique once in canonical vertex order.  Only the boundary
    // matrices through dimension three are retained; higher simplices are
    // streamed into their exact counts and alternating sum, so working memory
    // does not scale with the potentially exponential full nerve.
    let inventory = enumerate_full_nerve(
        n,
        &|simplex: &[usize]| {
            let (mass, _) = mutual_row_mass(charts, simplex);
            mass.is_finite() && mass > 0.0
        },
        &adjacency,
    )?;
    let admitted_edge_inventory: BTreeSet<AtlasHolonomyEdgeId> = edge_reports
        .iter()
        .filter(|edge| edge.admitted)
        .map(|edge| AtlasHolonomyEdgeId::new(edge.a, edge.b, edge.overlap))
        .collect::<Result<_, _>>()?;
    validate_holonomy_certificate(n, &admitted_edge_inventory, holonomy_certificate.as_ref())?;
    let certified_orientability = holonomy_certificate
        .as_ref()
        .and_then(AtlasHolonomyCertificate::certified_orientability);

    let covering_side = if sampled >= n {
        AtlasCoveringSide::AtOrAboveCoveringNumber
    } else {
        AtlasCoveringSide::BelowCoveringNumber
    };
    let betti = compute_betti(
        &inventory.vertices,
        &inventory.edges,
        &inventory.triangles,
        &inventory.tetrahedra,
    );
    let note = format!(
        "atlas nerve over {n} charts and {row_count} rows: sampled_support_size={sampled}, covering_side={}, certified_orientability={certified_orientability:?}, Euler={}, Betti=({}, {}, {:?})",
        covering_side.as_str(),
        inventory.euler_characteristic,
        betti.b0,
        betti.b1,
        betti.b2
    );

    Ok(AtlasNerveDiagram {
        betti,
        null_calibration: None,
        n_vertices: inventory.vertices.len(),
        n_edges: inventory.edges.len(),
        n_triangles: inventory.triangles.len(),
        n_tetrahedra: inventory.tetrahedra.len(),
        simplex_counts: inventory.counts,
        euler_characteristic: inventory.euler_characteristic,
        holonomy_certificate,
        sampled_support_size: sampled,
        covering_side,
        max_filtration,
        edges: edge_reports,
        note,
    })
}

#[cfg(test)]
mod tests {
    use super::{
        AtlasChart, AtlasTransferGate, PlanarTransferOrientation, PlanarTransferTest,
        build_atlas_nerve,
    };
    use crate::inference::atlas_holonomy::AtlasHolonomyEdgeId;
    use crate::manifold::{AtlasOrientability, GraphCompressionKind};
    use rand::rngs::StdRng;
    use rand::{RngExt, SeedableRng};
    use std::f64::consts::PI;

    /// Evenly spaced unit-circle codes.
    fn circle_codes(n: usize) -> Vec<[f32; 2]> {
        (0..n)
            .map(|i| {
                let angle = 2.0 * PI * i as f64 / n as f64;
                [angle.cos() as f32, angle.sin() as f32]
            })
            .collect()
    }

    /// `y_i = Aᵀx_i` evaluated exactly in f64 and stored as f32.
    fn mapped_codes(codes: &[[f32; 2]], operator: [[f64; 2]; 2]) -> Vec<[f32; 2]> {
        codes
            .iter()
            .map(|x| {
                let (x0, x1) = (f64::from(x[0]), f64::from(x[1]));
                [
                    (operator[0][0] * x0 + operator[1][0] * x1) as f32,
                    (operator[0][1] * x0 + operator[1][1] * x1) as f32,
                ]
            })
            .collect()
    }

    fn gate_from_operator(
        edge: AtlasHolonomyEdgeId,
        operator: [[f64; 2]; 2],
        level: Option<f64>,
    ) -> AtlasTransferGate {
        let x = circle_codes(16);
        let y = mapped_codes(&x, operator);
        AtlasTransferGate::from_test(edge, PlanarTransferTest::from_codes(&x, &y).unwrap(), level)
    }

    fn standard_normal(rng: &mut StdRng) -> f64 {
        let u: f64 = rng.random_range(f64::MIN_POSITIVE..1.0);
        let v: f64 = rng.random_range(0.0..1.0);
        (-2.0 * u.ln()).sqrt() * (2.0 * PI * v).cos()
    }

    /// `n` isotropic codes `x_i` and responses `y_i = Aᵀx_i + σε_i`.
    fn noisy_pair(
        rng: &mut StdRng,
        n: usize,
        operator: [[f64; 2]; 2],
        sigma: f64,
    ) -> (Vec<[f32; 2]>, Vec<[f32; 2]>) {
        let mut x = Vec::with_capacity(n);
        let mut y = Vec::with_capacity(n);
        for _ in 0..n {
            let x0 = standard_normal(rng);
            let x1 = standard_normal(rng);
            let y0 = operator[0][0] * x0 + operator[1][0] * x1 + sigma * standard_normal(rng);
            let y1 = operator[0][1] * x0 + operator[1][1] * x1 + sigma * standard_normal(rng);
            x.push([x0 as f32, x1 as f32]);
            y.push([y0 as f32, y1 as f32]);
        }
        (x, y)
    }

    /// Rejection rates of `p_value` at each level, and the Monte Carlo
    /// standard error `√(α(1 − α)/R)` a calibrated test would show.
    fn assert_calibrated(p_values: &[f64], label: &str) {
        let reps = p_values.len() as f64;
        for alpha in [0.01, 0.05, 0.2] {
            let rate = p_values.iter().filter(|&&p| p <= alpha).count() as f64 / reps;
            let mcse = (alpha * (1.0 - alpha) / reps).sqrt();
            assert!(
                (rate - alpha).abs() <= 4.0 * mcse,
                "{label}: rejection rate {rate} at level {alpha} is outside 4 MCSE ({mcse})"
            );
        }
    }

    fn rotation(scale: f64, angle: f64) -> [[f64; 2]; 2] {
        let (s, c) = angle.sin_cos();
        [[scale * c, -scale * s], [scale * s, scale * c]]
    }

    fn reflection(scale: f64, angle: f64) -> [[f64; 2]; 2] {
        let (s, c) = angle.sin_cos();
        [[scale * c, scale * s], [scale * s, -scale * c]]
    }

    fn charts_from_faces(n_charts: usize, faces: &[Vec<usize>]) -> Vec<AtlasChart> {
        let mut support_rows = vec![Vec::new(); n_charts];
        for (row, face) in faces.iter().enumerate() {
            for &chart in face {
                support_rows[chart].push(row);
            }
        }
        support_rows
            .into_iter()
            .enumerate()
            .map(|(chart, rows)| {
                let weights = vec![1.0; rows.len()];
                AtlasChart::from_sparse_weights(chart, faces.len(), rows, weights).unwrap()
            })
            .collect()
    }

    fn all_valid_pair_gates(n_charts: usize) -> Vec<AtlasTransferGate> {
        let identity = [[1.0, 0.0], [0.0, 1.0]];
        let mut gates = Vec::new();
        for a in 0..n_charts {
            for b in (a + 1)..n_charts {
                gates.push(gate_from_operator(
                    AtlasHolonomyEdgeId::new(a, b, 0).unwrap(),
                    identity,
                    Some(0.05),
                ));
            }
        }
        gates
    }

    #[test]
    fn full_nerve_euler_includes_five_way_overlap() {
        let maximal = vec![vec![0, 1, 2, 3, 4]];
        let charts = charts_from_faces(5, &maximal);
        let gates = all_valid_pair_gates(5);
        let diagram = build_atlas_nerve(&charts, &gates, None).unwrap();
        assert_eq!(diagram.simplex_counts, vec![5, 10, 10, 5, 1]);
        assert_eq!(diagram.euler_characteristic, 1);
        assert_eq!(
            diagram.n_vertices as i128 - diagram.n_edges as i128 + diagram.n_triangles as i128,
            5,
            "the old V-E+F truncation is deliberately wrong on this overlap"
        );
    }

    #[test]
    fn closed_nonorientable_surface_names_require_the_exact_orientation_row() {
        let rp2_betti = crate::manifold::BettiSignature {
            b0: 1,
            b1: 1,
            b2: Some(1),
        };
        let klein_betti = crate::manifold::BettiSignature {
            b0: 1,
            b1: 2,
            b2: Some(1),
        };
        assert_eq!(
            super::surface_from_invariants(rp2_betti, 1, AtlasOrientability::NonOrientable)
                .map(|row| row.0),
            Some(GraphCompressionKind::ProjectivePlane)
        );
        assert_eq!(
            super::surface_from_invariants(klein_betti, 0, AtlasOrientability::NonOrientable)
                .map(|row| row.0),
            Some(GraphCompressionKind::KleinBottle)
        );
        assert_eq!(
            super::surface_from_invariants(klein_betti, 0, AtlasOrientability::Orientable)
                .map(|row| row.0),
            Some(GraphCompressionKind::Torus),
            "the same F2 homology becomes a different surface only through certified orientation"
        );
        assert!(
            super::surface_from_invariants(rp2_betti, 1, AtlasOrientability::Orientable).is_none(),
            "an impossible orientable RP2 signature must not earn any name"
        );
    }

    /// #2280 acceptance shapes the table previously omitted: the contractible
    /// bounded surface a swiss roll glues to (sheet/disk, χ = 1) and the Möbius
    /// strip (χ = 0, non-orientable — the cylinder's F₂ homology separated only by
    /// the orientation cocycle).
    #[test]
    fn sheet_and_mobius_strip_earn_their_certified_names() {
        let disk_betti = crate::manifold::BettiSignature {
            b0: 1,
            b1: 0,
            b2: Some(0),
        };
        let strip_betti = crate::manifold::BettiSignature {
            b0: 1,
            b1: 1,
            b2: Some(0),
        };
        // Sheet: orientable, no handle, no closed 2-cycle, χ = 1.
        assert_eq!(
            super::surface_from_invariants(disk_betti, 1, AtlasOrientability::Orientable)
                .map(|row| row.0),
            Some(GraphCompressionKind::Disk)
        );
        // The sphere signature (b₂ = 1, χ = 2) must NOT collapse to a disk — χ and
        // b₂ are the robust simply-connected discriminant.
        let sphere_betti = crate::manifold::BettiSignature {
            b0: 1,
            b1: 0,
            b2: Some(1),
        };
        assert_eq!(
            super::surface_from_invariants(sphere_betti, 2, AtlasOrientability::Orientable)
                .map(|row| row.0),
            Some(GraphCompressionKind::Sphere)
        );
        // Möbius strip vs cylinder: identical F₂ homology (b₁ = 1, b₂ = 0, χ = 0),
        // separated ONLY by the certified orientation.
        assert_eq!(
            super::surface_from_invariants(strip_betti, 0, AtlasOrientability::NonOrientable)
                .map(|row| row.0),
            Some(GraphCompressionKind::MobiusStrip)
        );
        assert_eq!(
            super::surface_from_invariants(strip_betti, 0, AtlasOrientability::Orientable)
                .map(|row| row.0),
            Some(GraphCompressionKind::Cylinder),
            "the same F₂ homology becomes cylinder vs Möbius only through certified orientation"
        );
        // An orientable disk with a spurious non-orientable label is impossible.
        assert!(
            super::surface_from_invariants(disk_betti, 1, AtlasOrientability::NonOrientable)
                .is_none(),
            "a non-orientable contractible surface is not a named type in this table"
        );
    }

    #[test]
    fn inconsistent_transfer_rejects_cross_cluster_edge() {
        let faces = vec![vec![0, 1], vec![1, 2], vec![2, 3]];
        let charts = charts_from_faces(4, &faces);
        let identity = [[1.0, 0.0], [0.0, 1.0]];
        // An anisotropic stretch is linear but not conformal.
        let stretch = [[1.0, 0.0], [0.0, 2.0]];
        let gates = vec![
            gate_from_operator(
                AtlasHolonomyEdgeId::new(0, 1, 0).unwrap(),
                identity,
                Some(0.05),
            ),
            gate_from_operator(
                AtlasHolonomyEdgeId::new(1, 2, 0).unwrap(),
                stretch,
                Some(0.05),
            ),
            gate_from_operator(
                AtlasHolonomyEdgeId::new(2, 3, 0).unwrap(),
                identity,
                Some(0.05),
            ),
        ];
        let diagram = build_atlas_nerve(&charts, &gates, None).unwrap();
        assert_eq!(diagram.betti.b0, 2);
        assert_eq!(diagram.betti.b1, 0);
        let rejected = diagram
            .edges
            .iter()
            .find(|edge| edge.a == 1 && edge.b == 2)
            .unwrap();
        assert!(rejected.coactivation_mass >= rejected.coactivation_threshold);
        assert!(!rejected.transfer_valid);
        assert!(!rejected.admitted);
    }

    /// #3921: two independently learned blocks have unidentified relative
    /// radii, so a scaled rotation is a valid transfer. The old gate compared
    /// ‖QᵀQ − I‖ to √d·ε and refused it.
    #[test]
    fn scaled_rotation_and_reflection_transfers_are_admitted() {
        let edge = AtlasHolonomyEdgeId::new(0, 1, 0).unwrap();
        let scaled = gate_from_operator(edge, [[2.0, 0.0], [0.0, 2.0]], Some(0.05));
        assert!(scaled.valid, "{:?}", scaled.test);
        assert_eq!(
            scaled.test.orientation,
            PlanarTransferOrientation::Preserving
        );
        assert!((scaled.test.scale - 2.0).abs() < 1e-6);
        let turned = gate_from_operator(edge, rotation(0.3, 1.1), Some(0.05));
        assert!(turned.valid, "{:?}", turned.test);
        assert_eq!(
            turned.test.orientation,
            PlanarTransferOrientation::Preserving
        );
        let flipped = gate_from_operator(edge, reflection(0.7, 0.4), Some(0.05));
        assert!(flipped.valid, "{:?}", flipped.test);
        assert_eq!(
            flipped.test.orientation,
            PlanarTransferOrientation::Reversing
        );
        assert!((flipped.test.scale - 0.7).abs() < 1e-6);
    }

    /// Identical f32 codes carry only their own quantization noise: the
    /// covariance is the input resolution, not 0/0, and the gate is valid.
    #[test]
    fn identical_codes_are_a_valid_identity_transfer() {
        let x = circle_codes(8);
        let test = PlanarTransferTest::from_codes(&x, &x).unwrap();
        assert!(test.conformal_p_value > 0.5, "{test:?}");
        assert!(test.existence_p_value < 1e-12, "{test:?}");
        assert!(test.conformal_defect <= 4.0 * test.conformal_defect_se);
        let gate = AtlasTransferGate::from_test(
            AtlasHolonomyEdgeId::new(0, 1, 0).unwrap(),
            test,
            Some(0.05),
        );
        assert!(gate.valid);
        let uncertified = gate_from_operator(
            AtlasHolonomyEdgeId::new(0, 1, 0).unwrap(),
            [[1.0, 0.0], [0.0, 1.0]],
            None,
        );
        assert!(
            !uncertified.valid,
            "without a level a gate certifies nothing"
        );
        assert!(PlanarTransferTest::from_codes(&x[..2], &x[..2]).is_err());
    }

    #[test]
    fn conformality_test_is_calibrated_under_rotation_and_reflection_nulls() {
        let mut rng = StdRng::seed_from_u64(3921);
        for (label, operator, orientation) in [
            (
                "rotation",
                rotation(0.7, 1.1),
                PlanarTransferOrientation::Preserving,
            ),
            (
                "reflection",
                reflection(0.7, 1.1),
                PlanarTransferOrientation::Reversing,
            ),
        ] {
            let mut p_values = Vec::new();
            for _ in 0..1500 {
                let (x, y) = noisy_pair(&mut rng, 200, operator, 0.1);
                let test = PlanarTransferTest::from_codes(&x, &y).unwrap();
                assert_eq!(test.orientation, orientation);
                p_values.push(test.conformal_p_value);
            }
            assert_calibrated(&p_values, label);
        }
    }

    #[test]
    fn existence_test_is_calibrated_when_no_transfer_exists() {
        let mut rng = StdRng::seed_from_u64(39210);
        let p_values: Vec<f64> = (0..1500)
            .map(|_| {
                let (x, y) = noisy_pair(&mut rng, 200, [[0.0, 0.0], [0.0, 0.0]], 0.1);
                PlanarTransferTest::from_codes(&x, &y)
                    .unwrap()
                    .existence_p_value
            })
            .collect();
        assert_calibrated(&p_values, "existence");
    }

    #[test]
    fn a_shear_transfer_is_refused() {
        let mut rng = StdRng::seed_from_u64(39211);
        let edge = AtlasHolonomyEdgeId::new(0, 1, 0).unwrap();
        for _ in 0..50 {
            let (x, y) = noisy_pair(&mut rng, 200, [[1.0, 0.3], [0.0, 1.0]], 0.1);
            let test = PlanarTransferTest::from_codes(&x, &y).unwrap();
            assert!(test.conformal_p_value < 1e-6, "{test:?}");
            assert!(!AtlasTransferGate::from_test(edge, test, Some(0.01)).valid);
        }
    }
}
