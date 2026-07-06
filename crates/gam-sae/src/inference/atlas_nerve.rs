//! Certificate-gated atlas nerves for block/chart dictionaries.
//!
//! Single chart atoms expose exact within-chart graph readouts, but they cannot
//! see cross-chart topology such as an atlas covering a sphere or torus. This
//! module builds the nerve over dictionary charts: a chart is a vertex, a pair
//! becomes an edge only when the supports co-activate and the chart-transfer
//! certificate is valid, and higher simplices are admitted from mutual
//! co-activation. The resulting filtered clique complex is read by exact GF(2)
//! homology through H2.

use crate::assignment::SupportMeasure;
use crate::chart_transfer::TransferCertificate;
use crate::inference::layer_transport::FittedTransport;
use crate::manifold::{BettiSignature, GraphCompressionKind, GraphCompressionReport};
use crate::null_battery::ClaimNullCalibration;
use ndarray::Array1;
use std::collections::{BTreeSet, HashMap};

/// Which side of the chart-covering sample count the diagram sits on.
///
/// Nerve invariance is a covering theorem: once the sampled support resolves at
/// least the chart cover, refining samples should not change the cover nerve.
/// Below that count, a diagram is still a measurement, but it is marked as
/// under-resolved.
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
    pub row_weights: Array1<f64>,
    pub support_mass: f64,
    pub support_ess: f64,
}

impl AtlasChart {
    #[must_use = "atlas chart construction errors must be handled"]
    pub fn from_support(support: SupportMeasure) -> Result<Self, String> {
        let weights = support.weights().to_owned();
        Self::from_weights(support.atom_idx(), weights)
    }

    #[must_use = "atlas chart construction errors must be handled"]
    pub fn from_weights(chart_idx: usize, row_weights: Array1<f64>) -> Result<Self, String> {
        let support = SupportMeasure::from_weights(chart_idx, row_weights.clone())?;
        Ok(Self {
            chart_idx,
            row_weights,
            support_mass: support.mass(),
            support_ess: support.ess(),
        })
    }
}

/// Existing transfer evidence for a chart pair, compressed to the validity
/// verdict the nerve gate needs.
#[derive(Clone, Debug)]
pub struct AtlasTransferGate {
    pub a: usize,
    pub b: usize,
    pub valid: bool,
    pub transport_defect: f64,
    pub equivariance_defect: f64,
}

impl AtlasTransferGate {
    pub fn from_square_transfer(
        a: usize,
        b: usize,
        certificate: TransferCertificate,
        chart_dim: usize,
    ) -> Self {
        let scale = (chart_dim.max(1) as f64).sqrt() * f64::EPSILON;
        let valid = certificate.transport_defect.is_finite()
            && certificate.equivariance_defect.is_finite()
            && certificate.transport_defect <= scale
            && certificate.equivariance_defect <= scale;
        Self {
            a,
            b,
            valid,
            transport_defect: certificate.transport_defect,
            equivariance_defect: certificate.equivariance_defect,
        }
    }

    pub fn from_fitted_transport(a: usize, b: usize, transport: &FittedTransport) -> Self {
        let valid = transport.topology_preserved
            && transport.isometry_defect.is_finite()
            && transport.isometry_defect_se.is_finite()
            && transport.isometry_defect <= transport.isometry_defect_se;
        Self {
            a,
            b,
            valid,
            transport_defect: transport.isometry_defect,
            equivariance_defect: transport.residual_rms,
        }
    }
}

/// One admitted or rejected pair after both gates are evaluated.
#[derive(Clone, Debug)]
pub struct AtlasNerveEdge {
    pub a: usize,
    pub b: usize,
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
    pub sampled_support_size: usize,
    pub covering_side: AtlasCoveringSide,
    pub max_filtration: f64,
    pub edges: Vec<AtlasNerveEdge>,
    pub note: String,
}

impl AtlasNerveDiagram {
    /// Certified named compression of the atlas nerve. The name is a codebook
    /// compression of the exact nerve homology, never an input topology choice.
    pub fn certified_compression(&self) -> GraphCompressionReport {
        let max_edges = self
            .n_vertices
            .saturating_mul(self.n_vertices.saturating_sub(1))
            / 2;
        let max_triangles = if self.n_vertices >= 3 {
            self.n_vertices * (self.n_vertices - 1) * (self.n_vertices - 2) / 6
        } else {
            0
        };
        let max_tetrahedra = if self.n_vertices >= 4 {
            self.n_vertices
                * (self.n_vertices - 1)
                * (self.n_vertices - 2)
                * (self.n_vertices - 3)
                / 24
        } else {
            0
        };
        let generic = crate::description_length::selection_bits(
            max_edges as i64,
            self.n_edges as i64,
        )
            + crate::description_length::selection_bits(
                max_triangles as i64,
                self.n_triangles as i64,
            )
            + crate::description_length::selection_bits(
                max_tetrahedra as i64,
                self.n_tetrahedra as i64,
            );
        let log_vertices = (self.n_vertices.max(2) as f64).log2();
        let named = match (self.betti.b0, self.betti.b1, self.betti.b2) {
            (1, 2, Some(1)) => {
                Some((GraphCompressionKind::Torus, "torus", 2.0 * log_vertices))
            }
            (1, 1, Some(0)) => {
                Some((GraphCompressionKind::Cylinder, "cylinder", 2.0 * log_vertices))
            }
            (1, 0, Some(1)) => Some((GraphCompressionKind::Sphere, "sphere", log_vertices)),
            _ => None,
        };
        if let Some((kind, name, named_bits)) = named {
            let report = GraphCompressionReport::certified(kind, name, generic, named_bits);
            if report.bits_saved > 0.0 {
                return report;
            }
        }
        GraphCompressionReport::unnamed(generic)
    }
}

fn validate_charts(charts: &[AtlasChart]) -> Result<usize, String> {
    let Some(first) = charts.first() else {
        return Ok(0);
    };
    let n = first.row_weights.len();
    for (pos, chart) in charts.iter().enumerate() {
        if chart.chart_idx != pos {
            return Err(format!(
                "atlas chart indices must be contiguous; chart at position {pos} has index {}",
                chart.chart_idx
            ));
        }
        if chart.row_weights.len() != n {
            return Err(format!(
                "atlas chart {} has {} rows but expected {n}",
                chart.chart_idx,
                chart.row_weights.len()
            ));
        }
        if chart
            .row_weights
            .iter()
            .any(|&w| !(w.is_finite() && w >= 0.0))
        {
            return Err(format!(
                "atlas chart {} has non-finite or negative row weights",
                chart.chart_idx
            ));
        }
    }
    Ok(n)
}

fn gate_key(a: usize, b: usize) -> (usize, usize) {
    if a < b { (a, b) } else { (b, a) }
}

fn mutual_row_mass(charts: &[AtlasChart], simplex: &[usize]) -> (f64, f64) {
    let n = charts[0].row_weights.len();
    let mut total = 0.0_f64;
    let mut positive_count = 0usize;
    for row in 0..n {
        let mut row_mass = f64::INFINITY;
        for &chart_idx in simplex {
            row_mass = row_mass.min(charts[chart_idx].row_weights[row]);
        }
        if row_mass.is_finite() && row_mass > 0.0 {
            total += row_mass;
            positive_count += 1;
        }
    }
    let threshold = if positive_count > 0 {
        total / positive_count as f64
    } else {
        f64::INFINITY
    };
    (total, threshold)
}

fn sampled_support_size(charts: &[AtlasChart]) -> usize {
    if charts.is_empty() {
        return 0;
    }
    let n = charts[0].row_weights.len();
    let mut count = 0usize;
    for row in 0..n {
        if charts.iter().any(|chart| chart.row_weights[row] > 0.0) {
            count += 1;
        }
    }
    count
}

fn chart_distance(a: &AtlasChart, b: &AtlasChart) -> f64 {
    let mut overlap = 0.0_f64;
    for row in 0..a.row_weights.len() {
        overlap += a.row_weights[row].min(b.row_weights[row]);
    }
    let denom = a.support_mass.min(b.support_mass);
    if denom > 0.0 {
        (1.0 - overlap / denom).clamp(0.0, 1.0)
    } else {
        1.0
    }
}

fn dtm_radii(charts: &[AtlasChart], distances: &[Vec<f64>]) -> Vec<f64> {
    let n = charts.len();
    if n <= 1 {
        return vec![0.0; n];
    }
    let total_mass = charts.iter().map(|chart| chart.support_mass).sum::<f64>();
    if !(total_mass.is_finite() && total_mass > 0.0) {
        return vec![0.0; n];
    }
    let target_mass = total_mass / n as f64;
    let mut radii = vec![0.0_f64; n];
    for i in 0..n {
        let mut neighbors = Vec::with_capacity(n - 1);
        for j in 0..n {
            if i != j && charts[j].support_mass > 0.0 {
                neighbors.push((distances[i][j], charts[j].support_mass));
            }
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

fn compute_betti(
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

/// Build the certificate-gated atlas nerve and read Betti numbers through H2.
#[must_use = "atlas nerve construction errors must be handled"]
pub fn build_atlas_nerve(
    charts: &[AtlasChart],
    transfer_gates: &[AtlasTransferGate],
) -> Result<AtlasNerveDiagram, String> {
    let row_count = validate_charts(charts)?;
    let n = charts.len();
    if n == 0 {
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
            sampled_support_size: 0,
            covering_side: AtlasCoveringSide::BelowCoveringNumber,
            max_filtration: 0.0,
            edges: Vec::new(),
            note: "empty atlas nerve".to_string(),
        });
    }

    let mut gate_map = HashMap::with_capacity(transfer_gates.len());
    for gate in transfer_gates {
        if gate.a >= n || gate.b >= n || gate.a == gate.b {
            return Err(format!(
                "transfer gate ({}, {}) is outside the {n}-chart atlas or self-linked",
                gate.a, gate.b
            ));
        }
        gate_map.insert(gate_key(gate.a, gate.b), gate);
    }

    let mut distances = vec![vec![0.0_f64; n]; n];
    for i in 0..n {
        for j in (i + 1)..n {
            let d = chart_distance(&charts[i], &charts[j]);
            distances[i][j] = d;
            distances[j][i] = d;
        }
    }
    let dtm = dtm_radii(charts, &distances);

    let vertices: Vec<Vec<usize>> = (0..n).map(|i| vec![i]).collect();
    let mut edge_set = vec![vec![false; n]; n];
    let mut edge_filtration = vec![vec![0.0_f64; n]; n];
    let mut edge_reports = Vec::new();
    let mut edge_simplices = Vec::new();
    let mut max_filtration = 0.0_f64;
    for a in 0..n {
        for b in (a + 1)..n {
            let pair = [a, b];
            let (coactivation, threshold) = mutual_row_mass(charts, &pair);
            let transfer_valid = gate_map
                .get(&gate_key(a, b))
                .is_some_and(|gate| gate.valid);
            let coactive = coactivation.is_finite() && coactivation >= threshold;
            let filtration = distances[a][b].max(dtm[a]).max(dtm[b]);
            let admitted = coactive && transfer_valid;
            if admitted {
                edge_set[a][b] = true;
                edge_set[b][a] = true;
                edge_filtration[a][b] = filtration;
                edge_filtration[b][a] = filtration;
                edge_simplices.push(vec![a, b]);
                max_filtration = max_filtration.max(filtration);
            }
            edge_reports.push(AtlasNerveEdge {
                a,
                b,
                coactivation_mass: coactivation,
                coactivation_threshold: threshold,
                transfer_valid,
                admitted,
                filtration,
            });
        }
    }

    let mut triangles = Vec::new();
    let mut triangle_set = BTreeSet::new();
    for a in 0..n {
        for b in (a + 1)..n {
            for c in (b + 1)..n {
                if edge_set[a][b] && edge_set[a][c] && edge_set[b][c] {
                    let simplex = [a, b, c];
                    let (coactivation, threshold) = mutual_row_mass(charts, &simplex);
                    if coactivation.is_finite() && coactivation >= threshold {
                        let tri = vec![a, b, c];
                        triangle_set.insert(tri.clone());
                        triangles.push(tri);
                        let filt = edge_filtration[a][b]
                            .max(edge_filtration[a][c])
                            .max(edge_filtration[b][c]);
                        max_filtration = max_filtration.max(filt);
                    }
                }
            }
        }
    }

    let mut tetrahedra = Vec::new();
    for a in 0..n {
        for b in (a + 1)..n {
            for c in (b + 1)..n {
                for d in (c + 1)..n {
                    let faces = [
                        vec![a, b, c],
                        vec![a, b, d],
                        vec![a, c, d],
                        vec![b, c, d],
                    ];
                    if faces.iter().all(|face| triangle_set.contains(face)) {
                        let simplex = [a, b, c, d];
                        let (coactivation, threshold) = mutual_row_mass(charts, &simplex);
                        if coactivation.is_finite() && coactivation >= threshold {
                            tetrahedra.push(vec![a, b, c, d]);
                        }
                    }
                }
            }
        }
    }

    let sampled = sampled_support_size(charts);
    let covering_side = if sampled >= n {
        AtlasCoveringSide::AtOrAboveCoveringNumber
    } else {
        AtlasCoveringSide::BelowCoveringNumber
    };
    let betti = compute_betti(&vertices, &edge_simplices, &triangles, &tetrahedra);
    let note = format!(
        "atlas nerve over {n} charts and {row_count} rows: sampled_support_size={sampled}, covering_side={}, Betti=({}, {}, {:?})",
        covering_side.as_str(),
        betti.b0,
        betti.b1,
        betti.b2
    );

    Ok(AtlasNerveDiagram {
        betti,
        null_calibration: None,
        n_vertices: vertices.len(),
        n_edges: edge_simplices.len(),
        n_triangles: triangles.len(),
        n_tetrahedra: tetrahedra.len(),
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
        AtlasChart, AtlasCoveringSide, AtlasTransferGate, build_atlas_nerve,
    };
    use crate::assignment::SupportMeasure;
    use crate::chart_transfer::certify_square_transfer;
    use crate::manifold::GraphCompressionKind;
    use ndarray::{Array2, arr2};
    use std::collections::BTreeSet;

    fn charts_from_faces(n_charts: usize, faces: &[Vec<usize>]) -> Vec<AtlasChart> {
        let mut assignments = Array2::<f64>::zeros((faces.len(), n_charts));
        for (row, face) in faces.iter().enumerate() {
            for &chart in face {
                assignments[[row, chart]] = 1.0;
            }
        }
        (0..n_charts)
            .map(|chart| {
                let support =
                    SupportMeasure::from_assignment_matrix(assignments.view(), chart).unwrap();
                AtlasChart::from_support(support).unwrap()
            })
            .collect()
    }

    fn all_valid_pair_gates(n_charts: usize) -> Vec<AtlasTransferGate> {
        let generator = Array2::<f64>::zeros((2, 2));
        let identity = arr2(&[[1.0, 0.0], [0.0, 1.0]]);
        let mut gates = Vec::new();
        for a in 0..n_charts {
            for b in (a + 1)..n_charts {
                let cert =
                    certify_square_transfer(identity.view(), generator.view(), generator.view())
                        .unwrap();
                gates.push(AtlasTransferGate::from_square_transfer(a, b, cert, 2));
            }
        }
        gates
    }

    #[test]
    fn charts_tiling_synthetic_sphere_have_h2() {
        let faces = vec![vec![0, 1, 2], vec![0, 1, 3], vec![0, 2, 3], vec![1, 2, 3]];
        let charts = charts_from_faces(4, &faces);
        let gates = all_valid_pair_gates(4);
        let diagram = build_atlas_nerve(&charts, &gates).unwrap();
        assert_eq!(diagram.betti.b0, 1);
        assert_eq!(diagram.betti.b1, 0);
        assert_eq!(diagram.betti.b2, Some(1));
        assert_eq!(diagram.n_triangles, 4);
        assert_eq!(diagram.n_tetrahedra, 0);
        assert_eq!(diagram.covering_side, AtlasCoveringSide::AtOrAboveCoveringNumber);
    }

    #[test]
    fn inconsistent_transfer_rejects_cross_cluster_edge() {
        let faces = vec![vec![0, 1], vec![1, 2], vec![2, 3]];
        let charts = charts_from_faces(4, &faces);
        let generator = Array2::<f64>::zeros((2, 2));
        let identity = arr2(&[[1.0, 0.0], [0.0, 1.0]]);
        let scaled = arr2(&[[2.0, 0.0], [0.0, 2.0]]);
        let valid_cert =
            certify_square_transfer(identity.view(), generator.view(), generator.view()).unwrap();
        let invalid_cert =
            certify_square_transfer(scaled.view(), generator.view(), generator.view()).unwrap();
        let gates = vec![
            AtlasTransferGate::from_square_transfer(0, 1, valid_cert, 2),
            AtlasTransferGate::from_square_transfer(1, 2, invalid_cert, 2),
            AtlasTransferGate::from_square_transfer(2, 3, valid_cert, 2),
        ];
        let diagram = build_atlas_nerve(&charts, &gates).unwrap();
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

    #[test]
    fn charts_tiling_synthetic_torus_have_two_h1_cycles() {
        let side = 4usize;
        let vertex = |i: usize, j: usize| -> usize { (i % side) * side + (j % side) };
        let mut faces = BTreeSet::new();
        for i in 0..side {
            for j in 0..side {
                let a = vertex(i, j);
                let b = vertex(i + 1, j);
                let c = vertex(i + 1, j + 1);
                let d = vertex(i, j + 1);
                let mut first = vec![a, b, c];
                first.sort_unstable();
                faces.insert(first);
                let mut second = vec![a, c, d];
                second.sort_unstable();
                faces.insert(second);
            }
        }
        let faces: Vec<Vec<usize>> = faces.into_iter().collect();
        let charts = charts_from_faces(side * side, &faces);
        let gates = all_valid_pair_gates(side * side);
        let diagram = build_atlas_nerve(&charts, &gates).unwrap();
        assert_eq!(diagram.betti.b0, 1);
        assert_eq!(diagram.betti.b1, 2);
        assert_eq!(diagram.betti.b2, Some(1));
        assert_eq!(diagram.covering_side, AtlasCoveringSide::AtOrAboveCoveringNumber);
        let compression = diagram.certified_compression();
        assert_eq!(compression.kind, GraphCompressionKind::Torus);
        assert_eq!(compression.name, "torus");
        assert!(compression.bits_saved > 0.0);
    }
}
