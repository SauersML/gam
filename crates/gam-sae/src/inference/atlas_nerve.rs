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

use crate::chart_transfer::TransferCertificate;
use crate::inference::layer_transport::FittedTransport;
use crate::manifold::{BettiSignature, GraphCompressionKind, GraphCompressionReport};
use crate::null_battery::ClaimNullCalibration;
use std::cmp::Reverse;
use std::collections::{BTreeMap, BTreeSet, BinaryHeap, HashMap};

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

    pub fn support_weights(&self) -> &[f64] {
        &self.support_weights
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

/// A structural proof that one non-empty chart intersection is contractible.
///
/// The proof says that the complete intersection named by `charts` is a convex
/// subset of `witness_chart`'s fitted local coordinate domain.  Convexity gives
/// an explicit straight-line contraction, so this is a theorem witness rather
/// than a sample-count heuristic.  The atlas builder that owns the local patch
/// geometry is responsible for emitting these proofs; the nerve verifies that
/// there is exactly one proof for every non-empty finite intersection it finds.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ConvexIntersectionProof {
    charts: Vec<usize>,
    witness_chart: usize,
}

impl ConvexIntersectionProof {
    #[must_use = "intersection proof construction errors must be handled"]
    pub fn new(charts: Vec<usize>, witness_chart: usize) -> Result<Self, String> {
        if charts.is_empty() {
            return Err("a contractible-intersection proof cannot name an empty simplex".into());
        }
        if charts.windows(2).any(|pair| pair[0] >= pair[1]) {
            return Err(
                "contractible-intersection chart indices must be strictly increasing".into(),
            );
        }
        if charts.binary_search(&witness_chart).is_err() {
            return Err(format!(
                "convexity witness chart {witness_chart} is not in intersection {charts:?}"
            ));
        }
        Ok(Self {
            charts,
            witness_chart,
        })
    }

    #[must_use]
    pub fn charts(&self) -> &[usize] {
        &self.charts
    }

    #[must_use]
    pub fn witness_chart(&self) -> usize {
        self.witness_chart
    }
}

/// Explicit good-cover proof for one fixed atlas.
///
/// A good cover requires *every* non-empty finite intersection to be
/// contractible.  Consequently this object is deliberately exhaustive: its
/// proof keys must equal the full nerve, not merely its graph or its
/// intersections through some chosen dimension.  [`build_atlas_nerve`] rejects
/// both missing and surplus keys.  Merely observing at least as many rows as
/// charts can never construct this certificate.
#[derive(Clone, Debug)]
pub struct AtlasGoodCoverCertificate {
    chart_count: usize,
    proofs: BTreeMap<Vec<usize>, ConvexIntersectionProof>,
}

impl AtlasGoodCoverCertificate {
    #[must_use = "good-cover certificate construction errors must be handled"]
    pub fn new(
        chart_count: usize,
        proofs: Vec<ConvexIntersectionProof>,
    ) -> Result<Self, String> {
        let mut indexed = BTreeMap::new();
        for proof in proofs {
            if proof.charts.iter().any(|&chart| chart >= chart_count) {
                return Err(format!(
                    "good-cover proof {:?} is outside the {chart_count}-chart atlas",
                    proof.charts
                ));
            }
            let key = proof.charts.clone();
            if indexed.insert(key.clone(), proof).is_some() {
                return Err(format!(
                    "duplicate contractibility proof for chart intersection {key:?}"
                ));
            }
        }
        Ok(Self {
            chart_count,
            proofs: indexed,
        })
    }
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
    /// Whether every non-empty intersection was matched to an explicit
    /// contractibility proof.
    pub good_cover_certified: bool,
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
        let generic = self
            .simplex_counts
            .iter()
            .enumerate()
            .map(|(dimension, &present)| {
                simplex_selection_bits(self.n_vertices, dimension + 1, present)
            })
            .sum();
        if !self.good_cover_certified {
            return GraphCompressionReport::unnamed(generic);
        }
        let log_vertices = (self.n_vertices.max(2) as f64).log2();
        let named = match (self.betti.b0, self.betti.b1, self.betti.b2) {
            (1, 2, Some(1)) => Some((GraphCompressionKind::Torus, "torus", 2.0 * log_vertices)),
            (1, 1, Some(0)) => Some((
                GraphCompressionKind::Cylinder,
                "cylinder",
                2.0 * log_vertices,
            )),
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

fn binomial_usize(n: usize, k: usize) -> Option<usize> {
    if k > n {
        return Some(0);
    }
    let k = k.min(n - k);
    let mut value = 1usize;
    for divisor in 1..=k {
        value = value.checked_mul(n - k + divisor)? / divisor;
    }
    Some(value)
}

fn simplex_selection_bits(vertices: usize, cardinality: usize, present: usize) -> f64 {
    let Some(slots) = binomial_usize(vertices, cardinality) else {
        return f64::INFINITY;
    };
    if present > slots {
        return f64::INFINITY;
    }
    let Ok(slots_i64) = i64::try_from(slots) else {
        return f64::INFINITY;
    };
    let selected = present.min(slots - present);
    let Ok(selected_i64) = i64::try_from(selected) else {
        return f64::INFINITY;
    };
    crate::description_length::selection_bits(slots_i64, selected_i64)
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

fn gate_key(a: usize, b: usize) -> (usize, usize) {
    if a < b { (a, b) } else { (b, a) }
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

#[derive(Debug)]
struct SimplexInventory {
    counts: Vec<usize>,
    vertices: Vec<Vec<usize>>,
    edges: Vec<Vec<usize>>,
    triangles: Vec<Vec<usize>>,
    tetrahedra: Vec<Vec<usize>>,
    euler_characteristic: i128,
}

fn record_simplex(
    simplex: &[usize],
    inventory: &mut SimplexInventory,
    unmatched_proofs: &mut Option<BTreeSet<Vec<usize>>>,
) -> Result<(), String> {
    let cardinality = simplex.len();
    inventory.counts[cardinality - 1] = inventory.counts[cardinality - 1]
        .checked_add(1)
        .ok_or_else(|| "atlas nerve simplex count overflowed usize".to_string())?;
    let count = i128::try_from(inventory.counts[cardinality - 1])
        .map_err(|_| "atlas nerve simplex count overflowed i128".to_string())?;
    let previous = count - 1;
    inventory.euler_characteristic = if cardinality % 2 == 1 {
        inventory
            .euler_characteristic
            .checked_add(count - previous)
    } else {
        inventory
            .euler_characteristic
            .checked_sub(count - previous)
    }
    .ok_or_else(|| "atlas nerve Euler characteristic overflowed i128".to_string())?;

    if let Some(unmatched) = unmatched_proofs {
        if !unmatched.remove(simplex) {
            return Err(format!(
                "good-cover certificate has no contractibility proof for non-empty intersection {simplex:?}"
            ));
        }
    }
    match cardinality {
        1 => inventory.vertices.push(simplex.to_vec()),
        2 => inventory.edges.push(simplex.to_vec()),
        3 => inventory.triangles.push(simplex.to_vec()),
        4 => inventory.tetrahedra.push(simplex.to_vec()),
        _ => {}
    }
    Ok(())
}

fn enumerate_simplices_from(
    charts: &[AtlasChart],
    adjacency: &[BTreeSet<usize>],
    prefix: &mut Vec<usize>,
    candidates: &[usize],
    inventory: &mut SimplexInventory,
    unmatched_proofs: &mut Option<BTreeSet<Vec<usize>>>,
) -> Result<(), String> {
    for (position, &vertex) in candidates.iter().enumerate() {
        prefix.push(vertex);
        let nonempty = prefix.len() == 1 || {
            let (mass, _) = mutual_row_mass(charts, prefix);
            mass.is_finite() && mass > 0.0
        };
        if nonempty {
            record_simplex(prefix, inventory, unmatched_proofs)?;
            let next: Vec<usize> = candidates[(position + 1)..]
                .iter()
                .copied()
                .filter(|candidate| adjacency[vertex].contains(candidate))
                .collect();
            enumerate_simplices_from(
                charts,
                adjacency,
                prefix,
                &next,
                inventory,
                unmatched_proofs,
            )?;
        }
        prefix.pop();
    }
    Ok(())
}

fn enumerate_full_nerve(
    charts: &[AtlasChart],
    adjacency: &[BTreeSet<usize>],
    certificate: Option<&AtlasGoodCoverCertificate>,
) -> Result<SimplexInventory, String> {
    if let Some(certificate) = certificate {
        if certificate.chart_count != charts.len() {
            return Err(format!(
                "good-cover certificate is for {} charts but the atlas has {}",
                certificate.chart_count,
                charts.len()
            ));
        }
    }
    let mut unmatched_proofs = certificate
        .map(|certificate| certificate.proofs.keys().cloned().collect::<BTreeSet<_>>());
    let mut inventory = SimplexInventory {
        counts: vec![0; charts.len()],
        vertices: Vec::with_capacity(charts.len()),
        edges: Vec::new(),
        triangles: Vec::new(),
        tetrahedra: Vec::new(),
        euler_characteristic: 0,
    };
    let candidates: Vec<usize> = (0..charts.len()).collect();
    enumerate_simplices_from(
        charts,
        adjacency,
        &mut Vec::new(),
        &candidates,
        &mut inventory,
        &mut unmatched_proofs,
    )?;
    if let Some(unmatched) = unmatched_proofs {
        if !unmatched.is_empty() {
            return Err(format!(
                "good-cover certificate names intersections absent from the nerve: {unmatched:?}"
            ));
        }
    }
    Ok(inventory)
}

/// Build the certificate-gated atlas nerve and read Betti numbers through H2.
#[must_use = "atlas nerve construction errors must be handled"]
pub fn build_atlas_nerve(
    charts: &[AtlasChart],
    transfer_gates: &[AtlasTransferGate],
    good_cover: Option<&AtlasGoodCoverCertificate>,
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
            simplex_counts: Vec::new(),
            euler_characteristic: 0,
            good_cover_certified: good_cover.is_some(),
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
        if gate_map.insert(gate_key(gate.a, gate.b), gate).is_some() {
            return Err(format!(
                "duplicate transfer gate for atlas pair ({}, {})",
                gate.a.min(gate.b),
                gate.a.max(gate.b)
            ));
        }
    }

    let (overlaps, sampled) = sparse_pair_overlaps(charts);
    let dtm = dtm_radii(charts, &overlaps);

    let vertices: Vec<Vec<usize>> = (0..n).map(|i| vec![i]).collect();
    let mut adjacency = vec![BTreeSet::<usize>::new(); n];
    let mut edge_filtration = BTreeMap::<(usize, usize), f64>::new();
    let mut edge_reports = Vec::new();
    let mut edge_simplices = Vec::new();
    let mut max_filtration = 0.0_f64;
    for (&(a, b), overlap) in &overlaps {
        let threshold = overlap.mass / overlap.positive_rows as f64;
        let transfer_valid = gate_map.get(&gate_key(a, b)).is_some_and(|gate| gate.valid);
        let coactive = overlap.mass.is_finite() && overlap.mass >= threshold;
        let filtration = chart_distance(&charts[a], &charts[b], overlap.mass)
            .max(dtm[a])
            .max(dtm[b]);
        let admitted = coactive && transfer_valid;
        if admitted {
            adjacency[a].insert(b);
            adjacency[b].insert(a);
            edge_filtration.insert((a, b), filtration);
            edge_simplices.push(vec![a, b]);
            max_filtration = max_filtration.max(filtration);
        }
        edge_reports.push(AtlasNerveEdge {
            a,
            b,
            coactivation_mass: overlap.mass,
            coactivation_threshold: threshold,
            transfer_valid,
            admitted,
            filtration,
        });
    }

    let mut triangles = Vec::new();
    let mut triangle_set = BTreeSet::new();
    for a in 0..n {
        for &b in adjacency[a].iter().filter(|&&candidate| candidate > a) {
            for &c in adjacency[a]
                .intersection(&adjacency[b])
                .filter(|&&candidate| candidate > b)
            {
                let simplex = [a, b, c];
                let (coactivation, threshold) = mutual_row_mass(charts, &simplex);
                if coactivation.is_finite() && coactivation >= threshold {
                    let tri = vec![a, b, c];
                    triangle_set.insert(tri.clone());
                    triangles.push(tri);
                    let filt = edge_filtration[&(a, b)]
                        .max(edge_filtration[&(a, c)])
                        .max(edge_filtration[&(b, c)]);
                    max_filtration = max_filtration.max(filt);
                }
            }
        }
    }

    let mut tetrahedra = Vec::new();
    for triangle in &triangles {
        let [a, b, c] = [triangle[0], triangle[1], triangle[2]];
        for &d in adjacency[a].iter().filter(|&&candidate| candidate > c) {
            let faces = [vec![a, b, c], vec![a, b, d], vec![a, c, d], vec![b, c, d]];
            if faces.iter().all(|face| triangle_set.contains(face)) {
                let simplex = [a, b, c, d];
                let (coactivation, threshold) = mutual_row_mass(charts, &simplex);
                if coactivation.is_finite() && coactivation >= threshold {
                    tetrahedra.push(vec![a, b, c, d]);
                }
            }
        }
    }

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
    use super::{AtlasChart, AtlasCoveringSide, AtlasTransferGate, build_atlas_nerve};
    use crate::chart_transfer::certify_square_transfer;
    use crate::manifold::GraphCompressionKind;
    use ndarray::{Array2, arr2};
    use std::collections::BTreeSet;

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

    #[test]
    fn chart_storage_tracks_sparse_support_not_row_count() {
        let chart =
            AtlasChart::from_sparse_weights(0, 1_000_000, vec![7, 900_001], vec![0.25, 0.75])
                .unwrap();
        assert_eq!(chart.row_count(), 1_000_000);
        assert_eq!(chart.support_rows(), &[7, 900_001]);
        assert_eq!(chart.support_weights(), &[0.25, 0.75]);
        assert_eq!(chart.support_mass, 1.0);
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
        assert_eq!(
            diagram.covering_side,
            AtlasCoveringSide::AtOrAboveCoveringNumber
        );
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
        assert_eq!(
            diagram.covering_side,
            AtlasCoveringSide::AtOrAboveCoveringNumber
        );
        let compression = diagram.certified_compression();
        assert_eq!(compression.kind, GraphCompressionKind::Torus);
        assert_eq!(compression.name, "torus");
        assert!(compression.bits_saved > 0.0);
    }
}
