// PyO3 boundary for the SAE spectral / routing-geometry diagnostics landed in
// the `gam-sae` crate (`gam::terms::sae`): the dimension spectrometer, the
// block-firing circle-coordinate readout, the routability floor + audit, the
// sparse-dictionary dual certificate, and the contract-composition / loop
// holonomy calculus.
//
// NB: plain `//` comments, NOT `//!` inner-doc — this file is an `include!`
// fragment textually inlined into `lib.rs` AFTER other items, where an inner
// doc comment is a hard error (E0753: inner attributes/doc must lead the
// enclosing module). Every sibling entrypoint fragment (`model_ffi.rs`, …)
// leads with plain comments for the same reason.
//
// The fragment shares the crate-root namespace with the other entrypoint
// fragments and reaches the shared prelude items (`PyDict`, `PyReadonlyArray2`,
// `detach_py_result`, …) by bare name. As with the other fragments, the engine
// functions are referenced through their fully-qualified `gam::terms::sae::…`
// paths and the Python surface is a thin wrapper (SPEC rule 8): every number is
// computed in Rust, the FFI only marshals arrays and dicts.

#[derive(Clone)]
struct AuditSparseRoute {
    indices: ndarray::Array2<u32>,
    values: ndarray::Array3<f32>,
    n_units: usize,
    block_size: usize,
}

impl AuditSparseRoute {
    fn new(
        indices: ndarray::Array2<u32>,
        values: ndarray::Array3<f32>,
        n_units: usize,
        block_size: usize,
        label: &str,
    ) -> Result<Self, String> {
        if block_size == 0 {
            return Err("audit_sae block_size must be >= 1".to_string());
        }
        if n_units == 0 {
            return Err(format!(
                "audit_sae {label} requires at least one routing unit"
            ));
        }
        let (n_rows, width) = indices.dim();
        if n_rows == 0 || width == 0 {
            return Err(format!(
                "audit_sae {label} must be a non-empty N×s route; got {:?}",
                indices.dim()
            ));
        }
        if values.shape() != [n_rows, width, block_size] {
            return Err(format!(
                "audit_sae {label} values shape {:?} does not match indices {:?} and block_size {block_size}",
                values.shape(),
                indices.dim()
            ));
        }
        for row in 0..n_rows {
            let mut live = std::collections::HashSet::with_capacity(width);
            for slot in 0..width {
                let unit = indices[[row, slot]] as usize;
                if unit >= n_units {
                    return Err(format!(
                        "audit_sae {label} index {unit} at row {row}, slot {slot} is outside 0..{n_units}"
                    ));
                }
                let mut norm2 = 0.0_f64;
                for offset in 0..block_size {
                    let value = values[[row, slot, offset]] as f64;
                    if !value.is_finite() {
                        return Err(format!(
                            "audit_sae {label} value at row {row}, slot {slot}, offset {offset} is not finite"
                        ));
                    }
                    norm2 += value * value;
                }
                if norm2 > 0.0 && !live.insert(unit) {
                    return Err(format!(
                        "audit_sae {label} repeats live unit {unit} in row {row}"
                    ));
                }
            }
        }
        Ok(Self {
            indices,
            values,
            n_units,
            block_size,
        })
    }

    fn nrows(&self) -> usize {
        self.indices.nrows()
    }

    fn width(&self) -> usize {
        self.indices.ncols()
    }

    fn gate(&self, row: usize, slot: usize) -> f64 {
        let mut norm2 = 0.0_f64;
        for offset in 0..self.block_size {
            let value = self.values[[row, slot, offset]] as f64;
            norm2 += value * value;
        }
        norm2.sqrt()
    }

    fn reconstruct(
        &self,
        decoder: ndarray::ArrayView2<'_, f32>,
    ) -> Result<ndarray::Array2<f32>, String> {
        if self.block_size == 1 {
            gam::terms::sae::sparse_dict::reconstruct_sparse_rows(
                decoder,
                self.indices.view(),
                self.values.index_axis(ndarray::Axis(2), 0),
            )
        } else {
            gam::terms::sae::sparse_dict::reconstruct_block_sparse_rows(
                decoder,
                self.indices.view(),
                self.values.view(),
                self.block_size,
            )
        }
    }
}

fn residuals_from_sparse_sae(
    data: ndarray::ArrayView2<'_, f32>,
    decoder: ndarray::ArrayView2<'_, f32>,
    route: &AuditSparseRoute,
) -> Result<ndarray::Array2<f32>, String> {
    if data.ncols() != decoder.ncols() || data.nrows() != route.nrows() {
        return Err(format!(
            "audit_sae data shape {:?} is incompatible with decoder {:?} and {} route rows",
            data.dim(),
            decoder.dim(),
            route.nrows()
        ));
    }
    let fitted = route.reconstruct(decoder)?;
    let mut residuals = data.to_owned();
    residuals -= &fitted;
    Ok(residuals)
}

/// Build the fit-report sub-dict for a [`DualCertificateReport`] — the lane's
/// global-optimality certificate channel: the certified fraction of rows, the
/// per-row optimality-ratio quantiles, and the top strictly-improving
/// `(row, atom, η)` birth candidates. Emitted on every lane fit so the global
/// -optimality certificate sits beside the reconstruction/EV report the way the
/// first-order LAML audit sits beside the exact-manifold fit.
///
/// This is the single crate-root definition, shared (by bare name, via the
/// `include!`-flat namespace) with the sparse-dictionary / block entrypoints in
/// `manifold/geometry_ffi.rs`. Returns a borrowed `Bound` handle; callers that
/// need an owned `Py<PyDict>` `.unbind()` it, and `set_item` accepts either.
fn dual_certificate_report_dict<'py>(
    py: Python<'py>,
    report: &gam::terms::sae::dual_certificate::DualCertificateReport,
) -> PyResult<Bound<'py, PyDict>> {
    let out = PyDict::new(py);
    out.set_item("n_rows", report.n_rows)?;
    out.set_item("frac_certified", report.frac_certified)?;
    out.set_item(
        "optimality_ratio_quantiles",
        report.optimality_ratio_quantiles.clone(),
    )?;
    out.set_item("birth_candidates", report.birth_candidates.clone())?;
    Ok(out)
}

fn block_coordinate_report_dict<'py>(
    py: Python<'py>,
    report: &gam::terms::sae::sparse_dict::BlockCoordinateReport,
) -> PyResult<Bound<'py, PyDict>> {
    let n = report.firings.len();
    let mut firing_block = Vec::with_capacity(n);
    let mut firing_row = Vec::with_capacity(n);
    let mut firing_t = Vec::with_capacity(n);
    let mut firing_amplitude = Vec::with_capacity(n);
    let mut firing_t_se = Vec::with_capacity(n);
    let mut firing_amplitude_se = Vec::with_capacity(n);
    let mut firing_t_se_clamped = Vec::with_capacity(n);
    for f in &report.firings {
        firing_block.push(f.block as u64);
        firing_row.push(f.row as u64);
        firing_t.push(f.t);
        firing_amplitude.push(f.amplitude);
        firing_t_se.push(f.t_se);
        firing_amplitude_se.push(f.amplitude_se);
        firing_t_se_clamped.push(f.t_se_clamped);
    }

    let out = PyDict::new(py);
    out.set_item("sigma_hat", report.sigma_hat)?;
    out.set_item("mean_radius", report.mean_radius)?;
    out.set_item("n_firings", report.n_firings)?;
    out.set_item(
        "block",
        ndarray::Array1::from_vec(firing_block).into_pyarray(py),
    )?;
    out.set_item(
        "row",
        ndarray::Array1::from_vec(firing_row).into_pyarray(py),
    )?;
    out.set_item("t", ndarray::Array1::from_vec(firing_t).into_pyarray(py))?;
    out.set_item(
        "amplitude",
        ndarray::Array1::from_vec(firing_amplitude).into_pyarray(py),
    )?;
    out.set_item(
        "t_se",
        ndarray::Array1::from_vec(firing_t_se).into_pyarray(py),
    )?;
    out.set_item(
        "amplitude_se",
        ndarray::Array1::from_vec(firing_amplitude_se).into_pyarray(py),
    )?;
    out.set_item("t_se_clamped", firing_t_se_clamped)?;
    Ok(out)
}

#[derive(Clone, Copy, Default)]
struct SparsePairAccum {
    n_joint: usize,
    sum_a: f64,
    sum_b: f64,
    sum_a2: f64,
    sum_b2: f64,
    sum_ab: f64,
}

struct AbsorptionPairReport {
    a: usize,
    b: usize,
    n_obs: usize,
    n_a: usize,
    n_b: usize,
    n_joint: usize,
    p_a_given_b: f64,
    p_b_given_a: f64,
    lift: f64,
    weight_correlation: f64,
    dependence: f64,
    fusion_evidence: f64,
    absorption_asymmetry: f64,
}

struct AbsorptionAuditReport {
    n_units: usize,
    activation_threshold: f32,
    pairs: Vec<AbsorptionPairReport>,
}

fn absorption_audit(
    route: &AuditSparseRoute,
    activation_threshold: f32,
    max_pairs: usize,
) -> AbsorptionAuditReport {
    let mut marginals = vec![0usize; route.n_units];
    let mut accumulators = std::collections::BTreeMap::<(usize, usize), SparsePairAccum>::new();
    for row in 0..route.nrows() {
        let mut live = Vec::with_capacity(route.width());
        for slot in 0..route.width() {
            let weight = route.gate(row, slot);
            if weight > activation_threshold as f64 {
                let unit = route.indices[[row, slot]] as usize;
                marginals[unit] += 1;
                live.push((unit, weight));
            }
        }
        live.sort_unstable_by_key(|(unit, _)| *unit);
        for left in 0..live.len() {
            for right in (left + 1)..live.len() {
                let (a, wa) = live[left];
                let (b, wb) = live[right];
                let acc = accumulators.entry((a, b)).or_default();
                acc.n_joint += 1;
                acc.sum_a += wa;
                acc.sum_b += wb;
                acc.sum_a2 += wa * wa;
                acc.sum_b2 += wb * wb;
                acc.sum_ab += wa * wb;
            }
        }
    }
    let n_obs = route.nrows();
    let mut pairs = accumulators
        .into_iter()
        .map(|((a, b), acc)| {
            let n_a = marginals[a];
            let n_b = marginals[b];
            let conditional = |joint: usize, marginal: usize| {
                if marginal == 0 {
                    0.0
                } else {
                    joint as f64 / marginal as f64
                }
            };
            let p_a_given_b = conditional(acc.n_joint, n_b);
            let p_b_given_a = conditional(acc.n_joint, n_a);
            let lift = if n_a == 0 || n_b == 0 || n_obs == 0 {
                0.0
            } else {
                acc.n_joint as f64 * n_obs as f64 / (n_a as f64 * n_b as f64)
            };
            let weight_correlation = if acc.n_joint < 2 {
                0.0
            } else {
                let n = acc.n_joint as f64;
                let covariance = acc.sum_ab - acc.sum_a * acc.sum_b / n;
                let variance_a = acc.sum_a2 - acc.sum_a * acc.sum_a / n;
                let variance_b = acc.sum_b2 - acc.sum_b * acc.sum_b / n;
                if variance_a > 0.0 && variance_b > 0.0 {
                    (covariance / (variance_a.sqrt() * variance_b.sqrt())).clamp(-1.0, 1.0)
                } else {
                    0.0
                }
            };
            let dependence = p_a_given_b.min(p_b_given_a);
            AbsorptionPairReport {
                a,
                b,
                n_obs,
                n_a,
                n_b,
                n_joint: acc.n_joint,
                p_a_given_b,
                p_b_given_a,
                lift,
                weight_correlation,
                dependence,
                fusion_evidence: dependence * weight_correlation.abs(),
                absorption_asymmetry: (p_a_given_b - p_b_given_a).abs(),
            }
        })
        .collect::<Vec<_>>();
    pairs.sort_by(|left, right| {
        right
            .absorption_asymmetry
            .total_cmp(&left.absorption_asymmetry)
    });
    pairs.truncate(max_pairs);
    AbsorptionAuditReport {
        n_units: route.n_units,
        activation_threshold,
        pairs,
    }
}

fn absorption_audit_dict<'py>(
    py: Python<'py>,
    report: &AbsorptionAuditReport,
) -> PyResult<Bound<'py, PyDict>> {
    let pair_list = PyList::empty(py);
    for entry in &report.pairs {
        let pair = PyDict::new(py);
        pair.set_item("a", entry.a)?;
        pair.set_item("b", entry.b)?;
        pair.set_item("n_obs", entry.n_obs)?;
        pair.set_item("n_a", entry.n_a)?;
        pair.set_item("n_b", entry.n_b)?;
        pair.set_item("n_joint", entry.n_joint)?;
        pair.set_item("p_a_given_b", entry.p_a_given_b)?;
        pair.set_item("p_b_given_a", entry.p_b_given_a)?;
        pair.set_item("lift", entry.lift)?;
        pair.set_item("weight_correlation", entry.weight_correlation)?;
        pair.set_item("dependence", entry.dependence)?;
        pair.set_item("fusion_evidence", entry.fusion_evidence)?;
        pair.set_item("absorption_asymmetry", entry.absorption_asymmetry)?;
        pair_list.append(pair)?;
    }

    let out = PyDict::new(py);
    out.set_item("n_units", report.n_units)?;
    out.set_item("activation_threshold", report.activation_threshold)?;
    out.set_item("pairs", pair_list)?;
    Ok(out)
}

fn transport_report_dict<'py>(
    py: Python<'py>,
    report: &gam::terms::sae::inference::transport_class::CircleTransportReport,
) -> PyResult<Bound<'py, PyDict>> {
    let class_name = match report.class {
        gam::terms::sae::inference::transport_class::CircleTransportClass::Shift => "shift",
        gam::terms::sae::inference::transport_class::CircleTransportClass::Reflect => "reflect",
        gam::terms::sae::inference::transport_class::CircleTransportClass::Mixing => "mixing",
    };
    let out = PyDict::new(py);
    out.set_item("layer_from", report.layer_from)?;
    out.set_item("layer_to", report.layer_to)?;
    out.set_item("n_samples", report.n_samples)?;
    out.set_item("winding", report.winding)?;
    out.set_item("phase", report.phase)?;
    out.set_item("phase_degrees", report.phase_degrees())?;
    out.set_item("defect", report.defect)?;
    out.set_item("resultant_shift", report.resultant_shift)?;
    out.set_item("resultant_reflect", report.resultant_reflect)?;
    out.set_item("class", class_name)?;
    Ok(out)
}

#[derive(Clone)]
struct AuditTopologyRecord {
    atom: usize,
    support_size: usize,
    landmark_count: usize,
    covering_side: String,
    measured_betti: gam::terms::sae::manifold::BettiSignature,
    expected_betti: gam::terms::sae::manifold::BettiSignature,
    contested: bool,
    dominant_h1_persistence: f64,
    dominant_h2_persistence: f64,
    note: String,
}

struct AuditAtlasReport {
    chart_blocks: Vec<usize>,
    diagram: gam::terms::sae::inference::atlas_nerve::AtlasNerveDiagram,
}

fn betti_signature_dict<'py>(
    py: Python<'py>,
    betti: gam::terms::sae::manifold::BettiSignature,
) -> PyResult<Bound<'py, PyDict>> {
    let out = PyDict::new(py);
    out.set_item("b0", betti.b0)?;
    out.set_item("b1", betti.b1)?;
    match betti.b2 {
        Some(value) => out.set_item("b2", value)?,
        None => out.set_item("b2", py.None())?,
    }
    Ok(out)
}

fn topology_records_dict<'py>(
    py: Python<'py>,
    records: &[AuditTopologyRecord],
    calibration: Option<&gam::terms::sae::null_battery::ClaimNullCalibration>,
) -> PyResult<Bound<'py, PyDict>> {
    let atoms = PyList::empty(py);
    let mut circles = 0usize;
    let mut tori = 0usize;
    let mut lines_or_points = 0usize;
    let mut contested = 0usize;
    for record in records {
        if record.contested {
            contested += 1;
        }
        if record.measured_betti.b1 == 2 && record.measured_betti.b2 == Some(1) {
            tori += 1;
        } else if record.measured_betti.b1 == 1 {
            circles += 1;
        } else if record.measured_betti.b1 == 0 {
            lines_or_points += 1;
        }

        let row = PyDict::new(py);
        row.set_item("atom", record.atom)?;
        row.set_item("support_size", record.support_size)?;
        row.set_item("landmark_count", record.landmark_count)?;
        row.set_item("covering_side", &record.covering_side)?;
        row.set_item(
            "measured_betti",
            betti_signature_dict(py, record.measured_betti)?,
        )?;
        row.set_item(
            "expected_betti",
            betti_signature_dict(py, record.expected_betti)?,
        )?;
        row.set_item("contested", record.contested)?;
        row.set_item("dominant_h1_persistence", record.dominant_h1_persistence)?;
        row.set_item("dominant_h2_persistence", record.dominant_h2_persistence)?;
        row.set_item("note", &record.note)?;
        atoms.append(row)?;
    }

    let summary = PyDict::new(py);
    summary.set_item("n_atoms", records.len())?;
    summary.set_item("circles", circles)?;
    summary.set_item("tori", tori)?;
    summary.set_item("lines_or_points", lines_or_points)?;
    summary.set_item("contested", contested)?;
    set_null_calibration_items(&summary, calibration)?;

    let out = PyDict::new(py);
    out.set_item("summary", summary)?;
    out.set_item("atoms", atoms)?;
    Ok(out)
}

fn topology_records_from_codes(
    coordinate_reports: &[gam::terms::sae::sparse_dict::BlockCoordinateReport],
    block_size: usize,
    activation_threshold: f32,
) -> Vec<AuditTopologyRecord> {
    let threshold = activation_threshold as f64;
    if block_size == 1 {
        // A scalar external SAE feature is a point/line chart, not a manifold:
        // there is no circle/torus topology to audit. This mirrors the atlas
        // nerve, which likewise declines the scalar `block_size == 1` shape, so a
        // scalar dictionary reports zero topology atoms rather than one trivial
        // point record per column.
        return Vec::new();
    }

    let expected = gam::terms::sae::manifold::BettiSignature {
        b0: 1,
        b1: 1,
        b2: None,
    };
    let mut records = Vec::with_capacity(coordinate_reports.len());
    for report in coordinate_reports {
        let live: Vec<_> = report
            .firings
            .iter()
            .filter(|firing| firing.amplitude > threshold)
            .collect();
        if live.len() < 4 {
            let measured = gam::terms::sae::manifold::BettiSignature {
                b0: if live.is_empty() { 0 } else { 1 },
                b1: 0,
                b2: None,
            };
            records.push(AuditTopologyRecord {
                atom: report.firings.first().map_or(0, |firing| firing.block),
                support_size: live.len(),
                landmark_count: live.len(),
                covering_side: "below_covering_number".to_string(),
                measured_betti: measured,
                expected_betti: expected,
                contested: measured != expected,
                dominant_h1_persistence: 0.0,
                dominant_h2_persistence: 0.0,
                note: "under-resolved harmonic-circle block: fewer than four firing coordinates"
                    .to_string(),
            });
            continue;
        }

        let mut points = ndarray::Array2::<f64>::zeros((live.len(), 2));
        for (row_idx, firing) in live.iter().enumerate() {
            let phase = std::f64::consts::TAU * firing.t;
            let (sin_phase, cos_phase) = phase.sin_cos();
            points[[row_idx, 0]] = cos_phase;
            points[[row_idx, 1]] = sin_phase;
        }
        if let Some(verdict) = gam::terms::sae::manifold::topology_persistence_verdict(
            points.view(),
            &gam::terms::sae::manifold::SaeAtomBasisKind::Periodic,
        ) {
            records.push(AuditTopologyRecord {
                atom: live[0].block,
                support_size: verdict.support_size,
                landmark_count: verdict.landmark_count,
                covering_side: verdict.covering_side.as_str().to_string(),
                measured_betti: verdict.measured_betti,
                expected_betti: verdict.expected_betti,
                contested: verdict.contested,
                dominant_h1_persistence: verdict.dominant_h1_persistence,
                dominant_h2_persistence: verdict.dominant_h2_persistence,
                note: verdict.note,
            });
        }
    }
    records
}

/// Genuine chart-transfer certificate for one atlas-nerve gate between two
/// charts, read from the frozen code matrix.
///
/// A gate stamped `valid = true` with zero transport/equivariance defect is a
/// FABRICATED certificate: it admits every co-active chart pair as a nerve edge
/// without running any transport test, so the reported topology is manufactured,
/// not measured. The real certificate needs a square (≤2-D) chart-to-chart
/// operator; only the harmonic circle lane (`block_size == 2`) exposes a 2-D
/// per-row coordinate from which the empirical transfer operator `A` (least
/// squares `X_a A ≈ X_b` over the rows that fire in BOTH charts) can be formed.
/// `A` is certified against isometry (`‖AᵀA − I‖_F`) and SO(2) equivariance
/// (`‖A·G − G·A‖_F`) by [`certify_square_transfer`], and validity is the
/// library's own gate ([`AtlasTransferGate::from_square_transfer`]). Any other
/// block width, fewer than two co-firing rows, a singular coordinate Gram, or a
/// non-finite operator exposes no certifiable transfer at this boundary, so the
/// gate is UNCERTIFIED (`valid = false`, unknown/`inf` defects) — never
/// fabricated valid. `block_a`/`block_b` index the dictionary blocks the two
/// charts read; `chart_a`/`chart_b` are the nerve-vertex labels.
fn chart_transfer_gate_sparse(
    route: &AuditSparseRoute,
    support_a: &gam::terms::sae::inference::atlas_nerve::AtlasChart,
    support_b: &gam::terms::sae::inference::atlas_nerve::AtlasChart,
    block_a: usize,
    block_b: usize,
    chart_a: usize,
    chart_b: usize,
) -> gam::terms::sae::inference::atlas_nerve::AtlasTransferGate {
    use gam::terms::sae::inference::atlas_nerve::AtlasTransferGate;
    let uncertified = || AtlasTransferGate {
        a: chart_a,
        b: chart_b,
        valid: false,
        transport_defect: f64::INFINITY,
        equivariance_defect: f64::INFINITY,
    };
    if route.block_size != 2 {
        return uncertified();
    }
    let mut xa: Vec<f64> = Vec::new();
    let mut xb: Vec<f64> = Vec::new();
    let mut position_a = 0usize;
    let mut position_b = 0usize;
    while position_a < support_a.support_rows().len() && position_b < support_b.support_rows().len()
    {
        let row_a = support_a.support_rows()[position_a];
        let row_b = support_b.support_rows()[position_b];
        if row_a < row_b {
            position_a += 1;
            continue;
        }
        if row_b < row_a {
            position_b += 1;
            continue;
        }
        let row = row_a;
        let mut a = None;
        let mut b = None;
        for slot in 0..route.width() {
            let unit = route.indices[[row, slot]] as usize;
            let value = [
                route.values[[row, slot, 0]] as f64,
                route.values[[row, slot, 1]] as f64,
            ];
            if value[0] * value[0] + value[1] * value[1] == 0.0 {
                continue;
            }
            if unit == block_a {
                a = Some(value);
            } else if unit == block_b {
                b = Some(value);
            }
        }
        if let (Some([a0, a1]), Some([b0, b1])) = (a, b) {
            xa.extend([a0, a1]);
            xb.extend([b0, b1]);
        }
        position_a += 1;
        position_b += 1;
    }
    let n_co = xa.len() / 2;
    if n_co < 2 {
        return uncertified();
    }
    let (Ok(x_a), Ok(x_b)) = (
        ndarray::Array2::from_shape_vec((n_co, 2), xa),
        ndarray::Array2::from_shape_vec((n_co, 2), xb),
    ) else {
        return uncertified();
    };
    // Empirical chart-to-chart transfer operator `A = (X_aᵀX_a)⁻¹ X_aᵀX_b`
    // solving `X_a A ≈ X_b` over the co-firing rows.
    let Ok(operator) =
        gam::terms::sae::chart_transfer::pulled_back_operator(x_a.view(), x_b.view())
    else {
        return uncertified();
    };
    // Both charts are circles, so the shared infinitesimal-rotation generator is
    // the SO(2) generator `[[0,−1],[1,0]]`.
    let generator = ndarray::array![[0.0_f64, -1.0], [1.0, 0.0]];
    match gam::terms::sae::chart_transfer::certify_square_transfer(
        operator.view(),
        generator.view(),
        generator.view(),
    ) {
        Ok(cert) => AtlasTransferGate::from_square_transfer(chart_a, chart_b, cert, 2),
        Err(_) => uncertified(),
    }
}

fn atlas_nerve_from_sparse_route(
    route: &AuditSparseRoute,
    activation_threshold: f32,
    requested_blocks: Option<&[usize]>,
) -> Result<Option<AuditAtlasReport>, String> {
    if route.block_size == 1 {
        return Ok(None);
    }
    let n_blocks = route.n_units;
    let chart_blocks: Vec<usize> = match requested_blocks {
        Some(blocks) => blocks.to_vec(),
        None => (0..n_blocks).collect(),
    };
    if chart_blocks.len() < 2 {
        return Ok(None);
    }
    let mut chart_positions = std::collections::HashMap::with_capacity(chart_blocks.len());
    for (chart_idx, &block) in chart_blocks.iter().enumerate() {
        if block >= n_blocks {
            return Err(format!(
                "audit_sae atlas block {block} out of range 0..{n_blocks}"
            ));
        }
        if chart_positions.insert(block, chart_idx).is_some() {
            return Err(format!(
                "audit_sae atlas block {block} is selected more than once"
            ));
        }
    }
    let mut chart_rows = vec![Vec::<usize>::new(); chart_blocks.len()];
    let mut chart_weights = vec![Vec::<f64>::new(); chart_blocks.len()];
    let mut coactive_pairs = std::collections::BTreeSet::<(usize, usize)>::new();
    for row in 0..route.nrows() {
        let mut live_charts = Vec::with_capacity(route.width());
        for slot in 0..route.width() {
            let unit = route.indices[[row, slot]] as usize;
            if let Some(&chart_idx) = chart_positions.get(&unit) {
                let gate = route.gate(row, slot);
                if gate > activation_threshold as f64 {
                    chart_rows[chart_idx].push(row);
                    chart_weights[chart_idx].push(gate);
                    live_charts.push(chart_idx);
                }
            }
        }
        live_charts.sort_unstable();
        for left in 0..live_charts.len() {
            for right in (left + 1)..live_charts.len() {
                coactive_pairs.insert((live_charts[left], live_charts[right]));
            }
        }
    }
    let mut charts = Vec::with_capacity(chart_blocks.len());
    for (chart_idx, (rows, weights)) in chart_rows.into_iter().zip(chart_weights).enumerate() {
        charts.push(
            gam::terms::sae::inference::atlas_nerve::AtlasChart::from_sparse_weights(
                chart_idx,
                route.nrows(),
                rows,
                weights,
            )?,
        );
    }
    let mut gates = Vec::with_capacity(coactive_pairs.len());
    for (a, b) in coactive_pairs {
        gates.push(chart_transfer_gate_sparse(
            route,
            &charts[a],
            &charts[b],
            chart_blocks[a],
            chart_blocks[b],
            a,
            b,
        ));
    }
    let diagram = gam::terms::sae::inference::atlas_nerve::build_atlas_nerve(&charts, &gates)?;
    Ok(Some(AuditAtlasReport {
        chart_blocks,
        diagram,
    }))
}

fn atlas_nerve_dict<'py>(
    py: Python<'py>,
    report: Option<&AuditAtlasReport>,
    skipped_reason: &str,
) -> PyResult<Bound<'py, PyDict>> {
    let out = PyDict::new(py);
    let Some(report) = report else {
        out.set_item("computed", false)?;
        out.set_item("reason", skipped_reason)?;
        return Ok(out);
    };
    let diagram = &report.diagram;
    out.set_item("computed", true)?;
    out.set_item("chart_blocks", report.chart_blocks.clone())?;
    out.set_item("betti", betti_signature_dict(py, diagram.betti)?)?;
    out.set_item("n_vertices", diagram.n_vertices)?;
    out.set_item("n_edges", diagram.n_edges)?;
    out.set_item("n_triangles", diagram.n_triangles)?;
    out.set_item("n_tetrahedra", diagram.n_tetrahedra)?;
    out.set_item("sampled_support_size", diagram.sampled_support_size)?;
    out.set_item("covering_side", diagram.covering_side.as_str())?;
    out.set_item("max_filtration", diagram.max_filtration)?;
    out.set_item("note", &diagram.note)?;
    Ok(out)
}

/// Standalone atlas-nerve diagram from fixed-width sparse block routing (#985 /
/// E1 FFI completeness). This exposes the same Čech-nerve reduction `audit_sae`
/// computes internally as its own front-door accessor: build one `AtlasChart` per
/// requested `b`-wide block from sparse per-row block-energy support, certify only
/// genuinely co-active chart transfers, and reduce the nerve to its Betti signature
/// and simplex counts. Every number is computed in the `gam::terms::sae::inference::
/// atlas_nerve` core; this only marshals the sparse route in and the diagram dict
/// out. Returns `{computed: false, reason}` for shapes the nerve does not apply to
/// (scalar `block_size == 1` or fewer than two selected charts), matching
/// `atlas_nerve_dict`'s skipped-report contract.
#[pyfunction(signature = (
    indices,
    values,
    n_units,
    block_size,
    activation_threshold = 1.0e-6,
    blocks = None
))]
fn atlas_nerve_diagram<'py>(
    py: Python<'py>,
    indices: PyReadonlyArray2<'py, u32>,
    values: PyReadonlyArray3<'py, f32>,
    n_units: usize,
    block_size: usize,
    activation_threshold: f32,
    blocks: Option<Vec<usize>>,
) -> PyResult<Py<PyDict>> {
    let route = AuditSparseRoute::new(
        indices.as_array().to_owned(),
        values.as_array().to_owned(),
        n_units,
        block_size,
        "atlas route",
    )
    .map_err(PyValueError::new_err)?;
    let report = detach_py_result(py, "atlas_nerve_diagram", move || {
        atlas_nerve_from_sparse_route(&route, activation_threshold, blocks.as_deref())
    })?;
    let reason = if block_size == 1 {
        "scalar block_size == 1 exposes no atlas nerve"
    } else {
        "atlas nerve requires at least two selected block charts"
    };
    Ok(atlas_nerve_dict(py, report.as_ref(), reason)?.unbind())
}

/// Dimension spectrometer: fit a single-atom (`s = 1`) sparse dictionary at each
/// rung of the doubling ladder `k_min·2^j`, `j = 0..=n_doublings`, and invert the
/// fitted reconstruction-loss scaling law `L(K) − σ² ∝ K^{-2/d}` into an
/// intrinsic-dimension estimate `d̂ = −2/m` with delta-method standard errors.
/// The forwarded dictionary template mirrors `sparse_dictionary_fit`'s fit
/// knobs; `active` is forced to 1 per rung by the engine regardless.
#[pyfunction(signature = (
    data,
    k_min = 4,
    n_doublings = 6,
    active = 1,
    minibatch = 512,
    max_epochs = 30,
    score_tile = 4096,
    code_ridge = 1.0e-6,
    decoder_ridge = 1.0e-9,
    tolerance = 1.0e-6,
    score_mode = "required"
))]
fn dimension_spectrometer<'py>(
    py: Python<'py>,
    data: PyReadonlyArray2<'py, f32>,
    k_min: usize,
    n_doublings: usize,
    active: usize,
    minibatch: usize,
    max_epochs: usize,
    score_tile: usize,
    code_ridge: f32,
    decoder_ridge: f32,
    tolerance: f64,
    score_mode: &str,
) -> PyResult<Py<PyDict>> {
    let score_mode = parse_sparse_dict_score_mode(score_mode)?;
    let data_values = data.as_array().to_owned();
    let config = gam::terms::sae::spectrometer::SpectrometerConfig {
        k_min,
        n_doublings,
        dict: SparseDictConfig {
            n_atoms: k_min,
            active,
            minibatch,
            max_epochs,
            score_tile,
            code_ridge,
            decoder_ridge,
            tolerance,
            score_mode,
        },
    };
    let report = detach_py_result(py, "dimension_spectrometer", move || {
        gam::terms::sae::spectrometer::dimension_spectrometer(data_values.view(), &config)
    })?;
    let out = PyDict::new(py);
    out.set_item("rungs", report.rungs)?;
    out.set_item("noise_floor", report.noise_floor)?;
    out.set_item("slope", report.slope)?;
    out.set_item("slope_se", report.slope_se)?;
    out.set_item("d_hat", report.d_hat)?;
    out.set_item("d_hat_se", report.d_hat_se)?;
    out.set_item("floor_saturated", report.floor_saturated)?;
    Ok(out.unbind())
}

/// Per-firing circle-coordinate readout for one `b = 2` block of a fitted
/// block-sparse dictionary: phase `t̂ ∈ [0,1)`, amplitude `‖z‖`, and their
/// closed-form SEs (`σ̂` from the block's radial scatter). Takes the block-lane
/// route as `blocks[N,s]` / `codes[N,s,2]`; no fit status or redundant gate
/// matrix is constructed. The firings are returned as aligned columns.
#[pyfunction(signature = (decoder, blocks, codes, block))]
fn block_firing_coordinates<'py>(
    py: Python<'py>,
    decoder: PyReadonlyArray2<'py, f32>,
    blocks: PyReadonlyArray2<'py, u32>,
    codes: PyReadonlyArray3<'py, f32>,
    block: usize,
) -> PyResult<Py<PyDict>> {
    let block_values = blocks.as_array().to_owned();
    let code_values = codes.as_array().to_owned();
    let block_size = code_values.shape()[2];
    if block_size == 0 || decoder.as_array().nrows() % block_size != 0 {
        return Err(PyValueError::new_err(format!(
            "block_firing_coordinates decoder has K={} rows, not a multiple of block_size {block_size}",
            decoder.as_array().nrows()
        )));
    }
    let n_blocks = decoder.as_array().nrows() / block_size;
    let report = detach_py_result(py, "block_firing_coordinates", move || {
        gam::terms::sae::sparse_dict::block_route_firing_coordinates(
            block_values.view(),
            code_values.view(),
            n_blocks,
            block,
        )
    })?;
    Ok(block_coordinate_report_dict(py, &report)?.unbind())
}

/// Build the dict form of a [`RoutabilityFloor`].
fn routability_floor_dict<'py>(
    py: Python<'py>,
    floor: &gam::terms::sae::routability::RoutabilityFloor,
) -> PyResult<Bound<'py, PyDict>> {
    let out = PyDict::new(py);
    out.set_item("p", floor.p)?;
    out.set_item("n_blocks", floor.n_blocks)?;
    out.set_item("b_max", floor.b_max)?;
    out.set_item("delta", floor.delta)?;
    out.set_item("floor", floor.floor)?;
    out.set_item(
        "minimum_routable_energy",
        gam::terms::sae::routability::minimum_routable_energy(floor),
    )?;
    Ok(out)
}

/// Closed-form routability floor `√(b_max/p) + √(2·ln(K/δ)/p)` on the routable
/// energy fraction, plus the derived `minimum_routable_energy`. Degenerate
/// configurations (`p`, `n_blocks`, `b_max` zero; `b_max > p`; non-finite or
/// non-positive `delta`) are rejected as `ValueError` before the (asserting)
/// engine call.
#[pyfunction(signature = (p, n_blocks, b_max, delta))]
fn routability_floor(
    py: Python<'_>,
    p: usize,
    n_blocks: usize,
    b_max: usize,
    delta: f64,
) -> PyResult<Py<PyDict>> {
    if p == 0 || n_blocks == 0 || b_max == 0 {
        return Err(PyValueError::new_err(
            "routability_floor requires p >= 1, n_blocks >= 1, b_max >= 1",
        ));
    }
    if b_max > p {
        return Err(PyValueError::new_err(
            "routability_floor requires b_max <= p (a b-frame must fit in R^p)",
        ));
    }
    if !delta.is_finite() || delta <= 0.0 {
        return Err(PyValueError::new_err(
            "routability_floor requires a finite delta > 0",
        ));
    }
    let floor = gam::terms::sae::routability::routability_floor(p, n_blocks, b_max, delta);
    Ok(routability_floor_dict(py, &floor)?.unbind())
}

/// Empirical routability audit: measure a fitted dictionary's max-cross-gate
/// distribution against real residual rows and compare it to the closed-form
/// floor. `decoder` is `K×P`; `block_size` is 1 for the linear atom lane, `b`
/// for the block lane (must divide `K`). Returns the floor (nested), the
/// requested quantiles, and the coherence-excess summary.
#[pyfunction(signature = (decoder, residuals, block_size, delta, quantile_levels))]
fn routability_audit<'py>(
    py: Python<'py>,
    decoder: PyReadonlyArray2<'py, f32>,
    residuals: PyReadonlyArray2<'py, f32>,
    block_size: usize,
    delta: f64,
    quantile_levels: Vec<f64>,
) -> PyResult<Py<PyDict>> {
    let decoder_values = decoder.as_array().to_owned();
    let residual_values = residuals.as_array().to_owned();
    let report = detach_py_result(py, "routability_audit", move || {
        gam::terms::sae::routability::routability_audit(
            decoder_values.view(),
            residual_values.view(),
            block_size,
            delta,
            &quantile_levels,
        )
    })?;
    let out = PyDict::new(py);
    out.set_item("n_rows", report.n_rows)?;
    out.set_item("floor", routability_floor_dict(py, &report.floor)?)?;
    out.set_item("quantiles", report.quantiles)?;
    out.set_item("empirical_mean", report.empirical_mean)?;
    out.set_item("empirical_max", report.empirical_max)?;
    out.set_item("confidence_quantile", report.confidence_quantile)?;
    out.set_item("coherence_excess", report.coherence_excess)?;
    out.set_item("fraction_below_floor", report.fraction_below_floor)?;
    Ok(out.unbind())
}

/// Global-optimality dual certificate for the collapsed linear lane: for each
/// row of `data`, fold the fitted-routing residual's dual value over the whole
/// dictionary and form the scale-free optimality ratio, reporting the certified
/// fraction, the ratio quantiles, and the top strictly-improving birth
/// candidates. The fit is passed as its columnar arrays.
#[pyfunction(signature = (data, decoder, indices, codes, max_candidates = 16))]
fn sparse_dict_dual_certificate<'py>(
    py: Python<'py>,
    data: PyReadonlyArray2<'py, f32>,
    decoder: PyReadonlyArray2<'py, f32>,
    indices: PyReadonlyArray2<'py, u32>,
    codes: PyReadonlyArray2<'py, f32>,
    max_candidates: usize,
) -> PyResult<Py<PyDict>> {
    let data_values = data.as_array().to_owned();
    let decoder_values = decoder.as_array().to_owned();
    let index_values = indices.as_array().to_owned();
    let code_values = codes.as_array().to_owned();
    let report = detach_py_result(py, "sparse_dict_dual_certificate", move || {
        gam::terms::sae::dual_certificate::sparse_route_dual_certificate(
            data_values.view(),
            decoder_values.view(),
            index_values.view(),
            code_values.view(),
            max_candidates,
        )
    })?;
    let out = PyDict::new(py);
    out.set_item("n_rows", report.n_rows)?;
    out.set_item("frac_certified", report.frac_certified)?;
    out.set_item(
        "optimality_ratio_quantiles",
        report.optimality_ratio_quantiles,
    )?;
    out.set_item("birth_candidates", report.birth_candidates)?;
    Ok(out.unbind())
}

/// Attach the compact null-battery / spike-in calibration a topology or
/// atlas-nerve claim ships with. When `calibration` is present the observed
/// statistic, its architecture-matched-donor p-value / z-score, and the
/// spike-in detection power at the claimed operating point are emitted; when it
/// is absent (a shape carrying no topological claim) the same keys are emitted
/// as `None` so the report schema is stable.
fn set_null_calibration_items(
    dict: &Bound<'_, PyDict>,
    calibration: Option<&gam::terms::sae::null_battery::ClaimNullCalibration>,
) -> PyResult<()> {
    match calibration {
        Some(cal) => {
            dict.set_item("null_pvalue", cal.null_pvalue)?;
            dict.set_item("null_z", cal.null_z)?;
            dict.set_item("null_observed_statistic", cal.observed_statistic)?;
            dict.set_item("spikein_power", cal.spikein_power)?;
            dict.set_item("null_claimed_snr", cal.claimed_snr)?;
            dict.set_item(
                "null_claimed_false_positive_rate",
                cal.claimed_false_positive_rate,
            )?;
        }
        None => {
            let py = dict.py();
            dict.set_item("null_pvalue", py.None())?;
            dict.set_item("null_z", py.None())?;
            dict.set_item("null_observed_statistic", py.None())?;
            dict.set_item("spikein_power", py.None())?;
            dict.set_item("null_claimed_snr", py.None())?;
            dict.set_item("null_claimed_false_positive_rate", py.None())?;
        }
    }
    Ok(())
}

/// Monte-Carlo operating points for the standing null battery / spike-in
/// calibration `audit_sae` attaches to its topology and atlas-nerve claims.
/// Surfaced on the FFI so callers can widen the null replicate count or move the
/// spike-in operating point; the FFI defaults are the reporting operating
/// points.
#[derive(Clone, Copy)]
struct StandingCalibrationConfig {
    null_replicates: usize,
    null_seed: u64,
    spikein_trials: usize,
    spikein_snr: f64,
    spikein_false_positive_rate: f64,
}

/// Atlas-nerve topological-richness statistic over a fixed-width sparse route.
/// The statistic never materializes the logical `N×K` code matrix.
fn sparse_atlas_nerve_richness_statistic(
    route: &AuditSparseRoute,
    chart_blocks: &[usize],
    activation_threshold: f64,
) -> Result<f64, String> {
    let report =
        atlas_nerve_from_sparse_route(route, activation_threshold as f32, Some(chart_blocks))?
            .ok_or_else(|| "atlas null statistic requires at least two block charts".to_string())?;
    Ok((report.diagram.n_edges + report.diagram.n_triangles + report.diagram.n_tetrahedra) as f64)
}

#[derive(Clone, Copy, Default)]
struct LiveAmplitudeMoments {
    count: usize,
    sum: f64,
    sum2: f64,
}

impl LiveAmplitudeMoments {
    fn mean(self) -> f64 {
        if self.count == 0 {
            0.0
        } else {
            self.sum / self.count as f64
        }
    }

    fn sd(self) -> f64 {
        if self.count < 2 {
            0.0
        } else {
            let n = self.count as f64;
            ((self.sum2 - self.sum * self.sum / n) / (n - 1.0))
                .max(0.0)
                .sqrt()
        }
    }
}

fn live_amplitude_moments(route: &AuditSparseRoute) -> Vec<LiveAmplitudeMoments> {
    let mut moments = vec![LiveAmplitudeMoments::default(); route.n_units];
    for row in 0..route.nrows() {
        for slot in 0..route.width() {
            let gate = route.gate(row, slot);
            if gate > 0.0 {
                let unit = route.indices[[row, slot]] as usize;
                moments[unit].count += 1;
                moments[unit].sum += gate;
                moments[unit].sum2 += gate * gate;
            }
        }
    }
    moments
}

fn resample_sparse_architecture_null<R: rand::Rng + ?Sized>(
    observed: &AuditSparseRoute,
    donor: &AuditSparseRoute,
    rng: &mut R,
) -> Result<AuditSparseRoute, String> {
    use rand::RngExt;
    let observed_moments = live_amplitude_moments(observed);
    let donor_moments = live_amplitude_moments(donor);
    let mut indices = ndarray::Array2::<u32>::zeros((observed.nrows(), donor.width()));
    let mut values =
        ndarray::Array3::<f32>::zeros((observed.nrows(), donor.width(), donor.block_size));
    for row in 0..observed.nrows() {
        let source = rng.random_range(0..donor.nrows());
        for slot in 0..donor.width() {
            let unit = donor.indices[[source, slot]] as usize;
            indices[[row, slot]] = unit as u32;
            let gate = donor.gate(source, slot);
            if gate == 0.0 {
                continue;
            }
            let observed_moment = observed_moments[unit];
            let donor_moment = donor_moments[unit];
            if observed_moment.count == 0 {
                continue;
            }
            let donor_sd = donor_moment.sd();
            let target_gate = if donor_sd > 0.0 {
                (observed_moment.mean()
                    + (gate - donor_moment.mean()) * observed_moment.sd() / donor_sd)
                    .max(0.0)
            } else {
                observed_moment.mean()
            };
            if target_gate == 0.0 {
                continue;
            }
            let scale = target_gate / gate;
            for offset in 0..donor.block_size {
                values[[row, slot, offset]] =
                    (donor.values[[source, slot, offset]] as f64 * scale) as f32;
            }
        }
    }
    AuditSparseRoute::new(
        indices,
        values,
        observed.n_units,
        observed.block_size,
        "architecture-matched null route",
    )
}

/// Build the standing sparse donor null + residual spike-in calibration. Both
/// observed and donor routing remain `N×s×b`; implicit zeros are never expanded.
fn standing_sparse_null_calibration(
    route: &AuditSparseRoute,
    donor: &AuditSparseRoute,
    residuals_f64: ndarray::ArrayView2<'_, f64>,
    chart_blocks: &[usize],
    activation_threshold: f64,
    cfg: &StandingCalibrationConfig,
) -> Result<Option<gam::terms::sae::null_battery::ClaimNullCalibration>, String> {
    use gam::terms::sae::null_battery as nb;
    if chart_blocks.len() < 2
        || donor.n_units != route.n_units
        || donor.block_size != route.block_size
        || residuals_f64.nrows() < 4
        || residuals_f64.ncols() < 2
        || cfg.null_replicates == 0
        || cfg.spikein_trials == 0
    {
        return Ok(None);
    }
    use rand::SeedableRng;
    let observed =
        sparse_atlas_nerve_richness_statistic(route, chart_blocks, activation_threshold)?;
    let mut rng = rand::rngs::StdRng::seed_from_u64(cfg.null_seed);
    let mut samples = Vec::with_capacity(cfg.null_replicates);
    for _ in 0..cfg.null_replicates {
        let surrogate = resample_sparse_architecture_null(route, donor, &mut rng)?;
        samples.push(sparse_atlas_nerve_richness_statistic(
            &surrogate,
            chart_blocks,
            activation_threshold,
        )?);
    }
    let null_summary = nb::summarize_null_distribution(
        nb::NullKind::ArchitectureMatchedRandomWeight,
        observed,
        samples,
        nb::Tail::Larger,
    )?;
    let nulls = nb::NullBatteryReport {
        observed,
        summaries: vec![null_summary],
    };
    // Spike-in power: plant a synthetic circle into the real audit residuals and
    // measure the default block-chart/topology detector's recovery rate at the
    // requested false-positive operating point. Bootstrapping the empirical
    // residual rows keeps the real post-fit covariance and tails in the loop.
    let mut roc_config = nb::SpikeInRocConfig::circle(
        vec![0.0, cfg.spikein_snr],
        cfg.spikein_trials,
        cfg.null_seed,
    );
    roc_config.noise_mode = nb::SpikeInNoiseMode::EmpiricalResidualBootstrap;
    roc_config.fpr_levels = vec![cfg.spikein_false_positive_rate];
    let roc = nb::default_spike_in_roc_curve(residuals_f64, &roc_config)?;
    let report = nb::calibrated_roc_claim_report(
        "audit_sae.topology_atlas_nerve",
        cfg.spikein_snr,
        cfg.spikein_false_positive_rate,
        nulls,
        roc,
    )?;
    Ok(Some(nb::ClaimNullCalibration::from_calibrated_roc(report)))
}

/// Typed knob bundle for `audit_sae`, decoded from the single Python-side
/// `options` dict so the FFI entrypoint keeps a small fixed arity while the
/// audit keeps its full tuning surface. Field defaults are the audit's
/// long-standing keyword defaults; [`SaeAuditOptions::from_pydict`] overlays
/// caller-provided keys on top of them and hard-errors on unknown keys so a
/// typo can never silently fall back to a default.
#[derive(Debug)]
struct SaeAuditOptions {
    block_size: usize,
    delta: f64,
    quantile_levels: Option<Vec<f64>>,
    max_candidates: usize,
    coordinate_blocks: Option<Vec<usize>>,
    activation_threshold: f32,
    max_absorption_pairs: usize,
    transport_theta_in: Option<Vec<f64>>,
    transport_theta_out: Option<Vec<f64>>,
    transport_layer_from: usize,
    transport_layer_to: usize,
    null_replicates: usize,
    null_seed: u64,
    spikein_trials: usize,
    spikein_snr: f64,
    spikein_false_positive_rate: f64,
}

impl SaeAuditOptions {
    /// Every key `from_pydict` accepts, in documentation order; kept as a
    /// single list so the unknown-key error names the full valid vocabulary.
    const KNOWN_KEYS: [&'static str; 16] = [
        "block_size",
        "delta",
        "quantile_levels",
        "max_candidates",
        "coordinate_blocks",
        "activation_threshold",
        "max_absorption_pairs",
        "transport_theta_in",
        "transport_theta_out",
        "transport_layer_from",
        "transport_layer_to",
        "null_replicates",
        "null_seed",
        "spikein_trials",
        "spikein_snr",
        "spikein_false_positive_rate",
    ];

    /// The audit's historical keyword defaults (the pre-dict FFI signature).
    fn defaults() -> Self {
        Self {
            block_size: 1,
            delta: 0.05,
            quantile_levels: None,
            max_candidates: 16,
            coordinate_blocks: None,
            activation_threshold: 0.0,
            max_absorption_pairs: 32,
            transport_theta_in: None,
            transport_theta_out: None,
            transport_layer_from: 0,
            transport_layer_to: 1,
            null_replicates: 64,
            null_seed: 0x5AE0_A0D1,
            spikein_trials: 32,
            spikein_snr: 2.0,
            spikein_false_positive_rate: 0.05,
        }
    }

    /// Wrap an extraction failure for one options entry with the key name so
    /// the caller sees which knob was malformed.
    fn knob_error(key: &str, err: &PyErr) -> PyErr {
        PyValueError::new_err(format!("audit_sae options[{key:?}]: {err}"))
    }

    /// `None`-passthrough extraction for the optional index-list knob
    /// (`coordinate_blocks`).
    fn optional_usize_vec(value: &Bound<'_, PyAny>, key: &str) -> PyResult<Option<Vec<usize>>> {
        if value.is_none() {
            return Ok(None);
        }
        value
            .extract::<Vec<usize>>()
            .map(Some)
            .map_err(|err| Self::knob_error(key, &err))
    }

    /// `None`-passthrough extraction for the optional float-vector knobs
    /// (`quantile_levels`, `transport_theta_in`, `transport_theta_out`).
    /// The transport angles arrive from the Python facade as contiguous 1-D
    /// float64 numpy arrays, so a `PyReadonlyArray1<f64>` fast path is tried
    /// first; plain float sequences (lists/tuples) fall through to the
    /// `Vec<f64>` extraction.
    fn optional_f64_vec(value: &Bound<'_, PyAny>, key: &str) -> PyResult<Option<Vec<f64>>> {
        if value.is_none() {
            return Ok(None);
        }
        if let Ok(array) = value.extract::<PyReadonlyArray1<f64>>() {
            return Ok(Some(array.as_array().to_vec()));
        }
        value
            .extract::<Vec<f64>>()
            .map(Some)
            .map_err(|err| Self::knob_error(key, &err))
    }

    /// Decode the `options` dict: start from [`Self::defaults`], overlay every
    /// provided key, and reject unknown keys and non-string keys outright.
    fn from_pydict(options: Option<&Bound<'_, PyDict>>) -> PyResult<Self> {
        let mut cfg = Self::defaults();
        let Some(options) = options else {
            return Ok(cfg);
        };
        for (key_any, value) in options.iter() {
            let key = key_any
                .extract::<String>()
                .map_err(|_| PyValueError::new_err("audit_sae options keys must be strings"))?;
            match key.as_str() {
                "block_size" => {
                    cfg.block_size = value
                        .extract::<usize>()
                        .map_err(|err| Self::knob_error(&key, &err))?;
                }
                "delta" => {
                    cfg.delta = value
                        .extract::<f64>()
                        .map_err(|err| Self::knob_error(&key, &err))?;
                }
                "quantile_levels" => {
                    cfg.quantile_levels = Self::optional_f64_vec(&value, &key)?;
                }
                "max_candidates" => {
                    cfg.max_candidates = value
                        .extract::<usize>()
                        .map_err(|err| Self::knob_error(&key, &err))?;
                }
                "coordinate_blocks" => {
                    cfg.coordinate_blocks = Self::optional_usize_vec(&value, &key)?;
                }
                "activation_threshold" => {
                    cfg.activation_threshold = value
                        .extract::<f32>()
                        .map_err(|err| Self::knob_error(&key, &err))?;
                }
                "max_absorption_pairs" => {
                    cfg.max_absorption_pairs = value
                        .extract::<usize>()
                        .map_err(|err| Self::knob_error(&key, &err))?;
                }
                "transport_theta_in" => {
                    cfg.transport_theta_in = Self::optional_f64_vec(&value, &key)?;
                }
                "transport_theta_out" => {
                    cfg.transport_theta_out = Self::optional_f64_vec(&value, &key)?;
                }
                "transport_layer_from" => {
                    cfg.transport_layer_from = value
                        .extract::<usize>()
                        .map_err(|err| Self::knob_error(&key, &err))?;
                }
                "transport_layer_to" => {
                    cfg.transport_layer_to = value
                        .extract::<usize>()
                        .map_err(|err| Self::knob_error(&key, &err))?;
                }
                "null_replicates" => {
                    cfg.null_replicates = value
                        .extract::<usize>()
                        .map_err(|err| Self::knob_error(&key, &err))?;
                }
                "null_seed" => {
                    cfg.null_seed = value
                        .extract::<u64>()
                        .map_err(|err| Self::knob_error(&key, &err))?;
                }
                "spikein_trials" => {
                    cfg.spikein_trials = value
                        .extract::<usize>()
                        .map_err(|err| Self::knob_error(&key, &err))?;
                }
                "spikein_snr" => {
                    cfg.spikein_snr = value
                        .extract::<f64>()
                        .map_err(|err| Self::knob_error(&key, &err))?;
                }
                "spikein_false_positive_rate" => {
                    cfg.spikein_false_positive_rate = value
                        .extract::<f64>()
                        .map_err(|err| Self::knob_error(&key, &err))?;
                }
                other => {
                    return Err(PyValueError::new_err(format!(
                        "audit_sae options has unknown key {other:?}; valid keys: {}",
                        Self::KNOWN_KEYS.join(", ")
                    )));
                }
            }
        }
        Ok(cfg)
    }
}

/// One-shot audit over an externally supplied, frozen SAE dictionary. Routing
/// crosses the boundary only as fixed-width sparse `(indices, values)` arrays:
/// `indices` is `N×s`, `values` is `N×s×b`, and implicit zeros are never
/// expanded into `N×K`. The architecture-matched donor uses the same sparse
/// representation and may have a different row count / route width.
///
/// All tuning knobs travel in the single optional `options` dict; omitted keys
/// (or `options = None`) take the audit's long-standing defaults, and unknown
/// keys are a hard error. Accepted keys and their defaults:
///
/// * `block_size = 1` — dictionary block width; `>= 2` selects the block lane.
/// * `delta = 0.05` — routability confidence level.
/// * `quantile_levels = None` — routability quantiles (default `[0.5, 0.9, 0.99]`).
/// * `max_candidates = 16` — dual-certificate candidate budget.
/// * `coordinate_blocks = None` — block indices for the harmonic coordinate readout.
/// * `activation_threshold = 0.0` — firing threshold for topology/atlas/absorption.
/// * `max_absorption_pairs = 32` — absorption-audit pair budget.
/// * `transport_theta_in = None` / `transport_theta_out = None` — paired circle
///   coordinates (1-D float64 array or float sequence) for the transport class.
/// * `transport_layer_from = 0` / `transport_layer_to = 1` — transport layer labels.
/// * `null_replicates = 64`, `null_seed = 0x5AE0_A0D1` — donor null battery.
/// * `spikein_trials = 32`, `spikein_snr = 2.0`,
///   `spikein_false_positive_rate = 0.05` — residual spike-in ROC calibration.
#[pyfunction(signature = (
    decoder,
    route_indices,
    route_values,
    data,
    donor_indices,
    donor_values,
    options = None
))]
fn audit_sae<'py>(
    py: Python<'py>,
    decoder: PyReadonlyArray2<'py, f32>,
    route_indices: PyReadonlyArray2<'py, u32>,
    route_values: PyReadonlyArray3<'py, f32>,
    data: PyReadonlyArray2<'py, f32>,
    donor_indices: PyReadonlyArray2<'py, u32>,
    donor_values: PyReadonlyArray3<'py, f32>,
    options: Option<&Bound<'py, PyDict>>,
) -> PyResult<Py<PyDict>> {
    let SaeAuditOptions {
        block_size,
        delta,
        quantile_levels,
        max_candidates,
        coordinate_blocks,
        activation_threshold,
        max_absorption_pairs,
        transport_theta_in: theta_in_values,
        transport_theta_out: theta_out_values,
        transport_layer_from,
        transport_layer_to,
        null_replicates,
        null_seed,
        spikein_trials,
        spikein_snr,
        spikein_false_positive_rate,
    } = SaeAuditOptions::from_pydict(options)?;
    let decoder_values = decoder.as_array().to_owned();
    let route_indices = route_indices.as_array().to_owned();
    let route_values = route_values.as_array().to_owned();
    let data_values = data.as_array().to_owned();
    let donor_indices = donor_indices.as_array().to_owned();
    let donor_values = donor_values.as_array().to_owned();
    if decoder_values.nrows() == 0 || decoder_values.ncols() == 0 {
        return Err(PyValueError::new_err(
            "audit_sae requires a non-empty decoder matrix",
        ));
    }
    if block_size == 0 {
        return Err(PyValueError::new_err("audit_sae block_size must be >= 1"));
    }
    if decoder_values.nrows() % block_size != 0 {
        return Err(PyValueError::new_err(format!(
            "audit_sae decoder has K={} rows, not a multiple of block_size {block_size}",
            decoder_values.nrows()
        )));
    }
    if decoder_values.iter().any(|value| !value.is_finite())
        || data_values.iter().any(|value| !value.is_finite())
    {
        return Err(PyValueError::new_err(
            "audit_sae decoder and activations must be finite",
        ));
    }
    let n_units = decoder_values.nrows() / block_size;
    let route = AuditSparseRoute::new(
        route_indices,
        route_values,
        n_units,
        block_size,
        "observed route",
    )
    .map_err(PyValueError::new_err)?;
    let donor = AuditSparseRoute::new(
        donor_indices,
        donor_values,
        n_units,
        block_size,
        "random-weight donor route",
    )
    .map_err(PyValueError::new_err)?;
    if data_values.nrows() != route.nrows() || data_values.ncols() != decoder_values.ncols() {
        return Err(PyValueError::new_err(format!(
            "audit_sae data shape {:?} is incompatible with {} route rows and decoder {:?}",
            data_values.dim(),
            route.nrows(),
            decoder_values.dim()
        )));
    }
    if !delta.is_finite() || delta <= 0.0 {
        return Err(PyValueError::new_err(
            "audit_sae requires a finite delta > 0",
        ));
    }
    if activation_threshold < 0.0 || !activation_threshold.is_finite() {
        return Err(PyValueError::new_err(
            "audit_sae activation_threshold must be finite and non-negative",
        ));
    }
    if null_replicates == 0 {
        return Err(PyValueError::new_err(
            "audit_sae null_replicates must be >= 1",
        ));
    }
    if spikein_trials == 0 {
        return Err(PyValueError::new_err(
            "audit_sae spikein_trials must be >= 1",
        ));
    }
    if !spikein_snr.is_finite() || spikein_snr < 0.0 {
        return Err(PyValueError::new_err(
            "audit_sae spikein_snr must be finite and non-negative",
        ));
    }
    if !spikein_false_positive_rate.is_finite()
        || spikein_false_positive_rate <= 0.0
        || spikein_false_positive_rate >= 1.0
    {
        return Err(PyValueError::new_err(
            "audit_sae spikein_false_positive_rate must be in (0, 1)",
        ));
    }

    let calibration_cfg = StandingCalibrationConfig {
        null_replicates,
        null_seed,
        spikein_trials,
        spikein_snr,
        spikein_false_positive_rate,
    };
    let quantiles = quantile_levels.unwrap_or_else(|| vec![0.5, 0.9, 0.99]);
    let decoder_shape = decoder_values.dim();
    let route_rows = route.nrows();
    let route_width = route.width();

    let audit = detach_py_result(py, "audit_sae", move || {
        let residuals =
            residuals_from_sparse_sae(data_values.view(), decoder_values.view(), &route)?;
        let routability = gam::terms::sae::routability::routability_audit(
            decoder_values.view(),
            residuals.view(),
            block_size,
            delta,
            &quantiles,
        )?;

        let (dual, coordinate_reports) = if block_size == 1 {
            let report = gam::terms::sae::dual_certificate::sparse_route_dual_certificate(
                data_values.view(),
                decoder_values.view(),
                route.indices.view(),
                route.values.index_axis(ndarray::Axis(2), 0),
                max_candidates,
            )?;
            (report, Vec::new())
        } else {
            let report = gam::terms::sae::dual_certificate::block_route_dual_certificate(
                data_values.view(),
                decoder_values.view(),
                route.indices.view(),
                route.values.view(),
                block_size,
                max_candidates,
            )?;
            let mut coordinates = Vec::new();
            if block_size >= 2 && block_size % 2 == 0 {
                let total_blocks = decoder_values.nrows() / block_size;
                let blocks = coordinate_blocks
                    .clone()
                    .unwrap_or_else(|| (0..total_blocks).collect());
                for block in blocks {
                    if block >= total_blocks {
                        return Err(format!(
                            "audit_sae coordinate block {block} out of range 0..{total_blocks}"
                        ));
                    }
                    coordinates.push(
                        gam::terms::sae::sparse_dict::harmonic_route_firing_coordinates(
                            route.indices.view(),
                            route.values.view(),
                            n_units,
                            block,
                        )?,
                    );
                }
            }
            (report, coordinates)
        };

        let topology_records =
            topology_records_from_codes(&coordinate_reports, block_size, activation_threshold);
        let atlas_nerve = atlas_nerve_from_sparse_route(
            &route,
            activation_threshold,
            coordinate_blocks.as_deref(),
        )?;
        let absorption = absorption_audit(&route, activation_threshold, max_absorption_pairs);

        let transport = match (theta_in_values, theta_out_values) {
            (Some(theta_in), Some(theta_out)) => Some(
                gam::terms::sae::inference::transport_class::classify_circle_transport(
                    &theta_in,
                    &theta_out,
                    transport_layer_from,
                    transport_layer_to,
                )?,
            ),
            (None, None) => None,
            _ => {
                return Err(
                    "audit_sae transport requires both transport_theta_in and transport_theta_out"
                        .to_string(),
                );
            }
        };

        // Standing null battery + spike-in calibration for the audit's
        // topological claims: re-invoke the atlas-richness audit on the
        // architecture-matched random-weight donor and plant a circle into the
        // real residuals. Gated on a selected atlas chart set (block dictionaries
        // with >= 2 charts); scalar/degenerate shapes carry no such claim.
        let calibration = match atlas_nerve.as_ref() {
            Some(atlas) => {
                let residuals_f64 = residuals.mapv(|value| value as f64);
                standing_sparse_null_calibration(
                    &route,
                    &donor,
                    residuals_f64.view(),
                    &atlas.chart_blocks,
                    activation_threshold as f64,
                    &calibration_cfg,
                )?
            }
            None => None,
        };

        Ok::<_, String>((
            routability,
            dual,
            coordinate_reports,
            topology_records,
            atlas_nerve,
            absorption,
            transport,
            calibration,
        ))
    })?;

    let (
        routability,
        dual,
        coordinate_reports,
        topology_records,
        atlas_nerve,
        absorption,
        transport,
        calibration,
    ) = audit;
    let out = PyDict::new(py);
    out.set_item("decoder_shape", decoder_shape)?;
    out.set_item("codes_shape", (route_rows, decoder_shape.0))?;

    let route = PyDict::new(py);
    route.set_item("block_size", block_size)?;
    route.set_item("n_units", n_units)?;
    route.set_item("width", route_width)?;
    out.set_item("routing", route)?;

    let routability_dict = PyDict::new(py);
    routability_dict.set_item("n_rows", routability.n_rows)?;
    routability_dict.set_item("floor", routability_floor_dict(py, &routability.floor)?)?;
    routability_dict.set_item("quantiles", routability.quantiles)?;
    routability_dict.set_item("empirical_mean", routability.empirical_mean)?;
    routability_dict.set_item("empirical_max", routability.empirical_max)?;
    routability_dict.set_item("confidence_quantile", routability.confidence_quantile)?;
    routability_dict.set_item("coherence_excess", routability.coherence_excess)?;
    routability_dict.set_item("fraction_below_floor", routability.fraction_below_floor)?;
    routability_dict.set_item(
        "dark_matter_fraction",
        1.0 - routability.fraction_below_floor,
    )?;
    out.set_item("routability", routability_dict)?;
    out.set_item("dual_certificate", dual_certificate_report_dict(py, &dual)?)?;
    out.set_item("absorption", absorption_audit_dict(py, &absorption)?)?;

    let coordinate_list = PyList::empty(py);
    for report in &coordinate_reports {
        coordinate_list.append(block_coordinate_report_dict(py, report)?)?;
    }
    out.set_item("coordinate_se", coordinate_list)?;
    out.set_item(
        "topology",
        topology_records_dict(py, &topology_records, calibration.as_ref())?,
    )?;
    let atlas_dict = atlas_nerve_dict(
        py,
        atlas_nerve.as_ref(),
        "atlas nerve requires a block dictionary with at least two selected charts",
    )?;
    set_null_calibration_items(&atlas_dict, calibration.as_ref())?;
    out.set_item("atlas_nerve", atlas_dict)?;

    match transport {
        Some(report) => out.set_item("transport", transport_report_dict(py, &report)?)?,
        None => out.set_item("transport", py.None())?,
    }
    Ok(out.unbind())
}

#[cfg(test)]
mod sae_spectral_ffi_tests {
    use super::*;

    fn assert_dict_has_key(dict: &Bound<'_, PyDict>, key: &str) {
        assert!(
            dict.get_item(key)
                .unwrap_or_else(|err| panic!("read {key}: {err}"))
                .is_some(),
            "missing key {key}"
        );
    }

    #[test]
    fn audit_sae_round_trip_surfaces_external_dictionary_diagnostics() {
        Python::attach(|py| {
            let decoder = ndarray::array![[1.0_f32, 0.0_f32], [0.0_f32, 1.0_f32]];
            let codes = ndarray::array![
                [1.0_f32, 0.0_f32],
                [0.0_f32, 1.0_f32],
                [0.5_f32, 0.5_f32],
                [1.0_f32, 1.0_f32],
                [0.25_f32, 0.75_f32],
                [0.75_f32, 0.25_f32],
                [0.2_f32, 0.0_f32],
                [0.0_f32, 0.2_f32],
            ];
            let data = codes.clone();
            let random_weight_codes = ndarray::array![
                [0.31_f32, 0.72_f32],
                [0.64_f32, 0.18_f32],
                [0.12_f32, 0.55_f32],
                [0.83_f32, 0.27_f32],
                [0.49_f32, 0.61_f32],
                [0.22_f32, 0.44_f32],
                [0.71_f32, 0.09_f32],
                [0.38_f32, 0.86_f32],
            ];
            let indices =
                ndarray::Array2::from_shape_fn((codes.nrows(), 2), |(_, slot)| slot as u32);
            let route_values =
                ndarray::Array3::from_shape_fn((codes.nrows(), 2, 1), |(row, slot, _)| {
                    codes[[row, slot]]
                });
            let donor_values = ndarray::Array3::from_shape_fn(
                (random_weight_codes.nrows(), 2, 1),
                |(row, slot, _)| random_weight_codes[[row, slot]],
            );
            let theta_in = ndarray::array![
                0.0_f64,
                std::f64::consts::FRAC_PI_4,
                std::f64::consts::FRAC_PI_2,
                3.0 * std::f64::consts::FRAC_PI_4,
                std::f64::consts::PI,
                5.0 * std::f64::consts::FRAC_PI_4,
                3.0 * std::f64::consts::FRAC_PI_2,
                7.0 * std::f64::consts::FRAC_PI_4,
            ];
            let theta_out = theta_in.mapv(|theta| theta + 0.25);

            let decoder_py = decoder.into_pyarray(py);
            let indices_py = indices.into_pyarray(py);
            let route_py = route_values.into_pyarray(py);
            let data_py = data.into_pyarray(py);
            let donor_py = donor_values.into_pyarray(py);
            let theta_in_py = theta_in.into_pyarray(py);
            let theta_out_py = theta_out.into_pyarray(py);

            let options = PyDict::new(py);
            options
                .set_item("quantile_levels", vec![0.5, 0.9])
                .expect("set quantile_levels");
            options
                .set_item("max_candidates", 4)
                .expect("set max_candidates");
            options
                .set_item("max_absorption_pairs", 4)
                .expect("set max_absorption_pairs");
            options
                .set_item("transport_theta_in", &theta_in_py)
                .expect("set transport_theta_in");
            options
                .set_item("transport_theta_out", &theta_out_py)
                .expect("set transport_theta_out");
            options
                .set_item("transport_layer_from", 3)
                .expect("set transport_layer_from");
            options
                .set_item("transport_layer_to", 4)
                .expect("set transport_layer_to");

            let payload = audit_sae(
                py,
                decoder_py.readonly(),
                indices_py.readonly(),
                route_py.readonly(),
                data_py.readonly(),
                indices_py.readonly(),
                donor_py.readonly(),
                Some(&options),
            )
            .expect("audit_sae FFI round-trip");

            let report = payload.bind(py);
            for key in [
                "dual_certificate",
                "routability",
                "coordinate_se",
                "absorption",
                "transport",
                "topology",
                "atlas_nerve",
            ] {
                assert_dict_has_key(report, key);
            }
            let transport_any = report
                .get_item("transport")
                .expect("read transport")
                .expect("transport present");
            let transport = transport_any.cast::<PyDict>().expect("transport dict");
            assert_eq!(
                transport
                    .get_item("class")
                    .expect("read transport class")
                    .expect("transport class present")
                    .extract::<String>()
                    .expect("transport class string"),
                "shift"
            );
            assert_eq!(
                transport
                    .get_item("n_samples")
                    .expect("read transport n_samples")
                    .expect("transport n_samples present")
                    .extract::<usize>()
                    .expect("transport n_samples usize"),
                8
            );
        });
    }

    #[test]
    fn audit_sae_options_rejects_unknown_keys() {
        Python::attach(|py| {
            let options = PyDict::new(py);
            options
                .set_item("block_sise", 2)
                .expect("set misspelled key");
            let err = SaeAuditOptions::from_pydict(Some(&options))
                .expect_err("typo'd options key must error, not fall back to defaults");
            let message = err.to_string();
            assert!(
                message.contains("unknown key \"block_sise\"") && message.contains("block_size"),
                "unknown-key error must name the typo and the valid vocabulary; got {message}"
            );
        });
    }
}

/// Compose a chain of component contracts into one end-to-end shadowing bound.
/// Each contract is `(name, domain_radius, defect, lipschitz)`; returns the
/// total defect, its per-stage additive contributions, and the drift-only
/// domain-feasibility flag.
#[pyfunction(signature = (chain,))]
fn compose_contracts(py: Python<'_>, chain: Vec<(String, f64, f64, f64)>) -> PyResult<Py<PyDict>> {
    let contracts: Vec<gam::terms::sae::inference::contracts::Contract> = chain
        .into_iter()
        .map(|(name, domain_radius, defect, lipschitz)| {
            gam::terms::sae::inference::contracts::Contract {
                name,
                domain_radius,
                defect,
                lipschitz,
            }
        })
        .collect();
    let report = gam::terms::sae::inference::contracts::compose_contracts(&contracts);
    let out = PyDict::new(py);
    out.set_item("total_defect", report.total_defect)?;
    out.set_item("per_stage_contribution", report.per_stage_contribution)?;
    out.set_item("domain_ok", report.domain_ok)?;
    Ok(out.unbind())
}

/// Candès–Fernández-Granda super-resolution separation threshold `Δ ≈ 2/H` for a
/// harmonic atom carrying `n_harmonics = H` harmonics: the minimum wrap-around
/// spike separation at which exact convex recovery is guaranteed.
#[pyfunction(signature = (n_harmonics,))]
fn separation_limit(n_harmonics: usize) -> f64 {
    gam::terms::sae::super_resolution::separation_limit(n_harmonics)
}

/// Recover the point masses `{(a_j, t_j)}` underlying a harmonic atom's Fourier
/// coefficients by the matrix-pencil / Prony method. `fourier_coeffs[h] =
/// (c_{h+1}, s_{h+1})` covers harmonics `1..H`; `sigma` is the per-component
/// coefficient-noise SD (`≤ 0` selects the noiseless numerical-rank path).
/// Returns the recovered spikes (position `t` + amplitude columns), the selected
/// model order, the fit residual, and the Hankel singular spectrum.
#[pyfunction(signature = (fourier_coeffs, sigma = 0.0))]
fn recover_spikes(
    py: Python<'_>,
    fourier_coeffs: Vec<(f64, f64)>,
    sigma: f64,
) -> PyResult<Py<PyDict>> {
    let recovery = detach_py_result(py, "recover_spikes", move || {
        gam::terms::sae::super_resolution::recover_spikes(&fourier_coeffs, sigma)
    })?;
    let n = recovery.spikes.len();
    let mut spike_t = Vec::with_capacity(n);
    let mut spike_amplitude = Vec::with_capacity(n);
    for spike in &recovery.spikes {
        spike_t.push(spike.t);
        spike_amplitude.push(spike.amplitude);
    }
    let out = PyDict::new(py);
    out.set_item("t", ndarray::Array1::from_vec(spike_t).into_pyarray(py))?;
    out.set_item(
        "amplitude",
        ndarray::Array1::from_vec(spike_amplitude).into_pyarray(py),
    )?;
    out.set_item("model_order", recovery.model_order)?;
    out.set_item("residual", recovery.residual)?;
    out.set_item(
        "hankel_singular_values",
        ndarray::Array1::from_vec(recovery.hankel_singular_values).into_pyarray(py),
    )?;
    Ok(out.unbind())
}

/// Compose a closed loop of circle isometries and report the net `O(2)` element.
/// `edges` are `(sign, angle)` (`sign = +1` rotation, `−1` reflection); `defects`
/// are the per-edge `O(2)` departures whose sum is the derived trivial-verdict
/// tolerance. Returns the net sign/angle and the measure-don't-latch triviality
/// verdict.
#[pyfunction(signature = (edges, defects))]
fn loop_holonomy(py: Python<'_>, edges: Vec<(i8, f64)>, defects: Vec<f64>) -> PyResult<Py<PyDict>> {
    let report = gam::terms::sae::inference::contracts::loop_holonomy(&edges, &defects);
    let out = PyDict::new(py);
    out.set_item("loop_len", report.loop_len)?;
    out.set_item("net_sign", report.net_sign)?;
    out.set_item("net_angle", report.net_angle)?;
    out.set_item("is_trivial", report.is_trivial)?;
    out.set_item("angle_tolerance", report.angle_tolerance)?;
    Ok(out.unbind())
}

/// Per-row influence contributions for the conditional coactivation probability
/// `P(gate_j active | gate_i active)` over the selected sample. `active_i` /
/// `active_j` are the two gate activity streams; `rows` selects the sampled row
/// indices and `likelihood_weights` their honesty weights. Closed form — the
/// engine computes every number, the FFI only marshals arrays/dict. Returns the
/// weighted conditional probability, gate-i active mass, the per-row influence
/// values `psi`, and the normalized weights.
#[pyfunction(signature = (active_i, active_j, rows, likelihood_weights))]
fn conditional_coactivation_influence(
    py: Python<'_>,
    active_i: Vec<bool>,
    active_j: Vec<bool>,
    rows: Vec<usize>,
    likelihood_weights: Vec<f64>,
) -> PyResult<Py<PyDict>> {
    let report = detach_py_result(py, "conditional_coactivation_influence", move || {
        gam::terms::sae::coactivation_conditionality::conditional_coactivation_influence_values(
            &active_i,
            &active_j,
            &rows,
            &likelihood_weights,
        )
    })?;
    let out = PyDict::new(py);
    out.set_item("conditional_probability", report.conditional_probability)?;
    out.set_item("active_mass_i", report.active_mass_i)?;
    out.set_item("psi", report.psi)?;
    out.set_item("normalized_weights", report.normalized_weights)?;
    Ok(out.unbind())
}

/// Weighted-Pearson coupling influence and its KL-robustness certificate over
/// the selected sample. Computes the closed-form influence values
/// `psi_i = g̃_i·h̃_i − (rho/2)(g̃_i² + h̃_i²)`, the pooled coupling `rho`, the
/// influence variance / mean-abs, the robustness radius `epsilon*`, and the
/// first-order worst-case coupling after an arbitrary KL-`epsilon` shift.
#[pyfunction(signature = (gate_i, gate_j, rows, likelihood_weights, epsilon = 0.0))]
fn coupling_robustness_certificate(
    py: Python<'_>,
    gate_i: Vec<f64>,
    gate_j: Vec<f64>,
    rows: Vec<usize>,
    likelihood_weights: Vec<f64>,
    epsilon: f64,
) -> PyResult<Py<PyDict>> {
    let (influence, certificate, worst_case) =
        detach_py_result(py, "coupling_robustness_certificate", move || {
            let influence =
                gam::terms::sae::coactivation_conditionality::coupling_influence_values(
                    &gate_i,
                    &gate_j,
                    &rows,
                    &likelihood_weights,
                )?;
            let certificate = influence.certificate();
            let worst_case = certificate.worst_case_coupling(epsilon)?;
            Ok((influence, certificate, worst_case))
        })?;
    let out = PyDict::new(py);
    out.set_item("rho", influence.rho)?;
    out.set_item("psi", influence.psi)?;
    out.set_item("normalized_weights", influence.normalized_weights)?;
    out.set_item("influence_variance", certificate.influence_variance)?;
    out.set_item("influence_mean_abs", certificate.influence_mean_abs)?;
    out.set_item(
        "robustness_radius_epsilon",
        certificate.robustness_radius_epsilon,
    )?;
    out.set_item("epsilon", epsilon)?;
    out.set_item("worst_case_coupling", worst_case)?;
    Ok(out.unbind())
}

/// Effect-weighted atom retention ledger. `variance[a]` is the optional
/// reconstruction charge evidence `(delta_deviance_nats, charge_nats)` for atom
/// `a` (the list length is the atom count); `firings` are streamed
/// `(atom, fisher_quadratic_kl_nats)` ablated-firing contributions that build the
/// Fisher local-KL effect ledger through the real streaming accumulator.
/// Retention is an OR of the variance margin (`delta_deviance − charge`) and the
/// Fisher-effect margin (mean local-KL over the per-atom one-degree BIC price).
/// Returns per-atom `{atom, variance, effect, retained_by_variance,
/// retained_by_effect, retained}` under `atoms`.
#[pyfunction(signature = (variance, firings))]
fn effect_weighted_retention(
    py: Python<'_>,
    variance: Vec<Option<(f64, f64)>>,
    firings: Vec<(usize, f64)>,
) -> PyResult<Py<PyDict>> {
    let evidence = detach_py_result(py, "effect_weighted_retention", move || {
        let atom_count = variance.len();
        let variance_evidence: Vec<Option<gam::terms::sae::effect_weight::VarianceChargeEvidence>> =
            variance
                .iter()
                .map(|entry| {
                    entry.map(|(delta_deviance, charge)| {
                        gam::terms::sae::effect_weight::VarianceChargeEvidence {
                            delta_deviance,
                            charge,
                        }
                    })
                })
                .collect();
        let mut accumulator =
            gam::terms::sae::effect_weight::StreamingFisherEffectAccumulator::new(atom_count);
        for (atom, fisher_quadratic_kl_nats) in &firings {
            accumulator.accumulate_firing_local_kl(*atom, *fisher_quadratic_kl_nats)?;
        }
        let effect = accumulator.finish();
        gam::terms::sae::effect_weight::effect_weighted_retention(&variance_evidence, &effect)
    })?;
    let atoms = pyo3::types::PyList::empty(py);
    for ev in &evidence {
        let d = PyDict::new(py);
        d.set_item("atom", ev.atom)?;
        d.set_item("retained_by_variance", ev.retained_by_variance)?;
        d.set_item("retained_by_effect", ev.retained_by_effect)?;
        d.set_item("retained", ev.retained)?;
        match ev.variance {
            Some(v) => {
                let vd = PyDict::new(py);
                vd.set_item("delta_deviance", v.delta_deviance)?;
                vd.set_item("charge", v.charge)?;
                vd.set_item("margin", v.margin())?;
                d.set_item("variance", vd)?;
            }
            None => d.set_item("variance", py.None())?,
        }
        match ev.effect {
            Some(e) => {
                let ed = PyDict::new(py);
                ed.set_item("atom", e.atom)?;
                ed.set_item(
                    "mean_fisher_quadratic_kl_nats",
                    e.mean_fisher_quadratic_kl_nats,
                )?;
                ed.set_item(
                    "max_fisher_quadratic_kl_nats",
                    e.max_fisher_quadratic_kl_nats,
                )?;
                ed.set_item("n_firings", e.n_firings)?;
                ed.set_item("threshold_nats", e.threshold_nats)?;
                ed.set_item("margin", e.margin())?;
                d.set_item("effect", ed)?;
            }
            None => d.set_item("effect", py.None())?,
        }
        atoms.append(d)?;
    }
    let out = PyDict::new(py);
    out.set_item("atoms", atoms)?;
    Ok(out.unbind())
}

/// Score chart-coordinate interpretability against cyclic ground-truth labels.
///
/// `observations` are `(recovered_turns, label_turns, weight)` triples: the
/// recovered chart coordinate and the ground-truth cyclic label, both in turns
/// (values are wrapped modulo one), plus a non-negative posterior/evidence weight
/// per row. `null_observation_draws` contains complete ledgers produced by the
/// closed `null_protocol`; the declared draw count must match the artifact. The
/// scorer is fail-closed: neither provenance nor null draws are optional.
#[pyfunction(signature = (observations, null_observation_draws, null_protocol, null_seed, expected_draws, significance_level))]
fn chart_interp_score(
    py: Python<'_>,
    observations: Vec<(f64, f64, f64)>,
    null_observation_draws: Vec<Vec<(f64, f64, f64)>>,
    null_protocol: String,
    null_seed: u64,
    expected_draws: usize,
    significance_level: f64,
) -> PyResult<Py<PyDict>> {
    let report = detach_py_result(py, "chart_interp_score", move || {
        let rows: Vec<gam::terms::sae::saebench_metrics::ChartInterpObservation> = observations
            .iter()
            .map(|&(recovered_turns, label_turns, weight)| {
                gam::terms::sae::saebench_metrics::ChartInterpObservation {
                    recovered_turns,
                    label_turns,
                    weight,
                }
            })
            .collect();
        let null_draws: Vec<Vec<gam::terms::sae::saebench_metrics::ChartInterpObservation>> =
            null_observation_draws
                .iter()
                .map(|draw| {
                    draw.iter()
                        .map(|&(recovered_turns, label_turns, weight)| {
                            gam::terms::sae::saebench_metrics::ChartInterpObservation {
                                recovered_turns,
                                label_turns,
                                weight,
                            }
                        })
                        .collect()
                })
                .collect();
        let protocol =
            gam::terms::sae::saebench_metrics::ChartInterpNullProtocol::parse(&null_protocol)?;
        let calibration =
            gam::terms::sae::saebench_metrics::ChartInterpNullCalibration::new(
                protocol,
                null_seed,
                expected_draws,
                null_draws,
            )?;
        gam::terms::sae::saebench_metrics::chart_interp_score(
            &rows,
            &calibration,
            significance_level,
        )
    })?;
    let observed = PyDict::new(py);
    observed.set_item(
        "circular_correlation",
        report.observed.circular_correlation,
    )?;
    observed.set_item(
        "signed_circular_correlation",
        report.observed.signed_circular_correlation,
    )?;
    observed.set_item("effective_weight", report.observed.effective_weight)?;

    let null = &report.calibration.null_distribution;
    let calibration = PyDict::new(py);
    calibration.set_item("statistic", report.calibration.statistic.as_str())?;
    calibration.set_item("protocol", report.calibration.protocol.as_str())?;
    calibration.set_item("null_kind", null.kind.as_str())?;
    calibration.set_item(
        "draw_policy",
        report.calibration.protocol.draw_policy().as_str(),
    )?;
    calibration.set_item("seed", report.calibration.seed)?;
    calibration.set_item("tail", null.tail.as_str())?;
    calibration.set_item("draws", null.n)?;
    calibration.set_item("observed_statistic", null.observed)?;
    calibration.set_item("mean", null.mean)?;
    calibration.set_item("sd", null.sd)?;
    calibration.set_item("min", null.min)?;
    calibration.set_item("q25", null.q25)?;
    calibration.set_item("median", null.median)?;
    calibration.set_item("q75", null.q75)?;
    calibration.set_item("max", null.max)?;
    calibration.set_item("z", null.z)?;
    calibration.set_item("p_value", null.p_value)?;
    calibration.set_item(
        "monte_carlo_standard_error",
        null.monte_carlo_standard_error,
    )?;
    calibration.set_item("extreme_draws", null.extreme_draws)?;
    calibration.set_item("null_statistics", &null.samples)?;

    let out = PyDict::new(py);
    out.set_item("statistic", report.statistic.as_str())?;
    out.set_item("observed", observed)?;
    out.set_item("calibration", calibration)?;
    out.set_item("significance_level", report.significance_level)?;
    out.set_item("verdict", report.verdict.as_str())?;
    Ok(out.unbind())
}

/// Fit the output-Fisher dose-response calibration ledger.
///
/// `observations` are `(arc_length, predicted_nats, measured_nats, weight)`
/// rows along a steered arc: the unit-speed path coordinate, the local
/// output-Fisher prediction in nats, the measured KL/behaviour change in nats,
/// and a non-negative weight. Returns `{slope_through_origin, r2_through_origin,
/// mean_measured_nats_per_arc_squared, cv_measured_nats_per_arc_squared,
/// effective_weight}` from
/// the audited [`saebench_metrics::dose_response_calibration`] — the #1942
/// dose-response calibration figure (through-origin slope + weighted R², plus
/// the unit-speed constancy kill-test via the nats-per-arc coefficient of
/// variation).
#[pyfunction(signature = (observations,))]
fn dose_response_calibration(
    py: Python<'_>,
    observations: Vec<(f64, f64, f64, f64)>,
) -> PyResult<Py<PyDict>> {
    let report = detach_py_result(py, "dose_response_calibration", move || {
        let rows: Vec<gam::terms::sae::saebench_metrics::DoseResponseObservation> = observations
            .iter()
            .map(|&(arc_length, predicted_nats, measured_nats, weight)| {
                gam::terms::sae::saebench_metrics::DoseResponseObservation {
                    arc_length,
                    predicted_nats,
                    measured_nats,
                    weight,
                }
            })
            .collect();
        gam::terms::sae::saebench_metrics::dose_response_calibration(&rows)
    })?;
    let out = PyDict::new(py);
    out.set_item("slope_through_origin", report.slope_through_origin)?;
    out.set_item("r2_through_origin", report.r2_through_origin)?;
    out.set_item(
        "mean_measured_nats_per_arc_squared",
        report.mean_measured_nats_per_arc_squared,
    )?;
    out.set_item(
        "cv_measured_nats_per_arc_squared",
        report.cv_measured_nats_per_arc_squared,
    )?;
    out.set_item("effective_weight", report.effective_weight)?;
    Ok(out.unbind())
}

/// Invert a row-Hessian precision block into a per-coordinate posterior.
///
/// `mean` is the posterior mean coordinate the fit/encoder supplies and
/// `precision_row_major` is the row-major `d x d` precision (inverse-covariance)
/// block that the arrow solve already factors for that chart coordinate.
/// Returns `{mean, covariance_diag, covariance_trace, precision_weight}` from
/// the audited [`saebench_metrics::coordinate_posterior_from_precision`]: the
/// per-token coordinate posterior uncertainty that weights both #1942
/// manifold-native metrics and tightens the steering validity radius.
#[pyfunction(signature = (mean, precision_row_major))]
fn coordinate_posterior_from_precision(
    py: Python<'_>,
    mean: Vec<f64>,
    precision_row_major: Vec<f64>,
) -> PyResult<Py<PyDict>> {
    let posterior = detach_py_result(py, "coordinate_posterior_from_precision", move || {
        gam::terms::sae::saebench_metrics::coordinate_posterior_from_precision(
            &mean,
            &precision_row_major,
        )
    })?;
    let out = PyDict::new(py);
    out.set_item("mean", posterior.mean)?;
    out.set_item("covariance_diag", posterior.covariance_diag)?;
    out.set_item("covariance_trace", posterior.covariance_trace)?;
    out.set_item("precision_weight", posterior.precision_weight)?;
    Ok(out.unbind())
}

#[cfg(test)]
mod ffi_completeness_tests {
    use super::*;

    #[test]
    fn conditional_coactivation_influence_surfaces_conditional_probability() {
        Python::attach(|py| {
            // Rows 0,1 fire gate i; only row 0 also fires gate j. With equal
            // weights, P(j|i) = joint_mass / active_mass_i = (1/3)/(2/3) = 0.5.
            let out = conditional_coactivation_influence(
                py,
                vec![true, true, false],
                vec![true, false, false],
                vec![0, 1, 2],
                vec![1.0, 1.0, 1.0],
            )
            .expect("conditional coactivation influence");
            let d = out.bind(py);
            let cp: f64 = d
                .get_item("conditional_probability")
                .unwrap()
                .unwrap()
                .extract()
                .unwrap();
            assert!((cp - 0.5).abs() < 1e-12, "conditional_probability = {cp}");
            let mass: f64 = d
                .get_item("active_mass_i")
                .unwrap()
                .unwrap()
                .extract()
                .unwrap();
            assert!((mass - 2.0 / 3.0).abs() < 1e-12, "active_mass_i = {mass}");
            let psi: Vec<f64> = d.get_item("psi").unwrap().unwrap().extract().unwrap();
            assert_eq!(psi.len(), 3);
        });
    }

    #[test]
    fn coupling_robustness_certificate_surfaces_worst_case_coupling() {
        Python::attach(|py| {
            // Perfectly correlated gate streams -> rho = 1; at epsilon = 0 the
            // worst-case coupling equals rho (no distribution shift budget).
            let out = coupling_robustness_certificate(
                py,
                vec![0.0, 1.0, 0.0, 1.0],
                vec![0.0, 1.0, 0.0, 1.0],
                vec![0, 1, 2, 3],
                vec![1.0, 1.0, 1.0, 1.0],
                0.0,
            )
            .expect("coupling robustness certificate");
            let d = out.bind(py);
            let rho: f64 = d.get_item("rho").unwrap().unwrap().extract().unwrap();
            assert!((rho - 1.0).abs() < 1e-9, "rho = {rho}");
            let wc: f64 = d
                .get_item("worst_case_coupling")
                .unwrap()
                .unwrap()
                .extract()
                .unwrap();
            assert!((wc - rho).abs() < 1e-12, "worst_case_coupling = {wc}");
            assert!(d.get_item("influence_variance").unwrap().is_some());
            assert!(d.get_item("robustness_radius_epsilon").unwrap().is_some());
        });
    }

    #[test]
    fn effect_weighted_retention_ors_variance_and_effect_margins() {
        Python::attach(|py| {
            // Atom 0: variance margin 2.0-0.5>0 and two firings of local-KL 1.0
            // clear the BIC price -> retained. Atom 1: no variance, one firing of
            // 0.01 below the price -> not retained.
            let out = effect_weighted_retention(
                py,
                vec![Some((2.0, 0.5)), None],
                vec![(0, 1.0), (0, 1.0), (1, 0.01)],
            )
            .expect("effect weighted retention");
            let d = out.bind(py);
            let atoms_any = d.get_item("atoms").unwrap().unwrap();
            let atoms = atoms_any.cast::<pyo3::types::PyList>().unwrap();
            assert_eq!(atoms.len(), 2);
            let a0 = atoms.get_item(0).unwrap();
            let a0 = a0.cast::<PyDict>().unwrap();
            let r0: bool = a0.get_item("retained").unwrap().unwrap().extract().unwrap();
            assert!(r0, "atom 0 should be retained");
            let rv0: bool = a0
                .get_item("retained_by_variance")
                .unwrap()
                .unwrap()
                .extract()
                .unwrap();
            assert!(rv0, "atom 0 retained by variance");
            let a1 = atoms.get_item(1).unwrap();
            let a1 = a1.cast::<PyDict>().unwrap();
            let r1: bool = a1.get_item("retained").unwrap().unwrap().extract().unwrap();
            assert!(!r1, "atom 1 should not be retained");
            assert!(a1.get_item("variance").unwrap().unwrap().is_none());
        });
    }

    #[test]
    fn chart_interp_score_quotients_orientation_over_the_ffi_boundary() {
        Python::attach(|py| {
            // Recovered coordinate runs backwards relative to the cyclic label;
            // the orientation-quotiented score still locks phase, and the signed
            // score records the reversal.
            let observations = vec![
                (0.99, 0.01, 1.0),
                (0.24, 0.76, 1.0),
                (0.49, 0.51, 1.0),
                (0.74, 0.26, 1.0),
            ];
            let out = chart_interp_score(
                py,
                observations.clone(),
                vec![observations],
                "matched_spectrum_gaussian_chart_refit_v1".to_string(),
                17,
                1,
                0.05,
            )
            .expect("chart interp score");
            let d = out.bind(py);
            let observed = d
                .get_item("observed")
                .unwrap()
                .unwrap()
                .cast_into::<PyDict>()
                .unwrap();
            let cc: f64 = observed
                .get_item("circular_correlation")
                .unwrap()
                .unwrap()
                .extract()
                .unwrap();
            assert!(cc > 0.99, "circular_correlation = {cc}");
            let signed: f64 = observed
                .get_item("signed_circular_correlation")
                .unwrap()
                .unwrap()
                .extract()
                .unwrap();
            assert!(signed < 0.0, "signed_circular_correlation = {signed}");
            let calibration = d
                .get_item("calibration")
                .unwrap()
                .unwrap()
                .cast_into::<PyDict>()
                .unwrap();
            let protocol: String = calibration
                .get_item("protocol")
                .unwrap()
                .unwrap()
                .extract()
                .unwrap();
            assert_eq!(protocol, "matched_spectrum_gaussian_chart_refit_v1");
            let seed: u64 = calibration
                .get_item("seed")
                .unwrap()
                .unwrap()
                .extract()
                .unwrap();
            assert_eq!(seed, 17);
            let samples: Vec<f64> = calibration
                .get_item("null_statistics")
                .unwrap()
                .unwrap()
                .extract()
                .unwrap();
            assert_eq!(samples.len(), 1);
            let verdict: String = d
                .get_item("verdict")
                .unwrap()
                .unwrap()
                .extract()
                .unwrap();
            assert_eq!(verdict, "null_compatible");
        });
    }

    #[test]
    fn dose_response_calibration_reports_slope_and_unit_speed_constancy() {
        Python::attach(|py| {
            let out = dose_response_calibration(
                py,
                vec![
                    (1.0, 0.5, 1.0, 1.0),
                    (2.0, 2.0, 4.0, 1.0),
                    (3.0, 4.5, 9.0, 1.0),
                ],
            )
            .expect("dose response calibration");
            let d = out.bind(py);
            let slope: f64 = d
                .get_item("slope_through_origin")
                .unwrap()
                .unwrap()
                .extract()
                .unwrap();
            assert!((slope - 2.0).abs() < 1e-9, "slope = {slope}");
            let cv: f64 = d
                .get_item("cv_measured_nats_per_arc_squared")
                .unwrap()
                .unwrap()
                .extract()
                .unwrap();
            assert!(cv < 1e-9, "unit-speed cv = {cv}");
        });
    }

    #[test]
    fn coordinate_posterior_inverts_precision_block_over_the_ffi_boundary() {
        Python::attach(|py| {
            let out =
                coordinate_posterior_from_precision(py, vec![0.25, 0.75], vec![4.0, 1.0, 1.0, 3.0])
                    .expect("coordinate posterior");
            let d = out.bind(py);
            let diag: Vec<f64> = d
                .get_item("covariance_diag")
                .unwrap()
                .unwrap()
                .extract()
                .unwrap();
            assert!((diag[0] - 3.0 / 11.0).abs() < 1e-9, "diag0 = {}", diag[0]);
            assert!((diag[1] - 4.0 / 11.0).abs() < 1e-9, "diag1 = {}", diag[1]);
            let pw: f64 = d
                .get_item("precision_weight")
                .unwrap()
                .unwrap()
                .extract()
                .unwrap();
            assert!((pw - 11.0 / 7.0).abs() < 1e-9, "precision_weight = {pw}");
        });
    }
}
