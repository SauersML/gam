#![cfg(test)]

use super::*;
use super::super::law::{JointLikelihood, JointSpecification};

fn strings(values: &[&str]) -> Vec<String> {
    values.iter().map(|v| v.to_string()).collect()
}

fn declarations() -> JointDeclarations {
    JointDeclarations {
        marks: Some(vec![
            ("flu".to_string(), MarkKind::Recurrent),
            ("diabetes".to_string(), MarkKind::Once),
            ("death".to_string(), MarkKind::Terminal),
            ("visit".to_string(), MarkKind::Recurrent),
        ]),
        channels: vec![
            ("hba1c".to_string(), "student_t".to_string()),
            ("smoker".to_string(), "probit:2".to_string()),
            ("prescriptions".to_string(), "negative_binomial".to_string()),
        ],
        score_names: strings(&["prs"]),
        visit_process: VisitProcess::Informative {
            mark: "visit".to_string(),
            event_date_state: EventDateState::BeforeEvents,
        },
        baseline_formula: "1".to_string(),
        population_formula: "1".to_string(),
        drive_formula: None,
        entry_formula: None,
        resolution: NodeResolution {
            quadrature_order: 1,
            state_width: Some(100.0),
        },
    }
}

/// Subject A enters at 50 with a diabetes record at 45, has two flu records,
/// misses the visit at 54, and at the visit at 51 has its smoking channel
/// unmeasured; at 58 it records a negative smoking answer. Subject B has a NaN
/// score, a prescription count with its exposure, a diabetes record inside
/// follow-up, and dies at its exit.
fn tables() -> JointTables {
    JointTables {
        subjects: SubjectTable {
            id: strings(&["A", "B"]),
            entry: vec![50.0, 40.0],
            exit: vec![60.0, 48.0],
        },
        events: EventTable {
            id: strings(&["A", "A", "A", "B", "B"]),
            time: vec![45.0, 52.0, 57.0, 44.0, 48.0],
            mark: strings(&["diabetes", "flu", "flu", "diabetes", "death"]),
        },
        covariates: CovariateTable {
            id: strings(&["A", "A", "B"]),
            start: vec![40.0, 55.0, 40.0],
            names: strings(&["bmi", "site"]),
            columns: vec![
                CovariateCells::Numbers(vec![25.0, 27.0, 31.0]),
                CovariateCells::Labels(strings(&["north", "north", "south"])),
            ],
        },
        measurements: MeasurementTable {
            id: strings(&["A", "A", "A", "B", "B"]),
            time: vec![51.0, 51.0, 58.0, 42.0, 42.0],
            channel: strings(&["hba1c", "smoker", "smoker", "hba1c", "prescriptions"]),
            value: vec![Some(6.1), None, Some(0.0), Some(7.0), Some(3.0)],
            exposure: vec![None, None, None, None, Some(0.5)],
        },
        visits: VisitTable {
            id: strings(&["A", "A", "A", "B"]),
            time: vec![51.0, 54.0, 58.0, 42.0],
            attended: vec![true, false, true, true],
        },
        genetics: GeneticTable {
            id: strings(&["A", "B"]),
            score_names: strings(&["prs"]),
            scores: vec![vec![Some(0.3), Some(f64::NAN)]],
        },
    }
}

/// The rows of one subject, in the same table order.
fn only(tables: &JointTables, id: &str) -> JointTables {
    fn keep<T: Clone>(ids: &[String], values: &[T], id: &str) -> Vec<T> {
        ids.iter()
            .zip(values)
            .filter(|(row, _)| row.as_str() == id)
            .map(|(_, value)| value.clone())
            .collect()
    }
    let c = &tables.covariates;
    let columns = c
        .columns
        .iter()
        .map(|cells| match cells {
            CovariateCells::Numbers(values) => CovariateCells::Numbers(keep(&c.id, values, id)),
            CovariateCells::Labels(values) => CovariateCells::Labels(keep(&c.id, values, id)),
        })
        .collect();
    let g = &tables.genetics;
    JointTables {
        subjects: SubjectTable {
            id: keep(&tables.subjects.id, &tables.subjects.id, id),
            entry: keep(&tables.subjects.id, &tables.subjects.entry, id),
            exit: keep(&tables.subjects.id, &tables.subjects.exit, id),
        },
        events: EventTable {
            id: keep(&tables.events.id, &tables.events.id, id),
            time: keep(&tables.events.id, &tables.events.time, id),
            mark: keep(&tables.events.id, &tables.events.mark, id),
        },
        covariates: CovariateTable {
            id: keep(&c.id, &c.id, id),
            start: keep(&c.id, &c.start, id),
            names: c.names.clone(),
            columns,
        },
        measurements: MeasurementTable {
            id: keep(&tables.measurements.id, &tables.measurements.id, id),
            time: keep(&tables.measurements.id, &tables.measurements.time, id),
            channel: keep(&tables.measurements.id, &tables.measurements.channel, id),
            value: keep(&tables.measurements.id, &tables.measurements.value, id),
            exposure: keep(&tables.measurements.id, &tables.measurements.exposure, id),
        },
        visits: VisitTable {
            id: keep(&tables.visits.id, &tables.visits.id, id),
            time: keep(&tables.visits.id, &tables.visits.time, id),
            attended: keep(&tables.visits.id, &tables.visits.attended, id),
        },
        genetics: GeneticTable {
            id: keep(&g.id, &g.id, id),
            score_names: g.score_names.clone(),
            scores: g.scores.iter().map(|column| keep(&g.id, column, id)).collect(),
        },
    }
}

/// What an encoded history says was observed after entry, read back as rows.
struct Decoded {
    events: Vec<(f64, String)>,
    visits: Vec<f64>,
    measurements: Vec<(f64, String, Option<f64>, Option<f64>)>,
}

fn decode(schema: &FrozenJointSchema, h: &JointHistory) -> Decoded {
    let mut events = Vec::new();
    let mut visits = Vec::new();
    for (&time, fired) in h.times.iter().zip(&h.events) {
        for &d in fired {
            if schema.mark_names[d] == "visit" {
                visits.push(time);
            } else {
                events.push((time, schema.mark_names[d].clone()));
            }
        }
    }
    let measurements = h
        .measurements
        .iter()
        .map(|r| {
            (
                h.times[r.node],
                schema.channels[r.channel].name.clone(),
                r.value,
                r.exposure,
            )
        })
        .collect();
    Decoded {
        events,
        visits,
        measurements,
    }
}

type Flat = (
    (Vec<f64>, Vec<(usize, f64)>, Vec<Vec<usize>>, Vec<bool>),
    (Array2<f64>, Array2<f64>, Array2<f64>, Vec<f64>),
    Vec<Option<f64>>,
    Vec<(usize, usize, Option<f64>, Option<f64>, bool)>,
);

fn flat(h: &JointHistory) -> Flat {
    (
        (
            h.times.clone(),
            h.points.iter().map(|p| (p.node, p.weight)).collect(),
            h.events.clone(),
            h.initially_at_risk.clone(),
        ),
        (
            h.baseline_design.clone(),
            h.population_design.clone(),
            h.drive_design.clone(),
            h.entry_design.clone(),
        ),
        h.genetics.clone(),
        h.measurements
            .iter()
            .map(|r| (r.node, r.channel, r.value, r.exposure, r.after_event))
            .collect(),
    )
}

#[test]
fn long_tables_round_trip_through_joint_histories_the_law_accepts() {
    let (schema, subjects) = FrozenJointSchema::fit(&declarations(), &tables()).unwrap();
    assert_eq!(
        subjects.iter().map(|s| s.id.as_str()).collect::<Vec<_>>(),
        ["A", "B"]
    );

    let a = &subjects[0].history;
    assert_eq!((a.times[0], *a.times.last().unwrap()), (50.0, 60.0));
    let decoded = decode(&schema, a);
    // The diabetes record at 45 is entry information, not a count node.
    assert_eq!(
        decoded.events,
        vec![(52.0, "flu".to_string()), (57.0, "flu".to_string())]
    );
    assert_eq!(a.initially_at_risk, vec![true, false, true, true]);
    // Attended visits are events of the visit mark; the missed visit adds
    // nothing, not even a node.
    assert_eq!(decoded.visits, vec![51.0, 58.0]);
    assert!(!a.times.contains(&54.0));
    // The unmeasured channel and the recorded negative stay different rows.
    assert_eq!(
        decoded.measurements,
        vec![
            (51.0, "hba1c".to_string(), Some(6.1), None),
            (51.0, "smoker".to_string(), None, None),
            (58.0, "smoker".to_string(), Some(0.0), None),
        ]
    );
    assert_eq!(a.genetics, vec![Some(0.3)]);

    let b = &subjects[1].history;
    let decoded = decode(&schema, b);
    assert_eq!(
        decoded.events,
        vec![(44.0, "diabetes".to_string()), (48.0, "death".to_string())]
    );
    assert_eq!(b.events.last().unwrap(), &vec![2]);
    assert_eq!(decoded.visits, vec![42.0]);
    // Only the count channel carries an exposure.
    assert_eq!(
        decoded.measurements,
        vec![
            (42.0, "hba1c".to_string(), Some(7.0), None),
            (42.0, "prescriptions".to_string(), Some(3.0), Some(0.5)),
        ]
    );
    assert_eq!(b.initially_at_risk, vec![true; 4]);
    // A NaN score is missing, never filled.
    assert_eq!(b.genetics, vec![None]);

    // Each subject's covariate changes inside follow-up: A changes at 55, B never.
    for (subject, changes) in subjects.iter().zip([vec![55.0_f64], Vec::new()]) {
        let h = &subject.history;
        // Two routes to each node's cell length t_n − t_{n−1}: its points' weights
        // summed in the encoder's order, and the rounded endpoint subtraction.
        // Running error (Higham, Accuracy and Stability, ch. 3): every rounded
        // operation errs by at most ε times the magnitude of its result, so the
        // summation contributes ε·μ with μ = Σ_j (|w_j| + |partial sum after j|).
        // Each weight is a rounded half piece width times the Gauss-Legendre weight,
        // adding ε·|w_j| for the product and ε·max(|a|, |b|) for its piece's
        // subtraction, and the cell length adds ε·max(|t_{n−1}|, |t_n|). The
        // fixture's order-1 rule has the weight 2 = 2/((1 − z²)·P₁′(z)²) with P₁′ ≡ 1
        // and a Newton root |z| ~ 1e-17, which rounds to exactly 2, so its certified
        // weight error is zero. A higher order would add
        // `gauss_legendre_certified(n)`'s weight error here.
        for n in 1..h.times.len() {
            let (left, right) = (h.times[n - 1], h.times[n]);
            let mut total = 0.0_f64;
            let mut mu = 0.0_f64;
            for weight in h.points.iter().filter(|p| p.node == n && p.weight > 0.0).map(|p| p.weight) {
                total += weight;
                mu += weight.abs() + total.abs() + weight.abs();
            }
            let mut cuts = vec![left];
            cuts.extend(changes.iter().copied().filter(|&c| c > left && c < right));
            cuts.push(right);
            let pieces: f64 = cuts.windows(2).map(|c| c[0].abs().max(c[1].abs())).sum();
            let length = right - left;
            let bar = f64::EPSILON * (mu + pieces + left.abs().max(right.abs()));
            assert!(length > bar);
            assert!((total - length).abs() <= bar, "node {n}: {total} vs {length}, bar {bar}");
        }
        // Sorted by node; each node's first point is its zero-weight anchor, and
        // the entry node owns nothing else.
        assert!(h.points.windows(2).all(|w| w[0].node <= w[1].node));
        for node in 0..h.times.len() {
            let first = h.points.iter().position(|p| p.node == node).unwrap();
            assert_eq!(h.points[first].weight, 0.0);
            // The anchor sits at its node time; the cell's points follow in time
            // order inside (t_{n−1}, t_n].
            assert_eq!(h.points[first].time, h.times[node]);
            let cell: Vec<f64> = h
                .points
                .iter()
                .filter(|p| p.node == node && p.weight > 0.0)
                .map(|p| p.time)
                .collect();
            assert!(cell.windows(2).all(|w| w[0] <= w[1]));
            if node > 0 {
                assert!(cell.iter().all(|&t| t > h.times[node - 1] && t <= h.times[node]));
            }
        }
        assert_eq!(h.points.iter().filter(|p| p.node == 0).count(), 1);
        assert_eq!(h.baseline_design.nrows(), h.points.len());
        assert_eq!(h.population_design.nrows(), h.points.len());
        assert!(h.population_design.column(0).iter().all(|&v| v == 1.0));
        assert_eq!(h.drive_design.dim(), (h.times.len() - 1, 0));
        // The declared weight error is the certified rule's, with its two roundings.
        assert_eq!(
            h.weight_error,
            declared_weight_error(&gam_math::special::gauss_legendre_certified(1)).unwrap()
        );
        let law = JointLikelihood::new(JointSpecification {
            signatures: 0,
            marks: schema.mark_kinds.clone(),
            baseline_columns: h.baseline_design.ncols(),
            population_columns: h.population_design.ncols(),
            drive_columns: 0,
            entry_columns: 0,
            // The encoder's frozen penalty blocks, through the law's own shape check.
            baseline_penalties: schema.baseline.penalties.clone(),
            drive_penalties: Vec::new(),
            population_penalties: schema.population.penalties.clone(),
            measurements: schema.channels.iter().map(|c| c.family.clone()).collect(),
            genetic_mean: vec![0.0],
            genetic_precision: Array2::eye(1),
        })
        .unwrap();
        assert_eq!(law.validate_history(h), Ok(()));
        // The law's check is not vacuous: weight moved onto a node's anchor keeps
        // the window's total but is refused.
        let anchor = h.points.iter().position(|p| p.node == 1).unwrap();
        let mut broken = h.clone();
        broken.points[anchor].weight = broken.points[anchor + 1].weight;
        broken.points[anchor + 1].weight = 0.0;
        assert!(law.validate_history(&broken).is_err());
    }
}

#[test]
fn latent_nodes_keep_every_date_and_split_gaps_to_the_state_width() {
    let times = latent_node_times(50.0, 60.0, [52.0, 51.0, 52.0, 45.0, 60.0], Some(3.0)).unwrap();
    // Dates inside follow-up stay nodes; a date at or outside the ends adds nothing.
    for date in [50.0, 51.0, 52.0, 60.0] {
        assert!(times.contains(&date));
    }
    assert!(!times.contains(&45.0));
    // (52, 60) is split into three equal gaps no wider than 3.
    assert_eq!(times.len(), 6);
    // The widest gap is 8/3, far inside the width, so no rounding bar enters.
    assert!(times.windows(2).all(|w| w[0] < w[1] && w[1] - w[0] <= 3.0));
    assert_eq!((times[2], times[5]), (52.0, 60.0));
    // A width wider than the follow-up, or no latent state, leaves only the dated nodes.
    assert_eq!(latent_node_times(50.0, 60.0, [51.0], Some(100.0)).unwrap(), vec![50.0, 51.0, 60.0]);
    assert_eq!(latent_node_times(50.0, 60.0, [51.0], None).unwrap(), vec![50.0, 51.0, 60.0]);
    for width in [0.0, f64::INFINITY, f64::NAN] {
        assert!(latent_node_times(50.0, 60.0, [51.0], Some(width)).is_err());
    }
}

#[test]
fn compensator_cells_belong_to_their_right_node_and_split_at_covariate_changes() {
    let nodes = [50.0, 51.0, 60.0];
    let events = vec![vec![], vec![0], vec![]];
    // The order-1 certified rule is exact: node 0, weight 2.
    let rule = gam_math::special::gauss_legendre_certified(1);
    let points = placed_points(&nodes, &events, &[40.0, 55.0], &rule);
    // Sorted by node, and each node's first point is its anchor.
    assert!(points.windows(2).all(|w| w[0].node <= w[1].node));
    for n in 0..nodes.len() {
        assert_eq!(points.iter().find(|p| p.node == n).unwrap().weight, 0.0);
    }
    // One zero-weight anchor per node at its time; an event node reads its
    // left limit.
    let anchors: Vec<&PlacedPoint> = points.iter().filter(|p| p.weight == 0.0).collect();
    assert_eq!(anchors.len(), 3);
    assert!(anchors.iter().all(|p| p.time == nodes[p.node]));
    assert_eq!(
        anchors.iter().map(|p| p.left_limit).collect::<Vec<_>>(),
        [false, true, false]
    );
    let cells = |n: usize| {
        points
            .iter()
            .filter(|p| p.node == n && p.weight > 0.0)
            .map(|p| (p.time, p.weight))
            .collect::<Vec<_>>()
    };
    // The entry node owns no cell; (50, 51] is one piece; (51, 60] splits at 55.
    assert!(cells(0).is_empty());
    assert_eq!(cells(1), [(50.5, 1.0)]);
    assert_eq!(cells(2), [(53.0, 4.0), (57.5, 5.0)]);
    for n in 1..nodes.len() {
        let total: f64 = cells(n).iter().map(|&(_, w)| w).sum();
        assert_eq!(total, nodes[n] - nodes[n - 1]);
    }
}

#[test]
fn a_declined_quadrature_certificate_is_refused_not_clamped() {
    let certified = gam_math::special::gauss_legendre_certified(5);
    assert!(certified.weight_relative_error.is_finite());
    let declared = declared_weight_error(&certified).unwrap();
    // Two half-ulp roundings per weight on top of the certified error.
    assert_eq!(declared, certified.weight_relative_error + f64::EPSILON);
    assert!(declared < 1.0);
    let mut declined = certified.clone();
    declined.weight_relative_error = f64::INFINITY;
    assert!(matches!(
        declared_weight_error(&declined),
        Err(JointDataError::QuadratureCertificate { order: 5, .. })
    ));
}

#[test]
fn measurement_families_parse_by_one_rule() {
    assert!(matches!(
        parse_measurement_family(" Probit:4 "),
        Ok(MeasurementFamily::Probit { categories: 4 })
    ));
    assert!(matches!(
        parse_measurement_family("student_t"),
        Ok(MeasurementFamily::StudentT)
    ));
    assert!(matches!(
        parse_measurement_family("probit:1"),
        Err(JointDataError::UnknownFamily { .. })
    ));
    assert!(matches!(
        parse_measurement_family("binary_probit"),
        Err(JointDataError::UnknownFamily { .. })
    ));
    assert!(matches!(
        parse_measurement_family("gaussian"),
        Err(JointDataError::UnknownFamily { .. })
    ));
}

#[test]
fn a_reloaded_schema_encodes_fit_and_served_histories_identically() {
    let tables = tables();
    let (schema, fitted) = FrozenJointSchema::fit(&declarations(), &tables).unwrap();
    let text = serde_json::to_string(&schema).unwrap();
    let reloaded: FrozenJointSchema = serde_json::from_str(&text).unwrap();
    let again = reloaded.encode(&tables).unwrap();
    assert_eq!(again.len(), fitted.len());
    for (fit, served) in fitted.iter().zip(&again) {
        assert_eq!(fit.id, served.id);
        assert_eq!(flat(&fit.history), flat(&served.history));
    }
    // One subject served alone encodes as it did inside the cohort.
    let alone = reloaded.encode(&only(&tables, "A")).unwrap();
    assert_eq!(alone.len(), 1);
    assert_eq!(flat(&alone[0].history), flat(&fitted[0].history));
}

#[test]
fn a_served_history_with_an_unknown_level_is_refused() {
    let (schema, _) = FrozenJointSchema::fit(&declarations(), &tables()).unwrap();
    let mut served = only(&tables(), "B");
    assert!(schema.encode(&served).is_ok());
    served.covariates.columns[1] = CovariateCells::Labels(strings(&["east"]));
    let error = schema.encode(&served).err();
    assert!(
        matches!(&error, Some(JointDataError::Cohort(EventHistoryError::InvalidInput { reason })) if reason.contains("unknown level")),
        "{error:?}"
    );
}

/// Fit the valid fixture (the positive control), apply one change, and return
/// the refusal, or `None` when the changed tables fit. Callers assert
/// `Some(pattern)`, so a check that stops refusing fails its own assertion.
fn refusal(change: impl FnOnce(&mut JointDeclarations, &mut JointTables)) -> Option<JointDataError> {
    let (mut declared, mut data) = (declarations(), tables());
    assert!(FrozenJointSchema::fit(&declared, &data).is_ok());
    change(&mut declared, &mut data);
    FrozenJointSchema::fit(&declared, &data).err()
}

/// Every float an encoded history carries, as bits: with [`flat`], equality is bitwise.
fn float_bits(h: &JointHistory) -> Vec<Option<u64>> {
    h.times
        .iter()
        .chain(h.points.iter().flat_map(|p| [&p.time, &p.weight]))
        .chain(h.baseline_design.iter())
        .chain(h.population_design.iter())
        .chain(h.drive_design.iter())
        .chain(&h.entry_design)
        .chain([&h.weight_error])
        .map(|v| Some(v.to_bits()))
        .chain(h.genetics.iter().map(|&v| v.map(f64::to_bits)))
        .chain(
            h.measurements
                .iter()
                .flat_map(|r| [r.value, r.exposure])
                .map(|v| v.map(f64::to_bits)),
        )
        .collect()
}

/// The fixture with rows that tie: a second hba1c row at A's visit on 51, a second
/// prescriptions row at B's visit on 42 differing only in its exposure, and a flu
/// record on B's diabetes date.
fn with_tied_rows() -> JointTables {
    let mut data = tables();
    let m = &mut data.measurements;
    for (id, time, channel, value, exposure) in [
        ("A", 51.0, "hba1c", Some(6.3), None),
        ("B", 42.0, "prescriptions", Some(3.0), Some(0.7)),
    ] {
        m.id.push(id.to_string());
        m.time.push(time);
        m.channel.push(channel.to_string());
        m.value.push(value);
        m.exposure.push(exposure);
    }
    data.events.id.push("B".to_string());
    data.events.time.push(44.0);
    data.events.mark.push("flu".to_string());
    data
}

/// Every table listed backwards, so rows that tie reverse among themselves too.
fn reversed(mut data: JointTables) -> JointTables {
    let s = &mut data.subjects;
    s.id.reverse();
    s.entry.reverse();
    s.exit.reverse();
    let e = &mut data.events;
    e.id.reverse();
    e.time.reverse();
    e.mark.reverse();
    let c = &mut data.covariates;
    c.id.reverse();
    c.start.reverse();
    for cells in &mut c.columns {
        match cells {
            CovariateCells::Numbers(values) => values.reverse(),
            CovariateCells::Labels(values) => values.reverse(),
        }
    }
    let m = &mut data.measurements;
    m.id.reverse();
    m.time.reverse();
    m.channel.reverse();
    m.value.reverse();
    m.exposure.reverse();
    let v = &mut data.visits;
    v.id.reverse();
    v.time.reverse();
    v.attended.reverse();
    let g = &mut data.genetics;
    g.id.reverse();
    for column in &mut g.scores {
        column.reverse();
    }
    data
}

/// Each subject's encoding, bitwise, ordered by id.
fn by_id(subjects: &[EncodedSubject]) -> Vec<(String, Flat, Vec<Option<u64>>)> {
    let mut rows: Vec<_> = subjects
        .iter()
        .map(|s| (s.id.clone(), flat(&s.history), float_bits(&s.history)))
        .collect();
    rows.sort_by(|a, b| a.0.cmp(&b.0));
    rows
}

#[test]
fn rows_in_any_order_encode_identically() {
    // Every record carries its own time, so no permutation of a table's rows changes an
    // encoding. Here every table is listed backwards, and rows that tie on a date reverse
    // among themselves.
    let (schema, ordered) = FrozenJointSchema::fit(&declarations(), &with_tied_rows()).unwrap();
    let reordered = FrozenJointSchema::fit(&declarations(), &reversed(with_tied_rows()));
    // A lost sort either refuses the reversed tables or changes their encoding; both fail here.
    assert!(reordered.is_ok(), "{:?}", reordered.as_ref().err());
    let (reordered_schema, reordered) = reordered.unwrap();
    assert_eq!(schema.covariate_levels, reordered_schema.covariate_levels);
    assert_eq!(by_id(&ordered), by_id(&reordered));
    // The ties are there, in canonical order: records by time, channel, value and
    // exposure, and a node's marks by index.
    let a = decode(&schema, &ordered[0].history);
    assert_eq!(
        a.measurements,
        vec![
            (51.0, "hba1c".to_string(), Some(6.1), None),
            (51.0, "hba1c".to_string(), Some(6.3), None),
            (51.0, "smoker".to_string(), None, None),
            (58.0, "smoker".to_string(), Some(0.0), None),
        ]
    );
    let b = decode(&schema, &ordered[1].history);
    assert_eq!(
        b.measurements,
        vec![
            (42.0, "hba1c".to_string(), Some(7.0), None),
            (42.0, "prescriptions".to_string(), Some(3.0), Some(0.5)),
            (42.0, "prescriptions".to_string(), Some(3.0), Some(0.7)),
        ]
    );
    assert_eq!(
        b.events,
        vec![
            (44.0, "flu".to_string()),
            (44.0, "diabetes".to_string()),
            (48.0, "death".to_string()),
        ]
    );
    // Positive control: moving a record's time does change the encoding.
    let mut moved = with_tied_rows();
    moved.events.time[1] = 53.0;
    let (_, changed) = FrozenJointSchema::fit(&declarations(), &moved).unwrap();
    assert_ne!(flat(&ordered[0].history), flat(&changed[0].history));
}

#[test]
fn an_undeclared_channel_is_refused() {
    let error = refusal(|_, t| t.measurements.channel[0] = "ldl".to_string());
    assert!(
        matches!(&error, Some(JointDataError::UnknownChannel { channel, .. }) if channel == "ldl"),
        "{error:?}"
    );
}

#[test]
fn a_value_outside_its_familys_support_is_refused() {
    let error = refusal(|_, t| t.measurements.value[2] = Some(0.5));
    assert!(
        matches!(&error, Some(JointDataError::OutsideSupport { channel, value, .. }) if channel == "smoker" && *value == 0.5),
        "{error:?}"
    );
    let error = refusal(|_, t| t.measurements.value[0] = Some(f64::NAN));
    assert!(
        matches!(&error, Some(JointDataError::OutsideSupport { channel, value, .. }) if channel == "hba1c" && value.is_nan()),
        "{error:?}"
    );
}

#[test]
fn a_measurement_at_a_missed_visit_is_refused() {
    let error = refusal(|_, t| t.measurements.time[2] = 54.0);
    assert!(
        matches!(&error, Some(JointDataError::MeasurementWithoutVisit { subject, time }) if subject == "A" && *time == 54.0),
        "{error:?}"
    );
}

#[test]
fn events_recorded_at_one_time_share_one_node() {
    let mut data = tables();
    // A flu record on the visit date: the node carries both marks, unordered.
    data.events.time[1] = 51.0;
    let (schema, subjects) = FrozenJointSchema::fit(&declarations(), &data).unwrap();
    let a = &subjects[0].history;
    let node = a.times.iter().position(|&t| t == 51.0).unwrap();
    let mut fired: Vec<&str> = a.events[node]
        .iter()
        .map(|&d| schema.mark_names[d].as_str())
        .collect();
    fired.sort_unstable();
    assert_eq!(fired, ["flu", "visit"]);
    // The date is one latent node, whose first point is its zero-weight anchor.
    assert_eq!(a.times.iter().filter(|&&t| t == 51.0).count(), 1);
    assert_eq!(a.points.iter().find(|p| p.node == node).unwrap().weight, 0.0);
}

#[test]
fn a_count_channel_needs_an_exposure_and_no_other_channel_takes_one() {
    let error = refusal(|_, t| t.measurements.exposure[4] = None);
    assert!(
        matches!(&error, Some(JointDataError::Exposure { subject, channel, .. }) if subject == "B" && channel == "prescriptions"),
        "{error:?}"
    );
    let error = refusal(|_, t| t.measurements.exposure[4] = Some(0.0));
    assert!(
        matches!(&error, Some(JointDataError::Exposure { channel, .. }) if channel == "prescriptions"),
        "{error:?}"
    );
    let error = refusal(|_, t| t.measurements.exposure[3] = Some(1.0));
    assert!(
        matches!(&error, Some(JointDataError::Exposure { channel, .. }) if channel == "hba1c"),
        "{error:?}"
    );
}

#[test]
fn a_subject_without_a_genetics_row_has_every_score_missing() {
    // Positive control: A's score is read from its row.
    let mut data = tables();
    let (_, subjects) = FrozenJointSchema::fit(&declarations(), &data).unwrap();
    assert_eq!(subjects[0].history.genetics, vec![Some(0.3)]);
    // Only B's row goes: B's score is missing, never filled, and A's is unchanged.
    let g = &mut data.genetics;
    g.id.truncate(1);
    g.scores[0].truncate(1);
    let (_, subjects) = FrozenJointSchema::fit(&declarations(), &data).unwrap();
    assert_eq!(subjects[0].history.genetics, vec![Some(0.3)]);
    assert_eq!(subjects[1].history.genetics, vec![None]);
}

#[test]
fn frozen_bases_carry_their_penalties_as_basis_width_blocks() {
    let mut declared = declarations();
    declared.population_formula = "s(time, double_penalty=true)".to_string();
    let (schema, subjects) = FrozenJointSchema::fit(&declared, &tables()).unwrap();
    let width = subjects[0].history.population_design.ncols();
    assert!(width > 1);
    assert!(!schema.population.penalties.is_empty());
    let mut ranks: Vec<(std::ops::Range<usize>, usize)> = Vec::new();
    for penalty in &schema.population.penalties {
        let span = penalty.columns.len();
        assert_eq!(penalty.local.dim(), (span, span));
        assert!(penalty.columns.end <= width);
        // The unit constant is in no penalized range.
        assert!(penalty.columns.start >= 1);
        assert!(penalty.rank >= 1 && penalty.rank <= span);
        assert!(penalty.local.iter().any(|&v| v != 0.0));
        match ranks.iter_mut().find(|(columns, _)| *columns == penalty.columns) {
            Some((_, total)) => *total += penalty.rank,
            None => ranks.push((penalty.columns.clone(), penalty.rank)),
        }
    }
    // A double-penalized smooth: its blocks' declared ranks fill its range,
    // and every non-constant column sits in a penalized range.
    for (columns, total) in &ranks {
        assert_eq!(*total, columns.len());
    }
    let covered: usize = ranks.iter().map(|(columns, _)| columns.len()).sum();
    assert_eq!(covered, width - 1);
    // An intercept-only basis has nothing to penalize.
    assert!(schema.baseline.penalties.is_empty());
}

/// The fixture with a flu record on B's diabetes date, listed before or after
/// the diabetes row.
fn with_tied_flu(before_diabetes: bool) -> JointTables {
    let mut data = tables();
    let at = if before_diabetes { 3 } else { 4 };
    data.events.id.insert(at, "B".to_string());
    data.events.time.insert(at, 44.0);
    data.events.mark.insert(at, "flu".to_string());
    data
}

#[test]
fn a_date_encodes_identically_whatever_order_its_rows_list_its_events() {
    let (schema, first) = FrozenJointSchema::fit(&declarations(), &with_tied_flu(true)).unwrap();
    let (_, second) = FrozenJointSchema::fit(&declarations(), &with_tied_flu(false)).unwrap();
    assert_eq!(flat(&first[1].history), flat(&second[1].history));
    let b = &first[1].history;
    let node = b.times.iter().position(|&t| t == 44.0).unwrap();
    let fired: Vec<&str> = b.events[node]
        .iter()
        .map(|&d| schema.mark_names[d].as_str())
        .collect();
    assert_eq!(fired, ["flu", "diabetes"]);
    assert_eq!(b.points.iter().find(|p| p.node == node).unwrap().weight, 0.0);
}

#[test]
fn two_rows_of_a_recurrent_mark_on_one_date_are_two_events() {
    let mut data = tables();
    // A second flu record on A's flu date 52.
    data.events.id.insert(2, "A".to_string());
    data.events.time.insert(2, 52.0);
    data.events.mark.insert(2, "flu".to_string());
    let (schema, subjects) = FrozenJointSchema::fit(&declarations(), &data).unwrap();
    let a = &subjects[0].history;
    let node = a.times.iter().position(|&t| t == 52.0).unwrap();
    let fired: Vec<&str> = a.events[node]
        .iter()
        .map(|&d| schema.mark_names[d].as_str())
        .collect();
    assert_eq!(fired, ["flu", "flu"]);
    // A once-only mark cannot repeat, on one date or across dates.
    let mut data = tables();
    data.events.id.insert(4, "B".to_string());
    data.events.time.insert(4, 44.0);
    data.events.mark.insert(4, "diabetes".to_string());
    assert!(FrozenJointSchema::fit(&declarations(), &data).is_err());
}

#[test]
fn the_visit_contract_declares_what_a_measurement_on_an_event_date_observes() {
    let (_, before) = FrozenJointSchema::fit(&declarations(), &tables()).unwrap();
    assert!(before
        .iter()
        .flat_map(|s| &s.history.measurements)
        .all(|r| !r.after_event));
    let mut declared = declarations();
    declared.visit_process = VisitProcess::Informative {
        mark: "visit".to_string(),
        event_date_state: EventDateState::AfterEvents,
    };
    let (_, after) = FrozenJointSchema::fit(&declared, &tables()).unwrap();
    // Every fixture measurement sits on its visit's node.
    for subject in &after {
        assert!(!subject.history.measurements.is_empty());
        for record in &subject.history.measurements {
            assert!(record.after_event);
            assert!(!subject.history.events[record.node].is_empty());
        }
    }
    // A visit on B's death date with a measurement: after termination there
    // is no state to observe, while the pre-event state is observable.
    let mut data = tables();
    data.visits.id.push("B".to_string());
    data.visits.time.push(48.0);
    data.visits.attended.push(true);
    let m = &mut data.measurements;
    m.id.push("B".to_string());
    m.time.push(48.0);
    m.channel.push("hba1c".to_string());
    m.value.push(Some(6.5));
    m.exposure.push(None);
    assert!(FrozenJointSchema::fit(&declarations(), &data).is_ok());
    let error = FrozenJointSchema::fit(&declared, &data).err();
    assert!(
        matches!(&error, Some(JointDataError::MeasurementAfterTermination { subject, time }) if subject == "B" && *time == 48.0),
        "{error:?}"
    );
}

#[test]
fn a_model_without_covariates_needs_no_covariate_table() {
    let mut data = tables();
    data.covariates = CovariateTable::default();
    let (_, subjects) = FrozenJointSchema::fit(&declarations(), &data).unwrap();
    assert_eq!(subjects.len(), 2);
    for subject in &subjects {
        let h = &subject.history;
        assert_eq!(h.baseline_design.nrows(), h.points.len());
        assert!(h.population_design.column(0).iter().all(|&v| v == 1.0));
    }
    // Declared columns without rows leave every subject without a segment.
    let mut data = tables();
    data.covariates.id.clear();
    data.covariates.start.clear();
    data.covariates.columns = vec![
        CovariateCells::Numbers(Vec::new()),
        CovariateCells::Labels(Vec::new()),
    ];
    assert!(FrozenJointSchema::fit(&declarations(), &data).is_err());
}

#[test]
fn the_identity_and_shape_refusals_the_surfaces_no_longer_repeat() {
    // The CLI, pyffi and gamfit used to check these themselves; the encoder now owns
    // every one, naming the offender. The fixture itself is the accepted neighbour.
    let error = refusal(|_, t| t.subjects.id[1] = "A".to_string());
    assert!(
        matches!(&error, Some(JointDataError::DuplicateSubject { table: "subjects", id }) if id == "A"),
        "{error:?}"
    );
    let error = refusal(|_, t| t.events.id[3] = "Z".to_string());
    assert!(
        matches!(&error, Some(JointDataError::UnknownSubject { table: "events", id }) if id == "Z"),
        "{error:?}"
    );
    let error = refusal(|_, t| {
        t.events.time.pop();
    });
    assert!(
        matches!(&error, Some(JointDataError::ColumnLength { table: "events", column, .. }) if column == "time"),
        "{error:?}"
    );
    let error = refusal(|d, _| d.channels.push(("hba1c".to_string(), "student_t".to_string())));
    assert!(
        matches!(&error, Some(JointDataError::DuplicateChannel { channel }) if channel == "hba1c"),
        "{error:?}"
    );
    let error = refusal(|_, t| t.covariates.names.push("region".to_string()));
    assert!(
        matches!(&error, Some(JointDataError::CovariateColumns { .. })),
        "{error:?}"
    );
    // Attendance comes from the visits table alone; the events table may not also record it.
    let error = refusal(|_, t| {
        t.events.id.insert(3, "A".to_string());
        t.events.time.insert(3, 58.0);
        t.events.mark.insert(3, "visit".to_string());
    });
    assert!(
        matches!(&error, Some(JointDataError::VisitMark { mark, reason }) if mark == "visit" && reason.contains("records it too")),
        "{error:?}"
    );
}

#[test]
fn records_a_rank_zero_history_could_not_hold_are_refused_by_the_cohort_rules() {
    // These were JointSpecification::history's own refusals. The encoder reaches each through
    // EventHistoryCohort::validate, as JointDataError::Cohort, named by its message.
    let cohort_refusal = |error: &Option<JointDataError>, reason: &str| {
        matches!(error, Some(JointDataError::Cohort(EventHistoryError::InvalidInput { reason: text })) if text.contains(reason))
    };
    // A terminal event before the exit.
    let error = refusal(|_, t| t.events.time[4] = 46.0);
    assert!(cohort_refusal(&error, "must end follow-up"), "{error:?}");
    // A once-only mark recorded twice.
    let error = refusal(|_, t| {
        t.events.id.insert(4, "B".to_string());
        t.events.time.insert(4, 45.0);
        t.events.mark.insert(4, "diabetes".to_string());
    });
    assert!(cohort_refusal(&error, "can fire at most once"), "{error:?}");
    // Two terminal events at the exit.
    let error = refusal(|_, t| {
        t.events.id.push("B".to_string());
        t.events.time.push(48.0);
        t.events.mark.push("death".to_string());
    });
    assert!(cohort_refusal(&error, "can fire at most once"), "{error:?}");
    // An event past the exit. Only the cohort refuses it, before encoding: the encoder places an
    // event at the first node at or after its date, and past the exit there is none, so without
    // this refusal the encoder would index out of bounds and panic before this assertion.
    let error = refusal(|_, t| {
        t.events.id.insert(3, "A".to_string());
        t.events.time.insert(3, 61.0);
        t.events.mark.insert(3, "flu".to_string());
    });
    assert!(cohort_refusal(&error, "after its exit"), "{error:?}");
    // A terminal event at the entry.
    let error = refusal(|_, t| {
        t.events.id.insert(1, "A".to_string());
        t.events.time.insert(1, 50.0);
        t.events.mark.insert(1, "death".to_string());
    });
    assert!(cohort_refusal(&error, "no follow-up to model"), "{error:?}");
    // Entry equal to exit, on a table with nothing else to refuse first.
    let empty_window = JointTables {
        subjects: SubjectTable {
            id: strings(&["C"]),
            entry: vec![5.0],
            exit: vec![5.0],
        },
        ..JointTables::default()
    };
    let mut declared = declarations();
    declared.score_names.clear();
    let error = FrozenJointSchema::fit(&declared, &empty_window).err();
    assert!(cohort_refusal(&error, "needs finite entry < exit"), "{error:?}");
}

#[test]
fn an_informative_visit_mark_must_be_declared_recurrent() {
    let error = refusal(|d, _| {
        if let Some(marks) = d.marks.as_mut() {
            marks[3].1 = MarkKind::Once;
        }
    });
    assert!(
        matches!(&error, Some(JointDataError::VisitMark { mark, .. }) if mark == "visit"),
        "{error:?}"
    );
}
