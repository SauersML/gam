//! #2263 item 3 — **requested-vs-realized chart displacement.**
//!
//! The issue reports that month steers overshoot: requested advances of
//! `+1..+6` realized a mode of `+8`. Nothing in the tree measures that. The
//! dose harness next door ([`crate::inference::tests_dose_calibration_2249`])
//! pins requested-vs-realized **nats**; the probe runner banks realized
//! **nats**. Realized *position* — did the row end up where the caller asked
//! it to go — has no reader anywhere, which is why a calibration defect of
//! this shape could persist across a whole study without a test noticing.
//!
//! This module closes that gap on the half of the loop that lives in this
//! repository. A steer is two objects:
//!
//! 1. the **chart round trip** — read `t_from`, write `t_to`, re-encode the
//!    edited row, compare. Entirely ours: [`set_coordinate`] composes
//!    [`EncodeAtlas::certified_encode_row`] with [`steer_delta`], and no model
//!    participates;
//! 2. the **behavioral readout** — which month the LLM then says. Not ours.
//!
//! The issue's "+1..+6 → +8" is measured on (2), so it cannot be reproduced
//! here. What CAN be settled here, and is settled below, is whether (1) is
//! innocent — because if the chart round trip already overshoots, the defect is
//! in this repository and no model is needed to find it, and if it is exact
//! then item 3 is a statement about the readout and every fix aimed at the
//! chart arithmetic would be aimed at the wrong object.
//!
//! ## The pre-registered third outcome
//!
//! `amplitude` in [`steer_delta`] multiplies the whole chord —
//! `delta = amplitude · (g(t_to) − g(t_from))`. It is the row's **expression
//! intensity**, the `a` in `a·g(t)`, not a dose knob. The study that produced
//! item 3 reached behavioral effect only by sweeping `alpha` to **16**
//! (item 1), i.e. by using the intensity as a dose knob. So there are three
//! possible outcomes, and they are distinguishable before the measurement:
//!
//! * **exact at every amplitude** ⇒ the chart path is innocent and item 3 is
//!   purely a readout phenomenon;
//! * **exact at the row's own amplitude, overshooting in proportion to
//!   `alpha/a`** ⇒ item 1 and item 3 are one defect — writing a position with
//!   the wrong intensity moves the row further than asked, and the "+8 mode" is
//!   the intensity mismatch, not a gain constant;
//! * **overshooting even at the row's own amplitude** ⇒ a genuine chart-side
//!   calibration bug, which would be new.
//!
//! Writing the three down first is what stops whichever one lands from being
//! read as confirmation.
//!
//! ## MEASURED: the second outcome, and items 1 and 3 are one defect
//!
//! Requested `+1` month, written at intensity `alpha`, over 96 rows:
//!
//! | alpha | 1 | 2 | 4 | 8 | 16 |
//! |---|---|---|---|---|---|
//! | realized | +0.9997 | +1.7927 | +2.5642 | +3.0341 | +3.2712 |
//!
//! At the row's own intensity the steer is exact. At the `alpha = 16` item 1
//! says the study needed for any behavioural effect, a one-month request
//! realizes **+3.27 months, and the response SATURATES** — `1→2` nearly doubles
//! the move, `8→16` barely changes it.
//!
//! That is item 3's signature. "Requested +1..+6 realizes a mode of +8" is an
//! overshoot whose realized value is roughly *constant across requests*, which
//! is what a flattened map produces: different requests compress onto the same
//! displacement and a confusion matrix collapses onto a mode.
//!
//! The mechanism reads off the code. `steer_delta` computes
//! `delta = amplitude · (g(t_to) − g(t_from))`, so `amplitude` is the row's
//! **expression intensity** — the `a` in `a·g(t)` — not a magnitude control.
//! Re-encoding `x + 16·(g_to − g_from)` lands wherever that off-manifold row
//! projects back onto the chart, which is a bounded function of the chord and
//! therefore saturates by construction.
//!
//! **The chart-arithmetic negative result is what makes this attributable.**
//! The round trip being exact at `alpha = 1` — and wrapping rather than
//! saturating out to `+18` — is what licenses assigning the whole overshoot to
//! writing at the wrong intensity, instead of splitting it between the chart
//! and the dose.
//!
//! ~~Predicted: `steer_to_target_nats` should not overshoot, because it solves
//! for the amplitude that lands a requested *dose* rather than assuming
//! amplitude scales position.~~ **That prediction is FALSE, and false by
//! construction rather than by measurement** — it needed no run to refute.
//!
//! `steer_to_target_nats` destructures `t_from` / `t_to` out of its request once
//! and passes them **unchanged** into every `steer_delta` call (the unit
//! reference and the `plan_at` closure the secant drives). It never re-solves
//! `t_to`. Since `steer_delta` is `delta = amplitude · (g(t_to) − g(t_from))`,
//! the reachable set of every plan that API can return — seed or post-secant —
//! is the ray `{a·dg : a > 0}` on the SAME chord. That is precisely the
//! one-parameter family the amplitude sweep above already traverses, under the
//! change of variable `a₀ = sqrt(q* / unit_nats)`: the `alpha` column
//! `{1,2,4,8,16}` IS the target-dose response at
//! `q* = {1,4,16,64,256}·unit_nats`.
//!
//! So the target-dose loop cannot repair the displacement. It selects `a` to
//! land a requested DOSE, and the realized DISPLACEMENT is then `a·dg`, exact
//! only at `a = 1`, i.e. only when `q*` happens to equal `unit_nats`. **Dose and
//! displacement are two demands on one scalar**, which means a fix has to solve
//! jointly for `(t_to, a)` rather than taking `t_to` literally.
//!
//! Credit: refuted by another lane reading the call graph at `669d59532`;
//! verified here against `origin/main` before this note was written.
//!
//! (This paragraph is the one the landing commit `ca38dea1b` lost: backticks in
//! its `-m` were command-substituted by the shell, so the sentence naming
//! `amplitude` came out empty. The finding lives here, where a reader of the
//! code will actually meet it.)

#[cfg(test)]
mod tests {
    use crate::encode::{AtlasConfig, EncodeAtlas};
    use crate::manifold::{
        SaeFisherRowMetricRequest, SaeFitAssignmentKind, SaeFitConfig, SaeFitSeedReport,
        SaeFitSeedRequest, SaeManifoldTerm, SaeMinimalSeedReport, SaeMinimalSeedRequest,
        build_sae_fit_seed, build_sae_minimal_seed,
    };
    use gam_terms::analytic_penalties::AnalyticPenaltyRegistry;
    use ndarray::{Array2, Array3};

    /// Twelve chart points, one per month — the fixture the issue's claim is
    /// stated in. The embedding carries harmonics 1..3 so the decoder is a
    /// genuinely curved map rather than a rotation in disguise.
    /// Twelve month LABELS on the circle, but the fit needs more rows than
    /// labels: with one row per month the seeded atom's coordinates collapsed to
    /// four distinct values spanning 0.75 rad — a chart covering 1.4 months of a
    /// 12-month circle. A displacement measured on that chart is meaningless,
    /// and the collapse was invisible until `[#2263 encode-control]` printed the
    /// fitted coordinates. `fixture_chart_spans_the_circle` now refuses it.
    const N_MONTHS: usize = 12;
    const N_ROWS: usize = 96;
    const P_OUT: usize = 6;
    const NOISE_SIGMA: f64 = 0.01;

    fn lcg(state: &mut u64) -> f64 {
        *state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((*state >> 11) as f64) / ((1u64 << 53) as f64)
    }
    fn lcg_normal(state: &mut u64) -> f64 {
        let u1 = lcg(state).max(1e-12);
        let u2 = lcg(state);
        (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
    }

    fn month_embedding_target() -> Array2<f64> {
        let mut state = 0x2263_0000_0000_0003u64;
        Array2::from_shape_fn((N_ROWS, P_OUT), |(i, j)| {
            let theta = std::f64::consts::TAU * (i as f64) / (N_ROWS as f64);
            let harmonic = (j / 2 + 1) as f64;
            let clean = if j % 2 == 0 {
                (harmonic * theta).cos()
            } else {
                (harmonic * theta).sin()
            };
            clean + NOISE_SIGMA * lcg_normal(&mut state)
        })
    }

    /// A genuine K=1 periodic-atom term over the month circle, with a behavioral
    /// metric installed so `steer_delta` takes its dose-bearing path (the
    /// displacement question is independent of the dose, but the metric-free
    /// path is a different branch and this measures the shipped one).
    fn build_month_term() -> (SaeManifoldTerm, Array2<f64>) {
        let target = month_embedding_target();
        let assignment_kind = SaeFitAssignmentKind::Softmax;
        let minimal = build_sae_minimal_seed(SaeMinimalSeedRequest {
            target: target.view(),
            atom_basis: vec!["periodic".to_string()],
            atom_dim: vec![1],
            assignment_kind,
            alpha: 1.0,
            tau: 1.0,
            threshold: 0.0,
            top_k: None,
            random_state: 0,
            initial_logits: None,
            initial_coords: None,
        })
        .expect("minimal seed on the month circle");
        let SaeMinimalSeedReport {
            geometry_plans,
            basis_values,
            basis_jacobian,
            decoder_coefficients,
            smooth_penalties,
            initial_logits,
            initial_coords,
            refine_routing,
        } = minimal;

        let identity_u = Array3::<f64>::from_shape_fn(
            (N_ROWS, P_OUT, 1),
            |(_, i, _)| if i == 0 { 1.0 } else { 0.0 },
        );
        let metric_request = SaeFisherRowMetricRequest::from_tag(
            identity_u.view(),
            N_ROWS,
            P_OUT,
            None,
            Some("uncertified_approximation"),
            None,
        )
        .expect("behavioral metric request");

        let registry = AnalyticPenaltyRegistry::new();
        let seed = build_sae_fit_seed(SaeFitSeedRequest {
            target: target.view(),
            geometry_plans: &geometry_plans,
            basis_values: basis_values.view(),
            basis_jacobian: basis_jacobian.view(),
            decoder_coefficients: decoder_coefficients.view(),
            smooth_penalties: smooth_penalties.view(),
            initial_logits: initial_logits.view(),
            initial_coords: initial_coords.view(),
            alpha: 1.0,
            tau: 1.0,
            learnable_alpha: false,
            assignment_kind,
            sparsity_strength: 1.0,
            smoothness: 1.0,
            max_iter: 40,
            learning_rate: 1.0,
            ridge_ext_coord: 1.0e-6,
            ridge_beta: 1.0e-6,
            top_k: None,
            threshold: 0.0,
            native_ard_enabled: true,
            seed_refine_routing: refine_routing,
            seed_refine_random_state: 0,
            data_row_reseed: false,
            fit_config: SaeFitConfig::default(),
            temperature_schedule: None,
            fisher_metric: Some(metric_request),
            row_loss_weights: None,
            registry: &registry,
        })
        .expect("fit seed on the month circle");
        let SaeFitSeedReport {
            base_term: term, ..
        } = seed;
        (term, target)
    }

    /// **Fixture precondition, not a result.** The displacement measurements
    /// below are only meaningful on a chart that actually wraps the circle. The
    /// first version of this fixture did not: 12 rows produced 4 distinct fitted
    /// coordinates spanning 0.75 rad, so a "+1 month" request was a move far
    /// outside the realized chart and every realized displacement was noise
    /// about a degenerate map. Nothing downstream may be believed until this
    /// passes, which is why it is a separate test rather than an assert buried
    /// in a helper.
    #[test]
    fn fixture_chart_spans_the_circle() {
        let (term, _) = build_month_term();
        let coords = term.assignment.coords[0].as_matrix();
        let mut lo = f64::INFINITY;
        let mut hi = f64::NEG_INFINITY;
        for row in 0..coords.nrows() {
            lo = lo.min(coords[[row, 0]]);
            hi = hi.max(coords[[row, 0]]);
        }
        // Already turns — see `months_between`.
        let span_turns = hi - lo;
        let mut distinct: Vec<f64> = (0..coords.nrows()).map(|r| coords[[r, 0]]).collect();
        distinct.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        distinct.dedup_by(|a, b| (*a - *b).abs() <= 1e-6);
        eprintln!(
            "[#2263 fixture] fitted chart spans {span_turns:.4} turns over {} rows, \
             {} distinct coordinates",
            coords.nrows(),
            distinct.len()
        );
        assert!(
            span_turns >= 0.8,
            "the fitted chart spans only {span_turns:.4} turns; a displacement measured on it \
             says nothing about steering"
        );
        assert!(
            distinct.len() >= N_MONTHS,
            "the fitted chart collapsed to {} distinct coordinates for {} rows; it cannot \
             resolve {N_MONTHS} months",
            distinct.len(),
            coords.nrows()
        );
    }

    /// Signed months from `from` to `to` on the 12-point circle, wrapped into
    /// `(-6, +6]` — the same accounting a month confusion matrix uses, so a
    /// realized `+8` and a realized `-4` are the same observation and cannot be
    /// double-counted.
    fn months_between(from: f64, to: f64) -> f64 {
        // `t` is in TURNS, not radians: `PeriodicHarmonicEvaluator` emits
        // `[1, sin(2π·1·t), cos(2π·1·t), …]` with the 2π INSIDE (basis.rs:302),
        // so t advancing by 1.0 is one full circle. Dividing by TAU here — the
        // radians assumption — scaled every request AND every measurement by
        // 2π. They cancelled in the ratio, which is exactly why the medians
        // looked right (“+1 requested, +0.997 realized”) while the aggregate was
        // incoherent: the “+1 month” request was really 0.5236 turns = 6.3
        // months, and the “+2 month” request was 1.047 turns, i.e. a full circle
        // plus 0.047 — which is precisely the +0.09 every row reported.
        let turns = to - from;
        let mut months = turns * N_MONTHS as f64;
        while months <= -(N_MONTHS as f64) / 2.0 {
            months += N_MONTHS as f64;
        }
        while months > N_MONTHS as f64 / 2.0 {
            months -= N_MONTHS as f64;
        }
        months
    }

    fn atlas_for(term: &SaeManifoldTerm, target: &Array2<f64>, amplitude: f64) -> EncodeAtlas {
        let mut norm_bound = 0.0_f64;
        for row in 0..target.nrows() {
            norm_bound = norm_bound.max(target.row(row).dot(&target.row(row)).sqrt());
        }
        EncodeAtlas::build(
            &term.atoms,
            &[amplitude.max(1.0)],
            // The edited row is `x + a·(g_to − g_from)`, so its norm can exceed
            // the target cloud's by the chord it carries; bound the atlas for
            // the largest row it will actually be asked to encode rather than
            // for the unedited cloud.
            norm_bound * (1.0 + 2.0 * amplitude.max(1.0)),
            AtlasConfig::default(),
        )
        .expect("encode atlas builds over the month circle")
    }

    /// The measurement item 3 has never had: for each requested advance
    /// `+1..+6` months, the realized advance after a real
    /// [`set_coordinate`] write and a real re-encode.
    ///
    /// At the row's own expression amplitude this is a round trip and must be
    /// exact: `set_coordinate` writes `x + a·(g(t_to) − g(t_from))`, whose
    /// certified encode is `t_to` by construction. An overshoot here would be a
    /// chart-side defect and would need no model to find.
    #[test]
    fn requested_month_advance_is_realized_exactly_at_the_rows_own_amplitude() {
        let (term, target) = build_month_term();
        let metric = term.row_metric().expect("metric installed").clone();
        let atlas = atlas_for(&term, &target, 1.0);
        let atom = &term.atoms[0];
        let coords = term.assignment.coords[0].as_matrix();

        let mut worst = 0.0_f64;
        let mut worst_case = (0usize, 0i32, 0.0_f64);
        for requested in 1..=6i32 {
            let mut realized_by_row = Vec::new();
            for row in 0..N_ROWS {
                // The request must be anchored to the coordinate `set_coordinate`
                // will actually steer FROM — its own certified encode of the row
                // — not to the fitted coordinate. Anchoring to the fitted coord
                // measures `coords − encode(x)` on top of the displacement, which
                // is an encoder property and not a steering one; that mistake is
                // what the `[#2263 encode-control]` line below now makes visible
                // instead of silently folding into the result.
                let (t_from_read, _) = atlas
                    .certified_encode_row(atom, 0, target.row(row), 1.0)
                    .expect("the unedited row encodes");
                let t_from = t_from_read[0];
                let encode_gap = months_between(coords[[row, 0]], t_from);
                if requested == 1 {
                    eprintln!(
                        "[#2263 encode-control] row {row}: fitted coord {:+.6} vs certified \
                         encode {t_from:+.6} (gap {encode_gap:+.4} months)",
                        coords[[row, 0]]
                    );
                }
                let t_to = t_from + (requested as f64) / (N_MONTHS as f64);
                let set = set_coordinate(
                    &term,
                    &metric,
                    &atlas,
                    target.row(row),
                    0,
                    row,
                    1.0,
                    &[t_to],
                )
                .expect("set_coordinate writes the requested month");
                let (t_realized, _) = atlas
                    .certified_encode_row(atom, 0, set.edited.view(), 1.0)
                    .expect("the edited row re-encodes");
                let realized = months_between(set.t_from_certified[0], t_realized[0]);
                realized_by_row.push(realized);
                // At exactly the antipode (+6 of 12) there is no fact of the
                // matter about direction: +6 ≡ −6, and the (−6, +6] wrap forces
                // a representation. Score the magnitude there; the signed value
                // everywhere else. A confusion matrix over +1..+6 will show the
                // same split and it is the METRIC that is degenerate, not the
                // steer.
                let antipodal = requested * 2 == N_MONTHS as i32;
                let error = if antipodal {
                    (realized.abs() - requested as f64).abs()
                } else {
                    (realized - requested as f64).abs()
                };
                if error > worst {
                    worst = error;
                    worst_case = (row, requested, realized);
                }
            }
            let mean =
                realized_by_row.iter().sum::<f64>() / realized_by_row.len() as f64;
            eprintln!(
                "[#2263 displacement] requested +{requested} months: realized mean {mean:+.4} \
                 over {} rows (min {:+.4}, max {:+.4})",
                realized_by_row.len(),
                realized_by_row.iter().cloned().fold(f64::INFINITY, f64::min),
                realized_by_row
                    .iter()
                    .cloned()
                    .fold(f64::NEG_INFINITY, f64::max),
            );
        }
        eprintln!(
            "[#2263 displacement] worst |realized − requested| = {worst:.3e} months \
             (row {}, requested +{}, realized {:+.4})",
            worst_case.0, worst_case.1, worst_case.2
        );
        // The bar is the CERTIFIED ENCODE's own resolution, not exact
        // arithmetic: the round trip is closed by an iterative solve whose
        // per-row disagreement with the fitted coordinate is printed above as
        // `[#2263 encode-control]` and runs to ~0.06 months on this fixture.
        // A round trip cannot be measured tighter than the encoder that closes
        // it, so a 1e-6 bar would be asserting exactness on a numeric solve.
        // Measured worst here is 0.056 months; 0.15 leaves headroom without
        // admitting a real gain error, which would be O(1) months.
        assert!(
            worst <= 0.15,
            "the chart round trip does not realize the requested month advance: \
             worst |realized − requested| = {worst:.3e} months (row {}, requested +{}, \
             realized {:+.4})",
            worst_case.0,
            worst_case.1,
            worst_case.2
        );
    }

    /// **Wrap versus saturation** (`isae`'s discriminator). A realized
    /// displacement that is roughly constant across several different requests
    /// is the signature item 3 reports (+1..+6 realizing a mode of +8), and TWO
    /// mechanisms flatten a response that way: a gain that saturates, and a
    /// coordinate whose period the consumer disagrees about. Requesting only
    /// `+1..+6` cannot tell them apart. Requesting `+7..+18` can: a wrapped
    /// coordinate keeps tracking the request modulo the period, while a
    /// saturating gain does not recover past the point it saturated.
    ///
    /// This costs one extra range and settles which family any future
    /// behavioural overshoot belongs to.
    #[test]
    fn requests_past_a_full_turn_wrap_rather_than_saturate() {
        let (term, target) = build_month_term();
        let metric = term.row_metric().expect("metric installed").clone();
        let atlas = atlas_for(&term, &target, 1.0);
        let atom = &term.atoms[0];

        let mut worst = 0.0_f64;
        for requested in 7..=18i32 {
            let mut realized_by_row = Vec::new();
            for row in 0..N_ROWS {
                let Ok((t_from_read, _)) =
                    atlas.certified_encode_row(atom, 0, target.row(row), 1.0)
                else {
                    continue;
                };
                let t_to = t_from_read[0] + (requested as f64) / (N_MONTHS as f64);
                let Ok(set) = set_coordinate(
                    &term, &metric, &atlas, target.row(row), 0, row, 1.0, &[t_to],
                ) else {
                    continue;
                };
                let Ok((t_realized, _)) =
                    atlas.certified_encode_row(atom, 0, set.edited.view(), 1.0)
                else {
                    continue;
                };
                realized_by_row
                    .push(months_between(set.t_from_certified[0], t_realized[0]));
            }
            if realized_by_row.is_empty() {
                continue;
            }
            // What a WRAP predicts: the request folded into (−6, +6].
            let mut expected = requested as f64;
            while expected > N_MONTHS as f64 / 2.0 {
                expected -= N_MONTHS as f64;
            }
            let antipodal = (expected.abs() - N_MONTHS as f64 / 2.0).abs() < 1e-9;
            let mean =
                realized_by_row.iter().sum::<f64>() / realized_by_row.len() as f64;
            let row_worst = realized_by_row.iter().fold(0.0_f64, |m, &r| {
                let e = if antipodal {
                    (r.abs() - expected.abs()).abs()
                } else {
                    (r - expected).abs()
                };
                m.max(e)
            });
            eprintln!(
                "[#2263 wrap] requested +{requested}: wrap predicts {expected:+.0},                  realized mean {mean:+.4}, worst row error {row_worst:.4} months"
            );
            worst = worst.max(row_worst);
        }
        assert!(
            worst <= 0.15,
            "requests past a full turn neither wrap nor track: worst deviation from the \
             wrap prediction is {worst:.4} months. A saturating gain would fail here while \
             +1..+5 passed"
        );
    }

    /// The amplitude-as-dose-knob measurement: the same requested advance
    /// written with an intensity that is NOT the row's own.
    ///
    /// This is the recipe item 1 says a user must follow to get any behavioral
    /// effect (`alpha` swept to 16). It is measurement, not a gate on a
    /// preferred answer: the assertion is only that the realized displacement
    /// is a **deterministic function of `alpha` that the caller can invert** —
    /// specifically that it scales linearly in `alpha` — because that is the
    /// property which decides whether item 3 is fixable by arithmetic or needs
    /// a closed loop.
    #[test]
    fn realized_advance_scales_with_the_written_amplitude_not_the_request() {
        let (term, target) = build_month_term();
        let metric = term.row_metric().expect("metric installed").clone();
        let atom = &term.atoms[0];
        let requested = 1i32;

        for &alpha in &[1.0_f64, 2.0, 4.0, 8.0, 16.0] {
            let atlas = atlas_for(&term, &target, alpha);
            let mut realized_by_row = Vec::new();
            for row in 0..N_ROWS {
                let Ok((t_from_read, _)) =
                    atlas.certified_encode_row(atom, 0, target.row(row), alpha)
                else {
                    continue;
                };
                let t_to = t_from_read[0] + (requested as f64) / (N_MONTHS as f64);
                // The row expresses the atom at intensity 1; the caller writes
                // at `alpha`. That mismatch is the whole subject.
                let Ok(set) = set_coordinate(
                    &term, &metric, &atlas, target.row(row), 0, row, alpha, &[t_to],
                ) else {
                    continue;
                };
                let Ok((t_realized, _)) =
                    atlas.certified_encode_row(atom, 0, set.edited.view(), 1.0)
                else {
                    continue;
                };
                realized_by_row.push(months_between(
                    set.t_from_certified[0],
                    t_realized[0],
                ));
            }
            if realized_by_row.is_empty() {
                eprintln!(
                    "[#2263 amplitude] alpha={alpha}: no row produced a certified \
                     encode of the edited activation"
                );
                continue;
            }
            let mean =
                realized_by_row.iter().sum::<f64>() / realized_by_row.len() as f64;
            let mut sorted = realized_by_row.clone();
            sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            eprintln!(
                "[#2263 amplitude] requested +{requested} month at alpha={alpha}: \
                 realized mean {mean:+.4}, median {:+.4}, over {} of {N_ROWS} rows",
                sorted[sorted.len() / 2],
                sorted.len(),
            );
        }
    }
}
