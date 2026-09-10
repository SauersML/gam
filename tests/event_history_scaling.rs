//! How an event-history fit scales with the number of marks.
//!
//! The accumulation that forms the gradient and Hessian contracts every mark's
//! node function against every coefficient carried, so its cost grows with the
//! square of the mark count. A multi-disease model is exactly the regime where
//! that matters, and a three-mark fixture is exactly the regime where it does
//! not show. This measures the fit at several mark counts and reports what the
//! growth actually is, so a change to that accumulation can be judged against a
//! number rather than an argument.
//!
//! It is a measurement, not a threshold: the assertion is only that every fit
//! converged, since a timing gate on a shared machine measures the machine.

use gam::families::custom_family::BlockwiseFitOptions;
use gam::event_history::{
    CovariateSegment, Event, EventHistoryCohort, MarkKind, SubjectHistory,
    fit_event_history_formula,
};
use ndarray::Array2;
use std::time::Instant;

/// A deterministic uniform stream, so the fixture is the same at every commit
/// this is compared across.
struct Stream(u64);

impl Stream {
    fn next(&mut self) -> f64 {
        // SplitMix64, taken to the unit interval.
        self.0 = self.0.wrapping_add(0x9E3779B97F4A7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
        ((z ^ (z >> 31)) >> 11) as f64 / (1u64 << 53) as f64
    }
}

/// A cohort of `subjects` people followed for `follow_up`, each at risk for
/// `marks` first-occurrence marks with a shared covariate.
fn cohort(subjects: usize, marks: usize, follow_up: f64, seed: u64) -> EventHistoryCohort {
    let mut stream = Stream(seed);
    let mut covariates = Array2::<f64>::zeros((subjects, 1));
    let mut histories = Vec::with_capacity(subjects);
    for s in 0..subjects {
        let x = 2.0 * stream.next() - 1.0;
        covariates[[s, 0]] = x;
        let mut events: Vec<Event> = Vec::new();
        for d in 0..marks {
            // One rate per mark, spread so the marks are not copies.
            let rate = (0.12 + 0.03 * (d % 5) as f64) * (1.0 + 0.4 * x);
            let time = -stream.next().max(1e-12).ln() / rate.max(1e-6);
            if time < follow_up {
                events.push(Event { time, mark: d });
            }
        }
        events.sort_by(|a, b| a.time.total_cmp(&b.time));
        histories.push(SubjectHistory {
            id: format!("s{s}"),
            entry: 0.0,
            exit: follow_up,
            events,
            segments: vec![CovariateSegment { start: 0.0, row: s }],
        });
    }
    EventHistoryCohort {
        mark_names: (0..marks).map(|d| format!("m{d}")).collect(),
        mark_kinds: vec![MarkKind::Once; marks],
        covariate_names: vec!["x".to_string()],
        covariate_levels: vec![Vec::new()],
        covariates,
        subjects: histories,
    }
}

#[test]
fn the_fit_reports_its_cost_against_the_mark_count() {
    let subjects = 150;
    let follow_up = 4.0;
    let mut measured: Vec<(usize, f64)> = Vec::new();
    for &marks in &[2usize, 4, 8, 16] {
        let mut data = cohort(subjects, marks, follow_up, 20_260_909);
        let started = Instant::now();
        let fit = fit_event_history_formula(&mut data, "x", BlockwiseFitOptions::default())
            .expect("the scaling fixture must fit");
        let seconds = started.elapsed().as_secs_f64();
        // Every fit object comes from a converged optimisation, so reaching
        // one at all is the assertion; the rank is the evidence's verdict and
        // is reported, not required.
        assert!(
            fit.fit.log_likelihood.is_finite(),
            "{marks} marks: the fit's log-likelihood is {}",
            fit.fit.log_likelihood
        );
        println!(
            "[scaling] {marks:2} marks: {seconds:8.3}s  rank {}  loglik {:.3}",
            fit.rank(),
            fit.fit.log_likelihood
        );
        measured.push((marks, seconds));
    }
    let growth: Vec<String> = measured
        .windows(2)
        .map(|w| {
            let (marks, seconds) = w[1];
            let (previous_marks, previous) = w[0];
            format!(
                "{previous_marks}→{marks}: ×{:.2}",
                seconds / previous.max(1e-9)
            )
        })
        .collect();
    println!(
        "[scaling] cost per doubling of the mark count: {}",
        growth.join(", ")
    );
}
