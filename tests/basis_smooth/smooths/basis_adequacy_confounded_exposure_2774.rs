//! gam#2774 — a certified 16-D Duchon binomial fit that silently underfits the
//! PC confounding it was asked to adjust for.
//!
//! The filed report is a biobank-shaped association model: a null exposure
//! correlated with 16 population PCs, an outcome with a rotated, anisotropic
//! NONLINEAR PC effect and no exposure effect conditional on the PCs, adjusted
//! by one native `duchon(pc1..pc16, centers=24)` term. It converges, reports
//! `certified = true`, and returns the null exposure at `p = 6.2e-5`.
//!
//! What this file pins is NOT the p-value of the exposure — a single seed at CI
//! scale cannot pin that, and chasing it would make the test a coin flip. It
//! pins the thing that was actually missing: **the fit now says, at fit time,
//! that the basis it converged on cannot represent the structure its residuals
//! still carry**, and it says nothing of the sort when the basis is adequate.
//!
//! Three arms, and the third is the one the issue is really about:
//!
//! * **flagged** — the rotated curved 2-D truth. The 16-D Duchon's 17-column
//!   linear null space leaves 24 − 17 = 7 nonlinear columns, which cannot reach
//!   `0.70 sin(1.6u) + 0.38(v² − 1)`. The residual lack-of-fit test must
//!   detect that, and the fit must carry a user-facing note.
//! * **not flagged** — the SAME model, the same basis, the same `n`, with a
//!   truth that is exactly linear in the PCs and therefore exactly inside the
//!   Duchon kernel's own polynomial null space — a function the fitted basis
//!   represents with no error at all. Nothing is missing, and the check must
//!   say so. Without this arm the first one is satisfied by a diagnostic
//!   that fires on everything.
//! * **certified is not adequate** — the flagged fit still certifies. The two
//!   verdicts are about different things, and that is now an executable
//!   statement rather than a documentation claim.
//!
//! Every arm runs at one scale, [`FIXTURE_ROWS`], and nothing here is
//! `#[ignore]`d — that is a build ban in this tree, and a smoke test nobody
//! runs is not a smoke test. The constant is 50 000 rather than the filed
//! 200 000 or the 3 000 these arms were first written at; the measured sweep
//! that decides it is recorded beside the constant, and the short version is
//! that 3 000 rows do not detect this alternative at all while 50 000 detect it
//! by sixteen orders of magnitude, at ~15 s per fit. The `O(n·q²)` accumulation
//! runs against a real row count there and the enrichment budget actually binds
//! (`q` is capped by the flop budget, not by 4×`k`), which the filed scale would
//! not add to.

use gam::data::EncodedDataset;
use gam::inference::model::{ColumnKindTag, DataSchema, SchemaColumn};
use gam::{FitConfig, FitResult, fit_from_formula_with_notes};

/// Number of population PCs in the fixture. Sixteen is load-bearing: the Duchon
/// KERNEL's polynomial null space is `d + 1 = 17` columns, so `centers=24`
/// leaves seven nonlinear columns for the whole 16-D surface. That ratio IS the
/// defect. It is NOT the same object as the adequacy row's `nullspace_dim`,
/// which is the joint null space of the term's active penalties and is 0 here.
const PC_DIMENSION: usize = 16;

/// The filed `centers=24`. Explicitly pinned by the user, which is also why the
/// engine's saturation-driven resolution loop never sees this term:
/// `adaptive_spatial_term_mask` admits only `CenterStrategy::Auto`.
const CENTERS: usize = 24;

/// Rows every arm in this file runs at.
///
/// ONE scale for all four, so the arms differ in what they claim and not in the
/// data they claim it about — which is what makes the negative control a
/// control. 50 000 is chosen from a measured sweep of this exact fixture and
/// seed, statistic `T` against its reference `r = 75`:
///
/// | `n` | `T` curved | `p` curved | `T` linear | `p` linear |
/// |---:|---:|---:|---:|---:|
/// | 3 000 | 85.5 | 1.9e-1 | 68.4 | 6.9e-1 |
/// | 6 000 | 213.3 | 3.4e-15 | 74.8 | 4.8e-1 |
/// | 12 000 | 127.8 | 1.4e-4 | 82.9 | 2.5e-1 |
/// | 25 000 | 161.0 | 3.2e-8 | 72.7 | 5.5e-1 |
/// | 50 000 | 234.3 | 2.7e-18 | 50.4 | 9.9e-1 |
/// | 200 000 | 838.6 | 2.0e-129 | — | — |
///
/// The excess `T − r` runs 10 → 764 over a 67× row increase, which is the
/// `n`-linear non-centrality the construction predicts. Two facts in that table
/// set the constant. **`n = 3 000` does not detect this alternative at all**
/// (`p = 0.19`), which is what the first version of these arms asserted and is
/// why they were red. And a single seed at small `n` is noisy in `λ̂` — 12 000
/// reads weaker than 6 000 — so an arm gated at `p < 1e-3` needs a margin in
/// orders of magnitude, not a factor of seven. 50 000 is the smallest swept
/// scale that has one, at ~15 s per fit; the filed 200 000 buys 111 more orders
/// of margin for ~80 s and nothing else.
const FIXTURE_ROWS: usize = 50_000;

/// Deterministic normal draws. A fixture that decides whether a diagnostic fires
/// may not depend on which sampler version is linked.
struct Lcg(u64);

impl Lcg {
    fn next_uniform(&mut self) -> f64 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        ((self.0 >> 11) as f64) / ((1u64 << 53) as f64)
    }

    fn next_normal(&mut self) -> f64 {
        let u1 = self.next_uniform().max(1e-12);
        let u2 = self.next_uniform();
        (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
    }

    fn next_bernoulli(&mut self, probability: f64) -> f64 {
        f64::from(self.next_uniform() < probability)
    }
}

fn logistic(value: f64) -> f64 {
    1.0 / (1.0 + (-value).exp())
}

/// Which PC effect the outcome carries.
#[derive(Clone, Copy)]
enum PcEffect {
    /// `0.70 sin(1.6u) + 0.38(v² − 1)` on the rotated coordinates
    /// `u = (z₁+z₂)/√2`, `v = (z₁−z₂)/√2` — the filed alternative. Nonlinear,
    /// rotated, and anisotropic, so it is outside the Duchon null space and
    /// beyond what seven kernel columns in 16 dimensions can reach.
    RotatedCurved,
    /// `0.6z₁ − 0.4z₂ + 0.3z₃` — exactly inside the smooth's own unpenalized
    /// linear null space. The basis is adequate BY CONSTRUCTION here, which is
    /// what makes this the honest negative control.
    Linear,
}

fn pc_effect(effect: PcEffect, z: &[f64]) -> f64 {
    match effect {
        PcEffect::RotatedCurved => {
            let u = (z[0] + z[1]) / std::f64::consts::SQRT_2;
            let v = (z[0] - z[1]) / std::f64::consts::SQRT_2;
            0.70 * (1.6 * u).sin() + 0.38 * (v * v - 1.0)
        }
        PcEffect::Linear => 0.6 * z[0] - 0.4 * z[1] + 0.3 * z[2],
    }
}

/// The filed data-generating process: an exposure correlated with the PCs, and
/// an outcome with a PC effect and **no exposure effect conditional on the PCs**.
fn confounded_dataset(n: usize, seed: u64, effect: PcEffect) -> EncodedDataset {
    let mut rng = Lcg(seed);
    let mut latent = vec![0.0_f64; n * PC_DIMENSION];
    let mut dosage = vec![0.0_f64; n];
    let mut effects = vec![0.0_f64; n];
    // Geometric PC scales, 3.0 down to 0.45, as filed: the covariates the model
    // sees are NOT on a common scale, which is why the enrichment standardizes.
    let scales: Vec<f64> = (0..PC_DIMENSION)
        .map(|j| {
            let t = j as f64 / (PC_DIMENSION - 1) as f64;
            3.0_f64 * (0.45_f64 / 3.0_f64).powf(t)
        })
        .collect();
    for row in 0..n {
        let z: Vec<f64> = (0..PC_DIMENSION).map(|_| rng.next_normal()).collect();
        let frequency = logistic(-0.7 + 0.75 * z[0] - 0.55 * z[1] + 0.25 * z[2]);
        dosage[row] = rng.next_bernoulli(frequency) + rng.next_bernoulli(frequency);
        effects[row] = pc_effect(effect, &z);
        for (column, value) in z.iter().enumerate() {
            latent[row * PC_DIMENSION + column] = value * scales[column];
        }
    }
    // Intercept by bisection for ~10% prevalence, exactly as filed. The
    // prevalence matters: at 10% the Bernoulli residual variance dwarfs the
    // missing structure, which is precisely the regime that defeats local
    // residual differencing.
    let (mut lo, mut hi) = (-20.0_f64, 20.0_f64);
    for _ in 0..80 {
        let mid = 0.5 * (lo + hi);
        let mean = effects.iter().map(|e| logistic(mid + e)).sum::<f64>() / n as f64;
        if mean < 0.10 { lo = mid } else { hi = mid }
    }
    let intercept = 0.5 * (lo + hi);

    let mut headers = vec!["y".to_string(), "dosage".to_string()];
    headers.extend((1..=PC_DIMENSION).map(|j| format!("pc{j}")));
    let width = headers.len();
    let mut values = Vec::<f64>::with_capacity(n * width);
    for row in 0..n {
        values.push(rng.next_bernoulli(logistic(intercept + effects[row])));
        values.push(dosage[row]);
        for column in 0..PC_DIMENSION {
            values.push(latent[row * PC_DIMENSION + column]);
        }
    }
    let mut columns = vec![
        SchemaColumn {
            name: "y".to_string(),
            kind: ColumnKindTag::Binary,
            levels: vec![],
        },
        SchemaColumn {
            name: "dosage".to_string(),
            kind: ColumnKindTag::Continuous,
            levels: vec![],
        },
    ];
    columns.extend((1..=PC_DIMENSION).map(|j| SchemaColumn {
        name: format!("pc{j}"),
        kind: ColumnKindTag::Continuous,
        levels: vec![],
    }));
    // Parallel to `headers`, and DERIVED from `columns` rather than written out
    // a second time: the two are required to agree, and a fixture that states
    // the column kinds twice can disagree with itself.
    let column_kinds = columns.iter().map(|column| column.kind).collect();
    EncodedDataset {
        headers,
        values: ndarray::Array2::from_shape_vec((n, width), values)
            .expect("confounded fixture shape"),
        schema: DataSchema { columns },
        column_kinds,
    }
}

fn formula() -> String {
    let covariates = (1..=PC_DIMENSION)
        .map(|j| format!("pc{j}"))
        .collect::<Vec<_>>()
        .join(", ");
    format!("y ~ dosage + duchon({covariates}, centers={CENTERS})")
}

struct FitOutcome {
    p_value: Option<f64>,
    basis_dim: usize,
    nullspace_dim: usize,
    enrichment_rank: Option<usize>,
    provenance: String,
    notes: Vec<String>,
    certified: bool,
}

fn fit_confounded(n: usize, seed: u64, effect: PcEffect) -> FitOutcome {
    let dataset = confounded_dataset(n, seed, effect);
    let mut config = FitConfig::default();
    config.family = Some("binomial".to_string());
    let outcome = fit_from_formula_with_notes(&formula(), &dataset, &config)
        .expect("confounded 16-D Duchon fixture must fit");
    let FitResult::Standard(standard) = &outcome.result else {
        panic!("the confounded fixture is an ordinary standard GAM");
    };
    assert_eq!(
        standard.basis_adequacy.len(),
        1,
        "one smooth term, one adequacy row"
    );
    let row = &standard.basis_adequacy[0];
    FitOutcome {
        p_value: row.p_value,
        basis_dim: row.basis_dim,
        nullspace_dim: row.nullspace_dim,
        enrichment_rank: row.enrichment_rank,
        provenance: row.provenance.label().to_string(),
        notes: outcome.inference_notes.advisories.clone(),
        // The same predicate `summary()` reports as `convergence.certified`:
        // no outer coordinate optimized ⇒ the converged inner mode IS the proof.
        certified: standard
            .fit
            .convergence_evidence()
            .outer_certificate()
            .is_none_or(|certificate| certificate.certifies()),
    }
}

/// The filed failure: a nonlinear rotated PC surface that seven kernel columns
/// in 16 dimensions cannot reach. The residual lack-of-fit test must see it, and
/// the fit must carry a note the caller cannot miss.
#[test]
fn underfitted_confounder_smooth_is_flagged_at_fit_time() {
    let outcome = fit_confounded(FIXTURE_ROWS, 20_260_820, PcEffect::RotatedCurved);
    assert_eq!(
        outcome.provenance, "radial_enrichment",
        "the check must actually run on a 16-D continuous smooth"
    );
    // The realized width and its unpenalized part are the two numbers that make
    // the summary table's EDF column readable, so pin both: the report may not
    // quietly drop the decomposition.
    //
    // `basis_dim` is one column short of `centers` — the term collection's
    // identifiability centering costs exactly one.
    assert_eq!(
        outcome.basis_dim,
        CENTERS - 1,
        "the realized width is `centers` minus the one column identifiability \
         centering removes; if this moves, the summary's EDF column moved with it"
    );
    // `nullspace_dim` is `dim null(Σ_k S_k)` over the term's ACTIVE PENALTIES —
    // NOT the Duchon KERNEL's `d + 1 = 17`-column polynomial null space, which
    // is a different object and the one this assertion used to name. For this
    // shipped term the active penalties sum to full rank on the realized 23
    // columns, so the answer is 0: every coefficient direction is reached by
    // some penalty, the polynomial part included. Measured, not assumed. A
    // report that confused the two objects would read 17 here.
    assert_eq!(
        outcome.nullspace_dim, 0,
        "no direction of this term escapes every active penalty"
    );
    assert!(
        outcome.basis_dim >= outcome.nullspace_dim,
        "realized width {} cannot be below its own null space {}",
        outcome.basis_dim,
        outcome.nullspace_dim
    );
    let rank = outcome
        .enrichment_rank
        .expect("a running check reports its reference d.f.");
    assert!(
        rank > outcome.basis_dim,
        "the alternative must carry MORE resolution than the fitted basis; \
         rank={rank} vs basis_dim={}",
        outcome.basis_dim
    );
    let p_value = outcome.p_value.expect("a running check reports a p-value");
    assert!(
        p_value < 1.0e-3,
        "the underfitted 16-D Duchon must be detected; got p={p_value:e}"
    );
    assert!(
        outcome
            .notes
            .iter()
            .any(|note| note.contains("basis adequacy")),
        "the fit must carry a user-facing basis-adequacy note; got {:?}",
        outcome.notes
    );
}

/// The negative control, and the reason the arm above is not vacuous: the same
/// model, the same basis, the same `n`, the same seed — and a truth the fitted
/// basis represents EXACTLY, because it lives in the Duchon kernel's own
/// polynomial null space. Nothing is missing, and the check has to say so.
///
/// "Represents exactly", not "leaves unpenalized": the fitted term shrinks that
/// part too (see the `nullspace_dim` pin above). What makes this the honest
/// control is that no amount of extra resolution could improve the fit, which
/// is exactly the null the statistic tests.
///
/// The threshold is the fit-time note level rather than a nominal 5 %, because
/// that is the level at which a note is actually raised. The linear arm of the
/// sweep beside [`FIXTURE_ROWS`] reads `p` between 0.25 and 0.99 across five
/// scales — nowhere near it.
#[test]
fn adequate_basis_on_the_same_design_is_not_flagged() {
    let outcome = fit_confounded(FIXTURE_ROWS, 20_260_820, PcEffect::Linear);
    assert_eq!(outcome.provenance, "radial_enrichment");
    let p_value = outcome.p_value.expect("a running check reports a p-value");
    assert!(
        p_value > 1.0e-3,
        "a truth the fitted basis represents exactly must not be flagged; got p={p_value:e}"
    );
    assert!(
        !outcome
            .notes
            .iter()
            .any(|note| note.contains("basis adequacy")),
        "no note may be raised on an adequate basis; got {:?}",
        outcome.notes
    );
}

/// The contract the issue is actually about: `certified` covers the optimizer,
/// and a fit can be certified AND inadequate at the same time. Documentation
/// says so; this makes it executable.
#[test]
fn certification_does_not_imply_basis_adequacy() {
    let outcome = fit_confounded(FIXTURE_ROWS, 20_260_820, PcEffect::RotatedCurved);
    assert!(
        outcome.certified,
        "the filed fit converges and certifies — that is the whole complaint"
    );
    assert!(
        outcome.p_value.is_some_and(|p| p < 1.0e-3),
        "and is simultaneously inadequate"
    );
}

/// The smoke tier: the same defect held to a bar four orders tighter than the
/// note level, plus the note itself reaching the caller.
///
/// It shares [`FIXTURE_ROWS`] with the arm above rather than running its own
/// larger fit, because at that scale the measured `p` is 2.7e-18 — the `1e-8`
/// bar gates the mechanism, not the seed — and the enrichment budget already
/// binds there (`q` capped by the flop budget, not by 4×`k`). What it adds over
/// `underfitted_confounder_smooth_is_flagged_at_fit_time` is the strength of
/// the bar: a regression that merely weakened the statistic would still clear
/// `1e-3` and would not clear this.
#[test]
fn large_scale_confounded_fit_is_flagged() {
    let outcome = fit_confounded(FIXTURE_ROWS, 20_260_820, PcEffect::RotatedCurved);
    assert_eq!(outcome.provenance, "radial_enrichment");
    let p_value = outcome.p_value.expect("a running check reports a p-value");
    assert!(
        p_value < 1.0e-8,
        "at biobank scale the underfit must be overwhelming; got p={p_value:e}"
    );
    assert!(
        outcome
            .notes
            .iter()
            .any(|note| note.contains("basis adequacy")),
        "the large-n fit must carry the note too; got {:?}",
        outcome.notes
    );
}
