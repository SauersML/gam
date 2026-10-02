//! Counterexample-guided refinement of the two-part-code decomposition (#2951).
//!
//! `mpd_cegar_2951 MODEL_DIR N`
//!
//! `MODEL_DIR` is a `gam_mpd::import` export whose declared domain is a product: a residual MLP on
//! bit vectors, or a transformer whose every position ranges over the tokens it holds in the
//! export's samples (modular addition: every `(a, b)` in `Z_p²` followed by `=`). `N` is the
//! observations of each row.
//!
//! The family starts as the export's first sample alone and grows by `engine::decompose_refined`:
//! each round the engine selects a program by its two-part code on the family, and the verifier
//! ascends `KL(model ‖ program)` from every row; every endpoint certified worse than the family's
//! worst row joins it. Because these domains enumerate, the final program is then checked against
//! the exact supremum over the whole domain and scored on it, and compared with the program the
//! engine selects on the whole domain directly.
//!
//! Report (JSON on stdout): per round the rows, the family's certified worst KL, the ascents'
//! worst, the counterexamples and evaluations, and the adversary's ladder (worst KL after k ascent
//! steps); then the refined program's bits, its exhaustive supremum and the full-domain baseline.

use gam_mpd::cegar::{Input, InputDomain, LADDER, SlotDomain, SlotValue, ascend, exhaustive_worst, input_at, ladder};
use gam_mpd::contract::Contract;
use gam_mpd::engine::{Budget, decompose, decompose_refined, library};
use gam_mpd::import::import;
use gam_mpd::operator_program::{Slot, SlotValues};
use ndarray::Array1;
use serde_json::json;
use std::path::PathBuf;
use std::time::Instant;

/// The declared product domain of a contract's slots: a raw slot holding only bits is the bit
/// cube, a token slot is every token its values take on the declared samples.
fn product_domain(contract: &Contract) -> Result<InputDomain, String> {
    let slots = contract
        .declarations
        .slots
        .iter()
        .zip(&contract.family.slots)
        .map(|(slot, values)| match (slot, values) {
            (Slot::Token { .. }, SlotValues::Tokens(tokens)) => {
                let mut set: Vec<u32> = tokens.clone();
                set.sort_unstable();
                set.dedup();
                Ok(SlotDomain::Tokens(set))
            }
            (Slot::Raw { width }, SlotValues::Raw(rows)) if rows.iter().all(|v| *v == 0.0 || *v == 1.0) => {
                Ok(SlotDomain::Corners { lower: Array1::zeros(*width), upper: Array1::ones(*width) })
            }
            _ => Err("only token slots and bit-vector slots enumerate".to_string()),
        })
        .collect::<Result<Vec<_>, String>>()?;
    Ok(InputDomain { slots })
}

fn main() -> Result<(), String> {
    let args: Vec<String> = std::env::args().collect();
    let usage = "mpd_cegar_2951 MODEL_DIR N";
    let dir = PathBuf::from(args.get(1).ok_or(usage)?);
    let observations: u64 = args.get(2).ok_or(usage)?.parse().map_err(|e| format!("N: {e}"))?;
    let imported = import(&dir)?;
    let (model, mut full) = (imported.program, imported.contract);
    full.observations = observations;
    let domain = product_domain(&full)?;
    let cardinality = domain.cardinality().ok_or("the domain does not enumerate")?;
    let budget = Budget { screenings: 1 << 20, certifications: 1 << 12, ..Budget::default() };
    let library = library();
    let readouts = full.readouts;
    let mut start = full.clone();
    start.family = full.family.select(&[0]);
    if let gam_mpd::contract::FamilyKind::Sample { units, .. } = &mut start.kind {
        units.truncate(1);
    }
    let started = Instant::now();
    let refinement = decompose_refined(&model, &start, &library, &budget, &domain, &[]).map_err(|e| e.to_string())?;
    let refined_seconds = started.elapsed().as_secs_f64();
    let program = &refinement.decomposition.program;
    let (worst_input, worst_kl, sup_upper) =
        exhaustive_worst(&model, program, &domain, readouts, 1024).map_err(|e| e.to_string())?;
    let reference = full.logits(&model).map_err(|e| e.to_string())?;
    let on_domain = full.score(program, &reference).map_err(|e| e.to_string())?;
    // The ascent from every input of the domain: the adversary unrestricted in its starts.
    let every: Vec<Input> = (0..full.family.rows).map(|row| input_at(&full.family, row)).collect();
    let everywhere = ascend(&model, program, &domain, &every, readouts).map_err(|e| e.to_string())?;
    let started = Instant::now();
    let direct = decompose(&model, &full, &library, &budget).map_err(|e| e.to_string())?;
    let direct_seconds = started.elapsed().as_secs_f64();
    let describe = |input: &Input| -> Vec<serde_json::Value> {
        input
            .iter()
            .map(|v| match v {
                SlotValue::Token(t) => json!(t),
                SlotValue::Raw(x) => json!(x.to_vec()),
            })
            .collect()
    };
    let report = json!({
        "model": format!("{} ({})", imported.name, imported.kind),
        "export": imported.record,
        "observations": observations,
        "domain_inputs": cardinality,
        "native_bits": model.code_bits().map_err(|e| e.to_string())?,
        "rounds": refinement.rounds.iter().map(|r| json!({
            "rows": r.rows, "family_worst_kl_upper": r.data_worst_upper, "ascent_worst_kl": r.ascent_worst,
            "counterexamples": r.counterexamples, "evaluations": r.evaluations, "ladder": r.ladder,
        })).collect::<Vec<_>>(),
        "refined": {
            "rows": start.family.rows + refinement.added,
            "seconds": refined_seconds,
            "program_bits": on_domain.program_bits,
            "data_bits_on_domain": on_domain.data_bits,
            "family_worst_kl_upper": refinement.rounds.last().map(|r| r.data_worst_upper),
            "domain_sup_kl": worst_kl.value,
            "domain_sup_kl_upper": sup_upper,
            "domain_worst_input": describe(&worst_input),
            "argmax_disagreements_on_domain": on_domain.evaluation.argmax_disagreements,
            "ladder_from_every_input": ladder(&everywhere, &LADDER),
        },
        "direct": {
            "rows": full.family.rows,
            "seconds": direct_seconds,
            "program_bits": direct.score.program_bits,
            "data_bits_on_domain": direct.score.data_bits,
            "domain_sup_kl_upper": direct.score.evaluation.max_kl.upper_bound(),
            "argmax_disagreements_on_domain": direct.score.evaluation.argmax_disagreements,
            "stop": format!("{:?}", direct.stop),
        },
    });
    println!("{}", serde_json::to_string_pretty(&report).map_err(|e| e.to_string())?);
    Ok(())
}
