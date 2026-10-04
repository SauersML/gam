//! A language model's MLPs accounted for by explicit rules between subcomponent amplitudes
//! (#2951, `gam_mpd::mlp_account`), fitted layer by layer on hybrid states and checked end to end.
//!
//! `mpd_mlp_functions_2951 TRAIN_EXPORT EVAL_EXPORT LIBRARY_DIR OUT_DIR [KEY=VALUE ...]`
//!
//! Keys (defaults): `layers=0,1`, `train=64`, `valid=8` (sequences of `TRAIN_EXPORT`, the held-out
//! ones after the training ones), `eval=16` (sequences of `EVAL_EXPORT`), `context=512`,
//! `observations=1e6` (`n` of the remainder's price), `grow_rows=8192` (training inputs the rules
//! are grown on, evenly strided), `iterations=200`, `patience=10` (each refinement's), `rounds=2`, `draws=2`
//! (sampled-label reverse passes per sequence for the Fisher), `starts=vpd,neurons`, `load=DIR` (a
//! layer whose `vpd` account `DIR` holds, as this driver writes it, is loaded instead of fitted).
//!
//! Per layer in order, every MLP before it replaced by its account (the `vpd` start's): the MLP's
//! inputs `h` on the training, held-out and eval sequences and its written Fisher at that hybrid
//! state; its native output `y = W_out σ(W_in h)` there. Then per start:
//!
//! * `vpd`: reads from `LIBRARY_DIR/blocks.L.c_fc.v.f64`, one rule per subcomponent of
//!   `blocks.L.down_proj` grown to its amplitude `qᵀσ(W_in h)`, writes from its `u`;
//! * `neurons`: the dense map as an account (`Account::neurons`), the baseline every account is
//!   measured against;
//!
//! each per round pruned and refined and (between rounds) its rules grown again on the moved reads
//! (`mlp_account::regrow`), pruned a last time and its rules shared. Every comparison of totals
//! prices the remainder with the Fisher scaled to the exact KL the account adds on the held-out
//! sequences (its shape from the Fisher, its scale from the KL). `OUT_DIR/functions.json`
//! gets per layer and start, after every stage: the account's sizes and bits, its remainder on each
//! split (nats per input, second order), its total at `n`, and the KL(model ‖ program) per eval
//! token this MLP's account adds at the hybrid state (local) and with every MLP so far replaced
//! (end to end); beside them
//! the plain factorization's and the dense map's bits, and the shared bodies.
//! `OUT_DIR/L{layer}.{start}.rules.json` holds the rules; `.reads.f64` and `.writes.f64` the
//! directions (row-major float64).

use gam_mpd::derivatives::vjp;
use gam_mpd::gates::gelu;
use gam_mpd::import::import_language_model;
use gam_mpd::masked::{Site, Target, kl_score_only, matrix, read_values, sampled_label_cotangent, sites};
use gam_mpd::mlp_account::{self, Account, dense_bits, plain_bits};
use gam_mpd::operator_program::{FamilyInputs, OperatorProgram, Trace};
use gam_linalg::faer_ndarray::{fast_abt, fast_atb};
use ndarray::{Array1, Array2, Axis};
use serde_json::{Value, json};
use std::collections::BTreeMap;
use std::f64::consts::LN_2;
use std::path::{Path, PathBuf};
use std::time::Instant;

fn read_rows(path: &Path, cols: usize) -> Result<Array2<f64>, String> {
    let bytes = std::fs::read(path).map_err(|e| format!("{}: {e}", path.display()))?;
    let values: Vec<f64> = bytes.chunks_exact(8).map(|c| f64::from_le_bytes([c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]])).collect();
    Array2::from_shape_vec((values.len() / cols, cols), values).map_err(|e| e.to_string())
}

fn write_rows(path: &Path, rows: &Array2<f64>) -> Result<(), String> {
    let bytes: Vec<u8> = rows.iter().flat_map(|v| v.to_le_bytes()).collect();
    std::fs::write(path, bytes).map_err(|e| format!("{}: {e}", path.display()))
}

/// One MLP: its input and output sites and maps.
struct Mlp {
    layer: String,
    input: Site,
    output: Site,
    w_in: Array2<f64>,
    w_out: Array2<f64>,
}

/// The trace of `inputs` with each MLP of `replaced` (in execution order) computing its account's
/// output from its hybrid input instead of its own.
fn hybrid(program: &OperatorProgram, inputs: &FamilyInputs, replaced: &[(&Mlp, &Account)]) -> Result<Trace, String> {
    let mut trace = program.execute(inputs, false).map_err(|e| e.to_string())?;
    for (mlp, account) in replaced {
        let h = read_values(&trace, &mlp.input)?;
        let active = read_values(&trace, &mlp.output)?;
        let node = mlp.output.writes[0];
        let own = fast_abt(&active, &mlp.w_out);
        let value = &trace.values[node] - &own + &account.apply(h.view());
        trace.values.truncate(node);
        trace.values.push(value);
        program.execute_from(inputs, &mut trace, node + 1).map_err(|e| e.to_string())?;
    }
    Ok(trace)
}

/// The MLP's inputs on `batches` at the hybrid state, and (when `draws > 0`) its written Fisher there.
fn collect(program: &OperatorProgram, batches: &[FamilyInputs], replaced: &[(&Mlp, &Account)], mlp: &Mlp, draws: usize) -> Result<(Array2<f64>, Option<Array2<f64>>), String> {
    let mut rows = Vec::new();
    let mut fisher: Option<Array2<f64>> = None;
    let mut count = 0usize;
    let no_rows = Target { logits: Array2::zeros((0, 0)), scored: None };
    for (b, inputs) in batches.iter().enumerate() {
        let trace = hybrid(program, inputs, replaced)?;
        rows.push(read_values(&trace, &mlp.input)?);
        for d in 0..draws {
            let seed = 0x317E_u64.wrapping_add((b * draws + d) as u64).wrapping_mul(0x9E37_79B9);
            let cotangent = sampled_label_cotangent(&trace.values[program.output], &no_rows, seed);
            let back = vjp(program, inputs, &trace, cotangent).map_err(|e| e.to_string())?;
            let node = mlp.output.writes[0];
            let g = back[node].clone().unwrap_or_else(|| Array2::zeros(trace.values[node].dim()));
            let outer = fast_atb(&g, &g);
            fisher = Some(match fisher.take() {
                Some(f) => f + outer,
                None => outer,
            });
            count += g.nrows();
        }
    }
    let views: Vec<_> = rows.iter().map(|r| r.view()).collect();
    let h = ndarray::concatenate(Axis(0), &views).map_err(|e| e.to_string())?;
    Ok((h, fisher.map(|f| f / count.max(1) as f64)))
}

/// Mean KL(model ‖ program) per token on `batches` with `replaced` accounting for their MLPs.
fn end_to_end(program: &OperatorProgram, batches: &[FamilyInputs], replaced: &[(&Mlp, &Account)]) -> Result<f64, String> {
    let (mut total, mut tokens) = (0.0, 0usize);
    for inputs in batches {
        let native = program.execute(inputs, false).map_err(|e| e.to_string())?;
        let target = Target::every_row(native.values[program.output].clone());
        drop(native);
        let trace = hybrid(program, inputs, replaced)?;
        total += kl_score_only(&target, &trace.values[program.output]).sum();
        tokens += inputs.rows;
    }
    Ok(total / tokens.max(1) as f64)
}

/// Mean KL per token between the program with `replaced` upstream and this MLP native, and the
/// same with `account` in this MLP's place: the KL this MLP's account adds at the hybrid state.
fn local_kl(program: &OperatorProgram, batches: &[FamilyInputs], replaced: &[(&Mlp, &Account)], mlp: &Mlp, account: &Account) -> Result<f64, String> {
    let (mut total, mut tokens) = (0.0, 0usize);
    let mut with: Vec<(&Mlp, &Account)> = replaced.to_vec();
    with.push((mlp, account));
    for inputs in batches {
        let before = hybrid(program, inputs, replaced)?;
        let target = Target::every_row(before.values[program.output].clone());
        drop(before);
        let after = hybrid(program, inputs, &with)?;
        total += kl_score_only(&target, &after.values[program.output]).sum();
        tokens += inputs.rows;
    }
    Ok(total / tokens.max(1) as f64)
}

struct Split {
    h: Array2<f64>,
    y: Array2<f64>,
}

fn main() -> Result<(), String> {
    gam_mpd::engine::log_to_stderr();
    let args: Vec<String> = std::env::args().collect();
    let usage = "mpd_mlp_functions_2951 TRAIN_EXPORT EVAL_EXPORT LIBRARY_DIR OUT_DIR [KEY=VALUE ...]";
    if args.len() < 5 {
        return Err(usage.to_string());
    }
    let (train_export, eval_export) = (PathBuf::from(&args[1]), PathBuf::from(&args[2]));
    let library = PathBuf::from(&args[3]);
    let out = PathBuf::from(&args[4]);
    let mut keys: BTreeMap<String, String> = BTreeMap::new();
    for kv in &args[5..] {
        let (k, v) = kv.split_once('=').ok_or(format!("{kv}: not KEY=VALUE"))?;
        keys.insert(k.to_string(), v.to_string());
    }
    let get = |k: &str, default: &str| keys.get(k).cloned().unwrap_or_else(|| default.to_string());
    let number = |k: &str, default: &str| -> Result<f64, String> { get(k, default).parse::<f64>().map_err(|e| format!("{k}: {e}")) };
    let layers: Vec<String> = get("layers", "0,1").split(',').map(str::to_string).collect();
    let starts: Vec<String> = get("starts", "vpd,neurons").split(',').map(str::to_string).collect();
    let (train, valid, eval, context) = (number("train", "64")? as usize, number("valid", "8")? as usize, number("eval", "16")? as usize, number("context", "512")? as usize);
    let observations = number("observations", "1e6")?;
    let grow_rows = number("grow_rows", "8192")? as usize;
    let (iterations, patience, draws) = (number("iterations", "200")? as usize, number("patience", "10")? as usize, number("draws", "2")? as usize);
    let rounds = number("rounds", "2")? as usize;
    let load = keys.get("load").map(PathBuf::from);
    std::fs::create_dir_all(&out).map_err(|e| format!("{}: {e}", out.display()))?;
    let started = Instant::now();

    let training = import_language_model(&train_export, train + valid, context)?;
    let model = &training.program;
    let evaluation = import_language_model(&eval_export, eval, context)?;
    let split = |family: &FamilyInputs, range: std::ops::Range<usize>| -> Vec<FamilyInputs> {
        range.map(|q| family.select(&(q * context..(q + 1) * context).collect::<Vec<_>>())).collect()
    };
    let train_batches = split(&training.contract.family, 0..train);
    let valid_batches = split(&training.contract.family, train..train + valid);
    let eval_batches = split(&evaluation.contract.family, 0..eval);

    let all = sites(model);
    let mut mlps = Vec::new();
    for layer in &layers {
        let find = |suffix: &str| all.iter().find(|s| s.name == format!("blocks.{layer}.{suffix}")).cloned().ok_or(format!("no site blocks.{layer}.{suffix}"));
        let (input, output) = (find("c_fc")?, find("down_proj")?);
        let (w_in, w_out) = (matrix(model, &input)?, matrix(model, &output)?);
        mlps.push(Mlp { layer: layer.clone(), input, output, w_in, w_out });
    }

    let mut report: Vec<Value> = Vec::new();
    let mut accounts: Vec<Account> = Vec::new();
    for (index, mlp) in mlps.iter().enumerate() {
        let replaced: Vec<(&Mlp, &Account)> = mlps[..index].iter().zip(&accounts).collect();
        let clock = Instant::now();
        let (h_train, fisher) = collect(model, &train_batches, &replaced, mlp, draws)?;
        let fisher = fisher.ok_or("no Fisher")?;
        let (h_valid, _) = collect(model, &valid_batches, &replaced, mlp, 0)?;
        let (h_eval, _) = collect(&evaluation.program, &eval_batches, &replaced, mlp, 0)?;
        let make = |h: Array2<f64>| {
            let y = mlp_account::native(&mlp.w_in, &mlp.w_out, h.view());
            Split { h, y }
        };
        let (tr, va, ev) = (make(h_train), make(h_valid), make(h_eval));
        let (hidden, d_in, d_out) = (mlp.w_in.nrows(), mlp.w_in.ncols(), mlp.w_out.nrows());
        let energy = |sp: &Split| 0.5 * (&sp.y * &sp.y.dot(&fisher)).sum() / sp.h.nrows() as f64;
        eprintln!(
            "layer {}: {} train, {} held-out, {} eval inputs at the hybrid state; ½E‖y‖²_F {:.4} nats; {:.0}s",
            mlp.layer,
            tr.h.nrows(),
            va.h.nrows(),
            ev.h.nrows(),
            energy(&tr),
            clock.elapsed().as_secs_f64()
        );
        let mut layer_report = json!({
            "layer": mlp.layer, "train_inputs": tr.h.nrows(), "valid_inputs": va.h.nrows(), "eval_inputs": ev.h.nrows(),
            "output_energy_nats": {"train": energy(&tr), "eval": energy(&ev)},
            "dense_bits": dense_bits(d_in, hidden, d_out),
        });
        let saved = load.as_ref().map(|dir| dir.join(format!("L{}.vpd", mlp.layer))).filter(|b| b.with_extension("rules.json").exists());
        if let Some(base) = saved {
            let text = std::fs::read_to_string(base.with_extension("rules.json")).map_err(|e| e.to_string())?;
            let account = Account {
                reads: read_rows(&base.with_extension("reads.f64"), d_in)?,
                rules: serde_json::from_str(&text).map_err(|e| e.to_string())?,
                writes: read_rows(&base.with_extension("writes.f64"), d_out)?,
                offset: read_rows(&base.with_extension("offset.f64"), d_out)?.row(0).to_owned(),
            };
            let mut replaced_now = replaced.clone();
            replaced_now.push((mlp, &account));
            let together = end_to_end(&evaluation.program, &eval_batches, &replaced_now)?;
            let r_eval = account.remainder(ev.h.view(), ev.y.view(), &fisher);
            eprintln!("layer {} loaded from {}: remainder eval {r_eval:.4}, KL all {together:.4}", mlp.layer, base.display());
            layer_report["loaded"] = json!({"from": base, "remainder_eval": r_eval, "kl_all": together, "incoming": account.reads.nrows(), "outgoing": account.rules.len()});
            report.push(layer_report);
            accounts.push(account);
            continue;
        }
        let mut chosen: Option<Account> = None;
        for start in &starts {
            let mut stages: Vec<Value> = Vec::new();
            let mut account = match start.as_str() {
                "neurons" => Account::neurons(&mlp.w_in, &mlp.w_out),
                "vpd" => {
                    let input_name = format!("blocks.{}.c_fc", mlp.layer);
                    let output_name = format!("blocks.{}.down_proj", mlp.layer);
                    let reads = read_rows(&library.join(format!("{input_name}.v.f64")), d_in)?;
                    let u_in = read_rows(&library.join(format!("{input_name}.u.f64")), hidden)?;
                    let q = read_rows(&library.join(format!("{output_name}.v.f64")), hidden)?;
                    let writes = read_rows(&library.join(format!("{output_name}.u.f64")), d_out)?;
                    layer_report["plain_bits"] = json!(plain_bits(d_in, hidden, d_out, reads.nrows(), q.nrows()));
                    layer_report["library"] = json!({"incoming": reads.nrows(), "outgoing": q.nrows(), "u_in_rows": u_in.nrows()});
                    // Grow on evenly strided training inputs, each step checked on the held-out ones.
                    let stride = tr.h.nrows().div_ceil(grow_rows.max(1)).max(1);
                    let picked: Vec<usize> = (0..tr.h.nrows()).step_by(stride).collect();
                    let h_grow = tr.h.select(Axis(0), &picked);
                    let amplitudes_targets = |h: &Array2<f64>| (fast_abt(h, &reads), fast_abt(&fast_abt(h, &mlp.w_in).mapv(gelu), &q));
                    let (a, targets) = amplitudes_targets(&h_grow);
                    let (a_check, t_check) = amplitudes_targets(&va.h);
                    let pfp = writes.dot(&fisher).dot(&writes.t()).diag().to_owned();
                    let prices: Vec<f64> = pfp.iter().map(|w| 0.5 * observations * w).collect();
                    let clock = Instant::now();
                    let rules = mlp_account::grow((&a, &targets), (&a_check, &t_check), &prices);
                    eprintln!("layer {} vpd: {} rules grown on {} inputs, {:.0}s", mlp.layer, rules.len(), picked.len(), clock.elapsed().as_secs_f64());
                    Account { reads, rules, writes, offset: Array1::zeros(d_out) }
                }
                other => return Err(format!("start {other}: vpd or neurons")),
            };
            let mut stage = |name: &str, account: &Account, extra: Value| -> Result<(), String> {
                let replaced_now: Vec<(&Mlp, &Account)> = mlps[..index].iter().zip(&accounts).chain(std::iter::once((mlp, account))).collect();
                let alone = local_kl(&evaluation.program, &eval_batches, &replaced, mlp, account)?;
                let together = if index == 0 { alone } else { end_to_end(&evaluation.program, &eval_batches, &replaced_now)? };
                let remainder = |sp: &Split| account.remainder(sp.h.view(), sp.y.view(), &fisher);
                let (r_train, r_valid, r_eval) = (remainder(&tr), remainder(&va), remainder(&ev));
                let mut sizes = BTreeMap::<String, usize>::new();
                for r in &account.rules {
                    *sizes.entry(format!("{} in, {} units", r.inputs.len(), r.units.len())).or_default() += 1;
                }
                let structure = account.structure_bits();
                eprintln!(
                    "layer {} {start} {name}: m {} K {} literals {} ({:.3e} bits); remainder train {r_train:.4} held-out {r_valid:.4} eval {r_eval:.4}; KL local {alone:.4} all {together:.4}; {:.0}s",
                    mlp.layer,
                    account.reads.nrows(),
                    account.rules.len(),
                    account.literals(),
                    structure,
                    started.elapsed().as_secs_f64()
                );
                stages.push(json!({
                    "stage": name, "incoming": account.reads.nrows(), "outgoing": account.rules.len(), "literals": account.literals(),
                    "structure_bits": structure, "rule_literals": account.rules.iter().map(|r| r.literals()).sum::<usize>(),
                    "remainder_nats": {"train": r_train, "valid": r_valid, "eval": r_eval},
                    "total_bits_eval": structure + observations * r_eval / LN_2,
                    "kl_local": alone, "kl_all": together, "sizes": sizes, "extra": extra,
                }));
                Ok(())
            };
            let initial_stage = if start == "vpd" { "grown" } else { "exact" };
            stage(initial_stage, &account, json!({}))?;
            // The Fisher at the exact KL's scale: the KL this account adds on the held-out
            // sequences over its second-order remainder there (the metric's shape from the Fisher,
            // its scale from the exact KL).
            let calibrated = |account: &Account| -> Result<(Array2<f64>, f64), String> {
                let r = account.remainder(va.h.view(), va.y.view(), &fisher);
                if !(r > 0.0) {
                    return Ok((fisher.clone(), 1.0));
                }
                let kl = local_kl(model, &valid_batches, &replaced, mlp, account)?;
                let scale = if kl > 0.0 { kl / r } else { 1.0 };
                eprintln!("layer {} {start}: held-out KL {kl:.4} over remainder {r:.4}: Fisher scale {scale:.3}", mlp.layer);
                Ok((&fisher * scale, scale))
            };
            let grow_picked: Vec<usize> = (0..tr.h.nrows()).step_by(tr.h.nrows().div_ceil(grow_rows.max(1)).max(1)).collect();
            for round in 1..=rounds {
                let (metric, scale) = calibrated(&account)?;
                let pruned = mlp_account::prune(&mut account, (tr.h.view(), tr.y.view()), (va.h.view(), va.y.view()), &metric, observations)?;
                stage(&format!("pruned {round}"), &account, json!({"pruned": pruned, "fisher_scale": scale}))?;
                let refined = mlp_account::refine(&mut account, (tr.h.view(), tr.y.view()), (va.h.view(), va.y.view()), &fisher, iterations, patience);
                stage(&format!("refined {round}"), &account, json!(refined))?;
                if round < rounds {
                    let (metric, scale) = calibrated(&account)?;
                    let replaced = mlp_account::regrow(&mut account, (tr.h.view(), tr.y.view()), (va.h.view(), va.y.view()), &metric, observations, &grow_picked)?;
                    stage(&format!("regrown {round}"), &account, json!({"replaced": replaced, "fisher_scale": scale}))?;
                }
            }
            let (metric, scale) = calibrated(&account)?;
            let pruned = mlp_account::prune(&mut account, (tr.h.view(), tr.y.view()), (va.h.view(), va.y.view()), &metric, observations)?;
            stage("final", &account, json!({"pruned": pruned, "fisher_scale": scale}))?;
            mlp_account::canonical(&mut account);
            let sample: Vec<usize> = (0..tr.h.nrows()).step_by(tr.h.nrows().div_ceil(4096).max(1)).collect();
            let (h_s, y_s) = (tr.h.select(Axis(0), &sample), tr.y.select(Axis(0), &sample));
            let (metric, _) = calibrated(&account)?;
            let bodies = mlp_account::share(&mut account, h_s.view(), y_s.view(), &metric, observations);
            let rule_bits = mlp_account::shared_rule_bits(&account, &bodies);
            let shared_total = (account.reads.len() + account.writes.len() + account.offset.len()) as f64 * mlp_account::LITERAL_BITS + rule_bits;
            let shown: Vec<&mlp_account::Body> = bodies.iter().take(20).collect();
            stage("shared", &account, json!({"bodies": bodies.len(), "rule_bits_shared": rule_bits, "structure_bits_shared": shared_total, "top": shown}))?;
            let base = out.join(format!("L{}.{start}", mlp.layer));
            std::fs::write(base.with_extension("rules.json"), serde_json::to_string(&account.rules).map_err(|e| e.to_string())?).map_err(|e| e.to_string())?;
            write_rows(&base.with_extension("reads.f64"), &account.reads)?;
            write_rows(&base.with_extension("writes.f64"), &account.writes)?;
            write_rows(&base.with_extension("offset.f64"), &account.offset.clone().insert_axis(Axis(0)))?;
            layer_report[start.as_str()] = json!(stages);
            if start == "vpd" || chosen.is_none() {
                chosen = Some(account);
            }
            let mut written = report.clone();
            written.push(layer_report.clone());
            let text = serde_json::to_string_pretty(&json!({"observations": observations, "layers": written})).map_err(|e| e.to_string())?;
            std::fs::write(out.join("functions.json"), text).map_err(|e| e.to_string())?;
        }
        report.push(layer_report);
        accounts.push(chosen.ok_or("no start")?);
    }
    eprintln!("done in {:.0}s", started.elapsed().as_secs_f64());
    Ok(())
}
