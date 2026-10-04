//! The named vocabulary of a decomposition's per-word sets (#2951), for the natural-language
//! autoencoder `bench/vpd_2951/vpd_nl_autoencoder.py`, where the text is the only channel.
//!
//! `mpd_nl_concepts_2951 SETS TRAIN EVAL OBSERVATIONS LABEL_BITS SECONDS REPRICE OUT ORACLE...`
//!
//! `SETS` holds, raw little-endian: `indptr.i64` and `indices.i64` (CSR over every sequence's
//! positions, sequence after sequence: the subcomponents each word may run), optionally `start.u8`
//! (per set member, 1 where it runs at the start; every member when absent), `missing.f32` (per
//! set member, its price in nats at the start's programs, KL with it off minus with it on),
//! `program.f64` (per subcomponent, its description bits under the library's own description), and
//! `meta.json` (`universe`, `context`). Any library's sets work. `TRAIN` and `EVAL` are sequence ranges
//! `lo:hi`. The vocabulary is fitted on `TRAIN` ([`gam_mpd::concepts::fit`], every concept's name
//! costing `LABEL_BITS` in the library, the error `OBSERVATIONS · KL / ln 2`), then the words of
//! `EVAL` are coded with it frozen ([`gam_mpd::concepts::Model::encode`]). The fit gets the share
//! of `SECONDS` its words are of all the words coded, the coding the rest; `REPRICE` (0 or 1) says
//! whether exact prices are measured again when nothing is proposed at carried prices.
//!
//! `ORACLE...` is a command that runs the model on the sequences `lo:hi` it is given as one more
//! argument (`vpd_nl_autoencoder.py oracle`), started once for `TRAIN` and once for `EVAL`. It
//! reads requests on stdin, each `u64 kind, u64 words, u64 members, (words + 1) × u64 row pointer,
//! members × u32` (every word's program), and for kinds 1 and 2 also the sets (`u64 count, (words
//! + 1) × u64 row pointer, count × u32`). It answers kind 0 with `words × f32`, the exact KL in
//! nats of the model running each word's program, kind 1 with `count × f32`, each set member's
//! price in nats at the programs (its word's KL with it off minus with it on), and kind 2 with
//! `count × f32`, each set member's slope `−∂(Σ_s KL_s)/∂m_tj`.
//!
//! Writes to `OUT`:
//!
//! * `concepts.json`: the fit's exact totals and rounds, the coding's rounds, and the vocabulary's
//!   members, rates and program bits;
//! * `invoked.indptr.i64`, `invoked.indices.i64`: per word of `TRAIN` then `EVAL`, the concepts it names;
//! * `bits.f64`: per word, its names, program and `n KL / ln 2` bits (exact), and its own set's program bits.

use gam_mpd::concepts::{Bits, Model, Oracle, Run, Sets, Words, fit};
use serde_json::json;
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::process::{Child, ChildStdin, ChildStdout, Command, Stdio};
use std::time::{Duration, Instant};

fn read<const W: usize>(path: &Path) -> Result<Vec<[u8; W]>, String> {
    let bytes = std::fs::read(path).map_err(|e| format!("{}: {e}", path.display()))?;
    if bytes.len() % W != 0 {
        return Err(format!("{}: {} bytes are not {W}-byte values", path.display(), bytes.len()));
    }
    Ok(bytes.chunks_exact(W).map(|c| c.try_into().expect("chunk width")).collect())
}

fn range(spec: &str) -> Result<(usize, usize), String> {
    let (lo, hi) = spec.split_once(':').ok_or(format!("{spec}: not lo:hi"))?;
    Ok((lo.parse().map_err(|e| format!("{spec}: {e}"))?, hi.parse().map_err(|e| format!("{spec}: {e}"))?))
}

/// The model as a child process on one range of sequences.
struct Process {
    child: Child,
    to: std::io::BufWriter<ChildStdin>,
    from: std::io::BufReader<ChildStdout>,
}

impl Process {
    fn start(command: &[String], rows: &str) -> Result<Self, String> {
        let mut child = Command::new(&command[0])
            .args(&command[1..])
            .arg(rows)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .spawn()
            .map_err(|e| format!("oracle: {e}"))?;
        let to = std::io::BufWriter::new(child.stdin.take().ok_or("oracle stdin")?);
        let from = std::io::BufReader::new(child.stdout.take().ok_or("oracle stdout")?);
        Ok(Self { child, to, from })
    }

    fn rows(message: &mut Vec<u8>, rows: &[Vec<u32>]) {
        let mut at = 0u64;
        message.extend(at.to_le_bytes());
        for p in rows {
            at += p.len() as u64;
            message.extend(at.to_le_bytes());
        }
        for p in rows {
            for j in p {
                message.extend(j.to_le_bytes());
            }
        }
    }

    fn ask(&mut self, message: &[u8], answers: usize) -> Result<Vec<f64>, String> {
        self.to.write_all(message).and_then(|()| self.to.flush()).map_err(|e| format!("oracle: {e}"))?;
        let mut answer = vec![0u8; 4 * answers];
        self.from.read_exact(&mut answer).map_err(|e| format!("oracle: {e}"))?;
        Ok(answer.chunks_exact(4).map(|c| f64::from(f32::from_le_bytes([c[0], c[1], c[2], c[3]]))).collect())
    }

    fn finish(self) -> Result<(), String> {
        let Self { mut child, to, from } = self;
        drop(to);
        drop(from);
        child.wait().map_err(|e| format!("oracle: {e}"))?;
        Ok(())
    }
}

impl Oracle for Process {
    fn kl(&mut self, programs: &[Vec<u32>]) -> Result<Vec<f64>, String> {
        let members: usize = programs.iter().map(Vec::len).sum();
        let mut message = Vec::with_capacity(24 + 8 * (programs.len() + 1) + 4 * members);
        for x in [0, programs.len() as u64, members as u64] {
            message.extend(x.to_le_bytes());
        }
        Self::rows(&mut message, programs);
        self.ask(&message, programs.len())
    }

    fn prices(&mut self, programs: &[Vec<u32>], sets: &Sets) -> Result<Vec<f64>, String> {
        self.per_member(1, programs, sets)
    }

    fn slopes(&mut self, programs: &[Vec<u32>], sets: &Sets) -> Result<Vec<f64>, String> {
        self.per_member(2, programs, sets)
    }
}

impl Process {
    /// A request answered per set member: kind 1 (exact prices) or 2 (slopes).
    fn per_member(&mut self, kind: u64, programs: &[Vec<u32>], sets: &Sets) -> Result<Vec<f64>, String> {
        let members: usize = programs.iter().map(Vec::len).sum();
        let mut message = Vec::with_capacity(32 + 16 * (programs.len() + 1) + 4 * (members + sets.indices.len()));
        for x in [kind, programs.len() as u64, members as u64] {
            message.extend(x.to_le_bytes());
        }
        Self::rows(&mut message, programs);
        message.extend((sets.indices.len() as u64).to_le_bytes());
        message.extend(sets.indptr.iter().flat_map(|p| (*p as u64).to_le_bytes()));
        message.extend(sets.indices.iter().flat_map(|j| j.to_le_bytes()));
        self.ask(&message, sets.indices.len())
    }
}

fn main() -> Result<(), String> {
    gam_mpd::engine::log_to_stderr();
    let started = Instant::now();
    let args: Vec<String> = std::env::args().collect();
    let usage = "usage: mpd_nl_concepts_2951 SETS TRAIN EVAL OBSERVATIONS LABEL_BITS SECONDS REPRICE OUT ORACLE...";
    let dir = PathBuf::from(args.get(1).ok_or(usage)?);
    let train = range(args.get(2).ok_or(usage)?)?;
    let eval = range(args.get(3).ok_or(usage)?)?;
    let observations: f64 = args.get(4).ok_or(usage)?.parse().map_err(|e| format!("OBSERVATIONS: {e}"))?;
    let label_bits: f64 = args.get(5).ok_or(usage)?.parse().map_err(|e| format!("LABEL_BITS: {e}"))?;
    let seconds: f64 = args.get(6).ok_or(usage)?.parse().map_err(|e| format!("SECONDS: {e}"))?;
    let reprice = match args.get(7).map(String::as_str) {
        Some("1") => true,
        Some("0") => false,
        _ => return Err(format!("REPRICE is 0 or 1; {usage}")),
    };
    let out = PathBuf::from(args.get(8).ok_or(usage)?);
    let command = args.get(9..).filter(|c| !c.is_empty()).ok_or(usage)?;
    let meta: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(dir.join("meta.json")).map_err(|e| e.to_string())?).map_err(|e| e.to_string())?;
    let universe = meta["universe"].as_u64().ok_or("meta.json: universe")? as usize;
    let context = meta["context"].as_u64().ok_or("meta.json: context")? as usize;
    let indptr: Vec<usize> = read::<8>(&dir.join("indptr.i64"))?.into_iter().map(|b| i64::from_le_bytes(b) as usize).collect();
    let indices: Vec<u32> = read::<8>(&dir.join("indices.i64"))?.into_iter().map(|b| i64::from_le_bytes(b) as u32).collect();
    let program: Vec<f64> = read::<8>(&dir.join("program.f64"))?.into_iter().map(f64::from_le_bytes).collect();
    let missing: Vec<f64> = read::<4>(&dir.join("missing.f32"))?.into_iter().map(|b| f64::from(f32::from_le_bytes(b))).collect();
    let start: Vec<bool> =
        if dir.join("start.u8").exists() { read::<1>(&dir.join("start.u8"))?.into_iter().map(|b| b[0] == 1).collect() } else { vec![true; indices.len()] };
    if program.len() != universe || missing.len() != indices.len() || start.len() != indices.len() {
        return Err("program.f64 needs one value per subcomponent, missing.f32 and start.u8 one per set member".to_string());
    }
    let sequences = (indptr.len() - 1) / context;
    if train.1 > sequences || eval.1 > sequences || train.0 >= train.1 || eval.0 >= eval.1 {
        return Err(format!("ranges outside the {sequences} sequences"));
    }
    let rows = |(lo, hi): (usize, usize)| -> Result<(Sets, Vec<bool>, Vec<f64>), String> {
        let (a, b) = (indptr[lo * context], indptr[hi * context]);
        let sets = Sets::new(universe, indptr[lo * context..=hi * context].iter().map(|p| p - a).collect(), indices[a..b].to_vec())?;
        Ok((sets, start[a..b].to_vec(), missing[a..b].to_vec()))
    };
    let ((fitted_on, fit_start, fit_prices), (coded_on, code_start, code_prices)) = (rows(train)?, rows(eval)?);
    let share = fitted_on.rows() as f64 / (fitted_on.rows() + coded_on.rows()) as f64;
    let mut oracle = Process::start(command, &format!("{}:{}", train.0, train.1))?;
    let run = |deadline: Instant| Run { observations, deadline, reprice };
    let fitted = fit(
        &Words { sets: &fitted_on, start: &fit_start, prices: &fit_prices },
        &program,
        label_bits,
        run(started + Duration::from_secs_f64(share * seconds)),
        &mut oracle,
    )?;
    oracle.finish()?;
    let fit_seconds = started.elapsed().as_secs_f64();
    let words = fitted_on.rows() as f64;
    let scale = observations / std::f64::consts::LN_2;
    let mean_kl = fitted.kl.iter().sum::<f64>() / words;
    println!(
        "fit {fit_seconds:.0}s, {} rounds: exact {:.1} bits/word (names + program + library + n KL/ln 2, KL {mean_kl:.4}) from the own sets' {:.1}",
        fitted.rounds.len(),
        fitted.total_bits / words,
        fitted.start_bits / words
    );
    let model = Model::new(&fitted);
    let mut oracle = Process::start(command, &format!("{}:{}", eval.0, eval.1))?;
    let coded = model.encode(&Words { sets: &coded_on, start: &code_start, prices: &code_prices }, run(started + Duration::from_secs_f64(seconds)), &mut oracle)?;
    oracle.finish()?;
    std::fs::create_dir_all(&out).map_err(|e| e.to_string())?;
    let mut ptr: Vec<i64> = vec![0];
    let mut invoked: Vec<i64> = Vec::new();
    let mut bits: Vec<f64> = Vec::new();
    let mut summary = Vec::new();
    // The fitted words' bits under the frozen model, with the fit's exact KL.
    let silent: f64 = model.concepts.iter().map(|c| -(1.0 - c.invoked).log2()).sum();
    let fitted_bits: Vec<Bits> = model
        .fitted
        .iter()
        .zip(&fitted.kl)
        .map(|(cs, k)| Bits {
            names: silent + cs.iter().map(|c| -model.concepts[*c as usize].invoked.log2() + (1.0 - model.concepts[*c as usize].invoked).log2()).sum::<f64>(),
            program: cs.iter().map(|c| model.concepts[*c as usize].program).sum(),
            kl: *k,
        })
        .collect();
    for (name, spec, sets, starts, invocations, coded_bits) in
        [("train", train, &fitted_on, &fit_start, &model.fitted, &fitted_bits), ("eval", eval, &coded_on, &code_start, &coded.invoked, &coded.bits)]
    {
        // The start's program bits (the own sets').
        let own: Vec<f64> = (0..sets.rows())
            .map(|t| (sets.indptr[t]..sets.indptr[t + 1]).filter(|k| starts[*k]).map(|k| program[sets.indices[k] as usize]).sum())
            .collect();
        let mut sum = [0.0; 4];
        for ((cs, b), o) in invocations.iter().zip(coded_bits).zip(&own) {
            invoked.extend(cs.iter().map(|c| i64::from(*c)));
            ptr.push(invoked.len() as i64);
            let row = [b.names, b.program, scale * b.kl, *o];
            for (s, x) in sum.iter_mut().zip(row) {
                *s += x;
            }
            bits.extend(row);
        }
        let n = sets.rows() as f64;
        println!(
            "{name}: per word names {:.1} + program {:.1} + n KL/ln 2 {:.1} = {:.1} bits; the own sets' program {:.1}",
            sum[0] / n,
            sum[1] / n,
            sum[2] / n,
            (sum[0] + sum[1] + sum[2]) / n,
            sum[3] / n
        );
        summary.push(json!({"rows": name, "sequences": [spec.0, spec.1], "words": sets.rows(),
            "names": sum[0] / n, "program": sum[1] / n, "error": sum[2] / n, "own_program": sum[3] / n}));
    }
    let write = |name: &str, bytes: Vec<u8>| std::fs::write(out.join(name), bytes).map_err(|e| e.to_string());
    write("invoked.indptr.i64", ptr.iter().flat_map(|x| x.to_le_bytes()).collect())?;
    write("invoked.indices.i64", invoked.iter().flat_map(|x| x.to_le_bytes()).collect())?;
    write("bits.f64", bits.iter().flat_map(|x| x.to_le_bytes()).collect())?;
    let report = json!({
        "universe": universe, "context": context, "observations": observations, "label_bits": label_bits,
        "fit_seconds": fit_seconds, "train_words": fitted_on.rows(), "start_bits": fitted.start_bits, "total_bits": fitted.total_bits,
        "train_kl": mean_kl, "rounds": fitted.rounds, "coding_rounds": coded.rounds, "coded": summary, "concepts": model.concepts,
    });
    write("concepts.json", serde_json::to_vec(&report).map_err(|e| e.to_string())?)
}
