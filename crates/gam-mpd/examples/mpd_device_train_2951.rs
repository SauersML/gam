//! A library trained on a device (#2951, `gam_mpd::device_train`).
//!
//! `mpd_device_train_2951 EXPORT_DIR LIBRARY_DIR OUT_DIR OBSERVATIONS TRAIN [BATCH] [STEPS] [RATE] [MICRO] [EVAL] [SYNC] [CONTEXT]`
//!
//! `LIBRARY_DIR` holds a library per site (`{site}.v.f64`, `{site}.u.f64`, as `mpd_e2e_2951` and
//! `mpd_site_fit_2951` write them; a site without files stays native), and where `mpd_e2e_2951` fitted
//! it, the statistics it was fitted in (`{site}.fisher.f64`, `{site}.moment.f64`, on the inputs
//! under the explanation upstream): each site's code and description use those, else the native
//! model's statistics on the first training sequences. The library trains on the corner run's code
//! (`gam_mpd::device_train`, module note) at `n = OBSERVATIONS`. Every update sums the
//! gradients of `BATCH` training sequences (default 32, of the export's first `TRAIN`, cycled),
//! `MICRO` at a time (default 8), and takes one Adam step of `RATE` (default 3e-4, a share of
//! each factor's root mean square entry); `STEPS` updates (default 1000). Every `SYNC` updates
//! (default 25) and at the end, the library goes to `OUT_DIR/library/` as `mpd_e2e_2951` reads a
//! given library (`{site}.{v,u,fisher,moment}.f64`, `{site}.json` with its blocks and bits), so
//! `mpd_e2e_2951 … given=OUT_DIR/library` scores it end to end, and the code of the `EVAL` sequences
//! after the training ones (default 8) in float64 to `OUT_DIR/evals.json`. With
//! several CUDA devices each holds a replica: a batch's sequences are split among them, their
//! gradients summed, and every replica takes the same update. Without CUDA it trains on the Apple
//! GPU where there is one (f32 steps); that device has no float64, so each eval runs on the host,
//! a float64 trainer built from the synced library.

use gam_gpu::tensor::{Arithmetic, Device};
use gam_mpd::describe::{Geometry, Metric, Structured, declared_charts};
use gam_mpd::device_train::{Settings, Tally, Trainer, statistics};
use gam_mpd::import::import_language_model;
use gam_mpd::masked::{Library, Masked, matrix, sites};
use ndarray::{Array1, Array2};
use serde_json::json;
use std::path::{Path, PathBuf};
use std::time::Instant;

fn read_f64(path: &Path, cols: usize) -> Result<Array2<f64>, String> {
    let bytes = std::fs::read(path).map_err(|e| format!("{}: {e}", path.display()))?;
    let values: Vec<f64> = bytes.chunks_exact(8).map(|c| f64::from_le_bytes([c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]])).collect();
    Array2::from_shape_vec((values.len() / cols, cols), values).map_err(|e| format!("{}: {e}", path.display()))
}

fn write_f64(path: &Path, m: &Array2<f64>) -> Result<(), String> {
    let bytes: Vec<u8> = m.iter().flat_map(|v| v.to_le_bytes()).collect();
    let partial = path.with_extension("partial");
    std::fs::write(&partial, bytes).map_err(|e| format!("{}: {e}", partial.display()))?;
    std::fs::rename(&partial, path).map_err(|e| format!("{}: {e}", path.display()))
}

fn arg<T: std::str::FromStr>(args: &[String], i: usize, default: T, name: &str) -> Result<T, String>
where
    T::Err: std::fmt::Display,
{
    args.get(i).map_or(Ok(default), |v| v.parse().map_err(|e| format!("{name}: {e}")))
}

/// A replica's result, or its panic's message.
fn joined<T>(handle: std::thread::ScopedJoinHandle<'_, Result<T, String>>) -> Result<T, String> {
    handle.join().map_err(|payload| {
        let message = payload.downcast_ref::<&str>().map(|s| (*s).to_string()).or_else(|| payload.downcast_ref::<String>().cloned());
        format!("a replica panicked: {}", message.as_deref().unwrap_or("(no message)"))
    })?
}

fn main() -> Result<(), String> {
    gam_mpd::engine::log_to_stderr();
    let args: Vec<String> = std::env::args().collect();
    let usage = "mpd_device_train_2951 EXPORT_DIR LIBRARY_DIR OUT_DIR OBSERVATIONS TRAIN [BATCH] [STEPS] [RATE] [MICRO] [EVAL] [SYNC] [CONTEXT]";
    let export = PathBuf::from(args.get(1).ok_or(usage)?);
    let given = PathBuf::from(args.get(2).ok_or(usage)?);
    let out = PathBuf::from(args.get(3).ok_or(usage)?);
    let observations: f64 = args.get(4).ok_or(usage)?.parse().map_err(|e| format!("OBSERVATIONS: {e}"))?;
    let train: usize = args.get(5).ok_or(usage)?.parse().map_err(|e| format!("TRAIN: {e}"))?;
    let batch: usize = arg(&args, 6, 32, "BATCH")?;
    let steps: usize = arg(&args, 7, 1000, "STEPS")?;
    let rate: f64 = arg(&args, 8, 3e-4, "RATE")?;
    let micro: usize = arg(&args, 9, 8, "MICRO")?;
    let eval: usize = arg(&args, 10, 8, "EVAL")?;
    let sync: usize = arg(&args, 11, 25, "SYNC")?;
    let context: usize = arg(&args, 12, 512, "CONTEXT")?;
    if train == 0 || batch == 0 || micro == 0 || sync == 0 {
        return Err(usage.to_string());
    }
    std::fs::create_dir_all(&out).map_err(|e| format!("{}: {e}", out.display()))?;
    let imported = import_language_model(&export, train + eval, context)?;
    let model = &imported.program;
    let family = &imported.contract.family;
    let sequences = |list: &[usize]| family.select(&list.iter().flat_map(|s| s * context..(s + 1) * context).collect::<Vec<_>>());
    let (mut chosen, mut libraries, mut fitted) = (Vec::new(), Vec::new(), Vec::new());
    for site in sites(model) {
        let v_path = given.join(format!("{}.v.f64", site.name));
        if !v_path.exists() {
            continue;
        }
        let w = matrix(model, &site)?;
        let (d_out, d_in) = w.dim();
        libraries.push(Library { v: read_f64(&v_path, d_in)?, u: read_f64(&given.join(format!("{}.u.f64", site.name)), d_out)?, mean: Array1::zeros(d_in) });
        // The statistics the library was fitted in, when it was.
        let (f_path, m_path) = (given.join(format!("{}.fisher.f64", site.name)), given.join(format!("{}.moment.f64", site.name)));
        fitted.push(if f_path.exists() && m_path.exists() {
            Some(gam_mpd::pieces::Site { mean: Array1::zeros(d_in), second_moment: read_f64(&m_path, d_in)?, fisher: read_f64(&f_path, d_out)?, w })
        } else {
            None
        });
        chosen.push(site);
    }
    if chosen.is_empty() {
        return Err(format!("{}: no library", given.display()));
    }
    let mut devices = Device::accelerators(gam_gpu::global_policy()).map_err(|e| e.to_string())?;
    if devices.is_empty() {
        devices.extend(Device::single_precision(gam_gpu::global_policy()).map_err(|e| e.to_string())?);
    }
    if devices.is_empty() {
        devices.push(Device::host());
    }
    let device = devices[0].clone();
    let arithmetic = if device.is_host() { Arithmetic::F64 } else { Arithmetic::Tf32 };
    let started = Instant::now();
    let chunks = |range: std::ops::Range<usize>| -> Vec<Vec<usize>> { range.collect::<Vec<_>>().chunks(micro).map(<[usize]>::to_vec).collect() };
    let measured: Vec<gam_mpd::pieces::Site> = if fitted.iter().all(Option::is_some) {
        fitted.into_iter().flatten().collect()
    } else {
        let statistic_batches: Vec<_> = chunks(0..train.min(batch)).iter().map(|c| sequences(c)).collect();
        let native = statistics(&device, model, &chosen, &statistic_batches, 2, 0xDE5C)?;
        fitted.into_iter().zip(native).map(|(f, n)| f.unwrap_or(n)).collect()
    };
    let fishers: Vec<Array2<f64>> = measured.iter().map(|m| m.fisher.clone()).collect();
    let moments: Vec<Array2<f64>> = measured.iter().map(|m| m.second_moment.clone()).collect();
    // Each subcomponent priced by its description in its site's declared charts, as site_fit and
    // e2e price it (gam_mpd::describe::Structured).
    let describe = Structured::new(
        chosen
            .iter()
            .zip(&measured)
            .map(|(site, statistics)| {
                let (writers, readers) = declared_charts(model, site)?;
                Geometry::new(Metric::of(statistics, observations), writers, readers)
            })
            .collect::<Result<_, String>>()?,
    );
    drop(measured);
    let settings = Settings { observations, rate, betas: (0.9, 0.999), epsilon: 1e-12 };
    let mut replicas = Vec::new();
    for replica in &devices {
        let masked = Masked::build(model, chosen.clone(), libraries.clone())?;
        replicas.push(Trainer::new(replica, (model, &chosen), masked, (&describe, &fishers), settings, arithmetic)?);
    }
    drop(libraries);
    eprintln!("{} sites on {} × {} ({arithmetic:?}), statistics and start {:.0}s", chosen.len(), devices.len(), device.name(), started.elapsed().as_secs_f64());
    let evals: Vec<_> = chunks(train..train + eval).iter().map(|c| sequences(c)).collect();
    let mut log = Vec::new();
    let library_dir = out.join("library");
    std::fs::create_dir_all(&library_dir).map_err(|e| format!("{}: {e}", library_dir.display()))?;
    let evaluate = |trainer: &mut Trainer, step: usize, log: &mut Vec<serde_json::Value>| -> Result<(), String> {
        trainer.sync(&describe)?;
        let bits = trainer.bits()?;
        let masked = trainer.masked();
        for (k, site) in chosen.iter().enumerate() {
            let library = masked.library(k)?;
            let name = &site.name;
            write_f64(&library_dir.join(format!("{name}.v.f64")), &library.v)?;
            write_f64(&library_dir.join(format!("{name}.u.f64")), &library.u)?;
            write_f64(&library_dir.join(format!("{name}.fisher.f64")), &fishers[k])?;
            write_f64(&library_dir.join(format!("{name}.moment.f64")), &moments[k])?;
            let record = json!({"site": name, "ranks": vec![1; library.v.nrows()], "bits": bits[k],
                                "trained": {"from": given, "steps": step, "n": observations, "rate": rate, "batch": batch, "train": train, "context": context}});
            std::fs::write(library_dir.join(format!("{name}.json")), record.to_string()).map_err(|e| e.to_string())?;
        }
        if evals.is_empty() {
            return Ok(());
        }
        let mut tally = Tally::default();
        if device.float64() {
            for inputs in &evals {
                tally.add(&trainer.evaluate(inputs)?);
            }
        } else {
            // The first replica's library, evaluated in float64 on the host.
            let libraries = (0..chosen.len()).map(|k| masked.library(k)).collect::<Result<Vec<_>, _>>()?;
            let masked = Masked::build(model, chosen.clone(), libraries)?;
            let mut host = Trainer::new(&Device::host(), (model, &chosen), masked, (&describe, &fishers), settings, Arithmetic::F64)?;
            for inputs in &evals {
                tally.add(&host.evaluate(inputs)?);
            }
        }
        let rows = tally.rows.max(1) as f64;
        eprintln!(
            "eval after {step} steps: code {:.1} bits per token (subcomponents on {:.1}, corner KL {:.4} nats per token)",
            tally.code(observations),
            tally.l0 / rows,
            tally.kl / rows
        );
        log.push(json!({"step": step, "code": tally.code(observations), "l0": tally.l0 / rows, "kl": tally.kl / rows, "description": tally.description / rows}));
        std::fs::write(out.join("evals.json"), json!({"observations": observations, "evals": log}).to_string()).map_err(|e| e.to_string())
    };
    if eval > 0 {
        evaluate(&mut replicas[0], 0, &mut log)?;
    }
    let (mut trained, mut busy) = (0usize, 0.0f64);
    for step in 0..steps {
        let started = Instant::now();
        let order: Vec<usize> = (0..batch).map(|i| (step * batch + i) % train).collect();
        let share = order.len().div_ceil(replicas.len());
        let tallies: Vec<Result<Tally, String>> = std::thread::scope(|scope| {
            let running: Vec<_> = replicas
                .iter_mut()
                .zip(order.chunks(share))
                .map(|(trainer, mine)| {
                    let sequences = &sequences;
                    scope.spawn(move || -> Result<Tally, String> {
                        let mut tally = Tally::default();
                        for part in mine.chunks(micro) {
                            tally.add(&trainer.train(&sequences(part))?);
                        }
                        Ok(tally)
                    })
                })
                .collect();
            running.into_iter().map(|h| joined(h)).collect()
        });
        let mut tally = Tally::default();
        for t in tallies {
            tally.add(&t?);
        }
        if replicas.len() > 1 {
            let mut total = replicas[0].gradients()?;
            for replica in &replicas[1..] {
                for (sum, g) in total.iter_mut().zip(replica.gradients()?) {
                    *sum += &g;
                }
            }
            for replica in &mut replicas {
                replica.set_gradients(&total)?;
            }
        }
        std::thread::scope(|scope| -> Result<(), String> {
            let running: Vec<_> = replicas.iter_mut().map(|trainer| scope.spawn(move || trainer.update())).collect();
            running.into_iter().try_for_each(|h| joined(h))
        })?;
        for d in &devices {
            d.synchronize().map_err(|e| e.to_string())?;
        }
        let seconds = started.elapsed().as_secs_f64();
        trained += batch;
        busy += seconds;
        let rows = tally.rows.max(1) as f64;
        eprintln!(
            "step {step}: code {:.1} bits per token (subcomponents on {:.1}, corner KL {:.4}); {:.1} ms per sequence, {:.2} sequences/s",
            tally.code(observations),
            tally.l0 / rows,
            tally.kl / rows,
            1e3 * seconds / batch as f64,
            trained as f64 / busy
        );
        if (step + 1) % sync == 0 || step + 1 == steps {
            // Every replica restores its map and prices its bits again; the first is evaluated.
            std::thread::scope(|scope| -> Result<(), String> {
                let describe = &describe;
                let running: Vec<_> = replicas[1..].iter_mut().map(|trainer| scope.spawn(move || trainer.sync(describe).map(|_| ()))).collect();
                running.into_iter().try_for_each(|h| joined(h))
            })?;
            evaluate(&mut replicas[0], step + 1, &mut log)?;
        }
    }
    eprintln!(
        "throughput: {:.2} sequences/s ({:.1} ms per sequence-step) at batch {batch} on {} × {}",
        trained as f64 / busy.max(1e-9),
        1e3 * busy / trained.max(1) as f64,
        devices.len(),
        device.name()
    );
    Ok(())
}
