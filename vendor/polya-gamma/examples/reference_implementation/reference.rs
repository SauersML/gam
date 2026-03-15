use polya_gamma::PolyaGamma;
use rand::SeedableRng;
use rand::rngs::StdRng;
use std::fmt::Write as WriteTrait;
use std::io::Write;
use std::path::PathBuf;
use std::time::Instant;

/// Draws `N` Polya-Gamma samples for each of the command line arguments
/// (interpreted as `z` values) and writes them to a single CSV file
/// called `data/pg_samples.csv`. The first row is a header row with
/// column names like "z=0.5", the second row is a header row with
/// column names like "z=1.0", etc. The remaining rows are the actual
/// samples.
///
/// Usage: `cargo run --release --example reference_implementation -- [z1 z2 ...] [--seed <u64>]`
///
/// If no command line arguments are given, defaults to drawing samples
/// for `z` values `[0.5, 1.0, 2.0, 3.2, 5.0]`.
///
/// If `--seed <u64>` is given, seeds the PRNG with that value (otherwise
/// uses a fixed default).
fn main() {
    let mut b = 1.0;
    let mut zs = Vec::new();
    let mut had_input = false;
    let mut seed: u64 = 0;
    let mut args = std::env::args().skip(1).peekable();
    while let Some(arg) = args.next() {
        if arg == "--seed" {
            if let Some(val) = args.next() {
                if let Ok(parsed) = val.parse::<u64>() {
                    seed = parsed;
                } else {
                    eprintln!("Invalid value for --seed: {} (using default 0)", val);
                }
            } else {
                eprintln!("--seed given but no value (using default 0)");
            }
        } else if arg.eq_ignore_ascii_case("--b") {
            if let Some(val) = args.next() {
                if let Ok(parsed) = val.parse::<f64>() {
                    b = parsed;
                } else {
                    eprintln!("Invalid value for --b: {} (using default 1.0)", val);
                }
            } else {
                eprintln!("--b given but no value (using default 1.0)");
            }
        } else {
            let l = arg.trim();
            if l.is_empty() {
                continue;
            }
            had_input = true;
            if let Ok(z) = l.parse::<f64>() {
                zs.push(z);
            } else {
                eprintln!("Could not parse '{}' as f64; skipping.", l);
            }
        }
    }

    if !had_input || zs.is_empty() {
        zs = vec![0.5, 1.0, 2.0, 3.2, 5.0];
        eprintln!("Using default z values: {:.1?}", zs);
    }

    const N: usize = 1_000_000;

    // Time only the sample generation
    let sample_start = Instant::now();
    let mut all_samples: Vec<Vec<f64>> = Vec::with_capacity(zs.len());
    let mut rng = StdRng::seed_from_u64(seed);
    let pg = PolyaGamma::new(b);
    for &z in &zs {
        let out = pg.draw_vec_par_deterministic(&mut rng, &vec![z; N]);
        all_samples.push(out);
    }
    let sample_dur = sample_start.elapsed();
    println!(
        "[Rust] Cumulative sample generation time: {:.3} seconds",
        sample_dur.as_secs_f64()
    );

    // Write everything into a single CSV file with columns = z values

    let mut out_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    out_path.push("examples/reference_implementation/data/pg_samples.csv");
    let mut f = std::fs::File::create(&out_path).unwrap();
    // Header row

    let mut header = String::with_capacity(zs.len() * 8);
    for (i, &z) in zs.iter().enumerate() {
        if i > 0 {
            header.push(',');
        }

        let _ = write!(header, "z={:.1}", z);
    }
    let _ = writeln!(f, "{header}");

    // Data rows
    let n_cols = all_samples.len();
    let mut row_buf = String::with_capacity(n_cols * 8);
    for row in 0..N {
        row_buf.clear();
        for (col, sample) in all_samples.iter().enumerate() {
            if col > 0 {
                row_buf.push(',');
            }

            let _ = write!(row_buf, "{}", sample[row]);
        }
        let _ = writeln!(f, "{row_buf}");
    }
}
