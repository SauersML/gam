//! Unsupervised overcomplete dictionary on a modern LLM's residual stream (#2502).
//!
//! One arm per invocation. The arm is entirely described by `(K, b, k)`:
//!
//! * `--atoms K --block-size 2 --block-topk k` — the **overcomplete block**
//!   dictionary, `K > p`;
//! * `--atoms p --block-size 2 --block-topk k` — the **critically complete**
//!   control at the same rate;
//! * `--atoms K --block-size 1 --block-topk 2k` — the **flat TopK** arm, i.e. the
//!   standard sparse-dictionary decoder form at matched active scalars;
//! * `--atoms m --block-size m --block-topk 1` — **PCA(m)**: with one block the
//!   model is `x̂ = γ (x D₁ᵀ) D₁` over a single Stiefel frame, whose minimiser is
//!   the top-`m` principal subspace of the centred train data. Same engine, same
//!   loop, zero selection bits — the dense linear baseline.
//!
//! Everything numeric happens here, in Rust: centring, the alternating
//! block-TopK fit (`gam_sae::sparse_dict`), the held-out transform, the
//! reconstruction, the explained variance and the rate accounting. The Python in
//! `experiments/issue-2502/overcomplete_llm/` only moves bytes (activation
//! harvest through PyTorch, plots, token lookups).
//!
//! Usage:
//! ```text
//! issue_2502_overcomplete_llm --train train.npy --eval eval.npy --out DIR \
//!     --arm NAME --atoms K --block-size b --block-topk k [--epochs N] \
//!     [--minibatch M] [--block-tile T] [--tolerance t] [--gpu auto|required|off] \
//!     [--aux-k N] [--seed-policy rows|coordinate] \
//!     [--rows N] [--eval-rows N] [--dump-recon] [--dump-codes]
//! ```

use gam_sae::sparse_dict::{
    BlockSparseConfig, BlockSparseStreamState, block_sparse_dictionary_transform,
    coordinate_partition_frames, reconstruct_block_sparse_rows,
};
use memmap2::Mmap;
use ndarray::{Array1, Array2, ArrayView2, ArrayViewMut2};
use rayon::prelude::*;
use serde_json::json;
use std::fs::{self, File};
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};
use std::time::Instant;

#[path = "support/npy_header.rs"]
mod npy_header;
use npy_header::parse_npy_header;

#[path = "support/f16.rs"]
mod f16;
use f16::f16_to_f32;

struct Npy {
    mmap: Mmap,
    rows: usize,
    cols: usize,
    elem: usize,
    is_f4: bool,
    data_off: usize,
}

impl Npy {
    fn open(path: &Path) -> Result<Self, String> {
        let file = File::open(path).map_err(|e| format!("open {}: {e}", path.display()))?;
        // SAFETY: the activation bank is opened read-only and never written
        // through this mapping for the lifetime of the run.
        let mmap =
            unsafe { Mmap::map(&file).map_err(|e| format!("mmap {}: {e}", path.display()))? };
        let (rows, cols, elem, is_f4, data_off) = parse_npy_header(&mmap, path)?;
        let need = data_off
            .checked_add(
                rows.checked_mul(cols)
                    .and_then(|c| c.checked_mul(elem))
                    .ok_or("npy size overflow")?,
            )
            .ok_or("npy end overflow")?;
        if need > mmap.len() {
            return Err(format!("{} is truncated", path.display()));
        }
        Ok(Self {
            mmap,
            rows,
            cols,
            elem,
            is_f4,
            data_off,
        })
    }

    /// Copy `[row0, row0+take)` into `out`, subtracting `centre` if given.
    fn read_into(&self, row0: usize, mut out: ArrayViewMut2<'_, f32>, centre: Option<&[f32]>) {
        let take = out.nrows();
        let p = self.cols;
        for i in 0..take {
            let base = self.data_off + (row0 + i) * p * self.elem;
            let mut row = out.row_mut(i);
            for c in 0..p {
                let off = base + c * self.elem;
                let v = if self.is_f4 {
                    f32::from_le_bytes([
                        self.mmap[off],
                        self.mmap[off + 1],
                        self.mmap[off + 2],
                        self.mmap[off + 3],
                    ])
                } else {
                    f16_to_f32(u16::from_le_bytes([self.mmap[off], self.mmap[off + 1]]))
                };
                row[c] = match centre {
                    Some(m) => v - m[c],
                    None => v,
                };
            }
        }
    }
}

/// Least-squares reconstruction of one row on the support the router chose.
///
/// The block lane's own decode is `x̂ = γ Σ_{g∈S} x P_g`: a SUM of subspace
/// projections, which double-counts wherever two selected blocks overlap, with a
/// single shared `γ` absorbing the average of that double count. The
/// least-squares reconstruction on the SAME support is the orthogonal projection
/// of `x` onto the span of the selected atoms, which is what the dictionary can
/// actually represent. Computing both is the Rust-side statement of this
/// campaign's amortisation-gap thesis: solve the codes before comparing
/// dictionaries.
///
/// The span is built by modified Gram-Schmidt over the selected atom rows. A
/// direction whose residual norm falls below `sqrt(eps) x` its original norm is
/// linearly dependent on the ones already taken at f32 resolution and is
/// dropped, so a rank-deficient support is handled without a ridge.
fn joint_ls_row(
    row: &[f32],
    decoder: ArrayView2<'_, f32>,
    blocks: &[u32],
    gates: &[f32],
    block_size: usize,
    out: &mut [f32],
) {
    let p = row.len();
    assert_eq!(
        out.len(),
        p,
        "joint_ls_row writes one reconstruction entry per input feature"
    );
    let drop_ratio = f32::EPSILON.sqrt();
    let mut basis: Vec<Vec<f32>> = Vec::with_capacity(blocks.len() * block_size);
    for (j, &gsel) in blocks.iter().enumerate() {
        if gates[j] == 0.0 {
            continue;
        }
        for r in 0..block_size {
            let atom = decoder.row(gsel as usize * block_size + r);
            let mut q: Vec<f32> = atom.iter().copied().collect();
            let norm0 = q.iter().map(|v| v * v).sum::<f32>().sqrt();
            if !(norm0 > 0.0) {
                continue;
            }
            for e in basis.iter() {
                let d: f32 = q.iter().zip(e.iter()).map(|(a, b)| a * b).sum();
                for (a, b) in q.iter_mut().zip(e.iter()) {
                    *a -= d * b;
                }
            }
            let norm = q.iter().map(|v| v * v).sum::<f32>().sqrt();
            if !(norm > drop_ratio * norm0) {
                continue;
            }
            for v in q.iter_mut() {
                *v /= norm;
            }
            basis.push(q);
        }
    }
    for v in out.iter_mut() {
        *v = 0.0;
    }
    for e in basis.iter() {
        let c: f32 = row.iter().zip(e.iter()).map(|(a, b)| a * b).sum();
        for (o, b) in out.iter_mut().zip(e.iter()) {
            *o += c * b;
        }
    }
}

struct Args {
    train: PathBuf,
    eval: PathBuf,
    out: PathBuf,
    arm: String,
    atoms: usize,
    block_size: usize,
    block_topk: usize,
    epochs: usize,
    minibatch: usize,
    block_tile: usize,
    tolerance: f64,
    frame_ridge: f64,
    aux_k: usize,
    rows: usize,
    eval_rows: usize,
    dump_recon: bool,
    dump_codes: bool,
    seed_policy: SeedPolicy,
    load_decoder: PathBuf,
    load_gamma: f32,
    gpu: gam_gpu::GpuPolicy,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum SeedPolicy {
    /// `gam_sae::sparse_dict::coordinate_partition_frames`: `b` signed unit
    /// coordinate axes per block, no corpus pass.
    Coordinate,
    /// `b` distinct centred training rows per block, orthonormalised. The
    /// library's own `CoordinatePartition` docs note that at `K` far above the
    /// intrinsic rank most coordinate-seeded blocks are structurally spurious and
    /// depend on AuxK revival to become useful; seeding on data puts every block
    /// inside the cloud from the first epoch instead.
    Rows,
}

/// splitmix64, so the row seed is reproducible without an RNG dependency.
fn splitmix64_next(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// `G` blocks of `b` orthonormal rows, each block grown from `b` distinct centred
/// training rows by modified Gram-Schmidt. A block whose rows are degenerate
/// (a zero row, or rows that collapse) keeps the coordinate-partition frame for
/// that block, so the result is always a valid `St(b, P)` block dictionary.
fn row_sample_frames(
    train: &Npy,
    n_train: usize,
    centre: &[f32],
    g: usize,
    b: usize,
) -> Result<(Array2<f32>, usize), String> {
    let p = train.cols;
    let mut decoder = coordinate_partition_frames(g, b, p);
    let mut state = 0x2502_0000_0000_2502u64;
    let mut scratch = Array2::<f32>::zeros((1, p));
    let mut fallbacks = 0usize;
    for block in 0..g {
        let mut rows: Vec<Vec<f32>> = Vec::with_capacity(b);
        for _ in 0..b {
            let idx = (splitmix64_next(&mut state) % n_train as u64) as usize;
            train.read_into(idx, scratch.view_mut(), Some(centre));
            rows.push(scratch.row(0).to_vec());
        }
        let mut basis: Vec<Vec<f32>> = Vec::with_capacity(b);
        for mut row in rows {
            for q in basis.iter() {
                let dot: f32 = row.iter().zip(q.iter()).map(|(a, c)| a * c).sum();
                for (a, c) in row.iter_mut().zip(q.iter()) {
                    *a -= dot * c;
                }
            }
            let norm = row.iter().map(|v| v * v).sum::<f32>().sqrt();
            if !norm.is_finite() || norm == 0.0 {
                break;
            }
            for v in row.iter_mut() {
                *v /= norm;
            }
            basis.push(row);
        }
        if basis.len() < b {
            fallbacks += 1;
            continue;
        }
        for (r, q) in basis.iter().enumerate() {
            for (c, v) in q.iter().enumerate() {
                decoder[[block * b + r, c]] = *v;
            }
        }
    }
    Ok((decoder, fallbacks))
}

fn parse_args() -> Result<Args, String> {
    let raw: Vec<String> = std::env::args().collect();
    let mut a = Args {
        train: PathBuf::new(),
        eval: PathBuf::new(),
        out: PathBuf::new(),
        arm: "arm".to_string(),
        atoms: 0,
        block_size: 2,
        block_topk: 0,
        epochs: 60,
        minibatch: 8192,
        block_tile: 1024,
        tolerance: 1.0e-4,
        frame_ridge: 1.0e-9,
        aux_k: 0,
        rows: usize::MAX,
        eval_rows: usize::MAX,
        dump_recon: false,
        dump_codes: false,
        seed_policy: SeedPolicy::Rows,
        load_decoder: PathBuf::new(),
        load_gamma: f32::NAN,
        gpu: gam_gpu::GpuPolicy::Auto,
    };
    let mut i = 1usize;
    while i < raw.len() {
        let key = raw[i].as_str();
        if key == "--dump-recon" {
            a.dump_recon = true;
            i += 1;
            continue;
        }
        if key == "--dump-codes" {
            a.dump_codes = true;
            i += 1;
            continue;
        }
        let value = raw
            .get(i + 1)
            .ok_or_else(|| format!("missing value for {key}"))?;
        match key {
            "--train" => a.train = PathBuf::from(value),
            "--eval" => a.eval = PathBuf::from(value),
            "--out" => a.out = PathBuf::from(value),
            "--arm" => a.arm = value.clone(),
            "--atoms" => a.atoms = value.parse().map_err(|e| format!("--atoms: {e}"))?,
            "--block-size" => {
                a.block_size = value.parse().map_err(|e| format!("--block-size: {e}"))?
            }
            "--block-topk" => {
                a.block_topk = value.parse().map_err(|e| format!("--block-topk: {e}"))?
            }
            "--epochs" => a.epochs = value.parse().map_err(|e| format!("--epochs: {e}"))?,
            "--minibatch" => {
                a.minibatch = value.parse().map_err(|e| format!("--minibatch: {e}"))?
            }
            "--block-tile" => {
                a.block_tile = value.parse().map_err(|e| format!("--block-tile: {e}"))?
            }
            "--tolerance" => {
                a.tolerance = value.parse().map_err(|e| format!("--tolerance: {e}"))?
            }
            "--frame-ridge" => {
                a.frame_ridge = value.parse().map_err(|e| format!("--frame-ridge: {e}"))?
            }
            "--aux-k" => a.aux_k = value.parse().map_err(|e| format!("--aux-k: {e}"))?,
            "--rows" => a.rows = value.parse().map_err(|e| format!("--rows: {e}"))?,
            "--eval-rows" => {
                a.eval_rows = value.parse().map_err(|e| format!("--eval-rows: {e}"))?
            }
            "--seed-policy" => {
                a.seed_policy = match value.as_str() {
                    "coordinate" => SeedPolicy::Coordinate,
                    "rows" => SeedPolicy::Rows,
                    other => {
                        return Err(format!(
                            "--seed-policy must be coordinate|rows, got {other}"
                        ));
                    }
                }
            }
            "--load-decoder" => a.load_decoder = PathBuf::from(value),
            "--load-gamma" => {
                a.load_gamma = value.parse().map_err(|e| format!("--load-gamma: {e}"))?
            }
            "--gpu" => {
                a.gpu = gam_gpu::GpuPolicy::parse(value)
                    .ok_or_else(|| format!("--gpu must be required|auto|off, got {value}"))?;
            }
            other => return Err(format!("unknown argument {other}")),
        }
        i += 2;
    }
    if a.train.as_os_str().is_empty() || a.eval.as_os_str().is_empty() {
        return Err("--train and --eval are required".to_string());
    }
    if a.out.as_os_str().is_empty() {
        return Err("--out is required".to_string());
    }
    if a.atoms == 0 || a.block_size == 0 || a.block_topk == 0 {
        return Err("--atoms, --block-size and --block-topk must be positive".to_string());
    }
    if a.atoms % a.block_size != 0 {
        return Err("--atoms must be divisible by --block-size".to_string());
    }
    Ok(a)
}

fn write_f32(path: &Path, values: impl Iterator<Item = f32>) -> Result<usize, String> {
    let file = File::create(path).map_err(|e| format!("create {}: {e}", path.display()))?;
    let mut out = BufWriter::new(file);
    let mut n = 0usize;
    for v in values {
        out.write_all(&v.to_le_bytes())
            .map_err(|e| format!("write {}: {e}", path.display()))?;
        n += 1;
    }
    out.flush()
        .map_err(|e| format!("flush {}: {e}", path.display()))?;
    Ok(n * 4)
}

fn main() -> Result<(), String> {
    let args = parse_args()?;
    fs::create_dir_all(&args.out).map_err(|e| format!("mkdir {}: {e}", args.out.display()))?;
    let started = Instant::now();

    gam_gpu::configure_global_policy(args.gpu);
    let train = Npy::open(&args.train)?;
    let eval = Npy::open(&args.eval)?;
    if train.cols != eval.cols {
        return Err(format!(
            "train has p={} but eval has p={}",
            train.cols, eval.cols
        ));
    }
    let p = train.cols;
    let n_train = args.rows.min(train.rows);
    let n_eval = args.eval_rows.min(eval.rows);
    let g = args.atoms / args.block_size;
    let k = args.block_topk.min(g);

    let route_plan = gam_gpu::DictionaryScoreRoutePlan::default_for_shape(
        args.minibatch.min(n_train),
        args.atoms,
        p,
    );
    println!(
        "[a5] arm={} engine=gam_sae::sparse_dict::BlockSparseStreamState n_train={n_train} \
         n_eval={n_eval} p={p} K={} G={g} b={} topk={k} overcomplete_ratio={:.4} gpu={:?} \
         device_admitted={} break_even={}",
        args.arm,
        args.atoms,
        args.block_size,
        args.atoms as f64 / p as f64,
        args.gpu,
        route_plan.device_admitted,
        route_plan.device_min_score_elems,
    );

    // Tier-0: the train-split mean, held fixed for eval. Streaming so the whole
    // bank never has to be resident.
    let mean_start = Instant::now();
    let mut mean = vec![0.0f64; p];
    {
        let chunk = args.minibatch.min(n_train).max(1);
        let mut buf = Array2::<f32>::zeros((chunk, p));
        let mut row0 = 0usize;
        while row0 < n_train {
            let take = (n_train - row0).min(chunk);
            train.read_into(row0, buf.slice_mut(ndarray::s![0..take, ..]), None);
            for i in 0..take {
                let row = buf.row(i);
                for c in 0..p {
                    mean[c] += row[c] as f64;
                }
            }
            row0 += take;
        }
        for m in mean.iter_mut() {
            *m /= n_train as f64;
        }
    }
    let mean_f32: Vec<f32> = mean.iter().map(|v| *v as f32).collect();
    println!(
        "[a5] tier0 mean over {n_train} rows in {:.1}s",
        mean_start.elapsed().as_secs_f64()
    );

    let mut cfg = BlockSparseConfig::new(g, args.block_size);
    cfg.block_topk = k;
    cfg.max_epochs = args.epochs;
    cfg.minibatch = args.minibatch;
    cfg.block_tile = args.block_tile;
    cfg.tolerance = args.tolerance;
    cfg.frame_ridge = args.frame_ridge;
    cfg.aux_k = args.aux_k;

    let reload = !args.load_decoder.as_os_str().is_empty();
    let (seed, seed_fallbacks) = match args.seed_policy {
        SeedPolicy::Coordinate => (coordinate_partition_frames(g, args.block_size, p), 0),
        SeedPolicy::Rows => row_sample_frames(&train, n_train, &mean_f32, g, args.block_size)?,
    };
    println!(
        "[a5] seed_policy={:?} coordinate_fallback_blocks={seed_fallbacks}/{g}",
        args.seed_policy
    );
    let mut state = BlockSparseStreamState::new_with_decoder(seed, &cfg)?;

    let train_start = Instant::now();
    let mut shard = Array2::<f32>::zeros((args.minibatch.min(n_train).max(1), p));
    let mut epoch_rows: Vec<serde_json::Value> = Vec::new();
    let mut converged = false;
    let mut last_ev = f64::NAN;
    for epoch in 0..(if reload { 0 } else { args.epochs }) {
        let mut row0 = 0usize;
        while row0 < n_train {
            let take = (n_train - row0).min(shard.nrows());
            train.read_into(
                row0,
                shard.slice_mut(ndarray::s![0..take, ..]),
                Some(&mean_f32),
            );
            state.partial_fit(shard.slice(ndarray::s![0..take, ..]))?;
            row0 += take;
        }
        let stats = state.end_epoch()?;
        last_ev = stats.explained_variance;
        println!(
            "[a5] arm={} epoch {}/{} train_ev={:.6} gamma={:.6} dead={} accepted_births={} \
             elapsed={:.1}s",
            args.arm,
            epoch + 1,
            args.epochs,
            stats.explained_variance,
            stats.gamma,
            stats.dead,
            stats.accepted_births,
            started.elapsed().as_secs_f64(),
        );
        epoch_rows.push(json!({
            "epoch": epoch + 1,
            "train_explained_variance": stats.explained_variance,
            "gamma": stats.gamma,
            "dead": stats.dead,
            "accepted_births": stats.accepted_births,
            "converged": stats.converged,
            "seconds": started.elapsed().as_secs_f64(),
        }));
        if stats.converged {
            converged = true;
            break;
        }
    }
    let train_seconds = train_start.elapsed().as_secs_f64();
    // `--load-decoder` re-scores an ALREADY FITTED dictionary: the same held-out
    // pass, no training. The decoder file is the `K*p*4` byte dump this binary
    // wrote for that arm, so a re-score cannot silently read a different fit.
    let (decoder, gamma) = if reload {
        let bytes = fs::read(&args.load_decoder)
            .map_err(|e| format!("read {}: {e}", args.load_decoder.display()))?;
        if bytes.len() != args.atoms * p * 4 {
            return Err(format!(
                "{} is {} bytes, expected K*p*4 = {}",
                args.load_decoder.display(),
                bytes.len(),
                args.atoms * p * 4
            ));
        }
        if !args.load_gamma.is_finite() {
            return Err("--load-decoder requires --load-gamma".to_string());
        }
        let mut d = Array2::<f32>::zeros((args.atoms, p));
        for (i, chunk) in bytes.chunks_exact(4).enumerate() {
            d[[i / p, i % p]] = f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
        }
        println!(
            "[a5] reloaded decoder {} ({} bytes) gamma={}",
            args.load_decoder.display(),
            bytes.len(),
            args.load_gamma
        );
        (d, args.load_gamma)
    } else {
        (state.decoder().to_owned(), state.gamma())
    };

    // Held-out pass: transform, reconstruct, and accumulate the sums the FVU and
    // the usage census are made of. Chunked so the N x K score matrix and the
    // N x p reconstruction never both have to be resident.
    let eval_start = Instant::now();
    let mut eval_mean = vec![0.0f64; p];
    {
        let chunk = args.minibatch.min(n_eval).max(1);
        let mut buf = Array2::<f32>::zeros((chunk, p));
        let mut row0 = 0usize;
        while row0 < n_eval {
            let take = (n_eval - row0).min(chunk);
            eval.read_into(row0, buf.slice_mut(ndarray::s![0..take, ..]), None);
            for i in 0..take {
                let row = buf.row(i);
                for c in 0..p {
                    eval_mean[c] += row[c] as f64;
                }
            }
            row0 += take;
        }
        for m in eval_mean.iter_mut() {
            *m /= n_eval as f64;
        }
    }

    let mut rss = 0.0f64;
    let mut rss_ls = 0.0f64;
    let mut tss_about_eval_mean = 0.0f64;
    let mut tss_about_train_mean = 0.0f64;
    let mut block_hits = vec![0u64; g];
    let mut atom_abs_max = vec![0.0f32; args.atoms];
    let mut recon_file = if args.dump_recon {
        Some(BufWriter::new(
            File::create(args.out.join("eval_recon.f32"))
                .map_err(|e| format!("create eval_recon.f32: {e}"))?,
        ))
    } else {
        None
    };
    let mut recon_ls_file = if args.dump_recon {
        Some(BufWriter::new(
            File::create(args.out.join("eval_recon_joint_ls.f32"))
                .map_err(|e| format!("create eval_recon_joint_ls.f32: {e}"))?,
        ))
    } else {
        None
    };
    let mut blocks_file = if args.dump_codes {
        Some(BufWriter::new(
            File::create(args.out.join("eval_blocks.u32"))
                .map_err(|e| format!("create eval_blocks.u32: {e}"))?,
        ))
    } else {
        None
    };
    let mut codes_file = if args.dump_codes {
        Some(BufWriter::new(
            File::create(args.out.join("eval_codes.f32"))
                .map_err(|e| format!("create eval_codes.f32: {e}"))?,
        ))
    } else {
        None
    };

    let chunk = args.minibatch.min(n_eval).max(1);
    let mut buf = Array2::<f32>::zeros((chunk, p));
    let mut row0 = 0usize;
    while row0 < n_eval {
        let take = (n_eval - row0).min(chunk);
        eval.read_into(
            row0,
            buf.slice_mut(ndarray::s![0..take, ..]),
            Some(&mean_f32),
        );
        let view = buf.slice(ndarray::s![0..take, ..]);
        let (blocks, _gates, codes) = block_sparse_dictionary_transform(
            view,
            decoder.view(),
            gamma,
            args.block_size,
            k,
            args.block_tile,
        )?;
        let recon = reconstruct_block_sparse_rows(
            decoder.view(),
            blocks.view(),
            codes.view(),
            args.block_size,
        )?;
        // Same support, least-squares amplitudes: the orthogonal projection onto
        // the span of the selected atoms. Row-independent, so it parallelises.
        let mut recon_ls = Array2::<f32>::zeros((take, p));
        recon_ls
            .axis_iter_mut(ndarray::Axis(0))
            .into_par_iter()
            .enumerate()
            .for_each(|(i, mut out_row)| {
                let row: Vec<f32> = view.row(i).to_vec();
                let sel: Vec<u32> = (0..k).map(|j| blocks[[i, j]]).collect();
                let gate: Vec<f32> = (0..k)
                    .map(|j| {
                        (0..args.block_size)
                            .map(|r| codes[[i, j, r]] * codes[[i, j, r]])
                            .sum::<f32>()
                    })
                    .collect();
                let slice = out_row.as_slice_mut().expect("contiguous row");
                joint_ls_row(&row, decoder.view(), &sel, &gate, args.block_size, slice);
            });
        for i in 0..take {
            for c in 0..p {
                let target = view[[i, c]] as f64;
                let fitted = recon[[i, c]] as f64;
                let d = target - fitted;
                rss += d * d;
                let d_ls = target - recon_ls[[i, c]] as f64;
                rss_ls += d_ls * d_ls;
                // Both totals are about a CONSTANT predictor; the eval-mean one is
                // the standard FVU denominator, the train-mean one is what an
                // honest out-of-sample Tier-0 would actually have achieved.
                let about_eval = target + mean[c] - eval_mean[c];
                tss_about_eval_mean += about_eval * about_eval;
                tss_about_train_mean += target * target;
            }
            for j in 0..k {
                let gsel = blocks[[i, j]] as usize;
                block_hits[gsel] += 1;
                for r in 0..args.block_size {
                    let v = codes[[i, j, r]].abs();
                    let atom = gsel * args.block_size + r;
                    if v > atom_abs_max[atom] {
                        atom_abs_max[atom] = v;
                    }
                }
            }
        }
        if let Some(f) = recon_file.as_mut() {
            for v in recon.iter() {
                f.write_all(&v.to_le_bytes())
                    .map_err(|e| format!("write recon: {e}"))?;
            }
        }
        if let Some(f) = recon_ls_file.as_mut() {
            for v in recon_ls.iter() {
                f.write_all(&v.to_le_bytes())
                    .map_err(|e| format!("write recon_ls: {e}"))?;
            }
        }
        if let Some(f) = blocks_file.as_mut() {
            for v in blocks.iter() {
                f.write_all(&v.to_le_bytes())
                    .map_err(|e| format!("write blocks: {e}"))?;
            }
        }
        if let Some(f) = codes_file.as_mut() {
            for v in codes.iter() {
                f.write_all(&v.to_le_bytes())
                    .map_err(|e| format!("write codes: {e}"))?;
            }
        }
        row0 += take;
    }
    if let Some(mut f) = recon_file {
        f.flush().map_err(|e| format!("flush recon: {e}"))?;
    }
    if let Some(mut f) = recon_ls_file {
        f.flush().map_err(|e| format!("flush recon_ls: {e}"))?;
    }
    if let Some(mut f) = blocks_file {
        f.flush().map_err(|e| format!("flush blocks: {e}"))?;
    }
    if let Some(mut f) = codes_file {
        f.flush().map_err(|e| format!("flush codes: {e}"))?;
    }
    let eval_seconds = eval_start.elapsed().as_secs_f64();

    let blocks_used = block_hits.iter().filter(|c| **c > 0).count();
    let atoms_used = atom_abs_max.iter().filter(|v| **v > 0.0).count();
    let ev = 1.0 - rss / tss_about_eval_mean;
    let fvu = rss / tss_about_eval_mean;
    let fvu_ls = rss_ls / tss_about_eval_mean;

    // Fixed-width rate accounting, the #2283 currency: every firing pays a
    // selection index over G blocks plus b amplitude scalars. Reported as counts
    // so the amplitude word width stays the reader's choice.
    let selection_bits_per_token = k as f64 * (g as f64).log2();
    let scalars_per_token = k * args.block_size;

    // Provenance: the decoder bytes the analysis reads must be exactly K*p*4.
    let decoder_bytes = write_f32(&args.out.join("decoder.f32"), decoder.iter().copied())?;
    let mean_bytes = write_f32(&args.out.join("train_mean.f32"), mean_f32.iter().copied())?;
    let util: Array1<f64> = Array1::from(
        block_hits
            .iter()
            .map(|c| *c as f64 / n_eval as f64)
            .collect::<Vec<f64>>(),
    );
    write_f32(
        &args.out.join("eval_block_utilization.f32"),
        util.iter().map(|v| *v as f32),
    )?;

    let report = json!({
        "issue": 2502,
        "arm": args.arm,
        "engine": "gam_sae::sparse_dict::BlockSparseStreamState",
        "crate_version": env!("CARGO_PKG_VERSION"),
        "inputs": {
            "train": args.train.display().to_string(),
            "eval": args.eval.display().to_string(),
            "n_train": n_train,
            "n_eval": n_eval,
            "p": p,
        },
        "dictionary": {
            "K_atoms": args.atoms,
            "G_blocks": g,
            "block_size": args.block_size,
            "block_topk": k,
            "overcomplete_ratio_K_over_p": args.atoms as f64 / p as f64,
            "gamma": gamma,
            "decoder_bytes": decoder_bytes,
            "decoder_bytes_expected": args.atoms * p * 4,
            "train_mean_bytes": mean_bytes,
        },
        "fit": {
            "epochs_run": state.epochs_run(),
            "converged": converged,
            "train_explained_variance": last_ev,
            "train_seconds": train_seconds,
            "gpu_policy": format!("{:?}", args.gpu),
            "seed_policy": format!("{:?}", args.seed_policy),
            "seed_coordinate_fallback_blocks": seed_fallbacks,
            "aux_k": args.aux_k,
            "device_admitted": route_plan.device_admitted,
        },
        "heldout": {
            "rows_scored": n_eval,
            "rss": rss,
            "tss_about_eval_mean": tss_about_eval_mean,
            "tss_about_train_mean": tss_about_train_mean,
            "explained_variance": ev,
            "fvu": fvu,
            "fvu_joint_ls": fvu_ls,
            "explained_variance_joint_ls": 1.0 - fvu_ls,
            "rss_joint_ls": rss_ls,
            "fvu_about_train_mean": rss / tss_about_train_mean,
            "seconds": eval_seconds,
        },
        "overcompleteness": {
            "K_atoms": args.atoms,
            "ambient_p": p,
            "blocks_used_on_heldout": blocks_used,
            "G_blocks": g,
            "atoms_used_on_heldout": atoms_used,
            "atoms_used_exceeds_ambient_p": atoms_used > p,
            "note": "atoms_used > p implies the used sub-dictionary is linearly dependent",
        },
        "rate": {
            "active_scalars_per_token": scalars_per_token,
            "selection_bits_per_token": selection_bits_per_token,
            "firings_per_token": k,
        },
        "epochs": epoch_rows,
        "wall_seconds": started.elapsed().as_secs_f64(),
    });
    let numbers = args.out.join("numbers.json");
    fs::write(
        &numbers,
        format!(
            "{}\n",
            serde_json::to_string_pretty(&report).map_err(|e| e.to_string())?
        ),
    )
    .map_err(|e| format!("write {}: {e}", numbers.display()))?;
    println!(
        "[a5] arm={} heldout_ev={:.6} fvu={:.6} fvu_joint_ls={:.6} blocks_used={}/{} \
         atoms_used={}/{} (p={}) scalars/token={} sel_bits/token={:.2} wrote {}",
        args.arm,
        ev,
        fvu,
        fvu_ls,
        blocks_used,
        g,
        atoms_used,
        args.atoms,
        p,
        scalars_per_token,
        selection_bits_per_token,
        numbers.display()
    );
    Ok(())
}
