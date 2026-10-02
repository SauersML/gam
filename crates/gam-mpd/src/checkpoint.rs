//! Checkpoints of a streaming masked fit (#2951), so a run on an interruptible host resumes
//! where it stopped and continues exactly as an uninterrupted run would.
//!
//! A checkpoint is everything a fit carries from one training sequence to the next: the
//! libraries (`V`, `U`, `μ` per site), the [`Context`] counts, the [`Running`] preconditioners, each
//! training sequence's current sets, and the driver's own state (indices, points, scalars) as
//! JSON. Every random draw of the fit is seeded from those indices, so no generator state is kept.
//!
//! On disk, in one directory: `manifest.json` names generation `N` and lays out its arrays, all
//! float64 values in `g{N}.f64.npy` and all set indices in `g{N}.u32.npy` (one-dimensional, little
//! endian, in the manifest's order, so numpy reads them as they are). A save writes generation
//! `N + 1`'s arrays in full and syncs them, then replaces the manifest by renaming a synced
//! temporary file over it, which commits the generation atomically; only then are older
//! generations removed. A run killed at any point leaves the last committed generation whole.
//! Values are stored bit for bit (raw float64, and JSON numbers that round-trip exactly).

use super::masked::{Context, Library, Running};
use ndarray::{Array1, Array2};
use serde_json::{Value, json};
use std::fs::File;
use std::io::{BufReader, BufWriter, Read, Write};
use std::path::{Path, PathBuf};

/// One sequence's sets, sparse: per site, CSR over its positions (`indptr`, `rows + 1` long) of the
/// pieces on (`indices`, ascending within a position).
pub type SparseSets = Vec<(Vec<u32>, Vec<u32>)>;

/// What a save writes (borrowed from the fit).
pub struct Saved<'a> {
    pub driver: &'a Value,
    pub libraries: &'a [Library],
    pub context: &'a Context,
    pub running: &'a Running,
    /// Per training sequence, its current sets when it has any.
    pub sets: Vec<Option<&'a SparseSets>>,
}

/// What a load returns (owned).
pub struct Loaded {
    pub driver: Value,
    pub libraries: Vec<Library>,
    pub context: Context,
    pub running: Running,
    pub sets: Vec<Option<SparseSets>>,
}

const MANIFEST: &str = "manifest.json";

fn io(path: &Path) -> impl Fn(std::io::Error) -> String + '_ {
    move |e| format!("{}: {e}", path.display())
}

/// The header of a one-dimensional little-endian `.npy` array of `count` values of type `descr`.
fn npy_header(descr: &str, count: usize) -> Vec<u8> {
    let mut header = format!("{{'descr': '{descr}', 'fortran_order': False, 'shape': ({count},), }}");
    while (10 + header.len() + 1) % 64 != 0 {
        header.push(' ');
    }
    header.push('\n');
    let mut bytes = b"\x93NUMPY\x01\x00".to_vec();
    bytes.extend_from_slice(&(header.len() as u16).to_le_bytes());
    bytes.extend_from_slice(header.as_bytes());
    bytes
}

/// Opens `path` and checks that its header is [`npy_header`]`(descr, count)`.
fn open_npy(path: &Path, descr: &str, count: usize) -> Result<BufReader<File>, String> {
    let mut reader = BufReader::new(File::open(path).map_err(io(path))?);
    let expected = npy_header(descr, count);
    let mut header = vec![0u8; expected.len()];
    reader.read_exact(&mut header).map_err(io(path))?;
    if header != expected {
        return Err(format!("{}: not a {descr} array of {count} values", path.display()));
    }
    Ok(reader)
}

/// Writes `count` values, each `width` bytes, from `chunks` (whose total must be `count`), then
/// syncs the file.
fn write_npy<'a>(path: &Path, descr: &str, count: usize, width: usize, chunks: impl Iterator<Item = Box<dyn Iterator<Item = [u8; 8]> + 'a>>) -> Result<(), String> {
    let file = File::create(path).map_err(io(path))?;
    let mut writer = BufWriter::new(file);
    writer.write_all(&npy_header(descr, count)).map_err(io(path))?;
    let mut written = 0usize;
    for chunk in chunks {
        for bytes in chunk {
            writer.write_all(&bytes[..width]).map_err(io(path))?;
            written += 1;
        }
    }
    if written != count {
        return Err(format!("{}: wrote {written} values of {count}", path.display()));
    }
    let file = writer.into_inner().map_err(|e| format!("{}: {e}", path.display()))?;
    file.sync_all().map_err(io(path))
}

fn f64_bytes<'a>(values: impl Iterator<Item = &'a f64> + 'a) -> Box<dyn Iterator<Item = [u8; 8]> + 'a> {
    Box::new(values.map(|v| v.to_le_bytes()))
}

fn u32_bytes<'a>(values: &'a [u32]) -> Box<dyn Iterator<Item = [u8; 8]> + 'a> {
    Box::new(values.iter().map(|v| {
        let b = v.to_le_bytes();
        [b[0], b[1], b[2], b[3], 0, 0, 0, 0]
    }))
}

fn read_f64s(reader: &mut impl Read, count: usize, path: &Path) -> Result<Vec<f64>, String> {
    let mut bytes = vec![0u8; count * 8];
    reader.read_exact(&mut bytes).map_err(io(path))?;
    Ok(bytes.chunks_exact(8).map(|c| f64::from_le_bytes([c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]])).collect())
}

fn read_u32s(reader: &mut impl Read, count: usize, path: &Path) -> Result<Vec<u32>, String> {
    let mut bytes = vec![0u8; count * 4];
    reader.read_exact(&mut bytes).map_err(io(path))?;
    Ok(bytes.chunks_exact(4).map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect())
}

fn matrix(reader: &mut impl Read, (rows, cols): (usize, usize), path: &Path) -> Result<Array2<f64>, String> {
    Array2::from_shape_vec((rows, cols), read_f64s(reader, rows * cols, path)?).map_err(|e| e.to_string())
}

fn dims(m: &Array2<f64>) -> Value {
    json!([m.nrows(), m.ncols()])
}

fn usize_of(value: &Value) -> Result<usize, String> {
    value.as_u64().map(|v| v as usize).ok_or_else(|| format!("checkpoint manifest: {value} is not a count"))
}

fn pair_of(value: &Value) -> Result<(usize, usize), String> {
    match value.as_array().map(Vec::as_slice) {
        Some([a, b]) => Ok((usize_of(a)?, usize_of(b)?)),
        _ => Err(format!("checkpoint manifest: {value} is not a pair of counts")),
    }
}

fn list<'a>(manifest: &'a Value, key: &str) -> Result<&'a Vec<Value>, String> {
    manifest[key].as_array().ok_or_else(|| format!("checkpoint manifest: no {key} list"))
}

fn generation_file(dir: &Path, generation: u64, kind: &str) -> PathBuf {
    dir.join(format!("g{generation}.{kind}.npy"))
}

/// The generation the manifest in `dir` commits, if any.
fn committed(dir: &Path) -> Result<Option<(u64, Value)>, String> {
    let path = dir.join(MANIFEST);
    let text = match std::fs::read_to_string(&path) {
        Ok(text) => text,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(e) => return Err(format!("{}: {e}", path.display())),
    };
    let manifest: Value = serde_json::from_str(&text).map_err(|e| format!("{}: {e}", path.display()))?;
    let generation = manifest["generation"].as_u64().ok_or_else(|| format!("{}: no generation", path.display()))?;
    Ok(Some((generation, manifest)))
}

/// Saves `saved` to `dir` as its next generation (module note), creating `dir` when needed.
pub fn save(dir: &Path, saved: &Saved<'_>) -> Result<(), String> {
    std::fs::create_dir_all(dir).map_err(io(dir))?;
    let generation = committed(dir)?.map_or(0, |(g, _)| g + 1);
    let libraries: Vec<Value> = saved.libraries.iter().map(|l| json!({"v": dims(&l.v), "u": dims(&l.u), "mean": l.mean.len()})).collect();
    let context: Vec<usize> = saved.context.new.iter().map(Array1::len).collect();
    if saved.context.stayed.iter().map(Array1::len).ne(context.iter().copied()) || saved.context.was_on.iter().map(Array1::len).ne(context.iter().copied()) {
        return Err("checkpoint: the context's counts disagree in their pieces".to_string());
    }
    let sets: Vec<Value> = saved
        .sets
        .iter()
        .map(|s| s.map_or(Value::Null, |sites| json!(sites.iter().map(|(indptr, indices)| json!([indptr.len(), indices.len()])).collect::<Vec<_>>())))
        .collect();
    let f64_count = saved.libraries.iter().map(|l| l.v.len() + l.u.len() + l.mean.len()).sum::<usize>()
        + 3 * context.iter().sum::<usize>()
        + saved.running.covariances.iter().chain(&saved.running.fishers).map(Array2::len).sum::<usize>();
    let u32_count = saved.sets.iter().flatten().flat_map(|sites| sites.iter()).map(|(p, i)| p.len() + i.len()).sum::<usize>();
    let manifest = json!({
        "generation": generation,
        "driver": saved.driver,
        "libraries": libraries,
        "context": context,
        "running": {
            "rows": saved.running.rows,
            "covariances": saved.running.covariances.iter().map(dims).collect::<Vec<_>>(),
            "fishers": saved.running.fishers.iter().map(dims).collect::<Vec<_>>(),
        },
        "sets": sets,
        "f64": f64_count,
        "u32": u32_count,
    });
    let reals = saved
        .libraries
        .iter()
        .flat_map(|l| [f64_bytes(l.v.iter()), f64_bytes(l.u.iter()), f64_bytes(l.mean.iter())])
        .chain(saved.context.stayed.iter().chain(&saved.context.was_on).chain(&saved.context.new).map(|c| f64_bytes(c.iter())))
        .chain(saved.running.covariances.iter().chain(&saved.running.fishers).map(|m| f64_bytes(m.iter())));
    write_npy(&generation_file(dir, generation, "f64"), "<f8", f64_count, 8, reals)?;
    let indices = saved.sets.iter().flatten().flat_map(|sites| sites.iter()).flat_map(|(p, i)| [u32_bytes(p), u32_bytes(i)]);
    write_npy(&generation_file(dir, generation, "u32"), "<u4", u32_count, 4, indices)?;
    let temporary = dir.join(format!("{MANIFEST}.tmp"));
    {
        let mut file = File::create(&temporary).map_err(io(&temporary))?;
        file.write_all(manifest.to_string().as_bytes()).map_err(io(&temporary))?;
        file.sync_all().map_err(io(&temporary))?;
    }
    std::fs::rename(&temporary, dir.join(MANIFEST)).map_err(io(dir))?;
    File::open(dir).and_then(|d| d.sync_all()).map_err(io(dir))?;
    // The new generation is committed: older ones (and any partial one) go.
    let current = [generation_file(dir, generation, "f64"), generation_file(dir, generation, "u32")];
    for entry in std::fs::read_dir(dir).map_err(io(dir))? {
        let path = entry.map_err(io(dir))?.path();
        let name = path.file_name().and_then(|n| n.to_str()).unwrap_or("");
        if name.starts_with('g') && name.ends_with(".npy") && !current.contains(&path) {
            std::fs::remove_file(&path).map_err(io(&path))?;
        }
    }
    Ok(())
}

/// The checkpoint committed in `dir`, or `None` when there is none.
pub fn load(dir: &Path) -> Result<Option<Loaded>, String> {
    let Some((generation, manifest)) = committed(dir)? else { return Ok(None) };
    let f64_path = generation_file(dir, generation, "f64");
    let mut reals = open_npy(&f64_path, "<f8", usize_of(&manifest["f64"])?)?;
    let mut libraries = Vec::new();
    for entry in list(&manifest, "libraries")? {
        let v = matrix(&mut reals, pair_of(&entry["v"])?, &f64_path)?;
        let u = matrix(&mut reals, pair_of(&entry["u"])?, &f64_path)?;
        let mean = Array1::from(read_f64s(&mut reals, usize_of(&entry["mean"])?, &f64_path)?);
        libraries.push(Library { v, u, mean });
    }
    let pieces: Vec<usize> = list(&manifest, "context")?.iter().map(usize_of).collect::<Result<_, _>>()?;
    let mut counts = || -> Result<Vec<Array1<f64>>, String> { pieces.iter().map(|p| Ok(Array1::from(read_f64s(&mut reals, *p, &f64_path)?))).collect() };
    let (stayed, was_on, new) = (counts()?, counts()?, counts()?);
    let running_manifest = &manifest["running"];
    let rows = running_manifest["rows"].as_f64().ok_or("checkpoint manifest: no running rows")?;
    let mut matrices = |key: &str| -> Result<Vec<Array2<f64>>, String> {
        list(running_manifest, key)?.iter().map(|d| matrix(&mut reals, pair_of(d)?, &f64_path)).collect()
    };
    let (covariances, fishers) = (matrices("covariances")?, matrices("fishers")?);
    let u32_path = generation_file(dir, generation, "u32");
    let mut indices = open_npy(&u32_path, "<u4", usize_of(&manifest["u32"])?)?;
    let mut sets = Vec::new();
    for entry in list(&manifest, "sets")? {
        if entry.is_null() {
            sets.push(None);
            continue;
        }
        let sites = entry.as_array().ok_or("checkpoint manifest: a sequence's sets are not a list")?;
        let mut sequence = Vec::new();
        for site in sites {
            let (p, i) = pair_of(site)?;
            sequence.push((read_u32s(&mut indices, p, &u32_path)?, read_u32s(&mut indices, i, &u32_path)?));
        }
        sets.push(Some(sequence));
    }
    Ok(Some(Loaded {
        driver: manifest["driver"].clone(),
        libraries,
        context: Context { stayed, was_on, new },
        running: Running { covariances, fishers, rows },
        sets,
    }))
}
