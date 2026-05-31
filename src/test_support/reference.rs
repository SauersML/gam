//! End-to-end quality comparison against mature, standard statistical tools.
//!
//! The harness lets a `cargo test` integration test fit the *same* data with a
//! trusted reference implementation and assert that gam's fitted function,
//! coefficients, effective degrees of freedom, predictions, or uncertainty
//! agree with what practitioners already trust. It is deliberately
//! tool-agnostic: a test supplies an arbitrary R or Python body and the harness
//! handles all of the data plumbing and result parsing.
//!
//! Reference toolchains supported today:
//!   * **R** via `Rscript` — `mgcv`, `gamlss`, `survival`, and any package the
//!     body chooses to `library()`.
//!   * **Python** via `python3` — `scikit-learn`, `scipy`, `statsmodels`,
//!     `lifelines`, `scikit-survival`, and anything else importable.
//!
//! There is **no skip path**. If the interpreter or a required package is not
//! installed, `run_r`/`run_python` fail loudly and the test fails — a missing
//! reference dependency is a real failure, not a silent pass. CI is expected to
//! provision the reference stack. (Only genuine hardware gates, e.g. CUDA, are
//! allowed to skip; that lives in `tests/common/gpu_gate.rs`, not here.)
//!
//! Wire protocol (kept dependency-free on purpose — no JSON crate on the R/
//! Python side): the test body calls `emit("key", numeric_vector)` for every
//! quantity it wants to return. The harness reads these back as
//! `key: v1 v2 v3 ...` lines and exposes them as `f64` scalars / vectors.

use std::collections::BTreeMap;
use std::io::Write;
use std::path::PathBuf;
use std::process::Command;
use std::sync::atomic::{AtomicU64, Ordering};

/// Parsed results emitted by a reference-tool body via `emit(key, values)`.
pub struct ReferenceResult {
    values: BTreeMap<String, Vec<f64>>,
}

impl ReferenceResult {
    /// Fetch a single scalar emitted under `key`. Fails the test loudly when
    /// the key is missing or did not carry exactly one value.
    pub fn scalar(&self, key: &str) -> f64 {
        let v = self.vector(key);
        assert_eq!(
            v.len(),
            1,
            "reference key {key:?} carried {} values, expected a scalar",
            v.len()
        );
        v[0]
    }

    /// Fetch the vector emitted under `key`. Fails the test when the key is
    /// missing.
    pub fn vector(&self, key: &str) -> &[f64] {
        let msg = format!(
            "reference did not emit key {key:?}; emitted keys: {:?}",
            self.keys()
        );
        self.values.get(key).expect(&msg).as_slice()
    }

    /// Keys the reference body emitted, for diagnostics.
    pub fn keys(&self) -> Vec<&str> {
        self.values.keys().map(String::as_str).collect()
    }
}

/// A named numeric column handed to the reference body as a `data.frame` column
/// (R) or a NumPy array `df["name"]` (Python).
pub struct Column<'a> {
    /// Column header, referenced verbatim inside the reference body.
    pub name: &'a str,
    /// Column values, one per row. Length must match across all columns.
    pub data: &'a [f64],
}

impl<'a> Column<'a> {
    /// Convenience constructor.
    pub fn new(name: &'a str, data: &'a [f64]) -> Self {
        Self { name, data }
    }
}

fn unique_scratch_dir(tag: &str) -> PathBuf {
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    let n = COUNTER.fetch_add(1, Ordering::Relaxed);
    let mut dir = std::env::temp_dir();
    dir.push(format!(
        "gam_reference_{}_{}_{}",
        tag,
        std::process::id(),
        n
    ));
    std::fs::create_dir_all(&dir).expect("create reference scratch dir");
    dir
}

fn write_columns_csv(path: &std::path::Path, columns: &[Column<'_>]) {
    assert!(
        !columns.is_empty(),
        "reference run needs at least one column"
    );
    let nrows = columns[0].data.len();
    for c in columns {
        assert_eq!(
            c.data.len(),
            nrows,
            "reference column {:?} has {} rows, expected {}",
            c.name,
            c.data.len(),
            nrows
        );
    }
    let mut s = String::new();
    s.push_str(
        &columns
            .iter()
            .map(|c| c.name.to_string())
            .collect::<Vec<_>>()
            .join(","),
    );
    s.push('\n');
    for row in 0..nrows {
        let line = columns
            .iter()
            .map(|c| format!("{:.17e}", c.data[row]))
            .collect::<Vec<_>>()
            .join(",");
        s.push_str(&line);
        s.push('\n');
    }
    let mut f = std::fs::File::create(path).expect("write reference data csv");
    f.write_all(s.as_bytes()).expect("flush reference data csv");
}

fn parse_emitted(text: &str) -> BTreeMap<String, Vec<f64>> {
    let mut out: BTreeMap<String, Vec<f64>> = BTreeMap::new();
    for line in text.lines() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let Some((key, rest)) = line.split_once(':') else {
            continue;
        };
        let key = key.trim();
        if key.is_empty() {
            continue;
        }
        let values: Vec<f64> = rest
            .split_whitespace()
            .map(|tok| match tok {
                "NA" | "na" | "NaN" | "nan" => f64::NAN,
                "Inf" | "inf" => f64::INFINITY,
                "-Inf" | "-inf" => f64::NEG_INFINITY,
                other => other
                    .parse::<f64>()
                    .expect("reference emitted an unparsable numeric token"),
            })
            .collect();
        out.insert(key.to_string(), values);
    }
    out
}

/// Per-language specifics for a reference subprocess run: scratch-dir tag,
/// script filename, the interpreter `Command` (with any fixed leading args
/// already applied), the script preamble/epilogue wrapped around the test body,
/// and the human-readable name used in failure messages.
struct ReferenceKind {
    /// Short tag for the scratch directory (`"r"`, `"py"`).
    tag: &'static str,
    /// Script filename written into the scratch directory.
    script_name: &'static str,
    /// Code prepended before the test body (exposes `df`, output path, `emit`).
    preamble: &'static str,
    /// Code appended after the test body (e.g. Python's flush-to-file step).
    epilogue: &'static str,
    /// `.expect` message text for the spawn failure.
    spawn_expect: &'static str,
    /// `.expect` message text for the script-write failure.
    write_expect: &'static str,
    /// Human-readable language name used in the non-zero-exit assertion.
    display: &'static str,
}

impl ReferenceKind {
    fn r() -> Self {
        ReferenceKind {
            tag: "r",
            script_name: "script.R",
            preamble: "\
args <- commandArgs(trailingOnly = TRUE)\n\
df <- read.csv(args[1])\n\
.OUT <- args[2]\n\
emit <- function(key, x) {\n\
  cat(sprintf('%s:%s\\n', key, paste(format(as.numeric(x), digits = 17, scientific = TRUE), collapse = ' ')),\n\
      file = .OUT, append = TRUE)\n\
}\n",
            epilogue: "",
            spawn_expect: "spawn Rscript (install R to run reference-comparison tests)",
            write_expect: "write reference R script",
            display: "R",
        }
    }

    fn python() -> Self {
        ReferenceKind {
            tag: "py",
            script_name: "script.py",
            preamble: "\
import sys\n\
import numpy as np\n\
_data_csv, _out = sys.argv[1], sys.argv[2]\n\
try:\n\
    import pandas as pd\n\
    df = pd.read_csv(_data_csv)\n\
except Exception:\n\
    import csv as _csv\n\
    with open(_data_csv) as _fh:\n\
        _r = _csv.DictReader(_fh)\n\
        _cols = {k: [] for k in _r.fieldnames}\n\
        for _row in _r:\n\
            for _k, _v in _row.items():\n\
                _cols[_k].append(float(_v))\n\
    df = {k: np.asarray(v, dtype=float) for k, v in _cols.items()}\n\
_lines = []\n\
def emit(key, x):\n\
    arr = np.asarray(x, dtype=float).reshape(-1)\n\
    _lines.append(str(key) + ':' + ' '.join(repr(float(v)) for v in arr))\n",
            epilogue: "\nopen(_out, 'w').write('\\n'.join(_lines) + '\\n')\n",
            spawn_expect: "spawn python3 (install python3 to run reference-comparison tests)",
            write_expect: "write reference python script",
            display: "Python",
        }
    }

    /// Build the interpreter command with its fixed leading arguments (before
    /// the script path / data CSV / output path are appended by the runner).
    fn command(&self) -> Command {
        match self.tag {
            "r" => {
                let mut cmd = Command::new("Rscript");
                cmd.arg("--vanilla");
                cmd
            }
            _ => Command::new("python3"),
        }
    }
}

/// Run a reference body in the interpreter described by `kind`. The columns are
/// written to a CSV the script reads; the wrapped script exposes `df`, the
/// output path, and `emit("key", values)`, runs `body`, and emits results back
/// over the line protocol. Fails the test with captured stderr/stdout when the
/// interpreter exits non-zero — a broken or unavailable reference run (missing
/// interpreter, missing package, runtime error) is a hard failure, never a
/// silent skip.
fn run_subprocess(kind: &ReferenceKind, columns: &[Column<'_>], body: &str) -> ReferenceResult {
    let dir = unique_scratch_dir(kind.tag);
    let data_csv = dir.join("data.csv");
    let out_txt = dir.join("out.txt");
    let script = dir.join(kind.script_name);
    write_columns_csv(&data_csv, columns);

    let preamble = kind.preamble;
    let epilogue = kind.epilogue;
    let full = format!("{preamble}\n{body}\n{epilogue}");
    std::fs::write(&script, full).expect(kind.write_expect);

    let output = kind
        .command()
        .arg(&script)
        .arg(&data_csv)
        .arg(&out_txt)
        .output()
        .expect(kind.spawn_expect);

    let stderr = String::from_utf8_lossy(&output.stderr);
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        output.status.success(),
        "reference {} body failed (status {:?})\n--- stderr ---\n{stderr}\n--- stdout ---\n{stdout}",
        kind.display,
        output.status.code()
    );

    let emitted = std::fs::read_to_string(&out_txt).unwrap_or_default();
    let parsed = parse_emitted(&emitted);
    std::fs::remove_dir_all(&dir).ok();
    ReferenceResult { values: parsed }
}

/// Run an R reference body. The columns are exposed as a `data.frame` named
/// `df`; the body calls `emit("key", numeric_vector)` to return results. The
/// harness prepends the `df`, output path, and `emit` helper. Fails the test
/// with the captured stderr when R exits non-zero — a broken or unavailable
/// reference run (missing `Rscript`, missing package, R error) is a hard test
/// failure, never a silent skip.
pub fn run_r(columns: &[Column<'_>], body: &str) -> ReferenceResult {
    run_subprocess(&ReferenceKind::r(), columns, body)
}

/// Run a Python reference body. The columns are exposed as a pandas `df` (or,
/// when pandas is unavailable, a dict of NumPy arrays). The body calls
/// `emit("key", iterable)` to return results. Fails the test with captured
/// stderr when Python exits non-zero (missing `python3`, missing module, or a
/// raised exception).
pub fn run_python(columns: &[Column<'_>], body: &str) -> ReferenceResult {
    run_subprocess(&ReferenceKind::python(), columns, body)
}

/// Relative L2 distance `||a - b|| / max(||b||, eps)` — the natural
/// scale-free measure of how closely a fitted function tracks a reference
/// function evaluated on the same grid.
pub fn relative_l2(a: &[f64], b: &[f64]) -> f64 {
    assert_eq!(a.len(), b.len(), "relative_l2 length mismatch");
    let num: f64 = a.iter().zip(b).map(|(x, y)| (x - y) * (x - y)).sum();
    let den: f64 = b.iter().map(|y| y * y).sum();
    (num / den.max(1e-300)).sqrt()
}

/// Root-mean-square difference between two equal-length vectors.
pub fn rmse(a: &[f64], b: &[f64]) -> f64 {
    assert_eq!(a.len(), b.len(), "rmse length mismatch");
    let s: f64 = a.iter().zip(b).map(|(x, y)| (x - y) * (x - y)).sum();
    (s / a.len().max(1) as f64).sqrt()
}

/// Maximum absolute difference between two equal-length vectors.
pub fn max_abs_diff(a: &[f64], b: &[f64]) -> f64 {
    assert_eq!(a.len(), b.len(), "max_abs_diff length mismatch");
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0, f64::max)
}

/// Pearson correlation between two equal-length vectors.
pub fn pearson(a: &[f64], b: &[f64]) -> f64 {
    assert_eq!(a.len(), b.len(), "pearson length mismatch");
    let n = a.len() as f64;
    let ma = a.iter().sum::<f64>() / n;
    let mb = b.iter().sum::<f64>() / n;
    let mut sab = 0.0;
    let mut saa = 0.0;
    let mut sbb = 0.0;
    for (x, y) in a.iter().zip(b) {
        let da = x - ma;
        let db = y - mb;
        sab += da * db;
        saa += da * da;
        sbb += db * db;
    }
    sab / (saa.sqrt() * sbb.sqrt()).max(1e-300)
}
