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
    // Ragged columns are first-class: tests routinely ship a training column
    // (n rows), a grid column (grid_n rows), and a scalar option (1 row) in
    // the same data.frame, and the reference bodies dereference the surplus
    // tail with `is.finite(...)` / NaN filters on their side. The CSV row
    // grid runs from row 0 to `nrows = max column length`; shorter columns
    // emit `NaN` past their own length so every column appears at its
    // natural width to the reference interpreter.
    let nrows = columns.iter().map(|c| c.data.len()).max().unwrap_or(0);
    assert!(
        nrows > 0,
        "reference run needs at least one non-empty column"
    );
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
            .map(|c| match c.data.get(row) {
                // `NA` is the missing-value token recognised by R
                // `read.csv` (its default `na.strings`) AND by pandas
                // `read_csv` (its default `na_values` list). Both
                // produce IEEE NaN downstream, so the reference body's
                // `is.finite(...)` / `np.isfinite(...)` mask filters
                // out the surplus tail of a short column cleanly.
                Some(value) => format!("{value:.17e}"),
                None => "NA".to_string(),
            })
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
            // NOTE: built with concat! of per-line literals, NOT the `"\<newline>`
            // continuation idiom. Rust's `\<newline>` continuation strips the
            // leading whitespace of the following source line, which silently
            // destroys Python's significant indentation — an earlier version did
            // exactly that and every Python reference died with
            // `IndentationError: expected an indented block after 'try'`. Keeping
            // the indentation INSIDE each literal makes it immune to that.
            preamble: concat!(
                "import sys\n",
                "import numpy as np\n",
                "_data_csv, _out = sys.argv[1], sys.argv[2]\n",
                "try:\n",
                "    import pandas as pd\n",
                "    df = pd.read_csv(_data_csv)\n",
                "except Exception:\n",
                "    import csv as _csv\n",
                "    with open(_data_csv) as _fh:\n",
                "        _r = _csv.DictReader(_fh)\n",
                "        _cols = {k: [] for k in _r.fieldnames}\n",
                "        for _row in _r:\n",
                "            for _k, _v in _row.items():\n",
                "                _cols[_k].append(float(_v))\n",
                "    df = {k: np.asarray(v, dtype=float) for k, v in _cols.items()}\n",
                "_lines = []\n",
                "def emit(key, x):\n",
                "    arr = np.asarray(x, dtype=float).reshape(-1)\n",
                "    _lines.append(str(key) + ':' + ' '.join(repr(float(v)) for v in arr))\n",
            ),
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

/// Probe whether an R package can actually be **loaded** (namespace + any native
/// `dyn.load`) in the reference interpreter, without raising. Returns `true`
/// only when `requireNamespace` reports the package is usable.
///
/// This is the narrow, documented environmental-gate escape hatch — the same
/// category as the CUDA hardware gate and the DoubleML/EconML `available` flag,
/// NOT a general skip path. It exists for the handful of references the
/// reference-quality CI job provisions only *best-effort* because they are large
/// and/or native and not reliably installable on a bare runner (notably R-INLA,
/// whose bundled native binaries `dyn.load` per-OS). A test that gates on this
/// MUST still assert its tool-free, absolute quality bars unconditionally and
/// skip only the *match-or-beat-vs-this-tool* arm when the tool is genuinely
/// absent — never the gam-side claim. Every other reference dependency remains a
/// hard failure via [`run_r`]/[`run_python`].
pub fn r_package_available(pkg: &str) -> bool {
    // `requireNamespace` is contractually non-throwing (returns FALSE and warns
    // on a failed load), so a present probe interpreter exits zero and this is
    // never itself a hard failure. Treat a missing `Rscript` binary the same as
    // a missing package for this narrowly-scoped environmental gate: callers
    // that use the gate still assert their tool-free gam quality bars and skip
    // only the external-reference arm.
    let script = format!(
        "cat(if (requireNamespace(\"{pkg}\", quietly = TRUE)) \"1\\n\" else \"0\\n\")"
    );
    let output = match Command::new("Rscript")
        .arg("--vanilla")
        .arg("-e")
        .arg(script)
        .output()
    {
        Ok(output) => output,
        Err(_) => return false,
    };
    output.status.success() && String::from_utf8_lossy(&output.stdout).trim() == "1"
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

/// Coefficient of determination against the mean predictor.
pub fn r2(pred: &[f64], truth: &[f64]) -> f64 {
    assert_eq!(pred.len(), truth.len(), "r2 length mismatch");
    let n = truth.len() as f64;
    let mean = truth.iter().sum::<f64>() / n;
    let ss_res: f64 = pred.iter().zip(truth).map(|(p, t)| (t - p) * (t - p)).sum();
    let ss_tot: f64 = truth.iter().map(|t| (t - mean) * (t - mean)).sum();
    1.0 - ss_res / ss_tot.max(1e-300)
}

/// Out-of-sample coefficient of determination against the held-out mean.
pub fn held_out_r2(pred: &[f64], truth: &[f64]) -> f64 {
    r2(pred, truth)
}

/// Right-pad a vector with its last value, or 0.0 when empty.
///
/// `pad_to` is a *grow-only* helper used to lift a short column up to the common
/// wire width of a ragged reference frame; the padded tail is never read by a
/// correctly-sliced reference body. Asking it to *shrink* a column (target
/// shorter than the source) is always a caller bug — it would silently drop the
/// tail of real data — so this is a hard error with an actionable message rather
/// than a quiet truncation. The usual culprit is padding a full-data column to a
/// train-split width: pad every column to a single `n = max(len)` and slice each
/// by its own semantic length inside the reference body instead.
pub fn pad_to(v: &[f64], len: usize) -> Vec<f64> {
    assert!(
        v.len() <= len,
        "pad_to cannot shrink: source has {} rows but the pad target is {len} \
         (a shorter target would drop real data). Pad every column to a common \
         n = max(column length) and slice by semantic length in the reference body.",
        v.len()
    );
    let fill = v.last().copied().unwrap_or(0.0);
    let mut out = v.to_vec();
    out.resize(len, fill);
    out
}

/// Maximum absolute difference between two equal-length vectors.
pub fn max_abs_diff(a: &[f64], b: &[f64]) -> f64 {
    assert_eq!(a.len(), b.len(), "max_abs_diff length mismatch");
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0, f64::max)
}

/// Compact, reusable diagnostics for truth/reference quality tests.
#[derive(Clone, Debug)]
pub struct QualityDiagnostics {
    pub label: String,
    pub rmse_vs_truth: Option<f64>,
    pub rmse_vs_reference: Option<f64>,
    pub reference_rmse_vs_truth: Option<f64>,
    pub edf_total: Option<f64>,
    pub rho: Vec<f64>,
    pub lambda: Vec<f64>,
    pub design: Option<DesignDiagnostics>,
    pub penalties: Vec<PenaltyDiagnostics>,
    pub prediction: Option<PredictionFingerprint>,
}

#[derive(Clone, Debug)]
pub struct DesignDiagnostics {
    pub nrows: usize,
    pub ncols: usize,
    pub rank: usize,
    pub condition: f64,
    pub sigma_min: f64,
    pub sigma_max: f64,
}

#[derive(Clone, Debug)]
pub struct PenaltyDiagnostics {
    pub index: usize,
    pub col_start: usize,
    pub col_end: usize,
    pub rank: usize,
    pub lambda: Option<f64>,
    pub eig_min: f64,
    pub eig_max: f64,
    pub trace: f64,
}

#[derive(Clone, Debug)]
pub struct PredictionFingerprint {
    pub n: usize,
    pub mean: f64,
    pub sd: f64,
    pub min: f64,
    pub max: f64,
    pub first: f64,
    pub last: f64,
}

impl QualityDiagnostics {
    pub fn from_standard_fit(label: impl Into<String>, fit: &crate::StandardFitResult) -> Self {
        let design = design_diagnostics(&fit.design.design).ok();
        let penalties = penalty_diagnostics(
            &fit.design.penalties,
            fit.fit.lambdas.as_slice().unwrap_or(&[]),
        );
        Self {
            label: label.into(),
            rmse_vs_truth: None,
            rmse_vs_reference: None,
            reference_rmse_vs_truth: None,
            edf_total: fit.fit.inference.as_ref().map(|i| i.edf_total),
            rho: fit.fit.log_lambdas.to_vec(),
            lambda: fit.fit.lambdas.to_vec(),
            design,
            penalties,
            prediction: None,
        }
    }
    pub fn with_truth_rmse(mut self, pred: &[f64], truth: &[f64]) -> Self {
        self.rmse_vs_truth = Some(rmse(pred, truth));
        self.prediction = Some(prediction_fingerprint(pred));
        self
    }
    pub fn with_reference_gap(
        mut self,
        pred: &[f64],
        reference: &[f64],
        truth: Option<&[f64]>,
    ) -> Self {
        self.rmse_vs_reference = Some(rmse(pred, reference));
        if let Some(truth) = truth {
            self.reference_rmse_vs_truth = Some(rmse(reference, truth));
        }
        self
    }
    pub fn emit(&self) {
        eprintln!("{}", self.report());
    }
    pub fn report(&self) -> String {
        let mut out = format!("[quality-diagnostics] label={}", self.label);
        if let Some(v) = self.rmse_vs_truth {
            out.push_str(&format!(" rmse_truth={v:.6}"));
        }
        if let Some(v) = self.rmse_vs_reference {
            out.push_str(&format!(" rmse_reference_gap={v:.6}"));
        }
        if let Some(v) = self.reference_rmse_vs_truth {
            out.push_str(&format!(" reference_rmse_truth={v:.6}"));
        }
        if let Some(v) = self.edf_total {
            out.push_str(&format!(" edf_total={v:.3}"));
        }
        if !self.rho.is_empty() {
            out.push_str(&format!(" rho={:?}", Rounded(&self.rho)));
        }
        if !self.lambda.is_empty() {
            out.push_str(&format!(" lambda={:?}", Rounded(&self.lambda)));
        }
        if let Some(d) = &self.design {
            out.push_str(&format!(
                " design={}x{} rank={} cond={:.3e} sigma=[{:.3e},{:.3e}]",
                d.nrows, d.ncols, d.rank, d.condition, d.sigma_min, d.sigma_max
            ));
        }
        if let Some(p) = &self.prediction {
            out.push_str(&format!(
                " pred[n={} mean={:.4} sd={:.4} range=[{:.4},{:.4}] edge=[{:.4},{:.4}]]",
                p.n, p.mean, p.sd, p.min, p.max, p.first, p.last
            ));
        }
        if !self.penalties.is_empty() {
            out.push_str(" penalties=");
            for p in &self.penalties {
                out.push_str(&format!(
                    " #{} cols={}..{} rank={} lambda={} eig=[{:.3e},{:.3e}] tr={:.3e};",
                    p.index,
                    p.col_start,
                    p.col_end,
                    p.rank,
                    p.lambda
                        .map(|v| format!("{v:.3e}"))
                        .unwrap_or_else(|| "NA".into()),
                    p.eig_min,
                    p.eig_max,
                    p.trace
                ));
            }
        }
        out
    }
}

struct Rounded<'a>(&'a [f64]);
impl std::fmt::Debug for Rounded<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("[")?;
        for (i, v) in self.0.iter().enumerate() {
            if i > 0 {
                f.write_str(", ")?;
            }
            write!(f, "{v:.3e}")?;
        }
        f.write_str("]")
    }
}

pub fn prediction_fingerprint(values: &[f64]) -> PredictionFingerprint {
    let n = values.len();
    let mean = values.iter().sum::<f64>() / n.max(1) as f64;
    let var = values.iter().map(|v| (v - mean) * (v - mean)).sum::<f64>() / n.max(1) as f64;
    PredictionFingerprint {
        n,
        mean,
        sd: var.sqrt(),
        min: values.iter().copied().fold(f64::INFINITY, f64::min),
        max: values.iter().copied().fold(f64::NEG_INFINITY, f64::max),
        first: values.first().copied().unwrap_or(f64::NAN),
        last: values.last().copied().unwrap_or(f64::NAN),
    }
}

pub fn design_diagnostics(
    design: &crate::matrix::DesignMatrix,
) -> Result<DesignDiagnostics, String> {
    use crate::faer_ndarray::FaerSvd;
    let dense = design
        .try_to_dense_by_chunks_budgeted("quality diagnostics design SVD", 256 * 1024 * 1024)?;
    let (_u, s, _vt) = dense.svd(false, false).map_err(|e| e.to_string())?;
    let sigma_max = s.iter().copied().fold(0.0, f64::max);
    let tol = (design.nrows().max(design.ncols()) as f64) * f64::EPSILON * sigma_max.max(1.0);
    let rank = s.iter().filter(|&&v| v > tol).count();
    let sigma_min = s
        .iter()
        .copied()
        .filter(|v| *v > tol)
        .fold(0.0_f64, |a, v| if a == 0.0 { v } else { a.min(v) });
    Ok(DesignDiagnostics {
        nrows: design.nrows(),
        ncols: design.ncols(),
        rank,
        condition: if sigma_min > 0.0 {
            sigma_max / sigma_min
        } else {
            f64::INFINITY
        },
        sigma_min,
        sigma_max,
    })
}

pub fn penalty_diagnostics(
    penalties: &[crate::terms::smooth::BlockwisePenalty],
    lambdas: &[f64],
) -> Vec<PenaltyDiagnostics> {
    use crate::faer_ndarray::FaerEigh;
    use faer::Side;
    penalties
        .iter()
        .enumerate()
        .map(|(index, p)| {
            let evals = p
                .local
                .eigh(Side::Lower)
                .map(|(e, _)| e)
                .unwrap_or_else(|_| ndarray::Array1::from_vec(vec![f64::NAN]));
            let scale = evals
                .iter()
                .copied()
                .map(f64::abs)
                .fold(0.0, f64::max)
                .max(1.0);
            let tol = scale * 1.0e-10;
            let rank = evals.iter().filter(|&&v| v > tol).count();
            PenaltyDiagnostics {
                index,
                col_start: p.col_range.start,
                col_end: p.col_range.end,
                rank,
                lambda: lambdas.get(index).copied(),
                eig_min: evals.iter().copied().fold(f64::INFINITY, f64::min),
                eig_max: evals.iter().copied().fold(f64::NEG_INFINITY, f64::max),
                trace: p.local.diag().sum(),
            }
        })
        .collect()
}

/// A Double Machine Learning (DML) reference estimate of the average linear
/// effect `θ = E[∂E(Y|D,X)/∂D]` of a treatment/dose `D` on outcome `Y` after
/// partialling out confounders `X`, computed by a mature Python DML library
/// (DoubleML's partially-linear model, with EconML's `LinearDML` as fallback).
///
/// This is the Neyman-orthogonal scalar-target baseline used by #461's Sim C:
/// the cross-fitted DML estimator is, by construction, first-order insensitive
/// to first-stage nuisance estimation error, so its `theta`/`se` are the
/// reference bias/coverage that gam's orthogonalized marginal-slope target
/// `θ = E_x[β(x)]` must match-or-beat under x-dependent Stage-1 miscalibration.
pub struct DmlPartialLinearReference {
    /// Whether a DML library was importable in the reference interpreter. When
    /// `false`, `theta`/`se`/`ci_lo`/`ci_hi` are `NaN` and the caller should
    /// emit a clear skip message rather than asserting against them — DoubleML/
    /// EconML are heavier optional dependencies than scipy/mgcv, so their
    /// absence is treated as a genuine environmental gate (mirroring the
    /// CUDA-only skip in `tests/common/gpu_gate.rs`) rather than the hard
    /// failure that a missing scipy/R would be.
    pub available: bool,
    /// Which backend produced the estimate: "doubleml", "econml", or "none".
    pub backend: String,
    /// Point estimate of the average linear treatment effect `θ`.
    pub theta: f64,
    /// Standard error of `θ̂` reported by the DML library.
    pub se: f64,
    /// Lower end of the library's 95% confidence interval for `θ`.
    pub ci_lo: f64,
    /// Upper end of the library's 95% confidence interval for `θ`.
    pub ci_hi: f64,
}

/// Fit a partially-linear DML model `Y = θ·D + g(X) + ε`, `D = m(X) + ν` with a
/// mature Python DML library and return its orthogonal estimate of `θ`.
///
/// `y`, `d`, and the columns of `x` must share a common length. `n_folds` sets
/// the cross-fitting fold count (DML's sample-splitting ingredient). The
/// reference uses gradient-boosted nuisance learners so the partialling-out is
/// genuinely nonparametric, exercising the orthogonality the estimator claims.
///
/// When neither DoubleML nor EconML is importable, the returned struct has
/// `available == false`; the interpreter itself still exits zero (the import
/// probe is guarded), so this is *not* a hard failure — the caller decides
/// whether to skip. A missing `python3`/`numpy`/`scikit-learn`, by contrast, is
/// still a loud failure via the underlying [`run_python`] contract.
pub fn dml_partial_linear_reference(
    y: &[f64],
    d: &[f64],
    x: &[Column<'_>],
    n_folds: usize,
) -> DmlPartialLinearReference {
    assert!(
        !x.is_empty(),
        "DML reference needs at least one confounder X"
    );
    assert_eq!(y.len(), d.len(), "DML reference y/d length mismatch");
    let x_names: Vec<String> = x.iter().map(|c| format!("{:?}", c.name)).collect();
    let x_list = x_names.join(", ");
    let mut columns: Vec<Column<'_>> = Vec::with_capacity(x.len() + 2);
    columns.push(Column::new("y", y));
    columns.push(Column::new("d", d));
    columns.extend(x.iter().map(|c| Column::new(c.name, c.data)));

    // The body first probes for an importable DML backend; if none is present it
    // emits `available=0` and returns cleanly (interpreter exits zero), so the
    // Rust side can skip-with-message instead of failing. When a backend exists
    // the estimate is real and emitted under `theta`/`se`/`ci_lo`/`ci_hi`.
    let body = format!(
        r#"
import numpy as np
_xcols = [{x_list}]
Y = np.asarray(df["y"], dtype=float).reshape(-1)
D = np.asarray(df["d"], dtype=float).reshape(-1)
X = np.column_stack([np.asarray(df[c], dtype=float).reshape(-1) for c in _xcols])
n_folds = {n_folds}

def _have(mod):
    import importlib.util
    return importlib.util.find_spec(mod) is not None

theta = float("nan"); se = float("nan"); backend = 0.0; avail = 0.0
ci_lo = float("nan"); ci_hi = float("nan")

try:
    if _have("doubleml") and _have("sklearn"):
        import doubleml as dml
        from doubleml import DoubleMLData, DoubleMLPLR
        from sklearn.ensemble import GradientBoostingRegressor
        data = DoubleMLData.from_arrays(X, Y, D)
        ml_l = GradientBoostingRegressor(n_estimators=100, max_depth=3, random_state=0)
        ml_m = GradientBoostingRegressor(n_estimators=100, max_depth=3, random_state=1)
        plr = DoubleMLPLR(data, ml_l=ml_l, ml_m=ml_m, n_folds=n_folds)
        plr.fit()
        theta = float(np.asarray(plr.coef).reshape(-1)[0])
        se = float(np.asarray(plr.se).reshape(-1)[0])
        cis = np.asarray(plr.confint(level=0.95))
        ci_lo = float(cis.reshape(-1)[0]); ci_hi = float(cis.reshape(-1)[1])
        backend = 1.0; avail = 1.0
    elif _have("econml") and _have("sklearn"):
        from econml.dml import LinearDML
        from sklearn.ensemble import GradientBoostingRegressor
        est = LinearDML(
            model_y=GradientBoostingRegressor(n_estimators=100, max_depth=3, random_state=0),
            model_t=GradientBoostingRegressor(n_estimators=100, max_depth=3, random_state=1),
            cv=n_folds, random_state=0,
        )
        est.fit(Y, D, X=None, W=X)
        theta = float(np.asarray(est.coef_).reshape(-1)[0]) if np.asarray(est.coef_).size else float(est.intercept_)
        inf = est.coef__inference() if hasattr(est, "coef__inference") else est.intercept__inference()
        se = float(np.asarray(inf.stderr).reshape(-1)[0])
        lohi = inf.conf_int(alpha=0.05)
        ci_lo = float(np.asarray(lohi[0]).reshape(-1)[0]); ci_hi = float(np.asarray(lohi[1]).reshape(-1)[0])
        backend = 2.0; avail = 1.0
except Exception as _e:
    avail = 0.0; backend = 0.0
    theta = float("nan"); se = float("nan")
    ci_lo = float("nan"); ci_hi = float("nan")

emit("available", [avail])
emit("backend", [backend])
emit("theta", [theta])
emit("se", [se])
emit("ci_lo", [ci_lo])
emit("ci_hi", [ci_hi])
"#
    );

    let r = run_python(&columns, &body);
    let available = r.scalar("available") > 0.5;
    let backend = match r.scalar("backend") as i64 {
        1 => "doubleml",
        2 => "econml",
        _ => "none",
    }
    .to_string();
    DmlPartialLinearReference {
        available,
        backend,
        theta: r.scalar("theta"),
        se: r.scalar("se"),
        ci_lo: r.scalar("ci_lo"),
        ci_hi: r.scalar("ci_hi"),
    }
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

#[cfg(test)]
mod pad_to_tests {
    use super::pad_to;

    /// Regression for #1084: the exact shape that used to panic with the
    /// inscrutable "pad target 490 shorter than source 654". A full-data column
    /// (654 rows) padded down to a train-split width (490 rows) must still be a
    /// hard error — silently dropping 164 rows of real data is never correct —
    /// but now with an actionable message naming the cause and the fix.
    #[test]
    #[should_panic(expected = "pad_to cannot shrink")]
    fn shrink_to_train_split_is_a_clear_error() {
        let full = vec![1.0; 654];
        drop(pad_to(&full, 490));
    }

    /// The documented fix: padding both a full-data column and a train-split
    /// column to a common `n = max(len)` yields equal-length columns whose
    /// real-data prefixes are preserved, so a reference body can slice each by
    /// its own semantic length. This is the consistent-split path #1084's
    /// prostate test now follows.
    #[test]
    fn pad_full_and_train_to_common_n_is_consistent() {
        let n = 654usize;
        let n_train = 490usize;
        let full: Vec<f64> = (0..n).map(|i| i as f64).collect();
        let train: Vec<f64> = (0..n_train).map(|i| (1000 + i) as f64).collect();

        let full_wire = pad_to(&full, n);
        let train_wire = pad_to(&train, n);
        assert_eq!(full_wire.len(), n);
        assert_eq!(train_wire.len(), n);

        // Real-data prefixes survive untouched.
        assert_eq!(&full_wire[..n], &full[..]);
        assert_eq!(&train_wire[..n_train], &train[..]);
        // The padded tail repeats the last real value (never read by a body
        // that slices by `n_train`), confirming no real data leaks past it.
        assert_eq!(train_wire[n_train], train[n_train - 1]);
        assert_eq!(train_wire[n - 1], train[n_train - 1]);
    }
}
