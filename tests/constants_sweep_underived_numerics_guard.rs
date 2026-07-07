//! Constants-sweep guard (SPEC.md line 19: "avoid magic constants").
//!
//! The new SAE research modules introduce a lot of numeric behavior. This guard
//! scans the *production* (non-test) region of those modules for magic numeric
//! literals that carry NEITHER a derivation comment (on the line, or on the
//! comment line directly above it) NOR an explicit reporting-only tag. A magic
//! literal with no such justification is a silent, un-auditable constant — the
//! exact thing SPEC.md line 19 bans — so the test FAILS and lists every offender
//! with its file, line, and the literal, telling the author to either derive it
//! (a `//` comment explaining where the number comes from) or tag it
//! `// reporting-only` / `// fd-ok` (a value that only shapes a reported/plotted
//! quantity, not an estimate), or promote it to a named `const`.
//!
//! Scope is a FIXED list of the freshly-landed research modules (below). Test
//! modules (`#[cfg(test)] mod ...` at the bottom of each file) are out of scope,
//! matching build.rs's convention of exempting test contexts: assertion
//! tolerances are not production magic constants.
//!
//! Deliberately-permitted literals (no justification required):
//!   * trivial values 0, 1, 2, 3, 0.5, 0.25, 10, 100, 1000 (and their `.0`
//!     float forms) — loop/step/round scaffolding;
//!   * single-digit integers (0..=9) — array indices, small dimensions;
//!   * power-of-two sizes (8..=8192) — buffer/batch/dimension sizes;
//!   * any literal inside an array-index position `[ ... ]`;
//!   * any literal on a bit-manipulation / hash line (`>>`, `<<`, `rotate_*`,
//!     `wrapping_*`, or a `0x` hex literal) — RNG shift amounts and mixing
//!     constants are structural (splitmix64 / npy parsing), not tuning knobs;
//!   * any literal on a `const NAME` / `static NAME` line — a named const is
//!     self-documenting by its identifier.
//!
//! The scanner is hand-rolled (no `regex` dev-dependency), mirroring the string/
//! comment-masking style build.rs uses for its own line-level ban scanners. It
//! masks `"..."`/`'...'` literal interiors and truncates at `//` per line, so a
//! numeral inside a string or a comment is never counted as code.

use std::path::PathBuf;

/// Fixed set of research modules this guard audits, relative to the workspace
/// root (`CARGO_MANIFEST_DIR` is the root `gam` crate).
const TARGET_FILES: &[&str] = &[
    "crates/gam-sae/src/spectrometer.rs",
    "crates/gam-sae/src/nuisance_atlas.rs",
    "crates/gam-sae/src/dual_certificate.rs",
    "crates/gam-sae/src/front_door.rs",
    "crates/gam-sae/src/null_battery.rs",
    "crates/gam-sae/examples/scale_k.rs",
];

fn workspace_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

/// Strip a single line down to its *code* bytes: `"..."` and `'...'` literal
/// interiors are replaced by spaces (so digits inside them are not scanned) and
/// everything from a `//` line comment onward is dropped. State does not carry
/// across lines (matching the per-line scan); multi-line string interiors are
/// out of scope, as they are for the analogous build.rs line scanners.
///
/// The returned string has the same byte length as the input up to the point a
/// `//` comment truncates it, so `stripped.len() < line.len()` is exactly "this
/// line carries a `//` comment" — used as the inline-justification signal.
fn strip_code(line: &str) -> String {
    let bytes = line.as_bytes();
    let mut out: Vec<u8> = Vec::with_capacity(bytes.len());
    let mut i = 0usize;
    let mut in_str = false;
    let mut quote = 0u8;
    while i < bytes.len() {
        let c = bytes[i];
        if in_str {
            if c == b'\\' && i + 1 < bytes.len() {
                out.push(b' ');
                out.push(b' ');
                i += 2;
                continue;
            }
            if c == quote {
                in_str = false;
                out.push(b' ');
            } else {
                out.push(b' ');
            }
            i += 1;
            continue;
        }
        if c == b'/' && i + 1 < bytes.len() && bytes[i + 1] == b'/' {
            break;
        }
        if c == b'"' || c == b'\'' {
            in_str = true;
            quote = c;
            out.push(b' ');
            i += 1;
            continue;
        }
        out.push(c);
        i += 1;
    }
    String::from_utf8(out).unwrap_or_default()
}

/// The first line index at which the file's test region begins: the first line
/// whose trimmed text starts with `#[cfg(test)]`. Lines from there to EOF are
/// out of scope. If there is no such marker (e.g. an example binary), the whole
/// file is in scope.
fn test_region_start(lines: &[&str]) -> usize {
    lines
        .iter()
        .position(|l| l.trim_start().starts_with("#[cfg(test)]"))
        .unwrap_or(lines.len())
}

/// A line is exempt wholesale if it declares a named const/static, or if it is a
/// bit-manipulation / hash line whose numerals are structural (shift amounts,
/// mixing constants, npy/hex parsing) rather than tuning knobs.
fn line_is_exempt(code: &str) -> bool {
    if declares_named_const(code) {
        return true;
    }
    const BIT_MARKERS: &[&str] = &[
        ">>",
        "<<",
        "rotate_left",
        "rotate_right",
        "wrapping_mul",
        "wrapping_add",
        "0x",
    ];
    BIT_MARKERS.iter().any(|m| code.contains(m))
}

/// True if the code contains `const `/`static ` immediately followed (after
/// whitespace) by an uppercase-or-underscore identifier start — a named const.
fn declares_named_const(code: &str) -> bool {
    for kw in ["const ", "static "] {
        let mut from = 0usize;
        while let Some(rel) = code[from..].find(kw) {
            let after = from + rel + kw.len();
            // keyword must be at a word boundary on the left.
            let left_ok = rel + from == 0
                || !code.as_bytes()[from + rel - 1].is_ascii_alphanumeric()
                    && code.as_bytes()[from + rel - 1] != b'_';
            let next = code[after..].trim_start().as_bytes().first().copied();
            if left_ok
                && matches!(next, Some(b) if b.is_ascii_uppercase() || b == b'_')
            {
                return true;
            }
            from = after;
        }
    }
    false
}

/// Whether the nearest non-blank line strictly above `idx` is a `//` comment,
/// i.e. a derivation/tag comment attached directly above the flagged line.
fn has_comment_above(lines: &[&str], idx: usize) -> bool {
    let mut j = idx;
    while j > 0 {
        j -= 1;
        let t = lines[j].trim();
        if t.is_empty() {
            continue;
        }
        return t.starts_with("//");
    }
    false
}

/// A numeric literal found in a stripped code line: its core text (digits with
/// an optional fractional/exponent part, underscores kept, type suffix removed)
/// and its byte start within the stripped line.
struct NumLit {
    core: String,
    start: usize,
}

/// Scan one stripped code line for numeric literals. A literal starts at an
/// ASCII digit not preceded by an identifier char or `.` (so `x2`, tuple access
/// `.0`, and the second dot of a `..` range never start one). It consumes the
/// integer part, an optional `.<digits>` fraction, an optional `[eE][+-]?<digits>`
/// exponent, and any trailing type-suffix letters (`f64`, `usize`, ...), which
/// are dropped from `core`.
fn numeric_literals(code: &str) -> Vec<NumLit> {
    let b = code.as_bytes();
    let mut out = Vec::new();
    let mut i = 0usize;
    while i < b.len() {
        let c = b[i];
        if !c.is_ascii_digit() {
            i += 1;
            continue;
        }
        let prev = if i == 0 { 0u8 } else { b[i - 1] };
        if prev.is_ascii_alphanumeric() || prev == b'_' || prev == b'.' {
            // Part of an identifier or a `.N` tuple/field access; skip the whole
            // digit run so we do not re-enter it.
            while i < b.len() && (b[i].is_ascii_alphanumeric() || b[i] == b'_') {
                i += 1;
            }
            continue;
        }
        let start = i;
        // integer part
        while i < b.len() && (b[i].is_ascii_digit() || b[i] == b'_') {
            i += 1;
        }
        // fraction: `.` followed by a digit (so `0..n` and `x.0` handled by the
        // prev-char rule; here we only extend on a real `.<digit>`).
        if i + 1 < b.len() && b[i] == b'.' && b[i + 1].is_ascii_digit() {
            i += 1;
            while i < b.len() && (b[i].is_ascii_digit() || b[i] == b'_') {
                i += 1;
            }
        }
        let core_end = {
            // exponent
            let mut k = i;
            if k < b.len() && (b[k] == b'e' || b[k] == b'E') {
                let mut m = k + 1;
                if m < b.len() && (b[m] == b'+' || b[m] == b'-') {
                    m += 1;
                }
                if m < b.len() && b[m].is_ascii_digit() {
                    while m < b.len() && b[m].is_ascii_digit() {
                        m += 1;
                    }
                    k = m;
                }
            }
            k
        };
        let core = code[start..core_end].to_string();
        // consume any trailing type-suffix letters/digits so the outer loop
        // resumes past the whole token.
        i = core_end;
        while i < b.len() && (b[i].is_ascii_alphanumeric() || b[i] == b'_') {
            i += 1;
        }
        out.push(NumLit { core, start });
    }
    out
}

/// Numeric cores that need no justification (trivial scaffolding values).
fn is_trivial_core(core: &str) -> bool {
    const TRIVIAL: &[&str] = &[
        "0", "1", "2", "3", "0.0", "1.0", "2.0", "3.0", "0.5", "0.25", "10", "100", "1000",
    ];
    if TRIVIAL.contains(&core) {
        return true;
    }
    // Integer-valued cores: single digits, powers of two, round decades.
    let digits: String = core.chars().filter(|c| *c != '_').collect();
    if let Ok(v) = digits.parse::<u128>() {
        if v <= 9 {
            return true;
        }
        const POW2: &[u128] = &[8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192];
        if POW2.contains(&v) {
            return true;
        }
        if matches!(v, 10 | 100 | 1000) {
            return true;
        }
    }
    false
}

fn scan_file(rel: &str) -> Vec<(usize, String, String)> {
    let path = workspace_root().join(rel);
    let content = std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("read {}: {e}", path.display()));
    let lines: Vec<&str> = content.lines().collect();
    let end = test_region_start(&lines);
    let mut offenders = Vec::new();
    for idx in 0..end {
        let raw = lines[idx];
        let code = strip_code(raw);
        if line_is_exempt(&code) {
            continue;
        }
        // Inline `//` comment on this very line justifies its literals.
        let has_inline_comment = code.len() < raw.len();
        for lit in numeric_literals(&code) {
            if is_trivial_core(&lit.core) {
                continue;
            }
            // array-index position: literal directly after `[`.
            if code[..lit.start].trim_end().ends_with('[') {
                continue;
            }
            if has_inline_comment {
                continue;
            }
            if has_comment_above(&lines, idx) {
                continue;
            }
            offenders.push((idx + 1, lit.core.clone(), raw.trim().to_string()));
        }
    }
    offenders
}

#[test]
fn sae_research_modules_have_no_underived_magic_numerics() {
    let mut report = String::new();
    let mut total = 0usize;
    for rel in TARGET_FILES {
        let offenders = scan_file(rel);
        if offenders.is_empty() {
            continue;
        }
        total += offenders.len();
        report.push_str(&format!("\n{rel}: {} offender(s)\n", offenders.len()));
        for (line, core, text) in &offenders {
            report.push_str(&format!("  line {line}: literal `{core}`  |  {text}\n"));
        }
    }
    assert!(
        total == 0,
        "SPEC.md line 19 (avoid magic constants): {total} untagged numeric literal(s) in the SAE \
         research modules carry no derivation comment, no `// reporting-only`/`// fd-ok` tag, and \
         are not named consts. Derive each (a `//` comment on the line or directly above), tag it \
         reporting-only, or promote it to a named const:\n{report}"
    );
}

/// Guards the guard: the scanner must recognize each exemption/flag path, so a
/// future refactor cannot silently turn it into a no-op.
#[test]
fn scanner_recognizes_flags_and_exemptions() {
    // Bare untagged tuning constant is flagged.
    let code = strip_code("    let threshold = 0.37;");
    assert!(!line_is_exempt(&code));
    let lits = numeric_literals(&code);
    assert!(lits.iter().any(|l| l.core == "0.37"));
    assert!(!is_trivial_core("0.37"));

    // Trivial and structural values are exempt.
    assert!(is_trivial_core("0.5"));
    assert!(is_trivial_core("128"));
    assert!(is_trivial_core("7"));
    assert!(!is_trivial_core("0.12"));

    // Bit-op / hash lines and named consts are wholesale-exempt.
    assert!(line_is_exempt(&strip_code("    z = (z >> 30).wrapping_mul(0xBF58);")));
    assert!(line_is_exempt(&strip_code("const CLAIMED_SNR: f64 = 3.7;")));
    assert!(!line_is_exempt(&strip_code("    let a = foo(3.7);")));

    // Numerals inside strings/comments are masked out.
    assert!(numeric_literals(&strip_code("    log(\"snr=0.37 dB\"); // note 0.99")).is_empty());

    // Range and tuple access do not spawn literals beyond the leading `0`.
    let range = numeric_literals(&strip_code("    for i in 0..n {"));
    assert!(range.iter().all(|l| l.core == "0"));
    let tuple = numeric_literals(&strip_code("    let x = pair.0;"));
    assert!(tuple.is_empty(), "tuple .0 must not be a literal: {:?}", tuple.iter().map(|l| &l.core).collect::<Vec<_>>());
}
