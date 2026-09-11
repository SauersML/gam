//! Guard for #2704: `LIBRARY_ROW_CHUNK_TARGET_BYTES` must stay the ONE copy of
//! the streamed row-chunk working-set target.
//!
//! Twelve module-local transcriptions of that value across six crates were
//! collapsed onto `gam_runtime::resource::LIBRARY_ROW_CHUNK_TARGET_BYTES` in
//! `5c445ce6d`. Nothing stopped them coming back: every one of the twelve was a
//! plain literal that compiled, passed review, and read as a local decision.
//! This test is the thing that stops it — it fails the moment the value is
//! spelled out anywhere in the workspace outside the canonical definition and a
//! small, reasoned exemption table.
//!
//! ## What is checked, and what is deliberately NOT
//!
//! The scan is on the VALUE, because a transcription is by definition not
//! reachable by name. That makes it blind in both directions the issue thread
//! established, and neither blindness is a defect of this guard:
//!
//!   * **Same value, different quantity.** Three sites spell 8 MiB and mean
//!     something else (an I/O read-ahead window, a dense-vs-streamed route
//!     threshold, a rung of a reduction task-cap ladder). Collapsing those onto
//!     the row-chunk target would be a regression dressed as a cleanup, so they
//!     are exempted BY NAME with the reason recorded, not by a blanket rule.
//!   * **Same quantity, different value.** `dense_diag_gram_chunkrows` is this
//!     exact rule at 2 MiB. No value scan can see it, and unifying it would be a
//!     silent 4x behaviour change. It is out of scope here, and stays open.
//!
//! The exemption table is itself checked for staleness: an entry that matches
//! nothing is a failure, so a deleted or renamed site cannot leave a permanent
//! hole behind it.

use std::fs;
use std::path::{Path, PathBuf};

/// Workspace subtrees scanned for transcriptions. Everything first-party that
/// can declare a Rust constant lives under one of these.
const SCAN_ROOTS: &[&str] = &["crates", "src", "tests", "bench"];

/// This file, relative to the workspace root. It is skipped because it builds
/// the very spellings it hunts for; a scanner that trips on its own needles
/// reports a violation that does not exist. Taken from `file!()` rather than
/// spelled out: a hand-written path went stale when the guard moved into
/// `tests/misc/misc/`, and the guard then reported its own doc comments.
const THIS_GUARD: &str = file!();

/// Sites that spell the canonical value but mean a DIFFERENT quantity, plus the
/// canonical definition itself. Each entry is
/// `(path suffix, substring that must appear on the offending line, reason)`.
/// Both the path and the line marker must match, so an exemption cannot spread
/// to a new literal that later appears in the same file.
const EXEMPTIONS: &[(&str, &str, &str)] = &[
    (
        "crates/gam-runtime/src/resource.rs",
        "pub const LIBRARY_ROW_CHUNK_TARGET_BYTES",
        "the canonical definition — the one copy every other site must import",
    ),
    (
        "crates/gam-linalg/src/parallel.rs",
        "} else if bytes <=",
        "a rung of the reduction task-cap ladder (64 KiB / 1 MiB / 8 MiB), which \
         caps task COUNT by working-set size; it is not a chunk size and shares \
         no consumer with the row-chunk target",
    ),
    (
        "crates/gam-models/src/fit_orchestration/drivers/design_construction.rs",
        "const STREAMING_BYTES_THRESHOLD",
        "a dense-vs-streamed ROUTE threshold on the total n*p*8 work, not a chunk \
         size; unifying it would tie a routing decision to a tile budget",
    ),
    (
        "crates/gam-sae/src/corpus/shard_reader.rs",
        "const DEFAULT_PREFETCH_WINDOW_BYTES",
        "an I/O read-ahead window on a shard reader, sized by storage latency \
         rather than by a compute working set",
    ),
];

/// Every way an author plausibly writes `value` in Rust source, derived FROM the
/// constant rather than transcribed — this guard must not itself contain the
/// literal it bans.
fn value_spellings(value: usize) -> Vec<String> {
    let mut out = vec![value.to_string(), grouped_decimal(value)];
    // `<factor> * <unit>` for every power-of-two byte unit that divides the
    // value exactly, in both the collapsed and the nested form authors write.
    for (unit, unit_text) in [
        (1024usize, "1024"),
        (1024usize * 1024, "1048576"),
        (1024usize * 1024, "1024*1024"),
        (1024usize * 1024 * 1024, "1073741824"),
        (1024usize * 1024 * 1024, "1024*1024*1024"),
    ] {
        if value % unit == 0 && value / unit > 0 {
            out.push(format!("{}*{}", value / unit, unit_text));
        }
    }
    if value.is_power_of_two() {
        let bits = value.trailing_zeros();
        out.push(format!("1<<{bits}"));
        for shift in [10u32, 20, 30] {
            if shift < bits {
                out.push(format!("{}<<{}", value >> shift, shift));
            }
        }
    }
    out.sort();
    out.dedup();
    out
}

/// `8388608` -> `8_388_608`, the underscore-grouped spelling rustfmt tolerates.
fn grouped_decimal(value: usize) -> String {
    let digits = value.to_string();
    let mut grouped = String::with_capacity(digits.len() * 4 / 3);
    for (index, digit) in digits.chars().enumerate() {
        if index > 0 && (digits.len() - index) % 3 == 0 {
            grouped.push('_');
        }
        grouped.push(digit);
    }
    grouped
}

/// Whitespace carries no meaning inside a numeric expression, so compare on the
/// whitespace-free form and let one needle cover every formatting of it.
fn squeeze(line: &str) -> String {
    line.chars().filter(|ch| !ch.is_whitespace()).collect()
}

/// Drop a Rust integer-literal type suffix so `8388608usize` is still a hit.
fn strip_int_suffix(rest: &str) -> &str {
    for suffix in [
        "usize", "u128", "u64", "u32", "u16", "u8", "isize", "i128", "i64", "i32", "i16", "i8",
    ] {
        if let Some(tail) = rest.strip_prefix(suffix) {
            return tail;
        }
    }
    rest
}

/// True when the squeezed line spells the value as a standalone quantity.
///
/// Two boundaries do the discriminating work. A hit preceded by an identifier
/// character is a digit of a LARGER number (`128 * 1024 * 1024` contains the
/// 8 MiB product), and a hit followed by a further numeric factor is a larger
/// magnitude (`8 * 1024 * 1024 * 1024` is 8 GiB). Both are rejected. A hit
/// followed by an operator that SHRINKS or consumes the value — the
/// `target / bytes_per_row` division every chunk site performs — is kept, since
/// that is precisely the transcription this guard exists to catch.
fn spells_value(squeezed: &str, needles: &[String]) -> bool {
    for needle in needles {
        let mut cursor = 0usize;
        while let Some(offset) = squeezed[cursor..].find(needle.as_str()) {
            let start = cursor + offset;
            let end = start + needle.len();
            let preceded_by_ident = squeezed[..start]
                .chars()
                .next_back()
                .is_some_and(|ch| ch.is_ascii_alphanumeric() || ch == '_' || ch == '.');
            let rest = strip_int_suffix(&squeezed[end..]);
            let followed_by_factor = match rest.chars().next() {
                None => false,
                Some(ch) if ch.is_ascii_alphanumeric() || ch == '_' => true,
                Some('*') => rest[1..].starts_with(|ch: char| ch.is_ascii_digit()),
                Some(_) => false,
            };
            if !preceded_by_ident && !followed_by_factor {
                return true;
            }
            cursor = start + 1;
        }
    }
    false
}

/// Collect every `.rs` file under `dir`, recursively, skipping build output.
fn rust_files(dir: &Path, out: &mut Vec<PathBuf>) {
    let entries = match fs::read_dir(dir) {
        Ok(entries) => entries,
        Err(_) => return,
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            if path.file_name().and_then(|name| name.to_str()) == Some("target") {
                continue;
            }
            rust_files(&path, out);
        } else if path.extension().and_then(|ext| ext.to_str()) == Some("rs") {
            out.push(path);
        }
    }
}

fn workspace_root() -> PathBuf {
    // `CARGO_MANIFEST_DIR` is the workspace root (the `gam` crate lives there).
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

fn relative_slash_path(root: &Path, path: &Path) -> String {
    path.strip_prefix(root)
        .unwrap_or(path)
        .to_string_lossy()
        .replace('\\', "/")
}

/// Non-vacuity control. A generator that produced no usable needle, or a
/// boundary rule that rejected everything, would make the sweep below pass on a
/// tree full of transcriptions. Prove the detector fires on each spelling it
/// generates, and does NOT fire on the two shapes the boundaries exist to
/// exclude.
#[test]
fn the_transcription_detector_fires_and_is_bounded() {
    let value = gam_runtime::resource::LIBRARY_ROW_CHUNK_TARGET_BYTES;
    let needles = value_spellings(value);
    assert!(
        needles.len() >= 3,
        "expected several spellings of {value}, got {needles:?}"
    );

    for needle in &needles {
        let positive = squeeze(&format!("const PROBE: usize = {needle};"));
        assert!(
            spells_value(&positive, &needles),
            "detector missed its own spelling {needle:?} in {positive:?}"
        );

        let larger_magnitude = squeeze(&format!("const PROBE: usize = {needle} * 1024;"));
        assert!(
            !spells_value(&larger_magnitude, &needles),
            "detector claimed {larger_magnitude:?} spells {value}, but it is 1024x that"
        );

        let inside_bigger_number = squeeze(&format!("const PROBE: usize = 1{needle};"));
        assert!(
            !spells_value(&inside_bigger_number, &needles),
            "detector matched {needle:?} inside the larger literal {inside_bigger_number:?}"
        );
    }

    // The division every chunk site performs must still read as a hit.
    let divided = squeeze(&format!(
        "let rows = {} / bytes_per_row;",
        needles.first().expect("at least one spelling")
    ));
    assert!(
        spells_value(&divided, &needles),
        "detector missed the chunk-sizing division {divided:?}"
    );
}

#[test]
fn the_row_chunk_target_has_exactly_one_declaration() {
    let value = gam_runtime::resource::LIBRARY_ROW_CHUNK_TARGET_BYTES;
    let needles = value_spellings(value);
    let root = workspace_root();

    let mut files = Vec::new();
    for scan_root in SCAN_ROOTS {
        rust_files(&root.join(scan_root), &mut files);
    }
    files.sort();
    assert!(
        files.len() > 100,
        "expected the workspace sweep to reach the whole first-party tree, \
         found only {} .rs files under {SCAN_ROOTS:?}",
        files.len()
    );

    let mut offenders: Vec<String> = Vec::new();
    let mut exemptions_hit = vec![false; EXEMPTIONS.len()];

    for file in &files {
        let relative = relative_slash_path(&root, file);
        if relative == THIS_GUARD {
            continue;
        }
        let contents = match fs::read_to_string(file) {
            Ok(contents) => contents,
            Err(_) => continue,
        };
        for (index, line) in contents.lines().enumerate() {
            if !spells_value(&squeeze(line), &needles) {
                continue;
            }
            let exemption = EXEMPTIONS
                .iter()
                .enumerate()
                .find(|(_, (path, marker, _))| relative.ends_with(*path) && line.contains(*marker));
            match exemption {
                Some((slot, _)) => exemptions_hit[slot] = true,
                None => offenders.push(format!("{relative}:{}: {}", index + 1, line.trim())),
            }
        }
    }

    assert!(
        offenders.is_empty(),
        "{} site(s) re-declare the canonical streamed row-chunk target ({value} bytes) \
         instead of importing `gam_runtime::resource::LIBRARY_ROW_CHUNK_TARGET_BYTES` \
         (#2704). Import it, or derive from it under a documented name; if the site \
         genuinely means a DIFFERENT quantity that merely coincides at this value, add \
         it to EXEMPTIONS in this file with the reason. Offenders:\n{}",
        offenders.len(),
        offenders.join("\n")
    );

    let stale: Vec<String> = EXEMPTIONS
        .iter()
        .zip(&exemptions_hit)
        .filter(|(_, hit)| !**hit)
        .map(|((path, marker, reason), _)| format!("{path} [{marker}] — exempted because {reason}"))
        .collect();
    assert!(
        stale.is_empty(),
        "{} EXEMPTIONS entry/entries matched nothing — the site moved, was renamed, or \
         was already fixed, and the exemption is now a hole a future transcription can \
         slip through. Delete or update each of:\n{}",
        stale.len(),
        stale.join("\n")
    );
}
