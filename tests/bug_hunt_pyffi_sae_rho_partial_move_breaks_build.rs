//! Regression: the `gam-pyffi` crate — and therefore the entire `gamfit`
//! Python wheel — does not compile from a clean checkout at HEAD.
//!
//! In `crates/gam-pyffi/src/latent_basis_and_sae_ffi.rs`, the function
//! `sae_manifold_fit_inner` owns a `SaeManifoldRho` value (`let mut rho =
//! fitted_result.rho;`). It then *moves the `log_lambda_smooth` field out of
//! `rho` by value* into the result dict:
//!
//! ```ignore
//!     out.set_item("log_lambda_smooth", rho.log_lambda_smooth)?;   // line ~3190
//! ```
//!
//! `SaeManifoldRho::log_lambda_smooth` is a `Vec<f64>` (not `Copy`), so this is
//! a *partial move* of `rho`. Further down the same function, the still-owned
//! `rho` is borrowed again to compute the co-trained amortized-encoder report:
//!
//! ```ignore
//!     if let Ok(consistency) =
//!         term.amortized_encoder_consistency(z_view.view(), &rho) { ... }   // line ~3249
//! ```
//!
//! Borrowing `&rho` after one of its fields was moved out is a borrow-check
//! error (`E0382: borrow of partially moved value: rho`), so `cargo build -p
//! gam-pyffi`, `maturin develop`, and `maturin build` all abort and the
//! `gamfit._rust` extension cannot be produced. The `#1162` WIP commit
//! (9259c277b) added the second `&rho` borrow after the `#780` flatten
//! (e71bbba5f) had already introduced the by-value field move, leaving the
//! Python FFI crate red on `main`. (The top-level `gam` crate itself still
//! builds, which is why the `cargo test`/`cargo build --lib` CI on that crate
//! stays green and never exercises `gam-pyffi`.)
//!
//! The minimal fix is to NOT move the field out while `rho` is still needed —
//! e.g. `rho.log_lambda_smooth.clone()`, a borrow, or reordering the report
//! computation before the move. Any of those makes `gam-pyffi` compile again.
//!
//! This test asserts the borrow-check property *statically* (the `gam` crate it
//! lives in compiles fine, so it can run while `gam-pyffi` is broken): within a
//! single function, the owned `rho` must not be partially moved (its non-`Copy`
//! `log_lambda_smooth` field taken by value) BEFORE a subsequent whole-value
//! `&rho` borrow. Every compiling fix (clone / borrow / reorder) satisfies this,
//! so the test starts passing once the crate builds again, with no further edits.

use std::fs;
use std::path::Path;

/// Replace `//` line-comment and `/* */` block-comment bodies with spaces
/// (preserving newlines and byte offsets) so the scan only sees real code.
fn strip_comments(src: &str) -> String {
    let bytes = src.as_bytes();
    let mut out = String::with_capacity(src.len());
    let mut i = 0;
    let mut in_block = false;
    let mut in_line = false;
    let mut in_str = false;
    while i < bytes.len() {
        let b = bytes[i];
        if in_block {
            if b == b'*' && i + 1 < bytes.len() && bytes[i + 1] == b'/' {
                out.push_str("  ");
                i += 2;
                in_block = false;
                continue;
            }
            out.push(if b == b'\n' { '\n' } else { ' ' });
            i += 1;
            continue;
        }
        if in_line {
            if b == b'\n' {
                in_line = false;
                out.push('\n');
            } else {
                out.push(' ');
            }
            i += 1;
            continue;
        }
        if in_str {
            // Keep string bodies (so `set_item("log_lambda_smooth", ...)` survives),
            // but track escapes so a `\"` does not end the string early.
            out.push(b as char);
            if b == b'\\' && i + 1 < bytes.len() {
                out.push(bytes[i + 1] as char);
                i += 2;
                continue;
            }
            if b == b'"' {
                in_str = false;
            }
            i += 1;
            continue;
        }
        if b == b'/' && i + 1 < bytes.len() && bytes[i + 1] == b'*' {
            out.push_str("  ");
            i += 2;
            in_block = true;
            continue;
        }
        if b == b'/' && i + 1 < bytes.len() && bytes[i + 1] == b'/' {
            in_line = true;
            i += 1;
            continue;
        }
        if b == b'"' {
            in_str = true;
        }
        out.push(b as char);
        i += 1;
    }
    out
}

fn is_ident_byte(b: u8) -> bool {
    b.is_ascii_alphanumeric() || b == b'_'
}

/// Byte index just past the function header `fn <name>` (whole-word matched),
/// or `None` if the function is absent.
fn fn_start(code: &str, name: &str) -> Option<usize> {
    let needle = format!("fn {name}");
    let bytes = code.as_bytes();
    let mut from = 0usize;
    while let Some(rel) = code[from..].find(&needle) {
        let at = from + rel;
        let before_ok = at == 0 || !is_ident_byte(bytes[at - 1]);
        let after = at + needle.len();
        let after_ok = after >= bytes.len() || !is_ident_byte(bytes[after]);
        if before_ok && after_ok {
            return Some(at);
        }
        from = at + needle.len();
    }
    None
}

/// First by-value move of the `Vec<f64>` field `rho.log_lambda_smooth` in
/// `region` (a `rho.log_lambda_smooth` token that is NOT borrowed with a leading
/// `&` and NOT the receiver of a `.clone()`/`.as_slice()`/… chain). Returns the
/// byte offset of the move, or `None` if the field is never moved by value.
fn first_value_move_of_log_lambda_smooth(region: &str) -> Option<usize> {
    let needle = "rho.log_lambda_smooth";
    let bytes = region.as_bytes();
    let mut from = 0usize;
    while let Some(rel) = region[from..].find(needle) {
        let at = from + rel;
        // Preceded by `&` (skipping spaces) ⇒ a borrow, not a move.
        let mut p = at;
        while p > 0 && (bytes[p - 1] == b' ' || bytes[p - 1] == b'\n' || bytes[p - 1] == b'\t') {
            p -= 1;
        }
        let borrowed = p > 0 && bytes[p - 1] == b'&';
        // The token must be a whole field access of `rho` (not `my_rho.…`).
        let whole_word = at == 0 || !is_ident_byte(bytes[at - 1]);
        // Followed by `.` (skipping spaces) ⇒ a method/clone chain on the field,
        // which borrows the field rather than moving it out of `rho`.
        let after = at + needle.len();
        let mut q = after;
        while q < bytes.len() && (bytes[q] == b' ' || bytes[q] == b'\n' || bytes[q] == b'\t') {
            q += 1;
        }
        let chained = q < bytes.len() && bytes[q] == b'.';
        if whole_word && !borrowed && !chained {
            return Some(at);
        }
        from = at + needle.len();
    }
    None
}

/// Byte offset of the first whole-value `&rho` borrow in `region` at or after
/// `start` (a `&rho` token whose next char is neither an identifier byte nor
/// `.`, so it borrows the *whole* `rho`, not `&rho.some_field`). `None` if none.
fn first_whole_rho_borrow_after(region: &str, start: usize) -> Option<usize> {
    let bytes = region.as_bytes();
    let mut from = start;
    while let Some(rel) = region[from..].find("&rho") {
        let at = from + rel;
        let after = at + "&rho".len();
        let next = if after < bytes.len() { bytes[after] } else { b' ' };
        if !is_ident_byte(next) && next != b'.' {
            return Some(at);
        }
        from = at + "&rho".len();
    }
    None
}

#[test]
fn sae_manifold_fit_inner_does_not_borrow_rho_after_partial_move() {
    let manifest = env!("CARGO_MANIFEST_DIR");
    let path = Path::new(manifest).join("crates/gam-pyffi/src/latent_basis_and_sae_ffi.rs");
    let raw = fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("cannot read {}: {e}", path.display()));
    let code = strip_comments(&raw);

    // Scope to the body of `sae_manifold_fit_inner`: from its header up to the
    // next top-level `fn ` definition (so the unrelated second `log_lambda_smooth`
    // emit in `sae_manifold_predict_oos` is not mixed in).
    let start = fn_start(&code, "sae_manifold_fit_inner")
        .expect("sae_manifold_fit_inner must exist in latent_basis_and_sae_ffi.rs");
    let body_start = start + "fn sae_manifold_fit_inner".len();
    // Find the next top-level `\nfn ` boundary after the header.
    let rest = &code[body_start..];
    let end_rel = rest.find("\nfn ").unwrap_or(rest.len());
    let region = &rest[..end_rel];

    if let Some(move_at) = first_value_move_of_log_lambda_smooth(region) {
        let borrow = first_whole_rho_borrow_after(region, move_at);
        assert!(
            borrow.is_none(),
            "gam-pyffi does not compile: `sae_manifold_fit_inner` moves the non-Copy field \
             `rho.log_lambda_smooth` out of `rho` by value and then borrows the whole `&rho` \
             afterward (E0382: borrow of partially moved value). The gamfit wheel cannot be \
             built. Clone the field, borrow it, or reorder the `&rho` use before the move."
        );
    }
}
