//! Bug hunt: `warm_start::key::Fingerprinter` does NOT disambiguate primitive types,
//! contradicting its own module documentation.
//!
//! The `Fingerprinter` doc comment states (verbatim):
//!
//! > Each `absorb_*` writes a short type-tag + length before the data, so
//! > values of different primitive types absorbed under one tag cannot
//! > collide with `absorb_bytes` of the same payload.
//!
//! That promise was false when this was filed: `absorb_str`/`absorb_u64`
//! forwarded to `absorb_bytes` with the
//! caller's tag and the raw payload. None of them writes a per-*type*
//! discriminator — only the caller-supplied content tag and a length. So two
//! values of *different primitive types* that share a tag and whose byte
//! payloads coincide produce the *same* `Fingerprint`. This is exactly the
//! type-confusion the "type-tag" framing is documented to prevent.
//!
//! The crate's own unit test `tagged_absorptions_dont_collide` only ever checks
//! *different* tags (`b"x"` vs `b"y"`), so it never exercises the documented
//! same-tag, cross-type case.
//!
//! Each assertion below is a well-posed collision the docstring forbids; when a
//! per-type discriminator is added to the typed `absorb_*` methods, every pair
//! becomes distinct and this test passes unchanged.

use gam::warm_start::Fingerprinter;

#[test]
fn absorb_str_does_not_collide_with_equivalent_absorb_bytes() {
    // A string field vs a bytes field with the same tag and the same payload
    // bytes: distinct types, must hash differently per the type-tag contract.
    let mut a = Fingerprinter::new();
    a.absorb_str(b"k", "AB");
    let ka = a.finalize();

    let mut b = Fingerprinter::new();
    b.absorb_bytes(b"k", b"AB");
    let kb = b.finalize();

    assert_ne!(
        ka, kb,
        "absorb_str(b\"k\", \"AB\") and absorb_bytes(b\"k\", b\"AB\") must not \
         collide (str/bytes type confusion): {ka}"
    );
}
