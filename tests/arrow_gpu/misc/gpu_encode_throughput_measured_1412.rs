//! #1412 / #988 — the GPU encode-throughput deployment gate must assert against
//! a *measured* device rows/sec, not a CPU rate scaled by a hardcoded fudge
//! factor, and the device path must actually engage (false GPU routing is a
//! hard failure, never a silent CPU fallback dressed as a pass).
//!
//! This complements `tests/owed_1412.rs` (pure gate-logic contract, no device).
//! Here we run the production device-resident penalized solve on the REAL
//! device via `gam::gpu::encode_throughput::measure_resident_solve_throughput`
//! and:
//!
//!   * assert the device path ENGAGED (no false routing) when a device is
//!     present — `engaged == true` and `measured_rows_per_sec > 0`;
//!   * assert CPU↔GPU PARITY: the device solve matches the CPU oracle Cholesky
//!     solve of the same `(XᵀWX+ridge·I)β=rhs` system (this is the gate — the
//!     CPU implementation is truth, the GPU must agree);
//!   * REPORT the measured rows/sec and its fraction of the 100K target so the
//!     deployment decision sees a real device number;
//!   * assert that on at least one canonical shape the device measurement
//!     ESTABLISHES the 100K rows/sec/GPU target (the V100 in this fleet clears
//!     it on the wide-decoder shapes).
//!
//! On a CPU-only host the device path declines cleanly (`engaged == false`) and
//! the test asserts the decline is honest (a non-engaged result never claims to
//! meet the target) — it does NOT fabricate a GPU number.

