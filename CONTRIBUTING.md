# Contributing to gam

## Banned patterns and their principled replacements

`build.rs` enforces these rules at compile time; the build fails with a
report listing every offending site.  The entries below describe _why_
each pattern is banned and what to write instead.

---

### 1. `let _ = expr;` / `let _name = expr;` / all-underscore tuple patterns

**Why banned.** An underscore-prefixed `let` binding tells the compiler
"ignore this value".  It silences the unused-value diagnostic without
using the value, hiding real logic errors (a fallible call whose error is
dropped, a computation whose result is never consumed).

**What to use instead.**

```rust
// If you want side effects only:
do_something_with_side_effects();   // call as a statement; no binding needed

// If the value must be consumed to end a lifetime / trigger Drop:
drop(my_owned_value);               // NOT for Copy types — see rule 2

// If the result is fallible and you genuinely don't care:
let _ = ...;                        // BANNED — propagate with `?` instead
```

---

### 2. `drop(Copy)` / `drop(&T)`

**Why banned.** `drop` on a `Copy` type or a shared reference is a no-op:
the compiler does not move the value, so `Drop::drop` is never called.
Clippy hard-errors on this because the only plausible intent is to
silence an unused-variable warning — the same prohibited act as `let _`.

**What to use instead.**

```rust
// For a Copy type you want to "consume" to satisfy the borrow checker:
let v: u32 = compute();
// Just use v directly, or restructure so it isn't computed needlessly.

// For an owned non-Copy value at end of scope: let normal Drop handle it.
```

---

### 3. `_param: T` or bare `_: T` in function signatures

**Why banned.** An underscore-prefixed parameter name is the same
"rename to silence the warning" act as `let _ = name;`, collapsed into
the signature.  It hides unused parameters from the lint rather than
fixing them.

**What to use instead.**

```rust
// Use the parameter in the function body.
fn compute(scale: f64, offset: f64) -> f64 {
    scale * 2.0 + offset
}

// Or remove the parameter from the API if it is never needed.
fn compute(scale: f64) -> f64 {
    scale * 2.0
}
```

---

### 4. `hint::black_box(name)` / `std::hint::black_box(name)` as a silencer

**Why banned.** `black_box` is the second-round dodge for silencing an
unused-value warning after `let _ = name;` became illegal.  It prevents
optimizer elimination without actually using the value in any real logic.

**What to use instead.**

```rust
// Return the value, store it in a struct field, pass it to another fn —
// anything that makes it part of real computation.

// Exception: microbenchmarks under benches/ legitimately call black_box
// to prevent the compiler from eliminating the measured expression.
// That use is allowed in bench scope only.
```

---

### 5. `#[allow(unused_*)]` / `#[allow(dead_code)]` / `#[expect(...)]`

**Why banned.** Every file-level or item-level `allow` is an admission
that a lint fired on real code and the author chose to suppress the
signal instead of fix it.  `expect` is the promotion form of `allow` —
it silences identically and is banned for the same reason.

**What to use instead.**

```rust
// Rename the identifier, restructure the API, or delete the dead item.
// If the lint is wrong site-wide, add a project-level exception in
// build.rs — that is the single source of "we accept this category".

// NEVER write:
// #[allow(dead_code)]
// pub fn unused_helper() { ... }

// INSTEAD: delete the helper or add a production caller.
```

---

### 6. `std::env::var(...)` / `env::vars(...)` / `env::var_os(...)`

**Why banned.** Reading environment variables at runtime is an invisible
configuration surface — the program behaves differently depending on the
process environment with no declaration in the source.  All configuration
must flow through typed `const` values or auto-derived paths from problem
characteristics.

**What to use instead.**

```rust
// Route via a typed constant in the relevant module.
const DEFAULT_BACKEND: &str = "cpu";

// Or auto-derive from problem characteristics (e.g. GPU availability)
// at runtime using the existing resource-policy infrastructure, not via
// an env-var toggle.
```

---

### 7. `#[cfg(feature = "...")]` / Cargo `[features]` entries

**Why banned.** Feature gates carve the codebase into
conditionally-compiled forks.  `rustc`'s `dead_code` lint sees only one
fork at a time; the test suite has to enumerate every buildable
configuration; and the gate itself is the same "make the lint shut up
depending on context" family as `#[allow(dead_code)]`.

**What to use instead.**

```rust
// Auto-detect from problem characteristics at runtime.
// For example, GPU vs CPU dispatch uses the existing GpuPolicy
// infrastructure — no feature gate needed.

let backend = if gpu_available() {
    Backend::Gpu
} else {
    Backend::Cpu
};
```

---

### 8. `debug_assert!` / `debug_assert_eq!` / `debug_assert_ne!`

**Why banned.** A `debug_assert` compiles to nothing in release builds.
An invariant worth checking is worth checking unconditionally; one that
is not should be deleted.  Tests that assert only in debug mode silently
pass in release CI.

**What to use instead.**

```rust
// Use unconditional assert! / assert_eq! for correctness invariants.
assert!(value > 0.0, "expected positive value, got {value}");

// For solver invariants that are too expensive in production,
// express them as a Result check and return an EstimationError.
```

---

### 9. New CLI flags / env-var toggles for behavior selection ("magic by default")

**Why banned.** Every new flag multiplies the test-configuration space
and forces callers to know which flag to pass.  The project policy is
"magic by default": the solver auto-selects the right path from problem
characteristics (data size, sparsity, family, GPU availability).

**What to use instead.**

```rust
// Inspect problem characteristics inside the solver and branch silently.
let strategy = if n > BIOBANK_SCALE_THRESHOLD {
    SolveStrategy::ApproximateLanczos
} else {
    SolveStrategy::Exact
};

// Expose knobs only via existing typed config structs (FitConfig,
// ResourcePolicy, etc.) when genuinely needed, not via new flags.
```

---

### 10. `-> Result<_, String>` for internal solver / family errors

**Why banned.** Returning a bare `String` as the error type discards
structure; callers cannot match on the failure kind without parsing
strings.  Internal errors belong in the typed error enums already
defined in the codebase.

**What to use instead.**

```rust
// Internal solver / PIRLS / REML errors -> EstimationError
fn run_pirls(setup: &PirlsSetup) -> Result<PirlsResult, EstimationError> { ... }

// Custom-family evaluation errors -> CustomFamilyError
fn evaluate(&self, blocks: &[ParameterBlockState])
    -> Result<FamilyEvaluation, CustomFamilyError> { ... }

// External boundaries (Python FFI, file I/O, CLI argument parsing)
// are the ONLY sites where String errors are acceptable; the boundary
// converts via `.map_err(|e| e.to_string())`.
```

---

### 11. `#[cfg(any())]` / `#[cfg(all())]` (empty-arg cfg)

**Why banned.** `#[cfg(any())]` is permanently `false`; `#[cfg(all())]`
is permanently `true`.  Both are dead-by-construction guards — the
annotated item is either never compiled or always compiled, with no
runtime or build-time sensitivity.

**What to use instead.**

Delete the item or express the condition via a real `cfg` predicate
(`target_os`, `target_arch`, etc.) or a runtime branch.

---

### 12. `const FLAG: bool = false;` dead-guard constants

**Why banned.** A `const bool` used as a branch guard fools `rustc`'s
`dead_code` lint: the "dead" branch is syntactically reachable, so the
lint cannot report the code inside it.  Real toggles belong in `cfg`
attributes (compile-time) or runtime configuration structs (run-time).

**What to use instead.**

```rust
// Compile-time: use a real cfg predicate.
#[cfg(target_os = "linux")]
fn platform_specific() { ... }

// Run-time: use a typed config field with a documented default.
let policy = ResourcePolicy { use_gpu: false, .. };
```

---

### 13. `#[should_panic]` without `expected = "..."`

**Why banned.** A bare `#[should_panic]` catches _any_ panic, including
a panic from an unrelated bug introduced much later.  It masks failures
by design.

**What to use instead.**

```rust
#[test]
#[should_panic(expected = "entry_age must be positive")]
fn rejects_non_positive_entry_age() {
    build_spec(-1.0);
}
```

---

### 14. `vendor/` directories

**Why banned.** Vendored crates fork the dependency tree from
crates.io/git, hide upstream security fixes, and bypass the same lint
gate this scanner enforces inside the vendored code.

**What to use instead.**

Declare all dependencies in `Cargo.toml` using a crates.io version
specifier or `git = "..."`.  Use `cargo update` to pull security fixes.
