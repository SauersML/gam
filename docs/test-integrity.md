# Test integrity and removal evidence

The CLI, the Python extension and package, examples, and the production code of
each workspace crate are the users that keep an item in the supported surface.
Tests and benchmarks are roots of their own targets only, so an item that only
they use is not supported surface. Absence from a linked binary's symbol table is
never evidence that an item is unused: inlining and LTO can remove symbols for
live code. A production reachability sweep must not count `#[cfg(test)]` code as a
user of production code, and must not delete test code for being unreachable
from production. Test helpers remain test code; moving them to production or
duplicating them inline does not repair a faulty reachability rule.

Before removing code, trace its source references, identify its callers in every
supported target, and compile the affected targets. The public-surface rule and
concrete API decisions are recorded in
[`rust-library-surface.md`](rust-library-surface.md). A `pub` item that neither
the CLI, the Python extension, examples, nor another crate's production code uses
is deleted, or narrowed to `pub(crate)` when its own crate's production code uses
it. Being exported, or generic, does not make an item a root.
Tests whose production behavior was removed need a semantic retirement decision;
tests that exercise surviving behavior need repair. Compilation alone cannot tell
the two apart. Do not delete a failing test merely because its fixture was deleted.
