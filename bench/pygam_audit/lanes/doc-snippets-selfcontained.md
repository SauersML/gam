# doc-snippets-selfcontained

TITLE: Execute every docs/*.md snippet as written, with no injected df/X/y
WORK ITEM: Read audit/docs.md D17. The docs item fixes tour.md only.
Evidence: tests/test_documentation_examples.py injects df, X and y into the namespace of every doc snippet. Snippets that never define their data pass the test but fail when a user copies them.
Fix: remove the injection. Make each snippet in docs/*.md define or load its own data (gamfit.datasets or a small inline synthetic). Snippets that are intentionally partial must be marked explicitly, e.g. a `skip` fence tag, with a count cap asserted.
Coordinate: docs (owns tour.md and doc wording; this item owns the remaining docs/*.md and the test harness), thin-python (if datasets helpers move).
Acceptance: test_documentation_examples.py runs every fenced python block in a fresh namespace containing only builtins. It fails at HEAD on the snippets that relied on injection and passes after.
