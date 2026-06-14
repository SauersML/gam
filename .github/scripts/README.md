# GitHub Workflow Scripts

Helper scripts used by GitHub Actions workflows.

Scripts here should be non-interactive, deterministic, and safe to run in
CI logs. Keep workflow-specific shell fragments in the workflow YAML when
they are only used once; move them here when multiple jobs need the same
behavior.

Current scripts:

| Script | Purpose |
| --- | --- |
| `print-claude-transcript.sh` | Print a Claude transcript artifact in CI diagnostics. |
