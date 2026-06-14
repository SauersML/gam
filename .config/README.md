# Tool Configuration

Configuration files for local and CI test tooling.

Current files:

| File | Purpose |
| --- | --- |
| `nextest.toml` | `cargo nextest` profile and retry/timeout settings. |

Keep tool-specific configuration here when it is shared by local runs and
CI. Project metadata that package managers consume belongs at the repo
root instead.
