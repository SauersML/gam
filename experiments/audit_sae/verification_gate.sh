#!/usr/bin/env bash
set -euo pipefail

usage() {
    printf 'usage: %s <repo-dir> <target-dir> <artifact-dir> <build-env>\n' "$0" >&2
    exit 64
}

die() {
    printf 'verification_gate: %s\n' "$1" >&2
    exit 1
}

require_absolute_dir() {
    local label="$1"
    local value="$2"
    case "$value" in
        /*) ;;
        *) die "$label must be an absolute path" ;;
    esac
}

contains_rust_diagnostic() {
    local path="$1"
    LC_ALL=C grep -Eq '(^|[^[:alnum:]_])(error\[[Ee][0-9]{4}\]|[Ee][0-9]{4}|could not compile)([^[:alnum:]_]|$)' "$path"
}

copy_failed_streams() {
    local stdout_tmp="$1"
    local stderr_tmp="$2"
    local artifact_dir="$3"
    cp "$stdout_tmp" "$artifact_dir/verification-gate.failed.stdout.log"
    cp "$stderr_tmp" "$artifact_dir/verification-gate.failed.stderr.log"
}

append_log_tail() {
    local path="$1"
    if [ -s "$path" ]; then
        tail -n 20 "$path"
    else
        printf '(empty)\n'
    fi
}

write_report() {
    local repo_dir="$1"
    local target_dir="$2"
    local artifact_dir="$3"
    local build_env="$4"
    local stdout_log="$5"
    local stderr_log="$6"
    local report_tmp="$artifact_dir/results.md.tmp"
    local report_path="$artifact_dir/results.md"
    local run_date
    run_date="$(date -u +%Y-%m-%d)"

    {
        cat <<REPORT
# SAE Audit Surface

Date: $run_date

API shipped in this branch:

\`\`\`python
gamfit.audit_sae(checkpoint, activations, *, codes=None, decoder_key=None, active=None, ...)
\`\`\`

Supported external checkpoint format for the facade:

- \`.npy\` decoder matrix with shape \`K x P\`, one dictionary row per atom and one activation dimension per column.
- \`.npz\` / \`.safetensors\` containing a \`decoder\` tensor, or an explicit \`decoder_key\`.
- Python mapping or array with the same \`K x P\` decoder contract.

The facade is thin over Rust. If \`codes\` are supplied, Rust audits those frozen external encoder activations. If \`codes\` are absent, the facade calls the Rust sparse router against the frozen decoder, densifies the returned sparse layout, and calls the Rust audit entry point. The structured report includes the dual certificate, birth candidates, routability floor and empirical dark-matter fraction, per-firing coordinate SEs for harmonic blocks, per-atom Betti topology summaries, and an atlas-nerve report when a block dictionary has selected composable charts.

## MSI verification

Required crate-local gate, captured by \`experiments/audit_sae/verification_gate.sh\`:

\`\`\`text
cd $repo_dir
. $build_env
cargo check -p gam-sae --target-dir $target_dir
\`\`\`

Result: exit code 0. The harness captured stdout and stderr as separate artifacts and refused to write this verdict unless both streams were free of Rust compiler diagnostics.

Certified artifacts:

- stdout: \`$stdout_log\`
- stderr: \`$stderr_log\`

Last stdout lines:

\`\`\`text
REPORT
        append_log_tail "$stdout_log"
        cat <<'REPORT'
```

Last stderr lines:

```text
REPORT
        append_log_tail "$stderr_log"
        cat <<'REPORT'
```

## Gemma Scope 2 audit

Blocked before producing audit numbers.

What blocked it:

- The available MSI gamfit virtualenvs expose neither `gamfit.audit_sae` nor the Rust `audit_sae` pyfunction.
- No extension-build output is part of this verification artifact. The only certified gate here is the crate-local `gam-sae` check captured above.
- Because the updated Python extension was not available, I did not run the Gemma Scope 2 audit and did not fabricate dual-certified atom counts, dark-matter fractions, or Betti distributions.
REPORT
    } > "$report_tmp"

    mv "$report_tmp" "$report_path"
}

[ "$#" -eq 4 ] || usage

repo_dir="$1"
target_dir="$2"
artifact_dir="$3"
build_env="$4"

require_absolute_dir "repo-dir" "$repo_dir"
require_absolute_dir "target-dir" "$target_dir"
require_absolute_dir "artifact-dir" "$artifact_dir"
case "$build_env" in
    /*) ;;
    *) die "build-env must be an absolute path" ;;
esac

[ -d "$repo_dir" ] || die "repo-dir does not exist: $repo_dir"
[ -f "$repo_dir/Cargo.toml" ] || die "repo-dir is not a Cargo workspace root: $repo_dir"
[ -f "$build_env" ] || die "build-env does not exist: $build_env"
mkdir -p "$target_dir" "$artifact_dir"

tmp_dir="$(mktemp -d "$artifact_dir/.verification-gate.XXXXXX")"
trap 'rm -rf "$tmp_dir"' EXIT
stdout_tmp="$tmp_dir/stdout.log"
stderr_tmp="$tmp_dir/stderr.log"

cd "$repo_dir"
. "$build_env"

set +e
cargo check -p gam-sae --target-dir "$target_dir" > "$stdout_tmp" 2> "$stderr_tmp"
status="$?"
set -e

if [ "$status" -ne 0 ]; then
    copy_failed_streams "$stdout_tmp" "$stderr_tmp" "$artifact_dir"
    die "gate exited with status $status; failed streams were written without a certified verdict"
fi

if contains_rust_diagnostic "$stdout_tmp"; then
    copy_failed_streams "$stdout_tmp" "$stderr_tmp" "$artifact_dir"
    die "stdout contains Rust compiler diagnostics; failed streams were written without a certified verdict"
fi

if contains_rust_diagnostic "$stderr_tmp"; then
    copy_failed_streams "$stdout_tmp" "$stderr_tmp" "$artifact_dir"
    die "stderr contains Rust compiler diagnostics; failed streams were written without a certified verdict"
fi

stdout_log="$artifact_dir/verification-gate.stdout.log"
stderr_log="$artifact_dir/verification-gate.stderr.log"
cp "$stdout_tmp" "$stdout_log"
cp "$stderr_tmp" "$stderr_log"
write_report "$repo_dir" "$target_dir" "$artifact_dir" "$build_env" "$stdout_log" "$stderr_log"
printf 'wrote %s/results.md\n' "$artifact_dir"
