#!/usr/bin/env bash
# Codex overnight loop: 4-prompt cycle, same session, runs forever.
# Pauses if Codex is still working.

set -u

SESSION_ID=""
CODEX_PID=""
OUTFILE=""
BUSY=false
CYCLE=0
LOG="/tmp/codex_nightrun.log"

PROMPT_1="Run a wide variety of tests. Do NOT run 'cargo test' by itself — the full suite takes too long. Instead run specific test targets like 'cargo test --lib some_test_name' or 'cargo test -p crate_name'. Run cargo check. Run Python integration tests: tests/integration_marginal_slope.py (important marginal-slope model test), tests/integration_pit_pipeline.py, and the bench test scripts in tests/. Also try scripts in scripts/. Check recent CI logs with 'gh run list' and 'gh run view' for any failures. Read any errors or test failures carefully. Find real bugs, correctness issues, or dead code. Make principled, long-term improvements — no hacks, no shortcuts, no quick patches. Do not duplicate code — use unified paths and shared abstractions that already exist in the codebase. Think carefully before changing anything. If tests pass, look deeper: read the code for logic errors, numeric bugs, missed edge cases, or places where the math doesn't match the intent. Always run relevant tests after making changes to verify nothing broke."

PROMPT_2="Continue. Keep working on what you started. If you fixed something, run relevant tests again to confirm — specific cargo test targets, Python integration tests, etc. Do NOT run the full cargo test suite. If tests pass, look for the next real issue. Stay principled — only make changes that genuinely improve correctness or code quality. Do not duplicate code — reuse existing unified paths. Do not add unnecessary abstractions."

PROMPT_3="Finish up your current work. Make sure all changes compile (cargo check) and relevant tests pass. Run the specific tests related to what you changed. Summarize what you changed and why."

PROMPT_4="Now commit and push. Stage all modified source files (src/**) with git add, write a clear commit message describing what changed, then git push to main. Do not push non-source files. Do not amend previous commits."

log() {
    local ts
    ts=$(date '+%Y-%m-%d %H:%M:%S')
    echo "[$ts] $1" | tee -a "$LOG"
}

send_message() {
    local prompt="$1"

    OUTFILE=$(mktemp /tmp/codex_nr_out.XXXXXX)

    if [ -z "$SESSION_ID" ]; then
        codex exec --json "$prompt" >"$OUTFILE" 2>/dev/null &
    else
        codex exec resume --json "$SESSION_ID" "$prompt" >"$OUTFILE" 2>/dev/null &
    fi
    CODEX_PID=$!
    BUSY=true
}

check_done() {
    if kill -0 "$CODEX_PID" 2>/dev/null; then
        return 1
    fi
    return 0
}

harvest() {
    wait "$CODEX_PID" 2>/dev/null || true

    # Grab session ID from output
    if [ -z "$SESSION_ID" ] && [ -f "$OUTFILE" ]; then
        local sid
        sid=$(grep -o '"thread_id":"[^"]*"' "$OUTFILE" 2>/dev/null | head -1 | cut -d'"' -f4) || true
        if [ -n "${sid:-}" ]; then
            SESSION_ID="$sid"
            log "Session: $SESSION_ID"
        fi
    fi

    # Print last response text
    if [ -f "$OUTFILE" ]; then
        local text
        text=$(grep '"item.completed"' "$OUTFILE" 2>/dev/null | grep -o '"text":"[^"]*"' | cut -d'"' -f4 | tail -1) || true
        if [ -n "${text:-}" ]; then
            log "  codex> ${text:0:200}"
        fi

        # Log token usage
        local tokens
        tokens=$(grep '"turn.completed"' "$OUTFILE" 2>/dev/null | tail -1) || true
        if [ -n "${tokens:-}" ]; then
            log "  tokens: ${tokens:0:200}"
        fi

        rm -f "$OUTFILE"
    fi

    BUSY=false
    CODEX_PID=""
    OUTFILE=""
}

wait_for_codex() {
    while [ "$BUSY" = true ]; do
        if check_done; then
            harvest
        else
            sleep 5
        fi
    done
}

cleanup() {
    log "Shutting down..."
    if [ -n "$CODEX_PID" ]; then
        kill "$CODEX_PID" 2>/dev/null || true
    fi
    if [ -n "$OUTFILE" ]; then
        rm -f "$OUTFILE" 2>/dev/null || true
    fi
}
trap cleanup EXIT INT TERM

PROMPTS=("$PROMPT_1" "$PROMPT_2" "$PROMPT_3" "$PROMPT_4")
LABELS=("TEST+FIX" "CONTINUE" "FINISH" "COMMIT+PUSH")

log "=== Codex overnight run started ==="
log "Log: $LOG"

while true; do
    CYCLE=$((CYCLE + 1))
    log "--- Cycle $CYCLE ---"

    for i in 0 1 2 3; do
        wait_for_codex
        log "[${LABELS[$i]}] Sending prompt..."
        send_message "${PROMPTS[$i]}"
        wait_for_codex
    done

    log "Cycle $CYCLE complete. Waiting 60s before next cycle..."
    sleep 60
done
