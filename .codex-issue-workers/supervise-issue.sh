#!/bin/sh
set -u

ISSUE="$1"
REPO_DIR="/Users/user/gam"
LOG_DIR="$REPO_DIR/.codex-issue-workers"
REPO="SauersML/gam"

PROMPT="You are a persistent Codex worker assigned only to GitHub issue #${ISSUE} in /Users/user/gam. Read the GitHub issue yourself, understand the requested change, edit the code as needed, comment progress/findings on the issue, and close the issue honestly only when the work is actually complete and verified. DO NOT run local build/test commands: no cargo, rustc, maturin, pytest, nextest, or heavy local grep/log scans. The local machine is reserved for editing and light reads only; use MSI for acceptance verification, or comment the exact command needed if MSI is unavailable. Commit and push regularly: whenever you have a coherent local change, diagnostic, or verification artifact, stage only the files relevant to issue #${ISSUE}, commit with #${ISSUE} in the message, and push to origin/main before starting any wait, MSI job, sleep, or ending a reasoning cycle. Do not hoard local WIP. Do not use Azure. Do not create new GitHub issues except for real correctness regressions discovered while working. Do not use git restore, git revert, git checkout to revert files, or any git command to edit/revert files. You are not alone in the codebase: do not kill other agents or processes, do not revert others' changes, and work around unrelated unexpected changes. Backwards compatibility is not required; avoid fallbacks and dead code. Continue until issue #${ISSUE} is CLOSED."

cd "$REPO_DIR" || exit 1
mkdir -p "$LOG_DIR"

echo "[$(date)] supervisor starting for issue #${ISSUE}"

while true; do
  STATE="$(/Users/user/.local/bin/gh issue view "$ISSUE" --repo "$REPO" --json state -q .state 2>/dev/null || echo UNKNOWN)"
  echo "[$(date)] issue #${ISSUE} state=${STATE}"

  if [ "$STATE" = "CLOSED" ]; then
    echo "[$(date)] issue #${ISSUE} closed; supervisor exiting"
    exit 0
  fi

  /Users/user/.local/bin/codex exec --cd "$REPO_DIR" --dangerously-bypass-approvals-and-sandbox "$PROMPT" </dev/null
  RC="$?"
  echo "[$(date)] codex exec for issue #${ISSUE} exited rc=${RC}"
  sleep 30
done
