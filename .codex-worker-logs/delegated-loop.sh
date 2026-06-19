#!/usr/bin/env bash
set -u

issue="$1"
root="/Users/user/gam"
repo="SauersML/gam"
log="/Users/user/gam/.codex-worker-logs/delegated-issue-${issue}.log"
codex_bin="/Users/user/.local/bin/codex"

timestamp() {
  date '+%Y-%m-%dT%H:%M:%S%z'
}

issue_state() {
  /Users/user/.local/bin/gh issue view "$issue" --repo "$repo" --json state -q .state 2>&1
}

active_codex_pid() {
  ps ax -o pid=,command= | awk -v marker="#${issue}" '
    /codex exec/ && index($0, marker) { print $1; exit }
  '
}

prompt="$(cat <<PROMPT
You are a persistent long-term Codex worker assigned only to GitHub issue #${issue} in /Users/user/gam.

First read GitHub issue #${issue} with gh and understand the requested fix. Edit the code as needed, run relevant tests, comment honest progress and final results on the issue, and close GitHub issue #${issue} only when it is honestly fixed. If it cannot be fixed in the current run, comment the blocker clearly and exit nonzero so the restart loop can retry later.

Do not work on unrelated issues. Do not use Azure services, Azure CLIs, Azure configuration, or Azure workflows. Do not create new GitHub issues except for a real correctness regression that is distinct from issue #${issue} and cannot be handled in this issue. Do not use git restore, git checkout --, git reset --hard, git revert, or any git command to edit or revert files. Do not kill Claude, Codex, or other agent processes. You are not alone in the codebase; preserve unrelated changes and continue despite unexpected changes. Backwards compatibility is not required. Dead code and unused code are not allowed; remove it or wire it correctly.

Work in /Users/user/gam until issue #${issue} is closed or a clear blocker has been commented.
PROMPT
)"

mkdir -p "$(dirname "$log")"
cd "$root" || exit 1

echo "[$(timestamp)] delegated restart loop $$ started for issue #${issue}" >> "$log"

while :; do
  state="$(issue_state)"
  state_rc="$?"
  if [ "$state_rc" -eq 0 ] && [ "$state" = "CLOSED" ]; then
    echo "[$(timestamp)] issue #${issue} is CLOSED; loop exiting" >> "$log"
    exit 0
  fi

  if [ "$state_rc" -ne 0 ]; then
    echo "[$(timestamp)] failed to read issue #${issue} state: $state" >> "$log"
    sleep 60
    continue
  fi

  existing_pid="$(active_codex_pid)"
  if [ -n "$existing_pid" ] && kill -0 "$existing_pid" 2>/dev/null; then
    echo "[$(timestamp)] issue #${issue} state=${state}; monitoring existing codex pid ${existing_pid}" >> "$log"
    while kill -0 "$existing_pid" 2>/dev/null; do
      sleep 30
      state="$(issue_state)"
      state_rc="$?"
      if [ "$state_rc" -eq 0 ] && [ "$state" = "CLOSED" ]; then
        echo "[$(timestamp)] issue #${issue} is CLOSED while pid ${existing_pid} is active; loop exiting" >> "$log"
        exit 0
      fi
    done
    echo "[$(timestamp)] existing codex pid ${existing_pid} ended for issue #${issue}" >> "$log"
    sleep 10
    continue
  fi

  echo "[$(timestamp)] issue #${issue} state=${state}; launching codex exec" >> "$log"
  "$codex_bin" exec --cd "$root" --dangerously-bypass-approvals-and-sandbox "$prompt" >> "$log" 2>&1 </dev/null &
  child="$!"
  echo "[$(timestamp)] issue #${issue} launched codex pid ${child}" >> "$log"

  while kill -0 "$child" 2>/dev/null; do
    sleep 30
    state="$(issue_state)"
    state_rc="$?"
    if [ "$state_rc" -eq 0 ] && [ "$state" = "CLOSED" ]; then
      echo "[$(timestamp)] issue #${issue} is CLOSED while codex pid ${child} is active; loop exiting" >> "$log"
      exit 0
    fi
  done

  wait "$child"
  rc="$?"
  echo "[$(timestamp)] issue #${issue} codex pid ${child} exited rc=${rc}" >> "$log"
  sleep 60
done
