#!/usr/bin/env bash
# Render a readable transcript from a claude-code-action execution_file:
# just what Claude says (assistant text) plus a one-line marker per tool call,
# and the final result — none of the stream-json envelope (usage, ids,
# cache_* counters, request_id, …).
#
# Usage: print-claude-transcript.sh <execution_file>
set -uo pipefail

f="${1:-}"
if [ -z "$f" ] || [ ! -f "$f" ]; then
  echo "(no Claude transcript file at '${f:-<empty>}' — the agent step may have failed before producing output)"
  exit 0
fi

# The file is stream-json: either whitespace-separated JSON objects (one event
# per line) or a single JSON array. jq consumes each top-level value, so the
# `if type=="array"` guard handles both shapes.
jq -rj '
  (if type=="array" then .[] else . end)
  | select(.type=="assistant" or .type=="result")
  | if .type=="result" then
      "\n\n──────── result ────────\n" + ((.result // "") | tostring) + "\n"
    else
      ( .message.content[]?
        | if .type=="text" then .text + "\n"
          elif .type=="tool_use" then
            "\n· " + .name
            + ( if   (.input.command?)   then ": " + (.input.command  | tostring | split("\n")[0])
                elif (.input.pattern?)   then ": " + (.input.pattern   | tostring)
                elif (.input.query?)     then ": " + (.input.query     | tostring)
                elif (.input.file_path?) then ": " + (.input.file_path | tostring)
                elif (.input.path?)      then ": " + (.input.path      | tostring)
                else "" end )
            + "\n"
          else empty end )
    end
' "$f"
