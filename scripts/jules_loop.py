"""
Jules optimizer loop for GAM Lean files.

This script runs inside the GitHub Actions workflow and asks Jules to improve
existing Lean files under src/. It only commits patches that modify existing
src/**/*.lean files and do not regress a previously passing Lean check.
"""
import datetime
import os
import re
import subprocess
import sys
import time

import requests

JULES_API_URL = "https://jules.googleapis.com"
MAX_RETRIES = 2
RETRY_DELAY = 60
BRANCH_PATTERN = re.compile(r"^jules-improvement-\d{8}-\d{6}$")
DIRECT_PR_ERROR = (
    "Jules attempted to create a PR directly. "
    "This loop only accepts patches that modify existing src/**/*.lean files."
)


def strip_ansi(text):
    ansi_escape = re.compile(r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])")
    return ansi_escape.sub("", text)


def run_command(cmd, check=False):
    if isinstance(cmd, list):
        result = subprocess.run(cmd, check=False, capture_output=True, text=True)
    else:
        result = subprocess.run(cmd, shell=True, check=False, capture_output=True, text=True)
    if check and result.returncode != 0:
        sys.stderr.write(result.stdout)
        sys.stderr.write(result.stderr)
        sys.exit(result.returncode)
    return result.stdout.strip(), result.stderr.strip(), result.returncode


def filter_noise(logs):
    noise_patterns = [
        "Replayed Mathlib.",
        "✔ [",
        "Build completed successfully",
        "Checking ",
    ]
    filtered_lines = []
    for line in logs.splitlines():
        if any(pattern in line for pattern in noise_patterns):
            continue
        filtered_lines.append(line)
    return "\n".join(filtered_lines)


def get_run_info():
    local_log_file = os.environ["LOCAL_LOG_FILE"]
    local_status = os.environ["LOCAL_BUILD_STATUS"]
    with open(local_log_file, "r", encoding="utf-8") as handle:
        raw_logs = handle.read()
    logs = filter_noise(strip_ansi(raw_logs))
    max_len = 300000
    if len(logs) > max_len:
        logs = "..." + logs[-max_len:]
    return local_status, logs


def verify_lean_build():
    _, stderr, code = run_command("./scripts/lean-check-all.sh")
    if code != 0 and stderr:
        sys.stderr.write(stderr[:4000])
    return code == 0


def call_jules(prompt, attempt):
    api_key = os.environ["JULES_API_KEY"]
    repo = os.environ["GITHUB_REPOSITORY"]
    payload = {
        "prompt": prompt,
        "sourceContext": {
            "source": f"sources/github/{repo}",
            "githubRepoContext": {"startingBranch": "main"},
        },
    }
    response = requests.post(
        f"{JULES_API_URL}/v1alpha/sessions",
        headers={"X-Goog-Api-Key": api_key},
        json=payload,
        timeout=60,
    )
    if response.status_code != 200:
        print(f"Attempt {attempt} failed to create session: {response.text}")
        return None
    session_name = response.json()["name"]
    seen_ids = set()
    for _ in range(180):
        time.sleep(10)
        activities_response = requests.get(
            f"{JULES_API_URL}/v1alpha/{session_name}/activities",
            headers={"X-Goog-Api-Key": api_key},
            timeout=30,
        )
        if activities_response.status_code != 200:
            continue
        activities = activities_response.json().get("activities", [])
        activities.sort(key=lambda item: item.get("createTime", ""))
        latest_changeset = None
        for activity in activities:
            activity_id = activity.get("id")
            if activity_id in seen_ids:
                continue
            seen_ids.add(activity_id)
            for artifact in activity.get("artifacts", []):
                if "changeSet" in artifact:
                    latest_changeset = artifact["changeSet"]
                if "pullRequest" in artifact:
                    print(DIRECT_PR_ERROR)
                    return None
            if "sessionCompleted" in activity:
                return latest_changeset
        if latest_changeset:
            return latest_changeset
    return None


def validate_patch(patch):
    additions = 0
    deletions = 0
    deleted_theorem_lines = []
    new_files = []
    touched_files = set()
    current_file = None

    lines = patch.splitlines()
    for index, line in enumerate(lines):
        if line.startswith("--- /dev/null"):
            next_line = lines[index + 1] if index + 1 < len(lines) else ""
            if next_line.startswith("+++ b/"):
                new_files.append(next_line[6:])
        elif line.startswith("+++ b/"):
            current_file = line[6:]
            touched_files.add(current_file)
        elif line.startswith("+") and not line.startswith("+++"):
            additions += 1
        elif line.startswith("-") and not line.startswith("---"):
            deletions += 1
            if "theorem " in line:
                deleted_theorem_lines.append(line)

    if new_files:
        return False, "Patch creates new files"
    if deleted_theorem_lines:
        return False, "Patch deletes theorems"
    if deletions > 0 and additions == 0:
        return False, "Patch only deletes lines"
    if not touched_files:
        return False, "Patch does not modify any files"
    invalid_paths = [
        path for path in touched_files
        if not path.startswith("src/") or not path.endswith(".lean")
    ]
    if invalid_paths:
        return False, f"Patch modifies non-Lean src files: {', '.join(sorted(invalid_paths))}"
    return True, ""


def build_prompt(conclusion, logs):
    restrictions = (
        "\n\nRules:\n"
        "- Edit only existing files under src/ ending in .lean.\n"
        "- Do not create files.\n"
        "- Do not delete files.\n"
        "- Do not edit lakefile.lean, lean-toolchain, lake-manifest.json, workflows, scripts, or Rust files.\n"
        "- Do not delete theorems.\n"
        "- Do not submit patches that only delete lines.\n"
        "- Run the Lean checks yourself before submitting.\n"
        "- Prioritize meaningful proof or specification improvements over style changes.\n"
    )
    if conclusion == "success":
        return (
            "The GAM Lean checks passed. Improve an existing Lean file under src/. "
            "Strengthen proofs, remove weak or vacuous reasoning, or fix a real issue. "
            "Only edit existing src/**/*.lean files. Do not create new files."
            + restrictions
        )
    return (
        "The GAM Lean checks failed. Here are the logs:\n\n"
        f"{logs}\n\n"
        "Analyze the errors and improve the Lean files under src/. "
        "Only edit existing src/**/*.lean files. Do not create new files."
        + restrictions
    )


def main():
    conclusion, logs = get_run_info()
    prompt = build_prompt(conclusion, logs)

    changeset = None
    for attempt in range(1, MAX_RETRIES + 1):
        changeset = call_jules(prompt, attempt)
        if changeset:
            break
        if attempt < MAX_RETRIES:
            time.sleep(RETRY_DELAY)
    if not changeset:
        sys.exit(0)
    patch = changeset.get("gitPatch", {}).get("unidiffPatch")
    message = changeset.get("gitPatch", {}).get("suggestedCommitMessage", "Jules improvement")
    if not patch:
        sys.exit(0)

    is_valid, reason = validate_patch(patch)
    if not is_valid:
        print(reason)
        sys.exit(0)

    with open("jules.patch", "w", encoding="utf-8") as handle:
        handle.write(patch)

    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    branch_name = f"jules-improvement-{timestamp}"
    if not BRANCH_PATTERN.match(branch_name):
        sys.exit(1)

    run_command("git fetch origin main", check=True)
    run_command(f"git checkout -B {branch_name} origin/main", check=True)
    _, err, code = run_command("git apply jules.patch")
    if code != 0:
        print(err)
        sys.exit(0)

    run_command('git config user.name "Jules Bot"', check=True)
    run_command('git config user.email "jules-bot@google.com"', check=True)
    run_command("git add src", check=True)
    _, _, code = run_command("git diff --cached --quiet")
    if code == 0:
        sys.exit(0)

    if conclusion == "success" and not verify_lean_build():
        sys.exit(0)

    if conclusion != "success":
        verify_lean_build()

    run_command(["git", "commit", "-m", message], check=True)
    run_command(f"git push origin {branch_name}", check=True)

    github_token = os.environ["GITHUB_TOKEN"]
    subprocess.run(
        ["gh", "auth", "login", "--with-token"],
        input=github_token,
        check=True,
        capture_output=True,
        text=True,
    )
    title = message.splitlines()[0][:80]
    body = f"Automated improvement by Jules Loop.\n\nCommit message:\n```\n{message}\n```"
    run_command(
        [
            "gh",
            "pr",
            "create",
            "--title",
            title,
            "--body",
            body,
            "--base",
            "main",
            "--head",
            branch_name,
            "--label",
            "jules-loop",
        ],
        check=True,
    )


if __name__ == "__main__":
    main()
