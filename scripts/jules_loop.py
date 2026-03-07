"""
Jules optimizer loop for GAM Lean files.

This script runs inside the GitHub Actions workflow and asks Jules to improve
existing Lean files in the repository. It only commits patches that modify
existing `*.lean` files and do not regress a previously passing Lean check.
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
    "This loop only accepts patches that modify existing .lean files."
)


def strip_ansi(text):
    """Remove ANSI escape sequences from a log string."""
    ansi_escape = re.compile(r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])")
    return ansi_escape.sub("", text)


def run_command(cmd, check=False):
    """Run a shell command and return stripped stdout, stderr, and exit code."""
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
    """Drop routine build noise from Lean logs."""
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
    """Read the latest local build status and filtered logs from the environment."""
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
    """Run the local Lean check script and report whether it succeeded."""
    _, stderr, code = run_command("./scripts/lean-check-all.sh")
    if code != 0 and stderr:
        sys.stderr.write(stderr[:4000])
    return code == 0


def call_jules(prompt, attempt):
    """Submit a prompt to Jules and poll until a changeset is available."""
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
    """Reject patches that violate the repository's mutation rules."""
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
    invalid_paths = [path for path in touched_files if not path.endswith(".lean")]
    if invalid_paths:
        return False, f"Patch modifies non-Lean files: {', '.join(sorted(invalid_paths))}"
    return True, ""


def build_prompt(conclusion, logs):
    """Build the Jules prompt for the current build outcome."""
    # Common restrictions for all prompts
    version_restriction = (
        "\n\nNOTE:\n"
        "- You are encouraged to proactively search the web.\n"
        "- DO NOT modify 'lean-toolchain' - the Lean version is intentionally pinned.\n"
        "- DO NOT modify version specifiers in 'lakefile.lean' (e.g., mathlib version).\n"
        "- Focus ONLY on existing .lean files for improvements.\n"
        "- Always try to improve something--commit and finish. "
        "No further instruction will be given.\n"
        "- CRITICAL: DO NOT create new files. ONLY edit existing files.\n"
        "- CRITICAL: DO NOT delete theorems. DO NOT submit patches that only delete lines.\n"
    )

    # Build the prompt based on previous run status
    if conclusion == "success":
        prompt = (
            "The Lean Proof build passed successfully. "
            "Please find one thing to do or strengthen in an existing Lean "
            "file anywhere in the repository. Do not create new files. You "
            "must successfully compile the code yourself. If the build times "
            "out or fails, do not submit it and keep working. You are not "
            "allowed to use 'native_decide' or similar."
            "If the build executes and terminates the shell, count that as a "
            "failure. Always tail build logs. You can optimize code, "
            "strengthen proofs, replace 'sorry' or 'axiom' with actual proofs. "
            "Feel free to try big or multiple tasks. We should also remove and "
            "fix specification gaming or vacuous verification. It involves "
            "writing theorem statements that appear rigorous in natural "
            "language but are mathematically constructed to be trivial or "
            "tautological. The most common tactic is begging the question, "
            "where the theorem explicitly includes the desired conclusion "
            "within its own hypothesis, rendering the proof a simple "
            "restatement of the input. Another tactic is the trivial witness, "
            "where a property regarding a complex mathematical object is "
            "proven by providing a hardcoded constant that technically "
            "satisfies a loose inequality without actually computing or "
            "representing the complex object itself. Finally, ex post facto "
            "construction involves defining a bounding function or rule only "
            "after calculating the specific error value, ensuring the "
            "condition is met by definition rather than by deriving a "
            "meaningful general law. For all of these, if they occur, we need "
            "to address them well and improve the code. Once specification "
            "gaming is fixed, you may add new Lean proofs corresponding to "
            "the Rust code or get the Lean code to match what the Rust code "
            "does."
            "IMPORTANT: Ensure your changes compile and that all proofs are "
            "valid. Axioms are just as bad as sorrys and all axioms must be "
            "replaced with real proofs. Do not assume more than is necessary. "
            "Do not attempt low-importance small changes like style "
            "improvements, comments, etc."
            "Do not break existing functionality. DO NOT DELETE THEOREMS. DO "
            "NOT submit patches that only delete lines."
            + version_restriction
        )
    else:
        # Failure case
        prompt = (
            f"The Lean Proof build failed. "
            f"Here are the logs from the run (ANSI colors stripped):\n\n{logs}\n\n"
            "Please analyze the logs and fix the errors in the existing Lean "
            "files. If the code does not compile, you can commit a small "
            "improvement even if it is not a complete fix. You can search the "
            "web to find the latest documentation for the "
            "dependencies/libraries you're using. You can proactively find "
            "examples or code snippets that can help inform your edits. It's "
            "a good idea to web search."
            "You should check if your changes compile and that all proofs are "
            "valid. However, if the code does not compile, improve what you "
            "can as much as possible before submitting. It's okay if it still "
            "fails to compile as long as it is in a better state. At the same "
            "time, feel free to fix multiple issues at once, and don't afraid "
            "to make big improvements. You can do it!"
            + version_restriction
        )
    return prompt


def get_changeset(prompt):
    """Retry Jules until a changeset is returned or retries are exhausted."""
    for attempt in range(1, MAX_RETRIES + 1):
        changeset = call_jules(prompt, attempt)
        if changeset:
            return changeset
        if attempt < MAX_RETRIES:
            time.sleep(RETRY_DELAY)
    return None


def apply_patch_on_branch(patch):
    """Write, validate, and apply the generated patch on a fresh branch."""
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
    return branch_name


def create_pull_request(message, branch_name):
    """Push the branch and open a pull request with the commit message."""
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


def main():
    """Run the Jules improvement loop and open a PR for accepted changes."""
    conclusion, logs = get_run_info()
    prompt = build_prompt(conclusion, logs)
    changeset = get_changeset(prompt)
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

    branch_name = apply_patch_on_branch(patch)

    run_command('git config user.name "Jules Bot"', check=True)
    run_command('git config user.email "jules-bot@google.com"', check=True)
    run_command(["git", "add", "--", "*.lean"], check=True)
    _, _, code = run_command("git diff --cached --quiet")
    if code == 0:
        sys.exit(0)

    if conclusion == "success" and not verify_lean_build():
        sys.exit(0)

    if conclusion != "success":
        verify_lean_build()

    create_pull_request(message, branch_name)


if __name__ == "__main__":
    main()
