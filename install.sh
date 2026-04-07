#!/bin/bash
set -e

# ==============================================================================
#  gam Universal Installer
# ==============================================================================
#  Installs the latest release of gam for your platform.
#  Repo: https://github.com/SauersML/gam
# ==============================================================================

# --- Colors & Styles ---
BOLD="\033[1m"
GREEN="\033[1;32m"
CYAN="\033[1;36m"
YELLOW="\033[1;33m"
RED="\033[1;31m"
RESET="\033[0m"

# --- Icons ---
ICON_CHECK="${GREEN}✔${RESET}"
ICON_CROSS="${RED}✘${RESET}"
ICON_INFO="${CYAN}ℹ${RESET}"
ICON_PKG="${YELLOW}📦${RESET}"
ICON_ROCK="${GREEN}🚀${RESET}"

# --- Configuration ---
REPO_OWNER="SauersML"
REPO_NAME="gam"
BINARY_NAME="gam"
# Default to user-local install (no sudo required, works everywhere)
INSTALL_DIR="$HOME/.local/bin"

log_info() { echo -e "${ICON_INFO}  $1"; }
log_success() { echo -e "${ICON_CHECK}  $1"; }
log_error() { echo -e "${ICON_CROSS}  ${RED}$1${RESET}"; }
log_header() { echo -e "\n${BOLD}${CYAN}=== $1 ===${RESET}\n"; }

# --- 0. Bootstrap Latest Installer Script ---
# Raw GitHub content for branch refs can be stale due to caches.
# To avoid running an outdated installer body, resolve main -> commit SHA
# and execute install.sh at that immutable commit exactly once.
bootstrap_latest_installer() {
    if [ -n "${GAM_INSTALL_BOOTSTRAPPED:-}" ]; then
        return 0
    fi

    local api_url="https://api.github.com/repos/${REPO_OWNER}/${REPO_NAME}/commits/main"
    local script_sha_url=""
    local sha=""
    local response=""
    local tmp_script=""

    local curl_args=(-fsSL --retry 5 --retry-delay 2 --retry-connrefused --connect-timeout 5 --max-time 20)
    if [ -n "$GITHUB_TOKEN" ]; then
        curl_args+=(-H "Authorization: token $GITHUB_TOKEN")
    elif [ -n "$GH_TOKEN" ]; then
        curl_args+=(-H "Authorization: token $GH_TOKEN")
    fi

    response="$(curl "${curl_args[@]}" "$api_url" 2>/dev/null || true)"
    sha="$(echo "$response" | sed -n 's/^[[:space:]]*"sha":[[:space:]]*"\([0-9a-f]\{40\}\)".*/\1/p' | head -n 1)"

    if [ -z "$sha" ]; then
        log_info "Could not resolve latest main commit SHA; continuing with current installer body."
        return 0
    fi

    script_sha_url="https://raw.githubusercontent.com/${REPO_OWNER}/${REPO_NAME}/${sha}/install.sh"
    tmp_script="$(mktemp)"
    if ! curl "${curl_args[@]}" "$script_sha_url" -o "$tmp_script" 2>/dev/null; then
        rm -f "$tmp_script"
        log_info "Could not fetch installer at commit ${sha}; continuing with current installer body."
        return 0
    fi

    if ! head -n 1 "$tmp_script" | grep -q '^#!/'; then
        rm -f "$tmp_script"
        log_info "Fetched installer looked invalid; continuing with current installer body."
        return 0
    fi

    log_info "Refreshing installer from main commit ${sha}."
    GAM_INSTALL_BOOTSTRAPPED=1 bash "$tmp_script" "$@"
    local rc=$?
    rm -f "$tmp_script"
    exit $rc
}

bootstrap_latest_installer "$@"

# --- 1. Detect Platform ---
log_header "Detecting Platform"

OS="$(uname -s | tr '[:upper:]' '[:lower:]')"
ARCH="$(uname -m | tr '[:upper:]' '[:lower:]')"

if [[ "$OS" == mingw* ]] || [[ "$OS" == msys* ]] || [[ "$OS" == cygwin* ]]; then
    log_info "Raw Git Bash detection (uname): $OS / $ARCH"
    log_info "Windows Env: PROCESSOR_ARCHITECTURE='$PROCESSOR_ARCHITECTURE', PROCESSOR_ARCHITEW6432='$PROCESSOR_ARCHITEW6432'"
fi

TARGET_ASSET=""

case "$OS" in
    linux)
        case "$ARCH" in
            x86_64)  TARGET_ASSET="gam-linux-x64.tar.gz" ;;
            aarch64) TARGET_ASSET="gam-linux-arm64.tar.gz" ;;
            *)
                log_error "Unsupported Linux architecture: $ARCH"
                exit 1
                ;;
        esac
        ;;
    darwin)
        case "$ARCH" in
            x86_64) TARGET_ASSET="gam-macos-intel.tar.gz" ;;
            arm64)  TARGET_ASSET="gam-macos-arm64.tar.gz" ;;
            *)
                log_error "Unsupported macOS architecture: $ARCH"
                exit 1
                ;;
        esac
        ;;
    mingw*|msys*|cygwin*)
        RAW_OS="$OS"
        OS="windows"
        INSTALL_DIR="$HOME/bin"
        
        # Check for Windows ARM64 environment variables (Git Bash often runs as x64 emulated)
        # Fallback: check if uname string contains "arm64" (e.g. mingw64_nt-10.0-26200-arm64)
        if [[ "$PROCESSOR_ARCHITECTURE" == "ARM64" ]] || \
           [[ "$PROCESSOR_ARCHITEW6432" == "ARM64" ]] || \
           [[ "$RAW_OS" == *"arm64"* ]]; then
            ARCH="aarch64"
            log_info "Detected Windows ARM64 environment."
        fi

        case "$ARCH" in
            x86_64) TARGET_ASSET="gam-windows-x64.zip" ;;
            aarch64|arm64) TARGET_ASSET="gam-windows-arm64.zip" ;;
            *)
                log_error "Unsupported Windows architecture: $ARCH"
                exit 1
                ;;
        esac
        ;;
    *)
        log_error "Unsupported OS: $OS"
        exit 1
        ;;
esac

log_success "Detected: ${BOLD}${OS}/${ARCH}${RESET}"
log_info "Target release asset: ${BOLD}${TARGET_ASSET}${RESET}"

# --- 2. Find Latest Published Binary Release ---
log_header "Checking Published Binary Releases"

# Prefer the canonical latest-release endpoint for deterministic selection.
# Fallback to scanning all releases only if latest is missing this platform asset.
LATEST_API_URL="https://api.github.com/repos/${REPO_OWNER}/${REPO_NAME}/releases/latest?_=$(date +%s)"
RELEASES_API_URL="https://api.github.com/repos/${REPO_OWNER}/${REPO_NAME}/releases?per_page=100&_=$(date +%s)"

CURL_ARGS=(-sL --retry 5 --retry-delay 2 --retry-connrefused --connect-timeout 5 --max-time 20 -H "Cache-Control: no-cache" -H "Pragma: no-cache")
if [ -n "$GITHUB_TOKEN" ]; then
    log_info "Using GITHUB_TOKEN for authenticated API request."
    CURL_ARGS+=(-H "Authorization: token $GITHUB_TOKEN")
elif [ -n "$GH_TOKEN" ]; then
    log_info "Using GH_TOKEN for authenticated API request."
    CURL_ARGS+=(-H "Authorization: token $GH_TOKEN")
fi

DOWNLOAD_URL=""
RELEASE_TAG=""
log_info "Resolving latest published release with asset ${BOLD}${TARGET_ASSET}${RESET}."
LATEST_RESPONSE="$(curl "${CURL_ARGS[@]}" "${LATEST_API_URL}" || true)"
DOWNLOAD_URL="$(echo "$LATEST_RESPONSE" | \
    grep "browser_download_url.*${TARGET_ASSET}" | \
    cut -d '"' -f 4 | \
    head -n 1)"
RELEASE_TAG="$(echo "$LATEST_RESPONSE" | sed -n 's/^[[:space:]]*"tag_name":[[:space:]]*"\([^"]*\)".*/\1/p' | head -n 1)"

if [ -z "$DOWNLOAD_URL" ]; then
    log_info "Latest release did not include ${TARGET_ASSET}. Falling back to release scan."
    RELEASES_RESPONSE="$(curl "${CURL_ARGS[@]}" "${RELEASES_API_URL}" || true)"
    DOWNLOAD_URL="$(echo "$RELEASES_RESPONSE" | \
        grep "browser_download_url.*${TARGET_ASSET}" | \
        cut -d '"' -f 4 | \
        head -n 1)"
fi

if [ -z "$DOWNLOAD_URL" ]; then
    log_error "Could not find a download URL for ${TARGET_ASSET} in any release."
    log_error "GitHub API Response Preview:"
    echo "$LATEST_RESPONSE" | head -n 20
    log_error "..."
    log_error "Please check https://github.com/${REPO_OWNER}/${REPO_NAME}/releases manually."
    exit 1
fi

if [ -z "$RELEASE_TAG" ]; then
    RELEASE_TAG="$(echo "$DOWNLOAD_URL" | sed -n 's#.*/download/\([^/]*\)/.*#\1#p' | head -n 1)"
fi
log_success "Using latest published binary release."
if [ -n "$RELEASE_TAG" ]; then
    log_info "Selected release tag: ${BOLD}${RELEASE_TAG}${RESET}"
fi
log_info "Download URL: ${DOWNLOAD_URL}"

# --- 3. Download & Install ---
log_header "Installing gam"

TEMP_DIR=$(mktemp -d)
trap 'rm -rf "$TEMP_DIR"' EXIT

ARCHIVE_PATH="${TEMP_DIR}/${TARGET_ASSET}"

log_info "${ICON_PKG} Downloading..."
curl -L --progress-bar --retry 5 --retry-delay 3 --retry-connrefused --connect-timeout 10 --max-time 300 -o "$ARCHIVE_PATH" "$DOWNLOAD_URL"

log_info "Extracting..."
if [[ "$TARGET_ASSET" == *.zip ]]; then
    unzip -q "$ARCHIVE_PATH" -d "$TEMP_DIR"
else
    tar -xzf "$ARCHIVE_PATH" -C "$TEMP_DIR"
fi

# Determine binary location after extraction.
# The binary inside the tarball is named identically to the asset (minus extensions)
# e.g. gam-macos-arm64.tar.gz -> gam-macos-arm64
if [[ "$TARGET_ASSET" == *.zip ]]; then
    INTERNAL_BIN_NAME="${TARGET_ASSET%.zip}.exe"
    DEST_BINARY_NAME="${BINARY_NAME}.exe"
else
    INTERNAL_BIN_NAME="${TARGET_ASSET%.tar.gz}"
    DEST_BINARY_NAME="${BINARY_NAME}"
fi

# Search for this specific file
SOURCE_BIN=$(find "$TEMP_DIR" -type f -name "${INTERNAL_BIN_NAME}" | head -n 1)

# Fallback: if not found, look for any executable file that is not the archive itself
# For Windows, look for .exe
if [ -z "$SOURCE_BIN" ] || [ ! -f "$SOURCE_BIN" ]; then
    log_info "Exact match not found. Searching for executable..."
    if [[ "$OS" == "windows" ]]; then
         SOURCE_BIN=$(find "$TEMP_DIR" -type f -name "*.exe" | head -n 1)
    else
         SOURCE_BIN=$(find "$TEMP_DIR" -type f -perm /111 ! -name "*.*" | head -n 1)
    fi
fi

if [ -z "$SOURCE_BIN" ] || [ ! -f "$SOURCE_BIN" ]; then
    log_error "Binary not found in extracted archive."
    log_error "Contents of extraction: $(ls -R "$TEMP_DIR")"
    exit 1
fi

log_info "Installing to ${INSTALL_DIR}..."

# Create install dir if it doesn't exist
if [ ! -d "$INSTALL_DIR" ]; then
    log_info "Creating directory ${INSTALL_DIR}..."
    mkdir -p "$INSTALL_DIR" || {
         log_error "Failed to create ${INSTALL_DIR}."
         exit 1
    }
fi

mv "$SOURCE_BIN" "${INSTALL_DIR}/${DEST_BINARY_NAME}"
chmod +x "${INSTALL_DIR}/${DEST_BINARY_NAME}"

# --- 4. Verify ---
log_header "Verification"

INSTALLED_BIN="${INSTALL_DIR}/${DEST_BINARY_NAME}"

# Test the binary directly from install path
if [ -x "$INSTALLED_BIN" ] && "$INSTALLED_BIN" --help >/dev/null 2>&1; then
    log_success "Successfully installed gam!"

    # If another gam earlier in PATH shadows this install, update it too when writable.
    PATH_BIN="$(command -v gam 2>/dev/null || true)"
    if [ -n "$PATH_BIN" ] && [ "$PATH_BIN" != "$INSTALLED_BIN" ]; then
        log_info "Detected another gam in PATH before ${INSTALL_DIR}: ${PATH_BIN}"

        # On Git Bash/Windows, `command -v gam` may return .../gam while the
        # installed file is .../gam.exe; these can be the exact same file.
        SAME_FILE=0
        if [ -e "$PATH_BIN" ] && [ -e "$INSTALLED_BIN" ] && [ "$PATH_BIN" -ef "$INSTALLED_BIN" ]; then
            SAME_FILE=1
        elif [[ "$OS" == "windows" ]] && [[ "$INSTALLED_BIN" == *.exe ]]; then
            # Git Bash often reports extensionless command paths (e.g. .../gam)
            # even when the underlying file is .../gam.exe.
            PATH_BIN_EXE="${PATH_BIN}.exe"
            if [ "$PATH_BIN_EXE" = "$INSTALLED_BIN" ]; then
                SAME_FILE=1
            elif [ -e "$PATH_BIN_EXE" ] && [ "$PATH_BIN_EXE" -ef "$INSTALLED_BIN" ]; then
                SAME_FILE=1
            fi
        fi

        if [ "$SAME_FILE" -eq 1 ]; then
            log_info "PATH binary already resolves to installed binary; skipping shadow update."
        elif [ -w "$PATH_BIN" ]; then
            cp "$INSTALLED_BIN" "$PATH_BIN"
            chmod +x "$PATH_BIN"
            log_success "Updated shadowing binary at ${PATH_BIN}"
        else
            log_info "Cannot overwrite ${PATH_BIN} (not writable). Use ${INSTALLED_BIN} directly or adjust PATH."
        fi
    fi
    
    # Check if install dir is in PATH
    if [[ ":$PATH:" == *":$INSTALL_DIR:"* ]]; then
        echo -e "\n${ICON_ROCK}  ${BOLD}Run 'gam --help' to get started!${RESET}\n"
    else
        # Auto-add to PATH in shell config
        PATH_LINE="export PATH=\"${INSTALL_DIR}:\$PATH\""
        
        # Determine which shell config to update
        if [ -n "$ZSH_VERSION" ] || [[ "$SHELL" == *"zsh"* ]]; then
            SHELL_RC="$HOME/.zshrc"
        else
            SHELL_RC="$HOME/.bashrc"
        fi
        
        # Check if already in config (avoid duplicates)
        if ! grep -q '.local/bin' "$SHELL_RC" 2>/dev/null; then
            echo "" >> "$SHELL_RC"
            echo "# Added by gam installer" >> "$SHELL_RC"
            echo "$PATH_LINE" >> "$SHELL_RC"
            log_success "Added ${INSTALL_DIR} to PATH in ${SHELL_RC}"
        fi
        
        echo -e "\n${ICON_ROCK}  ${BOLD}Restart your terminal or run: source ${SHELL_RC}${RESET}"
        echo -e "    Then run: ${BOLD}gam --help${RESET}\n"
    fi
else
    log_error "Binary installed but failed to run."
    log_info "Try running: ${INSTALLED_BIN} --help"
    exit 1
fi
