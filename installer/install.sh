#!/usr/bin/env bash

set -euo pipefail

APP_NAME="ai-control-centre"
DISPLAY_NAME="AI Control Centre"

SOURCE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

INSTALL_DIR="${HOME}/.local/share/${APP_NAME}"
BIN_DIR="${HOME}/.local/bin"
APPLICATIONS_DIR="${HOME}/.local/share/applications"
ICON_DIR="${HOME}/.local/share/icons/hicolor/256x256/apps"

print_dependency_help() {
    echo
    echo "Install the missing prerequisites, then run this installer again."
    echo

    if command -v apt-get >/dev/null 2>&1; then
        echo "Ubuntu / Pop!_OS / Linux Mint / Debian:"
        echo "  sudo apt update"
        echo "  sudo apt install -y python3-venv python3-tk rsync"
    elif command -v dnf >/dev/null 2>&1; then
        echo "Fedora:"
        echo "  sudo dnf install python3 python3-tkinter rsync"
    elif command -v pacman >/dev/null 2>&1; then
        echo "Arch Linux:"
        echo "  sudo pacman -S python tk rsync"
    else
        echo "Required components:"
        echo "  Python 3.11 or newer"
        echo "  Python virtual-environment support (python3 -m venv)"
        echo "  Tkinter for Python 3"
        echo "  rsync"
        echo
        echo "Use your distribution's package manager to install the missing items."
    fi
}

preflight() {
    local failed=0
    local venv_test_dir=""
    local venv_error=""

    echo "Checking prerequisites..."

    if ! command -v python3 >/dev/null 2>&1; then
        echo "ERROR: python3 was not found."
        failed=1
    else
        if ! python3 -c 'import sys; raise SystemExit(0 if sys.version_info >= (3, 11) else 1)' >/dev/null 2>&1; then
            echo "ERROR: ${DISPLAY_NAME} requires Python 3.11 or newer."
            echo "       Found: $(python3 --version 2>&1)"
            failed=1
        fi

        if ! python3 -c 'import tkinter' >/dev/null 2>&1; then
            echo "ERROR: Python Tkinter support is not available."
            failed=1
        fi

        venv_test_dir="$(mktemp -d "${TMPDIR:-/tmp}/${APP_NAME}-venv-check.XXXXXX")"
        venv_error="${venv_test_dir}/venv-error.log"
        if ! python3 -m venv "${venv_test_dir}/venv" >/dev/null 2>"${venv_error}"; then
            echo "ERROR: Python virtual-environment support is not available or could not create an environment."
            echo "       The installer needs 'python3 -m venv' to create its private environment."
            if [[ -s "${venv_error}" ]]; then
                echo
                echo "Python reported:"
                sed -n '1,8{s/^/  /;p;}' "${venv_error}"
            fi
            failed=1
        fi
        rm -rf "${venv_test_dir}"
    fi

    if ! command -v rsync >/dev/null 2>&1; then
        echo "ERROR: rsync was not found."
        failed=1
    fi

    if [[ ! -f "${SOURCE_DIR}/pyproject.toml" ]]; then
        echo "ERROR: pyproject.toml is missing from the source tree."
        failed=1
    fi

    if [[ ! -f "${SOURCE_DIR}/assets/icons/ai-control-centre-256.png" ]]; then
        echo "ERROR: the application icon is missing from the source tree."
        failed=1
    fi

    if (( failed != 0 )); then
        print_dependency_help
        echo
        echo "No existing ${DISPLAY_NAME} installation was changed."
        exit 1
    fi

    echo "Prerequisites OK."
    echo
}


echo
echo "========================================"
echo "  AI Control Centre Installer"
echo "========================================"
echo

echo "Source:"
echo "  ${SOURCE_DIR}"
echo

echo "Install location:"
echo "  ${INSTALL_DIR}"
echo

# Run all dependency checks before changing an existing installation.
preflight

mkdir -p "${HOME}/.local/share"
mkdir -p "${BIN_DIR}"
mkdir -p "${APPLICATIONS_DIR}"
mkdir -p "${ICON_DIR}"

echo "Installing application files..."

rm -rf "${INSTALL_DIR}"
mkdir -p "${INSTALL_DIR}"

rsync -a \
    --exclude='.git/' \
    --exclude='.venv/' \
    --exclude='__pycache__/' \
    --exclude='.pytest_cache/' \
    "${SOURCE_DIR}/" \
    "${INSTALL_DIR}/"

echo "Creating Python environment..."

python3 -m venv "${INSTALL_DIR}/.venv"

"${INSTALL_DIR}/.venv/bin/python" \
    -m pip install --upgrade pip

"${INSTALL_DIR}/.venv/bin/python" \
    -m pip install "${INSTALL_DIR}"

echo "Installing launcher..."

cat > "${BIN_DIR}/ai-control-centre" <<EOF
#!/usr/bin/env bash
exec "${INSTALL_DIR}/.venv/bin/ai-control-centre" "\$@"
EOF

chmod +x "${BIN_DIR}/ai-control-centre"

echo "Installing application icon..."

cp \
    "${INSTALL_DIR}/assets/icons/ai-control-centre-256.png" \
    "${ICON_DIR}/ai-control-centre.png"

echo "Installing desktop entry..."

cat > "${APPLICATIONS_DIR}/ai-control-centre.desktop" <<EOF
[Desktop Entry]
Type=Application
Version=1.0
Name=AI Control Centre
Comment=Manage local AI models, services and workloads
Exec=${BIN_DIR}/ai-control-centre gui
Icon=ai-control-centre
Terminal=false
Categories=Utility;Development;
StartupNotify=true
Keywords=AI;LLM;llama;ComfyUI;local AI;
EOF

chmod +x "${APPLICATIONS_DIR}/ai-control-centre.desktop"

if command -v update-desktop-database >/dev/null 2>&1; then
    update-desktop-database "${APPLICATIONS_DIR}" || true
fi

if command -v gtk-update-icon-cache >/dev/null 2>&1; then
    gtk-update-icon-cache \
        -f \
        "${HOME}/.local/share/icons/hicolor" \
        >/dev/null 2>&1 || true
fi

echo
echo "========================================"
echo "Installation complete"
echo "========================================"
echo
echo "Launch from your Applications menu:"
echo "  AI Control Centre"
echo
echo "Or from a terminal:"
echo "  ai-control-centre gui"
echo
echo "User configuration is stored separately under:"
echo "  ~/.config/ai-control-centre"
echo
