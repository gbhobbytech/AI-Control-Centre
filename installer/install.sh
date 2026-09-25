#!/usr/bin/env bash

set -euo pipefail

APP_NAME="ai-control-centre"
DISPLAY_NAME="AI Control Centre"

SOURCE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

INSTALL_DIR="${HOME}/.local/share/${APP_NAME}"
BIN_DIR="${HOME}/.local/bin"
APPLICATIONS_DIR="${HOME}/.local/share/applications"
ICON_DIR="${HOME}/.local/share/icons/hicolor/256x256/apps"

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
