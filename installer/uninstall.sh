#!/usr/bin/env bash

set -euo pipefail

APP_NAME="ai-control-centre"

INSTALL_DIR="${HOME}/.local/share/${APP_NAME}"
BIN_FILE="${HOME}/.local/bin/ai-control-centre"
DESKTOP_FILE="${HOME}/.local/share/applications/ai-control-centre.desktop"
ICON_FILE="${HOME}/.local/share/icons/hicolor/256x256/apps/ai-control-centre.png"

echo
echo "Uninstalling AI Control Centre..."
echo

rm -rf "${INSTALL_DIR}"
rm -f "${BIN_FILE}"
rm -f "${DESKTOP_FILE}"
rm -f "${ICON_FILE}"

echo "Application removed."
echo
echo "Your settings have NOT been deleted:"
echo "  ~/.config/ai-control-centre"
echo
echo "Your runtime state has NOT been deleted:"
echo "  ~/.local/state/ai-control-centre"
echo
echo "Your models, llama.cpp, ComfyUI and other AI tools"
echo "have NOT been touched."
