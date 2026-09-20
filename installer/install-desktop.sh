#!/bin/sh
set -eu

APP_DIR=${1:-"$HOME/ai/control-centre"}
BIN="$APP_DIR/.venv/bin/ai-control-centre"
DEST_DIR="$HOME/.local/share/applications"
DEST="$DEST_DIR/ai-control-centre.desktop"

if [ ! -x "$BIN" ]; then
    echo "AI Control Centre executable not found: $BIN" >&2
    echo "Install the project into its venv first, or pass the project directory as argument 1." >&2
    exit 1
fi

mkdir -p "$DEST_DIR"
cat > "$DEST" <<EOF
[Desktop Entry]
Type=Application
Name=AI Control Centre
Comment=Task-oriented control panel for local AI services
Exec=$BIN gui
Terminal=false
Categories=Utility;Development;
StartupNotify=true
EOF
chmod 0644 "$DEST"
echo "Installed desktop launcher: $DEST"
