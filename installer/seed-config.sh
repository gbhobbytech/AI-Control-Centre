#!/usr/bin/env bash

set -euo pipefail

APP_NAME="ai-control-centre"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_CONFIG_DIR="${1:-${SCRIPT_DIR}/../packaging/default-config}"
CONFIG_DIR="${2:-${HOME}/.config/${APP_NAME}}"

CONFIG_FILES=(
    "settings.toml"
    "services.toml"
    "profiles.toml"
    "models.toml"
)

for config_file in "${CONFIG_FILES[@]}"; do
    if [[ ! -f "${DEFAULT_CONFIG_DIR}/${config_file}" ]]; then
        echo "ERROR: default configuration file is missing: ${DEFAULT_CONFIG_DIR}/${config_file}" >&2
        exit 1
    fi
done

mkdir -p "${CONFIG_DIR}"

created=0
preserved=0

for config_file in "${CONFIG_FILES[@]}"; do
    destination="${CONFIG_DIR}/${config_file}"
    if [[ -e "${destination}" ]]; then
        echo "  Preserved ${destination}"
        preserved=$((preserved + 1))
        continue
    fi

    cp "${DEFAULT_CONFIG_DIR}/${config_file}" "${destination}"
    echo "  Created   ${destination}"
    created=$((created + 1))
done

if (( created == 0 )); then
    echo "Existing user configuration preserved."
else
    echo "Created ${created} missing user configuration file(s); preserved ${preserved} existing file(s)."
fi
