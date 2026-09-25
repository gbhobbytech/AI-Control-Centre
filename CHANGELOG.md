# Changelog

Notable changes to AI Control Centre are recorded here.

This file is intentionally concise. Detailed implementation history remains available in the Git commit log.

## 1.0.7

- Added prerequisite checks before modifying an existing installation.
- Added first-run configuration bootstrap so a fresh installation can open the Setup Wizard immediately.
- Preserved existing user configuration during reinstall and update.
- Added a real virtual-environment creation test.
- Added clearer dependency guidance for Ubuntu, Pop!_OS, Linux Mint, Debian, Fedora and Arch Linux.
- Added validation for Python version, Tkinter, `rsync` and source-tree requirements.
- Preserved an existing installation when preflight checks fail.
- Aligned package and application version metadata at 1.0.7.
- Updated packaged install, update and uninstall guidance.

## 1.0.2

- Fixed a V0.x-to-V1 migration edge case involving a stale profile-level `open_service`.
- Agent profiles now ignore that legacy field when a valid V1 harness is selected.
- Setup and Preferences remove the obsolete field when migrated configuration is saved.

## 1.0.0

V1 established the current configuration and Agent-harness model.

Key changes included:

- V1 task configuration for Coding, Chat and Agent.
- Agent harness selection rather than permanently tying Agent to one automation runtime.
- V1 model-tuning review for selected task models.
- Migration handling for earlier V0.x Agent configuration.

## 0.9.2

- Added hardware-aware model-tuning suggestions.
- Added Conservative, Balanced and Maximum GPU starting presets.
- Used detected GPU VRAM, system RAM and GGUF size to produce cautious starting values.
- Added warnings when a model is close to available system-RAM capacity.
- Kept all suggested values editable rather than replacing manual tuning.

## 0.9.0

- Added per-model llama.cpp tuning for GPU layers, context length, reply length, K/V cache formats, flash attention and startup timeout.
- Added model-setting review before first launch under the new tuning system.
- Added Save & Launch behaviour using normal service and conflict checks.
- Added import of supported launch values from a matching running llama.cpp process.
- Added configurable light/dark appearance, primary/accent colours and text size.
- Added improved launch-failure reporting with exit codes and log paths.

## 0.8.0

- Added the Setup Wizard.
- Added permanent Settings and model-root configuration.
- Added recursive GGUF discovery from user-selected model roots.
- Added Coding, Chat and Agent default-model selectors.
- Added Prompt Helper model selection.
- Added Prompt Helper CPU, GPU and Auto processing modes.
- Added optional retained Prompt Helper behaviour.
- Added advanced Prompt Helper controls for GPU layers, context length, port, startup timeout and K/V cache types.
- Added atomic configuration writes.
- Added setup-completion tracking.
- Added Prompt Helper state information to the System panel.
- Added resource-conflict handling for Prompt Helper GPU use.

## 0.7

- Introduced the lower workspace layout with the System panel and Prompt Workshop.
- Added prompt drafting, refinement, copy and task hand-off controls.

## Earlier development

Earlier development history is preserved in the Git commit log.
