# AI Control Centre

**Current development version: 1.0.7** - Linux packaging and installer prerequisite handling.

AI Control Centre is a lightweight, task-oriented Linux control panel for running several local AI tools on one computer, particularly when GPU memory and other resources are limited.

Instead of keeping the entire AI stack running, the Control Centre starts the services needed for the selected task, checks readiness, exposes useful system status and keeps machine-specific paths and launch settings in user configuration.

The normal task controls are:

- **Coding** - launch the configured coding LLM environment
- **Chat** - launch the configured general chat LLM environment
- **Agent** - launch a selected model plus an Agent harness
- **Image** - launch the configured image-generation environment
- **Stop All** - stop launcher-owned services in dependency-aware order

The lower workspace combines a compact **System** panel with **Prompt Workshop** for prompt drafting and refinement.

## Install - recommended desktop application method

AI Control Centre currently targets desktop Linux and requires:

- Bash
- Python **3.11 or newer**
- Python virtual-environment support (`python3 -m venv`)
- Tkinter for Python 3
- `rsync`

Download or clone the project, then run:

~~~bash
chmod +x installer/install.sh installer/uninstall.sh
./installer/install.sh
~~~

The installer performs its prerequisite checks **before changing an existing installation**. It verifies that Python is new enough, Tkinter imports correctly, `rsync` exists, and `python3 -m venv` can actually create a temporary virtual environment.

This matters on clean Ubuntu-family systems where `python3` may already exist while the separate virtual-environment package is not installed.

If a prerequisite is missing, the installer stops and prints appropriate package guidance. It does **not** automatically run `sudo` or alter system packages.

For Ubuntu, Pop!_OS, Linux Mint or Debian, the usual prerequisite command is:

~~~bash
sudo apt update
sudo apt install -y python3-venv python3-tk rsync
~~~

For Fedora:

~~~bash
sudo dnf install python3 python3-tkinter rsync
~~~

For Arch Linux:

~~~bash
sudo pacman -S python tk rsync
~~~

The user-level installation is placed under:

~~~text
~/.local/share/ai-control-centre
~~~

The installer also creates:

~~~text
~/.local/bin/ai-control-centre
~/.local/share/applications/ai-control-centre.desktop
~/.local/share/icons/hicolor/256x256/apps/ai-control-centre.png
~~~

User configuration remains separate under:

~~~text
~/.config/ai-control-centre
~~~

Runtime state remains separate under:

~~~text
~/.local/state/ai-control-centre
~~~

Re-running the installer updates the application without deleting those user configuration or state directories.

## V1.0.7

V1.0.7 hardens the Linux application packaging introduced in V1.0.6 and brings the README and version metadata into line with the current application.

Changes include:

- prerequisite checks before the installer replaces application files
- a real temporary `venv` creation test rather than assuming virtual-environment support is installed
- clear dependency guidance for Ubuntu/Pop!_OS/Mint/Debian, Fedora and Arch
- Tkinter, Python-version, `rsync` and source-tree validation
- preservation of an existing installation when preflight fails
- package version metadata aligned at 1.0.7
- current packaged install, update and uninstall instructions
- remembered main-window size from the V1.0.6 packaging work

## Launch

Open **AI Control Centre** from the Linux Applications menu, or run:

~~~bash
ai-control-centre gui
~~~

If `~/.local/bin` is not yet on your shell `PATH`, use:

~~~bash
~/.local/bin/ai-control-centre gui
~~~

Run Setup explicitly with:

~~~bash
ai-control-centre setup
~~~

## First run and V1 setup

V1 deliberately treats pre-V1 main-model launch tuning as historical rather than active. Coding, Chat and Agent models must be reviewed and saved under the V1 tuning schema before they can be used as task defaults.

The Setup Wizard guides the user through:

1. detected host capabilities and model roots
2. local GGUF model discovery
3. Coding, Chat and Agent model selection
4. Agent harness selection
5. model tuning for every selected task model
6. Prompt Helper behaviour
7. review and save

Settings remain available under **Preferences > Settings...** and models can be rediscovered with **Preferences > Rescan Models**.

## Agent harnesses

Agent is **model + harness** rather than being permanently tied to one automation runtime.

A harness can combine one or more configured services plus the interface that should be opened when the harness becomes ready. The bundled Orson reference uses Computer DMZ, while migrated V0.x Agent configurations can be surfaced as an **Existing Agent Harness** rather than being silently discarded.

## Current V1 capabilities

The current application includes:

- recursive local GGUF discovery, including split models
- Coding, Chat and Agent task defaults
- Agent harness selection
- per-model GPU layers, context length, output length, K/V cache, flash-attention and startup-timeout settings
- Conservative, Balanced and Maximum GPU starting suggestions
- optional import of supported launch values from a matching running llama.cpp process
- Prompt Helper CPU, GPU and Auto processing modes
- Prompt Workshop with editable refinement output, clipboard copy and task hand-off
- service readiness checks, logs and dependency-aware profile startup
- live CPU, RAM, GPU, VRAM and temperature monitoring where supported
- appearance settings using the gbhobbytech deep-blue/copper defaults
- remembered main-window size
- user-level desktop installation and uninstall support

AI Control Centre does **not** install the complete AI stack for the user. Services such as llama.cpp, ComfyUI and Docker-based Agent harnesses remain user-installed and user-configured.

## Update

Stop launcher-owned services first, then run the installer from the newer downloaded or cloned source tree:

~~~bash
ai-control-centre stop-all
./installer/install.sh
~~~

The application files are replaced while `~/.config/ai-control-centre` and `~/.local/state/ai-control-centre` remain separate.

## Uninstall

From a downloaded or cloned source tree:

~~~bash
./installer/uninstall.sh
~~~

The uninstaller removes the installed application, launcher, desktop entry and icon. It deliberately leaves user configuration, runtime state, models, llama.cpp, ComfyUI, Docker data and other AI tools untouched.

## Developer install

For development, an editable virtual-environment install is still supported:

~~~bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
~~~

Run tests with:

~~~bash
PYTHONPATH=src python -m pytest -q
~~~

## Historical release notes

## V1.0.2: Agent harness migration hotfix

V1.0.2 fixes a V0.x-to-V1 migration edge case where a stale profile-level `open_service` could invalidate an Agent profile even though a valid V1 harness already owned the runtime/interface to open. V1 now ignores that legacy field whenever an Agent harness is selected and removes it when Setup or Preferences saves the migrated configuration.

Model configuration launched from Setup now uses a scrollable window with a pinned **Back / Save & Continue** action bar so controls remain reachable on shorter desktops. Secondary setup/preferences windows are sized against the current screen rather than assuming a fixed desktop height.

## V1.0.0: configuration reset and agent harnesses

V1.0 deliberately treats all pre-V1 main-model launch tuning as historical rather than active. The first V1 launch forces the Setup Wizard again. Coding, Chat and Agent models must each be configured under the V1 model-tuning schema before setup can finish or the model can be assigned as a task default. Historical values remain in `models.toml` for safety, but the V1 tuning screen starts from neutral baseline values until the user saves a new configuration.

Agent is now **model + harness**, rather than being permanently synonymous with Computer DMZ. Harnesses are declared in `profiles.toml` and can combine one or more configured services plus the interface to open. The bundled Orson reference uses `Computer DMZ`, while migrated V0.x Agent profiles are surfaced as an `Existing Agent Harness` so working environments are not silently lost.

The main menu is now **Preferences** rather than File. The normal five-button interface remains Coding, Chat, Agent, Image and Stop All.

### V1 setup flow

1. detect host capabilities and model roots
2. discover local GGUF models
3. choose Coding, Chat and Agent models
4. choose the Agent harness
5. configure every selected task model using Conservative, Balanced, Maximum GPU or manual settings
6. configure Prompt Helper behaviour
7. review and save

Settings remain available later under **Preferences > Settings...**, and newly discovered models cannot be saved as task defaults until they have V1 launch settings.

## V0.9.2: suggested starting settings

The **Model tuning** tab now detects available GPU VRAM and system RAM, reads the selected GGUF's total on-disk size, and offers three clearly labelled starting presets: **Conservative**, **Balanced** and **Maximum GPU**. The suggestions fill the existing manual fields; they do not hide or replace advanced tuning.

The ready reckoner intentionally uses cautious heuristics rather than claiming an exact optimum from a filename. GPU-layer suggestions are approximate because model architectures differ. Every preset is described as a starting point, and the UI still asks the user to confirm a successful launch and test a representative prompt before tuning further. If no supported GPU memory reading is available, GPU-oriented presets fall back to CPU-first values. The panel also warns when the model file is close to the detected system-RAM capacity.

AI Control Centre is a lightweight, task-oriented Linux control panel for running several local AI tools on one computer. It starts only the services required for the selected task, checks readiness, protects resource conflicts and keeps machine-specific details in user configuration rather than application code.


## V0.9.0: model tuning and appearance

Open **Preferences > Settings > Model tuning**, or use **Tune model…** on the main window.
Each main llama.cpp model has its own GPU layers, context length, maximum reply
length, K/V cache formats, flash attention mode and startup timeout. Short hints
explain the memory and compatibility trade-offs. Values are stored in models.toml.

Before the first launch, review and save that model's settings. This also applies
once to models configured in an earlier release. Existing explicit model values
are shown for review; missing values use 0 GPU layers, 4096 context tokens,
2048 reply tokens, f16 caches, automatic flash attention and a 180-second timeout.
These are conservative starting values, not a guarantee that the model fits RAM.
The shared service defaults no longer silently set the hardware tuning for a main
llama launch. Coding, Chat and Agent use the selected model's saved tuning.
Prompt Helper remains separately configured in its existing settings tab.

**Save model settings** does not alter a running process. **Save & launch** uses
the normal service start and conflict checks, selects this model as the temporary
LLM override, and requests permission before restarting a launcher-owned server
whose model or launch arguments differ. It does not change your task defaults.
Readiness means the health endpoint responds, not that generation performance has
been benchmarked. Watch the system panel and test a representative prompt before
increasing context or GPU layers further. If launch fails, use the service Logs
button; process exits now include their exit code and log path.

**Import from running server…** lists readable same-user Linux processes whose
executable and explicit model path match the configured main llama service and
selected model. Choose a PID, review the supported explicit values, then save.
Missing values stay unchanged. Import does not adopt, stop or restart the process.
Docker/remote servers, model selection through environment variables or -hf,
environment-only settings, and backend defaults are not inferred. Auto/all GPU
layer values and unlimited output must be replaced with explicit numeric limits
in this release. The importer reads only supported launch arguments into the UI;
it does not display credentials or copy arbitrary commands.

Under **Appearance**, choose light/dark, primary/accent colours and text size.
The default palette is gbhobbytech deep blue (#16324F), copper (#B8734F),
graphite and light neutral surfaces. These are editable UI shades based on the
gbhobbytech palette. Aptos is used if installed, with an available Linux fallback.
Apply and save appearance updates the open windows. Restore defaults resets
the fields; apply to save them. No new application dependencies are required.

### Upgrade from 0.8

1. Close AI Control Centre. Keep your existing user configuration.
2. Extract this release and replace the application source files in your existing
   project directory. Do not copy examples/orson over your user configuration.
3. In that project directory, activate your existing virtual environment and run:

   ```bash
   source .venv/bin/activate
   python -m pip install -e .
   ```

4. Relaunch. Review each main model's settings once before its next launch. You
   can import an already-running server before stopping it manually.

Existing configuration files remain readable. Saving writes only the relevant
model or appearance fields while retaining unrelated settings. TOML formatting
and comments may be rewritten. Each individual file is saved atomically.

### Validation and remaining limits

The automated suite covers existing workflows, V1 migration, model tuning, harness discovery, optional capabilities, validation, argument handling, restart detection and theme persistence. The V1.0.2 development build passes 74 automated tests. A Tkinter smoke test also instantiates the Setup Wizard, Preferences and Configure Model windows under a virtual display to catch construction and screen-sizing regressions.

Hardware-specific llama.cpp/GPU performance still needs testing on your computer.
This release fixes the unready-process retry acceptance and adds launch exit
codes; it does not claim to complete every item from the earlier audit. Startup
cancellation through Stop All, durable error-state history and VRAM admission
checks remain follow-up work. Conservative settings reduce assumptions but do
not diagnose the original Qwen crash without its log.

V0.8.0 was the **Setup & Configuration** release. It builds on the V0.7 Prompt Workshop and adds guided first-run configuration plus permanent settings for model discovery, task defaults and Prompt Helper performance.

## What V0.8.0 adds

- First-run **Setup Wizard**.
- `Preferences > Run Setup Wizard...` to rerun guided setup later.
- `Preferences > Settings...` for permanent configuration changes.
- `Preferences > Rescan Models` for newly added GGUF files.
- Editable LLM model roots with recursive model discovery.
- Coding, Chat and Agent default-model selectors populated from discovered models.
- Prompt Helper model selection with automatic lightest-model mode.
- Prompt Helper **CPU / GPU / Auto** processing modes.
- **Keep Prompt Helper loaded after use** option.
- Advanced Prompt Helper controls for GPU layers, context length, port, startup timeout and K/V cache types.
- Prompt Helper settings stored in `settings.toml`; task model defaults remain in `profiles.toml`.
- Atomic configuration writes so a save does not leave partially written TOML files.
- Setup completion tracking so the wizard opens automatically on first V0.8 configuration.
- System panel shows the Prompt Helper model, CPU/GPU decision and readiness state.
- Auto mode uses CPU while another configured GPU workload is active and GPU when the GPU is otherwise free.
- Auto/GPU Prompt Helper is treated as a potential resource blocker before starting a normal GPU workload; explicit CPU mode remains non-conflicting.

The existing Coding, Chat, Agent, Image, Stop All, service controls, model discovery, Prompt Workshop and dependency-aware transitions remain intact.

## Setup Wizard

On a configuration that has not completed V0.8 setup, the wizard opens automatically after the main window appears. It can also be launched manually:

```bash
ai-control-centre setup
```

or from:

```text
Preferences > Run Setup Wizard...
```

The wizard covers:

1. host capability summary
2. LLM model roots and model discovery
3. default models for Coding, Chat and Agent
4. Prompt Helper model policy, processing mode and keep-loaded behaviour
5. review and save

The wizard does not install packages or modify unrelated operating-system configuration.

## Settings

Open:

```text
Preferences > Settings...
```

### Models

Edit one recursive LLM model root per line and choose **Rescan Models**. Newly added GGUF files become available immediately after saving/reloading configuration.

### Task Defaults

Choose a default model for:

```text
Coding
Chat
Agent
```

These choices update `profiles.toml`. The main-window LLM override remains available for one-off task starts.

### Prompt Helper

The normal options are:

```text
Model            [ discovered model ]
[x] Automatically use lightest suitable model

Processing
( ) CPU  - lowest VRAM use
( ) GPU  - faster response
(*) Auto - balance performance and VRAM

[x] Keep Prompt Helper loaded after use
```

When automatic lightest-model selection is enabled, the complete discovered GGUF with the smallest total on-disk size is selected. Split-model shard sizes are summed before comparison.

### Advanced Prompt Helper

Expand **Advanced** to configure:

```text
GPU layers        Auto or integer
Context length    4096 default
Port              8081 default
Startup timeout   60 s default
KV cache K        q8_0 default
KV cache V        q8_0 default
```

**Restore recommended defaults** resets these controls to the V0.8 defaults.

## Prompt Helper processing modes

### CPU

Prompt Helper launches with:

```text
-ngl 0
```

This prioritises VRAM availability for ComfyUI or large task models.

### GPU

Prompt Helper uses the configured GPU-layer value. When GPU layers is `Auto`, the selected model's recommended GPU layers are used when available; otherwise llama.cpp is allowed to offload as many layers as practical.

### Auto

V0.8 uses a deliberately simple rule:

```text
another configured GPU workload active -> CPU
otherwise                             -> GPU
```

The System panel shows the current decision as `CPU` or `GPU` rather than hiding it.

If a Prompt Helper retained in Auto/GPU mode could conflict with a newly requested GPU workload, the existing resource-conflict flow can unload it after user approval.

## Keep loaded behaviour

When enabled, Prompt Helper remains resident after prompt refinement for faster repeated requests.

When disabled, the launcher stops its owned Prompt Helper after the refined prompt is returned.

`Stop All` always stops launcher-owned Prompt Helper regardless of this setting.

## Prompt Workshop

The lower workspace introduced in V0.7 remains:

```text
[ System ]   [ Prompt Workshop ]
```

Prompt Workshop includes:

- rough task input
- Improve Prompt
- Clear Input
- Clear Output
- Clear Both
- editable refined prompt
- Copy
- Send to Agent
- Send to Image

Send actions remain clipboard-first in V0.8; direct Computer task injection and direct ComfyUI workflow submission are still deferred.

## Orson reference task defaults

The bundled Orson reference configuration starts with:

```text
Coding          Qwen3-Coder 30B A3B Instruct Heretic Q4_K_M
Chat            Qwen3 30B A3B Instruct 2507 Q4_K_M
Agent           Qwen3 30B A3B Instruct 2507 Q4_K_M
Prompt Helper   automatic lightest complete model
Processing      Auto
Keep loaded     Yes
```

On the Orson reference library, automatic lightest-model selection should choose SmolLM3-3B Q8_0.

## Upgrade from V0.7.x

Stop launcher-owned services first:

```bash
ai-control-centre stop-all
```

Update the project files, activate the existing virtual environment and reinstall the editable package:

```bash
cd ~/ai/control-centre
source .venv/bin/activate
python -m pip install -e .
```

V0.8 understands V0.7 user configuration. If `[setup] completed = true` is not present, the Setup Wizard opens so the new choices can be reviewed and saved.

You do not need to overwrite `models.toml` to discover a newly added model. Use **Preferences > Rescan Models** or the Models page in Settings.

## Run

```bash
ai-control-centre gui
```

or:

```bash
ai-control-centre-gui
```

Run the setup wizard explicitly with:

```bash
ai-control-centre setup
```

## Optional desktop launcher

After the GUI works correctly from the terminal:

```bash
./installer/install-desktop.sh
```

This creates only the user-level desktop entry. It does not configure AI services to start at login.

## Tests

```bash
PYTHONPATH=src python -m pytest -q
```

The V1.0.2 development build passed 74 automated tests in the development environment.
