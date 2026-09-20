# AI Control Centre V0.6.2

## V0.6.2 task-opening hotfix

Primary task buttons now complete the whole task transition: after all required services are Ready, the configured default interface opens automatically. Coding and Chat open the llama.cpp web dialogue, Agent opens Computer DMZ, and Image opens ComfyUI. The manual service Open buttons remain available for troubleshooting or reopening a closed browser tab.

## V0.6.1 hotfix

- Replaces the LLM override `ttk.Combobox` with an anchored menu button to avoid a Linux/Tk popdown-positioning bug where the model list could appear at the top-left of the desktop.
- No backend, profile, model-selection, or service-management behaviour is changed.


V0.6 is the first usability-focused release of the task-oriented Linux AI control panel.
It keeps the proven V0.5 backend as the authority and improves how profiles, model changes,
resource conflicts and individual service controls are presented in the GUI.

## What V0.6 adds

- Five primary task controls in one row: **Coding**, **Chat**, **Agent**, **Image**, **Stop All**.
- A new **Chat** profile for general conversation and prompt-development work.
- Task-specific default models:
  - Coding: Qwen3-Coder 30B Heretic.
  - Chat: Qwen3 30B A3B Instruct 2507.
  - Agent: Qwen3 30B A3B Instruct 2507.
  - Image: ComfyUI, no LLM required.
- `profiles.toml` accepts the clearer `default_model` key while retaining compatibility with
  the earlier `selected_model` key.
- The GUI model selector is now an optional **LLM override**. Leave it on `Task default` for
  automatic task-model selection.
- Safe model transitions. Switching from a running Coding model to the Chat model, for example,
  asks before stopping affected launcher-owned services and reloading llama.cpp.
- Dependency-aware model transitions. Agent -> Coding stops Computer DMZ before reloading llama.
- Chat -> Agent reuses the already loaded general-purpose model and only starts Computer DMZ.
- Individual **Start** buttons use the same resource-conflict confirmation flow as task buttons.
- Friendlier service status text. Normal stopped services show `Not running` rather than raw
  connection-refused messages.
- llama.cpp shows the currently loaded model in the Services view.
- Clearer `Loading model` feedback while llama.cpp is starting.
- CLI support for `--replace-model` on service and profile starts.
- Optional user-level desktop launcher installer.

V0.6 retains process ownership, Docker ownership, HTTP readiness, dependency ordering, model
selection, split-GGUF discovery, NVIDIA monitoring and resource conflict protection from earlier
releases.

## Orson reference profiles

```toml
[profiles.coding]
display_name = "Coding"
services = ["llama"]
default_model = "qwen3_coder_30b_heretic"

[profiles.chat]
display_name = "Chat"
services = ["llama"]
open_service = "llama"
default_model = "qwen_qwen3_30b_a3b_instruct_2507_q4_k_m"

[profiles.agent]
display_name = "Agent"
services = ["llama", "computer_dmz"]
open_service = "computer_dmz"
default_model = "qwen_qwen3_30b_a3b_instruct_2507_q4_k_m"

[profiles.image]
display_name = "Image"
services = ["comfyui"]
open_service = "comfyui"
```

The Chat and Agent model uses the llama service's current fallback launch tuning until more precise
per-model values are recorded in `models.toml`.

## Upgrade from V0.5

Stop launcher-owned services first:

```bash
ai-control-centre stop-all
```

Copy the V0.6 files over the existing project, then from `~/ai/control-centre`:

```bash
source .venv/bin/activate
python -m pip install -e .
```

Back up and replace the reference profile/model files if you want the new Orson defaults:

```bash
cp ~/.config/ai-control-centre/profiles.toml \
   ~/.config/ai-control-centre/profiles.toml.v0.5.bak
cp ~/.config/ai-control-centre/models.toml \
   ~/.config/ai-control-centre/models.toml.v0.5.bak

cp examples/orson/profiles.toml ~/.config/ai-control-centre/profiles.toml
cp examples/orson/models.toml ~/.config/ai-control-centre/models.toml
```

`services.toml` and `settings.toml` do not need to be replaced for the V0.6 features.

## Run the GUI

```bash
ai-control-centre gui
```

or:

```bash
ai-control-centre-gui
```

Normal use should leave **LLM override** set to `Task default`.

## Expected task transitions

### Coding -> Chat

The GUI detects that llama.cpp is running the Coding model, asks for permission to change the
loaded model, then stops llama safely and starts the Chat model.

### Chat -> Agent

Because Chat and Agent share the same default model, llama.cpp remains running and only Computer
DMZ is started.

### Agent -> Coding

The GUI asks before a model change. On approval it stops Computer DMZ first, then llama.cpp,
loads the Coding model and starts the Coding profile.

### Coding/Chat/Agent -> Image

The existing large-GPU-workload conflict rule applies. The GUI asks before stopping launcher-owned
conflicting services and starting ComfyUI.

### Individual Start buttons

They use the same backend resource and model-transition rules as task buttons. They no longer just
surface the raw conflict error when a safe launcher-owned transition is available.

## CLI model transition

A model change still requires explicit permission:

```bash
ai-control-centre profile start chat --no-open
ai-control-centre profile start coding --no-open
```

The second command refuses while the Chat model is loaded. To approve the reload from the CLI:

```bash
ai-control-centre profile start coding --no-open --replace-model
```

Resource conflicts remain separate:

```bash
ai-control-centre profile start image --no-open --stop-conflicts
```

## Optional desktop launcher

After the GUI works correctly from the terminal, install a user-level `.desktop` entry with:

```bash
./installer/install-desktop.sh
```

This writes only:

```text
~/.local/share/applications/ai-control-centre.desktop
```

It does not configure AI services to start at login.

## Tests

Run:

```bash
PYTHONPATH=src pytest -q
```

V0.6 passes 22 automated tests in the development environment.
