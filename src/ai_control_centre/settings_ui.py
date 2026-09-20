from __future__ import annotations

from pathlib import Path
import tkinter as tk
from tkinter import colorchooser, messagebox, ttk

from .config import AppConfig, load_app_config
from .domain import HarnessConfig
from .harnesses import available_custom_harness_services, discover_agent_harnesses
from .models import model_total_size_bytes
from .preferences import detect_setup_environment, rescan_models, save_preferences, save_appearance
from .theme import DEFAULT_APPEARANCE, style_classic, validate_appearance
from .tuning_ui import ModelTuningPanel, ModelTuningWindow
from .ui_utils import ScrollableFrame, fit_window_to_screen

_TASKS = ("coding", "chat", "agent")
_CACHE_CHOICES = ("q8_0", "q4_0", "f16")


class _PreferenceState:
    def __init__(self, config_dir: Path, config: AppConfig):
        self.config_dir = config_dir
        self.config = config
        self.models = dict(config.models)
        self.roots = list(config.settings.llm_model_roots)

        self.task_models = {
            task: config.profiles.get(task).default_model if config.profiles.get(task) else None
            for task in _TASKS
        }
        agent_profile = config.profiles.get("agent")
        self.agent_harness = agent_profile.harness if agent_profile else None
        self.agent_base_services = tuple(
            sid for sid in (agent_profile.services if agent_profile else ())
            if sid in config.services
            and getattr(config.services[sid], "uses_model", False)
        )
        self.harnesses = dict(config.harnesses)
        prompt = config.settings.prompt_workshop
        self.prompt_enabled = prompt.enabled
        self.prompt_model = prompt.preferred_model
        self.auto_lightest = prompt.model_strategy == "smallest"
        self.processing_mode = prompt.processing_mode
        self.keep_loaded = prompt.keep_loaded
        self.gpu_layers = prompt.gpu_layers
        self.context_length = prompt.context_length
        self.port = prompt.port
        self.startup_timeout = prompt.startup_timeout_seconds
        self.cache_type_k = prompt.cache_type_k
        self.cache_type_v = prompt.cache_type_v

    def scan(self, roots: tuple[Path, ...]) -> None:
        self.roots = list(roots)
        self.models = rescan_models(self.config_dir, roots)

    def save(self) -> None:
        save_preferences(
            self.config_dir,
            model_roots=tuple(self.roots),
            task_models=self.task_models,
            agent_harness=self.agent_harness,
            agent_base_services=self.agent_base_services,
            harnesses=self.harnesses,
            prompt_model=self.prompt_model,
            prompt_enabled=self.prompt_enabled,
            prompt_auto_lightest=self.auto_lightest,
            prompt_processing_mode=self.processing_mode,
            prompt_keep_loaded=self.keep_loaded,
            prompt_gpu_layers=self.gpu_layers,
            prompt_context_length=self.context_length,
            prompt_port=self.port,
            prompt_startup_timeout=self.startup_timeout,
            prompt_cache_type_k=self.cache_type_k,
            prompt_cache_type_v=self.cache_type_v,
            setup_completed=True,
        )


def _model_label(model) -> str:
    size = model_total_size_bytes(model)
    size_text = f"{size / (1024 ** 3):.1f} GB" if size is not None else "size unknown"
    suffix = "" if model.complete else " - incomplete"
    return f"{model.display_name} [{model.id}] - {size_text}{suffix}"


class ModelMenu(ttk.Menubutton):
    def __init__(self, parent, variable: tk.StringVar, **kwargs):
        super().__init__(parent, textvariable=variable, direction="below", **kwargs)
        self.variable = variable
        self.menu = tk.Menu(self, tearoff=False)
        self.configure(menu=self.menu)
        self.label_to_id: dict[str, str] = {}
        self.id_to_label: dict[str, str] = {}

    def set_models(self, models, selected_id: str | None, *, allow_none: bool = False) -> None:
        self.menu.delete(0, "end")
        self.label_to_id.clear()
        self.id_to_label.clear()
        if allow_none:
            self.menu.add_command(label="None", command=lambda: self.variable.set("None"))
        for model_id, model in models.items():
            if not model.complete:
                continue
            label = _model_label(model)
            self.label_to_id[label] = model_id
            self.id_to_label[model_id] = label
            self.menu.add_command(label=label, command=lambda value=label: self.variable.set(value))
        if selected_id in self.id_to_label:
            self.variable.set(self.id_to_label[selected_id])
        elif allow_none:
            self.variable.set("None")
        elif self.id_to_label:
            self.variable.set(next(iter(self.id_to_label.values())))
        else:
            self.variable.set("No complete models found")

    def selected_id(self) -> str | None:
        return self.label_to_id.get(self.variable.get())


class HarnessMenu(ttk.Menubutton):
    def __init__(self, parent, variable: tk.StringVar, **kwargs):
        super().__init__(parent, textvariable=variable, direction="below", **kwargs)
        self.variable = variable
        self.menu = tk.Menu(self, tearoff=False)
        self.configure(menu=self.menu)
        self.label_to_id: dict[str, str] = {}
        self.id_to_label: dict[str, str] = {}

    def set_harnesses(self, harnesses, selected_id: str | None, *, allow_none: bool = False) -> None:
        self.menu.delete(0, "end")
        self.label_to_id.clear()
        self.id_to_label.clear()
        if allow_none:
            self.menu.add_command(label="Not configured", command=lambda: self.variable.set("Not configured"))
        for harness_id, harness in harnesses.items():
            label = harness.display_name
            self.label_to_id[label] = harness_id
            self.id_to_label[harness_id] = label
            self.menu.add_command(label=label, command=lambda value=label: self.variable.set(value))
        if selected_id in self.id_to_label:
            self.variable.set(self.id_to_label[selected_id])
        elif allow_none:
            self.variable.set("Not configured")
        elif self.id_to_label:
            self.variable.set(next(iter(self.id_to_label.values())))
        else:
            self.variable.set("No harnesses configured")

    def selected_id(self) -> str | None:
        return self.label_to_id.get(self.variable.get())


def _harness_slug(text: str) -> str:
    import re
    value = re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")
    return value or "agent_harness"


class CustomHarnessDialog(tk.Toplevel):
    """Create or edit a harness from already configured services."""

    def __init__(self, parent, config: AppConfig, harnesses, on_saved, existing: HarnessConfig | None = None):
        super().__init__(parent)
        self.title("Edit Agent Harness" if existing else "Add Custom Agent Harness")
        self.transient(parent)
        self.grab_set()
        self.config_data = config
        self.existing = existing
        self.existing_ids = set(harnesses)
        self.on_saved = on_saved
        prompt_service = config.settings.prompt_workshop.service
        self.services = available_custom_harness_services(config.services, prompt_helper_service=prompt_service)
        self.name_var = tk.StringVar(value=existing.display_name if existing else "")
        self.service_vars = {
            sid: tk.BooleanVar(value=bool(existing and sid in existing.services))
            for sid in self.services
        }
        self.open_var = tk.StringVar(value="None")
        self.open_label_to_id: dict[str, str] = {}

        outer = ttk.Frame(self, padding=16)
        outer.pack(fill="both", expand=True)
        outer.columnconfigure(1, weight=1)
        ttk.Label(outer, text="Harness name").grid(row=0, column=0, sticky="w", padx=(0, 12), pady=6)
        ttk.Entry(outer, textvariable=self.name_var).grid(row=0, column=1, sticky="ew", pady=6)
        ttk.Label(outer, text="Services", font="AppHeading").grid(row=1, column=0, columnspan=2, sticky="w", pady=(12, 6))

        services_scroll = ScrollableFrame(outer)
        services_scroll.grid(row=2, column=0, columnspan=2, sticky="nsew")
        services_frame = services_scroll.body
        outer.rowconfigure(2, weight=1)
        if not self.services:
            ttk.Label(
                services_frame,
                text="No suitable configured services are available. Configure an agent runtime service first, then scan again.",
                style="Muted.TLabel",
                wraplength=600,
            ).pack(anchor="w", pady=10)
        else:
            for sid, service in self.services.items():
                ttk.Checkbutton(
                    services_frame,
                    text=f"{service.display_name}  [{sid}]",
                    variable=self.service_vars[sid],
                    command=self._refresh_open_menu,
                ).pack(anchor="w", pady=3)

        ttk.Label(outer, text="Open when ready").grid(row=3, column=0, sticky="w", padx=(0, 12), pady=(12, 6))
        self.open_button = ttk.Menubutton(outer, textvariable=self.open_var, direction="below")
        self.open_menu = tk.Menu(self.open_button, tearoff=False)
        self.open_button.configure(menu=self.open_menu)
        self.open_button.grid(row=3, column=1, sticky="ew", pady=(12, 6))
        self._refresh_open_menu(existing.open_service if existing else None)

        actions = ttk.Frame(outer)
        actions.grid(row=4, column=0, columnspan=2, sticky="ew", pady=(16, 0))
        actions.columnconfigure(0, weight=1)
        ttk.Button(actions, text="Cancel", command=self.destroy).grid(row=0, column=1, padx=4)
        ttk.Button(actions, text="Save harness", style="Primary.TButton", command=self._save).grid(row=0, column=2, padx=(4, 0))
        style_classic(self)
        fit_window_to_screen(self, preferred_width=720, preferred_height=560, min_width=620, min_height=420)

    def _selected_services(self) -> tuple[str, ...]:
        return tuple(sid for sid, var in self.service_vars.items() if var.get())

    def _refresh_open_menu(self, selected_id: str | None = None):
        selected_id = selected_id or self.open_label_to_id.get(self.open_var.get())
        self.open_menu.delete(0, "end")
        self.open_label_to_id = {"None": ""}
        self.open_menu.add_command(label="None", command=lambda: self.open_var.set("None"))
        for sid in self._selected_services():
            service = self.services[sid]
            label = service.display_name
            self.open_label_to_id[label] = sid
            self.open_menu.add_command(label=label, command=lambda v=label: self.open_var.set(v))
        if selected_id:
            service = self.services.get(selected_id)
            if service and selected_id in self._selected_services():
                self.open_var.set(service.display_name)
                return
        self.open_var.set("None")

    def _save(self):
        name = self.name_var.get().strip()
        services = self._selected_services()
        if not name:
            messagebox.showerror("Agent Harness", "Enter a harness name.", parent=self)
            return
        if not services:
            messagebox.showerror("Agent Harness", "Select at least one service.", parent=self)
            return
        if self.existing:
            harness_id = self.existing.id
        else:
            base = _harness_slug(name)
            harness_id = base
            suffix = 2
            while harness_id in self.existing_ids:
                harness_id = f"{base}_{suffix}"
                suffix += 1
        open_service = self.open_label_to_id.get(self.open_var.get()) or None
        harness = HarnessConfig(
            id=harness_id,
            display_name=name,
            services=services,
            open_service=open_service,
        )
        self.on_saved(harness)
        self.destroy()


class SettingsWindow(tk.Toplevel):
    def __init__(self, parent, config_dir: Path, config: AppConfig, on_saved, *, initial_tab=None, model_id=None):
        super().__init__(parent)
        self.title("AI Control Centre Settings")
        self.transient(parent)
        self.config_dir = config_dir
        self.state_data = _PreferenceState(config_dir, config)
        self.on_saved = on_saved
        self.model_vars: dict[str, tk.StringVar] = {}
        self.model_menus: dict[str, ModelMenu] = {}
        self.harness_var = tk.StringVar()
        self.harness_menu: HarnessMenu | None = None
        self.prompt_model_var = tk.StringVar()
        self.root_var = tk.StringVar(value="\n".join(str(p) for p in self.state_data.roots))
        self.scan_status = tk.StringVar(value=f"{len(self.state_data.models)} model(s) discovered")
        self.prompt_enabled_var = tk.BooleanVar(value=self.state_data.prompt_enabled)
        self.auto_var = tk.BooleanVar(value=self.state_data.auto_lightest)
        self.mode_var = tk.StringVar(value=self.state_data.processing_mode)
        self.keep_var = tk.BooleanVar(value=self.state_data.keep_loaded)
        self.gpu_layers_var = tk.StringVar(value="Auto" if self.state_data.gpu_layers is None else str(self.state_data.gpu_layers))
        self.context_var = tk.StringVar(value=str(self.state_data.context_length))
        self.port_var = tk.StringVar(value=str(self.state_data.port))
        self.timeout_var = tk.StringVar(value=str(self.state_data.startup_timeout))
        self.cache_k_var = tk.StringVar(value=self.state_data.cache_type_k)
        self.cache_v_var = tk.StringVar(value=self.state_data.cache_type_v)
        self.advanced_visible = False
        self.appearance_vars = {key: tk.StringVar(self, value=str(value))
                                for key, value in {**DEFAULT_APPEARANCE, **config.settings.appearance}.items()}
        self._build()
        if model_id in self.tuning.models:
            self.tuning.select_model(model_id, force=True)
        if initial_tab == 'tuning':
            self.notebook.select(self.tuning_scroll)
        style_classic(self)
        fit_window_to_screen(self, preferred_width=1040, preferred_height=940, min_width=820, min_height=620, margin_y=110)
        self.protocol('WM_DELETE_WINDOW', self._cancel)

    def _build(self) -> None:
        outer = ttk.Frame(self, padding=12)
        outer.pack(fill="both", expand=True)
        outer.rowconfigure(0, weight=1)
        outer.columnconfigure(0, weight=1)
        self.notebook = ttk.Notebook(outer)
        self.notebook.grid(row=0, column=0, sticky="nsew")
        models = ttk.Frame(self.notebook, padding=12)
        tasks = ttk.Frame(self.notebook, padding=12)
        harnesses = ttk.Frame(self.notebook, padding=12)
        prompt = ttk.Frame(self.notebook, padding=12)
        self.notebook.add(models, text="Models")
        self.notebook.add(tasks, text="Task Defaults")
        self.notebook.add(harnesses, text="Agent Harnesses")
        self.notebook.add(prompt, text="Prompt Helper")
        self.tuning_scroll = ScrollableFrame(self.notebook)
        self.tuning = ModelTuningPanel(self.tuning_scroll.body, self.config_dir, self.state_data.config,
                                      self.on_saved, self._launch_tuned)
        self.tuning.pack(fill='both', expand=True)
        self.notebook.add(self.tuning_scroll, text='Model tuning')
        appearance = ttk.Frame(self.notebook, padding=16)
        self.notebook.add(appearance, text='Appearance')
        self._build_appearance(appearance)
        self._build_models(models)
        self._build_tasks(tasks)
        self._build_harnesses(harnesses)
        self._build_prompt(prompt)

        buttons = ttk.Frame(outer)
        buttons.grid(row=1, column=0, sticky="ew", pady=(12, 0))
        buttons.columnconfigure(0, weight=1)
        ttk.Button(buttons, text="Close", command=self._cancel).grid(row=0, column=1, padx=4)
        ttk.Button(buttons, text="Save settings", style="Primary.TButton", command=self._save).grid(row=0, column=2, padx=(4, 0))

    def _build_models(self, parent) -> None:
        parent.columnconfigure(0, weight=1)
        ttk.Label(parent, text="LLM model roots", font=("TkDefaultFont", 11, "bold")).grid(row=0, column=0, sticky="w")
        ttk.Label(parent, text="One directory per line. Model discovery is recursive.").grid(row=1, column=0, sticky="w", pady=(2, 6))
        self.root_text = tk.Text(parent, height=5, wrap="none")
        self.root_text.grid(row=2, column=0, sticky="ew")
        self.root_text.insert("1.0", self.root_var.get())
        row = ttk.Frame(parent)
        row.grid(row=3, column=0, sticky="ew", pady=8)
        ttk.Button(row, text="Rescan Models", command=self._rescan).pack(side="left")
        ttk.Label(row, textvariable=self.scan_status).pack(side="left", padx=10)
        self.model_list = tk.Listbox(parent, height=16)
        self.model_list.grid(row=4, column=0, sticky="nsew", pady=(4, 0))
        parent.rowconfigure(4, weight=1)
        self._refresh_model_list()

    def _build_tasks(self, parent) -> None:
        parent.columnconfigure(1, weight=1)
        ttk.Label(parent, text="Default model for each task", font=("TkDefaultFont", 11, "bold")).grid(row=0, column=0, columnspan=2, sticky="w", pady=(0, 12))
        labels = {"coding": "Coding", "chat": "Chat", "agent": "Agent"}
        for row, task in enumerate(_TASKS, start=1):
            ttk.Label(parent, text=labels[task]).grid(row=row, column=0, sticky="w", pady=6, padx=(0, 12))
            variable = tk.StringVar()
            menu = ModelMenu(parent, variable)
            menu.grid(row=row, column=1, sticky="ew", pady=6)
            self.model_vars[task] = variable
            self.model_menus[task] = menu
        ttk.Separator(parent).grid(row=4, column=0, columnspan=2, sticky="ew", pady=12)
        ttk.Label(parent, text="Agent harness").grid(row=5, column=0, sticky="w", pady=6, padx=(0, 12))
        self.harness_menu = HarnessMenu(parent, self.harness_var)
        self.harness_menu.grid(row=5, column=1, sticky="ew", pady=6)
        self.harness_menu.set_harnesses(self.state_data.harnesses, self.state_data.agent_harness, allow_none=True)
        ttk.Label(parent, text="The harness supplies the agent runtime/interface; the Agent task still uses the selected LLM.", style="Muted.TLabel", wraplength=620).grid(row=6, column=0, columnspan=2, sticky="w", pady=(2, 0))
        self._refresh_model_menus()

    def _build_harnesses(self, parent) -> None:
        parent.columnconfigure(0, weight=1)
        parent.rowconfigure(2, weight=1)
        ttk.Label(parent, text="Agent Harnesses", font=("TkDefaultFont", 11, "bold")).grid(row=0, column=0, sticky="w")
        ttk.Label(
            parent,
            text="Scan configured services for recognised agent runtimes, or build a custom harness from services already known to AI Control Centre.",
            style="Muted.TLabel",
            wraplength=720,
        ).grid(row=1, column=0, sticky="w", pady=(4, 10))
        self.harness_list = tk.Listbox(parent, height=12, exportselection=False)
        self.harness_list.grid(row=2, column=0, sticky="nsew")
        controls = ttk.Frame(parent)
        controls.grid(row=3, column=0, sticky="ew", pady=10)
        ttk.Button(controls, text="Scan for Harnesses", command=self._scan_harnesses).pack(side="left")
        ttk.Button(controls, text="Add Custom Harness", command=self._add_custom_harness).pack(side="left", padx=8)
        ttk.Button(controls, text="Edit", command=self._edit_selected_harness).pack(side="left")
        ttk.Button(controls, text="Remove", command=self._remove_selected_harness).pack(side="left", padx=8)
        self._refresh_harness_list()

    def _refresh_harness_list(self) -> None:
        if not hasattr(self, "harness_list"):
            return
        self.harness_list.delete(0, "end")
        self._harness_ids = []
        for harness_id, harness in self.state_data.harnesses.items():
            services = ", ".join(harness.services)
            label = f"{harness.display_name}  [{harness_id}]  -  {services}"
            self.harness_list.insert("end", label)
            self._harness_ids.append(harness_id)
        if self._harness_ids:
            self.harness_list.selection_set(0)

    def _refresh_harness_menu(self) -> None:
        if self.harness_menu is None:
            return
        selected = self.harness_menu.selected_id() or self.state_data.agent_harness
        self.harness_menu.set_harnesses(self.state_data.harnesses, selected, allow_none=True)

    def _scan_harnesses(self) -> None:
        discovered = discover_agent_harnesses(self.state_data.config.services, self.state_data.harnesses)
        added = [hid for hid in discovered if hid not in self.state_data.harnesses]
        self.state_data.harnesses = discovered
        self._refresh_harness_list()
        self._refresh_harness_menu()
        if added:
            messagebox.showinfo("Agent Harnesses", f"Found {len(added)} new harness suggestion(s). Review and save settings to keep them.", parent=self)
        else:
            messagebox.showinfo("Agent Harnesses", "No new recognised harnesses were found. You can add a custom harness from configured services.", parent=self)

    def _store_harness(self, harness: HarnessConfig) -> None:
        self.state_data.harnesses[harness.id] = harness
        if self.state_data.agent_harness is None:
            self.state_data.agent_harness = harness.id
        self._refresh_harness_list()
        self._refresh_harness_menu()

    def _add_custom_harness(self) -> None:
        CustomHarnessDialog(self, self.state_data.config, self.state_data.harnesses, self._store_harness)

    def _selected_harness_id(self) -> str | None:
        if not hasattr(self, "harness_list"):
            return None
        selection = self.harness_list.curselection()
        if not selection:
            return None
        index = selection[0]
        return self._harness_ids[index] if index < len(self._harness_ids) else None

    def _edit_selected_harness(self) -> None:
        harness_id = self._selected_harness_id()
        if harness_id is None:
            return
        CustomHarnessDialog(
            self, self.state_data.config, self.state_data.harnesses,
            self._store_harness, self.state_data.harnesses[harness_id],
        )

    def _remove_selected_harness(self) -> None:
        harness_id = self._selected_harness_id()
        if harness_id is None:
            return
        harness = self.state_data.harnesses[harness_id]
        if not messagebox.askyesno("Remove Harness", f"Remove '{harness.display_name}' from AI Control Centre configuration?", parent=self):
            return
        self.state_data.harnesses.pop(harness_id, None)
        if self.state_data.agent_harness == harness_id:
            self.state_data.agent_harness = None
        self._refresh_harness_list()
        self._refresh_harness_menu()

    def _build_prompt(self, parent) -> None:
        parent.columnconfigure(1, weight=1)
        row = 0
        ttk.Label(parent, text="Prompt Helper", font=("TkDefaultFont", 11, "bold")).grid(row=row, column=0, columnspan=2, sticky="w", pady=(0, 12)); row += 1
        ttk.Checkbutton(parent, text="Enable Prompt Helper", variable=self.prompt_enabled_var, command=self._update_prompt_controls).grid(row=row, column=0, columnspan=2, sticky="w", pady=(0, 8)); row += 1
        ttk.Label(parent, text="Model").grid(row=row, column=0, sticky="w", pady=5)
        self.prompt_model_menu = ModelMenu(parent, self.prompt_model_var)
        self.prompt_model_menu.grid(row=row, column=1, sticky="ew", pady=5); row += 1
        ttk.Checkbutton(parent, text="Automatically use lightest suitable model", variable=self.auto_var).grid(row=row, column=0, columnspan=2, sticky="w", pady=5); row += 1
        ttk.Label(parent, text="Processing").grid(row=row, column=0, sticky="nw", pady=5)
        modes = ttk.Frame(parent); modes.grid(row=row, column=1, sticky="w", pady=5)
        for text, value in (("CPU - lowest VRAM use", "cpu"), ("GPU - faster response", "gpu"), ("Auto - balance performance and VRAM", "auto")):
            ttk.Radiobutton(modes, text=text, value=value, variable=self.mode_var).pack(anchor="w")
        row += 1
        ttk.Checkbutton(parent, text="Keep Prompt Helper loaded after use", variable=self.keep_var).grid(row=row, column=0, columnspan=2, sticky="w", pady=5); row += 1
        self.advanced_button = ttk.Button(parent, text="Advanced >", command=self._toggle_advanced)
        self.advanced_button.grid(row=row, column=0, columnspan=2, sticky="w", pady=(10, 5)); row += 1
        self.advanced = ttk.LabelFrame(parent, text="Advanced Prompt Helper", padding=10)
        self.advanced.grid(row=row, column=0, columnspan=2, sticky="ew")
        self.advanced.columnconfigure(1, weight=1)
        fields = (
            ("GPU layers", self.gpu_layers_var),
            ("Context length", self.context_var),
            ("Port", self.port_var),
            ("Startup timeout (s)", self.timeout_var),
        )
        for i, (label, var) in enumerate(fields):
            ttk.Label(self.advanced, text=label).grid(row=i, column=0, sticky="w", pady=4, padx=(0, 10))
            ttk.Entry(self.advanced, textvariable=var).grid(row=i, column=1, sticky="ew", pady=4)
        ttk.Label(self.advanced, text="KV cache K").grid(row=4, column=0, sticky="w", pady=4)
        ttk.OptionMenu(self.advanced, self.cache_k_var, self.cache_k_var.get(), *_CACHE_CHOICES).grid(row=4, column=1, sticky="ew", pady=4)
        ttk.Label(self.advanced, text="KV cache V").grid(row=5, column=0, sticky="w", pady=4)
        ttk.OptionMenu(self.advanced, self.cache_v_var, self.cache_v_var.get(), *_CACHE_CHOICES).grid(row=5, column=1, sticky="ew", pady=4)
        ttk.Button(self.advanced, text="Restore recommended defaults", command=self._restore_prompt_defaults).grid(row=6, column=0, columnspan=2, sticky="w", pady=(8, 0))
        self.advanced.grid_remove()
        self._refresh_prompt_menu()
        self._update_prompt_controls()

    def _update_prompt_controls(self) -> None:
        state = "normal" if self.prompt_enabled_var.get() else "disabled"
        self.prompt_model_menu.configure(state=state)
        self.advanced_button.configure(state=state)
        if not self.prompt_enabled_var.get():
            self.advanced.grid_remove()
            self.advanced_visible = False
            self.advanced_button.configure(text="Advanced >")

    def _toggle_advanced(self) -> None:
        self.advanced_visible = not self.advanced_visible
        if self.advanced_visible:
            self.advanced.grid()
            self.advanced_button.configure(text="Advanced v")
        else:
            self.advanced.grid_remove()
            self.advanced_button.configure(text="Advanced >")

    def _restore_prompt_defaults(self) -> None:
        self.gpu_layers_var.set("Auto")
        self.context_var.set("4096")
        self.port_var.set("8081")
        self.timeout_var.set("60")
        self.cache_k_var.set("q8_0")
        self.cache_v_var.set("q8_0")

    def _roots_from_text(self) -> tuple[Path, ...]:
        values = [line.strip() for line in self.root_text.get("1.0", "end-1c").splitlines() if line.strip()]
        if not values:
            raise ValueError("At least one model root is required")
        return tuple(Path(value).expanduser() for value in values)

    def _rescan(self) -> None:
        if self.tuning.dirty and not messagebox.askyesno('Unsaved model settings', 'Rescanning will discard unsaved model changes. Continue?', parent=self):
            return
        try:
            roots = self._roots_from_text()
            self.state_data.scan(roots)
        except Exception as exc:
            messagebox.showerror("Rescan Models", str(exc), parent=self)
            return
        self.scan_status.set(f"{len(self.state_data.models)} model(s) discovered")
        self._refresh_model_list()
        self._refresh_model_menus()
        self._refresh_prompt_menu()
        self.tuning.set_models(self.state_data.models)

    def _refresh_model_list(self) -> None:
        self.model_list.delete(0, "end")
        for model in self.state_data.models.values():
            self.model_list.insert("end", _model_label(model))

    def _refresh_model_menus(self) -> None:
        for task, menu in self.model_menus.items():
            selected = menu.selected_id() or self.state_data.task_models.get(task)
            menu.set_models(self.state_data.models, selected, allow_none=(task == "agent"))

    def _refresh_prompt_menu(self) -> None:
        selected = self.prompt_model_menu.selected_id() if hasattr(self, "prompt_model_menu") else None
        if selected is None:
            selected = self.state_data.prompt_model
        self.prompt_model_menu.set_models(self.state_data.models, selected, allow_none=True)

    def _collect(self) -> None:
        self.state_data.roots = list(self._roots_from_text())
        harness_id = self.harness_menu.selected_id() if self.harness_menu is not None else None
        self.state_data.agent_harness = harness_id
        selected_task_models = {}
        for task, menu in self.model_menus.items():
            model_id = menu.selected_id()
            if task == "agent" and harness_id is None:
                self.state_data.task_models[task] = None
                continue
            if model_id is not None:
                selected_task_models[task] = model_id
                self.state_data.task_models[task] = model_id
            else:
                self.state_data.task_models[task] = None
        fresh = load_app_config(self.config_dir)
        unconfigured = [
            fresh.models[mid].display_name if mid in fresh.models else mid
            for mid in dict.fromkeys(selected_task_models.values())
            if mid not in fresh.models or not fresh.models[mid].tuning_reviewed
        ]
        if unconfigured:
            raise ValueError(
                "Configure selected task model(s) under Model tuning before saving: "
                + ", ".join(unconfigured)
            )
        self.state_data.prompt_enabled = self.prompt_enabled_var.get()
        self.state_data.prompt_model = self.prompt_model_menu.selected_id()
        self.state_data.auto_lightest = self.auto_var.get()
        if self.state_data.prompt_enabled and not self.state_data.auto_lightest and self.state_data.prompt_model is None:
            raise ValueError("Choose a Prompt Helper model or enable automatic lightest-model selection")
        self.state_data.processing_mode = self.mode_var.get()
        self.state_data.keep_loaded = self.keep_var.get()
        layers_text = self.gpu_layers_var.get().strip().lower()
        self.state_data.gpu_layers = None if layers_text in {"", "auto"} else int(layers_text)
        if self.state_data.gpu_layers is not None and self.state_data.gpu_layers < 0:
            raise ValueError("GPU layers must be Auto or zero/greater")
        self.state_data.context_length = int(self.context_var.get())
        self.state_data.port = int(self.port_var.get())
        self.state_data.startup_timeout = float(self.timeout_var.get())
        self.state_data.cache_type_k = self.cache_k_var.get().strip()
        self.state_data.cache_type_v = self.cache_v_var.get().strip()
        if self.state_data.context_length < 512:
            raise ValueError("Context length must be at least 512")
        if not 1 <= self.state_data.port <= 65535:
            raise ValueError("Port must be between 1 and 65535")
        if self.state_data.startup_timeout <= 0:
            raise ValueError("Startup timeout must be positive")

    def _save(self) -> None:
        if getattr(self._root(), 'operation_busy', False) or getattr(self._root(), 'prompt_busy', False):
            messagebox.showinfo('Settings', 'Wait for the current operation to finish before saving.', parent=self)
            return
        try:
            appearance = self._appearance_values()
            self._collect()
            if self.tuning.dirty and not self.tuning.save():
                return
            self.state_data.save()
            save_appearance(self.config_dir, appearance)
        except Exception as exc:
            messagebox.showerror("Settings", str(exc), parent=self)
            return
        self.on_saved()
        self.destroy()


    def _cancel(self):
        if self.tuning.dirty and not messagebox.askyesno('Unsaved model settings', 'Discard unsaved model changes and close?', parent=self):
            return
        self.destroy()

    def _launch_tuned(self, model_id):
        root = self._root()
        self.destroy()
        root.after_idle(lambda: root._launch_tuned_model(model_id))

    def _appearance_values(self):
        values = {key: var.get() for key, var in self.appearance_vars.items()}
        values['font_size'] = int(values['font_size'])
        return validate_appearance(values)

    def _save_appearance(self):
        if getattr(self._root(), 'operation_busy', False) or getattr(self._root(), 'prompt_busy', False):
            messagebox.showinfo('Appearance', 'Wait for the current operation to finish before saving.', parent=self)
            return
        try:
            save_appearance(self.config_dir, self._appearance_values())
            self.on_saved()
        except Exception as exc:
            messagebox.showerror('Appearance', str(exc), parent=self)

    def _build_appearance(self, frame):
        frame.columnconfigure(1, weight=1)
        ttk.Label(frame, text='gbhobbytech', font='AppHeading').grid(row=0, column=0, columnspan=3, sticky='w')
        ttk.Label(frame, text='Default deep blue and copper palette with graphite text and clean panel surfaces.', style='Muted.TLabel').grid(row=1, column=0, columnspan=3, sticky='w', pady=(4, 18))
        ttk.Label(frame, text='Appearance').grid(row=2, column=0, sticky='w', padx=(0, 16))
        ttk.OptionMenu(frame, self.appearance_vars['mode'], self.appearance_vars['mode'].get(), 'light', 'dark').grid(row=2, column=1, sticky='w', pady=8)
        for row, key, title in ((3, 'primary', 'Primary colour'), (4, 'accent', 'Accent colour')):
            ttk.Label(frame, text=title).grid(row=row, column=0, sticky='w')
            ttk.Entry(frame, textvariable=self.appearance_vars[key]).grid(row=row, column=1, sticky='ew', pady=8)
            ttk.Button(frame, text='Choose…', command=lambda k=key: self._pick_colour(k)).grid(row=row, column=2, padx=10)
        ttk.Label(frame, text='Text size (points)').grid(row=5, column=0, sticky='w')
        ttk.Spinbox(frame, from_=9, to=14, textvariable=self.appearance_vars['font_size'], width=8).grid(row=5, column=1, sticky='w', pady=8)
        ttk.Label(frame, text='Aptos is used when installed; otherwise the app uses an available Linux font.\nColour changes apply across the main window, settings and text editors.', wraplength=720, style='Muted.TLabel').grid(row=6, column=0, columnspan=3, sticky='w', pady=18)
        controls = ttk.Frame(frame)
        controls.grid(row=7, column=0, columnspan=3, sticky='w')
        ttk.Button(controls, text='Apply and save appearance', style='Primary.TButton', command=self._save_appearance).pack(side='left')
        ttk.Button(controls, text='Restore defaults', command=self._reset_appearance).pack(side='left', padx=10)

    def _pick_colour(self, key):
        try:
            current = self.appearance_vars[key].get()
            chosen = colorchooser.askcolor(current, parent=self, title='Choose colour')[1]
        except tk.TclError:
            chosen = colorchooser.askcolor(parent=self, title='Choose colour')[1]
        if chosen:
            self.appearance_vars[key].set(chosen)

    def _reset_appearance(self):
        for key, value in DEFAULT_APPEARANCE.items():
            self.appearance_vars[key].set(str(value))

class SetupWizard(tk.Toplevel):
    """Small first-run wizard. Advanced tuning remains in the Settings window."""

    def __init__(self, parent, config_dir: Path, config: AppConfig, on_saved):
        super().__init__(parent)
        self.title("AI Control Centre Setup Wizard")
        self.transient(parent)
        self.grab_set()
        self.state_data = _PreferenceState(config_dir, config)
        if not self.state_data.harnesses:
            self.state_data.harnesses = discover_agent_harnesses(config.services, self.state_data.harnesses)
        self.on_saved = on_saved
        self.page = 0
        self.pages: list[ttk.Frame] = []
        self.root_var = tk.StringVar(value="\n".join(str(p) for p in self.state_data.roots))
        self.task_vars = {task: tk.StringVar() for task in _TASKS}
        self.task_menus: dict[str, ModelMenu] = {}
        self.harness_var = tk.StringVar()
        self.harness_menu: HarnessMenu | None = None
        self.agent_enabled_var = tk.BooleanVar(value=bool(self.state_data.agent_harness and self.state_data.harnesses))
        self.harness_status_var = tk.StringVar(value="")
        self.prompt_var = tk.StringVar()
        self.prompt_enabled_var = tk.BooleanVar(value=self.state_data.prompt_enabled)
        self.auto_var = tk.BooleanVar(value=self.state_data.auto_lightest)
        self.mode_var = tk.StringVar(value=self.state_data.processing_mode)
        self.keep_var = tk.BooleanVar(value=self.state_data.keep_loaded)
        self.status_var = tk.StringVar(value="")
        self._build()
        self._show_page(0)
        style_classic(self)
        fit_window_to_screen(self, preferred_width=900, preferred_height=760, min_width=760, min_height=560, margin_y=110)
        self.protocol("WM_DELETE_WINDOW", self._close_setup)

    def _close_setup(self) -> None:
        if messagebox.askyesno("Setup Wizard", "Exit setup? Unsaved setup choices will be lost.", parent=self):
            self.destroy()

    def _build(self) -> None:
        outer = ttk.Frame(self, padding=16)
        outer.pack(fill="both", expand=True)
        outer.rowconfigure(1, weight=1)
        outer.columnconfigure(0, weight=1)
        ttk.Label(outer, text="AI Control Centre Setup", font=("TkDefaultFont", 16, "bold")).grid(row=0, column=0, sticky="w", pady=(0, 12))
        host = ttk.Frame(outer); models = ttk.Frame(outer); tasks = ttk.Frame(outer); configure = ttk.Frame(outer); prompt = ttk.Frame(outer); review = ttk.Frame(outer)
        self.pages = [host, models, tasks, configure, prompt, review]
        for frame in self.pages:
            frame.grid(row=1, column=0, sticky="nsew")
        self._page_host(host)
        self._page_models(models)
        self._page_tasks(tasks)
        self._page_model_configuration(configure)
        self._page_prompt(prompt)
        self._page_review(review)
        nav = ttk.Frame(outer); nav.grid(row=2, column=0, sticky="ew", pady=(12, 0)); nav.columnconfigure(1, weight=1)
        self.back = ttk.Button(nav, text="Back", command=self._back); self.back.grid(row=0, column=0)
        ttk.Label(nav, textvariable=self.status_var).grid(row=0, column=1)
        self.next = ttk.Button(nav, text="Next", command=self._next); self.next.grid(row=0, column=2)

    def _page_host(self, p) -> None:
        d = detect_setup_environment()
        ttk.Label(p, text="System detection", font=("TkDefaultFont", 12, "bold")).pack(anchor="w")
        for label, value in (("Operating system", d.os_name), ("Python", d.python_version), ("Docker", "Available" if d.docker_available else "Not found"), ("NVIDIA monitoring", "Available" if d.nvidia_smi_available else "Not found")):
            row=ttk.Frame(p); row.pack(fill="x", pady=5); ttk.Label(row,text=label,width=22).pack(side="left"); ttk.Label(row,text=value).pack(side="left")
        ttk.Label(p, text="The wizard does not install or alter system packages. It only writes AI Control Centre user configuration.", wraplength=650).pack(anchor="w", pady=(18,0))

    def _page_models(self, p) -> None:
        p.columnconfigure(0, weight=1); p.rowconfigure(3, weight=1)
        ttk.Label(p, text="Model discovery", font=("TkDefaultFont", 12, "bold")).grid(row=0,column=0,sticky="w")
        ttk.Label(p, text="Enter one LLM model root per line.").grid(row=1,column=0,sticky="w",pady=(4,6))
        self.root_text=tk.Text(p,height=5); self.root_text.grid(row=2,column=0,sticky="ew"); self.root_text.insert("1.0",self.root_var.get())
        self.model_list=tk.Listbox(p,height=12); self.model_list.grid(row=3,column=0,sticky="nsew",pady=8)
        ttk.Button(p,text="Rescan Models",command=self._scan).grid(row=4,column=0,sticky="w")
        self._fill_model_list()

    def _page_tasks(self, p) -> None:
        p.columnconfigure(1, weight=1)
        ttk.Label(p,text="Task model defaults",font=("TkDefaultFont",12,"bold")).grid(row=0,column=0,columnspan=3,sticky="w",pady=(0,10))

        # Coding and Chat are core LLM tasks.
        for row,task in enumerate(("coding", "chat"),start=1):
            ttk.Label(p,text=task.title()).grid(row=row,column=0,sticky="w",pady=6,padx=(0,12))
            menu=ModelMenu(p,self.task_vars[task]); menu.grid(row=row,column=1,columnspan=2,sticky="ew",pady=6); self.task_menus[task]=menu
            menu.set_models(self.state_data.models,self.state_data.task_models.get(task))

        ttk.Separator(p).grid(row=3,column=0,columnspan=3,sticky="ew",pady=12)
        ttk.Checkbutton(
            p, text="Configure Agent on this computer", variable=self.agent_enabled_var,
            command=self._update_agent_setup_controls,
        ).grid(row=4,column=0,columnspan=3,sticky="w",pady=(0,8))

        ttk.Label(p,text="Agent model").grid(row=5,column=0,sticky="w",pady=6,padx=(0,12))
        agent_menu=ModelMenu(p,self.task_vars["agent"]); agent_menu.grid(row=5,column=1,columnspan=2,sticky="ew",pady=6); self.task_menus["agent"]=agent_menu
        agent_menu.set_models(self.state_data.models,self.state_data.task_models.get("agent"))

        ttk.Label(p,text="Agent harness").grid(row=6,column=0,sticky="w",pady=6,padx=(0,12))
        self.harness_menu=HarnessMenu(p,self.harness_var); self.harness_menu.grid(row=6,column=1,columnspan=2,sticky="ew",pady=6)
        self.harness_menu.set_harnesses(self.state_data.harnesses,self.state_data.agent_harness)

        harness_actions=ttk.Frame(p); harness_actions.grid(row=7,column=1,columnspan=2,sticky="w",pady=(4,4))
        ttk.Button(harness_actions,text="Scan for Harnesses",command=self._setup_scan_harnesses).pack(side="left")
        ttk.Button(harness_actions,text="Add Custom Harness",command=self._setup_add_custom_harness).pack(side="left",padx=8)
        ttk.Label(p,textvariable=self.harness_status_var,style="Muted.TLabel",wraplength=650).grid(row=8,column=0,columnspan=3,sticky="w",pady=(2,0))
        ttk.Label(p,text="The harness supplies the agent runtime/interface; the Agent task still uses the selected LLM. You can skip Agent and configure it later under Preferences.",style="Muted.TLabel",wraplength=650).grid(row=9,column=0,columnspan=3,sticky="w",pady=(8,0))
        self._update_agent_setup_controls()

    def _update_agent_setup_controls(self) -> None:
        enabled = self.agent_enabled_var.get()
        state = "normal" if enabled else "disabled"
        if "agent" in self.task_menus:
            self.task_menus["agent"].configure(state=state)
        if self.harness_menu is not None:
            self.harness_menu.configure(state=state)
        if not enabled:
            self.harness_status_var.set("Agent will be unavailable until a harness is configured.")
        elif not self.state_data.harnesses:
            self.harness_status_var.set("No harness is configured. Scan for a recognised runtime or add a custom harness.")
        else:
            self.harness_status_var.set(f"{len(self.state_data.harnesses)} harness(es) available.")

    def _setup_scan_harnesses(self) -> None:
        discovered = discover_agent_harnesses(self.state_data.config.services, self.state_data.harnesses)
        added = [hid for hid in discovered if hid not in self.state_data.harnesses]
        self.state_data.harnesses = discovered
        selected = self.harness_menu.selected_id() if self.harness_menu is not None else self.state_data.agent_harness
        if self.harness_menu is not None:
            self.harness_menu.set_harnesses(self.state_data.harnesses, selected)
        if self.state_data.harnesses:
            self.agent_enabled_var.set(True)
        self.harness_status_var.set(
            f"Found {len(added)} new harness suggestion(s)." if added
            else ("No new recognised harnesses found. Add a custom harness if needed." if not self.state_data.harnesses else f"{len(self.state_data.harnesses)} harness(es) available; no new suggestions found.")
        )
        self._update_agent_setup_controls()

    def _setup_store_harness(self, harness: HarnessConfig) -> None:
        self.state_data.harnesses[harness.id] = harness
        self.state_data.agent_harness = harness.id
        self.agent_enabled_var.set(True)
        if self.harness_menu is not None:
            self.harness_menu.set_harnesses(self.state_data.harnesses, harness.id)
        self._update_agent_setup_controls()

    def _setup_add_custom_harness(self) -> None:
        CustomHarnessDialog(self, self.state_data.config, self.state_data.harnesses, self._setup_store_harness)

    def _page_model_configuration(self, p) -> None:
        p.columnconfigure(0, weight=1); p.rowconfigure(2, weight=1)
        ttk.Label(p,text="Configure selected models",font=("TkDefaultFont",12,"bold")).grid(row=0,column=0,sticky="w")
        ttk.Label(p,text="V1.0 does not trust pre-V1 launch settings. Each model selected for Coding, Chat or Agent must be configured again before setup can finish.",style="Muted.TLabel",wraplength=760).grid(row=1,column=0,sticky="w",pady=(4,10))
        self.config_model_list=tk.Listbox(p,height=12,exportselection=False); self.config_model_list.grid(row=2,column=0,sticky="nsew")
        controls=ttk.Frame(p); controls.grid(row=3,column=0,sticky="ew",pady=10)
        ttk.Button(controls,text="Configure selected model…",style="Primary.TButton",command=self._configure_selected_model).pack(side="left")
        ttk.Button(controls,text="Refresh status",command=self._refresh_model_configuration).pack(side="left",padx=8)

    def _page_prompt(self, p) -> None:
        p.columnconfigure(1,weight=1)
        ttk.Label(p,text="Prompt Helper",font=("TkDefaultFont",12,"bold")).grid(row=0,column=0,columnspan=2,sticky="w",pady=(0,10))
        ttk.Checkbutton(
            p, text="Enable Prompt Helper", variable=self.prompt_enabled_var,
            command=self._update_setup_prompt_controls,
        ).grid(row=1,column=0,columnspan=2,sticky="w",pady=(0,8))
        ttk.Label(p,text="Preferred model").grid(row=2,column=0,sticky="w",pady=5)
        self.prompt_menu=ModelMenu(p,self.prompt_var); self.prompt_menu.grid(row=2,column=1,sticky="ew",pady=5); self.prompt_menu.set_models(self.state_data.models,self.state_data.prompt_model,allow_none=True)
        self.prompt_auto_check = ttk.Checkbutton(p,text="Automatically use lightest suitable model",variable=self.auto_var)
        self.prompt_auto_check.grid(row=3,column=0,columnspan=2,sticky="w",pady=5)
        ttk.Label(p,text="Processing").grid(row=4,column=0,sticky="nw",pady=5)
        self.prompt_modes=ttk.Frame(p); self.prompt_modes.grid(row=4,column=1,sticky="w")
        self.prompt_mode_buttons=[]
        for text,value in (("CPU - lowest VRAM use","cpu"),("GPU - faster response","gpu"),("Auto - balance performance and VRAM","auto")):
            button=ttk.Radiobutton(self.prompt_modes,text=text,value=value,variable=self.mode_var); button.pack(anchor="w"); self.prompt_mode_buttons.append(button)
        self.prompt_keep_check = ttk.Checkbutton(p,text="Keep Prompt Helper loaded after use",variable=self.keep_var)
        self.prompt_keep_check.grid(row=5,column=0,columnspan=2,sticky="w",pady=8)
        ttk.Label(p,text="Advanced Prompt Helper settings can be changed later under Preferences > Settings. If disabled, Improve Prompt is greyed out but the rest of Prompt Workshop remains available.",wraplength=620,style="Muted.TLabel").grid(row=6,column=0,columnspan=2,sticky="w",pady=(12,0))
        self._update_setup_prompt_controls()

    def _update_setup_prompt_controls(self) -> None:
        enabled = self.prompt_enabled_var.get()
        state = "normal" if enabled else "disabled"
        if hasattr(self, "prompt_menu"):
            self.prompt_menu.configure(state=state)
        if hasattr(self, "prompt_auto_check"):
            self.prompt_auto_check.configure(state=state)
        if hasattr(self, "prompt_mode_buttons"):
            for button in self.prompt_mode_buttons:
                button.configure(state=state)
        if hasattr(self, "prompt_keep_check"):
            self.prompt_keep_check.configure(state=state)

    def _selected_task_models(self) -> list[str]:
        result: list[str] = []
        for task, menu in self.task_menus.items():
            if task == "agent" and not self.agent_enabled_var.get():
                continue
            model_id = menu.selected_id()
            if model_id and model_id not in result:
                result.append(model_id)
        return result

    def _refresh_model_configuration(self) -> None:
        try:
            fresh = load_app_config(self.state_data.config_dir)
            self.state_data.config = fresh
            self.state_data.models = dict(fresh.models)
        except Exception as exc:
            messagebox.showerror("Model configuration", str(exc), parent=self)
            return
        if not hasattr(self, "config_model_list"):
            return
        self.config_model_list.delete(0, "end")
        for model_id in self._selected_task_models():
            model = self.state_data.models.get(model_id)
            if model is None:
                self.config_model_list.insert("end", f"{model_id} - missing")
                continue
            status = "Configured for V1.0" if model.tuning_reviewed else "CONFIGURATION REQUIRED"
            self.config_model_list.insert("end", f"{model.display_name} - {status}")
        if self.config_model_list.size():
            self.config_model_list.selection_set(0)

    def _configure_selected_model(self) -> None:
        selected_ids = self._selected_task_models()
        if not selected_ids:
            messagebox.showinfo("Model configuration", "Choose task models first.", parent=self)
            return
        index = self.config_model_list.curselection()[0] if self.config_model_list.curselection() else 0
        index = min(index, len(selected_ids) - 1)
        model_id = selected_ids[index]
        try:
            fresh = load_app_config(self.state_data.config_dir)
        except Exception as exc:
            messagebox.showerror("Model configuration", str(exc), parent=self)
            return
        root = self._root()
        on_launch = getattr(root, "_launch_tuned_model", None)
        ModelTuningWindow(
            self, self.state_data.config_dir, fresh, self._refresh_model_configuration,
            on_launch, model_id, setup_mode=True,
            on_continue=self._model_configuration_continue,
            on_back=lambda _model_id: self._refresh_model_configuration(),
        )

    def _model_configuration_continue(self, _model_id) -> None:
        self._refresh_model_configuration()
        # Put the next unconfigured model under the selection so Save & Continue
        # returns naturally to the setup flow without requiring the window X.
        try:
            fresh = load_app_config(self.state_data.config_dir)
        except Exception:
            return
        selected_ids = self._selected_task_models()
        for index, model_id in enumerate(selected_ids):
            model = fresh.models.get(model_id)
            if model is None or not model.tuning_reviewed:
                self.config_model_list.selection_clear(0, "end")
                self.config_model_list.selection_set(index)
                self.config_model_list.see(index)
                return

    def _all_selected_models_configured(self) -> tuple[bool, list[str]]:
        try:
            fresh = load_app_config(self.state_data.config_dir)
        except Exception:
            fresh = self.state_data.config
        missing = []
        for model_id in self._selected_task_models():
            model = fresh.models.get(model_id)
            if model is None or not model.tuning_reviewed:
                missing.append(model.display_name if model else model_id)
        return not missing, missing

    def _page_review(self, p) -> None:
        ttk.Label(p,text="Review",font=("TkDefaultFont",12,"bold")).pack(anchor="w")
        self.review_text=tk.Text(p,height=18,wrap="word",state="disabled"); self.review_text.pack(fill="both",expand=True,pady=10)

    def _scan(self) -> None:
        try:
            roots=tuple(Path(x.strip()).expanduser() for x in self.root_text.get("1.0","end-1c").splitlines() if x.strip())
            if not roots: raise ValueError("At least one model root is required")
            self.state_data.scan(roots)
        except Exception as exc:
            messagebox.showerror("Model discovery",str(exc),parent=self); return
        self._fill_model_list()
        for task,menu in self.task_menus.items(): menu.set_models(self.state_data.models,menu.selected_id() or self.state_data.task_models.get(task))
        if hasattr(self,"prompt_menu"): self.prompt_menu.set_models(self.state_data.models,self.prompt_menu.selected_id() or self.state_data.prompt_model,allow_none=True)
        self.status_var.set(f"Found {len(self.state_data.models)} model(s)")

    def _fill_model_list(self) -> None:
        if not hasattr(self,"model_list"): return
        self.model_list.delete(0,"end")
        for m in self.state_data.models.values(): self.model_list.insert("end",_model_label(m))

    def _show_page(self,index:int) -> None:
        self.page=index
        for i,frame in enumerate(self.pages):
            if i==index: frame.tkraise()
        self.back.configure(state="disabled" if index==0 else "normal")
        self.next.configure(text="Save & Finish" if index==len(self.pages)-1 else "Next")
        if index==3: self._refresh_model_configuration()
        if index==len(self.pages)-1: self._update_review()

    def _back(self) -> None:
        if self.page>0: self._show_page(self.page-1)

    def _next(self) -> None:
        if self.page==1: self._scan()
        if self.page==2:
            for task in ("coding", "chat"):
                menu = self.task_menus[task]
                if menu.selected_id() is None:
                    messagebox.showerror("Setup Wizard", f"Choose a model for {task.title()}", parent=self); return
            if self.agent_enabled_var.get():
                if self.task_menus["agent"].selected_id() is None:
                    messagebox.showerror("Setup Wizard", "Choose a model for Agent", parent=self); return
                if self.harness_menu is None or self.harness_menu.selected_id() is None:
                    messagebox.showerror("Setup Wizard", "Choose an Agent harness, scan for one, add a custom harness, or disable Agent setup.", parent=self); return
        if self.page==3:
            ok, missing = self._all_selected_models_configured()
            if not ok:
                messagebox.showerror("Setup Wizard", "Configure these selected models before continuing:\n\n" + "\n".join(missing), parent=self); return
        if self.page==len(self.pages)-1:
            self._finish(); return
        self._show_page(self.page+1)

    def _update_review(self) -> None:
        task_lines=[]
        for task in ("coding", "chat"):
            task_lines.append(f"{task.title()}: {self.task_menus[task].selected_id() or 'Not selected'}")
        if self.agent_enabled_var.get():
            task_lines.append(f"Agent: {self.task_menus['agent'].selected_id() or 'Not selected'}")
            harness = self.harness_menu.selected_id() if self.harness_menu is not None else None
            harness_name = self.state_data.harnesses[harness].display_name if harness in self.state_data.harnesses else "Not selected"
        else:
            task_lines.append("Agent: Not configured")
            harness_name = "Not configured"
        if self.prompt_enabled_var.get():
            helper="Automatic lightest model" if self.auto_var.get() else (self.prompt_menu.selected_id() or "Not selected")
            prompt_text=f"Enabled\n  Model: {helper}\n  Processing: {self.mode_var.get().upper()}\n  Keep loaded: {'Yes' if self.keep_var.get() else 'No'}"
        else:
            prompt_text="Disabled"
        text=("Model roots:\n  " + "\n  ".join(str(p) for p in self.state_data.roots) + "\n\nTask defaults:\n  " + "\n  ".join(task_lines) + f"\n\nAgent harness:\n  {harness_name}\n\nPrompt Helper:\n  {prompt_text}")
        self.review_text.configure(state="normal"); self.review_text.delete("1.0","end"); self.review_text.insert("1.0",text); self.review_text.configure(state="disabled")

    def _finish(self) -> None:
        try:
            roots=tuple(Path(x.strip()).expanduser() for x in self.root_text.get("1.0","end-1c").splitlines() if x.strip())
            if not roots: raise ValueError("At least one model root is required")
            self.state_data.roots=list(roots)
            for task in ("coding", "chat"):
                model=self.task_menus[task].selected_id()
                if model is None: raise ValueError(f"Choose a model for {task.title()}")
                self.state_data.task_models[task]=model
            if self.agent_enabled_var.get():
                model=self.task_menus["agent"].selected_id()
                if model is None: raise ValueError("Choose a model for Agent")
                harness=self.harness_menu.selected_id() if self.harness_menu is not None else None
                if harness is None: raise ValueError("Choose an Agent harness or disable Agent setup")
                self.state_data.task_models["agent"]=model
                self.state_data.agent_harness=harness
            else:
                self.state_data.task_models["agent"]=None
                self.state_data.agent_harness=None
            ok, missing = self._all_selected_models_configured()
            if not ok: raise ValueError("Configure selected models before finishing: " + ", ".join(missing))
            self.state_data.prompt_enabled=self.prompt_enabled_var.get()
            self.state_data.prompt_model=self.prompt_menu.selected_id() if self.state_data.prompt_enabled else None
            self.state_data.auto_lightest=self.auto_var.get()
            if self.state_data.prompt_enabled and not self.state_data.auto_lightest and self.state_data.prompt_model is None: raise ValueError("Choose a Prompt Helper model")
            self.state_data.processing_mode=self.mode_var.get(); self.state_data.keep_loaded=self.keep_var.get()
            self.state_data.save()
        except Exception as exc:
            messagebox.showerror("Setup Wizard",str(exc),parent=self); return
        self.on_saved(); self.destroy()
