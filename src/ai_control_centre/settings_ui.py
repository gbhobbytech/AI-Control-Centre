from __future__ import annotations

from pathlib import Path
import tkinter as tk
from tkinter import colorchooser, messagebox, ttk

from .config import AppConfig
from .models import model_total_size_bytes
from .preferences import detect_setup_environment, rescan_models, save_preferences, save_appearance
from .theme import DEFAULT_APPEARANCE, style_classic, validate_appearance
from .tuning_ui import ModelTuningPanel

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
        prompt = config.settings.prompt_workshop
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
            prompt_model=self.prompt_model,
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


class SettingsWindow(tk.Toplevel):
    def __init__(self, parent, config_dir: Path, config: AppConfig, on_saved, *, initial_tab=None, model_id=None):
        super().__init__(parent)
        self.title("AI Control Centre Settings")
        self.geometry("1000x900")
        self.minsize(920, 820)
        self.transient(parent)
        self.config_dir = config_dir
        self.state_data = _PreferenceState(config_dir, config)
        self.on_saved = on_saved
        self.model_vars: dict[str, tk.StringVar] = {}
        self.model_menus: dict[str, ModelMenu] = {}
        self.prompt_model_var = tk.StringVar()
        self.root_var = tk.StringVar(value="\n".join(str(p) for p in self.state_data.roots))
        self.scan_status = tk.StringVar(value=f"{len(self.state_data.models)} model(s) discovered")
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
            self.notebook.select(self.tuning)
        style_classic(self)
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
        prompt = ttk.Frame(self.notebook, padding=12)
        self.notebook.add(models, text="Models")
        self.notebook.add(tasks, text="Task Defaults")
        self.notebook.add(prompt, text="Prompt Helper")
        self.tuning = ModelTuningPanel(self.notebook, self.config_dir, self.state_data.config,
                                      self.on_saved, self._launch_tuned)
        self.notebook.add(self.tuning, text='Model tuning')
        appearance = ttk.Frame(self.notebook, padding=16)
        self.notebook.add(appearance, text='Appearance')
        self._build_appearance(appearance)
        self._build_models(models)
        self._build_tasks(tasks)
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
        self._refresh_model_menus()

    def _build_prompt(self, parent) -> None:
        parent.columnconfigure(1, weight=1)
        row = 0
        ttk.Label(parent, text="Prompt Helper", font=("TkDefaultFont", 11, "bold")).grid(row=row, column=0, columnspan=2, sticky="w", pady=(0, 12)); row += 1
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
            menu.set_models(self.state_data.models, selected)

    def _refresh_prompt_menu(self) -> None:
        selected = self.prompt_model_menu.selected_id() if hasattr(self, "prompt_model_menu") else None
        if selected is None:
            selected = self.state_data.prompt_model
        self.prompt_model_menu.set_models(self.state_data.models, selected, allow_none=True)

    def _collect(self) -> None:
        self.state_data.roots = list(self._roots_from_text())
        for task, menu in self.model_menus.items():
            model_id = menu.selected_id()
            if model_id is not None:
                self.state_data.task_models[task] = model_id
        self.state_data.prompt_model = self.prompt_model_menu.selected_id()
        self.state_data.auto_lightest = self.auto_var.get()
        if not self.state_data.auto_lightest and self.state_data.prompt_model is None:
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
        ttk.Label(frame, text='Enlighten Tech', font='AppHeading').grid(row=0, column=0, columnspan=3, sticky='w')
        ttk.Label(frame, text='Deep blue and copper, with graphite text and clean panel surfaces.', style='Muted.TLabel').grid(row=1, column=0, columnspan=3, sticky='w', pady=(4, 18))
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
        ttk.Button(controls, text='Restore brand defaults', command=self._reset_appearance).pack(side='left', padx=10)

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
        self.geometry("760x590")
        self.minsize(700, 540)
        self.transient(parent)
        self.grab_set()
        self.state_data = _PreferenceState(config_dir, config)
        self.on_saved = on_saved
        self.page = 0
        self.pages: list[ttk.Frame] = []
        self.root_var = tk.StringVar(value="\n".join(str(p) for p in self.state_data.roots))
        self.task_vars = {task: tk.StringVar() for task in _TASKS}
        self.task_menus: dict[str, ModelMenu] = {}
        self.prompt_var = tk.StringVar()
        self.auto_var = tk.BooleanVar(value=self.state_data.auto_lightest)
        self.mode_var = tk.StringVar(value=self.state_data.processing_mode)
        self.keep_var = tk.BooleanVar(value=self.state_data.keep_loaded)
        self.status_var = tk.StringVar(value="")
        self._build()
        self._show_page(0)
        style_classic(self)

    def _build(self) -> None:
        outer = ttk.Frame(self, padding=16)
        outer.pack(fill="both", expand=True)
        outer.rowconfigure(1, weight=1)
        outer.columnconfigure(0, weight=1)
        ttk.Label(outer, text="AI Control Centre Setup", font=("TkDefaultFont", 16, "bold")).grid(row=0, column=0, sticky="w", pady=(0, 12))
        host = ttk.Frame(outer); models = ttk.Frame(outer); tasks = ttk.Frame(outer); prompt = ttk.Frame(outer); review = ttk.Frame(outer)
        self.pages = [host, models, tasks, prompt, review]
        for frame in self.pages:
            frame.grid(row=1, column=0, sticky="nsew")
        self._page_host(host)
        self._page_models(models)
        self._page_tasks(tasks)
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
        ttk.Label(p,text="Task model defaults",font=("TkDefaultFont",12,"bold")).grid(row=0,column=0,columnspan=2,sticky="w",pady=(0,10))
        for row,task in enumerate(_TASKS,start=1):
            ttk.Label(p,text=task.title()).grid(row=row,column=0,sticky="w",pady=6,padx=(0,12))
            menu=ModelMenu(p,self.task_vars[task]); menu.grid(row=row,column=1,sticky="ew",pady=6); self.task_menus[task]=menu
            menu.set_models(self.state_data.models,self.state_data.task_models.get(task))

    def _page_prompt(self, p) -> None:
        p.columnconfigure(1,weight=1)
        ttk.Label(p,text="Prompt Helper",font=("TkDefaultFont",12,"bold")).grid(row=0,column=0,columnspan=2,sticky="w",pady=(0,10))
        ttk.Label(p,text="Preferred model").grid(row=1,column=0,sticky="w",pady=5)
        self.prompt_menu=ModelMenu(p,self.prompt_var); self.prompt_menu.grid(row=1,column=1,sticky="ew",pady=5); self.prompt_menu.set_models(self.state_data.models,self.state_data.prompt_model,allow_none=True)
        ttk.Checkbutton(p,text="Automatically use lightest suitable model",variable=self.auto_var).grid(row=2,column=0,columnspan=2,sticky="w",pady=5)
        ttk.Label(p,text="Processing").grid(row=3,column=0,sticky="nw",pady=5)
        modes=ttk.Frame(p); modes.grid(row=3,column=1,sticky="w")
        for text,value in (("CPU - lowest VRAM use","cpu"),("GPU - faster response","gpu"),("Auto - balance performance and VRAM","auto")):
            ttk.Radiobutton(modes,text=text,value=value,variable=self.mode_var).pack(anchor="w")
        ttk.Checkbutton(p,text="Keep Prompt Helper loaded after use",variable=self.keep_var).grid(row=4,column=0,columnspan=2,sticky="w",pady=8)
        ttk.Label(p,text="Advanced Prompt Helper settings can be changed later under File > Settings.",wraplength=620).grid(row=5,column=0,columnspan=2,sticky="w",pady=(12,0))

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
        if index==len(self.pages)-1: self._update_review()

    def _back(self) -> None:
        if self.page>0: self._show_page(self.page-1)

    def _next(self) -> None:
        if self.page==1: self._scan()
        if self.page==len(self.pages)-1:
            self._finish(); return
        self._show_page(self.page+1)

    def _update_review(self) -> None:
        task_lines=[]
        for task,menu in self.task_menus.items(): task_lines.append(f"{task.title()}: {menu.selected_id() or 'Not selected'}")
        helper="Automatic lightest model" if self.auto_var.get() else (self.prompt_menu.selected_id() or "Not selected")
        text=("Model roots:\n  " + "\n  ".join(str(p) for p in self.state_data.roots) + "\n\nTask defaults:\n  " + "\n  ".join(task_lines) + f"\n\nPrompt Helper:\n  Model: {helper}\n  Processing: {self.mode_var.get().upper()}\n  Keep loaded: {'Yes' if self.keep_var.get() else 'No'}")
        self.review_text.configure(state="normal"); self.review_text.delete("1.0","end"); self.review_text.insert("1.0",text); self.review_text.configure(state="disabled")

    def _finish(self) -> None:
        try:
            roots=tuple(Path(x.strip()).expanduser() for x in self.root_text.get("1.0","end-1c").splitlines() if x.strip())
            if not roots: raise ValueError("At least one model root is required")
            self.state_data.roots=list(roots)
            for task,menu in self.task_menus.items():
                model=menu.selected_id()
                if model is None: raise ValueError(f"Choose a model for {task.title()}")
                self.state_data.task_models[task]=model
            self.state_data.prompt_model=self.prompt_menu.selected_id()
            self.state_data.auto_lightest=self.auto_var.get()
            if not self.state_data.auto_lightest and self.state_data.prompt_model is None: raise ValueError("Choose a Prompt Helper model")
            self.state_data.processing_mode=self.mode_var.get(); self.state_data.keep_loaded=self.keep_var.get()
            self.state_data.save()
        except Exception as exc:
            messagebox.showerror("Setup Wizard",str(exc),parent=self); return
        self.on_saved(); self.destroy()
