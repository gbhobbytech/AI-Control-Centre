"""Model tuning controls kept separate from service orchestration."""
from __future__ import annotations

from dataclasses import replace
import tkinter as tk
from tkinter import messagebox, ttk

from .domain import ProcessServiceConfig
from .models import model_total_size_bytes
from .monitoring import LinuxHostMonitor, create_gpu_monitor
from .preferences import save_model_tuning
from .theme import style_classic
from .ui_utils import ScrollableFrame, fit_window_to_screen
from .tuning import BASELINE, CACHE_TYPES, CURRENT_TUNING_SCHEMA, is_llama_service, model_values, running_settings, suggested_tuning_presets, validate_values

LABELS = {
    'recommended_gpu_layers': ('GPU layers', '0 keeps model layers in RAM. Increase gradually while watching VRAM.'),
    'recommended_context_length': ('Context tokens', 'Space for prompt, history and reply. Larger contexts need more memory.'),
    'max_output_tokens': ('Maximum reply tokens', 'Caps reply length. Leave context space for your input and history.'),
    'cache_type_k': ('K cache format', 'f16 is a compatibility starting point; q8_0/q4_0 can reduce memory.'),
    'cache_type_v': ('V cache format', 'Quantised V cache needs flash attention and backend/model support.'),
    'flash_attention': ('Flash attention', 'Auto lets llama.cpp choose. Support depends on your build and model.'),
    'startup_timeout_seconds': ('Startup timeout (s)', 'Time allowed for loading. A longer timeout cannot fix a crashed process.'),
}


class ModelTuningPanel(ttk.Frame):
    def __init__(self, parent, config_dir, config, on_saved, on_launch=None, initial_model=None, *, show_actions=True):
        super().__init__(parent, padding=12)
        self.config_dir, self.config = config_dir, config
        self.on_saved, self.on_launch = on_saved, on_launch
        self.show_actions = show_actions
        self.models = dict(config.models)
        self.model_id = None
        self.dirty = False
        self.filling = False
        self.source = 'manual'
        self.variables = {key: tk.StringVar(self) for key in BASELINE}
        self.model_label = tk.StringVar(self)
        self.status = tk.StringVar(self)
        self.hardware_summary = tk.StringVar(self, value='Detecting hardware…')
        self.preset_var = tk.StringVar(self, value='balanced')
        self.preset_summary = tk.StringVar(self, value='Select a model to calculate starting suggestions.')
        self.presets = {}
        self.gpu_status = None
        self.host_status = None
        self._detect_hardware()

        self.columnconfigure(1, weight=1)
        heading = ttk.Frame(self)
        heading.grid(row=0, column=0, columnspan=3, sticky='ew')
        ttk.Label(heading, text='Launch settings for this computer', font='AppHeading').pack(side='left')
        if hasattr(self._root(), 'vram_var'):
            ttk.Label(heading, textvariable=self._root().vram_var, style='Muted.TLabel').pack(side='right')
            ttk.Label(heading, text='VRAM in use: ', style='Muted.TLabel').pack(side='right')
        ttk.Label(self, text='Saved per model; Coding, Chat and Agent use the selected model’s settings.', style='Muted.TLabel').grid(row=1, column=0, columnspan=3, sticky='w', pady=(3, 12))
        ttk.Label(self, text='Model').grid(row=2, column=0, sticky='w', padx=(0, 12))
        self.picker = ttk.Menubutton(self, textvariable=self.model_label, direction='below')
        self.menu = tk.Menu(self.picker, tearoff=False)
        self.picker.configure(menu=self.menu)
        self.picker.grid(row=2, column=1, columnspan=2, sticky='ew')
        ttk.Label(self, textvariable=self.status, wraplength=700, style='Muted.TLabel').grid(row=3, column=0, columnspan=3, sticky='w', pady=(7, 12))

        reckoner = ttk.LabelFrame(self, text='Suggested starting point', padding=12)
        reckoner.grid(row=4, column=0, columnspan=3, sticky='ew', pady=(0, 12))
        reckoner.columnconfigure(0, weight=1)
        ttk.Label(reckoner, textvariable=self.hardware_summary, style='Muted.TLabel', wraplength=820).grid(row=0, column=0, columnspan=4, sticky='w')
        presets = ttk.Frame(reckoner)
        presets.grid(row=1, column=0, columnspan=4, sticky='ew', pady=(8, 6))
        for index, (value, label) in enumerate((('conservative', 'Conservative'), ('balanced', 'Balanced'), ('maximum', 'Maximum GPU'))):
            ttk.Radiobutton(presets, text=label, value=value, variable=self.preset_var, command=self._preset_changed).grid(row=0, column=index, sticky='w', padx=(0, 18))
        ttk.Button(presets, text='Apply suggested settings', style='Primary.TButton', command=self._apply_preset).grid(row=0, column=3, sticky='e')
        presets.columnconfigure(3, weight=1)
        ttk.Label(reckoner, textvariable=self.preset_summary, wraplength=820, style='Muted.TLabel').grid(row=2, column=0, columnspan=4, sticky='w', pady=(2, 2))
        ttk.Label(reckoner, text='These are practical starting values based on detected memory and model size. They are not guaranteed optimal settings; confirm a successful launch before tuning further.', wraplength=820, style='Muted.TLabel').grid(row=3, column=0, columnspan=4, sticky='w', pady=(4, 0))

        for row, (key, (label, help_text)) in enumerate(LABELS.items(), start=5):
            ttk.Label(self, text=label).grid(row=row, column=0, sticky='w', padx=(0, 12), pady=5)
            var = self.variables[key]
            choices = CACHE_TYPES if key.startswith('cache_type') else (('auto', 'on', 'off') if key == 'flash_attention' else None)
            if choices:
                field = ttk.OptionMenu(self, var, var.get(), *choices)
            else:
                field = ttk.Entry(self, textvariable=var, width=12)
            field.grid(row=row, column=1, sticky='ew', pady=5)
            ttk.Label(self, text=help_text, wraplength=330, style='Muted.TLabel').grid(row=row, column=2, sticky='w', padx=(14, 0), pady=5)
            var.trace_add('write', self._changed)
        tools = ttk.Frame(self)
        tools.grid(row=12, column=0, columnspan=3, sticky='ew', pady=(10, 6))
        ttk.Button(tools, text='Import from running server…', command=self._import).pack(side='left')
        ttk.Label(self, text='A successful launch confirms readiness, not performance. Watch VRAM and test a representative prompt before increasing context or GPU offload further.', wraplength=700, style='Muted.TLabel').grid(row=13, column=0, columnspan=3, sticky='w', pady=6)
        self.save_button = None
        self.launch_button = None
        if self.show_actions:
            actions = ttk.Frame(self)
            actions.grid(row=14, column=0, columnspan=3, sticky='ew', pady=(6, 0))
            self.save_button = ttk.Button(actions, text='Save model settings', style='Primary.TButton', command=self.save)
            self.save_button.pack(side='left')
            self.launch_button = ttk.Button(actions, text='Save & launch', command=lambda: self.save(launch=True))
            self.launch_button.pack(side='left', padx=8)
        self.set_models(self.models, initial_model)
        style_classic(self)

    def _detect_hardware(self):
        try:
            monitor = create_gpu_monitor(self.config.settings.gpu_backend)
            self.gpu_status = monitor.status() if monitor else None
        except Exception:
            self.gpu_status = None
        try:
            self.host_status = LinuxHostMonitor().status()
        except Exception:
            self.host_status = None

    def _refresh_reckoner(self):
        if self.model_id is None or self.model_id not in self.models:
            self.presets = {}
            self.hardware_summary.set('Select a model to calculate starting suggestions.')
            self.preset_summary.set('')
            return
        model = self.models[self.model_id]
        size = model_total_size_bytes(model)
        gpu_total = self.gpu_status.memory_total_mib if self.gpu_status and self.gpu_status.available else None
        host_total = self.host_status.memory_total_mib if self.host_status else None
        self.presets = suggested_tuning_presets(
            model_size_bytes=size,
            gpu_total_mib=gpu_total,
            host_total_mib=host_total,
        )
        parts = []
        if self.gpu_status and self.gpu_status.available:
            parts.append(f'GPU: {self.gpu_status.name or "detected GPU"} · {self.gpu_status.memory_total_mib / 1024:.1f} GB VRAM')
        else:
            parts.append('GPU: no supported VRAM reading detected')
        if host_total:
            parts.append(f'System RAM: {host_total / 1024:.1f} GB')
        if size is not None:
            parts.append(f'Model: {size / (1024 ** 3):.1f} GB on disk')
        else:
            parts.append('Model size: unavailable')
        self.hardware_summary.set('   |   '.join(parts))
        self._preset_changed()

    def _preset_changed(self):
        preset = self.presets.get(self.preset_var.get())
        if preset is None:
            self.preset_summary.set('No suggestion is available for this model.')
            return
        values = preset.values
        layers = values['recommended_gpu_layers']
        layer_text = 'full/offload-all request' if layers == 999 else str(layers)
        self.preset_summary.set(
            f'{preset.display_name}: GPU layers {layer_text}, context {values["recommended_context_length"]}, '
            f'K/V cache {values["cache_type_k"]}/{values["cache_type_v"]}. {preset.summary}'
        )

    def _apply_preset(self):
        preset = self.presets.get(self.preset_var.get())
        if preset is None:
            return
        self._fill(preset.values)
        self.dirty = True
        self.source = f'ready reckoner: {preset.display_name.lower()}'
        self.status.set(f'{preset.display_name} starting values applied. Review them, then save and launch. Successful loading is the next check.')

    def _changed(self, *_):
        if not self.filling:
            self.dirty = True
            self.source = 'manual'

    def set_models(self, models, selected=None):
        self.models = dict(models)
        self.menu.delete(0, 'end')
        for model in self.models.values():
            self.menu.add_command(label=model.display_name, command=lambda mid=model.id: self.select_model(mid))
        selected = selected or self.model_id or next(iter(self.models), None)
        if selected in self.models:
            self.select_model(selected, force=True)
        else:
            self.model_id = None
            self.model_label.set('No models found')
            if self.save_button is not None:
                self.save_button.state(['disabled'])
            if self.launch_button is not None:
                self.launch_button.state(['disabled'])

    def select_model(self, model_id, force=False):
        if self.dirty and not force and not messagebox.askyesno('Unsaved model settings', 'Discard your unsaved model changes?', parent=self):
            return
        self.model_id = model_id
        model = self.models[model_id]
        self.model_label.set(model.display_name)
        self._fill(model_values(model) if model.tuning_reviewed else BASELINE)
        self.source = model.tuning_source if model.tuning_reviewed else 'manual'
        state = 'Configured for V1.0' if model.tuning_reviewed else 'Configuration required for V1.0'
        detail = '' if model.tuning_reviewed else ' · Historical pre-V1 values are not active'
        self.status.set(f'{state}{detail} · {model.path}')
        self._refresh_reckoner()
        self.dirty = False
        if self.save_button is not None:
            self.save_button.state(['!disabled'])
        if self.launch_button is not None:
            self.launch_button.state(['!disabled'] if model.complete and self.on_launch else ['disabled'])

    def _fill(self, values):
        self.filling = True
        try:
            for key, value in values.items():
                if key in self.variables:
                    self.variables[key].set(str(value))
        finally:
            self.filling = False

    def _baseline(self):
        self._fill(BASELINE)
        self.dirty = True
        self.source = 'conservative starting values'
        self.status.set('Starting values selected. Review, save and launch; then increase GPU layers gradually.')

    def _service(self):
        helper = self.config.settings.prompt_workshop.service
        return next((service for sid, service in self.config.services.items()
                     if sid != helper and isinstance(service, ProcessServiceConfig) and is_llama_service(service)), None)

    def _import(self):
        if self.model_id is None:
            return
        service = self._service()
        if service is None:
            messagebox.showinfo('Import settings', 'No main llama.cpp process service is configured.', parent=self)
            return
        candidates = running_settings(self.models[self.model_id], service.executable)
        if not candidates:
            messagebox.showinfo('Import settings', 'No readable local server matches this executable and model file.\n\nImport reads explicit command-line settings from your Linux processes. Docker, remote servers, environment-only settings and server defaults are not imported.', parent=self)
            return
        window = tk.Toplevel(self)
        window.title('Review detected settings')
        window.transient(self.winfo_toplevel())
        window.geometry('680x470')
        frame = ttk.Frame(window, padding=16)
        frame.pack(fill='both', expand=True)
        ttk.Label(frame, text='Choose a process, then review the detected values.', font='AppHeading').pack(anchor='w')
        listing = tk.Listbox(frame, height=4, exportselection=False)
        listing.pack(fill='x', pady=10)
        details = tk.Text(frame, height=10, wrap='word')
        details.pack(fill='both', expand=True)
        for candidate in candidates:
            listing.insert('end', f'PID {candidate.pid} · {self.models[self.model_id].display_name}')
        def show(_event=None):
            selection = listing.curselection()
            if not selection:
                return
            values = candidates[selection[0]].values
            details.configure(state='normal')
            details.delete('1.0', 'end')
            details.insert('1.0', '\n'.join(f'{LABELS[key][0]}: {value}' for key, value in values.items()))
            details.insert('end', '\n\nOnly shown values are imported. Missing values remain unchanged.\nThe process is not adopted or stopped. Review and save before use.')
            details.configure(state='disabled')
        def use():
            selection = listing.curselection()
            if not selection:
                return
            candidate = candidates[selection[0]]
            self._fill(candidate.values)
            self.dirty = True
            self.source = 'imported command line (user reviewed)'
            self.status.set(f'Imported explicit values from PID {candidate.pid}. Check missing values and save. Auto/all GPU layers or unlimited output must be replaced with numeric limits.')
            window.destroy()
        listing.bind('<<ListboxSelect>>', show)
        listing.selection_set(0)
        show()
        ttk.Button(frame, text='Use detected values', style='Primary.TButton', command=use).pack(anchor='e', pady=(10, 0))
        style_classic(window)

    def save(self, launch=False):
        if self.model_id is None:
            return False
        root = self._root()
        if getattr(root, 'operation_busy', False) or getattr(root, 'prompt_busy', False):
            messagebox.showinfo('Model settings', 'Wait for the current operation to finish before saving.', parent=self)
            return False
        try:
            values = validate_values({key: var.get() for key, var in self.variables.items()})
            model = self.models[self.model_id]
            save_model_tuning(self.config_dir, model, values, self.source)
        except Exception as exc:
            messagebox.showerror('Model settings', str(exc), parent=self)
            return False
        self.models[model.id] = replace(
            model, **values, tuning_reviewed=True, tuning_source=self.source,
            tuning_schema_version=CURRENT_TUNING_SCHEMA
        )
        self.dirty = False
        self.status.set('Configured for V1.0. Applies on the next launch/restart; the running server has not been changed.')
        self.on_saved()
        if launch and self.on_launch:
            self.on_launch(model.id)
        return True


class ModelTuningWindow(tk.Toplevel):
    """Standalone model configuration window used by Setup Wizard.

    The content scrolls when the desktop is short, while the Back / Save &
    Continue action bar remains visible at all times.
    """
    def __init__(
        self, parent, config_dir, config, on_saved, on_launch=None, initial_model=None,
        *, setup_mode=False, on_continue=None, on_back=None,
    ):
        super().__init__(parent)
        self.title('Configure model')
        self.transient(parent)
        self.setup_mode = setup_mode
        self.on_continue = on_continue
        self.on_back = on_back

        outer = ttk.Frame(self, padding=(8, 8, 8, 10))
        outer.pack(fill='both', expand=True)
        outer.rowconfigure(0, weight=1)
        outer.columnconfigure(0, weight=1)

        scroll = ScrollableFrame(outer)
        scroll.grid(row=0, column=0, sticky='nsew')
        self.panel = ModelTuningPanel(
            scroll.body, config_dir, config, on_saved, on_launch, initial_model,
            show_actions=not setup_mode,
        )
        self.panel.pack(fill='both', expand=True)

        if setup_mode:
            actions = ttk.Frame(outer)
            actions.grid(row=1, column=0, sticky='ew', pady=(10, 0))
            actions.columnconfigure(1, weight=1)
            ttk.Button(actions, text='Back', command=self._back).grid(row=0, column=0, sticky='w')
            ttk.Button(
                actions, text='Save & Continue', style='Primary.TButton',
                command=self._save_and_continue,
            ).grid(row=0, column=2, sticky='e')
            self.protocol('WM_DELETE_WINDOW', self._back)

        style_classic(self)
        fit_window_to_screen(
            self, preferred_width=1040, preferred_height=940,
            min_width=820, min_height=600, margin_y=110,
        )

    def _back(self):
        if self.panel.dirty and not messagebox.askyesno(
            'Unsaved model settings',
            'Return to setup without saving these model settings?',
            parent=self,
        ):
            return
        if self.on_back is not None:
            self.on_back(self.panel.model_id)
        self.destroy()

    def _save_and_continue(self):
        model_id = self.panel.model_id
        if not self.panel.save(launch=False):
            return
        if self.on_continue is not None:
            self.on_continue(model_id)
        self.destroy()
