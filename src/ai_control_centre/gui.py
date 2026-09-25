from __future__ import annotations

import argparse
import queue
import threading
from collections.abc import Callable
from pathlib import Path

try:
    import tkinter as tk
    from tkinter import messagebox, ttk
except ImportError as exc:  # pragma: no cover - depends on host packaging
    raise RuntimeError(
        "Tkinter is not available. Install the distribution package that provides Python Tk support."
    ) from exc

from .config import ConfigError, load_app_config
from .domain import ProcessServiceConfig, ServiceState
from .manager import ServiceManager
from .models import smallest_complete_model
from .monitoring import GpuStatus, HostStatus, LinuxHostMonitor, create_gpu_monitor
from .preferences import save_window_size
from .prompt_client import PromptClient, prompt_service_action, select_prompt
from .settings_ui import SettingsWindow, SetupWizard
from .theme import apply_theme, style_classic
from .tuning import is_llama_service

_STATE_LABELS = {
    ServiceState.STOPPED: "Stopped",
    ServiceState.STARTING: "Starting",
    ServiceState.READY: "Ready",
    ServiceState.RUNNING_NOT_READY: "Starting",
    ServiceState.EXTERNAL: "External",
    ServiceState.ERROR: "Error",
    ServiceState.UNKNOWN: "Unknown",
}

_TASK_IDS = ("coding", "chat", "agent", "image")
_TASK_DEFAULT = "Task default"


class ControlCentreWindow(tk.Tk):
    def __init__(self, config_dir: Path, *, force_setup: bool = False):
        super().__init__()
        self.title("AI Control Centre")
        self.minsize(980, 720)

        self.config_dir = config_dir
        self.config_data = load_app_config(config_dir)
        self._restore_window_size()
        apply_theme(self, self.config_data.settings.appearance)
        self.manager = ServiceManager(self.config_data)
        self.gpu_monitor = create_gpu_monitor(self.config_data.settings.gpu_backend)
        self.host_monitor = LinuxHostMonitor()
        self.prompt_config = self.config_data.settings.prompt_workshop
        self.prompt_client = PromptClient(self.prompt_config)

        self.events: queue.Queue[tuple[str, object]] = queue.Queue()
        self.operation_busy = False
        self.prompt_busy = False
        self.status_poll_in_flight = False
        self.monitor_poll_in_flight = False
        self.active_task_id: str | None = None

        self.status_var = tk.StringVar(value="Ready - choose a task")
        self.model_var = tk.StringVar(value=_TASK_DEFAULT)
        self.task_buttons = {}
        self.state_labels = {}
        self.service_vars: dict[str, tuple[tk.StringVar, tk.StringVar]] = {}
        self.model_label_to_id: dict[str, str] = {}

        self.gpu_name_var = tk.StringVar(value="Checking...")
        self.vram_var = tk.StringVar(value="Checking...")
        self.gpu_load_var = tk.StringVar(value="Checking...")
        self.gpu_temp_var = tk.StringVar(value="Checking...")
        self.ram_var = tk.StringVar(value="Checking...")
        self.cpu_var = tk.StringVar(value="Checking...")
        self.active_task_var = tk.StringVar(value="Unknown")
        self.active_model_var = tk.StringVar(value="None")
        self.prompt_helper_var = tk.StringVar(value="Stopped")
        self.service_summary_var = tk.StringVar(value="Checking...")
        self.prompt_status_var = tk.StringVar(value="Draft a task or prompt to begin.")

        self._build_ui()
        self._update_capabilities()
        style_classic(self)
        self.after(100, self._drain_events)
        self.after(150, self._poll_status)
        self.after(250, self._poll_system)
        self.protocol("WM_DELETE_WINDOW", self._on_close)
        if force_setup or not self.config_data.settings.setup_completed:
            self.after(500, self._open_setup_wizard)

    def _restore_window_size(self) -> None:
        settings = self.config_data.settings
        screen_width = max(980, self.winfo_screenwidth())
        screen_height = max(720, self.winfo_screenheight())
        width = min(max(980, settings.window_width), screen_width)
        height = min(max(720, settings.window_height), screen_height)
        self.geometry(f"{width}x{height}")

    def _on_close(self) -> None:
        try:
            self.update_idletasks()
            save_window_size(self.config_dir, self.winfo_width(), self.winfo_height())
        finally:
            self.destroy()

    def _build_ui(self) -> None:
        menu_bar = tk.Menu(self)
        preferences_menu = tk.Menu(menu_bar, tearoff=False)
        preferences_menu.add_command(label="Settings...", command=self._open_settings)
        preferences_menu.add_command(label="Rescan Models", command=self._rescan_models)
        preferences_menu.add_command(label="Run Setup Wizard...", command=self._open_setup_wizard)
        preferences_menu.add_separator()
        preferences_menu.add_command(label="Exit", command=self._on_close)
        menu_bar.add_cascade(label="Preferences", menu=preferences_menu)
        self.configure(menu=menu_bar)

        outer = ttk.Frame(self, padding=16, style="Background.TFrame")
        outer.pack(fill="both", expand=True)
        outer.columnconfigure(0, weight=1)
        outer.rowconfigure(4, weight=1)

        header = ttk.Frame(outer)
        header.grid(row=0, column=0, sticky="ew")
        header.columnconfigure(0, weight=1)
        ttk.Label(
            header, text="AI Control Centre", style="Title.TLabel"
        ).grid(row=0, column=0, sticky="w")
        ttk.Label(header, text="gbhobbytech  |  Local AI services", style="Muted.TLabel").grid(
            row=0, column=1, sticky="e"
        )

        task = ttk.LabelFrame(outer, text="Tasks", padding=12)
        task.grid(row=1, column=0, sticky="ew", pady=(12, 10))
        for column in range(5):
            task.columnconfigure(column, weight=1)

        for column, profile_id in enumerate(_TASK_IDS):
            profile = self.config_data.profiles.get(profile_id)
            if profile is None:
                button = ttk.Button(task, text=profile_id.title(), state="disabled")
            else:
                button = ttk.Button(
                    task,
                    text=profile.display_name,
                    command=lambda pid=profile_id: self._start_profile(pid),
                )
            button.configure(style='Task.TButton')
            self.task_buttons[profile_id] = button
            button.grid(
                row=0, column=column, sticky="ew", padx=(0 if column == 0 else 4, 4)
            )

        ttk.Button(task, text="Stop All", style="Danger.TButton", command=self._stop_all).grid(
            row=0, column=4, sticky="ew", padx=(4, 0)
        )

        model_row = ttk.Frame(task)
        model_row.grid(row=1, column=0, columnspan=5, sticky="ew", pady=(12, 0))
        model_row.columnconfigure(1, weight=1)
        ttk.Label(model_row, text="LLM override:").grid(
            row=0, column=0, sticky="w", padx=(0, 8)
        )

        # A menu button avoids a Linux/Tk combobox popdown positioning bug.
        self.model_button = ttk.Menubutton(
            model_row,
            textvariable=self.model_var,
            direction="below",
        )
        self.model_menu = tk.Menu(self.model_button, tearoff=False)
        self.model_button.configure(menu=self.model_menu)
        self._rebuild_model_menu()
        self.model_button.grid(row=0, column=1, sticky="ew")
        ttk.Button(model_row, text="Tune model…", command=self._open_model_tuning).grid(row=0, column=2, padx=(10, 0))

        message = ttk.Frame(outer)
        message.grid(row=2, column=0, sticky="ew", pady=(0, 10))
        message.columnconfigure(0, weight=1)
        ttk.Label(message, textvariable=self.status_var).grid(
            row=0, column=0, sticky="w"
        )
        ttk.Button(message, text="Refresh", command=self._refresh_all).grid(
            row=0, column=1, sticky="e"
        )

        services = ttk.LabelFrame(outer, text="Services", padding=10)
        services.grid(row=3, column=0, sticky="ew", pady=(0, 10))
        services.columnconfigure(2, weight=1)

        ttk.Label(services, text="Service", font=("TkDefaultFont", 10, "bold")).grid(
            row=0, column=0, sticky="w"
        )
        ttk.Label(services, text="State", font=("TkDefaultFont", 10, "bold")).grid(
            row=0, column=1, sticky="w", padx=(16, 0)
        )
        ttk.Label(services, text="Detail", font=("TkDefaultFont", 10, "bold")).grid(
            row=0, column=2, sticky="w", padx=(16, 0)
        )

        row = 1
        for service_id, config in self.config_data.services.items():
            if service_id == self.prompt_config.service and service_id != "llama":
                continue
            state_var = tk.StringVar(value="Checking...")
            detail_var = tk.StringVar(value="")
            self.service_vars[service_id] = (state_var, detail_var)

            ttk.Label(services, text=config.display_name).grid(
                row=row, column=0, sticky="w", pady=4
            )
            state_label = ttk.Label(services, textvariable=state_var, width=18)
            state_label.grid(row=row, column=1, sticky="w", padx=(16, 0), pady=4)
            self.state_labels[service_id] = state_label
            ttk.Label(services, textvariable=detail_var).grid(
                row=row, column=2, sticky="ew", padx=(16, 10), pady=4
            )
            ttk.Button(
                services,
                text="Start",
                width=8,
                command=lambda sid=service_id: self._start_service(sid),
            ).grid(row=row, column=3, padx=3)
            ttk.Button(
                services,
                text="Stop",
                width=8,
                command=lambda sid=service_id: self._stop_service(sid),
            ).grid(row=row, column=4, padx=3)
            ttk.Button(
                services,
                text="Open",
                width=8,
                command=lambda sid=service_id: self._open_service(sid),
            ).grid(row=row, column=5, padx=3)
            ttk.Button(
                services,
                text="Logs",
                style="Quiet.TButton",
                width=8,
                command=lambda sid=service_id: self._show_logs(sid),
            ).grid(row=row, column=6, padx=3)
            row += 1

        workspace = ttk.Panedwindow(outer, orient=tk.HORIZONTAL)
        workspace.grid(row=4, column=0, sticky="nsew")
        system_panel = ttk.LabelFrame(workspace, text="System", padding=12)
        prompt_panel = ttk.LabelFrame(workspace, text="Prompt Workshop", padding=12)
        workspace.add(system_panel, weight=2)
        workspace.add(prompt_panel, weight=3)
        self._build_system_panel(system_panel)
        self._build_prompt_panel(prompt_panel)

    def _build_system_panel(self, panel: ttk.LabelFrame) -> None:
        panel.columnconfigure(1, weight=1)
        panel.columnconfigure(2, weight=2)

        ttk.Label(panel, text="GPU", font=("TkDefaultFont", 10, "bold")).grid(
            row=0, column=0, sticky="w", pady=(0, 8)
        )
        ttk.Label(panel, textvariable=self.gpu_name_var).grid(
            row=0, column=1, columnspan=2, sticky="w", pady=(0, 8)
        )

        self.vram_bar = self._metric_row(panel, 1, "VRAM", self.vram_var)
        self.gpu_load_bar = self._metric_row(panel, 2, "GPU load", self.gpu_load_var)
        ttk.Label(panel, text="Temp").grid(row=3, column=0, sticky="w", pady=4)
        ttk.Label(panel, textvariable=self.gpu_temp_var).grid(
            row=3, column=1, columnspan=2, sticky="w", pady=4
        )

        ttk.Separator(panel, orient="horizontal").grid(
            row=4, column=0, columnspan=3, sticky="ew", pady=10
        )
        self.ram_bar = self._metric_row(panel, 5, "RAM", self.ram_var)
        self.cpu_bar = self._metric_row(panel, 6, "CPU", self.cpu_var)

        ttk.Separator(panel, orient="horizontal").grid(
            row=7, column=0, columnspan=3, sticky="ew", pady=10
        )
        ttk.Label(panel, text="Active task").grid(row=8, column=0, sticky="nw", pady=4)
        ttk.Label(panel, textvariable=self.active_task_var, wraplength=230).grid(
            row=8, column=1, columnspan=2, sticky="w", padx=(10, 0), pady=4
        )
        ttk.Label(panel, text="Active model").grid(row=9, column=0, sticky="nw", pady=4)
        ttk.Label(panel, textvariable=self.active_model_var, wraplength=230).grid(
            row=9, column=1, columnspan=2, sticky="w", padx=(10, 0), pady=4
        )
        ttk.Label(panel, text="Prompt helper").grid(row=10, column=0, sticky="nw", pady=4)
        ttk.Label(panel, textvariable=self.prompt_helper_var, wraplength=230).grid(
            row=10, column=1, columnspan=2, sticky="w", padx=(10, 0), pady=4
        )
        ttk.Label(panel, text="Services").grid(row=11, column=0, sticky="nw", pady=4)
        ttk.Label(panel, textvariable=self.service_summary_var, wraplength=230).grid(
            row=11, column=1, columnspan=2, sticky="w", padx=(10, 0), pady=4
        )

    def _metric_row(
        self,
        panel: ttk.LabelFrame,
        row: int,
        label: str,
        variable: tk.StringVar,
    ) -> ttk.Progressbar:
        ttk.Label(panel, text=label).grid(row=row, column=0, sticky="w", pady=4)
        ttk.Label(panel, textvariable=variable, width=17).grid(
            row=row, column=1, sticky="w", padx=(8, 8), pady=4
        )
        bar = ttk.Progressbar(
            panel, orient="horizontal", mode="determinate", maximum=100
        )
        bar.grid(row=row, column=2, sticky="ew", pady=4)
        return bar

    def _build_prompt_panel(self, panel: ttk.LabelFrame) -> None:
        panel.columnconfigure(0, weight=1)
        panel.rowconfigure(1, weight=2)
        panel.rowconfigure(4, weight=3)

        ttk.Label(panel, text="Task:").grid(row=0, column=0, sticky="w")
        task_frame, self.task_text = self._text_editor(panel, height=5)
        task_frame.grid(row=1, column=0, sticky="nsew", pady=(4, 8))

        prompt_tools = ttk.Frame(panel)
        prompt_tools.grid(row=2, column=0, sticky="ew", pady=(0, 10))
        for column in range(4):
            prompt_tools.columnconfigure(column, weight=1)

        self.improve_button = ttk.Button(
            prompt_tools,
            text="Improve Prompt",
            style="Accent.TButton",
            command=self._improve_prompt,
        )
        self.improve_button.grid(row=0, column=0, sticky="ew", padx=(0, 4))
        ttk.Button(prompt_tools, text="Clear Input", style="Quiet.TButton", command=self._clear_input).grid(
            row=0, column=1, sticky="ew", padx=4
        )
        ttk.Button(prompt_tools, text="Clear Output", style="Quiet.TButton", command=self._clear_output).grid(
            row=0, column=2, sticky="ew", padx=4
        )
        ttk.Button(prompt_tools, text="Clear Both", style="Quiet.TButton", command=self._clear_both).grid(
            row=0, column=3, sticky="ew", padx=(4, 0)
        )

        ttk.Label(panel, text="Refined prompt:").grid(row=3, column=0, sticky="w")
        refined_frame, self.refined_text = self._text_editor(panel, height=7)
        refined_frame.grid(row=4, column=0, sticky="nsew", pady=(4, 8))

        actions = ttk.Frame(panel)
        actions.grid(row=5, column=0, sticky="ew")
        actions.columnconfigure(0, weight=1)
        actions.columnconfigure(1, weight=1)
        actions.columnconfigure(2, weight=1)
        ttk.Button(actions, text="Copy", command=self._copy_prompt).grid(
            row=0, column=0, sticky="ew", padx=(0, 4)
        )
        self.agent_button = ttk.Button(
            actions,
            text="Send to Agent",
            command=lambda: self._dispatch_prompt(
                self.prompt_config.agent_profile,
                "Agent",
            ),
        )
        self.agent_button.grid(row=0, column=1, sticky="ew", padx=4)
        self.image_button = ttk.Button(
            actions,
            text="Send to Image",
            command=lambda: self._dispatch_prompt(
                self.prompt_config.image_profile,
                "Image",
            ),
        )
        self.image_button.grid(row=0, column=2, sticky="ew", padx=(4, 0))
        ttk.Label(panel, textvariable=self.prompt_status_var, wraplength=600).grid(
            row=6, column=0, sticky="ew", pady=(8, 0)
        )

    def _text_editor(
        self, parent: ttk.LabelFrame, *, height: int
    ) -> tuple[ttk.Frame, tk.Text]:
        frame = ttk.Frame(parent)
        frame.columnconfigure(0, weight=1)
        frame.rowconfigure(0, weight=1)
        text = tk.Text(frame, height=height, wrap="word", undo=True)
        scrollbar = ttk.Scrollbar(frame, orient="vertical", command=text.yview)
        text.configure(yscrollcommand=scrollbar.set)
        text.grid(row=0, column=0, sticky="nsew")
        scrollbar.grid(row=0, column=1, sticky="ns")
        return frame, text

    def _run_worker(
        self,
        description: str,
        func: Callable[[], object],
        retry_callback: Callable[[str], None] | None = None,
        success_callback: Callable[[object], None] | None = None,
        error_callback: Callable[[Exception], None] | None = None,
    ) -> bool:
        if self.operation_busy:
            self.status_var.set("Another service operation is already in progress")
            return False
        self.operation_busy = True
        self.status_var.set(description)

        def worker() -> None:
            try:
                result = func()
                self.events.put(
                    ("operation_success", (description, result, success_callback))
                )
            except Exception as exc:  # noqa: BLE001 - controlled UI boundary
                self.events.put(
                    (
                        "operation_error",
                        (description, exc, retry_callback, error_callback),
                    )
                )

        threading.Thread(target=worker, daemon=True).start()
        return True

    def _rebuild_model_menu(self) -> None:
        self.model_menu.delete(0, "end")
        self.model_label_to_id.clear()
        values = [_TASK_DEFAULT]
        for model_id, model in self.config_data.models.items():
            label = f"{model.display_name}  [{model_id}]"
            self.model_label_to_id[label] = model_id
            values.append(label)
        for value in values:
            self.model_menu.add_command(
                label=value,
                command=lambda selected=value: self.model_var.set(selected),
            )
        if self.model_var.get() not in values:
            self.model_var.set(_TASK_DEFAULT)

    def _reload_configuration(self) -> None:
        if self.operation_busy or self.prompt_busy:
            self.status_var.set('Wait for the current operation before reloading configuration.')
            return
        try:
            config = load_app_config(self.config_dir)
        except Exception as exc:  # noqa: BLE001 - configuration UI boundary
            messagebox.showerror("Configuration", str(exc), parent=self)
            return
        self.config_data = config
        apply_theme(self, config.settings.appearance)
        self.manager = ServiceManager(config)
        self.prompt_config = config.settings.prompt_workshop
        self.prompt_client = PromptClient(self.prompt_config)
        self.gpu_monitor = create_gpu_monitor(config.settings.gpu_backend)
        self._rebuild_model_menu()
        self._update_capabilities()
        self.status_var.set("Configuration saved and reloaded")
        self._refresh_all()

    def _open_settings(self) -> None:
        SettingsWindow(self, self.config_dir, self.config_data, self._reload_configuration)

    def _open_model_tuning(self, model_id=None):
        if self.operation_busy or self.prompt_busy:
            self.status_var.set('Wait for the current operation before tuning a model.')
            return
        if model_id is None:
            model_id = self._selected_model_override()
        if model_id is None and self.active_task_id in self.config_data.profiles:
            model_id = self._model_for_profile(self.active_task_id)
        if model_id is None and 'chat' in self.config_data.profiles:
            model_id = self._model_for_profile('chat')
        SettingsWindow(self, self.config_dir, self.config_data, self._reload_configuration,
                       initial_tab='tuning', model_id=model_id)

    def _launch_tuned_model(self, model_id):
        for label, mid in self.model_label_to_id.items():
            if mid == model_id:
                self.model_var.set(label)
                break
        service = next((sid for sid, cfg in self.config_data.services.items()
                        if sid != self.prompt_config.service and isinstance(cfg, ProcessServiceConfig)
                        and is_llama_service(cfg)), None)
        if service is not None:
            self._start_service(service)

    def _review_model_before_start(self, requested, model_id):
        for sid in self.manager.dependency_order(requested):
            cfg = self.config_data.services[sid]
            if sid == self.prompt_config.service or not isinstance(cfg, ProcessServiceConfig) or not is_llama_service(cfg):
                continue
            selected = model_id or cfg.default_model
            model = self.config_data.models.get(selected)
            if model is not None and not model.tuning_reviewed:
                self.status_var.set(f'Review launch settings for {model.display_name} before starting.')
                self._open_model_tuning(selected)
                return False
        return True

    def _open_setup_wizard(self) -> None:
        SetupWizard(self, self.config_dir, self.config_data, self._reload_configuration)

    def _rescan_models(self) -> None:
        before = set(self.config_data.models)
        self._reload_configuration()
        after = set(self.config_data.models)
        added = sorted(after - before)
        if added:
            self.status_var.set(f"Model rescan found {len(added)} new model(s)")
        else:
            self.status_var.set(f"Model rescan complete - {len(after)} model(s) available")

    def _task_capability(self, profile_id: str) -> tuple[bool, str]:
        profile = self.config_data.profiles.get(profile_id)
        if profile is None:
            return False, f"{profile_id.title()} is not configured."
        for service_id in profile.services:
            if service_id not in self.config_data.services:
                return False, f"Required service '{service_id}' is not configured."
        if profile_id in {"coding", "chat", "agent"}:
            model_id = profile.default_model
            model = self.config_data.models.get(model_id) if model_id else None
            if model is None or not model.complete:
                return False, f"{profile.display_name} needs a complete default model."
            if not model.tuning_reviewed:
                return False, f"{model.display_name} needs launch configuration."
        if profile_id == "agent":
            if not profile.harness:
                return False, "Agent unavailable - no Agent harness is configured."
            harness = self.config_data.harnesses.get(profile.harness)
            if harness is None:
                return False, "Agent unavailable - the selected harness is missing."
            missing = [sid for sid in harness.services if sid not in self.config_data.services]
            if missing:
                return False, "Agent harness has missing service(s): " + ", ".join(missing)
        return True, "Ready"

    def _prompt_helper_capability(self) -> tuple[bool, str]:
        prompt = self.prompt_config
        if not prompt.enabled:
            return False, "Prompt Helper is disabled in Preferences."
        if prompt.service not in self.config_data.services:
            return False, "Prompt Helper service is not configured."
        if prompt.model_strategy == "smallest":
            model = smallest_complete_model(self.config_data.models)
            if model is None:
                return False, "No complete GGUF model is available for Prompt Helper."
            return True, "Ready"
        if prompt.model_strategy == "manual":
            model = self.config_data.models.get(prompt.preferred_model) if prompt.preferred_model else None
            if model is None or not model.complete:
                return False, "Choose a complete Prompt Helper model in Preferences."
            return True, "Ready"
        service = self.config_data.services.get(prompt.service)
        default_model = service.default_model if isinstance(service, ProcessServiceConfig) else None
        model = self.config_data.models.get(default_model) if default_model else None
        if model is None or not model.complete:
            return False, "Prompt Helper has no usable model."
        return True, "Ready"

    def _update_capabilities(self) -> None:
        self.capability_reasons = {}
        for profile_id, button in self.task_buttons.items():
            available, reason = self._task_capability(profile_id)
            self.capability_reasons[profile_id] = reason
            button.configure(state="normal" if available else "disabled")
        prompt_available, prompt_reason = self._prompt_helper_capability()
        self.capability_reasons["prompt_helper"] = prompt_reason
        if hasattr(self, "improve_button"):
            self.improve_button.configure(state="normal" if prompt_available else "disabled")
        agent_available, _ = self._task_capability(self.prompt_config.agent_profile)
        image_available, _ = self._task_capability(self.prompt_config.image_profile)
        if hasattr(self, "agent_button"):
            self.agent_button.configure(state="normal" if agent_available else "disabled")
        if hasattr(self, "image_button"):
            self.image_button.configure(state="normal" if image_available else "disabled")
        if not prompt_available and hasattr(self, "prompt_status_var"):
            self.prompt_status_var.set(prompt_reason)

    def _selected_model_override(self) -> str | None:
        value = self.model_var.get().strip()
        if not value or value == _TASK_DEFAULT:
            return None
        return self.model_label_to_id.get(value)

    def _model_display(self, model_id: str | None) -> str:
        if model_id is None:
            return "Unknown model"
        model = self.config_data.models.get(model_id)
        return model.display_name if model is not None else model_id

    def _model_for_profile(self, profile_id: str) -> str | None:
        if profile_id == self.prompt_config.startup_profile:
            return self._prompt_helper_model_id()
        override = self._selected_model_override()
        if override is not None:
            return override
        return self.config_data.profiles[profile_id].default_model

    def _model_for_service(self, service_id: str) -> str | None:
        if service_id == self.prompt_config.service:
            helper_model = self._prompt_helper_model_id()
            if helper_model is not None:
                return helper_model
        override = self._selected_model_override()
        if override is not None:
            return override
        config = self.config_data.services[service_id]
        if isinstance(config, ProcessServiceConfig) and config.uses_model:
            return config.default_model
        if service_id == self.prompt_config.service:
            profile = self.config_data.profiles.get(self.prompt_config.startup_profile)
            if profile is not None:
                return profile.default_model
        return None

    def _start_profile(
        self,
        profile_id: str,
        *,
        stop_conflicts: bool = False,
        replace_model: bool = False,
        open_interface: bool = True,
        on_ready: Callable[[object], None] | None = None,
        on_failure: Callable[[Exception], None] | None = None,
    ) -> bool:
        available, reason = self._task_capability(profile_id)
        if not available and profile_id != self.prompt_config.startup_profile:
            self.status_var.set(reason)
            if on_failure is not None:
                on_failure(RuntimeError(reason))
            return False
        profile = self.config_data.profiles[profile_id]
        model = (
            self._model_for_profile(profile_id)
            if any(
                isinstance(self.config_data.services[sid], ProcessServiceConfig)
                and self.config_data.services[sid].uses_model
                for sid in self.manager.dependency_order(profile.services)
            )
            else None
        )
        if not self._review_model_before_start(profile.services, model):
            return False
        model_text = f" with {self._model_display(model)}" if model else ""

        def operation():
            return self.manager.start_profile(
                profile_id,
                open_interface=open_interface,
                stop_conflicts=stop_conflicts,
                replace_model=replace_model,
                model_id=model,
            )

        def retry(reason: str) -> None:
            self._start_profile(
                profile_id,
                stop_conflicts=stop_conflicts or reason == "conflict",
                replace_model=replace_model or reason == "model",
                open_interface=open_interface,
                on_ready=on_ready,
                on_failure=on_failure,
            )

        def completed(results: object) -> None:
            self._set_active_task(profile_id)
            if on_ready is not None:
                on_ready(results)

        return self._run_worker(
            f"Starting {profile.display_name}{model_text}...",
            operation,
            retry,
            completed,
            on_failure,
        )

    def _start_service(
        self,
        service_id: str,
        *,
        stop_conflicts: bool = False,
        replace_model: bool = False,
    ) -> None:
        config = self.config_data.services[service_id]
        model = self._model_for_service(service_id)
        if not self._review_model_before_start([service_id], model):
            return

        def operation():
            return self.manager.start_service(
                service_id,
                stop_conflicts=stop_conflicts,
                replace_model=replace_model,
                model_id=model,
            )

        def retry(reason: str) -> None:
            self._start_service(
                service_id,
                stop_conflicts=stop_conflicts or reason == "conflict",
                replace_model=replace_model or reason == "model",
            )

        model_text = f" with {self._model_display(model)}" if model else ""
        self._run_worker(
            f"Starting {config.display_name}{model_text}...",
            operation,
            retry,
            lambda _results: self._set_active_task("custom"),
        )

    def _stop_service(self, service_id: str) -> None:
        config = self.config_data.services[service_id]
        self._run_worker(
            f"Stopping {config.display_name}...",
            lambda: [(service_id, self.manager.stop_service(service_id))],
            success_callback=lambda _results: self._set_active_task("custom"),
        )

    def _stop_all(self) -> None:
        self._run_worker(
            "Stopping launcher-owned AI services...",
            self.manager.stop_all,
            success_callback=lambda _results: self._set_active_task(None),
        )

    def _open_service(self, service_id: str) -> None:
        if not self.manager.get(service_id).open():
            self.status_var.set(
                f"No openable interface configured for {self.config_data.services[service_id].display_name}"
            )

    def _show_logs(self, service_id: str) -> None:
        try:
            lines = self.manager.get(service_id).logs(120)
        except Exception as exc:  # noqa: BLE001 - controlled UI boundary
            messagebox.showerror("Logs", str(exc), parent=self)
            return
        window = tk.Toplevel(self)
        window.title(
            f"{self.config_data.services[service_id].display_name} - Recent logs"
        )
        window.geometry("900x520")
        text = tk.Text(window, wrap="none")
        text.pack(fill="both", expand=True)
        text.insert("1.0", "\n".join(lines))
        text.configure(state="disabled")
        style_classic(window)

    def _set_active_task(self, profile_id: str | None) -> None:
        self.active_task_id = profile_id
        for task, button in self.task_buttons.items():
            button.configure(style='Active.Task.TButton' if task == profile_id else 'Task.TButton')
        if profile_id is None:
            self.active_task_var.set("None")
        elif profile_id == "custom":
            self.active_task_var.set("Custom service state")
        else:
            profile = self.config_data.profiles.get(profile_id)
            self.active_task_var.set(profile.display_name if profile else profile_id)

    def _refresh_async(self) -> None:
        if self.status_poll_in_flight:
            return
        self.status_poll_in_flight = True

        def worker() -> None:
            snapshot = {}
            for service_id in self.config_data.services:
                try:
                    snapshot[service_id] = self.manager.get(service_id).status()
                except Exception as exc:  # noqa: BLE001 - per-service UI boundary
                    snapshot[service_id] = exc
            self.events.put(("status_snapshot", snapshot))

        threading.Thread(target=worker, daemon=True).start()

    def _system_refresh_async(self) -> None:
        if self.monitor_poll_in_flight:
            return
        self.monitor_poll_in_flight = True

        def worker() -> None:
            try:
                if self.gpu_monitor is None:
                    gpu = GpuStatus(False, "none", detail="GPU monitoring disabled")
                else:
                    gpu = self.gpu_monitor.status()
            except Exception as exc:  # noqa: BLE001 - defensive adapter boundary
                gpu = GpuStatus(
                    False, "unknown", detail=f"GPU monitoring failed: {exc}"
                )
            try:
                host = self.host_monitor.status()
            except Exception as exc:  # noqa: BLE001 - defensive adapter boundary
                host = HostStatus(detail=f"Host monitoring failed: {exc}")
            self.events.put(("system_snapshot", (gpu, host)))

        threading.Thread(target=worker, daemon=True).start()

    def _refresh_all(self) -> None:
        self._refresh_async()
        self._system_refresh_async()

    def _poll_status(self) -> None:
        self._refresh_async()
        interval = max(1000, int(self.config_data.settings.monitoring_interval * 1000))
        self.after(interval, self._poll_status)

    def _poll_system(self) -> None:
        self._system_refresh_async()
        interval = max(1000, int(self.config_data.settings.monitoring_interval * 1000))
        self.after(interval, self._poll_system)

    def _friendly_service_status(self, service_id: str, status) -> tuple[str, str]:
        state = _STATE_LABELS.get(status.state, status.state.value)
        if status.state == ServiceState.STOPPED:
            return state, "Not running"
        if status.state == ServiceState.EXTERNAL:
            return state, "Running outside AI Control Centre"
        if status.state == ServiceState.RUNNING_NOT_READY:
            if service_id == self.prompt_config.service:
                model_id = self.manager.running_model_id(service_id)
                return "Loading model", self._model_display(model_id)
            return "Starting", "Waiting for readiness"
        if status.state == ServiceState.READY:
            if service_id == self.prompt_config.service:
                model_id = self.manager.running_model_id(service_id)
                detail = self._model_display(model_id)
                if status.pid is not None:
                    detail += f" | PID {status.pid}"
                return state, detail
            return state, "Ready for use"
        if status.state == ServiceState.ERROR:
            return state, status.detail
        return state, status.detail

    def _retry_with_permission(
        self,
        exc: Exception,
        retry_callback: Callable[[str], None] | None,
    ) -> tuple[bool, bool]:
        if retry_callback is None:
            return False, False
        text = str(exc)
        if "Retry with --stop-conflicts" in text:
            approved = messagebox.askyesno(
                "Resource conflict",
                f"{text}\n\nStop the launcher-owned conflicting service(s) and continue?",
                parent=self,
            )
            if approved:
                retry_callback("conflict")
            return True, approved
        if "Retry with --replace-model" in text:
            approved = messagebox.askyesno(
                "Change loaded model",
                f"{text}\n\nStop affected launcher-owned services, load the requested model and continue?",
                parent=self,
            )
            if approved:
                retry_callback("model")
            return True, approved
        return False, False

    def _task_input(self) -> str:
        return self.task_text.get("1.0", "end-1c").strip()

    def _selected_prompt(self) -> str:
        return select_prompt(
            self._task_input(),
            self.refined_text.get("1.0", "end-1c"),
        )

    def _clear_input(self) -> None:
        self.task_text.delete("1.0", "end")
        self.prompt_status_var.set("Input cleared.")

    def _clear_output(self) -> None:
        self.refined_text.delete("1.0", "end")
        self.prompt_status_var.set("Refined prompt cleared.")

    def _clear_both(self) -> None:
        self.task_text.delete("1.0", "end")
        self.refined_text.delete("1.0", "end")
        self.prompt_status_var.set("Prompt Workshop cleared.")

    def _prompt_helper_model_id(self) -> str | None:
        if self.prompt_config.model_strategy == "smallest":
            model = smallest_complete_model(self.config_data.models)
            return model.id if model is not None else None
        if self.prompt_config.preferred_model is not None:
            return self.prompt_config.preferred_model
        service = self.config_data.services[self.prompt_config.service]
        if isinstance(service, ProcessServiceConfig):
            return service.default_model
        return None

    def _set_prompt_busy(self, busy: bool) -> None:
        self.prompt_busy = busy
        if busy:
            self.improve_button.configure(state="disabled")
            self.agent_button.configure(state="disabled")
            self.image_button.configure(state="disabled")
        else:
            self._update_capabilities()

    def _copy_to_clipboard(self, prompt: str) -> bool:
        try:
            self.clipboard_clear()
            self.clipboard_append(prompt)
            self.update_idletasks()
            return True
        except tk.TclError as exc:
            self.prompt_status_var.set(f"Could not copy prompt: {exc}")
            return False

    def _copy_prompt(self) -> None:
        prompt = self._selected_prompt()
        if not prompt:
            self.prompt_status_var.set("Enter a task or refined prompt to copy.")
            return
        if self._copy_to_clipboard(prompt):
            self.prompt_status_var.set("Prompt copied to clipboard.")

    def _improve_prompt(self) -> None:
        available, reason = self._prompt_helper_capability()
        if not available:
            self.prompt_status_var.set(reason)
            return
        prompt = self._task_input()
        if not prompt:
            self.prompt_status_var.set("Enter a task to improve.")
            return
        if self.prompt_busy:
            self.prompt_status_var.set("A prompt operation is already in progress.")
            return

        self._set_prompt_busy(True)
        self.prompt_status_var.set("Checking local LLM...")

        def worker() -> None:
            try:
                status = self.manager.get(self.prompt_config.service).status()
                action = prompt_service_action(status.state)
                if action == "use":
                    refined = self.prompt_client.refine(prompt)
                    self.events.put(("prompt_success", refined))
                elif action == "wait":
                    self.events.put(("prompt_wait", None))
                elif action == "start":
                    self.events.put(("prompt_needs_start", prompt))
                else:
                    raise RuntimeError(
                        f"{self.config_data.services[self.prompt_config.service].display_name} is unavailable: "
                        f"{status.detail}"
                    )
            except Exception as exc:  # noqa: BLE001 - controlled prompt boundary
                self.events.put(("prompt_error", exc))

        threading.Thread(target=worker, daemon=True).start()

    def _submit_refine_request(self, prompt: str) -> None:
        self.prompt_status_var.set("Improving prompt...")

        def worker() -> None:
            try:
                self.events.put(("prompt_success", self.prompt_client.refine(prompt)))
            except Exception as exc:  # noqa: BLE001 - controlled prompt boundary
                self.events.put(("prompt_error", exc))

        threading.Thread(target=worker, daemon=True).start()

    def _start_chat_for_prompt(self, prompt: str) -> None:
        if self.operation_busy:
            self._set_prompt_busy(False)
            self.prompt_status_var.set(
                "Another service operation is already in progress."
            )
            return
        profile = self.config_data.profiles[self.prompt_config.startup_profile]
        self.prompt_status_var.set(
            f"Starting {profile.display_name} for prompt refinement..."
        )

        def failed(exc: Exception) -> None:
            self._set_prompt_busy(False)
            self.prompt_status_var.set(f"Could not prepare the local LLM: {exc}")

        started = self._start_profile(
            self.prompt_config.startup_profile,
            open_interface=False,
            on_ready=lambda _results: self._submit_refine_request(prompt),
            on_failure=failed,
        )
        if not started:
            failed(RuntimeError("another service operation is already in progress"))

    def _dispatch_prompt(self, profile_id: str, label: str) -> None:
        prompt = self._selected_prompt()
        if not prompt:
            self.prompt_status_var.set(
                f"Enter a task or refined prompt to send to {label}."
            )
            return
        if self.operation_busy or self.prompt_busy:
            self.prompt_status_var.set("Another operation is already in progress.")
            return
        if not self._copy_to_clipboard(prompt):
            return

        self.prompt_status_var.set(f"Preparing {label} environment...")

        def ready(_results: object) -> None:
            self.prompt_status_var.set(
                f"{label} environment ready. Prompt copied to clipboard."
            )

        def failed(exc: Exception) -> None:
            self.prompt_status_var.set(
                f"Prompt copied, but {label} could not be prepared: {exc}"
            )

        if not self._start_profile(profile_id, on_ready=ready, on_failure=failed):
            failed(RuntimeError("another service operation is already in progress"))

    def _drain_events(self) -> None:
        try:
            while True:
                kind, payload = self.events.get_nowait()
                if kind == "status_snapshot":
                    self.status_poll_in_flight = False
                    self._apply_status_snapshot(payload)
                elif kind == "system_snapshot":
                    self.monitor_poll_in_flight = False
                    gpu, host = payload
                    self._apply_system(gpu, host)
                elif kind == "operation_success":
                    description, results, success_callback = payload
                    self.operation_busy = False
                    if results:
                        summary = "; ".join(
                            f"{self.config_data.services[service_id].display_name}: {status.state.value}"
                            for service_id, status in results
                        )
                        self.status_var.set(summary)
                    else:
                        self.status_var.set(description.replace("...", " complete"))
                    if success_callback is not None:
                        success_callback(results)
                    self._refresh_all()
                elif kind == "operation_error":
                    description, exc, retry_callback, error_callback = payload
                    self.operation_busy = False
                    handled, retrying = self._retry_with_permission(exc, retry_callback)
                    if not retrying and error_callback is not None:
                        error_callback(exc)
                    if handled:
                        if not retrying:
                            self.status_var.set("Operation cancelled")
                    else:
                        self._set_active_task('custom')
                        self.status_var.set(f"Could not complete operation: {exc}")
                        messagebox.showerror("AI Control Centre", str(exc), parent=self)
                    self._refresh_all()
                elif kind == "prompt_success":
                    self.refined_text.delete("1.0", "end")
                    self.refined_text.insert("1.0", str(payload))
                    self._set_prompt_busy(False)
                    self.prompt_status_var.set("Prompt improved.")
                    if not self.prompt_config.keep_loaded:
                        helper_id = self.prompt_config.service
                        def stop_helper() -> None:
                            try:
                                self.manager.stop_service(helper_id)
                            except Exception:
                                pass
                            self.events.put(("prompt_helper_stopped", None))
                        threading.Thread(target=stop_helper, daemon=True).start()
                elif kind == "prompt_helper_stopped":
                    self.prompt_status_var.set("Prompt improved. Prompt Helper unloaded.")
                    self._refresh_all()
                elif kind == "prompt_wait":
                    self._set_prompt_busy(False)
                    self.prompt_status_var.set(
                        "Prompt helper is still loading. Try again shortly."
                    )
                elif kind == "prompt_needs_start":
                    # Prompt refinement is intentionally one-click: start the lightweight
                    # CPU-first helper automatically when it is not already running.
                    self._start_chat_for_prompt(str(payload))
                elif kind == "prompt_error":
                    self._set_prompt_busy(False)
                    self.prompt_status_var.set(f"Could not improve prompt: {payload}")
        except queue.Empty:
            pass
        self.after(100, self._drain_events)

    def _apply_status_snapshot(self, snapshot: dict[str, object]) -> None:
        counts: dict[str, int] = {}
        for service_id, value in snapshot.items():
            if service_id not in self.service_vars:
                continue
            state_var, detail_var = self.service_vars[service_id]
            if isinstance(value, Exception):
                state_var.set("Error")
                detail_var.set(str(value))
                counts["error"] = counts.get("error", 0) + 1
            else:
                state, detail = self._friendly_service_status(service_id, value)
                state_var.set(state)
                label = self.state_labels.get(service_id)
                if label is not None:
                    name = state if state in {'Ready', 'Starting', 'Error', 'Stopped', 'External'} else 'Stopped'
                    label.configure(style=name+'.TLabel')
                detail_var.set(detail)
                key = state.lower()
                counts[key] = counts.get(key, 0) + 1

        parts = [f"{count} {state}" for state, count in sorted(counts.items())]
        self.service_summary_var.set(
            ", ".join(parts) if parts else "No services configured"
        )

        # Keep the primary task model distinct from the lightweight Prompt Helper.
        main_service_id = next(
            (
                sid
                for sid, config in self.config_data.services.items()
                if sid != self.prompt_config.service
                and isinstance(config, ProcessServiceConfig)
                and config.uses_model
            ),
            None,
        )
        main_status = snapshot.get(main_service_id) if main_service_id else None
        if isinstance(main_status, Exception) or main_status is None:
            self.active_model_var.set("Unknown" if main_service_id else "None")
        elif main_status.state == ServiceState.EXTERNAL:
            self.active_model_var.set("External model (unknown)")
        elif main_status.state in {ServiceState.READY, ServiceState.RUNNING_NOT_READY}:
            model_id = self.manager.running_model_id(main_service_id)
            self.active_model_var.set(self._model_display(model_id))
        else:
            self.active_model_var.set("None")

        helper_status = snapshot.get(self.prompt_config.service)
        if not self.prompt_config.enabled:
            self.prompt_helper_var.set("Disabled")
        elif isinstance(helper_status, Exception) or helper_status is None:
            self.prompt_helper_var.set("Unavailable")
        elif helper_status.state in {ServiceState.READY, ServiceState.RUNNING_NOT_READY}:
            model_id = self.manager.running_model_id(self.prompt_config.service)
            mode = self.manager.prompt_helper_processing_mode().upper()
            state_text = "Ready" if helper_status.state == ServiceState.READY else "Loading"
            self.prompt_helper_var.set(f"{self._model_display(model_id)} • {mode} • {state_text}")
        elif helper_status.state == ServiceState.EXTERNAL:
            self.prompt_helper_var.set("External")
        else:
            self.prompt_helper_var.set("Stopped")

    def _apply_system(self, gpu: GpuStatus, host: HostStatus) -> None:
        if not gpu.available:
            self.gpu_name_var.set(f"Unavailable ({gpu.detail})")
            self.vram_var.set("Unavailable")
            self.gpu_load_var.set("Unavailable")
            self.gpu_temp_var.set("Unavailable")
            self.vram_bar["value"] = 0
            self.gpu_load_bar["value"] = 0
        else:
            self.gpu_name_var.set(gpu.name or "GPU")
            if gpu.memory_used_mib is not None and gpu.memory_total_mib:
                self.vram_var.set(
                    f"{gpu.memory_used_mib / 1024:.1f} / {gpu.memory_total_mib / 1024:.1f} GB"
                )
                self.vram_bar["value"] = max(
                    0,
                    min(100, gpu.memory_used_mib / gpu.memory_total_mib * 100),
                )
            else:
                self.vram_var.set("Unavailable")
                self.vram_bar["value"] = 0
            if gpu.utilization_percent is None:
                self.gpu_load_var.set("Unavailable")
                self.gpu_load_bar["value"] = 0
            else:
                value = max(0, min(100, gpu.utilization_percent))
                self.gpu_load_var.set(f"{value}%")
                self.gpu_load_bar["value"] = value
            self.gpu_temp_var.set(
                f"{gpu.temperature_c} C"
                if gpu.temperature_c is not None
                else "Unavailable"
            )

        if host.memory_used_mib is None or host.memory_total_mib is None:
            self.ram_var.set("Unavailable")
            self.ram_bar["value"] = 0
        else:
            self.ram_var.set(
                f"{host.memory_used_mib / 1024:.1f} / {host.memory_total_mib / 1024:.1f} GB"
            )
            self.ram_bar["value"] = host.memory_percent or 0

        if host.cpu_utilization_percent is None:
            self.cpu_var.set("Checking..." if host.detail == "OK" else "Unavailable")
            self.cpu_bar["value"] = 0
        else:
            value = max(0.0, min(100.0, host.cpu_utilization_percent))
            self.cpu_var.set(f"{value:.0f}%")
            self.cpu_bar["value"] = value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="AI Control Centre graphical interface"
    )
    parser.add_argument(
        "--config-dir",
        type=Path,
        default=Path("~/.config/ai-control-centre").expanduser(),
    )
    parser.add_argument("--setup", action="store_true", help="Open the setup wizard after launch")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        app = ControlCentreWindow(args.config_dir.expanduser(), force_setup=args.setup)
    except (ConfigError, FileNotFoundError, RuntimeError, ValueError) as exc:
        print(f"ERROR: {exc}")
        return 1
    app.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
