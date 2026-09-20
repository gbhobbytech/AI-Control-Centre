from __future__ import annotations

import argparse
import queue
import threading
from pathlib import Path
from typing import Callable

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
from .monitoring import GpuStatus, create_gpu_monitor


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
    def __init__(self, config_dir: Path):
        super().__init__()
        self.title("AI Control Centre")
        self.geometry("1040x670")
        self.minsize(900, 580)

        self.config_dir = config_dir
        self.config_data = load_app_config(config_dir)
        self.manager = ServiceManager(self.config_data)
        self.gpu_monitor = create_gpu_monitor(self.config_data.settings.gpu_backend)
        self.events: queue.Queue[tuple[str, object]] = queue.Queue()
        self.busy = False

        self.status_var = tk.StringVar(value="Ready - choose a task")
        self.gpu_var = tk.StringVar(value="GPU: checking...")
        self.model_var = tk.StringVar(value=_TASK_DEFAULT)
        self.service_vars: dict[str, tuple[tk.StringVar, tk.StringVar]] = {}
        self.model_label_to_id: dict[str, str] = {}

        self._build_ui()
        self.after(100, self._drain_events)
        self.after(150, self._poll_status)
        self.after(250, self._poll_gpu)

    def _build_ui(self) -> None:
        outer = ttk.Frame(self, padding=16)
        outer.pack(fill="both", expand=True)
        outer.columnconfigure(0, weight=1)
        outer.rowconfigure(3, weight=1)

        header = ttk.Frame(outer)
        header.grid(row=0, column=0, sticky="ew")
        header.columnconfigure(0, weight=1)
        ttk.Label(header, text="AI Control Centre", font=("TkDefaultFont", 18, "bold")).grid(
            row=0, column=0, sticky="w"
        )
        ttk.Label(header, textvariable=self.gpu_var).grid(row=0, column=1, sticky="e")

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
            button.grid(row=0, column=column, sticky="ew", padx=(0 if column == 0 else 4, 4))

        ttk.Button(task, text="Stop All", command=self._stop_all).grid(
            row=0, column=4, sticky="ew", padx=(4, 0)
        )

        model_row = ttk.Frame(task)
        model_row.grid(row=1, column=0, columnspan=5, sticky="ew", pady=(12, 0))
        model_row.columnconfigure(1, weight=1)
        ttk.Label(model_row, text="LLM override:").grid(row=0, column=0, sticky="w", padx=(0, 8))

        model_values = [_TASK_DEFAULT]
        for model_id, model in self.config_data.models.items():
            label = f"{model.display_name}  [{model_id}]"
            self.model_label_to_id[label] = model_id
            model_values.append(label)
        # Use a menu button instead of ttk.Combobox. On some Linux/Tk
        # combinations (including the Pop!_OS reference machine), the
        # native combobox popdown can be positioned at the top-left of
        # the screen rather than beneath the widget. A menu button keeps
        # the selector anchored correctly while preserving the same
        # StringVar-based model-selection behaviour.
        self.model_button = ttk.Menubutton(
            model_row,
            textvariable=self.model_var,
            direction="below",
        )
        self.model_menu = tk.Menu(self.model_button, tearoff=False)
        for value in model_values:
            self.model_menu.add_command(
                label=value,
                command=lambda selected=value: self.model_var.set(selected),
            )
        self.model_button.configure(menu=self.model_menu)
        self.model_button.grid(row=0, column=1, sticky="ew")
        ttk.Label(model_row, text="Leave on Task default for automatic model choice.").grid(
            row=0, column=2, sticky="e", padx=(10, 0)
        )

        message = ttk.Frame(outer)
        message.grid(row=2, column=0, sticky="ew", pady=(0, 10))
        message.columnconfigure(0, weight=1)
        ttk.Label(message, textvariable=self.status_var).grid(row=0, column=0, sticky="w")
        ttk.Button(message, text="Refresh", command=self._refresh_async).grid(row=0, column=1, sticky="e")

        services = ttk.LabelFrame(outer, text="Services", padding=10)
        services.grid(row=3, column=0, sticky="nsew")
        services.columnconfigure(2, weight=1)

        ttk.Label(services, text="Service", font=("TkDefaultFont", 10, "bold")).grid(row=0, column=0, sticky="w")
        ttk.Label(services, text="State", font=("TkDefaultFont", 10, "bold")).grid(
            row=0, column=1, sticky="w", padx=(16, 0)
        )
        ttk.Label(services, text="Detail", font=("TkDefaultFont", 10, "bold")).grid(
            row=0, column=2, sticky="w", padx=(16, 0)
        )

        row = 1
        for service_id, config in self.config_data.services.items():
            state_var = tk.StringVar(value="Checking...")
            detail_var = tk.StringVar(value="")
            self.service_vars[service_id] = (state_var, detail_var)

            ttk.Label(services, text=config.display_name).grid(row=row, column=0, sticky="w", pady=5)
            ttk.Label(services, textvariable=state_var, width=18).grid(
                row=row, column=1, sticky="w", padx=(16, 0), pady=5
            )
            ttk.Label(services, textvariable=detail_var).grid(
                row=row, column=2, sticky="ew", padx=(16, 10), pady=5
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
                width=8,
                command=lambda sid=service_id: self._show_logs(sid),
            ).grid(row=row, column=6, padx=3)
            row += 1

    def _run_worker(
        self,
        description: str,
        func: Callable[[], object],
        retry_callback: Callable[[str], None] | None = None,
    ) -> None:
        if self.busy:
            self.status_var.set("Another operation is already in progress")
            return
        self.busy = True
        self.status_var.set(description)

        def worker() -> None:
            try:
                result = func()
                self.events.put(("success", (description, result)))
            except Exception as exc:  # UI boundary: show controlled backend errors
                self.events.put(("error", (description, exc, retry_callback)))

        threading.Thread(target=worker, daemon=True).start()

    def _selected_model_override(self) -> str | None:
        value = self.model_var.get().strip()
        if not value or value == _TASK_DEFAULT:
            return None
        return self.model_label_to_id.get(value)

    def _model_display(self, model_id: str | None) -> str:
        if model_id is None:
            return "configured model"
        model = self.config_data.models.get(model_id)
        return model.display_name if model is not None else model_id

    def _model_for_profile(self, profile_id: str) -> str | None:
        override = self._selected_model_override()
        if override is not None:
            return override
        profile = self.config_data.profiles[profile_id]
        return profile.default_model

    def _model_for_service(self, service_id: str) -> str | None:
        override = self._selected_model_override()
        if override is not None:
            return override
        config = self.config_data.services[service_id]
        if isinstance(config, ProcessServiceConfig) and config.uses_model:
            return config.default_model
        if service_id == "computer_dmz":
            agent = self.config_data.profiles.get("agent")
            if agent is not None:
                return agent.default_model
        return None

    def _start_profile(
        self,
        profile_id: str,
        *,
        stop_conflicts: bool = False,
        replace_model: bool = False,
    ) -> None:
        profile = self.config_data.profiles[profile_id]
        model = self._model_for_profile(profile_id) if any(
            isinstance(self.config_data.services[sid], ProcessServiceConfig)
            and self.config_data.services[sid].uses_model
            for sid in self.manager._dependency_order(profile.services)
        ) else None
        model_text = f" with {self._model_display(model)}" if model else ""

        def operation():
            return self.manager.start_profile(
                profile_id,
                open_interface=True,
                stop_conflicts=stop_conflicts,
                replace_model=replace_model,
                model_id=model,
            )

        def retry(reason: str) -> None:
            self._start_profile(
                profile_id,
                stop_conflicts=stop_conflicts or reason == "conflict",
                replace_model=replace_model or reason == "model",
            )

        self._run_worker(
            f"Starting {profile.display_name}{model_text}...",
            operation,
            retry,
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
        self._run_worker(f"Starting {config.display_name}{model_text}...", operation, retry)

    def _stop_service(self, service_id: str) -> None:
        config = self.config_data.services[service_id]
        self._run_worker(
            f"Stopping {config.display_name}...",
            lambda: [(service_id, self.manager.stop_service(service_id))],
        )

    def _stop_all(self) -> None:
        self._run_worker("Stopping launcher-owned AI services...", self.manager.stop_all)

    def _open_service(self, service_id: str) -> None:
        if not self.manager.get(service_id).open():
            self.status_var.set(f"No openable interface configured for {self.config_data.services[service_id].display_name}")

    def _show_logs(self, service_id: str) -> None:
        try:
            lines = self.manager.get(service_id).logs(120)
        except Exception as exc:
            messagebox.showerror("Logs", str(exc), parent=self)
            return
        window = tk.Toplevel(self)
        window.title(f"{self.config_data.services[service_id].display_name} - Recent logs")
        window.geometry("900x520")
        text = tk.Text(window, wrap="none")
        text.pack(fill="both", expand=True)
        text.insert("1.0", "\n".join(lines))
        text.configure(state="disabled")

    def _refresh_async(self) -> None:
        def worker() -> None:
            snapshot = {}
            for service_id in self.config_data.services:
                try:
                    snapshot[service_id] = self.manager.get(service_id).status()
                except Exception as exc:
                    snapshot[service_id] = exc
            self.events.put(("status", snapshot))

        threading.Thread(target=worker, daemon=True).start()

    def _poll_status(self) -> None:
        self._refresh_async()
        self.after(max(1000, int(self.config_data.settings.monitoring_interval * 1000)), self._poll_status)

    def _gpu_refresh_async(self) -> None:
        def worker() -> None:
            if self.gpu_monitor is None:
                value = GpuStatus(False, "none", detail="GPU monitoring disabled")
            else:
                value = self.gpu_monitor.status()
            self.events.put(("gpu", value))

        threading.Thread(target=worker, daemon=True).start()

    def _poll_gpu(self) -> None:
        self._gpu_refresh_async()
        self.after(max(1000, int(self.config_data.settings.monitoring_interval * 1000)), self._poll_gpu)

    def _friendly_service_status(self, service_id: str, status) -> tuple[str, str]:
        state = _STATE_LABELS.get(status.state, status.state.value)
        if status.state == ServiceState.STOPPED:
            return state, "Not running"
        if status.state == ServiceState.EXTERNAL:
            return state, "Running outside AI Control Centre"
        if status.state == ServiceState.RUNNING_NOT_READY:
            if service_id == "llama":
                model_id = self.manager.running_model_id(service_id)
                return "Loading model", self._model_display(model_id)
            return "Starting", "Waiting for readiness"
        if status.state == ServiceState.READY:
            if service_id == "llama":
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
    ) -> bool:
        if retry_callback is None:
            return False
        text = str(exc)
        if "Retry with --stop-conflicts" in text:
            if messagebox.askyesno(
                "Resource conflict",
                f"{text}\n\nStop the launcher-owned conflicting service(s) and continue?",
                parent=self,
            ):
                retry_callback("conflict")
            return True
        if "Retry with --replace-model" in text:
            if messagebox.askyesno(
                "Change loaded model",
                f"{text}\n\nStop affected launcher-owned services, load the requested model and continue?",
                parent=self,
            ):
                retry_callback("model")
            return True
        return False

    def _drain_events(self) -> None:
        try:
            while True:
                kind, payload = self.events.get_nowait()
                if kind == "status":
                    for service_id, value in payload.items():
                        state_var, detail_var = self.service_vars[service_id]
                        if isinstance(value, Exception):
                            state_var.set("Error")
                            detail_var.set(str(value))
                        else:
                            state, detail = self._friendly_service_status(service_id, value)
                            state_var.set(state)
                            detail_var.set(detail)
                elif kind == "gpu":
                    self._apply_gpu(payload)
                elif kind == "success":
                    description, results = payload
                    self.busy = False
                    if results:
                        summary = "; ".join(
                            f"{self.config_data.services[service_id].display_name}: {status.state.value}"
                            for service_id, status in results
                        )
                        self.status_var.set(summary)
                    else:
                        self.status_var.set(description.replace("...", " complete"))
                    self._refresh_async()
                    self._gpu_refresh_async()
                elif kind == "error":
                    description, exc, retry_callback = payload
                    self.busy = False
                    if not self._retry_with_permission(exc, retry_callback):
                        self.status_var.set(f"Could not complete operation: {exc}")
                        messagebox.showerror("AI Control Centre", str(exc), parent=self)
                    self._refresh_async()
                    self._gpu_refresh_async()
        except queue.Empty:
            pass
        self.after(100, self._drain_events)

    def _apply_gpu(self, status: GpuStatus) -> None:
        if not status.available:
            self.gpu_var.set(f"GPU: unavailable ({status.detail})")
            return
        self.gpu_var.set(
            f"{status.name} | VRAM {status.memory_used_mib}/{status.memory_total_mib} MiB | "
            f"GPU {status.utilization_percent}% | {status.temperature_c} C"
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="AI Control Centre V0.6 graphical interface")
    parser.add_argument(
        "--config-dir",
        type=Path,
        default=Path("~/.config/ai-control-centre").expanduser(),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        app = ControlCentreWindow(args.config_dir.expanduser())
    except (ConfigError, FileNotFoundError, RuntimeError, ValueError) as exc:
        print(f"ERROR: {exc}")
        return 1
    app.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
