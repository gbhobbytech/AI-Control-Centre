from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .config import ConfigError, load_app_config
from .manager import ServiceManager
from .monitoring import create_gpu_monitor


def _print_status(service_id, status) -> None:
    pid = f" pid={status.pid}" if status.pid is not None else ""
    print(f"{service_id}: {status.state.value}{pid} - {status.detail}")


def _print_many(results) -> None:
    for service_id, status in results:
        _print_status(service_id, status)


def _print_models(config) -> None:
    if not config.models:
        print("No GGUF models discovered or configured")
        return
    for model_id, model in config.models.items():
        state = "ready" if model.complete else "INCOMPLETE"
        split = f" split {len(model.shard_paths)}/{model.expected_shards}" if model.split else ""
        quant = f" [{model.quant}]" if model.quant else ""
        print(f"{model_id}: {model.display_name}{quant} - {state}{split}")
        print(f"  {model.path}")
        if model.warning:
            print(f"  warning: {model.warning}")


def _show_model(config, model_id: str) -> None:
    try:
        model = config.models[model_id]
    except KeyError as exc:
        raise KeyError(f"Unknown model: {model_id}") from exc
    print(f"id: {model.id}")
    print(f"name: {model.display_name}")
    print(f"path: {model.path}")
    print(f"complete: {str(model.complete).lower()}")
    print(f"split: {str(model.split).lower()}")
    print(f"shards: {len(model.shard_paths)}/{model.expected_shards}")
    if model.family:
        print(f"family: {model.family}")
    if model.quant:
        print(f"quant: {model.quant}")
    if model.architecture:
        print(f"architecture: {model.architecture}")
    if model.recommended_gpu_layers is not None:
        print(f"recommended_gpu_layers: {model.recommended_gpu_layers}")
    if model.recommended_context_length is not None:
        print(f"recommended_context_length: {model.recommended_context_length}")
    if model.cache_type_k:
        print(f"cache_type_k: {model.cache_type_k}")
    if model.cache_type_v:
        print(f"cache_type_v: {model.cache_type_v}")
    if model.estimated_vram_mib is not None:
        print(f"estimated_vram_mib: {model.estimated_vram_mib}")
    if model.warning:
        print(f"warning: {model.warning}")
    if model.notes:
        print(f"notes: {model.notes}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="AI Control Centre CLI")
    parser.add_argument(
        "--config-dir",
        type=Path,
        default=Path("~/.config/ai-control-centre").expanduser(),
        help="Directory containing Control Centre TOML configuration",
    )

    sub = parser.add_subparsers(dest="command", required=True)

    status = sub.add_parser("status", help="Show service status")
    status.add_argument("service", nargs="?", default="all")

    start = sub.add_parser("start", help="Start a service and its dependencies")
    start.add_argument("service")
    start.add_argument("--model", help="Override the configured/default model for model-aware services")
    start.add_argument(
        "--stop-conflicts",
        action="store_true",
        help="Stop launcher-owned conflicting services in dependency-safe order before starting",
    )
    start.add_argument(
        "--replace-model",
        action="store_true",
        help="Safely stop affected launcher-owned services when changing the loaded model",
    )

    stop = sub.add_parser("stop", help="Stop a launcher-owned service")
    stop.add_argument("service")

    logs = sub.add_parser("logs", help="Show recent service logs")
    logs.add_argument("service")
    logs.add_argument("--lines", type=int, default=40)

    open_cmd = sub.add_parser("open", help="Open a service web interface")
    open_cmd.add_argument("service")

    profile = sub.add_parser("profile", help="Start or stop a task profile")
    profile.add_argument("action", choices=("start", "stop"))
    profile.add_argument("profile")
    profile.add_argument("--model", help="Override the profile's selected LLM")
    profile.add_argument("--no-open", action="store_true", help="Do not open the profile interface")
    profile.add_argument(
        "--stop-conflicts",
        action="store_true",
        help="Stop launcher-owned conflicting services in dependency-safe order before starting",
    )
    profile.add_argument(
        "--replace-model",
        action="store_true",
        help="Safely stop affected launcher-owned services when changing the loaded model",
    )

    models = sub.add_parser("models", help="List or inspect discovered GGUF models")
    model_sub = models.add_subparsers(dest="models_action", required=True)
    model_sub.add_parser("list", help="List discovered/configured models")
    show = model_sub.add_parser("show", help="Show one model")
    show.add_argument("model")

    sub.add_parser("stop-all", help="Stop launcher-owned services in reverse dependency order")
    sub.add_parser("gpu", help="Show current GPU status")
    sub.add_parser("gui", help="Open the graphical control centre")
    sub.add_parser("setup", help="Open the graphical control centre and run the setup wizard")

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        config = load_app_config(args.config_dir.expanduser())
        manager = ServiceManager(config)

        if args.command in {"gui", "setup"}:
            from .gui import main as gui_main
            gui_args = ["--config-dir", str(args.config_dir.expanduser())]
            if args.command == "setup":
                gui_args.append("--setup")
            return gui_main(gui_args)

        if args.command == "gpu":
            monitor = create_gpu_monitor(config.settings.gpu_backend)
            if monitor is None:
                print("GPU monitoring disabled")
                return 0
            status = monitor.status()
            if not status.available:
                print(f"GPU unavailable: {status.detail}")
                return 0
            print(
                f"{status.name}: {status.memory_used_mib}/{status.memory_total_mib} MiB VRAM, "
                f"{status.utilization_percent}% GPU, {status.temperature_c} C"
            )
            return 0

        if args.command == "models":
            if args.models_action == "list":
                _print_models(config)
            else:
                _show_model(config, args.model)
            return 0

        if args.command == "status":
            if args.service == "all":
                for service_id, service in manager.services.items():
                    _print_status(service_id, service.status())
            else:
                _print_status(args.service, manager.get(args.service).status())
            return 0

        if args.command == "start":
            _print_many(
                manager.start_service(
                    args.service,
                    stop_conflicts=args.stop_conflicts,
                    replace_model=args.replace_model,
                    model_id=args.model,
                )
            )
            return 0

        if args.command == "stop":
            _print_status(args.service, manager.stop_service(args.service))
            return 0

        if args.command == "logs":
            for line in manager.get(args.service).logs(args.lines):
                print(line)
            return 0

        if args.command == "open":
            if not manager.get(args.service).open():
                print(f"No openable interface configured for {args.service}")
            return 0

        if args.command == "profile":
            if args.action == "start":
                _print_many(
                    manager.start_profile(
                        args.profile,
                        open_interface=not args.no_open,
                        stop_conflicts=args.stop_conflicts,
                        replace_model=args.replace_model,
                        model_id=args.model,
                    )
                )
            else:
                _print_many(manager.stop_profile(args.profile))
            return 0

        if args.command == "stop-all":
            _print_many(manager.stop_all())
            return 0

    except (ConfigError, FileNotFoundError, RuntimeError, KeyError, PermissionError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
