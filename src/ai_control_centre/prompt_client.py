from __future__ import annotations

import json
import urllib.error
import urllib.request
from collections.abc import Callable
from typing import Any

from .domain import PromptWorkshopConfig, ServiceState


class PromptClientError(RuntimeError):
    """A controlled local prompt-service failure."""


def select_prompt(task_text: str, refined_text: str) -> str:
    refined = refined_text.strip()
    return refined if refined else task_text.strip()


def prompt_service_action(state: ServiceState) -> str:
    if state in {ServiceState.READY, ServiceState.EXTERNAL}:
        return "use"
    if state in {ServiceState.STARTING, ServiceState.RUNNING_NOT_READY}:
        return "wait"
    if state == ServiceState.STOPPED:
        return "start"
    return "unavailable"


def build_prompt_payload(prompt: str, config: PromptWorkshopConfig) -> bytes:
    value = prompt.strip()
    if not value:
        raise PromptClientError("Enter a task to improve.")
    payload = {
        "messages": [
            {"role": "system", "content": config.system_instruction},
            {"role": "user", "content": value},
        ],
        "temperature": config.temperature,
        "stream": False,
    }
    return json.dumps(payload).encode("utf-8")


def extract_refined_prompt(payload: Any) -> str:
    try:
        content = payload["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise PromptClientError(
            "The local LLM returned an unexpected response."
        ) from exc
    if not isinstance(content, str) or not content.strip():
        raise PromptClientError("The local LLM returned an empty refined prompt.")
    return content.strip()


class PromptClient:
    def __init__(
        self,
        config: PromptWorkshopConfig,
        opener: Callable[..., Any] | None = None,
    ):
        self.config = config
        self._opener = opener or urllib.request.urlopen

    def refine(self, prompt: str) -> str:
        request = urllib.request.Request(
            self.config.endpoint,
            data=build_prompt_payload(prompt, self.config),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with self._opener(
                request,
                timeout=self.config.request_timeout_seconds,
            ) as response:
                raw = response.read()
        except urllib.error.HTTPError as exc:
            raise PromptClientError(
                f"Local LLM request failed with HTTP {exc.code}."
            ) from exc
        except urllib.error.URLError as exc:
            reason = getattr(exc, "reason", exc)
            raise PromptClientError(f"Could not reach the local LLM: {reason}") from exc
        except TimeoutError as exc:
            raise PromptClientError("The local LLM request timed out.") from exc
        except OSError as exc:
            raise PromptClientError(f"Could not call the local LLM: {exc}") from exc

        try:
            payload = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise PromptClientError("The local LLM returned invalid JSON.") from exc
        return extract_refined_prompt(payload)
