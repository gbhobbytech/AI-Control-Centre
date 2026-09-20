from __future__ import annotations

import json
import urllib.error

import pytest

from ai_control_centre.domain import PromptWorkshopConfig, ServiceState
from ai_control_centre.prompt_client import (
    PromptClient,
    PromptClientError,
    build_prompt_payload,
    extract_refined_prompt,
    prompt_service_action,
    select_prompt,
)


class _Response:
    def __init__(self, payload: object):
        self._raw = json.dumps(payload).encode("utf-8")

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def read(self) -> bytes:
        return self._raw


class _RawResponse(_Response):
    def __init__(self, raw: bytes):
        self._raw = raw


def test_select_prompt_prefers_refined_text():
    assert select_prompt(" original ", " refined ") == "refined"
    assert select_prompt(" original ", "  ") == "original"


def test_prompt_service_action():
    assert prompt_service_action(ServiceState.READY) == "use"
    assert prompt_service_action(ServiceState.EXTERNAL) == "use"
    assert prompt_service_action(ServiceState.RUNNING_NOT_READY) == "wait"
    assert prompt_service_action(ServiceState.STOPPED) == "start"
    assert prompt_service_action(ServiceState.ERROR) == "unavailable"


def test_build_prompt_payload_contains_instruction_and_task():
    config = PromptWorkshopConfig(system_instruction="Improve it")
    payload = json.loads(build_prompt_payload("Build a robot", config))
    assert payload["messages"] == [
        {"role": "system", "content": "Improve it"},
        {"role": "user", "content": "Build a robot"},
    ]
    assert payload["stream"] is False


def test_extract_refined_prompt_rejects_invalid_shape():
    with pytest.raises(PromptClientError, match="unexpected"):
        extract_refined_prompt({"choices": []})


def test_prompt_client_returns_refined_content():
    captured = {}

    def opener(request, timeout):
        captured["url"] = request.full_url
        captured["timeout"] = timeout
        return _Response({"choices": [{"message": {"content": "  Better prompt  "}}]})

    config = PromptWorkshopConfig(request_timeout_seconds=7)
    result = PromptClient(config, opener=opener).refine("rough prompt")
    assert result == "Better prompt"
    assert captured == {"url": config.endpoint, "timeout": 7}


def test_prompt_client_reports_http_failure():
    def opener(request, timeout):
        raise urllib.error.HTTPError(request.full_url, 500, "error", {}, None)

    with pytest.raises(PromptClientError, match="HTTP 500"):
        PromptClient(PromptWorkshopConfig(), opener=opener).refine("test")


def test_prompt_client_reports_invalid_json():
    def opener(_request, timeout):
        return _RawResponse(b"not json")

    with pytest.raises(PromptClientError, match="invalid JSON"):
        PromptClient(PromptWorkshopConfig(), opener=opener).refine("test")


def test_prompt_client_reports_timeout():
    def opener(_request, timeout):
        raise TimeoutError

    with pytest.raises(PromptClientError, match="timed out"):
        PromptClient(PromptWorkshopConfig(), opener=opener).refine("test")
