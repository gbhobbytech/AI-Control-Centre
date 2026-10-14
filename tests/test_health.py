from __future__ import annotations

from unittest.mock import patch
from urllib.error import HTTPError, URLError

from ai_control_centre.domain import HealthCheckConfig
from ai_control_centre.health import check_http


def _challenge(code: int) -> HTTPError:
    return HTTPError("http://127.0.0.1:3080/", code, "Unauthorized", {}, None)


def test_401_is_ready_only_when_explicitly_configured():
    url = "http://127.0.0.1:3080/"
    with patch("ai_control_centre.health.urllib.request.urlopen", side_effect=_challenge(401)):
        ready = check_http(HealthCheckConfig(url=url, expected_status=401))
        assert ready.ok
        assert ready.status_code == 401
        default = check_http(HealthCheckConfig(url=url))
        assert not default.ok


def test_expected_401_with_body_requirement_is_not_ready():
    config = HealthCheckConfig(url="http://127.0.0.1:3080/", expected_status=401, body_contains="ready")
    with patch("ai_control_centre.health.urllib.request.urlopen", side_effect=_challenge(401)):
        assert not check_http(config).ok


def test_expected_401_does_not_hide_connection_failures():
    config = HealthCheckConfig(url="http://127.0.0.1:3080/", expected_status=401)
    with patch("ai_control_centre.health.urllib.request.urlopen", side_effect=URLError("connection refused")):
        assert not check_http(config).ok
