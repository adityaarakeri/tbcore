"""Host-controlled fake upstream routing for offline Docker integration."""

import http.client

import pytest

from tbcore.images.proxy.proxy import parse_upstream_overrides
from tbcore.sandbox import Sandbox


def test_offline_route_uses_plain_http_only_for_configured_model_host():
    routes = parse_upstream_overrides('{"api.openai.com":"http://fake-provider:8000"}')
    connection = routes("api.openai.com")
    assert isinstance(connection, http.client.HTTPConnection)
    assert not isinstance(connection, http.client.HTTPSConnection)
    assert (connection.host, connection.port) == ("fake-provider", 8000)
    assert isinstance(routes("api.anthropic.com"), http.client.HTTPSConnection)


@pytest.mark.parametrize(
    "route",
    [
        "http://127.0.0.1:8000",
        "https://fake-provider:8000",
        "http://fake-provider:8000/path",
        "http://user@fake-provider:8000",
    ],
)
def test_offline_route_rejects_unsafe_destination(route):
    with pytest.raises(ValueError, match="upstream"):
        parse_upstream_overrides('{"api.openai.com":"' + route + '"}')


def test_sandbox_requires_injected_host_for_offline_route():
    with pytest.raises(ValueError, match="upstream"):
        Sandbox("image", upstream_overrides={"api.openai.com": "http://fake-provider:8000"})
