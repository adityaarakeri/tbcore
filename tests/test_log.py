import io
import logging

import pytest

from tbcore import Client, ModelSpec, configure_logging
from tbcore.accounting import CallLog
from tbcore.client import TbError
from tbcore.log import LOGGER_NAME


@pytest.fixture(autouse=True)
def restore_logger():
    logger = logging.getLogger(LOGGER_NAME)
    handlers, level, propagate = list(logger.handlers), logger.level, logger.propagate
    yield
    logger.handlers[:] = handlers
    logger.setLevel(level)
    logger.propagate = propagate


@pytest.fixture(autouse=True)
def clean_color_env(monkeypatch):
    for name in ("NO_COLOR", "FORCE_COLOR", "TERM"):
        monkeypatch.delenv(name, raising=False)


def emit(stream, **kwargs):
    configure_logging("DEBUG", stream=stream, **kwargs)
    logging.getLogger("tbcore.test").warning("hello %s", "world")
    return stream.getvalue()


def test_library_installs_only_a_null_handler():
    handlers = logging.getLogger(LOGGER_NAME).handlers

    assert any(isinstance(h, logging.NullHandler) for h in handlers)


def test_plain_output_has_no_escape_codes():
    out = emit(io.StringIO(), color=False)

    assert "\x1b" not in out and "WARNING" in out and "tbcore.test: hello world" in out


def test_forced_color_wraps_level_and_timestamp():
    out = emit(io.StringIO(), color=True)

    assert "\x1b[33mWARNING\x1b[0m" in out  # yellow
    assert out.startswith("\x1b[2m") and "hello world" in out


def test_each_level_has_its_own_color():
    stream = io.StringIO()
    configure_logging("DEBUG", stream=stream, color=True)
    log = logging.getLogger("tbcore.test")
    log.debug("d")
    log.info("i")
    log.error("e")
    log.critical("c")
    out = stream.getvalue()

    for code in ("\x1b[36m", "\x1b[32m", "\x1b[31m", "\x1b[1;41m"):
        assert code in out


def test_other_handlers_keep_the_plain_level_name():
    seen = []

    class Capture(logging.Handler):
        def emit(self, record):
            seen.append(record.levelname)

    configure_logging("INFO", stream=io.StringIO(), color=True)
    logging.getLogger(LOGGER_NAME).addHandler(Capture())
    logging.getLogger("tbcore.test").info("x")

    assert seen == ["INFO"]


def test_auto_color_off_when_not_a_tty():
    assert "\x1b" not in emit(io.StringIO())


def test_no_color_beats_force_color(monkeypatch):
    monkeypatch.setenv("NO_COLOR", "1")
    monkeypatch.setenv("FORCE_COLOR", "1")

    assert "\x1b" not in emit(io.StringIO())


def test_force_color_env_enables_color(monkeypatch):
    monkeypatch.setenv("FORCE_COLOR", "1")

    assert "\x1b[33m" in emit(io.StringIO())


def test_force_color_zero_does_not_enable(monkeypatch):
    monkeypatch.setenv("FORCE_COLOR", "0")

    assert "\x1b" not in emit(io.StringIO())


def test_explicit_color_argument_beats_environment(monkeypatch):
    monkeypatch.setenv("NO_COLOR", "1")

    assert "\x1b[33m" in emit(io.StringIO(), color=True)


def test_reconfigure_replaces_handler_instead_of_duplicating():
    first, second = io.StringIO(), io.StringIO()
    configure_logging("INFO", stream=first, color=False)
    configure_logging("INFO", stream=second, color=False)
    logging.getLogger("tbcore.test").info("once")

    assert first.getvalue() == "" and second.getvalue().count("once") == 1


def test_level_filters_and_logger_does_not_propagate():
    stream = io.StringIO()
    configure_logging("WARNING", stream=stream, color=False)
    logging.getLogger("tbcore.test").info("quiet")

    assert stream.getvalue() == ""
    assert logging.getLogger(LOGGER_NAME).propagate is False


class FakeTransport:
    def __init__(self, response=None, error=None):
        self.response, self.error = response, error

    def send(self, request):
        if self.error:
            raise self.error
        return self.response


OPENAI_OK = {
    "choices": [{"message": {"content": "hi"}, "finish_reason": "stop"}],
    "usage": {"prompt_tokens": 10, "completion_tokens": 5},
}
SPEC = ModelSpec("m", "model-x", wire="openai-compat", api_key_required=False)


def test_client_logs_success_without_prompt_or_key(caplog):
    client = Client(SPEC, "sk-very-secret", transport=FakeTransport(OPENAI_OK), log=CallLog(None))

    with caplog.at_level(logging.INFO, logger="tbcore.client"):
        client.complete("system-prompt-text", [{"role": "user", "content": "user-prompt-text"}])

    [record] = caplog.records
    message = record.getMessage()
    assert message.startswith("call ok model=model-x in=10 out=5 latency=")
    for secret in ("sk-very-secret", "system-prompt-text", "user-prompt-text"):
        assert secret not in message


def test_client_logs_failure_with_type_only(caplog):
    boom = TbError("body contained sk-very-secret")
    client = Client(SPEC, "sk-very-secret", transport=FakeTransport(error=boom), log=CallLog(None))

    with caplog.at_level(logging.INFO, logger="tbcore.client"), pytest.raises(TbError):
        client.complete(None, [{"role": "user", "content": "x"}])

    [record] = caplog.records
    assert record.levelno == logging.WARNING
    assert record.getMessage() == "call failed model=model-x error=TbError"


class FakeTty(io.StringIO):
    def isatty(self):
        return True


def test_auto_color_on_for_a_capable_tty(monkeypatch):
    monkeypatch.setattr("tbcore.log._ansi_supported", lambda stream: True)

    assert "\x1b[33m" in emit(FakeTty())


def test_auto_color_off_when_terminal_lacks_ansi(monkeypatch):
    monkeypatch.setattr("tbcore.log._ansi_supported", lambda stream: False)

    assert "\x1b" not in emit(FakeTty())


def test_auto_color_off_for_dumb_terminal(monkeypatch):
    monkeypatch.setattr("tbcore.log._ansi_supported", lambda stream: True)
    monkeypatch.setenv("TERM", "dumb")

    assert "\x1b" not in emit(FakeTty())
