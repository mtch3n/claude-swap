"""Tests for the `cswap service` subcommand."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

from claude_swap import cli, service
from claude_swap.exceptions import ClaudeSwitchError


def _run(argv: list[str], capsys) -> tuple[int, str, str]:
    """Run `cswap service <argv>`; returns (exit_code, stdout, stderr)."""
    code = 0
    with patch.object(sys, "argv", ["claude-swap", "service", *argv]):
        try:
            cli.main()
        except SystemExit as e:
            code = e.code if isinstance(e.code, int) else 1
    out, err = capsys.readouterr()
    return code, out, err


def _status(**kwargs) -> service.ServiceStatus:
    defaults = dict(
        installed=True,
        enabled=True,
        active=True,
        unit_path=Path("/home/u/.config/systemd/user/claude-swap.service"),
        exec_start="/home/u/.local/bin/cswap auto",
        lingering=True,
    )
    defaults.update(kwargs)
    return service.ServiceStatus(**defaults)


class TestActions:
    @pytest.mark.parametrize(
        "action", ["install", "uninstall", "enable", "disable"]
    )
    def test_each_verb_calls_its_function(self, temp_home, capsys, action):
        with patch.object(service, action, return_value=True) as fn:
            code, out, _ = _run([action], capsys)
        assert code == 0
        fn.assert_called_once_with()
        assert out.strip(), "every action reports what it did"

    def test_install_reports_no_change_when_already_current(self, temp_home, capsys):
        with patch.object(service, "install", return_value=False):
            _, out, _ = _run(["install"], capsys)
        assert "already" in out.lower()

    def test_uninstall_when_absent_is_not_an_error(self, temp_home, capsys):
        with patch.object(service, "uninstall", return_value=False):
            code, out, err = _run(["uninstall"], capsys)
        assert code == 0
        assert "not installed" in (out + err).lower()

    def test_enable_mentions_that_it_starts_now(self, temp_home, capsys):
        """The user asked to turn it on; say plainly that it is running."""
        with patch.object(service, "enable"):
            _, out, _ = _run(["enable"], capsys)
        assert "running" in out.lower() or "started" in out.lower()


class TestStatus:
    def test_status_is_the_default_action(self, temp_home, capsys):
        with patch.object(service, "status", return_value=_status()) as fn:
            code, out, _ = _run([], capsys)
        assert code == 0
        fn.assert_called_once_with()
        assert "claude-swap.service" in out

    def test_shows_the_installed_command_and_unit_path(self, temp_home, capsys):
        with patch.object(service, "status", return_value=_status()):
            _, out, _ = _run(["status"], capsys)
        assert "/home/u/.local/bin/cswap auto" in out
        assert "/home/u/.config/systemd/user/claude-swap.service" in out

    def test_not_installed_says_how_to_install(self, temp_home, capsys):
        with patch.object(
            service,
            "status",
            return_value=_status(
                installed=False, enabled=False, active=False, exec_start=None
            ),
        ):
            _, out, _ = _run(["status"], capsys)
        assert "cswap service install" in out

    def test_warns_when_linger_is_off(self, temp_home, capsys):
        """Without linger the service dies with your last session — the single
        most confusing way for this feature to 'not work'."""
        with patch.object(service, "status", return_value=_status(lingering=False)):
            _, out, _ = _run(["status"], capsys)
        assert "enable-linger" in out

    def test_no_linger_advice_when_already_lingering(self, temp_home, capsys):
        with patch.object(service, "status", return_value=_status(lingering=True)):
            _, out, _ = _run(["status"], capsys)
        assert "enable-linger" not in out

    def test_unknown_linger_is_not_reported_as_off(self, temp_home, capsys):
        with patch.object(service, "status", return_value=_status(lingering=None)):
            _, out, _ = _run(["status"], capsys)
        assert "enable-linger" not in out

    def test_json_status(self, temp_home, capsys):
        with patch.object(service, "status", return_value=_status()):
            code, out, _ = _run(["status", "--json"], capsys)
        assert code == 0
        payload = json.loads(out)
        assert payload["installed"] is True
        assert payload["enabled"] is True
        assert payload["active"] is True
        assert payload["execStart"] == "/home/u/.local/bin/cswap auto"
        assert payload["lingering"] is True

    def test_json_rejected_for_mutating_actions(self, temp_home, capsys):
        code, _, err = _run(["enable", "--json"], capsys)
        assert code == 2
        assert "--json" in err


class TestErrors:
    def test_unsupported_platform_exits_1_with_the_message(self, temp_home, capsys):
        with patch.object(
            service, "install", side_effect=ClaudeSwitchError("Linux-only, sorry")
        ):
            code, _, err = _run(["install"], capsys)
        assert code == 1
        assert "Linux-only" in err

    def test_error_envelope_in_json_mode(self, temp_home, capsys):
        with patch.object(
            service, "status", side_effect=ClaudeSwitchError("no bus")
        ):
            code, out, _ = _run(["status", "--json"], capsys)
        assert code == 1
        assert "no bus" in json.dumps(json.loads(out))

    def test_unknown_action_exits_2(self, temp_home, capsys):
        code, _, _ = _run(["frobnicate"], capsys)
        assert code == 2


class TestHelp:
    def test_service_help_lists_every_verb(self, temp_home, capsys):
        code, out, _ = _run(["--help"], capsys)
        assert code == 0
        for verb in ("install", "uninstall", "enable", "disable", "status"):
            assert verb in out

    def test_main_help_mentions_service(self, temp_home, capsys):
        code = 0
        with patch.object(sys, "argv", ["claude-swap", "help"]):
            try:
                cli.main()
            except SystemExit as e:
                code = e.code if isinstance(e.code, int) else 1
        assert code == 0
        assert "service" in capsys.readouterr().out
