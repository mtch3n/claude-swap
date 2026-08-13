"""Tests for the systemd user service manager (service.py)."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

from claude_swap import service
from claude_swap.exceptions import ClaudeSwitchError
from claude_swap.models import Platform


class FakeSystemctl:
    """Records `systemctl --user ...` invocations and answers queries."""

    def __init__(self, **answers: tuple[int, str]) -> None:
        # answers maps the first argument ("is-active") to (returncode, stdout)
        self.answers = answers
        self.calls: list[list[str]] = []

    def __call__(self, cmd, **kwargs):
        self.calls.append(list(cmd))
        verb = cmd[2] if len(cmd) > 2 else ""
        code, out = self.answers.get(verb.replace("-", "_"), (0, ""))
        return subprocess.CompletedProcess(cmd, code, stdout=out, stderr="")

    @property
    def verbs(self) -> list[str]:
        return [c[2] for c in self.calls if len(c) > 2]


@pytest.fixture
def linux_systemd(tmp_path, monkeypatch):
    """A Linux box with systemctl, and XDG_CONFIG_HOME pointed at tmp."""
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))
    monkeypatch.delenv("CLAUDE_CONFIG_DIR", raising=False)
    monkeypatch.delenv("XDG_DATA_HOME", raising=False)
    with (
        patch.object(Platform, "detect", return_value=Platform.LINUX),
        patch.object(service.shutil, "which", side_effect=_which),
    ):
        yield


def _which(name: str):
    return {
        "systemctl": "/usr/bin/systemctl",
        "loginctl": "/usr/bin/loginctl",
        "cswap": "/home/u/.local/bin/cswap",
    }.get(name)


class TestUnitPath:
    def test_honours_xdg_config_home(self, linux_systemd, tmp_path):
        assert service.unit_path() == (
            tmp_path / "config" / "systemd" / "user" / "claude-swap.service"
        )

    def test_falls_back_to_dot_config(self, monkeypatch):
        monkeypatch.delenv("XDG_CONFIG_HOME", raising=False)
        assert service.unit_path() == (
            Path.home() / ".config" / "systemd" / "user" / "claude-swap.service"
        )

    def test_relative_xdg_config_home_is_ignored(self, monkeypatch):
        """Per the XDG spec a non-absolute value must be treated as unset."""
        monkeypatch.setenv("XDG_CONFIG_HOME", "relative/path")
        assert service.unit_path() == (
            Path.home() / ".config" / "systemd" / "user" / "claude-swap.service"
        )


class TestExecutableResolution:
    def test_prefers_cswap_on_path(self, linux_systemd):
        assert service.exec_start() == "/home/u/.local/bin/cswap auto"

    def test_falls_back_to_a_sibling_of_the_interpreter(self, tmp_path, monkeypatch):
        """uv/pipx venvs put cswap beside python, which may not be on PATH
        when the unit is written from a different shell."""
        venv_bin = tmp_path / "venv" / "bin"
        venv_bin.mkdir(parents=True)
        (venv_bin / "cswap").touch(mode=0o755)
        monkeypatch.setattr(sys, "executable", str(venv_bin / "python"))
        with patch.object(service.shutil, "which", return_value=None):
            assert service.exec_start() == f"{venv_bin / 'cswap'} auto"

    def test_last_resort_runs_the_module(self, monkeypatch):
        monkeypatch.setattr(sys, "executable", "/usr/bin/python3")
        with patch.object(service.shutil, "which", return_value=None):
            assert service.exec_start() == "/usr/bin/python3 -m claude_swap auto"

    def test_exec_start_is_absolute(self, linux_systemd):
        """A unit file gets no PATH lookup — a bare name would never start."""
        assert service.exec_start().startswith("/")

    @pytest.mark.skipif(sys.platform == "win32", reason="POSIX symlinks")
    def test_keeps_the_shim_path_rather_than_its_symlink_target(self, tmp_path):
        """uv/pipx shims are the stable name; what they point at moves on
        upgrade, and a unit pinned to today's target would break."""
        bin_dir = tmp_path / "bin"
        bin_dir.mkdir()
        real = tmp_path / "tools" / "claude-swap" / "bin" / "cswap"
        real.parent.mkdir(parents=True)
        real.touch()
        shim = bin_dir / "cswap"
        shim.symlink_to(real)
        with patch.object(service.shutil, "which", return_value=str(shim)):
            assert service.exec_start() == f"{shim} auto"


class TestCurrentUser:
    def test_returns_a_login_name(self):
        assert service.current_user()

    def test_degrades_to_empty_rather_than_raising(self):
        """Only used for a loginctl query and a hint line — never worth a crash."""
        with patch.object(service.getpass, "getuser", side_effect=OSError("no pwd")):
            assert service.current_user() == ""


class TestUnitContents:
    def test_runs_the_auto_loop_and_restarts_on_failure(self, linux_systemd):
        unit = service.unit_contents()
        assert "ExecStart=/home/u/.local/bin/cswap auto" in unit
        assert "Restart=on-failure" in unit
        assert "WantedBy=default.target" in unit

    def test_no_environment_lines_when_defaults_are_in_use(self, linux_systemd):
        assert "Environment=CLAUDE_CONFIG_DIR" not in service.unit_contents()
        assert "Environment=XDG_DATA_HOME" not in service.unit_contents()

    @pytest.mark.parametrize("var", ["CLAUDE_CONFIG_DIR", "XDG_DATA_HOME"])
    def test_non_default_store_location_is_baked_in(self, linux_systemd, monkeypatch, var):
        """systemd --user does not inherit the shell's environment. Without
        this the service would read a different account store than the CLI —
        silently, which is the worst way for it to be wrong."""
        monkeypatch.setenv(var, "/custom/location")
        assert f"Environment={var}=/custom/location" in service.unit_contents()


class TestInstall:
    def test_writes_the_unit_and_reloads(self, linux_systemd):
        fake = FakeSystemctl()
        with patch.object(service.subprocess, "run", fake):
            changed = service.install()
        assert changed is True
        assert service.unit_path().read_text() == service.unit_contents()
        assert "daemon-reload" in fake.verbs

    def test_is_idempotent_and_reports_no_change(self, linux_systemd):
        with patch.object(service.subprocess, "run", FakeSystemctl()):
            service.install()
            assert service.install() is False

    def test_rewrites_when_the_unit_changed(self, linux_systemd):
        with patch.object(service.subprocess, "run", FakeSystemctl()):
            service.install()
            service.unit_path().write_text("[Unit]\nDescription=stale\n")
            assert service.install() is True
        assert "stale" not in service.unit_path().read_text()

    def test_creates_the_unit_directory(self, linux_systemd):
        assert not service.unit_path().parent.exists()
        with patch.object(service.subprocess, "run", FakeSystemctl()):
            service.install()
        assert service.unit_path().parent.is_dir()


class TestEnableDisable:
    def test_enable_installs_first_then_starts_now(self, linux_systemd):
        """`cswap service enable` on a clean machine must just work."""
        fake = FakeSystemctl()
        with patch.object(service.subprocess, "run", fake):
            service.enable()
        assert service.unit_path().exists()
        enable_call = next(c for c in fake.calls if "enable" in c)
        assert "--now" in enable_call, "enable must also start it"

    def test_disable_stops_now(self, linux_systemd):
        fake = FakeSystemctl()
        with patch.object(service.subprocess, "run", fake):
            service.install()
            service.disable()
        disable_call = next(c for c in fake.calls if "disable" in c)
        assert "--now" in disable_call

    def test_every_call_is_user_scoped(self, linux_systemd):
        """Nothing in this feature touches system units or needs root."""
        fake = FakeSystemctl()
        with patch.object(service.subprocess, "run", fake):
            service.enable()
            service.disable()
        assert all(c[:2] == ["/usr/bin/systemctl", "--user"] for c in fake.calls)


class TestUninstall:
    def test_disables_removes_and_reloads(self, linux_systemd):
        fake = FakeSystemctl()
        with patch.object(service.subprocess, "run", fake):
            service.install()
            assert service.uninstall() is True
        assert not service.unit_path().exists()
        assert "disable" in fake.verbs
        assert fake.verbs.count("daemon-reload") >= 1

    def test_uninstall_when_not_installed_is_a_noop(self, linux_systemd):
        fake = FakeSystemctl()
        with patch.object(service.subprocess, "run", fake):
            assert service.uninstall() is False
        assert fake.calls == [], "nothing to disable, nothing to reload"


class TestStatus:
    def test_reports_installed_enabled_and_active(self, linux_systemd):
        fake = FakeSystemctl(
            is_active=(0, "active\n"), is_enabled=(0, "enabled\n"),
        )
        with patch.object(service.subprocess, "run", fake):
            service.install()
            got = service.status()
        assert (got.installed, got.enabled, got.active) == (True, True, True)
        assert got.exec_start == "/home/u/.local/bin/cswap auto"

    def test_reports_inactive_when_stopped(self, linux_systemd):
        fake = FakeSystemctl(
            is_active=(3, "inactive\n"), is_enabled=(1, "disabled\n"),
        )
        with patch.object(service.subprocess, "run", fake):
            service.install()
            got = service.status()
        assert got.installed is True
        assert (got.enabled, got.active) == (False, False)

    def test_not_installed_does_not_query_systemd(self, linux_systemd):
        fake = FakeSystemctl()
        with patch.object(service.subprocess, "run", fake):
            got = service.status()
        assert got.installed is False
        assert fake.calls == []

    def test_exec_start_is_read_back_from_the_installed_unit(self, linux_systemd):
        """Status must describe what is installed, not what would be."""
        with patch.object(service.subprocess, "run", FakeSystemctl()):
            service.install()
            service.unit_path().write_text(
                "[Service]\nExecStart=/elsewhere/cswap auto\n"
            )
            assert service.status().exec_start == "/elsewhere/cswap auto"

    def test_linger_is_reported(self, linux_systemd):
        fake = FakeSystemctl()

        def run(cmd, **kwargs):
            if cmd[0].endswith("loginctl"):
                return subprocess.CompletedProcess(cmd, 0, stdout="yes\n", stderr="")
            return fake(cmd, **kwargs)

        with patch.object(service.subprocess, "run", run):
            service.install()
            assert service.status().lingering is True

    def test_unknown_linger_is_not_a_failure(self, linux_systemd):
        """No loginctl (containers, trimmed images) — report unknown, not an error."""
        def which(name):
            return None if name == "loginctl" else _which(name)

        with (
            patch.object(service.shutil, "which", side_effect=which),
            patch.object(service.subprocess, "run", FakeSystemctl()),
        ):
            service.install()
            assert service.status().lingering is None


class TestGuards:
    @pytest.mark.parametrize(
        "platform", [Platform.MACOS, Platform.WINDOWS, Platform.UNKNOWN]
    )
    def test_non_linux_raises_a_clear_error(self, platform, tmp_path, monkeypatch):
        monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
        with patch.object(Platform, "detect", return_value=platform):
            with pytest.raises(ClaudeSwitchError, match="Linux"):
                service.install()

    def test_missing_systemctl_names_the_alternative(self, tmp_path, monkeypatch):
        monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
        with (
            patch.object(Platform, "detect", return_value=Platform.LINUX),
            patch.object(service.shutil, "which", return_value=None),
        ):
            with pytest.raises(ClaudeSwitchError, match="cswap auto"):
                service.install()

    def test_systemctl_failure_surfaces_its_message(self, linux_systemd):
        def run(cmd, **kwargs):
            return subprocess.CompletedProcess(cmd, 1, stdout="", stderr="Failed: no bus")

        with patch.object(service.subprocess, "run", run):
            with pytest.raises(ClaudeSwitchError, match="no bus"):
                service.enable()
