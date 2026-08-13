"""Tests for desktop notifications (notify.py)."""

from __future__ import annotations

import json
import subprocess
import sys
from unittest.mock import patch

import pytest

from claude_swap import cli
from claude_swap.autoswitch import (
    AllExhaustedEvent,
    ConfigWarningEvent,
    NoSwitchEvent,
    PollEvent,
    QuarantineEvent,
    SwitchEvent,
    UnquarantineEvent,
)
from claude_swap.models import Platform
from claude_swap.notify import APP_NAME, DesktopNotifier


class FakeClock:
    """Monotonic clock the dedupe window can be steered with."""

    def __init__(self, start: float = 1000.0) -> None:
        self.t = start

    def __call__(self) -> float:
        return self.t

    def advance(self, seconds: float) -> None:
        self.t += seconds


class Runs:
    """Records subprocess.run calls; returns a configurable exit code."""

    def __init__(self, returncode: int = 0, raises: Exception | None = None) -> None:
        self.calls: list[list[str]] = []
        self.returncode = returncode
        self.raises = raises

    def __call__(self, cmd, **kwargs):
        self.calls.append(list(cmd))
        self.kwargs = kwargs
        if self.raises is not None:
            raise self.raises
        return subprocess.CompletedProcess(cmd, self.returncode)


@pytest.fixture
def linux_with_notify_send():
    """A Linux box with notify-send on PATH."""
    with (
        patch.object(Platform, "detect", return_value=Platform.LINUX),
        patch("claude_swap.notify.shutil.which", return_value="/usr/bin/notify-send"),
    ):
        yield


def _switch_event(dry_run: bool = False) -> SwitchEvent:
    return SwitchEvent(
        trigger="proactive",
        from_ref={"number": "1", "email": "a@example.com"},
        to_ref={"number": "2", "email": "b@example.com"},
        dry_run=dry_run,
    )


class TestNotify:
    def test_invokes_notify_send_with_app_name_and_urgency(self, linux_with_notify_send):
        runs = Runs()
        with patch("claude_swap.notify.subprocess.run", runs):
            assert DesktopNotifier().notify("Title", "Body", urgency="critical") is True
        (cmd,) = runs.calls
        assert cmd[0] == "/usr/bin/notify-send"
        assert f"--app-name={APP_NAME}" in cmd
        assert "--urgency=critical" in cmd
        # Title and body are positional and last, after the `--` terminator, so
        # a body that happens to start with a dash is never read as a flag.
        assert cmd[-3:] == ["--", "Title", "Body"]

    def test_bounded_timeout_so_a_wedged_daemon_cannot_stall_the_loop(
        self, linux_with_notify_send
    ):
        runs = Runs()
        with patch("claude_swap.notify.subprocess.run", runs):
            DesktopNotifier().notify("Title", "Body")
        assert 0 < runs.kwargs["timeout"] <= 10

    def test_nonzero_exit_reports_failure(self, linux_with_notify_send):
        with patch("claude_swap.notify.subprocess.run", Runs(returncode=1)):
            assert DesktopNotifier().notify("Title", "Body") is False

    @pytest.mark.parametrize(
        "boom",
        [
            OSError("no such file"),
            subprocess.TimeoutExpired("notify-send", 5),
            subprocess.SubprocessError("broken"),
        ],
    )
    def test_subprocess_failures_are_swallowed(self, linux_with_notify_send, boom):
        """A notifier that raises would take the watcher down with it."""
        with patch("claude_swap.notify.subprocess.run", Runs(raises=boom)):
            assert DesktopNotifier().notify("Title", "Body") is False


class TestAvailability:
    @pytest.mark.parametrize(
        "platform", [Platform.MACOS, Platform.WINDOWS, Platform.UNKNOWN]
    )
    def test_non_linux_never_shells_out(self, platform):
        runs = Runs()
        with (
            patch.object(Platform, "detect", return_value=platform),
            patch("claude_swap.notify.subprocess.run", runs),
        ):
            assert DesktopNotifier().notify("Title", "Body") is False
        assert runs.calls == []

    def test_wsl_is_supported(self):
        runs = Runs()
        with (
            patch.object(Platform, "detect", return_value=Platform.WSL),
            patch("claude_swap.notify.shutil.which", return_value="/usr/bin/notify-send"),
            patch("claude_swap.notify.subprocess.run", runs),
        ):
            assert DesktopNotifier().notify("Title", "Body") is True
        assert len(runs.calls) == 1

    def test_missing_binary_no_ops(self):
        runs = Runs()
        with (
            patch.object(Platform, "detect", return_value=Platform.LINUX),
            patch("claude_swap.notify.shutil.which", return_value=None),
            patch("claude_swap.notify.subprocess.run", runs),
        ):
            assert DesktopNotifier().notify("Title", "Body") is False
        assert runs.calls == []

    def test_binary_is_resolved_once_not_per_notification(self):
        """The watcher runs for weeks; PATH must not be re-scanned per event."""
        with (
            patch.object(Platform, "detect", return_value=Platform.LINUX),
            patch(
                "claude_swap.notify.shutil.which", return_value="/usr/bin/notify-send"
            ) as which,
            patch("claude_swap.notify.subprocess.run", Runs()),
        ):
            notifier = DesktopNotifier()
            for i in range(5):
                notifier.notify("Title", f"Body {i}")
        assert which.call_count == 1


class TestDedupe:
    def test_identical_notification_suppressed_inside_the_window(
        self, linux_with_notify_send
    ):
        clock, runs = FakeClock(), Runs()
        notifier = DesktopNotifier(dedupe_window_s=900.0, clock=clock)
        with patch("claude_swap.notify.subprocess.run", runs):
            assert notifier.notify("Title", "Body") is True
            clock.advance(899.0)
            assert notifier.notify("Title", "Body") is False
        assert len(runs.calls) == 1

    def test_repeats_again_once_the_window_passes(self, linux_with_notify_send):
        clock, runs = FakeClock(), Runs()
        notifier = DesktopNotifier(dedupe_window_s=900.0, clock=clock)
        with patch("claude_swap.notify.subprocess.run", runs):
            notifier.notify("Title", "Body")
            clock.advance(901.0)
            assert notifier.notify("Title", "Body") is True
        assert len(runs.calls) == 2

    def test_different_body_is_not_suppressed(self, linux_with_notify_send):
        runs = Runs()
        notifier = DesktopNotifier(clock=FakeClock())
        with patch("claude_swap.notify.subprocess.run", runs):
            notifier.notify("Title", "First")
            notifier.notify("Title", "Second")
        assert len(runs.calls) == 2

    def test_repeated_all_exhausted_ticks_notify_once(self, linux_with_notify_send):
        """The engine re-emits all-exhausted on every blocked tick
        (autoswitch.py _blocked_wait_long resets per tick), so without the
        window an exhausted fleet would notify forever."""
        runs = Runs()
        notifier = DesktopNotifier(clock=FakeClock())
        event = AllExhaustedEvent(earliest_reset_at="2026-08-14T00:00:00Z")
        with patch("claude_swap.notify.subprocess.run", runs):
            for _ in range(10):
                notifier.notify_event(event)
        assert len(runs.calls) == 1


class TestNotifyEvent:
    @pytest.mark.parametrize(
        "event, title",
        [
            (_switch_event(), "Switched account"),
            (
                QuarantineEvent(number="2", email="b@example.com", reason="dead token"),
                "Account quarantined",
            ),
            (AllExhaustedEvent(earliest_reset_at=None), "All accounts exhausted"),
            (ConfigWarningEvent(message="model Fable matches nothing"), "Configuration warning"),
        ],
    )
    def test_notifiable_kinds_use_the_engine_rendering_as_the_body(
        self, linux_with_notify_send, event, title
    ):
        runs = Runs()
        with patch("claude_swap.notify.subprocess.run", runs):
            assert DesktopNotifier().notify_event(event) is True
        (cmd,) = runs.calls
        assert cmd[-3:] == ["--", title, event.human()]

    @pytest.mark.parametrize(
        "event",
        [
            PollEvent(
                active={"number": "1", "email": "a@example.com"},
                headroom={"1": 40.0},
                threshold=90.0,
            ),
            NoSwitchEvent(reason="cooldown"),
            UnquarantineEvent(number="2", email="b@example.com"),
        ],
    )
    def test_routine_events_never_notify(self, linux_with_notify_send, event):
        runs = Runs()
        with patch("claude_swap.notify.subprocess.run", runs):
            assert DesktopNotifier().notify_event(event) is False
        assert runs.calls == []

    def test_dry_run_switch_never_notifies(self, linux_with_notify_send):
        """Nothing actually switched — claiming otherwise would be a lie."""
        runs = Runs()
        with patch("claude_swap.notify.subprocess.run", runs):
            assert DesktopNotifier().notify_event(_switch_event(dry_run=True)) is False
        assert runs.calls == []

    def test_quarantine_and_exhaustion_are_critical(self, linux_with_notify_send):
        runs = Runs()
        with patch("claude_swap.notify.subprocess.run", runs):
            notifier = DesktopNotifier()
            notifier.notify_event(
                QuarantineEvent(number="2", email="b@example.com", reason="dead")
            )
            notifier.notify_event(AllExhaustedEvent(earliest_reset_at=None))
        assert all("--urgency=critical" in cmd for cmd in runs.calls)

    def test_switch_is_normal_urgency(self, linux_with_notify_send):
        runs = Runs()
        with patch("claude_swap.notify.subprocess.run", runs):
            DesktopNotifier().notify_event(_switch_event())
        assert "--urgency=normal" in runs.calls[0]


def _enable_notifications() -> None:
    with patch.object(
        sys, "argv", ["claude-swap", "config", "set", "notifications.enabled", "true"]
    ):
        cli.main()  # returns normally on success


def _run_auto_once(*extra_argv: str) -> list:
    """Run `cswap auto --once` against a fake engine that emits one switch.

    Returns the events the notifier was asked to raise.
    """
    notified: list = []
    captured: dict = {}

    class FakeEngine:
        def __init__(self, switcher, settings, on_event, *, dry_run=False,
                     state_path=None, clock=None):
            captured["on_event"] = on_event

        def tick(self):
            from claude_swap.autoswitch import TickOutcome

            captured["on_event"](_switch_event())
            return TickOutcome.SWITCHED

    with (
        patch("claude_swap.autoswitch.AutoSwitchEngine", FakeEngine),
        patch.object(
            DesktopNotifier,
            "notify_event",
            lambda self, event: bool(notified.append(event)) or True,
        ),
        patch("os.geteuid", return_value=1000, create=True),
        patch.object(sys, "argv", ["claude-swap", "auto", "--once", *extra_argv]),
    ):
        with pytest.raises(SystemExit):
            cli.main()
    return notified


class TestAutoCommandWiring:
    """`cswap auto` notifies only when the config says so — and prints either
    way. The systemd service runs this same code path."""

    def test_disabled_by_default(self, temp_home, capsys):
        notified = _run_auto_once()
        assert notified == []
        assert "Switched" in capsys.readouterr().out, "still printed"

    def test_enabled_notifies(self, temp_home, capsys):
        _enable_notifications()
        capsys.readouterr()
        assert [e.kind for e in _run_auto_once()] == ["switch"]

    def test_event_is_printed_as_well_as_notified(self, temp_home, capsys):
        """The terminal/journal line is the record; notification is extra."""
        _enable_notifications()
        capsys.readouterr()
        _run_auto_once()
        assert "Switched" in capsys.readouterr().out

    def test_json_mode_still_notifies_and_stays_parseable(self, temp_home, capsys):
        """--json is for scripts; notifying must not corrupt the stream."""
        _enable_notifications()
        capsys.readouterr()
        assert len(_run_auto_once("--json")) == 1
        for line in capsys.readouterr().out.splitlines():
            json.loads(line)  # every line still parses
