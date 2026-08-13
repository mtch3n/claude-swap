"""Desktop notifications for the auto-switch watcher (Linux/GNOME).

``cswap auto`` normally runs where nobody is looking — a spare terminal, or
the systemd user service (``cswap service``). The events worth interrupting
someone for are rare: a switch happened, an account's token died, the whole
fleet is spent, a config key is inert. This module puts those four on the
desktop and stays out of the way otherwise.

Delivery is ``notify-send`` (libnotify), shelled out to. That keeps the
dependency list untouched — libnotify is already there on any desktop with a
notification daemon — and a ``systemd --user`` unit inherits
``DBUS_SESSION_BUS_ADDRESS`` from the user manager, so the service reaches the
session bus with no extra plumbing.

Off by default; ``cswap config set notifications.enabled true`` turns it on.
macOS is deliberately excluded: the menu bar (``menubar.py``) already raises
native notifications for the same four events through rumps.

Nothing here raises. A notifier that can crash the watcher is strictly worse
than no notifier, so every failure path — absent binary, wedged daemon,
non-zero exit — degrades to a logged line and ``False``.
"""

from __future__ import annotations

import logging
import shutil
import subprocess
import time
from collections.abc import Callable

from claude_swap.models import Platform

_logger = logging.getLogger("claude-swap")

NOTIFY_BINARY = "notify-send"
APP_NAME = "claude-swap"

# notify-send returns as soon as the daemon accepts the message, so this only
# has to cover a wedged or starting daemon — never a user reading anything.
SEND_TIMEOUT_S = 5.0

# How long an identical (kind, body) notification is suppressed for. Not
# cosmetic: ``AllExhaustedEvent`` is emitted on every blocked tick, and the
# engine's ``_blocked_wait_long`` guard resets at the top of each tick, so an
# exhausted fleet would otherwise re-notify for as long as it stayed exhausted.
DEDUPE_WINDOW_S = 900.0

# Freedesktop icon-naming-spec names, present in every mainstream icon theme.
_ICONS = {"normal": "dialog-information", "critical": "dialog-warning"}

# The four notify-worthy event kinds, mapped to a title and an urgency. Kept
# in step with the macOS menu bar's own list (``menubar.py``): both surfaces
# answer "is this worth interrupting someone for?" the same way. Everything
# else the engine emits — polls, no-switches, sleeps, transient errors — is
# loop bookkeeping and belongs in the log, not on the screen.
NOTIFIABLE_KINDS: dict[str, tuple[str, str]] = {
    "switch": ("Switched account", "normal"),
    "account-quarantined": ("Account quarantined", "critical"),
    "all-exhausted": ("All accounts exhausted", "critical"),
    "config-warning": ("Configuration warning", "normal"),
}


class DesktopNotifier:
    """Sends desktop notifications, with the binary resolved once and repeats
    suppressed. One instance per ``cswap auto`` run; not thread-safe, which is
    fine — the engine emits from a single thread."""

    def __init__(
        self,
        *,
        dedupe_window_s: float = DEDUPE_WINDOW_S,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._dedupe_window_s = dedupe_window_s
        self._clock = clock
        self._binary: str | None = None
        self._resolved = False
        self._last_sent: dict[tuple[str, str], float] = {}

    # -- availability -------------------------------------------------------

    def _resolve_binary(self) -> str | None:
        """Absolute path to notify-send, or None if unusable here.

        Resolved once per instance: the watcher runs for weeks, and rescanning
        PATH per event would be pointless work in a hot-ish loop.
        """
        if not self._resolved:
            self._resolved = True
            self._binary = self._find_binary()
        return self._binary

    @staticmethod
    def _find_binary() -> str | None:
        if Platform.detect() not in (Platform.LINUX, Platform.WSL):
            _logger.debug("Desktop notifications are Linux-only; not notifying")
            return None
        found = shutil.which(NOTIFY_BINARY)
        if found is None:
            _logger.warning(
                "notifications.enabled is on but '%s' was not found on PATH; "
                "install libnotify (e.g. libnotify-bin) to receive them",
                NOTIFY_BINARY,
            )
        return found

    # -- sending ------------------------------------------------------------

    def notify(
        self,
        title: str,
        body: str,
        *,
        urgency: str = "normal",
        dedupe_key: tuple[str, str] | None = None,
    ) -> bool:
        """Raise one notification. True if the daemon accepted it."""
        binary = self._resolve_binary()
        if binary is None:
            return False
        if self._suppressed(dedupe_key or (title, body)):
            return False
        cmd = [
            binary,
            f"--app-name={APP_NAME}",
            f"--urgency={urgency}",
            f"--icon={_ICONS.get(urgency, _ICONS['normal'])}",
            # Option terminator: a body starting with '-' is a message, never
            # a flag. Both title and body are engine-rendered text.
            "--",
            title,
            body,
        ]
        try:
            completed = subprocess.run(
                cmd,
                timeout=SEND_TIMEOUT_S,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
            )
        except (OSError, subprocess.SubprocessError) as exc:
            _logger.debug("notify-send failed: %s", exc)
            return False
        if completed.returncode != 0:
            _logger.debug("notify-send exited %s", completed.returncode)
            return False
        return True

    def notify_event(self, event) -> bool:
        """Notify for one engine event, or skip it. True if one was sent.

        The body is the event's own ``human()`` — the same string the CLI
        prints and the menu bar shows — so the three surfaces cannot drift.
        """
        entry = NOTIFIABLE_KINDS.get(event.kind)
        if entry is None:
            return False
        # A dry-run switch never happened; saying otherwise would be a lie.
        if event.kind == "switch" and getattr(event, "dry_run", False):
            return False
        title, urgency = entry
        body = event.human()
        return self.notify(
            title, body, urgency=urgency, dedupe_key=(event.kind, body)
        )

    # -- dedupe -------------------------------------------------------------

    def _suppressed(self, key: tuple[str, str]) -> bool:
        """True if this exact notification went out inside the window.

        The attempt is recorded whether or not the send then succeeds: a
        failing daemon should not be retried once per tick either.
        """
        now = self._clock()
        last = self._last_sent.get(key)
        if last is not None and now - last < self._dedupe_window_s:
            return True
        self._last_sent = {
            k: t
            for k, t in self._last_sent.items()
            if now - t < self._dedupe_window_s  # bound the map on long runs
        }
        self._last_sent[key] = now
        return False
