"""systemd **user** service for the auto-switch loop (``cswap service``).

``cswap auto`` is the thing worth running unattended, and until now the only
ways to do that were a terminal you never close or a cron line. This module
writes a systemd user unit that runs it, and drives systemctl for the four
verbs a user actually wants: install, uninstall, enable, disable — plus a
status readout the TUI also reads.

Everything here is user-scoped. There is no system unit, no root, and no
sudo: ``systemctl --user`` manages units under the caller's own login. The
one thing that genuinely needs a privileged decision — lingering, which is
what keeps a user service alive after your last session closes — is
*reported* by ``status()`` and left to the user, because enabling it changes
how their login behaves and needs polkit authentication.

The engine already meets systemd halfway: the loop installs a SIGTERM handler
(``cli.py``) so ``systemctl --user stop`` ends a tick cleanly rather than
killing a switch mid-flight, and every event is one line on stdout, which is
exactly what journald wants.
"""

from __future__ import annotations

import getpass
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from claude_swap.exceptions import ClaudeSwitchError
from claude_swap.models import Platform

UNIT_NAME = "claude-swap.service"

# Long enough that a structural failure (no credential store, no network at
# boot) backs off instead of hammering the usage API in a restart loop. The
# loop handles its own transient errors internally, so an actual exit means
# something that a fast retry will not fix.
RESTART_DELAY_S = 30

# Environment the unit must carry explicitly. systemd --user starts from its
# own environment, not the shell's, so a user who redirects the account store
# in .bashrc would otherwise get a service reading a DIFFERENT store than
# their CLI — with no error, just two divergent views of "your accounts".
_INHERITED_ENV = ("CLAUDE_CONFIG_DIR", "XDG_DATA_HOME")


@dataclass(frozen=True)
class ServiceStatus:
    """What the background service is doing right now."""

    installed: bool
    enabled: bool  # starts at login
    active: bool  # running now
    unit_path: Path
    exec_start: str | None  # as recorded in the installed unit
    lingering: bool | None  # None = could not be determined


# ---------------------------------------------------------------------------
# Paths and unit contents
# ---------------------------------------------------------------------------


def current_user() -> str:
    """This user's login name, for loginctl and for the linger hint.

    ``getpass.getuser`` reads LOGNAME/USER/LNAME/USERNAME and falls back to
    the password database, so it still answers on a headless host where the
    shell never exported ``$USER`` — the same failure mode ``macos_keychain``
    documents. Empty string if even that fails; callers degrade rather than
    crash on a cosmetic lookup.
    """
    try:
        return getpass.getuser()
    except Exception:
        return ""


def unit_dir() -> Path:
    """``$XDG_CONFIG_HOME/systemd/user``, the standard user-unit location.

    Per the XDG spec a value that is unset, empty, or non-absolute is ignored
    in favour of ``~/.config`` — matching ``paths.get_backup_root``'s reading
    of ``XDG_DATA_HOME``.
    """
    xdg = os.environ.get("XDG_CONFIG_HOME", "")
    if xdg:
        base = Path(os.path.expanduser(xdg))
        if base.is_absolute():
            return base / "systemd" / "user"
    return Path.home() / ".config" / "systemd" / "user"


def unit_path() -> Path:
    return unit_dir() / UNIT_NAME


def exec_start() -> str:
    """The ``ExecStart`` command line, always absolute.

    A unit file gets no PATH lookup, so a bare ``cswap`` would simply never
    start. Three sources, in order of how well they survive the user's
    installer: PATH (uv tool / pipx shims), a sibling of the running
    interpreter (a venv whose bin dir is not on the PATH of the shell that
    ran ``cswap service install``), and finally the module entry point.

    Deliberately NOT resolved through symlinks. ``~/.local/bin/cswap`` is the
    stable name; what it points at is an implementation detail of uv/pipx that
    an upgrade is free to move, and a unit pinned to today's target would
    break the next time it did.
    """
    found = shutil.which("cswap")
    if found:
        return f"{found} auto"
    sibling = Path(sys.executable).parent / "cswap"
    if sibling.exists():
        return f"{sibling} auto"
    return f"{sys.executable} -m claude_swap auto"


def unit_contents() -> str:
    """Render the unit file for this machine."""
    environment = "".join(
        f"Environment={var}={os.environ[var]}\n"
        for var in _INHERITED_ENV
        if os.environ.get(var)
    )
    return f"""\
[Unit]
Description=claude-swap automatic account switcher
Documentation=https://github.com/realiti4/claude-swap
After=default.target

[Service]
Type=simple
ExecStart={exec_start()}
Restart=on-failure
RestartSec={RESTART_DELAY_S}
Environment=PYTHONUNBUFFERED=1
{environment}
[Install]
WantedBy=default.target
"""


# ---------------------------------------------------------------------------
# systemctl plumbing
# ---------------------------------------------------------------------------


def _systemctl_path() -> str:
    """Absolute systemctl, or a refusal that names the alternative."""
    if Platform.detect() not in (Platform.LINUX, Platform.WSL):
        raise ClaudeSwitchError(
            "cswap service manages a systemd unit, which is Linux-only. "
            "On macOS, run the menu bar app; elsewhere run 'cswap auto'."
        )
    found = shutil.which("systemctl")
    if not found:
        raise ClaudeSwitchError(
            "systemctl was not found — this system does not use systemd. "
            "Run 'cswap auto' directly, or supervise it with whatever init "
            "this system does use."
        )
    return found


def _systemctl(*args: str, check: bool = True) -> subprocess.CompletedProcess:
    completed = subprocess.run(
        [_systemctl_path(), "--user", *args],
        capture_output=True,
        text=True,
        check=False,
    )
    if check and completed.returncode != 0:
        detail = (completed.stderr or completed.stdout or "").strip()
        raise ClaudeSwitchError(
            f"systemctl --user {' '.join(args)} failed"
            + (f": {detail}" if detail else "")
        )
    return completed


# ---------------------------------------------------------------------------
# Actions
# ---------------------------------------------------------------------------


def install() -> bool:
    """Write the unit and reload systemd. True if anything changed.

    Idempotent, and safe to re-run after switching installers: the unit is
    rewritten whenever its rendered contents differ, so a moved ``cswap``
    binary is fixed by ``cswap service install``.
    """
    _systemctl_path()  # refuse early on an unsupported platform
    path = unit_path()
    wanted = unit_contents()
    if path.exists() and path.read_text(encoding="utf-8") == wanted:
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(wanted, encoding="utf-8")
    _systemctl("daemon-reload")
    return True


def uninstall() -> bool:
    """Stop, disable, and remove the unit. False if it wasn't installed."""
    path = unit_path()
    if not path.exists():
        return False
    # Not check=True: a unit that was never enabled makes this non-zero, and
    # that is a fine state to be uninstalling from.
    _systemctl("disable", "--now", UNIT_NAME, check=False)
    path.unlink()
    _systemctl("daemon-reload")
    return True


def enable() -> None:
    """Start the service now and at every login.

    ``--now`` collapses systemd's enabled-at-boot / running-now distinction,
    which is not what someone asking to turn on their account switcher is
    thinking about. Installs first so this works on a clean machine.
    """
    install()
    _systemctl("enable", "--now", UNIT_NAME)


def disable() -> None:
    """Stop the service now and leave it stopped at login."""
    _systemctl("disable", "--now", UNIT_NAME)


def status() -> ServiceStatus:
    """Current service state. Never raises for a merely-absent unit."""
    path = unit_path()
    if not path.exists():
        return ServiceStatus(
            installed=False,
            enabled=False,
            active=False,
            unit_path=path,
            exec_start=None,
            lingering=None,
        )
    return ServiceStatus(
        installed=True,
        # is-enabled/is-active report state through the exit code; a non-zero
        # one is the answer "no", not a failure.
        enabled=_systemctl("is-enabled", UNIT_NAME, check=False).returncode == 0,
        active=_systemctl("is-active", UNIT_NAME, check=False).returncode == 0,
        unit_path=path,
        exec_start=_installed_exec_start(path),
        lingering=_linger_enabled(),
    )


def _installed_exec_start(path: Path) -> str | None:
    """ExecStart as actually installed — which may predate an upgrade."""
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.startswith("ExecStart="):
                return line.split("=", 1)[1].strip()
    except OSError:
        pass
    return None


def _linger_enabled() -> bool | None:
    """Whether this user's services survive their last session logging out.

    None when it cannot be determined (no loginctl, no logind) — an unknown
    answer is reported as unknown rather than guessed at, since the whole
    point of surfacing it is to explain a service that mysteriously stops.
    """
    found = shutil.which("loginctl")
    if not found:
        return None
    try:
        completed = subprocess.run(
            [found, "show-user", current_user(), "--property=Linger", "--value"],
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError:
        return None
    if completed.returncode != 0:
        return None
    answer = completed.stdout.strip().lower()
    if answer in ("yes", "no"):
        return answer == "yes"
    return None
