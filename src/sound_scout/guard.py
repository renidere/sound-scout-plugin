"""Process-wide network ban for the offline default (R09, T07).

`install_network_guard` adds a `sys.addaudithook` hook that raises `NetworkDenied` on the
socket audit events through which a Python process reaches the network (`socket.connect`,
name resolution, datagram sends, and `socket.bind` on an internet socket — a listener; an
`AF_UNIX` bind stays allowed), and tells the Hugging Face libraries to stay offline. An audit
hook cannot be removed: the ban lasts for the life of the process, which is intended.
Child processes are not covered by the hook; the decoder is kept offline by its protocol
whitelist (`proc.ffmpeg_argv` / `proc.ffprobe_argv`).

`disable_telemetry` switches library telemetry off whatever `network.allow` says: no telemetry
is part of R09 always, not of the offline mode. The hook does not see native sockets, so
onnxruntime's 1DS telemetry (a device ID under `~/.cache` and a `.ses` file in the temporary
directory, written on import) is turned off by `ORT_DISABLE_TELEMETRY` before the first import:
`install_network_guard` calls it on every entry point, and the lazy import sites call it again
so a library caller is covered too.

`network_guard_active` is the `doctor` check: it reports the module flag, since the hook
itself cannot be inspected. `denied_events` lists every event the hook denied in this process,
so a denial that the code below caught and swallowed is still on record (T07).
"""

import os
import socket
import sys

from sound_scout.config import Config

BLOCKED_EVENTS = frozenset(
    {
        "socket.connect",
        "socket.getaddrinfo",
        "socket.gethostbyname",
        "socket.gethostbyaddr",
        "socket.getnameinfo",
        "socket.sendto",
        "socket.sendmsg",
    }
)
INTERNET_FAMILIES = frozenset({socket.AF_INET, socket.AF_INET6})
OFFLINE_ENV = {"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"}
TELEMETRY_OFF_ENV = {"ORT_DISABLE_TELEMETRY": "1"}

_installed = False
_denied: list[str] = []


class NetworkDenied(Exception):
    """A network operation was attempted while `network.allow = false`."""


def install_network_guard(config: Config) -> bool:
    """Ban network access in this process when the configuration does not allow it.

    Library telemetry is switched off in either case. Returns whether the guard is active
    afterwards. Idempotent: a second call does not add a second hook.
    """
    global _installed
    disable_telemetry()
    if config.network.allow:
        return False
    if not _installed:
        sys.addaudithook(_deny_network_events)
        _installed = True
    os.environ.update(OFFLINE_ENV)
    return True


def disable_telemetry() -> None:
    """Switch library telemetry off; call before a library that has it is first imported."""
    os.environ.update(TELEMETRY_OFF_ENV)


def network_guard_active() -> bool:
    return _installed


def denied_events() -> tuple[str, ...]:
    """The audit events the guard denied in this process so far, in order."""
    return tuple(_denied)


def _deny_network_events(event: str, args: tuple[object, ...]) -> None:
    if event in BLOCKED_EVENTS or (event == "socket.bind" and _is_internet_socket(args[0])):
        _denied.append(event)
        raise NetworkDenied(f"network access is disabled (network.allow = false): {event}")


def _is_internet_socket(sock: object) -> bool:
    return getattr(sock, "family", None) in INTERNET_FAMILIES
