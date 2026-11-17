"""Process-wide network ban for the offline default (R09, T07).

`install_network_guard` adds a `sys.addaudithook` hook that raises `NetworkDenied` on the
socket audit events through which a Python process reaches the network (`socket.connect`,
name resolution, datagram sends, and `socket.bind` on an internet socket — a listener; an
`AF_UNIX` bind stays allowed), and tells the Hugging Face libraries to stay offline. An audit
hook cannot be removed: the ban lasts for the life of the process, which is intended.
Child processes are not covered by the hook; the decoder is kept offline by its protocol
whitelist (`proc.ffmpeg_argv` / `proc.ffprobe_argv`).

`network_guard_active` is the `doctor` check: it reports the module flag, since the hook
itself cannot be inspected.
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

_installed = False


class NetworkDenied(Exception):
    """A network operation was attempted while `network.allow = false`."""


def install_network_guard(config: Config) -> bool:
    """Ban network access in this process when the configuration does not allow it.

    Returns whether the guard is active afterwards. Idempotent: a second call does not add
    a second hook.
    """
    global _installed
    if config.network.allow:
        return False
    if not _installed:
        sys.addaudithook(_deny_network_events)
        _installed = True
    os.environ.update(OFFLINE_ENV)
    return True


def network_guard_active() -> bool:
    return _installed


def _deny_network_events(event: str, args: tuple[object, ...]) -> None:
    if event in BLOCKED_EVENTS or (event == "socket.bind" and _is_internet_socket(args[0])):
        raise NetworkDenied(f"network access is disabled (network.allow = false): {event}")


def _is_internet_socket(sock: object) -> bool:
    return getattr(sock, "family", None) in INTERNET_FAMILIES
