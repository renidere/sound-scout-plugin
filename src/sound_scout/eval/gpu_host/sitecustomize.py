"""Network guard of every Python process that `hostrun.py serve` starts on the GPU host (#147).

`site` imports this file at start-up from `PYTHONPATH`, so the vLLM server and every child it
starts through `spawn` get the same audit hook before any library code runs. The hook allows
`AF_UNIX` and loopback (`127.0.0.0/8`, `::1`) and raises `PermissionError` on any other socket
connect, bind, send or name lookup, before the system call: no packet leaves. Every denial is
one JSON line in `$SS_GUARD_LOG`. Native sockets (C libraries) do not pass this hook;
`hostrun.py` sees them in `/proc` and kills the process group. Stdlib only, Python 3.10+, never
imports `sound_scout`: the file goes to the host alone.
"""

import ipaddress
import json
import os
import socket
import sys
import time

GUARD_ACTIVE = True
LOG_ENV = "SS_GUARD_LOG"
LOOKUP_NAMES = frozenset({"", "localhost", "127.0.0.1", "::1"})
ADDRESS_EVENTS = frozenset({"socket.connect", "socket.bind", "socket.sendto", "socket.sendmsg"})
LOOKUP_EVENTS = frozenset({"socket.getaddrinfo", "socket.gethostbyname", "socket.gethostbyaddr"})
INTERNET = frozenset({socket.AF_INET, socket.AF_INET6})


def is_loopback(host) -> bool:
    if isinstance(host, bytes):
        host = host.decode("ascii", "replace")
    if not isinstance(host, str) or not host:
        return False
    if host == "localhost":
        return True
    try:
        address = ipaddress.ip_address(host.split("%", 1)[0])
    except ValueError:
        return False
    mapped = getattr(address, "ipv4_mapped", None)
    return (mapped or address).is_loopback


def _deny(event: str, target) -> None:
    path = os.environ.get(LOG_ENV)
    if path:
        line = json.dumps({"t": time.time(), "pid": os.getpid(), "event": event, "target": target})
        try:
            fd = os.open(path, os.O_WRONLY | os.O_APPEND | os.O_CREAT | os.O_CLOEXEC, 0o600)
            try:
                os.write(fd, (line + "\n").encode())
            finally:
                os.close(fd)
        except OSError:
            pass
    raise PermissionError(f"sound-scout guard: {event} to {target!r} is denied")


def _hook(event: str, args: tuple) -> None:
    if not event.startswith("socket."):
        return
    if event in ADDRESS_EVENTS:
        sock, address = args[0], args[1]
        family = getattr(sock, "family", None)
        if family == getattr(socket, "AF_UNIX", None):
            return
        if family in INTERNET:
            if address is None and event == "socket.sendmsg":
                return
            host = address[0] if isinstance(address, tuple) and address else address
            if is_loopback(host):
                return
            _deny(event, repr(host))
        _deny(event, f"family {family!r}")
    elif event in LOOKUP_EVENTS:
        host = args[0]
        if isinstance(host, bytes):
            host = host.decode("ascii", "replace")
        if host is None or host in LOOKUP_NAMES or is_loopback(host):
            return
        _deny(event, repr(host))
    elif event == "socket.getnameinfo":
        sockaddr = args[0]
        host = sockaddr[0] if isinstance(sockaddr, tuple) and sockaddr else None
        if not is_loopback(host):
            _deny(event, repr(host))


sys.addaudithook(_hook)
