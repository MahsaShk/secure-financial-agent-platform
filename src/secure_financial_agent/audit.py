from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

from .schemas import AuditEvent


class AuditSink:
    """Base class for audit destinations. Subclasses implement write()."""

    def write(self, event: AuditEvent) -> None:
        raise NotImplementedError


class InMemoryAuditSink(AuditSink):
    """Keeps events in the events list. This is the default sink; events are lost when the process exits."""

    def __init__(self) -> None:
        self.events: list[AuditEvent] = []

    def write(self, event: AuditEvent) -> None:
        self.events.append(event)


class JsonlAuditSink(AuditSink):
    """Appends each event as one JSON line to a file. Not used by the app yet."""

    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)

    def write(self, event: AuditEvent) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with self._path.open("a", encoding="utf-8") as file:
            file.write(json.dumps(event.to_dict(), sort_keys=True) + "\n")


class CallbackAuditSink(AuditSink):
    """Passes each event to a caller-supplied function. Not used by the app yet."""

    def __init__(self, callback: Callable[[AuditEvent], None]) -> None:
        self._callback = callback

    def write(self, event: AuditEvent) -> None:
        self._callback(event)

